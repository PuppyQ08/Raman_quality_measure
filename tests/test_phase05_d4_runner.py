from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.sugar_quantitative import (  # noqa: E402
    CONFIG_SHA256,
    D4SugarCohort,
    D4WellSplit,
)
from rpe.runner.d4_sugar import (  # noqa: E402
    D4RunnerValidationError,
    prepare_d4_conditions,
    regression_metrics,
    run_d4_smoke_from_cohort,
    technical_lod_loq,
    validate_d4_smoke_result,
)


CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d4_sugar_protocol.json"
)
TARGET_NAMES = (
    "sucrose_nominal_mol_l",
    "fructose_nominal_mol_l",
    "maltose_nominal_mol_l",
    "glucose_nominal_mol_l",
)
CONTROL = "low_snr_raw_control"
SG = "low_snr_raw_plus_sg11"


def read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def synthetic_cohort() -> D4SugarCohort:
    generator = np.random.Generator(np.random.PCG64(20260817))
    well_count = 60
    records_per_well = 2
    feature_count = 40
    levels = np.array([0.0, 0.08, 0.2, 0.32], dtype=np.float64)
    well_targets = np.empty((well_count, 4), dtype="<f8")
    for fold in range(5):
        fold_wells = np.arange(fold, well_count, 5)
        for target_index in range(4):
            well_targets[fold_wells, target_index] = generator.permutation(
                np.tile(levels, fold_wells.size // levels.size)
            )
    basis = generator.normal(size=(4, feature_count))
    baseline = np.linspace(10.0, 12.0, feature_count)
    rows = []
    targets = []
    record_ids = []
    well_ids = []
    source_members = []
    rounds = []
    repetitions = []
    for well_index in range(well_count):
        well_id = f"W{well_index:03d}"
        for replicate in range(records_per_well):
            signal = (
                baseline
                + np.float64(25.0) * well_targets[well_index] @ basis
                + generator.normal(scale=0.05, size=feature_count)
            )
            rows.append(np.asarray(signal, dtype="<f4"))
            targets.append(well_targets[well_index])
            record_ids.append(f"record-{well_index:03d}-{replicate}")
            well_ids.append(well_id)
            source_members.append(f"source/{well_id}-{replicate}.csv")
            rounds.append(replicate + 1)
            repetitions.append(1)
    blank = np.stack(
        [
            baseline + generator.normal(scale=0.08, size=feature_count)
            for _ in range(8)
        ]
    ).astype("<f4")
    blank_targets = np.zeros((8, 4), dtype="<f8")
    folds = tuple(
        read_only(
            np.asarray(
                [
                    index
                    for index, well_id in enumerate(well_ids)
                    if int(well_id[1:]) % 5 == fold
                ],
                dtype="<i8",
            )
        )
        for fold in range(5)
    )
    test_fold = 0
    validation_fold = 1
    train_folds = (2, 3, 4)
    split = D4WellSplit(
        seed=0,
        train_indices=read_only(
            np.sort(np.concatenate([folds[fold] for fold in train_folds]))
            .astype("<i8")
        ),
        validation_indices=folds[validation_fold],
        test_indices=folds[test_fold],
        train_folds=train_folds,
        validation_fold=validation_fold,
        test_fold=test_fold,
    )
    return D4SugarCohort(
        protocol_config_sha256=CONFIG_SHA256,
        intensity=read_only(np.stack(rows).astype("<f4")),
        targets=read_only(np.stack(targets).astype("<f8")),
        wavenumber=read_only(
            np.linspace(150.0, 3500.0, feature_count, dtype="<f4")
        ),
        target_names=TARGET_NAMES,
        record_ids=tuple(record_ids),
        well_ids=tuple(well_ids),
        source_members=tuple(source_members),
        rounds=read_only(np.asarray(rounds, dtype="<i8")),
        repetitions=read_only(np.asarray(repetitions, dtype="<i8")),
        blank_intensity=read_only(blank),
        blank_targets=read_only(blank_targets),
        blank_record_ids=tuple(f"blank-{index}" for index in range(8)),
        blank_well_ids=("blank-well",) * 8,
        blank_source_members=tuple(
            f"source/blank-{index}.csv" for index in range(8)
        ),
        folds=folds,
        splits=(split,),
    )


def recursive_keys(value: object) -> set[str]:
    keys = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(key)
            keys.update(recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(recursive_keys(item))
    return keys


class D4ConditionAndMetricTest(unittest.TestCase):
    def test_conditions_apply_exact_sg_to_same_rows_without_mutation(self):
        cohort = synthetic_cohort()
        before = cohort.intensity.copy()

        control, sg = prepare_d4_conditions(cohort, CONFIG)

        self.assertEqual(control.condition_id, CONTROL)
        self.assertEqual(sg.condition_id, SG)
        self.assertEqual(control.protocol_config_sha256, CONFIG_SHA256)
        self.assertEqual(sg.protocol_config_sha256, CONFIG_SHA256)
        self.assertEqual(control.record_ids, cohort.record_ids)
        self.assertEqual(sg.record_ids, cohort.record_ids)
        np.testing.assert_array_equal(control.values, before)
        np.testing.assert_array_equal(
            sg.values,
            np.asarray(
                savgol_filter(
                    before,
                    window_length=11,
                    polyorder=3,
                    deriv=0,
                    axis=-1,
                    mode="interp",
                ),
                dtype="<f4",
            ),
        )
        np.testing.assert_array_equal(cohort.intensity, before)
        self.assertFalse(control.values.flags.writeable)
        self.assertFalse(sg.values.flags.writeable)
        self.assertFalse(np.shares_memory(control.values, cohort.intensity))
        self.assertFalse(np.shares_memory(sg.values, cohort.intensity))

    def test_regression_metrics_match_hand_calculation(self):
        true = np.array(
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
            ],
            dtype="<f8",
        )
        predicted = np.array(
            [
                [0.1, 0.0, 0.3, 0.2],
                [0.0, 0.3, 0.2, 0.5],
                [0.3, 0.2, 0.5, 0.4],
            ],
            dtype="<f8",
        )

        metrics = regression_metrics(true, predicted, TARGET_NAMES)

        errors = predicted - true
        expected_rmse = np.sqrt(np.mean(errors**2, axis=0))
        expected_mae = np.mean(np.abs(errors), axis=0)
        expected_r2 = 1.0 - np.sum(errors**2, axis=0) / np.sum(
            (true - true.mean(axis=0)) ** 2,
            axis=0,
        )
        self.assertEqual(
            metrics["macro_normalized_rmse"],
            float(np.mean(expected_rmse / 0.32)),
        )
        self.assertEqual(
            metrics["macro_mae_mol_l"],
            float(expected_mae.mean()),
        )
        self.assertEqual(metrics["macro_r2"], float(expected_r2.mean()))
        for index, target_name in enumerate(TARGET_NAMES):
            current = metrics["per_analyte"][target_name]
            self.assertEqual(current["rmse_mol_l"], float(expected_rmse[index]))
            self.assertEqual(current["mae_mol_l"], float(expected_mae[index]))
            self.assertEqual(current["r2"], float(expected_r2[index]))

    def test_technical_lod_loq_matches_known_slopes_and_sample_sigma(self):
        true_axis = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        slopes = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
        intercepts = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        train_true = np.stack([true_axis] * 4, axis=1)
        train_predicted = intercepts + train_true * slopes
        blank_predicted = np.stack(
            (
                intercepts - 1.0,
                intercepts,
                intercepts + 1.0,
            )
        )

        estimates = technical_lod_loq(
            train_true,
            train_predicted,
            blank_predicted,
            TARGET_NAMES,
        )

        for index, target_name in enumerate(TARGET_NAMES):
            current = estimates[target_name]
            self.assertAlmostEqual(current["slope"], float(slopes[index]))
            self.assertEqual(current["sigma"], 1.0)
            self.assertAlmostEqual(
                current["ich_lod_mol_l"],
                3.3 / slopes[index],
            )
            self.assertAlmostEqual(
                current["ich_loq_mol_l"],
                10.0 / slopes[index],
            )
            self.assertAlmostEqual(
                current["iupac_lod_mol_l"],
                3.0 / slopes[index],
            )
            self.assertEqual(
                current["status"],
                "technical_repeatability_estimate_not_validated_analytical_limit",
            )

        invalid = train_predicted.copy()
        invalid[:, 0] = invalid[::-1, 0]
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "positive finite slope",
        ):
            technical_lod_loq(
                train_true,
                invalid,
                blank_predicted,
                TARGET_NAMES,
            )


class D4SyntheticRunnerTest(unittest.TestCase):
    def setUp(self):
        self.cohort = synthetic_cohort()

    def test_smoke_selects_components_and_emits_paired_predictions(self):
        result = run_d4_smoke_from_cohort(
            self.cohort,
            CONFIG,
            seed=0,
        )

        self.assertEqual(result["schema_version"], "phase05-d4-result-v1")
        self.assertEqual(result["experiment_id"], "d4_sugar_low_snr_pls2_sg11")
        self.assertEqual(result["result_level"], "smoke")
        self.assertEqual(result["result_label"], "SMOKE — PROVIDED COHORT")
        self.assertEqual(result["seed"], 0)
        self.assertEqual(result["protocol_config"]["sha256"], CONFIG_SHA256)
        self.assertEqual(result["cohort"]["record_count"], 120)
        self.assertEqual(result["cohort"]["well_count"], 60)
        self.assertEqual(result["cohort"]["blank_record_count"], 8)
        self.assertEqual(result["cohort"]["feature_count"], 40)
        self.assertEqual(
            result["cohort"]["matrix_sha256"],
            hashlib.sha256(
                np.ascontiguousarray(self.cohort.intensity).tobytes(order="C")
            ).hexdigest(),
        )
        self.assertEqual(
            set(result["code"]),
            {
                "rpe/downstream/sugar_quantitative.py",
                "rpe/runner/d4_sugar.py",
                "tools/run_phase05_d4.py",
            },
        )
        self.assertEqual(result["split"]["train_count"], 72)
        self.assertEqual(result["split"]["validation_count"], 24)
        self.assertEqual(result["split"]["test_count"], 24)
        self.assertEqual(
            [condition["condition_id"] for condition in result["conditions"]],
            [CONTROL, SG],
        )
        for condition in result["conditions"]:
            self.assertIn(condition["selected_n_components"], (2, 4, 8, 16, 32))
            self.assertEqual(
                [row["n_components"] for row in condition["validation_scores"]],
                [2, 4, 8, 16, 32],
            )
            validation_values = [
                row["macro_normalized_rmse"]
                for row in condition["validation_scores"]
            ]
            expected = min(
                zip(validation_values, (2, 4, 8, 16, 32)),
                key=lambda item: (item[0], item[1]),
            )[1]
            self.assertEqual(condition["selected_n_components"], expected)
            self.assertEqual(len(condition["test_predictions"]), 24)
            self.assertEqual(
                set(condition["lod_loq"]),
                set(TARGET_NAMES),
            )
        self.assertEqual(len(result["paired_predictions"]), 24)
        for index, row in enumerate(result["paired_predictions"]):
            test_index = int(self.cohort.splits[0].test_indices[index])
            self.assertEqual(row["record_id"], self.cohort.record_ids[test_index])
            self.assertEqual(row["well_id"], self.cohort.well_ids[test_index])
            np.testing.assert_array_equal(
                row["true_targets"],
                self.cohort.targets[test_index],
            )
            self.assertEqual(set(row["predictions"]), {CONTROL, SG})

        conditions = {
            item["condition_id"]: item for item in result["conditions"]
        }
        expected_effect = 100.0 * (
            conditions[CONTROL]["test_metrics"]["macro_normalized_rmse"]
            - conditions[SG]["test_metrics"]["macro_normalized_rmse"]
        )
        self.assertEqual(
            result["descriptive_effect_percentage_points"],
            expected_effect,
        )
        forbidden = {
            "bootstrap",
            "confidence_interval",
            "gate_success",
            "p_value",
            "permutation",
            "significant",
        }
        self.assertFalse(recursive_keys(result) & forbidden)
        validate_d4_smoke_result(result)

    def test_runner_is_deterministic_and_preserves_input(self):
        before_intensity = self.cohort.intensity.copy()
        before_targets = self.cohort.targets.copy()

        first = run_d4_smoke_from_cohort(self.cohort, CONFIG, seed=0)
        second = run_d4_smoke_from_cohort(self.cohort, CONFIG, seed=0)

        self.assertEqual(first, second)
        np.testing.assert_array_equal(self.cohort.intensity, before_intensity)
        np.testing.assert_array_equal(self.cohort.targets, before_targets)

    def test_runner_rejects_wrong_seed_config_or_split_leakage(self):
        with self.assertRaisesRegex(D4RunnerValidationError, "seed"):
            run_d4_smoke_from_cohort(self.cohort, CONFIG, seed=1)

        wrong_config = D4SugarCohort(
            **{
                **self.cohort.__dict__,
                "protocol_config_sha256": "f" * 64,
            }
        )
        with self.assertRaisesRegex(D4RunnerValidationError, "protocol config"):
            run_d4_smoke_from_cohort(wrong_config, CONFIG, seed=0)

        split = self.cohort.splits[0]
        leaking = D4WellSplit(
            seed=0,
            train_indices=read_only(
                np.concatenate((split.train_indices, split.test_indices[:1]))
                .astype("<i8")
            ),
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
            train_folds=split.train_folds,
            validation_fold=split.validation_fold,
            test_fold=split.test_fold,
        )
        leaking_cohort = D4SugarCohort(
            **{
                **self.cohort.__dict__,
                "splits": (leaking,),
            }
        )
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "split records",
        ):
            run_d4_smoke_from_cohort(leaking_cohort, CONFIG, seed=0)

        mislabeled = D4WellSplit(
            seed=0,
            train_indices=split.train_indices,
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
            train_folds=split.train_folds,
            validation_fold=split.test_fold,
            test_fold=split.validation_fold,
        )
        mislabeled_cohort = D4SugarCohort(
            **{
                **self.cohort.__dict__,
                "splits": (mislabeled,),
            }
        )
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "split folds",
        ):
            run_d4_smoke_from_cohort(mislabeled_cohort, CONFIG, seed=0)

        missing_level_targets = self.cohort.targets.copy()
        validation = split.validation_indices
        missing_level_targets[
            validation[
                missing_level_targets[validation, 0] == 0.08
            ],
            0,
        ] = 0.0
        missing_level_cohort = D4SugarCohort(
            **{
                **self.cohort.__dict__,
                "targets": read_only(missing_level_targets),
            }
        )
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "split target levels",
        ):
            run_d4_smoke_from_cohort(
                missing_level_cohort,
                CONFIG,
                seed=0,
            )

        crossed_folds = [
            read_only(values.copy())
            for values in self.cohort.folds
        ]
        fold_two = crossed_folds[2].copy()
        fold_three = crossed_folds[3].copy()
        fold_two[0], fold_three[0] = fold_three[0], fold_two[0]
        crossed_folds[2] = read_only(np.sort(fold_two).astype("<i8"))
        crossed_folds[3] = read_only(np.sort(fold_three).astype("<i8"))
        crossed_fold_cohort = D4SugarCohort(
            **{
                **self.cohort.__dict__,
                "folds": tuple(crossed_folds),
            }
        )
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "split folds",
        ):
            run_d4_smoke_from_cohort(
                crossed_fold_cohort,
                CONFIG,
                seed=0,
            )

    def test_runner_rejects_malformed_axis_or_identities(self):
        reversed_axis = D4SugarCohort(
            **{
                **self.cohort.__dict__,
                "wavenumber": read_only(self.cohort.wavenumber[::-1].copy()),
            }
        )
        with self.assertRaisesRegex(D4RunnerValidationError, "wavenumber"):
            run_d4_smoke_from_cohort(reversed_axis, CONFIG, seed=0)

        duplicate_ids = list(self.cohort.record_ids)
        train = self.cohort.splits[0].train_indices
        duplicate_ids[int(train[1])] = duplicate_ids[int(train[0])]
        duplicate_id_cohort = D4SugarCohort(
            **{
                **self.cohort.__dict__,
                "record_ids": tuple(duplicate_ids),
            }
        )
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "cohort identities",
        ):
            run_d4_smoke_from_cohort(
                duplicate_id_cohort,
                CONFIG,
                seed=0,
            )

        short_blank_ids = D4SugarCohort(
            **{
                **self.cohort.__dict__,
                "blank_record_ids": self.cohort.blank_record_ids[:-1],
            }
        )
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "blank identities",
        ):
            run_d4_smoke_from_cohort(short_blank_ids, CONFIG, seed=0)

    def test_validator_rejects_metric_or_prediction_drift(self):
        result = run_d4_smoke_from_cohort(self.cohort, CONFIG, seed=0)
        metric_drift = copy.deepcopy(result)
        metric_drift["conditions"][0]["test_metrics"][
            "macro_normalized_rmse"
        ] = 99.0
        with self.assertRaisesRegex(D4RunnerValidationError, "metrics"):
            validate_d4_smoke_result(metric_drift)

        prediction_drift = copy.deepcopy(result)
        prediction_drift["paired_predictions"][0]["predictions"][CONTROL][0] += 1
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "paired predictions",
        ):
            validate_d4_smoke_result(prediction_drift)

    def test_validator_rejects_claim_or_lod_drift(self):
        result = run_d4_smoke_from_cohort(self.cohort, CONFIG, seed=0)

        claim_drift = copy.deepcopy(result)
        claim_drift["claim_boundary"]["retained_sugar_comparison"] = True
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "claim boundary",
        ):
            validate_d4_smoke_result(claim_drift)

        lod_drift = copy.deepcopy(result)
        lod_drift["conditions"][0]["lod_loq"][TARGET_NAMES[0]][
            "ich_lod_mol_l"
        ] += 1.0
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "LOD/LOQ",
        ):
            validate_d4_smoke_result(lod_drift)

        code_drift = copy.deepcopy(result)
        code_drift["code"]["rpe/runner/d4_sugar.py"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "code provenance",
        ):
            validate_d4_smoke_result(code_drift)

        cohort_drift = copy.deepcopy(result)
        cohort_drift["cohort"]["matrix_sha256"] = "e" * 64
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "cohort provenance",
        ):
            validate_d4_smoke_result(cohort_drift)

    def test_validator_rejects_malformed_condition_or_split_types(self):
        result = run_d4_smoke_from_cohort(self.cohort, CONFIG, seed=0)

        malformed_condition = copy.deepcopy(result)
        malformed_condition["conditions"][0] = "not-an-object"
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "conditions",
        ):
            validate_d4_smoke_result(malformed_condition)

        malformed_split = copy.deepcopy(result)
        malformed_split["split"]["train_count"] = "72"
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "split",
        ):
            validate_d4_smoke_result(malformed_split)


if __name__ == "__main__":
    unittest.main()
