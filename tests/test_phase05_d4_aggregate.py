from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.sugar_quantitative import (  # noqa: E402
    ARCHIVE_BYTES,
    ARCHIVE_SHA256,
    AXIS_SHA256,
    BLANK_RECORD_IDS_SHA256,
    BLANK_SOURCE_MEMBERS_SHA256,
    CONFIG_BYTES,
    CONFIG_SHA256,
    FOLD_RECORD_IDS_SHA256,
    FOLD_SOURCE_MEMBERS_SHA256,
    MIXTURE_RECORD_IDS_SHA256,
    MIXTURE_SOURCE_MEMBERS_SHA256,
    TARGET_NAMES,
)
from rpe.runner.d4_aggregate import (  # noqa: E402
    D4AggregateValidationError,
    aggregate_d4_results,
    paired_well_primary_statistics,
)
from rpe.runner.d4_sugar import (  # noqa: E402
    CONDITION_IDS,
    CONTROL_CONDITION_ID,
    FOLD_WELL_IDS_SHA256,
    RETAINED_BLANK_MATRIX_SHA256,
    RETAINED_BLANK_WELL_IDS_SHA256,
    RETAINED_MATRIX_SHA256,
    RETAINED_REPETITIONS_SHA256,
    RETAINED_ROUNDS_SHA256,
    RETAINED_TARGETS_SHA256,
    RETAINED_WELL_IDS_SHA256,
    SG_CONDITION_ID,
)


AUDIT = ROOT / "reports" / "phase05" / "d4_step01_protocol_audit.json"
LOW_PREFIX = (
    "Raw data/Experimental data from sugar mixtures/Raw data files/"
    "Sugar_Concentration_Test_Fast/"
)
RUNNER_CODE_PATHS = (
    "rpe/downstream/sugar_quantitative.py",
    "rpe/runner/d4_sugar.py",
    "tools/run_phase05_d4.py",
)


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def current_runner_code() -> dict[str, object]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": hashlib.sha256(
                (ROOT / relative_path).read_bytes()
            ).hexdigest(),
        }
        for relative_path in RUNNER_CODE_PATHS
    }


def retained_cohort_document() -> dict[str, object]:
    return {
        "axis_sha256": AXIS_SHA256,
        "blank_matrix_sha256": RETAINED_BLANK_MATRIX_SHA256,
        "blank_record_count": 32,
        "blank_record_ids_sha256": BLANK_RECORD_IDS_SHA256,
        "blank_source_members_sha256": BLANK_SOURCE_MEMBERS_SHA256,
        "blank_well_ids_sha256": RETAINED_BLANK_WELL_IDS_SHA256,
        "feature_count": 2000,
        "matrix_sha256": RETAINED_MATRIX_SHA256,
        "record_count": 7680,
        "record_ids_sha256": MIXTURE_RECORD_IDS_SHA256,
        "repetitions_sha256": RETAINED_REPETITIONS_SHA256,
        "rounds_sha256": RETAINED_ROUNDS_SHA256,
        "source_members_sha256": MIXTURE_SOURCE_MEMBERS_SHA256,
        "targets_sha256": RETAINED_TARGETS_SHA256,
        "well_count": 240,
        "well_ids_sha256": RETAINED_WELL_IDS_SHA256,
    }


def direct_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    errors = predicted - true
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    denominator = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(errors**2, axis=0) / denominator
    return {
        "macro_mae_mol_l": float(mae.mean()),
        "macro_normalized_rmse": float(np.mean(rmse / 0.32)),
        "macro_r2": float(r2.mean()),
        "per_analyte": {
            name: {
                "mae_mol_l": float(mae[index]),
                "r2": float(r2[index]),
                "rmse_mol_l": float(rmse[index]),
            }
            for index, name in enumerate(TARGET_NAMES)
        },
        "sample_count": int(true.shape[0]),
    }


def lod_document(sigma: float) -> dict[str, object]:
    return {
        name: {
            "blank_replicates": 32,
            "ich_lod_mol_l": 3.3 * sigma,
            "ich_loq_mol_l": 10.0 * sigma,
            "iupac_lod_mol_l": 3.0 * sigma,
            "sigma": sigma,
            "slope": 1.0,
            "status": (
                "technical_repeatability_estimate_not_validated_"
                "analytical_limit"
            ),
        }
        for name in TARGET_NAMES
    }


def frozen_rows_by_seed() -> dict[int, list[dict[str, object]]]:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    all_wells = sorted(
        {
            well_id
            for fold in audit["folds"]
            for well_id in fold["well_ids"]
        }
    )
    well_position = {
        well_id: index
        for index, well_id in enumerate(all_wells)
    }
    levels = (0.0, 0.08, 0.2, 0.32)
    output = {}
    for seed in range(5):
        rows = []
        for well_id in audit["folds"][seed]["well_ids"]:
            cell, plate_text = well_id.split("_")
            row_letter = cell[0]
            column = int(cell[1:])
            plate = int(plate_text)
            sample = 12 * (ord(row_letter) - ord("A")) + column
            position = well_position[well_id]
            targets = [
                levels[(position + target_index) % len(levels)]
                for target_index in range(4)
            ]
            for source_round in range(1, 9):
                for repetition in range(1, 5):
                    record_id = (
                        f"low_snr-s{sample:03d}-"
                        f"{row_letter.lower()}{column:02d}-p{plate:02d}"
                        f"-r{source_round:02d}-m01-rep{repetition:02d}"
                    )
                    source_member = (
                        LOW_PREFIX
                        + f"Sugar_Concentration_Test_Fast_{sample}_"
                        + f"{well_id}_RD{source_round}_M1_R{repetition}.csv"
                    )
                    signed = -1.0 if repetition % 2 else 1.0
                    scale = 0.018 + 0.001 * (position % 5)
                    control_error = [
                        signed * scale * (1.0 + 0.1 * target_index)
                        for target_index in range(4)
                    ]
                    sg_error = [
                        0.65 * value
                        for value in control_error
                    ]
                    rows.append(
                        {
                            "predictions": {
                                CONTROL_CONDITION_ID: [
                                    targets[index] + control_error[index]
                                    for index in range(4)
                                ],
                                SG_CONDITION_ID: [
                                    targets[index] + sg_error[index]
                                    for index in range(4)
                                ],
                            },
                            "record_id": record_id,
                            "repetition": repetition,
                            "round": source_round,
                            "source_member": source_member,
                            "true_targets": targets,
                            "well_id": well_id,
                        }
                    )
        output[seed] = sorted(rows, key=lambda row: row["record_id"])
    return output


def seed_result(
    seed: int,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    true = np.asarray([row["true_targets"] for row in rows], dtype="<f8")
    condition_documents = []
    condition_metrics = {}
    for condition_index, condition_id in enumerate(CONDITION_IDS):
        predicted = np.asarray(
            [row["predictions"][condition_id] for row in rows],
            dtype="<f8",
        )
        metrics = direct_metrics(true, predicted)
        condition_metrics[condition_id] = metrics
        selected = 8 if condition_index == 0 else 16
        condition_documents.append(
            {
                "condition_id": condition_id,
                "condition_matrix_sha256": (
                    RETAINED_MATRIX_SHA256
                    if condition_index == 0
                    else "a" * 64
                ),
                "lod_loq": lod_document(
                    0.01 + 0.001 * seed + 0.002 * condition_index
                ),
                "selected_n_components": selected,
                "test_metrics": metrics,
                "test_predictions": predicted.tolist(),
                "validation_scores": [
                    {
                        "macro_normalized_rmse": (
                            0.2
                            + 0.01 * abs(n_components - selected)
                        ),
                        "n_components": n_components,
                    }
                    for n_components in (2, 4, 8, 16, 32)
                ],
            }
        )
    payload = b"".join(canonical_json_bytes(row) for row in rows)
    return {
        "claim_boundary": {
            "descriptive_only": True,
            "formal_inference": False,
            "retained_sugar_comparison": True,
        },
        "code": current_runner_code(),
        "cohort": retained_cohort_document(),
        "conditions": condition_documents,
        "descriptive_effect_percentage_points": 100.0 * (
            condition_metrics[CONTROL_CONDITION_ID][
                "macro_normalized_rmse"
            ]
            - condition_metrics[SG_CONDITION_ID][
                "macro_normalized_rmse"
            ]
        ),
        "experiment_id": "d4_sugar_low_snr_pls2_sg11",
        "paired_predictions": rows,
        "paired_predictions_artifact": {
            "bytes": len(payload),
            "lines": len(rows),
            "path": f"paired_predictions_seed{seed}_complete.jsonl",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "protocol_config": {
            "bytes": CONFIG_BYTES,
            "path": "d4_sugar_protocol.json",
            "sha256": CONFIG_SHA256,
        },
        "result_label": f"COMPLETE SEED — {seed + 1}/5",
        "result_level": "complete_seed",
        "schema_version": "phase05-d4-result-v1",
        "seed": seed,
        "seeds_completed": seed + 1,
        "seeds_required": 5,
        "source_archive": {
            "bytes": ARCHIVE_BYTES,
            "path": "Raw data.zip",
            "sha256": ARCHIVE_SHA256,
        },
        "split": {
            "seed": seed,
            "test_count": 1536,
            "test_fold": seed,
            "test_record_ids_sha256": FOLD_RECORD_IDS_SHA256[seed],
            "test_source_members_sha256": (
                FOLD_SOURCE_MEMBERS_SHA256[seed]
            ),
            "test_well_count": 48,
            "test_well_ids_sha256": FOLD_WELL_IDS_SHA256[seed],
            "train_count": 4608,
            "train_folds": [
                fold
                for fold in range(5)
                if fold not in {seed, (seed + 1) % 5}
            ],
            "train_well_count": 144,
            "validation_count": 1536,
            "validation_fold": (seed + 1) % 5,
            "validation_well_count": 48,
        },
        "status": "completed",
        "target_names": list(TARGET_NAMES),
    }


class D4WellStatisticsTest(unittest.TestCase):
    def test_recomputes_pooled_primary_metric_for_well_resamples(self):
        control_sse = np.asarray(
            [
                [4.0, 1.0, 9.0, 16.0],
                [1.0, 4.0, 4.0, 9.0],
                [9.0, 9.0, 1.0, 4.0],
                [16.0, 4.0, 9.0, 1.0],
            ],
            dtype="<f8",
        )
        sg_sse = np.asarray(
            [
                [1.0, 1.0, 4.0, 9.0],
                [1.0, 1.0, 4.0, 4.0],
                [4.0, 9.0, 1.0, 1.0],
                [9.0, 4.0, 4.0, 1.0],
            ],
            dtype="<f8",
        )
        counts = np.asarray([2, 2, 2, 2], dtype="<i8")

        summary = paired_well_primary_statistics(
            control_sse,
            sg_sse,
            counts,
            bootstrap_resamples=32,
            permutation_resamples=64,
            confidence_level=0.95,
            random_seed=7,
        )

        self.assertEqual(
            summary["control_macro_normalized_rmse"],
            5.5223170640536985,
        )
        self.assertEqual(
            summary["sg_macro_normalized_rmse"],
            4.205214109130424,
        )
        self.assertEqual(
            summary["effect_percentage_points"],
            131.71029549232748,
        )
        self.assertEqual(
            summary["bootstrap"]["effect_ci_percentage_points"],
            [107.5379006642347, 147.9683998451524],
        )
        self.assertEqual(
            summary["permutation"]["extreme_resamples"],
            11,
        )
        self.assertEqual(
            summary["permutation"]["p_value"],
            12 / 65,
        )
        self.assertEqual(
            summary["permutation"]["p_value_correction"],
            "plus_one",
        )
        self.assertEqual(summary["statistics_unit"], "physical_well")
        self.assertNotEqual(
            summary["effect_percentage_points"],
            124.29611388044779,
        )

    def test_statistics_reject_misaligned_nonfinite_or_invalid_counts(self):
        valid = np.ones((4, 4), dtype="<f8")
        counts = np.ones(4, dtype="<i8")
        cases = (
            (valid[:, :3], valid, counts),
            (valid, valid[:3], counts),
            (valid, valid, counts[:3]),
            (valid * np.nan, valid, counts),
            (valid, valid, np.asarray([1, 1, 0, 1], dtype="<i8")),
        )
        for control, sg, current_counts in cases:
            with self.subTest(
                control_shape=control.shape,
                sg_shape=sg.shape,
                count_shape=current_counts.shape,
            ):
                with self.assertRaises(D4AggregateValidationError):
                    paired_well_primary_statistics(
                        control,
                        sg,
                        current_counts,
                        bootstrap_resamples=32,
                        permutation_resamples=64,
                        confidence_level=0.95,
                        random_seed=7,
                    )


class D4AggregateContractTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        for seed, rows in frozen_rows_by_seed().items():
            result = seed_result(seed, rows)
            (self.root / f"seed{seed}_complete.json").write_bytes(
                canonical_json_bytes(result)
            )
            (
                self.root
                / f"paired_predictions_seed{seed}_complete.jsonl"
            ).write_bytes(
                b"".join(canonical_json_bytes(row) for row in rows)
            )

    def test_aggregate_reconstructs_exhaustive_wells_and_gate_contract(self):
        result = aggregate_d4_results(self.root)

        self.assertEqual(
            result["schema_version"],
            "phase05-d4-complete-cell-v1",
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_level"], "complete_cell")
        self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(result["statistics_unit"], "physical_well")
        self.assertEqual(len(result["per_seed"]), 5)
        self.assertEqual(len(result["per_well"]), 240)
        self.assertEqual(len(result["input_results"]), 5)
        self.assertEqual(len(result["paired_prediction_artifacts"]), 5)
        self.assertEqual(
            result["pooled_metrics"][CONTROL_CONDITION_ID][
                "sample_count"
            ],
            7680,
        )
        primary = result["primary_summary"]
        self.assertEqual(primary["bootstrap"]["sample_size"], 240)
        self.assertEqual(primary["bootstrap"]["resamples"], 10000)
        self.assertEqual(primary["permutation"]["sample_size"], 240)
        self.assertEqual(primary["permutation"]["resamples"], 100000)
        self.assertEqual(
            primary["permutation"]["p_value"],
            (
                primary["permutation"]["extreme_resamples"] + 1
            )
            / 100001,
        )
        self.assertEqual(
            result["effect_exceeds_threshold"],
            primary["effect_percentage_points"] > 2.0,
        )
        self.assertEqual(
            result["significant_at_configured_threshold"],
            primary["permutation"]["p_value"] < 0.05,
        )
        self.assertEqual(
            result["d4_gate_success"],
            result["effect_exceeds_threshold"]
            and result["significant_at_configured_threshold"],
        )
        canonical_json_bytes(result)

    def test_aggregate_rejects_payload_drift(self):
        path = self.root / "paired_predictions_seed3_complete.jsonl"
        path.write_bytes(path.read_bytes() + b"{}\n")
        with self.assertRaisesRegex(
            D4AggregateValidationError,
            "seed 3 paired predictions",
        ):
            aggregate_d4_results(self.root)

    def test_aggregate_rejects_code_or_frozen_split_drift(self):
        path = self.root / "seed2_complete.json"
        result = json.loads(path.read_bytes())
        result["code"]["rpe/runner/d4_sugar.py"]["sha256"] = "f" * 64
        path.write_bytes(canonical_json_bytes(result))
        with self.assertRaisesRegex(
            D4AggregateValidationError,
            "seed 2 result",
        ):
            aggregate_d4_results(self.root)

        rows = frozen_rows_by_seed()[2]
        result = seed_result(2, rows)
        result["split"]["test_record_ids_sha256"] = "e" * 64
        path.write_bytes(canonical_json_bytes(result))
        with self.assertRaisesRegex(
            D4AggregateValidationError,
            "seed 2 result",
        ):
            aggregate_d4_results(self.root)

    def test_missing_complete_seed_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(
                D4AggregateValidationError,
                "seed 0",
            ):
                aggregate_d4_results(root)

    def test_cli_reports_missing_seed_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = tuple(root.iterdir())
            completed = subprocess.run(
                [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "tools" / "aggregate_phase05_d4.py"),
                    "--result-root",
                    str(root),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=False,
            )

            self.assertEqual(completed.returncode, 1)
            self.assertEqual(completed.stdout, b"")
            self.assertEqual(completed.stderr.count(b"\n"), 1)
            error = json.loads(completed.stderr)
            self.assertEqual(error["status"], "failed")
            self.assertIn("seed 0", error["error"])
            self.assertEqual(completed.stderr, canonical_json_bytes(error))
            self.assertEqual(tuple(root.iterdir()), before)

    def test_cli_emits_one_canonical_line_without_writes(self):
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.root.iterdir()
        }
        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "aggregate_phase05_d4.py"),
                "--result-root",
                str(self.root),
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(completed.stdout.count(b"\n"), 1)
        result = json.loads(completed.stdout)
        self.assertEqual(completed.stdout, canonical_json_bytes(result))
        self.assertEqual(result["result_level"], "complete_cell")
        after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.root.iterdir()
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
