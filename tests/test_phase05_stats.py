from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.stats.paired import (  # noqa: E402
    PairedStatisticsValidationError,
    exact_two_sided_sign_flip,
    paired_seed_bootstrap,
)
from rpe.runner.d1_aggregate import (  # noqa: E402
    D1AggregateValidationError,
    aggregate_d1_results,
    load_d1_aggregate_config,
)


AGGREGATE_CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d1_five_seed_aggregate.json"
)
AGGREGATE_CONFIG_SHA256 = (
    "fd72d323beaaf1c9d4d52b7dee6a0bc1c8a4dc5987b3da218c8f480d6e6f4051"
)
RUNNER_CONFIG_SHA256 = (
    "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"
)
CONTROL = "released_input_control"
SG = "released_input_plus_sg"


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


def prediction(condition_id: str, seed: int, index: int) -> dict[str, object]:
    true_class = index % 30
    correct_count = (
        2100 + 3 * seed
        if condition_id == CONTROL
        else 2115 + 3 * seed
    )
    predicted_class = (
        true_class
        if index < correct_count
        else (true_class + 1) % 30
    )
    return {
        "correct": predicted_class == true_class,
        "predicted_class": predicted_class,
        "record_id": f"test-{index:06d}",
        "true_class": true_class,
    }


def seed_result(seed: int) -> dict[str, object]:
    conditions = []
    for condition_id, accuracy in (
        (CONTROL, 0.70 + seed * 0.001),
        (SG, 0.705 + seed * 0.001),
    ):
        predictions = [
            prediction(condition_id, seed, index)
            for index in range(3000)
        ]
        conditions.append(
            {
                "condition_id": condition_id,
                "metrics": {
                    "accuracy": accuracy,
                    "balanced_accuracy": accuracy,
                    "confusion_matrix": [[2, 0], [0, 2]],
                    "macro_f1": accuracy,
                    "per_class_recall": [accuracy, accuracy],
                },
                "model": {
                    "classifier": "logistic_regression",
                    "convergence_warnings": 0,
                },
                "predictions": predictions,
                "runtime_seconds": 1.0,
                "selected_c": 10.0,
                "validation_scores": [
                    {"accuracy": 0.5, "c": 0.01},
                    {"accuracy": 0.6, "c": 0.1},
                    {"accuracy": 0.7, "c": 1.0},
                    {"accuracy": 0.8, "c": 10.0},
                ],
            }
        )
    return {
        "code": {"runner.py": {"bytes": 1, "sha256": "a" * 64}},
        "conditions": conditions,
        "config": {
            "bytes": 1051,
            "path": "d1_bacteria_id_pca20_lr_sg11.json",
            "sha256": RUNNER_CONFIG_SHA256,
        },
        "dataset": {
            "dataset_id": "bacteria_id_reference",
            "files": {
                "SHA256SUMS.sha256": {
                    "bytes": 77,
                    "sha256": (
                        "605866e2953479534e1830d759790afe71f6a61f"
                        "fa38f3dbd239895dfa39be02"
                    ),
                }
            },
            "record_count": 66000,
            "split_counts": {
                "finetune": 3000,
                "reference": 60000,
                "test": 3000,
            },
        },
        "environment": {
            "git_commit": None,
            "thread_environment": {"OMP_NUM_THREADS": "1"},
            "vcs_status": "unavailable",
        },
        "experiment_id": "d1_bacteria_id_pca20_lr_sg11",
        "leakage_audit": {
            "exact_intensity_duplicate_pair_counts": {
                "train_test": 0,
                "train_validation": 0,
                "validation_test": 0,
            },
            "nearest_train_cosine_similarity_quantiles": {"1.0": 0.9},
            "record_id_overlap_counts": {
                "train_test": 0,
                "train_validation": 0,
                "validation_test": 0,
            },
            "source_coordinate_overlap_counts": {
                "train_test": 0,
                "train_validation": 0,
                "validation_test": 0,
            },
            "unique_source_coordinate_counts": {
                "test": 3000,
                "train": 62700,
                "validation": 300,
            },
        },
        "pipeline_configs": {
            SG: {
                "bytes": 338,
                "path": "d1_sg11_poly3_interp.json",
                "sha256": (
                    "775c3dba4c4d6bb13ab0ac44abf3e48011f90ce"
                    "8bc268f7f79632ed35cb039aa"
                ),
            }
        },
        "result_level": "provisional",
        "runtime_seconds": 2.0,
        "schema_version": "phase05-d1-result-v1",
        "seed": seed,
        "split": {
            "test": {
                "class_counts": {
                    str(class_label): 100
                    for class_label in range(30)
                },
                "count": 3000,
                "record_ids_sha256": hashlib.sha256(
                    (
                        "\n".join(
                            f"test-{index:06d}"
                            for index in range(3000)
                        )
                        + "\n"
                    ).encode("utf-8")
                ).hexdigest(),
            },
            "train": {
                "class_counts": {
                    str(class_label): 2090
                    for class_label in range(30)
                },
                "count": 62700,
                "record_ids_sha256": "c" * 64,
            },
            "validation": {
                "class_counts": {
                    str(class_label): 10
                    for class_label in range(30)
                },
                "count": 300,
                "record_ids_sha256": "d" * 64,
            },
            "validation_per_class": 10,
        },
        "status": "completed",
    }


def prediction_jsonl_bytes(result: dict[str, object]) -> bytes:
    lines = []
    for condition in result["conditions"]:
        for item in condition["predictions"]:
            lines.append(
                canonical_json_bytes(
                    {
                        "condition_id": condition["condition_id"],
                        **item,
                    }
                )
            )
    return b"".join(lines)


class PairedSeedStatisticsTest(unittest.TestCase):
    def test_constant_bootstrap_has_point_confidence_interval(self):
        summary = paired_seed_bootstrap(
            [0.7] * 5,
            [0.705] * 5,
            resamples=2000,
            confidence_level=0.95,
            random_seed=20260817,
            scale=100.0,
        )

        self.assertEqual(summary["sample_size"], 5)
        self.assertAlmostEqual(summary["left_mean"], 70.0)
        self.assertAlmostEqual(summary["right_mean"], 70.5)
        self.assertAlmostEqual(summary["difference_mean"], 0.5)
        self.assertAlmostEqual(summary["left_ci"][0], 70.0)
        self.assertAlmostEqual(summary["left_ci"][1], 70.0)
        self.assertAlmostEqual(summary["right_ci"][0], 70.5)
        self.assertAlmostEqual(summary["right_ci"][1], 70.5)
        self.assertAlmostEqual(summary["difference_ci"][0], 0.5)
        self.assertAlmostEqual(summary["difference_ci"][1], 0.5)
        self.assertEqual(summary["resamples"], 2000)
        self.assertEqual(summary["random_seed"], 20260817)

    def test_exact_two_sided_sign_flip_minimum_is_two_of_thirty_two(self):
        result = exact_two_sided_sign_flip(
            [0.1, 0.2, 0.3, 0.4, 0.5]
        )

        self.assertEqual(result["sample_size"], 5)
        self.assertEqual(result["patterns"], 32)
        self.assertAlmostEqual(result["observed_mean"], 0.3)
        self.assertEqual(result["extreme_patterns"], 2)
        self.assertEqual(result["p_value"], 0.0625)
        self.assertEqual(result["alternative"], "two-sided")
        self.assertEqual(result["zero_difference"], "included")

    def test_bootstrap_difference_mean_uses_direct_paired_differences(self):
        left = [
            0.7,
            0.703,
            0.6953333333333334,
            0.7023333333333334,
            0.7033333333333334,
        ]
        right = [
            0.7023333333333334,
            0.7056666666666667,
            0.6986666666666667,
            0.702,
            0.684,
        ]

        summary = paired_seed_bootstrap(
            left,
            right,
            resamples=2000,
            confidence_level=0.95,
            random_seed=20260817,
            scale=100.0,
        )

        expected = float(
            np.mean(
                np.asarray(right, dtype=np.float64)
                - np.asarray(left, dtype=np.float64)
            )
            * 100.0
        )
        self.assertEqual(summary["difference_mean"], expected)

    def test_statistics_reject_mismatched_nonfinite_or_wrong_sample_size(self):
        cases = (
            lambda: paired_seed_bootstrap(
                [0.1] * 5,
                [0.2] * 4,
                resamples=2000,
                confidence_level=0.95,
                random_seed=1,
                scale=100.0,
            ),
            lambda: paired_seed_bootstrap(
                [0.1] * 5,
                [0.2, 0.2, 0.2, 0.2, float("nan")],
                resamples=2000,
                confidence_level=0.95,
                random_seed=1,
                scale=100.0,
            ),
            lambda: exact_two_sided_sign_flip([0.1] * 4),
        )
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaises(PairedStatisticsValidationError):
                    case()


class D1AggregateTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = self.root / AGGREGATE_CONFIG.name
        self.config.write_bytes(AGGREGATE_CONFIG.read_bytes())
        self.result_root = self.root / RUNNER_CONFIG_SHA256
        self.result_root.mkdir()
        for seed in range(5):
            result = seed_result(seed)
            (self.result_root / f"{seed}.json").write_bytes(
                canonical_json_bytes(result)
            )
            (self.result_root / f"predictions_{seed}.jsonl").write_bytes(
                prediction_jsonl_bytes(result)
            )

    def test_config_loads_exact_frozen_statistics_contract(self):
        config = load_d1_aggregate_config(self.config)

        self.assertEqual(config.seeds, (0, 1, 2, 3, 4))
        self.assertEqual(config.resamples, 2000)
        self.assertEqual(config.confidence_level, 0.95)
        self.assertEqual(config.random_seed, 20260817)
        self.assertEqual(config.sign_flip_patterns, 32)
        self.assertEqual(config.effect_threshold_percentage_points, 2.0)
        self.assertEqual(
            hashlib.sha256(self.config.read_bytes()).hexdigest(),
            AGGREGATE_CONFIG_SHA256,
        )

    def test_aggregate_emits_complete_cell_with_seed_and_paired_statistics(self):
        result = aggregate_d1_results(self.config, self.result_root)

        self.assertEqual(result["schema_version"], "phase05-d1-complete-cell-v1")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_level"], "complete_cell")
        self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(
            [row["seed"] for row in result["per_seed"]],
            [0, 1, 2, 3, 4],
        )
        for value in [
            row["difference_percentage_points"]
            for row in result["per_seed"]
        ]:
            self.assertAlmostEqual(value, 0.5)
        summary = result["primary_metric_summary"]
        self.assertAlmostEqual(summary["control_mean_percentage"], 70.2)
        self.assertAlmostEqual(summary["sg_mean_percentage"], 70.7)
        self.assertAlmostEqual(summary["difference_mean_percentage_points"], 0.5)
        self.assertEqual(summary["bootstrap"]["resamples"], 2000)
        self.assertEqual(summary["permutation"]["patterns"], 32)
        self.assertEqual(summary["permutation"]["p_value"], 0.0625)
        self.assertFalse(result["effect_exceeds_threshold"])
        self.assertEqual(result["gate_role"], "control_not_counted")
        self.assertEqual(len(result["input_results"]), 5)
        self.assertEqual(
            set(result["aggregate_code"]),
            {
                "rpe/runner/d1_aggregate.py",
                "rpe/stats/paired.py",
                "tools/aggregate_phase05.py",
            },
        )
        canonical_json_bytes(result)

    def test_aggregate_rejects_missing_seed_or_identity_drift(self):
        (self.result_root / "4.json").unlink()
        with self.assertRaisesRegex(D1AggregateValidationError, "seed 4"):
            aggregate_d1_results(self.config, self.result_root)

        (self.result_root / "4.json").write_bytes(
            canonical_json_bytes(seed_result(4))
        )
        mutated = seed_result(3)
        mutated["config"]["sha256"] = "f" * 64
        (self.result_root / "3.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        with self.assertRaisesRegex(D1AggregateValidationError, "config"):
            aggregate_d1_results(self.config, self.result_root)

    def test_aggregate_rejects_missing_or_mismatched_prediction_artifact(self):
        (self.result_root / "predictions_4.jsonl").unlink()
        with self.assertRaisesRegex(
            D1AggregateValidationError,
            "predictions seed 4",
        ):
            aggregate_d1_results(self.config, self.result_root)

        result = seed_result(4)
        lines = prediction_jsonl_bytes(result).splitlines(keepends=True)
        item = json.loads(lines[0])
        item["predicted_class"] = 1 - item["predicted_class"]
        lines[0] = canonical_json_bytes(item)
        (self.result_root / "predictions_4.jsonl").write_bytes(b"".join(lines))
        with self.assertRaisesRegex(
            D1AggregateValidationError,
            "prediction projection",
        ):
            aggregate_d1_results(self.config, self.result_root)

    def test_aggregate_rejects_test_record_order_or_condition_drift(self):
        mutated = seed_result(2)
        mutated["split"]["test"]["record_ids_sha256"] = "e" * 64
        (self.result_root / "2.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        with self.assertRaisesRegex(D1AggregateValidationError, "test record"):
            aggregate_d1_results(self.config, self.result_root)

        (self.result_root / "2.json").write_bytes(
            canonical_json_bytes(seed_result(2))
        )
        mutated = deepcopy(seed_result(2))
        mutated["conditions"].reverse()
        (self.result_root / "2.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        with self.assertRaisesRegex(D1AggregateValidationError, "conditions"):
            aggregate_d1_results(self.config, self.result_root)

    def test_aggregate_rejects_metric_or_paired_prediction_drift(self):
        mutated = seed_result(1)
        mutated["conditions"][0]["metrics"]["accuracy"] = 0.99
        (self.result_root / "1.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        (self.result_root / "predictions_1.jsonl").write_bytes(
            prediction_jsonl_bytes(mutated)
        )
        with self.assertRaisesRegex(
            D1AggregateValidationError,
            "accuracy mismatch",
        ):
            aggregate_d1_results(self.config, self.result_root)

        mutated = seed_result(1)
        sg_predictions = mutated["conditions"][1]["predictions"]
        sg_predictions[0], sg_predictions[1] = (
            sg_predictions[1],
            sg_predictions[0],
        )
        (self.result_root / "1.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        (self.result_root / "predictions_1.jsonl").write_bytes(
            prediction_jsonl_bytes(mutated)
        )
        with self.assertRaisesRegex(
            D1AggregateValidationError,
            "paired prediction",
        ):
            aggregate_d1_results(self.config, self.result_root)

    def test_aggregate_rejects_code_or_environment_drift(self):
        mutated = seed_result(3)
        mutated["code"]["runner.py"]["sha256"] = "f" * 64
        (self.result_root / "3.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        (self.result_root / "predictions_3.jsonl").write_bytes(
            prediction_jsonl_bytes(mutated)
        )
        with self.assertRaisesRegex(
            D1AggregateValidationError,
            "code identity",
        ):
            aggregate_d1_results(self.config, self.result_root)

        mutated = seed_result(3)
        mutated["environment"]["thread_environment"][
            "OMP_NUM_THREADS"
        ] = "2"
        (self.result_root / "3.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        (self.result_root / "predictions_3.jsonl").write_bytes(
            prediction_jsonl_bytes(mutated)
        )
        with self.assertRaisesRegex(
            D1AggregateValidationError,
            "environment identity",
        ):
            aggregate_d1_results(self.config, self.result_root)

    def test_aggregate_rejects_split_or_leakage_contract_drift(self):
        mutated = seed_result(1)
        mutated["split"]["validation"]["count"] = 299
        (self.result_root / "1.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        (self.result_root / "predictions_1.jsonl").write_bytes(
            prediction_jsonl_bytes(mutated)
        )
        with self.assertRaisesRegex(
            D1AggregateValidationError,
            "split contract",
        ):
            aggregate_d1_results(self.config, self.result_root)

        mutated = seed_result(1)
        mutated["leakage_audit"]["record_id_overlap_counts"][
            "train_test"
        ] = 1
        (self.result_root / "1.json").write_bytes(
            canonical_json_bytes(mutated)
        )
        (self.result_root / "predictions_1.jsonl").write_bytes(
            prediction_jsonl_bytes(mutated)
        )
        with self.assertRaisesRegex(
            D1AggregateValidationError,
            "leakage contract",
        ):
            aggregate_d1_results(self.config, self.result_root)

    def test_aggregate_cli_emits_one_canonical_line_without_writes(self):
        before = {
            path.relative_to(self.root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "aggregate_phase05.py"),
                "--config",
                str(self.config),
                "--result-root",
                str(self.result_root),
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(completed.stdout.count(b"\n"), 1)
        document = json.loads(completed.stdout)
        self.assertEqual(
            completed.stdout,
            canonical_json_bytes(document),
        )
        self.assertEqual(document["result_level"], "complete_cell")
        after = {
            path.relative_to(self.root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
