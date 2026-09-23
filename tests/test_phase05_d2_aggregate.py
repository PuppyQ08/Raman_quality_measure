from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.d2_aggregate import (  # noqa: E402
    D2AggregateValidationError,
    aggregate_d2_results,
    load_d2_aggregate_config,
)


CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d2_three_shot_aggregate.json"
)
CONFIG_SHA256 = (
    "25162255aabaf073f3c6365172122513f85b3c3d8a2fa79cbe04d1faf3fe197d"
)
RUNNER_SHA = (
    "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7"
)
SELECTION_SHA = (
    "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36"
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


def prediction(
    condition_id: str,
    shot_count: int,
    seed: int,
    index: int,
) -> dict[str, object]:
    true_class = index % 30
    control_correct = 1200 + shot_count * 10 + seed
    gain = {5: 300, 10: 150, 20: 60}[shot_count]
    correct_count = (
        control_correct
        if condition_id == CONTROL
        else control_correct + gain
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


def result_document(shot_count: int, seed: int) -> dict[str, object]:
    conditions = []
    for condition_id in (CONTROL, SG):
        predictions = [
            prediction(condition_id, shot_count, seed, index)
            for index in range(3000)
        ]
        accuracy = sum(item["correct"] for item in predictions) / 3000
        conditions.append(
            {
                "condition_id": condition_id,
                "metrics": {
                    "accuracy": accuracy,
                    "balanced_accuracy": accuracy,
                    "confusion_matrix": [[1]],
                    "macro_f1": accuracy,
                    "per_class_recall": [accuracy] * 30,
                },
                "model": {
                    "classifier": "logistic_regression",
                    "convergence_warnings": 0,
                },
                "predictions": predictions,
                "runtime_seconds": 1.0,
                "selected_c": 10.0,
                "validation_scores": [
                    {"accuracy": 0.1, "c": 0.01},
                    {"accuracy": 0.2, "c": 0.1},
                    {"accuracy": 0.3, "c": 1.0},
                    {"accuracy": 0.4, "c": 10.0},
                ],
            }
        )
    test_digest = hashlib.sha256(
        (
            "\n".join(f"test-{index:06d}" for index in range(3000))
            + "\n"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "code": {"runner.py": {"bytes": 1, "sha256": "a" * 64}},
        "conditions": conditions,
        "config": {
            "bytes": 814,
            "path": "d2_bacteria_id_pca20_lr_sg11.json",
            "sha256": RUNNER_SHA,
        },
        "dataset": {
            "dataset_id": "bacteria_id_reference",
            "files": {"SHA256SUMS.sha256": {"bytes": 77, "sha256": "b" * 64}},
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
        "experiment_id": "d2_bacteria_id_few_shot_pca20_lr_sg11",
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
                "train": shot_count * 30,
                "validation": 300,
            },
        },
        "model_config": {
            "bytes": 1051,
            "path": "d1_bacteria_id_pca20_lr_sg11.json",
            "sha256": (
                "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"
            ),
        },
        "pipeline_configs": {
            SG: {
                "bytes": 338,
                "path": "d1_sg11_poly3_interp.json",
                "sha256": (
                    "775c3dba4c4d6bb13ab0ac44abf3e48011f90ce8bc268f7f79632ed35cb039aa"
                ),
            }
        },
        "result_level": "provisional",
        "runtime_seconds": 2.0,
        "schema_version": "phase05-d2-result-v1",
        "seed": seed,
        "selection": {
            "artifact_bytes": 254678,
            "artifact_path": "selection.json",
            "artifact_sha256": SELECTION_SHA,
            "code": {"selector.py": {"bytes": 1, "sha256": "c" * 64}},
            "config_sha256": (
                "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"
            ),
            "train_record_ids_sha256": (
                hashlib.sha256(
                    f"train-{shot_count}-{seed}".encode()
                ).hexdigest()
            ),
            "validation_record_ids_sha256": (
                hashlib.sha256(f"validation-{seed}".encode()).hexdigest()
            ),
        },
        "shot_count": shot_count,
        "split": {
            "test": {
                "class_counts": {str(label): 100 for label in range(30)},
                "count": 3000,
                "record_ids_sha256": test_digest,
            },
            "train": {
                "class_counts": {
                    str(label): shot_count
                    for label in range(30)
                },
                "count": shot_count * 30,
                "record_ids_sha256": hashlib.sha256(
                    f"train-{shot_count}-{seed}".encode()
                ).hexdigest(),
            },
            "validation": {
                "class_counts": {str(label): 10 for label in range(30)},
                "count": 300,
                "record_ids_sha256": hashlib.sha256(
                    f"validation-{seed}".encode()
                ).hexdigest(),
            },
            "validation_per_class": 10,
        },
        "status": "completed",
    }


def prediction_bytes(result: dict[str, object]) -> bytes:
    return b"".join(
        canonical_json_bytes(
            {"condition_id": condition["condition_id"], **item}
        )
        for condition in result["conditions"]
        for item in condition["predictions"]
    )


class D2AggregateTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.config = self.root / CONFIG.name
        self.config.write_bytes(CONFIG.read_bytes())
        for shot_count in (5, 10, 20):
            for seed in range(5):
                result = result_document(shot_count, seed)
                (self.root / f"seed{seed}_{shot_count}shot.json").write_bytes(
                    canonical_json_bytes(result)
                )
                (
                    self.root
                    / f"predictions_seed{seed}_{shot_count}shot.jsonl"
                ).write_bytes(prediction_bytes(result))

    def test_config_loads_three_shot_five_seed_contract(self):
        config = load_d2_aggregate_config(self.config)

        self.assertEqual(config.seeds, (0, 1, 2, 3, 4))
        self.assertEqual(config.shot_counts, (5, 10, 20))
        self.assertEqual(config.resamples, 2000)
        self.assertEqual(config.random_seed, 20260817)
        self.assertEqual(config.significance_threshold, 0.05)
        self.assertEqual(CONFIG.stat().st_size, 882)
        self.assertEqual(
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            CONFIG_SHA256,
        )

    def test_aggregate_emits_complete_statistics_for_each_shot(self):
        result = aggregate_d2_results(self.config, self.root)

        self.assertEqual(
            result["schema_version"],
            "phase05-d2-complete-cells-v1",
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_level"], "complete_cells")
        self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(result["shot_counts"], [5, 10, 20])
        self.assertEqual(set(result["cells"]), {"5", "10", "20"})
        for shot_count in (5, 10, 20):
            cell = result["cells"][str(shot_count)]
            self.assertEqual(cell["shot_count"], shot_count)
            self.assertEqual(len(cell["per_seed"]), 5)
            self.assertEqual(cell["bootstrap"]["resamples"], 2000)
            self.assertEqual(cell["permutation"]["patterns"], 32)
            self.assertEqual(
                cell["effect_exceeds_threshold"],
                shot_count in (5, 10),
            )
            self.assertEqual(
                cell["significant_at_configured_threshold"],
                False,
            )
            self.assertEqual(cell["gate_success"], False)
        self.assertEqual(result["gate_success_shot_counts"], [])
        self.assertFalse(result["d2_gate_success"])
        self.assertEqual(len(result["input_results"]), 15)
        self.assertEqual(len(result["prediction_artifacts"]), 15)
        canonical_json_bytes(result)

    def test_aggregate_rejects_missing_cell_or_selection_drift(self):
        (self.root / "seed4_20shot.json").unlink()
        with self.assertRaisesRegex(
            D2AggregateValidationError,
            "seed 4 shot 20",
        ):
            aggregate_d2_results(self.config, self.root)

        result = result_document(20, 4)
        (self.root / "seed4_20shot.json").write_bytes(
            canonical_json_bytes(result)
        )
        result = result_document(10, 2)
        result["selection"]["artifact_sha256"] = "f" * 64
        (self.root / "seed2_10shot.json").write_bytes(
            canonical_json_bytes(result)
        )
        (self.root / "predictions_seed2_10shot.jsonl").write_bytes(
            prediction_bytes(result)
        )
        with self.assertRaisesRegex(
            D2AggregateValidationError,
            "selection",
        ):
            aggregate_d2_results(self.config, self.root)

    def test_aggregate_rejects_metric_or_paired_prediction_drift(self):
        result = result_document(5, 1)
        result["conditions"][0]["metrics"]["accuracy"] = 0.99
        (self.root / "seed1_5shot.json").write_bytes(
            canonical_json_bytes(result)
        )
        (self.root / "predictions_seed1_5shot.jsonl").write_bytes(
            prediction_bytes(result)
        )
        with self.assertRaisesRegex(
            D2AggregateValidationError,
            "accuracy mismatch",
        ):
            aggregate_d2_results(self.config, self.root)

        result = result_document(5, 1)
        predictions = result["conditions"][1]["predictions"]
        predictions[0], predictions[1] = predictions[1], predictions[0]
        (self.root / "seed1_5shot.json").write_bytes(
            canonical_json_bytes(result)
        )
        (self.root / "predictions_seed1_5shot.jsonl").write_bytes(
            prediction_bytes(result)
        )
        with self.assertRaisesRegex(
            D2AggregateValidationError,
            "paired prediction",
        ):
            aggregate_d2_results(self.config, self.root)

    def test_cli_emits_one_canonical_line_without_writes(self):
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.root.iterdir()
            if path.is_file()
        }
        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "aggregate_phase05_d2.py"),
                "--config",
                str(self.config),
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
        document = json.loads(completed.stdout)
        self.assertEqual(completed.stdout, canonical_json_bytes(document))
        self.assertEqual(document["result_level"], "complete_cells")
        after = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.root.iterdir()
            if path.is_file()
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
