from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rpe.io.schema import PreprocessingStatus, Targets  # noqa: E402
from rpe.io.store import write_dataset  # noqa: E402
from rpe.runner.d1_bacteria_id import (  # noqa: E402
    D1RunnerValidationError,
    load_d1_runner_config,
    run_d1_experiment,
)
from unified_helpers import corrected_record  # noqa: E402


DATASET_ID = "bacteria_id_reference"
RUNNER_CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d1_bacteria_id_pca20_lr_sg11.json"
)
SG_CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d1_sg11_poly3_interp.json"
)
RUNNER_CONFIG_SHA256 = (
    "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"
)
SG_CONFIG_SHA256 = (
    "775c3dba4c4d6bb13ab0ac44abf3e48011f90ce8bc268f7f79632ed35cb039aa"
)
RESULT_KEYS = {
    "code",
    "conditions",
    "config",
    "dataset",
    "environment",
    "experiment_id",
    "leakage_audit",
    "pipeline_configs",
    "result_level",
    "runtime_seconds",
    "schema_version",
    "seed",
    "split",
    "status",
}
CODE_PATHS = (
    "rpe/downstream/bacteria_id.py",
    "rpe/io/schema.py",
    "rpe/io/store.py",
    "rpe/methods/classical/savitzky_golay.py",
    "rpe/runner/d1_bacteria_id.py",
    "tools/run_phase05.py",
)
THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
CONDITION_KEYS = {
    "condition_id",
    "metrics",
    "model",
    "predictions",
    "runtime_seconds",
    "selected_c",
    "validation_scores",
}
METRIC_KEYS = {
    "accuracy",
    "balanced_accuracy",
    "confusion_matrix",
    "macro_f1",
    "per_class_recall",
}


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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            file_sha256(path),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def synthetic_spectrum(
    class_label: int,
    replicate: int,
    source_split: str,
) -> np.ndarray:
    coordinate = np.linspace(-1.0, 1.0, 25, dtype=np.float32)
    theta = np.float32(2.0 * np.pi * class_label / 30.0)
    jitter = np.float32((replicate - 5.5) * 0.001)
    split_offset = np.float32(
        {
            "finetune": 0.0,
            "reference": 0.0005,
            "test": -0.0005,
        }[source_split]
    )
    return np.asarray(
        np.float32(1.0)
        + (np.cos(theta) + jitter + split_offset) * coordinate
        + (np.sin(theta) - jitter) * coordinate**2
        + np.float32(0.05) * coordinate**3,
        dtype="<f4",
    )


def synthetic_record(
    source_split: str,
    source_row: int,
    class_label: int,
    replicate: int,
):
    base = corrected_record()
    return replace(
        base,
        record_id=f"{source_split}-{source_row:06d}",
        intensity=synthetic_spectrum(
            class_label,
            replicate,
            source_split,
        ),
        wavenumber=np.linspace(
            1800.0,
            400.0,
            25,
            dtype=np.float32,
        ),
        meta=replace(
            base.meta,
            dataset_id=DATASET_ID,
            sample_id=None,
            preprocessing_status=PreprocessingStatus.KNOWN_CORRECTED,
            source_metadata={
                "source_split": source_split,
                "source_row": source_row,
                "split_role": {
                    "finetune": "optical_system_adaptation",
                    "reference": "pretraining",
                    "test": "independent_test",
                }[source_split],
                "label_space": "isolate",
            },
        ),
        targets=Targets(class_label=class_label),
    )


def synthetic_records() -> list:
    records = []
    for source_split, per_class in (
        ("finetune", 12),
        ("reference", 2),
        ("test", 2),
    ):
        source_row = 0
        for class_label in range(30):
            for replicate in range(per_class):
                records.append(
                    synthetic_record(
                        source_split,
                        source_row,
                        class_label,
                        replicate,
                    )
                )
                source_row += 1
    return records


class D1FixedConfigRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.dataset_path = self.root / DATASET_ID
        self.config_root = self.root / "configs"
        self.config_root.mkdir()
        self.runner_config = self.config_root / RUNNER_CONFIG.name
        self.sg_config = self.config_root / SG_CONFIG.name
        self.runner_config.write_bytes(RUNNER_CONFIG.read_bytes())
        self.sg_config.write_bytes(SG_CONFIG.read_bytes())
        write_dataset(
            synthetic_records(),
            self.dataset_path,
            dataset_id=DATASET_ID,
            class_labels={
                class_label: f"class_{class_label:02d}"
                for class_label in range(30)
            },
        )

    def test_frozen_runner_config_loads_exact_approved_contract(self):
        config = load_d1_runner_config(self.runner_config)

        self.assertEqual(config.experiment_id, "d1_bacteria_id_pca20_lr_sg11")
        self.assertEqual(config.seeds, (0, 1, 2, 3, 4))
        self.assertEqual(config.pca_n_components, 20)
        self.assertEqual(config.validation_per_class, 10)
        self.assertEqual(config.c_grid, (0.01, 0.1, 1.0, 10.0))
        self.assertEqual(config.max_iter, 1000)
        self.assertEqual(config.tol, 0.0001)
        self.assertEqual(config.l1_ratio, 0.0)
        self.assertEqual(config.conditions[0].condition_id, "released_input_control")
        self.assertEqual(
            config.conditions[1].condition_id,
            "released_input_plus_sg",
        )
        self.assertEqual(
            hashlib.sha256(self.runner_config.read_bytes()).hexdigest(),
            RUNNER_CONFIG_SHA256,
        )

    def test_api_smoke_runs_paired_conditions_with_complete_result_schema(self):
        before = tree_snapshot(self.root)

        result = run_d1_experiment(
            self.runner_config,
            self.dataset_path,
            seed=0,
            result_level="smoke",
        )

        after = tree_snapshot(self.root)
        self.assertEqual(after, before)
        self.assertEqual(set(result), RESULT_KEYS)
        self.assertEqual(result["schema_version"], "phase05-d1-result-v1")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_level"], "smoke")
        self.assertEqual(result["seed"], 0)
        self.assertEqual(set(result["code"]), set(CODE_PATHS))
        for relative_path in CODE_PATHS:
            source_path = ROOT / relative_path
            self.assertEqual(
                result["code"][relative_path],
                {
                    "bytes": source_path.stat().st_size,
                    "sha256": file_sha256(source_path),
                },
            )
        self.assertEqual(result["config"]["bytes"], 1051)
        self.assertEqual(
            result["config"]["sha256"],
            RUNNER_CONFIG_SHA256,
        )
        self.assertEqual(
            result["pipeline_configs"],
            {
                "released_input_plus_sg": {
                    "bytes": 338,
                    "path": SG_CONFIG.name,
                    "sha256": SG_CONFIG_SHA256,
                }
            },
        )
        self.assertEqual(
            result["dataset"]["split_counts"],
            {"finetune": 360, "reference": 60, "test": 60},
        )
        self.assertEqual(result["dataset"]["record_count"], 480)
        self.assertEqual(
            result["dataset"]["files"],
            {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
                for path in sorted(self.dataset_path.iterdir())
                if path.is_file()
            },
        )
        self.assertEqual(result["split"]["train"]["count"], 120)
        self.assertEqual(result["split"]["validation"]["count"], 300)
        self.assertEqual(result["split"]["test"]["count"], 60)
        self.assertEqual(result["split"]["validation_per_class"], 10)
        self.assertEqual(
            [condition["condition_id"] for condition in result["conditions"]],
            ["released_input_control", "released_input_plus_sg"],
        )
        for condition in result["conditions"]:
            self.assertEqual(set(condition), CONDITION_KEYS)
            self.assertEqual(condition["selected_c"], 0.1)
            self.assertEqual(
                [entry["c"] for entry in condition["validation_scores"]],
                [0.01, 0.1, 1.0, 10.0],
            )
            self.assertEqual(set(condition["metrics"]), METRIC_KEYS)
            self.assertEqual(condition["metrics"]["accuracy"], 1.0)
            self.assertEqual(condition["metrics"]["macro_f1"], 1.0)
            self.assertEqual(condition["metrics"]["balanced_accuracy"], 1.0)
            self.assertEqual(len(condition["metrics"]["per_class_recall"]), 30)
            self.assertEqual(
                np.asarray(condition["metrics"]["confusion_matrix"]).shape,
                (30, 30),
            )
            self.assertEqual(len(condition["predictions"]), 60)
            self.assertEqual(
                set(condition["predictions"][0]),
                {"correct", "predicted_class", "record_id", "true_class"},
            )
            self.assertEqual(condition["model"]["pca_n_components"], 20)
            self.assertEqual(
                condition["model"]["classifier"],
                "logistic_regression",
            )
            self.assertEqual(condition["model"]["convergence_warnings"], 0)
        self.assertEqual(
            result["conditions"][0]["predictions"],
            result["conditions"][1]["predictions"],
        )
        self.assertEqual(
            result["leakage_audit"]["record_id_overlap_counts"],
            {
                "train_test": 0,
                "train_validation": 0,
                "validation_test": 0,
            },
        )
        self.assertEqual(
            result["leakage_audit"]["source_coordinate_overlap_counts"],
            {
                "train_test": 0,
                "train_validation": 0,
                "validation_test": 0,
            },
        )
        self.assertEqual(
            result["leakage_audit"]["unique_source_coordinate_counts"],
            {"test": 60, "train": 120, "validation": 300},
        )
        self.assertEqual(
            result["leakage_audit"]["exact_intensity_duplicate_pair_counts"],
            {
                "train_test": 0,
                "train_validation": 0,
                "validation_test": 0,
            },
        )
        quantiles = result["leakage_audit"][
            "nearest_train_cosine_similarity_quantiles"
        ]
        self.assertEqual(list(quantiles), ["0.5", "0.9", "0.95", "0.99", "1.0"])
        self.assertTrue(all(np.isfinite(value) for value in quantiles.values()))
        self.assertIsNone(result["environment"]["git_commit"])
        self.assertEqual(result["environment"]["vcs_status"], "unavailable")
        self.assertEqual(result["environment"]["scikit_learn"], "1.9.0")
        self.assertEqual(result["environment"]["compute_backend"], "cpu")
        self.assertIsInstance(result["environment"]["cpu_count"], int)
        self.assertGreater(result["environment"]["cpu_count"], 0)
        self.assertTrue(result["environment"]["cpu_model"])
        self.assertIsNone(result["environment"]["gpu_name"])
        self.assertIsNone(result["environment"]["cuda_runtime"])
        self.assertEqual(
            result["environment"]["thread_environment"],
            {
                key: os.environ.get(key)
                for key in THREAD_ENVIRONMENT_KEYS
            },
        )
        canonical_json_bytes(result)

    def test_same_seed_repeats_scientific_outputs(self):
        first = run_d1_experiment(
            self.runner_config,
            self.dataset_path,
            seed=0,
            result_level="smoke",
        )
        second = run_d1_experiment(
            self.runner_config,
            self.dataset_path,
            seed=0,
            result_level="smoke",
        )

        first_conditions = [
            {
                key: value
                for key, value in condition.items()
                if key != "runtime_seconds"
            }
            for condition in first["conditions"]
        ]
        second_conditions = [
            {
                key: value
                for key, value in condition.items()
                if key != "runtime_seconds"
            }
            for condition in second["conditions"]
        ]
        self.assertEqual(first["split"], second["split"])
        self.assertEqual(first_conditions, second_conditions)
        self.assertEqual(first["leakage_audit"], second["leakage_audit"])

    def test_runner_rejects_unlisted_seed_bad_level_and_config_mutation(self):
        with self.assertRaisesRegex(D1RunnerValidationError, "seed"):
            run_d1_experiment(
                self.runner_config,
                self.dataset_path,
                seed=99,
                result_level="smoke",
            )
        with self.assertRaisesRegex(D1RunnerValidationError, "result_level"):
            run_d1_experiment(
                self.runner_config,
                self.dataset_path,
                seed=0,
                result_level="paper",
            )
        with self.assertRaisesRegex(
            D1RunnerValidationError,
            "complete_cell",
        ):
            run_d1_experiment(
                self.runner_config,
                self.dataset_path,
                seed=0,
                result_level="complete_cell",
            )
        with self.assertRaisesRegex(
            D1RunnerValidationError,
            "retained snapshot",
        ):
            run_d1_experiment(
                self.runner_config,
                self.dataset_path,
                seed=0,
                result_level="provisional",
            )
        document = json.loads(self.runner_config.read_text(encoding="utf-8"))
        document["pca"]["n_components"] = 19
        self.runner_config.write_bytes(canonical_json_bytes(document))
        with self.assertRaisesRegex(D1RunnerValidationError, "pca"):
            load_d1_runner_config(self.runner_config)

    def test_runner_rejects_source_split_missing_one_of_thirty_classes(self):
        incomplete_root = self.root / "incomplete"
        incomplete_root.mkdir()
        incomplete_dataset = incomplete_root / DATASET_ID
        records = [
            record
            for record in synthetic_records()
            if not (
                record.meta.source_metadata["source_split"] == "test"
                and record.targets.class_label == 29
            )
        ]
        write_dataset(
            records,
            incomplete_dataset,
            dataset_id=DATASET_ID,
            class_labels={
                class_label: f"class_{class_label:02d}"
                for class_label in range(30)
            },
        )

        with self.assertRaisesRegex(
            D1RunnerValidationError,
            "test class coverage",
        ):
            run_d1_experiment(
                self.runner_config,
                incomplete_dataset,
                seed=0,
                result_level="smoke",
            )

    def test_cli_emits_one_canonical_json_line_without_writing_files(self):
        before = tree_snapshot(self.root)
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "NO_PROXY": "*",
            }
        )

        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "run_phase05.py"),
                "--config",
                str(self.runner_config),
                "--dataset",
                str(self.dataset_path),
                "--seed",
                "0",
                "--result-level",
                "smoke",
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr.decode())
        self.assertEqual(completed.stderr, b"")
        self.assertEqual(completed.stdout.count(b"\n"), 1)
        result = json.loads(completed.stdout)
        self.assertEqual(
            completed.stdout,
            canonical_json_bytes(result),
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_level"], "smoke")
        self.assertEqual(tree_snapshot(self.root), before)


if __name__ == "__main__":
    unittest.main()
