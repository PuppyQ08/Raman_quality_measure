from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rpe.io.store import write_dataset  # noqa: E402
from rpe.runner.d2_bacteria_id import (  # noqa: E402
    D2RunnerValidationError,
    load_d2_runner_config,
    run_d2_few_shot_experiment,
)
from tests.test_phase05_runner import synthetic_record  # noqa: E402


D2_CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d2_bacteria_id_pca20_lr_sg11.json"
)
D1_CONFIG = (
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
SELECTION_CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d2_few_shot_selection.json"
)
D2_CONFIG_SHA256 = (
    "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7"
)
CONDITIONS = (
    "released_input_control",
    "released_input_plus_sg",
)
EXPECTED_CODE_PATHS = {
    "rpe/downstream/bacteria_id.py",
    "rpe/io/schema.py",
    "rpe/io/store.py",
    "rpe/methods/classical/savitzky_golay.py",
    "rpe/runner/d1_bacteria_id.py",
    "rpe/runner/d2_bacteria_id.py",
    "tools/run_phase05_d2.py",
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            sha256_file(path),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def synthetic_records() -> list:
    records = []
    for source_split, per_class in (
        ("finetune", 30),
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


def ids_digest(record_ids: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(record_ids) + "\n").encode("utf-8")
    ).hexdigest()


def synthetic_selection(dataset: Path) -> dict[str, object]:
    records = [
        json.loads(line)
        for line in (dataset / "records.jsonl").read_bytes().splitlines()
    ]
    candidates = {class_label: [] for class_label in range(30)}
    test_ids = []
    for record in records:
        split = record["meta"]["source_metadata"]["source_split"]
        if split == "finetune":
            candidates[record["targets"]["class_label"]].append(
                (
                    record["meta"]["source_metadata"]["source_row"],
                    record["record_id"],
                )
            )
        elif split == "test":
            test_ids.append(record["record_id"])
    for values in candidates.values():
        values.sort()
    selections = []
    for seed in range(5):
        classes = []
        global_train = {5: [], 10: [], 20: []}
        global_validation = []
        for class_label in range(30):
            ordered_candidates = [
                record_id
                for _, record_id in candidates[class_label]
            ]
            permutation = np.random.Generator(
                np.random.PCG64(
                    np.random.SeedSequence([seed, class_label])
                )
            ).permutation(len(ordered_candidates))
            permuted = [
                ordered_candidates[int(index)]
                for index in permutation
            ]
            ordered_train = permuted[:20]
            validation = permuted[20:30]
            train_ids = {
                str(shot_count): ordered_train[:shot_count]
                for shot_count in (5, 10, 20)
            }
            classes.append(
                {
                    "candidate_count": 30,
                    "class_label": class_label,
                    "ordered_train_record_ids": ordered_train,
                    "source_split": "finetune",
                    "train_record_ids": train_ids,
                    "train_record_ids_sha256": {
                        str(shot_count): ids_digest(
                            train_ids[str(shot_count)]
                        )
                        for shot_count in (5, 10, 20)
                    },
                    "validation_record_ids": validation,
                    "validation_record_ids_sha256": ids_digest(validation),
                }
            )
            for shot_count in (5, 10, 20):
                global_train[shot_count].extend(
                    train_ids[str(shot_count)]
                )
            global_validation.extend(validation)
        selections.append(
            {
                "classes": classes,
                "seed": seed,
                "train_record_ids_sha256": {
                    str(shot_count): ids_digest(
                        global_train[shot_count]
                    )
                    for shot_count in (5, 10, 20)
                },
                "validation_record_ids_sha256": ids_digest(
                    global_validation
                ),
            }
        )
    return {
        "code": {
            path: {
                "bytes": (ROOT / path).stat().st_size,
                "sha256": sha256_file(ROOT / path),
            }
            for path in (
                "rpe/downstream/bacteria_id.py",
                "rpe/io/schema.py",
                "rpe/io/store.py",
                "rpe/runner/d2_selection.py",
                "tools/freeze_d2_selection.py",
            )
        },
        "conditions": [
            "released_input_control",
            "released_input_plus_sg",
        ],
        "config": {
            "bytes": 858,
            "path": "d2_few_shot_selection.json",
            "sha256": (
                "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"
            ),
        },
        "dataset": {
            "dataset_id": "bacteria_id_reference",
            "files": {},
            "record_count": 1020,
            "split_counts": {
                "finetune": 900,
                "reference": 60,
                "test": 60,
            },
        },
        "experiment_id": "d2_bacteria_id_few_shot",
        "schema_version": "phase05-d2-selection-artifact-v1",
        "seeds": [0, 1, 2, 3, 4],
        "selections": selections,
        "shot_counts": [5, 10, 20],
        "status": "frozen",
        "test": {
            "count": 60,
            "record_ids_sha256": ids_digest(test_ids),
            "source_split": "test",
        },
        "validation_per_class": 10,
    }


class D2FiveShotRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        self.dataset = self.root / "bacteria_id_reference"
        self.config_root = self.root / "configs"
        self.config_root.mkdir()
        for source in (
            D2_CONFIG,
            D1_CONFIG,
            SG_CONFIG,
            SELECTION_CONFIG,
        ):
            (self.config_root / source.name).write_bytes(source.read_bytes())
        self.d2_config = self.config_root / D2_CONFIG.name
        self.d1_config = self.config_root / D1_CONFIG.name
        self.selection_config = self.config_root / SELECTION_CONFIG.name
        write_dataset(
            synthetic_records(),
            self.dataset,
            dataset_id="bacteria_id_reference",
            class_labels={
                class_label: f"class_{class_label:02d}"
                for class_label in range(30)
            },
        )
        self.selection = synthetic_selection(self.dataset)
        self.selection_path = self.root / "selection.json"
        self.selection_path.write_bytes(
            canonical_json_bytes(self.selection)
        )

    def test_config_loads_exact_model_and_selection_contract(self):
        config = load_d2_runner_config(self.d2_config)

        self.assertEqual(
            config.experiment_id,
            "d2_bacteria_id_few_shot_pca20_lr_sg11",
        )
        self.assertEqual(config.seeds, (0, 1, 2, 3, 4))
        self.assertEqual(config.shot_counts, (5, 10, 20))
        self.assertEqual(
            config.accepted_selection_sha256,
            "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36",
        )
        self.assertEqual(D2_CONFIG.stat().st_size, 814)
        self.assertEqual(sha256_file(D2_CONFIG), D2_CONFIG_SHA256)

    def test_smoke_runs_seed_zero_five_shot_with_paired_predictions(self):
        before = tree_snapshot(self.root)

        result = run_d2_few_shot_experiment(
            self.d2_config,
            self.dataset,
            self.selection_path,
            seed=0,
            shot_count=5,
            result_level="smoke",
        )

        self.assertEqual(tree_snapshot(self.root), before)
        self.assertEqual(
            result["schema_version"],
            "phase05-d2-result-v1",
        )
        self.assertEqual(result["result_level"], "smoke")
        self.assertEqual(result["seed"], 0)
        self.assertEqual(result["shot_count"], 5)
        self.assertEqual(set(result["code"]), EXPECTED_CODE_PATHS)
        self.assertEqual(
            result["selection"],
            {
                "artifact_bytes": self.selection_path.stat().st_size,
                "artifact_path": self.selection_path.name,
                "artifact_sha256": sha256_file(self.selection_path),
                "config_sha256": (
                    "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"
                ),
                "code": self.selection["code"],
                "train_record_ids_sha256": (
                    self.selection["selections"][0][
                        "train_record_ids_sha256"
                    ]["5"]
                ),
                "validation_record_ids_sha256": (
                    self.selection["selections"][0][
                        "validation_record_ids_sha256"
                    ]
                ),
            },
        )
        self.assertEqual(result["split"]["train"]["count"], 150)
        self.assertEqual(result["split"]["validation"]["count"], 300)
        self.assertEqual(result["split"]["test"]["count"], 60)
        self.assertEqual(
            set(result["split"]["train"]["class_counts"].values()),
            {5},
        )
        self.assertEqual(
            set(result["split"]["validation"]["class_counts"].values()),
            {10},
        )
        self.assertEqual(
            [condition["condition_id"] for condition in result["conditions"]],
            list(CONDITIONS),
        )
        for condition in result["conditions"]:
            self.assertEqual(len(condition["predictions"]), 60)
            self.assertEqual(condition["model"]["pca_n_components"], 20)
            self.assertEqual(condition["model"]["convergence_warnings"], 0)
            self.assertEqual(
                [row["c"] for row in condition["validation_scores"]],
                [0.01, 0.1, 1.0, 10.0],
            )
        self.assertEqual(
            [row["record_id"] for row in result["conditions"][0]["predictions"]],
            [row["record_id"] for row in result["conditions"][1]["predictions"]],
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
            result["leakage_audit"]["unique_source_coordinate_counts"],
            {"test": 60, "train": 150, "validation": 300},
        )
        canonical_json_bytes(result)

    def test_runner_rejects_selection_code_identity_drift(self):
        mutated = json.loads(self.selection_path.read_bytes())
        mutated["code"]["rpe/runner/d2_selection.py"]["sha256"] = "f" * 64
        self.selection_path.write_bytes(canonical_json_bytes(mutated))

        with self.assertRaisesRegex(
            D2RunnerValidationError,
            "selection code",
        ):
            run_d2_few_shot_experiment(
                self.d2_config,
                self.dataset,
                self.selection_path,
                seed=0,
                shot_count=5,
                result_level="smoke",
            )

    def test_same_seed_and_selection_repeat_scientific_output(self):
        first = run_d2_few_shot_experiment(
            self.d2_config,
            self.dataset,
            self.selection_path,
            seed=0,
            shot_count=5,
            result_level="smoke",
        )
        second = run_d2_few_shot_experiment(
            self.d2_config,
            self.dataset,
            self.selection_path,
            seed=0,
            shot_count=5,
            result_level="smoke",
        )

        def scientific(result):
            return {
                "conditions": [
                    {
                        key: value
                        for key, value in condition.items()
                        if key != "runtime_seconds"
                    }
                    for condition in result["conditions"]
                ],
                "leakage_audit": result["leakage_audit"],
                "selection": result["selection"],
                "split": result["split"],
            }

        self.assertEqual(scientific(first), scientific(second))

    def test_runner_rejects_unlisted_seed_shot_or_false_provisional(self):
        for seed, shot_count, result_level, expected in (
            (99, 5, "smoke", "seed"),
            (0, 7, "smoke", "shot_count"),
            (0, 5, "provisional", "selection"),
        ):
            with self.subTest(
                seed=seed,
                shot_count=shot_count,
                result_level=result_level,
            ):
                with self.assertRaisesRegex(
                    D2RunnerValidationError,
                    expected,
                ):
                    run_d2_few_shot_experiment(
                        self.d2_config,
                        self.dataset,
                        self.selection_path,
                        seed=seed,
                        shot_count=shot_count,
                        result_level=result_level,
                    )

    def test_cli_emits_one_canonical_line_without_writes(self):
        before = tree_snapshot(self.root)
        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "run_phase05_d2.py"),
                "--config",
                str(self.d2_config),
                "--dataset",
                str(self.dataset),
                "--selection",
                str(self.selection_path),
                "--seed",
                "0",
                "--shot-count",
                "5",
                "--result-level",
                "smoke",
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
        result = json.loads(completed.stdout)
        self.assertEqual(
            completed.stdout,
            canonical_json_bytes(result),
        )
        self.assertEqual(result["shot_count"], 5)
        self.assertEqual(tree_snapshot(self.root), before)


if __name__ == "__main__":
    unittest.main()
