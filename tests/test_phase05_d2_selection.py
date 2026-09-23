from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.d2_selection import (  # noqa: E402
    D2SelectionValidationError,
    build_d2_few_shot_selection,
    load_d2_selection_config,
    validate_d2_few_shot_selection,
)


CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d2_few_shot_selection.json"
)
DATASET = ROOT / "data" / "unified" / "bacteria_id_reference"
CONFIG_SHA256 = (
    "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"
)
CONDITIONS = (
    "released_input_control",
    "released_input_plus_sg",
)
CODE_PATHS = (
    "rpe/downstream/bacteria_id.py",
    "rpe/io/schema.py",
    "rpe/io/store.py",
    "rpe/runner/d2_selection.py",
    "tools/freeze_d2_selection.py",
)
SEEDS = tuple(range(5))
SHOT_COUNTS = (5, 10, 20)


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


class D2FewShotSelectionTest(unittest.TestCase):
    def test_config_loads_exact_frozen_sampling_contract(self):
        config = load_d2_selection_config(CONFIG)

        self.assertEqual(config.seeds, SEEDS)
        self.assertEqual(config.shot_counts, SHOT_COUNTS)
        self.assertEqual(config.validation_per_class, 10)
        self.assertEqual(config.source_records_per_class, 100)
        self.assertEqual(config.source_split, "finetune")
        self.assertEqual(config.test_split, "test")
        self.assertEqual(config.train_prefix_length, 20)
        self.assertEqual(config.validation_slice, (20, 30))
        self.assertEqual(config.conditions, CONDITIONS)
        self.assertEqual(CONFIG.stat().st_size, 858)
        self.assertEqual(sha256_file(CONFIG), CONFIG_SHA256)

    def test_retained_selection_is_balanced_nested_disjoint_and_paired(self):
        before = {
            path.name: (path.stat().st_size, sha256_file(path))
            for path in DATASET.iterdir()
            if path.is_file()
        }

        artifact = build_d2_few_shot_selection(CONFIG, DATASET)

        after = {
            path.name: (path.stat().st_size, sha256_file(path))
            for path in DATASET.iterdir()
            if path.is_file()
        }
        self.assertEqual(after, before)
        self.assertEqual(artifact["schema_version"], "phase05-d2-selection-artifact-v1")
        self.assertEqual(artifact["status"], "frozen")
        self.assertEqual(artifact["conditions"], list(CONDITIONS))
        self.assertEqual(artifact["seeds"], list(SEEDS))
        self.assertEqual(artifact["shot_counts"], list(SHOT_COUNTS))
        self.assertEqual(set(artifact["code"]), set(CODE_PATHS))
        for relative_path in CODE_PATHS:
            source_path = ROOT / relative_path
            self.assertEqual(
                artifact["code"][relative_path],
                {
                    "bytes": source_path.stat().st_size,
                    "sha256": sha256_file(source_path),
                },
            )
        self.assertEqual(
            artifact["dataset"]["split_counts"],
            {"finetune": 3000, "reference": 60000, "test": 3000},
        )
        self.assertEqual(artifact["test"], {
            "count": 3000,
            "record_ids_sha256": (
                "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"
            ),
            "source_split": "test",
        })
        self.assertEqual(len(artifact["selections"]), 5)
        for seed_document in artifact["selections"]:
            self.assertIn(seed_document["seed"], SEEDS)
            self.assertEqual(len(seed_document["classes"]), 30)
            all_train_ids = {shot_count: [] for shot_count in SHOT_COUNTS}
            all_validation_ids = []
            for class_document in seed_document["classes"]:
                class_label = class_document["class_label"]
                self.assertEqual(class_document["source_split"], "finetune")
                self.assertEqual(class_document["candidate_count"], 100)
                ordered = class_document["ordered_train_record_ids"]
                validation = class_document["validation_record_ids"]
                self.assertEqual(len(ordered), 20)
                self.assertEqual(len(validation), 10)
                self.assertEqual(len(set(ordered)), 20)
                self.assertEqual(len(set(validation)), 10)
                self.assertFalse(set(ordered) & set(validation))
                self.assertTrue(
                    all(record_id.startswith("finetune-") for record_id in ordered)
                )
                self.assertTrue(
                    all(
                        record_id.startswith("finetune-")
                        for record_id in validation
                    )
                )
                for shot_count in SHOT_COUNTS:
                    selected = class_document["train_record_ids"][str(shot_count)]
                    self.assertEqual(selected, ordered[:shot_count])
                    self.assertEqual(len(selected), shot_count)
                    all_train_ids[shot_count].extend(selected)
                all_validation_ids.extend(validation)
                self.assertEqual(
                    class_document["train_record_ids_sha256"],
                    {
                        str(shot_count): hashlib.sha256(
                            (
                                "\n".join(ordered[:shot_count])
                                + "\n"
                            ).encode("utf-8")
                        ).hexdigest()
                        for shot_count in SHOT_COUNTS
                    },
                )
                self.assertEqual(
                    class_document["validation_record_ids_sha256"],
                    hashlib.sha256(
                        ("\n".join(validation) + "\n").encode("utf-8")
                    ).hexdigest(),
                )
                self.assertEqual(class_document["class_label"], class_label)
            for shot_count in SHOT_COUNTS:
                selected = all_train_ids[shot_count]
                self.assertEqual(len(selected), shot_count * 30)
                self.assertEqual(len(set(selected)), shot_count * 30)
                self.assertEqual(
                    seed_document["train_record_ids_sha256"][str(shot_count)],
                    hashlib.sha256(
                        ("\n".join(selected) + "\n").encode("utf-8")
                    ).hexdigest(),
                )
            self.assertEqual(len(all_validation_ids), 300)
            self.assertEqual(len(set(all_validation_ids)), 300)
            self.assertFalse(set(all_train_ids[20]) & set(all_validation_ids))
            self.assertEqual(
                seed_document["validation_record_ids_sha256"],
                hashlib.sha256(
                    ("\n".join(all_validation_ids) + "\n").encode("utf-8")
                ).hexdigest(),
            )
        canonical_json_bytes(artifact)
        validate_d2_few_shot_selection(artifact, CONFIG, DATASET)

    def test_same_inputs_repeat_exact_artifact_and_seed_is_locally_isolated(self):
        first = build_d2_few_shot_selection(CONFIG, DATASET)
        second = build_d2_few_shot_selection(CONFIG, DATASET)

        self.assertEqual(first, second)
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        by_seed = {entry["seed"]: entry for entry in first["selections"]}
        self.assertNotEqual(
            by_seed[0]["train_record_ids_sha256"]["20"],
            by_seed[1]["train_record_ids_sha256"]["20"],
        )
        class_zero = {
            entry["seed"]: entry["classes"][0]
            for entry in first["selections"]
        }
        self.assertNotEqual(
            class_zero[0]["ordered_train_record_ids"],
            class_zero[1]["ordered_train_record_ids"],
        )

    def test_validator_rejects_non_nested_overlap_or_test_digest_mutation(self):
        artifact = build_d2_few_shot_selection(CONFIG, DATASET)
        mutations = []

        non_nested = json.loads(canonical_json_bytes(artifact))
        non_nested["selections"][0]["classes"][0]["train_record_ids"]["5"][0] = (
            non_nested["selections"][0]["classes"][0][
                "ordered_train_record_ids"
            ][6]
        )
        mutations.append(("nested", non_nested))

        overlap = json.loads(canonical_json_bytes(artifact))
        overlap["selections"][0]["classes"][0]["validation_record_ids"][0] = (
            overlap["selections"][0]["classes"][0]["ordered_train_record_ids"][0]
        )
        mutations.append(("overlap", overlap))

        bad_test = json.loads(canonical_json_bytes(artifact))
        bad_test["test"]["record_ids_sha256"] = "f" * 64
        mutations.append(("test", bad_test))

        for expected, mutated in mutations:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    D2SelectionValidationError,
                    expected,
                ):
                    validate_d2_few_shot_selection(
                        mutated,
                        CONFIG,
                        DATASET,
                    )

    def test_validator_rejects_valid_but_non_deterministic_selection(self):
        artifact = build_d2_few_shot_selection(CONFIG, DATASET)
        mutated = json.loads(canonical_json_bytes(artifact))
        seed_document = mutated["selections"][0]
        class_document = seed_document["classes"][0]
        selected = (
            class_document["ordered_train_record_ids"]
            + class_document["validation_record_ids"]
        )
        rotated = selected[1:] + selected[:1]
        ordered = rotated[:20]
        validation = rotated[20:30]
        class_document["ordered_train_record_ids"] = ordered
        class_document["validation_record_ids"] = validation
        class_document["train_record_ids"] = {
            str(shot_count): ordered[:shot_count]
            for shot_count in SHOT_COUNTS
        }
        class_document["train_record_ids_sha256"] = {
            str(shot_count): hashlib.sha256(
                (
                    "\n".join(ordered[:shot_count])
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
            for shot_count in SHOT_COUNTS
        }
        class_document["validation_record_ids_sha256"] = hashlib.sha256(
            ("\n".join(validation) + "\n").encode("utf-8")
        ).hexdigest()
        for shot_count in SHOT_COUNTS:
            all_ids = [
                record_id
                for item in seed_document["classes"]
                for record_id in item["train_record_ids"][str(shot_count)]
            ]
            seed_document["train_record_ids_sha256"][str(shot_count)] = (
                hashlib.sha256(
                    ("\n".join(all_ids) + "\n").encode("utf-8")
                ).hexdigest()
            )
        all_validation = [
            record_id
            for item in seed_document["classes"]
            for record_id in item["validation_record_ids"]
        ]
        seed_document["validation_record_ids_sha256"] = hashlib.sha256(
            ("\n".join(all_validation) + "\n").encode("utf-8")
        ).hexdigest()

        with self.assertRaisesRegex(
            D2SelectionValidationError,
            "deterministic",
        ):
            validate_d2_few_shot_selection(
                mutated,
                CONFIG,
                DATASET,
            )

    def test_cli_emits_one_canonical_line_and_does_not_write_files(self):
        before = {
            path.name: (path.stat().st_size, sha256_file(path))
            for path in DATASET.iterdir()
            if path.is_file()
        }
        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "freeze_d2_selection.py"),
                "--config",
                str(CONFIG),
                "--dataset",
                str(DATASET),
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
        artifact = json.loads(completed.stdout)
        self.assertEqual(
            completed.stdout,
            canonical_json_bytes(artifact),
        )
        validate_d2_few_shot_selection(artifact, CONFIG, DATASET)
        after = {
            path.name: (path.stat().st_size, sha256_file(path))
            for path in DATASET.iterdir()
            if path.is_file()
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
