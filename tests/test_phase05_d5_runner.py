from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.rruff import (  # noqa: E402
    CONFIG_SHA256,
    D5LibraryQuerySplit,
    D5RawCohort,
    load_d5_raw_cohort,
)
from rpe.runner.d5_rruff import (  # noqa: E402
    D5RunnerValidationError,
    paired_outcomes_jsonl_bytes,
    run_d5_provisional_from_cohort,
    validate_d5_provisional_result,
)


SG_CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d1_sg11_poly3_interp.json"
)
D5_CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d5_rruff_protocol.json"
)
DATASET = ROOT / "data" / "unified" / "rruff_raman_raw"
CONTROL = "raw_aligned_control"
SG = "raw_aligned_plus_sg11"


def read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


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


def tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def synthetic_cohort() -> D5RawCohort:
    axis = np.arange(200.0, 1800.0 + 1.0, 2.0, dtype="<f4")
    coordinate = np.arange(axis.size, dtype=np.float32)
    rows = []
    for role in ("query", "library"):
        for class_label in range(6):
            center = np.float32(80 + class_label * 110)
            width = np.float32(8 + class_label)
            signal = np.exp(
                -np.float32(0.5)
                * ((coordinate - center) / width) ** np.float32(2.0)
            )
            secondary = np.float32(0.35) * np.exp(
                -np.float32(0.5)
                * (
                    (
                        coordinate
                        - np.float32(680 - class_label * 70)
                    )
                    / np.float32(12)
                )
                ** np.float32(2.0)
            )
            perturbation = (
                np.float32(0.04)
                * np.sin(
                    np.float32(class_label + 3) * coordinate
                    / np.float32(23)
                )
                if role == "query"
                else np.float32(0.0)
            )
            rows.append(
                np.asarray(
                    np.float32(1.0) + signal + secondary + perturbation,
                    dtype="<f4",
                )
            )
    intensity = read_only(np.stack(rows).astype("<f4"))
    class_labels = read_only(
        np.array(list(range(6)) + list(range(6)), dtype="<i8")
    )
    query_indices = read_only(np.arange(0, 6, dtype="<i8"))
    library_indices = read_only(np.arange(6, 12, dtype="<i8"))
    split = D5LibraryQuerySplit(
        seed=0,
        query_indices=query_indices,
        library_indices=library_indices,
        split_sha256="a" * 64,
    )
    return D5RawCohort(
        protocol_config_sha256=CONFIG_SHA256,
        dataset_id="rruff_raman_raw",
        intensity=intensity,
        wavenumber=read_only(axis),
        class_labels=class_labels,
        record_ids=tuple(
            f"{role}-{class_label}"
            for role in ("query", "library")
            for class_label in range(6)
        ),
        mineral_names=tuple(
            f"mineral-{class_label}"
            for _ in range(2)
            for class_label in range(6)
        ),
        rruff_ids=tuple(
            f"R-{role}-{class_label}"
            for role in ("query", "library")
            for class_label in range(6)
        ),
        pin_ids=(None,) * 12,
        group_ids=tuple(
            f"group-{role}-{class_label}"
            for role in ("query", "library")
            for class_label in range(6)
        ),
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


def refresh_paired_outcomes_artifact(result: dict[str, object]) -> None:
    payload = b"".join(
        canonical_json_bytes(row)
        for row in result["paired_outcomes"]
    )
    result["paired_outcomes_artifact"] = {
        "bytes": len(payload),
        "lines": len(result["paired_outcomes"]),
        "path": f"paired_outcomes_seed{result['seed']}.jsonl",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class D5ProvisionalRunnerTest(unittest.TestCase):
    def setUp(self):
        self.cohort = synthetic_cohort()

    def test_smoke_builds_paired_outcomes_metrics_and_artifact_identity(self):
        result = run_d5_provisional_from_cohort(
            self.cohort,
            SG_CONFIG,
            seed=0,
            result_level="smoke",
        )

        self.assertEqual(result["schema_version"], "phase05-d5-result-v1")
        self.assertEqual(
            result["experiment_id"],
            "d5_rruff_raw_library_matching_sg11",
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_level"], "smoke")
        self.assertEqual(result["result_label"], "SMOKE — PROVIDED COHORT")
        self.assertEqual(result["seed"], 0)
        self.assertEqual(result["seeds_completed"], 1)
        self.assertEqual(result["seeds_required"], 5)
        self.assertEqual(result["protocol_config"]["sha256"], CONFIG_SHA256)
        self.assertEqual(result["split"]["query_count"], 6)
        self.assertEqual(result["split"]["library_count"], 6)
        self.assertEqual(result["split"]["class_count"], 6)
        self.assertEqual(
            [item["condition_id"] for item in result["conditions"]],
            [CONTROL, SG],
        )
        self.assertEqual(len(result["paired_outcomes"]), 6)
        for index, outcome in enumerate(result["paired_outcomes"]):
            self.assertEqual(outcome["query_index"], index)
            self.assertEqual(outcome["record_id"], f"query-{index}")
            self.assertEqual(outcome["true_class_label"], index)
            self.assertEqual(outcome["mineral_name"], f"mineral-{index}")
            self.assertEqual(outcome["rruff_id"], f"R-query-{index}")
            self.assertEqual(outcome["group_id"], f"group-query-{index}")
            for condition_id in (CONTROL, SG):
                condition = outcome["conditions"][condition_id]
                self.assertEqual(len(condition["top5_class_labels"]), 5)
                self.assertEqual(
                    condition["top1_class_label"],
                    condition["top5_class_labels"][0],
                )
                self.assertEqual(
                    condition["top1_correct"],
                    condition["top1_class_label"] == index,
                )
                self.assertEqual(
                    condition["top5_correct"],
                    index in condition["top5_class_labels"],
                )
                self.assertGreaterEqual(condition["top1_score"], -1.0)
                self.assertLessEqual(condition["top1_score"], 1.0)

        for condition in result["conditions"]:
            condition_id = condition["condition_id"]
            outcomes = [
                row["conditions"][condition_id]
                for row in result["paired_outcomes"]
            ]
            top1 = np.array(
                [row["top1_correct"] for row in outcomes],
                dtype=np.float64,
            )
            top5 = np.array(
                [row["top5_correct"] for row in outcomes],
                dtype=np.float64,
            )
            self.assertEqual(
                condition["metrics"]["top1_correct_count"],
                int(top1.sum()),
            )
            self.assertEqual(
                condition["metrics"]["top5_correct_count"],
                int(top5.sum()),
            )
            self.assertEqual(
                condition["metrics"]["top1_micro_accuracy"],
                float(top1.mean()),
            )
            self.assertEqual(
                condition["metrics"]["top5_micro_accuracy"],
                float(top5.mean()),
            )
            self.assertEqual(
                condition["metrics"]["top1_macro_class_accuracy"],
                float(top1.mean()),
            )
            self.assertEqual(
                condition["metrics"]["top5_macro_class_accuracy"],
                float(top5.mean()),
            )

        by_condition = {
            item["condition_id"]: item["metrics"]
            for item in result["conditions"]
        }
        for metric in (
            "top1_macro_class_accuracy",
            "top5_macro_class_accuracy",
            "top1_micro_accuracy",
            "top5_micro_accuracy",
        ):
            expected = 100.0 * (
                by_condition[SG][metric]
                - by_condition[CONTROL][metric]
            )
            self.assertEqual(
                result["descriptive_deltas_percentage_points"][metric],
                expected,
            )

        jsonl = paired_outcomes_jsonl_bytes(result)
        self.assertEqual(jsonl.count(b"\n"), 6)
        self.assertEqual(
            result["paired_outcomes_artifact"],
            {
                "bytes": len(jsonl),
                "lines": 6,
                "path": "paired_outcomes_seed0.jsonl",
                "sha256": hashlib.sha256(jsonl).hexdigest(),
            },
        )
        for line, expected in zip(
            jsonl.splitlines(keepends=True),
            result["paired_outcomes"],
            strict=True,
        ):
            self.assertEqual(line, canonical_json_bytes(expected))

        forbidden = {
            "bootstrap",
            "confidence_interval",
            "p_value",
            "permutation",
            "significant",
            "gate_success",
        }
        self.assertFalse(recursive_keys(result) & forbidden)
        validate_d5_provisional_result(result)
        canonical_json_bytes(result)

    def test_same_input_repeats_scientific_document(self):
        first = run_d5_provisional_from_cohort(
            self.cohort,
            SG_CONFIG,
            seed=0,
            result_level="smoke",
        )
        second = run_d5_provisional_from_cohort(
            self.cohort,
            SG_CONFIG,
            seed=0,
            result_level="smoke",
        )

        def scientific(result):
            return {
                key: value
                for key, value in result.items()
                if key not in {"runtime_seconds", "environment"}
            }

        self.assertEqual(scientific(first), scientific(second))
        self.assertEqual(
            paired_outcomes_jsonl_bytes(first),
            paired_outcomes_jsonl_bytes(second),
        )

    def test_provisional_rejects_seed_one_and_smoke_rejects_missing_seed(self):
        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "provisional seed",
        ):
            run_d5_provisional_from_cohort(
                self.cohort,
                SG_CONFIG,
                seed=1,
                result_level="provisional",
            )
        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "seed",
        ):
            run_d5_provisional_from_cohort(
                self.cohort,
                SG_CONFIG,
                seed=1,
                result_level="smoke",
            )

    def test_complete_seed_accepts_all_frozen_seeds_without_inference_fields(self):
        cohort = load_d5_raw_cohort(D5_CONFIG, DATASET)

        result = run_d5_provisional_from_cohort(
            cohort,
            SG_CONFIG,
            seed=1,
            result_level="complete_seed",
        )

        self.assertEqual(result["result_level"], "complete_seed")
        self.assertEqual(result["result_label"], "COMPLETE SEED — 2/5")
        self.assertEqual(result["seed"], 1)
        self.assertEqual(result["seeds_completed"], 2)
        self.assertEqual(result["seeds_required"], 5)
        self.assertFalse(result["claim_boundary"]["five_seed_cell"])
        self.assertFalse(result["claim_boundary"]["formal_inference"])
        forbidden = {
            "bootstrap",
            "confidence_interval",
            "p_value",
            "permutation",
            "significant",
            "gate_success",
        }
        self.assertFalse(recursive_keys(result) & forbidden)
        validate_d5_provisional_result(result)

    def test_validator_rejects_metric_or_paired_outcome_drift(self):
        result = run_d5_provisional_from_cohort(
            self.cohort,
            SG_CONFIG,
            seed=0,
            result_level="smoke",
        )
        metric_drift = copy.deepcopy(result)
        metric_drift["conditions"][0]["metrics"][
            "top1_micro_accuracy"
        ] = 0.123
        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "metrics",
        ):
            validate_d5_provisional_result(metric_drift)

        outcome_drift = copy.deepcopy(result)
        outcome_drift["paired_outcomes"][0]["conditions"][CONTROL][
            "top1_correct"
        ] = not outcome_drift["paired_outcomes"][0]["conditions"][CONTROL][
            "top1_correct"
        ]
        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "paired outcomes",
        ):
            validate_d5_provisional_result(outcome_drift)

    def test_validator_rejects_rank_metadata_or_result_label_drift(self):
        result = run_d5_provisional_from_cohort(
            self.cohort,
            SG_CONFIG,
            seed=0,
            result_level="smoke",
        )

        rank_drift = copy.deepcopy(result)
        condition = rank_drift["paired_outcomes"][0]["conditions"][CONTROL]
        condition["top5_class_labels"][1:3] = reversed(
            condition["top5_class_labels"][1:3]
        )
        condition["top5_scores"][1:3] = reversed(
            condition["top5_scores"][1:3]
        )
        refresh_paired_outcomes_artifact(rank_drift)
        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "ranked scores",
        ):
            validate_d5_provisional_result(rank_drift)

        metadata_drift = copy.deepcopy(result)
        metadata_drift["paired_outcomes"][0]["mineral_name"] = "wrong-mineral"
        refresh_paired_outcomes_artifact(metadata_drift)
        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "outcome metadata",
        ):
            validate_d5_provisional_result(metadata_drift)

        provisional = copy.deepcopy(result)
        provisional["result_level"] = "provisional"
        provisional["result_label"] = "SMOKE — PROVIDED COHORT"
        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "result label",
        ):
            validate_d5_provisional_result(provisional)

    def test_validator_requires_frozen_retained_dataset_and_seed0_split(self):
        result = run_d5_provisional_from_cohort(
            self.cohort,
            SG_CONFIG,
            seed=0,
            result_level="smoke",
        )
        provisional = copy.deepcopy(result)
        provisional["result_level"] = "provisional"
        provisional["result_label"] = "PROVISIONAL — 1/5 SEEDS"
        provisional["claim_boundary"]["retained_raw_comparison"] = True

        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "provisional dataset",
        ):
            validate_d5_provisional_result(provisional)

        provisional["dataset"] = {
            "class_count": 681,
            "class_labels_sha256": (
                "289906f3282e0e7f7a82ea61de413bc4c474b6158383bea3aaad21aa956fb94c"
            ),
            "dataset_id": "rruff_raman_raw",
            "group_count": 1934,
            "group_ids_sha256": (
                "886613d52869b87c96e101fe1dc2b7bf74008ea9deb5ba9a6cfa237edc9644a3"
            ),
            "matrix_sha256": (
                "5a9776e787609b84e0ffb2f82685a77f507748243e74769695300b221d4b52ee"
            ),
            "record_count": 3770,
            "record_ids_sha256": (
                "840896eee008cb5df60116ebc727431a649529c124d63a14fb0219f778f0e72d"
            ),
            "rruff_id_count": 1936,
        }
        with self.assertRaisesRegex(
            D5RunnerValidationError,
            "provisional split",
        ):
            validate_d5_provisional_result(provisional)

    def test_cli_rejects_nonzero_provisional_seed_without_writes(self):
        before = tree_snapshot(ROOT / "results" / "phase05")
        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "run_phase05_d5.py"),
                "--config",
                str(D5_CONFIG),
                "--dataset",
                str(DATASET),
                "--sg-config",
                str(SG_CONFIG),
                "--seed",
                "1",
                "--result-level",
                "provisional",
            ],
            cwd=ROOT,
            env=os.environ.copy(),
            check=False,
            capture_output=True,
            text=False,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout, b"")
        self.assertEqual(completed.stderr.count(b"\n"), 1)
        error = json.loads(completed.stderr)
        self.assertEqual(error["status"], "failed")
        self.assertEqual(
            completed.stderr,
            canonical_json_bytes(error),
        )
        self.assertIn("provisional seed", error["error"])
        self.assertEqual(
            tree_snapshot(ROOT / "results" / "phase05"),
            before,
        )


if __name__ == "__main__":
    unittest.main()
