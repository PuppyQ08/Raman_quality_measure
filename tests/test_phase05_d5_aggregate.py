from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.d5_aggregate import (  # noqa: E402
    D5AggregateValidationError,
    aggregate_d5_results,
)
from rpe.runner.d5_rruff import FROZEN_SPLITS  # noqa: E402


PROTOCOL_SHA = (
    "94f59739a2e3583c6ae33ab5f4ab489cb1f26d3c9f8cdeb689a2448c9c01e009"
)
CONTROL = "raw_aligned_control"
SG = "raw_aligned_plus_sg11"
RUNNER_CODE_PATHS = (
    "rpe/downstream/rruff.py",
    "rpe/downstream/rruff_matching.py",
    "rpe/methods/classical/savitzky_golay.py",
    "rpe/runner/d5_rruff.py",
    "tools/run_phase05_d5.py",
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


def outcome(seed: int, index: int) -> dict[str, object]:
    class_label = index % 681
    control_top1 = class_label % 10 < 4
    sg_top1 = control_top1 or class_label % 100 == seed
    control_top5 = class_label % 10 < 7
    sg_top5 = control_top5 or class_label % 125 == seed

    def condition_document(
        *,
        top1_correct: bool,
        top5_correct: bool,
    ) -> dict[str, object]:
        labels = [class_label if top1_correct else 9999]
        if top5_correct and not top1_correct:
            labels.append(class_label)
        labels.extend(
            candidate
            for candidate in (10000, 10001, 10002, 10003, 10004)
            if candidate not in labels
        )
        labels = labels[:5]
        return {
            "top1_class_label": labels[0],
            "top1_correct": top1_correct,
            "top1_score": 0.9,
            "top5_class_labels": labels,
            "top5_correct": top5_correct,
            "top5_scores": [0.9, 0.8, 0.7, 0.6, 0.5],
        }

    return {
        "conditions": {
            CONTROL: condition_document(
                top1_correct=control_top1,
                top5_correct=control_top5,
            ),
            SG: condition_document(
                top1_correct=sg_top1,
                top5_correct=sg_top5,
            ),
        },
        "group_id": f"group-{seed}-{index}",
        "mineral_name": f"mineral-{class_label}",
        "query_index": index,
        "record_id": f"record-{seed}-{index}",
        "rruff_id": f"R-{seed}-{index}",
        "true_class_label": class_label,
    }


def seed_result(seed: int) -> dict[str, object]:
    outcomes = [
        outcome(seed, index)
        for index in range(FROZEN_SPLITS[seed]["query_count"])
    ]
    payload = b"".join(canonical_json_bytes(row) for row in outcomes)
    return {
        "claim_boundary": {
            "descriptive_only": True,
            "five_seed_cell": False,
            "formal_inference": False,
            "retained_raw_comparison": True,
        },
        "code": current_runner_code(),
        "conditions": [
            {
                "condition_id": condition_id,
                "condition_matrix_sha256": "b" * 64,
                "metrics": {"query_count": 681},
                "ranked_class_labels_sha256": "c" * 64,
                "ranked_class_scores_sha256": "d" * 64,
            }
            for condition_id in (CONTROL, SG)
        ],
        "dataset": {
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
        },
        "descriptive_deltas_percentage_points": {},
        "environment": {"python": "3.13"},
        "experiment_id": "d5_rruff_raw_library_matching_sg11",
        "paired_outcomes": outcomes,
        "paired_outcomes_artifact": {
            "bytes": len(payload),
            "lines": len(outcomes),
            "path": f"paired_outcomes_seed{seed}_complete.jsonl",
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "paired_summary": {},
        "protocol_config": {
            "bytes": 3855,
            "path": "d5_rruff_protocol.json",
            "sha256": PROTOCOL_SHA,
        },
        "result_label": f"COMPLETE SEED — {seed + 1}/5",
        "result_level": "complete_seed",
        "runtime_seconds": 1.0,
        "schema_version": "phase05-d5-result-v1",
        "seed": seed,
        "seeds_completed": seed + 1,
        "seeds_required": 5,
        "sg_config": {
            "bytes": 338,
            "path": "d1_sg11_poly3_interp.json",
            "sha256": (
                "775c3dba4c4d6bb13ab0ac44abf3e48011f90ce8bc268f7f79632ed35cb039aa"
            ),
        },
        "split": {
            **FROZEN_SPLITS[seed],
            "class_count": 681,
            "query_metadata_sha256": "6" * 64,
        },
        "status": "completed",
    }


class D5AggregateTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)
        for seed in range(5):
            result = seed_result(seed)
            (self.root / f"seed{seed}_complete.json").write_bytes(
                canonical_json_bytes(result)
            )
            (self.root / f"paired_outcomes_seed{seed}_complete.jsonl").write_bytes(
                b"".join(
                    canonical_json_bytes(row)
                    for row in result["paired_outcomes"]
                )
            )

    def test_aggregate_uses_681_class_values_and_frozen_inference(self):
        result = aggregate_d5_results(self.root)

        self.assertEqual(result["schema_version"], "phase05-d5-complete-cell-v1")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["result_level"], "complete_cell")
        self.assertEqual(result["seeds"], [0, 1, 2, 3, 4])
        self.assertEqual(result["class_count"], 681)
        self.assertEqual(result["statistics_unit"], "mineral_class_cluster")
        primary = result["primary_top1_macro_class_summary"]
        self.assertEqual(primary["bootstrap"]["sample_size"], 681)
        self.assertEqual(primary["bootstrap"]["resamples"], 10000)
        self.assertEqual(primary["permutation"]["sample_size"], 681)
        self.assertEqual(primary["permutation"]["resamples"], 100000)
        self.assertEqual(
            primary["permutation"]["p_value"],
            (
                primary["permutation"]["extreme_resamples"] + 1
            )
            / 100001,
        )
        self.assertEqual(len(result["per_class"]), 681)
        self.assertEqual(len(result["input_results"]), 5)
        self.assertEqual(len(result["paired_outcome_artifacts"]), 5)
        self.assertEqual(
            result["effect_exceeds_threshold"],
            primary["difference_mean_percentage_points"] > 2.0,
        )
        self.assertEqual(
            result["significant_at_configured_threshold"],
            primary["permutation"]["p_value"] < 0.05,
        )
        self.assertEqual(
            result["d5_gate_success"],
            result["effect_exceeds_threshold"]
            and result["significant_at_configured_threshold"],
        )
        canonical_json_bytes(result)

    def test_aggregate_rejects_missing_seed_or_payload_hash_drift(self):
        (self.root / "seed4_complete.json").unlink()
        with self.assertRaisesRegex(
            D5AggregateValidationError,
            "seed 4",
        ):
            aggregate_d5_results(self.root)

        result = seed_result(4)
        result["paired_outcomes_artifact"]["sha256"] = "f" * 64
        (self.root / "seed4_complete.json").write_bytes(
            canonical_json_bytes(result)
        )
        with self.assertRaisesRegex(
            D5AggregateValidationError,
            "paired outcome artifact",
        ):
            aggregate_d5_results(self.root)

    def test_aggregate_rejects_class_metadata_or_correctness_drift(self):
        result = seed_result(2)
        result["paired_outcomes"][0]["mineral_name"] = "wrong"
        payload = b"".join(
            canonical_json_bytes(row)
            for row in result["paired_outcomes"]
        )
        result["paired_outcomes_artifact"]["bytes"] = len(payload)
        result["paired_outcomes_artifact"]["sha256"] = hashlib.sha256(
            payload
        ).hexdigest()
        (self.root / "seed2_complete.json").write_bytes(
            canonical_json_bytes(result)
        )
        (
            self.root / "paired_outcomes_seed2_complete.jsonl"
        ).write_bytes(payload)
        with self.assertRaisesRegex(
            D5AggregateValidationError,
            "class metadata",
        ):
            aggregate_d5_results(self.root)

        result = seed_result(2)
        result["paired_outcomes"][0]["conditions"][CONTROL][
            "top1_correct"
        ] = not result["paired_outcomes"][0]["conditions"][CONTROL][
            "top1_correct"
        ]
        payload = b"".join(
            canonical_json_bytes(row)
            for row in result["paired_outcomes"]
        )
        result["paired_outcomes_artifact"]["bytes"] = len(payload)
        result["paired_outcomes_artifact"]["sha256"] = hashlib.sha256(
            payload
        ).hexdigest()
        (self.root / "seed2_complete.json").write_bytes(
            canonical_json_bytes(result)
        )
        (
            self.root / "paired_outcomes_seed2_complete.jsonl"
        ).write_bytes(payload)
        with self.assertRaisesRegex(
            D5AggregateValidationError,
            "correctness",
        ):
            aggregate_d5_results(self.root)

    def test_aggregate_rejects_consistent_but_stale_dataset_or_code(self):
        for seed in range(5):
            path = self.root / f"seed{seed}_complete.json"
            result = json.loads(path.read_bytes())
            result["dataset"]["matrix_sha256"] = "f" * 64
            path.write_bytes(canonical_json_bytes(result))
        with self.assertRaisesRegex(
            D5AggregateValidationError,
            "current dataset",
        ):
            aggregate_d5_results(self.root)

        for seed in range(5):
            result = seed_result(seed)
            result["code"] = {
                "rpe/runner/d5_rruff.py": {
                    "bytes": 1,
                    "sha256": "f" * 64,
                }
            }
            path = self.root / f"seed{seed}_complete.json"
            path.write_bytes(canonical_json_bytes(result))
        with self.assertRaisesRegex(
            D5AggregateValidationError,
            "current code",
        ):
            aggregate_d5_results(self.root)

    def test_aggregate_rejects_self_consistent_nonfrozen_split(self):
        result = seed_result(3)
        result["split"]["split_sha256"] = "f" * 64
        result["split"]["query_record_ids_sha256"] = "e" * 64
        (self.root / "seed3_complete.json").write_bytes(
            canonical_json_bytes(result)
        )

        with self.assertRaisesRegex(
            D5AggregateValidationError,
            "frozen split",
        ):
            aggregate_d5_results(self.root)

    def test_cli_emits_one_canonical_line_without_writes(self):
        before = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.root.iterdir()
            if path.is_file()
        }
        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "aggregate_phase05_d5.py"),
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
            if path.is_file()
        }
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
