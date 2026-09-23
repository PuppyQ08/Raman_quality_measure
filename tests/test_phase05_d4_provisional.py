from __future__ import annotations

import copy
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

from rpe.runner.d4_sugar import (  # noqa: E402
    D4RunnerValidationError,
    retained_sugar_comparison_for_level,
    paired_predictions_jsonl_bytes,
    run_d4_provisional_experiment,
    run_d4_result_from_cohort,
    validate_d4_result,
)
from tests.test_phase05_d4_runner import (  # noqa: E402
    CONFIG,
    synthetic_cohort,
)


ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "ramanbench"
    / "cache"
    / "10779223"
    / "Raw data.zip"
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


def tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


class D4ProvisionalContractTest(unittest.TestCase):
    def setUp(self):
        self.cohort = synthetic_cohort()

    def test_smoke_emits_canonical_paired_artifact_identity(self):
        result = run_d4_result_from_cohort(
            self.cohort,
            CONFIG,
            seed=0,
            result_level="smoke",
        )

        payload = paired_predictions_jsonl_bytes(result)

        self.assertEqual(payload.count(b"\n"), 24)
        self.assertEqual(
            result["paired_predictions_artifact"],
            {
                "bytes": len(payload),
                "lines": 24,
                "path": "paired_predictions_seed0.jsonl",
                "sha256": hashlib.sha256(payload).hexdigest(),
            },
        )
        for line, expected in zip(
            payload.splitlines(keepends=True),
            result["paired_predictions"],
            strict=True,
        ):
            self.assertEqual(line, canonical_json_bytes(expected))
        validate_d4_result(result)

    def test_complete_seed_is_a_retained_sugar_comparison(self):
        self.assertFalse(retained_sugar_comparison_for_level("smoke"))
        self.assertTrue(retained_sugar_comparison_for_level("provisional"))
        self.assertTrue(retained_sugar_comparison_for_level("complete_seed"))

    def test_synthetic_cohort_cannot_claim_retained_provisional(self):
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "provisional cohort",
        ):
            run_d4_result_from_cohort(
                self.cohort,
                CONFIG,
                seed=0,
                result_level="provisional",
            )

    def test_provisional_rejects_seed_one_before_archive_access(self):
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "provisional seed",
        ):
            run_d4_provisional_experiment(
                CONFIG,
                Path("/definitely/missing/Raw data.zip"),
                seed=1,
                result_level="provisional",
            )

    def test_complete_seed_requires_retained_cohort_and_seed_in_range(self):
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "provisional cohort",
        ):
            run_d4_result_from_cohort(
                self.cohort,
                CONFIG,
                seed=1,
                result_level="complete_seed",
            )

        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "complete seed",
        ):
            run_d4_provisional_experiment(
                CONFIG,
                Path("/definitely/missing/Raw data.zip"),
                seed=5,
                result_level="complete_seed",
            )

    def test_validator_rejects_artifact_or_claim_drift(self):
        result = run_d4_result_from_cohort(
            self.cohort,
            CONFIG,
            seed=0,
            result_level="smoke",
        )

        artifact_drift = copy.deepcopy(result)
        artifact_drift["paired_predictions_artifact"]["sha256"] = "f" * 64
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "paired predictions artifact",
        ):
            validate_d4_result(artifact_drift)

        provisional_drift = copy.deepcopy(result)
        provisional_drift["result_level"] = "provisional"
        provisional_drift["result_label"] = "PROVISIONAL — 1/5 SEEDS"
        provisional_drift["claim_boundary"]["retained_sugar_comparison"] = True
        with self.assertRaisesRegex(
            D4RunnerValidationError,
            "provisional cohort",
        ):
            validate_d4_result(provisional_drift)

    def test_cli_rejects_nonzero_provisional_seed_without_writes(self):
        results_root = ROOT / "results" / "phase05"
        before = tree_snapshot(results_root)
        completed = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "tools" / "run_phase05_d4.py"),
                "--config",
                str(CONFIG),
                "--archive",
                str(ARCHIVE),
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
        self.assertIn("provisional seed", error["error"])
        self.assertEqual(completed.stderr, canonical_json_bytes(error))
        self.assertEqual(tree_snapshot(results_root), before)

    def test_cli_reports_missing_archive_as_one_canonical_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            missing_archive = Path(temporary) / "Raw data.zip"
            completed = subprocess.run(
                [
                    str(ROOT / ".venv" / "bin" / "python"),
                    str(ROOT / "tools" / "run_phase05_d4.py"),
                    "--config",
                    str(CONFIG),
                    "--archive",
                    str(missing_archive),
                    "--seed",
                    "0",
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
        self.assertEqual(completed.stderr, canonical_json_bytes(error))


if __name__ == "__main__":
    unittest.main()
