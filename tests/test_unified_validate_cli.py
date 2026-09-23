import hashlib
import importlib.util
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.io import DatasetValidationError, validate_dataset  # noqa: E402
from rpe.io import validate as validation_cli  # noqa: E402


BUILDER_PATH = ROOT / "tools" / "build_unified_fixture.py"
SOURCE_ROOT = ROOT / "tests" / "fixtures" / "unified"
DATASET_ID = "fixture_mixed_axes"
EXPECTED_FILES = {
    "SHA256SUMS",
    "SHA256SUMS.sha256",
    "arrays.h5",
    "dataset.json",
    "records.jsonl",
}
EXPECTED_SOURCE_HASHES = {
    "source_a.txt": (
        "2e21e44d1bc1c4f6631499e6b5396f4f701f62ee5743227936d2ba83567660e1"
    ),
    "source_b.txt": (
        "afe2261a4060a159bfbfebf1a1e0767f2d3366cb9c585c76b44a11bd5ee4a7e1"
    ),
}
EXPECTED_VALIDATION = {
    "axis_group_count": 2,
    "checked_files": [
        "SHA256SUMS",
        "SHA256SUMS.sha256",
        "arrays.h5",
        "dataset.json",
        "records.jsonl",
    ],
    "dataset_id": DATASET_ID,
    "preprocessing_status_counts": {
        "known_corrected": 1,
        "known_raw": 1,
        "unknown": 1,
    },
    "record_count": 3,
    "status": "verified",
    "target_presence_counts": {
        "baseline": 1,
        "class_label": 2,
        "clean": 1,
        "concentration": 1,
        "concentrations": 1,
        "peaks": 1,
    },
}


def load_builder_module():
    specification = importlib.util.spec_from_file_location(
        "task5_build_unified_fixture",
        BUILDER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load fixture builder")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "rpe.io.validate",
            *arguments,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def update_checksum_index(dataset: Path) -> None:
    checksum_path = dataset / "SHA256SUMS"
    digest = hashlib.sha256(checksum_path.read_bytes()).hexdigest()
    (dataset / "SHA256SUMS.sha256").write_text(
        f"{digest}  SHA256SUMS\n",
        encoding="utf-8",
    )


class UnifiedFixtureBuilderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output = Path(self.temporary_directory.name) / DATASET_ID

    def test_source_fixture_bytes_and_hashes_are_exact(self):
        for name, expected_hash in EXPECTED_SOURCE_HASHES.items():
            path = SOURCE_ROOT / name
            self.assertEqual(
                path.read_bytes(),
                f"fixture source {'A' if name == 'source_a.txt' else 'B'}\n".encode(),
            )
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_hash,
            )

    def test_builder_main_succeeds_with_network_connections_disabled(self):
        builder = load_builder_module()

        with patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("network connection attempted"),
        ):
            with patch("builtins.print") as print_output:
                result = builder.main(["--output", str(self.output)])

        self.assertEqual(result, 0)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            EXPECTED_FILES,
        )
        self.assertEqual(len(print_output.call_args_list), 1)
        summary = json.loads(print_output.call_args.args[0])
        self.assertEqual(
            summary,
            {
                "axis_group_count": 2,
                "dataset_id": DATASET_ID,
                "path": self.output.as_posix(),
                "record_count": 3,
                "status": "written",
            },
        )

    def test_builder_subprocess_emits_one_json_summary(self):
        result = subprocess.run(
            [
                sys.executable,
                str(BUILDER_PATH),
                "--output",
                str(self.output),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "axis_group_count": 2,
                "dataset_id": DATASET_ID,
                "path": self.output.as_posix(),
                "record_count": 3,
                "status": "written",
            },
        )
        self.assertEqual(
            validate_dataset(self.output).record_count,
            3,
        )

    def test_fixture_records_reject_tampered_source_hash(self):
        builder = load_builder_module()
        temporary_sources = Path(self.temporary_directory.name) / "sources"
        shutil.copytree(SOURCE_ROOT, temporary_sources)
        (temporary_sources / "source_a.txt").write_text(
            "tampered\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "source_a.txt: unexpected SHA256",
        ):
            builder.fixture_records(temporary_sources)


class UnifiedValidationCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output = Path(self.temporary_directory.name) / DATASET_ID
        builder = load_builder_module()
        with patch("builtins.print"):
            self.assertEqual(
                builder.main(["--output", str(self.output)]),
                0,
            )

    def test_cli_success_emits_exact_json_summary(self):
        result = run_cli(str(self.output))

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        self.assertEqual(json.loads(result.stdout), EXPECTED_VALIDATION)

    def test_cli_main_succeeds_with_network_connections_disabled(self):
        stdout = io.StringIO()
        stderr = io.StringIO()

        with patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("network connection attempted"),
        ):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                result = validation_cli.main([str(self.output)])

        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue()), EXPECTED_VALIDATION)

    def test_cli_main_does_not_catch_keyboard_interrupt(self):
        with patch.object(
            validation_cli,
            "validate_dataset",
            side_effect=KeyboardInterrupt,
        ):
            with self.assertRaises(KeyboardInterrupt):
                validation_cli.main([str(self.output)])

    def test_cli_payload_corruption_returns_one_json_error_without_traceback(self):
        records_path = self.output / "records.jsonl"
        records_path.write_bytes(records_path.read_bytes() + b"\n")

        result = run_cli(str(self.output))

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(len(result.stderr.splitlines()), 1)
        error = json.loads(result.stderr)
        self.assertEqual(error["status"], "failed")
        self.assertEqual(error["error_type"], "DatasetValidationError")
        self.assertIn("records.jsonl: checksum mismatch", error["error"])
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_no_checksums_still_rejects_structural_corruption(self):
        records_path = self.output / "records.jsonl"
        records_path.write_bytes(records_path.read_bytes() + b"\n")

        result = run_cli("--no-checksums", str(self.output))

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertEqual(len(result.stderr.splitlines()), 1)
        error = json.loads(result.stderr)
        self.assertEqual(error["status"], "failed")
        self.assertEqual(error["error_type"], "DatasetValidationError")
        self.assertNotIn("checksum mismatch", error["error"])
        self.assertIn("canonical JSON line", error["error"])
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_no_checksums_ignores_only_digest_values(self):
        checksum_path = self.output / "SHA256SUMS"
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
        _, name = lines[0].split("  ", 1)
        lines[0] = f"{'0' * 64}  {name}"
        checksum_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
        update_checksum_index(self.output)

        strict = run_cli(str(self.output))
        no_checksums = run_cli("--no-checksums", str(self.output))

        self.assertNotEqual(strict.returncode, 0)
        self.assertEqual(no_checksums.returncode, 0, no_checksums.stderr)
        self.assertEqual(json.loads(no_checksums.stdout), EXPECTED_VALIDATION)

    def test_cli_missing_dataset_returns_json_error(self):
        missing = Path(self.temporary_directory.name) / "missing"

        result = run_cli(str(missing))

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        error = json.loads(result.stderr)
        self.assertEqual(error["status"], "failed")
        self.assertEqual(error["error_type"], "DatasetValidationError")
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
