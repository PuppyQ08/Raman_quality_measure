import hashlib
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.source_audit import (  # noqa: E402
    audit_file,
    validate_dataset_arrays,
    validate_record,
    write_jsonl,
)


class SourceAuditTest(unittest.TestCase):
    def test_verified_record_requires_hash_size_and_checks(self):
        with self.assertRaisesRegex(ValueError, "sha256"):
            validate_record(
                {
                    "source": "example",
                    "artifact_id": "missing-hash",
                    "status": "verified",
                    "local_path": "data/raw/example.bin",
                    "bytes": 1,
                    "checks": ["nonempty"],
                }
            )

    def test_blocked_record_requires_error_evidence(self):
        with self.assertRaisesRegex(ValueError, "error"):
            validate_record(
                {
                    "source": "example",
                    "artifact_id": "blocked",
                    "status": "blocked",
                }
            )

    def test_audit_file_hashes_and_crc_checks_zip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            archive = Path(temporary_directory) / "sample.zip"
            with zipfile.ZipFile(archive, "w") as zip_file:
                zip_file.writestr("spectrum.txt", "100 1.0\n101 2.0\n")

            record = audit_file("rruff", "sample", archive, "https://example.test/sample.zip")

            self.assertEqual(record["status"], "verified")
            self.assertEqual(record["bytes"], archive.stat().st_size)
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(archive.read_bytes()).hexdigest(),
            )
            self.assertIn("zip_crc_ok", record["checks"])
            self.assertEqual(record["format_details"]["member_count"], 1)

    def test_write_jsonl_rejects_invalid_verified_record(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "manifest.jsonl"
            with self.assertRaises(ValueError):
                write_jsonl(
                    output,
                    [
                        {
                            "source": "example",
                            "artifact_id": "invalid",
                            "status": "verified",
                        }
                    ],
                )
            self.assertFalse(output.exists())

    def test_write_jsonl_sorts_by_source_and_artifact(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "manifest.jsonl"
            records = [
                {
                    "source": "z",
                    "artifact_id": "b",
                    "status": "blocked",
                    "error": "credentials required",
                },
                {
                    "source": "a",
                    "artifact_id": "a",
                    "status": "failed",
                    "error": "download failed",
                },
            ]

            write_jsonl(output, records)

            parsed = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [(record["source"], record["artifact_id"]) for record in parsed],
                [("a", "a"), ("z", "b")],
            )

    def test_dataset_arrays_require_matching_sample_and_feature_dimensions(self):
        details = validate_dataset_arrays(
            spectra=np.ones((3, 4), dtype=np.float32),
            targets=np.array([0, 1, 0]),
            wavenumbers=np.arange(4, dtype=np.float32),
        )
        self.assertEqual(details["spectra_shape"], [3, 4])
        self.assertEqual(details["targets_shape"], [3])
        self.assertEqual(details["wavenumbers_shape"], [4])
        self.assertTrue(details["finite_spectra"])
        self.assertTrue(details["finite_wavenumbers"])

    def test_dataset_arrays_reject_nonfinite_spectra(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_dataset_arrays(
                spectra=np.array([[1.0, np.nan]]),
                targets=np.array([0]),
                wavenumbers=np.array([100.0, 101.0]),
            )

    def test_dataset_arrays_reject_mismatched_targets(self):
        with self.assertRaisesRegex(ValueError, "targets rows"):
            validate_dataset_arrays(
                spectra=np.ones((2, 3)),
                targets=np.array([0]),
                wavenumbers=np.arange(3),
            )

    def test_dataset_arrays_accept_numeric_object_wavenumbers(self):
        details = validate_dataset_arrays(
            spectra=np.ones((2, 3)),
            targets=np.array([0, 1]),
            wavenumbers=np.array([100.0, 101.0, 102.0], dtype=object),
        )
        self.assertEqual(details["wavenumbers_dtype"], "float64")

    def test_dataset_arrays_reject_nonnumeric_object_wavenumbers(self):
        with self.assertRaisesRegex(ValueError, "numeric"):
            validate_dataset_arrays(
                spectra=np.ones((1, 2)),
                targets=np.array([0]),
                wavenumbers=np.array([100.0, "invalid"], dtype=object),
            )


if __name__ == "__main__":
    unittest.main()
