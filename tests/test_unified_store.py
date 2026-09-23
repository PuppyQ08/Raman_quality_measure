import hashlib
import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rpe.io as rpe_io  # noqa: E402
from rpe.io.schema import SchemaValidationError  # noqa: E402
from rpe.io.store import (  # noqa: E402
    DatasetClosedError,
    DatasetSummary,
    DatasetValidationError,
    UnifiedDataset,
    ValidationSummary,
    validate_dataset,
    write_dataset,
)
from unified_helpers import (  # noqa: E402
    AXIS_DECREASING,
    AXIS_DECREASING_ID,
    AXIS_INCREASING,
    AXIS_INCREASING_ID,
    FIXTURE_DATASET_ID,
    SOURCE_A_SHA256,
    SOURCE_B_SHA256,
    corrected_record,
    fixture_records,
    raw_record,
    unknown_record,
)


EXPECTED_FILES = {
    "SHA256SUMS",
    "SHA256SUMS.sha256",
    "arrays.h5",
    "dataset.json",
    "records.jsonl",
}
CLASS_LABELS = {0: "class_a", 1: "class_b"}
CONCENTRATION_UNITS = {"acetate": "g/L", "glucose": "g/L"}


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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_checksums(path: Path) -> None:
    payload_names = ("arrays.h5", "dataset.json", "records.jsonl")
    checksum_path = path / "SHA256SUMS"
    checksum_path.write_text(
        "".join(
            f"{file_sha256(path / name)}  {name}\n"
            for name in payload_names
        ),
        encoding="utf-8",
    )
    (path / "SHA256SUMS.sha256").write_text(
        f"{file_sha256(checksum_path)}  SHA256SUMS\n",
        encoding="utf-8",
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(record) for record in records))


def assert_records_equal(
    testcase: unittest.TestCase,
    expected,
    actual,
) -> None:
    testcase.assertEqual(actual.record_id, expected.record_id)
    testcase.assertEqual(actual.meta, expected.meta)
    testcase.assertEqual(actual.provenance, expected.provenance)
    testcase.assertEqual(actual.targets.peaks, expected.targets.peaks)
    testcase.assertEqual(actual.targets.class_label, expected.targets.class_label)
    testcase.assertEqual(
        actual.targets.concentration,
        expected.targets.concentration,
    )
    testcase.assertEqual(
        actual.targets.concentrations,
        expected.targets.concentrations,
    )
    np.testing.assert_array_equal(actual.intensity, expected.intensity)
    np.testing.assert_array_equal(actual.wavenumber, expected.wavenumber)
    for field in ("clean", "baseline"):
        expected_array = getattr(expected.targets, field)
        actual_array = getattr(actual.targets, field)
        if expected_array is None:
            testcase.assertIsNone(actual_array)
        else:
            np.testing.assert_array_equal(actual_array, expected_array)


class UnifiedStoreWritingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.output_dir = self.temporary_root / FIXTURE_DATASET_ID

    def write_fixture(self):
        return write_dataset(
            fixture_records(),
            self.output_dir,
            dataset_id=FIXTURE_DATASET_ID,
            class_labels=CLASS_LABELS,
            concentration_unit="g/L",
            concentration_units=CONCENTRATION_UNITS,
        )

    def test_write_dataset_creates_exact_contract_and_summary(self):
        summary = self.write_fixture()

        self.assertEqual(summary.path, self.output_dir)
        self.assertEqual(summary.dataset_id, FIXTURE_DATASET_ID)
        self.assertEqual(summary.record_count, 3)
        self.assertEqual(summary.axis_group_count, 2)
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            EXPECTED_FILES,
        )
        self.assertFalse(
            any(".staging-" in path.name for path in self.temporary_root.iterdir())
        )

    def test_failed_first_publication_leaves_no_output_or_staging_directory(self):
        with patch(
            "rpe.io.store._write_hdf5",
            side_effect=OSError("injected HDF5 write failure"),
        ):
            with self.assertRaisesRegex(
                OSError,
                "injected HDF5 write failure",
            ):
                self.write_fixture()

        self.assertFalse(self.output_dir.exists())
        self.assertEqual(list(self.temporary_root.iterdir()), [])

    def test_dataset_manifest_is_canonical_and_complete(self):
        self.write_fixture()

        manifest_path = self.output_dir / "dataset.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(
            manifest,
            {
                "schema_version": "0.1.0",
                "dataset_id": FIXTURE_DATASET_ID,
                "record_count": 3,
                "axis_groups": [
                    {
                        "axis_id": AXIS_DECREASING_ID,
                        "length": 3,
                        "record_count": 1,
                    },
                    {
                        "axis_id": AXIS_INCREASING_ID,
                        "length": 4,
                        "record_count": 2,
                    },
                ],
                "class_labels": {"0": "class_a", "1": "class_b"},
                "concentration_unit": "g/L",
                "concentration_units": {
                    "acetate": "g/L",
                    "glucose": "g/L",
                },
                "source_artifacts": [
                    {
                        "source_url": "https://example.org/source",
                        "license": "CC0-1.0",
                        "license_status": "standardized",
                        "sha256": SOURCE_A_SHA256,
                        "retrieved_date": "2026-08-14",
                        "source_artifact": "source_a.txt",
                    },
                    {
                        "source_url": "https://example.org/source",
                        "license": "CC0-1.0",
                        "license_status": "standardized",
                        "sha256": SOURCE_B_SHA256,
                        "retrieved_date": "2026-08-14",
                        "source_artifact": "source_b.txt",
                    },
                ],
            },
        )
        self.assertEqual(manifest_path.read_bytes(), canonical_json_bytes(manifest))

    def test_records_jsonl_is_canonical_and_sorted_with_group_rows(self):
        self.write_fixture()

        records_path = self.output_dir / "records.jsonl"
        lines = records_path.read_bytes().splitlines(keepends=True)
        records = [json.loads(line) for line in lines]

        self.assertEqual(
            [record["record_id"] for record in records],
            [
                "record_a_raw",
                "record_b_corrected",
                "record_c_unknown",
            ],
        )
        self.assertEqual(
            [record["array_ref"] for record in records],
            [
                {"axis_id": AXIS_INCREASING_ID, "row": 0},
                {"axis_id": AXIS_INCREASING_ID, "row": 1},
                {"axis_id": AXIS_DECREASING_ID, "row": 0},
            ],
        )
        self.assertEqual(
            lines,
            [canonical_json_bytes(record) for record in records],
        )

    def test_hdf5_layout_preserves_native_axes_rows_and_optional_targets(self):
        self.write_fixture()

        with h5py.File(self.output_dir / "arrays.h5", "r") as arrays:
            self.assertEqual(dict(arrays.attrs), {
                "schema_version": "0.1.0",
                "dataset_id": FIXTURE_DATASET_ID,
            })
            self.assertEqual(set(arrays), {"axes"})
            self.assertEqual(
                set(arrays["axes"]),
                {AXIS_DECREASING_ID, AXIS_INCREASING_ID},
            )

            increasing = arrays["axes"][AXIS_INCREASING_ID]
            decreasing = arrays["axes"][AXIS_DECREASING_ID]
            self.assertEqual(dict(increasing.attrs), {"axis_id": AXIS_INCREASING_ID})
            self.assertEqual(dict(decreasing.attrs), {"axis_id": AXIS_DECREASING_ID})

            self.assertEqual(
                set(increasing),
                {
                    "baseline",
                    "baseline_present",
                    "clean",
                    "clean_present",
                    "intensity",
                    "wavenumber",
                },
            )
            self.assertEqual(set(decreasing), {"intensity", "wavenumber"})

            np.testing.assert_array_equal(
                increasing["wavenumber"][:],
                AXIS_INCREASING,
            )
            np.testing.assert_array_equal(
                decreasing["wavenumber"][:],
                AXIS_DECREASING,
            )
            np.testing.assert_array_equal(
                increasing["intensity"][:],
                np.stack(
                    [
                        raw_record().intensity,
                        corrected_record().intensity,
                    ]
                ),
            )
            np.testing.assert_array_equal(
                decreasing["intensity"][:],
                np.stack([unknown_record().intensity]),
            )
            np.testing.assert_array_equal(
                increasing["clean_present"][:],
                np.array([False, True], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                increasing["baseline_present"][:],
                np.array([False, True], dtype=np.bool_),
            )
            np.testing.assert_array_equal(
                increasing["clean"][:],
                np.stack(
                    [
                        np.zeros(4, dtype=np.float32),
                        corrected_record().targets.clean,
                    ]
                ),
            )
            np.testing.assert_array_equal(
                increasing["baseline"][:],
                np.stack(
                    [
                        np.zeros(4, dtype=np.float32),
                        corrected_record().targets.baseline,
                    ]
                ),
            )

            for axis_group in (increasing, decreasing):
                for dataset in axis_group.values():
                    self.assertEqual(dataset.compression, "gzip")
                    self.assertTrue(dataset.shuffle)
                    self.assertTrue(dataset.fletcher32)
                    if dataset.ndim == 1:
                        self.assertEqual(dataset.chunks, dataset.shape)
                    else:
                        self.assertEqual(
                            dataset.chunks,
                            (
                                min(dataset.shape[0], 256),
                                dataset.shape[1],
                            ),
                        )

            self.assertEqual(increasing["intensity"].dtype, np.dtype("float32"))
            self.assertEqual(decreasing["intensity"].dtype, np.dtype("float32"))
            self.assertEqual(increasing["clean_present"].dtype, np.dtype("bool"))

    def test_checksum_files_match_independent_hashes(self):
        self.write_fixture()

        checksum_path = self.output_dir / "SHA256SUMS"
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        parsed = []
        for line in lines:
            digest, relative_path = line.split("  ", 1)
            parsed.append((digest, relative_path))
        self.assertEqual(
            [relative_path for _, relative_path in parsed],
            ["arrays.h5", "dataset.json", "records.jsonl"],
        )
        for digest, relative_path in parsed:
            self.assertEqual(
                digest,
                file_sha256(self.output_dir / relative_path),
            )

        checksum_index = self.output_dir / "SHA256SUMS.sha256"
        self.assertEqual(
            checksum_index.read_text(encoding="utf-8"),
            f"{file_sha256(checksum_path)}  SHA256SUMS\n",
        )

    def test_validate_dataset_returns_basic_verified_summary(self):
        self.write_fixture()

        summary = validate_dataset(self.output_dir)

        self.assertEqual(summary.path, self.output_dir)
        self.assertEqual(summary.dataset_id, FIXTURE_DATASET_ID)
        self.assertEqual(summary.record_count, 3)
        self.assertEqual(summary.axis_group_count, 2)
        self.assertEqual(
            summary.target_presence_counts,
            {
                "baseline": 1,
                "class_label": 2,
                "clean": 1,
                "concentration": 1,
                "concentrations": 1,
                "peaks": 1,
            },
        )
        self.assertEqual(
            summary.preprocessing_status_counts,
            {
                "known_corrected": 1,
                "known_raw": 1,
                "unknown": 1,
            },
        )
        self.assertEqual(
            summary.checked_files,
            (
                "SHA256SUMS",
                "SHA256SUMS.sha256",
                "arrays.h5",
                "dataset.json",
                "records.jsonl",
            ),
        )

    def test_validate_dataset_detects_payload_checksum_mismatch(self):
        self.write_fixture()
        records_path = self.output_dir / "records.jsonl"
        records_path.write_bytes(records_path.read_bytes() + b"\n")

        with self.assertRaises(DatasetValidationError) as raised:
            validate_dataset(self.output_dir)

        self.assertEqual(raised.exception.path, "records.jsonl")

    def test_validate_dataset_rejects_extra_directory_entry(self):
        self.write_fixture()
        (self.output_dir / "unexpected").mkdir()

        with self.assertRaises(DatasetValidationError) as raised:
            validate_dataset(self.output_dir)

        self.assertEqual(raised.exception.path, "directory")

    def test_validate_dataset_no_checksums_skips_only_digest_comparison(self):
        self.write_fixture()
        checksum_path = self.output_dir / "SHA256SUMS"
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
        digest, name = lines[0].split("  ", 1)
        lines[0] = f"{'0' * len(digest)}  {name}"
        checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        checksum_index = self.output_dir / "SHA256SUMS.sha256"
        checksum_index.write_text(
            f"{'0' * 64}  SHA256SUMS\n",
            encoding="utf-8",
        )

        summary = validate_dataset(
            self.output_dir,
            verify_checksums=False,
        )

        self.assertEqual(summary.record_count, 3)

    def test_public_io_api_exports_task3_store_symbols(self):
        self.assertIs(rpe_io.DatasetSummary, DatasetSummary)
        self.assertIs(rpe_io.ValidationSummary, ValidationSummary)
        self.assertIs(rpe_io.DatasetValidationError, DatasetValidationError)
        self.assertIs(rpe_io.write_dataset, write_dataset)
        self.assertIs(rpe_io.validate_dataset, validate_dataset)


class UnifiedChecksumHardeningTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_dir = (
            Path(self.temporary_directory.name) / FIXTURE_DATASET_ID
        )
        write_dataset(
            fixture_records(),
            self.output_dir,
            dataset_id=FIXTURE_DATASET_ID,
            class_labels=CLASS_LABELS,
            concentration_unit="g/L",
            concentration_units=CONCENTRATION_UNITS,
        )

    def assert_validation_error(
        self,
        expected_path: str,
        mutation,
    ) -> None:
        mutation()
        with self.assertRaises(DatasetValidationError) as raised:
            validate_dataset(self.output_dir)
        self.assertEqual(raised.exception.path, expected_path)

    def test_validator_rejects_missing_payload_and_extra_entry(self):
        self.assert_validation_error(
            "directory",
            lambda: (self.output_dir / "arrays.h5").unlink(),
        )

    def test_validator_rejects_extra_sixth_file(self):
        self.assert_validation_error(
            "directory",
            lambda: (self.output_dir / "extra.txt").write_text(
                "extra",
                encoding="utf-8",
            ),
        )

    def test_checksum_parser_rejects_missing_and_duplicate_entries(self):
        checksum_path = self.output_dir / "SHA256SUMS"

        def missing_entry() -> None:
            lines = checksum_path.read_text(encoding="utf-8").splitlines()
            checksum_path.write_text(
                "\n".join(lines[:-1]) + "\n",
                encoding="utf-8",
            )

        self.assert_validation_error("SHA256SUMS", missing_entry)

    def test_checksum_parser_rejects_duplicate_entry(self):
        checksum_path = self.output_dir / "SHA256SUMS"

        def duplicate_entry() -> None:
            lines = checksum_path.read_text(encoding="utf-8").splitlines()
            checksum_path.write_text(
                "\n".join([*lines, lines[0]]) + "\n",
                encoding="utf-8",
            )

        self.assert_validation_error("SHA256SUMS", duplicate_entry)

    def test_checksum_parser_rejects_uppercase_hash(self):
        checksum_path = self.output_dir / "SHA256SUMS"

        def uppercase_hash() -> None:
            lines = checksum_path.read_text(encoding="utf-8").splitlines()
            digest, name = lines[0].split("  ", 1)
            lines[0] = f"{digest.upper()}  {name}"
            checksum_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

        self.assert_validation_error("SHA256SUMS", uppercase_hash)

    def test_checksum_parser_rejects_wrong_separator(self):
        checksum_path = self.output_dir / "SHA256SUMS"

        def one_space() -> None:
            lines = checksum_path.read_text(encoding="utf-8").splitlines()
            lines[0] = lines[0].replace("  ", " ", 1)
            checksum_path.write_text(
                "\n".join(lines) + "\n",
                encoding="utf-8",
            )

        self.assert_validation_error("SHA256SUMS", one_space)

    def test_checksum_parser_requires_exact_two_spaces_and_final_newline(self):
        checksum_path = self.output_dir / "SHA256SUMS"
        original = checksum_path.read_text(encoding="utf-8")

        for label, changed in (
            ("three spaces", original.replace("  ", "   ", 1)),
            ("missing final newline", original.rstrip("\n")),
            ("CRLF", original.replace("\n", "\r\n")),
        ):
            with self.subTest(label=label):
                checksum_path.write_bytes(changed.encode("utf-8"))
                with self.assertRaises(DatasetValidationError) as raised:
                    validate_dataset(self.output_dir)
                self.assertEqual(raised.exception.path, "SHA256SUMS")
                checksum_path.write_text(original, encoding="utf-8")

    def test_checksum_parser_rejects_absolute_and_traversal_paths(self):
        checksum_path = self.output_dir / "SHA256SUMS"

        for label, replacement in (
            ("absolute", "/tmp/arrays.h5"),
            ("traversal", "../arrays.h5"),
            ("windows", r"C:\tmp\arrays.h5"),
        ):
            with self.subTest(label=label):
                original = checksum_path.read_text(encoding="utf-8")
                lines = original.splitlines()
                digest, _ = lines[0].split("  ", 1)
                lines[0] = f"{digest}  {replacement}"
                checksum_path.write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaises(DatasetValidationError) as raised:
                    validate_dataset(self.output_dir)
                self.assertEqual(raised.exception.path, "SHA256SUMS")
                checksum_path.write_text(original, encoding="utf-8")

    def test_checksum_index_rejects_extra_line(self):
        index_path = self.output_dir / "SHA256SUMS.sha256"

        def extra_line() -> None:
            existing = index_path.read_text(encoding="utf-8")
            index_path.write_text(
                existing + existing,
                encoding="utf-8",
            )

        self.assert_validation_error("SHA256SUMS.sha256", extra_line)

    def test_checksum_index_rejects_wrong_digest(self):
        index_path = self.output_dir / "SHA256SUMS.sha256"

        def wrong_digest() -> None:
            index_path.write_text(
                f"{'0' * 64}  SHA256SUMS\n",
                encoding="utf-8",
            )

        self.assert_validation_error("SHA256SUMS", wrong_digest)


class UnifiedStoreOverwriteTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.parent = Path(self.temporary_directory.name)
        self.output_dir = self.parent / FIXTURE_DATASET_ID
        write_dataset(
            fixture_records(),
            self.output_dir,
            dataset_id=FIXTURE_DATASET_ID,
            class_labels=CLASS_LABELS,
            concentration_unit="g/L",
            concentration_units=CONCENTRATION_UNITS,
        )

    def file_hashes(self) -> dict[str, str]:
        return {
            path.name: file_sha256(path)
            for path in self.output_dir.iterdir()
            if path.is_file()
        }

    def overwrite(self):
        return write_dataset(
            fixture_records(),
            self.output_dir,
            dataset_id=FIXTURE_DATASET_ID,
            class_labels=CLASS_LABELS,
            concentration_unit="g/L",
            concentration_units=CONCENTRATION_UNITS,
            overwrite=True,
        )

    def assert_no_staging_artifacts(self) -> None:
        unexpected = [
            path.name
            for path in self.parent.iterdir()
            if path != self.output_dir
        ]
        self.assertEqual(unexpected, [])

    def test_overwrite_replaces_existing_dataset_and_keeps_valid_output(self):
        before = self.file_hashes()

        summary = self.overwrite()

        self.assertEqual(summary.record_count, 3)
        self.assertEqual(self.file_hashes(), before)
        self.assertEqual(validate_dataset(self.output_dir).record_count, 3)
        self.assert_no_staging_artifacts()

    def test_overwrite_publish_failure_restores_original_bytes(self):
        original_hashes = self.file_hashes()
        real_replace = os.replace

        def injected_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.name == FIXTURE_DATASET_ID
                and destination_path == self.output_dir
            ):
                raise OSError("injected publish failure")
            return real_replace(source, destination)

        with patch("rpe.io.store.os.replace", side_effect=injected_replace):
            with self.assertRaisesRegex(
                OSError,
                "injected publish failure",
            ):
                self.overwrite()

        self.assertTrue(self.output_dir.is_dir())
        self.assertEqual(self.file_hashes(), original_hashes)
        self.assertEqual(validate_dataset(self.output_dir).record_count, 3)
        self.assert_no_staging_artifacts()

    def test_overwrite_build_failure_preserves_original_without_backup(self):
        original_hashes = self.file_hashes()

        with patch(
            "rpe.io.store._write_hdf5",
            side_effect=OSError("injected build failure"),
        ):
            with self.assertRaisesRegex(
                OSError,
                "injected build failure",
            ):
                self.overwrite()

        self.assertEqual(self.file_hashes(), original_hashes)
        self.assertEqual(validate_dataset(self.output_dir).record_count, 3)
        self.assert_no_staging_artifacts()

    def test_overwrite_double_failure_preserves_backup_for_manual_recovery(self):
        original_hashes = self.file_hashes()
        real_replace = os.replace
        publish_failed = False

        def injected_replace(source, destination):
            nonlocal publish_failed
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.name == FIXTURE_DATASET_ID
                and destination_path == self.output_dir
            ):
                publish_failed = True
                raise OSError("injected publish failure")
            if (
                publish_failed
                and source_path.name == "backup"
                and destination_path == self.output_dir
            ):
                raise OSError("injected restore failure")
            return real_replace(source, destination)

        with patch("rpe.io.store.os.replace", side_effect=injected_replace):
            with self.assertRaisesRegex(
                OSError,
                "injected restore failure",
            ):
                self.overwrite()

        self.assertFalse(self.output_dir.exists())
        staging = [
            path
            for path in self.parent.iterdir()
            if path.name.startswith(f".{FIXTURE_DATASET_ID}.staging-")
        ]
        self.assertEqual(len(staging), 1)
        backup = staging[0] / "backup"
        self.assertTrue(backup.is_dir())
        backup_hashes = {
            path.name: file_sha256(path)
            for path in backup.iterdir()
            if path.is_file()
        }
        self.assertEqual(backup_hashes, original_hashes)
        self.assertEqual(
            {path.name for path in backup.iterdir()},
            EXPECTED_FILES,
        )


class UnifiedDatasetReaderTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.output_dir = (
            Path(self.temporary_directory.name) / FIXTURE_DATASET_ID
        )
        write_dataset(
            fixture_records(),
            self.output_dir,
            dataset_id=FIXTURE_DATASET_ID,
            class_labels=CLASS_LABELS,
            concentration_unit="g/L",
            concentration_units=CONCENTRATION_UNITS,
        )

    def test_reader_round_trip_order_lookup_and_indexing(self):
        expected_records = {
            record.record_id: record for record in fixture_records()
        }
        expected_ids = (
            "record_a_raw",
            "record_b_corrected",
            "record_c_unknown",
        )

        with UnifiedDataset.open(self.output_dir) as dataset:
            self.assertEqual(len(dataset), 3)
            self.assertEqual(dataset.record_ids, expected_ids)
            assert_records_equal(
                self,
                expected_records["record_b_corrected"],
                dataset.get("record_b_corrected"),
            )
            self.assertEqual(dataset[0].record_id, "record_a_raw")
            self.assertEqual(dataset[-1].record_id, "record_c_unknown")
            self.assertEqual(
                [record.record_id for record in dataset.iter_records()],
                list(expected_ids),
            )
            self.assertIsNone(dataset.get("record_a_raw").targets.clean)
            self.assertIsNone(dataset.get("record_c_unknown").targets.baseline)
            with self.assertRaises(KeyError):
                dataset.get("missing")
            with self.assertRaises(IndexError):
                _ = dataset[3]
            with self.assertRaises(IndexError):
                _ = dataset[-4]
            with self.assertRaises(TypeError):
                _ = dataset[True]
            with self.assertRaises(TypeError):
                _ = dataset["0"]

    def test_reader_returns_independent_array_copies(self):
        with UnifiedDataset.open(self.output_dir) as dataset:
            first = dataset.get("record_b_corrected")
            first.intensity[0] = -999.0
            first.wavenumber[0] = -999.0
            first.targets.clean[0] = -999.0
            first.targets.baseline[0] = -999.0

            second = dataset.get("record_b_corrected")

        np.testing.assert_array_equal(
            second.intensity,
            corrected_record().intensity,
        )
        np.testing.assert_array_equal(
            second.wavenumber,
            corrected_record().wavenumber,
        )
        np.testing.assert_array_equal(
            second.targets.clean,
            corrected_record().targets.clean,
        )
        np.testing.assert_array_equal(
            second.targets.baseline,
            corrected_record().targets.baseline,
        )

    def test_reader_close_is_idempotent_and_all_access_after_close_fails(self):
        dataset = UnifiedDataset.open(self.output_dir)
        dataset.close()
        dataset.close()

        accessors = (
            lambda: len(dataset),
            lambda: dataset.record_ids,
            lambda: dataset.get("record_a_raw"),
            lambda: dataset[0],
            lambda: list(dataset.iter_records()),
            lambda: dataset.__enter__(),
        )
        for access in accessors:
            with self.subTest(access=access):
                with self.assertRaises(DatasetClosedError):
                    access()

    def test_context_manager_closes_reader(self):
        dataset = UnifiedDataset.open(self.output_dir)

        with dataset as opened:
            self.assertIs(opened, dataset)
            self.assertEqual(opened.get("record_a_raw").record_id, "record_a_raw")

        with self.assertRaises(DatasetClosedError):
            dataset.get("record_a_raw")

    def test_open_is_lazy_at_record_array_boundary(self):
        with patch(
            "rpe.io.store._read_record_arrays",
            side_effect=AssertionError("record row read during open"),
        ):
            dataset = UnifiedDataset.open(self.output_dir)
            self.assertEqual(dataset.record_ids[0], "record_a_raw")
            dataset.close()

        with UnifiedDataset.open(self.output_dir) as dataset:
            self.assertEqual(dataset.get("record_a_raw").record_id, "record_a_raw")

    def test_open_does_not_read_intensity_or_optional_target_datasets(self):
        original_getitem = h5py.Dataset.__getitem__

        def reject_record_rows(dataset, key):
            dataset_name = dataset.name.rsplit("/", 1)[-1]
            if dataset_name in {"intensity", "clean", "baseline"}:
                raise AssertionError(
                    f"{dataset_name} read during UnifiedDataset.open"
                )
            return original_getitem(dataset, key)

        with patch.object(h5py.Dataset, "__getitem__", new=reject_record_rows):
            dataset = UnifiedDataset.open(self.output_dir)
            self.assertEqual(dataset.record_ids[0], "record_a_raw")
            dataset.close()

    def test_reader_verify_checksums_flag_matches_validator_semantics(self):
        checksum_path = self.output_dir / "SHA256SUMS"
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
        _, name = lines[0].split("  ", 1)
        lines[0] = f"{'0' * 64}  {name}"
        checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.output_dir / "SHA256SUMS.sha256").write_text(
            f"{'0' * 64}  SHA256SUMS\n",
            encoding="utf-8",
        )

        with self.assertRaises(DatasetValidationError):
            UnifiedDataset.open(self.output_dir)
        with UnifiedDataset.open(
            self.output_dir,
            verify_checksums=False,
        ) as dataset:
            self.assertEqual(dataset.record_ids[0], "record_a_raw")

    def test_public_io_api_exports_task4_reader_symbols(self):
        self.assertIs(rpe_io.UnifiedDataset, UnifiedDataset)
        self.assertIs(rpe_io.DatasetClosedError, DatasetClosedError)


class UnifiedDatasetStructuralValidationTest(unittest.TestCase):
    def assert_mutation_rejected(
        self,
        expected_path: str,
        mutation,
        *,
        exhaustive: bool = True,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / FIXTURE_DATASET_ID
            write_dataset(
                fixture_records(),
                output_dir,
                dataset_id=FIXTURE_DATASET_ID,
                class_labels=CLASS_LABELS,
                concentration_unit="g/L",
                concentration_units=CONCENTRATION_UNITS,
            )
            mutation(output_dir)
            rewrite_checksums(output_dir)

            operation = (
                lambda: validate_dataset(output_dir)
                if exhaustive
                else UnifiedDataset.open(output_dir)
            )
            with self.assertRaises(DatasetValidationError) as raised:
                result = operation()
                if not exhaustive:
                    result.close()
            self.assertEqual(raised.exception.path, expected_path)

    def test_validator_rejects_record_reference_corruption(self):
        def wrong_axis(path: Path) -> None:
            records_path = path / "records.jsonl"
            records = read_jsonl(records_path)
            records[0]["array_ref"]["axis_id"] = "0" * 64
            write_jsonl(records_path, records)

        def duplicate_row(path: Path) -> None:
            records_path = path / "records.jsonl"
            records = read_jsonl(records_path)
            records[1]["array_ref"]["row"] = 0
            write_jsonl(records_path, records)

        def out_of_range_row(path: Path) -> None:
            records_path = path / "records.jsonl"
            records = read_jsonl(records_path)
            records[1]["array_ref"]["row"] = 2
            write_jsonl(records_path, records)

        def mismatched_presence(path: Path) -> None:
            records_path = path / "records.jsonl"
            records = read_jsonl(records_path)
            records[1]["targets"]["clean_present"] = False
            write_jsonl(records_path, records)

        cases = (
            (
                "records.jsonl[0].array_ref.axis_id",
                wrong_axis,
            ),
            ("records.jsonl.array_ref.row", duplicate_row),
            ("records.jsonl.array_ref.row", out_of_range_row),
            (
                "records.jsonl[1].targets.clean_present",
                mismatched_presence,
            ),
        )
        for expected_path, mutation in cases:
            with self.subTest(expected_path=expected_path):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_validator_rejects_manifest_count_and_schema_corruption(self):
        def mutate_manifest(path: Path, mutation) -> None:
            manifest_path = path / "dataset.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutation(manifest)
            manifest_path.write_bytes(canonical_json_bytes(manifest))

        cases = (
            (
                "dataset.json.schema_version",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest.__setitem__(
                        "schema_version",
                        "9.9.9",
                    ),
                ),
            ),
            (
                "dataset.json.record_count",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest.__setitem__("record_count", 4),
                ),
            ),
            (
                "dataset.json.axis_groups",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest["axis_groups"][0].__setitem__(
                        "record_count",
                        99,
                    ),
                ),
            ),
            (
                "arrays.h5.axes."
                f"{AXIS_DECREASING_ID}.wavenumber",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest["axis_groups"][0].__setitem__(
                        "length",
                        4,
                    ),
                ),
            ),
            (
                "dataset.json",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest.__setitem__(
                        "unexpected",
                        True,
                    ),
                ),
            ),
        )
        for expected_path, mutation in cases:
            with self.subTest(expected_path=expected_path):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_validator_rejects_unsorted_and_duplicate_record_ids(self):
        def unsorted(path: Path) -> None:
            records_path = path / "records.jsonl"
            records = read_jsonl(records_path)
            records[0], records[1] = records[1], records[0]
            write_jsonl(records_path, records)

        def duplicate(path: Path) -> None:
            records_path = path / "records.jsonl"
            records = read_jsonl(records_path)
            records[1]["record_id"] = records[0]["record_id"]
            write_jsonl(records_path, records)

        for mutation in (unsorted, duplicate):
            with self.subTest(mutation=mutation):
                self.assert_mutation_rejected(
                    "records.jsonl.record_id",
                    mutation,
                )

    def test_validator_rejects_source_artifact_coverage_corruption(self):
        def mutate_manifest(path: Path, mutation) -> None:
            manifest_path = path / "dataset.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutation(manifest["source_artifacts"])
            manifest_path.write_bytes(canonical_json_bytes(manifest))

        def missing(path: Path) -> None:
            mutate_manifest(path, lambda artifacts: artifacts.pop())

        def unused(path: Path) -> None:
            def add(artifacts) -> None:
                extra = dict(artifacts[0])
                extra["sha256"] = "f" * 64
                extra["source_artifact"] = "unused.txt"
                artifacts.append(extra)

            mutate_manifest(path, add)

        for mutation in (missing, unused):
            with self.subTest(mutation=mutation):
                self.assert_mutation_rejected(
                    "dataset.json.source_artifacts",
                    mutation,
                )

    def test_validator_rejects_unsorted_source_artifacts(self):
        def reverse_artifacts(path: Path) -> None:
            manifest_path = path / "dataset.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_artifacts"].reverse()
            manifest_path.write_bytes(canonical_json_bytes(manifest))

        self.assert_mutation_rejected(
            "dataset.json.source_artifacts",
            reverse_artifacts,
        )

    def test_validator_uses_documented_source_artifact_sort_key(self):
        def noncanonical_sort(path: Path) -> None:
            manifest_path = path / "dataset.json"
            records_path = path / "records.jsonl"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            records = read_jsonl(records_path)

            manifest["source_artifacts"][0]["source_url"] = (
                "https://z.example/source"
            )
            manifest["source_artifacts"][0]["license"] = "A-license"
            manifest["source_artifacts"][1]["source_url"] = (
                "https://a.example/source"
            )
            manifest["source_artifacts"][1]["license"] = "Z-license"
            for record in records[:2]:
                record["provenance"]["source_url"] = "https://z.example/source"
                record["provenance"]["license"] = "A-license"
            records[2]["provenance"]["source_url"] = "https://a.example/source"
            records[2]["provenance"]["license"] = "Z-license"

            manifest_path.write_bytes(canonical_json_bytes(manifest))
            write_jsonl(records_path, records)

        self.assert_mutation_rejected(
            "dataset.json.source_artifacts",
            noncanonical_sort,
        )

    def test_validator_rejects_class_and_concentration_unit_coverage_corruption(
        self,
    ):
        def mutate_manifest(path: Path, mutation) -> None:
            manifest_path = path / "dataset.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            mutation(manifest)
            manifest_path.write_bytes(canonical_json_bytes(manifest))

        cases = (
            (
                "dataset.json.class_labels",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest["class_labels"].pop("1"),
                ),
            ),
            (
                "dataset.json.class_labels",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest["class_labels"].__setitem__(
                        "2",
                        "unused",
                    ),
                ),
            ),
            (
                "dataset.json.concentration_unit",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest.__setitem__(
                        "concentration_unit",
                        None,
                    ),
                ),
            ),
            (
                "dataset.json.concentration_units",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest[
                        "concentration_units"
                    ].pop("glucose"),
                ),
            ),
            (
                "dataset.json.concentration_units",
                lambda path: mutate_manifest(
                    path,
                    lambda manifest: manifest[
                        "concentration_units"
                    ].__setitem__("lactate", "g/L"),
                ),
            ),
        )
        for expected_path, mutation in cases:
            with self.subTest(expected_path=expected_path):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_validator_rejects_hdf5_layout_and_axis_corruption(self):
        def extra_root_attribute(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                arrays.attrs["unexpected"] = "value"

        def wrong_axis_data(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                axis = arrays["axes"][AXIS_INCREASING_ID]["wavenumber"]
                axis[0] = 101.0

        def mask_mismatch(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                arrays["axes"][AXIS_INCREASING_ID]["clean_present"][0] = True

        cases = (
            ("arrays.h5.attrs", extra_root_attribute),
            (
                f"arrays.h5.axes.{AXIS_INCREASING_ID}.wavenumber",
                wrong_axis_data,
            ),
            (
                "records.jsonl[0].targets.clean_present",
                mask_mismatch,
            ),
        )
        for expected_path, mutation in cases:
            with self.subTest(expected_path=expected_path):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_validator_rejects_nonzero_absent_optional_target_row(self):
        def nonzero_absent_row(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                arrays["axes"][AXIS_INCREASING_ID]["clean"][0, 0] = 1.0

        self.assert_mutation_rejected(
            f"arrays.h5.axes.{AXIS_INCREASING_ID}.clean[0]",
            nonzero_absent_row,
        )

    def test_validator_rejects_hdf5_dtype_and_filter_corruption(self):
        def replace_dataset(
            path: Path,
            *,
            dtype,
            compression,
        ) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                group = arrays["axes"][AXIS_INCREASING_ID]
                values = group["intensity"][:]
                del group["intensity"]
                group.create_dataset(
                    "intensity",
                    data=values.astype(dtype),
                    chunks=(2, 4),
                    compression=compression,
                    shuffle=True,
                    fletcher32=True,
                    track_times=False,
                )

        cases = (
            (
                "dtype",
                lambda path: replace_dataset(
                    path,
                    dtype=np.float64,
                    compression="gzip",
                ),
            ),
            (
                "filter",
                lambda path: replace_dataset(
                    path,
                    dtype=np.float32,
                    compression="lzf",
                ),
            ),
        )
        expected_path = f"arrays.h5.axes.{AXIS_INCREASING_ID}.intensity"
        for label, mutation in cases:
            with self.subTest(label=label):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_validator_rejects_hdf5_row_count_and_orphan_optional_dataset(self):
        def truncate_intensity(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                group = arrays["axes"][AXIS_INCREASING_ID]
                values = group["intensity"][:1]
                del group["intensity"]
                group.create_dataset(
                    "intensity",
                    data=values,
                    chunks=(1, 4),
                    compression="gzip",
                    shuffle=True,
                    fletcher32=True,
                    track_times=False,
                )

        def orphan_optional(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                group = arrays["axes"][AXIS_DECREASING_ID]
                group.create_dataset(
                    "clean",
                    data=np.zeros((1, 3), dtype=np.float32),
                    chunks=(1, 3),
                    compression="gzip",
                    shuffle=True,
                    fletcher32=True,
                    track_times=False,
                )

        cases = (
            (
                f"arrays.h5.axes.{AXIS_INCREASING_ID}.intensity",
                truncate_intensity,
            ),
            (
                f"arrays.h5.axes.{AXIS_DECREASING_ID}",
                orphan_optional,
            ),
        )
        for expected_path, mutation in cases:
            with self.subTest(expected_path=expected_path):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_validator_rejects_optional_pair_when_all_rows_are_absent(self):
        def all_absent_optional_pair(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                group = arrays["axes"][AXIS_DECREASING_ID]
                options = {
                    "compression": "gzip",
                    "shuffle": True,
                    "fletcher32": True,
                    "track_times": False,
                }
                group.create_dataset(
                    "clean",
                    data=np.zeros((1, 3), dtype=np.float32),
                    chunks=(1, 3),
                    **options,
                )
                group.create_dataset(
                    "clean_present",
                    data=np.array([False], dtype=np.bool_),
                    chunks=(1,),
                    **options,
                )

        self.assert_mutation_rejected(
            f"arrays.h5.axes.{AXIS_DECREASING_ID}.clean_present",
            all_absent_optional_pair,
        )

    def test_validator_rejects_wrong_hdf5_object_type_and_dataset_attributes(self):
        def intensity_as_group(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                axis_group = arrays["axes"][AXIS_INCREASING_ID]
                del axis_group["intensity"]
                axis_group.create_group("intensity")

        def extra_dataset_attribute(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                arrays["axes"][AXIS_INCREASING_ID]["intensity"].attrs[
                    "unexpected"
                ] = "value"

        expected_path = f"arrays.h5.axes.{AXIS_INCREASING_ID}.intensity"
        for label, mutation in (
            ("wrong object type", intensity_as_group),
            ("extra dataset attribute", extra_dataset_attribute),
        ):
            with self.subTest(label=label):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_validator_wraps_wrong_axes_and_axis_group_object_types(self):
        def axes_as_dataset(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                del arrays["axes"]
                arrays.create_dataset(
                    "axes",
                    data=np.array([1.0], dtype=np.float32),
                )

        def axis_as_dataset(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                axes = arrays["axes"]
                del axes[AXIS_INCREASING_ID]
                axes.create_dataset(
                    AXIS_INCREASING_ID,
                    data=np.array([1.0], dtype=np.float32),
                )

        cases = (
            ("arrays.h5.axes", axes_as_dataset),
            (
                f"arrays.h5.axes.{AXIS_INCREASING_ID}",
                axis_as_dataset,
            ),
        )
        for expected_path, mutation in cases:
            with self.subTest(expected_path=expected_path):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_validator_wraps_malformed_manifest_source_artifact(self):
        def malformed_artifact(path: Path) -> None:
            manifest_path = path / "dataset.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_artifacts"][0]["source_artifact"] = ["invalid"]
            manifest_path.write_bytes(canonical_json_bytes(manifest))

        self.assert_mutation_rejected(
            "dataset.json.source_artifacts[0]",
            malformed_artifact,
        )

    def test_validator_wraps_unhashable_manifest_artifact_fields(self):
        for field in ("license", "retrieved_date"):
            with self.subTest(field=field):
                def malformed_field(path: Path, current_field=field) -> None:
                    manifest_path = path / "dataset.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["source_artifacts"][0][current_field] = ["invalid"]
                    manifest_path.write_bytes(canonical_json_bytes(manifest))

                self.assert_mutation_rejected(
                    "dataset.json.source_artifacts[0]",
                    malformed_field,
                )

    def test_validator_wraps_unhashable_record_axis_id(self):
        def unhashable_axis_id(path: Path) -> None:
            records_path = path / "records.jsonl"
            records = read_jsonl(records_path)
            records[0]["array_ref"]["axis_id"] = ["invalid"]
            write_jsonl(records_path, records)

        self.assert_mutation_rejected(
            "records.jsonl[0].array_ref.axis_id",
            unhashable_axis_id,
        )

    def test_validator_rejects_non_scalar_core_hdf5_attributes(self):
        def root_attribute(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                arrays.attrs["dataset_id"] = np.array(
                    [FIXTURE_DATASET_ID],
                    dtype=h5py.string_dtype(),
                )

        def axis_attribute(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                arrays["axes"][AXIS_INCREASING_ID].attrs["axis_id"] = np.array(
                    [AXIS_INCREASING_ID],
                    dtype=h5py.string_dtype(),
                )

        for expected_path, mutation in (
            ("arrays.h5.attrs", root_attribute),
            (
                f"arrays.h5.axes.{AXIS_INCREASING_ID}.attrs",
                axis_attribute,
            ),
        ):
            with self.subTest(expected_path=expected_path):
                self.assert_mutation_rejected(expected_path, mutation)

    def test_exhaustive_validation_reads_arrays_but_open_remains_lazy(self):
        def nonfinite_intensity(path: Path) -> None:
            with h5py.File(path / "arrays.h5", "r+") as arrays:
                arrays["axes"][AXIS_INCREASING_ID]["intensity"][0, 0] = np.nan

        self.assert_mutation_rejected(
            "records.jsonl[0].intensity finite",
            nonfinite_intensity,
        )

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / FIXTURE_DATASET_ID
            write_dataset(
                fixture_records(),
                output_dir,
                dataset_id=FIXTURE_DATASET_ID,
                class_labels=CLASS_LABELS,
                concentration_unit="g/L",
                concentration_units=CONCENTRATION_UNITS,
            )
            nonfinite_intensity(output_dir)
            rewrite_checksums(output_dir)
            dataset = UnifiedDataset.open(output_dir)
            try:
                with self.assertRaises(SchemaValidationError):
                    dataset.get("record_a_raw")
            finally:
                dataset.close()


class UnifiedStoreArgumentValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.output_dir = self.temporary_root / FIXTURE_DATASET_ID

    def assert_write_error(
        self,
        expected_path: str,
        records,
        *,
        output_dir: Path | None = None,
        dataset_id: str = FIXTURE_DATASET_ID,
        class_labels=CLASS_LABELS,
        concentration_unit: str | None = "g/L",
        concentration_units=CONCENTRATION_UNITS,
    ):
        with self.assertRaises(DatasetValidationError) as raised:
            write_dataset(
                records,
                self.output_dir if output_dir is None else output_dir,
                dataset_id=dataset_id,
                class_labels=class_labels,
                concentration_unit=concentration_unit,
                concentration_units=concentration_units,
            )
        self.assertEqual(raised.exception.path, expected_path)

    def test_write_dataset_rejects_empty_duplicate_and_mixed_records(self):
        self.assert_write_error("records", [])

        duplicate = raw_record()
        self.assert_write_error(
            "records.record_id",
            [duplicate, duplicate],
            class_labels={0: "class_a"},
            concentration_unit=None,
            concentration_units={},
        )

        mixed = unknown_record()
        mixed = replace(
            mixed,
            meta=replace(mixed.meta, dataset_id="other_dataset"),
        )
        self.assert_write_error(
            "records.dataset_id",
            [raw_record(), mixed],
            concentration_unit=None,
        )

    def test_write_dataset_rejects_output_basename_mismatch(self):
        self.assert_write_error(
            "output_dir.name",
            fixture_records(),
            output_dir=self.temporary_root / "wrong_name",
        )

    def test_write_dataset_rejects_missing_and_unused_class_labels(self):
        self.assert_write_error(
            "class_labels missing",
            fixture_records(),
            class_labels={0: "class_a"},
        )
        self.assert_write_error(
            "class_labels unused",
            fixture_records(),
            class_labels={0: "class_a", 1: "class_b", 2: "class_c"},
        )
        self.assert_write_error(
            "class_labels missing",
            fixture_records(),
            class_labels=None,
        )

    def test_write_dataset_rejects_missing_scalar_concentration_unit(self):
        self.assert_write_error(
            "concentration_unit",
            fixture_records(),
            concentration_unit=None,
        )

    def test_write_dataset_rejects_missing_and_unused_named_units(self):
        self.assert_write_error(
            "concentration_units missing",
            fixture_records(),
            concentration_units={"glucose": "g/L"},
        )
        self.assert_write_error(
            "concentration_units unused",
            fixture_records(),
            concentration_units={
                "acetate": "g/L",
                "glucose": "g/L",
                "lactate": "g/L",
            },
        )

    def test_write_dataset_refuses_existing_output_without_overwrite(self):
        self.output_dir.mkdir()
        marker = self.output_dir / "marker.txt"
        marker.write_text("preserve me", encoding="utf-8")

        self.assert_write_error("output_dir", fixture_records())

        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve me")
        self.assertEqual(
            {path.name for path in self.output_dir.iterdir()},
            {"marker.txt"},
        )


if __name__ == "__main__":
    unittest.main()
