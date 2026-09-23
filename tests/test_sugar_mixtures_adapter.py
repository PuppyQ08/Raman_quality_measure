from __future__ import annotations

import csv
import errno
import hashlib
import importlib
import io
import json
import os
import pickle
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import weakref
import zipfile
from dataclasses import FrozenInstanceError, fields, replace
from datetime import date
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import rpe.io.sugar_mixtures as sugar_mixtures_module
import rpe.io.sugar_mixtures_companions as sugar_companions_module
import rpe.io.sugar_mixtures_source as sugar_source_module
import rpe.io.sugar_mixtures_receipt as sugar_receipt_module


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from sugar_mixtures_helpers import (  # noqa: E402
    CLAIM_README,
    EXPERIMENTAL_README,
    HIGH_DIRECTORY,
    HIGH_SPECTRA_ORACLE,
    LOW_DIRECTORY,
    PREPARED_PREFIX,
    TARGET_MEMBER,
    SOURCE_METADATA_KEYS,
    SyntheticZipEntry,
    build_member_payload,
    corrupt_synthetic_zip_member_data,
    create_synthetic_sugar_source,
    file_sha256,
    read_synthetic_zip_entries,
    refresh_synthetic_sugar_contract,
    rewrite_synthetic_zip,
    set_synthetic_zip_member_flag_bits,
    source_json_text,
    synthetic_sugar_contract,
)
from rpe.io.sugar_mixtures_models import (  # noqa: E402
    SugarMixturesCompanionSummary,
    SugarMixturesDatasetSummary,
    SugarMixturesValidationError,
    _SugarFinalState,
    _SugarOutputRootLock,
    _SugarPublicationResult,
    _SugarRecoveryPlan,
    _SugarRecoveryResult,
    _SugarStagedData,
)
from rpe.io.sugar_mixtures_companions import (  # noqa: E402
    _validate_derived_endmembers,
    _validate_auxiliary_axes,
    _validate_acquisition_json,
    _validate_views,
    _write_derived_endmembers,
    _write_auxiliary_axes,
    _write_acquisition_json,
    _write_views,
)
from rpe.io.sugar_mixtures import (  # noqa: E402
    _build_sugar_mixtures_staged,
    _build_sugar_core_staged,
    _compare_sugar_dataset_to_source,
    _inspect_sugar_mixtures,
    inspect_sugar_mixtures,
    iter_sugar_mixtures_records,
)
from rpe.io.sugar_mixtures_receipt import (  # noqa: E402
    _build_receipt_document,
    _validate_receipt,
    _write_receipt,
)
from rpe.io.sugar_mixtures_source import (  # noqa: E402
    _canonical_json_bytes,
    _inspect_sugar_mixtures_source,
    _parse_canonical_member_bytes,
    _reparse_verified_member,
)
from rpe.io.schema import (  # noqa: E402
    LicenseStatus,
    PreprocessingStatus,
    RamanRecord,
    axis_id,
    validate_record,
)
from rpe.io.store import (  # noqa: E402
    DATASET_FILES,
    DatasetClosedError,
    DatasetValidationError,
    UnifiedDataset,
    validate_dataset,
)


SOURCE_MEMBER = (
    "Raw data/Experimental data from sugar mixtures/Raw data files/"
    "Sugar_Concentration_Test/"
    "Sugar_Concentration_Test_1_A1_1_RD1_M1_R1.csv"
)


def csv_rows(payload: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(payload.decode("utf-8"))))


def payload_from_rows(rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def metadata_document(**updates):
    document = json.loads(source_json_text())
    document.update(updates)
    return document


def contains_numpy_array(value, seen: set[int] | None = None) -> bool:
    if isinstance(value, np.ndarray):
        return True
    if isinstance(value, (str, bytes, Path, int, float, bool, type(None))):
        return False
    current_seen = set() if seen is None else seen
    identity = id(value)
    if identity in current_seen:
        return False
    current_seen.add(identity)
    if isinstance(value, dict) or hasattr(value, "items"):
        return any(
            contains_numpy_array(key, current_seen)
            or contains_numpy_array(item, current_seen)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return any(
            contains_numpy_array(item, current_seen)
            for item in value
        )
    if hasattr(value, "__dataclass_fields__"):
        return any(
            contains_numpy_array(item, current_seen)
            for item in vars(value).values()
        )
    return False


def replace_zip_entry(
    entries: tuple[SyntheticZipEntry, ...],
    member: str,
    *,
    name: str | None = None,
    payload: bytes | None = None,
    compress_type: int | None = None,
    external_attr: int | None = None,
) -> tuple[SyntheticZipEntry, ...]:
    result = []
    replaced = False
    for entry in entries:
        if entry.name != member:
            result.append(entry)
            continue
        result.append(
            replace(
                entry,
                name=entry.name if name is None else name,
                payload=entry.payload if payload is None else payload,
                compress_type=(
                    entry.compress_type
                    if compress_type is None
                    else compress_type
                ),
                external_attr=(
                    entry.external_attr
                    if external_attr is None
                    else external_attr
                ),
            )
        )
        replaced = True
    if not replaced:
        raise KeyError(member)
    return tuple(result)


def target_rows(payload: bytes) -> list[list[str]]:
    return list(csv.reader(io.StringIO(payload.decode("utf-8"))))


def canonical_json_bytes(
    value: object,
    *,
    newline: bool,
) -> bytes:
    def project(current):
        if hasattr(current, "items"):
            return {
                key: project(item)
                for key, item in current.items()
            }
        if isinstance(current, (tuple, list)):
            return [project(item) for item in current]
        return current

    encoded = json.dumps(
        project(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if newline:
        encoded += "\n"
    return encoded.encode("utf-8")


def update_length_prefixed(digest, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def corrupt_first_filtered_chunk(path: Path, dataset_path: str) -> None:
    with h5py.File(path, "r") as artifact:
        dataset = artifact[dataset_path]
        coordinates = tuple(0 for _ in dataset.shape)
        chunk = dataset.id.get_chunk_info_by_coord(coordinates)
        byte_offset = chunk.byte_offset
        chunk_size = chunk.size
    with Path(path).open("r+b") as artifact:
        artifact.seek(byte_offset + chunk_size // 2)
        original = artifact.read(1)
        if len(original) != 1:
            raise AssertionError("filtered chunk byte is unavailable")
        artifact.seek(-1, os.SEEK_CUR)
        artifact.write(bytes((original[0] ^ 0x01,)))


class SugarMixturesMemberParserTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.payload = build_member_payload()

    def parse(self, payload: bytes | None = None):
        return _parse_canonical_member_bytes(
            self.payload if payload is None else payload,
            source_member=SOURCE_MEMBER,
            expected_points=16,
        )

    def assert_invalid(
        self,
        expected_code: str,
        payload: bytes,
    ) -> SugarMixturesValidationError:
        with self.assertRaises(SugarMixturesValidationError) as captured:
            self.parse(payload)
        self.assertEqual(captured.exception.code, expected_code)
        self.assertNotEqual(captured.exception.path, "")
        self.assertNotEqual(captured.exception.reason, "")
        self.assertIn(expected_code, str(captured.exception))
        return captured.exception

    def test_synthetic_source_is_byte_deterministic(self):
        first = create_synthetic_sugar_source(self.temporary_root / "first")
        second = create_synthetic_sugar_source(self.temporary_root / "second")

        self.assertEqual(first.archive_bytes, second.archive_bytes)
        self.assertEqual(first.archive_md5, second.archive_md5)
        self.assertEqual(first.archive_sha256, second.archive_sha256)
        self.assertEqual(
            file_sha256(first.archive_path),
            file_sha256(second.archive_path),
        )
        self.assertEqual(first.archive_members, second.archive_members)
        self.assertEqual(
            dict(first.member_sha256),
            dict(second.member_sha256),
        )
        self.assertEqual(first.canonical_members, second.canonical_members)
        self.assertEqual(first.record_ids, second.record_ids)
        self.assertEqual(first.sample_ids, second.sample_ids)
        self.assertEqual(len(first.canonical_members), 21)
        self.assertEqual(len(first.record_ids), 21)
        self.assertEqual(len(first.sample_ids), 7)
        self.assertEqual(len(first.role_members["canonical_high"]), 7)
        self.assertEqual(len(first.role_members["canonical_low"]), 14)
        self.assertEqual(len(first.role_members["target"]), 1)
        self.assertEqual(len(first.role_members["oracle"]), 4)
        self.assertEqual(len(first.role_members["prepared"]), 22)
        self.assertEqual(len(first.role_members["raw_support"]), 18)
        self.assertEqual(len(first.role_members["experimental_readme"]), 1)
        self.assertEqual(len(first.role_members["excluded_synthetic"]), 1)
        self.assertEqual(
            first.archive_members,
            tuple(sorted(first.archive_members)),
        )
        with zipfile.ZipFile(first.archive_path) as archive:
            self.assertIsNone(archive.testzip())
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(info.create_system, 3)
                self.assertEqual(info.external_attr, 0o100644 << 16)

    def test_synthetic_contract_matches_exact_archive_and_evidence(self):
        source = create_synthetic_sugar_source(self.temporary_root / "source")
        contract = synthetic_sugar_contract(source)
        inventory = hashlib.sha256(b"rpe-sugar-zip-inventory-v1\0")
        with zipfile.ZipFile(source.archive_path) as archive:
            for info in sorted(
                archive.infolist(),
                key=lambda value: value.filename.encode("utf-8"),
            ):
                encoded_path = info.filename.encode("utf-8")
                inventory.update(struct.pack("<Q", len(encoded_path)))
                inventory.update(encoded_path)
                inventory.update(struct.pack("<B", int(info.is_dir())))
                inventory.update(
                    struct.pack(
                        "<IQQHHI",
                        info.CRC,
                        info.file_size,
                        info.compress_size,
                        info.compress_type,
                        info.flag_bits,
                        info.external_attr,
                    )
                )

        self.assertEqual(contract.archive_name, "Raw data.zip")
        self.assertEqual(contract.archive_bytes, source.archive_bytes)
        self.assertEqual(contract.archive_md5, source.archive_md5)
        self.assertEqual(contract.archive_sha256, source.archive_sha256)
        self.assertEqual(
            contract.archive_member_count,
            len(source.archive_members),
        )
        self.assertEqual(
            contract.archive_uncompressed_bytes,
            sum(source.member_bytes.values()),
        )
        self.assertEqual(
            contract.central_directory_inventory_sha256,
            inventory.hexdigest(),
        )
        self.assertEqual(
            dict(contract.expected_counts),
            {
                "canonical_records": 21,
                "high_records": 7,
                "low_records": 14,
                "high_records_per_well": 1,
                "low_records_per_well": 2,
                "high_round_count": 1,
                "low_round_count": 1,
                "high_repetitions_per_round": 1,
                "low_repetitions_per_round": 2,
                "sample_count": 7,
                "points_per_record": 16,
                "raw_text_inventory_sha256": (
                    "d5b2b47d5ec0785d09c449f56e7575935e3f1f5b8d710d9d1d419fdc2835e3da"
                ),
                "normalized_metadata_sha256": (
                    "1e3ab230a2dce7054712d143cdc7262b23b74d35df63e2c557ab60af6dbb47d2"
                ),
                "core_metadata_sha256": (
                    "7614044838509ce26cedf5d6de7b74b90d544458dc6e4f7c4ff553e0528839e2"
                ),
                "well_target_contract_sha256": (
                    "f4f612748ffa11a077ed9833ca78391ea8cef0a2df50f0fa3bdc0f927f318b98"
                ),
                "record_target_map_sha256": (
                    "0b9c60bb0fd6cd4e6d13d0f5e57daecad138626892ee001682c440a2deb66727"
                ),
                "record_preprocessing_map_sha256": (
                    "30e2d841cc7290fe5decab17302d57b49eccb32ef8bc03193341811179791e5c"
                ),
                "preprocessing_evidence_document_sha256": (
                    "af13cffa89ffe8f9041493914029a355857990c552d75a1c4ce22c75bd0f70f3"
                ),
                "intensity_float32_max_abs_error": 0.0,
                "axis_float32_max_abs_error_cm1": (
                    1.4110402389633236e-05
                ),
                "wavelength_float32_max_abs_error_nm": 0.0,
            },
        )
        self.assertEqual(
            set(contract.expected_evidence_snapshots),
            {
                "evidence/zenodo/10779223.json",
                "receipts/sugar_mixtures_high_snr.json",
                "receipts/sugar_mixtures_low_snr.json",
            },
        )

    def test_parser_preserves_arrays_and_exact_json_text(self):
        parsed = self.parse()

        self.assertEqual(parsed.source_member, SOURCE_MEMBER)
        self.assertEqual(len(parsed.source_member_sha256), 64)
        np.testing.assert_array_equal(
            parsed.pixel,
            np.arange(16, dtype=np.float64),
        )
        expected_wavelength = np.linspace(800.0, 815.0, 16)
        expected_wavenumber = (
            1.0e7 / 785.0 - 1.0e7 / expected_wavelength
        )
        np.testing.assert_allclose(
            parsed.wavelength_nm,
            expected_wavelength,
            rtol=0.0,
            atol=5.0e-13,
        )
        np.testing.assert_allclose(
            parsed.wavenumber_cm1,
            expected_wavenumber,
            rtol=0.0,
            atol=5.0e-13,
        )
        np.testing.assert_array_equal(
            parsed.intensity,
            np.arange(100, 116, dtype=np.float64),
        )
        self.assertEqual(parsed.source_json_text, source_json_text())
        self.assertEqual(
            tuple(parsed.source_metadata),
            SOURCE_METADATA_KEYS,
        )
        np.testing.assert_array_equal(
            np.isnan(parsed.source_metadata["Laser_Offset"]),
            np.array([True, True]),
        )

    def test_parser_outputs_are_read_only_and_metadata_is_immutable(self):
        parsed = self.parse()

        for array in (
            parsed.pixel,
            parsed.wavelength_nm,
            parsed.wavenumber_cm1,
            parsed.intensity,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array[0] = 1
        with self.assertRaises(TypeError):
            parsed.source_metadata["Date"] = "changed"
        with self.assertRaises(FrozenInstanceError):
            parsed.source_member = "changed"

    def test_canonical_json_bytes_are_finite_sorted_and_newline_terminated(self):
        encoded = _canonical_json_bytes({"β": 2, "a": 1})

        self.assertEqual(encoded, '{"a":1,"β":2}\n'.encode("utf-8"))
        self.assertEqual(
            json.loads(encoded),
            {"a": 1, "β": 2},
        )
        with self.assertRaises(ValueError):
            _canonical_json_bytes({"bad": float("nan")})

    def test_parser_rejects_wrong_header(self):
        rows = csv_rows(self.payload)
        rows[0] = ["Pixel", "cm-1", "wl", "Intensity", "Metadata"]
        self.assert_invalid("HEADER_INVALID", payload_from_rows(rows))

    def test_parser_rejects_invalid_utf8(self):
        self.assert_invalid("ENCODING_INVALID", b"\xff\xfe\xfd")

    def test_parser_rejects_malformed_csv(self):
        payload = (
            b"Pixel,wl,cm-1,Intensity,Metadata\n"
            b"0,800,100,1,\"unterminated\n"
        )
        self.assert_invalid("CSV_INVALID", payload)

    def test_parser_wraps_csv_reader_error(self):
        with patch(
            "rpe.io.sugar_mixtures_source.csv.reader",
            side_effect=csv.Error("synthetic CSV failure"),
        ):
            self.assert_invalid("CSV_INVALID", self.payload)

    def test_parser_rejects_missing_or_extra_column(self):
        for mutate in (
            lambda row: row[:-1],
            lambda row: [*row, "extra"],
        ):
            with self.subTest(mutate=mutate):
                rows = csv_rows(self.payload)
                rows[1] = mutate(rows[1])
                self.assert_invalid(
                    "ROW_WIDTH_INVALID",
                    payload_from_rows(rows),
                )

    def test_parser_rejects_numeric_parse_failure_and_empty_field(self):
        for value in ("not-a-number", ""):
            with self.subTest(value=value):
                rows = csv_rows(self.payload)
                rows[1][3] = value
                self.assert_invalid(
                    "NUMERIC_ROW_INVALID",
                    payload_from_rows(rows),
                )

    def test_parser_rejects_nonfinite_numeric_value(self):
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                rows = csv_rows(self.payload)
                rows[1][1] = value
                self.assert_invalid(
                    "NUMERIC_NONFINITE",
                    payload_from_rows(rows),
                )

    def test_parser_rejects_wrong_point_count(self):
        rows = csv_rows(self.payload)
        del rows[-2]
        self.assert_invalid("POINT_COUNT_MISMATCH", payload_from_rows(rows))

        rows = csv_rows(self.payload)
        rows.insert(-1, list(rows[-2]))
        rows[-2][0] = "16"
        rows[-2][1] = "816"
        rows[-2][2] = "500"
        rows[-2][3] = "116"
        self.assert_invalid("POINT_COUNT_MISMATCH", payload_from_rows(rows))

    def test_parser_rejects_noninteger_or_duplicate_pixel(self):
        rows = csv_rows(self.payload)
        rows[1][0] = "0.5"
        self.assert_invalid("PIXEL_NONINTEGER", payload_from_rows(rows))

        rows = csv_rows(self.payload)
        rows[2][0] = rows[1][0]
        self.assert_invalid("PIXEL_DUPLICATE", payload_from_rows(rows))

    def test_parser_rejects_nonmonotonic_physical_axis(self):
        for column in (1, 2):
            with self.subTest(column=column):
                rows = csv_rows(self.payload)
                rows[3][column] = rows[1][column]
                self.assert_invalid(
                    "AXIS_NONMONOTONIC",
                    payload_from_rows(rows),
                )

    def test_parser_rejects_missing_duplicate_or_nonfinal_metadata_row(self):
        rows = csv_rows(self.payload)
        del rows[-1]
        self.assert_invalid("METADATA_ROW_MISSING", payload_from_rows(rows))

        rows = csv_rows(self.payload)
        rows.append(list(rows[-1]))
        self.assert_invalid("METADATA_ROW_DUPLICATE", payload_from_rows(rows))

        payload = build_member_payload(metadata_first=True)
        self.assert_invalid("METADATA_ROW_POSITION", payload)

    def test_parser_rejects_metadata_row_with_numeric_fields(self):
        rows = csv_rows(self.payload)
        rows[-1][0] = "0"
        self.assert_invalid(
            "METADATA_ROW_POSITION",
            payload_from_rows(rows),
        )

    def test_parser_rejects_five_empty_numeric_fields(self):
        rows = csv_rows(self.payload)
        rows.insert(-1, ["", "", "", "", ""])
        self.assert_invalid(
            "NUMERIC_ROW_INVALID",
            payload_from_rows(rows),
        )

    def test_parser_rejects_malformed_metadata_json(self):
        payload = build_member_payload(metadata_text="{not-json")
        self.assert_invalid("METADATA_JSON_INVALID", payload)

    def test_parser_rejects_nonobject_metadata_json(self):
        self.assert_invalid(
            "METADATA_TYPE_INVALID",
            build_member_payload(metadata_text="[]"),
        )

    def test_parser_rejects_unexpected_nonfinite_metadata_constant(self):
        for value in ("Infinity", "-Infinity"):
            with self.subTest(value=value):
                text = source_json_text().replace(
                    '"Spectro_Temp [C]": -59',
                    f'"Spectro_Temp [C]": {value}',
                )
                self.assert_invalid(
                    "METADATA_NONFINITE_INVALID",
                    build_member_payload(metadata_text=text),
                )

    def test_parser_rejects_wrong_laser_offset_semantics(self):
        for offset in ([0.0, float("nan")], [float("nan")], "NaN"):
            with self.subTest(offset=offset):
                document = metadata_document(Laser_Offset=offset)
                text = json.dumps(document, allow_nan=True)
                self.assert_invalid(
                    "METADATA_NONFINITE_INVALID",
                    build_member_payload(metadata_text=text),
                )

    def test_parser_rejects_missing_extra_or_reordered_metadata_key(self):
        document = metadata_document()
        document.pop("Spot_Size")
        self.assert_invalid(
            "METADATA_KEYS_INVALID",
            build_member_payload(
                metadata_text=json.dumps(document, allow_nan=True),
            ),
        )

        document = metadata_document()
        document["Unexpected"] = 1
        self.assert_invalid(
            "METADATA_KEYS_INVALID",
            build_member_payload(
                metadata_text=json.dumps(document, allow_nan=True),
            ),
        )

        document = metadata_document()
        reordered = {
            key: document[key]
            for key in reversed(tuple(document))
        }
        self.assert_invalid(
            "METADATA_KEYS_INVALID",
            build_member_payload(
                metadata_text=json.dumps(reordered, allow_nan=True),
            ),
        )

    def test_parser_rejects_duplicate_metadata_key(self):
        text = source_json_text()
        duplicate = (
            '{"Date":"duplicate",'
            + text.removeprefix("{")
        )
        self.assert_invalid(
            "METADATA_KEYS_INVALID",
            build_member_payload(metadata_text=duplicate),
        )

    def test_parser_rejects_wrong_metadata_type(self):
        invalid_updates = (
            {"Date": 123},
            {"Position [um]": "not-a-position"},
            {"Temperature [C]": 20.0},
            {"Spectro_Temp [C]": "cold"},
            {"Spectrometer integration time [s]": "5"},
            {"Spectrometer number of accumulations": 1.5},
            {"Objective_Magnification": True},
        )
        for updates in invalid_updates:
            with self.subTest(updates=updates):
                document = metadata_document(**updates)
                self.assert_invalid(
                    "METADATA_TYPE_INVALID",
                    build_member_payload(
                        metadata_text=json.dumps(
                            document,
                            allow_nan=True,
                        ),
                    ),
                )

    def test_parser_enforces_source_integer_and_float_field_types(self):
        invalid_updates = (
            {"Spectro_Temp [C]": -59.0},
            {"Laser power [mW]": 36},
            {"Excitation wavelength [nm]": 785.0},
            {"Objective_Magnification": 20},
            {"Spectrometer temperature [C]": -60.0},
            {"Magnification": 20},
            {"Spot_Size": 2},
        )
        for updates in invalid_updates:
            with self.subTest(updates=updates):
                document = metadata_document(**updates)
                self.assert_invalid(
                    "METADATA_TYPE_INVALID",
                    build_member_payload(
                        metadata_text=json.dumps(
                            document,
                            allow_nan=True,
                        ),
                    ),
                )

    def test_parser_rejects_nonfinite_metadata_number_outside_laser_offset(self):
        document = metadata_document(**{"Laser power [mW]": float("nan")})
        self.assert_invalid(
            "METADATA_NONFINITE_INVALID",
            build_member_payload(
                metadata_text=json.dumps(document, allow_nan=True),
            ),
        )


class SugarMixturesArchiveInspectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "valid"
        )
        self.contract = synthetic_sugar_contract(self.source)

    def create_source(self, name: str):
        source = create_synthetic_sugar_source(self.temporary_root / name)
        return source, synthetic_sugar_contract(source)

    def inspect(self, source=None, contract=None):
        selected_source = self.source if source is None else source
        selected_contract = self.contract if contract is None else contract
        return _inspect_sugar_mixtures_source(
            selected_source.raw_root,
            selected_contract,
        )

    def assert_invalid(
        self,
        expected_code: str,
        source=None,
        contract=None,
    ) -> SugarMixturesValidationError:
        with self.assertRaises(SugarMixturesValidationError) as captured:
            self.inspect(source, contract)
        self.assertEqual(captured.exception.code, expected_code)
        self.assertNotEqual(captured.exception.path, "")
        self.assertNotEqual(captured.exception.reason, "")
        self.assertIn(expected_code, str(captured.exception))
        return captured.exception

    def mutate_target(
        self,
        source,
        mutate,
    ):
        def transform(entries):
            result = []
            for entry in entries:
                if entry.name != TARGET_MEMBER:
                    result.append(entry)
                    continue
                rows = target_rows(entry.payload)
                mutate(rows)
                result.append(
                    replace(entry, payload=payload_from_rows(rows))
                )
            return tuple(result)

        rewrite_synthetic_zip(source, transform)
        return refresh_synthetic_sugar_contract(
            source,
            refresh_roles=True,
        )

    def rename_member_contract(
        self,
        source,
        old_name: str,
        new_name: str,
    ):
        rewrite_synthetic_zip(
            source,
            lambda entries: replace_zip_entry(
                entries,
                old_name,
                name=new_name,
            ),
        )
        role_members = {
            role: tuple(
                new_name if name == old_name else name
                for name in names
            )
            for role, names in source.role_members.items()
        }
        return refresh_synthetic_sugar_contract(
            source,
            refresh_roles=True,
            role_members=role_members,
        )

    def test_inspection_verifies_archive_roles_recipes_views_and_axes(self):
        inspection = self.inspect()
        statistics = inspection.source_statistics

        self.assertEqual(inspection.raw_root, self.source.raw_root.resolve())
        self.assertEqual(inspection.archive_md5, self.source.archive_md5)
        self.assertEqual(inspection.archive_sha256, self.source.archive_sha256)
        self.assertEqual(len(inspection.accepted_members), 21)
        self.assertEqual(len(inspection.relevant_evidence_members), 46)
        self.assertEqual(len(inspection.evidence_snapshots), 3)
        self.assertEqual(len(inspection.recipes), 7)
        self.assertEqual(statistics["record_count"], 21)
        self.assertEqual(statistics["sample_count"], 7)
        self.assertEqual(statistics["rejected_canonical_members"], 0)
        self.assertEqual(statistics["points_per_record"], 16)
        self.assertEqual(statistics["high_records"], 7)
        self.assertEqual(statistics["low_records"], 14)
        self.assertEqual(statistics["pure_reference_records"], 15)
        self.assertEqual(statistics["mixture_records"], 6)
        self.assertEqual(statistics["axis_groups"], 1)
        self.assertEqual(
            set(inspection.central_directory_safety),
            {
                "duplicate_paths",
                "absolute_or_traversal_paths",
                "backslash_paths",
                "control_or_nul_paths",
                "non_ascii_paths",
                "nfc_collisions",
                "casefold_collisions",
                "symlink_members",
                "encrypted_members",
                "unsupported_member_types",
                "directory_entries",
                "regular_deflated_files",
            },
        )
        self.assertEqual(
            set(inspection.member_roles),
            {
                "canonical_high",
                "canonical_low",
                "target",
                "oracle",
                "prepared",
                "raw_support",
                "experimental_readme",
            },
        )
        self.assertEqual(
            tuple(inspection.view_record_ids),
            (
                "high_snr",
                "high_snr_no_refs",
                "high_snr_pure_reference",
                "low_snr",
                "low_snr_no_refs",
                "low_snr_pure_reference",
                "no_refs",
                "pure_reference",
            ),
        )
        self.assertEqual(
            {
                name: len(record_ids)
                for name, record_ids in inspection.view_record_ids.items()
            },
            {
                "high_snr": 7,
                "high_snr_no_refs": 2,
                "high_snr_pure_reference": 5,
                "low_snr": 14,
                "low_snr_no_refs": 4,
                "low_snr_pure_reference": 10,
                "no_refs": 6,
                "pure_reference": 15,
            },
        )
        self.assertEqual(
            {
                name: len(record_ids)
                for name, record_ids
                in inspection.endmember_constituent_record_ids.items()
            },
            {
                f"{condition}/{component}": count
                for condition, count in (("high_snr", 1), ("low_snr", 2))
                for component in (
                    "sucrose",
                    "fructose",
                    "maltose",
                    "glucose",
                    "water",
                )
            },
        )
        self.assertEqual(
            tuple(member.record_id for member in inspection.accepted_members),
            tuple(sorted(self.source.record_ids, key=lambda value: value.encode())),
        )
        self.assertFalse(contains_numpy_array(inspection))

    def test_inspection_distinguishes_claim_readme_from_outer_readme(self):
        inspection = self.inspect()
        evidence_roles = {
            member.source_member: member.role
            for member in inspection.relevant_evidence_members
        }

        self.assertEqual(
            evidence_roles[
                "Raw data/Experimental data from sugar mixtures/"
                "Raw data files/README.md"
            ],
            "raw_support",
        )
        self.assertEqual(
            evidence_roles[
                "Raw data/Experimental data from sugar mixtures/README.txt"
            ],
            "experimental_readme",
        )

    def test_inspection_materializes_exact_member_recipe_and_identity_fields(self):
        inspection = self.inspect()
        members = {
            member.record_id: member
            for member in inspection.accepted_members
        }
        mixture = members["high_snr-s001-a01-p01-r01-m01-rep01"]
        water = members["high_snr-s049-e01-p03-r01-m01-rep01"]
        sucrose = members["low_snr-s050-e02-p03-r01-m01-rep02"]

        self.assertEqual(mixture.record_role, "mixture")
        self.assertIsNone(mixture.pure_component)
        self.assertEqual(mixture.well_key, "A1_1")
        self.assertEqual(mixture.sample_id, "sugar-well-a01-p01")
        self.assertEqual(mixture.acquisition_id, mixture.record_id)
        self.assertEqual(water.record_role, "pure_reference")
        self.assertEqual(water.pure_component, "water")
        self.assertEqual(sucrose.record_role, "pure_reference")
        self.assertEqual(sucrose.pure_component, "sucrose")
        self.assertEqual(
            inspection.recipes["A1_1"].component_volumes_ul,
            {
                "sucrose": 30,
                "fructose": 75,
                "maltose": 120,
                "glucose": 0,
                "water": 150,
            },
        )
        self.assertEqual(
            inspection.recipes["A1_1"].component_fractions["maltose"],
            120 / 375,
        )
        self.assertEqual(
            inspection.recipes["A1_1"].concentrations_mol_l,
            {
                "sucrose_nominal_mol_l": 30 / 375,
                "fructose_nominal_mol_l": 75 / 375,
                "maltose_nominal_mol_l": 120 / 375,
                "glucose_nominal_mol_l": 0.0,
            },
        )

    def test_inspection_axis_ids_follow_exact_stored_value_domains(self):
        inspection = self.inspect()
        first = _reparse_verified_member(
            self.source.raw_root,
            inspection.accepted_members[0],
        )
        core_values = np.ascontiguousarray(first.wavenumber_cm1, dtype="<f4")
        pixel_values = np.ascontiguousarray(first.pixel, dtype="<u2")
        wavelength_values = np.ascontiguousarray(
            first.wavelength_nm,
            dtype="<f4",
        )

        pixel_digest = hashlib.sha256(b"rpe-sugar-pixel-axis-v1\0")
        pixel_digest.update(struct.pack("<Q", pixel_values.size))
        pixel_digest.update(pixel_values.tobytes())
        wavelength_digest = hashlib.sha256(
            b"rpe-sugar-wavelength-axis-v1\0"
        )
        wavelength_digest.update(struct.pack("<Q", wavelength_values.size))
        wavelength_digest.update(wavelength_values.tobytes())
        set_digest = hashlib.sha256(b"rpe-sugar-aux-axis-set-v1\0")
        for value in (
            axis_id(core_values),
            pixel_digest.hexdigest(),
            wavelength_digest.hexdigest(),
        ):
            encoded = value.encode("utf-8")
            set_digest.update(struct.pack("<Q", len(encoded)))
            set_digest.update(encoded)

        self.assertEqual(inspection.core_axis_id, axis_id(core_values))
        self.assertEqual(
            inspection.pixel_axis_id,
            pixel_digest.hexdigest(),
        )
        self.assertEqual(
            inspection.wavelength_axis_id,
            wavelength_digest.hexdigest(),
        )
        self.assertEqual(
            inspection.auxiliary_axis_set_id,
            set_digest.hexdigest(),
        )

    def test_inspection_identity_digest_uses_length_prefixed_source_bindings(self):
        inspection = self.inspect()
        digest = hashlib.sha256(b"rpe-sugar-identity-map-v1\0")
        for member in inspection.accepted_members:
            for value in (
                member.record_id,
                member.sample_id,
                member.acquisition_id,
                member.source_member,
            ):
                encoded = value.encode("utf-8")
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)

        self.assertEqual(
            inspection.source_statistics["identity_map_sha256"],
            digest.hexdigest(),
        )
        self.assertEqual(
            inspection.source_statistics["maximum_record_id_bytes"],
            35,
        )

    def test_inspection_collections_are_transitively_immutable(self):
        inspection = self.inspect()

        with self.assertRaises(FrozenInstanceError):
            inspection.archive_sha256 = "changed"
        with self.assertRaises(TypeError):
            inspection.source_statistics["record_count"] = 0
        with self.assertRaises(TypeError):
            inspection.member_roles["target"]["member_count"] = 0
        with self.assertRaises(TypeError):
            inspection.recipes["A1_1"] = None
        with self.assertRaises(TypeError):
            inspection.recipes["A1_1"].component_volumes_ul["water"] = 0
        with self.assertRaises(TypeError):
            inspection.view_record_ids["high_snr"] = ()

    def test_public_inspection_uses_fixed_contract_and_private_wrapper(self):
        with patch(
            "rpe.io.sugar_mixtures._PRODUCTION_SOURCE_CONTRACT",
            self.contract,
        ):
            public = inspect_sugar_mixtures(self.source.raw_root)
        private = _inspect_sugar_mixtures(
            self.source.raw_root,
            self.contract,
        )

        self.assertEqual(public, private)
        self.assertEqual(public.archive_sha256, self.source.archive_sha256)

    def test_preflight_rejects_missing_and_unexpected_archive(self):
        source, contract = self.create_source("archive-paths")
        source.archive_path.rename(source.raw_root / "renamed.zip")
        self.assert_invalid("ARCHIVE_MISSING", source, contract)

        source, contract = self.create_source("unexpected-archive")
        (source.raw_root / "unexpected.zip").write_bytes(b"not a zip")
        self.assert_invalid("ARCHIVE_UNEXPECTED", source, contract)

    def test_preflight_rejects_archive_size_md5_and_sha256_drift(self):
        for field, value, code in (
            ("archive_bytes", self.contract.archive_bytes + 1, "ARCHIVE_BYTES_MISMATCH"),
            ("archive_md5", "0" * 32, "ARCHIVE_MD5_MISMATCH"),
            ("archive_sha256", "0" * 64, "ARCHIVE_SHA256_MISMATCH"),
        ):
            with self.subTest(field=field):
                self.assert_invalid(
                    code,
                    contract=replace(self.contract, **{field: value}),
                )

    def test_preflight_rejects_crc_member_count_bytes_and_inventory_drift(self):
        source, _ = self.create_source("crc")
        corrupt_synthetic_zip_member_data(
            source.archive_path,
            source.canonical_members[0],
        )
        contract = refresh_synthetic_sugar_contract(source)
        self.assert_invalid("ARCHIVE_CRC_FAILURE", source, contract)

        for field, value, code in (
            (
                "archive_member_count",
                self.contract.archive_member_count + 1,
                "ARCHIVE_MEMBER_COUNT_MISMATCH",
            ),
            (
                "archive_uncompressed_bytes",
                self.contract.archive_uncompressed_bytes + 1,
                "ARCHIVE_UNCOMPRESSED_BYTES_MISMATCH",
            ),
            (
                "central_directory_inventory_sha256",
                "0" * 64,
                "ARCHIVE_INVENTORY_MISMATCH",
            ),
        ):
            with self.subTest(field=field):
                self.assert_invalid(
                    code,
                    contract=replace(self.contract, **{field: value}),
                )

    def test_preflight_rejects_duplicate_member_path(self):
        source, _ = self.create_source("duplicate-path")
        entries = read_synthetic_zip_entries(source.archive_path)
        rewrite_synthetic_zip(source, lambda _: (*entries, entries[0]))
        contract = refresh_synthetic_sugar_contract(source)
        self.assert_invalid("MEMBER_PATH_DUPLICATE", source, contract)

    def test_preflight_rejects_unsafe_backslash_control_and_nonascii_paths(self):
        source_names = (
            ("/absolute.csv", "MEMBER_PATH_UNSAFE"),
            ("C:/absolute.csv", "MEMBER_PATH_UNSAFE"),
            ("../traversal.csv", "MEMBER_PATH_UNSAFE"),
            ("Raw data\\backslash.csv", "MEMBER_PATH_BACKSLASH"),
            ("Raw data/control\x01.csv", "MEMBER_PATH_CONTROL"),
            ("Raw data/café.csv", "MEMBER_PATH_NONASCII"),
        )
        for index, (new_name, code) in enumerate(source_names):
            with self.subTest(new_name=new_name):
                source, _ = self.create_source(f"unsafe-{index}")
                old_name = source.role_members["excluded_synthetic"][0]
                contract = self.rename_member_contract(
                    source,
                    old_name,
                    new_name,
                )
                self.assert_invalid(code, source, contract)

    def test_preflight_rejects_casefold_and_nfc_collisions(self):
        source, _ = self.create_source("casefold")
        old_name = source.role_members["excluded_synthetic"][0]
        collision = EXPERIMENTAL_README.swapcase()
        contract = self.rename_member_contract(source, old_name, collision)
        self.assert_invalid("MEMBER_PATH_CASEFOLD_COLLISION", source, contract)

        source, _ = self.create_source("nfc")
        first, second = source.role_members["raw_support"][:2]
        first_name = "Raw data/support/caf\u00e9.txt"
        second_name = "Raw data/support/cafe\u0301.txt"
        rewrite_synthetic_zip(
            source,
            lambda entries: replace_zip_entry(
                replace_zip_entry(entries, first, name=first_name),
                second,
                name=second_name,
            ),
        )
        roles = {
            role: tuple(
                first_name if name == first else second_name if name == second else name
                for name in names
            )
            for role, names in source.role_members.items()
        }
        contract = refresh_synthetic_sugar_contract(
            source,
            refresh_roles=True,
            role_members=roles,
        )
        self.assert_invalid("MEMBER_PATH_NFC_COLLISION", source, contract)

    def test_preflight_rejects_symlink_encrypted_type_and_compression(self):
        member = self.source.role_members["excluded_synthetic"][0]
        variants = (
            (
                "symlink",
                {"external_attr": (stat.S_IFLNK | 0o777) << 16},
                "MEMBER_SYMLINK",
            ),
            (
                "fifo",
                {"external_attr": (stat.S_IFIFO | 0o644) << 16},
                "MEMBER_TYPE_UNSUPPORTED",
            ),
            (
                "stored",
                {"compress_type": zipfile.ZIP_STORED},
                "MEMBER_COMPRESSION_UNSUPPORTED",
            ),
        )
        for name, updates, code in variants:
            with self.subTest(name=name):
                source, _ = self.create_source(name)
                rewrite_synthetic_zip(
                    source,
                    lambda entries, updates=updates: replace_zip_entry(
                        entries,
                        source.role_members["excluded_synthetic"][0],
                        **updates,
                    ),
                )
                contract = refresh_synthetic_sugar_contract(source)
                self.assert_invalid(code, source, contract)

        source, _ = self.create_source("encrypted")
        member = source.role_members["excluded_synthetic"][0]
        set_synthetic_zip_member_flag_bits(source.archive_path, member, 0x1)
        contract = refresh_synthetic_sugar_contract(source)
        self.assert_invalid("MEMBER_ENCRYPTED", source, contract)

    def test_preflight_accepts_safe_directory_entries(self):
        source, _ = self.create_source("directory-entry")
        entries = read_synthetic_zip_entries(source.archive_path)
        directory = SyntheticZipEntry(
            name="Raw data/empty/",
            payload=b"",
            compress_type=zipfile.ZIP_STORED,
            external_attr=(stat.S_IFDIR | 0o755) << 16,
        )
        rewrite_synthetic_zip(source, lambda _: (*entries, directory))
        contract = refresh_synthetic_sugar_contract(source)
        expected_counts = dict(contract.expected_counts)
        expected_counts.update(
            {
                "directory_entries": 1,
                "regular_deflated_files": 68,
            }
        )
        contract = replace(
            contract,
            expected_counts=MappingProxyType(expected_counts),
        )

        inspection = self.inspect(source, contract)

        self.assertEqual(
            inspection.central_directory_safety["directory_entries"],
            1,
        )
        self.assertEqual(
            inspection.central_directory_safety["regular_deflated_files"],
            68,
        )

    def test_role_contract_rejects_count_bytes_and_digest_drift(self):
        for field, changed, code in (
            (
                "expected_role_counts",
                {"canonical_high": 8},
                "ROLE_COUNT_MISMATCH",
            ),
            (
                "expected_role_bytes",
                {
                    "target": self.contract.expected_role_bytes["target"] + 1,
                },
                "ROLE_BYTES_MISMATCH",
            ),
            (
                "expected_role_sha256",
                {"oracle": "0" * 64},
                "ROLE_DIGEST_MISMATCH",
            ),
        ):
            with self.subTest(field=field):
                values = dict(getattr(self.contract, field))
                values.update(changed)
                self.assert_invalid(
                    code,
                    contract=replace(
                        self.contract,
                        **{field: MappingProxyType(values)},
                    ),
                )

    def test_role_digest_uses_domain_and_length_prefixed_paths(self):
        expected = {}
        for role, names in self.source.role_members.items():
            if role == "excluded_synthetic":
                continue
            digest = hashlib.sha256(b"rpe-sugar-member-content-v1\0")
            for name in sorted(names, key=lambda value: value.encode("utf-8")):
                encoded = name.encode("utf-8")
                digest.update(struct.pack("<Q", len(encoded)))
                digest.update(encoded)
                digest.update(bytes.fromhex(self.source.member_sha256[name]))
            expected[role] = digest.hexdigest()

        self.assertEqual(
            dict(self.contract.expected_role_sha256),
            expected,
        )

    def test_contract_rejects_missing_role_digest_or_evidence_snapshot(self):
        role_digests = dict(self.contract.expected_role_sha256)
        role_digests.pop("raw_support")
        self.assert_invalid(
            "ROLE_DIGEST_CONTRACT_INVALID",
            contract=replace(
                self.contract,
                expected_role_sha256=MappingProxyType(role_digests),
            ),
        )

        evidence = dict(self.contract.expected_evidence_snapshots)
        evidence.pop("receipts/sugar_mixtures_low_snr.json")
        self.assert_invalid(
            "EVIDENCE_CONTRACT_INVALID",
            contract=replace(
                self.contract,
                expected_evidence_snapshots=MappingProxyType(evidence),
            ),
        )

    def test_inspection_does_not_semantically_hash_excluded_payloads(self):
        excluded_members = set(
            self.source.role_members["excluded_synthetic"]
        )
        original_hash = sugar_source_module._hash_zip_member

        def reject_excluded_hash(archive, info):
            if info.filename in excluded_members:
                raise AssertionError(
                    "excluded subtree must not enter semantic member hashing"
                )
            return original_hash(archive, info)

        with patch(
            "rpe.io.sugar_mixtures_source._hash_zip_member",
            side_effect=reject_excluded_hash,
        ):
            inspection = self.inspect()

        self.assertEqual(
            inspection.source_statistics[
                "excluded_synthetic_regular_members"
            ],
            1,
        )

    def test_classification_rejects_unknown_and_unexpected_canonical_member(self):
        source, _ = self.create_source("unknown-role")
        old_name = source.role_members["excluded_synthetic"][0]
        contract = self.rename_member_contract(
            source,
            old_name,
            (
                "Raw data/Experimental data from sugar mixtures/"
                "unclassified.bin"
            ),
        )
        self.assert_invalid("MEMBER_ROLE_UNKNOWN", source, contract)

        source, _ = self.create_source("unexpected-canonical")
        entries = read_synthetic_zip_entries(source.archive_path)
        unexpected = SyntheticZipEntry(
            name=f"{HIGH_DIRECTORY}unexpected.txt",
            payload=b"unexpected\n",
        )
        rewrite_synthetic_zip(source, lambda _: (*entries, unexpected))
        contract = refresh_synthetic_sugar_contract(source)
        self.assert_invalid("CANONICAL_FILENAME_INVALID", source, contract)

    def test_canonical_rejects_filename_identity_and_duplicate_acquisition(self):
        source, _ = self.create_source("filename-identity")
        old_name = source.canonical_members[0]
        new_name = old_name.replace("_1_A1_1_", "_2_A1_1_")
        contract = self.rename_member_contract(source, old_name, new_name)
        self.assert_invalid("CANONICAL_IDENTITY_INVALID", source, contract)

        source, _ = self.create_source("filename-reversibility")
        old_name = next(
            name
            for name in source.canonical_members
            if "_1_A1_1_" in name and name.startswith(HIGH_DIRECTORY)
        )
        new_name = old_name.replace("_A1_", "_A01_")
        contract = self.rename_member_contract(source, old_name, new_name)
        self.assert_invalid("CANONICAL_IDENTITY_INVALID", source, contract)

        source, _ = self.create_source("filename-nested")
        old_name = source.canonical_members[0]
        new_name = (
            HIGH_DIRECTORY
            + "nested/"
            + Path(old_name).name
        )
        contract = self.rename_member_contract(source, old_name, new_name)
        self.assert_invalid("CANONICAL_FILENAME_INVALID", source, contract)

    def test_canonical_rejects_per_well_condition_cardinality_drift(self):
        source, _ = self.create_source("well-cardinality")
        old_name = next(
            name
            for name in source.canonical_members
            if "_2_A2_1_RD1_M1_R1.csv" in name
            and name.startswith(HIGH_DIRECTORY)
        )
        new_name = old_name.replace(
            "_2_A2_1_RD1_M1_R1.csv",
            "_1_A1_1_RD2_M1_R1.csv",
        )
        contract = self.rename_member_contract(source, old_name, new_name)

        self.assert_invalid(
            "CANONICAL_WELL_COUNT_MISMATCH",
            source,
            contract,
        )

    def test_canonical_rejects_incomplete_round_repetition_grid(self):
        source, _ = self.create_source("acquisition-grid")
        old_name = next(
            name
            for name in source.canonical_members
            if "_1_A1_1_RD1_M1_R2.csv" in name
            and name.startswith(LOW_DIRECTORY)
        )
        new_name = old_name.replace(
            "_1_A1_1_RD1_M1_R2.csv",
            "_1_A1_1_RD2_M1_R1.csv",
        )
        contract = self.rename_member_contract(source, old_name, new_name)

        self.assert_invalid(
            "CANONICAL_ACQUISITION_GRID_MISMATCH",
            source,
            contract,
        )

    def test_target_rejects_header_width_value_duplicate_and_volume_drift(self):
        mutations = (
            (
                "header",
                lambda rows: rows[0].__setitem__(0, "Wrong"),
                "TARGET_HEADER_INVALID",
            ),
            (
                "width",
                lambda rows: rows[1].pop(),
                "TARGET_ROW_WIDTH_INVALID",
            ),
            (
                "null",
                lambda rows: rows[1].__setitem__(5, ""),
                "TARGET_VALUE_INVALID",
            ),
            (
                "duplicate",
                lambda rows: rows.append(list(rows[1])),
                "TARGET_WELL_DUPLICATE",
            ),
            (
                "volume",
                lambda rows: rows[1].__setitem__(10, "374"),
                "TARGET_VOLUME_INVALID",
            ),
        )
        for name, mutate, code in mutations:
            with self.subTest(name=name):
                source, _ = self.create_source(f"target-{name}")
                contract = self.mutate_target(source, mutate)
                self.assert_invalid(code, source, contract)

    def test_target_rejects_missing_well_and_recipe_identity_drift(self):
        source, _ = self.create_source("missing-well")

        def replace_a2_with_a3(rows):
            rows[2][0] = "A3_1"
            rows[2][1] = "3"
            rows[2][3] = "3"

        contract = self.mutate_target(source, replace_a2_with_a3)
        self.assert_invalid("RECIPE_MISSING", source, contract)

        source, _ = self.create_source("recipe-identity")
        contract = self.mutate_target(
            source,
            lambda rows: rows[1].__setitem__(1, "2"),
        )
        self.assert_invalid("RECIPE_IDENTITY_INVALID", source, contract)

    def test_target_rejects_unapproved_sugar_level_and_duplicate_target_vector(self):
        source, _ = self.create_source("recipe-level")

        def unapproved_level(rows):
            rows[1][5] = "31"
            rows[1][9] = "149"

        contract = self.mutate_target(source, unapproved_level)
        self.assert_invalid("RECIPE_LEVEL_INVALID", source, contract)

        source, _ = self.create_source("duplicate-target")

        def duplicate_target(rows):
            rows[2][5:] = list(rows[1][5:])

        contract = self.mutate_target(source, duplicate_target)
        self.assert_invalid("RECIPE_TARGET_DUPLICATE", source, contract)

    def test_target_rejects_payload_drift_after_preflight_by_hash(self):
        original_read = zipfile.ZipFile.read

        def drift_target(archive, name, *args, **kwargs):
            payload = original_read(archive, name, *args, **kwargs)
            member_name = (
                name.filename
                if isinstance(name, zipfile.ZipInfo)
                else name
            )
            if member_name != TARGET_MEMBER:
                return payload
            rows = target_rows(payload)
            rows[1][5] = "31"
            rows[1][9] = "149"
            return payload_from_rows(rows)

        with patch.object(
            zipfile.ZipFile,
            "read",
            autospec=True,
            side_effect=drift_target,
        ):
            self.assert_invalid("MEMBER_HASH_MISMATCH")

    def test_inspection_rejects_shared_axis_and_exact_pixel_drift(self):
        variants = (
            ("wavelength", 1, "800.25", "AXIS_DRIFT"),
            ("pixel", 0, "1", "PIXEL_AXIS_INVALID"),
        )
        for name, column, value, code in variants:
            with self.subTest(name=name):
                source, _ = self.create_source(f"axis-{name}")
                member = source.canonical_members[-1]

                def transform(entries):
                    target = next(
                        entry for entry in entries if entry.name == member
                    )
                    rows = csv_rows(target.payload)
                    if name == "pixel":
                        for row in rows[1:-1]:
                            row[0] = str(int(row[0]) + 1)
                    else:
                        rows[1][column] = value
                    return replace_zip_entry(
                        entries,
                        member,
                        payload=payload_from_rows(rows),
                    )

                rewrite_synthetic_zip(source, transform)
                contract = refresh_synthetic_sugar_contract(
                    source,
                    refresh_roles=True,
                )
                self.assert_invalid(code, source, contract)

    def test_inspection_rejects_missing_hash_drifted_evidence_snapshot(self):
        evidence_root = self.source.raw_root.parent.parent
        relative = "evidence/zenodo/10779223.json"
        evidence = evidence_root / relative
        evidence.unlink()
        self.assert_invalid("EVIDENCE_MISSING")

        source, contract = self.create_source("evidence-size")
        evidence = source.raw_root.parent.parent / relative
        evidence.write_bytes(evidence.read_bytes() + b"x")
        self.assert_invalid("EVIDENCE_BYTES_MISMATCH", source, contract)

        source, contract = self.create_source("evidence-hash")
        evidence = source.raw_root.parent.parent / relative
        payload = bytearray(evidence.read_bytes())
        payload[0] ^= 1
        evidence.write_bytes(payload)
        adjusted = dict(contract.expected_evidence_snapshots)
        adjusted[relative] = (
            evidence.stat().st_size,
            contract.expected_evidence_snapshots[relative][1],
        )
        contract = replace(
            contract,
            expected_evidence_snapshots=MappingProxyType(adjusted),
        )
        self.assert_invalid("EVIDENCE_SHA256_MISMATCH", source, contract)

    def test_inspection_ignores_stale_extracted_cache(self):
        first = self.inspect()
        stale = (
            self.source.raw_root
            / "Raw data"
            / "Experimental data from sugar mixtures"
            / "Raw data files"
            / "Sugar_Concentrations.csv"
        )
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale extracted cache\n")
        second = self.inspect()

        self.assertEqual(first, second)

    def test_reparse_verified_member_rejects_post_inspection_archive_mutation(self):
        inspection = self.inspect()
        accepted = inspection.accepted_members[0]
        member = accepted.source_member
        rewrite_synthetic_zip(
            self.source,
            lambda entries: replace_zip_entry(
                entries,
                member,
                payload=build_member_payload(record_index=999),
            ),
        )

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _reparse_verified_member(self.source.raw_root, accepted)
        self.assertEqual(captured.exception.code, "MEMBER_HASH_MISMATCH")

    def test_reparse_verified_member_rejects_missing_member_binding(self):
        inspection = self.inspect()
        accepted = inspection.accepted_members[0]
        rewrite_synthetic_zip(
            self.source,
            lambda entries: tuple(
                entry
                for entry in entries
                if entry.name != accepted.source_member
            ),
        )

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _reparse_verified_member(self.source.raw_root, accepted)
        self.assertEqual(captured.exception.code, "MEMBER_BINDING_MISMATCH")


class SugarMixturesRecordMappingTest(unittest.TestCase):
    OUTER_SOURCE_KEYS = {
        "source_member",
        "source_member_sha256",
        "source_basename",
        "source_samp",
        "source_row",
        "source_column",
        "source_plate",
        "source_round",
        "source_measurement",
        "source_repetition",
        "source_acquisition_id",
        "source_acquisition_condition",
        "source_record_role",
        "source_auxiliary_axis_set_id",
        "source_acquisition_metadata",
        "source_sucrose_volume_ul",
        "source_fructose_volume_ul",
        "source_maltose_volume_ul",
        "source_glucose_volume_ul",
        "source_water_volume_ul",
        "source_total_volume_ul",
        "source_component_fractions",
        "source_preprocessing_state",
        "source_preprocessing_evidence_id",
    }
    NORMALIZED_KEYS = {
        "acquired_at_local",
        "acquired_at_timezone",
        "stage_position_um",
        "ambient_temperature_c",
        "ambient_temperature_source_text",
        "ambient_humidity_hundredth",
        "ambient_humidity_source_text",
        "spectro_temp_c",
        "laser_power_mw",
        "excitation_nm",
        "objective_name",
        "objective_maker",
        "objective_magnification",
        "objective_na",
        "objective_wd_source",
        "objective_immersion",
        "objective_tube_lens_f_source",
        "integration_time_s",
        "n_accumulations",
        "spectrometer_temperature_c",
        "spectrometer_acquisition_mode_code",
        "spectrometer_read_mode_code",
        "spectrometer_trigger_mode_code",
        "magnification",
        "spot_size_source",
        "laser_offset_source",
    }
    TARGET_NAMES = {
        "sucrose_nominal_mol_l",
        "fructose_nominal_mol_l",
        "maltose_nominal_mol_l",
        "glucose_nominal_mol_l",
    }

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.contract = synthetic_sugar_contract(self.source)
        self.inspection = _inspect_sugar_mixtures_source(
            self.source.raw_root,
            self.contract,
        )

    def records(self, raw_root=None, inspection=None):
        return list(
            iter_sugar_mixtures_records(
                self.source.raw_root if raw_root is None else raw_root,
                self.inspection if inspection is None else inspection,
            )
        )

    def mutate_member_metadata(self, source, member, updates):
        def transform(entries):
            entry = next(
                current
                for current in entries
                if current.name == member
            )
            rows = csv_rows(entry.payload)
            document = json.loads(rows[-1][4])
            document.update(updates)
            rows[-1][4] = json.dumps(
                document,
                ensure_ascii=False,
                separators=(", ", ": "),
                allow_nan=True,
            )
            return replace_zip_entry(
                entries,
                member,
                payload=payload_from_rows(rows),
            )

        rewrite_synthetic_zip(source, transform)

    def test_every_synthetic_record_maps_source_truth(self):
        records = self.records()
        members = {
            member.record_id: member
            for member in self.inspection.accepted_members
        }

        self.assertEqual(len(records), 21)
        self.assertTrue(all(isinstance(record, RamanRecord) for record in records))
        self.assertEqual(
            tuple(record.record_id for record in records),
            tuple(sorted(self.source.record_ids, key=lambda value: value.encode())),
        )
        for record in records:
            self.assertIs(validate_record(record), record)
            self.assertEqual(record.meta.dataset_id, "sugar_mixtures_raman")
            self.assertIs(
                record.meta.preprocessing_status,
                PreprocessingStatus.KNOWN_RAW,
            )
            self.assertEqual(record.meta.preprocessing_steps, ())
            self.assertTrue(
                record.meta.eligible_for_preprocessing_evaluation
            )
            self.assertIsNone(record.targets.clean)
            self.assertIsNone(record.targets.baseline)
            self.assertIsNone(record.targets.peaks)
            self.assertIsNone(record.targets.class_label)
            self.assertIsNone(record.targets.concentration)
            self.assertEqual(
                set(record.targets.concentrations),
                self.TARGET_NAMES,
            )
            self.assertEqual(record.intensity.dtype, np.dtype("<f4"))
            self.assertEqual(record.wavenumber.dtype, np.dtype("<f4"))
            self.assertEqual(
                record.intensity.flags.writeable,
                False,
            )
            self.assertEqual(
                record.wavenumber.flags.writeable,
                False,
            )
            parsed = _reparse_verified_member(
                self.source.raw_root,
                members[record.record_id],
            )
            np.testing.assert_array_equal(
                record.intensity,
                parsed.intensity.astype("<f4"),
            )
            np.testing.assert_array_equal(
                record.wavenumber,
                parsed.wavenumber_cm1.astype("<f4"),
            )

    def test_metadata_preserves_identity_condition_and_normalized_source(self):
        records = {
            record.record_id: record
            for record in self.records()
        }
        record = records["high_snr-s001-a01-p01-r01-m01-rep01"]
        source = record.meta.source_metadata
        acquisition = source["source_acquisition_metadata"]

        self.assertEqual(set(source), self.OUTER_SOURCE_KEYS)
        self.assertEqual(set(acquisition), self.NORMALIZED_KEYS)
        self.assertEqual(record.meta.sample_id, "sugar-well-a01-p01")
        self.assertEqual(
            source["source_acquisition_id"],
            record.record_id,
        )
        self.assertEqual(source["source_samp"], 1)
        self.assertEqual(source["source_row"], "A")
        self.assertEqual(source["source_column"], 1)
        self.assertEqual(source["source_plate"], 1)
        self.assertEqual(source["source_round"], 1)
        self.assertEqual(source["source_measurement"], 1)
        self.assertEqual(source["source_repetition"], 1)
        self.assertEqual(source["source_acquisition_condition"], "high_snr")
        self.assertEqual(source["source_record_role"], "mixture")
        self.assertEqual(
            source["source_auxiliary_axis_set_id"],
            self.inspection.auxiliary_axis_set_id,
        )
        self.assertEqual(
            acquisition["acquired_at_local"],
            "2023-09-06T19:00:17",
        )
        self.assertIsNone(acquisition["acquired_at_timezone"])
        self.assertEqual(
            acquisition["stage_position_um"],
            [0.0, 1.0, 0.0],
        )
        self.assertIsNone(acquisition["ambient_temperature_c"])
        self.assertEqual(
            acquisition["ambient_temperature_source_text"],
            "N/A",
        )
        self.assertIsNone(acquisition["ambient_humidity_hundredth"])
        self.assertEqual(
            acquisition["ambient_humidity_source_text"],
            "N/A",
        )
        self.assertEqual(
            acquisition["laser_offset_source"],
            [
                {"source_nonfinite_number": "NaN"},
                {"source_nonfinite_number": "NaN"},
            ],
        )
        self.assertNotIn("source_json_text", source)
        self.assertNotIn("pair_id", source)
        self.assertNotIn("split", source)

    def test_core_acquisition_fields_and_redundancy_are_exact(self):
        records = self.records()

        for record in records:
            acquisition = record.meta.source_metadata[
                "source_acquisition_metadata"
            ]
            condition = record.meta.source_metadata[
                "source_acquisition_condition"
            ]
            self.assertEqual(
                record.meta.instrument,
                "B-Raman custom Raman microspectroscopy platform",
            )
            self.assertEqual(record.meta.excitation_nm, 785.0)
            self.assertEqual(record.meta.n_accumulations, 1)
            self.assertIsNone(record.meta.grating)
            self.assertIsNone(record.meta.detector)
            self.assertEqual(
                record.meta.integration_time_s,
                5.0 if condition == "high_snr" else 0.5,
            )
            self.assertEqual(
                acquisition["laser_power_mw"],
                36.3 if condition == "high_snr" else 30.3,
            )
            self.assertEqual(
                acquisition["excitation_nm"],
                record.meta.excitation_nm,
            )
            self.assertEqual(
                acquisition["integration_time_s"],
                record.meta.integration_time_s,
            )
            self.assertEqual(
                acquisition["n_accumulations"],
                record.meta.n_accumulations,
            )
            self.assertEqual(
                acquisition["magnification"],
                acquisition["objective_magnification"],
            )
            objective = acquisition["objective_name"]
            self.assertIn(
                f"{acquisition['objective_magnification']:g}x",
                objective,
            )
            self.assertIn(
                f"/{acquisition['objective_na']:g}",
                objective,
            )

    def test_recipe_metadata_and_four_targets_are_exact(self):
        records = {
            record.record_id: record
            for record in self.records()
        }
        mixture = records["high_snr-s001-a01-p01-r01-m01-rep01"]
        water = records["high_snr-s049-e01-p03-r01-m01-rep01"]
        sucrose = records["low_snr-s050-e02-p03-r01-m01-rep02"]

        source = mixture.meta.source_metadata
        self.assertEqual(source["source_sucrose_volume_ul"], 30)
        self.assertEqual(source["source_fructose_volume_ul"], 75)
        self.assertEqual(source["source_maltose_volume_ul"], 120)
        self.assertEqual(source["source_glucose_volume_ul"], 0)
        self.assertEqual(source["source_water_volume_ul"], 150)
        self.assertEqual(source["source_total_volume_ul"], 375)
        self.assertEqual(
            source["source_component_fractions"],
            {
                "sucrose": 30 / 375,
                "fructose": 75 / 375,
                "maltose": 120 / 375,
                "glucose": 0.0,
                "water": 150 / 375,
            },
        )
        self.assertEqual(
            mixture.targets.concentrations,
            {
                "sucrose_nominal_mol_l": 30 / 375,
                "fructose_nominal_mol_l": 75 / 375,
                "maltose_nominal_mol_l": 120 / 375,
                "glucose_nominal_mol_l": 0.0,
            },
        )
        self.assertNotIn("water_nominal_mol_l", mixture.targets.concentrations)
        self.assertEqual(
            set(water.targets.concentrations.values()),
            {0.0},
        )
        self.assertEqual(
            sucrose.targets.concentrations["sucrose_nominal_mol_l"],
            1.0,
        )

    def test_preprocessing_evidence_is_bounded_and_exact(self):
        records = self.records()
        evidence = sugar_mixtures_module._preprocessing_evidence_document(
            self.source.raw_root,
            self.inspection,
        )

        self.assertEqual(
            evidence,
            {
                "evidence_id": "sugar-source-no-preprocessing-claim-v1",
                "evidence_type": "source_dataset_statement",
                "source_member": CLAIM_README,
                "source_member_sha256": (
                    "205573a04a62bcb9658a3e7b0ac4c28fa5014d747760f319f0c708369fc1b4ed"
                ),
                "source_member_bytes": 144,
                "source_line": 2,
                "source_line_text_exact": (
                    "Note that no pre-processing has been performed such that "
                    "the user can decide what is best for the analysis. "
                ),
                "claim_text_normalized": (
                    "Note that no pre-processing has been performed such that "
                    "the user can decide what is best for the analysis."
                ),
                "applies_to_source_directories": [
                    HIGH_DIRECTORY.rstrip("/"),
                    LOW_DIRECTORY.rstrip("/"),
                ],
                "mapped_preprocessing_status": "known_raw",
                "mapped_preprocessing_steps": [],
                "eligible_for_preprocessing_evaluation": True,
                "scope": (
                    "source-declared absence of dataset-level preprocessing "
                    "for canonical individual measurement CSVs"
                ),
                "limitations": [
                    "does not prove detector-native counts",
                    (
                        "does not exclude instrument-firmware or "
                        "acquisition/export-layer operations"
                    ),
                    "does not make High SNR a clean target for Low SNR",
                    (
                        "does not classify prepared analysis artifacts as "
                        "canonical records"
                    ),
                ],
            },
        )
        self.assertTrue(
            all(
                record.meta.source_metadata[
                    "source_preprocessing_state"
                ] == "source_declared_unprocessed"
                for record in records
            )
        )
        self.assertTrue(
            all(
                record.meta.source_metadata[
                    "source_preprocessing_evidence_id"
                ] == "sugar-source-no-preprocessing-claim-v1"
                for record in records
            )
        )

    def test_provenance_is_member_specific_and_standardized(self):
        members = {
            member.record_id: member
            for member in self.inspection.accepted_members
        }
        for record in self.records():
            member = members[record.record_id]
            self.assertEqual(
                record.provenance.source_url,
                "https://doi.org/10.5281/zenodo.10779223",
            )
            self.assertEqual(record.provenance.license, "CC BY 4.0")
            self.assertIs(
                record.provenance.license_status,
                LicenseStatus.STANDARDIZED,
            )
            self.assertEqual(
                record.provenance.retrieved_date,
                date(2026, 8, 14),
            )
            self.assertEqual(record.provenance.sha256, member.sha256)
            self.assertEqual(
                record.provenance.source_artifact,
                member.source_member,
            )
            self.assertEqual(
                record.meta.source_metadata["source_member_sha256"],
                member.sha256,
            )

    def test_iterator_is_sequential_and_does_not_materialize_all_records(self):
        real_reparse = sugar_mixtures_module._reparse_verified_member
        calls = []

        def counting_reparse(raw_root, member, **kwargs):
            calls.append(member.record_id)
            return real_reparse(raw_root, member, **kwargs)

        with patch(
            "rpe.io.sugar_mixtures._reparse_verified_member",
            side_effect=counting_reparse,
        ):
            iterator = iter_sugar_mixtures_records(
                self.source.raw_root,
                self.inspection,
            )
            self.assertEqual(calls, [])
            first = next(iterator)
            self.assertEqual(calls, [first.record_id])
            second = next(iterator)
            self.assertEqual(calls, [first.record_id, second.record_id])
            iterator.close()

    def test_iterator_indexes_zip_members_once_not_once_per_record(self):
        real_infolist = zipfile.ZipFile.infolist
        calls = []

        def counting_infolist(archive):
            calls.append(archive.filename)
            return real_infolist(archive)

        with patch.object(
            zipfile.ZipFile,
            "infolist",
            autospec=True,
            side_effect=counting_infolist,
        ):
            records = self.records()

        self.assertEqual(len(records), 21)
        self.assertLessEqual(len(calls), 3)

    def test_iterator_rejects_root_mismatch_and_stale_inspection(self):
        wrong_root = self.temporary_root / "wrong-root"
        with self.assertRaises(SugarMixturesValidationError) as captured:
            next(
                iter_sugar_mixtures_records(
                    wrong_root,
                    self.inspection,
                )
            )
        self.assertEqual(captured.exception.code, "INSPECTION_ROOT_MISMATCH")

        stale = replace(
            self.inspection,
            archive_sha256="0" * 64,
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            next(
                iter_sugar_mixtures_records(
                    self.source.raw_root,
                    stale,
                )
            )
        self.assertEqual(captured.exception.code, "INSPECTION_STALE")

    def test_iterator_rejects_member_mutation_after_inspection(self):
        member = self.inspection.accepted_members[0].source_member
        original_read = zipfile.ZipFile.read

        def drift_member(archive, name, *args, **kwargs):
            payload = original_read(archive, name, *args, **kwargs)
            member_name = (
                name.filename
                if isinstance(name, zipfile.ZipInfo)
                else name
            )
            if member_name == member:
                return build_member_payload(record_index=999)
            return payload

        with patch.object(
            zipfile.ZipFile,
            "read",
            autospec=True,
            side_effect=drift_member,
        ):
            with self.assertRaises(SugarMixturesValidationError) as captured:
                next(
                    iter_sugar_mixtures_records(
                        self.source.raw_root,
                        self.inspection,
                    )
                )
        self.assertEqual(captured.exception.code, "MEMBER_HASH_MISMATCH")

    def test_iterator_revalidates_target_and_preprocessing_authorities(self):
        original_read = zipfile.ZipFile.read

        for member_name, replacement, code in (
            (
                TARGET_MEMBER,
                b"Well,Samp,Row,Column,Plate\n",
                "TARGET_AUTHORITY_INVALID",
            ),
            (
                CLAIM_README,
                b"X" * self.source.member_bytes[CLAIM_README],
                "PREPROCESSING_EVIDENCE_INVALID",
            ),
        ):
            with self.subTest(member_name=member_name):
                def drift_member(archive, name, *args, **kwargs):
                    payload = original_read(archive, name, *args, **kwargs)
                    observed_name = (
                        name.filename
                        if isinstance(name, zipfile.ZipInfo)
                        else name
                    )
                    if observed_name == member_name:
                        return replacement
                    return payload

                with patch.object(
                    zipfile.ZipFile,
                    "read",
                    autospec=True,
                    side_effect=drift_member,
                ):
                    with self.assertRaises(
                        SugarMixturesValidationError
                    ) as captured:
                        next(
                            iter_sugar_mixtures_records(
                                self.source.raw_root,
                                self.inspection,
                            )
                        )
                self.assertEqual(captured.exception.code, code)

    def test_iterator_rejects_counterfeit_identity_and_recipe_inspection(self):
        first = self.inspection.accepted_members[0]
        counterfeit_member = replace(
            first,
            sample_id="sugar-well-h12-p99",
        )
        counterfeit_members = (
            counterfeit_member,
            *self.inspection.accepted_members[1:],
        )
        counterfeit = replace(
            self.inspection,
            accepted_members=counterfeit_members,
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            next(
                iter_sugar_mixtures_records(
                    self.source.raw_root,
                    counterfeit,
                )
            )
        self.assertEqual(
            captured.exception.code,
            "INSPECTION_IDENTITY_MISMATCH",
        )

        counterfeit_member = replace(
            first,
            source_samp=2,
            well_key="A2_1",
        )
        counterfeit = replace(
            self.inspection,
            accepted_members=(
                counterfeit_member,
                *self.inspection.accepted_members[1:],
            ),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            next(
                iter_sugar_mixtures_records(
                    self.source.raw_root,
                    counterfeit,
                )
            )
        self.assertEqual(
            captured.exception.code,
            "INSPECTION_IDENTITY_MISMATCH",
        )

        counterfeit_member = replace(
            first,
            record_role="pure_reference",
            pure_component="sucrose",
        )
        counterfeit = replace(
            self.inspection,
            accepted_members=(
                counterfeit_member,
                *self.inspection.accepted_members[1:],
            ),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            next(
                iter_sugar_mixtures_records(
                    self.source.raw_root,
                    counterfeit,
                )
            )
        self.assertEqual(
            captured.exception.code,
            "INSPECTION_RECORD_ROLE_MISMATCH",
        )

        counterfeit = replace(
            self.inspection,
            auxiliary_axis_set_id="0" * 64,
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            next(
                iter_sugar_mixtures_records(
                    self.source.raw_root,
                    counterfeit,
                )
            )
        self.assertEqual(
            captured.exception.code,
            "INSPECTION_STALE",
        )

        recipe = self.inspection.recipes["A1_1"]
        volumes = dict(recipe.component_volumes_ul)
        volumes["sucrose"] = 31
        volumes["water"] = 149
        bad_recipe = replace(
            recipe,
            component_volumes_ul=MappingProxyType(volumes),
            component_fractions=MappingProxyType(
                {
                    component: volume / 375
                    for component, volume in volumes.items()
                }
            ),
            concentrations_mol_l=MappingProxyType(
                {
                    f"{component}_nominal_mol_l": (
                        volumes[component] / 375
                    )
                    for component in (
                        "sucrose",
                        "fructose",
                        "maltose",
                        "glucose",
                    )
                }
            ),
        )
        recipes = dict(self.inspection.recipes)
        recipes["A1_1"] = bad_recipe
        counterfeit = replace(
            self.inspection,
            recipes=MappingProxyType(recipes),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            next(
                iter_sugar_mixtures_records(
                    self.source.raw_root,
                    counterfeit,
                )
            )
        self.assertEqual(
            captured.exception.code,
            "WELL_TARGET_CONTRACT_MISMATCH",
        )

    def test_iterator_checks_streaming_target_status_and_precision_contracts(self):
        cases = (
            (
                "record_target_map_sha256",
                "0" * 64,
                "RECORD_TARGET_CONTRACT_MISMATCH",
            ),
            (
                "record_preprocessing_map_sha256",
                "0" * 64,
                "RECORD_PREPROCESSING_CONTRACT_MISMATCH",
            ),
            (
                "axis_float32_max_abs_error_cm1",
                2.0 * self.inspection.source_statistics[
                    "axis_float32_max_abs_error_cm1"
                ],
                "FLOAT32_PRECISION_CONTRACT_MISMATCH",
            ),
        )
        for key, value, code in cases:
            with self.subTest(key=key):
                statistics = dict(self.inspection.source_statistics)
                statistics[key] = value
                inspection = replace(
                    self.inspection,
                    source_statistics=MappingProxyType(statistics),
                )
                with self.assertRaises(
                    SugarMixturesValidationError
                ) as captured:
                    list(
                        iter_sugar_mixtures_records(
                            self.source.raw_root,
                            inspection,
                        )
                    )
                self.assertEqual(captured.exception.code, code)

    def test_preprocessing_evidence_checks_expected_document_digest(self):
        statistics = dict(self.inspection.source_statistics)
        statistics["preprocessing_evidence_document_sha256"] = "0" * 64
        inspection = replace(
            self.inspection,
            source_statistics=MappingProxyType(statistics),
        )

        with self.assertRaises(SugarMixturesValidationError) as captured:
            sugar_mixtures_module._preprocessing_evidence_document(
                self.source.raw_root,
                inspection,
            )
        self.assertEqual(
            captured.exception.code,
            "PREPROCESSING_EVIDENCE_INVALID",
        )

    def test_iterator_rejects_normalized_redundancy_drift(self):
        cases = (
            (
                {"Magnification": 10.0},
                "METADATA_REDUNDANCY_MISMATCH",
            ),
            (
                {"Objective": "not-a-source-objective"},
                "METADATA_REDUNDANCY_MISMATCH",
            ),
            (
                {"Laser power [mW]": 30.3},
                "METADATA_CONDITION_MISMATCH",
            ),
            (
                {"Spectrometer integration time [s]": 0.5},
                "METADATA_CONDITION_MISMATCH",
            ),
            (
                {"Excitation wavelength [nm]": 633},
                "METADATA_REDUNDANCY_MISMATCH",
            ),
            (
                {"Spectrometer number of accumulations": 2},
                "METADATA_REDUNDANCY_MISMATCH",
            ),
        )
        for index, (updates, code) in enumerate(cases):
            with self.subTest(updates=updates):
                source = create_synthetic_sugar_source(
                    self.temporary_root / f"metadata-drift-{index}"
                )
                member = source.canonical_members[0]
                self.mutate_member_metadata(source, member, updates)
                contract = refresh_synthetic_sugar_contract(
                    source,
                    refresh_roles=True,
                )
                inspection = _inspect_sugar_mixtures_source(
                    source.raw_root,
                    contract,
                )
                with self.assertRaises(SugarMixturesValidationError) as captured:
                    next(
                        iter_sugar_mixtures_records(
                            source.raw_root,
                            inspection,
                        )
                    )
                self.assertEqual(captured.exception.code, code)

    def test_synthetic_record_digest_contracts_match_literals(self):
        records = self.records()
        members = {
            member.record_id: member
            for member in self.inspection.accepted_members
        }
        raw_text_digest = hashlib.sha256(
            b"rpe-sugar-acquisition-json-text-v1\0"
        )
        normalized_digest = hashlib.sha256(
            b"rpe-sugar-acquisition-metadata-v1\0"
        )
        core_digest = hashlib.sha256(
            b"rpe-sugar-core-acquisition-metadata-v1\0"
        )
        target_digest = hashlib.sha256(
            b"rpe-sugar-record-target-map-v1\0"
        )
        preprocessing_digest = hashlib.sha256(
            b"rpe-sugar-record-preprocessing-map-v1\0"
        )
        for record in records:
            member = members[record.record_id]
            parsed = _reparse_verified_member(
                self.source.raw_root,
                member,
            )
            path = member.source_member.encode("utf-8")
            raw_text = parsed.source_json_text.encode("utf-8")
            normalized = canonical_json_bytes(
                record.meta.source_metadata[
                    "source_acquisition_metadata"
                ],
                newline=False,
            )
            core = canonical_json_bytes(
                {
                    "instrument": record.meta.instrument,
                    "excitation_nm": record.meta.excitation_nm,
                    "integration_time_s": record.meta.integration_time_s,
                    "n_accumulations": record.meta.n_accumulations,
                    "grating": record.meta.grating,
                    "detector": record.meta.detector,
                },
                newline=False,
            )
            for digest, payload in (
                (raw_text_digest, raw_text),
                (normalized_digest, normalized),
                (core_digest, core),
            ):
                update_length_prefixed(digest, path)
                update_length_prefixed(digest, payload)
            target_entry = canonical_json_bytes(
                {
                    "record_id": record.record_id,
                    "concentrations": dict(record.targets.concentrations),
                },
                newline=True,
            )
            update_length_prefixed(target_digest, target_entry)
            status_entry = canonical_json_bytes(
                {
                    "record_id": record.record_id,
                    "preprocessing_status": (
                        record.meta.preprocessing_status.value
                    ),
                    "preprocessing_steps": [],
                    "eligible_for_preprocessing_evaluation": (
                        record.meta.eligible_for_preprocessing_evaluation
                    ),
                    "source_preprocessing_state": (
                        record.meta.source_metadata[
                            "source_preprocessing_state"
                        ]
                    ),
                    "source_preprocessing_evidence_id": (
                        record.meta.source_metadata[
                            "source_preprocessing_evidence_id"
                        ]
                    ),
                },
                newline=True,
            )
            update_length_prefixed(preprocessing_digest, status_entry)

        self.assertEqual(
            raw_text_digest.hexdigest(),
            "d5b2b47d5ec0785d09c449f56e7575935e3f1f5b8d710d9d1d419fdc2835e3da",
        )
        self.assertEqual(
            normalized_digest.hexdigest(),
            "1e3ab230a2dce7054712d143cdc7262b23b74d35df63e2c557ab60af6dbb47d2",
        )
        self.assertEqual(
            core_digest.hexdigest(),
            "7614044838509ce26cedf5d6de7b74b90d544458dc6e4f7c4ff553e0528839e2",
        )
        self.assertEqual(
            target_digest.hexdigest(),
            "0b9c60bb0fd6cd4e6d13d0f5e57daecad138626892ee001682c440a2deb66727",
        )
        self.assertEqual(
            preprocessing_digest.hexdigest(),
            "30e2d841cc7290fe5decab17302d57b49eccb32ef8bc03193341811179791e5c",
        )
        evidence = sugar_mixtures_module._preprocessing_evidence_document(
            self.source.raw_root,
            self.inspection,
        )
        evidence_digest = hashlib.sha256(
            b"rpe-sugar-preprocessing-evidence-v1\0"
            + canonical_json_bytes(evidence, newline=True)
        ).hexdigest()
        self.assertEqual(
            evidence_digest,
            "af13cffa89ffe8f9041493914029a355857990c552d75a1c4ce22c75bd0f70f3",
        )

    def test_synthetic_well_target_digest_and_cast_maxima_match_literals(self):
        components = ("sucrose", "fructose", "maltose", "glucose", "water")
        well_digest = hashlib.sha256(
            b"rpe-sugar-well-target-contract-v1\0"
        )
        for well_key in sorted(
            self.inspection.recipes,
            key=lambda value: value.encode("utf-8"),
        ):
            recipe = self.inspection.recipes[well_key]
            entry = canonical_json_bytes(
                {
                    "well": well_key,
                    "concentrations": dict(recipe.concentrations_mol_l),
                    "source_component_fractions": dict(
                        recipe.component_fractions
                    ),
                    "source_volumes_ul": {
                        name: recipe.component_volumes_ul[name]
                        for name in components
                    },
                    "source_total_volume_ul": recipe.total_volume_ul,
                },
                newline=True,
            )
            update_length_prefixed(well_digest, entry)
        self.assertEqual(
            well_digest.hexdigest(),
            "f4f612748ffa11a077ed9833ca78391ea8cef0a2df50f0fa3bdc0f927f318b98",
        )

        intensity_error = 0.0
        axis_error = 0.0
        wavelength_error = 0.0
        for member in self.inspection.accepted_members:
            parsed = _reparse_verified_member(
                self.source.raw_root,
                member,
            )
            intensity_error = max(
                intensity_error,
                float(
                    np.max(
                        np.abs(
                            parsed.intensity
                            - parsed.intensity.astype(np.float32).astype(
                                np.float64
                            )
                        )
                    )
                ),
            )
            axis_error = max(
                axis_error,
                float(
                    np.max(
                        np.abs(
                            parsed.wavenumber_cm1
                            - parsed.wavenumber_cm1.astype(np.float32).astype(
                                np.float64
                            )
                        )
                    )
                ),
            )
            wavelength_error = max(
                wavelength_error,
                float(
                    np.max(
                        np.abs(
                            parsed.wavelength_nm
                            - parsed.wavelength_nm.astype(np.float32).astype(
                                np.float64
                            )
                        )
                    )
                ),
            )
        self.assertEqual(intensity_error, 0.0)
        self.assertEqual(axis_error, 1.4110402389633236e-05)
        self.assertEqual(wavelength_error, 0.0)


class SugarMixturesTransparentCompanionTest(unittest.TestCase):
    VIEW_COUNTS = {
        "high_snr": 7,
        "high_snr_no_refs": 2,
        "high_snr_pure_reference": 5,
        "low_snr": 14,
        "low_snr_no_refs": 4,
        "low_snr_pure_reference": 10,
        "no_refs": 6,
        "pure_reference": 15,
    }
    VIEW_DIGESTS = {
        "high_snr": (
            "2567fee1bc2925587f380c7eae7ea0e6f9817b247905130854a368be2f4258c6",
            "2e497280cb2ecb2cd64626bc092e269f519beaa79fc59644546ee673a5d81947",
        ),
        "high_snr_no_refs": (
            "6eda85b985c86daa87918f832c1d1ac1a8f540a63cab21a118185d05853759c2",
            "6b27931d9befbeb7648f519925013dff05558f1c95e7299c4949331a63f4c5a8",
        ),
        "high_snr_pure_reference": (
            "4fbba2408549f1128526174b15a88bc272ab05c8b2aa62b0cb72355410455ecb",
            "7248372f06d8419917c8a42a4014379ba5d32062ec4f01f6af53380eddbdeb94",
        ),
        "low_snr": (
            "0c9b3ddaf62695a7c80dbf620888439c3be8cf1e324458fbe118db36b00bd0c0",
            "4b589849885bbb794f0656ca1099e095344d0608fad7317f35f7c29f305e49a3",
        ),
        "low_snr_no_refs": (
            "c96f323587e90e447ffe5d5c2e57786156f535098eed395ac713a58bbbe7cba6",
            "c5eaa9a562719a8c7a864e5635a0d72ee8bac16c9722a38a455edded03603021",
        ),
        "low_snr_pure_reference": (
            "b15df09dd41f1551d956d61e9f51e0fb37a396a5bc448977046ba09216fd9c6a",
            "d928f29337c3f4cc40887734341b15fe9c3e802ba49dbde2a146db8efa982c38",
        ),
        "no_refs": (
            "b020dece9efe3d2d0026e99be3681b8fbefe4595d131d88a444ac2f7f413fa99",
            "32d6494f6d3c6bf82fbc15ff66a8fc1cf9d086d7839431509db40b3cfe56eaeb",
        ),
        "pure_reference": (
            "cffca4809c2b0e02ae253a2371a649ca378f3adac7dd121a9cc676f016fac475",
            "e433070430d20e2189e2785816e47b4f02568bf05fa8f564ecd905783cd51234",
        ),
    }

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.inspection = _inspect_sugar_mixtures_source(
            self.source.raw_root,
            synthetic_sugar_contract(self.source),
        )
        self.record_ids = tuple(
            member.record_id
            for member in self.inspection.accepted_members
        )
        self.views_path = self.temporary_root / "views.json"
        self.jsonl_path = self.temporary_root / "acquisition.jsonl"

    def write_both(self, root: Path):
        root.mkdir(parents=True)
        views_path = root / "views.json"
        jsonl_path = root / "acquisition.jsonl"
        views = _write_views(views_path, self.inspection)
        acquisition = _write_acquisition_json(
            jsonl_path,
            self.source.raw_root,
            self.inspection,
        )
        return (
            views_path.read_bytes(),
            jsonl_path.read_bytes(),
            views,
            acquisition,
        )

    def mutate_views(self, mutate):
        document = json.loads(self.views_path.read_bytes())
        mutate(document)
        self.views_path.write_bytes(
            canonical_json_bytes(document, newline=True)
        )

    def assert_views_invalid(self, code: str):
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_views(
                self.views_path,
                self.inspection,
                record_ids=set(self.record_ids),
            )
        self.assertEqual(captured.exception.code, code)

    def jsonl_documents(self):
        return [
            json.loads(line)
            for line in self.jsonl_path.read_bytes().splitlines()
        ]

    def write_jsonl_documents(self, documents, *, trailing_newline=True):
        payload = b"".join(
            canonical_json_bytes(document, newline=True)
            for document in documents
        )
        if not trailing_newline:
            payload = payload.rstrip(b"\n")
        self.jsonl_path.write_bytes(payload)

    def assert_jsonl_invalid(self, code: str):
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_acquisition_json(
                self.jsonl_path,
                self.source.raw_root,
                self.inspection,
                record_ids=set(self.record_ids),
            )
        self.assertEqual(captured.exception.code, code)

    def test_views_are_canonical_and_rederived_from_records(self):
        summary = _write_views(self.views_path, self.inspection)
        result = _validate_views(
            self.views_path,
            self.inspection,
            record_ids=set(self.record_ids),
        )
        raw = self.views_path.read_bytes()
        document = json.loads(raw)

        self.assertEqual(summary.path, self.views_path)
        self.assertEqual(summary.artifact_role, "record_views")
        self.assertEqual(summary.bytes, 5556)
        self.assertEqual(
            summary.sha256,
            "6a1511ed1b7461558e77d0e4f584513e496a7aeb15d45fe04a8f3fddd63eca47",
        )
        self.assertIsNone(summary.logical_content_sha256)
        self.assertEqual(summary.item_count, 8)
        self.assertEqual(raw, canonical_json_bytes(document, newline=True))
        self.assertEqual(
            set(document),
            {
                "artifact_role",
                "artifact_schema_version",
                "dataset_id",
                "views",
            },
        )
        self.assertEqual(
            tuple(view["view_id"] for view in document["views"]),
            tuple(sorted(self.VIEW_COUNTS, key=lambda value: value.encode())),
        )
        self.assertEqual(result["view_count"], 8)
        self.assertEqual(result["view_memberships"], 63)
        self.assertEqual(result["view_counts"], self.VIEW_COUNTS)
        for view in document["views"]:
            expected_record, expected_source = self.VIEW_DIGESTS[
                view["view_id"]
            ]
            self.assertEqual(view["record_id_sha256"], expected_record)
            self.assertEqual(
                view["source_member_path_sha256"],
                expected_source,
            )
            self.assertFalse(view["train_test_semantics"])

    def test_views_reject_noncanonical_schema_order_ids_and_semantics(self):
        _write_views(self.views_path, self.inspection)
        raw = self.views_path.read_bytes()
        self.views_path.write_bytes(b'{ "views": [] }\n')
        self.assert_views_invalid("VIEWS_JSON_NONCANONICAL")
        self.views_path.write_bytes(raw)

        cases = (
            (
                lambda document: document.update({"extra": 1}),
                "VIEWS_KEYS_INVALID",
            ),
            (
                lambda document: document["views"][0].update({"extra": 1}),
                "VIEWS_KEYS_INVALID",
            ),
            (
                lambda document: document["views"].reverse(),
                "VIEWS_ORDER_INVALID",
            ),
            (
                lambda document: document["views"][0]["record_ids"].append(
                    document["views"][0]["record_ids"][0]
                ),
                "VIEWS_RECORD_IDS_INVALID",
            ),
            (
                lambda document: document["views"][0]["record_ids"].pop(),
                "VIEWS_RECORD_IDS_INVALID",
            ),
            (
                lambda document: document["views"][0]["record_ids"].append(
                    "unknown-record"
                ),
                "VIEWS_RECORD_IDS_INVALID",
            ),
            (
                lambda document: document["views"][0].update(
                    {"train_test_semantics": True}
                ),
                "VIEWS_SEMANTICS_INVALID",
            ),
        )
        for index, (mutate, code) in enumerate(cases):
            with self.subTest(index=index, code=code):
                self.views_path.write_bytes(raw)
                self.mutate_views(mutate)
                self.assert_views_invalid(code)

    def test_views_reject_selector_count_digests_and_broken_relations(self):
        _write_views(self.views_path, self.inspection)
        raw = self.views_path.read_bytes()
        cases = (
            (
                lambda document: document["views"][0].update(
                    {"selector": {"source_record_role": "mixture"}}
                ),
                "VIEWS_SELECTOR_INVALID",
            ),
            (
                lambda document: document["views"][0].update(
                    {"record_count": 999}
                ),
                "VIEWS_COUNT_INVALID",
            ),
            (
                lambda document: document["views"][0].update(
                    {"record_id_sha256": "0" * 64}
                ),
                "VIEWS_DIGEST_INVALID",
            ),
            (
                lambda document: document["views"][0].update(
                    {"source_member_path_sha256": "0" * 64}
                ),
                "VIEWS_DIGEST_INVALID",
            ),
            (
                lambda document: document["views"][0]["record_ids"].__setitem__(
                    0,
                    next(
                        view
                        for view in document["views"]
                        if view["view_id"] == "low_snr"
                    )["record_ids"][0],
                ),
                "VIEWS_RECORD_IDS_INVALID",
            ),
        )
        for index, (mutate, code) in enumerate(cases):
            with self.subTest(index=index, code=code):
                self.views_path.write_bytes(raw)
                self.mutate_views(mutate)
                self.assert_views_invalid(code)

    def test_views_reject_core_id_drift_and_counterfeit_inspection_cache(self):
        _write_views(self.views_path, self.inspection)
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_views(
                self.views_path,
                self.inspection,
                record_ids={*self.record_ids, "extra-core-record"},
            )
        self.assertEqual(
            captured.exception.code,
            "VIEWS_RECORD_IDS_INVALID",
        )

        cached_views = dict(self.inspection.view_record_ids)
        cached_views["high_snr"] = cached_views["high_snr"][:-1]
        counterfeit = replace(
            self.inspection,
            view_record_ids=MappingProxyType(cached_views),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _write_views(
                self.temporary_root / "counterfeit-views.json",
                counterfeit,
            )
        self.assertEqual(
            captured.exception.code,
            "VIEWS_INSPECTION_MISMATCH",
        )

        first = self.inspection.accepted_members[0]
        counterfeit_members = (
            replace(
                first,
                record_role="pure_reference",
                pure_component="sucrose",
            ),
            *self.inspection.accepted_members[1:],
        )
        counterfeit_view_ids = {}
        counterfeit_view_members = {}
        for view_id in self.inspection.view_record_ids:
            condition = (
                "high_snr"
                if view_id.startswith("high_snr")
                else "low_snr"
                if view_id.startswith("low_snr")
                else None
            )
            role = (
                "pure_reference"
                if view_id.endswith("pure_reference")
                or view_id == "pure_reference"
                else "mixture"
                if view_id.endswith("no_refs")
                or view_id == "no_refs"
                else None
            )
            selected = tuple(
                member
                for member in counterfeit_members
                if (condition is None or member.condition == condition)
                and (role is None or member.record_role == role)
            )
            counterfeit_view_ids[view_id] = tuple(
                member.record_id for member in selected
            )
            counterfeit_view_members[view_id] = tuple(
                sorted(
                    (member.source_member for member in selected),
                    key=lambda value: value.encode("utf-8"),
                )
            )
        counterfeit = replace(
            self.inspection,
            accepted_members=counterfeit_members,
            view_record_ids=MappingProxyType(counterfeit_view_ids),
            view_source_members=MappingProxyType(
                counterfeit_view_members
            ),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _write_views(
                self.temporary_root / "counterfeit-role-views.json",
                counterfeit,
            )
        self.assertEqual(
            captured.exception.code,
            "VIEWS_INSPECTION_MISMATCH",
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_views(
                self.views_path,
                counterfeit,
                record_ids=set(self.record_ids),
            )
        self.assertEqual(
            captured.exception.code,
            "VIEWS_INSPECTION_MISMATCH",
        )

        first = self.inspection.accepted_members[0]
        counterfeit = replace(
            self.inspection,
            accepted_members=(
                replace(
                    first,
                    sample_id="sugar-well-h12-p99",
                ),
                *self.inspection.accepted_members[1:],
            ),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _write_views(
                self.temporary_root / "counterfeit-identity-views.json",
                counterfeit,
            )
        self.assertEqual(
            captured.exception.code,
            "VIEWS_INSPECTION_MISMATCH",
        )

    def test_exact_jsonl_round_trips_source_text(self):
        summary = _write_acquisition_json(
            self.jsonl_path,
            self.source.raw_root,
            self.inspection,
        )
        result = _validate_acquisition_json(
            self.jsonl_path,
            self.source.raw_root,
            self.inspection,
            record_ids=set(self.record_ids),
        )
        lines = self.jsonl_path.read_bytes().splitlines(keepends=True)

        self.assertEqual(summary.path, self.jsonl_path)
        self.assertEqual(summary.artifact_role, "acquisition_json_text")
        self.assertEqual(summary.bytes, 22515)
        self.assertEqual(
            summary.sha256,
            "1c63cffc08de91d44b3935a935edc7bc255091082f17f0b8388113cdde1f6108",
        )
        self.assertIsNone(summary.logical_content_sha256)
        self.assertEqual(summary.item_count, 21)
        self.assertEqual(len(lines), 21)
        self.assertTrue(all(line.endswith(b"\n") for line in lines))
        self.assertEqual(result["lines"], 21)
        self.assertEqual(result["decoded_source_json_bytes"], 14541)
        self.assertEqual(result["minimum_source_json_bytes"], 690)
        self.assertEqual(result["maximum_source_json_bytes"], 694)
        self.assertEqual(result["unique_source_json_texts"], 21)
        self.assertEqual(
            result["raw_text_inventory_sha256"],
            "d5b2b47d5ec0785d09c449f56e7575935e3f1f5b8d710d9d1d419fdc2835e3da",
        )
        for line in lines:
            document = json.loads(line)
            self.assertEqual(
                line,
                canonical_json_bytes(document, newline=True),
            )
            self.assertEqual(
                set(document),
                {
                    "record_id",
                    "source_member",
                    "source_member_sha256",
                    "source_json_text",
                },
            )
            self.assertIn("NaN", document["source_json_text"])

    def test_jsonl_rejects_newline_order_id_hash_and_text_mutations(self):
        _write_acquisition_json(
            self.jsonl_path,
            self.source.raw_root,
            self.inspection,
        )
        original = self.jsonl_path.read_bytes()
        documents = self.jsonl_documents()

        self.write_jsonl_documents(documents, trailing_newline=False)
        self.assert_jsonl_invalid("ACQUISITION_JSONL_NEWLINE_INVALID")

        cases = (
            (
                lambda values: values.reverse(),
                "ACQUISITION_JSONL_ORDER_INVALID",
            ),
            (
                lambda values: values.pop(),
                "ACQUISITION_JSONL_RECORD_IDS_INVALID",
            ),
            (
                lambda values: values.append(dict(values[-1])),
                "ACQUISITION_JSONL_RECORD_IDS_INVALID",
            ),
            (
                lambda values: values[0].update(
                    {"record_id": "unknown-record"}
                ),
                "ACQUISITION_JSONL_RECORD_IDS_INVALID",
            ),
            (
                lambda values: values[0].update(
                    {"source_member_sha256": "0" * 64}
                ),
                "ACQUISITION_JSONL_MEMBER_MISMATCH",
            ),
            (
                lambda values: values[0].update(
                    {"source_json_text": values[0]["source_json_text"] + " "}
                ),
                "ACQUISITION_JSONL_TEXT_MISMATCH",
            ),
            (
                lambda values: values[0].update({"extra": 1}),
                "ACQUISITION_JSONL_KEYS_INVALID",
            ),
        )
        for index, (mutate, code) in enumerate(cases):
            with self.subTest(index=index, code=code):
                self.jsonl_path.write_bytes(original)
                values = self.jsonl_documents()
                mutate(values)
                self.write_jsonl_documents(values)
                self.assert_jsonl_invalid(code)

    def test_jsonl_rejects_parse_reemit_and_source_mutation(self):
        _write_acquisition_json(
            self.jsonl_path,
            self.source.raw_root,
            self.inspection,
        )
        documents = self.jsonl_documents()
        documents[0]["source_json_text"] = json.dumps(
            json.loads(documents[0]["source_json_text"]),
            sort_keys=True,
            allow_nan=True,
        )
        self.write_jsonl_documents(documents)
        self.assert_jsonl_invalid("ACQUISITION_JSONL_TEXT_MISMATCH")

        _write_acquisition_json(
            self.jsonl_path,
            self.source.raw_root,
            self.inspection,
        )
        member = self.inspection.accepted_members[0].source_member
        original_read = zipfile.ZipFile.read

        def drift_member(archive, name, *args, **kwargs):
            payload = original_read(archive, name, *args, **kwargs)
            observed_name = (
                name.filename
                if isinstance(name, zipfile.ZipInfo)
                else name
            )
            if observed_name == member:
                return build_member_payload(record_index=999)
            return payload

        with patch.object(
            zipfile.ZipFile,
            "read",
            autospec=True,
            side_effect=drift_member,
        ):
            self.assert_jsonl_invalid("MEMBER_HASH_MISMATCH")

    def test_jsonl_rejects_inspection_order_and_metadata_digest_drift(self):
        _write_acquisition_json(
            self.jsonl_path,
            self.source.raw_root,
            self.inspection,
        )
        reversed_inspection = replace(
            self.inspection,
            accepted_members=tuple(
                reversed(self.inspection.accepted_members)
            ),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_acquisition_json(
                self.jsonl_path,
                self.source.raw_root,
                reversed_inspection,
                record_ids=set(self.record_ids),
            )
        self.assertEqual(
            captured.exception.code,
            "INSPECTION_IDENTITY_MISMATCH",
        )

        for key in (
            "normalized_metadata_sha256",
            "core_metadata_sha256",
        ):
            with self.subTest(key=key):
                statistics = dict(self.inspection.source_statistics)
                statistics[key] = "0" * 64
                inspection = replace(
                    self.inspection,
                    source_statistics=MappingProxyType(statistics),
                )
                with self.assertRaises(
                    SugarMixturesValidationError
                ) as captured:
                    _validate_acquisition_json(
                        self.jsonl_path,
                        self.source.raw_root,
                        inspection,
                        record_ids=set(self.record_ids),
                    )
                self.assertEqual(
                    captured.exception.code,
                    "ACQUISITION_JSONL_METADATA_MISMATCH",
                )

    def test_jsonl_writer_checks_raw_text_contract_before_publication(self):
        statistics = dict(self.inspection.source_statistics)
        statistics["raw_text_inventory_sha256"] = "0" * 64
        inspection = replace(
            self.inspection,
            source_statistics=MappingProxyType(statistics),
        )

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _write_acquisition_json(
                self.jsonl_path,
                self.source.raw_root,
                inspection,
            )
        self.assertEqual(
            captured.exception.code,
            "ACQUISITION_JSONL_TEXT_MISMATCH",
        )
        self.assertFalse(self.jsonl_path.exists())

    def test_jsonl_writer_and_validator_are_streaming(self):
        real_reparse = (
            __import__(
                "rpe.io.sugar_mixtures_companions",
                fromlist=["_reparse_verified_member"],
            )._reparse_verified_member
        )

        class UnhashableSourceText(str):
            def __hash__(self):
                raise AssertionError(
                    "validator must not retain complete source text strings"
                )

        def unhashable_reparse(*args, **kwargs):
            parsed = real_reparse(*args, **kwargs)
            return replace(
                parsed,
                source_json_text=UnhashableSourceText(
                    parsed.source_json_text
                ),
            )

        with patch.object(
            Path,
            "read_bytes",
            side_effect=AssertionError("whole-file read forbidden"),
        ), patch(
            "rpe.io.sugar_mixtures_companions._reparse_verified_member",
            side_effect=unhashable_reparse,
        ):
            _write_acquisition_json(
                self.jsonl_path,
                self.source.raw_root,
                self.inspection,
            )
            result = _validate_acquisition_json(
                self.jsonl_path,
                self.source.raw_root,
                self.inspection,
                record_ids=set(self.record_ids),
            )
        self.assertEqual(result["lines"], 21)

    def test_independent_transparent_companion_writes_are_byte_identical(self):
        first = self.write_both(self.temporary_root / "first")
        second = self.write_both(self.temporary_root / "second")

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2].sha256, second[2].sha256)
        self.assertEqual(first[3].sha256, second[3].sha256)


class SugarMixturesAuxiliaryAxesTest(unittest.TestCase):
    EXPECTED_LOGICAL_SHA256 = (
        "6daef5ebeadd55cc8875d35b6da52bdce1bbd7f0589dd7df768242556a175a79"
    )
    EXPECTED_PHYSICAL_SHA256 = (
        "5cbabb66bfd538cde7d10e31f54b8dd3a5e788891de05f7d426f6eab2689c4eb"
    )
    EXPECTED_METADATA = {
        "artifact_role": "auxiliary_axes",
        "artifact_schema_version": "1.0.0",
        "dataset_id": "sugar_mixtures_raman",
        "auxiliary_axis_set_id": (
            "9513d0fa3ef818e81f9fa04f2c565f67d0f2b5daeba0bc136e84a1e2eaaae50d"
        ),
        "core_axis_id": (
            "976d06bd9dacdd5665a8b56bd785357587ab0d26e1995b31149edf0004560d7c"
        ),
        "point_count": 16,
        "record_coverage": 21,
        "source_excitation_nm": 785.0,
        "formula": (
            "raman_shift_cm1=1e7/excitation_nm-1e7/wavelength_nm"
        ),
        "source_formula_max_abs_residual_cm1": 4.547473508864641e-13,
        "source_formula_mean_abs_residual_cm1": 2.575717417130363e-13,
        "stored_formula_max_abs_residual_cm1": 1.4110402844380587e-05,
        "stored_formula_mean_abs_residual_cm1": 5.264165338303428e-06,
        "axes": {
            "pixel": {
                "axis_id": (
                    "d26d26a9514f04db2e168195a1153218876c7099d05c10f4fad13cfcc592ee53"
                ),
                "coordinate": "pixel",
                "unit": "detector_pixel_index",
                "source_dtype": "float64",
                "source_value_sha256": (
                    "799eb99a60dd83c57bfe43c1eb5b9e5334fab0ebc120369dee40028729c0004c"
                ),
                "stored_dtype": "uint16",
                "stored_value_sha256": (
                    "64a240d34d0c29ec867f653721a1532de6e665e602e7c03e0b853c9ef3094126"
                ),
                "source_to_stored_max_abs_error": 0.0,
            },
            "wavelength": {
                "axis_id": (
                    "ae112f2edaffa32fb256c5e828e9473ca55ae79084e310d13a9b0cc659cc7122"
                ),
                "coordinate": "wavelength",
                "unit": "nm",
                "source_dtype": "float64",
                "source_value_sha256": (
                    "da0cd1adbf7e6222e2f7e4f1a6c4143c69703d1cc61fb79eeec477dfdc069561"
                ),
                "stored_dtype": "float32",
                "stored_value_sha256": (
                    "94ed93d67d08332a5a790b464cd6b5dcfebfb9fcea0b6e1a2f1c2f95935787b9"
                ),
                "source_to_stored_max_abs_error": 0.0,
            },
        },
    }

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.inspection = _inspect_sugar_mixtures_source(
            self.source.raw_root,
            synthetic_sugar_contract(self.source),
        )
        self.path = self.temporary_root / "auxiliary.h5"

    def write(self):
        return _write_auxiliary_axes(self.path, self.inspection)

    def validate(self):
        return _validate_auxiliary_axes(self.path, self.inspection)

    def assert_auxiliary_invalid(self, code: str):
        with self.assertRaises(SugarMixturesValidationError) as captured:
            self.validate()
        self.assertEqual(captured.exception.code, code)
        self.assertNotEqual(captured.exception.path, "")
        self.assertNotEqual(captured.exception.reason, "")

    def metadata_bytes(self) -> bytes:
        with h5py.File(self.path, "r") as artifact:
            return artifact["metadata_json"][...].tobytes(order="C")

    def write_metadata_bytes(self, payload: bytes) -> None:
        with h5py.File(self.path, "r+") as artifact:
            dataset = artifact["metadata_json"]
            self.assertEqual(len(payload), dataset.shape[0])
            dataset[...] = np.frombuffer(payload, dtype=np.uint8)

    def assert_dataset_contract(
        self,
        dataset,
        *,
        dtype: str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
    ) -> None:
        self.assertEqual(dataset.dtype.str, dtype)
        self.assertEqual(dataset.shape, shape)
        self.assertEqual(dataset.chunks, chunks)
        self.assertEqual(dataset.maxshape, shape)
        self.assertEqual(dataset.compression, "gzip")
        self.assertEqual(dataset.compression_opts, 4)
        self.assertTrue(dataset.shuffle)
        self.assertTrue(dataset.fletcher32)
        self.assertIsNone(dataset.scaleoffset)
        self.assertFalse(dataset.is_virtual)
        creation = dataset.id.get_create_plist()
        self.assertEqual(creation.get_external_count(), 0)
        self.assertEqual(
            {
                creation.get_filter(index)[0]
                for index in range(creation.get_nfilters())
            },
            {
                h5py.h5z.FILTER_DEFLATE,
                h5py.h5z.FILTER_SHUFFLE,
                h5py.h5z.FILTER_FLETCHER32,
            },
        )
        self.assertEqual(set(dataset.attrs), set())

    def test_auxiliary_axes_exact_layout_metadata_and_summary(self):
        summary = self.write()
        result = self.validate()
        expected_metadata = canonical_json_bytes(
            self.EXPECTED_METADATA,
            newline=True,
        )

        self.assertEqual(summary.path, self.path)
        self.assertEqual(summary.artifact_role, "auxiliary_axes")
        self.assertEqual(summary.bytes, 15896)
        self.assertEqual(summary.sha256, self.EXPECTED_PHYSICAL_SHA256)
        self.assertEqual(
            summary.logical_content_sha256,
            self.EXPECTED_LOGICAL_SHA256,
        )
        self.assertEqual(summary.item_count, 1)
        self.assertEqual(len(expected_metadata), 1458)
        self.assertEqual(self.metadata_bytes(), expected_metadata)
        self.assertEqual(
            result["logical_content_sha256"],
            self.EXPECTED_LOGICAL_SHA256,
        )
        self.assertEqual(result["point_count"], 16)
        self.assertEqual(result["record_coverage"], 21)
        self.assertEqual(
            result["auxiliary_axis_set_id"],
            self.inspection.auxiliary_axis_set_id,
        )

        with h5py.File(self.path, "r") as artifact:
            names = []
            artifact.visit(names.append)
            self.assertEqual(
                set(names),
                {
                    "axes",
                    "axes/pixel",
                    "axes/wavelength_nm",
                    "metadata_json",
                },
            )
            self.assertEqual(
                set(artifact.attrs),
                {
                    "artifact_role",
                    "artifact_schema_version",
                    "dataset_id",
                    "logical_content_sha256",
                },
            )
            self.assertEqual(artifact.attrs["artifact_role"], "auxiliary_axes")
            self.assertEqual(
                artifact.attrs["artifact_schema_version"],
                "1.0.0",
            )
            self.assertEqual(
                artifact.attrs["dataset_id"],
                "sugar_mixtures_raman",
            )
            self.assertEqual(
                artifact.attrs["logical_content_sha256"],
                self.EXPECTED_LOGICAL_SHA256,
            )
            self.assertEqual(set(artifact["axes"].attrs), set())
            for name in ("axes", "metadata_json"):
                self.assertIsInstance(
                    artifact.get(name, getlink=True),
                    h5py.HardLink,
                )
            for name in ("pixel", "wavelength_nm"):
                self.assertIsInstance(
                    artifact["axes"].get(name, getlink=True),
                    h5py.HardLink,
                )
            self.assert_dataset_contract(
                artifact["metadata_json"],
                dtype="|u1",
                shape=(1458,),
                chunks=(1458,),
            )
            self.assert_dataset_contract(
                artifact["axes/pixel"],
                dtype="<u2",
                shape=(16,),
                chunks=(16,),
            )
            self.assert_dataset_contract(
                artifact["axes/wavelength_nm"],
                dtype="<f4",
                shape=(16,),
                chunks=(16,),
            )
            np.testing.assert_array_equal(
                artifact["axes/pixel"][...],
                np.arange(16, dtype="<u2"),
            )
            np.testing.assert_array_equal(
                artifact["axes/wavelength_nm"][...],
                np.arange(800.0, 816.0, dtype="<f4"),
            )

    def test_auxiliary_axes_reject_metadata_attribute_object_and_link_drift(self):
        def add_root_attribute():
            with h5py.File(self.path, "r+") as artifact:
                artifact.attrs["extra"] = 1

        def add_group_attribute():
            with h5py.File(self.path, "r+") as artifact:
                artifact["axes"].attrs["extra"] = 1

        def add_root_dataset():
            with h5py.File(self.path, "r+") as artifact:
                artifact.create_dataset(
                    "extra",
                    data=np.array([1], dtype=np.uint8),
                )

        def add_soft_link():
            with h5py.File(self.path, "r+") as artifact:
                artifact["axes"]["alias"] = h5py.SoftLink(
                    "/axes/pixel"
                )

        cases = (
            (
                add_root_attribute,
                "AUXILIARY_STRUCTURE_INVALID",
            ),
            (
                add_group_attribute,
                "AUXILIARY_STRUCTURE_INVALID",
            ),
            (
                add_root_dataset,
                "AUXILIARY_STRUCTURE_INVALID",
            ),
            (
                add_soft_link,
                "AUXILIARY_STRUCTURE_INVALID",
            ),
        )
        for index, (mutate, code) in enumerate(cases):
            with self.subTest(index=index, code=code):
                self.path.unlink(missing_ok=True)
                self.write()
                mutate()
                self.assert_auxiliary_invalid(code)

        self.path.unlink(missing_ok=True)
        self.write()
        original = self.metadata_bytes()
        changed = original.replace(
            b'"record_coverage":21',
            b'"record_coverage":20',
        )
        self.assertNotEqual(changed, original)
        self.write_metadata_bytes(changed)
        self.assert_auxiliary_invalid("AUXILIARY_METADATA_INVALID")

        self.path.unlink(missing_ok=True)
        self.write()
        with h5py.File(self.path, "r+") as artifact:
            artifact.attrs["logical_content_sha256"] = "0" * 64
        self.assert_auxiliary_invalid("AUXILIARY_LOGICAL_DIGEST_INVALID")

    def test_auxiliary_axes_reject_dtype_chunk_filter_and_value_drift(self):
        replacement_cases = (
            (
                "pixel",
                np.arange(16, dtype="<u2"),
                {"chunks": (8,), "compression": "gzip",
                 "compression_opts": 4, "shuffle": True,
                 "fletcher32": True, "track_times": False},
            ),
            (
                "pixel",
                np.arange(16, dtype="<u2"),
                {"chunks": (16,), "track_times": False},
            ),
            (
                "wavelength_nm",
                np.arange(800.0, 816.0, dtype="<f8"),
                {"chunks": (16,), "compression": "gzip",
                 "compression_opts": 4, "shuffle": True,
                 "fletcher32": True, "track_times": False},
            ),
        )
        for name, values, options in replacement_cases:
            with self.subTest(name=name, options=options):
                self.path.unlink(missing_ok=True)
                self.write()
                with h5py.File(self.path, "r+") as artifact:
                    del artifact[f"axes/{name}"]
                    artifact["axes"].create_dataset(
                        name,
                        data=values,
                        **options,
                    )
                self.assert_auxiliary_invalid(
                    "AUXILIARY_STRUCTURE_INVALID"
                )

        for name in ("pixel", "wavelength_nm"):
            with self.subTest(value_drift=name):
                self.path.unlink(missing_ok=True)
                self.write()
                with h5py.File(self.path, "r+") as artifact:
                    dataset = artifact[f"axes/{name}"]
                    values = dataset[...]
                    values[0] += 1
                    dataset[...] = values
                self.assert_auxiliary_invalid("AUXILIARY_AXIS_INVALID")

    def test_auxiliary_axes_reject_creation_order_tracking(self):
        self.write()
        metadata = self.metadata_bytes()
        with h5py.File(self.path, "r") as artifact:
            attributes = {
                name: artifact.attrs[name]
                for name in (
                    "artifact_role",
                    "artifact_schema_version",
                    "dataset_id",
                    "logical_content_sha256",
                )
            }
            pixel = artifact["axes/pixel"][...]
            wavelength = artifact["axes/wavelength_nm"][...]
        self.path.unlink()

        with h5py.File(
            self.path,
            "w",
            libver="earliest",
            track_order=True,
            track_times=False,
        ) as artifact:
            for name, value in attributes.items():
                artifact.attrs[name] = value
            artifact.create_dataset(
                "metadata_json",
                data=np.frombuffer(metadata, dtype=np.uint8),
                chunks=(len(metadata),),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                track_times=False,
            )
            axes = artifact.create_group(
                "axes",
                track_order=False,
                track_times=False,
            )
            axes.create_dataset(
                "pixel",
                data=pixel,
                chunks=pixel.shape,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                track_times=False,
            )
            axes.create_dataset(
                "wavelength_nm",
                data=wavelength,
                chunks=wavelength.shape,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                track_times=False,
            )

        self.assert_auxiliary_invalid("AUXILIARY_STRUCTURE_INVALID")

    def test_auxiliary_axes_reject_counterfeit_inspection_and_source_drift(self):
        self.write()
        statistics = dict(self.inspection.source_statistics)
        statistics["wavelength_float64_sha256"] = "0" * 64
        counterfeit = replace(
            self.inspection,
            source_statistics=MappingProxyType(statistics),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_auxiliary_axes(self.path, counterfeit)
        self.assertEqual(
            captured.exception.code,
            "AUXILIARY_INSPECTION_MISMATCH",
        )

        statistics = dict(self.inspection.source_statistics)
        statistics["axis_groups"] = 2
        counterfeit = replace(
            self.inspection,
            source_statistics=MappingProxyType(statistics),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_auxiliary_axes(self.path, counterfeit)
        self.assertEqual(
            captured.exception.code,
            "AUXILIARY_INSPECTION_MISMATCH",
        )

        original_read = zipfile.ZipFile.read
        source_member = self.inspection.accepted_members[0].source_member

        def drift_source(archive, name, *args, **kwargs):
            payload = original_read(archive, name, *args, **kwargs)
            observed_name = (
                name.filename if isinstance(name, zipfile.ZipInfo) else name
            )
            if observed_name == source_member:
                return build_member_payload(record_index=999)
            return payload

        with patch.object(
            zipfile.ZipFile,
            "read",
            autospec=True,
            side_effect=drift_source,
        ), self.assertRaises(SugarMixturesValidationError) as captured:
            self.validate()
        self.assertEqual(captured.exception.code, "MEMBER_HASH_MISMATCH")

    def test_auxiliary_axes_reject_formula_residual_above_fixed_ceiling(self):
        parsed = _reparse_verified_member(
            self.inspection.raw_root,
            self.inspection.accepted_members[0],
            expected_points=16,
        )
        drifted_wavenumber = np.ascontiguousarray(
            parsed.wavenumber_cm1 + 0.001,
            dtype="<f8",
        )
        stored_wavenumber = np.ascontiguousarray(
            drifted_wavenumber,
            dtype="<f4",
        )
        core_identifier = axis_id(stored_wavenumber)
        auxiliary_digest = hashlib.sha256(
            b"rpe-sugar-aux-axis-set-v1\0"
        )
        for value in (
            core_identifier,
            self.inspection.pixel_axis_id,
            self.inspection.wavelength_axis_id,
        ):
            encoded = value.encode("utf-8")
            auxiliary_digest.update(struct.pack("<Q", len(encoded)))
            auxiliary_digest.update(encoded)
        auxiliary_identifier = auxiliary_digest.hexdigest()
        statistics = dict(self.inspection.source_statistics)
        statistics.update(
            {
                "core_axis_id": core_identifier,
                "auxiliary_axis_set_id": auxiliary_identifier,
                "wavenumber_float64_sha256": hashlib.sha256(
                    drifted_wavenumber.tobytes(order="C")
                ).hexdigest(),
                "axis_float32_max_abs_error_cm1": float(
                    np.max(
                        np.abs(
                            drifted_wavenumber
                            - stored_wavenumber.astype(np.float64)
                        )
                    )
                ),
            }
        )
        counterfeit = replace(
            self.inspection,
            source_statistics=MappingProxyType(statistics),
            core_axis_id=core_identifier,
            auxiliary_axis_set_id=auxiliary_identifier,
        )
        drifted = replace(
            parsed,
            wavenumber_cm1=drifted_wavenumber,
        )

        with patch.object(
            sugar_companions_module,
            "_reparse_verified_member",
            return_value=drifted,
        ), self.assertRaises(SugarMixturesValidationError) as captured:
            _write_auxiliary_axes(
                self.temporary_root / "residual-drift.h5",
                counterfeit,
            )
        self.assertEqual(
            captured.exception.code,
            "AUXILIARY_FORMULA_RESIDUAL_INVALID",
        )

    def test_corrupt_filtered_chunk_raises_read_failure(self):
        self.write()
        corrupt_first_filtered_chunk(self.path, "/axes/wavelength_nm")

        with self.assertRaises(OSError):
            self.validate()

    def test_validator_independently_recomputes_writer_metadata_ids_and_digest(self):
        self.write()

        with patch.object(
            sugar_companions_module,
            "_auxiliary_source_values",
            side_effect=AssertionError("writer expectation builder reused"),
        ), patch.object(
            sugar_companions_module,
            "_axis_value_sha256",
            side_effect=AssertionError("writer value hash reused"),
        ), patch.object(
            sugar_companions_module,
            "_max_abs_error",
            side_effect=AssertionError("writer cast-error helper reused"),
        ), patch.object(
            sugar_companions_module,
            "_stored_axis_id",
            side_effect=AssertionError("source ID builder reused"),
        ), patch.object(
            sugar_companions_module,
            "_auxiliary_axis_set_id",
            side_effect=AssertionError("source set-ID builder reused"),
        ), patch.object(
            sugar_companions_module,
            "_auxiliary_logical_sha256",
            side_effect=AssertionError("writer logical digest reused"),
        ):
            result = self.validate()

        self.assertEqual(
            result["logical_content_sha256"],
            self.EXPECTED_LOGICAL_SHA256,
        )

    def test_writer_calls_pin_order_timestamps_filters_chunks_and_one_call_data(self):
        real_file = h5py.File
        calls = []

        class AttributeProxy:
            def __init__(self, delegate, owner):
                self.delegate = delegate
                self.owner = owner

            def __setitem__(self, key, value):
                calls.append(("attribute", self.owner, key, value))
                self.delegate[key] = value

        class GroupProxy:
            def __init__(self, delegate):
                self.delegate = delegate
                self.attrs = AttributeProxy(delegate.attrs, delegate.name)

            def create_group(self, name, **kwargs):
                calls.append(("group", name, dict(kwargs)))
                return GroupProxy(
                    self.delegate.create_group(name, **kwargs)
                )

            def create_dataset(self, name, **kwargs):
                data = kwargs["data"]
                calls.append(
                    (
                        "dataset",
                        f"{self.delegate.name.rstrip('/')}/{name}",
                        {
                            key: value
                            for key, value in kwargs.items()
                            if key != "data"
                        },
                        data.dtype.str,
                        data.shape,
                        data.flags.c_contiguous,
                    )
                )
                return self.delegate.create_dataset(name, **kwargs)

        class FileProxy(GroupProxy):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.delegate.__exit__(*args)

        def recording_file(*args, **kwargs):
            calls.append(("file", args[1:], dict(kwargs)))
            return FileProxy(real_file(*args, **kwargs))

        with patch.object(
            sugar_companions_module.h5py,
            "File",
            side_effect=recording_file,
        ):
            self.write()

        self.assertEqual(
            calls[0],
            (
                "file",
                ("w",),
                {
                    "libver": "earliest",
                    "track_order": False,
                    "track_times": False,
                },
            ),
        )
        self.assertEqual(
            [
                (entry[1], entry[2])
                for entry in calls
                if entry[0] == "attribute"
            ],
            [
                ("/", "artifact_role"),
                ("/", "artifact_schema_version"),
                ("/", "dataset_id"),
                ("/", "logical_content_sha256"),
            ],
        )
        self.assertEqual(
            [entry for entry in calls if entry[0] == "group"],
            [
                (
                    "group",
                    "axes",
                    {"track_order": False, "track_times": False},
                )
            ],
        )
        datasets = [entry for entry in calls if entry[0] == "dataset"]
        self.assertEqual(
            [entry[1] for entry in datasets],
            [
                "/metadata_json",
                "/axes/pixel",
                "/axes/wavelength_nm",
            ],
        )
        self.assertEqual(
            [(entry[3], entry[4], entry[5]) for entry in datasets],
            [
                ("|u1", (1458,), True),
                ("<u2", (16,), True),
                ("<f4", (16,), True),
            ],
        )
        for entry in datasets:
            self.assertEqual(
                entry[2],
                {
                    "chunks": entry[4],
                    "compression": "gzip",
                    "compression_opts": 4,
                    "shuffle": True,
                    "fletcher32": True,
                    "track_times": False,
                },
            )

    def test_delayed_cross_process_determinism_and_timestamp_negative_control(self):
        production_script = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "tests"))
from sugar_mixtures_helpers import (
    create_synthetic_sugar_source,
    synthetic_sugar_contract,
)
from rpe.io.sugar_mixtures_companions import _write_auxiliary_axes
from rpe.io.sugar_mixtures_source import _inspect_sugar_mixtures_source
source = create_synthetic_sugar_source(root / "source")
inspection = _inspect_sugar_mixtures_source(
    source.raw_root,
    synthetic_sugar_contract(source),
)
summary = _write_auxiliary_axes(root / "auxiliary.h5", inspection)
print(summary.sha256)
"""
        negative_script = """
import sys
from pathlib import Path
import h5py
import numpy as np
path = Path(sys.argv[1])
with h5py.File(
    path,
    "w",
    libver="earliest",
    track_order=False,
    track_times=True,
) as artifact:
    artifact.attrs["role"] = "negative_control"
    group = artifact.create_group(
        "axes",
        track_order=False,
        track_times=True,
    )
    group.create_dataset(
        "pixel",
        data=np.arange(16, dtype="<u2"),
        chunks=(16,),
        compression="gzip",
        compression_opts=4,
        shuffle=True,
        fletcher32=True,
        track_times=True,
    )
"""

        def run(script: str, path: Path) -> str:
            result = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()

        first_root = self.temporary_root / "process-first"
        second_root = self.temporary_root / "process-second"
        first_root.mkdir()
        second_root.mkdir()
        first = run(production_script, first_root)
        time.sleep(1.1)
        second = run(production_script, second_root)
        self.assertEqual(first, self.EXPECTED_PHYSICAL_SHA256)
        self.assertEqual(second, self.EXPECTED_PHYSICAL_SHA256)
        self.assertEqual(
            file_sha256(first_root / "auxiliary.h5"),
            file_sha256(second_root / "auxiliary.h5"),
        )

        negative_first = self.temporary_root / "negative-first.h5"
        negative_second = self.temporary_root / "negative-second.h5"
        run(negative_script, negative_first)
        time.sleep(1.1)
        run(negative_script, negative_second)
        self.assertNotEqual(
            file_sha256(negative_first),
            file_sha256(negative_second),
        )


class SugarMixturesDerivedEndmembersTest(unittest.TestCase):
    CONDITIONS = ("high_snr", "low_snr")
    COMPONENTS = ("sucrose", "fructose", "maltose", "glucose", "water")
    EXPECTED_LOGICAL_SHA256 = (
        "d4607ab5af5fb5943c127858f451d07cae20d3a1c0597d869107a669ff4a7b40"
    )
    EXPECTED_PHYSICAL_SHA256 = (
        "7a93d26fffa83bb115f3edbc0b73d4e681a9e4ad2e9aee533d8ec23911b5021a"
    )
    EXPECTED_RECORD_DIGESTS = {
        "high_snr/sucrose": (
            "7ec8e0869f346082b96780fc30ed31419665e8c7e062cbf5077fc0e08556e061"
        ),
        "high_snr/fructose": (
            "341bdd627f7e638195193aecac200a55da6ab43f800a5d999ca05ed35657b626"
        ),
        "high_snr/maltose": (
            "48da4ba7d887d7e910fcfd8e03de22be9353e61cbd377d28ff5d7e5b6f61bf62"
        ),
        "high_snr/glucose": (
            "bd24aa3eaae1a1173900ffcf7d5ef425f2a68989a51adad313751fe8f480f3ca"
        ),
        "high_snr/water": (
            "1c57162b110ab8273eacb8e8013c5a728f189f8b8c5203d866febcc68626984a"
        ),
        "low_snr/sucrose": (
            "e33398560ffba84be1cad23c6a2b1747a4c09b0ebebc3caf7125ba964a873f68"
        ),
        "low_snr/fructose": (
            "95ec85cb41eb5d4da54a6e6541c7560c1620cb90dbf0f69690b420e2d8761190"
        ),
        "low_snr/maltose": (
            "c2c4d983c6ac1cc96ed26df32bd32eb112165c264b8db0d6f93d28671146c719"
        ),
        "low_snr/glucose": (
            "dd48c16f8bc5d6c133d13e2d3ff57551c9bd1a691c0003af4c64b9a7f9524de9"
        ),
        "low_snr/water": (
            "3ee47913702958c6d4113ce9330949c347a6dee927e78091a881e973d063927a"
        ),
    }
    EXPECTED_SOURCE_MEMBER_DIGESTS = {
        "high_snr/sucrose": (
            "cc931d5e61672bfb533de008d320abaa06a08cb8fc8782e9d36067a4c2d89f20"
        ),
        "high_snr/fructose": (
            "a315f4d179bc542519ef17a2cf66d7e0896935540aa5bbdc78f7ae9e1d947c7e"
        ),
        "high_snr/maltose": (
            "66e9b04a8f9c84416214321ab877fef2409437e34c76302ab40ef85ea6ec5964"
        ),
        "high_snr/glucose": (
            "94c509c5b85b17421cb39408eb0f36397d1bd4c29b5396b434e79e7b4c213ab2"
        ),
        "high_snr/water": (
            "fb894ea5b66f314cab30d89282aa96241b9aeb1f80cd7a12fbc793dc59a2858b"
        ),
        "low_snr/sucrose": (
            "9f89f71efa71254479f29c8e617416d4406ef862a9c179e73b268e97cb9553bd"
        ),
        "low_snr/fructose": (
            "fc7812df09a2ccb94d4f46bcce3642364bc808dc264d0c903829c2b0ac05ba0b"
        ),
        "low_snr/maltose": (
            "85e9852216dfa8a449aee9a5c22eaf8511151a3975ffc75664e116a6cab2c92e"
        ),
        "low_snr/glucose": (
            "8a61bf3ae3915043984f4f5d20b920a66fd98bdae42b0f5486828dd5c58399e8"
        ),
        "low_snr/water": (
            "42266e896b7073d86f892801ac36be83ec60e9f8a558b4f070214a859d21f6aa"
        ),
    }
    EXPECTED_STORED_HASHES = {
        "high_snr/sucrose": (
            "eab59c3a61427839464d15b594505b5315eb8e1618b8fb64ab3f2955b5ae71ee"
        ),
        "high_snr/fructose": (
            "519fbebf47e16191d17ab827e20e9970423b291f9a1a856c5b4001593e173704"
        ),
        "high_snr/maltose": (
            "ab7d6dcb4cb691c451d7ab05b6cb5b4b59dba61dfa400f34034ca83fa121963f"
        ),
        "high_snr/glucose": (
            "5f618bfc7c055353f4911411e5abdef0af92127e82f6b5f3b599a316cbbcc0b0"
        ),
        "high_snr/water": (
            "067ec77143f677cd791dd2a6e17ac3b81f7fd76e3f6cc8d5a42c38d46aeb62f4"
        ),
        "low_snr/sucrose": (
            "cabc47f8ff478074d07b6d209005c2d9c7d7b342c99eb56198afb12cf9ffea2d"
        ),
        "low_snr/fructose": (
            "a3c3b4ab9acd338f21b51f9951a8840d17d3d3e8685cdd98e62b4a7cb0aeaf90"
        ),
        "low_snr/maltose": (
            "ae3bde9a97d0824e5d904f02337bc4af7040dfa22923de3b20a04980274d90ac"
        ),
        "low_snr/glucose": (
            "284c4d577fc9ed3f365e33840f9caf5d34beeead09d01489d32b7181058c40bc"
        ),
        "low_snr/water": (
            "b96ee824df1ec350716a70e408724f3d65d0589477934b8a76091c65df863938"
        ),
    }
    EXPECTED_SOURCE_HASHES = {
        "high_snr/sucrose": (
            "5390f34858cc29083082a78199d563d3e9764f2fa381f4d529ae7f499234544d"
        ),
        "high_snr/fructose": (
            "e8a8b4b270f6efd7071eb37726047746eeeb0cc03cbf0179be3b28c0e74b575e"
        ),
        "high_snr/maltose": (
            "863a65bc3bc173a25c5dd898296cdb5b5c49079ea7fb0cf56a5ea943a73bbeef"
        ),
        "high_snr/glucose": (
            "7b651da48d71e309b1faa60590c8c0840f09b1743fc0ee284dc831e60c0756ce"
        ),
        "high_snr/water": (
            "e8372fa204a219a51b6cfe4e7795f8097520beefc782e4d79c65e2fb873fdfd8"
        ),
        "low_snr/sucrose": (
            "918920cbebc14fca475e45ee5d31b83ecdbb7a1341187b6694016ed7d96add00"
        ),
        "low_snr/fructose": (
            "b2dd9299f25a6632612ba328b944ceb3cfaa13a4e4d4a1e1afd3c1ce9825bc67"
        ),
        "low_snr/maltose": (
            "fdcf737ea024f83b3b01b4f82a8edb0309ba3efe74ec36b3b6b461bf06348697"
        ),
        "low_snr/glucose": (
            "1a2d316bac5f0268bb18350b02554239c27fc8a080b81e9a35ed7d57adc7f9d5"
        ),
        "low_snr/water": (
            "e6806971d4329a9b0f3b3e7f107d16b5f957dbefc3d138691aa93f40d4f0e3f2"
        ),
    }
    EXPECTED_PREPARED_HASHES = {
        "high_snr": (
            "a8f7a67806e350bcbb580f54f43a24b307b8624bde6e3df5a9763450a8a7c7c7"
        ),
        "low_snr": (
            "dcabde4073df4173d486db16bcc95350c9c94a94c37df3854d0d3b296623d33d"
        ),
    }

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.raw_root = self.source.raw_root
        self.inspection = _inspect_sugar_mixtures_source(
            self.raw_root,
            synthetic_sugar_contract(self.source),
        )
        self.record_ids = {
            member.record_id
            for member in self.inspection.accepted_members
        }
        self.path = self.temporary_root / "endmembers.h5"

    def write(self):
        return _write_derived_endmembers(
            self.path,
            self.raw_root,
            self.inspection,
        )

    def validate(self):
        return _validate_derived_endmembers(
            self.path,
            self.raw_root,
            self.inspection,
            record_ids=set(self.record_ids),
        )

    def metadata(self):
        with h5py.File(self.path, "r") as artifact:
            payload = artifact["metadata_json"][...].tobytes(order="C")
        return payload, json.loads(payload)

    def write_metadata(self, document):
        payload = canonical_json_bytes(document, newline=True)
        with h5py.File(self.path, "r+") as artifact:
            dataset = artifact["metadata_json"]
            self.assertEqual(dataset.shape, (len(payload),))
            dataset[...] = np.frombuffer(payload, dtype=np.uint8)

    def assert_invalid(self, code: str):
        with self.assertRaises(SugarMixturesValidationError) as captured:
            self.validate()
        self.assertEqual(captured.exception.code, code)
        self.assertNotEqual(captured.exception.path, "")
        self.assertNotEqual(captured.exception.reason, "")

    def test_endmembers_are_direct_zip_medians_with_exact_lineage_and_layout(self):
        summary = self.write()
        result = self.validate()
        metadata_bytes, document = self.metadata()

        self.assertEqual(summary.path, self.path)
        self.assertEqual(
            summary.artifact_role,
            "derived_reference_endmembers",
        )
        self.assertEqual(summary.bytes, 14283)
        self.assertEqual(summary.sha256, self.EXPECTED_PHYSICAL_SHA256)
        self.assertEqual(
            summary.logical_content_sha256,
            self.EXPECTED_LOGICAL_SHA256,
        )
        self.assertEqual(summary.item_count, 10)
        self.assertEqual(result["endmember_count"], 10)
        self.assertEqual(result["constituent_records"], 15)
        self.assertTrue(result["equals_prepared"])
        self.assertEqual(
            result["high_float32_value_sha256"],
            "92849115f2f2b51104b68d042513d8412b6e14084d8bc051946da90857da9358",
        )
        self.assertEqual(
            result["low_float32_value_sha256"],
            "d474ab37c86fb7651e68b1f0457b2a38895a2fd9cb82235adbb2c5406d727204",
        )
        self.assertEqual(len(metadata_bytes), 8294)
        self.assertEqual(
            hashlib.sha256(metadata_bytes).hexdigest(),
            "6049884c4ec9e85434bac2cfeeeec956f53fe6fd4984f1c705d159dcc1e28266",
        )
        self.assertEqual(
            metadata_bytes,
            canonical_json_bytes(document, newline=True),
        )
        self.assertEqual(
            set(document),
            {
                "artifact_role",
                "artifact_schema_version",
                "dataset_id",
                "core_axis_id",
                "point_count",
                "condition_order",
                "component_order",
                "endmember_count",
                "entries",
                "source_semantic_term",
                "stored_semantic_term",
                "prepared_artifacts",
                "direct_zip_recomputation_equals_prepared",
            },
        )
        self.assertEqual(document["condition_order"], list(self.CONDITIONS))
        self.assertEqual(document["component_order"], list(self.COMPONENTS))
        self.assertEqual(document["source_semantic_term"], "gt_endmembers")
        self.assertEqual(
            document["stored_semantic_term"],
            "derived_reference_endmember",
        )
        self.assertTrue(document["direct_zip_recomputation_equals_prepared"])

        with h5py.File(self.path, "r") as artifact:
            self.assertEqual(set(artifact), {"metadata_json", "intensity"})
            self.assertEqual(
                set(artifact.attrs),
                {
                    "artifact_role",
                    "artifact_schema_version",
                    "dataset_id",
                    "logical_content_sha256",
                },
            )
            metadata_dataset = artifact["metadata_json"]
            intensity = artifact["intensity"]
            self.assertEqual(metadata_dataset.dtype.str, "|u1")
            self.assertEqual(metadata_dataset.shape, (8294,))
            self.assertEqual(metadata_dataset.chunks, (8294,))
            self.assertEqual(intensity.dtype.str, "<f4")
            self.assertEqual(intensity.shape, (2, 5, 16))
            self.assertEqual(intensity.chunks, (1, 1, 16))
            for dataset in (metadata_dataset, intensity):
                self.assertEqual(dataset.maxshape, dataset.shape)
                self.assertEqual(dataset.compression, "gzip")
                self.assertEqual(dataset.compression_opts, 4)
                self.assertTrue(dataset.shuffle)
                self.assertTrue(dataset.fletcher32)
                self.assertIsNone(dataset.scaleoffset)
                self.assertFalse(dataset.is_virtual)
                self.assertEqual(set(dataset.attrs), set())
            np.testing.assert_array_equal(
                intensity[:, :, 0],
                np.array(
                    [
                        [190, 220, 250, 280, 160],
                        [205, 235, 265, 295, 175],
                    ],
                    dtype="<f4",
                ),
            )
            np.testing.assert_array_equal(
                intensity[:, :, -1],
                np.array(
                    [
                        [205, 235, 265, 295, 175],
                        [220, 250, 280, 310, 190],
                    ],
                    dtype="<f4",
                ),
            )

        expected_order = [
            f"{condition}/{component}"
            for condition in self.CONDITIONS
            for component in self.COMPONENTS
        ]
        observed_order = []
        for entry in document["entries"]:
            key = (
                f"{entry['acquisition_condition']}/"
                f"{entry['component']}"
            )
            observed_order.append(key)
            self.assertEqual(
                set(entry),
                {
                    "acquisition_condition",
                    "component",
                    "semantic_role",
                    "derivation",
                    "constituent_count",
                    "constituent_record_ids",
                    "constituent_record_id_sha256",
                    "constituent_source_member_path_sha256",
                    "source_float64_value_sha256",
                    "stored_dtype",
                    "stored_value_sha256",
                    "source_to_stored_max_abs_error",
                },
            )
            self.assertEqual(
                entry["semantic_role"],
                "derived_reference_endmember",
            )
            self.assertNotIn("clean", entry)
            self.assertEqual(entry["derivation"], "pointwise_median")
            self.assertEqual(
                entry["constituent_count"],
                1 if entry["acquisition_condition"] == "high_snr" else 2,
            )
            self.assertEqual(
                entry["constituent_record_id_sha256"],
                self.EXPECTED_RECORD_DIGESTS[key],
            )
            self.assertEqual(
                entry["constituent_source_member_path_sha256"],
                self.EXPECTED_SOURCE_MEMBER_DIGESTS[key],
            )
            self.assertEqual(
                entry["stored_value_sha256"],
                self.EXPECTED_STORED_HASHES[key],
            )
            self.assertEqual(
                entry["source_float64_value_sha256"],
                self.EXPECTED_SOURCE_HASHES[key],
            )
            self.assertEqual(entry["stored_dtype"], "float32")
            self.assertEqual(entry["source_to_stored_max_abs_error"], 0.0)
            self.assertTrue(
                set(entry["constituent_record_ids"]) <= self.record_ids
            )
        self.assertEqual(observed_order, expected_order)

    def test_validator_independently_recomputes_writer_derivation_and_digest(self):
        self.write()

        with patch.object(
            sugar_companions_module,
            "_derive_endmember_content",
            side_effect=AssertionError("writer derivation reused"),
        ), patch.object(
            sugar_companions_module,
            "_endmember_logical_sha256",
            side_effect=AssertionError("writer logical digest reused"),
        ), patch.object(
            sugar_companions_module,
            "_axis_value_sha256",
            side_effect=AssertionError("writer value hash reused"),
        ), patch.object(
            sugar_companions_module,
            "_max_abs_error",
            side_effect=AssertionError("writer cast helper reused"),
        ):
            result = self.validate()

        self.assertEqual(
            result["logical_content_sha256"],
            self.EXPECTED_LOGICAL_SHA256,
        )

    def test_prepared_evidence_uses_verified_zip_bytesio_and_duplicate_hashes(self):
        real_read_pickle = pd.read_pickle
        calls = []

        def read_pickle_bytesio(value, *args, **kwargs):
            self.assertIsInstance(value, io.BytesIO)
            calls.append(value.getvalue())
            return real_read_pickle(value, *args, **kwargs)

        with patch.object(
            sugar_companions_module.pd,
            "read_pickle",
            side_effect=read_pickle_bytesio,
        ), patch.object(
            Path,
            "glob",
            side_effect=AssertionError("extracted cache access forbidden"),
        ):
            self.write()
            result = self.validate()

        self.assertEqual(len(calls), 8)
        self.assertEqual(
            [hashlib.sha256(payload).hexdigest() for payload in calls],
            [
                self.EXPECTED_PREPARED_HASHES["high_snr"],
                self.EXPECTED_PREPARED_HASHES["high_snr"],
                self.EXPECTED_PREPARED_HASHES["low_snr"],
                self.EXPECTED_PREPARED_HASHES["low_snr"],
            ]
            * 2,
        )
        _, document = self.metadata()
        for condition in self.CONDITIONS:
            artifacts = document["prepared_artifacts"][condition]
            self.assertEqual(len(artifacts), 2)
            self.assertEqual(
                {artifact["sha256"] for artifact in artifacts},
                {self.EXPECTED_PREPARED_HASHES[condition]},
            )
            self.assertEqual(
                tuple(artifact["source_member"] for artifact in artifacts),
                (
                    f"{PREPARED_PREFIX}"
                    f"{'High SNR' if condition == 'high_snr' else 'Low SNR'}"
                    "/gt_endmembers.pkl",
                    f"{PREPARED_PREFIX}"
                    f"{'High SNR' if condition == 'high_snr' else 'Low SNR'}"
                    " (no refs)/gt_endmembers.pkl",
                ),
            )
        self.assertTrue(result["equals_prepared"])

    def test_endmembers_reject_metadata_lineage_semantic_and_prepared_drift(self):
        self.write()
        original_bytes, original = self.metadata()
        cases = (
            (
                lambda document: document["condition_order"].reverse(),
                "ENDMEMBER_METADATA_INVALID",
            ),
            (
                lambda document: document["component_order"].reverse(),
                "ENDMEMBER_METADATA_INVALID",
            ),
            (
                lambda document: document.update({"endmember_count": 11}),
                "ENDMEMBER_METADATA_INVALID",
            ),
            (
                lambda document: document["entries"][0].update(
                    {"semantic_role": "x" * len("derived_reference_endmember")}
                ),
                "ENDMEMBER_METADATA_INVALID",
            ),
            (
                lambda document: document["entries"][0].update(
                    {"constituent_count": 2}
                ),
                "ENDMEMBER_LINEAGE_INVALID",
            ),
            (
                lambda document: document["entries"][0][
                    "constituent_record_ids"
                ].__setitem__(
                    0,
                    document["entries"][1]["constituent_record_ids"][0],
                ),
                "ENDMEMBER_LINEAGE_INVALID",
            ),
            (
                lambda document: document["entries"][0].update(
                    {"constituent_record_id_sha256": "0" * 64}
                ),
                "ENDMEMBER_LINEAGE_INVALID",
            ),
            (
                lambda document: document["entries"][0].update(
                    {"constituent_source_member_path_sha256": "0" * 64}
                ),
                "ENDMEMBER_LINEAGE_INVALID",
            ),
            (
                lambda document: document["entries"][0].update(
                    {"stored_value_sha256": "0" * 64}
                ),
                "ENDMEMBER_VALUE_INVALID",
            ),
            (
                lambda document: document["prepared_artifacts"][
                    "high_snr"
                ][0].update({"sha256": "0" * 64}),
                "ENDMEMBER_PREPARED_INVALID",
            ),
            (
                lambda document: document.update(
                    {"direct_zip_recomputation_equals_prepared": None}
                ),
                "ENDMEMBER_PREPARED_INVALID",
            ),
        )
        for index, (mutate, code) in enumerate(cases):
            with self.subTest(index=index, code=code):
                self.write_metadata(original)
                document = json.loads(original_bytes)
                mutate(document)
                self.write_metadata(document)
                self.assert_invalid(code)

    def test_endmembers_reject_structure_filter_and_value_drift(self):
        self.write()
        with h5py.File(self.path, "r+") as artifact:
            artifact.attrs["extra"] = 1
        self.assert_invalid("ENDMEMBER_STRUCTURE_INVALID")

        self.path.unlink()
        self.write()
        with h5py.File(self.path, "r+") as artifact:
            values = artifact["intensity"][...]
            del artifact["intensity"]
            artifact.create_dataset(
                "intensity",
                data=values,
                chunks=(2, 5, 16),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                track_times=False,
            )
        self.assert_invalid("ENDMEMBER_STRUCTURE_INVALID")

        self.path.unlink()
        self.write()
        with h5py.File(self.path, "r+") as artifact:
            values = artifact["intensity"][...]
            values[0, 0, 0] += 1
            artifact["intensity"][...] = values
        self.assert_invalid("ENDMEMBER_VALUE_INVALID")

        self.path.unlink()
        self.write()
        with h5py.File(self.path, "r+") as artifact:
            artifact.attrs["logical_content_sha256"] = "0" * 64
        self.assert_invalid("ENDMEMBER_LOGICAL_DIGEST_INVALID")

    def test_endmembers_reject_creation_order_tracking(self):
        self.write()
        with h5py.File(self.path, "r") as artifact:
            attributes = {
                name: artifact.attrs[name]
                for name in (
                    "artifact_role",
                    "artifact_schema_version",
                    "dataset_id",
                    "logical_content_sha256",
                )
            }
            metadata = artifact["metadata_json"][...]
            intensity = artifact["intensity"][...]
        self.path.unlink()

        with h5py.File(
            self.path,
            "w",
            libver="earliest",
            track_order=True,
            track_times=False,
        ) as artifact:
            for name, value in attributes.items():
                artifact.attrs[name] = value
            artifact.create_dataset(
                "metadata_json",
                data=metadata,
                chunks=metadata.shape,
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                track_times=False,
            )
            artifact.create_dataset(
                "intensity",
                data=intensity,
                chunks=(1, 1, 16),
                compression="gzip",
                compression_opts=4,
                shuffle=True,
                fletcher32=True,
                track_times=False,
            )

        self.assert_invalid("ENDMEMBER_STRUCTURE_INVALID")

    def test_programming_errors_from_pandas_propagate(self):
        with patch.object(
            sugar_companions_module.pd,
            "read_pickle",
            side_effect=TypeError("programming error"),
        ), self.assertRaises(TypeError):
            self.write()
        self.assertFalse(self.path.exists())

    def test_endmembers_reject_core_ids_nonpure_source_and_prepared_mismatch(self):
        self.write()
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_derived_endmembers(
                self.path,
                self.raw_root,
                self.inspection,
                record_ids={*self.record_ids, "extra"},
            )
        self.assertEqual(captured.exception.code, "ENDMEMBER_RECORD_IDS_INVALID")

        first = self.inspection.accepted_members[0]
        counterfeit = replace(
            self.inspection,
            accepted_members=(
                replace(
                    first,
                    record_role="pure_reference",
                    pure_component="sucrose",
                ),
                *self.inspection.accepted_members[1:],
            ),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _write_derived_endmembers(
                self.temporary_root / "counterfeit.h5",
                self.raw_root,
                counterfeit,
            )
        self.assertEqual(
            captured.exception.code,
            "ENDMEMBER_INSPECTION_MISMATCH",
        )

        statistics = dict(self.inspection.source_statistics)
        statistics["high_records"] = statistics["high_records"] - 1
        counterfeit = replace(
            self.inspection,
            source_statistics=MappingProxyType(statistics),
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _write_derived_endmembers(
                self.temporary_root / "count-drift.h5",
                self.raw_root,
                counterfeit,
            )
        self.assertEqual(
            captured.exception.code,
            "ENDMEMBER_INSPECTION_MISMATCH",
        )

        real_read = zipfile.ZipFile.read
        source_member = next(
            member.source_member
            for member in self.inspection.accepted_members
            if member.record_role == "pure_reference"
        )

        def drift_source(archive, name, *args, **kwargs):
            payload = real_read(archive, name, *args, **kwargs)
            observed_name = (
                name.filename if isinstance(name, zipfile.ZipInfo) else name
            )
            if observed_name == source_member:
                return build_member_payload(record_index=999)
            return payload

        with patch.object(
            zipfile.ZipFile,
            "read",
            autospec=True,
            side_effect=drift_source,
        ), self.assertRaises(SugarMixturesValidationError) as captured:
            self.validate()
        self.assertEqual(captured.exception.code, "MEMBER_HASH_MISMATCH")

        real_read_pickle = pd.read_pickle
        changed = False

        def drift_prepared(value, *args, **kwargs):
            nonlocal changed
            prepared = real_read_pickle(value, *args, **kwargs)
            if not changed:
                prepared = np.array(prepared, copy=True)
                prepared[0, 0] += 1
                changed = True
            return prepared

        with patch.object(
            sugar_companions_module.pd,
            "read_pickle",
            side_effect=drift_prepared,
        ), self.assertRaises(SugarMixturesValidationError) as captured:
            _write_derived_endmembers(
                self.temporary_root / "prepared-drift.h5",
                self.raw_root,
                self.inspection,
            )
        self.assertEqual(
            captured.exception.code,
            "ENDMEMBER_PREPARED_INVALID",
        )
        self.assertFalse(
            (self.temporary_root / "prepared-drift.h5").exists()
        )

    def test_endmembers_reject_duplicate_prepared_hash_drift(self):
        source = create_synthetic_sugar_source(
            self.temporary_root / "prepared-hash-drift-source"
        )
        member = (
            f"{PREPARED_PREFIX}"
            "High SNR (no refs)/gt_endmembers.pkl"
        )
        entries = read_synthetic_zip_entries(source.archive_path)

        def change_pickle(entry):
            if entry.name != member:
                return entry
            value = pd.read_pickle(io.BytesIO(entry.payload))
            payload = pickle.dumps(value, protocol=5)
            self.assertNotEqual(payload, entry.payload)
            self.assertTrue(
                np.array_equal(
                    pd.read_pickle(io.BytesIO(payload)),
                    value,
                )
            )
            return replace(entry, payload=payload)

        rewrite_synthetic_zip(
            source,
            lambda _: tuple(change_pickle(entry) for entry in entries),
        )
        contract = refresh_synthetic_sugar_contract(
            source,
            refresh_roles=True,
        )
        inspection = _inspect_sugar_mixtures_source(
            source.raw_root,
            contract,
        )

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _write_derived_endmembers(
                self.temporary_root / "prepared-hash-drift.h5",
                source.raw_root,
                inspection,
            )
        self.assertEqual(
            captured.exception.code,
            "ENDMEMBER_PREPARED_INVALID",
        )

    def test_corrupt_endmember_chunk_raises_read_failure(self):
        self.write()
        corrupt_first_filtered_chunk(self.path, "/intensity")

        with self.assertRaises(OSError):
            self.validate()

    def test_endmember_writer_calls_pin_physical_policy(self):
        real_file = h5py.File
        calls = []

        class AttributeProxy:
            def __init__(self, delegate):
                self.delegate = delegate

            def __setitem__(self, key, value):
                calls.append(("attribute", key))
                self.delegate[key] = value

        class FileProxy:
            def __init__(self, delegate):
                self.delegate = delegate
                self.attrs = AttributeProxy(delegate.attrs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return self.delegate.__exit__(*args)

            def create_dataset(self, name, **kwargs):
                data = kwargs["data"]
                calls.append(
                    (
                        "dataset",
                        name,
                        {
                            key: value
                            for key, value in kwargs.items()
                            if key != "data"
                        },
                        data.dtype.str,
                        data.shape,
                        data.flags.c_contiguous,
                    )
                )
                return self.delegate.create_dataset(name, **kwargs)

        def recording_file(*args, **kwargs):
            calls.append(("file", args[1:], dict(kwargs)))
            return FileProxy(real_file(*args, **kwargs))

        with patch.object(
            sugar_companions_module.h5py,
            "File",
            side_effect=recording_file,
        ):
            self.write()

        self.assertEqual(
            calls[0],
            (
                "file",
                ("w",),
                {
                    "libver": "earliest",
                    "track_order": False,
                    "track_times": False,
                },
            ),
        )
        self.assertEqual(
            [entry for entry in calls if entry[0] == "attribute"],
            [
                ("attribute", "artifact_role"),
                ("attribute", "artifact_schema_version"),
                ("attribute", "dataset_id"),
                ("attribute", "logical_content_sha256"),
            ],
        )
        datasets = [entry for entry in calls if entry[0] == "dataset"]
        self.assertEqual(
            [(entry[1], entry[3], entry[4], entry[5]) for entry in datasets],
            [
                ("metadata_json", "|u1", (8294,), True),
                ("intensity", "<f4", (2, 5, 16), True),
            ],
        )
        self.assertEqual(
            datasets[0][2],
            {
                "chunks": (8294,),
                "compression": "gzip",
                "compression_opts": 4,
                "shuffle": True,
                "fletcher32": True,
                "track_times": False,
            },
        )
        self.assertEqual(
            datasets[1][2],
            {
                "chunks": (1, 1, 16),
                "compression": "gzip",
                "compression_opts": 4,
                "shuffle": True,
                "fletcher32": True,
                "track_times": False,
            },
        )

    def test_delayed_cross_process_endmember_determinism(self):
        script = """
import sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd() / "tests"))
from sugar_mixtures_helpers import (
    create_synthetic_sugar_source,
    synthetic_sugar_contract,
)
from rpe.io.sugar_mixtures_companions import _write_derived_endmembers
from rpe.io.sugar_mixtures_source import _inspect_sugar_mixtures_source
source = create_synthetic_sugar_source(root / "source")
inspection = _inspect_sugar_mixtures_source(
    source.raw_root,
    synthetic_sugar_contract(source),
)
summary = _write_derived_endmembers(
    root / "endmembers.h5",
    source.raw_root,
    inspection,
)
print(summary.sha256)
"""

        def run(root):
            root.mkdir()
            return subprocess.run(
                [sys.executable, "-c", script, str(root)],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()

        first_root = self.temporary_root / "process-first"
        second_root = self.temporary_root / "process-second"
        first = run(first_root)
        time.sleep(1.1)
        second = run(second_root)

        self.assertEqual(first, self.EXPECTED_PHYSICAL_SHA256)
        self.assertEqual(second, self.EXPECTED_PHYSICAL_SHA256)
        self.assertEqual(
            file_sha256(first_root / "endmembers.h5"),
            file_sha256(second_root / "endmembers.h5"),
        )


class SugarMixturesStagedCoreTest(unittest.TestCase):
    EXPECTED_FILES = {
        "SHA256SUMS": (
            235,
            "784fe6ac3c901a8ba7345ad6d6e32867c7bfeb60a95406d1e141d9f48a7b0aed",
        ),
        "SHA256SUMS.sha256": (
            77,
            "449cf2da1abd2a2a556e122274bd878381abcc37360c13dcfd0b37dbf1046318",
        ),
        "arrays.h5": (
            15080,
            "19ed8de1363d5defc210b87ee2c2128ce423d863d81c5abeed86443f229a9d5c",
        ),
        "dataset.json": (
            8349,
            "4d9fdc3c9c29dcb2285d1bd5d943c09c7c732820bbf5ee23e5a036d74107f096",
        ),
        "records.jsonl": (
            64458,
            "fc2a9190f6a48a95aed01cd25b6460b61369ff575240c43ada83c0f64027e738",
        ),
    }
    EXPECTED_MATRIX_SHA256 = (
        "b43bb6cca1b88ca25e68c98ac6b98130ace8501529304209912806001ca49e4b"
    )
    CONCENTRATION_UNITS = {
        "fructose_nominal_mol_l": "mol/L",
        "glucose_nominal_mol_l": "mol/L",
        "maltose_nominal_mol_l": "mol/L",
        "sucrose_nominal_mol_l": "mol/L",
    }

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.raw_root = self.source.raw_root
        self.inspection = _inspect_sugar_mixtures_source(
            self.raw_root,
            synthetic_sugar_contract(self.source),
        )
        self.output_root = self.temporary_root / "output"
        self.output_root.mkdir()
        self.sentinel = self.output_root / "sentinel.txt"
        self.sentinel.write_text("unchanged\n", encoding="utf-8")
        self.core_path = self.temporary_root / "staging" / "sugar_mixtures_raman"
        self.source_sha256 = file_sha256(self.source.archive_path)

    def build(self, path: Path | None = None):
        return _build_sugar_core_staged(
            self.raw_root,
            self.core_path if path is None else path,
            self.inspection,
        )

    def compare(self, path: Path | None = None):
        return _compare_sugar_dataset_to_source(
            self.raw_root,
            self.core_path if path is None else path,
            self.inspection,
        )

    def documents(self, path: Path | None = None):
        current = self.core_path if path is None else path
        return [
            json.loads(line)
            for line in (current / "records.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    def write_documents(self, documents, path: Path | None = None):
        current = self.core_path if path is None else path
        (current / "records.jsonl").write_bytes(
            b"".join(
                canonical_json_bytes(document, newline=True)
                for document in documents
            )
        )

    def manifest(self, path: Path | None = None):
        current = self.core_path if path is None else path
        return json.loads((current / "dataset.json").read_bytes())

    def write_manifest(self, document, path: Path | None = None):
        current = self.core_path if path is None else path
        (current / "dataset.json").write_bytes(
            canonical_json_bytes(document, newline=True)
        )

    def assert_comparison_invalid(self, expected_path: str):
        with self.assertRaises(SugarMixturesValidationError) as captured:
            self.compare()
        self.assertEqual(
            captured.exception.code,
            "SOURCE_COMPARISON_MISMATCH",
        )
        self.assertEqual(captured.exception.path, expected_path)

    def test_core_staging_writes_exact_valid_five_file_dataset(self):
        summary = self.build()
        validated = validate_dataset(self.core_path)
        manifest = self.manifest()
        documents = self.documents()

        self.assertIsInstance(summary, SugarMixturesDatasetSummary)
        self.assertEqual(summary.path, self.core_path)
        self.assertEqual(summary.dataset_id, "sugar_mixtures_raman")
        self.assertEqual(summary.record_count, 21)
        self.assertEqual(summary.eligible_records, 21)
        self.assertEqual(summary.axis_group_count, 1)
        self.assertEqual(summary.output_bytes, 88199)
        self.assertEqual(set(summary.files), set(DATASET_FILES))
        for name, (expected_bytes, expected_sha256) in self.EXPECTED_FILES.items():
            self.assertEqual(
                summary.files[name],
                {"bytes": expected_bytes, "sha256": expected_sha256},
            )
        self.assertEqual(set(self.core_path.iterdir()), {
            self.core_path / name for name in DATASET_FILES
        })
        self.assertEqual(validated.record_count, 21)
        self.assertEqual(validated.axis_group_count, 1)
        self.assertEqual(
            dict(validated.preprocessing_status_counts),
            {"known_raw": 21, "known_corrected": 0, "unknown": 0},
        )
        self.assertEqual(
            dict(validated.target_presence_counts),
            {
                "baseline": 0,
                "class_label": 0,
                "clean": 0,
                "concentration": 0,
                "concentrations": 21,
                "peaks": 0,
            },
        )
        self.assertEqual(manifest["class_labels"], {})
        self.assertIsNone(manifest["concentration_unit"])
        self.assertEqual(
            manifest["concentration_units"],
            self.CONCENTRATION_UNITS,
        )
        self.assertEqual(len(manifest["source_artifacts"]), 21)
        self.assertEqual(
            manifest["axis_groups"],
            [
                {
                    "axis_id": self.inspection.core_axis_id,
                    "length": 16,
                    "record_count": 21,
                }
            ],
        )
        self.assertEqual(
            [document["record_id"] for document in documents],
            sorted(self.source.record_ids),
        )
        self.assertEqual(
            [document["array_ref"]["row"] for document in documents],
            list(range(21)),
        )
        self.assertTrue(
            all(
                document["targets"]["concentration"] is None
                and document["targets"]["concentrations"] is not None
                and set(document["targets"]["concentrations"])
                == set(self.CONCENTRATION_UNITS)
                for document in documents
            )
        )
        with h5py.File(self.core_path / "arrays.h5", "r") as artifact:
            group = artifact["axes"][self.inspection.core_axis_id]
            self.assertEqual(set(group), {"wavenumber", "intensity"})
            self.assertEqual(group["intensity"].shape, (21, 16))
            self.assertEqual(group["intensity"].chunks, (21, 16))
            self.assertEqual(
                hashlib.sha256(
                    group["intensity"][...].astype(
                        "<f4",
                        copy=False,
                    ).tobytes(order="C")
                ).hexdigest(),
                self.EXPECTED_MATRIX_SHA256,
            )
        self.assertEqual(self.sentinel.read_text(), "unchanged\n")
        self.assertEqual(file_sha256(self.source.archive_path), self.source_sha256)
        self.assertFalse(any(self.output_root.glob(".sugar*")))

    def test_exhaustive_comparison_counts_all_source_representations(self):
        self.build()
        summary = self.compare()

        self.assertEqual(summary.source_rows_compared, 21)
        self.assertEqual(summary.source_points_compared, 336)
        self.assertEqual(summary.consolidated_intensity_cells_compared, 336)
        self.assertEqual(summary.prepared_abundance_rows_compared, 27)
        self.assertEqual(summary.prepared_endmembers_compared, 10)
        for name in (
            "intensity_mismatches",
            "axis_mismatches",
            "target_mismatches",
            "metadata_mismatches",
            "preprocessing_status_mismatches",
            "companion_reference_mismatches",
        ):
            self.assertEqual(getattr(summary, name), 0)

    def test_comparator_rejects_counterfeit_comparison_count_contract(self):
        self.build()
        statistics = dict(self.inspection.source_statistics)
        statistics["record_count"] = statistics["record_count"] + 1
        counterfeit = replace(
            self.inspection,
            source_statistics=MappingProxyType(statistics),
        )

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _compare_sugar_dataset_to_source(
                self.raw_root,
                self.core_path,
                counterfeit,
            )
        self.assertEqual(
            captured.exception.code,
            "SOURCE_COMPARISON_MISMATCH",
        )
        self.assertEqual(
            captured.exception.path,
            "comparison.counts",
        )

    def test_comparator_propagates_pandas_programming_errors(self):
        self.build()
        with patch.object(
            sugar_mixtures_module.pd,
            "read_pickle",
            side_effect=TypeError("programming error"),
        ), self.assertRaises(TypeError):
            self.compare()

    def test_comparator_rejects_array_axis_identity_metadata_target_and_status_drift(self):
        self.build()
        original_records = self.documents()

        with h5py.File(self.core_path / "arrays.h5", "r+") as artifact:
            dataset = artifact["axes"][self.inspection.core_axis_id]["intensity"]
            dataset[0, 0] += 1
        self.assert_comparison_invalid(
            f"comparison.records.{original_records[0]['record_id']}.intensity"
        )

        shutil.rmtree(self.core_path, ignore_errors=True)
        self.build()
        with h5py.File(self.core_path / "arrays.h5", "r+") as artifact:
            dataset = artifact["axes"][self.inspection.core_axis_id][
                "wavenumber"
            ]
            dataset[0] += 1
        self.assert_comparison_invalid("comparison.dataset.arrays.h5")

        cases = (
            (
                lambda document: document["meta"]["source_metadata"].update(
                    {"source_samp": 999}
                ),
                "comparison.records.{}.identity",
            ),
            (
                lambda document: document["meta"]["source_metadata"][
                    "source_acquisition_metadata"
                ].update({"laser_power_mw": 999.0}),
                "comparison.records.{}.metadata",
            ),
            (
                lambda document: document["meta"]["source_metadata"].update(
                    {"source_sucrose_volume_ul": 999}
                ),
                "comparison.records.{}.recipe",
            ),
            (
                lambda document: document["targets"][
                    "concentrations"
                ].update({"sucrose_nominal_mol_l": 999.0}),
                "comparison.records.{}.targets",
            ),
            (
                lambda document: document["meta"].update(
                    {"preprocessing_status": "unknown"}
                ),
                "comparison.records.{}.preprocessing",
            ),
        )
        for index, (mutate, expected_path) in enumerate(cases):
            with self.subTest(index=index):
                shutil.rmtree(self.core_path)
                self.build()
                documents = self.documents()
                mutate(documents[0])
                self.write_documents(documents)
                self.assert_comparison_invalid(
                    expected_path.format(documents[0]["record_id"])
                )

    def test_comparator_rejects_provenance_manifest_and_oracle_drift(self):
        self.build()
        documents = self.documents()
        documents[0]["provenance"]["license"] = "wrong license"
        self.write_documents(documents)
        self.assert_comparison_invalid(
            f"comparison.records.{documents[0]['record_id']}.provenance"
        )

        shutil.rmtree(self.core_path)
        self.build()
        manifest = self.manifest()
        manifest["source_artifacts"][0]["license"] = "wrong license"
        self.write_manifest(manifest)
        self.assert_comparison_invalid(
            "comparison.dataset.source_artifacts"
        )

        shutil.rmtree(self.core_path)
        self.build()
        manifest = self.manifest()
        manifest["concentration_units"]["sucrose_nominal_mol_l"] = "mmol/L"
        self.write_manifest(manifest)
        self.assert_comparison_invalid(
            "comparison.dataset.concentration_units"
        )

        oracle_source = create_synthetic_sugar_source(
            self.temporary_root / "oracle-drift-source"
        )
        oracle_entries = read_synthetic_zip_entries(
            oracle_source.archive_path
        )

        def change_oracle(entry):
            if entry.name != HIGH_SPECTRA_ORACLE:
                return entry
            rows = list(
                csv.reader(
                    io.StringIO(entry.payload.decode("utf-8"))
                )
            )
            rows[1][1] = str(float(rows[1][1]) + 1.0)
            return replace(
                entry,
                payload=payload_from_rows(rows),
            )

        rewrite_synthetic_zip(
            oracle_source,
            lambda _: tuple(
                change_oracle(entry)
                for entry in oracle_entries
            ),
        )
        oracle_contract = refresh_synthetic_sugar_contract(
            oracle_source,
            refresh_roles=True,
        )
        oracle_inspection = _inspect_sugar_mixtures_source(
            oracle_source.raw_root,
            oracle_contract,
        )
        oracle_core = (
            self.temporary_root
            / "oracle-drift-staging"
            / "sugar_mixtures_raman"
        )
        _build_sugar_core_staged(
            oracle_source.raw_root,
            oracle_core,
            oracle_inspection,
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _compare_sugar_dataset_to_source(
                oracle_source.raw_root,
                oracle_core,
                oracle_inspection,
            )
        self.assertEqual(
            captured.exception.code,
            "SOURCE_COMPARISON_MISMATCH",
        )
        self.assertEqual(
            captured.exception.path,
            "comparison.oracle.high_snr",
        )

        shutil.rmtree(self.core_path)
        self.build()
        real_read_pickle = pd.read_pickle
        changed = False

        def drift_abundance(value, *args, **kwargs):
            nonlocal changed
            result = real_read_pickle(value, *args, **kwargs)
            if (
                not changed
                and isinstance(result, np.ndarray)
                and result.ndim == 2
                and result.shape[1] == 5
                and result.shape[0] not in {5}
            ):
                result = np.array(result, copy=True)
                result[0, 0] += 1.0
                changed = True
            return result

        with patch.object(
            sugar_mixtures_module.pd,
            "read_pickle",
            side_effect=drift_abundance,
        ):
            self.assert_comparison_invalid(
                "comparison.prepared.High SNR.abundance"
            )

    def test_comparator_rejects_prepared_view_membership_drift(self):
        source = create_synthetic_sugar_source(
            self.temporary_root / "prepared-membership-drift-source"
        )
        entries = read_synthetic_zip_entries(source.archive_path)
        view = "High SNR (no refs)"
        prefix = f"{PREPARED_PREFIX}{view}/"
        payload_by_name = {entry.name: entry.payload for entry in entries}
        metadata_lines = payload_by_name[
            f"{prefix}metadata.csv"
        ].splitlines(keepends=True)
        data = pd.read_pickle(
            io.BytesIO(payload_by_name[f"{prefix}data.pkl"])
        )
        abundance = pd.read_pickle(
            io.BytesIO(
                payload_by_name[f"{prefix}gt_abundance_image.pkl"]
            )
        )
        data = np.asarray(data)[:-1]
        abundance = np.asarray(abundance)[:-1]
        metadata_payload = b"".join(metadata_lines[:-1])
        replacements = {
            f"{prefix}metadata.csv": metadata_payload,
            f"{prefix}data.pkl": pickle.dumps(data, protocol=4),
            f"{prefix}gt_abundance_image.pkl": pickle.dumps(
                abundance,
                protocol=4,
            ),
        }
        rewrite_synthetic_zip(
            source,
            lambda _: tuple(
                replace(entry, payload=replacements[entry.name])
                if entry.name in replacements
                else entry
                for entry in entries
            ),
        )
        contract = refresh_synthetic_sugar_contract(
            source,
            refresh_roles=True,
        )
        inspection = _inspect_sugar_mixtures_source(
            source.raw_root,
            contract,
        )
        core = (
            self.temporary_root
            / "prepared-membership-drift-staging"
            / "sugar_mixtures_raman"
        )
        _build_sugar_core_staged(source.raw_root, core, inspection)

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _compare_sugar_dataset_to_source(
                source.raw_root,
                core,
                inspection,
            )
        self.assertEqual(
            captured.exception.code,
            "SOURCE_COMPARISON_MISMATCH",
        )
        self.assertEqual(
            captured.exception.path,
            "comparison.prepared.High SNR (no refs).membership",
        )

    def test_comparator_reads_hdf5_one_record_row_at_a_time(self):
        self.build()
        real_file = h5py.File
        reads = []

        class DatasetProxy:
            def __init__(self, delegate, name):
                self.delegate = delegate
                self.name = name

            def __getitem__(self, key):
                reads.append((self.name, key))
                if self.name.endswith("/intensity"):
                    self.assert_row_key(key)
                return self.delegate[key]

            @staticmethod
            def assert_row_key(key):
                if isinstance(key, tuple):
                    key = key[0]
                if not isinstance(key, (int, np.integer)):
                    raise AssertionError("full intensity matrix read forbidden")

            def __getattr__(self, name):
                return getattr(self.delegate, name)

        class GroupProxy:
            def __init__(self, delegate):
                self.delegate = delegate

            def __getitem__(self, key):
                value = self.delegate[key]
                if isinstance(value, h5py.Dataset):
                    return DatasetProxy(value, value.name)
                if isinstance(value, h5py.Group):
                    return GroupProxy(value)
                return value

            def __getattr__(self, name):
                return getattr(self.delegate, name)

        class FileProxy(GroupProxy):
            def __enter__(self):
                self.delegate.__enter__()
                return self

            def __exit__(self, *args):
                return self.delegate.__exit__(*args)

        with patch.object(
            sugar_mixtures_module.h5py,
            "File",
            side_effect=lambda *args, **kwargs: FileProxy(
                real_file(*args, **kwargs)
            ),
        ):
            summary = self.compare()
        self.assertEqual(summary.source_rows_compared, 21)
        intensity_reads = [
            key for name, key in reads if name.endswith("/intensity")
        ]
        self.assertEqual(len(intensity_reads), 21)
        axis_reads = [
            key for name, key in reads if name.endswith("/wavenumber")
        ]
        self.assertEqual(axis_reads, [slice(None, None, None)])

    def test_comparator_releases_complete_parsed_source_records(self):
        self.build()
        real_reparse = sugar_mixtures_module._reparse_verified_member
        parsed_references = []

        def tracked_reparse(*args, **kwargs):
            self.assertTrue(
                all(reference() is None for reference in parsed_references),
                "complete parsed source records must be released per row",
            )
            parsed = real_reparse(*args, **kwargs)
            parsed_references.append(weakref.ref(parsed))
            return parsed

        with patch.object(
            sugar_mixtures_module,
            "_reparse_verified_member",
            side_effect=tracked_reparse,
        ):
            summary = self.compare()

        self.assertEqual(summary.source_rows_compared, 21)
        self.assertEqual(len(parsed_references), 21)
        self.assertTrue(
            all(reference() is None for reference in parsed_references)
        )

    def test_independent_core_builds_are_five_file_deterministic(self):
        first = self.temporary_root / "first" / "sugar_mixtures_raman"
        second = self.temporary_root / "second" / "sugar_mixtures_raman"
        first_summary = self.build(first)
        second_summary = self.build(second)

        self.assertEqual(
            dict(first_summary.files),
            dict(second_summary.files),
        )
        self.assertEqual(
            {
                name: file_sha256(first / name)
                for name in DATASET_FILES
            },
            {
                name: file_sha256(second / name)
                for name in DATASET_FILES
            },
        )
        with h5py.File(first / "arrays.h5", "r") as first_hdf5, h5py.File(
            second / "arrays.h5",
            "r",
        ) as second_hdf5:
            first_matrix = first_hdf5["axes"][self.inspection.core_axis_id][
                "intensity"
            ][...]
            second_matrix = second_hdf5["axes"][self.inspection.core_axis_id][
                "intensity"
            ][...]
        np.testing.assert_array_equal(first_matrix, second_matrix)
        self.assertEqual(
            hashlib.sha256(first_matrix.tobytes(order="C")).hexdigest(),
            self.EXPECTED_MATRIX_SHA256,
        )


class SugarMixturesReceiptTest(unittest.TestCase):
    RECEIPT_KEYS = {
        "receipt_schema_version",
        "adapter_version",
        "schema_version",
        "dataset_id",
        "source_contract",
        "source_contract_sha256",
        "semantic_contract",
        "semantic_contract_sha256",
        "preprocessing_evidence",
        "license",
        "limitations",
        "policy_contract_sha256",
        "outputs",
    }
    SOURCE_KEYS = {
        "release",
        "archive",
        "central_directory_safety",
        "inventory_counts",
        "member_roles",
        "canonical_members",
        "relevant_evidence_members",
        "evidence_snapshots",
        "excluded_inventory",
    }
    SEMANTIC_KEYS = {
        "dataset",
        "provenance",
        "targets",
        "preprocessing",
        "precision",
        "views",
        "derived_endmembers",
        "auxiliary_axes",
        "acquisition_json",
        "comparison",
    }
    LIMITATIONS = {
        "adapter_defined_train_test_split": False,
        "condition_difference_not_integration_time_only": True,
        "croissant_included": False,
        "derived_endmembers_not_clean_targets": True,
        "high_low_not_pointwise_pairs": True,
        "high_snr_not_clean_target": True,
        "instrument_internal_operations_unverified": True,
        "local_timestamp_timezone_unknown": True,
        "lod_loq_not_in_adapter": True,
        "nominal_concentrations_not_assay_measured": True,
        "normalization_applied": False,
        "physical_clean_ground_truth_present": False,
        "prepared_views_not_canonical_records": True,
        "public_bundle_review_required": True,
        "resampling_applied": False,
        "sample_age_and_temporal_effects_present": True,
        "source_declared_unprocessed_not_detector_native": True,
        "source_nonfinite_metadata_tagged": True,
        "statistical_preprocessing_probe_not_in_adapter": True,
        "synthetic_subtree_excluded": True,
        "synthetic_subtree_excluded_members": 1,
        "water_not_analyte_target": True,
    }
    OUTPUT_NAMES = {
        "sugar_mixtures_raman",
        "sugar_mixtures_raman_views.json",
        "sugar_mixtures_raman_derived_reference_endmembers.h5",
        "sugar_mixtures_raman_auxiliary_axes.h5",
        "sugar_mixtures_raman_acquisition_json_text.jsonl",
    }
    ARTIFACT_NAMES = OUTPUT_NAMES | {
        "sugar_mixtures_raman_conversion.json"
    }
    EXPECTED_NON_SELF_HASHES = {
        "sugar_mixtures_raman/SHA256SUMS": (
            "784fe6ac3c901a8ba7345ad6d6e32867c7bfeb60a95406d1e141d9f48a7b0aed"
        ),
        "sugar_mixtures_raman/SHA256SUMS.sha256": (
            "449cf2da1abd2a2a556e122274bd878381abcc37360c13dcfd0b37dbf1046318"
        ),
        "sugar_mixtures_raman/arrays.h5": (
            "19ed8de1363d5defc210b87ee2c2128ce423d863d81c5abeed86443f229a9d5c"
        ),
        "sugar_mixtures_raman/dataset.json": (
            "4d9fdc3c9c29dcb2285d1bd5d943c09c7c732820bbf5ee23e5a036d74107f096"
        ),
        "sugar_mixtures_raman/records.jsonl": (
            "fc2a9190f6a48a95aed01cd25b6460b61369ff575240c43ada83c0f64027e738"
        ),
        "sugar_mixtures_raman_views.json": (
            "6a1511ed1b7461558e77d0e4f584513e496a7aeb15d45fe04a8f3fddd63eca47"
        ),
        "sugar_mixtures_raman_derived_reference_endmembers.h5": (
            "7a93d26fffa83bb115f3edbc0b73d4e681a9e4ad2e9aee533d8ec23911b5021a"
        ),
        "sugar_mixtures_raman_auxiliary_axes.h5": (
            "5cbabb66bfd538cde7d10e31f54b8dd3a5e788891de05f7d426f6eab2689c4eb"
        ),
        "sugar_mixtures_raman_acquisition_json_text.jsonl": (
            "1c63cffc08de91d44b3935a935edc7bc255091082f17f0b8388113cdde1f6108"
        ),
    }
    EXPECTED_CONTRACTS = {
        "source": (
            21743,
            "70af4a138d4ca700d8e4f6d5c1f5d88531446119885c2e42511c6135669d3cea",
        ),
        "semantic": (
            4054,
            "fe6ae562325f84e3221f304be4b9d2764ca2dbf68c92df6f47edb539f3c4876b",
        ),
        "policy": (
            2648,
            "b6a68fad3810288b546b2fa9c510b5766a322a26e7c307f4058cef6d250128c0",
        ),
        "receipt": (
            30353,
            "20e9d718f68d7289cdb21ab84faf326c594bfbaa5596f75498bc639a6c1952e9",
        ),
    }
    APPENDIX_WITNESSES = {
        "A": (
            6413,
            b"rpe-sugar-publication-transaction-v1\0",
            "6f31dcdded143ac7b57c48eba4e3af4cf9880e99479fb253b35814b85fabc589",
        ),
        "B": (
            7927,
            b"rpe-sugar-cli-contract-v1\0",
            "94e747c18c4af1e187550443d211199afa0da7408df03417802ea38f78814b54",
        ),
        "C": (
            7719,
            b"rpe-sugar-public-package-contract-v1\0",
            "e2f1138f5ca3f6e5115627f6d059eca71dabf4ffd6917e947c9507bacdef4b61",
        ),
        "D": (
            5149,
            b"rpe-sugar-hdf5-physical-contract-v1\0",
            "ec9facfb64ffec4f786ceac90f76f4fcf85150564a41d16f8b50ed1288efe42e",
        ),
        "E": (
            10855,
            b"rpe-preprocessing-state-audit-policy-v1\0",
            "7e48d875a8b2a4eb29d40e811a1f7a56c3949603b11ccaaec140649944cb0ca5",
        ),
    }

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.raw_root = self.source.raw_root
        self.inspection = _inspect_sugar_mixtures_source(
            self.raw_root,
            synthetic_sugar_contract(self.source),
        )
        self.staging = self.temporary_root / "staging"
        self.output_root = self.temporary_root / "output"
        self.output_root.mkdir()
        self.sentinel = self.output_root / "sentinel.txt"
        self.sentinel.write_bytes(b"unchanged\n")
        self.source_sha256 = file_sha256(self.source.archive_path)

    @staticmethod
    def companion_paths(staging: Path):
        return {
            "views": staging / "sugar_mixtures_raman_views.json",
            "derived_endmembers": (
                staging
                / "sugar_mixtures_raman_derived_reference_endmembers.h5"
            ),
            "auxiliary_axes": (
                staging / "sugar_mixtures_raman_auxiliary_axes.h5"
            ),
            "acquisition_json": (
                staging
                / "sugar_mixtures_raman_acquisition_json_text.jsonl"
            ),
        }

    def build_without_receipt(self, staging: Path | None = None):
        current = self.staging if staging is None else staging
        current.mkdir(parents=True)
        core = _build_sugar_core_staged(
            self.raw_root,
            current / "sugar_mixtures_raman",
            self.inspection,
        )
        paths = self.companion_paths(current)
        companions = {
            "views": _write_views(paths["views"], self.inspection),
            "derived_endmembers": _write_derived_endmembers(
                paths["derived_endmembers"],
                self.raw_root,
                self.inspection,
            ),
            "auxiliary_axes": _write_auxiliary_axes(
                paths["auxiliary_axes"],
                self.inspection,
            ),
            "acquisition_json": _write_acquisition_json(
                paths["acquisition_json"],
                self.raw_root,
                self.inspection,
            ),
        }
        comparison = _compare_sugar_dataset_to_source(
            self.raw_root,
            core.path,
            self.inspection,
        )
        return current, core, companions, comparison

    def build_document(self, staging: Path | None = None):
        current, core, companions, comparison = self.build_without_receipt(
            staging
        )
        document = _build_receipt_document(
            raw_root=self.raw_root,
            inspection=self.inspection,
            core=core,
            companions=companions,
            comparison=comparison,
        )
        return current, core, companions, comparison, document

    def write_and_validate(self, staging: Path | None = None):
        current, core, companions, comparison, document = (
            self.build_document(staging)
        )
        receipt_path = current / "sugar_mixtures_raman_conversion.json"
        receipt_bytes, receipt_sha256 = _write_receipt(
            receipt_path,
            document,
        )
        validated = _validate_receipt(
            receipt_path,
            raw_root=self.raw_root,
            inspection=self.inspection,
            core_path=core.path,
            companion_paths=self.companion_paths(current),
            comparison=comparison,
        )
        return (
            current,
            core,
            companions,
            comparison,
            document,
            receipt_path,
            receipt_bytes,
            receipt_sha256,
            validated,
        )

    @staticmethod
    def appendix_document(letter: str):
        text = (
            ROOT
            / "docs/superpowers/specs/"
            "2026-08-16-sugar-mixtures-adapter-design.md"
        ).read_text(encoding="utf-8")
        heading = f"## Appendix {letter}."
        section = text.split(heading, 1)[1]
        fenced = section.split("```json\n", 1)[1].split("\n```", 1)[0]
        payload = (fenced + "\n").encode("utf-8")
        return payload, json.loads(payload)

    @staticmethod
    def contract_identity(domain: bytes, value: object):
        payload = canonical_json_bytes(value, newline=True)
        return len(payload), hashlib.sha256(domain + payload).hexdigest()

    def test_policy_appendices_are_canonical_and_hdf5_owner_matches(self):
        for letter, (expected_bytes, domain, expected_sha256) in (
            self.APPENDIX_WITNESSES.items()
        ):
            with self.subTest(letter=letter):
                payload, document = self.appendix_document(letter)
                self.assertEqual(
                    payload,
                    canonical_json_bytes(document, newline=True),
                )
                self.assertEqual(len(payload), expected_bytes)
                self.assertEqual(
                    hashlib.sha256(domain + payload).hexdigest(),
                    expected_sha256,
                )
        _, hdf5_document = self.appendix_document("D")
        self.assertEqual(
            sugar_companions_module._HDF5_PHYSICAL_POLICY,
            hdf5_document,
        )
        for module in (
            sugar_mixtures_module,
            sugar_companions_module,
            sugar_source_module,
            sugar_receipt_module,
        ):
            self.assertFalse(hasattr(module, "_PUBLIC_PACKAGE_POLICY"))
            self.assertFalse(hasattr(module, "_STATISTICAL_AUDIT_POLICY"))

    def test_receipt_document_has_exact_contracts_and_attributions(self):
        _, core, companions, comparison, document = self.build_document()
        self.assertEqual(set(document), self.RECEIPT_KEYS)
        self.assertEqual(
            {
                key: document[key]
                for key in (
                    "receipt_schema_version",
                    "adapter_version",
                    "schema_version",
                    "dataset_id",
                )
            },
            {
                "receipt_schema_version": "1.0.0",
                "adapter_version": "0.1.0",
                "schema_version": "0.1.0",
                "dataset_id": "sugar_mixtures_raman",
            },
        )
        source = document["source_contract"]
        semantic = document["semantic_contract"]
        self.assertEqual(set(source), self.SOURCE_KEYS)
        self.assertEqual(set(semantic), self.SEMANTIC_KEYS)
        self.assertEqual(source["release"]["record_id"], 10779223)
        self.assertIs(type(source["release"]["record_id"]), int)
        self.assertEqual(len(source["canonical_members"]), 21)
        self.assertEqual(len(source["relevant_evidence_members"]), 46)
        self.assertEqual(len(source["evidence_snapshots"]), 3)
        self.assertEqual(
            source["canonical_members"],
            [
                {
                    "source_member": member.source_member,
                    "bytes": member.bytes,
                    "crc32": f"{member.crc32:08x}",
                    "sha256": member.sha256,
                    "condition": member.condition,
                    "record_id": member.record_id,
                }
                for member in self.inspection.accepted_members
            ],
        )
        self.assertEqual(
            source["relevant_evidence_members"],
            [
                {
                    "source_member": member.source_member,
                    "role": member.role,
                    "bytes": member.bytes,
                    "crc32": f"{member.crc32:08x}",
                    "sha256": member.sha256,
                }
                for member in self.inspection.relevant_evidence_members
            ],
        )
        self.assertEqual(
            source["excluded_inventory"],
            {
                "reason": "unrelated_synthetic_data_product",
                "regular_member_count": 1,
                "uncompressed_bytes": 8,
            },
        )
        self.assertEqual(
            self.contract_identity(
                b"rpe-sugar-receipt-source-contract-v1\0",
                source,
            ),
            self.EXPECTED_CONTRACTS["source"],
        )
        self.assertEqual(
            self.contract_identity(
                b"rpe-sugar-receipt-semantic-contract-v1\0",
                semantic,
            ),
            self.EXPECTED_CONTRACTS["semantic"],
        )
        policy = {
            "license": document["license"],
            "limitations": document["limitations"],
            "preprocessing_evidence": document["preprocessing_evidence"],
        }
        self.assertEqual(document["limitations"], self.LIMITATIONS)
        self.assertEqual(len(document["preprocessing_evidence"]), 1)
        self.assertEqual(
            self.contract_identity(
                b"rpe-sugar-receipt-policy-contract-v1\0",
                policy,
            ),
            self.EXPECTED_CONTRACTS["policy"],
        )
        self.assertEqual(set(document["outputs"]), self.OUTPUT_NAMES)
        attributed = {
            (
                f"{artifact_name}/{basename}"
                if artifact["artifact_type"] == "dataset"
                else basename
            ): details["sha256"]
            for artifact_name, artifact in document["outputs"].items()
            for basename, details in artifact["files"].items()
        }
        self.assertEqual(attributed, self.EXPECTED_NON_SELF_HASHES)
        self.assertNotIn("sugar_mixtures_raman_conversion.json", attributed)
        self.assertEqual(core.record_count, 21)
        self.assertEqual(set(companions), set(self.companion_paths(self.staging)))
        self.assertEqual(comparison.source_points_compared, 336)

    def test_receipt_round_trip_is_canonical_and_independently_validated(self):
        (
            _,
            _,
            _,
            _,
            document,
            receipt_path,
            receipt_bytes,
            receipt_sha256,
            validated,
        ) = self.write_and_validate()
        payload = receipt_path.read_bytes()
        self.assertEqual(payload, canonical_json_bytes(document, newline=True))
        self.assertEqual(
            (receipt_bytes, receipt_sha256),
            self.EXPECTED_CONTRACTS["receipt"],
        )
        self.assertEqual(validated, document)
        self.assertEqual(
            hashlib.sha256(payload).hexdigest(),
            receipt_sha256,
        )

    def test_receipt_rejects_noncanonical_nonfinite_and_contract_mutations(self):
        (
            current,
            core,
            _,
            comparison,
            document,
            receipt_path,
            _,
            _,
            _,
        ) = self.write_and_validate()

        def validate_mutated(mutated, *, raw_payload: bytes | None = None):
            receipt_path.write_bytes(
                canonical_json_bytes(mutated, newline=True)
                if raw_payload is None
                else raw_payload
            )
            with self.assertRaises(
                SugarMixturesValidationError
            ) as captured:
                _validate_receipt(
                    receipt_path,
                    raw_root=self.raw_root,
                    inspection=self.inspection,
                    core_path=core.path,
                    companion_paths=self.companion_paths(current),
                    comparison=comparison,
                )
            self.assertTrue(captured.exception.code.startswith("RECEIPT_"))
            self.assertNotEqual(captured.exception.path, "")

        validate_mutated(document, raw_payload=b" " + receipt_path.read_bytes())
        nonfinite = json.loads(canonical_json_bytes(document, newline=False))
        nonfinite["semantic_contract"]["precision"][
            "intensity_float32_max_abs_error"
        ] = float("nan")
        nonfinite_payload = (
            json.dumps(
                nonfinite,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            )
            + "\n"
        ).encode()
        validate_mutated(nonfinite, raw_payload=nonfinite_payload)

        def mutated(change):
            result = json.loads(
                canonical_json_bytes(document, newline=False)
            )
            change(result)
            return result

        cases = (
            lambda value: value.pop("dataset_id"),
            lambda value: value.update({"unexpected": 1}),
            lambda value: value["source_contract"]["archive"].update(
                {"bytes": True}
            ),
            lambda value: value["source_contract"]["inventory_counts"].update(
                {"canonical_members": 22}
            ),
            lambda value: value["source_contract"][
                "canonical_members"
            ].reverse(),
            lambda value: value["source_contract"]["canonical_members"][0].update(
                {"source_member": "wrong/source.csv"}
            ),
            lambda value: value.update({"source_contract_sha256": "0" * 64}),
            lambda value: value["semantic_contract"]["dataset"].update(
                {"record_count": 22}
            ),
            lambda value: value["semantic_contract"]["views"].update(
                {"companion_file": "/tmp/views.json"}
            ),
            lambda value: value.update({"semantic_contract_sha256": "0" * 64}),
            lambda value: value["license"].update({"text": "wrong"}),
            lambda value: value["limitations"].update(
                {"high_snr_not_clean_target": False}
            ),
            lambda value: value["limitations"].update(
                {"high_low_not_pointwise_pairs": False}
            ),
            lambda value: value["limitations"].update(
                {"adapter_defined_train_test_split": True}
            ),
            lambda value: value["limitations"].update(
                {"croissant_included": True}
            ),
            lambda value: value.update({"policy_contract_sha256": "0" * 64}),
            lambda value: value["outputs"]["sugar_mixtures_raman"]["files"][
                "arrays.h5"
            ].update({"bytes": 1}),
            lambda value: value["outputs"]["sugar_mixtures_raman"]["files"][
                "arrays.h5"
            ].update({"sha256": "0" * 64}),
            lambda value: value["outputs"].update(
                {
                    "sugar_mixtures_raman_conversion.json": {
                        "artifact_type": "file",
                        "files": {
                            "sugar_mixtures_raman_conversion.json": {
                                "bytes": 1,
                                "sha256": "0" * 64,
                            }
                        },
                    }
                }
            ),
            lambda value: value["semantic_contract"].update(
                {"runtime_seconds": 1.0}
            ),
        )
        for index, change in enumerate(cases):
            with self.subTest(index=index):
                validate_mutated(mutated(change))

    def test_receipt_rejects_whole_archive_drift_outside_relevant_members(self):
        (
            current,
            core,
            _,
            comparison,
            _,
            receipt_path,
            _,
            _,
            _,
        ) = self.write_and_validate()
        excluded_member = self.source.role_members["excluded_synthetic"][0]
        original_bytes = self.source.archive_path.stat().st_size
        corrupt_synthetic_zip_member_data(
            self.source.archive_path,
            excluded_member,
        )
        self.assertEqual(self.source.archive_path.stat().st_size, original_bytes)

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_receipt(
                receipt_path,
                raw_root=self.raw_root,
                inspection=self.inspection,
                core_path=core.path,
                companion_paths=self.companion_paths(current),
                comparison=comparison,
            )
        self.assertEqual(
            captured.exception.code,
            "RECEIPT_SOURCE_CONTRACT_INVALID",
        )
        self.assertEqual(
            captured.exception.path,
            "receipt.source_contract.archive.sha256",
        )

    def test_receipt_rejects_symlinked_output_attribution(self):
        (
            current,
            core,
            _,
            comparison,
            _,
            receipt_path,
            _,
            _,
            _,
        ) = self.write_and_validate()
        views_path = self.companion_paths(current)["views"]
        external = self.temporary_root / "external-views.json"
        shutil.copyfile(views_path, external)
        views_path.unlink()
        views_path.symlink_to(external)

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_receipt(
                receipt_path,
                raw_root=self.raw_root,
                inspection=self.inspection,
                core_path=core.path,
                companion_paths=self.companion_paths(current),
                comparison=comparison,
            )
        self.assertEqual(
            captured.exception.code,
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )
        self.assertEqual(
            captured.exception.path,
            "receipt.outputs.sugar_mixtures_raman_views.json",
        )

    def test_receipt_rejects_symlinked_staging_parent(self):
        (
            current,
            core,
            _,
            comparison,
            _,
            receipt_path,
            _,
            _,
            _,
        ) = self.write_and_validate()
        relocated = self.temporary_root / "relocated-staging"
        current.rename(relocated)
        current.symlink_to(relocated, target_is_directory=True)

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_receipt(
                current / receipt_path.name,
                raw_root=self.raw_root,
                inspection=self.inspection,
                core_path=current / core.path.name,
                companion_paths=self.companion_paths(current),
                comparison=comparison,
            )
        self.assertEqual(
            captured.exception.code,
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )
        self.assertEqual(captured.exception.path, "receipt.outputs")

    def test_receipt_rejects_duplicate_physical_file_attribution(self):
        (
            current,
            core,
            _,
            comparison,
            _,
            receipt_path,
            _,
            _,
            _,
        ) = self.write_and_validate()
        views = self.companion_paths(current)["views"]
        acquisition = self.companion_paths(current)["acquisition_json"]
        acquisition.unlink()
        os.link(views, acquisition)

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_receipt(
                receipt_path,
                raw_root=self.raw_root,
                inspection=self.inspection,
                core_path=core.path,
                companion_paths=self.companion_paths(current),
                comparison=comparison,
            )
        self.assertEqual(
            captured.exception.code,
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )
        self.assertEqual(captured.exception.path, "receipt.outputs")

    def test_receipt_preflights_dangling_receipt_and_core_symlinks(self):
        for target in ("receipt", "core"):
            with self.subTest(target=target):
                staging = self.temporary_root / f"dangling-{target}"
                (
                    current,
                    core,
                    _,
                    comparison,
                    _,
                    receipt_path,
                    _,
                    _,
                    _,
                ) = self.write_and_validate(staging)
                if target == "receipt":
                    receipt_path.unlink()
                    receipt_path.symlink_to(
                        self.temporary_root / "missing-receipt.json"
                    )
                    expected_path = (
                        "receipt.outputs."
                        "sugar_mixtures_raman_conversion.json"
                    )
                else:
                    shutil.rmtree(core.path)
                    core.path.symlink_to(
                        self.temporary_root / "missing-core",
                        target_is_directory=True,
                    )
                    expected_path = (
                        "receipt.outputs.sugar_mixtures_raman"
                    )
                with self.assertRaises(
                    SugarMixturesValidationError
                ) as captured:
                    _validate_receipt(
                        receipt_path,
                        raw_root=self.raw_root,
                        inspection=self.inspection,
                        core_path=core.path,
                        companion_paths=self.companion_paths(current),
                        comparison=comparison,
                    )
                self.assertEqual(
                    captured.exception.code,
                    "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
                )
                self.assertEqual(captured.exception.path, expected_path)

    def test_receipt_rejects_symlinked_retained_evidence(self):
        (
            current,
            core,
            _,
            comparison,
            _,
            receipt_path,
            _,
            _,
            _,
        ) = self.write_and_validate()
        evidence = (
            self.raw_root.parent.parent
            / "evidence/zenodo/10779223.json"
        )
        external = self.temporary_root / "external-zenodo.json"
        shutil.copyfile(evidence, external)
        evidence.unlink()
        evidence.symlink_to(external)

        with self.assertRaises(SugarMixturesValidationError) as captured:
            _validate_receipt(
                receipt_path,
                raw_root=self.raw_root,
                inspection=self.inspection,
                core_path=core.path,
                companion_paths=self.companion_paths(current),
                comparison=comparison,
            )
        self.assertEqual(
            captured.exception.code,
            "RECEIPT_SOURCE_CONTRACT_INVALID",
        )
        self.assertEqual(
            captured.exception.path,
            (
                "receipt.source_contract.evidence_snapshots."
                "evidence/zenodo/10779223.json"
            ),
        )

    def test_complete_staging_builds_exact_ten_file_topology(self):
        baseline_hdf5_handles = h5py.h5f.get_obj_count()
        staged = _build_sugar_mixtures_staged(
            self.raw_root,
            self.staging,
            self.inspection,
        )
        self.assertIsInstance(staged, _SugarStagedData)
        self.assertEqual(
            set(path.name for path in self.staging.iterdir()),
            self.ARTIFACT_NAMES,
        )
        regular_files = {
            path.relative_to(self.staging).as_posix()
            for path in self.staging.rglob("*")
            if path.is_file()
        }
        self.assertEqual(len(regular_files), 10)
        self.assertEqual(set(staged.output_hashes), regular_files)
        self.assertEqual(
            {
                name: digest
                for name, digest in staged.output_hashes.items()
                if name != "sugar_mixtures_raman_conversion.json"
            },
            self.EXPECTED_NON_SELF_HASHES,
        )
        self.assertEqual(
            (
                staged.receipt_bytes,
                staged.receipt_sha256,
            ),
            self.EXPECTED_CONTRACTS["receipt"],
        )
        self.assertEqual(staged.core.record_count, 21)
        self.assertEqual(staged.views.item_count, 8)
        self.assertEqual(staged.derived_endmembers.item_count, 10)
        self.assertEqual(staged.auxiliary_axes.item_count, 1)
        self.assertEqual(staged.acquisition_json.item_count, 21)
        self.assertEqual(staged.comparison.source_rows_compared, 21)
        self.assertEqual(h5py.h5f.get_obj_count(), baseline_hdf5_handles)
        self.assertEqual(file_sha256(self.source.archive_path), self.source_sha256)
        self.assertEqual(self.sentinel.read_bytes(), b"unchanged\n")
        self.assertFalse(any(self.output_root.glob("sugar_mixtures*")))

    def test_complete_staging_is_byte_deterministic_and_immutable(self):
        first_path = self.temporary_root / "first"
        second_path = self.temporary_root / "second"
        first = _build_sugar_mixtures_staged(
            self.raw_root,
            first_path,
            self.inspection,
        )
        time.sleep(1.1)
        second = _build_sugar_mixtures_staged(
            self.raw_root,
            second_path,
            self.inspection,
        )
        self.assertEqual(
            dict(first.output_hashes),
            dict(second.output_hashes),
        )
        self.assertEqual(
            first.output_hashes["sugar_mixtures_raman_conversion.json"],
            self.EXPECTED_CONTRACTS["receipt"][1],
        )
        with self.assertRaises(FrozenInstanceError):
            first.receipt_sha256 = "changed"
        with self.assertRaises(TypeError):
            first.output_hashes["new"] = "changed"
        with self.assertRaises(TypeError):
            first.core.files["arrays.h5"]["bytes"] = 0

    def test_complete_staging_requires_every_independent_validator(self):
        validators = (
            "_validate_views",
            "_validate_derived_endmembers",
            "_validate_auxiliary_axes",
            "_validate_acquisition_json",
            "_compare_sugar_dataset_to_source",
            "_validate_receipt",
        )
        for validator in validators:
            with self.subTest(validator=validator):
                staging = self.temporary_root / validator
                failure = SugarMixturesValidationError(
                    f"sentinel.{validator}",
                    "injected independent validation failure",
                    "SENTINEL_VALIDATION_FAILURE",
                )
                with patch.object(
                    sugar_mixtures_module,
                    validator,
                    side_effect=failure,
                ), self.assertRaises(
                    SugarMixturesValidationError
                ) as captured:
                    _build_sugar_mixtures_staged(
                        self.raw_root,
                        staging,
                        self.inspection,
                    )
                self.assertEqual(
                    captured.exception.code,
                    "SENTINEL_VALIDATION_FAILURE",
                )

    def test_complete_staging_rejects_nonempty_parent_without_publication(self):
        self.staging.mkdir()
        (self.staging / "unowned.txt").write_text(
            "preserve\n",
            encoding="utf-8",
        )
        with self.assertRaises(SugarMixturesValidationError):
            _build_sugar_mixtures_staged(
                self.raw_root,
                self.staging,
                self.inspection,
            )
        self.assertEqual(
            (self.staging / "unowned.txt").read_text(encoding="utf-8"),
            "preserve\n",
        )
        self.assertEqual(self.sentinel.read_bytes(), b"unchanged\n")
        self.assertFalse(any(self.output_root.glob("sugar_mixtures*")))

    def test_complete_staging_rejects_symlinked_staging_parent(self):
        real_staging = self.temporary_root / "real-staging"
        real_staging.mkdir()
        symlinked_staging = self.temporary_root / "symlinked-staging"
        symlinked_staging.symlink_to(
            real_staging,
            target_is_directory=True,
        )
        with self.assertRaises(SugarMixturesValidationError) as captured:
            _build_sugar_mixtures_staged(
                self.raw_root,
                symlinked_staging,
                self.inspection,
            )
        self.assertEqual(captured.exception.code, "STAGING_PARENT_INVALID")
        self.assertEqual(captured.exception.path, "staging_parent")
        self.assertEqual(list(real_staging.iterdir()), [])


SUGAR_RUNTIME_METRIC_FIELDS = (
    "total_seconds",
    "inspection_seconds",
    "staged_build_seconds",
    "core_build_seconds",
    "views_build_seconds",
    "derived_endmembers_build_seconds",
    "auxiliary_axes_build_seconds",
    "acquisition_json_build_seconds",
    "receipt_build_seconds",
    "validation_seconds",
    "peak_rss_raw",
    "peak_rss_bytes",
    "core_output_bytes",
    "views_output_bytes",
    "derived_endmembers_output_bytes",
    "auxiliary_axes_output_bytes",
    "acquisition_json_output_bytes",
    "receipt_bytes",
    "combined_output_bytes",
    "source_archive_bytes",
    "canonical_relevant_source_bytes",
    "compression_ratio_archive",
    "compression_ratio_relevant",
    "core_axis_groups",
    "core_lazy_open_seconds",
    "core_representative_record_read_seconds",
    "core_representative_record_id",
    "views_open_validate_seconds",
    "acquisition_json_stream_validate_seconds",
    "auxiliary_axes_open_validate_seconds",
    "derived_endmembers_open_validate_seconds",
)
SUGAR_STAGING_REVIEW_FIELDS = (
    "artifact_summaries",
    "receipt_sha256",
    "comparison",
    "source_statistics",
    "output_hashes",
    "metrics",
    "feasibility_failures",
)
SUGAR_RESOURCE_FAILURES = (
    "peak_rss_bytes",
    "combined_output_bytes",
    "inspection_seconds",
    "staged_build_seconds",
    "validation_seconds",
    "total_seconds",
    "core_lazy_open_seconds",
    "core_representative_record_read_seconds",
    "views_open_validate_seconds",
    "acquisition_json_stream_validate_seconds",
    "auxiliary_axes_open_validate_seconds",
    "derived_endmembers_open_validate_seconds",
)
SUGAR_SCIENTIFIC_FAILURES = (
    "canonical_records",
    "eligible_records",
    "core_axis_groups",
    "view_entries",
    "view_memberships",
    "endmembers",
    "auxiliary_axis_sets",
    "acquisition_json_lines",
    "staged_regular_files",
    "source_rows_compared",
    "source_points_compared",
    "consolidated_intensity_cells_compared",
    "prepared_abundance_rows_compared",
    "prepared_endmembers_compared",
    "intensity_mismatches",
    "axis_mismatches",
    "target_mismatches",
    "metadata_mismatches",
    "preprocessing_status_mismatches",
    "companion_reference_mismatches",
)


class SugarMixturesRuntimeAndFeasibilityTest(unittest.TestCase):
    REQUIRED_INTERFACES = (
        "_linux_ru_maxrss_to_bytes",
        "_measure_sugar_core_access",
        "_measure_sugar_staged",
        "_sugar_feasibility_failures",
        "_stage_sugar_mixtures_for_review",
    )

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.raw_root = self.source.raw_root
        self.inspection = _inspect_sugar_mixtures_source(
            self.raw_root,
            synthetic_sugar_contract(self.source),
        )
        self.staging = self.temporary_root / "staging"

    def require_interfaces(self):
        missing = [
            name
            for name in self.REQUIRED_INTERFACES
            if not hasattr(sugar_mixtures_module, name)
        ]
        self.assertEqual(
            missing,
            [],
            f"Task 9 interfaces are missing: {missing}",
        )

    def build(self):
        return _build_sugar_mixtures_staged(
            self.raw_root,
            self.staging,
            self.inspection,
        )

    def metrics(self, **changes):
        self.require_interfaces()
        values = {
            "total_seconds": 10.0,
            "inspection_seconds": 0.1,
            "staged_build_seconds": 0.75,
            "core_build_seconds": 0.125,
            "views_build_seconds": 0.125,
            "derived_endmembers_build_seconds": 0.125,
            "auxiliary_axes_build_seconds": 0.125,
            "acquisition_json_build_seconds": 0.125,
            "receipt_build_seconds": 0.125,
            "validation_seconds": 0.1,
            "peak_rss_raw": 1,
            "peak_rss_bytes": 1024,
            "core_output_bytes": 1,
            "views_output_bytes": 1,
            "derived_endmembers_output_bytes": 1,
            "auxiliary_axes_output_bytes": 1,
            "acquisition_json_output_bytes": 1,
            "receipt_bytes": 1,
            "combined_output_bytes": 6,
            "source_archive_bytes": 12,
            "canonical_relevant_source_bytes": 8,
            "compression_ratio_archive": 0.5,
            "compression_ratio_relevant": 0.75,
            "core_axis_groups": 1,
            "core_lazy_open_seconds": 0.1,
            "core_representative_record_read_seconds": 0.1,
            "core_representative_record_id": (
                "low_snr-s063-f03-p02-r02-m01-rep03"
            ),
            "views_open_validate_seconds": 0.1,
            "acquisition_json_stream_validate_seconds": 0.1,
            "auxiliary_axes_open_validate_seconds": 0.1,
            "derived_endmembers_open_validate_seconds": 0.1,
        }
        values.update(changes)
        return sugar_mixtures_module.SugarMixturesRuntimeMetrics(**values)

    def production_gate_staged(self):
        staged = self.build()
        output_hashes = {
            name: digest
            for name, digest in staged.output_hashes.items()
        }
        return replace(
            staged,
            core=replace(
                staged.core,
                record_count=9_800,
                eligible_records=9_800,
                axis_group_count=1,
                output_bytes=1,
            ),
            views=replace(staged.views, bytes=1, item_count=8),
            derived_endmembers=replace(
                staged.derived_endmembers,
                bytes=1,
                item_count=10,
            ),
            auxiliary_axes=replace(
                staged.auxiliary_axes,
                bytes=1,
                item_count=1,
            ),
            acquisition_json=replace(
                staged.acquisition_json,
                bytes=1,
                item_count=9_800,
            ),
            receipt_bytes=1,
            comparison=replace(
                staged.comparison,
                source_rows_compared=9_800,
                source_points_compared=19_600_000,
                consolidated_intensity_cells_compared=19_600_000,
                prepared_abundance_rows_compared=19_400,
                prepared_endmembers_compared=10,
            ),
            output_hashes=MappingProxyType(output_hashes),
        )

    def test_runtime_and_review_dataclasses_have_exact_frozen_fields(self):
        self.require_interfaces()
        self.assertEqual(
            tuple(
                field.name
                for field in fields(
                    sugar_mixtures_module.SugarMixturesRuntimeMetrics
                )
            ),
            SUGAR_RUNTIME_METRIC_FIELDS,
        )
        self.assertEqual(
            tuple(
                field.name
                for field in fields(
                    sugar_mixtures_module._SugarStagingReview
                )
            ),
            SUGAR_STAGING_REVIEW_FIELDS,
        )
        metrics = self.metrics()
        with self.assertRaises(FrozenInstanceError):
            metrics.total_seconds = 0.0

    def test_linux_ru_maxrss_conversion_rejects_invalid_values(self):
        self.require_interfaces()
        self.assertEqual(
            sugar_mixtures_module._linux_ru_maxrss_to_bytes(0),
            0,
        )
        self.assertEqual(
            sugar_mixtures_module._linux_ru_maxrss_to_bytes(1),
            1024,
        )
        self.assertEqual(
            sugar_mixtures_module._linux_ru_maxrss_to_bytes(4096),
            4_194_304,
        )
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    sugar_mixtures_module._linux_ru_maxrss_to_bytes(
                        invalid
                    )

    def test_all_strict_resource_thresholds_fail_at_equality(self):
        staged = self.production_gate_staged()
        staged = replace(
            staged,
            core=replace(
                staged.core,
                output_bytes=536_870_907,
            ),
        )
        metrics = self.metrics(
            peak_rss_bytes=2_147_483_648,
            peak_rss_raw=2_097_152,
            core_output_bytes=536_870_907,
            combined_output_bytes=536_870_912,
            compression_ratio_archive=44_739_242.666666664,
            compression_ratio_relevant=67_108_864.0,
            inspection_seconds=120.0,
            staged_build_seconds=600.0,
            receipt_build_seconds=599.375,
            validation_seconds=600.0,
            total_seconds=1_998.0,
            core_lazy_open_seconds=30.0,
            core_representative_record_read_seconds=2.0,
            views_open_validate_seconds=10.0,
            acquisition_json_stream_validate_seconds=120.0,
            auxiliary_axes_open_validate_seconds=10.0,
            derived_endmembers_open_validate_seconds=10.0,
        )
        self.assertEqual(
            sugar_mixtures_module._sugar_feasibility_failures(
                staged,
                metrics,
                staged.comparison,
            ),
            SUGAR_RESOURCE_FAILURES,
        )
        passing = self.metrics()
        passing_staged = replace(
            staged,
            core=replace(
                staged.core,
                output_bytes=1,
            ),
        )
        self.assertEqual(
            sugar_mixtures_module._sugar_feasibility_failures(
                passing_staged,
                passing,
                passing_staged.comparison,
            ),
            (),
        )

    def test_gate_fails_closed_for_every_invalid_metric_and_formula(self):
        staged = self.production_gate_staged()
        numeric_fields = (
            name
            for name in SUGAR_RUNTIME_METRIC_FIELDS
            if name != "core_representative_record_id"
        )
        for name in numeric_fields:
            with self.subTest(name=name, value="nan"):
                failures = (
                    sugar_mixtures_module._sugar_feasibility_failures(
                        staged,
                        replace(
                            self.metrics(),
                            **{name: float("nan")},
                        ),
                        staged.comparison,
                    )
                )
                self.assertIn(name, failures)
            with self.subTest(name=name, value="negative"):
                failures = (
                    sugar_mixtures_module._sugar_feasibility_failures(
                        staged,
                        replace(self.metrics(), **{name: -1}),
                        staged.comparison,
                    )
                )
                self.assertIn(name, failures)
            with self.subTest(name=name, value="boolean"):
                failures = (
                    sugar_mixtures_module._sugar_feasibility_failures(
                        staged,
                        replace(self.metrics(), **{name: True}),
                        staged.comparison,
                    )
                )
                self.assertIn(name, failures)
        formula_cases = (
            ("total_seconds", 0.1),
            ("staged_build_seconds", 0.7),
            ("peak_rss_bytes", 1025),
            ("combined_output_bytes", 7),
            ("compression_ratio_archive", 0.4),
            ("compression_ratio_relevant", 0.5),
            ("core_axis_groups", 2),
        )
        for name, value in formula_cases:
            with self.subTest(formula=name):
                failures = (
                    sugar_mixtures_module._sugar_feasibility_failures(
                        staged,
                        replace(self.metrics(), **{name: value}),
                        staged.comparison,
                    )
                )
                self.assertIn(name, failures)
        for name in (
            metric_name
            for metric_name in SUGAR_RUNTIME_METRIC_FIELDS
            if metric_name != "core_representative_record_id"
        ):
            with self.subTest(name=name, value="nonnumeric"):
                failures = (
                    sugar_mixtures_module._sugar_feasibility_failures(
                        staged,
                        replace(self.metrics(), **{name: "invalid"}),
                        staged.comparison,
                    )
                )
                self.assertIn(name, failures)

    def test_gate_requires_integer_fields_and_exact_ten_path_inventory(self):
        staged = self.production_gate_staged()
        integer_fields = (
            "peak_rss_raw",
            "peak_rss_bytes",
            "core_output_bytes",
            "views_output_bytes",
            "derived_endmembers_output_bytes",
            "auxiliary_axes_output_bytes",
            "acquisition_json_output_bytes",
            "receipt_bytes",
            "combined_output_bytes",
            "source_archive_bytes",
            "canonical_relevant_source_bytes",
            "core_axis_groups",
        )
        for name in integer_fields:
            with self.subTest(name=name):
                failures = (
                    sugar_mixtures_module._sugar_feasibility_failures(
                        staged,
                        replace(self.metrics(), **{name: 1.0}),
                        staged.comparison,
                    )
                )
                self.assertIn(name, failures)

        wrong_paths = dict(staged.output_hashes)
        wrong_paths.pop(next(iter(wrong_paths)))
        wrong_paths["unexpected.txt"] = "0" * 64
        staged = replace(
            staged,
            output_hashes=MappingProxyType(wrong_paths),
        )
        failures = sugar_mixtures_module._sugar_feasibility_failures(
            staged,
            self.metrics(),
            staged.comparison,
        )
        self.assertIn("staged_regular_files", failures)

        shutil.rmtree(self.staging)
        correct_paths = self.production_gate_staged()
        wrong_hashes = dict(correct_paths.output_hashes)
        first_path = next(iter(wrong_hashes))
        wrong_hashes[first_path] = "0" * 64
        correct_paths = replace(
            correct_paths,
            output_hashes=MappingProxyType(wrong_hashes),
        )
        failures = sugar_mixtures_module._sugar_feasibility_failures(
            correct_paths,
            self.metrics(),
            correct_paths.comparison,
        )
        self.assertIn("staged_regular_files", failures)

    def test_gate_returns_all_fixed_scientific_failures_in_order(self):
        staged = self.build()
        comparison = replace(
            staged.comparison,
            source_rows_compared=0,
            source_points_compared=0,
            consolidated_intensity_cells_compared=0,
            prepared_abundance_rows_compared=0,
            prepared_endmembers_compared=0,
            intensity_mismatches=1,
            axis_mismatches=1,
            target_mismatches=1,
            metadata_mismatches=1,
            preprocessing_status_mismatches=1,
            companion_reference_mismatches=1,
        )
        staged = replace(
            staged,
            core=replace(
                staged.core,
                record_count=0,
                eligible_records=0,
                axis_group_count=0,
            ),
            views=replace(staged.views, item_count=0),
            derived_endmembers=replace(
                staged.derived_endmembers,
                item_count=0,
            ),
            auxiliary_axes=replace(
                staged.auxiliary_axes,
                item_count=0,
            ),
            acquisition_json=replace(
                staged.acquisition_json,
                item_count=0,
            ),
            output_hashes=MappingProxyType({}),
        )
        receipt = json.loads(staged.receipt_path.read_bytes())
        role_bytes = {
            role["role"]: role["uncompressed_bytes"]
            for role in receipt["source_contract"]["member_roles"]
        }
        source_archive_bytes = receipt["source_contract"]["archive"]["bytes"]
        relevant_bytes = sum(
            role_bytes[name]
            for name in ("canonical_high", "canonical_low", "target")
        )
        combined_bytes = (
            staged.core.output_bytes
            + staged.views.bytes
            + staged.derived_endmembers.bytes
            + staged.auxiliary_axes.bytes
            + staged.acquisition_json.bytes
            + staged.receipt_bytes
        )
        self.assertEqual(
            sugar_mixtures_module._sugar_feasibility_failures(
                staged,
                self.metrics(
                    core_output_bytes=staged.core.output_bytes,
                    views_output_bytes=staged.views.bytes,
                    derived_endmembers_output_bytes=(
                        staged.derived_endmembers.bytes
                    ),
                    auxiliary_axes_output_bytes=staged.auxiliary_axes.bytes,
                    acquisition_json_output_bytes=(
                        staged.acquisition_json.bytes
                    ),
                    receipt_bytes=staged.receipt_bytes,
                    combined_output_bytes=combined_bytes,
                    source_archive_bytes=source_archive_bytes,
                    canonical_relevant_source_bytes=relevant_bytes,
                    compression_ratio_archive=(
                        combined_bytes / source_archive_bytes
                    ),
                    compression_ratio_relevant=(
                        combined_bytes / relevant_bytes
                    ),
                    core_axis_groups=0,
                ),
                comparison,
            ),
            SUGAR_SCIENTIFIC_FAILURES,
        )

    def test_core_access_probe_selects_literal_id_and_closes_handle(self):
        self.require_interfaces()
        staged = self.build()
        opened = []
        real_open = UnifiedDataset.open

        def recording_open(path, **kwargs):
            dataset = real_open(path, **kwargs)
            opened.append(dataset)
            return dataset

        with patch.object(
            sugar_mixtures_module.UnifiedDataset,
            "open",
            side_effect=recording_open,
        ):
            open_seconds, read_seconds, record_id = (
                sugar_mixtures_module._measure_sugar_core_access(
                    staged.core.path
                )
            )
        self.assertGreaterEqual(open_seconds, 0.0)
        self.assertGreaterEqual(read_seconds, 0.0)
        self.assertEqual(
            record_id,
            "low_snr-s049-e01-p03-r01-m01-rep01",
        )
        self.assertEqual(len(opened), 1)
        with self.assertRaises(DatasetClosedError):
            _ = opened[0].record_ids

    def test_measurement_uses_exact_formulas_and_complete_access_probes(self):
        self.require_interfaces()
        timings = {
            "sentinel": 1.0,
        }
        staged = _build_sugar_mixtures_staged(
            self.raw_root,
            self.staging,
            self.inspection,
            timings=timings,
        )
        with patch.object(
            sugar_mixtures_module.resource,
            "getrusage",
            return_value=type("Usage", (), {"ru_maxrss": 2})(),
        ):
            metrics = sugar_mixtures_module._measure_sugar_staged(
                staged,
                total_seconds=4.0,
                inspection_seconds=0.8,
                peak_rss_raw=2,
            )
        self.assertEqual(
            metrics.staged_build_seconds,
            sum(
                timings[name]
                for name in (
                    "core_build_seconds",
                    "views_build_seconds",
                    "derived_endmembers_build_seconds",
                    "auxiliary_axes_build_seconds",
                    "acquisition_json_build_seconds",
                    "receipt_build_seconds",
                )
            ),
        )
        self.assertGreaterEqual(metrics.validation_seconds, 0.0)
        self.assertEqual(metrics.peak_rss_bytes, 2048)
        self.assertEqual(
            metrics.combined_output_bytes,
            staged.core.output_bytes
            + staged.views.bytes
            + staged.derived_endmembers.bytes
            + staged.auxiliary_axes.bytes
            + staged.acquisition_json.bytes
            + staged.receipt_bytes,
        )
        receipt = json.loads(staged.receipt_path.read_bytes())
        role_bytes = {
            role["role"]: role["uncompressed_bytes"]
            for role in receipt["source_contract"]["member_roles"]
        }
        relevant_bytes = sum(
            role_bytes[name]
            for name in (
                "canonical_high",
                "canonical_low",
                "target",
            )
        )
        self.assertEqual(
            metrics.source_archive_bytes,
            self.source.archive_bytes,
        )
        self.assertEqual(
            metrics.canonical_relevant_source_bytes,
            relevant_bytes,
        )
        self.assertEqual(
            metrics.compression_ratio_archive,
            metrics.combined_output_bytes
            / metrics.source_archive_bytes,
        )
        self.assertEqual(
            metrics.compression_ratio_relevant,
            metrics.combined_output_bytes / relevant_bytes,
        )
        self.assertEqual(metrics.core_axis_groups, 1)
        self.assertEqual(
            metrics.core_representative_record_id,
            "low_snr-s049-e01-p03-r01-m01-rep01",
        )
        for name in (
            "core_lazy_open_seconds",
            "core_representative_record_read_seconds",
            "views_open_validate_seconds",
            "acquisition_json_stream_validate_seconds",
            "auxiliary_axes_open_validate_seconds",
            "derived_endmembers_open_validate_seconds",
        ):
            self.assertGreaterEqual(getattr(metrics, name), 0.0)
        for forbidden in SUGAR_RUNTIME_METRIC_FIELDS:
            self.assertNotIn(forbidden, receipt)

    def test_receipt_contract_validation_time_is_not_receipt_write_time(self):
        real_build_receipt = sugar_mixtures_module._build_receipt_document
        real_write_receipt = sugar_mixtures_module._write_receipt

        def delayed_build_receipt(*args, **kwargs):
            time.sleep(0.08)
            return real_build_receipt(*args, **kwargs)

        def delayed_write_receipt(*args, **kwargs):
            time.sleep(0.02)
            return real_write_receipt(*args, **kwargs)

        timings = {}
        with patch.object(
            sugar_mixtures_module,
            "_build_receipt_document",
            side_effect=delayed_build_receipt,
        ), patch.object(
            sugar_mixtures_module,
            "_write_receipt",
            side_effect=delayed_write_receipt,
        ):
            _build_sugar_mixtures_staged(
                self.raw_root,
                self.staging,
                self.inspection,
                timings=timings,
            )
        self.assertGreaterEqual(timings["validation_seconds"], 0.08)
        self.assertGreaterEqual(timings["receipt_build_seconds"], 0.02)
        self.assertLess(timings["receipt_build_seconds"], 0.08)

    def test_access_probes_reject_incomplete_role_semantics(self):
        staged = self.build()
        source_member_by_record_id = {
            member.record_id: member.source_member
            for member in self.inspection.accepted_members
        }
        source_bindings = {
            member.record_id: (member.source_member, member.sha256)
            for member in self.inspection.accepted_members
        }
        views_document = json.loads(staged.views.path.read_bytes())
        views_document["views"][0]["selector"] = {"wrong": "selector"}
        staged.views.path.write_bytes(
            canonical_json_bytes(views_document, newline=True)
        )
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_views_access(
                staged.views.path,
                source_member_by_record_id=source_member_by_record_id,
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        views_document = json.loads(staged.views.path.read_bytes())
        views_document["views"][0]["source_member_path_sha256"] = "0" * 64
        staged.views.path.write_bytes(
            canonical_json_bytes(views_document, newline=True)
        )
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_views_access(
                staged.views.path,
                source_member_by_record_id=source_member_by_record_id,
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        views_document = json.loads(staged.views.path.read_bytes())
        for view in views_document["views"]:
            view["record_ids"] = []
            view["record_count"] = 0
            view["record_id_sha256"] = hashlib.sha256(
                b"rpe-sugar-view-record-ids-v1\0"
            ).hexdigest()
            view["source_member_path_sha256"] = hashlib.sha256(
                b"rpe-sugar-view-source-members-v1\0"
            ).hexdigest()
        staged.views.path.write_bytes(
            canonical_json_bytes(views_document, newline=True)
        )
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_views_access(
                staged.views.path,
                source_member_by_record_id=source_member_by_record_id,
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        acquisition_documents = [
            json.loads(line)
            for line in staged.acquisition_json.path.read_bytes().splitlines()
        ]
        acquisition_documents[0]["record_id"] = "aaa-unknown"
        staged.acquisition_json.path.write_bytes(
            b"".join(
                canonical_json_bytes(document, newline=True)
                for document in acquisition_documents
            )
        )
        with UnifiedDataset.open(staged.core.path) as dataset:
            record_ids = set(dataset.record_ids)
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_acquisition_json_access(
                staged.acquisition_json.path,
                record_ids=record_ids,
                source_bindings=source_bindings,
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        acquisition_documents = [
            json.loads(line)
            for line in staged.acquisition_json.path.read_bytes().splitlines()
        ]
        acquisition_documents[0]["source_member_sha256"] = "0" * 64
        staged.acquisition_json.path.write_bytes(
            b"".join(
                canonical_json_bytes(document, newline=True)
                for document in acquisition_documents
            )
        )
        with UnifiedDataset.open(staged.core.path) as dataset:
            record_ids = set(dataset.record_ids)
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_acquisition_json_access(
                staged.acquisition_json.path,
                record_ids=record_ids,
                source_bindings=source_bindings,
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        with h5py.File(staged.auxiliary_axes.path, "r+") as artifact:
            artifact.attrs["logical_content_sha256"] = "0" * 64
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_auxiliary_axes_access(
                staged.auxiliary_axes.path,
                expected_logical_content_sha256=(
                    staged.auxiliary_axes.logical_content_sha256
                ),
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        with h5py.File(staged.auxiliary_axes.path, "r+") as artifact:
            artifact["axes"]["wavelength_nm"][0] += 1
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_auxiliary_axes_access(
                staged.auxiliary_axes.path,
                expected_logical_content_sha256=(
                    staged.auxiliary_axes.logical_content_sha256
                ),
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        with h5py.File(staged.derived_endmembers.path, "r+") as artifact:
            artifact.attrs["logical_content_sha256"] = "0" * 64
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_derived_endmembers_access(
                staged.derived_endmembers.path,
                expected_logical_content_sha256=(
                    staged.derived_endmembers.logical_content_sha256
                ),
            )

    def test_hdf5_access_probes_reject_wrong_dtype_chunks_and_filters(self):
        staged = self.build()
        auxiliary = staged.auxiliary_axes.path
        replacement = self.temporary_root / "wrong-auxiliary.h5"
        with h5py.File(auxiliary, "r") as source:
            attrs = dict(source.attrs)
            metadata = source["metadata_json"][...]
            pixel = source["axes"]["pixel"][...]
            wavelength = source["axes"]["wavelength_nm"][...]
        with h5py.File(replacement, "w") as target:
            for key, value in attrs.items():
                target.attrs[key] = value
            target.create_dataset("metadata_json", data=metadata)
            axes = target.create_group("axes")
            axes.create_dataset("pixel", data=pixel)
            axes.create_dataset(
                "wavelength_nm",
                data=wavelength.astype("<f8"),
            )
        os.replace(replacement, auxiliary)
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_auxiliary_axes_access(
                auxiliary,
                expected_logical_content_sha256=(
                    staged.auxiliary_axes.logical_content_sha256
                ),
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        endmembers = staged.derived_endmembers.path
        replacement = self.temporary_root / "wrong-endmembers.h5"
        with h5py.File(endmembers, "r") as source:
            attrs = dict(source.attrs)
            metadata = source["metadata_json"][...]
            intensity = source["intensity"][...]
        with h5py.File(replacement, "w") as target:
            for key, value in attrs.items():
                target.attrs[key] = value
            target.create_dataset("metadata_json", data=metadata)
            target.create_dataset(
                "intensity",
                data=intensity.astype("<f8"),
            )
        os.replace(replacement, endmembers)
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_derived_endmembers_access(
                endmembers,
                expected_logical_content_sha256=(
                    staged.derived_endmembers.logical_content_sha256
                ),
            )

    def test_hdf5_access_probes_reject_extra_attributes(self):
        staged = self.build()
        with h5py.File(staged.auxiliary_axes.path, "r+") as artifact:
            artifact.attrs["unexpected"] = "value"
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_auxiliary_axes_access(
                staged.auxiliary_axes.path,
                expected_logical_content_sha256=(
                    staged.auxiliary_axes.logical_content_sha256
                ),
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        with h5py.File(staged.derived_endmembers.path, "r+") as artifact:
            artifact.attrs["unexpected"] = "value"
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_derived_endmembers_access(
                staged.derived_endmembers.path,
                expected_logical_content_sha256=(
                    staged.derived_endmembers.logical_content_sha256
                ),
            )

    def test_hdf5_access_probes_reject_wrong_fixed_attributes(self):
        staged = self.build()
        with h5py.File(staged.auxiliary_axes.path, "r+") as artifact:
            artifact.attrs["artifact_role"] = "wrong"
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_auxiliary_axes_access(
                staged.auxiliary_axes.path,
                expected_logical_content_sha256=(
                    staged.auxiliary_axes.logical_content_sha256
                ),
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        with h5py.File(staged.derived_endmembers.path, "r+") as artifact:
            artifact.attrs["dataset_id"] = "wrong"
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_derived_endmembers_access(
                staged.derived_endmembers.path,
                expected_logical_content_sha256=(
                    staged.derived_endmembers.logical_content_sha256
                ),
            )

        shutil.rmtree(self.staging)
        staged = self.build()
        with h5py.File(staged.derived_endmembers.path, "r+") as artifact:
            artifact["intensity"][0, 0, 0] += 1
        with self.assertRaises(SugarMixturesValidationError):
            sugar_mixtures_module._measure_derived_endmembers_access(
                staged.derived_endmembers.path,
                expected_logical_content_sha256=(
                    staged.derived_endmembers.logical_content_sha256
                ),
            )


class SugarMixturesStagingReviewTest(unittest.TestCase):
    REQUIRED_INTERFACES = (
        "_measure_sugar_staged",
        "_sugar_feasibility_failures",
        "_stage_sugar_mixtures_for_review",
    )
    FINAL_NAMES = (
        "sugar_mixtures_raman",
        "sugar_mixtures_raman_views.json",
        "sugar_mixtures_raman_derived_reference_endmembers.h5",
        "sugar_mixtures_raman_auxiliary_axes.h5",
        "sugar_mixtures_raman_acquisition_json_text.jsonl",
        "sugar_mixtures_raman_conversion.json",
    )

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.raw_root = self.source.raw_root
        self.inspection = _inspect_sugar_mixtures_source(
            self.raw_root,
            synthetic_sugar_contract(self.source),
        )
        self.output_root = self.temporary_root / "output"

    def require_interfaces(self):
        missing = [
            name
            for name in self.REQUIRED_INTERFACES
            if not hasattr(sugar_mixtures_module, name)
        ]
        self.assertEqual(
            missing,
            [],
            f"Task 9 staging interfaces are missing: {missing}",
        )

    @staticmethod
    def snapshot(root: Path):
        if not root.exists():
            return {}
        result = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[relative] = ("symlink", os.readlink(path))
            elif path.is_dir():
                result[relative] = ("directory",)
            else:
                result[relative] = (
                    "file",
                    path.stat().st_size,
                    file_sha256(path),
                )
        return result

    def review(self, *, output_root: Path | None = None):
        self.require_interfaces()
        with patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            return_value=self.inspection,
        ), patch.object(
            sugar_mixtures_module.resource,
            "getrusage",
            return_value=type("Usage", (), {"ru_maxrss": 1})(),
        ):
            return sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root if output_root is None else output_root,
            )

    def test_review_returns_complete_immutable_evidence_and_cleans_gate_failure(self):
        review = self.review()
        self.assertIsInstance(
            review,
            sugar_mixtures_module._SugarStagingReview,
        )
        self.assertEqual(
            tuple(field.name for field in fields(review)),
            SUGAR_STAGING_REVIEW_FIELDS,
        )
        self.assertEqual(set(review.artifact_summaries), {
            "core",
            "views",
            "derived_endmembers",
            "auxiliary_axes",
            "acquisition_json",
            "receipt",
        })
        self.assertEqual(review.artifact_summaries["core"]["record_count"], 21)
        self.assertEqual(review.artifact_summaries["views"]["item_count"], 8)
        self.assertEqual(
            review.artifact_summaries["views"]["view_memberships"],
            63,
        )
        self.assertEqual(
            review.artifact_summaries["derived_endmembers"]["item_count"],
            10,
        )
        self.assertEqual(
            review.artifact_summaries["auxiliary_axes"]["item_count"],
            1,
        )
        self.assertEqual(
            review.artifact_summaries["acquisition_json"]["item_count"],
            21,
        )
        self.assertEqual(
            review.artifact_summaries["acquisition_json"]["line_count"],
            21,
        )
        self.assertEqual(
            review.artifact_summaries["auxiliary_axes"][
                "auxiliary_axis_set_id"
            ],
            self.inspection.auxiliary_axis_set_id,
        )
        self.assertEqual(len(review.output_hashes), 10)
        self.assertEqual(
            review.output_hashes["sugar_mixtures_raman_conversion.json"],
            review.receipt_sha256,
        )
        self.assertEqual(
            review.feasibility_failures,
            (
                "canonical_records",
                "eligible_records",
                "view_memberships",
                "acquisition_json_lines",
                "source_rows_compared",
                "source_points_compared",
                "consolidated_intensity_cells_compared",
                "prepared_abundance_rows_compared",
            ),
        )
        with self.assertRaises(TypeError):
            review.output_hashes["new"] = "0" * 64
        with self.assertRaises(TypeError):
            review.artifact_summaries["core"]["record_count"] = 0
        with self.assertRaises(FrozenInstanceError):
            review.receipt_sha256 = "changed"
        self.assertFalse(self.output_root.exists())
        self.assertFalse(hasattr(review, "staging_parent"))
        component_seconds = (
            review.metrics.inspection_seconds
            + review.metrics.staged_build_seconds
            + review.metrics.validation_seconds
            + review.metrics.core_lazy_open_seconds
            + review.metrics.core_representative_record_read_seconds
            + review.metrics.views_open_validate_seconds
            + review.metrics.acquisition_json_stream_validate_seconds
            + review.metrics.auxiliary_axes_open_validate_seconds
            + review.metrics.derived_endmembers_open_validate_seconds
        )
        self.assertGreaterEqual(review.metrics.total_seconds, component_seconds)
        self.assertEqual(
            file_sha256(self.source.archive_path),
            self.source.archive_sha256,
        )

    def test_review_allocates_direct_child_and_preserves_finals_and_siblings(self):
        self.output_root.mkdir()
        sentinel = self.output_root / "unowned.txt"
        sentinel.write_text("unchanged\n", encoding="utf-8")
        for name in self.FINAL_NAMES:
            path = self.output_root / name
            if "." not in name:
                path.mkdir()
                (path / "original.txt").write_text(
                    "original\n",
                    encoding="utf-8",
                )
            else:
                path.write_text("original\n", encoding="utf-8")
        before = self.snapshot(self.output_root)
        allocated = []
        real_mkdtemp = tempfile.mkdtemp

        def recording_mkdtemp(*args, **kwargs):
            path = Path(real_mkdtemp(*args, **kwargs))
            if kwargs.get("prefix") == ".sugar-mixtures.staging-":
                allocated.append(path)
            return str(path)

        with patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            return_value=self.inspection,
        ), patch.object(
            sugar_mixtures_module.resource,
            "getrusage",
            return_value=type("Usage", (), {"ru_maxrss": 1})(),
        ), patch.object(
            sugar_mixtures_module.tempfile,
            "mkdtemp",
            side_effect=recording_mkdtemp,
        ), patch.object(
            sugar_mixtures_module,
            "_sugar_feasibility_failures",
            return_value=(),
        ):
            review = sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertEqual(review.feasibility_failures, ())
        self.assertEqual(len(allocated), 1)
        self.assertEqual(allocated[0].parent, self.output_root)
        self.assertTrue(
            allocated[0].name.startswith(".sugar-mixtures.staging-")
        )
        self.assertFalse(allocated[0].exists())
        self.assertEqual(self.snapshot(self.output_root), before)

    def test_free_space_and_invalid_output_root_fail_before_source_access(self):
        self.require_interfaces()
        with patch.object(
            sugar_mixtures_module.shutil,
            "disk_usage",
            return_value=shutil._ntuple_diskusage(1, 1, 1_610_612_736),
        ), patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
        ) as inspect, self.assertRaises(
            SugarMixturesValidationError
        ) as captured:
            sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertEqual(captured.exception.code, "INSUFFICIENT_FREE_SPACE")
        self.assertEqual(
            captured.exception.path,
            "output_root.free_disk_bytes",
        )
        inspect.assert_not_called()
        self.assertFalse(self.output_root.exists())

        self.output_root.write_text("not a directory\n", encoding="utf-8")
        with patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
        ) as inspect, self.assertRaises(
            SugarMixturesValidationError
        ) as captured:
            sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertEqual(captured.exception.path, "output_root")
        inspect.assert_not_called()

    def test_staging_allocator_must_return_owned_direct_child(self):
        external = self.temporary_root / "external-staging"
        external.mkdir()
        with patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            return_value=self.inspection,
        ), patch.object(
            sugar_mixtures_module.tempfile,
            "mkdtemp",
            return_value=str(external),
        ), patch.object(
            sugar_mixtures_module,
            "_build_sugar_mixtures_staged",
        ) as build, self.assertRaises(
            SugarMixturesValidationError
        ) as captured:
            sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertEqual(captured.exception.code, "STAGING_PARENT_INVALID")
        self.assertEqual(captured.exception.path, "staging_parent")
        build.assert_not_called()
        self.assertTrue(external.is_dir())
        self.assertFalse(self.output_root.exists())

    def test_source_build_probe_and_cleanup_failures_propagate_after_cleanup(self):
        self.require_interfaces()
        source_error = SugarMixturesValidationError(
            "source.synthetic",
            "injected source failure",
            "SENTINEL_SOURCE_FAILURE",
        )
        with patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            side_effect=source_error,
        ), self.assertRaises(SugarMixturesValidationError) as captured:
            sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertIs(captured.exception, source_error)
        self.assertFalse(self.output_root.exists())

        for helper in (
            "_build_sugar_mixtures_staged",
            "_measure_sugar_staged",
        ):
            with self.subTest(helper=helper):
                failure = SugarMixturesValidationError(
                    f"{helper}.synthetic",
                    "injected structural failure",
                    "SENTINEL_STRUCTURAL_FAILURE",
                )
                with patch.object(
                    sugar_mixtures_module,
                    "_inspect_sugar_mixtures",
                    return_value=self.inspection,
                ), patch.object(
                    sugar_mixtures_module,
                    helper,
                    side_effect=failure,
                ), self.assertRaises(
                    SugarMixturesValidationError
                ) as captured:
                    sugar_mixtures_module._stage_sugar_mixtures_for_review(
                        self.raw_root,
                        self.output_root,
                    )
                self.assertIs(captured.exception, failure)
                self.assertFalse(self.output_root.exists())

    def test_total_seconds_includes_cleanup_and_gate_evaluation(self):
        self.require_interfaces()
        real_rmtree = shutil.rmtree
        real_gate = getattr(
            sugar_mixtures_module,
            "_sugar_feasibility_failures",
        )

        def delayed_rmtree(path):
            if Path(path).name.startswith(".sugar-mixtures.staging-"):
                time.sleep(0.04)
            return real_rmtree(path)

        def delayed_gate(*args, **kwargs):
            time.sleep(0.04)
            return real_gate(*args, **kwargs)

        with patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            return_value=self.inspection,
        ), patch.object(
            sugar_mixtures_module.shutil,
            "rmtree",
            side_effect=delayed_rmtree,
        ), patch.object(
            sugar_mixtures_module,
            "_sugar_feasibility_failures",
            side_effect=delayed_gate,
        ):
            review = sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertGreaterEqual(review.metrics.total_seconds, 0.08)
        self.assertFalse(self.output_root.exists())

    def test_final_total_threshold_is_evaluated_after_cleanup(self):
        external_staging = self.temporary_root / "external-staging"
        staged = _build_sugar_mixtures_staged(
            self.raw_root,
            external_staging,
            self.inspection,
        )
        base_metrics = sugar_mixtures_module.SugarMixturesRuntimeMetrics(
            total_seconds=1_499.0,
            inspection_seconds=0.0,
            staged_build_seconds=0.0,
            core_build_seconds=0.0,
            views_build_seconds=0.0,
            derived_endmembers_build_seconds=0.0,
            auxiliary_axes_build_seconds=0.0,
            acquisition_json_build_seconds=0.0,
            receipt_build_seconds=0.0,
            validation_seconds=0.0,
            peak_rss_raw=0,
            peak_rss_bytes=0,
            core_output_bytes=staged.core.output_bytes,
            views_output_bytes=staged.views.bytes,
            derived_endmembers_output_bytes=staged.derived_endmembers.bytes,
            auxiliary_axes_output_bytes=staged.auxiliary_axes.bytes,
            acquisition_json_output_bytes=staged.acquisition_json.bytes,
            receipt_bytes=staged.receipt_bytes,
            combined_output_bytes=(
                staged.core.output_bytes
                + staged.views.bytes
                + staged.derived_endmembers.bytes
                + staged.auxiliary_axes.bytes
                + staged.acquisition_json.bytes
                + staged.receipt_bytes
            ),
            source_archive_bytes=self.source.archive_bytes,
            canonical_relevant_source_bytes=30_826,
            compression_ratio_archive=(
                (
                    staged.core.output_bytes
                    + staged.views.bytes
                    + staged.derived_endmembers.bytes
                    + staged.auxiliary_axes.bytes
                    + staged.acquisition_json.bytes
                    + staged.receipt_bytes
                )
                / self.source.archive_bytes
            ),
            compression_ratio_relevant=(
                (
                    staged.core.output_bytes
                    + staged.views.bytes
                    + staged.derived_endmembers.bytes
                    + staged.auxiliary_axes.bytes
                    + staged.acquisition_json.bytes
                    + staged.receipt_bytes
                )
                / 30_826
            ),
            core_axis_groups=1,
            core_lazy_open_seconds=0.0,
            core_representative_record_read_seconds=0.0,
            core_representative_record_id=(
                "low_snr-s049-e01-p03-r01-m01-rep01"
            ),
            views_open_validate_seconds=0.0,
            acquisition_json_stream_validate_seconds=0.0,
            auxiliary_axes_open_validate_seconds=0.0,
            derived_endmembers_open_validate_seconds=0.0,
        )
        counter_values = iter(
            (0.0, 0.0, 0.0, 1_499.0, 1_499.0, 1_500.0)
        )

        with patch.object(
            sugar_mixtures_module.time,
            "perf_counter",
            side_effect=lambda: next(counter_values),
        ), patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            return_value=self.inspection,
        ), patch.object(
            sugar_mixtures_module,
            "_build_sugar_mixtures_staged",
            return_value=staged,
        ), patch.object(
            sugar_mixtures_module,
            "_measure_sugar_staged",
            return_value=base_metrics,
        ), patch.object(
            sugar_mixtures_module,
            "_sugar_feasibility_failures",
            return_value=(),
        ):
            review = sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertEqual(review.metrics.total_seconds, 1_500.0)
        self.assertIn("total_seconds", review.feasibility_failures)

    def test_final_total_consistency_failure_survives_cleanup(self):
        external_staging = self.temporary_root / "consistency-staging"
        staged = _build_sugar_mixtures_staged(
            self.raw_root,
            external_staging,
            self.inspection,
        )
        base_metrics = sugar_mixtures_module.SugarMixturesRuntimeMetrics(
            total_seconds=0.1,
            inspection_seconds=0.0,
            staged_build_seconds=1.0,
            core_build_seconds=1.0,
            views_build_seconds=0.0,
            derived_endmembers_build_seconds=0.0,
            auxiliary_axes_build_seconds=0.0,
            acquisition_json_build_seconds=0.0,
            receipt_build_seconds=0.0,
            validation_seconds=1.0,
            peak_rss_raw=1,
            peak_rss_bytes=1024,
            core_output_bytes=1,
            views_output_bytes=1,
            derived_endmembers_output_bytes=1,
            auxiliary_axes_output_bytes=1,
            acquisition_json_output_bytes=1,
            receipt_bytes=1,
            combined_output_bytes=6,
            source_archive_bytes=12,
            canonical_relevant_source_bytes=8,
            compression_ratio_archive=0.5,
            compression_ratio_relevant=0.75,
            core_axis_groups=1,
            core_lazy_open_seconds=0.0,
            core_representative_record_read_seconds=0.0,
            core_representative_record_id="synthetic",
            views_open_validate_seconds=0.0,
            acquisition_json_stream_validate_seconds=0.0,
            auxiliary_axes_open_validate_seconds=0.0,
            derived_endmembers_open_validate_seconds=0.0,
        )
        counter_values = iter((0.0, 0.0, 0.0, 0.1, 0.1, 0.5))
        with patch.object(
            sugar_mixtures_module.time,
            "perf_counter",
            side_effect=lambda: next(counter_values),
        ), patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            return_value=self.inspection,
        ), patch.object(
            sugar_mixtures_module,
            "_build_sugar_mixtures_staged",
            return_value=staged,
        ), patch.object(
            sugar_mixtures_module,
            "_measure_sugar_staged",
            return_value=base_metrics,
        ), patch.object(
            sugar_mixtures_module,
            "_sugar_feasibility_failures",
            return_value=("total_seconds",),
        ):
            review = sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertEqual(review.metrics.total_seconds, 0.5)
        self.assertIn("total_seconds", review.feasibility_failures)

    def test_final_peak_rss_is_sampled_after_cleanup_and_regated(self):
        base_metrics = sugar_mixtures_module.SugarMixturesRuntimeMetrics(
            total_seconds=1.0,
            inspection_seconds=0.0,
            staged_build_seconds=0.0,
            core_build_seconds=0.0,
            views_build_seconds=0.0,
            derived_endmembers_build_seconds=0.0,
            auxiliary_axes_build_seconds=0.0,
            acquisition_json_build_seconds=0.0,
            receipt_build_seconds=0.0,
            validation_seconds=0.0,
            peak_rss_raw=1,
            peak_rss_bytes=1024,
            core_output_bytes=1,
            views_output_bytes=1,
            derived_endmembers_output_bytes=1,
            auxiliary_axes_output_bytes=1,
            acquisition_json_output_bytes=1,
            receipt_bytes=1,
            combined_output_bytes=6,
            source_archive_bytes=12,
            canonical_relevant_source_bytes=8,
            compression_ratio_archive=0.5,
            compression_ratio_relevant=0.75,
            core_axis_groups=1,
            core_lazy_open_seconds=0.0,
            core_representative_record_read_seconds=0.0,
            core_representative_record_id="synthetic",
            views_open_validate_seconds=0.0,
            acquisition_json_stream_validate_seconds=0.0,
            auxiliary_axes_open_validate_seconds=0.0,
            derived_endmembers_open_validate_seconds=0.0,
        )
        usage_values = iter(
            (
                type("Usage", (), {"ru_maxrss": 1})(),
                type("Usage", (), {"ru_maxrss": 2_097_152})(),
            )
        )
        with patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            return_value=self.inspection,
        ), patch.object(
            sugar_mixtures_module,
            "_measure_sugar_staged",
            return_value=base_metrics,
        ), patch.object(
            sugar_mixtures_module,
            "_sugar_feasibility_failures",
            return_value=(),
        ), patch.object(
            sugar_mixtures_module.resource,
            "getrusage",
            side_effect=lambda *_: next(usage_values),
        ):
            review = sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertEqual(review.metrics.peak_rss_raw, 2_097_152)
        self.assertEqual(review.metrics.peak_rss_bytes, 2_147_483_648)
        self.assertIn("peak_rss_bytes", review.feasibility_failures)

    def test_cleanup_failure_propagates_and_preserves_staging_evidence(self):
        real_rmtree = shutil.rmtree
        preserved = []

        def fail_cleanup(path):
            if Path(path).name.startswith(".sugar-mixtures.staging-"):
                preserved.append(Path(path))
                raise OSError("injected staging cleanup failure")
            return real_rmtree(path)

        with patch.object(
            sugar_mixtures_module,
            "_inspect_sugar_mixtures",
            return_value=self.inspection,
        ), patch.object(
            sugar_mixtures_module.shutil,
            "rmtree",
            side_effect=fail_cleanup,
        ), self.assertRaisesRegex(
            OSError,
            "injected staging cleanup failure",
        ):
            sugar_mixtures_module._stage_sugar_mixtures_for_review(
                self.raw_root,
                self.output_root,
            )
        self.assertEqual(len(preserved), 1)
        self.assertTrue(preserved[0].is_dir())
        self.assertEqual(
            len([path for path in preserved[0].rglob("*") if path.is_file()]),
            10,
        )


class SugarMixturesTransactionTest(unittest.TestCase):
    PUBLICATION_ORDER = (
        "core",
        "views",
        "derived_endmembers",
        "auxiliary_axes",
        "acquisition_json",
        "receipt",
    )
    BACKUP_ORDER = (
        "receipt",
        "acquisition_json",
        "auxiliary_axes",
        "derived_endmembers",
        "views",
        "core",
    )
    ROLLBACK_REMOVE_ORDER = BACKUP_ORDER
    RESTORE_PAYLOAD_ORDER = (
        "core",
        "views",
        "derived_endmembers",
        "auxiliary_axes",
        "acquisition_json",
    )
    ARTIFACT_ROWS = (
        (
            "core",
            "dataset",
            "sugar_mixtures_raman",
            "backup-core",
        ),
        (
            "views",
            "file",
            "sugar_mixtures_raman_views.json",
            "backup-views",
        ),
        (
            "derived_endmembers",
            "file",
            "sugar_mixtures_raman_derived_reference_endmembers.h5",
            "backup-derived-endmembers",
        ),
        (
            "auxiliary_axes",
            "file",
            "sugar_mixtures_raman_auxiliary_axes.h5",
            "backup-auxiliary-axes",
        ),
        (
            "acquisition_json",
            "file",
            "sugar_mixtures_raman_acquisition_json_text.jsonl",
            "backup-acquisition-json",
        ),
        (
            "receipt",
            "file",
            "sugar_mixtures_raman_conversion.json",
            "backup-receipt",
        ),
    )
    FINAL_NAMES = tuple(row[2] for row in ARTIFACT_ROWS)
    JOURNAL_KEYS = {
        "transaction_schema_version",
        "transaction_id",
        "phase",
        "output_root_device",
        "output_root_inode",
        "old_snapshot_sha256",
        "new_snapshot_sha256",
        "publication_order",
        "backup_order",
        "rollback_remove_order",
        "restore_payload_order",
        "completed_backups",
        "completed_publications",
        "completed_removals",
        "completed_restores",
        "current_operation",
        "receipt_installed",
        "receipt_commit_fsynced",
        "artifacts",
    }
    RECOVERY_KEYS = {
        "status",
        "transaction_id",
        "phase",
        "committed_snapshot",
        "recommended_action",
        "recovery_plan_sha256",
        "recovery_required",
        "recovery_path",
        "finals",
        "backups",
        "unresolved",
    }
    PATH_STATE_KEYS = {
        "artifact_id",
        "relative_path",
        "lexical_type",
        "expected_identity_sha256",
        "observed_identity_sha256",
        "relation",
    }
    APPLY_KEYS = {
        "status",
        "transaction_id",
        "action",
        "recovery_plan_sha256",
        "final_state",
        "receipt_state",
        "removed_paths",
        "restored_paths",
        "remaining_recovery_paths",
    }

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.source = create_synthetic_sugar_source(
            self.temporary_root / "source"
        )
        self.inspection = _inspect_sugar_mixtures_source(
            self.source.raw_root,
            synthetic_sugar_contract(self.source),
        )
        self.output_root = self.temporary_root / "output"
        self.output_root.mkdir()
        self.transaction = importlib.import_module(
            "rpe.io.sugar_mixtures_transaction"
        )

    def build_staging(
        self,
        name: str,
        *,
        source=None,
        inspection=None,
    ):
        current_source = self.source if source is None else source
        current_inspection = (
            self.inspection if inspection is None else inspection
        )
        return _build_sugar_mixtures_staged(
            current_source.raw_root,
            self.output_root / name,
            current_inspection,
        )

    def variant_source(self, name: str):
        source = create_synthetic_sugar_source(
            self.temporary_root / name
        )
        excluded = source.role_members["excluded_synthetic"][0]
        entries = read_synthetic_zip_entries(source.archive_path)
        original = next(
            entry.payload for entry in entries if entry.name == excluded
        )
        replacement = bytes(
            ((byte + 1) % 256 for byte in original)
        )
        rewrite_synthetic_zip(
            source,
            lambda _: tuple(
                replace(entry, payload=replacement)
                if entry.name == excluded
                else entry
                for entry in entries
            ),
        )
        contract = refresh_synthetic_sugar_contract(source)
        inspection = _inspect_sugar_mixtures_source(
            source.raw_root,
            contract,
        )
        return source, inspection

    def publish(self, staged, *, overwrite: bool):
        tx = self.transaction
        with tx._lock_sugar_output_root(self.output_root) as lock:
            finals = [
                self.output_root / basename
                for basename in self.FINAL_NAMES
            ]
            present = [
                path.is_symlink() or path.exists()
                for path in finals
            ]
            if not any(present):
                previous = _SugarFinalState("absent", None)
            elif all(present) and overwrite:
                previous = _SugarFinalState(
                    "complete",
                    tx._validate_complete_snapshot(self.output_root),
                )
            else:
                previous = _SugarFinalState(
                    "partial_or_invalid",
                    None,
                )
            return tx._publish_sugar_artifacts(
                lock=lock,
                staging_parent=staged.receipt_path.parent,
                previous_state=previous,
            )

    def final_snapshot(self):
        result = {}
        for _, artifact_type, basename, _ in self.ARTIFACT_ROWS:
            path = self.output_root / basename
            if artifact_type == "dataset":
                for child in sorted(path.iterdir()):
                    result[f"{basename}/{child.name}"] = (
                        child.stat().st_size,
                        file_sha256(child),
                    )
            else:
                result[basename] = (
                    path.stat().st_size,
                    file_sha256(path),
                )
        return result

    def assert_no_staging(self):
        self.assertFalse(
            any(
                path.name.startswith(".sugar-mixtures.staging-")
                for path in self.output_root.iterdir()
            )
        )

    def test_policy_descriptors_orders_and_dataclasses_are_exact(self):
        tx = self.transaction
        spec_payload, spec_document = (
            SugarMixturesReceiptTest.appendix_document("A")
        )
        self.assertEqual(
            tx._PUBLICATION_TRANSACTION_POLICY,
            spec_document,
        )
        self.assertEqual(len(spec_payload), 6_413)
        self.assertEqual(
            hashlib.sha256(
                b"rpe-sugar-publication-transaction-v1\0"
                + spec_payload
            ).hexdigest(),
            "6f31dcdded143ac7b57c48eba4e3af4cf9880e99479fb253b35814b85fabc589",
        )
        self.assertEqual(
            tuple(
                (
                    descriptor.artifact_id,
                    descriptor.artifact_type,
                    descriptor.final_basename,
                    descriptor.backup_basename,
                )
                for descriptor in tx._ARTIFACT_DESCRIPTORS
            ),
            self.ARTIFACT_ROWS,
        )
        self.assertEqual(tx._PUBLICATION_ORDER, self.PUBLICATION_ORDER)
        self.assertEqual(tx._BACKUP_ORDER, self.BACKUP_ORDER)
        self.assertEqual(
            tx._ROLLBACK_REMOVE_ORDER,
            self.ROLLBACK_REMOVE_ORDER,
        )
        self.assertEqual(
            tx._RESTORE_PAYLOAD_ORDER,
            self.RESTORE_PAYLOAD_ORDER,
        )
        self.assertEqual(
            tuple(
                field.name
                for field in fields(_SugarOutputRootLock)
            ),
            ("output_root", "directory_fd", "created_output_root"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(_SugarFinalState)),
            ("classification", "output_snapshot_sha256"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(_SugarPublicationResult)),
            ("mode", "committed", "output_snapshot_sha256", "staging_parent"),
        )
        malformed = list(tx._ARTIFACT_DESCRIPTORS)
        malformed[1] = replace(
            malformed[1],
            final_basename="../escape",
        )
        with self.assertRaises(SugarMixturesValidationError):
            tx._validate_artifact_descriptors(tuple(malformed))
        duplicate = list(tx._ARTIFACT_DESCRIPTORS)
        duplicate[1] = replace(
            duplicate[1],
            backup_basename=duplicate[0].backup_basename,
        )
        with self.assertRaises(SugarMixturesValidationError):
            tx._validate_artifact_descriptors(tuple(duplicate))

    def test_publish_revalidates_supplied_previous_state(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-stale-state")
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._publish_sugar_artifacts(
                    lock=lock,
                    staging_parent=staged.receipt_path.parent,
                    previous_state=_SugarFinalState(
                        "complete",
                        "0" * 64,
                    ),
                )
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
            )
        )

    def test_publish_revalidates_staged_snapshot_before_final_mutation(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-invalid-content")
        receipt = json.loads(staged.receipt_path.read_bytes())
        receipt["adapter_version"] = "counterfeit"
        staged.receipt_path.write_bytes(
            canonical_json_bytes(receipt, newline=True)
        )
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError):
                tx._publish_sugar_artifacts(
                    lock=lock,
                    staging_parent=staged.receipt_path.parent,
                    previous_state=_SugarFinalState("absent", None),
                )
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
            )
        )

    def test_unknown_final_created_after_preflight_is_never_replaced(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-race")
        final = self.output_root / "sugar_mixtures_raman_views.json"
        real_gate = tx._durability_gate

        def create_unknown_after_gate(staging_parent):
            result = real_gate(staging_parent)
            final.write_text("unowned race path\n", encoding="utf-8")
            return result

        with tx._lock_sugar_output_root(self.output_root) as lock:
            with patch.object(
                tx,
                "_durability_gate",
                side_effect=create_unknown_after_gate,
            ), self.assertRaises(SugarMixturesValidationError) as caught:
                tx._publish_sugar_artifacts(
                    lock=lock,
                    staging_parent=staged.receipt_path.parent,
                    previous_state=_SugarFinalState("absent", None),
                )
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )
        self.assertEqual(
            final.read_text(encoding="utf-8"),
            "unowned race path\n",
        )

    def test_unknown_final_created_during_journal_update_is_never_replaced(
        self,
    ):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-journal-race"
        )
        final = self.output_root / "sugar_mixtures_raman_views.json"
        real_update = tx._update_journal
        injected = False

        def create_unknown_during_publication(
            staging_parent,
            journal,
            **updates,
        ):
            nonlocal injected
            result = real_update(staging_parent, journal, **updates)
            if (
                not injected
                and updates.get("phase") == "publication"
                and updates.get("current_operation") == "publish:views"
            ):
                final.write_text(
                    "unowned journal race path\n",
                    encoding="utf-8",
                )
                injected = True
            return result

        with tx._lock_sugar_output_root(self.output_root) as lock:
            with patch.object(
                tx,
                "_update_journal",
                side_effect=create_unknown_during_publication,
            ), self.assertRaises(SugarMixturesValidationError) as caught:
                tx._publish_sugar_artifacts(
                    lock=lock,
                    staging_parent=staged.receipt_path.parent,
                    previous_state=_SugarFinalState("absent", None),
                )
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RESTORE_FAILED",
        )
        self.assertEqual(
            final.read_text(encoding="utf-8"),
            "unowned journal race path\n",
        )
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
                if name != final.name
            )
        )
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue(
            (preserved[0] / "transaction_recovery.json").is_file()
        )

    def test_unknown_final_created_after_initial_journal_is_never_replaced(
        self,
    ):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-initial-journal-race"
        )
        final = self.output_root / "sugar_mixtures_raman_views.json"
        real_write = tx._write_journal

        def create_unknown_after_journal(
            staging_parent,
            journal,
            **kwargs,
        ):
            result = real_write(staging_parent, journal, **kwargs)
            final.write_text(
                "unowned initial-journal race path\n",
                encoding="utf-8",
            )
            return result

        with tx._lock_sugar_output_root(self.output_root) as lock:
            with patch.object(
                tx,
                "_write_journal",
                side_effect=create_unknown_after_journal,
            ), self.assertRaises(SugarMixturesValidationError) as caught:
                tx._publish_sugar_artifacts(
                    lock=lock,
                    staging_parent=staged.receipt_path.parent,
                    previous_state=_SugarFinalState("absent", None),
                )
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RESTORE_FAILED",
        )
        self.assertEqual(
            final.read_text(encoding="utf-8"),
            "unowned initial-journal race path\n",
        )
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
                if name != final.name
            )
        )
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue(
            (preserved[0] / "transaction_recovery.json").is_file()
        )

    def test_output_root_replaced_after_publication_journal_blocks_rename(
        self,
    ):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-publication-root-race"
        )
        moved = self.temporary_root / "moved-publication-output"
        sentinel = self.output_root / "replacement-sentinel.txt"
        real_update = tx._update_journal
        replaced = False

        def replace_root_after_publication_journal(
            staging_parent,
            journal,
            **updates,
        ):
            nonlocal replaced
            result = real_update(staging_parent, journal, **updates)
            if (
                not replaced
                and updates.get("phase") == "publication"
                and updates.get("current_operation") == "publish:core"
            ):
                self.output_root.rename(moved)
                self.output_root.mkdir()
                sentinel.write_text("replacement root\n", encoding="utf-8")
                replaced = True
            return result

        with tx._lock_sugar_output_root(self.output_root) as lock:
            with patch.object(
                tx,
                "_update_journal",
                side_effect=replace_root_after_publication_journal,
            ), self.assertRaises(SugarMixturesValidationError) as caught:
                tx._publish_sugar_artifacts(
                    lock=lock,
                    staging_parent=staged.receipt_path.parent,
                    previous_state=_SugarFinalState("absent", None),
                )
        self.assertEqual(caught.exception.code, "OUTPUT_ROOT_REPLACED")
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "replacement root\n",
        )
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
            )
        )
        moved_staging = moved / staged.receipt_path.parent.name
        self.assertTrue(
            (moved_staging / "transaction_recovery.json").is_file()
        )

    def test_output_root_replaced_after_backup_journal_blocks_rename(self):
        tx = self.transaction
        initial = self.build_staging(
            ".sugar-mixtures.staging-backup-root-old"
        )
        self.publish(initial, overwrite=False)
        source, inspection = self.variant_source("backup-root-source")
        staged = self.build_staging(
            ".sugar-mixtures.staging-backup-root-new",
            source=source,
            inspection=inspection,
        )
        moved = self.temporary_root / "moved-backup-output"
        sentinel = self.output_root / "replacement-sentinel.txt"
        real_update = tx._update_journal
        replaced = False

        def replace_root_after_backup_journal(
            staging_parent,
            journal,
            **updates,
        ):
            nonlocal replaced
            result = real_update(staging_parent, journal, **updates)
            if (
                not replaced
                and updates.get("phase") == "backup"
                and updates.get("current_operation") == "backup:receipt"
            ):
                self.output_root.rename(moved)
                self.output_root.mkdir()
                sentinel.write_text("replacement root\n", encoding="utf-8")
                replaced = True
            return result

        with patch.object(
            tx,
            "_update_journal",
            side_effect=replace_root_after_backup_journal,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=True)
        self.assertEqual(caught.exception.code, "OUTPUT_ROOT_REPLACED")
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "replacement root\n",
        )
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
            )
        )
        self.assertTrue(
            all((moved / name).exists() for name in self.FINAL_NAMES)
        )
        moved_staging = moved / staged.receipt_path.parent.name
        self.assertTrue(
            (moved_staging / "transaction_recovery.json").is_file()
        )

    def test_output_root_replaced_before_rollback_removal_blocks_mutation(
        self,
    ):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-rollback-root-race"
        )
        moved = self.temporary_root / "moved-rollback-output"
        sentinel = self.output_root / "replacement-sentinel.txt"

        def replace_root_then_fail_commit(*args, **kwargs):
            self.output_root.rename(moved)
            self.output_root.mkdir()
            sentinel.write_text("replacement root\n", encoding="utf-8")
            raise OSError("injected commit failure after root replacement")

        with patch.object(
            tx,
            "_commit_directories",
            side_effect=replace_root_then_fail_commit,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=False)
        self.assertEqual(caught.exception.code, "OUTPUT_ROOT_REPLACED")
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "replacement root\n",
        )
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
            )
        )
        self.assertTrue(
            all((moved / name).exists() for name in self.FINAL_NAMES)
        )
        moved_staging = moved / staged.receipt_path.parent.name
        self.assertTrue(
            (moved_staging / "transaction_recovery.json").is_file()
        )

    def test_output_root_replaced_during_initial_journal_write_is_stable(
        self,
    ):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-initial-journal-root-race"
        )
        moved = self.temporary_root / "moved-initial-journal-output"
        sentinel = self.output_root / "replacement-sentinel.txt"
        real_write = tx._write_journal_temp
        replaced = False

        def replace_root_after_temp_write(path, payload):
            nonlocal replaced
            result = real_write(path, payload)
            if not replaced:
                self.output_root.rename(moved)
                self.output_root.mkdir()
                sentinel.write_text(
                    "replacement root\n",
                    encoding="utf-8",
                )
                replaced = True
            return result

        with patch.object(
            tx,
            "_write_journal_temp",
            side_effect=replace_root_after_temp_write,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=False)
        self.assertEqual(caught.exception.code, "OUTPUT_ROOT_REPLACED")
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "replacement root\n",
        )
        moved_staging = moved / staged.receipt_path.parent.name
        self.assertTrue((moved_staging / ".transaction_recovery.tmp").is_file())
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
            )
        )

    def test_output_root_replaced_during_completed_journal_is_stable(self):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-completed-journal-root-race"
        )
        moved = self.temporary_root / "moved-completed-journal-output"
        sentinel = self.output_root / "replacement-sentinel.txt"
        real_write = tx._write_journal_temp
        replaced = False

        def replace_root_after_core_completion(path, payload):
            nonlocal replaced
            result = real_write(path, payload)
            journal = json.loads(payload)
            if (
                not replaced
                and journal["completed_publications"] == ["core"]
            ):
                self.output_root.rename(moved)
                self.output_root.mkdir()
                sentinel.write_text(
                    "replacement root\n",
                    encoding="utf-8",
                )
                replaced = True
            return result

        with patch.object(
            tx,
            "_write_journal_temp",
            side_effect=replace_root_after_core_completion,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=False)
        self.assertEqual(caught.exception.code, "OUTPUT_ROOT_REPLACED")
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "replacement root\n",
        )
        moved_staging = moved / staged.receipt_path.parent.name
        self.assertTrue((moved_staging / ".transaction_recovery.tmp").is_file())
        self.assertTrue((moved / "sugar_mixtures_raman").is_dir())
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
            )
        )

    def test_output_root_replaced_between_commit_fsyncs_is_stable(self):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-commit-fsync-root-race"
        )
        moved = self.temporary_root / "moved-commit-fsync-output"
        sentinel = self.output_root / "replacement-sentinel.txt"
        real_fsync = tx._fsync_directory
        staging_fsyncs_after_receipt = 0

        def replace_root_after_commit_staging_fsync(path):
            nonlocal staging_fsyncs_after_receipt
            path = Path(path)
            result = real_fsync(path)
            if (
                (
                    self.output_root
                    / "sugar_mixtures_raman_conversion.json"
                ).is_file()
                and path == staged.receipt_path.parent
            ):
                staging_fsyncs_after_receipt += 1
                if staging_fsyncs_after_receipt == 3:
                    self.output_root.rename(moved)
                    self.output_root.mkdir()
                    sentinel.write_text(
                        "replacement root\n",
                        encoding="utf-8",
                    )
            return result

        with patch.object(
            tx,
            "_fsync_directory",
            side_effect=replace_root_after_commit_staging_fsync,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=False)
        self.assertEqual(caught.exception.code, "OUTPUT_ROOT_REPLACED")
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "replacement root\n",
        )
        moved_staging = moved / staged.receipt_path.parent.name
        self.assertTrue(
            (moved_staging / "transaction_recovery.json").is_file()
        )
        self.assertTrue(
            (moved / "sugar_mixtures_raman_conversion.json").is_file()
        )

    def test_output_root_replaced_during_committed_journal_is_stable(self):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-committed-journal-root-race"
        )
        moved = self.temporary_root / "moved-committed-journal-output"
        sentinel = self.output_root / "replacement-sentinel.txt"
        real_write = tx._write_journal_temp
        replaced = False

        def replace_root_after_committed_temp(path, payload):
            nonlocal replaced
            result = real_write(path, payload)
            journal = json.loads(payload)
            if not replaced and journal["phase"] == "committed":
                self.output_root.rename(moved)
                self.output_root.mkdir()
                sentinel.write_text(
                    "replacement root\n",
                    encoding="utf-8",
                )
                replaced = True
            return result

        with patch.object(
            tx,
            "_write_journal_temp",
            side_effect=replace_root_after_committed_temp,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=False)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_COMMITTED_CLEANUP_FAILED",
        )
        self.assertIn("OUTPUT_ROOT_REPLACED", caught.exception.reason)
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "replacement root\n",
        )
        moved_staging = moved / staged.receipt_path.parent.name
        self.assertTrue((moved_staging / ".transaction_recovery.tmp").is_file())
        self.assertTrue(
            (moved / "sugar_mixtures_raman_conversion.json").is_file()
        )

    def test_unknown_old_final_during_backup_never_restores_receipt(self):
        tx = self.transaction
        initial = self.build_staging(
            ".sugar-mixtures.staging-backup-race-old"
        )
        self.publish(initial, overwrite=False)
        source, inspection = self.variant_source("backup-race-source")
        staged = self.build_staging(
            ".sugar-mixtures.staging-backup-race-new",
            source=source,
            inspection=inspection,
        )
        final = self.output_root / "sugar_mixtures_raman_views.json"
        real_update = tx._update_journal
        injected = False

        def replace_old_final_during_backup(
            staging_parent,
            journal,
            **updates,
        ):
            nonlocal injected
            result = real_update(staging_parent, journal, **updates)
            if (
                not injected
                and updates.get("phase") == "backup"
                and updates.get("current_operation") == "backup:views"
            ):
                final.write_text(
                    "unknown old-final race path\n",
                    encoding="utf-8",
                )
                injected = True
            return result

        with patch.object(
            tx,
            "_update_journal",
            side_effect=replace_old_final_during_backup,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RESTORE_FAILED",
        )
        self.assertEqual(
            final.read_text(encoding="utf-8"),
            "unknown old-final race path\n",
        )
        self.assertFalse(
            (
                self.output_root
                / "sugar_mixtures_raman_conversion.json"
            ).exists()
        )
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue((preserved[0] / "backup-receipt").is_file())

    def test_unknown_receipt_created_before_restore_is_never_replaced(self):
        tx = self.transaction
        initial = self.build_staging(
            ".sugar-mixtures.staging-receipt-race-old"
        )
        self.publish(initial, overwrite=False)
        source, inspection = self.variant_source("receipt-race-source")
        staged = self.build_staging(
            ".sugar-mixtures.staging-receipt-race-new",
            source=source,
            inspection=inspection,
        )
        receipt = (
            self.output_root / "sugar_mixtures_raman_conversion.json"
        )
        payload_finals = [
            self.output_root / basename
            for artifact_id, _, basename, _ in self.ARTIFACT_ROWS
            if artifact_id != "receipt"
        ]
        real_assert = tx._assert_lock_identity
        rollback_started = False
        injected = False

        def fail_commit(*args, **kwargs):
            nonlocal rollback_started
            rollback_started = True
            raise OSError("injected commit failure")

        def create_unknown_before_receipt_restore(lock):
            nonlocal injected
            result = real_assert(lock)
            if (
                rollback_started
                and not injected
                and not receipt.exists()
                and all(path.exists() for path in payload_finals)
                and (staged.receipt_path.parent / "backup-receipt").is_file()
            ):
                receipt.write_text(
                    "unknown receipt restore race\n",
                    encoding="utf-8",
                )
                injected = True
            return result

        with patch.object(
            tx,
            "_commit_directories",
            side_effect=fail_commit,
        ), patch.object(
            tx,
            "_assert_lock_identity",
            side_effect=create_unknown_before_receipt_restore,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RESTORE_FAILED",
        )
        self.assertEqual(
            receipt.read_text(encoding="utf-8"),
            "unknown receipt restore race\n",
        )
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue((preserved[0] / "backup-receipt").is_file())

    def test_unknown_payload_created_before_restore_is_never_replaced(self):
        tx = self.transaction
        initial = self.build_staging(
            ".sugar-mixtures.staging-payload-race-old"
        )
        self.publish(initial, overwrite=False)
        source, inspection = self.variant_source("payload-race-source")
        staged = self.build_staging(
            ".sugar-mixtures.staging-payload-race-new",
            source=source,
            inspection=inspection,
        )
        final = self.output_root / "sugar_mixtures_raman_views.json"
        receipt = (
            self.output_root / "sugar_mixtures_raman_conversion.json"
        )
        real_assert = tx._assert_lock_identity
        rollback_started = False
        injected = False
        core_final = self.output_root / "sugar_mixtures_raman"
        backup_core = staged.receipt_path.parent / "backup-core"
        backup_views = staged.receipt_path.parent / "backup-views"

        def fail_commit(*args, **kwargs):
            nonlocal rollback_started
            rollback_started = True
            raise OSError("injected commit failure")

        def create_unknown_before_payload_restore(lock):
            nonlocal injected
            result = real_assert(lock)
            if (
                rollback_started
                and not injected
                and core_final.is_dir()
                and not backup_core.exists()
                and not final.exists()
                and backup_views.is_file()
            ):
                final.write_text(
                    "unknown payload restore race\n",
                    encoding="utf-8",
                )
                injected = True
            return result

        with patch.object(
            tx,
            "_commit_directories",
            side_effect=fail_commit,
        ), patch.object(
            tx,
            "_assert_lock_identity",
            side_effect=create_unknown_before_payload_restore,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RESTORE_FAILED",
        )
        self.assertEqual(
            final.read_text(encoding="utf-8"),
            "unknown payload restore race\n",
        )
        self.assertFalse(receipt.exists())
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue((preserved[0] / "backup-receipt").is_file())
        self.assertTrue((preserved[0] / "backup-views").is_file())

    def test_lock_contention_preflight_states_and_root_identity(self):
        tx = self.transaction
        with tx._lock_sugar_output_root(self.output_root) as lock:
            self.assertIsInstance(lock, _SugarOutputRootLock)
            self.assertEqual(
                tx._preflight_sugar_final_state(
                    lock,
                    overwrite=False,
                ),
                _SugarFinalState("absent", None),
            )
            with self.assertRaises(SugarMixturesValidationError) as caught:
                with tx._lock_sugar_output_root(self.output_root):
                    pass
            self.assertEqual(caught.exception.code, "PUBLICATION_LOCKED")

        dangling = self.output_root / self.FINAL_NAMES[0]
        dangling.symlink_to(self.temporary_root / "missing")
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=False)
            self.assertEqual(caught.exception.code, "OUTPUT_EXISTS")
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=True)
            self.assertEqual(
                caught.exception.code,
                "PUBLICATION_RECOVERY_REQUIRED",
            )
        dangling.unlink()

        staged = self.build_staging(".sugar-mixtures.staging-first")
        self.publish(staged, overwrite=False)
        with tx._lock_sugar_output_root(self.output_root) as lock:
            complete = tx._preflight_sugar_final_state(
                lock,
                overwrite=True,
            )
            self.assertEqual(complete.classification, "complete")
            self.assertRegex(
                complete.output_snapshot_sha256,
                r"^[0-9a-f]{64}$",
            )
            original = self.output_root
            moved = self.temporary_root / "moved-output"
            original.rename(moved)
            original.mkdir()
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._assert_lock_identity(lock)
            self.assertEqual(
                caught.exception.code,
                "OUTPUT_ROOT_REPLACED",
            )

    def test_unsupported_flock_is_stable_and_removes_new_root(self):
        tx = self.transaction
        root = self.temporary_root / "unsupported-flock-root"
        calls = 0

        def unsupported_flock(_descriptor, _operation):
            nonlocal calls
            calls += 1
            raise OSError(errno.ENOSYS, "flock is unsupported")

        caught = None
        try:
            with patch.object(
                tx.fcntl,
                "flock",
                side_effect=unsupported_flock,
            ):
                with tx._lock_sugar_output_root(root):
                    self.fail("unsupported flock acquired a lock")
        except BaseException as error:
            caught = error
        self.assertIsInstance(caught, SugarMixturesValidationError)
        self.assertEqual(
            caught.code,
            "PUBLICATION_PLATFORM_UNSUPPORTED",
        )
        self.assertEqual(calls, 1)
        self.assertFalse(root.exists())

    def test_new_lock_context_never_removes_replacement_root(self):
        tx = self.transaction
        root = self.temporary_root / "replace-new-lock-root"
        moved = self.temporary_root / "moved-new-lock-root"
        with self.assertRaises(SugarMixturesValidationError) as caught:
            with tx._lock_sugar_output_root(root):
                root.rename(moved)
                root.mkdir()
        self.assertEqual(caught.exception.code, "OUTPUT_ROOT_REPLACED")
        self.assertTrue(root.is_dir())
        self.assertTrue(moved.is_dir())

    def test_open_failure_never_removes_replaced_new_root(self):
        tx = self.transaction
        root = self.temporary_root / "replace-open-failure-root"
        moved = self.temporary_root / "moved-open-failure-root"

        def replace_root_then_fail_open(_path, _flags):
            root.rename(moved)
            root.mkdir()
            raise OSError(errno.EIO, "injected directory open failure")

        with patch.object(
            tx.os,
            "open",
            side_effect=replace_root_then_fail_open,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            with tx._lock_sugar_output_root(root):
                self.fail("failed directory open acquired a lock")
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_PLATFORM_UNSUPPORTED",
        )
        self.assertTrue(root.is_dir())
        self.assertTrue(moved.is_dir())

    def test_lock_root_and_platform_open_fail_closed(self):
        tx = self.transaction
        invalid_file = self.temporary_root / "invalid-output-file"
        invalid_file.write_text("not a directory\n", encoding="utf-8")
        invalid_symlink = self.temporary_root / "invalid-output-symlink"
        invalid_symlink.symlink_to(invalid_file)
        for root in (invalid_file, invalid_symlink):
            with self.subTest(root=root.name):
                with self.assertRaises(
                    SugarMixturesValidationError
                ) as caught:
                    with tx._lock_sugar_output_root(root):
                        self.fail("invalid output root acquired a lock")
                self.assertEqual(
                    caught.exception.code,
                    "PUBLICATION_PLATFORM_UNSUPPORTED",
                )

        nofollow_root = self.temporary_root / "no-nofollow-root"
        with patch.object(tx.os, "O_NOFOLLOW", None):
            with self.assertRaises(
                SugarMixturesValidationError
            ) as caught:
                with tx._lock_sugar_output_root(nofollow_root):
                    self.fail("missing O_NOFOLLOW acquired a lock")
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_PLATFORM_UNSUPPORTED",
        )
        self.assertFalse(nofollow_root.exists())

        open_root = self.temporary_root / "open-failure-root"
        with patch.object(
            tx.os,
            "open",
            side_effect=OSError(errno.EIO, "injected directory open failure"),
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            with tx._lock_sugar_output_root(open_root):
                self.fail("failed directory open acquired a lock")
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_PLATFORM_UNSUPPORTED",
        )
        self.assertFalse(open_root.exists())

    def test_any_preserved_staging_and_corrupt_complete_state_block(self):
        tx = self.transaction
        preserved = self.output_root / ".sugar-mixtures.staging-preserved"
        preserved.mkdir()
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )
        preserved.rmdir()

        staged = self.build_staging(".sugar-mixtures.staging-corrupt")
        self.publish(staged, overwrite=False)
        receipt = self.output_root / "sugar_mixtures_raman_conversion.json"
        receipt.write_bytes(receipt.read_bytes() + b" ")
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )

    def test_complete_state_rejects_companion_logical_corruption(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-logical")
        self.publish(staged, overwrite=False)
        auxiliary = (
            self.output_root
            / "sugar_mixtures_raman_auxiliary_axes.h5"
        )
        with h5py.File(auxiliary, "r+") as artifact:
            artifact["axes"]["wavelength_nm"][0] += 1
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )

    def test_first_publication_overwrite_noop_orders_and_journal(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-first")
        expected_hashes = dict(staged.output_hashes)
        expected_snapshot_sha256 = (
            tx._output_snapshot_sha256_from_hashes(
                expected_hashes,
                staging_parent=staged.receipt_path.parent,
            )
        )
        operations = []
        real_replace = os.replace

        def recording_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.parent == staged.receipt_path.parent
                and destination_path.parent == self.output_root
                and destination_path.name in self.FINAL_NAMES
            ):
                operations.append(("publish", destination_path.name))
            return real_replace(source, destination)

        with patch.object(tx.os, "replace", side_effect=recording_replace):
            result = self.publish(staged, overwrite=False)
        self.assertEqual(
            result,
            _SugarPublicationResult(
                mode="first_publication",
                committed=True,
                output_snapshot_sha256=expected_snapshot_sha256,
                staging_parent=None,
            ),
        )
        self.assertEqual(
            operations,
            [
                ("publish", row[2])
                for row in self.ARTIFACT_ROWS
            ],
        )
        self.assertEqual(
            self.final_snapshot(),
            {
                name: (
                    (
                        self.output_root
                        / name
                    ).stat().st_size,
                    digest,
                )
                for name, digest in expected_hashes.items()
            },
        )
        self.assert_no_staging()

        identical = self.build_staging(
            ".sugar-mixtures.staging-identical"
        )
        before = {
            name: (
                path.stat().st_ino,
                path.stat().st_mtime_ns,
            )
            for name in self.FINAL_NAMES
            for path in (self.output_root / name,)
        }
        result = self.publish(identical, overwrite=True)
        self.assertEqual(result.mode, "identical_noop")
        self.assertTrue(result.committed)
        self.assertEqual(
            before,
            {
                name: (
                    path.stat().st_ino,
                    path.stat().st_mtime_ns,
                )
                for name in self.FINAL_NAMES
                for path in (self.output_root / name,)
            },
        )
        self.assert_no_staging()

        variant_source, variant_inspection = self.variant_source("variant")
        changed = self.build_staging(
            ".sugar-mixtures.staging-overwrite",
            source=variant_source,
            inspection=variant_inspection,
        )
        backup_operations = []
        publish_operations = []
        real_replace = os.replace

        def recording_overwrite(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.parent == self.output_root
                and destination_path.parent == changed.receipt_path.parent
                and destination_path.name.startswith("backup-")
            ):
                backup_operations.append(source_path.name)
            if (
                source_path.parent == changed.receipt_path.parent
                and destination_path.parent == self.output_root
                and destination_path.name in self.FINAL_NAMES
            ):
                publish_operations.append(destination_path.name)
            return real_replace(source, destination)

        with patch.object(tx.os, "replace", side_effect=recording_overwrite):
            result = self.publish(changed, overwrite=True)
        self.assertEqual(result.mode, "overwrite")
        self.assertEqual(
            backup_operations,
            [
                dict(
                    (row[0], row[2]) for row in self.ARTIFACT_ROWS
                )[artifact_id]
                for artifact_id in self.BACKUP_ORDER
            ],
        )
        self.assertEqual(
            publish_operations,
            [row[2] for row in self.ARTIFACT_ROWS],
        )
        self.assert_no_staging()

    def test_identical_noop_root_replacement_preserves_staging(self):
        tx = self.transaction
        initial = self.build_staging(
            ".sugar-mixtures.staging-noop-root-old"
        )
        self.publish(initial, overwrite=False)
        staged = self.build_staging(
            ".sugar-mixtures.staging-noop-root-new"
        )
        moved = self.temporary_root / "moved-noop-root"
        real_state = tx._actual_final_state_locked
        calls = 0

        def replace_root_after_second_state(lock):
            nonlocal calls
            calls += 1
            state = real_state(lock)
            if calls == 2:
                self.output_root.rename(moved)
                self.output_root.mkdir()
            return state

        with tx._lock_sugar_output_root(self.output_root) as lock:
            previous = _SugarFinalState(
                "complete",
                tx._validate_complete_snapshot(self.output_root),
            )
            with patch.object(
                tx,
                "_actual_final_state_locked",
                side_effect=replace_root_after_second_state,
            ), self.assertRaises(SugarMixturesValidationError) as caught:
                tx._publish_sugar_artifacts(
                    lock=lock,
                    staging_parent=staged.receipt_path.parent,
                    previous_state=previous,
                )
        self.assertEqual(caught.exception.code, "OUTPUT_ROOT_REPLACED")
        self.assertTrue(self.output_root.is_dir())
        self.assertTrue(
            (moved / staged.receipt_path.parent.name).is_dir()
        )

    def test_receipt_commit_order_is_replace_journal_then_two_directory_fsyncs(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-order")
        events = []
        real_replace = tx.os.replace
        real_update = tx._update_journal
        real_fsync = tx._fsync_directory

        def record_replace(source, destination):
            if (
                Path(source).name
                == "sugar_mixtures_raman_conversion.json"
                and Path(destination).parent == self.output_root
            ):
                events.append("replace_receipt")
            return real_replace(source, destination)

        def record_update(staging_parent, journal, **updates):
            if updates.get("receipt_installed") is True:
                events.append("journal_receipt_installed")
            return real_update(staging_parent, journal, **updates)

        def record_fsync(path):
            path = Path(path)
            if events and events[-1] == "journal_receipt_installed":
                events.append(
                    "fsync_staging"
                    if path.name.startswith(".sugar-mixtures.staging-")
                    else "fsync_output"
                )
            elif events and events[-1] == "fsync_staging":
                events.append("fsync_output")
            return real_fsync(path)

        with patch.object(
            tx.os,
            "replace",
            side_effect=record_replace,
        ), patch.object(
            tx,
            "_update_journal",
            side_effect=record_update,
        ), patch.object(
            tx,
            "_fsync_directory",
            side_effect=record_fsync,
        ):
            self.publish(staged, overwrite=False)
        start = events.index("replace_receipt")
        self.assertEqual(
            tuple(events[start : start + 4]),
            (
                "replace_receipt",
                "journal_receipt_installed",
                "fsync_staging",
                "fsync_output",
            ),
        )

    def test_each_rename_updates_journal_before_both_directory_fsyncs(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-rename-order")
        events = []
        real_replace = tx.os.replace
        real_update = tx._update_journal
        real_fsync_pair = tx._fsync_rename_directories

        def record_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                source_path.parent == staged.receipt_path.parent
                and destination_path.parent == self.output_root
                and destination_path.name in self.FINAL_NAMES
            ):
                events.append(f"replace:{destination_path.name}")
            return real_replace(source, destination)

        def record_update(staging_parent, journal, **updates):
            completed = updates.get("completed_publications")
            if completed:
                events.append(f"journal:{completed[-1]}")
            return real_update(staging_parent, journal, **updates)

        def record_fsync_pair(staging_parent, output_root, **kwargs):
            events.append("fsync:staging")
            events.append("fsync:output")
            return real_fsync_pair(
                staging_parent,
                output_root,
                **kwargs,
            )

        with patch.object(
            tx.os,
            "replace",
            side_effect=record_replace,
        ), patch.object(
            tx,
            "_update_journal",
            side_effect=record_update,
        ), patch.object(
            tx,
            "_fsync_rename_directories",
            side_effect=record_fsync_pair,
        ):
            self.publish(staged, overwrite=False)
        first = events.index("replace:sugar_mixtures_raman")
        self.assertEqual(
            events[first : first + 4],
            [
                "replace:sugar_mixtures_raman",
                "journal:core",
                "fsync:staging",
                "fsync:output",
            ],
        )

    def test_newly_created_lock_root_is_removed_on_exception(self):
        tx = self.transaction
        root = self.temporary_root / "new-output-root"
        with self.assertRaisesRegex(RuntimeError, "injected lock body failure"):
            with tx._lock_sugar_output_root(root):
                raise RuntimeError("injected lock body failure")
        self.assertFalse(root.exists())

        shared_root = self.temporary_root / "new-reader-root"
        with self.assertRaises(SugarMixturesValidationError):
            tx._read_sugar_snapshot_locked(
                shared_root,
                lambda document: document,
            )
        self.assertFalse(shared_root.exists())

    def test_complete_state_rejects_internal_contract_digest_drift(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-digest")
        self.publish(staged, overwrite=False)
        receipt_path = (
            self.output_root / "sugar_mixtures_raman_conversion.json"
        )
        receipt = json.loads(receipt_path.read_bytes())
        receipt["source_contract"]["release"]["title"] = "counterfeit"
        receipt_path.write_bytes(canonical_json_bytes(receipt, newline=True))
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )

    def test_complete_state_rejects_malformed_canonical_member_binding(self):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-malformed-binding"
        )
        self.publish(staged, overwrite=False)
        receipt_path = (
            self.output_root / "sugar_mixtures_raman_conversion.json"
        )
        receipt = json.loads(receipt_path.read_bytes())
        receipt["source_contract"]["canonical_members"][0] = {}
        receipt["source_contract_sha256"] = hashlib.sha256(
            b"rpe-sugar-receipt-source-contract-v1\0"
            + tx._canonical_json_bytes(receipt["source_contract"])
        ).hexdigest()
        receipt_path.write_bytes(canonical_json_bytes(receipt, newline=True))
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )

    def test_complete_state_rejects_malformed_nested_semantic_contract(self):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-malformed-semantic"
        )
        self.publish(staged, overwrite=False)
        receipt_path = (
            self.output_root / "sugar_mixtures_raman_conversion.json"
        )
        receipt = json.loads(receipt_path.read_bytes())
        receipt["semantic_contract"]["auxiliary_axes"] = {}
        receipt["semantic_contract_sha256"] = hashlib.sha256(
            b"rpe-sugar-receipt-semantic-contract-v1\0"
            + tx._canonical_json_bytes(receipt["semantic_contract"])
        ).hexdigest()
        receipt_path.write_bytes(canonical_json_bytes(receipt, newline=True))
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )

    def test_complete_state_rejects_fixed_receipt_scalar_drift(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-fixed-scalar")
        self.publish(staged, overwrite=False)
        receipt_path = (
            self.output_root / "sugar_mixtures_raman_conversion.json"
        )
        receipt = json.loads(receipt_path.read_bytes())
        receipt["adapter_version"] = "9.9.9"
        receipt_path.write_bytes(canonical_json_bytes(receipt, newline=True))
        with tx._lock_sugar_output_root(self.output_root) as lock:
            with self.assertRaises(SugarMixturesValidationError) as caught:
                tx._preflight_sugar_final_state(lock, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )

    def test_staged_contract_and_durability_fail_before_mutation(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-contract")
        companion = staged.views.path
        companion.unlink()
        companion.symlink_to(self.temporary_root / "missing")
        with tx._lock_sugar_output_root(self.output_root) as lock:
            previous = _SugarFinalState("absent", None)
            with self.assertRaises(SugarMixturesValidationError):
                tx._publish_sugar_artifacts(
                    lock=lock,
                    staging_parent=staged.receipt_path.parent,
                    previous_state=previous,
                )
        self.assertFalse(
            any((self.output_root / name).exists() for name in self.FINAL_NAMES)
        )

        for target in (
            *(
                f"sugar_mixtures_raman/{name}"
                for name in DATASET_FILES
            ),
            "sugar_mixtures_raman_views.json",
            "sugar_mixtures_raman_derived_reference_endmembers.h5",
            "sugar_mixtures_raman_auxiliary_axes.h5",
            "sugar_mixtures_raman_acquisition_json_text.jsonl",
            "sugar_mixtures_raman_conversion.json",
            "sugar_mixtures_raman",
            "staging_parent",
        ):
            with self.subTest(target=target):
                shutil.rmtree(
                    self.output_root / ".sugar-mixtures.staging-contract",
                    ignore_errors=True,
                )
                staged = self.build_staging(
                    ".sugar-mixtures.staging-contract"
                )
                real_fsync = tx._fsync_transaction_path

                def fail_fsync(path, *, label):
                    if label == target:
                        raise OSError(f"injected fsync failure {target}")
                    return real_fsync(path, label=label)

                with tx._lock_sugar_output_root(self.output_root) as lock:
                    previous = _SugarFinalState("absent", None)
                    with patch.object(
                        tx,
                        "_fsync_transaction_path",
                        side_effect=fail_fsync,
                    ), self.assertRaisesRegex(
                        OSError,
                        "injected fsync failure",
                    ):
                        tx._publish_sugar_artifacts(
                            lock=lock,
                            staging_parent=staged.receipt_path.parent,
                            previous_state=previous,
                        )
                self.assertFalse(
                    any(
                        (self.output_root / name).exists()
                        for name in self.FINAL_NAMES
                    )
                )

    def test_all_backup_and_publication_rename_failures_roll_back(self):
        tx = self.transaction
        for phase, positions in (
            ("first", self.PUBLICATION_ORDER),
            ("backup", self.BACKUP_ORDER),
            ("overwrite", self.PUBLICATION_ORDER),
        ):
            for target_id in positions:
                with self.subTest(phase=phase, target=target_id):
                    root = self.temporary_root / f"{phase}-{target_id}"
                    root.mkdir()
                    old_output_root = self.output_root
                    self.output_root = root
                    try:
                        old_snapshot = None
                        if phase != "first":
                            initial = self.build_staging(
                                ".sugar-mixtures.staging-initial"
                            )
                            self.publish(initial, overwrite=False)
                            old_snapshot = self.final_snapshot()
                            source, inspection = self.variant_source(
                                f"variant-{phase}-{target_id}"
                            )
                            staged = self.build_staging(
                                ".sugar-mixtures.staging-next",
                                source=source,
                                inspection=inspection,
                            )
                        else:
                            staged = self.build_staging(
                                ".sugar-mixtures.staging-next"
                            )
                        basename = dict(
                            (row[0], row[2])
                            for row in self.ARTIFACT_ROWS
                        )[target_id]
                        backup_name = dict(
                            (row[0], row[3])
                            for row in self.ARTIFACT_ROWS
                        )[target_id]
                        real_replace = os.replace

                        def fail_replace(source, destination):
                            source_path = Path(source)
                            destination_path = Path(destination)
                            if (
                                phase == "backup"
                                and source_path.parent == root
                                and source_path.name == basename
                                and destination_path.name == backup_name
                            ) or (
                                phase != "backup"
                                and source_path.parent
                                == staged.receipt_path.parent
                                and source_path.name == basename
                                and destination_path.parent == root
                            ):
                                raise OSError(
                                    f"injected {phase} failure {target_id}"
                                )
                            return real_replace(source, destination)

                        with patch.object(
                            tx.os,
                            "replace",
                            side_effect=fail_replace,
                        ), self.assertRaisesRegex(
                            OSError,
                            f"injected {phase} failure",
                        ):
                            self.publish(
                                staged,
                                overwrite=phase != "first",
                            )
                        if old_snapshot is None:
                            self.assertFalse(
                                any(
                                    (root / name).exists()
                                    for name in self.FINAL_NAMES
                                )
                            )
                        else:
                            self.assertEqual(
                                self.final_snapshot(),
                                old_snapshot,
                            )
                        self.assert_no_staging()
                    finally:
                        self.output_root = old_output_root

    def test_interrupts_rollback_and_restore_failures_preserve_evidence(self):
        tx = self.transaction
        for exception in (KeyboardInterrupt(), SystemExit(23)):
            for target_id in ("receipt", "core"):
                with self.subTest(
                    exception=type(exception).__name__,
                    target=target_id,
                ):
                    root = self.temporary_root / (
                        f"interrupt-{type(exception).__name__}-{target_id}"
                    )
                    root.mkdir()
                    old_root = self.output_root
                    self.output_root = root
                    try:
                        staged = self.build_staging(
                            ".sugar-mixtures.staging-interrupt"
                        )
                        basename = dict(
                            (row[0], row[2])
                            for row in self.ARTIFACT_ROWS
                        )[target_id]
                        real_replace = os.replace

                        def interrupt_replace(source, destination):
                            if (
                                Path(source).parent
                                == staged.receipt_path.parent
                                and Path(source).name == basename
                                and Path(destination).parent == root
                            ):
                                raise exception
                            return real_replace(source, destination)

                        with patch.object(
                            tx.os,
                            "replace",
                            side_effect=interrupt_replace,
                        ), self.assertRaises(type(exception)):
                            self.publish(staged, overwrite=False)
                        self.assertFalse(
                            any(
                                (root / name).exists()
                                for name in self.FINAL_NAMES
                            )
                        )
                        self.assert_no_staging()
                    finally:
                        self.output_root = old_root

        initial = self.build_staging(".sugar-mixtures.staging-old")
        self.publish(initial, overwrite=False)
        source, inspection = self.variant_source("restore-variant")
        staged = self.build_staging(
            ".sugar-mixtures.staging-restore",
            source=source,
            inspection=inspection,
        )
        real_commit = tx._commit_directories
        real_remove = tx._remove_owned_artifact

        def fail_commit(*args, **kwargs):
            raise OSError("injected commit failure")

        def fail_remove(descriptor, final_path, expected_identity):
            if descriptor.artifact_id == "receipt":
                raise OSError("injected receipt removal failure")
            return real_remove(
                descriptor,
                final_path,
                expected_identity,
            )

        with patch.object(
            tx,
            "_commit_directories",
            side_effect=fail_commit,
        ), patch.object(
            tx,
            "_remove_owned_artifact",
            side_effect=fail_remove,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=True)
        self.assertEqual(caught.exception.code, "PUBLICATION_RESTORE_FAILED")
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue(
            (preserved[0] / "transaction_recovery.json").is_file()
        )
        (self.output_root / "sugar_mixtures_raman_views.json").unlink()
        plan = tx._inspect_sugar_recovery(
            output_root=self.output_root,
            staging_parent=preserved[0],
        )
        self.assertIsInstance(plan, _SugarRecoveryPlan)
        self.assertEqual(set(plan.document), self.RECOVERY_KEYS)
        self.assertEqual(
            plan.document["recommended_action"],
            "restore_old_snapshot",
        )
        self.assertTrue(plan.document["recovery_required"])
        self.assertRegex(plan.recovery_plan_sha256, r"^[0-9a-f]{64}$")

    def test_journal_commit_and_cleanup_failure_recovery(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-cleanup")
        real_cleanup = tx._cleanup_staging

        def fail_cleanup(*args, **kwargs):
            raise OSError("injected postcommit cleanup failure")

        with patch.object(
            tx,
            "_cleanup_staging",
            side_effect=fail_cleanup,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=False)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_COMMITTED_CLEANUP_FAILED",
        )
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ]
        self.assertEqual(len(preserved), 1)
        journal_path = preserved[0] / "transaction_recovery.json"
        journal = json.loads(journal_path.read_bytes())
        self.assertEqual(set(journal), self.JOURNAL_KEYS)
        self.assertTrue(journal["receipt_installed"])
        self.assertTrue(journal["receipt_commit_fsynced"])
        self.assertEqual(
            journal["publication_order"],
            list(self.PUBLICATION_ORDER),
        )
        plan = tx._inspect_sugar_recovery(
            output_root=self.output_root,
            staging_parent=preserved[0],
        )
        self.assertEqual(
            plan.document["status"],
            "committed_cleanup_required",
        )
        self.assertEqual(
            plan.document["recommended_action"],
            "cleanup_committed_snapshot",
        )
        result = tx._apply_sugar_recovery(
            output_root=self.output_root,
            staging_parent=preserved[0],
            action="cleanup_committed_snapshot",
            transaction_id=journal["transaction_id"],
            expected_plan_sha256=plan.recovery_plan_sha256,
        )
        self.assertIsInstance(result, _SugarRecoveryResult)
        self.assertEqual(set(result.document), self.APPLY_KEYS)
        self.assertEqual(result.document["status"], "recovered")
        self.assertEqual(result.document["final_state"], "new_complete")
        self.assertFalse(preserved[0].exists())
        self.assertEqual(len(self.final_snapshot()), 10)
        with self.assertRaises(SugarMixturesValidationError):
            tx._apply_sugar_recovery(
                output_root=self.output_root,
                staging_parent=preserved[0],
                action="cleanup_committed_snapshot",
                transaction_id=journal["transaction_id"],
                expected_plan_sha256="0" * 64,
            )

    def test_recovery_scan_uses_physical_commit_and_device_inode(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-physical")

        def fail_cleanup(*args, **kwargs):
            raise OSError("injected cleanup failure")

        with patch.object(
            tx,
            "_cleanup_staging",
            side_effect=fail_cleanup,
        ), self.assertRaises(SugarMixturesValidationError):
            self.publish(staged, overwrite=False)
        preserved = next(
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        )
        journal_path = preserved / "transaction_recovery.json"
        journal = json.loads(journal_path.read_bytes())
        journal["receipt_commit_fsynced"] = False
        journal_path.write_bytes(canonical_json_bytes(journal, newline=True))
        plan = tx._inspect_sugar_recovery(
            output_root=self.output_root,
            staging_parent=preserved,
        )
        self.assertEqual(
            plan.document["status"],
            "committed_cleanup_required",
        )
        self.assertEqual(
            plan.document["recommended_action"],
            "cleanup_committed_snapshot",
        )

        journal["output_root_inode"] += 1
        journal_path.write_bytes(canonical_json_bytes(journal, newline=True))
        with self.assertRaises(SugarMixturesValidationError) as caught:
            tx._inspect_sugar_recovery(
                output_root=self.output_root,
                staging_parent=preserved,
            )
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RECOVERY_REQUIRED",
        )

    def test_recovery_refuses_old_receipt_when_old_payload_is_unavailable(self):
        tx = self.transaction
        initial = self.build_staging(".sugar-mixtures.staging-old-missing")
        self.publish(initial, overwrite=False)
        source, inspection = self.variant_source("missing-backup-source")
        staged = self.build_staging(
            ".sugar-mixtures.staging-new-missing",
            source=source,
            inspection=inspection,
        )
        real_remove = tx._remove_owned_artifact

        def fail_views_remove(descriptor, final_path, expected_identity):
            if descriptor.artifact_id == "views":
                raise OSError("injected views removal failure")
            return real_remove(descriptor, final_path, expected_identity)

        with patch.object(
            tx,
            "_commit_directories",
            side_effect=OSError("injected commit failure"),
        ), patch.object(
            tx,
            "_remove_owned_artifact",
            side_effect=fail_views_remove,
        ), self.assertRaises(SugarMixturesValidationError):
            self.publish(staged, overwrite=True)
        preserved = next(
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        )
        (self.output_root / "sugar_mixtures_raman_views.json").unlink()
        (preserved / "backup-views").unlink()
        plan = tx._inspect_sugar_recovery(
            output_root=self.output_root,
            staging_parent=preserved,
        )
        self.assertTrue(
            any(
                unresolved["artifact_id"] == "views"
                for unresolved in plan.document["unresolved"]
            )
        )
        with self.assertRaises(SugarMixturesValidationError):
            tx._apply_sugar_recovery(
                output_root=self.output_root,
                staging_parent=preserved,
                action="restore_old_snapshot",
                transaction_id=plan.document["transaction_id"],
                expected_plan_sha256=plan.recovery_plan_sha256,
            )
        self.assertTrue((preserved / "backup-receipt").is_file())
        self.assertFalse(
            (
                self.output_root
                / "sugar_mixtures_raman_conversion.json"
            ).exists()
        )

    def test_journal_directory_fsync_failure_preserves_recovery_evidence(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-journal-fsync")
        real_fsync = tx._fsync_journal_directory
        calls = 0

        def fail_after_replace(path):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected journal directory fsync failure")
            return real_fsync(path)

        with patch.object(
            tx,
            "_fsync_journal_directory",
            side_effect=fail_after_replace,
        ), self.assertRaisesRegex(
            OSError,
            "journal directory fsync failure",
        ):
            self.publish(staged, overwrite=False)
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ]
        self.assertEqual(len(preserved), 1)
        self.assertTrue(
            (preserved[0] / "transaction_recovery.json").is_file()
        )
        self.assertFalse(
            any(
                (self.output_root / name).exists()
                for name in self.FINAL_NAMES
            )
        )

    def test_journal_internal_failures_and_commit_fsync_roll_back(self):
        tx = self.transaction
        for helper in (
            "_write_journal_temp",
            "_fsync_journal_temp",
            "_replace_journal",
            "_fsync_journal_directory",
        ):
            with self.subTest(helper=helper):
                shutil.rmtree(
                    self.output_root / ".sugar-mixtures.staging-journal",
                    ignore_errors=True,
                )
                staged = self.build_staging(
                    ".sugar-mixtures.staging-journal"
                )
                with patch.object(
                    tx,
                    helper,
                    side_effect=OSError(f"injected {helper} failure"),
                ), self.assertRaisesRegex(OSError, f"injected {helper}"):
                    self.publish(staged, overwrite=False)
                self.assertFalse(
                    any(
                        (self.output_root / name).exists()
                        for name in self.FINAL_NAMES
                    )
                )
                if helper == "_fsync_journal_directory":
                    self.assertTrue(
                        any(
                            path.name.startswith(
                                ".sugar-mixtures.staging-"
                            )
                            and (
                                path / "transaction_recovery.json"
                            ).is_file()
                            for path in self.output_root.iterdir()
                        )
                    )
                else:
                    self.assertFalse(
                        any(
                            path.name.startswith(
                                ".sugar-mixtures.staging-"
                            )
                            for path in self.output_root.iterdir()
                        )
                    )

        for path in tuple(self.output_root.iterdir()):
            if path.name.startswith(".sugar-mixtures.staging-"):
                shutil.rmtree(path)
        for directory_name in ("staging_parent", "output_root"):
            with self.subTest(commit_fsync=directory_name):
                shutil.rmtree(
                    self.output_root / ".sugar-mixtures.staging-commit",
                    ignore_errors=True,
                )
                staged = self.build_staging(
                    ".sugar-mixtures.staging-commit"
                )
                with patch.object(
                    tx,
                    "_commit_directories",
                    side_effect=OSError(
                        f"injected {directory_name} commit fsync failure"
                    ),
                ), self.assertRaisesRegex(
                    OSError,
                    "commit fsync failure",
                ):
                    self.publish(staged, overwrite=False)
                self.assertFalse(
                    any(
                        (self.output_root / name).exists()
                        for name in self.FINAL_NAMES
                    )
                )

    def test_every_remove_restore_and_old_receipt_failure_preserves_false_receipt_safety(self):
        tx = self.transaction
        for target_id in self.ROLLBACK_REMOVE_ORDER:
            with self.subTest(remove=target_id):
                root = self.temporary_root / f"remove-{target_id}"
                root.mkdir()
                old_root = self.output_root
                self.output_root = root
                try:
                    staged = self.build_staging(
                        ".sugar-mixtures.staging-remove"
                    )
                    real_commit = tx._commit_directories
                    real_remove = tx._remove_owned_artifact

                    def fail_commit(*args, **kwargs):
                        raise OSError("injected commit failure")

                    def fail_remove(descriptor, final_path, expected_identity):
                        if descriptor.artifact_id == target_id:
                            raise OSError(
                                f"injected removal failure {target_id}"
                            )
                        return real_remove(
                            descriptor,
                            final_path,
                            expected_identity,
                        )

                    with patch.object(
                        tx,
                        "_commit_directories",
                        side_effect=fail_commit,
                    ), patch.object(
                        tx,
                        "_remove_owned_artifact",
                        side_effect=fail_remove,
                    ), self.assertRaises(
                        SugarMixturesValidationError
                    ) as caught:
                        self.publish(staged, overwrite=False)
                    self.assertEqual(
                        caught.exception.code,
                        "PUBLICATION_RESTORE_FAILED",
                    )
                    self.assertEqual(
                        target_id != "receipt",
                        not (root / "sugar_mixtures_raman_conversion.json").exists(),
                    )
                    self.assertTrue(
                        any(
                            path.name.startswith(
                                ".sugar-mixtures.staging-"
                            )
                            for path in root.iterdir()
                        )
                    )
                finally:
                    self.output_root = old_root

    def test_overwrite_remove_failure_does_not_overwrite_still_present_new_final(self):
        tx = self.transaction
        initial = self.build_staging(".sugar-mixtures.staging-old-remove")
        self.publish(initial, overwrite=False)
        source, inspection = self.variant_source("remove-overwrite-source")
        staged = self.build_staging(
            ".sugar-mixtures.staging-new-remove",
            source=source,
            inspection=inspection,
        )
        expected_new_views = staged.views.sha256
        real_remove = tx._remove_owned_artifact

        def fail_views_remove(descriptor, final_path, expected_identity):
            if descriptor.artifact_id == "views":
                raise OSError("injected overwrite views removal failure")
            return real_remove(descriptor, final_path, expected_identity)

        with patch.object(
            tx,
            "_commit_directories",
            side_effect=OSError("injected commit failure"),
        ), patch.object(
            tx,
            "_remove_owned_artifact",
            side_effect=fail_views_remove,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=True)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_RESTORE_FAILED",
        )
        final_views = self.output_root / "sugar_mixtures_raman_views.json"
        self.assertEqual(file_sha256(final_views), expected_new_views)
        preserved = next(
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        )
        self.assertTrue((preserved / "backup-views").is_file())
        self.assertFalse(
            (
                self.output_root
                / "sugar_mixtures_raman_conversion.json"
            ).exists()
        )


        for target_id in (*self.RESTORE_PAYLOAD_ORDER, "receipt"):
            with self.subTest(restore=target_id):
                root = self.temporary_root / f"restore-{target_id}"
                root.mkdir()
                old_root = self.output_root
                self.output_root = root
                try:
                    initial = self.build_staging(
                        ".sugar-mixtures.staging-old"
                    )
                    self.publish(initial, overwrite=False)
                    source, inspection = self.variant_source(
                        f"restore-source-{target_id}"
                    )
                    staged = self.build_staging(
                        ".sugar-mixtures.staging-new",
                        source=source,
                        inspection=inspection,
                    )
                    real_replace = os.replace

                    def fail_replace(source_path, destination_path):
                        source_path = Path(source_path)
                        destination_path = Path(destination_path)
                        if (
                            source_path.parent
                            == staged.receipt_path.parent
                            and source_path.name
                            == dict(
                                (row[0], row[3])
                                for row in self.ARTIFACT_ROWS
                            )[target_id]
                            and destination_path.parent == root
                        ):
                            raise OSError(
                                f"injected restore failure {target_id}"
                            )
                        return real_replace(source_path, destination_path)

                    with patch.object(
                        tx,
                        "_commit_directories",
                        side_effect=OSError("injected commit failure"),
                    ), patch.object(
                        tx.os,
                        "replace",
                        side_effect=fail_replace,
                    ), self.assertRaises(
                        SugarMixturesValidationError
                    ) as caught:
                        self.publish(staged, overwrite=True)
                    self.assertEqual(
                        caught.exception.code,
                        "PUBLICATION_RESTORE_FAILED",
                    )
                    if target_id != "receipt":
                        self.assertFalse(
                            (
                                root
                                / "sugar_mixtures_raman_conversion.json"
                            ).exists()
                        )
                        self.assertTrue(
                            (
                                staged.receipt_path.parent
                                / "backup-receipt"
                            ).is_file()
                        )
                finally:
                    self.output_root = old_root

    def test_commit_journal_update_failure_is_postcommit_cleanup_failure(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-commit-journal")
        real_update = tx._update_journal

        def fail_committed_update(staging_parent, journal, **updates):
            if updates.get("phase") == "committed":
                raise OSError("injected committed journal update failure")
            return real_update(staging_parent, journal, **updates)

        with patch.object(
            tx,
            "_update_journal",
            side_effect=fail_committed_update,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=False)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_COMMITTED_CLEANUP_FAILED",
        )
        self.assertEqual(len(self.final_snapshot()), 10)
        self.assertTrue(
            (
                self.output_root
                / "sugar_mixtures_raman_conversion.json"
            ).is_file()
        )
        self.assertTrue(
            any(
                path.name.startswith(".sugar-mixtures.staging-")
                for path in self.output_root.iterdir()
            )
        )

    def test_postcommit_root_replacement_skips_stale_cleanup(self):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-postcommit-root-race"
        )
        moved = self.temporary_root / "moved-postcommit-output"
        real_update = tx._update_journal
        real_cleanup = tx._cleanup_staging
        cleanup_called = False

        def replace_root_after_committed_journal(
            staging_parent,
            journal,
            **updates,
        ):
            result = real_update(staging_parent, journal, **updates)
            if updates.get("phase") == "committed":
                self.output_root.rename(moved)
                self.output_root.mkdir()
            return result

        def record_cleanup(*args, **kwargs):
            nonlocal cleanup_called
            cleanup_called = True
            return real_cleanup(*args, **kwargs)

        with patch.object(
            tx,
            "_update_journal",
            side_effect=replace_root_after_committed_journal,
        ), patch.object(
            tx,
            "_cleanup_staging",
            side_effect=record_cleanup,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            self.publish(staged, overwrite=False)
        self.assertEqual(
            caught.exception.code,
            "PUBLICATION_COMMITTED_CLEANUP_FAILED",
        )
        self.assertIn("OUTPUT_ROOT_REPLACED", caught.exception.reason)
        self.assertFalse(cleanup_called)
        self.assertTrue(self.output_root.is_dir())
        moved_staging = moved / staged.receipt_path.parent.name
        self.assertTrue(
            (moved_staging / "transaction_recovery.json").is_file()
        )

    def test_reader_shared_lock_and_lock_free_double_read(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-reader")
        self.publish(staged, overwrite=False)
        entered = threading.Event()
        release = threading.Event()
        result = []

        def reader_operation(document):
            entered.set()
            release.wait(timeout=5)
            return document["dataset_id"]

        def reader_thread():
            result.append(
                tx._read_sugar_snapshot_locked(
                    self.output_root,
                    reader_operation,
                )
            )

        thread = threading.Thread(target=reader_thread)
        thread.start()
        self.assertTrue(entered.wait(timeout=5))
        with self.assertRaises(SugarMixturesValidationError) as caught:
            with tx._lock_sugar_output_root(self.output_root):
                pass
        self.assertEqual(caught.exception.code, "PUBLICATION_LOCKED")
        release.set()
        thread.join(timeout=5)
        self.assertEqual(result, ["sugar_mixtures_raman"])

        receipt_path = (
            self.output_root / "sugar_mixtures_raman_conversion.json"
        )
        original = receipt_path.read_bytes()

        def mutate_between_reads(document):
            receipt_path.write_bytes(original + b" ")
            return document["dataset_id"]

        with self.assertRaises(SugarMixturesValidationError) as caught:
            tx._read_sugar_snapshot_lock_free(
                self.output_root,
                mutate_between_reads,
            )
        self.assertEqual(caught.exception.code, "READER_SNAPSHOT_CHANGED")
        receipt_path.unlink()
        with self.assertRaises(SugarMixturesValidationError) as caught:
            tx._read_sugar_snapshot_lock_free(
                self.output_root,
                lambda document: document,
            )
        self.assertEqual(caught.exception.code, "SNAPSHOT_UNAVAILABLE")

    def test_lock_free_reader_stable_success_and_second_read_error(self):
        tx = self.transaction
        staged = self.build_staging(
            ".sugar-mixtures.staging-reader-outcomes"
        )
        self.publish(staged, overwrite=False)
        self.assertEqual(
            tx._read_sugar_snapshot_lock_free(
                self.output_root,
                lambda document: document["dataset_id"],
            ),
            "sugar_mixtures_raman",
        )

        receipt_path = (
            self.output_root / "sugar_mixtures_raman_conversion.json"
        )
        real_read_bytes = Path.read_bytes
        operation_complete = False

        def fail_second_receipt_read(path):
            if Path(path) == receipt_path and operation_complete:
                raise OSError("injected second receipt read failure")
            return real_read_bytes(path)

        def complete_operation(document):
            nonlocal operation_complete
            operation_complete = True
            return document["dataset_id"]

        with patch.object(
            Path,
            "read_bytes",
            autospec=True,
            side_effect=fail_second_receipt_read,
        ), self.assertRaises(SugarMixturesValidationError) as caught:
            tx._read_sugar_snapshot_lock_free(
                self.output_root,
                complete_operation,
            )
        self.assertEqual(caught.exception.code, "READER_SNAPSHOT_CHANGED")

    def test_recovery_scan_overrides_stale_journal_and_unknown_is_not_mutated(self):
        tx = self.transaction
        staged = self.build_staging(".sugar-mixtures.staging-unknown")
        with patch.object(
            tx,
            "_commit_directories",
            side_effect=OSError("injected commit failure"),
        ), patch.object(
            tx,
            "_remove_owned_artifact",
            side_effect=OSError("injected removal failure"),
        ), self.assertRaises(SugarMixturesValidationError):
            self.publish(staged, overwrite=False)
        preserved = [
            path
            for path in self.output_root.iterdir()
            if path.name.startswith(".sugar-mixtures.staging-")
        ][0]
        journal_path = preserved / "transaction_recovery.json"
        journal = json.loads(journal_path.read_bytes())
        journal["completed_publications"] = []
        journal_path.write_bytes(canonical_json_bytes(journal, newline=True))
        final = self.output_root / "sugar_mixtures_raman_views.json"
        final.write_text("unknown identity\n", encoding="utf-8")
        unknown_before = file_sha256(final)
        plan = tx._inspect_sugar_recovery(
            output_root=self.output_root,
            staging_parent=preserved,
        )
        self.assertTrue(plan.document["unresolved"])
        self.assertEqual(
            set(plan.document["unresolved"][0]),
            {
                "artifact_id",
                "relative_path",
                "reason",
                "safe_automatic_action",
            },
        )
        with self.assertRaises(SugarMixturesValidationError):
            tx._apply_sugar_recovery(
                output_root=self.output_root,
                staging_parent=preserved,
                action=plan.document["recommended_action"],
                transaction_id=journal["transaction_id"],
                expected_plan_sha256=plan.recovery_plan_sha256,
            )
        self.assertEqual(file_sha256(final), unknown_before)


if __name__ == "__main__":
    unittest.main()
