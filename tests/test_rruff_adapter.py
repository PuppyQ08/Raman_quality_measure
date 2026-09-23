from __future__ import annotations

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
import time
import unittest
import zipfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, fields, replace
from datetime import date
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np
import h5py


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import rpe.io as rpe_io  # noqa: E402
import rpe.io.rruff as rruff_module  # noqa: E402
from rruff_helpers import (  # noqa: E402
    ARCHIVE_NAMES,
    CP1252_RAW_MEMBER,
    CALCITE_RAW_MEMBER,
    PAIRING_REJECTIONS,
    PAIRING_STATISTICS,
    PAIR_ALIGNMENT_KEY,
    PAIR_CROSS_ARCHIVE_KEY,
    PAIR_EXACT_KEY,
    PAIR_EXACT_PROCESSED_MEMBER_ID,
    PAIR_EXACT_RAW_MEMBER_ID,
    PAIR_NO_OVERLAP_KEY,
    PAIR_PROCESSED_SUBSET_KEY,
    PAIR_RAW_SUBSET_KEY,
    PAIR_REJECTED_DUPLICATE_KEY,
    QUARTZ_PROCESSED_MEMBER,
    QUARTZ_RAW_MEMBER,
    ZIP_MEMBER_TIMESTAMP,
    corrupt_archive_member_crc,
    create_synthetic_rruff_pairing_source,
    create_synthetic_rruff_source,
    file_sha256,
    replace_member_payload,
    rewrite_archive,
    synthetic_source_contract,
    synthetic_pairing_contract,
)
from rpe.io.rruff import (  # noqa: E402
    RruffAcceptedMember,
    RruffDatasetSummary,
    RruffInspection,
    _RruffComparisonSummary,
    _RruffInspectionBundle,
    _RruffStagedData,
    _build_rruff_data_staged,
    _build_pairs,
    _compare_rruff_dataset,
    _inspect_rruff,
    _inspect_rruff_bundle,
    inspect_rruff,
    iter_rruff_processed_records,
    iter_rruff_raw_records,
)
from rpe.io.rruff_pairing import (  # noqa: E402
    RruffPairInspection,
    RruffPairMember,
    _axis_relation,
    _canonical_json_bytes,
    _pair_id,
    _validate_rruff_pair_index,
    _write_rruff_pair_index,
)
from rpe.io.rruff_source import (  # noqa: E402
    PRODUCTION_REJECTIONS,
    PRODUCTION_SOURCE_CONTRACT,
    RruffHeaderEntry,
    RruffIgnoredLine,
    RruffValidationError,
    _ArchiveContract,
    _RruffAcquisitionFields,
    _RruffSourceContract,
    _RruffSourceInspection,
    _RruffSourceStatistics,
    _SourceAcceptedMember,
    _inspect_rruff_source,
    _parse_member_bytes,
    _promote_rruff_acquisition,
)
from rpe.io.schema import (  # noqa: E402
    LicenseStatus,
    PreprocessingStatus,
    PreprocessingStep,
    SchemaValidationError,
    Targets,
    validate_record,
)
from rpe.io.store import (  # noqa: E402
    DATASET_FILES,
    DatasetClosedError,
    DatasetValidationError,
    UnifiedDataset,
    validate_dataset,
)


RAW_MEMBER = (
    "Quartz__R000001__Raman__532__0__unoriented__"
    "Raman_Data_Raw__0000000000000000000000000001.txt"
)
PROCESSED_MEMBER = (
    "Quartz__R000001__Raman__532__0__unoriented__"
    "Raman_Data_Processed__0000000000000000000000000002.txt"
)
RAW_AXIS_ID = (
    "9e791ac3d82481c0485151c18e86f2ed41466176a7398b29c452b636743f21f4"
)
SIMPLE_RAW_SHA256 = (
    "a4156253561530a3ce21ad8fc9cf72937a4c863160b78418a11111d075b13528"
)
RRUFF_BUILDER_PATH = ROOT / "tools" / "build_rruff_unified.py"
PRODUCTION_RRUFF_RAW_ROOT = ROOT / "data" / "raw" / "rruff"


def load_rruff_builder_module():
    specification = importlib.util.spec_from_file_location(
        "task9_build_rruff_unified",
        RRUFF_BUILDER_PATH,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("unable to load RRUFF builder")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def member_payload(
    rows: tuple[str, ...],
    *,
    name: str = "Quartz",
    rruff_id: str = "R000001",
    filetype: str = "Raman RAW",
    wavelength: str = "532",
    optional_headers: tuple[str, ...] = (),
    body_prefix: tuple[str, ...] = (),
    encoding: str = "utf-8",
) -> bytes:
    lines = (
        f"##NAMES={name}",
        f"##RRUFFID={rruff_id}",
        f"##FILETYPE={filetype}",
        f"##RAMAN WAVELENGTH={wavelength}",
        f"##URL=https://rruff.info/{rruff_id}",
        *optional_headers,
        "",
        *body_prefix,
        *rows,
        "",
    )
    return "\n".join(lines).encode(encoding)


def parse(
    payload: bytes,
    *,
    source_member: str = RAW_MEMBER,
):
    return _parse_member_bytes(
        "LR-Raman.zip",
        source_member,
        payload,
    )


class RruffMemberParserTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)

    def assert_rejected(
        self,
        expected_code: str,
        payload: bytes,
        *,
        source_member: str = RAW_MEMBER,
    ):
        outcome = parse(payload, source_member=source_member)
        self.assertIsNone(outcome.accepted)
        self.assertIsNotNone(outcome.rejected)
        self.assertEqual(outcome.rejected.rejection_code, expected_code)
        self.assertNotEqual(outcome.rejected.reason, "")
        return outcome

    def test_synthetic_archives_are_byte_deterministic_and_rewritable(self):
        first = create_synthetic_rruff_source(
            self.temporary_root / "first",
        )
        second = create_synthetic_rruff_source(
            self.temporary_root / "second",
        )

        self.assertEqual(tuple(first.archive_sha256), ARCHIVE_NAMES)
        self.assertEqual(
            dict(first.archive_sha256),
            dict(second.archive_sha256),
        )
        self.assertEqual(
            dict(first.archive_bytes),
            dict(second.archive_bytes),
        )
        self.assertEqual(
            dict(first.archive_members),
            dict(second.archive_members),
        )
        self.assertIn(
            QUARTZ_RAW_MEMBER,
            first.archive_members["LR-Raman.zip"],
        )
        self.assertIn(
            QUARTZ_PROCESSED_MEMBER,
            first.archive_members["LR-Raman.zip"],
        )
        self.assertIn(
            CP1252_RAW_MEMBER,
            first.archive_members["fair_unoriented.zip"],
        )
        with self.assertRaises(TypeError):
            first.archive_sha256["LR-Raman.zip"] = "changed"

        for archive_name in ARCHIVE_NAMES:
            with zipfile.ZipFile(first.raw_root / archive_name) as archive:
                for info in archive.infolist():
                    self.assertEqual(info.date_time, ZIP_MEMBER_TIMESTAMP)
                    self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
                    self.assertEqual((info.external_attr >> 16) & 0o777, 0o644)

        original_hashes = dict(first.archive_sha256)

        def add_member(members):
            return [
                *members,
                (
                    "extra__R999999__Raman__532__0__unoriented__"
                    "Raman_Data_RAW__ffffffffffffffffffffffffffff.txt",
                    b"extra\n",
                ),
            ]

        rewritten = rewrite_archive(
            first,
            "poor_unoriented.zip",
            add_member,
        )
        self.assertNotEqual(
            rewritten.archive_sha256["poor_unoriented.zip"],
            original_hashes["poor_unoriented.zip"],
        )
        for archive_name in set(ARCHIVE_NAMES) - {"poor_unoriented.zip"}:
            self.assertEqual(
                rewritten.archive_sha256[archive_name],
                original_hashes[archive_name],
            )

    def test_utf8_comma_member_preserves_identity_arrays_and_statistics(self):
        payload = member_payload(("100,1", "101,2", "102,3"))
        self.assertEqual(hashlib.sha256(payload).hexdigest(), SIMPLE_RAW_SHA256)

        outcome = parse(payload)

        self.assertIsNone(outcome.rejected)
        parsed = outcome.accepted
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.archive, "LR-Raman.zip")
        self.assertEqual(parsed.source_member, RAW_MEMBER)
        self.assertEqual(
            parsed.measurement_key,
            "Quartz__R000001__Raman__532__0__unoriented",
        )
        self.assertEqual(
            parsed.member_id,
            "0000000000000000000000000001",
        )
        self.assertEqual(parsed.member_sha256, SIMPLE_RAW_SHA256)
        self.assertEqual(parsed.member_bytes, 129)
        self.assertEqual(parsed.kind, "raw")
        self.assertEqual(parsed.text_encoding, "utf-8")
        self.assertEqual(
            parsed.header_entries,
            (
                RruffHeaderEntry(1, "NAMES", "Quartz"),
                RruffHeaderEntry(2, "RRUFFID", "R000001"),
                RruffHeaderEntry(3, "FILETYPE", "Raman RAW"),
                RruffHeaderEntry(4, "RAMAN WAVELENGTH", "532"),
                RruffHeaderEntry(
                    5,
                    "URL",
                    "https://rruff.info/R000001",
                ),
            ),
        )
        self.assertEqual(parsed.ignored_lines, ())
        self.assertEqual(parsed.mineral_name, "Quartz")
        self.assertEqual(parsed.rruff_id, "R000001")
        self.assertIsNone(parsed.pin_id)
        self.assertIsNone(parsed.orientation)
        self.assertEqual(parsed.excitation_nm, 532.0)
        self.assertEqual(parsed.source_point_count, 3)
        np.testing.assert_array_equal(
            parsed.source_axis,
            np.array([100.0, 101.0, 102.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(
            parsed.source_intensity,
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
        )
        self.assertFalse(parsed.source_axis.flags.writeable)
        self.assertFalse(parsed.source_intensity.flags.writeable)
        self.assertFalse(parsed.source_axis.flags.owndata)
        self.assertFalse(parsed.source_intensity.flags.owndata)
        with self.assertRaises(ValueError):
            parsed.source_axis.setflags(write=True)
        with self.assertRaises(ValueError):
            parsed.source_intensity.setflags(write=True)
        with self.assertRaises(ValueError):
            parsed.source_axis[0] = 999.0
        with self.assertRaises(ValueError):
            parsed.source_intensity[0] = 999.0
        self.assertEqual(parsed.axis_direction, "increasing")
        self.assertEqual(parsed.axis_id, RAW_AXIS_ID)
        self.assertEqual(parsed.axis_float32_max_abs_error, 0.0)
        self.assertEqual(parsed.intensity_float32_max_abs_error, 0.0)
        self.assertEqual(parsed.comma_numeric_rows, 3)
        self.assertEqual(parsed.whitespace_numeric_rows, 0)
        self.assertEqual(outcome.statistics.parsed_numeric_rows, 3)
        self.assertEqual(outcome.statistics.comma_numeric_rows, 3)
        self.assertEqual(outcome.statistics.whitespace_numeric_rows, 0)
        self.assertEqual(outcome.statistics.ignored_comment_lines, 0)
        self.assertEqual(outcome.statistics.ignored_preamble_lines, 0)
        self.assertEqual(outcome.statistics.ignored_column_header_lines, 0)
        self.assertFalse(outcome.statistics.duplicate_header)

        with self.assertRaises(FrozenInstanceError):
            parsed.kind = "processed"

    def test_headers_ignored_lines_and_float_cast_errors_are_lossless(self):
        payload = member_payload(
            ("100.1,1.1", "101.2,2.2"),
            optional_headers=(
                "##PIN_ID=PIN-1",
                "##ORIENTATION=oriented",
                "##CELL PARAMETERS=coarse",
                "##CELL PARAMETERS=precise",
            ),
            body_prefix=(
                "# Instrument=LabRAM",
                "instrument export preamble",
                "X,Y",
            ),
        )

        outcome = parse(payload)

        parsed = outcome.accepted
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.pin_id, "PIN-1")
        self.assertEqual(parsed.orientation, "oriented")
        self.assertEqual(
            parsed.header_entries[-2:],
            (
                RruffHeaderEntry(8, "CELL PARAMETERS", "coarse"),
                RruffHeaderEntry(9, "CELL PARAMETERS", "precise"),
            ),
        )
        self.assertEqual(
            parsed.ignored_lines,
            (
                RruffIgnoredLine(11, "comment", "# Instrument=LabRAM"),
                RruffIgnoredLine(
                    12,
                    "preamble",
                    "instrument export preamble",
                ),
                RruffIgnoredLine(13, "column_header", "X,Y"),
            ),
        )
        self.assertEqual(
            parsed.axis_float32_max_abs_error,
            3.051757815342171e-06,
        )
        self.assertEqual(
            parsed.intensity_float32_max_abs_error,
            4.7683715642676816e-08,
        )
        self.assertEqual(outcome.statistics.ignored_comment_lines, 1)
        self.assertEqual(outcome.statistics.ignored_preamble_lines, 1)
        self.assertEqual(outcome.statistics.ignored_column_header_lines, 1)
        self.assertTrue(outcome.statistics.duplicate_header)

    def test_header_entries_preserve_value_whitespace_but_promotions_trim_it(self):
        payload = member_payload(
            ("100,1", "101,2"),
            name=" Quartz ",
            rruff_id=" R000001 ",
            filetype=" Raman RAW ",
            wavelength=" 532 ",
            optional_headers=(
                "##PIN_ID= PIN-1 ",
                "##ORIENTATION= oriented ",
                "##DESCRIPTION=source text with trailing space ",
            ),
        )

        outcome = parse(payload)

        parsed = outcome.accepted
        self.assertIsNotNone(parsed)
        self.assertEqual(
            parsed.header_entries[0],
            RruffHeaderEntry(1, "NAMES", " Quartz "),
        )
        self.assertEqual(
            parsed.header_entries[-1],
            RruffHeaderEntry(
                8,
                "DESCRIPTION",
                "source text with trailing space ",
            ),
        )
        self.assertEqual(parsed.mineral_name, "Quartz")
        self.assertEqual(parsed.rruff_id, "R000001")
        self.assertEqual(parsed.pin_id, "PIN-1")
        self.assertEqual(parsed.orientation, "oriented")
        self.assertEqual(parsed.excitation_nm, 532.0)

    def test_wavelength_promotion_accepts_nm_and_keeps_unknown_or_nonpositive_none(self):
        cases = (
            ("532 nm", 532.0),
            ("unknown", None),
            ("0", None),
        )
        for wavelength, expected in cases:
            with self.subTest(wavelength=wavelength):
                outcome = parse(
                    member_payload(
                        ("100,1", "101,2"),
                        wavelength=wavelength,
                    )
                )
                self.assertIsNotNone(outcome.accepted)
                self.assertEqual(outcome.accepted.excitation_nm, expected)

    def test_header_without_equals_is_retained_as_preamble(self):
        payload = member_payload(
            ("100,1", "101,2"),
            body_prefix=("##source export marker",),
        )

        outcome = parse(payload)

        self.assertIsNotNone(outcome.accepted)
        self.assertEqual(
            outcome.accepted.ignored_lines,
            (
                RruffIgnoredLine(
                    7,
                    "preamble",
                    "##source export marker",
                ),
            ),
        )
        self.assertEqual(outcome.statistics.ignored_preamble_lines, 1)

    def test_whitespace_processed_and_decreasing_axes_are_supported(self):
        payload = member_payload(
            ("102 0.3", "101 0.2", "100 0.1"),
            filetype="Raman Processed",
        )

        outcome = parse(payload, source_member=PROCESSED_MEMBER)

        parsed = outcome.accepted
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.kind, "processed")
        self.assertEqual(parsed.axis_direction, "decreasing")
        self.assertEqual(parsed.comma_numeric_rows, 0)
        self.assertEqual(parsed.whitespace_numeric_rows, 3)
        self.assertEqual(outcome.statistics.comma_numeric_rows, 0)
        self.assertEqual(outcome.statistics.whitespace_numeric_rows, 3)

    def test_cp1252_is_used_only_after_strict_utf8_failure(self):
        payload = member_payload(
            ("100,1", "101,2"),
            name="Sidérite",
            encoding="cp1252",
        )

        outcome = parse(payload)

        parsed = outcome.accepted
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.text_encoding, "cp1252")
        self.assertEqual(parsed.mineral_name, "Sidérite")

    def test_comments_after_numeric_data_are_retained_without_dropping_rows(self):
        payload = member_payload(
            ("100,1", "# acquisition note", "101,2"),
        )

        outcome = parse(payload)

        parsed = outcome.accepted
        self.assertIsNotNone(parsed)
        np.testing.assert_array_equal(
            parsed.source_axis,
            np.array([100.0, 101.0], dtype=np.float64),
        )
        self.assertEqual(
            parsed.ignored_lines,
            (RruffIgnoredLine(8, "comment", "# acquisition note"),),
        )
        self.assertEqual(outcome.statistics.parsed_numeric_rows, 2)

    def test_header_entries_after_numeric_data_are_retained(self):
        payload = member_payload(
            ("100,1", "101,2", "##END="),
        )

        outcome = parse(payload)

        self.assertIsNotNone(outcome.accepted)
        self.assertIsNone(outcome.rejected)
        self.assertEqual(outcome.statistics.parsed_numeric_rows, 2)
        self.assertEqual(
            outcome.accepted.header_entries,
            (
                RruffHeaderEntry(1, "NAMES", "Quartz"),
                RruffHeaderEntry(2, "RRUFFID", "R000001"),
                RruffHeaderEntry(3, "FILETYPE", "Raman RAW"),
                RruffHeaderEntry(4, "RAMAN WAVELENGTH", "532"),
                RruffHeaderEntry(
                    5,
                    "URL",
                    "https://rruff.info/R000001",
                ),
                RruffHeaderEntry(9, "END", ""),
            ),
        )

    def test_header_like_line_without_equals_after_numeric_data_is_rejected(self):
        payload = member_payload(
            ("100,1", "##source export marker", "101,2"),
        )

        outcome = self.assert_rejected("NUMERIC_ROW_INVALID", payload)

        self.assertEqual(outcome.statistics.parsed_numeric_rows, 2)
        self.assertEqual(len(outcome.rejected.header_entries), 5)

    def test_decode_failure_has_no_encoding_or_parsed_rows(self):
        outcome = self.assert_rejected("TEXT_DECODE_FAILURE", b"\x81")

        self.assertIsNone(outcome.rejected.text_encoding)
        self.assertEqual(outcome.rejected.header_entries, ())
        self.assertEqual(outcome.rejected.ignored_lines, ())
        self.assertEqual(outcome.statistics.parsed_numeric_rows, 0)

    def test_required_headers_and_identity_conflicts_fail_closed(self):
        missing_url = "\n".join(
            (
                "##NAMES=Quartz",
                "##RRUFFID=R000001",
                "##FILETYPE=Raman RAW",
                "##RAMAN WAVELENGTH=532",
                "100,1",
                "101,2",
                "",
            )
        ).encode("utf-8")
        conflict = member_payload(
            ("100,1", "101,2"),
            optional_headers=("##NAMES=Calcite",),
        )

        missing = self.assert_rejected(
            "REQUIRED_HEADER_MISSING",
            missing_url,
        )
        conflicting = self.assert_rejected(
            "IDENTITY_HEADER_CONFLICT",
            conflict,
        )

        self.assertEqual(missing.rejected.mineral_name, "Quartz")
        self.assertIsNone(conflicting.rejected.mineral_name)

    def test_required_header_keys_are_exact_and_not_whitespace_normalized(self):
        payload = "\n".join(
            (
                "## NAMES=Quartz",
                "##RRUFFID=R000001",
                "##FILETYPE=Raman RAW",
                "##RAMAN WAVELENGTH=532",
                "##URL=https://rruff.info/R000001",
                "",
                "100,1",
                "101,2",
                "",
            )
        ).encode("utf-8")

        outcome = self.assert_rejected(
            "REQUIRED_HEADER_MISSING",
            payload,
        )

        self.assertEqual(
            outcome.rejected.header_entries[0],
            RruffHeaderEntry(1, " NAMES", "Quartz"),
        )
        self.assertIsNone(outcome.rejected.mineral_name)

    def test_empty_required_header_value_is_missing(self):
        payload = member_payload(
            ("100,1", "101,2"),
            name="   ",
        )

        outcome = self.assert_rejected(
            "REQUIRED_HEADER_MISSING",
            payload,
        )

        self.assertEqual(
            outcome.rejected.header_entries[0],
            RruffHeaderEntry(1, "NAMES", "   "),
        )
        self.assertIsNone(outcome.rejected.mineral_name)

    def test_filename_and_filetype_must_agree(self):
        mismatch = member_payload(
            ("100,1", "101,2"),
            filetype="Raman Processed",
        )

        self.assert_rejected("FILETYPE_MISMATCH", mismatch)

        with self.assertRaises(RruffValidationError) as caught:
            parse(
                member_payload(("100,1", "101,2")),
                source_member="not-a-rruff-member.txt",
            )
        self.assertEqual(caught.exception.code, "MEMBER_NAME_INVALID")
        self.assertEqual(caught.exception.path, "source_member")

    def test_invalid_numeric_rows_before_and_after_data_are_rejected(self):
        one_numeric_field = member_payload(("100,-", "101,2"))
        malformed_after_data = member_payload(
            ("100,1", "101,2", "broken after data"),
        )

        before = self.assert_rejected(
            "NUMERIC_ROW_INVALID",
            one_numeric_field,
        )
        after = self.assert_rejected(
            "NUMERIC_ROW_INVALID",
            malformed_after_data,
        )

        self.assertEqual(before.statistics.parsed_numeric_rows, 1)
        self.assertEqual(
            before.rejected.ignored_lines,
            (RruffIgnoredLine(7, "preamble", "100,-"),),
        )
        self.assertEqual(before.statistics.ignored_preamble_lines, 1)
        self.assertEqual(after.statistics.parsed_numeric_rows, 2)
        self.assertEqual(after.statistics.comma_numeric_rows, 2)

    def test_numeric_like_rows_must_have_exactly_two_fields(self):
        one_field = member_payload(("100", "101,2"))
        three_fields = member_payload(("100,1,2", "101,2"))

        for payload in (one_field, three_fields):
            with self.subTest(payload=payload):
                outcome = self.assert_rejected(
                    "NUMERIC_ROW_INVALID",
                    payload,
                )
                self.assertEqual(outcome.statistics.parsed_numeric_rows, 1)
                self.assertEqual(
                    outcome.statistics.ignored_preamble_lines,
                    1,
                )

    def test_textual_preamble_may_contain_isolated_numeric_tokens(self):
        payload = member_payload(
            ("100,1", "101,2"),
            body_prefix=(
                "Fiducial mark perpendicular to laser is parallel "
                "to a [1 0 0]",
            ),
        )

        outcome = parse(payload)

        self.assertIsNotNone(outcome.accepted)
        self.assertIsNone(outcome.rejected)
        self.assertEqual(
            outcome.accepted.ignored_lines,
            (
                RruffIgnoredLine(
                    7,
                    "preamble",
                    (
                        "Fiducial mark perpendicular to laser is parallel "
                        "to a [1 0 0]"
                    ),
                ),
            ),
        )
        self.assertEqual(outcome.statistics.parsed_numeric_rows, 2)

    def test_at_least_two_numeric_rows_are_required(self):
        no_rows = member_payload((), body_prefix=("export preamble",))
        one_row = member_payload(("100,1",))

        self.assert_rejected("NUMERIC_DATA_MISSING", no_rows)
        one = self.assert_rejected("NUMERIC_DATA_MISSING", one_row)
        self.assertEqual(one.statistics.parsed_numeric_rows, 1)

    def test_nonfinite_source_values_are_rejected(self):
        for rows in (
            ("nan,1", "101,2"),
            ("100,inf", "101,2"),
        ):
            with self.subTest(rows=rows):
                self.assert_rejected(
                    "NONFINITE_VALUE",
                    member_payload(rows),
                )

    def test_float32_nonfinite_axis_and_intensity_have_distinct_codes(self):
        axis_overflow = member_payload(("3.5e38,1", "3.6e38,2"))
        intensity_overflow = member_payload(("100,3.5e38", "101,3.6e38"))

        self.assert_rejected("AXIS_FLOAT32_NONFINITE", axis_overflow)
        self.assert_rejected(
            "INTENSITY_FLOAT32_NONFINITE",
            intensity_overflow,
        )

    def test_float64_axis_duplicates_and_reversals_have_distinct_codes(self):
        duplicate = member_payload(("100,1", "100,2"))
        reversal = member_payload(("100,1", "102,2", "101,3"))

        self.assert_rejected("AXIS_DUPLICATE", duplicate)
        self.assert_rejected("AXIS_NONMONOTONIC", reversal)

    def test_float32_axis_collapse_is_rejected_without_source_repair(self):
        payload = member_payload(
            ("1.00000000,1", "1.00000001,2", "1.00000002,3"),
        )

        outcome = self.assert_rejected("AXIS_FLOAT32_COLLAPSE", payload)

        self.assertEqual(outcome.statistics.parsed_numeric_rows, 3)


class RruffArchiveInspectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_source(self.raw_root)

    def inspect(
        self,
        source=None,
        *,
        contract=None,
    ) -> _RruffSourceInspection:
        current_source = self.source if source is None else source
        current_contract = (
            synthetic_source_contract(current_source)
            if contract is None
            else contract
        )
        return _inspect_rruff_source(
            current_source.raw_root,
            current_contract,
        )

    def assert_fatal(
        self,
        expected_code: str,
        expected_path: str,
        *,
        source=None,
        contract=None,
    ) -> RruffValidationError:
        with self.assertRaises(RruffValidationError) as caught:
            self.inspect(source, contract=contract)
        self.assertEqual(caught.exception.code, expected_code)
        self.assertEqual(caught.exception.path, expected_path)
        return caught.exception

    def test_synthetic_contract_uses_independent_archive_facts(self):
        contract = synthetic_source_contract(self.source)

        self.assertIsInstance(contract, _RruffSourceContract)
        self.assertEqual(tuple(contract.archives), ARCHIVE_NAMES)
        self.assertEqual(contract.source_members, 4)
        self.assertEqual(contract.source_raw, 3)
        self.assertEqual(contract.source_processed, 1)
        self.assertEqual(contract.accepted_records, 4)
        self.assertEqual(contract.accepted_raw, 3)
        self.assertEqual(contract.accepted_processed, 1)
        self.assertEqual(contract.expected_rejections, frozenset())
        self.assertIsNone(contract.expected_statistics)
        self.assertIsNone(contract.expected_pair_statistics)
        self.assertEqual(
            contract.archives["LR-Raman.zip"],
            _ArchiveContract(
                archive="LR-Raman.zip",
                bytes=self.source.archive_bytes["LR-Raman.zip"],
                sha256=self.source.archive_sha256["LR-Raman.zip"],
                member_count=2,
                raw_members=1,
                processed_members=1,
            ),
        )
        with self.assertRaises(TypeError):
            contract.archives["LR-Raman.zip"] = contract.archives[
                "LR-Raman.zip"
            ]

    def test_preflight_rejects_missing_and_unexpected_archives(self):
        missing_root = self.temporary_root / "missing"
        missing = create_synthetic_rruff_source(missing_root)
        (missing.raw_root / "poor_unoriented.zip").unlink()
        missing = type(missing)(
            raw_root=missing.raw_root,
            archive_bytes=missing.archive_bytes,
            archive_sha256=missing.archive_sha256,
            archive_members=missing.archive_members,
        )

        self.assert_fatal(
            "ARCHIVE_MISSING",
            "raw_root.archives",
            source=missing,
            contract=synthetic_source_contract(self.source),
        )

        unexpected_root = self.temporary_root / "unexpected"
        unexpected = create_synthetic_rruff_source(unexpected_root)
        (unexpected.raw_root / "extra.zip").write_bytes(
            (unexpected.raw_root / "fair_oriented.zip").read_bytes()
        )
        self.assert_fatal(
            "ARCHIVE_UNEXPECTED",
            "raw_root.archives",
            source=unexpected,
            contract=synthetic_source_contract(self.source),
        )

    def test_preflight_rejects_archive_hash_and_crc_failures(self):
        hash_root = self.temporary_root / "hash"
        hash_source = create_synthetic_rruff_source(hash_root)
        hash_contract = synthetic_source_contract(hash_source)

        def mutate_equal_length(members):
            mutated = []
            for name, payload in members:
                if name == QUARTZ_RAW_MEMBER:
                    self.assertIn(b"100,1", payload)
                    payload = payload.replace(b"100,1", b"100,9", 1)
                mutated.append((name, payload))
            return mutated

        hash_source = rewrite_archive(
            hash_source,
            "LR-Raman.zip",
            mutate_equal_length,
        )
        self.assert_fatal(
            "ARCHIVE_HASH_MISMATCH",
            "archives.LR-Raman.zip.sha256",
            source=hash_source,
            contract=hash_contract,
        )

        crc_root = self.temporary_root / "crc"
        crc_source = create_synthetic_rruff_source(crc_root)
        crc_contract = synthetic_source_contract(crc_source)
        crc_source = corrupt_archive_member_crc(
            crc_source,
            "LR-Raman.zip",
            QUARTZ_RAW_MEMBER,
        )
        crc_archive = crc_source.raw_root / "LR-Raman.zip"
        crc_archives = dict(crc_contract.archives)
        crc_archives["LR-Raman.zip"] = _ArchiveContract(
            archive="LR-Raman.zip",
            bytes=crc_archive.stat().st_size,
            sha256=file_sha256(crc_archive),
            member_count=2,
            raw_members=1,
            processed_members=1,
        )
        crc_contract = _RruffSourceContract(
            archives=MappingProxyType(crc_archives),
            source_members=crc_contract.source_members,
            source_raw=crc_contract.source_raw,
            source_processed=crc_contract.source_processed,
            accepted_records=crc_contract.accepted_records,
            accepted_raw=crc_contract.accepted_raw,
            accepted_processed=crc_contract.accepted_processed,
            expected_rejections=crc_contract.expected_rejections,
            expected_statistics=None,
            expected_pair_statistics=None,
        )
        self.assert_fatal(
            "ARCHIVE_CRC_FAILURE",
            "archives.LR-Raman.zip.crc",
            source=crc_source,
            contract=crc_contract,
        )

    def test_preflight_rejects_source_member_count_and_invalid_names(self):
        count_contract = synthetic_source_contract(self.source)
        count_contract = _RruffSourceContract(
            archives=count_contract.archives,
            source_members=5,
            source_raw=count_contract.source_raw,
            source_processed=count_contract.source_processed,
            accepted_records=count_contract.accepted_records,
            accepted_raw=count_contract.accepted_raw,
            accepted_processed=count_contract.accepted_processed,
            expected_rejections=count_contract.expected_rejections,
            expected_statistics=None,
            expected_pair_statistics=None,
        )
        self.assert_fatal(
            "SOURCE_MEMBER_COUNT_MISMATCH",
            "source_members",
            contract=count_contract,
        )

        invalid_root = self.temporary_root / "invalid-name"
        invalid_source = create_synthetic_rruff_source(invalid_root)

        def add_invalid(members):
            return [*members, ("not-a-rruff-member.txt", b"not valid\n")]

        invalid_source = rewrite_archive(
            invalid_source,
            "poor_unoriented.zip",
            add_invalid,
        )
        invalid_contract = synthetic_source_contract(invalid_source)
        invalid_contract = _RruffSourceContract(
            archives=invalid_contract.archives,
            source_members=5,
            source_raw=3,
            source_processed=1,
            accepted_records=4,
            accepted_raw=3,
            accepted_processed=1,
            expected_rejections=frozenset(),
            expected_statistics=None,
            expected_pair_statistics=None,
        )
        self.assert_fatal(
            "MEMBER_NAME_INVALID",
            "archives.poor_unoriented.zip/not-a-rruff-member.txt",
            source=invalid_source,
            contract=invalid_contract,
        )

    def test_contract_count_drift_fails_closed_at_each_level(self):
        contract = synthetic_source_contract(self.source)
        archive_contracts = dict(contract.archives)
        archive_contracts["LR-Raman.zip"] = replace(
            archive_contracts["LR-Raman.zip"],
            member_count=3,
        )
        self.assert_fatal(
            "SOURCE_MEMBER_COUNT_MISMATCH",
            "archives.LR-Raman.zip.members",
            contract=replace(
                contract,
                archives=MappingProxyType(archive_contracts),
            ),
        )

        archive_contracts = dict(contract.archives)
        archive_contracts["LR-Raman.zip"] = replace(
            archive_contracts["LR-Raman.zip"],
            raw_members=2,
            processed_members=0,
        )
        self.assert_fatal(
            "SOURCE_MEMBER_COUNT_MISMATCH",
            "archives.LR-Raman.zip.kind_counts",
            contract=replace(
                contract,
                archives=MappingProxyType(archive_contracts),
            ),
        )

        self.assert_fatal(
            "SOURCE_MEMBER_COUNT_MISMATCH",
            "source_kind_counts",
            contract=replace(
                contract,
                source_raw=4,
                source_processed=0,
            ),
        )

        self.assert_fatal(
            "SOURCE_MEMBER_COUNT_MISMATCH",
            "accepted_members",
            contract=replace(
                contract,
                accepted_records=5,
            ),
        )

    def test_preflight_rejects_duplicate_member_paths_and_ids(self):
        path_root = self.temporary_root / "duplicate-path"
        path_source = create_synthetic_rruff_source(path_root)

        def duplicate_path(members):
            return [*members, members[0]]

        path_source = rewrite_archive(
            path_source,
            "LR-Raman.zip",
            duplicate_path,
        )
        path_contract = synthetic_source_contract(path_source)
        self.assert_fatal(
            "MEMBER_PATH_COLLISION",
            "archives.LR-Raman.zip.members",
            source=path_source,
            contract=path_contract,
        )

        id_root = self.temporary_root / "duplicate-id"
        id_source = create_synthetic_rruff_source(id_root)
        duplicate_id_member = (
            "Dolomite__R000099__Raman__532__0__unoriented__"
            "Raman_Data_RAW__0000000000000000000000000001.txt"
        )

        def duplicate_id(members):
            return [
                *members,
                (
                    duplicate_id_member,
                    member_payload(
                        ("400,1", "401,2"),
                        name="Dolomite",
                        rruff_id="R000099",
                    ),
                ),
            ]

        id_source = rewrite_archive(
            id_source,
            "poor_unoriented.zip",
            duplicate_id,
        )
        id_contract = synthetic_source_contract(id_source)
        self.assert_fatal(
            "MEMBER_ID_COLLISION",
            "member_ids.0000000000000000000000000001",
            source=id_source,
            contract=id_contract,
        )

    def test_archive_preflight_finishes_before_member_parser_runs(self):
        missing_root = self.temporary_root / "parser-order"
        source = create_synthetic_rruff_source(missing_root)
        (source.raw_root / "poor_unoriented.zip").unlink()

        with patch(
            "rpe.io.rruff_source._parse_member_bytes",
            side_effect=AssertionError("parser called before preflight"),
        ):
            self.assert_fatal(
                "ARCHIVE_MISSING",
                "raw_root.archives",
                source=source,
                contract=synthetic_source_contract(self.source),
            )

    def test_non_zip_evidence_is_ignored(self):
        evidence = self.raw_root / "evidence"
        evidence.mkdir()
        (evidence / "index.html").write_text(
            "<html>official listing</html>\n",
            encoding="utf-8",
        )

        inspection = self.inspect()

        self.assertEqual(inspection.statistics.source_members, 4)

    def test_acquisition_promotion_uses_exact_keys_and_safe_values(self):
        headers = (
            RruffHeaderEntry(1, "RAMAN WAVELENGTH", "unknown"),
        )
        ignored = (
            RruffIgnoredLine(2, "comment", "#Instrument=\tXploRA"),
            RruffIgnoredLine(3, "comment", "#Acq. time (s)=\t2.5"),
            RruffIgnoredLine(4, "comment", "#Accumulations=\t3"),
            RruffIgnoredLine(5, "comment", "#Grating=\t1200 gr/mm"),
            RruffIgnoredLine(6, "comment", "#Detector=\tSyncerity"),
            RruffIgnoredLine(7, "comment", "#Laser (nm)=\t785 nm"),
        )

        promoted = _promote_rruff_acquisition(headers, ignored)

        self.assertEqual(
            promoted,
            _RruffAcquisitionFields(
                instrument="XploRA",
                excitation_nm=785.0,
                integration_time_s=2.5,
                n_accumulations=3,
                grating="1200 gr/mm",
                detector="Syncerity",
            ),
        )

        exact_key_failure = _promote_rruff_acquisition(
            headers,
            (
                RruffIgnoredLine(2, "comment", "# Instrument=XploRA"),
                RruffIgnoredLine(3, "comment", "#laser (nm)=785"),
            ),
        )
        self.assertIsNone(exact_key_failure.instrument)
        self.assertIsNone(exact_key_failure.excitation_nm)

    def test_acquisition_conflicts_and_invalid_values_become_none(self):
        headers = (
            RruffHeaderEntry(1, "RAMAN WAVELENGTH", "532"),
        )
        ignored = (
            RruffIgnoredLine(2, "comment", "#Instrument=XploRA"),
            RruffIgnoredLine(3, "comment", "#Instrument=LabRAM"),
            RruffIgnoredLine(4, "comment", "#Acq. time (s)=-1"),
            RruffIgnoredLine(5, "comment", "#Accumulations=3.0"),
            RruffIgnoredLine(6, "comment", "#Grating="),
            RruffIgnoredLine(7, "comment", "#Detector=Syncerity"),
            RruffIgnoredLine(8, "comment", "#Laser (nm)=785"),
        )

        promoted = _promote_rruff_acquisition(headers, ignored)

        self.assertIsNone(promoted.instrument)
        self.assertIsNone(promoted.excitation_nm)
        self.assertIsNone(promoted.integration_time_s)
        self.assertIsNone(promoted.n_accumulations)
        self.assertIsNone(promoted.grating)
        self.assertEqual(promoted.detector, "Syncerity")

    def test_acquisition_empty_repeated_string_invalidates_promotion(self):
        headers = (
            RruffHeaderEntry(1, "RAMAN WAVELENGTH", "532"),
        )
        ignored = (
            RruffIgnoredLine(2, "comment", "#Instrument=XploRA"),
            RruffIgnoredLine(3, "comment", "#Instrument="),
            RruffIgnoredLine(4, "comment", "#Grating=1200 gr/mm"),
            RruffIgnoredLine(5, "comment", "#Grating=   "),
            RruffIgnoredLine(6, "comment", "#Detector=Syncerity"),
            RruffIgnoredLine(7, "comment", "#Detector="),
        )

        promoted = _promote_rruff_acquisition(headers, ignored)

        self.assertIsNone(promoted.instrument)
        self.assertIsNone(promoted.grating)
        self.assertIsNone(promoted.detector)

    def test_acquisition_equal_repeats_are_promoted(self):
        headers = (
            RruffHeaderEntry(1, "RAMAN WAVELENGTH", "532"),
        )
        ignored = (
            RruffIgnoredLine(2, "comment", "#Instrument=XploRA"),
            RruffIgnoredLine(3, "comment", "#Instrument=\tXploRA "),
            RruffIgnoredLine(4, "comment", "#Acq. time (s)=2.5"),
            RruffIgnoredLine(5, "comment", "#Acq. time (s)=2.500"),
            RruffIgnoredLine(6, "comment", "#Accumulations=3"),
            RruffIgnoredLine(7, "comment", "#Accumulations=03"),
            RruffIgnoredLine(8, "comment", "#Grating=1200 gr/mm"),
            RruffIgnoredLine(9, "comment", "#Detector=Syncerity"),
            RruffIgnoredLine(10, "comment", "#Laser (nm)=532"),
        )

        promoted = _promote_rruff_acquisition(headers, ignored)

        self.assertEqual(promoted.instrument, "XploRA")
        self.assertEqual(promoted.excitation_nm, 532.0)
        self.assertEqual(promoted.integration_time_s, 2.5)
        self.assertEqual(promoted.n_accumulations, 3)
        self.assertEqual(promoted.grating, "1200 gr/mm")
        self.assertEqual(promoted.detector, "Syncerity")

    def test_synthetic_source_inspection_has_exact_immutable_summary(self):
        inspection = self.inspect()

        self.assertIsInstance(inspection, _RruffSourceInspection)
        self.assertEqual(inspection.raw_root, self.raw_root)
        self.assertEqual(tuple(inspection.archive_sha256), ARCHIVE_NAMES)
        self.assertEqual(len(inspection.accepted_members), 4)
        self.assertEqual(inspection.rejected_members, ())
        self.assertEqual(
            [member.source_member for member in inspection.accepted_members],
            [
                QUARTZ_PROCESSED_MEMBER,
                QUARTZ_RAW_MEMBER,
                CALCITE_RAW_MEMBER,
                CP1252_RAW_MEMBER,
            ],
        )
        for member in inspection.accepted_members:
            self.assertIsInstance(member, _SourceAcceptedMember)
            self.assertFalse(
                any(
                    isinstance(value, np.ndarray)
                    for value in vars(member).values()
                )
            )
        self.assertEqual(
            dict(inspection.global_class_labels),
            {0: "Calcite", 1: "Quartz", 2: "Sidérite"},
        )
        self.assertEqual(
            dict(inspection.raw_class_labels),
            {0: "Calcite", 1: "Quartz", 2: "Sidérite"},
        )
        self.assertEqual(
            dict(inspection.processed_class_labels),
            {1: "Quartz"},
        )
        self.assertEqual(inspection.raw_rruff_ids, frozenset({"R000001", "R000002", "R000003"}))
        self.assertEqual(inspection.processed_rruff_ids, frozenset({"R000001"}))
        self.assertEqual(inspection.raw_axis_ids, frozenset(member.axis_id for member in inspection.accepted_members if member.kind == "raw"))
        self.assertEqual(inspection.processed_axis_ids, frozenset(member.axis_id for member in inspection.accepted_members if member.kind == "processed"))
        with self.assertRaises(TypeError):
            inspection.global_class_labels[0] = "changed"

        expected = _RruffSourceStatistics(
            source_members=4,
            source_raw=3,
            source_processed=1,
            accepted_records=4,
            rejected_records=0,
            accepted_raw=3,
            accepted_processed=1,
            accepted_spectral_points=12,
            rejected_member_parsed_numeric_rows=0,
            all_parsed_numeric_rows=12,
            mineral_classes=3,
            rruff_samples=3,
            raw_mineral_classes=3,
            processed_mineral_classes=1,
            raw_rruff_samples=3,
            processed_rruff_samples=1,
            utf8_members=3,
            cp1252_members=1,
            comma_numeric_rows=9,
            whitespace_numeric_rows=3,
            ignored_comment_lines=7,
            ignored_preamble_lines=1,
            ignored_column_header_lines=1,
            duplicate_header_records=1,
            increasing_axes=4,
            decreasing_axes=0,
            records_with_excitation_nm=4,
            records_with_excitation_nm_none=0,
            records_with_full_comment_acquisition_metadata=1,
            union_axis_groups=3,
            raw_axis_groups=3,
            processed_axis_groups=1,
            shared_axis_groups=1,
            raw_singleton_axis_groups=3,
            processed_singleton_axis_groups=1,
            raw_median_records_per_axis_group=1.0,
            processed_median_records_per_axis_group=1.0,
            raw_p95_records_per_axis_group=1.0,
            processed_p95_records_per_axis_group=1.0,
            raw_maximum_records_per_axis_group=1,
            processed_maximum_records_per_axis_group=1,
            axis_float32_max_abs_error_cm1=0.0,
            intensity_float32_max_abs_error=1.1920928966180355e-08,
            rejection_code_counts=MappingProxyType({}),
        )
        self.assertEqual(inspection.statistics, expected)

    def test_one_rruff_id_cannot_map_to_multiple_mineral_names(self):
        conflict_root = self.temporary_root / "rruff-id-conflict"
        source = create_synthetic_rruff_source(conflict_root)
        source = replace_member_payload(
            source,
            "excellent_oriented.zip",
            CALCITE_RAW_MEMBER,
            member_payload(
                ("200,4", "201,5", "202,6"),
                name="Calcite",
                rruff_id="R000001",
                wavelength="785 nm",
            ),
        )

        self.assert_fatal(
            "SOURCE_MEMBER_COUNT_MISMATCH",
            "rruff_ids.R000001.mineral_names",
            source=source,
            contract=synthetic_source_contract(source),
        )

    def test_full_comment_acquisition_count_only_includes_raw_records(self):
        processed_root = self.temporary_root / "processed-acquisition"
        source = create_synthetic_rruff_source(processed_root)
        source = replace_member_payload(
            source,
            "LR-Raman.zip",
            QUARTZ_PROCESSED_MEMBER,
            member_payload(
                ("100 0.1", "101 0.2", "102 0.3"),
                filetype="Raman Processed",
                body_prefix=(
                    "#Instrument=XploRA",
                    "#Acq. time (s)=2.5",
                    "#Accumulations=3",
                    "#Grating=1200 gr/mm",
                    "#Detector=Syncerity",
                ),
            ),
        )

        inspection = self.inspect(
            source,
            contract=synthetic_source_contract(source),
        )

        self.assertEqual(
            inspection.statistics.records_with_full_comment_acquisition_metadata,
            1,
        )

    def test_expected_rejection_set_and_statistics_fail_closed(self):
        rejected_root = self.temporary_root / "rejected"
        source = create_synthetic_rruff_source(rejected_root)
        rejected_payload = member_payload(("100,1", "100,2"))
        source = replace_member_payload(
            source,
            "LR-Raman.zip",
            QUARTZ_RAW_MEMBER,
            rejected_payload,
        )
        expected_rejection = frozenset(
            {
                (
                    "LR-Raman.zip",
                    QUARTZ_RAW_MEMBER,
                    "raw",
                    "AXIS_DUPLICATE",
                )
            }
        )
        contract = synthetic_source_contract(
            source,
            expected_rejections=expected_rejection,
        )
        inspection = self.inspect(source, contract=contract)
        self.assertEqual(len(inspection.accepted_members), 3)
        self.assertEqual(len(inspection.rejected_members), 1)
        self.assertEqual(
            inspection.statistics.rejection_code_counts,
            {"AXIS_DUPLICATE": 1},
        )

        wrong_rejection_contract = synthetic_source_contract(source)
        self.assert_fatal(
            "SOURCE_MEMBER_COUNT_MISMATCH",
            "rejected_members",
            source=source,
            contract=wrong_rejection_contract,
        )

        expected_statistics = {
            field.name: getattr(inspection.statistics, field.name)
            for field in inspection.statistics.__dataclass_fields__.values()
            if field.name != "rejection_code_counts"
        }
        expected_statistics["accepted_records"] = 99
        contract = _RruffSourceContract(
            archives=contract.archives,
            source_members=contract.source_members,
            source_raw=contract.source_raw,
            source_processed=contract.source_processed,
            accepted_records=contract.accepted_records,
            accepted_raw=contract.accepted_raw,
            accepted_processed=contract.accepted_processed,
            expected_rejections=contract.expected_rejections,
            expected_statistics=MappingProxyType(expected_statistics),
            expected_pair_statistics=None,
        )
        self.assert_fatal(
            "SOURCE_MEMBER_COUNT_MISMATCH",
            "statistics.accepted_records",
            source=source,
            contract=contract,
        )

    def test_production_contract_is_literal_and_not_synthetic_output(self):
        self.assertEqual(PRODUCTION_SOURCE_CONTRACT.source_members, 38056)
        self.assertEqual(PRODUCTION_SOURCE_CONTRACT.source_raw, 20674)
        self.assertEqual(PRODUCTION_SOURCE_CONTRACT.source_processed, 17382)
        self.assertEqual(PRODUCTION_SOURCE_CONTRACT.accepted_records, 38043)
        self.assertEqual(PRODUCTION_SOURCE_CONTRACT.accepted_raw, 20664)
        self.assertEqual(PRODUCTION_SOURCE_CONTRACT.accepted_processed, 17379)
        self.assertEqual(len(PRODUCTION_SOURCE_CONTRACT.archives), 8)
        self.assertEqual(len(PRODUCTION_REJECTIONS), 13)
        self.assertEqual(
            PRODUCTION_SOURCE_CONTRACT.expected_rejections,
            PRODUCTION_REJECTIONS,
        )
        self.assertEqual(
            PRODUCTION_SOURCE_CONTRACT.expected_statistics[
                "accepted_spectral_points"
            ],
            77368604,
        )
        self.assertEqual(
            PRODUCTION_SOURCE_CONTRACT.expected_statistics[
                "axis_float32_max_abs_error_cm1"
            ],
            0.00024218749967985786,
        )
        self.assertEqual(
            PRODUCTION_SOURCE_CONTRACT.expected_pair_statistics[
                "source_measurement_groups"
            ],
            22185,
        )


class RruffPairingInspectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.contract = synthetic_pairing_contract(self.source)

    def inspect(self) -> RruffInspection:
        return _inspect_rruff(self.raw_root, self.contract)

    def bundle(self) -> _RruffInspectionBundle:
        return _inspect_rruff_bundle(self.raw_root, self.contract)

    def pair(
        self,
        inspection: RruffInspection,
        archive: str,
        measurement_key: str,
    ) -> RruffPairInspection:
        return next(
            pair
            for pair in inspection.pairs
            if pair.archive == archive
            and pair.measurement_key == measurement_key
        )

    def test_pair_id_is_archive_scoped_and_domain_separated(self):
        expected = hashlib.sha256(
            b"rpe-rruff-pair-v1\0"
            + b"LR-Raman.zip"
            + b"\0"
            + PAIR_EXACT_KEY.encode("utf-8")
        ).hexdigest()

        self.assertEqual(
            _pair_id("LR-Raman.zip", PAIR_EXACT_KEY),
            expected,
        )
        self.assertNotEqual(
            _pair_id("fair_oriented.zip", PAIR_EXACT_KEY),
            expected,
        )
        self.assertEqual(len(expected), 64)

    def test_pairing_fixture_covers_every_status_and_relation(self):
        inspection = self.inspect()

        self.assertEqual(len(inspection.pairs), 13)
        status_counts = {
            status: sum(pair.pair_status == status for pair in inspection.pairs)
            for status in (
                "paired_unique",
                "raw_only",
                "processed_only",
                "ambiguous",
                "rejected_only",
            )
        }
        self.assertEqual(
            status_counts,
            {
                "paired_unique": 5,
                "raw_only": 3,
                "processed_only": 1,
                "ambiguous": 3,
                "rejected_only": 1,
            },
        )
        relation_counts = {
            relation: sum(
                pair.axis_relation == relation for pair in inspection.pairs
            )
            for relation in (
                "exact_equal",
                "processed_exact_contiguous_subset_of_raw",
                "raw_exact_contiguous_subset_of_processed",
                "overlap_requires_alignment",
                "no_axis_overlap",
            )
        }
        self.assertEqual(
            relation_counts,
            {
                "exact_equal": 1,
                "processed_exact_contiguous_subset_of_raw": 1,
                "raw_exact_contiguous_subset_of_processed": 1,
                "overlap_requires_alignment": 1,
                "no_axis_overlap": 1,
            },
        )

    def test_exact_and_subset_relations_have_exact_half_open_slices(self):
        inspection = self.inspect()

        exact = self.pair(
            inspection,
            "LR-Raman.zip",
            PAIR_EXACT_KEY,
        )
        self.assertEqual(exact.axis_relation, "exact_equal")
        self.assertEqual(exact.raw_slice, (0, 3))
        self.assertEqual(exact.processed_slice, (0, 3))

        processed_subset = self.pair(
            inspection,
            "LR-Raman.zip",
            PAIR_PROCESSED_SUBSET_KEY,
        )
        self.assertEqual(
            processed_subset.axis_relation,
            "processed_exact_contiguous_subset_of_raw",
        )
        self.assertEqual(processed_subset.raw_slice, (1, 3))
        self.assertEqual(processed_subset.processed_slice, (0, 2))

        raw_subset = self.pair(
            inspection,
            "LR-Raman.zip",
            PAIR_RAW_SUBSET_KEY,
        )
        self.assertEqual(
            raw_subset.axis_relation,
            "raw_exact_contiguous_subset_of_processed",
        )
        self.assertEqual(raw_subset.raw_slice, (0, 2))
        self.assertEqual(raw_subset.processed_slice, (1, 3))

    def test_axis_relations_support_decreasing_native_axes(self):
        cases = (
            (
                np.array([5.0, 4.0, 3.0, 2.0]),
                np.array([4.0, 3.0]),
                (
                    "processed_exact_contiguous_subset_of_raw",
                    (1, 3),
                    (0, 2),
                ),
            ),
            (
                np.array([4.0, 3.0]),
                np.array([5.0, 4.0, 3.0, 2.0]),
                (
                    "raw_exact_contiguous_subset_of_processed",
                    (0, 2),
                    (1, 3),
                ),
            ),
            (
                np.array([5.0, 4.0, 3.0]),
                np.array([4.5, 3.5, 2.5]),
                ( "overlap_requires_alignment", None, None),
            ),
            (
                np.array([5.0, 4.0]),
                np.array([2.0, 1.0]),
                ("no_axis_overlap", None, None),
            ),
        )
        for raw_axis, processed_axis, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    _axis_relation(raw_axis, processed_axis),
                    expected,
                )

    def test_alignment_and_no_overlap_have_no_slices(self):
        inspection = self.inspect()

        alignment = self.pair(
            inspection,
            "LR-Raman.zip",
            PAIR_ALIGNMENT_KEY,
        )
        self.assertEqual(
            alignment.axis_relation,
            "overlap_requires_alignment",
        )
        self.assertIsNone(alignment.raw_slice)
        self.assertIsNone(alignment.processed_slice)

        no_overlap = self.pair(
            inspection,
            "LR-Raman.zip",
            PAIR_NO_OVERLAP_KEY,
        )
        self.assertEqual(no_overlap.axis_relation, "no_axis_overlap")
        self.assertIsNone(no_overlap.raw_slice)
        self.assertIsNone(no_overlap.processed_slice)

    def test_rejected_duplicate_keeps_group_ambiguous(self):
        inspection = self.inspect()
        pair = self.pair(
            inspection,
            "LR-Raman.zip",
            PAIR_REJECTED_DUPLICATE_KEY,
        )

        self.assertEqual(pair.pair_status, "ambiguous")
        self.assertIsNone(pair.axis_relation)
        self.assertIsNone(pair.raw_slice)
        self.assertIsNone(pair.processed_slice)
        self.assertEqual(len(pair.raw_members), 2)
        accepted, rejected = pair.raw_members
        self.assertEqual(accepted.conversion_status, "accepted")
        self.assertEqual(
            accepted.record_id,
            "raw-1000000000000000000000000018",
        )
        self.assertIsNone(accepted.rejection_code)
        self.assertEqual(rejected.conversion_status, "rejected")
        self.assertIsNone(rejected.record_id)
        self.assertEqual(rejected.rejection_code, "AXIS_DUPLICATE")

    def test_rejected_only_group_has_no_output_reference(self):
        inspection = self.inspect()
        pair = next(
            pair
            for pair in inspection.pairs
            if pair.pair_status == "rejected_only"
        )

        self.assertEqual(len(pair.raw_members), 1)
        self.assertEqual(pair.processed_members, ())
        self.assertIsNone(pair.raw_members[0].record_id)
        self.assertEqual(
            pair.raw_members[0].rejection_code,
            "AXIS_DUPLICATE",
        )
        self.assertIsNone(pair.axis_relation)

    def test_cross_archive_measurement_keys_do_not_collapse(self):
        inspection = self.inspect()
        pairs = [
            pair
            for pair in inspection.pairs
            if pair.measurement_key == PAIR_CROSS_ARCHIVE_KEY
        ]

        self.assertEqual(
            [pair.archive for pair in pairs],
            ["LR-Raman.zip", "fair_oriented.zip"],
        )
        self.assertEqual([pair.pair_status for pair in pairs], ["raw_only", "raw_only"])
        self.assertNotEqual(pairs[0].pair_id, pairs[1].pair_id)
        self.assertNotEqual(
            pairs[0].raw_members[0].source_member_sha256,
            pairs[1].raw_members[0].source_member_sha256,
        )

    def test_pair_member_and_pair_order_is_deterministic(self):
        inspection = self.inspect()

        observed_pair_keys = [
            (
                pair.archive.encode("utf-8"),
                pair.measurement_key.encode("utf-8"),
            )
            for pair in inspection.pairs
        ]
        self.assertEqual(observed_pair_keys, sorted(observed_pair_keys))
        for pair in inspection.pairs:
            for members in (pair.raw_members, pair.processed_members):
                observed = [
                    (
                        member.member_id.encode("ascii"),
                        member.source_member.encode("utf-8"),
                    )
                    for member in members
                ]
                self.assertEqual(observed, sorted(observed))
            self.assertEqual(
                pair.mineral_names,
                tuple(
                    sorted(
                        set(pair.mineral_names),
                        key=lambda value: value.encode("utf-8"),
                    )
                ),
            )
            self.assertEqual(
                pair.rruff_ids,
                tuple(
                    sorted(
                        set(pair.rruff_ids),
                        key=lambda value: value.encode("utf-8"),
                    )
                ),
            )

    def test_public_inspection_is_array_free_and_members_have_pair_fields(self):
        bundle = self.bundle()
        inspection = bundle.inspection

        self.assertIsInstance(inspection, RruffInspection)
        self.assertIs(bundle.statistics, bundle.statistics)
        self.assertEqual(inspection.raw_root, self.raw_root)
        self.assertEqual(inspection.source_member_count, 21)
        self.assertEqual(len(inspection.accepted_members), 19)
        self.assertEqual(len(inspection.rejected_members), 2)
        self.assertEqual(len(inspection.pairs), 13)
        self.assertEqual(inspection.raw_axis_group_count, 10)
        self.assertEqual(inspection.processed_axis_group_count, 8)
        self.assertEqual(
            tuple(inspection.global_class_labels),
            tuple(sorted(inspection.global_class_labels)),
        )
        for member in inspection.accepted_members:
            self.assertIsInstance(member, RruffAcceptedMember)
            self.assertFalse(
                any(
                    isinstance(value, np.ndarray)
                    for value in vars(member).values()
                )
            )
            self.assertEqual(
                member.pair_id,
                _pair_id(member.archive, member.measurement_key),
            )
            pair = self.pair(
                inspection,
                member.archive,
                member.measurement_key,
            )
            self.assertEqual(member.pair_status, pair.pair_status)
        with self.assertRaises(TypeError):
            inspection.archive_sha256["LR-Raman.zip"] = "changed"
        with self.assertRaises(TypeError):
            inspection.global_class_labels[0] = "changed"

    def test_every_pair_member_reference_matches_public_member_inventory(self):
        inspection = self.inspect()
        accepted_by_id = {
            (
                f"{member.kind}-{member.member_id}"
            ): member
            for member in inspection.accepted_members
        }
        accepted_references = []
        rejected_references = []
        for pair in inspection.pairs:
            for member in (*pair.raw_members, *pair.processed_members):
                if member.conversion_status == "accepted":
                    self.assertIsNotNone(member.record_id)
                    self.assertIn(member.record_id, accepted_by_id)
                    public_member = accepted_by_id[member.record_id]
                    self.assertEqual(public_member.pair_id, pair.pair_id)
                    self.assertEqual(
                        public_member.pair_status,
                        pair.pair_status,
                    )
                    self.assertEqual(
                        public_member.source_member,
                        member.source_member,
                    )
                    self.assertEqual(
                        public_member.member_sha256,
                        member.source_member_sha256,
                    )
                    self.assertIsNone(member.rejection_code)
                    accepted_references.append(member.record_id)
                else:
                    self.assertEqual(member.conversion_status, "rejected")
                    self.assertIsNone(member.record_id)
                    self.assertIsNotNone(member.rejection_code)
                    rejected_references.append(member.member_id)
        self.assertEqual(
            sorted(accepted_references),
            sorted(accepted_by_id),
        )
        self.assertEqual(
            sorted(rejected_references),
            sorted(member.member_id for member in inspection.rejected_members),
        )

    def test_public_inspection_exports_final_rruff_api(self):
        import rpe.io as rpe_io

        for name in (
            "RruffAcceptedMember",
            "RruffConversionSummary",
            "RruffDatasetSummary",
            "RruffHeaderEntry",
            "RruffIgnoredLine",
            "RruffInspection",
            "RruffPairInspection",
            "RruffPairMember",
            "RruffRejectedMember",
            "RruffRuntimeMetrics",
            "build_rruff_unified",
            "inspect_rruff",
            "iter_rruff_raw_records",
            "iter_rruff_processed_records",
        ):
            self.assertIn(name, rpe_io.__all__)
            self.assertTrue(hasattr(rpe_io, name))

    def test_public_inspect_rruff_uses_production_contract(self):
        with patch(
            "rpe.io.rruff._inspect_rruff",
            return_value=self.inspect(),
        ) as mocked:
            result = inspect_rruff(self.raw_root)

        self.assertIsInstance(result, RruffInspection)
        mocked.assert_called_once()
        self.assertEqual(mocked.call_args.args[0], self.raw_root)
        self.assertIs(
            mocked.call_args.args[1],
            PRODUCTION_SOURCE_CONTRACT,
        )

    def test_paired_unique_metadata_mismatch_fails_closed(self):
        mismatch_root = self.temporary_root / "mismatch"
        source = create_synthetic_rruff_pairing_source(mismatch_root)
        processed_name = (
            f"{PAIR_EXACT_KEY}__Raman_Data_Processed__"
            f"{PAIR_EXACT_PROCESSED_MEMBER_ID}.txt"
        )
        source = replace_member_payload(
            source,
            "LR-Raman.zip",
            processed_name,
            member_payload(
                ("100,0.1", "101,0.2", "102,0.3"),
                name="Exactite",
                rruff_id="R100001",
                filetype="Raman Processed",
                wavelength="633",
                optional_headers=(
                    "##PIN_ID=PIN-EXACT",
                    "##ORIENTATION=unoriented",
                ),
            ),
        )
        contract = synthetic_pairing_contract(source)

        with self.assertRaises(RruffValidationError) as caught:
            _inspect_rruff(source.raw_root, contract)

        self.assertEqual(caught.exception.code, "PAIR_METADATA_MISMATCH")
        self.assertEqual(
            caught.exception.path,
            (
                "pairs.LR-Raman.zip/"
                f"{PAIR_EXACT_KEY}.RAMAN WAVELENGTH"
            ),
        )

    def test_pair_reparse_missing_member_fails_with_stable_error(self):
        source_inspection = _inspect_rruff_source(
            self.raw_root,
            self.contract,
        )
        processed_name = (
            f"{PAIR_EXACT_KEY}__Raman_Data_Processed__"
            f"{PAIR_EXACT_PROCESSED_MEMBER_ID}.txt"
        )

        def remove_member(members):
            return [
                item for item in members if item[0] != processed_name
            ]

        rewrite_archive(
            self.source,
            "LR-Raman.zip",
            remove_member,
        )

        with self.assertRaises(RruffValidationError) as caught:
            _build_pairs(
                self.raw_root,
                source_inspection,
                self.contract.expected_pair_statistics,
            )
        self.assertEqual(
            caught.exception.code,
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        self.assertEqual(
            caught.exception.path,
            f"pairs.LR-Raman.zip/{processed_name}",
        )

    def test_pair_reparse_changed_bytes_fails_member_hash_check(self):
        source_inspection = _inspect_rruff_source(
            self.raw_root,
            self.contract,
        )
        raw_name = (
            f"{PAIR_EXACT_KEY}__Raman_Data_RAW__"
            f"{PAIR_EXACT_RAW_MEMBER_ID}.txt"
        )

        def mutate_equal_length(members):
            mutated = []
            for name, payload in members:
                if name == raw_name:
                    payload = payload.replace(b"100,1", b"100,9", 1)
                mutated.append((name, payload))
            return mutated

        rewrite_archive(
            self.source,
            "LR-Raman.zip",
            mutate_equal_length,
        )

        with self.assertRaises(RruffValidationError) as caught:
            _build_pairs(
                self.raw_root,
                source_inspection,
                self.contract.expected_pair_statistics,
            )
        self.assertEqual(
            caught.exception.code,
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        self.assertEqual(
            caught.exception.path,
            f"pairs.LR-Raman.zip/{raw_name}.member_sha256",
        )

    def test_pairing_statistics_drift_fails_closed(self):
        wrong_statistics = dict(self.contract.expected_pair_statistics)
        wrong_statistics["source_measurement_groups"] = 99
        contract = replace(
            self.contract,
            expected_pair_statistics=MappingProxyType(wrong_statistics),
        )

        with self.assertRaises(RruffValidationError) as caught:
            _inspect_rruff(self.raw_root, contract)

        self.assertEqual(
            caught.exception.code,
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        self.assertEqual(
            caught.exception.path,
            "pairing_statistics.source_measurement_groups",
        )

    def test_production_pair_statistics_are_not_used_as_synthetic_oracle(self):
        self.assertEqual(dict(self.contract.expected_pair_statistics), dict(PAIRING_STATISTICS))
        self.assertEqual(self.contract.expected_rejections, PAIRING_REJECTIONS)
        self.assertEqual(
            PRODUCTION_SOURCE_CONTRACT.expected_pair_statistics[
                "source_measurement_groups"
            ],
            22185,
        )
        self.assertNotEqual(
            self.contract.expected_pair_statistics[
                "source_measurement_groups"
            ],
            PRODUCTION_SOURCE_CONTRACT.expected_pair_statistics[
                "source_measurement_groups"
            ],
        )


RRUFF_SOURCE_METADATA_KEYS = {
    "source_dataset_id",
    "source_archive",
    "source_archive_sha256",
    "source_member",
    "source_member_sha256",
    "source_member_bytes",
    "source_member_id",
    "source_measurement_key",
    "source_filetype",
    "source_text_encoding",
    "source_header_entries",
    "source_ignored_lines",
    "ignored_comment_line_count",
    "ignored_preamble_line_count",
    "ignored_column_header_line_count",
    "source_point_count",
    "source_axis_dtype",
    "stored_axis_dtype",
    "source_intensity_dtype",
    "stored_intensity_dtype",
    "archive_collection",
    "quality_rating",
    "orientation_collection",
    "mineral_name",
    "rruff_id",
    "pin_id",
    "pair_id",
    "pair_status",
}


class RruffRecordMappingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.contract = synthetic_pairing_contract(self.source)
        self.inspection = _inspect_rruff(self.raw_root, self.contract)

    def raw_records(self):
        return list(
            iter_rruff_raw_records(self.raw_root, self.inspection)
        )

    def processed_records(self):
        return list(
            iter_rruff_processed_records(
                self.raw_root,
                self.inspection,
            )
        )

    def expected_class_id(self, mineral_name: str) -> int:
        return next(
            class_id
            for class_id, name in self.inspection.global_class_labels.items()
            if name == mineral_name
        )

    def assert_common_record(self, record, member, *, dataset_id):
        self.assertIs(validate_record(record), record)
        self.assertEqual(record.meta.dataset_id, dataset_id)
        self.assertEqual(record.meta.sample_id, member.rruff_id)
        self.assertEqual(record.meta.excitation_nm, member.excitation_nm)
        self.assertEqual(
            record.targets,
            Targets(
                class_label=self.expected_class_id(member.mineral_name)
            ),
        )
        self.assertIsNone(record.targets.clean)
        self.assertIsNone(record.targets.baseline)
        self.assertIsNone(record.targets.peaks)
        self.assertIsNone(record.targets.concentration)
        self.assertIsNone(record.targets.concentrations)
        self.assertEqual(record.intensity.dtype, np.dtype(np.float32))
        self.assertEqual(record.wavenumber.dtype, np.dtype(np.float32))
        self.assertEqual(record.intensity.ndim, 1)
        self.assertEqual(record.wavenumber.ndim, 1)
        self.assertEqual(record.intensity.shape, record.wavenumber.shape)
        self.assertTrue(np.isfinite(record.intensity).all())
        self.assertTrue(np.isfinite(record.wavenumber).all())

        metadata = record.meta.source_metadata
        self.assertEqual(set(metadata), RRUFF_SOURCE_METADATA_KEYS)
        self.assertEqual(metadata["source_dataset_id"], "RRUFF Raman")
        self.assertEqual(metadata["source_archive"], member.archive)
        self.assertEqual(
            metadata["source_archive_sha256"],
            self.inspection.archive_sha256[member.archive],
        )
        self.assertEqual(metadata["source_member"], member.source_member)
        self.assertEqual(
            metadata["source_member_sha256"],
            member.member_sha256,
        )
        self.assertEqual(
            metadata["source_member_bytes"],
            member.member_bytes,
        )
        self.assertEqual(metadata["source_member_id"], member.member_id)
        self.assertEqual(
            metadata["source_measurement_key"],
            member.measurement_key,
        )
        self.assertEqual(
            metadata["source_filetype"],
            "Raman RAW" if member.kind == "raw" else "Raman Processed",
        )
        self.assertEqual(
            metadata["source_text_encoding"],
            member.text_encoding,
        )
        self.assertEqual(metadata["source_point_count"], member.source_point_count)
        self.assertEqual(metadata["source_axis_dtype"], "float64")
        self.assertEqual(metadata["stored_axis_dtype"], "float32")
        self.assertEqual(metadata["source_intensity_dtype"], "float64")
        self.assertEqual(metadata["stored_intensity_dtype"], "float32")
        self.assertEqual(metadata["mineral_name"], member.mineral_name)
        self.assertEqual(metadata["rruff_id"], member.rruff_id)
        self.assertEqual(metadata["pin_id"], member.pin_id)
        self.assertEqual(metadata["pair_id"], member.pair_id)
        self.assertEqual(metadata["pair_status"], member.pair_status)
        self.assertEqual(
            metadata["source_header_entries"],
            [
                {
                    "line": entry.line,
                    "key": entry.key,
                    "value": entry.value,
                }
                for entry in member.header_entries
            ],
        )
        self.assertEqual(
            metadata["source_ignored_lines"],
            [
                {
                    "line": ignored.line,
                    "category": ignored.category,
                    "text": ignored.text,
                }
                for ignored in member.ignored_lines
            ],
        )
        for category, key in (
            ("comment", "ignored_comment_line_count"),
            ("preamble", "ignored_preamble_line_count"),
            ("column_header", "ignored_column_header_line_count"),
        ):
            self.assertEqual(
                metadata[key],
                sum(
                    ignored.category == category
                    for ignored in member.ignored_lines
                ),
            )

        self.assertEqual(
            record.provenance.source_url,
            (
                "https://rruff.info/zipped_data_files/raman/"
                f"{member.archive}"
            ),
        )
        self.assertEqual(record.provenance.license, "not stated")
        self.assertIs(
            record.provenance.license_status,
            LicenseStatus.NOT_STATED,
        )
        self.assertEqual(
            record.provenance.sha256,
            self.inspection.archive_sha256[member.archive],
        )
        self.assertEqual(record.provenance.retrieved_date, date(2026, 8, 14))
        self.assertEqual(record.provenance.source_artifact, member.archive)

    def test_raw_records_have_truthful_status_targets_metadata_and_provenance(self):
        records = self.raw_records()
        members = [
            member
            for member in self.inspection.accepted_members
            if member.kind == "raw"
        ]

        self.assertEqual(len(records), len(members))
        self.assertEqual(
            [record.record_id for record in records],
            [f"raw-{member.member_id}" for member in members],
        )
        for record, member in zip(records, members, strict=True):
            with self.subTest(record_id=record.record_id):
                self.assert_common_record(
                    record,
                    member,
                    dataset_id="rruff_raman_raw",
                )
                self.assertIs(
                    record.meta.preprocessing_status,
                    PreprocessingStatus.KNOWN_RAW,
                )
                self.assertEqual(record.meta.preprocessing_steps, ())
                self.assertTrue(
                    record.meta.eligible_for_preprocessing_evaluation
                )

    def test_processed_records_are_unknown_and_never_clean_targets(self):
        records = self.processed_records()
        members = [
            member
            for member in self.inspection.accepted_members
            if member.kind == "processed"
        ]

        self.assertEqual(len(records), len(members))
        self.assertEqual(
            [record.record_id for record in records],
            [f"processed-{member.member_id}" for member in members],
        )
        for record, member in zip(records, members, strict=True):
            with self.subTest(record_id=record.record_id):
                self.assert_common_record(
                    record,
                    member,
                    dataset_id="rruff_raman_processed",
                )
                self.assertIs(
                    record.meta.preprocessing_status,
                    PreprocessingStatus.UNKNOWN,
                )
                self.assertEqual(
                    record.meta.preprocessing_steps,
                    (
                        PreprocessingStep(
                            operation="source_processed_state",
                            description=(
                                "Source FILETYPE labels this spectrum Raman "
                                "Processed; the exact processing operations "
                                "and parameters are unavailable"
                            ),
                            evidence=(
                                f"{member.archive}/{member.source_member} "
                                "FILETYPE header"
                            ),
                        ),
                    ),
                )
                self.assertFalse(
                    record.meta.eligible_for_preprocessing_evaluation
                )

    def test_archive_derived_metadata_is_exact(self):
        records = {
            record.record_id: record
            for record in (*self.raw_records(), *self.processed_records())
        }
        lr_member = next(
            member
            for member in self.inspection.accepted_members
            if member.archive == "LR-Raman.zip"
        )
        fair_member = next(
            member
            for member in self.inspection.accepted_members
            if member.archive == "fair_oriented.zip"
        )

        lr_record = records[f"{lr_member.kind}-{lr_member.member_id}"]
        fair_record = records[f"{fair_member.kind}-{fair_member.member_id}"]
        self.assertEqual(
            (
                lr_record.meta.source_metadata["archive_collection"],
                lr_record.meta.source_metadata["quality_rating"],
                lr_record.meta.source_metadata["orientation_collection"],
            ),
            ("long_range", None, "unoriented"),
        )
        self.assertEqual(
            (
                fair_record.meta.source_metadata["archive_collection"],
                fair_record.meta.source_metadata["quality_rating"],
                fair_record.meta.source_metadata["orientation_collection"],
            ),
            ("standard_range", "fair", "oriented"),
        )

    def test_complete_and_conflicting_acquisition_comments_map_to_records(self):
        baseline_root = self.temporary_root / "acquisition-baseline"
        source = create_synthetic_rruff_source(baseline_root)
        inspection = _inspect_rruff(
            source.raw_root,
            synthetic_source_contract(source),
        )
        records = list(
            iter_rruff_raw_records(source.raw_root, inspection)
        )
        calcite = next(
            record
            for record in records
            if record.meta.source_metadata["source_member"]
            == CALCITE_RAW_MEMBER
        )
        self.assertEqual(
            (
                calcite.meta.instrument,
                calcite.meta.excitation_nm,
                calcite.meta.integration_time_s,
                calcite.meta.n_accumulations,
                calcite.meta.grating,
                calcite.meta.detector,
            ),
            (
                "XploRA",
                785.0,
                2.5,
                3,
                "1200 gr/mm",
                "Syncerity",
            ),
        )

        conflict_root = self.temporary_root / "acquisition-conflict"
        source = create_synthetic_rruff_source(conflict_root)
        source = replace_member_payload(
            source,
            "excellent_oriented.zip",
            CALCITE_RAW_MEMBER,
            member_payload(
                ("200,4", "201,5", "202,6"),
                name="Calcite",
                rruff_id="R000002",
                wavelength="785",
                body_prefix=(
                    "#Instrument=XploRA",
                    "#Instrument=LabRAM",
                    "#Acq. time (s)=-1",
                    "#Accumulations=3.0",
                    "#Grating=",
                    "#Detector=Syncerity",
                    "#Laser (nm)=532",
                ),
            ),
        )
        inspection = _inspect_rruff(
            source.raw_root,
            synthetic_source_contract(source),
        )
        conflict_record = next(
            record
            for record in iter_rruff_raw_records(
                source.raw_root,
                inspection,
            )
            if record.meta.source_metadata["source_member"]
            == CALCITE_RAW_MEMBER
        )
        self.assertIsNone(conflict_record.meta.instrument)
        self.assertIsNone(conflict_record.meta.excitation_nm)
        self.assertIsNone(conflict_record.meta.integration_time_s)
        self.assertIsNone(conflict_record.meta.n_accumulations)
        self.assertIsNone(conflict_record.meta.grating)
        self.assertEqual(conflict_record.meta.detector, "Syncerity")
        self.assertEqual(
            conflict_record.meta.source_metadata[
                "ignored_comment_line_count"
            ],
            7,
        )

    def test_unoriented_archive_suffix_is_not_misclassified_as_oriented(self):
        baseline_root = self.temporary_root / "baseline"
        source = create_synthetic_rruff_source(baseline_root)
        inspection = _inspect_rruff(
            source.raw_root,
            synthetic_source_contract(source),
        )
        records = list(
            iter_rruff_raw_records(source.raw_root, inspection)
        )
        cp1252_record = next(
            record
            for record in records
            if record.meta.source_metadata["source_member"]
            == CP1252_RAW_MEMBER
        )

        self.assertEqual(
            (
                cp1252_record.meta.source_metadata["archive_collection"],
                cp1252_record.meta.source_metadata["quality_rating"],
                cp1252_record.meta.source_metadata[
                    "orientation_collection"
                ],
            ),
            ("standard_range", "fair", "unoriented"),
        )

    def test_record_arrays_are_independent_float32_copies(self):
        records = self.raw_records()
        by_axis = {}
        for record in records:
            by_axis.setdefault(
                next(
                    member.axis_id
                    for member in self.inspection.accepted_members
                    if member.member_id
                    == record.meta.source_metadata["source_member_id"]
                ),
                [],
            ).append(record)
        shared = next(
            current
            for current in by_axis.values()
            if len(current) >= 2
        )
        first, second = shared[:2]
        original_second_axis = second.wavenumber.copy()
        original_second_intensity = second.intensity.copy()

        first.wavenumber[0] += 1.0
        first.intensity[0] += 1.0

        np.testing.assert_array_equal(
            second.wavenumber,
            original_second_axis,
        )
        np.testing.assert_array_equal(
            second.intensity,
            original_second_intensity,
        )

    def test_wrong_inspection_root_is_rejected_before_archive_access(self):
        wrong_root = self.temporary_root / "other"
        with patch(
            "zipfile.ZipFile",
            side_effect=AssertionError("archive opened before root check"),
        ):
            with self.assertRaises(RruffValidationError) as caught:
                list(
                    iter_rruff_raw_records(
                        wrong_root,
                        self.inspection,
                    )
                )
        self.assertEqual(caught.exception.code, "SOURCE_MEMBER_COUNT_MISMATCH")
        self.assertEqual(caught.exception.path, "inspection.raw_root")

    def test_iterator_rechecks_member_hash_before_yield(self):
        inspection = self.inspection
        target = next(
            member
            for member in inspection.accepted_members
            if member.kind == "raw"
        )

        def mutate_equal_length(members):
            mutated = []
            for name, payload in members:
                if name == target.source_member:
                    payload = payload.replace(b",1\n", b",9\n", 1)
                mutated.append((name, payload))
            return mutated

        rewrite_archive(
            self.source,
            target.archive,
            mutate_equal_length,
        )

        with self.assertRaises(RruffValidationError) as caught:
            list(iter_rruff_raw_records(self.raw_root, inspection))
        self.assertEqual(caught.exception.code, "SOURCE_MEMBER_COUNT_MISMATCH")
        self.assertEqual(
            caught.exception.path,
            f"records.{target.source_member}.member_sha256",
        )

    def test_iterator_missing_member_fails_with_record_path(self):
        inspection = self.inspection
        target = next(
            member
            for member in inspection.accepted_members
            if member.kind == "raw"
        )

        def remove_member(members):
            return [
                item
                for item in members
                if item[0] != target.source_member
            ]

        rewrite_archive(
            self.source,
            target.archive,
            remove_member,
        )

        with self.assertRaises(RruffValidationError) as caught:
            list(iter_rruff_raw_records(self.raw_root, inspection))
        self.assertEqual(
            caught.exception.code,
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        self.assertEqual(
            caught.exception.path,
            f"records.{target.source_member}",
        )

    def test_iterator_crc_failure_has_stable_record_path(self):
        inspection = self.inspection
        target = next(
            member
            for member in inspection.accepted_members
            if member.kind == "raw"
        )
        corrupt_archive_member_crc(
            self.source,
            target.archive,
            target.source_member,
        )

        with self.assertRaises(RruffValidationError) as caught:
            list(iter_rruff_raw_records(self.raw_root, inspection))
        self.assertEqual(
            caught.exception.code,
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        self.assertEqual(
            caught.exception.path,
            f"records.{target.source_member}.read",
        )

    def test_iterator_reparses_selected_kind_once_and_never_other_kind(self):
        import rpe.io.rruff as rruff_module

        real_parser = rruff_module._parse_member_bytes
        raw_members = [
            member
            for member in self.inspection.accepted_members
            if member.kind == "raw"
        ]
        with patch(
            "rpe.io.rruff._parse_member_bytes",
            wraps=real_parser,
        ) as parser:
            records = list(
                iter_rruff_raw_records(
                    self.raw_root,
                    self.inspection,
                )
            )

        self.assertEqual(len(records), len(raw_members))
        self.assertEqual(parser.call_count, len(raw_members))
        self.assertEqual(
            [call.args[1] for call in parser.call_args_list],
            [member.source_member for member in raw_members],
        )
        self.assertFalse(
            any(
                "Processed" in call.args[1]
                for call in parser.call_args_list
            )
        )

    def test_iterator_rechecks_point_axis_and_cast_summaries(self):
        members = [
            member
            for member in self.inspection.accepted_members
            if member.kind == "raw"
        ]
        mutations = (
            ("source_point_count", members[0].source_point_count + 1),
            ("axis_id", "0" * 64),
            (
                "axis_float32_max_abs_error",
                members[0].axis_float32_max_abs_error + 1.0,
            ),
            (
                "intensity_float32_max_abs_error",
                members[0].intensity_float32_max_abs_error + 1.0,
            ),
        )
        for field_name, replacement in mutations:
            with self.subTest(field_name=field_name):
                target = members[0]
                mutated_members = tuple(
                    replace(
                        member,
                        **{field_name: replacement},
                    )
                    if member.member_id == target.member_id
                    else member
                    for member in self.inspection.accepted_members
                )
                mutated_inspection = replace(
                    self.inspection,
                    accepted_members=mutated_members,
                )
                with self.assertRaises(RruffValidationError) as caught:
                    list(
                        iter_rruff_raw_records(
                            self.raw_root,
                            mutated_inspection,
                        )
                    )
                self.assertEqual(
                    caught.exception.code,
                    "SOURCE_MEMBER_COUNT_MISMATCH",
                )
                self.assertEqual(
                    caught.exception.path,
                    f"records.{target.source_member}.{field_name}",
                )

    def test_iterator_rejects_unknown_kind_and_missing_archive_hash(self):
        target = next(
            member
            for member in self.inspection.accepted_members
            if member.kind == "raw"
        )
        unknown_kind_members = tuple(
            replace(member, kind="mystery")
            if member.member_id == target.member_id
            else member
            for member in self.inspection.accepted_members
        )
        unknown_kind = replace(
            self.inspection,
            accepted_members=unknown_kind_members,
        )
        with self.assertRaises(RruffValidationError) as caught:
            list(iter_rruff_raw_records(self.raw_root, unknown_kind))
        self.assertEqual(
            caught.exception.code,
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        self.assertEqual(
            caught.exception.path,
            f"records.{target.source_member}.kind",
        )

        missing_hash = replace(
            self.inspection,
            archive_sha256=MappingProxyType(
                {
                    name: digest
                    for name, digest in self.inspection.archive_sha256.items()
                    if name != target.archive
                }
            ),
        )
        with self.assertRaises(RruffValidationError) as caught:
            list(iter_rruff_raw_records(self.raw_root, missing_hash))
        self.assertEqual(
            caught.exception.code,
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        self.assertEqual(
            caught.exception.path,
            f"records.{target.source_member}.source_archive_sha256",
        )

    def test_iterator_exports_only_after_implementation(self):
        import rpe.io as rpe_io

        for name in (
            "iter_rruff_raw_records",
            "iter_rruff_processed_records",
            "build_rruff_unified",
        ):
            self.assertIn(name, rpe_io.__all__)
            self.assertTrue(hasattr(rpe_io, name))


PAIR_INDEX_KEYS = {
    "pair_id",
    "archive",
    "measurement_key",
    "mineral_names",
    "rruff_ids",
    "source_multiplicity",
    "accepted_multiplicity",
    "raw_members",
    "processed_members",
    "pair_status",
    "axis_relation",
    "pointwise_comparable_without_interpolation",
    "raw_slice",
    "processed_slice",
}
PAIR_MULTIPLICITY_KEYS = {"raw", "processed"}
PAIR_MEMBER_KEYS = {
    "member_id",
    "record_id",
    "source_member",
    "source_member_sha256",
    "conversion_status",
    "rejection_code",
}


def read_pair_index(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def write_pair_index_rows(
    path: Path,
    rows: list[dict[str, object]],
) -> None:
    path.write_bytes(
        b"".join(_canonical_json_bytes(row) for row in rows)
    )


class RruffPairIndexTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.inspection = _inspect_rruff(
            self.raw_root,
            synthetic_pairing_contract(self.source),
        )
        self.index_path = self.temporary_root / "rruff_raman_pairs.jsonl"

    def write(self) -> None:
        if self.index_path.exists():
            self.index_path.unlink()
        _write_rruff_pair_index(self.index_path, self.inspection)

    def validate(self, *, with_record_ids: bool = False):
        kwargs = {}
        if with_record_ids:
            kwargs = {
                "raw_record_ids": {
                    f"raw-{member.member_id}"
                    for member in self.inspection.accepted_members
                    if member.kind == "raw"
                },
                "processed_record_ids": {
                    f"processed-{member.member_id}"
                    for member in self.inspection.accepted_members
                    if member.kind == "processed"
                },
            }
        return _validate_rruff_pair_index(
            self.index_path,
            self.inspection,
            **kwargs,
        )

    def assert_invalid(
        self,
        expected_path: str,
        mutation,
        *,
        with_record_ids: bool = False,
    ) -> RruffValidationError:
        self.write()
        rows = read_pair_index(self.index_path)
        mutation(rows)
        write_pair_index_rows(self.index_path, rows)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate(with_record_ids=with_record_ids)
        self.assertEqual(caught.exception.path, expected_path)
        return caught.exception

    def test_writer_emits_exact_canonical_schema_and_order(self):
        self.write()

        raw = self.index_path.read_bytes()
        rows = read_pair_index(self.index_path)
        self.assertTrue(raw.endswith(b"\n"))
        self.assertEqual(
            raw,
            b"".join(_canonical_json_bytes(row) for row in rows),
        )
        self.assertEqual(len(rows), 13)
        self.assertEqual(
            [
                (
                    row["archive"].encode("utf-8"),
                    row["measurement_key"].encode("utf-8"),
                )
                for row in rows
            ],
            sorted(
                (
                    row["archive"].encode("utf-8"),
                    row["measurement_key"].encode("utf-8"),
                )
                for row in rows
            ),
        )
        for row in rows:
            self.assertEqual(set(row), PAIR_INDEX_KEYS)
            self.assertEqual(
                set(row["source_multiplicity"]),
                PAIR_MULTIPLICITY_KEYS,
            )
            self.assertEqual(
                set(row["accepted_multiplicity"]),
                PAIR_MULTIPLICITY_KEYS,
            )
            for members in (row["raw_members"], row["processed_members"]):
                for member in members:
                    self.assertEqual(set(member), PAIR_MEMBER_KEYS)
            encoded = json.dumps(row, ensure_ascii=False)
            for forbidden in (
                str(self.temporary_root),
                "total_seconds",
                "peak_rss",
                "timestamp",
                "created_at",
            ):
                self.assertNotIn(forbidden, encoded)

    def test_writer_maps_inspection_semantics_exactly(self):
        self.write()
        rows = read_pair_index(self.index_path)
        by_id = {row["pair_id"]: row for row in rows}

        for pair in self.inspection.pairs:
            row = by_id[pair.pair_id]
            self.assertEqual(row["archive"], pair.archive)
            self.assertEqual(row["measurement_key"], pair.measurement_key)
            self.assertEqual(row["mineral_names"], list(pair.mineral_names))
            self.assertEqual(row["rruff_ids"], list(pair.rruff_ids))
            self.assertEqual(row["pair_status"], pair.pair_status)
            self.assertEqual(row["axis_relation"], pair.axis_relation)
            self.assertEqual(
                row["pointwise_comparable_without_interpolation"],
                pair.axis_relation
                in {
                    "exact_equal",
                    "processed_exact_contiguous_subset_of_raw",
                    "raw_exact_contiguous_subset_of_processed",
                },
            )
            self.assertEqual(
                row["raw_slice"],
                None
                if pair.raw_slice is None
                else {
                    "start": pair.raw_slice[0],
                    "stop": pair.raw_slice[1],
                },
            )
            self.assertEqual(
                row["processed_slice"],
                None
                if pair.processed_slice is None
                else {
                    "start": pair.processed_slice[0],
                    "stop": pair.processed_slice[1],
                },
            )
            self.assertEqual(
                row["source_multiplicity"],
                {
                    "raw": len(pair.raw_members),
                    "processed": len(pair.processed_members),
                },
            )
            self.assertEqual(
                row["accepted_multiplicity"],
                {
                    "raw": sum(
                        member.conversion_status == "accepted"
                        for member in pair.raw_members
                    ),
                    "processed": sum(
                        member.conversion_status == "accepted"
                        for member in pair.processed_members
                    ),
                },
            )

    def test_validator_returns_exact_status_relation_and_row_counts(self):
        self.write()

        summary = self.validate(with_record_ids=True)

        self.assertEqual(
            dict(summary),
            {
                "rows": 13,
                "paired_unique": 5,
                "raw_only": 3,
                "processed_only": 1,
                "ambiguous": 3,
                "rejected_only": 1,
                "exact_equal": 1,
                "processed_exact_contiguous_subset_of_raw": 1,
                "raw_exact_contiguous_subset_of_processed": 1,
                "overlap_requires_alignment": 1,
                "no_axis_overlap": 1,
                "pointwise_comparable_pairs": 3,
            },
        )
        with self.assertRaises(TypeError):
            summary["rows"] = 0

    def test_writer_rejects_existing_path_and_missing_parent(self):
        self.index_path.write_text("existing\n", encoding="utf-8")
        with self.assertRaises(RruffValidationError) as caught:
            _write_rruff_pair_index(self.index_path, self.inspection)
        self.assertEqual(caught.exception.path, "pair_index")

        missing = self.temporary_root / "missing" / "pairs.jsonl"
        with self.assertRaises(RruffValidationError) as caught:
            _write_rruff_pair_index(missing, self.inspection)
        self.assertEqual(caught.exception.path, "pair_index.parent")

    def test_validator_rejects_unknown_missing_and_noncanonical_keys(self):
        self.assert_invalid(
            "pair_index[0]",
            lambda rows: rows[0].__setitem__("unexpected", True),
        )
        self.assert_invalid(
            "pair_index[0]",
            lambda rows: rows[0].pop("pair_status"),
        )

        self.write()
        rows = read_pair_index(self.index_path)
        noncanonical = (
            json.dumps(rows[0], ensure_ascii=False, indent=2).encode("utf-8")
            + b"\n"
            + b"".join(_canonical_json_bytes(row) for row in rows[1:])
        )
        self.index_path.write_bytes(noncanonical)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertEqual(caught.exception.path, "pair_index[0]")

    def test_validator_rejects_duplicate_unsorted_and_cross_archive_rows(self):
        self.assert_invalid(
            "pair_index.order",
            lambda rows: rows.insert(0, dict(rows[0])),
        )
        self.assert_invalid(
            "pair_index.order",
            lambda rows: rows.__setitem__(
                slice(0, 2),
                [rows[1], rows[0]],
            ),
        )
        self.assert_invalid(
            "pair_index.order",
            lambda rows: rows[0].__setitem__(
                "archive",
                next(
                    row["archive"]
                    for row in rows
                    if row["archive"] != rows[0]["archive"]
                ),
            ),
        )

    def test_validator_rejects_wrong_identity_status_and_multiplicity(self):
        self.assert_invalid(
            "pair_index[0].pair_id",
            lambda rows: rows[0].__setitem__("pair_id", "0" * 64),
        )
        self.assert_invalid(
            "pair_index[0].pair_status",
            lambda rows: rows[0].__setitem__("pair_status", "raw_only"),
        )
        self.assert_invalid(
            "pair_index[0].source_multiplicity",
            lambda rows: rows[0]["source_multiplicity"].__setitem__(
                "raw",
                99,
            ),
        )
        self.assert_invalid(
            "pair_index[0].accepted_multiplicity",
            lambda rows: rows[0]["accepted_multiplicity"].__setitem__(
                "processed",
                99,
            ),
        )

    def test_validator_rejects_wrong_relation_comparability_and_slices(self):
        paired_index = None

        def mutate_relation(rows):
            nonlocal paired_index
            paired_index = next(
                index
                for index, row in enumerate(rows)
                if row["pair_status"] == "paired_unique"
            )
            rows[paired_index]["axis_relation"] = "no_axis_overlap"

        self.write()
        rows = read_pair_index(self.index_path)
        mutate_relation(rows)
        write_pair_index_rows(self.index_path, rows)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertEqual(
            caught.exception.path,
            f"pair_index[{paired_index}].axis_relation",
        )

        self.write()
        rows = read_pair_index(self.index_path)
        paired_index = next(
            index
            for index, row in enumerate(rows)
            if row["pair_status"] == "paired_unique"
        )
        rows[paired_index][
            "pointwise_comparable_without_interpolation"
        ] = not rows[paired_index][
            "pointwise_comparable_without_interpolation"
        ]
        write_pair_index_rows(self.index_path, rows)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertEqual(
            caught.exception.path,
            (
                f"pair_index[{paired_index}]."
                "pointwise_comparable_without_interpolation"
            ),
        )

        self.write()
        rows = read_pair_index(self.index_path)
        subset_index = next(
            index
            for index, row in enumerate(rows)
            if row["raw_slice"] is not None
        )
        rows[subset_index]["raw_slice"]["stop"] += 1
        write_pair_index_rows(self.index_path, rows)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertEqual(
            caught.exception.path,
            f"pair_index[{subset_index}].raw_slice",
        )

    def test_validator_rejects_member_null_semantics_and_hash_drift(self):
        self.write()
        rows = read_pair_index(self.index_path)
        accepted_index = next(
            index
            for index, row in enumerate(rows)
            if any(
                member["conversion_status"] == "accepted"
                for member in (
                    *row["raw_members"],
                    *row["processed_members"],
                )
            )
        )
        accepted_member = next(
            member
            for member in (
                *rows[accepted_index]["raw_members"],
                *rows[accepted_index]["processed_members"],
            )
            if member["conversion_status"] == "accepted"
        )
        accepted_member["record_id"] = None
        write_pair_index_rows(self.index_path, rows)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertIn(
            caught.exception.path,
            {
                f"pair_index[{accepted_index}].raw_members",
                f"pair_index[{accepted_index}].processed_members",
            },
        )

        self.write()
        rows = read_pair_index(self.index_path)
        rejected_index = next(
            index
            for index, row in enumerate(rows)
            if any(
                member["conversion_status"] == "rejected"
                for member in (
                    *row["raw_members"],
                    *row["processed_members"],
                )
            )
        )
        rejected_member = next(
            member
            for member in (
                *rows[rejected_index]["raw_members"],
                *rows[rejected_index]["processed_members"],
            )
            if member["conversion_status"] == "rejected"
        )
        rejected_member["record_id"] = "raw-invented"
        rejected_member["rejection_code"] = None
        write_pair_index_rows(self.index_path, rows)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertIn(
            caught.exception.path,
            {
                f"pair_index[{rejected_index}].raw_members",
                f"pair_index[{rejected_index}].processed_members",
            },
        )

        self.write()
        rows = read_pair_index(self.index_path)
        first_member = (
            rows[0]["raw_members"] or rows[0]["processed_members"]
        )[0]
        first_member["source_member_sha256"] = "0" * 64
        write_pair_index_rows(self.index_path, rows)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertIn(caught.exception.path, {
            "pair_index[0].raw_members",
            "pair_index[0].processed_members",
        })

    def test_validator_rejects_unresolved_output_record_ids(self):
        self.write()
        raw_record_ids = {
            f"raw-{member.member_id}"
            for member in self.inspection.accepted_members
            if member.kind == "raw"
        }
        processed_record_ids = {
            f"processed-{member.member_id}"
            for member in self.inspection.accepted_members
            if member.kind == "processed"
        }
        missing_raw = next(iter(raw_record_ids))
        raw_record_ids.remove(missing_raw)

        with self.assertRaises(RruffValidationError) as caught:
            _validate_rruff_pair_index(
                self.index_path,
                self.inspection,
                raw_record_ids=raw_record_ids,
                processed_record_ids=processed_record_ids,
            )
        self.assertEqual(caught.exception.path, "pair_index.record_ids.raw")

    def test_output_record_id_sets_must_be_supplied_together(self):
        self.write()
        raw_record_ids = {
            f"raw-{member.member_id}"
            for member in self.inspection.accepted_members
            if member.kind == "raw"
        }

        with self.assertRaises(RruffValidationError) as caught:
            _validate_rruff_pair_index(
                self.index_path,
                self.inspection,
                raw_record_ids=raw_record_ids,
                processed_record_ids=None,
            )
        self.assertEqual(caught.exception.path, "pair_index.record_ids")

    def test_validator_rejects_extra_trailing_bytes_and_invalid_json(self):
        self.write()
        self.index_path.write_bytes(
            self.index_path.read_bytes() + b"trailing"
        )
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertEqual(caught.exception.path, "pair_index")

        self.index_path.write_bytes(b"{not-json}\n")
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertEqual(caught.exception.path, "pair_index[0]")

    def test_validator_wraps_nonfinite_json_and_non_ascii_member_ids(self):
        self.write()
        rows = read_pair_index(self.index_path)
        rows[0]["source_multiplicity"]["raw"] = float("nan")
        self.index_path.write_text(
            "\n".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=True,
                )
                for row in rows
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertEqual(caught.exception.path, "pair_index[0]")

        self.write()
        rows = read_pair_index(self.index_path)
        first_member = (
            rows[0]["raw_members"] or rows[0]["processed_members"]
        )[0]
        first_member["member_id"] = "mémbre"
        write_pair_index_rows(self.index_path, rows)
        with self.assertRaises(RruffValidationError) as caught:
            self.validate()
        self.assertIn(
            caught.exception.path,
            {
                "pair_index[0].raw_members",
                "pair_index[0].processed_members",
            },
        )

    def test_writer_removes_partial_file_after_serialization_failure(self):
        import rpe.io.rruff_pairing as pairing_module

        real_canonical = pairing_module._canonical_json_bytes
        calls = 0

        def fail_second(value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("synthetic serialization failure")
            return real_canonical(value)

        with patch(
            "rpe.io.rruff_pairing._canonical_json_bytes",
            side_effect=fail_second,
        ):
            with self.assertRaises(RruffValidationError) as caught:
                self.write()
        self.assertEqual(caught.exception.path, "pair_index")
        self.assertFalse(self.index_path.exists())

    def test_independent_writes_are_byte_identical(self):
        second_path = self.temporary_root / "second.jsonl"

        self.write()
        _write_rruff_pair_index(second_path, self.inspection)

        self.assertEqual(
            self.index_path.read_bytes(),
            second_path.read_bytes(),
        )


def artifact_snapshot(path: Path):
    if path.is_dir():
        return {
            item.relative_to(path).as_posix(): (
                "dir" if item.is_dir() else item.read_bytes()
            )
            for item in sorted(path.rglob("*"))
        }
    return path.read_bytes()


class RruffStagedDatasetTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.contract = synthetic_pairing_contract(self.source)
        self.bundle = _inspect_rruff_bundle(self.raw_root, self.contract)
        self.staging_parent = self.temporary_root / "staging"
        self.staging_parent.mkdir()
        self.output_root = self.temporary_root / "final-output"

    def build(self, staging_parent: Path | None = None):
        return _build_rruff_data_staged(
            self.raw_root,
            self.staging_parent
            if staging_parent is None
            else staging_parent,
            self.bundle,
        )

    def test_staged_build_creates_two_valid_datasets_and_pair_index(self):
        staged = self.build()

        self.assertIsInstance(staged, _RruffStagedData)
        self.assertIsInstance(staged.raw, RruffDatasetSummary)
        self.assertIsInstance(staged.processed, RruffDatasetSummary)
        self.assertIsInstance(
            staged.raw_comparison,
            _RruffComparisonSummary,
        )
        self.assertIsInstance(
            staged.processed_comparison,
            _RruffComparisonSummary,
        )
        self.assertEqual(
            set(path.name for path in self.staging_parent.iterdir()),
            {
                "rruff_raman_raw",
                "rruff_raman_processed",
                "rruff_raman_pairs.jsonl",
            },
        )
        for summary in (staged.raw, staged.processed):
            dataset_path = self.staging_parent / summary.dataset_id
            self.assertEqual(
                {path.name for path in dataset_path.iterdir()},
                set(DATASET_FILES),
            )
            validation = validate_dataset(dataset_path)
            self.assertEqual(validation.record_count, summary.record_count)
            self.assertEqual(
                validation.axis_group_count,
                summary.axis_group_count,
            )
            self.assertEqual(set(summary.files), set(DATASET_FILES))
            self.assertEqual(
                summary.output_bytes,
                sum(
                    int(details["bytes"])
                    for details in summary.files.values()
                ),
            )
            for name, details in summary.files.items():
                path = dataset_path / name
                self.assertEqual(details["bytes"], path.stat().st_size)
                self.assertEqual(
                    details["sha256"],
                    file_sha256(path),
                )
        self.assertEqual(staged.pair_index_path.name, "rruff_raman_pairs.jsonl")
        self.assertEqual(staged.pair_index_rows, 13)
        self.assertEqual(
            staged.pair_index_sha256,
            file_sha256(staged.pair_index_path),
        )
        self.assertEqual(
            staged.raw_comparison,
            _RruffComparisonSummary(11, 0, 0, 0),
        )
        self.assertEqual(
            staged.processed_comparison,
            _RruffComparisonSummary(8, 0, 0, 0),
        )
        self.assertFalse(self.output_root.exists())

    def test_staged_manifests_use_global_ids_without_renumbering(self):
        staged = self.build()
        raw_manifest = json.loads(
            (
                self.staging_parent
                / staged.raw.dataset_id
                / "dataset.json"
            ).read_text(encoding="utf-8")
        )
        processed_manifest = json.loads(
            (
                self.staging_parent
                / staged.processed.dataset_id
                / "dataset.json"
            ).read_text(encoding="utf-8")
        )
        expected_raw_ids = {
            str(member.targets.class_label)
            for member in iter_rruff_raw_records(
                self.raw_root,
                self.bundle.inspection,
            )
        }
        expected_processed_ids = {
            str(member.targets.class_label)
            for member in iter_rruff_processed_records(
                self.raw_root,
                self.bundle.inspection,
            )
        }
        self.assertEqual(
            set(raw_manifest["class_labels"]),
            expected_raw_ids,
        )
        self.assertEqual(
            set(processed_manifest["class_labels"]),
            expected_processed_ids,
        )
        for manifest in (raw_manifest, processed_manifest):
            for label, name in manifest["class_labels"].items():
                self.assertEqual(
                    self.bundle.inspection.global_class_labels[int(label)],
                    name,
                )

    def test_staged_records_have_exact_eligibility_status_and_no_targets(self):
        staged = self.build()
        raw_validation = validate_dataset(
            self.staging_parent / staged.raw.dataset_id
        )
        processed_validation = validate_dataset(
            self.staging_parent / staged.processed.dataset_id
        )
        self.assertEqual(
            raw_validation.preprocessing_status_counts,
            {
                "known_raw": 11,
                "known_corrected": 0,
                "unknown": 0,
            },
        )
        self.assertEqual(
            processed_validation.preprocessing_status_counts,
            {
                "known_raw": 0,
                "known_corrected": 0,
                "unknown": 8,
            },
        )
        for validation in (raw_validation, processed_validation):
            self.assertEqual(
                validation.target_presence_counts,
                {
                    "clean": 0,
                    "baseline": 0,
                    "peaks": 0,
                    "class_label": validation.record_count,
                    "concentration": 0,
                    "concentrations": 0,
                },
            )
        self.assertEqual(staged.raw.eligible_records, 11)
        self.assertEqual(staged.processed.eligible_records, 0)

    def test_build_materializes_and_releases_raw_before_processed(self):
        import rpe.io.rruff as rruff_module

        real_write_dataset = rruff_module.write_dataset
        calls = []

        def recording_write(records, output_dir, **kwargs):
            materialized = list(records)
            calls.append(
                (
                    kwargs["dataset_id"],
                    len(materialized),
                    [record.meta.dataset_id for record in materialized],
                )
            )
            return real_write_dataset(
                materialized,
                output_dir,
                **kwargs,
            )

        with patch(
            "rpe.io.rruff.write_dataset",
            side_effect=recording_write,
        ):
            self.build()

        self.assertEqual(
            [dataset_id for dataset_id, _, _ in calls],
            ["rruff_raman_raw", "rruff_raman_processed"],
        )
        self.assertEqual(calls[0][1], 11)
        self.assertEqual(calls[1][1], 8)
        self.assertEqual(set(calls[0][2]), {"rruff_raman_raw"})
        self.assertEqual(
            set(calls[1][2]),
            {"rruff_raman_processed"},
        )

    def test_pair_index_references_exact_output_record_ids(self):
        staged = self.build()
        with UnifiedDataset.open(
            self.staging_parent / staged.raw.dataset_id
        ) as raw_dataset:
            raw_ids = set(raw_dataset.record_ids)
        with UnifiedDataset.open(
            self.staging_parent / staged.processed.dataset_id
        ) as processed_dataset:
            processed_ids = set(processed_dataset.record_ids)

        summary = _validate_rruff_pair_index(
            staged.pair_index_path,
            self.bundle.inspection,
            raw_record_ids=raw_ids,
            processed_record_ids=processed_ids,
        )
        self.assertEqual(summary["rows"], 13)
        rows = read_pair_index(staged.pair_index_path)
        rejected_members = [
            member
            for row in rows
            for member in (
                *row["raw_members"],
                *row["processed_members"],
            )
            if member["conversion_status"] == "rejected"
        ]
        self.assertEqual(len(rejected_members), 2)
        self.assertTrue(
            all(member["record_id"] is None for member in rejected_members)
        )

    def test_independent_staged_builds_are_byte_deterministic_for_all_11_files(self):
        first = self.build()
        second_parent = self.temporary_root / "staging-second"
        second_parent.mkdir()
        second = self.build(second_parent)

        first_files = {
            path.relative_to(self.staging_parent).as_posix(): path.read_bytes()
            for path in sorted(self.staging_parent.rglob("*"))
            if path.is_file()
        }
        second_files = {
            path.relative_to(second_parent).as_posix(): path.read_bytes()
            for path in sorted(second_parent.rglob("*"))
            if path.is_file()
        }
        self.assertEqual(set(first_files), set(second_files))
        self.assertEqual(len(first_files), 11)
        self.assertEqual(first_files, second_files)
        self.assertEqual(first.pair_index_sha256, second.pair_index_sha256)

    def test_comparator_reads_one_hdf5_record_row_at_a_time(self):
        staged = self.build()
        real_getitem = h5py.Dataset.__getitem__
        intensity_keys = []
        events = []

        def recording_getitem(dataset, key):
            if dataset.name.endswith("/intensity"):
                intensity_keys.append(key)
                events.append("hdf5")
            return real_getitem(dataset, key)

        import rpe.io.rruff as rruff_module

        real_expected = rruff_module._expected_record_for_member

        def recording_expected(*args, **kwargs):
            events.append("source")
            return real_expected(*args, **kwargs)

        with patch.object(
            h5py.Dataset,
            "__getitem__",
            new=recording_getitem,
        ), patch(
            "rpe.io.rruff._expected_record_for_member",
            side_effect=recording_expected,
        ):
                comparison = _compare_rruff_dataset(
                    self.raw_root,
                    self.staging_parent / staged.raw.dataset_id,
                    self.bundle.inspection,
                    kind="raw",
                )
        self.assertEqual(comparison.source_rows_compared, 11)
        self.assertTrue(intensity_keys)
        self.assertTrue(
            all(
                isinstance(key, (int, np.integer))
                for key in intensity_keys
            )
        )
        self.assertEqual(len(events), 22)
        self.assertEqual(
            events,
            ["source", "hdf5"] * 11,
        )

    def test_comparator_rejects_mutated_intensity(self):
        staged = self.build()
        dataset_path = self.staging_parent / staged.raw.dataset_id
        with h5py.File(dataset_path / "arrays.h5", "r+") as arrays:
            group = next(iter(arrays["axes"].values()))
            group["intensity"][0, 0] += np.float32(0.25)

        with self.assertRaises(RruffValidationError) as caught:
            _compare_rruff_dataset(
                self.raw_root,
                dataset_path,
                self.bundle.inspection,
                kind="raw",
            )
        self.assertEqual(
            caught.exception.path,
            "rruff_raman_raw.arrays.h5.intensity",
        )

    def test_comparator_rejects_label_and_source_metadata_corruption(self):
        mutations = (
            (
                lambda records: records[0]["targets"].__setitem__(
                    "class_label",
                    999,
                ),
                "targets.class_label",
            ),
            (
                lambda records: records[0]["meta"]["source_metadata"].__setitem__(
                    "source_member",
                    "wrong-member.txt",
                ),
                "meta.source_metadata.source_member",
            ),
            (
                lambda records: records[0]["meta"].__setitem__(
                    "preprocessing_status",
                    "unknown",
                ),
                "meta.preprocessing_status",
            ),
            (
                lambda records: records[0]["meta"]["source_metadata"].__setitem__(
                    "pair_id",
                    "0" * 64,
                ),
                "meta.source_metadata.pair_id",
            ),
            (
                lambda records: records[0]["meta"]["source_metadata"].__setitem__(
                    "pair_status",
                    "ambiguous",
                ),
                "meta.source_metadata.pair_status",
            ),
        )
        for index, (mutation, suffix) in enumerate(mutations):
            with self.subTest(suffix=suffix):
                parent = self.temporary_root / f"corruption-{index}"
                parent.mkdir()
                staged = self.build(parent)
                dataset_path = parent / staged.raw.dataset_id
                records_path = dataset_path / "records.jsonl"
                records = [
                    json.loads(line)
                    for line in records_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                ]
                mutation(records)
                records_path.write_bytes(
                    b"".join(
                        _canonical_json_bytes(record)
                        for record in records
                    )
                )
                with self.assertRaises(RruffValidationError) as caught:
                    _compare_rruff_dataset(
                        self.raw_root,
                        dataset_path,
                        self.bundle.inspection,
                        kind="raw",
                    )
                self.assertEqual(
                    caught.exception.path,
                    f"rruff_raman_raw.records.jsonl[0].{suffix}",
                )

    def test_comparator_rejects_manifest_class_map_corruption(self):
        staged = self.build()
        dataset_path = self.staging_parent / staged.raw.dataset_id
        manifest_path = dataset_path / "dataset.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        first_label = next(iter(manifest["class_labels"]))
        manifest["class_labels"][first_label] = "Wrong mineral"
        manifest_path.write_bytes(_canonical_json_bytes(manifest))

        with self.assertRaises(RruffValidationError) as caught:
            _compare_rruff_dataset(
                self.raw_root,
                dataset_path,
                self.bundle.inspection,
                kind="raw",
            )
        self.assertEqual(
            caught.exception.path,
            "rruff_raman_raw.dataset.json.class_labels",
        )

    def test_staged_build_rejects_wrong_parent_and_existing_artifacts(self):
        missing_parent = self.temporary_root / "missing"
        with self.assertRaises(RruffValidationError) as caught:
            self.build(missing_parent)
        self.assertEqual(caught.exception.path, "staging_parent")

        existing_parent = self.temporary_root / "existing"
        existing_parent.mkdir()
        (existing_parent / "rruff_raman_pairs.jsonl").write_text(
            "existing\n",
            encoding="utf-8",
        )
        with self.assertRaises(RruffValidationError) as caught:
            self.build(existing_parent)
        self.assertEqual(
            caught.exception.path,
            "staging_parent/rruff_raman_pairs.jsonl",
        )

    def test_staged_build_rejects_bundle_statistics_drift(self):
        drifted_bundle = replace(
            self.bundle,
            statistics=replace(
                self.bundle.statistics,
                source_members=999,
            ),
        )

        with self.assertRaises(RruffValidationError) as caught:
            _build_rruff_data_staged(
                self.raw_root,
                self.staging_parent,
                drifted_bundle,
            )
        self.assertEqual(
            caught.exception.path,
            "bundle.statistics.source_members",
        )

    def test_staged_build_reports_exact_timing_keys(self):
        timings = {}

        staged = _build_rruff_data_staged(
            self.raw_root,
            self.staging_parent,
            self.bundle,
            timings=timings,
        )

        self.assertEqual(
            set(timings),
            {
                "raw_build_seconds",
                "processed_build_seconds",
                "pair_index_seconds",
                "validation_seconds",
            },
        )
        self.assertTrue(all(value >= 0.0 for value in timings.values()))
        self.assertEqual(staged.pair_index_rows, 13)

    def test_staged_build_does_not_mutate_source_or_final_output(self):
        source_hashes = dict(self.source.archive_sha256)
        self.output_root.mkdir()
        sentinel = self.output_root / "sentinel.txt"
        sentinel.write_text("owned by another task\n", encoding="utf-8")
        before = artifact_snapshot(self.output_root)

        self.build()

        self.assertEqual(
            {
                path.name: file_sha256(path)
                for path in sorted(self.raw_root.glob("*.zip"))
            },
            source_hashes,
        )
        self.assertEqual(artifact_snapshot(self.output_root), before)


RECEIPT_KEYS = {
    "adapter_version",
    "schema_version",
    "source_archives",
    "source_inventory",
    "parser",
    "rejected_members",
    "global_class_labels",
    "datasets",
    "pairing",
    "precision",
    "license",
    "limitations",
    "outputs",
}
SOURCE_ARCHIVE_KEYS = {
    "archive",
    "url",
    "bytes",
    "sha256",
    "retrieved_date",
    "member_count",
    "raw_members",
    "processed_members",
}
SOURCE_INVENTORY_KEYS = {
    "source_members",
    "source_raw",
    "source_processed",
    "accepted_records",
    "rejected_records",
    "accepted_raw",
    "accepted_processed",
    "accepted_spectral_points",
    "rejected_member_parsed_numeric_rows",
    "mineral_classes",
    "rruff_samples",
    "raw_mineral_classes",
    "processed_mineral_classes",
    "raw_rruff_samples",
    "processed_rruff_samples",
    "raw_axis_groups",
    "processed_axis_groups",
    "increasing_axes",
    "decreasing_axes",
    "records_with_excitation_nm",
    "records_with_excitation_nm_none",
    "records_with_full_comment_acquisition_metadata",
}
PARSER_KEYS = {
    "utf8_members",
    "cp1252_members",
    "comma_numeric_rows",
    "whitespace_numeric_rows",
    "ignored_comment_lines",
    "ignored_preamble_lines",
    "ignored_column_header_lines",
    "duplicate_header_records",
}
REJECTED_MEMBER_KEYS = {
    "archive",
    "source_member",
    "member_id",
    "member_sha256",
    "member_bytes",
    "measurement_key",
    "kind",
    "rejection_code",
    "reason",
}
DATASET_RECEIPT_KEYS = {
    "records",
    "eligible_records",
    "observed_class_ids",
    "axis_groups",
}
PAIRING_RECEIPT_KEYS = {
    "index_file",
    "index_rows",
    "status_counts",
    "axis_relation_counts",
    "pointwise_comparable_pairs",
}
PRECISION_KEYS = {
    "axis_float32_max_abs_error_cm1",
    "intensity_float32_max_abs_error",
}
LICENSE_KEYS = {
    "text",
    "status",
    "access_status",
    "redistribution_requires_review",
}
LIMITATION_KEYS = {
    "processed_is_not_clean_target",
    "processed_operations_unknown",
    "rejected_source_members",
    "alignment_required_pairs",
    "no_overlap_pairs",
    "taxonomy_normalization_applied",
    "resampling_applied",
}
OUTPUT_KEYS = {
    "rruff_raman_raw",
    "rruff_raman_processed",
    "rruff_raman_pairs.jsonl",
}
OUTPUT_ARTIFACT_KEYS = {"artifact_type", "files"}
OUTPUT_FILE_KEYS = {"bytes", "sha256"}


class RruffReceiptAndCompleteStagingTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.contract = synthetic_pairing_contract(self.source)
        self.bundle = _inspect_rruff_bundle(self.raw_root, self.contract)
        self.staging_parent = self.temporary_root / "staging"
        self.staging_parent.mkdir()

    def require_task7_interfaces(self):
        required = (
            "RruffConversionSummary",
            "_receipt_document",
            "_validate_rruff_receipt",
            "_build_rruff_staged",
        )
        missing = [
            name for name in required if not hasattr(rruff_module, name)
        ]
        self.assertEqual(
            missing,
            [],
            f"Task 7 interfaces are missing: {missing}",
        )

    def build(self, staging_parent: Path | None = None, *, timings=None):
        self.require_task7_interfaces()
        return rruff_module._build_rruff_staged(
            self.raw_root,
            self.staging_parent
            if staging_parent is None
            else staging_parent,
            self.bundle,
            timings=timings,
        )

    def receipt(self, staging_parent: Path | None = None):
        parent = (
            self.staging_parent
            if staging_parent is None
            else staging_parent
        )
        return json.loads(
            (parent / "rruff_raman_conversion.json").read_text(
                encoding="utf-8",
            )
        )

    def write_receipt(self, receipt):
        path = self.staging_parent / "rruff_raman_conversion.json"
        path.write_bytes(_canonical_json_bytes(receipt))
        return path

    def validate_receipt(self):
        return rruff_module._validate_rruff_receipt(
            self.staging_parent / "rruff_raman_conversion.json",
            self.bundle,
        )

    def test_task7_interfaces_exist(self):
        self.require_task7_interfaces()

    def test_receipt_has_exact_schema_semantics_and_11_file_attributions(self):
        summary, _, _ = self.build()
        receipt = self.receipt()

        self.assertEqual(set(receipt), RECEIPT_KEYS)
        self.assertEqual(receipt["adapter_version"], "0.1.0")
        self.assertEqual(receipt["schema_version"], "0.1.0")

        archives = receipt["source_archives"]
        self.assertEqual(
            [archive["archive"] for archive in archives],
            list(ARCHIVE_NAMES),
        )
        self.assertTrue(
            all(set(archive) == SOURCE_ARCHIVE_KEYS for archive in archives)
        )
        archive_counts = {
            archive["archive"]: (
                archive["member_count"],
                archive["raw_members"],
                archive["processed_members"],
            )
            for archive in archives
        }
        self.assertEqual(
            archive_counts,
            {
                "LR-Raman.zip": (20, 12, 8),
                "excellent_oriented.zip": (0, 0, 0),
                "excellent_unoriented.zip": (0, 0, 0),
                "fair_oriented.zip": (1, 1, 0),
                "fair_unoriented.zip": (0, 0, 0),
                "poor_unoriented.zip": (0, 0, 0),
                "unrated_oriented.zip": (0, 0, 0),
                "unrated_unoriented.zip": (0, 0, 0),
            },
        )
        for archive in archives:
            archive_path = self.raw_root / archive["archive"]
            self.assertEqual(archive["bytes"], archive_path.stat().st_size)
            self.assertEqual(archive["sha256"], file_sha256(archive_path))
            self.assertEqual(archive["retrieved_date"], "2026-08-14")
            self.assertEqual(
                archive["url"],
                (
                    "https://rruff.info/zipped_data_files/raman/"
                    f"{archive['archive']}"
                ),
            )

        inventory = receipt["source_inventory"]
        self.assertEqual(set(inventory), SOURCE_INVENTORY_KEYS)
        self.assertEqual(
            inventory,
            {
                "source_members": 21,
                "source_raw": 13,
                "source_processed": 8,
                "accepted_records": 19,
                "rejected_records": 2,
                "accepted_raw": 11,
                "accepted_processed": 8,
                "accepted_spectral_points": 46,
                "rejected_member_parsed_numeric_rows": 4,
                "mineral_classes": 11,
                "rruff_samples": 11,
                "raw_mineral_classes": 9,
                "processed_mineral_classes": 7,
                "raw_rruff_samples": 9,
                "processed_rruff_samples": 7,
                "raw_axis_groups": 10,
                "processed_axis_groups": 8,
                "increasing_axes": 19,
                "decreasing_axes": 0,
                "records_with_excitation_nm": 19,
                "records_with_excitation_nm_none": 0,
                "records_with_full_comment_acquisition_metadata": 0,
            },
        )
        parser = receipt["parser"]
        self.assertEqual(set(parser), PARSER_KEYS)
        self.assertEqual(
            parser,
            {
                "utf8_members": 21,
                "cp1252_members": 0,
                "comma_numeric_rows": 50,
                "whitespace_numeric_rows": 0,
                "ignored_comment_lines": 0,
                "ignored_preamble_lines": 0,
                "ignored_column_header_lines": 0,
                "duplicate_header_records": 0,
            },
        )

        rejected = receipt["rejected_members"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(
            all(set(member) == REJECTED_MEMBER_KEYS for member in rejected)
        )
        self.assertEqual(
            [
                (
                    member["archive"],
                    member["kind"],
                    member["rejection_code"],
                )
                for member in rejected
            ],
            [
                ("LR-Raman.zip", "raw", "AXIS_DUPLICATE"),
                ("LR-Raman.zip", "raw", "AXIS_DUPLICATE"),
            ],
        )

        self.assertEqual(
            receipt["global_class_labels"],
            {
                "0": "Alignite",
                "1": "Crossarchiveite",
                "2": "Exactite",
                "3": "Nooverlapite",
                "4": "Processedonlyite",
                "5": "Procsubsetite",
                "6": "Rawonlyite",
                "7": "Rawsubsetite",
                "8": "Rejectedduplicateite",
                "9": "Twoprocessedite",
                "10": "Tworawite",
            },
        )
        self.assertEqual(
            set(receipt["datasets"]),
            {"rruff_raman_raw", "rruff_raman_processed"},
        )
        self.assertTrue(
            all(
                set(dataset) == DATASET_RECEIPT_KEYS
                for dataset in receipt["datasets"].values()
            )
        )
        self.assertEqual(
            receipt["datasets"],
            {
                "rruff_raman_raw": {
                    "records": 11,
                    "eligible_records": 11,
                    "observed_class_ids": [0, 1, 2, 3, 5, 6, 7, 8, 10],
                    "axis_groups": 10,
                },
                "rruff_raman_processed": {
                    "records": 8,
                    "eligible_records": 0,
                    "observed_class_ids": [0, 2, 3, 4, 5, 7, 9],
                    "axis_groups": 8,
                },
            },
        )

        pairing = receipt["pairing"]
        self.assertEqual(set(pairing), PAIRING_RECEIPT_KEYS)
        self.assertEqual(pairing["index_file"], "rruff_raman_pairs.jsonl")
        self.assertEqual(pairing["index_rows"], 13)
        self.assertEqual(
            pairing["status_counts"],
            {
                "paired_unique": 5,
                "raw_only": 3,
                "processed_only": 1,
                "ambiguous": 3,
                "rejected_only": 1,
            },
        )
        self.assertEqual(
            pairing["axis_relation_counts"],
            {
                "exact_equal": 1,
                "processed_exact_contiguous_subset_of_raw": 1,
                "raw_exact_contiguous_subset_of_processed": 1,
                "overlap_requires_alignment": 1,
                "no_axis_overlap": 1,
            },
        )
        self.assertEqual(pairing["pointwise_comparable_pairs"], 3)

        self.assertEqual(set(receipt["precision"]), PRECISION_KEYS)
        self.assertEqual(
            receipt["precision"],
            {
                "axis_float32_max_abs_error_cm1": 0.0,
                "intensity_float32_max_abs_error": (
                    1.1920928966180355e-08
                ),
            },
        )
        self.assertEqual(set(receipt["license"]), LICENSE_KEYS)
        self.assertEqual(
            receipt["license"],
            {
                "text": "not stated",
                "status": "not_stated",
                "access_status": "free public access",
                "redistribution_requires_review": True,
            },
        )
        self.assertEqual(set(receipt["limitations"]), LIMITATION_KEYS)
        self.assertEqual(
            receipt["limitations"],
            {
                "processed_is_not_clean_target": True,
                "processed_operations_unknown": True,
                "rejected_source_members": 2,
                "alignment_required_pairs": 1,
                "no_overlap_pairs": 1,
                "taxonomy_normalization_applied": False,
                "resampling_applied": False,
            },
        )

        outputs = receipt["outputs"]
        self.assertEqual(set(outputs), OUTPUT_KEYS)
        attributed_files = []
        for artifact_name, output in outputs.items():
            self.assertEqual(set(output), OUTPUT_ARTIFACT_KEYS)
            expected_type = (
                "file"
                if artifact_name == "rruff_raman_pairs.jsonl"
                else "dataset"
            )
            self.assertEqual(output["artifact_type"], expected_type)
            expected_names = (
                {"rruff_raman_pairs.jsonl"}
                if expected_type == "file"
                else set(DATASET_FILES)
            )
            self.assertEqual(set(output["files"]), expected_names)
            for name, details in output["files"].items():
                self.assertEqual(set(details), OUTPUT_FILE_KEYS)
                path = (
                    self.staging_parent / name
                    if expected_type == "file"
                    else self.staging_parent / artifact_name / name
                )
                self.assertEqual(details["bytes"], path.stat().st_size)
                self.assertEqual(details["sha256"], file_sha256(path))
                attributed_files.append(path)
        self.assertEqual(len(attributed_files), 11)
        self.assertNotIn(
            "rruff_raman_conversion.json",
            {
                path.name
                for path in attributed_files
            },
        )
        self.assertEqual(
            summary.receipt_path.read_bytes(),
            _canonical_json_bytes(receipt),
        )

    def test_receipt_validator_rejects_schema_semantic_and_provenance_corruption(self):
        self.build()
        original = self.receipt()
        mutations = (
            (
                "missing top-level key",
                lambda value: value.pop("parser"),
                "receipt",
            ),
            (
                "unknown runtime field",
                lambda value: value.__setitem__("total_seconds", 1.0),
                "receipt",
            ),
            (
                "runtime timestamp",
                lambda value: value.__setitem__(
                    "generated_at",
                    "2026-08-15T00:00:00Z",
                ),
                "receipt",
            ),
            (
                "wrong structural type",
                lambda value: value.__setitem__("source_archives", {}),
                "receipt.source_archives",
            ),
            (
                "boolean as integer",
                lambda value: value["source_inventory"].__setitem__(
                    "accepted_records",
                    True,
                ),
                "receipt.source_inventory.accepted_records",
            ),
            (
                "count inconsistency",
                lambda value: value["source_inventory"].__setitem__(
                    "accepted_records",
                    18,
                ),
                "receipt.source_inventory.accepted_records",
            ),
            (
                "Croissant license claim",
                lambda value: value["license"].__setitem__(
                    "text",
                    "CC BY 4.0",
                ),
                "receipt.license.text",
            ),
            (
                "processed clean claim",
                lambda value: value["limitations"].__setitem__(
                    "processed_is_not_clean_target",
                    False,
                ),
                "receipt.limitations.processed_is_not_clean_target",
            ),
            (
                "absolute output path",
                lambda value: value["pairing"].__setitem__(
                    "index_file",
                    "/tmp/rruff_raman_pairs.jsonl",
                ),
                "receipt.pairing.index_file",
            ),
            (
                "staging path",
                lambda value: value["pairing"].__setitem__(
                    "index_file",
                    ".rruff.staging-123/rruff_raman_pairs.jsonl",
                ),
                "receipt.pairing.index_file",
            ),
            (
                "wrong attributed bytes",
                lambda value: value["outputs"][
                    "rruff_raman_pairs.jsonl"
                ]["files"]["rruff_raman_pairs.jsonl"].__setitem__(
                    "bytes",
                    1,
                ),
                (
                    "receipt.outputs.rruff_raman_pairs.jsonl.files."
                    "rruff_raman_pairs.jsonl.bytes"
                ),
            ),
            (
                "wrong attributed hash",
                lambda value: value["outputs"][
                    "rruff_raman_pairs.jsonl"
                ]["files"]["rruff_raman_pairs.jsonl"].__setitem__(
                    "sha256",
                    "0" * 64,
                ),
                (
                    "receipt.outputs.rruff_raman_pairs.jsonl.files."
                    "rruff_raman_pairs.jsonl.sha256"
                ),
            ),
        )
        for name, mutation, expected_path in mutations:
            with self.subTest(name=name):
                value = json.loads(json.dumps(original))
                mutation(value)
                self.write_receipt(value)
                with self.assertRaises(RruffValidationError) as caught:
                    self.validate_receipt()
                self.assertEqual(caught.exception.path, expected_path)

        path = self.write_receipt(original)
        path.write_bytes(b" " + path.read_bytes())
        with self.assertRaises(RruffValidationError) as caught:
            self.validate_receipt()
        self.assertEqual(caught.exception.path, "receipt")

        nonfinite = json.loads(json.dumps(original))
        nonfinite["precision"]["axis_float32_max_abs_error_cm1"] = float("nan")
        path.write_text(
            json.dumps(
                nonfinite,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=True,
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(RruffValidationError) as caught:
            self.validate_receipt()
        self.assertEqual(caught.exception.path, "receipt")

    def test_receipt_validator_rejects_attributed_file_corruption(self):
        self.build()
        pair_index = self.staging_parent / "rruff_raman_pairs.jsonl"
        pair_index.write_bytes(pair_index.read_bytes() + b"\n")

        with self.assertRaises(RruffValidationError) as caught:
            self.validate_receipt()
        self.assertEqual(
            caught.exception.path,
            (
                "receipt.outputs.rruff_raman_pairs.jsonl.files."
                "rruff_raman_pairs.jsonl.bytes"
            ),
        )

    def test_complete_staging_returns_frozen_summary_and_comparisons(self):
        summary, raw_comparison, processed_comparison = self.build()

        self.assertIsInstance(
            summary,
            rruff_module.RruffConversionSummary,
        )
        self.assertIsInstance(raw_comparison, _RruffComparisonSummary)
        self.assertIsInstance(processed_comparison, _RruffComparisonSummary)
        self.assertEqual(raw_comparison.source_rows_compared, 11)
        self.assertEqual(processed_comparison.source_rows_compared, 8)
        for comparison in (raw_comparison, processed_comparison):
            self.assertEqual(comparison.intensity_mismatches, 0)
            self.assertEqual(comparison.class_label_mismatches, 0)
            self.assertEqual(comparison.eligibility_violations, 0)
        self.assertEqual(summary.pair_index_rows, 13)
        self.assertEqual(summary.rejected_records, 2)
        self.assertEqual(
            summary.receipt_sha256,
            file_sha256(summary.receipt_path),
        )
        self.assertEqual(
            set(path.name for path in self.staging_parent.iterdir()),
            {
                "rruff_raman_raw",
                "rruff_raman_processed",
                "rruff_raman_pairs.jsonl",
                "rruff_raman_conversion.json",
            },
        )
        files = [
            path
            for path in self.staging_parent.rglob("*")
            if path.is_file()
        ]
        self.assertEqual(len(files), 12)
        with self.assertRaises(FrozenInstanceError):
            summary.rejected_records = 0
        with self.assertRaises(TypeError):
            summary.pair_status_counts["paired_unique"] = 0
        with self.assertRaises(TypeError):
            summary.raw.files["dataset.json"]["bytes"] = 0

    def test_complete_staging_preflights_existing_receipt_before_data_build(self):
        receipt_path = (
            self.staging_parent / "rruff_raman_conversion.json"
        )
        receipt_path.write_text("existing\n", encoding="utf-8")

        with self.assertRaises(RruffValidationError) as caught:
            self.build()
        self.assertEqual(
            caught.exception.path,
            "staging_parent/rruff_raman_conversion.json",
        )
        self.assertEqual(
            {path.name for path in self.staging_parent.iterdir()},
            {"rruff_raman_conversion.json"},
        )

    def test_complete_staging_rejects_archive_hash_drift_after_inspection(self):
        archive_path = self.raw_root / "excellent_oriented.zip"
        with archive_path.open("ab") as archive:
            archive.write(b"post-inspection drift")

        with self.assertRaises(RruffValidationError) as caught:
            self.build()
        self.assertEqual(
            caught.exception.path,
            (
                "receipt.source_archives."
                "excellent_oriented.zip.sha256"
            ),
        )
        self.assertEqual(list(self.staging_parent.iterdir()), [])

    def test_complete_staging_reports_receipt_timing_without_runtime_in_receipt(self):
        timings = {}
        self.build(timings=timings)

        self.assertEqual(
            set(timings),
            {
                "raw_build_seconds",
                "processed_build_seconds",
                "pair_index_seconds",
                "receipt_seconds",
                "validation_seconds",
            },
        )
        self.assertTrue(all(value >= 0.0 for value in timings.values()))
        receipt = self.receipt()
        self.assertTrue(
            {
                "total_seconds",
                "receipt_seconds",
                "validation_seconds",
                "peak_rss_bytes",
            }.isdisjoint(receipt),
        )

    def test_independent_complete_staging_builds_are_12_file_deterministic(self):
        first, _, _ = self.build()
        second_parent = self.temporary_root / "staging-second"
        second_parent.mkdir()
        second, _, _ = self.build(second_parent)

        first_files = {
            path.relative_to(self.staging_parent).as_posix(): (
                file_sha256(path)
            )
            for path in sorted(self.staging_parent.rglob("*"))
            if path.is_file()
        }
        second_files = {
            path.relative_to(second_parent).as_posix(): file_sha256(path)
            for path in sorted(second_parent.rglob("*"))
            if path.is_file()
        }
        self.assertEqual(len(first_files), 12)
        self.assertEqual(first_files, second_files)
        self.assertEqual(first.receipt_sha256, second.receipt_sha256)


RUNTIME_METRIC_FIELDS = (
    "total_seconds",
    "inspection_seconds",
    "staged_build_seconds",
    "raw_build_seconds",
    "processed_build_seconds",
    "pair_index_seconds",
    "receipt_seconds",
    "validation_seconds",
    "peak_rss_raw",
    "peak_rss_bytes",
    "raw_output_bytes",
    "processed_output_bytes",
    "pair_index_bytes",
    "receipt_bytes",
    "combined_output_bytes",
    "source_archive_bytes",
    "compression_ratio",
    "raw_axis_groups",
    "processed_axis_groups",
    "lazy_open_seconds_raw",
    "lazy_open_seconds_processed",
    "representative_record_read_seconds_raw",
    "representative_record_read_seconds_processed",
)
STAGING_REVIEW_FIELDS = (
    "raw",
    "processed",
    "pair_index_sha256",
    "pair_index_rows",
    "pair_status_counts",
    "axis_relation_counts",
    "receipt_sha256",
    "rejected_records",
    "raw_comparison",
    "processed_comparison",
    "source_statistics",
    "output_hashes",
    "metrics",
    "feasibility_failures",
)
RESOURCE_FAILURES = (
    "peak_rss_bytes",
    "combined_output_bytes",
    "staged_build_seconds",
    "validation_seconds",
    "lazy_open_seconds_raw",
    "lazy_open_seconds_processed",
    "representative_record_read_seconds_raw",
    "representative_record_read_seconds_processed",
)
FIXED_GATE_FAILURES = (
    "raw_axis_groups",
    "processed_axis_groups",
    "axis_groups_total",
    "accepted_records",
    "source_rows_compared",
    "intensity_mismatches",
    "class_label_mismatches",
    "eligibility_violations",
)


class RruffRuntimeAndFeasibilityTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.contract = synthetic_pairing_contract(self.source)
        self.bundle = _inspect_rruff_bundle(self.raw_root, self.contract)
        self.staging_parent = self.temporary_root / "staging"
        self.staging_parent.mkdir()

    def require_task8_interfaces(self):
        required = (
            "RruffRuntimeMetrics",
            "_RruffStagingReview",
            "_linux_ru_maxrss_to_bytes",
            "_measure_rruff_dataset_access",
            "_rruff_feasibility_failures",
            "_stage_rruff_for_review",
        )
        missing = [
            name for name in required if not hasattr(rruff_module, name)
        ]
        self.assertEqual(
            missing,
            [],
            f"Task 8 interfaces are missing: {missing}",
        )

    def build_complete(self):
        return rruff_module._build_rruff_staged(
            self.raw_root,
            self.staging_parent,
            self.bundle,
        )

    def metrics(self, **changes):
        self.require_task8_interfaces()
        values = {
            "total_seconds": 1.0,
            "inspection_seconds": 0.1,
            "staged_build_seconds": 0.4,
            "raw_build_seconds": 0.1,
            "processed_build_seconds": 0.1,
            "pair_index_seconds": 0.1,
            "receipt_seconds": 0.1,
            "validation_seconds": 0.1,
            "peak_rss_raw": 1,
            "peak_rss_bytes": 1024,
            "raw_output_bytes": 1,
            "processed_output_bytes": 1,
            "pair_index_bytes": 1,
            "receipt_bytes": 1,
            "combined_output_bytes": 4,
            "source_archive_bytes": 8,
            "compression_ratio": 0.5,
            "raw_axis_groups": 10_260,
            "processed_axis_groups": 11_009,
            "lazy_open_seconds_raw": 0.1,
            "lazy_open_seconds_processed": 0.1,
            "representative_record_read_seconds_raw": 0.1,
            "representative_record_read_seconds_processed": 0.1,
        }
        values.update(changes)
        return rruff_module.RruffRuntimeMetrics(**values)

    def production_gate_inputs(self):
        summary, raw_comparison, processed_comparison = self.build_complete()
        summary = replace(
            summary,
            raw=replace(
                summary.raw,
                record_count=20_664,
                axis_group_count=10_260,
            ),
            processed=replace(
                summary.processed,
                record_count=17_379,
                axis_group_count=11_009,
            ),
        )
        comparisons = (
            replace(raw_comparison, source_rows_compared=20_664),
            replace(processed_comparison, source_rows_compared=17_379),
        )
        return summary, comparisons

    def test_task8_dataclasses_have_exact_field_order(self):
        self.require_task8_interfaces()

        self.assertEqual(
            tuple(field.name for field in fields(
                rruff_module.RruffRuntimeMetrics
            )),
            RUNTIME_METRIC_FIELDS,
        )
        self.assertEqual(
            tuple(field.name for field in fields(
                rruff_module._RruffStagingReview
            )),
            STAGING_REVIEW_FIELDS,
        )
        metrics = self.metrics()
        with self.assertRaises(FrozenInstanceError):
            metrics.total_seconds = 2.0

    def test_linux_ru_maxrss_is_converted_from_kib_to_bytes(self):
        self.require_task8_interfaces()

        self.assertEqual(rruff_module._linux_ru_maxrss_to_bytes(0), 0)
        self.assertEqual(rruff_module._linux_ru_maxrss_to_bytes(1), 1024)
        self.assertEqual(
            rruff_module._linux_ru_maxrss_to_bytes(4096),
            4_194_304,
        )
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    rruff_module._linux_ru_maxrss_to_bytes(invalid)

    def test_all_strict_resource_thresholds_fail_at_equality(self):
        summary, comparisons = self.production_gate_inputs()
        metrics = self.metrics(
            peak_rss_bytes=8_589_934_592,
            combined_output_bytes=4_294_967_296,
            staged_build_seconds=3600.0,
            validation_seconds=1800.0,
            lazy_open_seconds_raw=60.0,
            lazy_open_seconds_processed=60.0,
            representative_record_read_seconds_raw=5.0,
            representative_record_read_seconds_processed=5.0,
        )

        self.assertEqual(
            rruff_module._rruff_feasibility_failures(
                summary,
                metrics,
                comparisons,
            ),
            RESOURCE_FAILURES,
        )
        passing = replace(
            metrics,
            peak_rss_bytes=8_589_934_591,
            combined_output_bytes=4_294_967_295,
            staged_build_seconds=3599.999,
            validation_seconds=1799.999,
            lazy_open_seconds_raw=59.999,
            lazy_open_seconds_processed=59.999,
            representative_record_read_seconds_raw=4.999,
            representative_record_read_seconds_processed=4.999,
        )
        self.assertEqual(
            rruff_module._rruff_feasibility_failures(
                summary,
                passing,
                comparisons,
            ),
            (),
        )

    def test_resource_thresholds_fail_closed_on_nonfinite_or_negative_values(self):
        summary, comparisons = self.production_gate_inputs()
        for name in RESOURCE_FAILURES:
            with self.subTest(name=name, value="nan"):
                failures = rruff_module._rruff_feasibility_failures(
                    summary,
                    replace(self.metrics(), **{name: float("nan")}),
                    comparisons,
                )
                self.assertIn(name, failures)
            with self.subTest(name=name, value="negative"):
                failures = rruff_module._rruff_feasibility_failures(
                    summary,
                    replace(self.metrics(), **{name: -0.1}),
                    comparisons,
                )
                self.assertIn(name, failures)

    def test_fixed_count_and_comparison_gates_report_all_failures(self):
        self.require_task8_interfaces()
        summary, comparisons = self.production_gate_inputs()
        summary = replace(
            summary,
            raw=replace(
                summary.raw,
                record_count=0,
                axis_group_count=0,
            ),
            processed=replace(
                summary.processed,
                axis_group_count=0,
            ),
        )
        comparisons = (
            replace(
                comparisons[0],
                source_rows_compared=0,
                intensity_mismatches=1,
                class_label_mismatches=1,
                eligibility_violations=1,
            ),
            comparisons[1],
        )

        self.assertEqual(
            rruff_module._rruff_feasibility_failures(
                summary,
                self.metrics(
                    raw_axis_groups=0,
                    processed_axis_groups=0,
                ),
                comparisons,
            ),
            FIXED_GATE_FAILURES,
        )

    def test_lazy_open_probe_selects_literal_ids_and_closes_handles(self):
        self.require_task8_interfaces()
        summary, _, _ = self.build_complete()
        opened = []
        real_open = UnifiedDataset.open

        def recording_open(path, **kwargs):
            dataset = real_open(path, **kwargs)
            opened.append(dataset)
            return dataset

        selected = {}
        with patch(
            "rpe.io.rruff.UnifiedDataset.open",
            side_effect=recording_open,
        ):
            for dataset_id in (
                "rruff_raman_raw",
                "rruff_raman_processed",
            ):
                selected[dataset_id] = (
                    rruff_module._measure_rruff_dataset_access(
                        self.staging_parent / dataset_id,
                        dataset_id,
                    )
                )

        self.assertEqual(
            {
                dataset_id: result[2]
                for dataset_id, result in selected.items()
            },
            {
                "rruff_raman_raw": (
                    "raw-1000000000000000000000000007"
                ),
                "rruff_raman_processed": (
                    "processed-1000000000000000000000000006"
                ),
            },
        )
        for open_seconds, read_seconds, _ in selected.values():
            self.assertGreaterEqual(open_seconds, 0.0)
            self.assertGreaterEqual(read_seconds, 0.0)
        self.assertEqual(len(opened), 2)
        for dataset in opened:
            with self.assertRaises(DatasetClosedError):
                _ = dataset.record_ids
        self.assertEqual(summary.raw.record_count, 11)
        self.assertEqual(summary.processed.record_count, 8)


class RruffStagingReviewTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.contract = synthetic_pairing_contract(self.source)
        self.bundle = _inspect_rruff_bundle(self.raw_root, self.contract)
        self.output_root = self.temporary_root / "output"
        required = (
            "RruffRuntimeMetrics",
            "_RruffStagingReview",
            "_linux_ru_maxrss_to_bytes",
            "_measure_rruff_dataset_access",
            "_rruff_feasibility_failures",
            "_stage_rruff_for_review",
        )
        missing = [
            name for name in required if not hasattr(rruff_module, name)
        ]
        self.assertEqual(
            missing,
            [],
            f"Task 8 interfaces are missing: {missing}",
        )

    def review(self, *, output_root: Path | None = None):
        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ):
            return rruff_module._stage_rruff_for_review(
                self.raw_root,
                self.output_root if output_root is None else output_root,
            )

    def assert_no_staging(self, output_root: Path) -> None:
        if not output_root.exists():
            return
        self.assertFalse(
            any(
                path.name.startswith(".rruff.staging-")
                for path in output_root.iterdir()
            )
        )

    def test_review_returns_complete_scalar_evidence_and_cleans_gate_failure(self):
        review = self.review()

        self.assertIsInstance(review, rruff_module._RruffStagingReview)
        self.assertEqual(review.raw.record_count, 11)
        self.assertEqual(review.processed.record_count, 8)
        self.assertEqual(review.raw.eligible_records, 11)
        self.assertEqual(review.processed.eligible_records, 0)
        self.assertEqual(review.pair_index_rows, 13)
        self.assertEqual(review.rejected_records, 2)
        self.assertEqual(
            dict(review.pair_status_counts),
            {
                "paired_unique": 5,
                "raw_only": 3,
                "processed_only": 1,
                "ambiguous": 3,
                "rejected_only": 1,
            },
        )
        self.assertEqual(
            dict(review.axis_relation_counts),
            {
                "exact_equal": 1,
                "processed_exact_contiguous_subset_of_raw": 1,
                "raw_exact_contiguous_subset_of_processed": 1,
                "overlap_requires_alignment": 1,
                "no_axis_overlap": 1,
            },
        )
        self.assertEqual(review.raw_comparison.source_rows_compared, 11)
        self.assertEqual(
            review.processed_comparison.source_rows_compared,
            8,
        )
        self.assertEqual(review.source_statistics.accepted_records, 19)
        self.assertEqual(review.source_statistics.rejected_records, 2)
        self.assertEqual(
            review.feasibility_failures,
            (
                "raw_axis_groups",
                "processed_axis_groups",
                "axis_groups_total",
                "accepted_records",
                "source_rows_compared",
            ),
        )
        self.assertEqual(len(review.output_hashes), 12)
        self.assertEqual(
            set(review.output_hashes),
            {
                *(f"rruff_raman_raw/{name}" for name in DATASET_FILES),
                *(
                    f"rruff_raman_processed/{name}"
                    for name in DATASET_FILES
                ),
                "rruff_raman_pairs.jsonl",
                "rruff_raman_conversion.json",
            },
        )
        self.assertEqual(
            review.output_hashes["rruff_raman_pairs.jsonl"],
            review.pair_index_sha256,
        )
        self.assertEqual(
            review.output_hashes["rruff_raman_conversion.json"],
            review.receipt_sha256,
        )
        with self.assertRaises(TypeError):
            review.output_hashes["new"] = "0" * 64
        with self.assertRaises(TypeError):
            review.pair_status_counts["paired_unique"] = 0
        with self.assertRaises(TypeError):
            review.source_statistics.rejection_code_counts[
                "AXIS_DUPLICATE"
            ] = 0
        with self.assertRaises(FrozenInstanceError):
            review.rejected_records = 0
        self.assertFalse(self.output_root.exists())

    def test_metrics_use_exact_formulas_and_never_enter_receipt(self):
        receipts = []
        real_build = rruff_module._build_rruff_staged

        def recording_build(*args, **kwargs):
            result = real_build(*args, **kwargs)
            receipts.append(
                json.loads(
                    result[0].receipt_path.read_text(encoding="utf-8")
                )
            )
            return result

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff._build_rruff_staged",
            side_effect=recording_build,
        ):
            review = rruff_module._stage_rruff_for_review(
                self.raw_root,
                self.output_root,
            )

        metrics = review.metrics
        self.assertEqual(
            tuple(field.name for field in fields(metrics)),
            RUNTIME_METRIC_FIELDS,
        )
        self.assertEqual(
            metrics.staged_build_seconds,
            metrics.raw_build_seconds
            + metrics.processed_build_seconds
            + metrics.pair_index_seconds
            + metrics.receipt_seconds,
        )
        self.assertEqual(metrics.raw_output_bytes, review.raw.output_bytes)
        self.assertEqual(
            metrics.processed_output_bytes,
            review.processed.output_bytes,
        )
        self.assertEqual(
            metrics.combined_output_bytes,
            metrics.raw_output_bytes
            + metrics.processed_output_bytes
            + metrics.pair_index_bytes
            + metrics.receipt_bytes,
        )
        self.assertEqual(
            metrics.source_archive_bytes,
            sum(
                (self.raw_root / name).stat().st_size
                for name in ARCHIVE_NAMES
            ),
        )
        self.assertEqual(
            metrics.compression_ratio,
            metrics.combined_output_bytes
            / metrics.source_archive_bytes,
        )
        self.assertEqual(metrics.raw_axis_groups, 10)
        self.assertEqual(metrics.processed_axis_groups, 8)
        self.assertEqual(
            metrics.peak_rss_bytes,
            rruff_module._linux_ru_maxrss_to_bytes(
                metrics.peak_rss_raw
            ),
        )
        component_seconds = (
            metrics.inspection_seconds
            + metrics.staged_build_seconds
            + metrics.validation_seconds
            + metrics.lazy_open_seconds_raw
            + metrics.lazy_open_seconds_processed
            + metrics.representative_record_read_seconds_raw
            + metrics.representative_record_read_seconds_processed
        )
        self.assertGreaterEqual(metrics.total_seconds, component_seconds)
        self.assertEqual(len(receipts), 1)
        for forbidden in RUNTIME_METRIC_FIELDS:
            self.assertNotIn(forbidden, receipts[0])
        self.assertFalse(self.output_root.exists())

    def test_review_allocates_one_direct_staging_child_and_preserves_siblings(self):
        self.output_root.mkdir()
        sentinel = self.output_root / "unowned-sentinel.txt"
        sentinel.write_text("keep me\n", encoding="utf-8")
        for dataset_id in (
            "rruff_raman_raw",
            "rruff_raman_processed",
        ):
            final = self.output_root / dataset_id
            final.mkdir()
            (final / "original.txt").write_text(
                f"original {dataset_id}\n",
                encoding="utf-8",
            )
        for file_name in (
            "rruff_raman_pairs.jsonl",
            "rruff_raman_conversion.json",
        ):
            (self.output_root / file_name).write_text(
                f"original {file_name}\n",
                encoding="utf-8",
            )
        before = artifact_snapshot(self.output_root)
        allocated = []
        real_mkdtemp = tempfile.mkdtemp

        def recording_mkdtemp(*args, **kwargs):
            path = Path(real_mkdtemp(*args, **kwargs))
            if (
                kwargs.get("prefix") == ".rruff.staging-"
                and Path(kwargs["dir"]) == self.output_root
            ):
                allocated.append(path)
            return str(path)

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff.tempfile.mkdtemp",
            side_effect=recording_mkdtemp,
        ), patch(
            "rpe.io.rruff._rruff_feasibility_failures",
            return_value=(),
        ):
            review = rruff_module._stage_rruff_for_review(
                self.raw_root,
                self.output_root,
            )

        self.assertEqual(review.feasibility_failures, ())
        self.assertEqual(len(allocated), 1)
        self.assertEqual(allocated[0].parent, self.output_root)
        self.assertTrue(allocated[0].name.startswith(".rruff.staging-"))
        self.assertFalse(allocated[0].exists())
        self.assertEqual(artifact_snapshot(self.output_root), before)
        self.assert_no_staging(self.output_root)

    def test_total_seconds_includes_staging_cleanup(self):
        real_rmtree = shutil.rmtree

        def delayed_rmtree(path):
            if Path(path).name.startswith(".rruff.staging-"):
                time.sleep(0.05)
            return real_rmtree(path)

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff.shutil.rmtree",
            side_effect=delayed_rmtree,
        ):
            review = rruff_module._stage_rruff_for_review(
                self.raw_root,
                self.output_root,
            )

        self.assertGreaterEqual(review.metrics.total_seconds, 0.05)
        self.assertFalse(self.output_root.exists())

    def test_total_and_component_timings_include_their_named_work(self):
        real_validate_receipt = rruff_module._validate_rruff_receipt
        real_gate = rruff_module._rruff_feasibility_failures

        def delayed_inspect(*args, **kwargs):
            time.sleep(0.03)
            return self.bundle

        def delayed_validate_receipt(*args, **kwargs):
            time.sleep(0.03)
            return real_validate_receipt(*args, **kwargs)

        def delayed_gate(*args, **kwargs):
            time.sleep(0.03)
            return real_gate(*args, **kwargs)

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            side_effect=delayed_inspect,
        ), patch(
            "rpe.io.rruff._validate_rruff_receipt",
            side_effect=delayed_validate_receipt,
        ), patch(
            "rpe.io.rruff._rruff_feasibility_failures",
            side_effect=delayed_gate,
        ):
            review = rruff_module._stage_rruff_for_review(
                self.raw_root,
                self.output_root,
            )

        self.assertGreaterEqual(review.metrics.inspection_seconds, 0.03)
        self.assertGreaterEqual(review.metrics.validation_seconds, 0.03)
        component_seconds = (
            review.metrics.inspection_seconds
            + review.metrics.staged_build_seconds
            + review.metrics.validation_seconds
            + review.metrics.lazy_open_seconds_raw
            + review.metrics.lazy_open_seconds_processed
            + review.metrics.representative_record_read_seconds_raw
            + review.metrics.representative_record_read_seconds_processed
        )
        self.assertGreaterEqual(
            review.metrics.total_seconds,
            component_seconds + 0.03,
        )
        self.assertFalse(self.output_root.exists())

    def test_source_and_structural_failures_propagate_after_cleanup(self):
        source_error = RruffValidationError(
            "source.synthetic",
            "injected source failure",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            side_effect=source_error,
        ):
            with self.assertRaises(RruffValidationError) as caught:
                rruff_module._stage_rruff_for_review(
                    self.raw_root,
                    self.output_root,
                )
        self.assertIs(caught.exception, source_error)
        self.assertFalse(self.output_root.exists())

        build_error = RruffValidationError(
            "staged.synthetic",
            "injected staged failure",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff._build_rruff_staged",
            side_effect=build_error,
        ):
            with self.assertRaises(RruffValidationError) as caught:
                rruff_module._stage_rruff_for_review(
                    self.raw_root,
                    self.output_root,
                )
        self.assertIs(caught.exception, build_error)
        self.assertFalse(self.output_root.exists())

    def test_staging_allocation_and_post_build_probe_failures_clean_up(self):
        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff.tempfile.mkdtemp",
            side_effect=OSError("injected allocation failure"),
        ):
            with self.assertRaisesRegex(
                OSError,
                "injected allocation failure",
            ):
                rruff_module._stage_rruff_for_review(
                    self.raw_root,
                    self.output_root,
                )
        self.assertFalse(self.output_root.exists())

        self.output_root.mkdir()
        sentinel = self.output_root / "unowned-sentinel.txt"
        sentinel.write_text("keep me\n", encoding="utf-8")
        for artifact_name in (
            "rruff_raman_raw",
            "rruff_raman_processed",
        ):
            artifact = self.output_root / artifact_name
            artifact.mkdir()
            (artifact / "original.txt").write_text(
                "original\n",
                encoding="utf-8",
            )
        for artifact_name in (
            "rruff_raman_pairs.jsonl",
            "rruff_raman_conversion.json",
        ):
            (self.output_root / artifact_name).write_text(
                "original\n",
                encoding="utf-8",
            )
        before = artifact_snapshot(self.output_root)
        probe_error = RruffValidationError(
            "probe.synthetic",
            "injected probe failure",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff._measure_rruff_dataset_access",
            side_effect=probe_error,
        ):
            with self.assertRaises(RruffValidationError) as caught:
                rruff_module._stage_rruff_for_review(
                    self.raw_root,
                    self.output_root,
                )
        self.assertIs(caught.exception, probe_error)
        self.assertEqual(artifact_snapshot(self.output_root), before)
        self.assert_no_staging(self.output_root)

    def test_non_directory_output_root_fails_without_source_access(self):
        self.output_root.write_text("not a directory\n", encoding="utf-8")

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle"
        ) as inspect:
            with self.assertRaises(RruffValidationError) as caught:
                rruff_module._stage_rruff_for_review(
                    self.raw_root,
                    self.output_root,
                )

        self.assertEqual(caught.exception.path, "output_root")
        inspect.assert_not_called()
        self.assertEqual(
            self.output_root.read_text(encoding="utf-8"),
            "not a directory\n",
        )


class RruffTransactionTest(unittest.TestCase):
    ARTIFACT_NAMES = (
        "rruff_raman_raw",
        "rruff_raman_processed",
        "rruff_raman_pairs.jsonl",
        "rruff_raman_conversion.json",
    )

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.contract = synthetic_pairing_contract(self.source)
        self.bundle = _inspect_rruff_bundle(self.raw_root, self.contract)
        required = (
            "_PublicationArtifact",
            "_publish_rruff_artifacts",
            "_build_rruff_unified_with_metrics",
            "build_rruff_unified",
        )
        missing = [
            name for name in required if not hasattr(rruff_module, name)
        ]
        self.assertEqual(
            missing,
            [],
            f"Task 9 transaction interfaces are missing: {missing}",
        )

    def output_root(self, name: str = "output") -> Path:
        return self.temporary_root / name

    def final_paths(self, output_root: Path) -> tuple[Path, ...]:
        return tuple(output_root / name for name in self.ARTIFACT_NAMES)

    def assert_no_staging(self, output_root: Path) -> None:
        if not output_root.exists():
            return
        self.assertFalse(
            any(
                path.name.startswith(".rruff.staging-")
                for path in output_root.iterdir()
            )
        )

    def seed_originals(self, output_root: Path) -> dict[str, object]:
        output_root.mkdir(parents=True)
        raw, processed, pairs, receipt = self.final_paths(output_root)
        for path in (raw, processed):
            path.mkdir()
            (path / "original.txt").write_text(
                f"original {path.name}\n",
                encoding="utf-8",
            )
        pairs.write_text("original pairs\n", encoding="utf-8")
        receipt.write_text("original receipt\n", encoding="utf-8")
        return {
            path.name: artifact_snapshot(path)
            for path in (raw, processed, pairs, receipt)
        }

    def assert_originals(
        self,
        output_root: Path,
        expected: dict[str, object],
    ) -> None:
        for path in self.final_paths(output_root):
            self.assertTrue(path.exists(), path)
            self.assertEqual(artifact_snapshot(path), expected[path.name])

    def build(
        self,
        output_root: Path,
        *,
        overwrite: bool = False,
    ):
        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff._rruff_feasibility_failures",
            return_value=(),
        ):
            return rruff_module.build_rruff_unified(
                self.raw_root,
                output_root,
                overwrite=overwrite,
            )

    def test_first_publication_is_valid_ordered_and_preserves_sibling(self):
        output_root = self.output_root()
        output_root.mkdir()
        sentinel = output_root / "unowned-sentinel.txt"
        sentinel.write_text("keep me\n", encoding="utf-8")
        source_hashes = dict(self.source.archive_sha256)
        publication_order = []
        real_replace = os.replace

        def recording_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                destination_path.parent == output_root
                and destination_path.name in self.ARTIFACT_NAMES
                and source_path.parent.name.startswith(
                    ".rruff.staging-"
                )
            ):
                publication_order.append(destination_path.name)
            return real_replace(source, destination)

        with patch(
            "rpe.io.rruff.os.replace",
            side_effect=recording_replace,
        ):
            summary = self.build(output_root)

        raw, processed, pairs, receipt = self.final_paths(output_root)
        self.assertEqual(publication_order, list(self.ARTIFACT_NAMES))
        self.assertEqual(validate_dataset(raw).record_count, 11)
        self.assertEqual(validate_dataset(processed).record_count, 8)
        with UnifiedDataset.open(raw) as raw_dataset:
            raw_ids = set(raw_dataset.record_ids)
        with UnifiedDataset.open(processed) as processed_dataset:
            processed_ids = set(processed_dataset.record_ids)
        pair_validation = _validate_rruff_pair_index(
            pairs,
            self.bundle.inspection,
            raw_record_ids=raw_ids,
            processed_record_ids=processed_ids,
        )
        self.assertEqual(pair_validation["rows"], 13)
        rruff_module._validate_rruff_receipt(receipt, self.bundle)
        self.assertEqual(summary.raw.dataset_id, "rruff_raman_raw")
        self.assertEqual(
            summary.pair_index_path,
            output_root / "rruff_raman_pairs.jsonl",
        )
        self.assertEqual(summary.pair_index_sha256, file_sha256(pairs))
        self.assertEqual(
            summary.receipt_path,
            output_root / "rruff_raman_conversion.json",
        )
        self.assertEqual(summary.receipt_sha256, file_sha256(receipt))
        self.assertEqual(
            sentinel.read_text(encoding="utf-8"),
            "keep me\n",
        )
        self.assert_no_staging(output_root)
        self.assertEqual(
            {
                path.name: file_sha256(path)
                for path in sorted(self.raw_root.glob("*.zip"))
            },
            source_hashes,
        )

    def test_overwrite_is_12_file_deterministic(self):
        output_root = self.output_root()
        first = self.build(output_root)
        before = {
            path.relative_to(output_root).as_posix(): file_sha256(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file()
        }

        second = self.build(output_root, overwrite=True)
        after = {
            path.relative_to(output_root).as_posix(): file_sha256(path)
            for path in sorted(output_root.rglob("*"))
            if path.is_file()
        }

        self.assertEqual(len(before), 12)
        self.assertEqual(after, before)
        self.assertEqual(second.pair_index_sha256, first.pair_index_sha256)
        self.assertEqual(second.receipt_sha256, first.receipt_sha256)
        self.assert_no_staging(output_root)

    def test_existing_output_without_overwrite_fails_before_source_access(self):
        output_root = self.output_root()
        originals = self.seed_originals(output_root)
        missing_raw_root = self.temporary_root / "missing-raw"

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
        ) as inspect:
            with self.assertRaises(RruffValidationError) as caught:
                rruff_module.build_rruff_unified(
                    missing_raw_root,
                    output_root,
                )

        self.assertEqual(
            caught.exception.path,
            "output_root/rruff_raman_raw",
        )
        inspect.assert_not_called()
        self.assert_originals(output_root, originals)
        self.assert_no_staging(output_root)

    def test_dangling_final_symlink_is_existing_output_before_source_access(self):
        output_root = self.output_root()
        output_root.mkdir()
        final = output_root / "rruff_raman_raw"
        final.symlink_to(output_root / "missing-target")
        missing_raw_root = self.temporary_root / "missing-raw"

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
        ) as inspect:
            with self.assertRaises(RruffValidationError) as caught:
                rruff_module.build_rruff_unified(
                    missing_raw_root,
                    output_root,
                )

        self.assertEqual(
            caught.exception.path,
            "output_root/rruff_raman_raw",
        )
        inspect.assert_not_called()
        self.assertTrue(final.is_symlink())
        self.assertEqual(
            final.readlink(),
            output_root / "missing-target",
        )
        self.assert_no_staging(output_root)

    def test_all_prepublication_failures_clean_up_and_preserve_finals(self):
        scenarios = (
            (
                "allocation",
                "rpe.io.rruff.tempfile.mkdtemp",
                OSError("injected allocation failure"),
            ),
            (
                "build",
                "rpe.io.rruff._build_rruff_staged",
                RruffValidationError(
                    "staged.synthetic",
                    "injected build failure",
                    "SOURCE_MEMBER_COUNT_MISMATCH",
                ),
            ),
            (
                "validation",
                "rpe.io.rruff._measure_rruff_dataset_access",
                DatasetValidationError(
                    "probe.synthetic",
                    "injected validation failure",
                ),
            ),
        )
        for name, target, failure in scenarios:
            with self.subTest(name=name):
                output_root = self.output_root(name)
                originals = self.seed_originals(output_root)
                with patch(
                    "rpe.io.rruff._inspect_rruff_bundle",
                    return_value=self.bundle,
                ), patch(
                    "rpe.io.rruff._rruff_feasibility_failures",
                    return_value=(),
                ), patch(target, side_effect=failure):
                    with self.assertRaises(type(failure)):
                        rruff_module.build_rruff_unified(
                            self.raw_root,
                            output_root,
                            overwrite=True,
                        )
                self.assert_originals(output_root, originals)
                self.assert_no_staging(output_root)

    def test_gate_failure_blocks_publication_and_returns_all_names(self):
        output_root = self.output_root()
        output_root.mkdir()
        sentinel = output_root / "sentinel.txt"
        sentinel.write_text("keep me\n", encoding="utf-8")
        before = artifact_snapshot(output_root)

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff._rruff_feasibility_failures",
            return_value=(
                "peak_rss_bytes",
                "combined_output_bytes",
            ),
        ), patch(
            "rpe.io.rruff._publish_rruff_artifacts",
        ) as publish:
            with self.assertRaises(RruffValidationError) as caught:
                rruff_module.build_rruff_unified(
                    self.raw_root,
                    output_root,
                )

        self.assertEqual(caught.exception.path, "feasibility")
        self.assertIn("peak_rss_bytes", str(caught.exception))
        self.assertIn("combined_output_bytes", str(caught.exception))
        publish.assert_not_called()
        self.assertEqual(artifact_snapshot(output_root), before)
        self.assert_no_staging(output_root)

    def test_publish_failure_each_artifact_restores_every_original(self):
        for target_name in self.ARTIFACT_NAMES:
            with self.subTest(target_name=target_name):
                output_root = self.output_root(f"publish-{target_name}")
                originals = self.seed_originals(output_root)
                sentinel = output_root / "sentinel.txt"
                sentinel.write_text("keep me\n", encoding="utf-8")
                target = output_root / target_name
                real_replace = os.replace

                def injected_replace(source, destination):
                    source_path = Path(source)
                    destination_path = Path(destination)
                    if (
                        destination_path == target
                        and source_path.parent.name.startswith(
                            ".rruff.staging-"
                        )
                        and not source_path.name.startswith("backup-")
                    ):
                        raise OSError(
                            f"injected publish failure for {target_name}"
                        )
                    return real_replace(source, destination)

                with patch(
                    "rpe.io.rruff.os.replace",
                    side_effect=injected_replace,
                ):
                    with self.assertRaisesRegex(
                        OSError,
                        "injected publish failure",
                    ):
                        self.build(output_root, overwrite=True)

                self.assert_originals(output_root, originals)
                self.assertEqual(
                    sentinel.read_text(encoding="utf-8"),
                    "keep me\n",
                )
                self.assert_no_staging(output_root)

    def test_first_publish_failure_removes_only_new_finals(self):
        for target_name in self.ARTIFACT_NAMES:
            with self.subTest(target_name=target_name):
                output_root = self.output_root(f"first-{target_name}")
                output_root.mkdir()
                sentinel = output_root / "sentinel.txt"
                sentinel.write_text("keep me\n", encoding="utf-8")
                target = output_root / target_name
                real_replace = os.replace

                def injected_replace(source, destination):
                    source_path = Path(source)
                    destination_path = Path(destination)
                    if (
                        destination_path == target
                        and source_path.parent.name.startswith(
                            ".rruff.staging-"
                        )
                    ):
                        raise OSError(
                            f"injected first failure for {target_name}"
                        )
                    return real_replace(source, destination)

                with patch(
                    "rpe.io.rruff.os.replace",
                    side_effect=injected_replace,
                ):
                    with self.assertRaisesRegex(
                        OSError,
                        "injected first failure",
                    ):
                        self.build(output_root)

                for final in self.final_paths(output_root):
                    self.assertFalse(final.exists(), final)
                self.assertEqual(
                    sentinel.read_text(encoding="utf-8"),
                    "keep me\n",
                )
                self.assert_no_staging(output_root)

    def test_backup_failure_restores_prior_backups(self):
        for target_name in self.ARTIFACT_NAMES:
            with self.subTest(target_name=target_name):
                output_root = self.output_root(f"backup-{target_name}")
                originals = self.seed_originals(output_root)
                target = output_root / target_name
                real_replace = os.replace

                def injected_replace(source, destination):
                    source_path = Path(source)
                    destination_path = Path(destination)
                    if (
                        source_path == target
                        and destination_path.parent.name.startswith(
                            ".rruff.staging-"
                        )
                        and destination_path.name.startswith("backup-")
                    ):
                        raise OSError(
                            f"injected backup failure for {target_name}"
                        )
                    return real_replace(source, destination)

                with patch(
                    "rpe.io.rruff.os.replace",
                    side_effect=injected_replace,
                ):
                    with self.assertRaisesRegex(
                        OSError,
                        "injected backup failure",
                    ):
                        self.build(output_root, overwrite=True)

                self.assert_originals(output_root, originals)
                self.assert_no_staging(output_root)

    def test_success_metrics_total_seconds_include_publication_cleanup(self):
        output_root = self.output_root()
        real_publish = rruff_module._publish_rruff_artifacts

        def delayed_publish(*args, **kwargs):
            time.sleep(0.05)
            return real_publish(*args, **kwargs)

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff._rruff_feasibility_failures",
            return_value=(),
        ), patch(
            "rpe.io.rruff._publish_rruff_artifacts",
            side_effect=delayed_publish,
        ):
            _, metrics = rruff_module._build_rruff_unified_with_metrics(
                self.raw_root,
                output_root,
            )

        component_seconds = (
            metrics.inspection_seconds
            + metrics.staged_build_seconds
            + metrics.validation_seconds
            + metrics.lazy_open_seconds_raw
            + metrics.lazy_open_seconds_processed
            + metrics.representative_record_read_seconds_raw
            + metrics.representative_record_read_seconds_processed
        )
        self.assertGreaterEqual(
            metrics.total_seconds,
            component_seconds + 0.05,
        )
        self.assert_no_staging(output_root)

    def test_keyboard_interrupt_rolls_back_and_propagates(self):
        output_root = self.output_root()
        originals = self.seed_originals(output_root)
        processed = output_root / "rruff_raman_processed"
        real_replace = os.replace

        def injected_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                destination_path == processed
                and source_path.parent.name.startswith(
                    ".rruff.staging-"
                )
                and not source_path.name.startswith("backup-")
            ):
                raise KeyboardInterrupt
            return real_replace(source, destination)

        with patch(
            "rpe.io.rruff.os.replace",
            side_effect=injected_replace,
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.build(output_root, overwrite=True)

        self.assert_originals(output_root, originals)
        self.assert_no_staging(output_root)

    def test_restore_failure_preserves_backup_and_manual_recovery_path(self):
        output_root = self.output_root()
        originals = self.seed_originals(output_root)
        raw, processed, _, _ = self.final_paths(output_root)
        real_replace = os.replace
        publish_failed = False

        def injected_replace(source, destination):
            nonlocal publish_failed
            source_path = Path(source)
            destination_path = Path(destination)
            if (
                destination_path == processed
                and source_path.parent.name.startswith(
                    ".rruff.staging-"
                )
                and not source_path.name.startswith("backup-")
            ):
                publish_failed = True
                raise OSError("injected publish failure")
            if (
                publish_failed
                and destination_path == raw
                and source_path.name == "backup-raw"
            ):
                raise OSError("injected restore failure")
            return real_replace(source, destination)

        with patch(
            "rpe.io.rruff.os.replace",
            side_effect=injected_replace,
        ):
            with self.assertRaises(RruffValidationError) as caught:
                self.build(output_root, overwrite=True)

        self.assertEqual(caught.exception.path, "publication.restore")
        staging = [
            path
            for path in output_root.iterdir()
            if path.name.startswith(".rruff.staging-")
        ]
        self.assertEqual(len(staging), 1)
        self.assertIn(str(staging[0]), str(caught.exception))
        backup = staging[0] / "backup-raw"
        self.assertTrue(backup.is_dir())
        self.assertEqual(
            artifact_snapshot(backup),
            originals["rruff_raman_raw"],
        )
        self.assertFalse(raw.exists())
        for path in self.final_paths(output_root)[1:]:
            self.assertEqual(
                artifact_snapshot(path),
                originals[path.name],
            )

    def test_publication_contract_requires_exactly_four_artifacts(self):
        staging = self.temporary_root / "manual-staging"
        output_root = self.output_root("manual-output")
        staging.mkdir()
        output_root.mkdir()
        staged = staging / "artifact"
        staged.write_text("new\n", encoding="utf-8")
        artifact = rruff_module._PublicationArtifact(
            staged=staged,
            final=output_root / "artifact",
            backup_name="backup-artifact",
        )

        with self.assertRaises(FrozenInstanceError):
            artifact.backup_name = "changed"
        with self.assertRaises(RruffValidationError) as caught:
            rruff_module._publish_rruff_artifacts(
                staging,
                (artifact,),
                overwrite=False,
            )
        self.assertEqual(caught.exception.path, "publication.artifacts")
        self.assertTrue(staged.is_file())

    def test_publication_contract_rejects_malformed_artifacts_before_moves(self):
        staging = self.temporary_root / "contract-staging"
        output_root = self.output_root("contract-output")
        staging.mkdir()
        output_root.mkdir()
        artifacts = []
        for index, name in enumerate(self.ARTIFACT_NAMES):
            staged = staging / name
            if "." in name:
                staged.write_text(f"new-{index}\n", encoding="utf-8")
            else:
                staged.mkdir()
                (staged / "value.txt").write_text(
                    f"new-{index}\n",
                    encoding="utf-8",
                )
            artifacts.append(
                rruff_module._PublicationArtifact(
                    staged=staged,
                    final=output_root / name,
                    backup_name=f"backup-{index}",
                )
            )

        cases = (
            (
                "duplicate final",
                (
                    artifacts[0],
                    replace(artifacts[1], final=artifacts[0].final),
                    artifacts[2],
                    artifacts[3],
                ),
                "publication.artifacts.final",
            ),
            (
                "duplicate backup",
                (
                    artifacts[0],
                    replace(
                        artifacts[1],
                        backup_name=artifacts[0].backup_name,
                    ),
                    artifacts[2],
                    artifacts[3],
                ),
                "publication.artifacts.backup_name",
            ),
            (
                "nonportable backup",
                (
                    replace(
                        artifacts[0],
                        backup_name="../backup",
                    ),
                    artifacts[1],
                    artifacts[2],
                    artifacts[3],
                ),
                "publication.artifacts.backup_name",
            ),
            (
                "staged outside parent",
                (
                    replace(
                        artifacts[0],
                        staged=self.temporary_root / "outside",
                    ),
                    artifacts[1],
                    artifacts[2],
                    artifacts[3],
                ),
                "publication.artifacts.staged",
            ),
            (
                "final inside staging",
                (
                    replace(
                        artifacts[0],
                        final=staging / "inside",
                    ),
                    artifacts[1],
                    artifacts[2],
                    artifacts[3],
                ),
                "publication.artifacts.final",
            ),
            (
                "final nested inside staging",
                (
                    replace(
                        artifacts[0],
                        final=staging / "nested" / "inside",
                    ),
                    artifacts[1],
                    artifacts[2],
                    artifacts[3],
                ),
                "publication.artifacts.final",
            ),
        )
        for name, malformed, expected_path in cases:
            with self.subTest(name=name):
                with self.assertRaises(RruffValidationError) as caught:
                    rruff_module._publish_rruff_artifacts(
                        staging,
                        malformed,
                        overwrite=False,
                    )
                self.assertEqual(caught.exception.path, expected_path)
                self.assertTrue(all(path.staged.exists() for path in artifacts))
                self.assertFalse(any(output_root.iterdir()))


class RruffCliTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temporary_root = Path(self.temporary_directory.name)
        self.raw_root = self.temporary_root / "raw"
        self.output_root = self.temporary_root / "output"
        self.source = create_synthetic_rruff_pairing_source(self.raw_root)
        self.contract = synthetic_pairing_contract(self.source)
        self.bundle = _inspect_rruff_bundle(self.raw_root, self.contract)
        missing = []
        if not RRUFF_BUILDER_PATH.is_file():
            missing.append("tools/build_rruff_unified.py")
        for name in (
            "_build_rruff_unified_with_metrics",
            "build_rruff_unified",
        ):
            if not hasattr(rruff_module, name):
                missing.append(name)
        self.assertEqual(
            missing,
            [],
            f"Task 9 CLI interfaces are missing: {missing}",
        )

    def run_main(self, *extra_arguments: str):
        builder = load_rruff_builder_module()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff._rruff_feasibility_failures",
            return_value=(),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            result = builder.main(
                [
                    "--raw-root",
                    str(self.raw_root),
                    "--output-root",
                    str(self.output_root),
                    *extra_arguments,
                ]
            )
        return result, stdout.getvalue(), stderr.getvalue()

    def test_cli_success_emits_one_exact_canonical_json_line_offline(self):
        builder = load_rruff_builder_module()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            return_value=self.bundle,
        ), patch(
            "rpe.io.rruff._rruff_feasibility_failures",
            return_value=(),
        ), patch.object(
            socket.socket,
            "connect",
            side_effect=AssertionError("network connection attempted"),
        ), redirect_stdout(stdout), redirect_stderr(stderr):
            result = builder.main(
                [
                    "--raw-root",
                    str(self.raw_root),
                    "--output-root",
                    str(self.output_root),
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        document = json.loads(stdout.getvalue())
        self.assertEqual(
            stdout.getvalue(),
            _canonical_json_bytes(document).decode("utf-8"),
        )
        self.assertEqual(
            set(document),
            {"status", "raw", "processed", "pairing", "receipt", "metrics"},
        )
        self.assertEqual(document["status"], "written")
        for name, records, eligible, groups in (
            ("raw", 11, 11, 10),
            ("processed", 8, 0, 8),
        ):
            self.assertEqual(
                set(document[name]),
                {
                    "dataset_id",
                    "record_count",
                    "eligible_records",
                    "axis_group_count",
                    "output_bytes",
                    "files",
                },
            )
            self.assertEqual(document[name]["record_count"], records)
            self.assertEqual(document[name]["eligible_records"], eligible)
            self.assertEqual(document[name]["axis_group_count"], groups)
            self.assertEqual(set(document[name]["files"]), set(DATASET_FILES))
        self.assertEqual(
            set(document["pairing"]),
            {
                "path",
                "rows",
                "bytes",
                "sha256",
                "status_counts",
                "axis_relation_counts",
            },
        )
        self.assertEqual(document["pairing"]["rows"], 13)
        self.assertEqual(
            set(document["receipt"]),
            {"path", "bytes", "sha256", "rejected_records"},
        )
        self.assertEqual(document["receipt"]["rejected_records"], 2)
        self.assertEqual(
            set(document["metrics"]),
            set(RUNTIME_METRIC_FIELDS),
        )
        self.assertEqual(
            document["pairing"]["path"],
            (self.output_root / "rruff_raman_pairs.jsonl").as_posix(),
        )
        self.assertEqual(
            document["receipt"]["path"],
            (
                self.output_root / "rruff_raman_conversion.json"
            ).as_posix(),
        )

    def test_cli_overwrite_and_expected_failure_protocol(self):
        first_result, _, first_stderr = self.run_main()
        self.assertEqual(first_result, 0, first_stderr)

        failed_result, failed_stdout, failed_stderr = self.run_main()
        self.assertEqual(failed_result, 1)
        self.assertEqual(failed_stdout, "")
        self.assertEqual(len(failed_stderr.splitlines()), 1)
        error = json.loads(failed_stderr)
        self.assertEqual(
            set(error),
            {"status", "error_type", "error"},
        )
        self.assertEqual(error["status"], "failed")
        self.assertEqual(error["error_type"], "RruffValidationError")
        self.assertNotIn("Traceback", failed_stderr)
        self.assertEqual(
            failed_stderr,
            _canonical_json_bytes(error).decode("utf-8"),
        )

        overwrite_result, _, overwrite_stderr = self.run_main("--overwrite")
        self.assertEqual(overwrite_result, 0, overwrite_stderr)

    def test_cli_catches_only_documented_expected_failures(self):
        builder = load_rruff_builder_module()
        expected = (
            RruffValidationError(
                "synthetic",
                "failure",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            ),
            DatasetValidationError("synthetic", "failure"),
            SchemaValidationError("synthetic", "failure"),
            OSError("failure"),
            ValueError("failure"),
            json.JSONDecodeError("failure", "x", 0),
        )
        arguments = [
            "--raw-root",
            str(self.raw_root),
            "--output-root",
            str(self.output_root),
        ]
        for failure in expected:
            with self.subTest(error_type=type(failure).__name__):
                stdout = io.StringIO()
                stderr = io.StringIO()
                with patch.object(
                    builder,
                    "_build_rruff_unified_with_metrics",
                    side_effect=failure,
                ), redirect_stdout(stdout), redirect_stderr(stderr):
                    result = builder.main(arguments)
                self.assertEqual(result, 1)
                self.assertEqual(stdout.getvalue(), "")
                document = json.loads(stderr.getvalue())
                self.assertEqual(
                    document["error_type"],
                    type(failure).__name__,
                )
                self.assertNotIn("Traceback", stderr.getvalue())

        for failure in (
            KeyboardInterrupt(),
            SystemExit(7),
            TypeError("undocumented failure"),
        ):
            with self.subTest(error_type=type(failure).__name__):
                with patch.object(
                    builder,
                    "_build_rruff_unified_with_metrics",
                    side_effect=failure,
                ):
                    with self.assertRaises(type(failure)):
                        builder.main(arguments)

    def test_cli_subprocess_converts_synthetic_source_with_offline_env(self):
        script = r"""
import os
import sys
from pathlib import Path
assert not any("proxy" in key.lower() for key in os.environ)
assert os.environ["HF_HUB_OFFLINE"] == "1"
assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
sys.path.insert(0, str(Path.cwd() / "tests"))
from rruff_helpers import create_synthetic_rruff_pairing_source, synthetic_pairing_contract
import rpe.io.rruff as rruff
from tools.build_rruff_unified import main
raw_root = Path(sys.argv[1])
source = create_synthetic_rruff_pairing_source(raw_root)
contract = synthetic_pairing_contract(source)
real_inspect = rruff._inspect_rruff_bundle
rruff._inspect_rruff_bundle = lambda root, ignored: real_inspect(
    Path(root), contract
)
rruff._rruff_feasibility_failures = lambda summary, metrics, comparisons: ()
raise SystemExit(main([
    "--raw-root", str(raw_root),
    "--output-root", sys.argv[2],
]))
"""
        subprocess_raw_root = self.temporary_root / "subprocess-raw"
        subprocess_output_root = self.temporary_root / "subprocess-output"
        environment = {
            key: value
            for key, value in os.environ.items()
            if "proxy" not in key.lower()
        }
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(subprocess_raw_root),
                str(subprocess_output_root),
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        self.assertEqual(len(result.stdout.splitlines()), 1)
        document = json.loads(result.stdout)
        self.assertEqual(document["status"], "written")
        self.assertEqual(
            validate_dataset(
                subprocess_output_root / "rruff_raman_raw"
            ).record_count,
            11,
        )
        self.assertEqual(
            validate_dataset(
                subprocess_output_root / "rruff_raman_processed"
            ).record_count,
            8,
        )
        self.assertFalse(
            any(
                path.name.startswith(".rruff.staging-")
                for path in subprocess_output_root.iterdir()
            )
        )


@unittest.skipUnless(
    all(
        (PRODUCTION_RRUFF_RAW_ROOT / archive).is_file()
        for archive in ARCHIVE_NAMES
    ),
    "retained RRUFF source archives are unavailable",
)
class RruffProductionInspectionTest(unittest.TestCase):
    def test_public_api_inspects_complete_retained_source(self):
        captured = []
        real_inspect_bundle = rruff_module._inspect_rruff_bundle

        def capturing_inspect_bundle(raw_root, contract):
            bundle = real_inspect_bundle(raw_root, contract)
            captured.append(bundle)
            return bundle

        with patch(
            "rpe.io.rruff._inspect_rruff_bundle",
            side_effect=capturing_inspect_bundle,
        ):
            inspection = inspect_rruff(PRODUCTION_RRUFF_RAW_ROOT)

        self.assertEqual(len(captured), 1)
        bundle = captured[0]
        statistics = bundle.statistics
        self.assertIs(inspection, bundle.inspection)
        self.assertEqual(
            dict(inspection.archive_sha256),
            {
                "LR-Raman.zip": (
                    "45f95e7003e00c196d5f6d0c326483f9d7a19476b499b4e1afd4048b81d802d7"
                ),
                "excellent_oriented.zip": (
                    "51b026d8087fd010d6d87b40cb626e486d96e971512a7d62287457672c0e762b"
                ),
                "excellent_unoriented.zip": (
                    "c244b65856b6c46bcbef125a92500da8d1cef22cfdc6b3d30dcf75077cfb2a73"
                ),
                "fair_oriented.zip": (
                    "7126e24d174e051d584f6cd054f235c09b8c4ec0a498ecb6f56949ecd140dcd4"
                ),
                "fair_unoriented.zip": (
                    "23a02af37c792b47310f1bc20906b88ba947c885fe882006de8054925eead787"
                ),
                "poor_unoriented.zip": (
                    "46f92fa5b229f79a04375b654eceec62b5695a472ea55596e1abe5964e9cc82c"
                ),
                "unrated_oriented.zip": (
                    "f931603eaa10ae8b94f8b2591d2c8b0fec30e8e92fe17ef64d8c145dda83f7f5"
                ),
                "unrated_unoriented.zip": (
                    "ecd9dbdc0cd8c559c1aaa077d8bb136cdbe2e8a208d34fcd619bf6405dc2e64b"
                ),
            },
        )
        expected_statistics = {
            "source_members": 38_056,
            "source_raw": 20_674,
            "source_processed": 17_382,
            "accepted_records": 38_043,
            "rejected_records": 13,
            "accepted_raw": 20_664,
            "accepted_processed": 17_379,
            "accepted_spectral_points": 77_368_604,
            "rejected_member_parsed_numeric_rows": 36_385,
            "all_parsed_numeric_rows": 77_404_989,
            "mineral_classes": 2_508,
            "rruff_samples": 4_150,
            "raw_mineral_classes": 2_480,
            "processed_mineral_classes": 2_487,
            "raw_rruff_samples": 4_106,
            "processed_rruff_samples": 4_072,
            "utf8_members": 38_053,
            "cp1252_members": 3,
            "comma_numeric_rows": 76_901_660,
            "whitespace_numeric_rows": 503_329,
            "ignored_comment_lines": 148,
            "ignored_preamble_lines": 1_462,
            "ignored_column_header_lines": 6,
            "duplicate_header_records": 728,
            "increasing_axes": 38_043,
            "decreasing_axes": 0,
            "records_with_excitation_nm": 38_035,
            "records_with_excitation_nm_none": 8,
            "records_with_full_comment_acquisition_metadata": 3,
            "union_axis_groups": 14_951,
            "raw_axis_groups": 10_260,
            "processed_axis_groups": 11_009,
            "shared_axis_groups": 6_318,
            "raw_singleton_axis_groups": 8_516,
            "processed_singleton_axis_groups": 8_890,
            "raw_median_records_per_axis_group": 1.0,
            "processed_median_records_per_axis_group": 1.0,
            "raw_p95_records_per_axis_group": 4.0,
            "processed_p95_records_per_axis_group": 4.0,
            "raw_maximum_records_per_axis_group": 214,
            "processed_maximum_records_per_axis_group": 113,
            "axis_float32_max_abs_error_cm1": (
                0.00024218749967985786
            ),
            "intensity_float32_max_abs_error": (
                0.025000000023283064
            ),
        }
        self.assertEqual(
            {
                name: getattr(statistics, name)
                for name in expected_statistics
            },
            expected_statistics,
        )
        self.assertEqual(
            dict(statistics.rejection_code_counts),
            {
                "AXIS_DUPLICATE": 8,
                "AXIS_NONMONOTONIC": 2,
                "NUMERIC_ROW_INVALID": 3,
            },
        )
        self.assertEqual(inspection.source_member_count, 38_056)
        self.assertEqual(len(inspection.accepted_members), 38_043)
        self.assertEqual(len(inspection.rejected_members), 13)
        self.assertEqual(len(inspection.global_class_labels), 2_508)
        self.assertEqual(inspection.raw_axis_group_count, 10_260)
        self.assertEqual(inspection.processed_axis_group_count, 11_009)
        self.assertEqual(
            len({member.rruff_id for member in inspection.accepted_members}),
            4_150,
        )
        self.assertEqual(
            {
                status: sum(
                    pair.pair_status == status
                    for pair in inspection.pairs
                )
                for status in (
                    "paired_unique",
                    "raw_only",
                    "processed_only",
                    "ambiguous",
                    "rejected_only",
                )
            },
            {
                "paired_unique": 15_861,
                "raw_only": 4_793,
                "processed_only": 1_516,
                "ambiguous": 6,
                "rejected_only": 9,
            },
        )
        self.assertEqual(
            {
                relation: sum(
                    pair.axis_relation == relation
                    for pair in inspection.pairs
                )
                for relation in (
                    "exact_equal",
                    "processed_exact_contiguous_subset_of_raw",
                    "raw_exact_contiguous_subset_of_processed",
                    "overlap_requires_alignment",
                    "no_axis_overlap",
                )
            },
            {
                "exact_equal": 9_391,
                "processed_exact_contiguous_subset_of_raw": 6_345,
                "raw_exact_contiguous_subset_of_processed": 28,
                "overlap_requires_alignment": 96,
                "no_axis_overlap": 1,
            },
        )
        self.assertEqual(
            {
                (
                    member.archive,
                    member.source_member,
                    member.kind,
                    member.rejection_code,
                )
                for member in inspection.rejected_members
            },
            PRODUCTION_REJECTIONS,
        )


if __name__ == "__main__":
    unittest.main()
