from __future__ import annotations

import hashlib
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from rpe.io.schema import axis_id


_MEMBER_PATTERN = re.compile(
    r"^(?P<measurement_key>.+)__Raman_Data_"
    r"(?P<file_kind>Raw|RAW|Processed)__"
    r"(?P<member_id>[0-9a-f]+)\.txt$"
)
_IDENTITY_HEADERS = (
    "NAMES",
    "RRUFFID",
    "FILETYPE",
    "RAMAN WAVELENGTH",
    "URL",
)
_COLUMN_HEADERS = {
    ("x", "y"),
    ("wave", "intensity"),
}
_ARCHIVE_URL_PREFIX = "https://rruff.info/zipped_data_files/raman/"


class RruffValidationError(ValueError):
    def __init__(self, path: str, reason: str, code: str) -> None:
        self.path = path
        self.reason = reason
        self.code = code
        super().__init__(f"{path}: [{code}] {reason}")


@dataclass(frozen=True)
class RruffHeaderEntry:
    line: int
    key: str
    value: str


@dataclass(frozen=True)
class RruffIgnoredLine:
    line: int
    category: str
    text: str


@dataclass(frozen=True)
class RruffRejectedMember:
    archive: str
    source_member: str
    member_id: str
    member_sha256: str
    member_bytes: int
    measurement_key: str
    kind: str
    text_encoding: str | None
    header_entries: tuple[RruffHeaderEntry, ...]
    ignored_lines: tuple[RruffIgnoredLine, ...]
    mineral_name: str | None
    rruff_id: str | None
    rejection_code: str
    reason: str


@dataclass(frozen=True)
class _ParsedRruffMember:
    archive: str
    source_member: str
    member_id: str
    member_sha256: str
    member_bytes: int
    measurement_key: str
    kind: str
    text_encoding: str
    header_entries: tuple[RruffHeaderEntry, ...]
    ignored_lines: tuple[RruffIgnoredLine, ...]
    mineral_name: str
    rruff_id: str
    pin_id: str | None
    orientation: str | None
    excitation_nm: float | None
    source_point_count: int
    source_axis: np.ndarray
    source_intensity: np.ndarray
    axis_direction: str
    axis_id: str
    axis_float32_max_abs_error: float
    intensity_float32_max_abs_error: float
    comma_numeric_rows: int
    whitespace_numeric_rows: int


@dataclass(frozen=True)
class _MemberParseStatistics:
    parsed_numeric_rows: int
    comma_numeric_rows: int
    whitespace_numeric_rows: int
    ignored_comment_lines: int
    ignored_preamble_lines: int
    ignored_column_header_lines: int
    duplicate_header: bool


@dataclass(frozen=True)
class _MemberParseOutcome:
    accepted: _ParsedRruffMember | None
    rejected: RruffRejectedMember | None
    statistics: _MemberParseStatistics


@dataclass(frozen=True)
class _ArchiveContract:
    archive: str
    bytes: int
    sha256: str
    member_count: int
    raw_members: int
    processed_members: int


@dataclass(frozen=True)
class _RruffSourceContract:
    archives: Mapping[str, _ArchiveContract]
    source_members: int
    source_raw: int
    source_processed: int
    accepted_records: int
    accepted_raw: int
    accepted_processed: int
    expected_rejections: frozenset[tuple[str, str, str, str]]
    expected_statistics: Mapping[str, int | float] | None
    expected_pair_statistics: Mapping[str, int] | None


@dataclass(frozen=True)
class _SourceAcceptedMember:
    archive: str
    source_member: str
    member_id: str
    member_sha256: str
    member_bytes: int
    measurement_key: str
    kind: str
    text_encoding: str
    header_entries: tuple[RruffHeaderEntry, ...]
    ignored_lines: tuple[RruffIgnoredLine, ...]
    mineral_name: str
    rruff_id: str
    pin_id: str | None
    orientation: str | None
    excitation_nm: float | None
    source_point_count: int
    axis_id: str
    axis_float32_max_abs_error: float
    intensity_float32_max_abs_error: float


@dataclass(frozen=True)
class _RruffSourceStatistics:
    source_members: int
    source_raw: int
    source_processed: int
    accepted_records: int
    rejected_records: int
    accepted_raw: int
    accepted_processed: int
    accepted_spectral_points: int
    rejected_member_parsed_numeric_rows: int
    all_parsed_numeric_rows: int
    mineral_classes: int
    rruff_samples: int
    raw_mineral_classes: int
    processed_mineral_classes: int
    raw_rruff_samples: int
    processed_rruff_samples: int
    utf8_members: int
    cp1252_members: int
    comma_numeric_rows: int
    whitespace_numeric_rows: int
    ignored_comment_lines: int
    ignored_preamble_lines: int
    ignored_column_header_lines: int
    duplicate_header_records: int
    increasing_axes: int
    decreasing_axes: int
    records_with_excitation_nm: int
    records_with_excitation_nm_none: int
    records_with_full_comment_acquisition_metadata: int
    union_axis_groups: int
    raw_axis_groups: int
    processed_axis_groups: int
    shared_axis_groups: int
    raw_singleton_axis_groups: int
    processed_singleton_axis_groups: int
    raw_median_records_per_axis_group: float
    processed_median_records_per_axis_group: float
    raw_p95_records_per_axis_group: float
    processed_p95_records_per_axis_group: float
    raw_maximum_records_per_axis_group: int
    processed_maximum_records_per_axis_group: int
    axis_float32_max_abs_error_cm1: float
    intensity_float32_max_abs_error: float
    rejection_code_counts: Mapping[str, int]


@dataclass(frozen=True)
class _RruffAcquisitionFields:
    instrument: str | None
    excitation_nm: float | None
    integration_time_s: float | None
    n_accumulations: int | None
    grating: str | None
    detector: str | None


@dataclass(frozen=True)
class _RruffSourceInspection:
    raw_root: Path
    archive_sha256: Mapping[str, str]
    accepted_members: tuple[_SourceAcceptedMember, ...]
    rejected_members: tuple[RruffRejectedMember, ...]
    global_class_labels: Mapping[int, str]
    raw_class_labels: Mapping[int, str]
    processed_class_labels: Mapping[int, str]
    raw_rruff_ids: frozenset[str]
    processed_rruff_ids: frozenset[str]
    raw_axis_ids: frozenset[str]
    processed_axis_ids: frozenset[str]
    statistics: _RruffSourceStatistics


_PRODUCTION_ARCHIVES = MappingProxyType(
    {
        "LR-Raman.zip": _ArchiveContract(
            "LR-Raman.zip",
            238_134_984,
            "45f95e7003e00c196d5f6d0c326483f9d7a19476b499b4e1afd4048b81d802d7",
            9_941,
            6_566,
            3_375,
        ),
        "excellent_oriented.zip": _ArchiveContract(
            "excellent_oriented.zip",
            80_871_357,
            "51b026d8087fd010d6d87b40cb626e486d96e971512a7d62287457672c0e762b",
            7_668,
            3_799,
            3_869,
        ),
        "excellent_unoriented.zip": _ArchiveContract(
            "excellent_unoriented.zip",
            240_547_259,
            "c244b65856b6c46bcbef125a92500da8d1cef22cfdc6b3d30dcf75077cfb2a73",
            11_415,
            5_628,
            5_787,
        ),
        "fair_oriented.zip": _ArchiveContract(
            "fair_oriented.zip",
            277_354,
            "7126e24d174e051d584f6cd054f235c09b8c4ec0a498ecb6f56949ecd140dcd4",
            25,
            12,
            13,
        ),
        "fair_unoriented.zip": _ArchiveContract(
            "fair_unoriented.zip",
            61_534_507,
            "23a02af37c792b47310f1bc20906b88ba947c885fe882006de8054925eead787",
            2_951,
            1_466,
            1_485,
        ),
        "poor_unoriented.zip": _ArchiveContract(
            "poor_unoriented.zip",
            34_808_272,
            "46f92fa5b229f79a04375b654eceec62b5695a472ea55596e1abe5964e9cc82c",
            1_639,
            829,
            810,
        ),
        "unrated_oriented.zip": _ArchiveContract(
            "unrated_oriented.zip",
            41_353_106,
            "f931603eaa10ae8b94f8b2591d2c8b0fec30e8e92fe17ef64d8c145dda83f7f5",
            3_778,
            1_927,
            1_851,
        ),
        "unrated_unoriented.zip": _ArchiveContract(
            "unrated_unoriented.zip",
            12_910_186,
            "ecd9dbdc0cd8c559c1aaa077d8bb136cdbe2e8a208d34fcd619bf6405dc2e64b",
            639,
            447,
            192,
        ),
    }
)

PRODUCTION_REJECTIONS = frozenset(
    {
        (
            "LR-Raman.zip",
            "Dacostaite__R250010__Broad_Scan__633__0__unoriented__"
            "Raman_Data_Raw__05da6e0a321e9a50ab6eb2a7053b.txt",
            "raw",
            "AXIS_DUPLICATE",
        ),
        (
            "LR-Raman.zip",
            "Hydromagnesite__R220007__Broad_Scan__532__0__unoriented__"
            "Raman_Data_Raw__46a513be613e8137077c7b08cfad.txt",
            "raw",
            "NUMERIC_ROW_INVALID",
        ),
        (
            "LR-Raman.zip",
            "Kusachiite__R220013__Broad_Scan__532__0__unoriented__"
            "Raman_Data_Raw__eab0ba8dd872e0e100a9b093991c.txt",
            "raw",
            "NUMERIC_ROW_INVALID",
        ),
        (
            "LR-Raman.zip",
            "Heflikite__R250030__Broad_Scan__532__0__unoriented__"
            "Raman_Data_Raw__681f2b18cc0a5af300e1a671f267.txt",
            "raw",
            "AXIS_DUPLICATE",
        ),
        (
            "LR-Raman.zip",
            "Clino-ferro-suenoite__R250076__Broad_Scan__515__0__unoriented__"
            "Raman_Data_Raw__7ddd3c1d5fd1f28fe928079a460d.txt",
            "raw",
            "AXIS_NONMONOTONIC",
        ),
        (
            "excellent_unoriented.zip",
            "Chromatite__R130086__Raman__532______Raman_Data_Processed__"
            "8df90c8a30428ef64e3211251d9a.txt",
            "processed",
            "AXIS_DUPLICATE",
        ),
        (
            "excellent_unoriented.zip",
            "Chromatite__R130086__Raman__532______Raman_Data_RAW__"
            "dc12a6f092ee4f7bbf71445eb0b1.txt",
            "raw",
            "AXIS_DUPLICATE",
        ),
        (
            "excellent_unoriented.zip",
            "Dacostaite__R250010__Raman__633______Raman_Data_Processed__"
            "02d16cd4200aa2e7929d7fa061ae.txt",
            "processed",
            "AXIS_DUPLICATE",
        ),
        (
            "excellent_unoriented.zip",
            "Heflikite__R250030__Raman__532______Raman_Data_Processed__"
            "731e9bd345be1f0b0d637399d62e.txt",
            "processed",
            "AXIS_DUPLICATE",
        ),
        (
            "excellent_unoriented.zip",
            "Heflikite__R250030__Raman__532______Raman_Data_RAW__"
            "366c1d42927874f6161cb9208372.txt",
            "raw",
            "AXIS_DUPLICATE",
        ),
        (
            "excellent_unoriented.zip",
            "Hydromagnesite__R220007__Raman__532______Raman_Data_RAW__"
            "b3d7c6d02ad14e42f551a3a7d28a.txt",
            "raw",
            "NUMERIC_ROW_INVALID",
        ),
        (
            "poor_unoriented.zip",
            "Straczekite__R050325__Raman__514______Raman_Data_RAW__"
            "0e9d250e3934f16dabba885455d4.txt",
            "raw",
            "AXIS_NONMONOTONIC",
        ),
        (
            "unrated_unoriented.zip",
            "Dacostaite__R250010__Raman__633______Raman_Data_RAW__"
            "5a2c9c8aafa65e4144efaebe93fd.txt",
            "raw",
            "AXIS_DUPLICATE",
        ),
    }
)

_PRODUCTION_STATISTICS = MappingProxyType(
    {
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
        "axis_float32_max_abs_error_cm1": 0.00024218749967985786,
        "intensity_float32_max_abs_error": 0.025000000023283064,
    }
)

_PRODUCTION_PAIR_STATISTICS = MappingProxyType(
    {
        "source_measurement_groups": 22_185,
        "source_multiplicity_1_1": 15_865,
        "source_multiplicity_1_0": 4_799,
        "source_multiplicity_0_1": 1_515,
        "source_multiplicity_2_0": 5,
        "source_multiplicity_0_2": 1,
        "paired_unique": 15_861,
        "raw_only": 4_793,
        "processed_only": 1_516,
        "ambiguous": 6,
        "rejected_only": 9,
        "exact_equal": 9_391,
        "processed_exact_contiguous_subset_of_raw": 6_345,
        "raw_exact_contiguous_subset_of_processed": 28,
        "overlap_requires_alignment": 96,
        "no_axis_overlap": 1,
        "pointwise_comparable_pairs": 15_764,
    }
)

PRODUCTION_SOURCE_CONTRACT = _RruffSourceContract(
    archives=_PRODUCTION_ARCHIVES,
    source_members=38_056,
    source_raw=20_674,
    source_processed=17_382,
    accepted_records=38_043,
    accepted_raw=20_664,
    accepted_processed=17_379,
    expected_rejections=PRODUCTION_REJECTIONS,
    expected_statistics=_PRODUCTION_STATISTICS,
    expected_pair_statistics=_PRODUCTION_PAIR_STATISTICS,
)


def _parse_filename(
    source_member: str,
) -> tuple[str, str, str]:
    match = _MEMBER_PATTERN.fullmatch(source_member)
    if match is None:
        raise RruffValidationError(
            "source_member",
            (
                "must match '<measurement-key>__Raman_Data_"
                "<Raw|RAW|Processed>__<lowercase-hex>.txt'"
            ),
            "MEMBER_NAME_INVALID",
        )
    file_kind = match.group("file_kind")
    kind = "raw" if file_kind in {"Raw", "RAW"} else "processed"
    return (
        match.group("measurement_key"),
        kind,
        match.group("member_id"),
    )


def _decode_payload(payload: bytes) -> tuple[str, str]:
    try:
        return payload.decode("utf-8", errors="strict"), "utf-8"
    except UnicodeDecodeError:
        return payload.decode("cp1252", errors="strict"), "cp1252"


def _header_values(
    entries: tuple[RruffHeaderEntry, ...],
) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = {}
    for entry in entries:
        key = entry.key
        value = entry.value.strip()
        if value != "":
            values.setdefault(key, []).append(value)
    return {
        key: tuple(current_values)
        for key, current_values in values.items()
    }


def _identity_fields(
    entries: tuple[RruffHeaderEntry, ...],
) -> tuple[dict[str, str], str | None, str | None]:
    values = _header_values(entries)
    identity: dict[str, str] = {}
    for key in _IDENTITY_HEADERS:
        current_values = values.get(key, ())
        if not current_values:
            raise RruffValidationError(
                f"headers.{key}",
                "required non-empty header is missing",
                "REQUIRED_HEADER_MISSING",
            )
        if len(set(current_values)) != 1:
            raise RruffValidationError(
                f"headers.{key}",
                "repeated non-empty identity values conflict",
                "IDENTITY_HEADER_CONFLICT",
            )
        identity[key] = current_values[-1]
    pin_values = values.get("PIN_ID", ())
    orientation_values = values.get("ORIENTATION", ())
    return (
        identity,
        pin_values[-1] if pin_values else None,
        orientation_values[-1] if orientation_values else None,
    )


def _expected_filetype(kind: str) -> str:
    return "Raman RAW" if kind == "raw" else "Raman Processed"


def _parse_excitation(value: str) -> float | None:
    match = re.fullmatch(
        r"\s*"
        r"(?P<number>[+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"(?:\s+nm)?"
        r"\s*",
        value,
        re.IGNORECASE,
    )
    if match is None:
        return None
    parsed = float(match.group("number"))
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _comment_values(
    ignored_lines: tuple[RruffIgnoredLine, ...],
) -> dict[str, tuple[str, ...]]:
    values: dict[str, list[str]] = {}
    for ignored_line in ignored_lines:
        if ignored_line.category != "comment":
            continue
        body = ignored_line.text[1:]
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        values.setdefault(key, []).append(value.strip())
    return {
        key: tuple(current_values)
        for key, current_values in values.items()
    }


def _unique_nonempty_string(
    values: tuple[str, ...],
) -> str | None:
    if (
        not values
        or any(value == "" for value in values)
        or len(set(values)) != 1
    ):
        return None
    return values[0]


def _unique_positive_float(
    values: tuple[str, ...],
) -> float | None:
    parsed_values: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except ValueError:
            return None
        if not math.isfinite(parsed) or parsed <= 0:
            return None
        parsed_values.append(parsed)
    if not parsed_values or len(set(parsed_values)) != 1:
        return None
    return parsed_values[0]


def _unique_positive_integer(
    values: tuple[str, ...],
) -> int | None:
    parsed_values: list[int] = []
    for value in values:
        if re.fullmatch(r"[0-9]+", value) is None:
            return None
        parsed = int(value, 10)
        if parsed <= 0:
            return None
        parsed_values.append(parsed)
    if not parsed_values or len(set(parsed_values)) != 1:
        return None
    return parsed_values[0]


def _unique_wavelength(
    values: tuple[str, ...],
) -> float | None:
    parsed_values: list[float] = []
    for value in values:
        parsed = _parse_excitation(value)
        if parsed is None:
            return None
        parsed_values.append(parsed)
    if not parsed_values or len(set(parsed_values)) != 1:
        return None
    return parsed_values[0]


def _promote_rruff_acquisition(
    header_entries: tuple[RruffHeaderEntry, ...],
    ignored_lines: tuple[RruffIgnoredLine, ...],
) -> _RruffAcquisitionFields:
    header_values = _header_values(header_entries)
    comment_values = _comment_values(ignored_lines)
    header_wavelength = _unique_wavelength(
        header_values.get("RAMAN WAVELENGTH", ())
    )
    comment_wavelength = _unique_wavelength(
        comment_values.get("Laser (nm)", ())
    )
    if (
        header_wavelength is not None
        and comment_wavelength is not None
        and header_wavelength != comment_wavelength
    ):
        excitation_nm = None
    elif header_wavelength is not None:
        excitation_nm = header_wavelength
    else:
        excitation_nm = comment_wavelength
    return _RruffAcquisitionFields(
        instrument=_unique_nonempty_string(
            comment_values.get("Instrument", ())
        ),
        excitation_nm=excitation_nm,
        integration_time_s=_unique_positive_float(
            comment_values.get("Acq. time (s)", ())
        ),
        n_accumulations=_unique_positive_integer(
            comment_values.get("Accumulations", ())
        ),
        grating=_unique_nonempty_string(
            comment_values.get("Grating", ())
        ),
        detector=_unique_nonempty_string(
            comment_values.get("Detector", ())
        ),
    )


def _split_numeric_candidate(
    stripped: str,
) -> tuple[list[str], str | None]:
    if "," in stripped:
        return [field.strip() for field in stripped.split(",")], "comma"
    return stripped.split(), "whitespace"


def _try_float(value: str) -> tuple[bool, float | None]:
    try:
        return True, float(value)
    except ValueError:
        return False, None


def _statistics(
    *,
    parsed_numeric_rows: int,
    comma_numeric_rows: int,
    whitespace_numeric_rows: int,
    ignored_lines: tuple[RruffIgnoredLine, ...],
    duplicate_header: bool,
) -> _MemberParseStatistics:
    return _MemberParseStatistics(
        parsed_numeric_rows=parsed_numeric_rows,
        comma_numeric_rows=comma_numeric_rows,
        whitespace_numeric_rows=whitespace_numeric_rows,
        ignored_comment_lines=sum(
            line.category == "comment" for line in ignored_lines
        ),
        ignored_preamble_lines=sum(
            line.category == "preamble" for line in ignored_lines
        ),
        ignored_column_header_lines=sum(
            line.category == "column_header" for line in ignored_lines
        ),
        duplicate_header=duplicate_header,
    )


def _rejected_outcome(
    *,
    archive: str,
    source_member: str,
    member_id: str,
    member_sha256: str,
    member_bytes: int,
    measurement_key: str,
    kind: str,
    text_encoding: str | None,
    header_entries: tuple[RruffHeaderEntry, ...],
    ignored_lines: tuple[RruffIgnoredLine, ...],
    rejection_code: str,
    reason: str,
    parsed_numeric_rows: int,
    comma_numeric_rows: int,
    whitespace_numeric_rows: int,
    duplicate_header: bool,
) -> _MemberParseOutcome:
    values = _header_values(header_entries)
    mineral_values = values.get("NAMES", ())
    rruff_values = values.get("RRUFFID", ())
    mineral_name = (
        mineral_values[-1]
        if mineral_values and len(set(mineral_values)) == 1
        else None
    )
    rruff_id = (
        rruff_values[-1]
        if rruff_values and len(set(rruff_values)) == 1
        else None
    )
    return _MemberParseOutcome(
        accepted=None,
        rejected=RruffRejectedMember(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            mineral_name=mineral_name,
            rruff_id=rruff_id,
            rejection_code=rejection_code,
            reason=reason,
        ),
        statistics=_statistics(
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            ignored_lines=ignored_lines,
            duplicate_header=duplicate_header,
        ),
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise RruffValidationError(
            str(path),
            f"cannot read archive: {error}",
            "ARCHIVE_HASH_MISMATCH",
        ) from error
    return digest.hexdigest()


def _fatal(path: str, reason: str, code: str) -> RruffValidationError:
    return RruffValidationError(path, reason, code)


def _preflight_archives(
    raw_root: Path,
    contract: _RruffSourceContract,
) -> tuple[
    Mapping[str, str],
    Mapping[str, tuple[zipfile.ZipInfo, ...]],
]:
    expected_names = set(contract.archives)
    observed_names = {
        path.name
        for path in raw_root.glob("*.zip")
        if path.is_file()
    }
    missing = expected_names - observed_names
    if missing:
        raise _fatal(
            "raw_root.archives",
            f"missing archives: {sorted(missing)}",
            "ARCHIVE_MISSING",
        )
    unexpected = observed_names - expected_names
    if unexpected:
        raise _fatal(
            "raw_root.archives",
            f"unexpected archives: {sorted(unexpected)}",
            "ARCHIVE_UNEXPECTED",
        )

    archive_sha256: dict[str, str] = {}
    archive_infos: dict[str, tuple[zipfile.ZipInfo, ...]] = {}
    observed_total = 0
    for archive_name in sorted(
        contract.archives,
        key=lambda value: value.encode("utf-8"),
    ):
        archive_contract = contract.archives[archive_name]
        archive_path = raw_root / archive_name
        try:
            archive_bytes = archive_path.stat().st_size
        except OSError as error:
            raise _fatal(
                f"archives.{archive_name}.bytes",
                str(error),
                "ARCHIVE_HASH_MISMATCH",
            ) from error
        if archive_bytes != archive_contract.bytes:
            raise _fatal(
                f"archives.{archive_name}.bytes",
                (
                    f"expected {archive_contract.bytes}, "
                    f"observed {archive_bytes}"
                ),
                "ARCHIVE_HASH_MISMATCH",
            )
        observed_hash = _sha256_file(archive_path)
        if observed_hash != archive_contract.sha256:
            raise _fatal(
                f"archives.{archive_name}.sha256",
                (
                    f"expected {archive_contract.sha256}, "
                    f"observed {observed_hash}"
                ),
                "ARCHIVE_HASH_MISMATCH",
            )
        archive_sha256[archive_name] = observed_hash
        try:
            with zipfile.ZipFile(archive_path) as archive:
                infos = tuple(
                    info for info in archive.infolist() if not info.is_dir()
                )
                duplicate_paths = [
                    name
                    for name, count in Counter(
                        info.filename for info in infos
                    ).items()
                    if count > 1
                ]
                if duplicate_paths:
                    raise _fatal(
                        f"archives.{archive_name}.members",
                        f"duplicate paths: {sorted(duplicate_paths)}",
                        "MEMBER_PATH_COLLISION",
                    )
                failed_member = archive.testzip()
                if failed_member is not None:
                    raise _fatal(
                        f"archives.{archive_name}.crc",
                        f"CRC failure in {failed_member}",
                        "ARCHIVE_CRC_FAILURE",
                    )
        except RruffValidationError:
            raise
        except (OSError, zipfile.BadZipFile) as error:
            raise _fatal(
                f"archives.{archive_name}.crc",
                str(error),
                "ARCHIVE_CRC_FAILURE",
            ) from error
        if len(infos) != archive_contract.member_count:
            raise _fatal(
                f"archives.{archive_name}.members",
                (
                    f"expected {archive_contract.member_count}, "
                    f"observed {len(infos)}"
                ),
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        archive_infos[archive_name] = infos
        observed_total += len(infos)
    if observed_total != contract.source_members:
        raise _fatal(
            "source_members",
            f"expected {contract.source_members}, observed {observed_total}",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    return (
        MappingProxyType(archive_sha256),
        MappingProxyType(archive_infos),
    )


def _percentile95(counts: tuple[int, ...]) -> float:
    if not counts:
        return 0.0
    return float(np.quantile(np.asarray(counts), 0.95))


def _median(counts: tuple[int, ...]) -> float:
    if not counts:
        return 0.0
    return float(np.median(np.asarray(counts)))


def _immutable_int_mapping(
    values: Mapping[str, int],
) -> Mapping[str, int]:
    return MappingProxyType(dict(values))


def _statistics_value(
    statistics: _RruffSourceStatistics,
    name: str,
) -> int | float:
    return getattr(statistics, name)


def _parse_member_bytes(
    archive: str,
    source_member: str,
    payload: bytes,
) -> _MemberParseOutcome:
    measurement_key, kind, member_id = _parse_filename(source_member)
    member_sha256 = hashlib.sha256(payload).hexdigest()
    member_bytes = len(payload)
    try:
        text, text_encoding = _decode_payload(payload)
    except UnicodeDecodeError:
        return _rejected_outcome(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=None,
            header_entries=(),
            ignored_lines=(),
            rejection_code="TEXT_DECODE_FAILURE",
            reason="payload is neither strict UTF-8 nor strict CP1252",
            parsed_numeric_rows=0,
            comma_numeric_rows=0,
            whitespace_numeric_rows=0,
            duplicate_header=False,
        )

    headers: list[RruffHeaderEntry] = []
    ignored: list[RruffIgnoredLine] = []
    axis_values: list[float] = []
    intensity_values: list[float] = []
    comma_numeric_rows = 0
    whitespace_numeric_rows = 0
    numeric_started = False
    rejection: tuple[str, str] | None = None

    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped == "":
            continue
        if line.startswith("##"):
            remainder = line[2:]
            if "=" in remainder:
                key, value = remainder.split("=", 1)
                headers.append(
                    RruffHeaderEntry(
                        line=line_number,
                        key=key,
                        value=value,
                    )
                )
                continue
            if numeric_started:
                if rejection is None:
                    rejection = (
                        "NUMERIC_ROW_INVALID",
                        f"line {line_number} is malformed after numeric data",
                    )
                continue
            ignored.append(
                RruffIgnoredLine(
                    line=line_number,
                    category="preamble",
                    text=line,
                )
            )
            continue
        if line.startswith("#"):
            ignored.append(
                RruffIgnoredLine(
                    line=line_number,
                    category="comment",
                    text=line,
                )
            )
            continue

        fields, delimiter = _split_numeric_candidate(stripped)
        parsed_fields = tuple(_try_float(field) for field in fields)
        numeric_field_count = sum(result[0] for result in parsed_fields)
        first_numeric = len(fields) == 2 and parsed_fields[0][0]
        second_numeric = len(fields) == 2 and parsed_fields[1][0]
        first_value = parsed_fields[0][1] if len(fields) == 2 else None
        second_value = parsed_fields[1][1] if len(fields) == 2 else None

        if len(fields) == 2 and first_numeric and second_numeric:
            axis_values.append(float(first_value))
            intensity_values.append(float(second_value))
            numeric_started = True
            if delimiter == "comma":
                comma_numeric_rows += 1
            else:
                whitespace_numeric_rows += 1
            continue

        if (
            len(fields) != 2
            and numeric_field_count == len(fields)
            and numeric_field_count > 0
        ):
            if not numeric_started:
                ignored.append(
                    RruffIgnoredLine(
                        line=line_number,
                        category="preamble",
                        text=line,
                    )
                )
            if rejection is None:
                rejection = (
                    "NUMERIC_ROW_INVALID",
                    (
                        f"line {line_number} contains {len(fields)} fields; "
                        "numeric rows require exactly two"
                    ),
                )
            continue
        if len(fields) == 2 and (first_numeric or second_numeric):
            if not numeric_started:
                ignored.append(
                    RruffIgnoredLine(
                        line=line_number,
                        category="preamble",
                        text=line,
                    )
                )
            if rejection is None:
                rejection = (
                    "NUMERIC_ROW_INVALID",
                    f"line {line_number} contains exactly one numeric field",
                )
            continue
        if numeric_started:
            if rejection is None:
                rejection = (
                    "NUMERIC_ROW_INVALID",
                    f"line {line_number} is malformed after numeric data",
                )
            continue
        category = (
            "column_header"
            if len(fields) == 2
            and tuple(field.casefold() for field in fields) in _COLUMN_HEADERS
            else "preamble"
        )
        ignored.append(
            RruffIgnoredLine(
                line=line_number,
                category=category,
                text=line,
            )
        )

    header_entries = tuple(headers)
    ignored_lines = tuple(ignored)
    header_keys = [entry.key for entry in header_entries]
    duplicate_header = len(header_keys) != len(set(header_keys))
    parsed_numeric_rows = len(axis_values)

    if rejection is not None:
        code, reason = rejection
        return _rejected_outcome(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            rejection_code=code,
            reason=reason,
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            duplicate_header=duplicate_header,
        )

    try:
        identity, pin_id, orientation = _identity_fields(header_entries)
    except RruffValidationError as error:
        return _rejected_outcome(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            rejection_code=error.code,
            reason=error.reason,
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            duplicate_header=duplicate_header,
        )

    if identity["FILETYPE"] != _expected_filetype(kind):
        return _rejected_outcome(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            rejection_code="FILETYPE_MISMATCH",
            reason=(
                f"filename kind {kind!r} conflicts with "
                f"FILETYPE {identity['FILETYPE']!r}"
            ),
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            duplicate_header=duplicate_header,
        )

    if parsed_numeric_rows < 2:
        return _rejected_outcome(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            rejection_code="NUMERIC_DATA_MISSING",
            reason="at least two complete numeric rows are required",
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            duplicate_header=duplicate_header,
        )

    source_axis = np.asarray(axis_values, dtype=np.float64)
    source_intensity = np.asarray(intensity_values, dtype=np.float64)
    if not np.isfinite(source_axis).all() or not np.isfinite(
        source_intensity
    ).all():
        return _rejected_outcome(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            rejection_code="NONFINITE_VALUE",
            reason="source axis or intensity contains a non-finite value",
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            duplicate_header=duplicate_header,
        )

    differences = np.diff(source_axis)
    if np.any(differences == 0):
        code = "AXIS_DUPLICATE"
        reason = "source float64 axis contains duplicate values"
        axis_direction = ""
    elif np.all(differences > 0):
        code = ""
        reason = ""
        axis_direction = "increasing"
    elif np.all(differences < 0):
        code = ""
        reason = ""
        axis_direction = "decreasing"
    else:
        code = "AXIS_NONMONOTONIC"
        reason = "source float64 axis is not strictly monotonic"
        axis_direction = ""
    if code:
        return _rejected_outcome(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            rejection_code=code,
            reason=reason,
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            duplicate_header=duplicate_header,
        )

    with np.errstate(over="ignore", invalid="ignore"):
        stored_axis = np.asarray(source_axis, dtype=np.float32)
        stored_intensity = np.asarray(source_intensity, dtype=np.float32)
    if not np.isfinite(stored_axis).all():
        code = "AXIS_FLOAT32_NONFINITE"
        reason = "float32 axis conversion creates a non-finite value"
    elif not np.isfinite(stored_intensity).all():
        code = "INTENSITY_FLOAT32_NONFINITE"
        reason = "float32 intensity conversion creates a non-finite value"
    else:
        stored_differences = np.diff(stored_axis)
        expected_direction = (
            np.all(stored_differences > 0)
            if axis_direction == "increasing"
            else np.all(stored_differences < 0)
        )
        if not expected_direction:
            code = "AXIS_FLOAT32_COLLAPSE"
            reason = (
                "float32 axis conversion does not preserve strict monotonicity"
            )
        else:
            code = ""
            reason = ""
    if code:
        return _rejected_outcome(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            rejection_code=code,
            reason=reason,
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            duplicate_header=duplicate_header,
        )

    axis_cast_error = float(
        np.max(np.abs(source_axis - stored_axis.astype(np.float64)))
    )
    intensity_cast_error = float(
        np.max(
            np.abs(
                source_intensity
                - stored_intensity.astype(np.float64)
            )
        )
    )
    source_axis = np.frombuffer(
        np.asarray(source_axis, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    )
    source_intensity = np.frombuffer(
        np.asarray(source_intensity, dtype="<f8").tobytes(order="C"),
        dtype="<f8",
    )
    return _MemberParseOutcome(
        accepted=_ParsedRruffMember(
            archive=archive,
            source_member=source_member,
            member_id=member_id,
            member_sha256=member_sha256,
            member_bytes=member_bytes,
            measurement_key=measurement_key,
            kind=kind,
            text_encoding=text_encoding,
            header_entries=header_entries,
            ignored_lines=ignored_lines,
            mineral_name=identity["NAMES"],
            rruff_id=identity["RRUFFID"],
            pin_id=pin_id,
            orientation=orientation,
            excitation_nm=_parse_excitation(identity["RAMAN WAVELENGTH"]),
            source_point_count=parsed_numeric_rows,
            source_axis=source_axis,
            source_intensity=source_intensity,
            axis_direction=axis_direction,
            axis_id=axis_id(stored_axis),
            axis_float32_max_abs_error=axis_cast_error,
            intensity_float32_max_abs_error=intensity_cast_error,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
        ),
        rejected=None,
        statistics=_statistics(
            parsed_numeric_rows=parsed_numeric_rows,
            comma_numeric_rows=comma_numeric_rows,
            whitespace_numeric_rows=whitespace_numeric_rows,
            ignored_lines=ignored_lines,
            duplicate_header=duplicate_header,
        ),
    )


def _inspect_rruff_source(
    raw_root: Path,
    contract: _RruffSourceContract,
) -> _RruffSourceInspection:
    raw_root = Path(raw_root)
    archive_sha256, archive_infos = _preflight_archives(
        raw_root,
        contract,
    )

    member_ids: dict[str, tuple[str, str]] = {}
    observed_raw = 0
    observed_processed = 0
    for archive_name in sorted(
        archive_infos,
        key=lambda value: value.encode("utf-8"),
    ):
        archive_raw = 0
        archive_processed = 0
        for info in archive_infos[archive_name]:
            try:
                _, kind, member_id = _parse_filename(info.filename)
            except RruffValidationError as error:
                raise _fatal(
                    f"archives.{archive_name}/{info.filename}",
                    error.reason,
                    error.code,
                ) from error
            if member_id in member_ids:
                previous_archive, previous_member = member_ids[member_id]
                raise _fatal(
                    f"member_ids.{member_id}",
                    (
                        f"collision between {previous_archive}/"
                        f"{previous_member} and {archive_name}/"
                        f"{info.filename}"
                    ),
                    "MEMBER_ID_COLLISION",
                )
            member_ids[member_id] = (archive_name, info.filename)
            if kind == "raw":
                archive_raw += 1
            else:
                archive_processed += 1
        archive_contract = contract.archives[archive_name]
        if (
            archive_raw != archive_contract.raw_members
            or archive_processed != archive_contract.processed_members
        ):
            raise _fatal(
                f"archives.{archive_name}.kind_counts",
                (
                    f"expected raw/processed "
                    f"{archive_contract.raw_members}/"
                    f"{archive_contract.processed_members}, observed "
                    f"{archive_raw}/{archive_processed}"
                ),
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        observed_raw += archive_raw
        observed_processed += archive_processed
    if (
        observed_raw != contract.source_raw
        or observed_processed != contract.source_processed
    ):
        raise _fatal(
            "source_kind_counts",
            (
                f"expected raw/processed {contract.source_raw}/"
                f"{contract.source_processed}, observed "
                f"{observed_raw}/{observed_processed}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )

    accepted_members: list[_SourceAcceptedMember] = []
    rejected_members: list[RruffRejectedMember] = []
    rejection_code_counts: Counter[str] = Counter()
    parser_counts: Counter[str] = Counter()
    raw_axis_counts: Counter[str] = Counter()
    processed_axis_counts: Counter[str] = Counter()
    raw_names: set[str] = set()
    processed_names: set[str] = set()
    raw_rruff_ids: set[str] = set()
    processed_rruff_ids: set[str] = set()
    mineral_names_by_rruff_id: dict[str, set[str]] = {}
    acquisition_with_excitation = 0
    acquisition_without_excitation = 0
    full_comment_acquisition = 0
    accepted_spectral_points = 0
    rejected_parsed_rows = 0
    increasing_axes = 0
    decreasing_axes = 0
    axis_cast_maximum = 0.0
    intensity_cast_maximum = 0.0

    for archive_name in sorted(
        archive_infos,
        key=lambda value: value.encode("utf-8"),
    ):
        archive_path = raw_root / archive_name
        with zipfile.ZipFile(archive_path) as archive:
            for info in sorted(
                archive_infos[archive_name],
                key=lambda value: value.filename.encode("utf-8"),
            ):
                try:
                    payload = archive.read(info)
                except (OSError, zipfile.BadZipFile) as error:
                    raise _fatal(
                        f"archives.{archive_name}.crc",
                        str(error),
                        "ARCHIVE_CRC_FAILURE",
                    ) from error
                outcome = _parse_member_bytes(
                    archive_name,
                    info.filename,
                    payload,
                )
                parser_counts[
                    "comma_numeric_rows"
                ] += outcome.statistics.comma_numeric_rows
                parser_counts[
                    "whitespace_numeric_rows"
                ] += outcome.statistics.whitespace_numeric_rows
                parser_counts[
                    "ignored_comment_lines"
                ] += outcome.statistics.ignored_comment_lines
                parser_counts[
                    "ignored_preamble_lines"
                ] += outcome.statistics.ignored_preamble_lines
                parser_counts[
                    "ignored_column_header_lines"
                ] += outcome.statistics.ignored_column_header_lines
                parser_counts["duplicate_header_records"] += int(
                    outcome.statistics.duplicate_header
                )
                text_encoding = (
                    outcome.accepted.text_encoding
                    if outcome.accepted is not None
                    else outcome.rejected.text_encoding
                )
                if text_encoding is not None:
                    parser_counts[f"{text_encoding}_members"] += 1

                if outcome.rejected is not None:
                    rejected_members.append(outcome.rejected)
                    rejected_parsed_rows += (
                        outcome.statistics.parsed_numeric_rows
                    )
                    rejection_code_counts[
                        outcome.rejected.rejection_code
                    ] += 1
                    continue

                parsed = outcome.accepted
                if parsed is None:
                    raise AssertionError("parse outcome has no result")
                acquisition = _promote_rruff_acquisition(
                    parsed.header_entries,
                    parsed.ignored_lines,
                )
                accepted_members.append(
                    _SourceAcceptedMember(
                        archive=parsed.archive,
                        source_member=parsed.source_member,
                        member_id=parsed.member_id,
                        member_sha256=parsed.member_sha256,
                        member_bytes=parsed.member_bytes,
                        measurement_key=parsed.measurement_key,
                        kind=parsed.kind,
                        text_encoding=parsed.text_encoding,
                        header_entries=parsed.header_entries,
                        ignored_lines=parsed.ignored_lines,
                        mineral_name=parsed.mineral_name,
                        rruff_id=parsed.rruff_id,
                        pin_id=parsed.pin_id,
                        orientation=parsed.orientation,
                        excitation_nm=acquisition.excitation_nm,
                        source_point_count=parsed.source_point_count,
                        axis_id=parsed.axis_id,
                        axis_float32_max_abs_error=(
                            parsed.axis_float32_max_abs_error
                        ),
                        intensity_float32_max_abs_error=(
                            parsed.intensity_float32_max_abs_error
                        ),
                    )
                )
                accepted_spectral_points += parsed.source_point_count
                axis_cast_maximum = max(
                    axis_cast_maximum,
                    parsed.axis_float32_max_abs_error,
                )
                intensity_cast_maximum = max(
                    intensity_cast_maximum,
                    parsed.intensity_float32_max_abs_error,
                )
                if parsed.axis_direction == "increasing":
                    increasing_axes += 1
                else:
                    decreasing_axes += 1
                if acquisition.excitation_nm is None:
                    acquisition_without_excitation += 1
                else:
                    acquisition_with_excitation += 1
                if all(
                    value is not None
                    for value in (
                        acquisition.instrument,
                        acquisition.integration_time_s,
                        acquisition.n_accumulations,
                        acquisition.grating,
                        acquisition.detector,
                    )
                ) and parsed.kind == "raw":
                    full_comment_acquisition += 1
                mineral_names_by_rruff_id.setdefault(
                    parsed.rruff_id,
                    set(),
                ).add(parsed.mineral_name)
                if parsed.kind == "raw":
                    raw_axis_counts[parsed.axis_id] += 1
                    raw_names.add(parsed.mineral_name)
                    raw_rruff_ids.add(parsed.rruff_id)
                else:
                    processed_axis_counts[parsed.axis_id] += 1
                    processed_names.add(parsed.mineral_name)
                    processed_rruff_ids.add(parsed.rruff_id)

    accepted_members.sort(
        key=lambda member: (
            member.archive.encode("utf-8"),
            member.source_member.encode("utf-8"),
        )
    )
    rejected_members.sort(
        key=lambda member: (
            member.archive.encode("utf-8"),
            member.source_member.encode("utf-8"),
        )
    )
    observed_rejections = frozenset(
        (
            member.archive,
            member.source_member,
            member.kind,
            member.rejection_code,
        )
        for member in rejected_members
    )
    if observed_rejections != contract.expected_rejections:
        raise _fatal(
            "rejected_members",
            (
                f"expected {sorted(contract.expected_rejections)}, "
                f"observed {sorted(observed_rejections)}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    accepted_raw = sum(
        member.kind == "raw" for member in accepted_members
    )
    accepted_processed = len(accepted_members) - accepted_raw
    if (
        len(accepted_members) != contract.accepted_records
        or accepted_raw != contract.accepted_raw
        or accepted_processed != contract.accepted_processed
    ):
        raise _fatal(
            "accepted_members",
            (
                f"expected total/raw/processed "
                f"{contract.accepted_records}/{contract.accepted_raw}/"
                f"{contract.accepted_processed}, observed "
                f"{len(accepted_members)}/{accepted_raw}/"
                f"{accepted_processed}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    for rruff_id, mineral_names in mineral_names_by_rruff_id.items():
        if len(mineral_names) > 1:
            raise _fatal(
                f"rruff_ids.{rruff_id}.mineral_names",
                f"observed conflicting names: {sorted(mineral_names)}",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )

    ordered_names = sorted(
        raw_names | processed_names,
        key=lambda value: value.encode("utf-8"),
    )
    class_ids = {
        name: index for index, name in enumerate(ordered_names)
    }
    global_class_labels = MappingProxyType(
        {index: name for name, index in class_ids.items()}
    )
    raw_class_labels = MappingProxyType(
        {class_ids[name]: name for name in ordered_names if name in raw_names}
    )
    processed_class_labels = MappingProxyType(
        {
            class_ids[name]: name
            for name in ordered_names
            if name in processed_names
        }
    )
    raw_axis_values = tuple(raw_axis_counts.values())
    processed_axis_values = tuple(processed_axis_counts.values())
    statistics = _RruffSourceStatistics(
        source_members=observed_raw + observed_processed,
        source_raw=observed_raw,
        source_processed=observed_processed,
        accepted_records=len(accepted_members),
        rejected_records=len(rejected_members),
        accepted_raw=accepted_raw,
        accepted_processed=accepted_processed,
        accepted_spectral_points=accepted_spectral_points,
        rejected_member_parsed_numeric_rows=rejected_parsed_rows,
        all_parsed_numeric_rows=(
            parser_counts["comma_numeric_rows"]
            + parser_counts["whitespace_numeric_rows"]
        ),
        mineral_classes=len(ordered_names),
        rruff_samples=len(raw_rruff_ids | processed_rruff_ids),
        raw_mineral_classes=len(raw_names),
        processed_mineral_classes=len(processed_names),
        raw_rruff_samples=len(raw_rruff_ids),
        processed_rruff_samples=len(processed_rruff_ids),
        utf8_members=parser_counts["utf-8_members"],
        cp1252_members=parser_counts["cp1252_members"],
        comma_numeric_rows=parser_counts["comma_numeric_rows"],
        whitespace_numeric_rows=parser_counts["whitespace_numeric_rows"],
        ignored_comment_lines=parser_counts["ignored_comment_lines"],
        ignored_preamble_lines=parser_counts["ignored_preamble_lines"],
        ignored_column_header_lines=parser_counts[
            "ignored_column_header_lines"
        ],
        duplicate_header_records=parser_counts["duplicate_header_records"],
        increasing_axes=increasing_axes,
        decreasing_axes=decreasing_axes,
        records_with_excitation_nm=acquisition_with_excitation,
        records_with_excitation_nm_none=acquisition_without_excitation,
        records_with_full_comment_acquisition_metadata=(
            full_comment_acquisition
        ),
        union_axis_groups=len(
            set(raw_axis_counts) | set(processed_axis_counts)
        ),
        raw_axis_groups=len(raw_axis_counts),
        processed_axis_groups=len(processed_axis_counts),
        shared_axis_groups=len(
            set(raw_axis_counts) & set(processed_axis_counts)
        ),
        raw_singleton_axis_groups=sum(
            count == 1 for count in raw_axis_values
        ),
        processed_singleton_axis_groups=sum(
            count == 1 for count in processed_axis_values
        ),
        raw_median_records_per_axis_group=_median(raw_axis_values),
        processed_median_records_per_axis_group=_median(
            processed_axis_values
        ),
        raw_p95_records_per_axis_group=_percentile95(raw_axis_values),
        processed_p95_records_per_axis_group=_percentile95(
            processed_axis_values
        ),
        raw_maximum_records_per_axis_group=max(raw_axis_values, default=0),
        processed_maximum_records_per_axis_group=max(
            processed_axis_values,
            default=0,
        ),
        axis_float32_max_abs_error_cm1=axis_cast_maximum,
        intensity_float32_max_abs_error=intensity_cast_maximum,
        rejection_code_counts=_immutable_int_mapping(
            rejection_code_counts
        ),
    )
    if contract.expected_statistics is not None:
        for name, expected in contract.expected_statistics.items():
            observed = _statistics_value(statistics, name)
            if observed != expected:
                raise _fatal(
                    f"statistics.{name}",
                    f"expected {expected}, observed {observed}",
                    "SOURCE_MEMBER_COUNT_MISMATCH",
                )
    return _RruffSourceInspection(
        raw_root=raw_root,
        archive_sha256=archive_sha256,
        accepted_members=tuple(accepted_members),
        rejected_members=tuple(rejected_members),
        global_class_labels=global_class_labels,
        raw_class_labels=raw_class_labels,
        processed_class_labels=processed_class_labels,
        raw_rruff_ids=frozenset(raw_rruff_ids),
        processed_rruff_ids=frozenset(processed_rruff_ids),
        raw_axis_ids=frozenset(raw_axis_counts),
        processed_axis_ids=frozenset(processed_axis_counts),
        statistics=statistics,
    )
