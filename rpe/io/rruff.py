from __future__ import annotations

import zipfile
import hashlib
import json
import math
import os
import resource
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from time import perf_counter
from types import MappingProxyType
from typing import Iterator, Mapping

import numpy as np

from rpe.io.rruff_pairing import (
    RruffPairInspection,
    RruffPairMember,
    _axis_relation,
    _canonical_json_bytes,
    _pair_id,
    _pair_status,
    _record_id,
    _validate_rruff_pair_index,
    _write_rruff_pair_index,
)
from rpe.io.rruff_source import (
    PRODUCTION_SOURCE_CONTRACT,
    RruffHeaderEntry,
    RruffIgnoredLine,
    RruffRejectedMember,
    RruffValidationError,
    _RruffSourceContract,
    _RruffSourceInspection,
    _RruffSourceStatistics,
    _SourceAcceptedMember,
    _header_values,
    _inspect_rruff_source,
    _parse_member_bytes,
    _promote_rruff_acquisition,
)
from rpe.io.schema import (
    SCHEMA_VERSION,
    LicenseStatus,
    PreprocessingStatus,
    PreprocessingStep,
    Provenance,
    RamanRecord,
    SpectrumMetadata,
    Targets,
    validate_record,
)
from rpe.io.store import (
    DATASET_FILES,
    UnifiedDataset,
    validate_dataset,
    write_dataset,
)


RAW_DATASET_ID = "rruff_raman_raw"
PROCESSED_DATASET_ID = "rruff_raman_processed"
SOURCE_DATASET_ID = "RRUFF Raman"
SOURCE_LICENSE = "not stated"
ADAPTER_VERSION = "0.1.0"
RECEIPT_NAME = "rruff_raman_conversion.json"
RETRIEVED_DATE = date(2026, 8, 14)
_SOURCE_URL_PREFIX = "https://rruff.info/zipped_data_files/raman/"
_PAIR_INDEX_NAME = "rruff_raman_pairs.jsonl"
_PAIR_STATUSES = (
    "paired_unique",
    "raw_only",
    "processed_only",
    "ambiguous",
    "rejected_only",
)
_AXIS_RELATIONS = (
    "exact_equal",
    "processed_exact_contiguous_subset_of_raw",
    "raw_exact_contiguous_subset_of_processed",
    "overlap_requires_alignment",
    "no_axis_overlap",
)
_POINTWISE_RELATIONS = frozenset(_AXIS_RELATIONS[:3])

_RECEIPT_KEYS = {
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
_SOURCE_ARCHIVE_KEYS = {
    "archive",
    "url",
    "bytes",
    "sha256",
    "retrieved_date",
    "member_count",
    "raw_members",
    "processed_members",
}
_SOURCE_INVENTORY_FIELDS = (
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
)
_PARSER_FIELDS = (
    "utf8_members",
    "cp1252_members",
    "comma_numeric_rows",
    "whitespace_numeric_rows",
    "ignored_comment_lines",
    "ignored_preamble_lines",
    "ignored_column_header_lines",
    "duplicate_header_records",
)
_REJECTED_MEMBER_KEYS = {
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
_DATASET_RECEIPT_KEYS = {
    "records",
    "eligible_records",
    "observed_class_ids",
    "axis_groups",
}
_PAIRING_RECEIPT_KEYS = {
    "index_file",
    "index_rows",
    "status_counts",
    "axis_relation_counts",
    "pointwise_comparable_pairs",
}
_PRECISION_KEYS = {
    "axis_float32_max_abs_error_cm1",
    "intensity_float32_max_abs_error",
}
_LICENSE_KEYS = {
    "text",
    "status",
    "access_status",
    "redistribution_requires_review",
}
_LIMITATION_KEYS = {
    "processed_is_not_clean_target",
    "processed_operations_unknown",
    "rejected_source_members",
    "alignment_required_pairs",
    "no_overlap_pairs",
    "taxonomy_normalization_applied",
    "resampling_applied",
}
_OUTPUT_KEYS = {
    RAW_DATASET_ID,
    PROCESSED_DATASET_ID,
    _PAIR_INDEX_NAME,
}
_OUTPUT_ARTIFACT_KEYS = {"artifact_type", "files"}
_OUTPUT_FILE_KEYS = {"bytes", "sha256"}


@dataclass(frozen=True)
class RruffAcceptedMember:
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
    excitation_nm: float | None
    source_point_count: int
    axis_id: str
    axis_float32_max_abs_error: float
    intensity_float32_max_abs_error: float
    pair_id: str
    pair_status: str


@dataclass(frozen=True)
class RruffInspection:
    raw_root: Path
    archive_sha256: Mapping[str, str]
    source_member_count: int
    accepted_members: tuple[RruffAcceptedMember, ...]
    rejected_members: tuple[RruffRejectedMember, ...]
    pairs: tuple[RruffPairInspection, ...]
    global_class_labels: Mapping[int, str]
    raw_axis_group_count: int
    processed_axis_group_count: int


@dataclass(frozen=True)
class _RruffInspectionBundle:
    inspection: RruffInspection
    statistics: _RruffSourceStatistics


@dataclass(frozen=True)
class RruffDatasetSummary:
    dataset_id: str
    record_count: int
    eligible_records: int
    axis_group_count: int
    output_bytes: int
    files: Mapping[str, Mapping[str, int | str]]


@dataclass(frozen=True)
class RruffConversionSummary:
    raw: RruffDatasetSummary
    processed: RruffDatasetSummary
    pair_index_path: Path
    pair_index_sha256: str
    pair_index_rows: int
    pair_status_counts: Mapping[str, int]
    axis_relation_counts: Mapping[str, int]
    receipt_path: Path
    receipt_sha256: str
    rejected_records: int


@dataclass(frozen=True)
class RruffRuntimeMetrics:
    total_seconds: float
    inspection_seconds: float
    staged_build_seconds: float
    raw_build_seconds: float
    processed_build_seconds: float
    pair_index_seconds: float
    receipt_seconds: float
    validation_seconds: float
    peak_rss_raw: int
    peak_rss_bytes: int
    raw_output_bytes: int
    processed_output_bytes: int
    pair_index_bytes: int
    receipt_bytes: int
    combined_output_bytes: int
    source_archive_bytes: int
    compression_ratio: float
    raw_axis_groups: int
    processed_axis_groups: int
    lazy_open_seconds_raw: float
    lazy_open_seconds_processed: float
    representative_record_read_seconds_raw: float
    representative_record_read_seconds_processed: float


@dataclass(frozen=True)
class _RruffComparisonSummary:
    source_rows_compared: int
    intensity_mismatches: int
    class_label_mismatches: int
    eligibility_violations: int


@dataclass(frozen=True)
class _RruffStagedData:
    raw: RruffDatasetSummary
    processed: RruffDatasetSummary
    pair_index_path: Path
    pair_index_sha256: str
    pair_index_rows: int
    pair_status_counts: Mapping[str, int]
    axis_relation_counts: Mapping[str, int]
    raw_comparison: _RruffComparisonSummary
    processed_comparison: _RruffComparisonSummary


@dataclass(frozen=True)
class _RruffStagingReview:
    raw: RruffDatasetSummary
    processed: RruffDatasetSummary
    pair_index_sha256: str
    pair_index_rows: int
    pair_status_counts: Mapping[str, int]
    axis_relation_counts: Mapping[str, int]
    receipt_sha256: str
    rejected_records: int
    raw_comparison: _RruffComparisonSummary
    processed_comparison: _RruffComparisonSummary
    source_statistics: _RruffSourceStatistics
    output_hashes: Mapping[str, str]
    metrics: RruffRuntimeMetrics
    feasibility_failures: tuple[str, ...]


@dataclass(frozen=True)
class _PublicationArtifact:
    staged: Path
    final: Path
    backup_name: str


def _fatal(path: str, reason: str, code: str) -> RruffValidationError:
    return RruffValidationError(path, reason, code)


def _pair_member(
    member: _SourceAcceptedMember | RruffRejectedMember,
) -> RruffPairMember:
    if isinstance(member, _SourceAcceptedMember):
        return RruffPairMember(
            member_id=member.member_id,
            record_id=_record_id(member.kind, member.member_id),
            source_member=member.source_member,
            source_member_sha256=member.member_sha256,
            conversion_status="accepted",
            rejection_code=None,
        )
    return RruffPairMember(
        member_id=member.member_id,
        record_id=None,
        source_member=member.source_member,
        source_member_sha256=member.member_sha256,
        conversion_status="rejected",
        rejection_code=member.rejection_code,
    )


def _pair_metadata_value(
    member: _SourceAcceptedMember,
    key: str,
) -> str | None:
    if key == "NAMES":
        return member.mineral_name
    if key == "RRUFFID":
        return member.rruff_id
    if key == "PIN_ID":
        return member.pin_id
    if key == "ORIENTATION":
        return member.orientation
    if key == "RAMAN WAVELENGTH":
        values = _header_values(member.header_entries).get(key, ())
        return values[-1] if values else None
    raise KeyError(key)


def _require_pair_metadata_equal(
    *,
    archive: str,
    measurement_key: str,
    raw_member: _SourceAcceptedMember,
    processed_member: _SourceAcceptedMember,
) -> None:
    for key in (
        "NAMES",
        "RRUFFID",
        "RAMAN WAVELENGTH",
        "PIN_ID",
        "ORIENTATION",
    ):
        raw_value = _pair_metadata_value(raw_member, key)
        processed_value = _pair_metadata_value(processed_member, key)
        if raw_value != processed_value:
            raise _fatal(
                f"pairs.{archive}/{measurement_key}.{key}",
                (
                    f"raw value {raw_value!r} differs from "
                    f"processed value {processed_value!r}"
                ),
                "PAIR_METADATA_MISMATCH",
            )


def _reparse_pair_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    expected: _SourceAcceptedMember,
):
    outcome = _parse_member_bytes(
        expected.archive,
        expected.source_member,
        archive.read(info),
    )
    parsed = outcome.accepted
    if parsed is None:
        raise _fatal(
            f"pairs.{expected.archive}/{expected.source_member}",
            (
                "accepted member changed to rejection "
                f"{outcome.rejected.rejection_code}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    checks = {
        "member_sha256": (parsed.member_sha256, expected.member_sha256),
        "source_point_count": (
            parsed.source_point_count,
            expected.source_point_count,
        ),
        "axis_id": (parsed.axis_id, expected.axis_id),
        "axis_float32_max_abs_error": (
            parsed.axis_float32_max_abs_error,
            expected.axis_float32_max_abs_error,
        ),
        "intensity_float32_max_abs_error": (
            parsed.intensity_float32_max_abs_error,
            expected.intensity_float32_max_abs_error,
        ),
    }
    for name, (observed, expected_value) in checks.items():
        if observed != expected_value:
            raise _fatal(
                (
                    f"pairs.{expected.archive}/"
                    f"{expected.source_member}.{name}"
                ),
                f"expected {expected_value!r}, observed {observed!r}",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
    return parsed


def _required_info(
    infos: Mapping[str, zipfile.ZipInfo],
    member: _SourceAcceptedMember,
) -> zipfile.ZipInfo:
    try:
        return infos[member.source_member]
    except KeyError:
        raise _fatal(
            f"pairs.{member.archive}/{member.source_member}",
            "source member is missing during pair reparse",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from None


def _group_members(
    source_inspection: _RruffSourceInspection,
) -> Mapping[
    tuple[str, str],
    tuple[_SourceAcceptedMember | RruffRejectedMember, ...],
]:
    groups: dict[
        tuple[str, str],
        list[_SourceAcceptedMember | RruffRejectedMember],
    ] = defaultdict(list)
    for member in source_inspection.accepted_members:
        groups[(member.archive, member.measurement_key)].append(member)
    for member in source_inspection.rejected_members:
        groups[(member.archive, member.measurement_key)].append(member)
    return MappingProxyType(
        {
            key: tuple(
                sorted(
                    members,
                    key=lambda member: (
                        member.kind.encode("ascii"),
                        member.member_id.encode("ascii"),
                        member.source_member.encode("utf-8"),
                    ),
                )
            )
            for key, members in groups.items()
        }
    )


def _derive_relations(
    raw_root: Path,
    groups: Mapping[
        tuple[str, str],
        tuple[_SourceAcceptedMember | RruffRejectedMember, ...],
    ],
    statuses: Mapping[tuple[str, str], str],
) -> Mapping[
    tuple[str, str],
    tuple[
        str,
        tuple[int, int] | None,
        tuple[int, int] | None,
    ],
]:
    groups_by_archive: dict[
        str,
        list[
            tuple[
                tuple[str, str],
                tuple[_SourceAcceptedMember | RruffRejectedMember, ...],
            ]
        ],
    ] = defaultdict(list)
    for key, members in groups.items():
        if statuses[key] == "paired_unique":
            groups_by_archive[key[0]].append((key, members))

    relations = {}
    for archive_name in sorted(
        groups_by_archive,
        key=lambda value: value.encode("utf-8"),
    ):
        with zipfile.ZipFile(raw_root / archive_name) as archive:
            infos = {
                info.filename: info
                for info in archive.infolist()
                if not info.is_dir()
            }
            for (archive_value, measurement_key), members in sorted(
                groups_by_archive[archive_name],
                key=lambda item: item[0][1].encode("utf-8"),
            ):
                raw_member = next(
                    member
                    for member in members
                    if isinstance(member, _SourceAcceptedMember)
                    and member.kind == "raw"
                )
                processed_member = next(
                    member
                    for member in members
                    if isinstance(member, _SourceAcceptedMember)
                    and member.kind == "processed"
                )
                _require_pair_metadata_equal(
                    archive=archive_value,
                    measurement_key=measurement_key,
                    raw_member=raw_member,
                    processed_member=processed_member,
                )
                raw_parsed = _reparse_pair_member(
                    archive,
                    _required_info(infos, raw_member),
                    raw_member,
                )
                processed_parsed = _reparse_pair_member(
                    archive,
                    _required_info(infos, processed_member),
                    processed_member,
                )
                relations[(archive_value, measurement_key)] = _axis_relation(
                    raw_parsed.source_axis,
                    processed_parsed.source_axis,
                )
                del raw_parsed
                del processed_parsed
    return MappingProxyType(relations)


def _pairing_statistics(
    pairs: tuple[RruffPairInspection, ...],
    groups: Mapping[
        tuple[str, str],
        tuple[_SourceAcceptedMember | RruffRejectedMember, ...],
    ],
) -> Mapping[str, int]:
    counts: Counter[str] = Counter()
    counts["source_measurement_groups"] = len(groups)
    for members in groups.values():
        raw = sum(member.kind == "raw" for member in members)
        processed = len(members) - raw
        counts[f"source_multiplicity_{raw}_{processed}"] += 1
    for pair in pairs:
        counts[pair.pair_status] += 1
        if pair.axis_relation is not None:
            counts[pair.axis_relation] += 1
        if pair.axis_relation in {
            "exact_equal",
            "processed_exact_contiguous_subset_of_raw",
            "raw_exact_contiguous_subset_of_processed",
        }:
            counts["pointwise_comparable_pairs"] += 1
    return MappingProxyType(dict(counts))


def _require_pairing_statistics(
    observed: Mapping[str, int],
    expected: Mapping[str, int] | None,
) -> None:
    if expected is None:
        return
    for name, expected_value in expected.items():
        observed_value = observed.get(name, 0)
        if observed_value != expected_value:
            raise _fatal(
                f"pairing_statistics.{name}",
                f"expected {expected_value}, observed {observed_value}",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )


def _build_pairs(
    raw_root: Path,
    source_inspection: _RruffSourceInspection,
    expected_statistics: Mapping[str, int] | None,
) -> tuple[
    tuple[RruffPairInspection, ...],
    Mapping[tuple[str, str], str],
]:
    groups = _group_members(source_inspection)
    statuses: dict[tuple[str, str], str] = {}
    for key, members in groups.items():
        source_raw = sum(member.kind == "raw" for member in members)
        source_processed = len(members) - source_raw
        accepted_raw = sum(
            isinstance(member, _SourceAcceptedMember)
            and member.kind == "raw"
            for member in members
        )
        accepted_processed = sum(
            isinstance(member, _SourceAcceptedMember)
            and member.kind == "processed"
            for member in members
        )
        statuses[key] = _pair_status(
            source_raw=source_raw,
            source_processed=source_processed,
            accepted_raw=accepted_raw,
            accepted_processed=accepted_processed,
        )
    relations = _derive_relations(
        raw_root,
        groups,
        MappingProxyType(statuses),
    )

    pairs = []
    for archive, measurement_key in sorted(
        groups,
        key=lambda key: (
            key[0].encode("utf-8"),
            key[1].encode("utf-8"),
        ),
    ):
        members = groups[(archive, measurement_key)]
        raw_members = tuple(
            sorted(
                (
                    _pair_member(member)
                    for member in members
                    if member.kind == "raw"
                ),
                key=lambda member: (
                    member.member_id.encode("ascii"),
                    member.source_member.encode("utf-8"),
                ),
            )
        )
        processed_members = tuple(
            sorted(
                (
                    _pair_member(member)
                    for member in members
                    if member.kind == "processed"
                ),
                key=lambda member: (
                    member.member_id.encode("ascii"),
                    member.source_member.encode("utf-8"),
                ),
            )
        )
        mineral_names = tuple(
            sorted(
                {
                    member.mineral_name
                    for member in members
                    if member.mineral_name is not None
                },
                key=lambda value: value.encode("utf-8"),
            )
        )
        rruff_ids = tuple(
            sorted(
                {
                    member.rruff_id
                    for member in members
                    if member.rruff_id is not None
                },
                key=lambda value: value.encode("utf-8"),
            )
        )
        relation = relations.get((archive, measurement_key))
        pairs.append(
            RruffPairInspection(
                pair_id=_pair_id(archive, measurement_key),
                archive=archive,
                measurement_key=measurement_key,
                mineral_names=mineral_names,
                rruff_ids=rruff_ids,
                raw_members=raw_members,
                processed_members=processed_members,
                pair_status=statuses[(archive, measurement_key)],
                axis_relation=None if relation is None else relation[0],
                raw_slice=None if relation is None else relation[1],
                processed_slice=None if relation is None else relation[2],
            )
        )
    pair_tuple = tuple(pairs)
    _require_pairing_statistics(
        _pairing_statistics(pair_tuple, groups),
        expected_statistics,
    )
    return pair_tuple, MappingProxyType(statuses)


def _public_accepted_member(
    member: _SourceAcceptedMember,
    pair_status: str,
) -> RruffAcceptedMember:
    return RruffAcceptedMember(
        archive=member.archive,
        source_member=member.source_member,
        member_id=member.member_id,
        member_sha256=member.member_sha256,
        member_bytes=member.member_bytes,
        measurement_key=member.measurement_key,
        kind=member.kind,
        text_encoding=member.text_encoding,
        header_entries=member.header_entries,
        ignored_lines=member.ignored_lines,
        mineral_name=member.mineral_name,
        rruff_id=member.rruff_id,
        pin_id=member.pin_id,
        excitation_nm=member.excitation_nm,
        source_point_count=member.source_point_count,
        axis_id=member.axis_id,
        axis_float32_max_abs_error=member.axis_float32_max_abs_error,
        intensity_float32_max_abs_error=(
            member.intensity_float32_max_abs_error
        ),
        pair_id=_pair_id(member.archive, member.measurement_key),
        pair_status=pair_status,
    )


def _inspect_rruff_bundle(
    raw_root: Path,
    contract: _RruffSourceContract,
) -> _RruffInspectionBundle:
    raw_root = Path(raw_root)
    source_inspection = _inspect_rruff_source(raw_root, contract)
    pairs, statuses = _build_pairs(
        raw_root,
        source_inspection,
        contract.expected_pair_statistics,
    )
    accepted_members = tuple(
        _public_accepted_member(
            member,
            statuses[(member.archive, member.measurement_key)],
        )
        for member in source_inspection.accepted_members
    )
    inspection = RruffInspection(
        raw_root=raw_root,
        archive_sha256=source_inspection.archive_sha256,
        source_member_count=source_inspection.statistics.source_members,
        accepted_members=accepted_members,
        rejected_members=source_inspection.rejected_members,
        pairs=pairs,
        global_class_labels=source_inspection.global_class_labels,
        raw_axis_group_count=source_inspection.statistics.raw_axis_groups,
        processed_axis_group_count=(
            source_inspection.statistics.processed_axis_groups
        ),
    )
    return _RruffInspectionBundle(
        inspection=inspection,
        statistics=source_inspection.statistics,
    )


def _inspect_rruff(
    raw_root: Path,
    contract: _RruffSourceContract,
) -> RruffInspection:
    return _inspect_rruff_bundle(raw_root, contract).inspection


def inspect_rruff(raw_root: Path) -> RruffInspection:
    return _inspect_rruff(Path(raw_root), PRODUCTION_SOURCE_CONTRACT)


def _require_matching_inspection_root(
    raw_root: Path,
    inspection: RruffInspection,
) -> None:
    if inspection.raw_root != raw_root:
        raise _fatal(
            "inspection.raw_root",
            (
                f"expected {raw_root}, "
                f"observed {inspection.raw_root}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )


def _record_class_id(
    inspection: RruffInspection,
    mineral_name: str,
) -> int:
    matches = [
        class_id
        for class_id, name in inspection.global_class_labels.items()
        if name == mineral_name
    ]
    if len(matches) != 1:
        raise _fatal(
            f"class_labels.{mineral_name}",
            f"expected one class ID, observed {matches}",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    return matches[0]


def _archive_metadata(
    archive: str,
) -> tuple[str, str | None, str]:
    if archive == "LR-Raman.zip":
        return "long_range", None, "unoriented"
    oriented_suffix = "_oriented.zip"
    unoriented_suffix = "_unoriented.zip"
    if archive.endswith(oriented_suffix):
        return (
            "standard_range",
            archive[: -len(oriented_suffix)],
            "oriented",
        )
    if archive.endswith(unoriented_suffix):
        return (
            "standard_range",
            archive[: -len(unoriented_suffix)],
            "unoriented",
        )
    raise _fatal(
        f"records.archive.{archive}",
        "archive name does not map to RRUFF collection metadata",
        "SOURCE_MEMBER_COUNT_MISMATCH",
    )


def _source_metadata(
    *,
    member: RruffAcceptedMember,
    inspection: RruffInspection,
) -> dict[str, object]:
    archive_collection, quality_rating, orientation_collection = (
        _archive_metadata(member.archive)
    )
    source_filetype_values = _header_values(member.header_entries).get(
        "FILETYPE",
        (),
    )
    if len(source_filetype_values) != 1:
        raise _fatal(
            f"records.{member.source_member}.source_filetype",
            (
                "expected one promoted FILETYPE value, observed "
                f"{source_filetype_values}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    header_entries = [
        {
            "line": entry.line,
            "key": entry.key,
            "value": entry.value,
        }
        for entry in member.header_entries
    ]
    ignored_lines = [
        {
            "line": ignored.line,
            "category": ignored.category,
            "text": ignored.text,
        }
        for ignored in member.ignored_lines
    ]
    return {
        "source_dataset_id": SOURCE_DATASET_ID,
        "source_archive": member.archive,
        "source_archive_sha256": inspection.archive_sha256[member.archive],
        "source_member": member.source_member,
        "source_member_sha256": member.member_sha256,
        "source_member_bytes": member.member_bytes,
        "source_member_id": member.member_id,
        "source_measurement_key": member.measurement_key,
        "source_filetype": source_filetype_values[0],
        "source_text_encoding": member.text_encoding,
        "source_header_entries": header_entries,
        "source_ignored_lines": ignored_lines,
        "ignored_comment_line_count": sum(
            ignored.category == "comment"
            for ignored in member.ignored_lines
        ),
        "ignored_preamble_line_count": sum(
            ignored.category == "preamble"
            for ignored in member.ignored_lines
        ),
        "ignored_column_header_line_count": sum(
            ignored.category == "column_header"
            for ignored in member.ignored_lines
        ),
        "source_point_count": member.source_point_count,
        "source_axis_dtype": "float64",
        "stored_axis_dtype": "float32",
        "source_intensity_dtype": "float64",
        "stored_intensity_dtype": "float32",
        "archive_collection": archive_collection,
        "quality_rating": quality_rating,
        "orientation_collection": orientation_collection,
        "mineral_name": member.mineral_name,
        "rruff_id": member.rruff_id,
        "pin_id": member.pin_id,
        "pair_id": member.pair_id,
        "pair_status": member.pair_status,
    }


def _reparse_expected_record_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    member: RruffAcceptedMember,
):
    try:
        payload = archive.read(info)
    except (OSError, zipfile.BadZipFile) as error:
        raise _fatal(
            f"records.{member.source_member}.read",
            str(error),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error
    outcome = _parse_member_bytes(
        member.archive,
        member.source_member,
        payload,
    )
    parsed = outcome.accepted
    if parsed is None:
        raise _fatal(
            f"records.{member.source_member}",
            (
                "accepted member changed to rejection "
                f"{outcome.rejected.rejection_code}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    checks = {
        "member_sha256": (parsed.member_sha256, member.member_sha256),
        "source_point_count": (
            parsed.source_point_count,
            member.source_point_count,
        ),
        "axis_id": (parsed.axis_id, member.axis_id),
        "axis_float32_max_abs_error": (
            parsed.axis_float32_max_abs_error,
            member.axis_float32_max_abs_error,
        ),
        "intensity_float32_max_abs_error": (
            parsed.intensity_float32_max_abs_error,
            member.intensity_float32_max_abs_error,
        ),
    }
    for name, (observed, expected) in checks.items():
        if observed != expected:
            raise _fatal(
                f"records.{member.source_member}.{name}",
                f"expected {expected!r}, observed {observed!r}",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
    return parsed


def _required_record_info(
    infos: Mapping[str, zipfile.ZipInfo],
    member: RruffAcceptedMember,
) -> zipfile.ZipInfo:
    try:
        return infos[member.source_member]
    except KeyError:
        raise _fatal(
            f"records.{member.source_member}",
            "source member is missing during record reparse",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from None


def _record_from_parsed_member(
    parsed,
    member: RruffAcceptedMember,
    inspection: RruffInspection,
) -> RamanRecord:
    acquisition = _promote_rruff_acquisition(
        parsed.header_entries,
        parsed.ignored_lines,
    )
    if member.kind == "raw":
        dataset_id = RAW_DATASET_ID
        preprocessing_status = PreprocessingStatus.KNOWN_RAW
        preprocessing_steps: tuple[PreprocessingStep, ...] = ()
    else:
        dataset_id = PROCESSED_DATASET_ID
        preprocessing_status = PreprocessingStatus.UNKNOWN
        preprocessing_steps = (
            PreprocessingStep(
                operation="source_processed_state",
                description=(
                    "Source FILETYPE labels this spectrum Raman Processed; "
                    "the exact processing operations and parameters are "
                    "unavailable"
                ),
                evidence=(
                    f"{member.archive}/{member.source_member} "
                    "FILETYPE header"
                ),
            ),
        )
    record = RamanRecord(
        record_id=_record_id(member.kind, member.member_id),
        intensity=np.array(
            parsed.source_intensity,
            dtype=np.float32,
            copy=True,
        ),
        wavenumber=np.array(
            parsed.source_axis,
            dtype=np.float32,
            copy=True,
        ),
        meta=SpectrumMetadata(
            dataset_id=dataset_id,
            sample_id=member.rruff_id,
            instrument=acquisition.instrument,
            excitation_nm=acquisition.excitation_nm,
            integration_time_s=acquisition.integration_time_s,
            n_accumulations=acquisition.n_accumulations,
            grating=acquisition.grating,
            detector=acquisition.detector,
            preprocessing_status=preprocessing_status,
            preprocessing_steps=preprocessing_steps,
            source_metadata=_source_metadata(
                member=member,
                inspection=inspection,
            ),
        ),
        targets=Targets(
            class_label=_record_class_id(
                inspection,
                member.mineral_name,
            )
        ),
        provenance=Provenance(
            source_url=f"{_SOURCE_URL_PREFIX}{member.archive}",
            license=SOURCE_LICENSE,
            license_status=LicenseStatus.NOT_STATED,
            sha256=inspection.archive_sha256[member.archive],
            retrieved_date=RETRIEVED_DATE,
            source_artifact=member.archive,
        ),
    )
    return validate_record(record)


def _iter_rruff_records(
    raw_root: Path,
    inspection: RruffInspection,
    *,
    kind: str,
) -> Iterator[RamanRecord]:
    raw_root = Path(raw_root)
    _require_matching_inspection_root(raw_root, inspection)
    members_by_archive: dict[str, list[RruffAcceptedMember]] = defaultdict(list)
    for member in inspection.accepted_members:
        if member.kind not in {"raw", "processed"}:
            raise _fatal(
                f"records.{member.source_member}.kind",
                f"unexpected member kind {member.kind!r}",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        if member.kind == kind:
            if member.archive not in inspection.archive_sha256:
                raise _fatal(
                    (
                        f"records.{member.source_member}."
                        "source_archive_sha256"
                    ),
                    (
                        f"archive {member.archive!r} is missing from "
                        "inspection archive hashes"
                    ),
                    "SOURCE_MEMBER_COUNT_MISMATCH",
                )
            members_by_archive[member.archive].append(member)
    for archive_name in sorted(
        members_by_archive,
        key=lambda value: value.encode("utf-8"),
    ):
        with zipfile.ZipFile(raw_root / archive_name) as archive:
            infos = {
                info.filename: info
                for info in archive.infolist()
                if not info.is_dir()
            }
            for member in sorted(
                members_by_archive[archive_name],
                key=lambda value: value.source_member.encode("utf-8"),
            ):
                info = _required_record_info(infos, member)
                parsed = _reparse_expected_record_member(
                    archive,
                    info,
                    member,
                )
                yield _record_from_parsed_member(
                    parsed,
                    member,
                    inspection,
                )
                del parsed


def iter_rruff_raw_records(
    raw_root: Path,
    inspection: RruffInspection | None = None,
) -> Iterator[RamanRecord]:
    raw_root = Path(raw_root)
    if inspection is None:
        inspection = inspect_rruff(raw_root)
    yield from _iter_rruff_records(
        raw_root,
        inspection,
        kind="raw",
    )


def iter_rruff_processed_records(
    raw_root: Path,
    inspection: RruffInspection | None = None,
) -> Iterator[RamanRecord]:
    raw_root = Path(raw_root)
    if inspection is None:
        inspection = inspect_rruff(raw_root)
    yield from _iter_rruff_records(
        raw_root,
        inspection,
        kind="processed",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise _fatal(
            str(path),
            f"cannot read file: {error}",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error
    return digest.hexdigest()


def _dataset_file_summary(
    dataset_path: Path,
) -> Mapping[str, Mapping[str, int | str]]:
    files = {}
    for name in DATASET_FILES:
        path = dataset_path / name
        files[name] = MappingProxyType(
            {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return MappingProxyType(files)


def _dataset_summary(
    dataset_path: Path,
    *,
    dataset_id: str,
    eligible_records: int,
) -> RruffDatasetSummary:
    validation = validate_dataset(dataset_path)
    files = _dataset_file_summary(dataset_path)
    return RruffDatasetSummary(
        dataset_id=dataset_id,
        record_count=validation.record_count,
        eligible_records=eligible_records,
        axis_group_count=validation.axis_group_count,
        output_bytes=sum(
            int(details["bytes"]) for details in files.values()
        ),
        files=files,
    )


def _observed_class_labels(
    inspection: RruffInspection,
    *,
    kind: str,
) -> Mapping[int, str]:
    names = {
        member.mineral_name
        for member in inspection.accepted_members
        if member.kind == kind
    }
    labels = {
        class_id: name
        for class_id, name in inspection.global_class_labels.items()
        if name in names
    }
    if len(labels) != len(names):
        raise _fatal(
            f"class_labels.{kind}",
            "global class mapping does not cover observed mineral names",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    return MappingProxyType(labels)


def _read_json(path: Path, error_path: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _fatal(
            error_path,
            str(error),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error
    if not isinstance(value, Mapping):
        raise _fatal(
            error_path,
            "must be a JSON object",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    return value


def _read_jsonl(
    path: Path,
    error_path: str,
) -> list[Mapping[str, object]]:
    try:
        lines = path.read_bytes().splitlines()
    except OSError as error:
        raise _fatal(
            error_path,
            str(error),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error
    records = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _fatal(
                f"{error_path}[{index}]",
                str(error),
                "SOURCE_MEMBER_COUNT_MISMATCH",
            ) from error
        if not isinstance(value, Mapping):
            raise _fatal(
                f"{error_path}[{index}]",
                "must be a JSON object",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        records.append(value)
    return records


def _comparison_error(
    dataset_id: str,
    suffix: str,
    reason: str,
) -> RruffValidationError:
    return _fatal(
        f"{dataset_id}.{suffix}",
        reason,
        "SOURCE_MEMBER_COUNT_MISMATCH",
    )


def _expected_record_for_member(
    raw_root: Path,
    inspection: RruffInspection,
    member: RruffAcceptedMember,
) -> RamanRecord:
    archive_path = raw_root / member.archive
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = {
                info.filename: info
                for info in archive.infolist()
                if not info.is_dir()
            }
            info = _required_record_info(infos, member)
            parsed = _reparse_expected_record_member(
                archive,
                info,
                member,
            )
    except RruffValidationError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise _fatal(
            f"records.{member.source_member}.read",
            str(error),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error
    return _record_from_parsed_member(parsed, member, inspection)


def _compare_record(
    *,
    dataset_id: str,
    record_index: int,
    observed: RamanRecord,
    expected: RamanRecord,
) -> tuple[int, int, int]:
    record_path = f"records.jsonl[{record_index}]"
    if observed.record_id != expected.record_id:
        raise _comparison_error(
            dataset_id,
            f"{record_path}.record_id",
            (
                f"expected {expected.record_id!r}, "
                f"observed {observed.record_id!r}"
            ),
        )
    if observed.targets.class_label != expected.targets.class_label:
        raise _comparison_error(
            dataset_id,
            f"{record_path}.targets.class_label",
            (
                f"expected {expected.targets.class_label!r}, "
                f"observed {observed.targets.class_label!r}"
            ),
        )
    if observed.meta.preprocessing_status is not expected.meta.preprocessing_status:
        raise _comparison_error(
            dataset_id,
            f"{record_path}.meta.preprocessing_status",
            (
                f"expected {expected.meta.preprocessing_status.value!r}, "
                f"observed {observed.meta.preprocessing_status.value!r}"
            ),
        )
    expected_source_metadata = expected.meta.source_metadata
    observed_source_metadata = observed.meta.source_metadata
    for key in (
        "source_member",
        "pair_id",
        "pair_status",
    ):
        if observed_source_metadata.get(key) != expected_source_metadata.get(key):
            raise _comparison_error(
                dataset_id,
                f"{record_path}.meta.source_metadata.{key}",
                (
                    f"expected {expected_source_metadata.get(key)!r}, "
                    f"observed {observed_source_metadata.get(key)!r}"
                ),
            )
    if observed.meta != expected.meta:
        raise _comparison_error(
            dataset_id,
            f"{record_path}.meta",
            "stored metadata differs from source mapping",
        )
    if observed.targets != expected.targets:
        raise _comparison_error(
            dataset_id,
            f"{record_path}.targets",
            "stored targets differ from source mapping",
        )
    if observed.provenance != expected.provenance:
        raise _comparison_error(
            dataset_id,
            f"{record_path}.provenance",
            "stored provenance differs from source mapping",
        )
    intensity_mismatch = int(
        not np.array_equal(observed.intensity, expected.intensity)
    )
    if intensity_mismatch:
        raise _comparison_error(
            dataset_id,
            "arrays.h5.intensity",
            f"stored record {observed.record_id} differs from source",
        )
    if not np.array_equal(observed.wavenumber, expected.wavenumber):
        raise _comparison_error(
            dataset_id,
            "arrays.h5.wavenumber",
            f"stored record {observed.record_id} axis differs from source",
        )
    eligibility_violation = int(
        observed.meta.eligible_for_preprocessing_evaluation
        != expected.meta.eligible_for_preprocessing_evaluation
    )
    return intensity_mismatch, 0, eligibility_violation


def _compare_rruff_dataset(
    raw_root: Path,
    dataset_path: Path,
    inspection: RruffInspection,
    *,
    kind: str,
) -> _RruffComparisonSummary:
    raw_root = Path(raw_root)
    dataset_path = Path(dataset_path)
    _require_matching_inspection_root(raw_root, inspection)
    if kind == "raw":
        dataset_id = RAW_DATASET_ID
    elif kind == "processed":
        dataset_id = PROCESSED_DATASET_ID
    else:
        raise _comparison_error(
            str(dataset_path.name),
            "kind",
            f"unexpected kind {kind!r}",
        )
    if dataset_path.name != dataset_id:
        raise _comparison_error(
            dataset_id,
            "path",
            "dataset basename does not match dataset ID",
        )

    expected_class_labels = _observed_class_labels(
        inspection,
        kind=kind,
    )
    manifest = _read_json(
        dataset_path / "dataset.json",
        f"{dataset_id}.dataset.json",
    )
    observed_class_labels = manifest.get("class_labels")
    expected_manifest_labels = {
        str(label): name
        for label, name in expected_class_labels.items()
    }
    if observed_class_labels != expected_manifest_labels:
        raise _comparison_error(
            dataset_id,
            "dataset.json.class_labels",
            (
                f"expected {expected_manifest_labels!r}, "
                f"observed {observed_class_labels!r}"
            ),
        )
    records = _read_jsonl(
        dataset_path / "records.jsonl",
        f"{dataset_id}.records.jsonl",
    )
    members_by_record_id = {
        _record_id(member.kind, member.member_id): member
        for member in inspection.accepted_members
        if member.kind == kind
    }
    if len(records) != len(members_by_record_id):
        raise _comparison_error(
            dataset_id,
            "records.jsonl",
            (
                f"expected {len(members_by_record_id)} records, "
                f"observed {len(records)}"
            ),
        )

    expected_members: list[tuple[str, RruffAcceptedMember]] = []
    for index, record_document in enumerate(records):
        record_path = f"records.jsonl[{index}]"
        record_id = record_document.get("record_id")
        if not isinstance(record_id, str) or record_id not in members_by_record_id:
            raise _comparison_error(
                dataset_id,
                f"{record_path}.record_id",
                f"unexpected record ID {record_id!r}",
            )
        member = members_by_record_id[record_id]
        try:
            class_label = record_document["targets"]["class_label"]
            preprocessing_status = record_document["meta"][
                "preprocessing_status"
            ]
            source_metadata = record_document["meta"]["source_metadata"]
            source_member = source_metadata["source_member"]
            pair_id = source_metadata["pair_id"]
            pair_status = source_metadata["pair_status"]
        except (KeyError, TypeError) as error:
            raise _comparison_error(
                dataset_id,
                record_path,
                f"missing or invalid record field: {error}",
            ) from error
        expected_class_label = _record_class_id(
            inspection,
            member.mineral_name,
        )
        if class_label != expected_class_label:
            raise _comparison_error(
                dataset_id,
                f"{record_path}.targets.class_label",
                (
                    f"expected {expected_class_label!r}, "
                    f"observed {class_label!r}"
                ),
            )
        expected_status = (
            PreprocessingStatus.KNOWN_RAW.value
            if member.kind == "raw"
            else PreprocessingStatus.UNKNOWN.value
        )
        if (
            preprocessing_status
            != expected_status
        ):
            raise _comparison_error(
                dataset_id,
                f"{record_path}.meta.preprocessing_status",
                (
                    "expected "
                    f"{expected_status!r}, "
                    f"observed {preprocessing_status!r}"
                ),
            )
        for name, observed_value, expected_value in (
            (
                "source_member",
                source_member,
                member.source_member,
            ),
            ("pair_id", pair_id, member.pair_id),
            (
                "pair_status",
                pair_status,
                member.pair_status,
            ),
        ):
            if observed_value != expected_value:
                raise _comparison_error(
                    dataset_id,
                    f"{record_path}.meta.source_metadata.{name}",
                    (
                        f"expected {expected_value!r}, "
                        f"observed {observed_value!r}"
                    ),
                )
        expected_members.append((record_id, member))

    intensity_mismatches = 0
    class_label_mismatches = 0
    eligibility_violations = 0
    with UnifiedDataset.open(
        dataset_path,
        verify_checksums=False,
    ) as dataset:
        for index, (record_id, member) in enumerate(expected_members):
            expected_record = _expected_record_for_member(
                raw_root,
                inspection,
                member,
            )
            try:
                observed_record = dataset.get(record_id)
            except (KeyError, TypeError, ValueError) as error:
                raise _comparison_error(
                    dataset_id,
                    f"records.jsonl[{index}]",
                    str(error),
                ) from error
            intensity, labels, eligibility = _compare_record(
                dataset_id=dataset_id,
                record_index=index,
                observed=observed_record,
                expected=expected_record,
            )
            intensity_mismatches += intensity
            class_label_mismatches += labels
            eligibility_violations += eligibility
            del expected_record
            del observed_record
    return _RruffComparisonSummary(
        source_rows_compared=len(records),
        intensity_mismatches=intensity_mismatches,
        class_label_mismatches=class_label_mismatches,
        eligibility_violations=eligibility_violations,
    )


def _build_rruff_data_staged(
    raw_root: Path,
    staging_parent: Path,
    bundle: _RruffInspectionBundle,
    *,
    timings: dict[str, float] | None = None,
) -> _RruffStagedData:
    raw_root = Path(raw_root)
    staging_parent = Path(staging_parent)
    inspection = bundle.inspection
    _require_matching_inspection_root(raw_root, inspection)
    bundle_checks = {
        "source_members": (
            bundle.statistics.source_members,
            inspection.source_member_count,
        ),
        "accepted_records": (
            bundle.statistics.accepted_records,
            len(inspection.accepted_members),
        ),
        "rejected_records": (
            bundle.statistics.rejected_records,
            len(inspection.rejected_members),
        ),
        "accepted_raw": (
            bundle.statistics.accepted_raw,
            sum(
                member.kind == "raw"
                for member in inspection.accepted_members
            ),
        ),
        "accepted_processed": (
            bundle.statistics.accepted_processed,
            sum(
                member.kind == "processed"
                for member in inspection.accepted_members
            ),
        ),
        "raw_axis_groups": (
            bundle.statistics.raw_axis_groups,
            inspection.raw_axis_group_count,
        ),
        "processed_axis_groups": (
            bundle.statistics.processed_axis_groups,
            inspection.processed_axis_group_count,
        ),
    }
    for name, (observed, expected) in bundle_checks.items():
        if observed != expected:
            raise _fatal(
                f"bundle.statistics.{name}",
                f"expected {expected}, observed {observed}",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
    if not staging_parent.is_dir():
        raise _fatal(
            "staging_parent",
            "must be an existing directory",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    raw_path = staging_parent / RAW_DATASET_ID
    processed_path = staging_parent / PROCESSED_DATASET_ID
    pair_index_path = staging_parent / _PAIR_INDEX_NAME
    for path in (raw_path, processed_path, pair_index_path):
        if path.exists():
            raise _fatal(
                f"staging_parent/{path.name}",
                "staged artifact already exists",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )

    validation_seconds = 0.0
    raw_started = perf_counter()
    raw_records = list(
        iter_rruff_raw_records(raw_root, inspection)
    )
    write_dataset(
        raw_records,
        raw_path,
        dataset_id=RAW_DATASET_ID,
        class_labels=_observed_class_labels(inspection, kind="raw"),
        overwrite=False,
    )
    del raw_records
    raw_build_seconds = perf_counter() - raw_started
    validation_started = perf_counter()
    raw_comparison = _compare_rruff_dataset(
        raw_root,
        raw_path,
        inspection,
        kind="raw",
    )
    raw_summary = _dataset_summary(
        raw_path,
        dataset_id=RAW_DATASET_ID,
        eligible_records=raw_comparison.source_rows_compared,
    )
    validation_seconds += perf_counter() - validation_started

    processed_started = perf_counter()
    processed_records = list(
        iter_rruff_processed_records(raw_root, inspection)
    )
    write_dataset(
        processed_records,
        processed_path,
        dataset_id=PROCESSED_DATASET_ID,
        class_labels=_observed_class_labels(
            inspection,
            kind="processed",
        ),
        overwrite=False,
    )
    del processed_records
    processed_build_seconds = perf_counter() - processed_started
    validation_started = perf_counter()
    processed_comparison = _compare_rruff_dataset(
        raw_root,
        processed_path,
        inspection,
        kind="processed",
    )
    processed_summary = _dataset_summary(
        processed_path,
        dataset_id=PROCESSED_DATASET_ID,
        eligible_records=0,
    )
    validation_seconds += perf_counter() - validation_started

    pair_started = perf_counter()
    _write_rruff_pair_index(pair_index_path, inspection)
    pair_index_seconds = perf_counter() - pair_started
    with UnifiedDataset.open(raw_path) as raw_dataset:
        raw_record_ids = set(raw_dataset.record_ids)
    with UnifiedDataset.open(processed_path) as processed_dataset:
        processed_record_ids = set(processed_dataset.record_ids)
    validation_started = perf_counter()
    pair_summary = _validate_rruff_pair_index(
        pair_index_path,
        inspection,
        raw_record_ids=raw_record_ids,
        processed_record_ids=processed_record_ids,
    )
    validation_seconds += perf_counter() - validation_started

    if timings is not None:
        timings.update(
            {
                "raw_build_seconds": raw_build_seconds,
                "processed_build_seconds": processed_build_seconds,
                "pair_index_seconds": pair_index_seconds,
                "validation_seconds": validation_seconds,
            }
        )
    pair_status_counts = MappingProxyType(
        {
            status: int(pair_summary[status])
            for status in _PAIR_STATUSES
        }
    )
    axis_relation_counts = MappingProxyType(
        {
            relation: int(pair_summary[relation])
            for relation in _AXIS_RELATIONS
        }
    )
    return _RruffStagedData(
        raw=raw_summary,
        processed=processed_summary,
        pair_index_path=pair_index_path,
        pair_index_sha256=_sha256_file(pair_index_path),
        pair_index_rows=int(pair_summary["rows"]),
        pair_status_counts=pair_status_counts,
        axis_relation_counts=axis_relation_counts,
        raw_comparison=raw_comparison,
        processed_comparison=processed_comparison,
    )


def _source_archive_documents(
    raw_root: Path,
    inspection: RruffInspection,
) -> list[dict[str, object]]:
    source_counts: Counter[tuple[str, str]] = Counter()
    for member in inspection.accepted_members:
        source_counts[(member.archive, member.kind)] += 1
    for member in inspection.rejected_members:
        source_counts[(member.archive, member.kind)] += 1

    archives = []
    for archive_name in sorted(
        inspection.archive_sha256,
        key=lambda value: value.encode("utf-8"),
    ):
        archive_path = raw_root / archive_name
        try:
            archive_bytes = archive_path.stat().st_size
        except OSError as error:
            raise _fatal(
                f"receipt.source_archives.{archive_name}.bytes",
                str(error),
                "SOURCE_MEMBER_COUNT_MISMATCH",
            ) from error
        observed_sha256 = _sha256_file(archive_path)
        expected_sha256 = inspection.archive_sha256[archive_name]
        if observed_sha256 != expected_sha256:
            raise _fatal(
                f"receipt.source_archives.{archive_name}.sha256",
                (
                    f"expected {expected_sha256!r}, "
                    f"observed {observed_sha256!r}"
                ),
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        raw_members = source_counts[(archive_name, "raw")]
        processed_members = source_counts[(archive_name, "processed")]
        archives.append(
            {
                "archive": archive_name,
                "url": f"{_SOURCE_URL_PREFIX}{archive_name}",
                "bytes": archive_bytes,
                "sha256": observed_sha256,
                "retrieved_date": RETRIEVED_DATE.isoformat(),
                "member_count": raw_members + processed_members,
                "raw_members": raw_members,
                "processed_members": processed_members,
            }
        )
    return archives


def _statistics_document(
    statistics: _RruffSourceStatistics,
    fields: tuple[str, ...],
) -> dict[str, int]:
    return {
        field: int(getattr(statistics, field))
        for field in fields
    }


def _rejected_member_documents(
    inspection: RruffInspection,
) -> list[dict[str, object]]:
    return [
        {
            "archive": member.archive,
            "source_member": member.source_member,
            "member_id": member.member_id,
            "member_sha256": member.member_sha256,
            "member_bytes": member.member_bytes,
            "measurement_key": member.measurement_key,
            "kind": member.kind,
            "rejection_code": member.rejection_code,
            "reason": member.reason,
        }
        for member in sorted(
            inspection.rejected_members,
            key=lambda value: (
                value.archive.encode("utf-8"),
                value.source_member.encode("utf-8"),
            ),
        )
    ]


def _dataset_receipt_document(
    summary: RruffDatasetSummary,
    inspection: RruffInspection,
    *,
    kind: str,
) -> dict[str, object]:
    return {
        "records": summary.record_count,
        "eligible_records": summary.eligible_records,
        "observed_class_ids": sorted(
            _observed_class_labels(inspection, kind=kind)
        ),
        "axis_groups": summary.axis_group_count,
    }


def _dataset_output_document(
    summary: RruffDatasetSummary,
) -> dict[str, object]:
    return {
        "artifact_type": "dataset",
        "files": {
            name: dict(details)
            for name, details in summary.files.items()
        },
    }


def _receipt_document(
    raw_root: Path,
    bundle: _RruffInspectionBundle,
    staged: _RruffStagedData,
) -> dict[str, object]:
    raw_root = Path(raw_root)
    inspection = bundle.inspection
    statistics = bundle.statistics
    _require_matching_inspection_root(raw_root, inspection)
    return {
        "adapter_version": ADAPTER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "source_archives": _source_archive_documents(
            raw_root,
            inspection,
        ),
        "source_inventory": _statistics_document(
            statistics,
            _SOURCE_INVENTORY_FIELDS,
        ),
        "parser": _statistics_document(
            statistics,
            _PARSER_FIELDS,
        ),
        "rejected_members": _rejected_member_documents(inspection),
        "global_class_labels": {
            str(class_id): mineral_name
            for class_id, mineral_name in (
                inspection.global_class_labels.items()
            )
        },
        "datasets": {
            RAW_DATASET_ID: _dataset_receipt_document(
                staged.raw,
                inspection,
                kind="raw",
            ),
            PROCESSED_DATASET_ID: _dataset_receipt_document(
                staged.processed,
                inspection,
                kind="processed",
            ),
        },
        "pairing": {
            "index_file": _PAIR_INDEX_NAME,
            "index_rows": staged.pair_index_rows,
            "status_counts": dict(staged.pair_status_counts),
            "axis_relation_counts": dict(
                staged.axis_relation_counts
            ),
            "pointwise_comparable_pairs": sum(
                staged.axis_relation_counts[relation]
                for relation in _POINTWISE_RELATIONS
            ),
        },
        "precision": {
            "axis_float32_max_abs_error_cm1": (
                statistics.axis_float32_max_abs_error_cm1
            ),
            "intensity_float32_max_abs_error": (
                statistics.intensity_float32_max_abs_error
            ),
        },
        "license": {
            "text": SOURCE_LICENSE,
            "status": LicenseStatus.NOT_STATED.value,
            "access_status": "free public access",
            "redistribution_requires_review": True,
        },
        "limitations": {
            "processed_is_not_clean_target": True,
            "processed_operations_unknown": True,
            "rejected_source_members": statistics.rejected_records,
            "alignment_required_pairs": staged.axis_relation_counts[
                "overlap_requires_alignment"
            ],
            "no_overlap_pairs": staged.axis_relation_counts[
                "no_axis_overlap"
            ],
            "taxonomy_normalization_applied": False,
            "resampling_applied": False,
        },
        "outputs": {
            RAW_DATASET_ID: _dataset_output_document(staged.raw),
            PROCESSED_DATASET_ID: _dataset_output_document(
                staged.processed
            ),
            _PAIR_INDEX_NAME: {
                "artifact_type": "file",
                "files": {
                    _PAIR_INDEX_NAME: {
                        "bytes": staged.pair_index_path.stat().st_size,
                        "sha256": staged.pair_index_sha256,
                    }
                },
            },
        },
    }


def _read_canonical_receipt(path: Path) -> Mapping[str, object]:
    try:
        raw = Path(path).read_bytes()
        receipt = json.loads(
            raw,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise _fatal(
            "receipt",
            str(error),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error
    if not isinstance(receipt, Mapping):
        raise _fatal(
            "receipt",
            "must be a JSON object",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    try:
        canonical = _canonical_json_bytes(receipt)
    except (TypeError, ValueError) as error:
        raise _fatal(
            "receipt",
            str(error),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error
    if raw != canonical:
        raise _fatal(
            "receipt",
            "must use canonical JSON encoding",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    return receipt


def _require_receipt_keys(
    path: str,
    value: object,
    expected: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _fatal(
            path,
            "must be a JSON object",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    actual = set(value)
    if actual != expected:
        raise _fatal(
            path,
            (
                f"key mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    return value


def _require_receipt_list(
    path: str,
    value: object,
) -> list[object]:
    if not isinstance(value, list):
        raise _fatal(
            path,
            "must be a JSON array",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    return value


def _require_receipt_equal(
    path: str,
    observed: object,
    expected: object,
) -> None:
    if isinstance(expected, bool):
        valid_type = isinstance(observed, bool)
    elif isinstance(expected, int):
        valid_type = isinstance(observed, int) and not isinstance(
            observed,
            bool,
        )
    elif isinstance(expected, float):
        valid_type = (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and math.isfinite(observed)
        )
    elif isinstance(expected, str):
        valid_type = isinstance(observed, str)
    elif expected is None:
        valid_type = observed is None
    else:
        raise TypeError(
            f"unsupported receipt scalar expectation {type(expected)!r}"
        )
    if not valid_type:
        raise _fatal(
            path,
            f"has invalid type {type(observed).__name__}",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    if observed != expected:
        raise _fatal(
            path,
            f"expected {expected!r}, observed {observed!r}",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )


def _validate_scalar_mapping(
    path: str,
    observed: object,
    expected: Mapping[str, object],
) -> Mapping[str, object]:
    mapping = _require_receipt_keys(
        path,
        observed,
        set(expected),
    )
    for name, expected_value in expected.items():
        _require_receipt_equal(
            f"{path}.{name}",
            mapping[name],
            expected_value,
        )
    return mapping


def _expected_dataset_receipt(
    bundle: _RruffInspectionBundle,
    *,
    kind: str,
) -> dict[str, object]:
    statistics = bundle.statistics
    if kind == "raw":
        return {
            "records": statistics.accepted_raw,
            "eligible_records": statistics.accepted_raw,
            "observed_class_ids": sorted(
                _observed_class_labels(
                    bundle.inspection,
                    kind="raw",
                )
            ),
            "axis_groups": statistics.raw_axis_groups,
        }
    if kind == "processed":
        return {
            "records": statistics.accepted_processed,
            "eligible_records": 0,
            "observed_class_ids": sorted(
                _observed_class_labels(
                    bundle.inspection,
                    kind="processed",
                )
            ),
            "axis_groups": statistics.processed_axis_groups,
        }
    raise ValueError(f"unsupported RRUFF kind {kind!r}")


def _expected_pairing_receipt(
    inspection: RruffInspection,
) -> dict[str, object]:
    status_counts = {
        status: sum(
            pair.pair_status == status
            for pair in inspection.pairs
        )
        for status in _PAIR_STATUSES
    }
    relation_counts = {
        relation: sum(
            pair.axis_relation == relation
            for pair in inspection.pairs
        )
        for relation in _AXIS_RELATIONS
    }
    return {
        "index_file": _PAIR_INDEX_NAME,
        "index_rows": len(inspection.pairs),
        "status_counts": status_counts,
        "axis_relation_counts": relation_counts,
        "pointwise_comparable_pairs": sum(
            relation_counts[relation]
            for relation in _POINTWISE_RELATIONS
        ),
    }


def _validate_receipt_dataset(
    path: str,
    observed: object,
    expected: Mapping[str, object],
) -> None:
    dataset = _require_receipt_keys(
        path,
        observed,
        _DATASET_RECEIPT_KEYS,
    )
    for name in ("records", "eligible_records", "axis_groups"):
        _require_receipt_equal(
            f"{path}.{name}",
            dataset[name],
            expected[name],
        )
    observed_ids = _require_receipt_list(
        f"{path}.observed_class_ids",
        dataset["observed_class_ids"],
    )
    expected_ids = expected["observed_class_ids"]
    if len(observed_ids) != len(expected_ids):
        raise _fatal(
            f"{path}.observed_class_ids",
            (
                f"expected {len(expected_ids)} IDs, "
                f"observed {len(observed_ids)}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    for index, (observed_id, expected_id) in enumerate(
        zip(observed_ids, expected_ids)
    ):
        _require_receipt_equal(
            f"{path}.observed_class_ids[{index}]",
            observed_id,
            expected_id,
        )


def _validate_receipt_pairing(
    observed: object,
    expected: Mapping[str, object],
) -> None:
    path = "receipt.pairing"
    pairing = _require_receipt_keys(
        path,
        observed,
        _PAIRING_RECEIPT_KEYS,
    )
    for name in (
        "index_file",
        "index_rows",
        "pointwise_comparable_pairs",
    ):
        _require_receipt_equal(
            f"{path}.{name}",
            pairing[name],
            expected[name],
        )
    _validate_scalar_mapping(
        f"{path}.status_counts",
        pairing["status_counts"],
        expected["status_counts"],
    )
    _validate_scalar_mapping(
        f"{path}.axis_relation_counts",
        pairing["axis_relation_counts"],
        expected["axis_relation_counts"],
    )


def _validate_receipt_outputs(
    receipt_path: Path,
    observed: object,
) -> None:
    outputs = _require_receipt_keys(
        "receipt.outputs",
        observed,
        _OUTPUT_KEYS,
    )
    for artifact_name in (
        RAW_DATASET_ID,
        PROCESSED_DATASET_ID,
        _PAIR_INDEX_NAME,
    ):
        artifact_path = f"receipt.outputs.{artifact_name}"
        output = _require_receipt_keys(
            artifact_path,
            outputs[artifact_name],
            _OUTPUT_ARTIFACT_KEYS,
        )
        expected_type = (
            "file"
            if artifact_name == _PAIR_INDEX_NAME
            else "dataset"
        )
        _require_receipt_equal(
            f"{artifact_path}.artifact_type",
            output["artifact_type"],
            expected_type,
        )
        files = output["files"]
        expected_files = (
            {_PAIR_INDEX_NAME}
            if expected_type == "file"
            else set(DATASET_FILES)
        )
        file_mapping = _require_receipt_keys(
            f"{artifact_path}.files",
            files,
            expected_files,
        )
        for name in sorted(
            expected_files,
            key=lambda value: value.encode("utf-8"),
        ):
            details_path = f"{artifact_path}.files.{name}"
            details = _require_receipt_keys(
                details_path,
                file_mapping[name],
                _OUTPUT_FILE_KEYS,
            )
            output_path = (
                receipt_path.parent / name
                if expected_type == "file"
                else receipt_path.parent / artifact_name / name
            )
            try:
                output_bytes = output_path.stat().st_size
            except OSError as error:
                raise _fatal(
                    f"{details_path}.bytes",
                    str(error),
                    "SOURCE_MEMBER_COUNT_MISMATCH",
                ) from error
            _require_receipt_equal(
                f"{details_path}.bytes",
                details["bytes"],
                output_bytes,
            )
            _require_receipt_equal(
                f"{details_path}.sha256",
                details["sha256"],
                _sha256_file(output_path),
            )


def _validate_rruff_receipt(
    path: Path,
    bundle: _RruffInspectionBundle,
) -> Mapping[str, object]:
    path = Path(path)
    receipt = _read_canonical_receipt(path)
    _require_receipt_keys("receipt", receipt, _RECEIPT_KEYS)
    _require_receipt_equal(
        "receipt.adapter_version",
        receipt["adapter_version"],
        ADAPTER_VERSION,
    )
    _require_receipt_equal(
        "receipt.schema_version",
        receipt["schema_version"],
        SCHEMA_VERSION,
    )

    source_archives = _require_receipt_list(
        "receipt.source_archives",
        receipt["source_archives"],
    )
    expected_archives = _source_archive_documents(
        bundle.inspection.raw_root,
        bundle.inspection,
    )
    if len(source_archives) != len(expected_archives):
        raise _fatal(
            "receipt.source_archives",
            (
                f"expected {len(expected_archives)} archives, "
                f"observed {len(source_archives)}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    for index, (archive, expected) in enumerate(
        zip(source_archives, expected_archives)
    ):
        _validate_scalar_mapping(
            f"receipt.source_archives[{index}]",
            archive,
            expected,
        )

    _validate_scalar_mapping(
        "receipt.source_inventory",
        receipt["source_inventory"],
        _statistics_document(
            bundle.statistics,
            _SOURCE_INVENTORY_FIELDS,
        ),
    )
    _validate_scalar_mapping(
        "receipt.parser",
        receipt["parser"],
        _statistics_document(bundle.statistics, _PARSER_FIELDS),
    )
    if (
        bundle.statistics.comma_numeric_rows
        + bundle.statistics.whitespace_numeric_rows
        != bundle.statistics.all_parsed_numeric_rows
    ):
        raise _fatal(
            "receipt.parser",
            "delimiter counts do not cover all parsed numeric rows",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )

    rejected_members = _require_receipt_list(
        "receipt.rejected_members",
        receipt["rejected_members"],
    )
    expected_rejected = _rejected_member_documents(
        bundle.inspection
    )
    if len(rejected_members) != len(expected_rejected):
        raise _fatal(
            "receipt.rejected_members",
            (
                f"expected {len(expected_rejected)} members, "
                f"observed {len(rejected_members)}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    for index, (member, expected) in enumerate(
        zip(rejected_members, expected_rejected)
    ):
        member_path = f"receipt.rejected_members[{index}]"
        parsed_member = _require_receipt_keys(
            member_path,
            member,
            _REJECTED_MEMBER_KEYS,
        )
        for name, expected_value in expected.items():
            _require_receipt_equal(
                f"{member_path}.{name}",
                parsed_member[name],
                expected_value,
            )

    labels = receipt["global_class_labels"]
    expected_labels = {
        str(class_id): mineral_name
        for class_id, mineral_name in (
            bundle.inspection.global_class_labels.items()
        )
    }
    parsed_labels = _require_receipt_keys(
        "receipt.global_class_labels",
        labels,
        set(expected_labels),
    )
    expected_label_keys = {
        str(class_id)
        for class_id in range(len(expected_labels))
    }
    if set(parsed_labels) != expected_label_keys:
        raise _fatal(
            "receipt.global_class_labels",
            "class IDs must be contiguous decimal strings from zero",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    for class_id, mineral_name in expected_labels.items():
        _require_receipt_equal(
            f"receipt.global_class_labels.{class_id}",
            parsed_labels[class_id],
            mineral_name,
        )

    datasets = _require_receipt_keys(
        "receipt.datasets",
        receipt["datasets"],
        {RAW_DATASET_ID, PROCESSED_DATASET_ID},
    )
    _validate_receipt_dataset(
        f"receipt.datasets.{RAW_DATASET_ID}",
        datasets[RAW_DATASET_ID],
        _expected_dataset_receipt(bundle, kind="raw"),
    )
    _validate_receipt_dataset(
        f"receipt.datasets.{PROCESSED_DATASET_ID}",
        datasets[PROCESSED_DATASET_ID],
        _expected_dataset_receipt(bundle, kind="processed"),
    )
    _validate_receipt_pairing(
        receipt["pairing"],
        _expected_pairing_receipt(bundle.inspection),
    )
    _validate_scalar_mapping(
        "receipt.precision",
        receipt["precision"],
        {
            "axis_float32_max_abs_error_cm1": (
                bundle.statistics.axis_float32_max_abs_error_cm1
            ),
            "intensity_float32_max_abs_error": (
                bundle.statistics.intensity_float32_max_abs_error
            ),
        },
    )
    _validate_scalar_mapping(
        "receipt.license",
        receipt["license"],
        {
            "text": SOURCE_LICENSE,
            "status": LicenseStatus.NOT_STATED.value,
            "access_status": "free public access",
            "redistribution_requires_review": True,
        },
    )
    expected_pairing = _expected_pairing_receipt(
        bundle.inspection
    )
    _validate_scalar_mapping(
        "receipt.limitations",
        receipt["limitations"],
        {
            "processed_is_not_clean_target": True,
            "processed_operations_unknown": True,
            "rejected_source_members": (
                bundle.statistics.rejected_records
            ),
            "alignment_required_pairs": expected_pairing[
                "axis_relation_counts"
            ]["overlap_requires_alignment"],
            "no_overlap_pairs": expected_pairing[
                "axis_relation_counts"
            ]["no_axis_overlap"],
            "taxonomy_normalization_applied": False,
            "resampling_applied": False,
        },
    )
    _validate_receipt_outputs(path, receipt["outputs"])
    return receipt


def _build_rruff_staged(
    raw_root: Path,
    staging_parent: Path,
    bundle: _RruffInspectionBundle,
    *,
    timings: dict[str, float] | None = None,
) -> tuple[
    RruffConversionSummary,
    _RruffComparisonSummary,
    _RruffComparisonSummary,
]:
    raw_root = Path(raw_root)
    staging_parent = Path(staging_parent)
    _require_matching_inspection_root(raw_root, bundle.inspection)
    receipt_path = staging_parent / RECEIPT_NAME
    if receipt_path.exists():
        raise _fatal(
            f"staging_parent/{RECEIPT_NAME}",
            "staged artifact already exists",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    _source_archive_documents(raw_root, bundle.inspection)

    staged_timings: dict[str, float] = {}
    staged = _build_rruff_data_staged(
        raw_root,
        staging_parent,
        bundle,
        timings=staged_timings,
    )
    receipt_started = perf_counter()
    receipt = _receipt_document(raw_root, bundle, staged)
    try:
        receipt_path.write_bytes(_canonical_json_bytes(receipt))
    except (OSError, TypeError, ValueError) as error:
        raise _fatal(
            "receipt",
            str(error),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error
    receipt_seconds = perf_counter() - receipt_started

    validation_started = perf_counter()
    _validate_rruff_receipt(receipt_path, bundle)
    receipt_validation_seconds = perf_counter() - validation_started
    if timings is not None:
        timings.update(staged_timings)
        timings["receipt_seconds"] = receipt_seconds
        timings["validation_seconds"] = (
            staged_timings["validation_seconds"]
            + receipt_validation_seconds
        )

    summary = RruffConversionSummary(
        raw=staged.raw,
        processed=staged.processed,
        pair_index_path=staged.pair_index_path,
        pair_index_sha256=staged.pair_index_sha256,
        pair_index_rows=staged.pair_index_rows,
        pair_status_counts=staged.pair_status_counts,
        axis_relation_counts=staged.axis_relation_counts,
        receipt_path=receipt_path,
        receipt_sha256=_sha256_file(receipt_path),
        rejected_records=bundle.statistics.rejected_records,
    )
    return (
        summary,
        staged.raw_comparison,
        staged.processed_comparison,
    )


def _linux_ru_maxrss_to_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Linux ru_maxrss must be a non-negative integer")
    return value * 1024


def _measure_rruff_dataset_access(
    dataset_path: Path,
    dataset_id: str,
) -> tuple[float, float, str]:
    dataset_path = Path(dataset_path)
    open_started = perf_counter()
    dataset = UnifiedDataset.open(dataset_path)
    open_seconds = perf_counter() - open_started
    try:
        record_ids = sorted(dataset.record_ids)
        if not record_ids:
            raise _fatal(
                f"{dataset_id}.record_ids",
                "must not be empty",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        probe = hashlib.sha256(
            b"rpe-rruff-read-probe-v1\0"
            + dataset_id.encode("utf-8")
        ).digest()
        index = int.from_bytes(probe[:8], "little") % len(record_ids)
        record_id = record_ids[index]
        read_started = perf_counter()
        dataset.get(record_id)
        read_seconds = perf_counter() - read_started
    finally:
        dataset.close()
    return open_seconds, read_seconds, record_id


def _rruff_feasibility_failures(
    summary: RruffConversionSummary,
    metrics: RruffRuntimeMetrics,
    comparisons: tuple[
        _RruffComparisonSummary,
        _RruffComparisonSummary,
    ],
) -> tuple[str, ...]:
    raw_comparison, processed_comparison = comparisons
    failures = []
    thresholds = (
        ("peak_rss_bytes", metrics.peak_rss_bytes, 8_589_934_592),
        (
            "combined_output_bytes",
            metrics.combined_output_bytes,
            4_294_967_296,
        ),
        (
            "staged_build_seconds",
            metrics.staged_build_seconds,
            3600.0,
        ),
        (
            "validation_seconds",
            metrics.validation_seconds,
            1800.0,
        ),
        (
            "lazy_open_seconds_raw",
            metrics.lazy_open_seconds_raw,
            60.0,
        ),
        (
            "lazy_open_seconds_processed",
            metrics.lazy_open_seconds_processed,
            60.0,
        ),
        (
            "representative_record_read_seconds_raw",
            metrics.representative_record_read_seconds_raw,
            5.0,
        ),
        (
            "representative_record_read_seconds_processed",
            metrics.representative_record_read_seconds_processed,
            5.0,
        ),
    )
    for name, observed, limit in thresholds:
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(observed)
            or observed < 0
            or observed >= limit
        ):
            failures.append(name)

    if (
        summary.raw.axis_group_count != 10_260
        or metrics.raw_axis_groups != 10_260
    ):
        failures.append("raw_axis_groups")
    if (
        summary.processed.axis_group_count != 11_009
        or metrics.processed_axis_groups != 11_009
    ):
        failures.append("processed_axis_groups")
    if (
        summary.raw.axis_group_count
        + summary.processed.axis_group_count
        != 21_269
        or metrics.raw_axis_groups + metrics.processed_axis_groups
        != 21_269
    ):
        failures.append("axis_groups_total")
    if (
        summary.raw.record_count + summary.processed.record_count
        != 38_043
    ):
        failures.append("accepted_records")
    if (
        raw_comparison.source_rows_compared
        + processed_comparison.source_rows_compared
        != 38_043
    ):
        failures.append("source_rows_compared")
    if (
        raw_comparison.intensity_mismatches
        + processed_comparison.intensity_mismatches
        != 0
    ):
        failures.append("intensity_mismatches")
    if (
        raw_comparison.class_label_mismatches
        + processed_comparison.class_label_mismatches
        != 0
    ):
        failures.append("class_label_mismatches")
    if (
        raw_comparison.eligibility_violations
        + processed_comparison.eligibility_violations
        != 0
    ):
        failures.append("eligibility_violations")
    return tuple(failures)


def _rruff_output_hashes(
    staging_parent: Path,
) -> Mapping[str, str]:
    staging_parent = Path(staging_parent)
    expected = {
        *(f"{RAW_DATASET_ID}/{name}" for name in DATASET_FILES),
        *(
            f"{PROCESSED_DATASET_ID}/{name}"
            for name in DATASET_FILES
        ),
        _PAIR_INDEX_NAME,
        RECEIPT_NAME,
    }
    paths = {
        path.relative_to(staging_parent).as_posix(): path
        for path in staging_parent.rglob("*")
        if path.is_file()
    }
    if set(paths) != expected:
        raise _fatal(
            "staging_parent.files",
            (
                f"file mismatch: missing={sorted(expected - set(paths))}, "
                f"unexpected={sorted(set(paths) - expected)}"
            ),
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    return MappingProxyType(
        {
            name: _sha256_file(paths[name])
            for name in sorted(
                paths,
                key=lambda value: value.encode("utf-8"),
            )
        }
    )


def _measure_rruff_staged(
    raw_root: Path,
    staging_parent: Path,
    bundle: _RruffInspectionBundle,
    *,
    inspection_seconds: float,
) -> tuple[
    RruffConversionSummary,
    _RruffComparisonSummary,
    _RruffComparisonSummary,
    Mapping[str, str],
    RruffRuntimeMetrics,
    tuple[str, ...],
]:
    timings: dict[str, float] = {}
    summary, raw_comparison, processed_comparison = (
        _build_rruff_staged(
            raw_root,
            staging_parent,
            bundle,
            timings=timings,
        )
    )
    (
        lazy_open_seconds_raw,
        representative_record_read_seconds_raw,
        _,
    ) = _measure_rruff_dataset_access(
        staging_parent / RAW_DATASET_ID,
        RAW_DATASET_ID,
    )
    (
        lazy_open_seconds_processed,
        representative_record_read_seconds_processed,
        _,
    ) = _measure_rruff_dataset_access(
        staging_parent / PROCESSED_DATASET_ID,
        PROCESSED_DATASET_ID,
    )
    output_hashes = _rruff_output_hashes(staging_parent)
    raw_output_bytes = summary.raw.output_bytes
    processed_output_bytes = summary.processed.output_bytes
    pair_index_bytes = summary.pair_index_path.stat().st_size
    receipt_bytes = summary.receipt_path.stat().st_size
    combined_output_bytes = (
        raw_output_bytes
        + processed_output_bytes
        + pair_index_bytes
        + receipt_bytes
    )
    source_archive_bytes = sum(
        (raw_root / archive_name).stat().st_size
        for archive_name in bundle.inspection.archive_sha256
    )
    if source_archive_bytes <= 0:
        raise _fatal(
            "metrics.source_archive_bytes",
            "must be positive",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    peak_rss_raw = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    staged_build_seconds = (
        timings["raw_build_seconds"]
        + timings["processed_build_seconds"]
        + timings["pair_index_seconds"]
        + timings["receipt_seconds"]
    )
    metrics = RruffRuntimeMetrics(
        total_seconds=0.0,
        inspection_seconds=inspection_seconds,
        staged_build_seconds=staged_build_seconds,
        raw_build_seconds=timings["raw_build_seconds"],
        processed_build_seconds=timings["processed_build_seconds"],
        pair_index_seconds=timings["pair_index_seconds"],
        receipt_seconds=timings["receipt_seconds"],
        validation_seconds=timings["validation_seconds"],
        peak_rss_raw=peak_rss_raw,
        peak_rss_bytes=_linux_ru_maxrss_to_bytes(peak_rss_raw),
        raw_output_bytes=raw_output_bytes,
        processed_output_bytes=processed_output_bytes,
        pair_index_bytes=pair_index_bytes,
        receipt_bytes=receipt_bytes,
        combined_output_bytes=combined_output_bytes,
        source_archive_bytes=source_archive_bytes,
        compression_ratio=(
            combined_output_bytes / source_archive_bytes
        ),
        raw_axis_groups=summary.raw.axis_group_count,
        processed_axis_groups=summary.processed.axis_group_count,
        lazy_open_seconds_raw=lazy_open_seconds_raw,
        lazy_open_seconds_processed=lazy_open_seconds_processed,
        representative_record_read_seconds_raw=(
            representative_record_read_seconds_raw
        ),
        representative_record_read_seconds_processed=(
            representative_record_read_seconds_processed
        ),
    )
    comparisons = (raw_comparison, processed_comparison)
    failures = _rruff_feasibility_failures(
        summary,
        metrics,
        comparisons,
    )
    return (
        summary,
        raw_comparison,
        processed_comparison,
        output_hashes,
        metrics,
        failures,
    )


def _stage_rruff_for_review(
    raw_root: Path,
    output_root: Path,
) -> _RruffStagingReview:
    started = perf_counter()
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    if output_root.exists() and not output_root.is_dir():
        raise _fatal(
            "output_root",
            "must be a directory",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    output_root_existed = output_root.exists()

    inspection_started = perf_counter()
    bundle = _inspect_rruff_bundle(
        raw_root,
        PRODUCTION_SOURCE_CONTRACT,
    )
    inspection_seconds = perf_counter() - inspection_started

    output_root.mkdir(parents=True, exist_ok=True)
    staging_parent: Path | None = None
    review: _RruffStagingReview | None = None
    try:
        staging_parent = Path(
            tempfile.mkdtemp(
                prefix=".rruff.staging-",
                dir=output_root,
            )
        )
        (
            summary,
            raw_comparison,
            processed_comparison,
            output_hashes,
            metrics,
            failures,
        ) = _measure_rruff_staged(
            raw_root,
            staging_parent,
            bundle,
            inspection_seconds=inspection_seconds,
        )
        review = _RruffStagingReview(
            raw=summary.raw,
            processed=summary.processed,
            pair_index_sha256=summary.pair_index_sha256,
            pair_index_rows=summary.pair_index_rows,
            pair_status_counts=summary.pair_status_counts,
            axis_relation_counts=summary.axis_relation_counts,
            receipt_sha256=summary.receipt_sha256,
            rejected_records=summary.rejected_records,
            raw_comparison=raw_comparison,
            processed_comparison=processed_comparison,
            source_statistics=bundle.statistics,
            output_hashes=output_hashes,
            metrics=metrics,
            feasibility_failures=failures,
        )
    finally:
        if staging_parent is not None and staging_parent.exists():
            shutil.rmtree(staging_parent)
        if (
            not output_root_existed
            and output_root.is_dir()
            and not any(output_root.iterdir())
        ):
            output_root.rmdir()

    if review is None:
        raise RuntimeError("RRUFF staging review did not produce a result")
    return replace(
        review,
        metrics=replace(
            review.metrics,
            total_seconds=perf_counter() - started,
        ),
    )


def _remove_published_artifact(path: Path) -> None:
    path = Path(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _path_lexists(path: Path) -> bool:
    path = Path(path)
    return path.is_symlink() or path.exists()


def _publish_rruff_artifacts(
    staging_parent: Path,
    artifacts: tuple[
        _PublicationArtifact,
        _PublicationArtifact,
        _PublicationArtifact,
        _PublicationArtifact,
    ],
    *,
    overwrite: bool,
) -> None:
    staging_parent = Path(staging_parent)
    if not staging_parent.is_dir():
        raise _fatal(
            "staging_parent",
            "must be an existing directory",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    if len(artifacts) != 4:
        raise _fatal(
            "publication.artifacts",
            "must contain exactly four artifacts",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )

    final_paths = [Path(artifact.final) for artifact in artifacts]
    if len(final_paths) != len(set(final_paths)):
        raise _fatal(
            "publication.artifacts.final",
            "final paths must be unique",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    backup_names = [artifact.backup_name for artifact in artifacts]
    if (
        len(backup_names) != len(set(backup_names))
        or any(
            not name
            or Path(name).name != name
            or name in {".", ".."}
            for name in backup_names
        )
    ):
        raise _fatal(
            "publication.artifacts.backup_name",
            "backup names must be unique portable basenames",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    for artifact in artifacts:
        staged = Path(artifact.staged)
        final = Path(artifact.final)
        if staged.parent != staging_parent:
            raise _fatal(
                "publication.artifacts.staged",
                "staged artifacts must be direct staging children",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        if not staged.exists():
            raise _fatal(
                f"publication.staged.{staged.name}",
                "staged artifact is missing",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        if final == staging_parent or staging_parent in final.parents:
            raise _fatal(
                "publication.artifacts.final",
                "final artifacts must be outside staging",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        backup = staging_parent / artifact.backup_name
        if _path_lexists(backup):
            raise _fatal(
                f"publication.backup.{artifact.backup_name}",
                "backup path already exists",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )
        if _path_lexists(final) and not overwrite:
            raise _fatal(
                f"output_root/{final.name}",
                "already exists and overwrite is false",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )

    backups: list[tuple[_PublicationArtifact, Path]] = []
    published: list[_PublicationArtifact] = []
    preserve_staging = False
    try:
        try:
            for artifact in artifacts:
                final = Path(artifact.final)
                if _path_lexists(final):
                    backup = staging_parent / artifact.backup_name
                    os.replace(final, backup)
                    backups.append((artifact, backup))

            for artifact in artifacts:
                os.replace(artifact.staged, artifact.final)
                published.append(artifact)
        except BaseException as publication_error:
            recovery_errors: list[str] = []
            for artifact in reversed(published):
                final = Path(artifact.final)
                if _path_lexists(final):
                    try:
                        _remove_published_artifact(final)
                    except BaseException as error:
                        recovery_errors.append(
                            f"remove {final}: {type(error).__name__}: {error}"
                        )
            for artifact, backup in backups:
                if not _path_lexists(backup):
                    continue
                final = Path(artifact.final)
                if _path_lexists(final):
                    recovery_errors.append(
                        f"restore {final}: destination still exists"
                    )
                    continue
                try:
                    os.replace(backup, final)
                except BaseException as error:
                    recovery_errors.append(
                        f"restore {final}: {type(error).__name__}: {error}"
                    )
            if recovery_errors:
                preserve_staging = True
                raise _fatal(
                    "publication.restore",
                    (
                        f"manual recovery required from {staging_parent}: "
                        + "; ".join(recovery_errors)
                    ),
                    "PUBLICATION_RESTORE_FAILED",
                ) from publication_error
            raise
    finally:
        if staging_parent.exists() and not preserve_staging:
            shutil.rmtree(staging_parent)


def _preflight_rruff_final_paths(
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    output_root = Path(output_root)
    if output_root.exists() and not output_root.is_dir():
        raise _fatal(
            "output_root",
            "must be a directory",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    for name in (
        RAW_DATASET_ID,
        PROCESSED_DATASET_ID,
        _PAIR_INDEX_NAME,
        RECEIPT_NAME,
    ):
        if _path_lexists(output_root / name) and not overwrite:
            raise _fatal(
                f"output_root/{name}",
                "already exists and overwrite is false",
                "SOURCE_MEMBER_COUNT_MISMATCH",
            )


def _run_rruff_conversion(
    raw_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> tuple[RruffConversionSummary, RruffRuntimeMetrics]:
    started = perf_counter()
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    _preflight_rruff_final_paths(output_root, overwrite=overwrite)
    output_root_existed = output_root.exists()

    inspection_started = perf_counter()
    bundle = _inspect_rruff_bundle(
        raw_root,
        PRODUCTION_SOURCE_CONTRACT,
    )
    inspection_seconds = perf_counter() - inspection_started

    output_root.mkdir(parents=True, exist_ok=True)
    staging_parent: Path | None = None
    preserve_staging = False
    try:
        staging_parent = Path(
            tempfile.mkdtemp(
                prefix=".rruff.staging-",
                dir=output_root,
            )
        )
        (
            summary,
            _raw_comparison,
            _processed_comparison,
            _output_hashes,
            metrics,
            feasibility_failures,
        ) = _measure_rruff_staged(
            raw_root,
            staging_parent,
            bundle,
            inspection_seconds=inspection_seconds,
        )
        if feasibility_failures:
            raise _fatal(
                "feasibility",
                (
                    "failed gates: "
                    + ", ".join(feasibility_failures)
                ),
                "STORAGE_FEASIBILITY_FAILED",
            )

        artifacts = (
            _PublicationArtifact(
                staged=staging_parent / RAW_DATASET_ID,
                final=output_root / RAW_DATASET_ID,
                backup_name="backup-raw",
            ),
            _PublicationArtifact(
                staged=staging_parent / PROCESSED_DATASET_ID,
                final=output_root / PROCESSED_DATASET_ID,
                backup_name="backup-processed",
            ),
            _PublicationArtifact(
                staged=staging_parent / _PAIR_INDEX_NAME,
                final=output_root / _PAIR_INDEX_NAME,
                backup_name="backup-pairs",
            ),
            _PublicationArtifact(
                staged=staging_parent / RECEIPT_NAME,
                final=output_root / RECEIPT_NAME,
                backup_name="backup-receipt",
            ),
        )
        try:
            _publish_rruff_artifacts(
                staging_parent,
                artifacts,
                overwrite=overwrite,
            )
        except RruffValidationError as error:
            preserve_staging = error.path == "publication.restore"
            raise

        final_summary = replace(
            summary,
            pair_index_path=output_root / _PAIR_INDEX_NAME,
            receipt_path=output_root / RECEIPT_NAME,
        )
        final_metrics = replace(
            metrics,
            total_seconds=perf_counter() - started,
        )
        return final_summary, final_metrics
    finally:
        if (
            staging_parent is not None
            and staging_parent.exists()
            and not preserve_staging
        ):
            shutil.rmtree(staging_parent)
        if (
            not output_root_existed
            and output_root.is_dir()
            and not any(output_root.iterdir())
        ):
            output_root.rmdir()


def build_rruff_unified(
    raw_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> RruffConversionSummary:
    summary, _ = _run_rruff_conversion(
        raw_root,
        output_root,
        overwrite=overwrite,
    )
    return summary


def _build_rruff_unified_with_metrics(
    raw_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
) -> tuple[RruffConversionSummary, RruffRuntimeMetrics]:
    return _run_rruff_conversion(
        raw_root,
        output_root,
        overwrite=overwrite,
    )
