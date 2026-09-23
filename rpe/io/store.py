from __future__ import annotations

import hashlib
import json
import os
import operator
import re
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import h5py
import numpy as np

from rpe.io.schema import (
    SCHEMA_VERSION,
    ArrayRef,
    JsonValue,
    PreprocessingStatus,
    RamanRecord,
    RecordArrays,
    SchemaValidationError,
    axis_id,
    metadata_to_record,
    record_to_metadata,
    validate_record,
)


PAYLOAD_FILES = ("arrays.h5", "dataset.json", "records.jsonl")
DATASET_FILES = (
    "SHA256SUMS",
    "SHA256SUMS.sha256",
    "arrays.h5",
    "dataset.json",
    "records.jsonl",
)
DATASET_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class DatasetValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class DatasetClosedError(RuntimeError):
    pass


@dataclass(frozen=True)
class DatasetSummary:
    path: Path
    dataset_id: str
    record_count: int
    axis_group_count: int


@dataclass(frozen=True)
class ValidationSummary:
    path: Path
    dataset_id: str
    record_count: int
    axis_group_count: int
    target_presence_counts: Mapping[str, int]
    preprocessing_status_counts: Mapping[str, int]
    checked_files: tuple[str, ...]


@dataclass(frozen=True)
class _DatasetInspection:
    summary: ValidationSummary
    manifest: Mapping[str, JsonValue]
    records: tuple[Mapping[str, JsonValue], ...]


def _canonical_json_bytes(value: object) -> bytes:
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


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_dataset_id(dataset_id: object) -> str:
    if (
        not isinstance(dataset_id, str)
        or DATASET_ID_PATTERN.fullmatch(dataset_id) is None
    ):
        raise DatasetValidationError(
            "dataset_id",
            "must be a portable dataset identifier",
        )
    return dataset_id


def _normalize_class_labels(
    class_labels: Mapping[int, str] | None,
    observed_labels: set[int],
) -> dict[int, str]:
    normalized = {} if class_labels is None else dict(class_labels)
    for label, name in normalized.items():
        if isinstance(label, bool) or not isinstance(label, int):
            raise DatasetValidationError(
                "class_labels",
                "keys must be integer labels and not Boolean",
            )
        if not isinstance(name, str) or name == "":
            raise DatasetValidationError(
                "class_labels",
                "values must be non-empty strings",
            )
    missing = observed_labels - set(normalized)
    if missing:
        raise DatasetValidationError(
            "class_labels missing",
            f"missing labels: {sorted(missing)}",
        )
    unused = set(normalized) - observed_labels
    if unused:
        raise DatasetValidationError(
            "class_labels unused",
            f"unused labels: {sorted(unused)}",
        )
    return normalized


def _normalize_concentration_units(
    concentration_unit: str | None,
    concentration_units: Mapping[str, str] | None,
    *,
    scalar_present: bool,
    observed_names: set[str],
) -> tuple[str | None, dict[str, str]]:
    if scalar_present:
        if not isinstance(concentration_unit, str) or concentration_unit == "":
            raise DatasetValidationError(
                "concentration_unit",
                "a non-empty scalar concentration unit is required",
            )
    elif concentration_unit is not None:
        if not isinstance(concentration_unit, str) or concentration_unit == "":
            raise DatasetValidationError(
                "concentration_unit",
                "must be a non-empty string or None",
            )

    normalized = (
        {} if concentration_units is None else dict(concentration_units)
    )
    for name, unit in normalized.items():
        if not isinstance(name, str) or name == "":
            raise DatasetValidationError(
                "concentration_units",
                "keys must be non-empty strings",
            )
        if not isinstance(unit, str) or unit == "":
            raise DatasetValidationError(
                "concentration_units",
                "values must be non-empty strings",
            )
    missing = observed_names - set(normalized)
    if missing:
        raise DatasetValidationError(
            "concentration_units missing",
            f"missing names: {sorted(missing)}",
        )
    unused = set(normalized) - observed_names
    if unused:
        raise DatasetValidationError(
            "concentration_units unused",
            f"unused names: {sorted(unused)}",
        )
    return concentration_unit, normalized


def _source_artifact_dict(record: RamanRecord) -> dict[str, object]:
    provenance = record.provenance
    return {
        "source_url": provenance.source_url,
        "license": provenance.license,
        "license_status": provenance.license_status.value,
        "sha256": provenance.sha256,
        "retrieved_date": provenance.retrieved_date.isoformat(),
        "source_artifact": provenance.source_artifact,
    }


def _source_artifact_sort_key(
    artifact: Mapping[str, object],
) -> tuple[str, str, str]:
    source_artifact = artifact["source_artifact"]
    return (
        str(artifact["source_url"]),
        str(artifact["sha256"]),
        "" if source_artifact is None else str(source_artifact),
    )


def _write_hdf5(
    path: Path,
    dataset_id: str,
    groups: Mapping[str, list[RamanRecord]],
) -> None:
    with h5py.File(path, "w") as arrays:
        arrays.attrs["schema_version"] = SCHEMA_VERSION
        arrays.attrs["dataset_id"] = dataset_id
        axes_group = arrays.create_group("axes")
        for current_axis_id in sorted(groups):
            records = groups[current_axis_id]
            record_count = len(records)
            feature_count = records[0].wavenumber.size
            group = axes_group.create_group(current_axis_id)
            group.attrs["axis_id"] = current_axis_id

            dataset_options = {
                "compression": "gzip",
                "shuffle": True,
                "fletcher32": True,
                "track_times": False,
            }
            group.create_dataset(
                "wavenumber",
                data=np.asarray(records[0].wavenumber, dtype="<f4"),
                chunks=(feature_count,),
                **dataset_options,
            )
            intensity = np.stack(
                [
                    np.asarray(record.intensity, dtype="<f4")
                    for record in records
                ]
            )
            group.create_dataset(
                "intensity",
                data=intensity,
                chunks=(min(record_count, 256), feature_count),
                **dataset_options,
            )

            for target_name in ("clean", "baseline"):
                values = [
                    getattr(record.targets, target_name)
                    for record in records
                ]
                if not any(value is not None for value in values):
                    continue
                present = np.array(
                    [value is not None for value in values],
                    dtype=np.bool_,
                )
                matrix = np.zeros(
                    (record_count, feature_count),
                    dtype="<f4",
                )
                for row, value in enumerate(values):
                    if value is not None:
                        matrix[row] = np.asarray(value, dtype="<f4")
                group.create_dataset(
                    target_name,
                    data=matrix,
                    chunks=(min(record_count, 256), feature_count),
                    **dataset_options,
                )
                group.create_dataset(
                    f"{target_name}_present",
                    data=present,
                    chunks=(min(record_count, 256),),
                    **dataset_options,
                )


def _write_checksums(path: Path) -> None:
    lines = [
        f"{_sha256_file(path / name)}  {name}\n"
        for name in sorted(PAYLOAD_FILES)
    ]
    checksum_path = path / "SHA256SUMS"
    checksum_path.write_text("".join(lines), encoding="utf-8")
    (path / "SHA256SUMS.sha256").write_text(
        f"{_sha256_file(checksum_path)}  SHA256SUMS\n",
        encoding="utf-8",
    )


def _parse_basic_checksum_file(
    path: Path,
    *,
    expected_names: set[str],
) -> dict[str, str]:
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DatasetValidationError(path.name, str(error)) from error
    line_pattern = re.compile(
        rb"([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)\n"
    )
    parsed: dict[str, str] = {}
    offset = 0
    while offset < len(content):
        match = line_pattern.match(content, offset)
        if match is None:
            raise DatasetValidationError(
                path.name,
                "checksum lines must match '<lowercase sha256>  <basename>\\n'",
            )
        digest = match.group(1).decode("ascii")
        name = match.group(2).decode("ascii")
        if name in parsed:
            raise DatasetValidationError(path.name, "duplicate checksum entry")
        parsed[name] = digest
        offset = match.end()
    if offset != len(content):
        raise DatasetValidationError(path.name, "invalid trailing checksum data")
    if set(parsed) != expected_names:
        raise DatasetValidationError(
            path.name,
            "checksum entry set does not match expected files",
        )
    return parsed


def _require_object(
    path: str,
    value: object,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DatasetValidationError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise DatasetValidationError(path, "object keys must be strings")
    return value


def _require_exact_keys(
    path: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        raise DatasetValidationError(
            path,
            (
                f"key mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
        )


def _load_manifest(path: Path) -> Mapping[str, object]:
    manifest_path = path / "dataset.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError("dataset.json", str(error)) from error
    parsed = _require_object("dataset.json", manifest)
    _require_exact_keys(
        "dataset.json",
        parsed,
        {
            "schema_version",
            "dataset_id",
            "record_count",
            "axis_groups",
            "class_labels",
            "concentration_unit",
            "concentration_units",
            "source_artifacts",
        },
    )
    if raw != _canonical_json_bytes(parsed):
        raise DatasetValidationError(
            "dataset.json",
            "must use canonical JSON encoding",
        )
    return parsed


def _load_record_documents(
    path: Path,
) -> tuple[Mapping[str, JsonValue], ...]:
    records_path = path / "records.jsonl"
    try:
        raw_lines = records_path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise DatasetValidationError("records.jsonl", str(error)) from error
    records: list[Mapping[str, JsonValue]] = []
    for index, raw_line in enumerate(raw_lines):
        record_path = f"records.jsonl[{index}]"
        if not raw_line.endswith(b"\n") or raw_line == b"\n":
            raise DatasetValidationError(
                record_path,
                "must be one non-empty canonical JSON line",
            )
        try:
            value = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DatasetValidationError(record_path, str(error)) from error
        parsed = _require_object(record_path, value)
        if raw_line != _canonical_json_bytes(parsed):
            raise DatasetValidationError(
                record_path,
                "must use canonical JSON encoding",
            )
        records.append(parsed)
    return tuple(records)


def _artifact_key(
    path: str,
    artifact: object,
) -> tuple[object, ...]:
    parsed = _require_object(path, artifact)
    keys = {
        "source_url",
        "license",
        "license_status",
        "sha256",
        "retrieved_date",
        "source_artifact",
    }
    _require_exact_keys(path, parsed, keys)
    for field in (
        "source_url",
        "license",
        "license_status",
        "sha256",
        "retrieved_date",
    ):
        if not isinstance(parsed[field], str):
            raise DatasetValidationError(
                path,
                f"{field} must be a string",
            )
    if (
        parsed["source_artifact"] is not None
        and not isinstance(parsed["source_artifact"], str)
    ):
        raise DatasetValidationError(
            path,
            "source_artifact must be a string or null",
        )
    return tuple(parsed[key] for key in sorted(keys))


def _artifact_sort_key(
    path: str,
    artifact: object,
) -> tuple[str, str, int, str]:
    parsed = _require_object(path, artifact)
    source_url = parsed.get("source_url")
    sha256 = parsed.get("sha256")
    source_artifact = parsed.get("source_artifact")
    if (
        not isinstance(source_url, str)
        or not isinstance(sha256, str)
        or (source_artifact is not None and not isinstance(source_artifact, str))
    ):
        raise DatasetValidationError(
            path,
            "contains invalid source artifact sort fields",
        )
    return (
        source_url,
        sha256,
        0 if source_artifact is None else 1,
        "" if source_artifact is None else source_artifact,
    )


def _validate_hdf_dataset(
    path: str,
    dataset: h5py.Dataset | h5py.Group,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunks: tuple[int, ...],
) -> None:
    if not isinstance(dataset, h5py.Dataset):
        raise DatasetValidationError(path, "must be an HDF5 dataset")
    if set(dataset.attrs):
        raise DatasetValidationError(path, "must not contain attributes")
    if dataset.shape != shape:
        raise DatasetValidationError(
            path,
            f"shape {dataset.shape} does not match {shape}",
        )
    if dataset.dtype != dtype:
        raise DatasetValidationError(
            path,
            f"dtype {dataset.dtype} does not match {dtype}",
        )
    if dataset.chunks != chunks:
        raise DatasetValidationError(
            path,
            f"chunks {dataset.chunks} do not match {chunks}",
        )
    if (
        dataset.compression != "gzip"
        or not dataset.shuffle
        or not dataset.fletcher32
    ):
        raise DatasetValidationError(
            path,
            "must use gzip, shuffle, and Fletcher32",
        )


def _read_record_arrays(
    arrays: h5py.File,
    metadata: Mapping[str, JsonValue],
    *,
    dataset_cache: Mapping[str, Mapping[str, h5py.Dataset]] | None = None,
    wavenumber_cache: dict[str, np.ndarray] | None = None,
) -> RecordArrays:
    array_ref = metadata["array_ref"]
    current_axis_id = array_ref["axis_id"]
    row = array_ref["row"]
    datasets = (
        arrays["axes"][current_axis_id]
        if dataset_cache is None
        else dataset_cache[current_axis_id]
    )
    targets = metadata["targets"]
    clean = (
        np.array(datasets["clean"][row], copy=True)
        if targets["clean_present"]
        else None
    )
    baseline = (
        np.array(datasets["baseline"][row], copy=True)
        if targets["baseline_present"]
        else None
    )
    if wavenumber_cache is None:
        wavenumber = np.array(datasets["wavenumber"][:], copy=True)
    else:
        cached_wavenumber = wavenumber_cache.get(current_axis_id)
        if cached_wavenumber is None:
            cached_wavenumber = np.array(
                datasets["wavenumber"][:],
                copy=True,
            )
            cached_wavenumber.setflags(write=False)
            wavenumber_cache[current_axis_id] = cached_wavenumber
        wavenumber = np.array(cached_wavenumber, copy=True)
    return RecordArrays(
        intensity=np.array(datasets["intensity"][row], copy=True),
        wavenumber=wavenumber,
        clean=clean,
        baseline=baseline,
    )


def _wrap_schema_error(
    path: str,
    error: SchemaValidationError,
) -> DatasetValidationError:
    return DatasetValidationError(
        f"{path}.{error.path}",
        error.reason,
    )


def _inspect_dataset(
    path: Path,
    *,
    verify_checksums: bool,
    exhaustive_arrays: bool,
) -> _DatasetInspection:
    path = Path(path)
    if not path.is_dir():
        raise DatasetValidationError(
            path.name or path.as_posix(),
            "dataset directory does not exist",
        )
    actual_entries = {child.name for child in path.iterdir()}
    if actual_entries != set(DATASET_FILES):
        raise DatasetValidationError(
            "directory",
            "dataset must contain exactly the five contract files",
        )

    payload_checksums = _parse_basic_checksum_file(
        path / "SHA256SUMS",
        expected_names=set(PAYLOAD_FILES),
    )
    index_checksums = _parse_basic_checksum_file(
        path / "SHA256SUMS.sha256",
        expected_names={"SHA256SUMS"},
    )
    if verify_checksums:
        for name, expected in payload_checksums.items():
            actual = _sha256_file(path / name)
            if actual != expected:
                raise DatasetValidationError(name, "checksum mismatch")
        actual_index = _sha256_file(path / "SHA256SUMS")
        if actual_index != index_checksums["SHA256SUMS"]:
            raise DatasetValidationError(
                "SHA256SUMS",
                "checksum-of-checksums mismatch",
            )

    manifest = _load_manifest(path)
    records = _load_record_documents(path)
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise DatasetValidationError(
            "dataset.json.schema_version",
            f"expected {SCHEMA_VERSION!r}",
        )
    dataset_id = manifest["dataset_id"]
    if not isinstance(dataset_id, str) or dataset_id != path.name:
        raise DatasetValidationError(
            "dataset.json.dataset_id",
            "must be a string matching the dataset directory name",
        )
    record_count = manifest["record_count"]
    axis_groups = manifest["axis_groups"]
    if not isinstance(record_count, int) or record_count != len(records):
        raise DatasetValidationError(
            "dataset.json.record_count",
            "does not match records.jsonl",
        )
    if not isinstance(axis_groups, list):
        raise DatasetValidationError(
            "dataset.json.axis_groups",
            "must be a list",
        )
    record_ids = [record.get("record_id") for record in records]
    if (
        any(not isinstance(record_id, str) for record_id in record_ids)
        or record_ids != sorted(record_ids)
        or len(record_ids) != len(set(record_ids))
    ):
        raise DatasetValidationError(
            "records.jsonl.record_id",
            "record IDs must be unique and sorted",
        )

    expected_axis_entry_keys = {"axis_id", "length", "record_count"}
    axis_manifest: dict[str, tuple[int, int]] = {}
    axis_order: list[str] = []
    for index, entry in enumerate(axis_groups):
        entry_path = f"dataset.json.axis_groups[{index}]"
        parsed = _require_object(entry_path, entry)
        _require_exact_keys(entry_path, parsed, expected_axis_entry_keys)
        current_axis_id = parsed["axis_id"]
        length = parsed["length"]
        group_record_count = parsed["record_count"]
        if (
            not isinstance(current_axis_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", current_axis_id) is None
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            or isinstance(group_record_count, bool)
            or not isinstance(group_record_count, int)
            or group_record_count <= 0
            or current_axis_id in axis_manifest
        ):
            raise DatasetValidationError(
                "dataset.json.axis_groups",
                "contains an invalid or duplicate axis-group entry",
            )
        axis_order.append(current_axis_id)
        axis_manifest[current_axis_id] = (length, group_record_count)
    if axis_order != sorted(axis_order):
        raise DatasetValidationError(
            "dataset.json.axis_groups",
            "must be sorted by axis_id",
        )
    if sum(count for _, count in axis_manifest.values()) != record_count:
        raise DatasetValidationError(
            "dataset.json.axis_groups",
            "record counts do not sum to record_count",
        )

    record_refs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, record in enumerate(records):
        record_path = f"records.jsonl[{index}]"
        array_ref = _require_object(
            f"{record_path}.array_ref",
            record.get("array_ref"),
        )
        _require_exact_keys(
            f"{record_path}.array_ref",
            array_ref,
            {"axis_id", "row"},
        )
        current_axis_id = array_ref["axis_id"]
        row = array_ref["row"]
        if (
            not isinstance(current_axis_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", current_axis_id) is None
        ):
            raise DatasetValidationError(
                f"{record_path}.array_ref.axis_id",
                "must be a lowercase SHA256 string",
            )
        if current_axis_id not in axis_manifest:
            raise DatasetValidationError(
                f"{record_path}.array_ref.axis_id",
                "does not reference a manifest/HDF5 axis group",
            )
        if isinstance(row, bool) or not isinstance(row, int) or row < 0:
            raise DatasetValidationError(
                "records.jsonl.array_ref.row",
                "must be a non-negative integer",
            )
        record_refs[current_axis_id].append((row, index))
    for current_axis_id, (_, group_count) in axis_manifest.items():
        rows = [row for row, _ in record_refs[current_axis_id]]
        if sorted(rows) != list(range(group_count)):
            raise DatasetValidationError(
                "records.jsonl.array_ref.row",
                "each HDF5 row must be referenced exactly once",
            )

    target_names = (
        "baseline",
        "class_label",
        "clean",
        "concentration",
        "concentrations",
        "peaks",
    )
    target_presence_counts = {name: 0 for name in target_names}
    preprocessing_status_counts = {
        status.value: 0 for status in PreprocessingStatus
    }
    reconstructed_records: list[RamanRecord] = []
    try:
        arrays = h5py.File(path / "arrays.h5", "r")
    except (OSError, ValueError) as error:
        raise DatasetValidationError("arrays.h5", str(error)) from error
    with arrays:
        if set(arrays.attrs) != {"schema_version", "dataset_id"}:
            raise DatasetValidationError(
                "arrays.h5.attrs",
                "root attributes must be exactly schema_version and dataset_id",
            )
        root_schema_version = arrays.attrs["schema_version"]
        root_dataset_id = arrays.attrs["dataset_id"]
        if (
            not isinstance(root_schema_version, str)
            or not isinstance(root_dataset_id, str)
            or root_schema_version != SCHEMA_VERSION
            or root_dataset_id != dataset_id
        ):
            raise DatasetValidationError(
                "arrays.h5.attrs",
                "root attributes do not match the manifest",
            )
        if set(arrays) != {"axes"}:
            raise DatasetValidationError(
                "arrays.h5",
                "root must contain only the axes group",
            )
        axes = arrays["axes"]
        if not isinstance(axes, h5py.Group):
            raise DatasetValidationError(
                "arrays.h5.axes",
                "must be an HDF5 group",
            )
        if set(axes.attrs):
            raise DatasetValidationError(
                "arrays.h5.axes.attrs",
                "axes group must not contain attributes",
            )
        if set(axes) != set(axis_manifest):
            raise DatasetValidationError(
                "arrays.h5.axes",
                "axis-group set does not match the manifest",
            )

        axis_values: dict[str, np.ndarray] = {}
        mask_values: dict[tuple[str, str], np.ndarray] = {}
        for current_axis_id in axis_order:
            length, group_count = axis_manifest[current_axis_id]
            group_path = f"arrays.h5.axes.{current_axis_id}"
            group = axes[current_axis_id]
            if not isinstance(group, h5py.Group):
                raise DatasetValidationError(
                    group_path,
                    "must be an HDF5 group",
                )
            group_axis_id = group.attrs["axis_id"] if "axis_id" in group.attrs else None
            if (
                set(group.attrs) != {"axis_id"}
                or not isinstance(group_axis_id, str)
                or group_axis_id != current_axis_id
            ):
                raise DatasetValidationError(
                    f"{group_path}.attrs",
                    "must contain only the matching axis_id attribute",
                )
            expected_optional: set[str] = set()
            for target_name in ("clean", "baseline"):
                presence_key = f"{target_name}_present"
                target_exists = target_name in group
                mask_exists = presence_key in group
                if target_exists != mask_exists:
                    raise DatasetValidationError(
                        group_path,
                        f"{target_name} and {presence_key} must coexist",
                    )
                if target_exists:
                    expected_optional.update({target_name, presence_key})
            expected_datasets = {
                "intensity",
                "wavenumber",
                *expected_optional,
            }
            if set(group) != expected_datasets:
                raise DatasetValidationError(
                    group_path,
                    "dataset set does not match record target presence",
                )
            _validate_hdf_dataset(
                f"{group_path}.wavenumber",
                group["wavenumber"],
                shape=(length,),
                dtype=np.dtype("float32"),
                chunks=(length,),
            )
            _validate_hdf_dataset(
                f"{group_path}.intensity",
                group["intensity"],
                shape=(group_count, length),
                dtype=np.dtype("float32"),
                chunks=(min(group_count, 256), length),
            )
            wavenumber = np.array(group["wavenumber"][:], copy=True)
            try:
                computed_axis_id = axis_id(wavenumber)
            except SchemaValidationError as error:
                raise _wrap_schema_error(
                    f"{group_path}.wavenumber",
                    error,
                ) from error
            if computed_axis_id != current_axis_id:
                raise DatasetValidationError(
                    f"{group_path}.wavenumber",
                    "stored values do not match the axis_id",
                )
            axis_values[current_axis_id] = wavenumber
            for target_name in ("clean", "baseline"):
                presence_name = f"{target_name}_present"
                if target_name not in expected_optional:
                    continue
                _validate_hdf_dataset(
                    f"{group_path}.{target_name}",
                    group[target_name],
                    shape=(group_count, length),
                    dtype=np.dtype("float32"),
                    chunks=(min(group_count, 256), length),
                )
                _validate_hdf_dataset(
                    f"{group_path}.{presence_name}",
                    group[presence_name],
                    shape=(group_count,),
                    dtype=np.dtype("bool"),
                    chunks=(min(group_count, 256),),
                )
                mask_values[(current_axis_id, target_name)] = np.array(
                    group[presence_name][:],
                    copy=True,
                )
                if not np.any(mask_values[(current_axis_id, target_name)]):
                    raise DatasetValidationError(
                        f"{group_path}.{presence_name}",
                        "optional target datasets require at least one present row",
                    )
                if exhaustive_arrays:
                    matrix = np.array(group[target_name][:], copy=True)
                    absent_rows = np.flatnonzero(
                        ~mask_values[(current_axis_id, target_name)]
                    )
                    for absent_row in absent_rows:
                        if np.any(matrix[absent_row] != 0):
                            raise DatasetValidationError(
                                (
                                    f"{group_path}.{target_name}"
                                    f"[{int(absent_row)}]"
                                ),
                                "absent optional target rows must be zero-filled",
                            )

        for index, record_document in enumerate(records):
            record_path = f"records.jsonl[{index}]"
            array_ref = record_document["array_ref"]
            current_axis_id = array_ref["axis_id"]
            row = array_ref["row"]
            targets = _require_object(
                f"{record_path}.targets",
                record_document.get("targets"),
            )
            clean_present = targets.get("clean_present")
            baseline_present = targets.get("baseline_present")
            for target_name, present in (
                ("clean", clean_present),
                ("baseline", baseline_present),
            ):
                mask = mask_values.get((current_axis_id, target_name))
                mask_present = False if mask is None else bool(mask[row])
                if present != mask_present:
                    raise DatasetValidationError(
                        f"{record_path}.targets.{target_name}_present",
                        "does not match the HDF5 presence mask",
                    )
            dummy_arrays = RecordArrays(
                intensity=np.zeros(
                    axis_values[current_axis_id].shape,
                    dtype=np.float32,
                ),
                wavenumber=axis_values[current_axis_id],
                clean=(
                    np.zeros(
                        axis_values[current_axis_id].shape,
                        dtype=np.float32,
                    )
                    if clean_present
                    else None
                ),
                baseline=(
                    np.zeros(
                        axis_values[current_axis_id].shape,
                        dtype=np.float32,
                    )
                    if baseline_present
                    else None
                ),
            )
            try:
                scalar_record = metadata_to_record(
                    record_document,
                    dummy_arrays,
                )
            except SchemaValidationError as error:
                raise _wrap_schema_error(record_path, error) from error
            if scalar_record.meta.dataset_id != dataset_id:
                raise DatasetValidationError(
                    f"{record_path}.meta.dataset_id",
                    "does not match the manifest",
                )
            target_presence_counts["clean"] += int(clean_present)
            target_presence_counts["baseline"] += int(baseline_present)
            for name in (
                "class_label",
                "concentration",
                "concentrations",
                "peaks",
            ):
                target_presence_counts[name] += int(
                    getattr(scalar_record.targets, name) is not None
                )
            preprocessing_status_counts[
                scalar_record.meta.preprocessing_status.value
            ] += 1
            if exhaustive_arrays:
                try:
                    record_arrays = _read_record_arrays(
                        arrays,
                        record_document,
                    )
                    reconstructed_records.append(
                        metadata_to_record(
                            record_document,
                            record_arrays,
                        )
                    )
                except SchemaValidationError as error:
                    raise _wrap_schema_error(record_path, error) from error

    manifest_artifacts = manifest["source_artifacts"]
    if not isinstance(manifest_artifacts, list):
        raise DatasetValidationError(
            "dataset.json.source_artifacts",
            "must be a list",
        )
    try:
        manifest_artifact_keys = [
            _artifact_key(
                f"dataset.json.source_artifacts[{index}]",
                artifact,
            )
            for index, artifact in enumerate(manifest_artifacts)
        ]
        manifest_artifact_sort_keys = [
            _artifact_sort_key(
                f"dataset.json.source_artifacts[{index}]",
                artifact,
            )
            for index, artifact in enumerate(manifest_artifacts)
        ]
        record_artifact_keys = {
            _artifact_key(
                f"records.jsonl[{index}].provenance",
                record["provenance"],
            )
            for index, record in enumerate(records)
        }
    except (KeyError, TypeError) as error:
        raise DatasetValidationError(
            "dataset.json.source_artifacts",
            str(error),
        ) from error
    if (
        len(manifest_artifact_keys) != len(set(manifest_artifact_keys))
        or set(manifest_artifact_keys) != record_artifact_keys
        or manifest_artifact_sort_keys != sorted(manifest_artifact_sort_keys)
    ):
        raise DatasetValidationError(
            "dataset.json.source_artifacts",
            "must equal the unique record provenance set",
        )

    class_labels = _require_object(
        "dataset.json.class_labels",
        manifest["class_labels"],
    )
    observed_labels = {
        str(record["targets"]["class_label"])
        for record in records
        if record["targets"]["class_label"] is not None
    }
    if (
        set(class_labels) != observed_labels
        or any(
            not isinstance(value, str) or value == ""
            for value in class_labels.values()
        )
    ):
        raise DatasetValidationError(
            "dataset.json.class_labels",
            "must exactly cover observed class labels",
        )
    scalar_present = any(
        record["targets"]["concentration"] is not None
        for record in records
    )
    concentration_unit = manifest["concentration_unit"]
    if scalar_present and (
        not isinstance(concentration_unit, str)
        or concentration_unit == ""
    ):
        raise DatasetValidationError(
            "dataset.json.concentration_unit",
            "a non-empty scalar unit is required",
        )
    if not scalar_present and concentration_unit is not None and (
        not isinstance(concentration_unit, str)
        or concentration_unit == ""
    ):
        raise DatasetValidationError(
            "dataset.json.concentration_unit",
            "must be a non-empty string or null",
        )
    concentration_units = _require_object(
        "dataset.json.concentration_units",
        manifest["concentration_units"],
    )
    observed_names = {
        name
        for record in records
        if record["targets"]["concentrations"] is not None
        for name in record["targets"]["concentrations"]
    }
    if (
        set(concentration_units) != observed_names
        or any(
            not isinstance(value, str) or value == ""
            for value in concentration_units.values()
        )
    ):
        raise DatasetValidationError(
            "dataset.json.concentration_units",
            "must exactly cover observed named concentrations",
        )

    summary = ValidationSummary(
        path=path,
        dataset_id=dataset_id,
        record_count=record_count,
        axis_group_count=len(axis_groups),
        target_presence_counts=target_presence_counts,
        preprocessing_status_counts=preprocessing_status_counts,
        checked_files=tuple(sorted(DATASET_FILES)),
    )
    return _DatasetInspection(
        summary=summary,
        manifest=manifest,
        records=records,
    )


def validate_dataset(
    path: Path,
    *,
    verify_checksums: bool = True,
) -> ValidationSummary:
    return _inspect_dataset(
        Path(path),
        verify_checksums=verify_checksums,
        exhaustive_arrays=True,
    ).summary


class UnifiedDataset:
    def __init__(
        self,
        path: Path,
        inspection: _DatasetInspection,
        arrays: h5py.File,
    ) -> None:
        self._path = path
        self._manifest = inspection.manifest
        self._records = inspection.records
        self._record_ids = tuple(
            str(record["record_id"]) for record in self._records
        )
        self._metadata_by_id = {
            record_id: record
            for record_id, record in zip(
                self._record_ids,
                self._records,
                strict=True,
            )
        }
        self._arrays = arrays
        self._array_datasets = {
            current_axis_id: {
                name: group[name]
                for name in group
                if isinstance(group[name], h5py.Dataset)
            }
            for current_axis_id, group in arrays["axes"].items()
        }
        self._wavenumber_cache: dict[str, np.ndarray] = {}
        self._closed = False

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        verify_checksums: bool = True,
    ) -> UnifiedDataset:
        path = Path(path)
        inspection = _inspect_dataset(
            path,
            verify_checksums=verify_checksums,
            exhaustive_arrays=False,
        )
        try:
            arrays = h5py.File(path / "arrays.h5", "r")
        except (OSError, ValueError) as error:
            raise DatasetValidationError("arrays.h5", str(error)) from error
        return cls(path, inspection, arrays)

    def _require_open(self) -> None:
        if self._closed:
            raise DatasetClosedError("unified dataset is closed")

    def __enter__(self) -> UnifiedDataset:
        self._require_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __len__(self) -> int:
        self._require_open()
        return len(self._record_ids)

    @property
    def record_ids(self) -> tuple[str, ...]:
        self._require_open()
        return self._record_ids

    def get(self, record_id: str) -> RamanRecord:
        self._require_open()
        try:
            metadata = self._metadata_by_id[record_id]
        except KeyError:
            raise KeyError(record_id) from None
        arrays = _read_record_arrays(
            self._arrays,
            metadata,
            dataset_cache=self._array_datasets,
            wavenumber_cache=self._wavenumber_cache,
        )
        return metadata_to_record(metadata, arrays)

    def __getitem__(self, index: int) -> RamanRecord:
        self._require_open()
        if isinstance(index, bool):
            raise TypeError("Boolean indices are invalid")
        try:
            normalized = operator.index(index)
        except TypeError as error:
            raise TypeError("dataset index must be an integer") from error
        if normalized < 0:
            normalized += len(self._record_ids)
        if normalized < 0 or normalized >= len(self._record_ids):
            raise IndexError(index)
        return self.get(self._record_ids[normalized])

    def iter_records(self) -> Iterator[RamanRecord]:
        self._require_open()
        for record_id in self._record_ids:
            yield self.get(record_id)

    def close(self) -> None:
        if self._closed:
            return
        self._arrays.close()
        self._closed = True


def write_dataset(
    records: Iterable[RamanRecord],
    output_dir: Path,
    *,
    dataset_id: str,
    class_labels: Mapping[int, str] | None = None,
    concentration_unit: str | None = None,
    concentration_units: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> DatasetSummary:
    dataset_id = _validate_dataset_id(dataset_id)
    output_dir = Path(output_dir)
    if output_dir.name != dataset_id:
        raise DatasetValidationError(
            "output_dir.name",
            "must equal dataset_id",
        )
    if output_dir.exists():
        if not overwrite:
            raise DatasetValidationError(
                "output_dir",
                "already exists and overwrite is false",
            )

    materialized = list(records)
    if not materialized:
        raise DatasetValidationError("records", "must not be empty")
    for index, record in enumerate(materialized):
        try:
            validate_record(record)
        except SchemaValidationError as error:
            raise DatasetValidationError(
                f"records[{index}].{error.path}",
                error.reason,
            ) from error

    record_ids = [record.record_id for record in materialized]
    if len(record_ids) != len(set(record_ids)):
        raise DatasetValidationError(
            "records.record_id",
            "record IDs must be unique",
        )
    if any(record.meta.dataset_id != dataset_id for record in materialized):
        raise DatasetValidationError(
            "records.dataset_id",
            "all records must match dataset_id",
        )

    observed_labels = {
        int(record.targets.class_label)
        for record in materialized
        if record.targets.class_label is not None
    }
    normalized_labels = _normalize_class_labels(
        class_labels,
        observed_labels,
    )
    scalar_present = any(
        record.targets.concentration is not None
        for record in materialized
    )
    observed_concentration_names = {
        name
        for record in materialized
        if record.targets.concentrations is not None
        for name in record.targets.concentrations
    }
    concentration_unit, normalized_units = _normalize_concentration_units(
        concentration_unit,
        concentration_units,
        scalar_present=scalar_present,
        observed_names=observed_concentration_names,
    )

    ordered_records = sorted(
        materialized,
        key=lambda record: record.record_id,
    )
    groups: dict[str, list[RamanRecord]] = defaultdict(list)
    for record in ordered_records:
        groups[axis_id(record.wavenumber)].append(record)

    source_artifacts_by_key: dict[
        tuple[str, str, str | None], dict[str, object]
    ] = {}
    for record in ordered_records:
        artifact = _source_artifact_dict(record)
        key = (
            record.provenance.source_url,
            record.provenance.sha256,
            record.provenance.source_artifact,
        )
        source_artifacts_by_key[key] = artifact
    source_artifacts = sorted(
        source_artifacts_by_key.values(),
        key=_source_artifact_sort_key,
    )

    axis_groups = [
        {
            "axis_id": current_axis_id,
            "length": int(groups[current_axis_id][0].wavenumber.size),
            "record_count": len(groups[current_axis_id]),
        }
        for current_axis_id in sorted(groups)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "record_count": len(ordered_records),
        "axis_groups": axis_groups,
        "class_labels": {
            str(label): normalized_labels[label]
            for label in sorted(normalized_labels)
        },
        "concentration_unit": concentration_unit,
        "concentration_units": {
            name: normalized_units[name] for name in sorted(normalized_units)
        },
        "source_artifacts": source_artifacts,
    }

    rows_by_axis: dict[str, int] = defaultdict(int)
    record_documents = []
    for record in ordered_records:
        current_axis_id = axis_id(record.wavenumber)
        row = rows_by_axis[current_axis_id]
        rows_by_axis[current_axis_id] += 1
        record_documents.append(
            record_to_metadata(
                record,
                ArrayRef(axis_id=current_axis_id, row=row),
            )
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = Path(
        tempfile.mkdtemp(
            prefix=f".{dataset_id}.staging-",
            dir=output_dir.parent,
        )
    )
    staged_dataset = staging_parent / dataset_id
    staged_dataset.mkdir()
    backup = staging_parent / "backup"
    published = False
    preserve_staging = False
    try:
        (staged_dataset / "dataset.json").write_bytes(
            _canonical_json_bytes(manifest)
        )
        with (staged_dataset / "records.jsonl").open("wb") as records_file:
            for document in record_documents:
                records_file.write(_canonical_json_bytes(document))
        _write_hdf5(staged_dataset / "arrays.h5", dataset_id, groups)
        _write_checksums(staged_dataset)
        validation = validate_dataset(staged_dataset)
        if output_dir.exists():
            os.replace(output_dir, backup)
        try:
            os.replace(staged_dataset, output_dir)
        except BaseException:
            if backup.exists():
                try:
                    os.replace(backup, output_dir)
                except BaseException:
                    preserve_staging = True
                    raise
            raise
        published = True
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staging_parent.exists() and not preserve_staging:
            shutil.rmtree(staging_parent)

    if not published:
        raise DatasetValidationError(
            "output_dir",
            "dataset publication failed",
        )
    return DatasetSummary(
        path=output_dir,
        dataset_id=validation.dataset_id,
        record_count=validation.record_count,
        axis_group_count=validation.axis_group_count,
    )
