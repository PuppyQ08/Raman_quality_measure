from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from rpe.evaluation.contracts import Spectrum1D
from rpe.io.schema import JsonValue, PreprocessingStatus
from rpe.io.store import UnifiedDataset, validate_dataset
from rpe.runner.phase1_config import Phase1CoreConfig, REPO_ROOT


_SAMPLE_DOMAIN = b"rpe-phase1-core10k-sample-order-v1\0"
_RECORD_DOMAIN = b"rpe-phase1-core10k-record-order-v1\0"
_EXPECTED_SOURCE_FILES = (
    "SHA256SUMS",
    "SHA256SUMS.sha256",
    "arrays.h5",
    "dataset.json",
    "records.jsonl",
)
_EXPECTED_CHECKED_FILES = tuple(sorted(_EXPECTED_SOURCE_FILES))


class Phase1SelectionError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class SourceInventoryRow:
    record_id: str
    sample_id: str
    class_label: int
    mineral_name: str
    axis_id: str


@dataclass(frozen=True)
class SelectedSourceRow:
    selection_rank: int
    record_id: str
    sample_id: str
    class_label: int
    mineral_name: str
    axis_id: str


@dataclass(frozen=True)
class Phase1Source:
    selection: SelectedSourceRow
    spectrum: Spectrum1D
    original_axis_orientation: str
    source_axis_float32_sha256: str
    source_intensity_float32_sha256: str
    normalized_axis_float64_sha256: str
    normalized_intensity_float64_sha256: str
    provenance: Mapping[str, JsonValue]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _json_ready(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise Phase1SelectionError(
        "json",
        f"unsupported value type {type(value).__name__}",
    )


def _freeze_json(path: str, value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return tuple(
            _freeze_json(f"{path}[{index}]", item)
            for index, item in enumerate(value)
        )
    if isinstance(value, tuple):
        return tuple(
            _freeze_json(f"{path}[{index}]", item)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in sorted(value.items()):
            if not isinstance(key, str) or key == "":
                raise Phase1SelectionError(path, "must use nonempty string keys")
            frozen[key] = _freeze_json(f"{path}.{key}", item)
        return MappingProxyType(frozen)
    raise Phase1SelectionError(
        path,
        f"unsupported JSON value type {type(value).__name__}",
    )


def _length_prefixed_utf8(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _hash_payload(
    domain: bytes,
    *,
    seed: int,
    class_label: int,
    sample_id: str,
    record_id: str | None = None,
) -> bytes:
    payload = (
        domain
        + struct.pack("<Q", seed)
        + struct.pack("<q", class_label)
        + _length_prefixed_utf8(sample_id)
    )
    if record_id is not None:
        payload += _length_prefixed_utf8(record_id)
    return hashlib.sha256(payload).digest()


def _resolved_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_identity(path: str, file_path: Path, *, byte_count: int, sha256: str) -> None:
    resolved = _resolved_path(file_path)
    try:
        observed_size = resolved.stat().st_size
    except OSError as error:
        raise Phase1SelectionError(path, str(error)) from error
    if observed_size != byte_count:
        raise Phase1SelectionError(path, "byte_count mismatch")
    observed_sha256 = _sha256_file(resolved)
    if observed_sha256 != sha256:
        raise Phase1SelectionError(path, "sha256 mismatch")


def _require_object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase1SelectionError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise Phase1SelectionError(path, "object keys must be strings")
    return value


def _load_manifest(dataset_path: Path) -> Mapping[str, object]:
    manifest_path = dataset_path / "dataset.json"
    try:
        value = json.loads(manifest_path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase1SelectionError("dataset.json", str(error)) from error
    return _require_object("dataset.json", value)


def _load_record_rows(dataset_path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    records_path = dataset_path / "records.jsonl"
    try:
        with records_path.open("rb") as stream:
            for index, raw_line in enumerate(stream):
                path = f"records.jsonl[{index}]"
                if not raw_line.endswith(b"\n") or raw_line == b"\n":
                    raise Phase1SelectionError(
                        path,
                        "must be one non-empty canonical JSON line",
                    )
                try:
                    value = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise Phase1SelectionError(path, str(error)) from error
                rows.append(_require_object(path, value))
    except OSError as error:
        raise Phase1SelectionError("records.jsonl", str(error)) from error
    return tuple(rows)


def load_source_inventory(config: Phase1CoreConfig) -> tuple[SourceInventoryRow, ...]:
    dataset_path = _resolved_path(config.source_dataset_path)
    for name in _EXPECTED_SOURCE_FILES:
        identity = config.source_files[name]
        _verify_identity(
            name,
            dataset_path / name,
            byte_count=identity.byte_count,
            sha256=identity.sha256,
        )
    for name, located in config.related_provenance.items():
        _verify_identity(
            name,
            located.path,
            byte_count=located.identity.byte_count,
            sha256=located.identity.sha256,
        )

    summary = validate_dataset(dataset_path, verify_checksums=True)
    if summary.dataset_id != config.source_dataset_id:
        raise Phase1SelectionError("dataset_id", "validated dataset_id mismatch")
    if summary.record_count != config.source_record_count:
        raise Phase1SelectionError("record_count", "validated record_count mismatch")
    if tuple(summary.checked_files) != _EXPECTED_CHECKED_FILES:
        raise Phase1SelectionError("checked_files", "validated checked_files mismatch")

    manifest = _load_manifest(dataset_path)
    class_labels = _require_object("dataset.json.class_labels", manifest["class_labels"])
    if len(class_labels) != config.source_class_count:
        raise Phase1SelectionError("class_count", "validated class_count mismatch")

    inventory_rows: list[SourceInventoryRow] = []
    for index, document in enumerate(_load_record_rows(dataset_path)):
        row_path = f"records.jsonl[{index}]"
        record_id = document.get("record_id")
        if not isinstance(record_id, str) or record_id == "":
            raise Phase1SelectionError(f"{row_path}.record_id", "must be a nonempty string")
        array_ref = _require_object(f"{row_path}.array_ref", document.get("array_ref"))
        axis_id = array_ref.get("axis_id")
        if not isinstance(axis_id, str) or axis_id == "":
            raise Phase1SelectionError(f"{row_path}.array_ref.axis_id", "must be a nonempty string")
        meta = _require_object(f"{row_path}.meta", document.get("meta"))
        preprocessing_status = meta.get("preprocessing_status")
        if preprocessing_status != PreprocessingStatus.KNOWN_RAW.value:
            raise Phase1SelectionError(f"{row_path}.preprocessing_status", "must equal known_raw")
        sample_id = meta.get("sample_id")
        if not isinstance(sample_id, str) or sample_id == "":
            raise Phase1SelectionError(f"{row_path}.sample_id", "must be a nonempty string")
        targets = _require_object(f"{row_path}.targets", document.get("targets"))
        class_label = targets.get("class_label")
        if isinstance(class_label, bool) or not isinstance(class_label, int):
            raise Phase1SelectionError(f"{row_path}.class_label", "must be an integer")
        mineral_name = class_labels.get(str(class_label))
        if not isinstance(mineral_name, str) or mineral_name == "":
            raise Phase1SelectionError(f"{row_path}.mineral_name", "must be present in manifest class_labels")
        inventory_rows.append(
            SourceInventoryRow(
                record_id=record_id,
                sample_id=sample_id,
                class_label=class_label,
                mineral_name=mineral_name,
                axis_id=axis_id,
            )
        )

    inventory = tuple(sorted(inventory_rows, key=lambda row: row.record_id))
    if len(inventory) < config.subset_size:
        raise Phase1SelectionError("subset_size", "subset_size exceeds eligible source count")
    return inventory


def hamilton_class_quotas(
    rows: tuple[SourceInventoryRow, ...],
    subset_size: int,
) -> Mapping[int, int]:
    if subset_size < 0:
        raise Phase1SelectionError("subset_size", "must be nonnegative")
    counts: dict[int, int] = {}
    for row in rows:
        counts[row.class_label] = counts.get(row.class_label, 0) + 1
    if not counts:
        if subset_size == 0:
            return {}
        raise Phase1SelectionError("rows", "must not be empty")
    class_count = len(counts)
    if subset_size < class_count:
        raise Phase1SelectionError("subset_size", "must allocate at least one record per class")
    if subset_size > len(rows):
        raise Phase1SelectionError("subset_size", "subset_size exceeds eligible source count")

    quotas = {class_label: 1 for class_label in counts}
    additional = subset_size - class_count
    capacities = {
        class_label: count - 1
        for class_label, count in counts.items()
    }
    total_capacity = sum(capacities.values())
    if additional == 0:
        return dict(sorted(quotas.items()))
    if additional > total_capacity:
        raise Phase1SelectionError("subset_size", "subset_size exceeds class capacity")

    remainders: list[tuple[int, int]] = []
    for class_label in sorted(capacities):
        capacity = capacities[class_label]
        if total_capacity == 0:
            quotient = 0
            remainder = 0
        else:
            quotient, remainder = divmod(additional * capacity, total_capacity)
        quotas[class_label] += quotient
        remainders.append((class_label, remainder))
    remaining = additional - sum(quotas[class_label] - 1 for class_label in quotas)
    for class_label, _ in sorted(
        remainders,
        key=lambda item: (-item[1], item[0]),
    )[:remaining]:
        quotas[class_label] += 1
    return dict(sorted(quotas.items()))


def select_source_rows(
    rows: tuple[SourceInventoryRow, ...],
    *,
    global_seed: int,
    subset_size: int,
) -> tuple[SelectedSourceRow, ...]:
    quotas = hamilton_class_quotas(rows, subset_size)
    by_class: dict[int, list[SourceInventoryRow]] = {}
    for row in rows:
        by_class.setdefault(row.class_label, []).append(row)

    selected_rows: list[SourceInventoryRow] = []
    for class_label in sorted(quotas):
        quota = quotas[class_label]
        grouped: dict[str, list[SourceInventoryRow]] = {}
        for row in by_class[class_label]:
            grouped.setdefault(row.sample_id, []).append(row)
        ordered_groups = [
            sorted(
                grouped[sample_id],
                key=lambda row: (
                    _hash_payload(
                        _RECORD_DOMAIN,
                        seed=global_seed,
                        class_label=class_label,
                        sample_id=sample_id,
                        record_id=row.record_id,
                    ),
                    row.record_id.encode("utf-8"),
                ),
            )
            for sample_id in sorted(
                grouped,
                key=lambda sample_id: (
                    _hash_payload(
                        _SAMPLE_DOMAIN,
                        seed=global_seed,
                        class_label=class_label,
                        sample_id=sample_id,
                    ),
                    sample_id.encode("utf-8"),
                ),
            )
        ]
        class_selected: list[SourceInventoryRow] = []
        depth = 0
        while len(class_selected) < quota:
            for group in ordered_groups:
                if depth < len(group):
                    class_selected.append(group[depth])
                    if len(class_selected) == quota:
                        break
            depth += 1
        selected_rows.extend(class_selected)

    return tuple(
        SelectedSourceRow(
            selection_rank=index,
            record_id=row.record_id,
            sample_id=row.sample_id,
            class_label=row.class_label,
            mineral_name=row.mineral_name,
            axis_id=row.axis_id,
        )
        for index, row in enumerate(sorted(selected_rows, key=lambda row: row.record_id))
    )


def selected_records_jsonl_bytes(rows: tuple[SelectedSourceRow, ...]) -> bytes:
    payload = tuple(
        {
            "axis_id": row.axis_id,
            "class_label": row.class_label,
            "mineral_name": row.mineral_name,
            "record_id": row.record_id,
            "sample_id": row.sample_id,
            "selection_rank": row.selection_rank,
        }
        for row in rows
    )
    return b"".join(_canonical_json_bytes(document) for document in payload)


def load_phase1_source(dataset: UnifiedDataset, row: SelectedSourceRow) -> Phase1Source:
    try:
        record = dataset.get(row.record_id)
    except KeyError as error:
        raise Phase1SelectionError("record_id", f"unknown record_id {row.record_id!r}") from error

    axis_f4 = np.ascontiguousarray(np.asarray(record.wavenumber, dtype="<f4"))
    intensity_f4 = np.ascontiguousarray(np.asarray(record.intensity, dtype="<f4"))
    if axis_f4.ndim != 1 or intensity_f4.ndim != 1 or axis_f4.shape != intensity_f4.shape:
        raise Phase1SelectionError("array shape", "axis and intensity must be one-dimensional with equal length")
    if axis_f4.size == 0:
        raise Phase1SelectionError("array shape", "axis and intensity must be nonempty")
    if not np.isfinite(axis_f4).all():
        raise Phase1SelectionError("wavenumber finite", "contains non-finite values")
    if not np.isfinite(intensity_f4).all():
        raise Phase1SelectionError("intensity finite", "contains non-finite values")

    differences = np.diff(axis_f4)
    if np.all(differences > 0.0):
        orientation = "increasing"
        normalized_axis_f4 = axis_f4
        normalized_intensity_f4 = intensity_f4
    elif np.all(differences < 0.0):
        orientation = "decreasing"
        normalized_axis_f4 = np.ascontiguousarray(axis_f4[::-1])
        normalized_intensity_f4 = np.ascontiguousarray(intensity_f4[::-1])
    else:
        raise Phase1SelectionError("wavenumber monotonic", "axis must be strictly increasing or decreasing")

    axis_f8 = np.ascontiguousarray(normalized_axis_f4.astype("<f8"))
    intensity_f8 = np.ascontiguousarray(normalized_intensity_f4.astype("<f8"))
    spectrum = Spectrum1D(
        spectrum_id=f"{record.meta.dataset_id}::{record.record_id}",
        sample_id=record.meta.sample_id,
        axis_cm1=axis_f8,
        intensity=intensity_f8,
    )

    provenance = MappingProxyType(
        {
            "license": record.provenance.license,
            "license_status": record.provenance.license_status.value,
            "retrieved_date": record.provenance.retrieved_date.isoformat(),
            "sha256": record.provenance.sha256,
            "source_artifact": record.provenance.source_artifact,
            "source_url": record.provenance.source_url,
        }
    )
    return Phase1Source(
        selection=row,
        spectrum=spectrum,
        original_axis_orientation=orientation,
        source_axis_float32_sha256=hashlib.sha256(axis_f4.tobytes()).hexdigest(),
        source_intensity_float32_sha256=hashlib.sha256(intensity_f4.tobytes()).hexdigest(),
        normalized_axis_float64_sha256=hashlib.sha256(axis_f8.tobytes()).hexdigest(),
        normalized_intensity_float64_sha256=hashlib.sha256(intensity_f8.tobytes()).hexdigest(),
        provenance=provenance,
    )


def source_subset_jsonl_bytes(sources: tuple[Phase1Source, ...]) -> bytes:
    payload = tuple(
        {
            "axis_id": source.selection.axis_id,
            "class_label": source.selection.class_label,
            "mineral_name": source.selection.mineral_name,
            "normalized_axis_float64_sha256": source.normalized_axis_float64_sha256,
            "normalized_intensity_float64_sha256": source.normalized_intensity_float64_sha256,
            "original_axis_orientation": source.original_axis_orientation,
            "provenance": source.provenance,
            "record_id": source.selection.record_id,
            "sample_id": source.selection.sample_id,
            "selection_rank": source.selection.selection_rank,
            "source_axis_float32_sha256": source.source_axis_float32_sha256,
            "source_intensity_float32_sha256": source.source_intensity_float32_sha256,
        }
        for source in sources
    )
    return b"".join(_canonical_json_bytes(document) for document in payload)
