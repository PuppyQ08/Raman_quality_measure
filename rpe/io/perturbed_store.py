from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

from rpe.io.perturbed_schema import (
    SHARD_FILES,
    SHARD_SCHEMA_VERSION,
    STORE_SCHEMA_VERSION,
    PerturbedStoreError,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    float64_le_bytes,
    logical_array_digest,
)
from rpe.io.schema import JsonValue
from rpe.perturb.contracts import derive_perturbed_spectrum_id
from rpe.runner.phase1_config import ArtifactIdentity
from rpe.runner.phase1_types import Phase1ShardPayload


_HEX_DIGITS = frozenset("0123456789abcdef")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_CELL_ORDER = tuple(f"p{index:02d}" for index in range(1, 13))
_MATERIALIZED_ORDER = ("p01", "p02", "p03", "p04", "p05", "p08", "p09", "p10", "p11", "p12")
_DEFERRED_IDS = frozenset({"p06", "p07"})
_ALPHA_ORDER = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
_ALPHA_HEX_ORDER = tuple(struct.pack("<d", alpha).hex() for alpha in _ALPHA_ORDER)
_IDENTITY_ALPHA_HEX = struct.pack("<d", 0.0).hex()
_DEFERRED_REASON_CODE = "missing_explicit_baseline"
_DEFERRED_DEPENDENCY = "phase2_semisynthetic_or_separately_approved_physical_baseline"
_SOURCE_IDS_DOMAIN = b"rpe-phase1-shard-source-ids-v1\0"
_LOGICAL_CONTENT_DOMAIN = b"rpe-phase1-shard-logical-content-v1\0"
_AXIS_ID_DOMAIN = b"rpe-phase1-float64-axis-v1\0"
_EXPECTED_CELL_KEYS = (
    "actual_output_count",
    "dependency",
    "exception_message",
    "exception_path",
    "exception_type",
    "expected_output_count",
    "native_gate",
    "output_spectrum_ids",
    "perturbation_id",
    "reason_code",
    "run_id",
    "scientific_config_sha256",
    "source",
    "state_digest",
    "status",
    "sweep_config_sha256",
)
_EXPECTED_SOURCE_KEYS = (
    "axis_id",
    "axis_index",
    "class_label",
    "intensity_length",
    "intensity_offset",
    "mineral_name",
    "normalized_axis_float64_sha256",
    "normalized_intensity_float64_sha256",
    "original_axis_orientation",
    "provenance",
    "sample_id",
    "selection_axis_id",
    "selection_rank",
    "source_axis_float32_sha256",
    "source_dataset_id",
    "source_intensity_float32_sha256",
    "source_record_id",
    "source_spectrum_id",
    "source_index",
)
_EXPECTED_RECORD_KEYS = (
    "alpha",
    "alpha_float64_le_hex",
    "axis_behavior",
    "axis_changed",
    "axis_id",
    "axis_length",
    "class_label",
    "diagnostics",
    "intensity_changed",
    "intensity_length",
    "intensity_offset",
    "mineral_name",
    "output_spectrum_id",
    "perturbation_id",
    "run_id",
    "sample_id",
    "source_dataset_id",
    "source_index",
    "source_record_id",
    "source_selection_rank",
    "source_spectrum_id",
    "state_digest",
    "sweep_config_sha256",
)
_EXPECTED_CELL_KEYS_CANONICAL = tuple(sorted(_EXPECTED_CELL_KEYS))
_EXPECTED_SOURCE_KEYS_CANONICAL = tuple(sorted(_EXPECTED_SOURCE_KEYS))
_EXPECTED_RECORD_KEYS_CANONICAL = tuple(sorted(_EXPECTED_RECORD_KEYS))


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise PerturbedStoreError(path, "must be a nonempty string")
    return value


def _nonnegative_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PerturbedStoreError(path, "must be a nonnegative integer")
    return value


def _positive_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerturbedStoreError(path, "must be a positive integer")
    return value


def _finite_float(path: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PerturbedStoreError(path, "must be a finite real number")
    return float(value)


def _lower_hex(path: str, value: object, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise PerturbedStoreError(
            path,
            f"must be a lowercase {length}-character hexadecimal string",
        )
    return value


def _safe_run_id(value: object) -> str:
    run_id = _nonempty_string("run_id", value)
    if run_id in {".", ".."} or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise PerturbedStoreError(
            "run_id",
            "must be one safe nonempty path component",
        )
    if "/" in run_id or "\\" in run_id or "\x00" in run_id:
        raise PerturbedStoreError(
            "run_id",
            "must be one safe nonempty path component",
        )
    return run_id


def _freeze_json(path: str, value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        _reject_absolute_paths(path, value)
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PerturbedStoreError(path, "must be finite")
        return value
    if isinstance(value, np.generic):
        raise PerturbedStoreError(path, "must be canonical-JSON-compatible")
    if isinstance(value, tuple):
        return tuple(_freeze_json(f"{path}[{index}]", item) for index, item in enumerate(value))
    if isinstance(value, list):
        return tuple(_freeze_json(f"{path}[{index}]", item) for index, item in enumerate(value))
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in sorted(value.items()):
            if not isinstance(key, str) or key == "":
                raise PerturbedStoreError(path, "mapping keys must be nonempty strings")
            _reject_absolute_paths(f"{path}.{key}", key)
            frozen[key] = _freeze_json(f"{path}.{key}", item)
        return MappingProxyType(frozen)
    raise PerturbedStoreError(path, "must be canonical-JSON-compatible")


def _freeze_mapping(path: str, value: object) -> Mapping[str, JsonValue]:
    frozen = _freeze_json(path, value)
    if not isinstance(frozen, Mapping):
        raise PerturbedStoreError(path, "must be a mapping")
    return frozen


def _reject_absolute_paths(path: str, value: object) -> None:
    if isinstance(value, str) and value.startswith("/"):
        raise PerturbedStoreError(path, "must not contain absolute paths")


def _readonly_f8_copy(path: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PerturbedStoreError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise PerturbedStoreError(path, "dtype must equal little-endian float64")
    if value.ndim != 1:
        raise PerturbedStoreError(path, "must be one-dimensional")
    if value.size == 0:
        raise PerturbedStoreError(path, "must be nonempty")
    if not np.isfinite(value).all():
        raise PerturbedStoreError(path, "must contain only finite values")
    copied = np.ascontiguousarray(value, dtype="<f8").copy()
    copied.setflags(write=False)
    return copied


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _length_prefixed_utf8(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _float64_axis_id(values: np.ndarray) -> str:
    return hashlib.sha256(
        _AXIS_ID_DOMAIN
        + struct.pack("<Q", values.size)
        + float64_le_bytes(values)
    ).hexdigest()


def _source_ids_digest(source_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(_SOURCE_IDS_DOMAIN)
    digest.update(struct.pack("<Q", len(source_ids)))
    for source_id in source_ids:
        digest.update(_length_prefixed_utf8(source_id))
    return digest.hexdigest()


def _logical_content_digest(
    *,
    cells_jsonl_bytes: bytes,
    records_jsonl_bytes: bytes,
    axes: Sequence[np.ndarray],
    sources: Sequence[np.ndarray],
    records: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(_LOGICAL_CONTENT_DOMAIN)
    digest.update(struct.pack("<Q", len(cells_jsonl_bytes)))
    digest.update(cells_jsonl_bytes)
    digest.update(struct.pack("<Q", len(records_jsonl_bytes)))
    digest.update(records_jsonl_bytes)
    digest.update(bytes.fromhex(logical_array_digest(axes, sources, records)))
    return digest.hexdigest()


def _track_times_disabled(h5_object: h5py.Dataset | h5py.Group) -> bool | None:
    plist = h5_object.id.get_create_plist()
    getter = getattr(plist, "get_obj_track_times", None)
    if getter is None:
        return None
    return not bool(getter())


def _dataset_options(length: int) -> dict[str, object]:
    chunk_length = max(1, min(length, 4096))
    return {
        "chunks": (chunk_length,),
        "compression": "gzip",
        "compression_opts": 1,
        "shuffle": False,
        "fletcher32": False,
        "track_times": False,
    }


def _fsync_path(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY") and path.is_dir():
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_binary_file(path: Path, data: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _split_source_spectrum_id(path: str, source_spectrum_id: object, source_record_id: object) -> str:
    spectrum_id = _nonempty_string(path, source_spectrum_id)
    record_id = _nonempty_string(f"{path}.record", source_record_id)
    prefix = "::"
    if spectrum_id.count(prefix) != 1:
        raise PerturbedStoreError(path, "must equal <dataset_id>::<source_record_id>")
    dataset_id, observed_record_id = spectrum_id.split(prefix, 1)
    if dataset_id == "" or observed_record_id != record_id:
        raise PerturbedStoreError(path, "must equal <dataset_id>::<source_record_id>")
    return dataset_id


@dataclass(frozen=True)
class StoredSource:
    source_spectrum_id: str
    source_record_id: str
    sample_id: str
    class_label: int
    mineral_name: str
    axis_cm1: np.ndarray
    intensity: np.ndarray
    provenance: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_spectrum_id", _nonempty_string("source_spectrum_id", self.source_spectrum_id))
        object.__setattr__(self, "source_record_id", _nonempty_string("source_record_id", self.source_record_id))
        object.__setattr__(self, "sample_id", _nonempty_string("sample_id", self.sample_id))
        object.__setattr__(self, "class_label", _nonnegative_int("class_label", self.class_label))
        object.__setattr__(self, "mineral_name", _nonempty_string("mineral_name", self.mineral_name))
        axis = _readonly_f8_copy("axis_cm1", self.axis_cm1)
        intensity = _readonly_f8_copy("intensity", self.intensity)
        if axis.size != intensity.size:
            raise PerturbedStoreError("intensity", "length must match axis length")
        object.__setattr__(self, "axis_cm1", axis)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "provenance", _freeze_mapping("provenance", self.provenance))


@dataclass(frozen=True)
class StoredRecord:
    source_spectrum_id: str
    perturbation_id: str
    output_spectrum_id: str
    alpha: float
    alpha_float64_le_hex: str
    state_digest: str
    sweep_config_sha256: str
    axis_behavior: str
    axis_changed: bool
    intensity_changed: bool
    axis_cm1: np.ndarray
    intensity: np.ndarray
    diagnostics: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_spectrum_id", _nonempty_string("source_spectrum_id", self.source_spectrum_id))
        object.__setattr__(self, "perturbation_id", _nonempty_string("perturbation_id", self.perturbation_id))
        output_spectrum_id = _nonempty_string("output_spectrum_id", self.output_spectrum_id)
        alpha = _finite_float("alpha", self.alpha)
        object.__setattr__(self, "alpha", alpha)
        expected_hex = struct.pack("<d", alpha).hex()
        if self.alpha_float64_le_hex != expected_hex:
            raise PerturbedStoreError("alpha_float64_le_hex", "must equal little-endian float64 hex")
        object.__setattr__(self, "alpha_float64_le_hex", expected_hex)
        object.__setattr__(self, "state_digest", _lower_hex("state_digest", self.state_digest))
        object.__setattr__(self, "sweep_config_sha256", _lower_hex("sweep_config_sha256", self.sweep_config_sha256))
        axis_behavior = _nonempty_string("axis_behavior", self.axis_behavior)
        if axis_behavior not in {"preserve", "transform"}:
            raise PerturbedStoreError("axis_behavior", "must be a supported axis behavior value")
        object.__setattr__(self, "axis_behavior", axis_behavior)
        if not isinstance(self.axis_changed, bool):
            raise PerturbedStoreError("axis_changed", "must be bool")
        if not isinstance(self.intensity_changed, bool):
            raise PerturbedStoreError("intensity_changed", "must be bool")
        axis = _readonly_f8_copy("axis_cm1", self.axis_cm1)
        intensity = _readonly_f8_copy("intensity", self.intensity)
        if axis.size != intensity.size:
            raise PerturbedStoreError("intensity", "length must match axis length")
        object.__setattr__(self, "axis_cm1", axis)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "diagnostics", _freeze_mapping("diagnostics", self.diagnostics))
        expected_output_spectrum_id = derive_perturbed_spectrum_id(
            self.source_spectrum_id,
            self.perturbation_id,
            alpha,
            self.state_digest,
            self.sweep_config_sha256,
        )
        if output_spectrum_id != expected_output_spectrum_id:
            raise PerturbedStoreError("output_spectrum_id", "must match the canonical perturbed spectrum ID")
        object.__setattr__(self, "output_spectrum_id", expected_output_spectrum_id)


@dataclass(frozen=True)
class StoredCell:
    source_spectrum_id: str
    perturbation_id: str
    status: str
    reason_code: str | None
    state_digest: str | None
    records: tuple[StoredRecord, ...]
    native_gate: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_spectrum_id", _nonempty_string("source_spectrum_id", self.source_spectrum_id))
        perturbation_id = _nonempty_string("perturbation_id", self.perturbation_id)
        status = _nonempty_string("status", self.status)
        if status not in {"complete", "not_applicable", "failed"}:
            raise PerturbedStoreError("status", "must be one of complete/not_applicable/failed")
        object.__setattr__(self, "perturbation_id", perturbation_id)
        object.__setattr__(self, "status", status)
        if self.reason_code is not None:
            object.__setattr__(self, "reason_code", _nonempty_string("reason_code", self.reason_code))
        if self.state_digest is not None:
            object.__setattr__(self, "state_digest", _lower_hex("state_digest", self.state_digest))
        if not isinstance(self.records, tuple):
            raise PerturbedStoreError("records", "must be a tuple")
        for index, record in enumerate(self.records):
            if not isinstance(record, StoredRecord):
                raise PerturbedStoreError(f"records[{index}]", "must be StoredRecord")
            if record.source_spectrum_id != self.source_spectrum_id:
                raise PerturbedStoreError(f"records[{index}].source_spectrum_id", "must match cell source_spectrum_id")
            if record.perturbation_id != perturbation_id:
                raise PerturbedStoreError(f"records[{index}].perturbation_id", "must match cell perturbation_id")
            if self.state_digest is not None and record.state_digest != self.state_digest:
                raise PerturbedStoreError(f"records[{index}].state_digest", "must match cell state_digest")
        object.__setattr__(self, "native_gate", _freeze_mapping("native_gate", self.native_gate))
        if status == "complete":
            if len(self.records) != len(_ALPHA_ORDER):
                raise PerturbedStoreError("records", "must contain exactly nine outputs when status is complete")
            if self.reason_code is not None:
                raise PerturbedStoreError("reason_code", "must be absent when status is complete")
            if self.state_digest is None:
                raise PerturbedStoreError("state_digest", "must be present when status is complete")
            if not self.native_gate:
                raise PerturbedStoreError("native_gate", "must be nonempty when status is complete")
            seen_output_ids: set[str] = set()
            for index, record in enumerate(self.records):
                if record.output_spectrum_id in seen_output_ids:
                    raise PerturbedStoreError(
                        f"records[{index}].output_spectrum_id",
                        "must be unique within the cell",
                    )
                seen_output_ids.add(record.output_spectrum_id)
            observed_alpha_hex = tuple(record.alpha_float64_le_hex for record in self.records)
            if observed_alpha_hex != _ALPHA_HEX_ORDER:
                raise PerturbedStoreError("records", "must follow the frozen alpha order")
        else:
            if self.records:
                raise PerturbedStoreError("records", "must be empty unless status is complete")
            if self.native_gate:
                raise PerturbedStoreError("native_gate", "must be empty unless status is complete")
            if status == "not_applicable" and self.reason_code is None:
                raise PerturbedStoreError("reason_code", "must be present when status is not_applicable")
            if status == "failed" and self.reason_code is not None:
                raise PerturbedStoreError("reason_code", "must be absent when status is failed")


@dataclass(frozen=True)
class ShardReceipt:
    schema_version: str
    run_id: str
    shard_index: int
    source_count: int
    cell_count: int
    record_count: int
    source_ids_sha256: str
    logical_content_sha256: str
    files: Mapping[str, ArtifactIdentity]

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", _nonempty_string("schema_version", self.schema_version))
        if self.schema_version != SHARD_SCHEMA_VERSION:
            raise PerturbedStoreError("schema_version", "must equal shard schema version")
        object.__setattr__(self, "run_id", _safe_run_id(self.run_id))
        object.__setattr__(self, "shard_index", _nonnegative_int("shard_index", self.shard_index))
        object.__setattr__(self, "source_count", _positive_int("source_count", self.source_count))
        object.__setattr__(self, "cell_count", _positive_int("cell_count", self.cell_count))
        object.__setattr__(self, "record_count", _nonnegative_int("record_count", self.record_count))
        object.__setattr__(self, "source_ids_sha256", _lower_hex("source_ids_sha256", self.source_ids_sha256))
        object.__setattr__(self, "logical_content_sha256", _lower_hex("logical_content_sha256", self.logical_content_sha256))
        if not isinstance(self.files, Mapping):
            raise PerturbedStoreError("files", "must be a mapping")
        expected_keys = ("arrays.h5", "cells.jsonl", "records.jsonl")
        normalized: dict[str, ArtifactIdentity] = {}
        for key, value in self.files.items():
            if not isinstance(value, ArtifactIdentity):
                raise PerturbedStoreError("files", "values must be ArtifactIdentity")
            normalized[_nonempty_string("files.key", key)] = value
        if tuple(sorted(normalized)) != expected_keys or tuple(normalized) != expected_keys:
            raise PerturbedStoreError("files", "must contain exactly sorted arrays.h5/cells.jsonl/records.jsonl")
        object.__setattr__(self, "files", MappingProxyType(normalized))


@dataclass(frozen=True)
class StoredShard:
    run_id: str
    shard_index: int
    sources: tuple[StoredSource, ...]
    cells: tuple[StoredCell, ...]
    receipt: ShardReceipt

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _safe_run_id(self.run_id))
        object.__setattr__(self, "shard_index", _nonnegative_int("shard_index", self.shard_index))
        if not isinstance(self.sources, tuple) or not self.sources:
            raise PerturbedStoreError("sources", "must be a nonempty tuple")
        if not isinstance(self.cells, tuple) or not self.cells:
            raise PerturbedStoreError("cells", "must be a nonempty tuple")
        for index, source in enumerate(self.sources):
            if not isinstance(source, StoredSource):
                raise PerturbedStoreError(f"sources[{index}]", "must be StoredSource")
        if len(self.cells) != len(self.sources) * len(_CELL_ORDER):
            raise PerturbedStoreError("cells", "must contain exactly twelve cells per source")
        position = 0
        for source in self.sources:
            for perturbation_id in _CELL_ORDER:
                cell = self.cells[position]
                if not isinstance(cell, StoredCell):
                    raise PerturbedStoreError(f"cells[{position}]", "must be StoredCell")
                if cell.source_spectrum_id != source.source_spectrum_id:
                    raise PerturbedStoreError(f"cells[{position}].source_spectrum_id", "must follow source-major order")
                if cell.perturbation_id != perturbation_id:
                    raise PerturbedStoreError(f"cells[{position}].perturbation_id", "must follow exact P01-P12 order")
                for record_index, record in enumerate(cell.records):
                    axis_changed = not np.array_equal(source.axis_cm1, record.axis_cm1)
                    intensity_changed = not np.array_equal(source.intensity, record.intensity)
                    if record.axis_changed != axis_changed:
                        raise PerturbedStoreError(
                            f"cells[{position}].records[{record_index}].axis_changed",
                            "must equal the exact comparison against the owning source axis",
                        )
                    if record.intensity_changed != intensity_changed:
                        raise PerturbedStoreError(
                            f"cells[{position}].records[{record_index}].intensity_changed",
                            "must equal the exact comparison against the owning source intensity",
                        )
                    if record.axis_behavior == "preserve" and axis_changed:
                        raise PerturbedStoreError(
                            f"cells[{position}].records[{record_index}].axis_behavior",
                            "preserve records must not change the source axis",
                        )
                    if record.alpha_float64_le_hex == _IDENTITY_ALPHA_HEX and (
                        axis_changed or intensity_changed
                    ):
                        raise PerturbedStoreError(
                            f"cells[{position}].records[{record_index}].alpha",
                            "identity alpha must preserve exact axis and intensity identity",
                        )
                position += 1
        if not isinstance(self.receipt, ShardReceipt):
            raise PerturbedStoreError("receipt", "must be ShardReceipt")
        if self.receipt.run_id != self.run_id:
            raise PerturbedStoreError("receipt.run_id", "must match shard run_id")
        if self.receipt.shard_index != self.shard_index:
            raise PerturbedStoreError("receipt.shard_index", "must match shard_index")
        if self.receipt.source_count != self.source_count:
            raise PerturbedStoreError("receipt.source_count", "must match stored sources")
        if self.receipt.cell_count != self.cell_count:
            raise PerturbedStoreError("receipt.cell_count", "must match stored cells")
        if self.receipt.record_count != self.record_count:
            raise PerturbedStoreError("receipt.record_count", "must match stored records")

    @property
    def source_count(self) -> int:
        return len(self.sources)

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    @property
    def record_count(self) -> int:
        return sum(len(cell.records) for cell in self.cells)


@dataclass(frozen=True)
class _ProjectedSource:
    public: StoredSource
    source_index: int
    selection_rank: int
    source_dataset_id: str
    selection_axis_id: str
    source_axis_float32_sha256: str
    source_intensity_float32_sha256: str
    normalized_axis_float64_sha256: str
    normalized_intensity_float64_sha256: str
    original_axis_orientation: str
    axis_id: str


@dataclass(frozen=True)
class _ProjectedRecord:
    public: StoredRecord
    source_index: int
    source_record_id: str
    source_dataset_id: str
    sample_id: str
    class_label: int
    mineral_name: str
    source_selection_rank: int
    axis_id: str


@dataclass(frozen=True)
class _ProjectedCell:
    public: StoredCell
    source_index: int
    scientific_config_sha256: str
    sweep_config_sha256: str
    exception_type: str | None
    exception_path: str | None
    exception_message: str | None


@dataclass(frozen=True)
class _ProjectedShard:
    sources: tuple[_ProjectedSource, ...]
    cells: tuple[_ProjectedCell, ...]
    materialized_records: tuple[_ProjectedRecord, ...]
    source_snapshots: tuple[tuple[str, np.ndarray, np.ndarray], ...]
    record_snapshots: tuple[tuple[str, np.ndarray, np.ndarray], ...]


def _validate_payload(
    payload: Phase1ShardPayload,
    *,
    scientific_config_sha256: str,
    sweep_config_sha256: str,
) -> _ProjectedShard:
    if not isinstance(payload, Phase1ShardPayload):
        raise PerturbedStoreError("payload", "must be Phase1ShardPayload")
    _lower_hex("scientific_config_sha256", scientific_config_sha256)
    _lower_hex("sweep_config_sha256", sweep_config_sha256)
    projected_sources: list[_ProjectedSource] = []
    projected_cells: list[_ProjectedCell] = []
    materialized_records: list[_ProjectedRecord] = []
    source_snapshots: list[tuple[str, np.ndarray, np.ndarray]] = []
    record_snapshots: list[tuple[str, np.ndarray, np.ndarray]] = []
    seen_source_ids: set[str] = set()
    position = 0
    for source_index, source in enumerate(payload.sources):
        dataset_id = _split_source_spectrum_id(
            f"sources[{source_index}].source_spectrum_id",
            source.spectrum.spectrum_id,
            source.selection.record_id,
        )
        if source.spectrum.spectrum_id in seen_source_ids:
            raise PerturbedStoreError(
                f"sources[{source_index}].source_spectrum_id",
                "must be unique within the shard",
            )
        seen_source_ids.add(source.spectrum.spectrum_id)
        public_source = StoredSource(
            source_spectrum_id=source.spectrum.spectrum_id,
            source_record_id=source.selection.record_id,
            sample_id=source.selection.sample_id,
            class_label=source.selection.class_label,
            mineral_name=source.selection.mineral_name,
            axis_cm1=source.spectrum.axis_cm1,
            intensity=source.spectrum.intensity,
            provenance=source.provenance,
        )
        projected_source = _ProjectedSource(
            public=public_source,
            source_index=source_index,
            selection_rank=_nonnegative_int(
                f"sources[{source_index}].selection_rank",
                source.selection.selection_rank,
            ),
            source_dataset_id=dataset_id,
            selection_axis_id=_nonempty_string(
                f"sources[{source_index}].selection_axis_id",
                source.selection.axis_id,
            ),
            source_axis_float32_sha256=_lower_hex(
                f"sources[{source_index}].source_axis_float32_sha256",
                source.source_axis_float32_sha256,
            ),
            source_intensity_float32_sha256=_lower_hex(
                f"sources[{source_index}].source_intensity_float32_sha256",
                source.source_intensity_float32_sha256,
            ),
            normalized_axis_float64_sha256=_lower_hex(
                f"sources[{source_index}].normalized_axis_float64_sha256",
                source.normalized_axis_float64_sha256,
            ),
            normalized_intensity_float64_sha256=_lower_hex(
                f"sources[{source_index}].normalized_intensity_float64_sha256",
                source.normalized_intensity_float64_sha256,
            ),
            original_axis_orientation=_nonempty_string(
                f"sources[{source_index}].original_axis_orientation",
                source.original_axis_orientation,
            ),
            axis_id=_float64_axis_id(public_source.axis_cm1),
        )
        source_snapshots.append(
            (f"sources[{source_index}].axis_cm1", source.spectrum.axis_cm1, np.array(source.spectrum.axis_cm1, copy=True))
        )
        source_snapshots.append(
            (f"sources[{source_index}].intensity", source.spectrum.intensity, np.array(source.spectrum.intensity, copy=True))
        )
        projected_sources.append(projected_source)
        for perturbation_id in _CELL_ORDER:
            cell = payload.cells[position]
            if cell.source is not source:
                raise PerturbedStoreError(
                    f"cells[{position}].source",
                    "must match shard source object at this position",
                )
            if cell.perturbation_id != perturbation_id:
                raise PerturbedStoreError(
                    f"cells[{position}].perturbation_id",
                    "must follow exact P01-P12 order",
                )
            status = cell.evidence.status.value
            state_digest = None if cell.state is None else cell.state.state_digest
            exception_path = cell.evidence.exception_path
            if exception_path is not None and exception_path.startswith("/"):
                raise PerturbedStoreError(
                    f"cells[{position}].exception_path",
                    "must not contain absolute paths",
                )
            public_records: list[StoredRecord] = []
            if status == "complete":
                if not cell.evidence.native_gate:
                    raise PerturbedStoreError(f"cells[{position}].native_gate", "must be nonempty when complete")
                for record_index, live_record in enumerate(cell.records):
                    public_record = StoredRecord(
                        source_spectrum_id=live_record.result.source_spectrum_id,
                        perturbation_id=live_record.result.perturbation_id,
                        output_spectrum_id=live_record.result.output.spectrum_id,
                        alpha=live_record.result.alpha,
                        alpha_float64_le_hex=live_record.alpha_float64_le_hex,
                        state_digest=live_record.result.state_digest,
                        sweep_config_sha256=sweep_config_sha256,
                        axis_behavior=live_record.result.axis_behavior.value,
                        axis_changed=live_record.result.axis_changed,
                        intensity_changed=live_record.result.intensity_changed,
                        axis_cm1=live_record.result.output.axis_cm1,
                        intensity=live_record.result.output.intensity,
                        diagnostics=live_record.result.diagnostics,
                    )
                    if np.shares_memory(source.spectrum.axis_cm1, live_record.result.output.axis_cm1):
                        raise PerturbedStoreError(
                            f"cells[{position}].records[{record_index}].axis_cm1",
                            "must not share memory with source spectrum",
                        )
                    if np.shares_memory(source.spectrum.intensity, live_record.result.output.intensity):
                        raise PerturbedStoreError(
                            f"cells[{position}].records[{record_index}].intensity",
                            "must not share memory with source spectrum",
                        )
                    materialized_records.append(
                        _ProjectedRecord(
                            public=public_record,
                            source_index=source_index,
                            source_record_id=projected_source.public.source_record_id,
                            source_dataset_id=projected_source.source_dataset_id,
                            sample_id=projected_source.public.sample_id,
                            class_label=projected_source.public.class_label,
                            mineral_name=projected_source.public.mineral_name,
                            source_selection_rank=projected_source.selection_rank,
                            axis_id=_float64_axis_id(public_record.axis_cm1),
                        )
                    )
                    record_snapshots.append(
                        (
                            f"cells[{position}].records[{record_index}].axis_cm1",
                            live_record.result.output.axis_cm1,
                            np.array(live_record.result.output.axis_cm1, copy=True),
                        )
                    )
                    record_snapshots.append(
                        (
                            f"cells[{position}].records[{record_index}].intensity",
                            live_record.result.output.intensity,
                            np.array(live_record.result.output.intensity, copy=True),
                        )
                    )
                    public_records.append(public_record)
            public_cell = StoredCell(
                source_spectrum_id=projected_source.public.source_spectrum_id,
                perturbation_id=perturbation_id,
                status=status,
                reason_code=cell.evidence.reason_code,
                state_digest=state_digest,
                records=tuple(public_records),
                native_gate=cell.evidence.native_gate,
            )
            projected_cells.append(
                _ProjectedCell(
                    public=public_cell,
                    source_index=source_index,
                    scientific_config_sha256=scientific_config_sha256,
                    sweep_config_sha256=sweep_config_sha256,
                    exception_type=cell.evidence.exception_type,
                    exception_path=cell.evidence.exception_path,
                    exception_message=cell.evidence.exception_message,
                )
            )
            position += 1
    return _ProjectedShard(
        sources=tuple(projected_sources),
        cells=tuple(projected_cells),
        materialized_records=tuple(materialized_records),
        source_snapshots=tuple(source_snapshots),
        record_snapshots=tuple(record_snapshots),
    )


def _build_axis_registry(projected: _ProjectedShard) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    registry: dict[str, np.ndarray] = {}
    for source in projected.sources:
        registry[source.axis_id] = source.public.axis_cm1
    for record in projected.materialized_records:
        existing = registry.get(record.axis_id)
        if existing is None:
            registry[record.axis_id] = record.public.axis_cm1
        elif not np.array_equal(existing, record.public.axis_cm1):
            raise PerturbedStoreError(
                "axis_id",
                "same axis ID resolved to different float64 bytes",
            )
    sorted_ids = tuple(sorted(registry))
    return registry, sorted_ids


def _source_row(
    source: _ProjectedSource,
    *,
    axis_index: int,
    intensity_offset: int,
) -> dict[str, object]:
    return {
        "axis_id": source.axis_id,
        "axis_index": axis_index,
        "class_label": source.public.class_label,
        "intensity_length": source.public.intensity.size,
        "intensity_offset": intensity_offset,
        "mineral_name": source.public.mineral_name,
        "normalized_axis_float64_sha256": source.normalized_axis_float64_sha256,
        "normalized_intensity_float64_sha256": source.normalized_intensity_float64_sha256,
        "original_axis_orientation": source.original_axis_orientation,
        "provenance": source.public.provenance,
        "sample_id": source.public.sample_id,
        "selection_axis_id": source.selection_axis_id,
        "selection_rank": source.selection_rank,
        "source_axis_float32_sha256": source.source_axis_float32_sha256,
        "source_dataset_id": source.source_dataset_id,
        "source_intensity_float32_sha256": source.source_intensity_float32_sha256,
        "source_record_id": source.public.source_record_id,
        "source_spectrum_id": source.public.source_spectrum_id,
        "source_index": source.source_index,
    }


def _record_row(
    record: _ProjectedRecord,
    *,
    intensity_offset: int,
    run_id: str,
) -> dict[str, object]:
    return {
        "alpha": record.public.alpha,
        "alpha_float64_le_hex": record.public.alpha_float64_le_hex,
        "axis_behavior": record.public.axis_behavior,
        "axis_changed": record.public.axis_changed,
        "axis_id": record.axis_id,
        "axis_length": record.public.axis_cm1.size,
        "class_label": record.class_label,
        "diagnostics": record.public.diagnostics,
        "intensity_changed": record.public.intensity_changed,
        "intensity_length": record.public.intensity.size,
        "intensity_offset": intensity_offset,
        "mineral_name": record.mineral_name,
        "output_spectrum_id": record.public.output_spectrum_id,
        "perturbation_id": record.public.perturbation_id,
        "run_id": run_id,
        "sample_id": record.sample_id,
        "source_dataset_id": record.source_dataset_id,
        "source_index": record.source_index,
        "source_record_id": record.source_record_id,
        "source_selection_rank": record.source_selection_rank,
        "source_spectrum_id": record.public.source_spectrum_id,
        "state_digest": record.public.state_digest,
        "sweep_config_sha256": record.public.sweep_config_sha256,
    }


def _cell_row(
    cell: _ProjectedCell,
    *,
    source_row: Mapping[str, object],
) -> dict[str, object]:
    output_ids = None
    dependency = None
    native_gate: Mapping[str, JsonValue] | None = None
    expected_output_count = 0
    actual_output_count = 0
    reason_code = cell.public.reason_code
    exception_type = cell.exception_type
    exception_path = cell.exception_path
    exception_message = cell.exception_message
    if cell.public.status == "complete":
        expected_output_count = len(_ALPHA_ORDER)
        actual_output_count = len(cell.public.records)
        output_ids = [record.output_spectrum_id for record in cell.public.records]
        native_gate = cell.public.native_gate
        reason_code = None
        exception_type = None
        exception_path = None
        exception_message = None
    elif cell.public.perturbation_id in _DEFERRED_IDS:
        dependency = _DEFERRED_DEPENDENCY
        reason_code = _DEFERRED_REASON_CODE
        exception_type = None
        exception_path = None
        exception_message = None
    elif cell.public.status == "failed":
        expected_output_count = len(_ALPHA_ORDER)
    return {
        "actual_output_count": actual_output_count,
        "dependency": dependency,
        "exception_message": exception_message,
        "exception_path": exception_path,
        "exception_type": exception_type,
        "expected_output_count": expected_output_count,
        "native_gate": native_gate,
        "output_spectrum_ids": output_ids,
        "perturbation_id": cell.public.perturbation_id,
        "reason_code": reason_code,
        "run_id": None,  # populated later
        "scientific_config_sha256": cell.scientific_config_sha256,
        "source": source_row,
        "state_digest": cell.public.state_digest,
        "status": cell.public.status,
        "sweep_config_sha256": cell.sweep_config_sha256,
    }


def _build_rows(
    projected: _ProjectedShard,
    *,
    run_id: str,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
]:
    axis_registry, sorted_axis_ids = _build_axis_registry(projected)
    axis_index_by_id = {axis_id: index for index, axis_id in enumerate(sorted_axis_ids)}
    source_rows_by_index: dict[int, dict[str, object]] = {}
    source_arrays = tuple(source.public.intensity for source in projected.sources)
    source_offsets = [0]
    for source in projected.sources:
        source_rows_by_index[source.source_index] = _source_row(
            source,
            axis_index=axis_index_by_id[source.axis_id],
            intensity_offset=source_offsets[-1],
        )
        source_offsets.append(source_offsets[-1] + source.public.intensity.size)
    record_rows: list[dict[str, object]] = []
    record_arrays: list[np.ndarray] = []
    record_offsets = [0]
    for record in projected.materialized_records:
        record_rows.append(
            _record_row(
                record,
                intensity_offset=record_offsets[-1],
                run_id=run_id,
            )
        )
        record_arrays.append(record.public.intensity)
        record_offsets.append(record_offsets[-1] + record.public.intensity.size)
    cells_rows: list[dict[str, object]] = []
    for cell in projected.cells:
        row = _cell_row(
            cell,
            source_row=source_rows_by_index[cell.source_index],
        )
        row["run_id"] = run_id
        cells_rows.append(row)
    return (
        tuple(cells_rows),
        tuple(record_rows),
        tuple(axis_registry[axis_id] for axis_id in sorted_axis_ids),
        source_arrays,
        tuple(record_arrays),
    )


def _write_hdf5(
    path: Path,
    *,
    run_id: str,
    shard_index: int,
    source_rows: Sequence[Mapping[str, object]],
    record_rows: Sequence[Mapping[str, object]],
    axes: Sequence[np.ndarray],
    source_arrays: Sequence[np.ndarray],
    record_arrays: Sequence[np.ndarray],
) -> None:
    axis_ids = tuple(_float64_axis_id(axis) for axis in axes)
    axis_offsets = [0]
    for axis in axes:
        axis_offsets.append(axis_offsets[-1] + axis.size)
    source_offsets = [0]
    for values in source_arrays:
        source_offsets.append(source_offsets[-1] + values.size)
    record_offsets = [0]
    for values in record_arrays:
        record_offsets.append(record_offsets[-1] + values.size)
    with h5py.File(path, "w", track_order=False) as handle:
        handle.attrs["schema_version"] = SHARD_SCHEMA_VERSION
        handle.attrs["run_id"] = run_id
        handle.attrs["shard_index"] = np.uint64(shard_index)
        axes_group = handle.create_group("axes", track_order=False, track_times=False)
        sources_group = handle.create_group("sources", track_order=False, track_times=False)
        records_group = handle.create_group("records", track_order=False, track_times=False)
        axes_group.create_dataset(
            "ids",
            data=np.asarray(axis_ids, dtype="S64"),
            **_dataset_options(max(1, len(axis_ids))),
        )
        axes_group.create_dataset(
            "offsets",
            data=np.asarray(axis_offsets, dtype="<u8"),
            **_dataset_options(len(axis_offsets)),
        )
        axes_group.create_dataset(
            "values",
            data=np.concatenate([np.asarray(axis, dtype="<f8") for axis in axes]) if axes else np.zeros(0, dtype="<f8"),
            **_dataset_options(max(1, axis_offsets[-1])),
        )
        sources_group.create_dataset(
            "axis_index",
            data=np.asarray([row["axis_index"] for row in source_rows], dtype="<u4"),
            **_dataset_options(max(1, len(source_rows))),
        )
        sources_group.create_dataset(
            "intensity_offsets",
            data=np.asarray(source_offsets, dtype="<u8"),
            **_dataset_options(len(source_offsets)),
        )
        sources_group.create_dataset(
            "intensity_values",
            data=np.concatenate([np.asarray(values, dtype="<f8") for values in source_arrays]) if source_arrays else np.zeros(0, dtype="<f8"),
            **_dataset_options(max(1, source_offsets[-1])),
        )
        records_group.create_dataset(
            "intensity_offsets",
            data=np.asarray(record_offsets, dtype="<u8"),
            **_dataset_options(len(record_offsets)),
        )
        record_values = (
            np.concatenate([np.asarray(values, dtype="<f8") for values in record_arrays])
            if record_arrays
            else np.zeros(0, dtype="<f8")
        )
        if record_arrays:
            records_group.create_dataset(
                "intensity_values",
                data=record_values,
                **_dataset_options(record_offsets[-1]),
            )
        else:
            records_group.create_dataset(
                "intensity_values",
                data=record_values,
                track_times=False,
            )


def _receipt_payload(receipt: ShardReceipt) -> dict[str, object]:
    return {
        "schema_version": receipt.schema_version,
        "run_id": receipt.run_id,
        "shard_index": receipt.shard_index,
        "source_count": receipt.source_count,
        "cell_count": receipt.cell_count,
        "record_count": receipt.record_count,
        "source_ids_sha256": receipt.source_ids_sha256,
        "logical_content_sha256": receipt.logical_content_sha256,
        "files": {
            key: {
                "byte_count": value.byte_count,
                "sha256": value.sha256,
            }
            for key, value in receipt.files.items()
        },
    }


def _recheck_live_snapshots(snapshots: Sequence[tuple[str, np.ndarray, np.ndarray]]) -> None:
    for path, live, snapshot in snapshots:
        if live.shape != snapshot.shape or live.dtype != snapshot.dtype or not np.array_equal(live, snapshot):
            raise PerturbedStoreError(path, "caller array changed during write")


def write_perturbed_shard(
    payload: Phase1ShardPayload,
    output_dir: Path,
    *,
    run_id: str,
    scientific_config_sha256: str,
    sweep_config_sha256: str,
) -> ShardReceipt:
    run_id = _safe_run_id(run_id)
    _lower_hex("scientific_config_sha256", scientific_config_sha256)
    _lower_hex("sweep_config_sha256", sweep_config_sha256)
    output_dir = Path(output_dir)
    if output_dir.name != f"{payload.shard_index:05d}":
        raise PerturbedStoreError("output_dir", "name must equal zero-padded shard index")
    if os.path.lexists(output_dir):
        raise PerturbedStoreError("output_dir", "target must not already exist")
    parent = output_dir.parent
    if os.path.lexists(parent):
        if parent.is_symlink() or not parent.is_dir():
            raise PerturbedStoreError("output_dir.parent", "must be a non-symlink directory")
    else:
        parent.mkdir(parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise PerturbedStoreError("output_dir.parent", "must be a non-symlink directory")
    projected = _validate_payload(
        payload,
        scientific_config_sha256=scientific_config_sha256,
        sweep_config_sha256=sweep_config_sha256,
    )
    cells_rows, record_rows, axes, source_arrays, record_arrays = _build_rows(projected, run_id=run_id)
    cells_jsonl_bytes = canonical_jsonl_bytes(cells_rows)
    records_jsonl_bytes = canonical_jsonl_bytes(record_rows)
    logical_content_sha256 = _logical_content_digest(
        cells_jsonl_bytes=cells_jsonl_bytes,
        records_jsonl_bytes=records_jsonl_bytes,
        axes=axes,
        sources=source_arrays,
        records=record_arrays,
    )
    source_ids_sha256 = _source_ids_digest(
        tuple(source.public.source_spectrum_id for source in projected.sources)
    )
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=parent))
    staged_dir = temp_root / output_dir.name
    renamed = False
    try:
        staged_dir.mkdir()
        _write_binary_file(staged_dir / "cells.jsonl", cells_jsonl_bytes)
        _write_binary_file(staged_dir / "records.jsonl", records_jsonl_bytes)
        _write_hdf5(
            staged_dir / "arrays.h5",
            run_id=run_id,
            shard_index=payload.shard_index,
            source_rows=tuple(row["source"] for row in cells_rows[::12]),
            record_rows=record_rows,
            axes=axes,
            source_arrays=source_arrays,
            record_arrays=record_arrays,
        )
        _fsync_path(staged_dir / "arrays.h5")
        _fsync_path(staged_dir / "cells.jsonl")
        _fsync_path(staged_dir / "records.jsonl")
        _fsync_path(staged_dir)
        files = MappingProxyType(
            {
                name: ArtifactIdentity(
                    byte_count=(staged_dir / name).stat().st_size,
                    sha256=_sha256_file(staged_dir / name),
                )
                for name in ("arrays.h5", "cells.jsonl", "records.jsonl")
            }
        )
        receipt = ShardReceipt(
            schema_version=SHARD_SCHEMA_VERSION,
            run_id=run_id,
            shard_index=payload.shard_index,
            source_count=len(projected.sources),
            cell_count=len(projected.cells),
            record_count=len(projected.materialized_records),
            source_ids_sha256=source_ids_sha256,
            logical_content_sha256=logical_content_sha256,
            files=files,
        )
        _write_binary_file(
            staged_dir / "receipt.json",
            canonical_json_bytes(_receipt_payload(receipt)),
        )
        _fsync_path(staged_dir / "receipt.json")
        _fsync_path(staged_dir)
        read_perturbed_shard(staged_dir)
        _recheck_live_snapshots(projected.source_snapshots)
        _recheck_live_snapshots(projected.record_snapshots)
        if os.path.lexists(output_dir):
            raise PerturbedStoreError("output_dir", "target appeared during write")
        os.replace(staged_dir, output_dir)
        renamed = True
        _fsync_path(parent)
        temp_root.rmdir()
        return receipt
    except BaseException:
        if not renamed:
            shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _read_canonical_json(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise PerturbedStoreError(path.name, str(error)) from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PerturbedStoreError(path.name, str(error)) from error
    if not isinstance(value, Mapping):
        raise PerturbedStoreError(path.name, "must be a JSON object")
    if canonical_json_bytes(value) != raw:
        raise PerturbedStoreError(path.name, "must be canonical JSON with trailing newline")
    return value


def _read_canonical_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    try:
        with path.open("rb") as stream:
            for index, line in enumerate(stream):
                if not line.endswith(b"\n") or line == b"\n":
                    raise PerturbedStoreError(path.name, "must be canonical JSONL")
                try:
                    value = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PerturbedStoreError(f"{path.name}[{index}]", str(error)) from error
                if not isinstance(value, Mapping):
                    raise PerturbedStoreError(f"{path.name}[{index}]", "row must be a JSON object")
                if canonical_json_bytes(value) != line:
                    raise PerturbedStoreError(f"{path.name}[{index}]", "row must be canonical JSON")
                rows.append(value)
    except OSError as error:
        raise PerturbedStoreError(path.name, str(error)) from error
    return tuple(rows)


def _validate_receipt(receipt_document: Mapping[str, object]) -> ShardReceipt:
    expected_keys = (
        "cell_count",
        "files",
        "logical_content_sha256",
        "record_count",
        "run_id",
        "schema_version",
        "shard_index",
        "source_count",
        "source_ids_sha256",
    )
    if tuple(receipt_document.keys()) != expected_keys:
        raise PerturbedStoreError("receipt.json", "must use the exact receipt key set")
    files_object = receipt_document["files"]
    if not isinstance(files_object, Mapping):
        raise PerturbedStoreError("receipt.json.files", "must be a mapping")
    expected_file_keys = ("arrays.h5", "cells.jsonl", "records.jsonl")
    if tuple(files_object.keys()) != expected_file_keys:
        raise PerturbedStoreError("receipt.json.files", "must contain the exact file key set")
    files: dict[str, ArtifactIdentity] = {}
    for key in expected_file_keys:
        value = files_object[key]
        if not isinstance(value, Mapping):
            raise PerturbedStoreError(f"receipt.json.files.{key}", "must be a mapping")
        byte_count = _nonnegative_int(
            f"receipt.json.files.{key}.byte_count",
            value.get("byte_count"),
        )
        sha256 = _lower_hex(
            f"receipt.json.files.{key}.sha256",
            value.get("sha256"),
        )
        if set(value.keys()) != {"byte_count", "sha256"}:
            raise PerturbedStoreError(
                f"receipt.json.files.{key}",
                "must use the exact byte_count/sha256 key set",
            )
        files[key] = ArtifactIdentity(byte_count=byte_count, sha256=sha256)
    return ShardReceipt(
        schema_version=receipt_document["schema_version"],
        run_id=receipt_document["run_id"],
        shard_index=_nonnegative_int("receipt.json.shard_index", receipt_document["shard_index"]),
        source_count=_positive_int("receipt.json.source_count", receipt_document["source_count"]),
        cell_count=_positive_int("receipt.json.cell_count", receipt_document["cell_count"]),
        record_count=_nonnegative_int("receipt.json.record_count", receipt_document["record_count"]),
        source_ids_sha256=_lower_hex("receipt.json.source_ids_sha256", receipt_document["source_ids_sha256"]),
        logical_content_sha256=_lower_hex("receipt.json.logical_content_sha256", receipt_document["logical_content_sha256"]),
        files=files,
    )


def _validate_lexical_inventory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise PerturbedStoreError("path", str(error)) from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise PerturbedStoreError("path", "must be a non-symlink directory")
    expected_names = tuple(sorted(SHARD_FILES))
    for name in expected_names:
        entry = path / name
        if not os.path.lexists(entry):
            raise PerturbedStoreError("path", "must contain exactly arrays.h5/cells.jsonl/records.jsonl/receipt.json")
        try:
            entry_info = entry.lstat()
        except OSError as error:
            raise PerturbedStoreError(name, str(error)) from error
        if stat.S_ISLNK(entry_info.st_mode) or not stat.S_ISREG(entry_info.st_mode):
            raise PerturbedStoreError(name, "must be a non-symlink regular file")
    names = tuple(sorted(entry.name for entry in path.iterdir()))
    if expected_names != names:
        raise PerturbedStoreError("path", "must contain exactly arrays.h5/cells.jsonl/records.jsonl/receipt.json")


def _validate_file_identities(path: Path, receipt: ShardReceipt) -> None:
    for name, identity in receipt.files.items():
        file_path = path / name
        if file_path.stat().st_size != identity.byte_count:
            raise PerturbedStoreError(name, "byte_count mismatch")
        if _sha256_file(file_path) != identity.sha256:
            raise PerturbedStoreError(name, "sha256 mismatch")


def _load_hdf5_arrays(
    path: Path,
    *,
    run_id: str,
    shard_index: int,
    source_count: int,
    record_count: int,
) -> tuple[
    tuple[str, ...],
    dict[str, np.ndarray],
    tuple[np.ndarray, ...],
    tuple[np.ndarray, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    try:
        with h5py.File(path, "r") as handle:
            for name in ("axes", "sources", "records"):
                if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                    raise PerturbedStoreError(f"arrays.h5.{name}", "must be a hard link")
            if set(handle.keys()) != {"axes", "sources", "records"}:
                raise PerturbedStoreError("arrays.h5", "must contain exact root groups axes/sources/records")
            if set(handle.attrs.keys()) != {"schema_version", "run_id", "shard_index"}:
                raise PerturbedStoreError("arrays.h5", "must contain exact root attrs schema_version/run_id/shard_index")
            if handle.attrs["schema_version"] != SHARD_SCHEMA_VERSION:
                raise PerturbedStoreError("arrays.h5.schema_version", "must match shard schema version")
            if handle.attrs["run_id"] != run_id:
                raise PerturbedStoreError("arrays.h5.run_id", "must match receipt run_id")
            raw_shard_index = np.asarray(handle.attrs["shard_index"])
            if raw_shard_index.shape != () or raw_shard_index.dtype.kind != "u" or raw_shard_index.dtype.itemsize != 8:
                raise PerturbedStoreError("arrays.h5.shard_index", "must be stored as uint64")
            if int(raw_shard_index.item()) != shard_index:
                raise PerturbedStoreError("arrays.h5.shard_index", "must match receipt shard_index")
            axes_group = handle["axes"]
            sources_group = handle["sources"]
            records_group = handle["records"]
            for name, group, expected in (
                ("axes", axes_group, {"ids", "offsets", "values"}),
                ("sources", sources_group, {"axis_index", "intensity_offsets", "intensity_values"}),
                ("records", records_group, {"intensity_offsets", "intensity_values"}),
            ):
                if set(group.keys()) != expected:
                    raise PerturbedStoreError(f"arrays.h5.{name}", "must contain the exact dataset inventory")
                if len(group.attrs) != 0:
                    raise PerturbedStoreError(f"arrays.h5.{name}", "groups must not carry attrs")
                for child_name in expected:
                    if not isinstance(group.get(child_name, getlink=True), h5py.HardLink):
                        raise PerturbedStoreError(f"arrays.h5.{name}.{child_name}", "must be a hard link")
            datasets = {
                "axes.ids": axes_group["ids"],
                "axes.offsets": axes_group["offsets"],
                "axes.values": axes_group["values"],
                "sources.axis_index": sources_group["axis_index"],
                "sources.intensity_offsets": sources_group["intensity_offsets"],
                "sources.intensity_values": sources_group["intensity_values"],
                "records.intensity_offsets": records_group["intensity_offsets"],
                "records.intensity_values": records_group["intensity_values"],
            }
            expected_dtypes = {
                "axes.ids": np.dtype("S64"),
                "axes.offsets": np.dtype("<u8"),
                "axes.values": np.dtype("<f8"),
                "sources.axis_index": np.dtype("<u4"),
                "sources.intensity_offsets": np.dtype("<u8"),
                "sources.intensity_values": np.dtype("<f8"),
                "records.intensity_offsets": np.dtype("<u8"),
                "records.intensity_values": np.dtype("<f8"),
            }
            for name, dataset in datasets.items():
                if dataset.ndim != 1:
                    raise PerturbedStoreError(f"arrays.h5.{name}", "datasets must be one-dimensional")
                if dataset.dtype != expected_dtypes[name]:
                    raise PerturbedStoreError(f"arrays.h5.{name}", "dtype mismatch")
                if len(dataset.attrs) != 0:
                    raise PerturbedStoreError(f"arrays.h5.{name}", "datasets must not carry attrs")
                if bool(dataset.is_virtual):
                    raise PerturbedStoreError(f"arrays.h5.{name}", "virtual datasets are not allowed")
                external_storage = dataset.external
                if external_storage not in (None, ()):
                    raise PerturbedStoreError(f"arrays.h5.{name}", "external storage is not allowed")
                is_zero_record_values = (
                    name == "records.intensity_values"
                    and record_count == 0
                    and dataset.shape == (0,)
                )
                if is_zero_record_values:
                    if dataset.compression is not None or dataset.compression_opts is not None:
                        raise PerturbedStoreError(f"arrays.h5.{name}", "zero-record values must be stored uncompressed")
                    if dataset.chunks is not None:
                        raise PerturbedStoreError(f"arrays.h5.{name}", "zero-record values must be stored contiguous")
                else:
                    if dataset.compression != "gzip" or dataset.compression_opts != 1:
                        raise PerturbedStoreError(f"arrays.h5.{name}", "must use gzip level 1")
                    if bool(dataset.shuffle):
                        raise PerturbedStoreError(f"arrays.h5.{name}", "shuffle must be disabled")
                    if bool(dataset.fletcher32):
                        raise PerturbedStoreError(f"arrays.h5.{name}", "fletcher32 must be disabled")
                    if dataset.chunks is None:
                        raise PerturbedStoreError(f"arrays.h5.{name}", "must use explicit chunks")
                if dataset.maxshape != dataset.shape:
                    raise PerturbedStoreError(f"arrays.h5.{name}", "must be non-resizable")
            axis_ids = tuple(item.decode("ascii") for item in datasets["axes.ids"][...])
            if tuple(sorted(axis_ids)) != axis_ids:
                raise PerturbedStoreError("arrays.h5.axes.ids", "axis IDs must be sorted")
            axis_offsets = tuple(int(value) for value in datasets["axes.offsets"][...])
            source_offsets = tuple(int(value) for value in datasets["sources.intensity_offsets"][...])
            record_offsets = tuple(int(value) for value in datasets["records.intensity_offsets"][...])
            if len(axis_offsets) != len(axis_ids) + 1:
                raise PerturbedStoreError("arrays.h5.axes.offsets", "length must equal axis count + 1")
            if len(source_offsets) != source_count + 1:
                raise PerturbedStoreError("arrays.h5.sources.intensity_offsets", "length must equal source count + 1")
            if len(record_offsets) != record_count + 1:
                raise PerturbedStoreError("arrays.h5.records.intensity_offsets", "length must equal record count + 1")
            for name, offsets, values_length in (
                ("arrays.h5.axes.offsets", axis_offsets, datasets["axes.values"].shape[0]),
                ("arrays.h5.sources.intensity_offsets", source_offsets, datasets["sources.intensity_values"].shape[0]),
                ("arrays.h5.records.intensity_offsets", record_offsets, datasets["records.intensity_values"].shape[0]),
            ):
                if offsets[0] != 0:
                    raise PerturbedStoreError(name, "must begin at zero")
                if any(right < left for left, right in zip(offsets, offsets[1:])):
                    raise PerturbedStoreError(name, "must be nondecreasing")
                if offsets[-1] != values_length:
                    raise PerturbedStoreError(name, "must end at the values length")
            axis_by_id: dict[str, np.ndarray] = {}
            for axis_id, start, end in zip(axis_ids, axis_offsets, axis_offsets[1:]):
                axis = _readonly_f8_copy("arrays.h5.axes.values", np.asarray(datasets["axes.values"][start:end], dtype="<f8"))
                if _float64_axis_id(axis) != axis_id:
                    raise PerturbedStoreError("arrays.h5.axes.ids", "axis ID mismatch")
                axis_by_id[axis_id] = axis
            axis_index_values = tuple(int(value) for value in datasets["sources.axis_index"][...])
            if len(axis_index_values) != source_count:
                raise PerturbedStoreError("arrays.h5.sources.axis_index", "count must equal source count")
            source_arrays: list[np.ndarray] = []
            for axis_index, start, end in zip(axis_index_values, source_offsets, source_offsets[1:]):
                if axis_index < 0 or axis_index >= len(axis_ids):
                    raise PerturbedStoreError("arrays.h5.sources.axis_index", "must reference a valid axis row")
                values = _readonly_f8_copy(
                    "arrays.h5.sources.intensity_values",
                    np.asarray(datasets["sources.intensity_values"][start:end], dtype="<f8"),
                )
                if values.size != axis_by_id[axis_ids[axis_index]].size:
                    raise PerturbedStoreError("arrays.h5.sources.intensity_values", "length must match referenced axis length")
                source_arrays.append(values)
            record_arrays: list[np.ndarray] = []
            for start, end in zip(record_offsets, record_offsets[1:]):
                values = np.asarray(datasets["records.intensity_values"][start:end], dtype="<f8")
                if values.size == 0:
                    raise PerturbedStoreError("arrays.h5.records.intensity_values", "stored record intensities must be nonempty")
                record_arrays.append(_readonly_f8_copy("arrays.h5.records.intensity_values", values))
            return (
                axis_ids,
                axis_by_id,
                tuple(source_arrays),
                tuple(record_arrays),
                axis_index_values,
                source_offsets,
                record_offsets,
            )
    except OSError as error:
        raise PerturbedStoreError("arrays.h5", str(error)) from error


def _validate_source_row(row: Mapping[str, object], *, index: int) -> Mapping[str, object]:
    if tuple(row.keys()) != _EXPECTED_SOURCE_KEYS_CANONICAL:
        raise PerturbedStoreError(f"cells.jsonl[{index}].source", "must use the exact source key set")
    _lower_hex(f"cells.jsonl[{index}].source.axis_id", row["axis_id"])
    _nonnegative_int(f"cells.jsonl[{index}].source.axis_index", row["axis_index"])
    _nonnegative_int(f"cells.jsonl[{index}].source.class_label", row["class_label"])
    _positive_int(f"cells.jsonl[{index}].source.intensity_length", row["intensity_length"])
    _nonnegative_int(f"cells.jsonl[{index}].source.intensity_offset", row["intensity_offset"])
    _nonempty_string(f"cells.jsonl[{index}].source.mineral_name", row["mineral_name"])
    _lower_hex(f"cells.jsonl[{index}].source.normalized_axis_float64_sha256", row["normalized_axis_float64_sha256"])
    _lower_hex(f"cells.jsonl[{index}].source.normalized_intensity_float64_sha256", row["normalized_intensity_float64_sha256"])
    _nonempty_string(f"cells.jsonl[{index}].source.original_axis_orientation", row["original_axis_orientation"])
    _freeze_mapping(f"cells.jsonl[{index}].source.provenance", row["provenance"])
    _nonempty_string(f"cells.jsonl[{index}].source.sample_id", row["sample_id"])
    _nonempty_string(f"cells.jsonl[{index}].source.selection_axis_id", row["selection_axis_id"])
    _nonnegative_int(f"cells.jsonl[{index}].source.selection_rank", row["selection_rank"])
    _lower_hex(f"cells.jsonl[{index}].source.source_axis_float32_sha256", row["source_axis_float32_sha256"])
    dataset_id = _split_source_spectrum_id(
        f"cells.jsonl[{index}].source.source_spectrum_id",
        row["source_spectrum_id"],
        row["source_record_id"],
    )
    if row["source_dataset_id"] != dataset_id:
        raise PerturbedStoreError(f"cells.jsonl[{index}].source.source_dataset_id", "must be derived from source_spectrum_id")
    _lower_hex(f"cells.jsonl[{index}].source.source_intensity_float32_sha256", row["source_intensity_float32_sha256"])
    if _nonnegative_int(f"cells.jsonl[{index}].source.source_index", row["source_index"]) != index // 12:
        raise PerturbedStoreError(f"cells.jsonl[{index}].source.source_index", "must match source-major order")
    return row


def _source_rows_from_cells(cell_rows: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    if len(cell_rows) % len(_CELL_ORDER) != 0:
        raise PerturbedStoreError("cells.jsonl", "must contain exactly twelve rows per source")
    unique_rows: list[Mapping[str, object]] = []
    previous_rank: int | None = None
    for source_index in range(len(cell_rows) // len(_CELL_ORDER)):
        chunk = cell_rows[source_index * len(_CELL_ORDER):(source_index + 1) * len(_CELL_ORDER)]
        source_row = _validate_source_row(chunk[0]["source"], index=source_index * len(_CELL_ORDER))
        if previous_rank is not None and int(source_row["selection_rank"]) <= previous_rank:
            raise PerturbedStoreError("cells.jsonl.source.selection_rank", "must be strictly increasing")
        previous_rank = int(source_row["selection_rank"])
        for offset, row in enumerate(chunk):
            if tuple(row["source"].items()) != tuple(source_row.items()):
                raise PerturbedStoreError(f"cells.jsonl[{source_index * len(_CELL_ORDER) + offset}].source", "must repeat the exact source object")
            if row["perturbation_id"] != _CELL_ORDER[offset]:
                raise PerturbedStoreError(f"cells.jsonl[{source_index * len(_CELL_ORDER) + offset}].perturbation_id", "must follow exact P01-P12 order")
        unique_rows.append(source_row)
    return tuple(unique_rows)


def _validate_cell_rows(
    cell_rows: Sequence[Mapping[str, object]],
    *,
    receipt: ShardReceipt,
) -> tuple[Mapping[str, object], ...]:
    if len(cell_rows) != receipt.cell_count:
        raise PerturbedStoreError("cells.jsonl", "cell_count mismatch")
    shard_scientific_config_sha256: str | None = None
    shard_sweep_config_sha256: str | None = None
    for index, row in enumerate(cell_rows):
        if tuple(row.keys()) != _EXPECTED_CELL_KEYS_CANONICAL:
            raise PerturbedStoreError(f"cells.jsonl[{index}]", "must use the exact cell key set")
        if row["run_id"] != receipt.run_id:
            raise PerturbedStoreError(f"cells.jsonl[{index}].run_id", "must match receipt run_id")
        scientific_config_sha256 = _lower_hex(
            f"cells.jsonl[{index}].scientific_config_sha256",
            row["scientific_config_sha256"],
        )
        sweep_config_sha256 = _lower_hex(
            f"cells.jsonl[{index}].sweep_config_sha256",
            row["sweep_config_sha256"],
        )
        if shard_scientific_config_sha256 is None:
            shard_scientific_config_sha256 = scientific_config_sha256
        elif scientific_config_sha256 != shard_scientific_config_sha256:
            raise PerturbedStoreError(
                f"cells.jsonl[{index}].scientific_config_sha256",
                "must remain constant within the shard",
            )
        if shard_sweep_config_sha256 is None:
            shard_sweep_config_sha256 = sweep_config_sha256
        elif sweep_config_sha256 != shard_sweep_config_sha256:
            raise PerturbedStoreError(
                f"cells.jsonl[{index}].sweep_config_sha256",
                "must remain constant within the shard",
            )
        _nonempty_string(f"cells.jsonl[{index}].perturbation_id", row["perturbation_id"])
        status = _nonempty_string(f"cells.jsonl[{index}].status", row["status"])
        if status not in {"complete", "not_applicable", "failed"}:
            raise PerturbedStoreError(f"cells.jsonl[{index}].status", "unsupported status")
        if row["state_digest"] is not None:
            _lower_hex(f"cells.jsonl[{index}].state_digest", row["state_digest"])
        _validate_source_row(row["source"], index=index)
        if status == "complete":
            if row["expected_output_count"] != len(_ALPHA_ORDER) or row["actual_output_count"] != len(_ALPHA_ORDER):
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "complete rows must report nine outputs")
            if not isinstance(row["output_spectrum_ids"], list) or len(row["output_spectrum_ids"]) != len(_ALPHA_ORDER):
                raise PerturbedStoreError(f"cells.jsonl[{index}].output_spectrum_ids", "must contain nine ordered output IDs")
            if len(set(row["output_spectrum_ids"])) != len(_ALPHA_ORDER):
                raise PerturbedStoreError(f"cells.jsonl[{index}].output_spectrum_ids", "must be unique")
            if row["reason_code"] is not None or row["dependency"] is not None:
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "complete rows must not carry reason/dependency")
            if any(row[name] is not None for name in ("exception_type", "exception_path", "exception_message")):
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "complete rows must not carry exception triplets")
            if not isinstance(row["native_gate"], Mapping) or not row["native_gate"]:
                raise PerturbedStoreError(f"cells.jsonl[{index}].native_gate", "complete rows must carry nonempty native_gate")
            if row["state_digest"] is None:
                raise PerturbedStoreError(f"cells.jsonl[{index}].state_digest", "complete rows must carry state_digest")
            _freeze_mapping(f"cells.jsonl[{index}].native_gate", row["native_gate"])
        elif row["perturbation_id"] in _DEFERRED_IDS:
            if status != "not_applicable":
                raise PerturbedStoreError(f"cells.jsonl[{index}].status", "deferred rows must be not_applicable")
            if row["reason_code"] != _DEFERRED_REASON_CODE:
                raise PerturbedStoreError(f"cells.jsonl[{index}].reason_code", "deferred rows must use the exact deferred reason")
            if row["dependency"] != _DEFERRED_DEPENDENCY:
                raise PerturbedStoreError(f"cells.jsonl[{index}].dependency", "deferred rows must use the exact dependency")
            if row["expected_output_count"] != 0 or row["actual_output_count"] != 0:
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "deferred rows must have zero outputs")
            if row["output_spectrum_ids"] is not None or row["native_gate"] is not None:
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "deferred rows must not carry outputs/native_gate")
            if any(row[name] is not None for name in ("exception_type", "exception_path", "exception_message")):
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "deferred rows must not carry exception triplets")
        elif status == "not_applicable":
            if row["expected_output_count"] != 0 or row["actual_output_count"] != 0:
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "not_applicable rows must have zero outputs")
            if not isinstance(row["reason_code"], str) or row["reason_code"] == "":
                raise PerturbedStoreError(f"cells.jsonl[{index}].reason_code", "core not_applicable rows must carry a reason")
            if row["dependency"] is not None or row["output_spectrum_ids"] is not None or row["native_gate"] is not None:
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "core not_applicable rows must not carry dependency/outputs/native_gate")
            exception_values = tuple(row[name] for name in ("exception_type", "exception_path", "exception_message"))
            if any(value is not None for value in exception_values) and not all(isinstance(value, str) and value != "" for value in exception_values):
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "exception triplet must be all present or all absent")
        else:
            if row["expected_output_count"] != len(_ALPHA_ORDER) or row["actual_output_count"] != 0:
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "failed rows must advertise nine expected and zero actual outputs")
            if row["reason_code"] is not None or row["dependency"] is not None or row["output_spectrum_ids"] is not None or row["native_gate"] is not None:
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "failed rows must not carry reason/dependency/outputs/native_gate")
            exception_values = tuple(row[name] for name in ("exception_type", "exception_path", "exception_message"))
            if not all(isinstance(value, str) and value != "" for value in exception_values):
                raise PerturbedStoreError(f"cells.jsonl[{index}]", "failed rows must carry a complete exception triplet")
    return _source_rows_from_cells(cell_rows)


def _validate_record_rows(
    record_rows: Sequence[Mapping[str, object]],
    *,
    receipt: ShardReceipt,
    cell_rows: Sequence[Mapping[str, object]],
    source_rows: Sequence[Mapping[str, object]],
) -> None:
    if len(record_rows) != receipt.record_count:
        raise PerturbedStoreError("records.jsonl", "record_count mismatch")
    source_row_by_index = {
        int(source_row["source_index"]): source_row
        for source_row in source_rows
    }
    expected_complete_cells = tuple(
        (cell_index, row)
        for cell_index, row in enumerate(cell_rows)
        if row["status"] == "complete"
    )
    expected_index = 0
    for _, cell_row in expected_complete_cells:
        source_row = source_row_by_index[int(cell_row["source"]["source_index"])]
        cell_state_digest = str(cell_row["state_digest"])
        expected_output_ids = tuple(cell_row["output_spectrum_ids"])
        for alpha, alpha_hex, expected_output_id in zip(
            _ALPHA_ORDER,
            _ALPHA_HEX_ORDER,
            expected_output_ids,
            strict=True,
        ):
            row = record_rows[expected_index]
            if tuple(row.keys()) != _EXPECTED_RECORD_KEYS_CANONICAL:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}]", "must use the exact record key set")
            if row["run_id"] != receipt.run_id:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].run_id", "must match receipt run_id")
            if row["source_index"] != source_row["source_index"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].source_index", "must follow source-major order")
            if row["perturbation_id"] != cell_row["perturbation_id"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].perturbation_id", "must follow canonical perturbation order")
            if not math.isclose(float(row["alpha"]), alpha, rel_tol=0.0, abs_tol=0.0):
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].alpha", "must follow the frozen alpha order")
            if row["alpha_float64_le_hex"] != alpha_hex:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].alpha_float64_le_hex", "must equal the alpha little-endian hex")
            if row["source_dataset_id"] != source_row["source_dataset_id"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].source_dataset_id", "must match the source row")
            if row["source_record_id"] != source_row["source_record_id"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].source_record_id", "must match the source row")
            if row["source_spectrum_id"] != source_row["source_spectrum_id"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].source_spectrum_id", "must match the source row")
            if row["sample_id"] != source_row["sample_id"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].sample_id", "must match the source row")
            if row["class_label"] != source_row["class_label"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].class_label", "must match the source row")
            if row["mineral_name"] != source_row["mineral_name"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].mineral_name", "must match the source row")
            if row["source_selection_rank"] != source_row["selection_rank"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].source_selection_rank", "must match the source row")
            _lower_hex(f"records.jsonl[{expected_index}].axis_id", row["axis_id"])
            _positive_int(f"records.jsonl[{expected_index}].axis_length", row["axis_length"])
            _positive_int(f"records.jsonl[{expected_index}].intensity_length", row["intensity_length"])
            _nonnegative_int(f"records.jsonl[{expected_index}].intensity_offset", row["intensity_offset"])
            if row["axis_length"] != row["intensity_length"]:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}]", "axis_length must equal intensity_length")
            if row["axis_behavior"] not in {"preserve", "transform"}:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].axis_behavior", "must be a supported enum value")
            if not isinstance(row["axis_changed"], bool) or not isinstance(row["intensity_changed"], bool):
                raise PerturbedStoreError(f"records.jsonl[{expected_index}]", "axis_changed/intensity_changed must be bool")
            if row["state_digest"] != cell_state_digest:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].state_digest", "must match the owning cell state_digest")
            if row["sweep_config_sha256"] != cell_row["sweep_config_sha256"]:
                raise PerturbedStoreError(
                    f"records.jsonl[{expected_index}].sweep_config_sha256",
                    "must match the owning cell sweep_config_sha256",
                )
            _lower_hex(f"records.jsonl[{expected_index}].state_digest", row["state_digest"])
            _lower_hex(f"records.jsonl[{expected_index}].sweep_config_sha256", row["sweep_config_sha256"])
            _freeze_mapping(f"records.jsonl[{expected_index}].diagnostics", row["diagnostics"])
            canonical_output_id = derive_perturbed_spectrum_id(
                row["source_spectrum_id"],
                row["perturbation_id"],
                float(row["alpha"]),
                row["state_digest"],
                row["sweep_config_sha256"],
            )
            if row["output_spectrum_id"] != canonical_output_id:
                raise PerturbedStoreError(f"records.jsonl[{expected_index}].output_spectrum_id", "must match the canonical perturbed spectrum ID")
            if row["output_spectrum_id"] != expected_output_id:
                raise PerturbedStoreError(
                    f"records.jsonl[{expected_index}].output_spectrum_id",
                    "must match the owning cell output_spectrum_ids order",
                )
            expected_index += 1
    if expected_index != len(record_rows):
        raise PerturbedStoreError("records.jsonl", "must contain rows for complete cells only")


def read_perturbed_shard(path: Path) -> StoredShard:
    path = Path(path)
    _validate_lexical_inventory(path)
    receipt_document = _read_canonical_json(path / "receipt.json")
    receipt = _validate_receipt(receipt_document)
    _validate_file_identities(path, receipt)
    cell_rows = _read_canonical_jsonl(path / "cells.jsonl")
    source_rows = _validate_cell_rows(cell_rows, receipt=receipt)
    record_rows = _read_canonical_jsonl(path / "records.jsonl")
    _validate_record_rows(
        record_rows,
        receipt=receipt,
        cell_rows=cell_rows,
        source_rows=source_rows,
    )
    axis_ids, axis_by_id, source_arrays, record_arrays, source_axis_indexes, source_offsets, record_offsets = _load_hdf5_arrays(
        path / "arrays.h5",
        run_id=receipt.run_id,
        shard_index=receipt.shard_index,
        source_count=receipt.source_count,
        record_count=receipt.record_count,
    )
    sources: list[StoredSource] = []
    for source_position, (source_row, source_array, axis_index) in enumerate(
        zip(source_rows, source_arrays, source_axis_indexes, strict=True)
    ):
        axis_id = str(source_row["axis_id"])
        if int(source_row["axis_index"]) != axis_index:
            raise PerturbedStoreError("cells.jsonl.source.axis_index", "must agree with arrays.h5")
        if axis_id != axis_ids[axis_index]:
            raise PerturbedStoreError("cells.jsonl.source.axis_id", "must agree with arrays.h5 axis_index")
        if int(source_row["intensity_offset"]) != source_offsets[source_position]:
            raise PerturbedStoreError("cells.jsonl.source.intensity_offset", "must agree with arrays.h5")
        axis = axis_by_id[axis_id]
        if source_array.size != axis.size:
            raise PerturbedStoreError("arrays.h5.sources.intensity_values", "source intensity length must match source axis length")
        if int(source_row["intensity_length"]) != source_array.size:
            raise PerturbedStoreError("cells.jsonl.source.intensity_length", "must agree with arrays.h5")
        if hashlib.sha256(float64_le_bytes(axis)).hexdigest() != source_row["normalized_axis_float64_sha256"]:
            raise PerturbedStoreError("cells.jsonl.source.normalized_axis_float64_sha256", "must match stored axis bytes")
        if hashlib.sha256(float64_le_bytes(source_array)).hexdigest() != source_row["normalized_intensity_float64_sha256"]:
            raise PerturbedStoreError("cells.jsonl.source.normalized_intensity_float64_sha256", "must match stored intensity bytes")
        sources.append(
            StoredSource(
                source_spectrum_id=source_row["source_spectrum_id"],
                source_record_id=source_row["source_record_id"],
                sample_id=source_row["sample_id"],
                class_label=source_row["class_label"],
                mineral_name=source_row["mineral_name"],
                axis_cm1=axis,
                intensity=source_array,
                provenance=source_row["provenance"],
            )
        )
    records_by_cell: dict[tuple[int, str], list[StoredRecord]] = {}
    for record_index, row in enumerate(record_rows):
        axis = axis_by_id[row["axis_id"]]
        values = record_arrays[record_index]
        if axis.size != values.size or axis.size != int(row["axis_length"]) or values.size != int(row["intensity_length"]):
            raise PerturbedStoreError(f"records.jsonl[{record_index}]", "row lengths must agree with arrays.h5")
        if int(row["intensity_offset"]) != record_offsets[record_index]:
            raise PerturbedStoreError(f"records.jsonl[{record_index}].intensity_offset", "must agree with arrays.h5")
        records_by_cell.setdefault((int(row["source_index"]), str(row["perturbation_id"])), []).append(
            StoredRecord(
                source_spectrum_id=row["source_spectrum_id"],
                perturbation_id=row["perturbation_id"],
                output_spectrum_id=row["output_spectrum_id"],
                alpha=row["alpha"],
                alpha_float64_le_hex=row["alpha_float64_le_hex"],
                state_digest=row["state_digest"],
                sweep_config_sha256=row["sweep_config_sha256"],
                axis_behavior=row["axis_behavior"],
                axis_changed=row["axis_changed"],
                intensity_changed=row["intensity_changed"],
                axis_cm1=axis,
                intensity=values,
                diagnostics=row["diagnostics"],
            )
        )
    cells: list[StoredCell] = []
    for index, row in enumerate(cell_rows):
        source_index = int(row["source"]["source_index"])
        perturbation_id = str(row["perturbation_id"])
        cell_records = tuple(records_by_cell.get((source_index, perturbation_id), ()))
        if row["status"] == "complete":
            expected_output_ids = tuple(row["output_spectrum_ids"])
            observed_output_ids = tuple(record.output_spectrum_id for record in cell_records)
            if observed_output_ids != expected_output_ids:
                raise PerturbedStoreError(f"cells.jsonl[{index}].output_spectrum_ids", "must match records.jsonl order")
        elif cell_records:
            raise PerturbedStoreError(f"cells.jsonl[{index}]", "non-complete rows must not own records")
        cells.append(
            StoredCell(
                source_spectrum_id=row["source"]["source_spectrum_id"],
                perturbation_id=perturbation_id,
                status=row["status"],
                reason_code=row["reason_code"],
                state_digest=row["state_digest"],
                records=cell_records,
                native_gate={} if row["native_gate"] is None else row["native_gate"],
            )
        )
    cells_bytes = (path / "cells.jsonl").read_bytes()
    records_bytes = (path / "records.jsonl").read_bytes()
    expected_source_ids_sha = _source_ids_digest(tuple(source.source_spectrum_id for source in sources))
    if expected_source_ids_sha != receipt.source_ids_sha256:
        raise PerturbedStoreError("receipt.json.source_ids_sha256", "source IDs digest mismatch")
    expected_logical_content_sha = _logical_content_digest(
        cells_jsonl_bytes=cells_bytes,
        records_jsonl_bytes=records_bytes,
        axes=tuple(axis_by_id[axis_id] for axis_id in axis_ids),
        sources=tuple(source.intensity for source in sources),
        records=tuple(record.intensity for cell in cells for record in cell.records),
    )
    if expected_logical_content_sha != receipt.logical_content_sha256:
        raise PerturbedStoreError("receipt.json.logical_content_sha256", "logical content digest mismatch")
    return StoredShard(
        run_id=receipt.run_id,
        shard_index=receipt.shard_index,
        sources=tuple(sources),
        cells=tuple(cells),
        receipt=receipt,
    )
