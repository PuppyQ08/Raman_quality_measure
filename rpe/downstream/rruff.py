from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from rpe.evaluation import Spectrum1D

SCHEMA_VERSION = "phase05-d5-protocol-v1"
EXPERIMENT_ID = "d5_rruff_raw_library_matching_sg11"
DATASET_ID = "rruff_raman_raw"
CONFIG_BYTES = 3855
CONFIG_SHA256 = (
    "94f59739a2e3583c6ae33ab5f4ab489cb1f26d3c9f8cdeb689a2448c9c01e009"
)
AUDIT_BYTES = 5111
AUDIT_SHA256 = (
    "a673d0d0c4a0c6536a8a6d3a31514d209607f31599cd0456ba12956dab49ff70"
)
AUDIT_RELATIVE_PATH = "reports/phase05/d5_step01_protocol_audit.json"
SEEDS = (0, 1, 2, 3, 4)
GRID_START_CM1 = 200.0
GRID_STOP_CM1 = 1800.0
GRID_STEP_CM1 = 2.0
GRID_POINT_COUNT = 801
MAX_IN_RANGE_NATIVE_GAP_CM1 = 3.0
EXPECTED_CLASS_COUNT = 681
EXPECTED_RECORD_COUNT = 3770
EXPECTED_RRUFF_ID_COUNT = 1936
EXPECTED_GROUP_COUNT = 1934
EXPECTED_SOURCE_RECORD_COUNT = 20664
CLASS_LABELS_SHA256 = (
    "289906f3282e0e7f7a82ea61de413bc4c474b6158383bea3aaad21aa956fb94c"
)
RECORD_IDS_SHA256 = (
    "840896eee008cb5df60116ebc727431a649529c124d63a14fb0219f778f0e72d"
)
GROUP_IDS_SHA256 = (
    "886613d52869b87c96e101fe1dc2b7bf74008ea9deb5ba9a6cfa237edc9644a3"
)
SPLIT_SHA256 = (
    "5d292560b28bf41c207c6fbe88d9e5025c4f03764240a41d6abf0da14138e5e5",
    "ee6703717075c7ab4094fcd68886866edfa1237e204f386a5622cb73538dd7c1",
    "91c2bc9c8fd669bb39f7ee29b79eecd2baa3573e37b9941e89b2beebb6593335",
    "01d736bcd75a872fbffbd6fac7cd2cd406bc90ad532227a24ed41fdb19dbb1cb",
    "f3d08a18e88e159b1097a0c2e823bb83d1e2aa0405f4bac3390de57618d97788",
)
ROOT = Path(__file__).resolve().parents[2]
DATASET_FILE_SHA256 = {
    "SHA256SUMS": (
        "f0cb09af8d80cdf3d52af9ed01bc1ae48abba94fefdff712c30d6e8c8bfe4995"
    ),
    "SHA256SUMS.sha256": (
        "72ec71a034c3ed081ddd27252d04910d51caceebcf84a4e9e1203f99ac50f314"
    ),
    "arrays.h5": (
        "768e4f3abd30db76b1a45e0a94448a63f180d2c3021dcfed56ce6ad861b32d68"
    ),
    "dataset.json": (
        "e4e5ca82f154eb2215bbda3e13dec2fe9aaa812e46ea7d36b7e4266754e84b10"
    ),
    "records.jsonl": (
        "4cc815e261abda595dd6facc3af52a5735d5f37c7dd346b683837f830c4b906e"
    ),
}
PAIR_INDEX_SHA256 = (
    "efbb9549fb9bb0db721def69ea158c45e7fbd09e6515df51b1f109b0c912ec6d"
)
CONVERSION_RECEIPT_SHA256 = (
    "282701601d962e440561e452aa25adf89d626108d893974f24ca63fac2b3facc"
)


class D5LoaderValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class D5ProtocolConfig:
    path: Path
    sha256: str
    byte_count: int
    dataset_id: str
    seeds: tuple[int, ...]
    grid_start_cm1: float
    grid_stop_cm1: float
    grid_step_cm1: float
    grid_point_count: int
    max_in_range_native_gap_cm1: float
    expected_class_count: int
    expected_record_count: int
    expected_rruff_id_count: int
    expected_group_count: int
    audit_sha256: str
    split_sha256: tuple[str, ...]


@dataclass(frozen=True)
class D5LibraryQuerySplit:
    seed: int
    query_indices: np.ndarray
    library_indices: np.ndarray
    split_sha256: str


@dataclass(frozen=True)
class D5RawCohort:
    protocol_config_sha256: str
    dataset_id: str
    intensity: np.ndarray
    wavenumber: np.ndarray
    class_labels: np.ndarray
    record_ids: tuple[str, ...]
    mineral_names: tuple[str, ...]
    rruff_ids: tuple[str, ...]
    pin_ids: tuple[str | None, ...]
    group_ids: tuple[str, ...]
    splits: tuple[D5LibraryQuerySplit, ...]


@dataclass(frozen=True)
class _EligibleRecord:
    record_id: str
    intensity: np.ndarray
    class_label: int
    mineral_name: str
    rruff_id: str
    pin_id: str | None


@dataclass(frozen=True)
class _RetainedRawRecord:
    record_id: str
    intensity: np.ndarray
    wavenumber: np.ndarray
    class_label: int
    mineral_name: str
    rruff_id: str
    pin_id: str | None


class _ConnectedComponents:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        self._parent.setdefault(value, value)
        parent = self._parent[value]
        if parent != value:
            parent = self.find(parent)
            self._parent[value] = parent
        return parent

    def union_group(self, values: set[str]) -> None:
        ordered = sorted(values)
        if not ordered:
            return
        first = ordered[0]
        for other in ordered[1:]:
            left = self.find(first)
            right = self.find(other)
            if left != right:
                self._parent[max(left, right)] = min(left, right)


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


def _reject_nonfinite(value: str) -> None:
    raise D5LoaderValidationError(
        "config nonfinite",
        f"unsupported JSON constant {value!r}",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(values: set[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("utf-8")
    ).hexdigest()


def _class_labels_digest(values: set[int]) -> str:
    return hashlib.sha256(
        ("\n".join(str(value) for value in sorted(values)) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D5LoaderValidationError(path, "must be an object")
    return value


def load_d5_protocol_config(path: Path) -> D5ProtocolConfig:
    path = Path(path)
    try:
        raw = path.read_bytes()
        document = json.loads(raw, parse_constant=_reject_nonfinite)
    except D5LoaderValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D5LoaderValidationError(
            path.name or "config",
            str(error),
        ) from error
    if raw != _canonical_json_bytes(document):
        raise D5LoaderValidationError(
            "config noncanonical",
            "must use canonical JSON",
        )
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTES or digest != CONFIG_SHA256:
        raise D5LoaderValidationError(
            "config identity",
            "bytes or SHA256 mismatch",
        )
    root = _object("config", document)
    if (
        root.get("schema_version") != SCHEMA_VERSION
        or root.get("experiment_id") != EXPERIMENT_ID
    ):
        raise D5LoaderValidationError(
            "config contract",
            "schema or experiment identity mismatch",
        )
    return D5ProtocolConfig(
        path=path,
        sha256=digest,
        byte_count=len(raw),
        dataset_id=DATASET_ID,
        seeds=SEEDS,
        grid_start_cm1=GRID_START_CM1,
        grid_stop_cm1=GRID_STOP_CM1,
        grid_step_cm1=GRID_STEP_CM1,
        grid_point_count=GRID_POINT_COUNT,
        max_in_range_native_gap_cm1=MAX_IN_RANGE_NATIVE_GAP_CM1,
        expected_class_count=EXPECTED_CLASS_COUNT,
        expected_record_count=EXPECTED_RECORD_COUNT,
        expected_rruff_id_count=EXPECTED_RRUFF_ID_COUNT,
        expected_group_count=EXPECTED_GROUP_COUNT,
        audit_sha256=AUDIT_SHA256,
        split_sha256=SPLIT_SHA256,
    )


def _validate_protocol_audit() -> None:
    path = ROOT / AUDIT_RELATIVE_PATH
    raw = path.read_bytes()
    try:
        document = json.loads(raw, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as error:
        raise D5LoaderValidationError("protocol audit", str(error)) from error
    if raw != _canonical_json_bytes(document):
        raise D5LoaderValidationError(
            "protocol audit",
            "must use canonical JSON",
        )
    if len(raw) != AUDIT_BYTES or hashlib.sha256(raw).hexdigest() != AUDIT_SHA256:
        raise D5LoaderValidationError(
            "protocol audit identity",
            "bytes or SHA256 mismatch",
        )


def _validate_retained_inputs(dataset_path: Path) -> None:
    if dataset_path.name != DATASET_ID:
        raise D5LoaderValidationError(
            "dataset path",
            f"name must equal {DATASET_ID!r}",
        )
    for name, expected in DATASET_FILE_SHA256.items():
        path = dataset_path / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise D5LoaderValidationError(
                f"dataset file {name}",
                "missing or SHA256 mismatch",
            )
    companions = {
        dataset_path.parent / "rruff_raman_pairs.jsonl": PAIR_INDEX_SHA256,
        dataset_path.parent / "rruff_raman_conversion.json": (
            CONVERSION_RECEIPT_SHA256
        ),
    }
    for path, expected in companions.items():
        if not path.is_file() or _sha256_file(path) != expected:
            raise D5LoaderValidationError(
                path.name,
                "missing or SHA256 mismatch",
            )


def _required_string(
    record_id: str,
    metadata: Mapping[str, object],
    key: str,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or value == "":
        raise D5LoaderValidationError(
            f"{record_id}.{key}",
            "must be a non-empty string",
        )
    return value


def _retained_raw_record(
    document: object,
    arrays: h5py.File,
) -> _RetainedRawRecord:
    root = _object("record", document)
    record_id = root.get("record_id")
    if not isinstance(record_id, str) or record_id == "":
        raise D5LoaderValidationError(
            "record_id",
            "must be a non-empty string",
        )
    meta = _object(f"{record_id}.meta", root.get("meta"))
    if meta.get("dataset_id") != DATASET_ID:
        raise D5LoaderValidationError(
            f"{record_id}.dataset_id",
            f"must equal {DATASET_ID!r}",
        )
    if meta.get("preprocessing_status") != "known_raw":
        raise D5LoaderValidationError(
            f"{record_id}.preprocessing_status",
            "must equal known_raw",
        )
    if meta.get("preprocessing_steps") != []:
        raise D5LoaderValidationError(
            f"{record_id}.preprocessing_steps",
            "must be empty",
        )
    metadata = _object(
        f"{record_id}.source_metadata",
        meta.get("source_metadata"),
    )
    rruff_id = _required_string(record_id, metadata, "rruff_id")
    mineral_name = _required_string(record_id, metadata, "mineral_name")
    if meta.get("sample_id") != rruff_id:
        raise D5LoaderValidationError(
            f"{record_id}.sample_id",
            "must equal source rruff_id",
        )
    pin_id = metadata.get("pin_id")
    if pin_id is not None and (not isinstance(pin_id, str) or pin_id == ""):
        raise D5LoaderValidationError(
            f"{record_id}.pin_id",
            "must be null or a non-empty string",
        )
    targets = _object(f"{record_id}.targets", root.get("targets"))
    class_label = targets.get("class_label")
    if (
        isinstance(class_label, bool)
        or not isinstance(class_label, int)
        or class_label < 0
    ):
        raise D5LoaderValidationError(
            f"{record_id}.class_label",
            "must be a non-negative integer",
        )
    array_ref = _object(f"{record_id}.array_ref", root.get("array_ref"))
    axis_id = array_ref.get("axis_id")
    row = array_ref.get("row")
    if not isinstance(axis_id, str) or axis_id == "":
        raise D5LoaderValidationError(
            f"{record_id}.array_ref.axis_id",
            "must be a non-empty string",
        )
    if isinstance(row, bool) or not isinstance(row, int) or row < 0:
        raise D5LoaderValidationError(
            f"{record_id}.array_ref.row",
            "must be a non-negative integer",
        )
    try:
        group = arrays["axes"][axis_id]
        wavenumber = np.asarray(group["wavenumber"][:], dtype="<f4")
        intensity = np.asarray(group["intensity"][row], dtype="<f4")
    except (KeyError, IndexError, OSError, ValueError) as error:
        raise D5LoaderValidationError(
            f"{record_id}.array_ref",
            str(error),
        ) from error
    if (
        wavenumber.ndim != 1
        or intensity.ndim != 1
        or wavenumber.shape != intensity.shape
        or wavenumber.size == 0
        or not np.isfinite(wavenumber).all()
        or not np.isfinite(intensity).all()
        or not np.all(np.diff(wavenumber) > 0)
    ):
        raise D5LoaderValidationError(
            f"{record_id}.arrays",
            "must be finite, nonempty, aligned, and strictly increasing",
        )
    return _RetainedRawRecord(
        record_id=record_id,
        intensity=intensity,
        wavenumber=wavenumber,
        class_label=class_label,
        mineral_name=mineral_name,
        rruff_id=rruff_id,
        pin_id=pin_id,
    )


def _iter_retained_raw_records(
    dataset_path: Path,
):
    try:
        arrays = h5py.File(dataset_path / "arrays.h5", "r")
    except (OSError, ValueError) as error:
        raise D5LoaderValidationError("arrays.h5", str(error)) from error
    try:
        with (dataset_path / "records.jsonl").open(
            encoding="utf-8"
        ) as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    document = json.loads(
                        line,
                        parse_constant=_reject_nonfinite,
                    )
                except json.JSONDecodeError as error:
                    raise D5LoaderValidationError(
                        f"records.jsonl line {line_number}",
                        str(error),
                    ) from error
                if line.encode("utf-8") != _canonical_json_bytes(document):
                    raise D5LoaderValidationError(
                        f"records.jsonl line {line_number}",
                        "must use canonical JSON",
                    )
                yield _retained_raw_record(document, arrays)
    finally:
        arrays.close()


def _native_identity(record: _RetainedRawRecord) -> str:
    axis = np.ascontiguousarray(record.wavenumber, dtype="<f4")
    intensity = np.ascontiguousarray(record.intensity, dtype="<f4")
    return hashlib.sha256(
        b"rpe-d5-native-spectrum-v1\0"
        + axis.tobytes(order="C")
        + intensity.tobytes(order="C")
    ).hexdigest()


def _aligned_intensity(
    record: _RetainedRawRecord,
    grid: np.ndarray,
) -> np.ndarray | None:
    axis = np.asarray(record.wavenumber, dtype=np.float64)
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    if left < 0 or right >= axis.size:
        return None
    support = axis[left : right + 1]
    if float(np.max(np.diff(support))) > MAX_IN_RANGE_NATIVE_GAP_CM1:
        return None
    aligned = np.asarray(
        np.interp(
            grid,
            axis,
            np.asarray(record.intensity, dtype=np.float64),
        ),
        dtype="<f4",
    )
    if not np.isfinite(aligned).all():
        raise D5LoaderValidationError(
            f"{record.record_id}.aligned intensity",
            "contains non-finite values",
        )
    return aligned


def _resampled_identity(intensity: np.ndarray) -> str:
    return hashlib.sha256(
        b"rpe-d5-resampled-spectrum-v1\0"
        + np.ascontiguousarray(intensity, dtype="<f4").tobytes(order="C")
    ).hexdigest()


def _split_sha256(
    seed: int,
    by_class_group_indices: Mapping[int, Mapping[str, list[int]]],
) -> tuple[str, list[int], list[int]]:
    classes = {}
    query_indices = []
    library_indices = []
    for class_label in sorted(by_class_group_indices):
        groups = by_class_group_indices[class_label]
        ordered_groups = sorted(groups)
        generator = np.random.Generator(
            np.random.PCG64(
                np.random.SeedSequence([seed, class_label])
            )
        )
        permutation = generator.permutation(len(ordered_groups))
        query_group = ordered_groups[int(permutation[0])]
        class_query = sorted(groups[query_group])
        class_library_groups = [
            group for group in ordered_groups if group != query_group
        ]
        class_library = sorted(
            index
            for group in class_library_groups
            for index in groups[group]
        )
        query_indices.extend(class_query)
        library_indices.extend(class_library)
        classes[str(class_label)] = {
            "query_group": query_group,
            "query_record_count": len(class_query),
            "library_group_count": len(class_library_groups),
            "library_record_count": len(class_library),
        }
    payload = {"seed": seed, "classes": classes}
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return digest, sorted(query_indices), sorted(library_indices)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def load_d5_raw_cohort(
    config_path: Path,
    dataset_path: Path,
) -> D5RawCohort:
    config = load_d5_protocol_config(config_path)
    _validate_protocol_audit()
    dataset_path = Path(dataset_path)
    _validate_retained_inputs(dataset_path)
    grid64 = np.arange(
        config.grid_start_cm1,
        config.grid_stop_cm1 + config.grid_step_cm1 / 2.0,
        config.grid_step_cm1,
        dtype=np.float64,
    )
    if grid64.size != config.grid_point_count:
        raise D5LoaderValidationError(
            "common grid",
            "point count mismatch",
        )

    components = _ConnectedComponents()
    pin_groups: dict[str, set[str]] = defaultdict(set)
    native_groups: dict[str, set[str]] = defaultdict(set)
    resampled_groups: dict[str, set[str]] = defaultdict(set)
    classes_by_rruff_id: dict[str, set[int]] = defaultdict(set)
    eligible: list[_EligibleRecord] = []

    source_record_count = 0
    for record in _iter_retained_raw_records(dataset_path):
        source_record_count += 1
        components.find(record.rruff_id)
        classes_by_rruff_id[record.rruff_id].add(record.class_label)
        if record.pin_id is not None:
            pin_groups[record.pin_id].add(record.rruff_id)
        native_groups[_native_identity(record)].add(record.rruff_id)
        aligned = _aligned_intensity(record, grid64)
        if aligned is None:
            continue
        resampled_groups[_resampled_identity(aligned)].add(record.rruff_id)
        eligible.append(
            _EligibleRecord(
                record_id=record.record_id,
                intensity=aligned,
                class_label=record.class_label,
                mineral_name=record.mineral_name,
                rruff_id=record.rruff_id,
                pin_id=record.pin_id,
            )
        )
    if source_record_count != EXPECTED_SOURCE_RECORD_COUNT:
        raise D5LoaderValidationError(
            "source record count",
            (
                f"must equal {EXPECTED_SOURCE_RECORD_COUNT}; "
                f"observed {source_record_count}"
            ),
        )

    for groups in (
        pin_groups.values(),
        native_groups.values(),
        resampled_groups.values(),
    ):
        for values in groups:
            components.union_group(values)

    component_classes: dict[str, set[int]] = defaultdict(set)
    for rruff_id, class_labels in classes_by_rruff_id.items():
        component_classes[components.find(rruff_id)].update(class_labels)
    conflicted = {
        group_id
        for group_id, class_labels in component_classes.items()
        if len(class_labels) > 1
    }

    by_class_groups: dict[int, dict[str, list[_EligibleRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in eligible:
        group_id = components.find(record.rruff_id)
        if group_id not in conflicted:
            by_class_groups[record.class_label][group_id].append(record)
    retained_classes = {
        class_label
        for class_label, groups in by_class_groups.items()
        if len(groups) >= 2
    }
    cohort_records = [
        record
        for record in eligible
        if record.class_label in retained_classes
        and components.find(record.rruff_id)
        in by_class_groups[record.class_label]
    ]

    record_ids = tuple(record.record_id for record in cohort_records)
    class_labels = _read_only(
        np.asarray(
            [record.class_label for record in cohort_records],
            dtype="<i8",
        )
    )
    mineral_names = tuple(record.mineral_name for record in cohort_records)
    rruff_ids = tuple(record.rruff_id for record in cohort_records)
    pin_ids = tuple(record.pin_id for record in cohort_records)
    group_ids = tuple(
        components.find(record.rruff_id) for record in cohort_records
    )
    intensity = _read_only(
        np.stack(
            [record.intensity for record in cohort_records],
            axis=0,
        ).astype("<f4", copy=False)
    )
    wavenumber = _read_only(np.asarray(grid64, dtype="<f4"))

    observed_class_labels = set(int(value) for value in class_labels)
    observed_record_ids = set(record_ids)
    observed_rruff_ids = set(rruff_ids)
    observed_group_ids = set(group_ids)
    observed = (
        len(observed_class_labels),
        len(record_ids),
        len(observed_rruff_ids),
        len(observed_group_ids),
    )
    expected = (
        config.expected_class_count,
        config.expected_record_count,
        config.expected_rruff_id_count,
        config.expected_group_count,
    )
    if observed != expected:
        raise D5LoaderValidationError(
            "cohort counts",
            f"expected {expected!r}; observed {observed!r}",
        )
    identities = (
        _class_labels_digest(observed_class_labels),
        _ids_digest(observed_record_ids),
        _ids_digest(observed_group_ids),
    )
    expected_identities = (
        CLASS_LABELS_SHA256,
        RECORD_IDS_SHA256,
        GROUP_IDS_SHA256,
    )
    if identities != expected_identities:
        raise D5LoaderValidationError(
            "cohort identities",
            "class, record, or group digest mismatch",
        )

    by_class_group_indices: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for index, (class_label, group_id) in enumerate(
        zip(class_labels, group_ids, strict=True)
    ):
        by_class_group_indices[int(class_label)][group_id].append(index)
    splits = []
    all_indices = set(range(len(record_ids)))
    for seed, expected_split_sha256 in zip(
        config.seeds,
        config.split_sha256,
        strict=True,
    ):
        digest, query, library = _split_sha256(
            seed,
            by_class_group_indices,
        )
        if digest != expected_split_sha256:
            raise D5LoaderValidationError(
                f"split seed {seed}",
                "SHA256 mismatch",
            )
        if set(query) & set(library) or set(query) | set(library) != all_indices:
            raise D5LoaderValidationError(
                f"split seed {seed}",
                "record partition is not disjoint and complete",
            )
        query_groups = {group_ids[index] for index in query}
        library_groups = {group_ids[index] for index in library}
        if query_groups & library_groups:
            raise D5LoaderValidationError(
                f"split seed {seed}",
                "leakage group crosses query and library",
            )
        splits.append(
            D5LibraryQuerySplit(
                seed=seed,
                query_indices=_read_only(np.asarray(query, dtype="<i8")),
                library_indices=_read_only(
                    np.asarray(library, dtype="<i8")
                ),
                split_sha256=digest,
            )
        )

    return D5RawCohort(
        protocol_config_sha256=config.sha256,
        dataset_id=config.dataset_id,
        intensity=intensity,
        wavenumber=wavenumber,
        class_labels=class_labels,
        record_ids=record_ids,
        mineral_names=mineral_names,
        rruff_ids=rruff_ids,
        pin_ids=pin_ids,
        group_ids=group_ids,
        splits=tuple(splits),
    )


def load_d5_native_spectra(
    dataset_path: Path,
    record_ids: tuple[str, ...],
) -> tuple[Spectrum1D, ...]:
    if not isinstance(record_ids, tuple) or not record_ids:
        raise D5LoaderValidationError(
            "record_ids",
            "must be a nonempty tuple",
        )
    if any(not isinstance(record_id, str) or record_id == "" for record_id in record_ids):
        raise D5LoaderValidationError(
            "record_ids",
            "must contain only nonempty strings",
        )
    if len(set(record_ids)) != len(record_ids):
        raise D5LoaderValidationError(
            "record_ids",
            "must contain unique values",
        )

    dataset_path = Path(dataset_path)
    _validate_retained_inputs(dataset_path)
    requested = set(record_ids)
    located: dict[str, _RetainedRawRecord] = {}
    for record in _iter_retained_raw_records(dataset_path):
        if record.record_id in requested:
            located[record.record_id] = record
    missing = sorted(requested - set(located))
    if missing:
        raise D5LoaderValidationError(
            "record_ids missing",
            f"not found in retained dataset: {missing!r}",
        )

    return tuple(
        Spectrum1D(
            spectrum_id=f"{DATASET_ID}::{record.record_id}",
            sample_id=record.rruff_id,
            axis_cm1=np.asarray(record.wavenumber, dtype="<f8"),
            intensity=np.asarray(record.intensity, dtype="<f8"),
        )
        for record in (located[record_id] for record_id in record_ids)
    )


__all__ = [
    "D5LibraryQuerySplit",
    "D5LoaderValidationError",
    "D5ProtocolConfig",
    "D5RawCohort",
    "load_d5_native_spectra",
    "load_d5_protocol_config",
    "load_d5_raw_cohort",
]
