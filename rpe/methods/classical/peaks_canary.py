from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from rpe.evaluation import Spectrum1D
from rpe.io.store import UnifiedDataset
from rpe.methods.catalog import (
    Phase3ClassicalCatalog,
    Phase3System,
    TaskLine,
    load_classical_catalog,
)
from rpe.methods.classical.baseline_canary import load_rruff_baseline_canary_sources
from rpe.methods.classical.peaks import (
    DetectedPeak1D,
    PeakDetectionRunResult,
    PeakRunStatus,
    PeakWarning,
    peak_document,
    run_peak_detection_system,
)


CONFIG_BYTE_COUNT = 2891
CONFIG_SHA256 = "2a6636de43c87ae95cd77d2ab2cc42470f188f60e8b66c3ed654a76402b72d7d"
_RUN_DOMAIN = b"rpe-phase3-peak-detection-v1-canary-v1\0"
_SUCCESS = frozenset({PeakRunStatus.COMPLETE.value, PeakRunStatus.COMPLETE_WITH_WARNING.value})


class PeakCanaryError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class PeakCanaryConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    catalog_path: str
    catalog_sha256: str
    catalog_id: str
    phase3_lock_path: str
    phase3_lock_sha256: str
    design_path: str
    design_sha256: str
    source_ledger_sha256: str
    system_ids: tuple[str, ...]
    rruff: Mapping[str, object]
    bacteria: Mapping[str, object]
    expected: Mapping[str, int]
    claim_boundary: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "rruff", _freeze_mapping(self.rruff))
        object.__setattr__(self, "bacteria", _freeze_mapping(self.bacteria))
        object.__setattr__(self, "expected", MappingProxyType(dict(self.expected)))


@dataclass(frozen=True)
class PeakCanarySource:
    cohort_id: str
    record_id: str
    sample_id: str | None
    mineral_name: str | None
    class_label: int
    source_row: int
    point_count: int
    axis_sha256: str
    intensity_sha256: str
    spectrum: Spectrum1D


@dataclass(frozen=True)
class PeakCanarySummary:
    path: Path
    run_id: str
    system_count: int
    source_count: int
    receipt_count: int
    peak_count: int
    empty_receipt_count: int
    status_counts: Mapping[str, int]
    canary_passed: bool
    is_frozen_canary: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "status_counts", MappingProxyType(dict(sorted(self.status_counts.items()))))


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _reject_nonfinite(token: str) -> object:
    raise PeakCanaryError("JSON", f"non-finite constant {token}")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _freeze(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PeakCanaryError("mapping", "contains non-finite float")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    raise PeakCanaryError("mapping", "contains unsupported value")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _identity(root: Path, value: Mapping[str, object], label: str) -> None:
    path = root / str(value["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(value["byte_count"])
        or _sha(path) != str(value["sha256"])
    ):
        raise PeakCanaryError(label, "identity mismatch")


def load_peak_canary_config(path: Path, *, project_root: Path) -> PeakCanaryConfig:
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw, parse_constant=_reject_nonfinite)
    if not isinstance(document, Mapping) or raw != _canonical(document):
        raise PeakCanaryError("config", "must be canonical finite JSON")
    observed_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTE_COUNT or observed_sha != CONFIG_SHA256:
        raise PeakCanaryError("config", "frozen identity mismatch")
    root = Path(project_root)
    for label in ("catalog", "phase3_lock", "design"):
        value = document[label]
        if not isinstance(value, Mapping):
            raise PeakCanaryError(label, "must be an object")
        _identity(root, value, label)
    catalog_value = document["catalog"]
    lock_value = document["phase3_lock"]
    design_value = document["design"]
    assert isinstance(catalog_value, Mapping)
    assert isinstance(lock_value, Mapping)
    assert isinstance(design_value, Mapping)
    catalog = load_classical_catalog(root / str(catalog_value["path"]), project_root=root)
    expected_ids = tuple(
        system.system_id
        for system in catalog.systems
        if system.task_line is TaskLine.PEAK_DETECTION
        and system.family_id in {"find_peaks", "find_peaks_cwt"}
    )
    configured_ids = tuple(str(value) for value in document["system_ids"])
    if configured_ids != tuple(sorted(expected_ids)) or len(configured_ids) != 24:
        raise PeakCanaryError("system_ids", "do not match SciPy catalog view")
    expected_value = document["expected"]
    if not isinstance(expected_value, Mapping):
        raise PeakCanaryError("expected", "must be an object")
    expected = {str(key): int(value) for key, value in expected_value.items()}
    if expected != {"receipt_count": 144, "source_count": 6, "system_count": 24}:
        raise PeakCanaryError("expected", "does not match frozen design")
    rruff = document["rruff"]
    bacteria = document["bacteria"]
    if not isinstance(rruff, Mapping) or not isinstance(bacteria, Mapping):
        raise PeakCanaryError("cohorts", "must be objects")
    return PeakCanaryConfig(
        path=path,
        byte_count=len(raw),
        sha256=observed_sha,
        schema_version=str(document["schema_version"]),
        catalog_path=str(catalog_value["path"]),
        catalog_sha256=str(catalog_value["sha256"]),
        catalog_id=str(catalog_value["catalog_id"]),
        phase3_lock_path=str(lock_value["path"]),
        phase3_lock_sha256=str(lock_value["sha256"]),
        design_path=str(design_value["path"]),
        design_sha256=str(design_value["sha256"]),
        source_ledger_sha256=str(document["source_ledger_sha256"]),
        system_ids=configured_ids,
        rruff=rruff,
        bacteria=bacteria,
        expected=expected,
        claim_boundary=str(document["claim_boundary"]),
    )


def source_document(source: PeakCanarySource) -> dict[str, object]:
    return {
        "axis_sha256": source.axis_sha256,
        "class_label": source.class_label,
        "cohort_id": source.cohort_id,
        "intensity_sha256": source.intensity_sha256,
        "mineral_name": source.mineral_name,
        "point_count": source.point_count,
        "record_id": source.record_id,
        "sample_id": source.sample_id,
        "source_row": source.source_row,
    }


def _materialize_bacteria(dataset: UnifiedDataset, row: Mapping[str, object]) -> PeakCanarySource:
    record_id = str(row["record_id"])
    record = dataset.get(record_id)
    axis = np.asarray(record.wavenumber, dtype="<f8")
    intensity = np.asarray(record.intensity, dtype="<f8")
    if np.all(np.diff(axis) < 0.0):
        axis = np.ascontiguousarray(axis[::-1])
        intensity = np.ascontiguousarray(intensity[::-1])
    spectrum = Spectrum1D(record_id, None, axis, intensity)
    return PeakCanarySource(
        cohort_id="bacteria_test",
        record_id=record_id,
        sample_id=None,
        mineral_name=None,
        class_label=int(row["targets"]["class_label"]),  # type: ignore[index]
        source_row=int(row["meta"]["source_metadata"]["source_row"]),  # type: ignore[index]
        point_count=spectrum.intensity.size,
        axis_sha256=_array_sha(spectrum.axis_cm1),
        intensity_sha256=_array_sha(spectrum.intensity),
        spectrum=spectrum,
    )


def load_peak_canary_sources(
    config: PeakCanaryConfig, *, project_root: Path
) -> tuple[PeakCanarySource, ...]:
    if not isinstance(config, PeakCanaryConfig):
        raise PeakCanaryError("config", "must be PeakCanaryConfig")
    root = Path(project_root)
    rruff = load_rruff_baseline_canary_sources(
        root / str(config.rruff["phase1_run_path"]),
        root / str(config.rruff["dataset_path"]),
    )
    sources = [
        PeakCanarySource(
            cohort_id=value.band_id,
            record_id=value.record_id,
            sample_id=value.sample_id,
            mineral_name=value.mineral_name,
            class_label=value.class_label,
            source_row=value.selection_rank,
            point_count=value.point_count,
            axis_sha256=value.axis_sha256,
            intensity_sha256=value.intensity_sha256,
            spectrum=value.spectrum,
        )
        for value in rruff
    ]
    bacteria_path = root / str(config.bacteria["dataset_path"])
    if _sha(bacteria_path / "SHA256SUMS") != config.bacteria["dataset_sha256sums_sha256"]:
        raise PeakCanaryError("Bacteria SHA256SUMS", "identity mismatch")
    rows = {row["record_id"]: row for row in (
        json.loads(line, parse_constant=_reject_nonfinite)
        for line in (bacteria_path / "records.jsonl").read_bytes().splitlines()
    )}
    with UnifiedDataset.open(bacteria_path, verify_checksums=True) as dataset:
        for record_id in config.bacteria["record_ids"]:
            if record_id not in rows:
                raise PeakCanaryError("Bacteria record", f"missing {record_id}")
            sources.append(_materialize_bacteria(dataset, rows[record_id]))
    frozen = tuple(sources)
    payload = b"".join(_canonical(source_document(value)) for value in frozen)
    if hashlib.sha256(payload).hexdigest() != config.source_ledger_sha256:
        raise PeakCanaryError("source ledger", "SHA256 mismatch")
    return frozen


def _system_document(system: Phase3System) -> dict[str, object]:
    return {
        "family_id": system.family_id,
        "hyperparameters": _json_ready(system.hyperparameters),
        "method_id": system.method_id,
        "system_id": system.system_id,
    }


def _code_identity(root: Path) -> dict[str, str]:
    paths = (
        "rpe/methods/catalog.py",
        "rpe/methods/classical/peaks.py",
        "rpe/methods/classical/peaks_canary.py",
    )
    return {path: _sha(root / path) for path in paths}


def _software_identity() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy")
    }


def _warning_documents(
    warnings: Sequence[PeakWarning], *, system_id: str, record_id: str
) -> list[dict[str, object]]:
    return [
        {
            "category": value.category,
            "message": value.message,
            "record_id": record_id,
            "sequence": index,
            "system_id": system_id,
        }
        for index, value in enumerate(warnings)
    ]


def _result_documents(
    source: PeakCanarySource,
    result: PeakDetectionRunResult,
    peak_start: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    peak_rows = [
        {
            **peak_document(value),
            "record_id": source.record_id,
            "sequence": sequence,
            "system_id": result.system_id,
        }
        for sequence, value in enumerate(result.peaks)
    ]
    record = {
        "cohort_id": source.cohort_id,
        "diagnostics": _json_ready(result.diagnostics),
        "error_code": result.error_code,
        "error_message": result.error_message,
        "family_id": result.family_id,
        "method_id": result.method_id,
        "peak_count": len(result.peaks),
        "peak_start": peak_start,
        "peaks_sha256": result.peaks_sha256,
        "point_count": source.point_count,
        "record_id": source.record_id,
        "source_row": source.source_row,
        "status": result.status.value,
        "system_id": result.system_id,
        "warnings": [
            {"category": value.category, "message": value.message}
            for value in result.warnings
        ],
    }
    return record, peak_rows


def build_peak_canary_artifact(
    sources: Sequence[PeakCanarySource],
    systems: Sequence[Phase3System],
    output_root: Path,
    *,
    config: PeakCanaryConfig,
    catalog: Phase3ClassicalCatalog,
    project_root: Path,
) -> PeakCanarySummary:
    frozen_sources = tuple(sources)
    frozen_systems = tuple(sorted(systems, key=lambda value: value.system_id))
    if not frozen_sources or len({value.record_id for value in frozen_sources}) != len(frozen_sources):
        raise PeakCanaryError("sources", "must be nonempty and unique")
    if not frozen_systems or len({value.system_id for value in frozen_systems}) != len(frozen_systems):
        raise PeakCanaryError("systems", "must be nonempty and unique")
    catalog_ids = {value.system_id for value in catalog.systems}
    if any(
        value.task_line is not TaskLine.PEAK_DETECTION
        or value.family_id not in {"find_peaks", "find_peaks_cwt"}
        or value.system_id not in catalog_ids
        for value in frozen_systems
    ):
        raise PeakCanaryError("systems", "must be catalog SciPy peak systems")
    for source in frozen_sources:
        if (
            source.point_count != source.spectrum.intensity.size
            or source.axis_sha256 != _array_sha(source.spectrum.axis_cm1)
            or source.intensity_sha256 != _array_sha(source.spectrum.intensity)
        ):
            raise PeakCanaryError("source identity", "does not match spectrum")
    source_payload = b"".join(_canonical(source_document(value)) for value in frozen_sources)
    source_sha = hashlib.sha256(source_payload).hexdigest()
    system_ids = tuple(value.system_id for value in frozen_systems)
    root = Path(project_root)
    identity = {
        "catalog_id": catalog.catalog_id,
        "catalog_sha256": catalog.sha256,
        "code_identity": _code_identity(root),
        "config_sha256": config.sha256,
        "phase3_lock_sha256": config.phase3_lock_sha256,
        "software_identity": _software_identity(),
        "source_ledger_sha256": source_sha,
        "system_ids": list(system_ids),
    }
    run_id = hashlib.sha256(_RUN_DOMAIN + _canonical(identity)).hexdigest()
    path = Path(output_root) / f"phase3-peak-detection-v1-canary-{run_id}"
    if path.exists():
        raise PeakCanaryError("output", "run path already exists")
    path.mkdir(parents=True)
    record_rows: list[dict[str, object]] = []
    peak_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    for current_system in frozen_systems:
        for source in frozen_sources:
            result = run_peak_detection_system(current_system, source.spectrum)
            record, current_peaks = _result_documents(source, result, len(peak_rows))
            record_rows.append(record)
            peak_rows.extend(current_peaks)
            warning_rows.extend(
                _warning_documents(
                    result.warnings,
                    system_id=current_system.system_id,
                    record_id=source.record_id,
                )
            )
    status_counts = Counter(str(value["status"]) for value in record_rows)
    empty_count = sum(
        str(value["status"]) in _SUCCESS and int(value["peak_count"]) == 0
        for value in record_rows
    )
    rejection_totals: Counter[str] = Counter()
    for row in record_rows:
        diagnostics = row["diagnostics"]
        if isinstance(diagnostics, Mapping):
            characterize = diagnostics.get("characterization_rejections", {})
            if isinstance(characterize, Mapping):
                rejection_totals.update({str(k): int(v) for k, v in characterize.items()})
            for key in ("rejected_duplicate", "rejected_non_admissible", "rejected_out_of_range"):
                if key in diagnostics:
                    rejection_totals[key] += int(diagnostics[key])
    is_frozen = (
        system_ids == config.system_ids
        and len(frozen_sources) == config.expected["source_count"]
        and len(frozen_systems) == config.expected["system_count"]
        and len(record_rows) == config.expected["receipt_count"]
        and source_sha == config.source_ledger_sha256
    )
    all_success = set(status_counts).issubset(_SUCCESS)
    canary_passed = is_frozen and all_success
    summary = {
        "canary_passed": canary_passed,
        "claim_boundary": config.claim_boundary,
        "empty_receipt_count": empty_count,
        "is_frozen_canary": is_frozen,
        "peak_count": len(peak_rows),
        "receipt_count": len(record_rows),
        "rejection_totals": dict(sorted(rejection_totals.items())),
        "run_id": run_id,
        "source_count": len(frozen_sources),
        "status_counts": dict(sorted(status_counts.items())),
        "system_count": len(frozen_systems),
        "warning_count": len(warning_rows),
    }
    manifest = {
        **identity,
        "canary_passed": canary_passed,
        "claim_boundary": config.claim_boundary,
        "is_frozen_canary": is_frozen,
        "peak_count": len(peak_rows),
        "receipt_count": len(record_rows),
        "run_id": run_id,
        "schema_version": "phase3-peak-detection-v1-canary-run-v1",
        "source_count": len(frozen_sources),
        "system_count": len(frozen_systems),
    }
    marker_name = "complete.json" if canary_passed else "failed.json"
    marker = {
        "canary_passed": canary_passed,
        "is_frozen_canary": is_frozen,
        "run_id": run_id,
        "status": "complete" if canary_passed else "failed",
    }
    payloads = {
        "manifest.json": _canonical(manifest),
        "peaks.jsonl": b"".join(_canonical(value) for value in peak_rows),
        "records.jsonl": b"".join(_canonical(value) for value in record_rows),
        "sources.jsonl": source_payload,
        "summary.json": _canonical(summary),
        "systems.jsonl": b"".join(_canonical(_system_document(value)) for value in frozen_systems),
        "warnings.jsonl": b"".join(_canonical(value) for value in warning_rows),
        marker_name: _canonical(marker),
    }
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    (path / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
            for name in sorted(payloads)
        ),
        encoding="utf-8",
    )
    return PeakCanarySummary(
        path=path,
        run_id=run_id,
        system_count=len(frozen_systems),
        source_count=len(frozen_sources),
        receipt_count=len(record_rows),
        peak_count=len(peak_rows),
        empty_receipt_count=empty_count,
        status_counts=status_counts,
        canary_passed=canary_passed,
        is_frozen_canary=is_frozen,
    )


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw, parse_constant=_reject_nonfinite)
    if not isinstance(value, Mapping) or raw != _canonical(value):
        raise PeakCanaryError(label, "must be canonical JSON object")
    return value


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, object]]:
    values = []
    for index, raw in enumerate(path.read_bytes().splitlines(keepends=True), start=1):
        value = json.loads(raw, parse_constant=_reject_nonfinite)
        if not isinstance(value, Mapping) or raw != _canonical(value):
            raise PeakCanaryError(label, f"line {index} is not canonical")
        values.append(value)
    return values


def _peak_from_row(row: Mapping[str, object]) -> DetectedPeak1D:
    return DetectedPeak1D(
        index=int(row["index"]),
        position_cm1=float(row["position_cm1"]),
        height=float(row["height"]),
        prominence=float(row["prominence"]),
        fwhm_cm1=float(row["fwhm_cm1"]),
        area=float(row["area"]),
        left_base_index=int(row["left_base_index"]),
        right_base_index=int(row["right_base_index"]),
        contour_height=float(row["contour_height"]),
        area_left_cm1=float(row["area_left_cm1"]),
        area_right_cm1=float(row["area_right_cm1"]),
    )


def _record_document(
    source: PeakCanarySource,
    result: PeakDetectionRunResult,
    peak_start: int,
) -> dict[str, object]:
    return _result_documents(source, result, peak_start)[0]


def verify_peak_canary_artifact(path: Path, *, project_root: Path) -> PeakCanarySummary:
    path = Path(path)
    checksum_raw = (path / "SHA256SUMS").read_text(encoding="utf-8")
    checks: dict[str, str] = {}
    for line in checksum_raw.splitlines():
        try:
            digest, name = line.split("  ")
        except ValueError as error:
            raise PeakCanaryError("checksum", "invalid line") from error
        if name in checks or len(digest) != 64:
            raise PeakCanaryError("checksum", "invalid or duplicate entry")
        checks[name] = digest
        if not (path / name).is_file() or _sha(path / name) != digest:
            raise PeakCanaryError("checksum", f"{name} mismatch")
    if checksum_raw != "".join(f"{checks[name]}  {name}\n" for name in sorted(checks)):
        raise PeakCanaryError("checksum", "must be canonically sorted")
    actual = {
        value.relative_to(path).as_posix()
        for value in path.rglob("*")
        if value.is_file() and value.name != "SHA256SUMS"
    }
    if actual != set(checks):
        raise PeakCanaryError("checksum inventory", "does not match files")
    marker_names = sorted(actual & {"complete.json", "failed.json"})
    if len(marker_names) != 1:
        raise PeakCanaryError("marker", "must contain exactly one")
    expected_inventory = {
        "manifest.json", "peaks.jsonl", "records.jsonl", "sources.jsonl",
        "summary.json", "systems.jsonl", "warnings.jsonl", marker_names[0],
    }
    if actual != expected_inventory:
        raise PeakCanaryError("artifact inventory", "does not match protocol")
    manifest = _read_json(path / "manifest.json", "manifest")
    summary = _read_json(path / "summary.json", "summary")
    source_rows = _read_jsonl(path / "sources.jsonl", "sources")
    system_rows = _read_jsonl(path / "systems.jsonl", "systems")
    records = _read_jsonl(path / "records.jsonl", "records")
    peaks = _read_jsonl(path / "peaks.jsonl", "peak ledger")
    warnings = _read_jsonl(path / "warnings.jsonl", "warnings")
    root = Path(project_root)
    config = load_peak_canary_config(
        root / "experiments/phase3/configs/peak_detection_v1_canary.json",
        project_root=root,
    )
    catalog = load_classical_catalog(root / config.catalog_path, project_root=root)
    catalog_by_id = {value.system_id: value for value in catalog.systems}
    system_ids = tuple(str(value["system_id"]) for value in system_rows)
    if system_ids != tuple(sorted(set(system_ids))):
        raise PeakCanaryError("systems", "must be unique and sorted")
    try:
        systems = tuple(catalog_by_id[value] for value in system_ids)
    except KeyError as error:
        raise PeakCanaryError("systems", "contains unknown ID") from error
    if system_rows != [_system_document(value) for value in systems]:
        raise PeakCanaryError("systems", "do not match catalog")
    current_sources = load_peak_canary_sources(config, project_root=root)
    current_by_id = {value.record_id: value for value in current_sources}
    source_ids = tuple(str(value["record_id"]) for value in source_rows)
    try:
        sources = tuple(current_by_id[value] for value in source_ids)
    except KeyError as error:
        raise PeakCanaryError("sources", "contains unknown frozen source") from error
    if source_rows != [source_document(value) for value in sources]:
        raise PeakCanaryError("sources", "do not match current frozen inputs")
    source_sha = hashlib.sha256(
        b"".join(_canonical(value) for value in source_rows)
    ).hexdigest()
    expected_pairs = tuple(
        (system.system_id, source.record_id)
        for system in systems
        for source in sources
    )
    observed_pairs = tuple(
        (str(value["system_id"]), str(value["record_id"])) for value in records
    )
    if observed_pairs != expected_pairs:
        raise PeakCanaryError("records", "Cartesian order mismatch")
    expected_warnings = []
    recomputed_records = []
    recomputed_peaks = []
    for system, source, stored in zip(
        (system for system in systems for _ in sources),
        (source for _ in systems for source in sources),
        records,
        strict=True,
    ):
        start = int(stored["peak_start"])
        count = int(stored["peak_count"])
        if start != len(recomputed_peaks) or count < 0 or start + count > len(peaks):
            raise PeakCanaryError("peak ledger", "slice bounds mismatch")
        stored_slice = peaks[start : start + count]
        for sequence, row in enumerate(stored_slice):
            if (
                row["system_id"] != system.system_id
                or row["record_id"] != source.record_id
                or int(row["sequence"]) != sequence
            ):
                raise PeakCanaryError("peak ledger", "key/order mismatch")
            _peak_from_row(row)
        rerun = run_peak_detection_system(system, source.spectrum)
        record_document, peak_documents = _result_documents(
            source, rerun, len(recomputed_peaks)
        )
        if stored != record_document or stored_slice != peak_documents:
            raise PeakCanaryError("peak ledger", "does not match re-execution")
        recomputed_records.append(record_document)
        recomputed_peaks.extend(peak_documents)
        expected_warnings.extend(
            _warning_documents(
                rerun.warnings, system_id=system.system_id, record_id=source.record_id
            )
        )
    if len(recomputed_peaks) != len(peaks):
        raise PeakCanaryError("peak ledger", "contains unattributed rows")
    if warnings != expected_warnings:
        raise PeakCanaryError("warnings", "do not match receipts")
    status_counts = Counter(str(value["status"]) for value in records)
    empty_count = sum(
        str(value["status"]) in _SUCCESS and int(value["peak_count"]) == 0
        for value in records
    )
    rejection_totals: Counter[str] = Counter()
    for row in records:
        diagnostics = row["diagnostics"]
        if isinstance(diagnostics, Mapping):
            characterize = diagnostics.get("characterization_rejections", {})
            if isinstance(characterize, Mapping):
                rejection_totals.update({str(k): int(v) for k, v in characterize.items()})
            for key in ("rejected_duplicate", "rejected_non_admissible", "rejected_out_of_range"):
                if key in diagnostics:
                    rejection_totals[key] += int(diagnostics[key])
    is_frozen = (
        system_ids == config.system_ids
        and source_sha == config.source_ledger_sha256
        and len(systems) == config.expected["system_count"]
        and len(sources) == config.expected["source_count"]
        and len(records) == config.expected["receipt_count"]
    )
    canary_passed = is_frozen and set(status_counts).issubset(_SUCCESS)
    expected_summary = {
        "canary_passed": canary_passed,
        "claim_boundary": config.claim_boundary,
        "empty_receipt_count": empty_count,
        "is_frozen_canary": is_frozen,
        "peak_count": len(peaks),
        "receipt_count": len(records),
        "rejection_totals": dict(sorted(rejection_totals.items())),
        "run_id": manifest["run_id"],
        "source_count": len(sources),
        "status_counts": dict(sorted(status_counts.items())),
        "system_count": len(systems),
        "warning_count": len(warnings),
    }
    if summary != expected_summary:
        raise PeakCanaryError("summary", "does not match re-executed receipts")
    identity = {
        "catalog_id": catalog.catalog_id,
        "catalog_sha256": catalog.sha256,
        "code_identity": _code_identity(root),
        "config_sha256": config.sha256,
        "phase3_lock_sha256": config.phase3_lock_sha256,
        "software_identity": _software_identity(),
        "source_ledger_sha256": source_sha,
        "system_ids": list(system_ids),
    }
    expected_run_id = hashlib.sha256(_RUN_DOMAIN + _canonical(identity)).hexdigest()
    expected_manifest = {
        **identity,
        "canary_passed": canary_passed,
        "claim_boundary": config.claim_boundary,
        "is_frozen_canary": is_frozen,
        "peak_count": len(peaks),
        "receipt_count": len(records),
        "run_id": expected_run_id,
        "schema_version": "phase3-peak-detection-v1-canary-run-v1",
        "source_count": len(sources),
        "system_count": len(systems),
    }
    if manifest != expected_manifest:
        raise PeakCanaryError("manifest", "does not match verified identities")
    marker_name = "complete.json" if canary_passed else "failed.json"
    if marker_names != [marker_name]:
        raise PeakCanaryError("marker", "does not match status")
    marker = _read_json(path / marker_name, "marker")
    expected_marker = {
        "canary_passed": canary_passed,
        "is_frozen_canary": is_frozen,
        "run_id": expected_run_id,
        "status": "complete" if canary_passed else "failed",
    }
    if marker != expected_marker:
        raise PeakCanaryError("marker", "does not match verified status")
    return PeakCanarySummary(
        path=path,
        run_id=expected_run_id,
        system_count=len(systems),
        source_count=len(sources),
        receipt_count=len(records),
        peak_count=len(peaks),
        empty_receipt_count=empty_count,
        status_counts=status_counts,
        canary_passed=canary_passed,
        is_frozen_canary=is_frozen,
    )


__all__ = [
    "PeakCanaryConfig",
    "PeakCanaryError",
    "PeakCanarySource",
    "PeakCanarySummary",
    "build_peak_canary_artifact",
    "load_peak_canary_config",
    "load_peak_canary_sources",
    "source_document",
    "verify_peak_canary_artifact",
]
