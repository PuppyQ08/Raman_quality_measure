from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from rpe.evaluation import Spectrum1D
from rpe.io.store import UnifiedDataset
from rpe.methods.catalog import Phase3ClassicalCatalog, Phase3System, TaskLine
from rpe.methods.classical.baseline import BaselineRunResult, BaselineRunStatus, run_baseline_system


_RUN_DOMAIN = b"rpe-phase3-baseline-canary-v1\0"
_PHASE1_SUBSET_SHA256 = "818bfddd9eb94cced9d486f4232a3e8b7e8f9505e8df6218e143b354b89a663f"
_PHASE1_SCIENTIFIC_CONFIG_SHA256 = "f44a1b51a9ff0eb7f4221d45d452ce7354bae5734432e54a0fda87617babd37e"
_PHASE1_SOURCE_SNAPSHOT_SHA256 = "8814d9a5e6b1f1c9b885a9fd84e1d8f3ceb3bfc9b69ebbcae62a0fd7aee774e7"
_BANDS = (
    ("b1_351_1726", 351, 1727),
    ("b2_1727_2326", 1727, 2327),
    ("b3_2327_3395", 2327, 3396),
    ("b4_3396_23775", 3396, 23776),
)


class BaselineCanaryError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class BaselineCanarySource:
    band_id: str
    selection_rank: int
    record_id: str
    sample_id: str
    class_label: int
    mineral_name: str
    point_count: int
    axis_sha256: str
    intensity_sha256: str
    spectrum: Spectrum1D


@dataclass(frozen=True)
class BaselineCanarySummary:
    path: Path
    run_id: str
    source_count: int
    system_count: int
    attempt_count: int
    status_counts: Mapping[str, int]
    canary_passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "status_counts", MappingProxyType(dict(sorted(self.status_counts.items()))))


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def load_rruff_baseline_canary_sources(
    phase1_run_path: Path, dataset_path: Path
) -> tuple[BaselineCanarySource, ...]:
    phase1_run_path = Path(phase1_run_path)
    subset_path = phase1_run_path / "source_subset.jsonl"
    if _sha(subset_path) != _PHASE1_SUBSET_SHA256:
        raise BaselineCanaryError("source_subset", "SHA256 mismatch")
    manifest = json.loads((phase1_run_path / "manifest.json").read_bytes())
    if manifest["selected_source_subset_sha256"] != _PHASE1_SUBSET_SHA256:
        raise BaselineCanaryError("manifest.selected_source_subset_sha256", "does not match")
    if manifest["scientific_config"]["sha256"] != _PHASE1_SCIENTIFIC_CONFIG_SHA256:
        raise BaselineCanaryError("manifest.scientific_config", "does not match")
    if manifest["source_snapshot_sha256"] != _PHASE1_SOURCE_SNAPSHOT_SHA256:
        raise BaselineCanaryError("manifest.source_snapshot_sha256", "does not match")
    rows = [json.loads(line) for line in subset_path.read_bytes().splitlines()]
    if len(rows) != 10000 or tuple(row["selection_rank"] for row in rows) != tuple(range(10000)):
        raise BaselineCanaryError("source_subset", "must contain the frozen consecutive 10k cohort")
    selected: list[BaselineCanarySource] = []
    with UnifiedDataset.open(Path(dataset_path), verify_checksums=True) as dataset:
        materialized = []
        for row in rows:
            record = dataset.get(str(row["record_id"]))
            axis = np.asarray(record.wavenumber, dtype="<f8")
            intensity = np.asarray(record.intensity, dtype="<f8")
            if np.all(np.diff(axis) < 0.0):
                axis = np.ascontiguousarray(axis[::-1])
                intensity = np.ascontiguousarray(intensity[::-1])
            spectrum = Spectrum1D(
                spectrum_id=f"{record.meta.dataset_id}::{record.record_id}",
                sample_id=record.meta.sample_id,
                axis_cm1=axis,
                intensity=intensity,
            )
            if _array_sha(spectrum.axis_cm1) != row.get(
                "normalized_axis_float64_sha256"
            ):
                raise BaselineCanaryError(
                    "source_subset.normalized axis", "SHA256 mismatch"
                )
            if _array_sha(spectrum.intensity) != row.get(
                "normalized_intensity_float64_sha256"
            ):
                raise BaselineCanaryError(
                    "source_subset.normalized intensity", "SHA256 mismatch"
                )
            materialized.append((row, spectrum))
        for band_id, lower, upper in _BANDS:
            candidates = [
                (row, spectrum)
                for row, spectrum in materialized
                if lower <= spectrum.intensity.size < upper
            ]
            candidates.sort(key=lambda item: (int(item[0]["selection_rank"]), str(item[0]["record_id"]).encode("utf-8")))
            if not candidates:
                raise BaselineCanaryError(band_id, "has no frozen source")
            row, spectrum = candidates[0]
            selected.append(
                BaselineCanarySource(
                    band_id=band_id, selection_rank=int(row["selection_rank"]),
                    record_id=str(row["record_id"]), sample_id=str(row["sample_id"]),
                    class_label=int(row["class_label"]), mineral_name=str(row["mineral_name"]),
                    point_count=spectrum.intensity.size, axis_sha256=_array_sha(spectrum.axis_cm1),
                    intensity_sha256=_array_sha(spectrum.intensity), spectrum=spectrum,
                )
            )
    return tuple(selected)


def _source_document(source: BaselineCanarySource) -> dict[str, object]:
    return {
        "axis_sha256": source.axis_sha256, "band_id": source.band_id,
        "class_label": source.class_label, "intensity_sha256": source.intensity_sha256,
        "mineral_name": source.mineral_name, "point_count": source.point_count,
        "record_id": source.record_id, "sample_id": source.sample_id,
        "selection_rank": source.selection_rank,
    }


def _result_document(source: BaselineCanarySource, result: BaselineRunResult) -> dict[str, object]:
    return {
        "band_id": source.band_id, "baseline_sha256": result.baseline_sha256,
        "corrected_sha256": result.corrected_sha256, "diagnostics": dict(result.diagnostics),
        "error_code": result.error_code, "error_message": result.error_message,
        "family_id": result.family_id, "method_id": result.method_id,
        "point_count": source.point_count, "record_id": source.record_id,
        "selection_rank": source.selection_rank, "status": result.status.value,
        "system_id": result.system_id,
        "warnings": [{"category": item.category, "message": item.message} for item in result.warnings],
    }


def build_baseline_canary_from_sources(
    sources: Sequence[BaselineCanarySource],
    systems: Sequence[Phase3System],
    output_root: Path,
    *,
    catalog: Phase3ClassicalCatalog,
    project_root: Path,
) -> BaselineCanarySummary:
    frozen_sources = tuple(sources)
    frozen_systems = tuple(sorted(systems, key=lambda item: item.system_id))
    if not frozen_sources or not frozen_systems:
        raise BaselineCanaryError("inputs", "must not be empty")
    if any(system.task_line is not TaskLine.BASELINE_CORRECTION for system in frozen_systems):
        raise BaselineCanaryError("systems", "must all be baseline correction")
    if len({source.record_id for source in frozen_sources}) != len(frozen_sources):
        raise BaselineCanaryError("sources", "record IDs must be unique")
    if len({system.system_id for system in frozen_systems}) != len(frozen_systems):
        raise BaselineCanaryError("systems", "system IDs must be unique")
    catalog_system_ids = {system.system_id for system in catalog.systems}
    if any(system.system_id not in catalog_system_ids for system in frozen_systems):
        raise BaselineCanaryError("catalog membership", "system is not in catalog")
    for source in frozen_sources:
        if source.point_count != source.spectrum.intensity.size:
            raise BaselineCanaryError(
                "source.point_count", "does not match spectrum"
            )
        if source.axis_sha256 != _array_sha(source.spectrum.axis_cm1):
            raise BaselineCanaryError("source.axis_sha256", "does not match spectrum")
        if source.intensity_sha256 != _array_sha(source.spectrum.intensity):
            raise BaselineCanaryError(
                "source.intensity_sha256", "does not match spectrum"
            )
    source_documents = [_source_document(source) for source in frozen_sources]
    code_paths = (
        "rpe/methods/catalog.py",
        "rpe/methods/classical/baseline.py",
        "rpe/methods/classical/baseline_canary.py",
    )
    code_identity = {path: _sha(Path(project_root) / path) for path in code_paths}
    identity = {
        "catalog_id": catalog.catalog_id, "catalog_sha256": catalog.sha256,
        "code_identity": code_identity,
        "phase1_scientific_config_sha256": _PHASE1_SCIENTIFIC_CONFIG_SHA256,
        "phase1_selected_source_subset_sha256": _PHASE1_SUBSET_SHA256,
        "phase1_source_snapshot_sha256": _PHASE1_SOURCE_SNAPSHOT_SHA256,
        "phase3_lock_sha256": catalog.dependency_lock.sha256,
        "source_subset_sha256": hashlib.sha256(b"".join(_canonical(value) for value in source_documents)).hexdigest(),
        "system_ids": [system.system_id for system in frozen_systems],
    }
    run_id = hashlib.sha256(_RUN_DOMAIN + _canonical(identity)).hexdigest()
    path = Path(output_root) / f"phase3-baseline-canary-{run_id}"
    if path.exists():
        raise BaselineCanaryError("output", "run path already exists")
    path.mkdir(parents=True)
    result_documents = []
    for system in frozen_systems:
        for source in frozen_sources:
            result_documents.append(_result_document(source, run_baseline_system(system, source.spectrum)))
    result_documents.sort(key=lambda value: (str(value["system_id"]), str(value["band_id"]), str(value["record_id"])))
    status_counts = Counter(str(value["status"]) for value in result_documents)
    attempt_count = len(result_documents)
    canary_passed = (
        attempt_count == len(frozen_sources) * len(frozen_systems)
        and not any(status in status_counts for status in (BaselineRunStatus.FAILED_RUNTIME.value,))
    )
    manifest = {
        **identity, "attempt_count": attempt_count,
        "claim_boundary": "infrastructure_canary_not_scientific_performance_or_phase5_power",
        "run_id": run_id, "schema_version": "phase3-baseline-canary-v1",
        "source_count": len(frozen_sources), "system_count": len(frozen_systems),
    }
    summary = {
        "attempt_count": attempt_count, "canary_passed": canary_passed,
        "claim_boundary": manifest["claim_boundary"], "run_id": run_id,
        "source_count": len(frozen_sources), "status_counts": dict(sorted(status_counts.items())),
        "system_count": len(frozen_systems),
    }
    scientific_payloads = {
        "manifest.json": _canonical(manifest),
        "source_subset.jsonl": b"".join(_canonical(value) for value in source_documents),
        "summary.json": _canonical(summary),
        "system_results.jsonl": b"".join(_canonical(value) for value in result_documents),
    }
    marker_name = "complete.json" if canary_passed else "failed.json"
    marker = {
        "canary_passed": canary_passed, "run_id": run_id,
        "status": "complete" if canary_passed else "failed",
    }
    payloads = {**scientific_payloads, marker_name: _canonical(marker)}
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    (path / "SHA256SUMS").write_text(
        "".join(f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in sorted(payloads)),
        encoding="utf-8",
    )
    return BaselineCanarySummary(path, run_id, len(frozen_sources), len(frozen_systems), attempt_count, status_counts, canary_passed)


def verify_baseline_canary_artifact(path: Path) -> BaselineCanarySummary:
    path = Path(path)
    try:
        checksum_lines = (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BaselineCanaryError("checksum", str(error)) from error
    observed_names = set()
    for line in checksum_lines:
        try:
            digest, name = line.split("  ")
        except ValueError as error:
            raise BaselineCanaryError("checksum", "invalid line") from error
        observed_names.add(name)
        if _sha(path / name) != digest:
            raise BaselineCanaryError("checksum", f"{name} mismatch")
    manifest = json.loads((path / "manifest.json").read_bytes())
    summary = json.loads((path / "summary.json").read_bytes())
    rows = [json.loads(line) for line in (path / "system_results.jsonl").read_bytes().splitlines()]
    if len(rows) != summary["attempt_count"]:
        raise BaselineCanaryError("system_results", "attempt count mismatch")
    status_counts = Counter(str(row["status"]) for row in rows)
    if dict(sorted(status_counts.items())) != summary["status_counts"]:
        raise BaselineCanaryError("summary.status_counts", "does not match rows")
    marker_name = "complete.json" if summary["canary_passed"] else "failed.json"
    expected_names = {
        "manifest.json", "source_subset.jsonl", "summary.json",
        "system_results.jsonl", marker_name,
    }
    if observed_names != expected_names:
        raise BaselineCanaryError("checksum", "inventory mismatch")
    if not (path / marker_name).is_file() or (path / ("failed.json" if marker_name == "complete.json" else "complete.json")).exists():
        raise BaselineCanaryError("marker", "must contain exactly one terminal marker")
    marker = json.loads((path / marker_name).read_bytes())
    if marker["run_id"] != manifest["run_id"] or marker["run_id"] != summary["run_id"]:
        raise BaselineCanaryError("run_id", "does not match")
    expected_marker_status = "complete" if summary["canary_passed"] else "failed"
    if (
        marker.get("status") != expected_marker_status
        or marker.get("canary_passed") is not summary["canary_passed"]
    ):
        raise BaselineCanaryError("marker", "does not match summary status")
    source_payload = (path / "source_subset.jsonl").read_bytes()
    if hashlib.sha256(source_payload).hexdigest() != manifest.get("source_subset_sha256"):
        raise BaselineCanaryError("source_subset_sha256", "does not match manifest")
    if manifest.get("attempt_count") != len(rows):
        raise BaselineCanaryError("manifest.attempt_count", "does not match rows")
    if manifest.get("source_count") != summary.get("source_count"):
        raise BaselineCanaryError("manifest.source_count", "does not match summary")
    if manifest.get("system_count") != summary.get("system_count"):
        raise BaselineCanaryError("manifest.system_count", "does not match summary")
    identity_keys = (
        "catalog_id", "catalog_sha256", "code_identity",
        "phase1_scientific_config_sha256",
        "phase1_selected_source_subset_sha256",
        "phase1_source_snapshot_sha256", "phase3_lock_sha256",
        "source_subset_sha256", "system_ids",
    )
    identity = {key: manifest[key] for key in identity_keys}
    expected_run_id = hashlib.sha256(_RUN_DOMAIN + _canonical(identity)).hexdigest()
    if expected_run_id != manifest.get("run_id"):
        raise BaselineCanaryError("run_id", "does not match identity")
    return BaselineCanarySummary(
        path=path, run_id=str(summary["run_id"]), source_count=int(summary["source_count"]),
        system_count=int(summary["system_count"]), attempt_count=int(summary["attempt_count"]),
        status_counts=status_counts, canary_passed=bool(summary["canary_passed"]),
    )


__all__ = [
    "BaselineCanaryError", "BaselineCanarySource", "BaselineCanarySummary",
    "build_baseline_canary_from_sources", "load_rruff_baseline_canary_sources",
    "verify_baseline_canary_artifact",
]
