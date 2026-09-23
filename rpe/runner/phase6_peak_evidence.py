from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.evaluation import Peak1D, Spectrum1D
from rpe.io.perturbed_store import StoredCell, StoredSource, read_perturbed_shard
from rpe.methods.catalog import Phase3System, TaskLine, load_classical_catalog
from rpe.methods.classical.peaks import (
    DetectedPeak1D,
    PeakDetectionRunResult,
    PeakRunStatus,
    run_peak_detection_system,
)
from rpe.metrics.peak import _match_peaks
from rpe.perturb import PeakFamilyPreparedState, PerturbationContext
from rpe.perturb.sweep import load_perturbation_sweep_config
from rpe.runner.phase1_perturbations import operator_for


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/peak_evidence_v1.json"
CONFIG_BYTES = 12007
CONFIG_SHA256 = "837c0e3c16db1b651bcb2ffebe37b10551bdecdc19019685cd472e0ebd148959"
RUN_DOMAIN = b"rpe-phase6-peak-evidence-v1\0"
TRUTH_DOMAIN = b"rpe-phase6-peak-intervention-truth-v1\0"
SUCCESS = frozenset({PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING})
PERTURBATION_ORDER = ("p01", "p02", "p03", "p04", "p05")
ENDPOINTS_BY_PERTURBATION = {
    "p01": ("P1-retention",),
    "p02": ("P2-affected-retention", "P2-unaffected-retention"),
    "p03": ("P3-retention", "P3-position-mae", "P3-signed-log2-fwhm-ratio"),
    "p04": ("P4-deleted-event-disappearance", "P4-undeleted-retention"),
    "p05": ("P5-inserted-event-detection-gain", "P5-native-peak-retention"),
}
SYNTHETIC_ENDPOINT_ORDER = tuple(
    endpoint
    for perturbation_id in PERTURBATION_ORDER
    for endpoint in ENDPOINTS_BY_PERTURBATION[perturbation_id]
)
_WORKER_INPUTS: "PeakEvidenceInputs | None" = None
_WORKER_SYSTEMS: tuple[Phase3System, ...] = ()


class PeakEvidenceError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class PeakEvidenceConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    authorities: Mapping[str, Mapping[str, object]]
    artifact_contract: Mapping[str, object]
    bootstrap: Mapping[str, object]
    cohort: Mapping[str, object]
    endpoint_manifest: tuple[Mapping[str, object], ...]
    expected: Mapping[str, int]
    positive_alpha_grid: tuple[float, ...]
    alpha_grid: tuple[float, ...]
    tolerances_cm1: tuple[float, ...]
    promoted_system_ids: tuple[str, ...]
    blocked_systems: tuple[Mapping[str, str], ...]
    phase5_power_state: str
    source_revision_status: str
    claim_boundary: str
    document: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "authorities", _freeze_mapping(self.authorities))
        object.__setattr__(self, "artifact_contract", _freeze_mapping(self.artifact_contract))
        object.__setattr__(self, "bootstrap", _freeze_mapping(self.bootstrap))
        object.__setattr__(self, "cohort", _freeze_mapping(self.cohort))
        object.__setattr__(self, "expected", _freeze_mapping(self.expected))
        object.__setattr__(self, "endpoint_manifest", tuple(_freeze_mapping(row) for row in self.endpoint_manifest))
        object.__setattr__(self, "blocked_systems", tuple(_freeze_mapping(row) for row in self.blocked_systems))
        object.__setattr__(self, "document", _freeze_mapping(self.document))


@dataclass(frozen=True)
class PeakEvidenceCondition:
    perturbation_id: str
    alpha: float
    alpha_float64_le_hex: str
    state_digest: str
    output_spectrum_id: str
    output_intensity_sha256: str
    truth: Mapping[str, object]
    truth_digest: str
    spectrum: Spectrum1D

    def __post_init__(self) -> None:
        object.__setattr__(self, "truth", _freeze_mapping(self.truth))


@dataclass(frozen=True)
class PeakEvidenceIntervention:
    perturbation_id: str
    state_digest: str
    conditions: tuple[PeakEvidenceCondition, ...]


@dataclass(frozen=True)
class PeakEvidenceSource:
    class_label: int
    selection_rank: int
    source_record_id: str
    spectrum: Spectrum1D
    interventions: tuple[PeakEvidenceIntervention, ...]


@dataclass(frozen=True)
class PeakEvidenceInputs:
    synthetic_fixture: bool
    cohort_id: str
    cohort_projection: bytes
    sources: tuple[PeakEvidenceSource, ...]
    cohort_identity: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cohort_projection", bytes(self.cohort_projection))
        object.__setattr__(self, "cohort_identity", _freeze_mapping(self.cohort_identity))

    @property
    def source_count(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class PeakEvidenceSummary:
    path: Path
    run_id: str
    status: str
    system_count: int
    source_count: int
    intervention_truth_row_count: int
    detector_receipt_row_count: int
    detector_call_count: int
    condition_row_count: int
    class_row_count: int
    bootstrap_row_count: int
    method_evidence_row_count: int


def _canonical(value: object) -> bytes:
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
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _freeze_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    raise PeakEvidenceError("freeze", f"unsupported value type {type(value).__name__}")


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in sorted(value.items())}
    )


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _normalize_identity(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PeakEvidenceError(path, "must be an object")
    byte_count = value.get("byte_count")
    if not isinstance(value.get("path"), str) or not value["path"]:
        raise PeakEvidenceError(path, "path must be nonempty")
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
        raise PeakEvidenceError(path, "byte_count must be nonnegative integer")
    sha256 = value.get("sha256")
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise PeakEvidenceError(path, "sha256 must be lowercase hexadecimal")
    return MappingProxyType({"path": value["path"], "byte_count": byte_count, "sha256": sha256})


def load_phase6_peak_evidence_config(path: Path = DEFAULT_CONFIG) -> PeakEvidenceConfig:
    path = Path(path)
    raw = path.read_bytes()
    document = json.loads(raw)
    if raw != _canonical(document):
        raise PeakEvidenceError("config", "must be canonical JSON")
    if path.resolve() == DEFAULT_CONFIG.resolve() and (
        len(raw) != CONFIG_BYTES or hashlib.sha256(raw).hexdigest() != CONFIG_SHA256
    ):
        raise PeakEvidenceError("config", "frozen byte identity mismatch")
    if document.get("schema_version") != "phase6-peak-evidence-v1":
        raise PeakEvidenceError("schema_version", "unexpected config schema")
    authorities = {
        str(name): _normalize_identity(f"authorities.{name}", identity)
        for name, identity in dict(document["authorities"]).items()
    }
    promoted = tuple(str(value) for value in document["promoted_system_ids"])
    blocked = tuple(dict(value) for value in document["blocked_systems"])
    if len(promoted) != 21 or len(set(promoted)) != 21:
        raise PeakEvidenceError("promoted_system_ids", "must contain exactly 21 unique IDs")
    if len(blocked) != 15 or len({str(row["system_id"]) for row in blocked}) != 15:
        raise PeakEvidenceError("blocked_systems", "must contain exactly 15 unique systems")
    promotion = json.loads((ROOT / str(authorities["peak_promotion"]["path"])).read_text(encoding="utf-8"))
    if tuple(promotion.get("promoted_system_ids", ())) != promoted:
        raise PeakEvidenceError("promoted_system_ids", "must match frozen promotion order")
    catalog = load_classical_catalog(ROOT / str(authorities["classical_catalog"]["path"]))
    all_peak = {system.system_id: system for system in catalog.systems if system.task_line is TaskLine.PEAK_DETECTION}
    if set(promoted) | {str(row["system_id"]) for row in blocked} != set(all_peak):
        raise PeakEvidenceError("system inventory", "must cover the exact 36 planned peak systems")
    payloads = tuple(document["artifact_contract"]["payload_files"])
    expected_payloads = (
        "config.json", "authority_bridge.json", "preflight.json",
        "system_status.jsonl", "intervention_truth.jsonl",
        "detector_receipts.jsonl", "synthetic_condition_rows.jsonl",
        "synthetic_class_rows.jsonl", "bootstrap_results.jsonl",
        "method_evidence_rows.csv", "family_projection.csv", "manifest.json",
    )
    if payloads != expected_payloads:
        raise PeakEvidenceError("artifact_contract", "payload inventory mismatch")
    endpoint_manifest = tuple(dict(row) for row in document["endpoint_manifest"])
    if tuple(str(row["endpoint_id"]) for row in endpoint_manifest[3:]) != SYNTHETIC_ENDPOINT_ORDER:
        raise PeakEvidenceError("endpoint_manifest", "synthetic endpoint order mismatch")
    return PeakEvidenceConfig(
        path=path,
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        schema_version=str(document["schema_version"]),
        authorities=authorities,
        artifact_contract=dict(document["artifact_contract"]),
        bootstrap=dict(document["bootstrap"]),
        cohort=dict(document["cohort"]),
        endpoint_manifest=endpoint_manifest,
        expected={str(key): int(value) for key, value in dict(document["expected"]).items()},
        positive_alpha_grid=tuple(float(value) for value in document["positive_alpha_grid"]),
        alpha_grid=tuple(float(value) for value in document["alpha_grid"]),
        tolerances_cm1=tuple(float(value) for value in document["tolerances_cm1"]),
        promoted_system_ids=promoted,
        blocked_systems=blocked,
        phase5_power_state=str(document["phase5_power_state"]),
        source_revision_status=str(document["source_revision_status"]),
        claim_boundary=str(document["claim_boundary"]),
        document=document,
    )


def _validate_authorities(config: PeakEvidenceConfig, project_root: Path) -> None:
    for name, identity in config.authorities.items():
        path = project_root / str(identity["path"])
        if not path.is_file():
            raise PeakEvidenceError(name, "authority path is missing")
        if path.stat().st_size != int(identity["byte_count"]):
            raise PeakEvidenceError(name, "authority byte count mismatch")
        if _sha_file(path) != str(identity["sha256"]):
            raise PeakEvidenceError(name, "authority SHA256 mismatch")


def _verify_phase1_artifact_checksums(run_path: Path) -> None:
    run_path = Path(run_path)
    ledger_path = run_path / "SHA256SUMS"
    if not ledger_path.is_file() or ledger_path.is_symlink():
        raise PeakEvidenceError("phase1 checksum ledger", "is missing or a symlink")
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    entries: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        fields = line.split("  ", 1)
        if len(fields) != 2:
            raise PeakEvidenceError("phase1 checksum ledger", f"invalid line {index}")
        digest, relative = fields
        relative_path = Path(relative)
        if (
            len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise PeakEvidenceError("phase1 checksum ledger", f"invalid member {relative!r}")
        entries.append((relative, digest))
    names = tuple(relative for relative, _ in entries)
    if names != tuple(sorted(names)) or len(names) != len(set(names)):
        raise PeakEvidenceError("phase1 checksum ledger", "members must be unique and sorted")
    observed_names = tuple(
        sorted(
            path.relative_to(run_path).as_posix()
            for path in run_path.rglob("*")
            if path.is_file()
            and path.name not in {"SHA256SUMS", "complete.json", "failed.json"}
        )
    )
    if names != observed_names:
        raise PeakEvidenceError("phase1 checksum ledger", "member inventory mismatch")
    for relative, digest in entries:
        target = run_path / relative
        if target.is_symlink() or _sha_file(target) != digest:
            raise PeakEvidenceError("phase1 checksum ledger", f"checksum mismatch for {relative}")


def derive_peak_evidence_cohort_projection(phase1_run_path: Path) -> bytes:
    run_path = Path(phase1_run_path)
    selected: dict[int, Mapping[str, object]] = {}
    cell_paths = sorted((run_path / "shards").glob("[0-9][0-9][0-9][0-9][0-9]/cells.jsonl"))
    if not cell_paths:
        raise PeakEvidenceError("phase1 shards", "no cell ledgers found")
    complete_source_count = 0
    for path in cell_paths:
        with path.open("rb") as stream:
            for raw in stream:
                row = json.loads(raw)
                if row.get("perturbation_id") != "p05" or row.get("status") != "complete":
                    continue
                complete_source_count += 1
                source = row["source"]
                class_label = int(source["class_label"])
                candidate = {
                    "class_label": class_label,
                    "selection_rank": int(source["selection_rank"]),
                    "source_record_id": str(source["source_record_id"]),
                }
                previous = selected.get(class_label)
                if previous is None or int(candidate["selection_rank"]) < int(previous["selection_rank"]):
                    selected[class_label] = candidate
    rows = sorted(selected.values(), key=lambda row: (int(row["class_label"]), int(row["selection_rank"])))
    if complete_source_count != 7819:
        raise PeakEvidenceError("P5 complete cohort", f"expected 7819 sources, observed {complete_source_count}")
    if len(rows) != 2250:
        raise PeakEvidenceError("cohort projection", f"expected 2250 classes, observed {len(rows)}")
    return _jsonl_bytes(rows)


def _as_peak_tuple(values: Sequence[Peak1D | DetectedPeak1D]) -> tuple[Peak1D, ...]:
    converted = tuple(value.to_peak1d() if isinstance(value, DetectedPeak1D) else value for value in values)
    if any(not isinstance(value, Peak1D) for value in converted):
        raise PeakEvidenceError("peaks", "must contain Peak1D or DetectedPeak1D values")
    return converted


def _position_peaks(values: Sequence[float]) -> tuple[Peak1D, ...]:
    return tuple(Peak1D(float(value), None, None, None, None) for value in values)


def _matched_indexes(
    reference: Sequence[Peak1D | DetectedPeak1D],
    candidate: Sequence[Peak1D | DetectedPeak1D],
    tolerance_cm1: float,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (pair.reference_index, pair.candidate_index)
        for pair in _match_peaks(
            _as_peak_tuple(reference),
            _as_peak_tuple(candidate),
            tolerance_cm1=float(tolerance_cm1),
        )
    )


def _endpoint_row(
    endpoint_id: str,
    numerator: float | int,
    denominator: int,
    *,
    empty_reason: str,
) -> Mapping[str, object]:
    if denominator == 0:
        return {
            "endpoint_id": endpoint_id,
            "numerator": 0,
            "denominator": 0,
            "value": None,
            "state": "complete_empty_risk_set",
            "reason_code": empty_reason,
        }
    return {
        "endpoint_id": endpoint_id,
        "numerator": numerator,
        "denominator": denominator,
        "value": float(numerator) / denominator,
        "state": "complete_numeric",
        "reason_code": "",
    }


def evaluate_mechanism_condition(
    *,
    perturbation_id: str,
    baseline_peaks: Sequence[Peak1D | DetectedPeak1D],
    candidate_peaks: Sequence[Peak1D | DetectedPeak1D],
    truth: Mapping[str, object],
    tolerance_cm1: float,
) -> tuple[Mapping[str, object], ...]:
    baseline = _as_peak_tuple(baseline_peaks)
    candidate = _as_peak_tuple(candidate_peaks)
    tolerance = float(tolerance_cm1)
    matches = _matched_indexes(baseline, candidate, tolerance)
    matched_baseline = {left for left, _ in matches}

    if perturbation_id == "p01":
        return (_endpoint_row("P1-retention", len(matches), len(baseline), empty_reason="no_baseline_peaks"),)

    if perturbation_id == "p02":
        supports = tuple(tuple(float(part) for part in pair) for pair in truth.get("affected_supports_cm1", ()))
        affected = {
            index
            for index, value in enumerate(baseline)
            if any(lower <= value.position_cm1 <= upper for lower, upper in supports)
        }
        unaffected = set(range(len(baseline))) - affected
        return (
            _endpoint_row("P2-affected-retention", len(affected & matched_baseline), len(affected), empty_reason="no_affected_baseline_peaks"),
            _endpoint_row("P2-unaffected-retention", len(unaffected & matched_baseline), len(unaffected), empty_reason="no_unaffected_baseline_peaks"),
        )

    if perturbation_id == "p03":
        retention = _endpoint_row("P3-retention", len(matches), len(baseline), empty_reason="no_baseline_peaks")
        if not matches:
            empty = {
                "numerator": 0, "denominator": 0, "value": None,
                "state": "complete_empty_risk_set",
                "reason_code": "no_matched_baseline_peaks",
            }
            return (retention, {"endpoint_id": "P3-position-mae", **empty}, {"endpoint_id": "P3-signed-log2-fwhm-ratio", **empty})
        position_total = sum(abs(candidate[right].position_cm1 - baseline[left].position_cm1) for left, right in matches)
        width_total = sum(math.log2(float(candidate[right].fwhm_cm1) / float(baseline[left].fwhm_cm1)) for left, right in matches)
        return (
            retention,
            _endpoint_row("P3-position-mae", position_total, len(matches), empty_reason="no_matched_baseline_peaks"),
            _endpoint_row("P3-signed-log2-fwhm-ratio", width_total, len(matches), empty_reason="no_matched_baseline_peaks"),
        )

    if perturbation_id == "p04":
        event_peaks = _position_peaks(tuple(float(value) for value in truth.get("deleted_centers_cm1", ())))
        event_to_baseline = _matched_indexes(event_peaks, baseline, tolerance)
        eligible_event_indexes = tuple(left for left, _ in event_to_baseline)
        deleted_baseline_indexes = {right for _, right in event_to_baseline}
        eligible_events = tuple(event_peaks[index] for index in eligible_event_indexes)
        event_to_candidate_local = _matched_indexes(eligible_events, candidate, tolerance)
        consumed_candidate = {right for _, right in event_to_candidate_local}
        disappearance = _endpoint_row(
            "P4-deleted-event-disappearance",
            len(eligible_events) - len(event_to_candidate_local),
            len(eligible_events),
            empty_reason="no_baseline_matched_deleted_event",
        )
        remaining_baseline = tuple(value for index, value in enumerate(baseline) if index not in deleted_baseline_indexes)
        remaining_candidate = tuple(value for index, value in enumerate(candidate) if index not in consumed_candidate)
        remaining_matches = _matched_indexes(remaining_baseline, remaining_candidate, tolerance)
        retention = _endpoint_row("P4-undeleted-retention", len(remaining_matches), len(remaining_baseline), empty_reason="no_undeleted_baseline_peaks")
        return disappearance, retention

    if perturbation_id == "p05":
        inserted = _position_peaks(tuple(float(value) for value in truth.get("inserted_centers_cm1", ())))
        if not inserted:
            return tuple(
                {
                    "endpoint_id": endpoint_id,
                    "numerator": 0,
                    "denominator": 0,
                    "value": None,
                    "state": "not_applicable_zero_synthetic_events",
                    "reason_code": "not_applicable_zero_synthetic_events",
                }
                for endpoint_id in ENDPOINTS_BY_PERTURBATION["p05"]
            )
        inserted_to_baseline = _matched_indexes(inserted, baseline, tolerance)
        inserted_to_candidate = _matched_indexes(inserted, candidate, tolerance)
        gain = _endpoint_row(
            "P5-inserted-event-detection-gain",
            len(inserted_to_candidate) - len(inserted_to_baseline),
            len(inserted),
            empty_reason="not_applicable_zero_synthetic_events",
        )
        consumed_baseline = {right for _, right in inserted_to_baseline}
        consumed_candidate = {right for _, right in inserted_to_candidate}
        remaining_baseline = tuple(value for index, value in enumerate(baseline) if index not in consumed_baseline)
        remaining_candidate = tuple(value for index, value in enumerate(candidate) if index not in consumed_candidate)
        remaining_matches = _matched_indexes(remaining_baseline, remaining_candidate, tolerance)
        retention = _endpoint_row("P5-native-peak-retention", len(remaining_matches), len(remaining_baseline), empty_reason="no_native_baseline_peaks")
        return gain, retention

    raise PeakEvidenceError("perturbation_id", f"unsupported {perturbation_id!r}")


def _weak_order(state: PeakFamilyPreparedState) -> tuple[int, ...]:
    return tuple(
        index
        for _, _, index in sorted(
            (
                (state.prominences[index], state.peak_positions_cm1[index], index)
                for index in range(len(state.peak_indices))
            ),
            key=lambda value: (value[0], value[1]),
        )
    )


def _condition_truth(
    perturbation_id: str,
    state: PeakFamilyPreparedState,
    source: Spectrum1D,
    alpha: float,
    diagnostics: Mapping[str, object],
) -> Mapping[str, object]:
    count = len(state.peak_indices)
    if perturbation_id == "p01":
        return {"attenuation_fraction": alpha, "affected_component_count": count}
    if perturbation_id in {"p02", "p04"}:
        selected_count = min(int(math.ceil(alpha * count)), count)
        selected = _weak_order(state)[:selected_count]
        supports = tuple(
            (float(source.axis_cm1[state.support_bounds[index][0]]), float(source.axis_cm1[state.support_bounds[index][1]]))
            for index in selected
        )
        centers = tuple(float(state.peak_positions_cm1[index]) for index in selected)
        if perturbation_id == "p02":
            return {"affected_component_count": selected_count, "affected_supports_cm1": supports, "attenuation_fraction": 0.5}
        return {"deleted_component_count": selected_count, "deleted_centers_cm1": centers, "deleted_supports_cm1": supports}
    if perturbation_id == "p03":
        return {
            "replacement_centers_cm1": tuple(float(value) for value in state.peak_positions_cm1),
            "replacement_component_count": count,
            "equal_area": True,
            "imposed_sigma_cm1": float(diagnostics["broadened_sigma_cm1"]),
        }
    if perturbation_id == "p05":
        selected_count = min(int(math.floor(alpha * count)), len(state.candidate_false_centers_cm1))
        return {
            "inserted_component_count": selected_count,
            "inserted_centers_cm1": tuple(float(value) for value in state.candidate_false_centers_cm1[:selected_count]),
            "inserted_height": float(diagnostics["false_peak_height"]),
            "inserted_fwhm_cm1": float(diagnostics["false_peak_fwhm_cm1"]),
            "nested_prefix": True,
        }
    raise PeakEvidenceError("perturbation_id", f"unsupported {perturbation_id!r}")


def _build_intervention(
    *,
    source: Spectrum1D,
    perturbation_id: str,
    sweep,
    stored_cell: StoredCell | None,
) -> PeakEvidenceIntervention:
    operator = operator_for(perturbation_id, sweep)
    context = PerturbationContext(sweep.sweep_id, sweep.sha256, sweep.global_seed)
    state = operator.prepare(source, context)
    if not isinstance(state, PeakFamilyPreparedState):
        raise PeakEvidenceError("intervention state", "must be PeakFamilyPreparedState")
    if stored_cell is not None:
        if stored_cell.status != "complete" or stored_cell.state_digest != state.state_digest:
            raise PeakEvidenceError("intervention state", "does not match Phase-1 stored cell")
        stored_by_alpha = {record.alpha: record for record in stored_cell.records}
    else:
        stored_by_alpha = {}
    conditions: list[PeakEvidenceCondition] = []
    for alpha in sweep.alpha_grid:
        result = operator.apply(source, alpha, state)
        stored = stored_by_alpha.get(alpha)
        if stored_cell is not None:
            if stored is None:
                raise PeakEvidenceError("intervention output", "stored alpha is missing")
            if (
                stored.output_spectrum_id != result.output.spectrum_id
                or stored.state_digest != result.state_digest
                or not np.array_equal(stored.axis_cm1, result.output.axis_cm1)
                or not np.array_equal(stored.intensity, result.output.intensity)
            ):
                raise PeakEvidenceError("intervention output", "does not match Phase-1 stored bytes")
        if alpha == 0.0:
            if not np.array_equal(result.output.axis_cm1, source.axis_cm1) or not np.array_equal(result.output.intensity, source.intensity):
                raise PeakEvidenceError("alpha zero", "must preserve source exactly")
            continue
        truth = _condition_truth(perturbation_id, state, source, float(alpha), result.diagnostics)
        truth_digest = hashlib.sha256(TRUTH_DOMAIN + _canonical(truth)).hexdigest()
        conditions.append(
            PeakEvidenceCondition(
                perturbation_id=perturbation_id,
                alpha=float(alpha),
                alpha_float64_le_hex=struct.pack("<d", float(alpha)).hex(),
                state_digest=state.state_digest,
                output_spectrum_id=result.output.spectrum_id,
                output_intensity_sha256=_array_sha(result.output.intensity),
                truth=truth,
                truth_digest=truth_digest,
                spectrum=result.output,
            )
        )
    return PeakEvidenceIntervention(perturbation_id, state.state_digest, tuple(conditions))


def _source_input(
    *,
    class_label: int,
    selection_rank: int,
    source_record_id: str,
    spectrum: Spectrum1D,
    sweep,
    stored_cells: Mapping[str, StoredCell] | None = None,
) -> PeakEvidenceSource:
    interventions = tuple(
        _build_intervention(
            source=spectrum,
            perturbation_id=perturbation_id,
            sweep=sweep,
            stored_cell=None if stored_cells is None else stored_cells[perturbation_id],
        )
        for perturbation_id in PERTURBATION_ORDER
    )
    return PeakEvidenceSource(class_label, selection_rank, source_record_id, spectrum, interventions)


def make_synthetic_peak_evidence_inputs(
    *, config: PeakEvidenceConfig | None = None
) -> PeakEvidenceInputs:
    config = load_phase6_peak_evidence_config() if config is None else config
    sweep = load_perturbation_sweep_config(ROOT / str(config.authorities["shared_sweep"]["path"]))
    axis = np.linspace(100.0, 500.0, 401, dtype="<f8")
    sources = []
    for index, shift in enumerate((0.0, 3.0)):
        intensity = np.full(axis.size, 1.0, dtype="<f8")
        for center, height, sigma in ((145.0 + shift, 9.0, 2.5), (220.0 + shift, 6.0, 3.0), (310.0 + shift, 12.0, 2.0), (405.0 + shift, 7.0, 3.5)):
            intensity += height * np.exp(-0.5 * ((axis - center) / sigma) ** 2)
        spectrum = Spectrum1D(f"fixture-source-{index}", f"fixture-sample-{index}", axis, intensity)
        sources.append(_source_input(class_label=index + 1, selection_rank=index, source_record_id=f"fixture-record-{index}", spectrum=spectrum, sweep=sweep))
    projection = _jsonl_bytes(
        {"class_label": source.class_label, "selection_rank": source.selection_rank, "source_record_id": source.source_record_id}
        for source in sources
    )
    return PeakEvidenceInputs(
        synthetic_fixture=True,
        cohort_id="synthetic_peak_evidence_fixture",
        cohort_projection=projection,
        sources=tuple(sources),
        cohort_identity={"class_count": len(sources), "projection_sha256": hashlib.sha256(projection).hexdigest()},
    )


def _load_real_inputs(config: PeakEvidenceConfig, project_root: Path) -> PeakEvidenceInputs:
    _validate_authorities(config, project_root)
    phase1_run = (project_root / str(config.authorities["phase1_manifest"]["path"])).parent
    _verify_phase1_artifact_checksums(phase1_run)
    projection = derive_peak_evidence_cohort_projection(phase1_run)
    projection_sha = hashlib.sha256(projection).hexdigest()
    if projection_sha != str(config.cohort["projection_sha256"]):
        raise PeakEvidenceError("cohort projection", "SHA256 mismatch")
    projection_rows = [json.loads(line) for line in projection.splitlines()]
    selected_by_shard: dict[int, list[Mapping[str, object]]] = {}
    for row in projection_rows:
        selected_by_shard.setdefault(int(row["selection_rank"]) // 100, []).append(row)
    sweep = load_perturbation_sweep_config(project_root / str(config.authorities["shared_sweep"]["path"]))
    built: dict[int, PeakEvidenceSource] = {}
    for shard_index in sorted(selected_by_shard):
        shard = read_perturbed_shard(phase1_run / "shards" / f"{shard_index:05d}")
        for row in selected_by_shard[shard_index]:
            rank = int(row["selection_rank"])
            local_index = rank - shard_index * 100
            stored_source: StoredSource = shard.sources[local_index]
            if stored_source.source_record_id != row["source_record_id"] or stored_source.class_label != int(row["class_label"]):
                raise PeakEvidenceError("selected source", "shard source does not match projection")
            cells = shard.cells[local_index * 12 : (local_index + 1) * 12]
            by_id = {cell.perturbation_id: cell for cell in cells}
            if any(by_id[name].status != "complete" for name in PERTURBATION_ORDER):
                raise PeakEvidenceError("selected source", "must have complete P1-P5 cells")
            spectrum = Spectrum1D(
                stored_source.source_spectrum_id,
                stored_source.sample_id,
                stored_source.axis_cm1,
                stored_source.intensity,
            )
            built[rank] = _source_input(
                class_label=int(row["class_label"]),
                selection_rank=rank,
                source_record_id=str(row["source_record_id"]),
                spectrum=spectrum,
                sweep=sweep,
                stored_cells=by_id,
            )
    sources = tuple(built[int(row["selection_rank"])] for row in projection_rows)
    return PeakEvidenceInputs(
        synthetic_fixture=False,
        cohort_id="rruff_core10k_p5_complete_class_balanced_2250",
        cohort_projection=projection,
        sources=sources,
        cohort_identity={
            "class_count": len(sources),
            "projection_sha256": projection_sha,
            "source_record_ids_sha256": hashlib.sha256(("\n".join(source.source_record_id for source in sources) + "\n").encode()).hexdigest(),
        },
    )


# Artifact orchestration is implemented below in the same module so the public
# surface is available from the first GREEN state.


def _systems_from_config(
    config: PeakEvidenceConfig, project_root: Path
) -> tuple[Phase3System, ...]:
    catalog = load_classical_catalog(
        project_root / str(config.authorities["classical_catalog"]["path"])
    )
    by_id = {system.system_id: system for system in catalog.systems}
    systems: list[Phase3System] = []
    for system_id in config.promoted_system_ids:
        system = by_id.get(system_id)
        if system is None or system.task_line is not TaskLine.PEAK_DETECTION:
            raise PeakEvidenceError("catalog", f"invalid promoted system {system_id}")
        if system.family_id not in {"find_peaks", "find_peaks_cwt"}:
            raise PeakEvidenceError("catalog", f"unsupported promoted family {system.family_id}")
        systems.append(system)
    return tuple(systems)


def _detector_document(result: PeakDetectionRunResult) -> Mapping[str, object]:
    return {
        "status": result.status.value,
        "peak_count": len(result.peaks),
        "peaks_sha256": result.peaks_sha256,
        "warning_count": len(result.warnings),
        "warnings": tuple(
            {"category": warning.category, "message": warning.message}
            for warning in result.warnings
        ),
        "error_code": result.error_code,
        "error_message": result.error_message,
    }


def _closed_endpoint_rows(
    perturbation_id: str, state: str, reason_code: str
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "endpoint_id": endpoint_id,
            "numerator": 0,
            "denominator": 0,
            "value": None,
            "state": state,
            "reason_code": reason_code,
        }
        for endpoint_id in ENDPOINTS_BY_PERTURBATION[perturbation_id]
    )


def _event_count(condition: PeakEvidenceCondition) -> int:
    for key in (
        "inserted_component_count",
        "deleted_component_count",
        "affected_component_count",
        "replacement_component_count",
    ):
        if key in condition.truth:
            return int(condition.truth[key])
    return 0


def _evaluate_system_source(
    system: Phase3System, source: PeakEvidenceSource, tolerances: Sequence[float]
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    baseline = run_peak_detection_system(system, source.spectrum)
    receipt_conditions: list[Mapping[str, object]] = []
    condition_rows: list[Mapping[str, object]] = []
    class_values: dict[str, list[Mapping[str, object]]] = {
        endpoint_id: [] for endpoint_id in SYNTHETIC_ENDPOINT_ORDER
    }
    for intervention in source.interventions:
        for condition in intervention.conditions:
            candidate = run_peak_detection_system(system, condition.spectrum)
            by_tolerance: dict[str, Mapping[str, Mapping[str, object]]] = {}
            for tolerance in tolerances:
                tolerance_key = format(float(tolerance), "g")
                if baseline.status not in SUCCESS:
                    rows = _closed_endpoint_rows(
                        intervention.perturbation_id,
                        "not_evaluable_detector_failure",
                        f"alpha_zero_detector:{baseline.status.value}:{baseline.error_code or 'unknown'}",
                    )
                elif candidate.status not in SUCCESS:
                    rows = _closed_endpoint_rows(
                        intervention.perturbation_id,
                        "not_evaluable_detector_failure",
                        f"positive_detector:{candidate.status.value}:{candidate.error_code or 'unknown'}",
                    )
                else:
                    rows = evaluate_mechanism_condition(
                        perturbation_id=intervention.perturbation_id,
                        baseline_peaks=baseline.peaks,
                        candidate_peaks=candidate.peaks,
                        truth=condition.truth,
                        tolerance_cm1=float(tolerance),
                    )
                by_tolerance[tolerance_key] = {str(row["endpoint_id"]): row for row in rows}
            condition_row = {
                "system_id": system.system_id,
                "family_id": system.family_id,
                "method_id": system.method_id,
                "class_label": source.class_label,
                "selection_rank": source.selection_rank,
                "source_record_id": source.source_record_id,
                "perturbation_id": intervention.perturbation_id,
                "alpha": condition.alpha,
                "alpha_float64_le_hex": condition.alpha_float64_le_hex,
                "generator_event_count": _event_count(condition),
                "truth_digest": condition.truth_digest,
                "baseline_detector_status": baseline.status.value,
                "baseline_peak_count": len(baseline.peaks),
                "baseline_peaks_sha256": baseline.peaks_sha256,
                "candidate_detector_status": candidate.status.value,
                "candidate_peak_count": len(candidate.peaks),
                "candidate_peaks_sha256": candidate.peaks_sha256,
                "endpoints_by_tolerance_cm1": by_tolerance,
            }
            condition_rows.append(condition_row)
            for endpoint_id in ENDPOINTS_BY_PERTURBATION[intervention.perturbation_id]:
                class_values[endpoint_id].append(
                    {
                        "alpha": condition.alpha,
                        "alpha_float64_le_hex": condition.alpha_float64_le_hex,
                        "by_tolerance_cm1": {
                            key: by_tolerance[key][endpoint_id]
                            for key in (format(float(value), "g") for value in tolerances)
                        },
                    }
                )
            receipt_conditions.append(
                {
                    "perturbation_id": intervention.perturbation_id,
                    "alpha": condition.alpha,
                    "alpha_float64_le_hex": condition.alpha_float64_le_hex,
                    "output_spectrum_id": condition.output_spectrum_id,
                    "output_intensity_sha256": condition.output_intensity_sha256,
                    "truth_digest": condition.truth_digest,
                    **_detector_document(candidate),
                }
            )
    class_rows = tuple(
        {
            "system_id": system.system_id,
            "family_id": system.family_id,
            "method_id": system.method_id,
            "class_label": source.class_label,
            "selection_rank": source.selection_rank,
            "source_record_id": source.source_record_id,
            "endpoint_id": endpoint_id,
            "responses": tuple(class_values[endpoint_id]),
        }
        for endpoint_id in SYNTHETIC_ENDPOINT_ORDER
    )
    receipt = {
        "system_id": system.system_id,
        "family_id": system.family_id,
        "method_id": system.method_id,
        "class_label": source.class_label,
        "selection_rank": source.selection_rank,
        "source_record_id": source.source_record_id,
        "alpha_zero": _detector_document(baseline),
        "conditions": tuple(receipt_conditions),
        "detector_call_count": 1 + len(receipt_conditions),
    }
    return receipt, tuple(condition_rows), class_rows


def _initialize_worker(inputs: PeakEvidenceInputs, systems: Sequence[Phase3System]) -> None:
    global _WORKER_INPUTS, _WORKER_SYSTEMS
    _WORKER_INPUTS = inputs
    _WORKER_SYSTEMS = tuple(systems)


def _worker_task(task: tuple[int, int, tuple[float, ...]]):
    system_index, source_index, tolerances = task
    if _WORKER_INPUTS is None:
        raise PeakEvidenceError("worker", "inputs are not initialized")
    with threadpool_limits(limits=1):
        return _evaluate_system_source(
            _WORKER_SYSTEMS[system_index],
            _WORKER_INPUTS.sources[source_index],
            tolerances,
        )


def _bounded_ordered_map(
    executor: ProcessPoolExecutor,
    tasks: Iterable[tuple[int, int, tuple[float, ...]]],
    *,
    window: int,
):
    iterator = iter(tasks)
    pending = []
    for _ in range(window):
        try:
            pending.append(executor.submit(_worker_task, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.pop(0)
        yield future.result()
        try:
            pending.append(executor.submit(_worker_task, next(iterator)))
        except StopIteration:
            pass


def _truth_rows(inputs: PeakEvidenceInputs) -> Iterable[Mapping[str, object]]:
    for source in inputs.sources:
        for intervention in source.interventions:
            yield {
                "class_label": source.class_label,
                "selection_rank": source.selection_rank,
                "source_record_id": source.source_record_id,
                "source_spectrum_id": source.spectrum.spectrum_id,
                "source_axis_sha256": _array_sha(source.spectrum.axis_cm1),
                "source_intensity_sha256": _array_sha(source.spectrum.intensity),
                "perturbation_id": intervention.perturbation_id,
                "state_digest": intervention.state_digest,
                "conditions": tuple(
                    {
                        "alpha": condition.alpha,
                        "alpha_float64_le_hex": condition.alpha_float64_le_hex,
                        "output_spectrum_id": condition.output_spectrum_id,
                        "output_intensity_sha256": condition.output_intensity_sha256,
                        "truth": condition.truth,
                        "truth_digest": condition.truth_digest,
                    }
                    for condition in intervention.conditions
                ),
            }


def _bootstrap_rows(
    *,
    class_rows: Sequence[Mapping[str, object]],
    systems: Sequence[Phase3System],
    config: PeakEvidenceConfig,
) -> list[Mapping[str, object]]:
    rows_by_key = {
        (str(row["system_id"]), str(row["endpoint_id"])): []
        for row in class_rows
    }
    for row in class_rows:
        rows_by_key[(str(row["system_id"]), str(row["endpoint_id"]))].append(row)
    source_count = len(class_rows) // (len(systems) * len(SYNTHETIC_ENDPOINT_ORDER))
    rng = np.random.Generator(np.random.PCG64(int(config.bootstrap["seed"])))
    sample_indexes = rng.integers(
        0, source_count, size=(int(config.bootstrap["resamples"]), source_count)
    )
    output: list[Mapping[str, object]] = []
    for system in systems:
        for endpoint_id in SYNTHETIC_ENDPOINT_ORDER:
            endpoint_rows = rows_by_key[(system.system_id, endpoint_id)]
            if len(endpoint_rows) != source_count:
                raise PeakEvidenceError("class rows", "incomplete system endpoint grid")
            curve_by_tolerance: dict[str, object] = {}
            summary_for_two: Mapping[str, object] | None = None
            for tolerance in config.tolerances_cm1:
                key = format(tolerance, "g")
                matrix = np.full((source_count, len(config.positive_alpha_grid)), np.nan, dtype="<f8")
                fatal = False
                for source_index, class_row in enumerate(endpoint_rows):
                    responses = tuple(class_row["responses"])
                    if len(responses) != len(config.positive_alpha_grid):
                        raise PeakEvidenceError("class row responses", "alpha grid mismatch")
                    for alpha_index, response in enumerate(responses):
                        cell = response["by_tolerance_cm1"][key]
                        state = str(cell["state"])
                        if state == "complete_numeric":
                            matrix[source_index, alpha_index] = float(cell["value"])
                        elif state not in {"complete_empty_risk_set", "not_applicable_zero_synthetic_events"}:
                            fatal = True
                alpha_estimates: list[float | None] = []
                contributing: list[int] = []
                for alpha_index in range(matrix.shape[1]):
                    finite = np.isfinite(matrix[:, alpha_index])
                    contributing.append(int(np.sum(finite)))
                    alpha_estimates.append(None if not np.any(finite) else float(np.mean(matrix[finite, alpha_index])))
                if fatal or any(value is None for value in alpha_estimates):
                    summary = {
                        "estimate": None, "interval_lower": None, "interval_upper": None,
                        "state": "not_evaluable_incomplete_system_grid" if fatal else "not_evaluable_no_contributing_classes",
                    }
                else:
                    bootstrap_values = np.empty(sample_indexes.shape[0], dtype="<f8")
                    for replicate, indexes in enumerate(sample_indexes):
                        alpha_values = []
                        for alpha_index in range(matrix.shape[1]):
                            selected = matrix[indexes, alpha_index]
                            finite = np.isfinite(selected)
                            alpha_values.append(float(np.mean(selected[finite])) if np.any(finite) else float(alpha_estimates[alpha_index]))
                        bootstrap_values[replicate] = float(np.mean(alpha_values))
                    summary = {
                        "estimate": float(np.mean(np.asarray(alpha_estimates, dtype="<f8"))),
                        "interval_lower": float(np.quantile(bootstrap_values, 0.025)),
                        "interval_upper": float(np.quantile(bootstrap_values, 0.975)),
                        "state": "complete",
                    }
                curve_by_tolerance[key] = {
                    "alpha_estimates": tuple(alpha_estimates),
                    "contributing_class_counts": tuple(contributing),
                    **summary,
                }
                if tolerance == 2.0:
                    summary_for_two = summary
            assert summary_for_two is not None
            output.append(
                {
                    "system_id": system.system_id,
                    "family_id": system.family_id,
                    "method_id": system.method_id,
                    "endpoint_id": endpoint_id,
                    "design_class_count": source_count,
                    "resamples": int(config.bootstrap["resamples"]),
                    "seed": int(config.bootstrap["seed"]),
                    "confidence_level": float(config.bootstrap["confidence_level"]),
                    "curve_by_tolerance_cm1": curve_by_tolerance,
                    "estimate": summary_for_two["estimate"],
                    "interval_lower": summary_for_two["interval_lower"],
                    "interval_upper": summary_for_two["interval_upper"],
                    "state": summary_for_two["state"],
                }
            )
    return output


def _csv_bytes(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> bytes:
    pieces: list[str] = []

    class Writer:
        def write(self, value: str) -> int:
            pieces.append(value)
            return len(value)

    writer = csv.DictWriter(Writer(), fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        values = {}
        for name in fieldnames:
            value = row.get(name)
            values[name] = "" if value is None else format(value, ".17g") if isinstance(value, float) else value
        writer.writerow(values)
    return "".join(pieces).encode("utf-8")


def _method_rows(
    *,
    systems: Sequence[Phase3System],
    blocked_systems: Sequence[Mapping[str, str]],
    status_rows: Sequence[Mapping[str, object]],
    bootstrap_rows: Sequence[Mapping[str, object]],
    config: PeakEvidenceConfig,
) -> list[dict[str, object]]:
    status_by_id = {str(row["system_id"]): row for row in status_rows}
    bootstrap_by_key = {(str(row["system_id"]), str(row["endpoint_id"])): row for row in bootstrap_rows}
    rows: list[dict[str, object]] = []
    for system in systems:
        for endpoint in config.endpoint_manifest:
            endpoint_id = str(endpoint["endpoint_id"])
            if endpoint_id == "availability":
                status = status_by_id[system.system_id]
                state, reason, source_path = status["status"], status["reason_code"], "system_status.jsonl"
                estimate = lower = upper = None
            elif endpoint_id == "direct_gt":
                state = reason = "not_evaluated_no_real_peak_assignments"
                source_path = "system_status.jsonl"; estimate = lower = upper = None
            elif endpoint_id == "downstream":
                state = reason = "not_evaluated_no_real_peak_downstream_target"
                source_path = "system_status.jsonl"; estimate = lower = upper = None
            else:
                bootstrap = bootstrap_by_key[(system.system_id, endpoint_id)]
                state = str(bootstrap["state"]); reason = "" if state == "complete" else state
                estimate, lower, upper = bootstrap["estimate"], bootstrap["interval_lower"], bootstrap["interval_upper"]
                source_path = "synthetic_class_rows.jsonl"
            rows.append({
                "evidence_id": f"{system.system_id}:{endpoint_id}", "task_line": "peak_detection",
                "system_id": system.system_id, "family_id": system.family_id,
                "endpoint_id": endpoint_id, "protocol_id": endpoint["protocol_id"],
                "cohort_id": "rruff_core10k_p5_complete_class_balanced_2250",
                "evidence_component": endpoint["evidence_component"], "metric_output_id": endpoint["metric_output_id"],
                "estimate": estimate, "interval_lower": lower, "interval_upper": upper,
                "preferred_direction": endpoint["preferred_direction"], "state": state, "reason_code": reason,
                "phase5_power_state": config.phase5_power_state, "source_path": source_path, "source_sha256": None,
            })
    for blocked in blocked_systems:
        rows.append({
            "evidence_id": f"{blocked['system_id']}:availability", "task_line": "peak_detection",
            "system_id": blocked["system_id"], "family_id": blocked["family_id"],
            "endpoint_id": "availability", "protocol_id": "availability",
            "cohort_id": "rruff_core10k_p5_complete_class_balanced_2250",
            "evidence_component": "availability", "metric_output_id": "availability",
            "estimate": None, "interval_lower": None, "interval_upper": None, "preferred_direction": "not_applicable",
            "state": blocked["reason_code"], "reason_code": blocked["reason_code"],
            "phase5_power_state": config.phase5_power_state, "source_path": "system_status.jsonl", "source_sha256": None,
        })
    return rows


def _family_rows(method_rows: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in method_rows:
        grouped.setdefault((str(row["family_id"]), str(row["endpoint_id"])), []).append(row)
    output = []
    for (family_id, endpoint_id), items in sorted(grouped.items()):
        values = [float(row["estimate"]) for row in items if row["estimate"] is not None]
        output.append({
            "family_id": family_id, "endpoint_id": endpoint_id,
            "registered_system_count": len(items), "complete_system_count": len(values),
            "median_estimate": None if not values else float(np.median(values)),
            "min_estimate": None if not values else float(np.min(values)),
            "max_estimate": None if not values else float(np.max(values)),
        })
    return output


def _code_authority(project_root: Path) -> Mapping[str, Mapping[str, object]]:
    paths = (
        "experiments/phase6/configs/peak_evidence_v1.json",
        "rpe/io/perturbed_store.py",
        "rpe/methods/catalog.py",
        "rpe/methods/classical/peaks.py",
        "rpe/metrics/peak.py",
        "rpe/perturb/peak_family.py",
        "rpe/runner/phase6_peak_evidence.py",
        "rpe/runner/phase6_peak_evidence_verifier.py",
        "tools/run_phase6_peak_evidence.py",
    )
    output = {}
    for relative in paths:
        path = project_root / relative
        if not path.is_file():
            raise PeakEvidenceError("code authority", f"missing {relative}")
        output[relative] = {"byte_count": path.stat().st_size, "sha256": _sha_file(path)}
    return output


def _run_id(config: PeakEvidenceConfig, inputs: PeakEvidenceInputs, systems: Sequence[Phase3System], project_root: Path) -> str:
    identity = {
        "config_sha256": config.sha256, "cohort_identity": inputs.cohort_identity,
        "system_ids": tuple(system.system_id for system in systems),
        "synthetic_fixture": inputs.synthetic_fixture, "code_authority": _code_authority(project_root),
    }
    return str(config.artifact_contract["run_prefix"]) + hashlib.sha256(RUN_DOMAIN + _canonical(identity)).hexdigest()


def _sha256sums(path: Path, names: Sequence[str]) -> bytes:
    return "".join(f"{_sha_file(path / name)}  {name}\n" for name in sorted(names)).encode("utf-8")


def _validate_formal_scope(
    inputs: PeakEvidenceInputs,
    systems: Sequence[Phase3System],
    config: PeakEvidenceConfig,
) -> None:
    if inputs.synthetic_fixture:
        return
    expected = config.expected
    if tuple(system.system_id for system in systems) != config.promoted_system_ids:
        raise PeakEvidenceError("formal system IDs", "must match frozen promoted order")
    if inputs.source_count != int(config.cohort["class_count"]):
        raise PeakEvidenceError("formal source count", "must equal 2250")
    if len({source.class_label for source in inputs.sources}) != inputs.source_count:
        raise PeakEvidenceError("formal source classes", "must be one source per class")
    if hashlib.sha256(inputs.cohort_projection).hexdigest() != str(config.cohort["projection_sha256"]):
        raise PeakEvidenceError("formal cohort projection", "SHA256 mismatch")
    if any(
        tuple(item.perturbation_id for item in source.interventions) != PERTURBATION_ORDER
        or any(len(item.conditions) != len(config.positive_alpha_grid) for item in source.interventions)
        for source in inputs.sources
    ):
        raise PeakEvidenceError("formal intervention grid", "must contain P1-P5 and eight positive alphas")
    static_counts = {
        "promoted_system_count": len(systems),
        "blocked_system_count": len(config.blocked_systems),
        "intervention_truth_row_count": inputs.source_count * len(PERTURBATION_ORDER),
        "detector_receipt_row_count": len(systems) * inputs.source_count,
        "detector_call_count": len(systems) * inputs.source_count * (1 + len(PERTURBATION_ORDER) * len(config.positive_alpha_grid)),
        "positive_detector_call_count": len(systems) * inputs.source_count * len(PERTURBATION_ORDER) * len(config.positive_alpha_grid),
        "synthetic_condition_row_count": len(systems) * inputs.source_count * len(PERTURBATION_ORDER) * len(config.positive_alpha_grid),
        "synthetic_class_row_count": len(systems) * inputs.source_count * len(SYNTHETIC_ENDPOINT_ORDER),
        "bootstrap_row_count": len(systems) * len(SYNTHETIC_ENDPOINT_ORDER),
        "promoted_method_evidence_row_count": len(systems) * len(config.endpoint_manifest),
        "method_evidence_row_count": len(systems) * len(config.endpoint_manifest) + len(config.blocked_systems),
        "system_status_row_count": len(systems) + len(config.blocked_systems),
        "family_projection_row_count": 27,
    }
    for key, observed in static_counts.items():
        if observed != int(expected[key]):
            raise PeakEvidenceError(f"formal expected.{key}", f"expected {expected[key]}, observed {observed}")


def _validate_formal_observed(
    *,
    inputs: PeakEvidenceInputs,
    config: PeakEvidenceConfig,
    status_rows: Sequence[Mapping[str, object]],
    truth_row_count: int,
    detector_receipt_row_count: int,
    detector_call_count: int,
    condition_row_count: int,
    class_rows: Sequence[Mapping[str, object]],
    bootstrap_rows: Sequence[Mapping[str, object]],
    method_rows: Sequence[Mapping[str, object]],
    family_rows: Sequence[Mapping[str, object]],
) -> None:
    if inputs.synthetic_fixture:
        return
    expected = config.expected
    observed = {
        "system_status_row_count": len(status_rows),
        "intervention_truth_row_count": truth_row_count,
        "detector_receipt_row_count": detector_receipt_row_count,
        "detector_call_count": detector_call_count,
        "synthetic_condition_row_count": condition_row_count,
        "synthetic_class_row_count": len(class_rows),
        "bootstrap_row_count": len(bootstrap_rows),
        "method_evidence_row_count": len(method_rows),
        "family_projection_row_count": len(family_rows),
    }
    for key, value in observed.items():
        if value != int(expected[key]):
            raise PeakEvidenceError(f"formal expected.{key}", f"expected {expected[key]}, observed {value}")


def _build_phase6_peak_evidence_from_inputs_unchecked(
    inputs: PeakEvidenceInputs,
    systems: Sequence[Phase3System],
    output_root: Path,
    *,
    config: PeakEvidenceConfig,
    worker_count: int,
    project_root: Path = ROOT,
) -> PeakEvidenceSummary:
    if worker_count <= 0:
        raise PeakEvidenceError("worker_count", "must be positive")
    systems = tuple(systems)
    if not systems or any(system.task_line is not TaskLine.PEAK_DETECTION for system in systems):
        raise PeakEvidenceError("systems", "must contain peak-detection systems")
    if len({system.system_id for system in systems}) != len(systems):
        raise PeakEvidenceError("systems", "must be unique")
    run_id = _run_id(config, inputs, systems, project_root)
    path = Path(output_root) / run_id
    if path.exists():
        raise PeakEvidenceError("output", "run path already exists")
    path.mkdir(parents=True)
    (path / "config.json").write_bytes(_canonical(config.document))
    truth_row_count = inputs.source_count * len(PERTURBATION_ORDER)
    (path / "intervention_truth.jsonl").write_bytes(_jsonl_bytes(_truth_rows(inputs)))
    tasks = (
        (system_index, source_index, config.tolerances_cm1)
        for system_index in range(len(systems))
        for source_index in range(inputs.source_count)
    )
    global _WORKER_INPUTS, _WORKER_SYSTEMS
    _WORKER_INPUTS, _WORKER_SYSTEMS = inputs, systems
    if worker_count == 1:
        results = map(_worker_task, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("fork"),
            initializer=_initialize_worker,
            initargs=(inputs, systems),
        )
        results = _bounded_ordered_map(executor, tasks, window=max(2, worker_count * 2))
    class_rows: list[Mapping[str, object]] = []
    status_counts = {system.system_id: {"complete": 0, "complete_with_warning": 0, "not_applicable": 0, "failed_runtime": 0} for system in systems}
    detector_call_count = 0
    condition_row_count = 0
    try:
        with (path / "detector_receipts.jsonl").open("wb") as receipt_stream, (path / "synthetic_condition_rows.jsonl").open("wb") as condition_stream, (path / "synthetic_class_rows.jsonl").open("wb") as class_stream:
            for receipt, condition_group, class_group in results:
                receipt_stream.write(_canonical(receipt))
                detector_call_count += int(receipt["detector_call_count"])
                for detector in (receipt["alpha_zero"], *receipt["conditions"]):
                    state = str(detector["status"])
                    status_counts[str(receipt["system_id"])][state] = status_counts[str(receipt["system_id"])].get(state, 0) + 1
                for row in condition_group:
                    condition_stream.write(_canonical(row)); condition_row_count += 1
                for row in class_group:
                    class_stream.write(_canonical(row)); class_rows.append(row)
    finally:
        if executor is not None:
            executor.shutdown()
    blocked = () if inputs.synthetic_fixture else config.blocked_systems
    status_rows: list[Mapping[str, object]] = []
    for system in systems:
        counts = status_counts[system.system_id]
        bad = counts.get("not_applicable", 0) + counts.get("failed_runtime", 0)
        status_rows.append({
            "system_id": system.system_id, "family_id": system.family_id, "method_id": system.method_id,
            "status": "complete" if bad == 0 else "complete_with_reason_closed_conditions",
            "reason_code": "" if bad == 0 else "detector_condition_failure",
            "detector_call_count": sum(counts.values()), "detector_status_counts": counts,
        })
    for row in blocked:
        status_rows.append({
            "system_id": row["system_id"], "family_id": row["family_id"], "method_id": row["family_id"],
            "status": row["reason_code"], "reason_code": row["reason_code"],
            "detector_call_count": 0, "detector_status_counts": {},
        })
    (path / "system_status.jsonl").write_bytes(_jsonl_bytes(status_rows))
    bootstrap_rows = _bootstrap_rows(class_rows=class_rows, systems=systems, config=config)
    (path / "bootstrap_results.jsonl").write_bytes(_jsonl_bytes(bootstrap_rows))
    method_rows = _method_rows(systems=systems, blocked_systems=blocked, status_rows=status_rows, bootstrap_rows=bootstrap_rows, config=config)
    method_source_hashes = {
        "system_status.jsonl": _sha_file(path / "system_status.jsonl"),
        "synthetic_class_rows.jsonl": _sha_file(path / "synthetic_class_rows.jsonl"),
    }
    for row in method_rows:
        row["source_sha256"] = method_source_hashes[str(row["source_path"])]
    method_fields = ("evidence_id", "task_line", "system_id", "family_id", "endpoint_id", "protocol_id", "cohort_id", "evidence_component", "metric_output_id", "estimate", "interval_lower", "interval_upper", "preferred_direction", "state", "reason_code", "phase5_power_state", "source_path", "source_sha256")
    (path / "method_evidence_rows.csv").write_bytes(_csv_bytes(method_rows, method_fields))
    family_rows = _family_rows(method_rows)
    family_fields = ("family_id", "endpoint_id", "registered_system_count", "complete_system_count", "median_estimate", "min_estimate", "max_estimate")
    (path / "family_projection.csv").write_bytes(_csv_bytes(family_rows, family_fields))
    _validate_formal_observed(
        inputs=inputs,
        config=config,
        status_rows=status_rows,
        truth_row_count=truth_row_count,
        detector_receipt_row_count=len(systems) * inputs.source_count,
        detector_call_count=detector_call_count,
        condition_row_count=condition_row_count,
        class_rows=class_rows,
        bootstrap_rows=bootstrap_rows,
        method_rows=method_rows,
        family_rows=family_rows,
    )
    authority_bridge = {
        "authorities": config.authorities, "code_authority": _code_authority(project_root),
        "cohort_identity": inputs.cohort_identity, "source_revision_status": config.source_revision_status,
        "promoted_system_ids_sha256": hashlib.sha256(("\n".join(system.system_id for system in systems) + "\n").encode()).hexdigest(),
    }
    (path / "authority_bridge.json").write_bytes(_canonical(authority_bridge))
    preflight = {
        "status": "passed", "cohort_identity": inputs.cohort_identity,
        "promoted_system_count": len(systems), "blocked_system_count": len(blocked),
        "condition_count_per_system_source": 41, "alpha_zero_reuse": "once_per_system_source",
        "anti_circular_truth_boundary": "mechanism_facts_only_not_internal_peak_list_gt",
    }
    (path / "preflight.json").write_bytes(_canonical(preflight))
    manifest = {
        "schema_version": "phase6-peak-evidence-artifact-v1", "run_id": run_id, "status": "complete",
        "synthetic_fixture": inputs.synthetic_fixture, "system_count": len(systems), "blocked_system_count": len(blocked),
        "source_count": inputs.source_count, "intervention_truth_row_count": truth_row_count,
        "detector_receipt_row_count": len(systems) * inputs.source_count, "detector_call_count": detector_call_count,
        "synthetic_condition_row_count": condition_row_count, "synthetic_class_row_count": len(class_rows),
        "bootstrap_row_count": len(bootstrap_rows), "method_evidence_row_count": len(method_rows),
        "family_projection_row_count": len(family_rows), "configured_payload_count": 12,
        "payload_files": tuple(config.artifact_contract["payload_files"]),
        "source_revision_status": config.source_revision_status,
    }
    (path / "manifest.json").write_bytes(_canonical(manifest))
    (path / "complete.json").write_bytes(_canonical({"run_id": run_id, "status": "complete"}))
    payload_names = tuple(config.artifact_contract["payload_files"]) + ("complete.json",)
    (path / "SHA256SUMS").write_bytes(_sha256sums(path, payload_names))
    return PeakEvidenceSummary(
        path=path, run_id=run_id, status="complete", system_count=len(systems), source_count=inputs.source_count,
        intervention_truth_row_count=truth_row_count, detector_receipt_row_count=len(systems) * inputs.source_count,
        detector_call_count=detector_call_count, condition_row_count=condition_row_count, class_row_count=len(class_rows),
        bootstrap_row_count=len(bootstrap_rows), method_evidence_row_count=len(method_rows),
    )


def build_phase6_peak_evidence_from_inputs(
    inputs: PeakEvidenceInputs,
    systems: Sequence[Phase3System],
    output_root: Path,
    *,
    config: PeakEvidenceConfig,
    worker_count: int,
    project_root: Path = ROOT,
) -> PeakEvidenceSummary:
    systems = tuple(systems)
    run_id = _run_id(config, inputs, systems, project_root)
    path = Path(output_root) / run_id
    if path.exists():
        raise PeakEvidenceError("output", "run path already exists")
    try:
        _validate_formal_scope(inputs, systems, config)
        return _build_phase6_peak_evidence_from_inputs_unchecked(
            inputs, systems, output_root, config=config, worker_count=worker_count, project_root=project_root
        )
    except Exception as error:
        path.mkdir(parents=True, exist_ok=True)
        complete = path / "complete.json"
        if complete.exists():
            complete.unlink()
        reason = error.reason if isinstance(error, PeakEvidenceError) else str(error) or type(error).__name__
        failed = {
            "error_type": type(error).__name__,
            "reason": reason,
            "run_id": run_id,
            "status": "failed",
        }
        (path / "failed.json").write_bytes(_canonical(failed))
        raise


def build_phase6_peak_evidence(
    output_root: Path, *, worker_count: int = 32, project_root: Path = ROOT
) -> PeakEvidenceSummary:
    config = load_phase6_peak_evidence_config(project_root / "experiments/phase6/configs/peak_evidence_v1.json")
    inputs = _load_real_inputs(config, project_root)
    systems = _systems_from_config(config, project_root)
    return build_phase6_peak_evidence_from_inputs(
        inputs, systems, Path(output_root), config=config, worker_count=worker_count, project_root=project_root
    )


__all__ = [
    "PeakEvidenceConfig", "PeakEvidenceCondition", "PeakEvidenceError",
    "PeakEvidenceInputs", "PeakEvidenceIntervention", "PeakEvidenceSource",
    "PeakEvidenceSummary", "build_phase6_peak_evidence",
    "build_phase6_peak_evidence_from_inputs", "derive_peak_evidence_cohort_projection",
    "evaluate_mechanism_condition", "load_phase6_peak_evidence_config",
    "make_synthetic_peak_evidence_inputs",
]
