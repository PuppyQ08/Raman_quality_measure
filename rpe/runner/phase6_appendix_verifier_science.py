"""Independent record-science and deterministic-byte primitives for Step 7 verification.

This module deliberately owns its transforms, ranking rules, duplicate scan and
serialization.  Dataset adapters may use it without depending on any Step-7
production implementation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy import interpolate, stats

from rpe.evaluation import SingleSpectrumInput, Spectrum1D, SpectrumPairInput, evaluate_metric
from rpe.metrics import (
    ISLikeStructureToNoiseMetric, MAEMetric, MSEMetric, NMSEMetric,
    PearsonRMetric, RMSEMetric, SAMMetric, Wasserstein1Metric,
)
from rpe.metrics.peak import _match_peaks
from rpe.methods.catalog import Phase3System
from rpe.methods.classical.peaks import PeakRunStatus, run_peak_detection_system
from rpe.perturb import PerturbationSweepConfig
from rpe.runner.phase1_config import Phase1CoreConfig
from rpe.runner.phase1_perturbations import P10MemoryAdmission, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source
from rpe.runner.phase1_types import CellStatus


v_METRIC_IDS = (
    "mse", "rmse", "mae", "sam", "pearson_r", "nmse",
    "wasserstein_1_cm1", "is_like_structure_to_noise", "precision",
    "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio",
)
v_SCALAR_METRIC_IDS = v_METRIC_IDS[:8]
v_DOMAIN = b"rpe-step7-exact-spectrum-v1"


@dataclass(frozen=True)
class VerifierDatasetSensitivityResult:
    """Immutable verifier-side record result suitable for dataset adapters."""

    record_id: str
    condition_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    resampling_ids: tuple[str, ...]
    normalization_ids: tuple[str, ...]
    tolerance_ids: tuple[str, ...]
    values_by_condition: Mapping[str, Mapping[str, float]]
    closure_reasons: Mapping[str, str]
    resampling_harms: np.ndarray
    normalization_harms: np.ndarray
    peak_tolerance_harms: np.ndarray
    identity: Mapping[str, object]


def pack_verifier_record_sensitivity_result(value: VerifierDatasetSensitivityResult) -> tuple[object, ...]:
    """Convert an immutable verifier result to process-queue-safe primitives."""
    return (
        value.record_id, tuple(value.condition_ids), tuple(value.metric_ids), tuple(value.resampling_ids), tuple(value.normalization_ids), tuple(value.tolerance_ids),
        {str(key): dict(item) for key, item in value.values_by_condition.items()},
        dict(value.closure_reasons), np.ascontiguousarray(value.resampling_harms, dtype="<f8"),
        np.ascontiguousarray(value.normalization_harms, dtype="<f8"), np.ascontiguousarray(value.peak_tolerance_harms, dtype="<f8"), dict(value.identity),
    )


def unpack_verifier_record_sensitivity_result(payload: tuple[object, ...]) -> VerifierDatasetSensitivityResult:
    if len(payload) != 12:
        raise ValueError("verifier record transport payload shape drift")
    record_id, condition_ids, metric_ids, resampling_ids, normalization_ids, tolerance_ids, values, reasons, resampling, normalization, tolerance, identity = payload
    if not all(isinstance(item, Mapping) for item in (values, reasons, identity)):
        raise ValueError("verifier record transport mapping drift")
    return VerifierDatasetSensitivityResult(
        str(record_id), tuple(condition_ids), tuple(metric_ids), tuple(resampling_ids), tuple(normalization_ids), tuple(tolerance_ids),
        MappingProxyType({str(key): MappingProxyType(dict(item)) for key, item in values.items()}),
        MappingProxyType(dict(sorted((str(key), str(value)) for key, value in reasons.items()))),
        v_freeze_array(np.asarray(resampling, dtype="<f8")), v_freeze_array(np.asarray(normalization, dtype="<f8")), v_freeze_array(np.asarray(tolerance, dtype="<f8")),
        MappingProxyType(dict(identity)),
    )


def v_array(name: str, value: object) -> np.ndarray:
    result = np.asarray(value, dtype="<f8")
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name}: expected finite one-dimensional vector")
    return np.ascontiguousarray(result, dtype="<f8")


def v_target_axis(start_cm1: float, spacing_cm1: float, point_count: int) -> np.ndarray:
    if not math.isfinite(start_cm1) or not math.isfinite(spacing_cm1) or spacing_cm1 <= 0 or point_count < 2:
        raise ValueError("target axis: invalid start, spacing, or count")
    return np.ascontiguousarray(np.float64(start_cm1) + np.float64(spacing_cm1) * np.arange(point_count, dtype="<f8"), dtype="<f8")


def v_resample(axis: object, intensity: object, target: object, interpolator: str, *, max_in_range_native_gap_cm1: float | None = None) -> np.ndarray:
    x, y, grid = v_array("axis", axis), v_array("intensity", intensity), v_array("target", target)
    if x.shape != y.shape or not np.all(np.diff(x) > 0) or not np.all(np.diff(grid) > 0):
        raise ValueError("resampling: source and target axes must strictly increase")
    if grid[0] < x[0] or grid[-1] > x[-1]:
        raise ValueError("resampling: extrapolation is forbidden")
    if max_in_range_native_gap_cm1 is not None:
        if not math.isfinite(max_in_range_native_gap_cm1) or max_in_range_native_gap_cm1 <= 0.0:
            raise ValueError("resampling: native-gap maximum must be finite and positive")
        native_in_range = x[(x >= grid[0]) & (x <= grid[-1])]
        if native_in_range.size < 2 or float(np.max(np.diff(native_in_range))) > max_in_range_native_gap_cm1:
            raise ValueError("resampling: native gap exceeds frozen maximum")
    if interpolator == "linear":
        result = np.interp(grid, x, y)
    elif interpolator == "cubic":
        result = interpolate.CubicSpline(x, y, bc_type="not-a-knot", extrapolate=False)(grid)
    elif interpolator == "pchip":
        result = interpolate.PchipInterpolator(x, y, extrapolate=False)(grid)
    else:
        raise ValueError("resampling: unsupported interpolator")
    result = np.ascontiguousarray(result, dtype="<f8")
    if result.shape != grid.shape or not np.isfinite(result).all():
        raise ValueError("resampling: output is nonfinite or shape drifted")
    return result


def v_normalize(axis: object, intensity: object, method: str) -> np.ndarray:
    x, y = v_array("axis", axis), v_array("intensity", intensity)
    if x.shape != y.shape or not np.all(np.diff(x) > 0):
        raise ValueError("normalization: invalid axis")
    if method == "none":
        output = y.copy()
    elif method == "maximum":
        denominator = float(np.max(y))
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("normalization: maximum must be finite and positive")
        output = y / denominator
    elif method == "area":
        denominator = float(np.trapezoid(np.abs(y), x))
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("normalization: physical L1 area must be finite and positive")
        output = y / denominator
    elif method == "snv":
        denominator = float(np.std(y, ddof=0))
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("normalization: SNV standard deviation must be finite and positive")
        output = (y - float(np.mean(y))) / denominator
    else:
        raise ValueError("normalization: unsupported method")
    if not np.isfinite(output).all():
        raise ValueError("normalization: nonfinite output")
    return np.ascontiguousarray(output, dtype="<f8")


def v_invariance_holds(none_value: float, transformed_value: float) -> bool:
    return bool(math.isfinite(none_value) and math.isfinite(transformed_value) and abs(transformed_value - none_value) <= 1e-12 + 1e-10 * abs(none_value))


def v_tie_count(values: np.ndarray) -> int:
    _, counts = np.unique(values, return_counts=True)
    return int(np.sum(counts[counts > 1] - 1))


def v_rank_stability(reference: Mapping[str, object], candidate: Mapping[str, object], *, statistic: str, identities: Sequence[str]) -> dict[str, object]:
    ids = tuple(identities)
    base: dict[str, object] = {"statistic": statistic, "tau_b": None, "reference_tie_count": None, "candidate_tie_count": None, "max_abs_rank_displacement": None, "stable": False}
    if statistic not in {"ag", "acc_cross"}:
        raise ValueError("rank stability: statistic must be ag or acc_cross")
    if set(reference) != set(ids) or set(candidate) != set(ids):
        return {**base, "state": "not_evaluable", "reason_code": "not_evaluable_incomplete_metric_set"}
    ref, cur = np.asarray([reference[key] for key in ids], dtype="<f8"), np.asarray([candidate[key] for key in ids], dtype="<f8")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all():
        return {**base, "state": "not_evaluable", "reason_code": "not_evaluable_incomplete_metric_set"}
    direction = 1.0 if statistic == "ag" else -1.0
    ref_rank, cur_rank = stats.rankdata(direction * ref, method="average"), stats.rankdata(direction * cur, method="average")
    ref_ties, cur_ties = v_tie_count(ref), v_tie_count(cur)
    if ref_ties == len(ids) - 1 or cur_ties == len(ids) - 1:
        return {**base, "reference_tie_count": ref_ties, "candidate_tie_count": cur_ties, "state": "not_evaluable", "reason_code": "not_evaluable_all_tied_ranking"}
    tau = float(stats.kendalltau(ref_rank, cur_rank, variant="b", nan_policy="propagate").statistic)
    stable = tau > 0.9
    return {**base, "tau_b": tau, "reference_tie_count": ref_ties, "candidate_tie_count": cur_ties, "max_abs_rank_displacement": float(np.max(np.abs(ref_rank - cur_rank))), "stable": stable, "state": "complete_numeric", "reason_code": "" if stable else "complete_ranking_instability_or_reversal"}


def v_scalar_metrics(axis: object, reference: object, candidate: object) -> dict[str, float]:
    x, ref, cur = v_array("axis", axis), v_array("reference", reference), v_array("candidate", candidate)
    if x.shape != ref.shape or ref.shape != cur.shape or not np.all(np.diff(x) > 0):
        raise ValueError("metric core: spectra must share a strictly increasing axis")
    reference_spectrum, candidate_spectrum = Spectrum1D("reference", "reference", x, ref), Spectrum1D("candidate", "candidate", x, cur)
    definitions = (
        ("mse", MSEMetric(), SpectrumPairInput(reference_spectrum, candidate_spectrum)),
        ("rmse", RMSEMetric(), SpectrumPairInput(reference_spectrum, candidate_spectrum)),
        ("mae", MAEMetric(), SpectrumPairInput(reference_spectrum, candidate_spectrum)),
        ("sam", SAMMetric(), SpectrumPairInput(reference_spectrum, candidate_spectrum)),
        ("pearson_r", PearsonRMetric(), SpectrumPairInput(reference_spectrum, candidate_spectrum)),
        ("nmse", NMSEMetric(), SpectrumPairInput(reference_spectrum, candidate_spectrum)),
        ("wasserstein_1_cm1", Wasserstein1Metric(), SpectrumPairInput(reference_spectrum, candidate_spectrum)),
        ("is_like_structure_to_noise", ISLikeStructureToNoiseMetric(), SingleSpectrumInput(candidate_spectrum)),
    )
    values: dict[str, float] = {}
    for name, metric, request in definitions:
        rows = [item for item in evaluate_metric(metric, request).outputs if item.output_id == name]
        if len(rows) != 1 or not math.isfinite(float(rows[0].value)):
            raise ValueError(f"metric core: {name} did not return one finite scalar")
        values[name] = float(rows[0].value)
    return values


def v_peak_metrics(reference_peaks: Sequence[object], candidate_peaks: Sequence[object], *, tolerance_cm1: float) -> dict[str, float]:
    """Compute the five frozen peak scores from already detected peak lists."""
    if not math.isfinite(tolerance_cm1) or tolerance_cm1 < 0:
        raise ValueError("peak metrics: invalid tolerance")
    reference, candidate = tuple(reference_peaks), tuple(candidate_peaks)
    matches = _match_peaks(reference, candidate, tolerance_cm1=float(tolerance_cm1))
    true_positive, reference_count, candidate_count = len(matches), len(reference), len(candidate)
    precision = true_positive / candidate_count if candidate_count else 1.0
    recall = true_positive / reference_count if reference_count else 1.0
    f1 = 2.0 * true_positive / (reference_count + candidate_count) if reference_count + candidate_count else 1.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "artifact_peak_ratio": (candidate_count - true_positive) / candidate_count if candidate_count else 0.0,
        "missing_peak_ratio": (reference_count - true_positive) / reference_count if reference_count else 0.0,
    }


def v_freeze_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype="<f8")
    result.setflags(write=False)
    return result


def v_array_digest(value: object) -> str:
    array = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(struct.pack("<Q", array.size) + array.tobytes()).hexdigest()


def v_condition_id(perturbation_id: str, alpha: float) -> str:
    return f"{perturbation_id}:{struct.pack('<d', float(alpha)).hex()}"


def v_rematerialize_p8_p12_conditions(source: Phase1Source, phase1_config: Phase1CoreConfig, sweep: PerturbationSweepConfig, p10_memory_budget_bytes: int) -> tuple[tuple[str, ...], tuple[Spectrum1D, ...]]:
    """Run the frozen source-side P8--P12 sweep and return its 40 positive cells."""
    if not isinstance(source, Phase1Source):
        raise ValueError("rematerialization: source must be Phase1Source")
    if not isinstance(p10_memory_budget_bytes, int) or p10_memory_budget_bytes <= 0:
        raise ValueError("rematerialization: P10 memory budget must be positive")
    expected = (0.0, .05, .1, .2, .3, .4, .5, .65, .8)
    if tuple(float(alpha) for alpha in sweep.alpha_grid) != expected:
        raise ValueError("rematerialization: frozen alpha grid drift")
    identifiers, outputs = [], []
    admission = P10MemoryAdmission(p10_memory_budget_bytes)
    for perturbation_id in ("p08", "p09", "p10", "p11", "p12"):
        cell = run_perturbation_cell(source, perturbation_id, phase1_config, sweep, p10_admission=admission if perturbation_id == "p10" else None)
        if cell.status is not CellStatus.COMPLETE or tuple(float(row.alpha) for row in cell.records) != expected:
            raise ValueError(f"rematerialization: {perturbation_id} cell is incomplete or grid drifted")
        zero = cell.records[0].result.output
        if not np.array_equal(zero.axis_cm1, source.spectrum.axis_cm1) or not np.array_equal(zero.intensity, source.spectrum.intensity):
            raise ValueError(f"rematerialization: {perturbation_id} alpha-zero identity failure")
        for row in cell.records[1:]:
            identifiers.append(v_condition_id(perturbation_id, row.alpha))
            outputs.append(row.result.output)
    if len(identifiers) != 40 or len(set(identifiers)) != 40:
        raise ValueError("rematerialization: expected 40 distinct positive conditions")
    return tuple(identifiers), tuple(outputs)


def v_detect_peaks(system: Phase3System, spectrum: Spectrum1D) -> tuple[object, ...]:
    outcome = run_peak_detection_system(system, spectrum)
    if getattr(outcome, "status", None) not in (PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING, "complete", "complete_with_warning"):
        raise ValueError(f"detector: unacceptable status {getattr(outcome, 'status', None)}")
    return tuple(item.to_peak1d() if hasattr(item, "to_peak1d") else item for item in outcome.peaks)


def v_harm(current: float, baseline: float, metric_id: str) -> float:
    lower = {"mse", "rmse", "mae", "sam", "nmse", "wasserstein_1_cm1", "artifact_peak_ratio", "missing_peak_ratio"}
    return current - baseline if metric_id in lower else baseline - current


def v_view(identifier: str, axis: object, intensity: object) -> Spectrum1D:
    return Spectrum1D(identifier, identifier, np.ascontiguousarray(axis, dtype="<f8"), np.ascontiguousarray(intensity, dtype="<f8"))


def v_evaluate_record_sensitivity(source: Spectrum1D, condition_ids: Sequence[str], condition_spectra: Sequence[Spectrum1D], *, record_id: str, support_start_cm1: float, support_stop_cm1: float, support_point_counts: Mapping[float, int], support_max_gap_cm1: float | None = None, cwt_system: Phase3System) -> VerifierDatasetSensitivityResult:
    ids, candidates = tuple(condition_ids), tuple(condition_spectra)
    if not isinstance(source, Spectrum1D) or not record_id or len(ids) != 40 or len(candidates) != 40 or len(set(ids)) != 40:
        raise ValueError("record sensitivity: source, record ID, or 40-condition grid invalid")
    resampling_ids = tuple(f"{kind}:{spacing:g}" for kind in ("linear", "cubic", "pchip") for spacing in (.5, 1., 2.))
    normalization_ids, tolerance_ids = ("none", "maximum", "area", "snv"), ("1", "2", "4", "8")
    resampling = np.full((9, 40, 13), np.nan, dtype="<f8")
    normalization = np.full((4, 40, 13), np.nan, dtype="<f8")
    tolerance = np.full((4, 40, 5), np.nan, dtype="<f8")
    reasons, values_by_condition, detector_calls = {}, {}, 0

    def detect(spectrum: Spectrum1D) -> tuple[object, ...]:
        nonlocal detector_calls
        detector_calls += 1
        return v_detect_peaks(cwt_system, spectrum)

    for ri, label in enumerate(resampling_ids):
        kind, spacing_text = label.split(":"); spacing = float(spacing_text)
        try:
            axis = v_target_axis(support_start_cm1, spacing, int(support_point_counts[spacing]))
            if float(axis[-1]) != float(support_stop_cm1): raise ValueError("target endpoint mismatch")
            ref = v_view(f"{source.spectrum_id}:{label}", axis, v_resample(source.axis_cm1, source.intensity, axis, kind, max_in_range_native_gap_cm1=support_max_gap_cm1))
            baseline_scalar, baseline_peaks = v_scalar_metrics(axis, ref.intensity, ref.intensity), detect(ref)
            baseline_peak = v_peak_metrics(baseline_peaks, baseline_peaks, tolerance_cm1=2.)
        except Exception as error:
            reasons[f"resampling:{label}"] = type(error).__name__; continue
        for ci, candidate in enumerate(candidates):
            try:
                view = v_view(f"{candidate.spectrum_id}:{label}", axis, v_resample(candidate.axis_cm1, candidate.intensity, axis, kind, max_in_range_native_gap_cm1=support_max_gap_cm1))
                scalar, peaks = v_scalar_metrics(axis, ref.intensity, view.intensity), detect(view)
                peak = v_peak_metrics(baseline_peaks, peaks, tolerance_cm1=2.)
                joined, baseline = scalar | peak, baseline_scalar | baseline_peak
                resampling[ri, ci, :] = [v_harm(joined[metric], baseline[metric], metric) for metric in v_METRIC_IDS]
            except Exception as error:
                reasons[f"resampling:{label}:{ids[ci]}"] = type(error).__name__

    none_values, none_peaks = [None] * 40, [None] * 40
    for ni, method in enumerate(normalization_ids):
        try:
            ref = v_view(f"{source.spectrum_id}:normalization:{method}", source.axis_cm1, v_normalize(source.axis_cm1, source.intensity, method))
            baseline_scalar, baseline_peaks = v_scalar_metrics(ref.axis_cm1, ref.intensity, ref.intensity), detect(ref)
            baseline_peak = v_peak_metrics(baseline_peaks, baseline_peaks, tolerance_cm1=2.)
        except Exception as error:
            reasons[f"normalization:{method}"] = type(error).__name__; continue
        for ci, candidate in enumerate(candidates):
            try:
                view = v_view(f"{candidate.spectrum_id}:normalization:{method}", candidate.axis_cm1, v_normalize(candidate.axis_cm1, candidate.intensity, method))
                scalar, peaks = v_scalar_metrics(ref.axis_cm1, ref.intensity, view.intensity), detect(view)
                peak, joined, baseline = v_peak_metrics(baseline_peaks, peaks, tolerance_cm1=2.), scalar | v_peak_metrics(baseline_peaks, peaks, tolerance_cm1=2.), baseline_scalar | baseline_peak
                normalization[ni, ci, :] = [v_harm(joined[metric], baseline[metric], metric) for metric in v_METRIC_IDS]
                if method == "none":
                    none_values[ci], none_peaks[ci], values_by_condition[ids[ci]] = joined, (baseline_peaks, peaks), MappingProxyType(dict(joined))
                else:
                    for metric in (("pearson_r", "is_like_structure_to_noise", "sam", "wasserstein_1_cm1") if method in {"maximum", "area"} else ("pearson_r", "is_like_structure_to_noise")):
                        if none_values[ci] is not None and not v_invariance_holds(none_values[ci][metric], joined[metric]): reasons[f"normalization:{method}"] = "failed_scale_invariance_control"
            except Exception as error:
                reasons[f"normalization:{method}:{ids[ci]}"] = type(error).__name__

    for ci, pair in enumerate(none_peaks):
        if pair is None:
            reasons[f"peak_tolerance:{ids[ci]}"] = "missing_normalization_none_peaks"; continue
        baseline = v_peak_metrics(pair[0], pair[0], tolerance_cm1=2.)
        for ti, level in enumerate((1., 2., 4., 8.)):
            current = v_peak_metrics(pair[0], pair[1], tolerance_cm1=level)
            tolerance[ti, ci, :] = [v_harm(current[metric], baseline[metric], metric) for metric in v_METRIC_IDS[8:]]
    identity = MappingProxyType({"source_axis_sha256": v_array_digest(source.axis_cm1), "source_intensity_sha256": v_array_digest(source.intensity), "condition_axis_sha256": tuple(v_array_digest(item.axis_cm1) for item in candidates), "condition_intensity_sha256": tuple(v_array_digest(item.intensity) for item in candidates), "detector_call_count": detector_calls})
    return VerifierDatasetSensitivityResult(record_id, ids, v_METRIC_IDS, resampling_ids, normalization_ids, tolerance_ids, MappingProxyType(values_by_condition), MappingProxyType(dict(sorted(reasons.items()))), v_freeze_array(resampling), v_freeze_array(normalization), v_freeze_array(tolerance), identity)


def v_parent_downstream_collapse(rows: Sequence[Mapping[str, object]], *, metric_ids: Sequence[str] = v_METRIC_IDS) -> Mapping[tuple[str, str, float], float]:
    """Collapse copied downstream harms and reject any non-bit-identical metric copy."""
    expected, grouped = tuple(metric_ids), {}
    for row in rows:
        cluster = str(row.get("cluster_id", row.get("class_label", row.get("well_id", ""))))
        key = (cluster, str(row["perturbation_id"]), float(row["alpha"]))
        metric, harm = str(row["metric_output_id"]), float(row["downstream_harm"])
        if metric not in expected or not math.isfinite(harm):
            raise ValueError("parent downstream: invalid metric or harm")
        values = grouped.setdefault(key, {})
        if metric in values:
            raise ValueError("parent downstream: duplicate metric condition")
        values[metric] = harm
    output = {}
    for key, values in grouped.items():
        if tuple(values) != expected or any(values[metric] != values[expected[0]] for metric in expected):
            raise ValueError("parent downstream: incomplete grid or nonidentical copied harm")
        output[key] = values[expected[0]]
    return MappingProxyType(output)


def v_condition_grid(perturbation_ids: Sequence[str] = ("P8", "P9", "P10", "P11", "P12"), positive_alphas: Sequence[float] = (.05, .1, .2, .3, .4, .5, .65, .8)) -> tuple[tuple[str, float], ...]:
    grid = tuple((str(perturbation), float(alpha)) for perturbation in perturbation_ids for alpha in positive_alphas)
    if len(grid) != 40 or len(set(grid)) != 40:
        raise ValueError("condition grid: expected the frozen 40 unique P8-P12 positive conditions")
    return grid


def v_run_id(config_bytes: bytes, protocol_bytes: bytes, identities: Mapping[str, object], *, prefix: str = "phase6-appendix-audits-") -> str:
    """Content ID excluding host, path, worker count, and wall-clock time."""
    digest = hashlib.sha256(b"rpe-phase6-appendix-audits-v1\0" + bytes(config_bytes) + bytes(protocol_bytes) + v_canonical_json_bytes(identities)).hexdigest()
    return f"{prefix}{digest}"


def v_manifest(payloads: Mapping[str, bytes], *, payload_order: Sequence[str]) -> Mapping[str, object]:
    names = tuple(payload_order)
    if tuple(payloads) != names or len(names) != 16:
        raise ValueError("manifest: payload names or order drifted")
    return MappingProxyType({"payload_files": list(names), "sha256": {name: hashlib.sha256(payloads[name]).hexdigest() for name in names}})


def v_sha256sums(payloads: Mapping[str, bytes], *, payload_order: Sequence[str], complete_bytes: bytes) -> bytes:
    entries = [(name, payloads[name]) for name in payload_order] + [("complete.json", complete_bytes)]
    return "".join(f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in entries).encode("utf-8")


def v_exact_spectrum_sha256(axis: object, intensity: object) -> str:
    x, y = v_array("axis", axis), v_array("intensity", intensity)
    if x.shape != y.shape or not np.all(np.diff(x) > 0):
        raise ValueError("exact spectrum hash: invalid spectrum")
    payload = v_DOMAIN + struct.pack("<Q", x.size) + x.astype("<f8", copy=False).tobytes() + struct.pack("<Q", y.size) + y.astype("<f4").tobytes()
    return hashlib.sha256(payload).hexdigest()


def v_select_calibration_pairs(records: Sequence[Mapping[str, object]], *, seed: int = 20260817, target: int = 100000, batch_size: int = 262144, max_batches: int = 64) -> tuple[tuple[str, str], ...]:
    ordered = tuple(sorted(records, key=lambda row: str(row["record_id"])))
    ids, entities = tuple(str(row["record_id"]) for row in ordered), tuple(str(row["entity_id"]) for row in ordered)
    if len(set(ids)) != len(ids) or min(target, batch_size, max_batches) <= 0:
        raise ValueError("calibration: invalid records or limits")
    entity_counts: dict[str, int] = {}
    for entity in entities:
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
    admissible_count = len(ids) * (len(ids) - 1) // 2 - sum(count * (count - 1) // 2 for count in entity_counts.values())
    if admissible_count < target:
        return tuple((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)) if entities[i] != entities[j])
    rng, accepted, seen = np.random.Generator(np.random.PCG64(seed)), [], set()
    for _ in range(max_batches):
        draws = rng.integers(0, len(ids), size=(batch_size, 2))
        for left, right in draws:
            i, j = int(left), int(right)
            if i == j or entities[i] == entities[j]:
                continue
            pair = (ids[i], ids[j]) if i < j else (ids[j], ids[i])
            if pair not in seen:
                seen.add(pair); accepted.append(pair)
                if len(accepted) == target:
                    return tuple(accepted)
    raise ValueError("calibration: maximum draw batches did not fill target")


def v_snv_on_support(record: Mapping[str, object], support_axis: object) -> tuple[np.ndarray | None, str]:
    try:
        values = v_resample(record["axis"], record["intensity"], support_axis, "linear")
        return v_normalize(support_axis, values, "snv"), ""
    except (KeyError, TypeError, ValueError):
        return None, "typed_closure_constant_or_nonfinite_spectrum"


def v_scan_near_duplicates(records: Sequence[Mapping[str, object]], *, left_role_id: str, right_role_id: str, support_axis: object, threshold: float = 1e-3, block_size: int = 4096) -> tuple[dict[str, object], ...]:
    if not math.isfinite(threshold) or threshold < 0 or block_size <= 0:
        raise ValueError("near scan: invalid threshold or block size")
    left = tuple(row for row in records if str(row.get("role_id")) == left_role_id)
    right = tuple(row for row in records if str(row.get("role_id")) == right_role_id)
    left_data, right_data = [(row, *v_snv_on_support(row, support_axis)) for row in left], [(row, *v_snv_on_support(row, support_axis)) for row in right]
    rows: list[dict[str, object]] = []
    def closed_row(left_row: Mapping[str, object] | None, right_row: Mapping[str, object] | None, reason: str) -> dict[str, object]:
        return {"left_role_id": left_role_id if left_row else "", "right_role_id": right_role_id if right_row else "", "left_record_id": str(left_row.get("record_id", "")) if left_row else "", "right_record_id": str(right_row.get("record_id", "")) if right_row else "", "left_entity_id": str(left_row.get("entity_id", "")) if left_row else "", "right_entity_id": str(right_row.get("entity_id", "")) if right_row else "", "distance": None, "correlation": None, "left_source_sha256": str(left_row.get("source_sha256", "")) if left_row else "", "right_source_sha256": str(right_row.get("source_sha256", "")) if right_row else "", "status": "not_evaluable", "reason_code": reason}
    for row, vector, reason in left_data:
        if vector is None: rows.append(closed_row(row, None, reason))
    for row, vector, reason in right_data:
        if vector is None: rows.append(closed_row(None, row, reason))
    for li in range(0, len(left_data), block_size):
        for ri in range(0, len(right_data), block_size):
            left_block = [(row, vector) for row, vector, _ in left_data[li:li + block_size] if vector is not None]
            right_block = [(row, vector) for row, vector, _ in right_data[ri:ri + block_size] if vector is not None]
            if not left_block or not right_block:
                continue
            left_matrix = np.stack([vector for _, vector in left_block])
            right_matrix = np.stack([vector for _, vector in right_block])
            correlations = (left_matrix @ right_matrix.T) / left_matrix.shape[1]
            squared = np.maximum(0.0, 2.0 - 2.0 * correlations)
            hit_rows, hit_columns = np.nonzero(squared <= threshold * threshold)
            for local_left, local_right in zip(hit_rows, hit_columns, strict=True):
                left_row, left_vector = left_block[int(local_left)]
                right_row, right_vector = right_block[int(local_right)]
                distance = float(np.sqrt(np.mean((left_vector - right_vector) ** 2)))
                if distance > threshold:
                    continue
                exact = v_exact_spectrum_sha256(left_row["axis"], left_row["intensity"]) == v_exact_spectrum_sha256(right_row["axis"], right_row["intensity"])
                status = "fail" if exact else "warning_near_duplicate_bridge"
                reason = "fail_exact_duplicate_bridge" if exact else "warning_near_duplicate_bridge"
                rows.append({"left_role_id": left_role_id, "right_role_id": right_role_id, "left_record_id": str(left_row["record_id"]), "right_record_id": str(right_row["record_id"]), "left_entity_id": str(left_row["entity_id"]), "right_entity_id": str(right_row["entity_id"]), "distance": distance, "correlation": float(correlations[int(local_left), int(local_right)]), "left_source_sha256": str(left_row.get("source_sha256", "")), "right_source_sha256": str(right_row.get("source_sha256", "")), "status": status, "reason_code": reason})
    return tuple(sorted(rows, key=lambda row: (row["left_record_id"], row["right_record_id"], row["reason_code"])))


def v_ready(value: object) -> object:
    if isinstance(value, Mapping): return {str(key): v_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray): return [v_ready(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)): return [v_ready(item) for item in value]
    if isinstance(value, np.generic): return value.item()
    return value


def v_canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(v_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def v_canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(v_canonical_json_bytes(row) for row in rows)


def v_canonical_csv_bytes(schema_and_rows: tuple[Sequence[str], Iterable[Mapping[str, object]]]) -> bytes:
    columns, rows = tuple(schema_and_rows[0]), schema_and_rows[1]
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([format(value, ".17g") if isinstance(value, (float, np.floating)) else value for value in (row[column] for column in columns)])
    return stream.getvalue().encode("utf-8")
