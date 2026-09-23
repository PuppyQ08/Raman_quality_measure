"""Record-local scientific kernel for the Step-7 appendix sensitivity audit."""
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from rpe.evaluation import PeakPairInput, PreferredDirection, SingleSpectrumInput, Spectrum1D, SpectrumPairInput, evaluate_metric
from rpe.methods.catalog import Phase3System
from rpe.methods.classical.peaks import PeakRunStatus, run_peak_detection_system
from rpe.metrics import (
    ISLikeStructureToNoiseMetric, MAEMetric, MSEMetric, NMSEMetric,
    PeakDetectionCurvesMetric, PearsonRMetric, RMSEMetric, SAMMetric, Wasserstein1Metric,
)
from rpe.perturb import PerturbationSweepConfig
from rpe.runner.phase1_config import Phase1CoreConfig
from rpe.runner.phase1_perturbations import P10MemoryAdmission, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase6_appendix_audits import normalize_spectrum, resample_spectrum, target_axis


METRIC_IDS = (
    "mse", "rmse", "mae", "sam", "pearson_r", "nmse", "wasserstein_1_cm1",
    "is_like_structure_to_noise", "precision", "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio",
)
PEAK_METRIC_IDS = METRIC_IDS[8:]
PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
NORMALIZATION_IDS = ("none", "maximum", "area", "snv")
TOLERANCE_IDS = ("1", "2", "4", "8")
_DIRECTIONS = {
    **{name: PreferredDirection.LOWER_IS_BETTER for name in ("mse", "rmse", "mae", "sam", "nmse", "wasserstein_1_cm1", "artifact_peak_ratio", "missing_peak_ratio")},
    **{name: PreferredDirection.HIGHER_IS_BETTER for name in ("pearson_r", "is_like_structure_to_noise", "precision", "recall", "f1")},
}
_INVARIANTS = {
    "maximum": ("pearson_r", "is_like_structure_to_noise", "sam", "wasserstein_1_cm1"),
    "area": ("pearson_r", "is_like_structure_to_noise", "sam", "wasserstein_1_cm1"),
    "snv": ("pearson_r", "is_like_structure_to_noise"),
}


@dataclass(frozen=True)
class RecordSensitivityValues:
    record_id: str
    condition_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    resampling_ids: tuple[str, ...]
    normalization_ids: tuple[str, ...]
    tolerance_ids: tuple[str, ...]
    resampling_harms: np.ndarray
    normalization_harms: np.ndarray
    peak_tolerance_harms: np.ndarray
    closure_reasons: Mapping[str, str]
    identity: Mapping[str, object]


def _freeze_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype="<f8")
    result.setflags(write=False)
    return result


def pack_record_sensitivity_values(value: RecordSensitivityValues) -> tuple[object, ...]:
    """Convert an immutable record result to process-queue-safe primitives."""
    return (
        value.record_id, tuple(value.condition_ids), tuple(value.metric_ids),
        tuple(value.resampling_ids), tuple(value.normalization_ids), tuple(value.tolerance_ids),
        np.ascontiguousarray(value.resampling_harms, dtype="<f8"),
        np.ascontiguousarray(value.normalization_harms, dtype="<f8"),
        np.ascontiguousarray(value.peak_tolerance_harms, dtype="<f8"),
        dict(value.closure_reasons), dict(value.identity),
    )


def unpack_record_sensitivity_values(payload: tuple[object, ...]) -> RecordSensitivityValues:
    if len(payload) != 11:
        raise ValueError("record transport payload shape drift")
    record_id, condition_ids, metric_ids, resampling_ids, normalization_ids, tolerance_ids, resampling, normalization, tolerance, reasons, identity = payload
    if not isinstance(reasons, Mapping) or not isinstance(identity, Mapping):
        raise ValueError("record transport mapping drift")
    return RecordSensitivityValues(
        str(record_id), tuple(condition_ids), tuple(metric_ids), tuple(resampling_ids), tuple(normalization_ids), tuple(tolerance_ids),
        _freeze_array(np.asarray(resampling, dtype="<f8")), _freeze_array(np.asarray(normalization, dtype="<f8")), _freeze_array(np.asarray(tolerance, dtype="<f8")),
        MappingProxyType(dict(sorted((str(key), str(value)) for key, value in reasons.items()))),
        MappingProxyType(dict(identity)),
    )


def _digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(struct.pack("<Q", array.size) + array.tobytes()).hexdigest()


def _condition_id(perturbation_id: str, alpha: float) -> str:
    return f"{perturbation_id}:{struct.pack('<d', float(alpha)).hex()}"


def rematerialize_p8_p12_conditions(source: Phase1Source, phase1_config: Phase1CoreConfig, sweep: PerturbationSweepConfig, *, p10_memory_budget_bytes: int) -> tuple[tuple[str, ...], tuple[Spectrum1D, ...]]:
    if not isinstance(source, Phase1Source):
        raise ValueError("source must be Phase1Source")
    if not isinstance(p10_memory_budget_bytes, int) or p10_memory_budget_bytes <= 0:
        raise ValueError("p10_memory_budget_bytes must be positive")
    expected_alphas = tuple(float(value) for value in sweep.alpha_grid)
    if expected_alphas != (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8):
        raise ValueError("sweep must use the frozen alpha grid")
    ids: list[str] = []
    outputs: list[Spectrum1D] = []
    admission = P10MemoryAdmission(p10_memory_budget_bytes)
    for perturbation_id in PERTURBATION_IDS:
        cell = run_perturbation_cell(source, perturbation_id, phase1_config, sweep, p10_admission=admission if perturbation_id == "p10" else None)
        if cell.status is not CellStatus.COMPLETE or tuple(float(row.alpha) for row in cell.records) != expected_alphas:
            raise ValueError(f"{perturbation_id}: perturbation cell incomplete or alpha grid drift")
        zero = cell.records[0].result.output
        if not np.array_equal(zero.axis_cm1, source.spectrum.axis_cm1) or not np.array_equal(zero.intensity, source.spectrum.intensity):
            raise ValueError(f"{perturbation_id}: alpha-zero identity failure")
        for item in cell.records[1:]:
            ids.append(_condition_id(perturbation_id, item.alpha))
            outputs.append(item.result.output)
    if len(ids) != 40:
        raise ValueError("P8-P12 must yield exactly forty positive conditions")
    return tuple(ids), tuple(outputs)


def _scalar_values(reference: Spectrum1D, candidate: Spectrum1D) -> dict[str, float]:
    metrics = {
        "mse": MSEMetric(), "rmse": RMSEMetric(), "mae": MAEMetric(), "sam": SAMMetric(),
        "pearson_r": PearsonRMetric(), "nmse": NMSEMetric(), "wasserstein_1_cm1": Wasserstein1Metric(),
        "is_like_structure_to_noise": ISLikeStructureToNoiseMetric(),
    }
    values: dict[str, float] = {}
    for name, metric in metrics.items():
        request = SingleSpectrumInput(candidate) if name == "is_like_structure_to_noise" else SpectrumPairInput(reference, candidate)
        result = evaluate_metric(metric, request)
        rows = [row for row in result.outputs if row.output_id == name]
        if len(rows) != 1 or not math.isfinite(float(rows[0].value)):
            raise ValueError(f"metric {name} produced no finite scalar")
        values[name] = float(rows[0].value)
    return values


def _peaks(system: Phase3System, spectrum: Spectrum1D):
    result = run_peak_detection_system(system, spectrum)
    status = getattr(result, "status", None)
    if status not in (PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING, "complete", "complete_with_warning"):
        raise ValueError(f"detector status {getattr(status, 'value', status)}")
    return tuple(item.to_peak1d() if hasattr(item, "to_peak1d") else item for item in result.peaks)


def _peak_values(reference, candidate, tolerance: float) -> dict[str, float]:
    result = evaluate_metric(PeakDetectionCurvesMetric(), PeakPairInput(reference, candidate, tolerance, (0.0,)))
    rows = {row.output_id: float(row.value) for row in result.outputs if row.output_id in PEAK_METRIC_IDS}
    if tuple(rows) != PEAK_METRIC_IDS or not all(math.isfinite(value) for value in rows.values()):
        raise ValueError("peak metric output incomplete")
    return rows


def _harm(current: float, baseline: float, metric_id: str) -> float:
    return current - baseline if _DIRECTIONS[metric_id] is PreferredDirection.LOWER_IS_BETTER else baseline - current


def _view(identifier: str, axis: np.ndarray, values: np.ndarray) -> Spectrum1D:
    return Spectrum1D(identifier, identifier, np.ascontiguousarray(axis, dtype="<f8"), np.ascontiguousarray(values, dtype="<f8"))


def evaluate_record_sensitivity(source: Spectrum1D, condition_ids: Sequence[str], condition_spectra: Sequence[Spectrum1D], *, record_id: str, support_start_cm1: float, support_stop_cm1: float, support_point_counts: Mapping[float, int], support_max_gap_cm1: float | None = None, cwt_system: Phase3System) -> RecordSensitivityValues:
    ids, candidates = tuple(condition_ids), tuple(condition_spectra)
    if not isinstance(source, Spectrum1D) or not record_id or len(ids) != 40 or len(ids) != len(candidates):
        raise ValueError("source, record ID, and condition grid are invalid")
    if len(set(ids)) != len(ids):
        raise ValueError("condition IDs must be unique")
    expected_resampling = tuple(f"{kind}:{spacing:g}" for kind in ("linear", "cubic", "pchip") for spacing in (0.5, 1.0, 2.0))
    r_harms = np.full((9, len(ids), len(METRIC_IDS)), np.nan, dtype="<f8")
    n_harms = np.full((4, len(ids), len(METRIC_IDS)), np.nan, dtype="<f8")
    t_harms = np.full((4, len(ids), len(PEAK_METRIC_IDS)), np.nan, dtype="<f8")
    reasons: dict[str, str] = {}
    calls = 0

    def detect(spectrum: Spectrum1D):
        nonlocal calls
        result = _peaks(cwt_system, spectrum)
        calls += 1
        return result

    for r_index, (kind, spacing) in enumerate((pair for pair in ((kind, spacing) for kind in ("linear", "cubic", "pchip") for spacing in (0.5, 1.0, 2.0)))):
        key = f"{kind}:{spacing:g}"
        try:
            count = support_point_counts[spacing]
            axis = target_axis(support_start_cm1, spacing, count)
            if float(axis[-1]) != float(support_stop_cm1): raise ValueError("target endpoint mismatch")
            ref = _view(f"{source.spectrum_id}:{key}", axis, resample_spectrum(source.axis_cm1, source.intensity, axis, interpolator=kind, max_in_range_native_gap_cm1=support_max_gap_cm1))
            ref_scalar, ref_peaks = _scalar_values(ref, ref), detect(ref)
            ref_peak = _peak_values(ref_peaks, ref_peaks, 2.0)
        except Exception as error:
            reasons[f"resampling:{key}"] = type(error).__name__
            continue
        for c_index, candidate in enumerate(candidates):
            try:
                view = _view(f"{candidate.spectrum_id}:{key}", axis, resample_spectrum(candidate.axis_cm1, candidate.intensity, axis, interpolator=kind, max_in_range_native_gap_cm1=support_max_gap_cm1))
                scalar, current_peaks = _scalar_values(ref, view), detect(view)
                peak = _peak_values(ref_peaks, current_peaks, 2.0)
                r_harms[r_index, c_index, :] = [_harm((scalar | peak)[metric], (ref_scalar | ref_peak)[metric], metric) for metric in METRIC_IDS]
            except Exception as error:
                reasons[f"resampling:{key}:{ids[c_index]}"] = type(error).__name__

    none_values: list[dict[str, float] | None] = [None] * len(ids)
    none_peaks: list[tuple[tuple[object, ...], tuple[object, ...]] | None] = [None] * len(ids)
    for n_index, method in enumerate(NORMALIZATION_IDS):
        try:
            ref = _view(f"{source.spectrum_id}:normalization:{method}", source.axis_cm1, normalize_spectrum(source.axis_cm1, source.intensity, method=method))
            ref_scalar, ref_peaks = _scalar_values(ref, ref), detect(ref)
            ref_peak = _peak_values(ref_peaks, ref_peaks, 2.0)
        except Exception as error:
            reasons[f"normalization:{method}"] = type(error).__name__
            continue
        for c_index, candidate in enumerate(candidates):
            try:
                view = _view(f"{candidate.spectrum_id}:normalization:{method}", candidate.axis_cm1, normalize_spectrum(candidate.axis_cm1, candidate.intensity, method=method))
                scalar, cp = _scalar_values(ref, view), detect(view)
                rp, peak = ref_peaks, _peak_values(ref_peaks, cp, 2.0)
                values = scalar | peak
                baseline = ref_scalar | ref_peak
                n_harms[n_index, c_index, :] = [_harm(values[metric], baseline[metric], metric) for metric in METRIC_IDS]
                if method == "none":
                    none_values[c_index] = values
                    none_peaks[c_index] = (rp, cp)
                elif none_values[c_index] is not None:
                    for metric in _INVARIANTS.get(method, ()):
                        if abs(values[metric] - none_values[c_index][metric]) > 1e-12 + 1e-10 * abs(none_values[c_index][metric]):
                            reasons[f"normalization:{method}"] = "failed_scale_invariance_control"
                            break
            except Exception as error:
                reasons[f"normalization:{method}:{ids[c_index]}"] = type(error).__name__

    for c_index, peak_pair in enumerate(none_peaks):
        if peak_pair is None:
            reasons[f"peak_tolerance:{ids[c_index]}"] = "missing_normalization_none_peaks"
            continue
        try:
            baseline = _peak_values(peak_pair[0], peak_pair[0], 2.0)
            for t_index, tolerance in enumerate((1.0, 2.0, 4.0, 8.0)):
                current = _peak_values(peak_pair[0], peak_pair[1], tolerance)
                t_harms[t_index, c_index, :] = [_harm(current[metric], baseline[metric], metric) for metric in PEAK_METRIC_IDS]
        except Exception as error:
            reasons[f"peak_tolerance:{ids[c_index]}"] = type(error).__name__

    identity = MappingProxyType({
        "source_axis_sha256": _digest(source.axis_cm1), "source_intensity_sha256": _digest(source.intensity),
        "condition_axis_sha256": tuple(_digest(value.axis_cm1) for value in candidates),
        "condition_intensity_sha256": tuple(_digest(value.intensity) for value in candidates),
        "detector_call_count": calls,
    })
    return RecordSensitivityValues(record_id, ids, METRIC_IDS, expected_resampling, NORMALIZATION_IDS, TOLERANCE_IDS, _freeze_array(r_harms), _freeze_array(n_harms), _freeze_array(t_harms), MappingProxyType(dict(sorted(reasons.items()))), identity)
