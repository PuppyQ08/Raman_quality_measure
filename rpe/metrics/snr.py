from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rpe.evaluation import (
    AxisPolicy,
    MetricInput,
    MetricInputKind,
    MetricResult,
    PreferredDirection,
    ScalarMetricOutput,
    SpectralRegion,
    SpectrumRegionsInput,
)


class SNRMetricError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class _ScaledAnchoredStatistics:
    intensity_scale_power2_exponent: int
    scaled_signal: np.ndarray
    scaled_denominator: np.ndarray
    scaled_denominator_anchor: float
    scaled_denominator_mean_delta: float
    scaled_denominator_rms: float


@dataclass(frozen=True)
class _SignedRatioComputation:
    scaled_signed_numerator: float
    scaled_denominator: float
    ratio: float


@dataclass(frozen=True)
class _ScaledRelativeTrapezoidWeights:
    axis_scale_power2_exponent: int
    normalized_weights: np.ndarray
    scaled_weight_l2_norm: float


def _require_regions_request(request: MetricInput) -> SpectrumRegionsInput:
    if not isinstance(request, SpectrumRegionsInput):
        raise SNRMetricError(
            "request",
            "must be SpectrumRegionsInput",
        )
    return request


def _consumed_regions(
    request: SpectrumRegionsInput,
    denominator_role: str,
) -> tuple[SpectralRegion, SpectralRegion]:
    signal_regions = [
        region for region in request.regions if region.role == "signal"
    ]
    denominator_regions = [
        region
        for region in request.regions
        if region.role == denominator_role
    ]
    if len(signal_regions) != 1:
        raise SNRMetricError(
            "regions.signal",
            "must contain exactly one consumed region",
        )
    if len(denominator_regions) != 1:
        raise SNRMetricError(
            f"regions.{denominator_role}",
            "must contain exactly one consumed region",
        )
    signal_region = signal_regions[0]
    denominator_region = denominator_regions[0]
    if not (
        signal_region.end_cm1 < denominator_region.start_cm1
        or denominator_region.end_cm1 < signal_region.start_cm1
    ):
        raise SNRMetricError(
            "regions overlap",
            "consumed intervals must be disjoint as closed intervals",
        )
    return signal_region, denominator_region


def _closed_region_indices(
    axis: np.ndarray,
    region: SpectralRegion,
    *,
    min_points: int,
) -> np.ndarray:
    mask = (axis >= region.start_cm1) & (axis <= region.end_cm1)
    indices = np.flatnonzero(mask)
    if indices.size < min_points:
        raise SNRMetricError(
            f"{region.role}_region point_count",
            f"must select at least {min_points} native samples",
        )
    return indices


def _require_finite_float(path: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise SNRMetricError(path, "produced non-finite value")
    return scalar


def _scale_exponent(values: np.ndarray) -> int:
    if values.size == 0:
        return 0
    max_abs = float(np.max(np.abs(values)))
    if max_abs == 0.0:
        return 0
    _, exponent = math.frexp(max_abs)
    return exponent - 1


def _power2_scale(exponent: int) -> float:
    return math.ldexp(1.0, exponent)


def _scale_values(path: str, values: np.ndarray, exponent: int) -> np.ndarray:
    scaled = np.asarray(
        values / _power2_scale(exponent),
        dtype=np.float64,
    )
    if not np.isfinite(scaled).all():
        raise SNRMetricError(path, "produced non-finite scaled values")
    lost = (values != 0.0) & (scaled == 0.0)
    if np.any(lost):
        raise SNRMetricError(
            path,
            "required nonzero value became zero after scaling",
        )
    return scaled


def _checked_subtract(path: str, left: float, right: float) -> float:
    difference = float(left - right)
    if not math.isfinite(difference):
        raise SNRMetricError(path, "produced non-finite difference")
    if left != right and difference == 0.0:
        raise SNRMetricError(
            path,
            "required nonzero difference became zero",
        )
    return difference


def _checked_normalized_values(
    path: str,
    values: np.ndarray,
) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(values))) if values.size else 0.0
    if max_abs == 0.0:
        return np.asarray(values, dtype=np.float64), 0.0
    exponent = _scale_exponent(values)
    normalized = np.asarray(
        values / _power2_scale(exponent),
        dtype=np.float64,
    )
    if not np.isfinite(normalized).all():
        raise SNRMetricError(path, "produced non-finite normalized values")
    lost = (values != 0.0) & (normalized == 0.0)
    if np.any(lost):
        raise SNRMetricError(
            path,
            "required nonzero term became zero after normalization",
        )
    return normalized, _power2_scale(exponent)


def _scaled_anchored_statistics(
    signal_values: np.ndarray,
    denominator_values: np.ndarray,
) -> _ScaledAnchoredStatistics:
    consumed = np.concatenate((signal_values, denominator_values))
    exponent = _scale_exponent(consumed)
    scaled_signal = _scale_values(
        "signal values",
        signal_values,
        exponent,
    )
    scaled_denominator = _scale_values(
        "denominator values",
        denominator_values,
        exponent,
    )

    anchor = float(scaled_denominator[0])
    deltas = np.asarray(
        [
            _checked_subtract(
                "denominator delta",
                float(value),
                anchor,
            )
            for value in scaled_denominator
        ],
        dtype=np.float64,
    )
    mean_delta = _require_finite_float(
        "denominator mean delta",
        math.fsum(float(value) for value in deltas) / deltas.size,
    )
    residuals = np.asarray(
        [
            _checked_subtract(
                "denominator residual",
                float(delta),
                mean_delta,
            )
            for delta in deltas
        ],
        dtype=np.float64,
    )
    normalized_residuals, residual_scale = _checked_normalized_values(
        "denominator residual",
        residuals,
    )
    if residual_scale == 0.0:
        raise SNRMetricError(
            "denominator rms",
            "zero centered population RMS",
        )
    normalized_squares: list[float] = []
    for value in normalized_residuals:
        square = float(value * value)
        if value != 0.0 and square == 0.0:
            raise SNRMetricError(
                "denominator residual square",
                "required nonzero L2 term became zero",
            )
        if not math.isfinite(square):
            raise SNRMetricError(
                "denominator residual square",
                "produced non-finite value",
            )
        normalized_squares.append(square)
    sum_squares = _require_finite_float(
        "denominator residual squares",
        math.fsum(normalized_squares),
    )
    rms = _require_finite_float(
        "denominator rms",
        residual_scale * math.sqrt(sum_squares / residuals.size),
    )
    if rms == 0.0:
        raise SNRMetricError(
            "denominator rms",
            "zero centered population RMS",
        )

    return _ScaledAnchoredStatistics(
        intensity_scale_power2_exponent=exponent,
        scaled_signal=scaled_signal,
        scaled_denominator=scaled_denominator,
        scaled_denominator_anchor=anchor,
        scaled_denominator_mean_delta=mean_delta,
        scaled_denominator_rms=rms,
    )


def _signed_ratio(
    numerator: float,
    denominator: float,
) -> _SignedRatioComputation:
    numerator_value = _require_finite_float("signed numerator", numerator)
    denominator_value = _require_finite_float(
        "denominator",
        denominator,
    )
    if denominator_value <= 0.0:
        raise SNRMetricError(
            "denominator",
            "must be positive",
        )
    ratio = _require_finite_float(
        "ratio",
        numerator_value / denominator_value,
    )
    return _SignedRatioComputation(
        scaled_signed_numerator=numerator_value,
        scaled_denominator=denominator_value,
        ratio=ratio,
    )


def _shared_diagnostics(
    signal_region: SpectralRegion,
    denominator_region: SpectralRegion,
    signal_count: int,
    denominator_count: int,
    statistics: _ScaledAnchoredStatistics,
    signed_numerator: float,
) -> dict[str, object]:
    return {
        "signal_region": {
            "region_id": signal_region.region_id,
            "role": signal_region.role,
            "start_cm1": float(signal_region.start_cm1),
            "end_cm1": float(signal_region.end_cm1),
            "point_count": int(signal_count),
        },
        "denominator_region": {
            "region_id": denominator_region.region_id,
            "role": denominator_region.role,
            "start_cm1": float(denominator_region.start_cm1),
            "end_cm1": float(denominator_region.end_cm1),
            "point_count": int(denominator_count),
        },
        "region_endpoint_policy": "closed_native_samples",
        "region_overlap_policy": "disjoint_closed_physical_intervals",
        "signed_output": True,
        "noise_rms_convention": "centered_population_rms",
        "intensity_scale_power2_exponent": int(
            statistics.intensity_scale_power2_exponent
        ),
        "scaled_denominator_anchor": float(
            statistics.scaled_denominator_anchor
        ),
        "scaled_denominator_mean_delta": float(
            statistics.scaled_denominator_mean_delta
        ),
        "scaled_denominator_rms": float(
            statistics.scaled_denominator_rms
        ),
        "scaled_signed_numerator": float(signed_numerator),
    }


def _scaled_relative_trapezoid_weights(
    signal_axis: np.ndarray,
) -> _ScaledRelativeTrapezoidWeights:
    axis_values = np.asarray(signal_axis, dtype=np.float64)
    if axis_values.size < 2:
        raise SNRMetricError(
            "signal_region point_count",
            "must select at least 2 native samples",
        )

    axis_exponent = _scale_exponent(axis_values)
    axis_scale = _power2_scale(axis_exponent)
    scaled_axis = np.asarray(axis_values / axis_scale, dtype=np.float64)
    if not np.isfinite(scaled_axis).all():
        raise SNRMetricError("signal axis", "produced non-finite scaled values")

    anchor = float(scaled_axis[0])
    anchored_axis = np.asarray(
        [
            _checked_subtract("scaled axis coordinate", float(value), anchor)
            for value in scaled_axis
        ],
        dtype=np.float64,
    )

    intervals: list[float] = []
    for index in range(anchored_axis.size - 1):
        left_original = float(axis_values[index + 1])
        right_original = float(axis_values[index])
        original_interval = float(left_original - right_original)
        if not math.isfinite(original_interval):
            raise SNRMetricError("axis interval", "produced non-finite value")
        left_anchored = float(anchored_axis[index + 1])
        right_anchored = float(anchored_axis[index])
        interval = float(left_anchored - right_anchored)
        if not math.isfinite(interval):
            raise SNRMetricError("axis interval", "produced non-finite value")
        if (
            original_interval > 0.0
            and interval == 0.0
        ):
            raise SNRMetricError(
                "axis interval",
                "positive original interval became zero after scaling",
            )
        if interval <= 0.0:
            raise SNRMetricError(
                "axis interval",
                "must be positive and finite",
            )
        intervals.append(interval)

    weights = np.empty(axis_values.size, dtype=np.float64)
    weights[0] = intervals[0] / 2.0
    weights[-1] = intervals[-1] / 2.0
    for index in range(1, axis_values.size - 1):
        weights[index] = (intervals[index - 1] + intervals[index]) / 2.0
    if not np.isfinite(weights).all():
        raise SNRMetricError("relative weight", "produced non-finite value")
    if np.any(weights <= 0.0):
        raise SNRMetricError("relative weight", "must be positive and finite")

    normalized_weights, _ = _checked_normalized_values("relative weight", weights)
    squares: list[float] = []
    for value in normalized_weights:
        square = float(value * value)
        if value != 0.0 and square == 0.0:
            raise SNRMetricError(
                "weight square",
                "required nonzero L2 term became zero",
            )
        if not math.isfinite(square):
            raise SNRMetricError("weight square", "produced non-finite value")
        squares.append(square)
    squared_norm = _require_finite_float(
        "weight squares",
        math.fsum(squares),
    )
    l2_norm = _require_finite_float(
        "weight L2 norm",
        math.sqrt(squared_norm),
    )
    if l2_norm <= 0.0:
        raise SNRMetricError("weight L2 norm", "must be positive")

    return _ScaledRelativeTrapezoidWeights(
        axis_scale_power2_exponent=axis_exponent,
        normalized_weights=normalized_weights,
        scaled_weight_l2_norm=l2_norm,
    )


@dataclass(frozen=True, init=False)
class SignedPeakHeightRmsNoiseSnrMetric:
    """Signed standardized contrast; not a power SNR or dB SNR."""

    metric_id: str = "signed_snr_peak_height_rms_noise"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SPECTRUM_REGIONS,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.SINGLE_AXIS,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = _require_regions_request(request)
        signal_region, noise_region = _consumed_regions(
            typed_request,
            "noise",
        )
        axis = typed_request.spectrum.axis_cm1
        intensity = typed_request.spectrum.intensity
        signal_indices = _closed_region_indices(
            axis,
            signal_region,
            min_points=1,
        )
        noise_indices = _closed_region_indices(
            axis,
            noise_region,
            min_points=2,
        )
        statistics = _scaled_anchored_statistics(
            intensity[signal_indices],
            intensity[noise_indices],
        )
        scaled_peak = _require_finite_float(
            "signal peak",
            float(np.max(statistics.scaled_signal)),
        )
        anchored_peak = _checked_subtract(
            "peak anchor difference",
            scaled_peak,
            statistics.scaled_denominator_anchor,
        )
        numerator = _checked_subtract(
            "signed numerator",
            anchored_peak,
            statistics.scaled_denominator_mean_delta,
        )
        ratio = _signed_ratio(
            numerator,
            statistics.scaled_denominator_rms,
        )
        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    output_id=self.metric_id,
                    value=ratio.ratio,
                    unit="ratio",
                    preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                    target_value=None,
                ),
            ),
            diagnostics=_shared_diagnostics(
                signal_region,
                noise_region,
                int(signal_indices.size),
                int(noise_indices.size),
                statistics,
                ratio.scaled_signed_numerator,
            ),
        )


@dataclass(frozen=True, init=False)
class SignedIntegratedAreaRmsNoiseSnrMetric:
    """Signed standardized contrast; not a power SNR or dB SNR."""

    metric_id: str = "signed_snr_integrated_area_rms_noise"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SPECTRUM_REGIONS,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.SINGLE_AXIS,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = _require_regions_request(request)
        signal_region, noise_region = _consumed_regions(
            typed_request,
            "noise",
        )
        axis = typed_request.spectrum.axis_cm1
        intensity = typed_request.spectrum.intensity
        signal_indices = _closed_region_indices(
            axis,
            signal_region,
            min_points=2,
        )
        noise_indices = _closed_region_indices(
            axis,
            noise_region,
            min_points=2,
        )
        statistics = _scaled_anchored_statistics(
            intensity[signal_indices],
            intensity[noise_indices],
        )
        weights = _scaled_relative_trapezoid_weights(axis[signal_indices])

        ordinates = np.asarray(
            [
                _require_finite_float(
                    "signal ordinate",
                    math.fsum(
                        (
                            float(value),
                            -statistics.scaled_denominator_anchor,
                            -statistics.scaled_denominator_mean_delta,
                        )
                    ),
                )
                for value in statistics.scaled_signal
            ],
            dtype=np.float64,
        )
        weighted_terms: list[float] = []
        for weight, ordinate in zip(weights.normalized_weights, ordinates, strict=True):
            term = float(weight * ordinate)
            if weight != 0.0 and ordinate != 0.0 and term == 0.0:
                raise SNRMetricError(
                    "weighted signal term",
                    "required nonzero product became zero",
                )
            if not math.isfinite(term):
                raise SNRMetricError(
                    "weighted signal term",
                    "produced non-finite value",
                )
            weighted_terms.append(term)
        numerator = _require_finite_float(
            "signed numerator",
            math.fsum(weighted_terms),
        )
        denominator = _require_finite_float(
            "denominator",
            statistics.scaled_denominator_rms
            * weights.scaled_weight_l2_norm,
        )
        ratio = _signed_ratio(numerator, denominator)

        diagnostics = _shared_diagnostics(
            signal_region,
            noise_region,
            int(signal_indices.size),
            int(noise_indices.size),
            statistics,
            ratio.scaled_signed_numerator,
        )
        diagnostics.update(
            {
                "axis_scale_power2_exponent": int(
                    weights.axis_scale_power2_exponent
                ),
                "scaled_weight_l2_norm": float(
                    weights.scaled_weight_l2_norm
                ),
                "area_quadrature": "native_trapezoid_node_weights",
                "noise_model": (
                    "conditional_iid_signal_samples_with_fixed_estimated_baseline"
                ),
            }
        )
        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    output_id=self.metric_id,
                    value=ratio.ratio,
                    unit="ratio",
                    preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                    target_value=None,
                ),
            ),
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, init=False)
class SignedReferenceIntervalMeanSnrMetric:
    """Signed standardized contrast; not a power SNR or dB SNR."""

    metric_id: str = "signed_snr_reference_interval_mean"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SPECTRUM_REGIONS,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.SINGLE_AXIS,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = _require_regions_request(request)
        signal_region, reference_region = _consumed_regions(
            typed_request,
            "reference",
        )
        axis = typed_request.spectrum.axis_cm1
        intensity = typed_request.spectrum.intensity
        signal_indices = _closed_region_indices(
            axis,
            signal_region,
            min_points=1,
        )
        reference_indices = _closed_region_indices(
            axis,
            reference_region,
            min_points=2,
        )
        statistics = _scaled_anchored_statistics(
            intensity[signal_indices],
            intensity[reference_indices],
        )

        anchored_signal_terms = np.asarray(
            [
                _checked_subtract(
                    "signal anchor difference",
                    float(value),
                    statistics.scaled_denominator_anchor,
                )
                for value in statistics.scaled_signal
            ],
            dtype=np.float64,
        )
        anchored_signal_mean = _require_finite_float(
            "signal mean anchor difference",
            math.fsum(float(value) for value in anchored_signal_terms)
            / anchored_signal_terms.size,
        )
        numerator = _require_finite_float(
            "signed numerator",
            _checked_subtract(
                "signed numerator",
                anchored_signal_mean,
                statistics.scaled_denominator_mean_delta,
            ),
        )
        ratio = _signed_ratio(
            numerator,
            statistics.scaled_denominator_rms,
        )
        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    output_id=self.metric_id,
                    value=ratio.ratio,
                    unit="ratio",
                    preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                    target_value=None,
                ),
            ),
            diagnostics=_shared_diagnostics(
                signal_region,
                reference_region,
                int(signal_indices.size),
                int(reference_indices.size),
                statistics,
                ratio.scaled_signed_numerator,
            ),
        )


__all__ = [
    "SNRMetricError",
    "SignedIntegratedAreaRmsNoiseSnrMetric",
    "SignedPeakHeightRmsNoiseSnrMetric",
    "SignedReferenceIntervalMeanSnrMetric",
]
