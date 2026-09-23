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
    SpectrumPairInput,
)


class TransportMetricError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class _ScaledProducts:
    values: np.ndarray
    shift_exponent: int
    positive_mask: np.ndarray
    underflow_mask: np.ndarray


def _require_spectrum_pair(request: MetricInput) -> SpectrumPairInput:
    if not isinstance(request, SpectrumPairInput):
        raise TransportMetricError(
            "request",
            "must be SpectrumPairInput",
        )
    return request


def _stable_positive_intervals(
    path: str,
    axis: np.ndarray,
) -> np.ndarray:
    if axis.size < 2:
        raise TransportMetricError(
            f"{path} point count",
            "must be at least 2",
        )
    try:
        with np.errstate(over="raise", invalid="raise"):
            intervals = axis[1:] - axis[:-1]
    except FloatingPointError as error:
        raise TransportMetricError(
            f"{path} intervals",
            "contains non-finite derived values",
        ) from error
    if not np.isfinite(intervals).all():
        raise TransportMetricError(
            f"{path} intervals",
            "contains non-finite derived values",
        )
    if np.any(intervals <= 0.0):
        raise TransportMetricError(
            f"{path} intervals",
            "must be positive",
        )
    return np.asarray(intervals, dtype=np.float64)


def _trapezoid_node_widths(
    path: str,
    axis: np.ndarray,
) -> np.ndarray:
    intervals = _stable_positive_intervals(path, axis)
    widths = np.empty(axis.shape, dtype=np.float64)
    try:
        with np.errstate(over="raise", invalid="raise"):
            widths[0] = intervals[0] / 2.0
            widths[-1] = intervals[-1] / 2.0
            if intervals.size > 1:
                widths[1:-1] = (intervals[:-1] + intervals[1:]) / 2.0
    except FloatingPointError as error:
        raise TransportMetricError(
            f"{path} widths",
            "contains non-finite derived values",
        ) from error
    if not np.isfinite(widths).all():
        raise TransportMetricError(
            f"{path} widths",
            "contains non-finite derived values",
        )
    if np.any(widths <= 0.0):
        raise TransportMetricError(
            f"{path} widths",
            "must be positive",
        )
    return widths


def _scaled_nonnegative_products(
    values: np.ndarray,
    widths: np.ndarray,
    *,
    shift_exponent: int | None = None,
) -> _ScaledProducts:
    products = np.zeros(values.shape, dtype=np.float64)
    positive_mask = values > 0.0
    if not np.any(positive_mask):
        chosen_shift = 0 if shift_exponent is None else shift_exponent
        return _ScaledProducts(
            values=products,
            shift_exponent=chosen_shift,
            positive_mask=positive_mask,
            underflow_mask=np.zeros(values.shape, dtype=bool),
        )

    value_mantissa, value_exponent = np.frexp(values[positive_mask])
    width_mantissa, width_exponent = np.frexp(widths[positive_mask])
    combined_exponent = (
        value_exponent.astype(np.int64) + width_exponent.astype(np.int64)
    )
    chosen_shift = (
        int(np.max(combined_exponent))
        if shift_exponent is None
        else shift_exponent
    )
    scaled_exponent = combined_exponent - chosen_shift
    try:
        with np.errstate(over="raise", invalid="raise"):
            products[positive_mask] = np.ldexp(
                value_mantissa * width_mantissa,
                scaled_exponent,
            )
    except FloatingPointError as error:
        raise TransportMetricError(
            "scaled product",
            "contains non-finite derived values",
        ) from error
    underflow_mask = np.zeros(values.shape, dtype=bool)
    underflow_mask[positive_mask] = products[positive_mask] == 0.0
    return _ScaledProducts(
        values=products,
        shift_exponent=chosen_shift,
        positive_mask=positive_mask,
        underflow_mask=underflow_mask,
    )


def _positive_probability(
    path: str,
    axis: np.ndarray,
    intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    widths = _trapezoid_node_widths(path, axis)
    positive_intensity = np.maximum(intensity, 0.0)
    scaled_mass = _scaled_nonnegative_products(
        positive_intensity,
        widths,
    )
    if np.any(scaled_mass.underflow_mask):
        raise TransportMetricError(
            f"{path} positive mass",
            "positive mass underflowed during normalization",
        )
    total_mass = math.fsum(float(value) for value in scaled_mass.values)
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise TransportMetricError(
            f"{path} positive mass",
            "must be positive and finite",
        )
    try:
        with np.errstate(divide="raise", invalid="raise"):
            probability = np.asarray(
                scaled_mass.values / total_mass,
                dtype=np.float64,
            )
    except FloatingPointError as error:
        raise TransportMetricError(
            f"{path} positive mass",
            "must be positive and finite",
        ) from error
    if not np.isfinite(probability).all():
        raise TransportMetricError(
            f"{path} positive mass",
            "must be positive and finite",
        )
    probability_zero_mask = np.logical_and(
        scaled_mass.values > 0.0,
        probability == 0.0,
    )
    if np.any(probability_zero_mask):
        raise TransportMetricError(
            f"{path} positive mass",
            "positive mass underflowed during normalization",
        )
    probability_sum = math.fsum(float(value) for value in probability)
    if not math.isfinite(probability_sum) or probability_sum <= 0.0:
        raise TransportMetricError(
            f"{path} positive mass",
            "must be positive and finite",
        )
    probability = np.asarray(
        probability / probability_sum,
        dtype=np.float64,
    )
    if not np.isfinite(probability).all():
        raise TransportMetricError(
            f"{path} positive mass",
            "must be positive and finite",
        )
    renormalized_zero_mask = np.logical_and(
        scaled_mass.values > 0.0,
        probability == 0.0,
    )
    if np.any(renormalized_zero_mask):
        raise TransportMetricError(
            f"{path} positive mass",
            "positive mass underflowed during normalization",
        )
    return probability, widths


def _negative_area_fraction(
    path: str,
    widths: np.ndarray,
    intensity: np.ndarray,
) -> float:
    negative_intensity = np.maximum(-intensity, 0.0)
    absolute_intensity = np.abs(intensity)
    scaled_absolute = _scaled_nonnegative_products(
        absolute_intensity,
        widths,
    )
    if np.any(scaled_absolute.underflow_mask):
        raise TransportMetricError(
            f"{path} negative area fraction",
            "weighted contribution underflowed during scaling",
        )
    absolute_area = math.fsum(float(value) for value in scaled_absolute.values)
    if not math.isfinite(absolute_area) or absolute_area <= 0.0:
        raise TransportMetricError(
            f"{path} absolute area",
            "must be positive and finite",
        )
    scaled_negative = _scaled_nonnegative_products(
        negative_intensity,
        widths,
        shift_exponent=scaled_absolute.shift_exponent,
    )
    if np.any(scaled_negative.underflow_mask):
        raise TransportMetricError(
            f"{path} negative area fraction",
            "weighted contribution underflowed during scaling",
        )
    negative_area = math.fsum(float(value) for value in scaled_negative.values)
    fraction = negative_area / absolute_area
    if not math.isfinite(fraction):
        raise TransportMetricError(
            f"{path} negative area fraction",
            "produced non-finite value",
        )
    return float(min(max(fraction, 0.0), 1.0))


def _cdf_at_left_edges(
    path: str,
    axis: np.ndarray,
    probability: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    cumulative = np.cumsum(probability, dtype=np.float64)
    if not np.isfinite(cumulative).all():
        raise TransportMetricError(
            f"{path} cdf",
            "contains non-finite derived values",
        )
    previous = np.concatenate(
        (np.array([0.0], dtype=np.float64), cumulative[:-1])
    )
    lost_probability_mask = np.logical_and(
        probability > 0.0,
        cumulative == previous,
    )
    if np.any(lost_probability_mask):
        raise TransportMetricError(
            f"{path} cdf",
            "positive probability lost during accumulation",
        )
    cumulative = np.clip(cumulative, 0.0, 1.0)
    cumulative[-1] = 1.0
    indices = np.searchsorted(axis, edges, side="right") - 1
    cdf = np.zeros(edges.shape, dtype=np.float64)
    populated = indices >= 0
    cdf[populated] = cumulative[indices[populated]]
    if not np.isfinite(cdf).all():
        raise TransportMetricError(
            f"{path} cdf",
            "contains non-finite derived values",
        )
    return np.clip(cdf, 0.0, 1.0)


@dataclass(frozen=True, init=False)
class Wasserstein1Metric:
    metric_id: str = "wasserstein_1_cm1"
    output_id: str = "wasserstein_1_cm1"
    unit: str = "cm^-1"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SPECTRUM_PAIR,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.INDEPENDENT_PHYSICAL_AXES,
        init=False,
    )
    preferred_direction: PreferredDirection = field(
        default=PreferredDirection.LOWER_IS_BETTER,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = _require_spectrum_pair(request)
        reference_probability, reference_widths = _positive_probability(
            "reference",
            typed_request.reference.axis_cm1,
            typed_request.reference.intensity,
        )
        candidate_probability, candidate_widths = _positive_probability(
            "candidate",
            typed_request.candidate.axis_cm1,
            typed_request.candidate.intensity,
        )
        reference_negative_area_fraction = _negative_area_fraction(
            "reference",
            reference_widths,
            typed_request.reference.intensity,
        )
        candidate_negative_area_fraction = _negative_area_fraction(
            "candidate",
            candidate_widths,
            typed_request.candidate.intensity,
        )
        reference_positive_intensity = np.maximum(
            typed_request.reference.intensity,
            0.0,
        )
        candidate_positive_intensity = np.maximum(
            typed_request.candidate.intensity,
            0.0,
        )

        supports = np.unique(
            np.concatenate(
                (
                    typed_request.reference.axis_cm1,
                    typed_request.candidate.axis_cm1,
                )
            )
        )
        identical_native_positive_mass_measure = bool(
            np.array_equal(
                typed_request.reference.axis_cm1,
                typed_request.candidate.axis_cm1,
            )
            and reference_positive_intensity.shape
            == candidate_positive_intensity.shape
            and np.array_equal(
                reference_positive_intensity,
                candidate_positive_intensity,
            )
        )
        if identical_native_positive_mass_measure:
            value = 0.0
        else:
            support_intervals = _stable_positive_intervals("support", supports)
            reference_cdf = _cdf_at_left_edges(
                "reference",
                typed_request.reference.axis_cm1,
                reference_probability,
                supports[:-1],
            )
            candidate_cdf = _cdf_at_left_edges(
                "candidate",
                typed_request.candidate.axis_cm1,
                candidate_probability,
                supports[:-1],
            )
            try:
                with np.errstate(over="raise", invalid="raise"):
                    interval_terms = np.abs(reference_cdf - candidate_cdf) * support_intervals
            except FloatingPointError as error:
                raise TransportMetricError(
                    self.output_id,
                    "produced non-finite value",
                ) from error
            if not np.isfinite(interval_terms).all():
                raise TransportMetricError(
                    self.output_id,
                    "produced non-finite value",
                )
            try:
                value = math.fsum(float(term) for term in interval_terms)
            except OverflowError as error:
                raise TransportMetricError(
                    self.output_id,
                    "produced non-finite value",
                ) from error
            if not math.isfinite(value):
                raise TransportMetricError(
                    self.output_id,
                    "produced non-finite value",
                )

        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    output_id=self.output_id,
                    value=value,
                    unit=self.unit,
                    preferred_direction=self.preferred_direction,
                    target_value=None,
                ),
            ),
            diagnostics={
                "axis_equal": bool(
                    np.array_equal(
                        typed_request.reference.axis_cm1,
                        typed_request.candidate.axis_cm1,
                    )
                ),
                "candidate_negative_area_fraction": candidate_negative_area_fraction,
                "candidate_point_count": int(
                    typed_request.candidate.axis_cm1.size
                ),
                "mass_construction": "positive_part_trapezoid_node_width",
                "mass_normalized": True,
                "reference_negative_area_fraction": reference_negative_area_fraction,
                "reference_point_count": int(
                    typed_request.reference.axis_cm1.size
                ),
            },
        )


__all__ = [
    "TransportMetricError",
    "Wasserstein1Metric",
]
