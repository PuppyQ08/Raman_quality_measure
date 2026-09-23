from __future__ import annotations

import math
from dataclasses import dataclass, field

from rpe.evaluation import (
    AxisPolicy,
    CurveMetricOutput,
    CurveSeries,
    MetricInput,
    MetricInputKind,
    MetricResult,
    Peak1D,
    PeakPairInput,
    PreferredDirection,
    ScalarMetricOutput,
)


class PeakMetricError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class _PeakCounts:
    reference_count: int
    candidate_count: int
    true_positive_count: int


@dataclass(frozen=True)
class _MatchedPair:
    reference_index: int
    candidate_index: int
    reference_peak: Peak1D
    candidate_peak: Peak1D
    distance_cm1: float


def _require_peak_request(request: MetricInput) -> PeakPairInput:
    if not isinstance(request, PeakPairInput):
        raise PeakMetricError(
            "request",
            "must be PeakPairInput",
        )
    return request


def _require_finite_value(path: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise PeakMetricError(path, "produced non-finite value")
    return scalar


def _match_peaks(
    reference_peaks: tuple[Peak1D, ...],
    candidate_peaks: tuple[Peak1D, ...],
    *,
    tolerance_cm1: float,
) -> tuple[_MatchedPair, ...]:
    valid_pairs: list[_MatchedPair] = []
    for reference_index, reference_peak in enumerate(reference_peaks):
        for candidate_index, candidate_peak in enumerate(candidate_peaks):
            distance = abs(
                candidate_peak.position_cm1 - reference_peak.position_cm1
            )
            if distance <= tolerance_cm1:
                valid_pairs.append(
                    _MatchedPair(
                        reference_index=reference_index,
                        candidate_index=candidate_index,
                        reference_peak=reference_peak,
                        candidate_peak=candidate_peak,
                        distance_cm1=distance,
                    )
                )
    valid_pairs.sort(
        key=lambda pair: (
            pair.distance_cm1,
            pair.reference_peak.position_cm1,
            pair.candidate_peak.position_cm1,
        )
    )
    consumed_reference: set[int] = set()
    consumed_candidate: set[int] = set()
    accepted: list[_MatchedPair] = []
    for pair in valid_pairs:
        if pair.reference_index in consumed_reference:
            continue
        if pair.candidate_index in consumed_candidate:
            continue
        consumed_reference.add(pair.reference_index)
        consumed_candidate.add(pair.candidate_index)
        accepted.append(pair)
    return tuple(accepted)


def _counts(
    reference_peaks: tuple[Peak1D, ...],
    candidate_peaks: tuple[Peak1D, ...],
    *,
    tolerance_cm1: float,
) -> _PeakCounts:
    matches = _match_peaks(
        reference_peaks,
        candidate_peaks,
        tolerance_cm1=tolerance_cm1,
    )
    return _PeakCounts(
        reference_count=len(reference_peaks),
        candidate_count=len(candidate_peaks),
        true_positive_count=len(matches),
    )


def _precision(counts: _PeakCounts) -> float:
    if counts.candidate_count == 0:
        return 1.0
    return counts.true_positive_count / counts.candidate_count


def _recall(counts: _PeakCounts) -> float:
    if counts.reference_count == 0:
        return 1.0
    return counts.true_positive_count / counts.reference_count


def _f1(counts: _PeakCounts) -> float:
    denominator = counts.reference_count + counts.candidate_count
    if denominator == 0:
        return 1.0
    return (2.0 * counts.true_positive_count) / denominator


def _artifact_peak_ratio(counts: _PeakCounts) -> float:
    if counts.candidate_count == 0:
        return 0.0
    false_positive_count = counts.candidate_count - counts.true_positive_count
    return false_positive_count / counts.candidate_count


def _missing_peak_ratio(counts: _PeakCounts) -> float:
    if counts.reference_count == 0:
        return 0.0
    false_negative_count = counts.reference_count - counts.true_positive_count
    return false_negative_count / counts.reference_count


def _classification_values(counts: _PeakCounts) -> tuple[float, float, float, float, float]:
    return (
        _precision(counts),
        _recall(counts),
        _f1(counts),
        _artifact_peak_ratio(counts),
        _missing_peak_ratio(counts),
    )


def _prominence(
    peaks: tuple[Peak1D, ...],
    *,
    side_name: str,
) -> tuple[float, ...]:
    prominences: list[float] = []
    for index, peak in enumerate(peaks):
        if peak.prominence is None:
            raise PeakMetricError(
                f"{side_name}[{index}].prominence",
                "must be present for prominence scan",
            )
        prominences.append(peak.prominence)
    return tuple(prominences)


def _filter_by_prominence(
    peaks: tuple[Peak1D, ...],
    prominences: tuple[float, ...],
    *,
    threshold: float,
) -> tuple[Peak1D, ...]:
    return tuple(
        peak
        for peak, prominence in zip(peaks, prominences)
        if prominence >= threshold
    )


def _mean_abs(values: tuple[float, ...], *, path: str) -> float:
    if not values:
        raise PeakMetricError(path, "must contain at least one matched pair")
    absolute_values: list[float] = []
    for value in values:
        absolute = abs(value)
        if not math.isfinite(absolute):
            raise PeakMetricError(path, "produced non-finite value")
        absolute_values.append(absolute)
    mean = math.fsum(absolute_values) / len(absolute_values)
    return _require_finite_value(path, mean)


def _required_matched_property(
    pair: _MatchedPair,
    *,
    side_name: str,
    property_name: str,
) -> float:
    peak = pair.reference_peak if side_name == "reference_peaks" else pair.candidate_peak
    value = getattr(peak, property_name)
    if value is None:
        raise PeakMetricError(
            f"{side_name}[{pair.reference_index if side_name == 'reference_peaks' else pair.candidate_index}].{property_name}",
            "must be present for matched error evaluation",
        )
    return value


@dataclass(frozen=True, init=False)
class PeakDetectionCurvesMetric:
    metric_id: str = "peak_detection_curves"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.PEAK_PAIR,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.NOT_APPLICABLE,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = _require_peak_request(request)
        reference_prominences = _prominence(
            typed_request.reference_peaks,
            side_name="reference_peaks",
        )
        candidate_prominences = _prominence(
            typed_request.candidate_peaks,
            side_name="candidate_peaks",
        )

        request_counts = _counts(
            typed_request.reference_peaks,
            typed_request.candidate_peaks,
            tolerance_cm1=typed_request.position_tolerance_cm1,
        )
        request_values = _classification_values(request_counts)

        tolerance_scan_counts = tuple(
            _counts(
                typed_request.reference_peaks,
                typed_request.candidate_peaks,
                tolerance_cm1=tolerance_cm1,
            )
            for tolerance_cm1 in (1.0, 2.0, 4.0, 8.0)
        )
        prominence_scan_counts = tuple(
            _counts(
                _filter_by_prominence(
                    typed_request.reference_peaks,
                    reference_prominences,
                    threshold=threshold,
                ),
                _filter_by_prominence(
                    typed_request.candidate_peaks,
                    candidate_prominences,
                    threshold=threshold,
                ),
                tolerance_cm1=typed_request.position_tolerance_cm1,
            )
            for threshold in typed_request.prominence_thresholds
        )

        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    output_id="precision",
                    value=request_values[0],
                    unit="ratio",
                    preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                    target_value=None,
                ),
                ScalarMetricOutput(
                    output_id="recall",
                    value=request_values[1],
                    unit="ratio",
                    preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                    target_value=None,
                ),
                ScalarMetricOutput(
                    output_id="f1",
                    value=request_values[2],
                    unit="ratio",
                    preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                    target_value=None,
                ),
                ScalarMetricOutput(
                    output_id="artifact_peak_ratio",
                    value=request_values[3],
                    unit="ratio",
                    preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                    target_value=None,
                ),
                ScalarMetricOutput(
                    output_id="missing_peak_ratio",
                    value=request_values[4],
                    unit="ratio",
                    preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                    target_value=None,
                ),
                CurveMetricOutput(
                    output_id="tolerance_scan",
                    x_name="position_tolerance",
                    x_values=(1.0, 2.0, 4.0, 8.0),
                    x_unit="cm^-1",
                    series=(
                        CurveSeries(
                            series_id="precision",
                            values=tuple(
                                _precision(counts)
                                for counts in tolerance_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                            target_value=None,
                        ),
                        CurveSeries(
                            series_id="recall",
                            values=tuple(
                                _recall(counts)
                                for counts in tolerance_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                            target_value=None,
                        ),
                        CurveSeries(
                            series_id="f1",
                            values=tuple(
                                _f1(counts)
                                for counts in tolerance_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                            target_value=None,
                        ),
                        CurveSeries(
                            series_id="artifact_peak_ratio",
                            values=tuple(
                                _artifact_peak_ratio(counts)
                                for counts in tolerance_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                            target_value=None,
                        ),
                        CurveSeries(
                            series_id="missing_peak_ratio",
                            values=tuple(
                                _missing_peak_ratio(counts)
                                for counts in tolerance_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                            target_value=None,
                        ),
                    ),
                ),
                CurveMetricOutput(
                    output_id="prominence_scan",
                    x_name="prominence_threshold",
                    x_values=typed_request.prominence_thresholds,
                    x_unit="intensity",
                    series=(
                        CurveSeries(
                            series_id="precision",
                            values=tuple(
                                _precision(counts)
                                for counts in prominence_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                            target_value=None,
                        ),
                        CurveSeries(
                            series_id="recall",
                            values=tuple(
                                _recall(counts)
                                for counts in prominence_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                            target_value=None,
                        ),
                        CurveSeries(
                            series_id="f1",
                            values=tuple(
                                _f1(counts)
                                for counts in prominence_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                            target_value=None,
                        ),
                        CurveSeries(
                            series_id="artifact_peak_ratio",
                            values=tuple(
                                _artifact_peak_ratio(counts)
                                for counts in prominence_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                            target_value=None,
                        ),
                        CurveSeries(
                            series_id="missing_peak_ratio",
                            values=tuple(
                                _missing_peak_ratio(counts)
                                for counts in prominence_scan_counts
                            ),
                            unit="ratio",
                            preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                            target_value=None,
                        ),
                    ),
                ),
            ),
            diagnostics={
                "matching_policy": (
                    "distance_sorted_greedy_reference_then_candidate"
                ),
                "tolerance_boundary": "inclusive",
                "position_tolerance_cm1": typed_request.position_tolerance_cm1,
                "tolerance_scan_grid_cm1": (1, 2, 4, 8),
                "prominence_filter": (
                    "both_sides_inclusive_requires_present"
                ),
                "reference_peak_count": len(typed_request.reference_peaks),
                "candidate_peak_count": len(typed_request.candidate_peaks),
                "matched_pair_count_at_request_tolerance": (
                    request_counts.true_positive_count
                ),
            },
        )


@dataclass(frozen=True, init=False)
class MatchedPeakErrorMetric:
    metric_id: str = "matched_peak_errors"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.PEAK_PAIR,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.NOT_APPLICABLE,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = _require_peak_request(request)
        matches = _match_peaks(
            typed_request.reference_peaks,
            typed_request.candidate_peaks,
            tolerance_cm1=typed_request.position_tolerance_cm1,
        )
        if not matches:
            raise PeakMetricError(
                "matches",
                "must contain at least one matched pair",
            )

        position_errors = tuple(
            pair.candidate_peak.position_cm1 - pair.reference_peak.position_cm1
            for pair in matches
        )

        relative_height_errors: list[float] = []
        fwhm_errors: list[float] = []
        area_errors: list[float] = []
        for pair in matches:
            reference_height = _required_matched_property(
                pair,
                side_name="reference_peaks",
                property_name="height",
            )
            candidate_height = _required_matched_property(
                pair,
                side_name="candidate_peaks",
                property_name="height",
            )
            reference_fwhm = _required_matched_property(
                pair,
                side_name="reference_peaks",
                property_name="fwhm_cm1",
            )
            candidate_fwhm = _required_matched_property(
                pair,
                side_name="candidate_peaks",
                property_name="fwhm_cm1",
            )
            reference_area = _required_matched_property(
                pair,
                side_name="reference_peaks",
                property_name="area",
            )
            candidate_area = _required_matched_property(
                pair,
                side_name="candidate_peaks",
                property_name="area",
            )
            if reference_height == 0.0:
                raise PeakMetricError(
                    f"reference_peaks[{pair.reference_index}].height",
                    "must be nonzero for relative height error",
                )

            relative_height_errors.append(
                _require_finite_value(
                    "relative_peak_height_mae",
                    (candidate_height - reference_height)
                    / abs(reference_height),
                )
            )
            fwhm_errors.append(
                _require_finite_value(
                    "peak_fwhm_mae_cm1",
                    candidate_fwhm - reference_fwhm,
                )
            )
            area_errors.append(
                _require_finite_value(
                    "peak_area_mae",
                    candidate_area - reference_area,
                )
            )

        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    output_id="peak_position_mae_cm1",
                    value=_mean_abs(
                        position_errors,
                        path="peak_position_mae_cm1",
                    ),
                    unit="cm^-1",
                    preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                    target_value=None,
                ),
                ScalarMetricOutput(
                    output_id="relative_peak_height_mae",
                    value=_mean_abs(
                        tuple(relative_height_errors),
                        path="relative_peak_height_mae",
                    ),
                    unit="ratio",
                    preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                    target_value=None,
                ),
                ScalarMetricOutput(
                    output_id="peak_fwhm_mae_cm1",
                    value=_mean_abs(
                        tuple(fwhm_errors),
                        path="peak_fwhm_mae_cm1",
                    ),
                    unit="cm^-1",
                    preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                    target_value=None,
                ),
                ScalarMetricOutput(
                    output_id="peak_area_mae",
                    value=_mean_abs(
                        tuple(area_errors),
                        path="peak_area_mae",
                    ),
                    unit="intensity_cm^-1",
                    preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                    target_value=None,
                ),
            ),
            diagnostics={
                "matching_policy": (
                    "distance_sorted_greedy_reference_then_candidate"
                ),
                "tolerance_boundary": "inclusive",
                "position_tolerance_cm1": typed_request.position_tolerance_cm1,
                "reference_peak_count": len(typed_request.reference_peaks),
                "candidate_peak_count": len(typed_request.candidate_peaks),
                "matched_pair_count_at_request_tolerance": len(matches),
                "evaluated_pair_count": len(matches),
            },
        )


__all__ = [
    "MatchedPeakErrorMetric",
    "PeakDetectionCurvesMetric",
    "PeakMetricError",
]
