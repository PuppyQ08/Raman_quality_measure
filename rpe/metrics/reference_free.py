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
    SingleSpectrumInput,
)


GAUSSIAN_MAD_CONSTANT = 0.6744897501960817


class ISLikeMetricError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _require_single_spectrum_request(
    request: MetricInput,
) -> SingleSpectrumInput:
    if not isinstance(request, SingleSpectrumInput):
        raise ISLikeMetricError(
            "request",
            "must be SingleSpectrumInput",
        )
    return request


def _require_finite_scalar(path: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ISLikeMetricError(path, "produced non-finite value")
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
        raise ISLikeMetricError(path, "produced non-finite scaled values")
    lost = (values != 0.0) & (scaled == 0.0)
    if np.any(lost):
        raise ISLikeMetricError(
            path,
            "required nonzero value became zero after scaling",
        )
    return scaled


def _mean(values: np.ndarray, *, path: str) -> float:
    mean = math.fsum(float(value) for value in values) / values.size
    return _require_finite_scalar(path, mean)


def _checked_difference(path: str, left: float, right: float) -> float:
    difference = float(left - right)
    if not math.isfinite(difference):
        raise ISLikeMetricError(path, "produced non-finite difference")
    if left != right and difference == 0.0:
        raise ISLikeMetricError(
            path,
            "required nonzero difference became zero",
        )
    return difference


def _stable_centered_rms_scaled(values: np.ndarray) -> tuple[float, int]:
    exponent = _scale_exponent(values)
    scaled_values = _scale_values("intensity", values, exponent)
    scaled_mean = _mean(scaled_values, path="mean intensity")
    centered = np.asarray(
        [
            _checked_difference("centered intensity", float(value), scaled_mean)
            for value in scaled_values
        ],
        dtype=np.float64,
    )
    if not np.isfinite(centered).all():
        raise ISLikeMetricError(
            "centered intensity",
            "produced non-finite values",
        )
    centered_square_sum = math.fsum(
        float(value) * float(value) for value in centered
    )
    signal_rms_scaled = _require_finite_scalar(
        "signal_rms_scaled",
        math.sqrt(centered_square_sum / centered.size),
    )
    if signal_rms_scaled <= 0.0:
        raise ISLikeMetricError(
            "signal_rms",
            "must be positive and finite",
        )
    return signal_rms_scaled, exponent


def _stable_noise_sigma_scaled(values: np.ndarray, exponent: int) -> float:
    scaled_values = _scale_values("intensity", values, exponent)
    scaled_differences = np.asarray(
        [
            _checked_difference(
                "differences",
                float(right),
                float(left),
            )
            for left, right in zip(scaled_values, scaled_values[1:])
        ],
        dtype=np.float64,
    )
    if not np.isfinite(scaled_differences).all():
        raise ISLikeMetricError(
            "differences",
            "produced non-finite values",
        )
    median_difference = _require_finite_scalar(
        "median difference",
        float(np.median(scaled_differences)),
    )
    absolute_deviations = np.asarray(
        [
            abs(
                _checked_difference(
                    "absolute deviation",
                    float(value),
                    median_difference,
                )
            )
            for value in scaled_differences
        ],
        dtype=np.float64,
    )
    mad_scaled = _require_finite_scalar(
        "mad_scaled",
        float(np.median(absolute_deviations)),
    )
    noise_sigma_scaled = _require_finite_scalar(
        "noise_sigma_scaled",
        mad_scaled / GAUSSIAN_MAD_CONSTANT / math.sqrt(2.0),
    )
    if noise_sigma_scaled <= 0.0:
        raise ISLikeMetricError(
            "noise_sigma",
            "must be positive and finite",
        )
    return noise_sigma_scaled


@dataclass(frozen=True, init=False)
class ISLikeStructureToNoiseMetric:
    metric_id: str = "is_like_structure_to_noise"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SINGLE_SPECTRUM,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.SINGLE_AXIS,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = _require_single_spectrum_request(request)
        intensity = np.asarray(
            typed_request.spectrum.intensity,
            dtype=np.float64,
        )
        point_count = int(intensity.size)
        if point_count < 3:
            raise ISLikeMetricError(
                "point_count",
                "must be at least 3",
            )

        signal_rms_scaled, exponent = _stable_centered_rms_scaled(intensity)
        noise_sigma_scaled = _stable_noise_sigma_scaled(intensity, exponent)
        score = _require_finite_scalar(
            "score",
            signal_rms_scaled / noise_sigma_scaled,
        )
        if score <= 0.0:
            raise ISLikeMetricError(
                "score",
                "must be positive and finite",
            )

        diagnostics = {
            "definition_boundary": "classical_is_like_not_learned_score",
            "gaussian_mad_constant": GAUSSIAN_MAD_CONSTANT,
            "intensity_scale_power2_exponent": exponent,
            "noise_estimator": "first_difference_mad_gaussian_sigma",
            "noise_sigma_scaled": noise_sigma_scaled,
            "point_count": point_count,
            "signal_rms_scaled": signal_rms_scaled,
        }

        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    output_id="is_like_structure_to_noise",
                    value=score,
                    unit="ratio",
                    preferred_direction=PreferredDirection.HIGHER_IS_BETTER,
                    target_value=None,
                ),
            ),
            diagnostics=diagnostics,
        )
