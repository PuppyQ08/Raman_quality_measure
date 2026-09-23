from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rpe.evaluation import (
    AxisPolicy,
    CalibrationInput,
    MetricInput,
    MetricInputKind,
    MetricResult,
    PreferredDirection,
    ScalarMetricOutput,
)


class AnalyticalMetricError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _finite_vector(path: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise AnalyticalMetricError(path, "must be one-dimensional")
    if not np.isfinite(array).all():
        raise AnalyticalMetricError(path, "contains non-finite values")
    return array


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _positive_finite_scalar(path: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise AnalyticalMetricError(path, "must be positive and finite")
    return scalar


def _finite_scalar(path: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise AnalyticalMetricError(path, "must be finite")
    return scalar


def _stable_sample_sd(values: np.ndarray) -> float:
    scale = _max_abs(values)
    if scale == 0.0:
        return 0.0
    scaled = values / scale
    centered = scaled - float(np.mean(scaled))
    variance = float(np.dot(centered, centered) / (values.size - 1))
    sd = scale * float(np.sqrt(variance))
    return _finite_scalar("blank sample sd", sd)


def _stable_ols_slope_intercept(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
) -> tuple[float, float]:
    x_scale = _max_abs(true_values)
    y_scale = _max_abs(predicted_values)
    if x_scale == 0.0:
        raise AnalyticalMetricError(
            "true concentrations",
            "must contain at least two distinct values",
        )
    if y_scale == 0.0:
        return 0.0, 0.0

    x_scaled = true_values / x_scale
    y_scaled = predicted_values / y_scale
    x_centered = x_scaled - float(np.mean(x_scaled))
    y_centered = y_scaled - float(np.mean(y_scaled))
    x_var = float(np.dot(x_centered, x_centered))
    if x_var == 0.0:
        raise AnalyticalMetricError(
            "true concentrations",
            "must contain at least two distinct values",
        )

    covariance = float(np.dot(x_centered, y_centered))
    slope = (y_scale / x_scale) * (covariance / x_var)
    intercept = float(np.mean(predicted_values)) - slope * float(
        np.mean(true_values)
    )
    return (
        _finite_scalar("ols slope", slope),
        _finite_scalar("ols intercept", intercept),
    )


@dataclass(frozen=True, init=False)
class TechnicalRepeatabilityLodLoqMetric:
    metric_id: str = "technical_repeatability_lod_loq"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.CALIBRATION,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.NOT_APPLICABLE,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        if not isinstance(request, CalibrationInput):
            raise AnalyticalMetricError(
                "request",
                "must be CalibrationInput",
            )
        if len(request.target_names) != 1:
            raise AnalyticalMetricError(
                "target_names",
                "must contain exactly one target",
            )
        if request.true_concentrations.shape[0] < 2:
            raise AnalyticalMetricError(
                "training rows",
                "must be at least 2",
            )

        true_values = _finite_vector(
            "true concentrations",
            request.true_concentrations[:, 0],
        )
        predicted_values = _finite_vector(
            "predicted concentrations",
            request.predicted_concentrations[:, 0],
        )
        blank_values = _finite_vector(
            "blank predictions",
            request.blank_predictions[:, 0],
        )
        if np.unique(true_values).size < 2:
            raise AnalyticalMetricError(
                "true concentrations",
                "must contain at least two distinct values",
            )

        slope, intercept = _stable_ols_slope_intercept(
            true_values,
            predicted_values,
        )
        slope = _positive_finite_scalar("ols slope", slope)
        sigma = _positive_finite_scalar(
            "blank sample sd",
            _stable_sample_sd(blank_values),
        )

        ich_lod = 3.3 * sigma / slope
        ich_loq = 10.0 * sigma / slope
        iupac_lod = 3.0 * sigma / slope
        derived = {
            "ich_lod": float(ich_lod),
            "ich_loq": float(ich_loq),
            "iupac_lod": float(iupac_lod),
        }
        if not all(math.isfinite(value) for value in derived.values()):
            raise AnalyticalMetricError(
                "derived outputs",
                "must be finite",
            )

        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=tuple(
                ScalarMetricOutput(
                    output_id=output_id,
                    value=value,
                    unit="input_concentration_unit",
                    preferred_direction=PreferredDirection.LOWER_IS_BETTER,
                    target_value=None,
                )
                for output_id, value in derived.items()
            ),
            diagnostics={
                "target_name": request.target_names[0],
                "training_row_count": int(true_values.size),
                "blank_repeat_count": int(blank_values.size),
                "ols_slope": slope,
                "ols_intercept": intercept,
                "blank_sample_sd": sigma,
                "sigma_convention": "sample_sd_ddof_1",
                "slope_convention": "ols_predicted_on_true",
                "claim_boundary": (
                    "supplied_prediction_technical_repeatability"
                ),
            },
        )


__all__ = [
    "AnalyticalMetricError",
    "TechnicalRepeatabilityLodLoqMetric",
]
