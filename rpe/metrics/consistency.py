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
    ReplicatePairInput,
    ScalarMetricOutput,
)


class ConsistencyMetricError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _require_finite_scalar(path: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ConsistencyMetricError(path, "produced non-finite value")
    return scalar


def _stable_centered_unit(path: str, values: np.ndarray) -> np.ndarray:
    max_abs = _max_abs(values)
    if max_abs == 0.0:
        raise ConsistencyMetricError(path, "zero variance")
    _, exponent = math.frexp(max_abs)
    scale = math.ldexp(1.0, exponent - 1)
    scaled_values = values / scale
    anchored = scaled_values - float(scaled_values[0])
    if not np.isfinite(anchored).all():
        raise ConsistencyMetricError(path, "zero variance")
    anchored_scale = _max_abs(anchored)
    if anchored_scale == 0.0:
        raise ConsistencyMetricError(path, "zero variance")
    scaled = anchored / anchored_scale
    centered = scaled - float(np.mean(scaled))
    centered_scale = _max_abs(centered)
    if centered_scale == 0.0:
        raise ConsistencyMetricError(path, "zero variance")
    centered = centered / centered_scale
    norm = float(np.linalg.norm(centered))
    if norm == 0.0 or not math.isfinite(norm):
        raise ConsistencyMetricError(path, "zero variance")
    return centered / norm


@dataclass(frozen=True, init=False)
class HalfSplitPearsonConsistencyMetric:
    metric_id: str = "half_split_pearson_consistency"
    output_id: str = "half_split_pearson_consistency"
    unit: str = "correlation"
    input_kind: MetricInputKind = field(
        default=MetricInputKind.REPLICATE_PAIR,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.EXACT_AXIS,
        init=False,
    )
    preferred_direction: PreferredDirection = field(
        default=PreferredDirection.HIGHER_IS_BETTER,
        init=False,
    )

    def evaluate(self, request: MetricInput) -> MetricResult:
        if not isinstance(request, ReplicatePairInput):
            raise ConsistencyMetricError(
                "request",
                "must be ReplicatePairInput",
            )
        left = _stable_centered_unit(
            "left zero variance",
            request.left.intensity,
        )
        right = _stable_centered_unit(
            "right zero variance",
            request.right.intensity,
        )
        value = _require_finite_scalar(
            self.output_id,
            float(np.clip(np.dot(left, right), -1.0, 1.0)),
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
                        request.left.axis_cm1,
                        request.right.axis_cm1,
                    )
                ),
                "point_count": int(request.left.intensity.size),
                "left_spectrum_id": request.left.spectrum_id,
                "right_spectrum_id": request.right.spectrum_id,
                "consistency_definition": "exact_axis_population_pearson_r",
                "affine_invariant": True,
            },
        )
