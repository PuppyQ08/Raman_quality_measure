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


class FidelityMetricError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _float64_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise FidelityMetricError(name, "must be one-dimensional")
    if not np.isfinite(array).all():
        raise FidelityMetricError(name, "contains non-finite values")
    return array


def _max_abs(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def _require_finite_scalar(path: str, value: float) -> float:
    scalar = float(value)
    if not math.isfinite(scalar):
        raise FidelityMetricError(path, "produced non-finite value")
    return scalar


def _stable_mean_square(values: np.ndarray) -> float:
    scale = _max_abs(values)
    if scale == 0.0:
        return 0.0
    scaled = values / scale
    mean_square = scale * scale * float(np.mean(scaled * scaled))
    return _require_finite_scalar("mean square", mean_square)


def _stable_mean_abs(values: np.ndarray) -> float:
    scale = _max_abs(values)
    if scale == 0.0:
        return 0.0
    mean_abs = scale * float(np.mean(np.abs(values) / scale))
    return _require_finite_scalar("mean absolute", mean_abs)


def _stable_rms(values: np.ndarray) -> float:
    scale = _max_abs(values)
    if scale == 0.0:
        return 0.0
    scaled = values / scale
    rms = scale * float(np.sqrt(np.mean(scaled * scaled)))
    return _require_finite_scalar("root mean square", rms)


def _stable_centered_unit(path: str, values: np.ndarray) -> np.ndarray:
    max_abs = _max_abs(values)
    if max_abs == 0.0:
        raise FidelityMetricError(path, "zero variance")
    _, exponent = math.frexp(max_abs)
    scale = math.ldexp(1.0, exponent - 1)
    scaled_values = values / scale
    anchored = scaled_values - float(scaled_values[0])
    if not np.isfinite(anchored).all():
        raise FidelityMetricError(path, "zero variance")
    anchored_scale = _max_abs(anchored)
    if anchored_scale == 0.0:
        raise FidelityMetricError(path, "zero variance")
    scaled = anchored / anchored_scale
    centered = scaled - float(np.mean(scaled))
    centered_scale = _max_abs(centered)
    if centered_scale == 0.0:
        raise FidelityMetricError(path, "zero variance")
    centered = centered / centered_scale
    norm = float(np.linalg.norm(centered))
    if norm == 0.0 or not math.isfinite(norm):
        raise FidelityMetricError(path, "zero variance")
    return centered / norm


@dataclass(frozen=True, init=False)
class _SpectrumPairMetric:
    metric_id: str = ""
    output_id: str = ""
    unit: str = ""
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SPECTRUM_PAIR,
        init=False,
    )
    axis_policy: AxisPolicy = field(
        default=AxisPolicy.INDEX_ALIGNED,
        init=False,
    )
    preferred_direction: PreferredDirection = field(
        default=PreferredDirection.LOWER_IS_BETTER,
        init=False,
    )

    def _arrays(self, request: MetricInput) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(request, SpectrumPairInput):
            raise FidelityMetricError(
                "request",
                "must be SpectrumPairInput",
            )
        if request.reference.intensity.shape != request.candidate.intensity.shape:
            raise FidelityMetricError(
                "point count",
                "reference and candidate must align by index",
            )
        reference = _float64_array("reference intensity", request.reference.intensity)
        candidate = _float64_array("candidate intensity", request.candidate.intensity)
        return reference, candidate

    def _difference(self, request: MetricInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference, candidate = self._arrays(request)
        difference = np.asarray(candidate - reference, dtype=np.float64)
        if not np.isfinite(difference).all():
            raise FidelityMetricError(
                "difference",
                "contains non-finite values",
            )
        return reference, candidate, difference

    def _result(
        self,
        request: SpectrumPairInput,
        value: float,
    ) -> MetricResult:
        return MetricResult(
            metric_id=self.metric_id,
            input_kind=self.input_kind,
            outputs=(
                ScalarMetricOutput(
                    output_id=self.output_id,
                    value=_require_finite_scalar(self.output_id, value),
                    unit=self.unit,
                    preferred_direction=self.preferred_direction,
                    target_value=None,
                ),
            ),
            diagnostics={
                "axis_equal": bool(
                    np.array_equal(
                        request.reference.axis_cm1,
                        request.candidate.axis_cm1,
                    )
                ),
                "point_count": int(request.reference.intensity.size),
            },
        )


@dataclass(frozen=True, init=False)
class MSEMetric(_SpectrumPairMetric):
    metric_id: str = "mse"
    output_id: str = "mse"
    unit: str = "intensity_squared"

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = request
        _, _, difference = self._difference(typed_request)
        return self._result(
            typed_request,
            _stable_mean_square(difference),
        )


@dataclass(frozen=True, init=False)
class RMSEMetric(_SpectrumPairMetric):
    metric_id: str = "rmse"
    output_id: str = "rmse"
    unit: str = "intensity"

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = request
        _, _, difference = self._difference(typed_request)
        return self._result(
            typed_request,
            _stable_rms(difference),
        )


@dataclass(frozen=True, init=False)
class MAEMetric(_SpectrumPairMetric):
    metric_id: str = "mae"
    output_id: str = "mae"
    unit: str = "intensity"

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = request
        _, _, difference = self._difference(typed_request)
        return self._result(
            typed_request,
            _stable_mean_abs(difference),
        )


@dataclass(frozen=True, init=False)
class SAMMetric(_SpectrumPairMetric):
    metric_id: str = "sam"
    output_id: str = "sam"
    unit: str = "radian"

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = request
        reference, candidate = self._arrays(typed_request)
        reference_scale = _max_abs(reference)
        candidate_scale = _max_abs(candidate)
        if reference_scale == 0.0 or candidate_scale == 0.0:
            raise FidelityMetricError(
                "zero norm",
                "reference and candidate norms must be positive",
            )
        reference_scaled = reference / reference_scale
        candidate_scaled = candidate / candidate_scale
        reference_norm = float(np.linalg.norm(reference_scaled))
        candidate_norm = float(np.linalg.norm(candidate_scaled))
        if reference_norm == 0.0 or candidate_norm == 0.0:
            raise FidelityMetricError(
                "zero norm",
                "reference and candidate norms must be positive",
            )
        dot = float(np.dot(reference_scaled, candidate_scaled))
        if not all(
            math.isfinite(value)
            for value in (reference_norm, candidate_norm, dot)
        ):
            raise FidelityMetricError(
                "sam",
                "produced non-finite intermediates",
            )
        cosine = dot / (reference_norm * candidate_norm)
        value = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        return self._result(typed_request, value)


@dataclass(frozen=True, init=False)
class PearsonRMetric(_SpectrumPairMetric):
    metric_id: str = "pearson_r"
    output_id: str = "pearson_r"
    unit: str = "correlation"
    preferred_direction: PreferredDirection = PreferredDirection.HIGHER_IS_BETTER

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = request
        reference, candidate = self._arrays(typed_request)
        left = _stable_centered_unit("reference zero variance", reference)
        right = _stable_centered_unit("candidate zero variance", candidate)
        value = float(np.clip(np.dot(left, right), -1.0, 1.0))
        return self._result(typed_request, value)


@dataclass(frozen=True, init=False)
class NMSEMetric(_SpectrumPairMetric):
    metric_id: str = "nmse"
    output_id: str = "nmse"
    unit: str = "ratio"

    def evaluate(self, request: MetricInput) -> MetricResult:
        typed_request = request
        reference, _, difference = self._difference(typed_request)
        reference_scale = _max_abs(reference)
        if reference_scale == 0.0:
            raise FidelityMetricError(
                "reference energy",
                "must be positive",
            )
        scale = max(reference_scale, _max_abs(difference))
        if scale == 0.0:
            return self._result(typed_request, 0.0)
        reference_scaled = reference / scale
        difference_scaled = difference / scale
        denominator = float(np.sum(reference_scaled * reference_scaled))
        if denominator == 0.0:
            raise FidelityMetricError(
                "reference energy",
                "must be positive",
            )
        numerator = float(np.sum(difference_scaled * difference_scaled))
        value = numerator / denominator
        return self._result(typed_request, value)
