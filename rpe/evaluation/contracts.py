from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, TypeAlias, runtime_checkable

import numpy as np


JsonValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | tuple["JsonValue", ...]
    | Mapping[str, "JsonValue"]
)


class EvaluationContractError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class MetricInputKind(str, Enum):
    SPECTRUM_PAIR = "spectrum_pair"
    SPECTRUM_REGIONS = "spectrum_regions"
    PEAK_PAIR = "peak_pair"
    SINGLE_SPECTRUM = "single_spectrum"
    CALIBRATION = "calibration"
    REPLICATE_PAIR = "replicate_pair"


class AxisPolicy(str, Enum):
    INDEX_ALIGNED = "index_aligned"
    EXACT_AXIS = "exact_axis"
    INDEPENDENT_PHYSICAL_AXES = "independent_physical_axes"
    SINGLE_AXIS = "single_axis"
    NOT_APPLICABLE = "not_applicable"


class PreferredDirection(str, Enum):
    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"
    TARGET_VALUE = "target_value"
    NON_MONOTONIC = "non_monotonic"


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise EvaluationContractError(path, "must be a nonempty string")
    return value


def _finite_number(path: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise EvaluationContractError(path, "must be a finite real number")
    return float(value)


def _positive_number(path: str, value: object) -> float:
    converted = _finite_number(path, value)
    if converted <= 0.0:
        raise EvaluationContractError(path, "must be positive")
    return converted


def _read_only_float64_vector(
    path: str,
    value: np.ndarray,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise EvaluationContractError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise EvaluationContractError(
            f"{path} dtype",
            "must be little-endian float64",
        )
    if value.ndim != 1:
        raise EvaluationContractError(
            f"{path} dimension",
            "must be one-dimensional",
        )
    if value.size == 0:
        raise EvaluationContractError(
            f"{path} shape",
            "must be nonempty",
        )
    if not np.isfinite(value).all():
        raise EvaluationContractError(
            f"{path} finite",
            "contains non-finite values",
        )
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _read_only_float64_matrix(
    path: str,
    value: np.ndarray,
    *,
    min_rows: int,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise EvaluationContractError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise EvaluationContractError(
            f"{path} dtype",
            "must be little-endian float64",
        )
    if (
        value.ndim != 2
        or value.shape[0] < min_rows
        or value.shape[1] == 0
    ):
        raise EvaluationContractError(
            f"{path} shape",
            f"must be two-dimensional with at least {min_rows} rows",
        )
    if not np.isfinite(value).all():
        raise EvaluationContractError(
            f"{path} finite",
            "contains non-finite values",
        )
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _freeze_json(path: str, value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvaluationContractError(
                f"{path} finite",
                "must be finite",
            )
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(f"{path}[{index}]", item)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        if any(
            not isinstance(key, str) or key == ""
            for key in value
        ):
            raise EvaluationContractError(
                path,
                "mapping keys must be nonempty strings",
            )
        return MappingProxyType(
            {
                key: _freeze_json(f"{path}.{key}", item)
                for key, item in sorted(value.items())
            }
        )
    raise EvaluationContractError(
        path,
        "must be canonical-JSON-compatible",
    )


def _json_mapping_keys(value: JsonValue) -> set[str]:
    keys = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(key)
            keys.update(_json_mapping_keys(item))
    elif isinstance(value, tuple):
        for item in value:
            keys.update(_json_mapping_keys(item))
    return keys


def _finite_tuple(
    path: str,
    values: object,
    *,
    nonempty: bool,
) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise EvaluationContractError(path, "must be a tuple")
    if nonempty and not values:
        raise EvaluationContractError(path, "must be nonempty")
    return tuple(
        _finite_number(f"{path}[{index}]", value)
        for index, value in enumerate(values)
    )


def _strictly_increasing(path: str, values: tuple[float, ...]) -> None:
    if any(
        right <= left
        for left, right in zip(values, values[1:])
    ):
        raise EvaluationContractError(
            path,
            "must be strictly increasing",
        )


@dataclass(frozen=True)
class Spectrum1D:
    spectrum_id: str
    sample_id: str | None
    axis_cm1: np.ndarray
    intensity: np.ndarray

    def __post_init__(self) -> None:
        _nonempty_string("spectrum_id", self.spectrum_id)
        if self.sample_id is not None:
            _nonempty_string("sample_id", self.sample_id)
        axis = _read_only_float64_vector("axis", self.axis_cm1)
        intensity = _read_only_float64_vector(
            "intensity",
            self.intensity,
        )
        if axis.shape != intensity.shape:
            raise EvaluationContractError(
                "array length",
                "axis and intensity must have equal length",
            )
        if not np.all(np.diff(axis) > 0.0):
            raise EvaluationContractError(
                "axis increasing",
                "must be strictly increasing",
            )
        object.__setattr__(self, "axis_cm1", axis)
        object.__setattr__(self, "intensity", intensity)


@dataclass(frozen=True)
class SpectralRegion:
    region_id: str
    start_cm1: float
    end_cm1: float
    role: str

    def __post_init__(self) -> None:
        _nonempty_string("region_id", self.region_id)
        _nonempty_string("region role", self.role)
        start = _finite_number("region start_cm1", self.start_cm1)
        end = _finite_number("region end_cm1", self.end_cm1)
        if start >= end:
            raise EvaluationContractError(
                "region bounds",
                "start_cm1 must be less than end_cm1",
            )
        object.__setattr__(self, "start_cm1", start)
        object.__setattr__(self, "end_cm1", end)


@dataclass(frozen=True)
class Peak1D:
    position_cm1: float
    height: float | None
    fwhm_cm1: float | None
    area: float | None
    prominence: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_cm1",
            _finite_number("peak position", self.position_cm1),
        )
        if self.height is not None:
            object.__setattr__(
                self,
                "height",
                _finite_number("peak height", self.height),
            )
        if self.fwhm_cm1 is not None:
            object.__setattr__(
                self,
                "fwhm_cm1",
                _positive_number("peak fwhm", self.fwhm_cm1),
            )
        if self.area is not None:
            object.__setattr__(
                self,
                "area",
                _finite_number("peak area", self.area),
            )
        if self.prominence is not None:
            object.__setattr__(
                self,
                "prominence",
                _positive_number(
                    "peak prominence",
                    self.prominence,
                ),
            )


@dataclass(frozen=True)
class SpectrumPairInput:
    reference: Spectrum1D
    candidate: Spectrum1D
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SPECTRUM_PAIR,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Spectrum1D):
            raise EvaluationContractError(
                "reference",
                "must be Spectrum1D",
            )
        if not isinstance(self.candidate, Spectrum1D):
            raise EvaluationContractError(
                "candidate",
                "must be Spectrum1D",
            )


@dataclass(frozen=True)
class SpectrumRegionsInput:
    spectrum: Spectrum1D
    regions: tuple[SpectralRegion, ...]
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SPECTRUM_REGIONS,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spectrum, Spectrum1D):
            raise EvaluationContractError(
                "spectrum",
                "must be Spectrum1D",
            )
        if not isinstance(self.regions, tuple) or not self.regions:
            raise EvaluationContractError(
                "regions",
                "must be a nonempty tuple",
            )
        if any(
            not isinstance(region, SpectralRegion)
            for region in self.regions
        ):
            raise EvaluationContractError(
                "regions",
                "must contain only SpectralRegion values",
            )
        region_ids = [region.region_id for region in self.regions]
        if len(set(region_ids)) != len(region_ids):
            raise EvaluationContractError(
                "region_ids",
                "must be unique",
            )
        lower = float(self.spectrum.axis_cm1[0])
        upper = float(self.spectrum.axis_cm1[-1])
        if any(
            region.start_cm1 < lower or region.end_cm1 > upper
            for region in self.regions
        ):
            raise EvaluationContractError(
                "region domain",
                "regions must lie inside the spectrum domain",
            )


def _validate_peak_tuple(
    path: str,
    peaks: object,
) -> tuple[Peak1D, ...]:
    if not isinstance(peaks, tuple):
        raise EvaluationContractError(path, "must be a tuple")
    if any(not isinstance(peak, Peak1D) for peak in peaks):
        raise EvaluationContractError(
            path,
            "must contain only Peak1D values",
        )
    positions = tuple(peak.position_cm1 for peak in peaks)
    if any(
        right <= left
        for left, right in zip(positions, positions[1:])
    ):
        raise EvaluationContractError(
            f"{path} position",
            "peak positions must be strictly increasing",
        )
    return peaks


@dataclass(frozen=True)
class PeakPairInput:
    reference_peaks: tuple[Peak1D, ...]
    candidate_peaks: tuple[Peak1D, ...]
    position_tolerance_cm1: float
    prominence_thresholds: tuple[float, ...]
    input_kind: MetricInputKind = field(
        default=MetricInputKind.PEAK_PAIR,
        init=False,
    )

    def __post_init__(self) -> None:
        reference = _validate_peak_tuple(
            "reference_peaks",
            self.reference_peaks,
        )
        candidate = _validate_peak_tuple(
            "candidate_peaks",
            self.candidate_peaks,
        )
        tolerance = _positive_number(
            "position_tolerance",
            self.position_tolerance_cm1,
        )
        thresholds = _finite_tuple(
            "prominence_thresholds",
            self.prominence_thresholds,
            nonempty=True,
        )
        if any(value < 0.0 for value in thresholds):
            raise EvaluationContractError(
                "prominence_thresholds",
                "must be nonnegative",
            )
        _strictly_increasing(
            "prominence_thresholds",
            thresholds,
        )
        object.__setattr__(self, "reference_peaks", reference)
        object.__setattr__(self, "candidate_peaks", candidate)
        object.__setattr__(
            self,
            "position_tolerance_cm1",
            tolerance,
        )
        object.__setattr__(
            self,
            "prominence_thresholds",
            thresholds,
        )


@dataclass(frozen=True)
class SingleSpectrumInput:
    spectrum: Spectrum1D
    input_kind: MetricInputKind = field(
        default=MetricInputKind.SINGLE_SPECTRUM,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.spectrum, Spectrum1D):
            raise EvaluationContractError(
                "spectrum",
                "must be Spectrum1D",
            )


@dataclass(frozen=True)
class CalibrationInput:
    target_names: tuple[str, ...]
    true_concentrations: np.ndarray
    predicted_concentrations: np.ndarray
    blank_predictions: np.ndarray
    input_kind: MetricInputKind = field(
        default=MetricInputKind.CALIBRATION,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.target_names, tuple) or not self.target_names:
            raise EvaluationContractError(
                "target_names",
                "must be a nonempty tuple",
            )
        if any(
            not isinstance(name, str) or name == ""
            for name in self.target_names
        ):
            raise EvaluationContractError(
                "target_names",
                "must contain nonempty strings",
            )
        if len(set(self.target_names)) != len(self.target_names):
            raise EvaluationContractError(
                "target_names",
                "must be unique",
            )
        true = _read_only_float64_matrix(
            "true_concentrations",
            self.true_concentrations,
            min_rows=1,
        )
        predicted = _read_only_float64_matrix(
            "predicted_concentrations",
            self.predicted_concentrations,
            min_rows=1,
        )
        blank = _read_only_float64_matrix(
            "blank_predictions",
            self.blank_predictions,
            min_rows=2,
        )
        target_count = len(self.target_names)
        if predicted.shape != true.shape:
            raise EvaluationContractError(
                "predicted_concentrations shape",
                "must equal true_concentrations shape",
            )
        if true.shape[1] != target_count:
            raise EvaluationContractError(
                "true_concentrations shape",
                "column count must equal target_names",
            )
        if blank.shape[1] != target_count:
            raise EvaluationContractError(
                "blank_predictions shape",
                "column count must equal target_names",
            )
        object.__setattr__(self, "true_concentrations", true)
        object.__setattr__(
            self,
            "predicted_concentrations",
            predicted,
        )
        object.__setattr__(self, "blank_predictions", blank)


@dataclass(frozen=True)
class ReplicatePairInput:
    left: Spectrum1D
    right: Spectrum1D
    input_kind: MetricInputKind = field(
        default=MetricInputKind.REPLICATE_PAIR,
        init=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.left, Spectrum1D):
            raise EvaluationContractError(
                "left",
                "must be Spectrum1D",
            )
        if not isinstance(self.right, Spectrum1D):
            raise EvaluationContractError(
                "right",
                "must be Spectrum1D",
            )


MetricInput: TypeAlias = (
    SpectrumPairInput
    | SpectrumRegionsInput
    | PeakPairInput
    | SingleSpectrumInput
    | CalibrationInput
    | ReplicatePairInput
)


def _validate_preference(
    path: str,
    preferred_direction: object,
    target_value: object,
) -> float | None:
    if not isinstance(preferred_direction, PreferredDirection):
        raise EvaluationContractError(
            f"{path} preferred_direction",
            "must be PreferredDirection",
        )
    if preferred_direction is PreferredDirection.TARGET_VALUE:
        if target_value is None:
            raise EvaluationContractError(
                f"{path} target_value",
                "is required for target_value direction",
            )
        return _finite_number(
            f"{path} target_value",
            target_value,
        )
    if target_value is not None:
        raise EvaluationContractError(
            f"{path} target_value",
            "must be None unless direction is target_value",
        )
    return None


@dataclass(frozen=True)
class ScalarMetricOutput:
    output_id: str
    value: float
    unit: str
    preferred_direction: PreferredDirection
    target_value: float | None

    def __post_init__(self) -> None:
        _nonempty_string("output_id", self.output_id)
        _nonempty_string("output unit", self.unit)
        value = _finite_number("value finite", self.value)
        target = _validate_preference(
            "scalar output",
            self.preferred_direction,
            self.target_value,
        )
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "target_value", target)


@dataclass(frozen=True)
class CurveSeries:
    series_id: str
    values: tuple[float, ...]
    unit: str
    preferred_direction: PreferredDirection
    target_value: float | None

    def __post_init__(self) -> None:
        _nonempty_string("series_id", self.series_id)
        _nonempty_string("series unit", self.unit)
        values = _finite_tuple(
            "series values",
            self.values,
            nonempty=True,
        )
        target = _validate_preference(
            "curve series",
            self.preferred_direction,
            self.target_value,
        )
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "target_value", target)


@dataclass(frozen=True)
class CurveMetricOutput:
    output_id: str
    x_name: str
    x_values: tuple[float, ...]
    x_unit: str
    series: tuple[CurveSeries, ...]

    def __post_init__(self) -> None:
        _nonempty_string("output_id", self.output_id)
        _nonempty_string("x_name", self.x_name)
        _nonempty_string("x_unit", self.x_unit)
        values = _finite_tuple(
            "x_values",
            self.x_values,
            nonempty=True,
        )
        _strictly_increasing("x_values", values)
        if not isinstance(self.series, tuple) or not self.series:
            raise EvaluationContractError(
                "series",
                "must be a nonempty tuple",
            )
        if any(
            not isinstance(series, CurveSeries)
            for series in self.series
        ):
            raise EvaluationContractError(
                "series",
                "must contain only CurveSeries values",
            )
        series_ids = [series.series_id for series in self.series]
        if len(set(series_ids)) != len(series_ids):
            raise EvaluationContractError(
                "series_ids",
                "must be unique",
            )
        if any(
            len(series.values) != len(values)
            for series in self.series
        ):
            raise EvaluationContractError(
                "series values",
                "must align with x_values",
            )
        object.__setattr__(self, "x_values", values)


MetricOutput: TypeAlias = ScalarMetricOutput | CurveMetricOutput


@dataclass(frozen=True)
class MetricResult:
    metric_id: str
    input_kind: MetricInputKind
    outputs: tuple[MetricOutput, ...]
    diagnostics: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        _nonempty_string("metric_id", self.metric_id)
        if not isinstance(self.input_kind, MetricInputKind):
            raise EvaluationContractError(
                "input_kind",
                "must be MetricInputKind",
            )
        if not isinstance(self.outputs, tuple) or not self.outputs:
            raise EvaluationContractError(
                "outputs",
                "must be a nonempty tuple",
            )
        if any(
            not isinstance(
                output,
                (ScalarMetricOutput, CurveMetricOutput),
            )
            for output in self.outputs
        ):
            raise EvaluationContractError(
                "outputs",
                "must contain metric outputs",
            )
        output_ids = [output.output_id for output in self.outputs]
        if len(set(output_ids)) != len(output_ids):
            raise EvaluationContractError(
                "output_ids",
                "must be unique",
            )
        diagnostics = _freeze_json(
            "diagnostics",
            self.diagnostics,
        )
        if not isinstance(diagnostics, Mapping):
            raise EvaluationContractError(
                "diagnostics",
                "must be a mapping",
            )
        primary_ids = set(output_ids)
        for output in self.outputs:
            if isinstance(output, CurveMetricOutput):
                primary_ids.update(
                    series.series_id for series in output.series
                )
        duplicated = primary_ids & _json_mapping_keys(diagnostics)
        if duplicated:
            raise EvaluationContractError(
                "diagnostics",
                (
                    "must not duplicate primary output IDs: "
                    + ",".join(sorted(duplicated))
                ),
            )
        object.__setattr__(self, "diagnostics", diagnostics)


@runtime_checkable
class Metric(Protocol):
    metric_id: str
    input_kind: MetricInputKind
    axis_policy: AxisPolicy

    def evaluate(self, request: MetricInput) -> MetricResult:
        pass


def validate_metric_request(request: MetricInput) -> None:
    if not isinstance(
        request,
        (
            SpectrumPairInput,
            SpectrumRegionsInput,
            PeakPairInput,
            SingleSpectrumInput,
            CalibrationInput,
            ReplicatePairInput,
        ),
    ):
        raise EvaluationContractError(
            "request",
            "unsupported request type",
        )


def validate_metric_result(result: MetricResult) -> None:
    if not isinstance(result, MetricResult):
        raise EvaluationContractError(
            "metric result",
            "must be MetricResult",
        )
    if not isinstance(result.metric_id, str) or result.metric_id == "":
        raise EvaluationContractError(
            "metric result metric_id",
            "must be a nonempty string",
        )
    if not isinstance(result.input_kind, MetricInputKind):
        raise EvaluationContractError(
            "metric result input_kind",
            "must be MetricInputKind",
        )
    if not isinstance(result.outputs, tuple) or not result.outputs:
        raise EvaluationContractError(
            "metric result outputs",
            "must be a nonempty tuple",
        )
    output_ids = []
    for index, output in enumerate(result.outputs):
        if isinstance(output, ScalarMetricOutput):
            if not math.isfinite(output.value):
                raise EvaluationContractError(
                    f"metric result outputs[{index}].value",
                    "must be finite",
                )
        elif isinstance(output, CurveMetricOutput):
            if (
                not output.x_values
                or any(
                    not math.isfinite(value)
                    for value in output.x_values
                )
                or any(
                    right <= left
                    for left, right in zip(
                        output.x_values,
                        output.x_values[1:],
                    )
                )
            ):
                raise EvaluationContractError(
                    f"metric result outputs[{index}].x_values",
                    "must be finite and strictly increasing",
                )
            for series_index, series in enumerate(output.series):
                if (
                    len(series.values) != len(output.x_values)
                    or any(
                        not math.isfinite(value)
                        for value in series.values
                    )
                ):
                    raise EvaluationContractError(
                        (
                            f"metric result outputs[{index}]"
                            f".series[{series_index}].values"
                        ),
                        "must align with finite x values",
                    )
        else:
            raise EvaluationContractError(
                f"metric result outputs[{index}]",
                "must be ScalarMetricOutput or CurveMetricOutput",
            )
        output_ids.append(output.output_id)
    if len(set(output_ids)) != len(output_ids):
        raise EvaluationContractError(
            "metric result output_ids",
            "must be unique",
        )


def _validate_axis_policy(
    metric: Metric,
    request: MetricInput,
) -> None:
    if not isinstance(metric.axis_policy, AxisPolicy):
        raise EvaluationContractError(
            "metric axis_policy",
            "must be AxisPolicy",
        )
    policy = metric.axis_policy
    if isinstance(request, SpectrumPairInput):
        if policy not in {
            AxisPolicy.INDEX_ALIGNED,
            AxisPolicy.EXACT_AXIS,
            AxisPolicy.INDEPENDENT_PHYSICAL_AXES,
        }:
            raise EvaluationContractError(
                "metric axis_policy",
                "is invalid for spectrum pair",
            )
        if (
            policy is AxisPolicy.INDEX_ALIGNED
            and request.reference.intensity.shape
            != request.candidate.intensity.shape
        ):
            raise EvaluationContractError(
                "index-aligned point count",
                "must be equal",
            )
        if policy is AxisPolicy.EXACT_AXIS and (
            request.reference.intensity.shape
            != request.candidate.intensity.shape
            or not np.array_equal(
                request.reference.axis_cm1,
                request.candidate.axis_cm1,
            )
        ):
            raise EvaluationContractError(
                "exact axis",
                "spectra must have exactly equal axes",
            )
        return
    if isinstance(
        request,
        (SpectrumRegionsInput, SingleSpectrumInput),
    ):
        if policy is not AxisPolicy.SINGLE_AXIS:
            raise EvaluationContractError(
                "metric axis_policy",
                "must be single_axis",
            )
        return
    if isinstance(request, (PeakPairInput, CalibrationInput)):
        if policy is not AxisPolicy.NOT_APPLICABLE:
            raise EvaluationContractError(
                "metric axis_policy",
                "must be not_applicable",
            )
        return
    if isinstance(request, ReplicatePairInput):
        if policy is not AxisPolicy.EXACT_AXIS:
            raise EvaluationContractError(
                "replicate axis_policy",
                "must be exact_axis",
            )
        if (
            request.left.intensity.shape != request.right.intensity.shape
            or not np.array_equal(
                request.left.axis_cm1,
                request.right.axis_cm1,
            )
        ):
            raise EvaluationContractError(
                "replicate exact axis",
                "replicates must have exactly equal axes",
            )
        return
    raise EvaluationContractError(
        "request",
        "unsupported request type",
    )


def evaluate_metric(
    metric: Metric,
    request: MetricInput,
) -> MetricResult:
    if not isinstance(metric, Metric):
        raise EvaluationContractError(
            "metric",
            "must implement Metric",
        )
    if not isinstance(metric.metric_id, str) or metric.metric_id == "":
        raise EvaluationContractError(
            "metric metric_id",
            "must be a nonempty string",
        )
    if not isinstance(metric.input_kind, MetricInputKind):
        raise EvaluationContractError(
            "metric input_kind",
            "must be MetricInputKind",
        )
    validate_metric_request(request)
    if metric.input_kind is not request.input_kind:
        raise EvaluationContractError(
            "metric input_kind",
            "does not match request",
        )
    _validate_axis_policy(metric, request)
    result = metric.evaluate(request)
    if not isinstance(result, MetricResult):
        raise EvaluationContractError(
            "metric result",
            "must be MetricResult",
        )
    validate_metric_result(result)
    if result.metric_id != metric.metric_id:
        raise EvaluationContractError(
            "metric result metric_id",
            "does not match metric",
        )
    if result.input_kind is not request.input_kind:
        raise EvaluationContractError(
            "metric result input_kind",
            "does not match request",
        )
    return result


__all__ = [
    "AxisPolicy",
    "CalibrationInput",
    "CurveMetricOutput",
    "CurveSeries",
    "EvaluationContractError",
    "JsonValue",
    "Metric",
    "MetricInput",
    "MetricInputKind",
    "MetricOutput",
    "MetricResult",
    "Peak1D",
    "PeakPairInput",
    "PreferredDirection",
    "ReplicatePairInput",
    "ScalarMetricOutput",
    "SingleSpectrumInput",
    "SpectralRegion",
    "Spectrum1D",
    "SpectrumPairInput",
    "SpectrumRegionsInput",
    "evaluate_metric",
    "validate_metric_request",
    "validate_metric_result",
]
