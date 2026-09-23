from __future__ import annotations

import hashlib
import json
import math
import warnings as python_warnings
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt, peak_prominences, peak_widths

from rpe.evaluation import Peak1D, Spectrum1D
from rpe.methods.catalog import Phase3System, TaskLine


_PEAKS_DOMAIN = b"rpe-phase3-detected-peaks-v1\0"
_SUPPORTED_FAMILIES = frozenset({"find_peaks", "find_peaks_cwt"})


class PeakWrapperError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class PeakRunStatus(str, Enum):
    COMPLETE = "complete"
    COMPLETE_WITH_WARNING = "complete_with_warning"
    NOT_APPLICABLE = "not_applicable"
    FAILED_DOMAIN = "failed_domain"
    FAILED_RUNTIME = "failed_runtime"


@dataclass(frozen=True)
class PeakWarning:
    category: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category:
            raise PeakWrapperError("warning.category", "must be nonempty")
        if not isinstance(self.message, str) or not self.message:
            raise PeakWrapperError("warning.message", "must be nonempty")


def _finite(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise PeakWrapperError(path, "must be numeric")
    current = float(value)
    if not math.isfinite(current):
        raise PeakWrapperError(path, "must be finite")
    return current


def _positive(path: str, value: object) -> float:
    current = _finite(path, value)
    if current <= 0.0:
        raise PeakWrapperError(path, "must be positive")
    return current


@dataclass(frozen=True)
class DetectedPeak1D:
    index: int
    position_cm1: float
    height: float
    prominence: float
    fwhm_cm1: float
    area: float
    left_base_index: int
    right_base_index: int
    contour_height: float
    area_left_cm1: float
    area_right_cm1: float

    def __post_init__(self) -> None:
        for name in ("index", "left_base_index", "right_base_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PeakWrapperError(name, "must be a nonnegative integer")
        if not self.left_base_index < self.index < self.right_base_index:
            raise PeakWrapperError("peak bases", "must strictly bracket index")
        object.__setattr__(self, "position_cm1", _finite("position_cm1", self.position_cm1))
        object.__setattr__(self, "height", _finite("height", self.height))
        object.__setattr__(self, "prominence", _positive("prominence", self.prominence))
        object.__setattr__(self, "fwhm_cm1", _positive("fwhm_cm1", self.fwhm_cm1))
        object.__setattr__(self, "area", _positive("area", self.area))
        object.__setattr__(self, "contour_height", _finite("contour_height", self.contour_height))
        left = _finite("area_left_cm1", self.area_left_cm1)
        right = _finite("area_right_cm1", self.area_right_cm1)
        if not left < self.position_cm1 < right:
            raise PeakWrapperError("area bounds", "must strictly bracket peak position")
        object.__setattr__(self, "area_left_cm1", left)
        object.__setattr__(self, "area_right_cm1", right)

    def to_peak1d(self) -> Peak1D:
        return Peak1D(
            position_cm1=self.position_cm1,
            height=self.height,
            fwhm_cm1=self.fwhm_cm1,
            area=self.area,
            prominence=self.prominence,
        )


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def peak_document(peak: DetectedPeak1D) -> dict[str, object]:
    return {
        "area": peak.area,
        "area_left_cm1": peak.area_left_cm1,
        "area_right_cm1": peak.area_right_cm1,
        "contour_height": peak.contour_height,
        "fwhm_cm1": peak.fwhm_cm1,
        "height": peak.height,
        "index": peak.index,
        "left_base_index": peak.left_base_index,
        "position_cm1": peak.position_cm1,
        "prominence": peak.prominence,
        "right_base_index": peak.right_base_index,
    }


def _freeze_value(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PeakWrapperError("diagnostics", "contains non-finite float")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in sorted(value.items())}
        )
    raise PeakWrapperError("diagnostics", "contains unsupported value")


@dataclass(frozen=True)
class PeakDetectionRunResult:
    system_id: str
    family_id: str
    method_id: str
    spectrum_id: str
    status: PeakRunStatus
    peaks: tuple[DetectedPeak1D, ...]
    peaks_sha256: str | None
    warnings: tuple[PeakWarning, ...]
    diagnostics: Mapping[str, object]
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PeakRunStatus):
            raise PeakWrapperError("status", "must be PeakRunStatus")
        for name in ("system_id", "family_id", "method_id", "spectrum_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise PeakWrapperError(name, "must be nonempty")
        if not isinstance(self.peaks, tuple) or any(
            not isinstance(value, DetectedPeak1D) for value in self.peaks
        ):
            raise PeakWrapperError("peaks", "must be a DetectedPeak1D tuple")
        indexes = tuple(value.index for value in self.peaks)
        positions = tuple(value.position_cm1 for value in self.peaks)
        if indexes != tuple(sorted(set(indexes))) or positions != tuple(sorted(positions)):
            raise PeakWrapperError("peaks", "must be unique and position ordered")
        successful = self.status in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}
        if successful:
            digest = hashlib.sha256(
                _PEAKS_DOMAIN
                + b"".join(_canonical(peak_document(value)) for value in self.peaks)
            ).hexdigest()
            if self.peaks_sha256 is not None and self.peaks_sha256 != digest:
                raise PeakWrapperError("peaks_sha256", "does not match peaks")
            object.__setattr__(self, "peaks_sha256", digest)
        elif self.peaks or self.peaks_sha256 is not None:
            raise PeakWrapperError("peaks", "failure must not carry peaks")
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(value, PeakWarning) for value in self.warnings
        ):
            raise PeakWrapperError("warnings", "must be a PeakWarning tuple")
        frozen = _freeze_value(self.diagnostics)
        if not isinstance(frozen, Mapping):
            raise PeakWrapperError("diagnostics", "must be a mapping")
        object.__setattr__(self, "diagnostics", frozen)


class _NotApplicable(PeakWrapperError):
    pass


def _warning_tuple(caught: Sequence[python_warnings.WarningMessage]) -> tuple[PeakWarning, ...]:
    return tuple(
        PeakWarning(type(value.message).__name__, str(value.message))
        for value in caught
    )


def _convert_width_bank(
    axis_cm1: np.ndarray, widths_cm1: Sequence[float]
) -> tuple[tuple[int, ...], Mapping[str, object]]:
    axis = np.asarray(axis_cm1, dtype="<f8")
    if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all():
        raise PeakWrapperError("axis", "must be finite 1-D with at least two points")
    spacing = np.diff(axis)
    if np.any(spacing <= 0.0):
        raise PeakWrapperError("axis", "must be strictly increasing")
    minimum = float(np.min(spacing))
    median = float(np.median(spacing))
    maximum = float(np.max(spacing))
    pairs = tuple(
        (float(width), math.floor(float(width) / median + 0.5))
        for width in widths_cm1
    )
    converted = tuple(sorted({value for _, value in pairs if value > 0}))
    if len(converted) < 2:
        raise PeakWrapperError(
            "unsupported_width_bank", "must yield at least two unique positive widths"
        )
    return converted, MappingProxyType(
        {
            "max_spacing": maximum,
            "median_spacing": median,
            "min_spacing": minimum,
            "physical_to_point": pairs,
            "spacing_irregularity_ratio": maximum / minimum,
        }
    )


def _interpolated_crossing(
    axis: np.ndarray,
    intensity: np.ndarray,
    contour: float,
    inner: int,
    outer: int,
) -> float:
    y_inner = float(intensity[inner] - contour)
    y_outer = float(intensity[outer] - contour)
    if y_inner < 0.0 or y_outer > 0.0:
        raise PeakWrapperError("area crossing", "does not bracket contour")
    if y_inner == 0.0:
        return float(axis[inner])
    if y_outer == 0.0:
        return float(axis[outer])
    fraction = y_inner / (y_inner - y_outer)
    return float(axis[inner] + fraction * (axis[outer] - axis[inner]))


def _peak_area(
    axis: np.ndarray,
    intensity: np.ndarray,
    *,
    index: int,
    left_base: int,
    right_base: int,
    contour: float,
) -> tuple[float, float, float]:
    left_inner = index
    while left_inner > left_base and intensity[left_inner - 1] > contour:
        left_inner -= 1
    left_outer = max(left_inner - 1, left_base)
    right_inner = index
    while right_inner < right_base and intensity[right_inner + 1] > contour:
        right_inner += 1
    right_outer = min(right_inner + 1, right_base)
    left_position = _interpolated_crossing(
        axis, intensity, contour, left_inner, left_outer
    )
    right_position = _interpolated_crossing(
        axis, intensity, contour, right_inner, right_outer
    )
    middle_start = left_inner if float(axis[left_inner]) > left_position else left_inner + 1
    middle_stop = right_inner + 1
    middle_axis = axis[middle_start:middle_stop]
    middle_height = np.maximum(intensity[middle_start:middle_stop] - contour, 0.0)
    area_axis = np.concatenate(([left_position], middle_axis, [right_position]))
    area_height = np.concatenate(([0.0], middle_height, [0.0]))
    keep = np.concatenate(([True], np.diff(area_axis) > 0.0))
    area_axis = area_axis[keep]
    area_height = area_height[keep]
    if area_axis.size < 2 or not left_position < float(axis[index]) < right_position:
        raise PeakWrapperError("area", "invalid connected support")
    area = float(np.trapezoid(area_height, area_axis))
    if not math.isfinite(area) or area <= 0.0:
        raise PeakWrapperError("area", "must be finite and positive")
    return area, left_position, right_position


def _characterize_candidates(
    spectrum: Spectrum1D, candidates: Sequence[int]
) -> tuple[tuple[DetectedPeak1D, ...], Counter[str]]:
    axis = spectrum.axis_cm1
    intensity = spectrum.intensity
    rejection: Counter[str] = Counter()
    peaks: list[DetectedPeak1D] = []
    for index in sorted(set(int(value) for value in candidates)):
        if index <= 0 or index >= intensity.size - 1:
            rejection["candidate_out_of_range_or_endpoint"] += 1
            continue
        try:
            prominence_array, left_array, right_array = peak_prominences(
                intensity, np.array([index], dtype=np.intp), wlen=None
            )
            prominence = float(prominence_array[0])
            left_base = int(left_array[0])
            right_base = int(right_array[0])
            if not math.isfinite(prominence) or prominence <= 0.0:
                raise PeakWrapperError("prominence", "must be finite and positive")
            contour = float(max(intensity[left_base], intensity[right_base]))
            width_values, _, left_ips, right_ips = peak_widths(
                intensity,
                np.array([index], dtype=np.intp),
                rel_height=0.5,
                prominence_data=(prominence_array, left_array, right_array),
            )
            index_axis = np.arange(axis.size, dtype="<f8")
            left_position = float(np.interp(float(left_ips[0]), index_axis, axis))
            right_position = float(np.interp(float(right_ips[0]), index_axis, axis))
            fwhm = right_position - left_position
            if not math.isfinite(float(width_values[0])) or not math.isfinite(fwhm) or fwhm <= 0.0:
                raise PeakWrapperError("fwhm", "must be finite and positive")
            area, area_left, area_right = _peak_area(
                axis,
                intensity,
                index=index,
                left_base=left_base,
                right_base=right_base,
                contour=contour,
            )
            peaks.append(
                DetectedPeak1D(
                    index=index,
                    position_cm1=float(axis[index]),
                    height=float(intensity[index]),
                    prominence=prominence,
                    fwhm_cm1=fwhm,
                    area=area,
                    left_base_index=left_base,
                    right_base_index=right_base,
                    contour_height=contour,
                    area_left_cm1=area_left,
                    area_right_cm1=area_right,
                )
            )
        except PeakWrapperError as error:
            rejection[f"invalid_{error.path}"] += 1
    return tuple(peaks), rejection


def _result(
    system: Phase3System,
    spectrum: Spectrum1D,
    status: PeakRunStatus,
    *,
    peaks: tuple[DetectedPeak1D, ...] = (),
    warnings: tuple[PeakWarning, ...] = (),
    diagnostics: Mapping[str, object] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> PeakDetectionRunResult:
    return PeakDetectionRunResult(
        system_id=system.system_id,
        family_id=system.family_id,
        method_id=system.method_id,
        spectrum_id=spectrum.spectrum_id,
        status=status,
        peaks=peaks,
        peaks_sha256=None,
        warnings=warnings,
        diagnostics={} if diagnostics is None else diagnostics,
        error_code=error_code,
        error_message=error_message,
    )


def run_peak_detection_system(
    system: Phase3System, spectrum: Spectrum1D
) -> PeakDetectionRunResult:
    if not isinstance(system, Phase3System) or system.task_line is not TaskLine.PEAK_DETECTION:
        raise PeakWrapperError("system", "must be a peak-detection Phase3System")
    if system.family_id == "mspd":
        raise PeakWrapperError("MSPD", "catalog v1 algorithm contract does not match published MSPD")
    if system.family_id not in _SUPPORTED_FAMILIES:
        raise PeakWrapperError("system", "has unsupported peak family")
    if not isinstance(spectrum, Spectrum1D):
        raise PeakWrapperError("spectrum", "must be Spectrum1D")
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            if system.family_id == "find_peaks":
                robust_range = float(
                    np.quantile(spectrum.intensity, 0.99, method="linear")
                    - np.quantile(spectrum.intensity, 0.01, method="linear")
                )
                if not math.isfinite(robust_range) or robust_range <= 0.0:
                    raise _NotApplicable(
                        "nonpositive_robust_range", "q99-q01 must be positive"
                    )
                values = dict(system.hyperparameters)
                fraction = float(values.pop("prominence_fraction"))
                candidates, _ = find_peaks(
                    spectrum.intensity,
                    prominence=fraction * robust_range,
                    **values,
                )
                diagnostics: dict[str, object] = {
                    "candidate_count": int(candidates.size),
                    "prominence_threshold": fraction * robust_range,
                    "robust_range": robust_range,
                }
            else:
                values = dict(system.hyperparameters)
                widths_cm1 = tuple(float(value) for value in values.pop("widths_cm1"))
                converted, width_diagnostics = _convert_width_bank(
                    spectrum.axis_cm1, widths_cm1
                )
                raw_candidates = tuple(
                    int(value)
                    for value in find_peaks_cwt(
                        spectrum.intensity,
                        converted,
                        **values,
                    )
                )
                admissible = set(int(value) for value in find_peaks(spectrum.intensity)[0])
                retained = []
                seen = set()
                rejected_duplicate = 0
                rejected_non_admissible = 0
                rejected_out_of_range = 0
                for index in raw_candidates:
                    if index <= 0 or index >= spectrum.intensity.size - 1:
                        rejected_out_of_range += 1
                    elif index in seen:
                        rejected_duplicate += 1
                    elif index not in admissible:
                        rejected_non_admissible += 1
                    else:
                        seen.add(index)
                        retained.append(index)
                candidates = np.asarray(sorted(retained), dtype=np.intp)
                diagnostics = {
                    **dict(width_diagnostics),
                    "candidate_count": len(raw_candidates),
                    "rejected_duplicate": rejected_duplicate,
                    "rejected_non_admissible": rejected_non_admissible,
                    "rejected_out_of_range": rejected_out_of_range,
                    "retained_candidate_count": int(candidates.size),
                }
            peaks, rejection = _characterize_candidates(spectrum, candidates)
            diagnostics["characterization_rejections"] = dict(sorted(rejection.items()))
        captured = _warning_tuple(caught)
        status = PeakRunStatus.COMPLETE_WITH_WARNING if captured else PeakRunStatus.COMPLETE
        return _result(
            system,
            spectrum,
            status,
            peaks=peaks,
            warnings=captured,
            diagnostics=diagnostics,
        )
    except _NotApplicable as error:
        return _result(
            system,
            spectrum,
            PeakRunStatus.NOT_APPLICABLE,
            error_code=error.path,
            error_message=error.reason,
        )
    except PeakWrapperError as error:
        return _result(
            system,
            spectrum,
            PeakRunStatus.NOT_APPLICABLE,
            error_code=error.path,
            error_message=error.reason,
        )
    except Exception as error:
        return _result(
            system,
            spectrum,
            PeakRunStatus.FAILED_RUNTIME,
            error_code=type(error).__name__,
            error_message=str(error) or type(error).__name__,
        )


__all__ = [
    "DetectedPeak1D",
    "PeakDetectionRunResult",
    "PeakRunStatus",
    "PeakWarning",
    "PeakWrapperError",
    "peak_document",
    "run_peak_detection_system",
]
