from __future__ import annotations

import hashlib
import math
import warnings as python_warnings
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np
from pybaselines import Baseline

from rpe.evaluation import Spectrum1D
from rpe.methods.catalog import Phase3System, TaskLine


class BaselineWrapperError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class BaselineRunStatus(str, Enum):
    COMPLETE = "complete"
    COMPLETE_WITH_WARNING = "complete_with_warning"
    NOT_APPLICABLE = "not_applicable"
    FAILED_CONVERGENCE = "failed_convergence"
    FAILED_RUNTIME = "failed_runtime"


@dataclass(frozen=True)
class CapturedWarning:
    category: str
    message: str

    def __post_init__(self) -> None:
        if not self.category or not self.message:
            raise BaselineWrapperError("warning", "must be nonempty")


@dataclass(frozen=True)
class BaselineRunResult:
    system_id: str
    family_id: str
    method_id: str
    spectrum_id: str
    status: BaselineRunStatus
    baseline_estimate: np.ndarray | None
    corrected_intensity: np.ndarray | None
    baseline_sha256: str | None
    corrected_sha256: str | None
    warnings: tuple[CapturedWarning, ...]
    diagnostics: Mapping[str, object]
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, BaselineRunStatus):
            raise BaselineWrapperError("status", "must be BaselineRunStatus")
        for name in ("system_id", "family_id", "method_id", "spectrum_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise BaselineWrapperError(name, "must be nonempty")
        arrays = (self.baseline_estimate, self.corrected_intensity)
        if self.status in {
            BaselineRunStatus.COMPLETE,
            BaselineRunStatus.COMPLETE_WITH_WARNING,
            BaselineRunStatus.FAILED_CONVERGENCE,
        }:
            if any(value is None for value in arrays):
                raise BaselineWrapperError("outputs", "terminal output status requires arrays")
            copied = []
            for name, value in zip(("baseline_estimate", "corrected_intensity"), arrays):
                assert isinstance(value, np.ndarray)
                if value.dtype != np.dtype("<f8") or value.ndim != 1 or not np.isfinite(value).all():
                    raise BaselineWrapperError(name, "must be a finite float64 vector")
                current = np.ascontiguousarray(value).copy()
                current.setflags(write=False)
                copied.append(current)
            object.__setattr__(self, "baseline_estimate", copied[0])
            object.__setattr__(self, "corrected_intensity", copied[1])
        elif any(value is not None for value in arrays):
            raise BaselineWrapperError("outputs", "failure without output must not carry arrays")
        object.__setattr__(self, "diagnostics", _freeze_mapping("diagnostics", self.diagnostics))


def _freeze_value(path: str, value: object) -> object:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BaselineWrapperError(path, "must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_value(f"{path}[{index}]", item) for index, item in enumerate(value))
    if isinstance(value, Mapping):
        return _freeze_mapping(path, value)
    raise BaselineWrapperError(path, "must be JSON-compatible")


def _freeze_mapping(path: str, value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BaselineWrapperError(path, "must be a mapping")
    return MappingProxyType(
        {
            key: _freeze_value(f"{path}.{key}", item)
            for key, item in sorted(value.items())
            if isinstance(key, str) and key
        }
    )


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _terminal(
    system: Phase3System,
    spectrum: Spectrum1D,
    status: BaselineRunStatus,
    *,
    baseline: np.ndarray | None = None,
    corrected: np.ndarray | None = None,
    warnings: tuple[CapturedWarning, ...] = (),
    diagnostics: Mapping[str, object] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> BaselineRunResult:
    return BaselineRunResult(
        system_id=system.system_id,
        family_id=system.family_id,
        method_id=system.method_id,
        spectrum_id=spectrum.spectrum_id,
        status=status,
        baseline_estimate=baseline,
        corrected_intensity=corrected,
        baseline_sha256=None if baseline is None else _array_sha256(baseline),
        corrected_sha256=None if corrected is None else _array_sha256(corrected),
        warnings=warnings,
        diagnostics={} if diagnostics is None else diagnostics,
        error_code=error_code,
        error_message=error_message,
    )


def _converted_window(system: Phase3System, spectrum: Spectrum1D) -> tuple[int | None, str | None]:
    if system.family_id not in {"snip", "morphological"}:
        return None, None
    spacing = float(np.median(np.diff(spectrum.axis_cm1)))
    key = "max_half_window_cm1" if system.family_id == "snip" else "half_window_cm1"
    width = float(system.hyperparameters[key])
    converted = int(round(width / spacing))
    maximum = (spectrum.axis_cm1.size - 1) // 2
    if converted < 1 or converted > maximum:
        return None, "converted_half_window_out_of_domain"
    return converted, None


def _backend_kwargs(system: Phase3System, converted_window: int | None) -> dict[str, object]:
    values = dict(system.hyperparameters)
    if system.family_id == "snip":
        values.pop("max_half_window_cm1")
        values["max_half_window"] = converted_window
    elif system.family_id == "morphological":
        values.pop("half_window_cm1")
        values["half_window"] = converted_window
    return values


def run_baseline_system(system: Phase3System, spectrum: Spectrum1D) -> BaselineRunResult:
    if not isinstance(system, Phase3System) or system.task_line is not TaskLine.BASELINE_CORRECTION:
        raise BaselineWrapperError("system", "must be a baseline-correction Phase3System")
    if not isinstance(spectrum, Spectrum1D):
        raise BaselineWrapperError("spectrum", "must be Spectrum1D")
    converted, window_error = _converted_window(system, spectrum)
    if window_error is not None:
        return _terminal(
            system,
            spectrum,
            BaselineRunStatus.NOT_APPLICABLE,
            diagnostics={"point_count": spectrum.axis_cm1.size},
            error_code=window_error,
            error_message="physical half-window cannot map to a valid index half-window",
        )
    backend_name = "mor" if system.family_id == "morphological" else system.method_id
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            baseline, params = getattr(Baseline(x_data=spectrum.axis_cm1), backend_name)(
                spectrum.intensity,
                **_backend_kwargs(system, converted),
            )
        baseline_array = np.asarray(baseline, dtype="<f8")
        if baseline_array.shape != spectrum.intensity.shape or not np.isfinite(baseline_array).all():
            raise BaselineWrapperError("baseline", "backend returned an invalid array")
        corrected = np.asarray(spectrum.intensity - baseline_array, dtype="<f8")
        captured = tuple(
            CapturedWarning(type(item.message).__name__, str(item.message)) for item in caught
        )
        diagnostics: dict[str, object] = {
            "converted_half_window": converted,
            "point_count": spectrum.axis_cm1.size,
        }
        history = np.asarray(params.get("tol_history", ()), dtype="<f8")
        if history.size:
            diagnostics["iteration_count"] = int(history.size)
            diagnostics["tol_last"] = float(history[-1])
            diagnostics["tol_threshold"] = float(system.hyperparameters["tol"])
        if "half_window" in params:
            diagnostics["returned_half_window"] = int(params["half_window"])
        if "alpha" in params:
            diagnostics["alpha_sha256"] = _array_sha256(np.asarray(params["alpha"], dtype="<f8"))
        if "signal" in params:
            diagnostics["signal_sha256"] = _array_sha256(np.asarray(params["signal"], dtype="<f8"))
        convergence_warning = any(warning.category == "ParameterWarning" for warning in captured)
        convergence_failed = convergence_warning or (
            history.size > 0
            and float(history[-1]) > float(system.hyperparameters["tol"])
        )
        if convergence_failed:
            status = BaselineRunStatus.FAILED_CONVERGENCE
            error_code = "convergence_evidence_failed"
            error_message = "warning or tolerance history indicates nonconvergence"
        elif captured:
            status = BaselineRunStatus.COMPLETE_WITH_WARNING
            error_code = None
            error_message = None
        else:
            status = BaselineRunStatus.COMPLETE
            error_code = None
            error_message = None
        return _terminal(
            system,
            spectrum,
            status,
            baseline=baseline_array,
            corrected=corrected,
            warnings=captured,
            diagnostics=diagnostics,
            error_code=error_code,
            error_message=error_message,
        )
    except Exception as error:
        return _terminal(
            system,
            spectrum,
            BaselineRunStatus.FAILED_RUNTIME,
            diagnostics={"converted_half_window": converted, "point_count": spectrum.axis_cm1.size},
            error_code=type(error).__name__,
            error_message=str(error) or type(error).__name__,
        )


__all__ = [
    "BaselineRunResult",
    "BaselineRunStatus",
    "BaselineWrapperError",
    "CapturedWarning",
    "run_baseline_system",
]
