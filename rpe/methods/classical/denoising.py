from __future__ import annotations

import hashlib
import json
import math
import warnings as python_warnings
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import pywt
from scipy.linalg import solveh_banded
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA, TruncatedSVD

from rpe.evaluation import Spectrum1D
from rpe.methods.catalog import Phase3System, TaskLine


_STATE_DOMAIN = b"rpe-phase3-denoising-state-v1\0"
_CONTEXT_DOMAIN = b"rpe-phase3-denoising-fit-context-v1\0"
_STATELESS_FAMILIES = frozenset(
    {"savitzky_golay", "wavelet", "whittaker_smoothing"}
)
_FITTED_FAMILIES = frozenset({"pca_reconstruction", "svd_reconstruction"})


class DenoisingWrapperError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class DenoisingRunStatus(str, Enum):
    COMPLETE = "complete"
    COMPLETE_WITH_WARNING = "complete_with_warning"
    NOT_APPLICABLE = "not_applicable"
    FAILED_DOMAIN = "failed_domain"
    FAILED_FIT = "failed_fit"
    FAILED_RUNTIME = "failed_runtime"


@dataclass(frozen=True)
class DenoisingWarning:
    category: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.category, str) or not self.category:
            raise DenoisingWrapperError("warning.category", "must be nonempty")
        if not isinstance(self.message, str) or not self.message:
            raise DenoisingWrapperError("warning.message", "must be nonempty")


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


def _array(value: object, *, ndim: int, path: str) -> np.ndarray:
    current = np.ascontiguousarray(value, dtype="<f8").copy()
    if current.ndim != ndim or current.size == 0:
        raise DenoisingWrapperError(path, f"must be a nonempty {ndim}-D array")
    if not np.isfinite(current).all():
        raise DenoisingWrapperError(path, "must be finite")
    current.setflags(write=False)
    return current


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _freeze_value(path: str, value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DenoisingWrapperError(path, "must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_value(f"{path}[{index}]", item)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        return _freeze_mapping(path, value)
    raise DenoisingWrapperError(path, "must be JSON-compatible")


def _freeze_mapping(path: str, value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise DenoisingWrapperError(path, "must be a mapping")
    frozen: dict[str, object] = {}
    for key, item in sorted(value.items()):
        if not isinstance(key, str) or not key:
            raise DenoisingWrapperError(path, "keys must be nonempty strings")
        frozen[key] = _freeze_value(f"{path}.{key}", item)
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class DenoisingFitContext:
    split_id: str
    representation_id: str
    record_ids: tuple[str, ...]
    context_sha256: str = ""

    def __post_init__(self) -> None:
        for name in ("split_id", "representation_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DenoisingWrapperError(name, "must be nonempty")
        if (
            not isinstance(self.record_ids, tuple)
            or not self.record_ids
            or any(not isinstance(value, str) or not value for value in self.record_ids)
            or len(set(self.record_ids)) != len(self.record_ids)
        ):
            raise DenoisingWrapperError(
                "record IDs", "must be a nonempty unique string tuple"
            )
        document = {
            "record_ids": list(self.record_ids),
            "representation_id": self.representation_id,
            "split_id": self.split_id,
        }
        expected = hashlib.sha256(_CONTEXT_DOMAIN + _canonical(document)).hexdigest()
        if self.context_sha256 and self.context_sha256 != expected:
            raise DenoisingWrapperError("context_sha256", "does not match context")
        object.__setattr__(self, "context_sha256", expected)


@dataclass(frozen=True)
class DenoisingRunResult:
    system_id: str
    family_id: str
    method_id: str
    spectrum_id: str
    status: DenoisingRunStatus
    denoised_intensity: np.ndarray | None
    output_sha256: str | None
    warnings: tuple[DenoisingWarning, ...]
    diagnostics: Mapping[str, object]
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.status, DenoisingRunStatus):
            raise DenoisingWrapperError("status", "must be DenoisingRunStatus")
        for name in ("system_id", "family_id", "method_id", "spectrum_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DenoisingWrapperError(name, "must be nonempty")
        successful = self.status in {
            DenoisingRunStatus.COMPLETE,
            DenoisingRunStatus.COMPLETE_WITH_WARNING,
        }
        if successful:
            if self.denoised_intensity is None:
                raise DenoisingWrapperError("denoised_intensity", "is required")
            output = _array(self.denoised_intensity, ndim=1, path="denoised_intensity")
            digest = _array_sha(output)
            if self.output_sha256 is not None and self.output_sha256 != digest:
                raise DenoisingWrapperError("output_sha256", "does not match output")
            object.__setattr__(self, "denoised_intensity", output)
            object.__setattr__(self, "output_sha256", digest)
        elif self.denoised_intensity is not None or self.output_sha256 is not None:
            raise DenoisingWrapperError("output", "failure status must not carry output")
        if not isinstance(self.warnings, tuple) or any(
            not isinstance(value, DenoisingWarning) for value in self.warnings
        ):
            raise DenoisingWrapperError("warnings", "must be a warning tuple")
        object.__setattr__(self, "diagnostics", _freeze_mapping("diagnostics", self.diagnostics))


@dataclass(frozen=True)
class FittedDenoiser:
    system_id: str
    family_id: str
    method_id: str
    n_components: int
    axis_sha256: str
    training_record_ledger_sha256: str
    training_matrix_sha256: str
    context_sha256: str
    mean: np.ndarray | None
    components: np.ndarray
    explained_variance: np.ndarray
    warnings: tuple[DenoisingWarning, ...]
    state_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "system_id",
            "family_id",
            "method_id",
            "axis_sha256",
            "training_record_ledger_sha256",
            "training_matrix_sha256",
            "context_sha256",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise DenoisingWrapperError(name, "must be nonempty")
        if isinstance(self.n_components, bool) or not isinstance(self.n_components, int) or self.n_components <= 0:
            raise DenoisingWrapperError("n_components", "must be positive")
        components = _array(self.components, ndim=2, path="components")
        explained = _array(self.explained_variance, ndim=1, path="explained_variance")
        if components.shape[0] != self.n_components or explained.size != self.n_components:
            raise DenoisingWrapperError("fitted state shape", "does not match n_components")
        mean = None if self.mean is None else _array(self.mean, ndim=1, path="mean")
        if mean is not None and mean.size != components.shape[1]:
            raise DenoisingWrapperError("mean", "does not match component width")
        if not isinstance(self.warnings, tuple):
            raise DenoisingWrapperError("warnings", "must be a tuple")
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "explained_variance", explained)
        object.__setattr__(self, "mean", mean)
        document = {
            "axis_sha256": self.axis_sha256,
            "components_sha256": _array_sha(components),
            "context_sha256": self.context_sha256,
            "explained_variance_sha256": _array_sha(explained),
            "family_id": self.family_id,
            "mean_sha256": None if mean is None else _array_sha(mean),
            "method_id": self.method_id,
            "n_components": self.n_components,
            "system_id": self.system_id,
            "training_matrix_sha256": self.training_matrix_sha256,
            "training_record_ledger_sha256": self.training_record_ledger_sha256,
            "warnings": [
                {"category": value.category, "message": value.message}
                for value in self.warnings
            ],
        }
        expected = hashlib.sha256(_STATE_DOMAIN + _canonical(document)).hexdigest()
        if self.state_sha256 and self.state_sha256 != expected:
            raise DenoisingWrapperError("state_sha256", "does not match state")
        object.__setattr__(self, "state_sha256", expected)


class DenoisingFitError(DenoisingWrapperError):
    def __init__(self, path: str, reason: str, status: DenoisingRunStatus) -> None:
        self.status = status
        super().__init__(path, reason)


def _warning_tuple(caught: Sequence[python_warnings.WarningMessage]) -> tuple[DenoisingWarning, ...]:
    return tuple(
        DenoisingWarning(type(value.message).__name__, str(value.message))
        for value in caught
    )


def _result(
    system: Phase3System,
    spectrum: Spectrum1D,
    status: DenoisingRunStatus,
    *,
    output: np.ndarray | None = None,
    warnings: tuple[DenoisingWarning, ...] = (),
    diagnostics: Mapping[str, object] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
) -> DenoisingRunResult:
    return DenoisingRunResult(
        system_id=system.system_id,
        family_id=system.family_id,
        method_id=system.method_id,
        spectrum_id=spectrum.spectrum_id,
        status=status,
        denoised_intensity=output,
        output_sha256=None,
        warnings=warnings,
        diagnostics={} if diagnostics is None else diagnostics,
        error_code=error_code,
        error_message=error_message,
    )


def _validate_system(system: Phase3System, families: frozenset[str]) -> None:
    if (
        not isinstance(system, Phase3System)
        or system.task_line is not TaskLine.DENOISING
        or system.family_id not in families
    ):
        raise DenoisingWrapperError("system", "is outside the requested denoising API")


def _wavelet_noise_sigma(detail: np.ndarray) -> float:
    values = np.asarray(detail, dtype="<f8")
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise DenoisingWrapperError("wavelet detail", "must be finite nonempty 1-D")
    median = float(np.median(values))
    return float(np.median(np.abs(values - median)) / 0.6744897501960817)


def _wavelet_thresholds(
    details: Sequence[np.ndarray],
    *,
    sigma: float,
    strategy: str,
    signal_size: int,
) -> tuple[float, ...]:
    if not math.isfinite(sigma) or sigma < 0.0:
        raise DenoisingWrapperError("wavelet sigma", "must be finite and nonnegative")
    if isinstance(signal_size, bool) or not isinstance(signal_size, int) or signal_size <= 0:
        raise DenoisingWrapperError("signal_size", "must be positive")
    if strategy == "universal_mad":
        threshold = sigma * math.sqrt(2.0 * math.log(signal_size))
        return tuple(threshold for _ in details)
    if strategy != "bayes_shrink":
        raise DenoisingWrapperError("threshold_strategy", "is unsupported")
    thresholds = []
    for detail in details:
        values = np.asarray(detail, dtype="<f8")
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise DenoisingWrapperError("wavelet detail", "must be finite nonempty 1-D")
        variance = float(np.mean(values * values))
        signal_sigma = math.sqrt(max(variance - sigma * sigma, 0.0))
        thresholds.append(math.inf if signal_sigma == 0.0 else sigma * sigma / signal_sigma)
    return tuple(thresholds)


def _wavelet_output(system: Phase3System, spectrum: Spectrum1D) -> tuple[np.ndarray, Mapping[str, object]]:
    values = system.hyperparameters
    wavelet = pywt.Wavelet(str(values["wavelet"]))
    level = min(
        int(values["max_level"]),
        pywt.dwt_max_level(spectrum.intensity.size, wavelet.dec_len),
    )
    if level < 1:
        raise DenoisingFitError(
            "wavelet level",
            "maximum feasible level is zero",
            DenoisingRunStatus.NOT_APPLICABLE,
        )
    backend_input = np.array(spectrum.intensity, dtype="<f8", copy=True)
    coefficients = pywt.wavedec(
        backend_input,
        wavelet,
        mode=str(values["extension_mode"]),
        level=level,
    )
    details = coefficients[1:]
    sigma = _wavelet_noise_sigma(np.asarray(details[-1], dtype="<f8"))
    thresholds = _wavelet_thresholds(
        tuple(np.asarray(value, dtype="<f8") for value in details),
        sigma=sigma,
        strategy=str(values["threshold_strategy"]),
        signal_size=spectrum.intensity.size,
    )
    filtered = [np.asarray(coefficients[0], dtype="<f8")]
    for detail, threshold in zip(details, thresholds, strict=True):
        if math.isinf(threshold):
            current = np.zeros_like(detail, dtype="<f8")
        else:
            current = pywt.threshold(
                np.asarray(detail, dtype="<f8"),
                threshold,
                mode=str(values["threshold_mode"]),
            )
        filtered.append(np.asarray(current, dtype="<f8"))
    reconstructed = np.asarray(
        pywt.waverec(filtered, wavelet, mode=str(values["extension_mode"])),
        dtype="<f8",
    )[: spectrum.intensity.size]
    if reconstructed.shape != spectrum.intensity.shape or not np.isfinite(reconstructed).all():
        raise DenoisingWrapperError("wavelet output", "has invalid shape or values")
    return reconstructed, {
        "level": level,
        "noise_sigma": sigma,
        "thresholds": tuple(
            "infinity" if math.isinf(value) else value for value in thresholds
        ),
    }


def _whittaker_banded_matrix(size: int, lam: float) -> np.ndarray:
    if isinstance(size, bool) or not isinstance(size, int) or size < 3:
        raise DenoisingWrapperError("point count", "must be at least three")
    if not math.isfinite(lam) or lam <= 0.0:
        raise DenoisingWrapperError("lambda", "must be finite and positive")
    banded = np.zeros((3, size), dtype="<f8")
    banded[0] = 1.0
    for start in range(size - 2):
        banded[0, start] += lam
        banded[0, start + 1] += 4.0 * lam
        banded[0, start + 2] += lam
        banded[1, start] -= 2.0 * lam
        banded[1, start + 1] -= 2.0 * lam
        banded[2, start] += lam
    return banded


def _whittaker_output(system: Phase3System, spectrum: Spectrum1D) -> np.ndarray:
    size = spectrum.intensity.size
    if size < 3:
        raise DenoisingFitError(
            "point count",
            "Whittaker smoothing requires at least three points",
            DenoisingRunStatus.NOT_APPLICABLE,
        )
    lam = float(system.hyperparameters["lambda"])
    banded = _whittaker_banded_matrix(size, lam)
    return np.asarray(
        solveh_banded(
            banded,
            spectrum.intensity,
            lower=True,
            overwrite_ab=False,
            overwrite_b=False,
            check_finite=True,
        ),
        dtype="<f8",
    )


def run_stateless_denoising_system(
    system: Phase3System, spectrum: Spectrum1D
) -> DenoisingRunResult:
    _validate_system(system, _STATELESS_FAMILIES)
    if not isinstance(spectrum, Spectrum1D):
        raise DenoisingWrapperError("spectrum", "must be Spectrum1D")
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            diagnostics: Mapping[str, object]
            if system.family_id == "savitzky_golay":
                window = int(system.hyperparameters["window_length"])
                if spectrum.intensity.size < window:
                    raise DenoisingFitError(
                        "point count",
                        "is shorter than the frozen SG window",
                        DenoisingRunStatus.NOT_APPLICABLE,
                    )
                output = np.asarray(
                    savgol_filter(spectrum.intensity, **dict(system.hyperparameters)),
                    dtype="<f8",
                )
                diagnostics = {"point_count": spectrum.intensity.size}
            elif system.family_id == "wavelet":
                output, diagnostics = _wavelet_output(system, spectrum)
            else:
                output = _whittaker_output(system, spectrum)
                diagnostics = {
                    "difference_order": 2,
                    "lambda": float(system.hyperparameters["lambda"]),
                }
        captured = _warning_tuple(caught)
        status = (
            DenoisingRunStatus.COMPLETE_WITH_WARNING
            if captured
            else DenoisingRunStatus.COMPLETE
        )
        return _result(
            system,
            spectrum,
            status,
            output=output,
            warnings=captured,
            diagnostics=diagnostics,
        )
    except DenoisingFitError as error:
        return _result(
            system,
            spectrum,
            error.status,
            error_code=error.path,
            error_message=error.reason,
        )
    except Exception as error:
        return _result(
            system,
            spectrum,
            DenoisingRunStatus.FAILED_RUNTIME,
            error_code=type(error).__name__,
            error_message=str(error) or type(error).__name__,
        )


def _training_matrix(
    training_spectra: tuple[Spectrum1D, ...], context: DenoisingFitContext
) -> tuple[np.ndarray, str, str]:
    if not isinstance(training_spectra, tuple) or not training_spectra:
        raise DenoisingFitError(
            "training_spectra", "must be nonempty", DenoisingRunStatus.FAILED_DOMAIN
        )
    if len(training_spectra) != len(context.record_ids):
        raise DenoisingFitError(
            "record IDs",
            "count does not match training spectra",
            DenoisingRunStatus.FAILED_DOMAIN,
        )
    observed_ids = tuple(value.spectrum_id for value in training_spectra)
    if observed_ids != context.record_ids:
        raise DenoisingFitError(
            "record IDs",
            "do not match ordered training spectra",
            DenoisingRunStatus.FAILED_DOMAIN,
        )
    first_axis = training_spectra[0].axis_cm1
    if any(not np.array_equal(value.axis_cm1, first_axis) for value in training_spectra[1:]):
        raise DenoisingFitError(
            "training axis",
            "must be exactly shared",
            DenoisingRunStatus.FAILED_DOMAIN,
        )
    matrix = np.ascontiguousarray(
        [value.intensity for value in training_spectra], dtype="<f8"
    )
    ledger_payload = b"".join(
        _canonical({"record_id": value}) for value in context.record_ids
    )
    return matrix, _array_sha(first_axis), hashlib.sha256(ledger_payload).hexdigest()


def fit_denoising_system(
    system: Phase3System,
    training_spectra: tuple[Spectrum1D, ...],
    context: DenoisingFitContext,
) -> FittedDenoiser:
    _validate_system(system, _FITTED_FAMILIES)
    if not isinstance(context, DenoisingFitContext):
        raise DenoisingWrapperError("context", "must be DenoisingFitContext")
    matrix, axis_sha, ledger_sha = _training_matrix(training_spectra, context)
    components = int(system.hyperparameters["n_components"])
    if components > min(matrix.shape):
        raise DenoisingFitError(
            "n_components",
            "exceeds the frozen training matrix domain",
            DenoisingRunStatus.NOT_APPLICABLE,
        )
    try:
        with python_warnings.catch_warnings(record=True) as caught:
            python_warnings.simplefilter("always")
            if system.family_id == "pca_reconstruction":
                model = PCA(**dict(system.hyperparameters))
                model.fit(matrix)
                mean = np.asarray(model.mean_, dtype="<f8")
                fitted_components = np.asarray(model.components_, dtype="<f8")
                explained = np.asarray(model.explained_variance_, dtype="<f8")
            else:
                model = TruncatedSVD(**dict(system.hyperparameters))
                model.fit(matrix)
                mean = None
                fitted_components = np.asarray(model.components_, dtype="<f8")
                explained = np.asarray(model.explained_variance_, dtype="<f8")
        return FittedDenoiser(
            system_id=system.system_id,
            family_id=system.family_id,
            method_id=system.method_id,
            n_components=components,
            axis_sha256=axis_sha,
            training_record_ledger_sha256=ledger_sha,
            training_matrix_sha256=_array_sha(matrix),
            context_sha256=context.context_sha256,
            mean=mean,
            components=fitted_components,
            explained_variance=explained,
            warnings=_warning_tuple(caught),
        )
    except DenoisingFitError:
        raise
    except Exception as error:
        raise DenoisingFitError(
            type(error).__name__,
            str(error) or type(error).__name__,
            DenoisingRunStatus.FAILED_FIT,
        ) from error


def transform_fitted_denoiser(
    fitted: FittedDenoiser, spectrum: Spectrum1D
) -> DenoisingRunResult:
    if not isinstance(fitted, FittedDenoiser):
        raise DenoisingWrapperError("fitted", "must be FittedDenoiser")
    if not isinstance(spectrum, Spectrum1D):
        raise DenoisingWrapperError("spectrum", "must be Spectrum1D")
    pseudo_system = Phase3System(
        system_id=fitted.system_id,
        task_line=TaskLine.DENOISING,
        family_id=fitted.family_id,
        method_id=fitted.method_id,
        backend={},
        hyperparameters={},
        input_contract={},
        output_contract={},
        ordered_composition=(),
        method_seed=None,
        metric_eligibility=(),
        downstream_eligibility=(),
        protocol_eligibility=(),
        availability=None,  # type: ignore[arg-type]
        evidence={},
    )
    if _array_sha(spectrum.axis_cm1) != fitted.axis_sha256:
        return _result(
            pseudo_system,
            spectrum,
            DenoisingRunStatus.FAILED_DOMAIN,
            error_code="axis_identity_mismatch",
            error_message="transform axis does not match fitted state",
        )
    try:
        centered = spectrum.intensity if fitted.mean is None else spectrum.intensity - fitted.mean
        scores = centered @ fitted.components.T
        output = scores @ fitted.components
        if fitted.mean is not None:
            output = output + fitted.mean
        status = (
            DenoisingRunStatus.COMPLETE_WITH_WARNING
            if fitted.warnings
            else DenoisingRunStatus.COMPLETE
        )
        return _result(
            pseudo_system,
            spectrum,
            status,
            output=np.asarray(output, dtype="<f8"),
            warnings=fitted.warnings,
            diagnostics={"fitted_state_sha256": fitted.state_sha256},
        )
    except Exception as error:
        return _result(
            pseudo_system,
            spectrum,
            DenoisingRunStatus.FAILED_RUNTIME,
            error_code=type(error).__name__,
            error_message=str(error) or type(error).__name__,
        )


__all__ = [
    "DenoisingFitContext",
    "DenoisingFitError",
    "DenoisingRunResult",
    "DenoisingRunStatus",
    "DenoisingWarning",
    "DenoisingWrapperError",
    "FittedDenoiser",
    "fit_denoising_system",
    "run_stateless_denoising_system",
    "transform_fitted_denoiser",
]
