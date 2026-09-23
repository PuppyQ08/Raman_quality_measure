from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field

import numpy as np

from rpe.evaluation import Spectrum1D
from rpe.perturb.contracts import (
    AxisBehavior,
    PerturbationContext,
    PerturbationContractError,
    PerturbationResult,
    PerturbationState,
    derive_perturbed_spectrum_id,
    validate_perturbation_result,
)
from rpe.perturb.sweep import (
    PerturbationSweepConfig,
    PerturbationSweepConfigError,
    is_frozen_alpha,
    validate_perturbation_sweep_config,
)


AXIS_BEHAVIOR = AxisBehavior.PRESERVE
BASELINE_SOURCE = "explicit_external_reference"
_HEX_DIGITS = frozenset("0123456789abcdef")


class BaselineResidualError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _config_error_from_path(path: str, reason: str) -> BaselineResidualError:
    return BaselineResidualError(f"config.{path}", reason)


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise BaselineResidualError(path, "must be a nonempty string")
    return value


def _lower_hex_64(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise BaselineResidualError(
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _positive_finite_float(path: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise BaselineResidualError(path, "must be a positive finite real number")
    return float(value)


def _read_only_float64_vector(path: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise BaselineResidualError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise BaselineResidualError(
            f"{path} dtype",
            "must be little-endian float64",
        )
    if value.ndim != 1:
        raise BaselineResidualError(
            f"{path} dimension",
            "must be one-dimensional",
        )
    if value.size == 0:
        raise BaselineResidualError(f"{path} shape", "must be nonempty")
    if not np.isfinite(value).all():
        raise BaselineResidualError(
            f"{path} finite",
            "contains non-finite values",
        )
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(
        array.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()


def _scaled_rms(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    scaled = values / scale
    return float(scale * np.sqrt(np.mean(scaled * scaled)))


def _length_prefixed_utf8(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


@dataclass(frozen=True)
class BaselineReference:
    spectrum_id: str
    provenance_id: str
    axis_cm1: np.ndarray = field(compare=False, repr=False)
    baseline_intensity: np.ndarray = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "spectrum_id",
            _nonempty_string("spectrum_id", self.spectrum_id),
        )
        object.__setattr__(
            self,
            "provenance_id",
            _nonempty_string("provenance_id", self.provenance_id),
        )
        axis = _read_only_float64_vector("axis_cm1", self.axis_cm1)
        baseline = _read_only_float64_vector(
            "baseline_intensity",
            self.baseline_intensity,
        )
        if axis.shape != baseline.shape:
            raise BaselineResidualError(
                "array length",
                "axis and baseline must have equal shape",
            )
        if not np.all(np.diff(axis) > 0.0):
            raise BaselineResidualError(
                "axis_cm1",
                "must be strictly increasing",
            )
        baseline_rms = _scaled_rms(baseline)
        if not math.isfinite(baseline_rms) or baseline_rms <= 0.0:
            raise BaselineResidualError(
                "baseline RMS",
                "must be positive and finite",
            )
        object.__setattr__(self, "axis_cm1", axis)
        object.__setattr__(self, "baseline_intensity", baseline)


def _state_digest(
    perturbation_id: str,
    spectrum_id: str,
    sweep_config_sha256: str,
    source_axis_sha256: str,
    source_intensity_sha256: str,
    baseline_provenance_id: str,
    baseline_sha256: str,
    baseline_rms: float,
    baseline_intensity: np.ndarray,
) -> str:
    perturbation_name = _nonempty_string("perturbation_id", perturbation_id)
    spectrum_name = _nonempty_string("spectrum_id", spectrum_id)
    config_hash = _lower_hex_64("sweep_config_sha256", sweep_config_sha256)
    axis_hash = _lower_hex_64("source_axis_sha256", source_axis_sha256)
    intensity_hash = _lower_hex_64(
        "source_intensity_sha256",
        source_intensity_sha256,
    )
    provenance = _nonempty_string(
        "baseline_provenance_id",
        baseline_provenance_id,
    )
    baseline_hash = _lower_hex_64("baseline_sha256", baseline_sha256)
    rms = _positive_finite_float("baseline_rms", baseline_rms)
    baseline = _read_only_float64_vector(
        "baseline_intensity",
        baseline_intensity,
    )
    payload = (
        b"rpe-baseline-residual-state-digest-v1\0"
        + _length_prefixed_utf8(perturbation_name)
        + _length_prefixed_utf8(spectrum_name)
        + bytes.fromhex(config_hash)
        + bytes.fromhex(axis_hash)
        + bytes.fromhex(intensity_hash)
        + _length_prefixed_utf8(provenance)
        + bytes.fromhex(baseline_hash)
        + struct.pack("<d", rms)
        + struct.pack("<Q", baseline.size)
        + baseline.astype("<f8", copy=False).tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BaselineResidualState:
    perturbation_id: str
    spectrum_id: str
    sweep_config_sha256: str
    state_digest: str
    source_axis_sha256: str
    source_intensity_sha256: str
    baseline_provenance_id: str
    baseline_sha256: str
    baseline_rms: float
    baseline_intensity: np.ndarray = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        perturbation_id = _nonempty_string(
            "perturbation_id",
            self.perturbation_id,
        )
        if perturbation_id not in ("p06", "p07"):
            raise BaselineResidualError(
                "perturbation_id",
                "must equal p06 or p07",
            )
        object.__setattr__(self, "perturbation_id", perturbation_id)
        object.__setattr__(
            self,
            "spectrum_id",
            _nonempty_string("spectrum_id", self.spectrum_id),
        )
        for name in (
            "sweep_config_sha256",
            "state_digest",
            "source_axis_sha256",
            "source_intensity_sha256",
            "baseline_sha256",
        ):
            object.__setattr__(self, name, _lower_hex_64(name, getattr(self, name)))
        object.__setattr__(
            self,
            "baseline_provenance_id",
            _nonempty_string(
                "baseline_provenance_id",
                self.baseline_provenance_id,
            ),
        )
        object.__setattr__(
            self,
            "baseline_rms",
            _positive_finite_float("baseline_rms", self.baseline_rms),
        )
        object.__setattr__(
            self,
            "baseline_intensity",
            _read_only_float64_vector(
                "baseline_intensity",
                self.baseline_intensity,
            ),
        )


class _BaselineResidualBase:
    perturbation_id = ""
    residual_sign = 0
    axis_behavior = AXIS_BEHAVIOR

    def __init__(
        self,
        config: PerturbationSweepConfig,
        baseline: BaselineReference,
    ) -> None:
        if not isinstance(config, PerturbationSweepConfig):
            raise BaselineResidualError(
                "config",
                "must be PerturbationSweepConfig",
            )
        try:
            validate_perturbation_sweep_config(config)
        except PerturbationSweepConfigError as exc:
            raise _config_error_from_path(exc.path, exc.reason) from exc
        if not isinstance(baseline, BaselineReference):
            raise BaselineResidualError(
                "baseline",
                "must be BaselineReference",
            )
        self._config = config
        self._baseline = BaselineReference(
            spectrum_id=baseline.spectrum_id,
            provenance_id=baseline.provenance_id,
            axis_cm1=baseline.axis_cm1,
            baseline_intensity=baseline.baseline_intensity,
        )
        self._baseline_sha256 = _sha256_bytes(self._baseline.baseline_intensity)
        self._baseline_rms = _scaled_rms(self._baseline.baseline_intensity)

    def _validate_context(self, context: PerturbationContext) -> None:
        if not isinstance(context, PerturbationContext):
            raise BaselineResidualError(
                "context",
                "must be PerturbationContext",
            )
        if context.sweep_id != self._config.sweep_id:
            raise BaselineResidualError(
                "context.sweep_id",
                "must match the retained sweep config",
            )
        if context.sweep_config_sha256 != self._config.sha256:
            raise BaselineResidualError(
                "context.sweep_config_sha256",
                "must match the retained sweep config",
            )
        if context.global_seed != self._config.global_seed:
            raise BaselineResidualError(
                "context.global_seed",
                "must match the retained sweep config",
            )

    def _validated_state(
        self,
        state: PerturbationState,
    ) -> BaselineResidualState:
        if not isinstance(state, BaselineResidualState):
            raise BaselineResidualError(
                "state",
                "must be BaselineResidualState",
            )
        if state.perturbation_id != self.perturbation_id:
            raise BaselineResidualError(
                "state.perturbation_id",
                f"must match {self.perturbation_id}",
            )
        if state.sweep_config_sha256 != self._config.sha256:
            raise BaselineResidualError(
                "state.sweep_config_sha256",
                "must match the retained sweep config",
            )
        state_baseline_sha256 = _sha256_bytes(state.baseline_intensity)
        if state_baseline_sha256 != state.baseline_sha256:
            raise BaselineResidualError(
                "state.baseline_intensity",
                "must match state.baseline_sha256",
            )
        state_baseline_rms = _scaled_rms(state.baseline_intensity)
        if state_baseline_rms != state.baseline_rms:
            raise BaselineResidualError(
                "state.baseline_rms",
                "must match the committed baseline intensity",
            )
        expected_digest = _state_digest(
            state.perturbation_id,
            state.spectrum_id,
            state.sweep_config_sha256,
            state.source_axis_sha256,
            state.source_intensity_sha256,
            state.baseline_provenance_id,
            state.baseline_sha256,
            state.baseline_rms,
            state.baseline_intensity,
        )
        if state.state_digest != expected_digest:
            raise BaselineResidualError(
                "state.state_digest",
                "must match the deterministic state digest",
            )
        if state.baseline_provenance_id != self._baseline.provenance_id:
            raise BaselineResidualError(
                "state.baseline_provenance_id",
                "must match the operator baseline provenance",
            )
        if state.baseline_sha256 != self._baseline_sha256:
            raise BaselineResidualError(
                "state.baseline_sha256",
                "must match the operator baseline content",
            )
        if _sha256_bytes(self._baseline.axis_cm1) != state.source_axis_sha256:
            raise BaselineResidualError(
                "state.baseline axis",
                "must match the prepared source axis",
            )
        return state

    def prepare(
        self,
        spectrum: Spectrum1D,
        context: PerturbationContext,
    ) -> BaselineResidualState:
        if not isinstance(spectrum, Spectrum1D):
            raise BaselineResidualError("spectrum", "must be Spectrum1D")
        self._validate_context(context)
        if spectrum.spectrum_id != self._baseline.spectrum_id:
            raise BaselineResidualError(
                "baseline spectrum_id",
                "must match the source spectrum_id",
            )
        if not np.array_equal(spectrum.axis_cm1, self._baseline.axis_cm1):
            raise BaselineResidualError(
                "baseline axis",
                "must exactly match the source axis",
            )
        source_axis_sha256 = _sha256_bytes(spectrum.axis_cm1)
        source_intensity_sha256 = _sha256_bytes(spectrum.intensity)
        state_digest = _state_digest(
            self.perturbation_id,
            spectrum.spectrum_id,
            self._config.sha256,
            source_axis_sha256,
            source_intensity_sha256,
            self._baseline.provenance_id,
            self._baseline_sha256,
            self._baseline_rms,
            self._baseline.baseline_intensity,
        )
        return BaselineResidualState(
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
            sweep_config_sha256=self._config.sha256,
            state_digest=state_digest,
            source_axis_sha256=source_axis_sha256,
            source_intensity_sha256=source_intensity_sha256,
            baseline_provenance_id=self._baseline.provenance_id,
            baseline_sha256=self._baseline_sha256,
            baseline_rms=self._baseline_rms,
            baseline_intensity=self._baseline.baseline_intensity,
        )

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        if not isinstance(spectrum, Spectrum1D):
            raise BaselineResidualError("spectrum", "must be Spectrum1D")
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise BaselineResidualError(
                "alpha",
                "must be one of the frozen alpha values",
            )
        alpha_value = float(alpha)
        if spectrum.spectrum_id != validated_state.spectrum_id:
            raise BaselineResidualError(
                "source spectrum_id",
                "must match the prepared state",
            )
        if spectrum.intensity.shape != validated_state.baseline_intensity.shape:
            raise BaselineResidualError(
                "source shape",
                "must match the committed baseline",
            )
        if _sha256_bytes(spectrum.axis_cm1) != validated_state.source_axis_sha256:
            raise BaselineResidualError(
                "source axis",
                "must match the prepared source axis",
            )
        if (
            _sha256_bytes(spectrum.intensity)
            != validated_state.source_intensity_sha256
        ):
            raise BaselineResidualError(
                "source intensity",
                "must match the prepared source intensity",
            )
        if alpha_value == self._config.identity_alpha:
            output_intensity = np.array(spectrum.intensity, dtype="<f8", copy=True)
            realized_residual_rmse = 0.0
        else:
            with np.errstate(over="ignore", invalid="ignore", under="ignore"):
                residual = (self.residual_sign * alpha_value) * (
                    validated_state.baseline_intensity
                )
                output_intensity = np.asarray(
                    spectrum.intensity + residual,
                    dtype="<f8",
                )
            if not np.isfinite(residual).all() or not np.isfinite(output_intensity).all():
                raise BaselineResidualError(
                    "finite output",
                    "residual and output must remain finite",
                )
            if np.array_equal(output_intensity, spectrum.intensity):
                raise BaselineResidualError(
                    "positive perturbation",
                    "rounds entirely to identity",
                )
            realized_residual = np.asarray(
                output_intensity - spectrum.intensity,
                dtype="<f8",
            )
            realized_residual_rmse = _scaled_rms(realized_residual)
            if not math.isfinite(realized_residual_rmse):
                raise BaselineResidualError(
                    "finite output",
                    "realized residual RMSE must remain finite",
                )
        output = Spectrum1D(
            spectrum_id=derive_perturbed_spectrum_id(
                spectrum.spectrum_id,
                self.perturbation_id,
                alpha_value,
                validated_state.state_digest,
                self._config.sha256,
            ),
            sample_id=spectrum.sample_id,
            axis_cm1=np.array(spectrum.axis_cm1, dtype="<f8", copy=True),
            intensity=np.array(output_intensity, dtype="<f8", copy=True),
        )
        result = PerturbationResult(
            source_spectrum_id=spectrum.spectrum_id,
            perturbation_id=self.perturbation_id,
            alpha=alpha_value,
            output=output,
            axis_behavior=self.axis_behavior,
            state_digest=validated_state.state_digest,
            axis_changed=False,
            intensity_changed=not np.array_equal(
                spectrum.intensity,
                output.intensity,
            ),
            diagnostics={
                "baseline_provenance_id": validated_state.baseline_provenance_id,
                "baseline_sha256": validated_state.baseline_sha256,
                "baseline_rms": validated_state.baseline_rms,
                "residual_sign": self.residual_sign,
                "requested_alpha": alpha_value,
                "realized_residual_rmse": realized_residual_rmse,
                "axis_preserved": True,
                "baseline_source": BASELINE_SOURCE,
            },
        )
        try:
            validate_perturbation_result(
                spectrum,
                validated_state,
                result,
                self._config,
            )
        except PerturbationContractError as exc:
            raise BaselineResidualError(exc.path, exc.reason) from exc
        return result


class P6BaselineUndercorrection(_BaselineResidualBase):
    perturbation_id = "p06"
    residual_sign = 1


class P7BaselineOvercorrection(_BaselineResidualBase):
    perturbation_id = "p07"
    residual_sign = -1


__all__ = [
    "AXIS_BEHAVIOR",
    "BASELINE_SOURCE",
    "BaselineReference",
    "BaselineResidualError",
    "BaselineResidualState",
    "P6BaselineUndercorrection",
    "P7BaselineOvercorrection",
]
