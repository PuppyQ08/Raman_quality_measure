from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

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


P11_PERTURBATION_ID = "p11"
P12_PERTURBATION_ID = "p12"
AXIS_BEHAVIOR = AxisBehavior.TRANSFORM
_HEX_DIGITS = frozenset("0123456789abcdef")


class _AxisTransformError(ValueError):
    label = "AxisTransformError"

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class P11AxisTransformError(_AxisTransformError):
    label = "P11AxisTransformError"


class P12AxisTransformError(_AxisTransformError):
    label = "P12AxisTransformError"


def _error_for(perturbation_id: str) -> type[_AxisTransformError]:
    if perturbation_id == P11_PERTURBATION_ID:
        return P11AxisTransformError
    if perturbation_id == P12_PERTURBATION_ID:
        return P12AxisTransformError
    raise ValueError(f"unsupported perturbation_id {perturbation_id!r}")


def _raise(perturbation_id: str, path: str, reason: str) -> None:
    raise _error_for(perturbation_id)(path, reason)


def _config_error_from_path(
    perturbation_id: str,
    path: str,
    reason: str,
) -> _AxisTransformError:
    return _error_for(perturbation_id)(f"config.{path}", reason)


def _nonempty_string(
    perturbation_id: str,
    path: str,
    value: object,
) -> str:
    if not isinstance(value, str) or value == "":
        _raise(perturbation_id, path, "must be a nonempty string")
    return value


def _lower_hex_64(
    perturbation_id: str,
    path: str,
    value: object,
) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        _raise(
            perturbation_id,
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _read_only_float64_vector(
    perturbation_id: str,
    path: str,
    value: object,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        _raise(perturbation_id, path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        _raise(perturbation_id, f"{path} dtype", "must be little-endian float64")
    if value.ndim != 1:
        _raise(perturbation_id, f"{path} dimension", "must be one-dimensional")
    if value.size == 0:
        _raise(perturbation_id, f"{path} shape", "must be nonempty")
    if not np.isfinite(value).all():
        _raise(perturbation_id, f"{path} finite", "contains non-finite values")
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(array.astype("<f8", copy=False).tobytes(order="C")).hexdigest()


def _state_digest(
    perturbation_id: str,
    spectrum_id: str,
    sweep_config_sha256: str,
    source_axis_sha256: str,
    source_intensity_sha256: str,
) -> str:
    payload = (
        b"rpe-axis-transform-state-digest-v1\0"
        + struct.pack("<Q", len(perturbation_id.encode("utf-8")))
        + perturbation_id.encode("utf-8")
        + struct.pack("<Q", len(spectrum_id.encode("utf-8")))
        + spectrum_id.encode("utf-8")
        + bytes.fromhex(sweep_config_sha256)
        + bytes.fromhex(source_axis_sha256)
        + bytes.fromhex(source_intensity_sha256)
    )
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class P11AxisTransformState:
    perturbation_id: str
    spectrum_id: str
    sweep_config_sha256: str
    state_digest: str
    source_axis_sha256: str
    source_intensity_sha256: str

    def __post_init__(self) -> None:
        perturbation_id = _nonempty_string(
            P11_PERTURBATION_ID,
            "perturbation_id",
            self.perturbation_id,
        )
        if perturbation_id not in (P11_PERTURBATION_ID, P12_PERTURBATION_ID):
            _raise(
                P11_PERTURBATION_ID,
                "perturbation_id",
                "must equal p11 or p12",
            )
        object.__setattr__(self, "perturbation_id", perturbation_id)
        object.__setattr__(
            self,
            "spectrum_id",
            _nonempty_string(
                perturbation_id,
                "spectrum_id",
                self.spectrum_id,
            ),
        )
        object.__setattr__(
            self,
            "sweep_config_sha256",
            _lower_hex_64(
                perturbation_id,
                "sweep_config_sha256",
                self.sweep_config_sha256,
            ),
        )
        object.__setattr__(
            self,
            "state_digest",
            _lower_hex_64(
                perturbation_id,
                "state_digest",
                self.state_digest,
            ),
        )
        object.__setattr__(
            self,
            "source_axis_sha256",
            _lower_hex_64(
                perturbation_id,
                "source_axis_sha256",
                self.source_axis_sha256,
            ),
        )
        object.__setattr__(
            self,
            "source_intensity_sha256",
            _lower_hex_64(
                perturbation_id,
                "source_intensity_sha256",
                self.source_intensity_sha256,
            ),
        )


def _validated_state(
    perturbation_id: str,
    config: PerturbationSweepConfig,
    state: PerturbationState,
) -> P11AxisTransformState:
    if not isinstance(state, P11AxisTransformState):
        _raise(perturbation_id, "state", "must be P11AxisTransformState")
    if state.perturbation_id != perturbation_id:
        _raise(
            perturbation_id,
            "state.perturbation_id",
            f"must match {perturbation_id}",
        )
    if state.sweep_config_sha256 != config.sha256:
        _raise(
            perturbation_id,
            "state.sweep_config_sha256",
            "must match the retained sweep config",
        )
    expected_digest = _state_digest(
        state.perturbation_id,
        state.spectrum_id,
        state.sweep_config_sha256,
        state.source_axis_sha256,
        state.source_intensity_sha256,
    )
    if state.state_digest != expected_digest:
        _raise(
            perturbation_id,
            "state.state_digest",
            "must match the deterministic state digest",
        )
    return state


def _validate_context(
    perturbation_id: str,
    config: PerturbationSweepConfig,
    context: PerturbationContext,
) -> None:
    if not isinstance(context, PerturbationContext):
        _raise(perturbation_id, "context", "must be PerturbationContext")
    if context.sweep_id != config.sweep_id:
        _raise(
            perturbation_id,
            "context.sweep_id",
            "must match the retained sweep config",
        )
    if context.sweep_config_sha256 != config.sha256:
        _raise(
            perturbation_id,
            "context.sweep_config_sha256",
            "must match the retained sweep config",
        )
    if context.global_seed != config.global_seed:
        _raise(
            perturbation_id,
            "context.global_seed",
            "must match the retained sweep config",
        )


def _source_hashes(
    perturbation_id: str,
    spectrum: Spectrum1D,
) -> tuple[str, str]:
    axis = _read_only_float64_vector(perturbation_id, "axis_cm1", spectrum.axis_cm1)
    intensity = _read_only_float64_vector(perturbation_id, "intensity", spectrum.intensity)
    return _sha256_bytes(axis), _sha256_bytes(intensity)


def _copy_output(
    spectrum: Spectrum1D,
    axis_cm1: np.ndarray,
    intensity: np.ndarray,
    *,
    spectrum_id: str,
) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id=spectrum.sample_id,
        axis_cm1=np.array(axis_cm1, dtype="<f8", copy=True),
        intensity=np.array(intensity, dtype="<f8", copy=True),
    )


def _global_shifted_axis(
    perturbation_id: str,
    source_axis: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    delta = float(4.0 * alpha)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = source_axis + delta
    if not np.isfinite(shifted).all():
        _raise(
            perturbation_id,
            "output axis finite",
            "must remain finite after the global shift",
        )
    if np.array_equal(shifted, source_axis):
        _raise(
            perturbation_id,
            "realized axis transform",
            "must not round back to the exact source axis",
        )
    if not np.all(np.diff(shifted) > 0.0):
        _raise(
            perturbation_id,
            "axis increasing",
            "must remain strictly increasing after the global shift",
        )
    realized = float(np.max(np.abs(shifted - source_axis)))
    if not math.isfinite(realized):
        _raise(
            perturbation_id,
            "realized_max_abs_offset_cm1",
            "must be finite",
        )
    return np.asarray(shifted, dtype="<f8"), realized


def _quadratic_warp_axis(
    perturbation_id: str,
    source_axis: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    lo = float(source_axis[0])
    hi = float(source_axis[-1])
    span = hi - lo
    if not math.isfinite(lo) or not math.isfinite(hi) or not math.isfinite(span):
        _raise(
            perturbation_id,
            "axis span",
            "must be finite",
        )
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = (source_axis - lo) / span
        offsets = 4.0 * alpha * np.square(normalized)
        warped = source_axis + offsets
    if (
        not np.isfinite(normalized).all()
        or not np.isfinite(offsets).all()
        or not np.isfinite(warped).all()
    ):
        _raise(
            perturbation_id,
            "output axis finite",
            "normalized coordinates, offsets, and output must remain finite",
        )
    expected_max_shift = float(4.0 * alpha)
    realized_max_shift = float(np.max(offsets))
    if alpha > 0.0 and realized_max_shift != expected_max_shift:
        _raise(
            perturbation_id,
            "maximum shift",
            "must preserve the positive maximum shift exactly",
        )
    if alpha > 0.0 and np.array_equal(warped, source_axis):
        _raise(
            perturbation_id,
            "realized axis transform",
            "must not round back to the exact source axis",
        )
    if not np.all(np.diff(warped) > 0.0):
        _raise(
            perturbation_id,
            "axis increasing",
            "must remain strictly increasing after the quadratic warp",
        )
    if not np.all(np.diff(offsets) >= 0.0):
        _raise(
            perturbation_id,
            "offset monotonicity",
            "must remain nondecreasing",
        )
    return np.asarray(warped, dtype="<f8"), realized_max_shift


class _BaseAxisTransform:
    perturbation_id = ""
    axis_behavior = AXIS_BEHAVIOR
    transform_name = ""

    def __init__(self, config: PerturbationSweepConfig) -> None:
        if not isinstance(config, PerturbationSweepConfig):
            _raise(
                self.perturbation_id,
                "config",
                "must be PerturbationSweepConfig",
            )
        try:
            validate_perturbation_sweep_config(config)
        except PerturbationSweepConfigError as exc:
            raise _config_error_from_path(
                self.perturbation_id,
                exc.path,
                exc.reason,
            ) from exc
        self._config = config

    def prepare(
        self,
        spectrum: Spectrum1D,
        context: PerturbationContext,
    ) -> P11AxisTransformState:
        if not isinstance(spectrum, Spectrum1D):
            _raise(self.perturbation_id, "spectrum", "must be Spectrum1D")
        _validate_context(self.perturbation_id, self._config, context)
        source_axis_sha256, source_intensity_sha256 = _source_hashes(
            self.perturbation_id,
            spectrum,
        )
        state = P11AxisTransformState(
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
            sweep_config_sha256=self._config.sha256,
            state_digest=_state_digest(
                self.perturbation_id,
                spectrum.spectrum_id,
                self._config.sha256,
                source_axis_sha256,
                source_intensity_sha256,
            ),
            source_axis_sha256=source_axis_sha256,
            source_intensity_sha256=source_intensity_sha256,
        )
        return state

    def _validated_source(
        self,
        spectrum: Spectrum1D,
        state: P11AxisTransformState,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(spectrum, Spectrum1D):
            _raise(self.perturbation_id, "spectrum", "must be Spectrum1D")
        if spectrum.spectrum_id != state.spectrum_id:
            _raise(
                self.perturbation_id,
                "spectrum.spectrum_id",
                "must match the prepared state",
            )
        source_axis = _read_only_float64_vector(
            self.perturbation_id,
            "axis_cm1",
            spectrum.axis_cm1,
        )
        source_intensity = _read_only_float64_vector(
            self.perturbation_id,
            "intensity",
            spectrum.intensity,
        )
        if _sha256_bytes(source_axis) != state.source_axis_sha256:
            _raise(
                self.perturbation_id,
                "source axis",
                "must match the prepared source axis",
            )
        if _sha256_bytes(source_intensity) != state.source_intensity_sha256:
            _raise(
                self.perturbation_id,
                "source intensity",
                "must match the prepared source intensity",
            )
        return source_axis, source_intensity

    def _build_result(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: P11AxisTransformState,
        output_axis: np.ndarray,
        output_intensity: np.ndarray,
        diagnostics: dict[str, object],
    ) -> PerturbationResult:
        result = PerturbationResult(
            source_spectrum_id=spectrum.spectrum_id,
            perturbation_id=self.perturbation_id,
            alpha=alpha,
            output=_copy_output(
                spectrum,
                output_axis,
                output_intensity,
                spectrum_id=derive_perturbed_spectrum_id(
                    spectrum.spectrum_id,
                    self.perturbation_id,
                    alpha,
                    state.state_digest,
                    self._config.sha256,
                ),
            ),
            axis_behavior=self.axis_behavior,
            state_digest=state.state_digest,
            axis_changed=not np.array_equal(spectrum.axis_cm1, output_axis),
            intensity_changed=not np.array_equal(spectrum.intensity, output_intensity),
            diagnostics=diagnostics,
        )
        try:
            validate_perturbation_result(
                spectrum,
                state,
                result,
                self._config,
            )
        except PerturbationContractError as exc:
            _raise(self.perturbation_id, exc.path, exc.reason)
        return result


class P11GlobalWavenumberShift(_BaseAxisTransform):
    perturbation_id = P11_PERTURBATION_ID
    transform_name = "global_additive_shift"

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        validated_state = _validated_state(self.perturbation_id, self._config, state)
        if not is_frozen_alpha(self._config, alpha):
            _raise(self.perturbation_id, "alpha", "must be one of the frozen alpha values")
        source_axis, source_intensity = self._validated_source(spectrum, validated_state)
        if alpha == self._config.identity_alpha:
            output_axis = np.array(source_axis, dtype="<f8", copy=True)
            realized = 0.0
        else:
            output_axis, realized = _global_shifted_axis(
                self.perturbation_id,
                source_axis,
                float(alpha),
            )
        output_intensity = np.array(source_intensity, dtype="<f8", copy=True)
        return self._build_result(
            spectrum,
            float(alpha),
            validated_state,
            output_axis,
            output_intensity,
            diagnostics={
                "transform": self.transform_name,
                "requested_max_abs_offset_cm1": float(4.0 * alpha),
                "realized_max_abs_offset_cm1": float(realized),
                "intensity_preserved": True,
                "interpolation": "none",
            },
        )


class P12QuadraticWavenumberWarp(_BaseAxisTransform):
    perturbation_id = P12_PERTURBATION_ID
    transform_name = "lower_anchored_quadratic_warp"

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        validated_state = _validated_state(self.perturbation_id, self._config, state)
        if not is_frozen_alpha(self._config, alpha):
            _raise(self.perturbation_id, "alpha", "must be one of the frozen alpha values")
        source_axis, source_intensity = self._validated_source(spectrum, validated_state)
        if alpha == self._config.identity_alpha:
            output_axis = np.array(source_axis, dtype="<f8", copy=True)
            realized = 0.0
        else:
            output_axis, realized = _quadratic_warp_axis(
                self.perturbation_id,
                source_axis,
                float(alpha),
            )
        output_intensity = np.array(source_intensity, dtype="<f8", copy=True)
        return self._build_result(
            spectrum,
            float(alpha),
            validated_state,
            output_axis,
            output_intensity,
            diagnostics={
                "transform": self.transform_name,
                "requested_max_abs_offset_cm1": float(4.0 * alpha),
                "realized_max_abs_offset_cm1": float(realized),
                "intensity_preserved": True,
                "interpolation": "none",
                "lower_anchor_cm1": float(source_axis[0]),
                "upper_anchor_cm1": float(source_axis[-1]),
                "normalized_coordinate": "endpoint_minmax",
            },
        )


__all__ = [
    "AXIS_BEHAVIOR",
    "P11AxisTransformError",
    "P11AxisTransformState",
    "P11GlobalWavenumberShift",
    "P11_PERTURBATION_ID",
    "P12AxisTransformError",
    "P12QuadraticWavenumberWarp",
    "P12_PERTURBATION_ID",
]
