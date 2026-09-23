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
    derive_state_seed_material,
    is_frozen_alpha,
    validate_perturbation_sweep_config,
)


PERTURBATION_ID = "p09"
AXIS_BEHAVIOR = AxisBehavior.PRESERVE
_HEX_DIGITS = frozenset("0123456789abcdef")


def _config_error_from_path(path: str, reason: str) -> GaussianNoiseError:
    return GaussianNoiseError(f"config.{path}", reason)


class GaussianNoiseError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise GaussianNoiseError(path, "must be a nonempty string")
    return value


def _lower_hex_64(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise GaussianNoiseError(
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _read_only_float64_vector(
    path: str,
    value: object,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise GaussianNoiseError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise GaussianNoiseError(
            f"{path} dtype",
            "must be little-endian float64",
        )
    if value.ndim != 1:
        raise GaussianNoiseError(
            f"{path} dimension",
            "must be one-dimensional",
        )
    if value.size == 0:
        raise GaussianNoiseError(
            f"{path} shape",
            "must be nonempty",
        )
    if not np.isfinite(value).all():
        raise GaussianNoiseError(
            f"{path} finite",
            "contains non-finite values",
        )
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _state_digest(
    perturbation_id: str,
    spectrum_id: str,
    sweep_config_sha256: str,
    standard_noise: np.ndarray,
) -> str:
    perturbation_name = _nonempty_string("perturbation_id", perturbation_id)
    spectrum_name = _nonempty_string("spectrum_id", spectrum_id)
    config_hash = _lower_hex_64(
        "sweep_config_sha256",
        sweep_config_sha256,
    )
    noise = _read_only_float64_vector("standard_noise", standard_noise)
    payload = (
        b"rpe-perturbation-state-digest-v1\0"
        + struct.pack("<Q", len(perturbation_name.encode("utf-8")))
        + perturbation_name.encode("utf-8")
        + struct.pack("<Q", len(spectrum_name.encode("utf-8")))
        + spectrum_name.encode("utf-8")
        + bytes.fromhex(config_hash)
        + struct.pack("<Q", noise.size)
        + noise.astype("<f8", copy=False).tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def _binding_key(state: P9GaussianNoiseState | PerturbationState) -> tuple[str, str, str, str]:
    return (
        state.perturbation_id,
        state.spectrum_id,
        state.sweep_config_sha256,
        state.state_digest,
    )


def _binding_payload(axis_cm1: np.ndarray, intensity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        _read_only_float64_vector("prepared_axis_cm1", axis_cm1),
        _read_only_float64_vector("prepared_intensity", intensity),
    )


def _scaled_rms(values: np.ndarray) -> float:
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0
    scaled = values / scale
    return float(scale * np.sqrt(np.mean(scaled * scaled)))


@dataclass(frozen=True)
class P9GaussianNoiseState:
    perturbation_id: str
    spectrum_id: str
    sweep_config_sha256: str
    state_digest: str
    standard_noise: np.ndarray

    def __post_init__(self) -> None:
        perturbation_id = _nonempty_string(
            "perturbation_id",
            self.perturbation_id,
        )
        if perturbation_id != PERTURBATION_ID:
            raise GaussianNoiseError(
                "perturbation_id",
                "must equal p09",
            )
        object.__setattr__(self, "perturbation_id", perturbation_id)
        object.__setattr__(
            self,
            "spectrum_id",
            _nonempty_string("spectrum_id", self.spectrum_id),
        )
        object.__setattr__(
            self,
            "sweep_config_sha256",
            _lower_hex_64(
                "sweep_config_sha256",
                self.sweep_config_sha256,
            ),
        )
        noise = _read_only_float64_vector(
            "standard_noise",
            self.standard_noise,
        )
        object.__setattr__(self, "standard_noise", noise)
        object.__setattr__(
            self,
            "state_digest",
            _lower_hex_64("state_digest", self.state_digest),
        )


class P9GaussianWhiteNoise:
    perturbation_id = PERTURBATION_ID
    axis_behavior = AXIS_BEHAVIOR

    def __init__(self, config: PerturbationSweepConfig) -> None:
        if not isinstance(config, PerturbationSweepConfig):
            raise GaussianNoiseError(
                "config",
                "must be PerturbationSweepConfig",
            )
        try:
            validate_perturbation_sweep_config(config)
        except PerturbationSweepConfigError as exc:
            raise _config_error_from_path(exc.path, exc.reason) from exc
        self._config = config
        self._prepared_bindings: dict[
            tuple[str, str, str, str],
            tuple[tuple[np.ndarray, np.ndarray], ...],
        ] = {}

    def _validate_context(self, context: PerturbationContext) -> None:
        if not isinstance(context, PerturbationContext):
            raise GaussianNoiseError(
                "context",
                "must be PerturbationContext",
            )
        if context.sweep_id != self._config.sweep_id:
            raise GaussianNoiseError(
                "context.sweep_id",
                "must match the retained sweep config",
            )
        if context.sweep_config_sha256 != self._config.sha256:
            raise GaussianNoiseError(
                "context.sweep_config_sha256",
                "must match the retained sweep config",
            )
        if context.global_seed != self._config.global_seed:
            raise GaussianNoiseError(
                "context.global_seed",
                "must match the retained sweep config",
            )

    def _validated_state(self, state: PerturbationState) -> P9GaussianNoiseState:
        if not isinstance(state, P9GaussianNoiseState):
            raise GaussianNoiseError(
                "state",
                "must be P9GaussianNoiseState",
            )
        if state.perturbation_id != self.perturbation_id:
            raise GaussianNoiseError(
                "state.perturbation_id",
                "must match p09",
            )
        if state.sweep_config_sha256 != self._config.sha256:
            raise GaussianNoiseError(
                "state.sweep_config_sha256",
                "must match the retained sweep config",
            )
        expected_digest = _state_digest(
            state.perturbation_id,
            state.spectrum_id,
            state.sweep_config_sha256,
            state.standard_noise,
        )
        if state.state_digest != expected_digest:
            raise GaussianNoiseError(
                "state.state_digest",
                "must match the deterministic state digest",
            )
        return state

    def prepare(
        self,
        spectrum: Spectrum1D,
        context: PerturbationContext,
    ) -> P9GaussianNoiseState:
        if not isinstance(spectrum, Spectrum1D):
            raise GaussianNoiseError(
                "spectrum",
                "must be Spectrum1D",
            )
        self._validate_context(context)
        seed_material = derive_state_seed_material(
            self._config,
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
        )
        generator = np.random.Generator(
            np.random.PCG64(np.random.SeedSequence(seed_material))
        )
        standard_noise = np.asarray(
            generator.standard_normal(spectrum.intensity.size),
            dtype="<f8",
        )
        state_digest = _state_digest(
            self.perturbation_id,
            spectrum.spectrum_id,
            self._config.sha256,
            standard_noise,
        )
        state = P9GaussianNoiseState(
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
            sweep_config_sha256=self._config.sha256,
            state_digest=state_digest,
            standard_noise=standard_noise,
        )
        binding_key = _binding_key(state)
        binding = _binding_payload(
            spectrum.axis_cm1,
            spectrum.intensity,
        )
        registered_bindings = self._prepared_bindings.get(binding_key, ())
        if not any(
            np.array_equal(binding[0], registered_axis)
            and np.array_equal(binding[1], registered_intensity)
            for registered_axis, registered_intensity in registered_bindings
        ):
            self._prepared_bindings[binding_key] = (
                *registered_bindings,
                binding,
            )
        return state

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        if not isinstance(spectrum, Spectrum1D):
            raise GaussianNoiseError(
                "spectrum",
                "must be Spectrum1D",
            )
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise GaussianNoiseError(
                "alpha",
                "must be one of the frozen alpha values",
            )
        if spectrum.spectrum_id != validated_state.spectrum_id:
            raise GaussianNoiseError(
                "spectrum.spectrum_id",
                "must match the prepared state",
            )
        if spectrum.intensity.size != validated_state.standard_noise.size:
            raise GaussianNoiseError(
                "point count",
                "must match the prepared state",
            )
        prepared_bindings = self._prepared_bindings.get(
            _binding_key(validated_state)
        )
        if prepared_bindings is None:
            raise GaussianNoiseError(
                "prepared state",
                "must be produced by this P9GaussianWhiteNoise instance",
            )
        axis_bindings = tuple(
            registered_intensity
            for registered_axis, registered_intensity in prepared_bindings
            if np.array_equal(spectrum.axis_cm1, registered_axis)
        )
        if not axis_bindings:
            raise GaussianNoiseError(
                "source axis",
                "must match the prepared source axis",
            )
        if not any(
            np.array_equal(spectrum.intensity, registered_intensity)
            for registered_intensity in axis_bindings
        ):
            raise GaussianNoiseError(
                "source intensity",
                "must match the prepared source intensity",
            )
        with np.errstate(over="ignore", invalid="ignore"):
            signal_rms = _scaled_rms(spectrum.intensity)
            sigma = float(alpha * signal_rms)
            perturbed = (
                spectrum.intensity
                + sigma * validated_state.standard_noise
            )
        if (
            not math.isfinite(signal_rms)
            or not math.isfinite(sigma)
            or not np.isfinite(perturbed).all()
        ):
            raise GaussianNoiseError(
                "output finite",
                "signal RMS, sigma, and output must remain finite",
            )
        if alpha == self._config.identity_alpha or sigma == 0.0:
            output_intensity = np.array(
                spectrum.intensity,
                dtype="<f8",
                copy=True,
            )
        else:
            output_intensity = np.asarray(perturbed, dtype="<f8")
        output = Spectrum1D(
            spectrum_id=derive_perturbed_spectrum_id(
                spectrum.spectrum_id,
                self.perturbation_id,
                alpha,
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
            alpha=alpha,
            output=output,
            axis_behavior=self.axis_behavior,
            state_digest=validated_state.state_digest,
            axis_changed=False,
            intensity_changed=not np.array_equal(
                spectrum.intensity,
                output.intensity,
            ),
            diagnostics={
                "signal_rms": signal_rms,
                "sigma": sigma,
                "noise_mean_square": float(
                    np.mean(validated_state.standard_noise**2)
                ),
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
            raise GaussianNoiseError(exc.path, exc.reason) from exc
        return result


__all__ = [
    "AXIS_BEHAVIOR",
    "GaussianNoiseError",
    "P9GaussianNoiseState",
    "P9GaussianWhiteNoise",
    "PERTURBATION_ID",
]
