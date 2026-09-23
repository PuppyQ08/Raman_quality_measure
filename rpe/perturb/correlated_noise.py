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


PERTURBATION_ID = "p10"
AXIS_BEHAVIOR = AxisBehavior.PRESERVE
CORRELATION_LENGTH_CM1 = 20.0
_HEX_DIGITS = frozenset("0123456789abcdef")


def _config_error_from_path(path: str, reason: str) -> CorrelatedNoiseError:
    return CorrelatedNoiseError(f"config.{path}", reason)


class CorrelatedNoiseError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise CorrelatedNoiseError(path, "must be a nonempty string")
    return value


def _lower_hex_64(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise CorrelatedNoiseError(
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _read_only_float64_vector(
    path: str,
    value: object,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise CorrelatedNoiseError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise CorrelatedNoiseError(
            f"{path} dtype",
            "must be little-endian float64",
        )
    if value.ndim != 1:
        raise CorrelatedNoiseError(
            f"{path} dimension",
            "must be one-dimensional",
        )
    if value.size == 0:
        raise CorrelatedNoiseError(
            f"{path} shape",
            "must be nonempty",
        )
    if not np.isfinite(value).all():
        raise CorrelatedNoiseError(
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
    standard_correlated_noise: np.ndarray,
) -> str:
    perturbation_name = _nonempty_string("perturbation_id", perturbation_id)
    spectrum_name = _nonempty_string("spectrum_id", spectrum_id)
    config_hash = _lower_hex_64(
        "sweep_config_sha256",
        sweep_config_sha256,
    )
    noise = _read_only_float64_vector(
        "standard_correlated_noise",
        standard_correlated_noise,
    )
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


def _binding_key(
    state: P10CorrelatedNoiseState | PerturbationState,
) -> tuple[str, str, str, str]:
    return (
        state.perturbation_id,
        state.spectrum_id,
        state.sweep_config_sha256,
        state.state_digest,
    )


def _binding_payload(
    axis_cm1: np.ndarray,
    intensity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
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


def _prepare_standard_correlated_noise(axis_cm1: np.ndarray, seed_material: tuple[int, int, int, int]) -> np.ndarray:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(seed_material))
    )
    white_noise = np.asarray(
        generator.standard_normal(axis_cm1.size),
        dtype="<f8",
    )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        distances = axis_cm1[:, None] - axis_cm1[None, :]
        scaled = distances / CORRELATION_LENGTH_CM1
        kernel = np.exp(-0.5 * scaled * scaled, dtype=np.float64)
        row_sums = np.sum(kernel, axis=1)
        normalized_kernel = kernel / row_sums[:, None]
        correlated_raw = normalized_kernel @ white_noise
        centered = correlated_raw - np.mean(correlated_raw)
        centered_rms = _scaled_rms(centered)
        if centered_rms == 0.0:
            raise CorrelatedNoiseError(
                "prepared noise RMS",
                "must be nonzero after centering",
            )
        normalized = centered / centered_rms
    if (
        not np.isfinite(distances).all()
        or not np.isfinite(kernel).all()
        or not np.isfinite(row_sums).all()
        or np.any(row_sums <= 0.0)
        or not np.isfinite(normalized_kernel).all()
        or not np.isfinite(correlated_raw).all()
        or not np.isfinite(centered).all()
        or not math.isfinite(centered_rms)
        or not np.isfinite(normalized).all()
    ):
        raise CorrelatedNoiseError(
            "prepared noise finite",
            "derived distances, kernel, filtering, centering, and normalization must remain finite",
        )
    return np.asarray(normalized, dtype="<f8")


@dataclass(frozen=True)
class P10CorrelatedNoiseState:
    perturbation_id: str
    spectrum_id: str
    sweep_config_sha256: str
    state_digest: str
    standard_correlated_noise: np.ndarray

    def __post_init__(self) -> None:
        perturbation_id = _nonempty_string(
            "perturbation_id",
            self.perturbation_id,
        )
        if perturbation_id != PERTURBATION_ID:
            raise CorrelatedNoiseError(
                "perturbation_id",
                "must equal p10",
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
            "standard_correlated_noise",
            self.standard_correlated_noise,
        )
        object.__setattr__(self, "standard_correlated_noise", noise)
        object.__setattr__(
            self,
            "state_digest",
            _lower_hex_64("state_digest", self.state_digest),
        )


class P10CorrelatedNoise:
    perturbation_id = PERTURBATION_ID
    axis_behavior = AXIS_BEHAVIOR

    def __init__(self, config: PerturbationSweepConfig) -> None:
        if not isinstance(config, PerturbationSweepConfig):
            raise CorrelatedNoiseError(
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
            raise CorrelatedNoiseError(
                "context",
                "must be PerturbationContext",
            )
        if context.sweep_id != self._config.sweep_id:
            raise CorrelatedNoiseError(
                "context.sweep_id",
                "must match the retained sweep config",
            )
        if context.sweep_config_sha256 != self._config.sha256:
            raise CorrelatedNoiseError(
                "context.sweep_config_sha256",
                "must match the retained sweep config",
            )
        if context.global_seed != self._config.global_seed:
            raise CorrelatedNoiseError(
                "context.global_seed",
                "must match the retained sweep config",
            )

    def _validated_state(
        self,
        state: PerturbationState,
    ) -> P10CorrelatedNoiseState:
        if not isinstance(state, P10CorrelatedNoiseState):
            raise CorrelatedNoiseError(
                "state",
                "must be P10CorrelatedNoiseState",
            )
        if state.perturbation_id != self.perturbation_id:
            raise CorrelatedNoiseError(
                "state.perturbation_id",
                "must match p10",
            )
        if state.sweep_config_sha256 != self._config.sha256:
            raise CorrelatedNoiseError(
                "state.sweep_config_sha256",
                "must match the retained sweep config",
            )
        expected_digest = _state_digest(
            state.perturbation_id,
            state.spectrum_id,
            state.sweep_config_sha256,
            state.standard_correlated_noise,
        )
        if state.state_digest != expected_digest:
            raise CorrelatedNoiseError(
                "state.state_digest",
                "must match the deterministic state digest",
            )
        return state

    def prepare(
        self,
        spectrum: Spectrum1D,
        context: PerturbationContext,
    ) -> P10CorrelatedNoiseState:
        if not isinstance(spectrum, Spectrum1D):
            raise CorrelatedNoiseError(
                "spectrum",
                "must be Spectrum1D",
            )
        self._validate_context(context)
        seed_material = derive_state_seed_material(
            self._config,
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
        )
        standard_correlated_noise = _prepare_standard_correlated_noise(
            spectrum.axis_cm1,
            seed_material,
        )
        state_digest = _state_digest(
            self.perturbation_id,
            spectrum.spectrum_id,
            self._config.sha256,
            standard_correlated_noise,
        )
        state = P10CorrelatedNoiseState(
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
            sweep_config_sha256=self._config.sha256,
            state_digest=state_digest,
            standard_correlated_noise=standard_correlated_noise,
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
            raise CorrelatedNoiseError(
                "spectrum",
                "must be Spectrum1D",
            )
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise CorrelatedNoiseError(
                "alpha",
                "must be one of the frozen alpha values",
            )
        if spectrum.spectrum_id != validated_state.spectrum_id:
            raise CorrelatedNoiseError(
                "spectrum.spectrum_id",
                "must match the prepared state",
            )
        if (
            spectrum.intensity.size
            != validated_state.standard_correlated_noise.size
        ):
            raise CorrelatedNoiseError(
                "point count",
                "must match the prepared state",
            )
        prepared_bindings = self._prepared_bindings.get(
            _binding_key(validated_state)
        )
        if prepared_bindings is None:
            raise CorrelatedNoiseError(
                "prepared state",
                "must be produced by this P10CorrelatedNoise instance",
            )
        axis_bindings = tuple(
            registered_intensity
            for registered_axis, registered_intensity in prepared_bindings
            if np.array_equal(spectrum.axis_cm1, registered_axis)
        )
        if not axis_bindings:
            raise CorrelatedNoiseError(
                "source axis",
                "must match the prepared source axis",
            )
        if not any(
            np.array_equal(spectrum.intensity, registered_intensity)
            for registered_intensity in axis_bindings
        ):
            raise CorrelatedNoiseError(
                "source intensity",
                "must match the prepared source intensity",
            )
        standard_noise_mean = float(
            np.mean(validated_state.standard_correlated_noise)
        )
        standard_noise_rms = _scaled_rms(
            validated_state.standard_correlated_noise
        )
        with np.errstate(over="ignore", invalid="ignore"):
            signal_rms = _scaled_rms(spectrum.intensity)
            sigma = float(alpha * signal_rms)
            perturbed = (
                spectrum.intensity
                + sigma * validated_state.standard_correlated_noise
            )
        if (
            not math.isfinite(standard_noise_mean)
            or not math.isfinite(standard_noise_rms)
            or not math.isfinite(signal_rms)
            or not math.isfinite(sigma)
            or not np.isfinite(perturbed).all()
        ):
            raise CorrelatedNoiseError(
                "output finite",
                "signal RMS, sigma, output, and diagnostics must remain finite",
            )
        if alpha == self._config.identity_alpha or signal_rms == 0.0:
            output_intensity = np.array(
                spectrum.intensity,
                dtype="<f8",
                copy=True,
            )
        else:
            output_intensity = np.asarray(perturbed, dtype="<f8")
            if np.array_equal(output_intensity, spectrum.intensity):
                raise CorrelatedNoiseError(
                    "positive perturbation",
                    "rounds entirely back to identity",
                )
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
                "noise_model": "native_axis_gaussian_correlated",
                "correlation_length_cm1": CORRELATION_LENGTH_CM1,
                "kernel_normalization": "row_sum_one",
                "boundary_mode": "full_native_gaussian_kernel",
                "standard_noise_mean": standard_noise_mean,
                "standard_noise_rms": standard_noise_rms,
                "signal_rms": signal_rms,
                "sigma": sigma,
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
            raise CorrelatedNoiseError(exc.path, exc.reason) from exc
        return result


__all__ = [
    "AXIS_BEHAVIOR",
    "CORRELATION_LENGTH_CM1",
    "CorrelatedNoiseError",
    "P10CorrelatedNoise",
    "P10CorrelatedNoiseState",
    "PERTURBATION_ID",
]
