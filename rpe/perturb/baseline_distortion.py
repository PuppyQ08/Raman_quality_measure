from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre as npleg

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


PERTURBATION_ID = "p08"
AXIS_BEHAVIOR = AxisBehavior.PRESERVE
BASELINE_MODEL = "deterministic_legendre_orders_2_to_4"
_HEX_DIGITS = frozenset("0123456789abcdef")


def _config_error_from_path(path: str, reason: str) -> BaselineDistortionError:
    return BaselineDistortionError(f"config.{path}", reason)


class BaselineDistortionError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise BaselineDistortionError(path, "must be a nonempty string")
    return value


def _lower_hex_64(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise BaselineDistortionError(
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _read_only_float64_vector(path: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise BaselineDistortionError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise BaselineDistortionError(
            f"{path} dtype",
            "must be little-endian float64",
        )
    if value.ndim != 1:
        raise BaselineDistortionError(
            f"{path} dimension",
            "must be one-dimensional",
        )
    if value.size == 0:
        raise BaselineDistortionError(
            f"{path} shape",
            "must be nonempty",
        )
    if not np.isfinite(value).all():
        raise BaselineDistortionError(
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


def _coefficient_tuple(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise BaselineDistortionError(path, "must be a tuple")
    if len(value) not in (1, 2, 3):
        raise BaselineDistortionError(
            path,
            "must contain one coefficient per Legendre order from 2 to degree",
        )
    converted: list[float] = []
    for index, item in enumerate(value):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise BaselineDistortionError(
                f"{path}[{index}]",
                "must be a finite real number",
            )
        converted.append(float(item))
    return tuple(converted)


def _state_digest(
    perturbation_id: str,
    spectrum_id: str,
    sweep_config_sha256: str,
    degree: int,
    coefficients: tuple[float, ...],
    standard_baseline_shape: np.ndarray,
    source_axis_sha256: str,
    source_intensity_sha256: str,
) -> str:
    perturbation_name = _nonempty_string("perturbation_id", perturbation_id)
    spectrum_name = _nonempty_string("spectrum_id", spectrum_id)
    config_hash = _lower_hex_64(
        "sweep_config_sha256",
        sweep_config_sha256,
    )
    axis_hash = _lower_hex_64("source_axis_sha256", source_axis_sha256)
    intensity_hash = _lower_hex_64(
        "source_intensity_sha256",
        source_intensity_sha256,
    )
    if not isinstance(degree, int) or degree not in (2, 3, 4):
        raise BaselineDistortionError("degree", "must be one of 2, 3, or 4")
    coeffs = _coefficient_tuple("coefficients", coefficients)
    if len(coeffs) != degree - 1:
        raise BaselineDistortionError(
            "coefficients",
            "must have one coefficient for each Legendre order from 2 to degree",
        )
    shape = _read_only_float64_vector(
        "standard_baseline_shape",
        standard_baseline_shape,
    )
    payload = (
        b"rpe-baseline-distortion-state-digest-v1\0"
        + struct.pack("<Q", len(perturbation_name.encode("utf-8")))
        + perturbation_name.encode("utf-8")
        + struct.pack("<Q", len(spectrum_name.encode("utf-8")))
        + spectrum_name.encode("utf-8")
        + bytes.fromhex(config_hash)
        + struct.pack("<Q", degree)
        + struct.pack("<Q", len(coeffs))
        + np.asarray(coeffs, dtype="<f8").tobytes(order="C")
        + struct.pack("<Q", shape.size)
        + shape.astype("<f8", copy=False).tobytes(order="C")
        + bytes.fromhex(axis_hash)
        + bytes.fromhex(intensity_hash)
    )
    return hashlib.sha256(payload).hexdigest()


def _prepare_standard_baseline_shape(
    axis_cm1: np.ndarray,
    seed_material: tuple[int, int, int, int],
) -> tuple[int, tuple[float, ...], np.ndarray]:
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(seed_material))
    )
    degree = int(generator.integers(2, 5))
    coefficients_array = np.asarray(
        generator.standard_normal(degree - 1),
        dtype="<f8",
    )
    coefficients = tuple(float(value) for value in coefficients_array)
    lo = float(axis_cm1[0])
    hi = float(axis_cm1[-1])
    span = hi - lo
    if not math.isfinite(lo) or not math.isfinite(hi) or not math.isfinite(span):
        raise BaselineDistortionError("axis span", "must be finite")
    if span <= 0.0:
        raise BaselineDistortionError("axis span", "must remain positive")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        u = -1.0 + 2.0 * ((axis_cm1 - lo) / span)
        legendre_coefficients = np.zeros(degree + 1, dtype="<f8")
        legendre_coefficients[2:] = coefficients_array
        evaluated = npleg.legval(u, legendre_coefficients)
        centered = evaluated - np.mean(evaluated)
        centered_rms = _scaled_rms(centered)
        if centered_rms == 0.0:
            raise BaselineDistortionError(
                "prepared baseline RMS",
                "must be nonzero after centering",
            )
        normalized = centered / centered_rms
    if (
        not np.isfinite(coefficients_array).all()
        or not np.isfinite(u).all()
        or not np.isfinite(evaluated).all()
        or not np.isfinite(centered).all()
        or not math.isfinite(centered_rms)
        or not np.isfinite(normalized).all()
    ):
        raise BaselineDistortionError(
            "prepared baseline finite",
            "coefficients, normalized coordinates, evaluated shape, centering, and normalization must remain finite",
        )
    return degree, coefficients, np.asarray(normalized, dtype="<f8")


@dataclass(frozen=True)
class P8BaselineDistortionState:
    perturbation_id: str
    spectrum_id: str
    sweep_config_sha256: str
    state_digest: str
    degree: int
    coefficients: tuple[float, ...]
    standard_baseline_shape: np.ndarray
    source_axis_sha256: str
    source_intensity_sha256: str

    def __post_init__(self) -> None:
        perturbation_id = _nonempty_string(
            "perturbation_id",
            self.perturbation_id,
        )
        if perturbation_id != PERTURBATION_ID:
            raise BaselineDistortionError(
                "perturbation_id",
                "must equal p08",
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
        object.__setattr__(
            self,
            "state_digest",
            _lower_hex_64("state_digest", self.state_digest),
        )
        if not isinstance(self.degree, int) or self.degree not in (2, 3, 4):
            raise BaselineDistortionError(
                "degree",
                "must be one of 2, 3, or 4",
            )
        object.__setattr__(self, "degree", self.degree)
        coefficients = _coefficient_tuple("coefficients", self.coefficients)
        if len(coefficients) != self.degree - 1:
            raise BaselineDistortionError(
                "coefficients",
                "must match the chosen Legendre degree",
            )
        object.__setattr__(self, "coefficients", coefficients)
        shape = _read_only_float64_vector(
            "standard_baseline_shape",
            self.standard_baseline_shape,
        )
        object.__setattr__(self, "standard_baseline_shape", shape)
        object.__setattr__(
            self,
            "source_axis_sha256",
            _lower_hex_64(
                "source_axis_sha256",
                self.source_axis_sha256,
            ),
        )
        object.__setattr__(
            self,
            "source_intensity_sha256",
            _lower_hex_64(
                "source_intensity_sha256",
                self.source_intensity_sha256,
            ),
        )


class P8LowOrderBaselineDistortion:
    perturbation_id = PERTURBATION_ID
    axis_behavior = AXIS_BEHAVIOR

    def __init__(self, config: PerturbationSweepConfig) -> None:
        if not isinstance(config, PerturbationSweepConfig):
            raise BaselineDistortionError(
                "config",
                "must be PerturbationSweepConfig",
            )
        try:
            validate_perturbation_sweep_config(config)
        except PerturbationSweepConfigError as exc:
            raise _config_error_from_path(exc.path, exc.reason) from exc
        self._config = config

    def _validate_context(self, context: PerturbationContext) -> None:
        if not isinstance(context, PerturbationContext):
            raise BaselineDistortionError(
                "context",
                "must be PerturbationContext",
            )
        if context.sweep_id != self._config.sweep_id:
            raise BaselineDistortionError(
                "context.sweep_id",
                "must match the retained sweep config",
            )
        if context.sweep_config_sha256 != self._config.sha256:
            raise BaselineDistortionError(
                "context.sweep_config_sha256",
                "must match the retained sweep config",
            )
        if context.global_seed != self._config.global_seed:
            raise BaselineDistortionError(
                "context.global_seed",
                "must match the retained sweep config",
            )

    def _validated_state(
        self,
        state: PerturbationState,
    ) -> P8BaselineDistortionState:
        if not isinstance(state, P8BaselineDistortionState):
            raise BaselineDistortionError(
                "state",
                "must be P8BaselineDistortionState",
            )
        if state.perturbation_id != self.perturbation_id:
            raise BaselineDistortionError(
                "state.perturbation_id",
                "must match p08",
            )
        if state.sweep_config_sha256 != self._config.sha256:
            raise BaselineDistortionError(
                "state.sweep_config_sha256",
                "must match the retained sweep config",
            )
        expected_digest = _state_digest(
            state.perturbation_id,
            state.spectrum_id,
            state.sweep_config_sha256,
            state.degree,
            state.coefficients,
            state.standard_baseline_shape,
            state.source_axis_sha256,
            state.source_intensity_sha256,
        )
        if state.state_digest != expected_digest:
            raise BaselineDistortionError(
                "state.state_digest",
                "must match the deterministic state digest",
            )
        return state

    def prepare(
        self,
        spectrum: Spectrum1D,
        context: PerturbationContext,
    ) -> P8BaselineDistortionState:
        if not isinstance(spectrum, Spectrum1D):
            raise BaselineDistortionError(
                "spectrum",
                "must be Spectrum1D",
            )
        self._validate_context(context)
        seed_material = derive_state_seed_material(
            self._config,
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
        )
        degree, coefficients, standard_baseline_shape = (
            _prepare_standard_baseline_shape(spectrum.axis_cm1, seed_material)
        )
        source_axis_sha256 = _sha256_bytes(spectrum.axis_cm1)
        source_intensity_sha256 = _sha256_bytes(spectrum.intensity)
        state_digest = _state_digest(
            self.perturbation_id,
            spectrum.spectrum_id,
            self._config.sha256,
            degree,
            coefficients,
            standard_baseline_shape,
            source_axis_sha256,
            source_intensity_sha256,
        )
        return P8BaselineDistortionState(
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
            sweep_config_sha256=self._config.sha256,
            state_digest=state_digest,
            degree=degree,
            coefficients=coefficients,
            standard_baseline_shape=standard_baseline_shape,
            source_axis_sha256=source_axis_sha256,
            source_intensity_sha256=source_intensity_sha256,
        )

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        if not isinstance(spectrum, Spectrum1D):
            raise BaselineDistortionError(
                "spectrum",
                "must be Spectrum1D",
            )
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise BaselineDistortionError(
                "alpha",
                "must be one of the frozen alpha values",
            )
        if spectrum.spectrum_id != validated_state.spectrum_id:
            raise BaselineDistortionError(
                "spectrum.spectrum_id",
                "must match the prepared state",
            )
        if spectrum.intensity.size != validated_state.standard_baseline_shape.size:
            raise BaselineDistortionError(
                "point count",
                "must match the prepared state",
            )
        source_axis_sha256 = _sha256_bytes(spectrum.axis_cm1)
        if source_axis_sha256 != validated_state.source_axis_sha256:
            raise BaselineDistortionError(
                "source axis",
                "must match the prepared source axis",
            )
        source_intensity_sha256 = _sha256_bytes(spectrum.intensity)
        if source_intensity_sha256 != validated_state.source_intensity_sha256:
            raise BaselineDistortionError(
                "source intensity",
                "must match the prepared source intensity",
            )
        standard_shape_rms = _scaled_rms(validated_state.standard_baseline_shape)
        with np.errstate(over="ignore", invalid="ignore"):
            signal_rms = _scaled_rms(spectrum.intensity)
            amplitude = float(alpha * signal_rms)
            added_baseline = amplitude * validated_state.standard_baseline_shape
            perturbed = spectrum.intensity + added_baseline
        realized_added_baseline_rms = _scaled_rms(
            np.asarray(added_baseline, dtype="<f8")
        )
        if (
            not math.isfinite(standard_shape_rms)
            or not math.isfinite(signal_rms)
            or not math.isfinite(amplitude)
            or not np.isfinite(added_baseline).all()
            or not np.isfinite(perturbed).all()
            or not math.isfinite(realized_added_baseline_rms)
        ):
            raise BaselineDistortionError(
                "output finite",
                "signal RMS, amplitude, added baseline, output, and diagnostics must remain finite",
            )
        if alpha == self._config.identity_alpha or signal_rms == 0.0:
            output_intensity = np.array(
                spectrum.intensity,
                dtype="<f8",
                copy=True,
            )
            realized_added_baseline_rms = 0.0
        else:
            output_intensity = np.asarray(perturbed, dtype="<f8")
            if np.array_equal(output_intensity, spectrum.intensity):
                raise BaselineDistortionError(
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
                "baseline_model": BASELINE_MODEL,
                "degree": validated_state.degree,
                "coefficients": validated_state.coefficients,
                "shape_centered": True,
                "standard_shape_rms": standard_shape_rms,
                "signal_rms": signal_rms,
                "amplitude": amplitude,
                "realized_added_baseline_rms": realized_added_baseline_rms,
                "axis_preserved": True,
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
            raise BaselineDistortionError(exc.path, exc.reason) from exc
        return result


__all__ = [
    "AXIS_BEHAVIOR",
    "BASELINE_MODEL",
    "BaselineDistortionError",
    "P8BaselineDistortionState",
    "P8LowOrderBaselineDistortion",
    "PERTURBATION_ID",
]
