from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths

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


DETECTOR_ID = "internal_find_peaks_prominence_5pct_range"
AXIS_BEHAVIOR = AxisBehavior.PRESERVE
_HEX_DIGITS = frozenset("0123456789abcdef")
_FALSE_PEAK_PROPOSAL_LIMIT = 10_000


def _config_error_from_path(path: str, reason: str) -> PeakFamilyError:
    return PeakFamilyError(f"config.{path}", reason)


class PeakFamilyError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise PeakFamilyError(path, "must be a nonempty string")
    return value


def _lower_hex_64(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise PeakFamilyError(
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _read_only_float64_vector(path: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PeakFamilyError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise PeakFamilyError(
            f"{path} dtype",
            "must be little-endian float64",
        )
    if value.ndim != 1:
        raise PeakFamilyError(
            f"{path} dimension",
            "must be one-dimensional",
        )
    if value.size == 0:
        raise PeakFamilyError(
            f"{path} shape",
            "must be nonempty",
        )
    if not np.isfinite(value).all():
        raise PeakFamilyError(
            f"{path} finite",
            "contains non-finite values",
        )
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _read_only_float64_matrix(path: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise PeakFamilyError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise PeakFamilyError(
            f"{path} dtype",
            "must be little-endian float64",
        )
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise PeakFamilyError(
            f"{path} shape",
            "must be two-dimensional and nonempty",
        )
    if not np.isfinite(value).all():
        raise PeakFamilyError(
            f"{path} finite",
            "contains non-finite values",
        )
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.asarray(array, dtype="<f8").tobytes(order="C")
    ).hexdigest()


def _length_prefixed_utf8(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _inclusive_bounds_tuple(
    path: str,
    value: object,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, tuple) or not value:
        raise PeakFamilyError(path, "must be a nonempty tuple")
    converted: list[tuple[int, int]] = []
    for index, item in enumerate(value):
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or any(
                isinstance(component, bool) or not isinstance(component, int)
                for component in item
            )
        ):
            raise PeakFamilyError(
                f"{path}[{index}]",
                "must be a pair of integers",
            )
        start, end = item
        if start < 0 or end < start:
            raise PeakFamilyError(
                f"{path}[{index}]",
                "must satisfy 0 <= start <= end",
            )
        converted.append((start, end))
    return tuple(converted)


def _finite_float_tuple(
    path: str,
    value: object,
    *,
    nonempty: bool,
) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise PeakFamilyError(path, "must be a tuple")
    if nonempty and not value:
        raise PeakFamilyError(path, "must be nonempty")
    converted: list[float] = []
    for index, item in enumerate(value):
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
        ):
            raise PeakFamilyError(
                f"{path}[{index}]",
                "must be a finite real number",
            )
        converted.append(float(item))
    return tuple(converted)


def _int_tuple(
    path: str,
    value: object,
    *,
    nonempty: bool,
) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise PeakFamilyError(path, "must be a tuple")
    if nonempty and not value:
        raise PeakFamilyError(path, "must be nonempty")
    converted: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise PeakFamilyError(
                f"{path}[{index}]",
                "must be an integer",
            )
        converted.append(item)
    return tuple(converted)


def _component_area(axis_cm1: np.ndarray, component: np.ndarray) -> float:
    return float(np.trapezoid(component, axis_cm1))


def _gaussian_with_area(
    axis_cm1: np.ndarray,
    *,
    center_cm1: float,
    sigma_cm1: float,
    area: float,
) -> np.ndarray:
    if sigma_cm1 <= 0.0 or not math.isfinite(sigma_cm1):
        raise PeakFamilyError(
            "sigma_cm1",
            "must be a positive finite real number",
        )
    amplitude = area / (sigma_cm1 * math.sqrt(2.0 * math.pi))
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        normalized = (axis_cm1 - center_cm1) / sigma_cm1
        gaussian = amplitude * np.exp(-0.5 * normalized * normalized)
    if not np.isfinite(gaussian).all():
        raise PeakFamilyError(
            "gaussian finite",
            "must remain finite",
        )
    gaussian = np.asarray(gaussian, dtype="<f8")
    realized_area = _component_area(axis_cm1, gaussian)
    if realized_area <= 0.0 or not math.isfinite(realized_area):
        raise PeakFamilyError(
            "gaussian area",
            "must remain positive and finite",
        )
    return gaussian * (area / realized_area)


def _gaussian_with_height(
    axis_cm1: np.ndarray,
    *,
    center_cm1: float,
    sigma_cm1: float,
    height: float,
) -> np.ndarray:
    if sigma_cm1 <= 0.0 or not math.isfinite(sigma_cm1):
        raise PeakFamilyError(
            "sigma_cm1",
            "must be a positive finite real number",
        )
    if height <= 0.0 or not math.isfinite(height):
        raise PeakFamilyError(
            "height",
            "must be a positive finite real number",
        )
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        normalized = (axis_cm1 - center_cm1) / sigma_cm1
        gaussian = height * np.exp(-0.5 * normalized * normalized)
    if not np.isfinite(gaussian).all():
        raise PeakFamilyError(
            "gaussian finite",
            "must remain finite",
        )
    return np.asarray(gaussian, dtype="<f8")


def _peak_model_digest(
    perturbation_id: str,
    spectrum_id: str,
    sweep_config_sha256: str,
    detector_id: str,
    source_axis_sha256: str,
    source_intensity_sha256: str,
    peak_indices: tuple[int, ...],
    peak_positions_cm1: tuple[float, ...],
    prominences: tuple[float, ...],
    support_bounds: tuple[tuple[int, int], ...],
    component_matrix: np.ndarray,
    component_heights: tuple[float, ...],
    component_areas: tuple[float, ...],
    median_fwhm_cm1: float,
    candidate_false_centers_cm1: tuple[float, ...],
    candidate_false_peak: np.ndarray,
    deterministic_random_material: tuple[int, ...],
) -> str:
    perturbation_name = _nonempty_string("perturbation_id", perturbation_id)
    spectrum_name = _nonempty_string("spectrum_id", spectrum_id)
    config_hash = _lower_hex_64(
        "sweep_config_sha256",
        sweep_config_sha256,
    )
    detector_name = _nonempty_string("detector_id", detector_id)
    axis_hash = _lower_hex_64("source_axis_sha256", source_axis_sha256)
    intensity_hash = _lower_hex_64(
        "source_intensity_sha256",
        source_intensity_sha256,
    )
    peak_indices_tuple = _int_tuple("peak_indices", peak_indices, nonempty=True)
    peak_positions_tuple = _finite_float_tuple(
        "peak_positions_cm1",
        peak_positions_cm1,
        nonempty=True,
    )
    prominences_tuple = _finite_float_tuple(
        "prominences",
        prominences,
        nonempty=True,
    )
    support_bounds_tuple = _inclusive_bounds_tuple(
        "support_bounds",
        support_bounds,
    )
    heights_tuple = _finite_float_tuple(
        "component_heights",
        component_heights,
        nonempty=True,
    )
    areas_tuple = _finite_float_tuple(
        "component_areas",
        component_areas,
        nonempty=True,
    )
    candidates_tuple = _finite_float_tuple(
        "candidate_false_centers_cm1",
        candidate_false_centers_cm1,
        nonempty=False,
    )
    random_material_tuple = _int_tuple(
        "deterministic_random_material",
        deterministic_random_material,
        nonempty=True,
    )
    matrix = _read_only_float64_matrix("component_matrix", component_matrix)
    false_peak = _read_only_float64_vector(
        "candidate_false_peak",
        candidate_false_peak,
    )
    if not math.isfinite(median_fwhm_cm1) or median_fwhm_cm1 <= 0.0:
        raise PeakFamilyError(
            "median_fwhm_cm1",
            "must be a positive finite real number",
        )
    payload = (
        b"rpe-peak-family-state-digest-v1\0"
        + _length_prefixed_utf8(perturbation_name)
        + _length_prefixed_utf8(spectrum_name)
        + bytes.fromhex(config_hash)
        + _length_prefixed_utf8(detector_name)
        + bytes.fromhex(axis_hash)
        + bytes.fromhex(intensity_hash)
        + struct.pack("<Q", len(peak_indices_tuple))
        + np.asarray(peak_indices_tuple, dtype="<q").tobytes(order="C")
        + struct.pack("<Q", len(peak_positions_tuple))
        + np.asarray(peak_positions_tuple, dtype="<f8").tobytes(order="C")
        + struct.pack("<Q", len(prominences_tuple))
        + np.asarray(prominences_tuple, dtype="<f8").tobytes(order="C")
        + struct.pack("<Q", len(support_bounds_tuple))
        + np.asarray(support_bounds_tuple, dtype="<q").tobytes(order="C")
        + struct.pack("<Q", matrix.shape[0])
        + struct.pack("<Q", matrix.shape[1])
        + matrix.astype("<f8", copy=False).tobytes(order="C")
        + struct.pack("<Q", len(heights_tuple))
        + np.asarray(heights_tuple, dtype="<f8").tobytes(order="C")
        + struct.pack("<Q", len(areas_tuple))
        + np.asarray(areas_tuple, dtype="<f8").tobytes(order="C")
        + struct.pack("<d", median_fwhm_cm1)
        + struct.pack("<Q", len(candidates_tuple))
        + np.asarray(candidates_tuple, dtype="<f8").tobytes(order="C")
        + struct.pack("<Q", false_peak.size)
        + false_peak.astype("<f8", copy=False).tobytes(order="C")
        + struct.pack("<Q", len(random_material_tuple))
        + np.asarray(random_material_tuple, dtype="<Q").tobytes(order="C")
    )
    return hashlib.sha256(payload).hexdigest()


def _validate_context(
    config: PerturbationSweepConfig,
    context: PerturbationContext,
) -> None:
    if not isinstance(context, PerturbationContext):
        raise PeakFamilyError(
            "context",
            "must be PerturbationContext",
        )
    if context.sweep_id != config.sweep_id:
        raise PeakFamilyError(
            "context.sweep_id",
            "must match the retained sweep config",
        )
    if context.sweep_config_sha256 != config.sha256:
        raise PeakFamilyError(
            "context.sweep_config_sha256",
            "must match the retained sweep config",
        )
    if context.global_seed != config.global_seed:
        raise PeakFamilyError(
            "context.global_seed",
            "must match the retained sweep config",
        )


def _detect_peak_components(
    spectrum: Spectrum1D,
) -> tuple[
    tuple[int, ...],
    tuple[float, ...],
    tuple[float, ...],
    tuple[tuple[int, int], ...],
    np.ndarray,
    tuple[float, ...],
    tuple[float, ...],
    float,
]:
    axis = _read_only_float64_vector("source axis", spectrum.axis_cm1)
    intensity = _read_only_float64_vector("source intensity", spectrum.intensity)
    if intensity.size < 5:
        raise PeakFamilyError(
            "point count",
            "must contain at least five points",
        )
    intensity_range = float(np.max(intensity) - np.min(intensity))
    if not math.isfinite(intensity_range) or intensity_range <= 0.0:
        raise PeakFamilyError(
            "intensity range",
            "must have a positive finite intensity range",
        )
    prominence_threshold = 0.05 * intensity_range
    peak_indices_array, _ = find_peaks(
        intensity,
        prominence=prominence_threshold,
    )
    if peak_indices_array.size == 0:
        raise PeakFamilyError(
            "detected peaks",
            "must contain at least one detected peak",
        )
    prominences_array = np.asarray(
        peak_prominences(intensity, peak_indices_array)[0],
        dtype="<f8",
    )
    widths, _, left_ips, right_ips = peak_widths(
        intensity,
        peak_indices_array,
        rel_height=0.5,
    )
    widths = np.asarray(widths, dtype="<f8")
    left_ips = np.asarray(left_ips, dtype="<f8")
    right_ips = np.asarray(right_ips, dtype="<f8")
    index_axis = np.arange(axis.size, dtype="<f8")
    left_positions = np.interp(left_ips, index_axis, axis)
    right_positions = np.interp(right_ips, index_axis, axis)
    fwhm_values = np.asarray(right_positions - left_positions, dtype="<f8")
    if not np.isfinite(fwhm_values).all() or np.any(fwhm_values <= 0.0):
        raise PeakFamilyError(
            "median_fwhm_cm1",
            "must remain positive and finite",
        )
    boundaries: list[int] = [0]
    for left_peak, right_peak in zip(
        peak_indices_array[:-1],
        peak_indices_array[1:],
        strict=True,
    ):
        boundaries.append(int((int(left_peak) + int(right_peak)) // 2))
    boundaries.append(axis.size - 1)
    support_bounds: list[tuple[int, int]] = []
    for index in range(len(peak_indices_array)):
        support_bounds.append((boundaries[index], boundaries[index + 1]))
    component_rows: list[np.ndarray] = []
    heights: list[float] = []
    areas: list[float] = []
    peak_positions_cm1 = tuple(
        float(axis[int(index)])
        for index in peak_indices_array
    )
    peak_indices = tuple(int(index) for index in peak_indices_array)
    prominences = tuple(float(value) for value in prominences_array)
    for peak_index, (start, end) in zip(
        peak_indices,
        support_bounds,
        strict=True,
    ):
        baseline = np.interp(
            axis[start : end + 1],
            np.array([axis[start], axis[end]], dtype="<f8"),
            np.array([intensity[start], intensity[end]], dtype="<f8"),
        )
        component = np.zeros_like(intensity, dtype="<f8")
        local = intensity[start : end + 1] - baseline
        component[start : end + 1] = np.maximum(local, 0.0)
        height = float(component[peak_index])
        area = _component_area(axis, component)
        if height <= 0.0 or area <= 0.0 or not math.isfinite(height) or not math.isfinite(area):
            raise PeakFamilyError(
                "component validity",
                "each detected component must have positive finite height and area",
            )
        component_rows.append(component)
        heights.append(height)
        areas.append(area)
    component_matrix = np.ascontiguousarray(component_rows, dtype="<f8")
    component_matrix.setflags(write=False)
    return (
        peak_indices,
        peak_positions_cm1,
        prominences,
        tuple(support_bounds),
        component_matrix,
        tuple(heights),
        tuple(areas),
        float(np.median(fwhm_values)),
    )


def _generate_false_peak_candidates(
    axis_cm1: np.ndarray,
    peak_positions_cm1: tuple[float, ...],
    *,
    median_fwhm_cm1: float,
    false_peak_height: float,
    seed_material: tuple[int, int, int, int],
    count: int,
) -> tuple[tuple[float, ...], np.ndarray]:
    if count <= 0:
        zero = np.zeros(axis_cm1.size, dtype="<f8")
        zero.setflags(write=False)
        return (), zero
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence(seed_material))
    )
    centers: list[float] = []
    protected_radius = 2.0 * median_fwhm_cm1
    sigma_cm1 = median_fwhm_cm1 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    false_peak = np.zeros(axis_cm1.size, dtype="<f8")
    lower = float(axis_cm1[0])
    upper = float(axis_cm1[-1])
    for _ in range(_FALSE_PEAK_PROPOSAL_LIMIT):
        if len(centers) >= count:
            break
        candidate = float(generator.uniform(lower, upper))
        if any(abs(candidate - position) < protected_radius for position in peak_positions_cm1):
            continue
        if any(abs(candidate - position) < protected_radius for position in centers):
            continue
        centers.append(candidate)
        if len(centers) == 1:
            false_peak = _gaussian_with_height(
                axis_cm1,
                center_cm1=candidate,
                sigma_cm1=sigma_cm1,
                height=false_peak_height,
            )
    if len(centers) < count:
        raise PeakFamilyError(
            "false peak candidates",
            "could not place enough separated peaks after 10000 deterministic proposals",
        )
    false_peak = np.asarray(false_peak, dtype="<f8")
    false_peak.setflags(write=False)
    return tuple(centers), false_peak


@dataclass(frozen=True)
class PeakFamilyPreparedState:
    perturbation_id: str
    spectrum_id: str
    sweep_config_sha256: str
    state_digest: str
    detector_id: str
    source_axis_sha256: str
    source_intensity_sha256: str
    peak_indices: tuple[int, ...]
    peak_positions_cm1: tuple[float, ...]
    prominences: tuple[float, ...]
    support_bounds: tuple[tuple[int, int], ...]
    component_matrix: np.ndarray
    component_heights: tuple[float, ...]
    component_areas: tuple[float, ...]
    median_fwhm_cm1: float
    candidate_false_centers_cm1: tuple[float, ...]
    candidate_false_peak: np.ndarray
    deterministic_random_material: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "perturbation_id",
            _nonempty_string("perturbation_id", self.perturbation_id),
        )
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
        object.__setattr__(
            self,
            "detector_id",
            _nonempty_string("detector_id", self.detector_id),
        )
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
        peak_indices = _int_tuple("peak_indices", self.peak_indices, nonempty=True)
        object.__setattr__(self, "peak_indices", peak_indices)
        positions = _finite_float_tuple(
            "peak_positions_cm1",
            self.peak_positions_cm1,
            nonempty=True,
        )
        object.__setattr__(self, "peak_positions_cm1", positions)
        prominences = _finite_float_tuple(
            "prominences",
            self.prominences,
            nonempty=True,
        )
        object.__setattr__(self, "prominences", prominences)
        bounds = _inclusive_bounds_tuple("support_bounds", self.support_bounds)
        object.__setattr__(self, "support_bounds", bounds)
        matrix = _read_only_float64_matrix(
            "component_matrix",
            self.component_matrix,
        )
        object.__setattr__(self, "component_matrix", matrix)
        heights = _finite_float_tuple(
            "component_heights",
            self.component_heights,
            nonempty=True,
        )
        object.__setattr__(self, "component_heights", heights)
        areas = _finite_float_tuple(
            "component_areas",
            self.component_areas,
            nonempty=True,
        )
        object.__setattr__(self, "component_areas", areas)
        if not math.isfinite(self.median_fwhm_cm1) or self.median_fwhm_cm1 <= 0.0:
            raise PeakFamilyError(
                "median_fwhm_cm1",
                "must be a positive finite real number",
            )
        candidates = _finite_float_tuple(
            "candidate_false_centers_cm1",
            self.candidate_false_centers_cm1,
            nonempty=False,
        )
        object.__setattr__(self, "candidate_false_centers_cm1", candidates)
        false_peak = _read_only_float64_vector(
            "candidate_false_peak",
            self.candidate_false_peak,
        )
        object.__setattr__(self, "candidate_false_peak", false_peak)
        random_material = _int_tuple(
            "deterministic_random_material",
            self.deterministic_random_material,
            nonempty=True,
        )
        object.__setattr__(
            self,
            "deterministic_random_material",
            random_material,
        )


class _PeakFamilyBase:
    perturbation_id = ""
    axis_behavior = AXIS_BEHAVIOR

    def __init__(self, config: PerturbationSweepConfig) -> None:
        if not isinstance(config, PerturbationSweepConfig):
            raise PeakFamilyError(
                "config",
                "must be PerturbationSweepConfig",
            )
        try:
            validate_perturbation_sweep_config(config)
        except PerturbationSweepConfigError as exc:
            raise _config_error_from_path(exc.path, exc.reason) from exc
        self._config = config

    def _validated_state(self, state: PerturbationState) -> PeakFamilyPreparedState:
        if not isinstance(state, PeakFamilyPreparedState):
            raise PeakFamilyError(
                "state",
                "must be PeakFamilyPreparedState",
            )
        if state.perturbation_id != self.perturbation_id:
            raise PeakFamilyError(
                "state.perturbation_id",
                f"must match {self.perturbation_id}",
            )
        if state.sweep_config_sha256 != self._config.sha256:
            raise PeakFamilyError(
                "state.sweep_config_sha256",
                "must match the retained sweep config",
            )
        expected_digest = _peak_model_digest(
            state.perturbation_id,
            state.spectrum_id,
            state.sweep_config_sha256,
            state.detector_id,
            state.source_axis_sha256,
            state.source_intensity_sha256,
            state.peak_indices,
            state.peak_positions_cm1,
            state.prominences,
            state.support_bounds,
            state.component_matrix,
            state.component_heights,
            state.component_areas,
            state.median_fwhm_cm1,
            state.candidate_false_centers_cm1,
            state.candidate_false_peak,
            state.deterministic_random_material,
        )
        if state.state_digest != expected_digest:
            raise PeakFamilyError(
                "state.state_digest",
                "must match the deterministic state digest",
            )
        return state

    def prepare(
        self,
        spectrum: Spectrum1D,
        context: PerturbationContext,
    ) -> PeakFamilyPreparedState:
        if not isinstance(spectrum, Spectrum1D):
            raise PeakFamilyError(
                "spectrum",
                "must be Spectrum1D",
            )
        _validate_context(self._config, context)
        (
            peak_indices,
            peak_positions_cm1,
            prominences,
            support_bounds,
            component_matrix,
            component_heights,
            component_areas,
            median_fwhm_cm1,
        ) = _detect_peak_components(spectrum)
        source_axis_sha256 = _sha256_bytes(spectrum.axis_cm1)
        source_intensity_sha256 = _sha256_bytes(spectrum.intensity)
        seed_material = derive_state_seed_material(
            self._config,
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
        )
        if self.perturbation_id == "p05":
            max_insert_count = int(
                math.floor(max(self._config.alpha_grid) * len(peak_indices))
            )
            false_peak_height = float(
                np.median(np.asarray(component_heights, dtype="<f8"))
            )
            candidate_false_centers_cm1, candidate_false_peak = (
                _generate_false_peak_candidates(
                    spectrum.axis_cm1,
                    peak_positions_cm1,
                    median_fwhm_cm1=median_fwhm_cm1,
                    false_peak_height=false_peak_height,
                    seed_material=seed_material,
                    count=max_insert_count,
                )
            )
        else:
            candidate_false_centers_cm1 = ()
            candidate_false_peak = np.zeros(
                spectrum.axis_cm1.size,
                dtype="<f8",
            )
            candidate_false_peak.setflags(write=False)
        state_digest = _peak_model_digest(
            self.perturbation_id,
            spectrum.spectrum_id,
            self._config.sha256,
            DETECTOR_ID,
            source_axis_sha256,
            source_intensity_sha256,
            peak_indices,
            peak_positions_cm1,
            prominences,
            support_bounds,
            component_matrix,
            component_heights,
            component_areas,
            median_fwhm_cm1,
            candidate_false_centers_cm1,
            candidate_false_peak,
            seed_material,
        )
        return PeakFamilyPreparedState(
            perturbation_id=self.perturbation_id,
            spectrum_id=spectrum.spectrum_id,
            sweep_config_sha256=self._config.sha256,
            state_digest=state_digest,
            detector_id=DETECTOR_ID,
            source_axis_sha256=source_axis_sha256,
            source_intensity_sha256=source_intensity_sha256,
            peak_indices=peak_indices,
            peak_positions_cm1=peak_positions_cm1,
            prominences=prominences,
            support_bounds=support_bounds,
            component_matrix=component_matrix,
            component_heights=component_heights,
            component_areas=component_areas,
            median_fwhm_cm1=median_fwhm_cm1,
            candidate_false_centers_cm1=candidate_false_centers_cm1,
            candidate_false_peak=candidate_false_peak,
            deterministic_random_material=seed_material,
        )

    def _validated_source(
        self,
        spectrum: Spectrum1D,
        state: PeakFamilyPreparedState,
    ) -> None:
        if spectrum.spectrum_id != state.spectrum_id:
            raise PeakFamilyError(
                "source spectrum_id",
                "must match the prepared state",
            )
        axis_hash = _sha256_bytes(spectrum.axis_cm1)
        if axis_hash != state.source_axis_sha256:
            raise PeakFamilyError(
                "source axis",
                "must match the prepared source axis",
            )
        intensity_hash = _sha256_bytes(spectrum.intensity)
        if intensity_hash != state.source_intensity_sha256:
            raise PeakFamilyError(
                "source intensity",
                "must match the prepared source intensity",
            )

    def _weak_peak_order(self, state: PeakFamilyPreparedState) -> tuple[int, ...]:
        return tuple(
            index
            for _, _, index in sorted(
                (
                    (state.prominences[index], state.peak_positions_cm1[index], index)
                    for index in range(len(state.peak_indices))
                ),
                key=lambda item: (item[0], item[1]),
            )
        )

    def _build_result(
        self,
        spectrum: Spectrum1D,
        state: PeakFamilyPreparedState,
        alpha: float,
        output_intensity: np.ndarray,
        diagnostics: dict[str, object],
    ) -> PerturbationResult:
        output = Spectrum1D(
            spectrum_id=derive_perturbed_spectrum_id(
                spectrum.spectrum_id,
                self.perturbation_id,
                alpha,
                state.state_digest,
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
            state_digest=state.state_digest,
            axis_changed=False,
            intensity_changed=not np.array_equal(
                spectrum.intensity,
                output.intensity,
            ),
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
            raise PeakFamilyError(exc.path, exc.reason) from exc
        return result

    def _ensure_positive_alpha_not_identity(
        self,
        spectrum: Spectrum1D,
        output_intensity: np.ndarray,
        *,
        alpha: float,
        should_change: bool,
    ) -> None:
        if alpha > self._config.identity_alpha and should_change and np.array_equal(
            spectrum.intensity,
            output_intensity,
        ):
            raise PeakFamilyError(
                "positive perturbation",
                "rounds entirely to identity",
            )


class P1GlobalPeakAttenuation(_PeakFamilyBase):
    perturbation_id = "p01"

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        if not isinstance(spectrum, Spectrum1D):
            raise PeakFamilyError("spectrum", "must be Spectrum1D")
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise PeakFamilyError("alpha", "must be one of the frozen alpha values")
        self._validated_source(spectrum, validated_state)
        components = validated_state.component_matrix
        removed = alpha * np.sum(components, axis=0)
        output_intensity = np.asarray(
            spectrum.intensity - removed,
            dtype="<f8",
        )
        should_change = bool(
            alpha > 0.0 and np.sum(validated_state.component_matrix) > 0.0
        )
        self._ensure_positive_alpha_not_identity(
            spectrum,
            output_intensity,
            alpha=alpha,
            should_change=should_change,
        )
        diagnostics = {
            "internal_peak_model": validated_state.detector_id,
            "detected_peak_count": len(validated_state.peak_indices),
            "selected_or_inserted_peak_count": len(validated_state.peak_indices) if alpha > 0.0 else 0,
            "alpha": alpha,
            "source_component_area": float(sum(validated_state.component_areas)),
            "removed_component_area": float(alpha * sum(validated_state.component_areas)),
            "axis_preserved": True,
        }
        return self._build_result(
            spectrum,
            validated_state,
            alpha,
            np.array(output_intensity, dtype="<f8", copy=True),
            diagnostics,
        )


class P2SelectiveWeakPeakAttenuation(_PeakFamilyBase):
    perturbation_id = "p02"

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        if not isinstance(spectrum, Spectrum1D):
            raise PeakFamilyError("spectrum", "must be Spectrum1D")
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise PeakFamilyError("alpha", "must be one of the frozen alpha values")
        self._validated_source(spectrum, validated_state)
        weak_order = self._weak_peak_order(validated_state)
        selected_count = 0 if alpha == 0.0 else int(math.ceil(alpha * len(weak_order)))
        selected_count = min(selected_count, len(weak_order))
        selected_indices = weak_order[:selected_count]
        removed_components = np.sum(
            validated_state.component_matrix[np.asarray(selected_indices, dtype=int), :],
            axis=0,
        ) if selected_indices else np.zeros(spectrum.intensity.size, dtype="<f8")
        output_intensity = np.asarray(
            spectrum.intensity - 0.5 * removed_components,
            dtype="<f8",
        )
        self._ensure_positive_alpha_not_identity(
            spectrum,
            output_intensity,
            alpha=alpha,
            should_change=selected_count > 0,
        )
        diagnostics = {
            "internal_peak_model": validated_state.detector_id,
            "detected_peak_count": len(validated_state.peak_indices),
            "selected_or_inserted_peak_count": selected_count,
            "alpha": alpha,
            "source_component_area": float(sum(validated_state.component_areas)),
            "removed_component_area": float(
                0.5 * sum(validated_state.component_areas[index] for index in selected_indices)
            ),
            "axis_preserved": True,
        }
        return self._build_result(
            spectrum,
            validated_state,
            alpha,
            np.array(output_intensity, dtype="<f8", copy=True),
            diagnostics,
        )


class P3PeakBroadening(_PeakFamilyBase):
    perturbation_id = "p03"

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        if not isinstance(spectrum, Spectrum1D):
            raise PeakFamilyError("spectrum", "must be Spectrum1D")
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise PeakFamilyError("alpha", "must be one of the frozen alpha values")
        self._validated_source(spectrum, validated_state)
        sigma_cm1 = float(alpha * validated_state.median_fwhm_cm1)
        if alpha == 0.0:
            output_intensity = np.array(spectrum.intensity, dtype="<f8", copy=True)
            inserted_area = float(sum(validated_state.component_areas))
        else:
            broadened = np.zeros(spectrum.intensity.size, dtype="<f8")
            for center_cm1, area in zip(
                validated_state.peak_positions_cm1,
                validated_state.component_areas,
                strict=True,
            ):
                broadened += _gaussian_with_area(
                    spectrum.axis_cm1,
                    center_cm1=center_cm1,
                    sigma_cm1=sigma_cm1,
                    area=area,
                )
            removed = np.sum(validated_state.component_matrix, axis=0)
            output_intensity = np.asarray(
                spectrum.intensity - removed + broadened,
                dtype="<f8",
            )
            inserted_area = float(_component_area(spectrum.axis_cm1, broadened))
        self._ensure_positive_alpha_not_identity(
            spectrum,
            output_intensity,
            alpha=alpha,
            should_change=alpha > 0.0,
        )
        diagnostics = {
            "internal_peak_model": validated_state.detector_id,
            "detected_peak_count": len(validated_state.peak_indices),
            "selected_or_inserted_peak_count": len(validated_state.peak_indices),
            "alpha": alpha,
            "source_component_area": float(sum(validated_state.component_areas)),
            "removed_component_area": float(sum(validated_state.component_areas)),
            "inserted_component_area": inserted_area,
            "broadened_sigma_cm1": sigma_cm1,
            "axis_preserved": True,
        }
        return self._build_result(
            spectrum,
            validated_state,
            alpha,
            np.array(output_intensity, dtype="<f8", copy=True),
            diagnostics,
        )


class P4WeakPeakDeletion(_PeakFamilyBase):
    perturbation_id = "p04"

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        if not isinstance(spectrum, Spectrum1D):
            raise PeakFamilyError("spectrum", "must be Spectrum1D")
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise PeakFamilyError("alpha", "must be one of the frozen alpha values")
        self._validated_source(spectrum, validated_state)
        weak_order = self._weak_peak_order(validated_state)
        selected_count = 0 if alpha == 0.0 else int(math.ceil(alpha * len(weak_order)))
        selected_count = min(selected_count, len(weak_order))
        selected_indices = weak_order[:selected_count]
        removed = np.sum(
            validated_state.component_matrix[np.asarray(selected_indices, dtype=int), :],
            axis=0,
        ) if selected_indices else np.zeros(spectrum.intensity.size, dtype="<f8")
        output_intensity = np.asarray(
            spectrum.intensity - removed,
            dtype="<f8",
        )
        self._ensure_positive_alpha_not_identity(
            spectrum,
            output_intensity,
            alpha=alpha,
            should_change=selected_count > 0,
        )
        diagnostics = {
            "internal_peak_model": validated_state.detector_id,
            "detected_peak_count": len(validated_state.peak_indices),
            "selected_or_inserted_peak_count": selected_count,
            "alpha": alpha,
            "source_component_area": float(sum(validated_state.component_areas)),
            "removed_component_area": float(
                sum(validated_state.component_areas[index] for index in selected_indices)
            ),
            "axis_preserved": True,
        }
        return self._build_result(
            spectrum,
            validated_state,
            alpha,
            np.array(output_intensity, dtype="<f8", copy=True),
            diagnostics,
        )


class P5FalsePeakInsertion(_PeakFamilyBase):
    perturbation_id = "p05"

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        if not isinstance(spectrum, Spectrum1D):
            raise PeakFamilyError("spectrum", "must be Spectrum1D")
        validated_state = self._validated_state(state)
        if not is_frozen_alpha(self._config, alpha):
            raise PeakFamilyError("alpha", "must be one of the frozen alpha values")
        self._validated_source(spectrum, validated_state)
        selected_count = 0 if alpha == 0.0 else int(
            math.floor(alpha * len(validated_state.peak_indices))
        )
        selected_count = min(
            selected_count,
            len(validated_state.candidate_false_centers_cm1),
        )
        inserted = np.zeros(spectrum.intensity.size, dtype="<f8")
        sigma_cm1 = validated_state.median_fwhm_cm1 / (
            2.0 * math.sqrt(2.0 * math.log(2.0))
        )
        peak_height = float(
            np.median(np.asarray(validated_state.component_heights, dtype="<f8"))
        )
        for center_cm1 in validated_state.candidate_false_centers_cm1[:selected_count]:
            inserted += _gaussian_with_height(
                spectrum.axis_cm1,
                center_cm1=center_cm1,
                sigma_cm1=sigma_cm1,
                height=peak_height,
            )
        output_intensity = np.asarray(
            spectrum.intensity + inserted,
            dtype="<f8",
        )
        self._ensure_positive_alpha_not_identity(
            spectrum,
            output_intensity,
            alpha=alpha,
            should_change=selected_count > 0,
        )
        diagnostics = {
            "internal_peak_model": validated_state.detector_id,
            "detected_peak_count": len(validated_state.peak_indices),
            "selected_or_inserted_peak_count": selected_count,
            "alpha": alpha,
            "source_component_area": float(sum(validated_state.component_areas)),
            "inserted_component_area": float(_component_area(spectrum.axis_cm1, inserted)),
            "inserted_peak_centers_cm1": validated_state.candidate_false_centers_cm1[:selected_count],
            "false_peak_height": peak_height,
            "false_peak_fwhm_cm1": validated_state.median_fwhm_cm1,
            "axis_preserved": True,
        }
        return self._build_result(
            spectrum,
            validated_state,
            alpha,
            np.array(output_intensity, dtype="<f8", copy=True),
            diagnostics,
        )


__all__ = [
    "AXIS_BEHAVIOR",
    "DETECTOR_ID",
    "P1GlobalPeakAttenuation",
    "P2SelectiveWeakPeakAttenuation",
    "P3PeakBroadening",
    "P4WeakPeakDeletion",
    "P5FalsePeakInsertion",
    "PeakFamilyError",
    "PeakFamilyPreparedState",
]
