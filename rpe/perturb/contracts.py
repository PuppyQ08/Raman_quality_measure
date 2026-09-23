from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Protocol, runtime_checkable

import numpy as np

from rpe.evaluation import JsonValue, Spectrum1D
from rpe.perturb.sweep import (
    PerturbationSweepConfig,
    PerturbationSweepConfigError,
    is_frozen_alpha,
    validate_perturbation_sweep_config,
)


_HEX_DIGITS = frozenset("0123456789abcdef")


class PerturbationContractError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise PerturbationContractError(
            path,
            "must be a nonempty string",
        )
    return value


def _finite_real(path: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PerturbationContractError(
            path,
            "must be a finite real number",
        )
    return float(value)


def _lower_hex_64(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise PerturbationContractError(
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _freeze_json(path: str, value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PerturbationContractError(
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
            raise PerturbationContractError(
                path,
                "mapping keys must be nonempty strings",
            )
        return MappingProxyType(
            {
                key: _freeze_json(f"{path}.{key}", item)
                for key, item in sorted(value.items())
            }
        )
    raise PerturbationContractError(
        path,
        "must be canonical-JSON-compatible",
    )


def _length_prefixed_utf8(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


class AxisBehavior(str, Enum):
    PRESERVE = "preserve"
    TRANSFORM = "transform"


@dataclass(frozen=True)
class PerturbationContext:
    sweep_id: str
    sweep_config_sha256: str
    global_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "sweep_id",
            _nonempty_string("sweep_id", self.sweep_id),
        )
        object.__setattr__(
            self,
            "sweep_config_sha256",
            _lower_hex_64(
                "sweep_config_sha256",
                self.sweep_config_sha256,
            ),
        )
        if (
            isinstance(self.global_seed, bool)
            or not isinstance(self.global_seed, int)
            or self.global_seed < 0
        ):
            raise PerturbationContractError(
                "global_seed",
                "must be a nonnegative integer",
            )


@runtime_checkable
class PerturbationState(Protocol):
    perturbation_id: str
    spectrum_id: str
    sweep_config_sha256: str
    state_digest: str


@dataclass(frozen=True)
class PerturbationResult:
    source_spectrum_id: str
    perturbation_id: str
    alpha: float
    output: Spectrum1D
    axis_behavior: AxisBehavior
    state_digest: str
    axis_changed: bool
    intensity_changed: bool
    diagnostics: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_spectrum_id",
            _nonempty_string(
                "source_spectrum_id",
                self.source_spectrum_id,
            ),
        )
        object.__setattr__(
            self,
            "perturbation_id",
            _nonempty_string(
                "perturbation_id",
                self.perturbation_id,
            ),
        )
        object.__setattr__(
            self,
            "alpha",
            _finite_real("alpha", self.alpha),
        )
        if not isinstance(self.output, Spectrum1D):
            raise PerturbationContractError(
                "output",
                "must be Spectrum1D",
            )
        if not isinstance(self.axis_behavior, AxisBehavior):
            raise PerturbationContractError(
                "axis_behavior",
                "must be AxisBehavior",
            )
        object.__setattr__(
            self,
            "state_digest",
            _lower_hex_64("state_digest", self.state_digest),
        )
        if not isinstance(self.axis_changed, bool):
            raise PerturbationContractError(
                "axis_changed",
                "must be bool",
            )
        if not isinstance(self.intensity_changed, bool):
            raise PerturbationContractError(
                "intensity_changed",
                "must be bool",
            )
        diagnostics = _freeze_json("diagnostics", self.diagnostics)
        if not isinstance(diagnostics, Mapping):
            raise PerturbationContractError(
                "diagnostics",
                "must be a mapping",
            )
        object.__setattr__(self, "diagnostics", diagnostics)


def derive_perturbed_spectrum_id(
    source_spectrum_id: str,
    perturbation_id: str,
    alpha: float,
    state_digest: str,
    sweep_config_sha256: str,
) -> str:
    source_id = _nonempty_string(
        "source_spectrum_id",
        source_spectrum_id,
    )
    perturbation_name = _nonempty_string(
        "perturbation_id",
        perturbation_id,
    )
    alpha_value = _finite_real("alpha", alpha)
    state_hash = _lower_hex_64("state_digest", state_digest)
    config_hash = _lower_hex_64(
        "sweep_config_sha256",
        sweep_config_sha256,
    )
    payload = (
        b"rpe-perturbed-spectrum-id-v1\0"
        + _length_prefixed_utf8(source_id)
        + _length_prefixed_utf8(perturbation_name)
        + struct.pack("<d", alpha_value)
        + bytes.fromhex(state_hash)
        + bytes.fromhex(config_hash)
    )
    return "perturbed-" + hashlib.sha256(payload).hexdigest()


@runtime_checkable
class Perturbation(Protocol):
    perturbation_id: str
    axis_behavior: AxisBehavior

    def prepare(
        self,
        spectrum: Spectrum1D,
        context: PerturbationContext,
    ) -> PerturbationState:
        pass

    def apply(
        self,
        spectrum: Spectrum1D,
        alpha: float,
        state: PerturbationState,
    ) -> PerturbationResult:
        pass


def validate_perturbation_result(
    source: Spectrum1D,
    state: PerturbationState,
    result: PerturbationResult,
    config: PerturbationSweepConfig,
) -> None:
    if not isinstance(source, Spectrum1D):
        raise PerturbationContractError(
            "source",
            "must be Spectrum1D",
        )
    if not isinstance(state, PerturbationState):
        raise PerturbationContractError(
            "state",
            "must implement PerturbationState",
        )
    if not isinstance(result, PerturbationResult):
        raise PerturbationContractError(
            "result",
            "must be PerturbationResult",
        )
    if not isinstance(config, PerturbationSweepConfig):
        raise PerturbationContractError(
            "config",
            "must be PerturbationSweepConfig",
        )
    try:
        validate_perturbation_sweep_config(config)
    except PerturbationSweepConfigError as exc:
        raise PerturbationContractError(
            f"config.{exc.path}",
            exc.reason,
        ) from exc
    if not is_frozen_alpha(config, result.alpha):
        raise PerturbationContractError(
            "alpha",
            "must be one of the frozen alpha values",
        )
    state_perturbation_id = _nonempty_string(
        "state.perturbation_id",
        state.perturbation_id,
    )
    state_spectrum_id = _nonempty_string(
        "state.spectrum_id",
        state.spectrum_id,
    )
    state_config_sha = _lower_hex_64(
        "state.sweep_config_sha256",
        state.sweep_config_sha256,
    )
    state_digest = _lower_hex_64(
        "state.state_digest",
        state.state_digest,
    )
    if state_perturbation_id not in config.perturbation_ids:
        raise PerturbationContractError(
            "state.perturbation_id",
            "must be one of the frozen perturbation IDs",
        )
    if state_spectrum_id != source.spectrum_id:
        raise PerturbationContractError(
            "state.spectrum_id",
            "must match source.spectrum_id",
        )
    if result.source_spectrum_id != source.spectrum_id:
        raise PerturbationContractError(
            "result.source_spectrum_id",
            "must match source.spectrum_id",
        )
    if result.perturbation_id != state_perturbation_id:
        raise PerturbationContractError(
            "result.perturbation_id",
            "must match state.perturbation_id",
        )
    if state_config_sha != config.sha256:
        raise PerturbationContractError(
            "state.sweep_config_sha256",
            "must match config.sha256",
        )
    if result.state_digest != state_digest:
        raise PerturbationContractError(
            "result.state_digest",
            "must match state.state_digest",
        )
    expected_output_id = derive_perturbed_spectrum_id(
        source.spectrum_id,
        result.perturbation_id,
        result.alpha,
        state_digest,
        config.sha256,
    )
    if result.output.spectrum_id != expected_output_id:
        raise PerturbationContractError(
            "result.output.spectrum_id",
            "must match the deterministic perturbed spectrum identity",
        )
    axis_changed = not np.array_equal(
        source.axis_cm1,
        result.output.axis_cm1,
    )
    intensity_changed = not np.array_equal(
        source.intensity,
        result.output.intensity,
    )
    if result.axis_changed != axis_changed:
        raise PerturbationContractError(
            "result.axis_changed",
            "must equal the exact axis comparison",
        )
    if result.intensity_changed != intensity_changed:
        raise PerturbationContractError(
            "result.intensity_changed",
            "must equal the exact intensity comparison",
        )
    if result.axis_behavior is AxisBehavior.PRESERVE and axis_changed:
        raise PerturbationContractError(
            "result.axis_behavior",
            "preserve perturbations must not change the axis",
        )
    if result.alpha == config.identity_alpha and (
        axis_changed or intensity_changed
    ):
        raise PerturbationContractError(
            "result.alpha",
            "alpha zero must preserve exact axis and intensity identity",
        )
