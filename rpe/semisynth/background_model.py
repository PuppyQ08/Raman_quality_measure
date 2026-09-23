from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from pybaselines import Baseline
from scipy.stats import chi2
from sklearn.covariance import LedoitWolf

from rpe.io.schema import JsonValue
from rpe.semisynth.contracts import Phase2Config
from rpe.semisynth.rruff_pairs import RruffEqualAxisPair
from rpe.semisynth.splits import RruffPairRole


_HEX = frozenset("0123456789abcdef")
_STRATA = frozenset({"green_514", "green_532", "nir_780", "nir_785"})
_MODEL_ID_DOMAIN = b"rpe-phase2-background-parameter-model-v1\0"


class BackgroundModelError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class AttemptStatus(str, Enum):
    VALID = "valid"
    FAILED = "failed"


def _nonempty(path: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise BackgroundModelError(path, "must be a nonempty string")
    return value


def _lower_hex(path: str, value: object) -> str:
    parsed = _nonempty(path, value)
    if len(parsed) != 64 or any(character not in _HEX for character in parsed):
        raise BackgroundModelError(path, "must be lowercase SHA256")
    return parsed


def _readonly_vector(path: str, value: object, *, size: int | None = None) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise BackgroundModelError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise BackgroundModelError(f"{path}.dtype", "must be float64")
    if value.ndim != 1 or (size is not None and value.size != size):
        raise BackgroundModelError(f"{path}.shape", "has the wrong shape")
    if not np.isfinite(value).all():
        raise BackgroundModelError(f"{path}.finite", "contains non-finite values")
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _readonly_matrix(path: str, value: object, *, size: int) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise BackgroundModelError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8") or value.shape != (size, size):
        raise BackgroundModelError(f"{path}.shape", "must be a float64 square matrix")
    if not np.isfinite(value).all():
        raise BackgroundModelError(f"{path}.finite", "contains non-finite values")
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _freeze_diagnostics(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise BackgroundModelError("diagnostics", "must be a mapping")
    frozen: dict[str, JsonValue] = {}
    for key, item in sorted(value.items()):
        parsed_key = _nonempty("diagnostics.key", key)
        if item is not None and not isinstance(item, (bool, int, float, str)):
            raise BackgroundModelError(
                f"diagnostics.{parsed_key}", "must be a scalar JSON value"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise BackgroundModelError(
                f"diagnostics.{parsed_key}", "must be finite"
            )
        frozen[parsed_key] = item
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class BackgroundExtractionAttempt:
    pair_id: str
    extractor_id: str
    excitation_stratum: str
    status: AttemptStatus
    baseline_raw: np.ndarray | None
    diagnostics: Mapping[str, JsonValue]
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        for name in ("pair_id", "extractor_id", "excitation_stratum"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.excitation_stratum not in _STRATA:
            raise BackgroundModelError("excitation_stratum", "is not frozen")
        if not isinstance(self.status, AttemptStatus):
            raise BackgroundModelError("status", "must be AttemptStatus")
        if self.status is AttemptStatus.VALID:
            if self.baseline_raw is None or self.error_code is not None or self.error_message is not None:
                raise BackgroundModelError("attempt", "valid attempt fields disagree")
            object.__setattr__(
                self, "baseline_raw", _readonly_vector("baseline_raw", self.baseline_raw)
            )
        else:
            if self.baseline_raw is not None:
                raise BackgroundModelError("attempt", "failed attempt has a baseline")
            object.__setattr__(self, "error_code", _nonempty("error_code", self.error_code))
            object.__setattr__(
                self, "error_message", _nonempty("error_message", self.error_message)
            )
        object.__setattr__(self, "diagnostics", _freeze_diagnostics(self.diagnostics))


@dataclass(frozen=True)
class BackgroundParameterAttempt:
    pair_id: str
    extractor_id: str
    excitation_stratum: str
    status: AttemptStatus
    vector: np.ndarray | None
    reconstruction_nrmse: float | None
    input_sha256: str
    error_code: str | None
    error_message: str | None

    def __post_init__(self) -> None:
        for name in ("pair_id", "extractor_id", "excitation_stratum"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.excitation_stratum not in _STRATA:
            raise BackgroundModelError("excitation_stratum", "is not frozen")
        if not isinstance(self.status, AttemptStatus):
            raise BackgroundModelError("status", "must be AttemptStatus")
        object.__setattr__(self, "input_sha256", _lower_hex("input_sha256", self.input_sha256))
        if self.status is AttemptStatus.VALID:
            if (
                self.vector is None
                or self.reconstruction_nrmse is None
                or not math.isfinite(float(self.reconstruction_nrmse))
                or self.reconstruction_nrmse < 0.0
                or self.error_code is not None
                or self.error_message is not None
            ):
                raise BackgroundModelError("attempt", "valid parameter fields disagree")
            object.__setattr__(self, "vector", _readonly_vector("vector", self.vector, size=8))
        else:
            if self.vector is not None or self.reconstruction_nrmse is not None:
                raise BackgroundModelError("attempt", "failed parameter has outputs")
            object.__setattr__(self, "error_code", _nonempty("error_code", self.error_code))
            object.__setattr__(
                self, "error_message", _nonempty("error_message", self.error_message)
            )


@dataclass(frozen=True)
class BackgroundParameterModelReceipt:
    model_id: str
    extractor_id: str
    excitation_stratum: str
    config_sha256: str
    input_digest: str
    input_count: int
    valid_count: int
    invalid_count: int
    valid_fraction: float
    median_reconstruction_nrmse: float | None
    p95_reconstruction_nrmse: float | None
    mean: np.ndarray | None
    covariance: np.ndarray | None
    precision: np.ndarray | None
    shrinkage: float | None
    log_amplitude_bounds: tuple[float, float] | None
    mahalanobis_squared_max: float
    gate_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _lower_hex("model_id", self.model_id))
        object.__setattr__(self, "config_sha256", _lower_hex("config_sha256", self.config_sha256))
        object.__setattr__(self, "input_digest", _lower_hex("input_digest", self.input_digest))
        _nonempty("extractor_id", self.extractor_id)
        if self.excitation_stratum not in _STRATA:
            raise BackgroundModelError("excitation_stratum", "is not frozen")
        if (
            self.input_count <= 0
            or self.valid_count < 0
            or self.invalid_count != self.input_count - self.valid_count
            or self.valid_fraction != self.valid_count / self.input_count
        ):
            raise BackgroundModelError("counts", "are inconsistent")
        model_values = (self.mean, self.covariance, self.precision, self.shrinkage, self.log_amplitude_bounds)
        if all(value is None for value in model_values):
            if not self.gate_failures:
                raise BackgroundModelError("model", "passing receipt lacks model values")
        else:
            if any(value is None for value in model_values):
                raise BackgroundModelError("model", "has partial model values")
            object.__setattr__(self, "mean", _readonly_vector("mean", self.mean, size=8))
            object.__setattr__(self, "covariance", _readonly_matrix("covariance", self.covariance, size=8))
            object.__setattr__(self, "precision", _readonly_matrix("precision", self.precision, size=8))
            assert self.log_amplitude_bounds is not None
            lower, upper = self.log_amplitude_bounds
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise BackgroundModelError("log_amplitude_bounds", "are invalid")
        if not isinstance(self.gate_failures, tuple):
            raise BackgroundModelError("gate_failures", "must be a tuple")

    @property
    def gate_passed(self) -> bool:
        return not self.gate_failures


@dataclass(frozen=True, eq=False)
class BackgroundParameterSample:
    model_id: str
    extractor_id: str
    excitation_stratum: str
    sample_index: int
    seed_sha256: str
    rejected_attempts: int
    vector: np.ndarray
    mahalanobis_squared: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _lower_hex("model_id", self.model_id))
        object.__setattr__(self, "seed_sha256", _lower_hex("seed_sha256", self.seed_sha256))
        _nonempty("extractor_id", self.extractor_id)
        if self.excitation_stratum not in _STRATA:
            raise BackgroundModelError("excitation_stratum", "is not frozen")
        if isinstance(self.sample_index, bool) or not isinstance(self.sample_index, int) or self.sample_index < 0:
            raise BackgroundModelError("sample_index", "must be nonnegative integer")
        if self.rejected_attempts < 0:
            raise BackgroundModelError("rejected_attempts", "must be nonnegative")
        object.__setattr__(self, "vector", _readonly_vector("vector", self.vector, size=8))
        if not math.isfinite(self.mahalanobis_squared) or self.mahalanobis_squared < 0.0:
            raise BackgroundModelError("mahalanobis_squared", "must be nonnegative finite")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BackgroundParameterSample):
            return NotImplemented
        return (
            self.model_id == other.model_id
            and self.extractor_id == other.extractor_id
            and self.excitation_stratum == other.excitation_stratum
            and self.sample_index == other.sample_index
            and self.seed_sha256 == other.seed_sha256
            and self.rejected_attempts == other.rejected_attempts
            and self.mahalanobis_squared == other.mahalanobis_squared
            and np.array_equal(self.vector, other.vector)
        )


@dataclass(frozen=True)
class BackgroundFitArtifactSummary:
    path: Path
    run_id: str
    pair_count: int
    attempt_count: int
    model_count: int
    gate_passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise BackgroundModelError("path", "must be pathlib.Path")
        object.__setattr__(self, "run_id", _lower_hex("run_id", self.run_id))
        if min(self.pair_count, self.attempt_count, self.model_count) <= 0:
            raise BackgroundModelError("artifact counts", "must be positive")


def _rms(value: np.ndarray) -> float:
    scale = float(np.max(np.abs(value)))
    if not math.isfinite(scale) or scale == 0.0:
        return 0.0
    return scale * math.sqrt(float(np.mean((value / scale) ** 2)))


def _extractor_kwargs(config: Phase2Config, extractor_id: str) -> dict[str, object]:
    dependencies = config.document["dependencies"]
    assert isinstance(dependencies, Mapping)
    pybaselines = dependencies["pybaselines"]
    assert isinstance(pybaselines, Mapping)
    values = pybaselines[extractor_id]
    assert isinstance(values, Mapping)
    return dict(values)


def extract_pair_backgrounds(
    pair: RruffEqualAxisPair, config: Phase2Config
) -> tuple[BackgroundExtractionAttempt, ...]:
    if not isinstance(pair, RruffEqualAxisPair):
        raise BackgroundModelError("pair", "must be RruffEqualAxisPair")
    if pair.excitation_stratum is None:
        raise BackgroundModelError("pair.excitation_stratum", "is unstratified")
    if pair.role is not RruffPairRole.EXTRACTION_FIT:
        raise BackgroundModelError(
            "pair.role", "only extraction_fit pairs may fit extractors"
        )
    attempts: list[BackgroundExtractionAttempt] = []
    for extractor_id in config.extractor_ids:
        try:
            baseline, params = getattr(
                Baseline(x_data=pair.axis_cm1), extractor_id
            )(pair.raw_intensity, **_extractor_kwargs(config, extractor_id))
            baseline_array = np.asarray(baseline, dtype="<f8")
            if baseline_array.shape != pair.axis_cm1.shape or not np.isfinite(baseline_array).all():
                raise BackgroundModelError("baseline_raw", "is invalid")
            if extractor_id in {"airpls", "arpls"}:
                history = np.asarray(params["tol_history"], dtype="<f8")
                diagnostics: Mapping[str, JsonValue] = {
                    "iteration_count": int(history.size),
                    "final_tolerance": (float(history[-1]) if history.size else None),
                }
            else:
                diagnostics = {"half_window": int(params["half_window"])}
            attempts.append(
                BackgroundExtractionAttempt(
                    pair_id=pair.pair_id,
                    extractor_id=extractor_id,
                    excitation_stratum=pair.excitation_stratum,
                    status=AttemptStatus.VALID,
                    baseline_raw=baseline_array,
                    diagnostics=diagnostics,
                    error_code=None,
                    error_message=None,
                )
            )
        except Exception as error:
            attempts.append(
                BackgroundExtractionAttempt(
                    pair_id=pair.pair_id,
                    extractor_id=extractor_id,
                    excitation_stratum=pair.excitation_stratum,
                    status=AttemptStatus.FAILED,
                    baseline_raw=None,
                    diagnostics={},
                    error_code=type(error).__name__,
                    error_message=str(error) or type(error).__name__,
                )
            )
    return tuple(attempts)


def _parameter_input_sha(
    pair: RruffEqualAxisPair, extraction: BackgroundExtractionAttempt
) -> str:
    digest = hashlib.sha256()
    digest.update(b"rpe-phase2-background-parameter-input-v1\0")
    for value in (
        pair.pair_id,
        extraction.extractor_id,
        pair.raw_source_member_sha256,
        pair.processed_source_member_sha256,
    ):
        raw = value.encode("utf-8")
        digest.update(struct.pack("<Q", len(raw)))
        digest.update(raw)
    digest.update(pair.axis_cm1.tobytes())
    digest.update(pair.processed_intensity.tobytes())
    if extraction.baseline_raw is not None:
        digest.update(extraction.baseline_raw.tobytes())
    return digest.hexdigest()


def parameterize_background(
    pair: RruffEqualAxisPair,
    extraction: BackgroundExtractionAttempt,
    config: Phase2Config,
) -> BackgroundParameterAttempt:
    input_sha = _parameter_input_sha(pair, extraction)
    if (
        extraction.pair_id != pair.pair_id
        or extraction.excitation_stratum != pair.excitation_stratum
        or extraction.extractor_id not in config.extractor_ids
    ):
        raise BackgroundModelError("extraction", "does not match pair/config")
    if extraction.status is AttemptStatus.FAILED:
        return BackgroundParameterAttempt(
            pair_id=pair.pair_id,
            extractor_id=extraction.extractor_id,
            excitation_stratum=extraction.excitation_stratum,
            status=AttemptStatus.FAILED,
            vector=None,
            reconstruction_nrmse=None,
            input_sha256=input_sha,
            error_code=extraction.error_code,
            error_message=extraction.error_message,
        )
    try:
        assert extraction.baseline_raw is not None
        q = extraction.baseline_raw - float(np.min(extraction.baseline_raw))
        q_rms = _rms(q)
        template_rms = _rms(pair.processed_intensity)
        if q_rms <= 0.0:
            raise BackgroundModelError("q.rms", "must be positive")
        if template_rms <= 0.0:
            raise BackgroundModelError("s_template.rms", "must be positive")
        normalized_axis = (
            2.0
            * (pair.axis_cm1 - pair.axis_cm1[0])
            / (pair.axis_cm1[-1] - pair.axis_cm1[0])
            - 1.0
        )
        q_unit = q / q_rms
        design = np.polynomial.legendre.legvander(
            normalized_axis, config.legendre_degree
        )
        coefficients = np.linalg.lstsq(design, q_unit, rcond=None)[0]
        reconstructed = design @ coefficients
        reconstruction_nrmse = _rms(reconstructed - q_unit)
        vector = np.concatenate(
            (coefficients, np.array([math.log(q_rms / template_rms)], dtype="<f8"))
        ).astype("<f8", copy=False)
        return BackgroundParameterAttempt(
            pair_id=pair.pair_id,
            extractor_id=extraction.extractor_id,
            excitation_stratum=extraction.excitation_stratum,
            status=AttemptStatus.VALID,
            vector=vector,
            reconstruction_nrmse=reconstruction_nrmse,
            input_sha256=input_sha,
            error_code=None,
            error_message=None,
        )
    except Exception as error:
        return BackgroundParameterAttempt(
            pair_id=pair.pair_id,
            extractor_id=extraction.extractor_id,
            excitation_stratum=extraction.excitation_stratum,
            status=AttemptStatus.FAILED,
            vector=None,
            reconstruction_nrmse=None,
            input_sha256=input_sha,
            error_code=(error.path if isinstance(error, BackgroundModelError) else type(error).__name__),
            error_message=str(error) or type(error).__name__,
        )


def _canonical_json(value: object) -> bytes:
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


def _attempt_document(attempt: BackgroundParameterAttempt) -> dict[str, object]:
    return {
        "error_code": attempt.error_code,
        "extractor_id": attempt.extractor_id,
        "excitation_stratum": attempt.excitation_stratum,
        "input_sha256": attempt.input_sha256,
        "pair_id": attempt.pair_id,
        "reconstruction_nrmse": attempt.reconstruction_nrmse,
        "status": attempt.status.value,
        "vector": attempt.vector.tolist() if attempt.vector is not None else None,
    }


def fit_background_parameter_model(
    attempts: Sequence[BackgroundParameterAttempt],
    *,
    extractor_id: str,
    excitation_stratum: str,
    config: Phase2Config,
) -> BackgroundParameterModelReceipt:
    ordered = tuple(sorted(attempts, key=lambda value: value.pair_id.encode("utf-8")))
    if not ordered:
        raise BackgroundModelError("attempts", "must not be empty")
    if len({attempt.pair_id for attempt in ordered}) != len(ordered):
        raise BackgroundModelError("attempts.pair_id", "must be unique")
    if extractor_id not in config.extractor_ids or excitation_stratum not in _STRATA:
        raise BackgroundModelError("model key", "is not frozen")
    if any(
        attempt.extractor_id != extractor_id
        or attempt.excitation_stratum != excitation_stratum
        for attempt in ordered
    ):
        raise BackgroundModelError("attempts", "contain another model key")
    input_payload = b"".join(_canonical_json(_attempt_document(attempt)) for attempt in ordered)
    input_digest = hashlib.sha256(
        b"rpe-phase2-background-parameter-attempts-v1\0" + input_payload
    ).hexdigest()
    valid = tuple(attempt for attempt in ordered if attempt.status is AttemptStatus.VALID)
    input_count = len(ordered)
    valid_count = len(valid)
    valid_fraction = valid_count / input_count
    fit_gate = config.document["background_model"]
    assert isinstance(fit_gate, Mapping)
    fit_gate = fit_gate["fit_gate"]
    assert isinstance(fit_gate, Mapping)
    failures: list[str] = []
    min_valid_count = int(fit_gate["min_valid_count"])
    if valid_fraction < float(fit_gate["min_valid_fraction"]):
        failures.append("valid_fraction")
    if valid_count < min_valid_count:
        failures.append("valid_count")

    if valid:
        matrix = np.stack([attempt.vector for attempt in valid])
        errors = np.array(
            [attempt.reconstruction_nrmse for attempt in valid], dtype="<f8"
        )
        median_error = float(np.median(errors))
        p95_error = float(np.quantile(errors, 0.95, method="linear"))
        if median_error > float(fit_gate["median_reconstruction_nrmse_max"]):
            failures.append("median_reconstruction_nrmse")
        if p95_error > float(fit_gate["p95_reconstruction_nrmse_max"]):
            failures.append("p95_reconstruction_nrmse")
        if valid_count >= min_valid_count:
            model = LedoitWolf(
                store_precision=True, assume_centered=False, block_size=1000
            ).fit(matrix)
            background_config = config.document["background_model"]
            assert isinstance(background_config, Mapping)
            percentiles = background_config["log_amplitude_percentiles"]
            assert isinstance(percentiles, tuple)
            bounds_array = np.quantile(
                matrix[:, -1],
                tuple(float(value) for value in percentiles),
                method="linear",
            )
            bounds: tuple[float, float] | None = (
                float(bounds_array[0]),
                float(bounds_array[1]),
            )
            mean: np.ndarray | None = np.asarray(model.location_, dtype="<f8")
            covariance: np.ndarray | None = np.asarray(
                model.covariance_, dtype="<f8"
            )
            precision: np.ndarray | None = np.asarray(
                model.precision_, dtype="<f8"
            )
            shrinkage: float | None = float(model.shrinkage_)
        else:
            mean = None
            covariance = None
            precision = None
            shrinkage = None
            bounds = None
    else:
        median_error = None
        p95_error = None
        mean = None
        covariance = None
        precision = None
        shrinkage = None
        bounds = None
    background_config = config.document["background_model"]
    assert isinstance(background_config, Mapping)
    mahalanobis_max = float(
        chi2.ppf(
            float(background_config["mahalanobis_probability"]),
            df=int(background_config["latent_dimensions"]),
        )
    )
    model_document = {
        "config_sha256": config.sha256,
        "covariance": covariance.tolist() if covariance is not None else None,
        "extractor_id": extractor_id,
        "excitation_stratum": excitation_stratum,
        "gate_failures": failures,
        "input_count": input_count,
        "input_digest": input_digest,
        "log_amplitude_bounds": bounds,
        "mahalanobis_squared_max": mahalanobis_max,
        "mean": mean.tolist() if mean is not None else None,
        "median_reconstruction_nrmse": median_error,
        "p95_reconstruction_nrmse": p95_error,
        "precision": precision.tolist() if precision is not None else None,
        "shrinkage": shrinkage,
        "valid_count": valid_count,
    }
    model_id = hashlib.sha256(_MODEL_ID_DOMAIN + _canonical_json(model_document)).hexdigest()
    return BackgroundParameterModelReceipt(
        model_id=model_id,
        extractor_id=extractor_id,
        excitation_stratum=excitation_stratum,
        config_sha256=config.sha256,
        input_digest=input_digest,
        input_count=input_count,
        valid_count=valid_count,
        invalid_count=input_count - valid_count,
        valid_fraction=valid_fraction,
        median_reconstruction_nrmse=median_error,
        p95_reconstruction_nrmse=p95_error,
        mean=mean,
        covariance=covariance,
        precision=precision,
        shrinkage=shrinkage,
        log_amplitude_bounds=bounds,
        mahalanobis_squared_max=mahalanobis_max,
        gate_failures=tuple(failures),
    )


def _length_prefixed(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def sample_background_parameters(
    receipt: BackgroundParameterModelReceipt,
    sample_index: int,
    config: Phase2Config,
) -> BackgroundParameterSample:
    if receipt.config_sha256 != config.sha256:
        raise BackgroundModelError("config identity", "does not match receipt")
    if not receipt.gate_passed:
        raise BackgroundModelError("receipt.gate_failures", "model did not pass")
    assert receipt.mean is not None
    assert receipt.covariance is not None
    assert receipt.precision is not None
    assert receipt.log_amplitude_bounds is not None
    schedule = config.document["schedule"]
    assert isinstance(schedule, Mapping)
    seed_bytes = hashlib.sha256(
        str(schedule["background_rng_domain"]).encode("utf-8")
        + b"\0"
        + struct.pack("<Q", config.global_seed)
        + _length_prefixed(receipt.model_id)
        + _length_prefixed(receipt.extractor_id)
        + _length_prefixed(receipt.excitation_stratum)
        + struct.pack("<Q", sample_index)
    ).digest()
    rng = np.random.Generator(
        np.random.PCG64(int.from_bytes(seed_bytes[:16], "little"))
    )
    transform = np.linalg.cholesky(receipt.covariance)
    lower, upper = receipt.log_amplitude_bounds
    for rejected in range(10_001):
        candidate = receipt.mean + transform @ rng.standard_normal(8)
        difference = candidate - receipt.mean
        mahalanobis = float(difference @ receipt.precision @ difference)
        if (
            mahalanobis <= receipt.mahalanobis_squared_max
            and lower <= candidate[-1] <= upper
        ):
            return BackgroundParameterSample(
                model_id=receipt.model_id,
                extractor_id=receipt.extractor_id,
                excitation_stratum=receipt.excitation_stratum,
                sample_index=sample_index,
                seed_sha256=seed_bytes.hex(),
                rejected_attempts=rejected,
                vector=np.asarray(candidate, dtype="<f8"),
                mahalanobis_squared=mahalanobis,
            )
    raise BackgroundModelError("sampling", "rejection limit exceeded")


def reconstruct_background(
    sample: BackgroundParameterSample,
    axis_cm1: np.ndarray,
    s_template: np.ndarray,
) -> np.ndarray:
    axis = _readonly_vector("axis_cm1", axis_cm1)
    template = _readonly_vector("s_template", s_template, size=axis.size)
    if axis.size < 2 or not np.all(np.diff(axis) > 0.0):
        raise BackgroundModelError("axis_cm1.increasing", "must be strict")
    template_rms = _rms(template)
    if template_rms <= 0.0:
        raise BackgroundModelError("s_template.rms", "must be positive")
    normalized_axis = 2.0 * (axis - axis[0]) / (axis[-1] - axis[0]) - 1.0
    shape = np.polynomial.legendre.legval(normalized_axis, sample.vector[:7])
    shape = shape - float(np.min(shape))
    shape_rms = _rms(shape)
    if shape_rms <= 0.0:
        raise BackgroundModelError("sample.shape.rms", "must be positive")
    background = (
        shape
        / shape_rms
        * math.exp(float(sample.vector[-1]))
        * template_rms
    ).astype("<f8", copy=False)
    background = np.ascontiguousarray(background).copy()
    background.setflags(write=False)
    return background


def _pair_identity_document(pair: RruffEqualAxisPair) -> dict[str, object]:
    return {
        "axis_relation": pair.axis_relation,
        "axis_sha256": hashlib.sha256(pair.axis_cm1.tobytes()).hexdigest(),
        "excitation_stratum": pair.excitation_stratum,
        "group_key": pair.group_key,
        "pair_id": pair.pair_id,
        "processed_record_id": pair.processed_record_id,
        "processed_sha256": hashlib.sha256(
            pair.processed_intensity.tobytes()
        ).hexdigest(),
        "processed_source_member_sha256": pair.processed_source_member_sha256,
        "processed_template_qualifier": pair.processed_template_qualifier,
        "raw_record_id": pair.raw_record_id,
        "raw_sha256": hashlib.sha256(pair.raw_intensity.tobytes()).hexdigest(),
        "raw_source_member_sha256": pair.raw_source_member_sha256,
        "role": pair.role.value,
    }


def _receipt_document(
    receipt: BackgroundParameterModelReceipt,
) -> dict[str, object]:
    return {
        "config_sha256": receipt.config_sha256,
        "covariance": (
            receipt.covariance.tolist() if receipt.covariance is not None else None
        ),
        "excitation_stratum": receipt.excitation_stratum,
        "extractor_id": receipt.extractor_id,
        "gate_failures": list(receipt.gate_failures),
        "input_count": receipt.input_count,
        "input_digest": receipt.input_digest,
        "invalid_count": receipt.invalid_count,
        "log_amplitude_bounds": receipt.log_amplitude_bounds,
        "mahalanobis_squared_max": receipt.mahalanobis_squared_max,
        "mean": receipt.mean.tolist() if receipt.mean is not None else None,
        "median_reconstruction_nrmse": receipt.median_reconstruction_nrmse,
        "model_id": receipt.model_id,
        "p95_reconstruction_nrmse": receipt.p95_reconstruction_nrmse,
        "precision": (
            receipt.precision.tolist() if receipt.precision is not None else None
        ),
        "shrinkage": receipt.shrinkage,
        "valid_count": receipt.valid_count,
        "valid_fraction": receipt.valid_fraction,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_background_fit_artifact(
    pairs: Sequence[RruffEqualAxisPair],
    output_root: Path,
    *,
    config: Phase2Config,
    project_root: Path,
) -> BackgroundFitArtifactSummary:
    ordered_pairs = tuple(
        sorted(pairs, key=lambda value: value.pair_id.encode("utf-8"))
    )
    if not ordered_pairs or len({pair.pair_id for pair in ordered_pairs}) != len(
        ordered_pairs
    ):
        raise BackgroundModelError("pairs", "must be nonempty and unique")
    if any(pair.role is not RruffPairRole.EXTRACTION_FIT for pair in ordered_pairs):
        raise BackgroundModelError("pairs.role", "must all be extraction_fit")
    if any(pair.excitation_stratum is None for pair in ordered_pairs):
        raise BackgroundModelError("pairs.excitation_stratum", "must be stratified")
    project_root = Path(project_root)
    code_paths = (
        "rpe/semisynth/contracts.py",
        "rpe/semisynth/splits.py",
        "rpe/semisynth/rruff_pairs.py",
        "rpe/semisynth/background_model.py",
    )
    code_identity = {
        path: _file_sha256(project_root / path) for path in code_paths
    }
    pair_documents = tuple(_pair_identity_document(pair) for pair in ordered_pairs)
    pair_digest = hashlib.sha256(
        b"rpe-phase2-background-fit-pairs-v1\0"
        + b"".join(_canonical_json(document) for document in pair_documents)
    ).hexdigest()
    identity = {
        "code_identity": code_identity,
        "config_sha256": config.sha256,
        "pair_count": len(ordered_pairs),
        "pair_digest": pair_digest,
        "schema_version": "phase2-background-fit-artifact-v1",
    }
    run_id = hashlib.sha256(
        b"rpe-phase2-background-fit-run-v1\0" + _canonical_json(identity)
    ).hexdigest()
    path = Path(output_root) / f"phase2-background-fit-{run_id}"
    if path.exists():
        raise BackgroundModelError("output", "run path already exists")
    path.mkdir(parents=True)

    attempts: list[BackgroundParameterAttempt] = []
    for pair in ordered_pairs:
        for extraction in extract_pair_backgrounds(pair, config):
            attempts.append(parameterize_background(pair, extraction, config))
    attempts.sort(
        key=lambda value: (
            value.extractor_id.encode("utf-8"),
            value.excitation_stratum.encode("utf-8"),
            value.pair_id.encode("utf-8"),
        )
    )
    receipts: list[BackgroundParameterModelReceipt] = []
    for extractor_id in config.extractor_ids:
        for stratum in sorted(_STRATA):
            current = tuple(
                attempt
                for attempt in attempts
                if attempt.extractor_id == extractor_id
                and attempt.excitation_stratum == stratum
            )
            if not current:
                raise BackgroundModelError(
                    "pairs", f"missing {extractor_id}/{stratum}"
                )
            receipts.append(
                fit_background_parameter_model(
                    current,
                    extractor_id=extractor_id,
                    excitation_stratum=stratum,
                    config=config,
                )
            )
    receipt_documents = tuple(_receipt_document(receipt) for receipt in receipts)
    gate_passed = all(receipt.gate_passed for receipt in receipts)
    manifest = {
        **identity,
        "attempt_count": len(attempts),
        "extractor_ids": list(config.extractor_ids),
        "model_count": len(receipts),
        "run_id": run_id,
        "strata": sorted(_STRATA),
    }
    gate = {
        "gate_passed": gate_passed,
        "model_gates": [
            {
                "excitation_stratum": receipt.excitation_stratum,
                "extractor_id": receipt.extractor_id,
                "gate_failures": list(receipt.gate_failures),
                "model_id": receipt.model_id,
            }
            for receipt in receipts
        ],
        "run_id": run_id,
    }
    payloads = {
        "gate.json": _canonical_json(gate),
        "manifest.json": _canonical_json(manifest),
        "model_receipts.json": _canonical_json(
            {"models": receipt_documents, "run_id": run_id}
        ),
        "parameter_attempts.jsonl": b"".join(
            _canonical_json(_attempt_document(attempt)) for attempt in attempts
        ),
    }
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    checksum_lines = "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
        for name in sorted(payloads)
    )
    (path / "SHA256SUMS").write_text(checksum_lines, encoding="utf-8")
    marker = {
        "gate_passed": gate_passed,
        "run_id": run_id,
        "status": "complete" if gate_passed else "failed",
    }
    (path / ("complete.json" if gate_passed else "failed.json")).write_bytes(
        _canonical_json(marker)
    )
    return BackgroundFitArtifactSummary(
        path=path,
        run_id=run_id,
        pair_count=len(ordered_pairs),
        attempt_count=len(attempts),
        model_count=len(receipts),
        gate_passed=gate_passed,
    )


__all__ = [
    "AttemptStatus",
    "BackgroundExtractionAttempt",
    "BackgroundModelError",
    "BackgroundParameterAttempt",
    "BackgroundParameterModelReceipt",
    "BackgroundParameterSample",
    "BackgroundFitArtifactSummary",
    "build_background_fit_artifact",
    "extract_pair_backgrounds",
    "fit_background_parameter_model",
    "parameterize_background",
    "reconstruct_background",
    "sample_background_parameters",
]
