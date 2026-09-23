from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from rpe.io.schema import JsonValue


CONFIG_SCHEMA_VERSION = "phase2-semisynth-config-v1"
CONFIG_BYTE_COUNT = 4722
CONFIG_SHA256 = "8ba338509dbbb4b74a8675f1875bf1048d40359ea4f617b0d087b146f60b0af1"
_HEX = frozenset("0123456789abcdef")
_EXTRACTORS = ("airpls", "arpls", "mor")
_STRATA = ("green_514", "green_532", "nir_780", "nir_785")
_QUALIFIERS = (
    "algorithmically_processed_signal_template_not_physical_clean_gt",
    "high_snr_acquisition_template_not_clean_gt",
)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LARGE_SOURCE_MIN_BYTES = 1_000_000_000


class Phase2ConfigError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class SemiSyntheticContractError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty(path: str, value: object, error_type: type[ValueError]) -> str:
    if not isinstance(value, str) or not value:
        raise error_type(path, "must be a nonempty string")
    return value


def _lower_hex(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise Phase2ConfigError(
            path, "must be a lowercase 64-character hexadecimal string"
        )
    return value


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase2ConfigError(path, "must be an object")
    return value


def _exact_keys(
    path: str, value: object, expected: tuple[str, ...]
) -> Mapping[str, object]:
    parsed = _object(path, value)
    if tuple(sorted(parsed)) != tuple(sorted(expected)):
        raise Phase2ConfigError(path, "has an unexpected key set")
    return parsed


def _positive_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase2ConfigError(path, "must be a positive integer")
    return value


def _finite(path: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Phase2ConfigError(path, "must be a finite real number")
    return float(value)


def _freeze_json(path: str, value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SemiSyntheticContractError(path, "must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(f"{path}[{index}]", item)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in sorted(value.items()):
            parsed_key = _nonempty(
                f"{path}.key", key, SemiSyntheticContractError
            )
            frozen[parsed_key] = _freeze_json(f"{path}.{parsed_key}", item)
        return MappingProxyType(frozen)
    raise SemiSyntheticContractError(path, "must be JSON-compatible")


def _vector(path: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise SemiSyntheticContractError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise SemiSyntheticContractError(
            f"{path}.dtype", "must be little-endian float64"
        )
    if value.ndim != 1:
        raise SemiSyntheticContractError(
            f"{path}.dimension", "must be one-dimensional"
        )
    if value.size == 0:
        raise SemiSyntheticContractError(f"{path}.shape", "must be nonempty")
    if not np.isfinite(value).all():
        raise SemiSyntheticContractError(f"{path}.finite", "contains non-finite values")
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "path", _nonempty("source.path", self.path, Phase2ConfigError)
        )
        if Path(self.path).is_absolute() or ".." in Path(self.path).parts:
            raise Phase2ConfigError("source.path", "must be normalized and relative")
        object.__setattr__(
            self, "byte_count", _positive_int("source.byte_count", self.byte_count)
        )
        object.__setattr__(self, "sha256", _lower_hex("source.sha256", self.sha256))


@dataclass(frozen=True)
class Phase2Config:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    experiment_id: str
    global_seed: int
    legendre_degree: int
    pilot_per_extractor: int
    full_per_extractor: int
    extractor_ids: tuple[str, ...]
    minimum_systems: int
    minimum_method_families: int
    tau_b_threshold: float
    split_domain: str
    ledger_domain: str
    split_seed: int
    fit_threshold: float
    holdout_threshold: float
    split_strata: Mapping[str, Mapping[str, float]]
    expected_pair_counts: Mapping[str, int]
    expected_group_counts: Mapping[str, int]
    expected_stratum_pair_counts: Mapping[str, Mapping[str, int]]
    expected_ledger_sha256: str
    template_qualifiers: Mapping[str, str]
    sources: Mapping[str, SourceIdentity]
    document: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise Phase2ConfigError("path", "must be a pathlib.Path")
        if self.schema_version != CONFIG_SCHEMA_VERSION:
            raise Phase2ConfigError("schema_version", "does not match")
        if self.extractor_ids != _EXTRACTORS:
            raise Phase2ConfigError("extractor_ids", "must equal airpls/arpls/mor")
        if not 0.0 < self.fit_threshold < self.holdout_threshold < 1.0:
            raise Phase2ConfigError("split thresholds", "must satisfy 0 < fit < holdout < 1")
        if not 0.0 <= self.tau_b_threshold <= 1.0:
            raise Phase2ConfigError("tau_b_threshold", "must lie in [0, 1]")
        if tuple(sorted(self.split_strata)) != tuple(sorted(_STRATA)):
            raise Phase2ConfigError("split.strata", "does not match frozen names")
        frozen_strata: dict[str, Mapping[str, float]] = {}
        for name, raw_bounds in sorted(self.split_strata.items()):
            bounds = dict(raw_bounds)
            expected_upper = (
                "upper_inclusive" if name == "nir_785" else "upper_exclusive"
            )
            if set(bounds) != {"lower_inclusive", expected_upper}:
                raise Phase2ConfigError(
                    f"split.strata.{name}", "has an unexpected key set"
                )
            lower = _finite(
                f"split.strata.{name}.lower_inclusive",
                bounds["lower_inclusive"],
            )
            upper = _finite(
                f"split.strata.{name}.{expected_upper}", bounds[expected_upper]
            )
            if not lower < upper:
                raise Phase2ConfigError(
                    f"split.strata.{name}", "must have lower < upper"
                )
            frozen_strata[name] = MappingProxyType(
                {"lower_inclusive": lower, expected_upper: upper}
            )
        object.__setattr__(
            self, "split_strata", MappingProxyType(frozen_strata)
        )
        object.__setattr__(
            self,
            "expected_pair_counts",
            MappingProxyType(dict(sorted(self.expected_pair_counts.items()))),
        )
        object.__setattr__(
            self,
            "expected_group_counts",
            MappingProxyType(dict(sorted(self.expected_group_counts.items()))),
        )
        object.__setattr__(
            self,
            "expected_stratum_pair_counts",
            MappingProxyType(
                {
                    role: MappingProxyType(dict(sorted(counts.items())))
                    for role, counts in sorted(
                        self.expected_stratum_pair_counts.items()
                    )
                }
            ),
        )
        object.__setattr__(
            self,
            "template_qualifiers",
            MappingProxyType(dict(sorted(self.template_qualifiers.items()))),
        )
        object.__setattr__(
            self, "sources", MappingProxyType(dict(sorted(self.sources.items())))
        )
        frozen_document = _freeze_json("document", self.document)
        assert isinstance(frozen_document, Mapping)
        object.__setattr__(self, "document", frozen_document)


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


def _verify_bound_file(
    project_root: Path,
    *,
    label: str,
    relative_path: str,
    byte_count: int,
    sha256: str,
    verify_sha256: bool,
) -> None:
    bound_path = project_root / relative_path
    try:
        observed_size = bound_path.stat().st_size
    except OSError as error:
        raise Phase2ConfigError(label, str(error)) from error
    if observed_size != byte_count:
        raise Phase2ConfigError(label, "byte count mismatch")
    if not verify_sha256:
        return
    digest = hashlib.sha256()
    try:
        with bound_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise Phase2ConfigError(label, str(error)) from error
    if digest.hexdigest() != sha256:
        raise Phase2ConfigError(label, "SHA256 mismatch")


def load_phase2_config(
    path: Path, *, project_root: Path | None = None
) -> Phase2Config:
    path = Path(path)
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase2ConfigError("config", str(error)) from error
    observed_sha = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTE_COUNT or observed_sha != CONFIG_SHA256:
        raise Phase2ConfigError("config identity", "bytes or SHA256 mismatch")
    if not isinstance(document, Mapping) or raw != _canonical_json(document):
        raise Phase2ConfigError("config canonical", "must use canonical JSON bytes")
    root = _exact_keys(
        "config.keys",
        document,
        (
            "background_model",
            "dependencies",
            "experiment_id",
            "fidelity",
            "noise",
            "schedule",
            "schema_version",
            "sources",
            "split",
            "stability",
            "template_qualifiers",
        ),
    )
    schedule = _object("schedule", root["schedule"])
    background = _object("background_model", root["background_model"])
    stability = _object("stability", root["stability"])
    split = _object("split", root["split"])
    dependencies = _object("dependencies", root["dependencies"])
    sources_value = _object("sources", root["sources"])
    sources = {
        name: SourceIdentity(
            path=str(_object(f"sources.{name}", value)["path"]),
            byte_count=int(_object(f"sources.{name}", value)["byte_count"]),
            sha256=str(_object(f"sources.{name}", value)["sha256"]),
        )
        for name, value in sources_value.items()
    }
    extractors = tuple(stability["extractors"])
    template_qualifiers = {
        str(key): str(value)
        for key, value in _object(
            "template_qualifiers", root["template_qualifiers"]
        ).items()
    }
    if tuple(template_qualifiers.values()) != _QUALIFIERS:
        raise Phase2ConfigError("template_qualifiers", "do not match frozen values")
    root_path = _PROJECT_ROOT if project_root is None else Path(project_root)
    lock_path = _nonempty(
        "dependencies.lock_path",
        dependencies.get("lock_path"),
        Phase2ConfigError,
    )
    lock_sha256 = _lower_hex(
        "dependencies.lock_sha256", dependencies.get("lock_sha256")
    )
    _verify_bound_file(
        root_path,
        label="dependency lock",
        relative_path=lock_path,
        byte_count=126,
        sha256=lock_sha256,
        verify_sha256=True,
    )
    pybaselines = _object(
        "dependencies.pybaselines", dependencies.get("pybaselines")
    )
    required_pybaselines = _nonempty(
        "dependencies.pybaselines.version",
        pybaselines.get("version"),
        Phase2ConfigError,
    )
    try:
        installed_pybaselines = importlib.metadata.version("pybaselines")
    except importlib.metadata.PackageNotFoundError as error:
        raise Phase2ConfigError(
            "dependencies.pybaselines.version", "package is not installed"
        ) from error
    if installed_pybaselines != required_pybaselines:
        raise Phase2ConfigError(
            "dependencies.pybaselines.version",
            f"requires {required_pybaselines}, observed {installed_pybaselines}",
        )
    for name, identity in sorted(sources.items()):
        _verify_bound_file(
            root_path,
            label=f"sources.{name}",
            relative_path=identity.path,
            byte_count=identity.byte_count,
            sha256=identity.sha256,
            verify_sha256=identity.byte_count < _LARGE_SOURCE_MIN_BYTES,
        )
    return Phase2Config(
        path=path,
        byte_count=len(raw),
        sha256=observed_sha,
        schema_version=str(root["schema_version"]),
        experiment_id=str(root["experiment_id"]),
        global_seed=int(schedule["global_seed"]),
        legendre_degree=int(background["legendre_degree"]),
        pilot_per_extractor=int(schedule["pilot_per_extractor"]),
        full_per_extractor=int(schedule["full_per_extractor"]),
        extractor_ids=extractors,
        minimum_systems=int(stability["minimum_systems"]),
        minimum_method_families=int(stability["minimum_method_families"]),
        tau_b_threshold=float(stability["tau_b_threshold"]),
        split_domain=str(split["domain"]),
        ledger_domain=str(split["ledger_domain"]),
        split_seed=int(split["seed"]),
        fit_threshold=float(split["fit_threshold"]),
        holdout_threshold=float(split["holdout_threshold"]),
        split_strata={
            str(name): {str(key): float(value) for key, value in bounds.items()}
            for name, raw_bounds in _object(
                "split.strata", split["strata"]
            ).items()
            for bounds in (_object(f"split.strata.{name}", raw_bounds),)
        },
        expected_pair_counts={
            str(key): int(value)
            for key, value in _object(
                "split.expected_pair_counts", split["expected_pair_counts"]
            ).items()
        },
        expected_group_counts={
            str(key): int(value)
            for key, value in _object(
                "split.expected_group_counts", split["expected_group_counts"]
            ).items()
        },
        expected_stratum_pair_counts={
            str(role): {
                str(key): int(value)
                for key, value in _object(
                    f"split.stratum_pair_counts.{role}", counts
                ).items()
            }
            for role, counts in _object(
                "split.stratum_pair_counts", split["stratum_pair_counts"]
            ).items()
        },
        expected_ledger_sha256=str(split["expected_ledger_sha256"]),
        template_qualifiers=template_qualifiers,
        sources=sources,
        document=document,
    )


@dataclass(frozen=True)
class SemiSyntheticRecord:
    record_id: str
    extractor_id: str
    excitation_stratum: str
    template_id: str
    template_qualifier: str
    axis_cm1: np.ndarray
    y_observed: np.ndarray
    s_template: np.ndarray
    b_true: np.ndarray
    n_true: np.ndarray
    peaks_template: tuple[object, ...]
    provenance: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "extractor_id",
            "excitation_stratum",
            "template_id",
            "template_qualifier",
        ):
            object.__setattr__(
                self,
                name,
                _nonempty(name, getattr(self, name), SemiSyntheticContractError),
            )
        if self.extractor_id not in _EXTRACTORS:
            raise SemiSyntheticContractError("extractor_id", "is not supported")
        if self.excitation_stratum not in _STRATA:
            raise SemiSyntheticContractError("excitation_stratum", "is not supported")
        if self.template_qualifier not in _QUALIFIERS:
            raise SemiSyntheticContractError("template_qualifier", "is not supported")
        arrays = {
            name: _vector(name, getattr(self, name))
            for name in (
                "axis_cm1",
                "y_observed",
                "s_template",
                "b_true",
                "n_true",
            )
        }
        sizes = {array.size for array in arrays.values()}
        if len(sizes) != 1:
            raise SemiSyntheticContractError(
                "array shape", "all spectral arrays must have equal shape"
            )
        if not np.all(np.diff(arrays["axis_cm1"]) > 0.0):
            raise SemiSyntheticContractError("axis_cm1.increasing", "must be strict")
        if np.any(arrays["b_true"] < 0.0):
            raise SemiSyntheticContractError("b_true.nonnegative", "must be nonnegative")
        reconstructed = (
            arrays["s_template"] + arrays["b_true"] + arrays["n_true"]
        )
        if not np.array_equal(arrays["y_observed"], reconstructed):
            raise SemiSyntheticContractError(
                "decomposition", "y_observed must exactly equal s_template+b_true+n_true"
            )
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        if not isinstance(self.peaks_template, tuple):
            raise SemiSyntheticContractError("peaks_template", "must be a tuple")
        frozen = _freeze_json("provenance", self.provenance)
        if not isinstance(frozen, Mapping):
            raise SemiSyntheticContractError("provenance", "must be a mapping")
        object.__setattr__(self, "provenance", frozen)


__all__ = [
    "CONFIG_BYTE_COUNT",
    "CONFIG_SCHEMA_VERSION",
    "CONFIG_SHA256",
    "Phase2Config",
    "Phase2ConfigError",
    "SemiSyntheticContractError",
    "SemiSyntheticRecord",
    "SourceIdentity",
    "load_phase2_config",
]
