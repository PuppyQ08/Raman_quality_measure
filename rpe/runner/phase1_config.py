from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping, Sequence

from rpe.perturb.sweep import load_perturbation_sweep_config


REPO_ROOT = Path(__file__).resolve().parents[2]
_HEX_DIGITS = frozenset("0123456789abcdef")
_CONFIG_BYTE_COUNT = 2350
_CONFIG_SHA256 = "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd"
_SCIENTIFIC_BYTE_COUNT = 2122
_SCIENTIFIC_SHA256 = "f44a1b51a9ff0eb7f4221d45d452ce7354bae5734432e54a0fda87617babd37e"
_SOURCE_SNAPSHOT_SHA256 = (
    "8814d9a5e6b1f1c9b885a9fd84e1d8f3ceb3bfc9b69ebbcae62a0fd7aee774e7"
)
_CONFIG_ROOT_KEYS = (
    "core_gate",
    "deferred_perturbations",
    "distribution_status",
    "experiment_id",
    "materialized_perturbation_ids",
    "peak_not_applicable_reason_codes",
    "schema_version",
    "selection",
    "shared_sweep",
    "source",
    "storage",
)
_CORE_GATE_KEYS = (
    "allowed_failed_cell_count",
    "float_relative_tolerance",
    "p01_p04_min_class_fraction",
    "p01_p04_min_source_fraction",
    "p05_min_class_fraction",
    "p05_min_source_fraction",
    "p08_p12_required_source_fraction",
)
_DEFERRED_KEYS = ("dependency", "perturbation_ids", "reason_code")
_SELECTION_KEYS = ("algorithm", "subset_size")
_SHARED_SWEEP_KEYS = ("byte_count", "config_path", "sha256")
_SOURCE_KEYS = (
    "class_count",
    "dataset_id",
    "dataset_path",
    "files",
    "record_count",
    "related_provenance",
)
_STORAGE_KEYS = ("schema_version", "shard_source_count")
_FILE_KEYS = ("byte_count", "sha256")
_PROVENANCE_KEYS = ("artifact_path", "byte_count", "sha256")
_EXPECTED_SOURCE_FILES = (
    "SHA256SUMS",
    "SHA256SUMS.sha256",
    "arrays.h5",
    "dataset.json",
    "records.jsonl",
)
_EXPECTED_PROVENANCE = ("conversion_receipt", "pair_index")
_EXPECTED_MATERIALIZED = (
    "p01",
    "p02",
    "p03",
    "p04",
    "p05",
    "p08",
    "p09",
    "p10",
    "p11",
    "p12",
)
_EXPECTED_DEFERRED = ("p06", "p07")
_EXPECTED_PEAK_REASON_CODES = (
    "false_peak_placement_impossible",
    "insufficient_points_for_peak_model",
    "invalid_peak_component",
    "no_detected_peak",
    "nonpositive_intensity_range",
    "zero_false_peak_insertion_capacity",
)
_EXPECTED_SOURCE_FILE_IDENTITIES = {
    "SHA256SUMS": (235, "f0cb09af8d80cdf3d52af9ed01bc1ae48abba94fefdff712c30d6e8c8bfe4995"),
    "SHA256SUMS.sha256": (
        77,
        "72ec71a034c3ed081ddd27252d04910d51caceebcf84a4e9e1203f99ac50f314",
    ),
    "arrays.h5": (
        223602272,
        "768e4f3abd30db76b1a45e0a94448a63f180d2c3021dcfed56ce6ad861b32d68",
    ),
    "dataset.json": (
        1181717,
        "e4e5ca82f154eb2215bbda3e13dec2fe9aaa812e46ea7d36b7e4266754e84b10",
    ),
    "records.jsonl": (
        64389257,
        "4cc815e261abda595dd6facc3af52a5735d5f37c7dd346b683837f830c4b906e",
    ),
}
_EXPECTED_PROVENANCE_IDENTITIES = {
    "conversion_receipt": (
        "data/unified/rruff_raman_conversion.json",
        86193,
        "282701601d962e440561e452aa25adf89d626108d893974f24ca63fac2b3facc",
    ),
    "pair_index": (
        "data/unified/rruff_raman_pairs.jsonl",
        25513597,
        "efbb9549fb9bb0db721def69ea158c45e7fbd09e6515df51b1f109b0c912ec6d",
    ),
}

PHASE1_DATA_CODE_PATHS = (
    "rpe/evaluation/__init__.py",
    "rpe/evaluation/contracts.py",
    "rpe/io/__init__.py",
    "rpe/io/perturbed_schema.py",
    "rpe/io/perturbed_store.py",
    "rpe/io/schema.py",
    "rpe/io/store.py",
    "rpe/perturb/__init__.py",
    "rpe/perturb/axis_transform.py",
    "rpe/perturb/baseline_distortion.py",
    "rpe/perturb/baseline_residual.py",
    "rpe/perturb/contracts.py",
    "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py",
    "rpe/perturb/peak_family.py",
    "rpe/perturb/sweep.py",
    "rpe/runner/__init__.py",
    "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_formal.py",
    "rpe/runner/phase1_gates.py",
    "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py",
    "rpe/runner/phase1_types.py",
)
PHASE1_SUMMARY_CODE_PATHS: tuple[str, ...] = ()


class Phase1ConfigError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class ArtifactIdentity:
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "byte_count",
            _nonnegative_int("byte_count", self.byte_count),
        )
        object.__setattr__(
            self,
            "sha256",
            _lower_hex_64("sha256", self.sha256),
        )


@dataclass(frozen=True)
class LocatedArtifactIdentity:
    path: Path
    identity: ArtifactIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise Phase1ConfigError("path", "must be a pathlib.Path")
        if not isinstance(self.identity, ArtifactIdentity):
            raise Phase1ConfigError("identity", "must be ArtifactIdentity")


@dataclass(frozen=True)
class Phase1CoreConfig:
    path: Path
    file_byte_count: int
    file_sha256: str
    scientific_config_byte_count: int
    scientific_config_sha256: str
    schema_version: str
    experiment_id: str
    source_dataset_id: str
    source_dataset_path: Path
    source_record_count: int
    source_class_count: int
    source_files: Mapping[str, ArtifactIdentity]
    related_provenance: Mapping[str, LocatedArtifactIdentity]
    shared_sweep_path: Path
    shared_sweep_identity: ArtifactIdentity
    subset_size: int
    selection_algorithm: str
    shard_source_count: int
    materialized_perturbation_ids: tuple[str, ...]
    deferred_perturbation_ids: tuple[str, ...]
    deferred_reason_code: str
    deferred_dependency: str
    peak_not_applicable_reason_codes: tuple[str, ...]
    distribution_status: str
    core_gate: Mapping[str, int | float]

    def __post_init__(self) -> None:
        self._set_path("path", self.path)
        self._set_nonnegative_int("file_byte_count", self.file_byte_count)
        self._set_lower_hex("file_sha256", self.file_sha256)
        self._set_nonnegative_int(
            "scientific_config_byte_count",
            self.scientific_config_byte_count,
        )
        self._set_lower_hex(
            "scientific_config_sha256",
            self.scientific_config_sha256,
        )
        self._set_nonempty_string("schema_version", self.schema_version)
        self._set_nonempty_string("experiment_id", self.experiment_id)
        self._set_nonempty_string("source_dataset_id", self.source_dataset_id)
        self._set_path("source_dataset_path", self.source_dataset_path)
        self._set_nonnegative_int("source_record_count", self.source_record_count)
        self._set_nonnegative_int("source_class_count", self.source_class_count)
        self._set_mapping(
            "source_files",
            self.source_files,
            ArtifactIdentity,
        )
        self._set_mapping(
            "related_provenance",
            self.related_provenance,
            LocatedArtifactIdentity,
        )
        self._set_path("shared_sweep_path", self.shared_sweep_path)
        if not isinstance(self.shared_sweep_identity, ArtifactIdentity):
            raise Phase1ConfigError(
                "shared_sweep_identity",
                "must be ArtifactIdentity",
            )
        self._set_nonnegative_int("subset_size", self.subset_size)
        self._set_nonempty_string(
            "selection_algorithm",
            self.selection_algorithm,
        )
        self._set_nonnegative_int(
            "shard_source_count",
            self.shard_source_count,
        )
        self._set_sorted_unique_strings(
            "materialized_perturbation_ids",
            self.materialized_perturbation_ids,
        )
        self._set_sorted_unique_strings(
            "deferred_perturbation_ids",
            self.deferred_perturbation_ids,
        )
        self._set_nonempty_string(
            "deferred_reason_code",
            self.deferred_reason_code,
        )
        self._set_nonempty_string(
            "deferred_dependency",
            self.deferred_dependency,
        )
        self._set_sorted_unique_strings(
            "peak_not_applicable_reason_codes",
            self.peak_not_applicable_reason_codes,
        )
        self._set_nonempty_string(
            "distribution_status",
            self.distribution_status,
        )
        self._set_numeric_mapping("core_gate", self.core_gate)

    def _set_path(self, name: str, value: object) -> None:
        if not isinstance(value, Path):
            raise Phase1ConfigError(name, "must be a pathlib.Path")

    def _set_nonnegative_int(self, name: str, value: object) -> None:
        object.__setattr__(self, name, _nonnegative_int(name, value))

    def _set_lower_hex(self, name: str, value: object) -> None:
        object.__setattr__(self, name, _lower_hex_64(name, value))

    def _set_nonempty_string(self, name: str, value: object) -> None:
        object.__setattr__(self, name, _nonempty_string(name, value))

    def _set_mapping(
        self,
        name: str,
        value: object,
        expected_type: type[object],
    ) -> None:
        if not isinstance(value, Mapping):
            raise Phase1ConfigError(name, "must be a mapping")
        copied: dict[str, object] = {}
        for key, item in value.items():
            parsed_key = _nonempty_string(f"{name} key", key)
            item_name = f"{name}.{parsed_key}"
            if not isinstance(item, expected_type):
                raise Phase1ConfigError(
                    item_name,
                    f"must be {expected_type.__name__}",
                )
            copied[parsed_key] = item
        object.__setattr__(
            self,
            name,
            MappingProxyType(dict(sorted(copied.items()))),
        )

    def _set_sorted_unique_strings(self, name: str, value: object) -> None:
        if not isinstance(value, tuple):
            raise Phase1ConfigError(name, "must be a tuple")
        converted = tuple(
            _nonempty_string(f"{name}[{index}]", item)
            for index, item in enumerate(value)
        )
        if len(set(converted)) != len(converted):
            raise Phase1ConfigError(name, "must be unique")
        object.__setattr__(self, name, tuple(sorted(converted)))

    def _set_numeric_mapping(self, name: str, value: object) -> None:
        if not isinstance(value, Mapping):
            raise Phase1ConfigError(name, "must be a mapping")
        copied: dict[str, int | float] = {}
        for key, item in value.items():
            parsed_key = _nonempty_string(f"{name} key", key)
            item_path = f"{name}.{parsed_key}"
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise Phase1ConfigError(item_path, "must be an int or float")
            if not math.isfinite(float(item)):
                raise Phase1ConfigError(item_path, "must be finite")
            copied[parsed_key] = item
        object.__setattr__(
            self,
            name,
            MappingProxyType(dict(sorted(copied.items()))),
        )


def _reject_nonfinite(token: str) -> object:
    raise Phase1ConfigError(
        "config nonfinite",
        f"JSON constant {token!r} is forbidden",
    )


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return json.dumps(
        _json_ready(document),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonicalize(raw: bytes) -> tuple[str, dict[str, object]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Phase1ConfigError("config", "must be UTF-8 text") from exc
    try:
        payload = json.loads(text, parse_constant=_reject_nonfinite)
    except Phase1ConfigError:
        raise
    except json.JSONDecodeError as exc:
        raise Phase1ConfigError("config", "must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise Phase1ConfigError("config", "must be a JSON object")
    canonical = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )
    if text != canonical:
        raise Phase1ConfigError(
            "config",
            "must match the frozen canonical JSON encoding",
        )
    return text, payload


def _require_exact_object_keys(
    path: str,
    value: object,
    expected_keys: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise Phase1ConfigError(path, "must be an object")
    if set(value.keys()) != set(expected_keys):
        raise Phase1ConfigError(path, "must contain the exact frozen key set")
    return value


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise Phase1ConfigError(path, "must be a nonempty string")
    return value


def _relative_string_path(path: str, value: object) -> str:
    text = _nonempty_string(path, value)
    pure = PurePosixPath(text)
    if pure.is_absolute():
        raise Phase1ConfigError(path, "must be a relative path")
    if any(part in ("", ".", "..") for part in pure.parts):
        raise Phase1ConfigError(path, "must be a normalized relative path")
    return text


def _lower_hex_64(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise Phase1ConfigError(
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _nonnegative_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase1ConfigError(path, "must be a nonnegative integer")
    return value


def _finite_real(path: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise Phase1ConfigError(path, "must be a finite real number")
    return float(value)


def _fraction(path: str, value: object) -> float:
    fraction = _finite_real(path, value)
    if fraction < 0.0 or fraction > 1.0:
        raise Phase1ConfigError(path, "must be between 0.0 and 1.0 inclusive")
    return fraction


def _string_tuple(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise Phase1ConfigError(path, "must be a list")
    converted = tuple(
        _nonempty_string(f"{path}[{index}]", item)
        for index, item in enumerate(value)
    )
    if len(set(converted)) != len(converted):
        raise Phase1ConfigError(path, "must be unique")
    return converted


def _mapping_proxy(values: dict[str, object]) -> Mapping[str, object]:
    return MappingProxyType(values)


def _validate_frozen_value(path: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise Phase1ConfigError(path, f"must match frozen value {expected!r}")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key == "":
                raise Phase1ConfigError("document", "must use nonempty string keys")
            normalized[key] = _json_ready(item)
        return normalized
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Phase1ConfigError("document", "contains nonfinite numeric value")
        return value
    raise Phase1ConfigError(
        "document",
        f"unsupported value type {type(value).__name__}",
    )


def _scientific_projection(value: object, path: str) -> object:
    if isinstance(value, Mapping):
        projected: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or key == "":
                raise Phase1ConfigError(path, "must use nonempty string keys")
            if key.endswith("_path"):
                continue
            child_path = f"{path}.{key}"
            projected[key] = _scientific_projection(item, child_path)
        return projected
    if isinstance(value, (tuple, list)):
        return [
            _scientific_projection(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Phase1ConfigError(path, "must be finite")
        return value
    raise Phase1ConfigError(
        path,
        f"unsupported value type {type(value).__name__}",
    )


def scientific_config_bytes(document: Mapping[str, object]) -> bytes:
    if not isinstance(document, Mapping):
        raise Phase1ConfigError("document", "must be a mapping")
    projected = _scientific_projection(document, "document")
    if not isinstance(projected, dict):
        raise Phase1ConfigError("document", "scientific projection must be an object")
    return _canonical_json_bytes(projected) + b"\n"


def source_snapshot_bytes(config: Phase1CoreConfig) -> bytes:
    if not isinstance(config, Phase1CoreConfig):
        raise Phase1ConfigError("config", "must be Phase1CoreConfig")
    document: dict[str, object] = {
        "class_count": config.source_class_count,
        "dataset_id": config.source_dataset_id,
        "files": {
            name: {
                "byte_count": identity.byte_count,
                "sha256": identity.sha256,
            }
            for name, identity in config.source_files.items()
        },
        "record_count": config.source_record_count,
        "related_provenance": {
            name: {
                "byte_count": located.identity.byte_count,
                "sha256": located.identity.sha256,
            }
            for name, located in config.related_provenance.items()
        },
    }
    return _canonical_json_bytes(document) + b"\n"


def source_snapshot_digest(config: Phase1CoreConfig) -> str:
    if not isinstance(config, Phase1CoreConfig):
        raise Phase1ConfigError("config", "must be Phase1CoreConfig")
    payload = b"rpe-phase1-source-snapshot-v1\0" + source_snapshot_bytes(config)
    return hashlib.sha256(payload).hexdigest()


def _normalized_relative_path(index: int, relative_path: object) -> str:
    if not isinstance(relative_path, str) or relative_path == "":
        raise Phase1ConfigError(
            f"relative_paths[{index}]",
            "must be a nonempty string",
        )
    pure = PurePosixPath(relative_path)
    if pure.is_absolute():
        raise Phase1ConfigError(
            f"relative_paths[{index}]",
            "must be relative",
        )
    if any(part in ("", ".", "..") for part in pure.parts):
        raise Phase1ConfigError(
            f"relative_paths[{index}]",
            "must be normalized and stay within root",
        )
    return pure.as_posix()


def code_snapshot_document(
    root: Path,
    relative_paths: Sequence[str],
) -> Mapping[str, Mapping[str, int | str]]:
    if not isinstance(root, Path):
        raise Phase1ConfigError("root", "must be a pathlib.Path")
    if not isinstance(relative_paths, Sequence):
        raise Phase1ConfigError("relative_paths", "must be a sequence")
    normalized = tuple(
        _normalized_relative_path(index, relative_path)
        for index, relative_path in enumerate(relative_paths)
    )
    if len(set(normalized)) != len(normalized):
        raise Phase1ConfigError("relative_paths", "must be unique")
    if normalized != tuple(sorted(normalized)):
        raise Phase1ConfigError("relative_paths", "must be lexicographically sorted")
    resolved_root = root.resolve()
    snapshot: dict[str, Mapping[str, int | str]] = {}
    for index, relative_path in enumerate(normalized):
        candidate = resolved_root / relative_path
        resolved = candidate.resolve()
        if resolved_root not in resolved.parents and resolved != resolved_root:
            raise Phase1ConfigError(
                f"relative_paths[{index}]",
                "must resolve within root",
            )
        if not resolved.exists():
            raise Phase1ConfigError(
                f"relative_paths[{index}]",
                "must reference an existing file",
            )
        if not resolved.is_file():
            raise Phase1ConfigError(
                f"relative_paths[{index}]",
                "must reference a file",
            )
        raw = resolved.read_bytes()
        snapshot[relative_path] = _mapping_proxy(
            {
                "byte_count": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return _mapping_proxy(snapshot)


def code_snapshot_digest(root: Path, relative_paths: Sequence[str]) -> str:
    document = code_snapshot_document(root, relative_paths)
    payload = b"rpe-phase1-code-snapshot-v1\0" + _canonical_json_bytes(document)
    return hashlib.sha256(payload).hexdigest()


def load_phase1_core_config(path: Path) -> Phase1CoreConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    _, payload = _canonicalize(raw)
    root = _require_exact_object_keys("config.keys", payload, _CONFIG_ROOT_KEYS)
    core_gate = _require_exact_object_keys("core_gate", root["core_gate"], _CORE_GATE_KEYS)
    deferred = _require_exact_object_keys(
        "deferred_perturbations",
        root["deferred_perturbations"],
        _DEFERRED_KEYS,
    )
    selection = _require_exact_object_keys("selection", root["selection"], _SELECTION_KEYS)
    shared_sweep = _require_exact_object_keys(
        "shared_sweep",
        root["shared_sweep"],
        _SHARED_SWEEP_KEYS,
    )
    source = _require_exact_object_keys("source", root["source"], _SOURCE_KEYS)
    storage = _require_exact_object_keys("storage", root["storage"], _STORAGE_KEYS)
    files = _require_exact_object_keys("source.files", source["files"], _EXPECTED_SOURCE_FILES)
    related_provenance = _require_exact_object_keys(
        "source.related_provenance",
        source["related_provenance"],
        _EXPECTED_PROVENANCE,
    )

    materialized = _string_tuple(
        "materialized_perturbation_ids",
        root["materialized_perturbation_ids"],
    )
    deferred_ids = _string_tuple(
        "deferred_perturbations.perturbation_ids",
        deferred["perturbation_ids"],
    )
    peak_reason_codes = _string_tuple(
        "peak_not_applicable_reason_codes",
        root["peak_not_applicable_reason_codes"],
    )
    _validate_frozen_value("materialized_perturbation_ids", materialized, _EXPECTED_MATERIALIZED)
    _validate_frozen_value(
        "deferred_perturbations.perturbation_ids",
        deferred_ids,
        _EXPECTED_DEFERRED,
    )
    _validate_frozen_value(
        "peak_not_applicable_reason_codes",
        peak_reason_codes,
        _EXPECTED_PEAK_REASON_CODES,
    )

    gate_values = {
        "allowed_failed_cell_count": _nonnegative_int(
            "core_gate.allowed_failed_cell_count",
            core_gate["allowed_failed_cell_count"],
        ),
        "float_relative_tolerance": _finite_real(
            "core_gate.float_relative_tolerance",
            core_gate["float_relative_tolerance"],
        ),
        "p01_p04_min_class_fraction": _fraction(
            "core_gate.p01_p04_min_class_fraction",
            core_gate["p01_p04_min_class_fraction"],
        ),
        "p01_p04_min_source_fraction": _fraction(
            "core_gate.p01_p04_min_source_fraction",
            core_gate["p01_p04_min_source_fraction"],
        ),
        "p05_min_class_fraction": _fraction(
            "core_gate.p05_min_class_fraction",
            core_gate["p05_min_class_fraction"],
        ),
        "p05_min_source_fraction": _fraction(
            "core_gate.p05_min_source_fraction",
            core_gate["p05_min_source_fraction"],
        ),
        "p08_p12_required_source_fraction": _fraction(
            "core_gate.p08_p12_required_source_fraction",
            core_gate["p08_p12_required_source_fraction"],
        ),
    }
    _validate_frozen_value("core_gate.allowed_failed_cell_count", gate_values["allowed_failed_cell_count"], 0)
    _validate_frozen_value("core_gate.float_relative_tolerance", gate_values["float_relative_tolerance"], 1e-12)
    _validate_frozen_value("core_gate.p01_p04_min_class_fraction", gate_values["p01_p04_min_class_fraction"], 0.95)
    _validate_frozen_value("core_gate.p01_p04_min_source_fraction", gate_values["p01_p04_min_source_fraction"], 0.95)
    _validate_frozen_value("core_gate.p05_min_class_fraction", gate_values["p05_min_class_fraction"], 0.9)
    _validate_frozen_value("core_gate.p05_min_source_fraction", gate_values["p05_min_source_fraction"], 0.9)
    _validate_frozen_value(
        "core_gate.p08_p12_required_source_fraction",
        gate_values["p08_p12_required_source_fraction"],
        1.0,
    )

    source_files: dict[str, ArtifactIdentity] = {}
    for name in _EXPECTED_SOURCE_FILES:
        file_document = _require_exact_object_keys(
            f"source.files.{name}",
            files[name],
            _FILE_KEYS,
        )
        identity = ArtifactIdentity(
            byte_count=_nonnegative_int(
                f"source.files.{name}.byte_count",
                file_document["byte_count"],
            ),
            sha256=_lower_hex_64(
                f"source.files.{name}.sha256",
                file_document["sha256"],
            ),
        )
        expected_byte_count, expected_sha256 = _EXPECTED_SOURCE_FILE_IDENTITIES[name]
        _validate_frozen_value(
            f"source.files.{name}.byte_count",
            identity.byte_count,
            expected_byte_count,
        )
        _validate_frozen_value(
            f"source.files.{name}.sha256",
            identity.sha256,
            expected_sha256,
        )
        source_files[name] = identity

    provenance_entries: dict[str, LocatedArtifactIdentity] = {}
    for name in _EXPECTED_PROVENANCE:
        document = _require_exact_object_keys(
            f"source.related_provenance.{name}",
            related_provenance[name],
            _PROVENANCE_KEYS,
        )
        expected_path, expected_byte_count, expected_sha256 = _EXPECTED_PROVENANCE_IDENTITIES[name]
        artifact_path = Path(
            _relative_string_path(
                f"source.related_provenance.{name}.artifact_path",
                document["artifact_path"],
            )
        )
        _validate_frozen_value(
            f"source.related_provenance.{name}.artifact_path",
            artifact_path.as_posix(),
            expected_path,
        )
        identity = ArtifactIdentity(
            byte_count=_nonnegative_int(
                f"source.related_provenance.{name}.byte_count",
                document["byte_count"],
            ),
            sha256=_lower_hex_64(
                f"source.related_provenance.{name}.sha256",
                document["sha256"],
            ),
        )
        _validate_frozen_value(
            f"source.related_provenance.{name}.byte_count",
            identity.byte_count,
            expected_byte_count,
        )
        _validate_frozen_value(
            f"source.related_provenance.{name}.sha256",
            identity.sha256,
            expected_sha256,
        )
        provenance_entries[name] = LocatedArtifactIdentity(path=artifact_path, identity=identity)

    schema_version = _nonempty_string("schema_version", root["schema_version"])
    experiment_id = _nonempty_string("experiment_id", root["experiment_id"])
    distribution_status = _nonempty_string(
        "distribution_status",
        root["distribution_status"],
    )
    source_dataset_id = _nonempty_string("source.dataset_id", source["dataset_id"])
    source_dataset_path = Path(
        _relative_string_path("source.dataset_path", source["dataset_path"])
    )
    source_record_count = _nonnegative_int("source.record_count", source["record_count"])
    source_class_count = _nonnegative_int("source.class_count", source["class_count"])
    selection_algorithm = _nonempty_string(
        "selection.algorithm",
        selection["algorithm"],
    )
    subset_size = _nonnegative_int("selection.subset_size", selection["subset_size"])
    storage_schema_version = _nonempty_string(
        "storage.schema_version",
        storage["schema_version"],
    )
    shard_source_count = _nonnegative_int(
        "storage.shard_source_count",
        storage["shard_source_count"],
    )
    deferred_reason_code = _nonempty_string(
        "deferred_perturbations.reason_code",
        deferred["reason_code"],
    )
    deferred_dependency = _nonempty_string(
        "deferred_perturbations.dependency",
        deferred["dependency"],
    )
    shared_sweep_byte_count = _nonnegative_int(
        "shared_sweep.byte_count",
        shared_sweep["byte_count"],
    )
    shared_sweep_sha256 = _lower_hex_64(
        "shared_sweep.sha256",
        shared_sweep["sha256"],
    )
    shared_sweep_path = Path(
        _relative_string_path("shared_sweep.config_path", shared_sweep["config_path"])
    )

    _validate_frozen_value("schema_version", schema_version, "phase1-rruff-core10k-config-v1")
    _validate_frozen_value("experiment_id", experiment_id, "phase1_rruff_raw_core10k")
    _validate_frozen_value(
        "distribution_status",
        distribution_status,
        "local_rebuild_only_pending_source_license_clearance",
    )
    _validate_frozen_value("source.dataset_id", source_dataset_id, "rruff_raman_raw")
    _validate_frozen_value("source.dataset_path", source_dataset_path.as_posix(), "data/unified/rruff_raman_raw")
    _validate_frozen_value("source.record_count", source_record_count, 20664)
    _validate_frozen_value("source.class_count", source_class_count, 2480)
    _validate_frozen_value(
        "selection.algorithm",
        selection_algorithm,
        "rruff_class_min1_hamilton_capacity_sample_round_robin_sha256_v1",
    )
    _validate_frozen_value("selection.subset_size", subset_size, 10000)
    _validate_frozen_value(
        "storage.schema_version",
        storage_schema_version,
        "phase1-perturbed-store-v1",
    )
    _validate_frozen_value("storage.shard_source_count", shard_source_count, 100)
    _validate_frozen_value(
        "deferred_perturbations.reason_code",
        deferred_reason_code,
        "missing_explicit_baseline",
    )
    _validate_frozen_value(
        "deferred_perturbations.dependency",
        deferred_dependency,
        "phase2_semisynthetic_or_separately_approved_physical_baseline",
    )
    _validate_frozen_value(
        "shared_sweep.byte_count",
        shared_sweep_byte_count,
        559,
    )
    _validate_frozen_value(
        "shared_sweep.sha256",
        shared_sweep_sha256,
        "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186",
    )
    _validate_frozen_value(
        "shared_sweep.config_path",
        shared_sweep_path.as_posix(),
        "experiments/shared/raman_perturbation_sweep_v1.json",
    )

    loaded_shared_sweep = load_perturbation_sweep_config(REPO_ROOT / shared_sweep_path)
    _validate_frozen_value(
        "shared_sweep.byte_count",
        loaded_shared_sweep.byte_count,
        shared_sweep_byte_count,
    )
    _validate_frozen_value(
        "shared_sweep.sha256",
        loaded_shared_sweep.sha256,
        shared_sweep_sha256,
    )

    scientific_bytes = scientific_config_bytes(root)
    scientific_byte_count = len(scientific_bytes)
    scientific_sha256 = hashlib.sha256(scientific_bytes).hexdigest()
    _validate_frozen_value(
        "scientific_config.byte_count",
        scientific_byte_count,
        _SCIENTIFIC_BYTE_COUNT,
    )
    _validate_frozen_value(
        "scientific_config.sha256",
        scientific_sha256,
        _SCIENTIFIC_SHA256,
    )

    config = Phase1CoreConfig(
        path=config_path,
        file_byte_count=len(raw),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        scientific_config_byte_count=scientific_byte_count,
        scientific_config_sha256=scientific_sha256,
        schema_version=schema_version,
        experiment_id=experiment_id,
        source_dataset_id=source_dataset_id,
        source_dataset_path=source_dataset_path,
        source_record_count=source_record_count,
        source_class_count=source_class_count,
        source_files=_mapping_proxy(source_files),
        related_provenance=_mapping_proxy(provenance_entries),
        shared_sweep_path=shared_sweep_path,
        shared_sweep_identity=ArtifactIdentity(
            byte_count=shared_sweep_byte_count,
            sha256=shared_sweep_sha256,
        ),
        subset_size=subset_size,
        selection_algorithm=selection_algorithm,
        shard_source_count=shard_source_count,
        materialized_perturbation_ids=materialized,
        deferred_perturbation_ids=deferred_ids,
        deferred_reason_code=deferred_reason_code,
        deferred_dependency=deferred_dependency,
        peak_not_applicable_reason_codes=peak_reason_codes,
        distribution_status=distribution_status,
        core_gate=_mapping_proxy(gate_values),
    )
    _validate_frozen_value(
        "source_snapshot.sha256",
        source_snapshot_digest(config),
        _SOURCE_SNAPSHOT_SHA256,
    )
    _validate_frozen_value("config identity.byte_count", config.file_byte_count, _CONFIG_BYTE_COUNT)
    _validate_frozen_value("config identity.sha256", config.file_sha256, _CONFIG_SHA256)
    return config
