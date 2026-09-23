from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path


SCHEMA_VERSION = "phase1-phase4-perturbation-sweep-v1"
SWEEP_ID = "raman_perturbation_alpha_v1"
CONFIG_BYTES = 559
CONFIG_SHA256 = "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
PERTURBATION_IDS = tuple(f"p{number:02d}" for number in range(1, 13))
GLOBAL_SEED = 20260817
BIT_GENERATOR = "PCG64"
SEED_DERIVATION_DIGEST = "sha256"
SEED_DERIVATION_DOMAIN = "rpe-perturbation-state-v1"
SEED_DERIVATION_ENCODING = (
    "uint64_le_seed_then_uint64_le_length_prefixed_utf8_fields"
)
SEED_DERIVATION_FIELDS = ("perturbation_id", "spectrum_id")
SEED_DERIVATION_SEED_MATERIAL = (
    "first_128_digest_bits_as_four_uint32_le"
)
_HEX_DIGITS = frozenset("0123456789abcdef")
_EXPECTED_ROOT_KEYS = (
    "alpha_grid",
    "bit_generator",
    "global_seed",
    "identity_alpha",
    "perturbation_ids",
    "schema_version",
    "seed_derivation",
    "sweep_id",
)
_EXPECTED_SEED_KEYS = (
    "digest",
    "domain",
    "encoding",
    "fields",
    "seed_material",
)


class PerturbationSweepConfigError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise PerturbationSweepConfigError(
            path,
            "must be a nonempty string",
        )
    return value


def _lower_hex_64(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise PerturbationSweepConfigError(
            path,
            "must be a lowercase 64-character hexadecimal string",
        )
    return value


def _finite_real(path: str, value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise PerturbationSweepConfigError(
            path,
            "must be a finite real number",
        )
    return float(value)


def _nonnegative_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PerturbationSweepConfigError(
            path,
            "must be a nonnegative integer",
        )
    return value


def _string_tuple(
    path: str,
    value: object,
    *,
    nonempty: bool,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise PerturbationSweepConfigError(path, "must be a tuple")
    if nonempty and not value:
        raise PerturbationSweepConfigError(path, "must be nonempty")
    converted = tuple(
        _nonempty_string(f"{path}[{index}]", item)
        for index, item in enumerate(value)
    )
    if len(set(converted)) != len(converted):
        raise PerturbationSweepConfigError(path, "must be unique")
    return converted


def _alpha_tuple(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        raise PerturbationSweepConfigError(
            path,
            "must be a nonempty tuple",
        )
    return tuple(
        _finite_real(f"{path}[{index}]", item)
        for index, item in enumerate(value)
    )


def _reject_nonfinite(token: str) -> object:
    raise PerturbationSweepConfigError(
        "config",
        f"non-finite JSON constant {token!r} is forbidden",
    )


def _length_prefixed_utf8(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


@dataclass(frozen=True)
class SeedDerivationConfig:
    digest: str
    domain: str
    encoding: str
    fields: tuple[str, ...]
    seed_material: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "digest",
            _nonempty_string("seed_derivation.digest", self.digest),
        )
        object.__setattr__(
            self,
            "domain",
            _nonempty_string("seed_derivation.domain", self.domain),
        )
        object.__setattr__(
            self,
            "encoding",
            _nonempty_string("seed_derivation.encoding", self.encoding),
        )
        object.__setattr__(
            self,
            "fields",
            _string_tuple(
                "seed_derivation.fields",
                self.fields,
                nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "seed_material",
            _nonempty_string(
                "seed_derivation.seed_material",
                self.seed_material,
            ),
        )


@dataclass(frozen=True)
class PerturbationSweepConfig:
    path: Path
    sha256: str
    byte_count: int
    schema_version: str
    sweep_id: str
    alpha_grid: tuple[float, ...]
    identity_alpha: float
    perturbation_ids: tuple[str, ...]
    global_seed: int
    bit_generator: str
    seed_derivation: SeedDerivationConfig

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise PerturbationSweepConfigError(
                "path",
                "must be a pathlib.Path",
            )
        object.__setattr__(
            self,
            "sha256",
            _lower_hex_64("sha256", self.sha256),
        )
        object.__setattr__(
            self,
            "byte_count",
            _nonnegative_int("byte_count", self.byte_count),
        )
        object.__setattr__(
            self,
            "schema_version",
            _nonempty_string("schema_version", self.schema_version),
        )
        object.__setattr__(
            self,
            "sweep_id",
            _nonempty_string("sweep_id", self.sweep_id),
        )
        alpha_grid = _alpha_tuple("alpha_grid", self.alpha_grid)
        object.__setattr__(self, "alpha_grid", alpha_grid)
        identity_alpha = _finite_real(
            "identity_alpha",
            self.identity_alpha,
        )
        if identity_alpha not in alpha_grid:
            raise PerturbationSweepConfigError(
                "identity_alpha",
                "must appear in alpha_grid",
            )
        object.__setattr__(self, "identity_alpha", identity_alpha)
        object.__setattr__(
            self,
            "perturbation_ids",
            _string_tuple(
                "perturbation_ids",
                self.perturbation_ids,
                nonempty=True,
            ),
        )
        object.__setattr__(
            self,
            "global_seed",
            _nonnegative_int("global_seed", self.global_seed),
        )
        object.__setattr__(
            self,
            "bit_generator",
            _nonempty_string("bit_generator", self.bit_generator),
        )
        if not isinstance(self.seed_derivation, SeedDerivationConfig):
            raise PerturbationSweepConfigError(
                "seed_derivation",
                "must be SeedDerivationConfig",
            )


def _float64_bytes(value: float) -> bytes:
    return struct.pack("<d", value)


def _same_float64(value: float, expected: float) -> bool:
    return _float64_bytes(value) == _float64_bytes(expected)


def _same_float64_tuple(
    values: tuple[float, ...],
    expected: tuple[float, ...],
) -> bool:
    return len(values) == len(expected) and all(
        _same_float64(value, expected_value)
        for value, expected_value in zip(values, expected, strict=True)
    )


def validate_perturbation_sweep_config(
    config: PerturbationSweepConfig,
) -> None:
    """Reject any config that is not the one frozen Phase 1/4 authority."""
    if not isinstance(config, PerturbationSweepConfig):
        raise PerturbationSweepConfigError(
            "config",
            "must be PerturbationSweepConfig",
        )
    if not isinstance(config.path, Path):
        raise PerturbationSweepConfigError(
            "path",
            "must be a pathlib.Path",
        )
    if config.sha256 != CONFIG_SHA256:
        raise PerturbationSweepConfigError(
            "sha256",
            "must match the frozen shared config SHA256",
        )
    if config.byte_count != CONFIG_BYTES:
        raise PerturbationSweepConfigError(
            "byte_count",
            "must match the frozen shared config byte count",
        )
    if config.schema_version != SCHEMA_VERSION:
        raise PerturbationSweepConfigError(
            "schema_version",
            "must match the frozen schema version",
        )
    if config.sweep_id != SWEEP_ID:
        raise PerturbationSweepConfigError(
            "sweep_id",
            "must match the frozen sweep_id",
        )
    if not _same_float64_tuple(config.alpha_grid, ALPHA_GRID):
        raise PerturbationSweepConfigError(
            "alpha_grid",
            "must match the frozen alpha grid",
        )
    if not _same_float64(config.identity_alpha, ALPHA_GRID[0]):
        raise PerturbationSweepConfigError(
            "identity_alpha",
            "must match the frozen identity alpha",
        )
    if config.perturbation_ids != PERTURBATION_IDS:
        raise PerturbationSweepConfigError(
            "perturbation_ids",
            "must match the frozen perturbation inventory",
        )
    if config.global_seed != GLOBAL_SEED:
        raise PerturbationSweepConfigError(
            "global_seed",
            "must match the frozen global seed",
        )
    if config.bit_generator != BIT_GENERATOR:
        raise PerturbationSweepConfigError(
            "bit_generator",
            "must match the frozen bit generator",
        )
    if config.seed_derivation.digest != SEED_DERIVATION_DIGEST:
        raise PerturbationSweepConfigError(
            "seed_derivation.digest",
            "must match the frozen digest",
        )
    if config.seed_derivation.domain != SEED_DERIVATION_DOMAIN:
        raise PerturbationSweepConfigError(
            "seed_derivation.domain",
            "must match the frozen domain",
        )
    if config.seed_derivation.encoding != SEED_DERIVATION_ENCODING:
        raise PerturbationSweepConfigError(
            "seed_derivation.encoding",
            "must match the frozen encoding",
        )
    if config.seed_derivation.fields != SEED_DERIVATION_FIELDS:
        raise PerturbationSweepConfigError(
            "seed_derivation.fields",
            "must match the frozen field order",
        )
    if config.seed_derivation.seed_material != SEED_DERIVATION_SEED_MATERIAL:
        raise PerturbationSweepConfigError(
            "seed_derivation.seed_material",
            "must match the frozen seed-material rule",
        )


def is_frozen_alpha(config: PerturbationSweepConfig, alpha: object) -> bool:
    """Return whether alpha is byte-identical to one frozen grid value."""
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(float(alpha))
    ):
        return False
    alpha_value = float(alpha)
    return any(
        _same_float64(alpha_value, grid_value)
        for grid_value in config.alpha_grid
    )


def _require_exact_object_keys(
    path: str,
    value: object,
    expected_keys: tuple[str, ...],
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise PerturbationSweepConfigError(path, "must be an object")
    keys = tuple(value.keys())
    if set(keys) != set(expected_keys):
        raise PerturbationSweepConfigError(
            path,
            "must contain the exact frozen key set",
        )
    return value


def _canonicalize(raw: bytes) -> tuple[str, dict[str, object]]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PerturbationSweepConfigError(
            "config",
            "must be UTF-8 text",
        ) from exc
    try:
        payload = json.loads(
            text,
            parse_constant=_reject_nonfinite,
        )
    except PerturbationSweepConfigError:
        raise
    except json.JSONDecodeError as exc:
        raise PerturbationSweepConfigError(
            "config",
            "must be valid JSON",
        ) from exc
    if not isinstance(payload, dict):
        raise PerturbationSweepConfigError(
            "config",
            "must be a JSON object",
        )
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
        raise PerturbationSweepConfigError(
            "config",
            "must match the frozen canonical JSON encoding",
        )
    return text, payload


def load_perturbation_sweep_config(
    path: Path,
) -> PerturbationSweepConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    _, payload = _canonicalize(raw)
    root = _require_exact_object_keys(
        "config",
        payload,
        _EXPECTED_ROOT_KEYS,
    )
    seed_derivation = _require_exact_object_keys(
        "seed_derivation",
        root["seed_derivation"],
        _EXPECTED_SEED_KEYS,
    )
    alpha_grid = root["alpha_grid"]
    if not isinstance(alpha_grid, list):
        raise PerturbationSweepConfigError(
            "alpha_grid",
            "must be a list",
        )
    parsed_alpha_grid = tuple(
        _finite_real(f"alpha_grid[{index}]", value)
        for index, value in enumerate(alpha_grid)
    )
    perturbation_ids = root["perturbation_ids"]
    if not isinstance(perturbation_ids, list):
        raise PerturbationSweepConfigError(
            "perturbation_ids",
            "must be a list",
        )
    parsed_perturbation_ids = tuple(
        _nonempty_string(f"perturbation_ids[{index}]", value)
        for index, value in enumerate(perturbation_ids)
    )
    if len(set(parsed_perturbation_ids)) != len(parsed_perturbation_ids):
        raise PerturbationSweepConfigError(
            "perturbation_ids",
            "must be unique",
        )
    fields = seed_derivation["fields"]
    if not isinstance(fields, list):
        raise PerturbationSweepConfigError(
            "seed_derivation.fields",
            "must be a list",
        )
    parsed_fields = tuple(
        _nonempty_string(
            f"seed_derivation.fields[{index}]",
            value,
        )
        for index, value in enumerate(fields)
    )
    config = PerturbationSweepConfig(
        path=config_path,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        schema_version=_nonempty_string(
            "schema_version",
            root["schema_version"],
        ),
        sweep_id=_nonempty_string("sweep_id", root["sweep_id"]),
        alpha_grid=parsed_alpha_grid,
        identity_alpha=_finite_real(
            "identity_alpha",
            root["identity_alpha"],
        ),
        perturbation_ids=parsed_perturbation_ids,
        global_seed=_nonnegative_int(
            "global_seed",
            root["global_seed"],
        ),
        bit_generator=_nonempty_string(
            "bit_generator",
            root["bit_generator"],
        ),
        seed_derivation=SeedDerivationConfig(
            digest=_nonempty_string(
                "seed_derivation.digest",
                seed_derivation["digest"],
            ),
            domain=_nonempty_string(
                "seed_derivation.domain",
                seed_derivation["domain"],
            ),
            encoding=_nonempty_string(
                "seed_derivation.encoding",
                seed_derivation["encoding"],
            ),
            fields=parsed_fields,
            seed_material=_nonempty_string(
                "seed_derivation.seed_material",
                seed_derivation["seed_material"],
            ),
        ),
    )
    validate_perturbation_sweep_config(config)
    return config


def derive_state_seed_material(
    config: PerturbationSweepConfig,
    *,
    perturbation_id: str,
    spectrum_id: str,
) -> tuple[int, int, int, int]:
    if not isinstance(config, PerturbationSweepConfig):
        raise PerturbationSweepConfigError(
            "config",
            "must be PerturbationSweepConfig",
        )
    if (
        not isinstance(perturbation_id, str)
        or perturbation_id not in config.perturbation_ids
    ):
        raise PerturbationSweepConfigError(
            "perturbation_id",
            "must be one of the frozen perturbation IDs",
        )
    if not isinstance(spectrum_id, str) or spectrum_id == "":
        raise PerturbationSweepConfigError(
            "spectrum_id",
            "must be a nonempty string",
        )
    payload = (
        b"rpe-perturbation-state-v1\0"
        + struct.pack("<Q", config.global_seed)
        + _length_prefixed_utf8(perturbation_id)
        + _length_prefixed_utf8(spectrum_id)
    )
    digest = hashlib.sha256(payload).digest()
    return struct.unpack("<4I", digest[:16])
