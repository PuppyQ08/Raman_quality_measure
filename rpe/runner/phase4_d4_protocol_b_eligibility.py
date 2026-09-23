from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from rpe.downstream.sugar_quantitative import D4SugarCohort, load_d4_sugar_cohort


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d4_protocol_b_all_role_eligibility_v1.json"
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d4_protocol_b_eligibility_authority.py"
PHASE05_D4_PROTOCOL_RELATIVE_PATH = "experiments/phase05/configs/d4_sugar_protocol.json"
ARCHIVE_RELATIVE_PATH = "data/raw/ramanbench/cache/10779223/Raw data.zip"
PARENT_RUN_RELATIVE_PATH = (
    "results/phase4/d4_protocol_a_full_domain_eligibility_v1/"
    "phase4-d4-protocol-a-full-domain-eligibility-"
    "0f45ae5a0815ecb852b410549ad4582c4916dbe16fa0e3e20e2079f7d088097f"
)
SCHEMA_VERSION = "phase4-d4-protocol-b-all-role-eligibility-config-v1"
EXPERIMENT_ID = "phase4-d4-protocol-b-all-role-eligibility-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d4-protocol-b-all-role-eligibility-artifact-v1"
MARKER_SCHEMA_VERSION = "phase4-d4-protocol-b-all-role-eligibility-marker-v1"
RUN_PREFIX = "phase4-d4-protocol-b-all-role-eligibility-"
PROTOCOL = "B"
ACTIVE_PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALL_PARENT_PERTURBATION_IDS = tuple(f"p{index:02d}" for index in range(1, 13))
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
CONDITION_IDS = ("alpha0",) + tuple(
    f"{perturbation_id}:{struct.pack('<d', alpha).hex()}"
    for perturbation_id in ACTIVE_PERTURBATION_IDS
    for alpha in POSITIVE_ALPHAS
)
ROLE_ORDER = ("train", "validation", "test")
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "parent_bridge.json",
    "role_condition_summaries.jsonl",
    "model_condition_readiness.jsonl",
    "blank_condition_summaries.jsonl",
    "gate.json",
    "manifest.json",
)
PARENT_INVENTORY = (
    "config.json",
    "source_records.jsonl",
    "well_folds.jsonl",
    "model_cells.jsonl",
    "model_role_occurrences.jsonl",
    "operator_cells.jsonl",
    "blank_cells.jsonl",
    "record_conditions.jsonl",
    "blank_conditions.jsonl",
    "well_summaries.jsonl",
    "common_support.jsonl",
    "gate.json",
    "manifest.json",
    "failed.json",
    "SHA256SUMS",
)
PARENT_LEDGER_ORDER = tuple(name for name in PARENT_INVENTORY if name != "SHA256SUMS")
REAL_PARENT_SHA256SUMS_SHA256 = (
    "86fc5d5635c2cc14877ab14b5cb0159fc6382ef494009697c58c77041a464d83"
)
REAL_PARENT_CONFIG_SHA256 = "d723c68ec7485b0a778287224f498884c54bec1ce0d1f688a201deacd922aff5"
FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "selected_n_components",
        "validation_scores",
        "prediction",
        "predicted_targets",
        "metric_value",
        "lod",
        "loq",
        "alignment",
        "bootstrap",
        "sign_flip",
        "holm",
        "table",
        "figure",
    }
)
FORBIDDEN_PATH_FRAGMENTS = (
    "results/phase4/d4_protocol_a_full_domain_v1/",
    "results/phase05/d4/",
)


class Phase4D4ProtocolBEligibilityError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D4ProtocolBEligibilityConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    protocol: str
    active_perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    canonical_condition_ids: tuple[str, ...]
    mixture_record_count: int
    mixture_well_count: int
    blank_record_count: int
    blank_well_count: int
    model_cell_count: int
    train_record_count: int
    validation_record_count: int
    test_record_count: int
    train_well_count: int
    validation_well_count: int
    test_well_count: int
    role_condition_summary_count: int
    model_condition_readiness_count: int
    blank_condition_summary_count: int
    parent_run_relative_path: str
    parent_sha256sums_sha256: str
    parent_required_inventory: tuple[str, ...]
    artifact_payload_files: tuple[str, ...]
    frozen_identities: Mapping[str, object]
    inherited_rulings: Mapping[str, object]
    authorities: Mapping[str, object]
    code_authority: Mapping[str, Mapping[str, object]]
    environment_authority: Mapping[str, object]
    trust_anchor: Mapping[str, object]
    claim_boundary: str


@dataclass(frozen=True)
class D4ProtocolBRoleInputs:
    source_records: tuple[Mapping[str, object], ...]
    parent_source_records: tuple[Mapping[str, object], ...]
    well_folds: tuple[Mapping[str, object], ...]
    model_cells: tuple[Mapping[str, object], ...]
    model_role_occurrences: tuple[Mapping[str, object], ...]
    blank_role_occurrences: tuple[Mapping[str, object], ...]
    source_record_ids_sha256: str
    source_well_ids_sha256: str
    model_cells_sha256: str
    model_role_occurrences_sha256: str
    condition_ids_sha256: str
    record_to_roles: Mapping[str, tuple[tuple[int, str], ...]]
    record_to_well: Mapping[str, str]
    seed_role_record_ids: Mapping[str, tuple[str, ...]]
    seed_role_well_ids: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class Phase4D4ProtocolBEligibilitySummary:
    path: Path
    run_id: str
    status: str
    marker_filename: str
    mixture_record_count: int
    model_cell_count: int
    role_condition_summary_count: int
    model_condition_readiness_count: int


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Phase4D4ProtocolBEligibilityError("json", f"unsupported {type(value).__name__}")


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def _frozen_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D4ProtocolBEligibilityError(path, "must be an object")
    return value


def _strings(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D4ProtocolBEligibilityError(path, "must be an array")
    result = tuple(str(item) for item in value)
    if any(not item for item in result):
        raise Phase4D4ProtocolBEligibilityError(path, "must contain nonempty strings")
    return result


def _floats(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D4ProtocolBEligibilityError(path, "must be an array")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in result):
        raise Phase4D4ProtocolBEligibilityError(path, "must contain finite numbers")
    return result


def _int(path: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Phase4D4ProtocolBEligibilityError(path, "must be an integer in range")
    return value


def _environment_document() -> dict[str, object]:
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "system": platform.system(),
    }


def _code_document() -> dict[str, Mapping[str, object]]:
    paths = (
        "rpe/downstream/sugar_quantitative.py",
        "rpe/runner/phase4_d4_protocol_b_eligibility.py",
        "rpe/runner/phase4_d4_protocol_b_eligibility_verifier.py",
        "tools/run_phase4_d4_protocol_b_eligibility.py",
        "tests/test_phase4_d4_protocol_b_eligibility.py",
    )
    return {
        relative: {"bytes": (ROOT / relative).stat().st_size, "sha256": _sha256_file(ROOT / relative)}
        for relative in paths
    }


def _sha256_member(archive_path: Path, member_path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_path) as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    return size, digest.hexdigest()


def _validate_structured_authorities(authorities: Mapping[str, object]) -> None:
    for name, value in authorities.items():
        receipt = _object(f"authorities.{name}", value)
        expected_bytes = _int(f"authorities.{name}.bytes", receipt.get("bytes"), minimum=1)
        expected_sha = str(receipt.get("sha256"))
        if len(expected_sha) != 64:
            raise Phase4D4ProtocolBEligibilityError(f"authorities.{name}", "sha receipt malformed")
        if "member_path" in receipt:
            archive_path = ROOT / str(receipt.get("archive_path", ""))
            if not archive_path.is_file():
                raise Phase4D4ProtocolBEligibilityError(f"authorities.{name}", "archive authority missing")
            observed_bytes, observed_sha = _sha256_member(archive_path, str(receipt["member_path"]))
        else:
            path = ROOT / str(receipt.get("path", ""))
            if not path.is_file():
                raise Phase4D4ProtocolBEligibilityError(f"authorities.{name}", "file authority missing")
            observed_bytes, observed_sha = path.stat().st_size, _sha256_file(path)
        if observed_bytes != expected_bytes or observed_sha != expected_sha:
            raise Phase4D4ProtocolBEligibilityError(f"authorities.{name}", "authority receipt sha/bytes mismatch")


def parse_phase4_d4_protocol_b_eligibility_config(
    path: Path,
    raw_bytes: bytes,
    *,
    require_frozen_identity: bool = True,
) -> Phase4D4ProtocolBEligibilityConfig:
    try:
        document_value = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D4ProtocolBEligibilityError("config", str(error)) from error
    document = _object("config", document_value)
    if raw_bytes != _canonical_json_bytes(document):
        raise Phase4D4ProtocolBEligibilityError("config", "must use canonical JSON")
    if require_frozen_identity:
        from rpe.runner.phase4_d4_protocol_b_eligibility_authority import (
            CONFIG_BYTES,
            CONFIG_SHA256,
        )

        if len(raw_bytes) != CONFIG_BYTES or _sha256_bytes(raw_bytes) != CONFIG_SHA256:
            raise Phase4D4ProtocolBEligibilityError(
                "frozen config identity", "bytes or SHA-256 mismatch"
            )
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("experiment_id", EXPERIMENT_ID),
        ("artifact_schema", ARTIFACT_SCHEMA_VERSION),
        ("marker_schema", MARKER_SCHEMA_VERSION),
        ("run_prefix", RUN_PREFIX),
        ("protocol", PROTOCOL),
    ):
        if document.get(key) != expected:
            raise Phase4D4ProtocolBEligibilityError(key, "frozen identifier mismatch")
    active = _strings("active_perturbation_ids", document.get("active_perturbation_ids"))
    alpha_grid = _floats("alpha_grid", document.get("alpha_grid"))
    conditions = _strings("canonical_condition_ids", document.get("canonical_condition_ids"))
    if active != ACTIVE_PERTURBATION_IDS or alpha_grid != ALPHA_GRID or conditions != CONDITION_IDS:
        raise Phase4D4ProtocolBEligibilityError("conditions", "canonical condition mismatch")
    payloads = _strings("artifact_payload_files", document.get("artifact_payload_files"))
    if payloads != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D4ProtocolBEligibilityError("artifact_payload_files", "order mismatch")
    denominators = _object("denominators", document.get("denominators"))
    parent = _object("parent_step27", document.get("parent_step27"))
    synthetic = bool(document.get("synthetic_fixture", False))
    authorities = _object("authorities", document.get("authorities", {}))
    code_authority = _object("code_authority", document.get("code_authority", {}))
    environment_authority = _object("environment_authority", document.get("environment_authority", {}))
    trust_anchor = _object("trust_anchor", document.get("trust_anchor", {}))
    if synthetic:
        if authorities or code_authority or environment_authority:
            raise Phase4D4ProtocolBEligibilityError(
                "synthetic authority", "synthetic fixtures require empty authority mappings"
            )
    else:
        if not authorities or not code_authority or not environment_authority:
            raise Phase4D4ProtocolBEligibilityError(
                "authority receipts", "real config requires complete authority receipts"
            )
        _validate_structured_authorities(authorities)
        if dict(environment_authority) != _environment_document():
            raise Phase4D4ProtocolBEligibilityError("environment_authority", "mismatch")
        if dict(code_authority) != _code_document():
            raise Phase4D4ProtocolBEligibilityError("code_authority", "mismatch")
        if parent.get("run_relative_path") != PARENT_RUN_RELATIVE_PATH:
            raise Phase4D4ProtocolBEligibilityError("parent_step27", "path mismatch")
        if parent.get("sha256sums_sha256") != REAL_PARENT_SHA256SUMS_SHA256:
            raise Phase4D4ProtocolBEligibilityError("parent_step27", "SHA256SUMS mismatch")
        if trust_anchor.get("config_authority_relative_path") != CONFIG_AUTHORITY_RELATIVE_PATH:
            raise Phase4D4ProtocolBEligibilityError("trust_anchor", "authority path mismatch")
    return Phase4D4ProtocolBEligibilityConfig(
        path=Path(path),
        raw_bytes=raw_bytes,
        sha256=_sha256_bytes(raw_bytes),
        document=_freeze(document),
        synthetic_fixture=synthetic,
        protocol=str(document["protocol"]),
        active_perturbation_ids=active,
        alpha_grid=alpha_grid,
        canonical_condition_ids=conditions,
        mixture_record_count=_int("denominators.mixture_record_count", denominators.get("mixture_record_count"), minimum=1),
        mixture_well_count=_int("denominators.mixture_well_count", denominators.get("mixture_well_count"), minimum=1),
        blank_record_count=_int("denominators.blank_record_count", denominators.get("blank_record_count"), minimum=1),
        blank_well_count=_int("denominators.blank_well_count", denominators.get("blank_well_count"), minimum=1),
        model_cell_count=_int("denominators.model_cell_count", denominators.get("model_cell_count"), minimum=1),
        train_record_count=_int("denominators.train_record_count", denominators.get("train_record_count"), minimum=1),
        validation_record_count=_int("denominators.validation_record_count", denominators.get("validation_record_count"), minimum=1),
        test_record_count=_int("denominators.test_record_count", denominators.get("test_record_count"), minimum=1),
        train_well_count=_int("denominators.train_well_count", denominators.get("train_well_count"), minimum=1),
        validation_well_count=_int("denominators.validation_well_count", denominators.get("validation_well_count"), minimum=1),
        test_well_count=_int("denominators.test_well_count", denominators.get("test_well_count"), minimum=1),
        role_condition_summary_count=_int("denominators.role_condition_summary_count", denominators.get("role_condition_summary_count"), minimum=1),
        model_condition_readiness_count=_int("denominators.model_condition_readiness_count", denominators.get("model_condition_readiness_count"), minimum=1),
        blank_condition_summary_count=_int("denominators.blank_condition_summary_count", denominators.get("blank_condition_summary_count"), minimum=1),
        parent_run_relative_path=str(parent["run_relative_path"]),
        parent_sha256sums_sha256=str(parent["sha256sums_sha256"]),
        parent_required_inventory=_strings("parent_step27.required_inventory", parent.get("required_inventory")),
        artifact_payload_files=payloads,
        frozen_identities=MappingProxyType(dict(_object("frozen_identities", document.get("frozen_identities", {})))),
        inherited_rulings=MappingProxyType(dict(_object("inherited_rulings", document.get("inherited_rulings", {})))),
        authorities=MappingProxyType(dict(authorities)),
        code_authority=MappingProxyType({str(k): MappingProxyType(dict(v)) for k, v in code_authority.items()}),
        environment_authority=MappingProxyType(dict(environment_authority)),
        trust_anchor=MappingProxyType(dict(trust_anchor)),
        claim_boundary=str(document.get("claim_boundary")),
    )


def load_phase4_d4_protocol_b_eligibility_config(
    path: Path,
) -> Phase4D4ProtocolBEligibilityConfig:
    return parse_phase4_d4_protocol_b_eligibility_config(
        Path(path),
        Path(path).read_bytes(),
        require_frozen_identity=True,
    )


def validate_protocol_b_outcome_blind_payload(value: object, *, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        bad = sorted(FORBIDDEN_OUTCOME_KEYS & {str(key) for key in value})
        if bad:
            raise Phase4D4ProtocolBEligibilityError(path, f"forbidden outcome keys {bad}")
        for key, item in value.items():
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        for fragment in FORBIDDEN_PATH_FRAGMENTS:
            if fragment in lowered:
                raise Phase4D4ProtocolBEligibilityError(path, f"forbidden step29/protocol_a_full_domain path {value}")


def reconstruct_d4_protocol_b_role_inputs(
    cohort: D4SugarCohort,
    config: Phase4D4ProtocolBEligibilityConfig,
) -> D4ProtocolBRoleInputs:
    record_ids = tuple(str(value) for value in cohort.record_ids)
    well_ids = tuple(str(value) for value in cohort.well_ids)
    blank_record_ids = tuple(str(value) for value in cohort.blank_record_ids)
    if len(record_ids) != config.mixture_record_count:
        raise Phase4D4ProtocolBEligibilityError("cohort.record_ids", "count mismatch")
    if len(set(well_ids)) != config.mixture_well_count:
        raise Phase4D4ProtocolBEligibilityError("cohort.well_ids", "well count mismatch")
    if len(blank_record_ids) != config.blank_record_count:
        raise Phase4D4ProtocolBEligibilityError("cohort.blank_record_ids", "count mismatch")
    record_to_well = dict(zip(record_ids, well_ids, strict=True))
    support_axis_sha256 = str(
        config.frozen_identities.get(
            "support_axis_sha256",
            config.frozen_identities.get("support_axis_f64_sha256", ""),
        )
    )
    source_records = tuple(
        {
            "record_id": record_id,
            "record_order": index,
            "source_member": str(cohort.source_members[index]),
            "well_id": well_ids[index],
        }
        for index, record_id in enumerate(record_ids)
    )
    parent_source_records = tuple(
        {
            "record_id": str(record_id),
            "scope": "blank_auxiliary",
            "support_axis_sha256": support_axis_sha256,
            "well_id": str(well_id),
        }
        for record_id, well_id in zip(cohort.blank_record_ids, cohort.blank_well_ids, strict=True)
    ) + tuple(
        {
            "record_id": str(record_id),
            "scope": "mixture",
            "support_axis_sha256": support_axis_sha256,
            "well_id": str(well_id),
        }
        for record_id, well_id in zip(cohort.record_ids, cohort.well_ids, strict=True)
    )
    well_folds = []
    model_cells = []
    role_occurrences = []
    seed_role_record_ids: dict[str, tuple[str, ...]] = {}
    seed_role_well_ids: dict[str, tuple[str, ...]] = {}
    record_to_roles: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for split in cohort.splits:
        seed = int(split.seed)
        role_index = {
            "train": tuple(int(value) for value in split.train_indices.tolist()),
            "validation": tuple(int(value) for value in split.validation_indices.tolist()),
            "test": tuple(int(value) for value in split.test_indices.tolist()),
        }
        role_records = {
            role: tuple(record_ids[index] for index in indices)
            for role, indices in role_index.items()
        }
        role_wells = {
            role: tuple(well_ids[index] for index in indices)
            for role, indices in role_index.items()
        }
        role_sources = {
            role: tuple(str(cohort.source_members[index]) for index in indices)
            for role, indices in role_index.items()
        }
        frozen_fold_well_ids = _frozen_strings(config.frozen_identities.get("fold_well_ids_sha256"))
        fold_well_ids_sha256 = (
            frozen_fold_well_ids[seed]
            if not config.synthetic_fixture and len(frozen_fold_well_ids) == config.model_cell_count
            else _ids_digest(tuple(dict.fromkeys(role_wells["test"])))
        )
        if (
            len(role_records["train"]) != config.train_record_count
            or len(role_records["validation"]) != config.validation_record_count
            or len(role_records["test"]) != config.test_record_count
        ):
            raise Phase4D4ProtocolBEligibilityError("role records", "count mismatch")
        if set(role_records["train"]) & set(role_records["validation"]) or set(role_records["train"]) & set(role_records["test"]) or set(role_records["validation"]) & set(role_records["test"]):
            raise Phase4D4ProtocolBEligibilityError("role records", "role overlap")
        if set().union(*(set(values) for values in role_records.values())) != set(record_ids):
            raise Phase4D4ProtocolBEligibilityError("role records", "union mismatch")
        model_cells.append(
            {
                "seed": seed,
                "train_record_count": len(role_records["train"]),
                "validation_record_count": len(role_records["validation"]),
                "test_record_count": len(role_records["test"]),
                "train_record_ids": list(role_records["train"]),
                "validation_record_ids": list(role_records["validation"]),
                "test_record_ids": list(role_records["test"]),
                "train_folds": list(split.train_folds),
                "validation_fold": int(split.validation_fold),
                "test_fold": int(split.test_fold),
                "train_record_ids_sha256": _ids_digest(role_records["train"]),
                "validation_record_ids_sha256": _ids_digest(role_records["validation"]),
                "test_record_ids_sha256": _ids_digest(role_records["test"]),
                "train_well_ids_sha256": _ids_digest(role_wells["train"]),
                "validation_well_ids_sha256": _ids_digest(role_wells["validation"]),
                "test_well_ids_sha256": _ids_digest(role_wells["test"]),
            }
        )
        well_folds.append(
            {
                "fold_index": seed,
                "record_ids_sha256": _ids_digest(role_records["test"]),
                "source_members_sha256": _ids_digest(role_sources["test"]),
                "well_ids_sha256": fold_well_ids_sha256,
                "seed": seed,
                "train_record_ids_sha256": _ids_digest(role_records["train"]),
                "validation_record_ids_sha256": _ids_digest(role_records["validation"]),
                "test_record_ids_sha256": _ids_digest(role_records["test"]),
                "train_well_ids_sha256": _ids_digest(role_wells["train"]),
                "validation_well_ids_sha256": _ids_digest(role_wells["validation"]),
                "test_well_ids_sha256": _ids_digest(role_wells["test"]),
            }
        )
        for role in ROLE_ORDER:
            seed_role_record_ids[f"{seed}:{role}"] = role_records[role]
            seed_role_well_ids[f"{seed}:{role}"] = role_wells[role]
            for record_id in role_records[role]:
                record_to_roles[record_id].append((seed, role))
                role_occurrences.append({"record_id": record_id, "role": role, "seed": seed})
    counts = Counter((record_id, role) for record_id, rows in record_to_roles.items() for _, role in rows)
    for record_id in record_ids:
        if counts[(record_id, "train")] != 3 or counts[(record_id, "validation")] != 1 or counts[(record_id, "test")] != 1:
            raise Phase4D4ProtocolBEligibilityError("role multiplicity", "3/1/1 mismatch")
    blank_occurrences = tuple(
        {"record_id": record_id, "role": "blank_auxiliary", "seed": seed}
        for seed in range(config.model_cell_count)
        for record_id in blank_record_ids
    )
    return D4ProtocolBRoleInputs(
        source_records=source_records,
        parent_source_records=parent_source_records,
        well_folds=tuple(well_folds),
        model_cells=tuple(model_cells),
        model_role_occurrences=tuple(role_occurrences),
        blank_role_occurrences=blank_occurrences,
        source_record_ids_sha256=_ids_digest(record_ids),
        source_well_ids_sha256=_ids_digest(well_ids),
        model_cells_sha256=_sha256_bytes(_jsonl_bytes(model_cells)),
        model_role_occurrences_sha256=_sha256_bytes(_jsonl_bytes(role_occurrences)),
        condition_ids_sha256=_ids_digest(config.canonical_condition_ids),
        record_to_roles=MappingProxyType({key: tuple(value) for key, value in record_to_roles.items()}),
        record_to_well=MappingProxyType(record_to_well),
        seed_role_record_ids=MappingProxyType(seed_role_record_ids),
        seed_role_well_ids=MappingProxyType(seed_role_well_ids),
    )


def _load_json(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Phase4D4ProtocolBEligibilityError(str(path), str(error)) from error
    if raw != _canonical_json_bytes(value):
        raise Phase4D4ProtocolBEligibilityError(str(path), "must be canonical JSON")
    return _object(str(path), value)


def _parse_sha256sums(path: Path) -> dict[str, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise Phase4D4ProtocolBEligibilityError("SHA256SUMS", "malformed ledger")
        result[parts[1]] = parts[0]
    return result


def _jsonl_documents(path: Path, *, apply_firewall: bool = True) -> tuple[Mapping[str, object], ...]:
    rows = []
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, start=1):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise Phase4D4ProtocolBEligibilityError(f"{path.name}:{number}", str(error)) from error
            if raw != _canonical_json_bytes(value):
                raise Phase4D4ProtocolBEligibilityError(f"{path.name}:{number}", "must be canonical JSON")
            row = _object(f"{path.name}:{number}", value)
            if apply_firewall:
                validate_protocol_b_outcome_blind_payload(row, path=f"{path.name}:{number}")
            rows.append(row)
    return tuple(rows)


def _parent_source_expected(inputs: D4ProtocolBRoleInputs, config: Phase4D4ProtocolBEligibilityConfig) -> tuple[Mapping[str, object], ...]:
    if config.synthetic_fixture:
        return tuple(
            {
                "record_id": str(row["record_id"]),
                "well_id": str(row["well_id"]),
                "source_member": str(row["source_member"]),
                "state": "complete",
            }
            for row in inputs.source_records
        )
    return tuple(inputs.parent_source_records)


def _parent_well_folds_expected(inputs: D4ProtocolBRoleInputs, config: Phase4D4ProtocolBEligibilityConfig) -> tuple[Mapping[str, object], ...]:
    if config.synthetic_fixture:
        return tuple(
            {
                "seed": row["seed"],
                "train_record_ids_sha256": row["train_record_ids_sha256"],
                "validation_record_ids_sha256": row["validation_record_ids_sha256"],
                "test_record_ids_sha256": row["test_record_ids_sha256"],
                "train_well_ids_sha256": row["train_well_ids_sha256"],
                "validation_well_ids_sha256": row["validation_well_ids_sha256"],
                "test_well_ids_sha256": row["test_well_ids_sha256"],
            }
            for row in inputs.well_folds
        )
    rows = []
    for index, row in enumerate(inputs.well_folds):
        rows.append(
            {
                "fold_index": index,
                "record_ids_sha256": row["test_record_ids_sha256"],
                "source_members_sha256": row["source_members_sha256"],
                "well_ids_sha256": row["well_ids_sha256"],
            }
        )
    return tuple(rows)


def _parent_model_cells_expected(inputs: D4ProtocolBRoleInputs, config: Phase4D4ProtocolBEligibilityConfig) -> tuple[Mapping[str, object], ...]:
    if config.synthetic_fixture:
        return tuple(
            {
                "seed": row["seed"],
                "train_record_count": row["train_record_count"],
                "validation_record_count": row["validation_record_count"],
                "test_record_count": row["test_record_count"],
            }
            for row in inputs.model_cells
        )
    return tuple(
        {
            "seed": row["seed"],
            "train_record_ids": row["train_record_ids"],
            "validation_record_ids": row["validation_record_ids"],
            "test_record_ids": row["test_record_ids"],
            "train_folds": row["train_folds"],
            "validation_fold": row["validation_fold"],
            "test_fold": row["test_fold"],
        }
        for row in inputs.model_cells
    )


def _parent_role_occurrences_expected(inputs: D4ProtocolBRoleInputs, config: Phase4D4ProtocolBEligibilityConfig) -> tuple[Mapping[str, object], ...]:
    if config.synthetic_fixture:
        return inputs.model_role_occurrences
    rows = []
    blank_by_seed: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in inputs.blank_role_occurrences:
        blank_by_seed[int(row["seed"])].append(row)
    for seed in range(config.model_cell_count):
        rows.extend(blank_by_seed[seed])
        for role in ("test", "train", "validation"):
            rows.extend(
                {"record_id": record_id, "role": role, "seed": seed}
                for record_id in inputs.seed_role_record_ids[f"{seed}:{role}"]
            )
    return tuple(rows)


def _assert_parent_rows_equal(name: str, actual: Sequence[Mapping[str, object]], expected: Sequence[Mapping[str, object]]) -> None:
    if len(actual) != len(expected):
        raise Phase4D4ProtocolBEligibilityError(name, "semantic row count mismatch")
    for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
        for key, value in right.items():
            if left.get(key) != value:
                raise Phase4D4ProtocolBEligibilityError(
                    name,
                    f"semantic mismatch at row {index}",
                )


def _validate_parent_static_ledgers(
    parent_run_path: Path,
    inputs: D4ProtocolBRoleInputs,
    config: Phase4D4ProtocolBEligibilityConfig,
) -> None:
    apply_firewall = config.synthetic_fixture
    _assert_parent_rows_equal(
        "source_records",
        _jsonl_documents(parent_run_path / "source_records.jsonl", apply_firewall=apply_firewall),
        _parent_source_expected(inputs, config),
    )
    _assert_parent_rows_equal(
        "well_folds",
        _jsonl_documents(parent_run_path / "well_folds.jsonl", apply_firewall=apply_firewall),
        _parent_well_folds_expected(inputs, config),
    )
    _assert_parent_rows_equal(
        "model_cells",
        _jsonl_documents(parent_run_path / "model_cells.jsonl", apply_firewall=apply_firewall),
        _parent_model_cells_expected(inputs, config),
    )
    _assert_parent_rows_equal(
        "model_role_occurrences",
        _jsonl_documents(parent_run_path / "model_role_occurrences.jsonl", apply_firewall=apply_firewall),
        _parent_role_occurrences_expected(inputs, config),
    )
    operator_rows = _jsonl_documents(parent_run_path / "operator_cells.jsonl", apply_firewall=apply_firewall)
    operator_ids = ALL_PARENT_PERTURBATION_IDS if not config.synthetic_fixture else ACTIVE_PERTURBATION_IDS
    expected_operator_count = config.mixture_record_count * len(operator_ids)
    if len(operator_rows) != expected_operator_count:
        raise Phase4D4ProtocolBEligibilityError("operator_cells", "semantic count mismatch")
    source_records = {str(row["record_id"]) for row in inputs.source_records}
    source_wells = set(inputs.record_to_well.values())
    operator_pairs = set()
    for row in operator_rows:
        record_id = str(row.get("record_id"))
        perturbation_id = str(row.get("perturbation_id"))
        if (
            record_id not in source_records
            or perturbation_id not in operator_ids
            or str(row.get("state")) not in {"complete", "not_applicable", "structurally_ineligible", "failed_runtime"}
            or (record_id, perturbation_id) in operator_pairs
        ):
            raise Phase4D4ProtocolBEligibilityError("operator_cells", "semantic schema mismatch")
        if not config.synthetic_fixture and str(row.get("well_id")) != inputs.record_to_well.get(record_id):
            raise Phase4D4ProtocolBEligibilityError("operator_cells", "semantic schema mismatch")
        operator_pairs.add((record_id, perturbation_id))
    blank_rows = _jsonl_documents(parent_run_path / "blank_cells.jsonl", apply_firewall=apply_firewall)
    blank_ids = {str(row["record_id"]) for row in inputs.blank_role_occurrences}
    if config.synthetic_fixture:
        if len(blank_rows) != config.blank_record_count:
            raise Phase4D4ProtocolBEligibilityError("blank_cells", "semantic count mismatch")
    elif len(blank_rows) != config.blank_record_count * len(ACTIVE_PERTURBATION_IDS):
        raise Phase4D4ProtocolBEligibilityError("blank_cells", "semantic count mismatch")
    if not config.synthetic_fixture:
        blank_pairs = set()
        for row in blank_rows:
            pair = (str(row.get("record_id")), str(row.get("perturbation_id")))
            if (
                pair[0] not in blank_ids
                or str(row.get("well_id")) != "E1_3"
                or pair[1] not in ACTIVE_PERTURBATION_IDS
                or str(row.get("state")) not in {"complete", "runtime_failure", "not_applicable"}
                or pair in blank_pairs
            ):
                raise Phase4D4ProtocolBEligibilityError("blank_cells", "semantic schema mismatch")
            blank_pairs.add(pair)
    well_rows = _jsonl_documents(parent_run_path / "well_summaries.jsonl", apply_firewall=apply_firewall)
    if config.synthetic_fixture:
        if len(well_rows) != config.mixture_well_count:
            raise Phase4D4ProtocolBEligibilityError("well_summaries", "semantic count mismatch")
    elif len(well_rows) != config.mixture_well_count * len(ALL_PARENT_PERTURBATION_IDS):
        raise Phase4D4ProtocolBEligibilityError("well_summaries", "semantic count mismatch")
    if not config.synthetic_fixture:
        well_pairs = set()
        for row in well_rows:
            pair = (str(row.get("well_id")), str(row.get("perturbation_id")))
            if (
                pair[0] not in source_wells
                or pair[1] not in ALL_PARENT_PERTURBATION_IDS
                or str(row.get("state")) not in {"complete", "not_evaluable_coverage"}
                or pair in well_pairs
            ):
                raise Phase4D4ProtocolBEligibilityError("well_summaries", "semantic schema mismatch")
            well_pairs.add(pair)
    support_rows = _jsonl_documents(parent_run_path / "common_support.jsonl", apply_firewall=apply_firewall)
    if config.synthetic_fixture:
        if len(support_rows) not in {len(config.canonical_condition_ids), 1}:
            raise Phase4D4ProtocolBEligibilityError("common_support", "semantic count mismatch")
    elif len(support_rows) != config.mixture_record_count + config.mixture_well_count:
        raise Phase4D4ProtocolBEligibilityError("common_support", "semantic count mismatch")
    if not config.synthetic_fixture:
        record_rows = support_rows[: config.mixture_record_count]
        well_support_rows = support_rows[config.mixture_record_count :]
        for expected_record, row in zip(inputs.source_records, record_rows, strict=True):
            if (
                row.get("scope") != "record"
                or row.get("record_id") != expected_record["record_id"]
                or row.get("well_id") != expected_record["well_id"]
                or not isinstance(row.get("complete"), bool)
            ):
                raise Phase4D4ProtocolBEligibilityError("common_support", "record semantic schema mismatch")
        for row in well_support_rows:
            if (
                row.get("scope") != "whole_well"
                or "record_id" in row
                or row.get("well_id") not in source_wells
                or not isinstance(row.get("complete"), bool)
            ):
                raise Phase4D4ProtocolBEligibilityError("common_support", "well semantic schema mismatch")


def validate_d4_protocol_b_parent_bridge(
    *,
    parent_run_path: Path,
    inputs: D4ProtocolBRoleInputs,
    config: Phase4D4ProtocolBEligibilityConfig,
    validate_condition_order: bool = True,
) -> Mapping[str, object]:
    parent_run_path = Path(parent_run_path)
    observed = tuple(sorted(item.name for item in parent_run_path.iterdir() if item.is_file()))
    expected = tuple(sorted(config.parent_required_inventory))
    if observed != expected:
        raise Phase4D4ProtocolBEligibilityError("parent inventory", "unexpected inventory")
    sha_path = parent_run_path / "SHA256SUMS"
    sha_digest = _sha256_file(sha_path)
    if not config.synthetic_fixture and sha_digest != config.parent_sha256sums_sha256:
        raise Phase4D4ProtocolBEligibilityError("parent SHA256SUMS", "checksum mismatch")
    ledger = _parse_sha256sums(sha_path)
    if tuple(ledger) != PARENT_LEDGER_ORDER:
        raise Phase4D4ProtocolBEligibilityError("parent SHA256SUMS", "ledger order mismatch")
    for name, expected_sha in ledger.items():
        if _sha256_file(parent_run_path / name) != expected_sha:
            raise Phase4D4ProtocolBEligibilityError(name, "SHA256 checksum mismatch")
    gate = _load_json(parent_run_path / "gate.json")
    full_domain = _object("parent.gate.full_domain_core", gate.get("full_domain_core"))
    marker_filename = str(gate.get("marker_filename"))
    if full_domain.get("state") != "evaluable" or marker_filename != "failed.json":
        raise Phase4D4ProtocolBEligibilityError(
            "parent gate marker", "full_domain_core evaluable must retain top-level failed.json marker"
        )
    validate_protocol_b_outcome_blind_payload(gate, path="parent.gate")
    manifest = _load_json(parent_run_path / "manifest.json")
    validate_protocol_b_outcome_blind_payload(manifest, path="parent.manifest")
    _validate_parent_static_ledgers(parent_run_path, inputs, config)
    if validate_condition_order:
        row_count = 0
        source_record_ids = tuple(inputs.record_to_well)
        for _, row in _read_jsonl(parent_run_path / "record_conditions.jsonl"):
            expected_record = source_record_ids[row_count // len(config.canonical_condition_ids)]
            expected_condition = config.canonical_condition_ids[row_count % len(config.canonical_condition_ids)]
            if str(row.get("record_id")) != expected_record or str(row.get("condition_id")) != expected_condition:
                raise Phase4D4ProtocolBEligibilityError("record_conditions", "condition canonical drift")
            row_count += 1
        if row_count != config.mixture_record_count * len(config.canonical_condition_ids):
            raise Phase4D4ProtocolBEligibilityError("record_conditions", "condition row count mismatch")
    return {
        "parent_run_path": str(parent_run_path),
        "parent_run_relative_path": config.parent_run_relative_path,
        "sha256sums_sha256": sha_digest,
        "ledger": ledger,
        "full_domain_core_state": "evaluable",
        "parent_marker_filename": marker_filename,
        "source_record_count": len(inputs.source_records),
        "model_cell_count": len(inputs.model_cells),
        "model_role_occurrence_count": len(inputs.model_role_occurrences),
        "blank_role_occurrence_count": len(inputs.blank_role_occurrences),
        "outcome_firewall": "passed",
    }


def _read_jsonl(path: Path):
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, start=1):
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise Phase4D4ProtocolBEligibilityError(f"{path.name}:{number}", str(error)) from error
            if raw != _canonical_json_bytes(value):
                raise Phase4D4ProtocolBEligibilityError(f"{path.name}:{number}", "must be canonical JSON")
            yield raw, _object(f"{path.name}:{number}", value)


def _condition_row_complete(row: Mapping[str, object]) -> bool:
    if row.get("state") != "complete":
        return False
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        if len(metrics) != 13:
            return False
        if any(not isinstance(value, Mapping) or value.get("state") != "complete" for value in metrics.values()):
            return False
    cwt = row.get("cwt")
    if isinstance(cwt, Mapping):
        if cwt.get("state") != "complete":
            return False
        for key in ("diagnostics_sha256", "peak_list_sha256"):
            value = cwt.get(key)
            if not isinstance(value, str) or len(value) != 64:
                return False
    else:
        for key in ("result_sha256", "diagnostics_sha256", "peak_list_sha256"):
            value = row.get(key)
            if not isinstance(value, str) or len(value) != 64:
                return False
    return True


def _blank_condition_row_complete(row: Mapping[str, object]) -> bool:
    if row.get("state") != "complete":
        return False
    for key in ("result_sha256", "diagnostics_sha256", "warning_sha256"):
        value = row.get(key)
        if not isinstance(value, str) or len(value) != 64:
            return False
    return True


def build_d4_protocol_b_role_lift(
    *,
    parent_run_path: Path,
    inputs: D4ProtocolBRoleInputs,
    config: Phase4D4ProtocolBEligibilityConfig,
) -> Mapping[str, object]:
    parent_bridge = validate_d4_protocol_b_parent_bridge(
        parent_run_path=parent_run_path,
        inputs=inputs,
        config=config,
        validate_condition_order=False,
    )
    condition_ids = config.canonical_condition_ids
    parent_condition_ids = condition_ids
    role_stats: dict[tuple[int, str, str], dict[str, object]] = {}
    for seed in range(config.model_cell_count):
        for role in ROLE_ORDER:
            records = inputs.seed_role_record_ids[f"{seed}:{role}"]
            wells = inputs.seed_role_well_ids[f"{seed}:{role}"]
            for condition_id in condition_ids:
                role_stats[(seed, role, condition_id)] = {
                    "observed_record_ids": set(),
                    "observed_well_ids": set(),
                    "complete": 0,
                    "not_applicable": 0,
                    "runtime": 0,
                    "missing": 0,
                    "duplicate": 0,
                    "parent_hasher": hashlib.sha256(),
                    "expected_record_count": len(records),
                    "expected_well_count": len(set(wells)),
                    "record_ids_sha256": _ids_digest(records),
                    "well_ids_sha256": _ids_digest(tuple(dict.fromkeys(wells))),
                }
    expected_row_count = config.mixture_record_count * len(condition_ids)
    seen_conditions: set[tuple[str, str]] = set()
    row_count = 0
    record_index = {record_id: index for index, record_id in enumerate(inputs.record_to_well)}
    for raw, row in _read_jsonl(Path(parent_run_path) / "record_conditions.jsonl"):
        record_id = str(row.get("record_id"))
        condition_id = str(row.get("condition_id"))
        if record_id not in inputs.record_to_roles:
            raise Phase4D4ProtocolBEligibilityError("record_conditions", "unknown record")
        validate_protocol_b_outcome_blind_payload(row, path="record_conditions")
        expected_condition = parent_condition_ids[row_count % len(condition_ids)]
        expected_record = tuple(inputs.record_to_well)[row_count // len(condition_ids)]
        if record_id != expected_record or condition_id != expected_condition:
            raise Phase4D4ProtocolBEligibilityError("record_conditions", "condition canonical drift")
        summary_condition_id = condition_ids[row_count % len(condition_ids)]
        key = (record_id, condition_id)
        if key in seen_conditions:
            for seed, role in inputs.record_to_roles[record_id]:
                role_stats[(seed, role, summary_condition_id)]["duplicate"] = int(role_stats[(seed, role, summary_condition_id)]["duplicate"]) + 1
            continue
        seen_conditions.add(key)
        complete = _condition_row_complete(row)
        for seed, role in inputs.record_to_roles[record_id]:
            stat = role_stats[(seed, role, summary_condition_id)]
            stat["observed_record_ids"].add(record_id)
            stat["observed_well_ids"].add(inputs.record_to_well[record_id])
            stat["parent_hasher"].update(raw)
            if complete:
                stat["complete"] = int(stat["complete"]) + 1
            elif row.get("state") == "not_applicable":
                stat["not_applicable"] = int(stat["not_applicable"]) + 1
            else:
                stat["runtime"] = int(stat["runtime"]) + 1
        row_count += 1
    if row_count != expected_row_count:
        raise Phase4D4ProtocolBEligibilityError("record_conditions", "row count mismatch")
    role_summaries = []
    role_summary_by_key: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for seed in range(config.model_cell_count):
        for role in ROLE_ORDER:
            for condition_id in condition_ids:
                stat = role_stats[(seed, role, condition_id)]
                missing = int(stat["expected_record_count"]) - len(stat["observed_record_ids"])
                state = "complete"
                reason = "complete"
                if missing or stat["duplicate"] or stat["not_applicable"] or stat["runtime"]:
                    state = "not_evaluable_coverage"
                    reason = "incomplete_parent_condition_receipts"
                row = {
                    "seed": seed,
                    "role": role,
                    "condition_id": condition_id,
                    "expected_record_count": stat["expected_record_count"],
                    "observed_record_count": len(stat["observed_record_ids"]),
                    "expected_well_count": stat["expected_well_count"],
                    "observed_well_count": len(stat["observed_well_ids"]),
                    "record_ids_sha256": stat["record_ids_sha256"],
                    "well_ids_sha256": stat["well_ids_sha256"],
                    "parent_condition_rows_sha256": stat["parent_hasher"].hexdigest(),
                    "complete_count": stat["complete"],
                    "not_applicable_count": stat["not_applicable"],
                    "runtime_failure_count": stat["runtime"],
                    "missing_count": missing,
                    "duplicate_count": stat["duplicate"],
                    "state": state,
                    "reason": reason,
                }
                role_summaries.append(row)
                role_summary_by_key[(seed, role, condition_id)] = row
    model_readiness = []
    for seed in range(config.model_cell_count):
        for condition_id in condition_ids:
            all_wells = []
            roles = [role_summary_by_key[(seed, role, condition_id)] for role in ROLE_ORDER]
            ready = all(row["state"] == "complete" for row in roles)
            role_sets = [set(inputs.seed_role_record_ids[f"{seed}:{role}"]) for role in ROLE_ORDER]
            well_sets = [set(inputs.seed_role_well_ids[f"{seed}:{role}"]) for role in ROLE_ORDER]
            if (
                role_sets[0] & role_sets[1]
                or role_sets[0] & role_sets[2]
                or role_sets[1] & role_sets[2]
                or well_sets[0] & well_sets[1]
                or well_sets[0] & well_sets[2]
                or well_sets[1] & well_sets[2]
            ):
                ready = False
            union_records = set().union(*role_sets)
            union_wells = set().union(*well_sets)
            if len(union_records) != config.mixture_record_count or len(union_wells) != config.mixture_well_count:
                ready = False
            for role in ROLE_ORDER:
                all_wells.extend(inputs.seed_role_well_ids[f"{seed}:{role}"])
            model_readiness.append(
                {
                    "seed": seed,
                    "condition_id": condition_id,
                    "role_summary_sha256": {
                        role: _sha256_bytes(_canonical_json_bytes(role_summary_by_key[(seed, role, condition_id)]))
                        for role in ROLE_ORDER
                    },
                    "union_record_count": len(union_records),
                    "union_well_count": len(union_wells),
                    "union_record_ids_sha256": _ids_digest(tuple(sorted(union_records, key=record_index.get))),
                    "union_well_ids_sha256": _ids_digest(tuple(dict.fromkeys(all_wells))),
                    "state": "ready" if ready else "not_ready",
                    "reason": "ready" if ready else "role_condition_not_complete",
                }
            )
    blank_stats = {
        condition_id: {"complete": 0, "failed": 0, "records": set(), "wells": set(), "hasher": hashlib.sha256()}
        for condition_id in condition_ids
    }
    blank_seen = set()
    blank_row_count = 0
    for raw, row in _read_jsonl(Path(parent_run_path) / "blank_conditions.jsonl"):
        record_id = str(row.get("record_id"))
        condition_id = str(row.get("condition_id"))
        validate_protocol_b_outcome_blind_payload(row, path="blank_conditions")
        if condition_id != parent_condition_ids[blank_row_count % len(condition_ids)]:
            raise Phase4D4ProtocolBEligibilityError("blank_conditions", "condition canonical drift")
        summary_condition_id = condition_ids[blank_row_count % len(condition_ids)]
        key = (record_id, condition_id)
        if key in blank_seen:
            blank_stats[summary_condition_id]["failed"] += 1
            continue
        blank_seen.add(key)
        stat = blank_stats[summary_condition_id]
        stat["records"].add(record_id)
        stat["wells"].add(str(row.get("well_id", "")))
        stat["hasher"].update(raw)
        if _blank_condition_row_complete(row):
            stat["complete"] += 1
        else:
            stat["failed"] += 1
        blank_row_count += 1
    if blank_row_count != config.blank_record_count * len(condition_ids):
        raise Phase4D4ProtocolBEligibilityError("blank_conditions", "row count mismatch")
    blank_summaries = []
    for condition_id in condition_ids:
        stat = blank_stats[condition_id]
        state = (
            "complete"
            if stat["complete"] == config.blank_record_count
            and not stat["failed"]
            and len(stat["wells"]) == config.blank_well_count
            else "auxiliary_not_evaluable"
        )
        blank_summaries.append(
            {
                "condition_id": condition_id,
                "expected_record_count": config.blank_record_count,
                "observed_record_count": len(stat["records"]),
                "expected_well_count": config.blank_well_count,
                "observed_well_count": len(stat["wells"]),
                "future_model_consumer_count": config.model_cell_count,
                "complete_count": stat["complete"],
                "failure_count": stat["failed"],
                "parent_condition_rows_sha256": stat["hasher"].hexdigest(),
                "state": state,
                "reason": "complete" if state == "complete" else "blank_auxiliary_incomplete",
            }
        )
    ready_count = sum(1 for row in model_readiness if row["state"] == "ready")
    role_complete_count = sum(1 for row in role_summaries if row["state"] == "complete")
    positive_by_p = {}
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        prefix = f"{perturbation_id}:"
        rows = [row for row in model_readiness if str(row["condition_id"]).startswith(prefix)]
        positive_by_p[perturbation_id] = {
            "ready_model_condition_count": sum(1 for row in rows if row["state"] == "ready"),
            "planned_model_condition_count": config.model_cell_count * len(POSITIVE_ALPHAS),
            "state": "ready" if all(row["state"] == "ready" for row in rows) else "not_ready",
        }
    primary_ready = ready_count == config.model_condition_readiness_count and role_complete_count == config.role_condition_summary_count
    marker = "complete.json" if primary_ready else "failed.json"
    gate = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "full_domain_core": {
            "state": "evaluable" if primary_ready else "not_evaluable_coverage",
            "reason": "all_role_conditions_ready" if primary_ready else "role_condition_not_complete",
            "ready_model_condition_count": ready_count,
            "planned_model_condition_count": config.model_condition_readiness_count,
        },
        "alpha0_ready_model_condition_count": sum(
            1 for row in model_readiness if row["condition_id"] == "alpha0" and row["state"] == "ready"
        ),
        "positive_conditions_by_perturbation": positive_by_p,
        "role_condition_complete_count": role_complete_count,
        "role_condition_summary_count": config.role_condition_summary_count,
        "blank_auxiliary_state": "complete" if all(row["state"] == "complete" for row in blank_summaries) else "auxiliary_not_evaluable",
        "inherited_rulings": dict(config.inherited_rulings),
        "overall_status": "complete" if primary_ready else "failed",
        "marker_filename": marker,
    }
    validate_protocol_b_outcome_blind_payload(parent_bridge, path="parent_bridge")
    validate_protocol_b_outcome_blind_payload(role_summaries, path="role_condition_summaries")
    validate_protocol_b_outcome_blind_payload(model_readiness, path="model_condition_readiness")
    validate_protocol_b_outcome_blind_payload(blank_summaries, path="blank_condition_summaries")
    validate_protocol_b_outcome_blind_payload(gate, path="gate")
    return {
        "parent_bridge": parent_bridge,
        "role_condition_summaries": role_summaries,
        "model_condition_readiness": model_readiness,
        "blank_condition_summaries": blank_summaries,
        "gate": gate,
    }


def _run_id(config: Phase4D4ProtocolBEligibilityConfig, inputs: D4ProtocolBRoleInputs) -> str:
    identity = {
        "blank_role_occurrences_sha256": _sha256_bytes(_jsonl_bytes(inputs.blank_role_occurrences)),
        "claim_boundary": config.claim_boundary,
        "condition_ids_sha256": inputs.condition_ids_sha256,
        "config_sha256": config.sha256,
        "folds_sha256": _sha256_bytes(_jsonl_bytes(inputs.well_folds)),
        "model_cells_sha256": inputs.model_cells_sha256,
        "mixture_role_occurrences_sha256": inputs.model_role_occurrences_sha256,
        "parent_sha256sums_sha256": config.parent_sha256sums_sha256,
        "source_record_ids_sha256": inputs.source_record_ids_sha256,
        "source_well_ids_sha256": inputs.source_well_ids_sha256,
    }
    return RUN_PREFIX + _sha256_bytes(_canonical_json_bytes(identity))


def _marker_payload(status: str, marker_filename: str, gate: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "marker_schema": MARKER_SCHEMA_VERSION,
        "status": status,
        "marker_filename": marker_filename,
        "full_domain_core_state": _object("gate.full_domain_core", gate["full_domain_core"])["state"],
    }


def _artifact_payloads(
    *,
    config: Phase4D4ProtocolBEligibilityConfig,
    inputs: D4ProtocolBRoleInputs,
    lift: Mapping[str, object],
    run_id: str,
) -> dict[str, bytes]:
    gate = _object("gate", lift["gate"])
    marker = str(gate["marker_filename"])
    status = str(gate["overall_status"])
    manifest = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "authority_identities": dict(config.authorities),
        "claim_boundary": config.claim_boundary,
        "code_authority": dict(config.code_authority),
        "config_sha256": config.sha256,
        "environment_authority": dict(config.environment_authority),
        "inherited_rulings": dict(config.inherited_rulings),
        "run_id": run_id,
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "reconstructed_digests": {
            "blank_role_occurrences_sha256": _sha256_bytes(_jsonl_bytes(inputs.blank_role_occurrences)),
            "condition_ids_sha256": inputs.condition_ids_sha256,
            "folds_sha256": _sha256_bytes(_jsonl_bytes(inputs.well_folds)),
            "model_cells_sha256": inputs.model_cells_sha256,
            "model_role_occurrences_sha256": inputs.model_role_occurrences_sha256,
            "source_record_ids_sha256": inputs.source_record_ids_sha256,
            "source_well_ids_sha256": inputs.source_well_ids_sha256,
        },
        "row_counts": {
            "role_condition_summaries": len(lift["role_condition_summaries"]),
            "model_condition_readiness": len(lift["model_condition_readiness"]),
            "blank_condition_summaries": len(lift["blank_condition_summaries"]),
        },
        "numerical_execution_count": 0,
        "status": status,
        "marker_filename": marker,
    }
    validate_protocol_b_outcome_blind_payload(manifest, path="manifest")
    payloads = {
        "config.json": config.raw_bytes,
        "parent_bridge.json": _canonical_json_bytes(lift["parent_bridge"]),
        "role_condition_summaries.jsonl": _jsonl_bytes(lift["role_condition_summaries"]),
        "model_condition_readiness.jsonl": _jsonl_bytes(lift["model_condition_readiness"]),
        "blank_condition_summaries.jsonl": _jsonl_bytes(lift["blank_condition_summaries"]),
        "gate.json": _canonical_json_bytes(gate),
        "manifest.json": _canonical_json_bytes(manifest),
        marker: _canonical_json_bytes(_marker_payload(status, marker, gate)),
    }
    payloads["SHA256SUMS"] = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n"
        for name in (*ARTIFACT_PAYLOAD_FILES, marker)
    ).encode("utf-8")
    return payloads


def build_phase4_d4_protocol_b_eligibility_from_inputs(
    output_dir: Path,
    *,
    parent_run_path: Path,
    inputs: D4ProtocolBRoleInputs,
    config: Phase4D4ProtocolBEligibilityConfig,
) -> Phase4D4ProtocolBEligibilitySummary:
    run_id = _run_id(config, inputs)
    path = Path(output_dir)
    if path.exists():
        raise Phase4D4ProtocolBEligibilityError(str(path), "output exists; append-only refusal")
    lift = build_d4_protocol_b_role_lift(
        parent_run_path=parent_run_path,
        inputs=inputs,
        config=config,
    )
    payloads = _artifact_payloads(config=config, inputs=inputs, lift=lift, run_id=run_id)
    path.mkdir(parents=True)
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    gate = _object("gate", lift["gate"])
    return Phase4D4ProtocolBEligibilitySummary(
        path=path,
        run_id=run_id,
        status=str(gate["overall_status"]),
        marker_filename=str(gate["marker_filename"]),
        mixture_record_count=config.mixture_record_count,
        model_cell_count=config.model_cell_count,
        role_condition_summary_count=len(lift["role_condition_summaries"]),
        model_condition_readiness_count=len(lift["model_condition_readiness"]),
    )


def build_phase4_d4_protocol_b_eligibility(
    output_root: Path,
) -> Phase4D4ProtocolBEligibilitySummary:
    config = load_phase4_d4_protocol_b_eligibility_config(ROOT / CONFIG_RELATIVE_PATH)
    cohort = load_d4_sugar_cohort(
        ROOT / PHASE05_D4_PROTOCOL_RELATIVE_PATH,
        ROOT / ARCHIVE_RELATIVE_PATH,
    )
    inputs = reconstruct_d4_protocol_b_role_inputs(cohort, config)
    run_id = _run_id(config, inputs)
    return build_phase4_d4_protocol_b_eligibility_from_inputs(
        Path(output_root) / run_id,
        parent_run_path=ROOT / config.parent_run_relative_path,
        inputs=inputs,
        config=config,
    )


def verify_phase4_d4_protocol_b_eligibility_from_inputs(
    path: Path,
    *,
    parent_run_path: Path,
    cohort: D4SugarCohort,
    config_path: Path,
) -> Phase4D4ProtocolBEligibilitySummary:
    from rpe.runner.phase4_d4_protocol_b_eligibility_verifier import (
        verify_phase4_d4_protocol_b_eligibility_from_inputs as _verify,
    )

    return _verify(path, parent_run_path=parent_run_path, cohort=cohort, config_path=config_path)


def verify_phase4_d4_protocol_b_eligibility(path: Path) -> Phase4D4ProtocolBEligibilitySummary:
    from rpe.runner.phase4_d4_protocol_b_eligibility_verifier import (
        verify_phase4_d4_protocol_b_eligibility as _verify,
    )

    return _verify(path)

