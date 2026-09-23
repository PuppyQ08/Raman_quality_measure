from __future__ import annotations

import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import h5py
import numpy as np
import scipy
import threadpoolctl
from threadpoolctl import threadpool_limits

from rpe.downstream.rruff import (
    D5RawCohort,
    load_d5_native_spectra,
    load_d5_raw_cohort,
)
from rpe.evaluation import Spectrum1D
from rpe.perturb import PerturbationSweepConfig, load_perturbation_sweep_config
from rpe.runner.phase1_config import Phase1CoreConfig, load_phase1_core_config
from rpe.runner.phase1_perturbations import (
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase4_d5_protocol_b_eligibility_authority import (
    CONFIG_BYTES,
    CONFIG_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d5_protocol_b_all_role_eligibility_v1.json"
D5_CONFIG_RELATIVE_PATH = "experiments/phase05/configs/d5_rruff_protocol.json"
DATASET_RELATIVE_PATH = "data/unified/rruff_raman_raw"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d5_protocol_b_eligibility_authority.py"

SCHEMA_VERSION = "phase4-d5-protocol-b-all-role-eligibility-config-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d5-protocol-b-all-role-eligibility-artifact-v1"
EXPERIMENT_ID = "phase4-d5-protocol-b-all-role-eligibility-v1"
RUN_PREFIX = "phase4-d5-protocol-b-all-role-eligibility-"
PROTOCOL = "B"
ACTIVE_PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
INACTIVE_PERTURBATIONS = ("p06", "p07")
ALL_PERTURBATIONS = ACTIVE_PERTURBATIONS + INACTIVE_PERTURBATIONS
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
TERMINAL_STATES = (
    "complete",
    "not_applicable",
    "failed_runtime",
    "structurally_ineligible",
)
STRUCTURAL_REASON = "structurally_ineligible_missing_explicit_baseline"
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "unique_records.jsonl",
    "role_occurrences.jsonl",
    "groups.jsonl",
    "operator_cells.jsonl",
    "record_conditions.jsonl",
    "class_summaries.jsonl",
    "gate.json",
    "manifest.json",
)
CODE_RELATIVE_PATHS = (
    "rpe/downstream/rruff.py",
    "rpe/evaluation/contracts.py",
    "rpe/perturb/axis_transform.py",
    "rpe/perturb/baseline_distortion.py",
    "rpe/perturb/contracts.py",
    "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py",
    "rpe/perturb/sweep.py",
    "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_gates.py",
    "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py",
    "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d5_protocol_b_eligibility.py",
    "rpe/runner/phase4_d5_protocol_b_eligibility_verifier.py",
    "tools/run_phase4_d5_protocol_b_eligibility.py",
)
FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "accuracy",
        "alignment_gap",
        "bootstrap",
        "correct",
        "downstream_harm",
        "figure",
        "inference",
        "matcher",
        "metric",
        "metric_value",
        "outcome",
        "p_value",
        "prediction",
        "score",
        "table",
        "top1_correct",
        "top5_correct",
    }
)
FORBIDDEN_KEY_FRAGMENTS = (
    "alignment",
    "bootstrap",
    "correct",
    "figure",
    "matcher",
    "metric",
    "outcome",
    "p_value",
    "predict",
    "score",
    "table",
    "top1",
    "top5",
)


class Phase4D5ProtocolBEligibilityError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D5ProtocolBEligibilityConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    protocol: str
    active_perturbation_ids: tuple[str, ...]
    inactive_perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    unique_record_count: int
    query_role_occurrence_count: int
    library_role_occurrence_count: int
    role_occurrence_count: int
    group_count: int
    class_count: int
    expected_operator_cell_count: int
    expected_apply_check_count: int
    expected_positive_record_condition_count: int
    expected_canonical_record_condition_count: int
    expected_class_summary_count: int
    p10_memory_budget_bytes: int
    native_gate_relative_tolerance: float
    support_start_cm1: float
    support_stop_cm1: float
    support_step_cm1: float
    support_point_count: int
    support_max_gap_cm1: float
    authorities: Mapping[str, str]
    frozen_identities: Mapping[str, object]
    inherited_rulings: Mapping[str, object]
    code_authority: Mapping[str, Mapping[str, object]]
    environment_authority: Mapping[str, object]
    claim_boundary: str
    artifact_payload_files: tuple[str, ...]
    trust_anchor: Mapping[str, object]


@dataclass(frozen=True)
class D5ProtocolBRoleLedgers:
    unique_records: tuple[Mapping[str, object], ...]
    role_occurrences: tuple[Mapping[str, object], ...]
    groups: tuple[Mapping[str, object], ...]
    unique_group_count: int
    unique_class_count: int


@dataclass(frozen=True)
class Phase4D5ProtocolBEligibilitySummary:
    path: Path
    run_id: str
    status: str
    unique_record_count: int
    role_occurrence_count: int
    operator_cell_count: int
    record_condition_count: int
    class_summary_count: int


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
    raise Phase4D5ProtocolBEligibilityError(
        "json", f"unsupported value type {type(value).__name__}"
    )


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


def _array_sha256(value: np.ndarray, *, dtype: str = "<f8") -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    return _sha256_bytes(array.tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _class_labels_digest(values: Sequence[int]) -> str:
    ordered = sorted({int(value) for value in values})
    return _sha256_bytes(
        ("\n".join(str(value) for value in ordered) + "\n").encode("utf-8")
    )


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D5ProtocolBEligibilityError(path, "must be an object")
    return value


def _strings(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D5ProtocolBEligibilityError(path, "must be an array")
    converted = tuple(str(item) for item in value)
    if any(not item for item in converted) or len(set(converted)) != len(converted):
        raise Phase4D5ProtocolBEligibilityError(path, "must contain unique nonempty strings")
    return converted


def _floats(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D5ProtocolBEligibilityError(path, "must be an array")
    converted = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in converted):
        raise Phase4D5ProtocolBEligibilityError(path, "must contain finite floats")
    return converted


def _integer(path: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4D5ProtocolBEligibilityError(path, "must be an integer")
    if value < (1 if positive else 0):
        raise Phase4D5ProtocolBEligibilityError(path, "is outside the allowed range")
    return value


def _number(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D5ProtocolBEligibilityError(path, "must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise Phase4D5ProtocolBEligibilityError(path, "must be finite")
    return result


def _lower_hex(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase4D5ProtocolBEligibilityError(path, "must be lowercase SHA-256 hex")
    return value


def _environment_document() -> dict[str, object]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _code_document() -> dict[str, Mapping[str, object]]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha256_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _config_authority_document() -> dict[str, object]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    return {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _validate_code_and_environment(
    config: Phase4D5ProtocolBEligibilityConfig,
) -> tuple[dict[str, Mapping[str, object]], dict[str, object]]:
    if config.synthetic_fixture:
        code = {str(key): dict(value) for key, value in config.code_authority.items()}
        environment = (
            dict(config.environment_authority)
            if config.environment_authority
            else _environment_document()
        )
        return code, environment
    if tuple(config.code_authority) != CODE_RELATIVE_PATHS:
        raise Phase4D5ProtocolBEligibilityError(
            "code_authority", "must contain the exact ordered frozen paths"
        )
    code = _code_document()
    expected_code = {key: dict(value) for key, value in config.code_authority.items()}
    if code != expected_code:
        raise Phase4D5ProtocolBEligibilityError(
            "code_authority", "current bytes or SHA-256 differ from config"
        )
    environment = _environment_document()
    if environment != dict(config.environment_authority):
        raise Phase4D5ProtocolBEligibilityError(
            "environment_authority", "current environment differs from config"
        )
    return code, environment


def parse_phase4_d5_protocol_b_eligibility_config(
    path: Path,
    raw: bytes,
    *,
    require_frozen_identity: bool,
) -> Phase4D5ProtocolBEligibilityConfig:
    try:
        document_value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5ProtocolBEligibilityError("config", str(error)) from error
    document = _object("config", document_value)
    if raw != _canonical_json_bytes(document):
        raise Phase4D5ProtocolBEligibilityError("config", "must use canonical JSON")
    if require_frozen_identity and (
        len(raw) != CONFIG_BYTES or _sha256_bytes(raw) != CONFIG_SHA256
    ):
        raise Phase4D5ProtocolBEligibilityError(
            "frozen config identity", "bytes or SHA-256 mismatch"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4D5ProtocolBEligibilityError("schema_version", "mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D5ProtocolBEligibilityError("experiment_id", "mismatch")
    protocol = document.get("protocol")
    if protocol != PROTOCOL:
        raise Phase4D5ProtocolBEligibilityError("protocol", "must equal 'B'")
    active = _strings("active_perturbation_ids", document.get("active_perturbation_ids"))
    inactive = _strings("inactive_perturbation_ids", document.get("inactive_perturbation_ids"))
    alpha_grid = _floats("alpha_grid", document.get("alpha_grid"))
    if active != ACTIVE_PERTURBATIONS or inactive != INACTIVE_PERTURBATIONS:
        raise Phase4D5ProtocolBEligibilityError("perturbation_ids", "frozen set mismatch")
    if alpha_grid != ALPHA_GRID:
        raise Phase4D5ProtocolBEligibilityError("alpha_grid", "frozen grid mismatch")

    artifact_payload_files = _strings(
        "artifact_payload_files", document.get("artifact_payload_files")
    )
    if artifact_payload_files != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D5ProtocolBEligibilityError(
            "artifact_payload_files", "frozen order mismatch"
        )

    denominators = _object("denominators", document.get("denominators"))
    unique_records = _integer(
        "denominators.unique_record_count",
        denominators.get("unique_record_count"),
        positive=True,
    )
    query_roles = _integer(
        "denominators.query_role_occurrence_count",
        denominators.get("query_role_occurrence_count"),
        positive=True,
    )
    library_roles = _integer(
        "denominators.library_role_occurrence_count",
        denominators.get("library_role_occurrence_count"),
        positive=True,
    )
    role_occurrences = _integer(
        "denominators.role_occurrence_count",
        denominators.get("role_occurrence_count"),
        positive=True,
    )
    groups = _integer("denominators.group_count", denominators.get("group_count"), positive=True)
    classes = _integer("denominators.class_count", denominators.get("class_count"), positive=True)
    if role_occurrences != query_roles + library_roles:
        raise Phase4D5ProtocolBEligibilityError("denominators", "role counts mismatch")

    derived = _object("derived_counts", document.get("derived_counts"))
    operator_cells = _integer(
        "derived_counts.operator_cell_count",
        derived.get("operator_cell_count"),
        positive=True,
    )
    apply_checks = _integer(
        "derived_counts.apply_check_count",
        derived.get("apply_check_count"),
        positive=True,
    )
    positive_conditions = _integer(
        "derived_counts.positive_record_condition_count",
        derived.get("positive_record_condition_count"),
        positive=True,
    )
    canonical_conditions = _integer(
        "derived_counts.canonical_record_condition_count",
        derived.get("canonical_record_condition_count"),
        positive=True,
    )
    class_summaries = _integer(
        "derived_counts.class_summary_count",
        derived.get("class_summary_count"),
        positive=True,
    )
    role_condition_refs = _integer(
        "derived_counts.role_condition_reference_count",
        derived.get("role_condition_reference_count"),
        positive=True,
    )
    if operator_cells != unique_records * len(ACTIVE_PERTURBATIONS):
        raise Phase4D5ProtocolBEligibilityError("derived_counts", "operator cell mismatch")
    if apply_checks != operator_cells * len(ALPHA_GRID):
        raise Phase4D5ProtocolBEligibilityError("derived_counts", "apply check mismatch")
    if positive_conditions != unique_records * len(ACTIVE_PERTURBATIONS) * len(POSITIVE_ALPHAS):
        raise Phase4D5ProtocolBEligibilityError("derived_counts", "positive condition mismatch")
    if canonical_conditions != unique_records * (1 + len(ACTIVE_PERTURBATIONS) * len(POSITIVE_ALPHAS)):
        raise Phase4D5ProtocolBEligibilityError("derived_counts", "canonical condition mismatch")
    if class_summaries != classes * len(ACTIVE_PERTURBATIONS):
        raise Phase4D5ProtocolBEligibilityError("derived_counts", "class summary mismatch")
    if role_condition_refs != role_occurrences * (1 + len(ACTIVE_PERTURBATIONS) * len(POSITIVE_ALPHAS)):
        raise Phase4D5ProtocolBEligibilityError("derived_counts", "role reference mismatch")

    p10 = _object("p10", document.get("p10"))
    budget = _integer("p10.memory_budget_bytes", p10.get("memory_budget_bytes"), positive=True)
    if (
        budget != 64 * 2**30
        or _number("p10.correlation_length_cm1", p10.get("correlation_length_cm1")) != 20.0
        or p10.get("peak_estimate_formula") != "32*N^2+64*N+2^30"
    ):
        raise Phase4D5ProtocolBEligibilityError("p10", "frozen resource contract mismatch")

    tolerance = _number(
        "phase1_native_gate_relative_tolerance",
        document.get("phase1_native_gate_relative_tolerance"),
    )
    if tolerance != 1e-12:
        raise Phase4D5ProtocolBEligibilityError(
            "phase1_native_gate_relative_tolerance", "must equal 1e-12"
        )

    support = _object("support_grid", document.get("support_grid"))
    support_values = (
        _number("support_grid.start_cm1", support.get("start_cm1")),
        _number("support_grid.stop_cm1", support.get("stop_cm1")),
        _number("support_grid.step_cm1", support.get("step_cm1")),
        _integer("support_grid.point_count", support.get("point_count"), positive=True),
        _number(
            "support_grid.max_in_range_native_gap_cm1",
            support.get("max_in_range_native_gap_cm1"),
        ),
    )
    if support_values != (204.0, 1800.0, 2.0, 799, 3.0):
        raise Phase4D5ProtocolBEligibilityError("support_grid", "frozen support mismatch")

    authorities_document = _object("authorities", document.get("authorities"))
    authorities = {
        str(key): _lower_hex(f"authorities.{key}", value)
        for key, value in authorities_document.items()
    }
    frozen_identities_document = _object(
        "frozen_identities", document.get("frozen_identities")
    )
    frozen_identities = dict(frozen_identities_document)
    _lower_hex("frozen_identities.record_ids_sha256", frozen_identities.get("record_ids_sha256"))
    _lower_hex("frozen_identities.group_ids_sha256", frozen_identities.get("group_ids_sha256"))
    _lower_hex("frozen_identities.class_labels_sha256", frozen_identities.get("class_labels_sha256"))
    split_digests = _strings(
        "frozen_identities.split_sha256",
        frozen_identities.get("split_sha256"),
    )
    for index, digest in enumerate(split_digests):
        _lower_hex(f"frozen_identities.split_sha256[{index}]", digest)
    frozen_identities["split_sha256"] = split_digests

    inherited_rulings = _object("inherited_rulings", document.get("inherited_rulings"))
    if set(inherited_rulings) != {
        "p01_p04_full_domain_core",
        "p05_full_domain_core",
        "p06",
        "p07",
        "peak_common_support",
    }:
        raise Phase4D5ProtocolBEligibilityError("inherited_rulings", "exact keys required")
    for key in ("p06", "p07"):
        value = _object(f"inherited_rulings.{key}", inherited_rulings[key])
        if value.get("state") != STRUCTURAL_REASON or value.get("reason") != STRUCTURAL_REASON:
            raise Phase4D5ProtocolBEligibilityError(f"inherited_rulings.{key}", "must stay structural")
    for key in ("p01_p04_full_domain_core", "p05_full_domain_core", "peak_common_support"):
        value = _object(f"inherited_rulings.{key}", inherited_rulings[key])
        if value.get("state") != "not_evaluable_coverage":
            raise Phase4D5ProtocolBEligibilityError(f"inherited_rulings.{key}", "must stay coverage-closed")

    code_document = _object("code_authority", document.get("code_authority", {}))
    code_authority: dict[str, Mapping[str, object]] = {}
    for relative, value in code_document.items():
        identity = _object(f"code_authority.{relative}", value)
        code_authority[str(relative)] = MappingProxyType(
            {
                "bytes": _integer(
                    f"code_authority.{relative}.bytes",
                    identity.get("bytes"),
                    positive=True,
                ),
                "sha256": _lower_hex(
                    f"code_authority.{relative}.sha256", identity.get("sha256")
                ),
            }
        )

    environment_authority = dict(
        _object("environment_authority", document.get("environment_authority", {}))
    )
    claim_boundary = document.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary:
        raise Phase4D5ProtocolBEligibilityError("claim_boundary", "must be nonempty")
    trust_anchor = dict(_object("trust_anchor", document.get("trust_anchor")))
    if trust_anchor.get("config_authority_relative_path") != CONFIG_AUTHORITY_RELATIVE_PATH:
        raise Phase4D5ProtocolBEligibilityError("trust_anchor", "authority path mismatch")
    synthetic = bool(document.get("synthetic_fixture", False))

    if require_frozen_identity:
        frozen_authorities = {
            "d5_config_sha256": "94f59739a2e3583c6ae33ab5f4ab489cb1f26d3c9f8cdeb689a2448c9c01e009",
            "dataset_sha256sums_sha256": "f0cb09af8d80cdf3d52af9ed01bc1ae48abba94fefdff712c30d6e8c8bfe4995",
            "parent_plan_sha256": "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5",
            "phase1_core_config_sha256": "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd",
            "phase4_step1_sha256": "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f",
            "phase4_step4_sha256": "2ae40b991d79935f173b16cf59ad8d7f531ad6a0f5717373712da96a802279e5",
            "phase4_step5_sha256": "2f1d30966e10dcce33caca86651b80747ddcc485109205ad7c1c7785ad8f4341",
            "phase4_step6_sha256": "571e22d1104c37536d8ce033a22756c2e2868ae82d021139ae7e75756f8a0100",
            "phase4_step7_sha256": "93ae3bb15c3f4b3d5e27f6a2c55e087e9e5541a0352783db98e11c01ca368375",
            "protocol_sha256": "d410659fb15c459850167e02dc82c4c81ca9561b3973838e7917ce8cb7d79fbb",
            "sweep_sha256": "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186",
        }
        if synthetic or authorities != frozen_authorities:
            raise Phase4D5ProtocolBEligibilityError("authorities", "frozen authorities mismatch")
        if (unique_records, query_roles, library_roles, role_occurrences, groups, classes) != (
            3770,
            6621,
            12229,
            18850,
            1934,
            681,
        ):
            raise Phase4D5ProtocolBEligibilityError("denominators", "frozen D5 values mismatch")
        if (operator_cells, apply_checks, positive_conditions, canonical_conditions, class_summaries) != (
            18850,
            169650,
            150800,
            154570,
            3405,
        ):
            raise Phase4D5ProtocolBEligibilityError("derived_counts", "frozen workload mismatch")

    config = Phase4D5ProtocolBEligibilityConfig(
        path=Path(path),
        raw_bytes=raw,
        sha256=_sha256_bytes(raw),
        document=_freeze(document),
        synthetic_fixture=synthetic,
        protocol=protocol,
        active_perturbation_ids=active,
        inactive_perturbation_ids=inactive,
        alpha_grid=alpha_grid,
        unique_record_count=unique_records,
        query_role_occurrence_count=query_roles,
        library_role_occurrence_count=library_roles,
        role_occurrence_count=role_occurrences,
        group_count=groups,
        class_count=classes,
        expected_operator_cell_count=operator_cells,
        expected_apply_check_count=apply_checks,
        expected_positive_record_condition_count=positive_conditions,
        expected_canonical_record_condition_count=canonical_conditions,
        expected_class_summary_count=class_summaries,
        p10_memory_budget_bytes=budget,
        native_gate_relative_tolerance=tolerance,
        support_start_cm1=support_values[0],
        support_stop_cm1=support_values[1],
        support_step_cm1=support_values[2],
        support_point_count=support_values[3],
        support_max_gap_cm1=support_values[4],
        authorities=MappingProxyType(authorities),
        frozen_identities=_freeze(frozen_identities),
        inherited_rulings=_freeze(inherited_rulings),
        code_authority=MappingProxyType(code_authority),
        environment_authority=_freeze(environment_authority),
        claim_boundary=claim_boundary,
        artifact_payload_files=artifact_payload_files,
        trust_anchor=_freeze(trust_anchor),
    )
    if require_frozen_identity:
        _validate_code_and_environment(config)
    return config


def load_phase4_d5_protocol_b_eligibility_config(
    path: Path,
) -> Phase4D5ProtocolBEligibilityConfig:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise Phase4D5ProtocolBEligibilityError("config", str(error)) from error
    return parse_phase4_d5_protocol_b_eligibility_config(
        Path(path), raw, require_frozen_identity=True
    )


def _support_grid(config: Phase4D5ProtocolBEligibilityConfig) -> np.ndarray:
    grid = np.arange(
        config.support_start_cm1,
        config.support_stop_cm1 + config.support_step_cm1 / 2.0,
        config.support_step_cm1,
        dtype="<f8",
    )
    if grid.size != config.support_point_count:
        raise Phase4D5ProtocolBEligibilityError("support_grid", "point count mismatch")
    return grid


def _project_support(
    spectrum: Spectrum1D,
    grid: np.ndarray,
    max_gap_cm1: float,
) -> tuple[str, int, float]:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5ProtocolBEligibilityError(
            "support projection", f"{spectrum.spectrum_id} would require extrapolation"
        )
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    if left < 0 or right >= axis.size:
        raise Phase4D5ProtocolBEligibilityError("support projection", "invalid bounds")
    support = axis[left : right + 1]
    max_gap = float(np.max(np.diff(support))) if support.size > 1 else math.inf
    if support.size < 2 or max_gap > max_gap_cm1:
        raise Phase4D5ProtocolBEligibilityError(
            "support projection", "native gap exceeds maximum"
        )
    values = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    if not np.isfinite(values).all():
        raise Phase4D5ProtocolBEligibilityError("support projection", "contains nonfinite values")
    norm = float(np.linalg.norm(values.astype(np.float64)))
    if not math.isfinite(norm) or norm <= 0.0:
        raise Phase4D5ProtocolBEligibilityError("support projection", "zero or nonfinite norm")
    return _array_sha256(values, dtype="<f4"), int(values.size), max_gap


def reconstruct_d5_protocol_b_role_ledgers(
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    config: Phase4D5ProtocolBEligibilityConfig,
) -> D5ProtocolBRoleLedgers:
    if not isinstance(cohort, D5RawCohort):
        raise Phase4D5ProtocolBEligibilityError("cohort", "must be D5RawCohort")
    if len(native_spectra) != len(cohort.record_ids):
        raise Phase4D5ProtocolBEligibilityError("native_spectra", "must align with cohort")
    expected_spectrum_ids = tuple(
        f"rruff_raman_raw::{record_id}" for record_id in cohort.record_ids
    )
    if tuple(spectrum.spectrum_id for spectrum in native_spectra) != expected_spectrum_ids:
        raise Phase4D5ProtocolBEligibilityError("native_spectra", "record order mismatch")
    if cohort.protocol_config_sha256 != config.authorities["d5_config_sha256"]:
        raise Phase4D5ProtocolBEligibilityError("cohort", "D5 config identity mismatch")
    observed_splits = tuple(split.split_sha256 for split in cohort.splits)
    configured_splits = tuple(config.frozen_identities["split_sha256"])
    if observed_splits != configured_splits:
        raise Phase4D5ProtocolBEligibilityError("splits", "frozen SHA-256 sequence mismatch")

    role_occurrences: list[dict[str, object]] = []
    record_roles: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"query": [], "library": []}
    )
    group_roles: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"query": [], "library": []}
    )
    for split in cohort.splits:
        query_indices = tuple(int(index) for index in split.query_indices)
        library_indices = tuple(int(index) for index in split.library_indices)
        overlap = set(query_indices) & set(library_indices)
        if overlap:
            raise Phase4D5ProtocolBEligibilityError("splits", "query/library overlap")
        if sorted(query_indices + library_indices) != list(range(len(cohort.record_ids))):
            raise Phase4D5ProtocolBEligibilityError("splits", "must partition full cohort")
        query_classes = {int(cohort.class_labels[index]) for index in query_indices}
        library_classes = {int(cohort.class_labels[index]) for index in library_indices}
        if len(query_classes) != config.class_count or len(library_classes) != config.class_count:
            raise Phase4D5ProtocolBEligibilityError("splits", "each role must cover all classes")
        for role, indices in (("query", query_indices), ("library", library_indices)):
            for role_order, cohort_index in enumerate(indices):
                record_roles[cohort_index][role].append(int(split.seed))
                group_roles[cohort.group_ids[cohort_index]][role].append(int(split.seed))
                role_occurrences.append(
                    {
                        "class_label": int(cohort.class_labels[cohort_index]),
                        "cohort_index": cohort_index,
                        "group_id": cohort.group_ids[cohort_index],
                        "record_id": cohort.record_ids[cohort_index],
                        "role": role,
                        "role_order": role_order,
                        "split_role_count": len(indices),
                        "split_seed": int(split.seed),
                        "split_sha256": split.split_sha256,
                    }
                )

    unique_records: list[dict[str, object]] = []
    for record_order, cohort_index in enumerate(range(len(cohort.record_ids))):
        spectrum = native_spectra[cohort_index]
        query_split_seeds = sorted(record_roles[cohort_index]["query"])
        library_split_seeds = sorted(record_roles[cohort_index]["library"])
        unique_records.append(
            {
                "class_label": int(cohort.class_labels[cohort_index]),
                "cohort_index": cohort_index,
                "group_id": cohort.group_ids[cohort_index],
                "library_occurrence_count": len(library_split_seeds),
                "library_split_seeds": library_split_seeds,
                "mineral_name": cohort.mineral_names[cohort_index],
                "native_axis_sha256": _array_sha256(spectrum.axis_cm1),
                "native_intensity_sha256": _array_sha256(spectrum.intensity),
                "pin_id": cohort.pin_ids[cohort_index],
                "point_count": int(spectrum.axis_cm1.size),
                "query_occurrence_count": len(query_split_seeds),
                "query_split_seeds": query_split_seeds,
                "record_id": cohort.record_ids[cohort_index],
                "record_order": record_order,
                "role_occurrence_count": len(query_split_seeds) + len(library_split_seeds),
                "rruff_id": cohort.rruff_ids[cohort_index],
            }
        )

    order_by_index = {
        int(row["cohort_index"]): int(row["record_order"]) for row in unique_records
    }
    for row in role_occurrences:
        row["unique_record_order"] = order_by_index[int(row["cohort_index"])]
    role_occurrences.sort(
        key=lambda row: (
            int(row["split_seed"]),
            0 if str(row["role"]) == "query" else 1,
            int(row["role_order"]),
        )
    )

    groups: list[dict[str, object]] = []
    group_to_records: dict[str, list[str]] = defaultdict(list)
    for row in unique_records:
        group_to_records[str(row["group_id"])].append(str(row["record_id"]))
    for group_id in sorted(group_to_records):
        records = sorted(group_to_records[group_id])
        query_count = sum(
            1
            for row in unique_records
            if row["group_id"] == group_id
            for _ in range(int(row["query_occurrence_count"]))
        )
        library_count = sum(
            1
            for row in unique_records
            if row["group_id"] == group_id
            for _ in range(int(row["library_occurrence_count"]))
        )
        groups.append(
            {
                "group_id": group_id,
                "library_role_occurrence_count": library_count,
                "query_role_occurrence_count": query_count,
                "record_count": len(records),
                "record_ids": records,
                "role_occurrence_count": query_count + library_count,
            }
        )

    record_ids = [str(row["record_id"]) for row in unique_records]
    group_ids = [str(row["group_id"]) for row in groups]
    class_labels = sorted({int(row["class_label"]) for row in unique_records})
    if _ids_digest(record_ids) != config.frozen_identities["record_ids_sha256"]:
        raise Phase4D5ProtocolBEligibilityError("record_ids", "digest mismatch")
    if _ids_digest(group_ids) != config.frozen_identities["group_ids_sha256"]:
        raise Phase4D5ProtocolBEligibilityError("group_ids", "digest mismatch")
    if _class_labels_digest(class_labels) != config.frozen_identities["class_labels_sha256"]:
        raise Phase4D5ProtocolBEligibilityError("class_labels", "digest mismatch")

    query_total = sum(int(row["query_occurrence_count"]) for row in unique_records)
    library_total = sum(int(row["library_occurrence_count"]) for row in unique_records)
    if (
        len(unique_records),
        query_total,
        library_total,
        len(role_occurrences),
        len(groups),
        len(class_labels),
    ) != (
        config.unique_record_count,
        config.query_role_occurrence_count,
        config.library_role_occurrence_count,
        config.role_occurrence_count,
        config.group_count,
        config.class_count,
    ):
        raise Phase4D5ProtocolBEligibilityError("ledger counts", "do not match config")

    return D5ProtocolBRoleLedgers(
        unique_records=tuple(MappingProxyType(row) for row in unique_records),
        role_occurrences=tuple(MappingProxyType(row) for row in role_occurrences),
        groups=tuple(MappingProxyType(row) for row in groups),
        unique_group_count=len(groups),
        unique_class_count=len(class_labels),
    )


def _phase1_source(record: Mapping[str, object], spectrum: Spectrum1D) -> Phase1Source:
    axis_f4 = np.ascontiguousarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.ascontiguousarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=int(record["record_order"]),
            record_id=str(record["record_id"]),
            sample_id=spectrum.sample_id or str(record["rruff_id"]),
            class_label=int(record["class_label"]),
            mineral_name=str(record["mineral_name"]),
            axis_id=f"native::{record['native_axis_sha256']}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha256(axis_f4, dtype="<f4"),
        source_intensity_float32_sha256=_array_sha256(intensity_f4, dtype="<f4"),
        normalized_axis_float64_sha256=str(record["native_axis_sha256"]),
        normalized_intensity_float64_sha256=str(record["native_intensity_sha256"]),
        provenance=MappingProxyType(
            {
                "d5_protocol_config_sha256": str(record.get("d5_protocol_config_sha256", "")),
                "dataset_id": "rruff_raman_raw",
                "record_id": str(record["record_id"]),
            }
        ),
    )


def _condition_id(perturbation_id: str, alpha: float) -> str:
    return f"{perturbation_id}:{np.float64(alpha).tobytes().hex()}"


def _cell_exception(cell) -> dict[str, object] | None:
    evidence = cell.evidence
    if evidence.exception_type is None:
        return None
    return {
        "message": evidence.exception_message or "unspecified",
        "path": evidence.exception_path or "unspecified",
        "type": evidence.exception_type,
    }


def _failed_runtime_receipt(
    record: Mapping[str, object],
    perturbation_id: str,
    error: BaseException,
    *,
    p10_estimate: int | None,
    state_digest: str | None = None,
) -> dict[str, object]:
    return {
        "class_label": int(record["class_label"]),
        "exception": {
            "message": str(error),
            "path": str(getattr(error, "path", "unexpected_exception")),
            "type": type(error).__name__,
        },
        "group_id": str(record["group_id"]),
        "native_gate": {},
        "output_count": 0,
        "outputs": [],
        "p10_estimated_peak_bytes": p10_estimate,
        "perturbation_id": perturbation_id,
        "reason_code": None,
        "record_id": str(record["record_id"]),
        "state": "failed_runtime",
        "state_digest": state_digest,
    }


def _cell_receipt(
    record: Mapping[str, object],
    cell,
    config: Phase4D5ProtocolBEligibilityConfig,
    grid: np.ndarray,
) -> dict[str, object]:
    p10_estimate = (
        estimate_p10_peak_bytes(int(record["point_count"]))
        if cell.perturbation_id == "p10"
        else None
    )
    state_digest = None if cell.state is None else cell.state.state_digest
    if cell.status is CellStatus.COMPLETE:
        try:
            outputs = []
            for perturbed in cell.records:
                output = perturbed.result.output
                support_hash, support_point_count, support_max_gap = _project_support(
                    output,
                    grid,
                    config.support_max_gap_cm1,
                )
                outputs.append(
                    {
                        "alpha": float(perturbed.result.alpha),
                        "alpha_float64_le_hex": perturbed.alpha_float64_le_hex,
                        "axis_changed": bool(perturbed.result.axis_changed),
                        "diagnostics": _json_ready(perturbed.result.diagnostics),
                        "intensity_changed": bool(perturbed.result.intensity_changed),
                        "output_axis_sha256": _array_sha256(output.axis_cm1),
                        "output_intensity_sha256": _array_sha256(output.intensity),
                        "output_spectrum_id": output.spectrum_id,
                        "support_max_in_range_gap_cm1": support_max_gap,
                        "support_point_count": support_point_count,
                        "support_projection_sha256": support_hash,
                    }
                )
        except Exception as error:
            return _failed_runtime_receipt(
                record,
                cell.perturbation_id,
                error,
                p10_estimate=p10_estimate,
                state_digest=state_digest,
            )
        return {
            "class_label": int(record["class_label"]),
            "exception": None,
            "group_id": str(record["group_id"]),
            "native_gate": _json_ready(cell.evidence.native_gate),
            "output_count": len(outputs),
            "outputs": outputs,
            "p10_estimated_peak_bytes": p10_estimate,
            "perturbation_id": cell.perturbation_id,
            "reason_code": None,
            "record_id": str(record["record_id"]),
            "state": "complete",
            "state_digest": state_digest,
        }
    if cell.status is CellStatus.NOT_APPLICABLE:
        state = "not_applicable"
        reason_code = cell.reason_code
    else:
        state = "failed_runtime"
        reason_code = None
    return {
        "class_label": int(record["class_label"]),
        "exception": _cell_exception(cell),
        "group_id": str(record["group_id"]),
        "native_gate": {},
        "output_count": 0,
        "outputs": [],
        "p10_estimated_peak_bytes": p10_estimate,
        "perturbation_id": cell.perturbation_id,
        "reason_code": reason_code,
        "record_id": str(record["record_id"]),
        "state": state,
        "state_digest": state_digest,
    }


def _run_record_operator_cells(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
    phase1_config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    config: Phase4D5ProtocolBEligibilityConfig,
    admission: P10MemoryAdmission,
    grid: np.ndarray,
) -> tuple[dict[str, object], ...]:
    source = _phase1_source(record, spectrum)
    receipts: list[dict[str, object]] = []
    for perturbation_id in ACTIVE_PERTURBATIONS:
        p10_estimate = (
            estimate_p10_peak_bytes(int(record["point_count"]))
            if perturbation_id == "p10"
            else None
        )
        try:
            cell = run_perturbation_cell(
                source,
                perturbation_id,
                phase1_config,
                sweep,
                p10_admission=admission,
            )
            receipts.append(_cell_receipt(record, cell, config, grid))
        except Exception as error:
            receipts.append(
                _failed_runtime_receipt(
                    record,
                    perturbation_id,
                    error,
                    p10_estimate=p10_estimate,
                )
            )
    return tuple(receipts)


def _build_record_conditions(
    records: Sequence[Mapping[str, object]],
    operator_cells: Sequence[Mapping[str, object]],
    native_spectra: tuple[Spectrum1D, ...],
    config: Phase4D5ProtocolBEligibilityConfig,
    grid: np.ndarray,
) -> list[dict[str, object]]:
    native_by_record = {
        str(row["record_id"]): native_spectra[int(row["cohort_index"])] for row in records
    }
    cell_by_key = {
        (str(cell["record_id"]), str(cell["perturbation_id"])): cell for cell in operator_cells
    }
    conditions: list[dict[str, object]] = []
    for record in records:
        record_id = str(record["record_id"])
        alpha_zero_hash, point_count, max_gap = _project_support(
            native_by_record[record_id],
            grid,
            config.support_max_gap_cm1,
        )
        conditions.append(
            {
                "alpha": 0.0,
                "alpha_float64_le_hex": np.float64(0.0).tobytes().hex(),
                "axis_sha256": str(record["native_axis_sha256"]),
                "class_label": int(record["class_label"]),
                "condition_id": "alpha0",
                "condition_kind": "alpha0",
                "group_id": str(record["group_id"]),
                "intensity_sha256": str(record["native_intensity_sha256"]),
                "perturbation_id": None,
                "record_id": record_id,
                "record_order": int(record["record_order"]),
                "state": "complete",
                "support_max_in_range_gap_cm1": max_gap,
                "support_point_count": point_count,
                "support_projection_sha256": alpha_zero_hash,
            }
        )
        alpha_zero_axis: str | None = None
        alpha_zero_intensity: str | None = None
        for perturbation_id in ACTIVE_PERTURBATIONS:
            cell = cell_by_key[(record_id, perturbation_id)]
            if str(cell["state"]) == "complete":
                outputs = list(cell["outputs"])
                alpha0 = outputs[0]
                alpha0_axis = str(alpha0["output_axis_sha256"])
                alpha0_intensity = str(alpha0["output_intensity_sha256"])
                if (
                    alpha0_axis != str(record["native_axis_sha256"])
                    or alpha0_intensity != str(record["native_intensity_sha256"])
                ):
                    raise Phase4D5ProtocolBEligibilityError(
                        "alpha-zero collapse",
                        "all active operators must preserve native alpha-zero bytes",
                    )
                if alpha_zero_axis is None:
                    alpha_zero_axis = alpha0_axis
                    alpha_zero_intensity = alpha0_intensity
                elif (
                    alpha_zero_axis != alpha0_axis
                    or alpha_zero_intensity != alpha0_intensity
                ):
                    raise Phase4D5ProtocolBEligibilityError(
                        "alpha-zero collapse",
                        "all active operators must share identical alpha-zero bytes",
                    )
                for output in outputs[1:]:
                    alpha = float(output["alpha"])
                    conditions.append(
                        {
                            "alpha": alpha,
                            "alpha_float64_le_hex": str(output["alpha_float64_le_hex"]),
                            "axis_sha256": str(output["output_axis_sha256"]),
                            "class_label": int(record["class_label"]),
                            "condition_id": _condition_id(perturbation_id, alpha),
                            "condition_kind": "positive",
                            "group_id": str(record["group_id"]),
                            "intensity_sha256": str(output["output_intensity_sha256"]),
                            "perturbation_id": perturbation_id,
                            "record_id": record_id,
                            "record_order": int(record["record_order"]),
                            "state": "complete",
                            "support_max_in_range_gap_cm1": float(
                                output["support_max_in_range_gap_cm1"]
                            ),
                            "support_point_count": int(output["support_point_count"]),
                            "support_projection_sha256": str(output["support_projection_sha256"]),
                        }
                    )
            else:
                for alpha in POSITIVE_ALPHAS:
                    conditions.append(
                        {
                            "alpha": float(alpha),
                            "alpha_float64_le_hex": np.float64(alpha).tobytes().hex(),
                            "axis_sha256": None,
                            "class_label": int(record["class_label"]),
                            "condition_id": _condition_id(perturbation_id, alpha),
                            "condition_kind": "positive",
                            "group_id": str(record["group_id"]),
                            "intensity_sha256": None,
                            "perturbation_id": perturbation_id,
                            "record_id": record_id,
                            "record_order": int(record["record_order"]),
                            "state": str(cell["state"]),
                            "support_max_in_range_gap_cm1": None,
                            "support_point_count": None,
                            "support_projection_sha256": None,
                        }
                    )
    return conditions


def evaluate_d5_protocol_b_all_role_gates(
    unique_records: Sequence[Mapping[str, object]],
    operator_cells: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, object]],
    role_occurrences: Sequence[Mapping[str, object]],
    config: Phase4D5ProtocolBEligibilityConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(unique_records) != config.unique_record_count:
        raise Phase4D5ProtocolBEligibilityError("unique_records", "denominator mismatch")
    if len(operator_cells) != config.expected_operator_cell_count:
        raise Phase4D5ProtocolBEligibilityError("operator_cells", "count mismatch")
    records_by_id = {str(row["record_id"]): row for row in unique_records}
    cell_by_key = {
        (str(row["record_id"]), str(row["perturbation_id"])): row for row in operator_cells
    }
    expected_keys = {
        (record_id, perturbation_id)
        for record_id in records_by_id
        for perturbation_id in ACTIVE_PERTURBATIONS
    }
    if set(cell_by_key) != expected_keys:
        raise Phase4D5ProtocolBEligibilityError("operator_cells", "grid keys mismatch")
    class_to_records: dict[int, list[str]] = defaultdict(list)
    group_to_records: dict[str, list[str]] = defaultdict(list)
    for row in unique_records:
        class_to_records[int(row["class_label"])].append(str(row["record_id"]))
        group_to_records[str(row["group_id"])].append(str(row["record_id"]))

    class_rows: list[dict[str, object]] = []
    operators: dict[str, dict[str, object]] = {}
    group_audits: dict[str, dict[str, object]] = {}
    role_audits: dict[str, dict[str, object]] = {}
    for perturbation_id in ACTIVE_PERTURBATIONS:
        states = Counter(
            str(cell_by_key[(record_id, perturbation_id)]["state"]) for record_id in records_by_id
        )
        complete_records = {
            record_id
            for record_id in records_by_id
            if cell_by_key[(record_id, perturbation_id)]["state"] == "complete"
        }
        failed_records = set(records_by_id) - complete_records
        complete_classes = 0
        for class_label in sorted(class_to_records):
            record_ids = sorted(class_to_records[class_label])
            complete_count = sum(record_id in complete_records for record_id in record_ids)
            complete = complete_count == len(record_ids)
            complete_classes += int(complete)
            class_rows.append(
                {
                    "class_label": class_label,
                    "complete": complete,
                    "complete_record_count": complete_count,
                    "perturbation_id": perturbation_id,
                    "required_record_count": len(record_ids),
                    "state": "complete" if complete else "closed_incomplete",
                    "state_counts": {
                        state: sum(
                            cell_by_key[(record_id, perturbation_id)]["state"] == state
                            for record_id in record_ids
                        )
                        for state in TERMINAL_STATES
                    },
                }
            )
        failed_groups = sum(
            not all(record_id in complete_records for record_id in group_to_records[group_id])
            for group_id in group_to_records
        )
        query_failed_count = sum(
            1
            for row in role_occurrences
            if str(row["role"]) == "query" and str(row["record_id"]) in failed_records
        )
        library_failed_count = sum(
            1
            for row in role_occurrences
            if str(row["role"]) == "library" and str(row["record_id"]) in failed_records
        )
        operators[perturbation_id] = {
            "complete_record_count": len(complete_records),
            "required_record_count": config.unique_record_count,
            "complete_class_count": complete_classes,
            "required_class_count": config.class_count,
            "failed_runtime_count": states.get("failed_runtime", 0),
            "not_applicable_count": states.get("not_applicable", 0),
            "state": (
                "evaluable"
                if len(complete_records) == config.unique_record_count
                and complete_classes == config.class_count
                and states.get("failed_runtime", 0) == 0
                and states.get("not_applicable", 0) == 0
                else "not_evaluable_coverage"
            ),
            "state_counts": {state: states.get(state, 0) for state in TERMINAL_STATES},
        }
        group_audits[perturbation_id] = {
            "complete_group_count": config.group_count - failed_groups,
            "failed_group_count": failed_groups,
            "required_group_count": config.group_count,
        }
        role_audits[perturbation_id] = {
            "query_complete_count": config.query_role_occurrence_count - query_failed_count,
            "query_failed_count": query_failed_count,
            "library_complete_count": config.library_role_occurrence_count - library_failed_count,
            "library_failed_count": library_failed_count,
        }

    overall = all(operators[perturbation_id]["state"] == "evaluable" for perturbation_id in ACTIVE_PERTURBATIONS)
    gate = {
        "protocol": PROTOCOL,
        "record_denominator": config.unique_record_count,
        "class_denominator": config.class_count,
        "group_denominator": config.group_count,
        "role_occurrence_denominator": config.role_occurrence_count,
        "operators": operators,
        "group_audits": group_audits,
        "role_audits": role_audits,
        "full_domain_core": {
            "perturbation_ids": list(ACTIVE_PERTURBATIONS),
            "state": "evaluable" if overall else "not_evaluable_coverage",
        },
        "inherited_rulings": _json_ready(config.inherited_rulings),
        "overall_status": "pass" if overall else "fail",
    }
    return class_rows, gate


def validate_protocol_b_outcome_blind_payload(
    value: object, *, path: str = "artifact"
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_EXACT_KEYS or any(
                fragment in key_text for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise Phase4D5ProtocolBEligibilityError(
                    "outcome-blind boundary", f"forbidden field {key!r} at {path}"
                )
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}[{index}]")


def _run_identity(
    config: Phase4D5ProtocolBEligibilityConfig,
    code: Mapping[str, object],
    environment: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    identity = {
        "authorities": config.authorities,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config_authority": _config_authority_document(),
        "config_sha256": config.sha256,
        "denominators": {
            "classes": config.class_count,
            "groups": config.group_count,
            "library_role_occurrences": config.library_role_occurrence_count,
            "query_role_occurrences": config.query_role_occurrence_count,
            "records": config.unique_record_count,
            "role_occurrences": config.role_occurrence_count,
        },
        "environment": environment,
        "frozen_identities": config.frozen_identities,
        "support_grid": {
            "max_gap_cm1": config.support_max_gap_cm1,
            "point_count": config.support_point_count,
            "start_cm1": config.support_start_cm1,
            "step_cm1": config.support_step_cm1,
            "stop_cm1": config.support_stop_cm1,
        },
        "terminal_taxonomy": list(TERMINAL_STATES),
        "trust_anchor": config.trust_anchor,
    }
    return RUN_PREFIX + _sha256_bytes(_canonical_json_bytes(identity)), identity


def _manifest_document(
    config: Phase4D5ProtocolBEligibilityConfig,
    code: Mapping[str, object],
    environment: Mapping[str, object],
    run_id: str,
    run_identity: Mapping[str, object],
    unique_records: Sequence[Mapping[str, object]],
    role_occurrences: Sequence[Mapping[str, object]],
    groups: Sequence[Mapping[str, object]],
    operator_cells: Sequence[Mapping[str, object]],
    record_conditions: Sequence[Mapping[str, object]],
    class_summaries: Sequence[Mapping[str, object]],
    gate: Mapping[str, object],
) -> dict[str, object]:
    operator_states = Counter(str(row["state"]) for row in operator_cells)
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_order": list(ARTIFACT_PAYLOAD_FILES),
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "class_summaries": len(class_summaries),
            "groups": len(groups),
            "operator_cells": len(operator_cells),
            "record_conditions": len(record_conditions),
            "role_occurrences": len(role_occurrences),
            "unique_records": len(unique_records),
        },
        "environment": environment,
        "experiment_id": EXPERIMENT_ID,
        "inherited_rulings": _json_ready(config.inherited_rulings),
        "protocol": PROTOCOL,
        "run_id": run_id,
        "run_identity": run_identity,
        "state_counts": {state: operator_states.get(state, 0) for state in TERMINAL_STATES},
        "synthetic_fixture": config.synthetic_fixture,
    }


def build_phase4_d5_protocol_b_eligibility_from_inputs(
    output_dir: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D5ProtocolBEligibilityConfig,
    worker_count: int,
) -> Phase4D5ProtocolBEligibilitySummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D5ProtocolBEligibilityError("worker_count", "must be a positive integer")
    if sweep.sha256 != config.authorities["sweep_sha256"] or tuple(sweep.alpha_grid) != config.alpha_grid:
        raise Phase4D5ProtocolBEligibilityError("sweep", "identity or alpha grid mismatch")
    if phase1_config.file_sha256 != config.authorities["phase1_core_config_sha256"]:
        raise Phase4D5ProtocolBEligibilityError("phase1_config", "identity mismatch")
    if phase1_config.core_gate["float_relative_tolerance"] != config.native_gate_relative_tolerance:
        raise Phase4D5ProtocolBEligibilityError("phase1_config", "native gate tolerance mismatch")
    code, environment = _validate_code_and_environment(config)
    ledgers = reconstruct_d5_protocol_b_role_ledgers(cohort, native_spectra, config)
    unique_records = [dict(row) for row in ledgers.unique_records]
    role_occurrences = [dict(row) for row in ledgers.role_occurrences]
    groups = [dict(row) for row in ledgers.groups]
    estimates = [estimate_p10_peak_bytes(int(row["point_count"])) for row in unique_records]
    if max(estimates) > config.p10_memory_budget_bytes:
        raise Phase4D5ProtocolBEligibilityError(
            "p10 admission", "at least one record exceeds the frozen 64 GiB budget"
        )

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Phase4D5ProtocolBEligibilityError("output_dir", "must not already exist")
    admission = P10MemoryAdmission(config.p10_memory_budget_bytes)
    native_by_index = {index: spectrum for index, spectrum in enumerate(native_spectra)}
    grid = _support_grid(config)
    results: dict[int, tuple[dict[str, object], ...]] = {}
    with threadpool_limits(limits=1, user_api="blas"):
        if worker_count == 1:
            for record in unique_records:
                order = int(record["record_order"])
                results[order] = _run_record_operator_cells(
                    record,
                    native_by_index[int(record["cohort_index"])],
                    phase1_config,
                    sweep,
                    config,
                    admission,
                    grid,
                )
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    int(record["record_order"]): executor.submit(
                        _run_record_operator_cells,
                        record,
                        native_by_index[int(record["cohort_index"])],
                        phase1_config,
                        sweep,
                        config,
                        admission,
                        grid,
                    )
                    for record in unique_records
                }
                for order, future in futures.items():
                    results[order] = future.result()
    operator_cells = [
        row
        for order in range(len(unique_records))
        for row in results[order]
    ]
    class_summaries, gate = evaluate_d5_protocol_b_all_role_gates(
        unique_records, operator_cells, groups, role_occurrences, config
    )
    record_conditions = _build_record_conditions(
        unique_records,
        operator_cells,
        native_spectra,
        config,
        grid,
    )
    if len(record_conditions) != config.expected_canonical_record_condition_count and record_conditions:
        raise Phase4D5ProtocolBEligibilityError("record_conditions", "count mismatch")
    if len(class_summaries) != config.expected_class_summary_count:
        raise Phase4D5ProtocolBEligibilityError("class_summaries", "count mismatch")

    run_id, run_identity = _run_identity(config, code, environment)
    manifest = _manifest_document(
        config,
        code,
        environment,
        run_id,
        run_identity,
        unique_records,
        role_occurrences,
        groups,
        operator_cells,
        record_conditions,
        class_summaries,
        gate,
    )
    marker_name = "complete.json" if gate["overall_status"] == "pass" else "failed.json"
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": gate["overall_status"],
    }

    payload_objects = [
        config.document,
        unique_records,
        role_occurrences,
        groups,
        operator_cells,
        record_conditions,
        class_summaries,
        gate,
        manifest,
        marker,
    ]
    for value in payload_objects:
        validate_protocol_b_outcome_blind_payload(value)

    payloads = {
        "config.json": config.raw_bytes,
        "unique_records.jsonl": b"".join(_canonical_json_bytes(row) for row in unique_records),
        "role_occurrences.jsonl": b"".join(_canonical_json_bytes(row) for row in role_occurrences),
        "groups.jsonl": b"".join(_canonical_json_bytes(row) for row in groups),
        "operator_cells.jsonl": b"".join(_canonical_json_bytes(row) for row in operator_cells),
        "record_conditions.jsonl": b"".join(_canonical_json_bytes(row) for row in record_conditions),
        "class_summaries.jsonl": b"".join(_canonical_json_bytes(row) for row in class_summaries),
        "gate.json": _canonical_json_bytes(gate),
        "manifest.json": _canonical_json_bytes(manifest),
        marker_name: _canonical_json_bytes(marker),
    }
    output_dir.mkdir(parents=True)
    ordered_payloads = (*ARTIFACT_PAYLOAD_FILES, marker_name)
    for name in ordered_payloads:
        (output_dir / name).write_bytes(payloads[name])
    checksum = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n" for name in ordered_payloads
    ).encode("utf-8")
    (output_dir / "SHA256SUMS").write_bytes(checksum)
    return Phase4D5ProtocolBEligibilitySummary(
        path=output_dir,
        run_id=run_id,
        status=str(gate["overall_status"]),
        unique_record_count=len(unique_records),
        role_occurrence_count=len(role_occurrences),
        operator_cell_count=len(operator_cells),
        record_condition_count=len(record_conditions),
        class_summary_count=len(class_summaries),
    )


def build_phase4_d5_protocol_b_eligibility(
    output_root: Path,
    *,
    worker_count: int = 16,
) -> Phase4D5ProtocolBEligibilitySummary:
    config = load_phase4_d5_protocol_b_eligibility_config(ROOT / CONFIG_RELATIVE_PATH)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1_config = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    cohort = load_d5_raw_cohort(ROOT / D5_CONFIG_RELATIVE_PATH, ROOT / DATASET_RELATIVE_PATH)
    native = load_d5_native_spectra(ROOT / DATASET_RELATIVE_PATH, cohort.record_ids)
    code, environment = _validate_code_and_environment(config)
    run_id, _ = _run_identity(config, code, environment)
    return build_phase4_d5_protocol_b_eligibility_from_inputs(
        Path(output_root) / run_id,
        cohort=cohort,
        native_spectra=native,
        sweep=sweep,
        phase1_config=phase1_config,
        config=config,
        worker_count=worker_count,
    )


__all__ = [
    "ACTIVE_PERTURBATIONS",
    "ALL_PERTURBATIONS",
    "ARTIFACT_PAYLOAD_FILES",
    "D5ProtocolBRoleLedgers",
    "INACTIVE_PERTURBATIONS",
    "Phase4D5ProtocolBEligibilityConfig",
    "Phase4D5ProtocolBEligibilityError",
    "Phase4D5ProtocolBEligibilitySummary",
    "build_phase4_d5_protocol_b_eligibility",
    "build_phase4_d5_protocol_b_eligibility_from_inputs",
    "evaluate_d5_protocol_b_all_role_gates",
    "load_phase4_d5_protocol_b_eligibility_config",
    "parse_phase4_d5_protocol_b_eligibility_config",
    "reconstruct_d5_protocol_b_role_ledgers",
    "validate_protocol_b_outcome_blind_payload",
]
