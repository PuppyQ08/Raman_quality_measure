from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import platform
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import h5py
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
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
from rpe.runner.phase4_d1_protocol_b_eligibility_authority import (
    CONFIG_BYTES,
    CONFIG_SHA256,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = (
    "experiments/phase4/configs/d1_protocol_b_all_role_eligibility_v1.json"
)
DATASET_RELATIVE_PATH = "data/unified/bacteria_id_reference"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
D1_PROTOCOL_A_CONFIG_RELATIVE_PATH = (
    "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json"
)
CONFIG_AUTHORITY_RELATIVE_PATH = (
    "rpe/runner/phase4_d1_protocol_b_eligibility_authority.py"
)
CODE_RELATIVE_PATHS = (
    "rpe/downstream/bacteria_id.py",
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
    "rpe/runner/phase4_d1_protocol_b_eligibility.py",
    "rpe/runner/phase4_d1_protocol_b_eligibility_verifier.py",
    "tools/run_phase4_d1_protocol_b_eligibility.py",
)

SCHEMA_VERSION = "phase4-d1-protocol-b-all-role-eligibility-config-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d1-protocol-b-all-role-eligibility-artifact-v1"
EXPERIMENT_ID = "phase4-d1-protocol-b-all-role-eligibility-v1"
RUN_PREFIX = "phase4-d1-protocol-b-all-role-eligibility-"
PROTOCOL = "B"
ENDPOINT_ID = "full_domain_core"
MODEL_SEEDS = (0, 1, 2, 3, 4)
ACTIVE_PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
CONDITION_COUNT = 1 + len(ACTIVE_PERTURBATION_IDS) * len(POSITIVE_ALPHAS)
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "source_records.jsonl",
    "model_cells.jsonl",
    "model_role_occurrences.jsonl",
    "operator_cells.jsonl",
    "record_conditions.jsonl",
    "condition_matrix_shards.jsonl",
    "class_summaries.jsonl",
    "gate.json",
    "manifest.json",
)
TERMINAL_STATES = ("complete", "not_applicable", "failed_runtime")

TEST_IDS_SHA256 = "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"
SOURCE_LEDGER_SHA256 = "a44e71531c5a639904390ca73738d035e0c3294766f46a8def82b2f9bd9fec1b"
MODEL_LEDGER_SHA256 = "1b77f2825e25265382597e64218f4a2771011cd9cc8f86c4c3f6e729e6c2c486"
ROLE_LEDGER_SHA256 = "211b087e21f3b511acd6acaa800b693ffef0db7c1aaa354c902d146ee2c2090c"
SUPPORT_F64_SHA256 = "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6"
SUPPORT_F32_SHA256 = "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59"
NATIVE_F64_SHA256 = "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d"

REAL_SOURCE_COUNT = 66000
REAL_CLASS_COUNT = 30
REAL_MODEL_CELL_COUNT = 5
REAL_ROLE_OCCURRENCE_COUNT = 330000
REAL_SHARD_SOURCE_COUNT = 1000
REAL_SUPPORT_POINT_COUNT = 997
P10_MEMORY_BUDGET_BYTES = 64 * 1024**3
CLAIM_BOUNDARY = "outcome_blind_protocol_b_all_role_eligibility_only"
REQUIRED_AUTHORITY_FILE_RECEIPTS = MappingProxyType(
    {
        "parent_plan": ("raman_preproc_benchmark_plan_v2.md", 46363, "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5"),
        "phase4_preregistration": ("reports/phase4/step01_phase4_feasibility_preregistration.md", 31265, "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f"),
        "phase4_step03": ("reports/phase4/step03_alignment_core.md", 8513, "76b1ffdc3544673cebbd6520803ae72aeff610a6a66a2306dd317c9402fdcbbe"),
        "step13_report": ("reports/phase4/step13_d2_protocol_a_eligibility_preflight.md", 13592, "ab06d47e2e18301c0c1bfb53aef57284fd616b086a9b05cb552869a826d155d9"),
        "step13_config": ("experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json", 23129, "f92427c2f18ab445db2bb54d5ea5a97cce21b05dd6ee80f50f3ed4082285a15b"),
        "step13_sha256sums": ("results/phase4/d2_protocol_a_full_domain_eligibility_v1/phase4-d2-protocol-a-full-domain-eligibility-0f43f329bfc6a7a1ff7232a336d115849b06dbca884de28d71f6b28502598ef8/SHA256SUMS", 1094, "b41638b4e246326b155ef1c6169b57d93c68266b47a41b9d16f05dc516546e9e"),
        "step16_design": ("reports/phase4/step16_d2_protocol_b_all_role_eligibility_design.md", 19740, "1039788a626c58ea989a7038aeca1b5fa25ef13edd33d982b0654f0e6dd92bc2"),
        "step17_report": ("reports/phase4/step17_d2_protocol_b_all_role_eligibility.md", 10396, "7d0a1842cd11acc095b37c403ea740d243075f3cb496b63dd0edadb0d1e12b9a"),
        "step20_design": ("reports/phase4/step20_d1_protocol_a_full_domain_outcome_design.md", 28296, "9a351a414988d0734851d798b82eb1cc2edad73ebe5b133aa235e5e903b4d00d"),
        "step21_report": ("reports/phase4/step21_d1_protocol_a_full_domain.md", 14391, "fb1990d030b5d16670b4799ad3deb17767131f0fbc628b28c7c11da678414225"),
        "d1_protocol_a_config": (D1_PROTOCOL_A_CONFIG_RELATIVE_PATH, 42260, "13342f4633afb215c02d10ae317e725e162721261169b38c9c0c0c57254d32be"),
        "d1_phase05_protocol": ("reports/phase05/d1_protocol_proposal.md", 14073, "cb8a1c5ba9f4cd716a7925c836fdb9baebb86e17b5d2c00d86dc1c6c1c1ee999"),
        "d1_recipe_config": ("experiments/phase05/configs/d1_bacteria_id_pca20_lr_sg11.json", 1051, "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"),
        "sweep": (SWEEP_RELATIVE_PATH, 559, "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"),
        "phase1_core_config": (PHASE1_CONFIG_RELATIVE_PATH, 2350, "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd"),
        "bacteria_id_sha256sums": ("data/unified/bacteria_id_reference/SHA256SUMS", 235, "6d6d1399ac0e5197a9e51a1a0cf32edf51c924ce8d7c12aeb9d58f9af93c7f3e"),
        "bacteria_id_retained_snapshot": ("data/unified/bacteria_id_reference/SHA256SUMS.sha256", 77, "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02"),
        "step22_design": ("reports/phase4/step22_d1_protocol_b_all_role_eligibility_design.md", 23333, "90654ea127eed2aaa7a4b62fd189ab77ea5afca37c504790c8d9db1d6dd4b37b"),
    }
)

FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "accuracy",
        "acc_cross",
        "ag",
        "alignment",
        "alignment_gap",
        "bootstrap",
        "coefficient",
        "coefficients",
        "correct",
        "correctness",
        "figure",
        "inference",
        "metric",
        "metric_value",
        "outcome",
        "p_value",
        "peak",
        "peak_count",
        "peak_list",
        "peak_lists",
        "peaks",
        "prediction",
        "predictions",
        "score",
        "selected_c",
        "table",
        "test_accuracy",
        "validation_accuracy",
        "validation_score",
    }
)
FORBIDDEN_KEY_FRAGMENTS = (
    "accuracy",
    "alignment",
    "bootstrap",
    "coefficient",
    "correct",
    "figure",
    "inference",
    "metric",
    "outcome",
    "p_value",
    "predict",
    "score",
    "selected_c",
    "table",
)

REAL_INHERITED_RULINGS = MappingProxyType(
    {
        "p01_p04_full_domain_core": MappingProxyType(
            {
                "bound": MappingProxyType(
                    {
                        "class_upper_bound_complete": 0,
                        "record_upper_bound_complete": 63208,
                        "required_class_count": 30,
                        "required_record_count": 66000,
                    }
                ),
                "reason": "zero_complete_test_class_bound_inherited_from_step13",
                "state": "not_evaluable_coverage",
            }
        ),
        "p05_full_domain_core": MappingProxyType(
            {
                "bound": MappingProxyType(
                    {
                        "class_upper_bound_complete": 0,
                        "record_upper_bound_complete": 63002,
                        "required_class_count": 30,
                        "required_record_count": 66000,
                    }
                ),
                "reason": "zero_complete_test_class_bound_inherited_from_step13",
                "state": "not_evaluable_coverage",
            }
        ),
        "p06": MappingProxyType(
            {
                "reason": "structurally_ineligible_missing_explicit_baseline",
                "state": "structurally_ineligible_missing_explicit_baseline",
            }
        ),
        "p07": MappingProxyType(
            {
                "reason": "structurally_ineligible_missing_explicit_baseline",
                "state": "structurally_ineligible_missing_explicit_baseline",
            }
        ),
    }
)


class Phase4D1ProtocolBEligibilityError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D1ProtocolBEligibilityConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    endpoint_count: int
    model_seeds: tuple[int, ...]
    active_perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    artifact_payload_files: tuple[str, ...]
    condition_matrix_shard_files: tuple[str, ...]
    source_record_count: int
    model_cell_count: int
    model_role_occurrence_count: int
    class_count: int
    expected_operator_cell_count: int
    expected_apply_check_count: int
    expected_positive_record_condition_count: int
    expected_record_condition_count: int
    expected_role_condition_audit_count: int
    expected_class_summary_count: int
    expected_numerical_value_count: int
    expected_numerical_byte_count: int
    support_coordinates_cm1: tuple[float, ...]
    support_point_count: int
    support_max_gap_cm1: float
    shard_source_count: int
    condition_matrix_shard_count: int
    p10_memory_budget_bytes: int
    native_gate_relative_tolerance: float
    authorities: Mapping[str, Mapping[str, object]]
    frozen_identities: Mapping[str, object]
    inherited_rulings: Mapping[str, object]
    code_authority: Mapping[str, Mapping[str, object]]
    environment_authority: Mapping[str, object]
    claim_boundary: str
    trust_anchor: Mapping[str, object]


@dataclass(frozen=True)
class D1ProtocolBEligibilityInputs:
    source_records: tuple[Mapping[str, object], ...]
    model_cells: tuple[Mapping[str, object], ...]
    model_role_occurrences: tuple[Mapping[str, object], ...]
    source_spectra: tuple[Spectrum1D, ...]
    source_projections: tuple[np.ndarray, ...]
    source_record_count: int
    model_cell_count: int
    role_occurrence_count: int
    source_ledger_sha256: str
    model_ledger_sha256: str
    role_ledger_sha256: str
    test_record_ids_sha256: str
    support_axis_f64_sha256: str
    support_axis_f32_sha256: str
    native_axis_f64_sha256: str
    native_axis_point_count: int
    support_axis_point_count: int
    role_overlap_detected: bool


@dataclass(frozen=True)
class Phase4D1ProtocolBEligibilitySummary:
    path: Path
    run_id: str
    status: str
    endpoint_count: int
    model_seed_count: int
    source_record_count: int
    model_cell_count: int
    role_occurrence_count: int
    operator_cell_count: int
    record_condition_count: int
    class_summary_count: int
    condition_matrix_shard_count: int
    condition_matrix_bytes: int


@dataclass(frozen=True)
class _ExecutedSource:
    operator_cells: tuple[Mapping[str, object], ...]
    condition_projections: tuple[np.ndarray | None, ...]


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Phase4D1ProtocolBEligibilityError(
        "json",
        f"unsupported value type {type(value).__name__}",
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
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha_bytes(("\n".join(str(value) for value in values) + "\n").encode("utf-8"))


def _canonical_ledger_digest(rows: Sequence[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical_json_bytes(row))
    return digest.hexdigest()


def _authority_document() -> dict[str, Mapping[str, object]]:
    return {
        key: {"bytes": size, "path": relative_path, "sha256": digest}
        for key, (relative_path, size, digest) in REQUIRED_AUTHORITY_FILE_RECEIPTS.items()
    }


def _code_document() -> dict[str, Mapping[str, object]]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha_file(ROOT / relative_path),
        }
        for relative_path in CODE_RELATIVE_PATHS
    }


def _validate_real_authority_receipts() -> None:
    for key, (relative_path, expected_bytes, expected_sha256) in (
        REQUIRED_AUTHORITY_FILE_RECEIPTS.items()
    ):
        path = ROOT / relative_path
        if not path.is_file():
            raise Phase4D1ProtocolBEligibilityError(
                f"authorities.{key}",
                "authority path is missing",
            )
        if path.stat().st_size != expected_bytes or _sha_file(path) != expected_sha256:
            raise Phase4D1ProtocolBEligibilityError(
                f"authorities.{key}",
                "live authority bytes or SHA-256 mismatch",
            )


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D1ProtocolBEligibilityError(path, "must be an object")
    return value


def _int(path: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4D1ProtocolBEligibilityError(path, "must be an integer")
    if value < minimum:
        raise Phase4D1ProtocolBEligibilityError(path, "is outside the allowed range")
    return int(value)


def _number(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D1ProtocolBEligibilityError(path, "must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise Phase4D1ProtocolBEligibilityError(path, "must be finite")
    return number


def _strings(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D1ProtocolBEligibilityError(path, "must be an array")
    converted = tuple(str(item) for item in value)
    if any(not item for item in converted):
        raise Phase4D1ProtocolBEligibilityError(path, "must contain nonempty strings")
    return converted


def _floats(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D1ProtocolBEligibilityError(path, "must be an array")
    converted = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in converted):
        raise Phase4D1ProtocolBEligibilityError(path, "must contain finite numbers")
    return converted


def _shard_name(start: int, end: int) -> str:
    return f"condition_matrices_{start:05d}_{end:05d}.f32le"


def _condition_matrix_shard_files(
    source_record_count: int,
    shard_source_count: int,
) -> tuple[str, ...]:
    return tuple(
        _shard_name(start, min(start + shard_source_count, source_record_count) - 1)
        for start in range(0, source_record_count, shard_source_count)
    )


def _complete_artifact_order(shard_files: Sequence[str]) -> tuple[str, ...]:
    return (
        ARTIFACT_PAYLOAD_FILES[:7]
        + tuple(str(name) for name in shard_files)
        + ARTIFACT_PAYLOAD_FILES[7:]
    )


def validate_protocol_b_outcome_blind_payload(
    value: object,
    *,
    path: str = "payload",
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_EXACT_KEYS or any(
                key_text.startswith(fragment) for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise Phase4D1ProtocolBEligibilityError(
                    "outcome-blind boundary",
                    f"forbidden field {key!r} at {path}",
                )
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}[{index}]")


def _default_support(point_count: int) -> tuple[float, ...]:
    return tuple(
        float(value)
        for value in np.linspace(
            386.6499938964844,
            1792.4000244140625,
            int(point_count),
            dtype="<f8",
        )
    )


def _real_support_coordinates() -> tuple[float, ...]:
    try:
        document = json.loads((ROOT / D1_PROTOCOL_A_CONFIG_RELATIVE_PATH).read_bytes())
        support = _object("d1_protocol_a.support_grid", document["support_grid"])
        return _floats(
            "d1_protocol_a.support_grid.coordinates_cm1",
            support["coordinates_cm1"],
        )
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D1ProtocolBEligibilityError(
            "support_grid.coordinates_cm1",
            f"bound D1 support unavailable: {error}",
        ) from error


def parse_phase4_d1_protocol_b_eligibility_config(
    path: Path,
    raw: bytes,
    *,
    require_frozen_identity: bool,
) -> Phase4D1ProtocolBEligibilityConfig:
    try:
        document_value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D1ProtocolBEligibilityError("config", str(error)) from error
    document = _object("config", document_value)
    if raw != _canonical_json_bytes(document):
        raise Phase4D1ProtocolBEligibilityError("config", "must use canonical JSON")
    validate_protocol_b_outcome_blind_payload(document, path="config")
    if require_frozen_identity and (
        len(raw) != CONFIG_BYTES or _sha_bytes(raw) != CONFIG_SHA256
    ):
        raise Phase4D1ProtocolBEligibilityError(
            "frozen config identity",
            "bytes or SHA-256 mismatch",
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4D1ProtocolBEligibilityError("schema_version", "mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D1ProtocolBEligibilityError("experiment_id", "mismatch")
    if document.get("protocol") != PROTOCOL:
        raise Phase4D1ProtocolBEligibilityError("protocol", "must equal 'B'")
    if str(document.get("endpoint_id")) != ENDPOINT_ID:
        raise Phase4D1ProtocolBEligibilityError("endpoint_id", "mismatch")
    synthetic_fixture = bool(document.get("synthetic_fixture", False))
    active = _strings("active_perturbation_ids", document.get("active_perturbation_ids"))
    if active != ACTIVE_PERTURBATION_IDS:
        raise Phase4D1ProtocolBEligibilityError("active_perturbation_ids", "mismatch")
    alpha_grid = _floats("alpha_grid", document.get("alpha_grid"))
    if alpha_grid != ALPHA_GRID:
        raise Phase4D1ProtocolBEligibilityError("alpha_grid", "mismatch")
    model_seeds = tuple(
        _int("model_seeds[]", value, minimum=0)
        for value in document.get("model_seeds", ())
    )
    if model_seeds != MODEL_SEEDS:
        raise Phase4D1ProtocolBEligibilityError("model_seeds", "mismatch")
    denominators = _object("denominators", document.get("denominators"))
    class_count = _int("denominators.class_count", denominators.get("class_count"), minimum=1)
    source_count = _int(
        "denominators.source_record_count",
        denominators.get("source_record_count"),
        minimum=1,
    )
    model_count = _int(
        "denominators.model_cell_count",
        denominators.get("model_cell_count"),
        minimum=1,
    )
    role_count = _int(
        "denominators.model_role_occurrence_count",
        denominators.get("model_role_occurrence_count"),
        minimum=1,
    )
    endpoint_count = _int("denominators.endpoint_count", denominators.get("endpoint_count"), minimum=1)
    expected = _object("expected", document.get("expected"))
    operator_count = _int(
        "expected.operator_cell_count",
        expected.get("operator_cell_count"),
        minimum=1,
    )
    apply_check_count = _int(
        "expected.apply_check_count",
        expected.get("apply_check_count", operator_count * len(ALPHA_GRID)),
        minimum=1,
    )
    positive_condition_count = _int(
        "expected.positive_record_condition_count",
        expected.get("positive_record_condition_count", source_count * len(ACTIVE_PERTURBATION_IDS) * len(POSITIVE_ALPHAS)),
        minimum=1,
    )
    condition_count = _int(
        "expected.record_condition_count",
        expected.get("record_condition_count"),
        minimum=1,
    )
    class_summary_count = _int(
        "expected.class_summary_count",
        expected.get("class_summary_count"),
        minimum=1,
    )
    role_condition_count = _int(
        "expected.role_condition_audit_count",
        expected.get("role_condition_audit_count", role_count * CONDITION_COUNT),
        minimum=1,
    )
    support_grid = _object("support_grid", document.get("support_grid"))
    support_point_count = _int("support_grid.point_count", support_grid.get("point_count"), minimum=1)
    support_coordinates_raw = support_grid.get("coordinates_cm1")
    if support_coordinates_raw is None:
        if synthetic_fixture:
            raise Phase4D1ProtocolBEligibilityError(
                "support_grid.coordinates_cm1",
                "synthetic fixtures must provide explicit coordinates",
            )
        support_coordinates = _real_support_coordinates()
    else:
        support_coordinates = _floats(
            "support_grid.coordinates_cm1",
            support_coordinates_raw,
        )
    if len(support_coordinates) != support_point_count:
        raise Phase4D1ProtocolBEligibilityError("support_grid", "point count mismatch")
    support_max_gap = _number(
        "support_grid.max_in_range_native_gap_cm1",
        support_grid.get("max_in_range_native_gap_cm1"),
    )
    storage = _object("condition_matrix_store", document.get("condition_matrix_store"))
    shard_source_count = _int(
        "condition_matrix_store.shard_source_count",
        storage.get("shard_source_count"),
        minimum=1,
    )
    shard_count = _int(
        "condition_matrix_store.shard_count",
        storage.get("shard_count"),
        minimum=1,
    )
    shard_files = _condition_matrix_shard_files(source_count, shard_source_count)
    if len(shard_files) != shard_count:
        raise Phase4D1ProtocolBEligibilityError(
            "condition_matrix_store.shard_count",
            "does not match source-range partition",
        )
    configured_shard_files = _strings(
        "condition_matrix_store.shard_files",
        storage.get("shard_files", shard_files),
    )
    if configured_shard_files != shard_files:
        raise Phase4D1ProtocolBEligibilityError(
            "condition_matrix_store.shard_files",
            "mismatch",
        )
    payload_files = _strings(
        "artifact_payload_files",
        document.get("artifact_payload_files"),
    )
    if payload_files != _complete_artifact_order(shard_files):
        raise Phase4D1ProtocolBEligibilityError(
            "artifact_payload_files",
            "mismatch",
        )
    p10 = _object("p10", document.get("p10"))
    memory_budget = _int("p10.memory_budget_bytes", p10.get("memory_budget_bytes"), minimum=1)
    if (
        _number("p10.correlation_length_cm1", p10.get("correlation_length_cm1")) != 20.0
        or p10.get("peak_estimate_formula") != "32*N^2+64*N+2^30"
    ):
        raise Phase4D1ProtocolBEligibilityError("p10", "frozen contract mismatch")
    native_tolerance = _number(
        "phase1_native_gate_relative_tolerance",
        document.get("phase1_native_gate_relative_tolerance"),
    )
    if native_tolerance != 1e-12:
        raise Phase4D1ProtocolBEligibilityError(
            "phase1_native_gate_relative_tolerance",
            "must equal 1e-12",
        )
    numerical_value_count = _int(
        "expected.numerical_value_count",
        expected.get("numerical_value_count", condition_count * support_point_count),
        minimum=1,
    )
    numerical_byte_count = _int(
        "expected.numerical_byte_count",
        expected.get("numerical_byte_count", numerical_value_count * 4),
        minimum=1,
    )
    authorities = MappingProxyType(
        {
            str(key): _freeze(value)
            for key, value in _object("authorities", document.get("authorities", {})).items()
        }
    )
    frozen = MappingProxyType(
        {
            str(key): _freeze(value)
            for key, value in _object("frozen_identities", document.get("frozen_identities", {})).items()
        }
    )
    inherited = MappingProxyType(
        {
            str(key): _freeze(value)
            for key, value in _object("inherited_rulings", document.get("inherited_rulings")).items()
        }
    )
    code_authority = MappingProxyType(
        {
            str(key): _freeze(value)
            for key, value in _object("code_authority", document.get("code_authority", {})).items()
        }
    )
    environment_authority = MappingProxyType(
        {
            str(key): _freeze(value)
            for key, value in _object("environment_authority", document.get("environment_authority", {})).items()
        }
    )
    trust_anchor = MappingProxyType(
        {
            str(key): _freeze(value)
            for key, value in _object("trust_anchor", document.get("trust_anchor", {})).items()
        }
    )
    claim_boundary = str(document.get("claim_boundary"))
    if claim_boundary != CLAIM_BOUNDARY:
        raise Phase4D1ProtocolBEligibilityError("claim_boundary", "mismatch")
    if not synthetic_fixture:
        if source_count != REAL_SOURCE_COUNT or class_count != REAL_CLASS_COUNT:
            raise Phase4D1ProtocolBEligibilityError("denominators", "real count mismatch")
        if model_count != REAL_MODEL_CELL_COUNT or role_count != REAL_ROLE_OCCURRENCE_COUNT:
            raise Phase4D1ProtocolBEligibilityError("denominators", "real role count mismatch")
        if shard_source_count != REAL_SHARD_SOURCE_COUNT or shard_count != 66:
            raise Phase4D1ProtocolBEligibilityError("condition_matrix_store", "real shard contract mismatch")
        if support_grid.get("coordinates_authority_path") != D1_PROTOCOL_A_CONFIG_RELATIVE_PATH:
            raise Phase4D1ProtocolBEligibilityError(
                "support_grid.coordinates_authority_path",
                "mismatch",
            )
        if _json_ready(authorities) != _authority_document():
            raise Phase4D1ProtocolBEligibilityError("authorities", "mismatch")
        _validate_real_authority_receipts()
        if tuple(code_authority) != CODE_RELATIVE_PATHS or _json_ready(code_authority) != _code_document():
            raise Phase4D1ProtocolBEligibilityError("code_authority", "mismatch")
        if _json_ready(environment_authority) != _environment_document():
            raise Phase4D1ProtocolBEligibilityError("environment_authority", "mismatch")
        if _json_ready(inherited) != _json_ready(REAL_INHERITED_RULINGS):
            raise Phase4D1ProtocolBEligibilityError("inherited_rulings", "mismatch")
        required_frozen = {
            "model_ledger_sha256": MODEL_LEDGER_SHA256,
            "model_seed_ids": list(MODEL_SEEDS),
            "native_axis_increasing_f64_sha256": NATIVE_F64_SHA256,
            "native_axis_point_count": 1000,
            "role_ledger_sha256": ROLE_LEDGER_SHA256,
            "source_ledger_sha256": SOURCE_LEDGER_SHA256,
            "support_axis_f32_sha256": SUPPORT_F32_SHA256,
            "support_axis_f64_sha256": SUPPORT_F64_SHA256,
            "support_point_count": REAL_SUPPORT_POINT_COUNT,
            "test_record_ids_sha256": TEST_IDS_SHA256,
        }
        if require_frozen_identity and _json_ready(frozen) != required_frozen:
            raise Phase4D1ProtocolBEligibilityError("frozen_identities", "mismatch")
        if _json_ready(trust_anchor) != {
            "config_authority_relative_path": CONFIG_AUTHORITY_RELATIVE_PATH,
            "config_binds_authority": False,
            "direction": "authority_to_config_only",
        }:
            raise Phase4D1ProtocolBEligibilityError("trust_anchor", "mismatch")
        expected_exact = {
            "operator_cell_count": source_count * len(ACTIVE_PERTURBATION_IDS),
            "apply_check_count": source_count * len(ACTIVE_PERTURBATION_IDS) * len(ALPHA_GRID),
            "positive_record_condition_count": source_count * len(ACTIVE_PERTURBATION_IDS) * len(POSITIVE_ALPHAS),
            "record_condition_count": source_count * CONDITION_COUNT,
            "role_condition_audit_count": role_count * CONDITION_COUNT,
            "class_summary_count": class_count * len(ACTIVE_PERTURBATION_IDS),
            "numerical_value_count": source_count * CONDITION_COUNT * support_point_count,
            "numerical_byte_count": source_count * CONDITION_COUNT * support_point_count * 4,
        }
        observed_exact = {
            "operator_cell_count": operator_count,
            "apply_check_count": apply_check_count,
            "positive_record_condition_count": positive_condition_count,
            "record_condition_count": condition_count,
            "role_condition_audit_count": role_condition_count,
            "class_summary_count": class_summary_count,
            "numerical_value_count": numerical_value_count,
            "numerical_byte_count": numerical_byte_count,
        }
        if observed_exact != expected_exact:
            raise Phase4D1ProtocolBEligibilityError("expected", "real workload mismatch")
        declared_artifact_counts = {
            "artifact_file_count": _int("expected.artifact_file_count", expected.get("artifact_file_count"), minimum=1),
            "checksum_entry_count": _int("expected.checksum_entry_count", expected.get("checksum_entry_count"), minimum=1),
            "configured_payload_count": _int("expected.configured_payload_count", expected.get("configured_payload_count"), minimum=1),
            "condition_matrix_shard_count": _int("expected.condition_matrix_shard_count", expected.get("condition_matrix_shard_count"), minimum=1),
            "terminal_marker_count": _int("expected.terminal_marker_count", expected.get("terminal_marker_count"), minimum=1),
        }
        expected_artifact_counts = {
            "artifact_file_count": len(payload_files) + 2,
            "checksum_entry_count": len(payload_files) + 1,
            "configured_payload_count": len(payload_files),
            "condition_matrix_shard_count": len(shard_files),
            "terminal_marker_count": 1,
        }
        if declared_artifact_counts != expected_artifact_counts:
            raise Phase4D1ProtocolBEligibilityError("expected", "artifact inventory mismatch")
        if (
            storage.get("condition_count") != CONDITION_COUNT
            or storage.get("dtype") != "little_endian_float32"
            or storage.get("layout") != "condition_major_source_major_support"
            or storage.get("bytes_per_full_shard") != CONDITION_COUNT * shard_source_count * support_point_count * 4
        ):
            raise Phase4D1ProtocolBEligibilityError("condition_matrix_store", "layout mismatch")
        if (
            p10.get("per_job_estimated_peak_bytes") != estimate_p10_peak_bytes(1000)
            or p10.get("max_admitted_workers") != P10_MEMORY_BUDGET_BYTES // estimate_p10_peak_bytes(1000)
            or p10.get("default_production_workers") != 16
            or p10.get("default_verifier_workers") != 12
        ):
            raise Phase4D1ProtocolBEligibilityError("p10", "worker/admission contract mismatch")
    elif authorities or code_authority or environment_authority or trust_anchor:
        raise Phase4D1ProtocolBEligibilityError(
            "synthetic authority",
            "synthetic fixtures may only use empty authority mappings",
        )
    return Phase4D1ProtocolBEligibilityConfig(
        path=Path(path),
        raw_bytes=raw,
        sha256=_sha_bytes(raw),
        document=_freeze(document),
        synthetic_fixture=synthetic_fixture,
        endpoint_count=endpoint_count,
        model_seeds=model_seeds,
        active_perturbation_ids=active,
        alpha_grid=alpha_grid,
        artifact_payload_files=payload_files,
        condition_matrix_shard_files=shard_files,
        source_record_count=source_count,
        model_cell_count=model_count,
        model_role_occurrence_count=role_count,
        class_count=class_count,
        expected_operator_cell_count=operator_count,
        expected_apply_check_count=apply_check_count,
        expected_positive_record_condition_count=positive_condition_count,
        expected_record_condition_count=condition_count,
        expected_role_condition_audit_count=role_condition_count,
        expected_class_summary_count=class_summary_count,
        expected_numerical_value_count=numerical_value_count,
        expected_numerical_byte_count=numerical_byte_count,
        support_coordinates_cm1=support_coordinates,
        support_point_count=support_point_count,
        support_max_gap_cm1=support_max_gap,
        shard_source_count=shard_source_count,
        condition_matrix_shard_count=shard_count,
        p10_memory_budget_bytes=memory_budget,
        native_gate_relative_tolerance=native_tolerance,
        authorities=authorities,
        frozen_identities=frozen,
        inherited_rulings=inherited,
        code_authority=code_authority,
        environment_authority=environment_authority,
        claim_boundary=claim_boundary,
        trust_anchor=trust_anchor,
    )


def load_phase4_d1_protocol_b_eligibility_config(
    path: Path,
) -> Phase4D1ProtocolBEligibilityConfig:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise Phase4D1ProtocolBEligibilityError(
            "frozen config identity",
            f"config unavailable or untrusted: {error}",
        ) from error
    return parse_phase4_d1_protocol_b_eligibility_config(
        path,
        raw,
        require_frozen_identity=True,
    )


def _synthetic_document(
    *,
    class_count: int,
    source_record_count: int,
    role_occurrence_count: int,
    shard_source_count: int,
    support_coordinates: Sequence[float],
) -> dict[str, object]:
    shard_count = math.ceil(source_record_count / shard_source_count)
    shard_files = _condition_matrix_shard_files(
        source_record_count,
        shard_source_count,
    )
    frozen = {
        "model_seed_ids": list(MODEL_SEEDS),
        "support_axis_f32_sha256": _array_sha(np.asarray(support_coordinates, dtype="<f8"), "<f4"),
        "support_axis_f64_sha256": _array_sha(np.asarray(support_coordinates, dtype="<f8"), "<f8"),
        "support_point_count": len(support_coordinates),
    }
    return {
        "active_perturbation_ids": list(ACTIVE_PERTURBATION_IDS),
        "alpha_grid": list(ALPHA_GRID),
        "artifact_payload_files": list(_complete_artifact_order(shard_files)),
        "authorities": {},
        "claim_boundary": CLAIM_BOUNDARY,
        "code_authority": {},
        "condition_matrix_store": {
            "condition_count": CONDITION_COUNT,
            "dtype": "little_endian_float32",
            "layout": "condition_major_source_major_support",
            "shard_count": shard_count,
            "shard_files": list(shard_files),
            "shard_source_count": shard_source_count,
        },
        "denominators": {
            "class_count": class_count,
            "endpoint_count": 1,
            "model_cell_count": len(MODEL_SEEDS),
            "model_role_occurrence_count": role_occurrence_count,
            "source_record_count": source_record_count,
        },
        "endpoint_id": ENDPOINT_ID,
        "expected": {
            "apply_check_count": source_record_count * len(ACTIVE_PERTURBATION_IDS) * len(ALPHA_GRID),
            "class_summary_count": class_count * len(ACTIVE_PERTURBATION_IDS),
            "numerical_byte_count": source_record_count * CONDITION_COUNT * len(support_coordinates) * 4,
            "numerical_value_count": source_record_count * CONDITION_COUNT * len(support_coordinates),
            "operator_cell_count": source_record_count * len(ACTIVE_PERTURBATION_IDS),
            "positive_record_condition_count": source_record_count * len(ACTIVE_PERTURBATION_IDS) * len(POSITIVE_ALPHAS),
            "record_condition_count": source_record_count * CONDITION_COUNT,
            "role_condition_audit_count": role_occurrence_count * CONDITION_COUNT,
        },
        "environment_authority": {},
        "experiment_id": EXPERIMENT_ID,
        "frozen_identities": frozen,
        "inherited_rulings": _json_ready(REAL_INHERITED_RULINGS),
        "model_seeds": list(MODEL_SEEDS),
        "p10": {
            "correlation_length_cm1": 20.0,
            "memory_budget_bytes": P10_MEMORY_BUDGET_BYTES,
            "peak_estimate_formula": "32*N^2+64*N+2^30",
        },
        "phase1_native_gate_relative_tolerance": 1e-12,
        "protocol": PROTOCOL,
        "schema_version": SCHEMA_VERSION,
        "support_grid": {
            "coordinates_cm1": list(support_coordinates),
            "max_in_range_native_gap_cm1": 2.0,
            "point_count": len(support_coordinates),
        },
        "synthetic_fixture": True,
        "trust_anchor": {},
    }


def make_synthetic_d1_protocol_b_config(
    inputs: D1ProtocolBEligibilityInputs,
) -> Phase4D1ProtocolBEligibilityConfig:
    support = tuple(float(value) for value in inputs.source_spectra[0].axis_cm1[: inputs.support_axis_point_count])
    document = _synthetic_document(
        class_count=len({int(row["class_label"]) for row in inputs.source_records}),
        source_record_count=inputs.source_record_count,
        role_occurrence_count=inputs.role_occurrence_count,
        shard_source_count=int(inputs.source_records[0].get("synthetic_shard_source_count", 2)),
        support_coordinates=support,
    )
    raw = _canonical_json_bytes(document)
    return parse_phase4_d1_protocol_b_eligibility_config(
        Path("<synthetic-d1-protocol-b-config.json>"),
        raw,
        require_frozen_identity=False,
    )


def _project_support(
    spectrum: Spectrum1D,
    config: Phase4D1ProtocolBEligibilityConfig,
) -> np.ndarray:
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    intensity = np.asarray(spectrum.intensity, dtype="<f8")
    support = np.asarray(config.support_coordinates_cm1, dtype="<f8")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.size != intensity.size:
        raise Phase4D1ProtocolBEligibilityError("support_projection", "axis/intensity mismatch")
    if not np.all(np.diff(axis) > 0.0):
        raise Phase4D1ProtocolBEligibilityError("support_projection", "axis must increase")
    if float(axis[0]) > float(support[0]) or float(axis[-1]) < float(support[-1]):
        raise Phase4D1ProtocolBEligibilityError("support_projection", "extrapolation required")
    left = int(np.searchsorted(axis, support[0], side="left"))
    right = int(np.searchsorted(axis, support[-1], side="right"))
    native_in_range = axis[left:right]
    if native_in_range.size < 2:
        raise Phase4D1ProtocolBEligibilityError("support_projection", "insufficient native support")
    if float(np.max(np.diff(native_in_range))) > config.support_max_gap_cm1:
        raise Phase4D1ProtocolBEligibilityError("support_projection", "native gap exceeds gate")
    projected = np.interp(support, axis, intensity)
    output = np.ascontiguousarray(projected, dtype="<f4")
    if output.size != config.support_point_count or not np.isfinite(output).all():
        raise Phase4D1ProtocolBEligibilityError("support_projection", "invalid projected row")
    if float(np.linalg.norm(output.astype(np.float64))) <= 0.0:
        raise Phase4D1ProtocolBEligibilityError("support_projection", "zero norm")
    return output


def make_synthetic_d1_protocol_b_inputs(
    *,
    class_count: int,
    records_per_class: int,
    model_seeds: Sequence[int],
    shard_source_count: int,
    native_point_count: int,
    support_point_count: int,
) -> D1ProtocolBEligibilityInputs:
    if tuple(model_seeds) != MODEL_SEEDS:
        raise Phase4D1ProtocolBEligibilityError("model_seeds", "synthetic fixture requires frozen seeds")
    axis = np.linspace(100.0, 200.0, native_point_count, dtype="<f8")
    support = np.ascontiguousarray(axis[:support_point_count], dtype="<f8")
    source_records: list[Mapping[str, object]] = []
    spectra: list[Spectrum1D] = []
    projections: list[np.ndarray] = []
    for class_label in range(class_count):
        for class_order in range(records_per_class):
            record_order = len(source_records)
            record_id = f"synthetic-c{class_label:02d}-r{class_order:03d}"
            intensity = np.ascontiguousarray(
                1.0 + class_label + class_order / 10.0 + np.linspace(0.0, 0.5, native_point_count),
                dtype="<f8",
            )
            spectrum = Spectrum1D(
                spectrum_id=f"synthetic::{record_id}",
                sample_id=None,
                axis_cm1=axis,
                intensity=intensity,
            )
            projected = np.ascontiguousarray(intensity[:support_point_count], dtype="<f4")
            row = {
                "class_label": class_label,
                "native_axis_sha256": _array_sha(axis, "<f8"),
                "native_intensity_sha256": _array_sha(intensity, "<f8"),
                "record_id": record_id,
                "record_order": record_order,
                "scope": "synthetic",
                "source_row": record_order,
                "source_split": "synthetic",
                "support_projection_sha256": _array_sha(projected, "<f4"),
                "synthetic_shard_source_count": shard_source_count,
            }
            source_records.append(MappingProxyType(row))
            spectra.append(spectrum)
            projections.append(projected)
    model_cells = tuple(
        MappingProxyType(
            {
                "endpoint_id": ENDPOINT_ID,
                "model_seed": int(seed),
                "test_count": len(source_records),
                "train_count": len(source_records),
                "validation_count": len(source_records),
            }
        )
        for seed in MODEL_SEEDS
    )
    role_rows: list[Mapping[str, object]] = []
    for seed in MODEL_SEEDS:
        for role in ("train", "validation", "test"):
            for row in source_records:
                role_rows.append(
                    {
                        "class_label": int(row["class_label"]),
                        "model_seed": seed,
                        "record_id": str(row["record_id"]),
                        "role": role,
                        "role_order": int(row["record_order"]),
                        "source_row": int(row["source_row"]),
                        "source_split": str(row["source_split"]),
                    }
                )
    return D1ProtocolBEligibilityInputs(
        source_records=tuple(source_records),
        model_cells=model_cells,
        model_role_occurrences=tuple(MappingProxyType(row) for row in role_rows),
        source_spectra=tuple(spectra),
        source_projections=tuple(projections),
        source_record_count=len(source_records),
        model_cell_count=len(model_cells),
        role_occurrence_count=len(role_rows),
        source_ledger_sha256=_canonical_ledger_digest(source_records),
        model_ledger_sha256=_canonical_ledger_digest(model_cells),
        role_ledger_sha256=_canonical_ledger_digest(role_rows),
        test_record_ids_sha256=_ids_digest([str(row["record_id"]) for row in source_records]),
        support_axis_f64_sha256=_array_sha(support, "<f8"),
        support_axis_f32_sha256=_array_sha(support, "<f4"),
        native_axis_f64_sha256=_array_sha(axis, "<f8"),
        native_axis_point_count=native_point_count,
        support_axis_point_count=support_point_count,
        role_overlap_detected=False,
    )


def make_synthetic_d1_protocol_b_inputs_with_one_failed_cell(
    inputs: D1ProtocolBEligibilityInputs,
) -> tuple[Mapping[str, object], ...]:
    cells = list(_synthetic_operator_cells(inputs))
    first = dict(cells[0])
    first["state"] = "failed_runtime"
    first["exception"] = {"message": "synthetic failure", "path": "synthetic", "type": "SyntheticFailure"}
    first["outputs"] = []
    first["output_count"] = 0
    cells[0] = MappingProxyType(first)
    return tuple(cells)


def _split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for label in range(REAL_CLASS_COUNT):
        indices = np.flatnonzero(labels == label)
        if len(indices) != 100:
            raise Phase4D1ProtocolBEligibilityError(
                "D1 inputs",
                f"finetune class {label} must contain 100 records",
            )
        shuffled = generator.permutation(indices)
        validation.extend(shuffled[:10].tolist())
        train.extend(shuffled[10:].tolist())
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def reconstruct_d1_protocol_b_inputs(
    dataset_path: Path,
    config: Phase4D1ProtocolBEligibilityConfig,
) -> D1ProtocolBEligibilityInputs:
    if config.synthetic_fixture:
        raise Phase4D1ProtocolBEligibilityError("D1 inputs", "use synthetic constructors for synthetic configs")
    dataset_path = Path(dataset_path)
    if (
        dataset_path.name.startswith("seed")
        or "prediction" in dataset_path.name
        or "selected_c" in dataset_path.name
        or dataset_path.name in {"complete_cell.json", "complete_cells.json"}
    ):
        raise Phase4D1ProtocolBEligibilityError(str(dataset_path), "forbidden outcome artifact")
    if not dataset_path.is_dir():
        raise Phase4D1ProtocolBEligibilityError(str(dataset_path), "retained dataset directory required")
    support = np.asarray(config.support_coordinates_cm1, dtype="<f8")
    if _array_sha(support, "<f8") != str(config.frozen_identities.get("support_axis_f64_sha256", SUPPORT_F64_SHA256)):
        raise Phase4D1ProtocolBEligibilityError("support_axis_f64_sha256", "support-axis digest mismatch")
    if _array_sha(support, "<f4") != str(config.frozen_identities.get("support_axis_f32_sha256", SUPPORT_F32_SHA256)):
        raise Phase4D1ProtocolBEligibilityError("support_axis_f32_sha256", "support-axis digest mismatch")

    split_records: dict[str, list[Mapping[str, object]]] = {"finetune": [], "reference": [], "test": []}
    split_projections: dict[str, list[np.ndarray]] = {"finetune": [], "reference": [], "test": []}
    split_native_f32: dict[str, list[np.ndarray]] = {"finetune": [], "reference": [], "test": []}
    split_labels: dict[str, list[int]] = {"finetune": [], "reference": [], "test": []}
    split_ids: dict[str, list[str]] = {"finetune": [], "reference": [], "test": []}
    split_rows: dict[str, list[int]] = {"finetune": [], "reference": [], "test": []}
    source_spectra: list[Spectrum1D] = []
    source_projections: list[np.ndarray] = []
    native_axis: np.ndarray | None = None
    with BacteriaIdBatchLoader(dataset_path, batch_size=4096) as loader:
        for batch in loader.iter_batches():
            split = str(batch.source_split)
            if split not in split_records:
                raise Phase4D1ProtocolBEligibilityError("BacteriaIdBatchLoader", "unexpected split")
            stored_axis = np.asarray(batch.wavenumber, dtype="<f4")
            axis = np.ascontiguousarray(stored_axis[::-1], dtype="<f8")
            if not np.all(np.diff(axis) > 0.0):
                raise Phase4D1ProtocolBEligibilityError("D1 inputs", "native axis reversal failed")
            if native_axis is None:
                native_axis = axis
            elif not np.array_equal(native_axis, axis):
                raise Phase4D1ProtocolBEligibilityError("D1 inputs", "shared native axis mismatch")
            for record_id, label, source_row, intensity in zip(
                batch.record_ids,
                batch.class_labels,
                batch.source_rows,
                batch.intensity,
                strict=True,
            ):
                record_id = str(record_id)
                native_intensity = np.ascontiguousarray(np.asarray(intensity, dtype="<f4")[::-1], dtype="<f8")
                spectrum = Spectrum1D(
                    spectrum_id=f"bacteria_id_reference::{record_id}",
                    sample_id=None,
                    axis_cm1=axis,
                    intensity=native_intensity,
                )
                projected = _project_support(spectrum, config)
                row = {
                    "class_label": int(label),
                    "native_axis_sha256": _array_sha(axis, "<f8"),
                    "native_intensity_sha256": _array_sha(native_intensity, "<f8"),
                    "record_id": record_id,
                    "record_order": 0,
                    "scope": split,
                    "source_row": int(source_row),
                    "source_split": split,
                    "support_projection_sha256": _array_sha(projected, "<f4"),
                }
                split_records[split].append(row)
                split_projections[split].append(projected)
                split_native_f32[split].append(np.ascontiguousarray(intensity, dtype="<f4"))
                split_labels[split].append(int(label))
                split_ids[split].append(record_id)
                split_rows[split].append(int(source_row))
    if native_axis is None or _array_sha(native_axis, "<f8") != NATIVE_F64_SHA256:
        raise Phase4D1ProtocolBEligibilityError("native_axis_increasing_f64_sha256", "native axis mismatch")
    expected_counts = {"finetune": 3000, "reference": 60000, "test": 3000}
    if {key: len(value) for key, value in split_records.items()} != expected_counts:
        raise Phase4D1ProtocolBEligibilityError("source_records", "source split cardinality mismatch")
    if _ids_digest(split_ids["test"]) != TEST_IDS_SHA256:
        raise Phase4D1ProtocolBEligibilityError("test_record_ids_sha256", "test identity mismatch")

    source_records: list[Mapping[str, object]] = []
    for split in ("finetune", "reference", "test"):
        for row, spectrum, projected in zip(
            split_records[split],
            [
                Spectrum1D(
                    spectrum_id=f"bacteria_id_reference::{record_id}",
                    sample_id=None,
                    axis_cm1=native_axis,
                    intensity=np.asarray(native_f32[::-1], dtype="<f8"),
                )
                for record_id, native_f32 in zip(split_ids[split], split_native_f32[split], strict=True)
            ],
            split_projections[split],
            strict=True,
        ):
            updated = {**row, "record_order": len(source_records)}
            source_records.append(MappingProxyType(updated))
            source_spectra.append(spectrum)
            source_projections.append(projected)

    source_digest = _canonical_ledger_digest(
        [
            {
                "class_label": int(row["class_label"]),
                "record_id": str(row["record_id"]),
                "source_row": int(row["source_row"]),
                "source_split": str(row["source_split"]),
            }
            for row in source_records
        ]
    )
    model_rows: list[Mapping[str, object]] = []
    role_rows: list[Mapping[str, object]] = []
    role_overlap = False
    reference_labels = np.asarray(split_labels["reference"], dtype="<i8")
    finetune_labels = np.asarray(split_labels["finetune"], dtype="<i8")
    test_labels = np.asarray(split_labels["test"], dtype="<i8")
    reference_native = np.asarray(split_native_f32["reference"], dtype="<f4")
    finetune_native = np.asarray(split_native_f32["finetune"], dtype="<f4")
    test_native = np.asarray(split_native_f32["test"], dtype="<f4")
    reference_projected = np.asarray(split_projections["reference"], dtype="<f4")
    finetune_projected = np.asarray(split_projections["finetune"], dtype="<f4")
    test_projected = np.asarray(split_projections["test"], dtype="<f4")
    for seed in config.model_seeds:
        train_idx, valid_idx = _split_indices(finetune_labels, seed)
        train_ids = tuple(split_ids["reference"] + [split_ids["finetune"][int(index)] for index in train_idx])
        valid_ids = tuple(split_ids["finetune"][int(index)] for index in valid_idx)
        test_ids = tuple(split_ids["test"])
        if any(set(a) & set(b) for a, b in ((train_ids, valid_ids), (train_ids, test_ids), (valid_ids, test_ids))):
            role_overlap = True
        source_hashes = {
            "train": {_sha_bytes(row.tobytes()) for row in np.concatenate((reference_native, finetune_native[train_idx]))},
            "validation": {_sha_bytes(row.tobytes()) for row in finetune_native[valid_idx]},
            "test": {_sha_bytes(row.tobytes()) for row in test_native},
        }
        projected_hashes = {
            "train": {_sha_bytes(row.tobytes()) for row in np.concatenate((reference_projected, finetune_projected[train_idx]))},
            "validation": {_sha_bytes(row.tobytes()) for row in finetune_projected[valid_idx]},
            "test": {_sha_bytes(row.tobytes()) for row in test_projected},
        }
        if any(
            left & right
            for groups in (source_hashes, projected_hashes)
            for left, right in (
                (groups["train"], groups["validation"]),
                (groups["train"], groups["test"]),
                (groups["validation"], groups["test"]),
            )
        ):
            raise Phase4D1ProtocolBEligibilityError("D1 inputs", "exact cross-role duplicate detected")
        model_rows.append(
            {
                "model_seed": int(seed),
                "test_count": len(test_ids),
                "test_record_ids_sha256": _ids_digest(test_ids),
                "train_count": len(train_ids),
                "train_record_ids_sha256": _ids_digest(train_ids),
                "validation_count": len(valid_ids),
                "validation_record_ids_sha256": _ids_digest(valid_ids),
            }
        )
        role_specs = (
            (
                "train",
                train_ids,
                np.concatenate((reference_labels, finetune_labels[train_idx])),
                tuple(split_rows["reference"]) + tuple(split_rows["finetune"][int(index)] for index in train_idx),
                ("reference",) * 60000 + ("finetune",) * 2700,
            ),
            (
                "validation",
                valid_ids,
                finetune_labels[valid_idx],
                tuple(split_rows["finetune"][int(index)] for index in valid_idx),
                ("finetune",) * 300,
            ),
            ("test", test_ids, test_labels, tuple(split_rows["test"]), ("test",) * 3000),
        )
        for role, ids, labels, rows, splits in role_specs:
            for order, (record_id, label, source_row, split) in enumerate(
                zip(ids, labels, rows, splits, strict=True)
            ):
                role_rows.append(
                    {
                        "class_label": int(label),
                        "model_seed": int(seed),
                        "record_id": str(record_id),
                        "role": str(role),
                        "role_order": int(order),
                        "source_row": int(source_row),
                        "source_split": str(split),
                    }
                )
    model_digest = _canonical_ledger_digest(model_rows)
    role_digest = _canonical_ledger_digest(role_rows)
    if source_digest != SOURCE_LEDGER_SHA256:
        raise Phase4D1ProtocolBEligibilityError("source_ledger_sha256", "frozen source ledger mismatch")
    if model_digest != MODEL_LEDGER_SHA256:
        raise Phase4D1ProtocolBEligibilityError("model_ledger_sha256", "frozen model ledger mismatch")
    if role_digest != ROLE_LEDGER_SHA256 or role_overlap:
        raise Phase4D1ProtocolBEligibilityError("role_ledger_sha256", "frozen role ledger mismatch")
    return D1ProtocolBEligibilityInputs(
        source_records=tuple(source_records),
        model_cells=tuple(MappingProxyType(row) for row in model_rows),
        model_role_occurrences=tuple(MappingProxyType(row) for row in role_rows),
        source_spectra=tuple(source_spectra),
        source_projections=tuple(source_projections),
        source_record_count=len(source_records),
        model_cell_count=len(model_rows),
        role_occurrence_count=len(role_rows),
        source_ledger_sha256=source_digest,
        model_ledger_sha256=model_digest,
        role_ledger_sha256=role_digest,
        test_record_ids_sha256=_ids_digest(split_ids["test"]),
        support_axis_f64_sha256=_array_sha(support, "<f8"),
        support_axis_f32_sha256=_array_sha(support, "<f4"),
        native_axis_f64_sha256=_array_sha(native_axis, "<f8"),
        native_axis_point_count=int(native_axis.size),
        support_axis_point_count=config.support_point_count,
        role_overlap_detected=role_overlap,
    )


def _alpha_hex(alpha: float) -> str:
    return np.float64(alpha).tobytes().hex()


def _condition_id(perturbation_id: str | None, alpha: float) -> str:
    if perturbation_id is None:
        return "alpha0"
    return f"{perturbation_id}:{_alpha_hex(alpha)}"


def _condition_specs() -> tuple[tuple[int, str | None, float], ...]:
    specs: list[tuple[int, str | None, float]] = [(0, None, 0.0)]
    order = 1
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        for alpha in POSITIVE_ALPHAS:
            specs.append((order, perturbation_id, alpha))
            order += 1
    return tuple(specs)


def _phase1_source(record: Mapping[str, object], spectrum: Spectrum1D) -> Phase1Source:
    axis_f4 = np.asarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.asarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=int(record["record_order"]),
            record_id=str(record["record_id"]),
            sample_id=str(record["record_id"]),
            class_label=int(record["class_label"]),
            mineral_name=f"bacteria-{record['class_label']}",
            axis_id=f"native::{record['native_axis_sha256']}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, "<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, "<f4"),
        normalized_axis_float64_sha256=str(record["native_axis_sha256"]),
        normalized_intensity_float64_sha256=str(record["native_intensity_sha256"]),
        provenance=MappingProxyType(
            {"dataset_id": "bacteria_id_reference", "record_id": str(record["record_id"])}
        ),
    )


def _cell_exception(cell) -> dict[str, object] | None:
    evidence = cell.evidence
    if evidence.exception_type is None:
        return None
    return {
        "message": evidence.exception_message or "unspecified",
        "path": evidence.exception_path or "unspecified",
        "type": evidence.exception_type,
    }


def _cell_receipt(
    record: Mapping[str, object],
    cell,
    config: Phase4D1ProtocolBEligibilityConfig,
    *,
    p10_estimate: int | None,
) -> tuple[Mapping[str, object], tuple[np.ndarray, ...]]:
    if cell.status is CellStatus.COMPLETE:
        outputs = []
        projections: list[np.ndarray] = []
        for perturbed in cell.records:
            output = perturbed.result.output
            projection = _project_support(output, config)
            projections.append(projection)
            outputs.append(
                {
                    "alpha": float(perturbed.result.alpha),
                    "alpha_float64_le_hex": perturbed.alpha_float64_le_hex,
                    "axis_changed": bool(perturbed.result.axis_changed),
                    "intensity_changed": bool(perturbed.result.intensity_changed),
                    "output_axis_sha256": _array_sha(output.axis_cm1, "<f8"),
                    "output_intensity_sha256": _array_sha(output.intensity, "<f8"),
                    "output_spectrum_id": output.spectrum_id,
                    "support_max_in_range_gap_cm1": float(
                        np.max(
                            np.diff(
                                output.axis_cm1[
                                    int(np.searchsorted(output.axis_cm1, config.support_coordinates_cm1[0], side="left")):
                                    int(np.searchsorted(output.axis_cm1, config.support_coordinates_cm1[-1], side="right"))
                                ]
                            )
                        )
                    ),
                    "support_point_count": int(projection.size),
                    "support_projection_sha256": _array_sha(projection, "<f4"),
                }
            )
        return (
            {
                "class_label": int(record["class_label"]),
                "exception": None,
                "native_gate": _json_ready(cell.evidence.native_gate),
                "output_count": len(outputs),
                "outputs": tuple(outputs),
                "p10_estimated_peak_bytes": p10_estimate,
                "perturbation_id": cell.perturbation_id,
                "reason_code": None,
                "record_id": str(record["record_id"]),
                "state": "complete",
                "state_digest": None if cell.state is None else cell.state.state_digest,
            },
            tuple(projections),
        )
    state = "not_applicable" if cell.status is CellStatus.NOT_APPLICABLE else "failed_runtime"
    return (
        {
            "class_label": int(record["class_label"]),
            "exception": _cell_exception(cell),
            "native_gate": {},
            "output_count": 0,
            "outputs": (),
            "p10_estimated_peak_bytes": p10_estimate,
            "perturbation_id": cell.perturbation_id,
            "reason_code": cell.reason_code if state == "not_applicable" else None,
            "record_id": str(record["record_id"]),
            "state": state,
            "state_digest": None if cell.state is None else cell.state.state_digest,
        },
        (),
    )


_PROCESS_SWEEP: PerturbationSweepConfig | None = None
_PROCESS_PHASE1_CONFIG: Phase1CoreConfig | None = None
_PROCESS_CONFIG: Phase4D1ProtocolBEligibilityConfig | None = None


def _initialize_process(
    sweep_path: str,
    phase1_config_path: str,
    config_path: str,
    config_raw: bytes,
) -> None:
    global _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG, _PROCESS_CONFIG
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    _PROCESS_SWEEP = load_perturbation_sweep_config(Path(sweep_path))
    _PROCESS_PHASE1_CONFIG = load_phase1_core_config(Path(phase1_config_path))
    _PROCESS_CONFIG = parse_phase4_d1_protocol_b_eligibility_config(
        Path(config_path),
        config_raw,
        require_frozen_identity=False,
    )


def _execute_initialized_record(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
) -> _ExecutedSource:
    if _PROCESS_SWEEP is None or _PROCESS_PHASE1_CONFIG is None or _PROCESS_CONFIG is None:
        raise Phase4D1ProtocolBEligibilityError("worker", "process not initialized")
    source = _phase1_source(record, spectrum)
    admission = P10MemoryAdmission(_PROCESS_CONFIG.p10_memory_budget_bytes)
    receipts: list[Mapping[str, object]] = []
    projections_by_perturbation: dict[str, tuple[np.ndarray, ...]] = {}
    base_projection = _project_support(spectrum, _PROCESS_CONFIG)
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        p10_estimate = estimate_p10_peak_bytes(int(spectrum.axis_cm1.size)) if perturbation_id == "p10" else None
        try:
            with threadpool_limits(limits=1, user_api="blas"):
                cell = run_perturbation_cell(
                    source,
                    perturbation_id,
                    _PROCESS_PHASE1_CONFIG,
                    _PROCESS_SWEEP,
                    p10_admission=admission if perturbation_id == "p10" else None,
                )
            receipt, projections = _cell_receipt(
                record,
                cell,
                _PROCESS_CONFIG,
                p10_estimate=p10_estimate,
            )
            if projections:
                if len(projections) != len(ALPHA_GRID):
                    raise Phase4D1ProtocolBEligibilityError(
                        "operator output",
                        "must contain the complete frozen alpha grid",
                    )
                if not np.array_equal(projections[0], base_projection):
                    raise Phase4D1ProtocolBEligibilityError(
                        "operator alpha zero",
                        "must equal the source projection exactly",
                    )
            receipts.append(receipt)
            projections_by_perturbation[perturbation_id] = projections
        except Exception as error:
            receipts.append(
                {
                        "class_label": int(record["class_label"]),
                        "exception": {
                            "message": str(error),
                            "path": str(getattr(error, "path", "unexpected_exception")),
                            "type": type(error).__name__,
                        },
                        "native_gate": {},
                        "output_count": 0,
                        "outputs": (),
                        "p10_estimated_peak_bytes": p10_estimate,
                        "perturbation_id": perturbation_id,
                        "reason_code": None,
                        "record_id": str(record["record_id"]),
                        "state": "failed_runtime",
                        "state_digest": None,
                }
            )
            projections_by_perturbation[perturbation_id] = ()
    condition_projections: list[np.ndarray | None] = [base_projection]
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        projections = projections_by_perturbation[perturbation_id]
        if projections:
            condition_projections.extend(projections[1:])
        else:
            condition_projections.extend([None] * len(POSITIVE_ALPHAS))
    return _ExecutedSource(
        operator_cells=tuple(receipts),
        condition_projections=tuple(condition_projections),
    )


def _bounded_process_count(
    requested_workers: int,
    point_counts: Sequence[int],
    memory_budget_bytes: int,
) -> int:
    if isinstance(requested_workers, bool) or not isinstance(requested_workers, int) or requested_workers < 1:
        raise Phase4D1ProtocolBEligibilityError("worker_count", "must be a positive integer")
    if not point_counts:
        return 1
    estimates = [estimate_p10_peak_bytes(int(point_count)) for point_count in point_counts]
    peak_estimate = max(estimates)
    if peak_estimate > memory_budget_bytes:
        raise Phase4D1ProtocolBEligibilityError(
            "p10 admission",
            "at least one record exceeds the frozen 64 GiB budget",
        )
    by_memory = max(1, memory_budget_bytes // peak_estimate)
    return max(1, min(int(requested_workers), int(by_memory), len(point_counts)))


def _synthetic_operator_cells(
    inputs: D1ProtocolBEligibilityInputs,
) -> tuple[Mapping[str, object], ...]:
    cells: list[Mapping[str, object]] = []
    for source_record in inputs.source_records:
        for perturbation_id in ACTIVE_PERTURBATION_IDS:
            outputs = []
            for alpha in ALPHA_GRID:
                outputs.append(
                    {
                        "alpha": float(alpha),
                        "alpha_float64_le_hex": _alpha_hex(alpha),
                        "support_projection_sha256": _synthetic_condition_sha(
                            inputs,
                            str(source_record["record_id"]),
                            perturbation_id,
                            float(alpha),
                        ),
                    }
                )
            cells.append(
                MappingProxyType(
                    {
                        "class_label": int(source_record["class_label"]),
                        "exception": None,
                        "output_count": len(outputs),
                        "outputs": tuple(outputs),
                        "p10_estimated_peak_bytes": (
                            estimate_p10_peak_bytes(inputs.native_axis_point_count)
                            if perturbation_id == "p10"
                            else None
                        ),
                        "perturbation_id": perturbation_id,
                        "record_id": str(source_record["record_id"]),
                        "state": "complete",
                        "state_digest": _sha_bytes(
                            f"{source_record['record_id']}:{perturbation_id}".encode("utf-8")
                        ),
                    }
                )
            )
    return tuple(cells)


def _synthetic_executed_source(
    inputs: D1ProtocolBEligibilityInputs,
    source_order: int,
) -> _ExecutedSource:
    source_record = inputs.source_records[source_order]
    record_id = str(source_record["record_id"])
    base = inputs.source_projections[source_order]
    projections: list[np.ndarray | None] = [np.ascontiguousarray(base, dtype="<f4")]
    cells: list[Mapping[str, object]] = []
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        output_rows = []
        for alpha in ALPHA_GRID:
            projection = _synthetic_condition_row(base, perturbation_id, alpha)
            output_axis_sha256 = str(source_record["native_axis_sha256"])
            output_intensity_sha256 = _array_sha(
                _synthetic_condition_row(
                    np.asarray(inputs.source_spectra[source_order].intensity, dtype="<f4"),
                    perturbation_id,
                    alpha,
                ),
                "<f8",
            )
            output_rows.append(
                {
                    "alpha": float(alpha),
                    "alpha_float64_le_hex": _alpha_hex(alpha),
                    "axis_changed": False,
                    "intensity_changed": bool(alpha > 0.0),
                    "output_axis_sha256": output_axis_sha256,
                    "output_intensity_sha256": output_intensity_sha256,
                    "output_spectrum_id": (
                        inputs.source_spectra[source_order].spectrum_id
                        if alpha == 0.0
                        else f"{inputs.source_spectra[source_order].spectrum_id}::{perturbation_id}::{_alpha_hex(alpha)}"
                    ),
                    "support_max_in_range_gap_cm1": 0.0,
                    "support_point_count": int(projection.size),
                    "support_projection_sha256": _array_sha(projection, "<f4"),
                }
            )
            if alpha > 0.0:
                projections.append(projection)
        cells.append(
            {
                    "class_label": int(source_record["class_label"]),
                    "exception": None,
                    "native_gate": {},
                    "output_count": len(output_rows),
                    "outputs": tuple(output_rows),
                    "p10_estimated_peak_bytes": (
                        estimate_p10_peak_bytes(inputs.native_axis_point_count)
                        if perturbation_id == "p10"
                        else None
                    ),
                    "perturbation_id": perturbation_id,
                    "reason_code": None,
                    "record_id": record_id,
                    "state": "complete",
                    "state_digest": _sha_bytes(
                        f"{record_id}:{perturbation_id}".encode("utf-8")
                    ),
                }
        )
    return _ExecutedSource(tuple(cells), tuple(projections))


def _iter_executed_batches(
    inputs: D1ProtocolBEligibilityInputs,
    sweep: PerturbationSweepConfig | None,
    phase1_config: Phase1CoreConfig | None,
    config: Phase4D1ProtocolBEligibilityConfig,
    worker_count: int,
) :
    point_counts = tuple(int(spectrum.axis_cm1.size) for spectrum in inputs.source_spectra)
    process_count = _bounded_process_count(worker_count, point_counts, config.p10_memory_budget_bytes)
    if config.synthetic_fixture:
        for start in range(0, inputs.source_record_count, config.shard_source_count):
            end = min(start + config.shard_source_count, inputs.source_record_count)
            yield start, tuple(
                _synthetic_executed_source(inputs, source_order)
                for source_order in range(start, end)
            )
        return
    if sweep is None or phase1_config is None:
        raise Phase4D1ProtocolBEligibilityError("phase1", "real execution requires sweep and phase1 config")
    if process_count == 1:
        global _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG, _PROCESS_CONFIG
        _PROCESS_SWEEP = sweep
        _PROCESS_PHASE1_CONFIG = phase1_config
        _PROCESS_CONFIG = config
        try:
            for start in range(0, inputs.source_record_count, config.shard_source_count):
                end = min(start + config.shard_source_count, inputs.source_record_count)
                yield start, tuple(
                    _execute_initialized_record(record, spectrum)
                    for record, spectrum in zip(
                        inputs.source_records[start:end],
                        inputs.source_spectra[start:end],
                        strict=True,
                    )
                )
        finally:
            _PROCESS_SWEEP = None
            _PROCESS_PHASE1_CONFIG = None
            _PROCESS_CONFIG = None
        return
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_process,
        initargs=(str(sweep.path), str(phase1_config.path), str(config.path), config.raw_bytes),
    ) as executor:
        for start in range(0, inputs.source_record_count, config.shard_source_count):
            end = min(start + config.shard_source_count, inputs.source_record_count)
            records = tuple(dict(row) for row in inputs.source_records[start:end])
            spectra = inputs.source_spectra[start:end]
            yield start, tuple(
                executor.map(
                    _execute_initialized_record,
                    records,
                    spectra,
                    chunksize=1,
                )
            )


def _synthetic_condition_row(
    base: np.ndarray,
    perturbation_id: str | None,
    alpha: float,
) -> np.ndarray:
    if perturbation_id is None or alpha == 0.0:
        return np.ascontiguousarray(base, dtype="<f4")
    perturbation_index = ACTIVE_PERTURBATION_IDS.index(perturbation_id) + 1
    return np.ascontiguousarray(base.astype(np.float32) + np.float32(alpha * perturbation_index / 100.0), dtype="<f4")


def _synthetic_condition_sha(
    inputs: D1ProtocolBEligibilityInputs,
    record_id: str,
    perturbation_id: str | None,
    alpha: float,
) -> str:
    index = next(
        int(row["record_order"])
        for row in inputs.source_records
        if str(row["record_id"]) == record_id
    )
    return _array_sha(_synthetic_condition_row(inputs.source_projections[index], perturbation_id, alpha), "<f4")


def evaluate_d1_protocol_b_all_role_gates(
    source_records: Sequence[Mapping[str, object]],
    model_cells: Sequence[Mapping[str, object]],
    model_role_occurrences: Sequence[Mapping[str, object]],
    operator_cells: Sequence[Mapping[str, object]],
    config: Phase4D1ProtocolBEligibilityConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(source_records) != config.source_record_count:
        raise Phase4D1ProtocolBEligibilityError("source_records", "denominator mismatch")
    if len(model_cells) != config.model_cell_count:
        raise Phase4D1ProtocolBEligibilityError("model_cells", "denominator mismatch")
    if len(model_role_occurrences) != config.model_role_occurrence_count:
        raise Phase4D1ProtocolBEligibilityError("model_role_occurrences", "denominator mismatch")
    if len(operator_cells) != config.expected_operator_cell_count:
        raise Phase4D1ProtocolBEligibilityError("operator_cells", "count mismatch")
    source_by_id = {str(row["record_id"]): row for row in source_records}
    cell_by_key = {
        (str(row["record_id"]), str(row["perturbation_id"])): row for row in operator_cells
    }
    expected_keys = {
        (record_id, perturbation_id)
        for record_id in source_by_id
        for perturbation_id in ACTIVE_PERTURBATION_IDS
    }
    if set(cell_by_key) != expected_keys:
        raise Phase4D1ProtocolBEligibilityError("operator_cells", "grid keys mismatch")
    for key, cell in cell_by_key.items():
        state = str(cell.get("state"))
        if state not in TERMINAL_STATES:
            raise Phase4D1ProtocolBEligibilityError(
                f"operator_cells.{key}",
                "unknown terminal state",
            )
        outputs_value = cell.get("outputs")
        outputs = () if outputs_value is None else tuple(outputs_value)
        if state == "complete":
            if int(cell.get("output_count", -1)) != len(ALPHA_GRID):
                raise Phase4D1ProtocolBEligibilityError(
                    f"operator_cells.{key}",
                    "complete cell must contain all nine alpha outputs",
                )
            if outputs_value is not None and (
                len(outputs) != len(ALPHA_GRID)
                or tuple(float(_object("operator output", row)["alpha"]) for row in outputs) != ALPHA_GRID
            ):
                raise Phase4D1ProtocolBEligibilityError(
                    f"operator_cells.{key}",
                    "alpha output grid mismatch",
                )
        elif int(cell.get("output_count", 0)) != 0 or outputs:
            raise Phase4D1ProtocolBEligibilityError(
                f"operator_cells.{key}",
                "noncomplete cell must not expose outputs",
            )
    class_to_records: dict[int, list[str]] = {}
    for record_id, source_record in source_by_id.items():
        class_to_records.setdefault(int(source_record["class_label"]), []).append(record_id)
    operators: dict[str, object] = {}
    class_rows: list[dict[str, object]] = []
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        states = Counter(
            str(cell_by_key[(record_id, perturbation_id)]["state"])
            for record_id in source_by_id
        )
        complete_records = {
            record_id
            for record_id in source_by_id
            if str(cell_by_key[(record_id, perturbation_id)]["state"]) == "complete"
        }
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
                }
            )
        state = (
            "evaluable"
            if len(complete_records) == len(source_by_id)
            and complete_classes == len(class_to_records)
            and states.get("failed_runtime", 0) == 0
            and states.get("not_applicable", 0) == 0
            else "not_evaluable_coverage"
        )
        operators[perturbation_id] = {
            "complete_class_count": complete_classes,
            "complete_record_count": len(complete_records),
            "failed_runtime_count": states.get("failed_runtime", 0),
            "not_applicable_count": states.get("not_applicable", 0),
            "required_class_count": len(class_to_records),
            "required_record_count": len(source_by_id),
            "state": state,
            "state_counts": {state_name: states.get(state_name, 0) for state_name in TERMINAL_STATES},
        }
    endpoint_state = (
        "evaluable"
        if all(
            _object(f"operators.{perturbation_id}", operators[perturbation_id])["state"] == "evaluable"
            for perturbation_id in ACTIVE_PERTURBATION_IDS
        )
        else "not_evaluable_coverage"
    )
    overall_status = "pass" if endpoint_state == "evaluable" else "fail"
    gate = {
        ENDPOINT_ID: {
            "class_denominator": len(class_to_records),
            "operators": operators,
            "record_denominator": len(source_by_id),
            "role_occurrence_denominator": len(model_role_occurrences),
            "state": endpoint_state,
        },
        "inherited_rulings": _json_ready(config.inherited_rulings),
        "marker_filename": "complete.json" if overall_status == "pass" else "failed.json",
        "overall_status": overall_status,
        "protocol": PROTOCOL,
    }
    return class_rows, gate


def _condition_receipt(
    *,
    source_record: Mapping[str, object],
    spectrum: Spectrum1D,
    executed: _ExecutedSource,
    condition_order: int,
    perturbation_id: str | None,
    alpha: float,
    filename: str,
    shard_id: int,
    shard_local_source_order: int,
    rows_per_shard: int,
    config: Phase4D1ProtocolBEligibilityConfig,
) -> tuple[dict[str, object], bytes]:
    projection = executed.condition_projections[condition_order]
    state = "complete"
    if perturbation_id is None:
        output_axis_sha256: str | None = str(source_record["native_axis_sha256"])
        output_intensity_sha256: str | None = str(source_record["native_intensity_sha256"])
        output_spectrum_id: str | None = spectrum.spectrum_id
    else:
        perturbation_order = ACTIVE_PERTURBATION_IDS.index(perturbation_id)
        cell = executed.operator_cells[perturbation_order]
        state = str(cell["state"])
        if state == "complete":
            alpha_order = ALPHA_GRID.index(alpha)
            outputs = tuple(cell["outputs"])
            if len(outputs) != len(ALPHA_GRID):
                raise Phase4D1ProtocolBEligibilityError(
                    "operator outputs",
                    "must contain the complete frozen alpha grid",
                )
            detail = _object("operator output", outputs[alpha_order])
            output_axis_sha256 = str(detail["output_axis_sha256"])
            output_intensity_sha256 = str(detail["output_intensity_sha256"])
            output_spectrum_id = str(detail["output_spectrum_id"])
        else:
            output_axis_sha256 = None
            output_intensity_sha256 = None
            output_spectrum_id = None
    if state == "complete":
        if projection is None:
            raise Phase4D1ProtocolBEligibilityError(
                "condition projection",
                "complete condition is missing its numerical row",
            )
        projected = np.ascontiguousarray(projection, dtype="<f4")
        if (
            projected.size != config.support_point_count
            or not np.isfinite(projected).all()
            or float(np.linalg.norm(projected.astype(np.float64))) <= 0.0
        ):
            raise Phase4D1ProtocolBEligibilityError(
                "condition projection",
                "complete condition has an invalid numerical row",
            )
        row_bytes = projected.tobytes(order="C")
        projection_sha256: str | None = _sha_bytes(row_bytes)
        if perturbation_id is None:
            if projection_sha256 != str(source_record["support_projection_sha256"]):
                raise Phase4D1ProtocolBEligibilityError(
                    "condition projection",
                    "alpha-zero projection differs from source receipt",
                )
        else:
            expected_projection_sha256 = str(detail["support_projection_sha256"])
            if projection_sha256 != expected_projection_sha256:
                raise Phase4D1ProtocolBEligibilityError(
                    "condition projection",
                    "projected row differs from operator receipt",
                )
    else:
        row_bytes = bytes(config.support_point_count * 4)
        projection_sha256 = None
    offset = (
        (condition_order * rows_per_shard + shard_local_source_order)
        * config.support_point_count
        * 4
    )
    receipt = {
        "alpha": float(alpha),
        "alpha_float64_le_hex": _alpha_hex(alpha),
        "class_label": int(source_record["class_label"]),
        "condition_id": _condition_id(perturbation_id, alpha),
        "condition_order": condition_order,
        "filename": filename,
        "native_axis_sha256": str(source_record["native_axis_sha256"]),
        "native_intensity_sha256": str(source_record["native_intensity_sha256"]),
        "output_axis_sha256": output_axis_sha256,
        "output_intensity_sha256": output_intensity_sha256,
        "output_spectrum_id": output_spectrum_id,
        "padding_not_scientific_output": state != "complete",
        "perturbation_id": perturbation_id,
        "record_id": str(source_record["record_id"]),
        "record_order": int(source_record["record_order"]),
        "shard_byte_offset": offset,
        "shard_id": shard_id,
        "shard_local_source_order": shard_local_source_order,
        "source_row": int(source_record["source_row"]),
        "source_split": str(source_record["source_split"]),
        "state": state,
        "support_point_count": config.support_point_count,
        "support_projection_sha256": projection_sha256,
    }
    return receipt, row_bytes


def _execute_and_write_condition_shards(
    output_path: Path,
    inputs: D1ProtocolBEligibilityInputs,
    config: Phase4D1ProtocolBEligibilityConfig,
    worker_count: int,
    sweep: PerturbationSweepConfig | None,
    phase1_config: Phase1CoreConfig | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int, int]:
    conditions = _condition_specs()
    gate_cells: list[dict[str, object]] = []
    shard_rows: list[dict[str, object]] = []
    total_bytes = 0
    operator_cell_count = 0
    record_condition_count = 0
    operator_path = output_path / "operator_cells.jsonl"
    condition_path = output_path / "record_conditions.jsonl"
    with operator_path.open("wb") as operator_stream, condition_path.open("wb") as condition_stream:
      for shard_id, (start, executed_sources) in enumerate(
          _iter_executed_batches(inputs, sweep, phase1_config, config, worker_count)
      ):
        end = start + len(executed_sources) - 1
        rows_this_shard = end - start + 1
        filename = config.condition_matrix_shard_files[shard_id]
        digest = hashlib.sha256()
        byte_count = 0
        batch_receipts: list[list[tuple[dict[str, object], bytes]]] = []
        for local_order, executed in enumerate(executed_sources):
            source_order = start + local_order
            source_record = inputs.source_records[source_order]
            spectrum = inputs.source_spectra[source_order]
            if len(executed.operator_cells) != len(ACTIVE_PERTURBATION_IDS):
                raise Phase4D1ProtocolBEligibilityError(
                    "operator_cells",
                    "one source must yield exactly five operator receipts",
                )
            if len(executed.condition_projections) != CONDITION_COUNT:
                raise Phase4D1ProtocolBEligibilityError(
                    "condition projections",
                    "one source must yield exactly 41 condition slots",
                )
            for cell in executed.operator_cells:
                validate_protocol_b_outcome_blind_payload(cell)
                operator_stream.write(_canonical_json_bytes(cell))
                operator_cell_count += 1
                gate_cells.append(
                    {
                        "output_count": int(cell["output_count"]),
                        "record_id": str(cell["record_id"]),
                        "perturbation_id": str(cell["perturbation_id"]),
                        "state": str(cell["state"]),
                    }
                )
            source_receipts: list[tuple[dict[str, object], bytes]] = []
            for condition_order, perturbation_id, alpha in conditions:
                receipt, row_bytes = _condition_receipt(
                    source_record=source_record,
                    spectrum=spectrum,
                    executed=executed,
                    condition_order=condition_order,
                    perturbation_id=perturbation_id,
                    alpha=alpha,
                    filename=filename,
                    shard_id=shard_id,
                    shard_local_source_order=local_order,
                    rows_per_shard=rows_this_shard,
                    config=config,
                )
                validate_protocol_b_outcome_blind_payload(receipt)
                condition_stream.write(_canonical_json_bytes(receipt))
                record_condition_count += 1
                source_receipts.append((receipt, row_bytes))
            batch_receipts.append(source_receipts)
        with (output_path / filename).open("wb") as stream:
            for condition_order in range(CONDITION_COUNT):
                for local_order in range(rows_this_shard):
                    row_bytes = batch_receipts[local_order][condition_order][1]
                    stream.write(row_bytes)
                    digest.update(row_bytes)
                    byte_count += len(row_bytes)
        shard_digest = digest.hexdigest()
        shard_rows.append(
            {
                "byte_count": byte_count,
                "condition_count": CONDITION_COUNT,
                "dtype": "little_endian_float32",
                "end_source_order": end,
                "filename": filename,
                "layout": "condition_major_source_major_support",
                "rows_per_shard": rows_this_shard,
                "sha256": shard_digest,
                "shard_id": shard_id,
                "start_source_order": start,
                "support_point_count": config.support_point_count,
            }
        )
        total_bytes += byte_count
    return gate_cells, shard_rows, total_bytes, operator_cell_count, record_condition_count


def _environment_document() -> dict[str, object]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _run_identity(
    config: Phase4D1ProtocolBEligibilityConfig,
    inputs: D1ProtocolBEligibilityInputs,
) -> tuple[str, dict[str, object]]:
    identity = {
        "claim_boundary": config.claim_boundary,
        "code_authority": _json_ready(config.code_authority),
        "config_sha256": config.sha256,
        "ledgers": {
            "model_ledger_sha256": inputs.model_ledger_sha256,
            "role_ledger_sha256": inputs.role_ledger_sha256,
            "source_ledger_sha256": inputs.source_ledger_sha256,
            "test_record_ids_sha256": inputs.test_record_ids_sha256,
        },
        "native": {
            "f64_sha256": inputs.native_axis_f64_sha256,
            "point_count": inputs.native_axis_point_count,
        },
        "protocol": PROTOCOL,
        "shard_layout": {
            "condition_count": CONDITION_COUNT,
            "dtype": "little_endian_float32",
            "files": list(config.condition_matrix_shard_files),
            "layout": "condition_major_source_major_support",
            "shard_source_count": config.shard_source_count,
        },
        "support": {
            "f32_sha256": inputs.support_axis_f32_sha256,
            "f64_sha256": inputs.support_axis_f64_sha256,
            "point_count": inputs.support_axis_point_count,
        },
    }
    return RUN_PREFIX + _sha_bytes(_canonical_json_bytes(identity)), identity


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _write_sha256sums(
    output_path: Path,
    compact_payloads: Mapping[str, bytes],
    config: Phase4D1ProtocolBEligibilityConfig,
    marker_name: str,
) -> None:
    ordered_names = list(config.artifact_payload_files) + [marker_name]
    lines = []
    for name in ordered_names:
        digest = (
            _sha_bytes(compact_payloads[name])
            if name in compact_payloads
            else _sha_file(output_path / name)
        )
        lines.append(f"{digest}  {name}")
    (output_path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_phase4_d1_protocol_b_eligibility_from_inputs(
    output_dir: Path,
    *,
    inputs: D1ProtocolBEligibilityInputs,
    config: Phase4D1ProtocolBEligibilityConfig,
    worker_count: int,
    sweep: PerturbationSweepConfig | None = None,
    phase1_config: Phase1CoreConfig | None = None,
) -> Phase4D1ProtocolBEligibilitySummary:
    output_path = Path(output_dir)
    if output_path.exists():
        raise Phase4D1ProtocolBEligibilityError(str(output_path), "append-only output target already exists")
    _bounded_process_count(
        worker_count,
        tuple(int(spectrum.axis_cm1.size) for spectrum in inputs.source_spectra),
        config.p10_memory_budget_bytes,
    )
    run_id, run_identity = _run_identity(config, inputs)
    output_path.mkdir(parents=True)
    (output_path / "config.json").write_bytes(config.raw_bytes)
    (output_path / "source_records.jsonl").write_bytes(
        _jsonl_bytes(inputs.source_records)
    )
    (output_path / "model_cells.jsonl").write_bytes(
        _jsonl_bytes(inputs.model_cells)
    )
    (output_path / "model_role_occurrences.jsonl").write_bytes(
        _jsonl_bytes(inputs.model_role_occurrences)
    )
    (
        operator_cells,
        shard_rows,
        matrix_bytes,
        operator_cell_count,
        record_condition_count,
    ) = _execute_and_write_condition_shards(
        output_path,
        inputs,
        config,
        worker_count,
        sweep,
        phase1_config,
    )
    if operator_cell_count != config.expected_operator_cell_count:
        raise Phase4D1ProtocolBEligibilityError("operator_cells", "count mismatch")
    if record_condition_count != config.expected_record_condition_count:
        raise Phase4D1ProtocolBEligibilityError("record_conditions", "count mismatch")
    if tuple(str(row["filename"]) for row in shard_rows) != config.condition_matrix_shard_files:
        raise Phase4D1ProtocolBEligibilityError("condition_matrix_shards", "inventory mismatch")
    if len(shard_rows) != config.condition_matrix_shard_count:
        raise Phase4D1ProtocolBEligibilityError("condition_matrix_shards", "count mismatch")
    if matrix_bytes != config.expected_numerical_byte_count:
        raise Phase4D1ProtocolBEligibilityError("condition_matrix_store", "total byte count mismatch")
    class_summaries, gate = evaluate_d1_protocol_b_all_role_gates(
        inputs.source_records,
        inputs.model_cells,
        inputs.model_role_occurrences,
        operator_cells,
        config,
    )
    if len(class_summaries) != config.expected_class_summary_count:
        raise Phase4D1ProtocolBEligibilityError("class_summaries", "count mismatch")
    marker_name = str(gate["marker_filename"])
    manifest = {
        "artifact_order": list(config.artifact_payload_files),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "authorities": _json_ready(config.authorities),
        "claim_boundary": config.claim_boundary,
        "code_authority": _json_ready(config.code_authority),
        "condition_matrix_bytes": matrix_bytes,
        "condition_matrix_shard_count": len(shard_rows),
        "counts": {
            "class_summaries": len(class_summaries),
            "model_cells": len(inputs.model_cells),
            "model_role_occurrences": len(inputs.model_role_occurrences),
            "operator_cells": operator_cell_count,
            "record_conditions": record_condition_count,
            "source_records": len(inputs.source_records),
        },
        "endpoint_count": config.endpoint_count,
        "environment_authority": _json_ready(config.environment_authority),
        "protocol": PROTOCOL,
        "run_id": run_id,
        "run_identity": run_identity,
        "status": gate["overall_status"],
        "synthetic_fixture": config.synthetic_fixture,
        "trust_anchor": _json_ready(config.trust_anchor),
    }
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": gate["overall_status"],
    }
    compact_payloads = {
        "class_summaries.jsonl": _jsonl_bytes(class_summaries),
        "condition_matrix_shards.jsonl": _jsonl_bytes(shard_rows),
        "gate.json": _canonical_json_bytes(gate),
        "manifest.json": _canonical_json_bytes(manifest),
        marker_name: _canonical_json_bytes(marker),
    }
    for payload in (
        config.document,
        inputs.source_records,
        inputs.model_cells,
        inputs.model_role_occurrences,
        operator_cells,
        class_summaries,
        shard_rows,
        gate,
        manifest,
        marker,
    ):
        validate_protocol_b_outcome_blind_payload(payload)
    for name, payload in compact_payloads.items():
        (output_path / name).write_bytes(payload)
    (output_path / marker_name).write_bytes(compact_payloads[marker_name])
    _write_sha256sums(output_path, compact_payloads, config, marker_name)
    return Phase4D1ProtocolBEligibilitySummary(
        path=output_path,
        run_id=run_id,
        status=str(gate["overall_status"]),
        endpoint_count=config.endpoint_count,
        model_seed_count=len(config.model_seeds),
        source_record_count=len(inputs.source_records),
        model_cell_count=len(inputs.model_cells),
        role_occurrence_count=len(inputs.model_role_occurrences),
        operator_cell_count=operator_cell_count,
        record_condition_count=record_condition_count,
        class_summary_count=len(class_summaries),
        condition_matrix_shard_count=len(shard_rows),
        condition_matrix_bytes=matrix_bytes,
    )


def build_phase4_d1_protocol_b_eligibility(
    output_root: Path,
    *,
    worker_count: int = 16,
) -> Phase4D1ProtocolBEligibilitySummary:
    config = load_phase4_d1_protocol_b_eligibility_config(ROOT / CONFIG_RELATIVE_PATH)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1_config = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    inputs = reconstruct_d1_protocol_b_inputs(ROOT / DATASET_RELATIVE_PATH, config)
    run_id, _ = _run_identity(config, inputs)
    return build_phase4_d1_protocol_b_eligibility_from_inputs(
        Path(output_root) / run_id,
        inputs=inputs,
        config=config,
        worker_count=worker_count,
        sweep=sweep,
        phase1_config=phase1_config,
    )


__all__ = [
    "ACTIVE_PERTURBATION_IDS",
    "ARTIFACT_PAYLOAD_FILES",
    "D1ProtocolBEligibilityInputs",
    "Phase4D1ProtocolBEligibilityConfig",
    "Phase4D1ProtocolBEligibilityError",
    "Phase4D1ProtocolBEligibilitySummary",
    "build_phase4_d1_protocol_b_eligibility",
    "build_phase4_d1_protocol_b_eligibility_from_inputs",
    "evaluate_d1_protocol_b_all_role_gates",
    "load_phase4_d1_protocol_b_eligibility_config",
    "make_synthetic_d1_protocol_b_config",
    "make_synthetic_d1_protocol_b_inputs",
    "make_synthetic_d1_protocol_b_inputs_with_one_failed_cell",
    "parse_phase4_d1_protocol_b_eligibility_config",
    "reconstruct_d1_protocol_b_inputs",
    "validate_protocol_b_outcome_blind_payload",
]
