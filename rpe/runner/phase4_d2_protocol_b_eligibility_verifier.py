from __future__ import annotations

import ast
import hashlib
import json
import math
import multiprocessing
import os
import platform
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import h5py
import numpy as np
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.evaluation import Spectrum1D
from rpe.perturb import PerturbationSweepConfig, load_perturbation_sweep_config
from rpe.runner.d2_selection import (
    D2SelectionValidationError,
    validate_d2_few_shot_selection,
)
from rpe.runner.phase1_config import Phase1CoreConfig, load_phase1_core_config
from rpe.runner.phase1_perturbations import (
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-d2-protocol-b-all-role-eligibility-config-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d2-protocol-b-all-role-eligibility-artifact-v1"
EXPERIMENT_ID = "phase4-d2-protocol-b-all-role-eligibility-v1"
RUN_PREFIX = "phase4-d2-protocol-b-all-role-eligibility-"
PROTOCOL = "B"
CONFIG_RELATIVE_PATH = (
    "experiments/phase4/configs/d2_protocol_b_all_role_eligibility_v1.json"
)
STEP13_CONFIG_RELATIVE_PATH = (
    "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
)
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
SELECTION_RELATIVE_PATH = (
    "results/phase05/d2/"
    "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/"
    "selection.json"
)
DATASET_RELATIVE_PATH = "data/unified/bacteria_id_reference"
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d2_protocol_b_eligibility_authority.py"

ACTIVE_PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "source_records.jsonl",
    "model_cells.jsonl",
    "model_role_occurrences.jsonl",
    "operator_cells.jsonl",
    "record_conditions.jsonl",
    "class_summaries.jsonl",
    "gate.json",
    "manifest.json",
)
TERMINAL_STATES = ("complete", "not_applicable", "failed_runtime")
FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "accuracy",
        "alignment",
        "alignment_gap",
        "bootstrap",
        "correct",
        "figure",
        "inference",
        "metric",
        "metric_value",
        "outcome",
        "prediction",
        "predictions",
        "score",
        "selected_c",
        "table",
    }
)
FORBIDDEN_KEY_FRAGMENTS = (
    "accuracy",
    "alignment",
    "bootstrap",
    "figure",
    "inference",
    "metric",
    "outcome",
    "predict",
    "score",
    "selected_c",
    "table",
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
    "rpe/runner/d2_selection.py",
    "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_gates.py",
    "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py",
    "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d2_protocol_b_eligibility.py",
    "rpe/runner/phase4_d2_protocol_b_eligibility_verifier.py",
    "tools/run_phase4_d2_protocol_b_eligibility.py",
)
REQUIRED_AUTHORITIES = {
    "bacteria_id_retained_snapshot_sha256": "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02",
    "bacteria_id_sha256sums_sha256": "6d6d1399ac0e5197a9e51a1a0cf32edf51c924ce8d7c12aeb9d58f9af93c7f3e",
    "d1_model_recipe_sha256": "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046",
    "d2_phase05_runner_config_sha256": "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7",
    "d2_selection_artifact_sha256": "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36",
    "d2_selection_config_sha256": "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138",
    "parent_plan_sha256": "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5",
    "phase1_core_config_sha256": "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd",
    "phase4_step01_sha256": "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f",
    "phase4_step12_design_sha256": "eeb2350ce6c359d84d9f1372962b8fcfb7fef7f623a77af18ae2e0abd03a4b38",
    "step13_config_sha256": "f92427c2f18ab445db2bb54d5ea5a97cce21b05dd6ee80f50f3ed4082285a15b",
    "step13_report_sha256": "ab06d47e2e18301c0c1bfb53aef57284fd616b086a9b05cb552869a826d155d9",
    "step13_sha256sums_sha256": "b41638b4e246326b155ef1c6169b57d93c68266b47a41b9d16f05dc516546e9e",
    "step15_report_sha256": "7ad98eb711bfdc3aa67f15d04704cfd293348a0c215280da78730983f257098e",
    "step16_design_sha256": "1039788a626c58ea989a7038aeca1b5fa25ef13edd33d982b0654f0e6dd92bc2",
    "sweep_sha256": "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186",
}
REQUIRED_AUTHORITY_FILE_RECEIPTS = {
    "d1_model_recipe_sha256": (
        "experiments/phase05/configs/d1_bacteria_id_pca20_lr_sg11.json",
        1051,
        "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046",
    ),
    "phase4_step12_design_sha256": (
        "reports/phase4/step12_d2_protocol_a_full_domain_eligibility_design.md",
        21066,
        "eeb2350ce6c359d84d9f1372962b8fcfb7fef7f623a77af18ae2e0abd03a4b38",
    ),
}
REQUIRED_REAL_INHERITED_RULINGS = {
    "p01_p04_full_domain_core": {
        "by_shot": {
            "5": {
                "class_upper_bound_complete": 0,
                "record_upper_bound_complete": 1898,
                "required_class_count": 30,
                "required_record_count": 4456,
            },
            "10": {
                "class_upper_bound_complete": 0,
                "record_upper_bound_complete": 2237,
                "required_class_count": 30,
                "required_record_count": 4778,
            },
            "20": {
                "class_upper_bound_complete": 0,
                "record_upper_bound_complete": 2721,
                "required_class_count": 30,
                "required_record_count": 5238,
            },
        },
        "state": "not_evaluable_coverage",
    },
    "p05_full_domain_core": {
        "by_shot": {
            "5": {
                "class_upper_bound_complete": 0,
                "record_upper_bound_complete": 1692,
                "required_class_count": 30,
                "required_record_count": 4221,
            },
            "10": {
                "class_upper_bound_complete": 0,
                "record_upper_bound_complete": 2031,
                "required_class_count": 30,
                "required_record_count": 4527,
            },
            "20": {
                "class_upper_bound_complete": 0,
                "record_upper_bound_complete": 2515,
                "required_class_count": 30,
                "required_record_count": 4962,
            },
        },
        "state": "not_evaluable_coverage",
    },
    "p06": {
        "reason": "structurally_ineligible_missing_explicit_baseline",
        "state": "structurally_ineligible_missing_explicit_baseline",
    },
    "p07": {
        "reason": "structurally_ineligible_missing_explicit_baseline",
        "state": "structurally_ineligible_missing_explicit_baseline",
    },
}


class Phase4D2ProtocolBEligibilityVerifierError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D2ProtocolBEligibilityConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    active_perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    artifact_payload_files: tuple[str, ...]
    source_record_count: int
    model_cell_count: int
    model_role_occurrence_count: int
    role_condition_audit_count: int
    shot_counts: tuple[int, ...]
    class_count: int
    shot_source_union_counts: Mapping[str, int]
    shot_role_occurrence_counts: Mapping[str, int]
    expected_operator_cell_count: int
    expected_apply_check_count: int
    expected_positive_record_condition_count: int
    expected_canonical_record_condition_count: int
    expected_class_summary_count: int
    support_coordinates_cm1: tuple[float, ...]
    support_point_count: int
    support_max_gap_cm1: float
    p10_memory_budget_bytes: int
    native_gate_relative_tolerance: float
    authorities: Mapping[str, object]
    frozen_identities: Mapping[str, object]
    inherited_rulings: Mapping[str, object]
    code_authority: Mapping[str, Mapping[str, object]]
    environment_authority: Mapping[str, object]
    claim_boundary: str
    trust_anchor: Mapping[str, object]


@dataclass(frozen=True)
class D2ProtocolBEligibilityInputs:
    source_records: tuple[Mapping[str, object], ...]
    model_cells: tuple[Mapping[str, object], ...]
    model_role_occurrences: tuple[Mapping[str, object], ...]
    source_spectra: tuple[Spectrum1D, ...]
    source_record_ids_sha256: str
    native_axis_point_count: int
    support_axis_point_count: int


@dataclass(frozen=True)
class Phase4D2ProtocolBEligibilitySummary:
    path: Path
    run_id: str
    status: str
    source_record_count: int
    model_cell_count: int
    role_occurrence_count: int
    operator_cell_count: int
    record_condition_count: int
    class_summary_count: int


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
    raise Phase4D2ProtocolBEligibilityVerifierError(
        "json",
        f"unsupported value type {type(value).__name__}",
    )


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, *, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha_bytes(("\n".join(values) + "\n").encode("utf-8"))


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D2ProtocolBEligibilityVerifierError(path, "must be an object")
    return value


def _strings(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D2ProtocolBEligibilityVerifierError(path, "must be an array")
    converted = tuple(str(item) for item in value)
    if any(not item for item in converted):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            path,
            "must contain nonempty strings",
        )
    return converted


def _floats(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D2ProtocolBEligibilityVerifierError(path, "must be an array")
    converted = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in converted):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            path,
            "must contain finite numbers",
        )
    return converted


def _int(path: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4D2ProtocolBEligibilityVerifierError(path, "must be an integer")
    if value < minimum:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            path,
            "is outside the allowed range",
        )
    return value


def _number(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D2ProtocolBEligibilityVerifierError(path, "must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise Phase4D2ProtocolBEligibilityVerifierError(path, "must be finite")
    return number


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


def _code_document() -> dict[str, Mapping[str, object]]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha_file(ROOT / relative_path),
        }
        for relative_path in CODE_RELATIVE_PATHS
    }


def _config_authority_document() -> dict[str, object]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    return {"bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _validate_real_authority_receipts() -> None:
    for key, (relative_path, expected_bytes, expected_sha256) in (
        REQUIRED_AUTHORITY_FILE_RECEIPTS.items()
    ):
        path = ROOT / relative_path
        if not path.is_file():
            raise Phase4D2ProtocolBEligibilityVerifierError(
                f"authorities.{key}",
                "authority path is missing",
            )
        if path.stat().st_size != expected_bytes or _sha_file(path) != expected_sha256:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                f"authorities.{key}",
                "live authority bytes or SHA-256 mismatch",
            )


def _load_frozen_config_identity() -> tuple[int, str]:
    module = ast.parse(
        (ROOT / CONFIG_AUTHORITY_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    config_bytes: int | None = None
    config_sha256: str | None = None
    for statement in module.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "CONFIG_BYTES":
            config_bytes = ast.literal_eval(statement.value)
        if target.id == "CONFIG_SHA256":
            config_sha256 = ast.literal_eval(statement.value)
    if not isinstance(config_bytes, int) or not isinstance(config_sha256, str):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            CONFIG_AUTHORITY_RELATIVE_PATH,
            "must define CONFIG_BYTES and CONFIG_SHA256",
        )
    return config_bytes, config_sha256


def _step13_support_coordinates() -> tuple[float, ...]:
    path = ROOT / STEP13_CONFIG_RELATIVE_PATH
    document = json.loads(path.read_text(encoding="utf-8"))
    support = _object("step13.support_grid", document["support_grid"])
    return _floats("step13.support_grid.coordinates_cm1", support["coordinates_cm1"])


def parse_phase4_d2_protocol_b_eligibility_config(
    path: Path,
    raw: bytes,
    *,
    require_frozen_identity: bool,
) -> Phase4D2ProtocolBEligibilityConfig:
    try:
        document_value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D2ProtocolBEligibilityVerifierError("config", str(error)) from error
    document = _object("config", document_value)
    if raw != _canonical_json_bytes(document):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "config",
            "must use canonical JSON",
        )
    if require_frozen_identity:
        config_bytes, config_sha256 = _load_frozen_config_identity()
        if len(raw) != config_bytes or _sha_bytes(raw) != config_sha256:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "frozen config identity",
                "bytes or SHA-256 mismatch",
            )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "schema_version",
            "mismatch",
        )
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "experiment_id",
            "mismatch",
        )
    if document.get("protocol") != PROTOCOL:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "protocol",
            "must equal 'B'",
        )

    active_perturbation_ids = _strings(
        "active_perturbation_ids",
        document.get("active_perturbation_ids"),
    )
    if active_perturbation_ids != ACTIVE_PERTURBATION_IDS:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "active_perturbation_ids",
            "frozen perturbation set mismatch",
        )
    alpha_grid = _floats("alpha_grid", document.get("alpha_grid"))
    if alpha_grid != ALPHA_GRID:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "alpha_grid",
            "frozen grid mismatch",
        )
    artifact_payload_files = _strings(
        "artifact_payload_files",
        document.get("artifact_payload_files"),
    )
    if artifact_payload_files != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "artifact_payload_files",
            "frozen payload order mismatch",
        )

    denominators = _object("denominators", document.get("denominators"))
    shot_source_union_counts = dict(
        _object(
            "denominators.shot_source_union_counts",
            denominators.get("shot_source_union_counts"),
        )
    )
    shot_role_occurrence_counts = dict(
        _object(
            "denominators.shot_role_occurrence_counts",
            denominators.get("shot_role_occurrence_counts"),
        )
    )
    source_record_count = _int(
        "denominators.source_record_count",
        denominators.get("source_record_count"),
        minimum=1,
    )
    model_cell_count = _int(
        "denominators.model_cell_count",
        denominators.get("model_cell_count"),
        minimum=1,
    )
    model_role_occurrence_count = _int(
        "denominators.model_role_occurrence_count",
        denominators.get("model_role_occurrence_count"),
        minimum=1,
    )
    role_condition_audit_count = _int(
        "denominators.role_condition_audit_count",
        denominators.get("role_condition_audit_count"),
        minimum=1,
    )
    class_count = _int(
        "denominators.class_count",
        denominators.get("class_count"),
        minimum=1,
    )
    shot_counts = tuple(
        _int("shot_counts[]", value, minimum=1)
        for value in document.get("shot_counts", ())
    )
    if shot_counts != (5, 10, 20):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "shot_counts",
            "must equal [5,10,20]",
        )

    expected = _object("expected", document.get("expected"))
    expected_operator_cell_count = _int(
        "expected.operator_cell_count",
        expected.get("operator_cell_count"),
        minimum=1,
    )
    expected_apply_check_count = _int(
        "expected.apply_check_count",
        expected.get("apply_check_count"),
        minimum=1,
    )
    expected_positive_record_condition_count = _int(
        "expected.positive_record_condition_count",
        expected.get("positive_record_condition_count"),
        minimum=1,
    )
    expected_canonical_record_condition_count = _int(
        "expected.canonical_record_condition_count",
        expected.get("canonical_record_condition_count"),
        minimum=1,
    )
    expected_class_summary_count = _int(
        "expected.class_summary_count",
        expected.get("class_summary_count"),
        minimum=1,
    )

    synthetic_fixture = bool(document.get("synthetic_fixture", False))
    support_grid = _object("support_grid", document.get("support_grid"))
    support_coordinates_raw = support_grid.get("coordinates_cm1")
    if support_coordinates_raw is None:
        if synthetic_fixture:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "support_grid.coordinates_cm1",
                "synthetic fixtures must provide explicit coordinates",
            )
        support_coordinates_cm1 = _step13_support_coordinates()
    else:
        support_coordinates_cm1 = _floats(
            "support_grid.coordinates_cm1",
            support_coordinates_raw,
        )
    support_point_count = _int(
        "support_grid.point_count",
        support_grid.get("point_count"),
        minimum=1,
    )
    support_max_gap_cm1 = _number(
        "support_grid.max_in_range_native_gap_cm1",
        support_grid.get("max_in_range_native_gap_cm1"),
    )
    if len(support_coordinates_cm1) != support_point_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "support_grid",
            "point count mismatch",
        )

    p10 = _object("p10", document.get("p10"))
    p10_memory_budget_bytes = _int(
        "p10.memory_budget_bytes",
        p10.get("memory_budget_bytes"),
        minimum=1,
    )
    if (
        _number("p10.correlation_length_cm1", p10.get("correlation_length_cm1"))
        != 20.0
        or p10.get("peak_estimate_formula") != "32*N^2+64*N+2^30"
    ):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "p10",
            "frozen contract mismatch",
        )
    native_gate_relative_tolerance = _number(
        "phase1_native_gate_relative_tolerance",
        document.get("phase1_native_gate_relative_tolerance"),
    )
    if native_gate_relative_tolerance != 1e-12:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "phase1_native_gate_relative_tolerance",
            "must equal 1e-12",
        )

    authorities = {
        str(key): value
        for key, value in _object(
            "authorities",
            document.get("authorities", {}),
        ).items()
    }
    frozen_identities = {
        str(key): value
        for key, value in _object(
            "frozen_identities",
            document.get("frozen_identities"),
        ).items()
    }
    inherited_rulings = {
        str(key): value
        for key, value in _object(
            "inherited_rulings",
            document.get("inherited_rulings"),
        ).items()
    }
    code_authority = {
        str(key): {
            "bytes": _int(
                f"code_authority.{key}.bytes",
                value["bytes"],
                minimum=1,
            ),
            "sha256": str(value["sha256"]),
        }
        for key, value in _object(
            "code_authority",
            document.get("code_authority", {}),
        ).items()
    }
    environment_authority = {
        str(key): value
        for key, value in _object(
            "environment_authority",
            document.get("environment_authority", {}),
        ).items()
    }
    claim_boundary = str(document.get("claim_boundary"))
    if not claim_boundary:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "claim_boundary",
            "must be nonempty",
        )
    trust_anchor = {
        str(key): value
        for key, value in _object(
            "trust_anchor",
            document.get("trust_anchor", {}),
        ).items()
    }
    if synthetic_fixture:
        if authorities or code_authority or environment_authority or trust_anchor:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "synthetic authority",
                "synthetic fixtures may only use empty authority mappings",
            )
    else:
        if dict(authorities) != REQUIRED_AUTHORITIES:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "authorities",
                "must contain the exact frozen authority identities",
            )
        _validate_real_authority_receipts()
        if trust_anchor.get("config_authority_relative_path") != CONFIG_AUTHORITY_RELATIVE_PATH:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "trust_anchor",
                "authority path mismatch",
            )
        if tuple(code_authority) != CODE_RELATIVE_PATHS:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "code_authority",
                "must contain the exact ordered frozen paths",
            )
        if dict(environment_authority) != _environment_document():
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "environment_authority",
                "must match the frozen runtime environment",
            )
        if _json_ready(inherited_rulings) != REQUIRED_REAL_INHERITED_RULINGS:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "inherited_rulings",
                "must match the frozen Step-16 inherited bounds",
            )

    return Phase4D2ProtocolBEligibilityConfig(
        path=Path(path),
        raw_bytes=raw,
        sha256=_sha_bytes(raw),
        document=document,
        synthetic_fixture=synthetic_fixture,
        active_perturbation_ids=active_perturbation_ids,
        alpha_grid=alpha_grid,
        artifact_payload_files=artifact_payload_files,
        source_record_count=source_record_count,
        model_cell_count=model_cell_count,
        model_role_occurrence_count=model_role_occurrence_count,
        role_condition_audit_count=role_condition_audit_count,
        shot_counts=shot_counts,
        class_count=class_count,
        shot_source_union_counts={
            str(key): _int(f"shot_source_union_counts.{key}", value, minimum=1)
            for key, value in shot_source_union_counts.items()
        },
        shot_role_occurrence_counts={
            str(key): _int(f"shot_role_occurrence_counts.{key}", value, minimum=1)
            for key, value in shot_role_occurrence_counts.items()
        },
        expected_operator_cell_count=expected_operator_cell_count,
        expected_apply_check_count=expected_apply_check_count,
        expected_positive_record_condition_count=expected_positive_record_condition_count,
        expected_canonical_record_condition_count=expected_canonical_record_condition_count,
        expected_class_summary_count=expected_class_summary_count,
        support_coordinates_cm1=support_coordinates_cm1,
        support_point_count=support_point_count,
        support_max_gap_cm1=support_max_gap_cm1,
        p10_memory_budget_bytes=p10_memory_budget_bytes,
        native_gate_relative_tolerance=native_gate_relative_tolerance,
        authorities=authorities,
        frozen_identities=frozen_identities,
        inherited_rulings=inherited_rulings,
        code_authority=code_authority,
        environment_authority=environment_authority,
        claim_boundary=claim_boundary,
        trust_anchor=trust_anchor,
    )


def load_phase4_d2_protocol_b_eligibility_config(
    path: Path,
) -> Phase4D2ProtocolBEligibilityConfig:
    raw = Path(path).read_bytes()
    return parse_phase4_d2_protocol_b_eligibility_config(
        path,
        raw,
        require_frozen_identity=True,
    )


def _validate_code_and_environment(
    config: Phase4D2ProtocolBEligibilityConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    if config.synthetic_fixture:
        return (
            {str(key): dict(value) for key, value in config.code_authority.items()},
            dict(config.environment_authority),
        )
    code = _code_document()
    if code != {
        str(key): dict(value) for key, value in config.code_authority.items()
    }:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "code_authority",
            "mismatch",
        )
    environment = _environment_document()
    if environment != dict(config.environment_authority):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "environment_authority",
            "mismatch",
        )
    return code, environment


def _load_json_document(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D2ProtocolBEligibilityVerifierError(str(path), str(error)) from error
    if raw != _canonical_json_bytes(value):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            str(path),
            "must be canonical JSON",
        )
    return _object(str(path), value)


def project_d2_protocol_b_support(
    spectrum: Spectrum1D,
    config: Phase4D2ProtocolBEligibilityConfig,
) -> np.ndarray:
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    intensity = np.asarray(spectrum.intensity, dtype="<f8")
    support = np.asarray(config.support_coordinates_cm1, dtype="<f8")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.size != intensity.size:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "spectrum",
            "axis/intensity mismatch",
        )
    if float(axis[0]) > float(support[0]) or float(axis[-1]) < float(support[-1]):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "support_projection",
            "extrapolation required",
        )
    left = int(np.searchsorted(axis, support[0], side="left"))
    right = int(np.searchsorted(axis, support[-1], side="right"))
    native_in_range = axis[left:right]
    if native_in_range.size < 2:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "support_projection",
            "insufficient in-range native support",
        )
    if float(np.max(np.diff(native_in_range))) > config.support_max_gap_cm1:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "max_in_range_native_gap_cm1",
            "support gap exceeds the frozen gate",
        )
    if all(float(value) in {float(item) for item in axis} for value in support):
        index_map = {float(value): index for index, value in enumerate(axis)}
        selected = np.asarray(
            [intensity[index_map[float(value)]] for value in support],
            dtype="<f8",
        )
    else:
        selected = np.interp(support, axis, intensity).astype("<f8")
    output = np.ascontiguousarray(selected, dtype="<f4")
    if output.size != config.support_point_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "support_projection",
            "point count mismatch",
        )
    if not np.isfinite(output).all():
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "support_projection",
            "must be finite",
        )
    if float(np.linalg.norm(output.astype(np.float64))) <= 0.0:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "support_projection",
            "zero norm",
        )
    return output


def _reconstruct_real_inputs(
    dataset_path: Path,
    selection_path: Path,
    config: Phase4D2ProtocolBEligibilityConfig,
) -> D2ProtocolBEligibilityInputs:
    selection = _load_json_document(selection_path)
    try:
        validate_d2_few_shot_selection(
            selection,
            ROOT / "experiments/phase05/configs/d2_few_shot_selection.json",
            dataset_path,
        )
    except D2SelectionValidationError as error:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "selection replay",
            str(error),
        ) from error
    selected_record_ids: dict[str, int] = {}
    source_shot_memberships: dict[str, set[int]] = {}
    cell_specs = []
    for seed_document in selection["selections"]:
        seed = _int("selection.seed", seed_document["seed"], minimum=0)
        per_class_validation: dict[int, list[str]] = {}
        per_class_train: dict[int, dict[int, list[str]]] = {}
        for class_document in seed_document["classes"]:
            class_label = _int(
                "selection.class_label",
                class_document["class_label"],
                minimum=0,
            )
            validation_ids = [
                str(value) for value in class_document["validation_record_ids"]
            ]
            per_class_validation[class_label] = validation_ids
            train_map = _object(
                "selection.train_record_ids",
                class_document["train_record_ids"],
            )
            per_class_train[class_label] = {
                int(shot): [str(value) for value in values]
                for shot, values in train_map.items()
            }
            for record_id in validation_ids:
                selected_record_ids.setdefault(record_id, class_label)
            for shot, values in per_class_train[class_label].items():
                for record_id in values:
                    selected_record_ids.setdefault(record_id, class_label)
                    source_shot_memberships.setdefault(record_id, set()).add(int(shot))
            for record_id in validation_ids:
                source_shot_memberships.setdefault(record_id, set()).update(
                    config.shot_counts
                )
        for shot_count in config.shot_counts:
            train_ids = sorted(
                value
                for class_values in per_class_train.values()
                for value in class_values[int(shot_count)]
            )
            validation_ids = sorted(
                value for values in per_class_validation.values() for value in values
            )
            if set(train_ids) & set(validation_ids):
                raise Phase4D2ProtocolBEligibilityVerifierError(
                    "selection",
                    "train/validation overlap",
                )
            cell_specs.append(
                {
                    "seed": seed,
                    "shot_count": int(shot_count),
                    "train_record_ids": tuple(train_ids),
                    "validation_record_ids": tuple(validation_ids),
                }
            )
    source_records = []
    source_spectra = []
    observed_model_seeds: set[int] = set()
    stored_axis_sha256: str | None = None
    native_axis_sha256: str | None = None
    test_record_ids: list[str] = []
    with BacteriaIdBatchLoader(dataset_path, batch_size=4096) as loader:
        for batch in loader.iter_batches():
            if batch.source_split not in {"finetune", "test"}:
                continue
            stored_axis = np.asarray(batch.wavenumber, dtype="<f4")
            if not np.all(np.diff(stored_axis) < 0.0):
                raise Phase4D2ProtocolBEligibilityVerifierError(
                    "BacteriaIdBatchLoader",
                    "stored axis must be strictly decreasing",
                )
            native_axis = np.ascontiguousarray(stored_axis[::-1], dtype="<f8")
            batch_stored_axis_sha256 = _array_sha(stored_axis, dtype="<f4")
            batch_native_axis_sha256 = _array_sha(native_axis, dtype="<f8")
            if stored_axis_sha256 is None:
                stored_axis_sha256 = batch_stored_axis_sha256
                native_axis_sha256 = batch_native_axis_sha256
            elif (
                stored_axis_sha256 != batch_stored_axis_sha256
                or native_axis_sha256 != batch_native_axis_sha256
            ):
                raise Phase4D2ProtocolBEligibilityVerifierError(
                    "BacteriaIdBatchLoader",
                    "all batches must share the same frozen native axis",
                )
            for record_id, class_label, stored_intensity, source_row in zip(
                batch.record_ids,
                batch.class_labels,
                batch.intensity,
                batch.source_rows,
                strict=True,
            ):
                record_id = str(record_id)
                if batch.source_split == "finetune" and record_id not in selected_record_ids:
                    continue
                intensity = np.ascontiguousarray(
                    np.asarray(stored_intensity, dtype="<f4")[::-1],
                    dtype="<f8",
                )
                spectrum = Spectrum1D(
                    spectrum_id=f"bacteria_id_reference::{record_id}",
                    sample_id=None,
                    axis_cm1=native_axis,
                    intensity=intensity,
                )
                source_spectra.append(spectrum)
                projected = project_d2_protocol_b_support(spectrum, config)
                memberships = (
                    sorted(source_shot_memberships.get(record_id, set(config.shot_counts)))
                    if batch.source_split == "finetune"
                    else list(config.shot_counts)
                )
                if batch.source_split == "test":
                    test_record_ids.append(record_id)
                    source_shot_memberships.setdefault(record_id, set()).update(
                        config.shot_counts
                    )
                source_records.append(
                    {
                        "class_label": int(class_label),
                        "native_axis_sha256": _array_sha(native_axis, dtype="<f8"),
                        "native_intensity_sha256": _array_sha(intensity, dtype="<f8"),
                        "record_id": record_id,
                        "record_order": len(source_records),
                        "scope": batch.source_split,
                        "shot_memberships": memberships,
                        "source_row": int(source_row),
                        "support_projection_sha256": _array_sha(projected, dtype="<f4"),
                    }
                )
    record_order = {
        str(row["record_id"]): int(row["record_order"]) for row in source_records
    }
    model_cells = []
    model_role_occurrences = []
    test_record_ids = sorted(test_record_ids)
    for cell in cell_specs:
        observed_model_seeds.add(int(cell["seed"]))
        model_cell = {**cell, "test_record_ids": tuple(test_record_ids)}
        model_cells.append(model_cell)
        for role_name in (
            "train_record_ids",
            "validation_record_ids",
            "test_record_ids",
        ):
            role = role_name.split("_", 1)[0]
            for record_id in model_cell[role_name]:
                model_role_occurrences.append(
                    {
                        "record_id": record_id,
                        "role": role,
                        "seed": int(model_cell["seed"]),
                        "shot_count": int(model_cell["shot_count"]),
                    }
                )
    role_counts = Counter(str(row["record_id"]) for row in model_role_occurrences)
    final_source_records = [
        {
            **row,
            "role_count": role_counts[str(row["record_id"])],
            "shot_memberships": sorted(row["shot_memberships"]),
        }
        for row in sorted(source_records, key=lambda item: str(item["record_id"]))
    ]
    if len(final_source_records) != config.source_record_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "source_records",
            "count mismatch",
        )
    if len(model_cells) != config.model_cell_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "model_cells",
            "count mismatch",
        )
    if len(model_role_occurrences) != config.model_role_occurrence_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "model_role_occurrences",
            "count mismatch",
        )
    _validate_real_reconstruction_identities(
        config,
        source_records=final_source_records,
        model_seed_ids=observed_model_seeds,
        test_record_ids=test_record_ids,
        stored_axis_sha256=stored_axis_sha256,
        native_axis_sha256=native_axis_sha256,
    )
    return D2ProtocolBEligibilityInputs(
        source_records=tuple(final_source_records),
        model_cells=tuple(
            sorted(
                model_cells,
                key=lambda row: (int(row["seed"]), int(row["shot_count"])),
            )
        ),
        model_role_occurrences=tuple(
            sorted(
                model_role_occurrences,
                key=lambda row: (
                    int(row["seed"]),
                    int(row["shot_count"]),
                    0
                    if str(row["role"]) == "train"
                    else 1
                    if str(row["role"]) == "validation"
                    else 2,
                    record_order[str(row["record_id"])],
                ),
            )
        ),
        source_spectra=tuple(
            source_spectra[record_order[str(row["record_id"])]]
            for row in final_source_records
        ),
        source_record_ids_sha256=_ids_digest(
            [str(row["record_id"]) for row in final_source_records]
        ),
        native_axis_point_count=1000,
        support_axis_point_count=config.support_point_count,
    )


def _validate_real_reconstruction_identities(
    config: Phase4D2ProtocolBEligibilityConfig,
    *,
    source_records: Sequence[Mapping[str, object]],
    model_seed_ids: set[int],
    test_record_ids: Sequence[str],
    stored_axis_sha256: str | None,
    native_axis_sha256: str | None,
) -> None:
    frozen = _object("frozen_identities", config.frozen_identities)
    expected_model_seeds = tuple(
        sorted(
            _int("frozen_identities.model_seed_ids[]", value, minimum=0)
            for value in frozen.get("model_seed_ids", ())
        )
    )
    if expected_model_seeds != tuple(sorted(model_seed_ids)):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "frozen_identities.model_seed_ids",
            "reconstructed model seed identities mismatch",
        )
    if str(frozen.get("native_axis_decreasing_f32_sha256")) != str(stored_axis_sha256):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "frozen_identities.native_axis_decreasing_f32_sha256",
            "reconstructed native decreasing axis digest mismatch",
        )
    if str(frozen.get("native_axis_increasing_f64_sha256")) != str(native_axis_sha256):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "frozen_identities.native_axis_increasing_f64_sha256",
            "reconstructed native increasing axis digest mismatch",
        )
    expected_test_digest = _ids_digest(sorted(str(record_id) for record_id in test_record_ids))
    if str(frozen.get("selection_test_record_ids_sha256")) != expected_test_digest:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "frozen_identities.selection_test_record_ids_sha256",
            "reconstructed test record digest mismatch",
        )
    support_f64 = np.ascontiguousarray(np.asarray(config.support_coordinates_cm1, dtype="<f8"))
    support_f32 = np.ascontiguousarray(support_f64, dtype="<f4")
    if str(frozen.get("support_axis_f64_sha256")) != _array_sha(support_f64, dtype="<f8"):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "frozen_identities.support_axis_f64_sha256",
            "support-axis float64 digest mismatch",
        )
    if str(frozen.get("support_axis_f32_sha256")) != _array_sha(support_f32, dtype="<f4"):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "frozen_identities.support_axis_f32_sha256",
            "support-axis float32 digest mismatch",
        )
    if _int(
        "frozen_identities.support_point_count",
        frozen.get("support_point_count"),
        minimum=1,
    ) != config.support_point_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "frozen_identities.support_point_count",
            "support point count mismatch",
        )
    shot_identities = _object(
        "frozen_identities.shot_source_union_record_ids_sha256",
        frozen.get("shot_source_union_record_ids_sha256"),
    )
    for shot_count in config.shot_counts:
        shot_key = str(int(shot_count))
        shot_record_ids = sorted(
            str(row["record_id"])
            for row in source_records
            if int(shot_count) in row["shot_memberships"]
        )
        if len(shot_record_ids) != int(config.shot_source_union_counts[shot_key]):
            raise Phase4D2ProtocolBEligibilityVerifierError(
                f"shot_source_union_counts.{shot_key}",
                "reconstructed shot-union count mismatch",
            )
        if _ids_digest(shot_record_ids) != str(shot_identities.get(shot_key)):
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "frozen_identities.shot_source_union_record_ids_sha256",
                f"shot {shot_key} digest mismatch",
            )


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
        source_axis_float32_sha256=_array_sha(axis_f4, dtype="<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, dtype="<f4"),
        normalized_axis_float64_sha256=str(record["native_axis_sha256"]),
        normalized_intensity_float64_sha256=str(record["native_intensity_sha256"]),
        provenance=MappingProxyType(
            {
                "dataset_id": "bacteria_id_reference",
                "record_id": str(record["record_id"]),
            }
        ),
    )


_PROCESS_SWEEP: PerturbationSweepConfig | None = None
_PROCESS_PHASE1_CONFIG: Phase1CoreConfig | None = None
_PROCESS_CONFIG: Phase4D2ProtocolBEligibilityConfig | None = None


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
    _PROCESS_CONFIG = parse_phase4_d2_protocol_b_eligibility_config(
        Path(config_path),
        config_raw,
        require_frozen_identity=False,
    )


def _project_support_receipt(
    spectrum: Spectrum1D,
    config: Phase4D2ProtocolBEligibilityConfig,
) -> tuple[str, int, float]:
    support = np.asarray(config.support_coordinates_cm1, dtype="<f8")
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    left = int(np.searchsorted(axis, support[0], side="left"))
    right = int(np.searchsorted(axis, support[-1], side="right"))
    native_in_range = axis[left:right]
    max_gap = float(np.max(np.diff(native_in_range)))
    projected = project_d2_protocol_b_support(spectrum, config)
    return _array_sha(projected, dtype="<f4"), int(projected.size), max_gap


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
    config: Phase4D2ProtocolBEligibilityConfig,
    *,
    p10_estimate: int | None,
) -> dict[str, object]:
    state_digest = None if cell.state is None else cell.state.state_digest
    if cell.status is CellStatus.COMPLETE:
        outputs = []
        for perturbed in cell.records:
            output = perturbed.result.output
            support_hash, support_point_count, support_max_gap = _project_support_receipt(
                output,
                config,
            )
            outputs.append(
                {
                    "alpha": float(perturbed.result.alpha),
                    "alpha_float64_le_hex": perturbed.alpha_float64_le_hex,
                    "axis_changed": bool(perturbed.result.axis_changed),
                    "diagnostics": _json_ready(perturbed.result.diagnostics),
                    "intensity_changed": bool(perturbed.result.intensity_changed),
                    "output_axis_sha256": _array_sha(output.axis_cm1, dtype="<f8"),
                    "output_intensity_sha256": _array_sha(output.intensity, dtype="<f8"),
                    "output_spectrum_id": output.spectrum_id,
                    "support_max_in_range_gap_cm1": support_max_gap,
                    "support_point_count": support_point_count,
                    "support_projection_sha256": support_hash,
                }
            )
        return {
            "class_label": int(record["class_label"]),
            "exception": None,
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


def _execute_initialized_record(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
) -> tuple[dict[str, object], ...]:
    if _PROCESS_SWEEP is None or _PROCESS_PHASE1_CONFIG is None or _PROCESS_CONFIG is None:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "worker",
            "process not initialized",
        )
    source = _phase1_source(record, spectrum)
    admission = P10MemoryAdmission(_PROCESS_CONFIG.p10_memory_budget_bytes)
    receipts: list[dict[str, object]] = []
    for perturbation_id in ACTIVE_PERTURBATION_IDS:
        p10_estimate = (
            estimate_p10_peak_bytes(int(spectrum.axis_cm1.size))
            if perturbation_id == "p10"
            else None
        )
        try:
            with threadpool_limits(limits=1, user_api="blas"):
                cell = run_perturbation_cell(
                    source,
                    perturbation_id,
                    _PROCESS_PHASE1_CONFIG,
                    _PROCESS_SWEEP,
                    p10_admission=admission if perturbation_id == "p10" else None,
                )
            if (
                perturbation_id == "p10"
                and p10_estimate is not None
                and p10_estimate > _PROCESS_CONFIG.p10_memory_budget_bytes
            ):
                raise Phase4D2ProtocolBEligibilityVerifierError(
                    "p10 admission",
                    "at least one record exceeds the frozen 64 GiB budget",
                )
            receipts.append(
                _cell_receipt(
                    record,
                    cell,
                    _PROCESS_CONFIG,
                    p10_estimate=p10_estimate,
                )
            )
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


def _bounded_worker_count(
    requested_workers: int,
    point_counts: Sequence[int],
    memory_budget_bytes: int,
) -> int:
    if requested_workers < 1:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "worker_count",
            "must be positive",
        )
    if not point_counts:
        return 1
    estimates = [estimate_p10_peak_bytes(int(point_count)) for point_count in point_counts]
    peak_estimate = max(estimates)
    if peak_estimate > memory_budget_bytes:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "p10 admission",
            "at least one record exceeds the frozen 64 GiB budget",
        )
    by_memory = max(1, memory_budget_bytes // peak_estimate)
    return max(1, min(int(requested_workers), int(by_memory), len(point_counts)))


def _run_record_jobs(
    inputs: D2ProtocolBEligibilityInputs,
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D2ProtocolBEligibilityConfig,
    worker_count: int,
) -> tuple[tuple[dict[str, object], ...], ...]:
    point_counts = tuple(spectrum.axis_cm1.size for spectrum in inputs.source_spectra)
    bounded_count = _bounded_worker_count(
        worker_count,
        point_counts,
        config.p10_memory_budget_bytes,
    )
    if bounded_count == 1:
        global _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG, _PROCESS_CONFIG
        _PROCESS_SWEEP = sweep
        _PROCESS_PHASE1_CONFIG = phase1_config
        _PROCESS_CONFIG = config
        try:
            return tuple(
                _execute_initialized_record(record, spectrum)
                for record, spectrum in zip(
                    inputs.source_records,
                    inputs.source_spectra,
                    strict=True,
                )
            )
        finally:
            _PROCESS_SWEEP = None
            _PROCESS_PHASE1_CONFIG = None
            _PROCESS_CONFIG = None
    with ProcessPoolExecutor(
        max_workers=bounded_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_process,
        initargs=(
            str(sweep.path),
            str(phase1_config.path),
            str(config.path),
            config.raw_bytes,
        ),
    ) as executor:
        futures = [
            executor.submit(
                _execute_initialized_record,
                record,
                spectrum,
            )
            for record, spectrum in zip(
                inputs.source_records,
                inputs.source_spectra,
                strict=True,
            )
        ]
        return tuple(
            future.result()
            for future in futures
        )


def _condition_id(perturbation_id: str, alpha: float) -> str:
    return f"{perturbation_id}:{np.float64(alpha).tobytes().hex()}"


def _build_record_conditions(
    inputs: D2ProtocolBEligibilityInputs,
    operator_cells: Sequence[Mapping[str, object]],
    config: Phase4D2ProtocolBEligibilityConfig,
) -> list[dict[str, object]]:
    spectrum_by_id = {
        str(row["record_id"]): spectrum
        for row, spectrum in zip(inputs.source_records, inputs.source_spectra, strict=True)
    }
    cell_by_key = {
        (str(row["record_id"]), str(row["perturbation_id"])): row
        for row in operator_cells
    }
    conditions = []
    for source_record in inputs.source_records:
        record_id = str(source_record["record_id"])
        spectrum = spectrum_by_id[record_id]
        support_hash, support_point_count, max_gap = _project_support_receipt(
            spectrum,
            config,
        )
        conditions.append(
            {
                "alpha": 0.0,
                "alpha_float64_le_hex": np.float64(0.0).tobytes().hex(),
                "axis_sha256": str(source_record["native_axis_sha256"]),
                "class_label": int(source_record["class_label"]),
                "condition_id": "alpha0",
                "condition_kind": "alpha0",
                "intensity_sha256": str(source_record["native_intensity_sha256"]),
                "perturbation_id": None,
                "record_id": record_id,
                "record_order": int(source_record["record_order"]),
                "state": "complete",
                "support_max_in_range_gap_cm1": max_gap,
                "support_point_count": support_point_count,
                "support_projection_sha256": support_hash,
            }
        )
        alpha_zero_axis = None
        alpha_zero_intensity = None
        for perturbation_id in ACTIVE_PERTURBATION_IDS:
            cell = cell_by_key[(record_id, perturbation_id)]
            if str(cell["state"]) == "complete":
                outputs = list(cell["outputs"])
                alpha0 = outputs[0]
                if (
                    str(alpha0["output_axis_sha256"])
                    != str(source_record["native_axis_sha256"])
                    or str(alpha0["output_intensity_sha256"])
                    != str(source_record["native_intensity_sha256"])
                ):
                    raise Phase4D2ProtocolBEligibilityVerifierError(
                        "alpha-zero collapse",
                        "all active operators must preserve native alpha-zero bytes",
                    )
                if alpha_zero_axis is None:
                    alpha_zero_axis = str(alpha0["output_axis_sha256"])
                    alpha_zero_intensity = str(alpha0["output_intensity_sha256"])
                elif (
                    alpha_zero_axis != str(alpha0["output_axis_sha256"])
                    or alpha_zero_intensity != str(alpha0["output_intensity_sha256"])
                ):
                    raise Phase4D2ProtocolBEligibilityVerifierError(
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
                            "class_label": int(source_record["class_label"]),
                            "condition_id": _condition_id(perturbation_id, alpha),
                            "condition_kind": "positive",
                            "intensity_sha256": str(output["output_intensity_sha256"]),
                            "perturbation_id": perturbation_id,
                            "record_id": record_id,
                            "record_order": int(source_record["record_order"]),
                            "state": "complete",
                            "support_max_in_range_gap_cm1": float(
                                output["support_max_in_range_gap_cm1"]
                            ),
                            "support_point_count": int(output["support_point_count"]),
                            "support_projection_sha256": str(
                                output["support_projection_sha256"]
                            ),
                        }
                    )
            else:
                for alpha in POSITIVE_ALPHAS:
                    conditions.append(
                        {
                            "alpha": float(alpha),
                            "alpha_float64_le_hex": np.float64(alpha).tobytes().hex(),
                            "axis_sha256": None,
                            "class_label": int(source_record["class_label"]),
                            "condition_id": _condition_id(perturbation_id, alpha),
                            "condition_kind": "positive",
                            "intensity_sha256": None,
                            "perturbation_id": perturbation_id,
                            "record_id": record_id,
                            "record_order": int(source_record["record_order"]),
                            "state": str(cell["state"]),
                            "support_max_in_range_gap_cm1": None,
                            "support_point_count": None,
                            "support_projection_sha256": None,
                        }
                    )
    return conditions


def evaluate_d2_protocol_b_all_role_gates(
    source_records: Sequence[Mapping[str, object]],
    model_cells: Sequence[Mapping[str, object]],
    model_role_occurrences: Sequence[Mapping[str, object]],
    operator_cells: Sequence[Mapping[str, object]],
    config: Phase4D2ProtocolBEligibilityConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if len(source_records) != config.source_record_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "source_records",
            "denominator mismatch",
        )
    if len(model_cells) != config.model_cell_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "model_cells",
            "denominator mismatch",
        )
    if len(model_role_occurrences) != config.model_role_occurrence_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "model_role_occurrences",
            "denominator mismatch",
        )
    if len(operator_cells) != config.expected_operator_cell_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "operator_cells",
            "count mismatch",
        )
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
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "operator_cells",
            "grid keys mismatch",
        )
    class_rows = []
    shots: dict[str, dict[str, object]] = {}
    for shot_count in config.shot_counts:
        shot_key = str(int(shot_count))
        shot_records = [
            str(row["record_id"])
            for row in source_records
            if int(shot_count) in row["shot_memberships"]
        ]
        class_to_records: dict[int, list[str]] = {}
        for record_id in shot_records:
            class_label = int(source_by_id[record_id]["class_label"])
            class_to_records.setdefault(class_label, []).append(record_id)
        role_occurrences_for_shot = [
            row for row in model_role_occurrences if int(row["shot_count"]) == int(shot_count)
        ]
        shot_operators = {}
        for perturbation_id in ACTIVE_PERTURBATION_IDS:
            states = Counter(
                str(cell_by_key[(record_id, perturbation_id)]["state"])
                for record_id in shot_records
            )
            complete_records = {
                record_id
                for record_id in shot_records
                if cell_by_key[(record_id, perturbation_id)]["state"] == "complete"
            }
            complete_classes = 0
            for class_label in sorted(class_to_records):
                record_ids = sorted(class_to_records[class_label])
                complete_record_count = sum(
                    record_id in complete_records for record_id in record_ids
                )
                complete = complete_record_count == len(record_ids)
                complete_classes += int(complete)
                class_rows.append(
                    {
                        "class_label": class_label,
                        "complete": complete,
                        "complete_record_count": complete_record_count,
                        "perturbation_id": perturbation_id,
                        "required_record_count": len(record_ids),
                        "shot_count": int(shot_count),
                        "state": "complete" if complete else "closed_incomplete",
                    }
                )
            shot_operators[perturbation_id] = {
                "complete_record_count": len(complete_records),
                "required_record_count": len(shot_records),
                "complete_class_count": complete_classes,
                "required_class_count": len(class_to_records),
                "failed_runtime_count": states.get("failed_runtime", 0),
                "not_applicable_count": states.get("not_applicable", 0),
                "state": (
                    "evaluable"
                    if len(complete_records) == len(shot_records)
                    and complete_classes == len(class_to_records)
                    and states.get("failed_runtime", 0) == 0
                    and states.get("not_applicable", 0) == 0
                    else "not_evaluable_coverage"
                ),
                "state_counts": {
                    state: states.get(state, 0) for state in TERMINAL_STATES
                },
            }
        shots[shot_key] = {
            "operators": shot_operators,
            "record_denominator": len(shot_records),
            "class_denominator": len(class_to_records),
            "role_occurrence_denominator": len(role_occurrences_for_shot),
            "state": (
                "evaluable"
                if all(
                    shot_operators[perturbation_id]["state"] == "evaluable"
                    for perturbation_id in ACTIVE_PERTURBATION_IDS
                )
                else "not_evaluable_coverage"
            ),
        }
    overall_status = (
        "pass"
        if all(shots[str(shot)]["state"] == "evaluable" for shot in config.shot_counts)
        else "fail"
    )
    gate = {
        "protocol": PROTOCOL,
        "shots": shots,
        "inherited_rulings": _json_ready(config.inherited_rulings),
        "overall_status": overall_status,
        "marker_filename": "complete.json" if overall_status == "pass" else "failed.json",
    }
    return class_rows, gate


def validate_protocol_b_outcome_blind_payload(
    value: object,
    *,
    path: str = "artifact",
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in FORBIDDEN_EXACT_KEYS or any(
                fragment in key_text for fragment in FORBIDDEN_KEY_FRAGMENTS
            ):
                raise Phase4D2ProtocolBEligibilityVerifierError(
                    "outcome-blind boundary",
                    f"forbidden field {key!r} at {path}",
                )
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            validate_protocol_b_outcome_blind_payload(item, path=f"{path}[{index}]")


def _run_identity(
    config: Phase4D2ProtocolBEligibilityConfig,
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
            "source_records": config.source_record_count,
            "model_cells": config.model_cell_count,
            "model_role_occurrences": config.model_role_occurrence_count,
            "shot_source_union_counts": dict(config.shot_source_union_counts),
            "shot_role_occurrence_counts": dict(config.shot_role_occurrence_counts),
        },
        "environment": environment,
        "frozen_identities": config.frozen_identities,
        "inherited_rulings": config.inherited_rulings,
        "support_grid": {
            "point_count": config.support_point_count,
            "max_gap_cm1": config.support_max_gap_cm1,
        },
        "trust_anchor": config.trust_anchor,
    }
    return RUN_PREFIX + _sha_bytes(_canonical_json_bytes(identity)), identity


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _rebuild_phase4_d2_protocol_b_eligibility_from_inputs(
    output_dir: Path,
    *,
    inputs: D2ProtocolBEligibilityInputs,
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D2ProtocolBEligibilityConfig,
    worker_count: int,
    code: Mapping[str, object],
    environment: Mapping[str, object],
) -> Phase4D2ProtocolBEligibilitySummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "worker_count",
            "must be a positive integer",
        )
    output_path = Path(output_dir)
    if output_path.exists():
        raise Phase4D2ProtocolBEligibilityVerifierError(
            str(output_path),
            "append-only output target already exists",
        )
    point_counts = tuple(spectrum.axis_cm1.size for spectrum in inputs.source_spectra)
    _bounded_worker_count(worker_count, point_counts, config.p10_memory_budget_bytes)
    operator_rows = _run_record_jobs(inputs, sweep, phase1_config, config, worker_count)
    operator_cells = [row for group in operator_rows for row in group]
    if len(operator_cells) != config.expected_operator_cell_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "operator_cells",
            "count mismatch",
        )
    record_conditions = _build_record_conditions(inputs, operator_cells, config)
    if len(record_conditions) != config.expected_canonical_record_condition_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "record_conditions",
            "count mismatch",
        )
    class_summaries, gate = evaluate_d2_protocol_b_all_role_gates(
        inputs.source_records,
        inputs.model_cells,
        inputs.model_role_occurrences,
        operator_cells,
        config,
    )
    if len(class_summaries) != config.expected_class_summary_count:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "class_summaries",
            "count mismatch",
        )
    run_id, run_identity = _run_identity(config, code, environment)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_order": list(ARTIFACT_PAYLOAD_FILES),
        "claim_boundary": config.claim_boundary,
        "counts": {
            "class_summaries": len(class_summaries),
            "model_cells": len(inputs.model_cells),
            "model_role_occurrences": len(inputs.model_role_occurrences),
            "operator_cells": len(operator_cells),
            "record_conditions": len(record_conditions),
            "source_records": len(inputs.source_records),
        },
        "environment": environment,
        "protocol": PROTOCOL,
        "run_id": run_id,
        "run_identity": run_identity,
        "synthetic_fixture": config.synthetic_fixture,
    }
    marker_name = str(gate["marker_filename"])
    marker = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": gate["overall_status"],
    }
    for payload in (
        config.document,
        inputs.source_records,
        inputs.model_cells,
        inputs.model_role_occurrences,
        operator_cells,
        record_conditions,
        class_summaries,
        gate,
        manifest,
        marker,
    ):
        validate_protocol_b_outcome_blind_payload(payload)
    payloads = {
        "config.json": config.raw_bytes,
        "source_records.jsonl": _jsonl_bytes(inputs.source_records),
        "model_cells.jsonl": _jsonl_bytes(inputs.model_cells),
        "model_role_occurrences.jsonl": _jsonl_bytes(inputs.model_role_occurrences),
        "operator_cells.jsonl": _jsonl_bytes(operator_cells),
        "record_conditions.jsonl": _jsonl_bytes(record_conditions),
        "class_summaries.jsonl": _jsonl_bytes(class_summaries),
        "gate.json": _canonical_json_bytes(gate),
        "manifest.json": _canonical_json_bytes(manifest),
        marker_name: _canonical_json_bytes(marker),
    }
    ordered_files = (*ARTIFACT_PAYLOAD_FILES, marker_name)
    output_path.mkdir(parents=True)
    for name in ordered_files:
        output_path.joinpath(name).write_bytes(payloads[name])
    checksum = "".join(
        f"{_sha_bytes(payloads[name])}  {name}\n" for name in ordered_files
    ).encode("utf-8")
    output_path.joinpath("SHA256SUMS").write_bytes(checksum)
    return Phase4D2ProtocolBEligibilitySummary(
        path=output_path,
        run_id=run_id,
        status=str(gate["overall_status"]),
        source_record_count=len(inputs.source_records),
        model_cell_count=len(inputs.model_cells),
        role_occurrence_count=len(inputs.model_role_occurrences),
        operator_cell_count=len(operator_cells),
        record_condition_count=len(record_conditions),
        class_summary_count=len(class_summaries),
    )


def _tree(path: Path) -> dict[str, bytes]:
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.iterdir()
        if item.is_file()
    }


def _compare_tree(candidate_path: Path, rebuilt_path: Path) -> None:
    if _tree(candidate_path) != _tree(rebuilt_path):
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "independent verifier",
            "rebuilt artifact bytes differ",
        )


def verify_phase4_d2_protocol_b_eligibility_from_inputs(
    path: Path,
    *,
    inputs: D2ProtocolBEligibilityInputs,
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config_path: Path,
    worker_count: int,
) -> Phase4D2ProtocolBEligibilitySummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "worker_count",
            "must be a positive integer",
        )
    path = Path(path)
    if not path.is_dir():
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "path",
            "must be an existing run directory",
        )
    raw = Path(config_path).read_bytes()
    document = json.loads(raw)
    config = parse_phase4_d2_protocol_b_eligibility_config(
        Path(config_path),
        raw,
        require_frozen_identity=not bool(document.get("synthetic_fixture", False)),
    )
    if not config.synthetic_fixture:
        frozen_path = ROOT / CONFIG_RELATIVE_PATH
        if frozen_path.is_file() and raw != frozen_path.read_bytes():
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "config.json",
                "does not match frozen experiment config",
            )
    if tuple(sweep.alpha_grid) != config.alpha_grid:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "sweep",
            "alpha grid mismatch",
        )
    if not config.synthetic_fixture:
        if sweep.sha256 != config.authorities["sweep_sha256"]:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "sweep",
                "identity mismatch",
            )
        if phase1_config.file_sha256 != config.authorities["phase1_core_config_sha256"]:
            raise Phase4D2ProtocolBEligibilityVerifierError(
                "phase1_config",
                "identity mismatch",
            )
    if phase1_config.core_gate["float_relative_tolerance"] != config.native_gate_relative_tolerance:
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "phase1_config",
            "native gate tolerance mismatch",
        )
    code, environment = _validate_code_and_environment(config)
    with tempfile.TemporaryDirectory(prefix="phase4-d2-protocol-b-verify-") as temporary:
        rebuilt_path = Path(temporary) / "rebuilt"
        summary = _rebuild_phase4_d2_protocol_b_eligibility_from_inputs(
            rebuilt_path,
            inputs=inputs,
            sweep=sweep,
            phase1_config=phase1_config,
            config=config,
            worker_count=worker_count,
            code=code,
            environment=environment,
        )
        _compare_tree(path, rebuilt_path)
    return Phase4D2ProtocolBEligibilitySummary(
        path=path,
        run_id=summary.run_id,
        status=summary.status,
        source_record_count=summary.source_record_count,
        model_cell_count=summary.model_cell_count,
        role_occurrence_count=summary.role_occurrence_count,
        operator_cell_count=summary.operator_cell_count,
        record_condition_count=summary.record_condition_count,
        class_summary_count=summary.class_summary_count,
    )


def verify_phase4_d2_protocol_b_eligibility(
    path: Path,
    *,
    worker_count: int = 12,
) -> Phase4D2ProtocolBEligibilitySummary:
    path = Path(path)
    if not path.is_dir():
        raise Phase4D2ProtocolBEligibilityVerifierError(
            "path",
            "must be an existing run directory",
        )
    raw = path.joinpath("config.json").read_bytes()
    document = json.loads(raw)
    config = parse_phase4_d2_protocol_b_eligibility_config(
        path / "config.json",
        raw,
        require_frozen_identity=not bool(document.get("synthetic_fixture", False)),
    )
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1_config = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    inputs = _reconstruct_real_inputs(
        ROOT / DATASET_RELATIVE_PATH,
        ROOT / SELECTION_RELATIVE_PATH,
        config,
    )
    return verify_phase4_d2_protocol_b_eligibility_from_inputs(
        path,
        inputs=inputs,
        sweep=sweep,
        phase1_config=phase1_config,
        config_path=path / "config.json",
        worker_count=worker_count,
    )


__all__ = [
    "verify_phase4_d2_protocol_b_eligibility",
    "verify_phase4_d2_protocol_b_eligibility_from_inputs",
]
