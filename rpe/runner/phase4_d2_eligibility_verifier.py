"""Independent verifier for Phase 4 D2 Protocol-A eligibility."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import multiprocessing
import os
import tempfile
import struct
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.evaluation import (
    PeakPairInput,
    SingleSpectrumInput,
    Spectrum1D,
    SpectrumPairInput,
    evaluate_metric,
)
from rpe.methods import load_classical_catalog
from rpe.methods.catalog import Phase3System, TaskLine
from rpe.methods.classical.peaks import PeakRunStatus, run_peak_detection_system
from rpe.metrics import (
    ISLikeStructureToNoiseMetric,
    MAEMetric,
    MSEMetric,
    NMSEMetric,
    PeakDetectionCurvesMetric,
    PearsonRMetric,
    RMSEMetric,
    SAMMetric,
    Wasserstein1Metric,
)
from rpe.perturb import PerturbationSweepConfig, load_perturbation_sweep_config
from rpe.runner.phase1_config import Phase1CoreConfig, load_phase1_core_config
from rpe.runner.phase1_perturbations import (
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus, Phase1Cell

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-d2-protocol-a-full-domain-eligibility-config-v1"
EXPERIMENT_ID = "phase4-d2-protocol-a-full-domain-eligibility-v1"
RUN_PREFIX = "phase4-d2-protocol-a-full-domain-eligibility-"
CONFIG_RELATIVE_PATH = (
    "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
)
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
CATALOG_RELATIVE_PATH = "experiments/phase3/configs/classical_system_catalog_v1.json"
SELECTION_RELATIVE_PATH = (
    "results/phase05/d2/"
    "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/"
    "selection.json"
)
DATASET_RELATIVE_PATH = "data/unified/bacteria_id_reference"
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d2_eligibility_authority.py"

ACTIVE_PERTURBATION_IDS = (
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
INACTIVE_PERTURBATION_IDS = ("p06", "p07")
ALL_PERTURBATION_IDS = tuple(f"p{index:02d}" for index in range(1, 13))
PEAK_PERTURBATION_IDS = ("p01", "p02", "p03", "p04", "p05")
FULL_DOMAIN_PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
METRIC_OUTPUT_IDS = (
    "mse",
    "rmse",
    "mae",
    "sam",
    "pearson_r",
    "nmse",
    "wasserstein_1_cm1",
    "is_like_structure_to_noise",
    "precision",
    "recall",
    "f1",
    "artifact_peak_ratio",
    "missing_peak_ratio",
)
CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
STRUCTURAL_REASON = "structurally_ineligible_missing_explicit_baseline"
FORBIDDEN_PHASE05_NAMES = frozenset(
    {
        "complete_cells.json",
    }
)
FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "accuracy",
        "acc_cross",
        "alignment_gap",
        "bootstrap",
        "correct",
        "figure",
        "metric_value",
        "peak_count",
        "peak_list",
        "peak_lists",
        "peaks",
        "prediction",
        "predictions",
        "selected_c",
    }
)
FORBIDDEN_KEY_FRAGMENTS = (
    "prediction",
    "metric_value",
    "peak_count",
    "peak_list",
    "bootstrap",
    "alignment",
    "figure",
)
ARTIFACT_STATIC_FILES = (
    "config.json",
    "source_records.jsonl",
    "model_cells.jsonl",
    "model_role_occurrences.jsonl",
    "cells.jsonl",
    "record_conditions.jsonl",
    "metric_statuses.jsonl",
    "cwt_receipts.jsonl",
    "class_summaries.jsonl",
    "common_support.jsonl",
    "gate.json",
    "manifest.json",
    "SHA256SUMS",
)
PAYLOAD_FILES = ARTIFACT_STATIC_FILES[:-1]
REQUIRED_CODE_AUTHORITY_PATHS = (
    "rpe/downstream/bacteria_id.py", "rpe/evaluation/contracts.py",
    "rpe/methods/classical/peaks.py",
    "rpe/metrics/fidelity.py", "rpe/metrics/peak.py",
    "rpe/metrics/reference_free.py", "rpe/metrics/transport.py",
    "rpe/perturb/contracts.py", "rpe/perturb/sweep.py",
    "rpe/runner/d2_selection.py", "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py", "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d2_eligibility.py",
    "rpe/runner/phase4_d2_eligibility_verifier.py",
    "tools/run_phase4_d2_eligibility.py",
)
REQUIRED_ENVIRONMENT_AUTHORITY = {
    "python": "3.13.11", "numpy": "2.5.2", "scipy": "1.18.0",
    "scikit_learn": "1.9.0", "h5py": "3.16.0",
    "threadpoolctl": "3.6.0", "system": "Linux", "machine": "x86_64",
}
REQUIRED_AUTHORITIES = {
    "parent_plan_sha256": "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5",
    "phase4_step01_sha256": "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f",
    "phase4_step11_sha256": "b2da0f26c60f930902c8b899091b4e1553bdbd160404b252608816785146ab18",
    "phase4_step12_design_sha256": "eeb2350ce6c359d84d9f1372962b8fcfb7fef7f623a77af18ae2e0abd03a4b38",
    "step13_plan_sha256": "d6d039a6381323aed5b14cac4a8a3a9a5dffc2b793141cf9f5646de2ea1447ae",
    "sweep_sha256": "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186",
    "phase1_core_config_sha256": "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd",
    "d2_selection_config_sha256": "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138",
    "d2_selection_artifact_path": SELECTION_RELATIVE_PATH,
    "d2_selection_artifact_sha256": "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36",
    "d2_phase05_runner_config_sha256": "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7",
    "d1_model_recipe_sha256": "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046",
    "classical_catalog_sha256": "8ad40b08df78b8905d75a67c84a2bb328531ef17cb12704ffad04f0a8f925d8f",
    "bacteria_id_sha256sums_sha256": "6d6d1399ac0e5197a9e51a1a0cf32edf51c924ce8d7c12aeb9d58f9af93c7f3e",
    "bacteria_id_retained_snapshot_sha256": "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02",
}
TRUST_ANCHOR = {"config_authority_relative_path": CONFIG_AUTHORITY_RELATIVE_PATH}


class Phase4D2EligibilityError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D2EligibilityConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    schema_version: str
    experiment_id: str
    synthetic_fixture: bool
    active_perturbation_ids: tuple[str, ...]
    inactive_perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_output_ids: tuple[str, ...]
    class_count: int
    test_record_count: int
    source_record_count: int
    finetune_union_record_count: int
    model_cell_count: int
    model_role_occurrence_count: int
    expected_active_cell_count: int
    expected_inactive_cell_count: int
    expected_cell_count: int
    expected_apply_call_count: int
    expected_condition_count: int
    expected_metric_status_count: int
    expected_cwt_receipt_count: int
    expected_class_summary_count: int
    gates: Mapping[str, float]
    authorities: Mapping[str, object]
    frozen_identities: Mapping[str, object]
    support_coordinates_cm1: tuple[float, ...]
    support_point_count: int
    support_max_gap_cm1: float
    p10_memory_budget_bytes: int
    claim_boundary: str


@dataclass(frozen=True)
class D2EligibilityInputs:
    source_records: tuple[Mapping[str, object], ...]
    model_cells: tuple[Mapping[str, object], ...]
    model_role_occurrences: tuple[Mapping[str, object], ...]
    test_spectra: tuple[Spectrum1D, ...]
    test_class_labels: tuple[int, ...]
    support_axis_cm1: np.ndarray
    test_record_ids_sha256: str
    native_axis_point_count: int
    support_axis_point_count: int


@dataclass(frozen=True)
class Phase4D2EligibilitySummary:
    path: Path
    run_id: str
    status: str
    test_record_count: int
    model_cell_count: int
    model_role_occurrence_count: int


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
    if is_dataclass(value):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, Enum):
        return value.value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Phase4D2EligibilityError("json", f"unsupported value {type(value).__name__}")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D2EligibilityError(path, "must be an object")
    return value


def _strings(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D2EligibilityError(path, "must be an array")
    output = tuple(str(item) for item in value)
    if tuple(dict.fromkeys(output)) != output:
        raise Phase4D2EligibilityError(path, "must be unique")
    return output


def _float(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D2EligibilityError(path, "must be numeric")
    current = float(value)
    if not math.isfinite(current):
        raise Phase4D2EligibilityError(path, "must be finite")
    return current


def _int(path: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Phase4D2EligibilityError(path, f"must be an integer >= {minimum}")
    return value


def _trusted_sha256_mapping(path: str, value: object) -> Mapping[str, str]:
    mapping = _object(path, value)
    for name, digest in mapping.items():
        if not isinstance(name, str) or not name:
            raise Phase4D2EligibilityError(path, "trust mapping keys must be nonempty strings")
        if not isinstance(digest, str) or len(digest) != 64:
            raise Phase4D2EligibilityError(f"{path}.{name}", "must be a SHA-256 hex digest")
        try:
            int(digest, 16)
        except ValueError as error:
            raise Phase4D2EligibilityError(f"{path}.{name}", "must be a SHA-256 hex digest") from error
    return mapping


def _authority_constants() -> tuple[int, str]:
    authority_path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    try:
        tree = ast.parse(
            authority_path.read_text(encoding="utf-8"),
            filename=str(authority_path),
        )
    except (OSError, SyntaxError) as error:
        raise Phase4D2EligibilityError("independent trust anchor", str(error)) from error
    values: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in {"CONFIG_BYTES", "CONFIG_SHA256"}
        ):
            values[node.targets[0].id] = ast.literal_eval(node.value)
    byte_count = values.get("CONFIG_BYTES")
    digest = values.get("CONFIG_SHA256")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
        or not isinstance(digest, str)
        or len(digest) != 64
    ):
        raise Phase4D2EligibilityError(
            "independent trust anchor", "CONFIG_BYTES/CONFIG_SHA256 missing"
        )
    return byte_count, digest


def _support_grid(config: Phase4D2EligibilityConfig) -> np.ndarray:
    values = np.asarray(config.support_coordinates_cm1, dtype="<f8")
    if values.size != config.support_point_count:
        raise Phase4D2EligibilityError("support_grid", "point count mismatch")
    return values


def _build_run_id(config: Phase4D2EligibilityConfig, inputs: D2EligibilityInputs) -> str:
    payload = (
        f"{config.sha256}\n"
        f"{inputs.test_record_ids_sha256}\n"
        f"{_canonical_json_bytes(config.authorities).decode('utf-8')}"
        f"{_canonical_json_bytes(config.frozen_identities).decode('utf-8')}"
        f"{config.model_cell_count}\n"
        f"{config.support_point_count}\n"
    ).encode("utf-8")
    return RUN_PREFIX + _sha_bytes(payload)


def parse_phase4_d2_eligibility_config(
    path: Path,
    raw: bytes,
    *,
    require_frozen_identity: bool,
) -> Phase4D2EligibilityConfig:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D2EligibilityError(str(path), str(error)) from error
    if raw != _canonical_json_bytes(document):
        raise Phase4D2EligibilityError(str(path), "must be canonical JSON")
    if require_frozen_identity:
        config_bytes, config_sha256 = _authority_constants()
        if len(raw) != config_bytes or _sha_bytes(raw) != config_sha256:
            raise Phase4D2EligibilityError(str(path), "frozen config identity mismatch")
    config = _object("config", document)
    schema_version = str(config.get("schema_version", ""))
    if schema_version != SCHEMA_VERSION:
        raise Phase4D2EligibilityError("schema_version", f"must equal {SCHEMA_VERSION!r}")
    experiment_id = str(config.get("experiment_id", ""))
    if experiment_id != EXPERIMENT_ID:
        raise Phase4D2EligibilityError("experiment_id", f"must equal {EXPERIMENT_ID!r}")
    active = _strings("active_perturbation_ids", config["active_perturbation_ids"])
    inactive = _strings("inactive_perturbation_ids", config["inactive_perturbation_ids"])
    if active != ACTIVE_PERTURBATION_IDS:
        raise Phase4D2EligibilityError("active_perturbation_ids", "must match the frozen D2 set")
    if inactive != INACTIVE_PERTURBATION_IDS:
        raise Phase4D2EligibilityError("inactive_perturbation_ids", "must match the frozen D2 set")
    alpha_grid = tuple(_float(f"alpha_grid[{index}]", value) for index, value in enumerate(config["alpha_grid"]))
    if alpha_grid != ALPHA_GRID:
        raise Phase4D2EligibilityError("alpha_grid", "must match the frozen alpha grid")
    metric_output_ids = _strings("metric_output_ids", config["metric_output_ids"])
    if metric_output_ids != METRIC_OUTPUT_IDS:
        raise Phase4D2EligibilityError("metric_output_ids", "must match the frozen metric manifest")

    denominators = _object("denominators", config["denominators"])
    class_count = _int("denominators.class_count", denominators["class_count"], minimum=1)
    test_record_count = _int("denominators.test_record_count", denominators["test_record_count"], minimum=1)
    source_record_count = _int("denominators.source_record_count", denominators["source_record_count"], minimum=1)
    finetune_union_record_count = _int(
        "denominators.finetune_union_record_count",
        denominators["finetune_union_record_count"],
        minimum=1,
    )
    model_cell_count = _int("denominators.model_cell_count", denominators["model_cell_count"], minimum=1)
    model_role_occurrence_count = _int(
        "denominators.model_role_occurrence_count",
        denominators["model_role_occurrence_count"],
        minimum=1,
    )
    if source_record_count < test_record_count:
        raise Phase4D2EligibilityError("denominators.source_record_count", "must include the test ledger")
    if source_record_count < finetune_union_record_count:
        raise Phase4D2EligibilityError("denominators.source_record_count", "must cover finetune union")

    synthetic_fixture = bool(config.get("synthetic_fixture", False))

    expected = _object("expected", config["expected"])
    expected_active_cell_count = _int("expected.active_cell_count", expected["active_cell_count"], minimum=1)
    expected_inactive_cell_count = _int("expected.inactive_cell_count", expected["inactive_cell_count"], minimum=1)
    expected_cell_count = _int("expected.cell_count", expected["cell_count"], minimum=1)
    expected_apply_call_count = _int("expected.apply_call_count", expected["apply_call_count"], minimum=1)
    expected_condition_count = _int("expected.condition_count", expected["condition_count"], minimum=1)
    expected_metric_status_count = _int(
        "expected.metric_status_count",
        expected["metric_status_count"],
        minimum=1,
    )
    expected_cwt_receipt_count = _int("expected.cwt_receipt_count", expected["cwt_receipt_count"], minimum=1)
    expected_class_summary_count = _int(
        "expected.class_summary_count",
        expected["class_summary_count"],
        minimum=1,
    )
    if expected_active_cell_count != test_record_count * len(ACTIVE_PERTURBATION_IDS):
        raise Phase4D2EligibilityError("expected.active_cell_count", "must equal test records x active perturbations")
    if expected_inactive_cell_count != test_record_count * len(INACTIVE_PERTURBATION_IDS):
        raise Phase4D2EligibilityError("expected.inactive_cell_count", "must equal test records x inactive perturbations")
    if expected_cell_count != test_record_count * len(ALL_PERTURBATION_IDS):
        raise Phase4D2EligibilityError("expected.cell_count", "must equal test records x all perturbations")
    if expected_apply_call_count != expected_active_cell_count * len(ALPHA_GRID):
        raise Phase4D2EligibilityError("expected.apply_call_count", "must equal active cells x alpha grid")
    expected_conditions = test_record_count * (1 + len(FULL_DOMAIN_PERTURBATION_IDS) * len(POSITIVE_ALPHAS))
    if expected_condition_count != expected_conditions:
        raise Phase4D2EligibilityError("expected.condition_count", "must match the full-domain condition grid")
    if expected_metric_status_count != expected_condition_count * len(METRIC_OUTPUT_IDS):
        raise Phase4D2EligibilityError("expected.metric_status_count", "must equal conditions x metric outputs")
    if expected_cwt_receipt_count != expected_condition_count:
        raise Phase4D2EligibilityError("expected.cwt_receipt_count", "must equal the condition grid")
    if expected_class_summary_count != class_count * len(ALL_PERTURBATION_IDS):
        raise Phase4D2EligibilityError("expected.class_summary_count", "must equal classes x perturbations")
    if not synthetic_fixture:
        seed_count = model_cell_count // 3
        expected_role_occurrence_count = (
            model_cell_count * test_record_count
            + seed_count * (150 + 300 + 600)
            + seed_count * 3 * 300
        )
    else:
        seed_count = model_cell_count // 3
        expected_role_occurrence_count = (
            model_cell_count * test_record_count
            + seed_count * class_count * (5 + 10 + 20)
            + seed_count * 3 * class_count * 10
        )
    if model_role_occurrence_count != expected_role_occurrence_count:
        raise Phase4D2EligibilityError(
            "model_role_occurrence_count",
            "must match the frozen 5/10/20-shot role ledger",
        )

    support_grid = _object("support_grid", config["support_grid"])
    support_point_count = _int("support_grid.point_count", support_grid["point_count"], minimum=1)
    support_max_gap_cm1 = _float(
        "support_grid.max_in_range_native_gap_cm1",
        support_grid["max_in_range_native_gap_cm1"],
    )
    if "coordinates_cm1" in support_grid:
        support_coordinates_cm1 = tuple(
            _float(f"support_grid.coordinates_cm1[{index}]", value)
            for index, value in enumerate(support_grid["coordinates_cm1"])
        )
    elif synthetic_fixture:
        support_start_cm1 = _float("support_grid.start_cm1", support_grid["start_cm1"])
        support_stop_cm1 = _float("support_grid.stop_cm1", support_grid["stop_cm1"])
        support_step_cm1 = _float("support_grid.step_cm1", support_grid["step_cm1"])
        if support_stop_cm1 < support_start_cm1:
            raise Phase4D2EligibilityError("support_grid", "must be increasing")
        if support_step_cm1 <= 0.0:
            raise Phase4D2EligibilityError("support_grid.step_cm1", "must be positive")
        support_coordinates_cm1 = tuple(
            np.arange(
                support_start_cm1,
                support_stop_cm1 + support_step_cm1 / 2.0,
                support_step_cm1,
                dtype="<f8",
            ).tolist()
        )
    else:
        raise Phase4D2EligibilityError(
            "support_grid.coordinates_cm1",
            "real frozen config must store literal support coordinates",
        )
    if len(support_coordinates_cm1) != support_point_count:
        raise Phase4D2EligibilityError("support_grid.point_count", "point count mismatch")
    if tuple(sorted(support_coordinates_cm1)) != support_coordinates_cm1:
        raise Phase4D2EligibilityError("support_grid.coordinates_cm1", "must be strictly increasing")

    p10 = _object("p10", config["p10"])
    p10_memory_budget_bytes = _int("p10.memory_budget_bytes", p10["memory_budget_bytes"], minimum=1)
    if p10.get("peak_estimate_formula") != "32*N^2+64*N+2^30":
        raise Phase4D2EligibilityError("p10.peak_estimate_formula", "must remain frozen")

    gates = {
        str(key): _float(f"gates.{key}", value)
        for key, value in _object("gates", config["gates"]).items()
    }
    authorities = MappingProxyType(dict(_object("authorities", config["authorities"])))
    artifact_payload_files = _strings(
        "artifact_payload_files", config.get("artifact_payload_files", ())
    )
    code_authority = _object("code_authority", config.get("code_authority", {}))
    environment_authority = _object(
        "environment_authority", config.get("environment_authority", {})
    )
    trust_anchor = _object("trust_anchor", config.get("trust_anchor", {}))
    if not synthetic_fixture:
        if artifact_payload_files != PAYLOAD_FILES:
            raise Phase4D2EligibilityError(
                "artifact_payload_files", "must be exact payload order through manifest"
            )
        if tuple(code_authority) != REQUIRED_CODE_AUTHORITY_PATHS:
            raise Phase4D2EligibilityError(
                "code_authority", "must have exact frozen authority paths and order"
            )
        for relative_path, receipt_value in code_authority.items():
            receipt = _object(f"code_authority.{relative_path}", receipt_value)
            target = ROOT / str(relative_path)
            if set(receipt) != {"bytes", "sha256"} or not target.is_file():
                raise Phase4D2EligibilityError(
                    f"code_authority.{relative_path}",
                    "invalid receipt or missing authority path",
                )
            if (
                _int(
                    f"code_authority.{relative_path}.bytes",
                    receipt["bytes"],
                    minimum=1,
                )
                != target.stat().st_size
                or str(receipt["sha256"]) != _sha_file(target)
            ):
                raise Phase4D2EligibilityError(
                    f"code_authority.{relative_path}", "bytes or SHA-256 mismatch"
                )
        if dict(environment_authority) != REQUIRED_ENVIRONMENT_AUTHORITY:
            raise Phase4D2EligibilityError(
                "environment_authority", "must match frozen runtime environment"
            )
        if dict(trust_anchor) != TRUST_ANCHOR:
            raise Phase4D2EligibilityError(
                "trust_anchor", "must bind production authority module"
            )
        if dict(authorities) != REQUIRED_AUTHORITIES:
            raise Phase4D2EligibilityError(
                "authorities", "must contain exact frozen authority identities"
            )
    elif artifact_payload_files or code_authority or environment_authority or trust_anchor:
        raise Phase4D2EligibilityError(
            "synthetic authority",
            "synthetic fixtures may only use empty authority mappings",
        )
    frozen_identities = MappingProxyType(dict(_object("frozen_identities", config["frozen_identities"])))
    claim_boundary = str(config["claim_boundary"])
    return Phase4D2EligibilityConfig(
        path=Path(path),
        raw_bytes=raw,
        sha256=_sha_bytes(raw),
        schema_version=schema_version,
        experiment_id=experiment_id,
        synthetic_fixture=synthetic_fixture,
        active_perturbation_ids=active,
        inactive_perturbation_ids=inactive,
        alpha_grid=alpha_grid,
        metric_output_ids=metric_output_ids,
        class_count=class_count,
        test_record_count=test_record_count,
        source_record_count=source_record_count,
        finetune_union_record_count=finetune_union_record_count,
        model_cell_count=model_cell_count,
        model_role_occurrence_count=model_role_occurrence_count,
        expected_active_cell_count=expected_active_cell_count,
        expected_inactive_cell_count=expected_inactive_cell_count,
        expected_cell_count=expected_cell_count,
        expected_apply_call_count=expected_apply_call_count,
        expected_condition_count=expected_condition_count,
        expected_metric_status_count=expected_metric_status_count,
        expected_cwt_receipt_count=expected_cwt_receipt_count,
        expected_class_summary_count=expected_class_summary_count,
        gates=MappingProxyType(gates),
        authorities=authorities,
        frozen_identities=frozen_identities,
        support_coordinates_cm1=support_coordinates_cm1,
        support_point_count=support_point_count,
        support_max_gap_cm1=support_max_gap_cm1,
        p10_memory_budget_bytes=p10_memory_budget_bytes,
        claim_boundary=claim_boundary,
    )


def load_phase4_d2_eligibility_config(path: Path) -> Phase4D2EligibilityConfig:
    raw = Path(path).read_bytes()
    return parse_phase4_d2_eligibility_config(path, raw, require_frozen_identity=True)


def _load_json_document(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D2EligibilityError(str(path), str(error)) from error
    if raw != _canonical_json_bytes(value):
        raise Phase4D2EligibilityError(str(path), "must be canonical JSON")
    return _object(str(path), value)


def _reverse_native_record(record: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    stored_axis = np.asarray(record["stored_axis_cm1"], dtype="<f4")
    stored_intensity = np.asarray(record["stored_intensity"], dtype="<f4")
    if stored_axis.ndim != 1 or stored_intensity.ndim != 1 or stored_axis.size != stored_intensity.size:
        raise Phase4D2EligibilityError("test_records", "axis and intensity must be one-dimensional with equal length")
    if not np.all(np.diff(stored_axis) < 0.0):
        raise Phase4D2EligibilityError("test_records.stored_axis_cm1", "must be strictly decreasing")
    axis = stored_axis[::-1].astype("<f8")
    intensity = stored_intensity[::-1].astype("<f8")
    if not np.all(np.diff(axis) > 0.0):
        raise Phase4D2EligibilityError("test_records.axis_cm1", "reversed axis must be strictly increasing")
    return axis, intensity


def reconstruct_d2_eligibility_inputs(
    dataset_path: Path,
    selection_path: Path,
    config: Phase4D2EligibilityConfig,
) -> D2EligibilityInputs:
    dataset_path = Path(dataset_path)
    selection_path = Path(selection_path)
    if selection_path.name in FORBIDDEN_PHASE05_NAMES or selection_path.name.startswith("seed") or selection_path.name.startswith("predictions_"):
        raise Phase4D2EligibilityError(str(selection_path), "forbidden Phase-0.5 outcome artifact")
    dataset = _load_json_document(dataset_path)
    selection = _load_json_document(selection_path)
    test_records_raw = selection.get("test", {})
    if not isinstance(test_records_raw, Mapping):
        raise Phase4D2EligibilityError("selection.test", "must be an object")
    test_records = tuple(_object("test_record", row) for row in dataset["test_records"])
    native_axis_reference = np.asarray(dataset["native_axis_cm1_decreasing_f32"], dtype="<f4")
    support_axis = _support_grid(config)

    test_spectra = []
    source_records: list[Mapping[str, object]] = []
    test_class_labels: list[int] = []
    test_record_ids: list[str] = []
    for row in test_records:
        axis, intensity = _reverse_native_record(row)
        if not np.array_equal(native_axis_reference[::-1].astype("<f8"), axis):
            raise Phase4D2EligibilityError("dataset.native_axis_cm1_decreasing_f32", "must bind the shared decreasing source axis")
        record_id = str(row["record_id"])
        class_label = _int("class_label", row["class_label"], minimum=0)
        test_record_ids.append(record_id)
        test_class_labels.append(class_label)
        spectrum = Spectrum1D(
            spectrum_id=f"bacteria_id_reference::{record_id}",
            sample_id=None,
            axis_cm1=axis,
            intensity=intensity,
        )
        test_spectra.append(spectrum)
        source_records.append(
            {
                "class_label": class_label,
                "native_axis_sha256": _sha_bytes(np.asarray(axis, dtype="<f8").tobytes(order="C")),
                "record_id": record_id,
                "role_count": 1,
                "scope": "test",
                "support_axis_sha256": _sha_bytes(np.asarray(support_axis, dtype="<f8").tobytes(order="C")),
            }
        )

    seen_finetune_ids: set[str] = set()
    model_cells: list[Mapping[str, object]] = []
    model_role_occurrences: list[Mapping[str, object]] = []
    for seed_document in selection["selections"]:
        seed = _int("selection.seed", seed_document["seed"], minimum=0)
        validation_ids: list[str] = []
        train_by_shot: dict[str, list[str]] = {str(shot): [] for shot in (5, 10, 20)}
        for class_document in seed_document["classes"]:
            validation = sorted(str(value) for value in class_document["validation_record_ids"])
            validation_ids.extend(validation)
            seen_finetune_ids.update(validation)
            train_map = _object("train_record_ids", class_document["train_record_ids"])
            for shot_key, values in train_map.items():
                selected = sorted(str(value) for value in values)
                train_by_shot[str(shot_key)].extend(selected)
                seen_finetune_ids.update(selected)
        validation_ids = sorted(validation_ids)
        for shot_key in ("5", "10", "20"):
            train_ids = tuple(sorted(train_by_shot[shot_key]))
            validation_tuple = tuple(validation_ids)
            if set(train_ids) & set(validation_tuple):
                raise Phase4D2EligibilityError("selection", "train/validation overlap within one cell")
            model_cell = {
                "seed": seed,
                "shot_count": int(shot_key),
                "train_record_ids": train_ids,
                "validation_record_ids": validation_tuple,
            }
            model_cells.append(model_cell)
            for record_id in train_ids:
                model_role_occurrences.append(
                    {
                        "record_id": record_id,
                        "role": "train",
                        "seed": seed,
                        "shot_count": int(shot_key),
                    }
                )
            for record_id in validation_tuple:
                model_role_occurrences.append(
                    {
                        "record_id": record_id,
                        "role": "validation",
                        "seed": seed,
                        "shot_count": int(shot_key),
                    }
                )
            for record_id in test_record_ids:
                model_role_occurrences.append(
                    {
                        "record_id": record_id,
                        "role": "test",
                        "seed": seed,
                        "shot_count": int(shot_key),
                    }
                )

    for record_id in sorted(seen_finetune_ids):
        source_records.append(
            {
                "class_label": None,
                "native_axis_sha256": None,
                "record_id": record_id,
                "role_count": sum(1 for row in model_role_occurrences if row["record_id"] == record_id),
                "scope": "finetune_union",
                "support_axis_sha256": None,
            }
        )
    if len(model_cells) != config.model_cell_count:
        raise Phase4D2EligibilityError("model_cells", "count mismatch")
    if len(model_role_occurrences) != config.model_role_occurrence_count:
        raise Phase4D2EligibilityError("model_role_occurrence_count", "count mismatch")
    if len(source_records) != config.source_record_count:
        raise Phase4D2EligibilityError("source_record_count", "count mismatch")
    expected_sha = str(test_records_raw.get("record_ids_sha256", ""))
    observed_sha = _sha_bytes(("\n".join(test_record_ids) + "\n").encode("utf-8"))
    if expected_sha and expected_sha != observed_sha:
        raise Phase4D2EligibilityError("selection.test.record_ids_sha256", "mismatch")
    return D2EligibilityInputs(
        source_records=tuple(source_records),
        model_cells=tuple(model_cells),
        model_role_occurrences=tuple(model_role_occurrences),
        test_spectra=tuple(test_spectra),
        test_class_labels=tuple(test_class_labels),
        support_axis_cm1=support_axis,
        test_record_ids_sha256=observed_sha,
        native_axis_point_count=int(test_spectra[0].axis_cm1.size) if test_spectra else 0,
        support_axis_point_count=int(support_axis.size),
    )


def project_d2_support(spectrum: Spectrum1D, config: Phase4D2EligibilityConfig) -> np.ndarray:
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    intensity = np.asarray(spectrum.intensity, dtype="<f8")
    support = _support_grid(config)
    if axis.ndim != 1 or intensity.ndim != 1 or axis.size != intensity.size:
        raise Phase4D2EligibilityError("spectrum", "axis and intensity must be aligned one-dimensional arrays")
    if float(axis[0]) > float(support[0]) or float(axis[-1]) < float(support[-1]):
        raise Phase4D2EligibilityError("support_projection", "extrapolation required")
    left = int(np.searchsorted(axis, support[0], side="left"))
    right = int(np.searchsorted(axis, support[-1], side="right"))
    native_in_range = axis[left:right]
    if native_in_range.size < 2:
        raise Phase4D2EligibilityError("support_projection", "insufficient in-range native support")
    if float(np.max(np.diff(native_in_range))) > config.support_max_gap_cm1:
        raise Phase4D2EligibilityError("max_in_range_native_gap_cm1", "support gap exceeds the frozen gate")
    index_map = {float(value): index for index, value in enumerate(axis)}
    if all(float(value) in index_map for value in support):
        selected = np.asarray(
            [intensity[index_map[float(value)]] for value in support],
            dtype="<f8",
        )
    else:
        selected = np.interp(support, axis, intensity).astype("<f8")
    output = np.ascontiguousarray(selected, dtype="<f4")
    if output.size != config.support_point_count:
        raise Phase4D2EligibilityError("support_projection", "point count mismatch")
    if not np.isfinite(output).all():
        raise Phase4D2EligibilityError("support_projection", "must be finite")
    if float(np.linalg.norm(output.astype(np.float64))) <= 0.0:
        raise Phase4D2EligibilityError("support_projection", "zero norm")
    return output


def evaluate_d2_eligibility_gates(
    cells: Sequence[Mapping[str, object]],
    class_labels: Sequence[int],
    config: Phase4D2EligibilityConfig,
    *,
    record_conditions: Sequence[Mapping[str, object]] | None = None,
    metric_statuses: Sequence[Mapping[str, object]] | None = None,
    cwt_receipts: Sequence[Mapping[str, object]] | None = None,
) -> Mapping[str, object]:
    per_perturbation: dict[str, dict[str, object]] = {}
    unique_classes = tuple(sorted(set(int(value) for value in class_labels)))
    unique_records = tuple(sorted({str(row["record_id"]) for row in cells}))
    by_perturbation_record: dict[str, dict[str, str]] = {pid: {} for pid in ALL_PERTURBATION_IDS}
    by_perturbation_class_records: dict[str, dict[int, set[str]]] = {
        pid: {class_label: set() for class_label in unique_classes}
        for pid in ALL_PERTURBATION_IDS
    }
    state_counts_by_perturbation: dict[str, dict[str, int]] = {
        pid: {"complete": 0, "not_applicable": 0, "failed_runtime": 0, "structurally_ineligible": 0}
        for pid in ALL_PERTURBATION_IDS
    }
    class_by_record = {str(row["record_id"]): int(row["class_label"]) for row in cells}
    for row in cells:
        perturbation_id = str(row["perturbation_id"])
        state = str(row["state"])
        record_id = str(row["record_id"])
        by_perturbation_record[perturbation_id][record_id] = state
        state_counts_by_perturbation[perturbation_id][state] += 1
        if state == "complete":
            by_perturbation_class_records[perturbation_id][class_by_record[record_id]].add(record_id)
    full_domain_states = {}
    for perturbation_id in FULL_DOMAIN_PERTURBATION_IDS:
        complete_records = sum(
            1 for record_id in unique_records if by_perturbation_record[perturbation_id].get(record_id) == "complete"
        )
        complete_classes = sum(
            1
            for class_label in unique_classes
            if len(by_perturbation_class_records[perturbation_id][class_label]) == sum(1 for value in class_labels if value == class_label)
        )
        state = "evaluable"
        if (
            complete_records != len(unique_records)
            or complete_classes != len(unique_classes)
            or state_counts_by_perturbation[perturbation_id]["not_applicable"] != 0
            or state_counts_by_perturbation[perturbation_id]["failed_runtime"] != 0
        ):
            state = "not_evaluable_coverage"
        full_domain_states[perturbation_id] = {
            "complete_class_count": complete_classes,
            "complete_record_count": complete_records,
            "state": state,
            "state_counts": dict(state_counts_by_perturbation[perturbation_id]),
        }

    peak_states = {}
    peak_complete_sets = []
    for perturbation_id in PEAK_PERTURBATION_IDS:
        complete_records = {
            record_id
            for record_id in unique_records
            if by_perturbation_record[perturbation_id].get(record_id) == "complete"
        }
        peak_complete_sets.append(complete_records)
        threshold = 0.9 if perturbation_id == "p05" else 0.95
        complete_classes = sum(
            1
            for class_label in unique_classes
            if len(by_perturbation_class_records[perturbation_id][class_label]) == sum(1 for value in class_labels if value == class_label)
        )
        minimum_records = math.ceil(len(unique_records) * threshold)
        minimum_classes = math.ceil(len(unique_classes) * threshold)
        state = "evaluable"
        if (
            len(complete_records) < minimum_records
            or complete_classes < minimum_classes
            or state_counts_by_perturbation[perturbation_id]["failed_runtime"] != 0
        ):
            state = "not_evaluable_coverage"
        peak_states[perturbation_id] = {
            "complete_class_count": complete_classes,
            "complete_record_count": len(complete_records),
            "state": state,
            "state_counts": dict(state_counts_by_perturbation[perturbation_id]),
        }
    common_support = set.intersection(*peak_complete_sets) if peak_complete_sets else set()
    common_classes = sum(
        1
        for class_label in unique_classes
        if all(
            record_id in common_support
            for record_id, row_class in ((record_id, class_by_record[record_id]) for record_id in unique_records)
            if row_class == class_label
        )
    )
    peak_common_state = "evaluable"
    if (
        len(common_support) < math.ceil(len(unique_records) * config.gates["peak_common_record_fraction"])
        or common_classes < math.ceil(len(unique_classes) * config.gates["peak_common_class_fraction"])
    ):
        peak_common_state = "not_evaluable_coverage"
    def grid_state(rows: Sequence[Mapping[str, object]], planned: int) -> Mapping[str, object]:
        complete = sum(1 for row in rows if row.get("state") == "complete")
        failed = planned - complete
        return {
            "planned": planned,
            "complete": complete,
            "failed": failed,
            "state": "evaluable" if complete == planned else "not_evaluable_incomplete_grid",
        }

    execution_conditions = tuple(
        row for row in (record_conditions or ())
        if (
            str(row.get("condition_id", "")) == "alpha0"
            or str(row.get("condition_id", "")).split(":", 1)[0]
            in FULL_DOMAIN_PERTURBATION_IDS
        )
    )
    support_grid = grid_state(execution_conditions, len(execution_conditions))
    metric_outputs = {
        output_id: grid_state(
            tuple(row for row in (metric_statuses or ()) if row.get("metric_output_id") == output_id),
            len(execution_conditions),
        )
        for output_id in METRIC_OUTPUT_IDS
    }
    cwt = grid_state(
        tuple(
            row for row in (cwt_receipts or ())
            if (
                str(row.get("condition_id", "")) == "alpha0"
                or str(row.get("condition_id", "")).split(":", 1)[0]
                in FULL_DOMAIN_PERTURBATION_IDS
            )
        ),
        len(execution_conditions),
    )
    has_execution_grids = record_conditions is not None
    full_domain_state = "evaluable" if (
        all(value["state"] == "evaluable" for value in full_domain_states.values())
        and (not has_execution_grids or (support_grid["state"] == "evaluable" and metric_outputs["mse"]["state"] == "evaluable"))
    ) else "not_evaluable_coverage"
    if has_execution_grids:
        for output_id in METRIC_OUTPUT_IDS[8:]:
            if cwt["state"] != "evaluable":
                metric_outputs[output_id] = {
                    **metric_outputs[output_id],
                    "state": "not_evaluable_incomplete_grid",
                }
    overall_status = "pass" if full_domain_state == "evaluable" and peak_common_state == "evaluable" else "fail"
    return MappingProxyType(
        {
            "full_domain_core": {
                "by_perturbation": full_domain_states,
                "state": full_domain_state,
            },
            "support_grid": support_grid,
            "metric_outputs": metric_outputs,
            "cwt": cwt,
            "marker_filename": "complete.json" if overall_status == "pass" else "failed.json",
            "overall_status": overall_status,
            "peak_common_support": {
                "common_class_count": common_classes,
                "common_record_count": len(common_support),
                "p01": peak_states["p01"],
                "p02": peak_states["p02"],
                "p03": peak_states["p03"],
                "p04": peak_states["p04"],
                "p05": peak_states["p05"],
                "state": peak_common_state,
            },
        }
    )


def validate_outcome_blind_d2_payload(value: object) -> None:
    def walk(path: str, current: object) -> None:
        if isinstance(current, Mapping):
            for key, item in current.items():
                name = str(key)
                if name in FORBIDDEN_PAYLOAD_KEYS or (
                    not name.endswith("_sha256")
                    and any(fragment in name for fragment in FORBIDDEN_KEY_FRAGMENTS)
                ):
                    raise Phase4D2EligibilityError(path, name)
                walk(f"{path}.{name}", item)
        elif isinstance(current, (list, tuple)):
            for index, item in enumerate(current):
                walk(f"{path}[{index}]", item)

    walk("payload", value)


def _array_sha256(value: np.ndarray, *, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _phase1_source_for_d2(
    spectrum: Spectrum1D, *, record_order: int, record_id: str, class_label: int
) -> Phase1Source:
    axis_f4 = np.asarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.asarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=record_order, record_id=record_id, sample_id=record_id,
            class_label=class_label, mineral_name=f"bacteria-{class_label}",
            axis_id=f"native::{_array_sha256(spectrum.axis_cm1)}",
        ),
        spectrum=spectrum, original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha256(axis_f4, dtype="<f4"),
        source_intensity_float32_sha256=_array_sha256(intensity_f4, dtype="<f4"),
        normalized_axis_float64_sha256=_array_sha256(spectrum.axis_cm1),
        normalized_intensity_float64_sha256=_array_sha256(spectrum.intensity),
        provenance=MappingProxyType({
            "license": None, "license_status": "not_stated",
            "retrieved_date": "2026-08-20", "sha256": "0" * 64,
            "source_artifact": "bacteria_id_reference", "source_url": "local://bacteria_id_reference",
        }),
    )


def _exception_payload(cell: Phase1Cell) -> Mapping[str, object] | None:
    evidence = cell.evidence
    if evidence.status is CellStatus.COMPLETE:
        return None
    return {
        "message": evidence.exception_message, "path": evidence.exception_path,
        "type": evidence.exception_type,
    }


def _classify_real_cell(cell: Phase1Cell) -> str:
    if cell.status is CellStatus.COMPLETE:
        return "complete"
    if cell.status is CellStatus.NOT_APPLICABLE:
        return "not_applicable"
    return "failed_runtime"


def _receipt_hash(value: object) -> str:
    return _sha_bytes(_canonical_json_bytes(value))


def _perturbation_result_hash(record: object) -> str:
    result = record.result
    return _receipt_hash({
        "alpha_float64_le_hex": record.alpha_float64_le_hex,
        "axis_behavior": result.axis_behavior.value,
        "axis_changed": result.axis_changed,
        "diagnostics": result.diagnostics,
        "intensity_changed": result.intensity_changed,
        "output_axis_sha256": _array_sha256(result.output.axis_cm1),
        "output_intensity_sha256": _array_sha256(result.output.intensity),
        "output_spectrum_id": result.output.spectrum_id,
        "perturbation_id": result.perturbation_id,
        "source_spectrum_id": result.source_spectrum_id,
        "state_digest": result.state_digest,
    })


def _metric_objects() -> Mapping[str, object]:
    return {
        "mse": MSEMetric(), "rmse": RMSEMetric(), "mae": MAEMetric(),
        "sam": SAMMetric(), "pearson_r": PearsonRMetric(), "nmse": NMSEMetric(),
        "wasserstein_1_cm1": Wasserstein1Metric(),
        "is_like_structure_to_noise": ISLikeStructureToNoiseMetric(),
    }


def _cwt_receipt_row(record_id: str, condition_id: str, receipt: object | None, state: str) -> Mapping[str, object]:
    if receipt is None:
        return {"condition_id": condition_id, "diagnostics_sha256": None, "peak_list_sha256": None, "record_id": record_id, "state": state, "warning_sha256": None}
    return {
        "condition_id": condition_id,
        "diagnostics_sha256": _receipt_hash(dict(receipt.diagnostics)),
        "peak_list_sha256": receipt.peaks_sha256,
        "record_id": record_id, "state": receipt.status.value,
        "warning_sha256": _receipt_hash(receipt.warnings),
    }


def _structure_metric_rows(
    *,
    record_id: str,
    condition_id: str,
    reference_receipt: object,
    candidate_receipt: object,
    state: str,
) -> tuple[Mapping[str, object], ...]:
    if state != "complete" or candidate_receipt is None:
        return tuple(
            {
                "condition_id": condition_id,
                "diagnostics_sha256": None,
                "metric_output_id": output_id,
                "record_id": record_id,
                "result_sha256": None,
                "state": state,
            }
            for output_id in METRIC_OUTPUT_IDS[8:]
        )
    try:
        result = evaluate_metric(
            PeakDetectionCurvesMetric(),
            PeakPairInput(
                reference_peaks=tuple(peak.to_peak1d() for peak in reference_receipt.peaks),
                candidate_peaks=tuple(peak.to_peak1d() for peak in candidate_receipt.peaks),
                position_tolerance_cm1=2.0,
                prominence_thresholds=(0.0,),
            ),
        )
    except Exception as error:
        exception = {"type": type(error).__name__, "message": str(error)}
        return tuple(
            {
                "condition_id": condition_id,
                "diagnostics_sha256": _receipt_hash(exception),
                "metric_output_id": output_id,
                "record_id": record_id,
                "result_sha256": None,
                "state": "failed_runtime",
            }
            for output_id in METRIC_OUTPUT_IDS[8:]
        )
    result_sha = _receipt_hash(result)
    diagnostics_sha = _receipt_hash(result.diagnostics)
    return tuple(
        {
            "condition_id": condition_id,
            "diagnostics_sha256": diagnostics_sha,
            "metric_output_id": output_id,
            "record_id": record_id,
            "result_sha256": result_sha,
            "state": "complete",
        }
        for output_id in METRIC_OUTPUT_IDS[8:]
    )


def _scalar_metric_rows(
    *,
    record_id: str,
    condition_id: str,
    source: Spectrum1D,
    output: Spectrum1D | None,
    state: str,
) -> tuple[Mapping[str, object], ...]:
    if output is None:
        return tuple(
            {
                "condition_id": condition_id,
                "diagnostics_sha256": None,
                "metric_output_id": output_id,
                "record_id": record_id,
                "result_sha256": None,
                "state": state,
            }
            for output_id in METRIC_OUTPUT_IDS[:8]
        )
    rows = []
    for output_id, metric in _metric_objects().items():
        request = (
            SingleSpectrumInput(output)
            if output_id == "is_like_structure_to_noise"
            else SpectrumPairInput(source, output)
        )
        try:
            result = evaluate_metric(metric, request)
            row = {
                "condition_id": condition_id,
                "diagnostics_sha256": _receipt_hash(result.diagnostics),
                "metric_output_id": output_id,
                "record_id": record_id,
                "result_sha256": _receipt_hash(result),
                "state": "complete",
            }
        except Exception as error:
            row = {
                "condition_id": condition_id,
                "diagnostics_sha256": _receipt_hash({"message": str(error), "type": type(error).__name__}),
                "metric_output_id": output_id,
                "record_id": record_id,
                "result_sha256": None,
                "state": "failed_runtime",
            }
        rows.append(row)
    return tuple(rows)


def _execute_d2_record(
    *,
    record_order: int,
    spectrum: Spectrum1D,
    class_label: int,
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D2EligibilityConfig,
    admission: P10MemoryAdmission,
    cwt_system: Phase3System,
) -> Mapping[str, object]:
    record_id = spectrum.spectrum_id.split("::", 1)[1]
    source = _phase1_source_for_d2(
        spectrum, record_order=record_order, record_id=record_id, class_label=class_label
    )
    cells: list[Mapping[str, object]] = []
    conditions: list[Mapping[str, object]] = []
    metrics: list[Mapping[str, object]] = []
    cwt_rows: list[Mapping[str, object]] = []
    try:
        reference_receipt = run_peak_detection_system(cwt_system, spectrum)
    except Exception as error:
        reference_receipt = None
        reference_state = "failed_runtime"
        reference_failure = _receipt_hash({"message": str(error), "type": type(error).__name__})
    else:
        reference_state = "complete"
        reference_failure = None
    alpha0_sha = _sha_bytes(project_d2_support(spectrum, config).tobytes(order="C"))
    alpha_zero_failures: list[str] = []
    conditions.append(
        {
            "class_label": class_label,
            "condition_id": "alpha0",
            "output_axis_sha256": _array_sha256(spectrum.axis_cm1),
            "output_intensity_sha256": _array_sha256(spectrum.intensity),
            "record_id": record_id,
            "result_sha256": alpha0_sha,
            "support_sha256": alpha0_sha,
            "state": "complete",
        }
    )
    cwt_rows.append(
        _cwt_receipt_row(record_id, "alpha0", reference_receipt, reference_state)
        if reference_receipt is not None
        else {
            "condition_id": "alpha0",
            "diagnostics_sha256": reference_failure,
            "peak_list_sha256": None,
            "record_id": record_id,
            "state": reference_state,
            "warning_sha256": None,
        }
    )
    metrics.extend(
        _scalar_metric_rows(
            record_id=record_id,
            condition_id="alpha0",
            source=spectrum,
            output=spectrum,
            state="complete",
        )
    )
    metrics.extend(
        _structure_metric_rows(
            record_id=record_id,
            condition_id="alpha0",
            reference_receipt=reference_receipt,
            candidate_receipt=reference_receipt,
            state=reference_state,
        )
        if reference_receipt is not None
        else _structure_metric_rows(
            record_id=record_id,
            condition_id="alpha0",
            reference_receipt=None,
            candidate_receipt=None,
            state=reference_state,
        )
    )
    for perturbation_id in ALL_PERTURBATION_IDS:
        if perturbation_id in INACTIVE_PERTURBATION_IDS:
            cells.append(
                {
                    "class_label": class_label,
                    "exception": None,
                    "native_gate": {},
                    "outputs": [],
                    "p10_estimated_peak_bytes": None,
                    "perturbation_id": perturbation_id,
                    "reason_code": STRUCTURAL_REASON,
                    "record_id": record_id,
                    "state": "structurally_ineligible",
                    "state_digest": None,
                }
            )
            continue
        with threadpool_limits(limits=1, user_api="blas"):
            cell = run_perturbation_cell(
                source,
                perturbation_id,
                phase1_config,
                sweep,
                p10_admission=admission if perturbation_id == "p10" else None,
            )
        state = _classify_real_cell(cell)
        support_hashes: dict[str, str] = {}
        support_failure: Mapping[str, object] | None = None
        if state == "complete":
            alpha_zero = next((record for record in cell.records if record.alpha == 0.0), None)
            if alpha_zero is None or not (
                np.array_equal(alpha_zero.result.output.axis_cm1, spectrum.axis_cm1)
                and np.array_equal(alpha_zero.result.output.intensity, spectrum.intensity)
            ):
                state = "failed_runtime"
                alpha_zero_failures.append(perturbation_id)
        if state == "complete":
            try:
                support_hashes = {
                    record.alpha_float64_le_hex: _sha_bytes(
                        project_d2_support(record.result.output, config).tobytes(order="C")
                    )
                    for record in cell.records
                }
            except Exception as error:
                state = "failed_runtime"
                support_failure = {
                    "message": str(error),
                    "path": getattr(error, "path", "support_projection"),
                    "type": type(error).__name__,
                }
        cells.append(
            {
                "class_label": class_label,
                "exception": support_failure if support_failure is not None else _exception_payload(cell),
                "native_gate": dict(cell.evidence.native_gate),
                "outputs": [
                    {
                        "alpha_float64_le_hex": record.alpha_float64_le_hex,
                        "diagnostics_sha256": _receipt_hash(record.result.diagnostics),
                        "output_axis_sha256": _array_sha256(record.result.output.axis_cm1),
                        "output_intensity_sha256": _array_sha256(record.result.output.intensity),
                        "output_spectrum_id": record.result.output.spectrum_id,
                        "support_sha256": support_hashes[record.alpha_float64_le_hex],
                    }
                    for record in cell.records
                ] if state == "complete" else [],
                "p10_estimated_peak_bytes": estimate_p10_peak_bytes(spectrum.axis_cm1.size) if perturbation_id == "p10" else None,
                "perturbation_id": perturbation_id,
                "reason_code": cell.reason_code,
                "record_id": record_id,
                "state": state,
                "state_digest": None if cell.state is None else cell.state.state_digest,
            }
        )
        if perturbation_id not in FULL_DOMAIN_PERTURBATION_IDS:
            continue
        by_alpha = {record.alpha: record for record in cell.records}
        for alpha in POSITIVE_ALPHAS:
            condition_id = f"{perturbation_id}:{np.float64(alpha).tobytes(order='C').hex()}"
            perturbed = by_alpha.get(alpha)
            output = None if perturbed is None or state != "complete" else perturbed.result.output
            condition_state = state
            support_sha = (
                None
                if perturbed is None or state != "complete"
                else support_hashes.get(perturbed.alpha_float64_le_hex)
            )
            conditions.append(
                {
                    "class_label": class_label,
                    "condition_id": condition_id,
                    "output_axis_sha256": None if output is None else _array_sha256(output.axis_cm1),
                    "output_intensity_sha256": None if output is None else _array_sha256(output.intensity),
                    "record_id": record_id,
                    "result_sha256": None if perturbed is None or output is None else _perturbation_result_hash(perturbed),
                    "support_sha256": support_sha,
                    "state": condition_state,
                }
            )
            try:
                candidate = None if output is None else run_peak_detection_system(cwt_system, output)
            except Exception as error:
                candidate = None
                candidate_state = "failed_runtime"
                candidate_failure = _receipt_hash({"message": str(error), "type": type(error).__name__})
            else:
                candidate_state = condition_state
                candidate_failure = None
            cwt_rows.append(
                _cwt_receipt_row(record_id, condition_id, candidate, candidate_state)
                if candidate is not None
                else {
                    "condition_id": condition_id,
                    "diagnostics_sha256": candidate_failure,
                    "peak_list_sha256": None,
                    "record_id": record_id,
                    "state": candidate_state,
                    "warning_sha256": None,
                }
            )
            metrics.extend(
                _scalar_metric_rows(
                    record_id=record_id,
                    condition_id=condition_id,
                    source=spectrum,
                    output=output,
                    state=condition_state,
                )
            )
            metrics.extend(
                _structure_metric_rows(
                    record_id=record_id,
                    condition_id=condition_id,
                    reference_receipt=reference_receipt,
                    candidate_receipt=candidate,
                    state=candidate_state,
                )
                if reference_receipt is not None
                else _structure_metric_rows(
                    record_id=record_id,
                    condition_id=condition_id,
                    reference_receipt=None,
                    candidate_receipt=None,
                    state="failed_runtime",
                )
            )
    if alpha_zero_failures:
        conditions[0] = {**conditions[0], "state": "failed_runtime", "support_sha256": None}
        metrics = [
            {**row, "state": "failed_runtime", "result_sha256": None}
            if row["condition_id"] == "alpha0"
            else row
            for row in metrics
        ]
    return {
        "record_order": record_order,
        "cells": tuple(cells),
        "record_conditions": tuple(conditions),
        "metric_statuses": tuple(metrics),
        "cwt_receipts": tuple(cwt_rows),
        "class_label": class_label,
        "record_id": record_id,
        "_worker_pid": os.getpid(),
    }


_PROCESS_SWEEP: PerturbationSweepConfig | None = None
_PROCESS_PHASE1_CONFIG: Phase1CoreConfig | None = None
_PROCESS_D2_CONFIG: Phase4D2EligibilityConfig | None = None
_PROCESS_CWT_SYSTEM: Phase3System | None = None


def _bounded_process_count(
    *,
    requested_workers: int,
    point_counts: Sequence[int],
    memory_budget_bytes: int,
) -> int:
    max_p10_estimate = max(estimate_p10_peak_bytes(int(value)) for value in point_counts)
    capacity = int(memory_budget_bytes) // max_p10_estimate
    if capacity < 1:
        raise Phase4D2EligibilityError(
            "p10.memory_budget_bytes",
            "cannot admit the largest source record",
        )
    return min(int(requested_workers), capacity)


def _initialize_d2_process(
    sweep_path: str,
    phase1_config_path: str,
    config_path: str,
    config_raw: bytes,
) -> None:
    global _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG, _PROCESS_D2_CONFIG, _PROCESS_CWT_SYSTEM
    _PROCESS_SWEEP = load_perturbation_sweep_config(Path(sweep_path))
    _PROCESS_PHASE1_CONFIG = load_phase1_core_config(Path(phase1_config_path))
    _PROCESS_D2_CONFIG = parse_phase4_d2_eligibility_config(
        Path(config_path),
        config_raw,
        require_frozen_identity=False,
    )
    catalog = load_classical_catalog(ROOT / CATALOG_RELATIVE_PATH)
    matches = [system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D2EligibilityError("CWT system", "must resolve exactly once")
    _PROCESS_CWT_SYSTEM = matches[0]


def _execute_initialized_d2_record(
    record_order: int,
    spectrum: Spectrum1D,
    class_label: int,
) -> Mapping[str, object]:
    if (
        _PROCESS_SWEEP is None
        or _PROCESS_PHASE1_CONFIG is None
        or _PROCESS_D2_CONFIG is None
        or _PROCESS_CWT_SYSTEM is None
    ):
        raise Phase4D2EligibilityError("process worker", "was not initialized")
    return _execute_d2_record(
        record_order=record_order,
        spectrum=spectrum,
        class_label=class_label,
        sweep=_PROCESS_SWEEP,
        phase1_config=_PROCESS_PHASE1_CONFIG,
        config=_PROCESS_D2_CONFIG,
        admission=P10MemoryAdmission(_PROCESS_D2_CONFIG.p10_memory_budget_bytes),
        cwt_system=_PROCESS_CWT_SYSTEM,
    )


def _run_d2_record_jobs(
    jobs: Sequence[tuple[int, tuple[Spectrum1D, int]]],
    *,
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D2EligibilityConfig,
    worker_count: int,
) -> tuple[Mapping[str, object], ...]:
    process_count = min(
        len(jobs),
        _bounded_process_count(
            requested_workers=worker_count,
            point_counts=tuple(spectrum.axis_cm1.size for _, (spectrum, _) in jobs),
            memory_budget_bytes=config.p10_memory_budget_bytes,
        ),
    )
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_d2_process,
        initargs=(
            str(sweep.path),
            str(phase1_config.path),
            str(config.path),
            config.raw_bytes,
        ),
    ) as executor:
        futures = [
            executor.submit(
                _execute_initialized_d2_record,
                record_order,
                spectrum,
                int(class_label),
            )
            for record_order, (spectrum, class_label) in jobs
        ]
        return tuple(
            sorted(
                (future.result() for future in futures),
                key=lambda row: int(row["record_order"]),
            )
        )


def build_phase4_d2_eligibility_from_inputs(
    output_dir: Path,
    *,
    inputs: D2EligibilityInputs,
    sweep: object,
    phase1_config: object,
    config: Phase4D2EligibilityConfig,
    worker_count: int,
) -> Phase4D2EligibilitySummary:
    if not isinstance(sweep, PerturbationSweepConfig):
        raise Phase4D2EligibilityError("sweep", "must be PerturbationSweepConfig")
    if not isinstance(phase1_config, Phase1CoreConfig):
        raise Phase4D2EligibilityError("phase1_config", "must be Phase1CoreConfig")
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise Phase4D2EligibilityError("worker_count", "must be a positive integer")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_id = _build_run_id(config, inputs)

    source_records = tuple(sorted(inputs.source_records, key=lambda row: (str(row["scope"]), str(row["record_id"]))))
    model_cells = tuple(sorted(inputs.model_cells, key=lambda row: (int(row["seed"]), int(row["shot_count"]))))
    model_role_occurrences = tuple(
        sorted(
            inputs.model_role_occurrences,
            key=lambda row: (int(row["seed"]), int(row["shot_count"]), str(row["role"]), str(row["record_id"])),
        )
    )
    cells: list[Mapping[str, object]] = []
    class_summaries: list[Mapping[str, object]] = []
    common_support_rows: list[Mapping[str, object]] = []
    record_conditions: list[Mapping[str, object]] = []
    metric_statuses: list[Mapping[str, object]] = []
    cwt_receipts: list[Mapping[str, object]] = []
    jobs = tuple(enumerate(zip(inputs.test_spectra, inputs.test_class_labels, strict=True)))
    catalog = load_classical_catalog(ROOT / CATALOG_RELATIVE_PATH)
    cwt_matches = [system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID]
    if len(cwt_matches) != 1:
        raise Phase4D2EligibilityError("CWT system", "must resolve exactly once")
    cwt_system = cwt_matches[0]
    records = _run_d2_record_jobs(
        jobs,
        sweep=sweep,
        phase1_config=phase1_config,
        config=config,
        worker_count=worker_count,
    )
    for record in records:
        for key, destination in (
            ("cells", cells),
            ("record_conditions", record_conditions),
            ("metric_statuses", metric_statuses),
            ("cwt_receipts", cwt_receipts),
        ):
            for row in record[key]:
                validate_outcome_blind_d2_payload(row)
                destination.append(row)
    gate = evaluate_d2_eligibility_gates(
        cells,
        inputs.test_class_labels,
        config,
        record_conditions=record_conditions,
        metric_statuses=metric_statuses,
        cwt_receipts=cwt_receipts,
    )

    for class_label in sorted(set(inputs.test_class_labels)):
        class_records = {
            spectrum.spectrum_id.split("::", 1)[1]
            for spectrum, value in zip(inputs.test_spectra, inputs.test_class_labels, strict=True)
            if value == class_label
        }
        for perturbation_id in ALL_PERTURBATION_IDS:
            complete_records = sum(
                1
                for row in cells
                if row["class_label"] == class_label
                and row["perturbation_id"] == perturbation_id
                and row["state"] == "complete"
            )
            summary = {
                "class_label": int(class_label),
                "complete": complete_records == len(class_records),
                "complete_record_count": complete_records,
                "perturbation_id": perturbation_id,
                "required_record_count": len(class_records),
                "state": "complete" if complete_records == len(class_records) else "not_evaluable_coverage",
            }
            validate_outcome_blind_d2_payload(summary)
            class_summaries.append(summary)

    peak_complete = {
        perturbation_id: {
            row["record_id"]
            for row in cells
            if row["perturbation_id"] == perturbation_id and row["state"] == "complete"
        }
        for perturbation_id in PEAK_PERTURBATION_IDS
    }
    common_records = set.intersection(*peak_complete.values())
    for spectrum, class_label in zip(inputs.test_spectra, inputs.test_class_labels, strict=True):
        record_id = spectrum.spectrum_id.split("::", 1)[1]
        common_support_rows.append(
            {
                "class_label": int(class_label),
                "complete": record_id in common_records,
                "record_id": record_id,
                "scope": "record",
            }
        )
    for class_label in sorted(set(inputs.test_class_labels)):
        class_records = {
            spectrum.spectrum_id.split("::", 1)[1]
            for spectrum, value in zip(inputs.test_spectra, inputs.test_class_labels, strict=True)
            if value == class_label
        }
        common_support_rows.append(
            {
                "class_label": int(class_label),
                "complete": class_records <= common_records,
                "required_record_count": len(class_records),
                "scope": "class",
            }
        )

    if len(cells) != config.expected_cell_count:
        raise Phase4D2EligibilityError("cells", "count mismatch")
    if len(record_conditions) != config.expected_condition_count:
        raise Phase4D2EligibilityError("record_conditions", "count mismatch")
    if len(metric_statuses) != config.expected_metric_status_count:
        raise Phase4D2EligibilityError("metric_statuses", "count mismatch")
    if len(cwt_receipts) != config.expected_cwt_receipt_count:
        raise Phase4D2EligibilityError("cwt_receipts", "count mismatch")
    if len(class_summaries) != config.expected_class_summary_count:
        raise Phase4D2EligibilityError("class_summaries", "count mismatch")

    files: dict[str, bytes] = {
        "config.json": config.raw_bytes,
        "source_records.jsonl": _jsonl_bytes(source_records),
        "model_cells.jsonl": _jsonl_bytes(model_cells),
        "model_role_occurrences.jsonl": _jsonl_bytes(model_role_occurrences),
        "cells.jsonl": _jsonl_bytes(cells),
        "record_conditions.jsonl": _jsonl_bytes(record_conditions),
        "metric_statuses.jsonl": _jsonl_bytes(metric_statuses),
        "cwt_receipts.jsonl": _jsonl_bytes(cwt_receipts),
        "class_summaries.jsonl": _jsonl_bytes(class_summaries),
        "common_support.jsonl": _jsonl_bytes(common_support_rows),
        "gate.json": _canonical_json_bytes(gate),
    }
    manifest = {
        "artifact_schema_version": "phase4-d2-protocol-a-full-domain-eligibility-artifact-v1",
        "files": {
            name: {"bytes": len(payload), "sha256": _sha_bytes(payload)}
            for name, payload in files.items()
        },
        "run_id": run_id,
        "status": gate["overall_status"],
    }
    files["manifest.json"] = _canonical_json_bytes(manifest)
    marker_name = str(gate["marker_filename"])
    files[marker_name] = _canonical_json_bytes(
        {
            "run_id": run_id,
            "schema_version": "phase4-d2-protocol-a-full-domain-eligibility-marker-v1",
            "status": gate["overall_status"],
        }
    )
    sha_lines = [
        f"{_sha_bytes(files[name])}  {name}\n"
        for name in ARTIFACT_STATIC_FILES
        if name != "SHA256SUMS"
    ]
    sha_lines.append(f"{_sha_bytes(files[marker_name])}  {marker_name}\n")
    files["SHA256SUMS"] = "".join(sha_lines).encode("utf-8")
    for name, payload in files.items():
        (output_path / name).write_bytes(payload)
    return Phase4D2EligibilitySummary(
        path=output_path,
        run_id=run_id,
        status=str(gate["overall_status"]),
        test_record_count=config.test_record_count,
        model_cell_count=config.model_cell_count,
        model_role_occurrence_count=config.model_role_occurrence_count,
    )


def build_phase4_d2_eligibility(
    output_root: Path,
    *,
    worker_count: int = 16,
) -> Phase4D2EligibilitySummary:
    config = load_phase4_d2_eligibility_config(ROOT / CONFIG_RELATIVE_PATH)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1_config = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    inputs = reconstruct_d2_eligibility_inputs(
        ROOT / DATASET_RELATIVE_PATH,
        ROOT / SELECTION_RELATIVE_PATH,
        config,
    )
    return build_phase4_d2_eligibility_from_inputs(
        output_root,
        inputs=inputs,
        sweep=sweep,
        phase1_config=phase1_config,
        config=config,
        worker_count=worker_count,
    )


def _compare_tree(candidate_path: Path, rebuilt_path: Path) -> None:
    candidate_files = sorted(path.name for path in candidate_path.iterdir() if path.is_file())
    rebuilt_files = sorted(path.name for path in rebuilt_path.iterdir() if path.is_file())
    if candidate_files != rebuilt_files:
        raise Phase4D2EligibilityError("artifact inventory", "candidate and rebuilt file sets differ")
    for name in candidate_files:
        if (candidate_path / name).read_bytes() != (rebuilt_path / name).read_bytes():
            raise Phase4D2EligibilityError(name, "candidate and rebuilt bytes differ")


def _reconstruct_real_bacteria_inputs(
    dataset_path: Path, selection_path: Path, config: Phase4D2EligibilityConfig
) -> D2EligibilityInputs:
    """Read the retained UnifiedDataset without depending on production D2 loaders."""
    selection = _load_json_document(selection_path)
    support_axis = _support_grid(config)
    test_spectra: list[Spectrum1D] = []
    test_labels: list[int] = []
    test_ids: list[str] = []
    selected_finetune_classes: dict[str, int] = {}
    for seed_document in selection["selections"]:
        for class_document in seed_document["classes"]:
            class_label = int(class_document["class_label"])
            identifiers = [str(item) for item in class_document["validation_record_ids"]]
            for values in _object("train_record_ids", class_document["train_record_ids"]).values():
                identifiers.extend(str(item) for item in values)
            for record_id in identifiers:
                prior = selected_finetune_classes.setdefault(record_id, class_label)
                if prior != class_label:
                    raise Phase4D2EligibilityError(
                        "independent selection class labels",
                        f"conflicting class for {record_id!r}",
                    )
    support_axis_sha = _sha_bytes(np.asarray(support_axis, dtype="<f8").tobytes(order="C"))
    test_source_records: list[Mapping[str, object]] = []
    finetune_source_records: dict[str, Mapping[str, object]] = {}
    with BacteriaIdBatchLoader(dataset_path, batch_size=4096) as loader:
        for batch in loader.iter_batches():
            if batch.source_split not in {"finetune", "test"}:
                continue
            for record_id, class_label, intensity, source_row in zip(
                batch.record_ids,
                batch.class_labels,
                batch.intensity,
                batch.source_rows,
                strict=True,
            ):
                record_id = str(record_id)
                class_label = int(class_label)
                if batch.source_split == "finetune" and record_id not in selected_finetune_classes:
                    continue
                stored_axis = np.asarray(batch.wavenumber, dtype="<f4")
                if not np.all(np.diff(stored_axis) < 0.0):
                    raise Phase4D2EligibilityError("BacteriaIdBatchLoader", "stored axis must be decreasing")
                axis = np.ascontiguousarray(stored_axis[::-1], dtype="<f8")
                values = np.ascontiguousarray(np.asarray(intensity, dtype="<f4")[::-1], dtype="<f8")
                spectrum = Spectrum1D(
                    spectrum_id=f"bacteria_id_reference::{record_id}",
                    sample_id=None, axis_cm1=axis, intensity=values,
                )
                projected = project_d2_support(spectrum, config)
                receipt = {
                    "class_label": class_label,
                    "native_axis_sha256": _array_sha256(axis),
                    "native_intensity_sha256": _array_sha256(values),
                    "record_id": record_id,
                    "role_count": 0,
                    "scope": batch.source_split if batch.source_split == "test" else "finetune_union",
                    "source_row": int(source_row),
                    "support_axis_sha256": support_axis_sha,
                    "support_intensity_sha256": _sha_bytes(projected.tobytes(order="C")),
                }
                if batch.source_split == "finetune":
                    if selected_finetune_classes[record_id] != class_label:
                        raise Phase4D2EligibilityError(
                            "independent finetune class labels",
                            f"selection/dataset mismatch for {record_id!r}",
                        )
                    finetune_source_records[record_id] = receipt
                else:
                    test_ids.append(record_id)
                    test_labels.append(class_label)
                    test_spectra.append(spectrum)
                    test_source_records.append(receipt)
    observed_ids_sha = _sha_bytes(("\n".join(test_ids) + "\n").encode("utf-8"))
    if observed_ids_sha != str(_object("selection.test", selection["test"])["record_ids_sha256"]):
        raise Phase4D2EligibilityError("selection.test.record_ids_sha256", "mismatch")
    source_records: list[Mapping[str, object]] = []
    seen_finetune: set[str] = set()
    model_cells: list[Mapping[str, object]] = []
    role_rows: list[Mapping[str, object]] = []
    for seed_document in selection["selections"]:
        seed = int(seed_document["seed"])
        by_shot = {str(shot): [] for shot in (5, 10, 20)}
        validation: list[str] = []
        for class_document in seed_document["classes"]:
            validation.extend(str(item) for item in class_document["validation_record_ids"])
            seen_finetune.update(validation)
            for shot, identifiers in _object("train_record_ids", class_document["train_record_ids"]).items():
                by_shot[str(shot)].extend(str(item) for item in identifiers)
                seen_finetune.update(str(item) for item in identifiers)
        validation_ids = tuple(sorted(validation))
        for shot in ("5", "10", "20"):
            train_ids = tuple(sorted(by_shot[shot]))
            if set(train_ids) & set(validation_ids):
                raise Phase4D2EligibilityError("selection", "train/validation overlap within one cell")
            model_cells.append({"seed": seed, "shot_count": int(shot), "train_record_ids": train_ids, "validation_record_ids": validation_ids})
            role_rows.extend({"record_id": identifier, "role": "train", "seed": seed, "shot_count": int(shot)} for identifier in train_ids)
            role_rows.extend({"record_id": identifier, "role": "validation", "seed": seed, "shot_count": int(shot)} for identifier in validation_ids)
            role_rows.extend({"record_id": identifier, "role": "test", "seed": seed, "shot_count": int(shot)} for identifier in test_ids)
    if set(finetune_source_records) != seen_finetune:
        raise Phase4D2EligibilityError(
            "independent finetune source ledger",
            "selected records were not reconstructed exactly",
        )
    role_counts: dict[str, int] = {}
    for row in role_rows:
        record_id = str(row["record_id"])
        role_counts[record_id] = role_counts.get(record_id, 0) + 1
    source_records.extend(
        {**row, "role_count": role_counts[str(row["record_id"])]}
        for row in test_source_records
    )
    source_records.extend(
        {
            **finetune_source_records[record_id],
            "role_count": role_counts[record_id],
        }
        for record_id in sorted(seen_finetune)
    )
    if len(source_records) != config.source_record_count or len(model_cells) != config.model_cell_count or len(role_rows) != config.model_role_occurrence_count:
        raise Phase4D2EligibilityError("real input ledger", "frozen denominator mismatch")
    return D2EligibilityInputs(
        source_records=tuple(source_records), model_cells=tuple(model_cells),
        model_role_occurrences=tuple(role_rows), test_spectra=tuple(test_spectra),
        test_class_labels=tuple(test_labels), support_axis_cm1=support_axis,
        test_record_ids_sha256=observed_ids_sha, native_axis_point_count=1000,
        support_axis_point_count=int(support_axis.size),
    )


def verify_phase4_d2_eligibility(
    path: Path,
    *,
    worker_count: int = 12,
) -> Phase4D2EligibilitySummary:
    """Independently bind frozen real authorities before rebuilding an artifact."""
    config_path = ROOT / CONFIG_RELATIVE_PATH
    config = load_phase4_d2_eligibility_config(config_path)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1_config = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    # The real retained dataset is a UnifiedDataset directory.  Its loader is
    # deliberately opened here rather than treating the directory as JSON.
    dataset_path = ROOT / DATASET_RELATIVE_PATH
    selection_path = ROOT / SELECTION_RELATIVE_PATH
    inputs = _reconstruct_real_bacteria_inputs(dataset_path, selection_path, config)
    return verify_phase4_d2_eligibility_from_inputs(
        path,
        inputs=inputs,
        sweep=sweep,
        phase1_config=phase1_config,
        config_path=config_path,
        worker_count=worker_count,
    )


def verify_phase4_d2_eligibility_from_inputs(
    path: Path,
    *,
    inputs: D2EligibilityInputs,
    sweep: object,
    phase1_config: object,
    config_path: Path,
    worker_count: int,
) -> Phase4D2EligibilitySummary:
    config = parse_phase4_d2_eligibility_config(
        config_path,
        Path(config_path).read_bytes(),
        require_frozen_identity=False,
    )
    with tempfile.TemporaryDirectory() as temporary_directory:
        rebuilt_path = Path(temporary_directory) / "rebuilt"
        summary = build_phase4_d2_eligibility_from_inputs(
            rebuilt_path,
            inputs=inputs,
            sweep=sweep,
            phase1_config=phase1_config,
            config=config,
            worker_count=worker_count,
        )
        _compare_tree(Path(path), rebuilt_path)
        return Phase4D2EligibilitySummary(
            path=Path(path),
            run_id=summary.run_id,
            status=summary.status,
            test_record_count=summary.test_record_count,
            model_cell_count=summary.model_cell_count,
            model_role_occurrence_count=summary.model_role_occurrence_count,
        )
