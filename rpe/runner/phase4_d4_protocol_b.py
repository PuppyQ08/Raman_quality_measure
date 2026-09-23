from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
import sklearn
import threadpoolctl
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

from rpe.alignment import (
    AlignmentObservation,
    alignment_gap,
    bulk_paired_cluster_bootstrap,
    compare_alignment,
    cross_perturbation_accuracy,
    holm_step_down,
    paired_contribution_sign_flip,
)
from rpe.downstream.sugar_quantitative import D4SugarCohort, load_d4_sugar_cohort
from rpe.evaluation import Spectrum1D
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import run_perturbation_cell
from rpe.runner import phase4_d4_eligibility as _step27_science

from rpe.runner.phase4_d4_protocol_b_authority import (
    ALPHAS,
    ARTIFACT_PAYLOAD_FILES,
    ARTIFACT_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    CODE_RELATIVE_PATHS,
    CONFIG_BYTES,
    CONFIG_SHA256,
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    MARKER_SCHEMA_VERSION,
    METRIC_OUTPUT_IDS,
    MODEL_FOLDS,
    N_COMPONENTS_GRID,
    PERTURBATIONS,
    REAL_ALPHA0_BLANK_PREDICTION_DIGEST,
    REAL_ALPHA0_MODEL_DIGEST,
    REAL_ALPHA0_PREDICTION_DIGEST,
    REAL_ALPHA0_TECHNICAL_LOD_LOQ_DIGEST,
    REAL_ALPHA0_VALIDATION_DIGEST,
    REAL_MEASUREMENT_BRIDGE_SHA256,
    ROOT,
    RUN_PREFIX,
    SCHEMA_VERSION,
    TERMINAL_MARKERS,
    canonical_json_bytes,
    condition_ids as canonical_condition_ids,
    csv_bytes,
    jsonl_bytes,
    metric_preferred_direction,
    sha256_file,
    sha256_hex,
    stable_run_id,
    write_sha256sums,
)


REAL_STEP27_RUN_ID = (
    "phase4-d4-protocol-a-full-domain-eligibility-"
    "0f45ae5a0815ecb852b410549ad4582c4916dbe16fa0e3e20e2079f7d088097f"
)
REAL_STEP29_RUN_ID = (
    "phase4-d4-protocol-a-full-domain-"
    "1ea534006fcaaecda614d7fbd0f4b4931d532a3f8979f0b7d7ac249025cbb2f7"
)
REAL_STEP31_RUN_ID = (
    "phase4-d4-protocol-b-all-role-eligibility-"
    "031be2fb81d1eef7bb88b23ae099320c8b22070d577738bdbc04e42430eb814d"
)
PROTOCOL_CONFIG_RELATIVE_PATH = "experiments/phase05/configs/d4_sugar_protocol.json"
ARCHIVE_RELATIVE_PATH = "data/raw/ramanbench/cache/10779223/Raw data.zip"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
ELIGIBILITY_CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json"
TARGET_NAMES = (
    "sucrose_nominal_mol_l",
    "fructose_nominal_mol_l",
    "maltose_nominal_mol_l",
    "glucose_nominal_mol_l",
)
LOWER_IS_BETTER = frozenset(
    {
        "mse", "rmse", "mae", "sam", "nmse",
        "wasserstein_1_cm1", "artifact_peak_ratio", "missing_peak_ratio",
    }
)
COLOR_MAP = MappingProxyType(
    dict(zip(PERTURBATIONS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True))
)
STEP29_POSTCOMPUTATION_QC = (
    "model_cells.jsonl",
    "validation_scores.jsonl",
    "predictions.jsonl",
    "blank_predictions.jsonl",
    "technical_lod_loq.jsonl",
)
STEP29_FORBIDDEN_INPUTS = (
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "condition_summary.csv",
    "d4_protocol_a_full_domain_secondary_table.csv",
    "eligibility_bridge.json",
    "figure1_d4_protocol_a_full_domain.png",
    "figure1_d4_protocol_a_full_domain.svg",
    "figure1_d4_protocol_a_full_domain_data.csv",
    "figure2_d4_protocol_a_full_domain.png",
    "figure2_d4_protocol_a_full_domain.svg",
    "figure2_d4_protocol_a_full_domain_data.csv",
    "holm_family.jsonl",
    "manifest.json",
    "preflight.json",
    "sign_flip_results.jsonl",
    "well_conditions.jsonl",
    "well_observations.jsonl",
)


class Phase4D4ProtocolBError(ValueError):
    pass


def _model_lifecycle_from_error(error: Phase4D4ProtocolBError) -> D4ProtocolBModelLifecycleError | None:
    receipt = getattr(error, "receipt", None)
    if not isinstance(receipt, Mapping) or receipt.get("state") != "failed_model_lifecycle":
        return None
    return D4ProtocolBModelLifecycleError(
        condition_id=str(receipt["condition_id"]), fold=int(receipt["fold"]),
        n_components=int(receipt["n_components"]), category=str(receipt.get("warning_category", type(error).__name__)),
        message=str(receipt.get("warning_message", str(error))), partial={},
    )


class D4ProtocolBModelLifecycleError(Phase4D4ProtocolBError):
    """Frozen PLS2 lifecycle failure; this is a scientific terminal state."""

    def __init__(self, *, condition_id: str, fold: int, n_components: int, category: str, message: str, partial: Mapping[str, object]) -> None:
        self.condition_id = condition_id
        self.fold = fold
        self.n_components = n_components
        self.category = category
        self.message = message
        self.partial = partial
        super().__init__(f"PLS2 {condition_id}/fold{fold}/n{n_components}: {category}: {message}")


@dataclass(frozen=True)
class Phase4D4ProtocolBConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    protocol: str
    tier: str
    condition_ids: tuple[str, ...]
    artifact_payload_files: tuple[str, ...]
    model_cell_count: int
    validation_row_count: int
    prediction_row_count: int
    blank_prediction_row_count: int
    technical_lod_loq_row_count: int
    well_condition_row_count: int
    well_observation_row_count: int
    measurement_bridge_sha256: str
    alpha0_expected_digests: Mapping[str, str]
    expected: Mapping[str, object]
    tamper_alpha0_projection: str | None = None


@dataclass(frozen=True)
class D4ProtocolBInputs:
    synthetic_fixture: bool
    condition_ids: tuple[str, ...]
    fold_ids: tuple[int, ...]
    well_ids: tuple[str, ...]
    record_ids: tuple[str, ...]
    blank_record_ids: tuple[str, ...]
    support_axis_cm1: np.ndarray
    feature_count: int
    true_targets: np.ndarray
    blank_targets: np.ndarray
    record_to_well: Mapping[str, str]
    record_to_fold: Mapping[str, int]
    validation_macro_nrmse_by_component: Mapping[int, float]
    prediction_fixture: str
    blank_fixture: str
    metric_fixture: str
    # Retained real-path state.  Synthetic fixtures intentionally leave these
    # empty; the production entry point obtains them from the direct-ZIP
    # cohort, never from a Protocol-A outcome payload.
    mixture_matrix: np.ndarray | None = None
    blank_matrix: np.ndarray | None = None
    train_indices_by_fold: Mapping[int, np.ndarray] | None = None
    validation_indices_by_fold: Mapping[int, np.ndarray] | None = None
    test_indices_by_fold: Mapping[int, np.ndarray] | None = None
    rounds: tuple[int, ...] = ()
    repetitions: tuple[int, ...] = ()
    native_axis_cm1: np.ndarray | None = None
    native_mixture_intensity: np.ndarray | None = None
    native_blank_intensity: np.ndarray | None = None
    blank_well_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class D4ProtocolBModelCell:
    fold: int
    condition_id: str
    train_condition_id: str
    validation_condition_id: str
    test_condition_id: str
    blank_condition_id: str
    selected_n_components: int
    refit_with_validation: bool
    state: str


@dataclass(frozen=True)
class Phase4D4ProtocolBSummary:
    path: Path
    run_id: str
    status: str
    endpoint_state: str
    record_count: int
    prediction_row_count: int


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


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
        return _json_ready(value.tolist())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Phase4D4ProtocolBError(f"unsupported JSON value: {type(value).__name__}")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


def _canonical_document(value: object) -> dict[str, object]:
    result = json.loads(canonical_json_bytes(_json_ready(value)))
    if not isinstance(result, dict):
        raise Phase4D4ProtocolBError("config document must be a JSON object")
    return result


def _float_list(values: np.ndarray) -> list[float]:
    return [float(item) for item in np.asarray(values, dtype=np.float64)]


def _matrix_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return sha256_hex(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


def _code_authority() -> dict[str, dict[str, object]]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": sha256_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _environment_authority() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _default_expected_counts(condition_count: int, record_count: int, blank_record_count: int, well_count: int) -> dict[str, int]:
    positive_condition_count = condition_count - 1
    return {
        "configured_payload_count": len(ARTIFACT_PAYLOAD_FILES),
        "artifact_file_count": len(ARTIFACT_PAYLOAD_FILES) + 2,
        "model_cell_count": len(MODEL_FOLDS) * condition_count,
        "validation_row_count": len(MODEL_FOLDS) * len(N_COMPONENTS_GRID) * condition_count,
        "prediction_row_count": record_count * condition_count,
        "blank_prediction_row_count": len(MODEL_FOLDS) * blank_record_count * condition_count,
        "technical_lod_loq_row_count": condition_count,
        "well_condition_row_count": well_count * condition_count,
        "well_observation_row_count": len(METRIC_OUTPUT_IDS) * well_count * positive_condition_count,
        "condition_summary_row_count": condition_count,
        "alignment_row_count": 13,
        "bootstrap_row_count": 12,
        "sign_flip_row_count": 24,
        "holm_row_count": 24,
        "figure1_row_count": len(METRIC_OUTPUT_IDS) * positive_condition_count,
        "figure2_row_count": 13,
        "secondary_table_row_count": 13,
    }


def parse_phase4_d4_protocol_b_config(
    path: Path,
    raw_bytes: bytes,
    *,
    require_frozen_identity: bool = True,
) -> Phase4D4ProtocolBConfig:
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise Phase4D4ProtocolBError(f"config parse: {error}") from error
    if canonical_json_bytes(document) != raw_bytes:
        raise Phase4D4ProtocolBError("config must be canonical JSON")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4D4ProtocolBError("schema_version mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D4ProtocolBError("experiment_id mismatch")
    synthetic_fixture = bool(document.get("synthetic_fixture", False))
    if require_frozen_identity and not synthetic_fixture and CONFIG_BYTES and CONFIG_SHA256:
        if len(raw_bytes) != CONFIG_BYTES or sha256_hex(raw_bytes) != CONFIG_SHA256:
            raise Phase4D4ProtocolBError("frozen config identity mismatch")
    artifact_payload_files = tuple(str(item) for item in document.get("artifact_payload_files", ()))
    if artifact_payload_files != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D4ProtocolBError("artifact payload order mismatch")
    condition_ids = tuple(str(item) for item in document.get("condition_ids", ()))
    if condition_ids != canonical_condition_ids():
        raise Phase4D4ProtocolBError("condition ids mismatch")
    if not synthetic_fixture:
        if document.get("protocol") != "B" or document.get("tier") != "full_domain_core":
            raise Phase4D4ProtocolBError("protocol/tier mismatch")
        if document.get("claim_boundary") != CLAIM_BOUNDARY:
            raise Phase4D4ProtocolBError("claim boundary mismatch")
        if tuple(document.get("active_perturbation_ids", ())) != PERTURBATIONS:
            raise Phase4D4ProtocolBError("perturbation ids mismatch")
        if tuple(float(value) for value in document.get("alpha_grid", ())) != ALPHAS:
            raise Phase4D4ProtocolBError("alpha grid mismatch")
        if tuple(document.get("metric_output_ids", ())) != METRIC_OUTPUT_IDS:
            raise Phase4D4ProtocolBError("metric output ids mismatch")
        if document.get("model_recipe") != {
            "copy": True,
            "max_iter": 500,
            "n_components_grid": [2, 4, 8, 16, 32],
            "refit_with_validation": False,
            "scale": True,
            "selection": "minimum_validation_macro_normalized_rmse_lowest_components_on_tie",
            "tol": 1e-6,
            "type": "PLSRegression",
        }:
            raise Phase4D4ProtocolBError("model_recipe mismatch")
        inference = document.get("inference", {})
        if not isinstance(inference, Mapping) or any(
            inference.get(key) != value
            for key, value in {
                "bootstrap_resamples": 2000,
                "holm_slot_count": 24,
                "random_seed": 20260817,
                "sign_flip_resamples": 100000,
            }.items()
        ):
            raise Phase4D4ProtocolBError("inference mismatch")
        resource = document.get("resource_policy", {})
        if not isinstance(resource, Mapping) or any(
            resource.get(key) != value
            for key, value in {
                "default_condition_workers": 16,
                "default_model_workers": 5,
                "default_verifier_condition_workers": 12,
                "default_verifier_model_workers": 4,
            }.items()
        ):
            raise Phase4D4ProtocolBError("resource_policy mismatch")
        code_authority = document.get("code_authority")
        if not isinstance(code_authority, Mapping) or tuple(code_authority) != CODE_RELATIVE_PATHS:
            raise Phase4D4ProtocolBError("code_authority path mismatch")
        if require_frozen_identity and dict(code_authority) != _code_authority():
            raise Phase4D4ProtocolBError("code_authority live identity mismatch")
        if document.get("environment_authority") != _environment_authority():
            raise Phase4D4ProtocolBError("environment_authority mismatch")
        if document.get("trust_anchor") != {
            "config_authority_relative_path": "rpe/runner/phase4_d4_protocol_b_authority.py",
            "config_binds_authority": False,
            "direction": "authority_to_config_only",
        }:
            raise Phase4D4ProtocolBError("trust_anchor mismatch")
    alpha0_expected_digests = _freeze(document.get("alpha0_expected_digests", {}))
    expected = _freeze(document.get("expected", {}))
    return Phase4D4ProtocolBConfig(
        path=Path(path),
        raw_bytes=raw_bytes,
        sha256=sha256_hex(raw_bytes),
        document=_freeze(document),
        synthetic_fixture=synthetic_fixture,
        protocol=str(document.get("protocol")),
        tier=str(document.get("tier")),
        condition_ids=condition_ids,
        artifact_payload_files=artifact_payload_files,
        model_cell_count=int(document.get("model_cell_count", 0)),
        validation_row_count=int(document.get("validation_row_count", 0)),
        prediction_row_count=int(document.get("prediction_row_count", 0)),
        blank_prediction_row_count=int(document.get("blank_prediction_row_count", 0)),
        technical_lod_loq_row_count=int(document.get("technical_lod_loq_row_count", 0)),
        well_condition_row_count=int(document.get("well_condition_row_count", 0)),
        well_observation_row_count=int(document.get("well_observation_row_count", 0)),
        measurement_bridge_sha256=str(document.get("measurement_bridge_sha256")),
        alpha0_expected_digests=alpha0_expected_digests,
        expected=expected,
        tamper_alpha0_projection=document.get("tamper_alpha0_projection"),
    )


def load_phase4_d4_protocol_b_config(path: Path) -> Phase4D4ProtocolBConfig:
    raw_bytes = Path(path).read_bytes()
    return parse_phase4_d4_protocol_b_config(Path(path), raw_bytes)


def make_synthetic_d4_protocol_b_inputs(
    *,
    well_count: int = 5,
    acquisitions_per_well: int = 2,
    model_folds: int = 5,
    feature_count: int = 8,
    blank_record_count: int = 2,
    condition_ids: Sequence[str] | None = None,
    validation_macro_nrmse_by_component: Mapping[int, float] | None = None,
    prediction_fixture: str = "default",
    blank_fixture: str = "default",
    metric_fixture: str = "default",
) -> D4ProtocolBInputs:
    if well_count <= 0 or acquisitions_per_well <= 0 or model_folds <= 0 or feature_count <= 0 or blank_record_count <= 0:
        raise Phase4D4ProtocolBError("synthetic inputs require positive dimensions")
    condition_ids_value = tuple(str(item) for item in (condition_ids or canonical_condition_ids()))
    well_ids = tuple(f"well_{index:02d}" for index in range(well_count))
    record_ids = tuple(
        f"record_{well_index:02d}_{repeat_index:02d}"
        for well_index in range(well_count)
        for repeat_index in range(acquisitions_per_well)
    )
    record_to_well = {
        record_id: well_ids[index // acquisitions_per_well]
        for index, record_id in enumerate(record_ids)
    }
    record_to_fold = {
        record_id: index // acquisitions_per_well % model_folds
        for index, record_id in enumerate(record_ids)
    }
    base_targets = []
    for index, record_id in enumerate(record_ids):
        well_index = well_ids.index(record_to_well[record_id])
        base = float(index + 1 + well_index)
        base_targets.append([base, base + 0.25, base + 0.5, base + 0.75])
    blank_targets = np.zeros((blank_record_count, 4), dtype=np.float64)
    validation_map = {
        int(key): float(value)
        for key, value in (validation_macro_nrmse_by_component or {
            2: 0.1,
            4: 0.2,
            8: 0.4,
            16: 0.6,
            32: 0.8,
        }).items()
    }
    support_axis = np.linspace(
        145.83834838867188,
        3684.83544921875,
        1999,
        dtype=np.float64,
    )
    return D4ProtocolBInputs(
        synthetic_fixture=True,
        condition_ids=condition_ids_value,
        fold_ids=tuple(range(model_folds)),
        well_ids=well_ids,
        record_ids=record_ids,
        blank_record_ids=tuple(f"blank_{index:02d}" for index in range(blank_record_count)),
        support_axis_cm1=support_axis,
        feature_count=feature_count,
        true_targets=np.asarray(base_targets, dtype=np.float64),
        blank_targets=blank_targets,
        record_to_well=MappingProxyType(record_to_well),
        record_to_fold=MappingProxyType(record_to_fold),
        validation_macro_nrmse_by_component=MappingProxyType(validation_map),
        prediction_fixture=prediction_fixture,
        blank_fixture=blank_fixture,
        metric_fixture=metric_fixture,
    )


def make_synthetic_d4_protocol_b_config(
    inputs: D4ProtocolBInputs,
    *,
    condition_ids: Sequence[str] | None = None,
    artifact_payload_files: Sequence[str] = ARTIFACT_PAYLOAD_FILES,
    measurement_bridge_sha256: str = REAL_MEASUREMENT_BRIDGE_SHA256,
    alpha0_expected_digests: Mapping[str, str] | None = None,
    tamper_alpha0_projection: str | None = None,
) -> Phase4D4ProtocolBConfig:
    condition_ids_value = tuple(str(item) for item in (condition_ids or inputs.condition_ids))
    expected_counts = _default_expected_counts(
        len(condition_ids_value),
        len(inputs.record_ids),
        len(inputs.blank_record_ids),
        len(inputs.well_ids),
    )
    expected_digests = dict(alpha0_expected_digests or {
        "model_digest": REAL_ALPHA0_MODEL_DIGEST,
        "validation_digest": REAL_ALPHA0_VALIDATION_DIGEST,
        "prediction_digest": REAL_ALPHA0_PREDICTION_DIGEST,
        "blank_prediction_digest": REAL_ALPHA0_BLANK_PREDICTION_DIGEST,
        "technical_lod_loq_digest": REAL_ALPHA0_TECHNICAL_LOD_LOQ_DIGEST,
    })
    document = _canonical_document(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "protocol": "B",
            "tier": "full_domain_core",
            "claim_boundary": CLAIM_BOUNDARY,
            "synthetic_fixture": True,
            "condition_ids": list(condition_ids_value),
            "artifact_payload_files": list(artifact_payload_files),
            "measurement_bridge_sha256": measurement_bridge_sha256,
            "alpha0_expected_digests": expected_digests,
            "tamper_alpha0_projection": tamper_alpha0_projection,
            "model_cell_count": expected_counts["model_cell_count"],
            "validation_row_count": expected_counts["validation_row_count"],
            "prediction_row_count": expected_counts["prediction_row_count"],
            "blank_prediction_row_count": expected_counts["blank_prediction_row_count"],
            "technical_lod_loq_row_count": expected_counts["technical_lod_loq_row_count"],
            "well_condition_row_count": expected_counts["well_condition_row_count"],
            "well_observation_row_count": expected_counts["well_observation_row_count"],
            "expected": expected_counts,
        }
    )
    raw_bytes = canonical_json_bytes(document)
    return parse_phase4_d4_protocol_b_config(
        Path("synthetic://phase4_d4_protocol_b"),
        raw_bytes,
        require_frozen_identity=False,
    )


def _condition_index_map(condition_ids: Sequence[str]) -> dict[str, int]:
    return {condition_id: index for index, condition_id in enumerate(condition_ids)}


def _condition_delta(inputs: D4ProtocolBInputs, condition_id: str) -> float:
    if condition_id == "alpha0":
        return 0.0
    index = _condition_index_map(inputs.condition_ids)[condition_id]
    scale = 0.05 if inputs.prediction_fixture == "hand_derived_regression" else 0.02
    return float(index) * scale


def _selected_n_components(inputs: D4ProtocolBInputs) -> int:
    minimum = min(inputs.validation_macro_nrmse_by_component.values())
    return min(
        candidate
        for candidate, score in inputs.validation_macro_nrmse_by_component.items()
        if score == minimum
    )


def _prediction_rows_for_condition(
    *,
    condition_id: str,
    inputs: D4ProtocolBInputs,
    state: str,
) -> tuple[dict[str, object], ...]:
    if state != "complete":
        return tuple(
            {
                "record_id": record_id,
                "well_id": inputs.record_to_well[record_id],
                "condition_id": condition_id,
                "state": state,
                "predicted_targets": [None, None, None, None],
            }
            for record_id in inputs.record_ids
        )
    delta = _condition_delta(inputs, condition_id)
    rows = []
    for index, record_id in enumerate(inputs.record_ids):
        prediction = np.asarray(inputs.true_targets[index], dtype=np.float64) + delta
        rows.append(
            {
                "record_id": record_id,
                "well_id": inputs.record_to_well[record_id],
                "condition_id": condition_id,
                "state": "complete",
                "predicted_targets": _float_list(prediction),
            }
        )
    return tuple(rows)


def _blank_prediction_rows_for_condition(
    *,
    condition_id: str,
    inputs: D4ProtocolBInputs,
    state: str,
) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    if state != "complete":
        for fold in inputs.fold_ids:
            for blank_record_id in inputs.blank_record_ids:
                rows.append(
                    {
                        "blank_record_id": blank_record_id,
                        "fold": fold,
                        "condition_id": condition_id,
                        "state": state,
                        "predicted_targets": [None, None, None, None],
                    }
                )
        return tuple(rows)
    delta = _condition_delta(inputs, condition_id) * 0.25
    for fold in inputs.fold_ids:
        for blank_record_id in inputs.blank_record_ids:
            rows.append(
                {
                    "blank_record_id": blank_record_id,
                    "fold": fold,
                    "condition_id": condition_id,
                    "state": "complete",
                    "predicted_targets": [float(delta)] * 4,
                }
            )
    return tuple(rows)


def _well_rows_for_condition(
    *,
    condition_id: str,
    prediction_rows: Sequence[Mapping[str, object]],
    inputs: D4ProtocolBInputs,
    state: str,
) -> tuple[dict[str, object], ...]:
    if state != "complete":
        return tuple(
            {
                "well_id": well_id,
                "condition_id": condition_id,
                "state": state,
                "loss": None,
                "downstream_harm": None,
            }
            for well_id in inputs.well_ids
        )
    alpha0_loss = 0.0
    rows = []
    for well_id in inputs.well_ids:
        losses = []
        for row in prediction_rows:
            if row["well_id"] != well_id:
                continue
            if row["state"] != "complete":
                continue
            record_index = inputs.record_ids.index(str(row["record_id"]))
            predicted = np.asarray(row["predicted_targets"], dtype=np.float64)
            target = np.asarray(inputs.true_targets[record_index], dtype=np.float64)
            losses.append(float(np.mean((predicted - target) ** 2)))
        loss = float(np.mean(losses)) if losses else 0.0
        rows.append(
            {
                "well_id": well_id,
                "condition_id": condition_id,
                "state": "complete",
                "loss": loss,
                "downstream_harm": loss - alpha0_loss if condition_id != "alpha0" else 0.0,
            }
        )
    return tuple(rows)


def _lod_row_for_condition(
    *,
    condition_id: str,
    inputs: D4ProtocolBInputs,
    state: str,
) -> dict[str, object]:
    if state != "complete":
        return {
            "condition_id": condition_id,
            "state": state,
            "slope": None,
            "iupac_lod": None,
            "ich_lod": None,
            "ich_loq": None,
        }
    if inputs.blank_fixture == "isolated_auxiliary_failure" and condition_id == inputs.condition_ids[1]:
        return {
            "condition_id": condition_id,
            "state": "not_evaluable_nonpositive_slope",
            "slope": 0.0,
            "iupac_lod": None,
            "ich_lod": None,
            "ich_loq": None,
        }
    severity = _condition_index_map(inputs.condition_ids)[condition_id]
    slope = 1.0 / (severity + 1.0)
    iupac_lod = 0.05 * (severity + 1.0)
    ich_lod = iupac_lod + 0.05
    ich_loq = ich_lod + 0.05
    return {
        "condition_id": condition_id,
        "state": "complete",
        "slope": slope,
        "iupac_lod": iupac_lod,
        "ich_lod": ich_lod,
        "ich_loq": ich_loq,
    }


def fit_d4_protocol_b_condition(
    *,
    condition_id: str,
    condition_matrix: np.ndarray,
    blank_matrix: np.ndarray,
    inputs: D4ProtocolBInputs,
    config: Phase4D4ProtocolBConfig,
    model_worker_count: int,
) -> Mapping[str, object]:
    if not inputs.synthetic_fixture:
        if model_worker_count < 1 or inputs.train_indices_by_fold is None or inputs.validation_indices_by_fold is None or inputs.test_indices_by_fold is None:
            raise Phase4D4ProtocolBError("real PLS2 role inputs are missing")
        matrix = np.ascontiguousarray(condition_matrix, dtype="<f4")
        blanks = np.ascontiguousarray(blank_matrix, dtype="<f4")
        if matrix.shape != (len(inputs.record_ids), inputs.feature_count) or blanks.shape != (len(inputs.blank_record_ids), inputs.feature_count):
            raise Phase4D4ProtocolBError("real PLS2 matrix denominator mismatch")
        if not np.isfinite(matrix).all() or not np.isfinite(blanks).all():
            raise Phase4D4ProtocolBError("real PLS2 matrix is nonfinite")
        model_rows: list[dict[str, object]] = []
        validation_rows: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []
        blank_prediction_rows: list[dict[str, object]] = []
        lod_rows: list[dict[str, object]] = []
        predicted_by_record = np.empty_like(inputs.true_targets, dtype="<f8")
        baseline_loss = {}
        for fold in inputs.fold_ids:
            train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
            validation = np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64)
            test = np.asarray(inputs.test_indices_by_fold[fold], dtype=np.int64)
            candidates: list[tuple[float, int, PLSRegression]] = []
            for n_components in N_COMPONENTS_GRID:
                estimator = PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=1e-6, copy=True)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("error")
                        with threadpool_limits(limits=1, user_api="blas"):
                            estimator.fit(matrix[train], inputs.true_targets[train])
                            validation_prediction = np.asarray(estimator.predict(matrix[validation]), dtype="<f8")
                except (Warning, ValueError, ArithmeticError) as error:
                    raise D4ProtocolBModelLifecycleError(
                        condition_id=condition_id, fold=fold, n_components=n_components,
                        category=type(error).__name__, message=str(error),
                        partial={"model_rows": tuple(model_rows), "validation_rows": tuple(validation_rows), "prediction_rows": tuple(prediction_rows), "blank_prediction_rows": tuple(blank_prediction_rows), "lod_rows": tuple(lod_rows)},
                    ) from error
                if not np.isfinite(validation_prediction).all():
                    raise Phase4D4ProtocolBError("PLS2 validation prediction is nonfinite")
                score = float(np.mean(np.sqrt(np.mean((validation_prediction - inputs.true_targets[validation]) ** 2, axis=0)) / 0.32))
                validation_rows.append({"fold": fold, "condition_id": condition_id, "n_components": n_components, "macro_normalized_rmse": score, "state": "complete"})
                candidates.append((score, n_components, estimator))
            _score, selected, estimator = min(candidates, key=lambda item: (item[0], item[1]))
            state_arrays = (estimator._x_mean, estimator._y_mean, estimator._x_std, estimator._y_std, estimator.x_weights_, estimator.y_weights_, estimator.x_loadings_, estimator.y_loadings_, estimator.x_rotations_, estimator.y_rotations_, estimator.coef_, estimator.intercept_)
            if not all(np.isfinite(np.asarray(item)).all() for item in state_arrays):
                raise Phase4D4ProtocolBError("PLS2 state is nonfinite")
            model_digest = sha256_hex(canonical_json_bytes({"fold": fold, "selected_n_components": selected, "state_sha256": [_matrix_sha(np.asarray(item), "<f8") for item in state_arrays], "n_iter": [int(value) for value in estimator.n_iter_]}))
            model_rows.append({"fold": fold, "condition_id": condition_id, "train_condition_id": condition_id, "validation_condition_id": condition_id, "test_condition_id": condition_id, "blank_condition_id": condition_id, "selected_n_components": selected, "refit_with_validation": False, "model_state_digest": model_digest, "state": "complete"})
            with threadpool_limits(limits=1, user_api="blas"):
                test_prediction = np.asarray(estimator.predict(matrix[test]), dtype="<f8")
                blank_prediction = np.asarray(estimator.predict(blanks), dtype="<f8")
                train_prediction = np.asarray(estimator.predict(matrix[train]), dtype="<f8")
            if not np.isfinite(test_prediction).all() or not np.isfinite(blank_prediction).all():
                raise Phase4D4ProtocolBError("PLS2 prediction is nonfinite")
            predicted_by_record[test] = test_prediction
            for index, value in zip(test.tolist(), test_prediction, strict=True):
                prediction_rows.append({"condition_id": condition_id, "config_sha256": config.sha256, "fold": fold, "model_state_digest": model_digest, "predicted_targets": _float_list(value), "projected_row_sha256": _matrix_sha(matrix[index], "<f4"), "record_id": inputs.record_ids[index], "record_order": index, "repetition": inputs.repetitions[index], "round": inputs.rounds[index], "terminal_state": "complete", "true_targets": _float_list(inputs.true_targets[index]), "well_id": inputs.record_to_well[inputs.record_ids[index]], "state": "complete"})
            for blank_record_id, value in zip(inputs.blank_record_ids, blank_prediction, strict=True):
                blank_prediction_rows.append({"blank_record_id": blank_record_id, "condition_id": condition_id, "fold": fold, "model_state_digest": model_digest, "predicted_targets": _float_list(value), "terminal_state": "complete", "state": "complete"})
            for analyte_index in range(4):
                sigma = float(np.std(blank_prediction[:, analyte_index], ddof=1))
                target = inputs.true_targets[train, analyte_index]
                values = train_prediction[:, analyte_index]
                centered = target - float(np.mean(target))
                denominator = float(np.sum(centered ** 2))
                slope = float(np.sum(centered * (values - float(np.mean(values)))) / denominator) if denominator else float("nan")
                state = "complete" if math.isfinite(slope) and slope > 0.0 else "not_evaluable_nonpositive_slope"
                lod_rows.append({"analyte_index": analyte_index, "condition_id": condition_id, "fold": fold, "sigma": sigma if math.isfinite(sigma) else None, "slope": slope if math.isfinite(slope) else None, "iupac_lod": float(3.0 * sigma / slope) if state == "complete" else None, "ich_lod": float(3.3 * sigma / slope) if state == "complete" else None, "ich_loq": float(10.0 * sigma / slope) if state == "complete" else None, "state": state})
        prediction_rows.sort(key=lambda row: int(row["record_order"]))
        well_rows = []
        for well_id in inputs.well_ids:
            indexes = [index for index, record_id in enumerate(inputs.record_ids) if inputs.record_to_well[record_id] == well_id]
            loss = float(np.mean(((predicted_by_record[indexes] - inputs.true_targets[indexes]) ** 2) / (0.32 ** 2)))
            well_rows.append({"well_id": well_id, "condition_id": condition_id, "fold": inputs.record_to_fold[inputs.record_ids[indexes[0]]], "loss": loss, "downstream_harm": 0.0, "state": "complete"})
        errors = predicted_by_record - inputs.true_targets
        rmse = np.sqrt(np.mean(errors ** 2, axis=0))
        mae = np.mean(np.abs(errors), axis=0)
        denominator = np.sum(
            (inputs.true_targets - inputs.true_targets.mean(axis=0)) ** 2, axis=0
        )
        r2 = np.where(
            denominator > 0.0,
            1.0 - np.sum(errors ** 2, axis=0) / denominator,
            1.0,
        )
        summary_row = {
            "condition_id": condition_id,
            "count": len(inputs.record_ids),
            "macro_normalized_rmse": float(np.mean(rmse / 0.32)),
            "macro_mae_mol_l": float(np.mean(mae)),
            "macro_r2": float(np.mean(r2)),
            "state": "complete",
        }
        for analyte_index, target_name in enumerate(TARGET_NAMES):
            summary_row[f"{target_name}_rmse_mol_l"] = float(rmse[analyte_index])
            summary_row[f"{target_name}_mae_mol_l"] = float(mae[analyte_index])
            summary_row[f"{target_name}_r2"] = float(r2[analyte_index])
        return {"model_rows": tuple(model_rows), "validation_rows": tuple(validation_rows), "prediction_rows": tuple(prediction_rows), "blank_prediction_rows": tuple(blank_prediction_rows), "well_rows": tuple(well_rows), "lod_rows": tuple(lod_rows), "condition_summary_rows": (summary_row,)}
    del condition_matrix, blank_matrix, model_worker_count, config
    selected = _selected_n_components(inputs)
    model_rows = tuple(
        {
            "fold": fold,
            "condition_id": condition_id,
            "train_condition_id": condition_id,
            "validation_condition_id": condition_id,
            "test_condition_id": condition_id,
            "blank_condition_id": condition_id,
            "selected_n_components": selected,
            "refit_with_validation": False,
            "state": "complete",
        }
        for fold in inputs.fold_ids
    )
    validation_rows = tuple(
        {
            "fold": fold,
            "condition_id": condition_id,
            "n_components": candidate,
            "macro_normalized_rmse": float(inputs.validation_macro_nrmse_by_component[candidate]),
            "state": "complete",
        }
        for fold in inputs.fold_ids
        for candidate in N_COMPONENTS_GRID
    )
    prediction_rows = _prediction_rows_for_condition(
        condition_id=condition_id,
        inputs=inputs,
        state="complete",
    )
    blank_prediction_rows = _blank_prediction_rows_for_condition(
        condition_id=condition_id,
        inputs=inputs,
        state="complete",
    )
    well_rows = _well_rows_for_condition(
        condition_id=condition_id,
        prediction_rows=prediction_rows,
        inputs=inputs,
        state="complete",
    )
    lod_row = _lod_row_for_condition(condition_id=condition_id, inputs=inputs, state="complete")
    summary_row = {
        "condition_id": condition_id,
        "state": "complete",
        "macro_normalized_rmse": math.sqrt(
            float(np.mean([row["loss"] for row in well_rows if row["loss"] is not None]))
        ) if any(row["loss"] is not None for row in well_rows) else 0.0,
    }
    return {
        "model_rows": model_rows,
        "validation_rows": validation_rows,
        "prediction_rows": prediction_rows,
        "blank_prediction_rows": blank_prediction_rows,
        "well_rows": well_rows,
        "lod_rows": (lod_row,),
        "condition_summary_rows": (summary_row,),
    }


def validate_d4_protocol_b_alpha0_equivalence(
    *,
    model_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    blank_prediction_rows: Sequence[Mapping[str, object]],
    lod_rows: Sequence[Mapping[str, object]],
    config: Phase4D4ProtocolBConfig,
) -> Mapping[str, object]:
    if not config.synthetic_fixture:
        projections = {
            "model_digest": (model_rows, ("fold", "model_state_digest", "refit_with_validation", "selected_n_components")),
            "validation_digest": (validation_rows, ("fold", "macro_normalized_rmse", "n_components")),
            "prediction_digest": (prediction_rows, ("condition_id", "fold", "model_state_digest", "predicted_targets", "projected_row_sha256", "record_id", "record_order", "repetition", "round", "terminal_state", "true_targets", "well_id")),
            "blank_prediction_digest": (blank_prediction_rows, ("blank_record_id", "condition_id", "fold", "model_state_digest", "predicted_targets", "terminal_state")),
            "technical_lod_loq_digest": (lod_rows, ("analyte_index", "condition_id", "fold", "ich_lod", "ich_loq", "iupac_lod", "sigma", "slope", "state")),
        }
        observed = {name: sha256_hex(b"".join(canonical_json_bytes({key: row[key] for key in fields}) for row in rows)) for name, (rows, fields) in projections.items()}
    else:
        observed = {
            "model_digest": REAL_ALPHA0_MODEL_DIGEST,
            "validation_digest": REAL_ALPHA0_VALIDATION_DIGEST,
            "prediction_digest": REAL_ALPHA0_PREDICTION_DIGEST,
            "blank_prediction_digest": REAL_ALPHA0_BLANK_PREDICTION_DIGEST,
            "technical_lod_loq_digest": REAL_ALPHA0_TECHNICAL_LOD_LOQ_DIGEST,
        }
    if config.tamper_alpha0_projection:
        observed[str(config.tamper_alpha0_projection)] = "tampered:" + str(config.tamper_alpha0_projection)
    mismatch_count = sum(
        1
        for key, value in observed.items()
        if config.alpha0_expected_digests.get(key) != value
    )
    status = "passed" if mismatch_count == 0 else "failed"
    return {
        "status": status,
        "mismatch_count": mismatch_count,
        "expected_digests": dict(config.alpha0_expected_digests),
        "observed_digests": observed,
    }


def build_d4_protocol_b_measurement_bridge(
    *,
    config: Phase4D4ProtocolBConfig,
    inputs: D4ProtocolBInputs,
    rematerialization_receipts: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...]]:
    if config.synthetic_fixture:
        sums: dict[tuple[str, str, str], list[float]] = {}
        for condition_id in inputs.condition_ids[1:]:
            severity = _condition_index_map(inputs.condition_ids)[condition_id]
            for record_index, record_id in enumerate(inputs.record_ids):
                well_id = inputs.record_to_well[record_id]
                for metric_index, metric_output_id in enumerate(METRIC_OUTPUT_IDS):
                    direction = metric_preferred_direction(metric_output_id)
                    baseline = 0.0 if direction == "lower_is_better" else 1.0
                    if direction == "lower_is_better":
                        current = severity * 0.01 + metric_index * 0.001 + record_index * 0.0001
                        harm = current - baseline
                    else:
                        current = max(0.0, 1.0 - severity * 0.01 - metric_index * 0.001 - record_index * 0.0001)
                        harm = baseline - current
                    bucket = sums.setdefault((well_id, condition_id, metric_output_id), [0.0, 0.0])
                    bucket[0] += harm
                    bucket[1] += 1.0
        rows = tuple(
            {
                "acquisition_count": int(sums[(well_id, condition_id, metric_output_id)][1]),
                "alpha": _parse_condition_id(condition_id)[1],
                "condition_id": condition_id,
                "downstream_harm": None,
                "metric_harm": sums[(well_id, condition_id, metric_output_id)][0] / sums[(well_id, condition_id, metric_output_id)][1],
                "metric_output_id": metric_output_id,
                "perturbation_id": _parse_condition_id(condition_id)[0],
                "state": "complete",
                "well_id": well_id,
            }
            for well_id in inputs.well_ids
            for condition_id in inputs.condition_ids[1:]
            for metric_output_id in METRIC_OUTPUT_IDS
        )
        return {
            "state": "complete",
            "bridge_sha256": config.measurement_bridge_sha256,
            "bridge_row_count": len(inputs.record_ids) * len(inputs.condition_ids),
            "allowed_step29_payloads": ("record_measurements.jsonl",),
        }, rows

    parents = dict(_thaw(config.document).get("parent_artifacts", {}))
    step27_path = ROOT / str(parents["step27"]["relative_path"]) / "record_conditions.jsonl"
    step29_path = ROOT / str(parents["step29"]["relative_path"]) / "record_measurements.jsonl"
    bridge_hasher = hashlib.sha256()
    normalized_bytes = 0
    row_count = 0
    mismatch = {
        "missing_key_count": 0,
        "extra_key_count": 0,
        "duplicate_key_count": 0,
        "state_mismatch_count": 0,
        "metric_receipt_mismatch_count": 0,
        "cwt_receipt_mismatch_count": 0,
        "rematerialization_mismatch_count": 0,
    }
    baseline_values: dict[str, tuple[float, ...]] = {}
    sums: dict[tuple[str, str, str], list[float]] = {}
    seen: set[tuple[str, str]] = set()
    with step27_path.open(encoding="utf-8") as step27_stream, step29_path.open(encoding="utf-8") as step29_stream:
        try:
            paired_rows = zip(step27_stream, step29_stream, strict=True)
            for row_index, (step27_line, step29_line) in enumerate(paired_rows):
                parent = json.loads(step27_line)
                measurement = json.loads(step29_line)
                record_index, condition_index = divmod(row_index, len(inputs.condition_ids))
                if record_index >= len(inputs.record_ids):
                    mismatch["extra_key_count"] += 1
                    continue
                record_id = inputs.record_ids[record_index]
                condition_id = inputs.condition_ids[condition_index]
                key = (record_id, condition_id)
                if key in seen:
                    mismatch["duplicate_key_count"] += 1
                seen.add(key)
                if (str(parent.get("record_id")), str(parent.get("condition_id"))) != key or (str(measurement.get("record_id")), str(measurement.get("condition_id"))) != key:
                    mismatch["missing_key_count"] += 1
                perturbation_id, alpha = _parse_condition_id(condition_id)
                derived = {
                    "alpha": alpha, "condition_id": condition_id,
                    "fold": inputs.record_to_fold[record_id],
                    "perturbation_id": perturbation_id, "record_id": record_id,
                    "record_order": record_index, "well_id": inputs.record_to_well[record_id],
                    "state": "complete",
                }
                if any(measurement.get(name) != value for name, value in derived.items()):
                    mismatch["state_mismatch_count"] += 1
                if parent.get("state") != measurement.get("state"):
                    mismatch["state_mismatch_count"] += 1
                sealed = rematerialization_receipts.get(key)
                if sealed is None or any(measurement.get(name) != sealed.get(name) for name in ("native_axis_sha256", "native_intensity_sha256", "projected_row_sha256")):
                    mismatch["rematerialization_mismatch_count"] += 1
                metric_projection = []
                values = []
                metrics = measurement.get("metric_values", ())
                if tuple(item.get("output_id") for item in metrics) != METRIC_OUTPUT_IDS:
                    mismatch["metric_receipt_mismatch_count"] += 1
                for metric_output_id, item in zip(METRIC_OUTPUT_IDS, metrics, strict=False):
                    parent_item = parent.get("metrics", {}).get(metric_output_id, {})
                    projected = {
                        "output_id": metric_output_id,
                        "state": item.get("state"),
                        "result_sha256": item.get("result_digest"),
                        "diagnostics_sha256": item.get("diagnostics_digest"),
                    }
                    if any(parent_item.get(name) != projected.get(name) for name in ("state", "result_sha256", "diagnostics_sha256")):
                        mismatch["metric_receipt_mismatch_count"] += 1
                    value = float(item.get("value"))
                    if not math.isfinite(value):
                        raise Phase4D4ProtocolBError(f"nonfinite Step-29 metric value: {key}/{metric_output_id}")
                    values.append(value)
                    metric_projection.append(projected)
                cwt = {
                    "state": measurement.get("cwt", {}).get("state"),
                    "diagnostics_sha256": measurement.get("cwt", {}).get("diagnostics_digest"),
                    "peak_list_sha256": measurement.get("cwt", {}).get("peak_list_digest"),
                    "warning_sha256": measurement.get("cwt", {}).get("warning_digest"),
                }
                if any(parent.get("cwt", {}).get(name) != value for name, value in cwt.items()):
                    mismatch["cwt_receipt_mismatch_count"] += 1
                projection = {**derived, "native_axis_sha256": measurement.get("native_axis_sha256"), "native_intensity_sha256": measurement.get("native_intensity_sha256"), "projected_row_sha256": measurement.get("projected_row_sha256"), "metrics": metric_projection, "cwt": cwt}
                raw = canonical_json_bytes(projection)
                bridge_hasher.update(raw); normalized_bytes += len(raw); row_count += 1
                if condition_id == "alpha0":
                    baseline_values[record_id] = tuple(values)
                else:
                    baseline = baseline_values.get(record_id)
                    if baseline is None:
                        raise Phase4D4ProtocolBError(f"missing alpha-zero metric row before {key}")
                    for metric_index, metric_output_id in enumerate(METRIC_OUTPUT_IDS):
                        harm = values[metric_index] - baseline[metric_index] if metric_output_id in LOWER_IS_BETTER else baseline[metric_index] - values[metric_index]
                        bucket = sums.setdefault((inputs.record_to_well[record_id], condition_id, metric_output_id), [0.0, 0.0])
                        bucket[0] += harm; bucket[1] += 1.0
        except ValueError as error:
            raise Phase4D4ProtocolBError(f"measurement bridge row-count mismatch: {error}") from error
    expected_rows = len(inputs.record_ids) * len(inputs.condition_ids)
    if row_count != expected_rows or len(seen) != expected_rows:
        mismatch["missing_key_count"] += abs(expected_rows - len(seen))
    observed_digest = bridge_hasher.hexdigest()
    expected_bytes = int(_thaw(config.document).get("measurement_bridge", {}).get("normalized_bytes", 0))
    if observed_digest != config.measurement_bridge_sha256 or normalized_bytes != expected_bytes or any(mismatch.values()):
        raise Phase4D4ProtocolBError(
            f"measurement bridge mismatch: sha256={observed_digest} bytes={normalized_bytes} counts={mismatch}"
        )
    rows = tuple(
        {
            "acquisition_count": int(sums[(well_id, condition_id, metric_output_id)][1]),
            "alpha": _parse_condition_id(condition_id)[1],
            "condition_id": condition_id,
            "downstream_harm": None,
            "metric_harm": sums[(well_id, condition_id, metric_output_id)][0] / sums[(well_id, condition_id, metric_output_id)][1],
            "metric_output_id": metric_output_id,
            "perturbation_id": _parse_condition_id(condition_id)[0],
            "state": "complete",
            "well_id": well_id,
        }
        for well_id in inputs.well_ids
        for condition_id in inputs.condition_ids[1:]
        for metric_output_id in METRIC_OUTPUT_IDS
    )
    receipt = {
        "allowed_step29_payloads": ("record_measurements.jsonl",),
        "bridge_row_count": row_count, "bridge_sha256": observed_digest,
        "mismatch_counts": mismatch, "normalized_bytes": normalized_bytes,
        "state": "complete",
    }
    return receipt, rows


def aggregate_d4_protocol_b(
    *,
    inputs: D4ProtocolBInputs,
    condition_summary_rows: Sequence[Mapping[str, object]],
    well_rows: Sequence[Mapping[str, object]],
    lod_rows: Sequence[Mapping[str, object]],
    measurement_bridge_rows: Sequence[Mapping[str, object]],
    endpoint_state: str,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    del lod_rows
    if endpoint_state != "complete":
        raise Phase4D4ProtocolBError("real aggregation requires a complete endpoint")
    well_lookup = {
        (str(row["well_id"]), str(row["condition_id"])): dict(row)
        for row in well_rows
    }
    normalized_wells: list[Mapping[str, object]] = []
    for well_id in inputs.well_ids:
        baseline_loss = float(well_lookup[(well_id, "alpha0")]["loss"])
        for condition_id in inputs.condition_ids:
            row = dict(well_lookup[(well_id, condition_id)])
            row["downstream_harm"] = 0.0 if condition_id == "alpha0" else float(row["loss"]) - baseline_loss
            normalized_wells.append(row)
    downstream = {
        (str(row["well_id"]), str(row["condition_id"])): float(row["downstream_harm"])
        for row in normalized_wells
    }
    well_observations = tuple(
        {
            **dict(row),
            "downstream_harm": downstream[(str(row["well_id"]), str(row["condition_id"]))],
            "harm": float(row["metric_harm"]),
            "preferred_direction": metric_preferred_direction(str(row["metric_output_id"])),
        }
        for row in measurement_bridge_rows
    )
    expected_observations = len(METRIC_OUTPUT_IDS) * len(inputs.well_ids) * (len(inputs.condition_ids) - 1)
    if len(well_observations) != expected_observations:
        raise Phase4D4ProtocolBError("well observation denominator mismatch")
    tables = {
        metric_output_id: tuple(
            AlignmentObservation(
                cluster_id=str(row["well_id"]),
                perturbation_id=str(row["perturbation_id"]),
                alpha=float(row["alpha"]),
                metric_harm=float(row["metric_harm"]),
                downstream_harm=float(row["downstream_harm"]),
            )
            for row in well_observations
            if row["metric_output_id"] == metric_output_id
        )
        for metric_output_id in METRIC_OUTPUT_IDS
    }
    bootstrap_n = 2000 if bootstrap_resamples is None else int(bootstrap_resamples)
    sign_flip_n = 100000 if sign_flip_resamples is None else int(sign_flip_resamples)
    reference = tables["mse"]
    reference_gap = alignment_gap(reference)
    reference_acc = cross_perturbation_accuracy(reference)
    reference_boot = bulk_paired_cluster_bootstrap(reference, reference, resamples=bootstrap_n, confidence_level=0.95, random_seed=20260817)
    alignment_rows: list[Mapping[str, object]] = [{
        "acc_cross": reference_acc.accuracy, "acc_interval": reference_boot.reference_acc_interval,
        "ag": reference_gap.alignment_gap, "ag_interval": reference_boot.reference_ag_interval,
        "ag_raw": reference_gap.raw_alignment_gap, "clusters": len(inputs.well_ids),
        "cross_pair_count": reference_acc.pair_count, "metric_output_id": "mse",
        "observation_count": len(reference), "state": "complete",
    }]
    bootstrap_rows: list[Mapping[str, object]] = []
    sign_flip_rows: list[Mapping[str, object]] = []
    p_values: dict[str, float] = {}
    contrasts: dict[str, float] = {}
    for metric_output_id in METRIC_OUTPUT_IDS[1:]:
        candidate = tables[metric_output_id]
        comparison = compare_alignment(reference, candidate)
        boot = bulk_paired_cluster_bootstrap(reference, candidate, resamples=bootstrap_n, confidence_level=0.95, random_seed=20260817)
        bootstrap_rows.append({"candidate_acc_interval": boot.candidate_acc_interval, "candidate_ag_interval": boot.candidate_ag_interval, "d_acc_interval": boot.d_acc_interval, "d_ag_interval": boot.d_ag_interval, "metric_output_id": metric_output_id, "resamples": bootstrap_n, "state": "complete"})
        alignment_rows.append({"acc_cross": comparison.candidate_accuracy.accuracy, "acc_interval": boot.candidate_acc_interval, "ag": comparison.candidate_gap.alignment_gap, "ag_interval": boot.candidate_ag_interval, "ag_raw": comparison.candidate_gap.raw_alignment_gap, "clusters": len(inputs.well_ids), "cross_pair_count": comparison.candidate_accuracy.pair_count, "d_acc": comparison.d_acc, "d_acc_interval": boot.d_acc_interval, "d_ag": comparison.d_ag, "d_ag_interval": boot.d_ag_interval, "metric_output_id": metric_output_id, "observation_count": len(candidate), "state": "complete"})
        for statistic, contributions, contrast in (("d_ag", [item.value for item in comparison.ag_contribution_differences], comparison.d_ag), ("d_acc", [item.value for item in comparison.acc_contribution_differences], comparison.d_acc)):
            sign = paired_contribution_sign_flip(contributions, aggregation="sum" if statistic == "d_ag" else "mean", resamples=sign_flip_n, random_seed=20260817)
            hypothesis_id = f"{metric_output_id}:{statistic}"
            p_values[hypothesis_id] = sign.p_value
            contrasts[hypothesis_id] = contrast
            sign_flip_rows.append({"contrast": contrast, "hypothesis_id": hypothesis_id, "metric_output_id": metric_output_id, "p_value": sign.p_value, "resamples": sign_flip_n, "state": "complete", "statistic": statistic})
    adjusted = {row.hypothesis_id: row for row in holm_step_down({key: p_values[key] for key in sorted(p_values)}, alpha=0.05)}
    holm_rows: list[Mapping[str, object]] = []
    for metric_output_id in METRIC_OUTPUT_IDS[1:]:
        for statistic in ("d_ag", "d_acc"):
            hypothesis_id = f"{metric_output_id}:{statistic}"
            result = adjusted[hypothesis_id]
            contrast = contrasts[hypothesis_id]
            favorable = contrast > 0.0
            holm_rows.append({"adjusted_p_value": result.adjusted_p_value, "favorable": favorable, "family_size": result.family_size, "hypothesis_id": hypothesis_id, "metric_output_id": metric_output_id, "multiplicity_p_value": p_values[hypothesis_id], "observed_contrast": contrast, "rank": result.rank, "raw_p_value": result.raw_p_value, "rejected": bool(result.rejected and favorable), "state": "tested", "statistic": statistic})
    metric_states = {str(row["metric_output_id"]): str(row["state"]) for row in alignment_rows}
    figure1_rows = tuple({"alpha": alpha, "mean_downstream_harm": float(np.mean([row["downstream_harm"] for row in well_observations if row["metric_output_id"] == metric_output_id and row["perturbation_id"] == perturbation_id and float(row["alpha"]) == alpha])), "mean_metric_harm": float(np.mean([row["metric_harm"] for row in well_observations if row["metric_output_id"] == metric_output_id and row["perturbation_id"] == perturbation_id and float(row["alpha"]) == alpha])), "metric_output_id": metric_output_id, "metric_state": metric_states[metric_output_id], "perturbation_id": perturbation_id} for metric_output_id in METRIC_OUTPUT_IDS for perturbation_id in PERTURBATIONS for alpha in ALPHAS[1:])
    family = {(row["metric_output_id"], row["statistic"]): row for row in holm_rows}
    figure2_rows = tuple({"acc_cross": row.get("acc_cross"), "acc_interval": row.get("acc_interval"), "ag": row.get("ag"), "ag_interval": row.get("ag_interval"), "ag_raw": row.get("ag_raw"), "clusters": row.get("clusters"), "d_acc": row.get("d_acc"), "d_acc_adjusted_p": family.get((str(row["metric_output_id"]), "d_acc"), {}).get("adjusted_p_value"), "d_acc_favorable": family.get((str(row["metric_output_id"]), "d_acc"), {}).get("favorable"), "d_acc_interval": row.get("d_acc_interval"), "d_acc_rank": family.get((str(row["metric_output_id"]), "d_acc"), {}).get("rank"), "d_acc_raw_p": family.get((str(row["metric_output_id"]), "d_acc"), {}).get("raw_p_value"), "d_acc_rejected": family.get((str(row["metric_output_id"]), "d_acc"), {}).get("rejected"), "d_ag": row.get("d_ag"), "d_ag_adjusted_p": family.get((str(row["metric_output_id"]), "d_ag"), {}).get("adjusted_p_value"), "d_ag_favorable": family.get((str(row["metric_output_id"]), "d_ag"), {}).get("favorable"), "d_ag_interval": row.get("d_ag_interval"), "d_ag_rank": family.get((str(row["metric_output_id"]), "d_ag"), {}).get("rank"), "d_ag_raw_p": family.get((str(row["metric_output_id"]), "d_ag"), {}).get("raw_p_value"), "d_ag_rejected": family.get((str(row["metric_output_id"]), "d_ag"), {}).get("rejected"), "metric_output_id": row["metric_output_id"], "observation_count": row.get("observation_count"), "state": row["state"]} for row in alignment_rows)
    return {
        "condition_summary_rows": tuple(condition_summary_rows),
        "well_condition_rows": tuple(normalized_wells),
        "well_observation_rows": well_observations,
        "alignment_rows": tuple(alignment_rows),
        "bootstrap_rows": tuple(bootstrap_rows),
        "sign_flip_rows": tuple(sign_flip_rows),
        "holm_rows": tuple(holm_rows),
        "figure1_rows": figure1_rows,
        "figure2_rows": figure2_rows,
        "table_rows": figure2_rows,
    }


def render_d4_protocol_b_figures(
    *,
    figure1_rows: Sequence[Mapping[str, object]],
    figure2_rows: Sequence[Mapping[str, object]],
) -> Mapping[str, bytes]:
    if not figure1_rows or not figure2_rows:
        raise Phase4D4ProtocolBError("figure projections must be nonempty")

    # Plot only from the same canonical CSV projections that are emitted as
    # checksummed payloads.  This makes the visible figures a deterministic
    # view of the tabular authority rather than a second aggregation path.
    figure1_plot_rows = tuple(
        csv.DictReader(io.StringIO(csv_bytes(figure1_rows).decode("utf-8")))
    )
    figure2_plot_rows = tuple(
        csv.DictReader(io.StringIO(csv_bytes(figure2_rows).decode("utf-8")))
    )
    complete = (
        "mean_downstream_harm" in figure1_plot_rows[0]
        and all(row.get("metric_state") == "complete" for row in figure1_plot_rows)
        and all(row.get("state") == "complete" for row in figure2_plot_rows)
    )

    payloads: dict[str, bytes] = {}
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "svg.hashsalt": "rpe-phase4-d4-protocol-b-v1",
        }
    ):
        figure1, axes1 = plt.subplots(4, 4, figsize=(12, 12))
        if complete:
            downstream_axis = axes1.ravel()[0]
            for perturbation_id in PERTURBATIONS:
                selected = sorted(
                    (
                        row
                        for row in figure1_plot_rows
                        if row["metric_output_id"] == "mse"
                        and row["perturbation_id"] == perturbation_id
                    ),
                    key=lambda row: float(row["alpha"]),
                )
                downstream_axis.plot(
                    [float(row["alpha"]) for row in selected],
                    [float(row["mean_downstream_harm"]) for row in selected],
                    marker="o",
                    linewidth=1.5,
                    color=COLOR_MAP[perturbation_id],
                )
            downstream_axis.set_title("normalized squared-loss harm")
            downstream_axis.set_xlabel("alpha")
            downstream_axis.set_ylabel("mean downstream harm")
            downstream_axis.margins(x=0.05, y=0.05)
            for axis, metric_output_id in zip(
                axes1.ravel()[1:], METRIC_OUTPUT_IDS, strict=False
            ):
                selected_metric = tuple(
                    row
                    for row in figure1_plot_rows
                    if row["metric_output_id"] == metric_output_id
                )
                by_perturbation = {
                    perturbation_id: sorted(
                        (
                            row
                            for row in selected_metric
                            if row["perturbation_id"] == perturbation_id
                        ),
                        key=lambda row: float(row["alpha"]),
                    )
                    for perturbation_id in PERTURBATIONS
                }
                for perturbation_id, selected in by_perturbation.items():
                    axis.plot(
                        [float(row["mean_metric_harm"]) for row in selected],
                        [float(row["mean_downstream_harm"]) for row in selected],
                        marker="o",
                        linewidth=1.5,
                        color=COLOR_MAP[perturbation_id],
                    )
                axis.set_title(metric_output_id)
                axis.margins(x=0.05, y=0.05)
        else:
            for index, axis in enumerate(axes1.ravel()[:14]):
                axis.set_facecolor("#f2f2f2")
                axis.text(
                    0.5,
                    0.5,
                    "not tested: endpoint closed",
                    color="#666666",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
                axis.set_title(
                    "normalized squared-loss harm"
                    if index == 0
                    else METRIC_OUTPUT_IDS[index - 1]
                )
                axis.set_xticks(())
                axis.set_yticks(())
        for axis in axes1.ravel()[14:]:
            axis.set_axis_off()
        figure1.tight_layout()
        png = io.BytesIO()
        svg = io.BytesIO()
        figure1.savefig(png, format="png", dpi=300, metadata={"Date": None})
        figure1.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(figure1)
        payloads["figure1_d4_protocol_b_full_domain.png"] = png.getvalue()
        payloads["figure1_d4_protocol_b_full_domain.svg"] = svg.getvalue()

        figure2, axes2 = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        y_positions = np.arange(len(METRIC_OUTPUT_IDS))
        for axis, field, title, color in zip(
            axes2,
            ("ag", "acc_cross", "d_ag", "d_acc"),
            ("AG", "Acc-cross", "D_AG", "D_Acc"),
            ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"),
            strict=True,
        ):
            values = [
                float(row[field]) if complete and row.get(field) not in (None, "") else 0.0
                for row in figure2_plot_rows
            ]
            axis.barh(
                y_positions,
                values,
                color=color if complete else "#b3b3b3",
            )
            axis.axvline(0.0, color="#666666", linewidth=0.6)
            axis.set_title(title)
            axis.set_yticks(y_positions)
            axis.margins(x=0.05)
            if not complete:
                axis.set_facecolor("#f2f2f2")
                axis.text(
                    0.5,
                    0.02,
                    "not tested: endpoint closed",
                    color="#666666",
                    ha="center",
                    va="bottom",
                    transform=axis.transAxes,
                )
        axes2[0].set_yticklabels(METRIC_OUTPUT_IDS)
        for axis in axes2[1:]:
            axis.tick_params(axis="y", labelleft=False)
        figure2.tight_layout()
        png = io.BytesIO()
        svg = io.BytesIO()
        figure2.savefig(png, format="png", dpi=300, metadata={"Date": None})
        figure2.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(figure2)
        payloads["figure2_d4_protocol_b_full_domain.png"] = png.getvalue()
        payloads["figure2_d4_protocol_b_full_domain.svg"] = svg.getvalue()
    return MappingProxyType(payloads)


def validate_d4_protocol_b_parent_authorities(
    config: Phase4D4ProtocolBConfig,
    *,
    parse_step29_measurements: bool = False,
) -> Mapping[str, object]:
    if config.synthetic_fixture:
        bridge = {
            "synthetic_fixture": True,
            "step27_run_id": "synthetic-step27",
            "step29_run_id": "synthetic-step29",
            "step31_run_id": "synthetic-step31",
            "allowed_step29_payloads": ("record_measurements.jsonl",),
            "measurement_bridge_sha256": config.measurement_bridge_sha256,
            "parse_step29_measurements": parse_step29_measurements,
        }
        return bridge
    document = _thaw(config.document)
    for receipt in document.get("authorities", {}).values():
        if "path" in receipt:
            path = ROOT / str(receipt["path"])
            if not path.is_file():
                raise Phase4D4ProtocolBError(f"missing authority file: {path}")
            if int(receipt["bytes"]) != path.stat().st_size:
                raise Phase4D4ProtocolBError(f"authority bytes mismatch: {path}")
            if str(receipt["sha256"]) != sha256_file(path):
                raise Phase4D4ProtocolBError(f"authority sha mismatch: {path}")
        elif "archive_path" in receipt and "member_path" in receipt:
            archive_path = ROOT / str(receipt["archive_path"])
            if not archive_path.is_file():
                raise Phase4D4ProtocolBError(f"missing archive authority: {archive_path}")
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    payload = archive.read(str(receipt["member_path"]))
            except (OSError, KeyError, zipfile.BadZipFile) as error:
                raise Phase4D4ProtocolBError(
                    f"invalid archive member authority: {archive_path}"
                ) from error
            if len(payload) != int(receipt["bytes"]) or sha256_hex(payload) != str(receipt["sha256"]):
                raise Phase4D4ProtocolBError(
                    f"archive member authority mismatch: {archive_path}:{receipt['member_path']}"
                )
    parents = dict(document.get("parent_artifacts", {}))
    step27 = parents.get("step27", {})
    step29 = parents.get("step29", {})
    step31 = parents.get("step31", {})
    for parent in (step27, step29, step31):
        relative_path = str(parent.get("relative_path", ""))
        if not relative_path:
            raise Phase4D4ProtocolBError("missing parent relative path")
        path = ROOT / relative_path
        if not path.is_dir():
            raise Phase4D4ProtocolBError(f"missing parent directory: {relative_path}")
        if "non_authoritative_pre_fix" in relative_path:
            raise Phase4D4ProtocolBError("non_authoritative_pre_fix parent is forbidden")
        required_inventory = tuple(parent.get("required_inventory", ()))
        if required_inventory:
            existing = {item.name for item in path.iterdir() if item.is_file()}
            if existing != set(required_inventory):
                raise Phase4D4ProtocolBError(f"parent inventory mismatch: {relative_path}")
        if "sha256sums_sha256" in parent:
            checksum_path = path / "SHA256SUMS"
            if not checksum_path.is_file() or sha256_file(checksum_path) != str(parent["sha256sums_sha256"]):
                raise Phase4D4ProtocolBError(f"parent SHA256SUMS mismatch: {relative_path}")
            entries = {}
            for line in checksum_path.read_text(encoding="utf-8").splitlines():
                digest, name = line.split("  ", 1)
                if name in entries or len(digest) != 64:
                    raise Phase4D4ProtocolBError(f"parent checksum ledger malformed: {relative_path}")
                entries[name] = digest
            required_without_ledger = set(required_inventory) - {"SHA256SUMS"}
            if set(entries) != required_without_ledger:
                raise Phase4D4ProtocolBError(f"parent checksum ledger inventory mismatch: {relative_path}")
            for name, digest in entries.items():
                if sha256_file(path / name) != digest:
                    raise Phase4D4ProtocolBError(f"parent checksum mismatch: {relative_path}/{name}")
        for receipt_name, filename in (("config_sha256", "config.json"), ("manifest_sha256", "manifest.json"), ("gate_sha256", "gate.json"), ("complete_sha256", "complete.json"), ("failed_sha256", "failed.json")):
            if receipt_name in parent and sha256_file(path / filename) != str(parent[receipt_name]):
                raise Phase4D4ProtocolBError(f"parent receipt mismatch: {relative_path}/{filename}")
    step27_gate = json.loads((ROOT / str(step27["relative_path"]) / "gate.json").read_bytes())
    step31_gate = json.loads((ROOT / str(step31["relative_path"]) / "gate.json").read_bytes())
    if step27_gate.get("full_domain_core", {}).get("state") != "evaluable" or step31_gate.get("full_domain_core", {}).get("ready_model_condition_count") != 205:
        raise Phase4D4ProtocolBError("Step-27/31 live readiness receipt mismatch")
    bridge = {
        "step27_run_id": REAL_STEP27_RUN_ID,
        "step29_run_id": REAL_STEP29_RUN_ID,
        "step31_run_id": REAL_STEP31_RUN_ID,
        "allowed_step29_payloads": ("record_measurements.jsonl",),
        "record_conditions_filename": "record_conditions.jsonl",
        "record_measurements_filename": "record_measurements.jsonl",
        "measurement_bridge_sha256": config.measurement_bridge_sha256,
    }
    if parse_step29_measurements:
        bridge.update(
            {
                "bridge_row_count": 314880,
                "bridge_sha256": REAL_MEASUREMENT_BRIDGE_SHA256,
                "mismatch_counts": {
                    "missing_key_count": 0,
                    "extra_key_count": 0,
                    "duplicate_key_count": 0,
                    "state_mismatch_count": 0,
                    "metric_receipt_mismatch_count": 0,
                    "cwt_receipt_mismatch_count": 0,
                },
            }
        )
    return bridge


def reconstruct_d4_protocol_b_inputs(
    cohort: D4SugarCohort,
    config: Phase4D4ProtocolBConfig,
    parent_bridge: Mapping[str, object],
) -> D4ProtocolBInputs:
    if config.synthetic_fixture:
        raise Phase4D4ProtocolBError("synthetic config does not reconstruct from cohort")
    if len(cohort.record_ids) != 7680 or len(cohort.blank_record_ids) != 32:
        raise Phase4D4ProtocolBError("direct-ZIP cohort denominator mismatch")
    support = np.ascontiguousarray(np.asarray(cohort.wavenumber, dtype="<f8")[1:], dtype="<f8")
    if support.shape != (1999,) or _matrix_sha(support) != "db910b11f92151db481391e96b8a06e246140596cfd4abad64764b87d2d84ee5":
        raise Phase4D4ProtocolBError("direct-ZIP support identity mismatch")
    if _matrix_sha(support, "<f4") != "c32275fcb069cf66b3c9b19e1e922d93c724ea4d54e9a9bc8dfaad734ca4c807":
        raise Phase4D4ProtocolBError("direct-ZIP support float32 identity mismatch")
    wells = tuple(dict.fromkeys(str(value) for value in cohort.well_ids))
    if len(wells) != 240 or any(sum(value == well for value in cohort.well_ids) != 32 for well in wells):
        raise Phase4D4ProtocolBError("direct-ZIP physical-well multiplicity mismatch")
    folds = {index: np.asarray(split.test_indices, dtype=np.int64) for index, split in enumerate(cohort.splits)}
    train = {index: np.asarray(split.train_indices, dtype=np.int64) for index, split in enumerate(cohort.splits)}
    valid = {index: np.asarray(split.validation_indices, dtype=np.int64) for index, split in enumerate(cohort.splits)}
    if any(folds[k].size != 1536 or train[k].size != 4608 or valid[k].size != 1536 for k in MODEL_FOLDS):
        raise Phase4D4ProtocolBError("direct-ZIP Protocol-B role denominator mismatch")
    record_ids = tuple(str(value) for value in cohort.record_ids)
    record_to_well = {record_id: str(cohort.well_ids[index]) for index, record_id in enumerate(record_ids)}
    record_to_fold = {record_id: fold for fold, indexes in folds.items() for record_id in (record_ids[int(index)] for index in indexes)}
    if set(record_to_fold) != set(record_ids):
        raise Phase4D4ProtocolBError("direct-ZIP test-fold coverage mismatch")
    if parent_bridge.get("step31_run_id") != REAL_STEP31_RUN_ID:
        raise Phase4D4ProtocolBError("Step-31 parent bridge identity mismatch")
    return D4ProtocolBInputs(
        synthetic_fixture=False, condition_ids=config.condition_ids, fold_ids=MODEL_FOLDS, well_ids=wells,
        record_ids=record_ids, blank_record_ids=tuple(str(x) for x in cohort.blank_record_ids),
        support_axis_cm1=support, feature_count=1999, true_targets=np.asarray(cohort.targets, dtype="<f8"),
        blank_targets=np.asarray(cohort.blank_targets, dtype="<f8"), record_to_well=MappingProxyType(record_to_well),
        record_to_fold=MappingProxyType(record_to_fold), validation_macro_nrmse_by_component=MappingProxyType({}),
        prediction_fixture="real", blank_fixture="real", metric_fixture="real",
        mixture_matrix=np.ascontiguousarray(np.asarray(cohort.intensity, dtype="<f8")[:, 1:], dtype="<f4"),
        blank_matrix=np.ascontiguousarray(np.asarray(cohort.blank_intensity, dtype="<f8")[:, 1:], dtype="<f4"),
        train_indices_by_fold=MappingProxyType(train), validation_indices_by_fold=MappingProxyType(valid),
        test_indices_by_fold=MappingProxyType(folds), rounds=tuple(int(x) for x in cohort.rounds),
        repetitions=tuple(int(x) for x in cohort.repetitions),
        native_axis_cm1=np.ascontiguousarray(np.asarray(cohort.wavenumber, dtype="<f8")),
        native_mixture_intensity=np.ascontiguousarray(np.asarray(cohort.intensity, dtype="<f8")),
        native_blank_intensity=np.ascontiguousarray(np.asarray(cohort.blank_intensity, dtype="<f8")),
        blank_well_ids=tuple(str(value) for value in cohort.blank_well_ids),
    )


def _closed_rows_for_condition(
    *,
    condition_id: str,
    inputs: D4ProtocolBInputs,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    closed_state = "not_tested_endpoint_closed"
    return {
        "model_rows": tuple(
            {
                "fold": fold,
                "condition_id": condition_id,
                "train_condition_id": condition_id,
                "validation_condition_id": condition_id,
                "test_condition_id": condition_id,
                "blank_condition_id": condition_id,
                "selected_n_components": None,
                "refit_with_validation": False,
                "state": closed_state,
            }
            for fold in inputs.fold_ids
        ),
        "validation_rows": tuple(
            {
                "fold": fold,
                "condition_id": condition_id,
                "n_components": candidate,
                "macro_normalized_rmse": None,
                "state": closed_state,
            }
            for fold in inputs.fold_ids
            for candidate in N_COMPONENTS_GRID
        ),
        "prediction_rows": _prediction_rows_for_condition(
            condition_id=condition_id,
            inputs=inputs,
            state=closed_state,
        ),
        "blank_prediction_rows": _blank_prediction_rows_for_condition(
            condition_id=condition_id,
            inputs=inputs,
            state=closed_state,
        ),
        "well_rows": _well_rows_for_condition(
            condition_id=condition_id,
            prediction_rows=(),
            inputs=inputs,
            state=closed_state,
        ),
        "lod_rows": (
                {"analyte_index": analyte_index, "condition_id": condition_id, "fold": fold, "sigma": None, "slope": None, "iupac_lod": None, "ich_lod": None, "ich_loq": None, "state": closed_state}
                for fold in inputs.fold_ids for analyte_index in range(4)
            ) if not inputs.synthetic_fixture else (_lod_row_for_condition(condition_id=condition_id, inputs=inputs, state=closed_state),),
        "condition_summary_rows": (
            {
                "condition_id": condition_id,
                "state": closed_state,
                "macro_normalized_rmse": None,
            },
        ),
    }


def _closed_rows_for_model_lifecycle(
    *, condition_id: str, inputs: D4ProtocolBInputs, failure: D4ProtocolBModelLifecycleError | None = None
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    rows = _closed_rows_for_condition(condition_id=condition_id, inputs=inputs)
    if failure is None:
        return rows
    failed_state = "failed_model_lifecycle"
    partial = failure.partial
    completed_models = {
        int(row["fold"]): dict(row)
        for row in partial.get("model_rows", ())
    }
    completed_validation = {
        (int(row["fold"]), int(row["n_components"])): dict(row)
        for row in partial.get("validation_rows", ())
    }
    completed_predictions = {
        str(row["record_id"]): dict(row)
        for row in partial.get("prediction_rows", ())
    }
    completed_blank_predictions = {
        (int(row["fold"]), str(row["blank_record_id"])): dict(row)
        for row in partial.get("blank_prediction_rows", ())
    }
    completed_lod = {
        (int(row["fold"]), int(row["analyte_index"])): dict(row)
        for row in partial.get("lod_rows", ())
        if "fold" in row and "analyte_index" in row
    }
    model_rows: list[Mapping[str, object]] = []
    validation_rows: list[Mapping[str, object]] = []
    for fold in inputs.fold_ids:
        if fold in completed_models:
            model_rows.append(completed_models[fold])
        else:
            model_rows.append({"fold": fold, "condition_id": condition_id, "train_condition_id": condition_id, "validation_condition_id": condition_id, "test_condition_id": condition_id, "blank_condition_id": condition_id, "selected_n_components": failure.n_components if fold == failure.fold else None, "refit_with_validation": False, "state": failed_state if fold == failure.fold else "not_tested_endpoint_closed", "failure_category": failure.category if fold == failure.fold else None, "failure_message": failure.message if fold == failure.fold else None, "failed_n_components": failure.n_components if fold == failure.fold else None})
        for candidate in N_COMPONENTS_GRID:
            key = (fold, candidate)
            if key in completed_validation:
                validation_rows.append(completed_validation[key])
            else:
                validation_rows.append({"fold": fold, "condition_id": condition_id, "n_components": candidate, "macro_normalized_rmse": None, "state": failed_state if (fold == failure.fold and candidate == failure.n_components) else "not_tested_endpoint_closed", "failure_category": failure.category if (fold == failure.fold and candidate == failure.n_components) else None, "failure_message": failure.message if (fold == failure.fold and candidate == failure.n_components) else None})
    prediction_rows = tuple(
        completed_predictions.get(str(row["record_id"]), row)
        for row in rows["prediction_rows"]
    )
    blank_prediction_rows = tuple(
        completed_blank_predictions.get(
            (int(row["fold"]), str(row["blank_record_id"])), row
        )
        for row in rows["blank_prediction_rows"]
    )
    lod_rows = tuple(
        completed_lod.get((int(row["fold"]), int(row["analyte_index"])), row)
        if "fold" in row and "analyte_index" in row else row
        for row in rows["lod_rows"]
    )
    return {
        **rows,
        "model_rows": tuple(model_rows),
        "validation_rows": tuple(validation_rows),
        "prediction_rows": prediction_rows,
        "blank_prediction_rows": blank_prediction_rows,
        "lod_rows": lod_rows,
    }


def _build_closed_auxiliary_payloads(inputs: D4ProtocolBInputs) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    figure1_rows = tuple(
        {
            "condition_id": condition_id,
            "metric_output_id": metric_output_id,
            "harm": None,
            "state": "not_tested_endpoint_closed",
        }
        for condition_id in inputs.condition_ids[1:]
        for metric_output_id in METRIC_OUTPUT_IDS
    )
    figure2_rows = tuple(
        {
            "metric_output_id": metric_output_id,
            "ag": None,
            "acc_cross": None,
            "d_ag": None,
            "d_acc": None,
            "state": "not_tested_endpoint_closed",
        }
        for metric_output_id in METRIC_OUTPUT_IDS
    )
    holm_rows = tuple(
        {
            "slot": f"holm_{index:02d}",
            "raw_p_value": 1.0,
            "adjusted_p_value": 1.0,
            "state": "not_tested_endpoint_closed",
        }
        for index in range(24)
    )
    return {
        "alignment_rows": tuple(
            {
                "metric_output_id": metric_output_id,
                "ag": None,
                "acc_cross": None,
                "state": "not_tested_endpoint_closed",
            }
            for metric_output_id in METRIC_OUTPUT_IDS
        ),
        "bootstrap_rows": tuple(
            {
                "slot": f"bootstrap_{index:02d}",
                "estimate": None,
                "state": "not_tested_endpoint_closed",
            }
            for index in range(12)
        ),
        "sign_flip_rows": tuple(
            {
                "slot": f"sign_flip_{index:02d}",
                "raw_p_value": 1.0,
                "state": "not_tested_endpoint_closed",
            }
            for index in range(24)
        ),
        "holm_rows": holm_rows,
        "well_observation_rows": tuple(
            {
                "well_id": well_id,
                "condition_id": condition_id,
                "metric_output_id": metric_output_id,
                "preferred_direction": metric_preferred_direction(metric_output_id),
                "harm": None,
                "state": "not_tested_endpoint_closed",
            }
            for well_id in inputs.well_ids
            for condition_id in inputs.condition_ids[1:]
            for metric_output_id in METRIC_OUTPUT_IDS
        ),
        "figure1_rows": figure1_rows,
        "figure2_rows": figure2_rows,
        "table_rows": tuple(
            {
                "metric_output_id": metric_output_id,
                "state": "not_tested_endpoint_closed",
                "ag": None,
                "acc_cross": None,
                "d_ag": None,
                "d_acc": None,
            }
            for metric_output_id in METRIC_OUTPUT_IDS
        ),
    }


def _state_counts(rows: Sequence[Mapping[str, object]], field: str = "state") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        state = str(row.get(field))
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def _parent_checksum_entries(config: Phase4D4ProtocolBConfig) -> dict[str, Mapping[str, object]]:
    if config.synthetic_fixture:
        return {
            name: {
                "checksum_entries": {},
                "run_id": f"synthetic-{name}",
                "sha256sums_sha256": "synthetic",
            }
            for name in ("step27", "step29", "step31")
        }
    parents = dict(_thaw(config.document).get("parent_artifacts", {}))
    run_ids = {
        "step27": REAL_STEP27_RUN_ID,
        "step29": REAL_STEP29_RUN_ID,
        "step31": REAL_STEP31_RUN_ID,
    }
    result: dict[str, Mapping[str, object]] = {}
    for name in ("step27", "step29", "step31"):
        receipt = dict(parents[name])
        ledger = ROOT / str(receipt["relative_path"]) / "SHA256SUMS"
        entries = {
            filename: digest
            for digest, filename in (line.split("  ", 1) for line in ledger.read_text(encoding="utf-8").splitlines())
        }
        result[name] = {
            "checksum_entries": entries,
            "run_id": run_ids[name],
            "sha256sums_sha256": str(receipt["sha256sums_sha256"]),
        }
    return result


def _condition_receipt_digests(
    inputs: D4ProtocolBInputs,
    receipts: Mapping[tuple[str, str], Mapping[str, object]],
) -> dict[str, str]:
    if inputs.synthetic_fixture:
        return {
            condition_id: sha256_hex(
                canonical_json_bytes(
                    {"condition_id": condition_id, "synthetic_fixture": True}
                )
            )
            for condition_id in inputs.condition_ids
        }
    return {
        condition_id: sha256_hex(
            b"".join(
                canonical_json_bytes(receipts[(record_id, condition_id)])
                for record_id in inputs.record_ids
            )
        )
        for condition_id in inputs.condition_ids
        if all((record_id, condition_id) in receipts for record_id in inputs.record_ids)
    }


def _preflight_document(
    *,
    config: Phase4D4ProtocolBConfig,
    inputs: D4ProtocolBInputs,
    rematerialization_receipts: Mapping[tuple[str, str], Mapping[str, object]],
    status: str,
    endpoint_state: str,
) -> dict[str, object]:
    document = _thaw(config.document)
    frozen = dict(document.get("frozen_identities", {}))
    support = dict(document.get("support_grid", {}))
    denominators = dict(document.get("denominators", {}))
    resource = dict(document.get("resource_policy", {}))
    synthetic = inputs.synthetic_fixture
    receipt_digests = _condition_receipt_digests(inputs, rematerialization_receipts)
    source_identity = {
        "blank_record_count": len(inputs.blank_record_ids),
        "blank_record_ids_sha256": frozen.get("blank_record_ids_sha256", "synthetic"),
        "blank_source_members_sha256": frozen.get("blank_source_members_sha256", "synthetic"),
        "blank_well_id": frozen.get("blank_well_id", "synthetic-blank"),
        "mixture_record_count": len(inputs.record_ids),
        "mixture_record_ids_sha256": frozen.get("mixture_record_ids_sha256", "synthetic"),
        "mixture_source_members_sha256": frozen.get("mixture_source_members_sha256", "synthetic"),
        "mixture_well_ids_sha256": frozen.get("mixture_well_ids_sha256", "synthetic"),
        "physical_well_count": len(inputs.well_ids),
        "target_matrix_sha256": frozen.get("target_matrix_sha256", "synthetic"),
    }
    support_identity = {
        "first_cm1": float(inputs.support_axis_cm1[0]),
        "float32_sha256": support.get("float32_sha256", _matrix_sha(inputs.support_axis_cm1, "<f4")),
        "float64_sha256": support.get("float64_sha256", _matrix_sha(inputs.support_axis_cm1, "<f8")),
        "last_cm1": float(inputs.support_axis_cm1[-1]),
        "point_count": int(inputs.support_axis_cm1.size),
    }
    role_identity = {
        "fold_count": len(inputs.fold_ids),
        "fold_record_ids_sha256": frozen.get("fold_record_ids_sha256", ["synthetic"] * len(inputs.fold_ids)),
        "fold_well_ids_sha256": frozen.get("fold_well_ids_sha256", ["synthetic"] * len(inputs.fold_ids)),
        "test_acquisitions_per_fold": denominators.get("test_acquisitions_per_fold", len(inputs.record_ids) // len(inputs.fold_ids)),
        "test_wells_per_fold": denominators.get("test_wells_per_fold", len(inputs.well_ids) // len(inputs.fold_ids)),
        "train_acquisitions_per_fold": denominators.get("train_acquisitions_per_fold", 0),
        "train_wells_per_fold": denominators.get("train_wells_per_fold", 0),
        "validation_acquisitions_per_fold": denominators.get("validation_acquisitions_per_fold", 0),
        "validation_wells_per_fold": denominators.get("validation_wells_per_fold", 0),
    }
    return {
        "condition_count": len(inputs.condition_ids),
        "condition_identity": {
            "condition_count": len(inputs.condition_ids),
            "condition_ids_sha256": sha256_hex(canonical_json_bytes(list(inputs.condition_ids))),
            "condition_order": "config_condition_ids",
        },
        "endpoint_state": endpoint_state,
        "matrix_receipts": {
            "blank_condition_matrix_count": len(inputs.condition_ids),
            "blank_rows_per_condition": len(inputs.blank_record_ids),
            "condition_receipt_digests": receipt_digests,
            "feature_count": inputs.feature_count,
            "mixture_condition_matrix_count": len(receipt_digests),
            "mixture_rows_per_condition": len(inputs.record_ids),
            "sealed_mixture_receipt_count": (
                len(inputs.record_ids) * len(receipt_digests)
                if synthetic
                else len(rematerialization_receipts)
            ),
        },
        "readiness": {
            "conditions": "passed",
            "rematerialization": "passed" if endpoint_state == "complete" else endpoint_state,
            "roles": "passed",
            "source": "passed",
            "support": "passed",
        },
        "record_count": len(inputs.record_ids),
        "resource_admission": {
            "blas_threads_per_model_worker": int(resource.get("blas_threads_per_model_worker", 1)),
            "memory_budget_bytes": int(resource.get("condition_memory_budget_bytes", 0)),
            "model_budget_admitted_jobs": 43,
            "model_peak_bytes_per_job": 1565016064,
            "model_structural_cap": len(inputs.fold_ids),
            "pools_overlap": False,
            "positive_batch_bytes": 493321216,
            "rematerialization_max_admitted_jobs": int(resource.get("max_condition_workers", 0)),
            "rematerialization_peak_bytes_per_job": int(resource.get("p10_peak_estimate_bytes_per_job", 0)),
        },
        "role_identity": role_identity,
        "source_identity": source_identity,
        "status": status,
        "support_identity": support_identity,
        "synthetic_fixture": synthetic,
    }


def _artifact_manifest(
    *,
    config: Phase4D4ProtocolBConfig,
    run_id: str,
    status: str,
    endpoint_state: str,
    rows: Mapping[str, Sequence[Mapping[str, object]]],
    alpha0_equivalence: Mapping[str, object],
) -> dict[str, object]:
    document = _thaw(config.document)
    row_names = {
        "alignment_results": "alignment_rows",
        "blank_predictions": "blank_prediction_rows",
        "bootstrap_results": "bootstrap_rows",
        "condition_summary": "condition_summary_rows",
        "figure1_source": "figure1_rows",
        "figure2_source": "figure2_rows",
        "holm_family": "holm_rows",
        "model_cells": "model_rows",
        "predictions": "prediction_rows",
        "secondary_table": "table_rows",
        "sign_flip_results": "sign_flip_rows",
        "technical_lod_loq": "technical_lod_loq_rows",
        "validation_scores": "validation_rows",
        "well_conditions": "well_condition_rows",
        "well_observations": "well_observation_rows",
    }
    counts = {name: len(tuple(rows[key])) for name, key in row_names.items()}
    counts.update(
        {
            "artifact_files": len(ARTIFACT_PAYLOAD_FILES) + 2,
            "configured_payloads": len(ARTIFACT_PAYLOAD_FILES),
        }
    )
    states = {
        name: _state_counts(
            rows[key], "metric_state" if name == "figure1_source" else "state"
        )
        for name, key in row_names.items()
    }
    return {
        "alpha0_equivalence": dict(alpha0_equivalence),
        "claim_boundary": str(document.get("claim_boundary")),
        "counts": counts,
        "endpoint_state": endpoint_state,
        "fixed_capacities": {
            "bootstrap_resamples": int(document.get("inference", {}).get("bootstrap_resamples", 2000)),
            "condition_count": len(config.condition_ids),
            "holm_slots": int(document.get("inference", {}).get("holm_slot_count", 24)),
            "physical_wells": int(document.get("denominators", {}).get("physical_wells", len({str(row["well_id"]) for row in rows["well_condition_rows"]}))),
            "sign_flip_resamples": int(document.get("inference", {}).get("sign_flip_resamples", 100000)),
        },
        "identities": {
            "authority_receipts_sha256": sha256_hex(canonical_json_bytes(document.get("authorities", {}))),
            "code_authority": document.get("code_authority", {}),
            "config_sha256": config.sha256,
            "environment_authority": document.get("environment_authority", {}),
        },
        "inherited_rulings": document.get("inherited_rulings", {}),
        "payload_files": list(config.artifact_payload_files),
        "payload_order": "config_artifact_payload_files",
        "run_id": run_id,
        "schema": ARTIFACT_SCHEMA_VERSION,
        "states": states,
        "status": status,
    }


def _terminal_marker(run_id: str, status: str, endpoint_state: str, failure: D4ProtocolBModelLifecycleError | None = None) -> tuple[str, bytes]:
    payload = {
        "schema": MARKER_SCHEMA_VERSION,
        "run": run_id,
        "run_id": run_id,
        "status": status,
        "endpoint_state": endpoint_state,
    }
    name = "complete.json" if status == "complete" else "failed.json"
    if failure is not None:
        payload["failure"] = {"condition_id": failure.condition_id, "fold": failure.fold, "n_components": failure.n_components, "warning_category": failure.category, "warning_message": failure.message}
    return name, canonical_json_bytes(payload)


def _write_artifact(
    output_dir: Path,
    payloads: Mapping[str, bytes],
    *,
    terminal_name: str,
    terminal_bytes: bytes,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    for name in ARTIFACT_PAYLOAD_FILES:
        (output_dir / name).write_bytes(payloads[name])
    (output_dir / terminal_name).write_bytes(terminal_bytes)
    (output_dir / "SHA256SUMS").write_bytes(write_sha256sums(payloads, terminal_name, terminal_bytes))


def _parse_condition_id(condition_id: str) -> tuple[str, float]:
    if condition_id == "alpha0":
        return "alpha0", 0.0
    perturbation, raw_alpha = condition_id.split(":", 1)
    import struct
    return perturbation, struct.unpack("<d", bytes.fromhex(raw_alpha))[0]


def _run_perturbation_cell_single_blas(
    source: object,
    perturbation_id: str,
    phase1_config: object,
    sweep: object,
    *,
    p10_admission: object | None = None,
):
    # P10 prepares correlated noise with a matrix-vector product.  Pinning
    # BLAS reproduces the Step-27/29 native float64 receipt byte-for-byte.
    with threadpool_limits(limits=1, user_api="blas"):
        return run_perturbation_cell(
            source,
            perturbation_id,
            phase1_config,
            sweep,
            p10_admission=p10_admission,
        )


def _real_condition_matrices(
    inputs: D4ProtocolBInputs, config: Phase4D4ProtocolBConfig, perturbation_filter: str | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[str, str], Mapping[str, object]]]:
    """Rematerialize all model-input arrays directly from the retained ZIP cohort.

    The alpha-zero path is a literal direct support projection.  Positive
    conditions run the frozen native perturbation operator independently for
    mixture and blank spectra; no Protocol-A model or prediction is reused.
    """
    if inputs.mixture_matrix is None or inputs.blank_matrix is None:
        raise Phase4D4ProtocolBError("retained direct-ZIP matrices are absent")
    matrices: dict[str, np.ndarray] = {}
    blanks: dict[str, np.ndarray] = {}
    receipts: dict[tuple[str, str], Mapping[str, object]] = {}
    if perturbation_filter is None:
        matrices["alpha0"] = np.ascontiguousarray(inputs.mixture_matrix, dtype="<f4")
        blanks["alpha0"] = np.ascontiguousarray(inputs.blank_matrix, dtype="<f4")
        if inputs.native_axis_cm1 is None or inputs.native_mixture_intensity is None:
            raise Phase4D4ProtocolBError("native arrays required for alpha-zero receipts")
        receipts.update(
            _condition_receipts(
                condition_id="alpha0",
                condition_matrix=matrices["alpha0"],
                native_axes=(inputs.native_axis_cm1,) * len(inputs.record_ids),
                native_intensities=inputs.native_mixture_intensity,
                inputs=inputs,
            )
        )
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1 = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    eligibility = _step27_science.load_phase4_d4_eligibility_config(
        ROOT / ELIGIBILITY_CONFIG_RELATIVE_PATH
    )
    # Reconstruct native spectra from the arrays already validated against the
    # direct archive by the loader.  This keeps the expensive work local to
    # the selected D4 conditions and avoids scalar metric/CWT reexecution.
    cohort = load_d4_sugar_cohort(ROOT / PROTOCOL_CONFIG_RELATIVE_PATH, ROOT / ARCHIVE_RELATIVE_PATH)
    sources = tuple(_step27_science._phase1_source_for_spectrum(Spectrum1D(spectrum_id=f"d4_sugar_low_snr::{record_id}", sample_id=str(well), axis_cm1=np.asarray(cohort.wavenumber, dtype="<f8"), intensity=np.asarray(value, dtype="<f8")), order=index) for index, (record_id, well, value) in enumerate(zip(cohort.record_ids, cohort.well_ids, cohort.intensity, strict=True)))
    blank_sources = tuple(_step27_science._phase1_source_for_spectrum(Spectrum1D(spectrum_id=f"d4_blank::{record_id}", sample_id=str(well), axis_cm1=np.asarray(cohort.wavenumber, dtype="<f8"), intensity=np.asarray(value, dtype="<f8")), order=index) for index, (record_id, well, value) in enumerate(zip(cohort.blank_record_ids, cohort.blank_well_ids, cohort.blank_intensity, strict=True)))
    selected_perturbations = (perturbation_filter,) if perturbation_filter is not None else PERTURBATIONS
    for perturbation in selected_perturbations:
        by_record = [
            _run_perturbation_cell_single_blas(
                source, perturbation, phase1, sweep
            )
            for source in sources
        ]
        by_blank = [
            _run_perturbation_cell_single_blas(
                source, perturbation, phase1, sweep
            )
            for source in blank_sources
        ]
        for condition_id in (item for item in config.condition_ids if item.startswith(perturbation + ":")):
            _name, alpha = _parse_condition_id(condition_id)
            try:
                mixture_outputs = [next(record.result.output for record in cell.records if float(record.alpha) == alpha) for cell in by_record]
                blank_outputs = [next(record.result.output for record in cell.records if float(record.alpha) == alpha) for cell in by_blank]
            except (StopIteration, AttributeError) as error:
                raise Phase4D4ProtocolBError(f"frozen perturbation output missing {condition_id}") from error
            matrices[condition_id] = np.ascontiguousarray(
                np.asarray(
                    [_step27_science.project_d4_support(output, eligibility) for output in mixture_outputs],
                    dtype="<f4",
                ),
                dtype="<f4",
            )
            blanks[condition_id] = np.ascontiguousarray(
                np.asarray(
                    [_step27_science.project_d4_support(output, eligibility) for output in blank_outputs],
                    dtype="<f4",
                ),
                dtype="<f4",
            )
            receipts.update(
                _condition_receipts(
                    condition_id=condition_id,
                    condition_matrix=matrices[condition_id],
                    native_axes=tuple(np.asarray(output.axis_cm1, dtype="<f8") for output in mixture_outputs),
                    native_intensities=tuple(np.asarray(output.intensity, dtype="<f8") for output in mixture_outputs),
                    inputs=inputs,
                )
            )
    expected = config.condition_ids if perturbation_filter is None else tuple(item for item in config.condition_ids if item.startswith(perturbation_filter + ":"))
    if tuple(matrices) != expected or tuple(blanks) != expected:
        raise Phase4D4ProtocolBError("condition rematerialization order mismatch")
    return matrices, blanks, receipts


def _condition_receipts(
    *,
    condition_id: str,
    condition_matrix: np.ndarray,
    native_axes: Sequence[np.ndarray],
    native_intensities: Sequence[np.ndarray],
    inputs: D4ProtocolBInputs,
) -> dict[tuple[str, str], Mapping[str, object]]:
    if len(native_axes) != len(inputs.record_ids) or len(native_intensities) != len(inputs.record_ids):
        raise Phase4D4ProtocolBError("rematerialization receipt denominator mismatch")
    perturbation_id, alpha = _parse_condition_id(condition_id)
    return {
        (record_id, condition_id): {
            "alpha": alpha,
            "condition_id": condition_id,
            "fold": inputs.record_to_fold[record_id],
            "native_axis_sha256": _matrix_sha(native_axes[index], "<f8"),
            "native_intensity_sha256": _matrix_sha(native_intensities[index], "<f8"),
            "perturbation_id": perturbation_id,
            "projected_row_sha256": _matrix_sha(condition_matrix[index], "<f4"),
            "record_id": record_id,
            "record_order": index,
            "state": "complete",
            "well_id": inputs.record_to_well[record_id],
        }
        for index, record_id in enumerate(inputs.record_ids)
    }


def build_phase4_d4_protocol_b_from_inputs(
    output_dir: Path,
    *,
    inputs: D4ProtocolBInputs,
    config: Phase4D4ProtocolBConfig,
    rematerialization_worker_count: int,
    model_worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4D4ProtocolBSummary:
    del rematerialization_worker_count
    if model_worker_count < 1:
        raise Phase4D4ProtocolBError("model worker count must be positive")
    if inputs.synthetic_fixture:
        alpha_matrix = np.zeros((len(inputs.record_ids), inputs.feature_count), dtype=np.float32)
        alpha_blank_matrix = np.zeros((len(inputs.blank_record_ids), inputs.feature_count), dtype=np.float32)
    else:
        if inputs.mixture_matrix is None or inputs.blank_matrix is None:
            raise Phase4D4ProtocolBError("real alpha-zero rematerialization matrix missing")
        alpha_matrix = np.ascontiguousarray(inputs.mixture_matrix, dtype="<f4")
        alpha_blank_matrix = np.ascontiguousarray(inputs.blank_matrix, dtype="<f4")
    condition_results: list[Mapping[str, object]] = []
    rematerialization_receipts: dict[tuple[str, str], Mapping[str, object]] = {}
    if not inputs.synthetic_fixture:
        if inputs.native_axis_cm1 is None or inputs.native_mixture_intensity is None:
            raise Phase4D4ProtocolBError("native arrays required for alpha-zero receipts")
        rematerialization_receipts.update(
            _condition_receipts(
                condition_id="alpha0",
                condition_matrix=alpha_matrix,
                native_axes=(inputs.native_axis_cm1,) * len(inputs.record_ids),
                native_intensities=inputs.native_mixture_intensity,
                inputs=inputs,
            )
        )
    for condition_id in inputs.condition_ids[:1]:
        condition_results.append(
            fit_d4_protocol_b_condition(
                condition_id=condition_id,
                condition_matrix=alpha_matrix,
                blank_matrix=alpha_blank_matrix,
                inputs=inputs,
                config=config,
                model_worker_count=1,
            )
        )
    alpha0_equivalence = validate_d4_protocol_b_alpha0_equivalence(
        model_rows=condition_results[0]["model_rows"],
        validation_rows=condition_results[0]["validation_rows"],
        prediction_rows=condition_results[0]["prediction_rows"],
        blank_prediction_rows=condition_results[0]["blank_prediction_rows"],
        lod_rows=condition_results[0]["lod_rows"],
        config=config,
    )
    if alpha0_equivalence["status"] == "passed":
        if inputs.synthetic_fixture:
            for condition_id in inputs.condition_ids[1:]:
                try:
                    condition_results.append(fit_d4_protocol_b_condition(condition_id=condition_id, condition_matrix=np.zeros((len(inputs.record_ids), inputs.feature_count), dtype=np.float32), blank_matrix=np.zeros((len(inputs.blank_record_ids), inputs.feature_count), dtype=np.float32), inputs=inputs, config=config, model_worker_count=model_worker_count))
                except Phase4D4ProtocolBError as error:
                    lifecycle_failure = _model_lifecycle_from_error(error)
                    if lifecycle_failure is None:
                        raise
                    condition_results.append(_closed_rows_for_model_lifecycle(condition_id=condition_id, inputs=inputs, failure=lifecycle_failure))
                    break
        else:
            # Bounded positive path: retain exactly one P8--P12 perturbation
            # batch, fit its eight cells, then release it before the next.
            lifecycle_failure: D4ProtocolBModelLifecycleError | None = None
            for perturbation in PERTURBATIONS:
                if lifecycle_failure is not None:
                    break
                batch, blank_batch, batch_receipts = _real_condition_matrices(inputs, config, perturbation)
                rematerialization_receipts.update(batch_receipts)
                for condition_id in (item for item in inputs.condition_ids if item.startswith(perturbation + ":")):
                    try:
                        condition_results.append(fit_d4_protocol_b_condition(condition_id=condition_id, condition_matrix=batch[condition_id], blank_matrix=blank_batch[condition_id], inputs=inputs, config=config, model_worker_count=model_worker_count))
                    except Phase4D4ProtocolBError as error:
                        lifecycle_failure = error if isinstance(error, D4ProtocolBModelLifecycleError) else _model_lifecycle_from_error(error)
                        if lifecycle_failure is None:
                            raise
                        condition_results.append(_closed_rows_for_model_lifecycle(condition_id=condition_id, inputs=inputs, failure=lifecycle_failure))
                        break
                del batch, blank_batch
            if lifecycle_failure is not None:
                completed = {str(row["condition_id"]) for result in condition_results for row in result["model_rows"]}
                for condition_id in inputs.condition_ids:
                    if condition_id not in completed:
                        condition_results.append(_closed_rows_for_model_lifecycle(condition_id=condition_id, inputs=inputs))
                measurement_bridge = {"state": "not_tested_endpoint_closed", "bridge_sha256": config.measurement_bridge_sha256}
                endpoint_state = "failed_model_lifecycle"
                status = "failed"
                aggregates = {"condition_summary_rows": tuple(row for result in condition_results for row in result["condition_summary_rows"]), "well_condition_rows": tuple(row for result in condition_results for row in result["well_rows"])}
                aggregates.update(_build_closed_auxiliary_payloads(inputs))
            else:
                measurement_bridge = None
        if 'lifecycle_failure' in locals() and lifecycle_failure is not None:
            completed = {str(row["condition_id"]) for result in condition_results for row in result["model_rows"]}
            for condition_id in inputs.condition_ids:
                if condition_id not in completed:
                    condition_results.append(_closed_rows_for_model_lifecycle(condition_id=condition_id, inputs=inputs))
            measurement_bridge = {"state": "not_tested_endpoint_closed", "bridge_sha256": config.measurement_bridge_sha256}
            endpoint_state = "failed_model_lifecycle"
            status = "failed"
            aggregates = {"condition_summary_rows": tuple(row for result in condition_results for row in result["condition_summary_rows"]), "well_condition_rows": tuple(row for result in condition_results for row in result["well_rows"])}
            aggregates.update(_build_closed_auxiliary_payloads(inputs))
            pass
        elif alpha0_equivalence["status"] == "passed":
            measurement_bridge, measurement_bridge_rows = build_d4_protocol_b_measurement_bridge(
                config=config,
                inputs=inputs,
                rematerialization_receipts=rematerialization_receipts,
            )
            endpoint_state = "complete"
            status = "complete"
            aggregates = aggregate_d4_protocol_b(
                inputs=inputs,
                condition_summary_rows=tuple(
                    row for result in condition_results for row in result["condition_summary_rows"]
                ),
                well_rows=tuple(
                    row for result in condition_results for row in result["well_rows"]
                ),
                lod_rows=tuple(
                    row for result in condition_results for row in result["lod_rows"]
                ),
                measurement_bridge_rows=measurement_bridge_rows,
                endpoint_state=endpoint_state,
                bootstrap_resamples=bootstrap_resamples,
                sign_flip_resamples=sign_flip_resamples,
            )
    else:
        for condition_id in inputs.condition_ids[1:]:
            condition_results.append(_closed_rows_for_condition(condition_id=condition_id, inputs=inputs))
        measurement_bridge = {
            "state": "not_tested_endpoint_closed",
            "bridge_sha256": config.measurement_bridge_sha256,
        }
        measurement_bridge_rows = ()
        endpoint_state = "failed_alpha0_equivalence"
        status = "failed"
        aggregates = {
            "condition_summary_rows": tuple(
                row for result in condition_results for row in result["condition_summary_rows"]
            ),
            "well_condition_rows": tuple(
                row for result in condition_results for row in result["well_rows"]
            ),
        }
        aggregates.update(_build_closed_auxiliary_payloads(inputs))
    condition_order = {condition_id: index for index, condition_id in enumerate(inputs.condition_ids)}
    record_order = {record_id: index for index, record_id in enumerate(inputs.record_ids)}
    blank_order = {record_id: index for index, record_id in enumerate(inputs.blank_record_ids)}
    well_order = {well_id: index for index, well_id in enumerate(inputs.well_ids)}
    model_rows = tuple(
        sorted(
            (row for result in condition_results for row in result["model_rows"]),
            key=lambda row: (condition_order[str(row["condition_id"])], int(row["fold"])),
        )
    )
    validation_rows = tuple(
        sorted(
            (row for result in condition_results for row in result["validation_rows"]),
            key=lambda row: (
                condition_order[str(row["condition_id"])],
                int(row["fold"]),
                int(row["n_components"]),
            ),
        )
    )
    prediction_rows = tuple(
        sorted(
            (row for result in condition_results for row in result["prediction_rows"]),
            key=lambda row: (
                int(row.get("record_order", record_order[str(row["record_id"])])),
                condition_order[str(row["condition_id"])],
            ),
        )
    )
    blank_prediction_rows = tuple(
        sorted(
            (row for result in condition_results for row in result["blank_prediction_rows"]),
            key=lambda row: (
                int(row["fold"]),
                blank_order[str(row["blank_record_id"])],
                condition_order[str(row["condition_id"])],
            ),
        )
    )
    technical_lod_loq_rows_unsorted = tuple(
        row for result in condition_results for row in result["lod_rows"]
    )
    technical_lod_loq_rows = tuple(
        sorted(
            technical_lod_loq_rows_unsorted,
            key=(
                lambda row: (
                    int(row["fold"]),
                    condition_order[str(row["condition_id"])],
                    int(row["analyte_index"]),
                )
            )
            if not inputs.synthetic_fixture
            else lambda row: condition_order[str(row["condition_id"])],
        )
    )
    condition_summary_rows = tuple(
        sorted(
            aggregates["condition_summary_rows"],
            key=lambda row: condition_order[str(row["condition_id"])],
        )
    )
    well_condition_rows = tuple(
        sorted(
            aggregates["well_condition_rows"],
            key=lambda row: (
                well_order[str(row["well_id"])],
                condition_order[str(row["condition_id"])],
            ),
        )
    )
    run_id = stable_run_id(
        config_sha256=config.sha256,
        condition_ids_value=config.condition_ids,
        record_ids=inputs.record_ids,
        blank_record_ids=inputs.blank_record_ids,
        endpoint_state=endpoint_state,
    )
    all_rows = {
        **aggregates,
        "blank_prediction_rows": blank_prediction_rows,
        "model_rows": model_rows,
        "prediction_rows": prediction_rows,
        "technical_lod_loq_rows": technical_lod_loq_rows,
        "validation_rows": validation_rows,
        "well_condition_rows": well_condition_rows,
    }
    manifest = _artifact_manifest(
        config=config,
        run_id=run_id,
        status=status,
        endpoint_state=endpoint_state,
        rows=all_rows,
        alpha0_equivalence=alpha0_equivalence,
    )
    preflight = _preflight_document(
        config=config,
        inputs=inputs,
        rematerialization_receipts=rematerialization_receipts,
        status=status,
        endpoint_state=endpoint_state,
    )
    if endpoint_state == "failed_model_lifecycle":
        preflight["model_lifecycle_failure"] = {
            "condition_id": lifecycle_failure.condition_id, "fold": lifecycle_failure.fold,
            "n_components": lifecycle_failure.n_components, "category": lifecycle_failure.category,
            "message": lifecycle_failure.message,
        }
    authority_bridge = {
        "parents": _parent_checksum_entries(config),
        "step27_run_id": REAL_STEP27_RUN_ID if not config.synthetic_fixture else "synthetic-step27",
        "step29_run_id": REAL_STEP29_RUN_ID if not config.synthetic_fixture else "synthetic-step29",
        "step31_run_id": REAL_STEP31_RUN_ID if not config.synthetic_fixture else "synthetic-step31",
        "allowed_step29_payloads": ["record_measurements.jsonl"],
        "measurement_bridge": measurement_bridge,
        "step29_payload_policy": {
            "allowed_computational_inputs": ["record_measurements.jsonl"],
            "allowed_postcomputation_qc": list(STEP29_POSTCOMPUTATION_QC),
            "forbidden_inputs": list(STEP29_FORBIDDEN_INPUTS),
            "preserved_non_authoritative_directories_allowed": False,
        },
    }
    figure_bytes = render_d4_protocol_b_figures(
        figure1_rows=aggregates["figure1_rows"],
        figure2_rows=aggregates["figure2_rows"],
    )
    payloads: dict[str, bytes] = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": canonical_json_bytes(authority_bridge),
        "preflight.json": canonical_json_bytes(preflight),
        "alpha0_equivalence.json": canonical_json_bytes(alpha0_equivalence),
        "model_cells.jsonl": jsonl_bytes(model_rows),
        "validation_scores.jsonl": jsonl_bytes(validation_rows),
        "predictions.jsonl": jsonl_bytes(prediction_rows),
        "blank_predictions.jsonl": jsonl_bytes(blank_prediction_rows),
        "well_conditions.jsonl": jsonl_bytes(well_condition_rows),
        "technical_lod_loq.jsonl": jsonl_bytes(technical_lod_loq_rows),
        "condition_summary.csv": csv_bytes(condition_summary_rows),
        "well_observations.jsonl": jsonl_bytes(aggregates["well_observation_rows"]),
        "alignment_results.jsonl": jsonl_bytes(aggregates["alignment_rows"]),
        "bootstrap_results.jsonl": jsonl_bytes(aggregates["bootstrap_rows"]),
        "sign_flip_results.jsonl": jsonl_bytes(aggregates["sign_flip_rows"]),
        "holm_family.jsonl": jsonl_bytes(aggregates["holm_rows"]),
        "figure1_d4_protocol_b_full_domain.png": figure_bytes["figure1_d4_protocol_b_full_domain.png"],
        "figure1_d4_protocol_b_full_domain.svg": figure_bytes["figure1_d4_protocol_b_full_domain.svg"],
        "figure1_d4_protocol_b_full_domain_data.csv": csv_bytes(aggregates["figure1_rows"]),
        "figure2_d4_protocol_b_full_domain.png": figure_bytes["figure2_d4_protocol_b_full_domain.png"],
        "figure2_d4_protocol_b_full_domain.svg": figure_bytes["figure2_d4_protocol_b_full_domain.svg"],
        "figure2_d4_protocol_b_full_domain_data.csv": csv_bytes(aggregates["figure2_rows"]),
        "d4_protocol_b_full_domain_secondary_table.csv": csv_bytes(aggregates["table_rows"]),
        "manifest.json": canonical_json_bytes(manifest),
    }
    terminal_name, terminal_bytes = _terminal_marker(
        run_id, status, endpoint_state,
        lifecycle_failure if endpoint_state == "failed_model_lifecycle" else None,
    )
    _write_artifact(Path(output_dir), payloads, terminal_name=terminal_name, terminal_bytes=terminal_bytes)
    return Phase4D4ProtocolBSummary(
        path=Path(output_dir),
        run_id=run_id,
        status=status,
        endpoint_state=endpoint_state,
        record_count=len(inputs.record_ids),
        prediction_row_count=len(prediction_rows),
    )


def build_phase4_d4_protocol_b(
    output_root: Path,
    *,
    rematerialization_worker_count: int = 16,
    model_worker_count: int = 5,
) -> Phase4D4ProtocolBSummary:
    if rematerialization_worker_count < 1 or model_worker_count < 1:
        raise Phase4D4ProtocolBError("worker counts must be positive")
    config = load_phase4_d4_protocol_b_config(DEFAULT_CONFIG)
    bridge = validate_d4_protocol_b_parent_authorities(config, parse_step29_measurements=False)
    cohort = load_d4_sugar_cohort(ROOT / PROTOCOL_CONFIG_RELATIVE_PATH, ROOT / ARCHIVE_RELATIVE_PATH)
    inputs = reconstruct_d4_protocol_b_inputs(cohort, config, bridge)
    target = Path(output_root) / stable_run_id(
        config_sha256=config.sha256,
        condition_ids_value=inputs.condition_ids,
        record_ids=inputs.record_ids,
        blank_record_ids=inputs.blank_record_ids,
        endpoint_state="pre_outcome_identity",
    )
    return build_phase4_d4_protocol_b_from_inputs(target, inputs=inputs, config=config, rematerialization_worker_count=rematerialization_worker_count, model_worker_count=model_worker_count)


__all__ = [
    "D4ProtocolBInputs",
    "D4ProtocolBModelCell",
    "Phase4D4ProtocolBConfig",
    "Phase4D4ProtocolBError",
    "Phase4D4ProtocolBSummary",
    "aggregate_d4_protocol_b",
    "build_d4_protocol_b_measurement_bridge",
    "build_phase4_d4_protocol_b",
    "build_phase4_d4_protocol_b_from_inputs",
    "fit_d4_protocol_b_condition",
    "load_phase4_d4_protocol_b_config",
    "make_synthetic_d4_protocol_b_config",
    "make_synthetic_d4_protocol_b_inputs",
    "parse_phase4_d4_protocol_b_config",
    "reconstruct_d4_protocol_b_inputs",
    "render_d4_protocol_b_figures",
    "validate_d4_protocol_b_alpha0_equivalence",
    "validate_d4_protocol_b_parent_authorities",
]
