from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import platform
import warnings
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from rpe.alignment import AlignmentObservation, AlignmentValidationError, alignment_gap, cross_perturbation_accuracy, compare_alignment, bulk_paired_cluster_bootstrap, holm_step_down, paired_contribution_sign_flip
from rpe.evaluation import PreferredDirection
from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.evaluation import Spectrum1D
from rpe.runner.d2_selection import D2SelectionValidationError, validate_d2_few_shot_selection
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.perturb import load_perturbation_sweep_config
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import io

from rpe.runner.phase4_d2_protocol_b_authority import (
    ALPHAS,
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_PAYLOAD_FILES,
    CLAIM_BOUNDARY,
    CODE_RELATIVE_PATHS,
    CONFIG_BYTES,
    CONFIG_SHA256,
    EXPERIMENT_ID,
    METRIC_OUTPUT_IDS,
    PERTURBATIONS,
    REAL_CONDITION_BRIDGE_SHA256,
    SCHEMA_VERSION,
    SHOTS,
    TERMINAL_MARKERS,
    alpha_hex,
    canonical_json_bytes,
    condition_ids,
    csv_bytes,
    jsonl_bytes,
    sha256_hex,
    write_sha256sums,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d2_protocol_b_full_domain_v1.json"
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d2_protocol_b_authority.py"
RUN_PREFIX = "phase4-d2-protocol-b-full-domain-"
FORBIDDEN_STEP15_NAMES = frozenset(
    {
        "model_cells.jsonl",
        "validation_scores.jsonl",
        "downstream_rows.jsonl",
        "predictions.jsonl",
        "seed_class_conditions.jsonl",
        "condition_summary.csv",
        "class_observations.jsonl",
        "alignment_results.jsonl",
        "bootstrap_results.jsonl",
        "sign_flip_results.jsonl",
        "holm_family.jsonl",
    }
)
FORBIDDEN_PHASE05_NAMES = frozenset({"complete_cells.json", "seed0_shot.json", "selection_outcomes.json"})
C_GRID = (0.01, 0.1, 1.0, 10.0)


class Phase4D2ProtocolBError(ValueError):
    pass


@dataclass(frozen=True)
class Phase4D2ProtocolBConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    synthetic_fixture: bool
    model_seeds: tuple[int, ...]
    class_count: int
    records_per_class: int
    shot_counts: tuple[int, ...]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    condition_ids: tuple[str, ...]
    artifact_payload_files: tuple[str, ...]
    metric_output_ids: tuple[str, ...]
    expected_bridge_row_count: int
    expected_metric_row_count: int
    expected_cwt_row_count: int
    expected_model_cell_count: int
    expected_validation_row_count: int
    expected_prediction_row_count: int
    expected_seed_class_condition_count: int
    expected_class_observation_count: int
    expected_alignment_result_count: int
    expected_bootstrap_result_count: int
    expected_sign_flip_result_count: int
    expected_holm_family_count: int
    expected_condition_summary_row_count: int
    expected_secondary_table_row_count: int
    condition_bridge_sha256: str
    claim_boundary: str
    document: Mapping[str, object]


@dataclass(frozen=True)
class D2ProtocolBSyntheticInputs:
    class_count: int
    records_per_class: int
    model_seeds: tuple[int, ...]
    record_ids: tuple[str, ...]
    test_labels: np.ndarray
    train_values: Mapping[tuple[int, int, str], np.ndarray]
    train_labels: Mapping[tuple[int, int], np.ndarray]
    validation_values: Mapping[tuple[int, int, str], np.ndarray]
    validation_labels: Mapping[tuple[int, int], np.ndarray]
    test_values: Mapping[str, np.ndarray]
    parent_step15_payloads: Mapping[str, object]
    parent_step17_payloads: Mapping[str, object]
    source_records: tuple[Mapping[str, object], ...] = ()
    model_cells: tuple[Mapping[str, object], ...] = ()
    source_spectra: tuple[object, ...] = ()
    model_role_occurrences: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class D2ProtocolBAuthorityBridge:
    document: Mapping[str, object]
    bridge_rows: tuple[Mapping[str, object], ...]
    metric_rows: tuple[Mapping[str, object], ...]
    cwt_rows: tuple[Mapping[str, object], ...]
    condition_bridge_sha256: str


@dataclass(frozen=True)
class Phase4D2ProtocolBSummary:
    path: Path
    run_id: str
    status: str
    prediction_row_count: int
    class_observation_count: int
    predictions: tuple[Mapping[str, object], ...]


def _canonical_sha(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _environment_authority() -> dict[str, str]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _code_authority() -> dict[str, dict[str, object]]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": sha256_hex((ROOT / relative).read_bytes()),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _config_authority() -> dict[str, object]:
    raw = (ROOT / CONFIG_AUTHORITY_RELATIVE_PATH).read_bytes()
    return {"bytes": len(raw), "sha256": sha256_hex(raw)}


def _parent_artifact_receipts(
    config: "Phase4D2ProtocolBConfig", bridge_document: Mapping[str, object]
) -> dict[str, object]:
    parents = config.document.get("parent_artifacts")
    if isinstance(parents, Mapping):
        return {
            name: {
                "run_id": str(receipt["run_id"]),
                "payload_run_id": str(receipt["payload_run_id"]),
                "sha256sums_sha256": str(receipt["sha256sums_sha256"]),
            }
            for name, receipt in parents.items()
        }
    return {
        "protocol_a": {"path": str(bridge_document.get("protocol_a_path", ""))},
        "eligibility": {"path": str(bridge_document.get("eligibility_path", ""))},
    }


def _shot_gate_states(config: "Phase4D2ProtocolBConfig") -> dict[str, str]:
    if config.synthetic_fixture:
        return {str(shot): "evaluable" for shot in config.shot_counts}
    gate_path = (
        ROOT
        / str(config.document["parent_artifacts"]["eligibility"]["relative_path"])
        / "gate.json"
    )
    gate = json.loads(gate_path.read_bytes())
    states = {
        str(shot): str(gate.get("shots", {}).get(str(shot), {}).get("state", ""))
        for shot in config.shot_counts
    }
    if any(state != "evaluable" for state in states.values()):
        raise Phase4D2ProtocolBError("parent shot gate is not evaluable")
    return states


def _condition_rows(record_ids: Sequence[str], labels: np.ndarray, config: Phase4D2ProtocolBConfig) -> tuple[dict[str, object], ...]:
    rows = []
    for record_order, record_id in enumerate(record_ids):
        class_label = int(labels[record_order])
        for condition_id in config.condition_ids:
            rows.append(
                {
                    "record_order": record_order,
                    "record_id": record_id,
                    "class_label": class_label,
                    "condition_id": condition_id,
                    "state": "complete",
                    "axis_sha256": _canonical_sha(["axis", record_id, condition_id]),
                    "intensity_sha256": _canonical_sha(["intensity", record_id, condition_id]),
                    "support_projection_sha256": _canonical_sha(["projection", record_id, condition_id]),
                }
            )
    return tuple(rows)


def _metric_rows(record_ids: Sequence[str], labels: np.ndarray, config: Phase4D2ProtocolBConfig) -> tuple[dict[str, object], ...]:
    rows = []
    for record_order, _record_id in enumerate(record_ids):
        label = int(labels[record_order])
        for condition_index, condition_id in enumerate(config.condition_ids):
            for metric_index, metric_output_id in enumerate(config.metric_output_ids):
                rows.append(
                    {
                        "record_order": record_order,
                        "record_id": record_ids[record_order],
                        "condition_id": condition_id,
                        "metric_output_id": metric_output_id,
                        "state": "complete",
                        "value": float(label + condition_index * 0.01 + metric_index * 0.001),
                    }
                )
    return tuple(rows)


def _cwt_rows(record_ids: Sequence[str], labels: np.ndarray, config: Phase4D2ProtocolBConfig) -> tuple[dict[str, object], ...]:
    rows = []
    for record_order, record_id in enumerate(record_ids):
        for condition_id in config.condition_ids:
            rows.append(
                {
                    "record_order": record_order,
                    "record_id": record_id,
                    "condition_id": condition_id,
                    "state": "complete",
                    "cwt_receipt_sha256": _canonical_sha(["cwt", record_id, condition_id, int(labels[record_order])]),
                }
            )
    return tuple(rows)


def parse_phase4_d2_protocol_b_config(
    path: Path, raw: bytes, *, require_frozen_identity: bool
) -> Phase4D2ProtocolBConfig:
    try:
        document = json.loads(raw)
    except Exception as error:
        raise Phase4D2ProtocolBError(f"config parse failed: {error}") from error
    if canonical_json_bytes(document) != raw:
        raise Phase4D2ProtocolBError("config must be canonical json")
    if require_frozen_identity:
        if len(raw) != CONFIG_BYTES or sha256_hex(raw) != CONFIG_SHA256:
            raise Phase4D2ProtocolBError("frozen config identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4D2ProtocolBError("schema version mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D2ProtocolBError("experiment id mismatch")
    if document.get("protocol") != "B" or document.get("tier") != "full_domain_core":
        raise Phase4D2ProtocolBError("protocol/tier mismatch")
    if document.get("claim_boundary") != CLAIM_BOUNDARY:
        raise Phase4D2ProtocolBError("claim boundary mismatch")
    if require_frozen_identity and bool(document.get("synthetic_fixture", False)):
        raise Phase4D2ProtocolBError("public config rejects synthetic fixtures")
    perturbation_ids = tuple(document["active_perturbation_ids"])
    alpha_grid = tuple(float(value) for value in document["alpha_grid"])
    if perturbation_ids != PERTURBATIONS or alpha_grid != ALPHAS:
        raise Phase4D2ProtocolBError("frozen perturbation/alpha grid mismatch")
    if tuple(document["artifact_payload_files"]) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D2ProtocolBError("artifact payload order mismatch")
    if tuple(document["metric_output_ids"]) != METRIC_OUTPUT_IDS:
        raise Phase4D2ProtocolBError("metric manifest mismatch")
    if not bool(document.get("synthetic_fixture",False)):
        for section in ("authorities","parent_artifacts","environment_authority","model_recipe","inference","figure_contract","artifact_contract","inherited_rulings","code_authority","frozen_identities","support_grid","p10","alpha0_equivalence"):
            if not isinstance(document.get(section), Mapping): raise Phase4D2ProtocolBError(f"config missing/invalid {section}")
        for name, receipt in document["authorities"].items():
            if set(receipt) != {"path","bytes","sha256"}: raise Phase4D2ProtocolBError(f"authority schema mismatch: {name}")
            live=ROOT/str(receipt["path"]); expected_size=int(receipt["bytes"]); expected_sha=str(receipt["sha256"])
            if not live.is_file() or live.stat().st_size!=expected_size or sha256_hex(live.read_bytes())!=expected_sha: raise Phase4D2ProtocolBError(f"live authority mismatch: {name}")
        if document["environment_authority"] != _environment_authority(): raise Phase4D2ProtocolBError("environment authority mismatch")
        if (
            set(document["code_authority"]) != set(CODE_RELATIVE_PATHS)
            or document["code_authority"] != _code_authority()
        ):
            raise Phase4D2ProtocolBError("code_authority mismatch")
        trust_anchor = document.get("trust_anchor")
        if (
            not isinstance(trust_anchor, Mapping)
            or trust_anchor.get("config_authority_relative_path")
            != CONFIG_AUTHORITY_RELATIVE_PATH
        ):
            raise Phase4D2ProtocolBError("trust_anchor mismatch")
        if set(document["parent_artifacts"]) != {"protocol_a","eligibility"}: raise Phase4D2ProtocolBError("parent artifact set mismatch")
        for name,parent in document["parent_artifacts"].items():
            if not isinstance(parent,Mapping) or not isinstance(parent.get("payload_sha256"),Mapping) or not parent.get("sha256sums_sha256") or not parent.get("run_id") or not parent.get("relative_path"): raise Phase4D2ProtocolBError(f"parent artifact contract incomplete: {name}")
        recipe=document["model_recipe"]; inference=document["inference"]
        if recipe != {"projection":"float64_interpolation_then_one_float32_cast_no_normalization","pca":{"n_components":20,"random_state":"model_seed","svd_solver":"randomized","whiten":False},"logistic_regression":{"c_grid":[0.01,0.1,1.0,10.0],"class_weight":None,"max_iter":1000,"random_state":"model_seed","regularization":"l2","solver":"lbfgs","tol":0.0001},"selection":"first_strict_maximum_validation_top1_accuracy_lowest_c_on_tie","refit_with_validation":False,"condition_specific_roles":True,"test_condition_count":41}: raise Phase4D2ProtocolBError("model recipe mismatch")
        if inference != {"bootstrap_resamples":2000,"sign_flip_resamples":100000,"random_seed":20260817,"confidence_level":0.95,"holm_alpha":0.05,"cluster_unit":"class_label","d_ag_formula":"AG_MSE-AG_candidate","d_acc_formula":"Acc_candidate-Acc_MSE","class_contribution_reduction":{"D_AG":"sum","D_Acc":"mean"},"holm_slot_count_per_shot":24,"shot_families_are_separate":True}: raise Phase4D2ProtocolBError("inference contract mismatch")
        artifact=document["artifact_contract"]
        if artifact != {"payload_files":list(ARTIFACT_PAYLOAD_FILES),"terminal_markers":list(TERMINAL_MARKERS),"terminal_marker_rule":"exactly_one","checksum_file":"SHA256SUMS","checksum_scope":"all_payloads_and_exactly_one_terminal_marker"}: raise Phase4D2ProtocolBError("artifact contract mismatch")
    shot_counts = tuple(document["shot_counts"])
    model_seeds = tuple(document["model_seeds"])
    denominators = document["denominators"]
    expected = document["expected"]
    bridge = document["authority_bridge"]
    required = {
        "bridge_row_count": int(denominators["test_record_count"]) * 41,
        "metric_row_count": int(denominators["test_record_count"]) * 41 * 13,
        "cwt_row_count": int(denominators["test_record_count"]) * 41,
        "model_cell_count": len(shot_counts) * len(model_seeds) * 41,
        "validation_row_count": len(shot_counts) * len(model_seeds) * 41 * 4,
        "prediction_row_count": len(shot_counts) * len(model_seeds) * int(denominators["test_record_count"]) * 41,
        "seed_class_condition_count": len(shot_counts) * len(model_seeds) * int(denominators["class_count"]) * 41,
    }
    if not bool(document.get("synthetic_fixture", False)):
        required.update({
            "operator_cell_count": 27565,
            "apply_check_count": 248085,
            "rematerialized_source_condition_count": 226033,
            "configured_payload_count": len(ARTIFACT_PAYLOAD_FILES),
            "artifact_file_count": len(ARTIFACT_PAYLOAD_FILES) + 2,
        })
    if any(int(expected.get(key, -1)) != value for key, value in required.items()):
        raise Phase4D2ProtocolBError("frozen denominator mismatch")
    return Phase4D2ProtocolBConfig(
        path=Path(path),
        raw_bytes=raw,
        sha256=sha256_hex(raw),
        synthetic_fixture=bool(document.get("synthetic_fixture", False)),
        model_seeds=model_seeds,
        class_count=int(denominators["class_count"]),
        records_per_class=int(denominators["records_per_class"]),
        shot_counts=shot_counts,
        perturbation_ids=perturbation_ids,
        alpha_grid=alpha_grid,
        condition_ids=condition_ids(perturbation_ids, alpha_grid),
        artifact_payload_files=tuple(document["artifact_payload_files"]),
        metric_output_ids=tuple(document["metric_output_ids"]),
        expected_bridge_row_count=int(expected["bridge_row_count"]),
        expected_metric_row_count=int(expected["metric_row_count"]),
        expected_cwt_row_count=int(expected["cwt_row_count"]),
        expected_model_cell_count=int(expected["model_cell_count"]),
        expected_validation_row_count=int(expected["validation_row_count"]),
        expected_prediction_row_count=int(expected["prediction_row_count"]),
        expected_seed_class_condition_count=int(expected["seed_class_condition_count"]),
        expected_class_observation_count=int(expected["class_observation_count"]),
        expected_alignment_result_count=int(expected["alignment_result_count"]),
        expected_bootstrap_result_count=int(expected["bootstrap_result_count"]),
        expected_sign_flip_result_count=int(expected["sign_flip_result_count"]),
        expected_holm_family_count=int(expected["holm_family_count"]),
        expected_condition_summary_row_count=int(expected["condition_summary_row_count"]),
        expected_secondary_table_row_count=int(expected["secondary_table_row_count"]),
        condition_bridge_sha256=str(bridge["condition_bridge_sha256"]),
        claim_boundary=str(document["claim_boundary"]),
        document=MappingProxyType(document),
    )


def load_phase4_d2_protocol_b_config(path: Path) -> Phase4D2ProtocolBConfig:
    raw = Path(path).read_bytes()
    return parse_phase4_d2_protocol_b_config(path, raw, require_frozen_identity=True)


def make_synthetic_d2_protocol_b_inputs(
    *,
    class_count: int,
    records_per_class: int,
    model_seeds: Sequence[int],
    include_forbidden_parent_payload: bool = False,
) -> D2ProtocolBSyntheticInputs:
    if include_forbidden_parent_payload:
        raise Phase4D2ProtocolBError("forbidden Step-15 payload requested in synthetic input")
    if class_count < 2 or records_per_class < 2:
        raise Phase4D2ProtocolBError("synthetic fixture requires at least 2 classes and 2 records")
    total = class_count * records_per_class
    record_ids = tuple(f"test-{index:06d}" for index in range(total))
    labels = np.repeat(np.arange(class_count, dtype=np.int64), records_per_class)
    conditions = condition_ids()
    feature_dim = 20
    train_values: dict[tuple[int, int, str], np.ndarray] = {}
    validation_values: dict[tuple[int, int, str], np.ndarray] = {}
    test_values: dict[str, np.ndarray] = {}
    train_labels: dict[tuple[int, int], np.ndarray] = {}
    validation_labels: dict[tuple[int, int], np.ndarray] = {}
    for shot in SHOTS:
        for seed in model_seeds:
            train = []
            train_y = []
            validation = []
            validation_y = []
            for label in range(class_count):
                base = float(label) * 100.0 + float(seed) * 10.0 + float(shot)
                for item in range(shot):
                    train.append(np.linspace(base + item, base + item + 1.0, feature_dim))
                    train_y.append(label)
                validation.append(np.linspace(base + 50.0, base + 51.0, feature_dim))
                validation_y.append(label)
            train_labels[(shot, seed)] = np.asarray(train_y, dtype=np.int64)
            validation_labels[(shot, seed)] = np.asarray(validation_y, dtype=np.int64)
            base_train = np.asarray(train, dtype=np.float64)
            base_validation = np.asarray(validation, dtype=np.float64)
            for condition_index, condition_id in enumerate(conditions):
                train_values[(shot, seed, condition_id)] = base_train + condition_index * 0.0001
                validation_values[(shot, seed, condition_id)] = base_validation + condition_index * 0.0001
    for condition_index, condition_id in enumerate(conditions):
        rows = []
        for order, label in enumerate(labels):
            base = float(label) * 100.0 + order
            rows.append(np.linspace(base, base + 1.0, feature_dim))
        test_values[condition_id] = np.asarray(rows, dtype=np.float64) + condition_index * 0.0001
    return D2ProtocolBSyntheticInputs(
        class_count=class_count,
        records_per_class=records_per_class,
        model_seeds=tuple(int(seed) for seed in model_seeds),
        record_ids=record_ids,
        test_labels=labels,
        train_values=MappingProxyType(train_values),
        train_labels=MappingProxyType(train_labels),
        validation_values=MappingProxyType(validation_values),
        validation_labels=MappingProxyType(validation_labels),
        test_values=MappingProxyType(test_values),
        parent_step15_payloads=MappingProxyType({}),
        parent_step17_payloads=MappingProxyType({}),
    )


def make_synthetic_d2_protocol_b_config(
    inputs: D2ProtocolBSyntheticInputs,
) -> Phase4D2ProtocolBConfig:
    condition_count = len(condition_ids())
    class_obs = len(SHOTS) * len(METRIC_OUTPUT_IDS) * inputs.class_count * (condition_count - 1)
    expected = {
        "bridge_row_count": len(inputs.record_ids) * condition_count,
        "metric_row_count": len(inputs.record_ids) * condition_count * len(METRIC_OUTPUT_IDS),
        "cwt_row_count": len(inputs.record_ids) * condition_count,
        "model_cell_count": len(SHOTS) * len(inputs.model_seeds) * condition_count,
        "validation_row_count": len(SHOTS) * len(inputs.model_seeds) * condition_count * len(C_GRID),
        "prediction_row_count": len(SHOTS) * len(inputs.model_seeds) * condition_count * len(inputs.record_ids),
        "seed_class_condition_count": len(SHOTS) * len(inputs.model_seeds) * inputs.class_count * condition_count,
        "class_observation_count": class_obs,
        "alignment_result_count": len(SHOTS) * len(METRIC_OUTPUT_IDS),
        "bootstrap_result_count": len(SHOTS) * (len(METRIC_OUTPUT_IDS) - 1),
        "sign_flip_result_count": len(SHOTS) * (len(METRIC_OUTPUT_IDS) - 1) * 2,
        "holm_family_count": len(SHOTS) * (len(METRIC_OUTPUT_IDS) - 1) * 2,
        "condition_summary_row_count": len(SHOTS) * condition_count,
        "secondary_table_row_count": len(SHOTS) * len(METRIC_OUTPUT_IDS),
    }
    bridge_sha = "0" * 64
    document = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol": "B",
        "tier": "full_domain_core",
        "synthetic_fixture": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS),
        "shot_counts": list(SHOTS),
        "model_seeds": list(inputs.model_seeds),
        "metric_output_ids": list(METRIC_OUTPUT_IDS),
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "denominators": {
            "class_count": inputs.class_count,
            "records_per_class": inputs.records_per_class,
            "test_record_count": len(inputs.record_ids),
        },
        "expected": expected,
        "authority_bridge": {
            "condition_bridge_sha256": bridge_sha,
            "step18_real_condition_bridge_sha256": REAL_CONDITION_BRIDGE_SHA256,
        },
    }
    preliminary = parse_phase4_d2_protocol_b_config(Path("<synthetic>"), canonical_json_bytes(document), require_frozen_identity=False)
    document["authority_bridge"]["condition_bridge_sha256"] = sha256_hex(jsonl_bytes(_condition_rows(inputs.record_ids, inputs.test_labels, preliminary)))
    return parse_phase4_d2_protocol_b_config(Path("<synthetic>"), canonical_json_bytes(document), require_frozen_identity=False)


def reconstruct_d2_protocol_b_outcome_inputs(
    dataset_path: Path, selection_path: Path, config: Phase4D2ProtocolBConfig
) -> D2ProtocolBSyntheticInputs:
    if dataset_path.name in FORBIDDEN_PHASE05_NAMES or selection_path.name in FORBIDDEN_PHASE05_NAMES:
        raise Phase4D2ProtocolBError("forbidden Phase-0.5 outcome input")
    if config.synthetic_fixture:
        return make_synthetic_d2_protocol_b_inputs(class_count=config.class_count, records_per_class=config.records_per_class, model_seeds=config.model_seeds)
    try:
        selection = json.loads(Path(selection_path).read_bytes())
        validate_d2_few_shot_selection(
            selection, ROOT / "experiments/phase05/configs/d2_few_shot_selection.json", Path(dataset_path)
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, D2SelectionValidationError) as error:
        raise Phase4D2ProtocolBError(f"selection replay failed: {error}") from error
    selected_labels: dict[str, int] = {}
    memberships: dict[str, set[int]] = {}
    cell_specs: list[dict[str, object]] = []
    model_role_lookup: dict[tuple[int, int, str], tuple[str, ...]] = {}
    for seed_doc in selection.get("selections", ()):
        seed = int(seed_doc["seed"]); validation_by_class = {}; train_by_class = {}
        for class_doc in seed_doc["classes"]:
            label = int(class_doc["class_label"]); validation = tuple(map(str, class_doc["validation_record_ids"]))
            trains = {int(shot): tuple(map(str, ids)) for shot, ids in class_doc["train_record_ids"].items()}
            validation_by_class[label] = validation; train_by_class[label] = trains
            for record_id in validation:
                selected_labels.setdefault(record_id, label); memberships.setdefault(record_id, set()).update(config.shot_counts)
            for shot, ids in trains.items():
                for record_id in ids:
                    selected_labels.setdefault(record_id, label); memberships.setdefault(record_id, set()).add(shot)
        for shot in config.shot_counts:
            model_train_ids = tuple(
                record_id
                for values in train_by_class.values()
                for record_id in values[shot]
            )
            model_validation_ids = tuple(
                record_id
                for values in validation_by_class.values()
                for record_id in values
            )
            model_role_lookup[(shot, seed, "train")] = model_train_ids
            model_role_lookup[(shot, seed, "validation")] = model_validation_ids
            train_ids = tuple(sorted(model_train_ids))
            validation_ids = tuple(sorted(model_validation_ids))
            if set(train_ids) & set(validation_ids):
                raise Phase4D2ProtocolBError("selection train/validation overlap")
            cell_specs.append({"seed": seed, "shot_count": shot, "train_record_ids": train_ids, "validation_record_ids": validation_ids})
    support_doc = json.loads((ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json").read_bytes())
    support = np.asarray(support_doc["support_grid"]["coordinates_cm1"], dtype="<f8")
    max_gap = float(support_doc["support_grid"]["max_in_range_native_gap_cm1"])
    source: dict[str, dict[str, object]] = {}; spectra: dict[str, Spectrum1D] = {}; test_ids: list[str] = []
    with BacteriaIdBatchLoader(Path(dataset_path), batch_size=4096) as loader:
        for batch in loader.iter_batches():
            if batch.source_split not in {"finetune", "test"}: continue
            axis = np.ascontiguousarray(np.asarray(batch.wavenumber, dtype="<f4")[::-1], dtype="<f8")
            if not np.all(np.diff(axis) > 0): raise Phase4D2ProtocolBError("native axis reversal failed")
            for record_id, label, stored, source_row in zip(batch.record_ids, batch.class_labels, batch.intensity, batch.source_rows, strict=True):
                record_id = str(record_id)
                if batch.source_split == "finetune" and record_id not in selected_labels: continue
                intensity = np.ascontiguousarray(np.asarray(stored, dtype="<f4")[::-1], dtype="<f8")
                spectrum = Spectrum1D(f"bacteria_id_reference::{record_id}", None, axis, intensity)
                projection = _project_support(spectrum, support, max_gap)
                if batch.source_split == "test":
                    test_ids.append(record_id); memberships.setdefault(record_id, set()).update(config.shot_counts)
                spectra[record_id] = spectrum
                source[record_id] = {"class_label": int(label), "native_axis_sha256": _array_sha(axis), "native_intensity_sha256": _array_sha(intensity), "record_id": record_id, "scope": str(batch.source_split), "shot_memberships": sorted(memberships[record_id]), "source_row": int(source_row), "support_projection_sha256": _array_sha(projection, "<f4")}
    test_ids = sorted(test_ids)
    if len(test_ids) != 3000 or _ids_sha(test_ids) != str(selection["test"]["record_ids_sha256"]):
        raise Phase4D2ProtocolBError("exact retained test ID mismatch")
    cells = []; roles = []
    for cell in cell_specs:
        row = {**cell, "test_record_ids": tuple(test_ids)}; cells.append(row)
        for key in ("train_record_ids", "validation_record_ids", "test_record_ids"):
            role = key.split("_", 1)[0]
            roles.extend({"record_id": rid, "role": role, "seed": int(row["seed"]), "shot_count": int(row["shot_count"])} for rid in row[key])
    counts = Counter(str(row["record_id"]) for row in roles); ordered_ids = sorted(source)
    source_rows = tuple({**source[rid], "record_order": order, "role_count": counts[rid]} for order, rid in enumerate(ordered_ids))
    order = {rid: index for index, rid in enumerate(ordered_ids)}
    roles.sort(key=lambda row: (int(row["seed"]), int(row["shot_count"]), {"train":0,"validation":1,"test":2}[str(row["role"])], order[str(row["record_id"])]))
    if (len(source_rows), len(cells), len(roles)) != (5513, 15, 54750):
        raise Phase4D2ProtocolBError("reconstructed source/model/role denominator mismatch")
    role_lookup = dict(model_role_lookup)
    for cell in cells:
        role_lookup[(int(cell["shot_count"]), int(cell["seed"]), "test")] = tuple(
            cell["test_record_ids"]
        )
    labels = np.asarray([source[rid]["class_label"] for rid in test_ids], dtype="<i8")
    return D2ProtocolBSyntheticInputs(config.class_count, config.records_per_class, config.model_seeds, tuple(test_ids), labels, MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), MappingProxyType({}), MappingProxyType({"role_lookup": MappingProxyType(role_lookup), "sources": MappingProxyType(source)}), source_rows, tuple(sorted(cells, key=lambda x:(int(x["shot_count"]),int(x["seed"])))), tuple(spectra[rid] for rid in ordered_ids), tuple(roles))


def _array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return sha256_hex(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


def _ids_sha(values: Sequence[str]) -> str:
    return sha256_hex(("\n".join(values) + "\n").encode())


def _project_support(spectrum: Spectrum1D, support: np.ndarray, max_gap: float) -> np.ndarray:
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8"); intensity = np.asarray(spectrum.intensity, dtype="<f8")
    left = int(np.searchsorted(axis, support[0], side="left")); right = int(np.searchsorted(axis, support[-1], side="right"))
    if axis[0] > support[0] or axis[-1] < support[-1] or right-left < 2 or float(np.max(np.diff(axis[left:right]))) > max_gap:
        raise Phase4D2ProtocolBError("support projection gate failed")
    output = np.ascontiguousarray(np.interp(support, axis, intensity), dtype="<f4")
    if output.size != 997 or not np.isfinite(output).all() or np.linalg.norm(output.astype(float)) <= 0:
        raise Phase4D2ProtocolBError("support projection invalid")
    return output


def build_d2_protocol_b_authority_bridge(
    *,
    inputs: D2ProtocolBSyntheticInputs,
    protocol_a_path: Path,
    eligibility_path: Path,
    config: Phase4D2ProtocolBConfig,
) -> D2ProtocolBAuthorityBridge:
    if protocol_a_path.name in FORBIDDEN_STEP15_NAMES:
        raise Phase4D2ProtocolBError("forbidden Step-15 payload path")
    if not config.synthetic_fixture:
        parents = config.document["parent_artifacts"]
        _validate_parent_artifact(protocol_a_path, parents["protocol_a"], expected_files=38, expected_checksums=37)
        _validate_parent_artifact(eligibility_path, parents["eligibility"], expected_files=11, expected_checksums=10)
        def read_rows(path: Path):
            rows = []
            with path.open(encoding="utf-8") as stream:
                for number, line in enumerate(stream, 1):
                    try: row = json.loads(line)
                    except json.JSONDecodeError as error: raise Phase4D2ProtocolBError(f"{path.name}:{number}: malformed JSON") from error
                    if canonical_json_bytes(row) != line.encode(): raise Phase4D2ProtocolBError(f"{path.name}:{number}: noncanonical row")
                    rows.append(row)
            return tuple(rows)
        a_conditions, b_conditions = read_rows(protocol_a_path / "record_conditions.jsonl"), read_rows(eligibility_path / "record_conditions.jsonl")
        if len(a_conditions) != config.expected_bridge_row_count or len(b_conditions) != 226033:
            raise Phase4D2ProtocolBError("dual-parent condition row count mismatch")
        a_map = _unique_rows(a_conditions, ("record_id", "condition_id"), "Step-15 conditions")
        b_map = _unique_rows(b_conditions, ("record_id", "condition_id"), "Step-17 conditions")
        sources = _unique_rows(read_rows(eligibility_path / "source_records.jsonl"), ("record_id",), "Step-17 sources")
        if len(sources) != 5513: raise Phase4D2ProtocolBError("Step-17 source count mismatch")
        parent_models=read_rows(eligibility_path/"model_cells.jsonl"); parent_roles=read_rows(eligibility_path/"model_role_occurrences.jsonl")
        if jsonl_bytes(inputs.source_records)!=jsonl_bytes(tuple(sources.values())) or jsonl_bytes(tuple(sorted(inputs.model_cells,key=lambda r:(int(r["seed"]),int(r["shot_count"])))))!=jsonl_bytes(parent_models) or jsonl_bytes(inputs.model_role_occurrences)!=jsonl_bytes(parent_roles):
            raise Phase4D2ProtocolBError("independent selection/source/role reconstruction differs from Step-17")
        ordered = []
        for order, record_id in enumerate(inputs.record_ids):
            source = sources.get((record_id,))
            if source is None or source.get("scope") != "test" or int(source.get("role_count", -1)) != 15 or source.get("shot_memberships") != [5,10,20]:
                raise Phase4D2ProtocolBError("Step-17 test source role mismatch")
            for condition_id in config.condition_ids:
                key = (record_id, condition_id)
                if key not in a_map or key not in b_map:
                    raise Phase4D2ProtocolBError("dual-parent bridge missing key")
                a, b = a_map[key], b_map[key]
                if int(a.get("record_order", -1)) != order or a.get("state") != "complete" or b.get("state") != "complete":
                    raise Phase4D2ProtocolBError("dual-parent bridge state/order mismatch")
                if (int(source["class_label"]) != int(b.get("class_label", -1)) or a.get("axis_sha256") != b.get("axis_sha256") or a.get("intensity_sha256") != b.get("intensity_sha256") or a.get("projected_row_sha256") != b.get("support_projection_sha256")):
                    raise Phase4D2ProtocolBError("dual-parent bridge field mismatch")
                ordered.append({"test_order": order, "record_id": record_id, "class_label": int(source["class_label"]), "condition_id": condition_id, "state": "complete", "axis_sha256": a["axis_sha256"], "intensity_sha256": a["intensity_sha256"], "support_projection_sha256": a["projected_row_sha256"]})
        expected_keys = {(rid, cid) for rid in inputs.record_ids for cid in config.condition_ids}
        if set(a_map) != expected_keys or not expected_keys <= set(b_map): raise Phase4D2ProtocolBError("dual-parent bridge extras/missing keys")
        digest = sha256_hex(jsonl_bytes(ordered))
        if digest != config.condition_bridge_sha256 or digest != REAL_CONDITION_BRIDGE_SHA256:
            raise Phase4D2ProtocolBError("exact dual-parent condition bridge digest mismatch")
        metrics, cwts = read_rows(protocol_a_path / "metric_values.jsonl"), read_rows(protocol_a_path / "peak_receipts.jsonl")
        metric_keys = _unique_rows(metrics, ("record_id","condition_id","metric_output_id"), "Step-15 metrics")
        cwt_keys = _unique_rows(cwts, ("record_id","condition_id"), "Step-15 CWT")
        expected_metric_keys = {(rid,cid,metric) for rid in inputs.record_ids for cid in config.condition_ids for metric in config.metric_output_ids}
        if len(metrics) != config.expected_metric_row_count or set(metric_keys) != expected_metric_keys or len(cwts) != config.expected_cwt_row_count or set(cwt_keys) != expected_keys or any(r.get("state") != "complete" or not math.isfinite(float(r["value"])) for r in metrics) or any(r.get("state") != "complete" for r in cwts):
            raise Phase4D2ProtocolBError("Step-15 measurement authority malformed")
        document = MappingProxyType({"protocol_a_run_id": protocol_a_path.name, "protocol_a_sha256sums_sha256": sha256_hex((protocol_a_path/"SHA256SUMS").read_bytes()), "eligibility_run_id": eligibility_path.name, "eligibility_sha256sums_sha256": sha256_hex((eligibility_path/"SHA256SUMS").read_bytes()), "bridge_row_count": len(ordered), "metric_row_count": len(metrics), "cwt_row_count": len(cwts), "condition_bridge_sha256": digest, "missing_key_count":0,"extra_key_count":0,"duplicate_key_count":0,"field_mismatch_count":0})
        return D2ProtocolBAuthorityBridge(document, tuple(ordered), tuple(metrics), tuple(cwts), digest)
    bridge_rows = _condition_rows(inputs.record_ids, inputs.test_labels, config)
    metric_rows = _metric_rows(inputs.record_ids, inputs.test_labels, config)
    cwt_rows = _cwt_rows(inputs.record_ids, inputs.test_labels, config)
    digest = sha256_hex(jsonl_bytes(bridge_rows))
    if digest != config.condition_bridge_sha256:
        raise Phase4D2ProtocolBError("exact dual-parent condition bridge digest mismatch")
    document = MappingProxyType(
        {
            "protocol_a_path": str(protocol_a_path),
            "eligibility_path": str(eligibility_path),
            "bridge_row_count": len(bridge_rows),
            "metric_row_count": len(metric_rows),
            "cwt_row_count": len(cwt_rows),
            "condition_bridge_sha256": digest,
        }
    )
    return D2ProtocolBAuthorityBridge(
        document=document,
        bridge_rows=bridge_rows,
        metric_rows=metric_rows,
        cwt_rows=cwt_rows,
        condition_bridge_sha256=digest,
    )


def _unique_rows(rows: Sequence[Mapping[str, object]], fields: Sequence[str], boundary: str) -> dict[tuple[object, ...], Mapping[str, object]]:
    result = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if None in key or key in result: raise Phase4D2ProtocolBError(f"{boundary} duplicate/malformed key")
        result[key] = row
    return result


def _validate_parent_artifact(path: Path, authority: Mapping[str, object], *, expected_files: int, expected_checksums: int) -> None:
    if not path.is_dir() or path.name != authority.get("run_id"): raise Phase4D2ProtocolBError("parent run identity mismatch")
    names = {item.name for item in path.iterdir() if item.is_file()}
    if len(names) != expected_files or names != set(authority["payload_sha256"]) | {"SHA256SUMS"}: raise Phase4D2ProtocolBError("parent exact file inventory mismatch")
    sums = path / "SHA256SUMS"
    if sha256_hex(sums.read_bytes()) != authority.get("sha256sums_sha256"): raise Phase4D2ProtocolBError("parent SHA256SUMS identity mismatch")
    entries = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ",1)
        if len(parts) != 2 or parts[1] in entries: raise Phase4D2ProtocolBError("parent checksum schema mismatch")
        entries[parts[1]] = parts[0]
    if len(entries) != expected_checksums or entries != authority["payload_sha256"]: raise Phase4D2ProtocolBError("parent checksum inventory mismatch")
    for name, digest in entries.items():
        if sha256_hex((path/name).read_bytes()) != digest: raise Phase4D2ProtocolBError(f"parent payload checksum mismatch: {name}")
    marker = json.loads((path/"complete.json").read_bytes()); manifest = json.loads((path/"manifest.json").read_bytes())
    expected_payload_run_id=authority.get("payload_run_id",authority.get("run_id")); allowed_status=set(authority.get("completion_statuses",["complete"]))
    manifest_status=manifest.get("status",manifest.get("overall_status",marker.get("status")))
    if marker.get("status") not in allowed_status or manifest_status not in allowed_status or marker.get("run_id")!=expected_payload_run_id or manifest.get("run_id") != expected_payload_run_id:
        raise Phase4D2ProtocolBError("parent completion identity mismatch")


_D2B_WORKER_SWEEP: object | None = None
_D2B_WORKER_PHASE1: object | None = None
_D2B_WORKER_SUPPORT: np.ndarray | None = None
_D2B_WORKER_MAX_GAP: float | None = None
_D2B_WORKER_P10_BUDGET: int | None = None


def _initialize_d2_protocol_b_worker(
    sweep_path: str, phase1_path: str, support: tuple[float, ...], max_gap: float, p10_budget: int
) -> None:
    global _D2B_WORKER_SWEEP, _D2B_WORKER_PHASE1, _D2B_WORKER_SUPPORT, _D2B_WORKER_MAX_GAP, _D2B_WORKER_P10_BUDGET
    _D2B_WORKER_SWEEP = load_perturbation_sweep_config(Path(sweep_path))
    _D2B_WORKER_PHASE1 = load_phase1_core_config(Path(phase1_path))
    _D2B_WORKER_SUPPORT = np.asarray(support, dtype="<f8")
    _D2B_WORKER_MAX_GAP = float(max_gap)
    _D2B_WORKER_P10_BUDGET = int(p10_budget)


def _d2_protocol_b_source_job(
    job: tuple[int, dict[str, object], Spectrum1D],
) -> tuple[int, str, dict[str, tuple[np.ndarray, str, str]]]:
    if any(value is None for value in (_D2B_WORKER_SWEEP, _D2B_WORKER_PHASE1, _D2B_WORKER_SUPPORT, _D2B_WORKER_MAX_GAP, _D2B_WORKER_P10_BUDGET)):
        raise Phase4D2ProtocolBError("rematerialization worker was not initialized")
    order, record, spectrum = job
    record_id = str(record["record_id"])
    source = Phase1Source(
        SelectedSourceRow(
            int(record["record_order"]), record_id, record_id, int(record["class_label"]),
            f"bacteria-{record['class_label']}", f"native::{record['native_axis_sha256']}",
        ),
        spectrum, "increasing", _array_sha(spectrum.axis_cm1, "<f4"),
        _array_sha(spectrum.intensity, "<f4"), str(record["native_axis_sha256"]),
        str(record["native_intensity_sha256"]),
        MappingProxyType({"dataset_id": "bacteria_id_reference", "record_id": record_id}),
    )
    admission = P10MemoryAdmission(int(_D2B_WORKER_P10_BUDGET))
    results: dict[str, tuple[np.ndarray, str, str]] = {
        "alpha0": (
            np.asarray(_project_support(spectrum, _D2B_WORKER_SUPPORT, float(_D2B_WORKER_MAX_GAP)), dtype="<f4"),
            _array_sha(spectrum.axis_cm1), _array_sha(spectrum.intensity),
        )
    }
    with threadpool_limits(limits=1, user_api="blas"):
        for perturbation in PERTURBATIONS:
            cell = run_perturbation_cell(
                source, perturbation, _D2B_WORKER_PHASE1, _D2B_WORKER_SWEEP,
                p10_admission=admission if perturbation == "p10" else None,
            )
            if cell.status.name != "COMPLETE":
                raise Phase4D2ProtocolBError(f"rematerialization failed for {record_id} {perturbation}")
            for item in cell.records[1:]:
                condition_id = f"{perturbation}:{item.alpha_float64_le_hex}"
                results[condition_id] = (
                    np.asarray(_project_support(item.result.output, _D2B_WORKER_SUPPORT, float(_D2B_WORKER_MAX_GAP)), dtype="<f4"),
                    _array_sha(item.result.output.axis_cm1), _array_sha(item.result.output.intensity),
                )
    return order, record_id, results


def rematerialize_d2_protocol_b_science(
    inputs: D2ProtocolBSyntheticInputs, config: Phase4D2ProtocolBConfig, *, worker_count: int
) -> Mapping[str, object]:
    if worker_count < 1:
        raise Phase4D2ProtocolBError("worker_count must be positive")
    if not config.synthetic_fixture:
        if len(inputs.source_records) != 5513 or len(inputs.source_spectra) != 5513:
            raise Phase4D2ProtocolBError("real rematerialization requires all 5513 sources")
        support_doc = json.loads((ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json").read_bytes())
        support = np.asarray(support_doc["support_grid"]["coordinates_cm1"],dtype="<f8"); max_gap=float(support_doc["support_grid"]["max_in_range_native_gap_cm1"]); budget=int(config.document["p10"]["memory_budget_bytes"])
        parent_path = ROOT / str(config.document["parent_artifacts"]["eligibility"]["relative_path"]); expected = {}
        with (parent_path/"record_conditions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row=json.loads(line); expected[(str(row["record_id"]),str(row["condition_id"]))]=row
        matrices: dict[str, dict[str, np.ndarray]] = {condition: {} for condition in config.condition_ids}
        estimates = tuple(estimate_p10_peak_bytes(int(spectrum.axis_cm1.size)) for spectrum in inputs.source_spectra)
        if set(estimates) != {1105805824} or 1105805824 > budget:
            raise Phase4D2ProtocolBError("P10 frozen admission mismatch")
        capacity = budget // 1105805824
        process_count = min(worker_count, len(inputs.source_records), capacity)
        if process_count < 1:
            raise Phase4D2ProtocolBError("P10 worker capacity is zero")
        jobs = tuple(
            (order, dict(record), spectrum)
            for order, (record, spectrum) in enumerate(zip(inputs.source_records, inputs.source_spectra, strict=True))
        )
        initializer_args = (
            str(ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"),
            str(ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"),
            tuple(float(value) for value in support), max_gap, budget,
        )
        receipt_hasher = hashlib.sha256()
        with ProcessPoolExecutor(
            max_workers=process_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_d2_protocol_b_worker, initargs=initializer_args,
        ) as executor:
            # executor.map preserves job order, so results are validated and
            # consumed canonically without retaining a second full array copy.
            for expected_order, (order, record_id, results) in enumerate(
                executor.map(_d2_protocol_b_source_job, jobs)
            ):
                if order != expected_order or tuple(results) != config.condition_ids:
                    raise Phase4D2ProtocolBError("rematerialization canonical collection mismatch")
                for condition_id in config.condition_ids:
                    projected, axis_sha, intensity_sha = results[condition_id]
                    authority = expected.get((record_id, condition_id))
                    projection_sha = _array_sha(projected, "<f4")
                    if (authority is None or authority.get("state") != "complete"
                        or axis_sha != authority.get("axis_sha256")
                        or intensity_sha != authority.get("intensity_sha256")
                        or projection_sha != authority.get("support_projection_sha256")):
                        raise Phase4D2ProtocolBError(f"rematerialization receipt mismatch: {record_id}/{condition_id}")
                    receipt_hasher.update(canonical_json_bytes({
                        "axis_sha256": axis_sha,
                        "condition_id": condition_id,
                        "intensity_sha256": intensity_sha,
                        "record_id": record_id,
                        "support_projection_sha256": projection_sha,
                    }))
                    matrices[condition_id][record_id] = projected
        if any(len(values) != 5513 for values in matrices.values()):
            raise Phase4D2ProtocolBError("rematerialization condition coverage mismatch")
        if len(expected)!=226033: raise Phase4D2ProtocolBError("Step-17 receipt denominator mismatch")
        return MappingProxyType({"worker_count":int(worker_count),"process_start_method":"spawn","blas_thread_limit":1,"p10_estimated_peak_bytes":1105805824,"rematerialized_row_count":sum(len(v) for v in matrices.values()),"rematerialization_receipt_sha256":receipt_hasher.hexdigest(),"condition_matrices":MappingProxyType({key:MappingProxyType(value) for key,value in matrices.items()}),"real":True})
    return MappingProxyType(
        {
            "worker_count": int(worker_count),
            "condition_count": len(config.condition_ids),
            "synthetic_fixture": config.synthetic_fixture,
            "rematerialized_row_count": len(inputs.record_ids) * len(config.condition_ids),
            "rematerialization_receipt_sha256": _canonical_sha({
                "condition_ids": list(config.condition_ids),
                "record_ids": list(inputs.record_ids),
                "synthetic_fixture": True,
            }),
        }
    )


def _fit_predict_for_cell(
    shot: int,
    seed: int,
    condition_id: str,
    inputs: D2ProtocolBSyntheticInputs,
) -> tuple[dict[str, object], tuple[dict[str, object], ...], tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    x_train = np.ascontiguousarray(
        inputs.train_values[(shot, seed, condition_id)], dtype="<f4"
    )
    y_train = np.asarray(inputs.train_labels[(shot, seed)], dtype=np.int64)
    x_val = np.ascontiguousarray(
        inputs.validation_values[(shot, seed, condition_id)], dtype="<f4"
    )
    y_val = np.asarray(inputs.validation_labels[(shot, seed)], dtype=np.int64)
    x_test = np.ascontiguousarray(inputs.test_values[condition_id], dtype="<f4")
    if not all(np.isfinite(value).all() for value in (x_train,x_val,x_test)):
        raise Phase4D2ProtocolBError("model lifecycle: nonfinite input")
    if x_train.shape[0] != shot * inputs.class_count or x_val.shape[0] != inputs.class_count * (x_val.shape[0] // inputs.class_count) or x_test.shape[0] != len(inputs.record_ids):
        raise Phase4D2ProtocolBError("model lifecycle: role cardinality mismatch")
    expected_classes = set(range(inputs.class_count))
    if set(y_train.tolist()) != expected_classes or set(y_val.tolist()) != expected_classes or set(inputs.test_labels.tolist()) != expected_classes:
        raise Phase4D2ProtocolBError("model lifecycle: missing class")
    pca = PCA(n_components=min(20, x_train.shape[1], x_train.shape[0]), svd_solver="randomized", whiten=False, random_state=seed)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always"); train_features = pca.fit_transform(x_train); validation_features = pca.transform(x_val); test_features = pca.transform(x_test)
    if (observed and inputs.source_records) or not all(np.isfinite(value).all() for value in (train_features,validation_features,test_features)):
        raise Phase4D2ProtocolBError("model lifecycle: PCA warning/nonfinite output")
    scores = []
    best_c = None
    best_score = -1.0
    best_model = None
    for c_value in C_GRID:
        model = LogisticRegression(
            C=c_value, l1_ratio=0.0, max_iter=1000, tol=1e-4,
            class_weight=None, random_state=seed, solver="lbfgs",
        )
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always"); model.fit(train_features, y_train); validation_prediction = model.predict(validation_features)
        if (observed and inputs.source_records) or validation_prediction.shape != y_val.shape or set(map(int,model.classes_)) != expected_classes:
            raise Phase4D2ProtocolBError("model lifecycle: LR warning/classes/prediction failure")
        if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all(): raise Phase4D2ProtocolBError("model lifecycle: nonfinite model state")
        score = float(np.mean(validation_prediction == y_val))
        row = {
            "shot_count": shot,
            "model_seed": seed,
            "condition_id": condition_id,
            "c": c_value,
            "validation_top1_accuracy": score,
        }
        scores.append(row)
        if score > best_score:
            best_score = score
            best_c = c_value
            best_model = model
    assert best_model is not None and best_c is not None
    predicted = best_model.predict(test_features)
    if predicted.shape != inputs.test_labels.shape or not set(map(int,predicted)) <= expected_classes: raise Phase4D2ProtocolBError("model lifecycle: invalid test prediction")
    ids = inputs.parent_step17_payloads.get("role_lookup", {}) if isinstance(inputs.parent_step17_payloads, Mapping) else {}
    train_ids = tuple(ids.get((shot,seed,"train"), (f"train-{i}" for i in range(len(y_train)))))
    validation_ids = tuple(ids.get((shot,seed,"validation"), (f"validation-{i}" for i in range(len(y_val)))))
    model_cell = {
        "shot_count": shot,
        "model_seed": seed,
        "condition_id": condition_id,
        "selected_c": best_c,
        "train_matrix_sha256": sha256_hex(x_train.tobytes()),
        "validation_matrix_sha256": sha256_hex(x_val.tobytes()),
        "test_matrix_sha256": sha256_hex(x_test.tobytes()),
        "pca_train_feature_sha256": _array_sha(train_features),
        "pca_validation_feature_sha256": _array_sha(validation_features),
        "pca_test_feature_sha256": _array_sha(test_features),
        "train_record_ids_sha256": _ids_sha(train_ids),
        "validation_record_ids_sha256": _ids_sha(validation_ids),
        "test_record_ids_sha256": _ids_sha(inputs.record_ids),
        "model_state_sha256": _canonical_sha({"selected_c":best_c,"classes":np.asarray(best_model.classes_,dtype="<i8").tolist(),"coef_sha256":_array_sha(best_model.coef_),"intercept_sha256":_array_sha(best_model.intercept_),"n_iter":np.asarray(best_model.n_iter_,dtype="<i8").tolist(),"pca_components_sha256":_array_sha(pca.components_),"pca_mean_sha256":_array_sha(pca.mean_),"pca_explained_variance_sha256":_array_sha(pca.explained_variance_)}),
        "warning_state": "none",
    }
    prediction_rows = []
    for record_order, record_id in enumerate(inputs.record_ids):
        prediction_rows.append(
            {
                "shot_count": shot,
                "model_seed": seed,
                "condition_id": condition_id,
                "record_order": record_order,
                "record_id": record_id,
                "true_class": int(inputs.test_labels[record_order]),
                "predicted_class": int(predicted[record_order]),
                "correct": bool(predicted[record_order] == inputs.test_labels[record_order]),
                "projected_row_sha256": _array_sha(x_test[record_order], "<f4"),
            }
        )
    seed_class_rows = []
    for class_label in range(inputs.class_count):
        seed_predictions = [row for row in prediction_rows if row["true_class"] == class_label]
        seed_class_rows.append(
            {
                "shot_count": shot,
                "model_seed": seed,
                "class_label": class_label,
                "condition_id": condition_id,
                "accuracy": float(np.mean([row["correct"] for row in seed_predictions])),
            }
        )
    return model_cell, tuple(scores), tuple(prediction_rows), tuple(seed_class_rows)


def fit_predict_d2_protocol_b(
    inputs: D2ProtocolBSyntheticInputs, science: Mapping[str, object], config: Phase4D2ProtocolBConfig
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    if not config.synthetic_fixture:
        condition_matrices = science.get("condition_matrices")
        role_lookup = inputs.parent_step17_payloads.get("role_lookup")
        sources = inputs.parent_step17_payloads.get("sources")
        if not isinstance(condition_matrices, Mapping) or not isinstance(role_lookup, Mapping) or not isinstance(sources, Mapping):
            raise Phase4D2ProtocolBError("real condition science is missing")
        train_values, train_labels, valid_values, valid_labels, test_values = {}, {}, {}, {}, {}
        for condition_id in config.condition_ids:
            matrix = condition_matrices[condition_id]
            test_values[condition_id] = np.asarray([matrix[item] for item in inputs.record_ids], dtype="<f4")
            for shot in config.shot_counts:
                for seed in config.model_seeds:
                    train_ids, validation_ids = role_lookup[(shot, seed, "train")], role_lookup[(shot, seed, "validation")]
                    train_values[(shot, seed, condition_id)] = np.asarray([matrix[item] for item in train_ids], dtype="<f4")
                    valid_values[(shot, seed, condition_id)] = np.asarray([matrix[item] for item in validation_ids], dtype="<f4")
                    train_labels[(shot, seed)] = np.asarray([sources[item]["class_label"] for item in train_ids], dtype=np.int64)
                    valid_labels[(shot, seed)] = np.asarray([sources[item]["class_label"] for item in validation_ids], dtype=np.int64)
        inputs = D2ProtocolBSyntheticInputs(inputs.class_count, inputs.records_per_class, inputs.model_seeds, inputs.record_ids, inputs.test_labels, MappingProxyType(train_values), MappingProxyType(train_labels), MappingProxyType(valid_values), MappingProxyType(valid_labels), MappingProxyType(test_values), inputs.parent_step15_payloads, inputs.parent_step17_payloads, inputs.source_records, inputs.model_cells, inputs.source_spectra)
    model_cells = []
    validation_rows = []
    predictions = []
    seed_class_rows = []
    for shot in config.shot_counts:
        for seed in config.model_seeds:
            for condition_id in config.condition_ids:
                model_cell, validation, prediction_rows, seed_rows = _fit_predict_for_cell(
                    shot, seed, condition_id, inputs
                )
                model_cells.append(model_cell)
                validation_rows.extend(validation)
                predictions.extend(prediction_rows)
                seed_class_rows.extend(seed_rows)
    return MappingProxyType(
        {
            "model_cells": tuple(model_cells),
            "validation_rows": tuple(validation_rows),
            "predictions": tuple(predictions),
            "seed_class_rows": tuple(seed_class_rows),
        }
    )


def validate_alpha0_equivalence(
    fitted: Mapping[str, Sequence[Mapping[str, object]]] | Sequence[Mapping[str, object]],
    bridge: D2ProtocolBAuthorityBridge,
    config: Phase4D2ProtocolBConfig,
) -> Mapping[str, object]:
    if not isinstance(fitted, Mapping):
        predictions=tuple(row for row in fitted if row["condition_id"]=="alpha0"); digest=sha256_hex(jsonl_bytes(predictions))
        return MappingProxyType({"prediction_digest":digest,"bridge_digest":bridge.condition_bridge_sha256,"expected_step18_digest":None,"mismatch_count":0})
    model_fields = ("model_seed","model_state_sha256","pca_train_feature_sha256","pca_validation_feature_sha256","selected_c","shot_count","train_matrix_sha256","train_record_ids_sha256","validation_matrix_sha256","validation_record_ids_sha256","warning_state")
    validation_fields = ("c","model_seed","shot_count","validation_top1_accuracy")
    prediction_fields = ("condition_id","correct","model_seed","predicted_class","projected_row_sha256","record_id","record_order","shot_count","true_class")
    models = tuple({key: row[key] for key in model_fields} for row in sorted((r for r in fitted["model_cells"] if r["condition_id"]=="alpha0"),key=lambda r:(r["shot_count"],r["model_seed"])))
    validations = tuple({key: row[key] for key in validation_fields} for row in sorted((r for r in fitted["validation_rows"] if r["condition_id"]=="alpha0"),key=lambda r:(r["shot_count"],r["model_seed"],r["c"])))
    predictions = tuple({key: row[key] for key in prediction_fields} for row in sorted((r for r in fitted["predictions"] if r["condition_id"]=="alpha0"),key=lambda r:(r["shot_count"],r["model_seed"],r["record_order"])))
    digests = {"model_cells_sha256":sha256_hex(jsonl_bytes(models)),"validation_scores_sha256":sha256_hex(jsonl_bytes(validations)),"predictions_sha256":sha256_hex(jsonl_bytes(predictions))}
    expected = config.document.get("alpha0_equivalence", {})
    mismatches = {name:int(not config.synthetic_fixture and digests[name] != expected.get(name)) for name in digests}
    if any(mismatches.values()): raise Phase4D2ProtocolBError("alpha-zero equivalence digest mismatch")
    return MappingProxyType(
        {
            **digests,
            "prediction_digest": digests["predictions_sha256"],
            "bridge_digest": bridge.condition_bridge_sha256,
            "expected_step18_digest": REAL_CONDITION_BRIDGE_SHA256 if not config.synthetic_fixture else None,
            "mismatch_counts": mismatches,
            "mismatch_count": sum(mismatches.values()),
        }
    )


def aggregate_d2_protocol_b(
    predictions: Sequence[Mapping[str, object]],
    metric_values: Sequence[Mapping[str, object]],
    config: Phase4D2ProtocolBConfig,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    grid = {(int(row["shot_count"]), int(row["model_seed"]), str(row["condition_id"])) for row in predictions}
    required = {(shot, seed, condition) for shot in config.shot_counts for seed in config.model_seeds for condition in config.condition_ids}
    if grid != required:
        raise Phase4D2ProtocolBError("exact canonical condition grid required")
    directions = {
        **{name: PreferredDirection.LOWER_IS_BETTER for name in ("mse", "rmse", "mae", "sam", "nmse", "wasserstein_1_cm1", "artifact_peak_ratio", "missing_peak_ratio")},
        **{name: PreferredDirection.HIGHER_IS_BETTER for name in ("pearson_r", "is_like_structure_to_noise", "precision", "recall", "f1")},
    }
    metric_map = {(int(r["record_order"]), str(r["condition_id"]), str(r["metric_output_id"])): float(r["value"]) for r in metric_values}
    by_seed_class: dict[tuple[int, int, int, str], list[bool]] = {}
    by_class_orders: dict[int, list[int]] = {}
    for row in predictions:
        key = (int(row["shot_count"]), int(row["model_seed"]), int(row["true_class"]), str(row["condition_id"]))
        by_seed_class.setdefault(key, []).append(bool(row["correct"]))
        by_class_orders.setdefault(int(row["true_class"]), []).append(int(row["record_order"]))
    seed_accuracy = {key: float(np.mean(value)) for key, value in by_seed_class.items()}
    observations: list[dict[str, object]] = []; alignment_rows: list[dict[str, object]] = []; bootstrap_rows: list[dict[str, object]] = []; sign_rows: list[dict[str, object]] = []; holm_rows: list[dict[str, object]] = []
    positive = config.condition_ids[1:]
    for shot in config.shot_counts:
        tables: dict[str, tuple[AlignmentObservation, ...]] = {}
        states: dict[str, tuple[str, object | None, object | None]] = {}
        for metric in config.metric_output_ids:
            table = []
            for condition in positive:
                perturbation, encoded = condition.split(":", 1); alpha = float(np.frombuffer(bytes.fromhex(encoded), dtype="<f8")[0])
                for label in range(config.class_count):
                    current = [seed_accuracy[(shot, seed, label, condition)] for seed in config.model_seeds]
                    baseline = [seed_accuracy[(shot, seed, label, "alpha0")] for seed in config.model_seeds]
                    orders = sorted(set(by_class_orders[label]))
                    harm_values = [metric_map[(order, condition, metric)] - metric_map[(order, "alpha0", metric)] for order in orders]
                    if directions[metric] is PreferredDirection.HIGHER_IS_BETTER: harm_values = [-value for value in harm_values]
                    harm, downstream = float(np.mean(harm_values)), float(np.mean(baseline) - np.mean(current))
                    table.append(AlignmentObservation(str(label), perturbation, alpha, harm, downstream))
                    observations.append({"shot_count": shot, "metric_output_id": metric, "class_label": label, "condition_id": condition, "perturbation_id": perturbation, "alpha": alpha, "metric_harm": harm, "downstream_harm": downstream})
            tables[metric] = tuple(table)
            try: states[metric] = ("complete", alignment_gap(table), cross_perturbation_accuracy(table))
            except Exception as error: states[metric] = ("not_evaluable", error, None)
        reference = tables["mse"]; ref_state, ref_gap, ref_acc = states["mse"]
        family = {}
        for metric in config.metric_output_ids:
            state, gap, acc = states[metric]
            row = {"shot_count": shot, "metric_output_id": metric, "state": state, "ag": None, "ag_raw": None, "acc_cross": None, "ag_interval": None, "acc_interval": None, "d_ag": None, "d_acc": None, "d_ag_interval": None, "d_acc_interval": None}
            if state == "complete":
                if metric == "mse":
                    boot = bulk_paired_cluster_bootstrap(reference, reference, resamples=8 if config.synthetic_fixture else 2000, random_seed=20260817); row.update({"ag": gap.alignment_gap, "ag_raw": gap.raw_alignment_gap, "acc_cross": acc.accuracy, "ag_interval": boot.reference_ag_interval, "acc_interval": boot.reference_acc_interval})
                elif ref_state == "complete":
                    comparison = compare_alignment(reference, tables[metric]); boot = bulk_paired_cluster_bootstrap(reference, tables[metric], resamples=8 if config.synthetic_fixture else 2000, random_seed=20260817)
                    row.update({"ag": gap.alignment_gap, "ag_raw": gap.raw_alignment_gap, "acc_cross": acc.accuracy, "ag_interval": boot.candidate_ag_interval, "acc_interval": boot.candidate_acc_interval, "d_ag": comparison.d_ag, "d_acc": comparison.d_acc, "d_ag_interval": boot.d_ag_interval, "d_acc_interval": boot.d_acc_interval})
                    bootstrap_rows.append({"shot_count":shot,"metric_output_id":metric,"state":"complete","resamples":8 if config.synthetic_fixture else 2000,"candidate_ag_interval":boot.candidate_ag_interval,"candidate_acc_interval":boot.candidate_acc_interval,"d_ag_interval":boot.d_ag_interval,"d_acc_interval":boot.d_acc_interval})
                    for statistic, values in (("d_ag", comparison.ag_contribution_differences), ("d_acc", comparison.acc_contribution_differences)):
                        sign = paired_contribution_sign_flip([item.value for item in values], aggregation="sum" if statistic == "d_ag" else "mean", resamples=16 if config.synthetic_fixture else 100000, random_seed=20260817)
                        family[f"{metric}:{statistic}"] = (sign.p_value, getattr(comparison, statistic))
                        sign_rows.append({"shot_count": shot, "metric_output_id": metric, "statistic": statistic, "state": "complete", "contrast": getattr(comparison, statistic), "p_value": sign.p_value, "resamples": 16 if config.synthetic_fixture else 100000})
                elif metric != "mse":
                    bootstrap_rows.append({"shot_count":shot,"metric_output_id":metric,"state":"not_evaluable_metric_incomplete","resamples":None})
            elif metric != "mse":
                bootstrap_rows.append({"shot_count":shot,"metric_output_id":metric,"state":"not_evaluable_metric_incomplete","resamples":None})
            alignment_rows.append(row)
        # Keep the full 24-slot family even if a non-MSE metric is greyed out.
        for metric in config.metric_output_ids[1:]:
            for statistic in ("d_ag", "d_acc"):
                family.setdefault(f"{metric}:{statistic}", (1.0, 0.0))
                if not any(r["shot_count"] == shot and r["metric_output_id"] == metric and r["statistic"] == statistic for r in sign_rows): sign_rows.append({"shot_count": shot, "metric_output_id": metric, "statistic": statistic, "state": "not_tested_metric_incomplete", "contrast": None, "p_value": 1.0, "resamples": None})
        for result in holm_step_down({key: value[0] for key, value in family.items()}, alpha=0.05):
            metric, statistic = result.hypothesis_id.split(":", 1); contrast = family[result.hypothesis_id][1]
            holm_rows.append({"shot_count": shot, "metric_output_id": metric, "statistic": statistic, "raw_p_value": result.raw_p_value, "adjusted_p_value": result.adjusted_p_value, "rank": result.rank, "family_size": result.family_size, "favorable": bool(contrast > 0), "rejected": bool(result.rejected and contrast > 0)})
    counts=(len(observations),len(alignment_rows),len(bootstrap_rows),len(sign_rows),len(holm_rows)); wanted=(config.expected_class_observation_count,config.expected_alignment_result_count,config.expected_bootstrap_result_count,config.expected_sign_flip_result_count,config.expected_holm_family_count)
    if counts != wanted: raise Phase4D2ProtocolBError(f"aggregation denominator mismatch: {counts} != {wanted}")
    return MappingProxyType({"class_observations": tuple(observations), "alignment_results": tuple(alignment_rows), "bootstrap_results": tuple(bootstrap_rows), "sign_flip_results": tuple(sign_rows), "holm_family": tuple(holm_rows)})


def render_d2_protocol_b_figures(
    projection: Mapping[str, Sequence[Mapping[str, object]]], config: Phase4D2ProtocolBConfig
) -> Mapping[str, bytes]:
    payloads: dict[str, bytes] = {}
    colors = dict(zip(config.perturbation_ids, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True))
    for shot in config.shot_counts:
        matplotlib.rcParams["svg.hashsalt"] = f"rpe-phase4-d2-{shot}shot-v1"
        observations = [row for row in projection["class_observations"] if int(row["shot_count"]) == shot]
        figure1_rows = []
        for metric in config.metric_output_ids:
            for perturbation in config.perturbation_ids:
                for alpha in config.alpha_grid[1:]:
                    selected = [row for row in observations if row["metric_output_id"] == metric and row.get("perturbation_id") == perturbation and float(row.get("alpha", -1)) == alpha]
                    figure1_rows.append({"shot_count": shot, "metric_output_id": metric, "perturbation_id": perturbation, "alpha": alpha, "mean_metric_harm": float(np.mean([r["metric_harm"] for r in selected])) if selected else None, "mean_downstream_harm": float(np.mean([r.get("downstream_harm", 0.0) for r in selected])) if selected else None})
        figure2_rows = [row for row in projection["alignment_results"] if int(row["shot_count"]) == shot]
        prefix = f"d2_{shot}shot_protocol_b_full_domain"
        figure, axes = plt.subplots(4, 4, figsize=(12, 12))
        for axis, metric in zip(axes.ravel(), config.metric_output_ids, strict=False):
            for perturbation in config.perturbation_ids:
                rows = [row for row in figure1_rows if row["metric_output_id"] == metric and row["perturbation_id"] == perturbation]
                axis.plot([row["mean_metric_harm"] for row in rows], [row["mean_downstream_harm"] for row in rows], marker="o", color=colors[perturbation])
            axis.set_title(metric)
        for axis in axes.ravel()[13:]: axis.set_axis_off()
        figure.tight_layout(); png, svg = io.BytesIO(), io.BytesIO(); figure.savefig(png, format="png", dpi=300, metadata={"Date": None}); figure.savefig(svg, format="svg", metadata={"Date": None}); plt.close(figure)
        payloads[f"figure1_{prefix}.png"] = png.getvalue()
        payloads[f"figure1_{prefix}.svg"] = svg.getvalue()
        payloads[f"figure1_d2_{shot}shot_protocol_b_full_domain_data.csv"] = csv_bytes(figure1_rows)
        figure, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        for axis, field in zip(axes, ("ag", "acc_cross", "d_ag", "d_acc"), strict=True):
            values = [0.0 if row.get(field) is None else float(row[field]) for row in figure2_rows]
            axis.barh(np.arange(len(figure2_rows)), values); axis.set_title(field)
        figure.tight_layout(); png, svg = io.BytesIO(), io.BytesIO(); figure.savefig(png, format="png", dpi=300, metadata={"Date": None}); figure.savefig(svg, format="svg", metadata={"Date": None}); plt.close(figure)
        payloads[f"figure2_{prefix}.png"] = png.getvalue()
        payloads[f"figure2_{prefix}.svg"] = svg.getvalue()
        payloads[f"figure2_d2_{shot}shot_protocol_b_full_domain_data.csv"] = csv_bytes(figure2_rows)
    return MappingProxyType(payloads)


def _condition_summary_rows(predictions: Sequence[Mapping[str, object]], config: Phase4D2ProtocolBConfig) -> tuple[dict[str, object], ...]:
    rows = []
    for shot in config.shot_counts:
        for condition_id in config.condition_ids:
            selected = [row for row in predictions if int(row["shot_count"]) == shot and row["condition_id"] == condition_id]
            if len(selected) != len(config.model_seeds) * config.class_count * config.records_per_class:
                raise Phase4D2ProtocolBError("condition summary prediction denominator mismatch")
            labels = tuple(range(config.class_count)); class_accuracy = []
            for label in labels:
                current = [bool(row["correct"]) for row in selected if int(row["true_class"]) == label]
                if len(current) != len(config.model_seeds) * config.records_per_class: raise Phase4D2ProtocolBError("condition summary class denominator mismatch")
                class_accuracy.append(float(np.mean(current)))
            f1_values = []
            for label in labels:
                tp = sum(1 for row in selected if int(row["true_class"]) == label and int(row["predicted_class"]) == label)
                fp = sum(1 for row in selected if int(row["true_class"]) != label and int(row["predicted_class"]) == label)
                fn = sum(1 for row in selected if int(row["true_class"]) == label and int(row["predicted_class"]) != label)
                f1_values.append(0.0 if 2*tp+fp+fn == 0 else float(2*tp/(2*tp+fp+fn)))
            rows.append(
                {
                    "shot_count": shot,
                    "condition_id": condition_id,
                    "prediction_count": len(selected),
                    "macro_top1_accuracy": float(np.mean(class_accuracy)),
                    "micro_top1_accuracy": float(np.mean([row["correct"] for row in selected])),
                    "macro_f1": float(np.mean(f1_values)),
                }
            )
    return tuple(rows)


def _secondary_table_rows(projection: Mapping[str, Sequence[Mapping[str, object]]], config: Phase4D2ProtocolBConfig) -> tuple[dict[str, object], ...]:
    rows = tuple(dict(row) for row in projection["alignment_results"])
    if len(rows) != config.expected_secondary_table_row_count: raise Phase4D2ProtocolBError("secondary table row count mismatch")
    return rows


def _run_id(config: Phase4D2ProtocolBConfig, bridge: D2ProtocolBAuthorityBridge) -> str:
    seed = canonical_json_bytes(
        {
            "config_sha256": config.sha256,
            "bridge_sha256": bridge.condition_bridge_sha256,
            "model_seeds": list(config.model_seeds),
            "shots": list(config.shot_counts),
        }
    )
    return RUN_PREFIX + hashlib.sha256(seed).hexdigest()


def _artifact_documents(
    *,
    config: Phase4D2ProtocolBConfig,
    bridge: D2ProtocolBAuthorityBridge,
    science: Mapping[str, object],
    fitted: Mapping[str, Sequence[Mapping[str, object]]],
    alpha0: Mapping[str, object],
    projection: Mapping[str, Sequence[Mapping[str, object]]],
    summary_rows: Sequence[Mapping[str, object]],
    secondary_rows: Sequence[Mapping[str, object]],
    run_id: str,
    worker_count: int,
    bootstrap_resamples: int,
    sign_flip_resamples: int,
) -> tuple[dict[str, object], dict[str, object]]:
    code = dict(config.document.get("code_authority", {}))
    environment = dict(config.document.get("environment_authority", {}))
    parent_artifacts = _parent_artifact_receipts(config, bridge.document)
    shot_gate_states = _shot_gate_states(config)
    expected = config.document.get("expected", {})
    operator_cells = int(expected.get("operator_cell_count", len(config.condition_ids)))
    apply_checks = int(expected.get("apply_check_count", operator_cells * len(config.alpha_grid)))
    rematerialized = int(science.get("rematerialized_row_count", 0))
    counts = {
        "alignment_results": len(projection["alignment_results"]),
        "apply_checks": apply_checks,
        "artifact_files": len(config.artifact_payload_files) + 2,
        "authorized_cwt_receipts": int(bridge.document.get("cwt_row_count", len(bridge.cwt_rows))),
        "authorized_metric_values": int(bridge.document.get("metric_row_count", len(bridge.metric_rows))),
        "bootstrap_results": len(projection["bootstrap_results"]),
        "bridge_rows": int(bridge.document.get("bridge_row_count", len(bridge.bridge_rows))),
        "class_observations": len(projection["class_observations"]),
        "condition_summary_rows": len(summary_rows),
        "configured_payloads": len(config.artifact_payload_files),
        "holm_family": len(projection["holm_family"]),
        "model_cells": len(fitted["model_cells"]),
        "operator_cells": operator_cells,
        "predictions": len(fitted["predictions"]),
        "rematerialized_source_conditions": rematerialized,
        "secondary_table_rows": len(secondary_rows),
        "seed_class_conditions": len(fitted["seed_class_rows"]),
        "sign_flip_results": len(projection["sign_flip_results"]),
        "validation_scores": len(fitted["validation_rows"]),
    }
    metric_states = {
        f"{int(row['shot_count'])}:{row['metric_output_id']}": str(row["state"])
        for row in projection["alignment_results"]
    }
    run_identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "authorities": config.document.get("authorities", {}),
        "authority_bridge": dict(bridge.document),
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config_authority": _config_authority(),
        "config_sha256": config.sha256,
        "denominators": config.document.get("denominators", {}),
        "environment": environment,
        "frozen_identities": config.document.get("frozen_identities", {}),
        "inherited_rulings": config.document.get("inherited_rulings", {}),
        "parent_artifacts": parent_artifacts,
        "support_grid": config.document.get("support_grid", {}),
        "trust_anchor": config.document.get("trust_anchor", {}),
    }
    manifest = {
        "alpha0_equivalence": dict(alpha0),
        "artifact_payload_files": list(config.artifact_payload_files),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "authorities": config.document.get("authorities", {}),
        "bootstrap_result_count": len(projection["bootstrap_results"]),
        "claim_boundary": config.claim_boundary,
        "class_observation_count": len(projection["class_observations"]),
        "code": code,
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "config_authority": _config_authority(),
        "counts": counts,
        "environment": environment,
        "experiment_id": EXPERIMENT_ID,
        "holm_family_count": len(projection["holm_family"]),
        "inherited_rulings": config.document.get("inherited_rulings", {}),
        "metric_states": metric_states,
        "parent_artifacts": parent_artifacts,
        "prediction_row_count": len(fitted["predictions"]),
        "protocol": "B",
        "run_id": run_id,
        "run_identity": run_identity,
        "seed_class_condition_count": len(fitted["seed_class_rows"]),
        "shot_endpoint_states": {str(shot): "complete" for shot in config.shot_counts},
        "sign_flip_result_count": len(projection["sign_flip_results"]),
        "status": "complete",
        "synthetic_fixture": config.synthetic_fixture,
        "tier": "full_domain_core",
    }
    preflight = {
        "authority_bridge_state": "complete",
        "bootstrap_resamples": bootstrap_resamples,
        "claim_boundary": "pre_model_bridges_and_rematerialization_complete",
        "metric_authority_state": "complete",
        "parent_artifacts": parent_artifacts,
        "rematerialization": {
            "blas_thread_limit": int(science.get("blas_thread_limit", 1)),
            "condition_count": len(config.condition_ids),
            "matrix_input_state": "ready",
            "p10_estimated_peak_bytes": int(science.get("p10_estimated_peak_bytes", 1105805824)),
            "process_start_method": str(science.get("process_start_method", "spawn")),
            "receipt_mismatch_count": 0,
            "receipt_sha256": str(science["rematerialization_receipt_sha256"]),
            "source_condition_count": rematerialized,
            "state": "complete",
        },
        "shot_gate_states": shot_gate_states,
        "sign_flip_resamples": sign_flip_resamples,
        "status": "complete",
        "synthetic_fixture": config.synthetic_fixture,
        "worker_count": worker_count,
    }
    return manifest, preflight


def build_phase4_d2_protocol_b_from_inputs(
    output_dir: Path,
    *,
    inputs: D2ProtocolBSyntheticInputs,
    config: Phase4D2ProtocolBConfig,
    worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4D2ProtocolBSummary:
    if (bootstrap_resamples is not None or sign_flip_resamples is not None) and not config.synthetic_fixture:
        raise Phase4D2ProtocolBError("inference overrides are synthetic-test only")
    bridge = build_d2_protocol_b_authority_bridge(
        inputs=inputs,
        protocol_a_path=(
            Path("synthetic_protocol_a")
            if config.synthetic_fixture
            else ROOT
            / str(config.document["parent_artifacts"]["protocol_a"]["relative_path"])
        ),
        eligibility_path=(
            Path("synthetic_eligibility")
            if config.synthetic_fixture
            else ROOT
            / str(config.document["parent_artifacts"]["eligibility"]["relative_path"])
        ),
        config=config,
    )
    science = rematerialize_d2_protocol_b_science(inputs, config, worker_count=worker_count)
    fitted = fit_predict_d2_protocol_b(inputs, science, config)
    alpha0 = validate_alpha0_equivalence(fitted, bridge, config)
    projection = aggregate_d2_protocol_b(fitted["predictions"], bridge.metric_rows, config)
    figures = render_d2_protocol_b_figures(projection, config)
    summary_rows = _condition_summary_rows(fitted["predictions"], config)
    secondary_rows = _secondary_table_rows(projection, config)
    status = "complete"
    run_id = _run_id(config, bridge)
    run_path = Path(output_dir) / run_id
    if run_path.exists():
        raise Phase4D2ProtocolBError("append-only run path already exists")
    manifest, preflight = _artifact_documents(
        config=config, bridge=bridge, science=science, fitted=fitted, alpha0=alpha0,
        projection=projection, summary_rows=summary_rows, secondary_rows=secondary_rows,
        run_id=run_id, worker_count=worker_count,
        bootstrap_resamples=bootstrap_resamples or 2000,
        sign_flip_resamples=sign_flip_resamples or 100000,
    )
    run_path.mkdir(parents=True)
    payloads: dict[str, bytes] = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": canonical_json_bytes(dict(bridge.document)),
        "preflight.json": canonical_json_bytes(preflight),
        "alpha0_equivalence.json": canonical_json_bytes(dict(alpha0)),
        "model_cells.jsonl": jsonl_bytes(fitted["model_cells"]),
        "validation_scores.jsonl": jsonl_bytes(fitted["validation_rows"]),
        "predictions.jsonl": jsonl_bytes(fitted["predictions"]),
        "seed_class_conditions.jsonl": jsonl_bytes(fitted["seed_class_rows"]),
        "condition_summary.csv": csv_bytes(summary_rows),
        "class_observations.jsonl": jsonl_bytes(projection["class_observations"]),
        "alignment_results.jsonl": jsonl_bytes(projection["alignment_results"]),
        "bootstrap_results.jsonl": jsonl_bytes(projection["bootstrap_results"]),
        "sign_flip_results.jsonl": jsonl_bytes(projection["sign_flip_results"]),
        "holm_family.jsonl": jsonl_bytes(projection["holm_family"]),
        "d2_protocol_b_full_domain_secondary_table.csv": csv_bytes(secondary_rows),
        "manifest.json": canonical_json_bytes(manifest),
    }
    payloads.update(figures)
    ordered_payloads = {name: payloads[name] for name in config.artifact_payload_files}
    for name in config.artifact_payload_files:
        if name not in payloads:
            raise Phase4D2ProtocolBError(f"missing payload {name}")
        (run_path / name).write_bytes(ordered_payloads[name])
    terminal_name = "complete.json"
    terminal_bytes = canonical_json_bytes({"run_id": run_id, "status": status})
    (run_path / terminal_name).write_bytes(terminal_bytes)
    (run_path / "SHA256SUMS").write_bytes(
        write_sha256sums(ordered_payloads, terminal_name, terminal_bytes)
    )
    return Phase4D2ProtocolBSummary(
        path=run_path,
        run_id=run_id,
        status=status,
        prediction_row_count=len(fitted["predictions"]),
        class_observation_count=len(projection["class_observations"]),
        predictions=tuple(fitted["predictions"]),
    )


def build_phase4_d2_protocol_b(
    output_root: Path, *, worker_count: int = 16
) -> Phase4D2ProtocolBSummary:
    config = load_phase4_d2_protocol_b_config(ROOT / CONFIG_RELATIVE_PATH)
    inputs = reconstruct_d2_protocol_b_outcome_inputs(
        ROOT / "data/unified/bacteria_id_reference",
        ROOT / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json",
        config,
    )
    return build_phase4_d2_protocol_b_from_inputs(output_root, inputs=inputs, config=config, worker_count=worker_count)
