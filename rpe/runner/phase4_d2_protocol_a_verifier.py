"""Structurally independent verifier for Phase 4 D2 Protocol-A artifacts.

This module deliberately owns its config/input/model/serialization path.  It
does not import the D2 outcome runner: only retained-data and scientific
primitives are shared with the primary implementation.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import multiprocessing
import platform
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy
import sklearn
import threadpoolctl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from rpe.alignment import (
    AlignmentObservation, AlignmentValidationError, alignment_gap,
    bulk_paired_cluster_bootstrap, compare_alignment, cross_perturbation_accuracy,
    holm_step_down, paired_contribution_sign_flip,
)
from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.evaluation import PeakPairInput, SingleSpectrumInput, SpectrumPairInput, Spectrum1D, evaluate_metric
from rpe.methods.catalog import load_classical_catalog
from rpe.methods.classical.peaks import PeakRunStatus, run_peak_detection_system
from rpe.metrics import ISLikeStructureToNoiseMetric, MAEMetric, MSEMetric, NMSEMetric, PeakDetectionCurvesMetric, PearsonRMetric, RMSEMetric, SAMMetric, Wasserstein1Metric
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.d2_selection import validate_d2_few_shot_selection
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase4_d2_protocol_a_authority import CONFIG_BYTES, CONFIG_SHA256
from rpe.runner.phase4_d2_eligibility_verifier import _canonical_json_bytes as _step13_canonical


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-d2-protocol-a-full-domain-config-v1"
EXPERIMENT_ID = "phase4-d2-protocol-a-full-domain-v1"
SHOTS = (5, 10, 20)
SEEDS = (0, 1, 2, 3, 4)
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
METRICS = (
    "mse", "rmse", "mae", "sam", "pearson_r", "nmse",
    "wasserstein_1_cm1", "is_like_structure_to_noise", "precision",
    "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio",
)
DIRECTIONS = MappingProxyType({
    **{name: "lower_is_better" for name in ("mse", "rmse", "mae", "sam", "nmse", "wasserstein_1_cm1", "artifact_peak_ratio", "missing_peak_ratio")},
    **{name: "higher_is_better" for name in ("pearson_r", "is_like_structure_to_noise", "precision", "recall", "f1")},
})
PAYLOADS = (
    "config.json", "eligibility_bridge.json", "model_cells.jsonl", "validation_scores.jsonl",
    "record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl", "downstream_rows.jsonl",
    "predictions.jsonl", "seed_class_conditions.jsonl", "condition_summary.csv", "class_observations.jsonl",
    "alignment_results.jsonl", "bootstrap_results.jsonl", "sign_flip_results.jsonl", "holm_family.jsonl",
    "figure1_d2_5shot_protocol_a_full_domain.png", "figure1_d2_5shot_protocol_a_full_domain.svg", "figure1_d2_5shot_protocol_a_full_domain_data.csv",
    "figure2_d2_5shot_protocol_a_full_domain.png", "figure2_d2_5shot_protocol_a_full_domain.svg", "figure2_d2_5shot_protocol_a_full_domain_data.csv",
    "figure1_d2_10shot_protocol_a_full_domain.png", "figure1_d2_10shot_protocol_a_full_domain.svg", "figure1_d2_10shot_protocol_a_full_domain_data.csv",
    "figure2_d2_10shot_protocol_a_full_domain.png", "figure2_d2_10shot_protocol_a_full_domain.svg", "figure2_d2_10shot_protocol_a_full_domain_data.csv",
    "figure1_d2_20shot_protocol_a_full_domain.png", "figure1_d2_20shot_protocol_a_full_domain.svg", "figure1_d2_20shot_protocol_a_full_domain_data.csv",
    "figure2_d2_20shot_protocol_a_full_domain.png", "figure2_d2_20shot_protocol_a_full_domain.svg", "figure2_d2_20shot_protocol_a_full_domain_data.csv",
    "d2_protocol_a_full_domain_secondary_table.csv", "manifest.json",
)
STEP13_RUN_ID = "phase4-d2-protocol-a-full-domain-eligibility-0f43f329bfc6a7a1ff7232a336d115849b06dbca884de28d71f6b28502598ef8"
STEP13_SUMS = "b41638b4e246326b155ef1c6169b57d93c68266b47a41b9d16f05dc516546e9e"
STEP13_CONFIG = "f92427c2f18ab445db2bb54d5ea5a97cce21b05dd6ee80f50f3ed4082285a15b"
CLAIM_BOUNDARY = "local_execution_artifact_redistribution_not_cleared"
CODE_PATHS = (
    "rpe/alignment/bulk.py", "rpe/alignment/contracts.py", "rpe/alignment/core.py", "rpe/alignment/inference.py",
    "rpe/downstream/bacteria_id.py", "rpe/evaluation/contracts.py", "rpe/methods/catalog.py",
    "rpe/methods/classical/peaks.py", "rpe/metrics/fidelity.py", "rpe/metrics/peak.py",
    "rpe/metrics/reference_free.py", "rpe/metrics/transport.py", "rpe/perturb/axis_transform.py",
    "rpe/perturb/baseline_distortion.py", "rpe/perturb/contracts.py", "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py", "rpe/perturb/sweep.py", "rpe/runner/d2_selection.py",
    "rpe/runner/phase1_config.py", "rpe/runner/phase1_gates.py", "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py", "rpe/runner/phase1_types.py", "rpe/runner/phase4_d2_eligibility.py",
    "rpe/runner/phase4_d2_eligibility_verifier.py", "rpe/runner/phase4_d2_protocol_a.py",
    "rpe/runner/phase4_d2_protocol_a_verifier.py", "tools/run_phase4_d2_protocol_a.py",
)


class Phase4D2ProtocolAVerifierError(ValueError):
    pass


@dataclass(frozen=True)
class _Config:
    path: Path
    raw: bytes
    sha256: str
    synthetic: bool
    class_count: int
    test_count: int
    seeds: tuple[int, ...]
    support_points: int
    document: Mapping[str, object]


@dataclass(frozen=True)
class _Inputs:
    test_values: np.ndarray
    test_labels: np.ndarray
    record_ids: tuple[str, ...]
    model_seeds: tuple[int, ...]
    train_values: Mapping[tuple[int, int], np.ndarray] | None
    train_labels: Mapping[tuple[int, int], np.ndarray] | None
    validation_values: Mapping[tuple[int, int], np.ndarray] | None
    validation_labels: Mapping[tuple[int, int], np.ndarray] | None
    model_cells: Mapping[tuple[int, int], Mapping[str, object]] | None
    support_axis: np.ndarray | None
    native_spectra: tuple[Spectrum1D, ...]
    native_labels: np.ndarray | None


@dataclass(frozen=True)
class _Model:
    shot: int
    seed: int
    selected_c: float
    pca: PCA
    classifier: LogisticRegression
    validation_scores: tuple[Mapping[str, object], ...]
    train_sha: str
    validation_sha: str
    train_feature_sha: str
    validation_feature_sha: str


@dataclass(frozen=True)
class Phase4D2ProtocolASummary:
    path: Path
    run_id: str
    status: str
    test_record_count: int
    model_cell_count: int


def _canonical(value: object) -> bytes:
    return (json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def _json_ready(value: object) -> object:
    if is_dataclass(value): return {item.name: _json_ready(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping): return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json_ready(item) for item in value]
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): value = float(value)
    if value is None or isinstance(value, (bool, int, str)) or (isinstance(value, float) and math.isfinite(value)): return value
    raise Phase4D2ProtocolAVerifierError(f"JSON: unsupported {type(value).__name__}")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _observed_environment() -> dict[str, str]:
    return {
        "h5py": h5py.__version__, "machine": platform.machine(), "matplotlib": matplotlib.__version__,
        "numpy": np.__version__, "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__, "system": platform.system(), "threadpoolctl": threadpoolctl.__version__,
    }


def _observed_code() -> dict[str, dict[str, object]]:
    return {name: {"bytes": (ROOT / name).stat().st_size, "sha256": _sha_file(ROOT / name)} for name in CODE_PATHS}


def _verify_frozen_config_contract(root: Mapping[str, object]) -> None:
    directions = {
        name: ("higher_is_better" if name in {"pearson_r", "is_like_structure_to_noise", "precision", "recall", "f1"} else "lower_is_better")
        for name in METRICS
    }
    required = {
        "protocol": "A", "tier": "full_domain_core", "claim_boundary": CLAIM_BOUNDARY,
        "shot_counts": list(SHOTS), "model_seeds": list(SEEDS), "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS), "metric_output_ids": list(METRICS),
        "metric_manifest": [{"output_id": name, "preferred_direction": directions[name]} for name in METRICS],
        "cwt_system_id": "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511",
        "denominators": {"class_count": 30, "test_record_count": 3000, "source_record_count": 5513, "model_cell_count": 15, "model_role_occurrence_count": 54750},
        "expected": {
            "operator_cell_count": 15000, "apply_check_count": 135000, "condition_count": 123000,
            "metric_value_count": 1599000, "cwt_receipt_count": 123000, "model_cell_count": 15, "pca_fit_count": 15,
            "lr_candidate_fit_count": 60, "frozen_model_condition_call_count": 615, "prediction_row_count": 1845000,
            "seed_class_condition_count": 18450, "class_observation_count": 46800, "alignment_result_count": 39,
            "bootstrap_result_count": 36, "sign_flip_result_count": 72, "holm_family_row_count": 72,
            "figure1_row_count_per_shot": 520, "figure2_row_count_per_shot": 13, "secondary_table_row_count": 39,
            "cross_perturbation_pair_count_per_class": 640, "cross_perturbation_pair_count_per_shot_metric": 19200,
            "configured_payload_count": 36, "terminal_marker_count": 1, "sha256sums_count": 1, "artifact_file_count": 38,
        },
        "support_grid": {"point_count": 997, "max_in_range_native_gap_cm1": 1.561, "support_axis_f32_sha256": "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59", "support_axis_f64_sha256": "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6"},
        "p10": {"correlation_length_cm1": 20, "memory_budget_bytes": 68719476736, "peak_estimate_formula": "32*N^2+64*N+2^30"},
        "model_recipe": {
            "projection": "float64_interpolation_then_one_float32_cast_no_normalization",
            "pca": {"n_components": 20, "random_state": "model_seed", "svd_solver": "randomized", "whiten": False},
            "logistic_regression": {"c_grid": [0.01, 0.1, 1.0, 10.0], "class_weight": None, "max_iter": 1000, "random_state": "model_seed", "regularization": "l2", "solver": "lbfgs", "tol": 0.0001},
            "selection": "first_strict_maximum_validation_top1_accuracy_lowest_c_on_tie", "refit_with_validation": False, "test_condition_count": 41,
        },
        "inference": {"bootstrap_resamples": 2000, "sign_flip_resamples": 100000, "random_seed": 20260817, "confidence_level": 0.95, "holm_alpha": 0.05, "cluster_unit": "class_label", "d_ag_formula": "AG_MSE-AG_candidate", "d_acc_formula": "Acc_candidate-Acc_MSE", "class_contribution_reduction": {"D_AG": "sum", "D_Acc": "mean"}, "holm_slot_count_per_shot": 24, "shot_families_are_separate": True},
        "figure_contract": {"colors": {"p08": "#1f77b4", "p09": "#ff7f0e", "p10": "#2ca02c", "p11": "#d62728", "p12": "#9467bd"}, "dpi": 300, "figure1_inches": [12, 12], "figure1_layout": [4, 4], "figure2_inches": [14, 8], "figure2_layout": [1, 4], "font_family": "DejaVu Sans", "line_width": 1.5, "marker": "o", "padding_fraction": 0.05, "svg_hashsalt_pattern": "rpe-phase4-d2-{shot_count}shot-v1"},
        "artifact_contract": {"payload_files": list(PAYLOADS), "terminal_markers": ["complete.json", "failed.json"], "terminal_marker_rule": "exactly_one", "checksum_file": "SHA256SUMS", "checksum_scope": "all_payloads_and_exactly_one_terminal_marker"},
        "artifact_payload_files": list(PAYLOADS), "trust_anchor": {"direction": "authority_to_config_only", "config_binds_authority": False},
    }
    for name, value in required.items():
        if root.get(name) != value:
            raise Phase4D2ProtocolAVerifierError(f"config: frozen {name} authority mismatch")

    authority_specs = {
        "parent_plan": ("raman_preproc_benchmark_plan_v2.md", 46363, "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5"),
        "phase4_preregistration": ("reports/phase4/step01_phase4_feasibility_preregistration.md", 31265, "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f"),
        "step12_design": ("reports/phase4/step12_d2_protocol_a_full_domain_eligibility_design.md", 21066, "eeb2350ce6c359d84d9f1372962b8fcfb7fef7f623a77af18ae2e0abd03a4b38"),
        "step13_report": ("reports/phase4/step13_d2_protocol_a_eligibility_preflight.md", 13592, "ab06d47e2e18301c0c1bfb53aef57284fd616b086a9b05cb552869a826d155d9"),
        "step13_config": ("experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json", 23129, STEP13_CONFIG),
        "sweep": ("experiments/shared/raman_perturbation_sweep_v1.json", 559, "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"),
        "phase1_core_config": ("experiments/phase1/configs/rruff_raw_core10k_v1.json", 2350, "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd"),
        "d2_selection_config": ("experiments/phase05/configs/d2_few_shot_selection.json", 858, "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"),
        "d2_selection_artifact": ("results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json", 254678, "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36"),
        "d2_phase05_runner_config": ("experiments/phase05/configs/d2_bacteria_id_pca20_lr_sg11.json", 814, "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7"),
        "pca_lr_recipe_config": ("experiments/phase05/configs/d1_bacteria_id_pca20_lr_sg11.json", 1051, "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"),
        "classical_catalog": ("experiments/phase3/configs/classical_system_catalog_v1.json", 376665, "8ad40b08df78b8905d75a67c84a2bb328531ef17cb12704ffad04f0a8f925d8f"),
        "step13_sha256sums": (f"results/phase4/d2_protocol_a_full_domain_eligibility_v1/{STEP13_RUN_ID}/SHA256SUMS", 1094, STEP13_SUMS),
        "bacteria_id_retained_snapshot": ("data/unified/bacteria_id_reference/SHA256SUMS.sha256", 77, "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02"),
        "bacteria_id_sha256sums": ("data/unified/bacteria_id_reference/SHA256SUMS", 235, "6d6d1399ac0e5197a9e51a1a0cf32edf51c924ce8d7c12aeb9d58f9af93c7f3e"),
    }
    expected_authorities = {name: {"path": path, "bytes": size, "sha256": digest} for name, (path, size, digest) in authority_specs.items()}
    if root.get("authorities") != expected_authorities:
        raise Phase4D2ProtocolAVerifierError("config: frozen scientific authorities mismatch")
    for path, size, digest in authority_specs.values():
        file = ROOT / path
        if file.exists() and (file.stat().st_size != size or _sha_file(file) != digest):
            raise Phase4D2ProtocolAVerifierError(f"config: live authority mismatch for {path}")

    payloads = {"cells.jsonl": "6e0241d9eab3e0f18f5de49706d70580bf71484f5468bc15116567843a0f6519", "class_summaries.jsonl": "ece89e8b548997fcd14fc2ed28d88d5767373a60d871caf04f582cb30fff0c80", "common_support.jsonl": "ec668c3e2aff9efaf9ad8ba14b2f8fd4ed20cb8a97394d3269ca1f258c5106e4", "config.json": STEP13_CONFIG, "cwt_receipts.jsonl": "38f082973fc7b40b116840cf1deadd518e574439af3d9ff11bee194b7676e25b", "failed.json": "e6b1ae6f6a1df48558b60e91a6d993cd9a58d1367e9e0be3c33bc3aa5e555fa9", "gate.json": "da68f7309416b9a6a40703b16be85c78183f46ff92733de11c4aa7997ae904f5", "manifest.json": "e8e584861259e1f9669b2cdaa2f2de55a81215f41c62d08cb9dec090520b77bd", "metric_statuses.jsonl": "2f9e085485b2eed18be828191c079156bc5d6038d62ed4dbfe243e3db6d3b7c4", "model_cells.jsonl": "950684c03ddbbd60857790cd40c4062c8f8c20ad743f447f5ce25996a02577d1", "model_role_occurrences.jsonl": "90c72e8e8d4f47a1a7d610cbbe82abaabb693a5dc20f6dd852840e73ea834d95", "record_conditions.jsonl": "3b57075e7cf0c941dfd7c2380375a51d86c9e746cac4a792720d391530e04bd0", "source_records.jsonl": "dfe10be4f77c0905c9f9108d0ea886d4cf5435fef4822b86d9f26c1971a570a0"}
    relative_parent = f"results/phase4/d2_protocol_a_full_domain_eligibility_v1/{STEP13_RUN_ID}"
    parent = {"relative_path": relative_parent, "run_id": STEP13_RUN_ID, "config_sha256": STEP13_CONFIG, "sha256sums_sha256": STEP13_SUMS, "payload_sha256": payloads}
    if root.get("step13_parent") != parent:
        raise Phase4D2ProtocolAVerifierError("config: frozen Step13 parent authority mismatch")
    parent_path = ROOT / relative_parent
    if parent_path.exists() and any(_sha_file(parent_path / name) != digest for name, digest in payloads.items()):
        raise Phase4D2ProtocolAVerifierError("config: live Step13 parent payload mismatch")
    identities = {"dataset_id": "bacteria_id_reference", "model_seed_ids": list(SEEDS), "native_axis_decreasing_f32_sha256": "dccc386b7fed68fc8302a1cb0f44f6486765c13efab7538d407cbd2c77b85b96", "native_axis_id": "91e468d92cd4215f23c6c785b4611dc6f1ffe39cbb9134d11c60345954f80378", "native_axis_increasing_f64_sha256": "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d", "selection_test_record_ids_sha256": "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd", "support_axis_f32_sha256": "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59", "support_axis_f64_sha256": "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6", "support_point_count": 997, "test_record_ids_sha256": "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"}
    if root.get("frozen_identities") != identities:
        raise Phase4D2ProtocolAVerifierError("config: frozen dataset/support/test identities mismatch")
    if root.get("environment_authority") != _observed_environment():
        raise Phase4D2ProtocolAVerifierError("config: environment authority mismatch")
    code = root.get("code_authority")
    if not isinstance(code, Mapping) or tuple(code) != CODE_PATHS or code != _observed_code():
        raise Phase4D2ProtocolAVerifierError("config: code authority mismatch")


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _csv(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


def _conditions() -> tuple[tuple[str, str | None, float], ...]:
    return (("alpha0", None, 0.0),) + tuple(
        (f"{perturbation}:{np.float64(alpha).tobytes().hex()}", perturbation, alpha)
        for perturbation in PERTURBATIONS for alpha in ALPHAS[1:]
    )


def _load_config(path: Path, *, require_frozen_identity: bool) -> _Config:
    raw = Path(path).read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D2ProtocolAVerifierError(f"config: {error}") from error
    if raw != _canonical(document):
        raise Phase4D2ProtocolAVerifierError("config: canonical JSON required")
    synthetic = bool(document.get("synthetic_fixture", False))
    if require_frozen_identity and (synthetic or len(raw) != CONFIG_BYTES or _sha_bytes(raw) != CONFIG_SHA256):
        raise Phase4D2ProtocolAVerifierError("config: frozen config identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D2ProtocolAVerifierError("config: schema identity mismatch")
    if tuple(document.get("shot_counts", ())) != SHOTS or tuple(document.get("active_perturbation_ids", ())) != PERTURBATIONS:
        raise Phase4D2ProtocolAVerifierError("config: endpoint/perturbation order mismatch")
    if tuple(float(item) for item in document.get("alpha_grid", ())) != ALPHAS or tuple(document.get("metric_output_ids", ())) != METRICS:
        raise Phase4D2ProtocolAVerifierError("config: alpha/metric manifest mismatch")
    if tuple(document.get("artifact_payload_files", ())) != PAYLOADS:
        raise Phase4D2ProtocolAVerifierError("config: payload inventory mismatch")
    denominator = document.get("denominators", {})
    expected = document.get("expected", {})
    if not isinstance(denominator, Mapping) or not isinstance(expected, Mapping):
        raise Phase4D2ProtocolAVerifierError("config: denominators and expected must be objects")
    classes, tests = int(denominator.get("class_count", 0)), int(denominator.get("test_record_count", 0))
    seeds = tuple(int(item) for item in document.get("model_seeds", SEEDS))
    required = {
        "condition_count": tests * 41, "metric_value_count": tests * 41 * 13,
        "cwt_receipt_count": tests * 41, "model_cell_count": 3 * len(seeds),
        "lr_candidate_fit_count": 12 * len(seeds), "prediction_row_count": 3 * len(seeds) * tests * 41,
        "seed_class_condition_count": 3 * len(seeds) * classes * 41,
    }
    if classes < 1 or tests < classes or not seeds or any(int(expected.get(key, -1)) != value for key, value in required.items()):
        raise Phase4D2ProtocolAVerifierError("config: frozen denominators mismatch")
    if not synthetic:
        _verify_frozen_config_contract(document)
    return _Config(Path(path), raw, _sha_bytes(raw), synthetic, classes, tests, seeds, int(document.get("support_point_count", 997)), MappingProxyType(document))


def _coerce_inputs(value: object, config: _Config) -> _Inputs:
    try:
        output = _Inputs(
            np.ascontiguousarray(np.asarray(value.test_values)), np.asarray(value.test_labels, dtype="<i8"),
            tuple(str(item) for item in value.record_ids), tuple(int(item) for item in value.model_seeds),
            getattr(value, "train_values", None), getattr(value, "train_labels", None),
            getattr(value, "validation_values", None), getattr(value, "validation_labels", None),
            getattr(value, "model_cells", None),
            getattr(value, "support_axis_cm1", None), tuple(getattr(value, "native_test_spectra", ())), getattr(value, "native_test_labels", None),
        )
    except AttributeError as error:
        raise Phase4D2ProtocolAVerifierError("inputs: missing D2 input contract fields") from error
    if output.test_values.ndim != 2 or output.test_values.shape != (len(output.record_ids), config.support_points):
        raise Phase4D2ProtocolAVerifierError("inputs: projected test matrix shape mismatch")
    if output.test_labels.shape != (len(output.record_ids),) or output.model_seeds != config.seeds:
        raise Phase4D2ProtocolAVerifierError("inputs: label/seed identity mismatch")
    return output


def _fit_models(inputs: _Inputs, config: _Config) -> tuple[_Model, ...]:
    models: list[_Model] = []
    for shot in SHOTS:
        for seed in config.seeds:
            key = (shot, seed)
            if inputs.train_values is None:
                train_index = np.arange(0, len(inputs.test_labels), 2, dtype=int)
                valid_index = np.arange(1, len(inputs.test_labels), 2, dtype=int)
                if len(np.unique(inputs.test_labels[train_index])) < 2 or valid_index.size == 0:
                    train_index = valid_index = np.arange(len(inputs.test_labels), dtype=int)
                train, train_y = inputs.test_values[train_index], inputs.test_labels[train_index]
                valid, valid_y = inputs.test_values[valid_index], inputs.test_labels[valid_index]
            else:
                train, train_y = np.asarray(inputs.train_values[key], dtype="<f4"), np.asarray(inputs.train_labels[key], dtype="<i8")
                valid, valid_y = np.asarray(inputs.validation_values[key], dtype="<f4"), np.asarray(inputs.validation_labels[key], dtype="<i8")
                if train.shape != (shot * config.class_count, config.support_points) or valid.shape != (300, config.support_points):
                    raise Phase4D2ProtocolAVerifierError("model: exact role cardinality mismatch")
                if set(train_y.tolist()) != set(range(config.class_count)) or set(valid_y.tolist()) != set(range(config.class_count)):
                    raise Phase4D2ProtocolAVerifierError("model: missing class")
            pca = PCA(n_components=min(20, train.shape[0], train.shape[1]), svd_solver="randomized", whiten=False, random_state=seed)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                train_x, valid_x = pca.fit_transform(train), pca.transform(valid)
            if observed or not np.isfinite(train_x).all() or not np.isfinite(valid_x).all():
                raise Phase4D2ProtocolAVerifierError("model: PCA warning/nonfinite failure")
            selected: LogisticRegression | None = None; selected_c = 0.0; best = -math.inf; scores: list[Mapping[str, object]] = []
            for c in (0.01, 0.1, 1.0, 10.0):
                candidate = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, tol=1e-4, class_weight=None, random_state=seed)
                with warnings.catch_warnings(record=True) as observed:
                    warnings.simplefilter("always"); candidate.fit(train_x, train_y)
                predicted = candidate.predict(valid_x)
                if observed or predicted.shape != valid_y.shape or not np.isfinite(candidate.coef_).all():
                    raise Phase4D2ProtocolAVerifierError("model: LR warning/nonfinite failure")
                score = float(np.mean(predicted == valid_y)); scores.append({"shot_count": shot, "model_seed": seed, "c": c, "validation_top1_accuracy": score})
                if score > best:
                    selected, selected_c, best = candidate, c, score
            assert selected is not None
            models.append(_Model(shot, seed, selected_c, pca, selected, tuple(scores), _array_sha(train, "<f4"), _array_sha(valid, "<f4"), _array_sha(train_x), _array_sha(valid_x)))
    return tuple(models)


def _receipt_sha(value: object) -> str:
    return _sha_bytes(_step13_canonical(value))


def _source(order: int, record_id: str, label: int, spectrum: Spectrum1D) -> Phase1Source:
    return Phase1Source(
        selection=SelectedSourceRow(order, record_id, spectrum.sample_id or record_id, label, f"bacteria-{label}", f"native::{_array_sha(spectrum.axis_cm1)}"),
        spectrum=spectrum, original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(spectrum.axis_cm1, "<f4"), source_intensity_float32_sha256=_array_sha(spectrum.intensity, "<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1), normalized_intensity_float64_sha256=_array_sha(spectrum.intensity),
        provenance=MappingProxyType({"license": None, "license_status": "not_stated", "retrieved_date": "2026-08-21", "sha256": "0" * 64, "source_artifact": "bacteria_id_reference", "source_url": "local://bacteria_id_reference"}),
    )


def _metric_objects() -> Mapping[str, object]:
    return MappingProxyType({"mse": MSEMetric(), "rmse": RMSEMetric(), "mae": MAEMetric(), "sam": SAMMetric(), "pearson_r": PearsonRMetric(), "nmse": NMSEMetric(), "wasserstein_1_cm1": Wasserstein1Metric(), "is_like_structure_to_noise": ISLikeStructureToNoiseMetric()})


def _projection(spectrum: Spectrum1D, support: np.ndarray) -> np.ndarray:
    axis, intensity = np.asarray(spectrum.axis_cm1, dtype="<f8"), np.asarray(spectrum.intensity, dtype="<f8")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.size != intensity.size or axis[0] > support[0] or axis[-1] < support[-1]:
        raise Phase4D2ProtocolAVerifierError("science: invalid support interpolation")
    return np.ascontiguousarray(np.interp(support, axis, intensity), dtype="<f4")


def _item_sha(item: object) -> str:
    result = item.result
    return _receipt_sha({"alpha_float64_le_hex": item.alpha_float64_le_hex, "axis_behavior": result.axis_behavior.value, "axis_changed": result.axis_changed, "diagnostics": result.diagnostics, "intensity_changed": result.intensity_changed, "output_axis_sha256": _array_sha(result.output.axis_cm1), "output_intensity_sha256": _array_sha(result.output.intensity), "output_spectrum_id": result.output.spectrum_id, "perturbation_id": result.perturbation_id, "source_spectrum_id": result.source_spectrum_id, "state_digest": result.state_digest})


def _science_record(order: int, record_id: str, label: int, spectrum: Spectrum1D, support: np.ndarray, sweep: object, phase1: object, cwt: object, admission: P10MemoryAdmission) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    source = _source(order, record_id, label, spectrum); matrices = {"alpha0": _projection(spectrum, support)}; spectra = {"alpha0": spectrum}; hashes = {"alpha0": _array_sha(matrices["alpha0"], "<f4")}
    for perturbation in PERTURBATIONS:
        cell = run_perturbation_cell(source, perturbation, phase1, sweep, p10_admission=admission if perturbation == "p10" else None)
        if cell.status is not CellStatus.COMPLETE: raise Phase4D2ProtocolAVerifierError(f"science: {record_id}/{perturbation} native gate failure")
        alpha0 = next((item for item in cell.records if item.result.alpha == 0.0), None)
        if alpha0 is None or not np.array_equal(alpha0.result.output.axis_cm1, spectrum.axis_cm1) or not np.array_equal(alpha0.result.output.intensity, spectrum.intensity): raise Phase4D2ProtocolAVerifierError("science: alpha0 identity failure")
        for item in cell.records:
            if item.result.alpha == 0.0: continue
            condition = f"{perturbation}:{np.float64(item.result.alpha).tobytes().hex()}"; spectra[condition] = item.result.output; matrices[condition] = _projection(item.result.output, support); hashes[condition] = _item_sha(item)
    reference = run_peak_detection_system(cwt, spectrum)
    if reference.status not in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}: raise Phase4D2ProtocolAVerifierError("science: reference CWT failure")
    conditions: list[dict[str, object]] = []; metrics: list[dict[str, object]] = []; peaks: list[dict[str, object]] = []; scalar = _metric_objects(); peak_metric = PeakDetectionCurvesMetric()
    for condition, _, _ in _conditions():
        output, matrix = spectra[condition], matrices[condition]
        conditions.append({"record_order": order, "record_id": record_id, "condition_id": condition, "source_spectrum_id": spectrum.spectrum_id, "output_spectrum_id": output.spectrum_id, "axis_sha256": _array_sha(output.axis_cm1), "intensity_sha256": _array_sha(output.intensity), "projected_row_sha256": _array_sha(matrix, "<f4"), "result_sha256": hashes[condition], "state": "complete"})
        current = reference if condition == "alpha0" else run_peak_detection_system(cwt, output)
        if current.status not in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}: raise Phase4D2ProtocolAVerifierError("science: CWT failure")
        peaks.append({"record_order": order, "record_id": record_id, "condition_id": condition, "state": current.status.value, "peak_list_sha256": current.peaks_sha256, "diagnostics_sha256": _receipt_sha(dict(current.diagnostics)), "warning_sha256": _receipt_sha(current.warnings)})
        for output_id, metric in scalar.items():
            result = evaluate_metric(metric, SingleSpectrumInput(output) if output_id == "is_like_structure_to_noise" else SpectrumPairInput(spectrum, output)); value = next(item for item in result.outputs if item.output_id == output_id)
            metrics.append({"record_order": order, "record_id": record_id, "condition_id": condition, "metric_output_id": output_id, "state": "complete", "value": float(value.value), "result_sha256": _receipt_sha(result), "diagnostics_sha256": _receipt_sha(result.diagnostics)})
        structure = evaluate_metric(peak_metric, PeakPairInput(tuple(item.to_peak1d() for item in reference.peaks), tuple(item.to_peak1d() for item in current.peaks), 2.0, (0.0,))); by_id = {item.output_id: item for item in structure.outputs}
        for output_id in METRICS[8:]: metrics.append({"record_order": order, "record_id": record_id, "condition_id": condition, "metric_output_id": output_id, "state": "complete", "value": float(by_id[output_id].value), "result_sha256": _receipt_sha(structure), "diagnostics_sha256": _receipt_sha(structure.diagnostics)})
    return matrices, conditions, metrics, peaks


_WORKER_SUPPORT: np.ndarray | None = None
_WORKER_SWEEP: object | None = None
_WORKER_PHASE1: object | None = None
_WORKER_CWT: object | None = None
_WORKER_P10_BUDGET: int | None = None


def _initialize_real_science_worker(support: np.ndarray, p10_budget: int) -> None:
    """Load frozen scientific authorities in each spawn worker."""
    global _WORKER_SUPPORT, _WORKER_SWEEP, _WORKER_PHASE1, _WORKER_CWT, _WORKER_P10_BUDGET
    _WORKER_SUPPORT = np.ascontiguousarray(support, dtype="<f8")
    _WORKER_P10_BUDGET = int(p10_budget)
    _WORKER_SWEEP = load_perturbation_sweep_config(ROOT / "experiments/shared/raman_perturbation_sweep_v1.json")
    _WORKER_PHASE1 = load_phase1_core_config(ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json")
    catalog = load_classical_catalog(ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json")
    matches = [system for system in catalog.systems if system.system_id == "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"]
    if len(matches) != 1:
        raise Phase4D2ProtocolAVerifierError("science worker: CWT authority mismatch")
    _WORKER_CWT = matches[0]


def _real_science_worker(job: tuple[int, str, int, Spectrum1D]) -> tuple[int, dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if any(value is None for value in (_WORKER_SUPPORT, _WORKER_SWEEP, _WORKER_PHASE1, _WORKER_CWT, _WORKER_P10_BUDGET)):
        raise Phase4D2ProtocolAVerifierError("science worker: uninitialized")
    order, record_id, label, spectrum = job
    with threadpool_limits(limits=1, user_api="blas"):
        bundle = _science_record(
            order, record_id, label, spectrum, _WORKER_SUPPORT, _WORKER_SWEEP, _WORKER_PHASE1, _WORKER_CWT,
            P10MemoryAdmission(_WORKER_P10_BUDGET),
        )
    return order, *bundle


def _real_science(inputs: _Inputs, config: _Config, worker_count: int) -> tuple[Mapping[str, np.ndarray], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    if inputs.support_axis is None or inputs.native_labels is None or len(inputs.native_spectra) != len(inputs.record_ids) or not np.array_equal(inputs.native_labels, inputs.test_labels): raise Phase4D2ProtocolAVerifierError("science: native retained inputs required")
    support = np.asarray(inputs.support_axis, dtype="<f8")
    estimate = estimate_p10_peak_bytes(max(item.axis_cm1.size for item in inputs.native_spectra)); budget = 64 * 1024**3
    if estimate > budget: raise Phase4D2ProtocolAVerifierError("science: P10 memory admission failed")
    jobs = tuple((order, record_id, int(label), spectrum) for order, (record_id, label, spectrum) in enumerate(zip(inputs.record_ids, inputs.native_labels, inputs.native_spectra, strict=True)))
    if config.synthetic:
        sweep = load_perturbation_sweep_config(ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"); phase1 = load_phase1_core_config(ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"); catalog = load_classical_catalog(ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json")
        matches = [system for system in catalog.systems if system.system_id == "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"]
        if len(matches) != 1: raise Phase4D2ProtocolAVerifierError("science: CWT authority mismatch")
        admission = P10MemoryAdmission(budget)
        bundles = [_science_record(order, record_id, label, spectrum, support, sweep, phase1, matches[0], admission) for order, record_id, label, spectrum in jobs]
    else:
        process_count = min(worker_count, len(jobs), budget // estimate)
        with ProcessPoolExecutor(
            max_workers=process_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_real_science_worker, initargs=(support, budget),
        ) as executor:
            completed = tuple(sorted(executor.map(_real_science_worker, jobs), key=lambda row: row[0]))
        bundles = [(matrices, condition_rows, metric_rows, peak_rows) for _, matrices, condition_rows, metric_rows, peak_rows in completed]
    by_condition = {condition: [] for condition, _, _ in _conditions()}; conditions: list[Mapping[str, object]] = []; metrics: list[Mapping[str, object]] = []; peaks: list[Mapping[str, object]] = []
    for matrices, condition_rows, metric_rows, peak_rows in bundles:
        for condition, matrix in matrices.items(): by_condition[condition].append(matrix)
        conditions.extend(condition_rows); metrics.extend(metric_rows); peaks.extend(peak_rows)
    return MappingProxyType({key: np.ascontiguousarray(value, dtype="<f4") for key, value in by_condition.items()}), tuple(conditions), tuple(metrics), tuple(peaks)


def _predict(models: Sequence[_Model], matrices: Mapping[str, np.ndarray], inputs: _Inputs, config: _Config) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    predictions: list[Mapping[str, object]] = []; seed_rows: list[Mapping[str, object]] = []; cells: list[Mapping[str, object]] = []
    for model in models:
        role = (inputs.model_cells or {}).get((model.shot, model.seed), {})
        cells.append({"shot_count": model.shot, "model_seed": model.seed, "selected_c": model.selected_c, "warning_state": "none", "train_record_ids_sha256": _sha_bytes(("\n".join(role.get("train_record_ids", ())) + "\n").encode()), "validation_record_ids_sha256": _sha_bytes(("\n".join(role.get("validation_record_ids", ())) + "\n").encode()), "train_matrix_sha256": model.train_sha, "validation_matrix_sha256": model.validation_sha, "pca_train_feature_sha256": model.train_feature_sha, "pca_validation_feature_sha256": model.validation_feature_sha, "model_state_sha256": _receipt_sha({"selected_c": model.selected_c, "classes": np.asarray(model.classifier.classes_, dtype="<i8").tolist(), "coef_sha256": _array_sha(np.asarray(model.classifier.coef_, dtype="<f8")), "intercept_sha256": _array_sha(np.asarray(model.classifier.intercept_, dtype="<f8")), "n_iter": np.asarray(model.classifier.n_iter_, dtype="<i8").tolist(), "pca_components_sha256": _array_sha(np.asarray(model.pca.components_, dtype="<f8")), "pca_mean_sha256": _array_sha(np.asarray(model.pca.mean_, dtype="<f8")), "pca_explained_variance_sha256": _array_sha(np.asarray(model.pca.explained_variance_, dtype="<f8"))})})
        for condition_id, matrix in matrices.items():
            predicted = model.classifier.predict(model.pca.transform(matrix))
            for order, choice in enumerate(predicted):
                truth = int(inputs.test_labels[order])
                predictions.append({"shot_count": model.shot, "model_seed": model.seed, "model_identity": f"{model.shot}:{model.seed}", "condition_id": condition_id, "record_order": order, "record_id": inputs.record_ids[order], "true_class": truth, "predicted_class": int(choice), "correct": bool(truth == choice), "projected_row_sha256": _array_sha(matrix[order], "<f4"), "config_sha256": config.sha256})
            for label in range(config.class_count):
                selected = [row["correct"] for row in predictions if row["shot_count"] == model.shot and row["model_seed"] == model.seed and row["condition_id"] == condition_id and row["true_class"] == label]
                seed_rows.append({"shot_count": model.shot, "model_seed": model.seed, "class_label": label, "condition_id": condition_id, "accuracy": float(np.mean(selected))})
    return tuple(predictions), tuple(seed_rows), tuple(cells)


def _aggregate(predictions: Sequence[Mapping[str, object]], metrics: Sequence[Mapping[str, object]], config: _Config, *, bootstrap_resamples: int, sign_flip_resamples: int) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    expected = tuple(item[0] for item in _conditions()); positive = expected[1:]
    seed_values: dict[tuple[int, int, int, str], float] = {}
    record_class: dict[int, int] = {}
    for row in predictions:
        key = (int(row["shot_count"]), int(row["model_seed"]), int(row["true_class"]), str(row["condition_id"]))
        seed_values.setdefault(key, []).append(bool(row["correct"]))  # type: ignore[union-attr]
        record_class[int(row["record_order"])] = int(row["true_class"])
    seed_values = {key: float(np.mean(value)) for key, value in seed_values.items()}
    if {key[3] for key in seed_values} != set(expected):
        raise Phase4D2ProtocolAVerifierError("aggregation: exact canonical condition grid required")
    metric_map = {(int(row["record_order"]), str(row["condition_id"]), str(row["metric_output_id"])): float(row["value"]) for row in metrics if row.get("value") is not None}
    observations: list[Mapping[str, object]] = []; alignment: list[Mapping[str, object]] = []; bootstrap: list[Mapping[str, object]] = []; signs: list[Mapping[str, object]] = []; holm: list[Mapping[str, object]] = []
    for shot in SHOTS:
        tables: dict[str, tuple[AlignmentObservation, ...]] = {}; results: dict[str, Mapping[str, object]] = {}
        for metric in METRICS:
            table: list[AlignmentObservation] = []
            for condition_id in positive:
                perturbation, encoded = condition_id.split(":", 1); alpha = float(np.frombuffer(bytes.fromhex(encoded), dtype="<f8")[0])
                for label in range(config.class_count):
                    current = [seed_values.get((shot, seed, label, condition_id)) for seed in config.seeds]; baseline = [seed_values.get((shot, seed, label, "alpha0")) for seed in config.seeds]
                    indexes = [index for index, observed in record_class.items() if observed == label]
                    pairs = [(metric_map.get((index, "alpha0", metric)), metric_map.get((index, condition_id, metric))) for index in indexes]
                    if any(value is None for value in (*current, *baseline)) or not pairs or any(left is None or right is None for left, right in pairs):
                        raise Phase4D2ProtocolAVerifierError("aggregation: incomplete class grid")
                    harm = float(np.mean([(right - left) if DIRECTIONS[metric] == "lower_is_better" else (left - right) for left, right in pairs]))
                    downstream = float(np.mean(baseline) - np.mean(current)); item = AlignmentObservation(str(label), perturbation, alpha, harm, downstream); table.append(item)
                    observations.append({"shot_count": shot, "metric_output_id": metric, "class_label": label, "condition_id": condition_id, "perturbation_id": perturbation, "alpha": alpha, "metric_harm": harm, "downstream_harm": downstream})
            tables[metric] = tuple(table)
            try: results[metric] = {"state": "complete", "gap": alignment_gap(table), "accuracy": cross_perturbation_accuracy(table)}
            except AlignmentValidationError as error:
                if error.path != "constant downstream": raise
                results[metric] = {"state": "not_evaluable_constant_downstream", "reason": str(error)}
        reference = results["mse"]; pvalues: dict[str, float] = {}; contrasts: dict[str, float] = {}; candidate: dict[str, Mapping[str, object]] = {}
        if reference["state"] == "complete":
            mse_boot = bulk_paired_cluster_bootstrap(tables["mse"], tables["mse"], resamples=bootstrap_resamples, random_seed=20260817)
            alignment.append({"shot_count": shot, "metric_output_id": "mse", "state": "complete", "ag": reference["gap"].alignment_gap, "ag_raw": reference["gap"].raw_alignment_gap, "acc_cross": reference["accuracy"].accuracy, "ag_interval": mse_boot.reference_ag_interval, "acc_interval": mse_boot.reference_acc_interval, "d_ag": None, "d_acc": None, "d_ag_interval": None, "d_acc_interval": None})
        else:
            alignment.append({"shot_count": shot, "metric_output_id": "mse", "state": reference["state"], "ag": None, "ag_raw": None, "acc_cross": None, "ag_interval": None, "acc_interval": None, "d_ag": None, "d_acc": None, "d_ag_interval": None, "d_acc_interval": None})
        for metric in METRICS[1:]:
            if reference["state"] != "complete" or results[metric]["state"] != "complete":
                candidate[metric] = {"state": "not_evaluable_metric_incomplete"}; bootstrap.append({"shot_count": shot, "metric_output_id": metric, "state": "not_evaluable_metric_incomplete", "resamples": None})
                for stat in ("d_ag", "d_acc"): pvalues[f"{metric}:{stat}"] = 1.; signs.append({"shot_count": shot, "metric_output_id": metric, "statistic": stat, "state": "not_tested_metric_incomplete", "contrast": None, "p_value": 1., "resamples": None})
                continue
            comparison = compare_alignment(tables["mse"], tables[metric]); boot = bulk_paired_cluster_bootstrap(tables["mse"], tables[metric], resamples=bootstrap_resamples, random_seed=20260817); candidate[metric] = {"state": "complete", "comparison": comparison, "bootstrap": boot}
            bootstrap.append({"shot_count": shot, "metric_output_id": metric, "state": "complete", "resamples": bootstrap_resamples, "candidate_ag_interval": boot.candidate_ag_interval, "candidate_acc_interval": boot.candidate_acc_interval, "d_ag_interval": boot.d_ag_interval, "d_acc_interval": boot.d_acc_interval})
            for stat, contrast, contributions, aggregation in (("d_ag", comparison.d_ag, [item.value for item in comparison.ag_contribution_differences], "sum"), ("d_acc", comparison.d_acc, [item.value for item in comparison.acc_contribution_differences], "mean")):
                sign = paired_contribution_sign_flip(contributions, aggregation=aggregation, resamples=sign_flip_resamples, random_seed=20260817); pvalues[f"{metric}:{stat}"] = sign.p_value; contrasts[f"{metric}:{stat}"] = contrast
                signs.append({"shot_count": shot, "metric_output_id": metric, "statistic": stat, "state": "complete", "contrast": contrast, "p_value": sign.p_value, "resamples": sign_flip_resamples})
        for metric in METRICS[1:]:
            item = candidate[metric]
            if item["state"] == "complete":
                comp, boot = item["comparison"], item["bootstrap"]; alignment.append({"shot_count": shot, "metric_output_id": metric, "state": "complete", "ag": comp.candidate_gap.alignment_gap, "ag_raw": comp.candidate_gap.raw_alignment_gap, "acc_cross": comp.candidate_accuracy.accuracy, "ag_interval": boot.candidate_ag_interval, "acc_interval": boot.candidate_acc_interval, "d_ag": comp.d_ag, "d_acc": comp.d_acc, "d_ag_interval": boot.d_ag_interval, "d_acc_interval": boot.d_acc_interval})
            else: alignment.append({"shot_count": shot, "metric_output_id": metric, "state": item["state"], "ag": None, "ag_raw": None, "acc_cross": None, "ag_interval": None, "acc_interval": None, "d_ag": None, "d_acc": None, "d_ag_interval": None, "d_acc_interval": None})
        for result in holm_step_down(pvalues, alpha=0.05):
            metric, stat = result.hypothesis_id.split(":", 1); sign = next(item for item in signs if item["shot_count"] == shot and item["metric_output_id"] == metric and item["statistic"] == stat); favorable = bool(contrasts.get(result.hypothesis_id, 0.) > 0.)
            holm.append({"shot_count": shot, "metric_output_id": metric, "statistic": stat, "raw_p_value": result.raw_p_value, "adjusted_p_value": result.adjusted_p_value, "rank": result.rank, "family_size": result.family_size, "family_state": "complete" if sign["state"] == "complete" else "not_tested_metric_incomplete", "favorable": favorable, "rejected": bool(result.rejected and favorable)})
    return tuple(observations), tuple(alignment), tuple(bootstrap), tuple(signs), tuple(holm)


def _render_figures(observations: Sequence[Mapping[str, object]], alignment: Sequence[Mapping[str, object]], holm: Sequence[Mapping[str, object]]) -> Mapping[str, bytes]:
    payloads: dict[str, bytes] = {}
    for shot in SHOTS:
        prefix = f"d2_{shot}shot_protocol_a_full_domain"
        f1 = [{"shot_count": shot, "metric_output_id": metric, "perturbation_id": perturbation, "alpha": alpha, "mean_metric_harm": float(np.mean([row["metric_harm"] for row in observations if row["shot_count"] == shot and row["metric_output_id"] == metric and row["perturbation_id"] == perturbation and row["alpha"] == alpha])), "mean_downstream_harm": float(np.mean([row["downstream_harm"] for row in observations if row["shot_count"] == shot and row["metric_output_id"] == metric and row["perturbation_id"] == perturbation and row["alpha"] == alpha])), "metric_state": next(row["state"] for row in alignment if row["shot_count"] == shot and row["metric_output_id"] == metric)} for metric in METRICS for perturbation in PERTURBATIONS for alpha in ALPHAS[1:]]
        f2 = [row for row in alignment if row["shot_count"] == shot]
        family = {(row["metric_output_id"], row["statistic"]): row for row in holm if row["shot_count"] == shot}
        fields = ("shot_count", "metric_output_id", "state", "ag", "ag_raw", "ag_interval", "acc_cross", "acc_interval", "d_ag", "d_ag_interval", "d_acc", "d_acc_interval", "d_ag_raw_p", "d_ag_adjusted_p", "d_ag_rank", "d_ag_favorable", "d_ag_rejected", "d_acc_raw_p", "d_acc_adjusted_p", "d_acc_rank", "d_acc_favorable", "d_acc_rejected")
        f2 = [{**row, **{f"d_ag_{name}": family.get((row["metric_output_id"], "d_ag"), {}).get(source) for name, source in (("raw_p", "raw_p_value"), ("adjusted_p", "adjusted_p_value"), ("rank", "rank"), ("favorable", "favorable"), ("rejected", "rejected"))}, **{f"d_acc_{name}": family.get((row["metric_output_id"], "d_acc"), {}).get(source) for name, source in (("raw_p", "raw_p_value"), ("adjusted_p", "adjusted_p_value"), ("rank", "rank"), ("favorable", "favorable"), ("rejected", "rejected"))}} for row in f2]
        payloads[f"figure1_{prefix}_data.csv"] = _csv(f1, ("shot_count", "metric_output_id", "perturbation_id", "alpha", "mean_metric_harm", "mean_downstream_harm", "metric_state"))
        payloads[f"figure2_{prefix}_data.csv"] = _csv([{field: row.get(field) for field in fields} for row in f2], fields)
        plt.rcParams.update({"font.family": "DejaVu Sans", "lines.linewidth": 1.5, "svg.hashsalt": f"d2-protocol-a-{shot}-shot"})
        colors = dict(zip(PERTURBATIONS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True))
        figure, axes = plt.subplots(4, 4, figsize=(12, 12))
        for axis, metric in zip(axes.ravel(), METRICS, strict=False):
            for perturbation in PERTURBATIONS:
                rows = [row for row in f1 if row["metric_output_id"] == metric and row["perturbation_id"] == perturbation]
                axis.plot([row["mean_metric_harm"] for row in rows], [row["mean_downstream_harm"] for row in rows], marker="o", color=colors[perturbation], label=perturbation)
            axis.set_title(metric); values = [value for row in f1 if row["metric_output_id"] == metric for value in (row["mean_metric_harm"], row["mean_downstream_harm"])]
            if values: pad = max(1e-12, (max(values) - min(values)) * 0.05); axis.set_xlim(min(values) - pad, max(values) + pad); axis.set_ylim(min(values) - pad, max(values) + pad)
        for axis in axes.ravel()[13:]: axis.set_axis_off()
        figure.tight_layout(); png, svg = io.BytesIO(), io.BytesIO(); figure.savefig(png, format="png", dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"}); figure.savefig(svg, format="svg", metadata={"Date": None, "Creator": "raman-preproc-eval"}); plt.close(figure)
        payloads[f"figure1_{prefix}.png"], payloads[f"figure1_{prefix}.svg"] = png.getvalue(), svg.getvalue()
        figure, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        for axis, field, title, color in zip(axes, ("ag", "acc_cross", "d_ag", "d_acc"), ("AG", "Acc-cross", "D_AG", "D_Acc"), ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"), strict=True):
            values = [row.get(field) for row in f2]; ypos = np.arange(len(METRICS)); colors_by_state = [color if value is not None else "#bdbdbd" for value in values]; errors = [0.0 if row.get(field + "_interval") is None else max(abs(value - row[field + "_interval"][0]), abs(row[field + "_interval"][1] - value)) for row, value in zip(f2, values)]
            axis.barh(ypos, [0.0 if value is None else value for value in values], xerr=errors, color=colors_by_state); axis.set_title(title); axis.set_yticks(ypos, METRICS if axis is axes[0] else [])
            for index, row in enumerate(f2):
                if values[index] is None: axis.text(0.0, index, "ineligible", color="#666666", va="center", fontsize=7)
                elif field in {"d_ag", "d_acc"}: axis.text(values[index], index, f" p={row.get(field + '_raw_p'):.3g}/{row.get(field + '_adjusted_p'):.3g} r={row.get(field + '_rank')}", va="center", fontsize=6)
            finite = [value for value in values if value is not None]
            if finite: pad = max(1e-12, (max(finite) - min(finite)) * 0.05); axis.set_xlim(min(finite) - pad, max(finite) + pad)
        figure.tight_layout(); png, svg = io.BytesIO(), io.BytesIO(); figure.savefig(png, format="png", dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"}); figure.savefig(svg, format="svg", metadata={"Date": None, "Creator": "raman-preproc-eval"}); plt.close(figure)
        payloads[f"figure2_{prefix}.png"], payloads[f"figure2_{prefix}.svg"] = png.getvalue(), svg.getvalue()
    return payloads


def _rebuild(inputs: _Inputs, config: _Config, worker_count: int, *, bootstrap_resamples: int, sign_flip_resamples: int) -> tuple[Mapping[str, bytes], str, str]:
    matrices, conditions, metrics, peaks = _real_science(inputs, config, worker_count); models = _fit_models(inputs, config); predictions, seed_rows, cells = _predict(models, matrices, inputs, config); observations, alignment, bootstrap, signs, holm = _aggregate(predictions, metrics, config, bootstrap_resamples=bootstrap_resamples, sign_flip_resamples=sign_flip_resamples)
    rows = []
    for shot in SHOTS:
        for condition_id, _, _ in _conditions():
            selected = [row for row in predictions if row["shot_count"] == shot and row["condition_id"] == condition_id]; labels = range(config.class_count); per_class = [float(np.mean([row["correct"] for row in selected if row["true_class"] == label])) for label in labels]
            f1 = []
            for label in labels:
                tp = sum(1 for row in selected if row["true_class"] == label and row["predicted_class"] == label); fp = sum(1 for row in selected if row["true_class"] != label and row["predicted_class"] == label); fn = sum(1 for row in selected if row["true_class"] == label and row["predicted_class"] != label)
                f1.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
            rows.append({"shot_count": shot, "condition_id": condition_id, "prediction_count": len(selected), "top1_macro_class_accuracy": float(np.mean(per_class)), "top1_micro_accuracy": float(np.mean([row["correct"] for row in selected])), "macro_f1": float(np.mean(f1))})
    bridge = _eligibility_bridge(config, conditions, metrics, peaks)
    run_id = _sha_bytes(config.raw + _canonical({"records": len(inputs.record_ids), "seeds": list(config.seeds)})); states = {str(shot): next(row["state"] for row in alignment if row["shot_count"] == shot and row["metric_output_id"] == "mse") == "complete" and "complete" or "failed" for shot in SHOTS}; status = "complete" if all(value == "complete" for value in states.values()) else "failed"
    payloads: dict[str, bytes] = {"config.json": config.raw, "eligibility_bridge.json": _canonical(bridge), "model_cells.jsonl": _jsonl(cells), "validation_scores.jsonl": _jsonl([score for model in models for score in model.validation_scores]), "record_conditions.jsonl": _jsonl(conditions), "metric_values.jsonl": _jsonl(metrics), "peak_receipts.jsonl": _jsonl(peaks), "downstream_rows.jsonl": _jsonl([{key: row[key] for key in ("record_order", "record_id", "condition_id", "projected_row_sha256")} for row in conditions]), "predictions.jsonl": _jsonl([{**row, "code_identity": _sha_file(ROOT / "rpe/runner/phase4_d2_protocol_a.py")} for row in predictions]), "seed_class_conditions.jsonl": _jsonl(seed_rows), "condition_summary.csv": _csv(rows, ("shot_count", "condition_id", "prediction_count", "top1_macro_class_accuracy", "top1_micro_accuracy", "macro_f1")), "class_observations.jsonl": _jsonl(observations), "alignment_results.jsonl": _jsonl(alignment), "bootstrap_results.jsonl": _jsonl(bootstrap), "sign_flip_results.jsonl": _jsonl(signs), "holm_family.jsonl": _jsonl(holm)}
    rendered = _render_figures(observations, alignment, holm)
    for name in PAYLOADS:
        if name in rendered:
            payloads[name] = rendered[name]
    table_fields = ("shot_count", "metric_output_id", "state", "ag", "ag_raw", "ag_interval", "acc_cross", "acc_interval", "d_ag", "d_ag_interval", "d_acc", "d_acc_interval")
    payloads["d2_protocol_a_full_domain_secondary_table.csv"] = _csv([{key: row.get(key) for key in table_fields} for row in alignment], table_fields)
    payloads["manifest.json"] = _canonical({"schema_version": "phase4-d2-protocol-a-full-domain-artifact-v1", "run_id": run_id, "status": status, "shared_condition_count": len(conditions), "shared_metric_value_count": len(metrics), "shared_cwt_receipt_count": len(peaks), "model_fit_count": len(models), "lr_candidate_fit_count": len(cells) * 4, "prediction_row_count": len(predictions), "seed_class_condition_count": len(seed_rows), "shot_endpoint_states": states, "metric_states": {f"{row['shot_count']}:{row['metric_output_id']}": row["state"] for row in alignment}, "payload_files": list(PAYLOADS)})
    if tuple(payloads) != PAYLOADS: raise Phase4D2ProtocolAVerifierError("rebuild: payload order mismatch")
    marker = "complete.json" if status == "complete" else "failed.json"
    marker_bytes = _canonical({"run_id": run_id, "status": status})
    sums = b"".join(f"{_sha_bytes(payloads[name])}  {name}\n".encode() for name in PAYLOADS) + f"{_sha_bytes(marker_bytes)}  {marker}\n".encode()
    return MappingProxyType({**payloads, marker: marker_bytes, "SHA256SUMS": sums}), marker, status


def _validate_inventory(path: Path, config: _Config) -> tuple[str, Mapping[str, str]]:
    manifest = json.loads((path / "manifest.json").read_text()); status = manifest.get("status")
    marker = "complete.json" if status == "complete" else "failed.json" if status == "failed" else None
    if marker is None or {item.name for item in path.iterdir()} != set(PAYLOADS) | {marker, "SHA256SUMS"}: raise Phase4D2ProtocolAVerifierError("artifact: inventory/terminal marker mismatch")
    sums: dict[str, str] = {}
    for line in (path / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1); sums[name] = digest
    if tuple(sums) != PAYLOADS + (marker,) or any(_sha_file(path / name) != digest for name, digest in sums.items()): raise Phase4D2ProtocolAVerifierError("artifact: checksum mismatch")
    endpoints = manifest.get("shot_endpoint_states");
    if not isinstance(endpoints, Mapping) or tuple(sorted(endpoints, key=int)) != tuple(str(item) for item in SHOTS): raise Phase4D2ProtocolAVerifierError("artifact: endpoint state mismatch")
    return marker, MappingProxyType(sums)


def _compare_payloads(path: Path, rebuilt: Mapping[str, bytes]) -> None:
    candidate = {item.name for item in path.iterdir()}
    if candidate != set(PAYLOADS) | set(rebuilt).difference(PAYLOADS):
        raise Phase4D2ProtocolAVerifierError("rebuild: candidate inventory mismatch")
    for name, value in rebuilt.items():
        if (path / name).read_bytes() != value:
            raise Phase4D2ProtocolAVerifierError(f"rebuild: byte mismatch for {name}")


def verify_phase4_d2_protocol_a_from_inputs(path: Path, *, inputs: object, config_path: Path, worker_count: int, inference_resamples: int | None = None) -> Phase4D2ProtocolASummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1: raise Phase4D2ProtocolAVerifierError("worker_count must be positive")
    artifact = Path(path); config = _load_config(Path(config_path), require_frozen_identity=False); _validate_inventory(artifact, config); local_inputs = _coerce_inputs(inputs, config)
    if not config.synthetic: raise Phase4D2ProtocolAVerifierError("real input verification requires frozen retained science reconstruction")
    if inference_resamples is None: raise Phase4D2ProtocolAVerifierError("synthetic verifier requires explicit test inference_resamples")
    rebuilt, _marker, _status = _rebuild(local_inputs, config, worker_count, bootstrap_resamples=int(inference_resamples), sign_flip_resamples=int(inference_resamples))
    _compare_payloads(artifact, rebuilt)
    manifest = json.loads((artifact / "manifest.json").read_text())
    return Phase4D2ProtocolASummary(artifact, str(manifest["run_id"]), str(manifest["status"]), int(manifest["shared_condition_count"]) // 41, int(manifest["model_fit_count"]))


def _validate_step13_parent(path: Path) -> None:
    required = {"config.json", "source_records.jsonl", "model_cells.jsonl", "model_role_occurrences.jsonl", "cells.jsonl", "record_conditions.jsonl", "metric_statuses.jsonl", "cwt_receipts.jsonl", "class_summaries.jsonl", "common_support.jsonl", "gate.json", "manifest.json", "failed.json", "SHA256SUMS"}
    if path.name != STEP13_RUN_ID or {item.name for item in path.iterdir()} != required or _sha_file(path / "SHA256SUMS") != STEP13_SUMS or _sha_file(path / "config.json") != STEP13_CONFIG: raise Phase4D2ProtocolAVerifierError("Step-13 parent identity mismatch")
    gate = json.loads((path / "gate.json").read_text())
    if gate.get("full_domain_core", {}).get("state") != "evaluable" or gate.get("peak_common_support", {}).get("state") != "not_evaluable_coverage": raise Phase4D2ProtocolAVerifierError("Step-13 tier authorization mismatch")


def _eligibility_bridge(config: _Config, conditions: Sequence[Mapping[str, object]], metrics: Sequence[Mapping[str, object]], peaks: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if config.synthetic:
        return {"parent_run_id": "synthetic-step13", "step13_config_sha256": "synthetic", "step13_sha256sums_sha256": "synthetic", "full_domain_state": "evaluable", "record_condition_count": config.test_count * 41, "metric_status_count": config.test_count * 41 * 13, "cwt_receipt_count": config.test_count * 41, "checksums": {}}
    parent = ROOT / "results/phase4/d2_protocol_a_full_domain_eligibility_v1" / STEP13_RUN_ID; _validate_step13_parent(parent)
    checksums = {}
    for line in (parent / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1); checksums[name] = digest
    expected_conditions = {(row["record_id"], row["condition_id"]): row for row in (json.loads(line) for line in (parent / "record_conditions.jsonl").read_text().splitlines())}
    expected_metrics = {(row["record_id"], row["condition_id"], row["metric_output_id"]): row for row in (json.loads(line) for line in (parent / "metric_statuses.jsonl").read_text().splitlines())}
    expected_peaks = {(row["record_id"], row["condition_id"]): row for row in (json.loads(line) for line in (parent / "cwt_receipts.jsonl").read_text().splitlines())}
    if len(conditions) != 123000 or len(metrics) != 1599000 or len(peaks) != 123000: raise Phase4D2ProtocolAVerifierError("Step-13 bridge count mismatch")
    for row in conditions:
        expected = expected_conditions.get((row["record_id"], row["condition_id"]))
        if expected is None or expected.get("state") != "complete" or expected.get("support_sha256") != row["projected_row_sha256"] or expected.get("result_sha256") != row["result_sha256"] or expected.get("output_axis_sha256") != row["axis_sha256"] or expected.get("output_intensity_sha256") != row["intensity_sha256"]: raise Phase4D2ProtocolAVerifierError("Step-13 condition bridge mismatch")
    for row in metrics:
        expected = expected_metrics.get((row["record_id"], row["condition_id"], row["metric_output_id"]))
        if expected is None or expected.get("state") != row["state"] or expected.get("result_sha256") != row["result_sha256"] or expected.get("diagnostics_sha256") != row["diagnostics_sha256"]: raise Phase4D2ProtocolAVerifierError("Step-13 metric bridge mismatch")
    for row in peaks:
        expected = expected_peaks.get((row["record_id"], row["condition_id"]))
        if expected is None or expected.get("state") != row["state"] or expected.get("peak_list_sha256") != row["peak_list_sha256"] or expected.get("diagnostics_sha256") != row["diagnostics_sha256"] or expected.get("warning_sha256") != row["warning_sha256"]: raise Phase4D2ProtocolAVerifierError("Step-13 CWT bridge mismatch")
    return {"parent_path": str(parent), "parent_run_id": STEP13_RUN_ID, "step13_config_sha256": STEP13_CONFIG, "step13_sha256sums_sha256": STEP13_SUMS, "full_domain_state": "evaluable", "peak_common_state": "not_evaluable_coverage", "record_condition_count": 123000, "metric_status_count": 1599000, "cwt_receipt_count": 123000, "source_record_count": 5513, "model_cell_count": 15, "checksums": dict(sorted(checksums.items()))}


def _reconstruct_real_inputs(config: _Config) -> _Inputs:
    selection_path = ROOT / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json"; dataset_path = ROOT / "data/unified/bacteria_id_reference"
    if _sha_file(selection_path) != "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36": raise Phase4D2ProtocolAVerifierError("selection identity mismatch")
    selection = json.loads(selection_path.read_bytes())
    try: validate_d2_few_shot_selection(selection, ROOT / "experiments/phase05/configs/d2_few_shot_selection.json", dataset_path)
    except Exception as error: raise Phase4D2ProtocolAVerifierError(f"selection validation failed: {error}") from error
    support = np.asarray(json.loads((ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json").read_bytes())["support_grid"]["coordinates_cm1"], dtype="<f8")
    roles: dict[tuple[int, int], Mapping[str, object]] = {}; required: set[str] = set()
    for seed_doc in selection["selections"]:
        seed = int(seed_doc["seed"]); by_shot = {shot: [] for shot in SHOTS}; validation: list[str] = []
        for class_doc in seed_doc["classes"]:
            validation.extend(str(item) for item in class_doc["validation_record_ids"]); [by_shot[shot].extend(str(item) for item in class_doc["train_record_ids"][str(shot)]) for shot in SHOTS]
        for shot in SHOTS:
            train = tuple(by_shot[shot]); valid = tuple(validation)
            if len(train) != shot * 30 or len(valid) != 300 or set(train) & set(valid): raise Phase4D2ProtocolAVerifierError("selection role cardinality mismatch")
            roles[(shot, seed)] = {"train_record_ids": train, "validation_record_ids": valid}; required.update(train); required.update(valid)
    values: dict[str, np.ndarray] = {}; labels: dict[str, int] = {}; test_ids: list[str] = []
    with BacteriaIdBatchLoader(dataset_path, batch_size=4096) as loader:
        for batch in loader.iter_batches():
            if batch.source_split not in {"finetune", "test"}: continue
            axis = np.asarray(batch.wavenumber, dtype="<f4")[::-1].astype("<f8")
            for record_id, label, intensity in zip(batch.record_ids, batch.class_labels, batch.intensity, strict=True):
                record_id = str(record_id)
                if batch.source_split == "finetune" and record_id not in required: continue
                values[record_id] = np.ascontiguousarray(np.interp(support, axis, np.asarray(intensity, dtype="<f4")[::-1]).astype("<f4")); labels[record_id] = int(label)
                if batch.source_split == "test": test_ids.append(record_id)
    if len(test_ids) != 3000 or _sha_bytes(("\n".join(test_ids) + "\n").encode()) != str(selection["test"]["record_ids_sha256"]): raise Phase4D2ProtocolAVerifierError("test ID identity mismatch")
    native = []
    # Native spectra are reread in deterministic test order for the independent science pass.
    with BacteriaIdBatchLoader(dataset_path, batch_size=4096) as loader:
        by_id = {}
        for batch in loader.iter_batches():
            if batch.source_split != "test": continue
            axis = np.asarray(batch.wavenumber, dtype="<f4")[::-1].astype("<f8")
            for record_id, intensity in zip(batch.record_ids, batch.intensity, strict=True): by_id[str(record_id)] = Spectrum1D(spectrum_id=f"bacteria_id_reference::{record_id}", sample_id=None, axis_cm1=axis, intensity=np.asarray(intensity, dtype="<f4")[::-1].astype("<f8"))
        native = [by_id[item] for item in test_ids]
    return _Inputs(np.asarray([values[item] for item in test_ids], dtype="<f4"), np.asarray([labels[item] for item in test_ids], dtype="<i8"), tuple(test_ids), config.seeds, MappingProxyType({key: np.asarray([values[item] for item in role["train_record_ids"]], dtype="<f4") for key, role in roles.items()}), MappingProxyType({key: np.asarray([labels[item] for item in role["train_record_ids"]], dtype="<i8") for key, role in roles.items()}), MappingProxyType({key: np.asarray([values[item] for item in role["validation_record_ids"]], dtype="<f4") for key, role in roles.items()}), MappingProxyType({key: np.asarray([labels[item] for item in role["validation_record_ids"]], dtype="<i8") for key, role in roles.items()}), MappingProxyType(roles), support, tuple(native), np.asarray([labels[item] for item in test_ids], dtype="<i8"))


def verify_phase4_d2_protocol_a(path: Path, *, worker_count: int = 12) -> Phase4D2ProtocolASummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1: raise Phase4D2ProtocolAVerifierError("worker_count must be positive")
    config_path = ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json"; config = _load_config(config_path, require_frozen_identity=True); _validate_step13_parent(ROOT / "results/phase4/d2_protocol_a_full_domain_eligibility_v1" / STEP13_RUN_ID); inputs = _reconstruct_real_inputs(config)
    artifact = Path(path); _validate_inventory(artifact, config)
    inference = config.document["inference"]
    rebuilt, _marker, _status = _rebuild(inputs, config, worker_count, bootstrap_resamples=int(inference["bootstrap_resamples"]), sign_flip_resamples=int(inference["sign_flip_resamples"]))
    _compare_payloads(artifact, rebuilt)
    manifest = json.loads((artifact / "manifest.json").read_text())
    return Phase4D2ProtocolASummary(artifact, str(manifest["run_id"]), str(manifest["status"]), int(manifest["shared_condition_count"]) // 41, int(manifest["model_fit_count"]))


__all__ = ["Phase4D2ProtocolASummary", "Phase4D2ProtocolAVerifierError", "verify_phase4_d2_protocol_a", "verify_phase4_d2_protocol_a_from_inputs"]
