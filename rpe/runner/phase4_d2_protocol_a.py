"""Deterministic D2 Protocol-A full-domain outcome assembly.

This module deliberately keeps full-domain test science separate from the
shot/seed model cells.  The public build entry point refuses synthetic inputs;
the small constructor helpers are only a focused-test fixture.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import multiprocessing
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
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

from rpe.alignment import (
    AlignmentObservation, alignment_gap, bulk_paired_cluster_bootstrap,
    compare_alignment, cross_perturbation_accuracy, holm_step_down,
    paired_contribution_sign_flip,
)
from rpe.alignment.contracts import AlignmentValidationError
from threadpoolctl import threadpool_limits
from rpe.evaluation import (
    PeakPairInput, PreferredDirection, SingleSpectrumInput, SpectrumPairInput,
    evaluate_metric,
)

from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.evaluation import Spectrum1D
from rpe.methods.catalog import load_classical_catalog
from rpe.methods.classical.peaks import PeakRunStatus, run_peak_detection_system
from rpe.metrics import (
    ISLikeStructureToNoiseMetric, MAEMetric, MSEMetric, NMSEMetric,
    PeakDetectionCurvesMetric, PearsonRMetric, RMSEMetric, SAMMetric,
    Wasserstein1Metric,
)
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.d2_selection import validate_d2_few_shot_selection
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase4_d2_eligibility import (
    load_phase4_d2_eligibility_config,
    project_d2_support,
)
from rpe.runner import phase4_d2_eligibility as _step13_science

from rpe.runner.phase4_d2_protocol_a_authority import CONFIG_BYTES, CONFIG_SHA256


SCHEMA_VERSION = "phase4-d2-protocol-a-full-domain-config-v1"
EXPERIMENT_ID = "phase4-d2-protocol-a-full-domain-v1"
SHOTS = (5, 10, 20)
SEEDS = (0, 1, 2, 3, 4)
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
METRIC_OUTPUT_IDS = (
    "mse", "rmse", "mae", "sam", "pearson_r", "nmse",
    "wasserstein_1_cm1", "is_like_structure_to_noise", "precision",
    "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio",
)
METRIC_DIRECTIONS = MappingProxyType({
    **{name: "lower_is_better" for name in ("mse", "rmse", "mae", "sam", "nmse", "wasserstein_1_cm1", "artifact_peak_ratio", "missing_peak_ratio")},
    **{name: "higher_is_better" for name in ("pearson_r", "is_like_structure_to_noise", "precision", "recall", "f1")},
})
ARTIFACT_PAYLOAD_FILES = (
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
FORBIDDEN_PHASE05_NAMES = frozenset({"complete_cells.json", "failed.json", "complete.json"})
STEP13_RUN_ID = (
    "phase4-d2-protocol-a-full-domain-eligibility-"
    "0f43f329bfc6a7a1ff7232a336d115849b06dbca884de28d71f6b28502598ef8"
)
STEP13_SHA256SUMS_SHA256 = "b41638b4e246326b155ef1c6169b57d93c68266b47a41b9d16f05dc516546e9e"
STEP13_CONFIG_SHA256 = "f92427c2f18ab445db2bb54d5ea5a97cce21b05dd6ee80f50f3ed4082285a15b"
ROOT = Path(__file__).resolve().parents[2]
_P10_MEMORY_BUDGET_BYTES = 64 * 1024**3
_CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
_CLAIM_BOUNDARY = "local_execution_artifact_redistribution_not_cleared"
_CODE_RELATIVE_PATHS = (
    "rpe/alignment/bulk.py", "rpe/alignment/contracts.py", "rpe/alignment/core.py",
    "rpe/alignment/inference.py", "rpe/downstream/bacteria_id.py",
    "rpe/evaluation/contracts.py", "rpe/methods/catalog.py",
    "rpe/methods/classical/peaks.py", "rpe/metrics/fidelity.py",
    "rpe/metrics/peak.py", "rpe/metrics/reference_free.py", "rpe/metrics/transport.py",
    "rpe/perturb/axis_transform.py", "rpe/perturb/baseline_distortion.py",
    "rpe/perturb/contracts.py", "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py", "rpe/perturb/sweep.py",
    "rpe/runner/d2_selection.py", "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_gates.py", "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py", "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d2_eligibility.py", "rpe/runner/phase4_d2_eligibility_verifier.py",
    "rpe/runner/phase4_d2_protocol_a.py", "rpe/runner/phase4_d2_protocol_a_verifier.py",
    "tools/run_phase4_d2_protocol_a.py",
)


class Phase4D2ProtocolAError(ValueError):
    pass


@dataclass(frozen=True)
class Phase4D2ProtocolAConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    synthetic_fixture: bool
    class_count: int
    test_record_count: int
    model_seeds: tuple[int, ...]
    support_point_count: int
    artifact_payload_files: tuple[str, ...]
    document: Mapping[str, object]


@dataclass(frozen=True)
class D2ProtocolAInputs:
    test_values: np.ndarray
    test_labels: np.ndarray
    record_ids: tuple[str, ...]
    model_seeds: tuple[int, ...]
    train_values: Mapping[tuple[int, int], np.ndarray] | None = None
    train_labels: Mapping[tuple[int, int], np.ndarray] | None = None
    validation_values: Mapping[tuple[int, int], np.ndarray] | None = None
    validation_labels: Mapping[tuple[int, int], np.ndarray] | None = None
    model_cells: Mapping[tuple[int, int], Mapping[str, object]] | None = None
    support_axis_cm1: np.ndarray | None = None
    # Native test records are the scientific source of truth.  The projected
    # test matrix is deliberately retained separately for model inference.
    native_test_spectra: tuple[Spectrum1D, ...] = ()
    native_test_labels: np.ndarray | None = None


@dataclass(frozen=True)
class D2FrozenModel:
    shot_count: int
    model_seed: int
    selected_c: float
    model: object
    validation_scores: tuple[Mapping[str, object], ...] = ()
    train_matrix_sha256: str = ""
    validation_matrix_sha256: str = ""
    pca_train_feature_sha256: str = ""
    pca_validation_feature_sha256: str = ""


@dataclass(frozen=True)
class Phase4D2ProtocolASummary:
    path: Path
    run_id: str
    status: str
    test_record_count: int
    model_cell_count: int


@dataclass(frozen=True)
class D2OutcomeProjection:
    class_observations: tuple[Mapping[str, object], ...]
    alignment_results: tuple[Mapping[str, object], ...]
    bootstrap_results: tuple[Mapping[str, object], ...]
    sign_flip_results: tuple[Mapping[str, object], ...]
    holm_family: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class D2SharedScience:
    """The one unmultiplied test-condition science ledger."""
    condition_matrices: Mapping[str, np.ndarray]
    record_conditions: tuple[Mapping[str, object], ...]
    metric_values: tuple[Mapping[str, object], ...]
    peak_receipts: tuple[Mapping[str, object], ...]


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n").encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _live_environment() -> dict[str, str]:
    return {
        "h5py": h5py.__version__, "machine": platform.machine(),
        "matplotlib": matplotlib.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__, "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _live_code_authority() -> dict[str, dict[str, object]]:
    return {
        relative: {"bytes": (ROOT / relative).stat().st_size, "sha256": _sha((ROOT / relative).read_bytes())}
        for relative in _CODE_RELATIVE_PATHS
    }


def _validate_real_config_authority(document: Mapping[str, object]) -> None:
    """Fail closed over the complete Step-14 config freeze."""
    metric_manifest = [
        {"output_id": output_id, "preferred_direction": METRIC_DIRECTIONS[output_id]}
        for output_id in METRIC_OUTPUT_IDS
    ]
    exact = {
        "protocol": "A", "tier": "full_domain_core", "claim_boundary": _CLAIM_BOUNDARY,
        "shot_counts": list(SHOTS), "model_seeds": list(SEEDS),
        "active_perturbation_ids": list(PERTURBATIONS), "alpha_grid": list(ALPHAS),
        "metric_output_ids": list(METRIC_OUTPUT_IDS), "metric_manifest": metric_manifest,
        "cwt_system_id": _CWT_SYSTEM_ID,
        "denominators": {
            "class_count": 30, "test_record_count": 3000, "source_record_count": 5513,
            "model_cell_count": 15, "model_role_occurrence_count": 54750,
        },
        "expected": {
            "operator_cell_count": 15000, "apply_check_count": 135000,
            "condition_count": 123000, "metric_value_count": 1599000,
            "cwt_receipt_count": 123000, "model_cell_count": 15, "pca_fit_count": 15,
            "lr_candidate_fit_count": 60, "frozen_model_condition_call_count": 615,
            "prediction_row_count": 1845000, "seed_class_condition_count": 18450,
            "class_observation_count": 46800, "alignment_result_count": 39,
            "bootstrap_result_count": 36, "sign_flip_result_count": 72,
            "holm_family_row_count": 72, "figure1_row_count_per_shot": 520,
            "figure2_row_count_per_shot": 13, "secondary_table_row_count": 39,
            "cross_perturbation_pair_count_per_class": 640,
            "cross_perturbation_pair_count_per_shot_metric": 19200,
            "configured_payload_count": 36, "terminal_marker_count": 1,
            "sha256sums_count": 1, "artifact_file_count": 38,
        },
        "support_grid": {
            "point_count": 997, "max_in_range_native_gap_cm1": 1.561,
            "support_axis_f32_sha256": "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59",
            "support_axis_f64_sha256": "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6",
        },
        "p10": {"correlation_length_cm1": 20, "memory_budget_bytes": 68719476736, "peak_estimate_formula": "32*N^2+64*N+2^30"},
        "model_recipe": {
            "projection": "float64_interpolation_then_one_float32_cast_no_normalization",
            "pca": {"n_components": 20, "random_state": "model_seed", "svd_solver": "randomized", "whiten": False},
            "logistic_regression": {"c_grid": [0.01, 0.1, 1.0, 10.0], "class_weight": None, "max_iter": 1000, "random_state": "model_seed", "regularization": "l2", "solver": "lbfgs", "tol": 0.0001},
            "selection": "first_strict_maximum_validation_top1_accuracy_lowest_c_on_tie",
            "refit_with_validation": False, "test_condition_count": 41,
        },
        "inference": {
            "bootstrap_resamples": 2000, "sign_flip_resamples": 100000,
            "random_seed": 20260817, "confidence_level": 0.95, "holm_alpha": 0.05,
            "cluster_unit": "class_label", "d_ag_formula": "AG_MSE-AG_candidate",
            "d_acc_formula": "Acc_candidate-Acc_MSE",
            "class_contribution_reduction": {"D_AG": "sum", "D_Acc": "mean"},
            "holm_slot_count_per_shot": 24, "shot_families_are_separate": True,
        },
        "figure_contract": {
            "colors": {"p08": "#1f77b4", "p09": "#ff7f0e", "p10": "#2ca02c", "p11": "#d62728", "p12": "#9467bd"},
            "dpi": 300, "figure1_inches": [12, 12], "figure1_layout": [4, 4],
            "figure2_inches": [14, 8], "figure2_layout": [1, 4],
            "font_family": "DejaVu Sans", "line_width": 1.5, "marker": "o",
            "padding_fraction": 0.05, "svg_hashsalt_pattern": "rpe-phase4-d2-{shot_count}shot-v1",
        },
        "artifact_contract": {
            "payload_files": list(ARTIFACT_PAYLOAD_FILES),
            "terminal_markers": ["complete.json", "failed.json"],
            "terminal_marker_rule": "exactly_one", "checksum_file": "SHA256SUMS",
            "checksum_scope": "all_payloads_and_exactly_one_terminal_marker",
        },
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "trust_anchor": {"direction": "authority_to_config_only", "config_binds_authority": False},
    }
    for key, value in exact.items():
        if document.get(key) != value:
            raise Phase4D2ProtocolAError(f"config: frozen {key} authority mismatch")

    authorities = {
        "parent_plan": ("raman_preproc_benchmark_plan_v2.md", 46363, "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5"),
        "phase4_preregistration": ("reports/phase4/step01_phase4_feasibility_preregistration.md", 31265, "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f"),
        "step12_design": ("reports/phase4/step12_d2_protocol_a_full_domain_eligibility_design.md", 21066, "eeb2350ce6c359d84d9f1372962b8fcfb7fef7f623a77af18ae2e0abd03a4b38"),
        "step13_report": ("reports/phase4/step13_d2_protocol_a_eligibility_preflight.md", 13592, "ab06d47e2e18301c0c1bfb53aef57284fd616b086a9b05cb552869a826d155d9"),
        "step13_config": ("experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json", 23129, STEP13_CONFIG_SHA256),
        "sweep": ("experiments/shared/raman_perturbation_sweep_v1.json", 559, "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"),
        "phase1_core_config": ("experiments/phase1/configs/rruff_raw_core10k_v1.json", 2350, "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd"),
        "d2_selection_config": ("experiments/phase05/configs/d2_few_shot_selection.json", 858, "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"),
        "d2_selection_artifact": ("results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json", 254678, "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36"),
        "d2_phase05_runner_config": ("experiments/phase05/configs/d2_bacteria_id_pca20_lr_sg11.json", 814, "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7"),
        "pca_lr_recipe_config": ("experiments/phase05/configs/d1_bacteria_id_pca20_lr_sg11.json", 1051, "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"),
        "classical_catalog": ("experiments/phase3/configs/classical_system_catalog_v1.json", 376665, "8ad40b08df78b8905d75a67c84a2bb328531ef17cb12704ffad04f0a8f925d8f"),
        "step13_sha256sums": (f"results/phase4/d2_protocol_a_full_domain_eligibility_v1/{STEP13_RUN_ID}/SHA256SUMS", 1094, STEP13_SHA256SUMS_SHA256),
        "bacteria_id_retained_snapshot": ("data/unified/bacteria_id_reference/SHA256SUMS.sha256", 77, "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02"),
        "bacteria_id_sha256sums": ("data/unified/bacteria_id_reference/SHA256SUMS", 235, "6d6d1399ac0e5197a9e51a1a0cf32edf51c924ce8d7c12aeb9d58f9af93c7f3e"),
    }
    expected_authorities = {key: {"path": path, "bytes": size, "sha256": digest} for key, (path, size, digest) in authorities.items()}
    if document.get("authorities") != expected_authorities:
        raise Phase4D2ProtocolAError("config: frozen scientific authorities mismatch")
    for path, size, digest in authorities.values():
        live = ROOT / path
        if live.exists() and (live.stat().st_size != size or _sha(live.read_bytes()) != digest):
            raise Phase4D2ProtocolAError(f"config: live authority mismatch for {path}")

    parent_payload_sha256 = {
        "cells.jsonl": "6e0241d9eab3e0f18f5de49706d70580bf71484f5468bc15116567843a0f6519", "class_summaries.jsonl": "ece89e8b548997fcd14fc2ed28d88d5767373a60d871caf04f582cb30fff0c80",
        "common_support.jsonl": "ec668c3e2aff9efaf9ad8ba14b2f8fd4ed20cb8a97394d3269ca1f258c5106e4", "config.json": STEP13_CONFIG_SHA256,
        "cwt_receipts.jsonl": "38f082973fc7b40b116840cf1deadd518e574439af3d9ff11bee194b7676e25b", "failed.json": "e6b1ae6f6a1df48558b60e91a6d993cd9a58d1367e9e0be3c33bc3aa5e555fa9",
        "gate.json": "da68f7309416b9a6a40703b16be85c78183f46ff92733de11c4aa7997ae904f5", "manifest.json": "e8e584861259e1f9669b2cdaa2f2de55a81215f41c62d08cb9dec090520b77bd",
        "metric_statuses.jsonl": "2f9e085485b2eed18be828191c079156bc5d6038d62ed4dbfe243e3db6d3b7c4", "model_cells.jsonl": "950684c03ddbbd60857790cd40c4062c8f8c20ad743f447f5ce25996a02577d1",
        "model_role_occurrences.jsonl": "90c72e8e8d4f47a1a7d610cbbe82abaabb693a5dc20f6dd852840e73ea834d95", "record_conditions.jsonl": "3b57075e7cf0c941dfd7c2380375a51d86c9e746cac4a792720d391530e04bd0",
        "source_records.jsonl": "dfe10be4f77c0905c9f9108d0ea886d4cf5435fef4822b86d9f26c1971a570a0",
    }
    parent = {"relative_path": f"results/phase4/d2_protocol_a_full_domain_eligibility_v1/{STEP13_RUN_ID}", "run_id": STEP13_RUN_ID, "config_sha256": STEP13_CONFIG_SHA256, "sha256sums_sha256": STEP13_SHA256SUMS_SHA256, "payload_sha256": parent_payload_sha256}
    if document.get("step13_parent") != parent:
        raise Phase4D2ProtocolAError("config: frozen Step13 parent authority mismatch")
    parent_path = ROOT / parent["relative_path"]
    if parent_path.exists():
        for name, digest in parent_payload_sha256.items():
            if _sha((parent_path / name).read_bytes()) != digest:
                raise Phase4D2ProtocolAError(f"config: live Step13 parent payload mismatch for {name}")

    identities = {
        "dataset_id": "bacteria_id_reference", "model_seed_ids": list(SEEDS),
        "native_axis_decreasing_f32_sha256": "dccc386b7fed68fc8302a1cb0f44f6486765c13efab7538d407cbd2c77b85b96",
        "native_axis_id": "91e468d92cd4215f23c6c785b4611dc6f1ffe39cbb9134d11c60345954f80378",
        "native_axis_increasing_f64_sha256": "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d",
        "selection_test_record_ids_sha256": "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd",
        "support_axis_f32_sha256": "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59",
        "support_axis_f64_sha256": "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6",
        "support_point_count": 997, "test_record_ids_sha256": "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd",
    }
    if document.get("frozen_identities") != identities:
        raise Phase4D2ProtocolAError("config: frozen dataset/support/test identities mismatch")
    if document.get("environment_authority") != _live_environment():
        raise Phase4D2ProtocolAError("config: environment authority mismatch")
    code = document.get("code_authority")
    if not isinstance(code, Mapping) or tuple(code) != _CODE_RELATIVE_PATHS or code != _live_code_authority():
        raise Phase4D2ProtocolAError("config: code authority mismatch")


def _receipt_sha(value: object) -> str:
    """Use the Step-13 canonical receipt encoding for native science hashes."""
    return _sha(_step13_science._canonical_json_bytes(value))


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _csv(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _condition_specs() -> tuple[tuple[str, str | None, float], ...]:
    return (("alpha0", None, 0.0),) + tuple(
        (f"{perturbation}:{np.float64(alpha).tobytes().hex()}", perturbation, alpha)
        for perturbation in PERTURBATIONS for alpha in ALPHAS[1:]
    )


def rematerialize_d2_protocol_a_science(
    inputs: D2ProtocolAInputs, config: Phase4D2ProtocolAConfig, *, worker_count: int
) -> D2SharedScience:
    """Public full-domain science entry point.

    The synthetic path retains real matrix semantics for focused tests.  The
    retained-data path is fail-closed until it is supplied with native spectra:
    projected arrays alone cannot run frozen Phase-1 gates, P10 admission,
    metrics, and independent CWT.
    """
    if worker_count < 1:
        raise Phase4D2ProtocolAError("worker_count must be positive")
    native = inputs.native_test_spectra
    native_labels = inputs.native_test_labels
    if len(native) != len(inputs.record_ids) or native_labels is None:
        raise Phase4D2ProtocolAError("rematerialize: retained native spectra are required for real P8-P12 science")
    if len(native_labels) != len(native) or not np.array_equal(native_labels, inputs.test_labels):
        raise Phase4D2ProtocolAError("rematerialize: native label ordering mismatch")
    return _rematerialize_native_science(inputs, config, worker_count=worker_count)


def parse_phase4_d2_protocol_a_config(path: Path, raw: bytes, *, require_frozen_identity: bool) -> Phase4D2ProtocolAConfig:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D2ProtocolAError(f"config: {error}") from error
    if raw != _canonical(document):
        raise Phase4D2ProtocolAError("config: must use canonical JSON")
    if require_frozen_identity and (len(raw) != CONFIG_BYTES or _sha(raw) != CONFIG_SHA256):
        raise Phase4D2ProtocolAError("config: frozen config identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D2ProtocolAError("config: schema or experiment mismatch")
    if tuple(document.get("shot_counts", ())) != SHOTS or tuple(document.get("active_perturbation_ids", ())) != PERTURBATIONS:
        raise Phase4D2ProtocolAError("config: frozen endpoint or perturbation order mismatch")
    if tuple(float(value) for value in document.get("alpha_grid", ())) != ALPHAS:
        raise Phase4D2ProtocolAError("config: alpha grid mismatch")
    if tuple(document.get("metric_output_ids", ())) != METRIC_OUTPUT_IDS:
        raise Phase4D2ProtocolAError("config: metric manifest mismatch")
    if tuple(document.get("artifact_payload_files", ())) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D2ProtocolAError("config: artifact inventory mismatch")
    den = document.get("denominators", {})
    if not isinstance(den, Mapping):
        raise Phase4D2ProtocolAError("config: denominators must be an object")
    synthetic = bool(document.get("synthetic_fixture", False))
    if require_frozen_identity and synthetic:
        raise Phase4D2ProtocolAError("config: public load rejects synthetic fixtures")
    class_count, test_count = int(den.get("class_count", 0)), int(den.get("test_record_count", 0))
    seeds = tuple(int(value) for value in document.get("model_seeds", SEEDS))
    if class_count < 1 or test_count < class_count or not seeds:
        raise Phase4D2ProtocolAError("config: invalid model denominator")
    expected = document.get("expected", {})
    if not isinstance(expected, Mapping):
        raise Phase4D2ProtocolAError("config: expected must be an object")
    checks = {
        "condition_count": test_count * 41,
        "metric_value_count": test_count * 41 * 13,
        "cwt_receipt_count": test_count * 41,
        "model_cell_count": len(SHOTS) * len(seeds),
        "lr_candidate_fit_count": len(SHOTS) * len(seeds) * 4,
        "prediction_row_count": len(SHOTS) * len(seeds) * test_count * 41,
        "seed_class_condition_count": len(SHOTS) * len(seeds) * class_count * 41,
    }
    if any(int(expected.get(key, -1)) != value for key, value in checks.items()):
        raise Phase4D2ProtocolAError("config: shared-science or model denominator mismatch")
    if not synthetic:
        _validate_real_config_authority(document)
    return Phase4D2ProtocolAConfig(Path(path), raw, _sha(raw), synthetic, class_count, test_count, seeds, int(document.get("support_point_count", 997)), ARTIFACT_PAYLOAD_FILES, MappingProxyType(document))


def load_phase4_d2_protocol_a_config(path: Path) -> Phase4D2ProtocolAConfig:
    return parse_phase4_d2_protocol_a_config(Path(path), Path(path).read_bytes(), require_frozen_identity=True)


def make_synthetic_d2_protocol_a_inputs(*, class_count: int, records_per_class: int, model_seeds: tuple[int, ...]) -> D2ProtocolAInputs:
    if records_per_class < 2:
        raise Phase4D2ProtocolAError("synthetic inputs: each class requires at least two records")
    # Keep native coverage wider than the exact model support: real P11/P12
    # change the axis, and a fixture must exercise the same no-extrapolation
    # support gate as retained Bacteria spectra.
    axis = np.linspace(350.0, 1830.0, 1000, dtype=np.float64)
    support_axis = np.linspace(386.65, 1792.4, 997, dtype=np.float64)
    values, labels, ids, native = [], [], [], []
    for label in range(class_count):
        for index in range(records_per_class):
            # Multiple resolved positive peaks keep the true CWT and all
            # native perturbation gates meaningful in the small fixture.
            centers = (520.0, 710.0, 910.0, 1130.0, 1370.0, 1610.0)
            intensity = 0.15 + sum(
                (1.0 + 0.08 * label + 0.02 * index + 0.03 * peak)
                * np.exp(-0.5 * ((axis - center) / (13.0 + peak)) ** 2)
                for peak, center in enumerate(centers)
            )
            record_id = f"c{label:02d}-r{index:02d}"
            values.append(np.interp(support_axis, axis, intensity)); labels.append(label); ids.append(record_id)
            native.append(Spectrum1D(
                spectrum_id=f"synthetic::{record_id}", sample_id=record_id,
                axis_cm1=np.ascontiguousarray(axis), intensity=np.ascontiguousarray(intensity, dtype=np.float64),
            ))
    test_values = np.asarray(values, dtype=np.float64)
    test_labels = np.asarray(labels, dtype=np.int64)
    return D2ProtocolAInputs(
        test_values, test_labels, tuple(ids), tuple(model_seeds),
        support_axis_cm1=np.ascontiguousarray(support_axis), native_test_spectra=tuple(native),
        native_test_labels=np.asarray(test_labels, dtype=np.int64),
    )


def make_synthetic_d2_protocol_a_config(inputs: D2ProtocolAInputs) -> Phase4D2ProtocolAConfig:
    count, classes, seeds = len(inputs.record_ids), len(set(inputs.test_labels.tolist())), inputs.model_seeds
    document = {
        "schema_version": SCHEMA_VERSION, "experiment_id": EXPERIMENT_ID, "synthetic_fixture": True,
        "shot_counts": list(SHOTS), "model_seeds": list(seeds), "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS), "metric_output_ids": list(METRIC_OUTPUT_IDS), "support_point_count": inputs.test_values.shape[1],
        "denominators": {"class_count": classes, "test_record_count": count},
        "expected": {"condition_count": count * 41, "metric_value_count": count * 41 * 13, "cwt_receipt_count": count * 41, "model_cell_count": 3 * len(seeds), "lr_candidate_fit_count": 12 * len(seeds), "prediction_row_count": 3 * len(seeds) * count * 41, "seed_class_condition_count": 3 * len(seeds) * classes * 41},
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
    }
    raw = _canonical(document)
    return parse_phase4_d2_protocol_a_config(Path("<synthetic>"), raw, require_frozen_identity=False)


def reconstruct_d2_protocol_a_inputs(dataset_path: Path, selection_path: Path, config: Phase4D2ProtocolAConfig) -> D2ProtocolAInputs:
    for path in (Path(dataset_path), Path(selection_path)):
        if path.name in FORBIDDEN_PHASE05_NAMES or path.name.startswith("seed") or "prediction" in path.name or "selected_c" in path.name:
            raise Phase4D2ProtocolAError(f"{path}: forbidden Phase-0.5 outcome artifact")
    dataset_path, selection_path = Path(dataset_path), Path(selection_path)
    if not dataset_path.is_dir() or selection_path.name != "selection.json":
        raise Phase4D2ProtocolAError("D2 inputs: require retained dataset directory and frozen selection.json")
    if config.synthetic_fixture:
        raise Phase4D2ProtocolAError("D2 inputs: public reconstruction does not accept synthetic config")
    selection_raw = selection_path.read_bytes()
    if _sha(selection_raw) != "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36":
        raise Phase4D2ProtocolAError("D2 inputs: frozen selection identity mismatch")
    selection = json.loads(selection_raw)
    try:
        validate_d2_few_shot_selection(
            selection,
            Path(__file__).resolve().parents[2] / "experiments/phase05/configs/d2_few_shot_selection.json",
            dataset_path,
        )
    except Exception as error:
        raise Phase4D2ProtocolAError(f"D2 inputs: invalid frozen selection: {error}") from error
    eligibility_config = load_phase4_d2_eligibility_config(
        Path(__file__).resolve().parents[2] / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
    )
    selected: dict[tuple[int, int], Mapping[str, object]] = {}
    required_finetune: set[str] = set()
    for seed_doc in selection["selections"]:
        seed = int(seed_doc["seed"])
        by_shot = {shot: [] for shot in SHOTS}
        validation: list[str] = []
        for class_doc in seed_doc["classes"]:
            validation.extend(str(value) for value in class_doc["validation_record_ids"])
            for shot in SHOTS:
                by_shot[shot].extend(str(value) for value in class_doc["train_record_ids"][str(shot)])
        for shot in SHOTS:
            train = tuple(by_shot[shot]); valid = tuple(validation)
            if len(train) != shot * 30 or len(valid) != 300 or set(train) & set(valid):
                raise Phase4D2ProtocolAError("D2 inputs: exact train/validation role mismatch")
            selected[(shot, seed)] = {"train_record_ids": train, "validation_record_ids": valid}
            required_finetune.update(train); required_finetune.update(valid)
    if tuple(sorted(selected)) != tuple((shot, seed) for shot in SHOTS for seed in SEEDS):
        raise Phase4D2ProtocolAError("D2 inputs: must reconstruct fifteen exact model cells")
    values: dict[str, np.ndarray] = {}; labels: dict[str, int] = {}; tests: list[str] = []; native_tests: list[Spectrum1D] = []
    with BacteriaIdBatchLoader(dataset_path, batch_size=4096) as loader:
        for batch in loader.iter_batches():
            if batch.source_split not in {"finetune", "test"}:
                continue
            axis = np.ascontiguousarray(np.asarray(batch.wavenumber, dtype="<f4")[::-1], dtype="<f8")
            if not np.all(np.diff(axis) > 0.0):
                raise Phase4D2ProtocolAError("D2 inputs: native axis reversal failed")
            for record_id, label, intensity in zip(batch.record_ids, batch.class_labels, batch.intensity, strict=True):
                record_id = str(record_id)
                if batch.source_split == "finetune" and record_id not in required_finetune:
                    continue
                spectrum = Spectrum1D(
                    spectrum_id=f"bacteria_id_reference::{record_id}", sample_id=None, axis_cm1=axis,
                    intensity=np.ascontiguousarray(np.asarray(intensity, dtype="<f4")[::-1], dtype="<f8"),
                )
                values[record_id] = project_d2_support(spectrum, eligibility_config)
                labels[record_id] = int(label)
                if batch.source_split == "test":
                    tests.append(record_id)
                    native_tests.append(spectrum)
    test_digest = _sha(("\n".join(tests) + "\n").encode())
    if len(tests) != 3000 or test_digest != str(selection["test"]["record_ids_sha256"]):
        raise Phase4D2ProtocolAError("D2 inputs: exact retained test ID mismatch")
    if set(values) != required_finetune | set(tests):
        raise Phase4D2ProtocolAError("D2 inputs: selected retained source rows missing")
    train_values: dict[tuple[int, int], np.ndarray] = {}; train_labels: dict[tuple[int, int], np.ndarray] = {}
    valid_values: dict[tuple[int, int], np.ndarray] = {}; valid_labels: dict[tuple[int, int], np.ndarray] = {}
    for key, cell in selected.items():
        train_ids, valid_ids = cell["train_record_ids"], cell["validation_record_ids"]
        train_values[key] = np.ascontiguousarray([values[item] for item in train_ids], dtype="<f4")
        train_labels[key] = np.asarray([labels[item] for item in train_ids], dtype="<i8")
        valid_values[key] = np.ascontiguousarray([values[item] for item in valid_ids], dtype="<f4")
        valid_labels[key] = np.asarray([labels[item] for item in valid_ids], dtype="<i8")
    return D2ProtocolAInputs(
        test_values=np.ascontiguousarray([values[item] for item in tests], dtype="<f4"),
        test_labels=np.asarray([labels[item] for item in tests], dtype="<i8"), record_ids=tuple(tests), model_seeds=SEEDS,
        train_values=MappingProxyType(train_values), train_labels=MappingProxyType(train_labels),
        validation_values=MappingProxyType(valid_values), validation_labels=MappingProxyType(valid_labels),
        model_cells=MappingProxyType(selected), support_axis_cm1=np.asarray(eligibility_config.support_coordinates_cm1, dtype="<f8"),
        native_test_spectra=tuple(native_tests), native_test_labels=np.asarray([labels[item] for item in tests], dtype="<i8"),
    )


def validate_d2_eligibility_parent(path: Path, config: Phase4D2ProtocolAConfig) -> Mapping[str, object]:
    path = Path(path)
    if path.name != STEP13_RUN_ID:
        raise Phase4D2ProtocolAError("eligibility parent: wrong final Step-13 run")
    required = (
        "config.json", "source_records.jsonl", "model_cells.jsonl",
        "model_role_occurrences.jsonl", "cells.jsonl", "record_conditions.jsonl",
        "metric_statuses.jsonl", "cwt_receipts.jsonl", "class_summaries.jsonl",
        "common_support.jsonl", "gate.json", "manifest.json", "failed.json", "SHA256SUMS",
    )
    names = {item.name for item in path.iterdir()} if path.is_dir() else set()
    if names != set(required):
        raise Phase4D2ProtocolAError("eligibility parent: exact file inventory mismatch")
    sums_path = path / "SHA256SUMS"
    if _sha(sums_path.read_bytes()) != STEP13_SHA256SUMS_SHA256:
        raise Phase4D2ProtocolAError("eligibility parent: SHA256SUMS identity mismatch")
    checksums: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        checksums[name] = digest
    if len(checksums) != 13 or set(checksums) != set(required) - {"SHA256SUMS"}:
        raise Phase4D2ProtocolAError("eligibility parent: checksum inventory mismatch")
    for name, digest in checksums.items():
        if _sha((path / name).read_bytes()) != digest:
            raise Phase4D2ProtocolAError(f"eligibility parent: checksum mismatch for {name}")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != STEP13_RUN_ID or manifest.get("status") != "fail":
        raise Phase4D2ProtocolAError("eligibility parent: manifest identity mismatch")
    if _sha((path / "config.json").read_bytes()) != STEP13_CONFIG_SHA256:
        raise Phase4D2ProtocolAError("eligibility parent: config identity mismatch")
    failed = json.loads((path / "failed.json").read_text(encoding="utf-8"))
    if failed != {"run_id": STEP13_RUN_ID, "schema_version": "phase4-d2-protocol-a-full-domain-eligibility-marker-v1", "status": "fail"}:
        raise Phase4D2ProtocolAError("eligibility parent: failed marker mismatch")
    gate = json.loads((path / "gate.json").read_text(encoding="utf-8"))
    if gate.get("full_domain_core", {}).get("state") != "evaluable" or gate.get("peak_common_support", {}).get("state") != "not_evaluable_coverage":
        raise Phase4D2ProtocolAError("eligibility parent: tier authorization mismatch")
    for perturbation in PERTURBATIONS:
        value = gate["full_domain_core"]["by_perturbation"].get(perturbation, {})
        if value.get("state") != "evaluable" or value.get("complete_record_count") != 3000 or value.get("complete_class_count") != 30:
            raise Phase4D2ProtocolAError(f"eligibility parent: {perturbation} coverage mismatch")
    for value in (gate.get("support_grid"), gate.get("cwt"), *gate.get("metric_outputs", {}).values()):
        if not isinstance(value, Mapping) or (value.get("planned"), value.get("complete"), value.get("state")) != (123000, 123000, "evaluable"):
            raise Phase4D2ProtocolAError("eligibility parent: execution-grid mismatch")
    counts: dict[str, int] = {}
    for name in ("source_records.jsonl", "model_cells.jsonl", "record_conditions.jsonl", "metric_statuses.jsonl", "cwt_receipts.jsonl"):
        with (path / name).open(encoding="utf-8") as stream:
            counts[name] = sum(1 for _ in stream)
    if counts != {"source_records.jsonl": 5513, "model_cells.jsonl": 15, "record_conditions.jsonl": 123000, "metric_statuses.jsonl": 1599000, "cwt_receipts.jsonl": 123000}:
        raise Phase4D2ProtocolAError("eligibility parent: payload row-count mismatch")
    return MappingProxyType({
        "parent_path": str(path), "parent_run_id": STEP13_RUN_ID, "step13_config_sha256": STEP13_CONFIG_SHA256,
        "step13_sha256sums_sha256": STEP13_SHA256SUMS_SHA256, "full_domain_state": "evaluable",
        "peak_common_state": "not_evaluable_coverage", "record_condition_count": 123000,
        "metric_status_count": 1599000, "cwt_receipt_count": 123000,
        "source_record_count": 5513, "model_cell_count": 15, "checksums": dict(sorted(checksums.items())),
    })


def fit_d2_protocol_a_models(inputs: D2ProtocolAInputs, config: Phase4D2ProtocolAConfig) -> tuple[D2FrozenModel, ...]:
    models: list[D2FrozenModel] = []
    for shot in SHOTS:
        for seed in config.model_seeds:
            key = (shot, seed)
            if inputs.train_values is None:
                x = inputs.test_values; y = inputs.test_labels
                train = np.asarray([i for i in range(len(y)) if i % 2 == 0], dtype=int)
                validation = np.asarray([i for i in range(len(y)) if i % 2 == 1], dtype=int)
                if len(np.unique(y[train])) < 2 or validation.size == 0:
                    train = np.arange(len(y), dtype=int); validation = train
                train_values, train_labels = x[train], y[train]
                valid_values, valid_labels = x[validation], y[validation]
            else:
                train_values = inputs.train_values[key]; train_labels = inputs.train_labels[key]
                valid_values = inputs.validation_values[key]; valid_labels = inputs.validation_labels[key]
                if train_values.shape[0] != shot * config.class_count or valid_values.shape[0] != 300:
                    raise Phase4D2ProtocolAError("model lifecycle: frozen train/validation cardinality mismatch")
                if set(train_labels.tolist()) != set(range(config.class_count)) or set(valid_labels.tolist()) != set(range(config.class_count)):
                    raise Phase4D2ProtocolAError("model lifecycle: missing class")
            pca = PCA(
                n_components=min(20, train_values.shape[0], train_values.shape[1]),
                svd_solver="randomized",
                whiten=False,
                random_state=seed,
            )
            with warnings.catch_warnings(record=True) as observed_warnings:
                warnings.simplefilter("always")
                train_x, valid_x = pca.fit_transform(train_values), pca.transform(valid_values)
            if observed_warnings:
                raise Phase4D2ProtocolAError("model lifecycle: PCA warning")
            best_c, best_score, best_model = None, -math.inf, None
            scores: list[Mapping[str, object]] = []
            for c in (0.01, 0.1, 1.0, 10.0):
                candidate = LogisticRegression(
                    C=c,
                    solver="lbfgs",
                    max_iter=1000,
                    tol=1e-4,
                    class_weight=None,
                    random_state=seed,
                )
                with warnings.catch_warnings(record=True) as observed_warnings:
                    warnings.simplefilter("always")
                    candidate.fit(train_x, train_labels)
                if observed_warnings:
                    raise Phase4D2ProtocolAError("model lifecycle: LR warning")
                predicted = candidate.predict(valid_x)
                if not np.isfinite(train_x).all() or not np.isfinite(valid_x).all() or predicted.shape != valid_labels.shape:
                    raise Phase4D2ProtocolAError("model lifecycle: invalid model output")
                score = float(np.mean(predicted == valid_labels))
                scores.append({"shot_count": shot, "model_seed": seed, "c": c, "validation_top1_accuracy": score})
                if score > best_score:
                    best_c, best_score, best_model = c, score, candidate
            models.append(D2FrozenModel(
                shot, seed, float(best_c), (pca, best_model), tuple(scores),
                _sha(np.ascontiguousarray(train_values, dtype="<f4").tobytes()),
                _sha(np.ascontiguousarray(valid_values, dtype="<f4").tobytes()),
                _array_sha(train_x), _array_sha(valid_x),
            ))
    return tuple(models)


def _array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return _sha(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


def _phase1_source(record_order: int, record_id: str, class_label: int, spectrum: Spectrum1D) -> Phase1Source:
    return Phase1Source(
        selection=SelectedSourceRow(record_order, record_id, spectrum.sample_id or record_id, class_label, f"bacteria-{class_label}", f"native::{_array_sha(spectrum.axis_cm1)}"),
        spectrum=spectrum, original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(spectrum.axis_cm1, "<f4"),
        source_intensity_float32_sha256=_array_sha(spectrum.intensity, "<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1),
        normalized_intensity_float64_sha256=_array_sha(spectrum.intensity),
        provenance=MappingProxyType({"license": None, "license_status": "not_stated", "retrieved_date": "2026-08-21", "sha256": "0" * 64, "source_artifact": "bacteria_id_reference", "source_url": "local://bacteria_id_reference"}),
    )


def _metric_objects() -> Mapping[str, object]:
    return MappingProxyType({
        "mse": MSEMetric(), "rmse": RMSEMetric(), "mae": MAEMetric(), "sam": SAMMetric(),
        "pearson_r": PearsonRMetric(), "nmse": NMSEMetric(), "wasserstein_1_cm1": Wasserstein1Metric(),
        "is_like_structure_to_noise": ISLikeStructureToNoiseMetric(),
    })


def _support_projection(spectrum: Spectrum1D, inputs: D2ProtocolAInputs, eligibility: object | None) -> np.ndarray:
    if eligibility is not None:
        return project_d2_support(spectrum, eligibility)
    support = inputs.support_axis_cm1
    if support is None:
        raise Phase4D2ProtocolAError("rematerialize: support axis is required")
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    if axis[0] > support[0] or axis[-1] < support[-1]:
        raise Phase4D2ProtocolAError("rematerialize: synthetic support requires extrapolation")
    return np.ascontiguousarray(np.interp(support, axis, spectrum.intensity), dtype="<f4")


def _condition_id(perturbation: str, alpha: float) -> str:
    return f"{perturbation}:{np.float64(alpha).tobytes().hex()}"


def _perturbation_result_sha(item: object) -> str:
    result = item.result
    return _receipt_sha({
        "alpha_float64_le_hex": item.alpha_float64_le_hex,
        "axis_behavior": result.axis_behavior.value, "axis_changed": result.axis_changed,
        "diagnostics": result.diagnostics, "intensity_changed": result.intensity_changed,
        "output_axis_sha256": _array_sha(result.output.axis_cm1),
        "output_intensity_sha256": _array_sha(result.output.intensity),
        "output_spectrum_id": result.output.spectrum_id, "perturbation_id": result.perturbation_id,
        "source_spectrum_id": result.source_spectrum_id, "state_digest": result.state_digest,
    })


def _native_record_science(
    *, record_order: int, record_id: str, class_label: int, spectrum: Spectrum1D,
    inputs: D2ProtocolAInputs, eligibility: object | None, sweep: object, phase1_config: object, cwt_system: object, admission: P10MemoryAdmission,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    source = _phase1_source(record_order, record_id, class_label, spectrum)
    projections: dict[str, np.ndarray] = {"alpha0": _support_projection(spectrum, inputs, eligibility)}
    condition_spectra: dict[str, Spectrum1D] = {"alpha0": spectrum}
    result_hashes: dict[str, str] = {"alpha0": _array_sha(projections["alpha0"], "<f4")}
    for perturbation in PERTURBATIONS:
        cell = run_perturbation_cell(source, perturbation, phase1_config, sweep, p10_admission=admission if perturbation == "p10" else None)
        if cell.status is not CellStatus.COMPLETE:
            raise Phase4D2ProtocolAError(f"science: {record_id}/{perturbation} did not complete native gates")
        alpha0 = next((item for item in cell.records if item.result.alpha == 0.0), None)
        if alpha0 is None or not np.array_equal(alpha0.result.output.axis_cm1, spectrum.axis_cm1) or not np.array_equal(alpha0.result.output.intensity, spectrum.intensity):
            raise Phase4D2ProtocolAError(f"science: {record_id}/{perturbation} alpha0 identity failure")
        for item in cell.records:
            if item.result.alpha == 0.0:
                continue
            condition_id = _condition_id(perturbation, item.result.alpha)
            condition_spectra[condition_id] = item.result.output
            projections[condition_id] = _support_projection(item.result.output, inputs, eligibility)
            result_hashes[condition_id] = _perturbation_result_sha(item)
    scalar = _metric_objects(); peak_metric = PeakDetectionCurvesMetric()
    reference_peaks = run_peak_detection_system(cwt_system, spectrum)
    if reference_peaks.status not in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}:
        raise Phase4D2ProtocolAError(f"science: {record_id} reference CWT failed")
    conditions: list[dict[str, object]] = []; metrics: list[dict[str, object]] = []; peak_rows: list[dict[str, object]] = []
    for condition_id, _, _ in _condition_specs():
        output = condition_spectra[condition_id]; projected = projections[condition_id]
        conditions.append({"record_order": record_order, "record_id": record_id, "condition_id": condition_id, "source_spectrum_id": spectrum.spectrum_id, "output_spectrum_id": output.spectrum_id, "axis_sha256": _array_sha(output.axis_cm1), "intensity_sha256": _array_sha(output.intensity), "projected_row_sha256": _array_sha(projected, "<f4"), "result_sha256": result_hashes[condition_id], "state": "complete"})
        current_peaks = reference_peaks if condition_id == "alpha0" else run_peak_detection_system(cwt_system, output)
        if current_peaks.status not in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}:
            raise Phase4D2ProtocolAError(f"science: {record_id}/{condition_id} CWT failed")
        peak_rows.append({"record_order": record_order, "record_id": record_id, "condition_id": condition_id, "state": current_peaks.status.value, "peak_list_sha256": current_peaks.peaks_sha256, "diagnostics_sha256": _receipt_sha(dict(current_peaks.diagnostics)), "warning_sha256": _receipt_sha(current_peaks.warnings)})
        for output_id, metric in scalar.items():
            request = SingleSpectrumInput(output) if output_id == "is_like_structure_to_noise" else SpectrumPairInput(spectrum, output)
            result = evaluate_metric(metric, request)
            value = next(item for item in result.outputs if item.output_id == output_id)
            metrics.append({"record_order": record_order, "record_id": record_id, "condition_id": condition_id, "metric_output_id": output_id, "state": "complete", "value": float(value.value), "result_sha256": _receipt_sha(result), "diagnostics_sha256": _receipt_sha(result.diagnostics)})
        structure = evaluate_metric(peak_metric, PeakPairInput(tuple(item.to_peak1d() for item in reference_peaks.peaks), tuple(item.to_peak1d() for item in current_peaks.peaks), 2.0, (0.0,)))
        by_id = {item.output_id: item for item in structure.outputs}
        for output_id in METRIC_OUTPUT_IDS[8:]:
            metrics.append({"record_order": record_order, "record_id": record_id, "condition_id": condition_id, "metric_output_id": output_id, "state": "complete", "value": float(by_id[output_id].value), "result_sha256": _receipt_sha(structure), "diagnostics_sha256": _receipt_sha(structure.diagnostics)})
    return projections, conditions, metrics, peak_rows


_PROCESS_ELIGIBILITY: object | None = None
_PROCESS_SWEEP: object | None = None
_PROCESS_PHASE1_CONFIG: object | None = None
_PROCESS_CWT_SYSTEM: object | None = None


def _initialize_real_science_worker() -> None:
    global _PROCESS_ELIGIBILITY, _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG, _PROCESS_CWT_SYSTEM
    _PROCESS_ELIGIBILITY = load_phase4_d2_eligibility_config(ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json")
    _PROCESS_SWEEP = load_perturbation_sweep_config(ROOT / "experiments/shared/raman_perturbation_sweep_v1.json")
    _PROCESS_PHASE1_CONFIG = load_phase1_core_config(ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json")
    catalog = load_classical_catalog(ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json")
    matches = [system for system in catalog.systems if system.system_id == _CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D2ProtocolAError("science: CWT authority did not resolve exactly once")
    _PROCESS_CWT_SYSTEM = matches[0]


def _real_science_worker(job: tuple[int, str, int, Spectrum1D]) -> tuple[int, dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if any(value is None for value in (_PROCESS_ELIGIBILITY, _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG, _PROCESS_CWT_SYSTEM)):
        raise Phase4D2ProtocolAError("science worker was not initialized")
    order, record_id, label, spectrum = job
    placeholder = D2ProtocolAInputs(np.empty((0, 0)), np.empty((0,), dtype=np.int64), (), ())
    with threadpool_limits(limits=1, user_api="blas"):
        bundle = _native_record_science(record_order=order, record_id=record_id, class_label=label, spectrum=spectrum, inputs=placeholder, eligibility=_PROCESS_ELIGIBILITY, sweep=_PROCESS_SWEEP, phase1_config=_PROCESS_PHASE1_CONFIG, cwt_system=_PROCESS_CWT_SYSTEM, admission=P10MemoryAdmission(_P10_MEMORY_BUDGET_BYTES))
    return (order, *bundle)


def _rematerialize_native_science(inputs: D2ProtocolAInputs, config: Phase4D2ProtocolAConfig, *, worker_count: int) -> D2SharedScience:
    eligibility = None
    if not config.synthetic_fixture:
        eligibility = load_phase4_d2_eligibility_config(ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json")
    sweep = load_perturbation_sweep_config(ROOT / "experiments/shared/raman_perturbation_sweep_v1.json")
    phase1_config = load_phase1_core_config(ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json")
    catalog = load_classical_catalog(ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json")
    matches = [system for system in catalog.systems if system.system_id == _CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D2ProtocolAError("science: CWT authority did not resolve exactly once")
    estimate = estimate_p10_peak_bytes(max(spectrum.axis_cm1.size for spectrum in inputs.native_test_spectra))
    if estimate > _P10_MEMORY_BUDGET_BYTES:
        raise Phase4D2ProtocolAError("science: P10 static 64 GiB memory admission failed")
    admission = P10MemoryAdmission(_P10_MEMORY_BUDGET_BYTES)
    by_condition: dict[str, list[np.ndarray]] = {condition_id: [] for condition_id, _, _ in _condition_specs()}
    conditions: list[dict[str, object]] = []; metrics: list[dict[str, object]] = []; peaks: list[dict[str, object]] = []
    jobs = tuple((order, record_id, int(label), spectrum) for order, (record_id, label, spectrum) in enumerate(zip(inputs.record_ids, inputs.native_test_labels, inputs.native_test_spectra, strict=True)))
    for _, record_id, _, spectrum in jobs:
        if spectrum.axis_cm1.ndim != 1 or spectrum.intensity.ndim != 1 or spectrum.axis_cm1.size != spectrum.intensity.size or spectrum.axis_cm1.size < 2 or not np.isfinite(spectrum.axis_cm1).all() or not np.isfinite(spectrum.intensity).all() or not np.all(np.diff(spectrum.axis_cm1) > 0.0):
            raise Phase4D2ProtocolAError(f"science: invalid native spectrum {record_id}")
    if not config.synthetic_fixture:
        capacity = _P10_MEMORY_BUDGET_BYTES // estimate
        process_count = min(worker_count, len(jobs), capacity)
        with ProcessPoolExecutor(max_workers=process_count, mp_context=multiprocessing.get_context("spawn"), initializer=_initialize_real_science_worker) as executor:
            completed = tuple(sorted(executor.map(_real_science_worker, jobs), key=lambda row: row[0]))
        bundles = tuple((matrices, condition_rows, metric_rows, peak_rows) for _, matrices, condition_rows, metric_rows, peak_rows in completed)
    else:
        bundles = []
        for order, record_id, label, spectrum in jobs:
            bundles.append(_native_record_science(record_order=order, record_id=record_id, class_label=label, spectrum=spectrum, inputs=inputs, eligibility=eligibility, sweep=sweep, phase1_config=phase1_config, cwt_system=matches[0], admission=admission))
    for matrices, condition_rows, metric_rows, peak_rows in bundles:
        for condition_id, values in matrices.items():
            by_condition[condition_id].append(values)
        conditions.extend(condition_rows); metrics.extend(metric_rows); peaks.extend(peak_rows)
    science = D2SharedScience(MappingProxyType({key: np.ascontiguousarray(value, dtype="<f4") for key, value in by_condition.items()}), tuple(conditions), tuple(metrics), tuple(peaks))
    if not config.synthetic_fixture:
        _validate_science_against_step13(science, inputs)
    return science


def _read_jsonl(path: Path) -> dict[tuple[object, ...], Mapping[str, object]]:
    rows: dict[tuple[object, ...], Mapping[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        key = (row["record_id"], row["condition_id"], row.get("metric_output_id"))
        if key in rows:
            raise Phase4D2ProtocolAError(f"science bridge: duplicate parent row {key}")
        rows[key] = row
    return rows


def _validate_science_against_step13(science: D2SharedScience, inputs: D2ProtocolAInputs) -> None:
    """Bridge every numerically rebuilt receipt to the frozen Step-13 authority."""
    parent = ROOT / "results/phase4/d2_protocol_a_full_domain_eligibility_v1" / STEP13_RUN_ID
    expected_conditions = _read_jsonl(parent / "record_conditions.jsonl")
    expected_metrics = _read_jsonl(parent / "metric_statuses.jsonl")
    expected_peaks = _read_jsonl(parent / "cwt_receipts.jsonl")
    for row in science.record_conditions:
        key = (row["record_id"], row["condition_id"], None)
        expected = expected_conditions.get(key)
        if expected is None or expected.get("state") != "complete" or expected.get("support_sha256") != row["projected_row_sha256"] or expected.get("result_sha256") != row["result_sha256"] or expected.get("output_axis_sha256") != row["axis_sha256"] or expected.get("output_intensity_sha256") != row["intensity_sha256"]:
            raise Phase4D2ProtocolAError(f"science bridge: record condition mismatch {key}")
    for row in science.metric_values:
        key = (row["record_id"], row["condition_id"], row["metric_output_id"])
        expected = expected_metrics.get(key)
        if expected is None or expected.get("state") != row["state"] or expected.get("result_sha256") != row["result_sha256"] or expected.get("diagnostics_sha256") != row["diagnostics_sha256"]:
            raise Phase4D2ProtocolAError(f"science bridge: metric mismatch {key}")
    for row in science.peak_receipts:
        key = (row["record_id"], row["condition_id"], None)
        expected = expected_peaks.get(key)
        if expected is None or expected.get("state") != row["state"] or expected.get("peak_list_sha256") != row["peak_list_sha256"] or expected.get("diagnostics_sha256") != row["diagnostics_sha256"] or expected.get("warning_sha256") != row["warning_sha256"]:
            raise Phase4D2ProtocolAError(f"science bridge: CWT mismatch {key}")
    if len(science.record_conditions) != len(expected_conditions) or len(science.metric_values) != len(expected_metrics) or len(science.peak_receipts) != len(expected_peaks):
        raise Phase4D2ProtocolAError("science bridge: parent row-count mismatch")


def predict_d2_protocol_a(models: Sequence[D2FrozenModel], science: D2SharedScience, config: Phase4D2ProtocolAConfig, *, inputs: D2ProtocolAInputs) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    """Predict each frozen model over the one shared native-science grid."""
    matrices = science.condition_matrices
    predictions: list[dict[str, object]] = []; seed_rows: list[dict[str, object]] = []; cells: list[dict[str, object]] = []
    for frozen in models:
        pca, classifier = frozen.model
        cell = inputs.model_cells.get((frozen.shot_count, frozen.model_seed), {}) if inputs.model_cells else {}
        cells.append({
            "shot_count": frozen.shot_count, "model_seed": frozen.model_seed, "selected_c": frozen.selected_c, "warning_state": "none",
            "train_record_ids_sha256": _sha(("\n".join(cell.get("train_record_ids", ())) + "\n").encode()),
            "validation_record_ids_sha256": _sha(("\n".join(cell.get("validation_record_ids", ())) + "\n").encode()),
            "train_matrix_sha256": frozen.train_matrix_sha256, "validation_matrix_sha256": frozen.validation_matrix_sha256,
            "pca_train_feature_sha256": frozen.pca_train_feature_sha256, "pca_validation_feature_sha256": frozen.pca_validation_feature_sha256,
            "model_state_sha256": _receipt_sha({"selected_c": frozen.selected_c, "classes": np.asarray(classifier.classes_, dtype="<i8").tolist(), "coef_sha256": _array_sha(np.asarray(classifier.coef_, dtype="<f8")), "intercept_sha256": _array_sha(np.asarray(classifier.intercept_, dtype="<f8")), "n_iter": np.asarray(classifier.n_iter_, dtype="<i8").tolist(), "pca_components_sha256": _array_sha(np.asarray(pca.components_, dtype="<f8")), "pca_mean_sha256": _array_sha(np.asarray(pca.mean_, dtype="<f8")), "pca_explained_variance_sha256": _array_sha(np.asarray(pca.explained_variance_, dtype="<f8"))}),
        })
        for condition_id, matrix in matrices.items():
            predicted = classifier.predict(pca.transform(matrix))
            for order, (truth, choice) in enumerate(zip(inputs.test_labels, predicted, strict=True)):
                predictions.append({"shot_count": frozen.shot_count, "model_seed": frozen.model_seed, "model_identity": f"{frozen.shot_count}:{frozen.model_seed}", "condition_id": condition_id, "record_order": order, "record_id": inputs.record_ids[order], "true_class": int(truth), "predicted_class": int(choice), "correct": bool(truth == choice), "projected_row_sha256": _array_sha(matrix[order], "<f4"), "config_sha256": config.sha256})
            for label in sorted(set(inputs.test_labels.tolist())):
                mask = inputs.test_labels == label
                seed_rows.append({"shot_count": frozen.shot_count, "model_seed": frozen.model_seed, "class_label": int(label), "condition_id": condition_id, "accuracy": float(np.mean(predicted[mask] == inputs.test_labels[mask]))})
    return tuple(predictions), tuple(seed_rows), tuple(cells)


def aggregate_d2_protocol_a(
    predictions: Sequence[Mapping[str, object]],
    metric_values: Sequence[Mapping[str, object]],
    config: Phase4D2ProtocolAConfig,
    *,
    inference_resamples: int | None = None,
) -> D2OutcomeProjection:
    """Aggregate record rows by class, then seed, and infer per shot."""
    if inference_resamples is not None and not config.synthetic_fixture:
        raise Phase4D2ProtocolAError("inference_resamples is test-only")
    count = 2000 if inference_resamples is None else int(inference_resamples)
    sign_count = 100000 if inference_resamples is None else int(inference_resamples)
    by_prediction: dict[tuple[int, int, int, str], list[bool]] = {}
    for row in predictions:
        key = (int(row["shot_count"]), int(row["model_seed"]), int(row.get("class_label", row.get("true_class"))), str(row["condition_id"]))
        by_prediction.setdefault(key, []).append(bool(row["correct"]))
    seed_accuracy = {key: float(np.mean(value)) for key, value in by_prediction.items()}
    expected_conditions = tuple(condition_id for condition_id, _, _ in _condition_specs())
    conditions = tuple(key[3] for key in seed_accuracy)
    if set(conditions) != set(expected_conditions) or len(set(conditions)) != len(expected_conditions):
        raise Phase4D2ProtocolAError("aggregation: exact canonical condition grid required")
    positive = expected_conditions[1:]
    metric_map: dict[tuple[int, str, str], list[float]] = {}
    for row in metric_values:
        if "value" not in row or row["value"] is None:
            continue
        key = (int(row["record_order"]), str(row["condition_id"]), str(row["metric_output_id"]))
        if key in metric_map:
            raise Phase4D2ProtocolAError(f"aggregation: duplicate metric entry {key}")
        metric_map[key] = [float(row["value"])]
    observations: list[dict[str, object]] = []
    alignment_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    sign_rows: list[dict[str, object]] = []
    holm_rows: list[dict[str, object]] = []
    classes = tuple(range(config.class_count))
    record_class: dict[int, int] = {}
    for row in predictions:
        record_class[int(row["record_order"])] = int(row.get("class_label", row.get("true_class")))
    for shot in SHOTS:
        metric_tables: dict[str, tuple[AlignmentObservation, ...]] = {}
        metric_results: dict[str, Mapping[str, object]] = {}
        for metric in METRIC_OUTPUT_IDS:
            table: list[AlignmentObservation] = []
            for condition_id in positive:
                perturbation, encoded = condition_id.split(":", 1)
                alpha = float(np.frombuffer(bytes.fromhex(encoded), dtype="<f8")[0])
                for label in classes:
                    seed_values = [seed_accuracy[(shot, seed, label, condition_id)] for seed in config.model_seeds if (shot, seed, label, condition_id) in seed_accuracy]
                    baseline_values = [seed_accuracy[(shot, seed, label, "alpha0")] for seed in config.model_seeds if (shot, seed, label, "alpha0") in seed_accuracy]
                    if len(seed_values) != len(config.model_seeds) or len(baseline_values) != len(config.model_seeds):
                        raise Phase4D2ProtocolAError("aggregation: incomplete seed/class condition grid")
                    indexes = sorted(index for index, observed_label in record_class.items() if observed_label == label)
                    current_keys = [(index, condition_id, metric) for index in indexes]
                    baseline_keys = [(index, "alpha0", metric) for index in indexes]
                    if any(key not in metric_map for key in (*current_keys, *baseline_keys)):
                        raise Phase4D2ProtocolAError("aggregation: incomplete metric grid")
                    current = [metric_map[key][0] for key in current_keys]
                    baseline = [metric_map[key][0] for key in baseline_keys]
                    direction = PreferredDirection(METRIC_DIRECTIONS[metric])
                    harm = float(np.mean([
                        value - base if direction is PreferredDirection.LOWER_IS_BETTER else base - value
                        for base, value in zip(baseline, current, strict=True)
                    ]))
                    downstream = float(np.mean(baseline_values) - np.mean(seed_values))
                    observation = AlignmentObservation(str(label), perturbation, alpha, harm, downstream)
                    table.append(observation)
                    observations.append({"shot_count": shot, "metric_output_id": metric, "class_label": label, "condition_id": condition_id, "perturbation_id": perturbation, "alpha": alpha, "metric_harm": harm, "downstream_harm": downstream})
            metric_tables[metric] = tuple(table)
            try:
                gap = alignment_gap(table); accuracy = cross_perturbation_accuracy(table)
                metric_results[metric] = {"state": "complete", "gap": gap, "accuracy": accuracy}
            except AlignmentValidationError as error:
                if error.path != "constant downstream":
                    raise
                metric_results[metric] = {"state": "not_evaluable_constant_downstream", "reason": str(error)}
        reference = metric_tables["mse"]
        reference_result = metric_results["mse"]
        p_values: dict[str, float] = {}; contrasts: dict[str, float] = {}; candidate_rows: dict[str, Mapping[str, object]] = {}
        if reference_result["state"] == "complete":
            try:
                mse_bootstrap = bulk_paired_cluster_bootstrap(reference, reference, resamples=count, random_seed=20260817)
                alignment_rows.append({"shot_count": shot, "metric_output_id": "mse", "state": "complete", "ag": reference_result["gap"].alignment_gap, "ag_raw": reference_result["gap"].raw_alignment_gap, "acc_cross": reference_result["accuracy"].accuracy, "ag_interval": mse_bootstrap.reference_ag_interval, "acc_interval": mse_bootstrap.reference_acc_interval, "d_ag": None, "d_acc": None, "d_ag_interval": None, "d_acc_interval": None})
            except AlignmentValidationError as error:
                metric_results["mse"] = {"state": "not_evaluable_constant_downstream", "reason": str(error)}
                alignment_rows.append({"shot_count": shot, "metric_output_id": "mse", "state": "not_evaluable_constant_downstream", "ag": None, "ag_raw": None, "acc_cross": None, "ag_interval": None, "acc_interval": None, "d_ag": None, "d_acc": None, "d_ag_interval": None, "d_acc_interval": None, "reason": str(error)})
        else:
            alignment_rows.append({"shot_count": shot, "metric_output_id": "mse", "state": reference_result["state"], "ag": None, "ag_raw": None, "acc_cross": None, "ag_interval": None, "acc_interval": None, "d_ag": None, "d_acc": None, "d_ag_interval": None, "d_acc_interval": None, "reason": reference_result.get("reason")})
        for metric in METRIC_OUTPUT_IDS[1:]:
            candidate = metric_tables[metric]
            if metric_results["mse"]["state"] != "complete" or metric_results[metric]["state"] != "complete":
                bootstrap_rows.append({"shot_count": shot, "metric_output_id": metric, "state": "not_evaluable_metric_incomplete", "resamples": None})
                candidate_rows[metric] = {"state": "not_evaluable_metric_incomplete"}
                for statistic in ("d_ag", "d_acc"):
                    p_values[f"{metric}:{statistic}"] = 1.0
                    sign_rows.append({"shot_count": shot, "metric_output_id": metric, "statistic": statistic, "state": "not_tested_metric_incomplete", "contrast": None, "p_value": 1.0, "resamples": None})
                continue
            try:
                comparison = compare_alignment(reference, candidate)
                bootstrap = bulk_paired_cluster_bootstrap(reference, candidate, resamples=count, random_seed=20260817)
                bootstrap_rows.append({"shot_count": shot, "metric_output_id": metric, "state": "complete", "resamples": count, "candidate_ag_interval": bootstrap.candidate_ag_interval, "candidate_acc_interval": bootstrap.candidate_acc_interval, "d_ag_interval": bootstrap.d_ag_interval, "d_acc_interval": bootstrap.d_acc_interval})
                candidate_rows[metric] = {"state": "complete", "comparison": comparison, "bootstrap": bootstrap}
                for statistic, values in (("d_ag", comparison.d_ag), ("d_acc", comparison.d_acc)):
                    contributions = (
                        [item.value for item in comparison.ag_contribution_differences]
                        if statistic == "d_ag"
                        else [item.value for item in comparison.acc_contribution_differences]
                    )
                    sign = paired_contribution_sign_flip(
                        contributions, aggregation="sum" if statistic == "d_ag" else "mean",
                        resamples=sign_count, random_seed=20260817,
                    )
                    hypothesis = f"{metric}:{statistic}"
                    p_values[hypothesis] = sign.p_value; contrasts[hypothesis] = values
                    sign_rows.append({"shot_count": shot, "metric_output_id": metric, "statistic": statistic, "state": "complete", "contrast": values, "p_value": sign.p_value, "resamples": sign_count})
            except AlignmentValidationError as error:
                bootstrap_rows.append({"shot_count": shot, "metric_output_id": metric, "state": "not_evaluable_metric_incomplete", "resamples": None, "reason": str(error)})
                candidate_rows[metric] = {"state": "not_evaluable_metric_incomplete", "reason": str(error)}
                for statistic in ("d_ag", "d_acc"):
                    hypothesis = f"{metric}:{statistic}"; p_values[hypothesis] = 1.0
                    sign_rows.append({"shot_count": shot, "metric_output_id": metric, "statistic": statistic, "state": "not_tested_metric_incomplete", "contrast": None, "p_value": 1.0, "resamples": None})
        family_results = {result.hypothesis_id: result for result in holm_step_down(p_values, alpha=0.05)}
        for metric in METRIC_OUTPUT_IDS[1:]:
            item = candidate_rows[metric]
            if item["state"] == "complete":
                comparison, bootstrap = item["comparison"], item["bootstrap"]
                alignment_rows.append({"shot_count": shot, "metric_output_id": metric, "state": "complete", "ag": comparison.candidate_gap.alignment_gap, "ag_raw": comparison.candidate_gap.raw_alignment_gap, "acc_cross": comparison.candidate_accuracy.accuracy, "ag_interval": bootstrap.candidate_ag_interval, "acc_interval": bootstrap.candidate_acc_interval, "d_ag": comparison.d_ag, "d_acc": comparison.d_acc, "d_ag_interval": bootstrap.d_ag_interval, "d_acc_interval": bootstrap.d_acc_interval})
            else:
                alignment_rows.append({"shot_count": shot, "metric_output_id": metric, "state": item["state"], "ag": None, "ag_raw": None, "acc_cross": None, "ag_interval": None, "acc_interval": None, "d_ag": None, "d_acc": None, "d_ag_interval": None, "d_acc_interval": None, "reason": item.get("reason")})
        for result in family_results.values():
            metric, statistic = result.hypothesis_id.split(":", 1)
            favorable = bool(contrasts.get(result.hypothesis_id, 0.0) > 0.0)
            sign_state = next(row["state"] for row in sign_rows if row["shot_count"] == shot and row["metric_output_id"] == metric and row["statistic"] == statistic)
            holm_rows.append({"shot_count": shot, "metric_output_id": metric, "statistic": statistic, "raw_p_value": result.raw_p_value, "adjusted_p_value": result.adjusted_p_value, "rank": result.rank, "family_size": result.family_size, "family_state": "complete" if sign_state == "complete" else "not_tested_metric_incomplete", "favorable": favorable, "rejected": bool(result.rejected and favorable)})
    return D2OutcomeProjection(tuple(observations), tuple(alignment_rows), tuple(bootstrap_rows), tuple(sign_rows), tuple(holm_rows))


def _matplotlib_bytes(figure: object, kind: str) -> bytes:
    stream = io.BytesIO()
    figure.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
    plt.close(figure)
    return stream.getvalue()


def render_d2_protocol_a_figures(projection: D2OutcomeProjection, config: Phase4D2ProtocolAConfig) -> Mapping[str, bytes]:
    payloads: dict[str, bytes] = {}
    colors = dict(zip(PERTURBATIONS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True))
    plt.rcParams.update({"font.family": "DejaVu Sans", "lines.linewidth": 1.5})
    for shot in SHOTS:
        figure1_rows = []
        for metric in METRIC_OUTPUT_IDS:
            for perturbation in PERTURBATIONS:
                for alpha in ALPHAS[1:]:
                    selected = [row for row in projection.class_observations if row["shot_count"] == shot and row["metric_output_id"] == metric and row["perturbation_id"] == perturbation and row["alpha"] == alpha]
                    figure1_rows.append({"shot_count": shot, "metric_output_id": metric, "perturbation_id": perturbation, "alpha": alpha, "mean_metric_harm": None if not selected else float(np.mean([row["metric_harm"] for row in selected])), "mean_downstream_harm": None if not selected else float(np.mean([row["downstream_harm"] for row in selected])), "metric_state": next((row["state"] for row in projection.alignment_results if row["shot_count"] == shot and row["metric_output_id"] == metric), "not_evaluable")})
        figure2_rows = [row for row in projection.alignment_results if row["shot_count"] == shot]
        prefix = f"d2_{shot}shot_protocol_a_full_domain"
        payloads[f"figure1_{prefix}_data.csv"] = _csv(figure1_rows, ("shot_count", "metric_output_id", "perturbation_id", "alpha", "mean_metric_harm", "mean_downstream_harm", "metric_state"))
        family = {(row["metric_output_id"], row["statistic"]): row for row in projection.holm_family if row["shot_count"] == shot}
        figure2_fields = ("shot_count", "metric_output_id", "state", "ag", "ag_raw", "ag_interval", "acc_cross", "acc_interval", "d_ag", "d_ag_interval", "d_acc", "d_acc_interval", "d_ag_raw_p", "d_ag_adjusted_p", "d_ag_rank", "d_ag_favorable", "d_ag_rejected", "d_acc_raw_p", "d_acc_adjusted_p", "d_acc_rank", "d_acc_favorable", "d_acc_rejected")
        figure2_rows = [{**row, **{f"d_ag_{name}": family.get((row["metric_output_id"], "d_ag"), {}).get(source) for name, source in (("raw_p", "raw_p_value"), ("adjusted_p", "adjusted_p_value"), ("rank", "rank"), ("favorable", "favorable"), ("rejected", "rejected"))}, **{f"d_acc_{name}": family.get((row["metric_output_id"], "d_acc"), {}).get(source) for name, source in (("raw_p", "raw_p_value"), ("adjusted_p", "adjusted_p_value"), ("rank", "rank"), ("favorable", "favorable"), ("rejected", "rejected"))}} for row in figure2_rows]
        payloads[f"figure2_{prefix}_data.csv"] = _csv(
            [{field: row.get(field) for field in figure2_fields} for row in figure2_rows], figure2_fields
        )
        plt.rcParams["svg.hashsalt"] = f"d2-protocol-a-{shot}-shot"
        figure, axes = plt.subplots(4, 4, figsize=(12, 12))
        for axis, metric in zip(axes.ravel(), METRIC_OUTPUT_IDS, strict=False):
            for perturbation in PERTURBATIONS:
                rows = [row for row in figure1_rows if row["metric_output_id"] == metric and row["perturbation_id"] == perturbation]
                axis.plot([row["mean_metric_harm"] for row in rows], [row["mean_downstream_harm"] for row in rows], marker="o", color=colors[perturbation], label=perturbation)
            axis.set_title(metric)
            values = [value for row in figure1_rows if row["metric_output_id"] == metric for value in (row["mean_metric_harm"], row["mean_downstream_harm"]) if value is not None]
            if values:
                pad = max(1e-12, (max(values) - min(values)) * 0.05)
                axis.set_xlim(min(values) - pad, max(values) + pad); axis.set_ylim(min(values) - pad, max(values) + pad)
        for axis in axes.ravel()[13:]: axis.set_axis_off()
        figure.tight_layout()
        png = io.BytesIO(); svg = io.BytesIO()
        figure.savefig(png, format="png", dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
        figure.savefig(svg, format="svg", metadata={"Date": None, "Creator": "raman-preproc-eval"})
        plt.close(figure)
        payloads[f"figure1_{prefix}.png"] = png.getvalue()
        payloads[f"figure1_{prefix}.svg"] = svg.getvalue()
        figure, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        for axis, field, title, color in zip(axes, ("ag", "acc_cross", "d_ag", "d_acc"), ("AG", "Acc-cross", "D_AG", "D_Acc"), ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"), strict=True):
            values = [row.get(field) for row in figure2_rows]
            ypos = np.arange(len(METRIC_OUTPUT_IDS))
            colors_by_state = [color if value is not None else "#bdbdbd" for value in values]
            errors = [0.0 if row.get(field + "_interval") is None else max(abs(value - row[field + "_interval"][0]), abs(row[field + "_interval"][1] - value)) for row, value in zip(figure2_rows, values)]
            axis.barh(ypos, [0.0 if value is None else value for value in values], xerr=errors, color=colors_by_state)
            axis.set_title(title); axis.set_yticks(ypos, METRIC_OUTPUT_IDS if axis is axes[0] else [])
            for index, row in enumerate(figure2_rows):
                if values[index] is None:
                    axis.text(0.0, index, "ineligible", color="#666666", va="center", fontsize=7)
                elif field in {"d_ag", "d_acc"}:
                    statistic_prefix = field
                    axis.text(values[index], index, f" p={row.get(statistic_prefix + '_raw_p'):.3g}/{row.get(statistic_prefix + '_adjusted_p'):.3g} r={row.get(statistic_prefix + '_rank')}", va="center", fontsize=6)
            finite = [value for value in values if value is not None]
            if finite:
                pad = max(1e-12, (max(finite) - min(finite)) * 0.05)
                axis.set_xlim(min(finite) - pad, max(finite) + pad)
        figure.tight_layout()
        png = io.BytesIO(); svg = io.BytesIO()
        figure.savefig(png, format="png", dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
        figure.savefig(svg, format="svg", metadata={"Date": None, "Creator": "raman-preproc-eval"})
        plt.close(figure)
        payloads[f"figure2_{prefix}.png"] = png.getvalue()
        payloads[f"figure2_{prefix}.svg"] = svg.getvalue()
    return payloads


def build_phase4_d2_protocol_a_from_inputs(output_dir: Path, *, inputs: D2ProtocolAInputs, config: Phase4D2ProtocolAConfig, worker_count: int, inference_resamples: int | None = None) -> Phase4D2ProtocolASummary:
    if worker_count < 1:
        raise Phase4D2ProtocolAError("worker_count must be positive")
    if inference_resamples is not None and not config.synthetic_fixture:
        raise Phase4D2ProtocolAError("inference_resamples is test-only")
    bridge: Mapping[str, object]
    if config.synthetic_fixture:
        bridge = MappingProxyType({"parent_run_id": "synthetic-step13", "step13_config_sha256": "synthetic", "step13_sha256sums_sha256": "synthetic", "full_domain_state": "evaluable", "record_condition_count": config.test_record_count * 41, "metric_status_count": config.test_record_count * 41 * 13, "cwt_receipt_count": config.test_record_count * 41, "checksums": {}})
    else:
        bridge = validate_d2_eligibility_parent(ROOT / "results/phase4/d2_protocol_a_full_domain_eligibility_v1" / STEP13_RUN_ID, config)
    science = rematerialize_d2_protocol_a_science(inputs, config, worker_count=worker_count)
    matrices, conditions, metrics, peaks = (
        science.condition_matrices, science.record_conditions, science.metric_values, science.peak_receipts
    )
    models = fit_d2_protocol_a_models(inputs, config)
    predictions, seed_rows, cells = predict_d2_protocol_a(models, science, config, inputs=inputs)
    aggregate_predictions = [
        {**row, "class_label": int(inputs.test_labels[int(row["record_order"])]), "correct": row["correct"]}
        for row in predictions
    ]
    projection = aggregate_d2_protocol_a(aggregate_predictions, metrics, config, inference_resamples=inference_resamples)
    observations, alignments, bootstraps, signs, holm = (
        projection.class_observations, projection.alignment_results, projection.bootstrap_results,
        projection.sign_flip_results, projection.holm_family,
    )
    expected = config.document["expected"]
    if len(conditions) != expected["condition_count"] or len(metrics) != expected["metric_value_count"] or len(predictions) != expected["prediction_row_count"]:
        raise Phase4D2ProtocolAError("outcome rows: frozen denominator mismatch")
    code_identity = _sha(Path(__file__).read_bytes())
    downstream_rows = tuple({"record_order": row["record_order"], "record_id": row["record_id"], "condition_id": row["condition_id"], "projected_row_sha256": row["projected_row_sha256"]} for row in conditions)
    if len({(row["record_id"], row["condition_id"]) for row in downstream_rows}) != len(conditions):
        raise Phase4D2ProtocolAError("downstream rows: record-condition uniqueness mismatch")
    prediction_summary = []
    for shot in SHOTS:
        for condition_id, _, _ in _condition_specs():
            selected = [row for row in predictions if row["shot_count"] == shot and row["condition_id"] == condition_id]
            labels = sorted({int(row["true_class"]) for row in selected})
            per_label = [float(np.mean([row["correct"] for row in selected if int(row["true_class"]) == label])) for label in labels]
            f1 = []
            for label in labels:
                tp = sum(1 for row in selected if row["true_class"] == label and row["predicted_class"] == label)
                fp = sum(1 for row in selected if row["true_class"] != label and row["predicted_class"] == label)
                fn = sum(1 for row in selected if row["true_class"] == label and row["predicted_class"] != label)
                f1.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
            prediction_summary.append({"shot_count": shot, "condition_id": condition_id, "prediction_count": len(selected), "top1_macro_class_accuracy": float(np.mean(per_label)), "top1_micro_accuracy": float(np.mean([row["correct"] for row in selected])), "macro_f1": float(np.mean(f1))})
    payloads: dict[str, bytes] = {
        "config.json": config.raw_bytes, "eligibility_bridge.json": _canonical(dict(bridge)),
        "model_cells.jsonl": _jsonl(cells), "validation_scores.jsonl": _jsonl([score for model in models for score in model.validation_scores]),
        "record_conditions.jsonl": _jsonl(conditions), "metric_values.jsonl": _jsonl(metrics), "peak_receipts.jsonl": _jsonl(peaks), "downstream_rows.jsonl": _jsonl(downstream_rows), "predictions.jsonl": _jsonl([{**row, "code_identity": code_identity} for row in predictions]), "seed_class_conditions.jsonl": _jsonl(seed_rows),
        "condition_summary.csv": _csv(prediction_summary, ("shot_count", "condition_id", "prediction_count", "top1_macro_class_accuracy", "top1_micro_accuracy", "macro_f1")), "class_observations.jsonl": _jsonl(observations), "alignment_results.jsonl": _jsonl(alignments), "bootstrap_results.jsonl": _jsonl(bootstraps), "sign_flip_results.jsonl": _jsonl(signs), "holm_family.jsonl": _jsonl(holm),
    }
    # Keep the legacy synthetic builder compact, but render actual figure bytes
    # from the outcome projection rather than signature-only placeholders.
    rendered = render_d2_protocol_a_figures(projection, config)
    for name in ARTIFACT_PAYLOAD_FILES:
        if name in rendered:
            payloads[name] = rendered[name]
    table_fields = tuple(next(iter(rendered.values()), b"")) if False else ("shot_count", "metric_output_id", "state", "ag", "ag_raw", "ag_interval", "acc_cross", "acc_interval", "d_ag", "d_ag_interval", "d_acc", "d_acc_interval")
    payloads["d2_protocol_a_full_domain_secondary_table.csv"] = _csv([{field: row.get(field) for field in table_fields} for row in alignments], table_fields)
    run_id = _sha(config.raw_bytes + _canonical({"records": len(inputs.record_ids), "seeds": list(config.model_seeds)}))
    shot_states = {str(shot): ("complete" if next(row for row in alignments if row["shot_count"] == shot and row["metric_output_id"] == "mse")["state"] == "complete" else "failed") for shot in SHOTS}
    terminal_status = "complete" if all(value == "complete" for value in shot_states.values()) else "failed"
    manifest = {"schema_version": "phase4-d2-protocol-a-full-domain-artifact-v1", "run_id": run_id, "status": terminal_status, "shared_condition_count": len(conditions), "shared_metric_value_count": len(metrics), "shared_cwt_receipt_count": len(peaks), "model_fit_count": len(models), "lr_candidate_fit_count": len(cells) * 4, "prediction_row_count": len(predictions), "seed_class_condition_count": len(seed_rows), "shot_endpoint_states": shot_states, "metric_states": {f"{row['shot_count']}:{row['metric_output_id']}": row["state"] for row in alignments}, "payload_files": list(ARTIFACT_PAYLOAD_FILES)}
    payloads["manifest.json"] = _canonical(manifest)
    if tuple(payloads) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D2ProtocolAError("artifact payload order mismatch")
    output = Path(output_dir)
    if output.exists():
        raise Phase4D2ProtocolAError("output: append-only target already exists")
    output.mkdir(parents=True)
    for name, value in payloads.items():
        (output / name).write_bytes(value)
    marker_name = "complete.json" if terminal_status == "complete" else "failed.json"
    (output / marker_name).write_bytes(_canonical({"run_id": run_id, "status": terminal_status}))
    sums = b"".join(f"{_sha((output / name).read_bytes())}  {name}\n".encode() for name in (*ARTIFACT_PAYLOAD_FILES, marker_name))
    (output / "SHA256SUMS").write_bytes(sums)
    return Phase4D2ProtocolASummary(output, run_id, terminal_status, len(inputs.record_ids), len(models))


def build_phase4_d2_protocol_a(output_root: Path, *, worker_count: int = 16) -> Phase4D2ProtocolASummary:
    root = Path(__file__).resolve().parents[2]
    config = load_phase4_d2_protocol_a_config(root / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json")
    parent = root / "results/phase4/d2_protocol_a_full_domain_eligibility_v1" / STEP13_RUN_ID
    validate_d2_eligibility_parent(parent, config)
    inputs = reconstruct_d2_protocol_a_inputs(
        root / "data/unified/bacteria_id_reference",
        root / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json",
        config,
    )
    target = Path(output_root) / _sha(config.raw_bytes + _canonical({"records": config.test_record_count, "stage": "d2-protocol-a"}))
    return build_phase4_d2_protocol_a_from_inputs(target, inputs=inputs, config=config, worker_count=worker_count)


__all__ = [name for name in globals() if name.startswith(("Phase4", "D2", "ARTIFACT", "METRIC", "build_", "fit_", "load_", "make_", "parse_", "predict_", "reconstruct_", "rematerialize_", "validate_"))]
