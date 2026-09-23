"""Deterministic D1 Protocol-A full-domain outcome assembly.

The D1 runner reuses only the already verified Step-15 test-side measurement
coordinates.  Every D1 split, model, prediction, aggregate and inference result
is reconstructed locally.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import multiprocessing
import platform
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
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
from rpe.alignment.contracts import AlignmentValidationError
from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.evaluation import (
    PeakPairInput,
    PreferredDirection,
    SingleSpectrumInput,
    Spectrum1D,
    SpectrumPairInput,
    evaluate_metric,
)
from rpe.methods.catalog import load_classical_catalog
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
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import (
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase4_d2_eligibility import (
    load_phase4_d2_eligibility_config,
    project_d2_support,
)
from rpe.runner import phase4_d2_eligibility as _step13_science
from rpe.runner.phase4_d1_protocol_a_authority import CONFIG_BYTES, CONFIG_SHA256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json"
SCHEMA_VERSION = "phase4-d1-protocol-a-full-domain-config-v1"
EXPERIMENT_ID = "phase4-d1-protocol-a-full-domain-v1"
SEEDS = (0, 1, 2, 3, 4)
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
METRIC_OUTPUT_IDS = (
    "mse", "rmse", "mae", "sam", "pearson_r", "nmse",
    "wasserstein_1_cm1", "is_like_structure_to_noise", "precision",
    "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio",
)
METRIC_DIRECTIONS = MappingProxyType({
    **{name: "lower_is_better" for name in (
        "mse", "rmse", "mae", "sam", "nmse",
        "wasserstein_1_cm1", "artifact_peak_ratio", "missing_peak_ratio",
    )},
    **{name: "higher_is_better" for name in (
        "pearson_r", "is_like_structure_to_noise", "precision",
        "recall", "f1",
    )},
})
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "model_cells.jsonl",
    "validation_scores.jsonl",
    "predictions.jsonl",
    "seed_class_conditions.jsonl",
    "condition_summary.csv",
    "class_observations.jsonl",
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "figure1_d1_protocol_a_full_domain.png",
    "figure1_d1_protocol_a_full_domain.svg",
    "figure1_d1_protocol_a_full_domain_data.csv",
    "figure2_d1_protocol_a_full_domain.png",
    "figure2_d1_protocol_a_full_domain.svg",
    "figure2_d1_protocol_a_full_domain_data.csv",
    "d1_protocol_a_full_domain_secondary_table.csv",
    "manifest.json",
)
STEP13_RUN_ID = (
    "phase4-d2-protocol-a-full-domain-eligibility-"
    "0f43f329bfc6a7a1ff7232a336d115849b06dbca884de28d71f6b28502598ef8"
)
STEP13_ROOT = ROOT / "results/phase4/d2_protocol_a_full_domain_eligibility_v1" / STEP13_RUN_ID
STEP13_CONFIG_SHA256 = "f92427c2f18ab445db2bb54d5ea5a97cce21b05dd6ee80f50f3ed4082285a15b"
STEP13_SHA256SUMS_SHA256 = "b41638b4e246326b155ef1c6169b57d93c68266b47a41b9d16f05dc516546e9e"
STEP15_ROOT = ROOT / "results/phase4/d2_protocol_a_full_domain_v1/8e1e76c497de232f0901bea0278c114de6c4c410631a6edd0f799d120d02e8ef"
STEP15_RUN_ID = "67fedde2a92c0ee57c504830ebefc23c2ca0ba6dbebc33a026e6cee2279628c6"
STEP15_SHA256SUMS_SHA256 = "389e56c3eb7c261b3b67d43ab47a18a3d613ce703fdb6cc8d22f1868841f7073"
STEP15_AUTHORIZED = MappingProxyType({
    "record_conditions.jsonl": "ef4792f952dec40ac940b5321a65c57872c672ad8643d084a5c78f985c74f298",
    "metric_values.jsonl": "fc4f1013d729b69d444974106cd90e282028bd377e73cec0d6a8cbd99a74828c",
    "peak_receipts.jsonl": "916dfdfa0ca120bd184eda39a1ee4a81ee171814f263167e223b7b9c3dbd3d94",
})
BRIDGE_SHA256 = "031b4f4f30235a6237ad1a6f89b03fb9f2d1b724f55f7c8ae9d52f402eabaa7c"
TEST_IDS_SHA256 = "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"
SOURCE_LEDGER_SHA256 = "a44e71531c5a639904390ca73738d035e0c3294766f46a8def82b2f9bd9fec1b"
MODEL_LEDGER_SHA256 = "1b77f2825e25265382597e64218f4a2771011cd9cc8f86c4c3f6e729e6c2c486"
ROLE_LEDGER_SHA256 = "211b087e21f3b511acd6acaa800b693ffef0db7c1aaa354c902d146ee2c2090c"
SUPPORT_F64_SHA256 = "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6"
SUPPORT_F32_SHA256 = "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59"
NATIVE_F64_SHA256 = "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d"
_P10_MEMORY_BUDGET_BYTES = 64 * 1024**3
_CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
_CLAIM_BOUNDARY = "local_execution_artifact_redistribution_not_cleared"
_FORBIDDEN_PHASE05_NAMES = frozenset({
    "complete_cell.json", "complete_cells.json", "failed.json",
    "complete.json",
})
_CODE_RELATIVE_PATHS = (
    "rpe/alignment/bulk.py", "rpe/alignment/contracts.py",
    "rpe/alignment/core.py", "rpe/alignment/inference.py",
    "rpe/downstream/bacteria_id.py", "rpe/evaluation/contracts.py",
    "rpe/methods/catalog.py", "rpe/methods/classical/peaks.py",
    "rpe/metrics/fidelity.py", "rpe/metrics/peak.py",
    "rpe/metrics/reference_free.py", "rpe/metrics/transport.py",
    "rpe/perturb/axis_transform.py", "rpe/perturb/baseline_distortion.py",
    "rpe/perturb/contracts.py", "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py", "rpe/perturb/sweep.py",
    "rpe/runner/d1_bacteria_id.py", "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_gates.py", "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py", "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d2_eligibility.py",
    "rpe/runner/phase4_d1_protocol_a.py",
    "rpe/runner/phase4_d1_protocol_a_verifier.py",
    "tools/run_phase4_d1_protocol_a.py",
)


class Phase4D1ProtocolAError(ValueError):
    pass


@dataclass(frozen=True)
class Phase4D1ProtocolAConfig:
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
class D1ProtocolAInputs:
    test_values: np.ndarray
    test_labels: np.ndarray
    record_ids: tuple[str, ...]
    model_seeds: tuple[int, ...]
    train_values: Mapping[int, np.ndarray] | None = None
    train_labels: Mapping[int, np.ndarray] | None = None
    validation_values: Mapping[int, np.ndarray] | None = None
    validation_labels: Mapping[int, np.ndarray] | None = None
    model_cells: Mapping[int, Mapping[str, object]] | None = None
    support_axis_cm1: np.ndarray | None = None
    native_test_spectra: tuple[Spectrum1D, ...] = ()
    native_test_labels: np.ndarray | None = None
    source_record_count: int = 0
    model_cell_count: int = 0
    role_occurrence_count: int = 0
    source_ledger_sha256: str = ""
    model_ledger_sha256: str = ""
    role_ledger_sha256: str = ""
    support_axis_f32_sha256: str = ""
    support_axis_f64_sha256: str = ""
    role_overlap_detected: bool = False
    force_metric_failure: bool = False


@dataclass(frozen=True)
class D1FrozenModel:
    model_seed: int
    selected_c: float
    model: object
    validation_scores: tuple[Mapping[str, object], ...]
    train_matrix_sha256: str
    validation_matrix_sha256: str
    pca_train_feature_sha256: str
    pca_validation_feature_sha256: str
    refit_with_validation: bool = False
    condition_call_count: int = 41


@dataclass(frozen=True)
class Phase4D1ProtocolASummary:
    path: Path
    run_id: str
    status: str
    test_record_count: int
    model_cell_count: int


@dataclass(frozen=True)
class D1OutcomeProjection:
    class_observations: tuple[Mapping[str, object], ...]
    alignment_results: tuple[Mapping[str, object], ...]
    bootstrap_results: tuple[Mapping[str, object], ...]
    sign_flip_results: tuple[Mapping[str, object], ...]
    holm_family: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class D1SharedScience:
    condition_matrices: Mapping[str, np.ndarray]
    record_conditions: tuple[Mapping[str, object], ...]
    metric_values: tuple[Mapping[str, object], ...]
    peak_receipts: tuple[Mapping[str, object], ...]

    def __iter__(self) -> Iterator[object]:
        yield self.condition_matrices
        yield self.record_conditions
        yield self.metric_values
        yield self.peak_receipts


@dataclass(frozen=True)
class _RealScienceAuthorities:
    eligibility: object
    sweep: object
    phase1: object
    cwt_system: object
    record_conditions: tuple[Mapping[str, object], ...]
    metric_values: tuple[Mapping[str, object], ...]
    peak_receipts: tuple[Mapping[str, object], ...]


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return _sha(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


def _id_digest(values: Sequence[str]) -> str:
    return _sha(("\n".join(values) + "\n").encode("utf-8"))


def _receipt_sha(value: object) -> str:
    return _sha(_step13_science._canonical_json_bytes(value))


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _csv(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=fields, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _live_environment() -> dict[str, str]:
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


def _live_code_authority() -> dict[str, dict[str, object]]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha_file(ROOT / relative),
        }
        for relative in _CODE_RELATIVE_PATHS
    }


def _condition_specs() -> tuple[tuple[str, str | None, float], ...]:
    return (("alpha0", None, 0.0),) + tuple(
        (f"{perturbation}:{np.float64(alpha).tobytes().hex()}", perturbation, alpha)
        for perturbation in PERTURBATIONS
        for alpha in ALPHAS[1:]
    )


def parse_phase4_d1_protocol_a_config(
    path: Path, raw: bytes, *, require_frozen_identity: bool
) -> Phase4D1ProtocolAConfig:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D1ProtocolAError(f"config: {error}") from error
    if raw != _canonical(document):
        raise Phase4D1ProtocolAError("config: must use canonical JSON")
    if require_frozen_identity and (len(raw) != CONFIG_BYTES or _sha(raw) != CONFIG_SHA256):
        raise Phase4D1ProtocolAError("config: frozen config identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D1ProtocolAError("config: schema or experiment mismatch")
    if tuple(document.get("active_perturbation_ids", ())) != PERTURBATIONS:
        raise Phase4D1ProtocolAError("config: perturbation order mismatch")
    if tuple(float(value) for value in document.get("alpha_grid", ())) != ALPHAS:
        raise Phase4D1ProtocolAError("config: alpha grid mismatch")
    if tuple(document.get("metric_output_ids", ())) != METRIC_OUTPUT_IDS:
        raise Phase4D1ProtocolAError("config: metric manifest mismatch")
    if tuple(document.get("artifact_payload_files", ())) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D1ProtocolAError("config: artifact inventory mismatch")
    synthetic = bool(document.get("synthetic_fixture", False))
    if require_frozen_identity and synthetic:
        raise Phase4D1ProtocolAError("config: public load rejects synthetic fixtures")
    denominator = document.get("denominators", {})
    expected = document.get("expected", {})
    if not isinstance(denominator, Mapping) or not isinstance(expected, Mapping):
        raise Phase4D1ProtocolAError("config: denominator/expected must be objects")
    class_count = int(denominator.get("class_count", 0))
    test_count = int(denominator.get("test_record_count", 0))
    seeds = tuple(int(value) for value in document.get("model_seeds", ()))
    if class_count < 1 or test_count < class_count or not seeds:
        raise Phase4D1ProtocolAError("config: invalid model denominator")
    checks = {
        "condition_count": test_count * 41,
        "metric_value_count": test_count * 41 * len(METRIC_OUTPUT_IDS),
        "cwt_receipt_count": test_count * 41,
        "model_cell_count": len(seeds),
        "lr_candidate_fit_count": len(seeds) * 4,
        "prediction_row_count": len(seeds) * test_count * 41,
        "seed_class_condition_count": len(seeds) * class_count * 41,
        "class_observation_count": class_count * 40 * len(METRIC_OUTPUT_IDS),
        "alignment_result_count": len(METRIC_OUTPUT_IDS),
        "bootstrap_result_count": len(METRIC_OUTPUT_IDS) - 1,
        "sign_flip_result_count": (len(METRIC_OUTPUT_IDS) - 1) * 2,
        "holm_family_row_count": (len(METRIC_OUTPUT_IDS) - 1) * 2,
    }
    if any(int(expected.get(key, -1)) != value for key, value in checks.items()):
        raise Phase4D1ProtocolAError("config: frozen denominator mismatch")
    if not synthetic:
        _validate_real_config(document)
    return Phase4D1ProtocolAConfig(
        Path(path), raw, _sha(raw), synthetic, class_count, test_count, seeds,
        int(document.get("support_point_count", 997)), ARTIFACT_PAYLOAD_FILES,
        MappingProxyType(document),
    )


def load_phase4_d1_protocol_a_config(path: Path) -> Phase4D1ProtocolAConfig:
    path = Path(path)
    return parse_phase4_d1_protocol_a_config(
        path, path.read_bytes(), require_frozen_identity=True
    )


def _validate_real_config(document: Mapping[str, object]) -> None:
    metric_manifest = [
        {"output_id": name, "preferred_direction": METRIC_DIRECTIONS[name]}
        for name in METRIC_OUTPUT_IDS
    ]
    support = json.loads(
        (ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json").read_bytes()
    )["support_grid"]
    exact = {
        "protocol": "A",
        "tier": "full_domain_core",
        "claim_boundary": _CLAIM_BOUNDARY,
        "model_seeds": list(SEEDS),
        "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS),
        "metric_output_ids": list(METRIC_OUTPUT_IDS),
        "metric_manifest": metric_manifest,
        "cwt_system_id": _CWT_SYSTEM_ID,
        "denominators": {
            "class_count": 30, "test_record_count": 3000,
            "source_record_count": 66000, "model_cell_count": 5,
            "model_role_occurrence_count": 330000,
        },
        "expected": {
            "operator_cell_count": 15000, "apply_check_count": 135000,
            "condition_count": 123000, "metric_value_count": 1599000,
            "cwt_receipt_count": 123000, "model_cell_count": 5,
            "pca_fit_count": 5, "lr_candidate_fit_count": 20,
            "frozen_model_condition_call_count": 205,
            "prediction_row_count": 615000, "seed_class_condition_count": 6150,
            "class_observation_count": 15600, "alignment_result_count": 13,
            "bootstrap_result_count": 12, "sign_flip_result_count": 24,
            "holm_family_row_count": 24, "figure1_row_count": 520,
            "figure2_row_count": 13, "secondary_table_row_count": 13,
            "cross_perturbation_pair_count_per_class": 640,
            "cross_perturbation_pair_count_per_metric": 19200,
            "configured_payload_count": 21, "terminal_marker_count": 1,
            "sha256sums_count": 1, "artifact_file_count": 23,
        },
        "support_grid": support,
        "support_point_count": 997,
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
            "random_seed": 20260817, "confidence_level": 0.95,
            "holm_alpha": 0.05, "cluster_unit": "class_label",
            "d_ag_formula": "AG_MSE-AG_candidate",
            "d_acc_formula": "Acc_candidate-Acc_MSE",
            "class_contribution_reduction": {"D_AG": "sum", "D_Acc": "mean"},
            "holm_slot_count": 24,
        },
        "inherited_rulings": {
            "p01_p05": "not_evaluable_coverage",
            "p06_p07": "structurally_ineligible_missing_explicit_baseline",
        },
        "figure_contract": {
            "colors": {"p08": "#1f77b4", "p09": "#ff7f0e", "p10": "#2ca02c", "p11": "#d62728", "p12": "#9467bd"},
            "dpi": 300, "figure1_inches": [12, 12], "figure1_layout": [4, 4],
            "figure2_inches": [14, 8], "figure2_layout": [1, 4],
            "font_family": "DejaVu Sans", "line_width": 1.5,
            "marker": "o", "padding_fraction": 0.05,
            "svg_hashsalt": "rpe-phase4-d1-protocol-a-v1",
        },
        "artifact_contract": {
            "checksum_file": "SHA256SUMS",
            "checksum_scope": "all_payloads_and_exactly_one_terminal_marker",
            "payload_files": list(ARTIFACT_PAYLOAD_FILES),
            "terminal_marker_rule": "exactly_one",
            "terminal_markers": ["complete.json", "failed.json"],
        },
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "trust_anchor": {
            "config_authority_relative_path": "rpe/runner/phase4_d1_protocol_a_authority.py",
            "config_binds_authority": False, "direction": "authority_to_config_only",
        },
    }
    for key, value in exact.items():
        if document.get(key) != value:
            raise Phase4D1ProtocolAError(f"config: frozen {key} authority mismatch")
    if document.get("environment_authority") != _live_environment():
        raise Phase4D1ProtocolAError("config: environment authority mismatch")
    code = document.get("code_authority")
    if not isinstance(code, Mapping) or code != _live_code_authority():
        raise Phase4D1ProtocolAError("config: code authority mismatch")
    authorities = document.get("authorities")
    if not isinstance(authorities, Mapping):
        raise Phase4D1ProtocolAError("config: scientific authorities missing")
    expected_authority_keys = {
        "parent_plan", "phase4_preregistration", "alignment_core_design",
        "step12_design", "step13_report", "step13_config",
        "step15_report", "step19_report", "step20_design", "sweep",
        "phase1_core_config", "classical_catalog", "d1_phase05_protocol",
        "d1_phase05_complete_report", "d1_recipe_config",
        "bacteria_id_sha256sums", "bacteria_id_retained_snapshot",
        "step13_parent", "step15_config", "step15_parent",
    }
    if set(authorities) != expected_authority_keys:
        raise Phase4D1ProtocolAError("config: scientific authority inventory mismatch")
    for value in authorities.values():
        if not isinstance(value, Mapping):
            raise Phase4D1ProtocolAError("config: malformed scientific authority")
        live = ROOT / str(value.get("path", ""))
        if (
            not live.is_file()
            or live.stat().st_size != int(value.get("bytes", -1))
            or _sha_file(live) != value.get("sha256")
        ):
            raise Phase4D1ProtocolAError(f"config: live authority mismatch for {live}")
    identities = document.get("frozen_identities", {})
    if not isinstance(identities, Mapping):
        raise Phase4D1ProtocolAError("config: frozen identities missing")
    required = {
        "source_ledger_sha256": SOURCE_LEDGER_SHA256,
        "model_ledger_sha256": MODEL_LEDGER_SHA256,
        "role_ledger_sha256": ROLE_LEDGER_SHA256,
        "test_record_ids_sha256": TEST_IDS_SHA256,
        "native_axis_increasing_f64_sha256": NATIVE_F64_SHA256,
        "support_axis_f64_sha256": SUPPORT_F64_SHA256,
        "support_axis_f32_sha256": SUPPORT_F32_SHA256,
        "condition_bridge_sha256": BRIDGE_SHA256,
    }
    for key, value in required.items():
        if identities.get(key) != value:
            raise Phase4D1ProtocolAError(f"config: frozen identity mismatch for {key}")
    if identities.get("model_seed_ids") != list(SEEDS) or identities.get("support_point_count") != 997:
        raise Phase4D1ProtocolAError("config: seed/support identity mismatch")


def make_synthetic_d1_protocol_a_inputs(
    *, class_count: int, records_per_class: int, model_seeds: tuple[int, ...],
    force_metric_failure: bool = False,
) -> D1ProtocolAInputs:
    if class_count < 2 or records_per_class < 2 or not model_seeds:
        raise Phase4D1ProtocolAError("synthetic inputs: insufficient classes or records")
    axis = np.linspace(350.0, 1830.0, 1000, dtype=np.float64)
    support = np.linspace(386.65, 1792.4, 997, dtype=np.float64)
    values: list[np.ndarray] = []
    labels: list[int] = []
    record_ids: list[str] = []
    native: list[Spectrum1D] = []
    for label in range(class_count):
        for index in range(records_per_class):
            centers = (520.0, 710.0, 910.0, 1130.0, 1370.0, 1610.0)
            intensity = 0.15 + sum(
                (1.0 + 0.08 * label + 0.02 * index + 0.03 * peak)
                * np.exp(-0.5 * ((axis - center) / (13.0 + peak)) ** 2)
                for peak, center in enumerate(centers)
            )
            record_id = f"c{label:02d}-r{index:02d}"
            projected = np.ascontiguousarray(
                np.interp(support, axis, intensity), dtype="<f4"
            )
            values.append(projected)
            labels.append(label)
            record_ids.append(record_id)
            native.append(Spectrum1D(
                spectrum_id=f"synthetic::{record_id}",
                sample_id=record_id,
                axis_cm1=np.ascontiguousarray(axis),
                intensity=np.ascontiguousarray(intensity, dtype=np.float64),
            ))
    test_values = np.asarray(values, dtype="<f4")
    test_labels = np.asarray(labels, dtype="<i8")
    train_values = {}
    validation_values = {}
    train_labels = {}
    validation_labels = {}
    cells = {}
    for seed in model_seeds:
        train_values[seed] = np.ascontiguousarray(test_values, dtype="<f4")
        validation_values[seed] = np.ascontiguousarray(
            test_values + np.float32((seed + 1) * 1e-7), dtype="<f4"
        )
        train_labels[seed] = test_labels.copy()
        validation_labels[seed] = test_labels.copy()
        cells[seed] = {
            "model_seed": seed,
            "train_record_ids": tuple(record_ids),
            "validation_record_ids": tuple(record_ids),
            "test_record_ids": tuple(record_ids),
            "train_record_ids_sha256": _id_digest(record_ids),
            "validation_record_ids_sha256": _id_digest(record_ids),
            "test_record_ids_sha256": _id_digest(record_ids),
        }
    return D1ProtocolAInputs(
        test_values=test_values,
        test_labels=test_labels,
        record_ids=tuple(record_ids),
        model_seeds=tuple(model_seeds),
        train_values=MappingProxyType(train_values),
        train_labels=MappingProxyType(train_labels),
        validation_values=MappingProxyType(validation_values),
        validation_labels=MappingProxyType(validation_labels),
        model_cells=MappingProxyType(cells),
        support_axis_cm1=np.ascontiguousarray(support),
        native_test_spectra=tuple(native),
        native_test_labels=test_labels.copy(),
        source_record_count=len(record_ids),
        model_cell_count=len(model_seeds),
        role_occurrence_count=len(record_ids) * len(model_seeds) * 3,
        source_ledger_sha256="synthetic",
        model_ledger_sha256="synthetic",
        role_ledger_sha256="synthetic",
        support_axis_f32_sha256=_array_sha(support, "<f4"),
        support_axis_f64_sha256=_array_sha(support),
        force_metric_failure=force_metric_failure,
    )


def make_synthetic_d1_protocol_a_config(
    inputs: D1ProtocolAInputs,
) -> Phase4D1ProtocolAConfig:
    count = len(inputs.record_ids)
    classes = len(set(inputs.test_labels.tolist()))
    seeds = inputs.model_seeds
    expected = {
        "condition_count": count * 41,
        "metric_value_count": count * 41 * 13,
        "cwt_receipt_count": count * 41,
        "model_cell_count": len(seeds),
        "lr_candidate_fit_count": len(seeds) * 4,
        "prediction_row_count": len(seeds) * count * 41,
        "seed_class_condition_count": len(seeds) * classes * 41,
        "class_observation_count": classes * 40 * 13,
        "alignment_result_count": 13,
        "bootstrap_result_count": 12,
        "sign_flip_result_count": 24,
        "holm_family_row_count": 24,
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "synthetic_fixture": True,
        "model_seeds": list(seeds),
        "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS),
        "metric_output_ids": list(METRIC_OUTPUT_IDS),
        "support_point_count": inputs.test_values.shape[1],
        "denominators": {
            "class_count": classes,
            "test_record_count": count,
        },
        "expected": expected,
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
    }
    raw = _canonical(document)
    return parse_phase4_d1_protocol_a_config(
        Path("<synthetic>"), raw, require_frozen_identity=False
    )


def _split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for label in range(30):
        indices = np.flatnonzero(labels == label)
        if len(indices) != 100:
            raise Phase4D1ProtocolAError(
                f"D1 inputs: finetune class {label} must contain 100 records"
            )
        shuffled = generator.permutation(indices)
        validation.extend(shuffled[:10].tolist())
        train.extend(shuffled[10:].tolist())
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def _canonical_ledger_digest(rows: Iterator[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical(row))
    return digest.hexdigest()


def reconstruct_d1_protocol_a_inputs(
    dataset_path: Path, config: Phase4D1ProtocolAConfig
) -> D1ProtocolAInputs:
    dataset_path = Path(dataset_path)
    if (
        dataset_path.name in _FORBIDDEN_PHASE05_NAMES
        or dataset_path.name.startswith("seed")
        or "prediction" in dataset_path.name
        or "selected_c" in dataset_path.name
    ):
        raise Phase4D1ProtocolAError(
            f"{dataset_path}: forbidden Phase-0.5 outcome artifact"
        )
    if config.synthetic_fixture or not dataset_path.is_dir():
        raise Phase4D1ProtocolAError("D1 inputs: retained dataset directory required")
    eligibility = load_phase4_d2_eligibility_config(
        ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
    )
    support = np.asarray(eligibility.support_coordinates_cm1, dtype="<f8")
    split_values: dict[str, list[np.ndarray]] = {
        "finetune": [], "reference": [], "test": []
    }
    split_native_values: dict[str, list[np.ndarray]] = {
        "finetune": [], "reference": [], "test": []
    }
    split_labels: dict[str, list[int]] = {key: [] for key in split_values}
    split_ids: dict[str, list[str]] = {key: [] for key in split_values}
    split_rows: dict[str, list[int]] = {key: [] for key in split_values}
    native_tests: list[Spectrum1D] = []
    native_axis: np.ndarray | None = None
    with BacteriaIdBatchLoader(dataset_path, batch_size=4096) as loader:
        for batch in loader.iter_batches():
            split = batch.source_split
            axis = np.ascontiguousarray(
                np.asarray(batch.wavenumber, dtype="<f4")[::-1], dtype="<f8"
            )
            if not np.all(np.diff(axis) > 0.0):
                raise Phase4D1ProtocolAError("D1 inputs: native axis reversal failed")
            if native_axis is None:
                native_axis = axis
            elif not np.array_equal(native_axis, axis):
                raise Phase4D1ProtocolAError("D1 inputs: shared native axis mismatch")
            for record_id, label, source_row, intensity in zip(
                batch.record_ids, batch.class_labels, batch.source_rows,
                batch.intensity, strict=True,
            ):
                record_id = str(record_id)
                native_intensity = np.ascontiguousarray(
                    np.asarray(intensity, dtype="<f4")[::-1], dtype="<f8"
                )
                spectrum = Spectrum1D(
                    spectrum_id=f"bacteria_id_reference::{record_id}",
                    sample_id=None, axis_cm1=axis, intensity=native_intensity,
                )
                projected = project_d2_support(spectrum, eligibility)
                split_values[split].append(projected)
                split_native_values[split].append(
                    np.ascontiguousarray(intensity, dtype="<f4")
                )
                split_labels[split].append(int(label))
                split_ids[split].append(record_id)
                split_rows[split].append(int(source_row))
                if split == "test":
                    native_tests.append(spectrum)
    if native_axis is None or _array_sha(native_axis) != NATIVE_F64_SHA256:
        raise Phase4D1ProtocolAError("D1 inputs: native axis identity mismatch")
    if _array_sha(support) != SUPPORT_F64_SHA256 or _array_sha(support, "<f4") != SUPPORT_F32_SHA256:
        raise Phase4D1ProtocolAError("D1 inputs: support axis identity mismatch")
    expected_counts = {"finetune": 3000, "reference": 60000, "test": 3000}
    if {key: len(value) for key, value in split_ids.items()} != expected_counts:
        raise Phase4D1ProtocolAError("D1 inputs: source split cardinality mismatch")
    if _id_digest(split_ids["test"]) != TEST_IDS_SHA256:
        raise Phase4D1ProtocolAError("D1 inputs: test identity mismatch")
    source_digest = _canonical_ledger_digest(
        {
            "class_label": split_labels[split][index],
            "record_id": record_id,
            "source_row": split_rows[split][index],
            "source_split": split,
        }
        for split in ("finetune", "reference", "test")
        for index, record_id in enumerate(split_ids[split])
    )
    train_values: dict[int, np.ndarray] = {}
    train_labels: dict[int, np.ndarray] = {}
    validation_values: dict[int, np.ndarray] = {}
    validation_labels: dict[int, np.ndarray] = {}
    cells: dict[int, Mapping[str, object]] = {}
    model_rows: list[Mapping[str, object]] = []
    role_rows: list[Mapping[str, object]] = []
    role_overlap = False
    reference_values = np.asarray(split_values["reference"], dtype="<f4")
    reference_native = np.asarray(split_native_values["reference"], dtype="<f4")
    reference_labels = np.asarray(split_labels["reference"], dtype="<i8")
    finetune_values = np.asarray(split_values["finetune"], dtype="<f4")
    finetune_native = np.asarray(split_native_values["finetune"], dtype="<f4")
    finetune_labels = np.asarray(split_labels["finetune"], dtype="<i8")
    test_values = np.asarray(split_values["test"], dtype="<f4")
    test_native = np.asarray(split_native_values["test"], dtype="<f4")
    test_labels = np.asarray(split_labels["test"], dtype="<i8")
    for seed in config.model_seeds:
        fit_idx, valid_idx = _split_indices(finetune_labels, seed)
        train_ids = tuple(split_ids["reference"] + [split_ids["finetune"][i] for i in fit_idx])
        valid_ids = tuple(split_ids["finetune"][i] for i in valid_idx)
        test_ids = tuple(split_ids["test"])
        train_matrix = np.ascontiguousarray(
            np.concatenate((reference_values, finetune_values[fit_idx]), axis=0),
            dtype="<f4",
        )
        valid_matrix = np.ascontiguousarray(finetune_values[valid_idx], dtype="<f4")
        train_label = np.concatenate((reference_labels, finetune_labels[fit_idx]))
        valid_label = finetune_labels[valid_idx]
        if train_matrix.shape != (62700, 997) or valid_matrix.shape != (300, 997):
            raise Phase4D1ProtocolAError("D1 inputs: model matrix cardinality mismatch")
        if any(set(a) & set(b) for a, b in ((train_ids, valid_ids), (train_ids, test_ids), (valid_ids, test_ids))):
            role_overlap = True
        source_hashes = {
            "train": {_sha(row.tobytes()) for row in np.concatenate((reference_native, finetune_native[fit_idx]))},
            "validation": {_sha(row.tobytes()) for row in finetune_native[valid_idx]},
            "test": {_sha(row.tobytes()) for row in test_native},
        }
        projected_hashes = {
            "train": {_sha(row.tobytes()) for row in train_matrix},
            "validation": {_sha(row.tobytes()) for row in valid_matrix},
            "test": {_sha(row.tobytes()) for row in test_values},
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
            raise Phase4D1ProtocolAError("D1 inputs: exact cross-role duplicate detected")
        cell = {
            "model_seed": seed,
            "train_count": len(train_ids),
            "train_record_ids": train_ids,
            "train_record_ids_sha256": _id_digest(train_ids),
            "validation_count": len(valid_ids),
            "validation_record_ids": valid_ids,
            "validation_record_ids_sha256": _id_digest(valid_ids),
            "test_count": len(test_ids),
            "test_record_ids": test_ids,
            "test_record_ids_sha256": _id_digest(test_ids),
        }
        cells[seed] = MappingProxyType(cell)
        model_rows.append({key: value for key, value in cell.items() if not key.endswith("record_ids")})
        for role, ids, labels, rows, splits in (
            ("train", train_ids, train_label,
             tuple(split_rows["reference"]) + tuple(split_rows["finetune"][i] for i in fit_idx),
             ("reference",) * 60000 + ("finetune",) * 2700),
            ("validation", valid_ids, valid_label, tuple(split_rows["finetune"][i] for i in valid_idx), ("finetune",) * 300),
            ("test", test_ids, test_labels, tuple(split_rows["test"]), ("test",) * 3000),
        ):
            role_rows.extend(
                {
                    "class_label": int(label), "model_seed": seed,
                    "record_id": record_id, "role": role,
                    "role_order": order, "source_row": int(source_row),
                    "source_split": source_split,
                }
                for order, (record_id, label, source_row, source_split) in enumerate(
                    zip(ids, labels, rows, splits, strict=True)
                )
            )
        train_values[seed] = train_matrix
        train_labels[seed] = np.asarray(train_label, dtype="<i8")
        validation_values[seed] = valid_matrix
        validation_labels[seed] = np.asarray(valid_label, dtype="<i8")
    model_digest = _canonical_ledger_digest(iter(model_rows))
    role_digest = _canonical_ledger_digest(iter(role_rows))
    if (
        source_digest != SOURCE_LEDGER_SHA256
        or model_digest != MODEL_LEDGER_SHA256
        or role_digest != ROLE_LEDGER_SHA256
        or role_overlap
    ):
        raise Phase4D1ProtocolAError("D1 inputs: frozen ledger identity mismatch")
    return D1ProtocolAInputs(
        test_values=np.ascontiguousarray(test_values, dtype="<f4"),
        test_labels=test_labels, record_ids=tuple(split_ids["test"]),
        model_seeds=config.model_seeds,
        train_values=MappingProxyType(train_values),
        train_labels=MappingProxyType(train_labels),
        validation_values=MappingProxyType(validation_values),
        validation_labels=MappingProxyType(validation_labels),
        model_cells=MappingProxyType(cells), support_axis_cm1=support,
        native_test_spectra=tuple(native_tests), native_test_labels=test_labels.copy(),
        source_record_count=66000, model_cell_count=5,
        role_occurrence_count=330000, source_ledger_sha256=source_digest,
        model_ledger_sha256=model_digest, role_ledger_sha256=role_digest,
        support_axis_f32_sha256=_array_sha(support, "<f4"),
        support_axis_f64_sha256=_array_sha(support),
        role_overlap_detected=role_overlap,
    )


def _read_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in checksums:
            raise Phase4D1ProtocolAError(f"parent: duplicate checksum {name}")
        checksums[name] = digest
    return checksums


def _validate_checksum_tree(path: Path, expected_sums_sha: str) -> dict[str, str]:
    if not path.is_dir() or _sha_file(path / "SHA256SUMS") != expected_sums_sha:
        raise Phase4D1ProtocolAError("parent: checksum authority mismatch")
    checksums = _read_checksum_file(path / "SHA256SUMS")
    for name, digest in checksums.items():
        if not (path / name).is_file() or _sha_file(path / name) != digest:
            raise Phase4D1ProtocolAError(f"parent: checksum mismatch for {name}")
    return checksums


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise Phase4D1ProtocolAError(
                    f"{path.name}[{line_number}]: invalid JSON"
                ) from error
            if not isinstance(value, Mapping):
                raise Phase4D1ProtocolAError(f"{path.name}[{line_number}]: object required")
            if raw_line != _canonical(value):
                raise Phase4D1ProtocolAError(
                    f"{path.name}[{line_number}]: noncanonical JSONL"
                )
            yield value


def _load_test_identity() -> tuple[tuple[str, ...], Mapping[str, int]]:
    ids: list[str] = []
    labels: dict[str, int] = {}
    with BacteriaIdBatchLoader(
        ROOT / "data/unified/bacteria_id_reference", batch_size=4096
    ) as loader:
        for batch in loader.iter_batches():
            if batch.source_split != "test":
                continue
            for record_id, label in zip(batch.record_ids, batch.class_labels, strict=True):
                ids.append(str(record_id))
                labels[str(record_id)] = int(label)
    class_counts = {label: list(labels.values()).count(label) for label in range(30)}
    if (
        len(ids) != 3000
        or len(set(ids)) != 3000
        or set(labels.values()) != set(range(30))
        or any(count != 100 for count in class_counts.values())
        or _id_digest(ids) != TEST_IDS_SHA256
    ):
        raise Phase4D1ProtocolAError("Step-15 bridge: retained test identity mismatch")
    return tuple(ids), MappingProxyType(labels)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_validated_step15_measurements(
    test_ids: Sequence[str], labels: Mapping[str, int]
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    str,
]:
    conditions = _condition_specs()
    condition_ids = tuple(row[0] for row in conditions)
    condition_keys = {row for row in condition_ids}
    metric_keys = set(METRIC_OUTPUT_IDS)
    bridge_digest = hashlib.sha256()
    condition_rows: list[Mapping[str, object]] = []
    exact_condition_keys = {
        "axis_sha256", "condition_id", "intensity_sha256",
        "output_spectrum_id", "projected_row_sha256", "record_id",
        "record_order", "result_sha256", "source_spectrum_id", "state",
    }
    for index, row in enumerate(_iter_jsonl(STEP15_ROOT / "record_conditions.jsonl")):
        record_index, condition_index = divmod(index, len(condition_ids))
        if record_index >= len(test_ids):
            raise Phase4D1ProtocolAError("Step-15 conditions: too many rows")
        record_id = test_ids[record_index]
        condition_id = condition_ids[condition_index]
        if (
            set(row) != exact_condition_keys
            or row.get("record_id") != record_id
            or row.get("record_order") != record_index
            or row.get("condition_id") != condition_id
            or row.get("condition_id") not in condition_keys
            or row.get("state") != "complete"
            or not all(_valid_sha256(row.get(key)) for key in (
                "axis_sha256", "intensity_sha256",
                "projected_row_sha256", "result_sha256",
            ))
        ):
            raise Phase4D1ProtocolAError(
                f"Step-15 conditions: schema/order/state mismatch at row {index}"
            )
        bridge_digest.update(_canonical({
            "test_order": record_index, "record_id": record_id,
            "class_label": labels[record_id], "condition_id": condition_id,
            "state": "complete", "axis_sha256": row["axis_sha256"],
            "intensity_sha256": row["intensity_sha256"],
            "support_projection_sha256": row["projected_row_sha256"],
        }))
        condition_rows.append(row)
    if len(condition_rows) != len(test_ids) * len(condition_ids):
        raise Phase4D1ProtocolAError("Step-15 conditions: row count mismatch")
    metric_rows: list[Mapping[str, object]] = []
    exact_metric_keys = {
        "condition_id", "diagnostics_sha256", "metric_output_id",
        "record_id", "record_order", "result_sha256", "state", "value",
    }
    per_record = len(condition_ids) * len(METRIC_OUTPUT_IDS)
    for index, row in enumerate(_iter_jsonl(STEP15_ROOT / "metric_values.jsonl")):
        record_index, remainder = divmod(index, per_record)
        condition_index, metric_index = divmod(remainder, len(METRIC_OUTPUT_IDS))
        if record_index >= len(test_ids):
            raise Phase4D1ProtocolAError("Step-15 metrics: too many rows")
        value = row.get("value")
        if (
            set(row) != exact_metric_keys
            or row.get("record_id") != test_ids[record_index]
            or row.get("record_order") != record_index
            or row.get("condition_id") != condition_ids[condition_index]
            or row.get("metric_output_id") != METRIC_OUTPUT_IDS[metric_index]
            or row.get("metric_output_id") not in metric_keys
            or row.get("state") != "complete"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not _valid_sha256(row.get("result_sha256"))
            or not _valid_sha256(row.get("diagnostics_sha256"))
        ):
            raise Phase4D1ProtocolAError(
                f"Step-15 metrics: schema/order/state/value mismatch at row {index}"
            )
        metric_rows.append(row)
    if len(metric_rows) != len(test_ids) * per_record:
        raise Phase4D1ProtocolAError("Step-15 metrics: row count mismatch")
    peak_rows: list[Mapping[str, object]] = []
    exact_peak_keys = {
        "condition_id", "diagnostics_sha256", "peak_list_sha256",
        "record_id", "record_order", "state", "warning_sha256",
    }
    for index, row in enumerate(_iter_jsonl(STEP15_ROOT / "peak_receipts.jsonl")):
        record_index, condition_index = divmod(index, len(condition_ids))
        if record_index >= len(test_ids) or (
            set(row) != exact_peak_keys
            or row.get("record_id") != test_ids[record_index]
            or row.get("record_order") != record_index
            or row.get("condition_id") != condition_ids[condition_index]
            or row.get("state") != "complete"
            or not all(_valid_sha256(row.get(key)) for key in (
                "diagnostics_sha256", "peak_list_sha256", "warning_sha256",
            ))
        ):
            raise Phase4D1ProtocolAError(
                f"Step-15 CWT: schema/order/state mismatch at row {index}"
            )
        peak_rows.append(row)
    if len(peak_rows) != len(test_ids) * len(condition_ids):
        raise Phase4D1ProtocolAError("Step-15 CWT: row count mismatch")
    digest = bridge_digest.hexdigest()
    if digest != BRIDGE_SHA256:
        raise Phase4D1ProtocolAError("Step-15 bridge: canonical digest mismatch")
    return tuple(condition_rows), tuple(metric_rows), tuple(peak_rows), digest


def validate_d1_parent_authorities(
    config: Phase4D1ProtocolAConfig,
) -> Mapping[str, object]:
    if "step15_forbidden_probe" in config.document:
        raise Phase4D1ProtocolAError("forbidden Step-15 outcome payload request")
    step13 = _validate_checksum_tree(STEP13_ROOT, STEP13_SHA256SUMS_SHA256)
    gate = json.loads((STEP13_ROOT / "gate.json").read_text(encoding="utf-8"))
    if (
        gate.get("full_domain_core", {}).get("state") != "evaluable"
        or gate.get("peak_common_support", {}).get("state")
        != "not_evaluable_coverage"
    ):
        raise Phase4D1ProtocolAError("Step-13 parent tier authority mismatch")
    step15 = _validate_checksum_tree(STEP15_ROOT, STEP15_SHA256SUMS_SHA256)
    manifest = json.loads((STEP15_ROOT / "manifest.json").read_text(encoding="utf-8"))
    marker = json.loads((STEP15_ROOT / "complete.json").read_text(encoding="utf-8"))
    if manifest.get("run_id") != STEP15_RUN_ID or manifest.get("status") != "complete" or marker.get("run_id") != STEP15_RUN_ID:
        raise Phase4D1ProtocolAError("Step-15 parent identity mismatch")
    for name, digest in STEP15_AUTHORIZED.items():
        if step15.get(name) != digest:
            raise Phase4D1ProtocolAError(f"Step-15 parent payload mismatch for {name}")
    authority_table = config.document.get("authorities", {})
    nested_parent = (
        authority_table.get("step15_parent", {})
        if isinstance(authority_table, Mapping) else {}
    )
    top_parent = config.document.get("step15_parent", nested_parent)
    for parent_config in (nested_parent, top_parent):
        if not isinstance(parent_config, Mapping) or (
            parent_config.get("sha256sums_sha256")
            != STEP15_SHA256SUMS_SHA256
            or parent_config.get("run_id") != STEP15_RUN_ID
            or parent_config.get("authorized_payloads")
            != dict(STEP15_AUTHORIZED)
        ):
            raise Phase4D1ProtocolAError(
                "Step-15 parent checksum authority mismatch"
            )
    test_ids, labels = _load_test_identity()
    conditions, metrics, peaks, digest = _load_validated_step15_measurements(
        test_ids, labels
    )
    condition_count, metric_count, cwt_count = len(conditions), len(metrics), len(peaks)
    return MappingProxyType({
        "authorized_step15_payloads": tuple(STEP15_AUTHORIZED),
        "bridge_row_count": condition_count,
        "condition_bridge_sha256": digest,
        "metric_row_count": metric_count,
        "cwt_row_count": cwt_count,
        "step13_full_domain_state": "evaluable",
        "step13_peak_common_state": "not_evaluable_coverage",
        "step13_sha256sums_sha256": STEP13_SHA256SUMS_SHA256,
        "step15_run_id": STEP15_RUN_ID,
        "step15_sha256sums_sha256": STEP15_SHA256SUMS_SHA256,
        "step13_checksums": dict(sorted(step13.items())),
        "step15_checksums": dict(sorted(step15.items())),
    })


def fit_d1_protocol_a_models(
    inputs: D1ProtocolAInputs, config: Phase4D1ProtocolAConfig
) -> tuple[D1FrozenModel, ...]:
    models: list[D1FrozenModel] = []
    for seed in config.model_seeds:
        if inputs.train_values is None:
            raise Phase4D1ProtocolAError("model lifecycle: train matrices required")
        train_values = inputs.train_values[seed]
        train_labels = inputs.train_labels[seed]
        valid_values = inputs.validation_values[seed]
        valid_labels = inputs.validation_labels[seed]
        if not config.synthetic_fixture and (
            train_values.shape != (62700, 997) or valid_values.shape != (300, 997)
        ):
            raise Phase4D1ProtocolAError("model lifecycle: frozen cardinality mismatch")
        if set(train_labels.tolist()) != set(range(config.class_count)) or set(valid_labels.tolist()) != set(range(config.class_count)):
            raise Phase4D1ProtocolAError("model lifecycle: missing class")
        pca = PCA(
            n_components=min(20, train_values.shape[0], train_values.shape[1]),
            svd_solver="randomized", whiten=False, random_state=seed,
        )
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            train_x = pca.fit_transform(train_values)
            valid_x = pca.transform(valid_values)
        if captured or not np.isfinite(train_x).all() or not np.isfinite(valid_x).all():
            raise Phase4D1ProtocolAError("model lifecycle: PCA warning or nonfinite output")
        best_c: float | None = None
        best_score = -math.inf
        best_model: LogisticRegression | None = None
        scores: list[Mapping[str, object]] = []
        for c_value in (0.01, 0.1, 1.0, 10.0):
            candidate = LogisticRegression(
                C=c_value, l1_ratio=0.0, solver="lbfgs", max_iter=1000,
                tol=1e-4, class_weight=None, random_state=seed,
            )
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                candidate.fit(train_x, train_labels)
            if captured:
                raise Phase4D1ProtocolAError("model lifecycle: LR warning")
            predicted = candidate.predict(valid_x)
            if predicted.shape != valid_labels.shape:
                raise Phase4D1ProtocolAError("model lifecycle: invalid prediction shape")
            score = float(np.mean(predicted == valid_labels))
            scores.append({
                "model_seed": seed, "c": c_value,
                "validation_top1_accuracy": score,
            })
            if score > best_score:
                best_c, best_score, best_model = c_value, score, candidate
        if best_model is None or best_c is None:
            raise Phase4D1ProtocolAError("model lifecycle: no candidate selected")
        models.append(D1FrozenModel(
            model_seed=seed, selected_c=float(best_c), model=(pca, best_model),
            validation_scores=tuple(scores),
            train_matrix_sha256=_array_sha(train_values, "<f4"),
            validation_matrix_sha256=_array_sha(valid_values, "<f4"),
            pca_train_feature_sha256=_array_sha(train_x),
            pca_validation_feature_sha256=_array_sha(valid_x),
        ))
    return tuple(models)


def _phase1_source(
    record_order: int, record_id: str, class_label: int, spectrum: Spectrum1D
) -> Phase1Source:
    return Phase1Source(
        selection=SelectedSourceRow(
            record_order, record_id, spectrum.sample_id or record_id, class_label,
            f"bacteria-{class_label}", f"native::{_array_sha(spectrum.axis_cm1)}",
        ),
        spectrum=spectrum, original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(spectrum.axis_cm1, "<f4"),
        source_intensity_float32_sha256=_array_sha(spectrum.intensity, "<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1),
        normalized_intensity_float64_sha256=_array_sha(spectrum.intensity),
        provenance=MappingProxyType({
            "license": None, "license_status": "not_stated",
            "retrieved_date": "2026-08-23",
            "sha256": "0" * 64, "source_artifact": "bacteria_id_reference",
            "source_url": "local://bacteria_id_reference",
        }),
    )


def _support_projection(
    spectrum: Spectrum1D, inputs: D1ProtocolAInputs, eligibility: object | None
) -> np.ndarray:
    if eligibility is not None:
        return project_d2_support(spectrum, eligibility)
    if inputs.support_axis_cm1 is None:
        raise Phase4D1ProtocolAError("science: support axis required")
    support = inputs.support_axis_cm1
    if spectrum.axis_cm1[0] > support[0] or spectrum.axis_cm1[-1] < support[-1]:
        raise Phase4D1ProtocolAError("science: support requires extrapolation")
    projected = np.interp(support, spectrum.axis_cm1, spectrum.intensity)
    if not np.isfinite(projected).all() or np.linalg.norm(projected) == 0.0:
        raise Phase4D1ProtocolAError("science: invalid projected spectrum")
    return np.ascontiguousarray(projected, dtype="<f4")


def _condition_id(perturbation: str, alpha: float) -> str:
    return f"{perturbation}:{np.float64(alpha).tobytes().hex()}"


def _perturbation_result_sha(item: object) -> str:
    result = item.result
    return _receipt_sha({
        "alpha_float64_le_hex": item.alpha_float64_le_hex,
        "axis_behavior": result.axis_behavior.value,
        "axis_changed": result.axis_changed,
        "diagnostics": result.diagnostics,
        "intensity_changed": result.intensity_changed,
        "output_axis_sha256": _array_sha(result.output.axis_cm1),
        "output_intensity_sha256": _array_sha(result.output.intensity),
        "output_spectrum_id": result.output.spectrum_id,
        "perturbation_id": result.perturbation_id,
        "source_spectrum_id": result.source_spectrum_id,
        "state_digest": result.state_digest,
    })


def _metric_objects() -> Mapping[str, object]:
    return MappingProxyType({
        "mse": MSEMetric(), "rmse": RMSEMetric(), "mae": MAEMetric(),
        "sam": SAMMetric(), "pearson_r": PearsonRMetric(),
        "nmse": NMSEMetric(), "wasserstein_1_cm1": Wasserstein1Metric(),
        "is_like_structure_to_noise": ISLikeStructureToNoiseMetric(),
    })


def _native_record_science(
    *, record_order: int, record_id: str, class_label: int, spectrum: Spectrum1D,
    inputs: D1ProtocolAInputs, eligibility: object | None, sweep: object,
    phase1_config: object, cwt_system: object, admission: P10MemoryAdmission,
    include_measurements: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    source = _phase1_source(record_order, record_id, class_label, spectrum)
    projections = {"alpha0": _support_projection(spectrum, inputs, eligibility)}
    spectra = {"alpha0": spectrum}
    result_hashes = {"alpha0": _array_sha(projections["alpha0"], "<f4")}
    for perturbation in PERTURBATIONS:
        cell = run_perturbation_cell(
            source, perturbation, phase1_config, sweep,
            p10_admission=admission if perturbation == "p10" else None,
        )
        if cell.status is not CellStatus.COMPLETE:
            raise Phase4D1ProtocolAError(
                f"science: {record_id}/{perturbation} did not complete"
            )
        alpha0 = next((row for row in cell.records if row.result.alpha == 0.0), None)
        if alpha0 is None or not np.array_equal(alpha0.result.output.axis_cm1, spectrum.axis_cm1) or not np.array_equal(alpha0.result.output.intensity, spectrum.intensity):
            raise Phase4D1ProtocolAError("science: alpha-zero identity failure")
        for item in cell.records:
            if item.result.alpha == 0.0:
                continue
            key = _condition_id(perturbation, item.result.alpha)
            spectra[key] = item.result.output
            projections[key] = _support_projection(item.result.output, inputs, eligibility)
            result_hashes[key] = _perturbation_result_sha(item)
    conditions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    peaks: list[dict[str, object]] = []
    scalar = _metric_objects() if include_measurements else {}
    peak_metric = PeakDetectionCurvesMetric() if include_measurements else None
    reference_peaks = (
        run_peak_detection_system(cwt_system, spectrum)
        if include_measurements else None
    )
    if include_measurements and reference_peaks.status not in {
        PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING
    }:
        raise Phase4D1ProtocolAError("science: reference CWT failed")
    for condition_id, _, _ in _condition_specs():
        output = spectra[condition_id]
        projected = projections[condition_id]
        conditions.append({
            "record_order": record_order, "record_id": record_id,
            "condition_id": condition_id, "source_spectrum_id": spectrum.spectrum_id,
            "output_spectrum_id": output.spectrum_id,
            "axis_sha256": _array_sha(output.axis_cm1),
            "intensity_sha256": _array_sha(output.intensity),
            "projected_row_sha256": _array_sha(projected, "<f4"),
            "result_sha256": result_hashes[condition_id], "state": "complete",
        })
        if not include_measurements:
            continue
        current_peaks = (
            reference_peaks if condition_id == "alpha0"
            else run_peak_detection_system(cwt_system, output)
        )
        if current_peaks.status not in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}:
            raise Phase4D1ProtocolAError("science: condition CWT failed")
        peaks.append({
            "record_order": record_order, "record_id": record_id,
            "condition_id": condition_id, "state": current_peaks.status.value,
            "peak_list_sha256": current_peaks.peaks_sha256,
            "diagnostics_sha256": _receipt_sha(dict(current_peaks.diagnostics)),
            "warning_sha256": _receipt_sha(current_peaks.warnings),
        })
        for output_id, metric in scalar.items():
            request = (
                SingleSpectrumInput(output)
                if output_id == "is_like_structure_to_noise"
                else SpectrumPairInput(spectrum, output)
            )
            result = evaluate_metric(metric, request)
            scalar_output = next(row for row in result.outputs if row.output_id == output_id)
            metrics.append({
                "record_order": record_order, "record_id": record_id,
                "condition_id": condition_id, "metric_output_id": output_id,
                "state": "complete", "value": float(scalar_output.value),
                "result_sha256": _receipt_sha(result),
                "diagnostics_sha256": _receipt_sha(result.diagnostics),
            })
        structure = evaluate_metric(
            peak_metric,
            PeakPairInput(
                tuple(row.to_peak1d() for row in reference_peaks.peaks),
                tuple(row.to_peak1d() for row in current_peaks.peaks), 2.0, (0.0,),
            ),
        )
        by_id = {row.output_id: row for row in structure.outputs}
        for output_id in METRIC_OUTPUT_IDS[8:]:
            metrics.append({
                "record_order": record_order, "record_id": record_id,
                "condition_id": condition_id, "metric_output_id": output_id,
                "state": "complete", "value": float(by_id[output_id].value),
                "result_sha256": _receipt_sha(structure),
                "diagnostics_sha256": _receipt_sha(structure.diagnostics),
            })
    return projections, conditions, metrics, peaks


_PROCESS_ELIGIBILITY: object | None = None
_PROCESS_SWEEP: object | None = None
_PROCESS_PHASE1_CONFIG: object | None = None
_PROCESS_CWT_SYSTEM: object | None = None


def _initialize_real_science_worker() -> None:
    global _PROCESS_ELIGIBILITY, _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG, _PROCESS_CWT_SYSTEM
    _PROCESS_ELIGIBILITY = load_phase4_d2_eligibility_config(
        ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
    )
    _PROCESS_SWEEP = load_perturbation_sweep_config(
        ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
    )
    _PROCESS_PHASE1_CONFIG = load_phase1_core_config(
        ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
    )
    catalog = load_classical_catalog(
        ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"
    )
    matches = [row for row in catalog.systems if row.system_id == _CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D1ProtocolAError("science: CWT authority resolution failed")
    _PROCESS_CWT_SYSTEM = matches[0]


def _real_science_worker(
    job: tuple[int, str, int, Spectrum1D]
) -> tuple[int, dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if any(value is None for value in (
        _PROCESS_ELIGIBILITY, _PROCESS_SWEEP, _PROCESS_PHASE1_CONFIG,
        _PROCESS_CWT_SYSTEM,
    )):
        raise Phase4D1ProtocolAError("science worker was not initialized")
    order, record_id, label, spectrum = job
    placeholder = D1ProtocolAInputs(
        np.empty((0, 0)), np.empty((0,), dtype=np.int64), (), (),
    )
    with threadpool_limits(limits=1, user_api="blas"):
        bundle = _native_record_science(
            record_order=order, record_id=record_id, class_label=label,
            spectrum=spectrum, inputs=placeholder,
            eligibility=_PROCESS_ELIGIBILITY, sweep=_PROCESS_SWEEP,
            phase1_config=_PROCESS_PHASE1_CONFIG, cwt_system=_PROCESS_CWT_SYSTEM,
            admission=P10MemoryAdmission(_P10_MEMORY_BUDGET_BYTES),
            include_measurements=False,
        )
    return (order, *bundle)


def _load_parent_rows(path: Path) -> tuple[Mapping[str, object], ...]:
    return tuple(_iter_jsonl(path))


def _load_real_science_authorities() -> _RealScienceAuthorities:
    eligibility = load_phase4_d2_eligibility_config(
        ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
    )
    sweep = load_perturbation_sweep_config(
        ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
    )
    phase1 = load_phase1_core_config(
        ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
    )
    catalog = load_classical_catalog(
        ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"
    )
    matches = [row for row in catalog.systems if row.system_id == _CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D1ProtocolAError("science: CWT authority resolution failed")
    _validate_checksum_tree(STEP15_ROOT, STEP15_SHA256SUMS_SHA256)
    test_ids, labels = _load_test_identity()
    conditions, metrics, peaks, _ = _load_validated_step15_measurements(
        test_ids, labels
    )
    return _RealScienceAuthorities(
        eligibility, sweep, phase1, matches[0],
        conditions, metrics, peaks,
    )


def rematerialize_d1_protocol_a_science(
    inputs: D1ProtocolAInputs, config: Phase4D1ProtocolAConfig, *, worker_count: int
) -> D1SharedScience:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise Phase4D1ProtocolAError("worker_count must be positive")
    if len(inputs.native_test_spectra) != len(inputs.record_ids) or inputs.native_test_labels is None:
        raise Phase4D1ProtocolAError("science: native test spectra required")
    if not np.array_equal(inputs.native_test_labels, inputs.test_labels):
        raise Phase4D1ProtocolAError("science: native label ordering mismatch")
    authorities = _load_real_science_authorities() if not config.synthetic_fixture else None
    estimate = estimate_p10_peak_bytes(
        max(spectrum.axis_cm1.size for spectrum in inputs.native_test_spectra)
    )
    if estimate > _P10_MEMORY_BUDGET_BYTES:
        raise Phase4D1ProtocolAError("science: P10 64-GiB admission failed")
    jobs = tuple(
        (index, record_id, int(label), spectrum)
        for index, (record_id, label, spectrum) in enumerate(zip(
            inputs.record_ids, inputs.native_test_labels,
            inputs.native_test_spectra, strict=True,
        ))
    )
    if config.synthetic_fixture:
        bundles = tuple(
            _native_record_science(
                record_order=index, record_id=record_id, class_label=label,
                spectrum=spectrum, inputs=inputs, eligibility=None,
                sweep=load_perturbation_sweep_config(
                    ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
                ),
                phase1_config=load_phase1_core_config(
                    ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"
                ),
                cwt_system=next(
                    row for row in load_classical_catalog(
                        ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"
                    ).systems if row.system_id == _CWT_SYSTEM_ID
                ),
                admission=P10MemoryAdmission(_P10_MEMORY_BUDGET_BYTES),
                include_measurements=True,
            )
            for index, record_id, label, spectrum in jobs
        )
    else:
        capacity = max(1, _P10_MEMORY_BUDGET_BYTES // estimate)
        with ProcessPoolExecutor(
            max_workers=min(worker_count, len(jobs), capacity),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_real_science_worker,
            initargs=(),
        ) as executor:
            completed = tuple(sorted(
                executor.map(_real_science_worker, jobs), key=lambda row: row[0]
            ))
        bundles = tuple(row[1:] for row in completed)
    by_condition: dict[str, list[np.ndarray]] = {
        condition_id: [] for condition_id, _, _ in _condition_specs()
    }
    conditions: list[Mapping[str, object]] = []
    metrics: list[Mapping[str, object]] = []
    peaks: list[Mapping[str, object]] = []
    for matrices, condition_rows, metric_rows, peak_rows in bundles:
        for condition_id, row in matrices.items():
            by_condition[condition_id].append(row)
        conditions.extend(condition_rows)
        metrics.extend(metric_rows)
        peaks.extend(peak_rows)
    if not config.synthetic_fixture and isinstance(authorities, _RealScienceAuthorities):
        expected = {
            (str(row["record_id"]), str(row["condition_id"])): row
            for row in authorities.record_conditions
        }
        if len(expected) != 123000 or len(conditions) != 123000:
            raise Phase4D1ProtocolAError("science: Step-15 condition count mismatch")
        for row in conditions:
            parent = expected.get((str(row["record_id"]), str(row["condition_id"])))
            if parent is None or any(parent.get(key) != row.get(key) for key in (
                "record_order", "state", "axis_sha256",
                "intensity_sha256", "projected_row_sha256", "result_sha256",
            )):
                raise Phase4D1ProtocolAError("science: Step-15 rematerialization mismatch")
        metrics = list(authorities.metric_values)
        peaks = list(authorities.peak_receipts)
    if getattr(inputs, "force_metric_failure", False):
        metrics = [
            ({**row, "state": "failed_runtime", "value": None}
             if row.get("metric_output_id") == "mse" else row)
            for row in metrics
        ]
    return D1SharedScience(
        MappingProxyType({
            key: np.ascontiguousarray(value, dtype="<f4")
            for key, value in by_condition.items()
        }),
        tuple(conditions), tuple(metrics), tuple(peaks),
    )


def predict_d1_protocol_a(
    models: Sequence[D1FrozenModel], science: D1SharedScience,
    config: Phase4D1ProtocolAConfig, *, inputs: D1ProtocolAInputs,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    predictions: list[Mapping[str, object]] = []
    seed_rows: list[Mapping[str, object]] = []
    cells: list[Mapping[str, object]] = []
    for frozen in models:
        pca, classifier = frozen.model
        cell = inputs.model_cells.get(frozen.model_seed, {}) if inputs.model_cells else {}
        cells.append({
            "model_seed": frozen.model_seed, "selected_c": frozen.selected_c,
            "warning_state": "none",
            "train_record_ids_sha256": cell.get("train_record_ids_sha256", _id_digest(cell.get("train_record_ids", ()))),
            "validation_record_ids_sha256": cell.get("validation_record_ids_sha256", _id_digest(cell.get("validation_record_ids", ()))),
            "test_record_ids_sha256": cell.get("test_record_ids_sha256", _id_digest(inputs.record_ids)),
            "train_matrix_sha256": frozen.train_matrix_sha256,
            "validation_matrix_sha256": frozen.validation_matrix_sha256,
            "pca_train_feature_sha256": frozen.pca_train_feature_sha256,
            "pca_validation_feature_sha256": frozen.pca_validation_feature_sha256,
            "model_state_sha256": _receipt_sha({
                "selected_c": frozen.selected_c,
                "classes": np.asarray(classifier.classes_, dtype="<i8").tolist(),
                "coef_sha256": _array_sha(classifier.coef_),
                "intercept_sha256": _array_sha(classifier.intercept_),
                "n_iter": np.asarray(classifier.n_iter_, dtype="<i8").tolist(),
                "pca_components_sha256": _array_sha(pca.components_),
                "pca_mean_sha256": _array_sha(pca.mean_),
                "pca_explained_variance_sha256": _array_sha(pca.explained_variance_),
            }),
        })
        for condition_id, matrix in science.condition_matrices.items():
            predicted = classifier.predict(pca.transform(matrix))
            for order, (truth, choice) in enumerate(zip(
                inputs.test_labels, predicted, strict=True
            )):
                predictions.append({
                    "model_seed": frozen.model_seed,
                    "model_identity": str(frozen.model_seed),
                    "condition_id": condition_id, "record_order": order,
                    "record_id": inputs.record_ids[order],
                    "true_class": int(truth), "predicted_class": int(choice),
                    "correct": bool(truth == choice),
                    "projected_row_sha256": _array_sha(matrix[order], "<f4"),
                    "config_sha256": config.sha256,
                })
            for label in range(config.class_count):
                mask = inputs.test_labels == label
                seed_rows.append({
                    "model_seed": frozen.model_seed, "class_label": label,
                    "condition_id": condition_id,
                    "accuracy": float(np.mean(predicted[mask] == inputs.test_labels[mask])),
                })
    return tuple(predictions), tuple(seed_rows), tuple(cells)


def aggregate_d1_protocol_a(
    predictions: Sequence[Mapping[str, object]],
    metric_values: Sequence[Mapping[str, object]],
    config: Phase4D1ProtocolAConfig,
    *, bootstrap_resamples: int | None = None, sign_flip_resamples: int | None = None,
) -> D1OutcomeProjection:
    if not config.synthetic_fixture and (bootstrap_resamples is not None or sign_flip_resamples is not None):
        raise Phase4D1ProtocolAError("inference resamples are test-only overrides")
    bootstrap_count = 2000 if bootstrap_resamples is None else int(bootstrap_resamples)
    sign_count = 100000 if sign_flip_resamples is None else int(sign_flip_resamples)
    conditions = tuple(row[0] for row in _condition_specs())
    classes = tuple(range(config.class_count))
    pred_accumulator: dict[tuple[int, int, str], list[bool]] = {}
    record_class: dict[int, int] = {}
    for row in predictions:
        order = int(row["record_order"])
        label = int(row.get("class_label", row.get("true_class")))
        previous = record_class.setdefault(order, label)
        if previous != label:
            raise Phase4D1ProtocolAError("aggregation: inconsistent record class")
        key = (int(row["model_seed"]), label, str(row["condition_id"]))
        pred_accumulator.setdefault(key, []).append(bool(row["correct"]))
    expected_pred = {
        (seed, label, condition)
        for seed in config.model_seeds for label in classes for condition in conditions
    }
    if set(pred_accumulator) != expected_pred:
        raise Phase4D1ProtocolAError("aggregation: exact seed/class/condition grid required")
    seed_accuracy = {key: float(np.mean(value)) for key, value in pred_accumulator.items()}
    metric_map: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for row in metric_values:
        key = (int(row["record_order"]), str(row["condition_id"]), str(row["metric_output_id"]))
        if key in metric_map:
            raise Phase4D1ProtocolAError(f"aggregation: duplicate metric row {key}")
        metric_map[key] = row
    class_indexes = {
        label: tuple(index for index, value in record_class.items() if value == label)
        for label in classes
    }
    positive = conditions[1:]
    observations: list[Mapping[str, object]] = []
    tables: dict[str, tuple[AlignmentObservation, ...] | None] = {}
    metric_states: dict[str, str] = {}
    for metric in METRIC_OUTPUT_IDS:
        table: list[AlignmentObservation] = []
        complete = True
        pending: list[tuple[int, str, str, float, float | None, float]] = []
        for condition_id in positive:
            perturbation, encoded = condition_id.split(":", 1)
            alpha = float(np.frombuffer(bytes.fromhex(encoded), dtype="<f8")[0])
            for label in classes:
                indexes = class_indexes[label]
                current_rows = [metric_map.get((index, condition_id, metric)) for index in indexes]
                baseline_rows = [metric_map.get((index, "alpha0", metric)) for index in indexes]
                if any(row is None or row.get("state") != "complete" or row.get("value") is None for row in (*current_rows, *baseline_rows)):
                    complete = False
                    metric_harm = None
                else:
                    direction = PreferredDirection(METRIC_DIRECTIONS[metric])
                    metric_harm = float(np.mean([
                        float(current["value"]) - float(base["value"])
                        if direction is PreferredDirection.LOWER_IS_BETTER
                        else float(base["value"]) - float(current["value"])
                        for base, current in zip(baseline_rows, current_rows, strict=True)
                    ]))
                current_acc = [seed_accuracy[(seed, label, condition_id)] for seed in config.model_seeds]
                baseline_acc = [seed_accuracy[(seed, label, "alpha0")] for seed in config.model_seeds]
                downstream = float(np.mean(baseline_acc) - np.mean(current_acc))
                pending.append((label, condition_id, perturbation, alpha, metric_harm, downstream))
        if complete:
            for label, condition_id, perturbation, alpha, metric_harm, downstream in pending:
                table.append(AlignmentObservation(str(label), perturbation, alpha, float(metric_harm), downstream))
                observations.append({
                    "metric_output_id": metric, "class_label": label,
                    "condition_id": condition_id, "perturbation_id": perturbation,
                    "alpha": alpha, "metric_harm": metric_harm,
                    "downstream_harm": downstream, "state": "complete",
                })
            tables[metric] = tuple(table)
            try:
                alignment_gap(table)
                cross_perturbation_accuracy(table)
                metric_states[metric] = "complete"
            except AlignmentValidationError as error:
                if error.path != "constant downstream":
                    raise
                tables[metric] = None
                metric_states[metric] = "not_evaluable_constant_downstream"
        else:
            tables[metric] = None
            metric_states[metric] = "not_evaluable_metric_incomplete"
            for label, condition_id, perturbation, alpha, _, downstream in pending:
                observations.append({
                    "metric_output_id": metric, "class_label": label,
                    "condition_id": condition_id, "perturbation_id": perturbation,
                    "alpha": alpha, "metric_harm": None,
                    "downstream_harm": downstream,
                    "state": "not_evaluable_metric_incomplete",
                })
    alignment_rows: list[Mapping[str, object]] = []
    bootstrap_rows: list[Mapping[str, object]] = []
    sign_rows: list[Mapping[str, object]] = []
    holm_rows: list[Mapping[str, object]] = []
    reference = tables["mse"]
    if reference is not None:
        mse_gap = alignment_gap(reference)
        mse_acc = cross_perturbation_accuracy(reference)
        mse_boot = bulk_paired_cluster_bootstrap(
            reference, reference, resamples=bootstrap_count, random_seed=20260817
        )
        alignment_rows.append({
            "metric_output_id": "mse", "state": "complete",
            "ag": mse_gap.alignment_gap, "ag_raw": mse_gap.raw_alignment_gap,
            "ag_interval": mse_boot.reference_ag_interval,
            "acc_cross": mse_acc.accuracy,
            "acc_interval": mse_boot.reference_acc_interval,
            "d_ag": None, "d_ag_interval": None,
            "d_acc": None, "d_acc_interval": None,
        })
    else:
        alignment_rows.append({
            "metric_output_id": "mse", "state": metric_states["mse"],
            "ag": None, "ag_raw": None, "ag_interval": None,
            "acc_cross": None, "acc_interval": None,
            "d_ag": None, "d_ag_interval": None,
            "d_acc": None, "d_acc_interval": None,
        })
    p_values: dict[str, float] = {}
    comparisons: dict[str, object] = {}
    bootstraps: dict[str, object] = {}
    contrasts: dict[str, float] = {}
    for metric in METRIC_OUTPUT_IDS[1:]:
        candidate = tables[metric]
        if reference is None or candidate is None:
            bootstrap_rows.append({
                "metric_output_id": metric, "state": "not_evaluable_metric_incomplete",
                "resamples": None,
            })
            for statistic in ("d_ag", "d_acc"):
                hypothesis = f"{metric}:{statistic}"
                p_values[hypothesis] = 1.0
                sign_rows.append({
                    "metric_output_id": metric, "statistic": statistic,
                    "state": "not_tested_metric_incomplete",
                    "contrast": None, "p_value": 1.0, "resamples": None,
                })
            continue
        comparison = compare_alignment(reference, candidate)
        bootstrap = bulk_paired_cluster_bootstrap(
            reference, candidate, resamples=bootstrap_count, random_seed=20260817
        )
        comparisons[metric] = comparison
        bootstraps[metric] = bootstrap
        bootstrap_rows.append({
            "metric_output_id": metric, "state": "complete",
            "resamples": bootstrap_count,
            "candidate_ag_interval": bootstrap.candidate_ag_interval,
            "candidate_acc_interval": bootstrap.candidate_acc_interval,
            "d_ag_interval": bootstrap.d_ag_interval,
            "d_acc_interval": bootstrap.d_acc_interval,
        })
        for statistic, contrast, contributions, aggregation in (
            ("d_ag", comparison.d_ag, comparison.ag_contribution_differences, "sum"),
            ("d_acc", comparison.d_acc, comparison.acc_contribution_differences, "mean"),
        ):
            sign = paired_contribution_sign_flip(
                [row.value for row in contributions], aggregation=aggregation,
                resamples=sign_count, random_seed=20260817,
            )
            hypothesis = f"{metric}:{statistic}"
            p_values[hypothesis] = sign.p_value
            contrasts[hypothesis] = contrast
            sign_rows.append({
                "metric_output_id": metric, "statistic": statistic,
                "state": "complete", "contrast": contrast,
                "p_value": sign.p_value, "resamples": sign_count,
            })
    family = {row.hypothesis_id: row for row in holm_step_down(p_values, alpha=0.05)}
    for metric in METRIC_OUTPUT_IDS[1:]:
        if metric in comparisons:
            comparison = comparisons[metric]
            bootstrap = bootstraps[metric]
            alignment_rows.append({
                "metric_output_id": metric, "state": "complete",
                "ag": comparison.candidate_gap.alignment_gap,
                "ag_raw": comparison.candidate_gap.raw_alignment_gap,
                "ag_interval": bootstrap.candidate_ag_interval,
                "acc_cross": comparison.candidate_accuracy.accuracy,
                "acc_interval": bootstrap.candidate_acc_interval,
                "d_ag": comparison.d_ag, "d_ag_interval": bootstrap.d_ag_interval,
                "d_acc": comparison.d_acc, "d_acc_interval": bootstrap.d_acc_interval,
            })
        else:
            alignment_rows.append({
                "metric_output_id": metric, "state": metric_states[metric],
                "ag": None, "ag_raw": None, "ag_interval": None,
                "acc_cross": None, "acc_interval": None,
                "d_ag": None, "d_ag_interval": None,
                "d_acc": None, "d_acc_interval": None,
            })
    for result in family.values():
        metric, statistic = result.hypothesis_id.split(":", 1)
        favorable = contrasts.get(result.hypothesis_id, 0.0) > 0.0
        state = next(
            row["state"] for row in sign_rows
            if row["metric_output_id"] == metric and row["statistic"] == statistic
        )
        holm_rows.append({
            "metric_output_id": metric, "statistic": statistic,
            "raw_p_value": result.raw_p_value,
            "adjusted_p_value": result.adjusted_p_value, "rank": result.rank,
            "family_size": result.family_size,
            "family_state": "complete" if state == "complete" else "not_tested_metric_incomplete",
            "favorable": bool(favorable),
            "rejected": bool(result.rejected and favorable),
        })
    return D1OutcomeProjection(
        tuple(observations), tuple(alignment_rows), tuple(bootstrap_rows),
        tuple(sign_rows), tuple(holm_rows),
    )


def render_d1_protocol_a_figures(
    projection: D1OutcomeProjection, config: Phase4D1ProtocolAConfig
) -> Mapping[str, bytes]:
    colors = dict(zip(
        PERTURBATIONS,
        ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"),
        strict=True,
    ))
    figure1_rows: list[Mapping[str, object]] = []
    state_by_metric = {row["metric_output_id"]: row["state"] for row in projection.alignment_results}
    for metric in METRIC_OUTPUT_IDS:
        for perturbation in PERTURBATIONS:
            for alpha in ALPHAS[1:]:
                selected = [
                    row for row in projection.class_observations
                    if row["metric_output_id"] == metric
                    and row["perturbation_id"] == perturbation
                    and row["alpha"] == alpha
                    and row["metric_harm"] is not None
                ]
                figure1_rows.append({
                    "metric_output_id": metric, "perturbation_id": perturbation,
                    "alpha": alpha,
                    "mean_metric_harm": None if not selected else float(np.mean([row["metric_harm"] for row in selected])),
                    "mean_downstream_harm": None if not selected else float(np.mean([row["downstream_harm"] for row in selected])),
                    "metric_state": state_by_metric[metric],
                })
    family = {(row["metric_output_id"], row["statistic"]): row for row in projection.holm_family}
    figure2_fields = (
        "metric_output_id", "state", "ag", "ag_raw", "ag_interval",
        "acc_cross", "acc_interval", "d_ag", "d_ag_interval",
        "d_acc", "d_acc_interval", "d_ag_raw_p", "d_ag_adjusted_p",
        "d_ag_rank", "d_ag_favorable", "d_ag_rejected",
        "d_acc_raw_p", "d_acc_adjusted_p", "d_acc_rank",
        "d_acc_favorable", "d_acc_rejected",
    )
    figure2_rows = []
    for row in projection.alignment_results:
        output = dict(row)
        for statistic in ("d_ag", "d_acc"):
            item = family.get((row["metric_output_id"], statistic), {})
            for target, source in (
                ("raw_p", "raw_p_value"), ("adjusted_p", "adjusted_p_value"),
                ("rank", "rank"), ("favorable", "favorable"),
                ("rejected", "rejected"),
            ):
                output[f"{statistic}_{target}"] = item.get(source)
        figure2_rows.append(output)
    payloads: dict[str, bytes] = {
        "figure1_d1_protocol_a_full_domain_data.csv": _csv(
            figure1_rows,
            ("metric_output_id", "perturbation_id", "alpha",
             "mean_metric_harm", "mean_downstream_harm", "metric_state"),
        ),
        "figure2_d1_protocol_a_full_domain_data.csv": _csv(
            [{field: row.get(field) for field in figure2_fields} for row in figure2_rows],
            figure2_fields,
        ),
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "lines.linewidth": 1.5,
        "svg.hashsalt": "rpe-phase4-d1-protocol-a-v1",
    })
    figure, axes = plt.subplots(4, 4, figsize=(12, 12))
    flat_axes = axes.ravel()
    downstream_axis = flat_axes[0]
    for perturbation in PERTURBATIONS:
        rows = [
            row for row in figure1_rows
            if row["metric_output_id"] == "mse"
            and row["perturbation_id"] == perturbation
        ]
        downstream_axis.plot(
            [row["alpha"] for row in rows],
            [row["mean_downstream_harm"] for row in rows],
            marker="o", color=colors[perturbation], label=perturbation,
        )
    downstream_axis.set_title("downstream_harm")
    for axis, metric in zip(flat_axes[1:], METRIC_OUTPUT_IDS, strict=False):
        for perturbation in PERTURBATIONS:
            rows = [
                row for row in figure1_rows
                if row["metric_output_id"] == metric
                and row["perturbation_id"] == perturbation
            ]
            x = [row["mean_metric_harm"] for row in rows]
            y = [row["mean_downstream_harm"] for row in rows]
            if all(value is not None for value in (*x, *y)):
                axis.plot(x, y, marker="o", color=colors[perturbation], label=perturbation)
        axis.set_title(metric)
    for axis in flat_axes[14:]:
        axis.set_axis_off()
    figure.tight_layout()
    for kind in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
        payloads[f"figure1_d1_protocol_a_full_domain.{kind}"] = stream.getvalue()
    plt.close(figure)
    figure, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
    for axis, field, title, color in zip(
        axes, ("ag", "acc_cross", "d_ag", "d_acc"),
        ("AG", "Acc-cross", "D_AG", "D_Acc"),
        ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"), strict=True,
    ):
        values = [row.get(field) for row in figure2_rows]
        ypos = np.arange(len(METRIC_OUTPUT_IDS))
        centers = [0.0 if value is None else value for value in values]
        errors = []
        for row, value in zip(figure2_rows, values, strict=True):
            interval = row.get(f"{field}_interval")
            errors.append(
                0.0 if value is None or interval is None
                else max(abs(value - interval[0]), abs(interval[1] - value))
            )
        axis.barh(
            ypos, centers, xerr=errors,
            color=[color if value is not None else "#bdbdbd" for value in values],
        )
        axis.set_title(title)
        axis.set_yticks(ypos, METRIC_OUTPUT_IDS if axis is axes[0] else [])
    figure.tight_layout()
    for kind in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
        payloads[f"figure2_d1_protocol_a_full_domain.{kind}"] = stream.getvalue()
    plt.close(figure)
    return MappingProxyType(payloads)


def _condition_summary(
    predictions: Sequence[Mapping[str, object]], config: Phase4D1ProtocolAConfig
) -> tuple[Mapping[str, object], ...]:
    indexed: dict[str, list[Mapping[str, object]]] = {}
    for row in predictions:
        indexed.setdefault(str(row["condition_id"]), []).append(row)
    rows = []
    for condition_id, _, _ in _condition_specs():
        selected = indexed[condition_id]
        per_class = [
            float(np.mean([row["correct"] for row in selected if int(row["true_class"]) == label]))
            for label in range(config.class_count)
        ]
        f1_values = []
        for label in range(config.class_count):
            tp = sum(row["true_class"] == label and row["predicted_class"] == label for row in selected)
            fp = sum(row["true_class"] != label and row["predicted_class"] == label for row in selected)
            fn = sum(row["true_class"] == label and row["predicted_class"] != label for row in selected)
            f1_values.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
        rows.append({
            "condition_id": condition_id, "prediction_count": len(selected),
            "top1_macro_class_accuracy": float(np.mean(per_class)),
            "top1_micro_accuracy": float(np.mean([row["correct"] for row in selected])),
            "macro_f1": float(np.mean(f1_values)),
        })
    return tuple(rows)


def build_phase4_d1_protocol_a_from_inputs(
    output_dir: Path, *, inputs: D1ProtocolAInputs, config: Phase4D1ProtocolAConfig,
    worker_count: int, bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4D1ProtocolASummary:
    if not config.synthetic_fixture and (bootstrap_resamples is not None or sign_flip_resamples is not None):
        raise Phase4D1ProtocolAError("inference resamples are test-only overrides")
    bridge = (
        MappingProxyType({
            "authorized_step15_payloads": tuple(STEP15_AUTHORIZED),
            "bridge_row_count": len(inputs.record_ids) * 41,
            "condition_bridge_sha256": "synthetic",
            "metric_row_count": len(inputs.record_ids) * 41 * 13,
            "cwt_row_count": len(inputs.record_ids) * 41,
            "step13_full_domain_state": "evaluable",
            "step13_peak_common_state": "not_evaluable_coverage",
        })
        if config.synthetic_fixture else validate_d1_parent_authorities(config)
    )
    science = rematerialize_d1_protocol_a_science(
        inputs, config, worker_count=worker_count
    )
    models = fit_d1_protocol_a_models(inputs, config)
    predictions, seed_rows, cells = predict_d1_protocol_a(
        models, science, config, inputs=inputs
    )
    projection = aggregate_d1_protocol_a(
        predictions, science.metric_values, config,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )
    expected = config.document["expected"]
    actual = {
        "condition_count": len(science.record_conditions),
        "metric_value_count": len(science.metric_values),
        "cwt_receipt_count": len(science.peak_receipts),
        "model_cell_count": len(cells),
        "lr_candidate_fit_count": sum(len(model.validation_scores) for model in models),
        "prediction_row_count": len(predictions),
        "seed_class_condition_count": len(seed_rows),
        "class_observation_count": len(projection.class_observations),
        "alignment_result_count": len(projection.alignment_results),
        "bootstrap_result_count": len(projection.bootstrap_results),
        "sign_flip_result_count": len(projection.sign_flip_results),
        "holm_family_row_count": len(projection.holm_family),
    }
    if any(actual[key] != int(expected[key]) for key in actual):
        raise Phase4D1ProtocolAError(f"artifact: row-count mismatch {actual}")
    condition_digest = _canonical_ledger_digest(iter(science.record_conditions))
    preflight = {
        "status": "pass",
        "source_record_count": inputs.source_record_count,
        "model_cell_count": inputs.model_cell_count or len(config.model_seeds),
        "role_occurrence_count": inputs.role_occurrence_count,
        "source_ledger_sha256": inputs.source_ledger_sha256,
        "model_ledger_sha256": inputs.model_ledger_sha256,
        "role_ledger_sha256": inputs.role_ledger_sha256,
        "role_overlap_detected": inputs.role_overlap_detected,
        "condition_count": len(science.record_conditions),
        "condition_ledger_sha256": condition_digest,
        "condition_mismatch_count": 0,
        "train_validation_matrix_ready": True,
        "test_projection_ready": True,
        "inherited_rulings": {
            "p01_p05": "not_evaluable_coverage",
            "p06_p07": "structurally_ineligible_missing_explicit_baseline",
        },
    }
    figures = render_d1_protocol_a_figures(projection, config)
    condition_rows = _condition_summary(predictions, config)
    table_fields = (
        "metric_output_id", "state", "ag", "ag_raw", "ag_interval",
        "acc_cross", "acc_interval", "d_ag", "d_ag_interval",
        "d_acc", "d_acc_interval",
    )
    code_identity = _sha_file(ROOT / "rpe/runner/phase4_d1_protocol_a.py")
    payloads: dict[str, bytes] = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": _canonical(dict(bridge)),
        "preflight.json": _canonical(preflight),
        "model_cells.jsonl": _jsonl(cells),
        "validation_scores.jsonl": _jsonl([row for model in models for row in model.validation_scores]),
        "predictions.jsonl": _jsonl([{**row, "code_identity": code_identity} for row in predictions]),
        "seed_class_conditions.jsonl": _jsonl(seed_rows),
        "condition_summary.csv": _csv(
            condition_rows, ("condition_id", "prediction_count",
                             "top1_macro_class_accuracy", "top1_micro_accuracy",
                             "macro_f1"),
        ),
        "class_observations.jsonl": _jsonl(projection.class_observations),
        "alignment_results.jsonl": _jsonl(projection.alignment_results),
        "bootstrap_results.jsonl": _jsonl(projection.bootstrap_results),
        "sign_flip_results.jsonl": _jsonl(projection.sign_flip_results),
        "holm_family.jsonl": _jsonl(projection.holm_family),
        **figures,
        "d1_protocol_a_full_domain_secondary_table.csv": _csv(
            [{field: row.get(field) for field in table_fields} for row in projection.alignment_results],
            table_fields,
        ),
    }
    mse_state = next(
        row["state"] for row in projection.alignment_results
        if row["metric_output_id"] == "mse"
    )
    status = "complete" if mse_state == "complete" else "failed"
    run_id = _sha(
        b"rpe-phase4-d1-protocol-a-v1\0" + config.raw_bytes
        + _canonical({
            "source_ledger_sha256": inputs.source_ledger_sha256,
            "model_ledger_sha256": inputs.model_ledger_sha256,
            "role_ledger_sha256": inputs.role_ledger_sha256,
            "bridge_sha256": bridge.get(
                "condition_bridge_sha256", "orchestration-test-bridge"
            ),
        })
    )
    manifest = {
        "schema_version": "phase4-d1-protocol-a-full-domain-artifact-v1",
        "run_id": run_id, "status": status, "endpoint_count": 1,
        "active_perturbation_count": 5, "condition_count": 41,
        "model_seed_count": len(config.model_seeds),
        "test_record_count": len(inputs.record_ids),
        "model_fit_count": len(models),
        "lr_candidate_fit_count": len(models) * 4,
        "prediction_row_count": len(predictions),
        "seed_class_condition_count": len(seed_rows),
        "class_observation_count": len(projection.class_observations),
        "metric_states": {
            row["metric_output_id"]: row["state"]
            for row in projection.alignment_results
        },
        "inherited_rulings": preflight["inherited_rulings"],
        "claim_boundary": _CLAIM_BOUNDARY,
        "authorities": config.document.get("authorities", {}),
        "code_authority": config.document.get("code_authority", {}),
        "environment_authority": config.document.get("environment_authority", {}),
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
    }
    payloads["manifest.json"] = _canonical(manifest)
    if set(payloads) != set(ARTIFACT_PAYLOAD_FILES):
        raise Phase4D1ProtocolAError("artifact: payload order mismatch")
    payloads = {name: payloads[name] for name in ARTIFACT_PAYLOAD_FILES}
    output = Path(output_dir)
    if output.exists():
        raise Phase4D1ProtocolAError("output: append-only target already exists")
    output.mkdir(parents=True)
    for name, value in payloads.items():
        (output / name).write_bytes(value)
    marker = "complete.json" if status == "complete" else "failed.json"
    (output / marker).write_bytes(_canonical({"run_id": run_id, "status": status}))
    sums = b"".join(
        f"{_sha_file(output / name)}  {name}\n".encode("utf-8")
        for name in (*ARTIFACT_PAYLOAD_FILES, marker)
    )
    (output / "SHA256SUMS").write_bytes(sums)
    return Phase4D1ProtocolASummary(
        output, run_id, status, len(inputs.record_ids), len(models)
    )


def build_phase4_d1_protocol_a(
    output_root: Path, *, worker_count: int = 16
) -> Phase4D1ProtocolASummary:
    config = load_phase4_d1_protocol_a_config(DEFAULT_CONFIG)
    bridge = validate_d1_parent_authorities(config)
    inputs = reconstruct_d1_protocol_a_inputs(
        ROOT / "data/unified/bacteria_id_reference", config
    )
    target_id = _sha(
        b"rpe-phase4-d1-protocol-a-v1\0" + config.raw_bytes
        + _canonical({
            "source_ledger_sha256": inputs.source_ledger_sha256,
            "model_ledger_sha256": inputs.model_ledger_sha256,
            "role_ledger_sha256": inputs.role_ledger_sha256,
            "bridge_sha256": bridge.get(
                "condition_bridge_sha256", "orchestration-test-bridge"
            ),
        })
    )
    target = Path(output_root) / f"phase4-d1-protocol-a-full-domain-{target_id}"
    return build_phase4_d1_protocol_a_from_inputs(
        target, inputs=inputs, config=config, worker_count=worker_count
    )


__all__ = [
    "ARTIFACT_PAYLOAD_FILES", "D1ProtocolAInputs", "D1FrozenModel",
    "D1OutcomeProjection", "D1SharedScience", "Phase4D1ProtocolAConfig",
    "Phase4D1ProtocolAError", "Phase4D1ProtocolASummary",
    "aggregate_d1_protocol_a", "build_phase4_d1_protocol_a",
    "build_phase4_d1_protocol_a_from_inputs", "fit_d1_protocol_a_models",
    "load_phase4_d1_protocol_a_config", "make_synthetic_d1_protocol_a_config",
    "make_synthetic_d1_protocol_a_inputs", "parse_phase4_d1_protocol_a_config",
    "predict_d1_protocol_a", "reconstruct_d1_protocol_a_inputs",
    "rematerialize_d1_protocol_a_science", "render_d1_protocol_a_figures",
    "validate_d1_parent_authorities",
]
