"""Structurally independent verifier for Phase 4 D1 Protocol-A artifacts.

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
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping, Sequence

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
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase4_d1_protocol_a_authority import CONFIG_BYTES, CONFIG_SHA256
from rpe.runner.phase4_d2_eligibility import (
    _canonical_json_bytes as _step13_canonical,
    load_phase4_d2_eligibility_config,
    project_d2_support,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase4/configs/d1_protocol_a_full_domain_v1.json"
SCHEMA_VERSION = "phase4-d1-protocol-a-full-domain-config-v1"
EXPERIMENT_ID = "phase4-d1-protocol-a-full-domain-v1"
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
    "config.json", "authority_bridge.json", "preflight.json",
    "model_cells.jsonl", "validation_scores.jsonl", "predictions.jsonl",
    "seed_class_conditions.jsonl", "condition_summary.csv",
    "class_observations.jsonl", "alignment_results.jsonl",
    "bootstrap_results.jsonl", "sign_flip_results.jsonl", "holm_family.jsonl",
    "figure1_d1_protocol_a_full_domain.png",
    "figure1_d1_protocol_a_full_domain.svg",
    "figure1_d1_protocol_a_full_domain_data.csv",
    "figure2_d1_protocol_a_full_domain.png",
    "figure2_d1_protocol_a_full_domain.svg",
    "figure2_d1_protocol_a_full_domain_data.csv",
    "d1_protocol_a_full_domain_secondary_table.csv", "manifest.json",
)
STEP13_RUN_ID = "phase4-d2-protocol-a-full-domain-eligibility-0f43f329bfc6a7a1ff7232a336d115849b06dbca884de28d71f6b28502598ef8"
STEP13_SUMS = "b41638b4e246326b155ef1c6169b57d93c68266b47a41b9d16f05dc516546e9e"
STEP13_CONFIG = "f92427c2f18ab445db2bb54d5ea5a97cce21b05dd6ee80f50f3ed4082285a15b"
STEP13_ROOT = ROOT / "results/phase4/d2_protocol_a_full_domain_eligibility_v1" / STEP13_RUN_ID
STEP15_ROOT = ROOT / "results/phase4/d2_protocol_a_full_domain_v1/8e1e76c497de232f0901bea0278c114de6c4c410631a6edd0f799d120d02e8ef"
STEP15_RUN_ID = "67fedde2a92c0ee57c504830ebefc23c2ca0ba6dbebc33a026e6cee2279628c6"
STEP15_SUMS = "389e56c3eb7c261b3b67d43ab47a18a3d613ce703fdb6cc8d22f1868841f7073"
STEP15_ALLOWED = MappingProxyType({
    "record_conditions.jsonl": "ef4792f952dec40ac940b5321a65c57872c672ad8643d084a5c78f985c74f298",
    "metric_values.jsonl": "fc4f1013d729b69d444974106cd90e282028bd377e73cec0d6a8cbd99a74828c",
    "peak_receipts.jsonl": "916dfdfa0ca120bd184eda39a1ee4a81ee171814f263167e223b7b9c3dbd3d94",
})
BRIDGE_DIGEST = "031b4f4f30235a6237ad1a6f89b03fb9f2d1b724f55f7c8ae9d52f402eabaa7c"
TEST_DIGEST = "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"
SOURCE_DIGEST = "a44e71531c5a639904390ca73738d035e0c3294766f46a8def82b2f9bd9fec1b"
MODEL_DIGEST = "1b77f2825e25265382597e64218f4a2771011cd9cc8f86c4c3f6e729e6c2c486"
ROLE_DIGEST = "211b087e21f3b511acd6acaa800b693ffef0db7c1aaa354c902d146ee2c2090c"
SUPPORT_F64 = "c682ec93f843362e1bb272d11037c4e0f33844dac47de0591496958dfca35dd6"
SUPPORT_F32 = "6bbef8640905114e63357df00e2bc5488ccde6dd594f0778bb167b0bafdb9c59"
NATIVE_F64 = "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d"
CLAIM_BOUNDARY = "local_execution_artifact_redistribution_not_cleared"
CODE_PATHS = (
    "rpe/alignment/bulk.py", "rpe/alignment/contracts.py", "rpe/alignment/core.py", "rpe/alignment/inference.py",
    "rpe/downstream/bacteria_id.py", "rpe/evaluation/contracts.py", "rpe/methods/catalog.py",
    "rpe/methods/classical/peaks.py", "rpe/metrics/fidelity.py", "rpe/metrics/peak.py",
    "rpe/metrics/reference_free.py", "rpe/metrics/transport.py", "rpe/perturb/axis_transform.py",
    "rpe/perturb/baseline_distortion.py", "rpe/perturb/contracts.py", "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py", "rpe/perturb/sweep.py", "rpe/runner/d1_bacteria_id.py",
    "rpe/runner/phase1_config.py", "rpe/runner/phase1_gates.py", "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py", "rpe/runner/phase1_types.py", "rpe/runner/phase4_d2_eligibility.py",
    "rpe/runner/phase4_d1_protocol_a.py",
    "rpe/runner/phase4_d1_protocol_a_verifier.py", "tools/run_phase4_d1_protocol_a.py",
)


class Phase4D1ProtocolAVerifierError(ValueError):
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
    train_values: Mapping[int, np.ndarray] | None
    train_labels: Mapping[int, np.ndarray] | None
    validation_values: Mapping[int, np.ndarray] | None
    validation_labels: Mapping[int, np.ndarray] | None
    model_cells: Mapping[int, Mapping[str, object]] | None
    support_axis: np.ndarray | None
    native_spectra: tuple[Spectrum1D, ...]
    native_labels: np.ndarray | None
    source_count: int
    model_count: int
    role_count: int
    source_digest: str
    model_digest: str
    role_digest: str
    support_f32: str
    support_f64: str
    role_overlap: bool
    force_metric_failure: bool


@dataclass(frozen=True)
class _Model:
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
class _ParentBundle:
    bridge: Mapping[str, object]
    record_conditions: tuple[Mapping[str, object], ...]
    metric_values: tuple[Mapping[str, object], ...]
    peak_receipts: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class Phase4D1ProtocolASummary:
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
    raise Phase4D1ProtocolAVerifierError(f"JSON: unsupported {type(value).__name__}")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _id_digest(values: Sequence[str]) -> str:
    return _sha_bytes(("\n".join(values) + "\n").encode("utf-8"))


def _ledger_digest(rows: Iterator[Mapping[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(_canonical(row))
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
    support = json.loads(
        (ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json").read_bytes()
    )["support_grid"]
    required = {
        "protocol": "A", "tier": "full_domain_core", "claim_boundary": CLAIM_BOUNDARY,
        "model_seeds": list(SEEDS), "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS), "metric_output_ids": list(METRICS),
        "metric_manifest": [{"output_id": name, "preferred_direction": directions[name]} for name in METRICS],
        "cwt_system_id": "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511",
        "denominators": {"class_count": 30, "test_record_count": 3000, "source_record_count": 66000, "model_cell_count": 5, "model_role_occurrence_count": 330000},
        "expected": {
            "operator_cell_count": 15000, "apply_check_count": 135000, "condition_count": 123000,
            "metric_value_count": 1599000, "cwt_receipt_count": 123000, "model_cell_count": 5, "pca_fit_count": 5,
            "lr_candidate_fit_count": 20, "frozen_model_condition_call_count": 205, "prediction_row_count": 615000,
            "seed_class_condition_count": 6150, "class_observation_count": 15600, "alignment_result_count": 13,
            "bootstrap_result_count": 12, "sign_flip_result_count": 24, "holm_family_row_count": 24,
            "figure1_row_count": 520, "figure2_row_count": 13, "secondary_table_row_count": 13,
            "cross_perturbation_pair_count_per_class": 640, "cross_perturbation_pair_count_per_metric": 19200,
            "configured_payload_count": 21, "terminal_marker_count": 1, "sha256sums_count": 1, "artifact_file_count": 23,
        },
        "support_grid": support,
        "support_point_count": 997,
        "p10": {"correlation_length_cm1": 20, "memory_budget_bytes": 68719476736, "peak_estimate_formula": "32*N^2+64*N+2^30"},
        "model_recipe": {
            "projection": "float64_interpolation_then_one_float32_cast_no_normalization",
            "pca": {"n_components": 20, "random_state": "model_seed", "svd_solver": "randomized", "whiten": False},
            "logistic_regression": {"c_grid": [0.01, 0.1, 1.0, 10.0], "class_weight": None, "max_iter": 1000, "random_state": "model_seed", "regularization": "l2", "solver": "lbfgs", "tol": 0.0001},
            "selection": "first_strict_maximum_validation_top1_accuracy_lowest_c_on_tie", "refit_with_validation": False, "test_condition_count": 41,
        },
        "inference": {"bootstrap_resamples": 2000, "sign_flip_resamples": 100000, "random_seed": 20260817, "confidence_level": 0.95, "holm_alpha": 0.05, "cluster_unit": "class_label", "d_ag_formula": "AG_MSE-AG_candidate", "d_acc_formula": "Acc_candidate-Acc_MSE", "class_contribution_reduction": {"D_AG": "sum", "D_Acc": "mean"}, "holm_slot_count": 24},
        "inherited_rulings": {"p01_p05": "not_evaluable_coverage", "p06_p07": "structurally_ineligible_missing_explicit_baseline"},
        "figure_contract": {"colors": {"p08": "#1f77b4", "p09": "#ff7f0e", "p10": "#2ca02c", "p11": "#d62728", "p12": "#9467bd"}, "dpi": 300, "figure1_inches": [12, 12], "figure1_layout": [4, 4], "figure2_inches": [14, 8], "figure2_layout": [1, 4], "font_family": "DejaVu Sans", "line_width": 1.5, "marker": "o", "padding_fraction": 0.05, "svg_hashsalt": "rpe-phase4-d1-protocol-a-v1"},
        "artifact_contract": {"checksum_file": "SHA256SUMS", "checksum_scope": "all_payloads_and_exactly_one_terminal_marker", "payload_files": list(PAYLOADS), "terminal_marker_rule": "exactly_one", "terminal_markers": ["complete.json", "failed.json"]},
        "artifact_payload_files": list(PAYLOADS), "trust_anchor": {"config_authority_relative_path": "rpe/runner/phase4_d1_protocol_a_authority.py", "config_binds_authority": False, "direction": "authority_to_config_only"},
    }
    for name, value in required.items():
        if root.get(name) != value:
            raise Phase4D1ProtocolAVerifierError(f"config: frozen {name} authority mismatch")

    authorities = root.get("authorities")
    expected_keys = {"parent_plan", "phase4_preregistration", "alignment_core_design", "step12_design", "step13_report", "step13_config", "step15_report", "step19_report", "step20_design", "sweep", "phase1_core_config", "classical_catalog", "d1_phase05_protocol", "d1_phase05_complete_report", "d1_recipe_config", "bacteria_id_sha256sums", "bacteria_id_retained_snapshot", "step13_parent", "step15_config", "step15_parent"}
    if not isinstance(authorities, Mapping) or set(authorities) != expected_keys:
        raise Phase4D1ProtocolAVerifierError("config: scientific authority inventory mismatch")
    for value in authorities.values():
        if not isinstance(value, Mapping):
            raise Phase4D1ProtocolAVerifierError("config: malformed authority")
        path = ROOT / str(value.get("path", ""))
        if not path.is_file() or path.stat().st_size != int(value.get("bytes", -1)) or _sha_file(path) != value.get("sha256"):
            raise Phase4D1ProtocolAVerifierError(f"config: live authority mismatch for {path}")
    identities = root.get("frozen_identities", {})
    required_identities = {"source_ledger_sha256": SOURCE_DIGEST, "model_ledger_sha256": MODEL_DIGEST, "role_ledger_sha256": ROLE_DIGEST, "test_record_ids_sha256": TEST_DIGEST, "native_axis_increasing_f64_sha256": NATIVE_F64, "support_axis_f64_sha256": SUPPORT_F64, "support_axis_f32_sha256": SUPPORT_F32, "condition_bridge_sha256": BRIDGE_DIGEST}
    if not isinstance(identities, Mapping) or any(identities.get(key) != value for key, value in required_identities.items()) or identities.get("model_seed_ids") != list(SEEDS) or identities.get("support_point_count") != 997:
        raise Phase4D1ProtocolAVerifierError("config: frozen dataset identity mismatch")
    if root.get("environment_authority") != _observed_environment():
        raise Phase4D1ProtocolAVerifierError("config: environment authority mismatch")
    code = root.get("code_authority")
    if not isinstance(code, Mapping) or code != _observed_code():
        raise Phase4D1ProtocolAVerifierError("config: code authority mismatch")


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
        raise Phase4D1ProtocolAVerifierError(f"config: {error}") from error
    if raw != _canonical(document):
        raise Phase4D1ProtocolAVerifierError("config: canonical JSON required")
    synthetic = bool(document.get("synthetic_fixture", False))
    if require_frozen_identity and (synthetic or len(raw) != CONFIG_BYTES or _sha_bytes(raw) != CONFIG_SHA256):
        raise Phase4D1ProtocolAVerifierError("config: frozen config identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D1ProtocolAVerifierError("config: schema identity mismatch")
    if tuple(document.get("active_perturbation_ids", ())) != PERTURBATIONS:
        raise Phase4D1ProtocolAVerifierError("config: perturbation order mismatch")
    if tuple(float(item) for item in document.get("alpha_grid", ())) != ALPHAS or tuple(document.get("metric_output_ids", ())) != METRICS:
        raise Phase4D1ProtocolAVerifierError("config: alpha/metric manifest mismatch")
    if tuple(document.get("artifact_payload_files", ())) != PAYLOADS:
        raise Phase4D1ProtocolAVerifierError("config: payload inventory mismatch")
    denominator = document.get("denominators", {})
    expected = document.get("expected", {})
    if not isinstance(denominator, Mapping) or not isinstance(expected, Mapping):
        raise Phase4D1ProtocolAVerifierError("config: denominators and expected must be objects")
    classes, tests = int(denominator.get("class_count", 0)), int(denominator.get("test_record_count", 0))
    seeds = tuple(int(item) for item in document.get("model_seeds", SEEDS))
    required = {
        "condition_count": tests * 41, "metric_value_count": tests * 41 * 13,
        "cwt_receipt_count": tests * 41, "model_cell_count": len(seeds),
        "lr_candidate_fit_count": 4 * len(seeds), "prediction_row_count": len(seeds) * tests * 41,
        "seed_class_condition_count": len(seeds) * classes * 41,
        "class_observation_count": classes * 40 * len(METRICS),
        "alignment_result_count": len(METRICS),
        "bootstrap_result_count": len(METRICS) - 1,
        "sign_flip_result_count": 2 * (len(METRICS) - 1),
        "holm_family_row_count": 2 * (len(METRICS) - 1),
    }
    if classes < 1 or tests < classes or not seeds or any(int(expected.get(key, -1)) != value for key, value in required.items()):
        raise Phase4D1ProtocolAVerifierError("config: frozen denominators mismatch")
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
            int(getattr(value, "source_record_count", 0)),
            int(getattr(value, "model_cell_count", 0)),
            int(getattr(value, "role_occurrence_count", 0)),
            str(getattr(value, "source_ledger_sha256", "")),
            str(getattr(value, "model_ledger_sha256", "")),
            str(getattr(value, "role_ledger_sha256", "")),
            str(getattr(value, "support_axis_f32_sha256", "")),
            str(getattr(value, "support_axis_f64_sha256", "")),
            bool(getattr(value, "role_overlap_detected", False)),
            bool(getattr(value, "force_metric_failure", False)),
        )
    except AttributeError as error:
        raise Phase4D1ProtocolAVerifierError("inputs: missing D1 input contract fields") from error
    if output.test_values.ndim != 2 or output.test_values.shape != (len(output.record_ids), config.support_points):
        raise Phase4D1ProtocolAVerifierError("inputs: projected test matrix shape mismatch")
    if output.test_labels.shape != (len(output.record_ids),) or output.model_seeds != config.seeds:
        raise Phase4D1ProtocolAVerifierError("inputs: label/seed identity mismatch")
    return output


def _fit_models(inputs: _Inputs, config: _Config) -> tuple[_Model, ...]:
    models: list[_Model] = []
    if any(value is None for value in (
        inputs.train_values, inputs.train_labels,
        inputs.validation_values, inputs.validation_labels,
    )):
        raise Phase4D1ProtocolAVerifierError("model: train/validation matrices required")
    for seed in config.seeds:
        train = np.asarray(inputs.train_values[seed], dtype="<f4")
        train_y = np.asarray(inputs.train_labels[seed], dtype="<i8")
        valid = np.asarray(inputs.validation_values[seed], dtype="<f4")
        valid_y = np.asarray(inputs.validation_labels[seed], dtype="<i8")
        if not config.synthetic and (
            train.shape != (62700, 997) or valid.shape != (300, 997)
        ):
            raise Phase4D1ProtocolAVerifierError(
                "model: exact D1 role cardinality mismatch"
            )
        if (
            set(train_y.tolist()) != set(range(config.class_count))
            or set(valid_y.tolist()) != set(range(config.class_count))
        ):
            raise Phase4D1ProtocolAVerifierError("model: missing class")
        pca = PCA(
            n_components=min(20, train.shape[0], train.shape[1]),
            svd_solver="randomized", whiten=False, random_state=seed,
        )
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            train_x = pca.fit_transform(train)
            valid_x = pca.transform(valid)
        if observed or not np.isfinite(train_x).all() or not np.isfinite(valid_x).all():
            raise Phase4D1ProtocolAVerifierError("model: PCA warning/nonfinite failure")
        selected: LogisticRegression | None = None
        selected_c: float | None = None
        best = -math.inf
        scores: list[Mapping[str, object]] = []
        for c_value in (0.01, 0.1, 1.0, 10.0):
            candidate = LogisticRegression(
                C=c_value, l1_ratio=0.0, solver="lbfgs", max_iter=1000,
                tol=1e-4, class_weight=None, random_state=seed,
            )
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                candidate.fit(train_x, train_y)
            if observed:
                raise Phase4D1ProtocolAVerifierError("model: LR warning")
            predicted = candidate.predict(valid_x)
            if predicted.shape != valid_y.shape:
                raise Phase4D1ProtocolAVerifierError("model: invalid prediction shape")
            score = float(np.mean(predicted == valid_y))
            scores.append({
                "model_seed": seed, "c": c_value,
                "validation_top1_accuracy": score,
            })
            if score > best:
                selected, selected_c, best = candidate, c_value, score
        if selected is None or selected_c is None:
            raise Phase4D1ProtocolAVerifierError("model: no candidate selected")
        models.append(_Model(
            seed, float(selected_c), pca, selected, tuple(scores),
            _array_sha(train, "<f4"), _array_sha(valid, "<f4"),
            _array_sha(train_x), _array_sha(valid_x),
        ))
    return tuple(models)


def _receipt_sha(value: object) -> str:
    return _sha_bytes(_step13_canonical(value))


def _source(order: int, record_id: str, label: int, spectrum: Spectrum1D) -> Phase1Source:
    return Phase1Source(
        selection=SelectedSourceRow(order, record_id, spectrum.sample_id or record_id, label, f"bacteria-{label}", f"native::{_array_sha(spectrum.axis_cm1)}"),
        spectrum=spectrum, original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(spectrum.axis_cm1, "<f4"), source_intensity_float32_sha256=_array_sha(spectrum.intensity, "<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1), normalized_intensity_float64_sha256=_array_sha(spectrum.intensity),
        provenance=MappingProxyType({"license": None, "license_status": "not_stated", "retrieved_date": "2026-08-23", "sha256": "0" * 64, "source_artifact": "bacteria_id_reference", "source_url": "local://bacteria_id_reference"}),
    )


def _metric_objects() -> Mapping[str, object]:
    return MappingProxyType({"mse": MSEMetric(), "rmse": RMSEMetric(), "mae": MAEMetric(), "sam": SAMMetric(), "pearson_r": PearsonRMetric(), "nmse": NMSEMetric(), "wasserstein_1_cm1": Wasserstein1Metric(), "is_like_structure_to_noise": ISLikeStructureToNoiseMetric()})


def _projection(
    spectrum: Spectrum1D, support: np.ndarray, eligibility: object | None = None
) -> np.ndarray:
    if eligibility is not None:
        return project_d2_support(spectrum, eligibility)
    axis, intensity = np.asarray(spectrum.axis_cm1, dtype="<f8"), np.asarray(spectrum.intensity, dtype="<f8")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.size != intensity.size or axis[0] > support[0] or axis[-1] < support[-1]:
        raise Phase4D1ProtocolAVerifierError("science: invalid support interpolation")
    return np.ascontiguousarray(np.interp(support, axis, intensity), dtype="<f4")


def _item_sha(item: object) -> str:
    result = item.result
    return _receipt_sha({"alpha_float64_le_hex": item.alpha_float64_le_hex, "axis_behavior": result.axis_behavior.value, "axis_changed": result.axis_changed, "diagnostics": result.diagnostics, "intensity_changed": result.intensity_changed, "output_axis_sha256": _array_sha(result.output.axis_cm1), "output_intensity_sha256": _array_sha(result.output.intensity), "output_spectrum_id": result.output.spectrum_id, "perturbation_id": result.perturbation_id, "source_spectrum_id": result.source_spectrum_id, "state_digest": result.state_digest})


def _science_record(
    order: int, record_id: str, label: int, spectrum: Spectrum1D,
    support: np.ndarray, sweep: object, phase1: object, cwt: object | None,
    admission: P10MemoryAdmission, *, include_measurements: bool,
    eligibility: object | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    source = _source(order, record_id, label, spectrum)
    matrices = {"alpha0": _projection(spectrum, support, eligibility)}
    spectra = {"alpha0": spectrum}
    hashes = {"alpha0": _array_sha(matrices["alpha0"], "<f4")}
    for perturbation in PERTURBATIONS:
        cell = run_perturbation_cell(source, perturbation, phase1, sweep, p10_admission=admission if perturbation == "p10" else None)
        if cell.status is not CellStatus.COMPLETE: raise Phase4D1ProtocolAVerifierError(f"science: {record_id}/{perturbation} native gate failure")
        alpha0 = next((item for item in cell.records if item.result.alpha == 0.0), None)
        if alpha0 is None or not np.array_equal(alpha0.result.output.axis_cm1, spectrum.axis_cm1) or not np.array_equal(alpha0.result.output.intensity, spectrum.intensity): raise Phase4D1ProtocolAVerifierError("science: alpha0 identity failure")
        for item in cell.records:
            if item.result.alpha == 0.0: continue
            condition = f"{perturbation}:{np.float64(item.result.alpha).tobytes().hex()}"
            spectra[condition] = item.result.output
            matrices[condition] = _projection(item.result.output, support, eligibility)
            hashes[condition] = _item_sha(item)
    reference = run_peak_detection_system(cwt, spectrum) if include_measurements else None
    if include_measurements and reference.status not in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}:
        raise Phase4D1ProtocolAVerifierError("science: reference CWT failure")
    conditions: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    peaks: list[dict[str, object]] = []
    scalar = _metric_objects() if include_measurements else {}
    peak_metric = PeakDetectionCurvesMetric() if include_measurements else None
    for condition, _, _ in _conditions():
        output, matrix = spectra[condition], matrices[condition]
        conditions.append({"record_order": order, "record_id": record_id, "condition_id": condition, "source_spectrum_id": spectrum.spectrum_id, "output_spectrum_id": output.spectrum_id, "axis_sha256": _array_sha(output.axis_cm1), "intensity_sha256": _array_sha(output.intensity), "projected_row_sha256": _array_sha(matrix, "<f4"), "result_sha256": hashes[condition], "state": "complete"})
        if not include_measurements:
            continue
        current = reference if condition == "alpha0" else run_peak_detection_system(cwt, output)
        if current.status not in {PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING}: raise Phase4D1ProtocolAVerifierError("science: CWT failure")
        peaks.append({"record_order": order, "record_id": record_id, "condition_id": condition, "state": current.status.value, "peak_list_sha256": current.peaks_sha256, "diagnostics_sha256": _receipt_sha(dict(current.diagnostics)), "warning_sha256": _receipt_sha(current.warnings)})
        for output_id, metric in scalar.items():
            result = evaluate_metric(metric, SingleSpectrumInput(output) if output_id == "is_like_structure_to_noise" else SpectrumPairInput(spectrum, output)); value = next(item for item in result.outputs if item.output_id == output_id)
            metrics.append({"record_order": order, "record_id": record_id, "condition_id": condition, "metric_output_id": output_id, "state": "complete", "value": float(value.value), "result_sha256": _receipt_sha(result), "diagnostics_sha256": _receipt_sha(result.diagnostics)})
        structure = evaluate_metric(peak_metric, PeakPairInput(tuple(item.to_peak1d() for item in reference.peaks), tuple(item.to_peak1d() for item in current.peaks), 2.0, (0.0,))); by_id = {item.output_id: item for item in structure.outputs}
        for output_id in METRICS[8:]: metrics.append({"record_order": order, "record_id": record_id, "condition_id": condition, "metric_output_id": output_id, "state": "complete", "value": float(by_id[output_id].value), "result_sha256": _receipt_sha(structure), "diagnostics_sha256": _receipt_sha(structure.diagnostics)})
    return matrices, conditions, metrics, peaks


_WORKER_SUPPORT: np.ndarray | None = None
_WORKER_ELIGIBILITY: object | None = None
_WORKER_SWEEP: object | None = None
_WORKER_PHASE1: object | None = None
_WORKER_CWT: object | None = None
_WORKER_P10_BUDGET: int | None = None


def _initialize_real_science_worker(p10_budget: int) -> None:
    """Load frozen scientific authorities in each spawn worker."""
    global _WORKER_SUPPORT, _WORKER_ELIGIBILITY, _WORKER_SWEEP, _WORKER_PHASE1, _WORKER_CWT, _WORKER_P10_BUDGET
    _WORKER_ELIGIBILITY = load_phase4_d2_eligibility_config(
        ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
    )
    _WORKER_SUPPORT = np.asarray(_WORKER_ELIGIBILITY.support_coordinates_cm1, dtype="<f8")
    _WORKER_P10_BUDGET = int(p10_budget)
    _WORKER_SWEEP = load_perturbation_sweep_config(ROOT / "experiments/shared/raman_perturbation_sweep_v1.json")
    _WORKER_PHASE1 = load_phase1_core_config(ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json")
    _WORKER_CWT = None


def _real_science_worker(job: tuple[int, str, int, Spectrum1D]) -> tuple[int, dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    if any(value is None for value in (_WORKER_SUPPORT, _WORKER_ELIGIBILITY, _WORKER_SWEEP, _WORKER_PHASE1, _WORKER_P10_BUDGET)):
        raise Phase4D1ProtocolAVerifierError("science worker: uninitialized")
    order, record_id, label, spectrum = job
    with threadpool_limits(limits=1, user_api="blas"):
        bundle = _science_record(
            order, record_id, label, spectrum, _WORKER_SUPPORT, _WORKER_SWEEP,
            _WORKER_PHASE1, None, P10MemoryAdmission(_WORKER_P10_BUDGET),
            include_measurements=False, eligibility=_WORKER_ELIGIBILITY,
        )
    return order, *bundle


def _real_science(
    inputs: _Inputs, config: _Config, worker_count: int
) -> tuple[
    Mapping[str, np.ndarray], tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...],
    Mapping[str, object],
]:
    if inputs.support_axis is None or inputs.native_labels is None or len(inputs.native_spectra) != len(inputs.record_ids) or not np.array_equal(inputs.native_labels, inputs.test_labels): raise Phase4D1ProtocolAVerifierError("science: native retained inputs required")
    support = np.asarray(inputs.support_axis, dtype="<f8")
    estimate = estimate_p10_peak_bytes(max(item.axis_cm1.size for item in inputs.native_spectra)); budget = 64 * 1024**3
    if estimate > budget: raise Phase4D1ProtocolAVerifierError("science: P10 memory admission failed")
    jobs = tuple((order, record_id, int(label), spectrum) for order, (record_id, label, spectrum) in enumerate(zip(inputs.record_ids, inputs.native_labels, inputs.native_spectra, strict=True)))
    if config.synthetic:
        sweep = load_perturbation_sweep_config(ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"); phase1 = load_phase1_core_config(ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"); catalog = load_classical_catalog(ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json")
        matches = [system for system in catalog.systems if system.system_id == "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"]
        if len(matches) != 1: raise Phase4D1ProtocolAVerifierError("science: CWT authority mismatch")
        admission = P10MemoryAdmission(budget)
        bundles = [
            _science_record(
                order, record_id, label, spectrum, support, sweep, phase1,
                matches[0], admission, include_measurements=True,
            )
            for order, record_id, label, spectrum in jobs
        ]
        bridge: Mapping[str, object] = MappingProxyType({
            "authorized_step15_payloads": tuple(STEP15_ALLOWED),
            "bridge_row_count": len(inputs.record_ids) * 41,
            "condition_bridge_sha256": "synthetic",
            "metric_row_count": len(inputs.record_ids) * 41 * 13,
            "cwt_row_count": len(inputs.record_ids) * 41,
            "step13_full_domain_state": "evaluable",
            "step13_peak_common_state": "not_evaluable_coverage",
        })
    else:
        parent = _load_parent_bundle(config)
        process_count = max(1, min(worker_count, len(jobs), budget // estimate))
        with ProcessPoolExecutor(
            max_workers=process_count, mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_real_science_worker, initargs=(budget,),
        ) as executor:
            completed = tuple(sorted(executor.map(_real_science_worker, jobs), key=lambda row: row[0]))
        bundles = [(matrices, condition_rows, metric_rows, peak_rows) for _, matrices, condition_rows, metric_rows, peak_rows in completed]
    by_condition = {condition: [] for condition, _, _ in _conditions()}; conditions: list[Mapping[str, object]] = []; metrics: list[Mapping[str, object]] = []; peaks: list[Mapping[str, object]] = []
    for matrices, condition_rows, metric_rows, peak_rows in bundles:
        for condition, matrix in matrices.items(): by_condition[condition].append(matrix)
        conditions.extend(condition_rows); metrics.extend(metric_rows); peaks.extend(peak_rows)
    if not config.synthetic:
        expected = {
            (str(row["record_id"]), str(row["condition_id"])): row
            for row in parent.record_conditions
        }
        if len(expected) != 123000 or len(conditions) != 123000:
            raise Phase4D1ProtocolAVerifierError("science: Step-15 condition count mismatch")
        for row in conditions:
            reference = expected.get((str(row["record_id"]), str(row["condition_id"])))
            if reference is None or any(
                reference.get(key) != row.get(key)
                for key in (
                    "record_order", "state", "axis_sha256",
                    "intensity_sha256", "projected_row_sha256",
                    "result_sha256",
                )
            ):
                raise Phase4D1ProtocolAVerifierError(
                    "science: Step-15 rematerialization mismatch"
                )
        metrics = list(parent.metric_values)
        peaks = list(parent.peak_receipts)
        bridge = parent.bridge
    if inputs.force_metric_failure:
        metrics = [
            ({**row, "state": "failed_runtime", "value": None}
             if row.get("metric_output_id") == "mse" else row)
            for row in metrics
        ]
    return (
        MappingProxyType({
            key: np.ascontiguousarray(value, dtype="<f4")
            for key, value in by_condition.items()
        }),
        tuple(conditions), tuple(metrics), tuple(peaks), bridge,
    )


def _predict(
    models: Sequence[_Model], matrices: Mapping[str, np.ndarray],
    inputs: _Inputs, config: _Config,
) -> tuple[
    tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    predictions: list[Mapping[str, object]] = []
    seed_rows: list[Mapping[str, object]] = []
    cells: list[Mapping[str, object]] = []
    for model in models:
        role = (inputs.model_cells or {}).get(model.seed, {})
        cells.append({
            "model_seed": model.seed,
            "selected_c": model.selected_c,
            "warning_state": "none",
            "train_record_ids_sha256": role.get(
                "train_record_ids_sha256",
                _id_digest(tuple(role.get("train_record_ids", ()))),
            ),
            "validation_record_ids_sha256": role.get(
                "validation_record_ids_sha256",
                _id_digest(tuple(role.get("validation_record_ids", ()))),
            ),
            "test_record_ids_sha256": role.get(
                "test_record_ids_sha256", _id_digest(inputs.record_ids)
            ),
            "train_matrix_sha256": model.train_sha,
            "validation_matrix_sha256": model.validation_sha,
            "pca_train_feature_sha256": model.train_feature_sha,
            "pca_validation_feature_sha256": model.validation_feature_sha,
            "model_state_sha256": _receipt_sha({
                "selected_c": model.selected_c,
                "classes": np.asarray(model.classifier.classes_, dtype="<i8").tolist(),
                "coef_sha256": _array_sha(model.classifier.coef_),
                "intercept_sha256": _array_sha(model.classifier.intercept_),
                "n_iter": np.asarray(model.classifier.n_iter_, dtype="<i8").tolist(),
                "pca_components_sha256": _array_sha(model.pca.components_),
                "pca_mean_sha256": _array_sha(model.pca.mean_),
                "pca_explained_variance_sha256": _array_sha(
                    model.pca.explained_variance_
                ),
            }),
        })
        for condition_id, matrix in matrices.items():
            predicted = model.classifier.predict(model.pca.transform(matrix))
            for order, (truth, choice) in enumerate(
                zip(inputs.test_labels, predicted, strict=True)
            ):
                predictions.append({
                    "model_seed": model.seed,
                    "model_identity": str(model.seed),
                    "condition_id": condition_id,
                    "record_order": order,
                    "record_id": inputs.record_ids[order],
                    "true_class": int(truth),
                    "predicted_class": int(choice),
                    "correct": bool(truth == choice),
                    "projected_row_sha256": _array_sha(matrix[order], "<f4"),
                    "config_sha256": config.sha256,
                })
            for label in range(config.class_count):
                mask = inputs.test_labels == label
                seed_rows.append({
                    "model_seed": model.seed,
                    "class_label": label,
                    "condition_id": condition_id,
                    "accuracy": float(
                        np.mean(predicted[mask] == inputs.test_labels[mask])
                    ),
                })
    return tuple(predictions), tuple(seed_rows), tuple(cells)


def _aggregate(
    predictions: Sequence[Mapping[str, object]],
    metrics: Sequence[Mapping[str, object]], config: _Config, *,
    bootstrap_resamples: int, sign_flip_resamples: int,
) -> tuple[
    tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    condition_ids = tuple(item[0] for item in _conditions())
    classes = tuple(range(config.class_count))
    accumulator: dict[tuple[int, int, str], list[bool]] = {}
    record_class: dict[int, int] = {}
    for row in predictions:
        order = int(row["record_order"])
        label = int(row["true_class"])
        if record_class.setdefault(order, label) != label:
            raise Phase4D1ProtocolAVerifierError(
                "aggregation: inconsistent record class"
            )
        key = (int(row["model_seed"]), label, str(row["condition_id"]))
        accumulator.setdefault(key, []).append(bool(row["correct"]))
    expected = {
        (seed, label, condition_id)
        for seed in config.seeds
        for label in classes
        for condition_id in condition_ids
    }
    if set(accumulator) != expected:
        raise Phase4D1ProtocolAVerifierError(
            "aggregation: exact seed/class/condition grid required"
        )
    seed_accuracy = {
        key: float(np.mean(values)) for key, values in accumulator.items()
    }
    metric_map: dict[tuple[int, str, str], Mapping[str, object]] = {}
    for row in metrics:
        key = (
            int(row["record_order"]), str(row["condition_id"]),
            str(row["metric_output_id"]),
        )
        if key in metric_map:
            raise Phase4D1ProtocolAVerifierError(
                f"aggregation: duplicate metric row {key}"
            )
        metric_map[key] = row
    class_indexes = {
        label: tuple(index for index, value in record_class.items() if value == label)
        for label in classes
    }
    observations: list[Mapping[str, object]] = []
    tables: dict[str, tuple[AlignmentObservation, ...] | None] = {}
    metric_states: dict[str, str] = {}
    for metric in METRICS:
        table: list[AlignmentObservation] = []
        complete = True
        pending: list[tuple[int, str, str, float, float | None, float]] = []
        for condition_id in condition_ids[1:]:
            perturbation, encoded = condition_id.split(":", 1)
            alpha = float(np.frombuffer(bytes.fromhex(encoded), dtype="<f8")[0])
            for label in classes:
                indexes = class_indexes[label]
                current_rows = [
                    metric_map.get((index, condition_id, metric)) for index in indexes
                ]
                baseline_rows = [
                    metric_map.get((index, "alpha0", metric)) for index in indexes
                ]
                if any(
                    row is None
                    or row.get("state") != "complete"
                    or row.get("value") is None
                    for row in (*current_rows, *baseline_rows)
                ):
                    complete = False
                    metric_harm = None
                else:
                    lower = DIRECTIONS[metric] == "lower_is_better"
                    metric_harm = float(np.mean([
                        float(current["value"]) - float(baseline["value"])
                        if lower
                        else float(baseline["value"]) - float(current["value"])
                        for baseline, current in zip(
                            baseline_rows, current_rows, strict=True
                        )
                    ]))
                current_acc = [
                    seed_accuracy[(seed, label, condition_id)]
                    for seed in config.seeds
                ]
                baseline_acc = [
                    seed_accuracy[(seed, label, "alpha0")]
                    for seed in config.seeds
                ]
                downstream = float(np.mean(baseline_acc) - np.mean(current_acc))
                pending.append((
                    label, condition_id, perturbation, alpha,
                    metric_harm, downstream,
                ))
        if complete:
            for label, condition_id, perturbation, alpha, harm, downstream in pending:
                table.append(AlignmentObservation(
                    str(label), perturbation, alpha, float(harm), downstream
                ))
                observations.append({
                    "metric_output_id": metric,
                    "class_label": label,
                    "condition_id": condition_id,
                    "perturbation_id": perturbation,
                    "alpha": alpha,
                    "metric_harm": harm,
                    "downstream_harm": downstream,
                    "state": "complete",
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
                    "metric_output_id": metric,
                    "class_label": label,
                    "condition_id": condition_id,
                    "perturbation_id": perturbation,
                    "alpha": alpha,
                    "metric_harm": None,
                    "downstream_harm": downstream,
                    "state": "not_evaluable_metric_incomplete",
                })
    alignment: list[Mapping[str, object]] = []
    bootstrap: list[Mapping[str, object]] = []
    signs: list[Mapping[str, object]] = []
    holm: list[Mapping[str, object]] = []
    reference = tables["mse"]
    if reference is not None:
        reference_gap = alignment_gap(reference)
        reference_accuracy = cross_perturbation_accuracy(reference)
        reference_bootstrap = bulk_paired_cluster_bootstrap(
            reference, reference, resamples=bootstrap_resamples,
            random_seed=20260817,
        )
        alignment.append({
            "metric_output_id": "mse", "state": "complete",
            "ag": reference_gap.alignment_gap,
            "ag_raw": reference_gap.raw_alignment_gap,
            "ag_interval": reference_bootstrap.reference_ag_interval,
            "acc_cross": reference_accuracy.accuracy,
            "acc_interval": reference_bootstrap.reference_acc_interval,
            "d_ag": None, "d_ag_interval": None,
            "d_acc": None, "d_acc_interval": None,
        })
    else:
        alignment.append({
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
    for metric in METRICS[1:]:
        candidate = tables[metric]
        if reference is None or candidate is None:
            bootstrap.append({
                "metric_output_id": metric,
                "state": "not_evaluable_metric_incomplete",
                "resamples": None,
            })
            for statistic in ("d_ag", "d_acc"):
                hypothesis = f"{metric}:{statistic}"
                p_values[hypothesis] = 1.0
                signs.append({
                    "metric_output_id": metric,
                    "statistic": statistic,
                    "state": "not_tested_metric_incomplete",
                    "contrast": None, "p_value": 1.0, "resamples": None,
                })
            continue
        comparison = compare_alignment(reference, candidate)
        inference = bulk_paired_cluster_bootstrap(
            reference, candidate, resamples=bootstrap_resamples,
            random_seed=20260817,
        )
        comparisons[metric] = comparison
        bootstraps[metric] = inference
        bootstrap.append({
            "metric_output_id": metric, "state": "complete",
            "resamples": bootstrap_resamples,
            "candidate_ag_interval": inference.candidate_ag_interval,
            "candidate_acc_interval": inference.candidate_acc_interval,
            "d_ag_interval": inference.d_ag_interval,
            "d_acc_interval": inference.d_acc_interval,
        })
        for statistic, contrast, contributions, reduction in (
            (
                "d_ag", comparison.d_ag,
                comparison.ag_contribution_differences, "sum",
            ),
            (
                "d_acc", comparison.d_acc,
                comparison.acc_contribution_differences, "mean",
            ),
        ):
            sign = paired_contribution_sign_flip(
                [item.value for item in contributions], aggregation=reduction,
                resamples=sign_flip_resamples, random_seed=20260817,
            )
            hypothesis = f"{metric}:{statistic}"
            p_values[hypothesis] = sign.p_value
            contrasts[hypothesis] = contrast
            signs.append({
                "metric_output_id": metric, "statistic": statistic,
                "state": "complete", "contrast": contrast,
                "p_value": sign.p_value, "resamples": sign_flip_resamples,
            })
    family = {
        row.hypothesis_id: row
        for row in holm_step_down(p_values, alpha=0.05)
    }
    for metric in METRICS[1:]:
        if metric in comparisons:
            comparison = comparisons[metric]
            inference = bootstraps[metric]
            alignment.append({
                "metric_output_id": metric, "state": "complete",
                "ag": comparison.candidate_gap.alignment_gap,
                "ag_raw": comparison.candidate_gap.raw_alignment_gap,
                "ag_interval": inference.candidate_ag_interval,
                "acc_cross": comparison.candidate_accuracy.accuracy,
                "acc_interval": inference.candidate_acc_interval,
                "d_ag": comparison.d_ag,
                "d_ag_interval": inference.d_ag_interval,
                "d_acc": comparison.d_acc,
                "d_acc_interval": inference.d_acc_interval,
            })
        else:
            alignment.append({
                "metric_output_id": metric, "state": metric_states[metric],
                "ag": None, "ag_raw": None, "ag_interval": None,
                "acc_cross": None, "acc_interval": None,
                "d_ag": None, "d_ag_interval": None,
                "d_acc": None, "d_acc_interval": None,
            })
    for result in family.values():
        metric, statistic = result.hypothesis_id.split(":", 1)
        state = next(
            row["state"] for row in signs
            if row["metric_output_id"] == metric
            and row["statistic"] == statistic
        )
        favorable = contrasts.get(result.hypothesis_id, 0.0) > 0.0
        holm.append({
            "metric_output_id": metric, "statistic": statistic,
            "raw_p_value": result.raw_p_value,
            "adjusted_p_value": result.adjusted_p_value,
            "rank": result.rank, "family_size": result.family_size,
            "family_state": (
                "complete" if state == "complete"
                else "not_tested_metric_incomplete"
            ),
            "favorable": bool(favorable),
            "rejected": bool(result.rejected and favorable),
        })
    return (
        tuple(observations), tuple(alignment), tuple(bootstrap),
        tuple(signs), tuple(holm),
    )


def _render_figures(
    observations: Sequence[Mapping[str, object]],
    alignment: Sequence[Mapping[str, object]],
    holm: Sequence[Mapping[str, object]],
) -> Mapping[str, bytes]:
    colors = dict(zip(
        PERTURBATIONS,
        ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"),
        strict=True,
    ))
    states = {row["metric_output_id"]: row["state"] for row in alignment}
    figure1_rows: list[Mapping[str, object]] = []
    for metric in METRICS:
        for perturbation in PERTURBATIONS:
            for alpha in ALPHAS[1:]:
                selected = [
                    row for row in observations
                    if row["metric_output_id"] == metric
                    and row["perturbation_id"] == perturbation
                    and row["alpha"] == alpha
                    and row["metric_harm"] is not None
                ]
                figure1_rows.append({
                    "metric_output_id": metric,
                    "perturbation_id": perturbation,
                    "alpha": alpha,
                    "mean_metric_harm": (
                        None if not selected else float(np.mean([
                            row["metric_harm"] for row in selected
                        ]))
                    ),
                    "mean_downstream_harm": (
                        None if not selected else float(np.mean([
                            row["downstream_harm"] for row in selected
                        ]))
                    ),
                    "metric_state": states[metric],
                })
    family = {
        (row["metric_output_id"], row["statistic"]): row for row in holm
    }
    figure2_fields = (
        "metric_output_id", "state", "ag", "ag_raw", "ag_interval",
        "acc_cross", "acc_interval", "d_ag", "d_ag_interval",
        "d_acc", "d_acc_interval", "d_ag_raw_p", "d_ag_adjusted_p",
        "d_ag_rank", "d_ag_favorable", "d_ag_rejected",
        "d_acc_raw_p", "d_acc_adjusted_p", "d_acc_rank",
        "d_acc_favorable", "d_acc_rejected",
    )
    figure2_rows = []
    for row in alignment:
        output = dict(row)
        for statistic in ("d_ag", "d_acc"):
            item = family.get((row["metric_output_id"], statistic), {})
            for target, source in (
                ("raw_p", "raw_p_value"),
                ("adjusted_p", "adjusted_p_value"),
                ("rank", "rank"),
                ("favorable", "favorable"),
                ("rejected", "rejected"),
            ):
                output[f"{statistic}_{target}"] = item.get(source)
        figure2_rows.append(output)
    payloads: dict[str, bytes] = {
        "figure1_d1_protocol_a_full_domain_data.csv": _csv(
            figure1_rows,
            (
                "metric_output_id", "perturbation_id", "alpha",
                "mean_metric_harm", "mean_downstream_harm", "metric_state",
            ),
        ),
        "figure2_d1_protocol_a_full_domain_data.csv": _csv(
            [
                {field: row.get(field) for field in figure2_fields}
                for row in figure2_rows
            ],
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
    for axis, metric in zip(flat_axes[1:], METRICS, strict=False):
        for perturbation in PERTURBATIONS:
            rows = [
                row for row in figure1_rows
                if row["metric_output_id"] == metric
                and row["perturbation_id"] == perturbation
            ]
            x = [row["mean_metric_harm"] for row in rows]
            y = [row["mean_downstream_harm"] for row in rows]
            if all(value is not None for value in (*x, *y)):
                axis.plot(
                    x, y, marker="o", color=colors[perturbation],
                    label=perturbation,
                )
        axis.set_title(metric)
    for axis in flat_axes[14:]:
        axis.set_axis_off()
    figure.tight_layout()
    for kind in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(
            stream, format=kind, dpi=300,
            metadata={"Date": None, "Creator": "raman-preproc-eval"},
        )
        payloads[f"figure1_d1_protocol_a_full_domain.{kind}"] = stream.getvalue()
    plt.close(figure)
    figure, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
    for axis, field, title, color in zip(
        axes, ("ag", "acc_cross", "d_ag", "d_acc"),
        ("AG", "Acc-cross", "D_AG", "D_Acc"),
        ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"), strict=True,
    ):
        values = [row.get(field) for row in figure2_rows]
        ypos = np.arange(len(METRICS))
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
        axis.set_yticks(ypos, METRICS if axis is axes[0] else [])
    figure.tight_layout()
    for kind in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(
            stream, format=kind, dpi=300,
            metadata={"Date": None, "Creator": "raman-preproc-eval"},
        )
        payloads[f"figure2_d1_protocol_a_full_domain.{kind}"] = stream.getvalue()
    plt.close(figure)
    return MappingProxyType(payloads)


def _condition_summary(
    predictions: Sequence[Mapping[str, object]], config: _Config,
) -> tuple[Mapping[str, object], ...]:
    indexed: dict[str, list[Mapping[str, object]]] = {}
    for row in predictions:
        indexed.setdefault(str(row["condition_id"]), []).append(row)
    summaries = []
    for condition_id, _, _ in _conditions():
        selected = indexed[condition_id]
        per_class = [
            float(np.mean([
                row["correct"] for row in selected
                if int(row["true_class"]) == label
            ]))
            for label in range(config.class_count)
        ]
        f1_values = []
        for label in range(config.class_count):
            tp = sum(
                row["true_class"] == label and row["predicted_class"] == label
                for row in selected
            )
            fp = sum(
                row["true_class"] != label and row["predicted_class"] == label
                for row in selected
            )
            fn = sum(
                row["true_class"] == label and row["predicted_class"] != label
                for row in selected
            )
            f1_values.append(
                0.0 if 2 * tp + fp + fn == 0
                else 2 * tp / (2 * tp + fp + fn)
            )
        summaries.append({
            "condition_id": condition_id,
            "prediction_count": len(selected),
            "top1_macro_class_accuracy": float(np.mean(per_class)),
            "top1_micro_accuracy": float(np.mean([
                row["correct"] for row in selected
            ])),
            "macro_f1": float(np.mean(f1_values)),
        })
    return tuple(summaries)


def _rebuild(
    inputs: _Inputs, config: _Config, worker_count: int, *,
    bootstrap_resamples: int, sign_flip_resamples: int,
) -> tuple[Mapping[str, bytes], str, str]:
    matrices, conditions, metrics, peaks, bridge = _real_science(
        inputs, config, worker_count
    )
    models = _fit_models(inputs, config)
    predictions, seed_rows, cells = _predict(models, matrices, inputs, config)
    observations, alignment, bootstrap, signs, holm = _aggregate(
        predictions, metrics, config,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )
    expected = config.document["expected"]
    actual = {
        "condition_count": len(conditions),
        "metric_value_count": len(metrics),
        "cwt_receipt_count": len(peaks),
        "model_cell_count": len(cells),
        "lr_candidate_fit_count": sum(
            len(model.validation_scores) for model in models
        ),
        "prediction_row_count": len(predictions),
        "seed_class_condition_count": len(seed_rows),
        "class_observation_count": len(observations),
        "alignment_result_count": len(alignment),
        "bootstrap_result_count": len(bootstrap),
        "sign_flip_result_count": len(signs),
        "holm_family_row_count": len(holm),
    }
    if any(actual[key] != int(expected[key]) for key in actual):
        raise Phase4D1ProtocolAVerifierError(
            f"rebuild: row-count mismatch {actual}"
        )
    preflight = {
        "status": "pass",
        "source_record_count": inputs.source_count,
        "model_cell_count": inputs.model_count or len(config.seeds),
        "role_occurrence_count": inputs.role_count,
        "source_ledger_sha256": inputs.source_digest,
        "model_ledger_sha256": inputs.model_digest,
        "role_ledger_sha256": inputs.role_digest,
        "role_overlap_detected": inputs.role_overlap,
        "condition_count": len(conditions),
        "condition_ledger_sha256": _ledger_digest(iter(conditions)),
        "condition_mismatch_count": 0,
        "train_validation_matrix_ready": True,
        "test_projection_ready": True,
        "inherited_rulings": {
            "p01_p05": "not_evaluable_coverage",
            "p06_p07": "structurally_ineligible_missing_explicit_baseline",
        },
    }
    figures = _render_figures(observations, alignment, holm)
    condition_rows = _condition_summary(predictions, config)
    table_fields = (
        "metric_output_id", "state", "ag", "ag_raw", "ag_interval",
        "acc_cross", "acc_interval", "d_ag", "d_ag_interval",
        "d_acc", "d_acc_interval",
    )
    code_identity = _sha_file(ROOT / "rpe/runner/phase4_d1_protocol_a.py")
    payloads: dict[str, bytes] = {
        "config.json": config.raw,
        "authority_bridge.json": _canonical(dict(bridge)),
        "preflight.json": _canonical(preflight),
        "model_cells.jsonl": _jsonl(cells),
        "validation_scores.jsonl": _jsonl([
            row for model in models for row in model.validation_scores
        ]),
        "predictions.jsonl": _jsonl([
            {**row, "code_identity": code_identity} for row in predictions
        ]),
        "seed_class_conditions.jsonl": _jsonl(seed_rows),
        "condition_summary.csv": _csv(
            condition_rows,
            (
                "condition_id", "prediction_count",
                "top1_macro_class_accuracy", "top1_micro_accuracy", "macro_f1",
            ),
        ),
        "class_observations.jsonl": _jsonl(observations),
        "alignment_results.jsonl": _jsonl(alignment),
        "bootstrap_results.jsonl": _jsonl(bootstrap),
        "sign_flip_results.jsonl": _jsonl(signs),
        "holm_family.jsonl": _jsonl(holm),
        **figures,
        "d1_protocol_a_full_domain_secondary_table.csv": _csv(
            [
                {field: row.get(field) for field in table_fields}
                for row in alignment
            ],
            table_fields,
        ),
    }
    mse_state = next(
        row["state"] for row in alignment if row["metric_output_id"] == "mse"
    )
    status = "complete" if mse_state == "complete" else "failed"
    run_id = _sha_bytes(
        b"rpe-phase4-d1-protocol-a-v1\0" + config.raw
        + _canonical({
            "source_ledger_sha256": inputs.source_digest,
            "model_ledger_sha256": inputs.model_digest,
            "role_ledger_sha256": inputs.role_digest,
            "bridge_sha256": bridge.get(
                "condition_bridge_sha256", "orchestration-test-bridge"
            ),
        })
    )
    manifest = {
        "schema_version": "phase4-d1-protocol-a-full-domain-artifact-v1",
        "run_id": run_id,
        "status": status,
        "endpoint_count": 1,
        "active_perturbation_count": 5,
        "condition_count": 41,
        "model_seed_count": len(config.seeds),
        "test_record_count": len(inputs.record_ids),
        "model_fit_count": len(models),
        "lr_candidate_fit_count": len(models) * 4,
        "prediction_row_count": len(predictions),
        "seed_class_condition_count": len(seed_rows),
        "class_observation_count": len(observations),
        "metric_states": {
            row["metric_output_id"]: row["state"] for row in alignment
        },
        "inherited_rulings": preflight["inherited_rulings"],
        "claim_boundary": CLAIM_BOUNDARY,
        "authorities": config.document.get("authorities", {}),
        "code_authority": config.document.get("code_authority", {}),
        "environment_authority": config.document.get("environment_authority", {}),
        "payload_files": list(PAYLOADS),
    }
    payloads["manifest.json"] = _canonical(manifest)
    if set(payloads) != set(PAYLOADS):
        raise Phase4D1ProtocolAVerifierError("rebuild: payload inventory mismatch")
    payloads = {name: payloads[name] for name in PAYLOADS}
    marker = "complete.json" if status == "complete" else "failed.json"
    marker_bytes = _canonical({"run_id": run_id, "status": status})
    sums = b"".join(
        f"{_sha_bytes(payloads[name])}  {name}\n".encode("utf-8")
        for name in PAYLOADS
    ) + f"{_sha_bytes(marker_bytes)}  {marker}\n".encode("utf-8")
    return (
        MappingProxyType({
            **payloads, marker: marker_bytes, "SHA256SUMS": sums,
        }),
        marker,
        status,
    )


def _validate_inventory(path: Path, config: _Config) -> tuple[str, Mapping[str, str]]:
    path = Path(path)
    if not path.is_dir():
        raise Phase4D1ProtocolAVerifierError("artifact: directory required")
    try:
        manifest_raw = (path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D1ProtocolAVerifierError("artifact: invalid manifest") from error
    if manifest_raw != _canonical(manifest):
        raise Phase4D1ProtocolAVerifierError("artifact: noncanonical manifest")
    status = manifest.get("status")
    marker = (
        "complete.json" if status == "complete"
        else "failed.json" if status == "failed"
        else None
    )
    expected_inventory = set(PAYLOADS) | {str(marker), "SHA256SUMS"}
    if marker is None or {item.name for item in path.iterdir()} != expected_inventory:
        raise Phase4D1ProtocolAVerifierError(
            "artifact: inventory/terminal marker mismatch"
        )
    if (
        manifest.get("endpoint_count") != 1
        or manifest.get("condition_count") != 41
        or manifest.get("model_seed_count") != len(config.seeds)
        or manifest.get("test_record_count") != config.test_count
        or manifest.get("payload_files") != list(PAYLOADS)
    ):
        raise Phase4D1ProtocolAVerifierError("artifact: D1 manifest mismatch")
    marker_raw = (path / marker).read_bytes()
    marker_document = json.loads(marker_raw)
    if (
        marker_raw != _canonical(marker_document)
        or marker_document != {
            "run_id": manifest.get("run_id"), "status": status,
        }
    ):
        raise Phase4D1ProtocolAVerifierError("artifact: terminal marker mismatch")
    sums: dict[str, str] = {}
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4D1ProtocolAVerifierError(
                "artifact: malformed checksum ledger"
            ) from error
        if name in sums or not _valid_sha256(digest):
            raise Phase4D1ProtocolAVerifierError(
                "artifact: malformed checksum ledger"
            )
        sums[name] = digest
    if tuple(sums) != PAYLOADS + (marker,) or any(
        _sha_file(path / name) != digest for name, digest in sums.items()
    ):
        raise Phase4D1ProtocolAVerifierError("artifact: checksum mismatch")
    return marker, MappingProxyType(sums)


def _compare_payloads(path: Path, rebuilt: Mapping[str, bytes]) -> None:
    if {item.name for item in path.iterdir()} != set(rebuilt):
        raise Phase4D1ProtocolAVerifierError(
            "rebuild: candidate inventory mismatch"
        )
    for name, value in rebuilt.items():
        if (path / name).read_bytes() != value:
            raise Phase4D1ProtocolAVerifierError(
                f"rebuild: byte mismatch for {name}"
            )


def verify_phase4_d1_protocol_a_from_inputs(
    path: Path, *, inputs: object, config_path: Path, worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4D1ProtocolASummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise Phase4D1ProtocolAVerifierError("worker_count must be positive")
    config = _load_config(Path(config_path), require_frozen_identity=False)
    if not config.synthetic:
        raise Phase4D1ProtocolAVerifierError(
            "real input verification requires frozen retained science reconstruction"
        )
    if bootstrap_resamples is None or sign_flip_resamples is None:
        raise Phase4D1ProtocolAVerifierError(
            "synthetic verifier requires explicit test inference resamples"
        )
    artifact = Path(path)
    _validate_inventory(artifact, config)
    rebuilt, _, _ = _rebuild(
        _coerce_inputs(inputs, config), config, worker_count,
        bootstrap_resamples=int(bootstrap_resamples),
        sign_flip_resamples=int(sign_flip_resamples),
    )
    _compare_payloads(artifact, rebuilt)
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    return Phase4D1ProtocolASummary(
        artifact, str(manifest["run_id"]), str(manifest["status"]),
        int(manifest["test_record_count"]), int(manifest["model_fit_count"]),
    )


def _read_checksum_tree(path: Path, expected_sums_sha: str) -> Mapping[str, str]:
    if not path.is_dir() or _sha_file(path / "SHA256SUMS") != expected_sums_sha:
        raise Phase4D1ProtocolAVerifierError("parent: checksum authority mismatch")
    checksums: dict[str, str] = {}
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4D1ProtocolAVerifierError(
                "parent: malformed checksum ledger"
            ) from error
        if name in checksums or not _valid_sha256(digest):
            raise Phase4D1ProtocolAVerifierError(
                "parent: malformed checksum ledger"
            )
        checksums[name] = digest
    for name, digest in checksums.items():
        if not (path / name).is_file() or _sha_file(path / name) != digest:
            raise Phase4D1ProtocolAVerifierError(
                f"parent: checksum mismatch for {name}"
            )
    return MappingProxyType(checksums)


def _iter_jsonl(path: Path) -> Iterator[Mapping[str, object]]:
    with path.open("rb") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise Phase4D1ProtocolAVerifierError(
                    f"{path.name}[{line_number}]: invalid JSON"
                ) from error
            if not isinstance(row, Mapping) or raw_line != _canonical(row):
                raise Phase4D1ProtocolAVerifierError(
                    f"{path.name}[{line_number}]: canonical object required"
                )
            yield row


def _validate_step13_parent(path: Path) -> Mapping[str, str]:
    required = {
        "config.json", "source_records.jsonl", "model_cells.jsonl",
        "model_role_occurrences.jsonl", "cells.jsonl",
        "record_conditions.jsonl", "metric_statuses.jsonl",
        "cwt_receipts.jsonl", "class_summaries.jsonl",
        "common_support.jsonl", "gate.json", "manifest.json",
        "failed.json", "SHA256SUMS",
    }
    if path.name != STEP13_RUN_ID or {item.name for item in path.iterdir()} != required:
        raise Phase4D1ProtocolAVerifierError("Step-13 parent identity mismatch")
    checksums = _read_checksum_tree(path, STEP13_SUMS)
    if _sha_file(path / "config.json") != STEP13_CONFIG:
        raise Phase4D1ProtocolAVerifierError("Step-13 config identity mismatch")
    gate = json.loads((path / "gate.json").read_bytes())
    if (
        gate.get("full_domain_core", {}).get("state") != "evaluable"
        or gate.get("peak_common_support", {}).get("state")
        != "not_evaluable_coverage"
    ):
        raise Phase4D1ProtocolAVerifierError(
            "Step-13 tier authorization mismatch"
        )
    return checksums


def _load_test_identity() -> tuple[tuple[str, ...], Mapping[str, int]]:
    record_ids: list[str] = []
    labels: dict[str, int] = {}
    with BacteriaIdBatchLoader(
        ROOT / "data/unified/bacteria_id_reference", batch_size=4096
    ) as loader:
        for batch in loader.iter_batches():
            if batch.source_split != "test":
                continue
            for record_id, label in zip(
                batch.record_ids, batch.class_labels, strict=True
            ):
                key = str(record_id)
                record_ids.append(key)
                labels[key] = int(label)
    counts = {label: 0 for label in range(30)}
    for label in labels.values():
        if label in counts:
            counts[label] += 1
    if (
        len(record_ids) != 3000
        or len(set(record_ids)) != 3000
        or set(labels.values()) != set(range(30))
        or any(value != 100 for value in counts.values())
        or _id_digest(record_ids) != TEST_DIGEST
    ):
        raise Phase4D1ProtocolAVerifierError(
            "Step-15 bridge: retained test identity mismatch"
        )
    return tuple(record_ids), MappingProxyType(labels)


def _load_parent_bundle(config: _Config) -> _ParentBundle:
    if "step15_forbidden_probe" in config.document:
        raise Phase4D1ProtocolAVerifierError(
            "forbidden Step-15 outcome payload request"
        )
    step13 = _validate_step13_parent(STEP13_ROOT)
    step15 = _read_checksum_tree(STEP15_ROOT, STEP15_SUMS)
    manifest = json.loads((STEP15_ROOT / "manifest.json").read_bytes())
    marker = json.loads((STEP15_ROOT / "complete.json").read_bytes())
    if (
        manifest.get("run_id") != STEP15_RUN_ID
        or manifest.get("status") != "complete"
        or marker.get("run_id") != STEP15_RUN_ID
    ):
        raise Phase4D1ProtocolAVerifierError("Step-15 parent identity mismatch")
    for name, digest in STEP15_ALLOWED.items():
        if step15.get(name) != digest:
            raise Phase4D1ProtocolAVerifierError(
                f"Step-15 parent payload mismatch for {name}"
            )
    authorities = config.document.get("authorities", {})
    nested = (
        authorities.get("step15_parent", {})
        if isinstance(authorities, Mapping) else {}
    )
    top = config.document.get("step15_parent", nested)
    for parent in (nested, top):
        if (
            not isinstance(parent, Mapping)
            or parent.get("sha256sums_sha256") != STEP15_SUMS
            or parent.get("run_id") != STEP15_RUN_ID
            or parent.get("authorized_payloads") != dict(STEP15_ALLOWED)
        ):
            raise Phase4D1ProtocolAVerifierError(
                "Step-15 parent checksum authority mismatch"
            )
    test_ids, labels = _load_test_identity()
    condition_ids = tuple(row[0] for row in _conditions())
    condition_rows: list[Mapping[str, object]] = []
    bridge_digest = hashlib.sha256()
    condition_keys = {
        "axis_sha256", "condition_id", "intensity_sha256",
        "output_spectrum_id", "projected_row_sha256", "record_id",
        "record_order", "result_sha256", "source_spectrum_id", "state",
    }
    for index, row in enumerate(
        _iter_jsonl(STEP15_ROOT / "record_conditions.jsonl")
    ):
        record_index, condition_index = divmod(index, len(condition_ids))
        if record_index >= len(test_ids):
            raise Phase4D1ProtocolAVerifierError(
                "Step-15 conditions: too many rows"
            )
        record_id = test_ids[record_index]
        condition_id = condition_ids[condition_index]
        if (
            set(row) != condition_keys
            or row.get("record_id") != record_id
            or row.get("record_order") != record_index
            or row.get("condition_id") != condition_id
            or row.get("state") != "complete"
            or not all(_valid_sha256(row.get(key)) for key in (
                "axis_sha256", "intensity_sha256",
                "projected_row_sha256", "result_sha256",
            ))
        ):
            raise Phase4D1ProtocolAVerifierError(
                f"Step-15 conditions: schema/order/state mismatch at row {index}"
            )
        bridge_digest.update(_canonical({
            "test_order": record_index,
            "record_id": record_id,
            "class_label": labels[record_id],
            "condition_id": condition_id,
            "state": "complete",
            "axis_sha256": row["axis_sha256"],
            "intensity_sha256": row["intensity_sha256"],
            "support_projection_sha256": row["projected_row_sha256"],
        }))
        condition_rows.append(row)
    expected_condition_count = len(test_ids) * len(condition_ids)
    if len(condition_rows) != expected_condition_count:
        raise Phase4D1ProtocolAVerifierError(
            "Step-15 conditions: row count mismatch"
        )
    metric_rows: list[Mapping[str, object]] = []
    metric_keys = {
        "condition_id", "diagnostics_sha256", "metric_output_id",
        "record_id", "record_order", "result_sha256", "state", "value",
    }
    per_record = len(condition_ids) * len(METRICS)
    for index, row in enumerate(
        _iter_jsonl(STEP15_ROOT / "metric_values.jsonl")
    ):
        record_index, remainder = divmod(index, per_record)
        condition_index, metric_index = divmod(remainder, len(METRICS))
        value = row.get("value")
        if (
            record_index >= len(test_ids)
            or set(row) != metric_keys
            or row.get("record_id") != test_ids[record_index]
            or row.get("record_order") != record_index
            or row.get("condition_id") != condition_ids[condition_index]
            or row.get("metric_output_id") != METRICS[metric_index]
            or row.get("state") != "complete"
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not _valid_sha256(row.get("result_sha256"))
            or not _valid_sha256(row.get("diagnostics_sha256"))
        ):
            raise Phase4D1ProtocolAVerifierError(
                f"Step-15 metrics: schema/order/state/value mismatch at row {index}"
            )
        metric_rows.append(row)
    if len(metric_rows) != len(test_ids) * per_record:
        raise Phase4D1ProtocolAVerifierError(
            "Step-15 metrics: row count mismatch"
        )
    peak_rows: list[Mapping[str, object]] = []
    peak_keys = {
        "condition_id", "diagnostics_sha256", "peak_list_sha256",
        "record_id", "record_order", "state", "warning_sha256",
    }
    for index, row in enumerate(
        _iter_jsonl(STEP15_ROOT / "peak_receipts.jsonl")
    ):
        record_index, condition_index = divmod(index, len(condition_ids))
        if (
            record_index >= len(test_ids)
            or set(row) != peak_keys
            or row.get("record_id") != test_ids[record_index]
            or row.get("record_order") != record_index
            or row.get("condition_id") != condition_ids[condition_index]
            or row.get("state") != "complete"
            or not all(_valid_sha256(row.get(key)) for key in (
                "diagnostics_sha256", "peak_list_sha256", "warning_sha256",
            ))
        ):
            raise Phase4D1ProtocolAVerifierError(
                f"Step-15 CWT: schema/order/state mismatch at row {index}"
            )
        peak_rows.append(row)
    if len(peak_rows) != expected_condition_count:
        raise Phase4D1ProtocolAVerifierError(
            "Step-15 CWT: row count mismatch"
        )
    digest = bridge_digest.hexdigest()
    if digest != BRIDGE_DIGEST:
        raise Phase4D1ProtocolAVerifierError(
            "Step-15 bridge: canonical digest mismatch"
        )
    bridge = MappingProxyType({
        "authorized_step15_payloads": tuple(STEP15_ALLOWED),
        "bridge_row_count": len(condition_rows),
        "condition_bridge_sha256": digest,
        "metric_row_count": len(metric_rows),
        "cwt_row_count": len(peak_rows),
        "step13_full_domain_state": "evaluable",
        "step13_peak_common_state": "not_evaluable_coverage",
        "step13_sha256sums_sha256": STEP13_SUMS,
        "step15_run_id": STEP15_RUN_ID,
        "step15_sha256sums_sha256": STEP15_SUMS,
        "step13_checksums": dict(sorted(step13.items())),
        "step15_checksums": dict(sorted(step15.items())),
    })
    return _ParentBundle(
        bridge, tuple(condition_rows), tuple(metric_rows), tuple(peak_rows)
    )


def _split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    for label in range(30):
        indexes = np.flatnonzero(labels == label)
        if len(indexes) != 100:
            raise Phase4D1ProtocolAVerifierError(
                f"D1 inputs: finetune class {label} must contain 100 records"
            )
        shuffled = generator.permutation(indexes)
        validation.extend(shuffled[:10].tolist())
        train.extend(shuffled[10:].tolist())
    return (
        np.asarray(sorted(train), dtype=np.int64),
        np.asarray(sorted(validation), dtype=np.int64),
    )


def _reconstruct_real_inputs(config: _Config) -> _Inputs:
    dataset = ROOT / "data/unified/bacteria_id_reference"
    eligibility = load_phase4_d2_eligibility_config(
        ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json"
    )
    support = np.asarray(eligibility.support_coordinates_cm1, dtype="<f8")
    split_values: dict[str, list[np.ndarray]] = {
        "finetune": [], "reference": [], "test": [],
    }
    split_native_values: dict[str, list[np.ndarray]] = {
        "finetune": [], "reference": [], "test": [],
    }
    split_labels: dict[str, list[int]] = {key: [] for key in split_values}
    split_ids: dict[str, list[str]] = {key: [] for key in split_values}
    split_rows: dict[str, list[int]] = {key: [] for key in split_values}
    native_tests: list[Spectrum1D] = []
    native_axis: np.ndarray | None = None
    with BacteriaIdBatchLoader(dataset, batch_size=4096) as loader:
        for batch in loader.iter_batches():
            split = batch.source_split
            if split not in split_values:
                raise Phase4D1ProtocolAVerifierError(
                    f"D1 inputs: unexpected source split {split}"
                )
            axis = np.ascontiguousarray(
                np.asarray(batch.wavenumber, dtype="<f4")[::-1], dtype="<f8"
            )
            if not np.all(np.diff(axis) > 0.0):
                raise Phase4D1ProtocolAVerifierError(
                    "D1 inputs: native axis reversal failed"
                )
            if native_axis is None:
                native_axis = axis
            elif not np.array_equal(native_axis, axis):
                raise Phase4D1ProtocolAVerifierError(
                    "D1 inputs: shared native axis mismatch"
                )
            for record_id, label, source_row, intensity in zip(
                batch.record_ids, batch.class_labels, batch.source_rows,
                batch.intensity, strict=True,
            ):
                key = str(record_id)
                native_intensity = np.ascontiguousarray(
                    np.asarray(intensity, dtype="<f4")[::-1], dtype="<f8"
                )
                spectrum = Spectrum1D(
                    spectrum_id=f"bacteria_id_reference::{key}",
                    sample_id=None, axis_cm1=axis, intensity=native_intensity,
                )
                split_values[split].append(project_d2_support(spectrum, eligibility))
                split_native_values[split].append(
                    np.ascontiguousarray(intensity, dtype="<f4")
                )
                split_labels[split].append(int(label))
                split_ids[split].append(key)
                split_rows[split].append(int(source_row))
                if split == "test":
                    native_tests.append(spectrum)
    if native_axis is None or _array_sha(native_axis) != NATIVE_F64:
        raise Phase4D1ProtocolAVerifierError(
            "D1 inputs: native axis identity mismatch"
        )
    if _array_sha(support) != SUPPORT_F64 or _array_sha(support, "<f4") != SUPPORT_F32:
        raise Phase4D1ProtocolAVerifierError(
            "D1 inputs: support axis identity mismatch"
        )
    if {key: len(value) for key, value in split_ids.items()} != {
        "finetune": 3000, "reference": 60000, "test": 3000,
    }:
        raise Phase4D1ProtocolAVerifierError(
            "D1 inputs: source split cardinality mismatch"
        )
    test_counts = np.bincount(
        np.asarray(split_labels["test"], dtype=np.int64), minlength=30
    )
    if (
        _id_digest(split_ids["test"]) != TEST_DIGEST
        or tuple(test_counts.tolist()) != (100,) * 30
    ):
        raise Phase4D1ProtocolAVerifierError("D1 inputs: test identity mismatch")
    source_digest = _ledger_digest(
        {
            "class_label": split_labels[split][index],
            "record_id": record_id,
            "source_row": split_rows[split][index],
            "source_split": split,
        }
        for split in ("finetune", "reference", "test")
        for index, record_id in enumerate(split_ids[split])
    )
    reference_values = np.asarray(split_values["reference"], dtype="<f4")
    reference_native = np.asarray(split_native_values["reference"], dtype="<f4")
    reference_labels = np.asarray(split_labels["reference"], dtype="<i8")
    finetune_values = np.asarray(split_values["finetune"], dtype="<f4")
    finetune_native = np.asarray(split_native_values["finetune"], dtype="<f4")
    finetune_labels = np.asarray(split_labels["finetune"], dtype="<i8")
    test_values = np.asarray(split_values["test"], dtype="<f4")
    test_native = np.asarray(split_native_values["test"], dtype="<f4")
    test_labels = np.asarray(split_labels["test"], dtype="<i8")
    train_values: dict[int, np.ndarray] = {}
    train_labels: dict[int, np.ndarray] = {}
    validation_values: dict[int, np.ndarray] = {}
    validation_labels: dict[int, np.ndarray] = {}
    cells: dict[int, Mapping[str, object]] = {}
    model_rows: list[Mapping[str, object]] = []
    role_rows: list[Mapping[str, object]] = []
    role_overlap = False
    split_identities = config.document.get("frozen_identities", {}).get(
        "split_record_ids_sha256", {}
    )
    for seed in config.seeds:
        fit_indexes, validation_indexes = _split_indices(finetune_labels, seed)
        train_ids = tuple(
            split_ids["reference"]
            + [split_ids["finetune"][index] for index in fit_indexes]
        )
        valid_ids = tuple(
            split_ids["finetune"][index] for index in validation_indexes
        )
        test_ids = tuple(split_ids["test"])
        train_matrix = np.ascontiguousarray(
            np.concatenate(
                (reference_values, finetune_values[fit_indexes]), axis=0
            ),
            dtype="<f4",
        )
        valid_matrix = np.ascontiguousarray(
            finetune_values[validation_indexes], dtype="<f4"
        )
        train_label = np.concatenate(
            (reference_labels, finetune_labels[fit_indexes])
        )
        valid_label = finetune_labels[validation_indexes]
        if train_matrix.shape != (62700, 997) or valid_matrix.shape != (300, 997):
            raise Phase4D1ProtocolAVerifierError(
                "D1 inputs: model matrix cardinality mismatch"
            )
        if any(
            set(left) & set(right)
            for left, right in (
                (train_ids, valid_ids), (train_ids, test_ids),
                (valid_ids, test_ids),
            )
        ):
            role_overlap = True
        source_hashes = {
            "train": {
                _sha_bytes(row.tobytes())
                for row in np.concatenate(
                    (reference_native, finetune_native[fit_indexes])
                )
            },
            "validation": {
                _sha_bytes(row.tobytes())
                for row in finetune_native[validation_indexes]
            },
            "test": {_sha_bytes(row.tobytes()) for row in test_native},
        }
        projected_hashes = {
            "train": {_sha_bytes(row.tobytes()) for row in train_matrix},
            "validation": {
                _sha_bytes(row.tobytes()) for row in valid_matrix
            },
            "test": {_sha_bytes(row.tobytes()) for row in test_values},
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
            raise Phase4D1ProtocolAVerifierError(
                "D1 inputs: exact cross-role duplicate detected"
            )
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
        expected_split = (
            split_identities.get(str(seed), {})
            if isinstance(split_identities, Mapping) else {}
        )
        if expected_split != {
            "train": cell["train_record_ids_sha256"],
            "validation": cell["validation_record_ids_sha256"],
            "test": cell["test_record_ids_sha256"],
        }:
            raise Phase4D1ProtocolAVerifierError(
                "D1 inputs: frozen split identity mismatch"
            )
        cells[seed] = MappingProxyType(cell)
        model_rows.append({
            key: value for key, value in cell.items()
            if not key.endswith("record_ids")
        })
        for role, ids, labels, source_rows, source_splits in (
            (
                "train", train_ids, train_label,
                tuple(split_rows["reference"])
                + tuple(split_rows["finetune"][index] for index in fit_indexes),
                ("reference",) * 60000 + ("finetune",) * 2700,
            ),
            (
                "validation", valid_ids, valid_label,
                tuple(split_rows["finetune"][index] for index in validation_indexes),
                ("finetune",) * 300,
            ),
            (
                "test", test_ids, test_labels, tuple(split_rows["test"]),
                ("test",) * 3000,
            ),
        ):
            role_rows.extend({
                "class_label": int(label),
                "model_seed": seed,
                "record_id": record_id,
                "role": role,
                "role_order": order,
                "source_row": int(source_row),
                "source_split": source_split,
            } for order, (record_id, label, source_row, source_split) in enumerate(
                zip(ids, labels, source_rows, source_splits, strict=True)
            ))
        train_values[seed] = train_matrix
        train_labels[seed] = np.asarray(train_label, dtype="<i8")
        validation_values[seed] = valid_matrix
        validation_labels[seed] = np.asarray(valid_label, dtype="<i8")
    model_digest = _ledger_digest(iter(model_rows))
    role_digest = _ledger_digest(iter(role_rows))
    if (
        source_digest != SOURCE_DIGEST
        or model_digest != MODEL_DIGEST
        or role_digest != ROLE_DIGEST
        or role_overlap
    ):
        raise Phase4D1ProtocolAVerifierError(
            "D1 inputs: frozen ledger identity mismatch"
        )
    return _Inputs(
        test_values=np.ascontiguousarray(test_values, dtype="<f4"),
        test_labels=test_labels,
        record_ids=tuple(split_ids["test"]),
        model_seeds=config.seeds,
        train_values=MappingProxyType(train_values),
        train_labels=MappingProxyType(train_labels),
        validation_values=MappingProxyType(validation_values),
        validation_labels=MappingProxyType(validation_labels),
        model_cells=MappingProxyType(cells),
        support_axis=support,
        native_spectra=tuple(native_tests),
        native_labels=test_labels.copy(),
        source_count=66000,
        model_count=5,
        role_count=330000,
        source_digest=source_digest,
        model_digest=model_digest,
        role_digest=role_digest,
        support_f32=_array_sha(support, "<f4"),
        support_f64=_array_sha(support),
        role_overlap=role_overlap,
        force_metric_failure=False,
    )


def verify_phase4_d1_protocol_a(
    path: Path, *, worker_count: int = 12
) -> Phase4D1ProtocolASummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise Phase4D1ProtocolAVerifierError("worker_count must be positive")
    config = _load_config(DEFAULT_CONFIG, require_frozen_identity=True)
    artifact = Path(path)
    _validate_inventory(artifact, config)
    inputs = _reconstruct_real_inputs(config)
    inference = config.document["inference"]
    rebuilt, _, _ = _rebuild(
        inputs, config, worker_count,
        bootstrap_resamples=int(inference["bootstrap_resamples"]),
        sign_flip_resamples=int(inference["sign_flip_resamples"]),
    )
    _compare_payloads(artifact, rebuilt)
    manifest = json.loads((artifact / "manifest.json").read_bytes())
    return Phase4D1ProtocolASummary(
        artifact, str(manifest["run_id"]), str(manifest["status"]),
        int(manifest["test_record_count"]), int(manifest["model_fit_count"]),
    )


__all__ = [
    "Phase4D1ProtocolASummary", "Phase4D1ProtocolAVerifierError",
    "verify_phase4_d1_protocol_a", "verify_phase4_d1_protocol_a_from_inputs",
]
