"""Independent verifier for the Phase 4 D4 Protocol-A outcome artifact.

The verifier owns its parsing, retained-input reconstruction, scientific
execution, model fitting, aggregation, rendering, and serialization paths.
Only frozen lower-level data/scientific/statistical primitives are shared.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import multiprocessing
import platform
import struct
import tempfile
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
from rpe.evaluation import PeakPairInput, SingleSpectrumInput, Spectrum1D, SpectrumPairInput, evaluate_metric
from rpe.methods import load_classical_catalog
from rpe.methods.classical.peaks import run_peak_detection_system
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
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus
from rpe.runner.phase4_d4_protocol_a_authority import CONFIG_BYTES, CONFIG_SHA256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d4_protocol_a_full_domain_v1.json"
ELIGIBILITY_CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json"
PROTOCOL_CONFIG_RELATIVE_PATH = "experiments/phase05/configs/d4_sugar_protocol.json"
ARCHIVE_RELATIVE_PATH = "data/raw/ramanbench/cache/10779223/Raw data.zip"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
CATALOG_RELATIVE_PATH = "experiments/phase3/configs/classical_system_catalog_v1.json"
PHASE05_AUDIT_RELATIVE_PATH = "reports/phase05/d4_step01_protocol_audit.json"
PARENT_RUN_ID = "phase4-d4-protocol-a-full-domain-eligibility-0f45ae5a0815ecb852b410549ad4582c4916dbe16fa0e3e20e2079f7d088097f"
PARENT_SHA256SUMS_SHA256 = "86fc5d5635c2cc14877ab14b5cb0159fc6382ef494009697c58c77041a464d83"
PARENT_CONFIG_SHA256 = "d723c68ec7485b0a778287224f498884c54bec1ce0d1f688a201deacd922aff5"
ELIGIBILITY_CONFIG_BYTES = 45_314
SCHEMA_VERSION = "phase4-d4-protocol-a-full-domain-config-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d4-protocol-a-full-domain-artifact-v1"
EXPERIMENT_ID = "phase4-d4-protocol-a-full-domain-v1"
RUN_PREFIX = "phase4-d4-protocol-a-full-domain-"
PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
CONDITION_IDS = ("alpha0",) + tuple(
    f"{perturbation_id}:{struct.pack('<d', alpha).hex()}"
    for perturbation_id in PERTURBATION_IDS
    for alpha in POSITIVE_ALPHAS
)
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
LOWER_IS_BETTER = frozenset(
    {
        "mse",
        "rmse",
        "mae",
        "sam",
        "nmse",
        "wasserstein_1_cm1",
        "artifact_peak_ratio",
        "missing_peak_ratio",
    }
)
TARGET_NAMES = (
    "sucrose_nominal_mol_l",
    "fructose_nominal_mol_l",
    "maltose_nominal_mol_l",
    "glucose_nominal_mol_l",
)
TARGET_RANGE_MOL_L = 0.32
CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
SUPPORT_F64_SHA256 = "db910b11f92151db481391e96b8a06e246140596cfd4abad64764b87d2d84ee5"
SUPPORT_F32_SHA256 = "c32275fcb069cf66b3c9b19e1e922d93c724ea4d54e9a9bc8dfaad734ca4c807"
COLOR_MAP = MappingProxyType(
    dict(zip(PERTURBATION_IDS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True))
)
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "eligibility_bridge.json",
    "preflight.json",
    "model_cells.jsonl",
    "validation_scores.jsonl",
    "record_measurements.jsonl",
    "predictions.jsonl",
    "blank_predictions.jsonl",
    "well_conditions.jsonl",
    "technical_lod_loq.jsonl",
    "condition_summary.csv",
    "well_observations.jsonl",
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "figure1_d4_protocol_a_full_domain.png",
    "figure1_d4_protocol_a_full_domain.svg",
    "figure1_d4_protocol_a_full_domain_data.csv",
    "figure2_d4_protocol_a_full_domain.png",
    "figure2_d4_protocol_a_full_domain.svg",
    "figure2_d4_protocol_a_full_domain_data.csv",
    "d4_protocol_a_full_domain_secondary_table.csv",
    "manifest.json",
)


class Phase4D4ProtocolAVerifierError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D4ProtocolASummary:
    path: Path
    run_id: str
    status: str
    record_count: int
    prediction_row_count: int


@dataclass(frozen=True)
class _Config:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_output_ids: tuple[str, ...]
    artifact_payload_files: tuple[str, ...]
    n_components_grid: tuple[int, ...]
    bootstrap_resamples: int
    sign_flip_resamples: int
    random_seed: int
    confidence_level: float
    holm_alpha: float
    target_range_mol_l: float


@dataclass(frozen=True)
class _EligibilityConfig:
    support_coordinates_cm1: tuple[float, ...]
    support_max_gap_cm1: float
    p10_memory_budget_bytes: int
    cwt_system_id: str
    frozen_identities: Mapping[str, object]


@dataclass(frozen=True)
class _Inputs:
    record_ids: tuple[str, ...]
    well_ids: tuple[str, ...]
    record_well_ids: tuple[str, ...]
    fold_by_well: Mapping[str, int]
    rounds: tuple[int, ...]
    repetitions: tuple[int, ...]
    targets: np.ndarray
    blank_record_ids: tuple[str, ...]
    support_axis_cm1: np.ndarray
    native_spectra: tuple[Spectrum1D, ...]
    native_blank_spectra: tuple[Spectrum1D, ...]
    alpha0_projected: np.ndarray
    train_indices_by_fold: Mapping[int, np.ndarray]
    validation_indices_by_fold: Mapping[int, np.ndarray]
    test_indices_by_fold: Mapping[int, np.ndarray]
    eligibility_config: _EligibilityConfig
    acquisitions_per_well: int = 32
    model_folds: int = 5
    feature_count: int = 1999


@dataclass(frozen=True)
class _Model:
    fold: int
    selected_n_components: int
    validation_scores: tuple[Mapping[str, object], ...]
    model_state_digest: str
    estimator: PLSRegression
    alpha0_train_predictions: np.ndarray


@dataclass(frozen=True)
class _Science:
    condition_spectra: Mapping[str, np.ndarray]
    blank_condition_spectra: Mapping[str, np.ndarray]
    record_measurements: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class _Projection:
    well_observations: tuple[Mapping[str, object], ...]
    alignment_results: tuple[Mapping[str, object], ...]
    bootstrap_results: tuple[Mapping[str, object], ...]
    sign_flip_results: tuple[Mapping[str, object], ...]
    holm_family: tuple[Mapping[str, object], ...]


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise Phase4D4ProtocolAVerifierError("json", "nonfinite float")
        return value
    raise Phase4D4ProtocolAVerifierError("json", f"unsupported value {type(value).__name__}")


def _receipt_ready(value: object) -> object:
    if is_dataclass(value):
        return {field.name: _receipt_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _receipt_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_receipt_ready(item) for item in value]
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
    raise Phase4D4ProtocolAVerifierError("receipt", f"unsupported value {type(value).__name__}")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _receipt_hash(value: object) -> str:
    raw = (
        json.dumps(_receipt_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields_: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields_, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields_})
    return stream.getvalue().encode("utf-8")


def _environment() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "machine": platform.machine(),
            "matplotlib": matplotlib.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "system": platform.system(),
            "threadpoolctl": threadpoolctl.__version__,
        }
    )


def _condition_alpha(condition_id: str) -> float:
    if condition_id == "alpha0":
        return 0.0
    return float(struct.unpack("<d", bytes.fromhex(condition_id.split(":", 1)[1]))[0])


def _condition_perturbation(condition_id: str) -> str:
    return "alpha0" if condition_id == "alpha0" else condition_id.split(":", 1)[0]


def _load_config(path: Path) -> _Config:
    path = Path(path)
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D4ProtocolAVerifierError("config.json", str(error)) from error
    if raw != _canonical(document):
        raise Phase4D4ProtocolAVerifierError("config.json", "must use canonical JSON")
    if len(raw) != CONFIG_BYTES or _sha_bytes(raw) != CONFIG_SHA256:
        raise Phase4D4ProtocolAVerifierError("config.json", "frozen identity mismatch")
    frozen = (ROOT / CONFIG_RELATIVE_PATH).read_bytes()
    if raw != frozen:
        raise Phase4D4ProtocolAVerifierError("config.json", "does not equal frozen experiment config")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D4ProtocolAVerifierError("config.json", "schema or experiment mismatch")
    if document.get("synthetic_fixture") is not False:
        raise Phase4D4ProtocolAVerifierError("config.json", "authoritative verification requires retained data")
    perturbations = tuple(str(value) for value in document.get("perturbation_ids", ()))
    alphas = tuple(float(value) for value in document.get("alpha_grid", ()))
    metrics = tuple(str(value) for value in document.get("metric_output_ids", ()))
    payloads = tuple(str(value) for value in document.get("artifact_payload_files", ()))
    if perturbations != PERTURBATION_IDS or alphas != ALPHA_GRID or metrics != METRIC_OUTPUT_IDS:
        raise Phase4D4ProtocolAVerifierError("config.json", "condition or metric order mismatch")
    if payloads != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D4ProtocolAVerifierError("config.json", "payload order mismatch")
    parent = document.get("parent")
    recipe = document.get("model_recipe")
    inference = document.get("inference")
    if not isinstance(parent, Mapping) or not isinstance(recipe, Mapping) or not isinstance(inference, Mapping):
        raise Phase4D4ProtocolAVerifierError("config.json", "parent/model/inference objects required")
    if parent.get("run_id") != PARENT_RUN_ID or parent.get("marker_filename") != "failed.json" or parent.get("full_domain_state") != "evaluable":
        raise Phase4D4ProtocolAVerifierError("config.json", "parent authority mismatch")
    authorities = document.get("authorities")
    code_authority = document.get("code_authority")
    if not isinstance(authorities, Mapping) or not isinstance(code_authority, Mapping):
        raise Phase4D4ProtocolAVerifierError("config.json", "authority receipts missing")
    for receipt in authorities.values():
        if not isinstance(receipt, Mapping):
            raise Phase4D4ProtocolAVerifierError("config authority", "receipt must be an object")
        relative = receipt.get("path")
        if not isinstance(relative, str):
            continue
        live = ROOT / relative
        if not live.is_file() or live.stat().st_size != int(receipt["bytes"]):
            raise Phase4D4ProtocolAVerifierError("config authority", f"size mismatch for {relative}")
        if relative != ARCHIVE_RELATIVE_PATH and _sha_file(live) != str(receipt["sha256"]):
            raise Phase4D4ProtocolAVerifierError("config authority", f"SHA256 mismatch for {relative}")
    for relative, receipt in code_authority.items():
        if not isinstance(relative, str) or not isinstance(receipt, Mapping):
            raise Phase4D4ProtocolAVerifierError("code authority", "invalid receipt")
        live = ROOT / relative
        if (
            not live.is_file()
            or live.stat().st_size != int(receipt["bytes"])
            or _sha_file(live) != str(receipt["sha256"])
        ):
            raise Phase4D4ProtocolAVerifierError("code authority", f"identity mismatch for {relative}")
    observed_environment = {**dict(_environment()), "matplotlib": matplotlib.__version__}
    if document.get("environment_authority") != observed_environment:
        raise Phase4D4ProtocolAVerifierError("config authority", "environment mismatch")
    return _Config(
        path=path,
        raw_bytes=raw,
        sha256=_sha_bytes(raw),
        document=MappingProxyType(document),
        perturbation_ids=perturbations,
        alpha_grid=alphas,
        metric_output_ids=metrics,
        artifact_payload_files=payloads,
        n_components_grid=tuple(int(value) for value in recipe["n_components_grid"]),
        bootstrap_resamples=int(inference["bootstrap_resamples"]),
        sign_flip_resamples=int(inference["sign_flip_resamples"]),
        random_seed=int(inference["random_seed"]),
        confidence_level=float(inference["confidence_level"]),
        holm_alpha=float(inference["holm_alpha"]),
        target_range_mol_l=float(document["target_range_mol_l"]),
    )


def _load_eligibility_config(path: Path) -> _EligibilityConfig:
    raw = Path(path).read_bytes()
    if len(raw) != ELIGIBILITY_CONFIG_BYTES or _sha_bytes(raw) != PARENT_CONFIG_SHA256:
        raise Phase4D4ProtocolAVerifierError("eligibility config", "frozen identity mismatch")
    document = json.loads(raw)
    if raw != _canonical(document):
        raise Phase4D4ProtocolAVerifierError("eligibility config", "must use canonical JSON")
    support = document.get("support_grid")
    p10 = document.get("p10")
    frozen = document.get("frozen_identities")
    if not isinstance(support, Mapping) or not isinstance(p10, Mapping) or not isinstance(frozen, Mapping):
        raise Phase4D4ProtocolAVerifierError("eligibility config", "required objects missing")
    coordinates = tuple(float(value) for value in support["coordinates_cm1"])
    if len(coordinates) != 1999:
        raise Phase4D4ProtocolAVerifierError("eligibility config", "support point count mismatch")
    return _EligibilityConfig(
        support_coordinates_cm1=coordinates,
        support_max_gap_cm1=float(support["max_in_range_native_gap_cm1"]),
        p10_memory_budget_bytes=int(p10["memory_budget_bytes"]),
        cwt_system_id=str(document["cwt_system_id"]),
        frozen_identities=MappingProxyType(dict(frozen)),
    )


def _parent_root() -> Path:
    return ROOT / "results/phase4/d4_protocol_a_full_domain_eligibility_v1" / PARENT_RUN_ID


def _read_jsonl_rows(path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise Phase4D4ProtocolAVerifierError(str(path), "JSONL row is not an object")
                rows.append(row)
    return tuple(rows)


def _checksum_inventory(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4D4ProtocolAVerifierError(str(path), "malformed checksum line") from error
        if name in result or len(digest) != 64:
            raise Phase4D4ProtocolAVerifierError(str(path), f"invalid checksum entry {name}")
        result[name] = digest
    return result


def _validate_parent(config: _Config) -> Mapping[str, object]:
    path = _parent_root()
    required = {
        "config.json", "model_cells.jsonl", "common_support.jsonl", "manifest.json",
        "gate.json", "well_summaries.jsonl", "model_role_occurrences.jsonl",
        "source_records.jsonl", "blank_conditions.jsonl", "operator_cells.jsonl",
        "blank_cells.jsonl", "record_conditions.jsonl", "well_folds.jsonl",
        "failed.json", "SHA256SUMS",
    }
    names = {item.name for item in path.iterdir()} if path.is_dir() else set()
    if names != required:
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "exact inventory mismatch")
    sums_path = path / "SHA256SUMS"
    if _sha_file(sums_path) != PARENT_SHA256SUMS_SHA256:
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "SHA256SUMS identity mismatch")
    checksums = _checksum_inventory(sums_path)
    if set(checksums) != required - {"SHA256SUMS"}:
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "checksum inventory mismatch")
    expected_parent = config.document.get("parent_artifacts", {}).get("eligibility", {})
    expected_payloads = expected_parent.get("payload_sha256", {}) if isinstance(expected_parent, Mapping) else {}
    if dict(checksums) != dict(expected_payloads):
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "config-bound checksum mapping mismatch")
    for name, digest in checksums.items():
        if _sha_file(path / name) != digest:
            raise Phase4D4ProtocolAVerifierError("eligibility parent", f"checksum mismatch for {name}")
    if checksums.get("config.json") != PARENT_CONFIG_SHA256:
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "config identity mismatch")
    manifest = json.loads((path / "manifest.json").read_bytes())
    gate = json.loads((path / "gate.json").read_bytes())
    marker = json.loads((path / "failed.json").read_bytes())
    if manifest.get("run_id") != PARENT_RUN_ID or manifest.get("status") != "fail":
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "manifest mismatch")
    if marker.get("run_id") != PARENT_RUN_ID or marker.get("status") != "fail":
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "terminal marker mismatch")
    full = gate.get("full_domain_core", {})
    if full.get("state") != "evaluable" or gate.get("marker_filename") != "failed.json":
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "full-domain authorization mismatch")
    for perturbation_id in PERTURBATION_IDS:
        row = full.get("by_perturbation", {}).get(perturbation_id, {})
        if row.get("state") != "evaluable" or row.get("complete_record_count") != 7680 or row.get("complete_well_count") != 240:
            raise Phase4D4ProtocolAVerifierError("eligibility parent", f"{perturbation_id} denominator mismatch")
        counts = row.get("state_counts", {})
        if counts.get("not_applicable") != 0 or counts.get("failed_runtime") != 0:
            raise Phase4D4ProtocolAVerifierError("eligibility parent", f"{perturbation_id} failures present")
    peak = gate.get("peak_common_support", {})
    if peak.get("state") != "not_evaluable_coverage" or peak.get("common_record_count") != 860 or peak.get("common_well_count") != 0:
        raise Phase4D4ProtocolAVerifierError("eligibility parent", "peak-common ruling mismatch")
    expected_rows = {
        "record_conditions.jsonl": 314880, "blank_conditions.jsonl": 1312,
        "source_records.jsonl": 7712, "model_cells.jsonl": 5,
        "model_role_occurrences.jsonl": 38560, "well_folds.jsonl": 5,
    }
    for name, expected in expected_rows.items():
        with (path / name).open(encoding="utf-8") as stream:
            if sum(1 for _ in stream) != expected:
                raise Phase4D4ProtocolAVerifierError("eligibility parent", f"{name} row-count mismatch")
    return MappingProxyType(
        {
            "checksums": dict(sorted(checksums.items())),
            "full_domain_state": "evaluable",
            "marker_filename": "failed.json",
            "parent_path": str(path),
            "parent_run_id": PARENT_RUN_ID,
            "peak_common_state": "not_evaluable_coverage",
            "record_condition_count": 314880,
            "blank_condition_count": 1312,
            "source_record_count": 7712,
            "model_cell_count": 5,
            "model_role_occurrence_count": 38560,
            "step27_config_sha256": PARENT_CONFIG_SHA256,
            "step27_sha256sums_sha256": PARENT_SHA256SUMS_SHA256,
        }
    )


def _reconstruct_inputs(
    cohort: D4SugarCohort, eligibility: _EligibilityConfig, bridge: Mapping[str, object]
) -> _Inputs:
    if len(cohort.record_ids) != 7680 or len(set(cohort.well_ids)) != 240 or len(cohort.blank_record_ids) != 32:
        raise Phase4D4ProtocolAVerifierError("retained cohort", "denominator mismatch")
    support = np.asarray(eligibility.support_coordinates_cm1, dtype="<f8")
    if support.shape != (1999,) or _array_sha(support) != SUPPORT_F64_SHA256 or _array_sha(support, "<f4") != SUPPORT_F32_SHA256:
        raise Phase4D4ProtocolAVerifierError("retained support", "identity mismatch")
    fold_rows = _read_jsonl_rows(_parent_root() / "well_folds.jsonl")
    role_rows = _read_jsonl_rows(_parent_root() / "model_role_occurrences.jsonl")
    if len(fold_rows) != 5 or len(role_rows) != 38560:
        raise Phase4D4ProtocolAVerifierError("retained splits", "parent ledger count mismatch")
    parent_by_fold = {int(row["fold_index"]): row for row in fold_rows}
    if set(parent_by_fold) != set(range(5)):
        raise Phase4D4ProtocolAVerifierError("retained splits", "parent fold indexes mismatch")
    try:
        audit_document = json.loads((ROOT / PHASE05_AUDIT_RELATIVE_PATH).read_bytes())
        audit_folds = {int(row["fold"]): row for row in audit_document["folds"]}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise Phase4D4ProtocolAVerifierError(
            "retained splits", f"invalid frozen Phase-0.5 fold audit: {error}"
        ) from error
    if set(audit_folds) != set(range(5)):
        raise Phase4D4ProtocolAVerifierError("retained splits", "audit fold indexes mismatch")
    frozen = eligibility.frozen_identities
    fold_by_well: dict[str, int] = {}
    for fold, indexes in enumerate(cohort.folds):
        values = np.asarray(indexes, dtype=np.int64)
        record_ids = tuple(str(cohort.record_ids[int(index)]) for index in values)
        source_members = tuple(str(cohort.source_members[int(index)]) for index in values)
        observed_wells = {str(cohort.well_ids[int(index)]) for index in values}
        well_ids = tuple(str(value) for value in audit_folds[fold]["well_ids"])
        computed = {
            "record_ids_sha256": _sha_bytes(("\n".join(sorted(record_ids)) + "\n").encode()),
            "source_members_sha256": _sha_bytes(("\n".join(source_members) + "\n").encode()),
            "well_ids_sha256": _sha_bytes(("\n".join(well_ids) + "\n").encode()),
        }
        expected = {
            "record_ids_sha256": str(frozen["fold_record_ids_sha256"][fold]),
            "source_members_sha256": str(frozen["fold_source_members_sha256"][fold]),
            "well_ids_sha256": str(frozen["fold_well_ids_sha256"][fold]),
        }
        if (
            values.size != 1536
            or len(well_ids) != 48
            or len(set(well_ids)) != 48
            or set(well_ids) != observed_wells
            or computed != expected
        ):
            raise Phase4D4ProtocolAVerifierError("retained splits", f"fold {fold} identity mismatch")
        parent_row = parent_by_fold.get(fold, {})
        if any(str(parent_row.get(key)) != value for key, value in expected.items()):
            raise Phase4D4ProtocolAVerifierError("retained splits", f"fold {fold} parent mismatch")
        for well_id in well_ids:
            if well_id in fold_by_well:
                raise Phase4D4ProtocolAVerifierError("retained splits", f"duplicate well {well_id}")
            fold_by_well[well_id] = fold
    if set(fold_by_well) != set(cohort.well_ids) or bridge.get("full_domain_state") != "evaluable":
        raise Phase4D4ProtocolAVerifierError("retained splits", "coverage mismatch")
    native_axis = np.asarray(cohort.wavenumber, dtype="<f8")
    if native_axis.shape != (2000,) or not np.array_equal(native_axis[1:], support):
        raise Phase4D4ProtocolAVerifierError("retained support", "literal drop-first mismatch")
    native = tuple(
        Spectrum1D(
            spectrum_id=f"d4_sugar_low_snr::{record_id}", sample_id=str(well_id),
            axis_cm1=native_axis, intensity=np.asarray(values, dtype="<f8"),
        )
        for record_id, well_id, values in zip(cohort.record_ids, cohort.well_ids, cohort.intensity, strict=True)
    )
    blanks = tuple(
        Spectrum1D(
            spectrum_id=f"d4_blank::{record_id}", sample_id=str(well_id),
            axis_cm1=native_axis, intensity=np.asarray(values, dtype="<f8"),
        )
        for record_id, well_id, values in zip(
            cohort.blank_record_ids, cohort.blank_well_ids, cohort.blank_intensity, strict=True
        )
    )
    alpha0 = np.ascontiguousarray(np.asarray(cohort.intensity, dtype="<f8")[:, 1:], dtype="<f4")
    train = {int(split.test_fold): np.asarray(split.train_indices, dtype="<i8") for split in cohort.splits}
    validation = {int(split.test_fold): np.asarray(split.validation_indices, dtype="<i8") for split in cohort.splits}
    test = {int(split.test_fold): np.asarray(split.test_indices, dtype="<i8") for split in cohort.splits}
    for fold in range(5):
        if (train[fold].size, validation[fold].size, test[fold].size) != (4608, 1536, 1536):
            raise Phase4D4ProtocolAVerifierError("retained splits", f"fold {fold} role denominator mismatch")
    expected_roles: set[tuple[int, str, str]] = set()
    for fold in range(5):
        for role, indexes in (("train", train[fold]), ("validation", validation[fold]), ("test", test[fold])):
            expected_roles.update((fold, str(cohort.record_ids[int(index)]), role) for index in indexes)
        expected_roles.update((fold, str(record_id), "blank_auxiliary") for record_id in cohort.blank_record_ids)
    actual_roles = {(int(row["seed"]), str(row["record_id"]), str(row["role"])) for row in role_rows}
    if actual_roles != expected_roles or len(actual_roles) != len(role_rows):
        raise Phase4D4ProtocolAVerifierError("retained splits", "model-role ledger mismatch")
    return _Inputs(
        record_ids=tuple(str(value) for value in cohort.record_ids),
        well_ids=tuple(dict.fromkeys(str(value) for value in cohort.well_ids)),
        record_well_ids=tuple(str(value) for value in cohort.well_ids),
        fold_by_well=MappingProxyType(dict(sorted(fold_by_well.items()))),
        rounds=tuple(int(value) for value in np.asarray(cohort.rounds, dtype="<i8").tolist()),
        repetitions=tuple(int(value) for value in np.asarray(cohort.repetitions, dtype="<i8").tolist()),
        targets=np.asarray(cohort.targets, dtype="<f8"),
        blank_record_ids=tuple(str(value) for value in cohort.blank_record_ids),
        support_axis_cm1=np.ascontiguousarray(support, dtype="<f8"),
        native_spectra=native,
        native_blank_spectra=blanks,
        alpha0_projected=alpha0,
        train_indices_by_fold=MappingProxyType(train),
        validation_indices_by_fold=MappingProxyType(validation),
        test_indices_by_fold=MappingProxyType(test),
        eligibility_config=eligibility,
    )


def _project_support(spectrum: Spectrum1D, config: _EligibilityConfig) -> np.ndarray:
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    intensity = np.asarray(spectrum.intensity, dtype="<f8")
    support = np.asarray(config.support_coordinates_cm1, dtype="<f8")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.shape != intensity.shape:
        raise Phase4D4ProtocolAVerifierError("support projection", "unaligned arrays")
    left = int(np.searchsorted(axis, support[0], side="left"))
    candidate_axis = np.asarray(axis[left:], dtype="<f8")
    if candidate_axis.size == support.size and np.array_equal(candidate_axis, support):
        result = np.ascontiguousarray(intensity[left:], dtype="<f4")
    else:
        if float(axis[0]) > float(support[0]) or float(axis[-1]) < float(support[-1]):
            raise Phase4D4ProtocolAVerifierError("support projection", "extrapolation required")
        right = int(np.searchsorted(axis, support[-1], side="right"))
        native = axis[left:right]
        if native.size < 2 or float(np.max(np.diff(native))) > config.support_max_gap_cm1:
            raise Phase4D4ProtocolAVerifierError("support projection", "invalid in-range support")
        result = np.ascontiguousarray(np.interp(support, axis, intensity), dtype="<f4")
    if result.shape != (1999,) or not np.isfinite(result).all():
        raise Phase4D4ProtocolAVerifierError("support projection", "invalid output")
    return result


def _phase1_source(spectrum: Spectrum1D, order: int) -> Phase1Source:
    record_id = spectrum.spectrum_id.split("::", 1)[1]
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=order,
            record_id=record_id,
            sample_id=str(spectrum.sample_id or record_id),
            class_label=0,
            mineral_name="d4-sugar",
            axis_id=f"native::{_array_sha(spectrum.axis_cm1)}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(spectrum.axis_cm1, "<f4"),
        source_intensity_float32_sha256=_array_sha(spectrum.intensity, "<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1),
        normalized_intensity_float64_sha256=_array_sha(spectrum.intensity),
        provenance=MappingProxyType(
            {
                "license": None,
                "license_status": "not_stated",
                "retrieved_date": "2026-08-24",
                "sha256": "0" * 64,
                "source_artifact": "d4_sugar_low_snr",
                "source_url": "local://d4_sugar_low_snr",
            }
        ),
    )


def _cwt_receipt(system: object, spectrum: Spectrum1D) -> Mapping[str, object]:
    try:
        receipt = run_peak_detection_system(system, spectrum)
    except Exception as error:
        return MappingProxyType(
            {
                "state": "runtime_failure",
                "diagnostics_sha256": _receipt_hash({"message": str(error), "type": type(error).__name__}),
                "peak_list_sha256": None,
                "warning_sha256": None,
            }
        )
    return MappingProxyType(
        {
            "state": "complete",
            "diagnostics_sha256": _receipt_hash(getattr(receipt, "diagnostics", None)),
            "peak_list_sha256": str(getattr(receipt, "peaks_sha256", _receipt_hash(getattr(receipt, "peaks", None)))),
            "warning_sha256": _receipt_hash(getattr(receipt, "warnings", None)),
            "_peak_receipt": receipt,
        }
    )


def _metric_objects() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "mse": MSEMetric(),
            "rmse": RMSEMetric(),
            "mae": MAEMetric(),
            "sam": SAMMetric(),
            "pearson_r": PearsonRMetric(),
            "nmse": NMSEMetric(),
            "wasserstein_1_cm1": Wasserstein1Metric(),
            "is_like_structure_to_noise": ISLikeStructureToNoiseMetric(),
        }
    )


def _metric_value(result: object, output_id: str) -> float:
    matches = [item for item in result.outputs if item.output_id == output_id]
    if len(matches) != 1:
        raise Phase4D4ProtocolAVerifierError("science", f"metric {output_id} did not resolve once")
    value = float(matches[0].value)
    if not math.isfinite(value):
        raise Phase4D4ProtocolAVerifierError("science", f"metric {output_id} is nonfinite")
    return value


def _measurement(
    *,
    record_order: int,
    record_id: str,
    well_id: str,
    fold: int,
    condition_id: str,
    source: Spectrum1D,
    output: Spectrum1D,
    projected: np.ndarray,
    source_cwt: Mapping[str, object],
    output_cwt: Mapping[str, object],
) -> Mapping[str, object]:
    metric_rows: dict[str, Mapping[str, object]] = {}
    for output_id, metric in _metric_objects().items():
        request = SingleSpectrumInput(output) if output_id == "is_like_structure_to_noise" else SpectrumPairInput(source, output)
        result = evaluate_metric(metric, request)
        metric_rows[output_id] = {
            "diagnostics_digest": _receipt_hash(result.diagnostics),
            "output_id": output_id,
            "result_digest": _receipt_hash(result.outputs),
            "state": "complete",
            "value": _metric_value(result, output_id),
        }
    peak_result = evaluate_metric(
        PeakDetectionCurvesMetric(),
        PeakPairInput(
            reference_peaks=tuple(item.to_peak1d() for item in source_cwt["_peak_receipt"].peaks),
            candidate_peaks=tuple(item.to_peak1d() for item in output_cwt["_peak_receipt"].peaks),
            position_tolerance_cm1=2.0,
            prominence_thresholds=(0.0,),
        ),
    )
    for output_id in METRIC_OUTPUT_IDS[8:]:
        metric_rows[output_id] = {
            "diagnostics_digest": _receipt_hash(peak_result.diagnostics),
            "output_id": output_id,
            "result_digest": _receipt_hash(peak_result.outputs),
            "state": "complete",
            "value": _metric_value(peak_result, output_id),
        }
    support_sha = _array_sha(projected, "<f4")
    peak_digest = str(output_cwt["peak_list_sha256"])
    if condition_id != "alpha0":
        peak_digest = _receipt_hash(
            {"cwt_peak_list_sha256": peak_digest, "support_intensity_sha256": support_sha}
        )
    return {
        "alpha": _condition_alpha(condition_id),
        "condition_id": condition_id,
        "cwt": {
            "diagnostics_digest": output_cwt["diagnostics_sha256"],
            "peak_count": len(output_cwt["_peak_receipt"].peaks),
            "peak_list_digest": peak_digest,
            "state": str(output_cwt["state"]),
            "warning_digest": output_cwt["warning_sha256"],
        },
        "fold": fold,
        "metric_values": [metric_rows[metric_id] for metric_id in METRIC_OUTPUT_IDS],
        "native_axis_sha256": _array_sha(output.axis_cm1),
        "native_intensity_sha256": _array_sha(output.intensity),
        "perturbation_id": _condition_perturbation(condition_id),
        "projected_row_sha256": support_sha,
        "record_id": record_id,
        "record_order": record_order,
        "state": "complete",
        "well_id": well_id,
    }


_WORKER_ELIGIBILITY: _EligibilityConfig | None = None
_WORKER_SWEEP = None
_WORKER_PHASE1 = None
_WORKER_CWT = None


def _initialize_worker(eligibility_path: str, sweep_path: str, phase1_path: str, catalog_path: str) -> None:
    global _WORKER_ELIGIBILITY, _WORKER_SWEEP, _WORKER_PHASE1, _WORKER_CWT
    _WORKER_ELIGIBILITY = _load_eligibility_config(Path(eligibility_path))
    _WORKER_SWEEP = load_perturbation_sweep_config(Path(sweep_path))
    _WORKER_PHASE1 = load_phase1_core_config(Path(phase1_path))
    catalog = load_classical_catalog(Path(catalog_path))
    matches = [system for system in catalog.systems if system.system_id == _WORKER_ELIGIBILITY.cwt_system_id]
    if len(matches) != 1:
        raise Phase4D4ProtocolAVerifierError("science worker", "CWT system did not resolve once")
    _WORKER_CWT = matches[0]


def _execute_science_record(job: tuple[str, int, int, Spectrum1D]) -> Mapping[str, object]:
    if _WORKER_ELIGIBILITY is None or _WORKER_SWEEP is None or _WORKER_PHASE1 is None or _WORKER_CWT is None:
        raise Phase4D4ProtocolAVerifierError("science worker", "not initialized")
    scope, record_order, fold, spectrum = job
    if scope not in {"mixture", "blank"}:
        raise Phase4D4ProtocolAVerifierError("science worker", f"invalid scope {scope}")
    blank = scope == "blank"
    record_id = spectrum.spectrum_id.split("::", 1)[1]
    well_id = str(spectrum.sample_id)
    projections: dict[str, np.ndarray] = {}
    measurements: list[Mapping[str, object]] = []
    blank_receipts: list[Mapping[str, object]] = []
    with threadpool_limits(limits=1):
        source_cwt = _cwt_receipt(_WORKER_CWT, spectrum)
        if source_cwt.get("state") != "complete":
            raise Phase4D4ProtocolAVerifierError("science", f"alpha-zero CWT failed {record_id}")
        alpha0 = _project_support(spectrum, _WORKER_ELIGIBILITY)
        projections["alpha0"] = alpha0
        if blank:
            blank_receipts.append(
                {
                    "condition_id": "alpha0",
                    "diagnostics_sha256": source_cwt["diagnostics_sha256"],
                    "record_id": record_id,
                    "result_sha256": _receipt_hash(
                        {
                            "cwt_peak_list_sha256": source_cwt["peak_list_sha256"],
                            "support_intensity_sha256": _array_sha(alpha0, "<f4"),
                        }
                    ),
                    "state": "complete",
                    "warning_sha256": source_cwt["warning_sha256"],
                }
            )
        else:
            measurements.append(
                _measurement(
                    record_order=record_order, record_id=record_id, well_id=well_id, fold=fold,
                    condition_id="alpha0", source=spectrum, output=spectrum, projected=alpha0,
                    source_cwt=source_cwt, output_cwt=source_cwt,
                )
            )
        source = _phase1_source(spectrum, record_order)
        for perturbation_id in PERTURBATION_IDS:
            cell = run_perturbation_cell(
                source, perturbation_id, _WORKER_PHASE1, _WORKER_SWEEP,
                p10_admission=P10MemoryAdmission(_WORKER_ELIGIBILITY.p10_memory_budget_bytes)
                if perturbation_id == "p10" else None,
            )
            if cell.status is not CellStatus.COMPLETE:
                raise Phase4D4ProtocolAVerifierError("science", f"operator failed {record_id}/{perturbation_id}")
            by_alpha = {float(item.alpha): item.result.output for item in cell.records}
            for alpha in POSITIVE_ALPHAS:
                condition_id = f"{perturbation_id}:{struct.pack('<d', alpha).hex()}"
                output = by_alpha.get(float(alpha))
                if output is None:
                    raise Phase4D4ProtocolAVerifierError("science", f"missing {record_id}/{condition_id}")
                projected = _project_support(output, _WORKER_ELIGIBILITY)
                projections[condition_id] = projected
                output_cwt = _cwt_receipt(_WORKER_CWT, output)
                if output_cwt.get("state") != "complete":
                    raise Phase4D4ProtocolAVerifierError("science", f"CWT failed {record_id}/{condition_id}")
                if blank:
                    blank_receipts.append(
                        {
                            "condition_id": condition_id,
                            "diagnostics_sha256": output_cwt["diagnostics_sha256"],
                            "record_id": record_id,
                            "result_sha256": _receipt_hash(
                                {
                                    "cwt_peak_list_sha256": output_cwt["peak_list_sha256"],
                                    "support_intensity_sha256": _array_sha(projected, "<f4"),
                                }
                            ),
                            "state": "complete",
                            "warning_sha256": output_cwt["warning_sha256"],
                        }
                    )
                else:
                    measurements.append(
                        _measurement(
                            record_order=record_order, record_id=record_id, well_id=well_id, fold=fold,
                            condition_id=condition_id, source=spectrum, output=output, projected=projected,
                            source_cwt=source_cwt, output_cwt=output_cwt,
                        )
                    )
    if tuple(projections) != CONDITION_IDS:
        raise Phase4D4ProtocolAVerifierError("science worker", f"condition order mismatch {record_id}")
    return {
        "blank_receipts": tuple(blank_receipts),
        "measurements": tuple(measurements),
        "projections": projections,
        "record_order": record_order,
        "scope": scope,
    }


def _validate_science_bridge(
    measurements: Sequence[Mapping[str, object]], blank_receipts: Sequence[Mapping[str, object]]
) -> None:
    expected_records = {
        (str(row["record_id"]), str(row["condition_id"])): row
        for row in _read_jsonl_rows(_parent_root() / "record_conditions.jsonl")
    }
    expected_blanks = {
        (str(row["record_id"]), str(row["condition_id"])): row
        for row in _read_jsonl_rows(_parent_root() / "blank_conditions.jsonl")
    }
    if len(expected_records) != len(measurements) or len(expected_blanks) != len(blank_receipts):
        raise Phase4D4ProtocolAVerifierError("science bridge", "condition count mismatch")
    for row in measurements:
        key = (str(row["record_id"]), str(row["condition_id"]))
        expected = expected_records.get(key)
        actual_metrics = {str(item["output_id"]): item for item in row["metric_values"]}
        if expected is None or expected.get("state") != row.get("state"):
            raise Phase4D4ProtocolAVerifierError("science bridge", f"condition state mismatch {key}")
        for metric_id in METRIC_OUTPUT_IDS:
            parent_metric = expected["metrics"].get(metric_id)
            actual = actual_metrics.get(metric_id)
            if parent_metric is None or actual is None or any(
                parent_metric.get(parent_name) != actual.get(actual_name)
                for parent_name, actual_name in (
                    ("state", "state"),
                    ("result_sha256", "result_digest"),
                    ("diagnostics_sha256", "diagnostics_digest"),
                )
            ):
                raise Phase4D4ProtocolAVerifierError("science bridge", f"metric mismatch {key}/{metric_id}")
        for parent_name, actual_name in (
            ("state", "state"),
            ("diagnostics_sha256", "diagnostics_digest"),
            ("peak_list_sha256", "peak_list_digest"),
            ("warning_sha256", "warning_digest"),
        ):
            if expected["cwt"].get(parent_name) != row["cwt"].get(actual_name):
                raise Phase4D4ProtocolAVerifierError("science bridge", f"CWT mismatch {key}/{parent_name}")
    for row in blank_receipts:
        key = (str(row["record_id"]), str(row["condition_id"]))
        expected = expected_blanks.get(key)
        if expected is None or any(
            expected.get(name) != row.get(name)
            for name in ("state", "diagnostics_sha256", "result_sha256", "warning_sha256")
        ):
            raise Phase4D4ProtocolAVerifierError("science bridge", f"blank mismatch {key}")


def _rematerialize(inputs: _Inputs, worker_count: int) -> _Science:
    jobs = tuple(
        ("mixture", order, int(inputs.fold_by_well[str(spectrum.sample_id)]), spectrum)
        for order, spectrum in enumerate(inputs.native_spectra)
    ) + tuple(("blank", order, -1, spectrum) for order, spectrum in enumerate(inputs.native_blank_spectra))
    jobs = tuple(sorted(jobs, key=lambda job: (0 if job[0] == "mixture" else 1, job[1])))
    capacity = inputs.eligibility_config.p10_memory_budget_bytes // max(
        estimate_p10_peak_bytes(spectrum.axis_cm1.size) for _, _, _, spectrum in jobs
    )
    process_count = min(worker_count, len(jobs), capacity)
    if process_count < 1:
        raise Phase4D4ProtocolAVerifierError("science", "P10 budget admits no worker")
    condition_rows: dict[str, list[np.ndarray]] = {condition_id: [] for condition_id in CONDITION_IDS}
    blank_rows: dict[str, list[np.ndarray]] = {condition_id: [] for condition_id in CONDITION_IDS}
    measurements: list[Mapping[str, object]] = []
    blank_receipts: list[Mapping[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_worker,
        initargs=(
            str(ROOT / ELIGIBILITY_CONFIG_RELATIVE_PATH),
            str(ROOT / SWEEP_RELATIVE_PATH),
            str(ROOT / PHASE1_CONFIG_RELATIVE_PATH),
            str(ROOT / CATALOG_RELATIVE_PATH),
        ),
    ) as executor:
        for expected_job, result in zip(jobs, executor.map(_execute_science_record, jobs), strict=True):
            if (str(result["scope"]), int(result["record_order"])) != (expected_job[0], expected_job[1]):
                raise Phase4D4ProtocolAVerifierError("science", "worker result order mismatch")
            target = blank_rows if result["scope"] == "blank" else condition_rows
            for condition_id in CONDITION_IDS:
                target[condition_id].append(result["projections"][condition_id])
            measurements.extend(result["measurements"])
            blank_receipts.extend(result["blank_receipts"])
    matrices = MappingProxyType(
        {name: np.ascontiguousarray(rows, dtype="<f4") for name, rows in condition_rows.items()}
    )
    blank_matrices = MappingProxyType(
        {name: np.ascontiguousarray(rows, dtype="<f4") for name, rows in blank_rows.items()}
    )
    if any(value.shape != (7680, 1999) for value in matrices.values()):
        raise Phase4D4ProtocolAVerifierError("science", "mixture matrix denominator mismatch")
    if any(value.shape != (32, 1999) for value in blank_matrices.values()):
        raise Phase4D4ProtocolAVerifierError("science", "blank matrix denominator mismatch")
    _validate_science_bridge(measurements, blank_receipts)
    return _Science(matrices, blank_matrices, tuple(measurements))


def _fit_models(inputs: _Inputs, config: _Config) -> tuple[_Model, ...]:
    models: list[_Model] = []
    if inputs.alpha0_projected.shape != (7680, 1999) or not np.isfinite(inputs.alpha0_projected).all():
        raise Phase4D4ProtocolAVerifierError("model selection", "invalid alpha-zero matrix")
    for fold in range(inputs.model_folds):
        train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
        validation = np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64)
        x_train = inputs.alpha0_projected[train]
        y_train = inputs.targets[train]
        x_validation = inputs.alpha0_projected[validation]
        y_validation = inputs.targets[validation]
        fitted: dict[int, PLSRegression] = {}
        score_rows: list[Mapping[str, object]] = []
        for n_components in config.n_components_grid:
            if n_components < 1 or n_components > min(x_train.shape[0] - 1, x_train.shape[1]):
                raise Phase4D4ProtocolAVerifierError("model selection", f"invalid component count {n_components}")
            estimator = PLSRegression(
                n_components=n_components, scale=True, max_iter=500, tol=1e-6, copy=True
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with threadpool_limits(limits=1, user_api="blas"):
                        estimator.fit(x_train, y_train)
                        predicted = estimator.predict(x_validation)
            except Warning as error:
                raise Phase4D4ProtocolAVerifierError(
                    "model selection", f"fold {fold} warning: {error}"
                ) from error
            except Exception as error:
                raise Phase4D4ProtocolAVerifierError(
                    "model selection", f"fold {fold} fit failed: {error}"
                ) from error
            if not np.isfinite(predicted).all():
                raise Phase4D4ProtocolAVerifierError(
                    "model selection", f"fold {fold} nonfinite validation prediction"
                )
            rmse = np.sqrt(np.mean((predicted - y_validation) ** 2, axis=0))
            score_rows.append(
                {
                    "fold": fold,
                    "macro_normalized_rmse": float(np.mean(rmse / config.target_range_mol_l)),
                    "n_components": int(n_components),
                }
            )
            fitted[int(n_components)] = estimator
        validation_scores = tuple(score_rows)
        selected = min(
            validation_scores,
            key=lambda row: (float(row["macro_normalized_rmse"]), int(row["n_components"])),
        )
        estimator = fitted[int(selected["n_components"])]
        with threadpool_limits(limits=1, user_api="blas"):
            train_predictions = np.asarray(estimator.predict(inputs.alpha0_projected[train]), dtype="<f8")
        state_arrays = (
            estimator._x_mean,
            estimator._y_mean,
            estimator._x_std,
            estimator._y_std,
            estimator.x_weights_,
            estimator.y_weights_,
            estimator.x_loadings_,
            estimator.y_loadings_,
            estimator.x_rotations_,
            estimator.y_rotations_,
            estimator.coef_,
            estimator.intercept_,
        )
        if not all(np.isfinite(np.asarray(value)).all() for value in state_arrays):
            raise Phase4D4ProtocolAVerifierError("model selection", f"fold {fold} nonfinite state")
        digest = _sha_bytes(
            _canonical(
                {
                    "fold": fold,
                    "selected_n_components": selected["n_components"],
                    "state_sha256": [_array_sha(np.asarray(value, dtype="<f8")) for value in state_arrays],
                    "n_iter": [int(value) for value in estimator.n_iter_],
                }
            )
        )
        models.append(
            _Model(
                fold=fold,
                selected_n_components=int(selected["n_components"]),
                validation_scores=validation_scores,
                model_state_digest=digest,
                estimator=estimator,
                alpha0_train_predictions=train_predictions,
            )
        )
    return tuple(models)


def _predictions(
    inputs: _Inputs, config: _Config, models: Sequence[_Model], science: _Science
) -> tuple[
    tuple[Mapping[str, object], ...],
    Mapping[str, np.ndarray],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    rows: list[Mapping[str, object]] = []
    blanks: list[Mapping[str, object]] = []
    predictions_by_condition = {name: np.zeros_like(inputs.targets) for name in CONDITION_IDS}
    measurements = {
        (str(row["record_id"]), str(row["condition_id"])): row
        for row in science.record_measurements
    }
    for model in models:
        test = np.asarray(inputs.test_indices_by_fold[model.fold], dtype=np.int64)
        for condition_id in CONDITION_IDS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error")
                    with threadpool_limits(limits=1, user_api="blas"):
                        predicted = np.asarray(
                            model.estimator.predict(science.condition_spectra[condition_id][test]), dtype="<f8"
                        )
            except Warning as error:
                raise Phase4D4ProtocolAVerifierError(
                    "prediction", f"fold {model.fold} warning: {error}"
                ) from error
            if predicted.shape != (test.size, 4) or not np.isfinite(predicted).all():
                raise Phase4D4ProtocolAVerifierError("prediction", f"fold {model.fold} invalid output")
            predictions_by_condition[condition_id][test] = predicted
            for index, values in zip(test.tolist(), predicted, strict=True):
                rows.append(
                    {
                        "condition_id": condition_id,
                        "config_sha256": config.sha256,
                        "fold": model.fold,
                        "model_state_digest": model.model_state_digest,
                        "predicted_targets": [float(value) for value in values],
                        "projected_row_sha256": _array_sha(
                            science.condition_spectra[condition_id][index], "<f4"
                        ),
                        "record_id": inputs.record_ids[index],
                        "record_order": index,
                        "repetition": int(inputs.repetitions[index]),
                        "round": int(inputs.rounds[index]),
                        "terminal_state": "complete",
                        "true_targets": [float(value) for value in inputs.targets[index]],
                        "well_id": inputs.record_well_ids[index],
                    }
                )
    for model in models:
        for condition_id in CONDITION_IDS:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with threadpool_limits(limits=1, user_api="blas"):
                    predicted = np.asarray(
                        model.estimator.predict(science.blank_condition_spectra[condition_id]), dtype="<f8"
                    )
            if predicted.shape != (32, 4) or not np.isfinite(predicted).all():
                raise Phase4D4ProtocolAVerifierError("prediction", f"fold {model.fold} invalid blank output")
            for record_id, values in zip(inputs.blank_record_ids, predicted, strict=True):
                blanks.append(
                    {
                        "blank_record_id": record_id,
                        "condition_id": condition_id,
                        "fold": model.fold,
                        "model_state_digest": model.model_state_digest,
                        "predicted_targets": [float(value) for value in values],
                        "terminal_state": "complete",
                    }
                )
    ordered_predictions = tuple(
        sorted(rows, key=lambda row: (int(row["record_order"]), CONDITION_IDS.index(str(row["condition_id"]))))
    )
    ordered_measurements = tuple(
        measurements[(record_id, condition_id)]
        for record_id in inputs.record_ids
        for condition_id in CONDITION_IDS
    )
    blank_order = {record_id: index for index, record_id in enumerate(inputs.blank_record_ids)}
    ordered_blanks = tuple(
        sorted(
            blanks,
            key=lambda row: (
                int(row["fold"]),
                blank_order[str(row["blank_record_id"])],
                CONDITION_IDS.index(str(row["condition_id"])),
            ),
        )
    )
    return ordered_predictions, MappingProxyType(predictions_by_condition), ordered_measurements, ordered_blanks


def _regression_metrics(true: np.ndarray, predicted: np.ndarray) -> Mapping[str, object]:
    errors = np.asarray(predicted, dtype="<f8") - np.asarray(true, dtype="<f8")
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    denominator = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(denominator > 0.0, 1.0 - np.sum(errors**2, axis=0) / denominator, 1.0)
    row: dict[str, object] = {
        "macro_mae_mol_l": float(np.mean(mae)),
        "macro_normalized_rmse": float(np.mean(rmse / TARGET_RANGE_MOL_L)),
        "macro_r2": float(np.mean(r2)),
    }
    for analyte, target_name in enumerate(TARGET_NAMES):
        row[f"{target_name}_rmse_mol_l"] = float(rmse[analyte])
        row[f"{target_name}_mae_mol_l"] = float(mae[analyte])
        row[f"{target_name}_r2"] = float(r2[analyte])
    return row


def _well_condition_rows(
    inputs: _Inputs, predictions_by_condition: Mapping[str, np.ndarray]
) -> tuple[Mapping[str, object], ...]:
    indexes_by_well: dict[str, list[int]] = {well_id: [] for well_id in inputs.well_ids}
    for index, well_id in enumerate(inputs.record_well_ids):
        indexes_by_well[well_id].append(index)
    rows: list[Mapping[str, object]] = []
    for well_id in inputs.well_ids:
        indexes = indexes_by_well[well_id]
        if len(indexes) != 32:
            raise Phase4D4ProtocolAVerifierError("well grid", f"{well_id} denominator mismatch")
        baseline: float | None = None
        for condition_id in CONDITION_IDS:
            delta = predictions_by_condition[condition_id][indexes] - inputs.targets[indexes]
            loss = float(np.mean((delta**2) / (TARGET_RANGE_MOL_L**2)))
            if condition_id == "alpha0":
                baseline = loss
                harm = 0.0
            else:
                harm = loss - float(baseline)
            rows.append(
                {
                    "condition_id": condition_id,
                    "downstream_harm": harm,
                    "fold": int(inputs.fold_by_well[well_id]),
                    "loss": loss,
                    "state": "complete",
                    "well_id": well_id,
                }
            )
    return tuple(rows)


def _lod_loq_rows(
    inputs: _Inputs, blank_predictions: Sequence[Mapping[str, object]], models: Sequence[_Model]
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    model_by_fold = {model.fold: model for model in models}
    for fold in range(5):
        model = model_by_fold[fold]
        train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
        for condition_id in CONDITION_IDS:
            selected = [
                row for row in blank_predictions
                if int(row["fold"]) == fold and str(row["condition_id"]) == condition_id
            ]
            predicted = np.asarray([row["predicted_targets"] for row in selected], dtype="<f8")
            for analyte in range(4):
                sigma = float(np.std(predicted[:, analyte], ddof=1))
                true_values = inputs.targets[train, analyte]
                predicted_values = model.alpha0_train_predictions[:, analyte]
                centered = true_values - float(np.mean(true_values))
                denominator = float(np.sum(centered**2))
                slope = (
                    float(np.sum(centered * (predicted_values - float(np.mean(predicted_values)))) / denominator)
                    if denominator > 0.0 else float("nan")
                )
                if not np.isfinite(slope) or slope <= 0.0:
                    rows.append(
                        {
                            "analyte_index": analyte, "condition_id": condition_id, "fold": fold,
                            "ich_lod": None, "ich_loq": None, "iupac_lod": None,
                            "sigma": sigma if np.isfinite(sigma) else None,
                            "slope": slope if np.isfinite(slope) else None,
                            "state": "not_evaluable_nonpositive_slope",
                        }
                    )
                else:
                    rows.append(
                        {
                            "analyte_index": analyte, "condition_id": condition_id, "fold": fold,
                            "ich_lod": float(3.3 * sigma / slope),
                            "ich_loq": float(10.0 * sigma / slope),
                            "iupac_lod": float(3.0 * sigma / slope),
                            "sigma": sigma, "slope": slope, "state": "complete",
                        }
                    )
    return tuple(rows)


def _metric_value_for_record(row: Mapping[str, object], metric_id: str) -> float:
    for item in row["metric_values"]:
        if item["output_id"] == metric_id:
            return float(item["value"])
    raise Phase4D4ProtocolAVerifierError("metric lookup", f"missing {metric_id}")


def _well_observations(
    inputs: _Inputs,
    record_measurements: Sequence[Mapping[str, object]],
    well_conditions: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    measurements = {(str(row["record_id"]), str(row["condition_id"])): row for row in record_measurements}
    losses = {(str(row["well_id"]), str(row["condition_id"])): row for row in well_conditions}
    indexes_by_well: dict[str, list[int]] = {well_id: [] for well_id in inputs.well_ids}
    for index, well_id in enumerate(inputs.record_well_ids):
        indexes_by_well[well_id].append(index)
    rows: list[Mapping[str, object]] = []
    for well_id in inputs.well_ids:
        indexes = indexes_by_well[well_id]
        baseline = {
            metric_id: [
                _metric_value_for_record(measurements[(inputs.record_ids[index], "alpha0")], metric_id)
                for index in indexes
            ]
            for metric_id in METRIC_OUTPUT_IDS
        }
        for condition_id in CONDITION_IDS[1:]:
            downstream_harm = float(losses[(well_id, condition_id)]["downstream_harm"])
            for metric_id in METRIC_OUTPUT_IDS:
                current = [
                    _metric_value_for_record(measurements[(inputs.record_ids[index], condition_id)], metric_id)
                    for index in indexes
                ]
                if metric_id in LOWER_IS_BETTER:
                    metric_harm = float(np.mean(np.asarray(current) - np.asarray(baseline[metric_id])))
                else:
                    metric_harm = float(np.mean(np.asarray(baseline[metric_id]) - np.asarray(current)))
                rows.append(
                    {
                        "acquisition_count": len(indexes),
                        "alpha": _condition_alpha(condition_id),
                        "condition_id": condition_id,
                        "downstream_harm": downstream_harm,
                        "metric_harm": metric_harm,
                        "metric_output_id": metric_id,
                        "perturbation_id": _condition_perturbation(condition_id),
                        "state": "complete",
                        "well_id": well_id,
                    }
                )
    return tuple(rows)


def _aggregate(well_observations: Sequence[Mapping[str, object]], config: _Config) -> _Projection:
    tables: dict[str, tuple[AlignmentObservation, ...]] = {}
    for metric_id in METRIC_OUTPUT_IDS:
        selected = [row for row in well_observations if row["metric_output_id"] == metric_id]
        tables[metric_id] = tuple(
            AlignmentObservation(
                cluster_id=str(row["well_id"]),
                perturbation_id=str(row["perturbation_id"]),
                alpha=float(row["alpha"]),
                metric_harm=float(row["metric_harm"]),
                downstream_harm=float(row["downstream_harm"]),
            )
            for row in selected
        )
    clusters = len({str(row["well_id"]) for row in well_observations})
    reference = tables["mse"]
    reference_gap = alignment_gap(reference)
    reference_acc = cross_perturbation_accuracy(reference)
    reference_boot = bulk_paired_cluster_bootstrap(
        reference,
        reference,
        resamples=config.bootstrap_resamples,
        confidence_level=config.confidence_level,
        random_seed=config.random_seed,
    )
    alignment_rows: list[Mapping[str, object]] = [
        {
            "acc_cross": reference_acc.accuracy,
            "acc_interval": reference_boot.reference_acc_interval,
            "ag": reference_gap.alignment_gap,
            "ag_interval": reference_boot.reference_ag_interval,
            "ag_raw": reference_gap.raw_alignment_gap,
            "clusters": clusters,
            "cross_pair_count": reference_acc.pair_count,
            "metric_output_id": "mse",
            "observation_count": len(reference),
            "state": "complete",
        }
    ]
    bootstrap_rows: list[Mapping[str, object]] = []
    sign_rows: list[Mapping[str, object]] = []
    p_values: dict[str, float] = {}
    contrasts: dict[str, float] = {}
    for metric_id in METRIC_OUTPUT_IDS[1:]:
        candidate = tables[metric_id]
        comparison = compare_alignment(reference, candidate)
        boot = bulk_paired_cluster_bootstrap(
            reference,
            candidate,
            resamples=config.bootstrap_resamples,
            confidence_level=config.confidence_level,
            random_seed=config.random_seed,
        )
        bootstrap_rows.append(
            {
                "candidate_acc_interval": boot.candidate_acc_interval,
                "candidate_ag_interval": boot.candidate_ag_interval,
                "d_acc_interval": boot.d_acc_interval,
                "d_ag_interval": boot.d_ag_interval,
                "metric_output_id": metric_id,
                "resamples": config.bootstrap_resamples,
                "state": "complete",
            }
        )
        alignment_rows.append(
            {
                "acc_cross": comparison.candidate_accuracy.accuracy,
                "acc_interval": boot.candidate_acc_interval,
                "ag": comparison.candidate_gap.alignment_gap,
                "ag_interval": boot.candidate_ag_interval,
                "ag_raw": comparison.candidate_gap.raw_alignment_gap,
                "clusters": clusters,
                "cross_pair_count": comparison.candidate_accuracy.pair_count,
                "d_acc": comparison.d_acc,
                "d_acc_interval": boot.d_acc_interval,
                "d_ag": comparison.d_ag,
                "d_ag_interval": boot.d_ag_interval,
                "metric_output_id": metric_id,
                "observation_count": len(candidate),
                "state": "complete",
            }
        )
        for statistic, contributions, contrast in (
            ("d_ag", [item.value for item in comparison.ag_contribution_differences], comparison.d_ag),
            ("d_acc", [item.value for item in comparison.acc_contribution_differences], comparison.d_acc),
        ):
            sign = paired_contribution_sign_flip(
                contributions,
                aggregation="sum" if statistic == "d_ag" else "mean",
                resamples=config.sign_flip_resamples,
                random_seed=config.random_seed,
            )
            hypothesis_id = f"{metric_id}:{statistic}"
            p_values[hypothesis_id] = sign.p_value
            contrasts[hypothesis_id] = contrast
            sign_rows.append(
                {
                    "contrast": contrast,
                    "hypothesis_id": hypothesis_id,
                    "metric_output_id": metric_id,
                    "p_value": sign.p_value,
                    "resamples": config.sign_flip_resamples,
                    "state": "complete",
                    "statistic": statistic,
                }
            )
    family_input = {hypothesis_id: p_values.get(hypothesis_id, 1.0) for hypothesis_id in sorted(p_values)}
    adjusted = {row.hypothesis_id: row for row in holm_step_down(family_input, alpha=config.holm_alpha)}
    holm_rows: list[Mapping[str, object]] = []
    for metric_id in METRIC_OUTPUT_IDS[1:]:
        for statistic in ("d_ag", "d_acc"):
            hypothesis_id = f"{metric_id}:{statistic}"
            result = adjusted[hypothesis_id]
            contrast = contrasts[hypothesis_id]
            favorable = contrast > 0.0
            holm_rows.append(
                {
                    "adjusted_p_value": result.adjusted_p_value,
                    "favorable": favorable,
                    "family_size": result.family_size,
                    "hypothesis_id": hypothesis_id,
                    "metric_output_id": metric_id,
                    "multiplicity_p_value": family_input[hypothesis_id],
                    "observed_contrast": contrast,
                    "rank": result.rank,
                    "raw_p_value": result.raw_p_value,
                    "rejected": bool(result.rejected and favorable),
                    "state": "tested",
                    "statistic": statistic,
                }
            )
    return _Projection(
        tuple(well_observations),
        tuple(alignment_rows),
        tuple(bootstrap_rows),
        tuple(sign_rows),
        tuple(holm_rows),
    )


def _render(projection: _Projection) -> Mapping[str, bytes]:
    metric_states = {str(row["metric_output_id"]): str(row["state"]) for row in projection.alignment_results}
    figure1_rows: list[Mapping[str, object]] = []
    for metric_id in METRIC_OUTPUT_IDS:
        for perturbation_id in PERTURBATION_IDS:
            for alpha in POSITIVE_ALPHAS:
                selected = [
                    row for row in projection.well_observations
                    if row["metric_output_id"] == metric_id
                    and row["perturbation_id"] == perturbation_id
                    and float(row["alpha"]) == float(alpha)
                ]
                figure1_rows.append(
                    {
                        "alpha": alpha,
                        "mean_downstream_harm": float(np.mean([row["downstream_harm"] for row in selected])),
                        "mean_metric_harm": float(np.mean([row["metric_harm"] for row in selected])),
                        "metric_output_id": metric_id,
                        "metric_state": metric_states[metric_id],
                        "perturbation_id": perturbation_id,
                    }
                )
    family = {(row["metric_output_id"], row["statistic"]): row for row in projection.holm_family}
    figure2_rows: list[Mapping[str, object]] = []
    for row in projection.alignment_results:
        metric_id = str(row["metric_output_id"])
        ag_family = family.get((metric_id, "d_ag"), {})
        acc_family = family.get((metric_id, "d_acc"), {})
        figure2_rows.append(
            {
                "acc_cross": row.get("acc_cross"),
                "acc_interval": row.get("acc_interval"),
                "ag": row.get("ag"),
                "ag_interval": row.get("ag_interval"),
                "ag_raw": row.get("ag_raw"),
                "clusters": row.get("clusters"),
                "d_acc": row.get("d_acc"),
                "d_acc_adjusted_p": acc_family.get("adjusted_p_value"),
                "d_acc_favorable": acc_family.get("favorable"),
                "d_acc_interval": row.get("d_acc_interval"),
                "d_acc_rank": acc_family.get("rank"),
                "d_acc_raw_p": acc_family.get("raw_p_value"),
                "d_acc_rejected": acc_family.get("rejected"),
                "d_ag": row.get("d_ag"),
                "d_ag_adjusted_p": ag_family.get("adjusted_p_value"),
                "d_ag_favorable": ag_family.get("favorable"),
                "d_ag_interval": row.get("d_ag_interval"),
                "d_ag_rank": ag_family.get("rank"),
                "d_ag_raw_p": ag_family.get("raw_p_value"),
                "d_ag_rejected": ag_family.get("rejected"),
                "metric_output_id": metric_id,
                "observation_count": row.get("observation_count"),
                "state": row.get("state"),
            }
        )
    payloads: dict[str, bytes] = {
        "figure1_d4_protocol_a_full_domain_data.csv": _csv_bytes(figure1_rows, tuple(figure1_rows[0])),
        "figure2_d4_protocol_a_full_domain_data.csv": _csv_bytes(figure2_rows, tuple(figure2_rows[0])),
        "d4_protocol_a_full_domain_secondary_table.csv": _csv_bytes(figure2_rows, tuple(figure2_rows[0])),
    }
    figure1_plot_rows = tuple(
        csv.DictReader(io.StringIO(payloads["figure1_d4_protocol_a_full_domain_data.csv"].decode()))
    )
    figure2_plot_rows = tuple(
        csv.DictReader(io.StringIO(payloads["figure2_d4_protocol_a_full_domain_data.csv"].decode()))
    )
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "svg.hashsalt": "rpe-phase4-d4-protocol-a-v1",
        }
    ):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        downstream_axis = axes.ravel()[0]
        for perturbation_id in PERTURBATION_IDS:
            selected = [row for row in figure1_plot_rows if row["perturbation_id"] == perturbation_id]
            by_alpha: dict[float, list[float]] = {}
            for row in selected:
                by_alpha.setdefault(float(row["alpha"]), []).append(float(row["mean_downstream_harm"]))
            downstream_axis.plot(
                sorted(by_alpha),
                [float(np.mean(by_alpha[alpha])) for alpha in sorted(by_alpha)],
                marker="o", linewidth=1.5, color=COLOR_MAP[perturbation_id],
            )
        downstream_axis.set_title("normalized squared-loss harm")
        for axis, metric_id in zip(axes.ravel()[1:], METRIC_OUTPUT_IDS, strict=False):
            metric_rows = [row for row in figure1_plot_rows if row["metric_output_id"] == metric_id]
            for perturbation_id in PERTURBATION_IDS:
                selected = [row for row in metric_rows if row["perturbation_id"] == perturbation_id]
                axis.plot(
                    [float(row["mean_metric_harm"]) for row in selected],
                    [float(row["mean_downstream_harm"]) for row in selected],
                    marker="o", linewidth=1.5, color=COLOR_MAP[perturbation_id],
                )
            axis.set_title(metric_id)
        for axis in axes.ravel()[14:]:
            axis.set_axis_off()
        fig.tight_layout()
        png, svg = io.BytesIO(), io.BytesIO()
        fig.savefig(png, format="png", dpi=300, metadata={"Date": None})
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
        payloads["figure1_d4_protocol_a_full_domain.png"] = png.getvalue()
        payloads["figure1_d4_protocol_a_full_domain.svg"] = svg.getvalue()

        fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        for axis, field, title, color in zip(
            axes,
            ("ag", "acc_cross", "d_ag", "d_acc"),
            ("AG", "Acc-cross", "D_AG", "D_Acc"),
            ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"),
            strict=True,
        ):
            values = [0.0 if not row.get(field) else float(row[field]) for row in figure2_plot_rows]
            ypos = np.arange(len(METRIC_OUTPUT_IDS))
            axis.barh(ypos, values, color=color)
            axis.set_title(title)
            axis.set_yticks(ypos, METRIC_OUTPUT_IDS if axis is axes[0] else [])
        fig.tight_layout()
        png, svg = io.BytesIO(), io.BytesIO()
        fig.savefig(png, format="png", dpi=300, metadata={"Date": None})
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
        payloads["figure2_d4_protocol_a_full_domain.png"] = png.getvalue()
        payloads["figure2_d4_protocol_a_full_domain.svg"] = svg.getvalue()
    return MappingProxyType(payloads)


def _build_payloads(
    *, inputs: _Inputs, config: _Config, bridge: Mapping[str, object], worker_count: int
) -> tuple[Mapping[str, bytes], Phase4D4ProtocolASummary]:
    science = _rematerialize(inputs, worker_count)
    models = _fit_models(inputs, config)
    prediction_rows, predictions_by_condition, record_measurements, blank_predictions = _predictions(
        inputs, config, models, science
    )
    well_conditions = _well_condition_rows(inputs, predictions_by_condition)
    lod_loq = _lod_loq_rows(inputs, blank_predictions, models)
    well_observations = _well_observations(inputs, record_measurements, well_conditions)
    projection = _aggregate(well_observations, config)
    summary_rows = [
        {
            "condition_id": condition_id,
            "count": len(inputs.record_ids),
            **_regression_metrics(inputs.targets, predictions_by_condition[condition_id]),
        }
        for condition_id in CONDITION_IDS
    ]
    figures = _render(projection)
    run_id = RUN_PREFIX + _sha_bytes(
        _canonical({"record_count": len(inputs.record_ids), "sha256": config.sha256})
    )
    model_rows = [
        {
            "condition_count": len(CONDITION_IDS),
            "fold": model.fold,
            "model_state_digest": model.model_state_digest,
            "refit_with_validation": False,
            "selected_n_components": model.selected_n_components,
        }
        for model in models
    ]
    validation_rows = [row for model in models for row in model.validation_scores]
    observed_counts = {
        "models": len(models), "validation": len(validation_rows),
        "measurements": len(record_measurements), "predictions": len(prediction_rows),
        "blanks": len(blank_predictions), "wells": len(well_conditions),
        "lod": len(lod_loq), "observations": len(well_observations),
    }
    expected_counts = {
        "models": 5, "validation": 25, "measurements": 314880,
        "predictions": 314880, "blanks": 6560, "wells": 9840,
        "lod": 820, "observations": 124800,
    }
    if observed_counts != expected_counts:
        raise Phase4D4ProtocolAVerifierError("artifact rows", f"denominator mismatch {observed_counts}")
    preflight = {
        "alignment_state": "complete",
        "bootstrap_resamples": config.bootstrap_resamples,
        "claim_boundary": "preflight_complete",
        "parent_bridge_state": bridge["full_domain_state"],
        "sign_flip_resamples": config.sign_flip_resamples,
        "synthetic_fixture": False,
        "worker_independence": "serialized_bytes_do_not_depend_on_worker_count",
    }
    manifest = {
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": "local_execution_artifact_redistribution_not_cleared",
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "alignment_results": len(projection.alignment_results),
            "blank_predictions": len(blank_predictions),
            "bootstrap_results": len(projection.bootstrap_results),
            "configured_payloads": len(ARTIFACT_PAYLOAD_FILES),
            "figure1_rows": 13 * 5 * 8,
            "figure2_rows": 13,
            "holm_family": len(projection.holm_family),
            "model_cells": len(model_rows),
            "prediction_rows": len(prediction_rows),
            "record_measurements": len(record_measurements),
            "sign_flip_results": len(projection.sign_flip_results),
            "technical_lod_loq": len(lod_loq),
            "validation_scores": len(validation_rows),
            "well_conditions": len(well_conditions),
            "well_observations": len(well_observations),
        },
        "environment": dict(_environment()),
        "experiment_id": EXPERIMENT_ID,
        "metric_states": {row["metric_output_id"]: row["state"] for row in projection.alignment_results},
        "protocol": "A",
        "run_id": run_id,
        "status": "complete",
        "synthetic_fixture": False,
        "tier": "full_domain_core",
    }
    payloads: dict[str, bytes] = {
        "config.json": config.raw_bytes,
        "eligibility_bridge.json": _canonical(dict(bridge)),
        "preflight.json": _canonical(preflight),
        "model_cells.jsonl": _jsonl_bytes(model_rows),
        "validation_scores.jsonl": _jsonl_bytes(validation_rows),
        "record_measurements.jsonl": _jsonl_bytes(record_measurements),
        "predictions.jsonl": _jsonl_bytes(prediction_rows),
        "blank_predictions.jsonl": _jsonl_bytes(blank_predictions),
        "well_conditions.jsonl": _jsonl_bytes(well_conditions),
        "technical_lod_loq.jsonl": _jsonl_bytes(lod_loq),
        "condition_summary.csv": _csv_bytes(summary_rows, tuple(summary_rows[0])),
        "well_observations.jsonl": _jsonl_bytes(projection.well_observations),
        "alignment_results.jsonl": _jsonl_bytes(projection.alignment_results),
        "bootstrap_results.jsonl": _jsonl_bytes(projection.bootstrap_results),
        "sign_flip_results.jsonl": _jsonl_bytes(projection.sign_flip_results),
        "holm_family.jsonl": _jsonl_bytes(projection.holm_family),
        "figure1_d4_protocol_a_full_domain.png": figures["figure1_d4_protocol_a_full_domain.png"],
        "figure1_d4_protocol_a_full_domain.svg": figures["figure1_d4_protocol_a_full_domain.svg"],
        "figure1_d4_protocol_a_full_domain_data.csv": figures["figure1_d4_protocol_a_full_domain_data.csv"],
        "figure2_d4_protocol_a_full_domain.png": figures["figure2_d4_protocol_a_full_domain.png"],
        "figure2_d4_protocol_a_full_domain.svg": figures["figure2_d4_protocol_a_full_domain.svg"],
        "figure2_d4_protocol_a_full_domain_data.csv": figures["figure2_d4_protocol_a_full_domain_data.csv"],
        "d4_protocol_a_full_domain_secondary_table.csv": figures["d4_protocol_a_full_domain_secondary_table.csv"],
        "manifest.json": _canonical(manifest),
    }
    marker = _canonical({"run_id": run_id, "status": "complete"})
    payloads["complete.json"] = marker
    payloads["SHA256SUMS"] = b"".join(
        f"{_sha_bytes(payloads[name])}  {name}\n".encode()
        for name in (*ARTIFACT_PAYLOAD_FILES, "complete.json")
    )
    return MappingProxyType(payloads), Phase4D4ProtocolASummary(
        Path("."), run_id, "complete", len(inputs.record_ids), len(prediction_rows)
    )


def _validate_candidate(path: Path) -> None:
    if not path.is_dir():
        raise Phase4D4ProtocolAVerifierError("artifact", "path must be an existing directory")
    observed = {item.name for item in path.iterdir()}
    expected = set(ARTIFACT_PAYLOAD_FILES) | {"complete.json", "SHA256SUMS"}
    if observed != expected:
        raise Phase4D4ProtocolAVerifierError("artifact inventory", f"unexpected files {sorted(observed)!r}")
    checksums = _checksum_inventory(path / "SHA256SUMS")
    expected_order = ARTIFACT_PAYLOAD_FILES + ("complete.json",)
    if tuple(checksums) != expected_order:
        raise Phase4D4ProtocolAVerifierError("SHA256SUMS", "canonical entry order mismatch")
    for name, digest in checksums.items():
        if _sha_file(path / name) != digest:
            raise Phase4D4ProtocolAVerifierError("SHA256SUMS", f"digest mismatch for {name}")
    manifest = json.loads((path / "manifest.json").read_bytes())
    marker = json.loads((path / "complete.json").read_bytes())
    if manifest.get("status") != "complete" or marker.get("status") != "complete":
        raise Phase4D4ProtocolAVerifierError("terminal marker", "status mismatch")
    if marker.get("run_id") != manifest.get("run_id"):
        raise Phase4D4ProtocolAVerifierError("terminal marker", "run identity mismatch")


def _write_rebuild(path: Path, payloads: Mapping[str, bytes]) -> None:
    path.mkdir(parents=True, exist_ok=False)
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)


def _compare(candidate: Path, rebuilt: Path) -> None:
    candidate_files = {item.name for item in candidate.iterdir()}
    rebuilt_files = {item.name for item in rebuilt.iterdir()}
    if candidate_files != rebuilt_files:
        raise Phase4D4ProtocolAVerifierError("artifact inventory", "candidate and rebuild differ")
    for name in sorted(candidate_files):
        if (candidate / name).read_bytes() != (rebuilt / name).read_bytes():
            raise Phase4D4ProtocolAVerifierError(name, "candidate and rebuilt bytes differ")


def verify_phase4_d4_protocol_a(path: Path, worker_count: int = 12) -> Phase4D4ProtocolASummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count < 1:
        raise Phase4D4ProtocolAVerifierError("worker_count", "must be a positive integer")
    candidate = Path(path)
    _validate_candidate(candidate)
    config = _load_config(candidate / "config.json")
    bridge = _validate_parent(config)
    cohort = load_d4_sugar_cohort(ROOT / PROTOCOL_CONFIG_RELATIVE_PATH, ROOT / ARCHIVE_RELATIVE_PATH)
    eligibility = _load_eligibility_config(ROOT / ELIGIBILITY_CONFIG_RELATIVE_PATH)
    inputs = _reconstruct_inputs(cohort, eligibility, bridge)
    payloads, rebuilt_summary = _build_payloads(
        inputs=inputs, config=config, bridge=bridge, worker_count=worker_count
    )
    with tempfile.TemporaryDirectory(prefix="phase4-d4-protocol-a-verify-") as temporary:
        rebuilt = Path(temporary) / rebuilt_summary.run_id
        _write_rebuild(rebuilt, payloads)
        _compare(candidate, rebuilt)
    return Phase4D4ProtocolASummary(
        candidate, rebuilt_summary.run_id, rebuilt_summary.status,
        rebuilt_summary.record_count, rebuilt_summary.prediction_row_count,
    )


__all__ = [
    "Phase4D4ProtocolASummary",
    "Phase4D4ProtocolAVerifierError",
    "verify_phase4_d4_protocol_a",
]
