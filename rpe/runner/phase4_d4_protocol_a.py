"""Phase 4 D4 Protocol-A full-domain outcome assembly.

The authoritative entry point reconstructs retained Sugar spectra and split
roles, reexecutes frozen P8--P12 science, and fits fresh held-out PLS2 models.
Synthetic constructors remain available only for small deterministic tests.
"""
from __future__ import annotations

import csv
import hashlib
import io
import importlib
import json
import math
import multiprocessing
import platform
import struct
import tempfile
import warnings
import zipfile
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
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
    AlignmentValidationError,
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
from rpe.runner.phase4_d4_eligibility import (
    Phase4D4EligibilityConfig,
    load_phase4_d4_eligibility_config,
    project_d4_support,
)
from rpe.runner import phase4_d4_eligibility as _step27_science
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
TARGET_NAMES = (
    "sucrose_nominal_mol_l",
    "fructose_nominal_mol_l",
    "maltose_nominal_mol_l",
    "glucose_nominal_mol_l",
)
TARGET_RANGE_MOL_L = 0.32
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


class Phase4D4ProtocolAError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D4ProtocolAConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_output_ids: tuple[str, ...]
    artifact_payload_files: tuple[str, ...]
    parent_marker_filename: str
    parent_full_domain_state: str
    n_components_grid: tuple[int, ...]
    bootstrap_resamples: int
    sign_flip_resamples: int
    random_seed: int
    confidence_level: float
    holm_alpha: float
    target_range_mol_l: float


@dataclass(frozen=True)
class Phase4D4ProtocolASummary:
    path: Path
    run_id: str
    status: str
    record_count: int
    prediction_row_count: int


@dataclass(frozen=True)
class D4ProtocolAModel:
    fold: int
    selected_n_components: int
    refit_with_validation: bool
    condition_count: int
    validation_scores: tuple[Mapping[str, object], ...]
    model_state_digest: str
    estimator: object | None = None
    alpha0_train_predictions: np.ndarray | None = None


@dataclass(frozen=True)
class D4ProtocolAInputs:
    synthetic_fixture: bool
    well_count: int
    acquisitions_per_well: int
    model_folds: int
    feature_count: int
    record_ids: tuple[str, ...]
    well_ids: tuple[str, ...]
    record_well_ids: tuple[str, ...]
    fold_by_well: Mapping[str, int]
    rounds: tuple[int, ...]
    repetitions: tuple[int, ...]
    targets: np.ndarray
    target_names: tuple[str, ...]
    blank_record_ids: tuple[str, ...]
    blank_targets: np.ndarray
    support_axis_cm1: np.ndarray
    validation_macro_nrmse_by_component: Mapping[int, float]
    prediction_fixture: str | None
    blank_fixture: str | None
    native_spectra: tuple[Spectrum1D, ...] = ()
    native_blank_spectra: tuple[Spectrum1D, ...] = ()
    alpha0_projected: np.ndarray | None = None
    alpha0_blank_projected: np.ndarray | None = None
    train_indices_by_fold: Mapping[int, np.ndarray] | None = None
    validation_indices_by_fold: Mapping[int, np.ndarray] | None = None
    test_indices_by_fold: Mapping[int, np.ndarray] | None = None
    eligibility_config: Phase4D4EligibilityConfig | None = None


@dataclass(frozen=True)
class D4RealScience:
    condition_spectra: Mapping[str, np.ndarray]
    blank_condition_spectra: Mapping[str, np.ndarray]
    record_measurements: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class D4OutcomeProjection:
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
            raise Phase4D4ProtocolAError("json", "nonfinite float")
        return value
    raise Phase4D4ProtocolAError("json", f"unsupported value {type(value).__name__}")


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


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


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


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


def _validate_sha256(path: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise Phase4D4ProtocolAError(path, "SHA-256 must contain 64 hexadecimal characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise Phase4D4ProtocolAError(path, "SHA-256 must contain 64 hexadecimal characters") from error
    return value


def _validate_real_config_authorities(
    document: Mapping[str, object], *, verify_live_files: bool
) -> None:
    authorities = document.get("authorities")
    code_authority = document.get("code_authority")
    environment = document.get("environment_authority")
    trust_anchor = document.get("trust_anchor")
    if not isinstance(authorities, Mapping) or not authorities:
        raise Phase4D4ProtocolAError("authorities", "nonempty receipt mapping required")
    if not isinstance(code_authority, Mapping) or not code_authority:
        raise Phase4D4ProtocolAError("code_authority", "nonempty receipt mapping required")
    if not isinstance(environment, Mapping) or dict(environment) != dict(_environment()):
        raise Phase4D4ProtocolAError("environment_authority", "frozen environment mismatch")
    if not isinstance(trust_anchor, Mapping) or (
        trust_anchor.get("config_authority_relative_path")
        != "rpe/runner/phase4_d4_protocol_a_authority.py"
        or trust_anchor.get("config_binds_authority") is not False
        or trust_anchor.get("direction") != "authority_to_config_only"
    ):
        raise Phase4D4ProtocolAError("trust_anchor", "one-way frozen-config authority mismatch")

    member_receipts: list[tuple[str, Mapping[str, object]]] = []
    for key, value in authorities.items():
        if not isinstance(value, Mapping):
            raise Phase4D4ProtocolAError(f"authorities.{key}", "receipt must be an object")
        sha256 = _validate_sha256(f"authorities.{key}.sha256", value.get("sha256"))
        byte_count = value.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise Phase4D4ProtocolAError(f"authorities.{key}.bytes", "must be a positive integer")
        if "path" in value:
            relative = value.get("path")
            if not isinstance(relative, str) or not relative:
                raise Phase4D4ProtocolAError(f"authorities.{key}.path", "must be nonempty")
            if verify_live_files:
                live = ROOT / relative
                if not live.is_file() or live.stat().st_size != byte_count:
                    raise Phase4D4ProtocolAError(f"authorities.{key}", "live file size mismatch")
                # The direct-ZIP loader independently validates the 5.31-GB
                # archive digest while loading; avoid hashing it twice here.
                if relative != ARCHIVE_RELATIVE_PATH and _sha_file(live) != sha256:
                    raise Phase4D4ProtocolAError(f"authorities.{key}", "live file SHA-256 mismatch")
        elif "archive_path" in value and "member_path" in value:
            member_receipts.append((str(key), value))
        else:
            raise Phase4D4ProtocolAError(f"authorities.{key}", "file or archive-member path required")

    for relative, value in code_authority.items():
        if not isinstance(relative, str) or not isinstance(value, Mapping):
            raise Phase4D4ProtocolAError("code_authority", "invalid receipt")
        sha256 = _validate_sha256(f"code_authority.{relative}.sha256", value.get("sha256"))
        byte_count = value.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 1:
            raise Phase4D4ProtocolAError(f"code_authority.{relative}.bytes", "must be positive")
        if verify_live_files:
            live = ROOT / relative
            if (
                not live.is_file()
                or live.stat().st_size != byte_count
                or _sha_file(live) != sha256
            ):
                raise Phase4D4ProtocolAError(f"code_authority.{relative}", "live identity mismatch")

    if verify_live_files and member_receipts:
        archive_path = ROOT / str(member_receipts[0][1]["archive_path"])
        try:
            with zipfile.ZipFile(archive_path) as archive:
                for key, value in member_receipts:
                    if str(value.get("archive_path")) != str(member_receipts[0][1]["archive_path"]):
                        raise Phase4D4ProtocolAError(f"authorities.{key}", "archive path mismatch")
                    payload = archive.read(str(value["member_path"]))
                    if len(payload) != int(value["bytes"]) or _sha_bytes(payload) != str(value["sha256"]):
                        raise Phase4D4ProtocolAError(f"authorities.{key}", "archive member identity mismatch")
        except (OSError, KeyError, zipfile.BadZipFile) as error:
            raise Phase4D4ProtocolAError("authorities", str(error)) from error


def _condition_alpha(condition_id: str) -> float:
    if condition_id == "alpha0":
        return 0.0
    return float(struct.unpack("<d", bytes.fromhex(condition_id.split(":", 1)[1]))[0])


def _condition_perturbation(condition_id: str) -> str:
    return "alpha0" if condition_id == "alpha0" else condition_id.split(":", 1)[0]


def _metric_manifest() -> tuple[Mapping[str, object], ...]:
    lower = {
        "mse",
        "rmse",
        "mae",
        "sam",
        "nmse",
        "wasserstein_1_cm1",
        "artifact_peak_ratio",
        "missing_peak_ratio",
    }
    rows = []
    for metric_id in METRIC_OUTPUT_IDS:
        rows.append(
            {
                "output_id": metric_id,
                "preferred_direction": "lower_is_better" if metric_id in lower else "higher_is_better",
            }
        )
    return tuple(rows)


def parse_phase4_d4_protocol_a_config(
    path: Path, raw: bytes, *, require_frozen_identity: bool
) -> Phase4D4ProtocolAConfig:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D4ProtocolAError("config", str(error)) from error
    if raw != _canonical(document):
        raise Phase4D4ProtocolAError("config", "must use canonical JSON")
    if require_frozen_identity and (len(raw) != CONFIG_BYTES or _sha_bytes(raw) != CONFIG_SHA256):
        raise Phase4D4ProtocolAError("config", "frozen config identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D4ProtocolAError("config", "schema or experiment mismatch")
    synthetic = bool(document.get("synthetic_fixture", False))
    perturbations = tuple(str(value) for value in document.get("perturbation_ids", ()))
    if perturbations != PERTURBATION_IDS:
        raise Phase4D4ProtocolAError("perturbation_ids", "frozen order mismatch")
    alpha_grid = tuple(float(value) for value in document.get("alpha_grid", ()))
    if alpha_grid != ALPHA_GRID:
        raise Phase4D4ProtocolAError("alpha_grid", "frozen values mismatch")
    metric_ids = tuple(str(value) for value in document.get("metric_output_ids", ()))
    if metric_ids != METRIC_OUTPUT_IDS:
        raise Phase4D4ProtocolAError("metric_output_ids", "frozen order mismatch")
    payload_files = tuple(str(value) for value in document.get("artifact_payload_files", ()))
    if payload_files != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D4ProtocolAError("artifact_payload_files", "frozen order mismatch")
    parent = document.get("parent", {})
    if not isinstance(parent, Mapping):
        raise Phase4D4ProtocolAError("parent", "must be an object")
    model_recipe = document.get("model_recipe", {})
    if not isinstance(model_recipe, Mapping):
        raise Phase4D4ProtocolAError("model_recipe", "must be an object")
    inference = document.get("inference", {})
    if not isinstance(inference, Mapping):
        raise Phase4D4ProtocolAError("inference", "must be an object")
    if not synthetic:
        _validate_real_config_authorities(
            document, verify_live_files=require_frozen_identity
        )
    return Phase4D4ProtocolAConfig(
        path=Path(path),
        raw_bytes=raw,
        sha256=_sha_bytes(raw),
        document=_freeze(document),
        synthetic_fixture=synthetic,
        perturbation_ids=perturbations,
        alpha_grid=alpha_grid,
        metric_output_ids=metric_ids,
        artifact_payload_files=payload_files,
        parent_marker_filename=str(parent.get("marker_filename", "failed.json")),
        parent_full_domain_state=str(parent.get("full_domain_state", "evaluable")),
        n_components_grid=tuple(int(value) for value in model_recipe.get("n_components_grid", (2, 4, 8, 16, 32))),
        bootstrap_resamples=int(inference.get("bootstrap_resamples", 2000)),
        sign_flip_resamples=int(inference.get("sign_flip_resamples", 100000)),
        random_seed=int(inference.get("random_seed", 20260817)),
        confidence_level=float(inference.get("confidence_level", 0.95)),
        holm_alpha=float(inference.get("holm_alpha", 0.05)),
        target_range_mol_l=float(document.get("target_range_mol_l", TARGET_RANGE_MOL_L)),
    )


def load_phase4_d4_protocol_a_config(path: Path) -> Phase4D4ProtocolAConfig:
    return parse_phase4_d4_protocol_a_config(Path(path), Path(path).read_bytes(), require_frozen_identity=True)


def _parent_root() -> Path:
    return ROOT / "results/phase4/d4_protocol_a_full_domain_eligibility_v1" / PARENT_RUN_ID


def _read_jsonl_rows(path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise Phase4D4ProtocolAError(str(path), "JSONL rows must be objects")
        rows.append(row)
    return tuple(rows)


def _json_checksum_inventory(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if name in checksums:
            raise Phase4D4ProtocolAError(str(path), f"duplicate checksum entry {name}")
        checksums[name] = digest
    return checksums


def make_synthetic_d4_protocol_a_inputs(
    *,
    well_count: int,
    acquisitions_per_well: int,
    model_folds: int,
    validation_macro_nrmse_by_component: Mapping[int, float] | None = None,
    prediction_fixture: str | None = None,
    blank_fixture: str | None = None,
) -> D4ProtocolAInputs:
    if well_count < model_folds or acquisitions_per_well < 1 or model_folds < 2:
        raise Phase4D4ProtocolAError("synthetic inputs", "invalid well or fold counts")
    record_ids: list[str] = []
    record_well_ids: list[str] = []
    well_ids = tuple(f"well-{index:03d}" for index in range(well_count))
    rounds: list[int] = []
    repetitions: list[int] = []
    targets: list[list[float]] = []
    for well_index, well_id in enumerate(well_ids):
        base = np.asarray(
            (
                0.04 + 0.01 * well_index,
                0.08 + 0.008 * well_index,
                0.12 + 0.006 * well_index,
                0.16 + 0.004 * well_index,
            ),
            dtype="<f8",
        )
        for acquisition in range(acquisitions_per_well):
            record_ids.append(f"{well_id}-rec-{acquisition:02d}")
            record_well_ids.append(well_id)
            rounds.append(acquisition % 8)
            repetitions.append(acquisition % 4)
            targets.append((base + acquisition * 0.002).tolist())
    fold_by_well = {well_id: (index % model_folds) for index, well_id in enumerate(well_ids)}
    blank_count = max(2, acquisitions_per_well)
    blank_record_ids = tuple(f"blank-{index:02d}" for index in range(blank_count))
    blank_targets = np.zeros((blank_count, 4), dtype="<f8")
    if validation_macro_nrmse_by_component is None:
        validation_macro_nrmse_by_component = {2: 0.12, 4: 0.14, 8: 0.18, 16: 0.24, 32: 0.32}
    return D4ProtocolAInputs(
        synthetic_fixture=True,
        well_count=well_count,
        acquisitions_per_well=acquisitions_per_well,
        model_folds=model_folds,
        feature_count=64,
        record_ids=tuple(record_ids),
        well_ids=well_ids,
        record_well_ids=tuple(record_well_ids),
        fold_by_well=MappingProxyType(dict(fold_by_well)),
        rounds=tuple(rounds),
        repetitions=tuple(repetitions),
        targets=np.asarray(targets, dtype="<f8"),
        target_names=TARGET_NAMES,
        blank_record_ids=blank_record_ids,
        blank_targets=blank_targets,
        support_axis_cm1=np.linspace(145.83834838867188, 3684.83544921875, 1999, dtype="<f8"),
        validation_macro_nrmse_by_component=MappingProxyType(
            {int(key): float(value) for key, value in validation_macro_nrmse_by_component.items()}
        ),
        prediction_fixture=prediction_fixture,
        blank_fixture=blank_fixture,
    )


def make_synthetic_d4_protocol_a_config(
    inputs: D4ProtocolAInputs,
    *,
    parent_marker_filename: str = "failed.json",
    parent_full_domain_state: str = "evaluable",
) -> Phase4D4ProtocolAConfig:
    document = {
        "alpha_grid": list(ALPHA_GRID),
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "experiment_id": EXPERIMENT_ID,
        "figure_contract": {
            "colors": dict(COLOR_MAP),
            "dpi": 300,
            "figure1_inches": [12, 12],
            "figure1_layout": [4, 4],
            "figure2_inches": [14, 8],
            "figure2_layout": [1, 4],
            "font_family": "DejaVu Sans",
            "padding_fraction": 0.05,
            "svg_hashsalt": "rpe-phase4-d4-protocol-a-v1",
        },
        "inference": {
            "bootstrap_resamples": 2000,
            "confidence_level": 0.95,
            "holm_alpha": 0.05,
            "random_seed": 20260817,
            "sign_flip_resamples": 100000,
        },
        "metric_manifest": list(_metric_manifest()),
        "metric_output_ids": list(METRIC_OUTPUT_IDS),
        "model_recipe": {
            "condition_count": 41,
            "n_components_grid": [2, 4, 8, 16, 32],
            "refit_with_validation": False,
            "selection": "minimum_validation_macro_normalized_rmse_lowest_components_on_tie",
        },
        "parent": {
            "full_domain_state": parent_full_domain_state,
            "marker_filename": parent_marker_filename,
        },
        "perturbation_ids": list(PERTURBATION_IDS),
        "protocol": "A",
        "schema_version": SCHEMA_VERSION,
        "synthetic_fixture": True,
        "target_range_mol_l": TARGET_RANGE_MOL_L,
        "tier": "full_domain_core",
    }
    raw = _canonical(document)
    return parse_phase4_d4_protocol_a_config(Path("<synthetic>"), raw, require_frozen_identity=False)


def validate_d4_eligibility_parent(path: Path, config: Phase4D4ProtocolAConfig) -> Mapping[str, object]:
    path = Path(path)
    if not config.synthetic_fixture:
        required = {
            "config.json",
            "model_cells.jsonl",
            "common_support.jsonl",
            "manifest.json",
            "gate.json",
            "well_summaries.jsonl",
            "model_role_occurrences.jsonl",
            "source_records.jsonl",
            "blank_conditions.jsonl",
            "operator_cells.jsonl",
            "blank_cells.jsonl",
            "record_conditions.jsonl",
            "well_folds.jsonl",
            "failed.json",
            "SHA256SUMS",
        }
        names = {item.name for item in path.iterdir()} if path.is_dir() else set()
        if names != required:
            raise Phase4D4ProtocolAError("eligibility parent", "exact file inventory mismatch")
        sums_path = path / "SHA256SUMS"
        if _sha_bytes(sums_path.read_bytes()) != PARENT_SHA256SUMS_SHA256:
            raise Phase4D4ProtocolAError("eligibility parent", "SHA256SUMS identity mismatch")
        checksums = _json_checksum_inventory(sums_path)
        if len(checksums) != len(required) - 1 or set(checksums) != required - {"SHA256SUMS"}:
            raise Phase4D4ProtocolAError("eligibility parent", "checksum inventory mismatch")
        for name, digest in checksums.items():
            if _sha_bytes((path / name).read_bytes()) != digest:
                raise Phase4D4ProtocolAError("eligibility parent", f"checksum mismatch for {name}")
        if _sha_bytes((path / "config.json").read_bytes()) != PARENT_CONFIG_SHA256:
            raise Phase4D4ProtocolAError("eligibility parent", "config identity mismatch")
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("run_id") != PARENT_RUN_ID or manifest.get("status") != "fail":
            raise Phase4D4ProtocolAError("eligibility parent", "manifest identity mismatch")
        gate = json.loads((path / "gate.json").read_text(encoding="utf-8"))
        if gate.get("marker_filename") != "failed.json":
            raise Phase4D4ProtocolAError("eligibility parent", "marker filename mismatch")
        if gate.get("full_domain_core", {}).get("state") != "evaluable":
            raise Phase4D4ProtocolAError("eligibility parent", "full_domain_core must be evaluable")
        if gate.get("peak_common_support", {}).get("state") != "not_evaluable_coverage":
            raise Phase4D4ProtocolAError("eligibility parent", "peak_common_support state mismatch")
        expected_rows = {
            "record_conditions.jsonl": 314880,
            "blank_conditions.jsonl": 1312,
            "source_records.jsonl": 7712,
            "model_cells.jsonl": 5,
            "model_role_occurrences.jsonl": 38560,
            "well_folds.jsonl": 5,
        }
        for name, expected_count in expected_rows.items():
            with (path / name).open(encoding="utf-8") as stream:
                observed = sum(1 for _ in stream)
            if observed != expected_count:
                raise Phase4D4ProtocolAError("eligibility parent", f"{name} row-count mismatch")
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
    marker_name = config.parent_marker_filename
    full_domain_state = config.parent_full_domain_state
    if marker_name == "failed.json" and full_domain_state != "evaluable":
        raise Phase4D4ProtocolAError(
            "failed.json",
            "full_domain_core must be evaluable before the failed marker is admissible",
        )
    return MappingProxyType(
        {
            "full_domain_state": full_domain_state,
            "marker_filename": marker_name,
            "parent_path": str(path),
        }
    )


def reconstruct_d4_protocol_a_inputs(
    cohort: D4SugarCohort,
    eligibility_config: Phase4D4EligibilityConfig,
    config: Phase4D4ProtocolAConfig,
    parent_bridge: Mapping[str, object],
) -> D4ProtocolAInputs:
    if config.synthetic_fixture:
        raise Phase4D4ProtocolAError("reconstruct", "real retained inputs require frozen config")
    if len(cohort.record_ids) != 7680 or len(set(cohort.well_ids)) != 240 or len(cohort.blank_record_ids) != 32:
        raise Phase4D4ProtocolAError("reconstruct", "cohort identity mismatch")
    support_axis = np.asarray(eligibility_config.support_coordinates_cm1, dtype="<f8")
    if support_axis.shape != (1999,):
        raise Phase4D4ProtocolAError("reconstruct", "support axis point count mismatch")
    if _array_sha(support_axis) != "db910b11f92151db481391e96b8a06e246140596cfd4abad64764b87d2d84ee5":
        raise Phase4D4ProtocolAError("reconstruct", "support axis float64 hash mismatch")
    if _array_sha(support_axis, "<f4") != "c32275fcb069cf66b3c9b19e1e922d93c724ea4d54e9a9bc8dfaad734ca4c807":
        raise Phase4D4ProtocolAError("reconstruct", "support axis float32 hash mismatch")
    fold_rows = _read_jsonl_rows(_parent_root() / "well_folds.jsonl")
    role_rows = _read_jsonl_rows(_parent_root() / "model_role_occurrences.jsonl")
    if len(fold_rows) != 5 or len(role_rows) != 38560:
        raise Phase4D4ProtocolAError("reconstruct", "parent split ledgers mismatch")
    parent_fold_rows = {int(row["fold_index"]): row for row in fold_rows}
    if set(parent_fold_rows) != set(range(5)):
        raise Phase4D4ProtocolAError("reconstruct", "parent fold indexes mismatch")
    try:
        audit = json.loads((ROOT / PHASE05_AUDIT_RELATIVE_PATH).read_text(encoding="utf-8"))
        audit_folds = {int(row["fold"]): row for row in audit["folds"]}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise Phase4D4ProtocolAError(
            "reconstruct", f"invalid frozen Phase-0.5 fold audit: {error}"
        ) from error
    if set(audit_folds) != set(range(5)):
        raise Phase4D4ProtocolAError("reconstruct", "frozen fold audit indexes mismatch")
    frozen = eligibility_config.frozen_identities
    fold_by_well: dict[str, int] = {}
    for fold_index, indexes in enumerate(cohort.folds):
        index_values = np.asarray(indexes, dtype=np.int64)
        record_ids = tuple(str(cohort.record_ids[int(index)]) for index in index_values)
        source_members = tuple(str(cohort.source_members[int(index)]) for index in index_values)
        observed_wells = {str(cohort.well_ids[int(index)]) for index in index_values}
        well_ids = tuple(str(value) for value in audit_folds[fold_index]["well_ids"])
        if index_values.size != 1536 or len(well_ids) != 48 or len(set(well_ids)) != 48:
            raise Phase4D4ProtocolAError("reconstruct", f"fold {fold_index} test denominator mismatch")
        if set(well_ids) != observed_wells:
            raise Phase4D4ProtocolAError("reconstruct", f"fold {fold_index} audit membership mismatch")
        computed = {
            "record_ids_sha256": _sha_bytes(("\n".join(sorted(record_ids)) + "\n").encode("utf-8")),
            "source_members_sha256": _sha_bytes(("\n".join(source_members) + "\n").encode("utf-8")),
            "well_ids_sha256": _sha_bytes(("\n".join(well_ids) + "\n").encode("utf-8")),
        }
        expected = {
            "record_ids_sha256": str(frozen["fold_record_ids_sha256"][fold_index]),
            "source_members_sha256": str(frozen["fold_source_members_sha256"][fold_index]),
            "well_ids_sha256": str(frozen["fold_well_ids_sha256"][fold_index]),
        }
        parent_row = parent_fold_rows[fold_index]
        if computed != expected or any(str(parent_row.get(key)) != value for key, value in expected.items()):
            raise Phase4D4ProtocolAError("reconstruct", f"fold {fold_index} identity mismatch")
        for well_id in well_ids:
            if well_id in fold_by_well:
                raise Phase4D4ProtocolAError("reconstruct", f"duplicate test-fold well {well_id}")
            fold_by_well[well_id] = fold_index
    if set(fold_by_well) != set(cohort.well_ids):
        raise Phase4D4ProtocolAError("reconstruct", "fold coverage does not match mixture wells")
    if parent_bridge.get("full_domain_state") != "evaluable":
        raise Phase4D4ProtocolAError("reconstruct", "parent bridge state mismatch")
    native_axis = np.asarray(cohort.wavenumber, dtype="<f8")
    if native_axis.shape != (2000,) or not np.all(np.diff(native_axis) > 0.0):
        raise Phase4D4ProtocolAError("reconstruct", "invalid native axis")
    native_spectra = tuple(
        Spectrum1D(
            spectrum_id=f"d4_sugar_low_snr::{record_id}",
            sample_id=str(well_id),
            axis_cm1=native_axis,
            intensity=np.asarray(values, dtype="<f8"),
        )
        for record_id, well_id, values in zip(
            cohort.record_ids, cohort.well_ids, cohort.intensity, strict=True
        )
    )
    native_blank_spectra = tuple(
        Spectrum1D(
            spectrum_id=f"d4_blank::{record_id}",
            sample_id=str(well_id),
            axis_cm1=native_axis,
            intensity=np.asarray(values, dtype="<f8"),
        )
        for record_id, well_id, values in zip(
            cohort.blank_record_ids, cohort.blank_well_ids, cohort.blank_intensity, strict=True
        )
    )
    alpha0_projected = np.ascontiguousarray(
        np.asarray(cohort.intensity, dtype="<f8")[:, 1:], dtype="<f4"
    )
    alpha0_blank_projected = np.ascontiguousarray(
        np.asarray(cohort.blank_intensity, dtype="<f8")[:, 1:], dtype="<f4"
    )
    if not np.array_equal(native_axis[1:], support_axis):
        raise Phase4D4ProtocolAError("reconstruct", "literal support selection mismatch")
    train_by_fold = {int(split.test_fold): np.asarray(split.train_indices, dtype="<i8") for split in cohort.splits}
    validation_by_fold = {
        int(split.test_fold): np.asarray(split.validation_indices, dtype="<i8") for split in cohort.splits
    }
    test_by_fold = {int(split.test_fold): np.asarray(split.test_indices, dtype="<i8") for split in cohort.splits}
    for fold in range(5):
        if (
            train_by_fold[fold].size != 4608
            or validation_by_fold[fold].size != 1536
            or test_by_fold[fold].size != 1536
        ):
            raise Phase4D4ProtocolAError("reconstruct", f"fold {fold} role denominator mismatch")
    expected_roles: set[tuple[int, str, str]] = set()
    for fold in range(5):
        for role, indexes in (
            ("train", train_by_fold[fold]),
            ("validation", validation_by_fold[fold]),
            ("test", test_by_fold[fold]),
        ):
            expected_roles.update((fold, str(cohort.record_ids[int(index)]), role) for index in indexes)
        expected_roles.update((fold, str(record_id), "blank_auxiliary") for record_id in cohort.blank_record_ids)
    actual_roles = {
        (int(row["seed"]), str(row["record_id"]), str(row["role"])) for row in role_rows
    }
    if actual_roles != expected_roles or len(actual_roles) != len(role_rows):
        raise Phase4D4ProtocolAError("reconstruct", "parent model-role ledger mismatch")
    return D4ProtocolAInputs(
        synthetic_fixture=False,
        well_count=240,
        acquisitions_per_well=32,
        model_folds=5,
        feature_count=int(support_axis.size),
        record_ids=tuple(str(value) for value in cohort.record_ids),
        well_ids=tuple(dict.fromkeys(str(value) for value in cohort.well_ids)),
        record_well_ids=tuple(str(value) for value in cohort.well_ids),
        fold_by_well=MappingProxyType(dict(sorted(fold_by_well.items()))),
        rounds=tuple(int(value) for value in np.asarray(cohort.rounds, dtype="<i8").tolist()),
        repetitions=tuple(int(value) for value in np.asarray(cohort.repetitions, dtype="<i8").tolist()),
        targets=np.asarray(cohort.targets, dtype="<f8"),
        target_names=tuple(str(value) for value in cohort.target_names),
        blank_record_ids=tuple(str(value) for value in cohort.blank_record_ids),
        blank_targets=np.asarray(cohort.blank_targets, dtype="<f8"),
        support_axis_cm1=np.ascontiguousarray(support_axis, dtype="<f8"),
        validation_macro_nrmse_by_component=MappingProxyType({2: 0.0, 4: 0.0, 8: 0.0, 16: 0.0, 32: 0.0}),
        prediction_fixture=None,
        blank_fixture=None,
        native_spectra=native_spectra,
        native_blank_spectra=native_blank_spectra,
        alpha0_projected=alpha0_projected,
        alpha0_blank_projected=alpha0_blank_projected,
        train_indices_by_fold=MappingProxyType(train_by_fold),
        validation_indices_by_fold=MappingProxyType(validation_by_fold),
        test_indices_by_fold=MappingProxyType(test_by_fold),
        eligibility_config=eligibility_config,
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
        raise Phase4D4ProtocolAError("science", f"metric output {output_id} did not resolve exactly once")
    value = float(matches[0].value)
    if not math.isfinite(value):
        raise Phase4D4ProtocolAError("science", f"metric output {output_id} is nonfinite")
    return value


def _measurement_for_condition(
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
    scalar_rows: dict[str, Mapping[str, object]] = {}
    for output_id, metric in _metric_objects().items():
        request = SingleSpectrumInput(output) if output_id == "is_like_structure_to_noise" else SpectrumPairInput(source, output)
        result = evaluate_metric(metric, request)
        scalar_rows[output_id] = {
            "diagnostics_digest": _step27_science._receipt_hash(result.diagnostics),
            "output_id": output_id,
            "result_digest": _step27_science._receipt_hash(result.outputs),
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
        scalar_rows[output_id] = {
            "diagnostics_digest": _step27_science._receipt_hash(peak_result.diagnostics),
            "output_id": output_id,
            "result_digest": _step27_science._receipt_hash(peak_result.outputs),
            "state": "complete",
            "value": _metric_value(peak_result, output_id),
        }
    support_sha = _array_sha(np.asarray(projected, dtype="<f4"), "<f4")
    peak_digest = str(output_cwt["peak_list_sha256"])
    if condition_id != "alpha0":
        peak_digest = _step27_science._receipt_hash(
            {
                "cwt_peak_list_sha256": peak_digest,
                "support_intensity_sha256": support_sha,
            }
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
        "metric_values": [scalar_rows[metric_id] for metric_id in METRIC_OUTPUT_IDS],
        "native_axis_sha256": _array_sha(np.asarray(output.axis_cm1, dtype="<f8")),
        "native_intensity_sha256": _array_sha(np.asarray(output.intensity, dtype="<f8")),
        "perturbation_id": _condition_perturbation(condition_id),
        "projected_row_sha256": support_sha,
        "record_id": record_id,
        "record_order": record_order,
        "state": "complete",
        "well_id": well_id,
    }


def _validate_measurements_against_parent(
    measurements: Sequence[Mapping[str, object]], blank_receipts: Sequence[Mapping[str, object]]
) -> None:
    parent = _parent_root()
    expected_records = {
        (str(row["record_id"]), str(row["condition_id"])): row
        for row in _read_jsonl_rows(parent / "record_conditions.jsonl")
    }
    expected_blanks = {
        (str(row["record_id"]), str(row["condition_id"])): row
        for row in _read_jsonl_rows(parent / "blank_conditions.jsonl")
    }
    if len(expected_records) != len(measurements) or len(expected_blanks) != len(blank_receipts):
        raise Phase4D4ProtocolAError("science bridge", "parent condition row-count mismatch")
    for row in measurements:
        key = (str(row["record_id"]), str(row["condition_id"]))
        expected = expected_records.get(key)
        actual_metrics = {str(item["output_id"]): item for item in row["metric_values"]}
        if expected is None or expected.get("state") != row.get("state"):
            raise Phase4D4ProtocolAError("science bridge", f"condition state mismatch {key}")
        for metric_id in METRIC_OUTPUT_IDS:
            parent_metric = expected["metrics"].get(metric_id)
            actual = actual_metrics[metric_id]
            if (
                parent_metric is None
                or parent_metric.get("state") != actual.get("state")
                or parent_metric.get("result_sha256") != actual.get("result_digest")
                or parent_metric.get("diagnostics_sha256") != actual.get("diagnostics_digest")
            ):
                raise Phase4D4ProtocolAError("science bridge", f"metric receipt mismatch {key}/{metric_id}")
        parent_cwt = expected["cwt"]
        actual_cwt = row["cwt"]
        for parent_name, actual_name in (
            ("state", "state"),
            ("diagnostics_sha256", "diagnostics_digest"),
            ("peak_list_sha256", "peak_list_digest"),
            ("warning_sha256", "warning_digest"),
        ):
            if parent_cwt.get(parent_name) != actual_cwt.get(actual_name):
                raise Phase4D4ProtocolAError("science bridge", f"CWT receipt mismatch {key}/{parent_name}")
    for row in blank_receipts:
        key = (str(row["record_id"]), str(row["condition_id"]))
        expected = expected_blanks.get(key)
        if expected is None or any(expected.get(name) != row.get(name) for name in ("state", "diagnostics_sha256", "result_sha256", "warning_sha256")):
            raise Phase4D4ProtocolAError("science bridge", f"blank receipt mismatch {key}")


_SCIENCE_ELIGIBILITY: Phase4D4EligibilityConfig | None = None
_SCIENCE_SWEEP = None
_SCIENCE_PHASE1_CONFIG = None
_SCIENCE_CWT_SYSTEM = None


def _initialize_d4_protocol_a_science_worker(
    eligibility_config_path: str, sweep_path: str, phase1_config_path: str, catalog_path: str
) -> None:
    global _SCIENCE_ELIGIBILITY, _SCIENCE_SWEEP, _SCIENCE_PHASE1_CONFIG, _SCIENCE_CWT_SYSTEM
    _SCIENCE_ELIGIBILITY = load_phase4_d4_eligibility_config(Path(eligibility_config_path))
    _SCIENCE_SWEEP = load_perturbation_sweep_config(Path(sweep_path))
    _SCIENCE_PHASE1_CONFIG = load_phase1_core_config(Path(phase1_config_path))
    catalog = load_classical_catalog(Path(catalog_path))
    matches = [system for system in catalog.systems if system.system_id == _SCIENCE_ELIGIBILITY.cwt_system_id]
    if len(matches) != 1:
        raise Phase4D4ProtocolAError("science worker", "CWT system did not resolve exactly once")
    _SCIENCE_CWT_SYSTEM = matches[0]


def _execute_d4_protocol_a_science_record(
    scope: str, record_order: int, fold: int, spectrum: Spectrum1D
) -> Mapping[str, object]:
    if (
        _SCIENCE_ELIGIBILITY is None
        or _SCIENCE_SWEEP is None
        or _SCIENCE_PHASE1_CONFIG is None
        or _SCIENCE_CWT_SYSTEM is None
    ):
        raise Phase4D4ProtocolAError("science worker", "was not initialized")
    if scope not in {"mixture", "blank"}:
        raise Phase4D4ProtocolAError("science worker", f"invalid scope {scope}")
    blank = scope == "blank"
    record_id = spectrum.spectrum_id.split("::", 1)[1]
    well_id = str(spectrum.sample_id)
    projections: dict[str, np.ndarray] = {}
    measurements: list[Mapping[str, object]] = []
    blank_receipts: list[Mapping[str, object]] = []
    with threadpool_limits(limits=1):
        source_cwt = _step27_science._cwt_receipt_from_spectrum(_SCIENCE_CWT_SYSTEM, spectrum)
        if source_cwt.get("state") != "complete":
            raise Phase4D4ProtocolAError("science", f"alpha-zero CWT failed {record_id}")
        alpha0 = np.ascontiguousarray(project_d4_support(spectrum, _SCIENCE_ELIGIBILITY), dtype="<f4")
        projections["alpha0"] = alpha0
        if blank:
            blank_receipts.append(
                {
                    "condition_id": "alpha0",
                    "diagnostics_sha256": source_cwt["diagnostics_sha256"],
                    "record_id": record_id,
                    "result_sha256": _step27_science._receipt_hash(
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
                _measurement_for_condition(
                    record_order=record_order, record_id=record_id, well_id=well_id, fold=fold,
                    condition_id="alpha0", source=spectrum, output=spectrum, projected=alpha0,
                    source_cwt=source_cwt, output_cwt=source_cwt,
                )
            )
        source = _step27_science._phase1_source_for_spectrum(spectrum, order=record_order)
        for perturbation_id in PERTURBATION_IDS:
            cell = run_perturbation_cell(
                source, perturbation_id, _SCIENCE_PHASE1_CONFIG, _SCIENCE_SWEEP,
                p10_admission=P10MemoryAdmission(_SCIENCE_ELIGIBILITY.p10_memory_budget_bytes)
                if perturbation_id == "p10" else None,
            )
            if _step27_science._classify_cell(cell) != "complete":
                raise Phase4D4ProtocolAError("science", f"operator failed {record_id}/{perturbation_id}")
            by_alpha = {float(item.alpha): item.result.output for item in cell.records}
            for alpha in POSITIVE_ALPHAS:
                condition_id = f"{perturbation_id}:{struct.pack('<d', alpha).hex()}"
                output = by_alpha.get(float(alpha))
                if output is None:
                    raise Phase4D4ProtocolAError("science", f"missing condition {record_id}/{condition_id}")
                projected = np.ascontiguousarray(project_d4_support(output, _SCIENCE_ELIGIBILITY), dtype="<f4")
                projections[condition_id] = projected
                output_cwt = _step27_science._cwt_receipt_from_spectrum(_SCIENCE_CWT_SYSTEM, output)
                if output_cwt.get("state") != "complete":
                    raise Phase4D4ProtocolAError("science", f"CWT failed {record_id}/{condition_id}")
                if blank:
                    blank_receipts.append(
                        {
                            "condition_id": condition_id,
                            "diagnostics_sha256": output_cwt["diagnostics_sha256"],
                            "record_id": record_id,
                            "result_sha256": _step27_science._receipt_hash(
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
                        _measurement_for_condition(
                            record_order=record_order, record_id=record_id, well_id=well_id, fold=fold,
                            condition_id=condition_id, source=spectrum, output=output, projected=projected,
                            source_cwt=source_cwt, output_cwt=output_cwt,
                        )
                    )
    if tuple(projections) != CONDITION_IDS:
        raise Phase4D4ProtocolAError("science worker", f"condition order mismatch {record_id}")
    return {
        "blank_receipts": tuple(blank_receipts),
        "measurements": tuple(measurements),
        "projections": projections,
        "record_order": record_order,
        "scope": scope,
    }


def _execute_initialized_d4_protocol_a_science_record(
    job: tuple[str, int, int, Spectrum1D]
) -> Mapping[str, object]:
    return _execute_d4_protocol_a_science_record(*job)


def rematerialize_d4_protocol_a_science(
    inputs: D4ProtocolAInputs, config: Phase4D4ProtocolAConfig, *, worker_count: int
) -> D4RealScience:
    if inputs.synthetic_fixture or config.synthetic_fixture:
        raise Phase4D4ProtocolAError("science", "real rematerialization requires retained inputs")
    if worker_count < 1 or inputs.eligibility_config is None:
        raise Phase4D4ProtocolAError("science", "invalid worker count or missing eligibility config")
    condition_rows: dict[str, list[np.ndarray]] = {condition_id: [] for condition_id in CONDITION_IDS}
    blank_rows: dict[str, list[np.ndarray]] = {condition_id: [] for condition_id in CONDITION_IDS}
    measurements: list[Mapping[str, object]] = []
    blank_receipts: list[Mapping[str, object]] = []

    jobs = tuple(
        ("mixture", order, int(inputs.fold_by_well[str(spectrum.sample_id)]), spectrum)
        for order, spectrum in enumerate(inputs.native_spectra)
    ) + tuple(
        ("blank", order, -1, spectrum)
        for order, spectrum in enumerate(inputs.native_blank_spectra)
    )
    jobs = tuple(sorted(jobs, key=lambda job: (0 if job[0] == "mixture" else 1, job[1])))
    max_p10_estimate = max(estimate_p10_peak_bytes(spectrum.axis_cm1.size) for _, _, _, spectrum in jobs)
    capacity = inputs.eligibility_config.p10_memory_budget_bytes // max_p10_estimate
    process_count = min(worker_count, len(jobs), capacity)
    if process_count < 1:
        raise Phase4D4ProtocolAError("science", "P10 memory budget admits no condition worker")
    with ProcessPoolExecutor(
        max_workers=process_count,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_initialize_d4_protocol_a_science_worker,
        initargs=(
            str(ROOT / ELIGIBILITY_CONFIG_RELATIVE_PATH),
            str(ROOT / SWEEP_RELATIVE_PATH),
            str(ROOT / PHASE1_CONFIG_RELATIVE_PATH),
            str(ROOT / CATALOG_RELATIVE_PATH),
        ),
    ) as executor:
        # executor.map emits in the explicitly sorted job order even when work
        # completes out of order, avoiding a second multi-gigabyte result copy.
        for expected_job, result in zip(
            jobs, executor.map(_execute_initialized_d4_protocol_a_science_record, jobs), strict=True
        ):
            if (str(result["scope"]), int(result["record_order"])) != (expected_job[0], expected_job[1]):
                raise Phase4D4ProtocolAError("science", "worker result order mismatch")
            target = blank_rows if result["scope"] == "blank" else condition_rows
            for condition_id in CONDITION_IDS:
                target[condition_id].append(result["projections"][condition_id])
            measurements.extend(result["measurements"])
            blank_receipts.extend(result["blank_receipts"])
    matrices = MappingProxyType(
        {key: np.ascontiguousarray(value, dtype="<f4") for key, value in condition_rows.items()}
    )
    blank_matrices = MappingProxyType(
        {key: np.ascontiguousarray(value, dtype="<f4") for key, value in blank_rows.items()}
    )
    if any(value.shape != (len(inputs.record_ids), inputs.feature_count) for value in matrices.values()):
        raise Phase4D4ProtocolAError("science", "mixture condition matrix denominator mismatch")
    if any(value.shape != (len(inputs.blank_record_ids), inputs.feature_count) for value in blank_matrices.values()):
        raise Phase4D4ProtocolAError("science", "blank condition matrix denominator mismatch")
    _validate_measurements_against_parent(measurements, blank_receipts)
    return D4RealScience(matrices, blank_matrices, tuple(measurements))


def fit_d4_protocol_a_models(
    inputs: D4ProtocolAInputs, config: Phase4D4ProtocolAConfig
) -> tuple[D4ProtocolAModel, ...]:
    if not inputs.synthetic_fixture:
        required = (
            inputs.alpha0_projected,
            inputs.train_indices_by_fold,
            inputs.validation_indices_by_fold,
            inputs.test_indices_by_fold,
        )
        if any(value is None for value in required):
            raise Phase4D4ProtocolAError("model selection", "retained matrices or split roles missing")
        matrix = np.asarray(inputs.alpha0_projected, dtype="<f4")
        if matrix.shape != (len(inputs.record_ids), inputs.feature_count) or not np.isfinite(matrix).all():
            raise Phase4D4ProtocolAError("model selection", "invalid alpha-zero matrix")
    rows: list[D4ProtocolAModel] = []
    for fold in range(inputs.model_folds):
        fitted: dict[int, PLSRegression] = {}
        score_rows: list[Mapping[str, object]] = []
        if inputs.synthetic_fixture:
            score_rows.extend(
                {
                    "fold": fold,
                    "macro_normalized_rmse": float(inputs.validation_macro_nrmse_by_component.get(n_components, 1.0)),
                    "n_components": int(n_components),
                }
                for n_components in config.n_components_grid
            )
        else:
            assert inputs.alpha0_projected is not None
            assert inputs.train_indices_by_fold is not None
            assert inputs.validation_indices_by_fold is not None
            train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
            validation = np.asarray(inputs.validation_indices_by_fold[fold], dtype=np.int64)
            x_train = inputs.alpha0_projected[train]
            y_train = inputs.targets[train]
            x_validation = inputs.alpha0_projected[validation]
            y_validation = inputs.targets[validation]
            for n_components in config.n_components_grid:
                if n_components < 1 or n_components > min(x_train.shape[0] - 1, x_train.shape[1]):
                    raise Phase4D4ProtocolAError("model selection", f"invalid component count {n_components}")
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
                    raise Phase4D4ProtocolAError("model selection", f"fold {fold} warning: {error}") from error
                except Exception as error:
                    raise Phase4D4ProtocolAError("model selection", f"fold {fold} fit failed: {error}") from error
                if not np.isfinite(predicted).all():
                    raise Phase4D4ProtocolAError("model selection", f"fold {fold} nonfinite validation prediction")
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
        estimator = fitted.get(int(selected["n_components"]))
        train_predictions = None
        digest_payload: dict[str, object] = {
            "fold": fold, "selected_n_components": selected["n_components"]
        }
        if estimator is not None:
            assert inputs.alpha0_projected is not None and inputs.train_indices_by_fold is not None
            train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
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
            if not all(np.isfinite(np.asarray(value)).all() for value in state_arrays) or not np.isfinite(train_predictions).all():
                raise Phase4D4ProtocolAError("model selection", f"fold {fold} nonfinite fitted state")
            digest_payload["state_sha256"] = [
                _array_sha(np.asarray(value, dtype="<f8")) for value in state_arrays
            ]
            digest_payload["n_iter"] = [int(value) for value in estimator.n_iter_]
        digest = _sha_bytes(_canonical(digest_payload))
        rows.append(
            D4ProtocolAModel(
                fold=fold,
                selected_n_components=int(selected["n_components"]),
                refit_with_validation=False,
                condition_count=len(CONDITION_IDS),
                validation_scores=validation_scores,
                model_state_digest=digest,
                estimator=estimator,
                alpha0_train_predictions=train_predictions,
            )
        )
    return tuple(rows)


def _record_indexes_for_well(inputs: D4ProtocolAInputs, well_id: str) -> tuple[int, ...]:
    return tuple(index for index, candidate in enumerate(inputs.record_well_ids) if candidate == well_id)


def _prediction_error(inputs: D4ProtocolAInputs, condition_id: str, record_index: int) -> float:
    if inputs.prediction_fixture == "hand_derived_regression":
        if condition_id == "alpha0":
            return 0.03872983346207417 if record_index < inputs.acquisitions_per_well else 0.03
        if condition_id == "p08:9a9999999999a93f":
            return 0.06324555320336758 if record_index < inputs.acquisitions_per_well else 0.05
    if condition_id == "alpha0":
        return 0.03
    perturbation_index = PERTURBATION_IDS.index(_condition_perturbation(condition_id))
    alpha = _condition_alpha(condition_id)
    return 0.03 + 0.02 * alpha + 0.004 * perturbation_index


def _regression_metrics(true: np.ndarray, predicted: np.ndarray) -> Mapping[str, object]:
    errors = np.asarray(predicted, dtype="<f8") - np.asarray(true, dtype="<f8")
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    denominator = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    r2 = np.where(denominator > 0.0, 1.0 - np.sum(errors**2, axis=0) / denominator, 1.0)
    row = {
        "macro_mae_mol_l": float(np.mean(mae)),
        "macro_normalized_rmse": float(np.mean(rmse / TARGET_RANGE_MOL_L)),
        "macro_r2": float(np.mean(r2)),
    }
    for analyte_index, target_name in enumerate(TARGET_NAMES):
        row[f"{target_name}_rmse_mol_l"] = float(rmse[analyte_index])
        row[f"{target_name}_mae_mol_l"] = float(mae[analyte_index])
        row[f"{target_name}_r2"] = float(r2[analyte_index])
    return row


def _summary_metrics(
    inputs: D4ProtocolAInputs,
    condition_id: str,
    predictions_by_condition: Mapping[str, np.ndarray],
) -> Mapping[str, object]:
    if inputs.prediction_fixture == "hand_derived_regression":
        if condition_id == "alpha0":
            row = {
                "macro_normalized_rmse": 0.09375,
                "macro_mae_mol_l": 0.03,
                "macro_r2": 0.9,
            }
        elif condition_id == "p08:9a9999999999a93f":
            row = {
                "macro_normalized_rmse": 0.15625,
                "macro_mae_mol_l": 0.05,
                "macro_r2": 0.75,
            }
        else:
            error = _prediction_error(inputs, condition_id, inputs.acquisitions_per_well)
            row = {
                "macro_normalized_rmse": float(error / TARGET_RANGE_MOL_L),
                "macro_mae_mol_l": float(error),
                "macro_r2": max(0.0, 1.0 - error),
            }
        for target_name in TARGET_NAMES:
            row[f"{target_name}_rmse_mol_l"] = float(row["macro_normalized_rmse"]) * TARGET_RANGE_MOL_L
            row[f"{target_name}_mae_mol_l"] = float(row["macro_mae_mol_l"])
            row[f"{target_name}_r2"] = float(row["macro_r2"])
        return row
    return _regression_metrics(inputs.targets, predictions_by_condition[condition_id])


def _blank_prediction_value(
    inputs: D4ProtocolAInputs,
    fold: int,
    condition_id: str,
    blank_index: int,
    analyte_index: int,
) -> float:
    if inputs.blank_fixture == "isolated_auxiliary_failure" and fold == 0 and condition_id == "alpha0" and analyte_index == 0:
        return (-0.02, 0.0, 0.02)[blank_index % 3]
    if inputs.blank_fixture == "isolated_auxiliary_failure" and fold == 1 and condition_id == "p08:9a9999999999a93f" and analyte_index == 0:
        return 0.01 * blank_index
    base = 0.002 * (fold + 1) + 0.001 * analyte_index + 0.0005 * blank_index
    return base + 0.01 * _condition_alpha(condition_id)


def _build_prediction_payloads(
    inputs: D4ProtocolAInputs,
    config: Phase4D4ProtocolAConfig,
    models: Sequence[D4ProtocolAModel],
    science: D4RealScience | None = None,
) -> tuple[
    tuple[Mapping[str, object], ...],
    Mapping[str, np.ndarray],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    predictions: list[Mapping[str, object]] = []
    predictions_by_condition = {condition_id: np.zeros_like(inputs.targets) for condition_id in CONDITION_IDS}
    record_measurements: list[Mapping[str, object]] = []
    blank_predictions: list[Mapping[str, object]] = []
    fold_model = {model.fold: model for model in models}
    if not inputs.synthetic_fixture:
        if science is None or any(model.estimator is None for model in models):
            raise Phase4D4ProtocolAError("prediction", "real science or fitted estimator missing")
        if inputs.test_indices_by_fold is None:
            raise Phase4D4ProtocolAError("prediction", "test split roles missing")
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
                    raise Phase4D4ProtocolAError("prediction", f"fold {model.fold} warning: {error}") from error
                except Exception as error:
                    raise Phase4D4ProtocolAError("prediction", f"fold {model.fold} failed: {error}") from error
                if predicted.shape != (test.size, len(TARGET_NAMES)) or not np.isfinite(predicted).all():
                    raise Phase4D4ProtocolAError("prediction", f"fold {model.fold} invalid test prediction")
                predictions_by_condition[condition_id][test] = predicted
                for index, values in zip(test.tolist(), predicted, strict=True):
                    record_id = inputs.record_ids[index]
                    predictions.append(
                        {
                            "condition_id": condition_id,
                            "config_sha256": config.sha256,
                            "fold": model.fold,
                            "model_state_digest": model.model_state_digest,
                            "predicted_targets": [float(value) for value in values],
                            "projected_row_sha256": _array_sha(
                                science.condition_spectra[condition_id][index], "<f4"
                            ),
                            "record_id": record_id,
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
                if predicted.shape != (len(inputs.blank_record_ids), len(TARGET_NAMES)) or not np.isfinite(predicted).all():
                    raise Phase4D4ProtocolAError("prediction", f"fold {model.fold} invalid blank prediction")
                for blank_record_id, values in zip(inputs.blank_record_ids, predicted, strict=True):
                    blank_predictions.append(
                        {
                            "blank_record_id": blank_record_id,
                            "condition_id": condition_id,
                            "fold": model.fold,
                            "model_state_digest": model.model_state_digest,
                            "predicted_targets": [float(value) for value in values],
                            "terminal_state": "complete",
                        }
                    )
        ordered_predictions = tuple(
            sorted(predictions, key=lambda row: (int(row["record_order"]), CONDITION_IDS.index(str(row["condition_id"]))))
        )
        ordered_measurements = tuple(
            measurements[(record_id, condition_id)]
            for record_id in inputs.record_ids
            for condition_id in CONDITION_IDS
        )
        ordered_blanks = tuple(
            sorted(
                blank_predictions,
                key=lambda row: (int(row["fold"]), inputs.blank_record_ids.index(str(row["blank_record_id"])), CONDITION_IDS.index(str(row["condition_id"]))),
            )
        )
        return ordered_predictions, MappingProxyType(predictions_by_condition), ordered_measurements, ordered_blanks
    for record_index, record_id in enumerate(inputs.record_ids):
        well_id = inputs.record_well_ids[record_index]
        fold = int(inputs.fold_by_well[well_id])
        model = fold_model[fold]
        for condition_id in CONDITION_IDS:
            error = _prediction_error(inputs, condition_id, record_index)
            predicted = inputs.targets[record_index] + error
            predictions_by_condition[condition_id][record_index] = predicted
            metric_vector = {
                "mse": float(np.mean((predicted - inputs.targets[record_index]) ** 2)),
                "rmse": float(np.sqrt(np.mean((predicted - inputs.targets[record_index]) ** 2))),
                "mae": float(np.mean(np.abs(predicted - inputs.targets[record_index]))),
                "sam": float(np.mean(np.abs(predicted - inputs.targets[record_index])) * 0.5),
                "pearson_r": float(max(0.0, 1.0 - error)),
                "nmse": float(np.mean((predicted - inputs.targets[record_index]) ** 2) / (TARGET_RANGE_MOL_L**2)),
                "wasserstein_1_cm1": float(error * 10.0),
                "is_like_structure_to_noise": float(max(0.0, 1.0 - error * 2.0)),
                "precision": float(max(0.0, 1.0 - error)),
                "recall": float(max(0.0, 1.0 - error * 1.1)),
                "f1": float(max(0.0, 1.0 - error * 1.05)),
                "artifact_peak_ratio": float(error * 0.25),
                "missing_peak_ratio": float(error * 0.2),
            }
            predictions.append(
                {
                    "condition_id": condition_id,
                    "config_sha256": config.sha256,
                    "fold": fold,
                    "model_state_digest": model.model_state_digest,
                    "predicted_targets": [float(value) for value in predicted],
                    "projected_row_sha256": _sha_bytes(
                        _canonical({"condition_id": condition_id, "record_id": record_id})
                    ),
                    "record_id": record_id,
                    "record_order": record_index,
                    "repetition": int(inputs.repetitions[record_index]),
                    "round": int(inputs.rounds[record_index]),
                    "terminal_state": "complete",
                    "true_targets": [float(value) for value in inputs.targets[record_index]],
                    "well_id": well_id,
                }
            )
            record_measurements.append(
                {
                    "condition_id": condition_id,
                    "cwt": {
                        "diagnostics_digest": _sha_bytes(_canonical({"record_id": record_id, "state": "complete"})),
                        "peak_count": 3,
                        "peak_list_digest": _sha_bytes(_canonical({"condition_id": condition_id, "record_id": record_id, "type": "peaks"})),
                        "state": "complete",
                        "warning_digest": _sha_bytes(_canonical({"warning": None})),
                    },
                    "fold": fold,
                    "metric_values": [
                        {
                            "diagnostics_digest": _sha_bytes(_canonical({"metric_output_id": metric_id, "record_id": record_id})),
                            "output_id": metric_id,
                            "result_digest": _sha_bytes(_canonical({"metric_output_id": metric_id, "value": metric_vector[metric_id]})),
                            "state": "complete",
                            "value": float(metric_vector[metric_id]),
                        }
                        for metric_id in METRIC_OUTPUT_IDS
                    ],
                    "record_id": record_id,
                    "record_order": record_index,
                    "well_id": well_id,
                }
            )
    for fold in range(inputs.model_folds):
        for blank_index, blank_record_id in enumerate(inputs.blank_record_ids):
            for condition_id in CONDITION_IDS:
                predicted = [
                    float(_blank_prediction_value(inputs, fold, condition_id, blank_index, analyte_index))
                    for analyte_index in range(4)
                ]
                blank_predictions.append(
                    {
                        "blank_record_id": blank_record_id,
                        "condition_id": condition_id,
                        "fold": fold,
                        "predicted_targets": predicted,
                        "terminal_state": "complete",
                    }
                )
    return (
        tuple(predictions),
        MappingProxyType(predictions_by_condition),
        tuple(record_measurements),
        tuple(blank_predictions),
    )


def _well_condition_rows(
    inputs: D4ProtocolAInputs, predictions_by_condition: Mapping[str, np.ndarray]
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for well_id in inputs.well_ids:
        indexes = _record_indexes_for_well(inputs, well_id)
        if len(indexes) != inputs.acquisitions_per_well:
            raise Phase4D4ProtocolAError("well grid", f"{well_id} acquisition denominator mismatch")
        baseline_loss = None
        for condition_id in CONDITION_IDS:
            delta = predictions_by_condition[condition_id][list(indexes)] - inputs.targets[list(indexes)]
            loss = float(np.mean((delta**2) / (TARGET_RANGE_MOL_L**2)))
            if condition_id == "alpha0":
                baseline_loss = loss
                downstream_harm = 0.0
            else:
                downstream_harm = loss - float(baseline_loss)
            rows.append(
                {
                    "condition_id": condition_id,
                    "downstream_harm": downstream_harm,
                    "fold": int(inputs.fold_by_well[well_id]),
                    "loss": loss,
                    "state": "complete",
                    "well_id": well_id,
                }
            )
    return tuple(rows)


def _lod_loq_rows(
    inputs: D4ProtocolAInputs,
    blank_predictions: Sequence[Mapping[str, object]],
    models: Sequence[D4ProtocolAModel] | None = None,
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    train_true = inputs.targets
    for fold in range(inputs.model_folds):
        for condition_id in CONDITION_IDS:
            if inputs.blank_fixture == "isolated_auxiliary_failure" and fold == 0 and condition_id == "alpha0":
                sigma_by_analyte = {0: 0.02}
                slope_by_analyte = {0: 0.4}
            else:
                sigma_by_analyte = {}
                slope_by_analyte = {}
            predicted_rows = [
                row for row in blank_predictions if int(row["fold"]) == fold and str(row["condition_id"]) == condition_id
            ]
            predicted = np.asarray([row["predicted_targets"] for row in predicted_rows], dtype="<f8")
            for analyte_index in range(4):
                sigma = sigma_by_analyte.get(analyte_index, float(np.std(predicted[:, analyte_index], ddof=1)))
                if (
                    inputs.blank_fixture == "isolated_auxiliary_failure"
                    and fold == 1
                    and condition_id == "p08:9a9999999999a93f"
                    and analyte_index == 0
                ):
                    slope = -0.1
                elif not inputs.synthetic_fixture:
                    if models is None or inputs.train_indices_by_fold is None:
                        raise Phase4D4ProtocolAError("technical LOD/LOQ", "model train predictions missing")
                    model = next(item for item in models if item.fold == fold)
                    if model.alpha0_train_predictions is None:
                        raise Phase4D4ProtocolAError("technical LOD/LOQ", f"fold {fold} train predictions missing")
                    train = np.asarray(inputs.train_indices_by_fold[fold], dtype=np.int64)
                    true_values = inputs.targets[train, analyte_index]
                    predicted_values = model.alpha0_train_predictions[:, analyte_index]
                    centered_true = true_values - float(np.mean(true_values))
                    denominator = float(np.sum(centered_true**2))
                    slope = float(
                        np.sum(centered_true * (predicted_values - float(np.mean(predicted_values)))) / denominator
                    ) if denominator > 0.0 else float("nan")
                else:
                    centered_true = train_true[:, analyte_index] - float(np.mean(train_true[:, analyte_index]))
                    denominator = float(np.sum(centered_true**2))
                    if denominator <= 0.0:
                        slope = float("nan")
                    else:
                        synthetic_pred = train_true[:, analyte_index] * 0.4 + 0.01 * fold + 0.005 * _condition_alpha(condition_id)
                        slope = float(
                            np.sum(centered_true * (synthetic_pred - float(np.mean(synthetic_pred)))) / denominator
                        )
                    if analyte_index in slope_by_analyte:
                        slope = slope_by_analyte[analyte_index]
                if not np.isfinite(slope) or slope <= 0.0:
                    rows.append(
                        {
                            "analyte_index": analyte_index,
                            "condition_id": condition_id,
                            "fold": fold,
                            "ich_lod": None,
                            "ich_loq": None,
                            "iupac_lod": None,
                            "sigma": float(sigma) if np.isfinite(sigma) else None,
                            "slope": float(slope) if np.isfinite(slope) else None,
                            "state": "not_evaluable_nonpositive_slope",
                        }
                    )
                else:
                    rows.append(
                        {
                            "analyte_index": analyte_index,
                            "condition_id": condition_id,
                            "fold": fold,
                            "ich_lod": float(3.3 * sigma / slope),
                            "ich_loq": float(10.0 * sigma / slope),
                            "iupac_lod": float(3.0 * sigma / slope),
                            "sigma": float(sigma),
                            "slope": float(slope),
                            "state": "complete",
                        }
                    )
    return tuple(rows)


def _metric_value_for_record(
    record_measurements: Mapping[str, object], metric_output_id: str
) -> float:
    for row in record_measurements["metric_values"]:
        if row["output_id"] == metric_output_id:
            return float(row["value"])
    raise Phase4D4ProtocolAError("metric lookup", f"missing {metric_output_id}")


def _well_observations(
    inputs: D4ProtocolAInputs,
    record_measurements: Sequence[Mapping[str, object]],
    well_conditions: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    measurements = {
        (str(row["record_id"]), str(row["condition_id"])): row for row in record_measurements
    }
    well_losses = {(str(row["well_id"]), str(row["condition_id"])): row for row in well_conditions}
    rows: list[Mapping[str, object]] = []
    for well_id in inputs.well_ids:
        indexes = _record_indexes_for_well(inputs, well_id)
        baseline_measurements = {
            metric_id: [
                _metric_value_for_record(measurements[(inputs.record_ids[index], "alpha0")], metric_id) for index in indexes
            ]
            for metric_id in METRIC_OUTPUT_IDS
        }
        for condition_id in CONDITION_IDS[1:]:
            perturbation_id = _condition_perturbation(condition_id)
            alpha = _condition_alpha(condition_id)
            downstream_harm = float(well_losses[(well_id, condition_id)]["downstream_harm"])
            for metric_id in METRIC_OUTPUT_IDS:
                current = [
                    _metric_value_for_record(measurements[(inputs.record_ids[index], condition_id)], metric_id)
                    for index in indexes
                ]
                baseline = baseline_measurements[metric_id]
                if metric_id in {
                    "mse",
                    "rmse",
                    "mae",
                    "sam",
                    "nmse",
                    "wasserstein_1_cm1",
                    "artifact_peak_ratio",
                    "missing_peak_ratio",
                }:
                    metric_harm = float(np.mean(np.asarray(current) - np.asarray(baseline)))
                else:
                    metric_harm = float(np.mean(np.asarray(baseline) - np.asarray(current)))
                rows.append(
                    {
                        "acquisition_count": len(indexes),
                        "alpha": alpha,
                        "condition_id": condition_id,
                        "downstream_harm": downstream_harm,
                        "metric_harm": metric_harm,
                        "metric_output_id": metric_id,
                        "perturbation_id": perturbation_id,
                        "state": "complete",
                        "well_id": well_id,
                    }
                )
    return tuple(rows)


def aggregate_d4_protocol_a(
    well_observations: Sequence[Mapping[str, object]],
    config: Phase4D4ProtocolAConfig,
    *,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> D4OutcomeProjection:
    bootstrap_n = config.bootstrap_resamples if bootstrap_resamples is None else int(bootstrap_resamples)
    sign_flip_n = config.sign_flip_resamples if sign_flip_resamples is None else int(sign_flip_resamples)
    tables: dict[str, tuple[AlignmentObservation, ...]] = {}
    states: dict[str, str] = {}
    for metric_id in METRIC_OUTPUT_IDS:
        selected = [row for row in well_observations if row["metric_output_id"] == metric_id]
        observations = tuple(
            AlignmentObservation(
                cluster_id=str(row["well_id"]),
                perturbation_id=str(row["perturbation_id"]),
                alpha=float(row["alpha"]),
                metric_harm=float(row["metric_harm"]),
                downstream_harm=float(row["downstream_harm"]),
            )
            for row in selected
        )
        tables[metric_id] = observations
        states[metric_id] = "complete"
    reference = tables["mse"]
    alignment_rows: list[Mapping[str, object]] = []
    bootstrap_rows: list[Mapping[str, object]] = []
    sign_rows: list[Mapping[str, object]] = []
    observed_p_values: dict[str, float] = {}
    observed_contrasts: dict[str, float] = {}
    reference_gap = alignment_gap(reference)
    reference_acc = cross_perturbation_accuracy(reference)
    reference_boot = bulk_paired_cluster_bootstrap(
        reference,
        reference,
        resamples=bootstrap_n,
        confidence_level=config.confidence_level,
        random_seed=config.random_seed,
    )
    alignment_rows.append(
        {
            "acc_cross": reference_acc.accuracy,
            "acc_interval": reference_boot.reference_acc_interval,
            "ag": reference_gap.alignment_gap,
            "ag_interval": reference_boot.reference_ag_interval,
            "ag_raw": reference_gap.raw_alignment_gap,
            "clusters": len(inputs_from_observations(well_observations)),
            "cross_pair_count": reference_acc.pair_count,
            "metric_output_id": "mse",
            "observation_count": len(reference),
            "state": "complete",
        }
    )
    for metric_id in METRIC_OUTPUT_IDS[1:]:
        candidate = tables[metric_id]
        comparison = compare_alignment(reference, candidate)
        boot = bulk_paired_cluster_bootstrap(
            reference,
            candidate,
            resamples=bootstrap_n,
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
                "resamples": bootstrap_n,
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
                "clusters": len(inputs_from_observations(well_observations)),
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
                resamples=sign_flip_n,
                random_seed=config.random_seed,
            )
            hypothesis_id = f"{metric_id}:{statistic}"
            observed_p_values[hypothesis_id] = sign.p_value
            observed_contrasts[hypothesis_id] = contrast
            sign_rows.append(
                {
                    "contrast": contrast,
                    "hypothesis_id": hypothesis_id,
                    "metric_output_id": metric_id,
                    "p_value": sign.p_value,
                    "resamples": sign_flip_n,
                    "state": "complete",
                    "statistic": statistic,
                }
            )
    family_input = {hypothesis_id: observed_p_values.get(hypothesis_id, 1.0) for hypothesis_id in sorted(observed_p_values)}
    adjusted = {row.hypothesis_id: row for row in holm_step_down(family_input, alpha=config.holm_alpha)}
    holm_rows: list[Mapping[str, object]] = []
    for metric_id in METRIC_OUTPUT_IDS[1:]:
        for statistic in ("d_ag", "d_acc"):
            hypothesis_id = f"{metric_id}:{statistic}"
            result = adjusted[hypothesis_id]
            contrast = observed_contrasts[hypothesis_id]
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
    return D4OutcomeProjection(
        tuple(well_observations),
        tuple(alignment_rows),
        tuple(bootstrap_rows),
        tuple(sign_rows),
        tuple(holm_rows),
    )


def inputs_from_observations(well_observations: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    return tuple(sorted({str(row["well_id"]) for row in well_observations}))


def render_d4_protocol_a_figures(
    projection: D4OutcomeProjection,
    config: Phase4D4ProtocolAConfig,
) -> Mapping[str, bytes]:
    figure1_rows: list[Mapping[str, object]] = []
    metric_states = {str(row["metric_output_id"]): str(row["state"]) for row in projection.alignment_results}
    for metric_id in METRIC_OUTPUT_IDS:
        for perturbation_id in PERTURBATION_IDS:
            for alpha in POSITIVE_ALPHAS:
                selected = [
                    row
                    for row in projection.well_observations
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
    figure2_rows = []
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
    figure1_fields = tuple(figure1_rows[0])
    figure2_fields = tuple(figure2_rows[0])
    payloads: dict[str, bytes] = {
        "figure1_d4_protocol_a_full_domain_data.csv": _csv_bytes(figure1_rows, figure1_fields),
        "figure2_d4_protocol_a_full_domain_data.csv": _csv_bytes(figure2_rows, figure2_fields),
        "d4_protocol_a_full_domain_secondary_table.csv": _csv_bytes(figure2_rows, figure2_fields),
    }
    # Render only from the serialized CSV projections that are themselves
    # checksummed artifact payloads.  This keeps table and figure inputs exact.
    figure1_plot_rows = tuple(
        csv.DictReader(io.StringIO(payloads["figure1_d4_protocol_a_full_domain_data.csv"].decode("utf-8")))
    )
    figure2_plot_rows = tuple(
        csv.DictReader(io.StringIO(payloads["figure2_d4_protocol_a_full_domain_data.csv"].decode("utf-8")))
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
            by_alpha = {}
            for row in selected:
                by_alpha.setdefault(float(row["alpha"]), []).append(float(row["mean_downstream_harm"]))
            downstream_axis.plot(
                sorted(by_alpha),
                [float(np.mean(by_alpha[alpha])) for alpha in sorted(by_alpha)],
                marker="o", linewidth=1.5, color=COLOR_MAP[perturbation_id],
            )
        downstream_axis.set_title("normalized squared-loss harm")
        for axis, metric_id in zip(axes.ravel()[1:], METRIC_OUTPUT_IDS, strict=False):
            selected_metric = [row for row in figure1_plot_rows if row["metric_output_id"] == metric_id]
            for perturbation_id in PERTURBATION_IDS:
                selected = [row for row in selected_metric if row["perturbation_id"] == perturbation_id]
                axis.plot(
                    [float(row["mean_metric_harm"]) for row in selected],
                    [float(row["mean_downstream_harm"]) for row in selected],
                    marker="o",
                    linewidth=1.5,
                    color=COLOR_MAP[perturbation_id],
                )
            axis.set_title(metric_id)
        for axis in axes.ravel()[14:]:
            axis.set_axis_off()
        fig.tight_layout()
        png = io.BytesIO()
        svg = io.BytesIO()
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
        png = io.BytesIO()
        svg = io.BytesIO()
        fig.savefig(png, format="png", dpi=300, metadata={"Date": None})
        fig.savefig(svg, format="svg", metadata={"Date": None})
        plt.close(fig)
        payloads["figure2_d4_protocol_a_full_domain.png"] = png.getvalue()
        payloads["figure2_d4_protocol_a_full_domain.svg"] = svg.getvalue()
    return MappingProxyType(payloads)


def build_phase4_d4_protocol_a_from_inputs(
    output_dir: Path,
    *,
    inputs: D4ProtocolAInputs,
    config: Phase4D4ProtocolAConfig,
    worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4D4ProtocolASummary:
    if worker_count < 1:
        raise Phase4D4ProtocolAError("worker_count", "must be positive")
    if not config.synthetic_fixture and (bootstrap_resamples is not None or sign_flip_resamples is not None):
        raise Phase4D4ProtocolAError("inference", "resample overrides are test-only")
    if inputs.synthetic_fixture != config.synthetic_fixture:
        raise Phase4D4ProtocolAError("inputs", "fixture mode must match config")
    output_path = Path(output_dir)
    if output_path.exists():
        raise Phase4D4ProtocolAError(str(output_path), "append-only output target already exists")
    bridge = validate_d4_eligibility_parent(
        Path("synthetic-parent") if config.synthetic_fixture else _parent_root(), config
    )
    science = None if config.synthetic_fixture else rematerialize_d4_protocol_a_science(
        inputs, config, worker_count=worker_count
    )
    models = fit_d4_protocol_a_models(inputs, config)
    prediction_rows, predictions_by_condition, record_measurements, blank_predictions = _build_prediction_payloads(
        inputs, config, models, science
    )
    well_conditions = _well_condition_rows(inputs, predictions_by_condition)
    technical_lod_loq = _lod_loq_rows(inputs, blank_predictions, models)
    well_observations = _well_observations(inputs, record_measurements, well_conditions)
    projection = aggregate_d4_protocol_a(
        well_observations,
        config,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )
    summary_rows = []
    for condition_id in CONDITION_IDS:
        metrics = _summary_metrics(inputs, condition_id, predictions_by_condition)
        summary_rows.append(
            {
                "condition_id": condition_id,
                "count": len(inputs.record_ids),
                **metrics,
            }
        )
    figures = render_d4_protocol_a_figures(projection, config)
    run_identity = _canonical({"record_count": len(inputs.record_ids), "sha256": config.sha256})
    run_id = RUN_PREFIX + _sha_bytes(run_identity)
    terminal_status = "complete"
    model_rows = [
        {
            "condition_count": model.condition_count,
            "fold": model.fold,
            "model_state_digest": model.model_state_digest,
            "refit_with_validation": model.refit_with_validation,
            "selected_n_components": model.selected_n_components,
        }
        for model in models
    ]
    validation_rows = [row for model in models for row in model.validation_scores]
    if not config.synthetic_fixture:
        expected_counts = {
            "models": 5,
            "validation": 25,
            "measurements": 314880,
            "predictions": 314880,
            "blanks": 6560,
            "wells": 9840,
            "lod": 820,
            "observations": 124800,
        }
        observed_counts = {
            "models": len(models),
            "validation": len(validation_rows),
            "measurements": len(record_measurements),
            "predictions": len(prediction_rows),
            "blanks": len(blank_predictions),
            "wells": len(well_conditions),
            "lod": len(technical_lod_loq),
            "observations": len(well_observations),
        }
        if observed_counts != expected_counts:
            raise Phase4D4ProtocolAError("artifact rows", f"frozen denominator mismatch {observed_counts}")
    preflight = {
        "alignment_state": "complete",
        "bootstrap_resamples": config.bootstrap_resamples if bootstrap_resamples is None else int(bootstrap_resamples),
        "claim_boundary": "preflight_complete",
        "parent_bridge_state": bridge["full_domain_state"],
        "sign_flip_resamples": config.sign_flip_resamples if sign_flip_resamples is None else int(sign_flip_resamples),
        "synthetic_fixture": config.synthetic_fixture,
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
            "technical_lod_loq": len(technical_lod_loq),
            "validation_scores": len(validation_rows),
            "well_conditions": len(well_conditions),
            "well_observations": len(well_observations),
        },
        "environment": dict(_environment()),
        "experiment_id": EXPERIMENT_ID,
        "metric_states": {row["metric_output_id"]: row["state"] for row in projection.alignment_results},
        "protocol": "A",
        "run_id": run_id,
        "status": terminal_status,
        "synthetic_fixture": config.synthetic_fixture,
        "tier": "full_domain_core",
    }
    payloads = {
        "config.json": config.raw_bytes,
        "eligibility_bridge.json": _canonical(dict(bridge)),
        "preflight.json": _canonical(preflight),
        "model_cells.jsonl": _jsonl_bytes(model_rows),
        "validation_scores.jsonl": _jsonl_bytes(validation_rows),
        "record_measurements.jsonl": _jsonl_bytes(record_measurements),
        "predictions.jsonl": _jsonl_bytes(prediction_rows),
        "blank_predictions.jsonl": _jsonl_bytes(blank_predictions),
        "well_conditions.jsonl": _jsonl_bytes(well_conditions),
        "technical_lod_loq.jsonl": _jsonl_bytes(technical_lod_loq),
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
    if tuple(payloads) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D4ProtocolAError("artifact payloads", "canonical order mismatch")
    output_path.mkdir(parents=True)
    for name, value in payloads.items():
        (output_path / name).write_bytes(value)
    marker_name = "complete.json"
    (output_path / marker_name).write_bytes(_canonical({"run_id": run_id, "status": terminal_status}))
    sums = b"".join(
        f"{_sha_bytes((output_path / name).read_bytes())}  {name}\n".encode("utf-8")
        for name in (*ARTIFACT_PAYLOAD_FILES, marker_name)
    )
    (output_path / "SHA256SUMS").write_bytes(sums)
    return Phase4D4ProtocolASummary(
        path=output_path,
        run_id=run_id,
        status=terminal_status,
        record_count=len(inputs.record_ids),
        prediction_row_count=len(prediction_rows),
    )


def _rebuild_candidate(
    path: Path,
    *,
    inputs: D4ProtocolAInputs,
    config: Phase4D4ProtocolAConfig,
    worker_count: int,
    bootstrap_resamples: int | None,
    sign_flip_resamples: int | None,
) -> tuple[Phase4D4ProtocolASummary, Path]:
    sandbox = Path(tempfile.mkdtemp(prefix="phase4-d4-protocol-a-"))
    rebuilt_path = sandbox / "rebuild"
    summary = build_phase4_d4_protocol_a_from_inputs(
        rebuilt_path,
        inputs=inputs,
        config=config,
        worker_count=worker_count,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )
    return summary, rebuilt_path


def verify_phase4_d4_protocol_a_from_inputs(
    path: Path,
    *,
    inputs: D4ProtocolAInputs,
    config: Phase4D4ProtocolAConfig,
    worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4D4ProtocolASummary:
    candidate = Path(path)
    rebuilt_summary, rebuilt_path = _rebuild_candidate(
        candidate,
        inputs=inputs,
        config=config,
        worker_count=worker_count,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )
    candidate_files = {item.name for item in candidate.iterdir()}
    rebuilt_files = {item.name for item in rebuilt_path.iterdir()}
    if candidate_files != rebuilt_files:
        raise Phase4D4ProtocolAError("artifact inventory", "candidate and rebuilt file sets differ")
    for name in sorted(candidate_files):
        if (candidate / name).read_bytes() != (rebuilt_path / name).read_bytes():
            raise Phase4D4ProtocolAError(name, "candidate and rebuilt bytes differ")
    return Phase4D4ProtocolASummary(
        path=candidate,
        run_id=rebuilt_summary.run_id,
        status=rebuilt_summary.status,
        record_count=rebuilt_summary.record_count,
        prediction_row_count=rebuilt_summary.prediction_row_count,
    )


def verify_phase4_d4_protocol_a(path: Path, worker_count: int = 12) -> Phase4D4ProtocolASummary:
    verifier = importlib.import_module("rpe.runner.phase4_d4_protocol_a_verifier")
    return verifier.verify_phase4_d4_protocol_a(path=path, worker_count=worker_count)


def _verify_phase4_d4_protocol_a_local(path: Path, worker_count: int = 12) -> Phase4D4ProtocolASummary:
    candidate = Path(path)
    marker = "complete.json" if (candidate / "complete.json").exists() else "failed.json"
    endpoint = json.loads((candidate / marker).read_text(encoding="utf-8"))
    predictions = (candidate / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return Phase4D4ProtocolASummary(
        path=candidate,
        run_id=str(endpoint["run_id"]),
        status=str(endpoint["status"]),
        record_count=len({json.loads(line)["record_id"] for line in predictions if line}),
        prediction_row_count=len(predictions),
    )


def build_phase4_d4_protocol_a(output_root: Path, *, worker_count: int = 16) -> Phase4D4ProtocolASummary:
    config = load_phase4_d4_protocol_a_config(ROOT / CONFIG_RELATIVE_PATH)
    parent = _parent_root()
    bridge = validate_d4_eligibility_parent(parent, config)
    cohort = load_d4_sugar_cohort(ROOT / PROTOCOL_CONFIG_RELATIVE_PATH, ROOT / ARCHIVE_RELATIVE_PATH)
    eligibility_config = load_phase4_d4_eligibility_config(ROOT / ELIGIBILITY_CONFIG_RELATIVE_PATH)
    inputs = reconstruct_d4_protocol_a_inputs(cohort, eligibility_config, config, bridge)
    target = Path(output_root) / (
        RUN_PREFIX
        + _sha_bytes(
            _canonical({"record_count": len(inputs.record_ids), "sha256": config.sha256})
        )
    )
    return build_phase4_d4_protocol_a_from_inputs(target, inputs=inputs, config=config, worker_count=worker_count)


__all__ = [
    "ARTIFACT_PAYLOAD_FILES",
    "D4ProtocolAInputs",
    "D4ProtocolAModel",
    "D4OutcomeProjection",
    "Phase4D4ProtocolAConfig",
    "Phase4D4ProtocolAError",
    "Phase4D4ProtocolASummary",
    "aggregate_d4_protocol_a",
    "build_phase4_d4_protocol_a",
    "build_phase4_d4_protocol_a_from_inputs",
    "fit_d4_protocol_a_models",
    "load_phase4_d4_protocol_a_config",
    "make_synthetic_d4_protocol_a_config",
    "make_synthetic_d4_protocol_a_inputs",
    "parse_phase4_d4_protocol_a_config",
    "reconstruct_d4_protocol_a_inputs",
    "rematerialize_d4_protocol_a_science",
    "render_d4_protocol_a_figures",
    "validate_d4_eligibility_parent",
    "verify_phase4_d4_protocol_a",
    "verify_phase4_d4_protocol_a_from_inputs",
]
