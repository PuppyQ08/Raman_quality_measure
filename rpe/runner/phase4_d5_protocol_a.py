from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import platform
import struct
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import h5py
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rpe-matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import threadpoolctl
from threadpoolctl import threadpool_limits

from rpe.alignment import (
    AlignmentObservation,
    AlignmentValidationError,
    alignment_gap,
    bulk_paired_cluster_bootstrap,
    compare_alignment,
    cross_perturbation_accuracy,
    holm_step_down,
    orient_harm,
    paired_contribution_sign_flip,
)
from rpe.downstream.rruff import D5LibraryQuerySplit, D5RawCohort, load_d5_native_spectra, load_d5_raw_cohort
from rpe.downstream.rruff_matching import D5MatchingResult, match_d5_protocol_a_values
from rpe.evaluation import (
    PeakPairInput,
    PreferredDirection,
    SingleSpectrumInput,
    Spectrum1D,
    SpectrumPairInput,
    evaluate_metric,
)
from rpe.methods.catalog import Phase3ClassicalCatalog, Phase3System, load_classical_catalog
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
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus, Phase1Cell
from rpe.runner.phase4_d5_protocol_a_authority import CONFIG_BYTES, CONFIG_SHA256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d5_protocol_a_full_domain_v1.json"
D5_CONFIG_RELATIVE_PATH = "experiments/phase05/configs/d5_rruff_protocol.json"
DATASET_RELATIVE_PATH = "data/unified/rruff_raman_raw"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
CATALOG_RELATIVE_PATH = "experiments/phase3/configs/classical_system_catalog_v1.json"
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d5_protocol_a_authority.py"

SCHEMA_VERSION = "phase4-d5-protocol-a-full-domain-config-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d5-protocol-a-full-domain-artifact-v1"
EXPERIMENT_ID = "phase4-d5-protocol-a-full-domain-v1"
RUN_PREFIX = "phase4-d5-protocol-a-full-domain-"
PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
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
CANDIDATE_OUTPUT_IDS = METRIC_OUTPUT_IDS[1:]
STRUCTURE_OUTPUT_IDS = METRIC_OUTPUT_IDS[8:]
METRIC_DIRECTIONS = MappingProxyType(
    {
        "mse": PreferredDirection.LOWER_IS_BETTER,
        "rmse": PreferredDirection.LOWER_IS_BETTER,
        "mae": PreferredDirection.LOWER_IS_BETTER,
        "sam": PreferredDirection.LOWER_IS_BETTER,
        "pearson_r": PreferredDirection.HIGHER_IS_BETTER,
        "nmse": PreferredDirection.LOWER_IS_BETTER,
        "wasserstein_1_cm1": PreferredDirection.LOWER_IS_BETTER,
        "is_like_structure_to_noise": PreferredDirection.HIGHER_IS_BETTER,
        "precision": PreferredDirection.HIGHER_IS_BETTER,
        "recall": PreferredDirection.HIGHER_IS_BETTER,
        "f1": PreferredDirection.HIGHER_IS_BETTER,
        "artifact_peak_ratio": PreferredDirection.LOWER_IS_BETTER,
        "missing_peak_ratio": PreferredDirection.LOWER_IS_BETTER,
    }
)
CONDITION_IDS = ("alpha0",) + tuple(
    f"{perturbation_id}:{struct.pack('<d', alpha).hex()}"
    for perturbation_id in PERTURBATION_IDS
    for alpha in POSITIVE_ALPHAS
)
COLORS = MappingProxyType(
    dict(zip(PERTURBATION_IDS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True))
)
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "preflight.json",
    "operator_cells.jsonl",
    "record_conditions.jsonl",
    "metric_values.jsonl",
    "peak_receipts.jsonl",
    "downstream_rows.jsonl",
    "matcher_predictions.jsonl",
    "condition_summary.csv",
    "class_observations.jsonl",
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "figure1_d5_protocol_a_full_domain.png",
    "figure1_d5_protocol_a_full_domain.svg",
    "figure1_d5_protocol_a_full_domain_data.csv",
    "figure2_d5_protocol_a_full_domain.png",
    "figure2_d5_protocol_a_full_domain.svg",
    "figure2_d5_protocol_a_full_domain_data.csv",
    "d5_full_domain_secondary_table.csv",
    "manifest.json",
    "complete.json",
)
CODE_RELATIVE_PATHS = (
    "rpe/alignment/bulk.py",
    "rpe/alignment/contracts.py",
    "rpe/alignment/core.py",
    "rpe/alignment/inference.py",
    "rpe/downstream/rruff.py",
    "rpe/downstream/rruff_matching.py",
    "rpe/evaluation/contracts.py",
    "rpe/methods/catalog.py",
    "rpe/methods/classical/peaks.py",
    "rpe/metrics/fidelity.py",
    "rpe/metrics/peak.py",
    "rpe/metrics/reference_free.py",
    "rpe/metrics/transport.py",
    "rpe/perturb/axis_transform.py",
    "rpe/perturb/baseline_distortion.py",
    "rpe/perturb/contracts.py",
    "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py",
    "rpe/perturb/sweep.py",
    "rpe/runner/phase1_gates.py",
    "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py",
    "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d5_protocol_a.py",
    "rpe/runner/phase4_d5_protocol_a_verifier.py",
    "tools/run_phase4_d5_protocol_a.py",
)


class Phase4D5ProtocolAError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D5ProtocolAConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_output_ids: tuple[str, ...]
    metric_directions: Mapping[str, PreferredDirection]
    cwt_system_id: str
    cohort_record_count: int
    query_record_count: int
    query_occurrence_count: int
    class_count: int
    split_count: int
    expected_operator_cell_count: int
    expected_apply_check_count: int
    expected_query_condition_count: int
    expected_projected_row_count: int
    expected_matcher_call_count: int
    expected_prediction_row_count: int
    expected_class_observation_count_per_metric: int
    expected_record_metric_row_count: int
    expected_figure1_row_count: int
    expected_figure2_row_count: int
    expected_table_row_count: int
    expected_holm_slot_count: int
    support_start_cm1: float
    support_stop_cm1: float
    support_step_cm1: float
    support_point_count: int
    support_max_gap_cm1: float
    bootstrap_resamples: int
    sign_flip_resamples: int
    random_seed: int
    confidence_level: float
    holm_alpha: float
    claim_boundary: str
    artifact_payload_files: tuple[str, ...]
    authorities: Mapping[str, str]
    frozen_identities: Mapping[str, object]
    code_authority: Mapping[str, Mapping[str, object]]
    environment_authority: Mapping[str, object]
    figure_contract: Mapping[str, object]


@dataclass(frozen=True)
class Phase4D5ProtocolASummary:
    path: Path
    run_id: str
    status: str
    query_record_count: int
    prediction_row_count: int
    metric_row_count: int
    class_observation_count: int


def _json_ready(value: object) -> object:
    if is_dataclass(value):
        return _json_ready(asdict(value))
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
    raise Phase4D5ProtocolAError("json", f"unsupported value {type(value).__name__}")


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
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha_bytes(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D5ProtocolAError(path, "must be an object")
    return value


def _strings(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise Phase4D5ProtocolAError(path, "must be an array")
    output = tuple(str(item) for item in value)
    if any(not item for item in output) or len(set(output)) != len(output):
        raise Phase4D5ProtocolAError(path, "must contain unique nonempty strings")
    return output


def _ints(path: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0):
        raise Phase4D5ProtocolAError(path, "must be an integer in range")
    return value


def _number(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise Phase4D5ProtocolAError(path, "must be finite numeric")
    return float(value)


def _lower_hex(path: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise Phase4D5ProtocolAError(path, "must be lowercase SHA-256")
    return value


def _environment() -> dict[str, object]:
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


def _code_document() -> dict[str, object]:
    return {
        relative: {"bytes": (ROOT / relative).stat().st_size, "sha256": _sha_file(ROOT / relative)}
        for relative in CODE_RELATIVE_PATHS
    }


def _authority_document() -> dict[str, object]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    return {"bytes": path.stat().st_size, "sha256": _sha_file(path)}


def parse_phase4_d5_protocol_a_config(
    path: Path, raw: bytes, *, require_frozen_identity: bool
) -> Phase4D5ProtocolAConfig:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5ProtocolAError("config", str(error)) from error
    root = _object("config", document)
    if raw != _canonical(root):
        raise Phase4D5ProtocolAError("config", "must use canonical JSON")
    if require_frozen_identity and (len(raw) != CONFIG_BYTES or _sha_bytes(raw) != CONFIG_SHA256):
        raise Phase4D5ProtocolAError("frozen config identity", "bytes or SHA-256 mismatch")
    if root.get("schema_version") != SCHEMA_VERSION or root.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D5ProtocolAError("config identity", "schema or experiment mismatch")
    synthetic = bool(root.get("synthetic_fixture", False))
    protocol = root.get("protocol", "A" if synthetic else None)
    if protocol != "A":
        raise Phase4D5ProtocolAError("protocol", "must equal A")
    perturbations = _strings("active_perturbation_ids", root.get("active_perturbation_ids"))
    if perturbations != PERTURBATION_IDS:
        raise Phase4D5ProtocolAError("active_perturbation_ids", "frozen order mismatch")
    alpha_grid = tuple(float(value) for value in root.get("alpha_grid", ()))
    if alpha_grid != ALPHA_GRID:
        raise Phase4D5ProtocolAError("alpha_grid", "frozen values mismatch")
    manifest = tuple(_object(f"metric_manifest[{index}]", row) for index, row in enumerate(root.get("metric_manifest", ())))
    metric_ids = tuple(str(row.get("output_id")) for row in manifest)
    if metric_ids != METRIC_OUTPUT_IDS:
        raise Phase4D5ProtocolAError("metric_manifest", "frozen output order mismatch")
    directions: dict[str, PreferredDirection] = {}
    for row in manifest:
        output_id = str(row["output_id"])
        try:
            direction = PreferredDirection(str(row["preferred_direction"]))
        except ValueError as error:
            raise Phase4D5ProtocolAError("metric direction", "invalid") from error
        if direction is not METRIC_DIRECTIONS[output_id]:
            raise Phase4D5ProtocolAError("metric direction", f"mismatch for {output_id}")
        directions[output_id] = direction
    den = _object("denominators", root.get("denominators"))
    cohort_count = _ints("cohort_record_count", den.get("cohort_record_count"), positive=True)
    query_count = _ints("query_record_count", den.get("query_record_count"), positive=True)
    occurrence_count = _ints("query_occurrence_count", den.get("query_occurrence_count"), positive=True)
    class_count = _ints("class_count", den.get("class_count"), positive=True)
    split_count = _ints("split_count", den.get("split_count"), positive=True)
    expected = _object("expected", root.get("expected"))
    expected_values = {
        key: _ints(f"expected.{key}", expected.get(key), positive=True)
        for key in (
            "operator_cell_count",
            "apply_check_count",
            "canonical_query_condition_count",
            "projected_unique_row_count",
            "matcher_call_count",
            "prediction_row_count",
            "class_observation_count_per_metric",
            "record_metric_row_count",
            "figure1_row_count",
            "figure2_row_count",
            "secondary_table_row_count",
            "holm_slot_count",
        )
    }
    derived = {
        "operator_cell_count": query_count * 5,
        "apply_check_count": query_count * 5 * 9,
        "canonical_query_condition_count": query_count * 41,
        "projected_unique_row_count": cohort_count + query_count * 40,
        "matcher_call_count": split_count * 41,
        "prediction_row_count": occurrence_count * 41,
        "class_observation_count_per_metric": class_count * 40,
        "record_metric_row_count": query_count * 41 * 13,
        "figure1_row_count": 520,
        "figure2_row_count": 13,
        "secondary_table_row_count": 13,
        "holm_slot_count": 24,
    }
    if expected_values != derived:
        raise Phase4D5ProtocolAError("expected", "does not match denominators")
    support = _object("support_grid", root.get("support_grid"))
    support_tuple = (
        _number("support.start", support.get("start_cm1")),
        _number("support.stop", support.get("stop_cm1")),
        _number("support.step", support.get("step_cm1")),
        _ints("support.count", support.get("point_count"), positive=True),
        _number("support.gap", support.get("max_in_range_native_gap_cm1")),
    )
    if support_tuple != (204.0, 1800.0, 2.0, 799, 3.0):
        raise Phase4D5ProtocolAError("support_grid", "frozen values mismatch")
    inference = _object("inference", root.get("inference"))
    bootstrap = _ints("bootstrap_resamples", inference.get("bootstrap_resamples"), positive=True)
    sign_flip = _ints("sign_flip_resamples", inference.get("sign_flip_resamples"), positive=True)
    seed = _ints("random_seed", inference.get("random_seed"))
    confidence = _number("confidence_level", inference.get("confidence_level"))
    holm_alpha = _number("holm_alpha", inference.get("holm_alpha"))
    if not synthetic and (bootstrap, sign_flip, seed, confidence, holm_alpha) != (2000, 100000, 20260817, 0.95, 0.05):
        raise Phase4D5ProtocolAError("inference", "frozen values mismatch")
    cwt_id = str(root.get("cwt_system_id"))
    if cwt_id != CWT_SYSTEM_ID:
        raise Phase4D5ProtocolAError("cwt_system_id", "frozen identity mismatch")
    payload_files = _strings("artifact_payload_files", root.get("artifact_payload_files"))
    if payload_files != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D5ProtocolAError("artifact_payload_files", "frozen order mismatch")
    authorities = {
        str(key): _lower_hex(f"authorities.{key}", value)
        for key, value in _object("authorities", root.get("authorities", {})).items()
    }
    frozen_ids = dict(_object("frozen_identities", root.get("frozen_identities", {})))
    code = {
        str(relative): MappingProxyType(dict(_object(f"code.{relative}", value)))
        for relative, value in _object("code_authority", root.get("code_authority", {})).items()
    }
    environment = dict(_object("environment_authority", root.get("environment_authority", {})))
    claim = str(root.get("claim_boundary"))
    if claim != "local_execution_artifact_redistribution_not_cleared":
        raise Phase4D5ProtocolAError("claim_boundary", "mismatch")
    config = Phase4D5ProtocolAConfig(
        path=Path(path), raw_bytes=raw, sha256=_sha_bytes(raw), document=_freeze(root), synthetic_fixture=synthetic,
        perturbation_ids=perturbations, alpha_grid=alpha_grid, metric_output_ids=metric_ids,
        metric_directions=MappingProxyType(directions), cwt_system_id=cwt_id,
        cohort_record_count=cohort_count, query_record_count=query_count, query_occurrence_count=occurrence_count,
        class_count=class_count, split_count=split_count,
        expected_operator_cell_count=expected_values["operator_cell_count"],
        expected_apply_check_count=expected_values["apply_check_count"],
        expected_query_condition_count=expected_values["canonical_query_condition_count"],
        expected_projected_row_count=expected_values["projected_unique_row_count"],
        expected_matcher_call_count=expected_values["matcher_call_count"],
        expected_prediction_row_count=expected_values["prediction_row_count"],
        expected_class_observation_count_per_metric=expected_values["class_observation_count_per_metric"],
        expected_record_metric_row_count=expected_values["record_metric_row_count"],
        expected_figure1_row_count=expected_values["figure1_row_count"],
        expected_figure2_row_count=expected_values["figure2_row_count"],
        expected_table_row_count=expected_values["secondary_table_row_count"],
        expected_holm_slot_count=expected_values["holm_slot_count"],
        support_start_cm1=support_tuple[0], support_stop_cm1=support_tuple[1], support_step_cm1=support_tuple[2],
        support_point_count=support_tuple[3], support_max_gap_cm1=support_tuple[4],
        bootstrap_resamples=bootstrap, sign_flip_resamples=sign_flip, random_seed=seed,
        confidence_level=confidence, holm_alpha=holm_alpha, claim_boundary=claim,
        artifact_payload_files=payload_files, authorities=MappingProxyType(authorities), frozen_identities=_freeze(frozen_ids),
        code_authority=MappingProxyType(code), environment_authority=_freeze(environment),
        figure_contract=_freeze(_object("figure_contract", root.get("figure_contract"))),
    )
    if require_frozen_identity:
        frozen_authorities = {
            "catalog_sha256": "8ad40b08df78b8905d75a67c84a2bb328531ef17cb12704ffad04f0a8f925d8f",
            "d5_config_sha256": "94f59739a2e3583c6ae33ab5f4ab489cb1f26d3c9f8cdeb689a2448c9c01e009",
            "dataset_sha256sums_sha256": "f0cb09af8d80cdf3d52af9ed01bc1ae48abba94fefdff712c30d6e8c8bfe4995",
            "parent_plan_sha256": "a299b5d7d08c4893523233146e9c66750937de9440f9eba0dd8141a64fc706d5",
            "peak_canary_config_sha256": "2a6636de43c87ae95cd77d2ab2cc42470f188f60e8b66c3ed654a76402b72d7d",
            "peak_report_sha256": "75616c89017771a5c57ee1c5a3ffaddc142c1209a0b6c8041f7dde29bf1cc484",
            "peak_wrapper_sha256": "271c25e3f18f80840f3e7ae5fde6d86214b41efc5aa6056b9c6ed34b83a6c003",
            "phase1_core_config_sha256": "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd",
            "phase4_step1_sha256": "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f",
            "phase4_step2_sha256": "4b9c784476d5add2f8551b928fc1d47055945c9584a47b2478996529a0c6dfe6",
            "phase4_step3_sha256": "76b1ffdc3544673cebbd6520803ae72aeff610a6a66a2306dd317c9402fdcbbe",
            "phase4_step4_sha256": "2ae40b991d79935f173b16cf59ad8d7f531ad6a0f5717373712da96a802279e5",
            "phase4_step5_sha256": "2f1d30966e10dcce33caca86651b80747ddcc485109205ad7c1c7785ad8f4341",
            "phase4_step5_sha256sums_sha256": "3a89c44e0667718c77116d9127635407094340f5016ba2254152dccbbff2c719",
            "phase4_step6_sha256": "571e22d1104c37536d8ce033a22756c2e2868ae82d021139ae7e75756f8a0100",
            "sweep_sha256": "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186",
        }
        if synthetic or authorities != frozen_authorities:
            raise Phase4D5ProtocolAError("authorities", "frozen scientific authorities mismatch")
        if (cohort_count, query_count, occurrence_count, class_count, split_count) != (3770, 3012, 6621, 681, 5):
            raise Phase4D5ProtocolAError("denominators", "frozen D5 values mismatch")
        observed_code = _code_document()
        if tuple(code) != CODE_RELATIVE_PATHS or observed_code != {k: dict(v) for k, v in code.items()}:
            raise Phase4D5ProtocolAError("code authority", "mismatch")
        if _environment() != dict(environment):
            raise Phase4D5ProtocolAError("environment authority", "mismatch")
    return config


def load_phase4_d5_protocol_a_config(path: Path) -> Phase4D5ProtocolAConfig:
    return parse_phase4_d5_protocol_a_config(Path(path), Path(path).read_bytes(), require_frozen_identity=True)


def _grid(config: Phase4D5ProtocolAConfig) -> np.ndarray:
    values = np.arange(config.support_start_cm1, config.support_stop_cm1 + config.support_step_cm1 / 2, config.support_step_cm1, dtype="<f8")
    if values.size != config.support_point_count:
        raise Phase4D5ProtocolAError("support grid", "point count mismatch")
    return values


def _project(spectrum: Spectrum1D, grid: np.ndarray, max_gap: float) -> np.ndarray:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5ProtocolAError("downstream support", "extrapolation required")
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    support = axis[left : right + 1]
    if left < 0 or right >= axis.size or support.size < 2 or float(np.max(np.diff(support))) > max_gap:
        raise Phase4D5ProtocolAError("downstream support", "invalid bounds or gap")
    output = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    if not np.isfinite(output).all() or float(np.linalg.norm(output.astype(np.float64))) <= 0:
        raise Phase4D5ProtocolAError("downstream projection", "nonfinite or zero norm")
    return output


def match_d5_protocol_a_799(
    cohort: D5RawCohort,
    split: D5LibraryQuerySplit,
    *,
    condition_id: str,
    query_record_ids: tuple[str, ...],
    library_record_ids: tuple[str, ...],
    query_values: np.ndarray,
    library_values: np.ndarray,
) -> D5MatchingResult:
    query = np.asarray(query_values)
    library = np.asarray(library_values)
    if query.ndim != 2 or library.ndim != 2 or query.shape[1] != 799 or library.shape[1] != 799:
        raise Phase4D5ProtocolAError("799 matcher", "must use exactly 799 features")
    return match_d5_protocol_a_values(
        cohort, split, condition_id=condition_id, query_record_ids=query_record_ids,
        library_record_ids=library_record_ids, query_values=query, library_values=library,
    )


def metric_preflight_states(counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, object]]:
    if tuple(counts) != METRIC_OUTPUT_IDS:
        raise Phase4D5ProtocolAError("metric preflight", "must follow exact manifest")
    states: dict[str, dict[str, object]] = {}
    for output_id in METRIC_OUTPUT_IDS:
        value = counts[output_id]
        complete = int(value["complete"])
        planned = int(value["planned"])
        state = "complete" if complete == planned else "not_evaluable_incomplete_grid"
        states[output_id] = {"complete": complete, "planned": planned, "state": state}
    if states["mse"]["state"] != "complete":
        raise Phase4D5ProtocolAError("MSE comparator", "must be complete on the full grid")
    return states


def fixed_holm_family(
    observed_p_values: Mapping[str, float],
    metric_states: Mapping[str, Mapping[str, object]],
    observed_contrasts: Mapping[str, float],
) -> list[dict[str, object]]:
    multiplicity: dict[str, float] = {}
    state_by_hypothesis: dict[str, str] = {}
    for metric_id in CANDIDATE_OUTPUT_IDS:
        eligible = metric_states[metric_id]["state"] == "complete"
        for statistic in ("d_ag", "d_acc"):
            hypothesis = f"{metric_id}:{statistic}"
            if eligible and hypothesis in observed_p_values:
                multiplicity[hypothesis] = float(observed_p_values[hypothesis])
                state_by_hypothesis[hypothesis] = "tested"
            else:
                multiplicity[hypothesis] = 1.0
                state_by_hypothesis[hypothesis] = "not_tested_metric_incomplete"
    adjusted = {row.hypothesis_id: row for row in holm_step_down(multiplicity, alpha=0.05)}
    output = []
    for hypothesis in sorted(multiplicity):
        tested = state_by_hypothesis[hypothesis] == "tested"
        if tested and hypothesis not in observed_contrasts:
            raise Phase4D5ProtocolAError("Holm family", f"missing observed contrast for {hypothesis}")
        contrast = None if not tested else float(observed_contrasts[hypothesis])
        if contrast is not None and not np.isfinite(contrast):
            raise Phase4D5ProtocolAError("Holm family", f"nonfinite observed contrast for {hypothesis}")
        favorable = None if contrast is None else contrast > 0.0
        output.append({
            "metric_output_id": hypothesis.rsplit(":", 1)[0],
            "statistic": hypothesis.rsplit(":", 1)[1],
            "hypothesis_id": hypothesis,
            "state": state_by_hypothesis[hypothesis],
            "raw_p_value": observed_p_values.get(hypothesis),
            "multiplicity_p_value": multiplicity[hypothesis],
            "adjusted_p_value": adjusted[hypothesis].adjusted_p_value,
            "rank": adjusted[hypothesis].rank,
            "family_size": adjusted[hypothesis].family_size,
            "observed_contrast": contrast,
            "favorable": favorable,
            "rejected": adjusted[hypothesis].rejected and tested and bool(favorable),
        })
    return output


def aggregate_class_observations(
    occurrence_rows: Sequence[Mapping[str, object]],
    metric_rows: Mapping[tuple[str, str, str], float],
    *,
    metric_output_id: str,
    preferred_direction: str,
    positive_conditions: Sequence[str],
) -> list[dict[str, object]]:
    direction = PreferredDirection(preferred_direction)
    by_class_condition: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in occurrence_rows:
        by_class_condition[(int(row["class_label"]), str(row["condition_id"]))].append(row)
    classes = sorted({key[0] for key in by_class_condition})
    output = []
    for class_label in classes:
        baseline_rows = by_class_condition[(class_label, "alpha0")]
        baseline_error = float(np.mean([not bool(row["top1_correct"]) for row in baseline_rows]))
        for condition_id in positive_conditions:
            rows = by_class_condition[(class_label, condition_id)]
            downstream_error = float(np.mean([not bool(row["top1_correct"]) for row in rows]))
            metric_harms = [
                orient_harm(
                    metric_rows[(str(row["record_id"]), "alpha0", metric_output_id)],
                    metric_rows[(str(row["record_id"]), condition_id, metric_output_id)],
                    direction,
                )
                for row in rows
            ]
            perturbation_id, alpha_hex = condition_id.split(":", 1)
            output.append(
                {
                    "cluster_id": str(class_label),
                    "perturbation_id": perturbation_id,
                    "alpha": struct.unpack("<d", bytes.fromhex(alpha_hex))[0],
                    "metric_output_id": metric_output_id,
                    "metric_harm": float(np.mean(metric_harms)),
                    "downstream_harm": downstream_error - baseline_error,
                    "occurrence_count": len(rows),
                    "state": "complete",
                }
            )
    return output


def _padding(values: Sequence[float]) -> tuple[float, float]:
    minimum = min(0.0, *values)
    maximum = max(0.0, *values)
    width = maximum - minimum
    pad = 0.05 * (width if width > 0 else max(1.0, abs(minimum), abs(maximum)))
    return minimum - pad, maximum + pad


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def render_figure_payloads(
    figure1_rows: Sequence[Mapping[str, object]], figure2_rows: Sequence[Mapping[str, object]]
) -> dict[str, bytes]:
    if len(figure1_rows) != 520 or len(figure2_rows) != 13:
        raise Phase4D5ProtocolAError("figure source", "must contain 520 and 13 rows")
    figure1_fields = tuple(figure1_rows[0])
    figure2_fields = tuple(figure2_rows[0])
    f1_csv = _csv_bytes(figure1_rows, figure1_fields)
    f2_csv = _csv_bytes(figure2_rows, figure2_fields)
    rc = {
        "font.family": "DejaVu Sans",
        "svg.hashsalt": "rpe-phase4-d5-v1",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharey=True)
        y_values = [
            float(row["mean_downstream_harm"])
            for row in figure1_rows
            if row.get("mean_downstream_harm") is not None
        ]
        if not y_values:
            raise Phase4D5ProtocolAError("figure1", "must retain at least one evaluable metric")
        y_lim = _padding(y_values)
        for metric_index, metric_id in enumerate(METRIC_OUTPUT_IDS):
            ax = axes.flat[metric_index]
            selected_metric = [row for row in figure1_rows if row["metric_output_id"] == metric_id]
            for perturbation_id in PERTURBATION_IDS:
                selected = [row for row in selected_metric if row["perturbation_id"] == perturbation_id]
                if selected and selected[0]["metric_state"] == "complete":
                    ax.plot(
                        [float(row["mean_metric_harm"]) for row in selected],
                        [float(row["mean_downstream_harm"]) for row in selected],
                        marker="o", linewidth=1.5, color=COLORS[perturbation_id], label=perturbation_id.upper(),
                    )
            x_values = [
                float(row["mean_metric_harm"])
                for row in selected_metric
                if row["metric_state"] == "complete" and row.get("mean_metric_harm") is not None
            ]
            if x_values:
                ax.set_xlim(*_padding(x_values))
            else:
                ax.text(0.5, 0.5, str(selected_metric[0]["metric_state"]), ha="center", va="center", transform=ax.transAxes, color="#777777", rotation=25)
            ax.set_ylim(*y_lim)
            ax.set_title(metric_id)
            ax.axhline(0.0, color="#999999", linewidth=0.5)
            ax.axvline(0.0, color="#999999", linewidth=0.5)
        for index in range(len(METRIC_OUTPUT_IDS), 16):
            axes.flat[index].axis("off")
        axes.flat[0].legend(loc="best", fontsize=7)
        fig.suptitle("D5 / Protocol A / full_domain_core / P8–P12")
        fig.tight_layout()
        f1_png = io.BytesIO(); f1_svg = io.BytesIO()
        fig.savefig(f1_png, format="png", dpi=300, metadata={"Software": "raman-preproc-eval"})
        fig.savefig(f1_svg, format="svg", metadata={"Date": None})
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        y = np.arange(len(METRIC_OUTPUT_IDS))
        specs = (
            ("ag", "ag_lower", "ag_upper", "AG"),
            ("acc", "acc_lower", "acc_upper", "Acc-cross"),
            ("d_ag", "d_ag_lower", "d_ag_upper", "D_AG"),
            ("d_acc", "d_acc_lower", "d_acc_upper", "D_Acc"),
        )
        for ax, (key, low_key, high_key, title) in zip(axes, specs, strict=True):
            finite = []
            for index, row in enumerate(figure2_rows):
                value = row.get(key)
                if value is None:
                    if row.get("metric_state") != "complete":
                        ax.scatter(0.0, index, marker="x", color="#777777", s=12)
                        ax.text(0.02, index, str(row["metric_state"]), transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=4.5, color="#777777", clip_on=True)
                    continue
                value = float(value); low = float(row[low_key]); high = float(row[high_key])
                finite.extend((low, value, high))
                ax.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color="#1f77b4")
                if key in {"d_ag", "d_acc"}:
                    raw = row[f"{key}_raw_p"]
                    adjusted_p = row[f"{key}_adjusted_p"]
                    direction = "favorable" if bool(row[f"{key}_favorable"]) else "unfavorable"
                    label = f"raw={float(raw):.6g}; adj={float(adjusted_p):.6g}; {direction}"
                    ax.text(0.02, index, label, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=4.5, color="#333333", clip_on=True)
            if finite:
                ax.set_xlim(*_padding(finite))
            ax.axvline(0.0, color="#999999", linewidth=0.5)
            ax.set_title(title)
        axes[0].set_yticks(y, METRIC_OUTPUT_IDS)
        fig.suptitle("D5 / Protocol A / full_domain_core / secondary")
        fig.tight_layout()
        f2_png = io.BytesIO(); f2_svg = io.BytesIO()
        fig.savefig(f2_png, format="png", dpi=300, metadata={"Software": "raman-preproc-eval"})
        fig.savefig(f2_svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    return {
        "figure1_d5_protocol_a_full_domain.png": f1_png.getvalue(),
        "figure1_d5_protocol_a_full_domain.svg": f1_svg.getvalue(),
        "figure1_d5_protocol_a_full_domain_data.csv": f1_csv,
        "figure2_d5_protocol_a_full_domain.png": f2_png.getvalue(),
        "figure2_d5_protocol_a_full_domain.svg": f2_svg.getvalue(),
        "figure2_d5_protocol_a_full_domain_data.csv": f2_csv,
    }


def _phase1_source(record_order: int, record_id: str, class_label: int, mineral_name: str, spectrum: Spectrum1D) -> Phase1Source:
    axis_f4 = np.asarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.asarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(record_order, record_id, spectrum.sample_id or record_id, class_label, mineral_name, f"native::{_array_sha(spectrum.axis_cm1)}"),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, "<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, "<f4"),
        normalized_axis_float64_sha256=_array_sha(spectrum.axis_cm1),
        normalized_intensity_float64_sha256=_array_sha(spectrum.intensity),
        provenance=MappingProxyType({"license": None, "license_status": "not_stated", "retrieved_date": "2026-08-19", "sha256": "0" * 64, "source_artifact": "rruff", "source_url": "https://rruff.info"}),
    )


def _condition_id(perturbation_id: str, alpha: float) -> str:
    return f"{perturbation_id}:{struct.pack('<d', float(alpha)).hex()}"


def _metric_objects() -> dict[str, object]:
    return {
        "mse": MSEMetric(), "rmse": RMSEMetric(), "mae": MAEMetric(), "sam": SAMMetric(),
        "pearson_r": PearsonRMetric(), "nmse": NMSEMetric(), "wasserstein_1_cm1": Wasserstein1Metric(),
        "is_like_structure_to_noise": ISLikeStructureToNoiseMetric(),
    }


def _resolve_cwt(catalog: Phase3ClassicalCatalog) -> Phase3System:
    matches = [system for system in catalog.systems if system.system_id == CWT_SYSTEM_ID]
    if len(matches) != 1:
        raise Phase4D5ProtocolAError("CWT system", "must resolve exactly once")
    return matches[0]


def _operator_record_bundle(
    record_order: int,
    cohort_index: int,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    phase1_config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    admission: P10MemoryAdmission,
    grid: np.ndarray,
    alpha0_projection: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, Spectrum1D], dict[str, np.ndarray], list[dict[str, object]], list[dict[str, object]]]:
    source_spectrum = native_spectra[cohort_index]
    record_id = cohort.record_ids[cohort_index]
    source = _phase1_source(
        record_order,
        record_id,
        int(cohort.class_labels[cohort_index]),
        cohort.mineral_names[cohort_index],
        source_spectrum,
    )
    condition_spectra = {"alpha0": source_spectrum}
    condition_projections = {"alpha0": alpha0_projection}
    condition_rows = [{
        "record_order": record_order,
        "record_id": record_id,
        "condition_id": "alpha0",
        "source_spectrum_id": source_spectrum.spectrum_id,
        "output_spectrum_id": source_spectrum.spectrum_id,
        "axis_sha256": _array_sha(source_spectrum.axis_cm1),
        "intensity_sha256": _array_sha(source_spectrum.intensity),
        "projection_sha256": _array_sha(alpha0_projection, "<f4"),
    }]
    downstream = []
    cells = []
    alpha_zero_hashes = []
    for perturbation_id in PERTURBATION_IDS:
        cell = run_perturbation_cell(
            source,
            perturbation_id,
            phase1_config,
            sweep,
            p10_admission=admission,
        )
        if cell.status is not CellStatus.COMPLETE:
            raise Phase4D5ProtocolAError(
                "operator cell", f"{record_id}/{perturbation_id} is not complete"
            )
        outputs = []
        for item in cell.records:
            output = item.result.output
            outputs.append({
                "alpha": item.result.alpha,
                "alpha_float64_le_hex": item.alpha_float64_le_hex,
                "output_spectrum_id": output.spectrum_id,
                "axis_sha256": _array_sha(output.axis_cm1),
                "intensity_sha256": _array_sha(output.intensity),
                "diagnostics": _json_ready(item.result.diagnostics),
            })
            if item.result.alpha == 0.0:
                alpha_zero_hashes.append(
                    (_array_sha(output.axis_cm1), _array_sha(output.intensity))
                )
                continue
            condition_id = _condition_id(perturbation_id, item.result.alpha)
            condition_spectra[condition_id] = output
            values = _project(output, grid, 3.0)
            condition_projections[condition_id] = values
            condition_rows.append({
                "record_order": record_order,
                "record_id": record_id,
                "condition_id": condition_id,
                "source_spectrum_id": source_spectrum.spectrum_id,
                "output_spectrum_id": output.spectrum_id,
                "axis_sha256": _array_sha(output.axis_cm1),
                "intensity_sha256": _array_sha(output.intensity),
                "projection_sha256": _array_sha(values, "<f4"),
            })
            downstream.append({
                "record_id": record_id,
                "condition_id": condition_id,
                "values_sha256": _array_sha(values, "<f4"),
            })
        cells.append({
            "record_id": record_id,
            "record_order": record_order,
            "perturbation_id": perturbation_id,
            "state_digest": cell.state.state_digest if cell.state else None,
            "native_gate": _json_ready(cell.evidence.native_gate),
            "outputs": outputs,
        })
    source_hash = (
        _array_sha(source_spectrum.axis_cm1),
        _array_sha(source_spectrum.intensity),
    )
    if len(alpha_zero_hashes) != 5 or any(value != source_hash for value in alpha_zero_hashes):
        raise Phase4D5ProtocolAError("alpha-zero collapse", "operator identities differ")
    return cells, condition_spectra, condition_projections, condition_rows, downstream


def _scalar_metric_record_bundle(
    record_order: int,
    record_id: str,
    source: Spectrum1D,
    condition_spectra: Mapping[str, Spectrum1D],
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], float], dict[str, list[dict[str, object]]]]:
    objects = _metric_objects()
    rows = []
    values: dict[tuple[str, str, str], float] = {}
    failures: dict[str, list[dict[str, object]]] = {
        output_id: [] for output_id in METRIC_OUTPUT_IDS[:8]
    }
    for condition_id in CONDITION_IDS:
        candidate = condition_spectra[condition_id]
        for output_id in METRIC_OUTPUT_IDS[:8]:
            try:
                request = (
                    SingleSpectrumInput(candidate)
                    if output_id == "is_like_structure_to_noise"
                    else SpectrumPairInput(source, candidate)
                )
                result = evaluate_metric(objects[output_id], request)
                scalar = next(value for value in result.outputs if value.output_id == output_id)
                value = float(scalar.value)
                values[(record_id, condition_id, output_id)] = value
                rows.append({
                    "record_order": record_order,
                    "record_id": record_id,
                    "condition_id": condition_id,
                    "metric_output_id": output_id,
                    "state": "complete",
                    "value": value,
                    "diagnostics": _json_ready(result.diagnostics),
                    "exception": None,
                })
            except Exception as error:
                failure = {
                    "record_id": record_id,
                    "condition_id": condition_id,
                    "type": type(error).__name__,
                    "message": str(error),
                }
                failures[output_id].append(failure)
                rows.append({
                    "record_order": record_order,
                    "record_id": record_id,
                    "condition_id": condition_id,
                    "metric_output_id": output_id,
                    "state": "failed",
                    "value": None,
                    "diagnostics": {},
                    "exception": {"type": type(error).__name__, "message": str(error)},
                })
    return rows, values, failures


def _peak_metric_record_bundle(
    record_order: int,
    record_id: str,
    source: Spectrum1D,
    condition_spectra: Mapping[str, Spectrum1D],
    cwt_system: Phase3System,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[str, str, str], float], dict[str, list[dict[str, object]]]]:
    peak_metric = PeakDetectionCurvesMetric()
    rows = []
    receipts = []
    values: dict[tuple[str, str, str], float] = {}
    failures: dict[str, list[dict[str, object]]] = {
        output_id: [] for output_id in STRUCTURE_OUTPUT_IDS
    }
    reference = run_peak_detection_system(cwt_system, source)
    reference_success = reference.status in {
        PeakRunStatus.COMPLETE,
        PeakRunStatus.COMPLETE_WITH_WARNING,
    }
    for condition_id in CONDITION_IDS:
        result = reference if condition_id == "alpha0" else run_peak_detection_system(
            cwt_system, condition_spectra[condition_id]
        )
        successful = result.status in {
            PeakRunStatus.COMPLETE,
            PeakRunStatus.COMPLETE_WITH_WARNING,
        }
        receipts.append({
            "record_order": record_order,
            "record_id": record_id,
            "condition_id": condition_id,
            "status": result.status.value,
            "peaks_sha256": result.peaks_sha256,
            "peak_count": len(result.peaks),
            "warnings": [asdict(value) for value in result.warnings],
            "diagnostics": _json_ready(result.diagnostics),
            "error_code": result.error_code,
            "error_message": result.error_message,
        })
        if successful and reference_success:
            try:
                metric_result = evaluate_metric(
                    peak_metric,
                    PeakPairInput(
                        tuple(value.to_peak1d() for value in reference.peaks),
                        tuple(value.to_peak1d() for value in result.peaks),
                        2.0,
                        (0.0,),
                    ),
                )
                by_id = {value.output_id: value for value in metric_result.outputs}
                for output_id in STRUCTURE_OUTPUT_IDS:
                    value = float(by_id[output_id].value)
                    values[(record_id, condition_id, output_id)] = value
                    rows.append({
                        "record_order": record_order,
                        "record_id": record_id,
                        "condition_id": condition_id,
                        "metric_output_id": output_id,
                        "state": "complete",
                        "value": value,
                        "diagnostics": {},
                        "exception": None,
                    })
                continue
            except Exception as error:
                failure_type = type(error).__name__
                failure_message = str(error)
        else:
            failure_type = "PeakRunStatus"
            failure_message = result.status.value
        for output_id in STRUCTURE_OUTPUT_IDS:
            failure = {
                "record_id": record_id,
                "condition_id": condition_id,
                "type": failure_type,
                "message": failure_message,
            }
            failures[output_id].append(failure)
            rows.append({
                "record_order": record_order,
                "record_id": record_id,
                "condition_id": condition_id,
                "metric_output_id": output_id,
                "state": "failed",
                "value": None,
                "diagnostics": {},
                "exception": {"type": failure_type, "message": failure_message},
            })
    return rows, receipts, values, failures


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _run_identity(config: Phase4D5ProtocolAConfig, code: Mapping[str, object], environment: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "authorities": config.authorities,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config_authority": _authority_document(),
        "config_sha256": config.sha256,
        "cwt_system_id": config.cwt_system_id,
        "environment": environment,
        "figure_contract": config.figure_contract,
        "frozen_identities": config.frozen_identities,
        "inference": {
            "bootstrap_resamples": config.bootstrap_resamples,
            "sign_flip_resamples": config.sign_flip_resamples,
            "random_seed": config.random_seed,
            "holm_slots": config.expected_holm_slot_count,
        },
        "metric_output_ids": config.metric_output_ids,
        "protocol": "A",
        "tier": "full_domain_core",
    }
    return RUN_PREFIX + _sha_bytes(_canonical(identity)), identity


@threadpool_limits.wrap(limits=1, user_api="blas")
def build_phase4_d5_protocol_a_from_inputs(
    output_dir: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    classical_catalog: Phase3ClassicalCatalog,
    config: Phase4D5ProtocolAConfig,
    worker_count: int,
    inference_resamples: int | None = None,
) -> Phase4D5ProtocolASummary:
    if not config.synthetic_fixture and inference_resamples is not None:
        raise Phase4D5ProtocolAError("inference_resamples", "test override forbidden for frozen run")
    if tuple(sweep.alpha_grid) != ALPHA_GRID or sweep.sha256 != config.authorities.get("sweep_sha256"):
        raise Phase4D5ProtocolAError("sweep", "identity mismatch")
    if phase1_config.file_sha256 != config.authorities.get("phase1_core_config_sha256"):
        raise Phase4D5ProtocolAError("phase1 config", "identity mismatch")
    if len(cohort.record_ids) != config.cohort_record_count or len(native_spectra) != len(cohort.record_ids):
        raise Phase4D5ProtocolAError("cohort", "count mismatch")
    expected_spectrum_ids = tuple(f"rruff_raman_raw::{record_id}" for record_id in cohort.record_ids)
    if tuple(value.spectrum_id for value in native_spectra) != expected_spectrum_ids:
        raise Phase4D5ProtocolAError("native spectra", "order mismatch")
    query_positions: dict[int, list[dict[str, int]]] = defaultdict(list)
    split_occurrences = []
    for split in cohort.splits:
        for query_order, raw_index in enumerate(split.query_indices):
            index = int(raw_index)
            query_positions[index].append({"split_seed": int(split.seed), "query_order": query_order})
            split_occurrences.append((int(split.seed), query_order, index))
    unique_indices = sorted(query_positions, key=lambda index: cohort.record_ids[index])
    if len(unique_indices) != config.query_record_count or len(split_occurrences) != config.query_occurrence_count:
        raise Phase4D5ProtocolAError("query ledgers", "count mismatch")
    observed_splits = tuple(split.split_sha256 for split in cohort.splits)
    expected_splits = tuple(config.frozen_identities.get("split_sha256", ()))
    if observed_splits != expected_splits:
        raise Phase4D5ProtocolAError("split identities", "do not match frozen config")
    query_record_ids = [cohort.record_ids[index] for index in unique_indices]
    query_group_ids = sorted({cohort.group_ids[index] for index in unique_indices})
    query_class_ids = sorted({str(int(cohort.class_labels[index])) for index in unique_indices})
    for key, observed in (
        ("query_record_ids_sha256", _ids_digest(query_record_ids)),
        ("query_group_ids_sha256", _ids_digest(query_group_ids)),
        ("query_class_labels_sha256", _ids_digest(query_class_ids)),
    ):
        if observed != config.frozen_identities.get(key):
            raise Phase4D5ProtocolAError(key, "literal query-union identity mismatch")
    grid = _grid(config)
    projected: dict[tuple[str, str], np.ndarray] = {}
    downstream_rows = []
    for index, spectrum in enumerate(native_spectra):
        values = _project(spectrum, grid, config.support_max_gap_cm1)
        record_id = cohort.record_ids[index]
        projected[(record_id, "alpha0")] = values
        downstream_rows.append({"record_id": record_id, "condition_id": "alpha0", "values_sha256": _array_sha(values, "<f4")})
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D5ProtocolAError("worker_count", "must be a positive integer")
    p10_admission = P10MemoryAdmission(64 * 2**30)
    estimates = [estimate_p10_peak_bytes(native_spectra[index].axis_cm1.size) for index in unique_indices]
    if max(estimates) > 64 * 2**30:
        raise Phase4D5ProtocolAError("P10 admission", "record exceeds frozen budget")
    bundles = {}
    with threadpool_limits(limits=1, user_api="blas"):
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                record_order: executor.submit(
                    _operator_record_bundle,
                    record_order,
                    cohort_index,
                    cohort,
                    native_spectra,
                    phase1_config,
                    sweep,
                    p10_admission,
                    grid,
                    projected[(cohort.record_ids[cohort_index], "alpha0")],
                )
                for record_order, cohort_index in enumerate(unique_indices)
            }
            for record_order, future in futures.items():
                bundles[record_order] = future.result()
    operator_cells = []
    conditions: dict[tuple[str, str], Spectrum1D] = {}
    record_condition_rows = []
    for record_order, cohort_index in enumerate(unique_indices):
        record_id = cohort.record_ids[cohort_index]
        cells, spectra, projections, rows, current_downstream = bundles[record_order]
        operator_cells.extend(cells)
        record_condition_rows.extend(rows)
        downstream_rows.extend(current_downstream)
        for condition_id, spectrum in spectra.items():
            conditions[(record_id, condition_id)] = spectrum
        for condition_id, values in projections.items():
            projected[(record_id, condition_id)] = values
    if len(record_condition_rows) != config.expected_query_condition_count or len(downstream_rows) != config.expected_projected_row_count:
        raise Phase4D5ProtocolAError("condition counts", "mismatch")

    metric_values: dict[tuple[str, str, str], float] = {}
    metric_failures: dict[str, list[dict[str, object]]] = {output_id: [] for output_id in METRIC_OUTPUT_IDS}
    metric_rows = []
    scalar_bundles = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            record_order: executor.submit(
                _scalar_metric_record_bundle,
                record_order,
                cohort.record_ids[cohort_index],
                native_spectra[cohort_index],
                {condition_id: conditions[(cohort.record_ids[cohort_index], condition_id)] for condition_id in CONDITION_IDS},
            )
            for record_order, cohort_index in enumerate(unique_indices)
        }
        for record_order, future in futures.items():
            scalar_bundles[record_order] = future.result()
    for record_order in range(len(unique_indices)):
        rows, values, failures = scalar_bundles[record_order]
        metric_rows.extend(rows)
        metric_values.update(values)
        for output_id, current in failures.items():
            metric_failures[output_id].extend(current)
    cwt_system = _resolve_cwt(classical_catalog)
    peak_receipts = []
    peak_bundles = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            record_order: executor.submit(
                _peak_metric_record_bundle,
                record_order,
                cohort.record_ids[cohort_index],
                native_spectra[cohort_index],
                {condition_id: conditions[(cohort.record_ids[cohort_index], condition_id)] for condition_id in CONDITION_IDS},
                cwt_system,
            )
            for record_order, cohort_index in enumerate(unique_indices)
        }
        for record_order, future in futures.items():
            peak_bundles[record_order] = future.result()
    for record_order in range(len(unique_indices)):
        rows, receipts, values, failures = peak_bundles[record_order]
        metric_rows.extend(rows)
        peak_receipts.extend(receipts)
        metric_values.update(values)
        for output_id, current in failures.items():
            metric_failures[output_id].extend(current)
    planned_metric_rows = config.expected_query_condition_count
    counts = {output_id: {"complete": planned_metric_rows - len(metric_failures[output_id]), "planned": planned_metric_rows} for output_id in METRIC_OUTPUT_IDS}
    metric_states = metric_preflight_states(counts)
    condition_order = {condition_id: index for index, condition_id in enumerate(CONDITION_IDS)}
    metric_order = {output_id: index for index, output_id in enumerate(METRIC_OUTPUT_IDS)}
    metric_rows.sort(
        key=lambda row: (
            int(row["record_order"]),
            condition_order[str(row["condition_id"])],
            metric_order[str(row["metric_output_id"])],
        )
    )
    preflight = {
        "claim_boundary": "outcome_blind_preflight_no_predictions_correctness_scores_alignment_or_pvalues",
        "downstream_role_state": "complete",
        "metric_states": metric_states,
        "operator_cell_count": len(operator_cells),
        "record_condition_count": len(record_condition_rows),
        "projected_row_count": len(downstream_rows),
    }

    prediction_rows = []
    for split in cohort.splits:
        query_indices = tuple(int(value) for value in split.query_indices)
        library_indices = tuple(int(value) for value in split.library_indices)
        query_ids = tuple(cohort.record_ids[index] for index in query_indices)
        library_ids = tuple(cohort.record_ids[index] for index in library_indices)
        library = np.stack([projected[(record_id, "alpha0")] for record_id in library_ids])
        for condition_id in CONDITION_IDS:
            query = np.stack([projected[(record_id, condition_id)] for record_id in query_ids])
            result = match_d5_protocol_a_799(cohort, split, condition_id=condition_id, query_record_ids=query_ids, library_record_ids=library_ids, query_values=query, library_values=library)
            for query_order, cohort_index in enumerate(query_indices):
                top_k = min(5, result.ranked_class_labels.shape[1])
                prediction_rows.append({"split_seed": int(split.seed), "split_sha256": split.split_sha256, "query_order": query_order, "cohort_index": cohort_index, "record_id": cohort.record_ids[cohort_index], "group_id": cohort.group_ids[cohort_index], "class_label": int(cohort.class_labels[cohort_index]), "condition_id": condition_id, "top1_class_label": int(result.ranked_class_labels[query_order, 0]), "top1_score": float(result.ranked_class_scores[query_order, 0]), "top1_correct": bool(result.top1_correct[query_order]), "top5_class_labels": [int(value) for value in result.ranked_class_labels[query_order, :top_k]], "top5_scores": [float(value) for value in result.ranked_class_scores[query_order, :top_k]], "top5_correct": bool(result.top5_correct[query_order])})
    if len(prediction_rows) != config.expected_prediction_row_count:
        raise Phase4D5ProtocolAError("prediction rows", "count mismatch")

    class_rows = []
    alignment_rows = []
    eligible_metric_ids = [output_id for output_id in METRIC_OUTPUT_IDS if metric_states[output_id]["state"] == "complete"]
    for metric_id in METRIC_OUTPUT_IDS:
        if metric_id in eligible_metric_ids:
            current = aggregate_class_observations(prediction_rows, metric_values, metric_output_id=metric_id, preferred_direction=config.metric_directions[metric_id].value, positive_conditions=CONDITION_IDS[1:])
            class_rows.extend(current)
        else:
            for class_label in sorted(set(int(value) for value in cohort.class_labels)):
                for condition_id in CONDITION_IDS[1:]:
                    perturbation_id, alpha_hex = condition_id.split(":", 1)
                    class_rows.append({"cluster_id": str(class_label), "perturbation_id": perturbation_id, "alpha": struct.unpack("<d", bytes.fromhex(alpha_hex))[0], "metric_output_id": metric_id, "metric_harm": None, "downstream_harm": None, "occurrence_count": 0, "state": "not_evaluable_incomplete_grid"})
    by_metric = defaultdict(list)
    for row in class_rows:
        if row.get("metric_harm") is not None:
            by_metric[row["metric_output_id"]].append(AlignmentObservation(row["cluster_id"], row["perturbation_id"], row["alpha"], row["metric_harm"], row["downstream_harm"]))
    reference = tuple(by_metric["mse"])
    metric_results: dict[str, dict[str, object]] = {}
    bootstrap_rows = []
    sign_rows = []
    raw_p_values: dict[str, float] = {}
    observed_contrasts: dict[str, float] = {}
    effective_resamples = config.bootstrap_resamples if inference_resamples is None else inference_resamples
    effective_sign_flips = config.sign_flip_resamples if inference_resamples is None else max(64, inference_resamples)
    downstream_constant = False
    try:
        reference_gap = alignment_gap(reference)
        reference_acc = cross_perturbation_accuracy(reference)
        mse_intervals = bulk_paired_cluster_bootstrap(
            reference,
            reference,
            resamples=effective_resamples,
            confidence_level=config.confidence_level,
            random_seed=config.random_seed,
        )
        metric_results["mse"] = {"metric_output_id": "mse", "metric_state": "complete", "ag": reference_gap.alignment_gap, "acc": reference_acc.accuracy, "comparison": None}
    except AlignmentValidationError as error:
        if error.path != "constant downstream":
            raise
        downstream_constant = True
        mse_intervals = None
        metric_results["mse"] = {"metric_output_id": "mse", "metric_state": "not_evaluable_constant_downstream", "ag": None, "acc": None, "comparison": None}
    for metric_id in CANDIDATE_OUTPUT_IDS:
        if metric_id not in by_metric:
            metric_results[metric_id] = {"metric_output_id": metric_id, "metric_state": metric_states[metric_id]["state"], "ag": None, "acc": None, "comparison": None}
            continue
        if downstream_constant:
            metric_results[metric_id] = {"metric_output_id": metric_id, "metric_state": "not_evaluable_constant_downstream", "ag": None, "acc": None, "comparison": None}
            continue
        candidate = tuple(by_metric[metric_id])
        comparison = compare_alignment(reference, candidate)
        bootstrap = bulk_paired_cluster_bootstrap(reference, candidate, resamples=effective_resamples, confidence_level=config.confidence_level, random_seed=config.random_seed)
        ag_flip = paired_contribution_sign_flip(tuple(value.value for value in comparison.ag_contribution_differences), aggregation="sum", resamples=effective_sign_flips, random_seed=config.random_seed)
        acc_flip = paired_contribution_sign_flip(tuple(value.value for value in comparison.acc_contribution_differences), aggregation="mean", resamples=effective_sign_flips, random_seed=config.random_seed)
        raw_p_values[f"{metric_id}:d_ag"] = ag_flip.p_value
        raw_p_values[f"{metric_id}:d_acc"] = acc_flip.p_value
        observed_contrasts[f"{metric_id}:d_ag"] = comparison.d_ag
        observed_contrasts[f"{metric_id}:d_acc"] = comparison.d_acc
        metric_results[metric_id] = {"metric_output_id": metric_id, "metric_state": "complete", "ag": comparison.candidate_gap.alignment_gap, "acc": comparison.candidate_accuracy.accuracy, "comparison": _json_ready(comparison)}
        bootstrap_rows.append({"metric_output_id": metric_id, **_json_ready(bootstrap)})
        sign_rows.extend(({"metric_output_id": metric_id, "statistic": "d_ag", **_json_ready(ag_flip)}, {"metric_output_id": metric_id, "statistic": "d_acc", **_json_ready(acc_flip)}))
    family = fixed_holm_family(raw_p_values, metric_states, observed_contrasts)
    figure2_rows = []
    bootstrap_by_metric = {row["metric_output_id"]: row for row in bootstrap_rows}
    family_by_key = {(row["metric_output_id"], row["statistic"]): row for row in family}
    for metric_id in METRIC_OUTPUT_IDS:
        result = metric_results[metric_id]
        if metric_id == "mse" and mse_intervals is not None:
            figure2_rows.append({
                "metric_output_id": metric_id, "preferred_direction": config.metric_directions[metric_id].value,
                "metric_state": "complete", "ag": result["ag"],
                "ag_lower": mse_intervals.reference_ag_interval[0], "ag_upper": mse_intervals.reference_ag_interval[1],
                "acc": result["acc"], "acc_lower": mse_intervals.reference_acc_interval[0], "acc_upper": mse_intervals.reference_acc_interval[1],
                "pair_count": reference_acc.pair_count, "strict_agreement_count": reference_acc.strict_agreement_count,
                "strict_disagreement_count": reference_acc.strict_disagreement_count, "metric_tie_count": reference_acc.metric_tie_count,
                "downstream_tie_count": reference_acc.downstream_tie_count, "double_tie_count": reference_acc.double_tie_count,
                "d_ag": None, "d_ag_lower": None, "d_ag_upper": None, "d_ag_favorable": None,
                "d_acc": None, "d_acc_lower": None, "d_acc_upper": None, "d_acc_favorable": None,
                "d_ag_raw_p": None, "d_ag_adjusted_p": None, "d_ag_holm_rank": None, "d_ag_rejected": None,
                "d_acc_raw_p": None, "d_acc_adjusted_p": None, "d_acc_holm_rank": None, "d_acc_rejected": None,
            })
        elif result["metric_state"] == "complete":
            boot = bootstrap_by_metric[metric_id]; comparison = result["comparison"]
            ag_holm = family_by_key[(metric_id, "d_ag")]; acc_holm = family_by_key[(metric_id, "d_acc")]
            candidate_acc = comparison["candidate_accuracy"]
            figure2_rows.append({
                "metric_output_id": metric_id, "preferred_direction": config.metric_directions[metric_id].value,
                "metric_state": "complete", "ag": result["ag"],
                "ag_lower": boot["candidate_ag_interval"][0], "ag_upper": boot["candidate_ag_interval"][1],
                "acc": result["acc"], "acc_lower": boot["candidate_acc_interval"][0], "acc_upper": boot["candidate_acc_interval"][1],
                "pair_count": candidate_acc["pair_count"], "strict_agreement_count": candidate_acc["strict_agreement_count"],
                "strict_disagreement_count": candidate_acc["strict_disagreement_count"], "metric_tie_count": candidate_acc["metric_tie_count"],
                "downstream_tie_count": candidate_acc["downstream_tie_count"], "double_tie_count": candidate_acc["double_tie_count"],
                "d_ag": comparison["d_ag"], "d_ag_lower": boot["d_ag_interval"][0], "d_ag_upper": boot["d_ag_interval"][1],
                "d_ag_favorable": comparison["d_ag"] > 0.0, "d_acc": comparison["d_acc"],
                "d_acc_lower": boot["d_acc_interval"][0], "d_acc_upper": boot["d_acc_interval"][1], "d_acc_favorable": comparison["d_acc"] > 0.0,
                "d_ag_raw_p": ag_holm["raw_p_value"], "d_ag_adjusted_p": ag_holm["adjusted_p_value"],
                "d_ag_holm_rank": ag_holm["rank"], "d_ag_rejected": ag_holm["rejected"],
                "d_acc_raw_p": acc_holm["raw_p_value"], "d_acc_adjusted_p": acc_holm["adjusted_p_value"],
                "d_acc_holm_rank": acc_holm["rank"], "d_acc_rejected": acc_holm["rejected"],
            })
        else:
            ag_holm = family_by_key.get((metric_id, "d_ag")); acc_holm = family_by_key.get((metric_id, "d_acc"))
            figure2_rows.append({
                "metric_output_id": metric_id, "preferred_direction": config.metric_directions[metric_id].value,
                "metric_state": result["metric_state"], "ag": None, "ag_lower": None, "ag_upper": None,
                "acc": None, "acc_lower": None, "acc_upper": None, "pair_count": None,
                "strict_agreement_count": None, "strict_disagreement_count": None, "metric_tie_count": None,
                "downstream_tie_count": None, "double_tie_count": None,
                "d_ag": None, "d_ag_lower": None, "d_ag_upper": None, "d_ag_favorable": None,
                "d_acc": None, "d_acc_lower": None, "d_acc_upper": None, "d_acc_favorable": None,
                "d_ag_raw_p": None if ag_holm is None else ag_holm["raw_p_value"],
                "d_ag_adjusted_p": 1.0 if ag_holm is None else ag_holm["adjusted_p_value"],
                "d_ag_holm_rank": None if ag_holm is None else ag_holm["rank"], "d_ag_rejected": False,
                "d_acc_raw_p": None if acc_holm is None else acc_holm["raw_p_value"],
                "d_acc_adjusted_p": 1.0 if acc_holm is None else acc_holm["adjusted_p_value"],
                "d_acc_holm_rank": None if acc_holm is None else acc_holm["rank"], "d_acc_rejected": False,
            })
    figure1_rows = []
    for metric_id in METRIC_OUTPUT_IDS:
        current_rows = [row for row in class_rows if row["metric_output_id"] == metric_id and row.get("metric_harm") is not None]
        for perturbation_id in PERTURBATION_IDS:
            for alpha in POSITIVE_ALPHAS:
                selected = [row for row in current_rows if row["perturbation_id"] == perturbation_id and row["alpha"] == alpha]
                figure1_rows.append({"metric_output_id": metric_id, "perturbation_id": perturbation_id, "alpha": alpha, "mean_metric_harm": None if not selected else float(np.mean([row["metric_harm"] for row in selected])), "mean_downstream_harm": None if not selected else float(np.mean([row["downstream_harm"] for row in selected])), "metric_state": metric_states[metric_id]["state"]})
    figure_payloads = render_figure_payloads(figure1_rows, figure2_rows)
    condition_summary_rows = []
    for condition_id in CONDITION_IDS:
        selected = [row for row in prediction_rows if row["condition_id"] == condition_id]
        labels = sorted({row["class_label"] for row in selected})
        macro1 = float(np.mean([np.mean([row["top1_correct"] for row in selected if row["class_label"] == label]) for label in labels]))
        macro5 = float(np.mean([np.mean([row["top5_correct"] for row in selected if row["class_label"] == label]) for label in labels]))
        condition_summary_rows.append({"condition_id": condition_id, "query_occurrence_count": len(selected), "top1_macro_class_accuracy": macro1, "top5_macro_class_accuracy": macro5, "top1_micro_accuracy": float(np.mean([row["top1_correct"] for row in selected])), "top5_micro_accuracy": float(np.mean([row["top5_correct"] for row in selected]))})
    code = {str(k): dict(v) for k, v in config.code_authority.items()} if config.synthetic_fixture else _code_document()
    environment = dict(config.environment_authority) if config.environment_authority else _environment()
    run_id, run_identity = _run_identity(config, code, environment)
    alignment_rows = [metric_results[metric_id] for metric_id in METRIC_OUTPUT_IDS]
    manifest = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "claim_boundary": config.claim_boundary, "code": code, "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256}, "counts": {"operator_cells": len(operator_cells), "record_conditions": len(record_condition_rows), "metric_values": len(metric_rows), "peak_receipts": len(peak_receipts), "downstream_rows": len(downstream_rows), "matcher_predictions": len(prediction_rows), "class_observations": len(class_rows), "alignment_results": len(alignment_rows), "bootstrap_results": len(bootstrap_rows), "sign_flip_results": len(sign_rows), "holm_family": len(family)}, "environment": environment, "experiment_id": EXPERIMENT_ID, "metric_states": metric_states, "perturbation_ids": list(PERTURBATION_IDS), "protocol": "A", "run_id": run_id, "run_identity": run_identity, "status": "complete", "synthetic_fixture": config.synthetic_fixture, "tier": "full_domain_core"}
    complete = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "run_id": run_id, "status": "complete"}
    condition_fields = tuple(condition_summary_rows[0])
    table_fields = tuple(figure2_rows[0])
    payloads = {
        "config.json": config.raw_bytes, "preflight.json": _canonical(preflight),
        "operator_cells.jsonl": _jsonl(operator_cells), "record_conditions.jsonl": _jsonl(record_condition_rows),
        "metric_values.jsonl": _jsonl(metric_rows), "peak_receipts.jsonl": _jsonl(peak_receipts),
        "downstream_rows.jsonl": _jsonl(downstream_rows), "matcher_predictions.jsonl": _jsonl(prediction_rows),
        "condition_summary.csv": _csv_bytes(condition_summary_rows, condition_fields),
        "class_observations.jsonl": _jsonl(class_rows), "alignment_results.jsonl": _jsonl(alignment_rows),
        "bootstrap_results.jsonl": _jsonl(bootstrap_rows), "sign_flip_results.jsonl": _jsonl(sign_rows),
        "holm_family.jsonl": _jsonl(family), "d5_full_domain_secondary_table.csv": _csv_bytes(figure2_rows, table_fields),
        "manifest.json": _canonical(manifest), "complete.json": _canonical(complete), **figure_payloads,
    }
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Phase4D5ProtocolAError("output_dir", "must not exist")
    output_dir.mkdir(parents=True)
    for name in ARTIFACT_PAYLOAD_FILES:
        (output_dir / name).write_bytes(payloads[name])
    (output_dir / "SHA256SUMS").write_text("".join(f"{_sha_bytes(payloads[name])}  {name}\n" for name in ARTIFACT_PAYLOAD_FILES), encoding="utf-8")
    return Phase4D5ProtocolASummary(output_dir, run_id, "complete", config.query_record_count, len(prediction_rows), len(metric_rows), len(class_rows))


def build_phase4_d5_protocol_a(output_root: Path, *, worker_count: int = 16) -> Phase4D5ProtocolASummary:
    config = load_phase4_d5_protocol_a_config(ROOT / CONFIG_RELATIVE_PATH)
    cohort = load_d5_raw_cohort(ROOT / D5_CONFIG_RELATIVE_PATH, ROOT / DATASET_RELATIVE_PATH)
    native = load_d5_native_spectra(ROOT / DATASET_RELATIVE_PATH, cohort.record_ids)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1 = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    catalog = load_classical_catalog(ROOT / CATALOG_RELATIVE_PATH, project_root=ROOT)
    code = _code_document(); environment = _environment(); run_id, _ = _run_identity(config, code, environment)
    return build_phase4_d5_protocol_a_from_inputs(Path(output_root) / run_id, cohort=cohort, native_spectra=native, sweep=sweep, phase1_config=phase1, classical_catalog=catalog, config=config, worker_count=worker_count)


__all__ = [
    "ARTIFACT_PAYLOAD_FILES", "METRIC_OUTPUT_IDS", "Phase4D5ProtocolAConfig", "Phase4D5ProtocolAError", "Phase4D5ProtocolASummary",
    "aggregate_class_observations", "build_phase4_d5_protocol_a", "build_phase4_d5_protocol_a_from_inputs", "fixed_holm_family",
    "load_phase4_d5_protocol_a_config", "match_d5_protocol_a_799", "metric_preflight_states", "parse_phase4_d5_protocol_a_config", "render_figure_payloads",
]
