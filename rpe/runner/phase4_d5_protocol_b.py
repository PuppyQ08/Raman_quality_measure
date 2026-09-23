from __future__ import annotations

import hashlib
import csv
import io
import json
import math
import os
import platform
import struct
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/rpe-matplotlib-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from rpe.downstream.rruff import (
    D5LibraryQuerySplit,
    D5RawCohort,
    load_d5_native_spectra,
    load_d5_raw_cohort,
)
from rpe.downstream.rruff_matching import D5MatchingResult, match_d5_protocol_a_values
from rpe.evaluation import PreferredDirection, Spectrum1D
from rpe.perturb import PerturbationSweepConfig, load_perturbation_sweep_config
from rpe.runner.phase1_config import Phase1CoreConfig, load_phase1_core_config
from rpe.runner.phase1_perturbations import (
    P10MemoryAdmission,
    estimate_p10_peak_bytes,
    run_perturbation_cell,
)
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.runner.phase1_types import CellStatus


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d5_protocol_b_full_domain_v1.json"
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d5_protocol_b_authority.py"
D5_CONFIG_RELATIVE_PATH = "experiments/phase05/configs/d5_rruff_protocol.json"
DATASET_RELATIVE_PATH = "data/unified/rruff_raman_raw"
SWEEP_RELATIVE_PATH = "experiments/shared/raman_perturbation_sweep_v1.json"
PHASE1_CONFIG_RELATIVE_PATH = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
PROTOCOL_A_RELATIVE_PATH = "results/phase4/d5_protocol_a_full_domain_v1/phase4-d5-protocol-a-full-domain-65e6a471f6c946de8973fee49e7eed8d0020393afef3831916349416274818fc"
ELIGIBILITY_RELATIVE_PATH = "results/phase4/d5_protocol_b_all_role_eligibility_v1/phase4-d5-protocol-b-all-role-eligibility-529f4793ca0047fd8cf09d331b8101a59c5f3bff40716aaf167f6dd7dd836ebb"
CODE_RELATIVE_PATHS = (
    "rpe/alignment/contracts.py",
    "rpe/alignment/core.py",
    "rpe/alignment/bulk.py",
    "rpe/alignment/inference.py",
    "rpe/downstream/rruff.py",
    "rpe/downstream/rruff_matching.py",
    "rpe/evaluation/contracts.py",
    "rpe/perturb/contracts.py",
    "rpe/perturb/sweep.py",
    "rpe/perturb/axis_transform.py",
    "rpe/perturb/baseline_distortion.py",
    "rpe/perturb/gaussian_noise.py",
    "rpe/perturb/correlated_noise.py",
    "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py",
    "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d5_protocol_b.py",
    "rpe/runner/phase4_d5_protocol_b_verifier.py",
    "tools/run_phase4_d5_protocol_b.py",
)
EXPERIMENT_ID = "phase4-d5-protocol-b-full-domain-v1"
PROTOCOL = "B"
TIER = "full_domain_core"
ARTIFACT_SCHEMA_VERSION = "phase4-d5-protocol-b-full-domain-artifact-v1"
RUN_PREFIX = "phase4-d5-protocol-b-full-domain-"
PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
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
CONDITION_BRIDGE_SHA256 = "00c27b1f6273b636c58156c3269d8c5877b7418be431646d84a0ba59bd219398"
OPERATOR_BRIDGE_SHA256 = "9ce29e2172a4bc4b2328b67a3e26e4409efac6efff7561d005a0910d74933cc0"
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "operator_cells.jsonl",
    "record_conditions.jsonl",
    "downstream_rows.jsonl",
    "matcher_predictions.jsonl",
    "condition_summary.csv",
    "class_observations.jsonl",
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "figure1_d5_protocol_b_full_domain.png",
    "figure1_d5_protocol_b_full_domain.svg",
    "figure1_d5_protocol_b_full_domain_data.csv",
    "figure2_d5_protocol_b_full_domain.png",
    "figure2_d5_protocol_b_full_domain.svg",
    "figure2_d5_protocol_b_full_domain_data.csv",
    "d5_protocol_b_full_domain_secondary_table.csv",
    "manifest.json",
)
PROTOCOL_A_ALLOWED_FILES = frozenset(
    {
        "config.json",
        "manifest.json",
        "complete.json",
        "operator_cells.jsonl",
        "record_conditions.jsonl",
        "metric_values.jsonl",
        "peak_receipts.jsonl",
        "SHA256SUMS",
    }
)
ELIGIBILITY_ALLOWED_FILES = frozenset(
    {
        "config.json",
        "manifest.json",
        "complete.json",
        "unique_records.jsonl",
        "role_occurrences.jsonl",
        "operator_cells.jsonl",
        "record_conditions.jsonl",
        "SHA256SUMS",
    }
)


class Phase4D5ProtocolBError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D5ProtocolBConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    protocol: str
    tier: str
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    condition_ids: tuple[str, ...]
    metric_output_ids: tuple[str, ...]
    metric_directions: Mapping[str, PreferredDirection]
    artifact_payload_files: tuple[str, ...]
    authorities: Mapping[str, str]
    parent_artifacts: Mapping[str, object]
    frozen_identities: Mapping[str, object]
    inherited_rulings: Mapping[str, object]
    claim_boundary: str
    condition_bridge_sha256: str
    operator_bridge_sha256: str
    full_record_count: int
    group_count: int
    class_count: int
    query_record_count: int
    query_occurrence_count: int
    library_occurrence_count: int
    split_count: int
    expected_operator_cell_count: int
    expected_apply_check_count: int
    expected_full_condition_count: int
    expected_query_condition_count: int
    expected_metric_row_count: int
    expected_peak_receipt_count: int
    expected_matcher_call_count: int
    expected_prediction_row_count: int
    expected_class_observation_count_per_metric: int
    expected_class_observation_count: int
    expected_holm_slot_count: int
    expected_figure1_row_count: int
    expected_figure2_row_count: int
    expected_secondary_table_row_count: int
    bootstrap_resamples: int
    sign_flip_resamples: int
    random_seed: int
    confidence_level: float
    holm_alpha: float
    support_start_cm1: float
    support_stop_cm1: float
    support_step_cm1: float
    support_point_count: int
    support_max_gap_cm1: float
    figure_contract: Mapping[str, object]
    code_authority: Mapping[str, object]
    environment_authority: Mapping[str, object]
    trust_anchor: Mapping[str, object]


@dataclass(frozen=True)
class ProtocolBAuthorityBridge:
    document: Mapping[str, object]
    metric_values: Mapping[tuple[str, str, str], float]
    peak_receipts: tuple[Mapping[str, object], ...]
    query_record_ids: tuple[str, ...]
    query_occurrences: tuple[Mapping[str, object], ...]
    protocol_a_conditions: tuple[Mapping[str, object], ...]
    eligibility_conditions: tuple[Mapping[str, object], ...]
    eligibility_operator_cells: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class Phase4D5ProtocolBSummary:
    path: Path
    run_id: str
    status: str
    full_record_count: int
    prediction_row_count: int
    class_observation_count: int


def _json_ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _canonical(value: object) -> bytes:
    try:
        text = json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise Phase4D5ProtocolBError("canonical JSON", str(error)) from error
    return (text + "\n").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(values: Sequence[str]) -> str:
    return _sha_bytes(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _class_digest(values: Sequence[int]) -> str:
    ordered = [str(value) for value in sorted(set(int(item) for item in values))]
    return _sha_bytes(("\n".join(ordered) + "\n").encode("utf-8"))


def _code_document() -> dict[str, object]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _authority_document() -> dict[str, object]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    return {"bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _lower_sha(path: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase4D5ProtocolBError(path, "must be a lowercase SHA-256")
    return value


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D5ProtocolBError(path, "must be an object")
    return {str(key): item for key, item in value.items()}


def _strings(path: str, value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise Phase4D5ProtocolBError(path, "must be a nonempty string list")
    return tuple(value)


def _integer(path: str, value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4D5ProtocolBError(path, "must be an integer")
    if positive and value <= 0:
        raise Phase4D5ProtocolBError(path, "must be positive")
    return value


def _number(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D5ProtocolBError(path, "must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise Phase4D5ProtocolBError(path, "must be finite")
    return converted


def _condition_ids(perturbations: Sequence[str], alpha_grid: Sequence[float]) -> tuple[str, ...]:
    return ("alpha0",) + tuple(
        f"{perturbation_id}:{np.float64(alpha).tobytes().hex()}"
        for perturbation_id in perturbations
        for alpha in alpha_grid[1:]
    )


def _environment_document() -> dict[str, object]:
    import h5py
    import matplotlib
    import scipy
    import sklearn
    import threadpoolctl

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


def parse_phase4_d5_protocol_b_config(
    path: Path,
    raw: bytes,
    *,
    require_frozen_identity: bool,
) -> Phase4D5ProtocolBConfig:
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5ProtocolBError("config", str(error)) from error
    if not isinstance(document, dict) or _canonical(document) != raw:
        raise Phase4D5ProtocolBError("config", "must use canonical JSON bytes")
    if require_frozen_identity:
        try:
            from rpe.runner.phase4_d5_protocol_b_authority import CONFIG_BYTES, CONFIG_SHA256
        except ImportError as error:
            raise Phase4D5ProtocolBError("frozen config identity", "trust anchor missing") from error
        if len(raw) != CONFIG_BYTES or _sha_bytes(raw) != CONFIG_SHA256:
            raise Phase4D5ProtocolBError("frozen config identity", "bytes or SHA-256 mismatch")

    synthetic = bool(document.get("synthetic_fixture", False))
    if document.get("schema_version") != "phase4-d5-protocol-b-full-domain-config-v1":
        raise Phase4D5ProtocolBError("schema_version", "mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D5ProtocolBError("experiment_id", "mismatch")
    protocol = str(document.get("protocol", ""))
    tier = str(document.get("tier", ""))
    if protocol != PROTOCOL:
        raise Phase4D5ProtocolBError("protocol", "must be B")
    if tier != TIER:
        raise Phase4D5ProtocolBError("tier", "must be full_domain_core")
    perturbations = _strings("active_perturbation_ids", document.get("active_perturbation_ids"))
    if perturbations != PERTURBATION_IDS:
        raise Phase4D5ProtocolBError("active_perturbation_ids", "must be P8-P12 only")
    alpha_raw = document.get("alpha_grid")
    if not isinstance(alpha_raw, list) or len(alpha_raw) < 2:
        raise Phase4D5ProtocolBError("alpha_grid", "must contain alpha zero and positive alphas")
    alpha_grid = tuple(_number("alpha_grid", value) for value in alpha_raw)
    if alpha_grid[0] != 0.0 or any(right <= left for left, right in zip(alpha_grid, alpha_grid[1:])):
        raise Phase4D5ProtocolBError("alpha_grid", "must start at zero and increase strictly")
    condition_ids = _condition_ids(perturbations, alpha_grid)

    artifact_files = _strings("artifact_payload_files", document.get("artifact_payload_files"))
    if artifact_files != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D5ProtocolBError("artifact_payload_files", "exact order mismatch")
    authorities_raw = _object("authorities", document.get("authorities"))
    authorities = {key: _lower_sha(f"authorities.{key}", value) for key, value in authorities_raw.items()}
    for required in ("d5_config_sha256", "sweep_sha256", "phase1_core_config_sha256", "parent_plan_sha256"):
        if required not in authorities:
            raise Phase4D5ProtocolBError("authorities", f"missing {required}")
    parent_artifacts = _object("parent_artifacts", document.get("parent_artifacts"))
    if set(parent_artifacts) != {"protocol_a", "eligibility"}:
        raise Phase4D5ProtocolBError("parent_artifacts", "must bind protocol_a and eligibility")
    for parent_name in ("protocol_a", "eligibility"):
        parent = _object(f"parent_artifacts.{parent_name}", parent_artifacts[parent_name])
        if not isinstance(parent.get("relative_path"), str) or not parent["relative_path"]:
            raise Phase4D5ProtocolBError(f"parent_artifacts.{parent_name}.relative_path", "must be nonempty")
        payloads = _object(f"parent_artifacts.{parent_name}.payload_sha256", parent.get("payload_sha256"))
        for name, value in payloads.items():
            _lower_sha(f"parent_artifacts.{parent_name}.{name}", value)

    bridge = _object("authority_bridge", document.get("authority_bridge"))
    condition_bridge_sha256 = _lower_sha("authority_bridge.condition_bridge_sha256", bridge.get("condition_bridge_sha256"))
    operator_bridge_sha256 = _lower_sha("authority_bridge.operator_bridge_sha256", bridge.get("operator_bridge_sha256"))
    metric_manifest = document.get("metric_manifest")
    if not isinstance(metric_manifest, list) or len(metric_manifest) != len(METRIC_OUTPUT_IDS):
        raise Phase4D5ProtocolBError("metric_manifest", "must contain 13 outputs")
    metric_ids: list[str] = []
    metric_directions: dict[str, PreferredDirection] = {}
    for index, raw_item in enumerate(metric_manifest):
        item = _object(f"metric_manifest[{index}]", raw_item)
        output_id = str(item.get("output_id", ""))
        if output_id not in METRIC_OUTPUT_IDS:
            raise Phase4D5ProtocolBError("metric_manifest", f"unknown output {output_id!r}")
        try:
            direction = PreferredDirection(str(item.get("preferred_direction", "")))
        except ValueError as error:
            raise Phase4D5ProtocolBError("metric_manifest", "invalid preferred direction") from error
        metric_ids.append(output_id)
        metric_directions[output_id] = direction
    if tuple(metric_ids) != METRIC_OUTPUT_IDS:
        raise Phase4D5ProtocolBError("metric_manifest", "order mismatch")

    denominators = _object("denominators", document.get("denominators"))
    full_record_count = _integer("denominators.full_record_count", denominators.get("full_record_count"), positive=True)
    group_count = _integer("denominators.group_count", denominators.get("group_count"), positive=True)
    class_count = _integer("denominators.class_count", denominators.get("class_count"), positive=True)
    query_record_count = _integer("denominators.query_record_count", denominators.get("query_record_count"), positive=True)
    query_occurrence_count = _integer("denominators.query_occurrence_count", denominators.get("query_occurrence_count"), positive=True)
    library_occurrence_count = _integer("denominators.library_occurrence_count", denominators.get("library_occurrence_count"), positive=True)
    split_count = _integer("denominators.split_count", denominators.get("split_count"), positive=True)
    expected = _object("expected", document.get("expected"))

    def expected_integer(name: str) -> int:
        return _integer(f"expected.{name}", expected.get(name), positive=True)

    values = {name: expected_integer(name) for name in (
        "operator_cell_count", "apply_check_count", "full_condition_count",
        "query_condition_count", "metric_row_count", "peak_receipt_count",
        "matcher_call_count", "prediction_row_count",
        "class_observation_count_per_metric", "class_observation_count",
        "holm_slot_count", "figure1_row_count", "figure2_row_count",
        "secondary_table_row_count",
    )}
    positive_condition_count = len(condition_ids) - 1
    derived = {
        "operator_cell_count": full_record_count * len(perturbations),
        "apply_check_count": full_record_count * len(perturbations) * len(alpha_grid),
        "full_condition_count": full_record_count * len(condition_ids),
        "query_condition_count": query_record_count * len(condition_ids),
        "metric_row_count": query_record_count * len(condition_ids) * len(METRIC_OUTPUT_IDS),
        "peak_receipt_count": query_record_count * len(condition_ids),
        "matcher_call_count": split_count * len(condition_ids),
        "prediction_row_count": query_occurrence_count * len(condition_ids),
        "class_observation_count_per_metric": class_count * positive_condition_count,
        "class_observation_count": class_count * positive_condition_count * len(METRIC_OUTPUT_IDS),
        "holm_slot_count": 24,
        "figure1_row_count": len(METRIC_OUTPUT_IDS) * positive_condition_count,
        "figure2_row_count": len(METRIC_OUTPUT_IDS),
        "secondary_table_row_count": len(METRIC_OUTPUT_IDS),
    }
    if values != derived:
        raise Phase4D5ProtocolBError("expected", f"derived counts mismatch: {derived!r}")
    frozen_identities = _object("frozen_identities", document.get("frozen_identities"))
    inherited_rulings = _object("inherited_rulings", document.get("inherited_rulings"))
    inference = _object("inference", document.get("inference"))
    bootstrap_resamples = _integer("inference.bootstrap_resamples", inference.get("bootstrap_resamples"), positive=True)
    sign_flip_resamples = _integer("inference.sign_flip_resamples", inference.get("sign_flip_resamples"), positive=True)
    random_seed = _integer("inference.random_seed", inference.get("random_seed"))
    confidence_level = _number("inference.confidence_level", inference.get("confidence_level"))
    holm_alpha = _number("inference.holm_alpha", inference.get("holm_alpha"))
    if not 0.0 < confidence_level < 1.0 or not 0.0 < holm_alpha <= 1.0:
        raise Phase4D5ProtocolBError("inference", "invalid probabilities")
    support = _object("support_grid", document.get("support_grid"))
    support_start = _number("support_grid.start_cm1", support.get("start_cm1"))
    support_stop = _number("support_grid.stop_cm1", support.get("stop_cm1"))
    support_step = _number("support_grid.step_cm1", support.get("step_cm1"))
    support_count = _integer("support_grid.point_count", support.get("point_count"), positive=True)
    support_gap = _number("support_grid.max_in_range_native_gap_cm1", support.get("max_in_range_native_gap_cm1"))
    if support_step <= 0 or int(round((support_stop - support_start) / support_step)) + 1 != support_count:
        raise Phase4D5ProtocolBError("support_grid", "point count mismatch")
    figure_contract = _object("figure_contract", document.get("figure_contract"))
    if figure_contract.get("svg_hashsalt") != "rpe-phase4-d5-protocol-b-v1":
        raise Phase4D5ProtocolBError("figure_contract", "SVG hash salt mismatch")
    claim_boundary = str(document.get("claim_boundary", ""))
    if claim_boundary != "local_execution_artifact_redistribution_not_cleared":
        raise Phase4D5ProtocolBError("claim_boundary", "mismatch")
    code_authority = _object("code_authority", document.get("code_authority"))
    environment_authority = _object("environment_authority", document.get("environment_authority"))
    trust_anchor = _object("trust_anchor", document.get("trust_anchor"))
    if trust_anchor.get("config_authority_relative_path") != CONFIG_AUTHORITY_RELATIVE_PATH:
        raise Phase4D5ProtocolBError("trust_anchor", "path mismatch")
    if not synthetic:
        if condition_bridge_sha256 != CONDITION_BRIDGE_SHA256 or operator_bridge_sha256 != OPERATOR_BRIDGE_SHA256:
            raise Phase4D5ProtocolBError("authority_bridge", "real bridge receipts mismatch")
        if (full_record_count, group_count, class_count, query_record_count, query_occurrence_count, library_occurrence_count, split_count) != (3770, 1934, 681, 3012, 6621, 12229, 5):
            raise Phase4D5ProtocolBError("denominators", "frozen real counts mismatch")
        if bootstrap_resamples != 2000 or sign_flip_resamples != 100000 or random_seed != 20260817:
            raise Phase4D5ProtocolBError("inference", "frozen real values mismatch")
        observed_code = _code_document()
        if set(code_authority) != set(CODE_RELATIVE_PATHS) or observed_code != dict(code_authority):
            raise Phase4D5ProtocolBError("code_authority", "mismatch")
        observed_environment = _environment_document()
        if observed_environment != dict(environment_authority):
            raise Phase4D5ProtocolBError("environment_authority", "mismatch")

    return Phase4D5ProtocolBConfig(
        path=Path(path), raw_bytes=raw, sha256=_sha_bytes(raw), document=MappingProxyType(document),
        synthetic_fixture=synthetic, protocol=protocol, tier=tier, perturbation_ids=perturbations,
        alpha_grid=alpha_grid, condition_ids=condition_ids, metric_output_ids=tuple(metric_ids),
        metric_directions=MappingProxyType(metric_directions), artifact_payload_files=artifact_files,
        authorities=MappingProxyType(authorities), parent_artifacts=MappingProxyType(dict(parent_artifacts)),
        frozen_identities=MappingProxyType(dict(frozen_identities)), inherited_rulings=MappingProxyType(dict(inherited_rulings)),
        claim_boundary=claim_boundary, condition_bridge_sha256=condition_bridge_sha256, operator_bridge_sha256=operator_bridge_sha256,
        full_record_count=full_record_count, group_count=group_count, class_count=class_count,
        query_record_count=query_record_count, query_occurrence_count=query_occurrence_count,
        library_occurrence_count=library_occurrence_count, split_count=split_count,
        expected_operator_cell_count=values["operator_cell_count"], expected_apply_check_count=values["apply_check_count"],
        expected_full_condition_count=values["full_condition_count"], expected_query_condition_count=values["query_condition_count"],
        expected_metric_row_count=values["metric_row_count"], expected_peak_receipt_count=values["peak_receipt_count"],
        expected_matcher_call_count=values["matcher_call_count"], expected_prediction_row_count=values["prediction_row_count"],
        expected_class_observation_count_per_metric=values["class_observation_count_per_metric"],
        expected_class_observation_count=values["class_observation_count"], expected_holm_slot_count=values["holm_slot_count"],
        expected_figure1_row_count=values["figure1_row_count"], expected_figure2_row_count=values["figure2_row_count"],
        expected_secondary_table_row_count=values["secondary_table_row_count"],
        bootstrap_resamples=bootstrap_resamples, sign_flip_resamples=sign_flip_resamples, random_seed=random_seed,
        confidence_level=confidence_level, holm_alpha=holm_alpha, support_start_cm1=support_start, support_stop_cm1=support_stop,
        support_step_cm1=support_step, support_point_count=support_count, support_max_gap_cm1=support_gap,
        figure_contract=MappingProxyType(dict(figure_contract)), code_authority=MappingProxyType(dict(code_authority)),
        environment_authority=MappingProxyType(dict(environment_authority)), trust_anchor=MappingProxyType(dict(trust_anchor)),
    )


def load_phase4_d5_protocol_b_config(path: Path) -> Phase4D5ProtocolBConfig:
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise Phase4D5ProtocolBError("config", str(error)) from error
    return parse_phase4_d5_protocol_b_config(Path(path), raw, require_frozen_identity=True)


def _load_json(path: Path, *, boundary: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5ProtocolBError(boundary, f"cannot load {path.name}: {error}") from error
    return _object(f"{boundary}.{path.name}", value)


def _load_jsonl(path: Path, *, boundary: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise Phase4D5ProtocolBError(boundary, f"{path.name}:{line_number}: {error}") from error
                rows.append(dict(_object(f"{path.name}:{line_number}", value)))
    except OSError as error:
        raise Phase4D5ProtocolBError(boundary, f"cannot open {path.name}: {error}") from error
    return rows


def _parent_payload_hashes(config: Phase4D5ProtocolBConfig, parent_name: str) -> Mapping[str, object]:
    parent = _object(f"parent_artifacts.{parent_name}", config.parent_artifacts[parent_name])
    return _object(f"parent_artifacts.{parent_name}.payload_sha256", parent.get("payload_sha256"))


def _validate_parent_file(
    path: Path,
    *,
    parent_name: str,
    config: Phase4D5ProtocolBConfig,
    boundary: str,
) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise Phase4D5ProtocolBError(boundary, f"cannot read {path.name}: {error}") from error
    expected = _parent_payload_hashes(config, parent_name).get(path.name)
    if expected is not None and _sha_bytes(raw) != expected:
        raise Phase4D5ProtocolBError(boundary, f"parent payload hash mismatch for {path.name}")
    return raw


def _condition_projection(row: Mapping[str, object], *, step9: bool) -> dict[str, object]:
    return {
        "record_id": str(row["record_id"]),
        "condition_id": str(row["condition_id"]),
        "axis_sha256": str(row["axis_sha256"]),
        "intensity_sha256": str(row["intensity_sha256"]),
        "projection_sha256": str(row["support_projection_sha256"] if step9 else row["projection_sha256"]),
    }


def _operator_projection(row: Mapping[str, object], *, step9: bool) -> dict[str, object]:
    outputs = []
    for raw_output in row["outputs"]:
        output = _object("operator output", raw_output)
        outputs.append(
            {
                "alpha": float(output["alpha"]),
                "alpha_float64_le_hex": str(output["alpha_float64_le_hex"]),
                "diagnostics": _json_ready(output["diagnostics"]),
                "output_spectrum_id": str(output["output_spectrum_id"]),
                "axis_sha256": str(output["output_axis_sha256"] if step9 else output["axis_sha256"]),
                "intensity_sha256": str(output["output_intensity_sha256"] if step9 else output["intensity_sha256"]),
            }
        )
    return {
        "record_id": str(row["record_id"]),
        "perturbation_id": str(row["perturbation_id"]),
        "state_digest": row.get("state_digest"),
        "native_gate": _json_ready(row["native_gate"]),
        "outputs": outputs,
    }


def _canonical_rows_digest(rows: Sequence[Mapping[str, object]], *, key_name: str) -> str:
    sorted_rows = sorted(rows, key=lambda row: f"{row['record_id']}|{row[key_name]}")
    return _sha_bytes(b"".join(_canonical(row) for row in sorted_rows))


def _unique_by_key(
    rows: Sequence[Mapping[str, object]],
    *,
    keys: tuple[str, str],
    boundary: str,
) -> dict[tuple[str, str], Mapping[str, object]]:
    output: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (str(row[keys[0]]), str(row[keys[1]]))
        if key in output:
            raise Phase4D5ProtocolBError(boundary, f"duplicate key {key!r}")
        output[key] = row
    return output


def _expected_roles(cohort: D5RawCohort) -> tuple[tuple[object, ...], ...]:
    rows = []
    for split in cohort.splits:
        for role, indices in (("query", split.query_indices), ("library", split.library_indices)):
            for role_order, raw_index in enumerate(indices):
                index = int(raw_index)
                rows.append(
                    (
                        int(split.seed), split.split_sha256, role, role_order, index,
                        cohort.record_ids[index], cohort.group_ids[index], int(cohort.class_labels[index]), index,
                    )
                )
    return tuple(rows)


def _observed_roles(rows: Sequence[Mapping[str, object]]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            int(row["split_seed"]), str(row["split_sha256"]), str(row["role"]),
            int(row["role_order"]), int(row["cohort_index"]), str(row["record_id"]),
            str(row["group_id"]), int(row["class_label"]), int(row["unique_record_order"]),
        )
        for row in rows
    )


def build_protocol_b_authority_bridge(
    *,
    cohort: D5RawCohort,
    protocol_a_path: Path,
    eligibility_path: Path,
    config: Phase4D5ProtocolBConfig,
) -> ProtocolBAuthorityBridge:
    protocol_a_path = Path(protocol_a_path)
    eligibility_path = Path(eligibility_path)
    if not protocol_a_path.is_dir() or not eligibility_path.is_dir():
        raise Phase4D5ProtocolBError("authority bridge", "parent path must be a directory")
    if len(cohort.record_ids) != config.full_record_count:
        raise Phase4D5ProtocolBError("cohort bridge", "record count mismatch")
    if cohort.protocol_config_sha256 != config.authorities["d5_config_sha256"]:
        raise Phase4D5ProtocolBError("cohort bridge", "D5 config identity mismatch")
    if tuple(split.split_sha256 for split in cohort.splits) != tuple(config.frozen_identities.get("split_sha256", ())):
        raise Phase4D5ProtocolBError("cohort bridge", "split identity mismatch")

    # Validate only the explicit scientific whitelist; other Protocol-A outcome files
    # may exist but must never be opened by this builder.
    for name in sorted(PROTOCOL_A_ALLOWED_FILES):
        _validate_parent_file(
            protocol_a_path / name, parent_name="protocol_a", config=config,
            boundary=("condition bridge" if name == "record_conditions.jsonl" else "operator bridge" if name == "operator_cells.jsonl" else "Protocol-A authority"),
        )
    for name in sorted(ELIGIBILITY_ALLOWED_FILES):
        _validate_parent_file(
            eligibility_path / name, parent_name="eligibility", config=config,
            boundary=("condition bridge" if name == "record_conditions.jsonl" else "operator bridge" if name == "operator_cells.jsonl" else "eligibility authority"),
        )
    step7_config = _load_json(protocol_a_path / "config.json", boundary="Protocol-A authority")
    step9_config = _load_json(eligibility_path / "config.json", boundary="eligibility authority")
    step7_manifest = _load_json(protocol_a_path / "manifest.json", boundary="Protocol-A authority")
    step9_manifest = _load_json(eligibility_path / "manifest.json", boundary="eligibility authority")
    step7_marker = _load_json(protocol_a_path / "complete.json", boundary="Protocol-A authority")
    step9_marker = _load_json(eligibility_path / "complete.json", boundary="eligibility authority")
    step7_config_protocol = step7_config.get("protocol")
    if (
        step7_config_protocol != "A"
        and not (bool(step7_config.get("synthetic_fixture", False)) and step7_config_protocol is None)
    ) or step7_manifest.get("protocol") != "A" or step7_marker.get("status") != "complete":
        raise Phase4D5ProtocolBError("Protocol-A authority", "not a complete Protocol-A parent")
    if step9_config.get("protocol") != "B" or step9_manifest.get("protocol") != "B" or step9_marker.get("status") not in {"complete", "pass"}:
        raise Phase4D5ProtocolBError("eligibility authority", "not a passing Protocol-B eligibility parent")

    unique_records = _load_jsonl(eligibility_path / "unique_records.jsonl", boundary="cohort bridge")
    role_rows = _load_jsonl(eligibility_path / "role_occurrences.jsonl", boundary="cohort bridge")
    if len(unique_records) != config.full_record_count or len(role_rows) != config.query_occurrence_count + config.library_occurrence_count:
        raise Phase4D5ProtocolBError("cohort bridge", "ledger count mismatch")
    for index, row in enumerate(unique_records):
        if (
            int(row["cohort_index"]) != index
            or str(row["record_id"]) != cohort.record_ids[index]
            or str(row["group_id"]) != cohort.group_ids[index]
            or int(row["class_label"]) != int(cohort.class_labels[index])
        ):
            raise Phase4D5ProtocolBError("cohort bridge", f"record identity mismatch at {index}")
    if _observed_roles(role_rows) != _expected_roles(cohort):
        raise Phase4D5ProtocolBError("cohort bridge", "role ledger mismatch")
    query_indices = sorted(
        {int(row["cohort_index"]) for row in role_rows if row["role"] == "query"},
        key=lambda index: cohort.record_ids[index],
    )
    query_record_ids = tuple(cohort.record_ids[index] for index in query_indices)
    configured_query_ids = config.frozen_identities.get("query_record_ids")
    if configured_query_ids is not None and tuple(configured_query_ids) != query_record_ids:
        raise Phase4D5ProtocolBError("cohort bridge", "query-union identity mismatch")
    if len(query_record_ids) != config.query_record_count:
        raise Phase4D5ProtocolBError("cohort bridge", "query record count mismatch")
    frozen_digests = {
        "record_ids_sha256": _ids_digest(list(cohort.record_ids)),
        "group_ids_sha256": _ids_digest(sorted(set(cohort.group_ids))),
        "class_labels_sha256": _class_digest([int(value) for value in cohort.class_labels]),
        "query_record_ids_sha256": _ids_digest(list(query_record_ids)),
    }
    for name, observed in frozen_digests.items():
        expected_digest = config.frozen_identities.get(name)
        if expected_digest is not None and expected_digest != observed:
            raise Phase4D5ProtocolBError("cohort bridge", f"{name} mismatch")
    query_occurrences = tuple(MappingProxyType(dict(row)) for row in role_rows if row["role"] == "query")

    step7_conditions = _load_jsonl(protocol_a_path / "record_conditions.jsonl", boundary="condition bridge")
    step9_conditions = _load_jsonl(eligibility_path / "record_conditions.jsonl", boundary="condition bridge")
    step7_by_key = _unique_by_key(step7_conditions, keys=("record_id", "condition_id"), boundary="condition bridge")
    step9_by_key = _unique_by_key(step9_conditions, keys=("record_id", "condition_id"), boundary="condition bridge")
    expected_condition_keys = {(record_id, condition_id) for record_id in query_record_ids for condition_id in config.condition_ids}
    if set(step7_by_key) != expected_condition_keys or set(step9_by_key) & expected_condition_keys != expected_condition_keys:
        raise Phase4D5ProtocolBError("condition bridge", "missing or extra query keys")
    condition_normalized: list[dict[str, object]] = []
    condition_mismatches = 0
    for key in sorted(expected_condition_keys):
        left = _condition_projection(step7_by_key[key], step9=False)
        right = _condition_projection(step9_by_key[key], step9=True)
        if left != right:
            condition_mismatches += 1
        condition_normalized.append(left)
    condition_sha = _canonical_rows_digest(condition_normalized, key_name="condition_id")
    if condition_mismatches or condition_sha != config.condition_bridge_sha256:
        raise Phase4D5ProtocolBError("condition bridge", f"mismatches={condition_mismatches}; sha256={condition_sha}")

    step7_operators = _load_jsonl(protocol_a_path / "operator_cells.jsonl", boundary="operator bridge")
    step9_operators = _load_jsonl(eligibility_path / "operator_cells.jsonl", boundary="operator bridge")
    step7_operator_by_key = _unique_by_key(step7_operators, keys=("record_id", "perturbation_id"), boundary="operator bridge")
    step9_operator_by_key = _unique_by_key(step9_operators, keys=("record_id", "perturbation_id"), boundary="operator bridge")
    expected_operator_keys = {(record_id, perturbation_id) for record_id in query_record_ids for perturbation_id in config.perturbation_ids}
    if set(step7_operator_by_key) != expected_operator_keys or set(step9_operator_by_key) & expected_operator_keys != expected_operator_keys:
        raise Phase4D5ProtocolBError("operator bridge", "missing or extra query cells")
    operator_normalized: list[dict[str, object]] = []
    operator_mismatches = 0
    for key in sorted(expected_operator_keys):
        left = _operator_projection(step7_operator_by_key[key], step9=False)
        right_source = step9_operator_by_key[key]
        if right_source.get("state") != "complete":
            operator_mismatches += 1
        right = _operator_projection(right_source, step9=True)
        if left != right:
            operator_mismatches += 1
        operator_normalized.append(left)
    operator_sha = _canonical_rows_digest(operator_normalized, key_name="perturbation_id")
    if operator_mismatches or operator_sha != config.operator_bridge_sha256:
        raise Phase4D5ProtocolBError("operator bridge", f"mismatches={operator_mismatches}; sha256={operator_sha}")

    metric_rows = _load_jsonl(protocol_a_path / "metric_values.jsonl", boundary="metric authority")
    if len(metric_rows) != config.expected_metric_row_count:
        raise Phase4D5ProtocolBError("metric authority", "row count mismatch")
    metric_values: dict[tuple[str, str, str], float] = {}
    for row in metric_rows:
        key = (str(row["record_id"]), str(row["condition_id"]), str(row["metric_output_id"]))
        if key in metric_values or key[0] not in query_record_ids or key[1] not in config.condition_ids or key[2] not in config.metric_output_ids:
            raise Phase4D5ProtocolBError("metric authority", f"invalid or duplicate key {key!r}")
        if row.get("state") != "complete":
            raise Phase4D5ProtocolBError("metric authority", f"incomplete row {key!r}")
        value = _number("metric authority value", row.get("value"))
        metric_values[key] = value
    peak_rows = _load_jsonl(protocol_a_path / "peak_receipts.jsonl", boundary="CWT authority")
    if len(peak_rows) != config.expected_peak_receipt_count:
        raise Phase4D5ProtocolBError("CWT authority", "row count mismatch")
    peak_keys: set[tuple[str, str]] = set()
    for row in peak_rows:
        key = (str(row["record_id"]), str(row["condition_id"]))
        if key in peak_keys or key[0] not in query_record_ids or key[1] not in config.condition_ids:
            raise Phase4D5ProtocolBError("CWT authority", f"invalid or duplicate key {key!r}")
        if row.get("status") not in {"complete", "complete_with_warning"}:
            raise Phase4D5ProtocolBError("CWT authority", f"failed receipt {key!r}")
        peak_keys.add(key)

    document = {
        "schema_version": "phase4-d5-protocol-b-authority-bridge-v1",
        "cohort_bridge": {
            "record_count": len(cohort.record_ids),
            "group_count": len(set(cohort.group_ids)),
            "class_count": len(set(int(value) for value in cohort.class_labels)),
            "class_labels": [int(value) for value in cohort.class_labels],
            "query_record_count": len(query_record_ids),
            "query_occurrence_count": len(query_occurrences),
            "library_occurrence_count": len(role_rows) - len(query_occurrences),
            "split_sha256": [split.split_sha256 for split in cohort.splits],
            "state": "complete",
        },
        "condition_bridge": {
            "row_count": len(condition_normalized),
            "mismatch_count": condition_mismatches,
            "sha256": condition_sha,
            "state": "complete",
        },
        "operator_bridge": {
            "cell_count": len(operator_normalized),
            "output_count": sum(len(row["outputs"]) for row in operator_normalized),
            "mismatch_count": operator_mismatches,
            "sha256": operator_sha,
            "state": "complete",
        },
        "metric_authority": {
            "metric_row_count": len(metric_values),
            "peak_receipt_count": len(peak_rows),
            "query_only": True,
            "state": "complete",
        },
        "protocol_a_parent_sha256sums_sha256": _sha_bytes((protocol_a_path / "SHA256SUMS").read_bytes()),
        "eligibility_parent_sha256sums_sha256": _sha_bytes((eligibility_path / "SHA256SUMS").read_bytes()),
    }
    return ProtocolBAuthorityBridge(
        document=MappingProxyType(document),
        metric_values=MappingProxyType(metric_values),
        peak_receipts=tuple(MappingProxyType(dict(row)) for row in peak_rows),
        query_record_ids=query_record_ids,
        query_occurrences=query_occurrences,
        protocol_a_conditions=tuple(MappingProxyType(dict(row)) for row in step7_conditions),
        eligibility_conditions=tuple(MappingProxyType(dict(row)) for row in step9_conditions),
        eligibility_operator_cells=tuple(MappingProxyType(dict(row)) for row in step9_operators),
    )


def match_d5_protocol_b_799(
    cohort: D5RawCohort,
    split: D5LibraryQuerySplit,
    *,
    query_condition_id: str,
    library_condition_id: str,
    query_record_ids: tuple[str, ...],
    library_record_ids: tuple[str, ...],
    query_values: np.ndarray,
    library_values: np.ndarray,
) -> D5MatchingResult:
    if query_condition_id != library_condition_id:
        raise Phase4D5ProtocolBError(
            "same condition matcher",
            "query and library must use the same condition",
        )
    query = np.asarray(query_values)
    library = np.asarray(library_values)
    if (
        query.ndim != 2
        or library.ndim != 2
        or query.shape[1] != 799
        or library.shape[1] != 799
    ):
        raise Phase4D5ProtocolBError("799 matcher", "must use exactly 799 features")
    return match_d5_protocol_a_values(
        cohort,
        split,
        condition_id=query_condition_id,
        query_record_ids=query_record_ids,
        library_record_ids=library_record_ids,
        query_values=query,
        library_values=library,
    )


def aggregate_protocol_b_class_observations(
    prediction_rows: Sequence[Mapping[str, object]],
    metric_values: Mapping[tuple[str, str, str], float],
    *,
    metric_output_id: str,
    preferred_direction: str,
    positive_conditions: Sequence[str],
) -> list[dict[str, object]]:
    try:
        direction = PreferredDirection(preferred_direction)
    except ValueError as error:
        raise Phase4D5ProtocolBError("preferred_direction", "invalid value") from error
    by_class_condition: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        by_class_condition[(int(row["class_label"]), str(row["condition_id"]))].append(row)
    classes = sorted({key[0] for key in by_class_condition})
    output: list[dict[str, object]] = []
    for class_label in classes:
        baseline = by_class_condition.get((class_label, "alpha0"), [])
        if not baseline:
            raise Phase4D5ProtocolBError("class observations", f"class {class_label} lacks alpha zero")
        baseline_error = float(np.mean([not bool(row["top1_correct"]) for row in baseline]))
        for condition_id in positive_conditions:
            rows = by_class_condition.get((class_label, str(condition_id)), [])
            if not rows:
                raise Phase4D5ProtocolBError("class observations", f"class {class_label} lacks {condition_id}")
            downstream_error = float(np.mean([not bool(row["top1_correct"]) for row in rows]))
            harms: list[float] = []
            for row in rows:
                record_id = str(row["record_id"])
                baseline_key = (record_id, "alpha0", metric_output_id)
                condition_key = (record_id, str(condition_id), metric_output_id)
                if baseline_key not in metric_values or condition_key not in metric_values:
                    raise Phase4D5ProtocolBError("class observations", f"missing query metric {condition_key!r}")
                harms.append(
                    orient_harm(
                        float(metric_values[baseline_key]),
                        float(metric_values[condition_key]),
                        direction,
                    )
                )
            perturbation_id, alpha_hex = str(condition_id).split(":", 1)
            output.append(
                {
                    "cluster_id": str(class_label),
                    "perturbation_id": perturbation_id,
                    "alpha": struct.unpack("<d", bytes.fromhex(alpha_hex))[0],
                    "metric_output_id": metric_output_id,
                    "metric_harm": float(np.mean(harms)),
                    "downstream_harm": downstream_error - baseline_error,
                    "occurrence_count": len(rows),
                    "state": "complete",
                }
            )
    return output


def fixed_protocol_b_holm_family(
    raw_p_values: Mapping[str, float],
    observed_contrasts: Mapping[str, float],
) -> list[dict[str, object]]:
    expected_ids = {
        f"{metric_id}:{statistic}"
        for metric_id in CANDIDATE_OUTPUT_IDS
        for statistic in ("d_ag", "d_acc")
    }
    if set(raw_p_values) != expected_ids or set(observed_contrasts) != expected_ids:
        raise Phase4D5ProtocolBError("Holm family", "must contain the separate 24 Protocol-B slots")
    adjusted = {row.hypothesis_id: row for row in holm_step_down(raw_p_values, alpha=0.05)}
    output = []
    for hypothesis_id in sorted(expected_ids):
        contrast = float(observed_contrasts[hypothesis_id])
        if not math.isfinite(contrast):
            raise Phase4D5ProtocolBError("Holm family", f"nonfinite contrast {hypothesis_id}")
        row = adjusted[hypothesis_id]
        metric_id, statistic = hypothesis_id.rsplit(":", 1)
        favorable = contrast > 0.0
        output.append(
            {
                "family_id": "d5_protocol_b_full_domain_secondary_24",
                "metric_output_id": metric_id,
                "statistic": statistic,
                "hypothesis_id": hypothesis_id,
                "raw_p_value": float(raw_p_values[hypothesis_id]),
                "adjusted_p_value": row.adjusted_p_value,
                "rank": row.rank,
                "family_size": row.family_size,
                "observed_contrast": contrast,
                "favorable": favorable,
                "rejected": bool(row.rejected and favorable),
            }
        )
    return output


def _padding(values: Sequence[float]) -> tuple[float, float]:
    minimum = min(0.0, *values)
    maximum = max(0.0, *values)
    width = maximum - minimum
    pad = 0.05 * (width if width > 0.0 else max(1.0, abs(minimum), abs(maximum)))
    return minimum - pad, maximum + pad


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return stream.getvalue().encode("utf-8")


def render_protocol_b_figure_payloads(
    figure1_rows: Sequence[Mapping[str, object]],
    figure2_rows: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    if len(figure1_rows) != 520 or len(figure2_rows) != 13:
        raise Phase4D5ProtocolBError("figure source", "must contain 520 and 13 rows")
    f1_csv = _csv_bytes(figure1_rows, tuple(figure1_rows[0]))
    f2_csv = _csv_bytes(figure2_rows, tuple(figure2_rows[0]))
    colors = dict(zip(PERTURBATION_IDS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True))
    rc = {
        "font.family": "DejaVu Sans",
        "svg.hashsalt": "rpe-phase4-d5-protocol-b-v1",
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharey=True)
        y_values = [float(row["mean_downstream_harm"]) for row in figure1_rows if row.get("mean_downstream_harm") is not None]
        if not y_values:
            raise Phase4D5ProtocolBError("figure1", "requires evaluable rows")
        y_lim = _padding(y_values)
        for metric_index, metric_id in enumerate(METRIC_OUTPUT_IDS):
            ax = axes.flat[metric_index]
            selected_metric = [row for row in figure1_rows if row["metric_output_id"] == metric_id]
            for perturbation_id in PERTURBATION_IDS:
                selected = [row for row in selected_metric if row["perturbation_id"] == perturbation_id]
                if selected and selected[0].get("metric_state") == "complete":
                    ax.plot(
                        [float(row["mean_metric_harm"]) for row in selected],
                        [float(row["mean_downstream_harm"]) for row in selected],
                        marker="o", linewidth=1.5, color=colors[perturbation_id], label=perturbation_id.upper(),
                    )
            x_values = [float(row["mean_metric_harm"]) for row in selected_metric if row.get("mean_metric_harm") is not None]
            if x_values:
                ax.set_xlim(*_padding(x_values))
            ax.set_ylim(*y_lim)
            ax.set_title(metric_id)
            ax.axhline(0.0, color="#999999", linewidth=0.5)
            ax.axvline(0.0, color="#999999", linewidth=0.5)
        for index in range(len(METRIC_OUTPUT_IDS), 16):
            axes.flat[index].axis("off")
        axes.flat[0].legend(loc="best", fontsize=7)
        fig.suptitle("D5 / matched-reference Protocol B / full_domain_core / P8–P12")
        fig.tight_layout()
        f1_png = io.BytesIO(); f1_svg = io.BytesIO()
        fig.savefig(f1_png, format="png", dpi=300, metadata={"Software": "raman-preproc-eval"})
        fig.savefig(f1_svg, format="svg", metadata={"Date": None})
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        specs = (("ag", "AG"), ("acc", "Acc-cross"), ("d_ag", "D_AG"), ("d_acc", "D_Acc"))
        y = np.arange(len(METRIC_OUTPUT_IDS))
        for ax, (key, title) in zip(axes, specs, strict=True):
            finite = []
            for index, row in enumerate(figure2_rows):
                value = row.get(key)
                if value is None:
                    continue
                value = float(value); low = float(row[f"{key}_lower"]); high = float(row[f"{key}_upper"])
                finite.extend((low, value, high))
                ax.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color="#1f77b4")
                if key in {"d_ag", "d_acc"}:
                    direction = "favorable" if bool(row[f"{key}_favorable"]) else "unfavorable"
                    label = f"raw={float(row[f'{key}_raw_p']):.6g}; adj={float(row[f'{key}_adjusted_p']):.6g}; {direction}"
                    ax.text(0.02, index, label, transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=4.5)
            if finite:
                ax.set_xlim(*_padding(finite))
            ax.axvline(0.0, color="#999999", linewidth=0.5)
            ax.set_title(title)
        axes[0].set_yticks(y, METRIC_OUTPUT_IDS)
        fig.suptitle("D5 / matched-reference Protocol B / full_domain_core / secondary")
        fig.tight_layout()
        f2_png = io.BytesIO(); f2_svg = io.BytesIO()
        fig.savefig(f2_png, format="png", dpi=300, metadata={"Software": "raman-preproc-eval"})
        fig.savefig(f2_svg, format="svg", metadata={"Date": None})
        plt.close(fig)
    return {
        "figure1_d5_protocol_b_full_domain.png": f1_png.getvalue(),
        "figure1_d5_protocol_b_full_domain.svg": f1_svg.getvalue(),
        "figure1_d5_protocol_b_full_domain_data.csv": f1_csv,
        "figure2_d5_protocol_b_full_domain.png": f2_png.getvalue(),
        "figure2_d5_protocol_b_full_domain.svg": f2_svg.getvalue(),
        "figure2_d5_protocol_b_full_domain_data.csv": f2_csv,
    }


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha_bytes(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _support_grid(config: Phase4D5ProtocolBConfig) -> np.ndarray:
    grid = np.arange(
        config.support_start_cm1,
        config.support_stop_cm1 + config.support_step_cm1 / 2.0,
        config.support_step_cm1,
        dtype="<f8",
    )
    if grid.size != config.support_point_count:
        raise Phase4D5ProtocolBError("support grid", "point count mismatch")
    return grid


def _project_values(
    spectrum: Spectrum1D,
    grid: np.ndarray,
    max_gap_cm1: float,
) -> tuple[np.ndarray, float]:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5ProtocolBError("support projection", "extrapolation required")
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    if left < 0 or right >= axis.size:
        raise Phase4D5ProtocolBError("support projection", "invalid bounds")
    support = axis[left : right + 1]
    max_gap = float(np.max(np.diff(support))) if support.size > 1 else math.inf
    if support.size < 2 or max_gap > max_gap_cm1:
        raise Phase4D5ProtocolBError("support projection", "native gap exceeds maximum")
    values = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    norm = float(np.linalg.norm(values.astype(np.float64)))
    if not np.isfinite(values).all() or not math.isfinite(norm) or norm <= 0.0:
        raise Phase4D5ProtocolBError("support projection", "nonfinite or zero norm")
    return values, max_gap


def _source_record(
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    index: int,
) -> dict[str, object]:
    spectrum = native_spectra[index]
    return {
        "class_label": int(cohort.class_labels[index]),
        "cohort_index": index,
        "group_id": cohort.group_ids[index],
        "mineral_name": cohort.mineral_names[index],
        "native_axis_sha256": _array_sha(spectrum.axis_cm1),
        "native_intensity_sha256": _array_sha(spectrum.intensity),
        "point_count": int(spectrum.axis_cm1.size),
        "record_id": cohort.record_ids[index],
        "record_order": index,
        "rruff_id": cohort.rruff_ids[index],
    }


def _phase1_source(record: Mapping[str, object], spectrum: Spectrum1D) -> Phase1Source:
    axis_f4 = np.ascontiguousarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.ascontiguousarray(spectrum.intensity, dtype="<f4")
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=int(record["record_order"]),
            record_id=str(record["record_id"]),
            sample_id=spectrum.sample_id or str(record["rruff_id"]),
            class_label=int(record["class_label"]),
            mineral_name=str(record["mineral_name"]),
            axis_id=f"native::{record['native_axis_sha256']}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, "<f4"),
        source_intensity_float32_sha256=_array_sha(intensity_f4, "<f4"),
        normalized_axis_float64_sha256=str(record["native_axis_sha256"]),
        normalized_intensity_float64_sha256=str(record["native_intensity_sha256"]),
        provenance=MappingProxyType(
            {
                "d5_protocol_config_sha256": "",
                "dataset_id": "rruff_raman_raw",
                "record_id": str(record["record_id"]),
            }
        ),
    )


def _rematerialize_record(
    record: Mapping[str, object],
    spectrum: Spectrum1D,
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D5ProtocolBConfig,
    grid: np.ndarray,
    admission: P10MemoryAdmission,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    source = _phase1_source(record, spectrum)
    record_id = str(record["record_id"])
    alpha0_values, alpha0_gap = _project_values(spectrum, grid, config.support_max_gap_cm1)
    projections: dict[str, np.ndarray] = {"alpha0": alpha0_values}
    conditions: list[dict[str, object]] = [
        {
            "alpha": 0.0,
            "alpha_float64_le_hex": np.float64(0.0).tobytes().hex(),
            "axis_sha256": str(record["native_axis_sha256"]),
            "class_label": int(record["class_label"]),
            "condition_id": "alpha0",
            "condition_kind": "alpha0",
            "group_id": str(record["group_id"]),
            "intensity_sha256": str(record["native_intensity_sha256"]),
            "perturbation_id": None,
            "record_id": record_id,
            "record_order": int(record["record_order"]),
            "state": "complete",
            "support_max_in_range_gap_cm1": alpha0_gap,
            "support_point_count": int(alpha0_values.size),
            "support_projection_sha256": _array_sha(alpha0_values, "<f4"),
        }
    ]
    cells: list[dict[str, object]] = []
    alpha0_hashes: list[tuple[str, str]] = []
    for perturbation_id in config.perturbation_ids:
        cell = run_perturbation_cell(
            source,
            perturbation_id,
            phase1_config,
            sweep,
            p10_admission=admission,
        )
        if cell.status is not CellStatus.COMPLETE:
            raise Phase4D5ProtocolBError(
                "rematerialization", f"{record_id}/{perturbation_id} is not complete"
            )
        outputs: list[dict[str, object]] = []
        for perturbed in cell.records:
            result = perturbed.result
            output = result.output
            values, max_gap = _project_values(output, grid, config.support_max_gap_cm1)
            axis_sha = _array_sha(output.axis_cm1)
            intensity_sha = _array_sha(output.intensity)
            outputs.append(
                {
                    "alpha": float(result.alpha),
                    "alpha_float64_le_hex": perturbed.alpha_float64_le_hex,
                    "axis_changed": bool(result.axis_changed),
                    "diagnostics": _json_ready(result.diagnostics),
                    "intensity_changed": bool(result.intensity_changed),
                    "output_axis_sha256": axis_sha,
                    "output_intensity_sha256": intensity_sha,
                    "output_spectrum_id": output.spectrum_id,
                    "support_max_in_range_gap_cm1": max_gap,
                    "support_point_count": int(values.size),
                    "support_projection_sha256": _array_sha(values, "<f4"),
                }
            )
            if result.alpha == 0.0:
                alpha0_hashes.append((axis_sha, intensity_sha))
                continue
            condition_id = f"{perturbation_id}:{perturbed.alpha_float64_le_hex}"
            projections[condition_id] = values
            conditions.append(
                {
                    "alpha": float(result.alpha),
                    "alpha_float64_le_hex": perturbed.alpha_float64_le_hex,
                    "axis_sha256": axis_sha,
                    "class_label": int(record["class_label"]),
                    "condition_id": condition_id,
                    "condition_kind": "positive",
                    "group_id": str(record["group_id"]),
                    "intensity_sha256": intensity_sha,
                    "perturbation_id": perturbation_id,
                    "record_id": record_id,
                    "record_order": int(record["record_order"]),
                    "state": "complete",
                    "support_max_in_range_gap_cm1": max_gap,
                    "support_point_count": int(values.size),
                    "support_projection_sha256": _array_sha(values, "<f4"),
                }
            )
        cells.append(
            {
                "class_label": int(record["class_label"]),
                "exception": None,
                "group_id": str(record["group_id"]),
                "native_gate": _json_ready(cell.evidence.native_gate),
                "output_count": len(outputs),
                "outputs": outputs,
                "p10_estimated_peak_bytes": (
                    estimate_p10_peak_bytes(int(record["point_count"]))
                    if perturbation_id == "p10"
                    else None
                ),
                "perturbation_id": perturbation_id,
                "reason_code": None,
                "record_id": record_id,
                "state": "complete",
                "state_digest": None if cell.state is None else cell.state.state_digest,
            }
        )
    source_hash = (str(record["native_axis_sha256"]), str(record["native_intensity_sha256"]))
    if len(alpha0_hashes) != len(config.perturbation_ids) or any(value != source_hash for value in alpha0_hashes):
        raise Phase4D5ProtocolBError("alpha-zero collapse", "operators do not preserve source bytes")
    return cells, conditions, projections


def _rematerialize_full_cohort(
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D5ProtocolBConfig,
    worker_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    if len(cohort.record_ids) != config.full_record_count or len(native_spectra) != config.full_record_count:
        raise Phase4D5ProtocolBError("rematerialization", "cohort count mismatch")
    expected_spectrum_ids = tuple(f"rruff_raman_raw::{record_id}" for record_id in cohort.record_ids)
    if tuple(value.spectrum_id for value in native_spectra) != expected_spectrum_ids:
        raise Phase4D5ProtocolBError("rematerialization", "native spectrum order mismatch")
    if tuple(sweep.alpha_grid) != config.alpha_grid:
        raise Phase4D5ProtocolBError("rematerialization", "alpha grid mismatch")
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D5ProtocolBError("worker_count", "must be positive")
    records = [_source_record(cohort, native_spectra, index) for index in range(config.full_record_count)]
    estimates = [estimate_p10_peak_bytes(int(record["point_count"])) for record in records]
    budget = 64 * 2**30
    if max(estimates) > budget:
        raise Phase4D5ProtocolBError("P10 admission", "record exceeds 64 GiB")
    admission = P10MemoryAdmission(budget)
    grid = _support_grid(config)
    results: dict[int, tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]] = {}
    with threadpool_limits(limits=1, user_api="blas"):
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                index: executor.submit(
                    _rematerialize_record,
                    records[index],
                    native_spectra[index],
                    sweep,
                    phase1_config,
                    config,
                    grid,
                    admission,
                )
                for index in range(config.full_record_count)
            }
            for index, future in futures.items():
                results[index] = future.result()
    cells: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    projected: dict[tuple[str, str], np.ndarray] = {}
    for index in range(config.full_record_count):
        current_cells, current_conditions, current_projected = results[index]
        cells.extend(current_cells)
        conditions.extend(current_conditions)
        record_id = cohort.record_ids[index]
        projected.update(
            {
                (record_id, condition_id): values
                for condition_id, values in current_projected.items()
            }
        )
    if len(cells) != config.expected_operator_cell_count or len(conditions) != config.expected_full_condition_count:
        raise Phase4D5ProtocolBError("rematerialization", "output count mismatch")
    return cells, conditions, projected


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _condition_summaries(
    prediction_rows: Sequence[Mapping[str, object]],
    condition_ids: Sequence[str],
) -> list[dict[str, object]]:
    output = []
    for condition_id in condition_ids:
        selected = [row for row in prediction_rows if row["condition_id"] == condition_id]
        if not selected:
            raise Phase4D5ProtocolBError("condition summary", f"missing {condition_id}")
        labels = sorted({int(row["class_label"]) for row in selected})
        macro1 = float(
            np.mean(
                [
                    np.mean([bool(row["top1_correct"]) for row in selected if int(row["class_label"]) == label])
                    for label in labels
                ]
            )
        )
        macro5 = float(
            np.mean(
                [
                    np.mean([bool(row["top5_correct"]) for row in selected if int(row["class_label"]) == label])
                    for label in labels
                ]
            )
        )
        output.append(
            {
                "condition_id": condition_id,
                "query_occurrence_count": len(selected),
                "top1_macro_class_accuracy": macro1,
                "top5_macro_class_accuracy": macro5,
                "top1_micro_accuracy": float(np.mean([bool(row["top1_correct"]) for row in selected])),
                "top5_micro_accuracy": float(np.mean([bool(row["top5_correct"]) for row in selected])),
            }
        )
    return output


def _terminal_holm_family(reason: str) -> list[dict[str, object]]:
    raw = {
        f"{metric}:{statistic}": 1.0
        for metric in CANDIDATE_OUTPUT_IDS
        for statistic in ("d_ag", "d_acc")
    }
    adjusted = {row.hypothesis_id: row for row in holm_step_down(raw, alpha=0.05)}
    return [
        {
            "family_id": "d5_protocol_b_full_domain_secondary_24",
            "metric_output_id": hypothesis.rsplit(":", 1)[0],
            "statistic": hypothesis.rsplit(":", 1)[1],
            "hypothesis_id": hypothesis,
            "state": reason,
            "raw_p_value": None,
            "multiplicity_p_value": 1.0,
            "adjusted_p_value": adjusted[hypothesis].adjusted_p_value,
            "rank": adjusted[hypothesis].rank,
            "family_size": adjusted[hypothesis].family_size,
            "observed_contrast": None,
            "favorable": None,
            "rejected": False,
        }
        for hypothesis in sorted(raw)
    ]


def _statistics_payloads(
    class_rows: Sequence[Mapping[str, object]],
    config: Phase4D5ProtocolBConfig,
    *,
    inference_resamples: int | None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    by_metric: dict[str, list[AlignmentObservation]] = defaultdict(list)
    for row in class_rows:
        by_metric[str(row["metric_output_id"])].append(
            AlignmentObservation(
                str(row["cluster_id"]),
                str(row["perturbation_id"]),
                float(row["alpha"]),
                float(row["metric_harm"]),
                float(row["downstream_harm"]),
            )
        )
    if tuple(by_metric) != METRIC_OUTPUT_IDS:
        raise Phase4D5ProtocolBError("alignment", "metric order or completeness mismatch")
    reference = tuple(by_metric["mse"])
    effective_bootstrap = config.bootstrap_resamples if inference_resamples is None else inference_resamples
    effective_sign_flips = config.sign_flip_resamples if inference_resamples is None else max(64, inference_resamples)
    if not config.synthetic_fixture and inference_resamples is not None:
        raise Phase4D5ProtocolBError("inference_resamples", "override forbidden for real config")
    metric_results: dict[str, dict[str, object]] = {}
    bootstrap_rows: list[dict[str, object]] = []
    sign_rows: list[dict[str, object]] = []
    raw_p_values: dict[str, float] = {}
    contrasts: dict[str, float] = {}
    downstream_constant = False
    try:
        reference_gap = alignment_gap(reference)
        reference_acc = cross_perturbation_accuracy(reference)
        mse_bootstrap = bulk_paired_cluster_bootstrap(
            reference,
            reference,
            resamples=effective_bootstrap,
            confidence_level=config.confidence_level,
            random_seed=config.random_seed,
        )
        metric_results["mse"] = {
            "metric_output_id": "mse",
            "metric_state": "complete",
            "ag": reference_gap.alignment_gap,
            "acc": reference_acc.accuracy,
            "comparison": None,
        }
    except AlignmentValidationError as error:
        if error.path != "constant downstream":
            raise
        downstream_constant = True
        reference_acc = None
        mse_bootstrap = None
        metric_results["mse"] = {
            "metric_output_id": "mse",
            "metric_state": "not_evaluable_constant_downstream",
            "ag": None,
            "acc": None,
            "comparison": None,
        }
    for metric_id in CANDIDATE_OUTPUT_IDS:
        if downstream_constant:
            metric_results[metric_id] = {
                "metric_output_id": metric_id,
                "metric_state": "not_evaluable_constant_downstream",
                "ag": None,
                "acc": None,
                "comparison": None,
            }
            bootstrap_rows.append({"metric_output_id": metric_id, "state": "not_evaluable_constant_downstream"})
            sign_rows.extend(
                {"metric_output_id": metric_id, "statistic": statistic, "state": "not_evaluable_constant_downstream"}
                for statistic in ("d_ag", "d_acc")
            )
            continue
        candidate = tuple(by_metric[metric_id])
        comparison = compare_alignment(reference, candidate)
        bootstrap = bulk_paired_cluster_bootstrap(
            reference,
            candidate,
            resamples=effective_bootstrap,
            confidence_level=config.confidence_level,
            random_seed=config.random_seed,
        )
        ag_flip = paired_contribution_sign_flip(
            tuple(value.value for value in comparison.ag_contribution_differences),
            aggregation="sum",
            resamples=effective_sign_flips,
            random_seed=config.random_seed,
        )
        acc_flip = paired_contribution_sign_flip(
            tuple(value.value for value in comparison.acc_contribution_differences),
            aggregation="mean",
            resamples=effective_sign_flips,
            random_seed=config.random_seed,
        )
        raw_p_values[f"{metric_id}:d_ag"] = ag_flip.p_value
        raw_p_values[f"{metric_id}:d_acc"] = acc_flip.p_value
        contrasts[f"{metric_id}:d_ag"] = comparison.d_ag
        contrasts[f"{metric_id}:d_acc"] = comparison.d_acc
        metric_results[metric_id] = {
            "metric_output_id": metric_id,
            "metric_state": "complete",
            "ag": comparison.candidate_gap.alignment_gap,
            "acc": comparison.candidate_accuracy.accuracy,
            "comparison": _json_ready(comparison),
        }
        bootstrap_rows.append({"metric_output_id": metric_id, "state": "complete", **_json_ready(bootstrap)})
        sign_rows.extend(
            (
                {"metric_output_id": metric_id, "statistic": "d_ag", "state": "complete", **_json_ready(ag_flip)},
                {"metric_output_id": metric_id, "statistic": "d_acc", "state": "complete", **_json_ready(acc_flip)},
            )
        )
    family = (
        _terminal_holm_family("not_tested_constant_downstream")
        if downstream_constant
        else fixed_protocol_b_holm_family(raw_p_values, contrasts)
    )
    family_by_key = {(row["metric_output_id"], row["statistic"]): row for row in family}
    bootstrap_by_metric = {row["metric_output_id"]: row for row in bootstrap_rows}
    figure2_rows: list[dict[str, object]] = []
    for metric_id in METRIC_OUTPUT_IDS:
        result = metric_results[metric_id]
        if metric_id == "mse" and mse_bootstrap is not None and reference_acc is not None:
            figure2_rows.append(
                {
                    "metric_output_id": metric_id, "metric_state": "complete",
                    "ag": result["ag"], "ag_lower": mse_bootstrap.reference_ag_interval[0], "ag_upper": mse_bootstrap.reference_ag_interval[1],
                    "acc": result["acc"], "acc_lower": mse_bootstrap.reference_acc_interval[0], "acc_upper": mse_bootstrap.reference_acc_interval[1],
                    "pair_count": reference_acc.pair_count, "strict_agreement_count": reference_acc.strict_agreement_count,
                    "strict_disagreement_count": reference_acc.strict_disagreement_count, "metric_tie_count": reference_acc.metric_tie_count,
                    "downstream_tie_count": reference_acc.downstream_tie_count, "double_tie_count": reference_acc.double_tie_count,
                    "d_ag": None, "d_ag_lower": None, "d_ag_upper": None, "d_ag_favorable": None,
                    "d_ag_raw_p": None, "d_ag_adjusted_p": None, "d_acc": None, "d_acc_lower": None,
                    "d_acc_upper": None, "d_acc_favorable": None, "d_acc_raw_p": None, "d_acc_adjusted_p": None,
                }
            )
        elif result["metric_state"] == "complete":
            boot = bootstrap_by_metric[metric_id]
            comparison = result["comparison"]
            ag_family = family_by_key[(metric_id, "d_ag")]
            acc_family = family_by_key[(metric_id, "d_acc")]
            candidate_acc = comparison["candidate_accuracy"]
            figure2_rows.append(
                {
                    "metric_output_id": metric_id, "metric_state": "complete",
                    "ag": result["ag"], "ag_lower": boot["candidate_ag_interval"][0], "ag_upper": boot["candidate_ag_interval"][1],
                    "acc": result["acc"], "acc_lower": boot["candidate_acc_interval"][0], "acc_upper": boot["candidate_acc_interval"][1],
                    "pair_count": candidate_acc["pair_count"], "strict_agreement_count": candidate_acc["strict_agreement_count"],
                    "strict_disagreement_count": candidate_acc["strict_disagreement_count"], "metric_tie_count": candidate_acc["metric_tie_count"],
                    "downstream_tie_count": candidate_acc["downstream_tie_count"], "double_tie_count": candidate_acc["double_tie_count"],
                    "d_ag": comparison["d_ag"], "d_ag_lower": boot["d_ag_interval"][0], "d_ag_upper": boot["d_ag_interval"][1],
                    "d_ag_favorable": ag_family["favorable"], "d_ag_raw_p": ag_family["raw_p_value"], "d_ag_adjusted_p": ag_family["adjusted_p_value"],
                    "d_acc": comparison["d_acc"], "d_acc_lower": boot["d_acc_interval"][0], "d_acc_upper": boot["d_acc_interval"][1],
                    "d_acc_favorable": acc_family["favorable"], "d_acc_raw_p": acc_family["raw_p_value"], "d_acc_adjusted_p": acc_family["adjusted_p_value"],
                }
            )
        else:
            figure2_rows.append(
                {
                    "metric_output_id": metric_id, "metric_state": result["metric_state"],
                    "ag": None, "ag_lower": None, "ag_upper": None, "acc": None, "acc_lower": None, "acc_upper": None,
                    "pair_count": None, "strict_agreement_count": None, "strict_disagreement_count": None,
                    "metric_tie_count": None, "downstream_tie_count": None, "double_tie_count": None,
                    "d_ag": None, "d_ag_lower": None, "d_ag_upper": None, "d_ag_favorable": None,
                    "d_ag_raw_p": None, "d_ag_adjusted_p": 1.0, "d_acc": None, "d_acc_lower": None,
                    "d_acc_upper": None, "d_acc_favorable": None, "d_acc_raw_p": None, "d_acc_adjusted_p": 1.0,
                }
            )
    alignment_rows = [metric_results[metric_id] for metric_id in METRIC_OUTPUT_IDS]
    figure1_rows: list[dict[str, object]] = []
    for metric_id in METRIC_OUTPUT_IDS:
        current = [row for row in class_rows if row["metric_output_id"] == metric_id]
        for perturbation_id in config.perturbation_ids:
            for alpha in config.alpha_grid[1:]:
                selected = [row for row in current if row["perturbation_id"] == perturbation_id and row["alpha"] == alpha]
                figure1_rows.append(
                    {
                        "metric_output_id": metric_id, "perturbation_id": perturbation_id, "alpha": alpha,
                        "mean_metric_harm": float(np.mean([row["metric_harm"] for row in selected])),
                        "mean_downstream_harm": float(np.mean([row["downstream_harm"] for row in selected])),
                        "metric_state": "complete",
                    }
                )
    return alignment_rows, bootstrap_rows, sign_rows, family, figure1_rows, figure2_rows


def _run_identity(
    config: Phase4D5ProtocolBConfig,
    code: Mapping[str, object],
    environment: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    identity = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "authorities": dict(config.authorities),
        "parent_artifacts": _json_ready(config.parent_artifacts),
        "authority_bridge": {
            "condition_bridge_sha256": config.condition_bridge_sha256,
            "operator_bridge_sha256": config.operator_bridge_sha256,
        },
        "claim_boundary": config.claim_boundary,
        "code": _json_ready(code),
        "config_authority": _authority_document(),
        "config_sha256": config.sha256,
        "environment": _json_ready(environment),
        "figure_contract": _json_ready(config.figure_contract),
        "frozen_identities": _json_ready(config.frozen_identities),
        "inference": {
            "bootstrap_resamples": config.bootstrap_resamples,
            "sign_flip_resamples": config.sign_flip_resamples,
            "random_seed": config.random_seed,
            "holm_slots": config.expected_holm_slot_count,
        },
        "metric_output_ids": list(config.metric_output_ids),
        "protocol": "B",
        "tier": config.tier,
    }
    return RUN_PREFIX + _sha_bytes(_canonical(identity)), identity


@threadpool_limits.wrap(limits=1, user_api="blas")
def build_phase4_d5_protocol_b_from_inputs(
    output_dir: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    phase1_config: Phase1CoreConfig,
    config: Phase4D5ProtocolBConfig,
    protocol_a_path: Path,
    eligibility_path: Path,
    worker_count: int,
    inference_resamples: int | None = None,
) -> Phase4D5ProtocolBSummary:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Phase4D5ProtocolBError("output_dir", "must not exist")
    if sweep.sha256 != config.authorities["sweep_sha256"]:
        raise Phase4D5ProtocolBError("sweep", "identity mismatch")
    if phase1_config.file_sha256 != config.authorities["phase1_core_config_sha256"]:
        raise Phase4D5ProtocolBError("phase1 config", "identity mismatch")
    bridge = build_protocol_b_authority_bridge(
        cohort=cohort,
        protocol_a_path=protocol_a_path,
        eligibility_path=eligibility_path,
        config=config,
    )
    operator_cells, record_conditions, projected = _rematerialize_full_cohort(
        cohort, native_spectra, sweep, phase1_config, config, worker_count
    )
    operator_bytes = _jsonl(operator_cells)
    condition_bytes = _jsonl(record_conditions)
    expected_operator_bytes = Path(eligibility_path).joinpath("operator_cells.jsonl").read_bytes()
    expected_condition_bytes = Path(eligibility_path).joinpath("record_conditions.jsonl").read_bytes()
    if operator_bytes != expected_operator_bytes:
        raise Phase4D5ProtocolBError("rematerialization", "operator bytes differ from Step 9")
    if condition_bytes != expected_condition_bytes:
        raise Phase4D5ProtocolBError("rematerialization", "condition bytes differ from Step 9")
    downstream_rows = [
        {
            "record_id": row["record_id"],
            "condition_id": row["condition_id"],
            "values_sha256": row["support_projection_sha256"],
        }
        for row in record_conditions
    ]
    prediction_rows: list[dict[str, object]] = []
    for split in cohort.splits:
        query_indices = tuple(int(value) for value in split.query_indices)
        library_indices = tuple(int(value) for value in split.library_indices)
        query_ids = tuple(cohort.record_ids[index] for index in query_indices)
        library_ids = tuple(cohort.record_ids[index] for index in library_indices)
        for condition_id in config.condition_ids:
            query = np.stack([projected[(record_id, condition_id)] for record_id in query_ids])
            library = np.stack([projected[(record_id, condition_id)] for record_id in library_ids])
            result = match_d5_protocol_b_799(
                cohort,
                split,
                query_condition_id=condition_id,
                library_condition_id=condition_id,
                query_record_ids=query_ids,
                library_record_ids=library_ids,
                query_values=query,
                library_values=library,
            )
            query_projection_sha256 = _sha_bytes(
                b"".join(
                    _array_sha(projected[(record_id, condition_id)], "<f4").encode("ascii")
                    for record_id in query_ids
                )
            )
            library_projection_sha256 = _sha_bytes(
                b"".join(
                    _array_sha(projected[(record_id, condition_id)], "<f4").encode("ascii")
                    for record_id in library_ids
                )
            )
            for query_order, cohort_index in enumerate(query_indices):
                top_k = min(5, result.ranked_class_labels.shape[1])
                prediction_rows.append(
                    {
                        "split_seed": int(split.seed),
                        "split_sha256": split.split_sha256,
                        "query_order": query_order,
                        "cohort_index": cohort_index,
                        "record_id": cohort.record_ids[cohort_index],
                        "group_id": cohort.group_ids[cohort_index],
                        "class_label": int(cohort.class_labels[cohort_index]),
                        "condition_id": condition_id,
                        "query_condition_id": condition_id,
                        "library_condition_id": condition_id,
                        "query_projection_set_sha256": query_projection_sha256,
                        "library_projection_set_sha256": library_projection_sha256,
                        "top1_class_label": int(result.ranked_class_labels[query_order, 0]),
                        "top1_score": float(result.ranked_class_scores[query_order, 0]),
                        "top1_correct": bool(result.top1_correct[query_order]),
                        "top5_class_labels": [int(value) for value in result.ranked_class_labels[query_order, :top_k]],
                        "top5_scores": [float(value) for value in result.ranked_class_scores[query_order, :top_k]],
                        "top5_correct": bool(result.top5_correct[query_order]),
                    }
                )
    if len(prediction_rows) != config.expected_prediction_row_count:
        raise Phase4D5ProtocolBError("matcher predictions", "row count mismatch")
    class_rows: list[dict[str, object]] = []
    for metric_id in config.metric_output_ids:
        class_rows.extend(
            aggregate_protocol_b_class_observations(
                prediction_rows,
                bridge.metric_values,
                metric_output_id=metric_id,
                preferred_direction=config.metric_directions[metric_id].value,
                positive_conditions=config.condition_ids[1:],
            )
        )
    if len(class_rows) != config.expected_class_observation_count:
        raise Phase4D5ProtocolBError("class observations", "row count mismatch")
    alignment_rows, bootstrap_rows, sign_rows, family, figure1_rows, figure2_rows = _statistics_payloads(
        class_rows, config, inference_resamples=inference_resamples
    )
    if (
        len(alignment_rows), len(bootstrap_rows), len(sign_rows), len(family),
        len(figure1_rows), len(figure2_rows),
    ) != (13, 12, 24, 24, config.expected_figure1_row_count, 13):
        raise Phase4D5ProtocolBError("statistics", "payload count mismatch")
    figure_payloads = render_protocol_b_figure_payloads(figure1_rows, figure2_rows)
    condition_summary_rows = _condition_summaries(prediction_rows, config.condition_ids)
    code = dict(config.code_authority) if config.synthetic_fixture else _code_document()
    environment = dict(config.environment_authority) if config.synthetic_fixture and config.environment_authority else _environment_document()
    run_id, run_identity = _run_identity(config, code, environment)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": config.claim_boundary,
        "code": _json_ready(code),
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "operator_cells": len(operator_cells),
            "record_conditions": len(record_conditions),
            "downstream_rows": len(downstream_rows),
            "matcher_predictions": len(prediction_rows),
            "class_observations": len(class_rows),
            "alignment_results": len(alignment_rows),
            "bootstrap_results": len(bootstrap_rows),
            "sign_flip_results": len(sign_rows),
            "holm_family": len(family),
        },
        "environment": environment,
        "experiment_id": EXPERIMENT_ID,
        "inherited_rulings": _json_ready(config.inherited_rulings),
        "metric_authority": _json_ready(bridge.document["metric_authority"]),
        "perturbation_ids": list(config.perturbation_ids),
        "protocol": "B",
        "run_id": run_id,
        "run_identity": run_identity,
        "status": "complete",
        "synthetic_fixture": config.synthetic_fixture,
        "tier": config.tier,
    }
    preflight = {
        "authority_bridge_state": "complete",
        "rematerialization_state": "complete",
        "metric_authority_state": "complete",
        "matcher_state": "complete",
        "claim_boundary": "pre_matcher_bridges_and_rematerialization_complete",
    }
    complete = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
    }
    payloads = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": _canonical(bridge.document),
        "preflight.json": _canonical(preflight),
        "operator_cells.jsonl": operator_bytes,
        "record_conditions.jsonl": condition_bytes,
        "downstream_rows.jsonl": _jsonl(downstream_rows),
        "matcher_predictions.jsonl": _jsonl(prediction_rows),
        "condition_summary.csv": _csv_bytes(condition_summary_rows, tuple(condition_summary_rows[0])),
        "class_observations.jsonl": _jsonl(class_rows),
        "alignment_results.jsonl": _jsonl(alignment_rows),
        "bootstrap_results.jsonl": _jsonl(bootstrap_rows),
        "sign_flip_results.jsonl": _jsonl(sign_rows),
        "holm_family.jsonl": _jsonl(family),
        "d5_protocol_b_full_domain_secondary_table.csv": _csv_bytes(figure2_rows, tuple(figure2_rows[0])),
        "manifest.json": _canonical(manifest),
        "complete.json": _canonical(complete),
        **figure_payloads,
    }
    output_dir.mkdir(parents=True)
    for name in ARTIFACT_PAYLOAD_FILES:
        output_dir.joinpath(name).write_bytes(payloads[name])
    output_dir.joinpath("complete.json").write_bytes(payloads["complete.json"])
    checksum_names = (*ARTIFACT_PAYLOAD_FILES, "complete.json")
    output_dir.joinpath("SHA256SUMS").write_text(
        "".join(f"{_sha_bytes(payloads[name])}  {name}\n" for name in checksum_names),
        encoding="utf-8",
    )
    return Phase4D5ProtocolBSummary(
        output_dir, run_id, "complete", config.full_record_count, len(prediction_rows), len(class_rows)
    )


def build_phase4_d5_protocol_b(
    output_root: Path,
    *,
    worker_count: int = 16,
) -> Phase4D5ProtocolBSummary:
    config = load_phase4_d5_protocol_b_config(ROOT / CONFIG_RELATIVE_PATH)
    cohort = load_d5_raw_cohort(ROOT / D5_CONFIG_RELATIVE_PATH, ROOT / DATASET_RELATIVE_PATH)
    native = load_d5_native_spectra(ROOT / DATASET_RELATIVE_PATH, cohort.record_ids)
    sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH)
    phase1 = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    code = dict(config.code_authority)
    environment = dict(config.environment_authority)
    run_id, _ = _run_identity(config, code, environment)
    return build_phase4_d5_protocol_b_from_inputs(
        Path(output_root) / run_id,
        cohort=cohort,
        native_spectra=native,
        sweep=sweep,
        phase1_config=phase1,
        config=config,
        protocol_a_path=ROOT / PROTOCOL_A_RELATIVE_PATH,
        eligibility_path=ROOT / ELIGIBILITY_RELATIVE_PATH,
        worker_count=worker_count,
    )


__all__ = [
    "ARTIFACT_PAYLOAD_FILES",
    "CONDITION_BRIDGE_SHA256",
    "METRIC_OUTPUT_IDS",
    "OPERATOR_BRIDGE_SHA256",
    "Phase4D5ProtocolBConfig",
    "Phase4D5ProtocolBError",
    "Phase4D5ProtocolBSummary",
    "ProtocolBAuthorityBridge",
    "aggregate_protocol_b_class_observations",
    "build_protocol_b_authority_bridge",
    "build_phase4_d5_protocol_b",
    "build_phase4_d5_protocol_b_from_inputs",
    "fixed_protocol_b_holm_family",
    "load_phase4_d5_protocol_b_config",
    "match_d5_protocol_b_799",
    "parse_phase4_d5_protocol_b_config",
    "render_protocol_b_figure_payloads",
]
