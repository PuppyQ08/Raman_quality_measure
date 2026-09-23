from __future__ import annotations

import csv
import ast
import hashlib
import io
import json
import math
import os
import platform
import struct
import tempfile
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
from rpe.downstream.rruff import D5RawCohort, load_d5_native_spectra, load_d5_raw_cohort
from rpe.downstream.rruff_matching import match_d5_protocol_a_values
from rpe.evaluation import PreferredDirection, Spectrum1D
from rpe.perturb import PerturbationSweepConfig, load_perturbation_sweep_config
from rpe.runner.phase1_config import Phase1CoreConfig, load_phase1_core_config
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
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
EXPERIMENT_ID = "phase4-d5-protocol-b-full-domain-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d5-protocol-b-full-domain-artifact-v1"
RUN_PREFIX = "phase4-d5-protocol-b-full-domain-"
PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
METRIC_OUTPUT_IDS = ("mse", "rmse", "mae", "sam", "pearson_r", "nmse", "wasserstein_1_cm1", "is_like_structure_to_noise", "precision", "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio")
CANDIDATE_OUTPUT_IDS = METRIC_OUTPUT_IDS[1:]
ARTIFACT_PAYLOAD_FILES = ("config.json", "authority_bridge.json", "preflight.json", "operator_cells.jsonl", "record_conditions.jsonl", "downstream_rows.jsonl", "matcher_predictions.jsonl", "condition_summary.csv", "class_observations.jsonl", "alignment_results.jsonl", "bootstrap_results.jsonl", "sign_flip_results.jsonl", "holm_family.jsonl", "figure1_d5_protocol_b_full_domain.png", "figure1_d5_protocol_b_full_domain.svg", "figure1_d5_protocol_b_full_domain_data.csv", "figure2_d5_protocol_b_full_domain.png", "figure2_d5_protocol_b_full_domain.svg", "figure2_d5_protocol_b_full_domain_data.csv", "d5_protocol_b_full_domain_secondary_table.csv", "manifest.json")
PROTOCOL_A_ALLOWED_FILES = frozenset({"config.json", "manifest.json", "complete.json", "operator_cells.jsonl", "record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl", "SHA256SUMS"})
ELIGIBILITY_ALLOWED_FILES = frozenset({"config.json", "manifest.json", "complete.json", "unique_records.jsonl", "role_occurrences.jsonl", "operator_cells.jsonl", "record_conditions.jsonl", "SHA256SUMS"})
CODE_RELATIVE_PATHS = ("rpe/alignment/contracts.py", "rpe/alignment/core.py", "rpe/alignment/bulk.py", "rpe/alignment/inference.py", "rpe/downstream/rruff.py", "rpe/downstream/rruff_matching.py", "rpe/evaluation/contracts.py", "rpe/perturb/contracts.py", "rpe/perturb/sweep.py", "rpe/perturb/axis_transform.py", "rpe/perturb/baseline_distortion.py", "rpe/perturb/gaussian_noise.py", "rpe/perturb/correlated_noise.py", "rpe/runner/phase1_perturbations.py", "rpe/runner/phase1_selection.py", "rpe/runner/phase1_types.py", "rpe/runner/phase4_d5_protocol_b.py", "rpe/runner/phase4_d5_protocol_b_verifier.py", "tools/run_phase4_d5_protocol_b.py")


class Phase4D5ProtocolBError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path; self.reason = reason; super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D5ProtocolBConfig:
    path: Path; raw_bytes: bytes; sha256: str; document: Mapping[str, object]; synthetic_fixture: bool
    protocol: str; tier: str; perturbation_ids: tuple[str, ...]; alpha_grid: tuple[float, ...]; condition_ids: tuple[str, ...]
    metric_output_ids: tuple[str, ...]; metric_directions: Mapping[str, PreferredDirection]; artifact_payload_files: tuple[str, ...]
    authorities: Mapping[str, str]; parent_artifacts: Mapping[str, object]; frozen_identities: Mapping[str, object]; inherited_rulings: Mapping[str, object]
    claim_boundary: str; condition_bridge_sha256: str; operator_bridge_sha256: str; full_record_count: int; group_count: int; class_count: int
    query_record_count: int; query_occurrence_count: int; library_occurrence_count: int; split_count: int
    expected_operator_cell_count: int; expected_apply_check_count: int; expected_full_condition_count: int; expected_query_condition_count: int
    expected_metric_row_count: int; expected_peak_receipt_count: int; expected_matcher_call_count: int; expected_prediction_row_count: int
    expected_class_observation_count_per_metric: int; expected_class_observation_count: int; expected_holm_slot_count: int
    expected_figure1_row_count: int; expected_figure2_row_count: int; expected_secondary_table_row_count: int
    bootstrap_resamples: int; sign_flip_resamples: int; random_seed: int; confidence_level: float; holm_alpha: float
    support_start_cm1: float; support_stop_cm1: float; support_step_cm1: float; support_point_count: int; support_max_gap_cm1: float
    figure_contract: Mapping[str, object]; code_authority: Mapping[str, object]; environment_authority: Mapping[str, object]; trust_anchor: Mapping[str, object]


@dataclass(frozen=True)
class Phase4D5ProtocolBSummary:
    path: Path; run_id: str; status: str; full_record_count: int; prediction_row_count: int; class_observation_count: int


def _ready(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return _ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canon(value: object) -> bytes:
    return (json.dumps(_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canon(row) for row in rows)


def _csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    fields = tuple(rows[0])
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: "" if row.get(field) is None else row.get(field) for field in fields})
    return stream.getvalue().encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha(np.ascontiguousarray(value, dtype=dtype).tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha(("\n".join(sorted(values)) + "\n").encode())


def _class_digest(values: Sequence[int]) -> str:
    ordered = [str(value) for value in sorted(set(int(item) for item in values))]
    return _sha(("\n".join(ordered) + "\n").encode())


def _authority_constants() -> tuple[int, str]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {"CONFIG_BYTES", "CONFIG_SHA256"}:
                values[name] = ast.literal_eval(node.value)
    if not isinstance(values.get("CONFIG_BYTES"), int) or not isinstance(values.get("CONFIG_SHA256"), str):
        raise Phase4D5ProtocolBError("independent trust anchor", "constants missing")
    return int(values["CONFIG_BYTES"]), str(values["CONFIG_SHA256"])


def _require_object(path: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict): raise Phase4D5ProtocolBError(path, "must be object")
    return value


def _require_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0: raise Phase4D5ProtocolBError(path, "must be positive integer")
    return value


def _require_sha(path: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value): raise Phase4D5ProtocolBError(path, "must be lowercase SHA256")
    return value


def parse_phase4_d5_protocol_b_config(path: Path, raw: bytes, *, require_frozen_identity: bool) -> Phase4D5ProtocolBConfig:
    try: document = json.loads(raw)
    except Exception as error: raise Phase4D5ProtocolBError("independent config", str(error)) from error
    if not isinstance(document, dict) or _canon(document) != raw: raise Phase4D5ProtocolBError("independent config", "not canonical")
    if require_frozen_identity:
        byte_count, expected_sha = _authority_constants()
        if len(raw) != byte_count or _sha(raw) != expected_sha: raise Phase4D5ProtocolBError("independent frozen config identity", "mismatch")
    synthetic = bool(document.get("synthetic_fixture", False))
    if document.get("schema_version") != "phase4-d5-protocol-b-full-domain-config-v1" or document.get("experiment_id") != EXPERIMENT_ID: raise Phase4D5ProtocolBError("independent config", "schema/experiment mismatch")
    if document.get("protocol") != "B" or document.get("tier") != "full_domain_core": raise Phase4D5ProtocolBError("independent config", "protocol/tier mismatch")
    perturbations = tuple(document.get("active_perturbation_ids", ()))
    if perturbations != PERTURBATION_IDS: raise Phase4D5ProtocolBError("independent config", "perturbations mismatch")
    alpha_raw = document.get("alpha_grid")
    if not isinstance(alpha_raw, list) or len(alpha_raw) < 2: raise Phase4D5ProtocolBError("independent config", "alpha grid")
    alpha_grid = tuple(float(value) for value in alpha_raw)
    if alpha_grid[0] != 0.0 or any(not math.isfinite(value) for value in alpha_grid) or any(b <= a for a,b in zip(alpha_grid,alpha_grid[1:])): raise Phase4D5ProtocolBError("independent config", "alpha grid order")
    condition_ids = ("alpha0",) + tuple(f"{pid}:{np.float64(alpha).tobytes().hex()}" for pid in perturbations for alpha in alpha_grid[1:])
    artifact = tuple(document.get("artifact_payload_files", ()))
    if artifact != ARTIFACT_PAYLOAD_FILES: raise Phase4D5ProtocolBError("independent config", "artifact order")
    authorities = _require_object("independent authorities", document.get("authorities"))
    for name in ("d5_config_sha256","sweep_sha256","phase1_core_config_sha256","parent_plan_sha256"): _require_sha(f"independent authorities.{name}",authorities.get(name))
    parents = _require_object("independent parent_artifacts", document.get("parent_artifacts"))
    if set(parents) != {"protocol_a","eligibility"}: raise Phase4D5ProtocolBError("independent parent_artifacts","keys")
    for parent_name in parents:
        parent = _require_object("independent parent",parents[parent_name]); hashes = _require_object("independent parent hashes",parent.get("payload_sha256"))
        if not isinstance(parent.get("relative_path"),str) or not parent["relative_path"]: raise Phase4D5ProtocolBError("independent parent","path")
        for name,value in hashes.items(): _require_sha(f"independent parent {name}",value)
    bridge = _require_object("independent bridge",document.get("authority_bridge")); condition_sha=_require_sha("independent condition bridge",bridge.get("condition_bridge_sha256")); operator_sha=_require_sha("independent operator bridge",bridge.get("operator_bridge_sha256"))
    manifest = document.get("metric_manifest")
    if not isinstance(manifest,list) or len(manifest)!=13: raise Phase4D5ProtocolBError("independent metric manifest","count")
    directions={}; metric_ids=[]
    for item in manifest:
        item=_require_object("independent metric",item); metric=str(item.get("output_id","")); metric_ids.append(metric)
        try: directions[metric]=PreferredDirection(str(item.get("preferred_direction","")))
        except ValueError as error: raise Phase4D5ProtocolBError("independent metric","direction") from error
    if tuple(metric_ids)!=METRIC_OUTPUT_IDS: raise Phase4D5ProtocolBError("independent metric manifest","order")
    den=_require_object("independent denominators",document.get("denominators")); full=_require_int("full_record_count",den.get("full_record_count")); groups=_require_int("group_count",den.get("group_count")); classes=_require_int("class_count",den.get("class_count")); query_records=_require_int("query_record_count",den.get("query_record_count")); query_occ=_require_int("query_occurrence_count",den.get("query_occurrence_count")); library_occ=_require_int("library_occurrence_count",den.get("library_occurrence_count")); splits=_require_int("split_count",den.get("split_count"))
    expected=_require_object("independent expected",document.get("expected")); keys=("operator_cell_count","apply_check_count","full_condition_count","query_condition_count","metric_row_count","peak_receipt_count","matcher_call_count","prediction_row_count","class_observation_count_per_metric","class_observation_count","holm_slot_count","figure1_row_count","figure2_row_count","secondary_table_row_count"); vals={name:_require_int(name,expected.get(name)) for name in keys}; positive=len(condition_ids)-1; derived={"operator_cell_count":full*5,"apply_check_count":full*5*len(alpha_grid),"full_condition_count":full*len(condition_ids),"query_condition_count":query_records*len(condition_ids),"metric_row_count":query_records*len(condition_ids)*13,"peak_receipt_count":query_records*len(condition_ids),"matcher_call_count":splits*len(condition_ids),"prediction_row_count":query_occ*len(condition_ids),"class_observation_count_per_metric":classes*positive,"class_observation_count":classes*positive*13,"holm_slot_count":24,"figure1_row_count":13*positive,"figure2_row_count":13,"secondary_table_row_count":13}
    if vals!=derived: raise Phase4D5ProtocolBError("independent expected","derived mismatch")
    inference=_require_object("independent inference",document.get("inference")); boot=_require_int("bootstrap",inference.get("bootstrap_resamples")); flips=_require_int("sign flips",inference.get("sign_flip_resamples")); seed=int(inference.get("random_seed")); confidence=float(inference.get("confidence_level")); holm=float(inference.get("holm_alpha"))
    support=_require_object("independent support",document.get("support_grid")); start=float(support["start_cm1"]); stop=float(support["stop_cm1"]); step=float(support["step_cm1"]); points=_require_int("support points",support.get("point_count")); max_gap=float(support["max_in_range_native_gap_cm1"]);
    if int(round((stop-start)/step))+1!=points: raise Phase4D5ProtocolBError("independent support","count")
    frozen=_require_object("independent frozen identities",document.get("frozen_identities")); rulings=_require_object("independent rulings",document.get("inherited_rulings")); figure=_require_object("independent figure",document.get("figure_contract")); code=_require_object("independent code",document.get("code_authority")); environment=_require_object("independent environment",document.get("environment_authority")); trust=_require_object("independent trust",document.get("trust_anchor")); claim=str(document.get("claim_boundary",""))
    if figure.get("svg_hashsalt")!="rpe-phase4-d5-protocol-b-v1" or trust.get("config_authority_relative_path")!=CONFIG_AUTHORITY_RELATIVE_PATH or claim!="local_execution_artifact_redistribution_not_cleared": raise Phase4D5ProtocolBError("independent config","figure/trust/claim")
    if not synthetic:
        if (full,groups,classes,query_records,query_occ,library_occ,splits)!=(3770,1934,681,3012,6621,12229,5) or boot!=2000 or flips!=100000 or seed!=20260817: raise Phase4D5ProtocolBError("independent real config","fixed values")
        current_code={relative:{"bytes": (ROOT/relative).stat().st_size,"sha256":_sha((ROOT/relative).read_bytes())} for relative in CODE_RELATIVE_PATHS}
        if set(code)!=set(CODE_RELATIVE_PATHS) or code!=current_code or environment!=_environment(): raise Phase4D5ProtocolBError("independent real identity","code/environment")
    return Phase4D5ProtocolBConfig(Path(path),raw,_sha(raw),MappingProxyType(document),synthetic,"B","full_domain_core",perturbations,alpha_grid,condition_ids,tuple(metric_ids),MappingProxyType(directions),artifact,MappingProxyType(authorities),MappingProxyType(parents),MappingProxyType(frozen),MappingProxyType(rulings),claim,condition_sha,operator_sha,full,groups,classes,query_records,query_occ,library_occ,splits,vals["operator_cell_count"],vals["apply_check_count"],vals["full_condition_count"],vals["query_condition_count"],vals["metric_row_count"],vals["peak_receipt_count"],vals["matcher_call_count"],vals["prediction_row_count"],vals["class_observation_count_per_metric"],vals["class_observation_count"],vals["holm_slot_count"],vals["figure1_row_count"],vals["figure2_row_count"],vals["secondary_table_row_count"],boot,flips,seed,confidence,holm,start,stop,step,points,max_gap,MappingProxyType(figure),MappingProxyType(code),MappingProxyType(environment),MappingProxyType(trust))


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_bytes())
    except Exception as error:
        raise Phase4D5ProtocolBError(label, f"cannot load {path.name}: {error}") from error
    if not isinstance(value, dict):
        raise Phase4D5ProtocolBError(label, f"{path.name} is not an object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, object]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Phase4D5ProtocolBError(label, f"{path.name}:{number} is not an object")
                rows.append(value)
    except Phase4D5ProtocolBError:
        raise
    except Exception as error:
        raise Phase4D5ProtocolBError(label, f"cannot load {path.name}: {error}") from error
    return rows


def _parent_hashes(config: Phase4D5ProtocolBConfig, name: str) -> Mapping[str, object]:
    parent = config.parent_artifacts[name]
    assert isinstance(parent, Mapping)
    hashes = parent["payload_sha256"]
    assert isinstance(hashes, Mapping)
    return hashes


def _validate_file(path: Path, config: Phase4D5ProtocolBConfig, parent: str, label: str) -> bytes:
    raw = path.read_bytes()
    expected = _parent_hashes(config, parent).get(path.name)
    if expected is not None and _sha(raw) != expected:
        raise Phase4D5ProtocolBError(label, f"hash mismatch for {path.name}")
    return raw


def _condition_view(row: Mapping[str, object], step9: bool) -> dict[str, object]:
    return {
        "record_id": str(row["record_id"]),
        "condition_id": str(row["condition_id"]),
        "axis_sha256": str(row["axis_sha256"]),
        "intensity_sha256": str(row["intensity_sha256"]),
        "projection_sha256": str(row["support_projection_sha256"] if step9 else row["projection_sha256"]),
    }


def _operator_view(row: Mapping[str, object], step9: bool) -> dict[str, object]:
    return {
        "record_id": str(row["record_id"]),
        "perturbation_id": str(row["perturbation_id"]),
        "state_digest": row.get("state_digest"),
        "native_gate": _ready(row["native_gate"]),
        "outputs": [
            {
                "alpha": float(output["alpha"]),
                "alpha_float64_le_hex": str(output["alpha_float64_le_hex"]),
                "diagnostics": _ready(output["diagnostics"]),
                "output_spectrum_id": str(output["output_spectrum_id"]),
                "axis_sha256": str(output["output_axis_sha256"] if step9 else output["axis_sha256"]),
                "intensity_sha256": str(output["output_intensity_sha256"] if step9 else output["intensity_sha256"]),
            }
            for output in row["outputs"]
        ],
    }


def _keyed(rows: Sequence[Mapping[str, object]], second: str, label: str) -> dict[tuple[str, str], Mapping[str, object]]:
    output = {}
    for row in rows:
        key = (str(row["record_id"]), str(row[second]))
        if key in output:
            raise Phase4D5ProtocolBError(label, f"duplicate key {key!r}")
        output[key] = row
    return output


def _rows_sha(rows: Sequence[Mapping[str, object]], second: str) -> str:
    ordered = sorted(rows, key=lambda row: f"{row['record_id']}|{row[second]}")
    return _sha(b"".join(_canon(row) for row in ordered))


def _bridge(
    cohort: D5RawCohort, protocol_a: Path, eligibility: Path, config: Phase4D5ProtocolBConfig
) -> tuple[dict[str, object], dict[tuple[str, str, str], float]]:
    for name in sorted(PROTOCOL_A_ALLOWED_FILES):
        _validate_file(protocol_a / name, config, "protocol_a", "independent Protocol-A authority")
    for name in sorted(ELIGIBILITY_ALLOWED_FILES):
        _validate_file(eligibility / name, config, "eligibility", "independent eligibility authority")
    a_config = _load_json(protocol_a / "config.json", "independent Protocol-A authority")
    a_manifest = _load_json(protocol_a / "manifest.json", "independent Protocol-A authority")
    a_complete = _load_json(protocol_a / "complete.json", "independent Protocol-A authority")
    b_config = _load_json(eligibility / "config.json", "independent eligibility authority")
    b_manifest = _load_json(eligibility / "manifest.json", "independent eligibility authority")
    b_complete = _load_json(eligibility / "complete.json", "independent eligibility authority")
    a_protocol = a_config.get("protocol")
    if (a_protocol != "A" and not (a_protocol is None and a_config.get("synthetic_fixture"))) or a_manifest.get("protocol") != "A" or a_complete.get("status") != "complete":
        raise Phase4D5ProtocolBError("independent Protocol-A authority", "invalid parent")
    if b_config.get("protocol") != "B" or b_manifest.get("protocol") != "B" or b_complete.get("status") not in {"pass", "complete"}:
        raise Phase4D5ProtocolBError("independent eligibility authority", "invalid parent")
    unique = _load_jsonl(eligibility / "unique_records.jsonl", "independent cohort bridge")
    roles = _load_jsonl(eligibility / "role_occurrences.jsonl", "independent cohort bridge")
    if len(unique) != config.full_record_count:
        raise Phase4D5ProtocolBError("independent cohort bridge", "record count")
    for index, row in enumerate(unique):
        if (int(row["cohort_index"]), str(row["record_id"]), str(row["group_id"]), int(row["class_label"])) != (index, cohort.record_ids[index], cohort.group_ids[index], int(cohort.class_labels[index])):
            raise Phase4D5ProtocolBError("independent cohort bridge", "record identity")
    expected_roles = []
    for split in cohort.splits:
        for role, indices in (("query", split.query_indices), ("library", split.library_indices)):
            for order, raw_index in enumerate(indices):
                index = int(raw_index)
                expected_roles.append((int(split.seed), split.split_sha256, role, order, index, cohort.record_ids[index], cohort.group_ids[index], int(cohort.class_labels[index]), index))
    observed_roles = [(int(row["split_seed"]), str(row["split_sha256"]), str(row["role"]), int(row["role_order"]), int(row["cohort_index"]), str(row["record_id"]), str(row["group_id"]), int(row["class_label"]), int(row["unique_record_order"])) for row in roles]
    if observed_roles != expected_roles:
        raise Phase4D5ProtocolBError("independent cohort bridge", "role identity")
    query_indices = sorted({int(row["cohort_index"]) for row in roles if row["role"] == "query"}, key=lambda index: cohort.record_ids[index])
    query_ids = tuple(cohort.record_ids[index] for index in query_indices)
    digests = {
        "record_ids_sha256": _ids_digest(list(cohort.record_ids)),
        "group_ids_sha256": _ids_digest(sorted(set(cohort.group_ids))),
        "class_labels_sha256": _class_digest([int(value) for value in cohort.class_labels]),
        "query_record_ids_sha256": _ids_digest(list(query_ids)),
    }
    for name, observed in digests.items():
        expected_digest = config.frozen_identities.get(name)
        if expected_digest is not None and expected_digest != observed:
            raise Phase4D5ProtocolBError("independent cohort bridge", f"{name} mismatch")
    a_conditions = _load_jsonl(protocol_a / "record_conditions.jsonl", "independent condition bridge")
    b_conditions = _load_jsonl(eligibility / "record_conditions.jsonl", "independent condition bridge")
    a_by = _keyed(a_conditions, "condition_id", "independent condition bridge")
    b_by = _keyed(b_conditions, "condition_id", "independent condition bridge")
    expected = {(record, condition) for record in query_ids for condition in config.condition_ids}
    if set(a_by) != expected or not expected.issubset(b_by):
        raise Phase4D5ProtocolBError("independent condition bridge", "key set")
    condition_rows = [_condition_view(a_by[key], False) for key in sorted(expected)]
    if any(_condition_view(a_by[key], False) != _condition_view(b_by[key], True) for key in expected):
        raise Phase4D5ProtocolBError("independent condition bridge", "mismatch")
    condition_sha = _rows_sha(condition_rows, "condition_id")
    if condition_sha != config.condition_bridge_sha256:
        raise Phase4D5ProtocolBError("independent condition bridge", "receipt SHA")
    a_cells = _load_jsonl(protocol_a / "operator_cells.jsonl", "independent operator bridge")
    b_cells = _load_jsonl(eligibility / "operator_cells.jsonl", "independent operator bridge")
    a_cell_by = _keyed(a_cells, "perturbation_id", "independent operator bridge")
    b_cell_by = _keyed(b_cells, "perturbation_id", "independent operator bridge")
    expected_cells = {(record, perturbation) for record in query_ids for perturbation in config.perturbation_ids}
    operator_rows = [_operator_view(a_cell_by[key], False) for key in sorted(expected_cells)]
    if set(a_cell_by) != expected_cells or not expected_cells.issubset(b_cell_by) or any(b_cell_by[key].get("state") != "complete" or _operator_view(a_cell_by[key], False) != _operator_view(b_cell_by[key], True) for key in expected_cells):
        raise Phase4D5ProtocolBError("independent operator bridge", "mismatch")
    operator_sha = _rows_sha(operator_rows, "perturbation_id")
    if operator_sha != config.operator_bridge_sha256:
        raise Phase4D5ProtocolBError("independent operator bridge", "receipt SHA")
    metric_rows = _load_jsonl(protocol_a / "metric_values.jsonl", "independent metric authority")
    if len(metric_rows) != config.expected_metric_row_count:
        raise Phase4D5ProtocolBError("independent metric authority", "count")
    metrics = {}
    for row in metric_rows:
        key = (str(row["record_id"]), str(row["condition_id"]), str(row["metric_output_id"]))
        if key in metrics or key[0] not in query_ids or row.get("state") != "complete":
            raise Phase4D5ProtocolBError("independent metric authority", "invalid row")
        metrics[key] = float(row["value"])
    peaks = _load_jsonl(protocol_a / "peak_receipts.jsonl", "independent CWT authority")
    if len(peaks) != config.expected_peak_receipt_count:
        raise Phase4D5ProtocolBError("independent CWT authority", "count")
    document = {
        "schema_version": "phase4-d5-protocol-b-authority-bridge-v1",
        "cohort_bridge": {"record_count": len(cohort.record_ids), "group_count": len(set(cohort.group_ids)), "class_count": len(set(int(value) for value in cohort.class_labels)), "class_labels": [int(value) for value in cohort.class_labels], "query_record_count": len(query_ids), "query_occurrence_count": sum(1 for row in roles if row["role"] == "query"), "library_occurrence_count": sum(1 for row in roles if row["role"] == "library"), "split_sha256": [split.split_sha256 for split in cohort.splits], "state": "complete"},
        "condition_bridge": {"row_count": len(condition_rows), "mismatch_count": 0, "sha256": condition_sha, "state": "complete"},
        "operator_bridge": {"cell_count": len(operator_rows), "output_count": sum(len(row["outputs"]) for row in operator_rows), "mismatch_count": 0, "sha256": operator_sha, "state": "complete"},
        "metric_authority": {"metric_row_count": len(metrics), "peak_receipt_count": len(peaks), "query_only": True, "state": "complete"},
        "protocol_a_parent_sha256sums_sha256": _sha((protocol_a / "SHA256SUMS").read_bytes()),
        "eligibility_parent_sha256sums_sha256": _sha((eligibility / "SHA256SUMS").read_bytes()),
    }
    return document, metrics


def _grid(config: Phase4D5ProtocolBConfig) -> np.ndarray:
    output = np.arange(config.support_start_cm1, config.support_stop_cm1 + config.support_step_cm1 / 2.0, config.support_step_cm1, dtype="<f8")
    if output.size != config.support_point_count:
        raise Phase4D5ProtocolBError("independent grid", "count")
    return output


def _project(spectrum: Spectrum1D, grid: np.ndarray, max_gap: float) -> tuple[np.ndarray, float]:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5ProtocolBError("independent projection", "extrapolation")
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    support = axis[left : right + 1]
    gap = float(np.max(np.diff(support))) if support.size > 1 else math.inf
    if left < 0 or right >= axis.size or support.size < 2 or gap > max_gap:
        raise Phase4D5ProtocolBError("independent projection", "bounds or gap")
    values = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    if not np.isfinite(values).all() or float(np.linalg.norm(values.astype(np.float64))) <= 0.0:
        raise Phase4D5ProtocolBError("independent projection", "nonfinite or zero norm")
    return values, gap


def _source(cohort: D5RawCohort, native: tuple[Spectrum1D, ...], index: int) -> tuple[dict[str, object], Phase1Source]:
    spectrum = native[index]
    record = {"class_label": int(cohort.class_labels[index]), "cohort_index": index, "group_id": cohort.group_ids[index], "mineral_name": cohort.mineral_names[index], "native_axis_sha256": _array_sha(spectrum.axis_cm1), "native_intensity_sha256": _array_sha(spectrum.intensity), "point_count": int(spectrum.axis_cm1.size), "record_id": cohort.record_ids[index], "record_order": index, "rruff_id": cohort.rruff_ids[index]}
    axis_f4 = np.ascontiguousarray(spectrum.axis_cm1, dtype="<f4")
    intensity_f4 = np.ascontiguousarray(spectrum.intensity, dtype="<f4")
    phase1 = Phase1Source(
        selection=SelectedSourceRow(index, cohort.record_ids[index], spectrum.sample_id or cohort.rruff_ids[index], int(cohort.class_labels[index]), cohort.mineral_names[index], f"native::{record['native_axis_sha256']}"),
        spectrum=spectrum, original_axis_orientation="increasing",
        source_axis_float32_sha256=_array_sha(axis_f4, "<f4"), source_intensity_float32_sha256=_array_sha(intensity_f4, "<f4"),
        normalized_axis_float64_sha256=str(record["native_axis_sha256"]), normalized_intensity_float64_sha256=str(record["native_intensity_sha256"]),
        provenance=MappingProxyType({"d5_protocol_config_sha256": "", "dataset_id": "rruff_raman_raw", "record_id": cohort.record_ids[index]}),
    )
    return record, phase1


def _record_materialization(
    cohort: D5RawCohort, native: tuple[Spectrum1D, ...], index: int, sweep: PerturbationSweepConfig, phase1_config: Phase1CoreConfig, config: Phase4D5ProtocolBConfig, grid: np.ndarray, admission: P10MemoryAdmission
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    record, source = _source(cohort, native, index)
    spectrum = native[index]; record_id = cohort.record_ids[index]
    alpha0, alpha0_gap = _project(spectrum, grid, config.support_max_gap_cm1)
    projections = {"alpha0": alpha0}
    conditions = [{"alpha": 0.0, "alpha_float64_le_hex": np.float64(0.0).tobytes().hex(), "axis_sha256": record["native_axis_sha256"], "class_label": record["class_label"], "condition_id": "alpha0", "condition_kind": "alpha0", "group_id": record["group_id"], "intensity_sha256": record["native_intensity_sha256"], "perturbation_id": None, "record_id": record_id, "record_order": index, "state": "complete", "support_max_in_range_gap_cm1": alpha0_gap, "support_point_count": int(alpha0.size), "support_projection_sha256": _array_sha(alpha0, "<f4")}]
    cells = []; zero_hashes = []
    for perturbation in config.perturbation_ids:
        cell = run_perturbation_cell(source, perturbation, phase1_config, sweep, p10_admission=admission)
        if cell.status is not CellStatus.COMPLETE:
            raise Phase4D5ProtocolBError("independent rematerialization", f"{record_id}/{perturbation}")
        outputs = []
        for perturbed in cell.records:
            result = perturbed.result; output = result.output
            values, gap = _project(output, grid, config.support_max_gap_cm1)
            axis_sha = _array_sha(output.axis_cm1); intensity_sha = _array_sha(output.intensity); projection_sha = _array_sha(values, "<f4")
            outputs.append({"alpha": float(result.alpha), "alpha_float64_le_hex": perturbed.alpha_float64_le_hex, "axis_changed": bool(result.axis_changed), "diagnostics": _ready(result.diagnostics), "intensity_changed": bool(result.intensity_changed), "output_axis_sha256": axis_sha, "output_intensity_sha256": intensity_sha, "output_spectrum_id": output.spectrum_id, "support_max_in_range_gap_cm1": gap, "support_point_count": int(values.size), "support_projection_sha256": projection_sha})
            if result.alpha == 0.0:
                zero_hashes.append((axis_sha, intensity_sha)); continue
            condition = f"{perturbation}:{perturbed.alpha_float64_le_hex}"; projections[condition] = values
            conditions.append({"alpha": float(result.alpha), "alpha_float64_le_hex": perturbed.alpha_float64_le_hex, "axis_sha256": axis_sha, "class_label": record["class_label"], "condition_id": condition, "condition_kind": "positive", "group_id": record["group_id"], "intensity_sha256": intensity_sha, "perturbation_id": perturbation, "record_id": record_id, "record_order": index, "state": "complete", "support_max_in_range_gap_cm1": gap, "support_point_count": int(values.size), "support_projection_sha256": projection_sha})
        cells.append({"class_label": record["class_label"], "exception": None, "group_id": record["group_id"], "native_gate": _ready(cell.evidence.native_gate), "output_count": len(outputs), "outputs": outputs, "p10_estimated_peak_bytes": estimate_p10_peak_bytes(int(record["point_count"])) if perturbation == "p10" else None, "perturbation_id": perturbation, "reason_code": None, "record_id": record_id, "state": "complete", "state_digest": None if cell.state is None else cell.state.state_digest})
    source_hash = (record["native_axis_sha256"], record["native_intensity_sha256"])
    if len(zero_hashes) != 5 or any(value != source_hash for value in zero_hashes):
        raise Phase4D5ProtocolBError("independent alpha zero", "mismatch")
    return cells, conditions, projections


def _rematerialize(cohort: D5RawCohort, native: tuple[Spectrum1D, ...], sweep: PerturbationSweepConfig, phase1: Phase1CoreConfig, config: Phase4D5ProtocolBConfig, workers: int) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[tuple[str, str], np.ndarray]]:
    if len(cohort.record_ids) != config.full_record_count or tuple(sweep.alpha_grid) != config.alpha_grid:
        raise Phase4D5ProtocolBError("independent rematerialization", "input identity")
    estimates = [estimate_p10_peak_bytes(spectrum.axis_cm1.size) for spectrum in native]
    budget = 64 * 2**30
    if max(estimates) > budget:
        raise Phase4D5ProtocolBError("independent P10", "budget")
    grid = _grid(config); admission = P10MemoryAdmission(budget); results = {}
    with threadpool_limits(limits=1, user_api="blas"):
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {index: executor.submit(_record_materialization, cohort, native, index, sweep, phase1, config, grid, admission) for index in range(config.full_record_count)}
            for index, future in futures.items(): results[index] = future.result()
    cells = []; conditions = []; projected = {}
    for index in range(config.full_record_count):
        current_cells, current_conditions, current_projected = results[index]; cells.extend(current_cells); conditions.extend(current_conditions)
        for condition, values in current_projected.items(): projected[(cohort.record_ids[index], condition)] = values
    return cells, conditions, projected


def _predict(cohort: D5RawCohort, projected: Mapping[tuple[str, str], np.ndarray], config: Phase4D5ProtocolBConfig) -> list[dict[str, object]]:
    rows = []
    with threadpool_limits(limits=1, user_api="blas"):
        for split in cohort.splits:
            query_indices = tuple(int(value) for value in split.query_indices); library_indices = tuple(int(value) for value in split.library_indices)
            query_ids = tuple(cohort.record_ids[index] for index in query_indices); library_ids = tuple(cohort.record_ids[index] for index in library_indices)
            for condition in config.condition_ids:
                query = np.stack([projected[(record, condition)] for record in query_ids]); library = np.stack([projected[(record, condition)] for record in library_ids])
                result = match_d5_protocol_a_values(cohort, split, condition_id=condition, query_record_ids=query_ids, library_record_ids=library_ids, query_values=query, library_values=library)
                q_sha = _sha(b"".join(_array_sha(projected[(record, condition)], "<f4").encode() for record in query_ids)); l_sha = _sha(b"".join(_array_sha(projected[(record, condition)], "<f4").encode() for record in library_ids))
                for order, cohort_index in enumerate(query_indices):
                    top_k = min(5, result.ranked_class_labels.shape[1])
                    rows.append({"split_seed": int(split.seed), "split_sha256": split.split_sha256, "query_order": order, "cohort_index": cohort_index, "record_id": cohort.record_ids[cohort_index], "group_id": cohort.group_ids[cohort_index], "class_label": int(cohort.class_labels[cohort_index]), "condition_id": condition, "query_condition_id": condition, "library_condition_id": condition, "query_projection_set_sha256": q_sha, "library_projection_set_sha256": l_sha, "top1_class_label": int(result.ranked_class_labels[order, 0]), "top1_score": float(result.ranked_class_scores[order, 0]), "top1_correct": bool(result.top1_correct[order]), "top5_class_labels": [int(value) for value in result.ranked_class_labels[order, :top_k]], "top5_scores": [float(value) for value in result.ranked_class_scores[order, :top_k]], "top5_correct": bool(result.top5_correct[order])})
    return rows


def _class_rows(predictions: Sequence[Mapping[str, object]], metrics: Mapping[tuple[str, str, str], float], config: Phase4D5ProtocolBConfig) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in predictions: grouped[(int(row["class_label"]), str(row["condition_id"]))].append(row)
    output = []
    for metric_id in config.metric_output_ids:
        direction = config.metric_directions[metric_id]
        for label in sorted({key[0] for key in grouped}):
            baseline = grouped[(label, "alpha0")]; baseline_error = float(np.mean([not bool(row["top1_correct"]) for row in baseline]))
            for condition in config.condition_ids[1:]:
                rows = grouped[(label, condition)]; downstream = float(np.mean([not bool(row["top1_correct"]) for row in rows])) - baseline_error
                harms = [orient_harm(metrics[(str(row["record_id"]), "alpha0", metric_id)], metrics[(str(row["record_id"]), condition, metric_id)], direction) for row in rows]
                perturbation, alpha_hex = condition.split(":", 1)
                output.append({"cluster_id": str(label), "perturbation_id": perturbation, "alpha": struct.unpack("<d", bytes.fromhex(alpha_hex))[0], "metric_output_id": metric_id, "metric_harm": float(np.mean(harms)), "downstream_harm": downstream, "occurrence_count": len(rows), "state": "complete"})
    return output


def _holm(raw: Mapping[str, float], contrasts: Mapping[str, float]) -> list[dict[str, object]]:
    adjusted = {row.hypothesis_id: row for row in holm_step_down(raw, alpha=0.05)}; output = []
    for hypothesis in sorted(raw):
        metric, statistic = hypothesis.rsplit(":", 1); contrast = float(contrasts[hypothesis]); favorable = contrast > 0.0; row = adjusted[hypothesis]
        output.append({"family_id": "d5_protocol_b_full_domain_secondary_24", "metric_output_id": metric, "statistic": statistic, "hypothesis_id": hypothesis, "raw_p_value": float(raw[hypothesis]), "adjusted_p_value": row.adjusted_p_value, "rank": row.rank, "family_size": row.family_size, "observed_contrast": contrast, "favorable": favorable, "rejected": bool(row.rejected and favorable)})
    return output


def _terminal_family() -> list[dict[str, object]]:
    raw = {f"{metric}:{statistic}": 1.0 for metric in CANDIDATE_OUTPUT_IDS for statistic in ("d_ag", "d_acc")}; adjusted = {row.hypothesis_id: row for row in holm_step_down(raw, alpha=0.05)}
    return [{"family_id": "d5_protocol_b_full_domain_secondary_24", "metric_output_id": hypothesis.rsplit(":", 1)[0], "statistic": hypothesis.rsplit(":", 1)[1], "hypothesis_id": hypothesis, "state": "not_tested_constant_downstream", "raw_p_value": None, "multiplicity_p_value": 1.0, "adjusted_p_value": adjusted[hypothesis].adjusted_p_value, "rank": adjusted[hypothesis].rank, "family_size": adjusted[hypothesis].family_size, "observed_contrast": None, "favorable": None, "rejected": False} for hypothesis in sorted(raw)]


def _stats(rows: Sequence[Mapping[str, object]], config: Phase4D5ProtocolBConfig, override: int | None) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    by_metric = defaultdict(list)
    for row in rows: by_metric[str(row["metric_output_id"])].append(AlignmentObservation(str(row["cluster_id"]), str(row["perturbation_id"]), float(row["alpha"]), float(row["metric_harm"]), float(row["downstream_harm"])))
    reference = tuple(by_metric["mse"]); boot_n = config.bootstrap_resamples if override is None else override; flip_n = config.sign_flip_resamples if override is None else max(64, override)
    if not config.synthetic_fixture and override is not None: raise Phase4D5ProtocolBError("independent inference", "override forbidden")
    results = {}; bootstrap_rows = []; sign_rows = []; raw = {}; contrasts = {}; constant = False
    try:
        ref_gap = alignment_gap(reference); ref_acc = cross_perturbation_accuracy(reference); mse_boot = bulk_paired_cluster_bootstrap(reference, reference, resamples=boot_n, confidence_level=config.confidence_level, random_seed=config.random_seed)
        results["mse"] = {"metric_output_id": "mse", "metric_state": "complete", "ag": ref_gap.alignment_gap, "acc": ref_acc.accuracy, "comparison": None}
    except AlignmentValidationError as error:
        if error.path != "constant downstream": raise
        constant = True; ref_acc = None; mse_boot = None; results["mse"] = {"metric_output_id": "mse", "metric_state": "not_evaluable_constant_downstream", "ag": None, "acc": None, "comparison": None}
    for metric in CANDIDATE_OUTPUT_IDS:
        if constant:
            results[metric] = {"metric_output_id": metric, "metric_state": "not_evaluable_constant_downstream", "ag": None, "acc": None, "comparison": None}; bootstrap_rows.append({"metric_output_id": metric, "state": "not_evaluable_constant_downstream"}); sign_rows.extend({"metric_output_id": metric, "statistic": statistic, "state": "not_evaluable_constant_downstream"} for statistic in ("d_ag", "d_acc")); continue
        candidate = tuple(by_metric[metric]); comparison = compare_alignment(reference, candidate); boot = bulk_paired_cluster_bootstrap(reference, candidate, resamples=boot_n, confidence_level=config.confidence_level, random_seed=config.random_seed)
        ag = paired_contribution_sign_flip(tuple(value.value for value in comparison.ag_contribution_differences), aggregation="sum", resamples=flip_n, random_seed=config.random_seed); acc = paired_contribution_sign_flip(tuple(value.value for value in comparison.acc_contribution_differences), aggregation="mean", resamples=flip_n, random_seed=config.random_seed)
        raw[f"{metric}:d_ag"] = ag.p_value; raw[f"{metric}:d_acc"] = acc.p_value; contrasts[f"{metric}:d_ag"] = comparison.d_ag; contrasts[f"{metric}:d_acc"] = comparison.d_acc
        results[metric] = {"metric_output_id": metric, "metric_state": "complete", "ag": comparison.candidate_gap.alignment_gap, "acc": comparison.candidate_accuracy.accuracy, "comparison": _ready(comparison)}; bootstrap_rows.append({"metric_output_id": metric, "state": "complete", **_ready(boot)}); sign_rows.extend(({"metric_output_id": metric, "statistic": "d_ag", "state": "complete", **_ready(ag)}, {"metric_output_id": metric, "statistic": "d_acc", "state": "complete", **_ready(acc)}))
    family = _terminal_family() if constant else _holm(raw, contrasts); family_by = {(row["metric_output_id"], row["statistic"]): row for row in family}; boot_by = {row["metric_output_id"]: row for row in bootstrap_rows}; figure2 = []
    for metric in METRIC_OUTPUT_IDS:
        result = results[metric]
        if metric == "mse" and mse_boot is not None and ref_acc is not None:
            figure2.append({"metric_output_id": metric, "metric_state": "complete", "ag": result["ag"], "ag_lower": mse_boot.reference_ag_interval[0], "ag_upper": mse_boot.reference_ag_interval[1], "acc": result["acc"], "acc_lower": mse_boot.reference_acc_interval[0], "acc_upper": mse_boot.reference_acc_interval[1], "pair_count": ref_acc.pair_count, "strict_agreement_count": ref_acc.strict_agreement_count, "strict_disagreement_count": ref_acc.strict_disagreement_count, "metric_tie_count": ref_acc.metric_tie_count, "downstream_tie_count": ref_acc.downstream_tie_count, "double_tie_count": ref_acc.double_tie_count, "d_ag": None, "d_ag_lower": None, "d_ag_upper": None, "d_ag_favorable": None, "d_ag_raw_p": None, "d_ag_adjusted_p": None, "d_acc": None, "d_acc_lower": None, "d_acc_upper": None, "d_acc_favorable": None, "d_acc_raw_p": None, "d_acc_adjusted_p": None})
        elif result["metric_state"] == "complete":
            boot = boot_by[metric]; comparison = result["comparison"]; ag = family_by[(metric, "d_ag")]; acc = family_by[(metric, "d_acc")]; ca = comparison["candidate_accuracy"]
            figure2.append({"metric_output_id": metric, "metric_state": "complete", "ag": result["ag"], "ag_lower": boot["candidate_ag_interval"][0], "ag_upper": boot["candidate_ag_interval"][1], "acc": result["acc"], "acc_lower": boot["candidate_acc_interval"][0], "acc_upper": boot["candidate_acc_interval"][1], "pair_count": ca["pair_count"], "strict_agreement_count": ca["strict_agreement_count"], "strict_disagreement_count": ca["strict_disagreement_count"], "metric_tie_count": ca["metric_tie_count"], "downstream_tie_count": ca["downstream_tie_count"], "double_tie_count": ca["double_tie_count"], "d_ag": comparison["d_ag"], "d_ag_lower": boot["d_ag_interval"][0], "d_ag_upper": boot["d_ag_interval"][1], "d_ag_favorable": ag["favorable"], "d_ag_raw_p": ag["raw_p_value"], "d_ag_adjusted_p": ag["adjusted_p_value"], "d_acc": comparison["d_acc"], "d_acc_lower": boot["d_acc_interval"][0], "d_acc_upper": boot["d_acc_interval"][1], "d_acc_favorable": acc["favorable"], "d_acc_raw_p": acc["raw_p_value"], "d_acc_adjusted_p": acc["adjusted_p_value"]})
        else:
            figure2.append({"metric_output_id": metric, "metric_state": result["metric_state"], "ag": None, "ag_lower": None, "ag_upper": None, "acc": None, "acc_lower": None, "acc_upper": None, "pair_count": None, "strict_agreement_count": None, "strict_disagreement_count": None, "metric_tie_count": None, "downstream_tie_count": None, "double_tie_count": None, "d_ag": None, "d_ag_lower": None, "d_ag_upper": None, "d_ag_favorable": None, "d_ag_raw_p": None, "d_ag_adjusted_p": 1.0, "d_acc": None, "d_acc_lower": None, "d_acc_upper": None, "d_acc_favorable": None, "d_acc_raw_p": None, "d_acc_adjusted_p": 1.0})
    figure1 = []
    for metric in METRIC_OUTPUT_IDS:
        current = [row for row in rows if row["metric_output_id"] == metric]
        for perturbation in config.perturbation_ids:
            for alpha in config.alpha_grid[1:]:
                selected = [row for row in current if row["perturbation_id"] == perturbation and row["alpha"] == alpha]; figure1.append({"metric_output_id": metric, "perturbation_id": perturbation, "alpha": alpha, "mean_metric_harm": float(np.mean([row["metric_harm"] for row in selected])), "mean_downstream_harm": float(np.mean([row["downstream_harm"] for row in selected])), "metric_state": "complete"})
    return [results[metric] for metric in METRIC_OUTPUT_IDS], bootstrap_rows, sign_rows, family, figure1, figure2


def _padding(values: Sequence[float]) -> tuple[float, float]:
    minimum = min(0.0, *values); maximum = max(0.0, *values); width = maximum - minimum; pad = 0.05 * (width if width > 0 else max(1.0, abs(minimum), abs(maximum))); return minimum - pad, maximum + pad


def _figures(figure1: Sequence[Mapping[str, object]], figure2: Sequence[Mapping[str, object]]) -> dict[str, bytes]:
    colors = dict(zip(PERTURBATION_IDS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"), strict=True)); rc = {"font.family": "DejaVu Sans", "svg.hashsalt": "rpe-phase4-d5-protocol-b-v1", "figure.dpi": 300, "savefig.dpi": 300}
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharey=True); y_values = [float(row["mean_downstream_harm"]) for row in figure1]; y_lim = _padding(y_values)
        for metric_index, metric in enumerate(METRIC_OUTPUT_IDS):
            ax = axes.flat[metric_index]; selected_metric = [row for row in figure1 if row["metric_output_id"] == metric]
            for perturbation in PERTURBATION_IDS:
                selected = [row for row in selected_metric if row["perturbation_id"] == perturbation]; ax.plot([float(row["mean_metric_harm"]) for row in selected], [float(row["mean_downstream_harm"]) for row in selected], marker="o", linewidth=1.5, color=colors[perturbation], label=perturbation.upper())
            ax.set_xlim(*_padding([float(row["mean_metric_harm"]) for row in selected_metric])); ax.set_ylim(*y_lim); ax.set_title(metric); ax.axhline(0.0, color="#999999", linewidth=0.5); ax.axvline(0.0, color="#999999", linewidth=0.5)
        for index in range(13, 16): axes.flat[index].axis("off")
        axes.flat[0].legend(loc="best", fontsize=7); fig.suptitle("D5 / matched-reference Protocol B / full_domain_core / P8–P12"); fig.tight_layout(); f1_png = io.BytesIO(); f1_svg = io.BytesIO(); fig.savefig(f1_png, format="png", dpi=300, metadata={"Software": "raman-preproc-eval"}); fig.savefig(f1_svg, format="svg", metadata={"Date": None}); plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True); specs = (("ag", "AG"), ("acc", "Acc-cross"), ("d_ag", "D_AG"), ("d_acc", "D_Acc")); y = np.arange(13)
        for ax, (key, title) in zip(axes, specs, strict=True):
            finite = []
            for index, row in enumerate(figure2):
                if row.get(key) is None: continue
                value = float(row[key]); low = float(row[f"{key}_lower"]); high = float(row[f"{key}_upper"]); finite.extend((low, value, high)); ax.errorbar(value, index, xerr=[[value - low], [high - value]], fmt="o", color="#1f77b4")
                if key in {"d_ag", "d_acc"}:
                    direction = "favorable" if bool(row[f"{key}_favorable"]) else "unfavorable"; ax.text(0.02, index, f"raw={float(row[f'{key}_raw_p']):.6g}; adj={float(row[f'{key}_adjusted_p']):.6g}; {direction}", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=4.5)
            if finite: ax.set_xlim(*_padding(finite))
            ax.axvline(0.0, color="#999999", linewidth=0.5); ax.set_title(title)
        axes[0].set_yticks(y, METRIC_OUTPUT_IDS); fig.suptitle("D5 / matched-reference Protocol B / full_domain_core / secondary"); fig.tight_layout(); f2_png = io.BytesIO(); f2_svg = io.BytesIO(); fig.savefig(f2_png, format="png", dpi=300, metadata={"Software": "raman-preproc-eval"}); fig.savefig(f2_svg, format="svg", metadata={"Date": None}); plt.close(fig)
    return {"figure1_d5_protocol_b_full_domain.png": f1_png.getvalue(), "figure1_d5_protocol_b_full_domain.svg": f1_svg.getvalue(), "figure1_d5_protocol_b_full_domain_data.csv": _csv(figure1), "figure2_d5_protocol_b_full_domain.png": f2_png.getvalue(), "figure2_d5_protocol_b_full_domain.svg": f2_svg.getvalue(), "figure2_d5_protocol_b_full_domain_data.csv": _csv(figure2)}


def _condition_summary(predictions: Sequence[Mapping[str, object]], conditions: Sequence[str]) -> list[dict[str, object]]:
    output = []
    for condition in conditions:
        selected = [row for row in predictions if row["condition_id"] == condition]; labels = sorted({int(row["class_label"]) for row in selected})
        output.append({"condition_id": condition, "query_occurrence_count": len(selected), "top1_macro_class_accuracy": float(np.mean([np.mean([bool(row["top1_correct"]) for row in selected if int(row["class_label"]) == label]) for label in labels])), "top5_macro_class_accuracy": float(np.mean([np.mean([bool(row["top5_correct"]) for row in selected if int(row["class_label"]) == label]) for label in labels])), "top1_micro_accuracy": float(np.mean([bool(row["top1_correct"]) for row in selected])), "top5_micro_accuracy": float(np.mean([bool(row["top5_correct"]) for row in selected]))})
    return output


def _run_identity(config: Phase4D5ProtocolBConfig, code: Mapping[str, object], environment: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    authority_path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    identity = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "authorities": dict(config.authorities), "parent_artifacts": _ready(config.parent_artifacts), "authority_bridge": {"condition_bridge_sha256": config.condition_bridge_sha256, "operator_bridge_sha256": config.operator_bridge_sha256}, "claim_boundary": config.claim_boundary, "code": _ready(code), "config_authority": {"bytes": authority_path.stat().st_size, "sha256": _sha(authority_path.read_bytes())}, "config_sha256": config.sha256, "environment": _ready(environment), "figure_contract": _ready(config.figure_contract), "frozen_identities": _ready(config.frozen_identities), "inference": {"bootstrap_resamples": config.bootstrap_resamples, "sign_flip_resamples": config.sign_flip_resamples, "random_seed": config.random_seed, "holm_slots": config.expected_holm_slot_count}, "metric_output_ids": list(config.metric_output_ids), "protocol": "B", "tier": config.tier}
    return RUN_PREFIX + _sha(_canon(identity)), identity


def _environment() -> dict[str, object]:
    import h5py, scipy, sklearn, threadpoolctl
    return {"h5py": h5py.__version__, "machine": platform.machine(), "matplotlib": matplotlib.__version__, "numpy": np.__version__, "python": platform.python_version(), "scikit_learn": sklearn.__version__, "scipy": scipy.__version__, "system": platform.system(), "threadpoolctl": threadpoolctl.__version__}


def _code() -> dict[str, object]:
    return {relative: {"bytes": (ROOT / relative).stat().st_size, "sha256": _sha((ROOT / relative).read_bytes())} for relative in CODE_RELATIVE_PATHS}


def _payloads(
    *, cohort: D5RawCohort, native: tuple[Spectrum1D, ...], sweep: PerturbationSweepConfig, phase1: Phase1CoreConfig, config: Phase4D5ProtocolBConfig, protocol_a: Path, eligibility: Path, workers: int, override: int | None
) -> tuple[dict[str, bytes], Phase4D5ProtocolBSummary]:
    bridge, metrics = _bridge(cohort, protocol_a, eligibility, config); cells, conditions, projected = _rematerialize(cohort, native, sweep, phase1, config, workers); cell_bytes = _jsonl(cells); condition_bytes = _jsonl(conditions)
    if cell_bytes != (eligibility / "operator_cells.jsonl").read_bytes() or condition_bytes != (eligibility / "record_conditions.jsonl").read_bytes(): raise Phase4D5ProtocolBError("independent rematerialization", "parent byte mismatch")
    downstream = [{"record_id": row["record_id"], "condition_id": row["condition_id"], "values_sha256": row["support_projection_sha256"]} for row in conditions]; predictions = _predict(cohort, projected, config); classes = _class_rows(predictions, metrics, config)
    alignment, bootstraps, signs, family, figure1, figure2 = _stats(classes, config, override); figures = _figures(figure1, figure2); summaries = _condition_summary(predictions, config.condition_ids)
    code = dict(config.code_authority) if config.synthetic_fixture else _code(); environment = dict(config.environment_authority) if config.synthetic_fixture and config.environment_authority else _environment(); run_id, identity = _run_identity(config, code, environment)
    manifest = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "claim_boundary": config.claim_boundary, "code": _ready(code), "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256}, "counts": {"operator_cells": len(cells), "record_conditions": len(conditions), "downstream_rows": len(downstream), "matcher_predictions": len(predictions), "class_observations": len(classes), "alignment_results": len(alignment), "bootstrap_results": len(bootstraps), "sign_flip_results": len(signs), "holm_family": len(family)}, "environment": environment, "experiment_id": EXPERIMENT_ID, "inherited_rulings": _ready(config.inherited_rulings), "metric_authority": _ready(bridge["metric_authority"]), "perturbation_ids": list(config.perturbation_ids), "protocol": "B", "run_id": run_id, "run_identity": identity, "status": "complete", "synthetic_fixture": config.synthetic_fixture, "tier": config.tier}
    preflight = {"authority_bridge_state": "complete", "rematerialization_state": "complete", "metric_authority_state": "complete", "matcher_state": "complete", "claim_boundary": "pre_matcher_bridges_and_rematerialization_complete"}; complete = {"artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "run_id": run_id, "status": "complete"}
    payloads = {"config.json": config.raw_bytes, "authority_bridge.json": _canon(bridge), "preflight.json": _canon(preflight), "operator_cells.jsonl": cell_bytes, "record_conditions.jsonl": condition_bytes, "downstream_rows.jsonl": _jsonl(downstream), "matcher_predictions.jsonl": _jsonl(predictions), "condition_summary.csv": _csv(summaries), "class_observations.jsonl": _jsonl(classes), "alignment_results.jsonl": _jsonl(alignment), "bootstrap_results.jsonl": _jsonl(bootstraps), "sign_flip_results.jsonl": _jsonl(signs), "holm_family.jsonl": _jsonl(family), "d5_protocol_b_full_domain_secondary_table.csv": _csv(figure2), "manifest.json": _canon(manifest), "complete.json": _canon(complete), **figures}
    return payloads, Phase4D5ProtocolBSummary(Path("."), run_id, "complete", config.full_record_count, len(predictions), len(classes))


def _tree(path: Path) -> dict[str, bytes]:
    return {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}


def verify_phase4_d5_protocol_b_from_inputs(
    path: Path, *, cohort: D5RawCohort, native_spectra: tuple[Spectrum1D, ...], sweep: PerturbationSweepConfig, phase1_config: Phase1CoreConfig, protocol_a_path: Path, eligibility_path: Path, worker_count: int, inference_resamples: int | None = None
) -> Phase4D5ProtocolBSummary:
    path = Path(path)
    if not path.is_dir(): raise Phase4D5ProtocolBError("independent verifier", "run path missing")
    raw = (path / "config.json").read_bytes(); document = json.loads(raw); config = parse_phase4_d5_protocol_b_config(path / "config.json", raw, require_frozen_identity=not bool(document.get("synthetic_fixture", False)))
    payloads, summary = _payloads(cohort=cohort, native=native_spectra, sweep=sweep, phase1=phase1_config, config=config, protocol_a=Path(protocol_a_path), eligibility=Path(eligibility_path), workers=worker_count, override=inference_resamples)
    with tempfile.TemporaryDirectory(prefix="phase4-d5-protocol-b-verify-") as temporary:
        rebuilt = Path(temporary) / summary.run_id; rebuilt.mkdir(); names = (*ARTIFACT_PAYLOAD_FILES, "complete.json")
        for name in names: (rebuilt / name).write_bytes(payloads[name])
        (rebuilt / "SHA256SUMS").write_text("".join(f"{_sha(payloads[name])}  {name}\n" for name in names), encoding="utf-8")
        observed_tree = _tree(path)
        rebuilt_tree = _tree(rebuilt)
        if observed_tree != rebuilt_tree:
            differing = sorted(
                name
                for name in set(observed_tree) | set(rebuilt_tree)
                if observed_tree.get(name) != rebuilt_tree.get(name)
            )
            raise Phase4D5ProtocolBError("independent verifier", f"artifact bytes differ: {differing!r}")
    return Phase4D5ProtocolBSummary(path, summary.run_id, summary.status, summary.full_record_count, summary.prediction_row_count, summary.class_observation_count)


def verify_phase4_d5_protocol_b(path: Path, *, worker_count: int = 12) -> Phase4D5ProtocolBSummary:
    run = Path(path); raw = (run / "config.json").read_bytes(); document = json.loads(raw); config = parse_phase4_d5_protocol_b_config(run / "config.json", raw, require_frozen_identity=not bool(document.get("synthetic_fixture", False)))
    if config.synthetic_fixture: raise Phase4D5ProtocolBError("independent verifier", "synthetic run requires from_inputs")
    cohort = load_d5_raw_cohort(ROOT / D5_CONFIG_RELATIVE_PATH, ROOT / DATASET_RELATIVE_PATH); native = load_d5_native_spectra(ROOT / DATASET_RELATIVE_PATH, cohort.record_ids); sweep = load_perturbation_sweep_config(ROOT / SWEEP_RELATIVE_PATH); phase1 = load_phase1_core_config(ROOT / PHASE1_CONFIG_RELATIVE_PATH)
    return verify_phase4_d5_protocol_b_from_inputs(run, cohort=cohort, native_spectra=native, sweep=sweep, phase1_config=phase1, protocol_a_path=ROOT / PROTOCOL_A_RELATIVE_PATH, eligibility_path=ROOT / ELIGIBILITY_RELATIVE_PATH, worker_count=worker_count)


__all__ = ["verify_phase4_d5_protocol_b", "verify_phase4_d5_protocol_b_from_inputs"]
