from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy import interpolate, stats

from rpe.alignment import AlignmentObservation, alignment_gap, cross_perturbation_accuracy

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/appendix_audits_v1.json"
DOMAIN = b"rpe-step7-exact-spectrum-v1"
_IDENTITY_CODE_FILES = (
    "rpe/runner/phase6_appendix_audits.py",
    "rpe/runner/phase6_appendix_audits_verifier.py",
    "rpe/runner/phase6_appendix_bacteria.py",
    "rpe/runner/phase6_appendix_sugar.py",
    "rpe/runner/phase6_appendix_rruff.py",
    "rpe/runner/phase6_appendix_leakage.py",
    "rpe/runner/phase6_appendix_metric_core.py",
    "rpe/runner/phase6_appendix_verifier_science.py",
    "rpe/runner/phase6_appendix_verify_bacteria.py",
    "rpe/runner/phase6_appendix_verify_sugar.py",
    "rpe/runner/phase6_appendix_verify_rruff.py",
    "rpe/runner/phase6_appendix_verify_leakage.py",
    "tools/run_phase6_appendix_audits.py",
    "experiments/phase6/configs/appendix_audits_v1.json",
)


class AppendixAuditsError(ValueError):
    def __init__(self, path: str, reason: str | None = None) -> None:
        self.path, self.reason = path, reason if reason is not None else path
        super().__init__(f"{path}: {reason}" if reason is not None else path)


@dataclass(frozen=True)
class AppendixAuditsConfig:
    raw: Mapping[str, object]
    path: Path
    raw_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class AppendixAuditsInputs:
    panel_inputs: tuple[object, ...]
    phase6_peak_rows: tuple[Mapping[str, object], ...]
    leakage_inputs: tuple[object, ...]
    authority_bridge: Mapping[str, object]
    identity: Mapping[str, object]


@dataclass(frozen=True)
class AppendixAuditsSummary:
    run_id: str
    path: Path
    status: str
    counts: Mapping[str, int]


@dataclass(frozen=True)
class Phase4SensitivityRows:
    """Aggregate-only Phase-4 result retained for the Task-2B seam.

    Native rematerializers own spectra and metric evaluation; this value owns no
    spectra and consequently cannot be mistaken for a downstream-model input.
    """

    status_rows: tuple[Mapping[str, object], ...]
    alignment_rows: tuple[Mapping[str, object], ...]
    rank_rows: tuple[Mapping[str, object], ...]


def _canonical(value: object) -> bytes:
    return (json.dumps(_plain_json(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _plain_json(value: object) -> object:
    """Recursively normalize immutable and NumPy values before canonical JSON."""
    if isinstance(value, AbcMapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_plain_json(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, list)):
        return [_plain_json(item) for item in value]
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_phase6_appendix_audits_config(path: Path = DEFAULT_CONFIG) -> AppendixAuditsConfig:
    path, raw = Path(path), Path(path).read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AppendixAuditsError("config", "invalid UTF-8 JSON") from error
    if not isinstance(document, dict) or raw != _canonical(document):
        raise AppendixAuditsError("config", "must be canonical sorted-key JSON")
    panels = ("d5_a", "d5_b", "d2_5_a", "d2_5_b", "d2_10_a", "d2_10_b", "d2_20_a", "d2_20_b", "d1_a", "d1_b_closed", "d4_a", "d4_b")
    metrics = ("mse", "rmse", "mae", "sam", "pearson_r", "nmse", "wasserstein_1_cm1", "is_like_structure_to_noise", "precision", "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio")
    counts = {"audit_status":259,"leakage_boundaries":15,"normalization_alignment":572,"normalization_rank_stability":88,"peak_tolerance_phase4_alignment":572,"peak_tolerance_phase4_rank_stability":88,"peak_tolerance_phase6_rank_stability":27,"peak_tolerance_phase6_system":840,"resampling_alignment":1287,"resampling_rank_stability":198}
    if document.get("schema_version") != "phase6-appendix-audits-v1":
        raise AppendixAuditsError("schema_version", "unexpected value")
    if tuple(document.get("fixed_panel_order", ())) != panels or tuple(document.get("metric_ids", ())) != metrics:
        raise AppendixAuditsError("config", "frozen panel or metric order mismatch")
    if document.get("expected_rows") != counts:
        raise AppendixAuditsError("expected_rows", "frozen counts mismatch")
    artifact = document.get("artifact_contract", {})
    if artifact.get("configured_payload_count") != 16 or len(artifact.get("payload_files", ())) != 16:
        raise AppendixAuditsError("artifact_contract", "invalid payload inventory")
    if sum(document.get("status_row_counts_by_component", {}).values()) != 259:
        raise AppendixAuditsError("status rows", "must total 259")
    return AppendixAuditsConfig(MappingProxyType(document), path, raw, _sha(raw))


def _array(name: str, values: object) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 2 or not np.isfinite(result).all():
        raise AppendixAuditsError(name, "must be a finite one-dimensional vector")
    return result


def target_axis(start: float, spacing: float, point_count: int) -> np.ndarray:
    if not math.isfinite(start) or not math.isfinite(spacing) or spacing <= 0 or point_count < 2:
        raise AppendixAuditsError("target_axis", "invalid parameters")
    return np.ascontiguousarray(np.float64(start) + np.float64(spacing) * np.arange(point_count, dtype="<f8"), dtype="<f8")


def resample_spectrum(axis: object, intensity: object, target_axis: object, *, interpolator: str, max_in_range_native_gap_cm1: float | None = None) -> np.ndarray:
    x, y, target = _array("axis", axis), _array("intensity", intensity), _array("target_axis", target_axis)
    if x.shape != y.shape or not np.all(np.diff(x) > 0.0):
        raise AppendixAuditsError("axis", "shape must match and axis must strictly increase")
    if not np.all(np.diff(target) > 0.0) or target[0] < x[0] or target[-1] > x[-1]:
        raise AppendixAuditsError("target_axis", "source support must bracket target; extrapolation forbidden")
    if max_in_range_native_gap_cm1 is not None:
        if not math.isfinite(max_in_range_native_gap_cm1) or max_in_range_native_gap_cm1 <= 0.0:
            raise AppendixAuditsError("max_in_range_native_gap_cm1", "must be finite and positive")
        native_in_range = x[(x >= target[0]) & (x <= target[-1])]
        if native_in_range.size < 2 or float(np.max(np.diff(native_in_range))) > max_in_range_native_gap_cm1:
            raise AppendixAuditsError("max_in_range_native_gap_cm1", "native gap exceeds frozen maximum")
    if interpolator == "linear":
        result = np.interp(target, x, y)
    elif interpolator == "cubic":
        result = interpolate.CubicSpline(x, y, bc_type="not-a-knot", extrapolate=False)(target)
    elif interpolator == "pchip":
        result = interpolate.PchipInterpolator(x, y, extrapolate=False)(target)
    else:
        raise AppendixAuditsError("interpolator", "unsupported")
    result = np.ascontiguousarray(result, dtype="<f8")
    if result.shape != target.shape or not np.isfinite(result).all():
        raise AppendixAuditsError("resampling", "nonfinite output or shape drift")
    return result


def normalize_spectrum(axis: object, intensity: object, *, method: str) -> np.ndarray:
    x, y = _array("axis", axis), _array("intensity", intensity)
    if x.shape != y.shape or not np.all(np.diff(x) > 0.0):
        raise AppendixAuditsError("normalization domain", "invalid axis")
    if method == "none": result = y.copy()
    elif method == "maximum":
        denominator = float(np.max(y))
        if denominator <= 0.0: raise AppendixAuditsError("normalization domain", "maximum must be positive")
        result = y / denominator
    elif method == "area":
        denominator = float(np.trapezoid(np.abs(y), x))
        if denominator <= 0.0: raise AppendixAuditsError("normalization domain", "area must be positive")
        result = y / denominator
    elif method == "snv":
        denominator = float(np.std(y, ddof=0))
        if denominator <= 0.0: raise AppendixAuditsError("normalization domain", "std must be positive; constant spectrum")
        result = (y - float(np.mean(y))) / denominator
    else: raise AppendixAuditsError("normalization method", "unsupported")
    if not np.isfinite(result).all(): raise AppendixAuditsError("normalization domain", "nonfinite result")
    return np.ascontiguousarray(result, dtype="<f8")


def scale_invariance_passes(none_value: float, transformed_value: float) -> bool:
    tolerance = 1e-12 + 1e-10 * abs(none_value)
    return bool(
        math.isfinite(none_value)
        and math.isfinite(transformed_value)
        and abs(transformed_value - none_value)
        <= tolerance
    )


def _ties(values: np.ndarray) -> int:
    _, counts = np.unique(values, return_counts=True)
    return int(np.sum(counts[counts > 1] - 1))


def rank_stability_rows(reference_by_metric: Mapping[str, object], candidate_by_metric: Mapping[str, object], *, statistic: str, metric_ids: Sequence[str]) -> dict[str, object]:
    ids = tuple(metric_ids)
    base = {"statistic":statistic,"tau_b":None,"reference_tie_count":None,"candidate_tie_count":None,"max_abs_rank_displacement":None,"stable":False}
    if statistic not in {"ag", "acc_cross"}: raise AppendixAuditsError("statistic", "unsupported")
    if set(reference_by_metric) != set(ids) or set(candidate_by_metric) != set(ids):
        return {**base,"state":"not_evaluable","reason_code":"not_evaluable_incomplete_metric_set"}
    reference = np.asarray([reference_by_metric[x] for x in ids], dtype=np.float64); candidate = np.asarray([candidate_by_metric[x] for x in ids], dtype=np.float64)
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all(): return {**base,"state":"not_evaluable","reason_code":"not_evaluable_incomplete_metric_set"}
    direction = 1.0 if statistic == "ag" else -1.0
    rr = stats.rankdata(direction*reference, method="average"); cr = stats.rankdata(direction*candidate, method="average")
    rt, ct = _ties(reference), _ties(candidate)
    if rt == len(ids)-1 or ct == len(ids)-1:
        return {**base,"reference_tie_count":rt,"candidate_tie_count":ct,"state":"not_evaluable","reason_code":"not_evaluable_all_tied_ranking"}
    tau = float(stats.kendalltau(rr, cr, variant="b", nan_policy="propagate").statistic)
    stable = tau > 0.9
    return {**base,"tau_b":tau,"reference_tie_count":rt,"candidate_tie_count":ct,"max_abs_rank_displacement":float(np.max(np.abs(rr-cr))),"stable":stable,"state":"complete_numeric","reason_code":"" if stable else "complete_ranking_instability_or_reversal"}


def _parent_downstream_by_condition(
    parent_rows: Sequence[Mapping[str, object]], *, metric_ids: Sequence[str]
) -> dict[tuple[str, str, float], float]:
    """Collapse immutable parent harms, rejecting metric-copy drift."""
    ids = tuple(metric_ids)
    grouped: dict[tuple[str, str, float], dict[str, float]] = {}
    for row in parent_rows:
        try:
            cluster_value = row.get("cluster_id")
            if cluster_value is None:
                cluster_value = row.get("class_label")
            if cluster_value is None:
                cluster_value = row["well_id"]
            cluster = str(cluster_value)
            key = (cluster, str(row["perturbation_id"]), float(row["alpha"]))
            metric = str(row["metric_output_id"])
            downstream = float(row["downstream_harm"])
        except (KeyError, TypeError, ValueError) as error:
            raise AppendixAuditsError("parent observations", "invalid row") from error
        if metric not in ids or not math.isfinite(downstream):
            raise AppendixAuditsError("parent observations", "invalid metric or downstream harm")
        values = grouped.setdefault(key, {})
        if metric in values:
            raise AppendixAuditsError("parent observations", "duplicate metric condition row")
        values[metric] = downstream
    output: dict[tuple[str, str, float], float] = {}
    for key, copies in grouped.items():
        if set(copies) != set(ids):
            raise AppendixAuditsError("parent observations", "incomplete 13-metric downstream grid")
        first = copies[ids[0]]
        if any(value != first for value in copies.values()):
            raise AppendixAuditsError("parent observations", "downstream harm differs across metric copies")
        output[key] = first
    return output


def assemble_phase4_sensitivity_rows(
    parent_rows: Sequence[Mapping[str, object]],
    metric_rows: Sequence[Mapping[str, object]],
    *,
    panel_id: str,
    endpoint_id: str,
    protocol_id: str,
    condition_id: str,
    metric_ids: Sequence[str],
    reference_by_metric: Mapping[str, tuple[float, float]],
    reference_condition_id: str = "native",
    require_identity: bool = False,
) -> Phase4SensitivityRows:
    """Pair reconstructed metric harms with immutable parent downstream harms.

    This deliberately accepts only aggregated record-side results.  It is used
    by dataset-specific native rematerializers after they have run P8--P12 and
    metric/CWT evaluation, so it has no route to predictions or model fitting.
    """
    ids = tuple(metric_ids)
    downstream = _parent_downstream_by_condition(parent_rows, metric_ids=ids)
    tables: dict[str, list[AlignmentObservation]] = {metric: [] for metric in ids}
    seen: set[tuple[str, str, float, str]] = set()
    for row in metric_rows:
        try:
            cluster_value = row.get("cluster_id")
            if cluster_value is None:
                cluster_value = row.get("class_label")
            if cluster_value is None:
                cluster_value = row["well_id"]
            cluster = str(cluster_value)
            perturbation, alpha, metric = str(row["perturbation_id"]), float(row["alpha"]), str(row["metric_output_id"])
            harm = float(row["metric_harm"])
        except (KeyError, TypeError, ValueError) as error:
            raise AppendixAuditsError("reconstructed metric rows", "invalid row") from error
        key = (cluster, perturbation, alpha, metric)
        if metric not in tables or key in seen or not math.isfinite(harm):
            raise AppendixAuditsError("reconstructed metric rows", "duplicate, nonfinite, or unexpected metric row")
        condition_key = key[:3]
        if condition_key not in downstream:
            raise AppendixAuditsError("reconstructed metric rows", "condition absent from parent downstream grid")
        seen.add(key)
        tables[metric].append(AlignmentObservation(cluster, perturbation, alpha, harm, downstream[condition_key]))
    expected = {(cluster, perturbation, alpha, metric) for cluster, perturbation, alpha in downstream for metric in ids}
    if seen != expected:
        raise AppendixAuditsError("reconstructed metric rows", "incomplete fixed cluster-condition-metric grid")
    ag: dict[str, float] = {}
    acc: dict[str, float] = {}
    for metric in ids:
        try:
            ag[metric] = alignment_gap(tables[metric]).alignment_gap
            acc[metric] = cross_perturbation_accuracy(tables[metric]).accuracy
        except Exception as error:
            raise AppendixAuditsError("phase4 aggregation", str(error)) from error
    if set(reference_by_metric) != set(ids):
        raise AppendixAuditsError("reference alignment", "incomplete metric set")
    if require_identity and any(ag[metric] != float(reference_by_metric[metric][0]) or acc[metric] != float(reference_by_metric[metric][1]) for metric in ids):
        raise AppendixAuditsError("identity", "tolerance-2 or normalization-none AG/Acc-cross mismatch")
    alignment_rows = tuple({
        "panel_id": panel_id, "endpoint_id": endpoint_id, "protocol_id": protocol_id,
        "condition_id": condition_id, "metric_output_id": metric, "ag": ag[metric],
        "acc_cross": acc[metric], "state": "complete_numeric", "reason_code": "",
    } for metric in ids)
    rank_rows = tuple({
        "panel_id": panel_id, "endpoint_id": endpoint_id, "protocol_id": protocol_id,
        "condition_id": condition_id, "reference_condition_id": reference_condition_id, **rank_stability_rows(
            {metric: reference_by_metric[metric][index] for metric in ids},
            values, statistic=statistic, metric_ids=ids,
        )
    } for index, (statistic, values) in enumerate((("ag", ag), ("acc_cross", acc))))
    status = ({
        "component": "phase4_sensitivity", "status_order": 0, "panel_id": panel_id,
        "endpoint_id": endpoint_id, "protocol_id": protocol_id, "condition_id": condition_id,
        "state": "complete_numeric", "reason_code": "", "numerical_row_count": len(alignment_rows),
        "rank_row_count": len(rank_rows), "evidence_key": "native_metric_reconstruction",
    },)
    return Phase4SensitivityRows(status, alignment_rows, rank_rows)


def exact_spectrum_sha256(axis: object, intensity: object) -> str:
    x, y = np.asarray(axis,dtype="<f8"), np.asarray(intensity,dtype="<f4")
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or not np.isfinite(x).all() or not np.isfinite(y).all(): raise AppendixAuditsError("exact hash", "invalid spectrum")
    return _sha(DOMAIN + np.uint64(x.size).astype("<u8").tobytes() + x.tobytes() + np.uint64(y.size).astype("<u8").tobytes() + y.tobytes())


def select_calibration_pairs(record_ids: Sequence[object], entity_ids: Sequence[object], *, seed: int, target_count: int, batch_size: int, max_batches: int) -> tuple[tuple[object, object], ...]:
    records, entities = tuple(record_ids), tuple(entity_ids)
    if len(records) != len(entities) or len(set(records)) != len(records): raise AppendixAuditsError("calibration", "record/entity IDs must align and records be unique")
    if min(target_count,batch_size,max_batches) <= 0: raise AppendixAuditsError("calibration", "counts must be positive")
    entity_counts: dict[object, int] = {}
    for entity in entities:
        entity_counts[entity] = entity_counts.get(entity, 0) + 1
    admissible = len(records) * (len(records) - 1) // 2 - sum(count * (count - 1) // 2 for count in entity_counts.values())
    if admissible < target_count:
        return tuple(sorted((records[i],records[j]) for i in range(len(records)) for j in range(i+1,len(records)) if entities[i] != entities[j]))
    rng = np.random.Generator(np.random.PCG64(seed)); selected: set[tuple[object,object]] = set(); accepted: list[tuple[object, object]] = []
    for _ in range(max_batches):
        left=rng.integers(0,len(records),size=batch_size); right=rng.integers(0,len(records),size=batch_size)
        for i,j in zip(left,right,strict=True):
            i,j=int(i),int(j)
            if i==j or entities[i]==entities[j]: continue
            pair=(records[i],records[j]) if i<j else (records[j],records[i])
            if pair in selected: continue
            selected.add(pair); accepted.append(pair)
            if len(accepted)==target_count: return tuple(accepted)
    raise AppendixAuditsError("calibration", "max batches did not fill target")


def _snv_record(record: Mapping[str,object]) -> tuple[np.ndarray|None,str]:
    try:
        axis,intensity=_array("axis",record["axis"]),_array("intensity",record["intensity"])
        if axis.shape != intensity.shape: raise AppendixAuditsError("shape","mismatch")
        std=float(np.std(intensity,ddof=0))
        if std<=0 or not math.isfinite(std): return None,"typed_closure_constant_or_nonfinite_spectrum"
        return np.ascontiguousarray((intensity-intensity.mean())/std),""
    except (AppendixAuditsError,KeyError,TypeError,ValueError): return None,"typed_closure_constant_or_nonfinite_spectrum"


def scan_near_duplicates(left_records: Sequence[Mapping[str,object]], right_records: Sequence[Mapping[str,object]], *, threshold: float, query_block_size: int, fit_block_size: int) -> tuple[dict[str,object],...]:
    if not math.isfinite(threshold) or threshold<0 or min(query_block_size,fit_block_size)<=0: raise AppendixAuditsError("near scan","invalid controls")
    left,right=tuple(left_records),tuple(right_records); ln=[_snv_record(x) for x in left]; rn=[_snv_record(x) for x in right]; rows=[]
    for side,records,normal in (("left",left,ln),("right",right,rn)):
        for record,(vector,reason) in zip(records,normal,strict=True):
            if vector is None:
                rows.append({"left_role_id":str(record.get("role_id","")) if side=="left" else "","right_role_id":str(record.get("role_id","")) if side=="right" else "","left_record_id":str(record.get("record_id","")) if side=="left" else "","right_record_id":str(record.get("record_id","")) if side=="right" else "","left_entity_id":str(record.get("entity_id","")) if side=="left" else "","right_entity_id":str(record.get("entity_id","")) if side=="right" else "","distance":None,"correlation":None,"left_source_sha256":str(record.get("source_sha256","")) if side=="left" else "","right_source_sha256":str(record.get("source_sha256","")) if side=="right" else "","status":"not_evaluable","reason_code":reason})
    for li in range(0,len(left),fit_block_size):
        for ri in range(0,len(right),query_block_size):
            lv=[(i,ln[i][0]) for i in range(li,min(li+fit_block_size,len(left))) if ln[i][0] is not None]; rv=[(j,rn[j][0]) for j in range(ri,min(ri+query_block_size,len(right))) if rn[j][0] is not None]
            if not lv or not rv: continue
            if len({v.size for _,v in (*lv,*rv)}) != 1: raise AppendixAuditsError("near scan","supports differ")
            lm=np.stack([v for _,v in lv]); rm=np.stack([v for _,v in rv]); squared=2.0-2.0*(lm@rm.T)/lm.shape[1]; squared[squared<0]=0
            for ai,bj in np.argwhere(squared<=threshold*threshold):
                i,j=lv[int(ai)][0],rv[int(bj)][0]; direct=float(np.sqrt(np.mean((ln[i][0]-rn[j][0])**2)))
                if direct<=threshold:
                    l,r=left[i],right[j]; rows.append({"left_role_id":str(l.get("role_id","")),"right_role_id":str(r.get("role_id","")),"left_record_id":str(l["record_id"]),"right_record_id":str(r["record_id"]),"left_entity_id":str(l["entity_id"]),"right_entity_id":str(r["entity_id"]),"distance":direct,"correlation":float(np.mean(ln[i][0]*rn[j][0])),"left_source_sha256":str(l.get("source_sha256","")),"right_source_sha256":str(r.get("source_sha256","")),"status":"warning_near_duplicate_bridge","reason_code":"warning_near_duplicate_bridge"})
    keys=("left_role_id","right_role_id","left_record_id","right_record_id","reason_code")
    return tuple(sorted(rows,key=lambda row:tuple(str(row[k]) for k in keys)))


def _curve_value(row: Mapping[str,object], tolerance: int) -> tuple[object,object,object,object,object,object]:
    value=row["curve_by_tolerance_cm1"][str(tolerance)]
    if isinstance(value,Mapping): return value["estimate"],value["interval_lower"],value["interval_upper"],value.get("alpha_estimates",()),value.get("contributing_class_counts",()),value.get("state",row.get("state","complete"))
    interval=row["interval_by_tolerance_cm1"][str(tolerance)]
    return value,interval[0],interval[1],row.get("alpha_estimates_json",{}),row.get("contributing_class_counts_json",{}),row.get("state","complete")


def project_phase6_peak_tolerance_rows(rows: Sequence[Mapping[str,object]], *, system_ids: Sequence[str], endpoint_manifest: Sequence[Mapping[str,object]], tolerances: Sequence[int]) -> tuple[tuple[dict[str,object],...],tuple[dict[str,object],...]]:
    systems,endpoints,tolerances=tuple(system_ids),tuple(endpoint_manifest),tuple(tolerances); by_key={(str(r["system_id"]),str(r["endpoint_id"])):r for r in rows}
    if set(by_key)!={(s,str(e["endpoint_id"])) for s in systems for e in endpoints}: raise AppendixAuditsError("phase6 peak rows","all systems/endpoints required; subset ranking forbidden")
    output=[]
    for system in systems:
        for endpoint in endpoints:
            source=by_key[(system,str(endpoint["endpoint_id"]))]
            for tolerance in tolerances:
                estimate,lower,upper,alpha,counts,state=_curve_value(source,tolerance)
                output.append({"system_id":system,"family_id":source["family_id"],"method_id":source["method_id"],"endpoint_id":endpoint["endpoint_id"],"preferred_direction":endpoint["preferred_direction"],"tolerance_cm1":tolerance,"estimate":estimate,"interval_lower":lower,"interval_upper":upper,"design_class_count":source["design_class_count"],"alpha_estimates_json":alpha,"contributing_class_counts_json":counts,"state":state,"reason_code":source.get("reason_code","")})
    ranks=[]
    for endpoint in endpoints:
        if endpoint["preferred_direction"]=="non_monotonic": continue
        eid=str(endpoint["endpoint_id"]); reference={s:float(_curve_value(by_key[(s,eid)],2)[0]) for s in systems}; statistic="acc_cross" if endpoint["preferred_direction"]=="higher_is_better" else "ag"
        for tolerance in tolerances:
            if tolerance==2: continue
            candidate={s:float(_curve_value(by_key[(s,eid)],tolerance)[0]) for s in systems}; rank=rank_stability_rows(reference,candidate,statistic=statistic,metric_ids=systems)
            ranks.append({"endpoint_id":eid,"preferred_direction":endpoint["preferred_direction"],"tolerance_cm1":tolerance,"reference_tolerance_cm1":2,**{k:v for k,v in rank.items() if k!="statistic"},"system_count":len(systems)})
    return tuple(output),tuple(ranks)


def _validate_ledger(run: Path, expected_digest: str | None = None, expected_files: Mapping[str,str] | None = None, *, terminal_name: str = "complete.json") -> dict[str,str]:
    ledger=run/"SHA256SUMS"
    if not ledger.is_file() or ledger.is_symlink() or (expected_digest and _sha_file(ledger)!=expected_digest): raise AppendixAuditsError(str(ledger),"missing or checksum mismatch")
    entries={}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        fields=line.split("  ",1)
        if len(fields)!=2 or len(fields[0])!=64 or fields[1] in entries: raise AppendixAuditsError(str(ledger),"malformed")
        entries[fields[1]]=fields[0]
    for name,digest in entries.items():
        member=run/name
        if not member.is_file() or member.is_symlink() or _sha_file(member)!=digest: raise AppendixAuditsError(str(member),"ledger member mismatch")
    if expected_files is not None and entries!=dict(expected_files): raise AppendixAuditsError(str(ledger),"verified inventory mismatch")
    terminal = run / terminal_name
    if terminal_name not in {"complete.json", "failed.json"} or not terminal.is_file():
        raise AppendixAuditsError(str(run), "declared authoritative terminal required")
    other = run / ("failed.json" if terminal_name == "complete.json" else "complete.json")
    if other.exists():
        raise AppendixAuditsError(str(run), "ambiguous terminal markers")
    return entries


def _validate_authorities(config: AppendixAuditsConfig, root: Path) -> Mapping[str,object]:
    receipts={}
    for name,raw in config.raw["authorities"].items():
        identity=dict(raw); relative=str(identity["path"]); lowered=relative.lower()
        if Path(relative).is_absolute() or any(label in lowered for label in ("non_authoritative","pre_","interrupted","superseded")): raise AppendixAuditsError(f"authorities.{name}.path","non-authoritative label")
        path=root/relative
        if not path.is_file() or path.stat().st_size!=int(identity["byte_count"]) or _sha_file(path)!=identity["sha256"]: raise AppendixAuditsError(f"authorities.{name}","byte identity mismatch")
        receipts[name]=identity
    index_path=root/config.raw["authorities"]["final_phase4_parent_panel_index"]["path"]
    with index_path.open(newline="",encoding="utf-8") as stream: index_rows=tuple(csv.DictReader(stream))
    panels=tuple(dict(x) for x in config.raw["panel_manifest"])
    if tuple(x["panel_id"] for x in index_rows)!=tuple(x["panel_id"] for x in panels): raise AppendixAuditsError("parent panel index","order mismatch")
    for declared,indexed in zip(panels,index_rows,strict=True):
        if declared["parent_path"]!=indexed["relative_path"] or declared["parent_key"]!=indexed["parent_key"]: raise AppendixAuditsError("parent panel index","config/index mismatch")
        run=root/declared["parent_path"]
        terminal_name = str(indexed["terminal_name"])
        _validate_ledger(run, indexed["ledger_sha256"], terminal_name=terminal_name)
        marker=run/indexed["terminal_name"]
        if _sha_file(marker)!=indexed["terminal_sha256"]: raise AppendixAuditsError(str(marker),"terminal identity mismatch")
    peak_ledger=root/config.raw["authorities"]["phase6_peak_checksum_ledger"]["path"]
    _validate_ledger(peak_ledger.parent,expected_files=config.raw["phase6_peak_retained_payloads"]["verified_inventory_sha256"])
    receipts["phase4_parent_panels"] = tuple({
        "panel_id": row["panel_id"], "parent_path": row["relative_path"],
        "ledger_sha256": row["ledger_sha256"], "terminal_name": row["terminal_name"],
        "terminal_sha256": row["terminal_sha256"],
    } for row in index_rows)
    return MappingProxyType(receipts)


def _csv_bytes(rows: Sequence[Mapping[str,object]], fields: Sequence[str]) -> bytes:
    stream=io.StringIO(newline=""); writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n",extrasaction="raise"); writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v,sort_keys=True,separators=(",",":")) if isinstance(v,(dict,list,tuple)) else format(v,".17g") if isinstance(v,float) else v for k,v in row.items()})
    return stream.getvalue().encode()


def _jsonl(rows: Sequence[Mapping[str,object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _rows(inputs: AppendixAuditsInputs, key: str) -> tuple[Mapping[str,object],...]:
    value=inputs.identity.get(key,())
    if not isinstance(value,(tuple,list)): raise AppendixAuditsError(f"inputs.identity.{key}","must be rows")
    return tuple(dict(row) for row in value)


def _validate_table(name: str, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    required = set(fields)
    for index, row in enumerate(rows):
        if set(row) != required:
            raise AppendixAuditsError(name, f"row {index} schema mismatch")
        state = str(row.get("state", row.get("status", "")))
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value) and state in {"complete", "complete_numeric", "pass", "pass_via_parent_hash"}:
                raise AppendixAuditsError(name, f"row {index} has nonfinite {key} in completed state")
    keys = {
        "audit_status": ("component", "status_order"),
        "resampling_alignment": ("panel_id", "condition_id", "metric_output_id"),
        "normalization_alignment": ("panel_id", "condition_id", "metric_output_id"),
        "peak_tolerance_phase4_alignment": ("panel_id", "condition_id", "metric_output_id"),
        "resampling_rank_stability": ("panel_id", "condition_id", "statistic"),
        "normalization_rank_stability": ("panel_id", "condition_id", "statistic"),
        "peak_tolerance_phase4_rank_stability": ("panel_id", "condition_id", "statistic"),
        "peak_tolerance_phase6_system": ("system_id", "endpoint_id", "tolerance_cm1"),
        "peak_tolerance_phase6_rank_stability": ("endpoint_id", "tolerance_cm1"),
        "leakage_boundaries": ("boundary_id",),
        "leakage_duplicate_candidates": ("boundary_id", "left_role_id", "right_role_id", "left_record_id", "right_record_id"),
    }.get(name)
    if keys is not None and len({tuple(row[key] for key in keys) for row in rows}) != len(rows):
        raise AppendixAuditsError(name, "duplicate canonical key")


def build_phase6_appendix_audits_from_inputs(inputs: AppendixAuditsInputs, output_root: Path, *, config: AppendixAuditsConfig, worker_count: int) -> AppendixAuditsSummary:
    if worker_count<=0: raise AppendixAuditsError("worker_count","must be positive")
    identity={"schema_version":config.raw["schema_version"],"config_sha256":config.sha256,"authority_bridge":inputs.authority_bridge,"identity":inputs.identity.get("run_identity",{}),"schemas":config.raw["schemas"],"payload_order":config.raw["artifact_contract"]["payload_files"]}
    run_id=config.raw["artifact_contract"]["run_prefix"]+_sha(_canonical(identity)); path=Path(output_root)/run_id
    tables={name:_rows(inputs,name) for name in ("audit_status","resampling_alignment","resampling_rank_stability","normalization_alignment","normalization_rank_stability","peak_tolerance_phase4_alignment","peak_tolerance_phase4_rank_stability","leakage_duplicate_candidates")}
    tables["leakage_boundaries"]=tuple(dict(x) for x in inputs.leakage_inputs)
    system,ranks=project_phase6_peak_tolerance_rows(inputs.phase6_peak_rows,system_ids=tuple(inputs.identity["phase6_system_ids"]),endpoint_manifest=tuple(config.raw["phase6_endpoint_manifest"]),tolerances=(1,2,4,8)); tables["peak_tolerance_phase6_system"]=system; tables["peak_tolerance_phase6_rank_stability"]=ranks
    created_here = False
    try:
        for name,count in config.raw["expected_rows"].items():
            if len(tables[name])!=count: raise AppendixAuditsError(name,f"expected {count} rows, observed {len(tables[name])}")
        schemas=config.raw["schemas"]; payloads={"config.json":config.raw_bytes,"authority_bridge.json":_canonical(inputs.authority_bridge),"preflight.json":_canonical({"status":"passed","d1_b_state":"not_evaluable_failed_alpha0_equivalence","worker_count_identity_excluded":True})}
        csv_schemas={"audit_status":"audit_status_csv_columns","resampling_alignment":"phase4_alignment_csv_columns","resampling_rank_stability":"phase4_rank_csv_columns","normalization_alignment":"phase4_alignment_csv_columns","normalization_rank_stability":"phase4_rank_csv_columns","peak_tolerance_phase4_alignment":"phase4_alignment_csv_columns","peak_tolerance_phase4_rank_stability":"phase4_rank_csv_columns","peak_tolerance_phase6_system":"phase6_system_csv_columns","peak_tolerance_phase6_rank_stability":"phase6_rank_csv_columns"}
        for name,schema in csv_schemas.items(): payloads[f"{name}.csv"]=_csv_bytes(tables[name],schemas[schema])
        for name, schema in csv_schemas.items(): _validate_table(name, tables[name], schemas[schema])
        _validate_table("leakage_boundaries", tables["leakage_boundaries"], schemas["leakage_boundaries_jsonl_fields"])
        _validate_table("leakage_duplicate_candidates", tables["leakage_duplicate_candidates"], schemas["leakage_duplicate_candidates_jsonl_fields"])
        payloads["leakage_boundaries.jsonl"]=_jsonl(tables["leakage_boundaries"]); payloads["leakage_duplicate_candidates.jsonl"]=_jsonl(tables["leakage_duplicate_candidates"])
        counts={name:len(value) for name,value in tables.items()}; outcomes={}
        for name, rows in tables.items():
            states={}
            for row in rows:
                state=str(row.get("state", row.get("status", ""))); states[state]=states.get(state,0)+1
            outcomes[name]=states
        payloads["summary.md"]=("# Phase 6 Appendix Audits\n\nStatus: complete.\n\n## Tables\n\n"+"\n".join(f"- {key}: {value}" for key,value in counts.items())+"\n\n## Outcomes\n\n"+"\n".join(f"- {key}: {json.dumps(value, sort_keys=True)}" for key,value in outcomes.items())+"\n").encode(); payloads["manifest.json"]=_canonical({"schema_version":"phase6-appendix-audits-artifact-v1","run_id":run_id,"status":"complete","counts":counts,"fixed_expected_counts":config.raw["expected_rows"],"outcome_counts":outcomes,"identity":identity,"payload_files":config.raw["artifact_contract"]["payload_files"]})
        ordered=tuple(config.raw["artifact_contract"]["payload_files"])
        if set(payloads)!=set(ordered): raise AppendixAuditsError("payloads","inventory mismatch")
        all_files={**payloads,"complete.json":_canonical({"run_id":run_id,"status":"complete"})}; ledger=b"".join(f"{_sha(all_files[n])}  {n}\n".encode() for n in (*ordered,"complete.json")); expected_files={**all_files,"SHA256SUMS":ledger}
        if path.exists():
            observed={p.name:p.read_bytes() for p in path.iterdir() if p.is_file()}
            if observed!=expected_files: raise AppendixAuditsError("output","existing content path bytes differ")
        else:
            path.mkdir(parents=True)
            created_here = True
            for name in (*ordered,"complete.json"): (path/name).write_bytes(all_files[name])
            (path/"SHA256SUMS").write_bytes(ledger)
        return AppendixAuditsSummary(run_id,path,"complete",MappingProxyType(counts))
    except Exception as error:
        if created_here:
            shutil.rmtree(path)
            path.mkdir(parents=True)
            (path/"failed.json").write_bytes(_canonical({"run_id":run_id,"status":"failed","error_type":type(error).__name__,"reason":str(error)}))
        raise


def _adapter_tables(results: Sequence[object]) -> dict[str, tuple[Mapping[str, object], ...]]:
    names = (
        "audit_status", "resampling_alignment", "resampling_rank_stability",
        "normalization_alignment", "normalization_rank_stability",
        "peak_tolerance_phase4_alignment", "peak_tolerance_phase4_rank_stability",
    )
    return {name: tuple(dict(row) for result in results for row in getattr(result, name)) for name in names}


def _condition_sort(component: str, condition: object) -> tuple[int, int]:
    value = str(condition)
    if component == "resampling":
        try:
            interpolation, spacing = value.split(":", 1)
            return (("linear", "cubic", "pchip").index(interpolation), ("0.5", "1", "2").index(f"{float(spacing):g}"))
        except (ValueError, IndexError): raise AppendixAuditsError("resampling condition", "not in frozen order")
    if component == "normalization":
        try: return (("none", "maximum", "area", "snv").index(value), 0)
        except ValueError: raise AppendixAuditsError("normalization condition", "not in frozen order")
    if component == "phase4_peak_tolerance":
        try: return ((1, 2, 4, 8).index(int(value.split(":", 1)[1])), 0)
        except (ValueError, IndexError): raise AppendixAuditsError("peak tolerance condition", "not in frozen order")
    return (0, 0)


def _canonicalize_inputs(
    config: AppendixAuditsConfig, tables: Mapping[str, Sequence[Mapping[str, object]]],
    leakage: object, phase6_rows: Sequence[Mapping[str, object]], system_ids: Sequence[str],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    panel_order = {str(value): index for index, value in enumerate(config.raw["fixed_panel_order"])}
    metric_order = {str(value): index for index, value in enumerate(config.raw["metric_ids"])}
    system_order = {str(value): index for index, value in enumerate(system_ids)}
    endpoint_order = {str(row["endpoint_id"]): index for index, row in enumerate(config.raw["phase6_endpoint_manifest"])}
    boundary_order = {str(row["boundary_id"]): index for index, row in enumerate(config.raw["leakage_boundary_definitions"])}
    result = {name: tuple(dict(row) for row in rows) for name, rows in tables.items()}
    for component, alignment, ranks in (
        ("resampling", "resampling_alignment", "resampling_rank_stability"),
        ("normalization", "normalization_alignment", "normalization_rank_stability"),
        ("phase4_peak_tolerance", "peak_tolerance_phase4_alignment", "peak_tolerance_phase4_rank_stability"),
    ):
        base = lambda row: (panel_order.get(str(row.get("panel_id")), 999), *_condition_sort(component, row.get("condition_id")))
        result[alignment] = tuple(sorted(result[alignment], key=lambda row: (*base(row), metric_order.get(str(row.get("metric_output_id")), 999))))
        result[ranks] = tuple(sorted(result[ranks], key=lambda row: (*base(row), ("ag", "acc_cross").index(str(row.get("statistic"))))))
    result["peak_tolerance_phase6_system"], result["peak_tolerance_phase6_rank_stability"] = project_phase6_peak_tolerance_rows(phase6_rows, system_ids=system_ids, endpoint_manifest=tuple(config.raw["phase6_endpoint_manifest"]), tolerances=(1,2,4,8))
    result["peak_tolerance_phase6_system"] = tuple(sorted(result["peak_tolerance_phase6_system"], key=lambda row: (system_order[str(row["system_id"])], endpoint_order[str(row["endpoint_id"])], int(row["tolerance_cm1"]))))
    result["peak_tolerance_phase6_rank_stability"] = tuple(sorted(result["peak_tolerance_phase6_rank_stability"], key=lambda row: (endpoint_order[str(row["endpoint_id"])], int(row["tolerance_cm1"]))))
    result["leakage_boundaries"] = tuple(sorted((dict(row) for row in leakage.boundary_rows), key=lambda row: boundary_order[str(row["boundary_id"])]))
    result["leakage_duplicate_candidates"] = tuple(sorted((dict(row) for row in leakage.candidate_rows), key=lambda row: (boundary_order[str(row["boundary_id"])], str(row["left_role_id"]), str(row["right_role_id"]), str(row["left_record_id"]), str(row["right_record_id"]))))
    statuses = []
    for row in tables["audit_status"]:
        current = dict(row); component = str(current["component"]); current["condition_id"] = str(current.get("condition_id", "")); statuses.append(current)
    for row in leakage.status_rows:
        current = dict(row); current.update(panel_id="", endpoint_id="", protocol_id="", condition_id=str(current.pop("boundary_id"))); statuses.append(current)
    for endpoint in config.raw["phase6_endpoint_manifest"]:
        for tolerance in (1, 2, 4, 8):
            statuses.append({"component": "phase6_peak_tolerance", "panel_id": "", "endpoint_id": endpoint["endpoint_id"], "protocol_id": "", "condition_id": f"tolerance:{tolerance}", "state": "complete_numeric", "reason_code": "", "numerical_row_count": len(system_ids), "rank_row_count": 0 if endpoint["preferred_direction"] == "non_monotonic" or tolerance == 2 else 1, "evidence_key": "step5_bootstrap_projection"})
    component_order = {"resampling": 0, "normalization": 1, "phase4_peak_tolerance": 2, "phase6_peak_tolerance": 3, "leakage_boundaries": 4}
    def status_key(row: Mapping[str, object]) -> tuple[object, ...]:
        component = str(row["component"]); condition = str(row["condition_id"]); panel = str(row.get("panel_id", ""))
        if component in {"resampling", "normalization", "phase4_peak_tolerance"}: return (component_order[component], panel_order[panel], *_condition_sort(component, condition))
        if component == "phase6_peak_tolerance": return (component_order[component], endpoint_order[str(row["endpoint_id"])], _condition_sort("phase4_peak_tolerance", condition)[0])
        return (component_order[component], boundary_order[condition])
    result["audit_status"] = tuple({**row, "status_order": index} for index, row in enumerate(sorted(statuses, key=status_key)))
    return result


def _load_real_inputs(config: AppendixAuditsConfig, project_root: Path, worker_count: int) -> AppendixAuditsInputs:
    authorities = _validate_authorities(config, project_root)
    from rpe.runner.phase6_appendix_bacteria import build_bacteria_phase4_sensitivity
    from rpe.runner.phase6_appendix_sugar import build_sugar_phase4_sensitivity
    from rpe.runner.phase6_appendix_rruff import build_rruff_phase4_sensitivity
    from rpe.runner.phase6_appendix_leakage import build_phase6_leakage_audit
    results = (build_bacteria_phase4_sensitivity(config.raw, project_root, worker_count), build_sugar_phase4_sensitivity(config.raw, project_root, worker_count), build_rruff_phase4_sensitivity(config.raw, project_root, worker_count))
    leakage = build_phase6_leakage_audit(config.raw, project_root, worker_count)
    peak_root = (project_root / str(config.raw["authorities"]["phase6_peak_checksum_ledger"]["path"])).parent
    peak_config = json.loads((peak_root / "config.json").read_text(encoding="utf-8"))
    system_ids = tuple(str(value) for value in peak_config.get("promoted_system_ids", ()))
    if len(system_ids) != 21 or len(set(system_ids)) != 21: raise AppendixAuditsError("Step-5 promoted systems", "expected exactly 21 ordered IDs")
    phase6_rows = tuple(json.loads(line) for line in (peak_root / "bootstrap_results.jsonl").read_text(encoding="utf-8").splitlines() if line)
    condition_path = peak_root / "synthetic_condition_rows.jsonl"
    condition_receipt = _validate_step5_condition_rows(condition_path, phase6_rows)
    expected_keys = {(system, str(endpoint["endpoint_id"])) for system in system_ids for endpoint in config.raw["phase6_endpoint_manifest"]}
    observed_keys = [(str(row.get("system_id")), str(row.get("endpoint_id"))) for row in phase6_rows]
    if len(phase6_rows) != 210 or len(set(observed_keys)) != 210 or set(observed_keys) != expected_keys: raise AppendixAuditsError("Step-5 bootstrap_results", "must be exactly one row per promoted system and configured endpoint")
    canonical = _canonicalize_inputs(config, _adapter_tables(results), leakage, phase6_rows, system_ids)
    identities = {name: dict(getattr(result, "identity")) for name, result in zip(("bacteria", "sugar", "rruff"), results, strict=True)}
    leakage_identity = dict(leakage.identity)
    bridge = MappingProxyType({**dict(authorities), "phase2_direct_split": leakage_identity.get("phase2_rruff", {}), "phase3_formal_denoising": leakage_identity.get("phase3_denoising", {}), "leakage_receipts": leakage_identity})
    identity = {**canonical, "phase6_system_ids": system_ids, "run_identity": _formal_identity_receipt(config, project_root, identities, leakage_identity, peak_root, system_ids, condition_receipt)}
    return AppendixAuditsInputs((), phase6_rows, canonical["leakage_boundaries"], bridge, MappingProxyType(identity))


def _validate_step5_condition_rows(path: Path, bootstrap_rows: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not path.is_file(): raise AppendixAuditsError("Step-5 synthetic conditions", "missing retained condition rows")
    expected: dict[tuple[str, str, str], list[Mapping[str, object]]] = {}
    for row in bootstrap_rows:
        for tolerance in ("1", "2", "4", "8"):
            expected.setdefault((str(row["system_id"]), str(row["endpoint_id"]), tolerance), [])
    count = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line); system = str(row["system_id"])
            for tolerance, endpoints in dict(row.get("endpoints_by_tolerance_cm1", {})).items():
                for endpoint, state in dict(endpoints).items():
                    key = (system, str(endpoint), str(tolerance))
                    if key not in expected: raise AppendixAuditsError("Step-5 synthetic conditions", "unexpected system/endpoint/tolerance row")
                    if not isinstance(state, Mapping) or "state" not in state or "denominator" not in state:
                        raise AppendixAuditsError("Step-5 synthetic conditions", "missing state or denominator")
                    expected[key].append(state); count += 1
    for (system, endpoint, tolerance), rows in expected.items():
        curve = next(row for row in bootstrap_rows if str(row["system_id"]) == system and str(row["endpoint_id"]) == endpoint)["curve_by_tolerance_cm1"][tolerance]
        contributing = tuple(curve.get("contributing_class_counts", ()))
        numerical = sum(str(row["state"]) == "complete_numeric" for row in rows)
        if contributing and sum(contributing) != numerical:
            raise AppendixAuditsError("Step-5 synthetic conditions", "contributing-count mismatch")
    return MappingProxyType({"synthetic_condition_rows_sha256": _sha_file(path), "validated_endpoint_rows": count})


def _formal_identity_receipt(config: AppendixAuditsConfig, project_root: Path, adapters: Mapping[str, object], leakage: Mapping[str, object], peak_root: Path, system_ids: Sequence[str], condition_receipt: Mapping[str, object]) -> Mapping[str, object]:
    code_manifest = {relative: _sha_file(project_root / relative) for relative in _IDENTITY_CODE_FILES}
    return MappingProxyType({"identity_schema_version": "phase6-step7-formal-identity-v1", "adapter_identities": adapters, "leakage_identity": leakage, "step5": {"bootstrap_results_sha256": _sha_file(peak_root / "bootstrap_results.jsonl"), "config_sha256": _sha_file(peak_root / "config.json"), "promoted_system_ids": tuple(system_ids), "condition_validation": condition_receipt}, "authorities": config.raw["authorities"], "code_manifest": code_manifest, "environment": {"python": __import__("platform").python_version(), "numpy": np.__version__, "scipy": __import__("scipy").__version__, "platform_system": __import__("platform").system(), "platform_machine": __import__("platform").machine()}, "fixed_ordering": {"panels": tuple(config.raw["fixed_panel_order"]), "metrics": tuple(config.raw["metric_ids"]), "boundaries": tuple(row["boundary_id"] for row in config.raw["leakage_boundary_definitions"])}})


def build_phase6_appendix_audits(output_root: Path, *, worker_count: int, project_root: Path=ROOT) -> AppendixAuditsSummary:
    config=load_phase6_appendix_audits_config(project_root/DEFAULT_CONFIG.relative_to(ROOT)); inputs=_load_real_inputs(config,project_root,worker_count)
    return build_phase6_appendix_audits_from_inputs(inputs,output_root,config=config,worker_count=worker_count)


__all__=["AppendixAuditsConfig","AppendixAuditsError","AppendixAuditsInputs","AppendixAuditsSummary","Phase4SensitivityRows","assemble_phase4_sensitivity_rows","build_phase6_appendix_audits","build_phase6_appendix_audits_from_inputs","exact_spectrum_sha256","load_phase6_appendix_audits_config","normalize_spectrum","project_phase6_peak_tolerance_rows","rank_stability_rows","resample_spectrum","scale_invariance_passes","scan_near_duplicates","select_calibration_pairs","target_axis"]
