from __future__ import annotations

import csv
import hashlib
import json
import math
import multiprocessing
import struct
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.evaluation import Peak1D, Spectrum1D
from rpe.io.perturbed_store import read_perturbed_shard
from rpe.methods.catalog import Phase3System, TaskLine, load_classical_catalog
from rpe.methods.classical.peaks import (
    DetectedPeak1D,
    PeakRunStatus,
    run_peak_detection_system,
)
from rpe.metrics.peak import _match_peaks
from rpe.perturb import PeakFamilyPreparedState, PerturbationContext
from rpe.perturb.sweep import load_perturbation_sweep_config
from rpe.runner.phase1_perturbations import operator_for


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/peak_evidence_v1.json"
CONFIG_BYTES = 12007
CONFIG_SHA256 = "837c0e3c16db1b651bcb2ffebe37b10551bdecdc19019685cd472e0ebd148959"
RUN_DOMAIN = b"rpe-phase6-peak-evidence-v1\0"
TRUTH_DOMAIN = b"rpe-phase6-peak-intervention-truth-v1\0"
PERTURBATIONS = ("p01", "p02", "p03", "p04", "p05")
ENDPOINTS = {
    "p01": ("P1-retention",),
    "p02": ("P2-affected-retention", "P2-unaffected-retention"),
    "p03": ("P3-retention", "P3-position-mae", "P3-signed-log2-fwhm-ratio"),
    "p04": ("P4-deleted-event-disappearance", "P4-undeleted-retention"),
    "p05": ("P5-inserted-event-detection-gain", "P5-native-peak-retention"),
}
ENDPOINT_ORDER = tuple(value for key in PERTURBATIONS for value in ENDPOINTS[key])
SUCCESS = frozenset({PeakRunStatus.COMPLETE, PeakRunStatus.COMPLETE_WITH_WARNING})
_WORKER_INPUTS = None
_WORKER_SYSTEMS: tuple[Phase3System, ...] = ()


class PeakEvidenceVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class _Config:
    raw: bytes
    sha256: str
    document: Mapping[str, object]
    authorities: Mapping[str, Mapping[str, object]]
    artifact_contract: Mapping[str, object]
    bootstrap: Mapping[str, object]
    cohort: Mapping[str, object]
    endpoint_manifest: tuple[Mapping[str, object], ...]
    expected: Mapping[str, int]
    positive_alphas: tuple[float, ...]
    tolerances: tuple[float, ...]
    promoted_ids: tuple[str, ...]
    blocked: tuple[Mapping[str, str], ...]


@dataclass(frozen=True)
class _Condition:
    perturbation_id: str
    alpha: float
    alpha_hex: str
    state_digest: str
    output_id: str
    output_sha256: str
    truth: Mapping[str, object]
    truth_digest: str
    spectrum: Spectrum1D


@dataclass(frozen=True)
class _Source:
    class_label: int
    selection_rank: int
    record_id: str
    spectrum: Spectrum1D
    state_digests: Mapping[str, str]
    interventions: Mapping[str, tuple[_Condition, ...]]


@dataclass(frozen=True)
class PeakEvidenceVerificationSummary:
    path: Path
    run_id: str
    status: str
    system_count: int
    source_count: int
    intervention_truth_row_count: int
    detector_receipt_row_count: int
    detector_call_count: int
    condition_row_count: int
    class_row_count: int
    bootstrap_row_count: int
    method_evidence_row_count: int


def _ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_ready(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _canon(value: object) -> bytes:
    return (json.dumps(_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(_canon(row) for row in rows)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arr_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype="<f8").tobytes()).hexdigest()


def _load_config(path: Path) -> _Config:
    raw = path.read_bytes()
    document = json.loads(raw)
    if raw != _canon(document):
        raise PeakEvidenceVerificationError("config is not canonical")
    if path.resolve() == DEFAULT_CONFIG.resolve() and (len(raw) != CONFIG_BYTES or hashlib.sha256(raw).hexdigest() != CONFIG_SHA256):
        raise PeakEvidenceVerificationError("frozen config identity mismatch")
    return _Config(
        raw=raw, sha256=hashlib.sha256(raw).hexdigest(), document=document,
        authorities=dict(document["authorities"]), artifact_contract=dict(document["artifact_contract"]),
        bootstrap=dict(document["bootstrap"]), cohort=dict(document["cohort"]),
        endpoint_manifest=tuple(dict(row) for row in document["endpoint_manifest"]),
        expected={str(key): int(value) for key, value in dict(document["expected"]).items()},
        positive_alphas=tuple(float(value) for value in document["positive_alpha_grid"]),
        tolerances=tuple(float(value) for value in document["tolerances_cm1"]),
        promoted_ids=tuple(str(value) for value in document["promoted_system_ids"]),
        blocked=tuple(dict(row) for row in document["blocked_systems"]),
    )


def _check_authorities(config: _Config, root: Path) -> None:
    for name, identity in config.authorities.items():
        path = root / str(identity["path"])
        if not path.is_file() or path.stat().st_size != int(identity["byte_count"]) or _sha(path) != str(identity["sha256"]):
            raise PeakEvidenceVerificationError(f"authority mismatch: {name}")


def _check_ledger(run: Path) -> None:
    entries = []
    for line in (run / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            raise PeakEvidenceVerificationError("invalid Phase-1 checksum ledger")
        entries.append((parts[1], parts[0]))
    names = tuple(name for name, _ in entries)
    actual = tuple(sorted(path.relative_to(run).as_posix() for path in run.rglob("*") if path.is_file() and path.name not in {"SHA256SUMS", "complete.json", "failed.json"}))
    if names != tuple(sorted(names)) or names != actual:
        raise PeakEvidenceVerificationError("Phase-1 checksum inventory mismatch")
    for name, digest in entries:
        if _sha(run / name) != digest:
            raise PeakEvidenceVerificationError(f"Phase-1 checksum mismatch: {name}")


def _projection(run: Path) -> bytes:
    chosen: dict[int, Mapping[str, object]] = {}
    count = 0
    for path in sorted((run / "shards").glob("*/cells.jsonl")):
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            if row.get("perturbation_id") == "p05" and row.get("status") == "complete":
                count += 1
                source = row["source"]; label = int(source["class_label"]); rank = int(source["selection_rank"])
                item = {"class_label": label, "selection_rank": rank, "source_record_id": str(source["source_record_id"])}
                if label not in chosen or rank < int(chosen[label]["selection_rank"]):
                    chosen[label] = item
    rows = sorted(chosen.values(), key=lambda row: (int(row["class_label"]), int(row["selection_rank"])))
    if count != 7819 or len(rows) != 2250:
        raise PeakEvidenceVerificationError("P5 cohort count mismatch")
    return _jsonl(rows)


def _weak(state: PeakFamilyPreparedState) -> tuple[int, ...]:
    return tuple(i for _, _, i in sorted((state.prominences[i], state.peak_positions_cm1[i], i) for i in range(len(state.peak_indices))))


def _truth(pid: str, state: PeakFamilyPreparedState, source: Spectrum1D, alpha: float, diagnostics: Mapping[str, object]) -> Mapping[str, object]:
    count = len(state.peak_indices)
    if pid == "p01": return {"attenuation_fraction": alpha, "affected_component_count": count}
    if pid in {"p02", "p04"}:
        indexes = _weak(state)[:min(int(math.ceil(alpha * count)), count)]
        supports = tuple((float(source.axis_cm1[state.support_bounds[i][0]]), float(source.axis_cm1[state.support_bounds[i][1]])) for i in indexes)
        if pid == "p02": return {"affected_component_count": len(indexes), "affected_supports_cm1": supports, "attenuation_fraction": 0.5}
        return {"deleted_component_count": len(indexes), "deleted_centers_cm1": tuple(float(state.peak_positions_cm1[i]) for i in indexes), "deleted_supports_cm1": supports}
    if pid == "p03": return {"replacement_centers_cm1": tuple(float(x) for x in state.peak_positions_cm1), "replacement_component_count": count, "equal_area": True, "imposed_sigma_cm1": float(diagnostics["broadened_sigma_cm1"])}
    selected = min(int(math.floor(alpha * count)), len(state.candidate_false_centers_cm1))
    return {"inserted_component_count": selected, "inserted_centers_cm1": tuple(float(x) for x in state.candidate_false_centers_cm1[:selected]), "inserted_height": float(diagnostics["false_peak_height"]), "inserted_fwhm_cm1": float(diagnostics["false_peak_fwhm_cm1"]), "nested_prefix": True}


def _build_source(stored_source, cells, sweep, label: int, rank: int, record_id: str) -> _Source:
    spectrum = Spectrum1D(stored_source.source_spectrum_id, stored_source.sample_id, stored_source.axis_cm1, stored_source.intensity)
    context = PerturbationContext(sweep.sweep_id, sweep.sha256, sweep.global_seed)
    states = {}; interventions = {}
    by_id = {cell.perturbation_id: cell for cell in cells}
    for pid in PERTURBATIONS:
        operator = operator_for(pid, sweep); state = operator.prepare(spectrum, context)
        if not isinstance(state, PeakFamilyPreparedState) or by_id[pid].state_digest != state.state_digest or by_id[pid].status != "complete":
            raise PeakEvidenceVerificationError("Phase-1 state mismatch")
        stored = {record.alpha: record for record in by_id[pid].records}; conditions=[]
        for alpha in sweep.alpha_grid:
            result = operator.apply(spectrum, alpha, state); record = stored.get(alpha)
            if record is None or record.output_spectrum_id != result.output.spectrum_id or not np.array_equal(record.intensity, result.output.intensity):
                raise PeakEvidenceVerificationError("Phase-1 output mismatch")
            if alpha == 0.0:
                if not np.array_equal(result.output.intensity, spectrum.intensity): raise PeakEvidenceVerificationError("alpha zero mismatch")
                continue
            truth = _truth(pid, state, spectrum, float(alpha), result.diagnostics)
            conditions.append(_Condition(pid, float(alpha), struct.pack("<d", float(alpha)).hex(), state.state_digest, result.output.spectrum_id, _arr_sha(result.output.intensity), truth, hashlib.sha256(TRUTH_DOMAIN + _canon(truth)).hexdigest(), result.output))
        states[pid]=state.state_digest; interventions[pid]=tuple(conditions)
    return _Source(label, rank, record_id, spectrum, states, interventions)


def _real_inputs(config: _Config, root: Path):
    _check_authorities(config, root); run=(root / str(config.authorities["phase1_manifest"]["path"])).parent; _check_ledger(run)
    projection=_projection(run)
    if hashlib.sha256(projection).hexdigest()!=str(config.cohort["projection_sha256"]): raise PeakEvidenceVerificationError("cohort digest mismatch")
    rows=[json.loads(line) for line in projection.splitlines()]; sweep=load_perturbation_sweep_config(root/str(config.authorities["shared_sweep"]["path"])); sources=[]
    by_shard={};
    for row in rows: by_shard.setdefault(int(row["selection_rank"])//100,[]).append(row)
    built={}
    for shard_index in sorted(by_shard):
        shard=read_perturbed_shard(run/"shards"/f"{shard_index:05d}")
        for row in by_shard[shard_index]:
            rank=int(row["selection_rank"]); local=rank-shard_index*100; source=shard.sources[local]
            if source.source_record_id!=row["source_record_id"]: raise PeakEvidenceVerificationError("selected source mismatch")
            built[rank]=_build_source(source, shard.cells[local*12:(local+1)*12], sweep, int(row["class_label"]), rank, str(row["source_record_id"]))
    sources=tuple(built[int(row["selection_rank"])] for row in rows)
    identity={"class_count":len(sources),"projection_sha256":hashlib.sha256(projection).hexdigest(),"source_record_ids_sha256":hashlib.sha256(("\n".join(x.record_id for x in sources)+"\n").encode()).hexdigest()}
    return False, "rruff_core10k_p5_complete_class_balanced_2250", projection, sources, identity


def _synthetic_inputs(config: _Config):
    sweep=load_perturbation_sweep_config(ROOT/str(config.authorities["shared_sweep"]["path"])); axis=np.linspace(100.,500.,401,dtype="<f8"); sources=[]
    for i,shift in enumerate((0.,3.)):
        intensity=np.ones(axis.size,dtype="<f8")
        for center,height,sigma in ((145+shift,9,2.5),(220+shift,6,3),(310+shift,12,2),(405+shift,7,3.5)): intensity += height*np.exp(-0.5*((axis-center)/sigma)**2)
        spectrum=Spectrum1D(f"fixture-source-{i}",f"fixture-sample-{i}",axis,intensity); fake=type("Stored",(),{})(); fake.source_spectrum_id=spectrum.spectrum_id; fake.sample_id=spectrum.sample_id; fake.axis_cm1=axis; fake.intensity=intensity
        context=PerturbationContext(sweep.sweep_id,sweep.sha256,sweep.global_seed); states={}; interventions={}
        for pid in PERTURBATIONS:
            operator=operator_for(pid,sweep); state=operator.prepare(spectrum,context); states[pid]=state.state_digest; values=[]
            for alpha in sweep.alpha_grid[1:]:
                result=operator.apply(spectrum,alpha,state); truth=_truth(pid,state,spectrum,float(alpha),result.diagnostics)
                values.append(_Condition(pid,float(alpha),struct.pack("<d",float(alpha)).hex(),state.state_digest,result.output.spectrum_id,_arr_sha(result.output.intensity),truth,hashlib.sha256(TRUTH_DOMAIN+_canon(truth)).hexdigest(),result.output))
            interventions[pid]=tuple(values)
        sources.append(_Source(i+1,i,f"fixture-record-{i}",spectrum,states,interventions))
    projection=_jsonl({"class_label":x.class_label,"selection_rank":x.selection_rank,"source_record_id":x.record_id} for x in sources)
    return True,"synthetic_peak_evidence_fixture",projection,tuple(sources),{"class_count":2,"projection_sha256":hashlib.sha256(projection).hexdigest()}


def _peaks(values: Sequence[Peak1D | DetectedPeak1D]) -> tuple[Peak1D,...]: return tuple(x.to_peak1d() if isinstance(x,DetectedPeak1D) else x for x in values)
def _pairs(a,b,t): return tuple((x.reference_index,x.candidate_index) for x in _match_peaks(_peaks(a),_peaks(b),tolerance_cm1=t))
def _point_peaks(values): return tuple(Peak1D(float(x),None,None,None,None) for x in values)
def _value(endpoint,numerator,denominator,reason):
    return {"endpoint_id":endpoint,"numerator":0 if denominator==0 else numerator,"denominator":denominator,"value":None if denominator==0 else float(numerator)/denominator,"state":"complete_empty_risk_set" if denominator==0 else "complete_numeric","reason_code":reason if denominator==0 else ""}


def _evaluate(pid, baseline_values, candidate_values, truth, tolerance):
    baseline=_peaks(baseline_values); candidate=_peaks(candidate_values); matches=_pairs(baseline,candidate,tolerance); matched={x for x,_ in matches}
    if pid=="p01": return (_value("P1-retention",len(matches),len(baseline),"no_baseline_peaks"),)
    if pid=="p02":
        supports=tuple(tuple(float(y) for y in x) for x in truth.get("affected_supports_cm1",())); affected={i for i,x in enumerate(baseline) if any(a<=x.position_cm1<=b for a,b in supports)}; unaffected=set(range(len(baseline)))-affected
        return (_value("P2-affected-retention",len(affected&matched),len(affected),"no_affected_baseline_peaks"),_value("P2-unaffected-retention",len(unaffected&matched),len(unaffected),"no_unaffected_baseline_peaks"))
    if pid=="p03":
        retention=_value("P3-retention",len(matches),len(baseline),"no_baseline_peaks")
        if not matches: return (retention,_value("P3-position-mae",0,0,"no_matched_baseline_peaks"),_value("P3-signed-log2-fwhm-ratio",0,0,"no_matched_baseline_peaks"))
        return (retention,_value("P3-position-mae",sum(abs(candidate[j].position_cm1-baseline[i].position_cm1) for i,j in matches),len(matches),""),_value("P3-signed-log2-fwhm-ratio",sum(math.log2(float(candidate[j].fwhm_cm1)/float(baseline[i].fwhm_cm1)) for i,j in matches),len(matches),""))
    if pid=="p04":
        events=_point_peaks(truth.get("deleted_centers_cm1",())); eb=_pairs(events,baseline,tolerance); eligible=tuple(events[i] for i,_ in eb); ec=_pairs(eligible,candidate,tolerance); rb=tuple(x for i,x in enumerate(baseline) if i not in {j for _,j in eb}); rc=tuple(x for i,x in enumerate(candidate) if i not in {j for _,j in ec}); rm=_pairs(rb,rc,tolerance)
        return (_value("P4-deleted-event-disappearance",len(eligible)-len(ec),len(eligible),"no_baseline_matched_deleted_event"),_value("P4-undeleted-retention",len(rm),len(rb),"no_undeleted_baseline_peaks"))
    inserted=_point_peaks(truth.get("inserted_centers_cm1",()))
    if not inserted: return tuple({"endpoint_id":endpoint,"numerator":0,"denominator":0,"value":None,"state":"not_applicable_zero_synthetic_events","reason_code":"not_applicable_zero_synthetic_events"} for endpoint in ENDPOINTS["p05"])
    ib=_pairs(inserted,baseline,tolerance); ic=_pairs(inserted,candidate,tolerance); rb=tuple(x for i,x in enumerate(baseline) if i not in {j for _,j in ib}); rc=tuple(x for i,x in enumerate(candidate) if i not in {j for _,j in ic}); rm=_pairs(rb,rc,tolerance)
    return (_value("P5-inserted-event-detection-gain",len(ic)-len(ib),len(inserted),""),_value("P5-native-peak-retention",len(rm),len(rb),"no_native_baseline_peaks"))


def _detector_doc(result): return {"status":result.status.value,"peak_count":len(result.peaks),"peaks_sha256":result.peaks_sha256,"warning_count":len(result.warnings),"warnings":tuple({"category":x.category,"message":x.message} for x in result.warnings),"error_code":result.error_code,"error_message":result.error_message}
def _closed(pid,state,reason): return tuple({"endpoint_id":endpoint,"numerator":0,"denominator":0,"value":None,"state":state,"reason_code":reason} for endpoint in ENDPOINTS[pid])


def _evaluate_pair(system: Phase3System, source: _Source, tolerances):
    baseline=run_peak_detection_system(system,source.spectrum); receipts=[]; conditions=[]; values={endpoint:[] for endpoint in ENDPOINT_ORDER}
    for pid in PERTURBATIONS:
        for condition in source.interventions[pid]:
            candidate=run_peak_detection_system(system,condition.spectrum); by={}
            for tolerance in tolerances:
                key=format(float(tolerance),"g")
                if baseline.status not in SUCCESS: rows=_closed(pid,"not_evaluable_detector_failure",f"alpha_zero_detector:{baseline.status.value}:{baseline.error_code or 'unknown'}")
                elif candidate.status not in SUCCESS: rows=_closed(pid,"not_evaluable_detector_failure",f"positive_detector:{candidate.status.value}:{candidate.error_code or 'unknown'}")
                else: rows=_evaluate(pid,baseline.peaks,candidate.peaks,condition.truth,float(tolerance))
                by[key]={str(row["endpoint_id"]):row for row in rows}
            event_count=next((int(condition.truth[k]) for k in ("inserted_component_count","deleted_component_count","affected_component_count","replacement_component_count") if k in condition.truth),0)
            conditions.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"class_label":source.class_label,"selection_rank":source.selection_rank,"source_record_id":source.record_id,"perturbation_id":pid,"alpha":condition.alpha,"alpha_float64_le_hex":condition.alpha_hex,"generator_event_count":event_count,"truth_digest":condition.truth_digest,"baseline_detector_status":baseline.status.value,"baseline_peak_count":len(baseline.peaks),"baseline_peaks_sha256":baseline.peaks_sha256,"candidate_detector_status":candidate.status.value,"candidate_peak_count":len(candidate.peaks),"candidate_peaks_sha256":candidate.peaks_sha256,"endpoints_by_tolerance_cm1":by})
            for endpoint in ENDPOINTS[pid]: values[endpoint].append({"alpha":condition.alpha,"alpha_float64_le_hex":condition.alpha_hex,"by_tolerance_cm1":{format(float(t),"g"):by[format(float(t),"g")][endpoint] for t in tolerances}})
            receipts.append({"perturbation_id":pid,"alpha":condition.alpha,"alpha_float64_le_hex":condition.alpha_hex,"output_spectrum_id":condition.output_id,"output_intensity_sha256":condition.output_sha256,"truth_digest":condition.truth_digest,**_detector_doc(candidate)})
    classes=tuple({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"class_label":source.class_label,"selection_rank":source.selection_rank,"source_record_id":source.record_id,"endpoint_id":endpoint,"responses":tuple(values[endpoint])} for endpoint in ENDPOINT_ORDER)
    receipt={"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"class_label":source.class_label,"selection_rank":source.selection_rank,"source_record_id":source.record_id,"alpha_zero":_detector_doc(baseline),"conditions":tuple(receipts),"detector_call_count":1+len(receipts)}
    return receipt,tuple(conditions),classes


def _init(inputs,systems):
    global _WORKER_INPUTS,_WORKER_SYSTEMS; _WORKER_INPUTS=inputs; _WORKER_SYSTEMS=tuple(systems)
def _task(task):
    si,ri,tolerances=task
    with threadpool_limits(limits=1): return _evaluate_pair(_WORKER_SYSTEMS[si],_WORKER_INPUTS[3][ri],tolerances)
def _ordered(executor,tasks,window):
    iterator=iter(tasks); pending=[]
    for _ in range(window):
        try: pending.append(executor.submit(_task,next(iterator)))
        except StopIteration: break
    while pending:
        yield pending.pop(0).result()
        try: pending.append(executor.submit(_task,next(iterator)))
        except StopIteration: pass


def _truth_rows(inputs):
    for source in inputs[3]:
        for pid in PERTURBATIONS:
            yield {"class_label":source.class_label,"selection_rank":source.selection_rank,"source_record_id":source.record_id,"source_spectrum_id":source.spectrum.spectrum_id,"source_axis_sha256":_arr_sha(source.spectrum.axis_cm1),"source_intensity_sha256":_arr_sha(source.spectrum.intensity),"perturbation_id":pid,"state_digest":source.state_digests[pid],"conditions":tuple({"alpha":x.alpha,"alpha_float64_le_hex":x.alpha_hex,"output_spectrum_id":x.output_id,"output_intensity_sha256":x.output_sha256,"truth":x.truth,"truth_digest":x.truth_digest} for x in source.interventions[pid])}


def _bootstrap(class_rows,systems,config):
    grouped={};
    for row in class_rows: grouped.setdefault((str(row["system_id"]),str(row["endpoint_id"])),[]).append(row)
    count=len(class_rows)//(len(systems)*10); rng=np.random.Generator(np.random.PCG64(int(config.bootstrap["seed"]))); indexes=rng.integers(0,count,size=(int(config.bootstrap["resamples"]),count)); output=[]
    for system in systems:
        for endpoint in ENDPOINT_ORDER:
            rows=grouped[(system.system_id,endpoint)]; curves={}; primary=None
            for tolerance in config.tolerances:
                key=format(tolerance,"g"); matrix=np.full((count,8),np.nan); fatal=False
                for ri,row in enumerate(rows):
                    for ai,response in enumerate(row["responses"]):
                        cell=response["by_tolerance_cm1"][key]; state=str(cell["state"]);
                        if state=="complete_numeric": matrix[ri,ai]=float(cell["value"])
                        elif state not in {"complete_empty_risk_set","not_applicable_zero_synthetic_events"}: fatal=True
                means=[]; contributing=[]
                for ai in range(8):
                    finite=np.isfinite(matrix[:,ai]); contributing.append(int(finite.sum())); means.append(None if not finite.any() else float(np.mean(matrix[finite,ai])))
                if fatal or any(x is None for x in means): summary={"estimate":None,"interval_lower":None,"interval_upper":None,"state":"not_evaluable_incomplete_system_grid" if fatal else "not_evaluable_no_contributing_classes"}
                else:
                    samples=np.empty(indexes.shape[0],dtype="<f8")
                    for replicate,sample in enumerate(indexes):
                        alpha_values=[]
                        for ai in range(8):
                            selected=matrix[sample,ai]; finite=np.isfinite(selected); alpha_values.append(float(np.mean(selected[finite])) if finite.any() else float(means[ai]))
                        samples[replicate]=float(np.mean(alpha_values))
                    summary={"estimate":float(np.mean(np.asarray(means,dtype="<f8"))),"interval_lower":float(np.quantile(samples,.025)),"interval_upper":float(np.quantile(samples,.975)),"state":"complete"}
                curves[key]={"alpha_estimates":tuple(means),"contributing_class_counts":tuple(contributing),**summary}
                if tolerance==2: primary=summary
            output.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"endpoint_id":endpoint,"design_class_count":count,"resamples":int(config.bootstrap["resamples"]),"seed":int(config.bootstrap["seed"]),"confidence_level":float(config.bootstrap["confidence_level"]),"curve_by_tolerance_cm1":curves,**primary})
    return output


def _csv(rows,fields):
    chunks=[]
    class Writer:
        def write(self,s): chunks.append(s); return len(s)
    writer=csv.DictWriter(Writer(),fieldnames=fields,lineterminator="\n"); writer.writeheader()
    for row in rows: writer.writerow({f:("" if row.get(f) is None else format(row[f],".17g") if isinstance(row.get(f),float) else row.get(f)) for f in fields})
    return "".join(chunks).encode()


def _method_rows(systems,blocked,status_rows,bootstrap,config):
    status={str(x["system_id"]):x for x in status_rows}; boots={(str(x["system_id"]),str(x["endpoint_id"])):x for x in bootstrap}; rows=[]
    for system in systems:
        for endpoint in config.endpoint_manifest:
            eid=str(endpoint["endpoint_id"]); estimate=lower=upper=None
            if eid=="availability": state=status[system.system_id]["status"]; reason=status[system.system_id]["reason_code"]; source="system_status.jsonl"
            elif eid=="direct_gt": state=reason="not_evaluated_no_real_peak_assignments"; source="system_status.jsonl"
            elif eid=="downstream": state=reason="not_evaluated_no_real_peak_downstream_target"; source="system_status.jsonl"
            else:
                boot=boots[(system.system_id,eid)]; state=str(boot["state"]); reason="" if state=="complete" else state; estimate=boot["estimate"]; lower=boot["interval_lower"]; upper=boot["interval_upper"]; source="synthetic_class_rows.jsonl"
            rows.append({"evidence_id":f"{system.system_id}:{eid}","task_line":"peak_detection","system_id":system.system_id,"family_id":system.family_id,"endpoint_id":eid,"protocol_id":endpoint["protocol_id"],"cohort_id":"rruff_core10k_p5_complete_class_balanced_2250","evidence_component":endpoint["evidence_component"],"metric_output_id":endpoint["metric_output_id"],"estimate":estimate,"interval_lower":lower,"interval_upper":upper,"preferred_direction":endpoint["preferred_direction"],"state":state,"reason_code":reason,"phase5_power_state":"not_powered_for_phase5","source_path":source,"source_sha256":None})
    for x in blocked: rows.append({"evidence_id":f"{x['system_id']}:availability","task_line":"peak_detection","system_id":x["system_id"],"family_id":x["family_id"],"endpoint_id":"availability","protocol_id":"availability","cohort_id":"rruff_core10k_p5_complete_class_balanced_2250","evidence_component":"availability","metric_output_id":"availability","estimate":None,"interval_lower":None,"interval_upper":None,"preferred_direction":"not_applicable","state":x["reason_code"],"reason_code":x["reason_code"],"phase5_power_state":"not_powered_for_phase5","source_path":"system_status.jsonl","source_sha256":None})
    return rows


def _family(method_rows):
    groups={};
    for row in method_rows: groups.setdefault((str(row["family_id"]),str(row["endpoint_id"])),[]).append(row)
    output=[]
    for (family,endpoint),items in sorted(groups.items()):
        values=[float(x["estimate"]) for x in items if x["estimate"] is not None]
        output.append({"family_id":family,"endpoint_id":endpoint,"registered_system_count":len(items),"complete_system_count":len(values),"median_estimate":None if not values else float(np.median(values)),"min_estimate":None if not values else float(np.min(values)),"max_estimate":None if not values else float(np.max(values))})
    return output


def _code(root):
    paths=("experiments/phase6/configs/peak_evidence_v1.json","rpe/io/perturbed_store.py","rpe/methods/catalog.py","rpe/methods/classical/peaks.py","rpe/metrics/peak.py","rpe/perturb/peak_family.py","rpe/runner/phase6_peak_evidence.py","rpe/runner/phase6_peak_evidence_verifier.py","tools/run_phase6_peak_evidence.py")
    return {name:{"byte_count":(root/name).stat().st_size,"sha256":_sha(root/name)} for name in paths}


def _rebuild(path: Path, config: _Config, inputs, systems, worker_count: int, root: Path):
    synthetic,cohort_id,projection,sources,cohort_identity=inputs; identity={"config_sha256":config.sha256,"cohort_identity":cohort_identity,"system_ids":tuple(x.system_id for x in systems),"synthetic_fixture":synthetic,"code_authority":_code(root)}; run_id=str(config.artifact_contract["run_prefix"])+hashlib.sha256(RUN_DOMAIN+_canon(identity)).hexdigest()
    truth=list(_truth_rows(inputs)); tasks=((si,ri,config.tolerances) for si in range(len(systems)) for ri in range(len(sources))); global _WORKER_INPUTS,_WORKER_SYSTEMS; _WORKER_INPUTS=inputs;_WORKER_SYSTEMS=tuple(systems)
    if worker_count==1: results=map(_task,tasks); executor=None
    else: executor=ProcessPoolExecutor(max_workers=worker_count,mp_context=multiprocessing.get_context("fork"),initializer=_init,initargs=(inputs,systems));results=_ordered(executor,tasks,max(2,2*worker_count))
    receipts=[];conditions=[];classes=[];counts={x.system_id:{"complete":0,"complete_with_warning":0,"not_applicable":0,"failed_runtime":0} for x in systems}
    try:
        for receipt,cr,cl in results:
            receipts.append(receipt);conditions.extend(cr);classes.extend(cl)
            for detector in (receipt["alpha_zero"],*receipt["conditions"]): counts[receipt["system_id"]][detector["status"]]=counts[receipt["system_id"]].get(detector["status"],0)+1
    finally:
        if executor is not None: executor.shutdown()
    blocked=() if synthetic else config.blocked; status=[]
    for system in systems:
        current=counts[system.system_id]; bad=current.get("not_applicable",0)+current.get("failed_runtime",0); status.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"status":"complete" if bad==0 else "complete_with_reason_closed_conditions","reason_code":"" if bad==0 else "detector_condition_failure","detector_call_count":sum(current.values()),"detector_status_counts":current})
    for row in blocked: status.append({"system_id":row["system_id"],"family_id":row["family_id"],"method_id":row["family_id"],"status":row["reason_code"],"reason_code":row["reason_code"],"detector_call_count":0,"detector_status_counts":{}})
    bootstrap=_bootstrap(classes,systems,config); method=_method_rows(systems,blocked,status,bootstrap,config); family=_family(method)
    payloads={"config.json":config.raw,"system_status.jsonl":_jsonl(status),"intervention_truth.jsonl":_jsonl(truth),"detector_receipts.jsonl":_jsonl(receipts),"synthetic_condition_rows.jsonl":_jsonl(conditions),"synthetic_class_rows.jsonl":_jsonl(classes),"bootstrap_results.jsonl":_jsonl(bootstrap)}
    source_hash={"system_status.jsonl":hashlib.sha256(payloads["system_status.jsonl"]).hexdigest(),"synthetic_class_rows.jsonl":hashlib.sha256(payloads["synthetic_class_rows.jsonl"]).hexdigest()}
    for row in method: row["source_sha256"]=source_hash[row["source_path"]]
    mfields=("evidence_id","task_line","system_id","family_id","endpoint_id","protocol_id","cohort_id","evidence_component","metric_output_id","estimate","interval_lower","interval_upper","preferred_direction","state","reason_code","phase5_power_state","source_path","source_sha256"); ffields=("family_id","endpoint_id","registered_system_count","complete_system_count","median_estimate","min_estimate","max_estimate")
    payloads["method_evidence_rows.csv"]=_csv(method,mfields);payloads["family_projection.csv"]=_csv(family,ffields)
    bridge={"authorities":config.authorities,"code_authority":_code(root),"cohort_identity":cohort_identity,"source_revision_status":"unavailable_no_valid_git_repository","promoted_system_ids_sha256":hashlib.sha256(("\n".join(x.system_id for x in systems)+"\n").encode()).hexdigest()};payloads["authority_bridge.json"]=_canon(bridge)
    payloads["preflight.json"]=_canon({"status":"passed","cohort_identity":cohort_identity,"promoted_system_count":len(systems),"blocked_system_count":len(blocked),"condition_count_per_system_source":41,"alpha_zero_reuse":"once_per_system_source","anti_circular_truth_boundary":"mechanism_facts_only_not_internal_peak_list_gt"})
    calls=sum(int(x["detector_call_count"]) for x in receipts);manifest={"schema_version":"phase6-peak-evidence-artifact-v1","run_id":run_id,"status":"complete","synthetic_fixture":synthetic,"system_count":len(systems),"blocked_system_count":len(blocked),"source_count":len(sources),"intervention_truth_row_count":len(truth),"detector_receipt_row_count":len(receipts),"detector_call_count":calls,"synthetic_condition_row_count":len(conditions),"synthetic_class_row_count":len(classes),"bootstrap_row_count":len(bootstrap),"method_evidence_row_count":len(method),"family_projection_row_count":len(family),"configured_payload_count":12,"payload_files":tuple(config.artifact_contract["payload_files"]),"source_revision_status":"unavailable_no_valid_git_repository"};payloads["manifest.json"]=_canon(manifest);payloads["complete.json"]=_canon({"run_id":run_id,"status":"complete"});payloads["SHA256SUMS"]="".join(f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n" for name in sorted(payloads)).encode();return payloads,manifest


def _artifact(path): return {x.relative_to(path).as_posix():x.read_bytes() for x in path.rglob("*") if x.is_file()}
def _check_artifact_ledger(path,expected):
    lines=(path/"SHA256SUMS").read_text().splitlines(); rows=[line.split("  ",1) for line in lines]
    if tuple(x[1] for x in rows)!=tuple(sorted(expected)): raise PeakEvidenceVerificationError("checksum inventory mismatch")
    for digest,name in rows:
        if _sha(path/name)!=digest: raise PeakEvidenceVerificationError(f"checksum mismatch for {name}")


def verify_phase6_peak_evidence(path: Path, *, worker_count: int, project_root: Path = ROOT) -> PeakEvidenceVerificationSummary:
    path=Path(path); config=_load_config(project_root/"experiments/phase6/configs/peak_evidence_v1.json"); manifest=json.loads((path/"manifest.json").read_bytes()); expected=tuple(config.artifact_contract["payload_files"])+("complete.json",); observed=set(_artifact(path))
    if observed!=set(expected)|{"SHA256SUMS"}: raise PeakEvidenceVerificationError("artifact inventory mismatch")
    _check_artifact_ledger(path,expected)
    catalog=load_classical_catalog(project_root/str(config.authorities["classical_catalog"]["path"])); by_id={x.system_id:x for x in catalog.systems}
    if bool(manifest["synthetic_fixture"]):
        inputs=_synthetic_inputs(config); status=[json.loads(x) for x in (path/"system_status.jsonl").read_bytes().splitlines()]; ids=tuple(str(x["system_id"]) for x in status if int(x["detector_call_count"])>0); systems=tuple(by_id[x] for x in ids)
    else:
        inputs=_real_inputs(config,project_root); systems=tuple(by_id[x] for x in config.promoted_ids); status=[json.loads(x) for x in (path/"system_status.jsonl").read_bytes().splitlines()]; expected_ids=tuple(x.system_id for x in systems)+tuple(str(x["system_id"]) for x in config.blocked)
        if tuple(str(x["system_id"]) for x in status)!=expected_ids: raise PeakEvidenceVerificationError("formal system status list mismatch")
    rebuilt, rebuilt_manifest=_rebuild(path,config,inputs,systems,worker_count,project_root)
    observed_files=_artifact(path)
    if observed_files!=rebuilt:
        mismatched=sorted(name for name in set(observed_files)|set(rebuilt) if observed_files.get(name)!=rebuilt.get(name))
        raise PeakEvidenceVerificationError("artifact byte mismatch after independent full reexecution: "+",".join(mismatched))
    return PeakEvidenceVerificationSummary(path,str(rebuilt_manifest["run_id"]),str(rebuilt_manifest["status"]),int(rebuilt_manifest["system_count"]),int(rebuilt_manifest["source_count"]),int(rebuilt_manifest["intervention_truth_row_count"]),int(rebuilt_manifest["detector_receipt_row_count"]),int(rebuilt_manifest["detector_call_count"]),int(rebuilt_manifest["synthetic_condition_row_count"]),int(rebuilt_manifest["synthetic_class_row_count"]),int(rebuilt_manifest["bootstrap_row_count"]),int(rebuilt_manifest["method_evidence_row_count"]))


__all__ = ["PeakEvidenceVerificationError", "PeakEvidenceVerificationSummary", "verify_phase6_peak_evidence"]
