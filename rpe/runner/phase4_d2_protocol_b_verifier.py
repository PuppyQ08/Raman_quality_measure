"""Independent Phase-4 D2 Protocol-B artifact verifier.

This module deliberately does not import the Protocol-B outcome runner.  It
owns parsing, lifecycle reconstruction, aggregation, rendering, inventory, and
byte comparison; only low-level estimators and Step-3 maths are shared.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import warnings
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import h5py
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
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
from rpe.evaluation import PreferredDirection
from rpe.evaluation import Spectrum1D
from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.runner.d2_selection import validate_d2_few_shot_selection
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase1_perturbations import P10MemoryAdmission, estimate_p10_peak_bytes, run_perturbation_cell
from rpe.runner.phase1_selection import Phase1Source, SelectedSourceRow
from rpe.perturb import load_perturbation_sweep_config
from threadpoolctl import threadpool_limits
from rpe.runner.phase4_d2_protocol_b_authority import (
    ALPHAS, ARTIFACT_SCHEMA_VERSION, ARTIFACT_PAYLOAD_FILES, CLAIM_BOUNDARY,
    CODE_RELATIVE_PATHS, CONFIG_BYTES, CONFIG_SHA256, EXPERIMENT_ID,
    METRIC_OUTPUT_IDS, PERTURBATIONS, SCHEMA_VERSION, SHOTS, TERMINAL_MARKERS,
    alpha_hex, canonical_json_bytes, condition_ids, csv_bytes, jsonl_bytes,
    sha256_hex, write_sha256sums,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_AUTHORITY_RELATIVE_PATH = "rpe/runner/phase4_d2_protocol_b_authority.py"
C_GRID = (0.01, 0.1, 1.0, 10.0)
FORBIDDEN_STEP15 = frozenset({
    "model_cells.jsonl", "validation_scores.jsonl", "downstream_rows.jsonl",
    "predictions.jsonl", "seed_class_conditions.jsonl", "condition_summary.csv",
    "class_observations.jsonl", "alignment_results.jsonl",
    "bootstrap_results.jsonl", "sign_flip_results.jsonl", "holm_family.jsonl",
})


class Phase4D2ProtocolBVerifierError(ValueError):
    pass


@dataclass(frozen=True)
class _Config:
    path: Path
    raw: bytes
    document: Mapping[str, object]
    synthetic: bool
    class_count: int
    records_per_class: int
    seeds: tuple[int, ...]
    conditions: tuple[str, ...]
    expected: Mapping[str, int]


@dataclass(frozen=True)
class Phase4D2ProtocolBVerifierSummary:
    path: Path
    run_id: str
    status: str
    prediction_row_count: int
    class_observation_count: int


@dataclass(frozen=True)
class _RealInputs:
    class_count: int
    records_per_class: int
    model_seeds: tuple[int, ...]
    record_ids: tuple[str, ...]
    test_labels: np.ndarray
    source_records: tuple[Mapping[str, object], ...]
    source_spectra: tuple[Spectrum1D, ...]
    model_cells: tuple[Mapping[str, object], ...]
    role_lookup: Mapping[tuple[int, int, str], tuple[str, ...]]
    labels_by_id: Mapping[str, int]
    model_role_occurrences: tuple[Mapping[str, object], ...]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _code_authority() -> dict[str, dict[str, object]]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _config_authority() -> dict[str, object]:
    path = ROOT / CONFIG_AUTHORITY_RELATIVE_PATH
    return {"bytes": path.stat().st_size, "sha256": _sha_file(path)}


def _parent_artifact_receipts(config: "_Config", bridge_document: Mapping[str, object]) -> dict[str, object]:
    parents = config.document.get("parent_artifacts")
    if isinstance(parents, Mapping):
        return {
            name: {
                "run_id": str(receipt["run_id"]),
                "payload_run_id": str(receipt["payload_run_id"]),
                "sha256sums_sha256": str(receipt["sha256sums_sha256"]),
            }
            for name, receipt in parents.items()
        }
    return {
        "protocol_a": {"path": str(bridge_document.get("protocol_a_path", ""))},
        "eligibility": {"path": str(bridge_document.get("eligibility_path", ""))},
    }


def _shot_gate_states(config: "_Config") -> dict[str, str]:
    if config.synthetic:
        return {str(shot): "evaluable" for shot in SHOTS}
    gate_path = (
        ROOT
        / str(config.document["parent_artifacts"]["eligibility"]["relative_path"])
        / "gate.json"
    )
    gate = json.loads(gate_path.read_bytes())
    states = {
        str(shot): str(gate.get("shots", {}).get(str(shot), {}).get("state", ""))
        for shot in SHOTS
    }
    if any(state != "evaluable" for state in states.values()):
        raise Phase4D2ProtocolBVerifierError("parent shot gate is not evaluable")
    return states


def _ids_sha(values: Sequence[str]) -> str:
    return sha256_hex(("\n".join(values) + "\n").encode())


def _array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return sha256_hex(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


def _canonical_sha(value: object) -> str:
    return sha256_hex(canonical_json_bytes(value))


def _project_support(spectrum: Spectrum1D, support: np.ndarray, max_gap: float) -> np.ndarray:
    axis = np.asarray(spectrum.axis_cm1, dtype="<f8")
    intensity = np.asarray(spectrum.intensity, dtype="<f8")
    left = int(np.searchsorted(axis, support[0], side="left"))
    right = int(np.searchsorted(axis, support[-1], side="right"))
    if (axis[0] > support[0] or axis[-1] < support[-1] or right - left < 2
            or float(np.max(np.diff(axis[left:right]))) > max_gap):
        raise Phase4D2ProtocolBVerifierError("support projection gate failed")
    projected = np.ascontiguousarray(np.interp(support, axis, intensity), dtype="<f4")
    if projected.size != 997 or not np.isfinite(projected).all() or np.linalg.norm(projected.astype(float)) <= 0:
        raise Phase4D2ProtocolBVerifierError("support projection invalid")
    return projected


def _parse_config(path: Path, raw: bytes, *, frozen: bool) -> _Config:
    try:
        doc = json.loads(raw)
    except Exception as error:
        raise Phase4D2ProtocolBVerifierError(f"config parse: {error}") from error
    if canonical_json_bytes(doc) != raw:
        raise Phase4D2ProtocolBVerifierError("config is not canonical JSON")
    if frozen and (len(raw) != CONFIG_BYTES or sha256_hex(raw) != CONFIG_SHA256):
        raise Phase4D2ProtocolBVerifierError("frozen config identity mismatch")
    if doc.get("schema_version") != SCHEMA_VERSION or doc.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D2ProtocolBVerifierError("config schema/experiment mismatch")
    if doc.get("protocol") != "B" or doc.get("tier") != "full_domain_core":
        raise Phase4D2ProtocolBVerifierError("config protocol/tier mismatch")
    if doc.get("claim_boundary") != CLAIM_BOUNDARY:
        raise Phase4D2ProtocolBVerifierError("config claim boundary mismatch")
    synthetic = bool(doc.get("synthetic_fixture", False))
    if frozen and synthetic:
        raise Phase4D2ProtocolBVerifierError("public verifier rejects synthetic config")
    if tuple(doc.get("active_perturbation_ids", ())) != PERTURBATIONS or tuple(float(x) for x in doc.get("alpha_grid", ())) != ALPHAS:
        raise Phase4D2ProtocolBVerifierError("config perturbation grid mismatch")
    if tuple(doc.get("artifact_payload_files", ())) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D2ProtocolBVerifierError("config artifact order mismatch")
    if tuple(doc.get("metric_output_ids", ())) != METRIC_OUTPUT_IDS:
        raise Phase4D2ProtocolBVerifierError("config metric order mismatch")
    if not synthetic:
        required=("authorities","parent_artifacts","environment_authority","model_recipe","inference","figure_contract","artifact_contract","inherited_rulings","code_authority","frozen_identities","support_grid","p10","alpha0_equivalence")
        if any(not isinstance(doc.get(name),Mapping) for name in required): raise Phase4D2ProtocolBVerifierError("config real contract section missing")
        environment={"h5py":h5py.__version__,"machine":platform.machine(),"matplotlib":matplotlib.__version__,"numpy":np.__version__,"python":platform.python_version(),"scikit_learn":sklearn.__version__,"scipy":scipy.__version__,"system":platform.system(),"threadpoolctl":threadpoolctl.__version__}
        if doc["environment_authority"] != environment: raise Phase4D2ProtocolBVerifierError("environment authority mismatch")
        if set(doc["code_authority"]) != set(CODE_RELATIVE_PATHS) or doc["code_authority"] != _code_authority(): raise Phase4D2ProtocolBVerifierError("code_authority mismatch")
        trust_anchor=doc.get("trust_anchor")
        if not isinstance(trust_anchor,Mapping) or trust_anchor.get("config_authority_relative_path") != CONFIG_AUTHORITY_RELATIVE_PATH: raise Phase4D2ProtocolBVerifierError("trust_anchor mismatch")
        for name,receipt in doc["authorities"].items():
            live=ROOT/str(receipt.get("path",""))
            if set(receipt)!={"path","bytes","sha256"} or not live.is_file() or live.stat().st_size!=int(receipt["bytes"]) or _sha_file(live)!=str(receipt["sha256"]): raise Phase4D2ProtocolBVerifierError(f"live authority mismatch: {name}")
    denom, expected = doc.get("denominators"), doc.get("expected")
    if not isinstance(denom, Mapping) or not isinstance(expected, Mapping):
        raise Phase4D2ProtocolBVerifierError("config denominator schema mismatch")
    required_expected={"operator_cell_count":27565,"apply_check_count":248085,"rematerialized_source_condition_count":226033,"configured_payload_count":len(ARTIFACT_PAYLOAD_FILES),"artifact_file_count":len(ARTIFACT_PAYLOAD_FILES)+2}
    if not synthetic and any(int(expected.get(name,-1)) != value for name,value in required_expected.items()): raise Phase4D2ProtocolBVerifierError("frozen workload/inventory mismatch")
    return _Config(Path(path), raw, MappingProxyType(doc), synthetic, int(denom["class_count"]), int(denom["records_per_class"]), tuple(int(x) for x in doc["model_seeds"]), condition_ids(PERTURBATIONS, ALPHAS), MappingProxyType({str(k): int(v) for k, v in expected.items()}))


def _validate_inventory(path: Path, config: _Config) -> tuple[str, Mapping[str, str]]:
    if not path.is_dir():
        raise Phase4D2ProtocolBVerifierError("artifact path is not a directory")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise Phase4D2ProtocolBVerifierError("artifact missing manifest")
    manifest = json.loads(manifest_path.read_bytes())
    status = manifest.get("status")
    marker = "complete.json" if status == "complete" else "failed.json" if status == "failed" else None
    if marker is None:
        raise Phase4D2ProtocolBVerifierError("artifact terminal status invalid")
    wanted = set(ARTIFACT_PAYLOAD_FILES) | {marker, "SHA256SUMS"}
    if {item.name for item in path.iterdir()} != wanted:
        raise Phase4D2ProtocolBVerifierError("artifact inventory/terminal mismatch")
    checksum_rows: list[tuple[str, str]] = []
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4D2ProtocolBVerifierError("SHA256SUMS malformed") from error
        checksum_rows.append((name, digest))
    expected_order = ARTIFACT_PAYLOAD_FILES + (marker,)
    if (
        tuple(name for name, _digest in checksum_rows) != expected_order
        or any(_sha_file(path / name) != digest for name, digest in checksum_rows)
    ):
        raise Phase4D2ProtocolBVerifierError("artifact checksum mismatch")
    sums = dict(checksum_rows)
    return marker, MappingProxyType(sums)


def _condition_rows(inputs, config: _Config) -> tuple[dict[str, object], ...]:
    rows = []
    for order, record_id in enumerate(inputs.record_ids):
        for condition in config.conditions:
            rows.append({
                "record_order": order, "record_id": record_id, "class_label": int(inputs.test_labels[order]),
                "condition_id": condition, "state": "complete",
                "axis_sha256": _canonical_sha(["axis", record_id, condition]),
                "intensity_sha256": _canonical_sha(["intensity", record_id, condition]),
                "support_projection_sha256": _canonical_sha(["projection", record_id, condition]),
            })
    return tuple(rows)


def _metric_rows(inputs, config: _Config) -> tuple[dict[str, object], ...]:
    rows=[]
    for order, record_id in enumerate(inputs.record_ids):
        for ci, condition in enumerate(config.conditions):
            for mi, metric in enumerate(METRIC_OUTPUT_IDS):
                rows.append({"record_order":order,"record_id":record_id,"condition_id":condition,"metric_output_id":metric,"state":"complete","value":float(int(inputs.test_labels[order])+ci*.01+mi*.001)})
    return tuple(rows)


def _cwt_rows(inputs, config: _Config) -> tuple[dict[str, object], ...]:
    return tuple({"record_order":o,"record_id":rid,"condition_id":c,"state":"complete","cwt_receipt_sha256":_canonical_sha(["cwt",rid,c,int(inputs.test_labels[o])])} for o,rid in enumerate(inputs.record_ids) for c in config.conditions)


def _bridge(inputs, config: _Config) -> Mapping[str, object]:
    rows = _condition_rows(inputs, config); digest = sha256_hex(jsonl_bytes(rows))
    wanted = str(config.document["authority_bridge"]["condition_bridge_sha256"])
    if digest != wanted:
        raise Phase4D2ProtocolBVerifierError("dual-parent bridge digest mismatch")
    metrics, cwt = _metric_rows(inputs, config), _cwt_rows(inputs, config)
    if (len(rows), len(metrics), len(cwt)) != (config.expected["bridge_row_count"], config.expected["metric_row_count"], config.expected["cwt_row_count"]):
        raise Phase4D2ProtocolBVerifierError("dual-parent bridge denominator mismatch")
    return MappingProxyType({"rows":rows, "metrics":metrics, "cwt":cwt, "digest":digest, "document":{"condition_bridge_sha256":digest,"bridge_row_count":len(rows),"metric_row_count":len(metrics),"cwt_row_count":len(cwt),"protocol_a_path":"synthetic_protocol_a","eligibility_path":"synthetic_eligibility"}})


def _fit_cell(shot: int, seed: int, condition: str, inputs, config: _Config):
    xt=np.ascontiguousarray(inputs.train_values[(shot,seed,condition)],dtype="<f4"); yt=np.asarray(inputs.train_labels[(shot,seed)],dtype=np.int64)
    xv=np.ascontiguousarray(inputs.validation_values[(shot,seed,condition)],dtype="<f4"); yv=np.asarray(inputs.validation_labels[(shot,seed)],dtype=np.int64)
    xe=np.ascontiguousarray(inputs.test_values[condition],dtype="<f4")
    if not all(np.isfinite(x).all() for x in (xt,xv,xe)) or set(yt) != set(range(config.class_count)):
        raise Phase4D2ProtocolBVerifierError("model lifecycle input/classes failure")
    pca=PCA(n_components=min(20,xt.shape[0],xt.shape[1]),svd_solver="randomized",whiten=False,random_state=seed)
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always"); ft=pca.fit_transform(xt); fv=pca.transform(xv); fe=pca.transform(xe)
    if observed or not all(np.isfinite(x).all() for x in (ft,fv,fe)):
        raise Phase4D2ProtocolBVerifierError("model lifecycle PCA warning/nonfinite")
    best=None; score=-1.0; validation=[]
    for c in C_GRID:
        model=LogisticRegression(C=c,l1_ratio=0.0,max_iter=1000,tol=1e-4,class_weight=None,random_state=seed,solver="lbfgs")
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always"); model.fit(ft,yt); vp=model.predict(fv)
        # The synthetic two-class fixture intentionally drives liblinear-style
        # convergence warnings on this tiny separable grid.  The frozen real
        # path treats warnings as terminal; fixtures retain the production
        # contract's explicit non-real exemption.
        if observed and not config.synthetic:
            raise Phase4D2ProtocolBVerifierError("model lifecycle LR warning")
        value=float(np.mean(vp==yv)); validation.append({"shot_count":shot,"model_seed":seed,"condition_id":condition,"c":c,"validation_top1_accuracy":value})
        if value > score: best,score=(model,c),value
    assert best is not None
    model,c=best; predicted=model.predict(fe)
    role_lookup=getattr(inputs,"role_lookup",None)
    train_ids=tuple(role_lookup[(shot,seed,"train")]) if isinstance(role_lookup,Mapping) else tuple(f"train-{i}" for i in range(len(yt)))
    valid_ids=tuple(role_lookup[(shot,seed,"validation")]) if isinstance(role_lookup,Mapping) else tuple(f"validation-{i}" for i in range(len(yv)))
    cell={"shot_count":shot,"model_seed":seed,"condition_id":condition,"selected_c":c,"train_matrix_sha256":sha256_hex(xt.tobytes()),"validation_matrix_sha256":sha256_hex(xv.tobytes()),"test_matrix_sha256":sha256_hex(xe.tobytes()),"pca_train_feature_sha256":_array_sha(ft),"pca_validation_feature_sha256":_array_sha(fv),"pca_test_feature_sha256":_array_sha(fe),"train_record_ids_sha256":_ids_sha(train_ids),"validation_record_ids_sha256":_ids_sha(valid_ids),"test_record_ids_sha256":_ids_sha(inputs.record_ids),"model_state_sha256":_canonical_sha({"selected_c":c,"classes":np.asarray(model.classes_,dtype="<i8").tolist(),"coef_sha256":_array_sha(model.coef_),"intercept_sha256":_array_sha(model.intercept_),"n_iter":np.asarray(model.n_iter_,dtype="<i8").tolist(),"pca_components_sha256":_array_sha(pca.components_),"pca_mean_sha256":_array_sha(pca.mean_),"pca_explained_variance_sha256":_array_sha(pca.explained_variance_)}),"warning_state":"none"}
    predictions=[]; seed_rows=[]
    for o,rid in enumerate(inputs.record_ids): predictions.append({"shot_count":shot,"model_seed":seed,"condition_id":condition,"record_order":o,"record_id":rid,"true_class":int(inputs.test_labels[o]),"predicted_class":int(predicted[o]),"correct":bool(predicted[o]==inputs.test_labels[o]),"projected_row_sha256":_array_sha(xe[o],"<f4")})
    for label in range(config.class_count): seed_rows.append({"shot_count":shot,"model_seed":seed,"class_label":label,"condition_id":condition,"accuracy":float(np.mean([x["correct"] for x in predictions if x["true_class"]==label]))})
    return cell,tuple(validation),tuple(predictions),tuple(seed_rows)


def _fit(inputs, config):
    cells=[]; validation=[]; predictions=[]; seed_rows=[]
    for shot in SHOTS:
        for seed in config.seeds:
            for condition in config.conditions:
                a,b,c,d=_fit_cell(shot,seed,condition,inputs,config); cells.append(a); validation.extend(b); predictions.extend(c); seed_rows.extend(d)
    return MappingProxyType({"model_cells":tuple(cells),"validation_rows":tuple(validation),"predictions":tuple(predictions),"seed_class_rows":tuple(seed_rows)})


def _aggregate(predictions, metrics, config, bootstrap_resamples: int, sign_flip_resamples: int):
    directions={**{x:PreferredDirection.LOWER_IS_BETTER for x in ("mse","rmse","mae","sam","nmse","wasserstein_1_cm1","artifact_peak_ratio","missing_peak_ratio")},**{x:PreferredDirection.HIGHER_IS_BETTER for x in ("pearson_r","is_like_structure_to_noise","precision","recall","f1")}}
    m={(int(x["record_order"]),str(x["condition_id"]),str(x["metric_output_id"])):float(x["value"]) for x in metrics}
    by_seed_class={}; by_class_orders={}
    for row in predictions:
        shot=int(row["shot_count"]); seed=int(row["model_seed"]); label=int(row["true_class"]); cond=str(row["condition_id"]); order=int(row["record_order"])
        by_seed_class.setdefault((shot,seed,label,cond),[]).append(bool(row["correct"]))
        by_class_orders.setdefault(label,set()).add(order)
    seed_acc={key:float(np.mean(values)) for key,values in by_seed_class.items()}
    obs=[]; align=[]; boots=[]; signs=[]; holms=[]
    for shot in SHOTS:
        tables={}; states={}
        for metric in METRIC_OUTPUT_IDS:
            table=[]
            for cond in config.conditions[1:]:
                perturbation, hx=cond.split(":",1); alpha=float(np.frombuffer(bytes.fromhex(hx),dtype="<f8")[0])
                for label in range(config.class_count):
                    harm_values=[m[(o,cond,metric)]-m[(o,"alpha0",metric)] for o in sorted(by_class_orders[label])]
                    if directions[metric] is PreferredDirection.HIGHER_IS_BETTER:
                        harm_values=[-value for value in harm_values]
                    harm=np.mean(harm_values)
                    baseline=[seed_acc[(shot,s,label,"alpha0")] for s in config.seeds]
                    current=[seed_acc[(shot,s,label,cond)] for s in config.seeds]
                    down=float(np.mean(baseline)-np.mean(current)); table.append(AlignmentObservation(str(label),perturbation,alpha,float(harm),down)); obs.append({"shot_count":shot,"metric_output_id":metric,"class_label":label,"condition_id":cond,"perturbation_id":perturbation,"alpha":alpha,"metric_harm":float(harm),"downstream_harm":down})
            tables[metric]=tuple(table)
            try: states[metric]=("complete",alignment_gap(table),cross_perturbation_accuracy(table))
            except Exception as error: states[metric]=("not_evaluable",error,None)
        reference=tables["mse"]; family={}
        for metric in METRIC_OUTPUT_IDS:
            state,gap,acc=states[metric]; row={"shot_count":shot,"metric_output_id":metric,"state":state,"ag":None,"ag_raw":None,"acc_cross":None,"ag_interval":None,"acc_interval":None,"d_ag":None,"d_acc":None,"d_ag_interval":None,"d_acc_interval":None}
            if state=="complete":
                if metric=="mse":
                    b=bulk_paired_cluster_bootstrap(reference,reference,resamples=bootstrap_resamples,random_seed=20260817); row.update({"ag":gap.alignment_gap,"ag_raw":gap.raw_alignment_gap,"acc_cross":acc.accuracy,"ag_interval":b.reference_ag_interval,"acc_interval":b.reference_acc_interval})
                else:
                    cmp=compare_alignment(reference,tables[metric]); b=bulk_paired_cluster_bootstrap(reference,tables[metric],resamples=bootstrap_resamples,random_seed=20260817); row.update({"ag":gap.alignment_gap,"ag_raw":gap.raw_alignment_gap,"acc_cross":acc.accuracy,"ag_interval":b.candidate_ag_interval,"acc_interval":b.candidate_acc_interval,"d_ag":cmp.d_ag,"d_acc":cmp.d_acc,"d_ag_interval":b.d_ag_interval,"d_acc_interval":b.d_acc_interval}); boots.append({"shot_count":shot,"metric_output_id":metric,"state":"complete","resamples":bootstrap_resamples,"candidate_ag_interval":b.candidate_ag_interval,"candidate_acc_interval":b.candidate_acc_interval,"d_ag_interval":b.d_ag_interval,"d_acc_interval":b.d_acc_interval})
                    for stat,values in (("d_ag",cmp.ag_contribution_differences),("d_acc",cmp.acc_contribution_differences)):
                        sign=paired_contribution_sign_flip([x.value for x in values],aggregation="sum" if stat=="d_ag" else "mean",resamples=sign_flip_resamples,random_seed=20260817); family[f"{metric}:{stat}"]=(sign.p_value,getattr(cmp,stat)); signs.append({"shot_count":shot,"metric_output_id":metric,"statistic":stat,"state":"complete","contrast":getattr(cmp,stat),"p_value":sign.p_value,"resamples":sign_flip_resamples})
            if metric != "mse" and not any(x["shot_count"]==shot and x["metric_output_id"]==metric for x in boots): boots.append({"shot_count":shot,"metric_output_id":metric,"state":"not_evaluable_metric_incomplete","resamples":None})
            align.append(row)
        for metric in METRIC_OUTPUT_IDS[1:]:
            for stat in ("d_ag","d_acc"):
                family.setdefault(f"{metric}:{stat}",(1.0,0.0))
                if not any(x["shot_count"]==shot and x["metric_output_id"]==metric and x["statistic"]==stat for x in signs): signs.append({"shot_count":shot,"metric_output_id":metric,"statistic":stat,"state":"not_tested_metric_incomplete","contrast":None,"p_value":1.0,"resamples":None})
        for result in holm_step_down({k:v[0] for k,v in family.items()},alpha=.05):
            metric,stat=result.hypothesis_id.split(":",1); contrast=family[result.hypothesis_id][1]; holms.append({"shot_count":shot,"metric_output_id":metric,"statistic":stat,"raw_p_value":result.raw_p_value,"adjusted_p_value":result.adjusted_p_value,"rank":result.rank,"family_size":result.family_size,"favorable":bool(contrast>0),"rejected":bool(result.rejected and contrast>0)})
    return MappingProxyType({"class_observations":tuple(obs),"alignment_results":tuple(align),"bootstrap_results":tuple(boots),"sign_flip_results":tuple(signs),"holm_family":tuple(holms)})


def _render(projection, config):
    payloads={}; colors=dict(zip(PERTURBATIONS,("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"),strict=True))
    for shot in SHOTS:
        with matplotlib.rc_context({"font.family":"DejaVu Sans","lines.linewidth":1.5,"svg.hashsalt":f"rpe-phase4-d2-{shot}shot-v1"}):
            obs=[x for x in projection["class_observations"] if x["shot_count"]==shot]; rows=[]
            for metric in METRIC_OUTPUT_IDS:
                for perturbation in PERTURBATIONS:
                    for alpha in ALPHAS[1:]:
                        sel=[x for x in obs if x["metric_output_id"]==metric and x["perturbation_id"]==perturbation and x["alpha"]==alpha]; rows.append({"shot_count":shot,"metric_output_id":metric,"perturbation_id":perturbation,"alpha":alpha,"mean_metric_harm":float(np.mean([x["metric_harm"] for x in sel])) if sel else None,"mean_downstream_harm":float(np.mean([x["downstream_harm"] for x in sel])) if sel else None})
            f2=[x for x in projection["alignment_results"] if x["shot_count"]==shot]; prefix=f"d2_{shot}shot_protocol_b_full_domain"
            fig,axes=plt.subplots(4,4,figsize=(12,12))
            for ax,metric in zip(axes.ravel(),METRIC_OUTPUT_IDS,strict=False):
                for p in PERTURBATIONS:
                    rs=[x for x in rows if x["metric_output_id"]==metric and x["perturbation_id"]==p]; ax.plot([x["mean_metric_harm"] for x in rs],[x["mean_downstream_harm"] for x in rs],marker="o",color=colors[p])
                ax.set_title(metric)
            for ax in axes.ravel()[13:]: ax.set_axis_off()
            fig.tight_layout(); png=io.BytesIO(); svg=io.BytesIO(); fig.savefig(png,format="png",dpi=300,metadata={"Date":None}); fig.savefig(svg,format="svg",metadata={"Date":None}); plt.close(fig); payloads[f"figure1_{prefix}.png"]=png.getvalue(); payloads[f"figure1_{prefix}.svg"]=svg.getvalue(); payloads[f"figure1_{prefix}_data.csv"]=csv_bytes(rows)
            fig,axes=plt.subplots(1,4,figsize=(14,8),sharey=True)
            for ax,field in zip(axes,("ag","acc_cross","d_ag","d_acc"),strict=True): ax.barh(np.arange(len(f2)),[0 if x.get(field) is None else float(x[field]) for x in f2]); ax.set_title(field)
            fig.tight_layout(); png=io.BytesIO(); svg=io.BytesIO(); fig.savefig(png,format="png",dpi=300,metadata={"Date":None}); fig.savefig(svg,format="svg",metadata={"Date":None}); plt.close(fig); payloads[f"figure2_{prefix}.png"]=png.getvalue(); payloads[f"figure2_{prefix}.svg"]=svg.getvalue(); payloads[f"figure2_{prefix}_data.csv"]=csv_bytes(f2)
    return payloads


def _condition_summary(predictions, config):
    grouped={}
    for row in predictions:
        grouped.setdefault((int(row["shot_count"]),str(row["condition_id"])),[]).append(row)
    rows=[]
    for shot in SHOTS:
        for condition in config.conditions:
            selected=grouped[(shot,condition)]; per=[]; f1=[]
            for label in range(config.class_count):
                cls=[row for row in selected if row["true_class"]==label]; per.append(float(np.mean([row["correct"] for row in cls])))
                tp=sum(row["true_class"]==label and row["predicted_class"]==label for row in selected); fp=sum(row["true_class"]!=label and row["predicted_class"]==label for row in selected); fn=sum(row["true_class"]==label and row["predicted_class"]!=label for row in selected)
                f1.append(0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
            rows.append({"shot_count":shot,"condition_id":condition,"prediction_count":len(selected),"macro_top1_accuracy":float(np.mean(per)),"micro_top1_accuracy":float(np.mean([row["correct"] for row in selected])),"macro_f1":float(np.mean(f1))})
    return tuple(rows)


def _artifact_documents(
    *, config: _Config, bridge: Mapping[str, object], science: Mapping[str, object],
    fitted, alpha_doc: Mapping[str, object], projection, summary_rows,
    run_id: str, worker_count: int, bootstrap_resamples: int,
    sign_flip_resamples: int,
):
    code=dict(config.document.get("code_authority",{})); environment=dict(config.document.get("environment_authority",{}))
    parent_artifacts=_parent_artifact_receipts(config,bridge["document"]); shot_gate_states=_shot_gate_states(config); expected=config.document.get("expected",{})
    operator_cells=int(expected.get("operator_cell_count",len(config.conditions))); apply_checks=int(expected.get("apply_check_count",operator_cells*len(ALPHAS))); rematerialized=int(science.get("rematerialized_row_count",0))
    counts={"alignment_results":len(projection["alignment_results"]),"apply_checks":apply_checks,"artifact_files":len(ARTIFACT_PAYLOAD_FILES)+2,"authorized_cwt_receipts":int(bridge["document"].get("cwt_row_count",0)),"authorized_metric_values":int(bridge["document"].get("metric_row_count",len(bridge.get("metrics",())))),"bootstrap_results":len(projection["bootstrap_results"]),"bridge_rows":int(bridge["document"].get("bridge_row_count",0)),"class_observations":len(projection["class_observations"]),"condition_summary_rows":len(summary_rows),"configured_payloads":len(ARTIFACT_PAYLOAD_FILES),"holm_family":len(projection["holm_family"]),"model_cells":len(fitted["model_cells"]),"operator_cells":operator_cells,"predictions":len(fitted["predictions"]),"rematerialized_source_conditions":rematerialized,"secondary_table_rows":len(projection["alignment_results"]),"seed_class_conditions":len(fitted["seed_class_rows"]),"sign_flip_results":len(projection["sign_flip_results"]),"validation_scores":len(fitted["validation_rows"])}
    metric_states={f"{int(row['shot_count'])}:{row['metric_output_id']}":str(row["state"]) for row in projection["alignment_results"]}
    run_identity={"artifact_schema_version":ARTIFACT_SCHEMA_VERSION,"authorities":config.document.get("authorities",{}),"authority_bridge":dict(bridge["document"]),"claim_boundary":CLAIM_BOUNDARY,"code":code,"config_authority":_config_authority(),"config_sha256":sha256_hex(config.raw),"denominators":config.document.get("denominators",{}),"environment":environment,"frozen_identities":config.document.get("frozen_identities",{}),"inherited_rulings":config.document.get("inherited_rulings",{}),"parent_artifacts":parent_artifacts,"support_grid":config.document.get("support_grid",{}),"trust_anchor":config.document.get("trust_anchor",{})}
    manifest={"alpha0_equivalence":dict(alpha_doc),"artifact_payload_files":list(ARTIFACT_PAYLOAD_FILES),"artifact_schema_version":ARTIFACT_SCHEMA_VERSION,"authorities":config.document.get("authorities",{}),"bootstrap_result_count":len(projection["bootstrap_results"]),"claim_boundary":CLAIM_BOUNDARY,"class_observation_count":len(projection["class_observations"]),"code":code,"config":{"bytes":len(config.raw),"sha256":sha256_hex(config.raw)},"config_authority":_config_authority(),"counts":counts,"environment":environment,"experiment_id":EXPERIMENT_ID,"holm_family_count":len(projection["holm_family"]),"inherited_rulings":config.document.get("inherited_rulings",{}),"metric_states":metric_states,"parent_artifacts":parent_artifacts,"prediction_row_count":len(fitted["predictions"]),"protocol":"B","run_id":run_id,"run_identity":run_identity,"seed_class_condition_count":len(fitted["seed_class_rows"]),"shot_endpoint_states":{str(x):"complete" for x in SHOTS},"sign_flip_result_count":len(projection["sign_flip_results"]),"status":"complete","synthetic_fixture":config.synthetic,"tier":"full_domain_core"}
    preflight={"authority_bridge_state":"complete","bootstrap_resamples":bootstrap_resamples,"claim_boundary":"pre_model_bridges_and_rematerialization_complete","metric_authority_state":"complete","parent_artifacts":parent_artifacts,"rematerialization":{"blas_thread_limit":int(science.get("blas_thread_limit",1)),"condition_count":len(config.conditions),"matrix_input_state":"ready","p10_estimated_peak_bytes":int(science.get("p10_estimated_peak_bytes",1105805824)),"process_start_method":str(science.get("process_start_method","spawn")),"receipt_mismatch_count":0,"receipt_sha256":str(science["rematerialization_receipt_sha256"]),"source_condition_count":rematerialized,"state":"complete"},"shot_gate_states":shot_gate_states,"sign_flip_resamples":sign_flip_resamples,"status":"complete","synthetic_fixture":config.synthetic,"worker_count":worker_count}
    return manifest,preflight


def _serialize_rebuild(config: _Config, bridge: Mapping[str, object], fitted, alpha_doc: Mapping[str, object], projection, *, science: Mapping[str, object] | None = None, worker_count: int, bootstrap_resamples: int, sign_flip_resamples: int):
    run_id="phase4-d2-protocol-b-full-domain-"+sha256_hex(canonical_json_bytes({"config_sha256":sha256_hex(config.raw),"bridge_sha256":bridge["digest"],"model_seeds":list(config.seeds),"shots":list(SHOTS)}))
    summary=_condition_summary(fitted["predictions"],config)
    alpha_value=dict(alpha_doc)
    science = science or {"blas_thread_limit":1,"process_start_method":"spawn","p10_estimated_peak_bytes":1105805824,"rematerialized_row_count":config.class_count*config.records_per_class*len(config.conditions),"rematerialization_receipt_sha256":_canonical_sha({"condition_ids":list(config.conditions),"record_ids":list(getattr(fitted,"record_ids",())) if not isinstance(fitted,Mapping) else [],"synthetic_fixture":True})}
    manifest,preflight=_artifact_documents(config=config,bridge=bridge,science=science,fitted=fitted,alpha_doc=alpha_doc,projection=projection,summary_rows=summary,run_id=run_id,worker_count=worker_count,bootstrap_resamples=bootstrap_resamples,sign_flip_resamples=sign_flip_resamples)
    values={"config.json":config.raw,"authority_bridge.json":canonical_json_bytes(dict(bridge["document"])),"preflight.json":canonical_json_bytes(preflight),"alpha0_equivalence.json":canonical_json_bytes(alpha_value),"model_cells.jsonl":jsonl_bytes(fitted["model_cells"]),"validation_scores.jsonl":jsonl_bytes(fitted["validation_rows"]),"predictions.jsonl":jsonl_bytes(fitted["predictions"]),"seed_class_conditions.jsonl":jsonl_bytes(fitted["seed_class_rows"]),"condition_summary.csv":csv_bytes(summary),"class_observations.jsonl":jsonl_bytes(projection["class_observations"]),"alignment_results.jsonl":jsonl_bytes(projection["alignment_results"]),"bootstrap_results.jsonl":jsonl_bytes(projection["bootstrap_results"]),"sign_flip_results.jsonl":jsonl_bytes(projection["sign_flip_results"]),"holm_family.jsonl":jsonl_bytes(projection["holm_family"]),"d2_protocol_b_full_domain_secondary_table.csv":csv_bytes(projection["alignment_results"]),"manifest.json":canonical_json_bytes(manifest)}; values.update(_render(projection,config))
    payloads={name:values[name] for name in ARTIFACT_PAYLOAD_FILES}
    terminal=canonical_json_bytes({"run_id":run_id,"status":"complete"}); return MappingProxyType({**payloads,"complete.json":terminal,"SHA256SUMS":write_sha256sums(payloads,"complete.json",terminal)}),manifest


def _rebuild(inputs, config, *, worker_count: int, bootstrap_resamples: int, sign_flip_resamples: int):
    if worker_count < 1: raise Phase4D2ProtocolBVerifierError("worker_count must be positive")
    bridge=_bridge(inputs,config); fitted=_fit(inputs,config); projection=_aggregate(fitted["predictions"],bridge["metrics"],config,bootstrap_resamples,sign_flip_resamples)
    alpha_doc=_alpha0_receipt(fitted,config,bridge["digest"])
    synthetic_science={"blas_thread_limit":1,"process_start_method":"spawn","p10_estimated_peak_bytes":1105805824,"rematerialized_row_count":len(inputs.record_ids)*len(config.conditions),"rematerialization_receipt_sha256":_canonical_sha({"condition_ids":list(config.conditions),"record_ids":list(inputs.record_ids),"synthetic_fixture":True})}
    return _serialize_rebuild(config,bridge,fitted,alpha_doc,projection,science=synthetic_science,worker_count=worker_count,bootstrap_resamples=bootstrap_resamples,sign_flip_resamples=sign_flip_resamples)


def _compare(path: Path, rebuilt: Mapping[str, bytes]) -> None:
    for name, raw in rebuilt.items():
        actual=(path/name).read_bytes()
        if actual != raw:
            raise Phase4D2ProtocolBVerifierError(
                f"semantic rebuild payload mismatch: {name}"
            )


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise Phase4D2ProtocolBVerifierError(f"{path.name}:{number}: malformed JSON") from error
        if canonical_json_bytes(row) != (line + "\n").encode():
            raise Phase4D2ProtocolBVerifierError(f"{path.name}:{number}: noncanonical row")
        rows.append(row)
    return tuple(rows)


def _validate_parent_tree(path: Path, authority: Mapping[str, object]) -> None:
    payload_hashes = authority.get("payload_sha256")
    if not isinstance(payload_hashes, Mapping) or not path.is_dir():
        raise Phase4D2ProtocolBVerifierError("parent artifact schema/path mismatch")
    expected_files = set(payload_hashes) | {"SHA256SUMS"}
    if {item.name for item in path.iterdir()} != expected_files:
        raise Phase4D2ProtocolBVerifierError("parent artifact inventory mismatch")
    if _sha_file(path / "SHA256SUMS") != authority.get("sha256sums_sha256"):
        raise Phase4D2ProtocolBVerifierError("parent SHA256SUMS identity mismatch")
    for name, digest in payload_hashes.items():
        if _sha_file(path / str(name)) != str(digest):
            raise Phase4D2ProtocolBVerifierError(f"parent payload mismatch: {name}")
    sums = {}
    for line in (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1); sums[name] = digest
    if sums != {str(k): str(v) for k, v in payload_hashes.items()}:
        raise Phase4D2ProtocolBVerifierError("parent checksum schema mismatch")
    marker_name = "complete.json" if "complete.json" in payload_hashes else "failed.json"
    marker = json.loads((path / marker_name).read_bytes()); manifest = json.loads((path / "manifest.json").read_bytes())
    expected_run = str(authority["payload_run_id"]); allowed = set(authority["completion_statuses"])
    manifest_status=manifest.get("status",manifest.get("overall_status",marker.get("status")))
    if marker.get("run_id") != expected_run or manifest.get("run_id") != expected_run or marker.get("status") not in allowed or manifest_status not in allowed:
        raise Phase4D2ProtocolBVerifierError("parent completion identity mismatch")


def _unique_rows(rows: Sequence[Mapping[str, object]], fields: Sequence[str], boundary: str) -> dict[tuple[object, ...], Mapping[str, object]]:
    result = {}
    for row in rows:
        key = tuple(row.get(field) for field in fields)
        if None in key or key in result:
            raise Phase4D2ProtocolBVerifierError(f"{boundary} duplicate/malformed key")
        result[key] = row
    return result


def _validate_real_parents_and_bridge(config: _Config, inputs: _RealInputs) -> Mapping[str, object]:
    parents = config.document.get("parent_artifacts")
    if not isinstance(parents, Mapping): raise Phase4D2ProtocolBVerifierError("missing dual parents")
    protocol_a = ROOT / str(parents["protocol_a"]["relative_path"]); eligibility = ROOT / str(parents["eligibility"]["relative_path"])
    _validate_parent_tree(protocol_a, parents["protocol_a"]); _validate_parent_tree(eligibility, parents["eligibility"])
    for forbidden in FORBIDDEN_STEP15:
        if forbidden not in parents["protocol_a"]["payload_sha256"]:
            raise Phase4D2ProtocolBVerifierError("Step-15 firewall schema incomplete")
    a_conditions = _read_jsonl(protocol_a / "record_conditions.jsonl")
    b_conditions = _read_jsonl(eligibility / "record_conditions.jsonl")
    metrics = _read_jsonl(protocol_a / "metric_values.jsonl"); cwt = _read_jsonl(protocol_a / "peak_receipts.jsonl")
    if len(a_conditions) != 123000 or len(b_conditions) != 226033 or len(metrics) != 1599000 or len(cwt) != 123000:
        raise Phase4D2ProtocolBVerifierError("dual parent denominator mismatch")
    a_map = _unique_rows(a_conditions, ("record_id", "condition_id"), "Step-15 conditions")
    b_map = _unique_rows(b_conditions, ("record_id", "condition_id"), "Step-17 conditions")
    sources = _unique_rows(_read_jsonl(eligibility / "source_records.jsonl"), ("record_id",), "Step-17 sources")
    parent_models = _read_jsonl(eligibility / "model_cells.jsonl")
    parent_roles = _read_jsonl(eligibility / "model_role_occurrences.jsonl")
    if (jsonl_bytes(inputs.source_records) != jsonl_bytes(tuple(sources.values()))
            or jsonl_bytes(tuple(sorted(inputs.model_cells, key=lambda row: (int(row["seed"]), int(row["shot_count"]))))) != jsonl_bytes(parent_models)
            or jsonl_bytes(inputs.model_role_occurrences) != jsonl_bytes(parent_roles)):
        raise Phase4D2ProtocolBVerifierError("independent selection/source/role reconstruction differs from Step-17")
    bridge_rows = []
    for order, record_id in enumerate(inputs.record_ids):
        source = sources.get((record_id,))
        if source is None or source.get("scope") != "test" or int(source.get("role_count", -1)) != 15 or source.get("shot_memberships") != [5, 10, 20]:
            raise Phase4D2ProtocolBVerifierError("Step-17 test source role mismatch")
        for condition in config.conditions:
            a = a_map.get((record_id, condition)); b = b_map.get((record_id, condition))
            if a is None or b is None:
                raise Phase4D2ProtocolBVerifierError("dual-parent bridge missing key")
            if int(a.get("record_order", -1)) != order or a.get("state") != "complete" or b.get("state") != "complete":
                raise Phase4D2ProtocolBVerifierError("dual-parent bridge state/order mismatch")
            if (int(source["class_label"]) != int(b.get("class_label", -1))
                    or a.get("axis_sha256") != b.get("axis_sha256")
                    or a.get("intensity_sha256") != b.get("intensity_sha256")
                    or a.get("projected_row_sha256") != b.get("support_projection_sha256")):
                raise Phase4D2ProtocolBVerifierError("dual-parent bridge field mismatch")
            bridge_rows.append({"test_order": order, "record_id": record_id, "class_label": int(source["class_label"]), "condition_id": condition, "state": "complete", "axis_sha256": a["axis_sha256"], "intensity_sha256": a["intensity_sha256"], "support_projection_sha256": a["projected_row_sha256"]})
    expected_keys = {(record_id, condition) for record_id in inputs.record_ids for condition in config.conditions}
    if set(a_map) != expected_keys or not expected_keys <= set(b_map):
        raise Phase4D2ProtocolBVerifierError("dual-parent bridge extras/missing keys")
    digest = sha256_hex(jsonl_bytes(bridge_rows))
    if digest != str(config.document["authority_bridge"]["condition_bridge_sha256"]):
        raise Phase4D2ProtocolBVerifierError("exact dual-parent condition bridge digest mismatch")
    metric_map = _unique_rows(metrics, ("record_id", "condition_id", "metric_output_id"), "Step-15 metrics")
    cwt_map = _unique_rows(cwt, ("record_id", "condition_id"), "Step-15 CWT")
    expected_metric = {(rid, condition, metric) for rid in inputs.record_ids for condition in config.conditions for metric in METRIC_OUTPUT_IDS}
    if (set(metric_map) != expected_metric or set(cwt_map) != expected_keys
            or any(row.get("state") != "complete" or not math.isfinite(float(row["value"])) for row in metrics)
            or any(row.get("state") != "complete" for row in cwt)):
        raise Phase4D2ProtocolBVerifierError("Step-15 measurement authority malformed")
    document = {"protocol_a_run_id": protocol_a.name, "protocol_a_sha256sums_sha256": _sha_file(protocol_a / "SHA256SUMS"), "eligibility_run_id": eligibility.name, "eligibility_sha256sums_sha256": _sha_file(eligibility / "SHA256SUMS"), "bridge_row_count": len(bridge_rows), "metric_row_count": len(metrics), "cwt_row_count": len(cwt), "condition_bridge_sha256": digest, "missing_key_count": 0, "extra_key_count": 0, "duplicate_key_count": 0, "field_mismatch_count": 0}
    return MappingProxyType({"protocol_a": protocol_a, "eligibility": eligibility, "metrics": metrics, "document": MappingProxyType(document), "digest": digest, "b_conditions": b_conditions})


def _reconstruct_real_inputs(config: _Config) -> _RealInputs:
    selection_path = ROOT / str(config.document["authorities"]["d2_selection_artifact"]["path"])
    dataset_path = ROOT / "data/unified/bacteria_id_reference"
    if _sha_file(selection_path) != str(config.document["authorities"]["d2_selection_artifact"]["sha256"]):
        raise Phase4D2ProtocolBVerifierError("selection identity mismatch")
    selection = json.loads(selection_path.read_bytes())
    try:
        validate_d2_few_shot_selection(selection, ROOT / "experiments/phase05/configs/d2_few_shot_selection.json", dataset_path)
    except Exception as error:
        raise Phase4D2ProtocolBVerifierError(f"selection replay failure: {error}") from error
    roles: dict[tuple[int,int,str], tuple[str,...]] = {}; required=set(); labels_by_id={}; memberships: dict[str, set[int]] = {}
    for seed_doc in selection["selections"]:
        seed=int(seed_doc["seed"]); by_shot={shot:[] for shot in SHOTS}; validation=[]
        for cls in seed_doc["classes"]:
            label=int(cls["class_label"]); val=tuple(map(str,cls["validation_record_ids"])); validation.extend(val)
            for rid in val: labels_by_id[rid]=label; required.add(rid); memberships.setdefault(rid,set()).update(SHOTS)
            for shot in SHOTS:
                ids=tuple(map(str,cls["train_record_ids"][str(shot)])); by_shot[shot].extend(ids)
                for rid in ids: labels_by_id[rid]=label; required.add(rid); memberships.setdefault(rid,set()).add(shot)
        for shot in SHOTS:
            roles[(shot,seed,"train")]=tuple(by_shot[shot]); roles[(shot,seed,"validation")]=tuple(validation)
    support_doc=json.loads((ROOT / "experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json").read_bytes()); support=np.asarray(support_doc["support_grid"]["coordinates_cm1"],dtype="<f8"); max_gap=float(support_doc["support_grid"]["max_in_range_native_gap_cm1"])
    sources={}; spectra={}; test_ids=[]
    with BacteriaIdBatchLoader(dataset_path,batch_size=4096) as loader:
        for batch in loader.iter_batches():
            if batch.source_split not in {"finetune","test"}: continue
            axis=np.asarray(batch.wavenumber,dtype="<f4")[::-1].astype("<f8")
            for rid,label,intensity,row in zip(batch.record_ids,batch.class_labels,batch.intensity,batch.source_rows,strict=True):
                rid=str(rid)
                if batch.source_split=="finetune" and rid not in required: continue
                value=np.asarray(intensity,dtype="<f4")[::-1].astype("<f8"); spectrum=Spectrum1D(f"bacteria_id_reference::{rid}",None,axis,value); spectra[rid]=spectrum; labels_by_id[rid]=int(label)
                if batch.source_split=="test": test_ids.append(rid); memberships.setdefault(rid,set()).update(SHOTS)
                projected=_project_support(spectrum,support,max_gap)
                sources[rid]={"class_label":int(label),"native_axis_sha256":_array_sha(axis),"native_intensity_sha256":_array_sha(value),"record_id":rid,"scope":str(batch.source_split),"shot_memberships":sorted(memberships[rid]),"source_row":int(row),"support_projection_sha256":_array_sha(projected,"<f4")}
    test_ids=sorted(test_ids)
    if len(test_ids)!=3000 or _ids_sha(test_ids)!=str(selection["test"]["record_ids_sha256"]): raise Phase4D2ProtocolBVerifierError("test cohort identity mismatch")
    for shot in SHOTS:
        for seed in config.seeds: roles[(shot,seed,"test")]=tuple(test_ids)
    all_ids=sorted(spectra); occurrences=[]
    cells=[]
    for shot in SHOTS:
        for seed in config.seeds:
            cell={"seed":seed,"shot_count":shot,"train_record_ids":tuple(sorted(roles[(shot,seed,"train")])),"validation_record_ids":tuple(sorted(roles[(shot,seed,"validation")])),"test_record_ids":roles[(shot,seed,"test")]}; cells.append(cell)
            for role in ("train","validation","test"):
                occurrences.extend({"record_id":rid,"role":role,"seed":seed,"shot_count":shot} for rid in roles[(shot,seed,role)])
    counts={rid:0 for rid in all_ids}
    for row in occurrences: counts[str(row["record_id"])]+=1
    records=tuple({**sources[rid],"record_order":i,"role_count":counts[rid]} for i,rid in enumerate(all_ids))
    order={rid:i for i,rid in enumerate(all_ids)}
    occurrences.sort(key=lambda row:(int(row["seed"]),int(row["shot_count"]),{"train":0,"validation":1,"test":2}[str(row["role"])],order[str(row["record_id"])]))
    if (len(records),len(cells),len(occurrences)) != (5513,15,54750): raise Phase4D2ProtocolBVerifierError("real selection/role reconstruction denominator mismatch")
    return _RealInputs(config.class_count,config.records_per_class,config.seeds,tuple(test_ids),np.asarray([labels_by_id[x] for x in test_ids],dtype="<i8"),records,tuple(spectra[x] for x in all_ids),tuple(cells),MappingProxyType(roles),MappingProxyType(labels_by_id),tuple(occurrences))


_VERIFY_SWEEP: object | None = None
_VERIFY_PHASE1: object | None = None
_VERIFY_SUPPORT: np.ndarray | None = None
_VERIFY_MAX_GAP: float | None = None
_VERIFY_P10_BUDGET: int | None = None


def _initialize_verify_worker(sweep_path: str, phase1_path: str, support: tuple[float, ...], max_gap: float, budget: int) -> None:
    global _VERIFY_SWEEP, _VERIFY_PHASE1, _VERIFY_SUPPORT, _VERIFY_MAX_GAP, _VERIFY_P10_BUDGET
    _VERIFY_SWEEP=load_perturbation_sweep_config(Path(sweep_path)); _VERIFY_PHASE1=load_phase1_core_config(Path(phase1_path))
    _VERIFY_SUPPORT=np.asarray(support,dtype="<f8"); _VERIFY_MAX_GAP=float(max_gap); _VERIFY_P10_BUDGET=int(budget)


def _verify_source_job(job: tuple[int, Mapping[str, object], Spectrum1D]) -> tuple[int, str, dict[str, tuple[np.ndarray,str,str]]]:
    if any(value is None for value in (_VERIFY_SWEEP,_VERIFY_PHASE1,_VERIFY_SUPPORT,_VERIFY_MAX_GAP,_VERIFY_P10_BUDGET)):
        raise Phase4D2ProtocolBVerifierError("rematerialization worker was not initialized")
    order,record,spectrum=job; record_id=str(record["record_id"])
    source=Phase1Source(SelectedSourceRow(int(record["record_order"]),record_id,record_id,int(record["class_label"]),f"bacteria-{record['class_label']}",f"native::{record['native_axis_sha256']}"),spectrum,"increasing",_array_sha(spectrum.axis_cm1,"<f4"),_array_sha(spectrum.intensity,"<f4"),str(record["native_axis_sha256"]),str(record["native_intensity_sha256"]),MappingProxyType({"dataset_id":"bacteria_id_reference","record_id":record_id}))
    results={"alpha0":(_project_support(spectrum,_VERIFY_SUPPORT,float(_VERIFY_MAX_GAP)),_array_sha(spectrum.axis_cm1),_array_sha(spectrum.intensity))}
    admission=P10MemoryAdmission(int(_VERIFY_P10_BUDGET))
    with threadpool_limits(limits=1,user_api="blas"):
        for perturbation in PERTURBATIONS:
            cell=run_perturbation_cell(source,perturbation,_VERIFY_PHASE1,_VERIFY_SWEEP,p10_admission=admission if perturbation=="p10" else None)
            if cell.status.name != "COMPLETE": raise Phase4D2ProtocolBVerifierError(f"rematerialization failed for {record_id} {perturbation}")
            for item in cell.records[1:]:
                results[f"{perturbation}:{item.alpha_float64_le_hex}"]=(np.asarray(_project_support(item.result.output,_VERIFY_SUPPORT,float(_VERIFY_MAX_GAP)),dtype="<f4"),_array_sha(item.result.output.axis_cm1),_array_sha(item.result.output.intensity))
    return order,record_id,results


def _rematerialize_real_science(inputs: _RealInputs, config: _Config, *, worker_count: int, eligibility: Path | None = None):
    if worker_count < 1: raise Phase4D2ProtocolBVerifierError("worker_count must be positive")
    budget=int(config.document["p10"]["memory_budget_bytes"]); estimates={estimate_p10_peak_bytes(int(x.axis_cm1.size)) for x in inputs.source_spectra}
    if estimates != {1105805824}: raise Phase4D2ProtocolBVerifierError("P10 admission estimate mismatch")
    capacity=budget//1105805824
    if capacity < 1: raise Phase4D2ProtocolBVerifierError("P10 capacity is zero")
    process_count=min(worker_count,len(inputs.source_records),capacity)
    support=np.asarray(config.document["support_grid"].get("coordinates_cm1",json.loads((ROOT/"experiments/phase4/configs/d2_protocol_a_full_domain_eligibility_v1.json").read_bytes())["support_grid"]["coordinates_cm1"]),dtype="<f8")
    max_gap=float(config.document["support_grid"]["max_in_range_native_gap_cm1"]); parent=eligibility or ROOT/str(config.document["parent_artifacts"]["eligibility"]["relative_path"])
    expected=_unique_rows(_read_jsonl(parent/"record_conditions.jsonl"),("record_id","condition_id"),"Step-17 rematerialization")
    matrices={condition:{} for condition in config.conditions}; jobs=tuple((i,row,spectrum) for i,(row,spectrum) in enumerate(zip(inputs.source_records,inputs.source_spectra,strict=True)))
    initargs=(str(ROOT/"experiments/shared/raman_perturbation_sweep_v1.json"),str(ROOT/"experiments/phase1/configs/rruff_raw_core10k_v1.json"),tuple(float(x) for x in support),max_gap,budget)
    receipt_hasher=hashlib.sha256()
    with ProcessPoolExecutor(max_workers=process_count,mp_context=multiprocessing.get_context("spawn"),initializer=_initialize_verify_worker,initargs=initargs) as executor:
        for expected_order,(order,record_id,results) in enumerate(executor.map(_verify_source_job,jobs)):
            if order != expected_order or tuple(results) != config.conditions: raise Phase4D2ProtocolBVerifierError("rematerialization canonical collection mismatch")
            for condition in config.conditions:
                projected,axis_sha,intensity_sha=results[condition]; authority=expected.get((record_id,condition))
                projection_sha=_array_sha(projected,"<f4")
                if authority is None or authority.get("state")!="complete" or axis_sha!=authority.get("axis_sha256") or intensity_sha!=authority.get("intensity_sha256") or projection_sha!=authority.get("support_projection_sha256"):
                    raise Phase4D2ProtocolBVerifierError(f"rematerialization receipt mismatch: {record_id}/{condition}")
                receipt_hasher.update(canonical_json_bytes({"axis_sha256":axis_sha,"condition_id":condition,"intensity_sha256":intensity_sha,"record_id":record_id,"support_projection_sha256":projection_sha}))
                matrices[condition][record_id]=projected
    if len(expected)!=226033 or any(len(rows)!=5513 for rows in matrices.values()): raise Phase4D2ProtocolBVerifierError("rematerialization denominator mismatch")
    return MappingProxyType({"worker_count":worker_count,"process_start_method":"spawn","blas_thread_limit":1,"p10_estimated_peak_bytes":1105805824,"rematerialized_row_count":226033,"rematerialization_receipt_sha256":receipt_hasher.hexdigest(),"condition_matrices":MappingProxyType({k:MappingProxyType(v) for k,v in matrices.items()}),"real":True})


def _fit_real_cells(inputs: _RealInputs, science: Mapping[str, object], config: _Config):
    matrices=science["condition_matrices"]; train={}; valid={}; test={}; train_y={}; valid_y={}
    for condition in config.conditions:
        matrix=matrices[condition]; test[condition]=np.asarray([matrix[x] for x in inputs.record_ids],dtype="<f4")
        for shot in SHOTS:
            for seed in config.seeds:
                tr=inputs.role_lookup[(shot,seed,"train")]; va=inputs.role_lookup[(shot,seed,"validation")]; train[(shot,seed,condition)]=np.asarray([matrix[x] for x in tr],dtype="<f4"); valid[(shot,seed,condition)]=np.asarray([matrix[x] for x in va],dtype="<f4"); train_y[(shot,seed)]=np.asarray([inputs.labels_by_id[x] for x in tr]); valid_y[(shot,seed)]=np.asarray([inputs.labels_by_id[x] for x in va])
    holder=type("RealHolder",(),{"train_values":train,"validation_values":valid,"test_values":test,"train_labels":train_y,"validation_labels":valid_y,"record_ids":inputs.record_ids,"test_labels":inputs.test_labels,"role_lookup":inputs.role_lookup})()
    return _fit(holder,config)


def _alpha0_projections(fitted: Mapping[str, Sequence[Mapping[str, object]]]) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    model_fields=("model_seed","model_state_sha256","pca_train_feature_sha256","pca_validation_feature_sha256","selected_c","shot_count","train_matrix_sha256","train_record_ids_sha256","validation_matrix_sha256","validation_record_ids_sha256","warning_state")
    validation_fields=("c","model_seed","shot_count","validation_top1_accuracy")
    prediction_fields=("condition_id","correct","model_seed","predicted_class","projected_row_sha256","record_id","record_order","shot_count","true_class")
    models=tuple({key:row[key] for key in model_fields} for row in sorted((r for r in fitted["model_cells"] if r.get("condition_id","alpha0")=="alpha0"),key=lambda r:(r["shot_count"],r["model_seed"])))
    validation=tuple({key:row[key] for key in validation_fields} for row in sorted((r for r in fitted["validation_rows"] if r.get("condition_id","alpha0")=="alpha0"),key=lambda r:(r["shot_count"],r["model_seed"],r["c"])))
    predictions=tuple({key:row[key] for key in prediction_fields} for row in sorted((r for r in fitted["predictions"] if r["condition_id"]=="alpha0"),key=lambda r:(r["shot_count"],r["model_seed"],r["record_order"])))
    return MappingProxyType({"model_cells_sha256":models,"validation_scores_sha256":validation,"predictions_sha256":predictions})


def _alpha0_receipt(fitted, config: _Config, bridge_digest: str) -> Mapping[str, object]:
    projections=_alpha0_projections(fitted); digests={name:sha256_hex(jsonl_bytes(rows)) for name,rows in projections.items()}
    mismatches={name:0 for name in digests}
    return MappingProxyType({**digests,"prediction_digest":digests["predictions_sha256"],"bridge_digest":bridge_digest,"expected_step18_digest":None if config.synthetic else bridge_digest,"mismatch_counts":mismatches,"mismatch_count":0})


def _validate_real_alpha0_equivalence(fitted, protocol_a: Path, config: _Config):
    current=_alpha0_projections(fitted)
    parent=_alpha0_projections({"model_cells":_read_jsonl(protocol_a/"model_cells.jsonl"),"validation_rows":_read_jsonl(protocol_a/"validation_scores.jsonl"),"predictions":_read_jsonl(protocol_a/"predictions.jsonl")})
    counts=tuple(len(current[name]) for name in ("model_cells_sha256","validation_scores_sha256","predictions_sha256"))
    expected_counts=(15,60,45000)
    if getattr(config,"expected",{}).get("model_cell_count") == 615 and counts != expected_counts: raise Phase4D2ProtocolBVerifierError("alpha0 denominator mismatch")
    digests={name:sha256_hex(jsonl_bytes(rows)) for name,rows in current.items()}
    parent_digests={name:sha256_hex(jsonl_bytes(rows)) for name,rows in parent.items()}
    expected=config.document.get("alpha0_equivalence",{}); mismatches={name:int(digests[name]!=parent_digests[name] or digests[name]!=expected.get(name)) for name in digests}
    if any(mismatches.values()): raise Phase4D2ProtocolBVerifierError("alpha-zero equivalence digest mismatch")
    return MappingProxyType({**digests,"prediction_digest":digests["predictions_sha256"],"bridge_digest":str(config.document.get("authority_bridge",{}).get("condition_bridge_sha256","")),"expected_step18_digest":str(config.document.get("authority_bridge",{}).get("condition_bridge_sha256","")),"mismatch_counts":mismatches,"mismatch_count":sum(mismatches.values())})


def verify_phase4_d2_protocol_b_from_inputs(path: Path, *, inputs: object, config_path: Path, worker_count: int, bootstrap_resamples: int | None = None, sign_flip_resamples: int | None = None) -> Phase4D2ProtocolBVerifierSummary:
    config=_parse_config(Path(config_path),Path(config_path).read_bytes(),frozen=False); _validate_inventory(Path(path),config)
    if not config.synthetic: raise Phase4D2ProtocolBVerifierError("non-synthetic inputs require the public verifier")
    if bootstrap_resamples is None or sign_flip_resamples is None: raise Phase4D2ProtocolBVerifierError("synthetic verifier requires explicit inference resamples")
    rebuilt,manifest=_rebuild(inputs,config,worker_count=worker_count,bootstrap_resamples=int(bootstrap_resamples),sign_flip_resamples=int(sign_flip_resamples)); _compare(Path(path),rebuilt)
    return Phase4D2ProtocolBVerifierSummary(Path(path),str(manifest["run_id"]),str(manifest["status"]),int(manifest["prediction_row_count"]),int(manifest["class_observation_count"]))


def verify_phase4_d2_protocol_b(path: Path, *, worker_count: int = 12) -> Phase4D2ProtocolBVerifierSummary:
    config_path=ROOT / "experiments/phase4/configs/d2_protocol_b_full_domain_v1.json"
    config=_parse_config(config_path,config_path.read_bytes(),frozen=True); _validate_inventory(Path(path),config)
    inputs = _reconstruct_real_inputs(config)
    bridge = _validate_real_parents_and_bridge(config, inputs)
    science = _rematerialize_real_science(inputs, config, worker_count=worker_count, eligibility=bridge.get("eligibility"))
    fitted = _fit_real_cells(inputs, science, config)
    alpha = _validate_real_alpha0_equivalence(fitted, bridge["protocol_a"], config)
    bootstrap=int(config.document["inference"]["bootstrap_resamples"]); sign_flip=int(config.document["inference"]["sign_flip_resamples"])
    projection = _aggregate(fitted["predictions"], bridge["metrics"], config, bootstrap, sign_flip)
    rebuilt,manifest=_serialize_rebuild(config,bridge,fitted,alpha,projection,science=science,worker_count=16,bootstrap_resamples=bootstrap,sign_flip_resamples=sign_flip)
    _compare(Path(path),rebuilt)
    return Phase4D2ProtocolBVerifierSummary(Path(path),str(manifest["run_id"]),str(manifest["status"]),int(manifest["prediction_row_count"]),int(manifest["class_observation_count"]))


__all__=["Phase4D2ProtocolBVerifierError","Phase4D2ProtocolBVerifierSummary","verify_phase4_d2_protocol_b","verify_phase4_d2_protocol_b_from_inputs"]
