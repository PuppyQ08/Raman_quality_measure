"""Frozen Phase-6 baseline evidence runner.

This module deliberately keeps the Phase-6 aggregation boundary small: baseline
transforms happen once on native spectra and all consumers read that same matrix.
The independent verifier is intentionally implemented in its own module.
"""
from __future__ import annotations

import csv
import warnings
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression

from rpe.downstream.rruff import D5RawCohort, load_d5_native_spectra, load_d5_raw_cohort
from rpe.downstream.rruff_matching import match_d5_protocol_a_values
from rpe.downstream.sugar_quantitative import TARGET_NAMES, load_d4_sugar_cohort
from rpe.evaluation import ReplicatePairInput, SingleSpectrumInput, Spectrum1D
from rpe.methods.catalog import Phase3System, TaskLine, load_classical_catalog
from rpe.methods.classical.baseline import BaselineRunStatus, run_baseline_system
from rpe.metrics.consistency import HalfSplitPearsonConsistencyMetric
from rpe.metrics.reference_free import ISLikeStructureToNoiseMetric
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/baseline_evidence_v1.json"
RUN_DOMAIN = b"rpe-phase6-baseline-evidence-v1\0"
_WORKER_CONTEXT: tuple[object, ...] | None = None


class BaselineEvidenceError(ValueError):
    pass


_SCIENTIFIC_EXCEPTIONS = (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError, Warning)


def _scientific_finite_or_close(value: object, source: str) -> str | None:
    """Return the frozen typed closure for non-finite scientific output.

    Programming/invariant exceptions intentionally propagate: they are artifact
    failures, not numerical scientific closures.
    """
    if isinstance(value, BaseException):
        if isinstance(value, _SCIENTIFIC_EXCEPTIONS):
            return "not_evaluable_metric_domain" if source == "metric" else "not_evaluable_consumer_failure"
        raise value
    array = np.asarray(value)
    if not np.isfinite(array).all():
        return "not_evaluable_metric_domain" if source == "metric" else "not_evaluable_consumer_failure"
    return None


def _canonical(value: object) -> bytes:
    return (json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


@dataclass(frozen=True)
class BaselineEvidenceConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    authorities: Mapping[str, Mapping[str, object]]
    artifact_contract: Mapping[str, object]
    endpoint_manifest: tuple[Mapping[str, object], ...]
    expected: Mapping[str, int]
    family_counts: Mapping[str, int]
    system_ids: tuple[str, ...]
    direct_gt_state: str
    document: Mapping[str, object]


@dataclass(frozen=True)
class BaselineEvidenceInputs:
    d5_axis_cm1: np.ndarray
    d5_matrix: np.ndarray
    d5_record_ids: tuple[str, ...]
    d5_classes: tuple[str, ...]
    d5_splits: tuple[tuple[np.ndarray, np.ndarray], ...]
    d4_axis_cm1: np.ndarray
    d4_matrix: np.ndarray
    d4_targets: np.ndarray
    d4_record_ids: tuple[str, ...]
    d4_well_ids: tuple[str, ...]
    d4_rounds: np.ndarray
    d4_repetitions: np.ndarray
    d4_fold_by_well: Mapping[str, int]
    d5_cohort: D5RawCohort | None = None
    d5_native_spectra: tuple[Spectrum1D, ...] | None = None

    @property
    def d5_record_count(self) -> int: return len(self.d5_record_ids)
    @property
    def d4_record_count(self) -> int: return len(self.d4_record_ids)


@dataclass(frozen=True)
class BaselineEvidenceSummary:
    path: Path
    run_id: str
    status: str
    system_count: int
    transform_receipt_row_count: int
    method_evidence_row_count: int
    bootstrap_result_row_count: int


def load_phase6_baseline_evidence_config(path: Path = DEFAULT_CONFIG) -> BaselineEvidenceConfig:
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if raw != _canonical(document):
        raise BaselineEvidenceError("config must be canonical JSON")
    if document.get("schema_version") != "phase6-baseline-evidence-v1":
        raise BaselineEvidenceError("unexpected config schema")
    inventory = document["inventory"]
    return BaselineEvidenceConfig(Path(path), len(raw), _sha(raw), str(document["schema_version"]),
        _freeze(document["authorities"]), _freeze(document["artifact_contract"]),
        tuple(_freeze(row) for row in document["endpoint_manifest"]), _freeze(document["expected"]),
        _freeze(inventory["family_counts"]), tuple(inventory["system_ids"]),
        str(document["direct_gt_state"]), _freeze(document))


def _validate_formal_system_inventory(config: BaselineEvidenceConfig, system_ids: Sequence[str], family_counts: Mapping[str, int]) -> None:
    if tuple(system_ids) != config.system_ids:
        raise BaselineEvidenceError("formal eligible system order mismatch")
    if len(system_ids) != 112:
        raise BaselineEvidenceError("formal system inventory must contain exactly 112 IDs")
    if dict(family_counts) != dict(config.family_counts):
        raise BaselineEvidenceError("formal family population mismatch")


def _d5_protocol_roles() -> dict[str, dict[str, str]]:
    return {"D5-A": {"query": "candidate", "library": "identity"}, "D5-B-matched-reference": {"query": "candidate", "library": "candidate"}}


def _d4_model_lifecycle() -> dict[str, dict[str, object]]:
    return {"D4-A": {"representation": "identity_train_validation", "candidate_roles": ("test",), "refit_per_system": False}, "D4-B": {"fit_roles": ("train",), "selection_roles": ("validation",), "refit_per_system": True}}


def _project_d5_corrected(axis: np.ndarray, corrected: np.ndarray, support: np.ndarray) -> np.ndarray:
    native_axis = np.asarray(axis, dtype="<f8")
    grid = np.asarray(support, dtype="<f8")
    if native_axis.ndim != 1 or native_axis.size < 2 or native_axis[0] > grid[0] or native_axis[-1] < grid[-1]:
        raise BaselineEvidenceError("D5 projection requires native support without extrapolation")
    left = int(np.searchsorted(native_axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(native_axis, grid[-1], side="left"))
    covered = native_axis[left:right + 1]
    if left < 0 or right >= native_axis.size or covered.size < 2 or float(np.max(np.diff(covered))) > 3.0:
        raise BaselineEvidenceError("D5 projection native gap exceeds 3 cm^-1")
    return np.asarray(np.interp(grid, native_axis, np.asarray(corrected, dtype="<f8")), dtype="<f4")


def _class_equal_top1_error(classes: Sequence[str], correct: Sequence[bool]) -> float:
    grouped: dict[str, list[bool]] = {}
    for label, value in zip(classes, correct, strict=True): grouped.setdefault(str(label), []).append(bool(value))
    return float(np.mean([1.0 - np.mean(values) for _, values in sorted(grouped.items())]))


def _normalized_squared_loss(predictions: np.ndarray, targets: np.ndarray) -> float:
    value = float(np.mean(((np.asarray(predictions) - np.asarray(targets)) ** 2) / (0.32 ** 2)))
    if _scientific_finite_or_close(value, "metric") is not None:
        raise BaselineEvidenceError("non-finite normalized loss")
    return value


def _candidate_minus_identity(candidate: float, identity: float) -> float: return float(candidate - identity)


def _half_split_record_ids(records: Sequence[tuple[int, int, str]]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ordered = sorted((int(round_id), int(repetition), str(record_id)) for round_id, repetition, record_id in records)
    return tuple(row[2] for row in ordered[0::2]), tuple(row[2] for row in ordered[1::2])


def _closed_endpoint_ids(reason: str) -> tuple[str, ...]:
    values = {
        "d5_a_matcher_failure": ("D5-A",), "d5_b_matcher_failure": ("D5-B-matched-reference",),
        "d5_reference_free_failure": ("D5-reference-free",), "d5_transform_failure": ("D5-A", "D5-B-matched-reference", "D5-reference-free"),
        "d4_a_consumer_failure": ("D4-A",), "d4_b_fit_or_consumer_failure": ("D4-B",),
        "d4_reference_free_failure": ("D4-reference-free",), "d4_half_split_failure": ("D4-half-split",),
        "d4_transform_failure": ("D4-A", "D4-B", "D4-reference-free", "D4-half-split"),
    }
    return values[reason]


def _fixed_method_rows(*, system_id: str, component_states: Mapping[str, tuple[float | None, str]], phase5_power_state: str, direct_gt_state: str) -> list[dict[str, object]]:
    rows = []
    for endpoint in ("availability", "direct_gt", "D5-A", "D5-B-matched-reference", "D5-reference-free", "D4-A", "D4-B", "D4-reference-free", "D4-half-split"):
        value, state = component_states.get(endpoint, (None, "not_evaluable_no_result"))
        if endpoint == "availability": value, state = 1.0, "complete"
        if endpoint == "direct_gt": value, state = None, direct_gt_state
        rows.append({"system_id": system_id, "endpoint_id": endpoint, "value": value, "state": state, "phase5_power_state": phase5_power_state})
    return rows


def _bootstrap_cluster_mean(values: np.ndarray, *, resamples: int, seed: int) -> dict[str, object]:
    values = np.asarray(values, dtype="<f8")
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, values.size, size=(resamples, values.size))
    means = values[draws].mean(axis=1)
    return {"estimate": float(values.mean()), "lower": float(np.percentile(means, 2.5)), "upper": float(np.percentile(means, 97.5)), "cluster_count": int(values.size)}


def make_synthetic_baseline_evidence_inputs(*, config: BaselineEvidenceConfig) -> BaselineEvidenceInputs:
    d5_axis = np.array([203., 204., 205., 206., 208., 1800.], dtype="<f8")
    d5_ids, d5_classes, d5_rows = [], [], []
    for cls in range(4):
        for replica in range(2):
            d5_ids.append(f"d5-{cls}-{replica}"); d5_classes.append(f"class-{cls}")
            d5_rows.append(np.sin(d5_axis / 100 + cls) + replica * .01 + .03 * d5_axis / 1800)
    indexes = np.arange(8, dtype=np.int64); splits = tuple((indexes[::2], indexes[1::2]) for _ in range(5))
    axis = np.linspace(100., 2100., 64, dtype="<f8"); x = np.linspace(0., 1., 64)
    rows=[]; targets=[]; ids=[]; wells=[]; rounds=[]; reps=[]; folds={}
    for well in range(10):
        wid=f"W{well:02d}"; folds[wid]=well % 5; target=np.array([(well+i)%5/15 for i in range(4)], dtype="<f8")
        for rnd in range(1,9):
            for rep in range(1,5):
                rows.append(.1 + target.sum()*np.exp(-((x-.4)/.12)**2) + .01*np.sin(20*x+rnd+rep)); targets.append(target); ids.append(f"{wid}-r{rnd}-m{rep}"); wells.append(wid); rounds.append(rnd); reps.append(rep)
    return BaselineEvidenceInputs(d5_axis, np.asarray(d5_rows,dtype="<f8"), tuple(d5_ids), tuple(d5_classes), splits, axis, np.asarray(rows,dtype="<f8"), np.asarray(targets,dtype="<f8"), tuple(ids), tuple(wells), np.asarray(rounds), np.asarray(reps), MappingProxyType(folds))


def _systems(config: BaselineEvidenceConfig, root: Path) -> tuple[Phase3System, ...]:
    catalog = load_classical_catalog(root / str(config.authorities["classical_catalog"]["path"]))
    found={s.system_id:s for s in catalog.systems if s.task_line is TaskLine.BASELINE_CORRECTION}
    try: return tuple(found[i] for i in config.system_ids)
    except KeyError as error: raise BaselineEvidenceError(f"catalog missing {error.args[0]}") from error


def _sha_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, dtype: str = "<f8") -> str:
    return _sha(np.ascontiguousarray(value, dtype=dtype).tobytes())


def _ordered_sha(values: Sequence[str]) -> str:
    return _sha(("\n".join(str(value) for value in values) + "\n").encode())


def _code_authority(root: Path) -> dict[str, dict[str, object]]:
    relative_paths = (
        "experiments/phase6/configs/baseline_evidence_v1.json",
        "rpe/methods/classical/baseline.py",
        "rpe/downstream/rruff.py",
        "rpe/downstream/rruff_matching.py",
        "rpe/downstream/sugar_quantitative.py",
        "rpe/metrics/reference_free.py",
        "rpe/metrics/consistency.py",
        "rpe/runner/phase6_baseline_evidence.py",
        "rpe/runner/phase6_baseline_evidence_verifier.py",
        "tools/run_phase6_baseline_evidence.py",
    )
    return {
        relative: {"byte_count": (root / relative).stat().st_size, "sha256": _sha_file(root / relative)}
        for relative in relative_paths if (root / relative).is_file()
    }


def _run_id(config: BaselineEvidenceConfig, systems: Sequence[Phase3System], inputs: BaselineEvidenceInputs, root: Path) -> str:
    identity={"config_sha256":config.sha256,"systems":[s.system_id for s in systems],"d5_record_ids_sha256":_ordered_sha(inputs.d5_record_ids),"d4_record_ids_sha256":_ordered_sha(inputs.d4_record_ids),"code_authority":_code_authority(root)}
    return str(config.artifact_contract["run_prefix"])+_sha(RUN_DOMAIN+_canonical(identity))


def _validate_authorities(config: BaselineEvidenceConfig, root: Path) -> None:
    for name, identity in config.authorities.items():
        path=root / str(identity["path"])
        if not path.is_file():
            raise BaselineEvidenceError(f"authority {name} is missing: {path}")
        if path.stat().st_size != int(identity["byte_count"]):
            raise BaselineEvidenceError(f"authority {name} byte count mismatch")
        if _sha_file(path) != str(identity["sha256"]):
            raise BaselineEvidenceError(f"authority {name} SHA256 mismatch")
    ledger_path = root / str(config.authorities["baseline_promotion_sha256sums"]["path"])
    ledger_root = ledger_path.parent
    entries = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = ledger_root / relative
        if not target.is_file() or _sha_file(target) != digest:
            raise BaselineEvidenceError(f"Phase-3 checksum tree mismatch: {relative}")
        entries.append(relative)
    if not entries:
        raise BaselineEvidenceError("Phase-3 checksum tree is empty")
    promotion = json.loads((root / str(config.authorities["baseline_promotion"]["path"])).read_text(encoding="utf-8"))
    if tuple(promotion.get("phase5_eligible_system_ids", ())) != config.system_ids:
        raise BaselineEvidenceError("Phase-3 eligible system order mismatch")
    _validate_formal_system_inventory(config, config.system_ids, dict(config.family_counts))
    d5_ledger = root / str(config.authorities["d5_raw_sha256sums"]["path"])
    if not d5_ledger.is_file() or not d5_ledger.read_text(encoding="utf-8").strip():
        raise BaselineEvidenceError("D5 raw checksum ledger is missing or empty")
    for line in d5_ledger.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        target = d5_ledger.parent / relative
        if not target.is_file() or _sha_file(target) != digest:
            raise BaselineEvidenceError(f"D5 raw checksum tree mismatch: {relative}")


def _load_real_inputs(config: BaselineEvidenceConfig, root: Path) -> BaselineEvidenceInputs:
    d5_root = root / "data/unified/rruff_raman_raw"
    d5 = load_d5_raw_cohort(root / str(config.authorities["d5_protocol"]["path"]), d5_root)
    d5_native = load_d5_native_spectra(d5_root, d5.record_ids)
    d4 = load_d4_sugar_cohort(root / str(config.authorities["d4_protocol"]["path"]), root / str(config.authorities["d4_source_archive"]["path"]))
    labels=tuple(str(v) for v in d5.class_labels.tolist()); folds={wid: next(i for i, f in enumerate(d4.folds) if np.any(np.asarray(d4.well_ids)[f] == wid)) for wid in sorted(set(d4.well_ids))}
    return BaselineEvidenceInputs(np.asarray(d5.wavenumber,dtype="<f8"),np.asarray(d5.intensity,dtype="<f8"),d5.record_ids,labels,tuple((s.query_indices,s.library_indices) for s in d5.splits),np.asarray(d4.wavenumber,dtype="<f8"),np.asarray(d4.intensity,dtype="<f8"),np.asarray(d4.targets,dtype="<f8"),d4.record_ids,d4.well_ids,np.asarray(d4.rounds),np.asarray(d4.repetitions),MappingProxyType(folds),d5,d5_native)


def _transform(system: Phase3System, axis: np.ndarray, matrix: np.ndarray, ids: Sequence[str], cohort: str):
    out=np.empty_like(matrix,dtype="<f8"); receipts=[]; failure=None
    for i,(row,rid) in enumerate(zip(matrix,ids,strict=True)):
        if failure is not None:
            receipts.append({"system_id":system.system_id,"cohort_id":cohort,"record_id":rid,"status":"not_run_cohort_transform_closed","output_sha256":None})
            continue
        result=run_baseline_system(system,Spectrum1D(f"{cohort}::{rid}",rid,axis,row))
        receipts.append({"system_id":system.system_id,"cohort_id":cohort,"record_id":rid,"status":result.status.value,"output_sha256":result.corrected_sha256})
        if result.status not in {BaselineRunStatus.COMPLETE,BaselineRunStatus.COMPLETE_WITH_WARNING} or result.corrected_intensity is None:
            failure=result.status.value
            continue
        out[i]=result.corrected_intensity
    return out,receipts,failure


def _transform_native_d5(system: Phase3System, spectra: Sequence[Spectrum1D], support: np.ndarray):
    projected=[]; corrected_native=[]; receipts=[]; failure=None
    for spectrum in spectra:
        result=run_baseline_system(system,spectrum)
        receipts.append({"system_id":system.system_id,"cohort_id":"D5","record_id":spectrum.spectrum_id.rsplit("::",1)[-1],"status":result.status.value,"output_sha256":result.corrected_sha256})
        if result.status not in {BaselineRunStatus.COMPLETE,BaselineRunStatus.COMPLETE_WITH_WARNING} or result.corrected_intensity is None:
            failure=result.status.value; continue
        try:
            corrected_native.append(np.asarray(result.corrected_intensity,dtype="<f8"))
            projected.append(_project_d5_corrected(spectrum.axis_cm1,result.corrected_intensity,support))
        except BaselineEvidenceError: failure="invalid_native_projection"
    if failure is not None or len(projected) != len(spectra): return None,None,receipts,failure or "incomplete"
    return np.vstack(projected),tuple(corrected_native),receipts,None


def _metric_value(metric: ISLikeStructureToNoiseMetric, spectrum: Spectrum1D) -> float:
    return float(metric.evaluate(SingleSpectrumInput(spectrum)).outputs[0].value)


def _native_metric_effects(axis: np.ndarray, identity: np.ndarray, candidate: np.ndarray, ids: Sequence[str]) -> np.ndarray:
    metric = ISLikeStructureToNoiseMetric()
    return np.asarray([_metric_value(metric, Spectrum1D(f"metric::{rid}", str(rid), axis, candidate[index])) - _metric_value(metric, Spectrum1D(f"identity::{rid}", str(rid), axis, identity[index])) for index, rid in enumerate(ids)], dtype="<f8")


def _fit_pls(x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, fold: int):
    choices=[]; validation=[]
    for n in (2,4,8,16,32):
        try:
            model=PLSRegression(n_components=n,scale=True,max_iter=500,tol=1e-6,copy=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with threadpool_limits(limits=1,user_api="blas"): model.fit(x_train,y_train); predicted=np.asarray(model.predict(x_validation),dtype="<f8")
            if _scientific_finite_or_close(predicted, "consumer") is not None:
                raise ValueError("non-finite PLS validation prediction")
            score=float(np.mean(np.sqrt(np.mean((predicted-y_validation)**2,axis=0))/0.32))
            validation.append({"fold":fold,"n_components":n,"macro_normalized_rmse":score,"state":"complete"}); choices.append((score,n,model))
        except _SCIENTIFIC_EXCEPTIONS as error:
            validation.append({"fold":fold,"n_components":n,"macro_normalized_rmse":None,"state":"failed_model_lifecycle","reason":type(error).__name__})
    if len(choices) != 5: raise BaselineEvidenceError(f"D4 fold {fold} incomplete PLS component grid")
    score,n,model=min(choices,key=lambda item:(item[0],item[1]))
    if _scientific_finite_or_close(score, "metric") is not None:
        raise BaselineEvidenceError(f"D4 fold {fold} non-finite validation score")
    digest=_sha(_canonical({"fold":fold,"n_components":n,"validation":validation,"coef_sha256":_sha(np.asarray(model.coef_,dtype="<f8").tobytes())}))
    return model, validation, n, digest


def _d4_roles(inputs: BaselineEvidenceInputs, fold: int):
    test=np.asarray([i for i,w in enumerate(inputs.d4_well_ids) if int(inputs.d4_fold_by_well[w])==fold],dtype=np.int64)
    validation=np.asarray([i for i,w in enumerate(inputs.d4_well_ids) if int(inputs.d4_fold_by_well[w])==(fold+1)%5],dtype=np.int64)
    train=np.asarray([i for i,w in enumerate(inputs.d4_well_ids) if i not in set(test.tolist()) and i not in set(validation.tolist())],dtype=np.int64)
    return train,validation,test


def _csv(rows: Sequence[Mapping[str,object]], names: Sequence[str]) -> bytes:
    class W:
        def __init__(self): self.parts=[]
        def write(self,x): self.parts.append(x); return len(x)
    w=W(); writer=csv.DictWriter(w,fieldnames=names,lineterminator="\n"); writer.writeheader()
    for row in rows: writer.writerow({k:("" if row.get(k) is None else (format(row[k],".17g") if isinstance(row.get(k),float) else row.get(k))) for k in names})
    return "".join(w.parts).encode()


def _fixed_cluster_rows(rows, systems, clusters, protocols, *, kind: str, closure_reasons: Mapping[tuple[str, str], str] | None = None):
    closure_reasons = {} if closure_reasons is None else closure_reasons
    existing={(str(row["system_id"]),str(row.get("protocol_id",row.get("cohort_id","method_native"))),str(row.get("cluster_id",row.get("well_id","")))) for row in rows}
    fixed=list(rows)
    for system in systems:
        for protocol in protocols:
            for cluster in clusters:
                key=(system.system_id,protocol,str(cluster))
                if key not in existing:
                    fixed.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"protocol_id":protocol,"cohort_id":protocol if kind == "reference" else None,"cluster_id":str(cluster),"well_id":str(cluster) if kind == "half" else None,"effect":None,"state":closure_reasons.get((system.system_id, protocol), "not_evaluable_component_closed")})
    return sorted(fixed, key=lambda row: (str(row["system_id"]), str(row.get("protocol_id", row.get("cohort_id", "method_native"))), str(row.get("cluster_id", row.get("well_id", "")))))


def _bootstrap_rows_from_clusters(systems, downstream, reference, half, *, resamples: int, seed: int):
    endpoint_sources={
        "D5-A":(downstream,"D5-A"), "D5-B-matched-reference":(downstream,"D5-B-matched-reference"),
        "D5-reference-free":(reference,"D5"), "D4-A":(downstream,"D4-A"), "D4-B":(downstream,"D4-B"),
        "D4-reference-free":(reference,"D4"), "D4-half-split":(half,"method_native"),
    }
    cluster_counts = {"D5": len({str(row["cluster_id"]) for row in reference if row.get("cohort_id") == "D5"}),
                      "D4": len({str(row["cluster_id"]) for row in reference if row.get("cohort_id") == "D4"})}
    rng = np.random.Generator(np.random.PCG64(seed))
    shared = {key: rng.integers(0, count, size=(resamples, count)) for key, count in cluster_counts.items()}
    output=[]
    for system in systems:
        for endpoint,(rows,key) in endpoint_sources.items():
            if endpoint == "D4-half-split": selected=[row for row in rows if row["system_id"]==system.system_id]
            elif endpoint.endswith("reference-free"):
                cohort=key; selected=[row for row in rows if row["system_id"]==system.system_id and row.get("cohort_id")==cohort]
            else: selected=[row for row in rows if row["system_id"]==system.system_id and row.get("protocol_id")==key]
            values=[row.get("effect") for row in selected]
            cohort = "D5" if endpoint.startswith("D5") else "D4"
            if len(values) != cluster_counts[cohort] or any(value is None for value in values):
                states={str(row.get("state")) for row in selected}
                if len(values) != cluster_counts[cohort] or any(value is not None for value in values) or len(states) != 1 or "complete" in states:
                    raise BaselineEvidenceError(f"non-canonical closed cluster grid: {system.system_id}/{endpoint}")
                output.append({"system_id":system.system_id,"endpoint_id":endpoint,"estimate":None,"interval_lower":None,"interval_upper":None,"cluster_count":len(values),"state":next(iter(states)),"bootstrap_seed":seed,"resamples":resamples})
            else:
                array=np.asarray(values,dtype="<f8"); means=array[shared[cohort]].mean(axis=1)
                output.append({"system_id":system.system_id,"endpoint_id":endpoint,"estimate":float(array.mean()),"interval_lower":float(np.percentile(means,2.5)),"interval_upper":float(np.percentile(means,97.5)),"cluster_count":len(array),"state":"complete","bootstrap_seed":seed,"resamples":resamples})
    return output


def _match_d5(inputs: BaselineEvidenceInputs, query_matrix: np.ndarray, library_matrix: np.ndarray, protocol: str):
    occurrences: list[tuple[str, bool]] = []; receipts=[]
    for split_number,(query,library) in enumerate(inputs.d5_splits):
        if inputs.d5_cohort is not None:
            result=match_d5_protocol_a_values(inputs.d5_cohort,inputs.d5_cohort.splits[split_number],condition_id=protocol,query_record_ids=tuple(inputs.d5_record_ids[int(i)] for i in query),library_record_ids=tuple(inputs.d5_record_ids[int(i)] for i in library),query_values=query_matrix[query],library_values=library_matrix[library])
            current=[(str(label),bool(ok)) for label,ok in zip(result.true_class_labels,result.top1_correct,strict=True)]
            receipts.append({"split":split_number,"split_sha256":inputs.d5_cohort.splits[split_number].split_sha256,"query_count":len(current),"top1_sha256":_array_sha(result.top1_correct,"|b1"),"top5_sha256":_array_sha(result.top5_correct,"|b1"),"ranking_sha256":_sha(result.ranked_class_labels.tobytes()+result.ranked_class_scores.tobytes())})
        else:
            q=np.asarray(query_matrix[query],dtype="<f8"); l=np.asarray(library_matrix[library],dtype="<f8")
            q=q/np.linalg.norm(q,axis=1)[:,None]; l=l/np.linalg.norm(l,axis=1)[:,None]
            scores=q@l.T; predicted=[inputs.d5_classes[int(library[int(np.argmax(row))])] for row in scores]
            truth=[inputs.d5_classes[int(i)] for i in query]; current=list(zip(truth,(a==b for a,b in zip(truth,predicted,strict=True)),strict=True))
            receipts.append({"split":split_number,"query_count":len(current),"top1_sha256":_sha(_canonical(current)),"top5_sha256":None,"ranking_sha256":_array_sha(scores)})
        occurrences.extend(current)
    by_class={label:[ok for current,ok in occurrences if current==label] for label in dict.fromkeys(label for label,_ in occurrences)}
    errors={label:float(1-np.mean(values)) for label,values in by_class.items()}
    return errors,{"protocol_id":protocol,"split_receipts":receipts,"occurrence_count":len(occurrences),"prediction_sha256":_sha(_canonical(occurrences)),"class_error_sha256":_sha(_canonical(errors))}


def _identity_context(inputs: BaselineEvidenceInputs, support: np.ndarray):
    raw_projected=(np.vstack([_project_d5_corrected(s.axis_cm1,s.intensity,support) for s in inputs.d5_native_spectra]) if inputs.d5_native_spectra is not None else np.vstack([_project_d5_corrected(inputs.d5_axis_cm1,row,support) for row in inputs.d5_matrix]))
    identity_errors,identity_matcher=_match_d5(inputs,raw_projected,raw_projected,"identity-D5-A")
    _, identity_matcher_b = _match_d5(inputs,raw_projected,raw_projected,"identity-D5-B")
    if identity_matcher["prediction_sha256"] != identity_matcher_b["prediction_sha256"] or identity_matcher["class_error_sha256"] != identity_matcher_b["class_error_sha256"]:
        raise BaselineEvidenceError("D5 A/B raw identity matcher mismatch")
    identity_matcher = {"D5-A": identity_matcher, "D5-B-matched-reference": identity_matcher_b, "exact_equal": True}
    def raw_identity_consumer():
        models={}; predictions={}; receipts=[]
        for fold in range(5):
            train,validation,test=_d4_roles(inputs,fold); model,rows,n,digest=_fit_pls(inputs.d4_matrix[train,1:].astype("<f4"),inputs.d4_targets[train],inputs.d4_matrix[validation,1:].astype("<f4"),inputs.d4_targets[validation],fold)
            with threadpool_limits(limits=1,user_api="blas"): predicted=np.asarray(model.predict(inputs.d4_matrix[test,1:].astype("<f4")),dtype="<f8")
            models[fold]=model; receipts.append({"fold":fold,"selected_n_components":n,"model_digest":digest,"validation_scores_sha256":_sha(_canonical(rows))})
            for index,value in zip(test,predicted,strict=True): predictions[int(index)]=value
        return models,predictions,receipts
    models,predictions,model_receipts=raw_identity_consumer()
    _, predictions_b, model_receipts_b=raw_identity_consumer()
    if model_receipts != model_receipts_b or _sha(_canonical(predictions)) != _sha(_canonical(predictions_b)):
        raise BaselineEvidenceError("D4 A/B raw identity model or prediction mismatch")
    model_receipts={"D4-A":model_receipts,"D4-B":model_receipts_b,"prediction_sha256":_sha(_canonical(predictions)),"exact_equal":True}
    identity_loss={well:_normalized_squared_loss(np.asarray([predictions[i] for i,w in enumerate(inputs.d4_well_ids) if w==well]),inputs.d4_targets[[i for i,w in enumerate(inputs.d4_well_ids) if w==well]]) for well in dict.fromkeys(inputs.d4_well_ids)}
    d5_metric=ISLikeStructureToNoiseMetric()
    if inputs.d5_native_spectra is not None: d5_identity_reference=np.asarray([_metric_value(d5_metric,s) for s in inputs.d5_native_spectra],dtype="<f8")
    else: d5_identity_reference=np.asarray([_metric_value(d5_metric,Spectrum1D(f"identity::{rid}",rid,inputs.d5_axis_cm1,row)) for rid,row in zip(inputs.d5_record_ids,inputs.d5_matrix,strict=True)],dtype="<f8")
    d4_identity_reference=np.asarray([_metric_value(d5_metric,Spectrum1D(f"identity::{rid}",rid,inputs.d4_axis_cm1,row)) for rid,row in zip(inputs.d4_record_ids,inputs.d4_matrix,strict=True)],dtype="<f8")
    half_metric=HalfSplitPearsonConsistencyMetric(); identity_half={}
    for well in dict.fromkeys(inputs.d4_well_ids):
        ix=[i for i,w in enumerate(inputs.d4_well_ids) if w==well]; ordered=sorted(ix,key=lambda i:(int(inputs.d4_rounds[i]),int(inputs.d4_repetitions[i]),inputs.d4_record_ids[i]))
        identity_half[well]=float(half_metric.evaluate(ReplicatePairInput(Spectrum1D(f"{well}-identity-a",well,inputs.d4_axis_cm1,np.mean(inputs.d4_matrix[ordered[::2]],axis=0)),Spectrum1D(f"{well}-identity-b",well,inputs.d4_axis_cm1,np.mean(inputs.d4_matrix[ordered[1::2]],axis=0)))).outputs[0].value)
    return raw_projected,identity_errors,identity_matcher,models,predictions,identity_loss,model_receipts,d5_identity_reference,d4_identity_reference,identity_half


def _initialize_worker(*context):
    global _WORKER_CONTEXT; _WORKER_CONTEXT=context


def _evaluate_system_task(index: int):
    if _WORKER_CONTEXT is None: raise BaselineEvidenceError("worker context is not initialized")
    inputs,systems,support,raw_projected,identity_errors,identity_models,identity_predictions,identity_loss,d5_identity_reference,d4_identity_reference,identity_half=_WORKER_CONTEXT
    return _evaluate_system(inputs,systems[index],support,raw_projected,identity_errors,identity_models,identity_predictions,identity_loss,d5_identity_reference,d4_identity_reference,identity_half)


def _evaluate_system(inputs,system,support,raw_projected,identity_errors,identity_models,identity_predictions,identity_loss,d5_identity_reference,d4_identity_reference,identity_half):
    receipts=[]; downstream=[]; reference=[]; half=[]; states={}; matcher_receipts={}; model_receipts=[]; prediction_receipts={}
    d5_consumer_digest=None; d4_native_digest=None
    classes=tuple(dict.fromkeys(inputs.d5_classes)); wells=tuple(dict.fromkeys(inputs.d4_well_ids))
    if inputs.d5_native_spectra is not None: d5,native,rec5,fail5=_transform_native_d5(system,inputs.d5_native_spectra,support)
    else: d5,rec5,fail5=_transform(system,inputs.d5_axis_cm1,inputs.d5_matrix,inputs.d5_record_ids,"D5"); native=None
    receipts.extend(rec5)
    if fail5: states.update({e:(None,"not_evaluable_transform_failure") for e in _closed_endpoint_ids("d5_transform_failure")})
    else:
        projected=d5 if native is not None else np.vstack([_project_d5_corrected(inputs.d5_axis_cm1,row,support) for row in d5])
        d5_consumer_digest=_array_sha(projected,"<f4")
        for protocol,library in (("D5-A",raw_projected),("D5-B-matched-reference",projected)):
            try:
                errors,receipt=_match_d5(inputs,projected,library,protocol); matcher_receipts[protocol]=receipt; values=[]
                for cls in classes:
                    effect=float(errors[str(cls)]-identity_errors[str(cls)]); values.append(effect); downstream.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"protocol_id":protocol,"cluster_id":str(cls),"identity_value":identity_errors[str(cls)],"candidate_value":errors[str(cls)],"effect":effect,"state":"complete"})
                states[protocol]=(float(np.mean(values)),"complete")
            except _SCIENTIFIC_EXCEPTIONS as error:
                downstream[:]=[row for row in downstream if not (row.get("system_id")==system.system_id and row.get("protocol_id")==protocol)]
                states[protocol]=(None,"not_evaluable_consumer_failure"); matcher_receipts[protocol]={"state":"not_evaluable_consumer_failure","reason":type(error).__name__}
        try:
            if native is not None:
                metric=ISLikeStructureToNoiseMetric(); effects=np.asarray([_metric_value(metric,Spectrum1D(s.spectrum_id,s.sample_id,s.axis_cm1,c))-d5_identity_reference[i] for i,(s,c) in enumerate(zip(inputs.d5_native_spectra,native,strict=True))])
            else:
                metric=ISLikeStructureToNoiseMetric(); effects=np.asarray([_metric_value(metric,Spectrum1D(f"candidate::{rid}",rid,inputs.d5_axis_cm1,row))-d5_identity_reference[i] for i,(rid,row) in enumerate(zip(inputs.d5_record_ids,d5,strict=True))])
            vals=[]
            for cls in classes:
                ix=[i for i,c in enumerate(inputs.d5_classes) if c==cls]; value=float(np.mean(effects[ix])); vals.append(value); reference.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"cohort_id":"D5","cluster_id":str(cls),"effect":value,"state":"complete"})
            states["D5-reference-free"]=(float(np.mean(vals)),"complete")
        except _SCIENTIFIC_EXCEPTIONS:
            reference[:]=[row for row in reference if not (row.get("system_id")==system.system_id and row.get("cohort_id")=="D5")]
            states["D5-reference-free"]=(None,"not_evaluable_metric_domain")
    del d5,native
    d4,rec4,fail4=_transform(system,inputs.d4_axis_cm1,inputs.d4_matrix,inputs.d4_record_ids,"D4"); receipts.extend(rec4)
    if fail4: states.update({e:(None,"not_evaluable_transform_failure") for e in _closed_endpoint_ids("d4_transform_failure")})
    else:
        d4_native_digest=_array_sha(d4)
        pa={}; pb={}; a_failed=False; b_failed=False
        for fold in range(5):
            train,validation,test=_d4_roles(inputs,fold)
            try:
                with threadpool_limits(limits=1,user_api="blas"): a=np.asarray(identity_models[fold].predict(d4[test,1:].astype("<f4")))
                for i,x in zip(test,a,strict=True): pa[int(i)]=x
            except _SCIENTIFIC_EXCEPTIONS: a_failed=True
            try:
                model,rows,n,digest=_fit_pls(d4[train,1:].astype("<f4"),inputs.d4_targets[train],d4[validation,1:].astype("<f4"),inputs.d4_targets[validation],fold)
                with threadpool_limits(limits=1,user_api="blas"): b=np.asarray(model.predict(d4[test,1:].astype("<f4")))
                model_receipts.append({"fold":fold,"selected_n_components":n,"model_digest":digest,"validation_scores_sha256":_sha(_canonical(rows))})
                for i,y in zip(test,b,strict=True): pb[int(i)]=y
            except _SCIENTIFIC_EXCEPTIONS as error: b_failed=True; model_receipts.append({"fold":fold,"state":"not_evaluable_fit_failure","reason":type(error).__name__})
        for protocol,predictions in (("D4-A",pa),("D4-B",pb)):
            if protocol=="D4-A" and a_failed: states[protocol]=(None,"not_evaluable_consumer_failure"); continue
            if protocol=="D4-B" and b_failed: states[protocol]=(None,"not_evaluable_fit_failure"); continue
            vals=[]
            try:
                for well in wells:
                    ix=[i for i,w in enumerate(inputs.d4_well_ids) if w==well]; candidate=_normalized_squared_loss(np.asarray([predictions[i] for i in ix]),inputs.d4_targets[ix]); effect=candidate-identity_loss[well]; vals.append(effect); downstream.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"protocol_id":protocol,"cluster_id":well,"identity_value":identity_loss[well],"candidate_value":candidate,"effect":effect,"state":"complete"})
                states[protocol]=(float(np.mean(vals)),"complete"); prediction_receipts[protocol]=_sha(_canonical({str(i):predictions[i] for i in sorted(predictions)}))
            except _SCIENTIFIC_EXCEPTIONS:
                downstream[:]=[row for row in downstream if not (row.get("system_id")==system.system_id and row.get("protocol_id")==protocol)]
                states[protocol]=(None,"not_evaluable_consumer_failure")
        try:
            metric=ISLikeStructureToNoiseMetric(); effects=np.asarray([_metric_value(metric,Spectrum1D(f"candidate::{rid}",rid,inputs.d4_axis_cm1,row))-d4_identity_reference[i] for i,(rid,row) in enumerate(zip(inputs.d4_record_ids,d4,strict=True))]); vals=[]
            for well in wells:
                ix=[i for i,w in enumerate(inputs.d4_well_ids) if w==well]; value=float(np.mean(effects[ix])); vals.append(value); reference.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"cohort_id":"D4","cluster_id":well,"effect":value,"state":"complete"})
            states["D4-reference-free"]=(float(np.mean(vals)),"complete")
        except _SCIENTIFIC_EXCEPTIONS:
            reference[:]=[row for row in reference if not (row.get("system_id")==system.system_id and row.get("cohort_id")=="D4")]
            states["D4-reference-free"]=(None,"not_evaluable_metric_domain")
        try:
            metric=HalfSplitPearsonConsistencyMetric(); vals=[]
            for well in wells:
                ix=[i for i,w in enumerate(inputs.d4_well_ids) if w==well]; ordered=sorted(ix,key=lambda i:(int(inputs.d4_rounds[i]),int(inputs.d4_repetitions[i]),inputs.d4_record_ids[i]))
                if len(ordered)!=32: raise BaselineEvidenceError("D4 half split requires 32 records")
                def value(matrix,prefix): return float(metric.evaluate(ReplicatePairInput(Spectrum1D(prefix+"a",well,inputs.d4_axis_cm1,np.mean(matrix[ordered[::2]],axis=0)),Spectrum1D(prefix+"b",well,inputs.d4_axis_cm1,np.mean(matrix[ordered[1::2]],axis=0)))).outputs[0].value)
                identity=identity_half[well]; candidate=value(d4,"candidate"); effect=candidate-identity; vals.append(effect); half.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"well_id":well,"identity_value":identity,"candidate_value":candidate,"effect":effect,"state":"complete"})
            states["D4-half-split"]=(float(np.mean(vals)),"complete")
        except _SCIENTIFIC_EXCEPTIONS:
            half[:]=[row for row in half if row.get("system_id") != system.system_id]; states["D4-half-split"]=(None,"not_evaluable_half_split_failure")
    status={"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"component_states":states,"d5":{"consumer_input_sha256":{"D5-A":d5_consumer_digest,"D5-B-matched-reference":d5_consumer_digest},"matcher_receipts":matcher_receipts},"d4":{"native_corrected_matrix_sha256":d4_native_digest,"consumer_input_sha256":{"D4-A":d4_native_digest,"D4-B":d4_native_digest},"model_receipts":model_receipts,"prediction_receipts":prediction_receipts}}
    return status,receipts,downstream,reference,half


def _build_phase6_baseline_evidence_from_inputs_unchecked(inputs: BaselineEvidenceInputs, systems: Sequence[Phase3System], output_root: Path, *, config: BaselineEvidenceConfig, worker_count: int, project_root: Path=ROOT) -> BaselineEvidenceSummary:
    statuses=[]; receipts=[]; downstream=[]; reference=[]; half=[]; methods=[]
    d5_support=np.arange(204.,1800.1,2.) if inputs.d5_axis_cm1.size > 10 else np.array([204.,206.,208.])
    d5_support=d5_support[(d5_support>=inputs.d5_axis_cm1.min())&(d5_support<=inputs.d5_axis_cm1.max())]
    wells=tuple(dict.fromkeys(inputs.d4_well_ids)); classes=tuple(dict.fromkeys(inputs.d5_classes)); power=str(config.document["gate"]["phase5_power_state"])
    formal_requested = tuple(system.system_id for system in systems) == config.system_ids
    if formal_requested:
        _validate_formal_system_inventory(config, tuple(system.system_id for system in systems), {family: sum(system.family_id == family for system in systems) for family in config.family_counts})
    raw_projected,identity_errors,identity_matcher,identity_models,identity_predictions,identity_loss,identity_model_receipts,d5_identity_reference,d4_identity_reference,identity_half=_identity_context(inputs,d5_support)
    context=(inputs,tuple(systems),d5_support,raw_projected,identity_errors,identity_models,identity_predictions,identity_loss,d5_identity_reference,d4_identity_reference,identity_half)
    indexes=tuple(range(len(systems))); workers=max(1,int(worker_count))
    if workers==1:
        _initialize_worker(*context); results=[_evaluate_system_task(i) for i in indexes]
    else:
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("fork"),initializer=_initialize_worker,initargs=context) as executor:
            results=list(executor.map(_evaluate_system_task,indexes,chunksize=1))
    for system,result in zip(systems,results,strict=True):
        status,rec,current_downstream,current_reference,current_half=result; statuses.append(status); receipts.extend(rec); downstream.extend(current_downstream); reference.extend(current_reference); half.extend(current_half); states=status["component_states"]
        for row in _fixed_method_rows(system_id=system.system_id,component_states=states,phase5_power_state=power,direct_gt_state=config.direct_gt_state): row.update(family_id=system.family_id,method_id=system.method_id); methods.append(row)
    closure_reasons = {}
    for status in statuses:
        for endpoint, (_, state) in status["component_states"].items():
            if state != "complete":
                protocol = {"D5-reference-free": "D5", "D4-reference-free": "D4", "D4-half-split": "method_native"}.get(endpoint, endpoint)
                closure_reasons[(str(status["system_id"]), protocol)] = str(state)
    downstream=(
        _fixed_cluster_rows([row for row in downstream if row.get("protocol_id") in {"D5-A","D5-B-matched-reference"}],systems,classes,("D5-A","D5-B-matched-reference"),kind="downstream",closure_reasons=closure_reasons)
        + _fixed_cluster_rows([row for row in downstream if row.get("protocol_id") in {"D4-A","D4-B"}],systems,wells,("D4-A","D4-B"),kind="downstream",closure_reasons=closure_reasons)
    )
    reference=(
        _fixed_cluster_rows([row for row in reference if row.get("cohort_id")=="D5"],systems,classes,("D5",),kind="reference",closure_reasons=closure_reasons)
        + _fixed_cluster_rows([row for row in reference if row.get("cohort_id")=="D4"],systems,wells,("D4",),kind="reference",closure_reasons=closure_reasons)
    )
    half=_fixed_cluster_rows(half,systems,wells,("method_native",),kind="half",closure_reasons=closure_reasons)
    bootstrap=_bootstrap_rows_from_clusters(systems,downstream,reference,half,resamples=int(config.document["bootstrap"]["resamples"]),seed=int(config.document["bootstrap"]["seed"]))
    bootstrap_by_key={(row["system_id"],row["endpoint_id"]):row for row in bootstrap}
    endpoint_by_id={str(row["endpoint_id"]):row for row in config.endpoint_manifest}
    full_methods=[]
    for row in methods:
        endpoint_id=str(row["endpoint_id"]); endpoint=endpoint_by_id[endpoint_id]; result=bootstrap_by_key.get((row["system_id"],endpoint_id))
        source_path=("system_status.jsonl" if endpoint_id in {"availability","direct_gt"} else "downstream_cluster_rows.jsonl" if endpoint_id in {"D5-A","D5-B-matched-reference","D4-A","D4-B"} else "reference_free_cluster_rows.jsonl" if endpoint_id.endswith("reference-free") else "half_split_well_rows.jsonl")
        estimate=(None if endpoint_id in {"availability","direct_gt"} else result["estimate"]); state=(row["state"] if result is None else result["state"])
        full_methods.append({"evidence_id":f"{row['system_id']}:{endpoint_id}","task_line":"baseline_correction","system_id":row["system_id"],"family_id":row["family_id"],"endpoint_id":endpoint_id,"protocol_id":endpoint["protocol_id"],"cohort_id":"D5_raw_rruff" if endpoint_id.startswith("D5") else "D4_low_snr_sugar" if endpoint_id.startswith("D4") else "baseline_method_inventory","evidence_component":endpoint["evidence_component"],"metric_output_id":endpoint["metric_output_id"],"estimate":estimate,"interval_lower":None if result is None else result["interval_lower"],"interval_upper":None if result is None else result["interval_upper"],"preferred_direction":endpoint["preferred_direction"],"state":state,"reason_code":"" if state=="complete" else state,"phase5_power_state":power,"source_path":source_path,"source_sha256":None})
    methods=full_methods
    family=[]
    for fid in config.family_counts:
        for endpoint in config.endpoint_manifest:
            endpoint_id=str(endpoint["endpoint_id"]); selected=[row for row in methods if row["family_id"]==fid and row["endpoint_id"]==endpoint_id and row["estimate"] is not None]; values=[float(row["estimate"]) for row in selected]
            family.append({"family_id":fid,"endpoint_id":endpoint_id,"registered_system_count":int(config.family_counts[fid]),"complete_system_count":len(values),"median_estimate":None if not values else float(np.median(values)),"min_estimate":None if not values else float(np.min(values)),"max_estimate":None if not values else float(np.max(values))})
    code_authority=_code_authority(project_root); identity={"config_sha256":config.sha256,"systems":[s.system_id for s in systems],"d5_record_ids_sha256":_ordered_sha(inputs.d5_record_ids),"d4_record_ids_sha256":_ordered_sha(inputs.d4_record_ids),"code_authority":code_authority}
    run_id=_run_id(config,systems,inputs,project_root)
    path=Path(output_root)/run_id
    if path.exists(): raise BaselineEvidenceError(f"run collision: {path} exists")
    path.mkdir(parents=True)
    jsonl=lambda rows:b"".join(_canonical(row) for row in rows)
    payloads={"config.json":_canonical(dict(config.document)),"authority_bridge.json":_canonical({"config_sha256":config.sha256,"run_identity":identity}),"preflight.json":_canonical({"phase5_power_state":power,"identity_d5_matcher":identity_matcher,"identity_d4_models":identity_model_receipts,"identity_d4_prediction_sha256":_sha(_canonical(identity_predictions)),"d5_record_count":inputs.d5_record_count,"d4_record_count":inputs.d4_record_count,"system_count":len(systems)}),"system_status.jsonl":jsonl(statuses),"transform_receipts.jsonl":jsonl(receipts),"downstream_cluster_rows.jsonl":jsonl(downstream),"reference_free_cluster_rows.jsonl":jsonl(reference),"half_split_well_rows.jsonl":jsonl(half),"bootstrap_results.jsonl":jsonl(bootstrap)}
    method_fields=("evidence_id","task_line","system_id","family_id","endpoint_id","protocol_id","cohort_id","evidence_component","metric_output_id","estimate","interval_lower","interval_upper","preferred_direction","state","reason_code","phase5_power_state","source_path","source_sha256")
    family_fields=("family_id","endpoint_id","registered_system_count","complete_system_count","median_estimate","min_estimate","max_estimate")
    source_sha={name:_sha(data) for name,data in payloads.items()}; methods=[dict(row,source_sha256=source_sha[row["source_path"]]) for row in methods]
    payloads["method_evidence_rows.csv"]=_csv(methods,method_fields); payloads["family_projection.csv"]=_csv(family,family_fields)
    formal=formal_requested
    observed={"system_status_row_count":len(statuses),"transform_receipt_row_count":len(receipts),"downstream_cluster_row_count":len(downstream),"reference_free_cluster_row_count":len(reference),"d4_half_split_well_row_count":len(half),"bootstrap_result_row_count":len(bootstrap),"fixed_slot_row_count":len(methods),"family_projection_row_count":len(family)}
    if formal:
        for key,value in observed.items():
            if int(config.expected[key]) != value: raise BaselineEvidenceError(f"formal count {key}: expected {config.expected[key]}, observed {value}")
    payloads["manifest.json"]=_canonical({"run_id":run_id,"status":"complete","observed_counts":observed,"payload_files":list(config.artifact_contract["payload_files"]),"phase5_power_state":power})
    payloads["complete.json"]=_canonical({"status":"complete","run_id":run_id,"observed_counts":observed})
    for name,data in payloads.items(): (path/name).write_bytes(data)
    (path/"SHA256SUMS").write_bytes(b"".join(f"{_sha(payloads[name])}  {name}\n".encode() for name in sorted(payloads)))
    return BaselineEvidenceSummary(path,run_id,"complete",len(systems),len(receipts),len(methods),len(bootstrap))


def build_phase6_baseline_evidence_from_inputs(inputs: BaselineEvidenceInputs, systems: Sequence[Phase3System], output_root: Path, *, config: BaselineEvidenceConfig, worker_count: int, project_root: Path=ROOT) -> BaselineEvidenceSummary:
    systems=tuple(systems)
    if worker_count < 1: raise BaselineEvidenceError("worker_count must be positive")
    run_id=_run_id(config,systems,inputs,project_root)
    path=Path(output_root)/run_id
    if path.exists():
        raise BaselineEvidenceError(f"run collision: {path} exists")
    try:
        return _build_phase6_baseline_evidence_from_inputs_unchecked(inputs,systems,output_root,config=config,worker_count=worker_count,project_root=project_root)
    except Exception as error:
        path.mkdir(parents=True,exist_ok=True)
        (path / "failed.json").write_bytes(_canonical({"status":"failed","run_id":run_id,"error_type":type(error).__name__,"reason":str(error) or type(error).__name__}))
        raise


def build_phase6_baseline_evidence(output_root: Path, *, worker_count: int=8, project_root: Path=ROOT) -> BaselineEvidenceSummary:
    config=load_phase6_baseline_evidence_config()
    _validate_authorities(config,project_root)
    return build_phase6_baseline_evidence_from_inputs(_load_real_inputs(config,project_root),_systems(config,project_root),Path(output_root),config=config,worker_count=worker_count,project_root=project_root)


__all__=["BaselineEvidenceError","BaselineEvidenceConfig","BaselineEvidenceInputs","BaselineEvidenceSummary","load_phase6_baseline_evidence_config","make_synthetic_baseline_evidence_inputs","build_phase6_baseline_evidence_from_inputs","build_phase6_baseline_evidence"]
