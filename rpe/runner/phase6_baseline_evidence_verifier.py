"""Independent re-execution verifier for Phase-6 baseline evidence artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

from rpe.downstream.rruff import load_d5_native_spectra, load_d5_raw_cohort
from rpe.downstream.rruff_matching import match_d5_protocol_a_values
from rpe.downstream.sugar_quantitative import load_d4_sugar_cohort
from rpe.evaluation import ReplicatePairInput, SingleSpectrumInput, Spectrum1D
from rpe.methods import TaskLine, load_classical_catalog
from rpe.methods.catalog import Phase3System
from rpe.methods.classical.baseline import BaselineRunStatus, run_baseline_system
from rpe.metrics.consistency import HalfSplitPearsonConsistencyMetric
from rpe.metrics.reference_free import ISLikeStructureToNoiseMetric

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "experiments/phase6/configs/baseline_evidence_v1.json"
_VERIFY_CONTEXT: tuple[object, ...] | None = None


class BaselineEvidenceVerificationError(ValueError):
    """The artifact does not satisfy the frozen independently re-run contract."""


_SCIENTIFIC_EXCEPTIONS = (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError, Warning)
_FIXTURE_SYSTEM_IDS = (
    "0372569fa54794e9f23b37477778de025295f22cd48a52d2aa762991b31119a0",
    "04c18275a33cca6b877282edca72e68808ea658d33b482bc428fbdccf6af8377",
)
_FIXTURE_RECEIPTS_PER_SYSTEM = 328


def _scientific_finite_or_close(value: object, source: str) -> str | None:
    if isinstance(value, BaseException):
        if isinstance(value, _SCIENTIFIC_EXCEPTIONS):
            return "not_evaluable_metric_domain" if source == "metric" else "not_evaluable_consumer_failure"
        raise value
    if not np.isfinite(np.asarray(value)).all():
        return "not_evaluable_metric_domain" if source == "metric" else "not_evaluable_consumer_failure"
    return None


@dataclass(frozen=True)
class BaselineEvidenceVerificationSummary:
    path: Path
    run_id: str
    status: str
    system_count: int
    transform_receipt_row_count: int
    method_evidence_row_count: int
    bootstrap_result_row_count: int


@dataclass(frozen=True)
class _Inputs:
    d5_axis: np.ndarray
    d5_values: np.ndarray
    d5_ids: tuple[str, ...]
    d5_classes: tuple[str, ...]
    d5_splits: tuple[tuple[np.ndarray, np.ndarray], ...]
    d4_axis: np.ndarray
    d4_values: np.ndarray
    d4_targets: np.ndarray
    d4_ids: tuple[str, ...]
    d4_wells: tuple[str, ...]
    d4_rounds: np.ndarray
    d4_repetitions: np.ndarray
    d4_fold_by_well: Mapping[str, int]
    d5_cohort: object | None = None


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
    return (json.dumps(_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_digest(value: np.ndarray, dtype: str = "<f8") -> str:
    return _digest(np.ascontiguousarray(value, dtype=dtype).tobytes())


def _ordered_digest(values: Sequence[str]) -> str:
    return _digest(("\n".join(str(value) for value in values) + "\n").encode())


def _code_authority(root: Path) -> dict[str, dict[str, object]]:
    names=("experiments/phase6/configs/baseline_evidence_v1.json","rpe/methods/classical/baseline.py","rpe/downstream/rruff.py","rpe/downstream/rruff_matching.py","rpe/downstream/sugar_quantitative.py","rpe/metrics/reference_free.py","rpe/metrics/consistency.py","rpe/runner/phase6_baseline_evidence.py","rpe/runner/phase6_baseline_evidence_verifier.py","tools/run_phase6_baseline_evidence.py")
    return {name:{"byte_count":(root/name).stat().st_size,"sha256":_file_digest(root/name)} for name in names if (root/name).is_file()}


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canon(row) for row in rows)


def _csv(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    class _Buffer:
        def __init__(self) -> None: self.parts: list[str] = []
        def write(self, text: str) -> int: self.parts.append(text); return len(text)
    output = _Buffer()
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: "" if row.get(key) is None else (format(row[key], ".17g") if isinstance(row.get(key), float) else row[key]) for key in fields})
    return "".join(output.parts).encode("utf-8")


def _load_config(root: Path) -> tuple[dict[str, object], bytes]:
    raw = (root / CONFIG.relative_to(ROOT)).read_bytes()
    document = json.loads(raw)
    if raw != _canon(document) or document.get("schema_version") != "phase6-baseline-evidence-v1":
        raise BaselineEvidenceVerificationError("canonical baseline config mismatch")
    return document, raw


def _check_authorities(document: Mapping[str, object], root: Path) -> None:
    for name, value in dict(document["authorities"]).items():
        identity = dict(value)
        source = root / str(identity["path"])
        if not source.is_file() or source.stat().st_size != int(identity["byte_count"]) or _file_digest(source) != str(identity["sha256"]):
            raise BaselineEvidenceVerificationError(f"source authority mismatch: {name}")
    authorities = dict(document["authorities"]); ledger = root / str(dict(authorities["baseline_promotion_sha256sums"])["path"])
    for line in ledger.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if not (ledger.parent / relative).is_file() or _file_digest(ledger.parent / relative) != digest:
            raise BaselineEvidenceVerificationError(f"Phase-3 checksum tree mismatch: {relative}")
    promotion = json.loads((root / str(dict(authorities["baseline_promotion"])["path"])).read_text(encoding="utf-8"))
    if tuple(promotion.get("phase5_eligible_system_ids", ())) != tuple(dict(document["inventory"])["system_ids"]):
        raise BaselineEvidenceVerificationError("Phase-3 eligible system order mismatch")
    d5_ledger = root / str(dict(authorities["d5_raw_sha256sums"])["path"])
    for line in d5_ledger.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        if not (d5_ledger.parent / relative).is_file() or _file_digest(d5_ledger.parent / relative) != digest:
            raise BaselineEvidenceVerificationError(f"D5 raw checksum tree mismatch: {relative}")


def _check_inventory_and_ledger(path: Path, document: Mapping[str, object]) -> None:
    expected = set(document["artifact_contract"]["payload_files"]) | {"complete.json", "SHA256SUMS"}
    found = {item.name for item in path.iterdir() if item.is_file()}
    if found != expected:
        raise BaselineEvidenceVerificationError(f"artifact inventory mismatch: expected {sorted(expected)}, found {sorted(found)}")
    ledger = (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed: dict[str, str] = {}
    for line in ledger:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in listed:
            raise BaselineEvidenceVerificationError("invalid SHA256SUMS ledger")
        listed[parts[1]] = parts[0]
    covered = found - {"SHA256SUMS"}
    if set(listed) != covered:
        raise BaselineEvidenceVerificationError("SHA256SUMS inventory mismatch")
    for name, digest in listed.items():
        if _file_digest(path / name) != digest:
            raise BaselineEvidenceVerificationError(f"SHA256SUMS mismatch: {name}")


def _select_rebuild_mode(ids: Sequence[str], receipt_count: int, document: Mapping[str, object]) -> str:
    ordered_ids = tuple(ids)
    if ordered_ids == _FIXTURE_SYSTEM_IDS:
        if receipt_count != len(_FIXTURE_SYSTEM_IDS) * _FIXTURE_RECEIPTS_PER_SYSTEM:
            raise BaselineEvidenceVerificationError("fixture transform receipt inventory mismatch")
        return "fixture"
    formal_ids = tuple(dict(document["inventory"])["system_ids"])
    formal_receipts = int(dict(document["expected"])["transform_receipt_row_count"])
    if ordered_ids == formal_ids and len(ordered_ids) == 112:
        if receipt_count != formal_receipts:
            raise BaselineEvidenceVerificationError("formal transform receipt inventory mismatch")
        return "formal"
    raise BaselineEvidenceVerificationError("system inventory is neither the exact known fixture nor the exact formal inventory")


def _fixture_inputs() -> _Inputs:
    d5_axis = np.array([203., 204., 205., 206., 208., 1800.], dtype="<f8")
    d5_ids: list[str] = []; d5_classes: list[str] = []; d5_rows: list[np.ndarray] = []
    for cls in range(4):
        for replica in range(2):
            d5_ids.append(f"d5-{cls}-{replica}"); d5_classes.append(f"class-{cls}")
            d5_rows.append(np.sin(d5_axis / 100 + cls) + replica * .01 + .03 * d5_axis / 1800)
    index = np.arange(8, dtype=np.int64)
    d5_splits = tuple((index[::2], index[1::2]) for _ in range(5))
    d4_axis = np.linspace(100., 2100., 64, dtype="<f8"); x = np.linspace(0., 1., 64)
    d4_rows: list[np.ndarray] = []; targets: list[np.ndarray] = []; ids: list[str] = []; wells: list[str] = []; rounds: list[int] = []; reps: list[int] = []; folds: dict[str, int] = {}
    for well in range(10):
        wid = f"W{well:02d}"; folds[wid] = well % 5
        target = np.array([(well + item) % 5 / 15 for item in range(4)], dtype="<f8")
        for round_id in range(1, 9):
            for rep in range(1, 5):
                d4_rows.append(.1 + target.sum() * np.exp(-((x-.4)/.12)**2) + .01*np.sin(20*x+round_id+rep))
                targets.append(target); ids.append(f"{wid}-r{round_id}-m{rep}"); wells.append(wid); rounds.append(round_id); reps.append(rep)
    return _Inputs(d5_axis, np.asarray(d5_rows, dtype="<f8"), tuple(d5_ids), tuple(d5_classes), d5_splits, d4_axis, np.asarray(d4_rows, dtype="<f8"), np.asarray(targets, dtype="<f8"), tuple(ids), tuple(wells), np.asarray(rounds), np.asarray(reps), folds)


def _real_inputs(document: Mapping[str, object], root: Path) -> tuple[_Inputs, tuple[Spectrum1D, ...]]:
    authorities = dict(document["authorities"])
    d5_root = root / "data/unified/rruff_raman_raw"
    d5 = load_d5_raw_cohort(root / str(dict(authorities["d5_protocol"])["path"]), d5_root)
    native = load_d5_native_spectra(d5_root, d5.record_ids)
    d4 = load_d4_sugar_cohort(root / str(dict(authorities["d4_protocol"])["path"]), root / str(dict(authorities["d4_source_archive"])["path"]))
    folds: dict[str, int] = {}
    for fold, split in enumerate(d4.folds):
        for index in split:
            folds[str(d4.well_ids[int(index)])] = fold
    inputs = _Inputs(np.asarray(d5.wavenumber, dtype="<f8"), np.asarray(d5.intensity, dtype="<f8"), tuple(d5.record_ids), tuple(str(value) for value in d5.class_labels), tuple((np.asarray(split.query_indices), np.asarray(split.library_indices)) for split in d5.splits), np.asarray(d4.wavenumber, dtype="<f8"), np.asarray(d4.intensity, dtype="<f8"), np.asarray(d4.targets, dtype="<f8"), tuple(d4.record_ids), tuple(d4.well_ids), np.asarray(d4.rounds), np.asarray(d4.repetitions), folds, d5)
    if len(inputs.d5_ids) != 3770 or len(inputs.d4_ids) != 7680 or len(folds) != 240:
        raise BaselineEvidenceVerificationError("real cohort inventory mismatch")
    return inputs, native


def _systems(root: Path, document: Mapping[str, object], ids: Sequence[str]) -> tuple[Phase3System, ...]:
    catalog_path = root / str(dict(document["authorities"])["classical_catalog"]["path"])
    catalog = load_classical_catalog(catalog_path)
    available = {item.system_id: item for item in catalog.systems if item.task_line is TaskLine.BASELINE_CORRECTION}
    try: return tuple(available[item] for item in ids)
    except KeyError as exc: raise BaselineEvidenceVerificationError(f"baseline catalog system missing: {exc.args[0]}") from exc


def _transform(system: Phase3System, axis: np.ndarray, matrix: np.ndarray, ids: Sequence[str], cohort: str) -> tuple[np.ndarray, list[dict[str, object]], str | None]:
    corrected = np.empty_like(matrix, dtype="<f8"); receipts: list[dict[str, object]] = []; failure: str | None = None
    for index, (row, record_id) in enumerate(zip(matrix, ids, strict=True)):
        if failure is not None:
            receipts.append({"system_id": system.system_id, "cohort_id": cohort, "record_id": record_id, "status": "not_run_cohort_transform_closed", "output_sha256": None}); continue
        result = run_baseline_system(system, Spectrum1D(f"{cohort}::{record_id}", record_id, axis, row))
        receipts.append({"system_id": system.system_id, "cohort_id": cohort, "record_id": record_id, "status": result.status.value, "output_sha256": result.corrected_sha256})
        if result.status not in {BaselineRunStatus.COMPLETE, BaselineRunStatus.COMPLETE_WITH_WARNING} or result.corrected_intensity is None:
            failure = result.status.value
        else: corrected[index] = result.corrected_intensity
    return corrected, receipts, failure


def _transform_native_d5(system: Phase3System, spectra: Sequence[Spectrum1D], support: np.ndarray) -> tuple[np.ndarray | None, tuple[np.ndarray, ...] | None, list[dict[str, object]], str | None]:
    projected=[]; native=[]; receipts=[]; failure=None
    for spectrum in spectra:
        result=run_baseline_system(system, spectrum)
        record_id=spectrum.spectrum_id.rsplit("::", 1)[-1]
        receipts.append({"system_id":system.system_id,"cohort_id":"D5","record_id":record_id,"status":result.status.value,"output_sha256":result.corrected_sha256})
        if result.status not in {BaselineRunStatus.COMPLETE, BaselineRunStatus.COMPLETE_WITH_WARNING} or result.corrected_intensity is None:
            failure=result.status.value; continue
        try:
            native.append(np.asarray(result.corrected_intensity,dtype="<f8")); projected.append(_project(spectrum.axis_cm1,result.corrected_intensity,support))
        except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError):
            failure="invalid_native_projection"
    if failure is not None or len(projected) != len(spectra): return None,None,receipts,failure or "incomplete"
    return np.vstack(projected),tuple(native),receipts,None


def _project(axis: np.ndarray, row: np.ndarray, support: np.ndarray) -> np.ndarray:
    native = np.asarray(axis, dtype="<f8"); values = np.asarray(row, dtype="<f8"); grid = np.asarray(support, dtype="<f8")
    left = int(np.searchsorted(native, grid[0], side="right") - 1); right = int(np.searchsorted(native, grid[-1], side="left"))
    if left < 0 or right >= native.size or float(np.max(np.diff(native[left:right + 1]))) > 3.:
        raise BaselineEvidenceVerificationError("D5 native support projection mismatch")
    return np.asarray(np.interp(grid, native, values), dtype="<f4")


def _roles(inputs: _Inputs, fold: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    test = np.asarray([i for i, well in enumerate(inputs.d4_wells) if inputs.d4_fold_by_well[well] == fold], dtype=np.int64)
    validation = np.asarray([i for i, well in enumerate(inputs.d4_wells) if inputs.d4_fold_by_well[well] == (fold + 1) % 5], dtype=np.int64)
    excluded = set(test.tolist()) | set(validation.tolist())
    train = np.asarray([i for i in range(len(inputs.d4_ids)) if i not in excluded], dtype=np.int64)
    return train, validation, test


def _fit(x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, y_valid: np.ndarray, fold: int):
    candidates = []; validations = []
    for components in (2, 4, 8, 16, 32):
        try:
            model = PLSRegression(n_components=components, scale=True, max_iter=500, tol=1e-6, copy=True)
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                with threadpool_limits(limits=1, user_api="blas"):
                    model.fit(x_train, y_train); prediction = np.asarray(model.predict(x_valid), dtype="<f8")
            if _scientific_finite_or_close(prediction, "consumer") is not None:
                raise ValueError("non-finite PLS validation prediction")
            score = float(np.mean(np.sqrt(np.mean((prediction-y_valid)**2, axis=0)) / .32))
            validations.append({"fold": fold, "n_components": components, "macro_normalized_rmse": score, "state": "complete"}); candidates.append((score, components, model))
        except _SCIENTIFIC_EXCEPTIONS as exc:
            validations.append({"fold": fold, "n_components": components, "macro_normalized_rmse": None, "state": "failed_model_lifecycle", "reason": type(exc).__name__})
    if len(candidates) != 5: raise BaselineEvidenceVerificationError(f"incomplete PLS grid for fold {fold}")
    score, selected, model = min(candidates, key=lambda item: (item[0], item[1]))
    if not np.isfinite(score): raise BaselineEvidenceVerificationError(f"non-finite PLS score for fold {fold}")
    digest = _digest(_canon({"fold": fold, "n_components": selected, "validation": validations, "coef_sha256": _digest(np.asarray(model.coef_, dtype="<f8").tobytes())}))
    return model, validations, selected, digest


def _fit_failure_receipt(fold: int, error: BaseException) -> dict[str, object]:
    reason = "BaselineEvidenceError" if isinstance(error, BaselineEvidenceVerificationError) else type(error).__name__
    return {"fold": fold, "state": "not_evaluable_fit_failure", "reason": reason}


def _loss(predicted: np.ndarray, targets: np.ndarray) -> float:
    value=float(np.mean(((np.asarray(predicted) - np.asarray(targets)) ** 2) / (.32 ** 2)))
    if not np.isfinite(value): raise BaselineEvidenceVerificationError("non-finite normalized loss")
    return value


def _is_effects(axis: np.ndarray, identity: np.ndarray, candidate: np.ndarray, ids: Sequence[str]) -> np.ndarray:
    metric = ISLikeStructureToNoiseMetric(); values = []
    for index, record_id in enumerate(ids):
        candidate_value = metric.evaluate(SingleSpectrumInput(Spectrum1D(f"candidate::{record_id}", record_id, axis, candidate[index]))).outputs[0].value
        identity_value = metric.evaluate(SingleSpectrumInput(Spectrum1D(f"identity::{record_id}", record_id, axis, identity[index]))).outputs[0].value
        values.append(float(candidate_value) - float(identity_value))
    return np.asarray(values, dtype="<f8")


def _class_error(labels: Sequence[str], correct: Sequence[bool]) -> float:
    grouped: dict[str, list[bool]] = {}
    for label, value in zip(labels, correct, strict=True): grouped.setdefault(str(label), []).append(bool(value))
    return float(np.mean([1 - np.mean(values) for _, values in sorted(grouped.items())]))


def _match(inputs: _Inputs, query_values: np.ndarray, library_values: np.ndarray, protocol: str) -> tuple[dict[str, float], dict[str, object]]:
    occurrences=[]; split_rows=[]
    for number,(query,library) in enumerate(inputs.d5_splits):
        if inputs.d5_cohort is None:
            q=np.asarray(query_values[query],dtype="<f8"); l=np.asarray(library_values[library],dtype="<f8")
            q=q/np.linalg.norm(q,axis=1)[:,None]; l=l/np.linalg.norm(l,axis=1)[:,None]; scores=q@l.T
            predicted=[inputs.d5_classes[int(library[int(np.argmax(row))])] for row in scores]; truth=[inputs.d5_classes[int(index)] for index in query]
            current=list(zip(truth,(left==right for left,right in zip(truth,predicted,strict=True)),strict=True))
            split_rows.append({"split":number,"query_count":len(current),"top1_sha256":_digest(_canon(current)),"top5_sha256":None,"ranking_sha256":_array_digest(scores)})
        else:
            cohort=inputs.d5_cohort; result=match_d5_protocol_a_values(cohort,cohort.splits[number],condition_id=protocol,query_record_ids=tuple(inputs.d5_ids[int(i)] for i in query),library_record_ids=tuple(inputs.d5_ids[int(i)] for i in library),query_values=query_values[query],library_values=library_values[library])
            current=[(str(label),bool(ok)) for label,ok in zip(result.true_class_labels,result.top1_correct,strict=True)]
            split_rows.append({"split":number,"split_sha256":cohort.splits[number].split_sha256,"query_count":len(current),"top1_sha256":_array_digest(result.top1_correct,"|b1"),"top5_sha256":_array_digest(result.top5_correct,"|b1"),"ranking_sha256":_digest(result.ranked_class_labels.tobytes()+result.ranked_class_scores.tobytes())})
        occurrences.extend(current)
    grouped={label:[ok for actual,ok in occurrences if actual==label] for label in dict.fromkeys(label for label,_ in occurrences)}
    errors={label:float(1-np.mean(values)) for label,values in grouped.items()}
    if _scientific_finite_or_close(list(errors.values()), "consumer") is not None:
        raise ValueError("non-finite D5 matcher class error")
    return errors,{"protocol_id":protocol,"split_receipts":split_rows,"occurrence_count":len(occurrences),"prediction_sha256":_digest(_canon(occurrences)),"class_error_sha256":_digest(_canon(errors))}


def _method_rows(system: Phase3System, states: Mapping[str, tuple[float | None, str]], power: str, direct: str) -> list[dict[str, object]]:
    rows=[]
    for endpoint in ("availability", "direct_gt", "D5-A", "D5-B-matched-reference", "D5-reference-free", "D4-A", "D4-B", "D4-reference-free", "D4-half-split"):
        value, state = states.get(endpoint, (None, "not_evaluable_no_result"))
        if endpoint == "availability": value, state = 1., "complete"
        if endpoint == "direct_gt": value, state = None, direct
        rows.append({"system_id": system.system_id, "family_id": system.family_id, "method_id": system.method_id, "endpoint_id": endpoint, "value": value, "state": state, "phase5_power_state": power})
    return rows


def _fixed_clusters(rows: list[dict[str, object]], systems: Sequence[Phase3System], clusters: Sequence[str], protocols: Sequence[str], kind: str, closure_reasons: Mapping[tuple[str, str], str] | None = None) -> list[dict[str, object]]:
    closure_reasons = {} if closure_reasons is None else closure_reasons
    present = {(str(row["system_id"]), str(row.get("protocol_id", row.get("cohort_id", "method_native"))), str(row.get("cluster_id", row.get("well_id", "")))) for row in rows}
    output = list(rows)
    for system in systems:
        for protocol in protocols:
            for cluster in clusters:
                if (system.system_id, protocol, str(cluster)) not in present:
                    output.append({"system_id": system.system_id, "family_id": system.family_id, "method_id": system.method_id, "protocol_id": protocol, "cohort_id": protocol if kind == "reference" else None, "cluster_id": str(cluster), "well_id": str(cluster) if kind == "half" else None, "effect": None, "state": closure_reasons.get((system.system_id, protocol), "not_evaluable_component_closed")})
    return sorted(output, key=lambda row: (str(row["system_id"]), str(row.get("protocol_id", row.get("cohort_id", "method_native"))), str(row.get("cluster_id", row.get("well_id", "")))))


def _bootstrap(systems: Sequence[Phase3System], downstream: Sequence[Mapping[str, object]], reference: Sequence[Mapping[str, object]], half: Sequence[Mapping[str, object]], resamples: int, seed: int) -> list[dict[str, object]]:
    sources = {"D5-A": (downstream, "D5-A"), "D5-B-matched-reference": (downstream, "D5-B-matched-reference"), "D5-reference-free": (reference, "D5"), "D4-A": (downstream, "D4-A"), "D4-B": (downstream, "D4-B"), "D4-reference-free": (reference, "D4"), "D4-half-split": (half, "method_native")}
    counts={"D5":len({str(row["cluster_id"]) for row in reference if row.get("cohort_id")=="D5"}),"D4":len({str(row["cluster_id"]) for row in reference if row.get("cohort_id")=="D4"})}
    rng=np.random.Generator(np.random.PCG64(seed)); shared={key:rng.integers(0,count,size=(resamples,count)) for key,count in counts.items()}
    output=[]
    for system in systems:
        for endpoint, (rows, marker) in sources.items():
            if endpoint == "D4-half-split": selected=[row for row in rows if row["system_id"] == system.system_id]
            elif endpoint.endswith("reference-free"): selected=[row for row in rows if row["system_id"] == system.system_id and row.get("cohort_id") == marker]
            else: selected=[row for row in rows if row["system_id"] == system.system_id and row.get("protocol_id") == marker]
            values=[row.get("effect") for row in selected]
            cohort="D5" if endpoint.startswith("D5") else "D4"
            if len(values) != counts[cohort] or any(value is None for value in values):
                states={str(row.get("state")) for row in selected}
                if len(values) != counts[cohort] or any(value is not None for value in values) or len(states) != 1 or "complete" in states:
                    raise BaselineEvidenceVerificationError(f"non-canonical closed cluster grid: {system.system_id}/{endpoint}")
                output.append({"system_id":system.system_id,"endpoint_id":endpoint,"estimate":None,"interval_lower":None,"interval_upper":None,"cluster_count":len(values),"state":next(iter(states)),"bootstrap_seed":seed,"resamples":resamples})
            else:
                array=np.asarray(values,dtype="<f8"); means=array[shared[cohort]].mean(axis=1); estimate=float(array.mean()); lower=float(np.percentile(means,2.5)); upper=float(np.percentile(means,97.5))
                if _scientific_finite_or_close((estimate, lower, upper), "metric") is not None:
                    output.append({"system_id":system.system_id,"endpoint_id":endpoint,"estimate":None,"interval_lower":None,"interval_upper":None,"cluster_count":len(array),"state":"not_evaluable_metric_domain","bootstrap_seed":seed,"resamples":resamples})
                else:
                    output.append({"system_id":system.system_id,"endpoint_id":endpoint,"estimate":estimate,"interval_lower":lower,"interval_upper":upper,"cluster_count":len(array),"state":"complete","bootstrap_seed":seed,"resamples":resamples})
    return output


def _initialize_verifier_worker(inputs: _Inputs, native_d5: tuple[Spectrum1D, ...] | None, support: np.ndarray, systems: Sequence[Phase3System], raw_projected: np.ndarray, identity_errors: Mapping[str, float], identity_models: Mapping[int, tuple[object, object, int, str]], d5_identity_reference: np.ndarray, d4_identity_reference: np.ndarray, identity_half: Mapping[str, float], identity_loss: Mapping[str, float]) -> None:
    global _VERIFY_CONTEXT
    _VERIFY_CONTEXT = (inputs, native_d5, support, tuple(systems), raw_projected, identity_errors, identity_models, d5_identity_reference, d4_identity_reference, identity_half, identity_loss)


def _evaluate_verifier_system(index: int):
    """Execute the native correction stage for one complete system in a fork worker.

    Returned matrices are subsequently consumed by the verifier-local canonical
    aggregation path; no candidate correction is re-run in the parent.
    """
    if _VERIFY_CONTEXT is None:
        raise BaselineEvidenceVerificationError("verifier worker context is not initialized")
    inputs, native_d5, support, systems, raw_projected, identity_errors, identity_models, d5_identity_reference, d4_identity_reference, identity_half, identity_loss = _VERIFY_CONTEXT
    system = systems[index]
    if native_d5 is None:
        d5, rec5, fail5 = _transform(system, inputs.d5_axis, inputs.d5_values, inputs.d5_ids, "D5")
        d5_native = None
    else:
        d5, d5_native, rec5, fail5 = _transform_native_d5(system, native_d5, support)
    d4, rec4, fail4 = _transform(system, inputs.d4_axis, inputs.d4_values, inputs.d4_ids, "D4")
    # Run all candidate consumer classes in the worker.  The returned receipts
    # let the parent merge in canonical system order without a parent-side
    # candidate transform/consumer reexecution.
    science = {"d5": {}, "d4": {}}
    if not fail5:
        projected = d5 if d5_native is not None else np.vstack([_project(inputs.d5_axis, row, support) for row in d5])
        for protocol, library in (("D5-A", raw_projected), ("D5-B-matched-reference", projected)):
            try:
                errors, receipt = _match(inputs, projected, library, protocol)
                science["d5"][protocol] = {"errors": errors, "receipt": receipt}
            except _SCIENTIFIC_EXCEPTIONS as exc:
                science["d5"][protocol] = {"closed": "not_evaluable_consumer_failure", "reason": type(exc).__name__}
        try:
            metric = ISLikeStructureToNoiseMetric(); values = d5_native if d5_native is not None else d5; axis = None if d5_native is not None else inputs.d5_axis; effects=[]
            for i, value in enumerate(values):
                spectrum = (Spectrum1D(f"candidate::{inputs.d5_ids[i]}", inputs.d5_ids[i], axis, value) if axis is not None else Spectrum1D(native_d5[i].spectrum_id, native_d5[i].sample_id, native_d5[i].axis_cm1, value))
                effects.append(float(metric.evaluate(SingleSpectrumInput(spectrum)).outputs[0].value) - float(d5_identity_reference[i]))
            science["d5"]["reference_free"] = np.asarray(effects, dtype="<f8")
        except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError):
            science["d5"]["reference_free"] = None
    if not fail4:
        predictions_a={}; predictions_b={}; model_rows=[]; a_closed=None; b_closed=None
        for fold in range(5):
            train, valid, test = _roles(inputs, fold); identity = identity_models[fold][0]
            try:
                with threadpool_limits(limits=1, user_api="blas"): a=np.asarray(identity.predict(d4[test,1:].astype("<f4")),dtype="<f8")
                if _scientific_finite_or_close(a, "consumer") is not None: raise ValueError("non-finite D4-A prediction")
                predictions_a.update({int(i):v for i,v in zip(test,a,strict=True)})
            except _SCIENTIFIC_EXCEPTIONS as exc: a_closed=type(exc).__name__
            try:
                candidate, rows, selected, digest = _fit(d4[train,1:].astype("<f4"), inputs.d4_targets[train], d4[valid,1:].astype("<f4"), inputs.d4_targets[valid], fold)
                with threadpool_limits(limits=1, user_api="blas"): b=np.asarray(candidate.predict(d4[test,1:].astype("<f4")),dtype="<f8")
                if _scientific_finite_or_close(b, "consumer") is not None: raise ValueError("non-finite D4-B prediction")
                predictions_b.update({int(i):v for i,v in zip(test,b,strict=True)}); model_rows.append({"fold":fold,"selected_n_components":selected,"model_digest":digest,"validation_scores_sha256":_digest(_canon(rows))})
            except _SCIENTIFIC_EXCEPTIONS as exc:
                b_closed=type(exc).__name__; model_rows.append(_fit_failure_receipt(fold, exc))
        try:
            metric=ISLikeStructureToNoiseMetric(); ref=np.asarray([float(metric.evaluate(SingleSpectrumInput(Spectrum1D(f"candidate::{rid}",rid,inputs.d4_axis,row))).outputs[0].value)-float(d4_identity_reference[i]) for i,(rid,row) in enumerate(zip(inputs.d4_ids,d4,strict=True))],dtype="<f8")
        except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError): ref=None
        try:
            half_metric=HalfSplitPearsonConsistencyMetric(); halves={}
            for well in dict.fromkeys(inputs.d4_wells):
                take=sorted([i for i,w in enumerate(inputs.d4_wells) if w==well],key=lambda i:(int(inputs.d4_rounds[i]),int(inputs.d4_repetitions[i]),inputs.d4_ids[i]))
                candidate=float(half_metric.evaluate(ReplicatePairInput(Spectrum1D("candidate-a",well,inputs.d4_axis,np.mean(d4[take[::2]],axis=0)),Spectrum1D("candidate-b",well,inputs.d4_axis,np.mean(d4[take[1::2]],axis=0)))).outputs[0].value); halves[well]=(candidate, candidate-float(identity_half[well]))
        except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError): halves=None
        science["d4"]={"predictions_a":predictions_a,"predictions_b":predictions_b,"model_rows":model_rows,"a_closed":a_closed,"b_closed":b_closed,"reference_free":ref,"half":halves}
    downstream=[]; reference=[]; half=[]; states={}; matcher_receipts={}; model_receipts=[]; prediction_receipts={}
    classes=tuple(dict.fromkeys(inputs.d5_classes)); wells=tuple(dict.fromkeys(inputs.d4_wells))
    if fail5:
        states.update({endpoint:(None,"not_evaluable_transform_failure") for endpoint in ("D5-A","D5-B-matched-reference","D5-reference-free")})
    else:
        for protocol in ("D5-A","D5-B-matched-reference"):
            try:
                data=science["d5"][protocol]
                if "closed" in data:
                    states[protocol]=(None,str(data["closed"])); matcher_receipts[protocol]={"state":data["closed"],"reason":data["reason"]}; continue
                matcher_receipts[protocol]=data["receipt"]; values=[]
                for cluster in classes:
                    effect=float(data["errors"][str(cluster)]-identity_errors[str(cluster)]); values.append(effect); downstream.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"protocol_id":protocol,"cluster_id":str(cluster),"identity_value":identity_errors[str(cluster)],"candidate_value":data["errors"][str(cluster)],"effect":effect,"state":"complete"})
                states[protocol]=(float(np.mean(values)),"complete")
            except (ValueError, ArithmeticError, FloatingPointError, np.linalg.LinAlgError) as exc:
                downstream[:]=[row for row in downstream if row.get("protocol_id") != protocol]; states[protocol]=(None,"not_evaluable_consumer_failure"); matcher_receipts[protocol]={"state":"not_evaluable_consumer_failure","reason":type(exc).__name__}
        effects=science["d5"]["reference_free"]
        if effects is None: states["D5-reference-free"]=(None,"not_evaluable_metric_domain")
        else:
            values=[]
            for cluster in classes:
                take=[i for i,v in enumerate(inputs.d5_classes) if v==cluster]; value=float(np.mean(effects[take])); values.append(value); reference.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"cohort_id":"D5","cluster_id":str(cluster),"effect":value,"state":"complete"})
            states["D5-reference-free"]=(float(np.mean(values)),"complete")
    if fail4:
        states.update({endpoint:(None,"not_evaluable_transform_failure") for endpoint in ("D4-A","D4-B","D4-reference-free","D4-half-split")})
    else:
        data=science["d4"]; model_receipts=data["model_rows"]
        for protocol,predictions in (("D4-A",data["predictions_a"]),("D4-B",data["predictions_b"])):
            closed = data["a_closed"] if protocol == "D4-A" else data["b_closed"]
            if closed is not None:
                states[protocol]=(None, "not_evaluable_consumer_failure" if protocol == "D4-A" else "not_evaluable_fit_failure"); continue
            values=[]
            for well in wells:
                take=[i for i,w in enumerate(inputs.d4_wells) if w==well]; candidate=_loss(np.asarray([predictions[i] for i in take]),inputs.d4_targets[take]); effect=candidate-float(identity_loss[well]); values.append(effect); downstream.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"protocol_id":protocol,"cluster_id":well,"identity_value":identity_loss[well],"candidate_value":candidate,"effect":effect,"state":"complete"})
            states[protocol]=(float(np.mean(values)),"complete"); prediction_receipts[protocol]=_digest(_canon({str(i):predictions[i] for i in sorted(predictions)}))
        if data["reference_free"] is None: states["D4-reference-free"]=(None,"not_evaluable_metric_domain")
        else:
            values=[]
            for well in wells:
                take=[i for i,w in enumerate(inputs.d4_wells) if w==well]; value=float(np.mean(data["reference_free"][take])); values.append(value); reference.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"cohort_id":"D4","cluster_id":well,"effect":value,"state":"complete"})
            states["D4-reference-free"]=(float(np.mean(values)),"complete")
        if data["half"] is None: states["D4-half-split"]=(None,"not_evaluable_metric_domain")
        else:
            values=[]
            for well,(candidate,effect) in data["half"].items(): values.append(effect); half.append({"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"well_id":well,"identity_value":identity_half[well],"candidate_value":candidate,"effect":effect,"state":"complete"})
            states["D4-half-split"]=(float(np.mean(values)),"complete")
    status={"system_id":system.system_id,"family_id":system.family_id,"method_id":system.method_id,"component_states":states,"d5":{"consumer_input_sha256":{"D5-A":None if fail5 else _array_digest(d5 if d5_native is not None else np.vstack([_project(inputs.d5_axis,row,support) for row in d5]),"<f4"),"D5-B-matched-reference":None if fail5 else _array_digest(d5 if d5_native is not None else np.vstack([_project(inputs.d5_axis,row,support) for row in d5]),"<f4")},"matcher_receipts":matcher_receipts},"d4":{"native_corrected_matrix_sha256":None if fail4 else _array_digest(d4),"consumer_input_sha256":{"D4-A":None if fail4 else _array_digest(d4),"D4-B":None if fail4 else _array_digest(d4)},"model_receipts":model_receipts,"prediction_receipts":prediction_receipts}}
    return status, rec5+rec4, downstream, reference, half


def _rebuild(path: Path, root: Path, worker_count: int) -> tuple[dict[str, bytes], BaselineEvidenceVerificationSummary]:
    document, config_bytes = _load_config(root); _check_authorities(document, root)
    status_rows = [json.loads(line) for line in (path / "system_status.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ids = tuple(str(row["system_id"]) for row in status_rows)
    if not ids or len(ids) != len(set(ids)):
        raise BaselineEvidenceVerificationError("system-status identity mismatch")
    receipt_count = sum(1 for _ in (path / "transform_receipts.jsonl").read_text(encoding="utf-8").splitlines() if _)
    mode = _select_rebuild_mode(ids, receipt_count, document)
    is_fixture = mode == "fixture"
    if is_fixture:
        inputs = _fixture_inputs(); native_d5 = None; support = np.array([204., 206., 208.], dtype="<f8")
    else:
        inputs, native_d5 = _real_inputs(document, root); support = np.arange(204., 1800.1, 2., dtype="<f8")
        expected_ids = tuple(dict(document["inventory"])["system_ids"])
        expected_families = dict(dict(document["inventory"])["family_counts"])
        if receipt_count != len(ids) * (len(inputs.d5_ids) + len(inputs.d4_ids)):
            raise BaselineEvidenceVerificationError("transform receipt inventory is neither synthetic nor canonical real cohort")
    systems = _systems(root, document, ids)
    if not is_fixture and {family: sum(system.family_id == family for system in systems) for family in expected_families} != expected_families:
        raise BaselineEvidenceVerificationError("formal verifier family population mismatch")
    power = str(document["gate"]["phase5_power_state"]); direct = str(document["direct_gt_state"])
    raw_projected=(np.vstack([_project(s.axis_cm1,s.intensity,support) for s in native_d5]) if native_d5 is not None else np.vstack([_project(inputs.d5_axis,row,support) for row in inputs.d5_values]))
    identity_errors, identity_matcher_a = _match(inputs,raw_projected,raw_projected,"identity-D5-A")
    _, identity_matcher_b = _match(inputs,raw_projected,raw_projected,"identity-D5-B")
    if identity_matcher_a["prediction_sha256"] != identity_matcher_b["prediction_sha256"] or identity_matcher_a["class_error_sha256"] != identity_matcher_b["class_error_sha256"]:
        raise BaselineEvidenceVerificationError("D5 A/B raw identity matcher mismatch")
    identity_matcher = {"D5-A": identity_matcher_a, "D5-B-matched-reference": identity_matcher_b, "exact_equal": True}
    def raw_identity_consumer():
        models={}; predictions={}; receipts=[]
        for fold in range(5):
            train, valid, test = _roles(inputs, fold); model, validation, selected, digest = _fit(inputs.d4_values[train,1:].astype("<f4"), inputs.d4_targets[train], inputs.d4_values[valid,1:].astype("<f4"), inputs.d4_targets[valid], fold)
            with threadpool_limits(limits=1, user_api="blas"): prediction=np.asarray(model.predict(inputs.d4_values[test,1:].astype("<f4")), dtype="<f8")
            models[fold] = (model, validation, selected, digest)
            receipts.append({"fold":fold,"selected_n_components":selected,"model_digest":digest,"validation_scores_sha256":_digest(_canon(validation))})
            predictions.update({int(index): value for index, value in zip(test, prediction, strict=True)})
        return models,predictions,receipts
    identity_models, identity_predictions, identity_model_receipts_a = raw_identity_consumer()
    _, identity_predictions_b, identity_model_receipts_b = raw_identity_consumer()
    if identity_model_receipts_a != identity_model_receipts_b or _digest(_canon(identity_predictions)) != _digest(_canon(identity_predictions_b)):
        raise BaselineEvidenceVerificationError("D4 A/B raw identity model or prediction mismatch")
    identity_model_receipts={"D4-A":identity_model_receipts_a,"D4-B":identity_model_receipts_b,"prediction_sha256":_digest(_canon(identity_predictions)),"exact_equal":True}
    wells=tuple(dict.fromkeys(inputs.d4_wells)); classes=tuple(dict.fromkeys(inputs.d5_classes)); identity_loss={well:_loss(np.asarray([identity_predictions[i] for i,w in enumerate(inputs.d4_wells) if w==well]),inputs.d4_targets[[i for i,w in enumerate(inputs.d4_wells) if w==well]]) for well in wells}
    metric=ISLikeStructureToNoiseMetric()
    d5_identity_reference=np.asarray([float(metric.evaluate(SingleSpectrumInput(s)).outputs[0].value) for s in native_d5],dtype="<f8") if native_d5 is not None else np.asarray([float(metric.evaluate(SingleSpectrumInput(Spectrum1D(f"identity::{rid}",rid,inputs.d5_axis,row))).outputs[0].value) for rid,row in zip(inputs.d5_ids,inputs.d5_values,strict=True)],dtype="<f8")
    d4_identity_reference=np.asarray([float(metric.evaluate(SingleSpectrumInput(Spectrum1D(f"identity::{rid}",rid,inputs.d4_axis,row))).outputs[0].value) for rid,row in zip(inputs.d4_ids,inputs.d4_values,strict=True)],dtype="<f8")
    half_metric=HalfSplitPearsonConsistencyMetric(); identity_half={}
    for well in wells:
        take=sorted([i for i,value in enumerate(inputs.d4_wells) if value==well],key=lambda i:(int(inputs.d4_rounds[i]),int(inputs.d4_repetitions[i]),inputs.d4_ids[i]))
        identity_half[well]=float(half_metric.evaluate(ReplicatePairInput(Spectrum1D(f"{well}-identity-a",well,inputs.d4_axis,np.mean(inputs.d4_values[take[::2]],axis=0)),Spectrum1D(f"{well}-identity-b",well,inputs.d4_axis,np.mean(inputs.d4_values[take[1::2]],axis=0)))).outputs[0].value)
    if worker_count == 1:
        _initialize_verifier_worker(inputs, native_d5, support, systems, raw_projected, identity_errors, identity_models, d5_identity_reference, d4_identity_reference, identity_half, identity_loss)
        worker_results = [_evaluate_verifier_system(index) for index in range(len(systems))]
    else:
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("fork"), initializer=_initialize_verifier_worker, initargs=(inputs, native_d5, support, systems, raw_projected, identity_errors, identity_models, d5_identity_reference, d4_identity_reference, identity_half, identity_loss)) as executor:
            worker_results = list(executor.map(_evaluate_verifier_system, range(len(systems)), chunksize=1))
    if tuple(result[0]["system_id"] for result in worker_results) != tuple(system.system_id for system in systems):
        raise BaselineEvidenceVerificationError("verifier canonical whole-system worker order mismatch")
    statuses=[]; receipts=[]; downstream=[]; reference=[]; half=[]; methods=[]
    for system, worker_result in zip(systems, worker_results, strict=True):
        status, current_receipts, current_downstream, current_reference, current_half = worker_result
        statuses.append(status); receipts.extend(current_receipts); downstream.extend(current_downstream); reference.extend(current_reference); half.extend(current_half)
        methods.extend(_method_rows(system, status["component_states"], power, direct))
    closure_reasons = {}
    for status in statuses:
        for endpoint, (_, state) in status["component_states"].items():
            if state != "complete": closure_reasons[(str(status["system_id"]), {"D5-reference-free": "D5", "D4-reference-free": "D4", "D4-half-split": "method_native"}.get(endpoint, endpoint))] = str(state)
    downstream = _fixed_clusters([row for row in downstream if row.get("protocol_id") in {"D5-A", "D5-B-matched-reference"}], systems, classes, ("D5-A", "D5-B-matched-reference"), "downstream", closure_reasons) + _fixed_clusters([row for row in downstream if row.get("protocol_id") in {"D4-A", "D4-B"}], systems, wells, ("D4-A", "D4-B"), "downstream", closure_reasons)
    reference = _fixed_clusters([row for row in reference if row.get("cohort_id") == "D5"], systems, classes, ("D5",), "reference", closure_reasons) + _fixed_clusters([row for row in reference if row.get("cohort_id") == "D4"], systems, wells, ("D4",), "reference", closure_reasons)
    half = _fixed_clusters(half, systems, wells, ("method_native",), "half", closure_reasons)
    bootstrap = _bootstrap(systems, downstream, reference, half, int(document["bootstrap"]["resamples"]), int(document["bootstrap"]["seed"]))
    by_key={(row["system_id"],row["endpoint_id"]):row for row in bootstrap}; endpoint_by_id={str(row["endpoint_id"]):row for row in document["endpoint_manifest"]}; full_methods=[]
    for row in methods:
        endpoint_id=str(row["endpoint_id"]); endpoint=endpoint_by_id[endpoint_id]; result=by_key.get((row["system_id"],endpoint_id))
        source=("system_status.jsonl" if endpoint_id in {"availability","direct_gt"} else "downstream_cluster_rows.jsonl" if endpoint_id in {"D5-A","D5-B-matched-reference","D4-A","D4-B"} else "reference_free_cluster_rows.jsonl" if endpoint_id.endswith("reference-free") else "half_split_well_rows.jsonl")
        full_methods.append({"evidence_id":f"{row['system_id']}:{endpoint_id}","task_line":"baseline_correction","system_id":row["system_id"],"family_id":row["family_id"],"endpoint_id":endpoint_id,"protocol_id":endpoint["protocol_id"],"cohort_id":"D5_raw_rruff" if endpoint_id.startswith("D5") else "D4_low_snr_sugar" if endpoint_id.startswith("D4") else "baseline_method_inventory","evidence_component":endpoint["evidence_component"],"metric_output_id":endpoint["metric_output_id"],"estimate":None if endpoint_id in {"availability","direct_gt"} else result["estimate"],"interval_lower":None if result is None else result["interval_lower"],"interval_upper":None if result is None else result["interval_upper"],"preferred_direction":endpoint["preferred_direction"],"state":row["state"] if result is None else result["state"],"reason_code":"" if (row["state"] if result is None else result["state"]) == "complete" else (row["state"] if result is None else result["state"]),"phase5_power_state":power,"source_path":source,"source_sha256":None})
    methods=full_methods; families=[]
    for family,count in dict(document["inventory"])["family_counts"].items():
        for endpoint in document["endpoint_manifest"]:
            endpoint_id=str(endpoint["endpoint_id"]); values=[float(row["estimate"]) for row in methods if row["family_id"]==family and row["endpoint_id"]==endpoint_id and row["estimate"] is not None]
            families.append({"family_id":family,"endpoint_id":endpoint_id,"registered_system_count":int(count),"complete_system_count":len(values),"median_estimate":None if not values else float(np.median(values)),"min_estimate":None if not values else float(np.min(values)),"max_estimate":None if not values else float(np.max(values))})
    config_sha=_digest(config_bytes); authority=_code_authority(root); identity={"config_sha256":config_sha,"systems":[item.system_id for item in systems],"d5_record_ids_sha256":_ordered_digest(inputs.d5_ids),"d4_record_ids_sha256":_ordered_digest(inputs.d4_ids),"code_authority":authority}; run_id=str(document["artifact_contract"]["run_prefix"])+_digest(b"rpe-phase6-baseline-evidence-v1\0"+_canon(identity))
    payloads={"config.json":_canon(document),"authority_bridge.json":_canon({"config_sha256":config_sha,"run_identity":identity}),"preflight.json":_canon({"phase5_power_state":power,"identity_d5_matcher":identity_matcher,"identity_d4_models":identity_model_receipts,"identity_d4_prediction_sha256":_digest(_canon(identity_predictions)),"d5_record_count":len(inputs.d5_ids),"d4_record_count":len(inputs.d4_ids),"system_count":len(systems)}),"system_status.jsonl":_jsonl(statuses),"transform_receipts.jsonl":_jsonl(receipts),"downstream_cluster_rows.jsonl":_jsonl(downstream),"reference_free_cluster_rows.jsonl":_jsonl(reference),"half_split_well_rows.jsonl":_jsonl(half),"bootstrap_results.jsonl":_jsonl(bootstrap)}
    method_fields=("evidence_id","task_line","system_id","family_id","endpoint_id","protocol_id","cohort_id","evidence_component","metric_output_id","estimate","interval_lower","interval_upper","preferred_direction","state","reason_code","phase5_power_state","source_path","source_sha256"); family_fields=("family_id","endpoint_id","registered_system_count","complete_system_count","median_estimate","min_estimate","max_estimate")
    sources={name:_digest(data) for name,data in payloads.items()}; methods=[dict(row,source_sha256=sources[row["source_path"]]) for row in methods]; payloads["method_evidence_rows.csv"]=_csv(methods,method_fields); payloads["family_projection.csv"]=_csv(families,family_fields)
    observed={"system_status_row_count":len(statuses),"transform_receipt_row_count":len(receipts),"downstream_cluster_row_count":len(downstream),"reference_free_cluster_row_count":len(reference),"d4_half_split_well_row_count":len(half),"bootstrap_result_row_count":len(bootstrap),"fixed_slot_row_count":len(methods),"family_projection_row_count":len(families)}
    payloads["manifest.json"]=_canon({"run_id":run_id,"status":"complete","observed_counts":observed,"payload_files":list(document["artifact_contract"]["payload_files"]),"phase5_power_state":power}); payloads["complete.json"]=_canon({"status":"complete","run_id":run_id,"observed_counts":observed})
    payloads["SHA256SUMS"] = "".join(f"{_digest(payloads[name])}  {name}\n" for name in sorted(payloads)).encode("utf-8")
    return payloads, BaselineEvidenceVerificationSummary(path,run_id,"complete",len(systems),len(receipts),len(methods),len(bootstrap))


def verify_phase6_baseline_evidence(path: Path, *, worker_count: int, project_root: Path = ROOT) -> BaselineEvidenceVerificationSummary:
    path=Path(path); root=Path(project_root)
    if worker_count < 1: raise BaselineEvidenceVerificationError("worker_count must be positive")
    if not path.is_dir(): raise BaselineEvidenceVerificationError("run path must exist")
    document, _ = _load_config(root); _check_inventory_and_ledger(path, document)
    expected, summary = _rebuild(path, root, worker_count)
    observed={item.name:item.read_bytes() for item in path.iterdir() if item.is_file()}
    mismatches=sorted(name for name in set(expected)|set(observed) if expected.get(name)!=observed.get(name))
    if mismatches: raise BaselineEvidenceVerificationError(f"artifact byte mismatch/tamper: {', '.join(mismatches)}")
    return summary


__all__ = ["BaselineEvidenceVerificationError", "BaselineEvidenceVerificationSummary", "verify_phase6_baseline_evidence"]
