"""D4 Sugar aggregate adapter for the Step-7 Phase-4 sensitivity audit."""
from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.downstream.sugar_quantitative import load_d4_sugar_cohort
from rpe.evaluation import Spectrum1D
from rpe.methods import load_classical_catalog
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner import phase4_d4_eligibility as d4_eligibility
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase4_d4_eligibility import load_phase4_d4_eligibility_config
from rpe.runner.phase6_appendix_audits import AppendixAuditsError, assemble_phase4_sensitivity_rows
from rpe.runner.phase6_appendix_metric_core import (
    METRIC_IDS, PEAK_METRIC_IDS, evaluate_record_sensitivity, pack_record_sensitivity_values,
    rematerialize_p8_p12_conditions, unpack_record_sensitivity_values,
)


ROOT = Path(__file__).resolve().parents[2]
_D4_PROTOCOL = "experiments/phase05/configs/d4_sugar_protocol.json"
_D4_ARCHIVE = "data/raw/ramanbench/cache/10779223/Raw data.zip"
_ELIGIBILITY = "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json"
_PHASE1 = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
_SWEEP = "experiments/shared/raman_perturbation_sweep_v1.json"
_CATALOG = "experiments/phase3/configs/classical_system_catalog_v1.json"


@dataclass(frozen=True)
class DatasetSensitivityResult:
    audit_status: tuple[Mapping[str, object], ...]
    resampling_alignment: tuple[Mapping[str, object], ...]
    resampling_rank_stability: tuple[Mapping[str, object], ...]
    normalization_alignment: tuple[Mapping[str, object], ...]
    normalization_rank_stability: tuple[Mapping[str, object], ...]
    peak_tolerance_phase4_alignment: tuple[Mapping[str, object], ...]
    peak_tolerance_phase4_rank_stability: tuple[Mapping[str, object], ...]
    identity: Mapping[str, object]


_WORKER: dict[str, object] = {}


def _conditions(config_raw: Mapping[str, object]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    resampling = config_raw.get("resampling_conditions", {})
    if not isinstance(resampling, Mapping):
        raise AppendixAuditsError("sugar adapter", "resampling conditions invalid")
    # Config JSON is canonical (therefore its object keys are sorted), while
    # the protocol explicitly fixes this scientific evaluation order.
    names = ("linear", "cubic", "pchip")
    declared = resampling.get("interpolators", {})
    if not isinstance(declared, Mapping) or set(declared) != set(names):
        raise AppendixAuditsError("sugar adapter", "frozen interpolator inventory mismatch")
    spacings = tuple(float(value) for value in resampling.get("spacing_cm1", ()))
    result = tuple(f"{name}:{spacing:g}" for name in names for spacing in spacings)
    normalizations = tuple(str(value) for value in config_raw.get("normalization_conditions", ()))
    if result != ("linear:0.5", "linear:1", "linear:2", "cubic:0.5", "cubic:1", "cubic:2", "pchip:0.5", "pchip:1", "pchip:2") or normalizations != ("none", "maximum", "area", "snv"):
        raise AppendixAuditsError("sugar adapter", "frozen condition order mismatch")
    return (("resampling", result), ("normalization", normalizations), ("phase4_peak_tolerance", ("tolerance:1", "tolerance:2", "tolerance:4", "tolerance:8")))


def _panels(config_raw: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    value = tuple(item for item in config_raw.get("panel_manifest", ()) if isinstance(item, Mapping) and str(item.get("panel_id")) in {"d4_a", "d4_b"})
    if tuple(str(item.get("panel_id")) for item in value) != ("d4_a", "d4_b"):
        raise AppendixAuditsError("sugar adapter", "frozen D4 panels absent")
    return value


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)


def _worker_init(root: str) -> None:
    os.environ.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    project = Path(root)
    eligibility = load_phase4_d4_eligibility_config(project / _ELIGIBILITY)
    catalog = load_classical_catalog(project / _CATALOG)
    systems = [item for item in catalog.systems if item.system_id == eligibility.cwt_system_id]
    if len(systems) != 1:
        raise AppendixAuditsError("sugar adapter", "CWT system did not resolve exactly once")
    _WORKER.update({
        "phase1": load_phase1_core_config(project / _PHASE1), "sweep": load_perturbation_sweep_config(project / _SWEEP),
        "cwt": systems[0], "p10": eligibility.p10_memory_budget_bytes,
        "support_counts": {0.5: 7077, 1.0: 3539, 2.0: 1770},
    })


def _evaluate_job(job: tuple[int, str, str, np.ndarray, np.ndarray]) -> tuple[int, str, object]:
    order, record_id, well_id, axis, intensity = job
    spectrum = Spectrum1D(f"d4_sugar_low_snr::{record_id}", well_id, np.asarray(axis, dtype="<f8"), np.asarray(intensity, dtype="<f8"))
    source = d4_eligibility._phase1_source_for_spectrum(spectrum, order=order)
    with threadpool_limits(limits=1, user_api="blas"):
        ids, spectra = rematerialize_p8_p12_conditions(source, _WORKER["phase1"], _WORKER["sweep"], p10_memory_budget_bytes=int(_WORKER["p10"]))
        values = evaluate_record_sensitivity(spectrum, ids, spectra, record_id=record_id, support_start_cm1=146.0, support_stop_cm1=3684.0, support_point_counts=_WORKER["support_counts"], support_max_gap_cm1=3.665, cwt_system=_WORKER["cwt"])
    return order, well_id, pack_record_sensitivity_values(values)


_pack_record_result = pack_record_sensitivity_values
_unpack_record_result = unpack_record_sensitivity_values


def _closed(panel: Mapping[str, object], component: str, condition: str, metric_ids: Sequence[str], reason: str):
    common = {"panel_id": str(panel["panel_id"]), "endpoint_id": str(panel["endpoint_id"]), "protocol_id": str(panel["protocol_id"]), "condition_id": condition}
    reference = {"resampling": "native", "normalization": "none", "phase4_peak_tolerance": "tolerance:2"}[component]
    alignment = tuple({**common, "metric_output_id": metric, "ag": None, "acc_cross": None, "state": "not_evaluable", "reason_code": reason} for metric in metric_ids)
    ranks = tuple({**common, "reference_condition_id": reference, "statistic": statistic, "tau_b": None, "reference_tie_count": None, "candidate_tie_count": None, "max_abs_rank_displacement": None, "stable": False, "state": "not_evaluable", "reason_code": "not_evaluable_incomplete_metric_set"} for statistic in ("ag", "acc_cross"))
    return {"component": component, "status_order": 0, **common, "state": "not_evaluable", "reason_code": reason, "numerical_row_count": 0, "rank_row_count": 2, "evidence_key": "metric_core_closure"}, alignment, ranks


def _aggregate_harms(values: Sequence[float], *, protocol_id: str) -> float:
    if not values:
        raise AppendixAuditsError("sugar adapter", "empty acquisition harms")
    if protocol_id == "a":
        return float(np.mean(np.asarray(values, dtype=np.float64)))
    if protocol_id == "b":
        total = 0.0
        for value in values:
            total += float(value)
        return total / len(values)
    raise AppendixAuditsError("sugar adapter", "unknown protocol reduction")


def _rows_from_records(records: Sequence[tuple[str, object]], *, surface: str, condition_index: int, metric_ids: tuple[str, ...], protocol_id: str) -> tuple[tuple[Mapping[str, object], ...] | None, str]:
    grouped: dict[tuple[str, str, float, str], list[float]] = {}
    for well_id, value in records:
        reason_keys = tuple(getattr(value, "closure_reasons").keys())
        if surface == "resampling":
            prefix = f"resampling:{getattr(value, 'resampling_ids')[condition_index]}"
        elif surface == "normalization":
            prefix = f"normalization:{getattr(value, 'normalization_ids')[condition_index]}"
        else:
            prefix = "peak_tolerance"
        if any(key == prefix or key.startswith(prefix + ":") for key in reason_keys):
            return None, "not_evaluable_metric_core_closure"
        if surface == "resampling": values = getattr(value, "resampling_harms")[condition_index]; ids = metric_ids
        elif surface == "normalization": values = getattr(value, "normalization_harms")[condition_index]; ids = metric_ids
        else:
            # Scalar harms are native `none`; tolerance only replaces five peak harms.
            scalar = getattr(value, "normalization_harms")[0, :, :8]
            peak = getattr(value, "peak_tolerance_harms")[condition_index]
            values = np.concatenate((scalar, peak), axis=1); ids = metric_ids
        for condition_id, row in zip(getattr(value, "condition_ids"), values, strict=True):
            perturbation, raw_alpha = condition_id.split(":", 1)
            alpha = float(np.frombuffer(bytes.fromhex(raw_alpha), dtype="<f8")[0])
            if not np.isfinite(row).all(): return None, "not_evaluable_metric_core_nonfinite"
            for metric, harm in zip(ids, row, strict=True): grouped.setdefault((well_id, perturbation, alpha, metric), []).append(float(harm))
    rows=[]
    for key, values in sorted(grouped.items()):
        if len(values) != 32: return None, "not_evaluable_incomplete_well_acquisition_grid"
        well_id, perturbation, alpha, metric = key
        rows.append({"well_id": well_id, "perturbation_id": perturbation, "alpha": alpha, "metric_output_id": metric, "metric_harm": _aggregate_harms(values, protocol_id=protocol_id)})
    if len(rows) != 240 * 40 * len(metric_ids): return None, "not_evaluable_incomplete_well_condition_grid"
    return tuple(rows), ""


def _aggregate(panel: Mapping[str, object], parent_rows: Sequence[Mapping[str, object]], parent_alignment: Mapping[str, tuple[float, float]], records: Sequence[tuple[str, object]], config_raw: Mapping[str, object], metric_ids: tuple[str, ...]):
    status=[]; ra=[]; rr=[]; na=[]; nr=[]; pa=[]; pr=[]; identity={"normalization_none": "not_run", "tolerance_2": "not_run"}
    for component, ids in _conditions(config_raw):
        for index, condition in enumerate(ids):
            fresh, reason = _rows_from_records(records, surface=component, condition_index=index, metric_ids=metric_ids, protocol_id=str(panel["protocol_id"]))
            if fresh is None:
                st, al, rk = _closed(panel, component, condition, metric_ids, reason)
            else:
                reference_id = "native" if component == "resampling" else ("none" if component == "normalization" else "tolerance:2")
                result = assemble_phase4_sensitivity_rows(parent_rows, fresh, panel_id=str(panel["panel_id"]), endpoint_id=str(panel["endpoint_id"]), protocol_id=str(panel["protocol_id"]), condition_id=condition, metric_ids=metric_ids, reference_by_metric=parent_alignment, reference_condition_id=reference_id, require_identity=(condition in {"none", "tolerance:2"}))
                st, al, rk = MappingProxyType({**result.status_rows[0], "component": component}), result.alignment_rows, result.rank_rows
                if condition == "none": identity["normalization_none"] = "passed"
                if condition == "tolerance:2": identity["tolerance_2"] = "passed"
            status.append(st)
            if component == "resampling": ra.extend(al); rr.extend(rk)
            elif component == "normalization": na.extend(al); nr.extend(rk)
            else: pa.extend(al); pr.extend(rk)
    return status, ra, rr, na, nr, pa, pr, MappingProxyType(identity)


def build_sugar_phase4_sensitivity(config_raw: Mapping[str, object], project_root: Path | None, worker_count: int, *, _synthetic: Mapping[str, object] | None = None) -> DatasetSensitivityResult:
    """Rematerialize D4 metric views only; never fit or invoke downstream models."""
    if worker_count < 1: raise AppendixAuditsError("sugar adapter", "worker count must be positive")
    metric_ids = tuple(str(value) for value in config_raw.get("metric_ids", ()))
    if metric_ids != METRIC_IDS: raise AppendixAuditsError("sugar adapter", "fixed metric order mismatch")
    root = Path(project_root) if project_root is not None else ROOT
    panels = _panels(config_raw)
    if _synthetic is not None:
        records = tuple(_synthetic["records"]); parents = _synthetic["parent_rows"]; alignments = _synthetic["parent_alignment"]
    else:
        cohort = load_d4_sugar_cohort(root / _D4_PROTOCOL, root / _D4_ARCHIVE)
        if len(cohort.record_ids) != 7680 or len(set(cohort.well_ids)) != 240: raise AppendixAuditsError("sugar adapter", "D4 cohort identity mismatch")
        jobs = tuple((index, str(record_id), str(well_id), np.asarray(cohort.wavenumber, dtype="<f8"), np.asarray(intensity, dtype="<f8")) for index, (record_id, well_id, intensity) in enumerate(zip(cohort.record_ids, cohort.well_ids, cohort.intensity, strict=True)))
        if worker_count == 1:
            _worker_init(str(root)); results = tuple(_evaluate_job(job) for job in jobs)
        else:
            with ProcessPoolExecutor(max_workers=min(worker_count, len(jobs)), initializer=_worker_init, initargs=(str(root),)) as executor: results = tuple(executor.map(_evaluate_job, jobs))
        records = tuple((well_id, unpack_record_sensitivity_values(payload)) for _, well_id, payload in results)
        parents = {str(panel["panel_id"]): _read_jsonl(root / str(panel["parent_path"]) / "well_observations.jsonl") for panel in panels}
        alignments = {str(panel["panel_id"]): {str(row["metric_output_id"]): (float(row["ag"]), float(row["acc_cross"])) for row in _read_jsonl(root / str(panel["parent_path"]) / "alignment_results.jsonl")} for panel in panels}
    audit=[]; ra=[]; rr=[]; na=[]; nr=[]; pa=[]; pr=[]; identity={}
    for panel in panels:
        panel_id=str(panel["panel_id"]); parent = parents[panel_id] if isinstance(parents, Mapping) else tuple(row for row in parents if str(row.get("panel_id")) == panel_id)
        alignment = alignments[panel_id] if isinstance(alignments, Mapping) else {}
        if set(alignment) != set(metric_ids): raise AppendixAuditsError("sugar adapter", f"{panel_id} parent alignment grid invalid")
        pieces = _aggregate(panel, parent, alignment, records, config_raw, metric_ids)
        audit.extend(pieces[0]); ra.extend(pieces[1]); rr.extend(pieces[2]); na.extend(pieces[3]); nr.extend(pieces[4]); pa.extend(pieces[5]); pr.extend(pieces[6]); identity[panel_id]=pieces[7]
    return DatasetSensitivityResult(tuple(audit), tuple(ra), tuple(rr), tuple(na), tuple(nr), tuple(pa), tuple(pr), MappingProxyType(identity))
