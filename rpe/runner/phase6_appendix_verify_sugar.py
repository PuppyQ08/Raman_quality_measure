"""Independent D4 Sugar sensitivity adapter used only by the Step-7 verifier."""
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

from rpe.alignment import AlignmentObservation, alignment_gap, cross_perturbation_accuracy
from rpe.downstream.sugar_quantitative import load_d4_sugar_cohort
from rpe.evaluation import Spectrum1D
from rpe.methods.catalog import load_classical_catalog
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase4_d4_eligibility import _phase1_source_for_spectrum, load_phase4_d4_eligibility_config
from rpe.runner.phase6_appendix_verifier_science import (
    v_METRIC_IDS, v_evaluate_record_sensitivity, v_parent_downstream_collapse,
    v_rank_stability, v_rematerialize_p8_p12_conditions,
    pack_verifier_record_sensitivity_result, unpack_verifier_record_sensitivity_result,
)


_PANELS = ("d4_a", "d4_b")
_COMPONENTS = ("resampling", "normalization", "phase4_peak_tolerance")
_PEAK_METRICS = v_METRIC_IDS[8:]
_V_EXPECTED_COUNTS = {
    "resampling": (18, 234, 36),
    "normalization": (8, 104, 16),
    "phase4_peak_tolerance": (8, 104, 16),
}
_D4_PROTOCOL = "experiments/phase05/configs/d4_sugar_protocol.json"
_D4_ARCHIVE = "data/raw/ramanbench/cache/10779223/Raw data.zip"
_ELIGIBILITY = "experiments/phase4/configs/d4_protocol_a_full_domain_eligibility_v1.json"
_PHASE1 = "experiments/phase1/configs/rruff_raw_core10k_v1.json"
_SWEEP = "experiments/shared/raman_perturbation_sweep_v1.json"
_CATALOG = "experiments/phase3/configs/classical_system_catalog_v1.json"


class VerifySugarError(ValueError):
    pass


@dataclass(frozen=True)
class VerifierSugarSensitivityResult:
    audit_status: tuple[Mapping[str, object], ...]
    resampling_alignment: tuple[Mapping[str, object], ...]
    resampling_rank_stability: tuple[Mapping[str, object], ...]
    normalization_alignment: tuple[Mapping[str, object], ...]
    normalization_rank_stability: tuple[Mapping[str, object], ...]
    peak_tolerance_phase4_alignment: tuple[Mapping[str, object], ...]
    peak_tolerance_phase4_rank_stability: tuple[Mapping[str, object], ...]
    identity: Mapping[str, object]


def _v_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        with path.open(encoding="utf-8") as stream:
            return tuple(json.loads(line) for line in stream if line.strip())
    except (OSError, json.JSONDecodeError) as error:
        raise VerifySugarError(f"parent artifact unreadable: {path}") from error


def _v_conditions(raw: Mapping[str, object]) -> Mapping[str, tuple[str, ...]]:
    resampling = raw.get("resampling_conditions")
    if not isinstance(resampling, Mapping):
        raise VerifySugarError("resampling conditions invalid")
    values = MappingProxyType({
        "resampling": tuple(f"{kind}:{float(spacing):g}" for kind in ("linear", "cubic", "pchip") for spacing in resampling.get("spacing_cm1", ())),
        "normalization": tuple(str(value) for value in raw.get("normalization_conditions", ())),
        "phase4_peak_tolerance": tuple(f"tolerance:{value}" for value in (1, 2, 4, 8)),
    })
    if tuple(map(len, values.values())) != (9, 4, 4):
        raise VerifySugarError("D4 condition grid drift")
    return values


def _v_meta(raw: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    manifest = raw.get("panel_manifest")
    if not isinstance(manifest, Sequence):
        raise VerifySugarError("panel manifest invalid")
    rows = {str(row.get("panel_id")): row for row in manifest if isinstance(row, Mapping)}
    if tuple(item for item in raw.get("fixed_panel_order", ()) if item in _PANELS) != _PANELS or set(_PANELS) - set(rows):
        raise VerifySugarError("D4 panel manifest drift")
    return MappingProxyType(rows)


def _v_downstream(rows: Sequence[Mapping[str, object]]) -> Mapping[tuple[str, str, float], float]:
    try:
        return v_parent_downstream_collapse(rows)
    except ValueError as error:
        raise VerifySugarError(f"parent downstream grid invalid: {error}") from error


def _v_parents(root: Path, meta: Mapping[str, Mapping[str, object]]) -> tuple[Mapping[str, tuple[Mapping[str, object], ...]], Mapping[str, Mapping[str, tuple[float, float]]]]:
    observations, references = {}, {}
    for panel in _PANELS:
        base = root / str(meta[panel]["parent_path"])
        rows, alignment = _v_jsonl(base / "well_observations.jsonl"), _v_jsonl(base / "alignment_results.jsonl")
        _v_downstream(rows)
        refs = {str(row["metric_output_id"]): (float(row["ag"]), float(row["acc_cross"])) for row in alignment if row.get("state") == "complete"}
        if tuple(refs) != v_METRIC_IDS:
            raise VerifySugarError(f"{panel}: parent alignment grid invalid")
        observations[panel], references[panel] = rows, MappingProxyType(refs)
    return MappingProxyType(observations), MappingProxyType(references)


def _v_status(component: str, meta: Mapping[str, object], condition: str, order: int, state: str, reason: str, numerical: int, ranks: int) -> Mapping[str, object]:
    return MappingProxyType({
        "component": component, "status_order": order, "panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"], "condition_id": condition, "state": state, "reason_code": reason, "numerical_row_count": numerical, "rank_row_count": ranks, "evidence_key": "native_metric_reconstruction" if not reason else "record_core_closure",
    })


def _v_closed(component: str, meta: Mapping[str, object], condition: str, reason: str) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    reference = "native" if component == "resampling" else ("none" if component == "normalization" else "tolerance:2")
    common = {"panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"], "condition_id": condition}
    alignment = tuple(MappingProxyType({**common, "metric_output_id": metric, "ag": None, "acc_cross": None, "state": "not_evaluable", "reason_code": reason}) for metric in v_METRIC_IDS)
    ranks = tuple(MappingProxyType({**common, "reference_condition_id": reference, "statistic": statistic, "tau_b": None, "reference_tie_count": None, "candidate_tie_count": None, "max_abs_rank_displacement": None, "stable": False, "state": "not_evaluable", "reason_code": reason}) for statistic in ("ag", "acc_cross"))
    return alignment, ranks


def _v_assemble(parent: Sequence[Mapping[str, object]], fresh_rows: Sequence[Mapping[str, object]], *, component: str, meta: Mapping[str, object], condition: str, references: Mapping[str, tuple[float, float]]) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    downstream = _v_downstream(parent)
    expected = {(well, perturbation, alpha, metric) for well, perturbation, alpha in downstream for metric in v_METRIC_IDS}
    fresh = {}
    for row in fresh_rows:
        key = (str(row["well_id"]), str(row["perturbation_id"]), float(row["alpha"]), str(row["metric_output_id"]))
        if key in fresh or key not in expected or not np.isfinite(float(row["metric_harm"])):
            raise VerifySugarError("fresh D4 metric grid invalid")
        fresh[key] = float(row["metric_harm"])
    if set(fresh) != expected:
        raise VerifySugarError("incomplete D4 well-condition-metric grid")
    ag, acc = {}, {}
    for metric in v_METRIC_IDS:
        observations = tuple(AlignmentObservation(well, perturbation, alpha, fresh[(well, perturbation, alpha, metric)], downstream[(well, perturbation, alpha)]) for well, perturbation, alpha in downstream)
        ag[metric], acc[metric] = alignment_gap(observations).alignment_gap, cross_perturbation_accuracy(observations).accuracy
    identity = (component == "normalization" and condition == "none") or (component == "phase4_peak_tolerance" and condition == "tolerance:2")
    if identity and any((ag[m], acc[m]) != references[m] for m in v_METRIC_IDS):
        raise VerifySugarError("normalization-none or tolerance-2 identity mismatch")
    reference_condition = "native" if component == "resampling" else ("none" if component == "normalization" else "tolerance:2")
    common = {"panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"], "condition_id": condition}
    alignment = tuple(MappingProxyType({**common, "metric_output_id": metric, "ag": ag[metric], "acc_cross": acc[metric], "state": "complete_numeric", "reason_code": ""}) for metric in v_METRIC_IDS)
    ranks = tuple(MappingProxyType({**common, "reference_condition_id": reference_condition, **v_rank_stability({m: references[m][index] for m in v_METRIC_IDS}, values, statistic=statistic, identities=v_METRIC_IDS)}) for index, (statistic, values) in enumerate((("ag", ag), ("acc_cross", acc))))
    return alignment, ranks


_WORKER: dict[str, object] = {}


def _v_worker_init(root: str) -> None:
    os.environ.update({name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")})
    project = Path(root)
    eligibility = load_phase4_d4_eligibility_config(project / _ELIGIBILITY)
    catalog = load_classical_catalog(project / _CATALOG)
    matched = tuple(system for system in catalog.systems if system.system_id == eligibility.cwt_system_id)
    if len(matched) != 1: raise VerifySugarError("CWT system did not resolve exactly once")
    _WORKER.update({"phase1": load_phase1_core_config(project / _PHASE1), "sweep": load_perturbation_sweep_config(project / _SWEEP), "cwt": matched[0], "p10": eligibility.p10_memory_budget_bytes})


def _v_worker(job: tuple[int, str, str, np.ndarray, np.ndarray]) -> tuple[str, object]:
    order, record_id, well_id, axis, intensity = job
    spectrum = Spectrum1D(f"d4_sugar_low_snr::{record_id}", well_id, np.asarray(axis, dtype="<f8"), np.asarray(intensity, dtype="<f8"))
    with threadpool_limits(limits=1, user_api="blas"):
        ids, spectra = v_rematerialize_p8_p12_conditions(_phase1_source_for_spectrum(spectrum, order=order), _WORKER["phase1"], _WORKER["sweep"], p10_memory_budget_bytes=int(_WORKER["p10"]))
        result = v_evaluate_record_sensitivity(spectrum, ids, spectra, record_id=record_id, support_start_cm1=146.0, support_stop_cm1=3684.0, support_point_counts={0.5: 7077, 1.0: 3539, 2.0: 1770}, support_max_gap_cm1=3.665, cwt_system=_WORKER["cwt"])
    return well_id, pack_verifier_record_sensitivity_result(result)


_pack_record_result = pack_verifier_record_sensitivity_result
_unpack_record_result = unpack_verifier_record_sensitivity_result


def _v_aggregate_harms(values: Sequence[float], *, protocol_id: str) -> float:
    if not values:
        raise VerifySugarError("empty acquisition harms")
    if protocol_id == "a":
        return float(np.mean(np.asarray(values, dtype=np.float64)))
    if protocol_id == "b":
        total = 0.0
        for value in values:
            total += float(value)
        return total / len(values)
    raise VerifySugarError("unknown protocol reduction")


def _v_metric_rows(values: Sequence[tuple[str, object]], *, component: str, condition: str, parent: Sequence[Mapping[str, object]], protocol_id: str) -> tuple[tuple[Mapping[str, object], ...], str | None]:
    surface_index = {"resampling": 0, "normalization": 1, "phase4_peak_tolerance": 2}[component]
    prefix = f"{component}:{condition}" if component != "phase4_peak_tolerance" else "peak_tolerance"
    for _, result in values:
        if any(key == prefix or key.startswith(prefix + ":") for key in result.closure_reasons): return (), "not_evaluable_metric_core_closure"
    means: dict[tuple[str, str, float, str], list[float]] = {}
    for well, result in values:
        surface = (result.resampling_harms, result.normalization_harms, result.peak_tolerance_harms)[surface_index]
        ids = (result.resampling_ids, result.normalization_ids, tuple(f"tolerance:{x}" for x in result.tolerance_ids))[surface_index]
        if condition not in ids: return (), "not_evaluable_surface_condition_drift"
        array, metrics = surface[ids.index(condition)], (_PEAK_METRICS if component == "phase4_peak_tolerance" else v_METRIC_IDS)
        for ci, cell in enumerate(result.condition_ids):
            perturbation, encoded = cell.split(":", 1); alpha = float(np.frombuffer(bytes.fromhex(encoded), dtype="<f8")[0])
            for mi, metric in enumerate(metrics):
                value = float(array[ci, mi])
                if not np.isfinite(value): return (), "not_evaluable_incomplete_metric_set"
                means.setdefault((well, perturbation, alpha, metric), []).append(value)
    rows = []
    for source in parent:
        metric = str(source["metric_output_id"]); key = (str(source["well_id"]), str(source["perturbation_id"]), float(source["alpha"]), metric)
        if component == "phase4_peak_tolerance" and metric not in _PEAK_METRICS: harm = float(source["metric_harm"])
        else:
            values_for_key = means.get(key, ())
            if len(values_for_key) != 32: return (), "not_evaluable_incomplete_well_acquisition_grid"
            harm = _v_aggregate_harms(values_for_key, protocol_id=protocol_id)
        rows.append({"well_id": key[0], "perturbation_id": key[1], "alpha": key[2], "metric_output_id": metric, "metric_harm": harm})
    return tuple(rows), None


def v_build_sugar_sensitivity(config_raw: Mapping[str, object], project_root: Path, worker_count: int) -> VerifierSugarSensitivityResult:
    """Rebuild D4 A/B metric sensitivity from 7,680 raw records; never fit PLS."""
    if not isinstance(config_raw, Mapping) or not isinstance(worker_count, int) or worker_count < 1 or tuple(config_raw.get("metric_ids", ())) != v_METRIC_IDS: raise VerifySugarError("invalid config, worker count, or metric manifest")
    root, meta, conditions = Path(project_root), _v_meta(config_raw), _v_conditions(config_raw)
    parents, references = _v_parents(root, meta)
    cohort = load_d4_sugar_cohort(root / _D4_PROTOCOL, root / _D4_ARCHIVE)
    if len(cohort.record_ids) != 7680 or len(set(cohort.well_ids)) != 240 or any(sum(w == well for w in cohort.well_ids) != 32 for well in set(cohort.well_ids)): raise VerifySugarError("D4 cohort identity/count drift")
    jobs = tuple((order, str(record), str(well), np.asarray(cohort.wavenumber, dtype="<f8"), np.asarray(intensity, dtype="<f8")) for order, (record, well, intensity) in enumerate(zip(cohort.record_ids, cohort.well_ids, cohort.intensity, strict=True)))
    if worker_count == 1:
        _v_worker_init(str(root)); values = tuple(_v_worker(job) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(jobs)), initializer=_v_worker_init, initargs=(str(root),)) as executor: values = tuple(executor.map(_v_worker, jobs))
    values = tuple((well_id, unpack_verifier_record_sensitivity_result(payload)) for well_id, payload in values)
    statuses, tables, ranks, order = [], {component: [] for component in _COMPONENTS}, {component: [] for component in _COMPONENTS}, 0
    for component in _COMPONENTS:
        for panel in _PANELS:
            for condition in conditions[component]:
                order += 1; metric_rows, reason = _v_metric_rows(values, component=component, condition=condition, parent=parents[panel], protocol_id=str(meta[panel]["protocol_id"]))
                if reason is None:
                    alignment, rank_rows = _v_assemble(parents[panel], metric_rows, component=component, meta=meta[panel], condition=condition, references=references[panel]); state = "complete_numeric"
                else:
                    alignment, rank_rows = _v_closed(component, meta[panel], condition, reason); state = "not_evaluable"
                statuses.append(_v_status(component, meta[panel], condition, order, state, reason or "", len(alignment) if reason is None else 0, len(rank_rows)))
                tables[component].extend(alignment); ranks[component].extend(rank_rows)
    for component, expected in _V_EXPECTED_COUNTS.items():
        actual = (sum(row["component"] == component for row in statuses), len(tables[component]), len(ranks[component]))
        if actual != expected: raise VerifySugarError(f"{component}: count drift {actual}")
    identity = MappingProxyType({"record_count": 7680, "well_count": 240, "records_per_well": 32, "detector_call_count": sum(int(result.identity["detector_call_count"]) for _, result in values)})
    return VerifierSugarSensitivityResult(tuple(statuses), tuple(tables["resampling"]), tuple(ranks["resampling"]), tuple(tables["normalization"]), tuple(ranks["normalization"]), tuple(tables["phase4_peak_tolerance"]), tuple(ranks["phase4_peak_tolerance"]), identity)


__all__ = ["VerifierSugarSensitivityResult", "VerifySugarError", "v_build_sugar_sensitivity"]
