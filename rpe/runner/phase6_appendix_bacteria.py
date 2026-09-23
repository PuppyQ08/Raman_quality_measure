"""Bacteria-ID aggregate adapter for the Step-7 Phase-4 sensitivity audit."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.methods.catalog import load_classical_catalog
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase4_d2_protocol_a import (
    _phase1_source, load_phase4_d2_protocol_a_config, reconstruct_d2_protocol_a_inputs,
)
from rpe.runner.phase6_appendix_audits import AppendixAuditsError, assemble_phase4_sensitivity_rows
from rpe.runner.phase6_appendix_metric_core import (
    METRIC_IDS, PEAK_METRIC_IDS, evaluate_record_sensitivity, pack_record_sensitivity_values,
    rematerialize_p8_p12_conditions, unpack_record_sensitivity_values,
)

_PANELS = ("d2_5_a", "d2_5_b", "d2_10_a", "d2_10_b", "d2_20_a", "d2_20_b", "d1_a", "d1_b_closed")
_COMPONENTS = ("resampling", "normalization", "phase4_peak_tolerance")
_D1B_REASON = "not_evaluable_failed_alpha0_equivalence"
_CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
_P10_BUDGET = 64 * 1024**3
_PROCESS_STATE: dict[str, object] = {}


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


def _panel_meta(raw: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = raw.get("panel_manifest")
    if not isinstance(rows, Sequence):
        raise AppendixAuditsError("bacteria adapter", "panel_manifest is required")
    found = {str(row["panel_id"]): row for row in rows if isinstance(row, Mapping)}
    if tuple(found) != tuple(str(x) for x in raw.get("fixed_panel_order", ())):
        raise AppendixAuditsError("bacteria adapter", "panel order drift")
    if any(panel not in found for panel in _PANELS):
        raise AppendixAuditsError("bacteria adapter", "Bacteria panel missing")
    return found


def _conditions(raw: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    r = raw.get("resampling_conditions", {})
    if not isinstance(r, Mapping): raise AppendixAuditsError("bacteria adapter", "resampling conditions invalid")
    return {
        "resampling": tuple(f"{name}:{float(spacing):g}" for name in ("linear", "cubic", "pchip") for spacing in r.get("spacing_cm1", ())),
        "normalization": tuple(str(x) for x in raw.get("normalization_conditions", ())),
        "phase4_peak_tolerance": tuple(f"tolerance:{x}" for x in (1, 2, 4, 8)),
    }


def _jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    with path.open(encoding="utf-8") as stream:
        return tuple(json.loads(line) for line in stream if line.strip())


def _status(component: str, meta: Mapping[str, object], condition: str, *, order: int, state: str, reason: str, numerical: int, ranks: int, evidence: str) -> Mapping[str, object]:
    return MappingProxyType({
        "component": component, "status_order": order, "panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"], "condition_id": condition,
        "state": state, "reason_code": reason, "numerical_row_count": numerical, "rank_row_count": ranks, "evidence_key": evidence,
    })


def _closed_rows(component: str, meta: Mapping[str, object], condition: str, metric_ids: Sequence[str], reason: str) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    alignment = tuple({"panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"], "condition_id": condition, "metric_output_id": metric, "ag": None, "acc_cross": None, "state": "not_evaluable", "reason_code": reason} for metric in metric_ids)
    rank = tuple({"panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"], "condition_id": condition, "reference_condition_id": "native" if component == "resampling" else ("none" if component == "normalization" else "tolerance:2"), "statistic": statistic, "tau_b": None, "reference_tie_count": None, "candidate_tie_count": None, "max_abs_rank_displacement": None, "stable": False, "state": "not_evaluable", "reason_code": reason} for statistic in ("ag", "acc_cross"))
    return alignment, rank


def _parent_artifacts(root: Path, meta: Mapping[str, Mapping[str, object]]) -> tuple[dict[str, tuple[Mapping[str, object], ...]], dict[str, Mapping[str, tuple[float, float]]]]:
    observations: dict[str, tuple[Mapping[str, object], ...]] = {}; references = {}
    for panel in _PANELS:
        if panel == "d1_b_closed": continue
        path = root / str(meta[panel]["parent_path"]); shot = int(panel.split("_")[1]) if panel.startswith("d2_") else None
        rows = _jsonl(path / "class_observations.jsonl")
        if shot is not None: rows = tuple(row for row in rows if int(row.get("shot_count", -1)) == shot)
        observations[panel] = rows
        aligned = _jsonl(path / "alignment_results.jsonl")
        if shot is not None: aligned = tuple(row for row in aligned if int(row.get("shot_count", -1)) == shot)
        refs = {str(row["metric_output_id"]): (float(row["ag"]), float(row["acc_cross"])) for row in aligned if row.get("state") == "complete"}
        if set(refs) != set(METRIC_IDS): raise AppendixAuditsError("bacteria adapter", f"{panel}: incomplete parent alignment")
        references[panel] = MappingProxyType(refs)
    return observations, references


def _worker(job: tuple[int, str, int, object], *, phase1: object, sweep: object, cwt: object, support: Mapping[str, object]):
    order, record_id, label, spectrum = job
    with threadpool_limits(limits=1, user_api="blas"):
        ids, spectra = rematerialize_p8_p12_conditions(_phase1_source(order, record_id, label, spectrum), phase1, sweep, p10_memory_budget_bytes=_P10_BUDGET)
        return label, evaluate_record_sensitivity(spectrum, ids, spectra, record_id=record_id, support_start_cm1=float(support["start_cm1"]), support_stop_cm1=float(support["stop_cm1"]), support_point_counts={float(k): int(v) for k, v in support["metric_view_point_counts_by_spacing_cm1"].items()}, support_max_gap_cm1=1.561, cwt_system=cwt)


def _init_process_worker(root_text: str) -> None:
    root = Path(root_text)
    catalog = load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json")
    matched = tuple(system for system in catalog.systems if system.system_id == _CWT_SYSTEM_ID)
    if len(matched) != 1:
        raise AppendixAuditsError("bacteria adapter", "CWT system did not resolve exactly once")
    _PROCESS_STATE.clear()
    _PROCESS_STATE.update({
        "phase1": load_phase1_core_config(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json"),
        "sweep": load_perturbation_sweep_config(root / "experiments/shared/raman_perturbation_sweep_v1.json"),
        "cwt": matched[0],
        "support": {"start_cm1": 387.0, "stop_cm1": 1791.0, "metric_view_point_counts_by_spacing_cm1": {"0.5": 2809, "1.0": 1405, "2.0": 703}},
    })


def _process_job(job: tuple[int, str, int, object]) -> tuple[int, object]:
    if not _PROCESS_STATE:
        raise AppendixAuditsError("bacteria adapter", "process worker was not initialized")
    label, value = _worker(job, phase1=_PROCESS_STATE["phase1"], sweep=_PROCESS_STATE["sweep"], cwt=_PROCESS_STATE["cwt"], support=_PROCESS_STATE["support"])
    return label, pack_record_sensitivity_values(value)


_pack_record_result = pack_record_sensitivity_values
_unpack_record_result = unpack_record_sensitivity_values


def _record_rows(values: Sequence[tuple[int, object]], *, component: str, condition: str, parent_rows: Sequence[Mapping[str, object]]) -> tuple[tuple[Mapping[str, object], ...], str | None]:
    index = {"resampling": 0, "normalization": 1, "phase4_peak_tolerance": 2}[component]
    for _, value in values:
        prefix = f"{component}:{condition}" if component != "phase4_peak_tolerance" else "peak_tolerance:"
        if any(key == prefix or key.startswith(prefix + ":") for key in value.closure_reasons):
            return (), f"not_evaluable_{next(key for key in value.closure_reasons if key == prefix or key.startswith(prefix + ':'))}"
    parent = {(str(row["class_label"]), str(row["perturbation_id"]), float(row["alpha"]), str(row["metric_output_id"])): row for row in parent_rows}
    if len(parent) != len(parent_rows): return (), "not_evaluable_parent_duplicate_grid"
    harms: dict[tuple[int, str, float, str], list[float]] = {}
    for label, value in values:
        surface = (value.resampling_harms, value.normalization_harms, value.peak_tolerance_harms)[index]
        surface_ids = (value.resampling_ids, value.normalization_ids, tuple(f"tolerance:{x}" for x in value.tolerance_ids))[index]
        if condition not in surface_ids: return (), "not_evaluable_surface_condition_drift"
        array = surface[surface_ids.index(condition)]
        for ci, cell in enumerate(value.condition_ids):
            perturbation, alpha_hex = cell.split(":", 1); alpha = float(np.frombuffer(bytes.fromhex(alpha_hex), dtype="<f8")[0])
            for mi, metric in enumerate(PEAK_METRIC_IDS if component == "phase4_peak_tolerance" else METRIC_IDS):
                number = float(array[ci, mi])
                if not np.isfinite(number): return (), "not_evaluable_incomplete_metric_set"
                harms.setdefault((label, perturbation, alpha, metric), []).append(number)
    rebuilt = []
    for key, row in parent.items():
        label, perturbation, alpha, metric = key
        if component == "phase4_peak_tolerance" and metric not in PEAK_METRIC_IDS:
            harm = float(row["metric_harm"])
        else:
            values_for_key = harms.get((int(label), perturbation, alpha, metric), ())
            if len(values_for_key) != 100: return (), "not_evaluable_incomplete_record_class_grid"
            harm = float(np.mean(values_for_key, dtype=np.float64))
        rebuilt.append({**row, "metric_harm": harm})
    return tuple(rebuilt), None


def _materialize(raw: Mapping[str, object], root: Path, worker_count: int) -> tuple[Sequence[tuple[int, object]], Mapping[str, object]]:
    d2_config = load_phase4_d2_protocol_a_config(root / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json")
    inputs = reconstruct_d2_protocol_a_inputs(root / "data/unified/bacteria_id_reference", root / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json", d2_config)
    catalog = load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json")
    matched = [system for system in catalog.systems if system.system_id == _CWT_SYSTEM_ID]
    if len(matched) != 1: raise AppendixAuditsError("bacteria adapter", "CWT system did not resolve exactly once")
    support = raw["supports"]["d1_d2_bacteria_id"]
    jobs = tuple((order, record_id, int(label), spectrum) for order, (record_id, label, spectrum) in enumerate(zip(inputs.record_ids, inputs.native_test_labels, inputs.native_test_spectra, strict=True)))
    if len(jobs) != 3000 or {label for _, _, label, _ in jobs} != set(range(30)): raise AppendixAuditsError("bacteria adapter", "shared D1/D2 test cohort drift")
    phase1 = load_phase1_core_config(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json"); sweep = load_perturbation_sweep_config(root / "experiments/shared/raman_perturbation_sweep_v1.json")
    # Worker construction stays top-level/picklable; deterministic map preserves input order.
    if worker_count == 1:
        values = tuple(_worker(job, phase1=phase1, sweep=sweep, cwt=matched[0], support=support) for job in jobs)
    else:
        # Bacteria spectra are record blocks.  Avoid nested closures in the process pool.
        with ProcessPoolExecutor(max_workers=min(worker_count, len(jobs)), initializer=_init_process_worker, initargs=(str(root),)) as executor:
            values = tuple((label, unpack_record_sensitivity_values(payload)) for label, payload in executor.map(_process_job, jobs))
    return values, MappingProxyType({"record_count": len(values), "class_count": 30, "detector_call_count": sum(int(value.identity["detector_call_count"]) for _, value in values)})


def build_bacteria_phase4_sensitivity(config_raw: Mapping[str, object], project_root: Path, worker_count: int) -> DatasetSensitivityResult:
    if not isinstance(config_raw, Mapping) or worker_count < 1: raise AppendixAuditsError("bacteria adapter", "invalid config or worker count")
    if tuple(config_raw.get("metric_ids", ())) != METRIC_IDS: raise AppendixAuditsError("bacteria adapter", "metric manifest drift")
    meta, conditions, root = _panel_meta(config_raw), _conditions(config_raw), Path(project_root)
    parent, references = _parent_artifacts(root, meta); values, identity = _materialize(config_raw, root, worker_count)
    statuses=[]; tables={name: [] for name in _COMPONENTS}; ranks={name: [] for name in _COMPONENTS}; order=0
    for component in _COMPONENTS:
        for panel in _PANELS:
            for condition in conditions[component]:
                order += 1; panel_meta = meta[panel]
                if panel == "d1_b_closed":
                    # D1-B is a status-only fixed closure: the frozen adapter
                    # counts reserve aggregate rows for the seven numerical panels.
                    statuses.append(_status(component, panel_meta, condition, order=order, state="not_evaluable", reason=_D1B_REASON, numerical=0, ranks=0, evidence="fixed_d1_b_closure")); continue
                metric_rows, reason = _record_rows(values, component=component, condition=condition, parent_rows=parent[panel])
                if reason:
                    alignment, rank = _closed_rows(component, panel_meta, condition, METRIC_IDS, reason); tables[component].extend(alignment); ranks[component].extend(rank)
                    statuses.append(_status(component, panel_meta, condition, order=order, state="not_evaluable", reason=reason, numerical=13, ranks=2, evidence="record_core_closure")); continue
                reference_id = "native" if component == "resampling" else ("none" if component == "normalization" else "tolerance:2")
                assembled = assemble_phase4_sensitivity_rows(parent[panel], metric_rows, panel_id=panel, endpoint_id=str(panel_meta["endpoint_id"]), protocol_id=str(panel_meta["protocol_id"]), condition_id=condition, metric_ids=METRIC_IDS, reference_by_metric=references[panel], reference_condition_id=reference_id, require_identity=(component == "normalization" and condition == "none") or (component == "phase4_peak_tolerance" and condition == "tolerance:2"))
                statuses.append(_status(component, panel_meta, condition, order=order, state="complete_numeric", reason="", numerical=13, ranks=2, evidence="native_metric_reconstruction")); tables[component].extend(assembled.alignment_rows); ranks[component].extend(assembled.rank_rows)
    expected = {"resampling": (72, 819, 126), "normalization": (32, 364, 56), "phase4_peak_tolerance": (32, 364, 56)}
    for component, (status_count, alignment_count, rank_count) in expected.items():
        actual = (sum(1 for row in statuses if row["component"] == component), len(tables[component]), len(ranks[component]))
        if actual != (status_count, alignment_count, rank_count): raise AppendixAuditsError("bacteria adapter", f"{component}: count drift {actual}")
    return DatasetSensitivityResult(tuple(statuses), tuple(tables["resampling"]), tuple(ranks["resampling"]), tuple(tables["normalization"]), tuple(ranks["normalization"]), tuple(tables["phase4_peak_tolerance"]), tuple(ranks["phase4_peak_tolerance"]), identity)


__all__ = ["DatasetSensitivityResult", "build_bacteria_phase4_sensitivity"]
