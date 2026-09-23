"""Independent D5/RRUFF verifier adapter for the Step-7 appendix audit.

This module intentionally reconstructs its aggregation without importing a
production Step-7 adapter, metric core, or artifact builder.  It returns only
aggregate Phase-4 rows and count receipts; native and transformed spectra never
leave the evaluation workers.
"""
from __future__ import annotations

import json
import math
import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.alignment import AlignmentObservation, alignment_gap, cross_perturbation_accuracy
from rpe.downstream.rruff import load_d5_native_spectra, load_d5_raw_cohort
from rpe.methods.catalog import load_classical_catalog
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase4_d5_protocol_a import _phase1_source, _resolve_cwt
from rpe.runner.phase6_appendix_verifier_science import (
    v_METRIC_IDS,
    v_evaluate_record_sensitivity,
    v_parent_downstream_collapse,
    v_rank_stability,
    v_rematerialize_p8_p12_conditions,
    pack_verifier_record_sensitivity_result,
    unpack_verifier_record_sensitivity_result,
)


_PANELS = ("d5_a", "d5_b")
_PEAK_METRICS = v_METRIC_IDS[8:]
_P10_BUDGET = 64 * 2**30
_PROCESS_STATE: dict[str, object] = {}


@dataclass(frozen=True)
class VerifierRruffSensitivityResult:
    audit_status: tuple[Mapping[str, object], ...]
    resampling_alignment: tuple[Mapping[str, object], ...]
    resampling_rank_stability: tuple[Mapping[str, object], ...]
    normalization_alignment: tuple[Mapping[str, object], ...]
    normalization_rank_stability: tuple[Mapping[str, object], ...]
    peak_tolerance_phase4_alignment: tuple[Mapping[str, object], ...]
    peak_tolerance_phase4_rank_stability: tuple[Mapping[str, object], ...]
    identity: Mapping[str, object]


def _v_rruff_worker_payload(payload: tuple[int, int, str, int, str, object, object, object, object, Mapping[float, int], Mapping[str, object]]) -> tuple[int, object]:
    order, index, record_id, class_label, mineral_name, spectrum, phase1, sweep, cwt, points, support = payload
    with threadpool_limits(limits=1, user_api="blas"):
        source = _phase1_source(order, record_id, class_label, mineral_name, spectrum)
        identifiers, conditions = v_rematerialize_p8_p12_conditions(source, phase1, sweep, _P10_BUDGET)
        return index, v_evaluate_record_sensitivity(spectrum, identifiers, conditions, record_id=record_id, support_start_cm1=float(support["start_cm1"]), support_stop_cm1=float(support["stop_cm1"]), support_point_counts=points, support_max_gap_cm1=3.0, cwt_system=cwt)


def _init_process_worker(root_text: str) -> None:
    root = Path(root_text)
    _PROCESS_STATE.clear()
    _PROCESS_STATE.update({
        "phase1": load_phase1_core_config(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json"),
        "sweep": load_perturbation_sweep_config(root / "experiments/shared/raman_perturbation_sweep_v1.json"),
        "cwt": _resolve_cwt(load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json")),
        "points": {0.5: 3193, 1.0: 1597, 2.0: 799},
        "support": {"start_cm1": 204.0, "stop_cm1": 1800.0},
    })


def _process_job(job: tuple[int, int, str, int, str, object]) -> tuple[int, object]:
    if not _PROCESS_STATE:
        raise RuntimeError("RRUFF verifier process worker was not initialized")
    order, index, record_id, class_label, mineral_name, spectrum = job
    index, value = _v_rruff_worker_payload((order, index, record_id, class_label, mineral_name, spectrum, _PROCESS_STATE["phase1"], _PROCESS_STATE["sweep"], _PROCESS_STATE["cwt"], _PROCESS_STATE["points"], _PROCESS_STATE["support"]))
    return index, pack_verifier_record_sensitivity_result(value)


_pack_record_result = pack_verifier_record_sensitivity_result
_unpack_record_result = unpack_verifier_record_sensitivity_result


def _v_panel_rows(config_raw: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = {str(row.get("panel_id")): row for row in config_raw.get("panel_manifest", ()) if isinstance(row, Mapping) and str(row.get("panel_id")) in _PANELS}
    if set(rows) != set(_PANELS):
        raise ValueError("RRUFF verifier: D5 panel manifest is incomplete")
    return rows


def _v_closed_rows(panel: Mapping[str, object], component: str, condition: str, reference: str, reason: str) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    base = {"panel_id": str(panel["panel_id"]), "endpoint_id": str(panel["endpoint_id"]), "protocol_id": str(panel["protocol_id"]), "condition_id": condition}
    alignment = tuple(MappingProxyType({**base, "metric_output_id": metric, "ag": None, "acc_cross": None, "state": "not_evaluable", "reason_code": reason}) for metric in v_METRIC_IDS)
    ranks = tuple(MappingProxyType({**base, "statistic": statistic, "reference_condition_id": reference, "tau_b": None, "reference_tie_count": None, "candidate_tie_count": None, "max_abs_rank_displacement": None, "stable": False, "state": "not_evaluable", "reason_code": reason}) for statistic in ("ag", "acc_cross"))
    return alignment, ranks


def _v_status(panel: Mapping[str, object], component: str, condition: str, state: str, reason: str, numerical: int, ranks: int) -> Mapping[str, object]:
    return MappingProxyType({"component": component, "status_order": 0, "panel_id": str(panel["panel_id"]), "endpoint_id": str(panel["endpoint_id"]), "protocol_id": str(panel["protocol_id"]), "condition_id": condition, "state": state, "reason_code": reason, "numerical_row_count": numerical, "rank_row_count": ranks, "evidence_key": "rruff_verifier_native_reconstruction"})


def _v_read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    with path.open(encoding="utf-8") as stream:
        return tuple(MappingProxyType(json.loads(line)) for line in stream if line.strip())


def _v_parent_authority(root: Path, panel: Mapping[str, object]) -> tuple[tuple[Mapping[str, object], ...], Mapping[str, tuple[float, float]]]:
    parent_root = root / str(panel["parent_path"])
    parent = _v_read_jsonl(parent_root / "class_observations.jsonl")
    alignment = _v_read_jsonl(parent_root / "alignment_results.jsonl")
    reference = {str(row["metric_output_id"]): (float(row["ag"]), float(row["acc"])) for row in alignment if row.get("metric_state") == "complete"}
    if len(parent) != 354_120 or set(reference) != set(v_METRIC_IDS):
        raise ValueError(f"{panel['panel_id']}: parent authority is incomplete")
    v_parent_downstream_collapse(parent)
    return parent, MappingProxyType(reference)


def _v_aggregate_rows(records: Mapping[int, object], occurrence_indices: Sequence[int], class_labels: np.ndarray, condition_ids: Sequence[str], *, field: str, surface_index: int, peak_only: bool) -> tuple[Mapping[str, object], ...]:
    metrics = _PEAK_METRICS if peak_only else v_METRIC_IDS
    conditions = tuple(
        (name, struct.unpack("<d", bytes.fromhex(alpha_hex))[0])
        for name, alpha_hex in (str(identifier).split(":", 1) for identifier in condition_ids)
    )
    if len(conditions) != 40 or len(set(conditions)) != 40:
        raise ValueError("not_evaluable_incomplete_record_condition_grid")
    occurrence_indices = tuple(int(index) for index in occurrence_indices)
    grouped: dict[tuple[str, str, float], list[np.ndarray]] = defaultdict(list)
    for index in occurrence_indices:
        values = getattr(records[index], field)[surface_index]
        if values.shape != (40, len(metrics)) or not np.isfinite(values).all():
            raise ValueError("not_evaluable_incomplete_record_condition_grid")
        cluster = str(int(class_labels[index]))
        for condition_index, (perturbation, alpha) in enumerate(conditions):
            grouped[(cluster, perturbation, alpha)].append(values[condition_index])
    expected = {(str(int(class_labels[index])), perturbation, alpha) for index in occurrence_indices for perturbation, alpha in conditions}
    if set(grouped) != expected:
        raise ValueError("not_evaluable_incomplete_record_condition_grid")
    rows = []
    for cluster, perturbation, alpha in sorted(grouped, key=lambda item: (int(item[0]), item[1], item[2])):
        values = grouped[(cluster, perturbation, alpha)]
        rows.extend(MappingProxyType({"cluster_id": cluster, "perturbation_id": perturbation, "alpha": alpha, "metric_output_id": metric, "metric_harm": float(np.mean([float(value[position]) for value in values])), "state": "complete", "occurrence_count": len(values)}) for position, metric in enumerate(metrics))
    return tuple(rows)


def _v_complete_rows(parent: Sequence[Mapping[str, object]], reference: Mapping[str, tuple[float, float]], panel: Mapping[str, object], condition: str, rows: Sequence[Mapping[str, object]], *, peak_only: bool, require_identity: bool, reference_condition: str) -> tuple[Mapping[str, object], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    source = (
        [row for row in parent if str(row["metric_output_id"]) not in _PEAK_METRICS] + list(rows)
        if peak_only else list(rows)
    )
    downstream = v_parent_downstream_collapse(parent)
    observed: dict[str, list[AlignmentObservation]] = {metric: [] for metric in v_METRIC_IDS}
    seen = set()
    for row in source:
        key = (str(row["cluster_id"]), str(row["perturbation_id"]), float(row["alpha"]), str(row["metric_output_id"]))
        if key in seen or key[3] not in observed or not math.isfinite(float(row["metric_harm"])):
            raise ValueError("reconstructed metric rows are invalid")
        seen.add(key)
        observed[key[3]].append(AlignmentObservation(*key[:3], float(row["metric_harm"]), downstream[key[:3]]))
    expected = {(cluster, perturbation, alpha, metric) for cluster, perturbation, alpha in downstream for metric in v_METRIC_IDS}
    if seen != expected:
        raise ValueError("reconstructed metric rows have incomplete fixed grid")
    ag = {metric: alignment_gap(observed[metric]).alignment_gap for metric in v_METRIC_IDS}
    acc = {metric: cross_perturbation_accuracy(observed[metric]).accuracy for metric in v_METRIC_IDS}
    if require_identity and any(ag[m] != reference[m][0] or acc[m] != reference[m][1] for m in v_METRIC_IDS):
        raise ValueError("RRUFF verifier identity AG/Acc-cross mismatch")
    base = {"panel_id": str(panel["panel_id"]), "endpoint_id": str(panel["endpoint_id"]), "protocol_id": str(panel["protocol_id"]), "condition_id": condition}
    alignment = tuple(MappingProxyType({**base, "metric_output_id": metric, "ag": ag[metric], "acc_cross": acc[metric], "state": "complete_numeric", "reason_code": ""}) for metric in v_METRIC_IDS)
    ranks = tuple(MappingProxyType({**base, "reference_condition_id": reference_condition, **v_rank_stability({m: reference[m][position] for m in v_METRIC_IDS}, values, statistic=statistic, identities=v_METRIC_IDS)}) for position, (statistic, values) in enumerate((("ag", ag), ("acc_cross", acc))))
    return _v_status(panel, "", condition, "complete_numeric", "", 13, 2), alignment, ranks


def _v_synthetic_result(config_raw: Mapping[str, object]) -> VerifierRruffSensitivityResult:
    panels = {panel: {"panel_id": panel, "endpoint_id": "d5", "protocol_id": panel[-1]} for panel in _PANELS}
    surfaces = (("resampling", ("linear:0.5", "linear:1", "linear:2", "cubic:0.5", "cubic:1", "cubic:2", "pchip:0.5", "pchip:1", "pchip:2"), "native"), ("normalization", ("none", "maximum", "area", "snv"), "none"), ("phase4_peak_tolerance", ("tolerance:1", "tolerance:2", "tolerance:4", "tolerance:8"), "tolerance:2"))
    outputs = {name: [] for name in ("status", "ra", "rr", "na", "nr", "pa", "pr")}
    for component, conditions, reference in surfaces:
        for panel in _PANELS:
            for condition in conditions:
                alignment, ranks = _v_closed_rows(panels[panel], component, condition, reference, "adapter_local_synthetic_closure")
                outputs["status"].append(_v_status(panels[panel], component, condition, "not_evaluable", "adapter_local_synthetic_closure", 13, 2))
                keys = ("ra", "rr") if component == "resampling" else ("na", "nr") if component == "normalization" else ("pa", "pr")
                outputs[keys[0]].extend(alignment); outputs[keys[1]].extend(ranks)
    return VerifierRruffSensitivityResult(*(tuple(outputs[key]) for key in ("status", "ra", "rr", "na", "nr", "pa", "pr")), MappingProxyType({"adapter": "rruff_verifier", "cohort_record_count": 0, "query_union_record_count": 0, "query_occurrence_count": 0, "mode": "synthetic"}))


def v_build_rruff_sensitivity(config_raw: Mapping[str, object], project_root: Path | str, worker_count: int, *, _testing_rows: Mapping[str, object] | None = None) -> VerifierRruffSensitivityResult:
    """Independently rebuild D5 sensitivity aggregate rows without matching."""
    if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count < 1:
        raise ValueError("worker_count must be a positive integer")
    if _testing_rows is not None:
        return _v_synthetic_result(config_raw)
    root, panels = Path(project_root), _v_panel_rows(config_raw)
    cohort = load_d5_raw_cohort(root / "experiments/phase05/configs/d5_rruff_protocol.json", root / "data/unified/rruff_raman_raw")
    occurrence_indices = tuple(int(index) for split in cohort.splits for index in split.query_indices)
    occurrences: dict[int, int] = defaultdict(int)
    for index in occurrence_indices:
        occurrences[index] += 1
    query_indices = tuple(sorted(occurrences))
    if len(cohort.record_ids) != 3770 or len(query_indices) != 3012 or sum(occurrences.values()) != 6621 or len(set(int(cohort.class_labels[index]) for index in query_indices)) != 681:
        raise ValueError("frozen D5 3770/3012/6621/681 occurrence identity gate failed")
    spectra = load_d5_native_spectra(root / "data/unified/rruff_raman_raw", tuple(cohort.record_ids[index] for index in query_indices))
    phase1 = load_phase1_core_config(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json")
    sweep = load_perturbation_sweep_config(root / "experiments/shared/raman_perturbation_sweep_v1.json")
    cwt = _resolve_cwt(load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json"))
    support = config_raw["supports"]["d5_raw_rruff"]
    points = {float(key): int(value) for key, value in support["metric_view_point_counts_by_spacing_cm1"].items()}

    jobs = tuple((order, index, cohort.record_ids[index], int(cohort.class_labels[index]), cohort.mineral_names[index], spectra[order]) for order, index in enumerate(query_indices))

    records: dict[int, object] = {}
    with threadpool_limits(limits=1, user_api="blas"):
        if worker_count == 1:
            completed = (_v_rruff_worker_payload((*job, phase1, sweep, cwt, points, support)) for job in jobs)
        else:
            executor = ProcessPoolExecutor(max_workers=min(worker_count, len(jobs)), initializer=_init_process_worker, initargs=(str(root),))
            completed = executor.map(_process_job, jobs)
        try:
            for index, values in completed:
                records[index] = values if worker_count == 1 else unpack_verifier_record_sensitivity_result(values)
        finally:
            if worker_count != 1:
                executor.shutdown()
    prototype = records[query_indices[0]]
    if any(values.condition_ids != prototype.condition_ids for values in records.values()):
        raise ValueError("RRUFF verifier condition order drift")
    authorities = {key: _v_parent_authority(root, panel) for key, panel in panels.items()}
    status: list[Mapping[str, object]] = []; ra: list[Mapping[str, object]] = []; rr: list[Mapping[str, object]] = []; na: list[Mapping[str, object]] = []; nr: list[Mapping[str, object]] = []; pa: list[Mapping[str, object]] = []; pr: list[Mapping[str, object]] = []
    for component, field, conditions, reference_condition, peak_only in (("resampling", "resampling_harms", prototype.resampling_ids, "native", False), ("normalization", "normalization_harms", prototype.normalization_ids, "none", False), ("phase4_peak_tolerance", "peak_tolerance_harms", tuple(f"tolerance:{item}" for item in prototype.tolerance_ids), "tolerance:2", True)):
        for surface_index, condition in enumerate(conditions):
            try:
                fresh = _v_aggregate_rows(records, occurrence_indices, cohort.class_labels, prototype.condition_ids, field=field, surface_index=surface_index, peak_only=peak_only)
            except ValueError as error:
                fresh, closure = None, str(error)
            for key in _PANELS:
                panel, (parent, reference) = panels[key], authorities[key]
                if fresh is None:
                    alignment, ranks = _v_closed_rows(panel, component, condition, reference_condition, closure)
                    current = _v_status(panel, component, condition, "not_evaluable", closure, 13, 2)
                else:
                    try:
                        current, alignment, ranks = _v_complete_rows(parent, reference, panel, condition, fresh, peak_only=peak_only, require_identity=(component == "normalization" and condition == "none") or (component == "phase4_peak_tolerance" and condition == "tolerance:2"), reference_condition=reference_condition)
                        current = MappingProxyType({**current, "component": component})
                    except Exception:
                        if (component == "normalization" and condition == "none") or (component == "phase4_peak_tolerance" and condition == "tolerance:2"):
                            raise
                        alignment, ranks = _v_closed_rows(panel, component, condition, reference_condition, "not_evaluable_incomplete_record_condition_grid")
                        current = _v_status(panel, component, condition, "not_evaluable", "not_evaluable_incomplete_record_condition_grid", 13, 2)
                status.append(current)
                target = (ra, rr) if component == "resampling" else (na, nr) if component == "normalization" else (pa, pr)
                target[0].extend(alignment); target[1].extend(ranks)
    if tuple(map(len, (status, ra, rr, na, nr, pa, pr))) != (34, 234, 36, 104, 16, 104, 16):
        raise ValueError("RRUFF verifier fixed row-count gate failed")
    identity = MappingProxyType({"adapter": "rruff_verifier", "cohort_record_count": len(cohort.record_ids), "query_union_record_count": len(query_indices), "query_occurrence_count": sum(occurrences.values()), "class_count": len(set(int(cohort.class_labels[index]) for index in query_indices))})
    return VerifierRruffSensitivityResult(tuple(status), tuple(ra), tuple(rr), tuple(na), tuple(nr), tuple(pa), tuple(pr), identity)


__all__ = ["VerifierRruffSensitivityResult", "v_build_rruff_sensitivity"]
