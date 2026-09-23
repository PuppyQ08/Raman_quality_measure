"""D5/RRUFF Phase-4 sensitivity adapter.

This module deliberately has no writer and never returns spectra.  The small
``_testing_rows`` seam exercises the aggregate contract without opening RRUFF.
"""
from __future__ import annotations

import json
import struct
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.downstream.rruff import load_d5_native_spectra, load_d5_raw_cohort
from rpe.methods.catalog import load_classical_catalog
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase4_d5_protocol_a import _phase1_source, _resolve_cwt
from rpe.runner.phase6_appendix_audits import assemble_phase4_sensitivity_rows
from rpe.runner.phase6_appendix_metric_core import (
    METRIC_IDS, PEAK_METRIC_IDS, evaluate_record_sensitivity, pack_record_sensitivity_values,
    rematerialize_p8_p12_conditions, unpack_record_sensitivity_values,
)


_METRICS = (
    "mse", "rmse", "mae", "sam", "pearson_r", "nmse",
    "wasserstein_1_cm1", "is_like_structure_to_noise", "precision",
    "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio",
)
_PANELS = ("d5_a", "d5_b")
_P10_BUDGET = 64 * 2**30
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


def _rruff_worker_payload(payload: tuple[int, int, str, int, str, object, object, object, object, Mapping[float, int], Mapping[str, object]]) -> tuple[int, object]:
    order, index, record_id, class_label, mineral_name, spectrum, phase1, sweep, cwt, points, support = payload
    with threadpool_limits(limits=1, user_api="blas"):
        source = _phase1_source(order, record_id, class_label, mineral_name, spectrum)
        condition_ids, conditions = rematerialize_p8_p12_conditions(source, phase1, sweep, p10_memory_budget_bytes=_P10_BUDGET)
        return index, evaluate_record_sensitivity(spectrum, condition_ids, conditions, record_id=record_id, support_start_cm1=float(support["start_cm1"]), support_stop_cm1=float(support["stop_cm1"]), support_point_counts=points, support_max_gap_cm1=3.0, cwt_system=cwt)


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
        raise RuntimeError("RRUFF process worker was not initialized")
    order, index, record_id, class_label, mineral_name, spectrum = job
    index, value = _rruff_worker_payload((order, index, record_id, class_label, mineral_name, spectrum, _PROCESS_STATE["phase1"], _PROCESS_STATE["sweep"], _PROCESS_STATE["cwt"], _PROCESS_STATE["points"], _PROCESS_STATE["support"]))
    return index, pack_record_sensitivity_values(value)


_pack_record_result = pack_record_sensitivity_values
_unpack_record_result = unpack_record_sensitivity_values


def _panel_rows(config_raw: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    rows = {str(row["panel_id"]): row for row in config_raw.get("panel_manifest", ())
            if str(row.get("panel_id")) in _PANELS}
    if set(rows) != set(_PANELS):
        # The synthetic TDD seam intentionally does not need the whole frozen config.
        return {panel: {"panel_id": panel, "endpoint_id": "d5", "protocol_id": panel[-1]} for panel in _PANELS}
    return rows


def _status(panel: Mapping[str, object], component: str, condition: str, reason: str = "adapter_local_synthetic_closure") -> Mapping[str, object]:
    return MappingProxyType({
        "component": component, "status_order": 0, "panel_id": panel["panel_id"],
        "endpoint_id": panel["endpoint_id"], "protocol_id": panel["protocol_id"],
        "condition_id": condition, "state": "not_evaluable",
        "reason_code": reason, "numerical_row_count": 13,
        "rank_row_count": 2, "evidence_key": "rruff_phase4_adapter",
    })


def _alignment(panel: Mapping[str, object], condition: str, reason: str = "adapter_local_synthetic_closure") -> tuple[Mapping[str, object], ...]:
    return tuple(MappingProxyType({
        "panel_id": panel["panel_id"], "endpoint_id": panel["endpoint_id"],
        "protocol_id": panel["protocol_id"], "condition_id": condition,
        "metric_output_id": metric, "ag": None, "acc_cross": None,
        "state": "not_evaluable", "reason_code": reason,
    }) for metric in _METRICS)


def _rank(panel: Mapping[str, object], condition: str, reference: str, reason: str = "adapter_local_synthetic_closure") -> tuple[Mapping[str, object], ...]:
    return tuple(MappingProxyType({
        "panel_id": panel["panel_id"], "endpoint_id": panel["endpoint_id"],
        "protocol_id": panel["protocol_id"], "condition_id": condition,
        "reference_condition_id": reference, "statistic": statistic, "tau_b": None,
        "reference_tie_count": None, "candidate_tie_count": None,
        "max_abs_rank_displacement": None, "stable": False, "state": "not_evaluable",
        "reason_code": reason,
    }) for statistic in ("ag", "acc_cross"))


def _synthetic_result(config_raw: Mapping[str, object]) -> DatasetSensitivityResult:
    panels = _panel_rows(config_raw)
    resampling = tuple(f"{interpolator}:{spacing:.17g}" for interpolator in ("linear", "cubic", "pchip") for spacing in (0.5, 1.0, 2.0))
    normalizations = ("none", "maximum", "area", "snv")
    tolerances = ("tolerance:1", "tolerance:2", "tolerance:4", "tolerance:8")
    status = tuple(_status(panels[p], "resampling", c) for p in _PANELS for c in resampling) + tuple(_status(panels[p], "normalization", c) for p in _PANELS for c in normalizations) + tuple(_status(panels[p], "phase4_peak_tolerance", c) for p in _PANELS for c in tolerances)
    def surface(conditions: tuple[str, ...], reference: str):
        return (tuple(row for p in _PANELS for c in conditions for row in _alignment(panels[p], c)), tuple(row for p in _PANELS for c in conditions for row in _rank(panels[p], c, reference)))
    ra, rr = surface(resampling, "native")
    na, nr = surface(normalizations, "none")
    pa, pr = surface(tolerances, "tolerance:2")
    return DatasetSensitivityResult(status, ra, rr, na, nr, pa, pr, MappingProxyType({"adapter": "rruff", "query_union_record_count": 0, "mode": "synthetic"}))


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    with path.open("r", encoding="utf-8") as stream:
        return tuple(MappingProxyType(json.loads(line)) for line in stream if line.strip())


def _parent_authority(root: Path, panel: Mapping[str, object]) -> tuple[tuple[Mapping[str, object], ...], dict[str, tuple[float, float]]]:
    parent = _read_jsonl(root / str(panel["parent_path"]) / "class_observations.jsonl")
    reference_rows = _read_jsonl(root / str(panel["parent_path"]) / "alignment_results.jsonl")
    reference = {str(row["metric_output_id"]): (float(row["ag"]), float(row["acc"])) for row in reference_rows if row.get("metric_state") == "complete"}
    if len(parent) != 354_120 or set(reference) != set(METRIC_IDS):
        raise ValueError(f"{panel['panel_id']}: incomplete parent authority")
    return parent, reference


def _aggregate_metric_rows(records, occurrence_indices, class_labels, condition_ids, *, surface_index: int, field: str, peak_only: bool) -> list[dict[str, object]]:
    metric_ids = PEAK_METRIC_IDS if peak_only else METRIC_IDS
    occurrence_indices = tuple(int(index) for index in occurrence_indices)
    expected_classes = {str(int(class_labels[index])) for index in occurrence_indices}
    rows: list[dict[str, object]] = []
    for condition_index, condition_id in enumerate(condition_ids):
        perturbation, alpha_hex = condition_id.split(":", 1)
        grouped: dict[str, list[np.ndarray]] = defaultdict(list)
        for index in occurrence_indices:
            values = getattr(records[index], field)[surface_index, condition_index, :]
            if not np.isfinite(values).all():
                raise ValueError("not_evaluable_incomplete_record_condition_grid")
            grouped[str(int(class_labels[index]))].append(values)
        if set(grouped) != expected_classes:
            raise ValueError("not_evaluable_incomplete_record_condition_grid")
        for cluster in sorted(grouped, key=int):
            for metric_index, metric in enumerate(metric_ids):
                metric_harms = [float(values[metric_index]) for values in grouped[cluster]]
                rows.append({"cluster_id": cluster, "perturbation_id": perturbation, "alpha": struct.unpack("<d", bytes.fromhex(alpha_hex))[0], "metric_output_id": metric, "metric_harm": float(np.mean(metric_harms)), "state": "complete", "occurrence_count": len(metric_harms)})
    return rows


def _complete_surface(panel, parent, reference, *, condition: str, rows, require_identity: bool, reference_condition_id: str):
    return assemble_phase4_sensitivity_rows(parent, rows, panel_id=str(panel["panel_id"]), endpoint_id=str(panel["endpoint_id"]), protocol_id=str(panel["protocol_id"]), condition_id=condition, metric_ids=METRIC_IDS, reference_by_metric=reference, reference_condition_id=reference_condition_id, require_identity=require_identity)


def _closed_surface(panel, component: str, condition: str, reference: str, reason: str):
    return _status(panel, component, condition, reason), _alignment(panel, condition, reason), _rank(panel, condition, reference, reason)


def build_rruff_phase4_sensitivity(
    config_raw: Mapping[str, object], project_root: Path | str, worker_count: int, *,
    _testing_rows: Mapping[str, object] | None = None,
) -> DatasetSensitivityResult:
    """Build D5 aggregate sensitivity surfaces; never reruns matching/predictions."""
    if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count < 1:
        raise ValueError("worker_count must be a positive integer")
    if _testing_rows is not None:
        return _synthetic_result(config_raw)
    root = Path(project_root)
    panels = _panel_rows(config_raw)
    cohort = load_d5_raw_cohort(root / "experiments/phase05/configs/d5_rruff_protocol.json", root / "data/unified/rruff_raman_raw")
    occurrence_indices = tuple(int(index) for split in cohort.splits for index in split.query_indices)
    occurrences: dict[int, int] = defaultdict(int)
    for index in occurrence_indices:
        occurrences[index] += 1
    query_indices = tuple(sorted(occurrences))
    if len(cohort.record_ids) != 3770 or len(query_indices) != 3012 or sum(occurrences.values()) != 6621:
        raise ValueError("frozen D5 query occurrence ledger mismatch")
    spectra = load_d5_native_spectra(root / "data/unified/rruff_raman_raw", tuple(cohort.record_ids[index] for index in query_indices))
    phase1 = load_phase1_core_config(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json")
    sweep = load_perturbation_sweep_config(root / "experiments/shared/raman_perturbation_sweep_v1.json")
    cwt = _resolve_cwt(load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json"))
    support = config_raw["supports"]["d5_raw_rruff"]
    points = {float(key): int(value) for key, value in support["metric_view_point_counts_by_spacing_cm1"].items()}

    jobs = tuple((order, index, cohort.record_ids[index], int(cohort.class_labels[index]), cohort.mineral_names[index], spectra[order]) for order, index in enumerate(query_indices))

    records = {}
    with threadpool_limits(limits=1, user_api="blas"):
        # Process workers are source-record blocks, with deterministic input-order reduction.
        if worker_count == 1:
            for job in jobs:
                index, values = _rruff_worker_payload((*job, phase1, sweep, cwt, points, support))
                records[index] = values
        else:
            with ProcessPoolExecutor(max_workers=min(worker_count, len(query_indices)), initializer=_init_process_worker, initargs=(str(root),)) as executor:
                for index, payload in executor.map(_process_job, jobs):
                    records[index] = unpack_record_sensitivity_values(payload)
    prototype = records[query_indices[0]]
    if any(values.condition_ids != prototype.condition_ids for values in records.values()):
        raise ValueError("D5 condition order drift")
    authorities = {key: _parent_authority(root, panels[key]) for key in _PANELS}
    status, ra, rr, na, nr, pa, pr = [], [], [], [], [], [], []
    surfaces = (
        ("resampling", "resampling_harms", prototype.resampling_ids, "native", False),
        ("normalization", "normalization_harms", prototype.normalization_ids, "none", False),
        ("phase4_peak_tolerance", "peak_tolerance_harms", tuple(f"tolerance:{x}" for x in prototype.tolerance_ids), "tolerance:2", True),
    )
    for component, field, surface_ids, reference_id, peak_only in surfaces:
        for surface_index, condition in enumerate(surface_ids):
            try:
                rows = _aggregate_metric_rows(records, occurrence_indices, cohort.class_labels, prototype.condition_ids, surface_index=surface_index, field=field, peak_only=peak_only)
            except ValueError as error:
                rows, closure_reason = None, str(error)
            for key in _PANELS:
                panel, (parent, reference) = panels[key], authorities[key]
                current = rows
                if current is not None and peak_only:
                    # Scalar rows are bit-identical native parent authority; only peak rows vary.
                    current = [dict(row) for row in parent if str(row["metric_output_id"]) not in PEAK_METRIC_IDS] + rows
                if current is None:
                    current_status, current_alignment, ranks = _closed_surface(panel, component, condition, reference_id, closure_reason)
                else:
                    try:
                        assembled = _complete_surface(panel, parent, reference, condition=condition, rows=current, reference_condition_id=reference_id, require_identity=(component == "normalization" and condition == "none") or (component == "phase4_peak_tolerance" and condition == "tolerance:2"))
                        current_status, current_alignment = assembled.status_rows[0], assembled.alignment_rows
                        ranks = tuple(MappingProxyType({**row, "reference_condition_id": reference_id}) for row in assembled.rank_rows)
                    except Exception:
                        if (component == "normalization" and condition == "none") or (component == "phase4_peak_tolerance" and condition == "tolerance:2"):
                            raise
                        current_status, current_alignment, ranks = _closed_surface(panel, component, condition, reference_id, "not_evaluable_incomplete_record_condition_grid")
                status.append(MappingProxyType({**current_status, "component": component}))
                if component == "resampling": ra.extend(current_alignment); rr.extend(ranks)
                elif component == "normalization": na.extend(current_alignment); nr.extend(ranks)
                else: pa.extend(current_alignment); pr.extend(ranks)
    observed = (len(status), len(ra), len(rr), len(na), len(nr), len(pa), len(pr))
    if observed != (34, 234, 36, 104, 16, 104, 16):
        raise ValueError(f"D5 sensitivity count mismatch: {observed!r}")
    return DatasetSensitivityResult(tuple(status), tuple(ra), tuple(rr), tuple(na), tuple(nr), tuple(pa), tuple(pr), MappingProxyType({"adapter": "rruff", "cohort_record_count": len(cohort.record_ids), "query_union_record_count": len(query_indices), "query_occurrence_count": sum(occurrences.values())}))


__all__ = ["DatasetSensitivityResult", "build_rruff_phase4_sensitivity"]
