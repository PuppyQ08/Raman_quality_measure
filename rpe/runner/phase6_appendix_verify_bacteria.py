"""Independent Bacteria-ID adapter used only by the Step-7 verifier."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.alignment import AlignmentObservation, alignment_gap, cross_perturbation_accuracy
from rpe.methods.catalog import load_classical_catalog
from rpe.perturb import load_perturbation_sweep_config
from rpe.runner.phase1_config import load_phase1_core_config
from rpe.runner.phase4_d2_protocol_a import (
    _phase1_source,
    load_phase4_d2_protocol_a_config,
    reconstruct_d2_protocol_a_inputs,
)
from rpe.runner.phase6_appendix_verifier_science import (
    v_METRIC_IDS,
    v_evaluate_record_sensitivity,
    v_parent_downstream_collapse,
    v_rank_stability,
    v_rematerialize_p8_p12_conditions,
    pack_verifier_record_sensitivity_result,
    unpack_verifier_record_sensitivity_result,
)


_PANELS = (
    "d2_5_a", "d2_5_b", "d2_10_a", "d2_10_b",
    "d2_20_a", "d2_20_b", "d1_a", "d1_b_closed",
)
_COMPONENTS = ("resampling", "normalization", "phase4_peak_tolerance")
_PEAK_METRICS = v_METRIC_IDS[8:]
_D1_B_REASON = "not_evaluable_failed_alpha0_equivalence"
_CWT_SYSTEM_ID = "6579e8210ea7fee9755ce133eeac1ecfe7012f59a79ce30d78d106f7993cc511"
_P10_BUDGET = 64 * 1024**3
_PROCESS_STATE: dict[str, object] = {}


class VerifyBacteriaError(ValueError):
    pass


@dataclass(frozen=True)
class VerifierBacteriaSensitivityResult:
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
        raise VerifyBacteriaError(f"parent artifact unreadable: {path}") from error


def _v_panel_meta(raw: Mapping[str, object]) -> Mapping[str, Mapping[str, object]]:
    rows = raw.get("panel_manifest")
    if not isinstance(rows, Sequence):
        raise VerifyBacteriaError("panel_manifest is required")
    found = {str(row.get("panel_id")): row for row in rows if isinstance(row, Mapping)}
    if tuple(found) != tuple(str(value) for value in raw.get("fixed_panel_order", ())):
        raise VerifyBacteriaError("panel order drift")
    if set(_PANELS) - set(found):
        raise VerifyBacteriaError("Bacteria panel missing")
    return MappingProxyType(found)


def _v_conditions(raw: Mapping[str, object]) -> Mapping[str, tuple[str, ...]]:
    resampling = raw.get("resampling_conditions")
    if not isinstance(resampling, Mapping):
        raise VerifyBacteriaError("resampling conditions invalid")
    values = MappingProxyType({
        "resampling": tuple(
            f"{kind}:{float(spacing):g}"
            for kind in ("linear", "cubic", "pchip")
            for spacing in resampling.get("spacing_cm1", ())
        ),
        "normalization": tuple(str(value) for value in raw.get("normalization_conditions", ())),
        "phase4_peak_tolerance": tuple(f"tolerance:{value}" for value in (1, 2, 4, 8)),
    })
    if tuple(map(len, (values["resampling"], values["normalization"], values["phase4_peak_tolerance"]))) != (9, 4, 4):
        raise VerifyBacteriaError("condition grid drift")
    return values


def _v_parent_artifacts(root: Path, metadata: Mapping[str, Mapping[str, object]]) -> tuple[Mapping[str, tuple[Mapping[str, object], ...]], Mapping[str, Mapping[str, tuple[float, float]]]]:
    observations, references = {}, {}
    for panel in _PANELS:
        if panel == "d1_b_closed":
            continue
        path = root / str(metadata[panel]["parent_path"])
        shot = int(panel.split("_")[1]) if panel.startswith("d2_") else None
        rows = _v_jsonl(path / "class_observations.jsonl")
        alignment = _v_jsonl(path / "alignment_results.jsonl")
        if shot is not None:
            rows = tuple(row for row in rows if int(row.get("shot_count", -1)) == shot)
            alignment = tuple(row for row in alignment if int(row.get("shot_count", -1)) == shot)
        refs = {str(row["metric_output_id"]): (float(row["ag"]), float(row["acc_cross"])) for row in alignment if row.get("state") == "complete"}
        if tuple(refs) != v_METRIC_IDS:
            raise VerifyBacteriaError(f"{panel}: incomplete parent alignment")
        # This consumes and validates the 13 exact downstream copies before any
        # fresh metric value is paired with them.
        v_parent_downstream_collapse(rows)
        observations[panel], references[panel] = rows, MappingProxyType(refs)
    return MappingProxyType(observations), MappingProxyType(references)


def _v_status(component: str, meta: Mapping[str, object], condition: str, order: int, state: str, reason: str, numerical: int, ranks: int, evidence: str) -> Mapping[str, object]:
    return MappingProxyType({
        "component": component, "status_order": order, "panel_id": meta["panel_id"],
        "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"],
        "condition_id": condition, "state": state, "reason_code": reason,
        "numerical_row_count": numerical, "rank_row_count": ranks, "evidence_key": evidence,
    })


def _v_closed_rows(component: str, meta: Mapping[str, object], condition: str, metric_ids: Sequence[str], reason: str) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    reference = "native" if component == "resampling" else ("none" if component == "normalization" else "tolerance:2")
    alignment = tuple(MappingProxyType({
        "panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"],
        "condition_id": condition, "metric_output_id": metric, "ag": None, "acc_cross": None,
        "state": "not_evaluable", "reason_code": reason,
    }) for metric in metric_ids)
    ranks = tuple(MappingProxyType({
        "panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"],
        "condition_id": condition, "reference_condition_id": reference, "statistic": statistic,
        "tau_b": None, "reference_tie_count": None, "candidate_tie_count": None,
        "max_abs_rank_displacement": None, "stable": False, "state": "not_evaluable", "reason_code": reason,
    }) for statistic in ("ag", "acc_cross"))
    return alignment, ranks


def _v_assemble(parent_rows: Sequence[Mapping[str, object]], metric_rows: Sequence[Mapping[str, object]], *, component: str, meta: Mapping[str, object], condition: str, references: Mapping[str, tuple[float, float]]) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    downstream = v_parent_downstream_collapse(parent_rows)
    expected = {(cluster, perturbation, alpha, metric) for cluster, perturbation, alpha in downstream for metric in v_METRIC_IDS}
    fresh: dict[tuple[str, str, float, str], float] = {}
    for row in metric_rows:
        cluster = str(row.get("class_label", row.get("cluster_id", "")))
        key = (cluster, str(row["perturbation_id"]), float(row["alpha"]), str(row["metric_output_id"]))
        harm = float(row["metric_harm"])
        if key in fresh or key not in expected or not np.isfinite(harm):
            raise VerifyBacteriaError("fresh metric grid duplicate or invalid")
        fresh[key] = harm
    if set(fresh) != expected:
        raise VerifyBacteriaError("incomplete fixed cluster-condition-metric grid")
    ag, acc = {}, {}
    for metric in v_METRIC_IDS:
        observations = tuple(AlignmentObservation(cluster, perturbation, alpha, fresh[(cluster, perturbation, alpha, metric)], downstream[(cluster, perturbation, alpha)]) for cluster, perturbation, alpha in downstream)
        ag[metric] = alignment_gap(observations).alignment_gap
        acc[metric] = cross_perturbation_accuracy(observations).accuracy
    identity_required = (component == "normalization" and condition == "none") or (component == "phase4_peak_tolerance" and condition == "tolerance:2")
    if identity_required and any((ag[m], acc[m]) != references[m] for m in v_METRIC_IDS):
        raise VerifyBacteriaError("normalization-none or tolerance-2 identity mismatch")
    reference_condition = "native" if component == "resampling" else ("none" if component == "normalization" else "tolerance:2")
    alignment = tuple(MappingProxyType({
        "panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"],
        "condition_id": condition, "metric_output_id": metric, "ag": ag[metric], "acc_cross": acc[metric],
        "state": "complete_numeric", "reason_code": "",
    }) for metric in v_METRIC_IDS)
    ranks = tuple(MappingProxyType({
        "panel_id": meta["panel_id"], "endpoint_id": meta["endpoint_id"], "protocol_id": meta["protocol_id"],
        "condition_id": condition, "reference_condition_id": reference_condition, **v_rank_stability(
            {metric: references[metric][index] for metric in v_METRIC_IDS}, values, statistic=statistic, identities=v_METRIC_IDS),
    }) for index, (statistic, values) in enumerate((("ag", ag), ("acc_cross", acc))))
    return alignment, ranks


def _v_worker(job: tuple[int, str, int, object], *, phase1: object, sweep: object, cwt: object, support: Mapping[str, object]) -> tuple[int, object]:
    order, record_id, label, spectrum = job
    with threadpool_limits(limits=1, user_api="blas"):
        condition_ids, spectra = v_rematerialize_p8_p12_conditions(_phase1_source(order, record_id, label, spectrum), phase1, sweep, p10_memory_budget_bytes=_P10_BUDGET)
        return label, v_evaluate_record_sensitivity(spectrum, condition_ids, spectra, record_id=record_id, support_start_cm1=float(support["start_cm1"]), support_stop_cm1=float(support["stop_cm1"]), support_point_counts={float(key): int(value) for key, value in support["metric_view_point_counts_by_spacing_cm1"].items()}, support_max_gap_cm1=1.561, cwt_system=cwt)


def _init_process_worker(root_text: str) -> None:
    root = Path(root_text)
    catalog = load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json")
    matched = tuple(system for system in catalog.systems if system.system_id == _CWT_SYSTEM_ID)
    if len(matched) != 1:
        raise VerifyBacteriaError("CWT system did not resolve exactly once")
    _PROCESS_STATE.clear()
    _PROCESS_STATE.update({
        "phase1": load_phase1_core_config(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json"),
        "sweep": load_perturbation_sweep_config(root / "experiments/shared/raman_perturbation_sweep_v1.json"),
        "cwt": matched[0],
        "support": {"start_cm1": 387.0, "stop_cm1": 1791.0, "metric_view_point_counts_by_spacing_cm1": {"0.5": 2809, "1.0": 1405, "2.0": 703}},
    })


def _process_job(job: tuple[int, str, int, object]) -> tuple[int, object]:
    if not _PROCESS_STATE:
        raise VerifyBacteriaError("process worker was not initialized")
    label, value = _v_worker(job, phase1=_PROCESS_STATE["phase1"], sweep=_PROCESS_STATE["sweep"], cwt=_PROCESS_STATE["cwt"], support=_PROCESS_STATE["support"])
    return label, pack_verifier_record_sensitivity_result(value)


_pack_record_result = pack_verifier_record_sensitivity_result
_unpack_record_result = unpack_verifier_record_sensitivity_result


def _v_metric_rows(values: Sequence[tuple[int, object]], *, component: str, condition: str, parent_rows: Sequence[Mapping[str, object]]) -> tuple[tuple[Mapping[str, object], ...], str | None]:
    index = {"resampling": 0, "normalization": 1, "phase4_peak_tolerance": 2}[component]
    prefix = f"{component}:{condition}" if component != "phase4_peak_tolerance" else "peak_tolerance:"
    for _, result in values:
        if any(key == prefix or key.startswith(prefix + ":") for key in result.closure_reasons):
            key = next(key for key in result.closure_reasons if key == prefix or key.startswith(prefix + ":"))
            return (), f"not_evaluable_{result.closure_reasons[key]}"
    means: dict[tuple[int, str, float, str], list[float]] = {}
    for label, result in values:
        surface = (result.resampling_harms, result.normalization_harms, result.peak_tolerance_harms)[index]
        ids = (result.resampling_ids, result.normalization_ids, tuple(f"tolerance:{item}" for item in result.tolerance_ids))[index]
        if condition not in ids:
            return (), "not_evaluable_surface_condition_drift"
        array = surface[ids.index(condition)]
        metric_ids = _PEAK_METRICS if component == "phase4_peak_tolerance" else v_METRIC_IDS
        for ci, cell in enumerate(result.condition_ids):
            perturbation, alpha_hex = cell.split(":", 1)
            alpha = float(np.frombuffer(bytes.fromhex(alpha_hex), dtype="<f8")[0])
            for mi, metric in enumerate(metric_ids):
                value = float(array[ci, mi])
                if not np.isfinite(value):
                    return (), "not_evaluable_incomplete_metric_set"
                means.setdefault((label, perturbation, alpha, metric), []).append(value)
    rows = []
    for parent in parent_rows:
        label, perturbation, alpha, metric = int(parent["class_label"]), str(parent["perturbation_id"]), float(parent["alpha"]), str(parent["metric_output_id"])
        if component == "phase4_peak_tolerance" and metric not in _PEAK_METRICS:
            harm = float(parent["metric_harm"])
        else:
            cell = means.get((label, perturbation, alpha, metric), ())
            if len(cell) != 100:
                return (), "not_evaluable_incomplete_record_class_grid"
            harm = float(np.mean(cell, dtype=np.float64))
        rows.append({**parent, "metric_harm": harm})
    return tuple(rows), None


def _v_materialize(raw: Mapping[str, object], root: Path, worker_count: int) -> tuple[tuple[tuple[int, object], ...], Mapping[str, object]]:
    config = load_phase4_d2_protocol_a_config(root / "experiments/phase4/configs/d2_protocol_a_full_domain_v1.json")
    inputs = reconstruct_d2_protocol_a_inputs(root / "data/unified/bacteria_id_reference", root / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json", config)
    catalog = load_classical_catalog(root / "experiments/phase3/configs/classical_system_catalog_v1.json")
    matched = tuple(system for system in catalog.systems if system.system_id == _CWT_SYSTEM_ID)
    if len(matched) != 1:
        raise VerifyBacteriaError("CWT system did not resolve exactly once")
    jobs = tuple((order, record_id, int(label), spectrum) for order, (record_id, label, spectrum) in enumerate(zip(inputs.record_ids, inputs.native_test_labels, inputs.native_test_spectra, strict=True)))
    if len(jobs) != 3000 or {label for _, _, label, _ in jobs} != set(range(30)):
        raise VerifyBacteriaError("shared D1/D2 test cohort drift")
    phase1 = load_phase1_core_config(root / "experiments/phase1/configs/rruff_raw_core10k_v1.json")
    sweep = load_perturbation_sweep_config(root / "experiments/shared/raman_perturbation_sweep_v1.json")
    support = raw["supports"]["d1_d2_bacteria_id"]
    if worker_count == 1:
        values = tuple(_v_worker(job, phase1=phase1, sweep=sweep, cwt=matched[0], support=support) for job in jobs)
    else:
        with ProcessPoolExecutor(max_workers=min(worker_count, len(jobs)), initializer=_init_process_worker, initargs=(str(root),)) as executor:
            values = tuple((label, unpack_verifier_record_sensitivity_result(payload)) for label, payload in executor.map(_process_job, jobs))
    return values, MappingProxyType({
        "record_count": len(values), "class_count": 30,
        "records_per_class": 100,
        "detector_call_count": sum(int(result.identity["detector_call_count"]) for _, result in values),
    })


def v_build_bacteria_sensitivity(config_raw: Mapping[str, object], project_root: Path, worker_count: int) -> VerifierBacteriaSensitivityResult:
    """Independently reconstruct all Bacteria Step-7 Phase-4 sensitivity rows."""
    if not isinstance(config_raw, Mapping) or not isinstance(worker_count, int) or worker_count < 1:
        raise VerifyBacteriaError("invalid config or worker count")
    if tuple(config_raw.get("metric_ids", ())) != v_METRIC_IDS:
        raise VerifyBacteriaError("metric manifest drift")
    root, meta, conditions = Path(project_root), _v_panel_meta(config_raw), _v_conditions(config_raw)
    parents, references = _v_parent_artifacts(root, meta)
    values, identity = _v_materialize(config_raw, root, worker_count)
    statuses, tables, ranks, order = [], {name: [] for name in _COMPONENTS}, {name: [] for name in _COMPONENTS}, 0
    for component in _COMPONENTS:
        for panel in _PANELS:
            for condition in conditions[component]:
                order += 1
                if panel == "d1_b_closed":
                    statuses.append(_v_status(component, meta[panel], condition, order, "not_evaluable", _D1_B_REASON, 0, 0, "fixed_d1_b_closure"))
                    continue
                metric_rows, reason = _v_metric_rows(values, component=component, condition=condition, parent_rows=parents[panel])
                if reason is not None:
                    alignment, rank_rows = _v_closed_rows(component, meta[panel], condition, v_METRIC_IDS, reason)
                    statuses.append(_v_status(component, meta[panel], condition, order, "not_evaluable", reason, 13, 2, "record_core_closure"))
                else:
                    alignment, rank_rows = _v_assemble(parents[panel], metric_rows, component=component, meta=meta[panel], condition=condition, references=references[panel])
                    statuses.append(_v_status(component, meta[panel], condition, order, "complete_numeric", "", 13, 2, "native_metric_reconstruction"))
                tables[component].extend(alignment)
                ranks[component].extend(rank_rows)
    expected = {"resampling": (72, 819, 126), "normalization": (32, 364, 56), "phase4_peak_tolerance": (32, 364, 56)}
    for component, count in expected.items():
        actual = (sum(row["component"] == component for row in statuses), len(tables[component]), len(ranks[component]))
        if actual != count:
            raise VerifyBacteriaError(f"{component}: count drift {actual}")
    return VerifierBacteriaSensitivityResult(tuple(statuses), tuple(tables["resampling"]), tuple(ranks["resampling"]), tuple(tables["normalization"]), tuple(ranks["normalization"]), tuple(tables["phase4_peak_tolerance"]), tuple(ranks["phase4_peak_tolerance"]), identity)


__all__ = ["VerifierBacteriaSensitivityResult", "VerifyBacteriaError", "v_build_bacteria_sensitivity"]
