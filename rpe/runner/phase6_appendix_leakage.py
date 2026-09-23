"""Phase-6 Step-7 leakage materializer.

This module deliberately returns only audit receipts: source spectra are kept
inside the scan and are never included in its result values.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rpe.runner.phase6_appendix_audits import (
    AppendixAuditsError,
    _validate_authorities,
    exact_spectrum_sha256,
    resample_spectrum,
    scan_near_duplicates,
    select_calibration_pairs,
)

_THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_THRESHOLD = 1e-3
_CANDIDATE_FIELDS = (
    "boundary_id", "left_role_id", "right_role_id", "left_record_id",
    "right_record_id", "left_entity_id", "right_entity_id", "distance",
    "correlation", "left_source_sha256", "right_source_sha256", "status",
    "reason_code",
)


@dataclass(frozen=True)
class LeakageAuditResult:
    boundary_rows: tuple[Mapping[str, object], ...]
    candidate_rows: tuple[Mapping[str, object], ...]
    status_rows: tuple[Mapping[str, object], ...]
    identity: Mapping[str, object]


def _digest(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bind_phase2(root: Path) -> dict[str, object]:
    from rpe.semisynth import load_phase2_config, split_rruff_pairs
    config_path = root / "experiments/phase2/configs/semisynth_v1.json"
    pair_path = root / "data/unified/rruff_raman_pairs.jsonl"
    raw_path = root / "data/unified/rruff_raman_raw"
    config = load_phase2_config(config_path, project_root=root)
    summary = split_rruff_pairs(pair_path, raw_path, config=config, verify_checksums=True)
    roles: dict[str, set[str]] = {}
    for assignment in summary.assignments: roles.setdefault(assignment.group_key, set()).add(assignment.role.value)
    if summary.assignment_count != 15764 or summary.group_count != 3919 or dict(summary.pair_counts) != {"extraction_fit": 9371, "signal_template": 3127, "real_holdout": 3266} or dict(summary.group_counts) != {"extraction_fit": 2333, "signal_template": 761, "real_holdout": 825} or summary.ledger_sha256 != "0b6323973a3afabc840c57a051a7eeb31cb89efcb2bdea0546c5b15141d95c5e" or any(len(value) != 1 for value in roles.values()):
        raise AppendixAuditsError("phase2 RRUFF role split", "frozen direct role identity mismatch")
    return {"assignment_count": summary.assignment_count, "group_count": summary.group_count, "pair_counts": dict(summary.pair_counts), "group_counts": dict(summary.group_counts), "ledger_sha256": summary.ledger_sha256, "one_role_per_group": True, "evidence_paths": (str(config_path.relative_to(root)), str(pair_path.relative_to(root)), str(raw_path.relative_to(root)), "reports/phase2/step02_contracts_splits.md")}


def _bind_phase3(root: Path) -> dict[str, object]:
    run = root / "results/phase3/denoising_v1_formal_coverage/phase3-denoising-v1-formal-9f8b856433ee007b96c0fff204fa87db3216a7605ac98e6bca940e872f0db458"
    ledger, complete, config = run / "SHA256SUMS", run / "complete.json", run / "config.json"
    expected = {ledger: "4f75a3236b0cd18e9c2b7af20a332b3139e60cdd8af9a19fab51e4080199fb31", complete: "4181516442b0b343732f278d6a24064091a47c6aabeb55fbb671d2ab5b80e908", config: "248b3fcfce517ada2cd3536865d809b5be6f684981c612776b84ba22670406ab"}
    if any(not path.is_file() or _sha_file(path) != digest for path, digest in expected.items()): raise AppendixAuditsError("phase3 denoising authority", "ledger, complete, or config digest mismatch")
    entries = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1); entries[name] = digest
        if _sha_file(run / name) != digest: raise AppendixAuditsError("phase3 denoising authority", "ledger member mismatch")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("system_count") != 60 or manifest.get("fit_receipt_count") != 24 or manifest.get("transform_receipt_count") != 439200: raise AppendixAuditsError("phase3 denoising authority", "manifest run/status/count mismatch")
    return {"ledger_sha256": expected[ledger], "complete_sha256": expected[complete], "config_sha256": expected[config], "system_count": 60, "fit_receipt_count": 24, "transform_receipt_count": 439200, "evidence_paths": tuple(str((run / name).relative_to(root)) for name in ("SHA256SUMS", "complete.json", "fit_receipts.jsonl", "transform_receipts.jsonl"))}


def _base(definition: Mapping[str, object]) -> dict[str, object]:
    return {
        "boundary_id": str(definition["boundary_id"]),
        "producer_authorities": (), "consumer_authorities": (),
        "fit_roles": tuple(definition.get("fit_roles", ())),
        "selection_roles": tuple(definition.get("selection_roles", ())),
        "evaluation_roles": tuple(definition.get("evaluation_roles", ())),
        "leakage_entity": definition.get("leakage_entity", ""),
        "expected_overlap_semantics": definition.get("expected_overlap_semantics", ""),
        "recomputed_digests": {}, "recomputed_counts": {},
        "exact_overlap_count": 0, "near_candidate_count": 0,
        "status": "pass", "reason_code": "", "evidence_paths": (),
    }


def fixed_boundary_rows(
    definitions: Sequence[Mapping[str, object]],
    role_ids: Mapping[str, Mapping[str, set[str]]],
    *,
    near_status: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[dict[str, object], ...]:
    """Construct role-boundary rows in declared order; useful for local TDD."""
    output = []
    for definition in definitions:
        row = _base(definition); boundary = str(row["boundary_id"]); roles = role_ids.get(boundary, {})
        names = tuple(row["fit_roles"]) + tuple(row["selection_roles"]) + tuple(row["evaluation_roles"])
        values = {name: set(roles.get(name, set())) for name in names}
        row["recomputed_counts"] = {name: len(values[name]) for name in names}
        row["recomputed_digests"] = {name: _digest(tuple(values[name])) for name in names}
        forbidden = 0
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1:]:
                forbidden += len(values[left_name] & values[right_name])
        row["exact_overlap_count"] = forbidden
        if forbidden:
            row.update(status="fail", reason_code="fail_role_overlap_bridge")
        elif near_status and boundary in near_status:
            row["status"], row["reason_code"] = near_status[boundary]
        output.append(row)
    return tuple(output)


def classify_duplicate_candidates(boundary_id: str, rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    output = []
    for source in rows:
        row = {key: source.get(key, "") for key in _CANDIDATE_FIELDS}
        row["boundary_id"] = boundary_id
        bridged = (str(row["left_entity_id"]) == str(row["right_entity_id"]) or
                   (row["left_source_sha256"] and row["left_source_sha256"] == row["right_source_sha256"]))
        row["status"] = "fail" if bridged else "warning_near_duplicate_bridge"
        row["reason_code"] = "fail_near_duplicate_identity_bridge" if bridged else "warning_near_duplicate_bridge"
        output.append(row)
    return tuple(sorted(output, key=lambda x: tuple(str(x[k]) for k in _CANDIDATE_FIELDS)))


def evaluate_direct_cells(
    cells: Sequence[Mapping[str, object]],
    prohibited_pairs: Sequence[tuple[str, str]],
    *,
    entity_key: str = "entity",
) -> dict[str, object]:
    """Validate direct role boundaries without ever mixing unrelated cells."""
    cell_receipts = []
    total = 0
    for cell in cells:
        cell_id = str(cell["cell_id"]); roles = cell["roles"]
        if not isinstance(roles, Mapping):
            raise AppendixAuditsError("direct cells", "roles must be a mapping")
        record = {name: set(str(value) for value in values) for name, values in roles.items()}
        for left, right in prohibited_pairs:
            if not record.get(left) or not record.get(right):
                raise AppendixAuditsError("direct cells", f"empty expected role in {cell_id}: {left}/{right}")
        overlaps = {f"{left}|{right}": len(record[left] & record[right]) for left, right in prohibited_pairs}
        total += sum(overlaps.values())
        cell_receipts.append({"cell_id": cell_id, "role_counts": {name: len(values) for name, values in record.items()}, "role_digests": {name: _digest(tuple(values)) for name, values in record.items()}, "exact_overlap_count": sum(overlaps.values()), "pair_overlap_counts": overlaps, "entity_key": entity_key})
    return {"exact_overlap_count": total, "cells": tuple(cell_receipts)}


def project_candidates_to_cells(
    boundary_id: str,
    candidates: Sequence[tuple[Mapping[str, object], Mapping[str, object], float, float]],
    cells: Sequence[Mapping[str, object]],
    prohibited_pairs: Sequence[tuple[str, str]],
) -> tuple[dict[str, object], ...]:
    """Project a deduplicated record-pair universe back only to legal cells."""
    output = []
    for left, right, distance, correlation in candidates:
        left_id, right_id = str(left["record_id"]), str(right["record_id"])
        for cell in cells:
            roles = cell["roles"]; cell_id = str(cell["cell_id"])
            for left_role, right_role in prohibited_pairs:
                if left_id in roles.get(left_role, set()) and right_id in roles.get(right_role, set()):
                    output.append({"boundary_id": boundary_id, "left_role_id": f"{left_role}@{cell_id}", "right_role_id": f"{right_role}@{cell_id}", "left_record_id": left_id, "right_record_id": right_id, "left_entity_id": str(left["entity_id"]), "right_entity_id": str(right["entity_id"]), "distance": distance, "correlation": correlation, "left_source_sha256": str(left["source_sha256"]), "right_source_sha256": str(right["source_sha256"]), "status": "", "reason_code": ""})
                if right_id in roles.get(left_role, set()) and left_id in roles.get(right_role, set()):
                    output.append({"boundary_id": boundary_id, "left_role_id": f"{left_role}@{cell_id}", "right_role_id": f"{right_role}@{cell_id}", "left_record_id": right_id, "right_record_id": left_id, "left_entity_id": str(right["entity_id"]), "right_entity_id": str(left["entity_id"]), "distance": distance, "correlation": correlation, "left_source_sha256": str(right["source_sha256"]), "right_source_sha256": str(left["source_sha256"]), "status": "", "reason_code": ""})
    return classify_duplicate_candidates(boundary_id, output)


def _record(record_id: str, entity_id: str, role_id: str, axis: np.ndarray, intensity: np.ndarray) -> dict[str, object]:
    axis = np.asarray(axis, dtype="<f8"); intensity = np.asarray(intensity, dtype="<f4")
    if axis.ndim != 1 or intensity.ndim != 1 or axis.shape != intensity.shape or not np.all(np.diff(axis) > 0):
        raise AppendixAuditsError("leakage source view", "requires increasing finite axis and aligned intensity")
    return {"record_id": record_id, "entity_id": entity_id, "role_id": role_id, "axis": axis, "intensity": intensity, "source_sha256": exact_spectrum_sha256(axis, intensity)}


def _cache(records: Sequence[Mapping[str, object]], start: float, stop: float) -> tuple[dict[str, object], ...]:
    target = np.arange(start, stop + 0.5, 1.0, dtype="<f8")
    result = []
    for record in records:
        result.append({**record, "intensity": resample_spectrum(record["axis"], record["intensity"], target, interpolator="linear"), "axis": target})
    return tuple(result)


def _calibration_pairs(records: Sequence[Mapping[str, object]]) -> tuple[tuple[str, str], ...]:
    """Protocol draw order; deliberately does not inherit main-runner sorting."""
    ids = tuple(str(r["record_id"]) for r in records); entities = tuple(str(r["entity_id"]) for r in records)
    counts: dict[str, int] = {}
    for entity in entities: counts[entity] = counts.get(entity, 0) + 1
    admissible_count = len(ids) * (len(ids) - 1) // 2 - sum(count * (count - 1) // 2 for count in counts.values())
    if admissible_count < 100000:
        return tuple((ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids)) if entities[i] != entities[j])
    rng = np.random.Generator(np.random.PCG64(20260817)); chosen = set(); ordered = []
    for _ in range(64):
        for i, j in zip(rng.integers(0, len(ids), 262144), rng.integers(0, len(ids), 262144), strict=True):
            i, j = int(i), int(j)
            if i == j or entities[i] == entities[j]: continue
            pair = (ids[i], ids[j]) if i < j else (ids[j], ids[i])
            if pair not in chosen:
                chosen.add(pair); ordered.append(pair)
                if len(ordered) == 100000: return tuple(ordered)
    raise AppendixAuditsError("calibration", "max batches did not fill target")


def _calibrate(records: Sequence[Mapping[str, object]]) -> tuple[int, int]:
    pairs = _calibration_pairs(records)
    by_id = {str(r["record_id"]): r for r in records}; hits = 0
    for left, right in pairs:
        candidate = scan_near_duplicates((by_id[str(left)],), (by_id[str(right)],), threshold=_THRESHOLD, query_block_size=1, fit_block_size=1)
        hits += sum(row["distance"] is not None for row in candidate)
    return len(pairs), hits


def _scan_boundary(boundary: str, roles: Mapping[str, Sequence[Mapping[str, object]]], *, support: tuple[float, float]) -> tuple[tuple[dict[str, object], ...], dict[str, object], tuple[str, str]]:
    names = tuple(roles); all_records = tuple(record for values in roles.values() for record in values)
    cached = {name: _cache(values, *support) for name, values in roles.items()}
    calibration_count, calibration_hits = _calibrate(tuple(record for values in cached.values() for record in values))
    receipt = {"calibration_pair_count": calibration_count, "calibration_hit_count": calibration_hits, "scan_pair_count": 0, "scan_tile_count": 0}
    if calibration_hits:
        return (), receipt, ("not_evaluable", "not_evaluable_threshold_not_specific")
    candidates = []
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1:]:
            left, right = cached[left_name], cached[right_name]
            receipt["scan_pair_count"] += len(left) * len(right); receipt["scan_tile_count"] += 1
            exact_left = {}
            for record in left: exact_left.setdefault(str(record["source_sha256"]), []).append(record)
            for record in right:
                for match in exact_left.get(str(record["source_sha256"]), ()):
                    candidates.append({"left_role_id": match["role_id"], "right_role_id": record["role_id"], "left_record_id": match["record_id"], "right_record_id": record["record_id"], "left_entity_id": match["entity_id"], "right_entity_id": record["entity_id"], "distance": 0.0, "correlation": 1.0, "left_source_sha256": match["source_sha256"], "right_source_sha256": record["source_sha256"], "status": "fail", "reason_code": "fail_exact_duplicate_bridge"})
            candidates.extend(scan_near_duplicates(left, right, threshold=_THRESHOLD, query_block_size=256, fit_block_size=256))
    classified = list(classify_duplicate_candidates(boundary, candidates))
    for row in classified:
        if float(row.get("distance") or 0.0) == 0.0:
            row["status"] = "fail"; row["reason_code"] = "fail_exact_duplicate_bridge"
    status = ("fail", "fail_exact_duplicate_bridge") if any(x["reason_code"] == "fail_exact_duplicate_bridge" for x in classified) else (("fail", "fail_near_duplicate_identity_bridge") if any(x["status"] == "fail" for x in classified) else (("warning_near_duplicate_bridge", "warning_near_duplicate_bridge") if classified else ("pass", "")))
    return tuple(classified), receipt, status


def _scan_cells(
    boundary: str, cells: Sequence[Mapping[str, object]], *, fit_role: str,
    prohibited_pairs: Sequence[tuple[str, str]], support: tuple[float, float],
) -> tuple[tuple[dict[str, object], ...], dict[str, object], tuple[str, str]]:
    """Stream role-pair tiles; do not materialize a prohibited pair universe."""
    by_id: dict[str, Mapping[str, object]] = {}
    fit_records = []
    for cell in cells:
        cell_id, records = str(cell["cell_id"]), cell["records"]
        fit_records.extend(records[fit_role])
        for values in records.values():
            for record in values: by_id[str(record["record_id"])] = record
    cached = {rid: _cache((record,), *support)[0] for rid, record in by_id.items()}
    calibration_count, calibration_hits = _calibrate(tuple({str(r["record_id"]): cached[str(r["record_id"])] for r in fit_records}.values()))
    receipt = {"calibration_pair_count": calibration_count, "calibration_hit_count": calibration_hits, "scan_pair_count": 0, "scan_tile_count": 0, "constant_or_nonfinite_count": 0}
    if calibration_hits: return (), receipt, ("not_evaluable", "not_evaluable_threshold_not_specific")
    raw = []; seen: set[tuple[str, str, str, str, str]] = set()
    for cell in cells:
        cell_id, roles = str(cell["cell_id"]), cell["records"]
        for left_role, right_role in prohibited_pairs:
            left_records, right_records = tuple(roles[left_role]), tuple(roles[right_role])
            receipt["scan_pair_count"] += len(left_records) * len(right_records)
            receipt["scan_tile_count"] += math.ceil(len(left_records) / 256) * math.ceil(len(right_records) / 256)
            found = scan_near_duplicates(left_records, right_records, threshold=_THRESHOLD, query_block_size=256, fit_block_size=256)
            for candidate in found:
                if candidate["distance"] is None:
                    receipt["constant_or_nonfinite_count"] += 1; continue
                key = (cell_id, left_role, right_role, str(candidate["left_record_id"]), str(candidate["right_record_id"]))
                if key in seen: continue
                seen.add(key)
                raw.append({**candidate, "boundary_id": boundary, "left_role_id": f"{left_role}@{cell_id}", "right_role_id": f"{right_role}@{cell_id}"})
    classified = list(classify_duplicate_candidates(boundary, raw))
    for row in classified:
        if row["distance"] == 0.0: row.update(status="fail", reason_code="fail_exact_duplicate_bridge")
    outcome = ("fail", "fail_exact_duplicate_bridge") if any(x["reason_code"] == "fail_exact_duplicate_bridge" for x in classified) else (("fail", "fail_near_duplicate_identity_bridge") if any(x["status"] == "fail" for x in classified) else (("warning_near_duplicate_bridge", "warning_near_duplicate_bridge") if classified else ("pass", "")))
    return tuple(classified), receipt, outcome


def _materialize_direct_roles(root: Path) -> tuple[dict[str, tuple[dict[str, object], ...]], dict[str, dict[str, tuple[dict[str, object], ...]]], dict[str, tuple[float, float]]]:
    from rpe.runner.d1_bacteria_id import _combine, _load_dataset, _stratified_split
    from rpe.runner.d2_bacteria_id import _read_selection, _selected_ids
    from rpe.downstream.sugar_quantitative import load_d4_sugar_cohort
    from rpe.downstream.rruff import load_d5_raw_cohort
    data, _ = _load_dataset(root / "data/unified/bacteria_id_reference", 4096)
    # Loader's shared wavenumber is retrieved directly to retain source orientation.
    from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
    with BacteriaIdBatchLoader(root / "data/unified/bacteria_id_reference", batch_size=4096) as loader: axis = next(loader.iter_batches()).wavenumber
    if not np.all(np.diff(axis) < 0):
        raise AppendixAuditsError("bacteria source axis", "retained source axis must be decreasing before source-view reversal")
    axis = axis[::-1]
    # Source rows are decreasing in this retained cohort; reverse values exactly once.
    def b(split, role): return tuple(_record(rid, rid, role, axis, row[::-1]) for rid, row in zip(split.record_ids, split.intensity, strict=True))
    d1_roles = {}
    for seed in range(5):
        fine, validation = _stratified_split(data["finetune"], seed=seed, validation_per_class=10, expected_class_count=30); train = _combine(data["reference"], fine)
        d1_roles.update({f"fit:{seed}": b(train, f"fit:{seed}"), f"selection:{seed}": b(validation, f"selection:{seed}"), f"test:{seed}": b(data["test"], f"test:{seed}")})
    selection, _ = _read_selection(root / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json")
    d2_roles = {}
    for seed in range(5):
        union = set()
        for shot in (5, 10, 20):
            train_ids, validation_ids, _ = _selected_ids(selection, seed=seed, shot_count=shot); union.update(train_ids); d2_roles[f"fit:{seed}:{shot}"] = tuple(r for r in b(data["finetune"], f"fit:{seed}:{shot}") if r["record_id"] in set(train_ids)); d2_roles[f"selection:{seed}:{shot}"] = tuple(r for r in b(data["finetune"], f"selection:{seed}:{shot}") if r["record_id"] in set(validation_ids))
        d2_roles[f"shot_union:{seed}"] = tuple(r for r in b(data["finetune"], f"shot_union:{seed}") if r["record_id"] in union); d2_roles[f"test:{seed}"] = b(data["test"], f"test:{seed}")
    d4 = load_d4_sugar_cohort(root / "experiments/phase05/configs/d4_sugar_protocol.json", root / "data/raw/ramanbench/cache/10779223/Raw data.zip")
    d4_roles = {}
    for split in d4.splits:
        for role, indices in (("fit", split.train_indices), ("selection", split.validation_indices), ("test", split.test_indices)):
            d4_roles[f"{role}:{split.seed}"] = tuple(_record(d4.record_ids[int(i)], d4.well_ids[int(i)], f"{role}:{split.seed}", d4.wavenumber, d4.intensity[int(i)]) for i in indices)
    d5 = load_d5_raw_cohort(root / "experiments/phase05/configs/d5_rruff_protocol.json", root / "data/unified/rruff_raman_raw")
    d5_roles = {}
    for split in d5.splits:
        for role, indices in (("query", split.query_indices), ("library", split.library_indices)):
            d5_roles[f"{role}:{split.seed}"] = tuple(_record(d5.record_ids[int(i)], d5.group_ids[int(i)], f"{role}:{split.seed}", d5.wavenumber, d5.intensity[int(i)]) for i in indices)
    ids = lambda records, field: {str(record[field]) for record in records}
    d1_cells = tuple({"cell_id": f"seed:{seed}", "roles": {"reference_plus_finetune_train": ids(d1_roles[f"fit:{seed}"], "record_id"), "validation": ids(d1_roles[f"selection:{seed}"], "record_id"), "test": ids(d1_roles[f"test:{seed}"], "record_id")}, "records": {"reference_plus_finetune_train": d1_roles[f"fit:{seed}"], "validation": d1_roles[f"selection:{seed}"], "test": d1_roles[f"test:{seed}"]}} for seed in range(5))
    d2_cells = tuple({"cell_id": f"seed:{seed}:shot:{shot}", "roles": {"training_selection": ids(d2_roles[f"fit:{seed}:{shot}"], "record_id"), "validation": ids(d2_roles[f"selection:{seed}:{shot}"], "record_id"), "frozen_test": ids(d2_roles[f"test:{seed}"], "record_id")}, "records": {"training_selection": d2_roles[f"fit:{seed}:{shot}"], "validation": d2_roles[f"selection:{seed}:{shot}"], "frozen_test": d2_roles[f"test:{seed}"]}} for seed in range(5) for shot in (5,10,20))
    d2_union_cells = tuple({"cell_id": f"seed:{seed}", "roles": {"protocol_b_training_selection": ids(d2_roles[f"shot_union:{seed}"], "record_id"), "protocol_b_validation": ids(d2_roles[f"selection:{seed}:20"], "record_id"), "frozen_test": ids(d2_roles[f"test:{seed}"], "record_id")}, "records": {"protocol_b_training_selection": d2_roles[f"shot_union:{seed}"], "protocol_b_validation": d2_roles[f"selection:{seed}:20"], "frozen_test": d2_roles[f"test:{seed}"]}} for seed in range(5))
    frozen_tests = [cell["roles"]["frozen_test"] for cell in d2_cells]
    if any(value != frozen_tests[0] for value in frozen_tests[1:]):
        raise AppendixAuditsError("d2 frozen test identity", "all 15 seed-shot cells must use one exact test set")
    for seed in range(5):
        train_sets = [ids(d2_roles[f"fit:{seed}:{shot}"], "record_id") for shot in (5, 10, 20)]
        if not (train_sets[0] <= train_sets[1] <= train_sets[2]):
            raise AppendixAuditsError("d2 protocol-b nesting", "5/10/20 selected training sets must be nested")
        union_cell = d2_union_cells[seed]["roles"]
        if union_cell["protocol_b_training_selection"] != train_sets[2] or union_cell["protocol_b_training_selection"] & union_cell["protocol_b_validation"] or union_cell["protocol_b_training_selection"] & union_cell["frozen_test"]:
            raise AppendixAuditsError("d2 protocol-b union", "union must equal 20-shot selection and remain disjoint from validation/test")
    d4_cells = tuple({"cell_id": f"fold:{seed}", "roles": {"train": ids(d4_roles[f"fit:{seed}"], "entity_id"), "validation": ids(d4_roles[f"selection:{seed}"], "entity_id"), "test": ids(d4_roles[f"test:{seed}"], "entity_id")}, "records": {"train": d4_roles[f"fit:{seed}"], "validation": d4_roles[f"selection:{seed}"], "test": d4_roles[f"test:{seed}"]}} for seed in range(5))
    d5_cells = tuple({"cell_id": f"split:{seed}", "roles": {"library": ids(d5_roles[f"library:{seed}"], "entity_id"), "query": ids(d5_roles[f"query:{seed}"], "entity_id")}, "records": {"library": d5_roles[f"library:{seed}"], "query": d5_roles[f"query:{seed}"]}} for seed in range(5))
    return {"d1_phase4_seed_role_split": d1_cells, "d2_phase4_test_identity": d2_cells, "d2_phase4_protocol_b_shot_union": d2_union_cells, "d4_phase4_fold_by_well": d4_cells, "d5_query_library_group_split": d5_cells}, {"d1": d1_roles, "d2": d2_roles, "d4": d4_roles, "d5": d5_roles}, {"d1": (387,1791), "d2": (387,1791), "d4": (146,3684), "d5": (204,1800)}


def build_phase6_leakage_audit(config_raw: Mapping[str, object], project_root: Path, worker_count: int) -> LeakageAuditResult:
    if worker_count <= 0: raise AppendixAuditsError("worker_count", "must be positive")
    for name in _THREAD_ENV: os.environ[name] = "1"
    root = Path(project_root); definitions = tuple(config_raw.get("leakage_boundary_definitions", ()))
    if len(definitions) != 15: raise AppendixAuditsError("leakage boundaries", "must contain exactly fifteen configured rows")
    # Parent authority validation is the mandatory inheritance gate.
    class _Config: raw = config_raw
    authorities = _validate_authorities(_Config(), root)
    phase2_receipt = _bind_phase2(root)
    phase3_receipt = _bind_phase3(root)
    direct_cells, _materialized, supports = _materialize_direct_roles(root)
    rows = [_base(definition) for definition in definitions]; candidates = []; receipts = {"authorities": authorities, "phase2_rruff": phase2_receipt, "phase3_denoising": phase3_receipt, "direct_cells": {}, "scans": {}}
    inherited = {"phase2_rruff_group_role_split", "phase3_denoising_fit_vs_transform", "phase6_baseline_d4_fit_selection_test_lifecycle", "phase6_denoising_identity_equivalence", "phase6_peak_claim_boundary"}
    for row in rows:
        boundary = str(row["boundary_id"])
        if boundary == "phase2_rruff_group_role_split":
            row.update(status="pass", reason_code="", recomputed_counts={"assignment_count": phase2_receipt["assignment_count"], "group_count": phase2_receipt["group_count"], **phase2_receipt["pair_counts"], **{f"group_{key}": value for key,value in phase2_receipt["group_counts"].items()}}, recomputed_digests={"ledger_sha256": phase2_receipt["ledger_sha256"]}, evidence_paths=phase2_receipt["evidence_paths"])
        elif boundary == "phase3_denoising_fit_vs_transform":
            row.update(status="pass_via_parent_hash", reason_code="", recomputed_counts={"system_count": 60, "fit_receipt_count": 24, "transform_receipt_count": 439200}, recomputed_digests={"ledger_sha256": phase3_receipt["ledger_sha256"], "complete_sha256": phase3_receipt["complete_sha256"], "config_sha256": phase3_receipt["config_sha256"]}, evidence_paths=phase3_receipt["evidence_paths"])
        elif boundary in inherited:
            paths = {
                "phase6_baseline_d4_fit_selection_test_lifecycle": (str(authorities["phase6_step3_baseline_evidence"]["path"]),),
                "phase6_denoising_identity_equivalence": (str(authorities["phase6_step4_denoising_evidence"]["path"]),),
                "phase6_peak_claim_boundary": (str(authorities["phase6_step5_peak_evidence"]["path"]), str(authorities["phase6_peak_checksum_ledger"]["path"])),
            }[boundary]
            row.update(status="pass_via_parent_hash", reason_code="", evidence_paths=paths)
        elif boundary == "cross_dataset_provenance_d1_d2_d4_d5": row.update(status="not_evaluable", reason_code="not_evaluable_insufficient_cross_dataset_identity")
        elif boundary in direct_cells:
            names = tuple(row["fit_roles"]) + tuple(row["selection_roles"]) + tuple(row["evaluation_roles"])
            pairs = tuple((names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names)))
            audit = evaluate_direct_cells(direct_cells[boundary], pairs, entity_key=str(row["leakage_entity"]))
            row["exact_overlap_count"] = audit["exact_overlap_count"]; row["recomputed_counts"] = {"cell_count": len(audit["cells"])}; row["recomputed_digests"] = {"cell_receipts_sha256": hashlib.sha256(json.dumps(audit["cells"], sort_keys=True, default=list).encode()).hexdigest()}; receipts["direct_cells"][boundary] = audit["cells"]
            if audit["exact_overlap_count"]: row.update(status="fail", reason_code="fail_role_overlap_bridge")
        elif boundary.endswith("exact_near_duplicate_bridge"):
            key = boundary.split("_", 1)[0]
            direct_boundary = {"d1": "d1_phase4_seed_role_split", "d2": "d2_phase4_test_identity", "d4": "d4_phase4_fold_by_well", "d5": "d5_query_library_group_split"}[key]
            names = {"d1": ("reference_plus_finetune_train", "validation", "test"), "d2": ("training_selection", "validation", "frozen_test"), "d4": ("train", "validation", "test"), "d5": ("library", "query")}[key]
            pairs = tuple((names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names)))
            fit_role = names[0]
            found, receipt, outcome = _scan_cells(boundary, direct_cells[direct_boundary], fit_role=fit_role, prohibited_pairs=pairs, support=supports[key]); candidates.extend(found); row.update(status=outcome[0], reason_code=outcome[1], near_candidate_count=len(found)); receipts["scans"][boundary] = receipt
        # Every boundary keeps a scalar row receipt; direct rows retain per-cell receipts above.
    candidates = tuple(sorted(candidates, key=lambda x: tuple(str(x[k]) for k in _CANDIDATE_FIELDS)))
    status_rows = tuple({"component": "leakage_boundaries", "status_order": index, "boundary_id": row["boundary_id"], "state": row["status"], "reason_code": row["reason_code"], "numerical_row_count": 0, "rank_row_count": 0, "evidence_key": "leakage_adapter"} for index,row in enumerate(rows))
    receipts["candidate_count"] = len(candidates); receipts["boundary_status"] = {str(row["boundary_id"]): str(row["status"]) for row in rows}
    return LeakageAuditResult(tuple(rows), candidates, status_rows, receipts)


__all__ = ["LeakageAuditResult", "build_phase6_leakage_audit", "classify_duplicate_candidates", "fixed_boundary_rows"]
