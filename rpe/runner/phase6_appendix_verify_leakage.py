"""Independent, verifier-only Step-7 leakage audit.

This adapter deliberately owns its role reconstruction, exact hashing and
blockwise duplicate scan.  It imports no production Step-7 implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from rpe.runner import phase6_appendix_verifier_science as science

_THREAD_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_DOMAIN = b"rpe-step7-exact-spectrum-v1"
_THRESHOLD = 1e-3
_FIELDS = (
    "boundary_id", "left_role_id", "right_role_id", "left_record_id",
    "right_record_id", "left_entity_id", "right_entity_id", "distance",
    "correlation", "left_source_sha256", "right_source_sha256", "status", "reason_code",
)


class VerifierLeakageError(ValueError):
    pass


@dataclass(frozen=True)
class VerifierLeakageResult:
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


def _base(definition: Mapping[str, object]) -> dict[str, object]:
    return {
        "boundary_id": str(definition["boundary_id"]), "producer_authorities": (), "consumer_authorities": (),
        "fit_roles": tuple(definition.get("fit_roles", ())), "selection_roles": tuple(definition.get("selection_roles", ())),
        "evaluation_roles": tuple(definition.get("evaluation_roles", ())), "leakage_entity": definition.get("leakage_entity", ""),
        "expected_overlap_semantics": definition.get("expected_overlap_semantics", ""), "recomputed_digests": {},
        "recomputed_counts": {}, "exact_overlap_count": 0, "near_candidate_count": 0,
        "status": "pass", "reason_code": "", "evidence_paths": (),
    }


def v_fixed_rows(definitions: Sequence[Mapping[str, object]], roles_by_boundary: Mapping[str, Mapping[str, set[str]]], *, near_status: Mapping[str, tuple[str, str]] | None = None) -> tuple[dict[str, object], ...]:
    """Return declared rows in order, classifying only within a boundary."""
    output = []
    for definition in definitions:
        row = _base(definition); names = tuple(row["fit_roles"]) + tuple(row["selection_roles"]) + tuple(row["evaluation_roles"])
        roles = roles_by_boundary.get(str(row["boundary_id"]), {})
        values = {name: set(str(value) for value in roles.get(name, set())) for name in names}
        overlap = sum(len(values[left] & values[right]) for index, left in enumerate(names) for right in names[index + 1:])
        row.update(recomputed_counts={name: len(values[name]) for name in names}, recomputed_digests={name: _digest(tuple(values[name])) for name in names}, exact_overlap_count=overlap)
        if overlap: row.update(status="fail", reason_code="fail_role_overlap_bridge")
        elif near_status and row["boundary_id"] in near_status: row["status"], row["reason_code"] = near_status[row["boundary_id"]]
        output.append(row)
    return tuple(output)


def v_direct_cells(cells: Sequence[Mapping[str, object]], prohibited_pairs: Sequence[tuple[str, str]]) -> dict[str, object]:
    receipts, total = [], 0
    for cell in cells:
        cell_id, supplied = str(cell["cell_id"]), cell["roles"]
        if not isinstance(supplied, Mapping): raise VerifierLeakageError("cell roles must be a mapping")
        roles = {str(name): {str(value) for value in values} for name, values in supplied.items()}
        counts = {}
        for left, right in prohibited_pairs:
            if not roles.get(left) or not roles.get(right): raise VerifierLeakageError(f"empty expected role in {cell_id}: {left}/{right}")
            counts[f"{left}|{right}"] = len(roles[left] & roles[right])
        total += sum(counts.values())
        receipts.append({"cell_id": cell_id, "role_counts": {key: len(value) for key, value in roles.items()}, "role_digests": {key: _digest(tuple(value)) for key, value in roles.items()}, "pair_overlap_counts": counts, "exact_overlap_count": sum(counts.values())})
    return {"exact_overlap_count": total, "cells": tuple(receipts)}


def _source_hash(axis: object, intensity: object) -> str:
    x, y = np.asarray(axis, dtype="<f8"), np.asarray(intensity, dtype="<f4")
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.all(np.diff(x) > 0):
        raise VerifierLeakageError("exact hash requires finite increasing source view")
    return hashlib.sha256(_DOMAIN + struct.pack("<Q", x.size) + x.tobytes() + struct.pack("<Q", y.size) + y.tobytes()).hexdigest()


def _record(record_id: str, entity_id: str, role_id: str, axis: object, intensity: object) -> dict[str, object]:
    x, y = np.asarray(axis, dtype="<f8"), np.asarray(intensity, dtype="<f4")
    return {"record_id": str(record_id), "entity_id": str(entity_id), "role_id": str(role_id), "axis": x, "intensity": y, "source_sha256": _source_hash(x, y)}


def _cached(record: Mapping[str, object], support: tuple[float, float]) -> dict[str, object]:
    target = np.arange(support[0], support[1] + .5, 1., dtype="<f8")
    intensity = science.v_resample(record["axis"], record["intensity"], target, "linear")
    return {**record, "axis": target, "intensity": intensity}


def _snv(value: object) -> np.ndarray | None:
    data = np.asarray(value, dtype="<f8")
    std = float(np.std(data, ddof=0))
    return None if not math.isfinite(std) or std <= 0 else np.ascontiguousarray((data - float(np.mean(data))) / std)


def _near(left: Mapping[str, object], right: Mapping[str, object]) -> tuple[float, float] | None:
    a, b = _snv(left["intensity"]), _snv(right["intensity"])
    if a is None or b is None: return None
    distance = float(np.sqrt(np.mean((a - b) ** 2)))
    return (distance, float(np.dot(a, b) / a.size)) if distance <= _THRESHOLD else None


def _calibration_pairs(records: Sequence[Mapping[str, object]]) -> tuple[tuple[str, str], ...]:
    ordered = tuple(sorted(records, key=lambda row: str(row["record_id"])))
    entity_counts: dict[str, int] = {}
    for row in ordered:
        entity = str(row["entity_id"]); entity_counts[entity] = entity_counts.get(entity, 0) + 1
    admissible_count = len(ordered) * (len(ordered) - 1) // 2 - sum(count * (count - 1) // 2 for count in entity_counts.values())
    if admissible_count < 100000:
        return tuple((str(left["record_id"]), str(right["record_id"])) for index, left in enumerate(ordered) for right in ordered[index + 1:] if left["entity_id"] != right["entity_id"])
    rng, selected, output = np.random.Generator(np.random.PCG64(20260817)), set(), []
    for _ in range(64):
        draws = rng.integers(0, len(ordered), size=(262144, 2))
        for first, second in draws:
            left, right = ordered[int(first)], ordered[int(second)]
            if first == second or left["entity_id"] == right["entity_id"]: continue
            pair = tuple(sorted((str(left["record_id"]), str(right["record_id"]))))
            if pair not in selected:
                selected.add(pair); output.append(pair)
                if len(output) == 100000: return tuple(output)
    raise VerifierLeakageError("calibration maximum batches did not fill target")


def v_classify_candidates(boundary_id: str, rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    output = []
    for supplied in rows:
        row = {field: supplied.get(field, "") for field in _FIELDS}; row["boundary_id"] = boundary_id
        bridge = row["left_entity_id"] == row["right_entity_id"] or (row["left_source_sha256"] and row["left_source_sha256"] == row["right_source_sha256"])
        row.update(status="fail" if bridge else "warning_near_duplicate_bridge", reason_code="fail_near_duplicate_identity_bridge" if bridge else "warning_near_duplicate_bridge")
        output.append(row)
    return tuple(sorted(output, key=lambda row: tuple(str(row[field]) for field in _FIELDS)))


def v_scan_cells(boundary_id: str, cells: Sequence[Mapping[str, object]], prohibited_pairs: Sequence[tuple[str, str]], support: tuple[float, float]) -> tuple[tuple[dict[str, object], ...], dict[str, object], tuple[str, str]]:
    """Exhaustive blockwise scan over only within-cell forbidden pairs."""
    by_id, fit = {}, []
    for cell in cells:
        roles = cell["records"]
        for records in roles.values():
            for record in records: by_id[str(record["record_id"])] = record
        fit.extend(next(iter(roles.values())))
    cached = {key: _cached(value, support) for key, value in by_id.items()}
    calibration = _calibration_pairs(tuple({str(row["record_id"]): cached[str(row["record_id"])] for row in fit}.values()))
    hits = sum(_near(cached[left], cached[right]) is not None for left, right in calibration)
    receipt = {"calibration_pair_count": len(calibration), "calibration_hit_count": hits, "scan_pair_count": 0, "scan_tile_count": 0, "constant_or_nonfinite_count": 0}
    if hits: return (), receipt, ("not_evaluable", "not_evaluable_threshold_not_specific")
    raw = []; seen: set[tuple[str, str, str, str, str]] = set()
    # Role blocks are streamed: no complete cross-role pair universe is retained.
    for cell in cells:
        cell_id, roles = str(cell["cell_id"]), cell["records"]
        for left_role, right_role in prohibited_pairs:
            left_records, right_records = tuple(roles[left_role]), tuple(roles[right_role])
            for left_start in range(0, len(left_records), 256):
                left_block = left_records[left_start:left_start + 256]
                for right_start in range(0, len(right_records), 256):
                    right_block = right_records[right_start:right_start + 256]
                    receipt["scan_tile_count"] += 1
                    receipt["scan_pair_count"] += len(left_block) * len(right_block)
                    left_ok = [(row, _snv(cached[str(row["record_id"])]["intensity"])) for row in left_block]
                    right_ok = [(row, _snv(cached[str(row["record_id"])]["intensity"])) for row in right_block]
                    receipt["constant_or_nonfinite_count"] += sum(vector is None for _, vector in left_ok) + sum(vector is None for _, vector in right_ok)
                    left_ok = [(row, vector) for row, vector in left_ok if vector is not None]; right_ok = [(row, vector) for row, vector in right_ok if vector is not None]
                    if not left_ok or not right_ok: continue
                    left_matrix = np.stack([vector for _, vector in left_ok]); right_matrix = np.stack([vector for _, vector in right_ok])
                    correlations = (left_matrix @ right_matrix.T) / left_matrix.shape[1]
                    for left_index, right_index in zip(*np.nonzero(np.maximum(0.0, 2.0 - 2.0 * correlations) <= _THRESHOLD * _THRESHOLD), strict=True):
                        left_raw, left_vector = left_ok[int(left_index)]; right_raw, right_vector = right_ok[int(right_index)]
                        distance = float(np.sqrt(np.mean((left_vector - right_vector) ** 2)))
                        if distance > _THRESHOLD: continue
                        key = (cell_id, left_role, right_role, str(left_raw["record_id"]), str(right_raw["record_id"]))
                        if key in seen: continue
                        seen.add(key)
                        left, right = cached[str(left_raw["record_id"])], cached[str(right_raw["record_id"])]
                        raw.append({"left_role_id": f"{left_role}@{cell_id}", "right_role_id": f"{right_role}@{cell_id}", "left_record_id": left["record_id"], "right_record_id": right["record_id"], "left_entity_id": left["entity_id"], "right_entity_id": right["entity_id"], "distance": distance, "correlation": float(correlations[int(left_index), int(right_index)]), "left_source_sha256": left["source_sha256"], "right_source_sha256": right["source_sha256"]})
    found = list(v_classify_candidates(boundary_id, raw))
    for row in found:
        if row["distance"] == 0.: row.update(status="fail", reason_code="fail_exact_duplicate_bridge")
    outcome = ("fail", "fail_exact_duplicate_bridge") if any(row["reason_code"] == "fail_exact_duplicate_bridge" for row in found) else (("fail", "fail_near_duplicate_identity_bridge") if any(row["status"] == "fail" for row in found) else (("warning_near_duplicate_bridge", "warning_near_duplicate_bridge") if found else ("pass", "")))
    return tuple(found), receipt, outcome


def _validate_authorities(config_raw: Mapping[str, object], root: Path) -> dict[str, Mapping[str, object]]:
    result = {}
    for name, values in dict(config_raw.get("authorities", {})).items():
        if not isinstance(values, Mapping): raise VerifierLeakageError(f"authority {name}: invalid descriptor")
        path = root / str(values["path"])
        if not path.is_file() or path.stat().st_size != int(values["byte_count"]) or _sha_file(path) != str(values["sha256"]): raise VerifierLeakageError(f"authority {name}: byte/hash mismatch")
        result[str(name)] = {"path": str(values["path"]), "sha256": str(values["sha256"])}
    return result


def _bind_phase2(root: Path) -> dict[str, object]:
    from rpe.semisynth import load_phase2_config, split_rruff_pairs
    config_path, pairs, raw = root / "experiments/phase2/configs/semisynth_v1.json", root / "data/unified/rruff_raman_pairs.jsonl", root / "data/unified/rruff_raman_raw"
    summary = split_rruff_pairs(pairs, raw, config=load_phase2_config(config_path, project_root=root), verify_checksums=True)
    roles: dict[str, set[str]] = {}
    for item in summary.assignments: roles.setdefault(item.group_key, set()).add(item.role.value)
    if summary.assignment_count != 15764 or summary.group_count != 3919 or any(len(value) != 1 for value in roles.values()): raise VerifierLeakageError("Phase-2 direct split drift")
    return {"assignment_count": summary.assignment_count, "group_count": summary.group_count, "ledger_sha256": summary.ledger_sha256, "evidence_paths": tuple(str(path.relative_to(root)) for path in (config_path, pairs, raw))}


def _bind_phase3(root: Path) -> dict[str, object]:
    run = root / "results/phase3/denoising_v1_formal_coverage/phase3-denoising-v1-formal-9f8b856433ee007b96c0fff204fa87db3216a7605ac98e6bca940e872f0db458"
    ledger = run / "SHA256SUMS"
    if not ledger.is_file(): raise VerifierLeakageError("Phase-3 checksum ledger missing")
    for line in ledger.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if _sha_file(run / name) != digest: raise VerifierLeakageError("Phase-3 ledger member mismatch")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or tuple(manifest.get(key) for key in ("system_count", "fit_receipt_count", "transform_receipt_count")) != (60, 24, 439200): raise VerifierLeakageError("Phase-3 terminal/count drift")
    return {"ledger_sha256": _sha_file(ledger), "system_count": 60, "fit_receipt_count": 24, "transform_receipt_count": 439200, "evidence_paths": tuple(str((run / name).relative_to(root)) for name in ("SHA256SUMS", "complete.json", "fit_receipts.jsonl", "transform_receipts.jsonl"))}


def _materialize_cells(root: Path):
    from rpe.runner.d1_bacteria_id import _combine, _load_dataset, _stratified_split
    from rpe.runner.d2_bacteria_id import _read_selection, _selected_ids
    from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
    from rpe.downstream.sugar_quantitative import load_d4_sugar_cohort
    from rpe.downstream.rruff import load_d5_raw_cohort
    data, _ = _load_dataset(root / "data/unified/bacteria_id_reference")
    with BacteriaIdBatchLoader(root / "data/unified/bacteria_id_reference", batch_size=4096) as loader: axis = next(loader.iter_batches()).wavenumber
    if not np.all(np.diff(axis) < 0): raise VerifierLeakageError("Bacteria source orientation drift")
    axis = axis[::-1]
    def bacteria(split, role): return tuple(_record(record_id, record_id, role, axis, intensity[::-1]) for record_id, intensity in zip(split.record_ids, split.intensity, strict=True))
    d1, d2 = [], []
    for seed in range(5):
        fine, validation = _stratified_split(data["finetune"], seed=seed, validation_per_class=10, expected_class_count=30)
        d1.append({"cell_id": f"seed:{seed}", "records": {"reference_plus_finetune_train": bacteria(_combine(data["reference"], fine), f"fit@{seed}"), "validation": bacteria(validation, f"validation@{seed}"), "test": bacteria(data["test"], f"test@{seed}")}})
    selection, _ = _read_selection(root / "results/phase05/d2/d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138/selection.json")
    d2b = []
    for seed in range(5):
        union, highest_validation, frozen = set(), None, bacteria(data["test"], f"test@{seed}")
        for shot in (5, 10, 20):
            train_ids, valid_ids, _ = _selected_ids(selection, seed=seed, shot_count=shot); union.update(train_ids); highest_validation = valid_ids
            material = bacteria(data["finetune"], f"d2@{seed}:{shot}")
            d2.append({"cell_id": f"seed:{seed}:shot:{shot}", "records": {"training_selection": tuple(row for row in material if row["record_id"] in set(train_ids)), "validation": tuple(row for row in material if row["record_id"] in set(valid_ids)), "frozen_test": frozen}})
        material = bacteria(data["finetune"], f"d2b@{seed}")
        d2b.append({"cell_id": f"seed:{seed}", "records": {"protocol_b_training_selection": tuple(row for row in material if row["record_id"] in union), "protocol_b_validation": tuple(row for row in material if row["record_id"] in set(highest_validation)), "frozen_test": frozen}})
    d4raw = load_d4_sugar_cohort(root / "experiments/phase05/configs/d4_sugar_protocol.json", root / "data/raw/ramanbench/cache/10779223/Raw data.zip")
    d4 = tuple({"cell_id": f"fold:{split.seed}", "records": {role: tuple(_record(d4raw.record_ids[int(i)], d4raw.well_ids[int(i)], f"{role}@{split.seed}", d4raw.wavenumber, d4raw.intensity[int(i)]) for i in indices) for role, indices in (("train", split.train_indices), ("validation", split.validation_indices), ("test", split.test_indices))}} for split in d4raw.splits)
    d5raw = load_d5_raw_cohort(root / "experiments/phase05/configs/d5_rruff_protocol.json", root / "data/unified/rruff_raman_raw")
    d5 = tuple({"cell_id": f"split:{split.seed}", "records": {role: tuple(_record(d5raw.record_ids[int(i)], d5raw.group_ids[int(i)], f"{role}@{split.seed}", d5raw.wavenumber, d5raw.intensity[int(i)]) for i in indices) for role, indices in (("library", split.library_indices), ("query", split.query_indices))}} for split in d5raw.splits)
    return {"d1_phase4_seed_role_split": tuple(d1), "d2_phase4_test_identity": tuple(d2), "d2_phase4_protocol_b_shot_union": tuple(d2b), "d4_phase4_fold_by_well": d4, "d5_query_library_group_split": d5}


def _roles(cells: Sequence[Mapping[str, object]], entity: bool) -> tuple[dict[str, object], ...]:
    return tuple({"cell_id": cell["cell_id"], "roles": {role: {str(row["entity_id"] if entity else row["record_id"]) for row in rows} for role, rows in cell["records"].items()}} for cell in cells)


def v_build_leakage(config_raw: Mapping[str, object], project_root: Path, worker_count: int) -> VerifierLeakageResult:
    if not isinstance(worker_count, int) or worker_count <= 0: raise VerifierLeakageError("worker_count must be positive")
    for name in _THREAD_ENV: os.environ[name] = "1"
    root, definitions = Path(project_root), tuple(config_raw.get("leakage_boundary_definitions", ()))
    if len(definitions) != 15: raise VerifierLeakageError("expected exactly fifteen leakage boundaries")
    authorities, phase2, phase3 = _validate_authorities(config_raw, root), _bind_phase2(root), _bind_phase3(root)
    cells = _materialize_cells(root); rows, candidates, direct_receipts, scan_receipts = [], [], {}, {}
    role_pairs = {"d1_phase4_seed_role_split": (("reference_plus_finetune_train", "validation"), ("reference_plus_finetune_train", "test"), ("validation", "test")), "d2_phase4_test_identity": (("training_selection", "validation"), ("training_selection", "frozen_test"), ("validation", "frozen_test")), "d2_phase4_protocol_b_shot_union": (("protocol_b_training_selection", "protocol_b_validation"), ("protocol_b_training_selection", "frozen_test"), ("protocol_b_validation", "frozen_test")), "d4_phase4_fold_by_well": (("train", "validation"), ("train", "test"), ("validation", "test")), "d5_query_library_group_split": (("library", "query"),)}
    supports = {"d1": (387., 1791.), "d2": (387., 1791.), "d4": (146., 3684.), "d5": (204., 1800.)}
    for definition in definitions:
        row, boundary = _base(definition), str(definition["boundary_id"])
        if boundary == "phase2_rruff_group_role_split": row.update(status="pass", recomputed_counts={"assignment_count": phase2["assignment_count"], "group_count": phase2["group_count"]}, recomputed_digests={"ledger_sha256": phase2["ledger_sha256"]}, evidence_paths=phase2["evidence_paths"])
        elif boundary == "phase3_denoising_fit_vs_transform": row.update(status="pass_via_parent_hash", recomputed_counts={key: phase3[key] for key in ("system_count", "fit_receipt_count", "transform_receipt_count")}, recomputed_digests={"ledger_sha256": phase3["ledger_sha256"]}, evidence_paths=phase3["evidence_paths"])
        elif boundary in {"phase6_baseline_d4_fit_selection_test_lifecycle", "phase6_denoising_identity_equivalence", "phase6_peak_claim_boundary"}:
            key = {"phase6_baseline_d4_fit_selection_test_lifecycle": "phase6_step3_baseline_evidence", "phase6_denoising_identity_equivalence": "phase6_step4_denoising_evidence", "phase6_peak_claim_boundary": "phase6_step5_peak_evidence"}[boundary]
            row.update(status="pass_via_parent_hash", evidence_paths=(authorities[key]["path"],))
        elif boundary == "cross_dataset_provenance_d1_d2_d4_d5": row.update(status="not_evaluable", reason_code="not_evaluable_insufficient_cross_dataset_identity")
        elif boundary in cells:
            audit = v_direct_cells(_roles(cells[boundary], entity=boundary in {"d4_phase4_fold_by_well", "d5_query_library_group_split"}), role_pairs[boundary]); direct_receipts[boundary] = audit["cells"]
            row.update(exact_overlap_count=audit["exact_overlap_count"], recomputed_counts={"cell_count": len(audit["cells"])}, recomputed_digests={"cell_receipts_sha256": hashlib.sha256(json.dumps(audit["cells"], sort_keys=True).encode()).hexdigest()})
            if audit["exact_overlap_count"]: row.update(status="fail", reason_code="fail_role_overlap_bridge")
        else:
            prefix, direct = boundary.split("_", 1)[0], {"d1": "d1_phase4_seed_role_split", "d2": "d2_phase4_test_identity", "d4": "d4_phase4_fold_by_well", "d5": "d5_query_library_group_split"}[boundary.split("_", 1)[0]]
            pairs = role_pairs[direct]; found, receipt, outcome = v_scan_cells(boundary, cells[direct], pairs, supports[prefix]); candidates.extend(found); scan_receipts[boundary] = receipt; row.update(status=outcome[0], reason_code=outcome[1], near_candidate_count=len(found))
        rows.append(row)
    candidates = tuple(sorted(candidates, key=lambda row: tuple(str(row[field]) for field in _FIELDS)))
    statuses = tuple({"component": "leakage_boundaries", "status_order": index, "boundary_id": row["boundary_id"], "state": row["status"], "reason_code": row["reason_code"], "numerical_row_count": 0, "rank_row_count": 0, "evidence_key": "verifier_leakage"} for index, row in enumerate(rows))
    identity = MappingProxyType({"authorities": MappingProxyType(authorities), "phase2": MappingProxyType(phase2), "phase3": MappingProxyType(phase3), "direct_cells": MappingProxyType(direct_receipts), "scans": MappingProxyType(scan_receipts), "candidate_count": len(candidates), "boundary_status": MappingProxyType({str(row["boundary_id"]): str(row["status"]) for row in rows})})
    return VerifierLeakageResult(tuple(rows), candidates, statuses, identity)


__all__ = ["VerifierLeakageResult", "v_build_leakage", "v_classify_candidates", "v_direct_cells", "v_fixed_rows", "v_scan_cells"]
