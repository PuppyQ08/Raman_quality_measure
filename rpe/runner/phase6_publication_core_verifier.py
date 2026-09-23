from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rpe.runner.phase6_publication_core_authority import (
    ARTIFACT_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    CODE_RECEIPT_PATHS,
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    FIXED_AUTHORITY_RUNS,
    FIGURE3_FIELDS,
    FIGURE4_EDGE_FIELDS,
    FIGURE4_NODE_FIELDS,
    PAYLOAD_FILES,
    REPORT_AUTHORITY_PATHS,
    ROOT,
    RasterCanvas,
    SCHEMA_VERSION,
    STEP7_DEFERRED_REPORT_STATE,
    SELECTED_PARENT_PAYLOAD_SPECS,
    TABLE1_FIELDS,
    TABLE2_FIELDS,
    TABLE_S3_FIELDS,
    canonical_json_bytes,
    csv_bytes,
    environment_receipt,
    file_identity,
    markdown_table,
    sha256_file,
    sha256_hex,
    svg_bytes,
    stable_run_id,
    validate_step7_deferred_report,
    write_sha256sums,
)


class PublicationCoreVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class PublicationCoreVerificationSummary:
    path: Path
    run_id: str
    status: str
    verified_file_count: int


@dataclass(frozen=True)
class _VerifierConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    artifact_payload_files: tuple[str, ...]
    expected: Mapping[str, int]
    figure3: Mapping[str, object]
    figure4: Mapping[str, object]
    report_authorities: Mapping[str, Mapping[str, object]]
    code_receipts: Mapping[str, Mapping[str, object]]
    step7_dependency: Mapping[str, object]


@dataclass(frozen=True)
class _VerifierInputs:
    table1_rows: tuple[Mapping[str, object], ...]
    table_s1_rows: tuple[Mapping[str, object], ...]
    table_s2_rows: tuple[Mapping[str, object], ...]
    table2_rows: tuple[Mapping[str, object], ...]
    table_s3_rows: tuple[Mapping[str, object], ...]
    figure3_rows: tuple[Mapping[str, object], ...]
    figure4_node_rows: tuple[Mapping[str, object], ...]
    figure4_edge_rows: tuple[Mapping[str, object], ...]
    captions: Mapping[str, str]
    authority_bridge: Mapping[str, object]


_TABLE1_ENDPOINT_SPECS = (
    ("d5_a", "d5", "a", "D5-A", "class", "681"),
    ("d5_b", "d5", "b", "D5-B", "class", "681"),
    ("d2_5_a", "d2_5", "a", "D2-5-A", "class", "30"),
    ("d2_5_b", "d2_5", "b", "D2-5-B", "class", "30"),
    ("d2_10_a", "d2_10", "a", "D2-10-A", "class", "30"),
    ("d2_10_b", "d2_10", "b", "D2-10-B", "class", "30"),
    ("d2_20_a", "d2_20", "a", "D2-20-A", "class", "30"),
    ("d2_20_b", "d2_20", "b", "D2-20-B", "class", "30"),
    ("d1_a", "d1", "a", "D1-A", "patient", "30"),
    ("d1_b_closed", "d1", "b", "D1-B-closed", "patient", "30"),
    ("d4_a", "d4", "a", "D4-A", "well", "240"),
    ("d4_b", "d4", "b", "D4-B", "well", "240"),
)

_FIGURE4_NODE_BLUEPRINTS = (
    ("phase0_audit", "Phase 0 audit", "complete", "#3b6699", "phase0_source_audit_report"),
    ("phase0_5_original_gate", "Phase 0.5 original gate", "not_met", "#b0533d", "phase05_internal_screening_decision_report"),
    ("phase0_5_internal_screening", "Internal screening continuation", "authorized_fallback", "#8e5da1", "phase05_internal_screening_decision_report"),
    ("phase1_metrics", "Phase 1 metrics/operators", "complete_with_p1_p5_and_p6_p7_closures", "#518772", "phase1_core10k_report"),
    ("phase2_model_adequacy", "Phase 2 model adequacy", "5/12 pass; 7/12 fail", "#b58730", "phase2_background_fit"),
    ("phase3_methods", "Phase 3 method states", "baseline 112/200 and 8/10 not_powered_for_phase5; denoising/peak applicability_only; DL 0 runnable", "#6c76a4", "phase3_status_report"),
    ("phase4_outcomes", "Phase 4 outcomes", "primary_not_evaluable_figures_complete", "#48848e", "phase4_closure_synthesis_report"),
    ("phase5_power", "Phase 5 power", "not_powered_for_phase5_not_run", "#88704a", "phase6_baseline_evidence"),
    ("phase6_publication", "Phase 6 publication", "evidence_first_publication_package", "#627b62", "phase6_step1_design_report"),
)

_FIGURE4_EDGE_BLUEPRINTS = (
    ("edge-0", "phase0_audit", "phase0_5_original_gate", "solid", "preregistered"),
    ("edge-1", "phase0_5_original_gate", "phase0_5_internal_screening", "dashed", "authorized fallback"),
    ("edge-2", "phase0_5_internal_screening", "phase1_metrics", "solid", "continued"),
    ("edge-3", "phase1_metrics", "phase2_model_adequacy", "solid", "dependency"),
    ("edge-4", "phase2_model_adequacy", "phase3_methods", "dashed", "three-part indirect evidence"),
    ("edge-5", "phase3_methods", "phase4_outcomes", "solid", "dependency"),
    ("edge-6", "phase4_outcomes", "phase5_power", "solid", "dependency"),
    ("edge-7", "phase5_power", "phase6_publication", "dashed", "status-aware package"),
)

_CAPTIONS = {
    "table1": "Task-native metric validity across the twelve frozen endpoint and protocol rows; D1-B-closed retains the failed alpha-zero equivalence closure.",
    "table2": "Long-form method evidence from the frozen Phase 6 baseline, denoising, and peak protocols; empty numerical cells are typed closures rather than imputed performance.",
    "figure3": "Figure 3 replacement: a 3x4 model-adequacy matrix from the frozen Phase 2 failed run with median, p95, and PASS or FAIL state per cell.",
    "figure4": "Figure 4 replacement: an evidence-flow and gate map in which solid arrows are preregistered dependencies and dashed arrows are authorized fallback or status-aware transitions.",
}


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PublicationCoreVerificationError(f"{path}: must be an object")
    return value


def _rows(name: str, value: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    rows = tuple(value)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PublicationCoreVerificationError(f"{name}[{index}]: must be an object")
    return rows


def _count_key(path: str, value: object) -> int:
    if not isinstance(value, int):
        raise PublicationCoreVerificationError(f"{path}: must be an integer")
    return value


def _csv_rows_from_path(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _receipt_matches(label: str, actual_path: Path, receipt_value: object) -> None:
    receipt = _object(label, receipt_value)
    actual = file_identity(actual_path)
    expected_path = receipt.get("path")
    if expected_path not in (None, actual["path"]):
        raise PublicationCoreVerificationError(f"{label}.path: frozen receipt mismatch")
    if receipt.get("byte_count") != actual["byte_count"]:
        raise PublicationCoreVerificationError(f"{label}.byte_count: frozen receipt mismatch")
    if receipt.get("sha256") != actual["sha256"]:
        raise PublicationCoreVerificationError(f"{label}.sha256: frozen receipt mismatch")


def _sha256sums_entries(run_path: Path) -> dict[str, str]:
    ledger = (run_path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    entries: dict[str, str] = {}
    for line in ledger:
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        entries[name] = digest
    return entries


def _validate_selected_parent_payloads(label: str, run_path: Path, *, spec_key: str) -> None:
    entries = _sha256sums_entries(run_path)
    spec = SELECTED_PARENT_PAYLOAD_SPECS[spec_key]
    selected_files = tuple(str(name) for name in spec["selected_files"])
    terminal_name = str(spec["terminal_name"])
    for name in selected_files:
        file_path = run_path / name
        if not file_path.is_file():
            raise PublicationCoreVerificationError(f"{label}: missing selected payload {name}")
        actual_sha = sha256_file(file_path)
        if entries.get(name) != actual_sha:
            raise PublicationCoreVerificationError(f"{label}: SHA256SUMS mismatch for {name}")
    terminal_path = run_path / terminal_name
    if not terminal_path.is_file():
        raise PublicationCoreVerificationError(f"{label}: missing terminal payload {terminal_name}")
    terminal = json.loads((run_path / terminal_name).read_text(encoding="utf-8"))
    if _object(f"{label}.{terminal_name}", terminal).get("status") != ("failed" if terminal_name == "failed.json" else "complete"):
        raise PublicationCoreVerificationError(f"{label}: unexpected terminal status")


def _resolve_frozen_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = ROOT / path
    return path


def _fixed_authority_paths(config: _VerifierConfig) -> tuple[dict[str, Path], Mapping[str, object]]:
    authorities_doc = _object("config.authorities", config.document.get("authorities"))
    resolved: dict[str, Path] = {}
    for key, relative_path in FIXED_AUTHORITY_RUNS.items():
        authority_path = ROOT / relative_path
        _receipt_matches(f"config.authorities.{key}", authority_path, authorities_doc.get(key))
        _validate_selected_parent_payloads(f"config.authorities.{key}", authority_path, spec_key=key)
        resolved[key] = authority_path
    deferred_report_receipt = _object(
        "config.step7_dependency.deferred_report",
        config.step7_dependency.get("deferred_report"),
    )
    deferred_report_path = _resolve_frozen_path(deferred_report_receipt.get("path"))
    validate_step7_deferred_report(deferred_report_path, error_type=PublicationCoreVerificationError)
    _receipt_matches(
        "config.step7_dependency.deferred_report",
        deferred_report_path,
        deferred_report_receipt,
    )
    for key, relative_path in REPORT_AUTHORITY_PATHS.items():
        _receipt_matches(f"config.report_authorities.{key}", ROOT / relative_path, config.report_authorities.get(key))
    for key, relative_path in CODE_RECEIPT_PATHS.items():
        _receipt_matches(f"config.code_receipts.{key}", ROOT / relative_path, config.code_receipts.get(key))
    return resolved, {
        "mode": "deferred_report",
        "state": STEP7_DEFERRED_REPORT_STATE,
        "deferred_report": file_identity(deferred_report_path),
    }


def _metric_vote_sets(rows: Sequence[Mapping[str, str]]) -> tuple[str, str, str]:
    jointly: list[str] = []
    ag_only: list[str] = []
    acc_only: list[str] = []
    for row in rows:
        metric_id = str(row["metric_output_id"])
        ag_rejected = str(row.get("d_ag_rejected", "")) == "True"
        acc_rejected = str(row.get("d_acc_rejected", "")) == "True"
        ag_favorable = str(row.get("d_ag_favorable", "")) == "True"
        acc_favorable = str(row.get("d_acc_favorable", "")) == "True"
        if ag_rejected and ag_favorable and acc_rejected and acc_favorable:
            jointly.append(metric_id)
        elif ag_rejected and ag_favorable:
            ag_only.append(metric_id)
        elif acc_rejected and acc_favorable:
            acc_only.append(metric_id)
    return ";".join(jointly), ";".join(ag_only), ";".join(acc_only)


def _derive_table1_rows(phase4_path: Path) -> tuple[Mapping[str, object], ...]:
    alignment_path = phase4_path / "figure2_alignment_data.csv"
    parent_index_path = phase4_path / "parent_panel_index.csv"
    alignment_rows = _csv_rows_from_path(alignment_path)
    parent_rows = _csv_rows_from_path(parent_index_path)
    parent_by_panel = {str(row["panel_id"]): row for row in parent_rows}
    alignment_sha = sha256_file(alignment_path)
    parent_sha = sha256_file(parent_index_path)
    rows: list[Mapping[str, object]] = []
    for panel_id, endpoint_key, protocol_id, label, cluster_kind, cluster_count in _TABLE1_ENDPOINT_SPECS:
        parent_row = parent_by_panel.get(panel_id)
        if parent_row is None:
            raise PublicationCoreVerificationError(f"table1: missing parent panel row for {panel_id}")
        if str(parent_row.get("endpoint_id")) != endpoint_key:
            raise PublicationCoreVerificationError(f"table1: endpoint mismatch for {panel_id}")
        if str(parent_row.get("protocol_id")) != protocol_id:
            raise PublicationCoreVerificationError(f"table1: protocol mismatch for {panel_id}")
        if panel_id == "d1_b_closed":
            if str(parent_row.get("state")) != "not_evaluable_failed_alpha0_equivalence":
                raise PublicationCoreVerificationError("table1: unexpected d1_b_closed closure state")
            rows.append(
                {
                    "endpoint_id": label,
                    "protocol_id": protocol_id,
                    "protocol_label": label,
                    "tier": "core",
                    "perturbation_scope": "P8-P12",
                    "cluster_kind": cluster_kind,
                    "cluster_count": cluster_count,
                    "state": "not_evaluable_failed_alpha0_equivalence",
                    "mse_ag": "",
                    "mse_ag_lower": "",
                    "mse_ag_upper": "",
                    "mse_acc_cross": "",
                    "mse_acc_cross_lower": "",
                    "mse_acc_cross_upper": "",
                    "jointly_favorable_metric_ids": "",
                    "ag_only_favorable_metric_ids": "",
                    "acc_only_favorable_metric_ids": "",
                    "holm_family_id": "d1:b:within_protocol",
                    "holm_family_size": "24",
                    "claim_scope": "within_protocol_family_only",
                    "source_path": "parent_panel_index.csv",
                    "source_sha256": parent_sha,
                }
            )
            continue
        panel_rows = [row for row in alignment_rows if str(row["panel_id"]) == panel_id]
        if len(panel_rows) != 13:
            raise PublicationCoreVerificationError(f"table1: expected 13 metric rows for {panel_id}")
        if any(str(row.get("endpoint_id")) != endpoint_key for row in panel_rows):
            raise PublicationCoreVerificationError(f"table1: alignment endpoint mismatch for {panel_id}")
        if any(str(row.get("protocol_id")) != protocol_id for row in panel_rows):
            raise PublicationCoreVerificationError(f"table1: alignment protocol mismatch for {panel_id}")
        mse_row = next((row for row in panel_rows if str(row["metric_output_id"]) == "mse"), None)
        if mse_row is None:
            raise PublicationCoreVerificationError(f"table1: missing mse row for {panel_id}")
        jointly, ag_only, acc_only = _metric_vote_sets(panel_rows)
        rows.append(
            {
                "endpoint_id": label,
                "protocol_id": protocol_id,
                "protocol_label": label,
                "tier": "core",
                "perturbation_scope": "P8-P12",
                "cluster_kind": cluster_kind,
                "cluster_count": cluster_count,
                "state": "complete",
                "mse_ag": float(mse_row["ag"]),
                "mse_ag_lower": float(mse_row["ag_lower"]),
                "mse_ag_upper": float(mse_row["ag_upper"]),
                "mse_acc_cross": float(mse_row["acc_cross"]),
                "mse_acc_cross_lower": float(mse_row["acc_cross_lower"]),
                "mse_acc_cross_upper": float(mse_row["acc_cross_upper"]),
                "jointly_favorable_metric_ids": jointly,
                "ag_only_favorable_metric_ids": ag_only,
                "acc_only_favorable_metric_ids": acc_only,
                "holm_family_id": str(mse_row["d_ag_family_id"]),
                "holm_family_size": str(mse_row["d_ag_family_size"]),
                "claim_scope": "within_protocol_family_only",
                "source_path": "figure2_alignment_data.csv",
                "source_sha256": alignment_sha,
            }
        )
    return tuple(rows)


def _component_state(rows: Sequence[Mapping[str, str]], default: str) -> str:
    if not rows:
        return default
    for row in rows:
        if str(row.get("state")) == "complete":
            return "complete"
    return str(rows[0].get("state", default))


def _classical_reason_code(rows: Sequence[Mapping[str, str]], fallback: str) -> str:
    for row in rows:
        reason = str(row.get("reason_code", ""))
        if reason:
            return reason
    return fallback


def _derive_table_s3_rows(
    *,
    table2_rows: Sequence[Mapping[str, object]],
    baseline_status_path: Path,
    denoising_status_path: Path,
    peak_status_path: Path,
    dl_table_path: Path,
    classical_catalog_path: Path,
) -> tuple[Mapping[str, object], ...]:
    baseline_status_sha = sha256_file(baseline_status_path)
    denoising_status_sha = sha256_file(denoising_status_path)
    peak_status_sha = sha256_file(peak_status_path)
    dl_sha = sha256_file(dl_table_path)
    catalog_sha = sha256_file(classical_catalog_path)
    catalog = _object("classical_catalog", json.loads(classical_catalog_path.read_text(encoding="utf-8")))
    systems = catalog.get("systems")
    if not isinstance(systems, list) or len(systems) != 306:
        raise PublicationCoreVerificationError("table_s3: classical catalog must contain 306 systems")
    evidence_by_system: dict[str, list[Mapping[str, str]]] = {}
    for row in table2_rows:
        system_id = str(row["system_id"])
        evidence_by_system.setdefault(system_id, []).append({str(key): str(value) for key, value in row.items()})
    classical_rows: list[Mapping[str, object]] = []
    for system in systems:
        item = _object("classical_system", system)
        task_line = str(item["task_line"])
        system_id = str(item["system_id"])
        family_id = str(item["family_id"])
        evidence_rows = evidence_by_system.get(system_id, [])
        if evidence_rows:
            by_component: dict[str, list[Mapping[str, str]]] = {}
            for row in evidence_rows:
                by_component.setdefault(str(row["evidence_component"]), []).append(row)
            source_path = {
                "baseline_correction": baseline_status_path.name,
                "denoising": denoising_status_path.name,
                "peak_detection": peak_status_path.name,
            }[task_line]
            source_sha = {
                "baseline_correction": baseline_status_sha,
                "denoising": denoising_status_sha,
                "peak_detection": peak_status_sha,
            }[task_line]
            direct_gt_default = {
                "baseline_correction": "not_evaluable_no_validated_clean_gt",
                "denoising": "not_evaluated_no_defensible_clean_target",
                "peak_detection": "not_evaluated_no_real_peak_assignments",
            }[task_line]
            classical_rows.append(
                {
                    "task_line": task_line,
                    "system_id": system_id,
                    "family_id": family_id,
                    "planned_K": "1",
                    "runnable_K": "1",
                    "coverage_promoted_K": "1",
                    "coverage_state": "coverage_promoted",
                    "direct_gt_state": _component_state(by_component.get("direct_gt", ()), direct_gt_default),
                    "downstream_state": _component_state(by_component.get("downstream", ()), "not_evaluable_no_common_consumer"),
                    "reference_free_state": _component_state(by_component.get("reference_free", ()), "not_evaluable_no_common_consumer"),
                    "half_split_state": _component_state(by_component.get("half_split", ()), "not_applicable"),
                    "phase5_eligible_K": "1",
                    "phase5_power_state": "not_powered_for_phase5",
                    "publication_disposition": "method_status_only",
                    "reason_code": _classical_reason_code(evidence_rows, "not_powered_for_phase5"),
                    "source_path": source_path,
                    "source_sha256": source_sha,
                }
            )
        else:
            classical_rows.append(
                {
                    "task_line": task_line,
                    "system_id": system_id,
                    "family_id": family_id,
                    "planned_K": "1",
                    "runnable_K": "0",
                    "coverage_promoted_K": "0",
                    "coverage_state": "status_only_nonpromoted",
                    "direct_gt_state": "not_evaluable_no_validated_clean_gt",
                    "downstream_state": "not_evaluable_no_common_consumer",
                    "reference_free_state": "not_evaluable_no_common_consumer",
                    "half_split_state": "not_applicable",
                    "phase5_eligible_K": "0",
                    "phase5_power_state": "not_powered_for_phase5",
                    "publication_disposition": "reported_only",
                    "reason_code": "not_promoted",
                    "source_path": classical_catalog_path.name,
                    "source_sha256": catalog_sha,
                }
            )
    dl_rows = _csv_rows_from_path(dl_table_path)
    if len(dl_rows) != 10:
        raise PublicationCoreVerificationError("table_s3: expected 10 deep-learning audit rows")
    for row in dl_rows:
        classical_rows.append(
            {
                "task_line": "deep_learning",
                "system_id": str(row["candidate_id"]),
                "family_id": str(row["family_id"]),
                "planned_K": "1",
                "runnable_K": "0",
                "coverage_promoted_K": "0",
                "coverage_state": "reported_only",
                "direct_gt_state": "not_applicable",
                "downstream_state": str(row.get("deepr_reproduction_state", "not_applicable")),
                "reference_free_state": "not_applicable",
                "half_split_state": "not_applicable",
                "phase5_eligible_K": "0",
                "phase5_power_state": "not_powered_for_phase5",
                "publication_disposition": str(row["publication_disposition"]),
                "reason_code": "not_reproducible_under_frozen_audit",
                "source_path": dl_table_path.name,
                "source_sha256": dl_sha,
            }
        )
    if len(classical_rows) != 316:
        raise PublicationCoreVerificationError("table_s3: row-count mismatch")
    return tuple(classical_rows)


def _derive_figure3_model_adequacy_rows(phase2_run_path: Path | str) -> tuple[Mapping[str, object], ...]:
    run_path = Path(phase2_run_path)
    model_receipts = _object("model_receipts", json.loads((run_path / "model_receipts.json").read_text(encoding="utf-8")))
    gate = _object("gate", json.loads((run_path / "gate.json").read_text(encoding="utf-8")))
    models = model_receipts.get("models")
    model_gates = gate.get("model_gates")
    if not isinstance(models, list) or not isinstance(model_gates, list) or len(models) != 12 or len(model_gates) != 12:
        raise PublicationCoreVerificationError("figure3 authorities: expected 12 model rows and 12 gate rows")
    gate_by_key = {
        (str(row["extractor_id"]), str(row["excitation_stratum"])): row
        for row in model_gates
    }
    rows: list[Mapping[str, object]] = []
    for row in models:
        extractor_id = str(row["extractor_id"])
        excitation = str(row["excitation_stratum"])
        gate_row = gate_by_key[(extractor_id, excitation)]
        failures = tuple(str(item) for item in gate_row.get("gate_failures", ()))
        p95_state = "pass" if not failures else "fail"
        rows.append(
            {
                "extractor_id": extractor_id,
                "excitation_stratum": excitation,
                "input_count": int(row["input_count"]),
                "valid_count": int(row["valid_count"]),
                "invalid_count": int(row["invalid_count"]),
                "median_reconstruction_nrmse": float(row["median_reconstruction_nrmse"]),
                "p95_reconstruction_nrmse": float(row["p95_reconstruction_nrmse"]),
                "p95_threshold": 0.25,
                "valid_fraction_state": "complete",
                "valid_count_state": "complete",
                "median_state": "complete",
                "p95_state": p95_state,
                "gate_state": "pass" if p95_state == "pass" else "fail_p95",
                "source_receipt_sha256": sha256_hex(
                    canonical_json_bytes(
                        {
                            "extractor_id": extractor_id,
                            "excitation_stratum": excitation,
                            "median_reconstruction_nrmse": row["median_reconstruction_nrmse"],
                            "p95_reconstruction_nrmse": row["p95_reconstruction_nrmse"],
                            "gate_failures": list(failures),
                        }
                    )
                ),
            }
        )
    if sum(int(row["valid_count"]) for row in rows) != 28092:
        raise PublicationCoreVerificationError("figure3: valid_count total mismatch")
    if sum(int(row["input_count"]) for row in rows) != 28092:
        raise PublicationCoreVerificationError("figure3: input_count total mismatch")
    if sum(row["gate_state"] == "pass" for row in rows) != 5:
        raise PublicationCoreVerificationError("figure3: pass-count mismatch")
    if sum(row["gate_state"] != "pass" for row in rows) != 7:
        raise PublicationCoreVerificationError("figure3: fail-count mismatch")
    return tuple(sorted(rows, key=lambda item: (str(item["extractor_id"]), str(item["excitation_stratum"]))))


def _derive_figure4_rows(
    *,
    step7_dependency: Mapping[str, object],
    authorities: Mapping[str, Path],
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    receipts = {
        **{key: file_identity(path) for key, path in authorities.items()},
        "step7_appendix_audits": _object("step7_dependency.deferred_report", step7_dependency["deferred_report"]),
        **{
            key: file_identity(ROOT / relative_path)
            for key, relative_path in REPORT_AUTHORITY_PATHS.items()
        },
    }
    node_rows = []
    for index, (node_id, label, state, color, receipt_key) in enumerate(_FIGURE4_NODE_BLUEPRINTS):
        receipt = receipts[receipt_key]
        node_rows.append(
            {
                "node_id": node_id,
                "label": label,
                "state": state,
                "shape": "box",
                "color": color,
                "authority_path": str(receipt["path"]),
                "authority_sha256": str(receipt["sha256"]),
                "x": str(index % 3),
                "y": str(index // 3),
            }
        )
    edge_rows = []
    phase6_receipt = receipts["phase6_step1_design_report"]
    for edge_id, source_node_id, target_node_id, edge_style, label in _FIGURE4_EDGE_BLUEPRINTS:
        edge_rows.append(
            {
                "edge_id": edge_id,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "edge_style": edge_style,
                "label": label,
                "authority_path": str(phase6_receipt["path"]),
                "authority_sha256": str(phase6_receipt["sha256"]),
            }
        )
    return tuple(node_rows), tuple(edge_rows)


def _derive_real_inputs(config: _VerifierConfig) -> _VerifierInputs:
    authorities, step7_dependency = _fixed_authority_paths(config)
    phase4_path = authorities["phase4_final_figures"]
    baseline_path = authorities["phase6_baseline_evidence"]
    denoising_path = authorities["phase6_denoising_evidence"]
    peak_path = authorities["phase6_peak_evidence"]
    phase3_dl_path = authorities["phase3_dl_audit"] / "reproducibility_table.csv"
    classical_catalog_path = ROOT / "experiments/phase3/configs/classical_system_catalog_v1.json"
    table1_rows = _derive_table1_rows(phase4_path)
    table_s1_rows = tuple(_csv_rows_from_path(phase4_path / "figure2_alignment_data.csv"))
    table_s2_rows = tuple(_csv_rows_from_path(phase4_path / "figure2_protocol_interaction_data.csv"))
    table2_rows = tuple(
        _csv_rows_from_path(baseline_path / "method_evidence_rows.csv")
        + _csv_rows_from_path(denoising_path / "method_evidence_rows.csv")
        + _csv_rows_from_path(peak_path / "method_evidence_rows.csv")
    )
    table_s3_rows = _derive_table_s3_rows(
        table2_rows=table2_rows,
        baseline_status_path=baseline_path / "system_status.jsonl",
        denoising_status_path=denoising_path / "system_status.jsonl",
        peak_status_path=peak_path / "system_status.jsonl",
        dl_table_path=phase3_dl_path,
        classical_catalog_path=classical_catalog_path,
    )
    figure3_rows = _derive_figure3_model_adequacy_rows(authorities["phase2_background_fit"])
    figure4_node_rows, figure4_edge_rows = _derive_figure4_rows(
        step7_dependency=step7_dependency,
        authorities=authorities,
    )
    authority_bridge = {
        "config_sha256": config.sha256,
        "fixed_authorities": {
            key: file_identity(path)
            for key, path in authorities.items()
        },
        "report_authorities": {
            key: file_identity(ROOT / relative_path)
            for key, relative_path in REPORT_AUTHORITY_PATHS.items()
        },
        "code_receipts": {
            key: file_identity(ROOT / relative_path)
            for key, relative_path in CODE_RECEIPT_PATHS.items()
        },
        "step7_dependency": step7_dependency,
    }
    return _VerifierInputs(
        table1_rows=table1_rows,
        table_s1_rows=table_s1_rows,
        table_s2_rows=table_s2_rows,
        table2_rows=table2_rows,
        table_s3_rows=table_s3_rows,
        figure3_rows=figure3_rows,
        figure4_node_rows=figure4_node_rows,
        figure4_edge_rows=figure4_edge_rows,
        captions=_CAPTIONS,
        authority_bridge=authority_bridge,
    )


def _load_phase6_publication_core_config(path: Path | str = DEFAULT_CONFIG) -> _VerifierConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    document = json.loads(raw)
    if raw != canonical_json_bytes(document):
        raise PublicationCoreVerificationError("config: must be canonical JSON")
    doc = _object("config", document)
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise PublicationCoreVerificationError("config.schema_version: unexpected value")
    if doc.get("experiment_id") != EXPERIMENT_ID:
        raise PublicationCoreVerificationError("config.experiment_id: unexpected value")
    if doc.get("claim_boundary") != CLAIM_BOUNDARY:
        raise PublicationCoreVerificationError("config.claim_boundary: unexpected value")
    payloads = tuple(str(item) for item in doc.get("artifact_payload_files", ()))
    if payloads != PAYLOAD_FILES:
        raise PublicationCoreVerificationError("config.artifact_payload_files: payload inventory mismatch")
    expected = _object("config.expected", doc.get("expected"))
    figure3 = _object("config.figure3", doc.get("figure3"))
    figure4 = _object("config.figure4", doc.get("figure4"))
    report_authorities_raw = doc.get("report_authorities", {})
    code_receipts_raw = doc.get("code_receipts", {})
    report_authorities = _object("config.report_authorities", report_authorities_raw)
    code_receipts = _object("config.code_receipts", code_receipts_raw)
    step7_dependency = _object("config.step7_dependency", doc.get("step7_dependency"))
    if step7_dependency.get("mode") != "deferred_report":
        raise PublicationCoreVerificationError("config.step7_dependency.mode: unexpected value")
    if step7_dependency.get("state") != STEP7_DEFERRED_REPORT_STATE:
        raise PublicationCoreVerificationError("config.step7_dependency.state: unexpected value")
    deferred_report_receipt = _object(
        "config.step7_dependency.deferred_report",
        step7_dependency.get("deferred_report"),
    )
    return _VerifierConfig(
        path=config_path,
        raw_bytes=raw,
        sha256=sha256_hex(raw),
        document=doc,
        synthetic_fixture=bool(doc.get("synthetic_fixture", False)),
        artifact_payload_files=payloads,
        expected={
            "table1": _count_key("config.expected.table1_row_count", expected.get("table1_row_count")),
            "table_s1": _count_key("config.expected.table_s1_row_count", expected.get("table_s1_row_count")),
            "table_s2": _count_key("config.expected.table_s2_row_count", expected.get("table_s2_row_count")),
            "table2": _count_key("config.expected.table2_row_count", expected.get("table2_row_count")),
            "table_s3": _count_key("config.expected.table_s3_row_count", expected.get("table_s3_row_count")),
            "figure3": _count_key("config.expected.figure3_row_count", expected.get("figure3_row_count")),
        },
        figure3=figure3,
        figure4=figure4,
        report_authorities={str(key): _object(f"config.report_authorities.{key}", value) for key, value in report_authorities.items()},
        code_receipts={str(key): _object(f"config.code_receipts.{key}", value) for key, value in code_receipts.items()},
        step7_dependency={
            "mode": "deferred_report",
            "state": STEP7_DEFERRED_REPORT_STATE,
            "deferred_report": deferred_report_receipt,
        },
    )


def _table_s1_fields(rows: tuple[Mapping[str, object], ...]) -> tuple[str, ...]:
    if not rows:
        raise PublicationCoreVerificationError("table_s1_rows: must not be empty")
    return tuple(str(field) for field in rows[0].keys())


def _table_s2_fields(rows: tuple[Mapping[str, object], ...]) -> tuple[str, ...]:
    if not rows:
        raise PublicationCoreVerificationError("table_s2_rows: must not be empty")
    return tuple(str(field) for field in rows[0].keys())


def _inputs_from_object(value: object) -> _VerifierInputs:
    table1_rows = tuple(_object("table1_rows[]", row) for row in value.table1_rows)
    table_s1_rows = tuple(_object("table_s1_rows[]", row) for row in value.table_s1_rows)
    table_s2_rows = tuple(_object("table_s2_rows[]", row) for row in value.table_s2_rows)
    table2_rows = tuple(_object("table2_rows[]", row) for row in value.table2_rows)
    table_s3_rows = tuple(_object("table_s3_rows[]", row) for row in value.table_s3_rows)
    figure3_rows = tuple(_object("figure3_rows[]", row) for row in value.figure3_rows)
    node_rows = tuple(_object("figure4_node_rows[]", row) for row in value.figure4_node_rows)
    edge_rows = tuple(_object("figure4_edge_rows[]", row) for row in value.figure4_edge_rows)
    captions = {str(key): str(item) for key, item in dict(value.captions).items()}
    authority_bridge = {str(key): item for key, item in dict(value.authority_bridge).items()}
    return _VerifierInputs(
        table1_rows=table1_rows,
        table_s1_rows=table_s1_rows,
        table_s2_rows=table_s2_rows,
        table2_rows=table2_rows,
        table_s3_rows=table_s3_rows,
        figure3_rows=figure3_rows,
        figure4_node_rows=node_rows,
        figure4_edge_rows=edge_rows,
        captions=captions,
        authority_bridge=authority_bridge,
    )


def _payload_projection(inputs: _VerifierInputs) -> Mapping[str, object]:
    return {
        "table1_sha256": sha256_hex(csv_bytes(inputs.table1_rows, TABLE1_FIELDS)),
        "table_s1_sha256": sha256_hex(csv_bytes(inputs.table_s1_rows, _table_s1_fields(inputs.table_s1_rows))),
        "table_s2_sha256": sha256_hex(csv_bytes(inputs.table_s2_rows, _table_s2_fields(inputs.table_s2_rows))),
        "table2_sha256": sha256_hex(csv_bytes(inputs.table2_rows, TABLE2_FIELDS)),
        "table_s3_sha256": sha256_hex(csv_bytes(inputs.table_s3_rows, TABLE_S3_FIELDS)),
        "figure3_sha256": sha256_hex(csv_bytes(inputs.figure3_rows, FIGURE3_FIELDS)),
        "figure4_nodes_sha256": sha256_hex(csv_bytes(inputs.figure4_node_rows, FIGURE4_NODE_FIELDS)),
        "figure4_edges_sha256": sha256_hex(csv_bytes(inputs.figure4_edge_rows, FIGURE4_EDGE_FIELDS)),
        "captions_sha256": sha256_hex(canonical_json_bytes(dict(inputs.captions))),
    }


def _validate_inputs(config: _VerifierConfig, inputs: _VerifierInputs) -> None:
    table1_rows = _rows("table1_rows", inputs.table1_rows)
    table_s1_rows = _rows("table_s1_rows", inputs.table_s1_rows)
    table_s2_rows = _rows("table_s2_rows", inputs.table_s2_rows)
    table2_rows = _rows("table2_rows", inputs.table2_rows)
    table_s3_rows = _rows("table_s3_rows", inputs.table_s3_rows)
    figure3_rows = _rows("figure3_rows", inputs.figure3_rows)
    if len(table1_rows) != config.expected["table1"]:
        raise PublicationCoreVerificationError("table1_rows: row-count mismatch")
    if len(table_s1_rows) != config.expected["table_s1"]:
        raise PublicationCoreVerificationError("table_s1_rows: row-count mismatch")
    if len(table_s2_rows) != config.expected["table_s2"]:
        raise PublicationCoreVerificationError("table_s2_rows: row-count mismatch")
    if len(table2_rows) != config.expected["table2"]:
        raise PublicationCoreVerificationError("table2_rows: row-count mismatch")
    if len(table_s3_rows) != config.expected["table_s3"]:
        raise PublicationCoreVerificationError("table_s3_rows: row-count mismatch")
    if len(figure3_rows) != config.expected["figure3"]:
        raise PublicationCoreVerificationError("figure3_rows: row-count mismatch")
    if sum(str(row.get("task_line")) == "deep_learning" for row in table_s3_rows) != 10:
        raise PublicationCoreVerificationError("table_s3_rows: expected 10 deep-learning rows")
    if sum(str(row.get("task_line")) != "deep_learning" for row in table_s3_rows) != 306:
        raise PublicationCoreVerificationError("table_s3_rows: expected 306 classical rows")


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _mix_color(
    low: tuple[int, int, int],
    high: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, float(ratio)))
    return tuple(
        _clamp_channel(low[index] + (high[index] - low[index]) * clamped)
        for index in range(3)
    )


def _escaped(value: object) -> str:
    return html.escape(str(value), quote=True)


def _figure3_orders(
    rows: Sequence[Mapping[str, object]],
    extractor_order: Sequence[str],
    excitation_order: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    row_extractors = {str(row["extractor_id"]) for row in rows}
    row_excitations = {str(row["excitation_stratum"]) for row in rows}
    ordered_extractors = tuple(item for item in extractor_order if item in row_extractors)
    ordered_excitations = tuple(item for item in excitation_order if item in row_excitations)
    return ordered_extractors, ordered_excitations


def _render_figure3_png(
    width: int,
    height: int,
    rows: Sequence[Mapping[str, object]],
    figure3_config: Mapping[str, object],
) -> bytes:
    canvas = RasterCanvas(width, height, (247, 247, 244))
    canvas.fill_rect(0, 0, width, 180, (228, 234, 240))
    canvas.fill_rect(0, height - 180, width, height, (244, 246, 248))
    ordered_extractors, ordered_excitations = _figure3_orders(
        rows,
        tuple(str(item) for item in figure3_config["extractor_order"]),
        tuple(str(item) for item in figure3_config["excitation_order"]),
    )
    by_key = {
        (str(row["extractor_id"]), str(row["excitation_stratum"])): row
        for row in rows
    }
    left = 340
    top = 320
    cell_width = max(220, (width - 520) // max(1, len(ordered_excitations)))
    cell_height = max(220, (height - 720) // max(1, len(ordered_extractors)))
    p95_threshold = float(figure3_config["p95_threshold"])
    for row_index, extractor in enumerate(ordered_extractors):
        for col_index, excitation in enumerate(ordered_excitations):
            row = by_key[(extractor, excitation)]
            median_ratio = min(1.0, float(row["median_reconstruction_nrmse"]) / 0.10)
            p95_ratio = min(1.0, float(row["p95_reconstruction_nrmse"]) / max(p95_threshold, 1e-9))
            median_color = _mix_color((220, 242, 228), (241, 196, 123), median_ratio)
            p95_color = _mix_color((210, 236, 222), (211, 90, 64), p95_ratio)
            status_color = (61, 139, 74) if str(row["gate_state"]) == "pass" else (181, 61, 41)
            x0 = left + col_index * cell_width
            y0 = top + row_index * cell_height
            x1 = x0 + cell_width - 28
            y1 = y0 + cell_height - 28
            canvas.fill_rect(x0, y0, x1, y0 + (cell_height // 2) - 14, median_color)
            canvas.fill_rect(x0, y0 + (cell_height // 2) - 14, x1, y1 - 34, p95_color)
            canvas.fill_rect(x0, y1 - 34, x1, y1, status_color)
            canvas.stroke_rect(x0, y0, x1, y1, (68, 73, 80), thickness=3)
            canvas.fill_rect(x0 + 18, y0 + 18, x0 + 64, y0 + 64, _mix_color((255, 255, 255), median_color, 0.55))
            canvas.fill_rect(x0 + 80, y0 + 18, x0 + 126, y0 + 64, _mix_color((255, 255, 255), p95_color, 0.55))
    grid_left = left - 24
    grid_top = top - 24
    grid_right = left + len(ordered_excitations) * cell_width - 28
    grid_bottom = top + len(ordered_extractors) * cell_height - 28
    canvas.stroke_rect(grid_left, grid_top, grid_right, grid_bottom, (48, 56, 68), thickness=4)
    for index in range(len(ordered_excitations) + 1):
        x = left + index * cell_width - 14
        canvas.draw_line(x, top - 24, x, grid_bottom, (126, 134, 146), thickness=2)
    for index in range(len(ordered_extractors) + 1):
        y = top + index * cell_height - 14
        canvas.draw_line(left - 24, y, grid_right, y, (126, 134, 146), thickness=2)
    legend_x = width - 760
    legend_y = 120
    legend_colors = (
        _mix_color((220, 242, 228), (241, 196, 123), 0.15),
        _mix_color((220, 242, 228), (241, 196, 123), 0.55),
        _mix_color((210, 236, 222), (211, 90, 64), 0.35),
        _mix_color((210, 236, 222), (211, 90, 64), 0.85),
        (61, 139, 74),
        (181, 61, 41),
    )
    for index, color in enumerate(legend_colors):
        x0 = legend_x + index * 62
        canvas.fill_rect(x0, legend_y, x0 + 42, legend_y + 42, color)
        canvas.stroke_rect(x0, legend_y, x0 + 42, legend_y + 42, (66, 70, 76), thickness=2)
    return canvas.to_png_bytes(label="figure3_model_adequacy")


def _render_figure3_svg(
    width: int,
    height: int,
    rows: Sequence[Mapping[str, object]],
    figure3_config: Mapping[str, object],
) -> bytes:
    ordered_extractors, ordered_excitations = _figure3_orders(
        rows,
        tuple(str(item) for item in figure3_config["extractor_order"]),
        tuple(str(item) for item in figure3_config["excitation_order"]),
    )
    by_key = {
        (str(row["extractor_id"]), str(row["excitation_stratum"])): row
        for row in rows
    }
    left = 340
    top = 320
    cell_width = max(220, (width - 520) // max(1, len(ordered_excitations)))
    cell_height = max(220, (height - 720) // max(1, len(ordered_extractors)))
    p95_threshold = float(figure3_config["p95_threshold"])
    lines = [
        '<rect x="0" y="0" width="100%" height="100%" fill="#f7f7f4"/>',
        '<text x="80" y="96" font-size="40" font-weight="700">Figure 3 Model Adequacy Matrix</text>',
        '<text x="80" y="146" font-size="22">3x4 matrix with Median, P95, and PASS or FAIL gate state derived from authoritative model receipts.</text>',
    ]
    for col_index, excitation in enumerate(ordered_excitations):
        x = left + col_index * cell_width + (cell_width // 2) - 14
        lines.append(f'<text x="{x}" y="260" font-size="24" text-anchor="middle">{_escaped(excitation)}</text>')
    for row_index, extractor in enumerate(ordered_extractors):
        y = top + row_index * cell_height + (cell_height // 2)
        lines.append(f'<text x="150" y="{y}" font-size="26" text-anchor="middle">{_escaped(extractor)}</text>')
    for row_index, extractor in enumerate(ordered_extractors):
        for col_index, excitation in enumerate(ordered_excitations):
            row = by_key[(extractor, excitation)]
            median_ratio = min(1.0, float(row["median_reconstruction_nrmse"]) / 0.10)
            p95_ratio = min(1.0, float(row["p95_reconstruction_nrmse"]) / max(p95_threshold, 1e-9))
            median_color = "#{:02x}{:02x}{:02x}".format(*_mix_color((220, 242, 228), (241, 196, 123), median_ratio))
            p95_color = "#{:02x}{:02x}{:02x}".format(*_mix_color((210, 236, 222), (211, 90, 64), p95_ratio))
            status_fill = "#3d8b4a" if str(row["gate_state"]) == "pass" else "#b53d29"
            status_text = "PASS" if str(row["gate_state"]) == "pass" else "FAIL"
            x0 = left + col_index * cell_width
            y0 = top + row_index * cell_height
            x1 = x0 + cell_width - 28
            y1 = y0 + cell_height - 28
            lines.append(f'<g class="matrix-cell" data-extractor="{_escaped(extractor)}" data-excitation="{_escaped(excitation)}">')
            lines.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{(cell_height // 2) - 14}" fill="{median_color}" stroke="#4b5563" stroke-width="2"/>')
            lines.append(f'<rect x="{x0}" y="{y0 + (cell_height // 2) - 14}" width="{x1 - x0}" height="{(cell_height // 2) - 48}" fill="{p95_color}" stroke="#4b5563" stroke-width="2"/>')
            lines.append(f'<rect x="{x0}" y="{y1 - 34}" width="{x1 - x0}" height="34" fill="{status_fill}"/>')
            lines.append(f'<text x="{x0 + 24}" y="{y0 + 42}" font-size="20" font-weight="600">Median {float(row["median_reconstruction_nrmse"]):.4f}</text>')
            lines.append(f'<text x="{x0 + 24}" y="{y0 + 84}" font-size="20" font-weight="600">P95 {float(row["p95_reconstruction_nrmse"]):.4f}</text>')
            lines.append(f'<text x="{x0 + 24}" y="{y1 - 10}" font-size="18" fill="#ffffff" font-weight="700">{status_text}</text>')
            lines.append("</g>")
    grid_left = left - 24
    grid_top = top - 24
    grid_right = left + len(ordered_excitations) * cell_width - 28
    grid_bottom = top + len(ordered_extractors) * cell_height - 28
    for index in range(len(ordered_excitations) + 1):
        x = left + index * cell_width - 14
        lines.append(f'<line x1="{x}" y1="{grid_top}" x2="{x}" y2="{grid_bottom}" stroke="#9ca3af" stroke-width="2"/>')
    for index in range(len(ordered_extractors) + 1):
        y = top + index * cell_height - 14
        lines.append(f'<line x1="{grid_left}" y1="{y}" x2="{grid_right}" y2="{y}" stroke="#9ca3af" stroke-width="2"/>')
    lines.append(f'<line x1="{grid_left}" y1="{grid_top}" x2="{grid_right}" y2="{grid_top}" stroke="#374151" stroke-width="4"/>')
    lines.append(f'<line x1="{grid_left}" y1="{grid_bottom}" x2="{grid_right}" y2="{grid_bottom}" stroke="#374151" stroke-width="4"/>')
    lines.append(f'<line x1="{grid_left}" y1="{grid_top}" x2="{grid_left}" y2="{grid_bottom}" stroke="#374151" stroke-width="4"/>')
    lines.append(f'<line x1="{grid_right}" y1="{grid_top}" x2="{grid_right}" y2="{grid_bottom}" stroke="#374151" stroke-width="4"/>')
    legend_y = height - 170
    legend_items = (
        ("Median low", "#d4ecdf"),
        ("Median high", "#f1c47b"),
        ("P95 low", "#caecd7"),
        ("P95 high", "#d35a40"),
        ("PASS", "#3d8b4a"),
        ("FAIL", "#b53d29"),
    )
    for index, (label, color) in enumerate(legend_items):
        x0 = 160 + index * 260
        lines.append(f'<rect x="{x0}" y="{legend_y}" width="40" height="40" fill="{color}" stroke="#4b5563" stroke-width="2"/>')
        lines.append(f'<text x="{x0 + 56}" y="{legend_y + 27}" font-size="20">{_escaped(label)}</text>')
    return svg_bytes(width, height, title="Figure 3", body_lines=lines)


def _render_figure4_png(
    width: int,
    height: int,
    node_rows: Sequence[Mapping[str, object]],
    edge_rows: Sequence[Mapping[str, object]],
) -> bytes:
    canvas = RasterCanvas(width, height, (249, 248, 244))
    band_colors = ((242, 238, 232), (234, 240, 246), (241, 244, 238))
    band_width = width // 3
    for index, color in enumerate(band_colors):
        canvas.fill_rect(index * band_width, 0, min(width, (index + 1) * band_width), height, color)
    positions: dict[str, tuple[int, int]] = {}
    columns = 3
    rows = 3
    x_step = width // (columns + 1)
    y_step = height // (rows + 1)
    for index, row in enumerate(node_rows):
        col = index % columns
        row_index = index // columns
        positions[str(row["node_id"])] = ((col + 1) * x_step, (row_index + 1) * y_step)
    edge_palette = (
        (54, 101, 153),
        (176, 83, 61),
        (102, 141, 60),
        (142, 93, 161),
        (211, 140, 48),
        (62, 126, 122),
        (95, 95, 95),
        (131, 64, 96),
    )
    for index, row in enumerate(edge_rows):
        source = positions[str(row["source_node_id"])]
        target = positions[str(row["target_node_id"])]
        color = edge_palette[index % len(edge_palette)]
        dash = (18, 12) if str(row["edge_style"]) == "dashed" else None
        canvas.draw_line(source[0], source[1], target[0], target[1], color, thickness=8, dash=dash)
        canvas.draw_line(target[0], target[1], target[0] - 24, target[1] - 12, color, thickness=6)
        canvas.draw_line(target[0], target[1], target[0] - 24, target[1] + 12, color, thickness=6)
    node_palette = (
        (59, 102, 153),
        (81, 135, 114),
        (181, 135, 48),
        (174, 88, 66),
        (108, 118, 164),
        (72, 132, 142),
        (136, 112, 74),
        (126, 86, 122),
        (98, 123, 98),
    )
    node_width = 640
    node_height = 220
    for index, row in enumerate(node_rows):
        center_x, center_y = positions[str(row["node_id"])]
        fill = node_palette[index % len(node_palette)]
        x0 = center_x - node_width // 2
        y0 = center_y - node_height // 2
        x1 = center_x + node_width // 2
        y1 = center_y + node_height // 2
        canvas.fill_rect(x0, y0, x1, y1, fill)
        canvas.fill_rect(x0, y0, x1, y0 + 38, _mix_color(fill, (255, 255, 255), 0.35))
        canvas.stroke_rect(x0, y0, x1, y1, (61, 61, 61), thickness=4)
    legend_x = width - 760
    legend_y = 120
    canvas.draw_line(legend_x, legend_y, legend_x + 170, legend_y, (54, 101, 153), thickness=8)
    canvas.draw_line(legend_x, legend_y + 54, legend_x + 170, legend_y + 54, (176, 83, 61), thickness=8, dash=(18, 12))
    for index, color in enumerate(((54, 101, 153), (176, 83, 61), (81, 135, 114), (181, 135, 48))):
        x0 = legend_x + index * 56
        canvas.fill_rect(x0, legend_y + 120, x0 + 40, legend_y + 160, color)
        canvas.stroke_rect(x0, legend_y + 120, x0 + 40, legend_y + 160, (61, 61, 61), thickness=2)
    return canvas.to_png_bytes(label="figure4_gate_map")


def _render_figure4_svg(
    width: int,
    height: int,
    node_rows: Sequence[Mapping[str, object]],
    edge_rows: Sequence[Mapping[str, object]],
) -> bytes:
    lines = [
        '<defs><marker id="arrowhead" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto"><path d="M0,0 L12,6 L0,12 z" fill="#475569"/></marker></defs>',
        '<rect x="0" y="0" width="100%" height="100%" fill="#f9f8f4"/>',
        '<rect x="0" y="0" width="1200" height="100%" fill="#f2eee8"/>',
        '<rect x="1200" y="0" width="1200" height="100%" fill="#eaf0f6"/>',
        '<rect x="2400" y="0" width="1200" height="100%" fill="#f1f4ee"/>',
        '<text x="80" y="96" font-size="40" font-weight="700">Figure 4 Evidence Gate Map</text>',
        '<text x="80" y="146" font-size="22">Solid edges are required dependencies; dashed edges represent fallback or status-aware transitions.</text>',
    ]
    positions: dict[str, tuple[int, int]] = {}
    columns = 3
    x_step = width // (columns + 1)
    y_step = height // 4
    for index, row in enumerate(node_rows):
        col = index % columns
        row_index = index // columns
        positions[str(row["node_id"])] = ((col + 1) * x_step, (row_index + 1) * y_step)
    edge_palette = ("#366599", "#b0533d", "#668d3c", "#8e5da1", "#d38c30", "#3e7e7a", "#6b7280", "#834060")
    for index, row in enumerate(edge_rows):
        source = positions[str(row["source_node_id"])]
        target = positions[str(row["target_node_id"])]
        dash = ' stroke-dasharray="18 12"' if str(row["edge_style"]) == "dashed" else ""
        color = edge_palette[index % len(edge_palette)]
        lines.append(
            f'<line x1="{source[0]}" y1="{source[1]}" x2="{target[0]}" y2="{target[1]}" '
            f'stroke="{color}" stroke-width="8"{dash} marker-end="url(#arrowhead)"/>'
        )
        mid_x = (source[0] + target[0]) // 2
        mid_y = (source[1] + target[1]) // 2 - 14
        lines.append(f'<text x="{mid_x}" y="{mid_y}" font-size="18" text-anchor="middle">{_escaped(row["label"])}</text>')
    node_palette = ("#3b6699", "#518772", "#b58730", "#ae5842", "#6c76a4", "#48848e", "#88704a", "#7e567a", "#627b62")
    node_width = 640
    node_height = 220
    for index, row in enumerate(node_rows):
        center_x, center_y = positions[str(row["node_id"])]
        fill = node_palette[index % len(node_palette)]
        x0 = center_x - node_width // 2
        y0 = center_y - node_height // 2
        lines.append(f'<rect x="{x0}" y="{y0}" width="{node_width}" height="{node_height}" rx="20" ry="20" fill="{fill}" stroke="#374151" stroke-width="4"/>')
        lines.append(f'<rect x="{x0}" y="{y0}" width="{node_width}" height="42" rx="20" ry="20" fill="rgba(255,255,255,0.18)"/>')
        lines.append(f'<text x="{center_x}" y="{center_y - 10}" font-size="24" fill="#ffffff" text-anchor="middle" font-weight="700">{_escaped(row["label"])}</text>')
        lines.append(f'<text x="{center_x}" y="{center_y + 28}" font-size="18" fill="#ffffff" text-anchor="middle">{_escaped(row["state"])}</text>')
    legend_y = height - 170
    lines.append('<line x1="2600" y1="{0}" x2="2780" y2="{0}" stroke="#366599" stroke-width="8" marker-end="url(#arrowhead)"/>'.format(legend_y))
    lines.append('<text x="2810" y="{0}" font-size="18">required dependency</text>'.format(legend_y + 6))
    lines.append('<line x1="2600" y1="{0}" x2="2780" y2="{0}" stroke="#b0533d" stroke-width="8" stroke-dasharray="18 12" marker-end="url(#arrowhead)"/>'.format(legend_y + 54))
    lines.append('<text x="2810" y="{0}" font-size="18">authorized fallback or status-aware step</text>'.format(legend_y + 60))
    return svg_bytes(width, height, title="Figure 4", body_lines=lines)


def _rebuild_payloads(config_raw: bytes, inputs: _VerifierInputs) -> tuple[str, Mapping[str, bytes], bytes, bytes]:
    config = _object("config", json.loads(config_raw))
    figure3 = _object("config.figure3", config.get("figure3"))
    figure4 = _object("config.figure4", config.get("figure4"))
    run_id = stable_run_id(
        config_sha256=sha256_hex(config_raw),
        payload_projection=_payload_projection(inputs),
    )
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "payload_files": list(PAYLOAD_FILES),
        "step7_dependency": {
            "mode": str(config["step7_dependency"]["mode"]),
            "state": str(config["step7_dependency"]["state"]),
            "authority_path": str(config["step7_dependency"]["deferred_report"]["path"]),
            "authority_sha256": str(config["step7_dependency"]["deferred_report"]["sha256"]),
        },
        "counts": {
            "table1": len(inputs.table1_rows),
            "table_s1": len(inputs.table_s1_rows),
            "table_s2": len(inputs.table_s2_rows),
            "table2": len(inputs.table2_rows),
            "table_s3": len(inputs.table_s3_rows),
            "figure3": len(inputs.figure3_rows),
        },
    }
    preflight = {
        "status": "complete",
        "synthetic_fixture": bool(config.get("synthetic_fixture", False)),
        "environment": environment_receipt(),
    }
    payloads = {
        "config.json": config_raw,
        "authority_bridge.json": canonical_json_bytes(dict(inputs.authority_bridge)),
        "preflight.json": canonical_json_bytes(preflight),
        "table1_metric_validity.csv": csv_bytes(inputs.table1_rows, TABLE1_FIELDS),
        "table1_metric_validity.md": markdown_table(inputs.table1_rows, ("endpoint_id", "state", "jointly_favorable_metric_ids")),
        "table_s1_full_alignment.csv": csv_bytes(inputs.table_s1_rows, _table_s1_fields(inputs.table_s1_rows)),
        "table_s2_protocol_interactions.csv": csv_bytes(inputs.table_s2_rows, _table_s2_fields(inputs.table_s2_rows)),
        "table2_method_evidence.csv": csv_bytes(inputs.table2_rows, TABLE2_FIELDS),
        "table2_method_evidence.md": markdown_table(inputs.table2_rows[:16], ("evidence_id", "task_line", "state", "reason_code")),
        "table_s3_system_status.csv": csv_bytes(inputs.table_s3_rows, TABLE_S3_FIELDS),
        "figure3_model_adequacy_data.csv": csv_bytes(inputs.figure3_rows, FIGURE3_FIELDS),
        "figure3_model_adequacy.png": _render_figure3_png(int(figure3["width_px"]), int(figure3["height_px"]), inputs.figure3_rows, figure3),
        "figure3_model_adequacy.svg": _render_figure3_svg(int(figure3["width_px"]), int(figure3["height_px"]), inputs.figure3_rows, figure3),
        "figure4_gate_nodes.csv": csv_bytes(inputs.figure4_node_rows, FIGURE4_NODE_FIELDS),
        "figure4_gate_edges.csv": csv_bytes(inputs.figure4_edge_rows, FIGURE4_EDGE_FIELDS),
        "figure4_gate_map.png": _render_figure4_png(int(figure4["width_px"]), int(figure4["height_px"]), inputs.figure4_node_rows, inputs.figure4_edge_rows),
        "figure4_gate_map.svg": _render_figure4_svg(int(figure4["width_px"]), int(figure4["height_px"]), inputs.figure4_node_rows, inputs.figure4_edge_rows),
        "captions.json": canonical_json_bytes(dict(inputs.captions)),
        "manifest.json": canonical_json_bytes(manifest),
    }
    complete = canonical_json_bytes({"run_id": run_id, "status": "complete"})
    sha256sums = write_sha256sums(payloads, "complete.json", complete)
    return run_id, payloads, complete, sha256sums


def verify_phase6_publication_core_from_inputs(
    run_path: Path | str,
    *,
    inputs: object,
    config_path: Path | str,
    worker_count: int,
) -> PublicationCoreVerificationSummary:
    if worker_count < 1:
        raise PublicationCoreVerificationError("worker_count: must be at least 1")
    path = Path(run_path)
    if not path.is_dir():
        raise PublicationCoreVerificationError("run_path: not a directory")
    actual = {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}
    if set(actual) != set(PAYLOAD_FILES) | {"complete.json", "SHA256SUMS"}:
        raise PublicationCoreVerificationError("payload inventory mismatch")
    config = _load_phase6_publication_core_config(config_path)
    if actual["config.json"] != config.raw_bytes:
        raise PublicationCoreVerificationError("config.json mismatch")
    verifier_inputs = _inputs_from_object(inputs)
    _validate_inputs(config, verifier_inputs)
    run_id, payloads, complete, sha256sums = _rebuild_payloads(config.raw_bytes, verifier_inputs)
    rebuilt = dict(payloads)
    rebuilt["complete.json"] = complete
    rebuilt["SHA256SUMS"] = sha256sums
    for name in (*PAYLOAD_FILES, "complete.json", "SHA256SUMS"):
        if actual[name] != rebuilt[name]:
            raise PublicationCoreVerificationError(f"payload rebuild semantic mismatch: {name}")
    return PublicationCoreVerificationSummary(path=path, run_id=run_id, status="complete", verified_file_count=21)


def verify_phase6_publication_core(
    run_path: Path | str,
    *,
    worker_count: int,
) -> PublicationCoreVerificationSummary:
    config = _load_phase6_publication_core_config(Path(run_path) / "config.json")
    if config.synthetic_fixture:
        raise PublicationCoreVerificationError("verify_phase6_publication_core: frozen real config required")
    inputs = _derive_real_inputs(config)
    return verify_phase6_publication_core_from_inputs(
        run_path,
        inputs=inputs,
        config_path=config.path,
        worker_count=worker_count,
    )


__all__ = [
    "PublicationCoreVerificationError",
    "PublicationCoreVerificationSummary",
    "verify_phase6_publication_core",
    "verify_phase6_publication_core_from_inputs",
]
