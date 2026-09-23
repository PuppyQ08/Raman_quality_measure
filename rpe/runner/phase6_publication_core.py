from __future__ import annotations

import csv
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
    render_figure3_png,
    render_figure3_svg,
    render_figure4_png,
    render_figure4_svg,
    sha256_file,
    sha256_hex,
    stable_run_id,
    validate_step7_deferred_report,
    write_sha256sums,
)


class PublicationCoreError(ValueError):
    pass


@dataclass(frozen=True)
class PublicationCoreConfig:
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
class PublicationCoreInputs:
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


@dataclass(frozen=True)
class PublicationCoreSummary:
    path: Path
    run_id: str
    status: str
    verified_row_counts: Mapping[str, int]


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise PublicationCoreError(f"{path}: must be an object")
    return value


def _rows(name: str, value: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    rows = tuple(value)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PublicationCoreError(f"{name}[{index}]: must be an object")
    return rows


def _count_key(path: str, value: object) -> int:
    if not isinstance(value, int):
        raise PublicationCoreError(f"{path}: must be an integer")
    return value


def load_phase6_publication_core_config(path: Path | str = DEFAULT_CONFIG) -> PublicationCoreConfig:
    config_path = Path(path)
    raw = config_path.read_bytes()
    document = json.loads(raw)
    if raw != canonical_json_bytes(document):
        raise PublicationCoreError("config: must be canonical JSON")
    doc = _object("config", document)
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise PublicationCoreError("config.schema_version: unexpected value")
    if doc.get("experiment_id") != EXPERIMENT_ID:
        raise PublicationCoreError("config.experiment_id: unexpected value")
    if doc.get("claim_boundary") != CLAIM_BOUNDARY:
        raise PublicationCoreError("config.claim_boundary: unexpected value")
    payloads = tuple(str(item) for item in doc.get("artifact_payload_files", ()))
    if payloads != PAYLOAD_FILES:
        raise PublicationCoreError("config.artifact_payload_files: payload inventory mismatch")
    expected = _object("config.expected", doc.get("expected"))
    figure3 = _object("config.figure3", doc.get("figure3"))
    figure4 = _object("config.figure4", doc.get("figure4"))
    report_authorities_raw = doc.get("report_authorities", {})
    code_receipts_raw = doc.get("code_receipts", {})
    report_authorities = _object("config.report_authorities", report_authorities_raw)
    code_receipts = _object("config.code_receipts", code_receipts_raw)
    step7_dependency = _object("config.step7_dependency", doc.get("step7_dependency"))
    if step7_dependency.get("mode") != "deferred_report":
        raise PublicationCoreError("config.step7_dependency.mode: unexpected value")
    if step7_dependency.get("state") != STEP7_DEFERRED_REPORT_STATE:
        raise PublicationCoreError("config.step7_dependency.state: unexpected value")
    deferred_report_receipt = _object(
        "config.step7_dependency.deferred_report",
        step7_dependency.get("deferred_report"),
    )
    return PublicationCoreConfig(
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


def derive_figure3_model_adequacy_rows(phase2_run_path: Path | str) -> tuple[dict[str, object], ...]:
    run_path = Path(phase2_run_path)
    model_receipts = _object("model_receipts", json.loads((run_path / "model_receipts.json").read_text(encoding="utf-8")))
    gate = _object("gate", json.loads((run_path / "gate.json").read_text(encoding="utf-8")))
    models = model_receipts.get("models")
    model_gates = gate.get("model_gates")
    if not isinstance(models, list) or not isinstance(model_gates, list) or len(models) != 12 or len(model_gates) != 12:
        raise PublicationCoreError("figure3 authorities: expected 12 model rows and 12 gate rows")
    gate_by_key = {
        (str(row["extractor_id"]), str(row["excitation_stratum"])): row
        for row in model_gates
    }
    rows: list[dict[str, object]] = []
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
        raise PublicationCoreError("figure3: valid_count total mismatch")
    if sum(int(row["input_count"]) for row in rows) != 28092:
        raise PublicationCoreError("figure3: input_count total mismatch")
    if sum(row["gate_state"] == "pass" for row in rows) != 5:
        raise PublicationCoreError("figure3: pass-count mismatch")
    if sum(row["gate_state"] != "pass" for row in rows) != 7:
        raise PublicationCoreError("figure3: fail-count mismatch")
    return tuple(sorted(rows, key=lambda item: (str(item["extractor_id"]), str(item["excitation_stratum"]))))


def _table_s1_fields(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    if not rows:
        raise PublicationCoreError("table_s1_rows: must not be empty")
    return tuple(str(field) for field in rows[0].keys())


def _table_s2_fields(rows: Sequence[Mapping[str, object]]) -> tuple[str, ...]:
    if not rows:
        raise PublicationCoreError("table_s2_rows: must not be empty")
    return tuple(str(field) for field in rows[0].keys())


def _payload_projection(
    *,
    table1_rows: Sequence[Mapping[str, object]],
    table_s1_rows: Sequence[Mapping[str, object]],
    table_s2_rows: Sequence[Mapping[str, object]],
    table2_rows: Sequence[Mapping[str, object]],
    table_s3_rows: Sequence[Mapping[str, object]],
    figure3_rows: Sequence[Mapping[str, object]],
    node_rows: Sequence[Mapping[str, object]],
    edge_rows: Sequence[Mapping[str, object]],
    captions: Mapping[str, str],
) -> Mapping[str, object]:
    return {
        "table1_sha256": sha256_hex(csv_bytes(table1_rows, TABLE1_FIELDS)),
        "table_s1_sha256": sha256_hex(csv_bytes(table_s1_rows, _table_s1_fields(table_s1_rows))),
        "table_s2_sha256": sha256_hex(csv_bytes(table_s2_rows, _table_s2_fields(table_s2_rows))),
        "table2_sha256": sha256_hex(csv_bytes(table2_rows, TABLE2_FIELDS)),
        "table_s3_sha256": sha256_hex(csv_bytes(table_s3_rows, TABLE_S3_FIELDS)),
        "figure3_sha256": sha256_hex(csv_bytes(figure3_rows, FIGURE3_FIELDS)),
        "figure4_nodes_sha256": sha256_hex(csv_bytes(node_rows, FIGURE4_NODE_FIELDS)),
        "figure4_edges_sha256": sha256_hex(csv_bytes(edge_rows, FIGURE4_EDGE_FIELDS)),
        "captions_sha256": sha256_hex(canonical_json_bytes(dict(captions))),
    }


def _validate_inputs(config: PublicationCoreConfig, inputs: PublicationCoreInputs) -> None:
    table1_rows = _rows("table1_rows", inputs.table1_rows)
    table_s1_rows = _rows("table_s1_rows", inputs.table_s1_rows)
    table_s2_rows = _rows("table_s2_rows", inputs.table_s2_rows)
    table2_rows = _rows("table2_rows", inputs.table2_rows)
    table_s3_rows = _rows("table_s3_rows", inputs.table_s3_rows)
    figure3_rows = _rows("figure3_rows", inputs.figure3_rows)
    if len(table1_rows) != config.expected["table1"]:
        raise PublicationCoreError("table1_rows: row-count mismatch")
    if len(table_s1_rows) != config.expected["table_s1"]:
        raise PublicationCoreError("table_s1_rows: row-count mismatch")
    if len(table_s2_rows) != config.expected["table_s2"]:
        raise PublicationCoreError("table_s2_rows: row-count mismatch")
    if len(table2_rows) != config.expected["table2"]:
        raise PublicationCoreError("table2_rows: row-count mismatch")
    if len(table_s3_rows) != config.expected["table_s3"]:
        raise PublicationCoreError("table_s3_rows: row-count mismatch")
    if len(figure3_rows) != config.expected["figure3"]:
        raise PublicationCoreError("figure3_rows: row-count mismatch")
    if sum(str(row.get("task_line")) == "deep_learning" for row in table_s3_rows) != 10:
        raise PublicationCoreError("table_s3_rows: expected 10 deep-learning rows")
    if sum(str(row.get("task_line")) != "deep_learning" for row in table_s3_rows) != 306:
        raise PublicationCoreError("table_s3_rows: expected 306 classical rows")


def _render_figure3_png(config: PublicationCoreConfig) -> bytes:
    raise PublicationCoreError("_render_figure3_png requires figure3 rows")


def _render_figure3_png_with_rows(
    config: PublicationCoreConfig,
    rows: Sequence[Mapping[str, object]],
) -> bytes:
    return render_figure3_png(
        width=int(config.figure3["width_px"]),
        height=int(config.figure3["height_px"]),
        rows=rows,
        extractor_order=tuple(str(item) for item in config.figure3["extractor_order"]),
        excitation_order=tuple(str(item) for item in config.figure3["excitation_order"]),
        p95_threshold=float(config.figure3["p95_threshold"]),
    )


def _render_figure3_svg(config: PublicationCoreConfig, rows: Sequence[Mapping[str, object]]) -> bytes:
    return render_figure3_svg(
        width=int(config.figure3["width_px"]),
        height=int(config.figure3["height_px"]),
        rows=rows,
        extractor_order=tuple(str(item) for item in config.figure3["extractor_order"]),
        excitation_order=tuple(str(item) for item in config.figure3["excitation_order"]),
        p95_threshold=float(config.figure3["p95_threshold"]),
    )


def _render_figure4_png(config: PublicationCoreConfig) -> bytes:
    raise PublicationCoreError("_render_figure4_png requires node and edge rows")


def _render_figure4_png_with_rows(
    config: PublicationCoreConfig,
    node_rows: Sequence[Mapping[str, object]],
    edge_rows: Sequence[Mapping[str, object]],
) -> bytes:
    return render_figure4_png(
        width=int(config.figure4["width_px"]),
        height=int(config.figure4["height_px"]),
        node_rows=node_rows,
        edge_rows=edge_rows,
    )


def _render_figure4_svg(
    config: PublicationCoreConfig,
    node_rows: Sequence[Mapping[str, object]],
    edge_rows: Sequence[Mapping[str, object]],
) -> bytes:
    return render_figure4_svg(
        width=int(config.figure4["width_px"]),
        height=int(config.figure4["height_px"]),
        node_rows=node_rows,
        edge_rows=edge_rows,
    )


def build_phase6_publication_core_from_inputs(
    output_path: Path | str,
    *,
    inputs: PublicationCoreInputs,
    config: PublicationCoreConfig,
    worker_count: int,
) -> PublicationCoreSummary:
    if worker_count < 1:
        raise PublicationCoreError("worker_count: must be at least 1")
    _validate_inputs(config, inputs)
    table1_rows = _rows("table1_rows", inputs.table1_rows)
    table_s1_rows = _rows("table_s1_rows", inputs.table_s1_rows)
    table_s2_rows = _rows("table_s2_rows", inputs.table_s2_rows)
    table2_rows = _rows("table2_rows", inputs.table2_rows)
    table_s3_rows = _rows("table_s3_rows", inputs.table_s3_rows)
    figure3_rows = _rows("figure3_rows", inputs.figure3_rows)
    node_rows = _rows("figure4_node_rows", inputs.figure4_node_rows)
    edge_rows = _rows("figure4_edge_rows", inputs.figure4_edge_rows)
    captions = {str(key): str(value) for key, value in dict(inputs.captions).items()}

    run_id = stable_run_id(
        config_sha256=config.sha256,
        payload_projection=_payload_projection(
            table1_rows=table1_rows,
            table_s1_rows=table_s1_rows,
            table_s2_rows=table_s2_rows,
            table2_rows=table2_rows,
            table_s3_rows=table_s3_rows,
            figure3_rows=figure3_rows,
            node_rows=node_rows,
            edge_rows=edge_rows,
            captions=captions,
        ),
    )
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "payload_files": list(PAYLOAD_FILES),
        "step7_dependency": {
            "mode": str(config.step7_dependency["mode"]),
            "state": str(config.step7_dependency["state"]),
            "authority_path": str(_object("config.step7_dependency.deferred_report", config.step7_dependency["deferred_report"])["path"]),
            "authority_sha256": str(_object("config.step7_dependency.deferred_report", config.step7_dependency["deferred_report"])["sha256"]),
        },
        "counts": {
            "table1": len(table1_rows),
            "table_s1": len(table_s1_rows),
            "table_s2": len(table_s2_rows),
            "table2": len(table2_rows),
            "table_s3": len(table_s3_rows),
            "figure3": len(figure3_rows),
        },
    }
    preflight = {
        "status": "complete",
        "synthetic_fixture": config.synthetic_fixture,
        "environment": environment_receipt(),
    }
    payloads = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": canonical_json_bytes(dict(inputs.authority_bridge)),
        "preflight.json": canonical_json_bytes(preflight),
        "table1_metric_validity.csv": csv_bytes(table1_rows, TABLE1_FIELDS),
        "table1_metric_validity.md": markdown_table(table1_rows, ("endpoint_id", "state", "jointly_favorable_metric_ids")),
        "table_s1_full_alignment.csv": csv_bytes(table_s1_rows, _table_s1_fields(table_s1_rows)),
        "table_s2_protocol_interactions.csv": csv_bytes(table_s2_rows, _table_s2_fields(table_s2_rows)),
        "table2_method_evidence.csv": csv_bytes(table2_rows, TABLE2_FIELDS),
        "table2_method_evidence.md": markdown_table(table2_rows[:16], ("evidence_id", "task_line", "state", "reason_code")),
        "table_s3_system_status.csv": csv_bytes(table_s3_rows, TABLE_S3_FIELDS),
        "figure3_model_adequacy_data.csv": csv_bytes(figure3_rows, FIGURE3_FIELDS),
        "figure3_model_adequacy.png": _render_figure3_png_with_rows(config, figure3_rows),
        "figure3_model_adequacy.svg": _render_figure3_svg(config, figure3_rows),
        "figure4_gate_nodes.csv": csv_bytes(node_rows, FIGURE4_NODE_FIELDS),
        "figure4_gate_edges.csv": csv_bytes(edge_rows, FIGURE4_EDGE_FIELDS),
        "figure4_gate_map.png": _render_figure4_png_with_rows(config, node_rows, edge_rows),
        "figure4_gate_map.svg": _render_figure4_svg(config, node_rows, edge_rows),
        "captions.json": canonical_json_bytes(captions),
        "manifest.json": canonical_json_bytes(manifest),
    }
    complete_bytes = canonical_json_bytes({"run_id": run_id, "status": "complete"})
    sha256sums = write_sha256sums(payloads, "complete.json", complete_bytes)
    target = Path(output_path)
    if target.exists():
        raise PublicationCoreError("output_path: must not already exist")
    target.mkdir(parents=True)
    for name in PAYLOAD_FILES:
        target.joinpath(name).write_bytes(payloads[name])
    target.joinpath("complete.json").write_bytes(complete_bytes)
    target.joinpath("SHA256SUMS").write_bytes(sha256sums)
    return PublicationCoreSummary(
        path=target,
        run_id=run_id,
        status="complete",
        verified_row_counts=manifest["counts"],
    )


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


def _csv_rows_from_path(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _jsonl_rows(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(_object(path.name, json.loads(line)))
    return rows


def _receipt_matches(label: str, actual_path: Path, receipt_value: object) -> None:
    receipt = _object(label, receipt_value)
    actual = file_identity(actual_path)
    expected_path = receipt.get("path")
    if expected_path not in (None, actual["path"]):
        raise PublicationCoreError(f"{label}.path: frozen receipt mismatch")
    if receipt.get("byte_count") != actual["byte_count"]:
        raise PublicationCoreError(f"{label}.byte_count: frozen receipt mismatch")
    if receipt.get("sha256") != actual["sha256"]:
        raise PublicationCoreError(f"{label}.sha256: frozen receipt mismatch")


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
            raise PublicationCoreError(f"{label}: missing selected payload {name}")
        actual_sha = sha256_file(file_path)
        if entries.get(name) != actual_sha:
            raise PublicationCoreError(f"{label}: SHA256SUMS mismatch for {name}")
    terminal_path = run_path / terminal_name
    if not terminal_path.is_file():
        raise PublicationCoreError(f"{label}: missing terminal payload {terminal_name}")
    terminal = json.loads((run_path / terminal_name).read_text(encoding="utf-8"))
    if _object(f"{label}.{terminal_name}", terminal).get("status") != ("failed" if terminal_name == "failed.json" else "complete"):
        raise PublicationCoreError(f"{label}: unexpected terminal status")


def _resolve_frozen_path(path_value: object) -> Path:
    path = Path(str(path_value))
    if not path.is_absolute():
        path = ROOT / path
    return path


def _fixed_authority_paths(config: PublicationCoreConfig) -> tuple[dict[str, Path], Mapping[str, object]]:
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
    validate_step7_deferred_report(deferred_report_path, error_type=PublicationCoreError)
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


def _cluster_count(label: str) -> str:
    if label.startswith("D2-5"):
        return "5"
    if label.startswith("D2-10"):
        return "10"
    if label.startswith("D2-20"):
        return "20"
    return ""


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
            raise PublicationCoreError(f"table1: missing parent panel row for {panel_id}")
        if str(parent_row.get("endpoint_id")) != endpoint_key:
            raise PublicationCoreError(f"table1: endpoint mismatch for {panel_id}")
        if str(parent_row.get("protocol_id")) != protocol_id:
            raise PublicationCoreError(f"table1: protocol mismatch for {panel_id}")
        if panel_id == "d1_b_closed":
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
            raise PublicationCoreError(f"table1: expected 13 metric rows for {panel_id}")
        if any(str(row.get("endpoint_id")) != endpoint_key for row in panel_rows):
            raise PublicationCoreError(f"table1: alignment endpoint mismatch for {panel_id}")
        if any(str(row.get("protocol_id")) != protocol_id for row in panel_rows):
            raise PublicationCoreError(f"table1: alignment protocol mismatch for {panel_id}")
        mse_row = next((row for row in panel_rows if str(row["metric_output_id"]) == "mse"), None)
        if mse_row is None:
            raise PublicationCoreError(f"table1: missing mse row for {panel_id}")
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
        raise PublicationCoreError("table_s3: classical catalog must contain 306 systems")
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
        raise PublicationCoreError("table_s3: expected 10 deep-learning audit rows")
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
        raise PublicationCoreError("table_s3: row-count mismatch")
    return tuple(classical_rows)


def _derive_figure4_rows(
    config: PublicationCoreConfig,
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


def _derive_real_inputs(config: PublicationCoreConfig) -> PublicationCoreInputs:
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
    figure3_rows = derive_figure3_model_adequacy_rows(authorities["phase2_background_fit"])
    figure4_node_rows, figure4_edge_rows = _derive_figure4_rows(
        config,
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
    return PublicationCoreInputs(
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


def build_phase6_publication_core(
    output_root: Path | str,
    *,
    worker_count: int,
) -> PublicationCoreSummary:
    config = load_phase6_publication_core_config(DEFAULT_CONFIG)
    if config.synthetic_fixture:
        raise PublicationCoreError("build_phase6_publication_core: frozen real config required")
    inputs = _derive_real_inputs(config)
    run_id = stable_run_id(
        config_sha256=config.sha256,
        payload_projection=_payload_projection(
            table1_rows=inputs.table1_rows,
            table_s1_rows=inputs.table_s1_rows,
            table_s2_rows=inputs.table_s2_rows,
            table2_rows=inputs.table2_rows,
            table_s3_rows=inputs.table_s3_rows,
            figure3_rows=inputs.figure3_rows,
            node_rows=inputs.figure4_node_rows,
            edge_rows=inputs.figure4_edge_rows,
            captions=inputs.captions,
        ),
    )
    return build_phase6_publication_core_from_inputs(
        Path(output_root) / run_id,
        inputs=inputs,
        config=config,
        worker_count=worker_count,
    )


__all__ = [
    "PublicationCoreConfig",
    "PublicationCoreError",
    "PublicationCoreInputs",
    "PublicationCoreSummary",
    "build_phase6_publication_core",
    "build_phase6_publication_core_from_inputs",
    "derive_figure3_model_adequacy_rows",
    "load_phase6_publication_core_config",
]
