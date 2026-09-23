from __future__ import annotations

import csv
import importlib
import importlib.machinery
import importlib.util
import io
import json
import struct
import sys
import tempfile
import types
import unittest
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SUBJECT_MODULE = "rpe.runner.phase6_publication_core"
VERIFIER_MODULE = "rpe.runner.phase6_publication_core_verifier"

EXPECTED_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "table1_metric_validity.csv",
    "table1_metric_validity.md",
    "table_s1_full_alignment.csv",
    "table_s2_protocol_interactions.csv",
    "table2_method_evidence.csv",
    "table2_method_evidence.md",
    "table_s3_system_status.csv",
    "figure3_model_adequacy_data.csv",
    "figure3_model_adequacy.png",
    "figure3_model_adequacy.svg",
    "figure4_gate_nodes.csv",
    "figure4_gate_edges.csv",
    "figure4_gate_map.png",
    "figure4_gate_map.svg",
    "captions.json",
    "manifest.json",
)
EXPECTED_INVENTORY = set(EXPECTED_PAYLOAD_FILES) | {"complete.json", "SHA256SUMS"}
FORMAL_REPORT_PATHS = {
    "phase0_source_audit_report": "reports/phase0/step01_data_source_audit.md",
    "phase05_internal_screening_decision_report": "reports/phase05/phase05_internal_screening_decision.md",
    "phase1_core10k_report": "reports/phase1/step16_core10k_build.md",
    "phase3_status_report": "reports/phase6/step01_publication_package_design.md",
    "phase4_closure_synthesis_report": "reports/phase4/step38_phase4_closure_synthesis.md",
    "phase6_step1_design_report": "reports/phase6/step01_publication_package_design.md",
}
FORMAL_CODE_PATHS = {
    "authority_module": "rpe/runner/phase6_publication_core_authority.py",
    "builder_module": "rpe/runner/phase6_publication_core.py",
    "verifier_module": "rpe/runner/phase6_publication_core_verifier.py",
    "run_cli": "tools/run_phase6_publication_core.py",
    "freeze_tool": "tools/freeze_phase6_publication_core_config.py",
    "focused_test": "tests/test_phase6_publication_core.py",
}


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _rewrite_sha256sums(run_path: Path) -> None:
    payloads = sorted(item.name for item in run_path.iterdir() if item.is_file() and item.name != "SHA256SUMS")
    lines = [
        f"{subject_hash((run_path / name).read_bytes())}  {name}"
        for name in payloads
    ]
    (run_path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def subject_hash(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()


def _path_receipt(path: Path) -> dict[str, object]:
    if path.is_dir():
        ledger = path / "SHA256SUMS"
        return {
            "path": str(path.relative_to(ROOT)),
            "byte_count": ledger.stat().st_size,
            "sha256": subject_hash(ledger.read_bytes()),
        }
    return {
        "path": str(path.relative_to(ROOT)),
        "byte_count": path.stat().st_size,
        "sha256": subject_hash(path.read_bytes()),
    }


def _png_unique_rgb_count(raw: bytes) -> int:
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError("expected PNG signature")
    offset = 8
    width = height = None
    idat_parts: list[bytes] = []
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        chunk_data = raw[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, flt, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if bit_depth != 8 or color_type != 2 or compression != 0 or flt != 0 or interlace != 0:
                raise AssertionError("unsupported PNG encoding for focused test")
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None:
        raise AssertionError("missing PNG IHDR")
    data = zlib.decompress(b"".join(idat_parts))
    stride = width * 3
    expected_length = height * (1 + stride)
    if len(data) != expected_length:
        raise AssertionError("unexpected PNG payload length")
    colors: set[tuple[int, int, int]] = set()
    cursor = 0
    for _ in range(height):
        if data[cursor] != 0:
            raise AssertionError("expected filter type 0 in focused PNG decoder")
        cursor += 1
        row = data[cursor : cursor + stride]
        cursor += stride
        sample_step = max(1, width // 64)
        for pixel in range(0, width, sample_step):
            base = pixel * 3
            colors.add((row[base], row[base + 1], row[base + 2]))
    return len(colors)


def _assert_real_figure3_visuals(test_case: unittest.TestCase, run_path: Path) -> None:
    png_colors = _png_unique_rgb_count((run_path / "figure3_model_adequacy.png").read_bytes())
    test_case.assertGreater(
        png_colors,
        12,
        "Figure 3 PNG should contain a nontrivial matrix palette, not a uniform fill",
    )
    svg = (run_path / "figure3_model_adequacy.svg").read_text(encoding="utf-8")
    for token in ("airpls", "arpls", "mor", "green_514", "green_532", "nir_780", "nir_785", "Median", "P95", "PASS", "FAIL"):
        test_case.assertIn(token, svg)
    test_case.assertGreaterEqual(svg.count("<rect"), 16)
    test_case.assertGreaterEqual(svg.count("<line"), 8)
    test_case.assertIn("matrix-cell", svg)


def _assert_real_figure4_visuals(test_case: unittest.TestCase, run_path: Path) -> None:
    png_colors = _png_unique_rgb_count((run_path / "figure4_gate_map.png").read_bytes())
    test_case.assertGreater(
        png_colors,
        10,
        "Figure 4 PNG should contain nodes, edges, and state colors, not a uniform fill",
    )
    svg = (run_path / "figure4_gate_map.svg").read_text(encoding="utf-8")
    for token in ("phase2_model_adequacy", "phase6_publication", "authorized fallback", "status-aware package"):
        test_case.assertIn(token, svg)
    test_case.assertGreaterEqual(svg.count("<rect"), 9)
    test_case.assertGreaterEqual(svg.count("<line"), 8)
    test_case.assertIn("marker-end", svg)
    test_case.assertIn("stroke-dasharray", svg)


def _make_config_document() -> dict[str, object]:
    synthetic_report = ROOT / "reports/phase6/step07_appendix_audits.md"
    return {
        "schema_version": "phase6-publication-core-config-v1",
        "experiment_id": "phase6-publication-core-v1",
        "claim_boundary": "publication_tables_and_figure3_figure4_only",
        "synthetic_fixture": True,
        "artifact_payload_files": list(EXPECTED_PAYLOAD_FILES),
        "expected": {
            "table1_row_count": 12,
            "table_s1_row_count": 143,
            "table_s2_row_count": 65,
            "table2_row_count": 1656,
            "table_s3_row_count": 316,
            "figure3_row_count": 12,
            "configured_payload_count": len(EXPECTED_PAYLOAD_FILES),
            "artifact_file_count": len(EXPECTED_PAYLOAD_FILES) + 2,
        },
        "figure3": {
            "width_px": 3600,
            "height_px": 2700,
            "dpi": 300,
            "p95_threshold": 0.25,
            "extractor_order": ["airpls", "arpls", "mor"],
            "excitation_order": ["green_514", "green_532", "nir_780", "nir_785"],
        },
        "figure4": {
            "width_px": 3600,
            "height_px": 2400,
            "dpi": 300,
        },
        "authorities": {
            "phase4_final_figures": {
                "run_id": "phase4-final-figures-04ae2a217ad7d5fc1beafd4742dbc5bf3794079aca2dadfc3d4cc9b46411e7c6",
            },
            "phase6_baseline_evidence": {
                "run_id": "phase6-baseline-evidence-41e258ebcbcfe8244783112d29facd222b70e77cc4729a3742b432ea6ae29b4f",
            },
            "phase6_denoising_evidence": {
                "run_id": "phase6-denoising-evidence-991db5224bb01234d881d92cdf8a2483c486135605019b6c528bd982eac85701",
            },
            "phase6_peak_evidence": {
                "run_id": "phase6-peak-evidence-5e8c7074799233232b3af6285b98ee75b11ea117476cf262f712738a901838ad",
            },
            "phase2_background_fit": {
                "run_id": "phase2-background-fit-6f8dc92c8b3edbefe2a422b805214b6687990c3ac13a34fbaed3faaa3fad6cdd",
            },
            "phase3_dl_audit": {
                "run_id": "phase3-dl-audit-b9bcfadabaf230832d129b85b295cf720fd01aaf2a47e786f5edb0e83b23cb3b",
            },
        },
        "step7_dependency": {
            "mode": "deferred_report",
            "state": "deferred_by_owner",
            "deferred_report": {
                "path": str(synthetic_report.relative_to(ROOT)),
                "byte_count": 0,
                "sha256": "0" * 64,
            },
        },
    }


def _make_step7_fixture(step7_path: Path) -> None:
    step7_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        step7_path / "config.json",
        {
            "schema_version": "phase6-appendix-audits-v1",
            "experiment_id": "phase6-appendix-audits-v1",
            "artifact_payload_files": ["audit_status.csv"],
        },
    )
    _write_json(
        step7_path / "manifest.json",
        {
            "status": "complete",
            "payload_files": ["config.json", "manifest.json"],
            "run_id": "phase6-appendix-audits-fixture",
        },
    )
    _write_json(step7_path / "complete.json", {"status": "complete"})
    _rewrite_sha256sums(step7_path)


def _make_step7_deferred_report(report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "---\n"
        "state: deferred_by_owner\n"
        "outcome_execution: not_admitted\n"
        "scientific_claims: not_admitted\n"
        "---\n"
        "\n"
        "# Phase 6 Step 7 Appendix Audits\n"
        "\n"
        "Owner-authorized defer status for Step 7.\n"
        "\n"
        "Earlier computational attempts may exist, but they are not admitted as Step 7 execution.\n"
        "\n"
        "state: deferred_by_owner\n",
        encoding="utf-8",
    )


def _make_formal_config_document(step7_path: Path) -> dict[str, object]:
    authorities = {}
    for key, relative_path in {
        "phase4_final_figures": "results/phase4/final_figures_v1/phase4-final-figures-04ae2a217ad7d5fc1beafd4742dbc5bf3794079aca2dadfc3d4cc9b46411e7c6",
        "phase6_baseline_evidence": "results/phase6/baseline_evidence_v1/phase6-baseline-evidence-41e258ebcbcfe8244783112d29facd222b70e77cc4729a3742b432ea6ae29b4f",
        "phase6_denoising_evidence": "results/phase6/denoising_evidence_v1/phase6-denoising-evidence-991db5224bb01234d881d92cdf8a2483c486135605019b6c528bd982eac85701",
        "phase6_peak_evidence": "results/phase6/peak_evidence_v1/phase6-peak-evidence-5e8c7074799233232b3af6285b98ee75b11ea117476cf262f712738a901838ad",
        "phase2_background_fit": "results/phase2/background_fit/phase2-background-fit-6f8dc92c8b3edbefe2a422b805214b6687990c3ac13a34fbaed3faaa3fad6cdd",
        "phase3_dl_audit": "results/phase3/dl_reproducibility_audit_v1/phase3-dl-audit-b9bcfadabaf230832d129b85b295cf720fd01aaf2a47e786f5edb0e83b23cb3b",
    }.items():
        path = ROOT / relative_path
        authorities[key] = _path_receipt(path)
    report_authorities = {
        key: _path_receipt(ROOT / relative_path)
        for key, relative_path in FORMAL_REPORT_PATHS.items()
    }
    code_receipts = {
        key: _path_receipt(ROOT / relative_path)
        for key, relative_path in FORMAL_CODE_PATHS.items()
    }
    return {
        "schema_version": "phase6-publication-core-config-v1",
        "experiment_id": "phase6-publication-core-v1",
        "claim_boundary": "publication_tables_and_figure3_figure4_only",
        "synthetic_fixture": False,
        "artifact_payload_files": list(EXPECTED_PAYLOAD_FILES),
        "expected": {
            "table1_row_count": 12,
            "table_s1_row_count": 143,
            "table_s2_row_count": 65,
            "table2_row_count": 1656,
            "table_s3_row_count": 316,
            "figure3_row_count": 12,
            "configured_payload_count": len(EXPECTED_PAYLOAD_FILES),
            "artifact_file_count": len(EXPECTED_PAYLOAD_FILES) + 2,
        },
        "figure3": {
            "width_px": 3600,
            "height_px": 2700,
            "dpi": 300,
            "p95_threshold": 0.25,
            "extractor_order": ["airpls", "arpls", "mor"],
            "excitation_order": ["green_514", "green_532", "nir_780", "nir_785"],
        },
        "figure4": {
            "width_px": 3600,
            "height_px": 2400,
            "dpi": 300,
        },
        "authorities": authorities,
        "report_authorities": report_authorities,
        "code_receipts": code_receipts,
        "step7_dependency": {
            "mode": "completed_run",
            "state": "complete",
            "run": _path_receipt(step7_path / "SHA256SUMS"),
            "run_path": str(step7_path.relative_to(ROOT)),
        },
    }


def _make_table1_rows() -> list[dict[str, object]]:
    endpoint_ids = (
        "D5-A",
        "D5-B",
        "D2-5-A",
        "D2-5-B",
        "D2-10-A",
        "D2-10-B",
        "D2-20-A",
        "D2-20-B",
        "D1-A",
        "D1-B-closed",
        "D4-A",
        "D4-B",
    )
    rows: list[dict[str, object]] = []
    for index, endpoint_id in enumerate(endpoint_ids):
        state = "not_evaluable_failed_alpha0_equivalence" if endpoint_id == "D1-B-closed" else "complete"
        rows.append(
            {
                "endpoint_id": endpoint_id,
                "protocol_id": endpoint_id.split("-")[-1].lower().replace("closed", "b"),
                "protocol_label": endpoint_id,
                "tier": "core",
                "perturbation_scope": "P8-P12",
                "cluster_kind": "class" if endpoint_id.startswith("D5") else "well",
                "cluster_count": "" if endpoint_id == "D1-B-closed" else str(index + 1),
                "state": state,
                "mse_ag": "" if endpoint_id == "D1-B-closed" else format(0.01 + index / 1000.0, ".17g"),
                "mse_ag_lower": "" if endpoint_id == "D1-B-closed" else format(0.008 + index / 1000.0, ".17g"),
                "mse_ag_upper": "" if endpoint_id == "D1-B-closed" else format(0.012 + index / 1000.0, ".17g"),
                "mse_acc_cross": "" if endpoint_id == "D1-B-closed" else format(0.55 + index / 100.0, ".17g"),
                "mse_acc_cross_lower": "" if endpoint_id == "D1-B-closed" else format(0.50 + index / 100.0, ".17g"),
                "mse_acc_cross_upper": "" if endpoint_id == "D1-B-closed" else format(0.60 + index / 100.0, ".17g"),
                "jointly_favorable_metric_ids": "mse;rmse" if endpoint_id != "D1-B-closed" else "",
                "ag_only_favorable_metric_ids": "mae" if endpoint_id != "D1-B-closed" else "",
                "acc_only_favorable_metric_ids": "sam" if endpoint_id != "D1-B-closed" else "",
                "holm_family_id": f"{endpoint_id}:within_protocol",
                "holm_family_size": "24",
                "claim_scope": "within_protocol_family_only",
                "source_path": "figure2_alignment_data.csv",
                "source_sha256": f"{index + 1:064x}",
            }
        )
    return rows


def _make_generic_rows(row_count: int, prefix: str, fields: tuple[str, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(row_count):
        row = {}
        for field in fields:
            if field.endswith("_id") or field.endswith("_path") or field.endswith("_sha256") or field in {
                "state",
                "reason_code",
                "publication_disposition",
                "preferred_direction",
                "coverage_state",
                "direct_gt_state",
                "downstream_state",
                "reference_free_state",
                "half_split_state",
                "phase5_power_state",
                "task_line",
                "family_id",
                "system_id",
                "protocol_id",
                "cohort_id",
                "metric_output_id",
                "evidence_component",
                "endpoint_id",
            }:
                row[field] = f"{prefix}_{field}_{index}"
            else:
                row[field] = str(index)
        rows.append(row)
    return rows


def _make_system_status_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(306):
        rows.append(
            {
                "task_line": "baseline_correction" if index < 210 else "denoising" if index < 270 else "peak_detection",
                "system_id": f"classical-{index:03d}",
                "family_id": f"family-{index % 14:02d}",
                "planned_K": "1",
                "runnable_K": "1" if index < 193 else "0",
                "coverage_promoted_K": "1" if index < 193 else "0",
                "coverage_state": "coverage_promoted" if index < 193 else "status_only_nonpromoted",
                "direct_gt_state": "not_evaluable_no_validated_clean_gt",
                "downstream_state": "complete" if index < 150 else "not_evaluable_no_common_consumer",
                "reference_free_state": "complete" if index < 150 else "not_evaluable_no_common_consumer",
                "half_split_state": "complete" if index < 112 else "not_applicable",
                "phase5_eligible_K": "1" if index < 112 else "0",
                "phase5_power_state": "not_powered_for_phase5",
                "publication_disposition": "reported_only" if index >= 193 else "method_status_only",
                "reason_code": "not_powered_for_phase5" if index < 193 else "not_promoted",
                "source_path": "classical_catalog_v1.json",
                "source_sha256": f"{index + 1000:064x}",
            }
        )
    for index in range(10):
        rows.append(
            {
                "task_line": "deep_learning",
                "system_id": f"DL{index + 1:02d}",
                "family_id": f"dl-family-{index:02d}",
                "planned_K": "1",
                "runnable_K": "0",
                "coverage_promoted_K": "0",
                "coverage_state": "reported_only",
                "direct_gt_state": "not_applicable",
                "downstream_state": "not_evaluable_data_missing",
                "reference_free_state": "not_applicable",
                "half_split_state": "not_applicable",
                "phase5_eligible_K": "0",
                "phase5_power_state": "not_powered_for_phase5",
                "publication_disposition": "reported_only",
                "reason_code": "not_reproducible_under_frozen_audit",
                "source_path": "reproducibility_table.csv",
                "source_sha256": f"{index + 2000:064x}",
            }
        )
    return rows


def _make_figure4_nodes() -> list[dict[str, object]]:
    labels = (
        "phase0_audit",
        "phase0_5_original_gate",
        "phase0_5_internal_screening",
        "phase1_metrics",
        "phase2_model_adequacy",
        "phase3_methods",
        "phase4_outcomes",
        "phase5_power",
        "phase6_publication",
    )
    rows = []
    for index, label in enumerate(labels):
        rows.append(
            {
                "node_id": label,
                "label": label,
                "state": "complete" if label != "phase5_power" else "not_powered_for_phase5",
                "shape": "box",
                "color": "#336699",
                "authority_path": f"authority-{index}",
                "authority_sha256": f"{index + 3000:064x}",
                "x": str(index),
                "y": str(index % 3),
            }
        )
    return rows


def _make_figure4_edges() -> list[dict[str, object]]:
    return [
        {
            "edge_id": f"edge-{index}",
            "source_node_id": source,
            "target_node_id": target,
            "edge_style": style,
            "label": label,
            "authority_path": f"edge-authority-{index}",
            "authority_sha256": f"{index + 4000:064x}",
        }
        for index, (source, target, style, label) in enumerate(
            (
                ("phase0_audit", "phase0_5_original_gate", "solid", "preregistered"),
                ("phase0_5_original_gate", "phase0_5_internal_screening", "dashed", "authorized fallback"),
                ("phase0_5_internal_screening", "phase1_metrics", "solid", "continued"),
                ("phase1_metrics", "phase2_model_adequacy", "solid", "dependency"),
                ("phase2_model_adequacy", "phase3_methods", "dashed", "fallback"),
                ("phase3_methods", "phase4_outcomes", "solid", "dependency"),
                ("phase4_outcomes", "phase5_power", "solid", "dependency"),
                ("phase5_power", "phase6_publication", "dashed", "status-aware package"),
            )
        )
    ]


def _make_inputs(subject: types.ModuleType) -> object:
    return subject.PublicationCoreInputs(
        table1_rows=tuple(_make_table1_rows()),
        table_s1_rows=tuple(
            _make_generic_rows(
                143,
                "s1",
                (
                    "panel_id",
                    "endpoint_id",
                    "protocol_id",
                    "metric_output_id",
                    "ag",
                    "ag_lower",
                    "ag_upper",
                    "acc_cross",
                    "acc_cross_lower",
                    "acc_cross_upper",
                    "d_ag",
                    "d_ag_lower",
                    "d_ag_upper",
                    "d_ag_raw_p_value",
                    "d_ag_adjusted_p_value",
                    "d_ag_rank",
                    "d_ag_family_size",
                    "d_ag_family_id",
                    "d_ag_favorable",
                    "d_ag_rejected",
                    "d_ag_direction",
                    "d_ag_state",
                    "d_ag_glyph",
                    "d_acc",
                    "d_acc_lower",
                    "d_acc_upper",
                    "d_acc_raw_p_value",
                    "d_acc_adjusted_p_value",
                    "d_acc_rank",
                    "d_acc_family_size",
                    "d_acc_family_id",
                    "d_acc_favorable",
                    "d_acc_rejected",
                    "d_acc_direction",
                    "d_acc_state",
                    "d_acc_glyph",
                    "panel_state",
                ),
            )
        ),
        table_s2_rows=tuple(
            _make_generic_rows(
                65,
                "s2",
                (
                    "cell_id",
                    "metric_output_id",
                    "delta_ag",
                    "delta_ag_lower",
                    "delta_ag_upper",
                    "delta_acc",
                    "delta_acc_lower",
                    "delta_acc_upper",
                    "i_ag",
                    "i_ag_lower",
                    "i_ag_upper",
                    "i_acc",
                    "i_acc_lower",
                    "i_acc_upper",
                    "i_ag_raw_p_value",
                    "i_ag_adjusted_p_value",
                    "i_ag_rank",
                    "i_ag_family_size",
                    "i_ag_family_id",
                    "i_ag_favorable",
                    "i_ag_rejected",
                    "i_ag_direction",
                    "i_ag_state",
                    "i_ag_glyph",
                    "i_acc_raw_p_value",
                    "i_acc_adjusted_p_value",
                    "i_acc_rank",
                    "i_acc_family_size",
                    "i_acc_family_id",
                    "i_acc_favorable",
                    "i_acc_rejected",
                    "i_acc_direction",
                    "i_acc_state",
                    "i_acc_glyph",
                    "state",
                ),
            )
        ),
        table2_rows=tuple(
            _make_generic_rows(
                1656,
                "table2",
                (
                    "evidence_id",
                    "task_line",
                    "system_id",
                    "family_id",
                    "endpoint_id",
                    "protocol_id",
                    "cohort_id",
                    "evidence_component",
                    "metric_output_id",
                    "estimate",
                    "interval_lower",
                    "interval_upper",
                    "preferred_direction",
                    "state",
                    "reason_code",
                    "phase5_power_state",
                    "source_path",
                    "source_sha256",
                ),
            )
        ),
        table_s3_rows=tuple(_make_system_status_rows()),
        figure3_rows=tuple(
            {
                "extractor_id": "airpls" if index < 4 else "arpls" if index < 8 else "mor",
                "excitation_stratum": ("green_514", "green_532", "nir_780", "nir_785")[index % 4],
                "input_count": "2341",
                "valid_count": "2341",
                "invalid_count": "0",
                "median_reconstruction_nrmse": format(0.02 + index / 100.0, ".17g"),
                "p95_reconstruction_nrmse": format(0.12 + index / 50.0, ".17g"),
                "p95_threshold": "0.25",
                "valid_fraction_state": "complete",
                "valid_count_state": "complete",
                "median_state": "complete",
                "p95_state": "pass" if index < 5 else "fail",
                "gate_state": "pass" if index < 5 else "fail_p95",
                "source_receipt_sha256": f"{index + 5000:064x}",
            }
            for index in range(12)
        ),
        figure4_node_rows=tuple(_make_figure4_nodes()),
        figure4_edge_rows=tuple(_make_figure4_edges()),
        captions={
            "table1": "Synthetic Table 1 caption",
            "table2": "Synthetic Table 2 caption",
            "figure3": "Synthetic Figure 3 caption",
            "figure4": "Synthetic Figure 4 caption",
        },
        authority_bridge={
            "synthetic_fixture": True,
            "fixed_parent_authorities": True,
        },
    )


class PublicationCoreTests(unittest.TestCase):
    def _load_module(self, module_name: str, relative_path: str) -> types.ModuleType:
        if "rpe" not in sys.modules:
            package = types.ModuleType("rpe")
            package.__path__ = [str(ROOT / "rpe")]
            package.__package__ = "rpe"
            package.__spec__ = importlib.machinery.ModuleSpec("rpe", loader=None, is_package=True)
            sys.modules["rpe"] = package
        if "rpe.runner" not in sys.modules:
            package = types.ModuleType("rpe.runner")
            package.__path__ = [str(ROOT / "rpe" / "runner")]
            package.__package__ = "rpe.runner"
            package.__spec__ = importlib.machinery.ModuleSpec("rpe.runner", loader=None, is_package=True)
            sys.modules["rpe.runner"] = package
        sys.modules.pop(module_name, None)
        spec = importlib.util.spec_from_file_location(module_name, ROOT / relative_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _load_subject(self) -> types.ModuleType:
        return self._load_module(
            SUBJECT_MODULE,
            "rpe/runner/phase6_publication_core.py",
        )

    def _load_verifier(self) -> types.ModuleType:
        return self._load_module(
            VERIFIER_MODULE,
            "rpe/runner/phase6_publication_core_verifier.py",
        )

    def _load_freeze_tool(self) -> types.ModuleType:
        return self._load_module(
            "tools.freeze_phase6_publication_core_config",
            "tools/freeze_phase6_publication_core_config.py",
        )

    def test_figure3_rows_derive_from_authoritative_phase2_receipts_and_gate(self) -> None:
        subject = self._load_subject()
        phase2_run = (
            ROOT
            / "results/phase2/background_fit/"
            / "phase2-background-fit-6f8dc92c8b3edbefe2a422b805214b6687990c3ac13a34fbaed3faaa3fad6cdd"
        )
        rows = subject.derive_figure3_model_adequacy_rows(phase2_run)
        self.assertEqual(len(rows), 12)
        by_key = {(row["extractor_id"], row["excitation_stratum"]): row for row in rows}
        airpls_514 = by_key[("airpls", "green_514")]
        self.assertAlmostEqual(float(airpls_514["median_reconstruction_nrmse"]), 0.020024842335016155)
        self.assertAlmostEqual(float(airpls_514["p95_reconstruction_nrmse"]), 0.12499345068991917)
        self.assertEqual(airpls_514["gate_state"], "pass")
        airpls_532 = by_key[("airpls", "green_532")]
        self.assertEqual(airpls_532["gate_state"], "fail_p95")
        self.assertEqual(sum(int(row["valid_count"]) for row in rows), 28092)
        self.assertEqual(sum(int(row["input_count"]) for row in rows), 28092)
        self.assertEqual(sum(row["gate_state"] == "pass" for row in rows), 5)
        self.assertEqual(sum(row["gate_state"] != "pass" for row in rows), 7)

    def test_synthetic_fixture_builds_expected_inventory_and_counts(self) -> None:
        subject = self._load_subject()
        config_document = _make_config_document()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            config_path = temp_root / "config.json"
            config_path.write_bytes(_canonical(config_document))
            config = subject.load_phase6_publication_core_config(config_path)
            inputs = _make_inputs(subject)
            summary = subject.build_phase6_publication_core_from_inputs(
                temp_root / "artifact",
                inputs=inputs,
                config=config,
                worker_count=3,
            )
            self.assertEqual(summary.status, "complete")
            self.assertEqual({path.name for path in summary.path.iterdir()}, EXPECTED_INVENTORY)
            manifest = json.loads((summary.path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["table1"], 12)
            self.assertEqual(manifest["counts"]["table_s1"], 143)
            self.assertEqual(manifest["counts"]["table_s2"], 65)
            self.assertEqual(manifest["counts"]["table2"], 1656)
            self.assertEqual(manifest["counts"]["table_s3"], 316)
            self.assertEqual(manifest["counts"]["figure3"], 12)
            self.assertEqual(len(_csv_rows(summary.path / "table1_metric_validity.csv")), 12)
            self.assertEqual(len(_csv_rows(summary.path / "table_s1_full_alignment.csv")), 143)
            self.assertEqual(len(_csv_rows(summary.path / "table_s2_protocol_interactions.csv")), 65)
            self.assertEqual(len(_csv_rows(summary.path / "table2_method_evidence.csv")), 1656)
            self.assertEqual(len(_csv_rows(summary.path / "table_s3_system_status.csv")), 316)
            self.assertEqual(len(_csv_rows(summary.path / "figure3_model_adequacy_data.csv")), 12)
            s3_rows = _csv_rows(summary.path / "table_s3_system_status.csv")
            self.assertEqual(sum(row["task_line"] == "deep_learning" for row in s3_rows), 10)
            self.assertEqual(sum(row["task_line"] != "deep_learning" for row in s3_rows), 306)
            d1_b = next(row for row in _csv_rows(summary.path / "table1_metric_validity.csv") if row["endpoint_id"] == "D1-B-closed")
            self.assertEqual(d1_b["state"], "not_evaluable_failed_alpha0_equivalence")
            self.assertEqual(d1_b["mse_ag"], "")
            _assert_real_figure3_visuals(self, summary.path)
            _assert_real_figure4_visuals(self, summary.path)

    def test_verifier_rebuilds_and_detects_tamper_without_importing_builder(self) -> None:
        subject = self._load_subject()
        config_document = _make_config_document()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_root = Path(tmpdir)
            config_path = temp_root / "config.json"
            config_path.write_bytes(_canonical(config_document))
            config = subject.load_phase6_publication_core_config(config_path)
            inputs = _make_inputs(subject)
            summary = subject.build_phase6_publication_core_from_inputs(
                temp_root / "artifact",
                inputs=inputs,
                config=config,
                worker_count=2,
            )

            blocker = types.ModuleType(SUBJECT_MODULE)

            def _blocked(name: str) -> object:
                raise AssertionError(f"verifier unexpectedly imported production module attribute {name}")

            blocker.__getattr__ = _blocked  # type: ignore[attr-defined]
            original = sys.modules.get(SUBJECT_MODULE)
            sys.modules.pop(VERIFIER_MODULE, None)
            sys.modules[SUBJECT_MODULE] = blocker
            try:
                verifier = self._load_verifier()
                verified = verifier.verify_phase6_publication_core_from_inputs(
                    summary.path,
                    inputs=inputs,
                    config_path=config_path,
                    worker_count=4,
                )
                self.assertEqual(verified.status, "complete")
                self.assertEqual(verified.verified_file_count, 21)
                manifest = summary.path / "manifest.json"
                manifest.write_text(manifest.read_text(encoding="utf-8").replace('"table1":12', '"table1":11'), encoding="utf-8")
                with self.assertRaises(verifier.PublicationCoreVerificationError):
                    verifier.verify_phase6_publication_core_from_inputs(
                        summary.path,
                        inputs=inputs,
                        config_path=config_path,
                        worker_count=1,
                    )
            finally:
                if original is None:
                    sys.modules.pop(SUBJECT_MODULE, None)
                else:
                    sys.modules[SUBJECT_MODULE] = original

    def test_public_formal_build_and_verify_support_owner_authorized_step7_deferred_report(self) -> None:
        subject = self._load_subject()
        freeze_tool = self._load_freeze_tool()
        with tempfile.TemporaryDirectory(dir=str(ROOT)) as tmpdir:
            temp_root = Path(tmpdir)
            deferred_report = temp_root / "reports/phase6/step07_appendix_audits.md"
            _make_step7_deferred_report(deferred_report)
            with self.assertRaises(SystemExit):
                freeze_tool.main(
                    [
                        "--step7-run-path",
                        str(temp_root / "step7_fixture"),
                        "--step7-deferred-report",
                        str(deferred_report),
                        "--output",
                        str(temp_root / "publication_core_mutex.json"),
                    ]
                )
            bad_config_path = temp_root / "publication_core_bad_v1.json"
            self.assertEqual(
                freeze_tool.main(
                    [
                        "--step7-deferred-report",
                        str(deferred_report),
                        "--output",
                        str(bad_config_path),
                    ]
                ),
                0,
            )
            deferred_report.write_text(
                deferred_report.read_text(encoding="utf-8").replace(
                    "state: deferred_by_owner",
                    "state: complete",
                    1,
                ),
                encoding="utf-8",
            )
            original_default = subject.DEFAULT_CONFIG
            subject.DEFAULT_CONFIG = bad_config_path
            try:
                with self.assertRaises(subject.PublicationCoreError):
                    subject.build_phase6_publication_core(temp_root / "bad-builds", worker_count=1)
            finally:
                subject.DEFAULT_CONFIG = original_default
            _make_step7_deferred_report(deferred_report)
            config_path = temp_root / "publication_core_v1.json"
            self.assertEqual(
                freeze_tool.main(
                    [
                        "--step7-deferred-report",
                        str(deferred_report),
                        "--output",
                        str(config_path),
                    ]
                ),
                0,
            )
            subject.DEFAULT_CONFIG = config_path
            try:
                summary = subject.build_phase6_publication_core(temp_root / "builds", worker_count=2)
            finally:
                subject.DEFAULT_CONFIG = original_default
            self.assertEqual(summary.status, "complete")
            self.assertEqual(summary.path.parent, temp_root / "builds")
            self.assertEqual(summary.path.name, summary.run_id)
            manifest = json.loads((summary.path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["table1"], 12)
            self.assertEqual(manifest["counts"]["table_s1"], 143)
            self.assertEqual(manifest["counts"]["table_s2"], 65)
            self.assertEqual(manifest["counts"]["table2"], 1656)
            self.assertEqual(manifest["counts"]["table_s3"], 316)
            self.assertEqual(manifest["counts"]["figure3"], 12)
            s3_rows = _csv_rows(summary.path / "table_s3_system_status.csv")
            self.assertEqual(sum(row["task_line"] == "deep_learning" for row in s3_rows), 10)
            self.assertEqual(sum(row["task_line"] != "deep_learning" for row in s3_rows), 306)
            table1_by_endpoint = {
                row["endpoint_id"]: row
                for row in _csv_rows(summary.path / "table1_metric_validity.csv")
            }
            self.assertEqual(
                {
                    endpoint_id: (row["cluster_kind"], row["cluster_count"])
                    for endpoint_id, row in table1_by_endpoint.items()
                },
                {
                    "D5-A": ("class", "681"),
                    "D5-B": ("class", "681"),
                    "D2-5-A": ("class", "30"),
                    "D2-5-B": ("class", "30"),
                    "D2-10-A": ("class", "30"),
                    "D2-10-B": ("class", "30"),
                    "D2-20-A": ("class", "30"),
                    "D2-20-B": ("class", "30"),
                    "D1-A": ("patient", "30"),
                    "D1-B-closed": ("patient", "30"),
                    "D4-A": ("well", "240"),
                    "D4-B": ("well", "240"),
                },
            )
            d1_b = table1_by_endpoint["D1-B-closed"]
            self.assertEqual(d1_b["state"], "not_evaluable_failed_alpha0_equivalence")
            self.assertEqual(d1_b["mse_ag"], "")
            config_document = json.loads((summary.path / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {key: value["path"] for key, value in config_document["report_authorities"].items()},
                FORMAL_REPORT_PATHS,
            )
            self.assertEqual(
                {key: value["path"] for key, value in config_document["code_receipts"].items()},
                FORMAL_CODE_PATHS,
            )
            self.assertEqual(config_document["step7_dependency"]["mode"], "deferred_report")
            self.assertEqual(config_document["step7_dependency"]["state"], "deferred_by_owner")
            self.assertEqual(
                config_document["step7_dependency"]["deferred_report"]["path"],
                str(deferred_report.relative_to(ROOT)),
            )
            self.assertNotIn("step7_run_path", config_document)
            self.assertNotIn("step7_appendix_audits", config_document)
            authority_bridge = json.loads((summary.path / "authority_bridge.json").read_text(encoding="utf-8"))
            self.assertEqual(
                {key: value["path"] for key, value in authority_bridge["report_authorities"].items()},
                FORMAL_REPORT_PATHS,
            )
            self.assertEqual(
                {key: value["path"] for key, value in authority_bridge["code_receipts"].items()},
                FORMAL_CODE_PATHS,
            )
            self.assertEqual(authority_bridge["step7_dependency"]["mode"], "deferred_report")
            self.assertEqual(authority_bridge["step7_dependency"]["state"], "deferred_by_owner")
            self.assertEqual(
                authority_bridge["step7_dependency"]["deferred_report"]["path"],
                str(deferred_report.relative_to(ROOT)),
            )
            manifest_dependency = manifest["step7_dependency"]
            self.assertEqual(manifest_dependency["mode"], "deferred_report")
            self.assertEqual(manifest_dependency["state"], "deferred_by_owner")
            self.assertEqual(
                manifest_dependency["authority_path"],
                str(deferred_report.relative_to(ROOT)),
            )
            node_rows = {
                row["node_id"]: row
                for row in _csv_rows(summary.path / "figure4_gate_nodes.csv")
            }
            self.assertEqual(node_rows["phase0_audit"]["authority_path"], FORMAL_REPORT_PATHS["phase0_source_audit_report"])
            self.assertEqual(node_rows["phase0_5_original_gate"]["authority_path"], FORMAL_REPORT_PATHS["phase05_internal_screening_decision_report"])
            self.assertEqual(node_rows["phase0_5_internal_screening"]["authority_path"], FORMAL_REPORT_PATHS["phase05_internal_screening_decision_report"])
            self.assertEqual(node_rows["phase1_metrics"]["authority_path"], FORMAL_REPORT_PATHS["phase1_core10k_report"])
            self.assertEqual(node_rows["phase3_methods"]["authority_path"], FORMAL_REPORT_PATHS["phase3_status_report"])
            self.assertEqual(node_rows["phase4_outcomes"]["authority_path"], FORMAL_REPORT_PATHS["phase4_closure_synthesis_report"])
            self.assertEqual(node_rows["phase6_publication"]["authority_path"], FORMAL_REPORT_PATHS["phase6_step1_design_report"])
            forbidden_phase4 = "results/phase4/final_figures_v1/phase4-final-figures-04ae2a217ad7d5fc1beafd4742dbc5bf3794079aca2dadfc3d4cc9b46411e7c6"
            for node_id in ("phase0_audit", "phase0_5_original_gate", "phase0_5_internal_screening", "phase1_metrics"):
                self.assertNotEqual(node_rows[node_id]["authority_path"], forbidden_phase4)
            edge_rows = _csv_rows(summary.path / "figure4_gate_edges.csv")
            self.assertEqual(
                {row["authority_path"] for row in edge_rows},
                {FORMAL_REPORT_PATHS["phase6_step1_design_report"]},
            )
            blocker = types.ModuleType(SUBJECT_MODULE)

            def _blocked(name: str) -> object:
                raise AssertionError(f"verifier unexpectedly imported production module attribute {name}")

            blocker.__getattr__ = _blocked  # type: ignore[attr-defined]
            authority_module = sys.modules["rpe.runner.phase6_publication_core_authority"]
            render_names = (
                "render_figure3_png",
                "render_figure3_svg",
                "render_figure4_png",
                "render_figure4_svg",
            )
            saved_renders = {name: getattr(authority_module, name) for name in render_names}

            def _blocked_render(*args: object, **kwargs: object) -> object:
                raise AssertionError("verifier unexpectedly used shared rendering function")

            original = sys.modules.get(SUBJECT_MODULE)
            sys.modules.pop(VERIFIER_MODULE, None)
            sys.modules[SUBJECT_MODULE] = blocker
            for name in render_names:
                setattr(authority_module, name, _blocked_render)
            try:
                verifier = self._load_verifier()
                verified = verifier.verify_phase6_publication_core(summary.path, worker_count=3)
                self.assertEqual(verified.status, "complete")
                self.assertEqual(verified.run_id, summary.run_id)
                deferred_report.write_text(
                    deferred_report.read_text(encoding="utf-8").replace(
                        "state: deferred_by_owner",
                        "state: complete",
                        1,
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(verifier.PublicationCoreVerificationError):
                    verifier.verify_phase6_publication_core(summary.path, worker_count=1)
            finally:
                for name, value in saved_renders.items():
                    setattr(authority_module, name, value)
                if original is None:
                    sys.modules.pop(SUBJECT_MODULE, None)
                else:
                    sys.modules[SUBJECT_MODULE] = original


if __name__ == "__main__":
    unittest.main()
