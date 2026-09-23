from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import io
import inspect
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SUBJECT_MODULE = "rpe.runner.phase4_final_figures"
VERIFIER_MODULE = "rpe.runner.phase4_final_figures_verifier"
CLI_MODULE = "tools.run_phase4_final_figures"
SUBJECT_FILE = ROOT / "rpe/runner/phase4_final_figures.py"
VERIFIER_FILE = ROOT / "rpe/runner/phase4_final_figures_verifier.py"
CLI_FILE = ROOT / "tools/run_phase4_final_figures.py"
REAL_CONFIG_PATH = ROOT / "experiments/phase4/configs/final_figures_v1.json"

DESIGN_REPORT_PATH = ROOT / "reports/phase4/step36_final_figure_integration_design.md"
PLAN_PATH = ROOT / "docs/superpowers/plans/2026-08-27-phase4-final-figures.md"

EXPECTED_SCOPE_IDS = (
    "p01_p05_not_evaluable_coverage",
    "p06_p07_structurally_ineligible_missing_explicit_baseline",
    "d3_inactive",
    "d1_protocol_b_failed_alpha0_equivalence",
    "d5_protocol_a_peak_common_primary_not_evaluable_coverage",
    "phase4_confirmatory_success_not_evaluable",
)
EXPECTED_ENDPOINT_ORDER = ("d5", "d2_5", "d2_10", "d2_20", "d1", "d4")
EXPECTED_PAIRED_CELL_ORDER = ("d5", "d2_5", "d2_10", "d2_20", "d4")
EXPECTED_PROTOCOL_ORDER = ("a", "b")
EXPECTED_PERTURBATION_ORDER = ("p08", "p09", "p10", "p11", "p12")
EXPECTED_ALPHA_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80)
EXPECTED_METRIC_ORDER = (
    "mse",
    "rmse",
    "mae",
    "sam",
    "pearson_r",
    "nmse",
    "wasserstein_1_cm1",
    "is_like_structure_to_noise",
    "precision",
    "recall",
    "f1",
    "artifact_peak_ratio",
    "missing_peak_ratio",
)
EXPECTED_SCOPE_ROW_COUNT = 6
EXPECTED_PARENT_PANEL_INDEX_ROW_COUNT = 12
EXPECTED_FIGURE1_RESPONSE_ROW_COUNT = 440
EXPECTED_FIGURE1_PROTOCOL_EFFECT_ROW_COUNT = 225
EXPECTED_FIGURE2_ALIGNMENT_ROW_COUNT = 143
EXPECTED_FIGURE2_PROTOCOL_INTERACTION_ROW_COUNT = 65
EXPECTED_PARENT_PANEL_ROWS = 12
EXPECTED_FIGURE1_SIZE = (5400, 3300)
EXPECTED_FIGURE2_SIZE = (5400, 3900)
EXPECTED_ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "scope_status.csv",
    "parent_panel_index.csv",
    "figure1_response_data.csv",
    "figure1_protocol_effect_data.csv",
    "figure1_phase4_response.png",
    "figure1_phase4_response.svg",
    "figure2_alignment_data.csv",
    "figure2_protocol_interaction_data.csv",
    "figure2_phase4_alignment.png",
    "figure2_phase4_alignment.svg",
    "captions.json",
    "manifest.json",
)
EXPECTED_FIGURE_TEXT = (
    "full_domain_core: P8-P12 only",
    "matched reference",
    "closed: failed exact alpha-zero equivalence; no positive-alpha outcome",
    "Task-native Phase-4 MSE/downstream response.",
    "Task-native metric alignment and secondary protocol interaction.",
)
EXPECTED_ENDPOINT_TITLES = ("D5", "D2-5", "D2-10", "D2-20", "D1", "D4")
EXPECTED_PERTURBATION_LABELS = ("P08", "P09", "P10", "P11", "P12")
EXPECTED_FIGURE2_COLUMN_LABELS = ("AG", "Acc-cross", "I_AG", "I_Acc", "Protocol A", "Protocol B")
EXPECTED_PROTOCOL_EFFECT_CLASSIFICATIONS = (
    "attenuation_by_b_rejected",
    "amplification_by_b_rejected",
    "not_rejected",
)
EXPECTED_WITHIN_PROTOCOL_GLYPHS = (
    "favorable_rejected",
    "adverse_rejected",
    "not_rejected",
)


def _hex_alpha(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _rewrite_sha256sums(path: Path) -> None:
    terminal_name = "complete.json" if (path / "complete.json").exists() else "failed.json"
    ordered = [*EXPECTED_ARTIFACT_PAYLOAD_FILES, terminal_name]
    lines = [
        f"{_sha256_bytes((path / name).read_bytes())}  {name}"
        for name in ordered
    ]
    (path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _csv_dict_rows(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def _assert_png_has_nontrivial_pixel_diversity(test_case: unittest.TestCase, raw: bytes) -> None:
    from PIL import Image

    with Image.open(io.BytesIO(raw)) as image:
        reduced = image.convert("RGBA").resize((128, 128))
        colors = reduced.getcolors(maxcolors=128 * 128)
    test_case.assertIsNotNone(colors, "rendered PNG must contain a bounded but diverse palette")
    assert colors is not None
    test_case.assertGreater(
        len(colors),
        32,
        "text-only or nearly blank PNGs collapse to too few colors",
    )


def _assert_svg_contains_real_plot_content(
    test_case: unittest.TestCase, raw: bytes, *, require_all_metrics: bool,
    require_perturbations: bool,
) -> None:
    text = raw.decode("utf-8")
    for title in EXPECTED_ENDPOINT_TITLES:
        test_case.assertIn(title, text)
    if require_perturbations:
        for label in EXPECTED_PERTURBATION_LABELS:
            test_case.assertIn(label, text)
    if require_all_metrics:
        for label in EXPECTED_METRIC_ORDER:
            test_case.assertIn(label, text)
    for label in EXPECTED_FIGURE2_COLUMN_LABELS:
        test_case.assertIn(label, text)
    test_case.assertGreaterEqual(text.count("<g"), 16)
    test_case.assertGreaterEqual(
        text.count("<path") + text.count("<circle") + text.count("<line") + text.count("<rect") + text.count("<use"),
        64,
        "SVG must contain real plot primitives and glyph groups, not only text",
    )
    test_case.assertTrue(
        any(token in text for token in ("line2d_", "PathCollection", "patch_", "clip-path")),
        "SVG should expose plotted trajectories, markers, or panel patches",
    )


def _load_subject():
    return importlib.import_module(SUBJECT_MODULE)


def _load_verifier():
    return importlib.import_module(VERIFIER_MODULE)


def _load_cli():
    return importlib.import_module(CLI_MODULE)


def _make_synthetic_config_document() -> dict[str, object]:
    return {
        "schema_version": "phase4-final-figures-config-v1",
        "experiment_id": "phase4-final-figures-v1",
        "claim_boundary": (
            "post_outcome_final_figure_secondary_phase4_no_cross_endpoint_claim"
        ),
        "artifact_payload_files": list(EXPECTED_ARTIFACT_PAYLOAD_FILES),
        "scope_ids": list(EXPECTED_SCOPE_IDS),
        "endpoint_order": list(EXPECTED_ENDPOINT_ORDER),
        "paired_cell_order": list(EXPECTED_PAIRED_CELL_ORDER),
        "protocol_order": list(EXPECTED_PROTOCOL_ORDER),
        "perturbation_ids": list(EXPECTED_PERTURBATION_ORDER),
        "alpha_grid": list(EXPECTED_ALPHA_GRID),
        "metric_output_ids": list(EXPECTED_METRIC_ORDER),
        "expected": {
            "scope_status_row_count": EXPECTED_SCOPE_ROW_COUNT,
            "parent_panel_index_row_count": EXPECTED_PARENT_PANEL_INDEX_ROW_COUNT,
            "figure1_response_row_count": EXPECTED_FIGURE1_RESPONSE_ROW_COUNT,
            "figure1_protocol_effect_row_count": EXPECTED_FIGURE1_PROTOCOL_EFFECT_ROW_COUNT,
            "figure2_alignment_row_count": EXPECTED_FIGURE2_ALIGNMENT_ROW_COUNT,
            "figure2_protocol_interaction_row_count": EXPECTED_FIGURE2_PROTOCOL_INTERACTION_ROW_COUNT,
            "configured_payload_count": len(EXPECTED_ARTIFACT_PAYLOAD_FILES),
            "artifact_file_count": len(EXPECTED_ARTIFACT_PAYLOAD_FILES) + 2,
        },
        "figure1": {
            "width_px": EXPECTED_FIGURE1_SIZE[0],
            "height_px": EXPECTED_FIGURE1_SIZE[1],
            "dpi": 300,
        },
        "figure2": {
            "width_px": EXPECTED_FIGURE2_SIZE[0],
            "height_px": EXPECTED_FIGURE2_SIZE[1],
            "dpi": 300,
        },
        "required_text": list(EXPECTED_FIGURE_TEXT),
        "style": {
            "perturbation_palette": {
                "p08": {"color": "#0072B2", "marker": "o"},
                "p09": {"color": "#56B4E9", "marker": "s"},
                "p10": {"color": "#009E73", "marker": "^"},
                "p11": {"color": "#E69F00", "marker": "D"},
                "p12": {"color": "#D55E00", "marker": "p"},
            },
            "protocol_effect_colors": {
                "attenuation_by_b_rejected": "#2166AC",
                "amplification_by_b_rejected": "#B2182B",
                "not_rejected": "#D0D0D0",
            },
        },
        "authorities": {
            "design_report": {
                "path": str(DESIGN_REPORT_PATH.relative_to(ROOT)),
                "sha256": _sha256_bytes(DESIGN_REPORT_PATH.read_bytes()),
                "bytes": DESIGN_REPORT_PATH.stat().st_size,
            },
            "implementation_plan": {
                "path": str(PLAN_PATH.relative_to(ROOT)),
                "sha256": _sha256_bytes(PLAN_PATH.read_bytes()),
                "bytes": PLAN_PATH.stat().st_size,
            },
        },
    }


def _make_synthetic_inputs_document() -> dict[str, object]:
    def response_rows(cell_id: str, protocol: str) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for perturbation_index, perturbation_id in enumerate(EXPECTED_PERTURBATION_ORDER):
            for alpha_index, alpha in enumerate(EXPECTED_ALPHA_GRID):
                metric_x = 0.1 + perturbation_index + alpha_index / 100.0
                rows.append(
                    {
                        "cell_id": cell_id,
                        "protocol_id": protocol,
                        "perturbation_id": perturbation_id,
                        "alpha": alpha,
                        "metric_output_id": "mse",
                        "protocol_a_mse_x": round(metric_x, 6),
                        "metric_x": round(metric_x + (0.5 if protocol == "b" and cell_id == "d4" else 0.0), 6),
                        "downstream_harm": round(2.0 + perturbation_index + alpha_index / 10.0, 6),
                    }
                )
        return rows

    figure1_protocol_effect_rows: list[dict[str, object]] = []
    for cell_index, cell_id in enumerate(EXPECTED_PAIRED_CELL_ORDER):
        for perturbation_index, perturbation_id in enumerate(EXPECTED_PERTURBATION_ORDER):
            for alpha_index, alpha in enumerate(EXPECTED_ALPHA_GRID):
                figure1_protocol_effect_rows.append(
                    {
                        "cell_id": cell_id,
                        "perturbation_id": perturbation_id,
                        "alpha": alpha,
                        "summary_type": "alpha_specific",
                        "g_harm": round((cell_index - perturbation_index) / 10.0, 6),
                        "classification": EXPECTED_PROTOCOL_EFFECT_CLASSIFICATIONS[
                            (cell_index + perturbation_index + alpha_index)
                            % len(EXPECTED_PROTOCOL_EFFECT_CLASSIFICATIONS)
                        ],
                    }
                )
            figure1_protocol_effect_rows.append(
                {
                    "cell_id": cell_id,
                    "perturbation_id": perturbation_id,
                    "alpha": None,
                    "summary_type": "integrated",
                    "g_harm": round((cell_index - perturbation_index) / 10.0, 6),
                    "classification": EXPECTED_PROTOCOL_EFFECT_CLASSIFICATIONS[
                        (cell_index + perturbation_index)
                        % len(EXPECTED_PROTOCOL_EFFECT_CLASSIFICATIONS)
                    ],
                }
            )

    figure2_alignment_rows: list[dict[str, object]] = []
    figure2_interaction_rows: list[dict[str, object]] = []
    for endpoint_index, endpoint_id in enumerate(EXPECTED_ENDPOINT_ORDER):
        for metric_index, metric_output_id in enumerate(EXPECTED_METRIC_ORDER):
            for protocol_id in (("a",) if endpoint_id == "d1" else EXPECTED_PROTOCOL_ORDER):
                favorable = (metric_index + (protocol_id == "b")) % 3 == 0
                rejected = metric_index % 3 != 2
                row = {
                    "panel_id": f"{endpoint_id}_{protocol_id}",
                    "endpoint_id": endpoint_id,
                    "protocol_id": protocol_id,
                    "metric_output_id": metric_output_id,
                    "ag": round(0.50 + endpoint_index / 100 + metric_index / 1000 + (0.03 if protocol_id == "b" else 0), 6),
                    "ag_lower": round(0.45 + endpoint_index / 100 + metric_index / 1000 + (0.03 if protocol_id == "b" else 0), 6),
                    "ag_upper": round(0.55 + endpoint_index / 100 + metric_index / 1000 + (0.03 if protocol_id == "b" else 0), 6),
                    "acc_cross": round(0.60 + endpoint_index / 100 + metric_index / 1000, 6),
                    "acc_cross_lower": round(0.57 + endpoint_index / 100 + metric_index / 1000, 6),
                    "acc_cross_upper": round(0.63 + endpoint_index / 100 + metric_index / 1000, 6),
                    "panel_state": "complete",
                }
                for statistic, statistic_favorable in (("d_ag", favorable), ("d_acc", not favorable)):
                    row.update({
                        statistic: round((metric_index - 1) / 100, 6),
                        f"{statistic}_lower": round((metric_index - 1) / 100 - 0.01, 6),
                        f"{statistic}_upper": round((metric_index - 1) / 100 + 0.01, 6),
                        f"{statistic}_raw_p_value": round(0.001 + metric_index / 1000, 6),
                        f"{statistic}_adjusted_p_value": round(0.011 + metric_index / 1000, 6),
                        f"{statistic}_rank": metric_index + 1,
                        f"{statistic}_family_size": 24,
                        f"{statistic}_family_id": f"{endpoint_id}:{protocol_id}:within_protocol",
                        f"{statistic}_favorable": statistic_favorable,
                        f"{statistic}_rejected": rejected,
                        f"{statistic}_direction": "favorable" if statistic_favorable else "adverse",
                        f"{statistic}_state": "complete",
                        f"{statistic}_glyph": "favorable_rejected" if rejected and statistic_favorable else "adverse_rejected" if rejected else "not_rejected",
                    })
                figure2_alignment_rows.append(row)
    for cell_index, cell_id in enumerate(EXPECTED_PAIRED_CELL_ORDER):
        for metric_index, metric_output_id in enumerate(EXPECTED_METRIC_ORDER):
            figure2_interaction_rows.append(
                {
                    "cell_id": cell_id,
                    "metric_output_id": metric_output_id,
                    "delta_ag": round((cell_index - metric_index) / 20.0, 6),
                    "delta_ag_lower": round((cell_index - metric_index) / 20.0 - 0.03, 6),
                    "delta_ag_upper": round((cell_index - metric_index) / 20.0 + 0.03, 6),
                    "delta_acc": round((metric_index - cell_index) / 30.0, 6),
                    "delta_acc_lower": round((metric_index - cell_index) / 30.0 - 0.02, 6),
                    "delta_acc_upper": round((metric_index - cell_index) / 30.0 + 0.02, 6),
                    "i_ag": round((cell_index - metric_index) / 40.0, 6),
                    "i_ag_lower": round((cell_index - metric_index) / 40.0 - 0.04, 6),
                    "i_ag_upper": round((cell_index - metric_index) / 40.0 + 0.04, 6),
                    "i_acc": round((metric_index - cell_index) / 50.0, 6),
                    "i_acc_lower": round((metric_index - cell_index) / 50.0 - 0.05, 6),
                    "i_acc_upper": round((metric_index - cell_index) / 50.0 + 0.05, 6),
                    **{
                        f"{statistic}_{field}": value
                        for statistic, values in (
                            ("i_ag", {"raw_p_value": 0.003, "adjusted_p_value": 0.033, "rank": metric_index + 1, "family_size": 29, "family_id": f"{cell_id}:protocol_interaction", "favorable": True, "rejected": metric_index % 3 != 2, "direction": "favorable", "state": "complete", "glyph": "favorable_rejected" if metric_index % 3 != 2 else "not_rejected"}),
                            ("i_acc", {"raw_p_value": 0.004, "adjusted_p_value": 0.034, "rank": metric_index + 1, "family_size": 29, "family_id": f"{cell_id}:protocol_interaction", "favorable": False, "rejected": metric_index % 3 != 2, "direction": "adverse", "state": "complete", "glyph": "adverse_rejected" if metric_index % 3 != 2 else "not_rejected"}),
                        )
                        for field, value in values.items()
                    },
                }
            )

    return {
        "scope_status_rows": [
            {"scope_id": scope_id, "state": "closed"} for scope_id in EXPECTED_SCOPE_IDS
        ],
        "parent_panel_rows": [
            {"panel_id": "d5_a", "endpoint_id": "d5", "protocol_id": "a", "numerical_source_allowed": True},
            {"panel_id": "d5_b", "endpoint_id": "d5", "protocol_id": "b", "numerical_source_allowed": True},
            {"panel_id": "d2_5_a", "endpoint_id": "d2_5", "protocol_id": "a", "numerical_source_allowed": True},
            {"panel_id": "d2_5_b", "endpoint_id": "d2_5", "protocol_id": "b", "numerical_source_allowed": True},
            {"panel_id": "d2_10_a", "endpoint_id": "d2_10", "protocol_id": "a", "numerical_source_allowed": True},
            {"panel_id": "d2_10_b", "endpoint_id": "d2_10", "protocol_id": "b", "numerical_source_allowed": True},
            {"panel_id": "d2_20_a", "endpoint_id": "d2_20", "protocol_id": "a", "numerical_source_allowed": True},
            {"panel_id": "d2_20_b", "endpoint_id": "d2_20", "protocol_id": "b", "numerical_source_allowed": True},
            {"panel_id": "d1_a", "endpoint_id": "d1", "protocol_id": "a", "numerical_source_allowed": True},
            {"panel_id": "d1_b_closed", "endpoint_id": "d1", "protocol_id": "b", "numerical_source_allowed": False},
            {"panel_id": "d4_a", "endpoint_id": "d4", "protocol_id": "a", "numerical_source_allowed": True},
            {"panel_id": "d4_b", "endpoint_id": "d4", "protocol_id": "b", "numerical_source_allowed": True},
        ],
        "figure1_response_rows": [
            *response_rows("d5", "a"),
            *response_rows("d5", "b"),
            *response_rows("d2_5", "a"),
            *response_rows("d2_5", "b"),
            *response_rows("d2_10", "a"),
            *response_rows("d2_10", "b"),
            *response_rows("d2_20", "a"),
            *response_rows("d2_20", "b"),
            *response_rows("d1", "a"),
            *response_rows("d4", "a"),
            *response_rows("d4", "b"),
        ],
        "figure1_protocol_effect_rows": figure1_protocol_effect_rows,
        "figure2_alignment_rows": figure2_alignment_rows,
        "figure2_interaction_rows": figure2_interaction_rows,
        "captions": {
            "figure1": EXPECTED_FIGURE_TEXT[3],
            "figure2": EXPECTED_FIGURE_TEXT[4],
        },
    }


class Phase4FinalFiguresContractTest(unittest.TestCase):
    def test_subject_surface_exists_with_expected_public_api(self) -> None:
        subject = _load_subject()
        self.assertTrue(hasattr(subject, "Phase4FinalFiguresError"))
        self.assertTrue(hasattr(subject, "Phase4FinalFiguresConfig"))
        self.assertTrue(hasattr(subject, "Phase4FinalFigureInputs"))
        self.assertTrue(hasattr(subject, "Phase4FinalFiguresSummary"))
        self.assertTrue(hasattr(subject, "parse_phase4_final_figures_config"))
        self.assertTrue(hasattr(subject, "load_phase4_final_figures_config"))
        self.assertTrue(hasattr(subject, "reconstruct_phase4_final_figure_inputs"))
        self.assertTrue(hasattr(subject, "build_phase4_final_figures_from_inputs"))
        self.assertTrue(hasattr(subject, "build_phase4_final_figures"))

    def test_synthetic_config_freezes_scopes_counts_orders_and_canvas_sizes(self) -> None:
        subject = _load_subject()

        config = subject.parse_phase4_final_figures_config(
            Path("synthetic.json"),
            _canonical(_make_synthetic_config_document()),
            require_frozen_identity=False,
        )
        self.assertEqual(config.scope_ids, EXPECTED_SCOPE_IDS)
        self.assertEqual(config.endpoint_order, EXPECTED_ENDPOINT_ORDER)
        self.assertEqual(config.paired_cell_order, EXPECTED_PAIRED_CELL_ORDER)
        self.assertEqual(config.protocol_order, EXPECTED_PROTOCOL_ORDER)
        self.assertEqual(config.perturbation_ids, EXPECTED_PERTURBATION_ORDER)
        self.assertEqual(config.alpha_grid, EXPECTED_ALPHA_GRID)
        self.assertEqual(config.metric_output_ids, EXPECTED_METRIC_ORDER)
        self.assertEqual(config.document["expected"]["scope_status_row_count"], EXPECTED_SCOPE_ROW_COUNT)
        self.assertEqual(
            config.document["expected"]["parent_panel_index_row_count"],
            EXPECTED_PARENT_PANEL_INDEX_ROW_COUNT,
        )
        self.assertEqual(
            config.document["expected"]["figure1_response_row_count"],
            EXPECTED_FIGURE1_RESPONSE_ROW_COUNT,
        )
        self.assertEqual(
            config.document["expected"]["figure1_protocol_effect_row_count"],
            EXPECTED_FIGURE1_PROTOCOL_EFFECT_ROW_COUNT,
        )
        self.assertEqual(
            config.document["expected"]["figure2_alignment_row_count"],
            EXPECTED_FIGURE2_ALIGNMENT_ROW_COUNT,
        )
        self.assertEqual(
            config.document["expected"]["figure2_protocol_interaction_row_count"],
            EXPECTED_FIGURE2_PROTOCOL_INTERACTION_ROW_COUNT,
        )
        self.assertEqual(
            (config.document["figure1"]["width_px"], config.document["figure1"]["height_px"]),
            EXPECTED_FIGURE1_SIZE,
        )
        self.assertEqual(
            (config.document["figure2"]["width_px"], config.document["figure2"]["height_px"]),
            EXPECTED_FIGURE2_SIZE,
        )

    def test_synthetic_projection_contract_keeps_order_scope_counts_and_d1b_closure(self) -> None:
        subject = _load_subject()

        config = subject.parse_phase4_final_figures_config(
            Path("synthetic.json"),
            _canonical(_make_synthetic_config_document()),
            require_frozen_identity=False,
        )
        inputs = subject.make_synthetic_phase4_final_figure_inputs(
            document=_make_synthetic_inputs_document(),
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = subject.build_phase4_final_figures_from_inputs(
                Path(directory) / "artifact",
                inputs=inputs,
                config=config,
                worker_count=1,
            )
            manifest = json.loads((summary.path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["counts"],
                {
                    "scope_status": EXPECTED_SCOPE_ROW_COUNT,
                    "parent_panel_index": EXPECTED_PARENT_PANEL_INDEX_ROW_COUNT,
                    "figure1_response_data": EXPECTED_FIGURE1_RESPONSE_ROW_COUNT,
                    "figure1_protocol_effect_data": EXPECTED_FIGURE1_PROTOCOL_EFFECT_ROW_COUNT,
                    "figure2_alignment_data": EXPECTED_FIGURE2_ALIGNMENT_ROW_COUNT,
                    "figure2_protocol_interaction_data": EXPECTED_FIGURE2_PROTOCOL_INTERACTION_ROW_COUNT,
                    "configured_payloads": len(EXPECTED_ARTIFACT_PAYLOAD_FILES),
                    "artifact_files": len(EXPECTED_ARTIFACT_PAYLOAD_FILES) + 2,
                },
            )
            scope_rows = (summary.path / "scope_status.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(scope_rows), 1 + EXPECTED_SCOPE_ROW_COUNT)
            self.assertEqual(
                [line.split(",", 1)[0] for line in scope_rows[1:]],
                list(EXPECTED_SCOPE_IDS),
            )
            panel_rows = (summary.path / "parent_panel_index.csv").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(panel_rows), 1 + EXPECTED_PARENT_PANEL_ROWS)
            self.assertIn("d1_b_closed", panel_rows[-3])
            self.assertIn(",False", panel_rows[-3])
            self.assertNotIn("d1,b,", (summary.path / "figure1_response_data.csv").read_text(encoding="utf-8"))

    def test_protocol_a_x_authority_and_d4_b_reconciliation_are_contractual(self) -> None:
        subject = _load_subject()

        config = subject.parse_phase4_final_figures_config(
            Path("synthetic.json"),
            _canonical(_make_synthetic_config_document()),
            require_frozen_identity=False,
        )
        inputs = subject.make_synthetic_phase4_final_figure_inputs(
            document=_make_synthetic_inputs_document(),
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = subject.build_phase4_final_figures_from_inputs(
                Path(directory) / "artifact",
                inputs=inputs,
                config=config,
                worker_count=1,
            )
            rows = (summary.path / "figure1_response_data.csv").read_text(encoding="utf-8").splitlines()
            header = rows[0].split(",")
            self.assertIn("protocol_a_mse_x", header)
            self.assertIn("metric_x", header)
            body = [dict(zip(header, row.split(","))) for row in rows[1:]]
            d4_b = [row for row in body if row["cell_id"] == "d4" and row["protocol_id"] == "b"]
            self.assertEqual(len(d4_b), 40)
            self.assertTrue(
                all(row["protocol_a_mse_x"] != row["metric_x"] for row in d4_b),
                "D4-B must retain parent-B metric as QC only; x stays on Protocol-A authority",
            )
            self.assertTrue(
                all(row["x_render_value"] == row["protocol_a_mse_x"] for row in d4_b),
                "render x must come from the Protocol-A coordinate",
            )

    def test_figure_text_sizes_inventory_and_bytes_are_deterministic(self) -> None:
        subject = _load_subject()

        config = subject.parse_phase4_final_figures_config(
            Path("synthetic.json"),
            _canonical(_make_synthetic_config_document()),
            require_frozen_identity=False,
        )
        inputs = subject.make_synthetic_phase4_final_figure_inputs(
            document=_make_synthetic_inputs_document(),
        )
        snapshots = []
        for worker_count in (1, 2):
            with tempfile.TemporaryDirectory() as directory:
                summary = subject.build_phase4_final_figures_from_inputs(
                    Path(directory) / "artifact",
                    inputs=inputs,
                    config=config,
                    worker_count=worker_count,
                )
                files = {
                    path.name: path.read_bytes()
                    for path in summary.path.iterdir()
                    if path.is_file()
                }
                self.assertEqual(
                    set(files),
                    set(EXPECTED_ARTIFACT_PAYLOAD_FILES) | {"complete.json", "SHA256SUMS"},
                )
                self.assertEqual(len(files), 17)
                captions = json.loads(files["captions.json"])
                self.assertIn("matched reference", json.dumps(captions, ensure_ascii=False))
                self.assertIn("failed exact alpha-zero equivalence", json.dumps(captions, ensure_ascii=False))
                self.assertEqual(
                    json.loads(files["manifest.json"])["figure1_size_px"],
                    list(EXPECTED_FIGURE1_SIZE),
                )
                self.assertEqual(
                    json.loads(files["manifest.json"])["figure2_size_px"],
                    list(EXPECTED_FIGURE2_SIZE),
                )
                for text in EXPECTED_FIGURE_TEXT:
                    blob = files["captions.json"] + files["figure1_phase4_response.svg"] + files["figure2_phase4_alignment.svg"]
                    self.assertIn(text.encode("utf-8"), blob)
                _assert_svg_contains_real_plot_content(
                    self, files["figure1_phase4_response.svg"],
                    require_all_metrics=False, require_perturbations=True,
                )
                _assert_svg_contains_real_plot_content(
                    self, files["figure2_phase4_alignment.svg"],
                    require_all_metrics=True, require_perturbations=False,
                )
                _assert_png_has_nontrivial_pixel_diversity(self, files["figure1_phase4_response.png"])
                _assert_png_has_nontrivial_pixel_diversity(self, files["figure2_phase4_alignment.png"])
                ledger = files["SHA256SUMS"].decode("utf-8").splitlines()
                self.assertEqual(len(ledger), 16)
                for line in ledger:
                    digest, name = line.split("  ", 1)
                    self.assertEqual(_sha256_bytes(files[name]), digest)
                snapshots.append((summary.run_id, files))
        self.assertEqual(snapshots[0], snapshots[1])

    def test_alignment_and_interaction_csvs_retain_exact_inference_fields(self) -> None:
        subject = _load_subject()

        config = subject.parse_phase4_final_figures_config(
            Path("synthetic.json"),
            _canonical(_make_synthetic_config_document()),
            require_frozen_identity=False,
        )
        inputs = subject.make_synthetic_phase4_final_figure_inputs(
            document=_make_synthetic_inputs_document(),
        )
        with tempfile.TemporaryDirectory() as directory:
            summary = subject.build_phase4_final_figures_from_inputs(
                Path(directory) / "artifact",
                inputs=inputs,
                config=config,
                worker_count=1,
            )
            alignment_rows = _csv_dict_rows((summary.path / "figure2_alignment_data.csv").read_bytes())
            interaction_rows = _csv_dict_rows((summary.path / "figure2_protocol_interaction_data.csv").read_bytes())
            self.assertEqual(len(alignment_rows), EXPECTED_FIGURE2_ALIGNMENT_ROW_COUNT)
            self.assertEqual(len(interaction_rows), EXPECTED_FIGURE2_PROTOCOL_INTERACTION_ROW_COUNT)
            alignment_header = tuple(alignment_rows[0].keys())
            for field_name in (
                "protocol_id", "ag_lower", "ag_upper",
                "acc_cross_lower", "acc_cross_upper",
                "d_ag_raw_p_value", "d_ag_adjusted_p_value",
                "d_ag_rank", "d_ag_family_size", "d_ag_family_id",
                "d_ag_favorable", "d_ag_rejected", "d_ag_direction",
                "d_ag_state", "d_ag_glyph",
                "d_acc_raw_p_value", "d_acc_adjusted_p_value",
                "d_acc_rank", "d_acc_family_size", "d_acc_family_id",
                "d_acc_favorable", "d_acc_rejected", "d_acc_direction",
                "d_acc_state", "d_acc_glyph", "panel_state",
            ):
                self.assertIn(field_name, alignment_header)
            interaction_header = tuple(interaction_rows[0].keys())
            for field_name in (
                "delta_ag_lower",
                "delta_ag_upper",
                "delta_acc_lower",
                "delta_acc_upper",
                "i_ag_lower",
                "i_ag_upper",
                "i_acc_lower",
                "i_acc_upper",
                "i_ag_raw_p_value", "i_ag_adjusted_p_value",
                "i_ag_rank", "i_ag_family_size", "i_ag_family_id",
                "i_ag_favorable", "i_ag_rejected", "i_ag_direction",
                "i_ag_state", "i_ag_glyph",
                "i_acc_raw_p_value", "i_acc_adjusted_p_value",
                "i_acc_rank", "i_acc_family_size", "i_acc_family_id",
                "i_acc_favorable", "i_acc_rejected", "i_acc_direction",
                "i_acc_state", "i_acc_glyph",
            ):
                self.assertIn(field_name, interaction_header)
            self.assertEqual(
                [row["protocol_id"] for row in alignment_rows if row["endpoint_id"] == "d1"],
                ["a"] * len(EXPECTED_METRIC_ORDER),
            )
            d5_mse = next(
                row for row in alignment_rows
                if row["endpoint_id"] == "d5" and row["protocol_id"] == "a"
                and row["metric_output_id"] == "mse"
            )
            self.assertEqual(d5_mse["d_ag_family_size"], "24")
            self.assertEqual(d5_mse["d_acc_family_size"], "24")
            self.assertEqual(d5_mse["d_ag_rank"], "")
            self.assertEqual(d5_mse["d_ag_direction"], "reference")
            self.assertEqual(d5_mse["d_ag_state"], "reference_no_contrast")
            d5_interaction = next(
                row for row in interaction_rows
                if row["cell_id"] == "d5" and row["metric_output_id"] == "mse"
            )
            self.assertEqual(d5_interaction["i_ag_family_size"], "29")
            self.assertEqual(d5_interaction["i_acc_family_size"], "29")
            self.assertEqual(d5_interaction["i_ag_rank"], "")
            self.assertEqual(d5_interaction["i_ag_direction"], "reference")
            self.assertEqual(d5_interaction["i_ag_state"], "reference_no_interaction")

    def test_real_config_reconstruction_surface_is_required(self) -> None:
        subject = _load_subject()

        config = subject.load_phase4_final_figures_config(REAL_CONFIG_PATH)
        inputs = subject.reconstruct_phase4_final_figure_inputs(config)
        self.assertEqual(config.path, REAL_CONFIG_PATH)
        self.assertEqual(len(inputs.scope_status_rows), EXPECTED_SCOPE_ROW_COUNT)
        self.assertEqual(len(inputs.parent_panel_rows), EXPECTED_PARENT_PANEL_INDEX_ROW_COUNT)


class Phase4FinalFiguresVerifierAndCliContractTest(unittest.TestCase):
    def test_verifier_is_independent_from_production_module(self) -> None:
        verifier = _load_verifier()

        tree = ast.parse(inspect.getsource(verifier))
        forbidden_modules = {
            SUBJECT_MODULE,
            "rpe.runner.phase4_final_figures_projection",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(
                    forbidden_modules.isdisjoint({alias.name for alias in node.names})
                )
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module, forbidden_modules)

    def test_verifier_rebuild_matches_and_detects_checksum_consistent_semantic_tampering(self) -> None:
        subject = _load_subject()
        verifier = _load_verifier()

        config = subject.parse_phase4_final_figures_config(
            Path("synthetic.json"),
            _canonical(_make_synthetic_config_document()),
            require_frozen_identity=False,
        )
        inputs = subject.make_synthetic_phase4_final_figure_inputs(
            document=_make_synthetic_inputs_document(),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            built = subject.build_phase4_final_figures_from_inputs(
                path,
                inputs=inputs,
                config=config,
                worker_count=1,
            )
            verified = verifier.verify_phase4_final_figures_from_inputs(
                built.path,
                inputs=inputs,
                config_path=built.path / "config.json",
                worker_count=1,
            )
            self.assertEqual(verified.run_id, built.run_id)

            rows = (built.path / "scope_status.csv").read_text(encoding="utf-8").splitlines()
            changed = rows[1].replace("p01_p05_not_evaluable_coverage", "tampered_scope")
            rows[1] = changed
            (built.path / "scope_status.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            _rewrite_sha256sums(built.path)
            with self.assertRaisesRegex(
                verifier.Phase4FinalFiguresVerifierError,
                "semantic|rebuild|payload",
            ):
                verifier.verify_phase4_final_figures_from_inputs(
                    built.path,
                    inputs=inputs,
                    config_path=built.path / "config.json",
                    worker_count=1,
                )

    def test_cli_surface_is_build_verify_only_and_verify_import_is_lazy(self) -> None:
        cli = _load_cli()
        self.assertTrue(hasattr(cli, "main"))
        with self.assertRaises(SystemExit):
            cli.main(["build"])
        with self.assertRaises(SystemExit):
            cli.main(["verify"])
        source = CLI_FILE.read_text(encoding="utf-8")
        self.assertIn('"build"', source)
        self.assertIn('"verify"', source)
        self.assertNotIn('"render"', source)
        self.assertNotIn('"repair"', source)
        self.assertNotIn("from rpe.runner.phase4_final_figures_verifier import", source)


if __name__ == "__main__":
    unittest.main()
