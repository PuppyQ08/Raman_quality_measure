from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import platform
import re
import struct
import zlib
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase6-publication-core-config-v1"
EXPERIMENT_ID = "phase6-publication-core-v1"
ARTIFACT_SCHEMA_VERSION = "phase6-publication-core-artifact-v1"
CLAIM_BOUNDARY = "publication_tables_and_figure3_figure4_only"
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/publication_core_v1.json"
FIXED_AUTHORITY_RUNS = {
    "phase4_final_figures": "results/phase4/final_figures_v1/phase4-final-figures-04ae2a217ad7d5fc1beafd4742dbc5bf3794079aca2dadfc3d4cc9b46411e7c6",
    "phase6_baseline_evidence": "results/phase6/baseline_evidence_v1/phase6-baseline-evidence-41e258ebcbcfe8244783112d29facd222b70e77cc4729a3742b432ea6ae29b4f",
    "phase6_denoising_evidence": "results/phase6/denoising_evidence_v1/phase6-denoising-evidence-991db5224bb01234d881d92cdf8a2483c486135605019b6c528bd982eac85701",
    "phase6_peak_evidence": "results/phase6/peak_evidence_v1/phase6-peak-evidence-5e8c7074799233232b3af6285b98ee75b11ea117476cf262f712738a901838ad",
    "phase2_background_fit": "results/phase2/background_fit/phase2-background-fit-6f8dc92c8b3edbefe2a422b805214b6687990c3ac13a34fbaed3faaa3fad6cdd",
    "phase3_dl_audit": "results/phase3/dl_reproducibility_audit_v1/phase3-dl-audit-b9bcfadabaf230832d129b85b295cf720fd01aaf2a47e786f5edb0e83b23cb3b",
}
REPORT_AUTHORITY_PATHS = {
    "phase0_source_audit_report": "reports/phase0/step01_data_source_audit.md",
    "phase05_internal_screening_decision_report": "reports/phase05/phase05_internal_screening_decision.md",
    "phase1_core10k_report": "reports/phase1/step16_core10k_build.md",
    "phase3_status_report": "reports/phase6/step01_publication_package_design.md",
    "phase4_closure_synthesis_report": "reports/phase4/step38_phase4_closure_synthesis.md",
    "phase6_step1_design_report": "reports/phase6/step01_publication_package_design.md",
}
CODE_RECEIPT_PATHS = {
    "authority_module": "rpe/runner/phase6_publication_core_authority.py",
    "builder_module": "rpe/runner/phase6_publication_core.py",
    "verifier_module": "rpe/runner/phase6_publication_core_verifier.py",
    "run_cli": "tools/run_phase6_publication_core.py",
    "freeze_tool": "tools/freeze_phase6_publication_core_config.py",
    "focused_test": "tests/test_phase6_publication_core.py",
}
SELECTED_PARENT_PAYLOAD_SPECS = {
    "phase4_final_figures": {
        "selected_files": (
            "parent_panel_index.csv",
            "figure2_alignment_data.csv",
            "figure2_protocol_interaction_data.csv",
        ),
        "terminal_name": "complete.json",
    },
    "phase6_baseline_evidence": {
        "selected_files": (
            "method_evidence_rows.csv",
            "system_status.jsonl",
        ),
        "terminal_name": "complete.json",
    },
    "phase6_denoising_evidence": {
        "selected_files": (
            "method_evidence_rows.csv",
            "system_status.jsonl",
        ),
        "terminal_name": "complete.json",
    },
    "phase6_peak_evidence": {
        "selected_files": (
            "method_evidence_rows.csv",
            "system_status.jsonl",
        ),
        "terminal_name": "complete.json",
    },
    "phase2_background_fit": {
        "selected_files": (
            "gate.json",
            "model_receipts.json",
        ),
        "terminal_name": "failed.json",
    },
    "phase3_dl_audit": {
        "selected_files": (
            "reproducibility_table.csv",
        ),
        "terminal_name": "complete.json",
    },
}

PAYLOAD_FILES = (
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

TABLE1_FIELDS = (
    "endpoint_id",
    "protocol_id",
    "protocol_label",
    "tier",
    "perturbation_scope",
    "cluster_kind",
    "cluster_count",
    "state",
    "mse_ag",
    "mse_ag_lower",
    "mse_ag_upper",
    "mse_acc_cross",
    "mse_acc_cross_lower",
    "mse_acc_cross_upper",
    "jointly_favorable_metric_ids",
    "ag_only_favorable_metric_ids",
    "acc_only_favorable_metric_ids",
    "holm_family_id",
    "holm_family_size",
    "claim_scope",
    "source_path",
    "source_sha256",
)

TABLE2_FIELDS = (
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
)

TABLE_S3_FIELDS = (
    "task_line",
    "system_id",
    "family_id",
    "planned_K",
    "runnable_K",
    "coverage_promoted_K",
    "coverage_state",
    "direct_gt_state",
    "downstream_state",
    "reference_free_state",
    "half_split_state",
    "phase5_eligible_K",
    "phase5_power_state",
    "publication_disposition",
    "reason_code",
    "source_path",
    "source_sha256",
)

FIGURE3_FIELDS = (
    "extractor_id",
    "excitation_stratum",
    "input_count",
    "valid_count",
    "invalid_count",
    "median_reconstruction_nrmse",
    "p95_reconstruction_nrmse",
    "p95_threshold",
    "valid_fraction_state",
    "valid_count_state",
    "median_state",
    "p95_state",
    "gate_state",
    "source_receipt_sha256",
)

FIGURE4_NODE_FIELDS = (
    "node_id",
    "label",
    "state",
    "shape",
    "color",
    "authority_path",
    "authority_sha256",
    "x",
    "y",
)

FIGURE4_EDGE_FIELDS = (
    "edge_id",
    "source_node_id",
    "target_node_id",
    "edge_style",
    "label",
    "authority_path",
    "authority_sha256",
)

STEP7_DEFERRED_REPORT_STATE = "deferred_by_owner"
STEP7_DEFERRED_REPORT_REQUIRED_TOKENS = {
    "state": "state: deferred_by_owner",
    "outcome_execution": "outcome_execution: not_admitted",
    "scientific_claims": "scientific_claims: not_admitted",
}


def canonical_json_bytes(value: object) -> bytes:
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


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> Mapping[str, object]:
    if path.is_dir():
        ledger = path / "SHA256SUMS"
        return {
            "path": str(path.relative_to(ROOT)),
            "byte_count": ledger.stat().st_size,
            "sha256": sha256_file(ledger),
        }
    return {
        "path": str(path.relative_to(ROOT)),
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _step7_dependency_error(error_type: type[Exception], message: str) -> None:
    raise error_type(message)


def validate_step7_deferred_report(
    path: Path,
    *,
    error_type: type[Exception] = ValueError,
) -> Mapping[str, object]:
    if not path.is_file():
        _step7_dependency_error(error_type, "config.step7_dependency.deferred_report.path: not a file")
    text = path.read_text(encoding="utf-8")
    front_matter = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
    if front_matter is None:
        _step7_dependency_error(error_type, "config.step7_dependency.deferred_report: missing YAML front matter")
    front_matter_text = front_matter.group(1)
    for label, token in STEP7_DEFERRED_REPORT_REQUIRED_TOKENS.items():
        if token not in front_matter_text and token not in text:
            _step7_dependency_error(
                error_type,
                f"config.step7_dependency.deferred_report: missing required token {label}",
            )
    return {
        "mode": "deferred_report",
        "state": STEP7_DEFERRED_REPORT_STATE,
        "deferred_report": file_identity(path),
    }


def csv_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".17g")
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def csv_bytes(rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=tuple(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: csv_cell(row.get(field, "")) for field in fieldnames})
    return buffer.getvalue().encode("utf-8")


def stable_run_id(*, config_sha256: str, payload_projection: Mapping[str, object]) -> str:
    return "phase6-publication-core-" + sha256_hex(
        canonical_json_bytes(
            {
                "config_sha256": config_sha256,
                "payload_projection": payload_projection,
            }
        )
    )


def write_sha256sums(payloads: Mapping[str, bytes], terminal_name: str, terminal_bytes: bytes) -> bytes:
    rows = [f"{sha256_hex(payloads[name])}  {name}" for name in PAYLOAD_FILES]
    rows.append(f"{sha256_hex(terminal_bytes)}  {terminal_name}")
    return ("\n".join(rows) + "\n").encode("utf-8")


class RasterCanvas:
    def __init__(self, width: int, height: int, background: tuple[int, int, int]) -> None:
        self.width = int(width)
        self.height = int(height)
        self._row_stride = self.width * 3
        self._pixels = bytearray(bytes(background) * (self.width * self.height))

    def _index(self, x: int, y: int) -> int:
        return (y * self.width + x) * 3

    def set_pixel(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            return
        index = self._index(x, y)
        self._pixels[index : index + 3] = bytes(color)

    def fill_rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        left = max(0, min(int(x0), int(x1)))
        right = min(self.width, max(int(x0), int(x1)))
        top = max(0, min(int(y0), int(y1)))
        bottom = min(self.height, max(int(y0), int(y1)))
        if left >= right or top >= bottom:
            return
        row = bytes(color) * (right - left)
        row_len = len(row)
        for y in range(top, bottom):
            start = self._index(left, y)
            self._pixels[start : start + row_len] = row

    def stroke_rect(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        *,
        thickness: int = 1,
    ) -> None:
        for step in range(max(1, int(thickness))):
            self.fill_rect(x0 + step, y0 + step, x1 - step, y0 + step + 1, color)
            self.fill_rect(x0 + step, y1 - step - 1, x1 - step, y1 - step, color)
            self.fill_rect(x0 + step, y0 + step, x0 + step + 1, y1 - step, color)
            self.fill_rect(x1 - step - 1, y0 + step, x1 - step, y1 - step, color)

    def draw_line(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        color: tuple[int, int, int],
        *,
        thickness: int = 1,
        dash: tuple[int, int] | None = None,
    ) -> None:
        x0_i = int(round(x0))
        y0_i = int(round(y0))
        x1_i = int(round(x1))
        y1_i = int(round(y1))
        dx = abs(x1_i - x0_i)
        sx = 1 if x0_i < x1_i else -1
        dy = -abs(y1_i - y0_i)
        sy = 1 if y0_i < y1_i else -1
        err = dx + dy
        total = max(abs(x1_i - x0_i), abs(y1_i - y0_i), 1)
        step = 0
        radius = max(0, int(thickness) // 2)
        dash_on = dash[0] if dash else 0
        dash_off = dash[1] if dash else 0
        dash_cycle = dash_on + dash_off
        while True:
            paint = True
            if dash and dash_cycle > 0:
                paint = (step % dash_cycle) < dash_on
            if paint:
                for ox in range(-radius, radius + 1):
                    for oy in range(-radius, radius + 1):
                        self.set_pixel(x0_i + ox, y0_i + oy, color)
            if x0_i == x1_i and y0_i == y1_i:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0_i += sx
            if e2 <= dx:
                err += dx
                y0_i += sy
            step += 1
            if step > total + 2:
                break

    def to_png_bytes(self, *, label: str) -> bytes:
        signature = b"\x89PNG\r\n\x1a\n"

        def chunk(name: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + name
                + data
                + struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
            )

        raw = bytearray()
        for y in range(self.height):
            raw.append(0)
            start = y * self._row_stride
            raw.extend(self._pixels[start : start + self._row_stride])
        text = chunk(b"tEXt", f"Title\x00{label}".encode("latin-1"))
        ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0))
        idat = chunk(b"IDAT", zlib.compress(bytes(raw), level=9))
        iend = chunk(b"IEND", b"")
        return signature + ihdr + text + idat + iend


def png_bytes(width: int, height: int, rgb: tuple[int, int, int], *, label: str) -> bytes:
    canvas = RasterCanvas(width, height, rgb)
    return canvas.to_png_bytes(label=label)


def svg_bytes(width: int, height: int, *, title: str, body_lines: Sequence[str]) -> bytes:
    body = "\n".join(body_lines)
    text = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f"<title>{title}</title>\n"
        f"{body}\n"
        "</svg>\n"
    )
    return text.encode("utf-8")


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


def render_figure3_png(
    *,
    width: int,
    height: int,
    rows: Sequence[Mapping[str, object]],
    extractor_order: Sequence[str],
    excitation_order: Sequence[str],
    p95_threshold: float,
) -> bytes:
    canvas = RasterCanvas(width, height, (247, 247, 244))
    canvas.fill_rect(0, 0, width, 180, (228, 234, 240))
    canvas.fill_rect(0, height - 180, width, height, (244, 246, 248))
    ordered_extractors, ordered_excitations = _figure3_orders(rows, extractor_order, excitation_order)
    by_key = {
        (str(row["extractor_id"]), str(row["excitation_stratum"])): row
        for row in rows
    }
    left = 340
    top = 320
    cell_width = max(220, (width - 520) // max(1, len(ordered_excitations)))
    cell_height = max(220, (height - 720) // max(1, len(ordered_extractors)))
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


def render_figure3_svg(
    *,
    width: int,
    height: int,
    rows: Sequence[Mapping[str, object]],
    extractor_order: Sequence[str],
    excitation_order: Sequence[str],
    p95_threshold: float,
) -> bytes:
    ordered_extractors, ordered_excitations = _figure3_orders(rows, extractor_order, excitation_order)
    by_key = {
        (str(row["extractor_id"]), str(row["excitation_stratum"])): row
        for row in rows
    }
    left = 340
    top = 320
    cell_width = max(220, (width - 520) // max(1, len(ordered_excitations)))
    cell_height = max(220, (height - 720) // max(1, len(ordered_extractors)))
    lines = [
        '<rect x="0" y="0" width="100%" height="100%" fill="#f7f7f4"/>',
        f'<text x="80" y="96" font-size="40" font-weight="700">Figure 3 Model Adequacy Matrix</text>',
        f'<text x="80" y="146" font-size="22">3x4 matrix with Median, P95, and PASS or FAIL gate state derived from authoritative model receipts.</text>',
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
            lines.append('</g>')
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


def render_figure4_png(
    *,
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


def render_figure4_svg(
    *,
    width: int,
    height: int,
    node_rows: Sequence[Mapping[str, object]],
    edge_rows: Sequence[Mapping[str, object]],
) -> bytes:
    lines = [
        '<defs><marker id="arrowhead" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto">'
        '<path d="M0,0 L12,6 L0,12 z" fill="#475569"/></marker></defs>',
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


def markdown_table(rows: Sequence[Mapping[str, object]], headers: Sequence[str]) -> bytes:
    ordered_headers = tuple(headers)
    parts = [
        "| " + " | ".join(ordered_headers) + " |",
        "| " + " | ".join("---" for _ in ordered_headers) + " |",
    ]
    for row in rows:
        parts.append("| " + " | ".join(csv_cell(row.get(header, "")) for header in ordered_headers) + " |")
    return ("\n".join(parts) + "\n").encode("utf-8")


def environment_receipt() -> Mapping[str, object]:
    return {
        "python": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
    }

