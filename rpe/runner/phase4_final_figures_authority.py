from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-final-figures-config-v1"
EXPERIMENT_ID = "phase4-final-figures-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-final-figures-artifact-v1"
MARKER_SCHEMA_VERSION = "phase4-final-figures-marker-v1"
RUN_PREFIX = "phase4-final-figures-"
CLAIM_BOUNDARY = "post_outcome_final_figure_secondary_phase4_no_cross_endpoint_claim"

SCOPE_IDS = (
    "p01_p05_not_evaluable_coverage",
    "p06_p07_structurally_ineligible_missing_explicit_baseline",
    "d3_inactive",
    "d1_protocol_b_failed_alpha0_equivalence",
    "d5_protocol_a_peak_common_primary_not_evaluable_coverage",
    "phase4_confirmatory_success_not_evaluable",
)
ENDPOINT_ORDER = ("d5", "d2_5", "d2_10", "d2_20", "d1", "d4")
PAIRED_CELL_ORDER = ("d5", "d2_5", "d2_10", "d2_20", "d4")
PROTOCOL_ORDER = ("a", "b")
PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHA_GRID = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80)
METRIC_OUTPUT_IDS = (
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

FIGURE1_SIZE = (5400, 3300)
FIGURE2_SIZE = (5400, 3900)
FIGURE_TEXT = (
    "full_domain_core: P8-P12 only",
    "matched reference",
    "closed: failed exact alpha-zero equivalence; no positive-alpha outcome",
    "Task-native Phase-4 MSE/downstream response.",
    "Task-native metric alignment and secondary protocol interaction.",
)
PROTOCOL_EFFECT_CLASSIFICATIONS = (
    "attenuation_by_b_rejected",
    "amplification_by_b_rejected",
    "not_rejected",
)
WITHIN_PROTOCOL_GLYPHS = (
    "favorable_rejected",
    "adverse_rejected",
    "not_rejected",
)

STYLE = {
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
}

ARTIFACT_PAYLOAD_FILES = (
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
TERMINAL_MARKERS = ("complete.json", "failed.json")
DEFAULT_CONFIG = ROOT / "experiments/phase4/configs/final_figures_v1.json"

SCOPE_STATUS_FIELDS = ("scope_id", "state")
PARENT_PANEL_FIELDS = (
    "panel_id",
    "endpoint_id",
    "protocol_id",
    "state",
    "numerical_source_allowed",
    "parent_key",
    "relative_path",
    "ledger_sha256",
    "terminal_name",
    "terminal_sha256",
    "figure1_path",
    "figure1_sha256",
    "figure2_path",
    "figure2_sha256",
    "holm_path",
    "holm_sha256",
)
FIGURE1_RESPONSE_FIELDS = (
    "cell_id",
    "protocol_id",
    "perturbation_id",
    "alpha",
    "metric_output_id",
    "protocol_a_mse_x",
    "metric_x",
    "x_render_value",
    "downstream_harm",
)
FIGURE1_PROTOCOL_EFFECT_FIELDS = (
    "cell_id",
    "perturbation_id",
    "alpha",
    "summary_type",
    "g_harm",
    "g_harm_lower",
    "g_harm_upper",
    "raw_p_value",
    "adjusted_p_value",
    "rank",
    "family_size",
    "family_id",
    "favorable",
    "rejected",
    "direction",
    "state",
    "classification",
)
FIGURE2_ALIGNMENT_FIELDS = (
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
)
FIGURE2_PROTOCOL_INTERACTION_FIELDS = (
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
)

# The real config is intentionally still pending; these are frozen only once
# the real-parent reconstruction path is implemented.
CONFIG_BYTES = 24965
CONFIG_SHA256 = "bd5aa825244c213ca01d2f521e0b7661992d60d323ab6a0e0321858c581c9a3e"


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def csv_bytes(
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def sha256_hex(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def alpha_hex(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


def stable_run_id(
    *,
    config_sha256: str,
    payload_projection: Mapping[str, object],
) -> str:
    document = {
        "config_sha256": config_sha256,
        "payload_projection": payload_projection,
    }
    return RUN_PREFIX + sha256_hex(canonical_json_bytes(document))


def write_sha256sums(
    payloads: Mapping[str, bytes],
    terminal_name: str,
    terminal_bytes: bytes,
) -> bytes:
    rows = [f"{sha256_hex(payloads[name])}  {name}" for name in ARTIFACT_PAYLOAD_FILES]
    rows.append(f"{sha256_hex(terminal_bytes)}  {terminal_name}")
    return ("\n".join(rows) + "\n").encode("utf-8")


__all__ = [
    "ALPHA_GRID",
    "ARTIFACT_PAYLOAD_FILES",
    "ARTIFACT_SCHEMA_VERSION",
    "CLAIM_BOUNDARY",
    "CONFIG_BYTES",
    "CONFIG_SHA256",
    "DEFAULT_CONFIG",
    "ENDPOINT_ORDER",
    "EXPERIMENT_ID",
    "FIGURE_TEXT",
    "FIGURE1_PROTOCOL_EFFECT_FIELDS",
    "FIGURE1_RESPONSE_FIELDS",
    "FIGURE1_SIZE",
    "FIGURE2_ALIGNMENT_FIELDS",
    "FIGURE2_PROTOCOL_INTERACTION_FIELDS",
    "FIGURE2_SIZE",
    "MARKER_SCHEMA_VERSION",
    "METRIC_OUTPUT_IDS",
    "PAIRED_CELL_ORDER",
    "PARENT_PANEL_FIELDS",
    "PERTURBATION_IDS",
    "PROTOCOL_EFFECT_CLASSIFICATIONS",
    "PROTOCOL_ORDER",
    "ROOT",
    "RUN_PREFIX",
    "SCHEMA_VERSION",
    "SCOPE_IDS",
    "SCOPE_STATUS_FIELDS",
    "STYLE",
    "TERMINAL_MARKERS",
    "WITHIN_PROTOCOL_GLYPHS",
    "alpha_hex",
    "canonical_json_bytes",
    "csv_bytes",
    "sha256_file",
    "sha256_hex",
    "stable_run_id",
    "write_sha256sums",
]
