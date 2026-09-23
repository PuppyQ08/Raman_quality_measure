from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-d4-protocol-b-full-domain-config-v1"
EXPERIMENT_ID = "phase4-d4-protocol-b-full-domain-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d4-protocol-b-full-domain-artifact-v1"
MARKER_SCHEMA_VERSION = "phase4-d4-protocol-b-full-domain-marker-v1"
RUN_PREFIX = "phase4-d4-protocol-b-full-domain-"
CLAIM_BOUNDARY = "secondary_phase4_endpoint_local_execution_artifact_redistribution_not_cleared"
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
MODEL_FOLDS = (0, 1, 2, 3, 4)
N_COMPONENTS_GRID = (2, 4, 8, 16, 32)
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
METRIC_DIRECTIONS = {
    "mse": "lower_is_better",
    "rmse": "lower_is_better",
    "mae": "lower_is_better",
    "sam": "lower_is_better",
    "pearson_r": "higher_is_better",
    "nmse": "lower_is_better",
    "wasserstein_1_cm1": "lower_is_better",
    "is_like_structure_to_noise": "higher_is_better",
    "precision": "higher_is_better",
    "recall": "higher_is_better",
    "f1": "higher_is_better",
    "artifact_peak_ratio": "lower_is_better",
    "missing_peak_ratio": "lower_is_better",
}
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "alpha0_equivalence.json",
    "model_cells.jsonl",
    "validation_scores.jsonl",
    "predictions.jsonl",
    "blank_predictions.jsonl",
    "well_conditions.jsonl",
    "technical_lod_loq.jsonl",
    "condition_summary.csv",
    "well_observations.jsonl",
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "figure1_d4_protocol_b_full_domain.png",
    "figure1_d4_protocol_b_full_domain.svg",
    "figure1_d4_protocol_b_full_domain_data.csv",
    "figure2_d4_protocol_b_full_domain.png",
    "figure2_d4_protocol_b_full_domain.svg",
    "figure2_d4_protocol_b_full_domain_data.csv",
    "d4_protocol_b_full_domain_secondary_table.csv",
    "manifest.json",
)
TERMINAL_MARKERS = ("complete.json", "failed.json")
CODE_RELATIVE_PATHS = (
    "rpe/runner/phase4_d4_protocol_b.py",
    "rpe/runner/phase4_d4_protocol_b_verifier.py",
    "tests/test_phase4_d4_protocol_b.py",
    "tools/run_phase4_d4_protocol_b.py",
)
DEFAULT_CONFIG = ROOT / "experiments/phase4/configs/d4_protocol_b_full_domain_v1.json"
REAL_ALPHA0_MODEL_DIGEST = "dcebe39171c6855570b0b3c52e5039e76de5cd12ccabba4bac123cd217e9afca"
REAL_ALPHA0_VALIDATION_DIGEST = "c94b1a28464e5ff36e14d40ac2443544a4d89f7095405146e710cb9a5b6eb215"
REAL_ALPHA0_PREDICTION_DIGEST = "2479d66588c660f14eb18b47987bbda74238c2efeb7c6d34151cfb60b4524ce4"
REAL_ALPHA0_BLANK_PREDICTION_DIGEST = "ac81670f74a8daac9e46ab2bd541c97b269e4413e70f261cb77792f146f68a03"
REAL_ALPHA0_TECHNICAL_LOD_LOQ_DIGEST = "d7f14517e8acd944577423c48294b93f0c828bccf60b6b6097882ddb826dd83a"
REAL_MEASUREMENT_BRIDGE_SHA256 = "1e0dac3d3ab54ca1e0faab1b55f66be85c3c51a181661ed0570366ac13a744c9"

# Frozen after the config bytes were finalized.
CONFIG_BYTES = 18809
CONFIG_SHA256 = "bf99332e23e82ecc95123701a74f1c335a2607eb554c5469a72df49f95039a35"

_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO3ZP1cAAAAASUVORK5CYII="
)


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("csv requires at least one row")
    fields = tuple(rows[0].keys())
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def alpha_hex(alpha: float) -> str:
    return struct.pack("<d", float(alpha)).hex()


def condition_ids(
    perturbation_ids: Sequence[str] = PERTURBATIONS,
    alpha_grid: Sequence[float] = ALPHAS,
) -> tuple[str, ...]:
    positives = tuple(
        f"{perturbation}:{alpha_hex(alpha)}"
        for perturbation in perturbation_ids
        for alpha in alpha_grid[1:]
    )
    return ("alpha0",) + positives


def metric_preferred_direction(metric_output_id: str) -> str:
    return METRIC_DIRECTIONS[metric_output_id]


def stable_run_id(
    *,
    config_sha256: str,
    condition_ids_value: Sequence[str],
    record_ids: Sequence[str],
    blank_record_ids: Sequence[str],
    endpoint_state: str,
) -> str:
    # The run identity is frozen before any outcome is observed.  Keep the
    # parameter for the already-published helper signature, but deliberately
    # exclude its value from the identity document.
    del endpoint_state
    document = {
        "blank_record_ids": list(blank_record_ids),
        "condition_ids": list(condition_ids_value),
        "config_sha256": config_sha256,
        "record_ids": list(record_ids),
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


def render_d4_protocol_b_figures(
    *,
    figure1_rows: Sequence[Mapping[str, object]],
    figure2_rows: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    figure1_lines = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='200'>",
        "<text x='20' y='24'>figure1_d4_protocol_b_full_domain</text>",
        f"<text x='20' y='48'>rows={len(figure1_rows)}</text>",
        "</svg>",
    ]
    figure2_lines = [
        "<svg xmlns='http://www.w3.org/2000/svg' width='900' height='600'>",
        "<text x='20' y='24'>figure2_d4_protocol_b_full_domain</text>",
    ]
    for index, metric_output_id in enumerate(METRIC_OUTPUT_IDS, start=1):
        y = 24 + index * 24
        figure2_lines.append(f"<text x='20' y='{y}'>{metric_output_id}</text>")
    figure2_lines.append(f"<text x='20' y='380'>rows={len(figure2_rows)}</text>")
    figure2_lines.append("</svg>")
    return {
        "figure1_d4_protocol_b_full_domain.png": _TINY_PNG,
        "figure1_d4_protocol_b_full_domain.svg": ("\n".join(figure1_lines) + "\n").encode("utf-8"),
        "figure2_d4_protocol_b_full_domain.png": _TINY_PNG,
        "figure2_d4_protocol_b_full_domain.svg": ("\n".join(figure2_lines) + "\n").encode("utf-8"),
    }


__all__ = [
    "ALPHAS",
    "ARTIFACT_PAYLOAD_FILES",
    "ARTIFACT_SCHEMA_VERSION",
    "CLAIM_BOUNDARY",
    "CODE_RELATIVE_PATHS",
    "CONFIG_BYTES",
    "CONFIG_SHA256",
    "DEFAULT_CONFIG",
    "EXPERIMENT_ID",
    "MARKER_SCHEMA_VERSION",
    "METRIC_DIRECTIONS",
    "METRIC_OUTPUT_IDS",
    "MODEL_FOLDS",
    "N_COMPONENTS_GRID",
    "PERTURBATIONS",
    "REAL_ALPHA0_BLANK_PREDICTION_DIGEST",
    "REAL_ALPHA0_MODEL_DIGEST",
    "REAL_ALPHA0_PREDICTION_DIGEST",
    "REAL_ALPHA0_TECHNICAL_LOD_LOQ_DIGEST",
    "REAL_ALPHA0_VALIDATION_DIGEST",
    "REAL_MEASUREMENT_BRIDGE_SHA256",
    "ROOT",
    "RUN_PREFIX",
    "SCHEMA_VERSION",
    "TERMINAL_MARKERS",
    "alpha_hex",
    "canonical_json_bytes",
    "condition_ids",
    "csv_bytes",
    "jsonl_bytes",
    "metric_preferred_direction",
    "render_d4_protocol_b_figures",
    "sha256_file",
    "sha256_hex",
    "stable_run_id",
    "write_sha256sums",
]
