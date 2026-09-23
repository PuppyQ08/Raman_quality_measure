from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-d1-protocol-b-full-domain-config-v1"
EXPERIMENT_ID = "phase4-d1-protocol-b-full-domain-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d1-protocol-b-full-domain-artifact-v1"
CLAIM_BOUNDARY = "local_execution_artifact_redistribution_not_cleared"
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
MODEL_SEEDS = (0, 1, 2, 3, 4)
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
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "alpha0_equivalence.json",
    "model_cells.jsonl",
    "validation_scores.jsonl",
    "predictions.jsonl",
    "seed_class_conditions.jsonl",
    "condition_summary.csv",
    "class_observations.jsonl",
    "alignment_results.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "figure1_d1_protocol_b_full_domain.png",
    "figure1_d1_protocol_b_full_domain.svg",
    "figure1_d1_protocol_b_full_domain_data.csv",
    "figure2_d1_protocol_b_full_domain.png",
    "figure2_d1_protocol_b_full_domain.svg",
    "figure2_d1_protocol_b_full_domain_data.csv",
    "d1_protocol_b_full_domain_secondary_table.csv",
    "manifest.json",
)
TERMINAL_MARKERS = ("complete.json", "failed.json")
CODE_RELATIVE_PATHS = (
    "rpe/runner/phase4_d1_protocol_b.py",
    "rpe/runner/phase4_d1_protocol_b_verifier.py",
    "tools/run_phase4_d1_protocol_b.py",
    "rpe/runner/__init__.py",
)
REAL_ALPHA0_MODEL_DIGEST = "8426f7873f53e22e78739f3d9f402c0a47b92751d85bb5539b9d4f6688523ab4"
REAL_ALPHA0_VALIDATION_DIGEST = "008db8a933f82a2a3af61107b14a2adab1d817bd04acee5c67b3fa63d73d4c09"
REAL_ALPHA0_PREDICTION_DIGEST = "9240974c53b568b041c08bfd4ad8f7830c9dc6f1ecb7e29a2fd06c91d9a4617b"

# Updated after the frozen config is written.
CONFIG_BYTES = 6492
CONFIG_SHA256 = "086203bf6f1bf52518e7fcac1feb5c6b488e338c2617013f4182c4918bdbb04a"


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


def jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json_bytes(row) for row in rows)


def csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        raise ValueError("csv requires at least one row")
    stream = io.StringIO(newline="")
    fields = tuple(rows[0].keys())
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def alpha_hex(alpha: float) -> str:
    return np.float64(alpha).tobytes().hex()


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


def write_sha256sums(
    payloads: Mapping[str, bytes],
    terminal_name: str,
    terminal_bytes: bytes,
) -> bytes:
    rows = [f"{sha256_hex(raw)}  {name}" for name, raw in payloads.items()]
    rows.append(f"{sha256_hex(terminal_bytes)}  {terminal_name}")
    return ("\n".join(rows) + "\n").encode("utf-8")


def render_protocol_b_figures(
    metric_output_ids: Sequence[str],
    figure1_rows: Sequence[Mapping[str, object]],
    figure2_rows: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    payloads: dict[str, bytes] = {}
    matplotlib.rcParams["svg.hashsalt"] = "rpe-phase4-d1-protocol-b-v1"

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    metric_names = list(metric_output_ids)
    metric_means = []
    for metric in metric_names:
        rows = [row for row in figure1_rows if str(row["metric_output_id"]) == metric]
        values = [float(row["harm"]) for row in rows]
        metric_means.append(float(np.mean(values)) if values else 0.0)
    axes[0].barh(np.arange(len(metric_names)), metric_means, color="#1f77b4")
    axes[0].set_yticks(np.arange(len(metric_names)), metric_names)
    axes[0].set_title("Mean Harm")

    condition_names = [str(row.get("metric_output_id", "")) for row in figure2_rows]
    ag_values = [0.0 if row.get("ag") is None else float(row["ag"]) for row in figure2_rows]
    axes[1].barh(np.arange(len(condition_names)), ag_values, color="#2ca02c")
    axes[1].set_yticks(np.arange(len(condition_names)), condition_names)
    axes[1].set_title("Alignment Gap")
    figure.tight_layout()
    for kind in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
        payloads[f"figure1_d1_protocol_b_full_domain.{kind}"] = stream.getvalue()
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
    for axis, field, color in zip(
        axes,
        ("ag", "acc_cross", "d_ag", "d_acc"),
        ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"),
        strict=True,
    ):
        values = [0.0 if row.get(field) is None else float(row[field]) for row in figure2_rows]
        axis.barh(np.arange(len(figure2_rows)), values, color=color)
        axis.set_title(field)
        axis.set_yticks(np.arange(len(figure2_rows)), condition_names if axis is axes[0] else [])
    figure.tight_layout()
    for kind in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
        payloads[f"figure2_d1_protocol_b_full_domain.{kind}"] = stream.getvalue()
    plt.close(figure)
    return payloads
