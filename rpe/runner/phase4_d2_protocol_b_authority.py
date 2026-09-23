from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-d2-protocol-b-full-domain-config-v1"
EXPERIMENT_ID = "phase4-d2-protocol-b-full-domain-v1"
CLAIM_BOUNDARY = "local_execution_artifact_redistribution_not_cleared"
ARTIFACT_SCHEMA_VERSION = "phase4-d2-protocol-b-full-domain-artifact-v1"
SHOTS = (5, 10, 20)
PERTURBATIONS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
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
    "figure1_d2_5shot_protocol_b_full_domain.png",
    "figure1_d2_5shot_protocol_b_full_domain.svg",
    "figure1_d2_5shot_protocol_b_full_domain_data.csv",
    "figure2_d2_5shot_protocol_b_full_domain.png",
    "figure2_d2_5shot_protocol_b_full_domain.svg",
    "figure2_d2_5shot_protocol_b_full_domain_data.csv",
    "figure1_d2_10shot_protocol_b_full_domain.png",
    "figure1_d2_10shot_protocol_b_full_domain.svg",
    "figure1_d2_10shot_protocol_b_full_domain_data.csv",
    "figure2_d2_10shot_protocol_b_full_domain.png",
    "figure2_d2_10shot_protocol_b_full_domain.svg",
    "figure2_d2_10shot_protocol_b_full_domain_data.csv",
    "figure1_d2_20shot_protocol_b_full_domain.png",
    "figure1_d2_20shot_protocol_b_full_domain.svg",
    "figure1_d2_20shot_protocol_b_full_domain_data.csv",
    "figure2_d2_20shot_protocol_b_full_domain.png",
    "figure2_d2_20shot_protocol_b_full_domain.svg",
    "figure2_d2_20shot_protocol_b_full_domain_data.csv",
    "d2_protocol_b_full_domain_secondary_table.csv",
    "manifest.json",
)
TERMINAL_MARKERS = ("complete.json", "failed.json")
REAL_CONDITION_BRIDGE_SHA256 = (
    "031b4f4f30235a6237ad1a6f89b03fb9f2d1b724f55f7c8ae9d52f402eabaa7c"
)
CODE_RELATIVE_PATHS = (
    "rpe/alignment/__init__.py",
    "rpe/alignment/bulk.py",
    "rpe/alignment/contracts.py",
    "rpe/alignment/core.py",
    "rpe/alignment/inference.py",
    "rpe/downstream/bacteria_id.py",
    "rpe/evaluation/contracts.py",
    "rpe/perturb/__init__.py",
    "rpe/perturb/axis_transform.py",
    "rpe/perturb/contracts.py",
    "rpe/perturb/correlated_noise.py",
    "rpe/perturb/gaussian_noise.py",
    "rpe/perturb/sweep.py",
    "rpe/runner/d2_selection.py",
    "rpe/runner/phase1_config.py",
    "rpe/runner/phase1_gates.py",
    "rpe/runner/phase1_perturbations.py",
    "rpe/runner/phase1_selection.py",
    "rpe/runner/phase1_types.py",
    "rpe/runner/phase4_d2_protocol_b.py",
    "rpe/runner/phase4_d2_protocol_b_verifier.py",
    "tools/run_phase4_d2_protocol_b.py",
)

# Updated after the config file is written.
CONFIG_BYTES = 24183
CONFIG_SHA256 = "f548fd588345949916b72b826ff4da001c9f4124fc4ed76ba03de76555996d00"


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


def write_sha256sums(payloads: Mapping[str, bytes], terminal_name: str, terminal_bytes: bytes) -> bytes:
    rows = []
    for name, raw in payloads.items():
        rows.append(f"{sha256_hex(raw)}  {name}")
    rows.append(f"{sha256_hex(terminal_bytes)}  {terminal_name}")
    return ("\n".join(rows) + "\n").encode("utf-8")
