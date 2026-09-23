from __future__ import annotations

import csv
import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase4-protocol-ab-contrast-config-v1"
EXPERIMENT_ID = "phase4-protocol-ab-contrast-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-protocol-ab-contrast-artifact-v1"
MARKER_SCHEMA_VERSION = "phase4-protocol-ab-contrast-marker-v1"
RUN_PREFIX = "phase4-protocol-ab-contrast-"
CLAIM_BOUNDARY = "post_outcome_pre_contrast_secondary_phase4_no_cross_endpoint_claim"
CELL_IDS = (
    "d5_full_domain_core",
    "d2_5shot",
    "d2_10shot",
    "d2_20shot",
    "d1_full_domain_core",
    "d4_full_domain_core",
)
EVALUABLE_CELL_IDS = tuple(value for value in CELL_IDS if value != "d1_full_domain_core")
PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
METRIC_OUTPUT_IDS = (
    "mse", "rmse", "mae", "sam", "pearson_r", "nmse",
    "wasserstein_1_cm1", "is_like_structure_to_noise", "precision",
    "recall", "f1", "artifact_peak_ratio", "missing_peak_ratio",
)
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "authority_bridge.json",
    "preflight.json",
    "endpoint_status.jsonl",
    "downstream_protocol_effects.jsonl",
    "metric_protocol_effects.jsonl",
    "bootstrap_results.jsonl",
    "sign_flip_results.jsonl",
    "holm_family.jsonl",
    "protocol_downstream_contrast_table.csv",
    "protocol_metric_interaction_table.csv",
    "manifest.json",
)
TERMINAL_MARKERS = ("complete.json", "failed.json")
CODE_RELATIVE_PATHS = (
    "rpe/runner/phase4_protocol_ab_contrast.py",
    "rpe/runner/phase4_protocol_ab_contrast_verifier.py",
    "tests/test_phase4_protocol_ab_contrast.py",
    "tools/run_phase4_protocol_ab_contrast.py",
)
DEFAULT_CONFIG = ROOT / "experiments/phase4/configs/protocol_ab_contrast_v1.json"

# Frozen only after all load-bearing code and the canonical real config are final.
CONFIG_BYTES = 22549
CONFIG_SHA256 = "ccabb4b0fc296a426b65d4321b35f763c97d90a18431f949ac050d2053a46937"

DOWNSTREAM_CSV_FIELDS = (
    "cell_id", "perturbation_id", "summary_type", "alpha", "mean_harm_a",
    "mean_harm_b", "gap", "interval_lower", "interval_upper", "cluster_count",
    "state",
)
METRIC_CSV_FIELDS = (
    "cell_id", "metric_output_id", "ag_a", "ag_b", "delta_ag",
    "delta_ag_lower", "delta_ag_upper", "acc_a", "acc_b", "delta_acc",
    "delta_acc_lower", "delta_acc_upper", "canonical_d_ag_a",
    "canonical_d_ag_b", "canonical_d_acc_a", "canonical_d_acc_b",
    "parent_ag_a", "parent_ag_b", "parent_acc_a", "parent_acc_b",
    "parent_d_ag_a", "parent_d_ag_b", "parent_d_acc_a",
    "parent_d_acc_b", "parent_a_exact", "parent_b_exact",
    "parent_ag_a_difference", "parent_ag_b_difference",
    "parent_acc_a_difference", "parent_acc_b_difference",
    "i_ag", "i_ag_lower", "i_ag_upper", "i_acc", "i_acc_lower",
    "i_acc_upper", "state",
)


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


def csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
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
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def alpha_hex(value: float) -> str:
    return struct.pack("<d", float(value)).hex()


def stable_run_id(
    *,
    config_sha256: str,
    parent_identity_projection: Mapping[str, object],
    code_authority: Mapping[str, object],
    environment_authority: Mapping[str, object],
) -> str:
    document = {
        "code_authority": code_authority,
        "config_sha256": config_sha256,
        "environment_authority": environment_authority,
        "parent_identity_projection": parent_identity_projection,
    }
    return RUN_PREFIX + sha256_hex(canonical_json_bytes(document))


def write_sha256sums(
    payloads: Mapping[str, bytes], terminal_name: str, terminal_bytes: bytes
) -> bytes:
    rows = [f"{sha256_hex(payloads[name])}  {name}" for name in ARTIFACT_PAYLOAD_FILES]
    rows.append(f"{sha256_hex(terminal_bytes)}  {terminal_name}")
    return ("\n".join(rows) + "\n").encode("utf-8")


__all__ = [
    "ALPHAS", "ARTIFACT_PAYLOAD_FILES", "ARTIFACT_SCHEMA_VERSION", "CELL_IDS",
    "CLAIM_BOUNDARY", "CODE_RELATIVE_PATHS", "CONFIG_BYTES", "CONFIG_SHA256",
    "DEFAULT_CONFIG", "DOWNSTREAM_CSV_FIELDS", "EVALUABLE_CELL_IDS",
    "EXPERIMENT_ID", "MARKER_SCHEMA_VERSION", "METRIC_CSV_FIELDS",
    "METRIC_OUTPUT_IDS", "PERTURBATION_IDS", "ROOT", "RUN_PREFIX",
    "SCHEMA_VERSION", "TERMINAL_MARKERS", "alpha_hex", "canonical_json_bytes",
    "csv_bytes", "jsonl_bytes", "sha256_file", "sha256_hex", "stable_run_id",
    "write_sha256sums",
]
