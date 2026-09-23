"""Independent verifier for the Phase-4 Protocol-A/B contrast artifact.

The verifier intentionally has no dependency on the production contrast
runner.  Both synthetic and retained-parent verification rebuild every
artifact byte from independently parsed inputs.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import platform
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
import scipy
from scipy.optimize import isotonic_regression
from sklearn import __version__ as sklearn_version
from threadpoolctl import __version__ as threadpoolctl_version

from rpe.alignment import (
    AlignmentObservation,
    alignment_gap,
    cross_perturbation_accuracy,
    holm_step_down,
    paired_contribution_sign_flip,
)


ROOT = Path(__file__).resolve().parents[2]
AUTHORITY_PATH = ROOT / "rpe/runner/phase4_protocol_ab_contrast_authority.py"
DEFAULT_CONFIG = ROOT / "experiments/phase4/configs/protocol_ab_contrast_v1.json"
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
PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALPHAS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
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
CODE_RELATIVE_PATHS = (
    "rpe/runner/phase4_protocol_ab_contrast.py",
    "rpe/runner/phase4_protocol_ab_contrast_verifier.py",
    "tests/test_phase4_protocol_ab_contrast.py",
    "tools/run_phase4_protocol_ab_contrast.py",
)
DOWNSTREAM_CSV_FIELDS = (
    "cell_id", "perturbation_id", "summary_type", "alpha",
    "mean_harm_a", "mean_harm_b", "gap", "interval_lower",
    "interval_upper", "cluster_count", "state",
)
METRIC_CSV_FIELDS = (
    "cell_id", "metric_output_id", "ag_a", "ag_b", "delta_ag",
    "delta_ag_lower", "delta_ag_upper", "acc_a", "acc_b",
    "delta_acc", "delta_acc_lower", "delta_acc_upper",
    "canonical_d_ag_a", "canonical_d_ag_b", "canonical_d_acc_a",
    "canonical_d_acc_b", "parent_ag_a", "parent_ag_b",
    "parent_acc_a", "parent_acc_b", "parent_d_ag_a",
    "parent_d_ag_b", "parent_d_acc_a", "parent_d_acc_b",
    "parent_a_exact", "parent_b_exact", "parent_ag_a_difference",
    "parent_ag_b_difference", "parent_acc_a_difference",
    "parent_acc_b_difference", "i_ag", "i_ag_lower",
    "i_ag_upper", "i_acc", "i_acc_lower", "i_acc_upper", "state",
)


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json(row) for row in rows)


def _csv(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=tuple(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    return stream.getvalue().encode("utf-8")


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _stable_run_id(
    *, config_sha256: str, parent_identity_projection: Mapping[str, object],
    code_authority: Mapping[str, object], environment_authority: Mapping[str, object],
) -> str:
    return RUN_PREFIX + _sha_bytes(
        _canonical_json(
            {
                "code_authority": code_authority,
                "config_sha256": config_sha256,
                "environment_authority": environment_authority,
                "parent_identity_projection": parent_identity_projection,
            }
        )
    )


def _checksum_ledger(
    payloads: Mapping[str, bytes], terminal_name: str, terminal: bytes
) -> bytes:
    lines = [f"{_sha_bytes(payloads[name])}  {name}" for name in ARTIFACT_PAYLOAD_FILES]
    lines.append(f"{_sha_bytes(terminal)}  {terminal_name}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _authority_literals() -> Mapping[str, object]:
    try:
        tree = ast.parse(AUTHORITY_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "cannot read authority literals"
        ) from error
    wanted = {
        "SCHEMA_VERSION", "EXPERIMENT_ID", "ARTIFACT_SCHEMA_VERSION",
        "MARKER_SCHEMA_VERSION", "RUN_PREFIX", "CLAIM_BOUNDARY",
        "CELL_IDS", "PERTURBATION_IDS", "ALPHAS",
        "METRIC_OUTPUT_IDS", "ARTIFACT_PAYLOAD_FILES",
        "CODE_RELATIVE_PATHS", "DOWNSTREAM_CSV_FIELDS",
        "METRIC_CSV_FIELDS", "CONFIG_BYTES", "CONFIG_SHA256",
    }
    values: dict[str, object] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id in wanted:
            try:
                values[target.id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
    if set(values) != wanted:
        raise Phase4ProtocolABContrastVerifierError(
            "authority literal schema mismatch"
        )
    local = {
        "SCHEMA_VERSION": SCHEMA_VERSION,
        "EXPERIMENT_ID": EXPERIMENT_ID,
        "ARTIFACT_SCHEMA_VERSION": ARTIFACT_SCHEMA_VERSION,
        "MARKER_SCHEMA_VERSION": MARKER_SCHEMA_VERSION,
        "RUN_PREFIX": RUN_PREFIX,
        "CLAIM_BOUNDARY": CLAIM_BOUNDARY,
        "CELL_IDS": CELL_IDS,
        "PERTURBATION_IDS": PERTURBATION_IDS,
        "ALPHAS": ALPHAS,
        "METRIC_OUTPUT_IDS": METRIC_OUTPUT_IDS,
        "ARTIFACT_PAYLOAD_FILES": ARTIFACT_PAYLOAD_FILES,
        "CODE_RELATIVE_PATHS": CODE_RELATIVE_PATHS,
        "DOWNSTREAM_CSV_FIELDS": DOWNSTREAM_CSV_FIELDS,
        "METRIC_CSV_FIELDS": METRIC_CSV_FIELDS,
    }
    if any(values[name] != value for name, value in local.items()):
        raise Phase4ProtocolABContrastVerifierError(
            "authority literals differ from verifier contract"
        )
    return MappingProxyType(values)


class Phase4ProtocolABContrastVerifierError(ValueError):
    """Raised when an artifact cannot be independently reproduced."""


@dataclass(frozen=True)
class _VerifierConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    payload_files: tuple[str, ...]
    cell_ids: tuple[str, ...]
    metric_output_ids: tuple[str, ...]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    synthetic_fixture: bool


@dataclass(frozen=True)
class Phase4ProtocolABContrastVerifierSummary:
    path: Path
    run_id: str
    status: str
    endpoint_status_count: int
    bootstrap_result_count: int
    holm_slot_count: int


@dataclass(frozen=True)
class _VerifierCell:
    cell_id: str
    cluster_kind: str
    cluster_ids: tuple[str, ...]
    metric_output_ids: tuple[str, ...]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_harm: np.ndarray
    downstream_a: np.ndarray
    downstream_b: np.ndarray
    metric_reconciliation: Mapping[str, object]
    parent_alignment_a: Mapping[str, Mapping[str, object]]
    parent_alignment_b: Mapping[str, Mapping[str, object]]
    parent_a_exact_required: bool
    parent_b_exact_required: bool


@dataclass(frozen=True)
class _VerifierInputs:
    cells: tuple[_VerifierCell, ...]
    endpoint_status_rows: tuple[Mapping[str, object], ...]
    authority_bridge: Mapping[str, object]
    preflight: Mapping[str, object]


def _require_mapping(
    document: Mapping[str, object], name: str
) -> Mapping[str, object]:
    value = document.get(name)
    if not isinstance(value, Mapping):
        raise Phase4ProtocolABContrastVerifierError(
            f"config section is missing or invalid: {name}"
        )
    return value


def _validate_real_config_sections(document: Mapping[str, object]) -> None:
    parent_names = {
        "d5_a", "d5_b", "d2_a", "d2_b",
        "d1_a", "d1_b", "d4_a", "d4_b",
    }
    parents = _require_mapping(document, "parent_artifacts")
    if set(parents) != parent_names:
        raise Phase4ProtocolABContrastVerifierError(
            "parent artifact identity set mismatch"
        )
    parent_fields = {
        "relative_path", "directory_name", "sha256sums_sha256",
        "required_inventory", "manifest_run_id", "terminal_name",
        "terminal_run_id", "status",
    }
    for name, receipt in parents.items():
        if not isinstance(receipt, Mapping) or not parent_fields <= set(receipt):
            raise Phase4ProtocolABContrastVerifierError(
                f"parent artifact schema mismatch: {name}"
            )
        expected_status = "failed" if name == "d1_b" else "complete"
        expected_terminal = "failed.json" if name == "d1_b" else "complete.json"
        if (
            receipt.get("status") != expected_status
            or receipt.get("terminal_name") != expected_terminal
        ):
            raise Phase4ProtocolABContrastVerifierError(
                f"parent terminal contract mismatch: {name}"
            )

    alpha0 = _require_mapping(document, "alpha0_admission")
    d5_summary = alpha0.get("d5_shared_summary")
    if (
        set(alpha0) != {
            "d1_sha256", "d2_sha256", "d4_sha256",
            "d5_shared_summary",
        }
        or
        not isinstance(alpha0.get("d2_sha256"), str)
        or not isinstance(alpha0.get("d4_sha256"), str)
        or not isinstance(alpha0.get("d1_sha256"), str)
        or not isinstance(d5_summary, Mapping)
        or set(d5_summary) != {"bytes", "sha256"}
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "alpha-zero admission schema mismatch"
        )

    evaluable = set(CELL_IDS) - {"d1_full_domain_core"}
    pairing = _require_mapping(document, "pairing_identities")
    reconciliation = _require_mapping(document, "metric_reconciliation")
    parent_alignment = _require_mapping(document, "parent_alignment")
    observations = _require_mapping(document, "observation_payloads")
    if set(pairing) != evaluable or set(reconciliation) != evaluable:
        raise Phase4ProtocolABContrastVerifierError(
            "per-cell pairing or reconciliation schema mismatch"
        )
    if set(parent_alignment) != evaluable:
        raise Phase4ProtocolABContrastVerifierError(
            "parent alignment schema mismatch"
        )
    if set(observations) != {"d5_a", "d5_b", "d2_a", "d2_b", "d4_a", "d4_b"}:
        raise Phase4ProtocolABContrastVerifierError(
            "observation payload schema mismatch"
        )
    for name, receipt in observations.items():
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"bytes", "sha256"}
            or isinstance(receipt.get("bytes"), bool)
            or int(receipt.get("bytes", -1)) <= 0
            or not isinstance(receipt.get("sha256"), str)
        ):
            raise Phase4ProtocolABContrastVerifierError(
                f"observation payload schema mismatch: {name}"
            )
    expected_parent_by_cell = {
        "d5_full_domain_core": ("d5_a", "d5_b", True),
        "d2_5shot": ("d2_a", "d2_b", True),
        "d2_10shot": ("d2_a", "d2_b", True),
        "d2_20shot": ("d2_a", "d2_b", True),
        "d4_full_domain_core": ("d4_a", "d4_b", False),
    }
    for cell_id in CELL_IDS:
        if cell_id == "d1_full_domain_core":
            continue
        identity = pairing[cell_id]
        metric = reconciliation[cell_id]
        alignment = parent_alignment[cell_id]
        expected_a, expected_b, b_exact = expected_parent_by_cell[cell_id]
        if (
            not isinstance(identity, Mapping)
            or set(identity) != {
                "cluster_count", "key_projection",
                "metric_projection", "row_count",
            }
            or any(
                not isinstance(identity.get(key), Mapping)
                or set(identity[key]) != {"bytes", "sha256"}
                for key in ("key_projection", "metric_projection")
            )
            or not isinstance(metric, Mapping)
            or set(metric) != {"mismatch_count", "max_abs_difference"}
            or not isinstance(alignment, Mapping)
            or alignment.get("protocol_a_parent") != expected_a
            or alignment.get("protocol_b_parent") != expected_b
            or alignment.get("protocol_a_exact_required") is not True
            or alignment.get("protocol_b_exact_required") is not b_exact
            or set(alignment) != {
                "protocol_a_parent", "protocol_b_parent",
                "protocol_a_payload_sha256", "protocol_b_payload_sha256",
                "protocol_a_exact_required", "protocol_b_exact_required",
            }
        ):
            raise Phase4ProtocolABContrastVerifierError(
                f"per-cell contract mismatch: {cell_id}"
            )


def _parse_config(
    path: Path, raw_bytes: bytes, *, require_frozen_identity: bool
) -> _VerifierConfig:
    authority = _authority_literals()
    try:
        document = json.loads(raw_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "config is not valid JSON"
        ) from error
    if not isinstance(document, dict) or _canonical_json(document) != raw_bytes:
        raise Phase4ProtocolABContrastVerifierError(
            "config is not canonical JSON"
        )
    digest = _sha_bytes(raw_bytes)
    if require_frozen_identity:
        config_bytes = int(authority["CONFIG_BYTES"])
        config_sha256 = str(authority["CONFIG_SHA256"])
        if config_bytes <= 0 or not config_sha256:
            raise Phase4ProtocolABContrastVerifierError(
                "real config identity is not frozen"
            )
        if len(raw_bytes) != config_bytes or digest != config_sha256:
            raise Phase4ProtocolABContrastVerifierError(
                "config frozen identity mismatch"
            )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4ProtocolABContrastVerifierError("config schema mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4ProtocolABContrastVerifierError("config experiment mismatch")
    if document.get("claim_boundary") != CLAIM_BOUNDARY:
        raise Phase4ProtocolABContrastVerifierError(
            "config claim boundary mismatch"
        )

    payloads = tuple(document.get("artifact_payload_files", ()))
    cells = tuple(document.get("cell_ids", ()))
    metrics = tuple(document.get("metric_output_ids", ()))
    perturbations = tuple(document.get("perturbation_ids", ()))
    try:
        alphas = tuple(float(value) for value in document.get("alpha_grid", ()))
    except (TypeError, ValueError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "config alpha grid is invalid"
        ) from error
    if payloads != ARTIFACT_PAYLOAD_FILES:
        raise Phase4ProtocolABContrastVerifierError(
            "artifact payload order mismatch"
        )
    if (
        not cells
        or len(set(cells)) != len(cells)
        or not metrics
        or metrics[0] != "mse"
        or len(set(metrics)) != len(metrics)
        or not perturbations
        or len(set(perturbations)) != len(perturbations)
        or not alphas
        or any(not math.isfinite(value) or value <= 0.0 for value in alphas)
        or len({value.hex() for value in alphas}) != len(alphas)
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "config scientific grid is invalid"
        )

    synthetic = bool(document.get("synthetic_fixture", False))
    if require_frozen_identity and synthetic:
        raise Phase4ProtocolABContrastVerifierError(
            "public verifier rejects synthetic config"
        )
    if not synthetic:
        if (
            cells != CELL_IDS
            or metrics != METRIC_OUTPUT_IDS
            or perturbations != PERTURBATION_IDS
            or alphas != ALPHAS
        ):
            raise Phase4ProtocolABContrastVerifierError(
                "real scientific grid mismatch"
            )
        if document.get("inference") != {
            "bootstrap_resamples": 2000,
            "confidence_level": 0.95,
            "holm_alpha": 0.05,
            "holm_slots_per_cell": 29,
            "random_seed": 20260817,
            "sign_flip_resamples": 100000,
        }:
            raise Phase4ProtocolABContrastVerifierError(
                "inference contract mismatch"
            )
        if document.get("code_authority") != _code_receipts():
            raise Phase4ProtocolABContrastVerifierError(
                "code authority mismatch"
            )
        if document.get("environment_authority") != _environment_receipt():
            raise Phase4ProtocolABContrastVerifierError(
                "environment authority mismatch"
            )
        if document.get("trust_anchor") != {
            "config_authority_relative_path": (
                "rpe/runner/phase4_protocol_ab_contrast_authority.py"
            ),
            "config_binds_authority": False,
            "direction": "authority_to_config_only",
        }:
            raise Phase4ProtocolABContrastVerifierError("trust anchor mismatch")
        if document.get("expected") != {
            "artifact_file_count": 14,
            "bootstrap_result_count": 475,
            "configured_payload_count": 12,
            "downstream_effect_row_count": 225,
            "endpoint_status_count": 6,
            "holm_slot_count": 145,
            "metric_effect_row_count": 65,
            "paired_observation_row_count": 525720,
            "sign_flip_result_count": 145,
        }:
            raise Phase4ProtocolABContrastVerifierError(
                "expected count contract mismatch"
            )
        authorities = document.get("authorities")
        if not isinstance(authorities, dict) or not authorities:
            raise Phase4ProtocolABContrastVerifierError(
                "scientific authorities are missing"
            )
        for name, receipt in authorities.items():
            try:
                source = ROOT / str(receipt["path"])
                expected_bytes = int(receipt["bytes"])
                expected_sha = str(receipt["sha256"])
            except (KeyError, TypeError, ValueError) as error:
                raise Phase4ProtocolABContrastVerifierError(
                    f"authority schema mismatch: {name}"
                ) from error
            if (
                not source.is_file()
                or source.stat().st_size != expected_bytes
                or _sha_file(source) != expected_sha
            ):
                raise Phase4ProtocolABContrastVerifierError(
                    f"scientific authority mismatch: {name}"
                )
        _validate_real_config_sections(document)
    return _VerifierConfig(
        path=Path(path),
        raw_bytes=raw_bytes,
        sha256=digest,
        document=MappingProxyType(document),
        payload_files=payloads,
        cell_ids=cells,
        metric_output_ids=metrics,
        perturbation_ids=perturbations,
        alpha_grid=alphas,
        synthetic_fixture=synthetic,
    )


def _environment_receipt() -> dict[str, str]:
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn_version,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl_version,
    }


def _code_receipts() -> dict[str, dict[str, object]]:
    receipts: dict[str, dict[str, object]] = {}
    for relative in CODE_RELATIVE_PATHS:
        target = ROOT / relative
        if not target.is_file():
            raise Phase4ProtocolABContrastVerifierError(
                f"missing code authority: {relative}"
            )
        receipts[relative] = {
            "bytes": target.stat().st_size,
            "sha256": _sha_file(target),
        }
    return receipts


def _read_json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            f"cannot read {label}"
        ) from error
    if not isinstance(value, dict):
        raise Phase4ProtocolABContrastVerifierError(f"{label} is not an object")
    return value


def _read_jsonl_rows(path: Path, label: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise Phase4ProtocolABContrastVerifierError(
                        f"{label} row is not an object: {line_number}"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            f"cannot read {label}"
        ) from error
    return rows


def _read_alpha0_csv(path: Path) -> Mapping[str, str]:
    try:
        with Path(path).open(encoding="utf-8", newline="") as stream:
            rows = [dict(row) for row in csv.DictReader(stream)]
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "cannot read D5 condition summary"
        ) from error
    alpha0 = [row for row in rows if row.get("condition_id") == "alpha0"]
    if len(rows) != 41 or len(alpha0) != 1:
        raise Phase4ProtocolABContrastVerifierError(
            "D5 alpha-zero summary denominator mismatch"
        )
    return MappingProxyType(alpha0[0])


def _validate_parent_tree(
    path: Path, expected: Mapping[str, object]
) -> Mapping[str, object]:
    parent = Path(path)
    ledger_path = parent / "SHA256SUMS"
    if not parent.is_dir() or not ledger_path.is_file():
        raise Phase4ProtocolABContrastVerifierError(
            "parent artifact or checksum ledger is missing"
        )
    if _sha_file(ledger_path) != str(expected.get("sha256sums_sha256", "")):
        raise Phase4ProtocolABContrastVerifierError(
            "parent ledger identity mismatch"
        )
    try:
        lines = ledger_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "parent checksum ledger cannot be read"
        ) from error
    entries: dict[str, str] = {}
    for line in lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4ProtocolABContrastVerifierError(
                "parent checksum ledger is malformed"
            ) from error
        if (
            name in entries
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Phase4ProtocolABContrastVerifierError(
                "parent checksum ledger is malformed"
            )
        entries[name] = digest
    inventory = expected.get("required_inventory")
    if isinstance(inventory, (str, bytes)) or not isinstance(inventory, Sequence):
        raise Phase4ProtocolABContrastVerifierError(
            "parent required inventory is malformed"
        )
    actual_inventory = sorted(item.name for item in parent.iterdir() if item.is_file())
    required_inventory = sorted(str(item) for item in inventory)
    if (
        actual_inventory != required_inventory
        or sorted([*entries, "SHA256SUMS"]) != required_inventory
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "parent artifact inventory mismatch"
        )
    for name, digest in entries.items():
        if _sha_file(parent / name) != digest:
            raise Phase4ProtocolABContrastVerifierError(
                f"parent payload checksum mismatch: {name}"
            )
    if parent.name != str(expected["directory_name"]):
        raise Phase4ProtocolABContrastVerifierError(
            "parent directory identity mismatch"
        )
    manifest_path = parent / "manifest.json"
    manifest = _read_json_object(manifest_path, "parent manifest")
    manifest_sha = _sha_file(manifest_path)
    if (
        "manifest_sha256" in expected
        and manifest_sha != str(expected["manifest_sha256"])
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "parent manifest identity mismatch"
        )
    manifest_run_id = str(manifest.get("run_id", manifest.get("run", "")))
    terminal_name = str(expected["terminal_name"])
    if terminal_name not in {"complete.json", "failed.json"}:
        raise Phase4ProtocolABContrastVerifierError(
            "parent terminal name is invalid"
        )
    terminal_path = parent / terminal_name
    terminal = _read_json_object(terminal_path, "parent terminal marker")
    terminal_sha = _sha_file(terminal_path)
    if (
        "terminal_sha256" in expected
        and terminal_sha != str(expected["terminal_sha256"])
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "parent terminal identity mismatch"
        )
    terminal_run_id = str(terminal.get("run_id", terminal.get("run", "")))
    if (
        manifest_run_id != str(expected["manifest_run_id"])
        or terminal_run_id != str(expected["terminal_run_id"])
        or manifest.get("status") != expected["status"]
        or terminal.get("status") != expected["status"]
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "parent terminal or run identity mismatch"
        )
    expected_endpoint_state = expected.get("endpoint_state")
    if expected_endpoint_state is not None:
        manifest_failure = manifest.get("failure")
        terminal_failure = terminal.get("failure")
        manifest_state = manifest.get(
            "endpoint_state",
            manifest_failure.get("state")
            if isinstance(manifest_failure, Mapping) else None,
        )
        terminal_state = terminal.get(
            "endpoint_state",
            terminal_failure.get("state")
            if isinstance(terminal_failure, Mapping) else None,
        )
        if (
            manifest_state != expected_endpoint_state
            or terminal_state != expected_endpoint_state
        ):
            raise Phase4ProtocolABContrastVerifierError(
                "parent endpoint state mismatch"
            )
    return MappingProxyType(
        {
            "checksum_entries": entries,
            "directory_name": parent.name,
            "manifest_run_id": manifest_run_id,
            "manifest_sha256": manifest_sha,
            "relative_path": str(expected["relative_path"]),
            "sha256sums_sha256": _sha_file(ledger_path),
            "status": str(terminal["status"]),
            "terminal_name": terminal_name,
            "terminal_run_id": terminal_run_id,
            "terminal_sha256": terminal_sha,
        }
    )


def _projection_receipt(
    cell_id: str,
    rows: Sequence[Mapping[str, object]],
    *,
    cluster_field: str,
    count_field: str | None,
    metrics: Sequence[str],
    perturbations: Sequence[str],
    alphas: Sequence[float],
    include_metric: bool,
) -> Mapping[str, object]:
    cluster_order: dict[str, int] = {}
    for row in rows:
        try:
            cluster_id = str(row[cluster_field])
        except KeyError as error:
            raise Phase4ProtocolABContrastVerifierError(
                f"projection cluster field missing: {cell_id}"
            ) from error
        cluster_order.setdefault(cluster_id, len(cluster_order))
    metric_order = {value: index for index, value in enumerate(metrics)}
    perturbation_order = {value: index for index, value in enumerate(perturbations)}
    alpha_order = {float(value).hex(): index for index, value in enumerate(alphas)}
    try:
        ordered = sorted(
            rows,
            key=lambda row: (
                metric_order[str(row["metric_output_id"])],
                perturbation_order[str(row["perturbation_id"])],
                alpha_order[float(row["alpha"]).hex()],
                cluster_order[str(row[cluster_field])],
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            f"projection row outside fixed grid: {cell_id}"
        ) from error
    digest = hashlib.sha256()
    byte_count = 0
    for row in ordered:
        item: dict[str, object] = {
            "alpha_hex": float(row["alpha"]).hex(),
            "cell_id": cell_id,
            "cluster_id": str(row[cluster_field]),
            "metric_output_id": str(row["metric_output_id"]),
            "perturbation_id": str(row["perturbation_id"]),
            "state": str(row.get("state", "complete")),
        }
        if include_metric:
            item["metric_harm_hex"] = float(row["metric_harm"]).hex()
        if count_field is not None:
            item["within_cluster_count"] = int(row[count_field])
        raw = _canonical_json(item)
        digest.update(raw)
        byte_count += len(raw)
    return MappingProxyType(
        {"bytes": byte_count, "sha256": digest.hexdigest()}
    )


def _normalize_parent_alignment(
    path: Path, shot_count: int | None
) -> Mapping[str, Mapping[str, object]]:
    rows = _read_jsonl_rows(path / "alignment_results.jsonl", "parent alignment")
    if shot_count is not None:
        rows = [row for row in rows if int(row.get("shot_count", -1)) == shot_count]
    if len(rows) != len(METRIC_OUTPUT_IDS):
        raise Phase4ProtocolABContrastVerifierError(
            "parent alignment row count mismatch"
        )
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        metric_id = str(row.get("metric_output_id", ""))
        if metric_id in result or metric_id not in METRIC_OUTPUT_IDS:
            raise Phase4ProtocolABContrastVerifierError(
                "parent alignment metric identity mismatch"
            )
        if row.get("state", row.get("metric_state")) != "complete":
            raise Phase4ProtocolABContrastVerifierError(
                "parent alignment state mismatch"
            )
        comparison = row.get("comparison")
        d_ag = row.get("d_ag")
        d_acc = row.get("d_acc")
        if isinstance(comparison, Mapping):
            d_ag = comparison.get("d_ag")
            d_acc = comparison.get("d_acc")
        ag = row.get("ag")
        acc = row.get("acc_cross", row.get("acc"))
        required = [ag, acc] if metric_id == "mse" else [ag, acc, d_ag, d_acc]
        if any(value is None or not math.isfinite(float(value)) for value in required):
            raise Phase4ProtocolABContrastVerifierError(
                "parent alignment numerical schema mismatch"
            )
        result[metric_id] = MappingProxyType(
            {
                "acc_cross": float(acc),
                "ag": float(ag),
                "d_acc": None if metric_id == "mse" else float(d_acc),
                "d_ag": None if metric_id == "mse" else float(d_ag),
            }
        )
    if tuple(result) != METRIC_OUTPUT_IDS:
        raise Phase4ProtocolABContrastVerifierError(
            "parent alignment metric order mismatch"
        )
    return MappingProxyType(result)


def _adapt_parent_rows(
    *,
    cell_id: str,
    rows_a: Sequence[Mapping[str, object]],
    rows_b: Sequence[Mapping[str, object]],
    cluster_field: str,
    cluster_kind: str,
    count_field: str | None,
    config: _VerifierConfig,
    legacy_missing_state: bool,
    parent_alignment_a: Mapping[str, Mapping[str, object]],
    parent_alignment_b: Mapping[str, Mapping[str, object]],
    parent_a_exact_required: bool,
    parent_b_exact_required: bool,
) -> _VerifierCell:
    cluster_ids: list[str] = []
    seen_clusters: set[str] = set()
    for row in rows_a:
        try:
            cluster_id = str(row[cluster_field])
        except KeyError as error:
            raise Phase4ProtocolABContrastVerifierError(
                f"observation cluster field missing: {cell_id}"
            ) from error
        if cluster_id not in seen_clusters:
            seen_clusters.add(cluster_id)
            cluster_ids.append(cluster_id)
    metrics = config.metric_output_ids
    perturbations = config.perturbation_ids
    alphas = config.alpha_grid
    expected_count = len(metrics) * len(cluster_ids) * len(perturbations) * len(alphas)
    if len(rows_a) != expected_count or len(rows_b) != expected_count:
        raise Phase4ProtocolABContrastVerifierError(
            f"observation grid row count mismatch: {cell_id}"
        )
    metric_index = {value: index for index, value in enumerate(metrics)}
    perturbation_index = {value: index for index, value in enumerate(perturbations)}
    alpha_index = {float(value).hex(): index for index, value in enumerate(alphas)}
    cluster_index = {value: index for index, value in enumerate(cluster_ids)}

    def keyed(
        rows: Sequence[Mapping[str, object]], side: str
    ) -> dict[tuple[str, str, str, str], tuple[float, float, int | None]]:
        result: dict[
            tuple[str, str, str, str], tuple[float, float, int | None]
        ] = {}
        for row in rows:
            if "state" not in row:
                if not legacy_missing_state:
                    raise Phase4ProtocolABContrastVerifierError(
                        f"{side} observation state missing"
                    )
            elif row["state"] != "complete":
                raise Phase4ProtocolABContrastVerifierError(
                    f"{side} observation state is not complete"
                )
            try:
                key = (
                    str(row["metric_output_id"]),
                    str(row["perturbation_id"]),
                    float(row["alpha"]).hex(),
                    str(row[cluster_field]),
                )
                metric_value = float(row["metric_harm"])
                downstream_value = float(row["downstream_harm"])
            except (KeyError, TypeError, ValueError) as error:
                raise Phase4ProtocolABContrastVerifierError(
                    f"{side} observation schema mismatch: {cell_id}"
                ) from error
            if key in result:
                raise Phase4ProtocolABContrastVerifierError(
                    f"{side} duplicate observation key: {cell_id}"
                )
            if (
                key[0] not in metric_index
                or key[1] not in perturbation_index
                or key[2] not in alpha_index
                or key[3] not in cluster_index
                or not math.isfinite(metric_value)
                or not math.isfinite(downstream_value)
            ):
                raise Phase4ProtocolABContrastVerifierError(
                    f"{side} observation outside fixed finite grid: {cell_id}"
                )
            count_value = None
            if count_field is not None:
                try:
                    count_value = int(row[count_field])
                except (KeyError, TypeError, ValueError) as error:
                    raise Phase4ProtocolABContrastVerifierError(
                        f"{side} within-cluster count is invalid: {cell_id}"
                    ) from error
                if count_value <= 0:
                    raise Phase4ProtocolABContrastVerifierError(
                        f"{side} within-cluster count is invalid: {cell_id}"
                    )
            result[key] = metric_value, downstream_value, count_value
        return result

    a_map = keyed(rows_a, "Protocol-A")
    b_map = keyed(rows_b, "Protocol-B")
    if set(a_map) != set(b_map):
        raise Phase4ProtocolABContrastVerifierError(
            f"A/B observation keys differ: {cell_id}"
        )
    metric_harm = np.empty(
        (len(metrics), len(cluster_ids), len(perturbations), len(alphas)),
        dtype=np.float64,
    )
    downstream_a = np.empty(metric_harm.shape[1:], dtype=np.float64)
    downstream_b = np.empty_like(downstream_a)
    seen_a: dict[tuple[str, str, str], float] = {}
    seen_b: dict[tuple[str, str, str], float] = {}
    mismatch_count = 0
    max_abs_difference = 0.0
    for metric_id in metrics:
        for perturbation_id in perturbations:
            for alpha in alphas:
                alpha_key = float(alpha).hex()
                for cluster_id in cluster_ids:
                    key = metric_id, perturbation_id, alpha_key, cluster_id
                    metric_a, harm_a, count_a = a_map[key]
                    metric_b, harm_b, count_b = b_map[key]
                    if count_a != count_b:
                        raise Phase4ProtocolABContrastVerifierError(
                            f"A/B within-cluster counts differ: {cell_id}"
                        )
                    mi = metric_index[metric_id]
                    ci = cluster_index[cluster_id]
                    pi = perturbation_index[perturbation_id]
                    ai = alpha_index[alpha_key]
                    metric_harm[mi, ci, pi, ai] = metric_a
                    condition = cluster_id, perturbation_id, alpha_key
                    for seen, value, side in (
                        (seen_a, harm_a, "A"), (seen_b, harm_b, "B")
                    ):
                        prior = seen.setdefault(condition, value)
                        if prior.hex() != value.hex():
                            raise Phase4ProtocolABContrastVerifierError(
                                f"Protocol-{side} downstream copies differ: {cell_id}"
                            )
                    downstream_a[ci, pi, ai] = harm_a
                    downstream_b[ci, pi, ai] = harm_b
                    if metric_a.hex() != metric_b.hex():
                        mismatch_count += 1
                        max_abs_difference = max(
                            max_abs_difference, abs(metric_a - metric_b)
                        )
    for value in (metric_harm, downstream_a, downstream_b):
        value.setflags(write=False)
    return _VerifierCell(
        cell_id=cell_id,
        cluster_kind=cluster_kind,
        cluster_ids=tuple(cluster_ids),
        metric_output_ids=metrics,
        perturbation_ids=perturbations,
        alpha_grid=alphas,
        metric_harm=metric_harm,
        downstream_a=downstream_a,
        downstream_b=downstream_b,
        metric_reconciliation=MappingProxyType(
            {
                "mismatch_count": mismatch_count,
                "max_abs_difference": max_abs_difference,
                "metric_authority": "protocol_a",
            }
        ),
        parent_alignment_a=parent_alignment_a,
        parent_alignment_b=parent_alignment_b,
        parent_a_exact_required=parent_a_exact_required,
        parent_b_exact_required=parent_b_exact_required,
    )


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise Phase4ProtocolABContrastVerifierError(
            f"{name} must be a positive integer"
        )
    return value


def _validated_cells(inputs: object, config: _VerifierConfig) -> tuple[object, ...]:
    try:
        cells = tuple(inputs.cells)
    except (AttributeError, TypeError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "synthetic inputs lack cells"
        ) from error
    expected_ids = tuple(
        cell_id for cell_id in config.cell_ids if cell_id != "d1_full_domain_core"
    )
    if tuple(getattr(cell, "cell_id", None) for cell in cells) != expected_ids:
        raise Phase4ProtocolABContrastVerifierError("input cell order mismatch")
    for cell in cells:
        cluster_ids = tuple(getattr(cell, "cluster_ids", ()))
        metrics = tuple(getattr(cell, "metric_output_ids", ()))
        perturbations = tuple(getattr(cell, "perturbation_ids", ()))
        alphas = tuple(float(value) for value in getattr(cell, "alpha_grid", ()))
        metric_harm = np.asarray(getattr(cell, "metric_harm", None), dtype=np.float64)
        downstream_a = np.asarray(getattr(cell, "downstream_a", None), dtype=np.float64)
        downstream_b = np.asarray(getattr(cell, "downstream_b", None), dtype=np.float64)
        parent_a = getattr(cell, "parent_alignment_a", {}) or {}
        parent_b = getattr(cell, "parent_alignment_b", {}) or {}
        metric_shape = (len(metrics), len(cluster_ids), len(perturbations), len(alphas))
        downstream_shape = metric_shape[1:]
        if (
            not cluster_ids
            or len(set(cluster_ids)) != len(cluster_ids)
            or metrics != config.metric_output_ids
            or perturbations != config.perturbation_ids
            or alphas != config.alpha_grid
            or metric_harm.shape != metric_shape
            or downstream_a.shape != downstream_shape
            or downstream_b.shape != downstream_shape
            or not np.isfinite(metric_harm).all()
            or not np.isfinite(downstream_a).all()
            or not np.isfinite(downstream_b).all()
        ):
            raise Phase4ProtocolABContrastVerifierError(
                f"invalid input cell: {getattr(cell, 'cell_id', '<unknown>')}"
            )
        if parent_a and (tuple(parent_a) != metrics or tuple(parent_b) != metrics):
            raise Phase4ProtocolABContrastVerifierError(
                f"parent alignment grid mismatch: {getattr(cell, 'cell_id', '<unknown>')}"
            )
    return cells


def _alignment_rows(
    cell: object, metric_index: int, downstream: np.ndarray
) -> tuple[AlignmentObservation, ...]:
    return tuple(
        AlignmentObservation(
            cluster_id=str(cluster_id),
            perturbation_id=str(perturbation_id),
            alpha=float(alpha),
            metric_harm=float(cell.metric_harm[metric_index, ci, pi, ai]),
            downstream_harm=float(downstream[ci, pi, ai]),
        )
        for ci, cluster_id in enumerate(cell.cluster_ids)
        for pi, perturbation_id in enumerate(cell.perturbation_ids)
        for ai, alpha in enumerate(cell.alpha_grid)
    )


@dataclass(frozen=True)
class _FixedXCache:
    pooled_inverse: np.ndarray
    pooled_group_count: int
    separate_inverse: tuple[np.ndarray, ...]
    separate_group_counts: tuple[int, ...]


def _fixed_x_cache(metric: np.ndarray) -> _FixedXCache:
    pooled_inverse = np.unique(metric.reshape(-1), return_inverse=True)[1]
    separate_inverse = tuple(
        np.unique(metric[:, perturbation, :].reshape(-1), return_inverse=True)[1]
        for perturbation in range(metric.shape[1])
    )
    pooled_inverse.setflags(write=False)
    for inverse in separate_inverse:
        inverse.setflags(write=False)
    return _FixedXCache(
        pooled_inverse=pooled_inverse,
        pooled_group_count=int(pooled_inverse.max()) + 1,
        separate_inverse=separate_inverse,
        separate_group_counts=tuple(
            int(inverse.max()) + 1 for inverse in separate_inverse
        ),
    )


def _weighted_pava_sse(
    inverse: np.ndarray,
    group_count: int,
    downstream: np.ndarray,
    observation_weights: np.ndarray,
) -> float:
    grouped_weights = np.bincount(
        inverse, weights=observation_weights, minlength=group_count
    )
    grouped_response = np.bincount(
        inverse, weights=observation_weights * downstream, minlength=group_count
    )
    grouped_response_squared = np.bincount(
        inverse,
        weights=observation_weights * downstream * downstream,
        minlength=group_count,
    )
    positive = grouped_weights > 0.0
    fitted = np.asarray(
        isotonic_regression(
            grouped_response[positive] / grouped_weights[positive],
            weights=grouped_weights[positive],
            increasing=True,
        ).x,
        dtype=np.float64,
    )
    return float(
        np.sum(
            grouped_response_squared[positive]
            - 2.0 * fitted * grouped_response[positive]
            + fitted * fitted * grouped_weights[positive]
        )
    )


def _weighted_alignment_gap(
    metric: np.ndarray,
    downstream: np.ndarray,
    cluster_weights: np.ndarray,
    cache: _FixedXCache,
) -> float:
    cluster_count, perturbation_count, alpha_count = downstream.shape
    observation_weights = np.repeat(
        cluster_weights, perturbation_count * alpha_count
    ).astype(np.float64)
    response = downstream.reshape(-1)
    total_weight = float(np.sum(observation_weights))
    mean = float(np.sum(observation_weights * response) / total_weight)
    sst = float(np.sum(observation_weights * (response - mean) ** 2))
    if not math.isfinite(sst) or sst <= 0.0:
        raise Phase4ProtocolABContrastVerifierError(
            "bootstrap downstream SST is not positive"
        )
    pooled_sse = _weighted_pava_sse(
        cache.pooled_inverse, cache.pooled_group_count, response, observation_weights
    )
    separate_sse = 0.0
    for perturbation_index in range(perturbation_count):
        current_response = downstream[:, perturbation_index, :].reshape(-1)
        current_weights = np.repeat(cluster_weights, alpha_count).astype(np.float64)
        separate_sse += _weighted_pava_sse(
            cache.separate_inverse[perturbation_index],
            cache.separate_group_counts[perturbation_index],
            current_response,
            current_weights,
        )
    raw = (pooled_sse - separate_sse) / sst
    if raw < -1e-12:
        raise Phase4ProtocolABContrastVerifierError(
            "bootstrap alignment gap is negative"
        )
    return 0.0 if raw < 0.0 else float(raw)


def _percentile_interval(values: np.ndarray) -> list[float]:
    tail_probability = (1.0 - 0.95) / 2.0
    lower, upper = np.quantile(
        values, [tail_probability, 1.0 - tail_probability], method="linear"
    )
    return [float(lower), float(upper)]


def _holm_rows(
    cell_id: str, slots: Sequence[tuple[str, float]]
) -> list[dict[str, object]]:
    if len({slot_id for slot_id, _ in slots}) != len(slots):
        raise Phase4ProtocolABContrastVerifierError(
            "Holm slot IDs must be unique"
        )
    adjusted = {
        row.hypothesis_id: row
        for row in holm_step_down(dict(slots), alpha=0.05)
    }
    return [
        {
            "adjusted_p_value": adjusted[slot_id].adjusted_p_value,
            "alpha": adjusted[slot_id].alpha,
            "cell_id": cell_id,
            "family_id": f"{cell_id}:protocol_ab",
            "family_size": adjusted[slot_id].family_size,
            "rank": adjusted[slot_id].rank,
            "raw_p_value": adjusted[slot_id].raw_p_value,
            "rejected": adjusted[slot_id].rejected,
            "slot_id": slot_id,
            "slot_order": order,
        }
        for order, (slot_id, _) in enumerate(slots)
    ]


def _recompute_cell(
    cell: object, *, bootstrap_resamples: int, sign_flip_resamples: int, seed: int
) -> Mapping[str, list[dict[str, object]]]:
    metric_harm = np.asarray(cell.metric_harm, dtype=np.float64)
    downstream_a = np.asarray(cell.downstream_a, dtype=np.float64)
    downstream_b = np.asarray(cell.downstream_b, dtype=np.float64)
    metric_count, cluster_count, perturbation_count, alpha_count = metric_harm.shape
    protocol_gap = downstream_a - downstream_b

    ag_a = np.empty(metric_count, dtype=np.float64)
    ag_b = np.empty(metric_count, dtype=np.float64)
    acc_a = np.empty(metric_count, dtype=np.float64)
    acc_b = np.empty(metric_count, dtype=np.float64)
    ag_cluster_a = np.empty((metric_count, cluster_count), dtype=np.float64)
    ag_cluster_b = np.empty_like(ag_cluster_a)
    acc_cluster_a = np.empty_like(ag_cluster_a)
    acc_cluster_b = np.empty_like(ag_cluster_a)
    for metric_index in range(metric_count):
        rows_a = _alignment_rows(cell, metric_index, downstream_a)
        rows_b = _alignment_rows(cell, metric_index, downstream_b)
        gap_a = alignment_gap(rows_a)
        gap_b = alignment_gap(rows_b)
        cross_a = cross_perturbation_accuracy(rows_a)
        cross_b = cross_perturbation_accuracy(rows_b)
        ag_a[metric_index] = gap_a.alignment_gap
        ag_b[metric_index] = gap_b.alignment_gap
        acc_a[metric_index] = cross_a.accuracy
        acc_b[metric_index] = cross_b.accuracy
        gap_a_by_cluster = {row.cluster_id: row.value for row in gap_a.cluster_contributions}
        gap_b_by_cluster = {row.cluster_id: row.value for row in gap_b.cluster_contributions}
        acc_a_by_cluster = {row.cluster_id: row.accuracy for row in cross_a.cluster_contributions}
        acc_b_by_cluster = {row.cluster_id: row.accuracy for row in cross_b.cluster_contributions}
        for cluster_index, cluster_id in enumerate(cell.cluster_ids):
            key = str(cluster_id)
            ag_cluster_a[metric_index, cluster_index] = gap_a_by_cluster[key]
            ag_cluster_b[metric_index, cluster_index] = gap_b_by_cluster[key]
            acc_cluster_a[metric_index, cluster_index] = acc_a_by_cluster[key]
            acc_cluster_b[metric_index, cluster_index] = acc_b_by_cluster[key]

    delta_ag = ag_b - ag_a
    delta_acc = acc_b - acc_a
    parent_d_ag_a = ag_a[0] - ag_a
    parent_d_ag_b = ag_b[0] - ag_b
    parent_d_acc_a = acc_a - acc_a[0]
    parent_d_acc_b = acc_b - acc_b[0]
    interaction_ag = parent_d_ag_b - parent_d_ag_a
    interaction_acc = parent_d_acc_b - parent_d_acc_a

    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(
        0, cluster_count, size=(bootstrap_resamples, cluster_count)
    )
    weights = np.zeros((bootstrap_resamples, cluster_count), dtype=np.int64)
    np.add.at(
        weights,
        (
            np.repeat(np.arange(bootstrap_resamples), cluster_count),
            draws.reshape(-1),
        ),
        1,
    )
    condition_bootstrap = np.empty(
        (bootstrap_resamples, perturbation_count, alpha_count), dtype=np.float64
    )
    integrated_bootstrap = np.empty(
        (bootstrap_resamples, perturbation_count), dtype=np.float64
    )
    ag_a_bootstrap = np.empty((bootstrap_resamples, metric_count), dtype=np.float64)
    ag_b_bootstrap = np.empty_like(ag_a_bootstrap)
    acc_a_bootstrap = np.empty_like(ag_a_bootstrap)
    acc_b_bootstrap = np.empty_like(ag_a_bootstrap)
    acc_a_bootstrap = weights @ acc_cluster_a.T / cluster_count
    acc_b_bootstrap = weights @ acc_cluster_b.T / cluster_count
    caches = tuple(_fixed_x_cache(metric_harm[index]) for index in range(metric_count))
    for resample_index, current_weights in enumerate(weights):
        condition_bootstrap[resample_index] = np.tensordot(
            current_weights, protocol_gap, axes=(0, 0)
        ) / cluster_count
        integrated_bootstrap[resample_index] = np.mean(
            condition_bootstrap[resample_index], axis=1
        )
        for metric_index in range(metric_count):
            ag_a_bootstrap[resample_index, metric_index] = _weighted_alignment_gap(
                metric_harm[metric_index], downstream_a, current_weights,
                caches[metric_index],
            )
            ag_b_bootstrap[resample_index, metric_index] = _weighted_alignment_gap(
                metric_harm[metric_index], downstream_b, current_weights,
                caches[metric_index],
            )

    delta_ag_bootstrap = ag_b_bootstrap - ag_a_bootstrap
    delta_acc_bootstrap = acc_b_bootstrap - acc_a_bootstrap
    interaction_ag_bootstrap = (
        ag_b_bootstrap[:, [0]] - ag_b_bootstrap
    ) - (ag_a_bootstrap[:, [0]] - ag_a_bootstrap)
    interaction_acc_bootstrap = (
        acc_b_bootstrap - acc_b_bootstrap[:, [0]]
    ) - (acc_a_bootstrap - acc_a_bootstrap[:, [0]])

    downstream_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    for perturbation_index, perturbation_id in enumerate(cell.perturbation_ids):
        for alpha_index, alpha in enumerate(cell.alpha_grid):
            interval = _percentile_interval(
                condition_bootstrap[:, perturbation_index, alpha_index]
            )
            downstream_rows.append(
                {
                    "alpha": float(alpha),
                    "cell_id": cell.cell_id,
                    "cluster_count": cluster_count,
                    "gap": float(np.mean(protocol_gap[:, perturbation_index, alpha_index])),
                    "interval": interval,
                    "mean_harm_a": float(
                        np.mean(downstream_a[:, perturbation_index, alpha_index])
                    ),
                    "mean_harm_b": float(
                        np.mean(downstream_b[:, perturbation_index, alpha_index])
                    ),
                    "perturbation_id": perturbation_id,
                    "state": "complete",
                    "summary_type": "alpha_specific",
                }
            )
            bootstrap_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "confidence_level": 0.95,
                    "interval": interval,
                    "metric_output_id": None,
                    "perturbation_id": perturbation_id,
                    "alpha": float(alpha),
                    "random_seed": seed,
                    "resamples": bootstrap_resamples,
                    "state": "complete",
                    "statistic": "gap",
                }
            )
        interval = _percentile_interval(
            integrated_bootstrap[:, perturbation_index]
        )
        downstream_rows.append(
            {
                "alpha": None,
                "cell_id": cell.cell_id,
                "cluster_count": cluster_count,
                "gap": float(np.mean(protocol_gap[:, perturbation_index, :])),
                "interval": interval,
                "mean_harm_a": float(
                    np.mean(downstream_a[:, perturbation_index, :])
                ),
                "mean_harm_b": float(
                    np.mean(downstream_b[:, perturbation_index, :])
                ),
                "perturbation_id": perturbation_id,
                "state": "complete",
                "summary_type": "integrated",
            }
        )
        bootstrap_rows.append(
            {
                "cell_id": cell.cell_id,
                "confidence_level": 0.95,
                "interval": interval,
                "metric_output_id": None,
                "perturbation_id": perturbation_id,
                "alpha": None,
                "random_seed": seed,
                "resamples": bootstrap_resamples,
                "state": "complete",
                "statistic": "g",
            }
        )

    metric_rows: list[dict[str, object]] = []
    parent_alignment_a = getattr(cell, "parent_alignment_a", {}) or {}
    parent_alignment_b = getattr(cell, "parent_alignment_b", {}) or {}
    for metric_index, metric_id in enumerate(cell.metric_output_ids):
        parent_a = parent_alignment_a.get(metric_id, {})
        parent_b = parent_alignment_b.get(metric_id, {})
        parent_acc_a = parent_a.get("acc_cross", parent_a.get("acc"))
        parent_acc_b = parent_b.get("acc_cross", parent_b.get("acc"))
        parent_ag_a = parent_a.get("ag")
        parent_ag_b = parent_b.get("ag")
        parent_ag_a_difference = (
            None if parent_ag_a is None
            else float(ag_a[metric_index] - float(parent_ag_a))
        )
        parent_ag_b_difference = (
            None if parent_ag_b is None
            else float(ag_b[metric_index] - float(parent_ag_b))
        )
        parent_acc_a_difference = (
            None if parent_acc_a is None
            else float(acc_a[metric_index] - float(parent_acc_a))
        )
        parent_acc_b_difference = (
            None if parent_acc_b is None
            else float(acc_b[metric_index] - float(parent_acc_b))
        )
        current = {
            "acc_a": float(acc_a[metric_index]),
            "acc_b": float(acc_b[metric_index]),
            "ag_a": float(ag_a[metric_index]),
            "ag_b": float(ag_b[metric_index]),
            "cell_id": cell.cell_id,
            "delta_acc": float(delta_acc[metric_index]),
            "delta_acc_interval": _percentile_interval(
                delta_acc_bootstrap[:, metric_index]
            ),
            "delta_ag": float(delta_ag[metric_index]),
            "delta_ag_interval": _percentile_interval(
                delta_ag_bootstrap[:, metric_index]
            ),
            "i_acc": None
            if metric_index == 0
            else float(interaction_acc[metric_index]),
            "i_acc_interval": None
            if metric_index == 0
            else _percentile_interval(interaction_acc_bootstrap[:, metric_index]),
            "i_ag": None
            if metric_index == 0
            else float(interaction_ag[metric_index]),
            "i_ag_interval": None
            if metric_index == 0
            else _percentile_interval(interaction_ag_bootstrap[:, metric_index]),
            "metric_output_id": metric_id,
            "canonical_d_acc_a": None
            if metric_index == 0
            else float(parent_d_acc_a[metric_index]),
            "canonical_d_acc_b": None
            if metric_index == 0
            else float(parent_d_acc_b[metric_index]),
            "canonical_d_ag_a": None
            if metric_index == 0
            else float(parent_d_ag_a[metric_index]),
            "canonical_d_ag_b": None
            if metric_index == 0
            else float(parent_d_ag_b[metric_index]),
            "parent_acc_a": None if parent_acc_a is None else float(parent_acc_a),
            "parent_acc_a_difference": parent_acc_a_difference,
            "parent_acc_b": None if parent_acc_b is None else float(parent_acc_b),
            "parent_acc_b_difference": parent_acc_b_difference,
            "parent_ag_a": None if parent_ag_a is None else float(parent_ag_a),
            "parent_ag_a_difference": parent_ag_a_difference,
            "parent_ag_b": None if parent_ag_b is None else float(parent_ag_b),
            "parent_ag_b_difference": parent_ag_b_difference,
            "parent_a_exact": None
            if parent_ag_a is None
            else parent_ag_a_difference == 0.0 and parent_acc_a_difference == 0.0,
            "parent_b_exact": None
            if parent_ag_b is None
            else parent_ag_b_difference == 0.0 and parent_acc_b_difference == 0.0,
            "parent_d_acc_a": None
            if metric_index == 0
            else (None if not parent_a else float(parent_a.get("d_acc"))),
            "parent_d_acc_b": None
            if metric_index == 0
            else (None if not parent_b else float(parent_b.get("d_acc"))),
            "parent_d_ag_a": None
            if metric_index == 0
            else (None if not parent_a else float(parent_a.get("d_ag"))),
            "parent_d_ag_b": None
            if metric_index == 0
            else (None if not parent_b else float(parent_b.get("d_ag"))),
            "state": "complete",
        }
        metric_rows.append(current)
    for statistic, values_by_metric in (
        ("delta_ag", delta_ag_bootstrap),
        ("delta_acc", delta_acc_bootstrap),
    ):
        for metric_index, metric_id in enumerate(cell.metric_output_ids):
            bootstrap_rows.append(
                {
                    "alpha": None,
                    "cell_id": cell.cell_id,
                    "confidence_level": 0.95,
                    "interval": _percentile_interval(
                        values_by_metric[:, metric_index]
                    ),
                    "metric_output_id": metric_id,
                    "perturbation_id": None,
                    "random_seed": seed,
                    "resamples": bootstrap_resamples,
                    "state": "complete",
                    "statistic": statistic,
                }
            )
    for statistic, values_by_metric in (
        ("i_ag", interaction_ag_bootstrap),
        ("i_acc", interaction_acc_bootstrap),
    ):
        for metric_index, metric_id in enumerate(
            cell.metric_output_ids[1:], start=1
        ):
            bootstrap_rows.append(
                {
                    "alpha": None,
                    "cell_id": cell.cell_id,
                    "confidence_level": 0.95,
                    "interval": _percentile_interval(
                        values_by_metric[:, metric_index]
                    ),
                    "metric_output_id": metric_id,
                    "perturbation_id": None,
                    "random_seed": seed,
                    "resamples": bootstrap_resamples,
                    "state": "complete",
                    "statistic": statistic,
                }
            )

    sign_flip_rows: list[dict[str, object]] = []
    slots: list[tuple[str, float]] = []
    for perturbation_index, perturbation_id in enumerate(cell.perturbation_ids):
        result = paired_contribution_sign_flip(
            tuple(
                float(value)
                for value in np.mean(
                    protocol_gap[:, perturbation_index, :], axis=1
                )
            ),
            aggregation="mean",
            resamples=sign_flip_resamples,
            random_seed=seed,
        )
        slot_id = f"downstream:{perturbation_id}"
        direction = (
            "attenuated_by_b"
            if result.observed > 0
            else "amplified_by_b"
            if result.observed < 0
            else "zero"
        )
        sign_flip_rows.append(
            {
                "aggregation": "mean",
                "cell_id": cell.cell_id,
                "direction": direction,
                "extreme_resamples": result.extreme_resamples,
                "metric_output_id": None,
                "observed": result.observed,
                "perturbation_id": perturbation_id,
                "p_value": result.p_value,
                "random_seed": seed,
                "resamples": sign_flip_resamples,
                "slot_id": slot_id,
                "state": "tested",
            }
        )
        slots.append((slot_id, result.p_value))

    for statistic in ("i_ag", "i_acc"):
        for metric_index, metric_id in enumerate(
            cell.metric_output_ids[1:], start=1
        ):
            if statistic == "i_ag":
                values = (
                    ag_cluster_b[0] - ag_cluster_b[metric_index]
                    - ag_cluster_a[0] + ag_cluster_a[metric_index]
                )
                aggregation = "sum"
                observed = interaction_ag[metric_index]
            else:
                values = (
                    acc_cluster_b[metric_index] - acc_cluster_b[0]
                    - acc_cluster_a[metric_index] + acc_cluster_a[0]
                )
                aggregation = "mean"
                observed = interaction_acc[metric_index]
            result = paired_contribution_sign_flip(
                tuple(float(value) for value in values),
                aggregation=aggregation,
                resamples=sign_flip_resamples,
                random_seed=seed,
            )
            if not math.isclose(
                result.observed, float(observed), rel_tol=0.0, abs_tol=1e-12
            ):
                raise Phase4ProtocolABContrastVerifierError(
                    "interaction contributions do not match point estimate"
                )
            slot_id = f"{statistic}:{metric_id}"
            direction = (
                "candidate_advantage_strengthened_under_b"
                if observed > 0
                else "candidate_advantage_weakened_under_b"
                if observed < 0
                else "zero"
            )
            sign_flip_rows.append(
                {
                    "aggregation": aggregation,
                    "cell_id": cell.cell_id,
                    "direction": direction,
                    "extreme_resamples": result.extreme_resamples,
                    "metric_output_id": metric_id,
                    "observed": float(observed),
                    "perturbation_id": None,
                    "p_value": result.p_value,
                    "random_seed": seed,
                    "resamples": sign_flip_resamples,
                    "slot_id": slot_id,
                    "state": "tested",
                }
            )
            slots.append((slot_id, result.p_value))

    holm_rows = _holm_rows(str(cell.cell_id), slots)
    sign_flip_by_slot = {row["slot_id"]: row for row in sign_flip_rows}
    for row in holm_rows:
        direction = sign_flip_by_slot[row["slot_id"]]["direction"]
        row["direction"] = direction
        row["directional_result"] = (
            direction if row["rejected"] else "not_rejected"
        )
        row["state"] = "tested"
    if parent_alignment_a:
        require_a = bool(getattr(cell, "parent_a_exact_required", True))
        require_b = bool(
            getattr(
                cell,
                "parent_b_exact_required",
                cell.cell_id != "d4_full_domain_core",
            )
        )
        a_contrasts_exact = all(
            row["metric_output_id"] == "mse"
            or (
                row["parent_d_ag_a"] == row["canonical_d_ag_a"]
                and row["parent_d_acc_a"] == row["canonical_d_acc_a"]
            )
            for row in metric_rows
        )
        b_contrasts_exact = all(
            row["metric_output_id"] == "mse"
            or (
                row["parent_d_ag_b"] == row["canonical_d_ag_b"]
                and row["parent_d_acc_b"] == row["canonical_d_acc_b"]
            )
            for row in metric_rows
        )
        if require_a and (
            not all(row["parent_a_exact"] is True for row in metric_rows)
            or not a_contrasts_exact
        ):
            raise Phase4ProtocolABContrastVerifierError(
                "Protocol-A parent alignment recomputation mismatch"
            )
        if require_b and (
            not all(row["parent_b_exact"] is True for row in metric_rows)
            or not b_contrasts_exact
        ):
            raise Phase4ProtocolABContrastVerifierError(
                "Protocol-B parent alignment recomputation mismatch"
            )
    return MappingProxyType(
        {
            "bootstrap_rows": bootstrap_rows,
            "downstream_rows": downstream_rows,
            "holm_rows": holm_rows,
            "metric_rows": metric_rows,
            "sign_flip_rows": sign_flip_rows,
        }
    )


def _downstream_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    projected = []
    for row in rows:
        interval = row["interval"]
        projected.append(
            {
                "cell_id": row["cell_id"],
                "perturbation_id": row["perturbation_id"],
                "summary_type": row["summary_type"],
                "alpha": row["alpha"],
                "mean_harm_a": row["mean_harm_a"],
                "mean_harm_b": row["mean_harm_b"],
                "gap": row["gap"],
                "interval_lower": interval[0],
                "interval_upper": interval[1],
                "cluster_count": row["cluster_count"],
                "state": row["state"],
            }
        )
    return _csv(projected, DOWNSTREAM_CSV_FIELDS)


def _metric_csv(rows: Sequence[Mapping[str, object]]) -> bytes:
    projected = []
    for row in rows:
        item = dict(row)
        for name in ("delta_ag", "delta_acc", "i_ag", "i_acc"):
            interval = item.pop(f"{name}_interval")
            item[f"{name}_lower"] = None if interval is None else interval[0]
            item[f"{name}_upper"] = None if interval is None else interval[1]
        projected.append(item)
    return _csv(projected, METRIC_CSV_FIELDS)


def _assemble_artifact(
    inputs: object,
    config: _VerifierConfig,
    *,
    worker_count: int,
    bootstrap_resamples: int,
    sign_flip_resamples: int,
) -> tuple[Mapping[str, bytes], Mapping[str, object]]:
    _positive_integer("worker_count", worker_count)
    bootstrap_count = _positive_integer(
        "bootstrap_resamples", bootstrap_resamples
    )
    sign_flip_count = _positive_integer(
        "sign_flip_resamples", sign_flip_resamples
    )
    cells = _validated_cells(inputs, config)
    seed = int(config.document.get("inference", {}).get("random_seed", 20260817))
    computations = tuple(
        _recompute_cell(
            cell,
            bootstrap_resamples=bootstrap_count,
            sign_flip_resamples=sign_flip_count,
            seed=seed,
        )
        for cell in cells
    )
    downstream_rows = [
        row for computation in computations for row in computation["downstream_rows"]
    ]
    metric_rows = [
        row for computation in computations for row in computation["metric_rows"]
    ]
    bootstrap_rows = [
        row for computation in computations for row in computation["bootstrap_rows"]
    ]
    sign_flip_rows = [
        row for computation in computations for row in computation["sign_flip_rows"]
    ]
    holm_rows = [
        row for computation in computations for row in computation["holm_rows"]
    ]
    try:
        endpoint_status_rows = tuple(inputs.endpoint_status_rows)
        authority_bridge = dict(inputs.authority_bridge)
        preflight = dict(inputs.preflight)
    except (AttributeError, TypeError, ValueError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "synthetic input metadata is invalid"
        ) from error
    if tuple(row.get("cell_id") for row in endpoint_status_rows) != config.cell_ids:
        raise Phase4ProtocolABContrastVerifierError(
            "endpoint status order mismatch"
        )

    code_authority = {} if config.synthetic_fixture else _code_receipts()
    environment_authority = _environment_receipt()
    parent_identity = dict(config.document.get("parent_artifacts", {}))
    run_id = _stable_run_id(
        config_sha256=config.sha256,
        parent_identity_projection=parent_identity,
        code_authority=code_authority,
        environment_authority=environment_authority,
    )
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": config.document.get("claim_boundary", CLAIM_BOUNDARY),
        "code_authority": code_authority,
        "config": {"bytes": len(config.raw_bytes), "sha256": config.sha256},
        "counts": {
            "artifact_files": len(ARTIFACT_PAYLOAD_FILES) + 2,
            "bootstrap_results": len(bootstrap_rows),
            "configured_payloads": len(ARTIFACT_PAYLOAD_FILES),
            "downstream_protocol_effects": len(downstream_rows),
            "endpoint_status": len(endpoint_status_rows),
            "holm_family": len(holm_rows),
            "metric_protocol_effects": len(metric_rows),
            "sign_flip_results": len(sign_flip_rows),
        },
        "environment_authority": environment_authority,
        "experiment_id": EXPERIMENT_ID,
        "failure": None,
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "run_id": run_id,
        "status": "complete",
        "synthetic_fixture": config.synthetic_fixture,
    }
    payloads = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": _canonical_json(authority_bridge),
        "preflight.json": _canonical_json(preflight),
        "endpoint_status.jsonl": _jsonl(endpoint_status_rows),
        "downstream_protocol_effects.jsonl": _jsonl(downstream_rows),
        "metric_protocol_effects.jsonl": _jsonl(metric_rows),
        "bootstrap_results.jsonl": _jsonl(bootstrap_rows),
        "sign_flip_results.jsonl": _jsonl(sign_flip_rows),
        "holm_family.jsonl": _jsonl(holm_rows),
        "protocol_downstream_contrast_table.csv": _downstream_csv(downstream_rows),
        "protocol_metric_interaction_table.csv": _metric_csv(metric_rows),
        "manifest.json": _canonical_json(manifest),
    }
    terminal = _canonical_json(
        {
            "endpoint_state": "complete",
            "run": run_id,
            "run_id": run_id,
            "schema": MARKER_SCHEMA_VERSION,
            "status": "complete",
        }
    )
    complete = {
        **payloads,
        "complete.json": terminal,
        "SHA256SUMS": _checksum_ledger(payloads, "complete.json", terminal),
    }
    return MappingProxyType(complete), MappingProxyType(manifest)


def _rebuild_synthetic(
    inputs: object,
    config: _VerifierConfig,
    *,
    worker_count: int,
    bootstrap_resamples: int,
    sign_flip_resamples: int,
) -> tuple[Mapping[str, bytes], Mapping[str, object]]:
    return _assemble_artifact(
        inputs,
        config,
        worker_count=worker_count,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )


def _validate_alpha0_admission(
    config: _VerifierConfig, parent_paths: Mapping[str, Path]
) -> Mapping[str, object]:
    alpha_contract = _require_mapping(config.document, "alpha0_admission")

    d2_path = parent_paths["d2_b"] / "alpha0_equivalence.json"
    d2 = _read_json_object(d2_path, "D2 alpha-zero receipt")
    d2_mismatches = d2.get("mismatch_counts")
    if (
        _sha_file(d2_path) != str(alpha_contract["d2_sha256"])
        or int(d2.get("mismatch_count", -1)) != 0
        or not isinstance(d2_mismatches, Mapping)
        or set(d2_mismatches) != {
            "model_cells_sha256", "predictions_sha256",
            "validation_scores_sha256",
        }
        or any(int(value) != 0 for value in d2_mismatches.values())
        or d2.get("bridge_digest") != d2.get("expected_step18_digest")
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "D2 alpha-zero admission failed"
        )

    d4_path = parent_paths["d4_b"] / "alpha0_equivalence.json"
    d4 = _read_json_object(d4_path, "D4 alpha-zero receipt")
    expected_d4_digests = d4.get("expected_digests")
    observed_d4_digests = d4.get("observed_digests")
    required_d4_digests = {
        "blank_prediction_digest", "model_digest", "prediction_digest",
        "technical_lod_loq_digest", "validation_digest",
    }
    if (
        _sha_file(d4_path) != str(alpha_contract["d4_sha256"])
        or d4.get("status") != "passed"
        or int(d4.get("mismatch_count", -1)) != 0
        or not isinstance(expected_d4_digests, Mapping)
        or set(expected_d4_digests) != required_d4_digests
        or expected_d4_digests != observed_d4_digests
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "D4 alpha-zero admission failed"
        )

    d5_bridge = _read_json_object(
        parent_paths["d5_b"] / "authority_bridge.json",
        "D5 authority bridge",
    )
    condition_bridge = d5_bridge.get("condition_bridge")
    operator_bridge = d5_bridge.get("operator_bridge")
    if (
        not isinstance(condition_bridge, Mapping)
        or condition_bridge.get("state") != "complete"
        or int(condition_bridge.get("mismatch_count", -1)) != 0
        or not isinstance(operator_bridge, Mapping)
        or operator_bridge.get("state") != "complete"
        or int(operator_bridge.get("mismatch_count", -1)) != 0
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "D5 authority bridge admission failed"
        )
    d5_a = _read_alpha0_csv(parent_paths["d5_a"] / "condition_summary.csv")
    d5_b = _read_alpha0_csv(parent_paths["d5_b"] / "condition_summary.csv")
    if dict(d5_a) != dict(d5_b):
        raise Phase4ProtocolABContrastVerifierError(
            "D5 shared alpha-zero row mismatch"
        )
    d5_raw = _canonical_json(dict(d5_a))
    d5_expected = alpha_contract["d5_shared_summary"]
    if (
        len(d5_raw) != int(d5_expected["bytes"])
        or _sha_bytes(d5_raw) != str(d5_expected["sha256"])
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "D5 shared alpha-zero identity mismatch"
        )

    d1_path = parent_paths["d1_b"] / "alpha0_equivalence.json"
    d1 = _read_json_object(d1_path, "D1 alpha-zero receipt")
    d1_mismatches = d1.get("mismatch_counts")
    if (
        _sha_file(d1_path) != str(alpha_contract["d1_sha256"])
        or
        d1.get("status") != "failed"
        or int(d1.get("mismatch_count", -1)) != 3
        or not isinstance(d1_mismatches, Mapping)
        or set(d1_mismatches) != {
            "model_digest", "prediction_digest", "validation_digest"
        }
        or any(int(value) != 1 for value in d1_mismatches.values())
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "D1 failed alpha-zero disposition mismatch"
        )
    d1_terminal = _read_json_object(
        parent_paths["d1_b"] / "failed.json", "D1 failed marker"
    )
    d1_failure = d1_terminal.get("failure")
    if (
        d1_terminal.get("status") != "failed"
        or d1_terminal.get("endpoint") != "failed_alpha0_equivalence"
        or not isinstance(d1_failure, Mapping)
        or d1_failure.get("condition_id") != "alpha0"
        or d1_failure.get("state") != "failed_alpha0_equivalence"
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "D1 failed terminal disposition mismatch"
        )
    return MappingProxyType(
        {
            "d2": {"mismatch_count": 0, "sha256": _sha_file(d2_path)},
            "d4": {
                "mismatch_count": 0, "sha256": _sha_file(d4_path),
                "status": "passed",
            },
            "d5": {
                "canonical_bytes": len(d5_raw),
                "canonical_sha256": _sha_bytes(d5_raw),
                "evidence": "shared_alpha0_summary",
            },
        }
    )


def reconstruct_phase4_protocol_ab_contrast_verifier_inputs(
    config: _VerifierConfig,
) -> _VerifierInputs:
    """Independently reconstruct the five cells and the closed D1 row."""

    if config.synthetic_fixture:
        raise Phase4ProtocolABContrastVerifierError(
            "real parent reconstruction rejects synthetic config"
        )
    parents = _require_mapping(config.document, "parent_artifacts")
    parent_order = (
        "d5_a", "d5_b", "d2_a", "d2_b",
        "d1_a", "d1_b", "d4_a", "d4_b",
    )
    parent_paths: dict[str, Path] = {}
    receipts: dict[str, Mapping[str, object]] = {}
    for name in parent_order:
        expected = parents[name]
        relative = Path(str(expected["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise Phase4ProtocolABContrastVerifierError(
                f"parent path is not workspace-relative: {name}"
            )
        parent_paths[name] = ROOT / relative
        receipts[name] = _validate_parent_tree(parent_paths[name], expected)

    alpha0_receipts = _validate_alpha0_admission(config, parent_paths)
    observation_contracts = _require_mapping(config.document, "observation_payloads")
    pairing_contracts = _require_mapping(config.document, "pairing_identities")
    reconciliation_contracts = _require_mapping(
        config.document, "metric_reconciliation"
    )
    alignment_contracts = _require_mapping(config.document, "parent_alignment")
    cell_specs = (
        ("d5_full_domain_core", "d5_a", "d5_b", "class_observations.jsonl", None, "cluster_id", "mineral_class", "occurrence_count"),
        ("d2_5shot", "d2_a", "d2_b", "class_observations.jsonl", 5, "class_label", "class_label", None),
        ("d2_10shot", "d2_a", "d2_b", "class_observations.jsonl", 10, "class_label", "class_label", None),
        ("d2_20shot", "d2_a", "d2_b", "class_observations.jsonl", 20, "class_label", "class_label", None),
        ("d4_full_domain_core", "d4_a", "d4_b", "well_observations.jsonl", None, "well_id", "physical_well", "acquisition_count"),
    )
    loaded_observations: dict[str, list[dict[str, object]]] = {}
    checked_payloads: set[str] = set()
    cells: list[_VerifierCell] = []
    status_rows: list[Mapping[str, object]] = []
    pairing_receipts: dict[str, Mapping[str, object]] = {}

    for (
        cell_id, a_name, b_name, filename, shot_count, cluster_field,
        cluster_kind, count_field,
    ) in cell_specs:
        for parent_name in (a_name, b_name):
            observation_path = parent_paths[parent_name] / filename
            observation_identity = observation_contracts[parent_name]
            if parent_name not in checked_payloads:
                if (
                    observation_path.stat().st_size
                    != int(observation_identity["bytes"])
                    or _sha_file(observation_path)
                    != str(observation_identity["sha256"])
                ):
                    raise Phase4ProtocolABContrastVerifierError(
                        f"observation payload identity mismatch: {parent_name}"
                    )
                checked_payloads.add(parent_name)
        alignment_contract = alignment_contracts[cell_id]
        if (
            alignment_contract["protocol_a_parent"] != a_name
            or alignment_contract["protocol_b_parent"] != b_name
        ):
            raise Phase4ProtocolABContrastVerifierError(
                f"parent alignment mapping mismatch: {cell_id}"
            )
        for side, parent_name in (("a", a_name), ("b", b_name)):
            alignment_path = parent_paths[parent_name] / "alignment_results.jsonl"
            if _sha_file(alignment_path) != str(
                alignment_contract[f"protocol_{side}_payload_sha256"]
            ):
                raise Phase4ProtocolABContrastVerifierError(
                    f"parent alignment payload mismatch: {cell_id}/{side}"
                )

        if a_name not in loaded_observations:
            loaded_observations[a_name] = _read_jsonl_rows(
                parent_paths[a_name] / filename, f"{a_name} observations"
            )
            loaded_observations[b_name] = _read_jsonl_rows(
                parent_paths[b_name] / filename, f"{b_name} observations"
            )
        rows_a: Sequence[Mapping[str, object]] = loaded_observations[a_name]
        rows_b: Sequence[Mapping[str, object]] = loaded_observations[b_name]
        if shot_count is not None:
            rows_a = [
                row for row in rows_a
                if int(row.get("shot_count", -1)) == shot_count
            ]
            rows_b = [
                row for row in rows_b
                if int(row.get("shot_count", -1)) == shot_count
            ]
        expected_pairing = pairing_contracts[cell_id]
        key_receipt = _projection_receipt(
            cell_id, rows_a, cluster_field=cluster_field, count_field=count_field,
            metrics=config.metric_output_ids, perturbations=config.perturbation_ids,
            alphas=config.alpha_grid, include_metric=False,
        )
        metric_receipt = _projection_receipt(
            cell_id, rows_a, cluster_field=cluster_field, count_field=count_field,
            metrics=config.metric_output_ids, perturbations=config.perturbation_ids,
            alphas=config.alpha_grid, include_metric=True,
        )
        if (
            dict(key_receipt) != dict(expected_pairing["key_projection"])
            or dict(metric_receipt) != dict(expected_pairing["metric_projection"])
        ):
            raise Phase4ProtocolABContrastVerifierError(
                f"pairing projection identity mismatch: {cell_id}"
            )
        parent_alignment_a = _normalize_parent_alignment(
            parent_paths[a_name], shot_count
        )
        parent_alignment_b = _normalize_parent_alignment(
            parent_paths[b_name], shot_count
        )
        cell = _adapt_parent_rows(
            cell_id=cell_id, rows_a=rows_a, rows_b=rows_b,
            cluster_field=cluster_field, cluster_kind=cluster_kind, config=config,
            count_field=count_field,
            legacy_missing_state=shot_count is not None,
            parent_alignment_a=parent_alignment_a,
            parent_alignment_b=parent_alignment_b,
            parent_a_exact_required=bool(
                alignment_contract["protocol_a_exact_required"]
            ),
            parent_b_exact_required=bool(
                alignment_contract["protocol_b_exact_required"]
            ),
        )
        expected_rows = int(expected_pairing["row_count"])
        if len(rows_a) != expected_rows or len(rows_b) != expected_rows:
            raise Phase4ProtocolABContrastVerifierError(
                f"pairing row count mismatch: {cell_id}"
            )
        if len(cell.cluster_ids) != int(expected_pairing["cluster_count"]):
            raise Phase4ProtocolABContrastVerifierError(
                f"cluster count mismatch: {cell_id}"
            )
        reconciliation = reconciliation_contracts[cell_id]
        if (
            int(cell.metric_reconciliation["mismatch_count"])
            != int(reconciliation["mismatch_count"])
            or float(cell.metric_reconciliation["max_abs_difference"]).hex()
            != float(reconciliation["max_abs_difference"]).hex()
        ):
            raise Phase4ProtocolABContrastVerifierError(
                f"metric reconciliation mismatch: {cell_id}"
            )
        cells.append(cell)
        pairing_receipts[cell_id] = {
            "cluster_count": len(cell.cluster_ids),
            "key_projection": dict(key_receipt),
            "metric_projection": dict(metric_receipt),
            "metric_reconciliation": dict(cell.metric_reconciliation),
            "row_count": expected_rows,
        }
        status_rows.append(
            {
                "alpha0_state": "passed",
                "cell_id": cell_id,
                "cluster_count": len(cell.cluster_ids),
                "cluster_kind": cluster_kind,
                "key_projection": dict(key_receipt),
                "metric_count": len(config.metric_output_ids),
                "metric_projection": dict(metric_receipt),
                "metric_reconciliation": dict(cell.metric_reconciliation),
                "paired_row_count": expected_rows,
                "perturbation_count": len(config.perturbation_ids),
                "protocol_a_parent": dict(receipts[a_name]),
                "protocol_b_parent": dict(receipts[b_name]),
                "reason": None,
                "state": "evaluable",
                "alpha0_evidence_type": (
                    "shared_alpha0_summary"
                    if cell_id == "d5_full_domain_core"
                    else "exact_projection_receipt"
                ),
            }
        )
        if shot_count is None:
            del loaded_observations[a_name]
            del loaded_observations[b_name]

    expected_total = int(config.document["expected"]["paired_observation_row_count"])
    if sum(int(receipt["row_count"]) for receipt in pairing_receipts.values()) != expected_total:
        raise Phase4ProtocolABContrastVerifierError(
            "paired observation total mismatch"
        )
    d1_status = {
        "alpha0_state": "failed_alpha0_equivalence",
        "alpha0_evidence_type": "exact_projection_receipt",
        "cell_id": "d1_full_domain_core",
        "cluster_count": 30,
        "cluster_kind": "class_label",
        "metric_count": 13,
        "paired_row_count": 0,
        "perturbation_count": 5,
        "reason": "protocol_b_failed_alpha0_equivalence",
        "protocol_a_parent": dict(receipts["d1_a"]),
        "protocol_b_parent": dict(receipts["d1_b"]),
        "state": "not_evaluable_failed_alpha0_equivalence",
    }
    status_by_cell = {str(row["cell_id"]): row for row in status_rows}
    status_by_cell["d1_full_domain_core"] = d1_status
    ordered_status = tuple(status_by_cell[cell_id] for cell_id in config.cell_ids)
    authority_bridge = MappingProxyType(
        {
            "alpha0_admission": dict(alpha0_receipts),
            "parents": {name: dict(receipts[name]) for name in parent_order},
            "state": "complete",
        }
    )
    preflight = MappingProxyType(
        {
            "cell_states": {
                str(row["cell_id"]): str(row["state"])
                for row in ordered_status
            },
            "evaluable_cell_count": 5,
            "fixed_cell_count": 6,
            "metric_authority": "protocol_a",
            "paired_observation_row_count": expected_total,
            "pairing_receipts": pairing_receipts,
            "state": "complete",
            "synthetic_fixture": False,
        }
    )
    return _VerifierInputs(
        cells=tuple(cells), endpoint_status_rows=ordered_status,
        authority_bridge=authority_bridge, preflight=preflight,
    )


def _validate_inventory(path: Path, config: _VerifierConfig) -> str:
    artifact = Path(path)
    if not artifact.is_dir():
        raise Phase4ProtocolABContrastVerifierError(
            "artifact path is not a directory"
        )
    try:
        manifest_raw = (artifact / "manifest.json").read_bytes()
        manifest = json.loads(manifest_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "artifact manifest is invalid"
        ) from error
    if not isinstance(manifest, dict) or _canonical_json(manifest) != manifest_raw:
        raise Phase4ProtocolABContrastVerifierError(
            "artifact manifest is not canonical"
        )
    status = manifest.get("status")
    if status == "complete":
        marker = "complete.json"
    elif status == "failed":
        marker = "failed.json"
    else:
        raise Phase4ProtocolABContrastVerifierError(
            "artifact terminal status is invalid"
        )
    expected_inventory = set(config.payload_files) | {marker, "SHA256SUMS"}
    if {entry.name for entry in artifact.iterdir()} != expected_inventory:
        raise Phase4ProtocolABContrastVerifierError(
            "artifact inventory/terminal mismatch"
        )
    if (
        manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION
        or manifest.get("experiment_id") != EXPERIMENT_ID
        or manifest.get("payload_files") != list(config.payload_files)
        or manifest.get("claim_boundary") != CLAIM_BOUNDARY
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "artifact manifest schema mismatch"
        )
    try:
        marker_raw = (artifact / marker).read_bytes()
        marker_document = json.loads(marker_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "artifact terminal marker is invalid"
        ) from error
    if (
        not isinstance(marker_document, dict)
        or _canonical_json(marker_document) != marker_raw
        or marker_document.get("run_id") != manifest.get("run_id")
        or marker_document.get("status") != status
        or marker_document.get("schema") != MARKER_SCHEMA_VERSION
    ):
        raise Phase4ProtocolABContrastVerifierError(
            "artifact terminal marker mismatch"
        )

    try:
        ledger_lines = (artifact / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise Phase4ProtocolABContrastVerifierError(
            "artifact checksum ledger is invalid"
        ) from error
    ledger: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line in ledger_lines:
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4ProtocolABContrastVerifierError(
                "artifact checksum ledger is malformed"
            ) from error
        if (
            name in seen
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise Phase4ProtocolABContrastVerifierError(
                "artifact checksum ledger is malformed"
            )
        seen.add(name)
        ledger.append((name, digest))
    expected_order = config.payload_files + (marker,)
    if tuple(name for name, _ in ledger) != expected_order:
        raise Phase4ProtocolABContrastVerifierError(
            "artifact checksum order mismatch"
        )
    for name, digest in ledger:
        if _sha_file(artifact / name) != digest:
            raise Phase4ProtocolABContrastVerifierError(
                f"artifact checksum mismatch for {name}"
            )
    return marker


def _compare_bytes(path: Path, rebuilt: Mapping[str, bytes]) -> None:
    actual_names = {entry.name for entry in Path(path).iterdir()}
    if actual_names != set(rebuilt):
        raise Phase4ProtocolABContrastVerifierError(
            "rebuilt artifact inventory mismatch"
        )
    for name, expected in rebuilt.items():
        if (Path(path) / name).read_bytes() != expected:
            raise Phase4ProtocolABContrastVerifierError(
                f"byte mismatch for {name}"
            )


def verify_phase4_protocol_ab_contrast_from_inputs(
    path: Path,
    *,
    inputs: object,
    config_path: Path,
    config_bytes: bytes,
    worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4ProtocolABContrastVerifierSummary:
    """Independently rebuild and compare a synthetic contrast artifact."""

    _positive_integer("worker_count", worker_count)
    config = _parse_config(
        Path(config_path), bytes(config_bytes), require_frozen_identity=False
    )
    if not config.synthetic_fixture:
        raise Phase4ProtocolABContrastVerifierError(
            "non-synthetic inputs require the public verifier"
        )
    if bootstrap_resamples is None or sign_flip_resamples is None:
        raise Phase4ProtocolABContrastVerifierError(
            "synthetic verifier requires explicit inference resamples"
        )
    _validate_inventory(Path(path), config)
    rebuilt, manifest = _rebuild_synthetic(
        inputs,
        config,
        worker_count=worker_count,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )
    _compare_bytes(Path(path), rebuilt)
    counts = manifest["counts"]
    return Phase4ProtocolABContrastVerifierSummary(
        path=Path(path),
        run_id=str(manifest["run_id"]),
        status=str(manifest["status"]),
        endpoint_status_count=int(counts["endpoint_status"]),
        bootstrap_result_count=int(counts["bootstrap_results"]),
        holm_slot_count=int(counts["holm_family"]),
    )


def verify_phase4_protocol_ab_contrast(
    path: Path, *, worker_count: int = 5
) -> Phase4ProtocolABContrastVerifierSummary:
    """Validate a retained real artifact before independent reconstruction."""

    _positive_integer("worker_count", worker_count)
    config_path = Path(DEFAULT_CONFIG)
    try:
        config_raw = config_path.read_bytes()
    except OSError as error:
        raise Phase4ProtocolABContrastVerifierError(
            "real config cannot be read"
        ) from error
    config = _parse_config(
        config_path, config_raw, require_frozen_identity=True
    )
    _validate_inventory(Path(path), config)
    inputs = reconstruct_phase4_protocol_ab_contrast_verifier_inputs(config)
    inference = config.document["inference"]
    rebuilt, manifest = _assemble_artifact(
        inputs,
        config,
        worker_count=worker_count,
        bootstrap_resamples=int(inference["bootstrap_resamples"]),
        sign_flip_resamples=int(inference["sign_flip_resamples"]),
    )
    _compare_bytes(Path(path), rebuilt)
    counts = manifest["counts"]
    return Phase4ProtocolABContrastVerifierSummary(
        path=Path(path),
        run_id=str(manifest["run_id"]),
        status=str(manifest["status"]),
        endpoint_status_count=int(counts["endpoint_status"]),
        bootstrap_result_count=int(counts["bootstrap_results"]),
        holm_slot_count=int(counts["holm_family"]),
    )


__all__ = [
    "Phase4ProtocolABContrastVerifierError",
    "Phase4ProtocolABContrastVerifierSummary",
    "reconstruct_phase4_protocol_ab_contrast_verifier_inputs",
    "verify_phase4_protocol_ab_contrast",
    "verify_phase4_protocol_ab_contrast_from_inputs",
]
