from __future__ import annotations

import hashlib
import json
import math
import platform
import struct
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from rpe.downstream.sugar_quantitative import D4SugarCohort, load_d4_sugar_cohort


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d4_protocol_b_all_role_eligibility_v1.json"
PHASE05_D4_PROTOCOL_RELATIVE_PATH = "experiments/phase05/configs/d4_sugar_protocol.json"
ARCHIVE_RELATIVE_PATH = "data/raw/ramanbench/cache/10779223/Raw data.zip"
PARENT_RUN_RELATIVE_PATH = (
    "results/phase4/d4_protocol_a_full_domain_eligibility_v1/"
    "phase4-d4-protocol-a-full-domain-eligibility-"
    "0f45ae5a0815ecb852b410549ad4582c4916dbe16fa0e3e20e2079f7d088097f"
)
SCHEMA_VERSION = "phase4-d4-protocol-b-all-role-eligibility-config-v1"
EXPERIMENT_ID = "phase4-d4-protocol-b-all-role-eligibility-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d4-protocol-b-all-role-eligibility-artifact-v1"
MARKER_SCHEMA_VERSION = "phase4-d4-protocol-b-all-role-eligibility-marker-v1"
RUN_PREFIX = "phase4-d4-protocol-b-all-role-eligibility-"
ACTIVE_PERTURBATION_IDS = ("p08", "p09", "p10", "p11", "p12")
ALL_PARENT_PERTURBATION_IDS = tuple(f"p{index:02d}" for index in range(1, 13))
ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
POSITIVE_ALPHAS = ALPHA_GRID[1:]
CONDITION_IDS = ("alpha0",) + tuple(
    f"{perturbation_id}:{struct.pack('<d', alpha).hex()}"
    for perturbation_id in ACTIVE_PERTURBATION_IDS
    for alpha in POSITIVE_ALPHAS
)
ROLE_ORDER = ("train", "validation", "test")
ARTIFACT_PAYLOAD_FILES = (
    "config.json",
    "parent_bridge.json",
    "role_condition_summaries.jsonl",
    "model_condition_readiness.jsonl",
    "blank_condition_summaries.jsonl",
    "gate.json",
    "manifest.json",
)
PARENT_INVENTORY = (
    "config.json",
    "source_records.jsonl",
    "well_folds.jsonl",
    "model_cells.jsonl",
    "model_role_occurrences.jsonl",
    "operator_cells.jsonl",
    "blank_cells.jsonl",
    "record_conditions.jsonl",
    "blank_conditions.jsonl",
    "well_summaries.jsonl",
    "common_support.jsonl",
    "gate.json",
    "manifest.json",
    "failed.json",
    "SHA256SUMS",
)
PARENT_LEDGER_ORDER = tuple(name for name in PARENT_INVENTORY if name != "SHA256SUMS")
FORBIDDEN_OUTCOME_KEYS = frozenset(
    {
        "selected_n_components",
        "validation_scores",
        "prediction",
        "predicted_targets",
        "metric_value",
        "lod",
        "loq",
        "alignment",
        "bootstrap",
        "sign_flip",
        "holm",
        "table",
        "figure",
    }
)
FORBIDDEN_PATH_FRAGMENTS = (
    "results/phase4/d4_protocol_a_full_domain_v1/",
    "results/phase05/d4/",
)


class Phase4D4ProtocolBEligibilityVerifierError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D4ProtocolBEligibilitySummary:
    path: Path
    run_id: str
    status: str
    marker_filename: str
    mixture_record_count: int
    model_cell_count: int
    role_condition_summary_count: int
    model_condition_readiness_count: int


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(_json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise Phase4D4ProtocolBEligibilityVerifierError("json", "unsupported value")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(values) + "\n").encode("utf-8"))


def _frozen_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def _environment_document() -> dict[str, object]:
    return {
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "system": platform.system(),
    }


def _sha256_member(archive_path: Path, member_path: str) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(member_path) as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                size += len(chunk)
                digest.update(chunk)
    return size, digest.hexdigest()


def _json(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if raw != _canonical_json_bytes(value):
        raise Phase4D4ProtocolBEligibilityVerifierError(str(path), "noncanonical JSON")
    if not isinstance(value, Mapping):
        raise Phase4D4ProtocolBEligibilityVerifierError(str(path), "must be an object")
    return value


def _jsonl_count(path: Path) -> int:
    count = 0
    with path.open("rb") as stream:
        for count, raw in enumerate(stream, start=1):
            value = json.loads(raw)
            if raw != _canonical_json_bytes(value):
                raise Phase4D4ProtocolBEligibilityVerifierError(str(path), "noncanonical JSONL")
    return count


def _parse_config(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if raw != _canonical_json_bytes(document):
        raise Phase4D4ProtocolBEligibilityVerifierError(str(path), "noncanonical config")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise Phase4D4ProtocolBEligibilityVerifierError("schema_version", "mismatch")
    if document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D4ProtocolBEligibilityVerifierError("experiment_id", "mismatch")
    if tuple(document.get("canonical_condition_ids", ())) != CONDITION_IDS:
        raise Phase4D4ProtocolBEligibilityVerifierError("canonical_condition_ids", "mismatch")
    if tuple(document.get("artifact_payload_files", ())) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D4ProtocolBEligibilityVerifierError("artifact_payload_files", "mismatch")
    if not document.get("synthetic_fixture", False):
        if dict(document.get("environment_authority", {})) != _environment_document():
            raise Phase4D4ProtocolBEligibilityVerifierError("environment_authority", "mismatch")
        for name, receipt_value in dict(document.get("authorities", {})).items():
            if not isinstance(receipt_value, Mapping):
                raise Phase4D4ProtocolBEligibilityVerifierError(f"authorities.{name}", "bad receipt")
            expected_bytes = int(receipt_value["bytes"])
            expected_sha = str(receipt_value["sha256"])
            if "member_path" in receipt_value:
                observed_bytes, observed_sha = _sha256_member(
                    ROOT / str(receipt_value["archive_path"]),
                    str(receipt_value["member_path"]),
                )
            else:
                live_path = ROOT / str(receipt_value["path"])
                observed_bytes, observed_sha = live_path.stat().st_size, _sha256_file(live_path)
            if observed_bytes != expected_bytes or observed_sha != expected_sha:
                raise Phase4D4ProtocolBEligibilityVerifierError(f"authorities.{name}", "authority receipt mismatch")
    return document


def _parse_sha256sums(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if len(digest) != 64:
            raise Phase4D4ProtocolBEligibilityVerifierError("SHA256SUMS", "bad digest")
        entries[name] = digest
    return entries


def _assert_artifact_bytes(path: Path, marker: str) -> None:
    expected_names = set(ARTIFACT_PAYLOAD_FILES) | {marker, "SHA256SUMS"}
    observed_names = {item.name for item in path.iterdir() if item.is_file()}
    if observed_names != expected_names:
        raise Phase4D4ProtocolBEligibilityVerifierError(str(path), "artifact inventory mismatch")
    ledger = _parse_sha256sums(path / "SHA256SUMS")
    if tuple(ledger) != (*ARTIFACT_PAYLOAD_FILES, marker):
        raise Phase4D4ProtocolBEligibilityVerifierError("SHA256SUMS", "ledger order mismatch")
    for name, expected in ledger.items():
        if _sha256_file(path / name) != expected:
            raise Phase4D4ProtocolBEligibilityVerifierError(name, "SHA256 mismatch")


def _firewall(value: object, path: str = "artifact") -> None:
    if isinstance(value, Mapping):
        bad = sorted(FORBIDDEN_OUTCOME_KEYS & {str(key) for key in value})
        if bad:
            raise Phase4D4ProtocolBEligibilityVerifierError(path, f"forbidden outcome keys {bad}")
        for key, item in value.items():
            _firewall(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _firewall(item, f"{path}[{index}]")
    elif isinstance(value, str):
        lowered = value.lower()
        if any(fragment in lowered for fragment in FORBIDDEN_PATH_FRAGMENTS):
            raise Phase4D4ProtocolBEligibilityVerifierError(path, "forbidden step29 path")


def _read_jsonl(path: Path):
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, start=1):
            value = json.loads(raw)
            if raw != _canonical_json_bytes(value):
                raise Phase4D4ProtocolBEligibilityVerifierError(f"{path.name}:{number}", "noncanonical JSONL")
            if not isinstance(value, Mapping):
                raise Phase4D4ProtocolBEligibilityVerifierError(f"{path.name}:{number}", "row must be object")
            yield raw, value


def _jsonl_documents(path: Path, *, apply_firewall: bool = True) -> tuple[Mapping[str, object], ...]:
    rows = []
    for _, row in _read_jsonl(path):
        if apply_firewall:
            _firewall(row, path.name)
        rows.append(row)
    return tuple(rows)


def _reconstruct_inputs(cohort: D4SugarCohort, config: Mapping[str, object]) -> dict[str, object]:
    den = config["denominators"]
    record_ids = tuple(str(value) for value in cohort.record_ids)
    well_ids = tuple(str(value) for value in cohort.well_ids)
    blank_ids = tuple(str(value) for value in cohort.blank_record_ids)
    support_sha = str(config.get("frozen_identities", {}).get("support_axis_sha256", ""))
    if len(record_ids) != den["mixture_record_count"] or len(blank_ids) != den["blank_record_count"]:
        raise Phase4D4ProtocolBEligibilityVerifierError("cohort", "count mismatch")
    record_to_well = dict(zip(record_ids, well_ids, strict=True))
    parent_source_records = tuple(
        {"record_id": str(record_id), "scope": "blank_auxiliary", "support_axis_sha256": support_sha, "well_id": str(well_id)}
        for record_id, well_id in zip(cohort.blank_record_ids, cohort.blank_well_ids, strict=True)
    ) + tuple(
        {"record_id": str(record_id), "scope": "mixture", "support_axis_sha256": support_sha, "well_id": str(well_id)}
        for record_id, well_id in zip(cohort.record_ids, cohort.well_ids, strict=True)
    )
    source_records = tuple(
        {"record_id": record_id, "record_order": index, "source_member": str(cohort.source_members[index]), "well_id": well_ids[index]}
        for index, record_id in enumerate(record_ids)
    )
    well_folds = []
    model_cells = []
    role_rows = []
    role_map: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seed_role_records = {}
    seed_role_wells = {}
    for split in cohort.splits:
        seed = int(split.seed)
        indices_by_role = {
            "train": tuple(int(value) for value in split.train_indices.tolist()),
            "validation": tuple(int(value) for value in split.validation_indices.tolist()),
            "test": tuple(int(value) for value in split.test_indices.tolist()),
        }
        records_by_role = {role: tuple(record_ids[index] for index in indices) for role, indices in indices_by_role.items()}
        wells_by_role = {role: tuple(well_ids[index] for index in indices) for role, indices in indices_by_role.items()}
        sources_by_role = {role: tuple(str(cohort.source_members[index]) for index in indices) for role, indices in indices_by_role.items()}
        frozen_fold_well_ids = _frozen_strings(config.get("frozen_identities", {}).get("fold_well_ids_sha256"))
        fold_well_ids_sha256 = (
            frozen_fold_well_ids[seed]
            if not bool(config.get("synthetic_fixture", False)) and len(frozen_fold_well_ids) == int(den["model_cell_count"])
            else _ids_digest(tuple(dict.fromkeys(wells_by_role["test"])))
        )
        model_cells.append(
            {
                "seed": seed,
                "train_record_count": len(records_by_role["train"]),
                "validation_record_count": len(records_by_role["validation"]),
                "test_record_count": len(records_by_role["test"]),
                "train_record_ids": list(records_by_role["train"]),
                "validation_record_ids": list(records_by_role["validation"]),
                "test_record_ids": list(records_by_role["test"]),
                "train_folds": list(split.train_folds),
                "validation_fold": int(split.validation_fold),
                "test_fold": int(split.test_fold),
                "train_record_ids_sha256": _ids_digest(records_by_role["train"]),
                "validation_record_ids_sha256": _ids_digest(records_by_role["validation"]),
                "test_record_ids_sha256": _ids_digest(records_by_role["test"]),
                "train_well_ids_sha256": _ids_digest(wells_by_role["train"]),
                "validation_well_ids_sha256": _ids_digest(wells_by_role["validation"]),
                "test_well_ids_sha256": _ids_digest(wells_by_role["test"]),
            }
        )
        well_folds.append(
            {
                "fold_index": seed,
                "record_ids_sha256": _ids_digest(records_by_role["test"]),
                "source_members_sha256": _ids_digest(sources_by_role["test"]),
                "well_ids_sha256": fold_well_ids_sha256,
                "seed": seed,
                "train_record_ids_sha256": _ids_digest(records_by_role["train"]),
                "validation_record_ids_sha256": _ids_digest(records_by_role["validation"]),
                "test_record_ids_sha256": _ids_digest(records_by_role["test"]),
                "train_well_ids_sha256": _ids_digest(wells_by_role["train"]),
                "validation_well_ids_sha256": _ids_digest(wells_by_role["validation"]),
                "test_well_ids_sha256": _ids_digest(wells_by_role["test"]),
            }
        )
        for role in ROLE_ORDER:
            seed_role_records[f"{seed}:{role}"] = records_by_role[role]
            seed_role_wells[f"{seed}:{role}"] = wells_by_role[role]
            for record_id in records_by_role[role]:
                role_map[record_id].append((seed, role))
                role_rows.append({"record_id": record_id, "role": role, "seed": seed})
    blank_role_rows = tuple(
        {"record_id": record_id, "role": "blank_auxiliary", "seed": seed}
        for seed in range(int(den["model_cell_count"]))
        for record_id in blank_ids
    )
    return {
        "source_records": source_records,
        "parent_source_records": parent_source_records,
        "well_folds": tuple(well_folds),
        "model_cells": tuple(model_cells),
        "model_role_occurrences": tuple(role_rows),
        "blank_role_occurrences": blank_role_rows,
        "record_to_well": record_to_well,
        "record_to_roles": {key: tuple(value) for key, value in role_map.items()},
        "seed_role_record_ids": seed_role_records,
        "seed_role_well_ids": seed_role_wells,
        "source_record_ids_sha256": _ids_digest(record_ids),
        "source_well_ids_sha256": _ids_digest(well_ids),
        "model_cells_sha256": _sha256_bytes(_jsonl_bytes(model_cells)),
        "model_role_occurrences_sha256": _sha256_bytes(_jsonl_bytes(role_rows)),
        "condition_ids_sha256": _ids_digest(CONDITION_IDS),
    }


def _subset_rows(rows: Sequence[Mapping[str, object]], expected: Sequence[Mapping[str, object]], name: str) -> None:
    if len(rows) != len(expected):
        raise Phase4D4ProtocolBEligibilityVerifierError(name, "semantic row count mismatch")
    for index, (row, want) in enumerate(zip(rows, expected, strict=True)):
        for key, value in want.items():
            if row.get(key) != value:
                raise Phase4D4ProtocolBEligibilityVerifierError(name, f"semantic mismatch at row {index}")


def _condition_complete(row: Mapping[str, object]) -> bool:
    if row.get("state") != "complete":
        return False
    metrics = row.get("metrics")
    if isinstance(metrics, Mapping) and metrics:
        if len(metrics) != 13:
            return False
        if any(not isinstance(value, Mapping) or value.get("state") != "complete" for value in metrics.values()):
            return False
    cwt = row.get("cwt")
    if isinstance(cwt, Mapping):
        return cwt.get("state") == "complete" and all(isinstance(cwt.get(key), str) and len(cwt[key]) == 64 for key in ("diagnostics_sha256", "peak_list_sha256"))
    return all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in ("result_sha256", "diagnostics_sha256", "peak_list_sha256"))


def _blank_condition_complete(row: Mapping[str, object]) -> bool:
    if row.get("state") != "complete":
        return False
    return all(isinstance(row.get(key), str) and len(row[key]) == 64 for key in ("result_sha256", "diagnostics_sha256", "warning_sha256"))


def _validate_parent(parent: Path, inputs: Mapping[str, object], config: Mapping[str, object]) -> Mapping[str, object]:
    observed = tuple(sorted(item.name for item in parent.iterdir() if item.is_file()))
    expected_inventory = tuple(sorted(config["parent_step27"]["required_inventory"]))
    if observed != expected_inventory:
        raise Phase4D4ProtocolBEligibilityVerifierError("parent inventory", "unexpected inventory")
    ledger = _parse_sha256sums(parent / "SHA256SUMS")
    if tuple(ledger) != PARENT_LEDGER_ORDER:
        raise Phase4D4ProtocolBEligibilityVerifierError("parent SHA256SUMS", "order mismatch")
    for name, expected in ledger.items():
        if _sha256_file(parent / name) != expected:
            raise Phase4D4ProtocolBEligibilityVerifierError(name, "checksum mismatch")
    gate = _json(parent / "gate.json")
    full = gate.get("full_domain_core", {})
    marker_filename = str(gate.get("marker_filename"))
    if not isinstance(full, Mapping) or full.get("state") != "evaluable" or marker_filename != "failed.json":
        raise Phase4D4ProtocolBEligibilityVerifierError("parent gate", "marker/state mismatch")
    _firewall(gate, "parent.gate")
    _firewall(_json(parent / "manifest.json"), "parent.manifest")
    synthetic = bool(config.get("synthetic_fixture", False))
    if synthetic:
        source_expected = tuple(
            {"record_id": row["record_id"], "well_id": row["well_id"], "source_member": row["source_member"], "state": "complete"}
            for row in inputs["source_records"]
        )
        fold_expected = tuple(
            {
                "seed": row["seed"],
                "train_record_ids_sha256": row["train_record_ids_sha256"],
                "validation_record_ids_sha256": row["validation_record_ids_sha256"],
                "test_record_ids_sha256": row["test_record_ids_sha256"],
                "train_well_ids_sha256": row["train_well_ids_sha256"],
                "validation_well_ids_sha256": row["validation_well_ids_sha256"],
                "test_well_ids_sha256": row["test_well_ids_sha256"],
            }
            for row in inputs["well_folds"]
        )
        model_expected = tuple(
            {"seed": row["seed"], "train_record_count": row["train_record_count"], "validation_record_count": row["validation_record_count"], "test_record_count": row["test_record_count"]}
            for row in inputs["model_cells"]
        )
        role_expected = inputs["model_role_occurrences"]
    else:
        source_expected = inputs["parent_source_records"]
        fold_expected = tuple(
            {"fold_index": i, "record_ids_sha256": row["test_record_ids_sha256"], "source_members_sha256": row["source_members_sha256"], "well_ids_sha256": row["well_ids_sha256"]}
            for i, row in enumerate(inputs["well_folds"])
        )
        model_expected = tuple(
            {"seed": row["seed"], "train_record_ids": row["train_record_ids"], "validation_record_ids": row["validation_record_ids"], "test_record_ids": row["test_record_ids"], "train_folds": row["train_folds"], "validation_fold": row["validation_fold"], "test_fold": row["test_fold"]}
            for row in inputs["model_cells"]
        )
        role_expected = []
        blank_by_seed: dict[int, list[Mapping[str, object]]] = defaultdict(list)
        for row in inputs["blank_role_occurrences"]:
            blank_by_seed[int(row["seed"])].append(row)
        for seed in range(int(config["denominators"]["model_cell_count"])):
            role_expected.extend(blank_by_seed[seed])
            for role in ("test", "train", "validation"):
                role_expected.extend(
                    {"record_id": record_id, "role": role, "seed": seed}
                    for record_id in inputs["seed_role_record_ids"][f"{seed}:{role}"]
                )
        role_expected = tuple(role_expected)
    apply_firewall = synthetic
    _subset_rows(_jsonl_documents(parent / "source_records.jsonl", apply_firewall=apply_firewall), source_expected, "source semantic mismatch")
    _subset_rows(_jsonl_documents(parent / "well_folds.jsonl", apply_firewall=apply_firewall), fold_expected, "fold semantic mismatch")
    _subset_rows(_jsonl_documents(parent / "model_cells.jsonl", apply_firewall=apply_firewall), model_expected, "model semantic mismatch")
    _subset_rows(_jsonl_documents(parent / "model_role_occurrences.jsonl", apply_firewall=apply_firewall), role_expected, "role semantic mismatch")
    operator_rows = _jsonl_documents(parent / "operator_cells.jsonl", apply_firewall=apply_firewall)
    operator_ids = ACTIVE_PERTURBATION_IDS if synthetic else ALL_PARENT_PERTURBATION_IDS
    if len(operator_rows) != int(config["denominators"]["mixture_record_count"]) * len(operator_ids):
        raise Phase4D4ProtocolBEligibilityVerifierError("operator_cells", "semantic mismatch")
    source_record_ids = {str(row["record_id"]) for row in inputs["source_records"]}
    source_wells = set(inputs["record_to_well"].values())
    operator_pairs = set()
    for row in operator_rows:
        record_id = str(row.get("record_id"))
        perturbation_id = str(row.get("perturbation_id"))
        if (
            record_id not in source_record_ids
            or perturbation_id not in operator_ids
            or str(row.get("state")) not in {"complete", "not_applicable", "structurally_ineligible", "failed_runtime"}
            or (record_id, perturbation_id) in operator_pairs
        ):
            raise Phase4D4ProtocolBEligibilityVerifierError("operator_cells", "semantic mismatch")
        if not synthetic and str(row.get("well_id")) != inputs["record_to_well"].get(record_id):
            raise Phase4D4ProtocolBEligibilityVerifierError("operator_cells", "semantic mismatch")
        operator_pairs.add((record_id, perturbation_id))
    blank_rows = _jsonl_documents(parent / "blank_cells.jsonl", apply_firewall=apply_firewall)
    if synthetic:
        if len(blank_rows) != int(config["denominators"]["blank_record_count"]):
            raise Phase4D4ProtocolBEligibilityVerifierError("blank_cells", "semantic mismatch")
    elif len(blank_rows) != int(config["denominators"]["blank_record_count"]) * len(ACTIVE_PERTURBATION_IDS):
        raise Phase4D4ProtocolBEligibilityVerifierError("blank_cells", "semantic mismatch")
    if not synthetic:
        blank_ids = {str(row["record_id"]) for row in inputs["blank_role_occurrences"]}
        blank_pairs = set()
        for row in blank_rows:
            pair = (str(row.get("record_id")), str(row.get("perturbation_id")))
            if (
                pair[0] not in blank_ids
                or str(row.get("well_id")) != "E1_3"
                or pair[1] not in ACTIVE_PERTURBATION_IDS
                or str(row.get("state")) not in {"complete", "runtime_failure", "not_applicable"}
                or pair in blank_pairs
            ):
                raise Phase4D4ProtocolBEligibilityVerifierError("blank_cells", "semantic mismatch")
            blank_pairs.add(pair)
    well_rows = _jsonl_documents(parent / "well_summaries.jsonl", apply_firewall=apply_firewall)
    if synthetic:
        if len(well_rows) != int(config["denominators"]["mixture_well_count"]):
            raise Phase4D4ProtocolBEligibilityVerifierError("well_summaries", "semantic mismatch")
    elif len(well_rows) != int(config["denominators"]["mixture_well_count"]) * len(ALL_PARENT_PERTURBATION_IDS):
        raise Phase4D4ProtocolBEligibilityVerifierError("well_summaries", "semantic mismatch")
    if not synthetic:
        well_pairs = set()
        for row in well_rows:
            pair = (str(row.get("well_id")), str(row.get("perturbation_id")))
            if (
                pair[0] not in source_wells
                or pair[1] not in ALL_PARENT_PERTURBATION_IDS
                or str(row.get("state")) not in {"complete", "not_evaluable_coverage"}
                or pair in well_pairs
            ):
                raise Phase4D4ProtocolBEligibilityVerifierError("well_summaries", "semantic mismatch")
            well_pairs.add(pair)
    support_rows = _jsonl_documents(parent / "common_support.jsonl", apply_firewall=apply_firewall)
    if synthetic:
        if len(support_rows) not in {1, len(CONDITION_IDS), int(config["denominators"]["mixture_record_count"])}:
            raise Phase4D4ProtocolBEligibilityVerifierError("common_support", "semantic mismatch")
    elif len(support_rows) != int(config["denominators"]["mixture_record_count"]) + int(config["denominators"]["mixture_well_count"]):
        raise Phase4D4ProtocolBEligibilityVerifierError("common_support", "semantic mismatch")
    if not synthetic:
        record_rows = support_rows[: int(config["denominators"]["mixture_record_count"])]
        well_support_rows = support_rows[int(config["denominators"]["mixture_record_count"]):]
        for expected_record, row in zip(inputs["source_records"], record_rows, strict=True):
            if (
                row.get("scope") != "record"
                or row.get("record_id") != expected_record["record_id"]
                or row.get("well_id") != expected_record["well_id"]
                or not isinstance(row.get("complete"), bool)
            ):
                raise Phase4D4ProtocolBEligibilityVerifierError("common_support", "semantic mismatch")
        for row in well_support_rows:
            if (
                row.get("scope") != "whole_well"
                or "record_id" in row
                or row.get("well_id") not in source_wells
                or not isinstance(row.get("complete"), bool)
            ):
                raise Phase4D4ProtocolBEligibilityVerifierError("common_support", "semantic mismatch")
    return {
        "parent_run_path": str(parent),
        "parent_run_relative_path": str(config["parent_step27"]["run_relative_path"]),
        "sha256sums_sha256": _sha256_file(parent / "SHA256SUMS"),
        "ledger": ledger,
        "full_domain_core_state": "evaluable",
        "parent_marker_filename": marker_filename,
        "source_record_count": len(inputs["source_records"]),
        "model_cell_count": len(inputs["model_cells"]),
        "model_role_occurrence_count": len(inputs["model_role_occurrences"]),
        "blank_role_occurrence_count": len(inputs["blank_role_occurrences"]),
        "outcome_firewall": "passed",
    }


def _build_lift(parent: Path, inputs: Mapping[str, object], config: Mapping[str, object]) -> Mapping[str, object]:
    den = config["denominators"]
    parent_bridge = _validate_parent(parent, inputs, config)
    stats = {}
    for seed in range(int(den["model_cell_count"])):
        for role in ROLE_ORDER:
            records = inputs["seed_role_record_ids"][f"{seed}:{role}"]
            wells = inputs["seed_role_well_ids"][f"{seed}:{role}"]
            for condition in CONDITION_IDS:
                stats[(seed, role, condition)] = {
                    "records": set(),
                    "wells": set(),
                    "complete": 0,
                    "not_applicable": 0,
                    "runtime": 0,
                    "duplicate": 0,
                    "hasher": hashlib.sha256(),
                    "expected_record_count": len(records),
                    "expected_well_count": len(set(wells)),
                    "record_ids_sha256": _ids_digest(records),
                    "well_ids_sha256": _ids_digest(tuple(dict.fromkeys(wells))),
                }
    seen = set()
    record_order = tuple(inputs["record_to_well"])
    row_count = 0
    for raw, row in _read_jsonl(parent / "record_conditions.jsonl"):
        _firewall(row, "record_conditions")
        record_id = str(row.get("record_id"))
        condition = str(row.get("condition_id"))
        if record_id != record_order[row_count // len(CONDITION_IDS)] or condition != CONDITION_IDS[row_count % len(CONDITION_IDS)]:
            raise Phase4D4ProtocolBEligibilityVerifierError("record_conditions", "condition canonical drift")
        key = (record_id, condition)
        if key in seen:
            for seed, role in inputs["record_to_roles"][record_id]:
                stats[(seed, role, condition)]["duplicate"] += 1
            continue
        seen.add(key)
        complete = _condition_complete(row)
        for seed, role in inputs["record_to_roles"][record_id]:
            stat = stats[(seed, role, condition)]
            stat["records"].add(record_id)
            stat["wells"].add(inputs["record_to_well"][record_id])
            stat["hasher"].update(raw)
            if complete:
                stat["complete"] += 1
            elif row.get("state") == "not_applicable":
                stat["not_applicable"] += 1
            else:
                stat["runtime"] += 1
        row_count += 1
    if row_count != int(den["mixture_record_count"]) * len(CONDITION_IDS):
        raise Phase4D4ProtocolBEligibilityVerifierError("record_conditions", "row count mismatch")
    role_summaries = []
    role_by_key = {}
    for seed in range(int(den["model_cell_count"])):
        for role in ROLE_ORDER:
            for condition in CONDITION_IDS:
                stat = stats[(seed, role, condition)]
                missing = int(stat["expected_record_count"]) - len(stat["records"])
                state = "complete" if not (missing or stat["duplicate"] or stat["not_applicable"] or stat["runtime"]) else "not_evaluable_coverage"
                row = {
                    "seed": seed,
                    "role": role,
                    "condition_id": condition,
                    "expected_record_count": stat["expected_record_count"],
                    "observed_record_count": len(stat["records"]),
                    "expected_well_count": stat["expected_well_count"],
                    "observed_well_count": len(stat["wells"]),
                    "record_ids_sha256": stat["record_ids_sha256"],
                    "well_ids_sha256": stat["well_ids_sha256"],
                    "parent_condition_rows_sha256": stat["hasher"].hexdigest(),
                    "complete_count": stat["complete"],
                    "not_applicable_count": stat["not_applicable"],
                    "runtime_failure_count": stat["runtime"],
                    "missing_count": missing,
                    "duplicate_count": stat["duplicate"],
                    "state": state,
                    "reason": "complete" if state == "complete" else "incomplete_parent_condition_receipts",
                }
                role_summaries.append(row)
                role_by_key[(seed, role, condition)] = row
    record_index = {record_id: i for i, record_id in enumerate(inputs["record_to_well"])}
    model_rows = []
    for seed in range(int(den["model_cell_count"])):
        for condition in CONDITION_IDS:
            role_sets = [set(inputs["seed_role_record_ids"][f"{seed}:{role}"]) for role in ROLE_ORDER]
            well_sets = [set(inputs["seed_role_well_ids"][f"{seed}:{role}"]) for role in ROLE_ORDER]
            ready = all(role_by_key[(seed, role, condition)]["state"] == "complete" for role in ROLE_ORDER)
            union_records = set().union(*role_sets)
            union_wells = set().union(*well_sets)
            if len(union_records) != int(den["mixture_record_count"]) or len(union_wells) != int(den["mixture_well_count"]):
                ready = False
            all_wells = []
            for role in ROLE_ORDER:
                all_wells.extend(inputs["seed_role_well_ids"][f"{seed}:{role}"])
            model_rows.append(
                {
                    "seed": seed,
                    "condition_id": condition,
                    "role_summary_sha256": {role: _sha256_bytes(_canonical_json_bytes(role_by_key[(seed, role, condition)])) for role in ROLE_ORDER},
                    "union_record_count": len(union_records),
                    "union_well_count": len(union_wells),
                    "union_record_ids_sha256": _ids_digest(tuple(sorted(union_records, key=record_index.get))),
                    "union_well_ids_sha256": _ids_digest(tuple(dict.fromkeys(all_wells))),
                    "state": "ready" if ready else "not_ready",
                    "reason": "ready" if ready else "role_condition_not_complete",
                }
            )
    blank_stats = {condition: {"complete": 0, "failed": 0, "records": set(), "wells": set(), "hasher": hashlib.sha256()} for condition in CONDITION_IDS}
    row_count = 0
    seen_blank = set()
    for raw, row in _read_jsonl(parent / "blank_conditions.jsonl"):
        _firewall(row, "blank_conditions")
        record_id = str(row.get("record_id"))
        condition = str(row.get("condition_id"))
        if condition != CONDITION_IDS[row_count % len(CONDITION_IDS)]:
            raise Phase4D4ProtocolBEligibilityVerifierError("blank_conditions", "condition canonical drift")
        if (record_id, condition) in seen_blank:
            blank_stats[condition]["failed"] += 1
            continue
        seen_blank.add((record_id, condition))
        stat = blank_stats[condition]
        stat["records"].add(record_id)
        stat["wells"].add(str(row.get("well_id", "")))
        stat["hasher"].update(raw)
        if _blank_condition_complete(row):
            stat["complete"] += 1
        else:
            stat["failed"] += 1
        row_count += 1
    blank_rows = []
    for condition in CONDITION_IDS:
        stat = blank_stats[condition]
        state = "complete" if stat["complete"] == int(den["blank_record_count"]) and not stat["failed"] and len(stat["wells"]) == int(den["blank_well_count"]) else "auxiliary_not_evaluable"
        blank_rows.append(
            {
                "condition_id": condition,
                "expected_record_count": int(den["blank_record_count"]),
                "observed_record_count": len(stat["records"]),
                "expected_well_count": int(den["blank_well_count"]),
                "observed_well_count": len(stat["wells"]),
                "future_model_consumer_count": int(den["model_cell_count"]),
                "complete_count": stat["complete"],
                "failure_count": stat["failed"],
                "parent_condition_rows_sha256": stat["hasher"].hexdigest(),
                "state": state,
                "reason": "complete" if state == "complete" else "blank_auxiliary_incomplete",
            }
        )
    ready_count = sum(1 for row in model_rows if row["state"] == "ready")
    role_complete_count = sum(1 for row in role_summaries if row["state"] == "complete")
    primary_ready = ready_count == int(den["model_condition_readiness_count"]) and role_complete_count == int(den["role_condition_summary_count"])
    gate = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "alpha0_ready_model_condition_count": sum(1 for row in model_rows if row["condition_id"] == "alpha0" and row["state"] == "ready"),
        "blank_auxiliary_state": "complete" if all(row["state"] == "complete" for row in blank_rows) else "auxiliary_not_evaluable",
        "full_domain_core": {
            "planned_model_condition_count": int(den["model_condition_readiness_count"]),
            "ready_model_condition_count": ready_count,
            "reason": "all_role_conditions_ready" if primary_ready else "role_condition_not_complete",
            "state": "evaluable" if primary_ready else "not_evaluable_coverage",
        },
        "inherited_rulings": dict(config.get("inherited_rulings", {})),
        "marker_filename": "complete.json" if primary_ready else "failed.json",
        "overall_status": "complete" if primary_ready else "failed",
        "positive_conditions_by_perturbation": {
            pid: {
                "planned_model_condition_count": 5 * len(POSITIVE_ALPHAS),
                "ready_model_condition_count": sum(1 for row in model_rows if row["condition_id"].startswith(f"{pid}:") and row["state"] == "ready"),
                "state": "ready" if all(row["state"] == "ready" for row in model_rows if row["condition_id"].startswith(f"{pid}:")) else "not_ready",
            }
            for pid in ACTIVE_PERTURBATION_IDS
        },
        "role_condition_complete_count": role_complete_count,
        "role_condition_summary_count": int(den["role_condition_summary_count"]),
    }
    for item, name in ((parent_bridge, "parent_bridge"), (role_summaries, "role_condition_summaries"), (model_rows, "model_readiness"), (blank_rows, "blank_summaries"), (gate, "gate")):
        _firewall(item, name)
    return {"parent_bridge": parent_bridge, "role_condition_summaries": role_summaries, "model_condition_readiness": model_rows, "blank_condition_summaries": blank_rows, "gate": gate}


def _run_id(config: Mapping[str, object], inputs: Mapping[str, object]) -> str:
    identity = {
        "blank_role_occurrences_sha256": _sha256_bytes(_jsonl_bytes(inputs["blank_role_occurrences"])),
        "claim_boundary": str(config["claim_boundary"]),
        "condition_ids_sha256": inputs["condition_ids_sha256"],
        "config_sha256": _sha256_bytes(_canonical_json_bytes(config)),
        "folds_sha256": _sha256_bytes(_jsonl_bytes(inputs["well_folds"])),
        "model_cells_sha256": inputs["model_cells_sha256"],
        "mixture_role_occurrences_sha256": inputs["model_role_occurrences_sha256"],
        "parent_sha256sums_sha256": str(config["parent_step27"]["sha256sums_sha256"]),
        "source_record_ids_sha256": inputs["source_record_ids_sha256"],
        "source_well_ids_sha256": inputs["source_well_ids_sha256"],
    }
    return RUN_PREFIX + _sha256_bytes(_canonical_json_bytes(identity))


def _payloads(config: Mapping[str, object], inputs: Mapping[str, object], lift: Mapping[str, object], run_id: str) -> dict[str, bytes]:
    gate = lift["gate"]
    marker = str(gate["marker_filename"])
    status = str(gate["overall_status"])
    manifest = {
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "authority_identities": dict(config.get("authorities", {})),
        "claim_boundary": str(config["claim_boundary"]),
        "code_authority": dict(config.get("code_authority", {})),
        "config_sha256": _sha256_bytes(_canonical_json_bytes(config)),
        "environment_authority": dict(config.get("environment_authority", {})),
        "inherited_rulings": dict(config.get("inherited_rulings", {})),
        "marker_filename": marker,
        "numerical_execution_count": 0,
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "reconstructed_digests": {
            "blank_role_occurrences_sha256": _sha256_bytes(_jsonl_bytes(inputs["blank_role_occurrences"])),
            "condition_ids_sha256": inputs["condition_ids_sha256"],
            "folds_sha256": _sha256_bytes(_jsonl_bytes(inputs["well_folds"])),
            "model_cells_sha256": inputs["model_cells_sha256"],
            "model_role_occurrences_sha256": inputs["model_role_occurrences_sha256"],
            "source_record_ids_sha256": inputs["source_record_ids_sha256"],
            "source_well_ids_sha256": inputs["source_well_ids_sha256"],
        },
        "row_counts": {
            "blank_condition_summaries": len(lift["blank_condition_summaries"]),
            "model_condition_readiness": len(lift["model_condition_readiness"]),
            "role_condition_summaries": len(lift["role_condition_summaries"]),
        },
        "run_id": run_id,
        "status": status,
    }
    payloads = {
        "config.json": _canonical_json_bytes(config),
        "parent_bridge.json": _canonical_json_bytes(lift["parent_bridge"]),
        "role_condition_summaries.jsonl": _jsonl_bytes(lift["role_condition_summaries"]),
        "model_condition_readiness.jsonl": _jsonl_bytes(lift["model_condition_readiness"]),
        "blank_condition_summaries.jsonl": _jsonl_bytes(lift["blank_condition_summaries"]),
        "gate.json": _canonical_json_bytes(gate),
        "manifest.json": _canonical_json_bytes(manifest),
        marker: _canonical_json_bytes({"full_domain_core_state": gate["full_domain_core"]["state"], "marker_filename": marker, "marker_schema": MARKER_SCHEMA_VERSION, "status": status}),
    }
    payloads["SHA256SUMS"] = "".join(f"{_sha256_bytes(payloads[name])}  {name}\n" for name in (*ARTIFACT_PAYLOAD_FILES, marker)).encode()
    return payloads


def verify_phase4_d4_protocol_b_eligibility_from_inputs(
    path: Path,
    *,
    parent_run_path: Path,
    cohort: D4SugarCohort,
    config_path: Path,
) -> Phase4D4ProtocolBEligibilitySummary:
    path = Path(path)
    config = _parse_config(Path(config_path))
    inputs = _reconstruct_inputs(cohort, config)
    expected_run_id = _run_id(config, inputs)
    lift = _build_lift(Path(parent_run_path), inputs, config)
    expected_payloads = _payloads(config, inputs, lift, expected_run_id)
    gate = _json(path / "gate.json")
    manifest = _json(path / "manifest.json")
    marker = str(gate["marker_filename"])
    _assert_artifact_bytes(path, marker)
    for name, expected_bytes in expected_payloads.items():
        observed_bytes = (path / name).read_bytes()
        if observed_bytes != expected_bytes:
            raise Phase4D4ProtocolBEligibilityVerifierError(name, "byte semantic rebuild mismatch")
    run_id = str(manifest["run_id"])
    if run_id != expected_run_id:
        raise Phase4D4ProtocolBEligibilityVerifierError("run_id", "rebuild mismatch")
    return Phase4D4ProtocolBEligibilitySummary(
        path=path,
        run_id=run_id,
        status=str(manifest["status"]),
        marker_filename=marker,
        mixture_record_count=len(cohort.record_ids),
        model_cell_count=len(cohort.splits),
        role_condition_summary_count=_jsonl_count(path / "role_condition_summaries.jsonl"),
        model_condition_readiness_count=_jsonl_count(path / "model_condition_readiness.jsonl"),
    )


def verify_phase4_d4_protocol_b_eligibility(path: Path) -> Phase4D4ProtocolBEligibilitySummary:
    path = Path(path)
    return verify_phase4_d4_protocol_b_eligibility_from_inputs(
        path,
        parent_run_path=ROOT / PARENT_RUN_RELATIVE_PATH,
        cohort=load_d4_sugar_cohort(
            ROOT / PHASE05_D4_PROTOCOL_RELATIVE_PATH,
            ROOT / ARCHIVE_RELATIVE_PATH,
        ),
        config_path=path / "config.json",
    )


__all__ = [
    "Phase4D4ProtocolBEligibilityVerifierError",
    "Phase4D4ProtocolBEligibilitySummary",
    "verify_phase4_d4_protocol_b_eligibility_from_inputs",
    "verify_phase4_d4_protocol_b_eligibility",
]
