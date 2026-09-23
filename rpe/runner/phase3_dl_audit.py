from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rpe.runner.phase3_dl_audit_authority import CONFIG_BYTES, CONFIG_SHA256


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase3/configs/dl_reproducibility_audit_v1.json"
SCHEMA_VERSION = "phase3-dl-reproducibility-audit-config-v1"
EXPERIMENT_ID = "phase3-dl-reproducibility-audit-v1"
ARTIFACT_SCHEMA_VERSION = "phase3-dl-reproducibility-audit-artifact-v1"
RUN_DOMAIN = b"rpe-phase3-dl-reproducibility-audit-v1\0"
EXPECTED_IDS = tuple(f"DL{index:02d}" for index in range(1, 11))
AVAILABILITY = {
    "runnable", "dependency_missing", "implementation_missing",
    "code_missing", "weights_missing", "data_missing",
    "reported_only", "excluded",
}
RUNNABLE_SMOKES = {"real_input_repeat_deterministic", "official_example_repeat_deterministic"}
OUTPUT_FILES = (
    "config.json", "candidates.jsonl", "source_checks.jsonl",
    "artifact_checks.jsonl", "environment_checks.jsonl",
    "smoke_checks.jsonl", "reproducibility_table.csv", "summary.json",
    "gate.json", "manifest.json", "complete.json",
)
CODE_PATHS = (
    "rpe/runner/phase3_dl_audit.py",
    "rpe/runner/phase3_dl_audit_verifier.py",
    "rpe/runner/phase3_dl_audit_authority.py",
    "tools/run_phase3_dl_audit.py",
)
AUTHORITY_PATHS = (
    "raman_preproc_benchmark_plan_v2.md",
    "reports/phase0/step01_data_source_audit.md",
    "reports/phase3/step01_classical_system_registry_design.md",
    "reports/phase3/step11_dl_reproducibility_audit_design.md",
    "data/raw/source_versions.json",
    "data/raw/source_manifest.jsonl",
    "data/raw/SHA256SUMS",
    "env/phase3-requirements.lock",
)
CSV_FIELDS = (
    "candidate_id", "label", "family_id", "task_role",
    "availability", "publication_disposition",
    "bibliographic_identity_state", "code_state", "weight_state",
    "data_state", "code_license_state", "weight_license_state",
    "data_license_state", "environment_state", "current_host_state",
    "smoke_level", "deepr_reproduction_state", "blocker_codes",
    "evidence_ids",
)


class DlAuditError(ValueError):
    pass


@dataclass(frozen=True)
class DlAuditSummary:
    path: Path
    run_id: str
    candidate_count: int
    runnable_candidate_count: int
    runnable_family_count: int
    parent_minimum_three_runnable_pass: bool


@dataclass(frozen=True)
class DlAuditConfig:
    path: Path
    byte_count: int
    sha256: str
    candidate_ids: tuple[str, ...]
    minimum_runnable_family_count: int
    authorities: Mapping[str, Mapping[str, object]]
    evidence: Mapping[str, Mapping[str, object]]
    document: Mapping[str, object]


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(dict(row)) for row in rows)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strings(value: object, label: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise DlAuditError(f"{label}: must be {'a' if allow_empty else 'a nonempty'} list")
    if any(not isinstance(item, str) or not item for item in value):
        raise DlAuditError(f"{label}: must contain nonempty strings")
    if len(set(value)) != len(value):
        raise DlAuditError(f"{label}: contains duplicates")
    return list(value)


def _nonempty_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise DlAuditError(f"{row.get('candidate_id', '<unknown>')}.{key}: must be nonempty")
    return value


def load_phase3_dl_audit_config(path: Path, *, project_root: Path = ROOT) -> DlAuditConfig:
    path = Path(path); raw = path.read_bytes(); document = json.loads(raw)
    if not isinstance(document, Mapping) or raw != _canonical(document):
        raise DlAuditError("config: must be canonical JSON object")
    if path.resolve() == DEFAULT_CONFIG.resolve() and (len(raw) != CONFIG_BYTES or _sha_bytes(raw) != CONFIG_SHA256):
        raise DlAuditError("config: frozen identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise DlAuditError("config: schema or experiment mismatch")
    ids = tuple(_strings(document.get("candidate_ids"), "candidate_ids"))
    if ids != EXPECTED_IDS: raise DlAuditError("candidate roster: must be exact ordered DL01-DL10")
    minimum = document.get("minimum_runnable_family_count")
    if minimum != 3: raise DlAuditError("minimum_runnable_family_count: must equal 3")
    root = Path(project_root); authorities = document.get("authorities"); evidence = document.get("evidence")
    if not isinstance(authorities, Mapping) or not isinstance(evidence, Mapping): raise DlAuditError("config: authorities/evidence missing")
    for group, values in (("authorities", authorities), ("evidence", evidence)):
        for name, identity in values.items():
            if not isinstance(identity, Mapping): raise DlAuditError(f"config.{group}.{name}: malformed")
            target=root/str(identity.get("path"))
            if not target.is_file() or target.stat().st_size != int(identity.get("byte_count", -1)) or _sha_file(target) != identity.get("sha256"):
                raise DlAuditError(f"config.{group}.{name}: identity mismatch")
    return DlAuditConfig(path,len(raw),_sha_bytes(raw),ids,minimum,dict(authorities),dict(evidence),dict(document))


def _positive_claim_has_evidence(row: Mapping[str, object]) -> None:
    positive = any(
        row.get(key) == "verified"
        for key in (
            "bibliographic_identity_state", "code_state", "weight_state",
            "data_state", "code_license_state", "weight_license_state",
            "data_license_state",
        )
    )
    evidence_ids = _strings(row.get("evidence_ids"), "evidence_ids", allow_empty=True)
    if positive and not evidence_ids:
        raise DlAuditError(f"{row.get('candidate_id', '<unknown>')}.evidence_ids: positive claims require evidence_ids")


def derive_dl_candidate_audit(candidate: Mapping[str, object], evidence: Mapping[str, object]) -> dict[str, object]:
    candidate_id = _nonempty_text(candidate, "candidate_id")
    if evidence.get("candidate_id") != candidate_id:
        raise DlAuditError(f"{candidate_id}: evidence candidate mismatch")
    label = _nonempty_text(candidate, "label")
    family_id = _nonempty_text(candidate, "family_id")
    _strings(candidate.get("variants"), f"{candidate_id}.variants")
    required = (
        "bibliographic_identity_state", "task_role", "code_state",
        "weight_state", "data_state", "code_license_state",
        "weight_license_state", "data_license_state", "environment_state",
        "current_host_state", "smoke_level",
    )
    values = {key: _nonempty_text(evidence, key) for key in required}
    blockers = _strings(evidence.get("blocker_codes"), f"{candidate_id}.blocker_codes", allow_empty=True)
    evidence_ids = _strings(evidence.get("evidence_ids"), f"{candidate_id}.evidence_ids", allow_empty=True)
    _positive_claim_has_evidence(evidence)

    role = values["task_role"]
    verified_or_unused = {"verified", "not_required", "not_required_for_synthetic_generator_smoke"}
    if values["bibliographic_identity_state"] != "verified":
        availability = "reported_only"
    elif role != "spectral_preprocessor":
        availability = "excluded"
    elif values["code_state"] not in {"verified", "not_required"}:
        availability = "code_missing"
    elif values["weight_state"] not in verified_or_unused:
        availability = "weights_missing"
    elif values["data_state"] not in verified_or_unused:
        availability = "data_missing"
    elif any(values[key] in {"restricted", "forbidden"} for key in ("code_license_state", "weight_license_state", "data_license_state")):
        availability = "excluded"
    elif any(values[key] not in verified_or_unused for key in ("code_license_state", "weight_license_state", "data_license_state")):
        availability = "reported_only"
    elif values["environment_state"] != "constructible" or values["current_host_state"] not in {"compatible", "cpu_compatible", "gpu_available"}:
        availability = "dependency_missing"
    elif values["smoke_level"] not in RUNNABLE_SMOKES:
        availability = "implementation_missing"
    else:
        availability = "runnable"
    if availability not in AVAILABILITY:
        raise AssertionError(availability)
    disposition = "executable_candidate" if availability == "runnable" else ("excluded" if availability == "excluded" else "reported_only")
    if candidate_id != "DL01":
        deeper = "not_applicable"
    elif values["data_state"] not in verified_or_unused or "official_paired_data_missing" in blockers:
        deeper = "not_evaluable_data_missing"
    elif evidence.get("deepr_endpoint_reproduction_verified") is True:
        deeper = "reproduced"
    else:
        deeper = "not_evaluated_smoke_only"
    return {
        "availability": availability, "bibliographic_identity_state": values["bibliographic_identity_state"],
        "blocker_codes": blockers, "candidate_id": candidate_id, "code_license_state": values["code_license_state"],
        "code_state": values["code_state"], "current_host_state": values["current_host_state"],
        "data_license_state": values["data_license_state"], "data_state": values["data_state"],
        "deepr_reproduction_state": deeper, "environment_state": values["environment_state"],
        "evidence_ids": evidence_ids, "family_id": family_id, "label": label,
        "publication_disposition": disposition, "smoke_level": values["smoke_level"],
        "task_role": role, "weight_license_state": values["weight_license_state"],
        "weight_state": values["weight_state"],
    }


def _projection(row: Mapping[str, object], keys: Sequence[str]) -> dict[str, object]:
    result = {key: row[key] for key in keys}
    for key in ("source_observations", "artifact_observations", "environment_observations", "smoke_observations"):
        if key in row and key in keys:
            result[key] = row[key]
    return result


def _csv_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        output = {key: row[key] for key in CSV_FIELDS}
        output["blocker_codes"] = ";".join(row["blocker_codes"])
        output["evidence_ids"] = ";".join(row["evidence_ids"])
        writer.writerow(output)
    return stream.getvalue().encode("utf-8")


def _code_identity(root: Path) -> dict[str, dict[str, object]]:
    result = {}
    for name in CODE_PATHS:
        path = root / name
        result[name] = {"byte_count": path.stat().st_size, "sha256": _sha_file(path)}
    return result


def _file_identity(root: Path, names: Sequence[str]) -> dict[str, dict[str, object]]:
    return {
        name: {"byte_count": (root / name).stat().st_size, "sha256": _sha_file(root / name)}
        for name in names
    }


def _environment() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "torch"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"packages": packages, "platform": platform.platform(), "python": platform.python_version()}


def _strip_observation_times(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _strip_observation_times(item) for key, item in value.items() if not str(key).endswith("_at_utc")}
    if isinstance(value, list):
        return [_strip_observation_times(item) for item in value]
    return value


def _scientific_config(config: Mapping[str, object]) -> object:
    return _strip_observation_times(
        {key: value for key, value in config.items() if key != "evidence"}
    )


def _artifact_payloads(*, config: Mapping[str, object], candidates: Sequence[Mapping[str, object]], evidence_rows: Sequence[Mapping[str, object]], root: Path) -> tuple[str, dict[str, bytes], DlAuditSummary]:
    if config.get("schema_version") != SCHEMA_VERSION or config.get("experiment_id") != EXPERIMENT_ID:
        raise DlAuditError("config: schema or experiment mismatch")
    ids = tuple(config.get("candidate_ids", ()))
    candidate_ids = tuple(value.get("candidate_id") for value in candidates)
    evidence_ids = tuple(value.get("candidate_id") for value in evidence_rows)
    if ids != EXPECTED_IDS or candidate_ids != EXPECTED_IDS or evidence_ids != EXPECTED_IDS:
        raise DlAuditError("candidate roster: must be exact ordered DL01-DL10")
    if len({str(value.get("family_id")) for value in candidates}) != len(candidates):
        raise DlAuditError("candidate roster: family IDs must be unique")
    derived = [derive_dl_candidate_audit(candidate, evidence) for candidate, evidence in zip(candidates, evidence_rows, strict=True)]
    source_keys = ("candidate_id", "bibliographic_identity_state", "task_role", "evidence_ids", "source_observations")
    artifact_keys = ("candidate_id", "code_state", "weight_state", "data_state", "code_license_state", "weight_license_state", "data_license_state", "artifact_observations")
    env_keys = ("candidate_id", "environment_state", "current_host_state", "environment_observations")
    smoke_keys = ("candidate_id", "smoke_level", "blocker_codes", "smoke_observations")
    def project(keys: Sequence[str]) -> list[dict[str, object]]:
        return [{key: row[key] for key in keys if key in row} for row in evidence_rows]
    base = {
        "config.json": _canonical(dict(config)),
        "candidates.jsonl": _jsonl(candidates),
        "source_checks.jsonl": _jsonl(project(source_keys)),
        "artifact_checks.jsonl": _jsonl(project(artifact_keys)),
        "environment_checks.jsonl": _jsonl(project(env_keys)),
        "smoke_checks.jsonl": _jsonl(project(smoke_keys)),
        "reproducibility_table.csv": _csv_bytes(derived),
    }
    runnable = [row for row in derived if row["availability"] == "runnable"]
    family_count = len({str(row["family_id"]) for row in runnable})
    minimum = int(config.get("minimum_runnable_family_count", -1))
    if minimum != 3:
        raise DlAuditError("minimum_runnable_family_count: must equal 3")
    gate = {
        "audit_complete": True, "deepr_reproduction_route_state": derived[0]["deepr_reproduction_state"],
        "excluded_count": sum(row["publication_disposition"] == "excluded" for row in derived),
        "parent_minimum_three_runnable_pass": family_count >= minimum,
        "reported_only_count": sum(row["publication_disposition"] == "reported_only" for row in derived),
        "runnable_candidate_count": len(runnable), "runnable_family_count": family_count,
    }
    summary = {
        "availability_counts": dict(sorted({state: sum(row["availability"] == state for row in derived) for state in AVAILABILITY if any(row["availability"] == state for row in derived)}.items())),
        "candidate_count": len(derived), "gate": gate,
    }
    base["summary.json"] = _canonical(summary); base["gate.json"] = _canonical(gate)
    input_identity = {name: {"byte_count": len(payload), "sha256": _sha_bytes(payload)} for name, payload in base.items() if name.endswith((".json", ".jsonl")) and name not in {"summary.json", "gate.json"}}
    scientific_inputs = {
        "config": _sha_bytes(_canonical(_scientific_config(config))),
        "candidates": _sha_bytes(_canonical(_strip_observation_times(list(candidates)))),
        "evidence": _sha_bytes(_canonical(_strip_observation_times(list(evidence_rows)))),
    }
    environment = _environment()
    run_identity = {
        "authority_identity": _file_identity(root, AUTHORITY_PATHS),
        "claim_boundary": str(config.get("claim_boundary")),
        "code_identity": _code_identity(root),
        "environment": environment,
        "scientific_input_identity": scientific_inputs,
    }
    run_id = _sha_bytes(RUN_DOMAIN + _canonical(run_identity))
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION, "audit_complete": True,
        "candidate_count": len(derived), "claim_boundary": str(config.get("claim_boundary")),
        "environment": environment, "input_file_identity": input_identity,
        "run_id": run_id, "run_identity": run_identity, "status": "complete",
    }
    base["manifest.json"] = _canonical(manifest)
    base["complete.json"] = _canonical({"audit_complete": True, "run_id": run_id, "status": "complete"})
    result = DlAuditSummary(Path(), run_id, len(derived), len(runnable), family_count, family_count >= minimum)
    return run_id, base, result


def build_phase3_dl_audit_from_documents(output_root: Path, *, config: Mapping[str, object], candidates: Sequence[Mapping[str, object]], evidence_rows: Sequence[Mapping[str, object]], project_root: Path = ROOT) -> DlAuditSummary:
    run_id, payloads, summary = _artifact_payloads(config=config, candidates=candidates, evidence_rows=evidence_rows, root=Path(project_root))
    path = Path(output_root) / f"phase3-dl-audit-{run_id}"
    if path.exists():
        raise DlAuditError(f"output: run path already exists: {path}")
    path.mkdir(parents=True)
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    checks = "".join(f"{_sha_file(path / name)}  {name}\n" for name in sorted(payloads))
    (path / "SHA256SUMS").write_text(checks, encoding="utf-8")
    return DlAuditSummary(path, summary.run_id, summary.candidate_count, summary.runnable_candidate_count, summary.runnable_family_count, summary.parent_minimum_three_runnable_pass)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            value = json.loads(raw)
            if not isinstance(value, Mapping) or raw != _canonical(value):
                raise DlAuditError(f"{path.name}:{number}: must be canonical JSON object")
            rows.append(dict(value))
    return rows


def build_phase3_dl_audit(output_root: Path, *, config_path: Path = DEFAULT_CONFIG, project_root: Path = ROOT) -> DlAuditSummary:
    parsed=load_phase3_dl_audit_config(config_path,project_root=project_root); config=parsed.document
    evidence = config.get("evidence")
    candidates = config.get("candidates")
    if not isinstance(evidence, Mapping) or not isinstance(candidates, list):
        raise DlAuditError("config: candidates/evidence contract missing")
    ledgers = {}
    for kind in ("source_checks", "artifact_checks", "environment_checks", "smoke_checks"):
        identity = evidence.get(kind)
        if not isinstance(identity, Mapping): raise DlAuditError(f"config.evidence.{kind}: missing")
        path = Path(project_root) / str(identity.get("path"))
        if path.stat().st_size != int(identity.get("byte_count", -1)) or _sha_file(path) != identity.get("sha256"):
            raise DlAuditError(f"config.evidence.{kind}: identity mismatch")
        ledgers[kind] = _load_jsonl(path)
    by_kind = {kind: {str(row["candidate_id"]): row for row in rows} for kind, rows in ledgers.items()}
    combined = []
    for candidate_id in EXPECTED_IDS:
        row = {"candidate_id": candidate_id}
        for kind in ("source_checks", "artifact_checks", "environment_checks", "smoke_checks"):
            if candidate_id not in by_kind[kind]: raise DlAuditError(f"{kind}: missing {candidate_id}")
            row.update({key: value for key, value in by_kind[kind][candidate_id].items() if key != "candidate_id"})
        combined.append(row)
    return build_phase3_dl_audit_from_documents(output_root, config=config, candidates=candidates, evidence_rows=combined, project_root=project_root)


__all__ = [
    "DlAuditConfig", "DlAuditError", "DlAuditSummary",
    "build_phase3_dl_audit", "build_phase3_dl_audit_from_documents",
    "derive_dl_candidate_audit", "load_phase3_dl_audit_config",
]
