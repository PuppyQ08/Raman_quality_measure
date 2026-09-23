from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from rpe.runner.phase3_dl_audit_authority import CONFIG_BYTES, CONFIG_SHA256


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_IDS = tuple(f"DL{index:02d}" for index in range(1, 11))
OUTPUT_FILES = {
    "config.json", "candidates.jsonl", "source_checks.jsonl",
    "artifact_checks.jsonl", "environment_checks.jsonl",
    "smoke_checks.jsonl", "reproducibility_table.csv", "summary.json",
    "gate.json", "manifest.json", "complete.json", "SHA256SUMS",
}
CSV_FIELDS = (
    "candidate_id", "label", "family_id", "task_role",
    "availability", "publication_disposition",
    "bibliographic_identity_state", "code_state", "weight_state",
    "data_state", "code_license_state", "weight_license_state",
    "data_license_state", "environment_state", "current_host_state",
    "smoke_level", "deepr_reproduction_state", "blocker_codes",
    "evidence_ids",
)
RUNNABLE_SMOKES = {"real_input_repeat_deterministic", "official_example_repeat_deterministic"}
CODE_PATHS = ("rpe/runner/phase3_dl_audit.py", "rpe/runner/phase3_dl_audit_verifier.py", "rpe/runner/phase3_dl_audit_authority.py", "tools/run_phase3_dl_audit.py")
AUTHORITY_PATHS = ("raman_preproc_benchmark_plan_v2.md", "reports/phase0/step01_data_source_audit.md", "reports/phase3/step01_classical_system_registry_design.md", "reports/phase3/step11_dl_reproducibility_audit_design.md", "data/raw/source_versions.json", "data/raw/source_manifest.jsonl", "data/raw/SHA256SUMS", "env/phase3-requirements.lock")


class DlAuditVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class DlAuditVerifiedSummary:
    path: Path
    run_id: str
    candidate_count: int
    runnable_candidate_count: int
    runnable_family_count: int
    parent_minimum_three_runnable_pass: bool


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _identity(root: Path, names: tuple[str, ...]) -> dict[str, dict[str, object]]:
    return {name: {"byte_count": (root / name).stat().st_size, "sha256": _sha(root / name)} for name in names}


def _environment() -> dict[str, object]:
    packages = {}
    for name in ("numpy", "torch"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {"packages": packages, "platform": platform.platform(), "python": platform.python_version()}


def _strip_times(value: object) -> object:
    if isinstance(value, Mapping): return {str(key): _strip_times(item) for key, item in value.items() if not str(key).endswith("_at_utc")}
    if isinstance(value, list): return [_strip_times(item) for item in value]
    return value


def _scientific_config(config: Mapping[str, object]) -> object:
    return _strip_times({key: value for key, value in config.items() if key != "evidence"})


def _json(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes(); value = json.loads(raw)
    if not isinstance(value, Mapping) or raw != _canonical(value): raise DlAuditVerificationError(f"{path.name}: noncanonical")
    return value


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            value = json.loads(raw)
            if not isinstance(value, Mapping) or raw != _canonical(value): raise DlAuditVerificationError(f"{path.name}:{number}: noncanonical")
            rows.append(dict(value))
    return rows


def _derive(candidate: Mapping[str, object], parts: Mapping[str, object]) -> dict[str, object]:
    candidate_id = str(candidate["candidate_id"]); blockers = list(parts["blocker_codes"]); role = str(parts["task_role"]); ok = {"verified", "not_required", "not_required_for_synthetic_generator_smoke"}
    if parts["bibliographic_identity_state"] != "verified": availability = "reported_only"
    elif role != "spectral_preprocessor": availability = "excluded"
    elif parts["code_state"] not in {"verified", "not_required"}: availability = "code_missing"
    elif parts["weight_state"] not in ok: availability = "weights_missing"
    elif parts["data_state"] not in ok: availability = "data_missing"
    elif any(parts[key] in {"restricted", "forbidden"} for key in ("code_license_state", "weight_license_state", "data_license_state")): availability = "excluded"
    elif any(parts[key] not in ok for key in ("code_license_state", "weight_license_state", "data_license_state")): availability = "reported_only"
    elif parts["environment_state"] != "constructible" or parts["current_host_state"] not in {"compatible", "cpu_compatible", "gpu_available"}: availability = "dependency_missing"
    elif parts["smoke_level"] not in RUNNABLE_SMOKES: availability = "implementation_missing"
    else: availability = "runnable"
    disposition = "executable_candidate" if availability == "runnable" else ("excluded" if availability == "excluded" else "reported_only")
    if candidate_id != "DL01": deeper = "not_applicable"
    elif parts["data_state"] not in ok or "official_paired_data_missing" in blockers: deeper = "not_evaluable_data_missing"
    elif parts.get("deepr_endpoint_reproduction_verified") is True: deeper = "reproduced"
    else: deeper = "not_evaluated_smoke_only"
    return {
        "availability": availability, "bibliographic_identity_state": parts["bibliographic_identity_state"], "blocker_codes": blockers,
        "candidate_id": candidate_id, "code_license_state": parts["code_license_state"], "code_state": parts["code_state"],
        "current_host_state": parts["current_host_state"], "data_license_state": parts["data_license_state"], "data_state": parts["data_state"],
        "deepr_reproduction_state": deeper, "environment_state": parts["environment_state"], "evidence_ids": list(parts["evidence_ids"]),
        "family_id": candidate["family_id"], "label": candidate["label"], "publication_disposition": disposition,
        "smoke_level": parts["smoke_level"], "task_role": role, "weight_license_state": parts["weight_license_state"], "weight_state": parts["weight_state"],
    }


def _table(rows: list[Mapping[str, object]]) -> bytes:
    stream = io.StringIO(newline=""); writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n"); writer.writeheader()
    for row in rows:
        value = {key: row[key] for key in CSV_FIELDS}; value["blocker_codes"] = ";".join(row["blocker_codes"]); value["evidence_ids"] = ";".join(row["evidence_ids"]); writer.writerow(value)
    return stream.getvalue().encode()


def verify_phase3_dl_audit(path: Path, *, project_root: Path | None = None) -> DlAuditVerifiedSummary:
    path = Path(path); root = ROOT if project_root is None else Path(project_root); actual = {value.name for value in path.iterdir() if value.is_file()}
    if actual != OUTPUT_FILES: raise DlAuditVerificationError("artifact inventory mismatch")
    checks = (path / "SHA256SUMS").read_text(); expected_lines = []
    for line in checks.splitlines():
        digest, name = line.split("  ", 1)
        if name == "SHA256SUMS" or name not in OUTPUT_FILES or _sha(path / name) != digest: raise DlAuditVerificationError(f"checksum mismatch: {name}")
        expected_lines.append((name, digest))
    if checks != "".join(f"{digest}  {name}\n" for name, digest in sorted(expected_lines)) or len(expected_lines) != 11: raise DlAuditVerificationError("checksum ledger mismatch")
    config_raw=(path / "config.json").read_bytes()
    config = _json(path / "config.json")
    if config.get("fixture") is not True and (len(config_raw) != CONFIG_BYTES or _sha_bytes(config_raw) != CONFIG_SHA256): raise DlAuditVerificationError("frozen config identity mismatch")
    candidates = _jsonl(path / "candidates.jsonl")
    if tuple(row.get("candidate_id") for row in candidates) != EXPECTED_IDS: raise DlAuditVerificationError("candidate roster mismatch")
    configured_candidates = config.get("candidates")
    if configured_candidates is not None and candidates != configured_candidates:
        raise DlAuditVerificationError("candidate roster mismatch with config")
    kinds = {name: _jsonl(path / f"{name}.jsonl") for name in ("source_checks", "artifact_checks", "environment_checks", "smoke_checks")}
    for name, rows in kinds.items():
        if tuple(row.get("candidate_id") for row in rows) != EXPECTED_IDS: raise DlAuditVerificationError(f"{name}: roster mismatch")
    configured_evidence = config.get("evidence")
    if isinstance(configured_evidence, Mapping):
        for name, identity in configured_evidence.items():
            if not isinstance(identity, Mapping): raise DlAuditVerificationError(f"config evidence identity malformed: {name}")
            source = root / str(identity["path"]); artifact = path / f"{name}.jsonl"
            if source.stat().st_size != int(identity["byte_count"]) or _sha(source) != identity["sha256"] or source.read_bytes() != artifact.read_bytes():
                raise DlAuditVerificationError(f"config evidence identity mismatch: {name}")
    derived=[]
    for index, candidate in enumerate(candidates):
        merged={}
        for rows in kinds.values(): merged.update(rows[index])
        evidence_ids=merged.get("evidence_ids")
        if not isinstance(evidence_ids,list) or not evidence_ids: raise DlAuditVerificationError("positive claim lacks evidence_ids")
        derived.append(_derive(candidate, merged))
    if (path / "reproducibility_table.csv").read_bytes() != _table(derived): raise DlAuditVerificationError("reproducibility table mismatch")
    runnable=[row for row in derived if row["availability"] == "runnable"]; families=len({str(row["family_id"]) for row in runnable})
    gate={"audit_complete":True,"deepr_reproduction_route_state":derived[0]["deepr_reproduction_state"],"excluded_count":sum(row["publication_disposition"]=="excluded" for row in derived),"parent_minimum_three_runnable_pass":families>=3,"reported_only_count":sum(row["publication_disposition"]=="reported_only" for row in derived),"runnable_candidate_count":len(runnable),"runnable_family_count":families}
    if _json(path / "gate.json") != gate: raise DlAuditVerificationError("gate mismatch")
    summary=_json(path / "summary.json")
    availability_counts={state:sum(row["availability"]==state for row in derived) for state in sorted({str(row["availability"]) for row in derived})}
    if summary != {"availability_counts":availability_counts,"candidate_count":10,"gate":gate}: raise DlAuditVerificationError("summary mismatch")
    manifest=_json(path / "manifest.json"); marker=_json(path / "complete.json")
    evidence_rows=[]
    for index, candidate_id in enumerate(EXPECTED_IDS):
        merged={"candidate_id":candidate_id}
        for rows in kinds.values(): merged.update({key:value for key,value in rows[index].items() if key != "candidate_id"})
        evidence_rows.append(merged)
    run_identity={"authority_identity":_identity(root,AUTHORITY_PATHS),"claim_boundary":str(config.get("claim_boundary")),"code_identity":_identity(root,CODE_PATHS),"environment":_environment(),"scientific_input_identity":{"config":_sha_bytes(_canonical(_scientific_config(config))),"candidates":_sha_bytes(_canonical(_strip_times(candidates))),"evidence":_sha_bytes(_canonical(_strip_times(evidence_rows)))}}
    run_id=_sha_bytes(b"rpe-phase3-dl-reproducibility-audit-v1\0"+_canonical(run_identity))
    if path.name != f"phase3-dl-audit-{run_id}": raise DlAuditVerificationError("run path identity mismatch")
    input_identity={name:{"byte_count":(path/name).stat().st_size,"sha256":_sha(path/name)} for name in ("config.json","candidates.jsonl","source_checks.jsonl","artifact_checks.jsonl","environment_checks.jsonl","smoke_checks.jsonl")}
    expected_manifest={"artifact_schema_version":"phase3-dl-reproducibility-audit-artifact-v1","audit_complete":True,"candidate_count":10,"claim_boundary":str(config.get("claim_boundary")),"environment":run_identity["environment"],"input_file_identity":input_identity,"run_id":run_id,"run_identity":run_identity,"status":"complete"}
    if manifest != expected_manifest or marker != {"audit_complete":True,"run_id":run_id,"status":"complete"}: raise DlAuditVerificationError("terminal identity mismatch")
    return DlAuditVerifiedSummary(path,str(manifest["run_id"]),10,len(runnable),families,families>=3)


__all__ = ["DlAuditVerificationError", "DlAuditVerifiedSummary", "verify_phase3_dl_audit"]
