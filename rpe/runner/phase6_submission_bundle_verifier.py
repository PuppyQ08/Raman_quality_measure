"""Independent verifier for Phase 6 Step 11 deterministic submission bundles."""
from __future__ import annotations

import csv
import hashlib
import json
import posixpath
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "phase6-submission-bundle-v1"
FORBIDDEN_PATH_RE = re.compile(
    r"(^|/)(raw|restricted|bacteria|rruff|non_authoritative[^/]*)(/|_|$)"
)
MARKDOWN_LINK_RE = re.compile(r"(?<!\!)\[[^\]]+\]\(([^)]+)\)")
CITATION_RE = re.compile(r"@([A-Za-z0-9_:-]+)")
BIBTEX_KEY_RE = re.compile(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,")


class SubmissionBundleVerificationError(ValueError):
    """The bundle fails the independent byte-rebuild contract."""


@dataclass(frozen=True)
class SubmissionBundleVerificationSummary:
    path: Path
    run_id: str
    status: str
    payload_count: int


def _canonical(value: object) -> bytes:
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


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_sha256sums_bytes(payload: bytes) -> dict[str, str]:
    ledger: dict[str, str] = {}
    for line in payload.decode("utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise SubmissionBundleVerificationError("invalid SHA256SUMS content")
        if parts[1] in ledger:
            raise SubmissionBundleVerificationError("duplicate SHA256SUMS entry")
        ledger[parts[1]] = parts[0]
    return ledger


def _relative_posix(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SubmissionBundleVerificationError(f"{field} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise SubmissionBundleVerificationError(f"{field} escapes the frozen root")
    normalized = pure.as_posix()
    if FORBIDDEN_PATH_RE.search(normalized.lower()) is not None:
        raise SubmissionBundleVerificationError(f"{field} matches a forbidden path pattern")
    return normalized


def _load_bundle(path: Path) -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    files = {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in path.rglob("*")
        if item.is_file()
    }
    for required in ("config.json", "manifest.json", "complete.json", "SHA256SUMS"):
        if required not in files:
            raise SubmissionBundleVerificationError(f"bundle inventory is missing {required}")
    config = json.loads(files["config.json"])
    manifest = json.loads(files["manifest.json"])
    if files["config.json"] != _canonical(config):
        raise SubmissionBundleVerificationError("config.json is not canonical")
    if files["manifest.json"] != _canonical(manifest):
        raise SubmissionBundleVerificationError("manifest.json is not canonical")
    if config.get("schema_version") != SCHEMA_VERSION:
        raise SubmissionBundleVerificationError("unexpected bundle schema version")
    if manifest.get("artifact_schema_version") != SCHEMA_VERSION:
        raise SubmissionBundleVerificationError("unexpected manifest schema version")
    return files, config, manifest


def _validate_inventory(files: Mapping[str, bytes], config: Mapping[str, object]) -> tuple[str, ...]:
    payloads = tuple(config["artifact_contract"]["payload_files"])
    expected = set(payloads) | {"complete.json", "SHA256SUMS"}
    if set(files) != expected:
        raise SubmissionBundleVerificationError("bundle inventory mismatch")
    ledger = _parse_sha256sums_bytes(files["SHA256SUMS"])
    if set(ledger) != expected - {"SHA256SUMS"}:
        raise SubmissionBundleVerificationError("SHA256SUMS inventory mismatch")
    for relative, digest in ledger.items():
        if _sha_bytes(files[relative]) != digest:
            raise SubmissionBundleVerificationError(f"SHA256SUMS mismatch for {relative}")
    return payloads


def _validate_parent(parent_name: str, parent_path: str, project_root: Path) -> dict[str, object]:
    if Path(parent_path).is_absolute():
        raise SubmissionBundleVerificationError(f"{parent_name} parent path must be relative")
    root = (project_root / parent_path).resolve()
    if not root.is_dir():
        raise SubmissionBundleVerificationError(f"{parent_name} parent path does not exist")
    inventory = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    if "complete.json" not in inventory or "SHA256SUMS" not in inventory:
        raise SubmissionBundleVerificationError(f"{parent_name} parent inventory is incomplete")
    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    if "complete" not in str(complete.get("status", "")):
        raise SubmissionBundleVerificationError(f"{parent_name} parent terminal mismatch")
    ledger = _parse_sha256sums_bytes((root / "SHA256SUMS").read_bytes())
    if set(ledger) != inventory - {"SHA256SUMS"}:
        raise SubmissionBundleVerificationError(f"{parent_name} parent ledger inventory mismatch")
    for relative, digest in ledger.items():
        if _sha_file(root / relative) != digest:
            raise SubmissionBundleVerificationError(f"{parent_name} parent ledger checksum mismatch: {relative}")
    return {"root": root, "path": parent_path, "sha256sums_sha256": _sha_file(root / "SHA256SUMS")}


def _collect_payloads(config: Mapping[str, object], parents: Mapping[str, Mapping[str, object]]) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for index, entry in enumerate(config["bundle_files"]):
        parent_name = entry["parent"]
        if parent_name not in parents:
            raise SubmissionBundleVerificationError(f"bundle_files[{index}] uses an unknown parent")
        source_path = _relative_posix(entry["source_path"], field=f"bundle_files[{index}].source_path")
        bundle_path = _relative_posix(entry["bundle_path"], field=f"bundle_files[{index}].bundle_path")
        source_file = Path(parents[parent_name]["root"]) / source_path
        if not source_file.is_file():
            raise SubmissionBundleVerificationError(f"bundle_files[{index}] source payload is missing")
        if bundle_path in payloads:
            raise SubmissionBundleVerificationError(f"duplicate bundle payload path {bundle_path}")
        payloads[bundle_path] = source_file.read_bytes()
    return payloads


def _bundle_source_to_target(config: Mapping[str, object]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for entry in config.get("bundle_files", ()):
        if not isinstance(entry, Mapping):
            continue
        parent_name = entry.get("parent")
        source_path = entry.get("source_path")
        bundle_path = entry.get("bundle_path")
        if isinstance(parent_name, str) and isinstance(source_path, str) and isinstance(bundle_path, str):
            mapping[f"{parent_name}:{source_path}"] = bundle_path
    return mapping


def _resolve_claim_payload_path(
    payload_path: str,
    *,
    project_root: Path,
    config: Mapping[str, object],
    parents: Mapping[str, Mapping[str, object]],
    bundle_payloads: Mapping[str, bytes],
) -> str:
    normalized_bundle = _relative_posix(payload_path, field="claim payload_path")
    if normalized_bundle in bundle_payloads:
        return normalized_bundle
    if Path(payload_path).is_absolute():
        raise SubmissionBundleVerificationError("claim payload_path must be relative")
    resolved = (project_root / payload_path).resolve()
    source_key = None
    for parent_name, parent_info in parents.items():
        root = Path(parent_info["root"]).resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            continue
        source_key = f"{parent_name}:{relative}"
        break
    if source_key is None:
        raise SubmissionBundleVerificationError(
            f"claim payload_path is outside declared parent authorities: {payload_path}"
        )
    mapping = _bundle_source_to_target(config)
    if source_key not in mapping:
        raise SubmissionBundleVerificationError(
            f"claim payload_path is not mapped into the bundle: {payload_path}"
        )
    return _relative_posix(mapping[source_key], field="claim payload_path")


def _markdown_links(text: str) -> tuple[str, ...]:
    links = []
    for raw in MARKDOWN_LINK_RE.findall(text):
        value = raw.strip()
        if value.startswith(("http://", "https://", "mailto:", "#")):
            continue
        links.append(value)
    return tuple(links)


def _citation_keys_from_text(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(CITATION_RE.findall(text))))


def _citation_keys_from_bibtex(text: str) -> tuple[str, ...]:
    keys = tuple(sorted(set(BIBTEX_KEY_RE.findall(text))))
    if not keys:
        raise SubmissionBundleVerificationError("references.bib does not define any BibTeX key")
    return keys


def _validate_markdown_links(payloads: Mapping[str, bytes]) -> None:
    manuscript_path = "paper/manuscript.md"
    text = payloads[manuscript_path].decode("utf-8")
    for target in _markdown_links(text):
        normalized = posixpath.normpath(
            PurePosixPath(manuscript_path).parent.joinpath(target).as_posix()
        )
        if normalized not in payloads:
            raise SubmissionBundleVerificationError(f"Markdown link target is missing: {target}")


def _validate_citations(payloads: Mapping[str, bytes]) -> tuple[str, ...]:
    manuscript = payloads["paper/manuscript.md"].decode("utf-8")
    references = payloads["paper/references.bib"].decode("utf-8")
    cited = set(_citation_keys_from_text(manuscript))
    defined = set(_citation_keys_from_bibtex(references))
    missing = sorted(cited - defined)
    if missing:
        raise SubmissionBundleVerificationError(f"unresolved BibTeX citation keys: {missing}")
    return tuple(sorted(cited))


def _validate_phase05_boundary(payloads: Mapping[str, bytes]) -> None:
    text = payloads["paper/manuscript.md"].decode("utf-8").lower()
    preserved_failure = re.search(
        r"(?:phase 0\.5\s+)?original(?:\s+phase 0\.5)?\s+confirmatory gate"
        r"\s+(?:(?:was|is|remained|at)\s+)?(?:not_met|not\s+met)\b",
        text,
    )
    if preserved_failure is None:
        raise SubmissionBundleVerificationError("Phase 0.5 confirmatory gate not_met wording is missing")
    if "internal screening" not in text:
        raise SubmissionBundleVerificationError("Phase 0.5 internal screening boundary is missing")
    if re.search(r"confirmatory gate\s+(?:was\s+)?(?:passed|met)\b", text):
        raise SubmissionBundleVerificationError("Phase 0.5 wording crosses the confirmatory/internal-screening boundary")


def _parse_selector(selector: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not selector:
        raise SubmissionBundleVerificationError("claim row_selector must be non-empty")
    if selector.startswith("{"):
        try:
            document = json.loads(selector)
        except json.JSONDecodeError as error:
            raise SubmissionBundleVerificationError("claim row_selector JSON is invalid") from error
        if (
            not isinstance(document, dict)
            or not document
            or any(not isinstance(key, str) or not key or value == "" for key, value in document.items())
        ):
            raise SubmissionBundleVerificationError("claim row_selector JSON must be a non-empty object")
        return {key: str(value) for key, value in document.items()}
    for item in selector.split(";"):
        if "=" not in item:
            raise SubmissionBundleVerificationError("claim row_selector must use key=value selectors")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or value == "":
            raise SubmissionBundleVerificationError("claim row_selector has an empty key or value")
        parsed[key] = value
    return parsed


def _select_csv_row(payload: bytes, selector: Mapping[str, str]) -> dict[str, str]:
    rows = list(csv.DictReader(payload.decode("utf-8").splitlines()))
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == value for key, value in selector.items())
    ]
    if len(matches) != 1:
        raise SubmissionBundleVerificationError("claim row_selector does not resolve to exactly one row")
    return matches[0]


def _validate_claims(
    payloads: Mapping[str, bytes],
    project_root: Path,
    config: Mapping[str, object],
    parents: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = list(csv.DictReader(payloads["paper/claims_matrix.csv"].decode("utf-8").splitlines()))
    claim_map: list[dict[str, object]] = []
    for row in rows:
        authority_path = row["authority_path"]
        if Path(authority_path).is_absolute():
            raise SubmissionBundleVerificationError("claim authority_path must be relative")
        authority = (project_root / authority_path).resolve()
        if authority.is_file():
            authority_digest = _sha_file(authority)
        elif authority.is_dir() and authority in {
            Path(parent["root"]).resolve() for parent in parents.values()
        }:
            ledger = authority / "SHA256SUMS"
            if not ledger.is_file():
                raise SubmissionBundleVerificationError("claim artifact authority has no checksum ledger")
            authority_digest = _sha_file(ledger)
        else:
            raise SubmissionBundleVerificationError(f"claim authority is missing: {authority_path}")
        if authority_digest != row["authority_sha256"]:
            raise SubmissionBundleVerificationError(f"claim authority digest mismatch: {authority_path}")
        payload_path = _resolve_claim_payload_path(
            row["payload_path"],
            project_root=project_root,
            config=config,
            parents=parents,
            bundle_payloads=payloads,
        )
        if payload_path not in payloads:
            raise SubmissionBundleVerificationError(f"claim payload_path is missing from bundle: {payload_path}")
        selector = _parse_selector(row["row_selector"])
        matched = _select_csv_row(payloads[payload_path], selector)
        if not row["allowed_scope"] or not row["prohibited_scope"]:
            raise SubmissionBundleVerificationError("claim scope fields must be non-empty")
        claim_map.append(
            {
                "claim_id": row["claim_id"],
                "payload_path": payload_path,
                "row_selector": selector,
                "authority_path": authority_path,
                "matched_row": matched,
            }
        )
    return claim_map


def _environment_receipt(project_root: Path) -> dict[str, object]:
    locks = []
    for relative in ("env/requirements.lock", "env/phase3-requirements.lock"):
        current = project_root / relative
        if current.is_file():
            locks.append(
                {
                    "path": relative,
                    "byte_count": current.stat().st_size,
                    "sha256": _sha_file(current),
                }
            )
    return {
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "nvidia_driver_visible": False,
        "lockfiles": locks,
    }


def _limitations() -> dict[str, object]:
    return {
        "license_gate": "pending_owner_license_selection",
        "redistribution_gate": "pending_owner_review_rruff_and_bacteria",
        "renderer_state": "deferred_pending_locked_renderer_and_venue_choice",
        "venue_state": "pending_owner_choice",
        "submission_state": "not_submitted_pending_owner_authorization",
        "leaderboard_state": "deferred_no_redistributable_hidden_gt",
    }


def _sha256sums_bytes(names: Iterable[str], payloads: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{_sha_bytes(payloads[name])}  {name}\n"
        for name in names
    ).encode("utf-8")


def verify_phase6_submission_bundle(
    run_path: Path,
    *,
    project_root: Path = ROOT,
) -> SubmissionBundleVerificationSummary:
    run_path = Path(run_path)
    project_root = Path(project_root)
    files, config, manifest = _load_bundle(run_path)
    payload_names = _validate_inventory(files, config)
    parents = {
        name: _validate_parent(name, parent_path, project_root)
        for name, parent_path in sorted(dict(manifest["parent_paths"]).items())
    }
    required_parents = {"step8", "step9", "step10"}
    if config["source_mode"] == "formal":
        required_parents = {"step4_final_figures", "step8", "step9", "step10"}
    if set(parents) != required_parents:
        raise SubmissionBundleVerificationError("parent path set mismatch")
    payloads = _collect_payloads(config, parents)
    _validate_markdown_links(payloads)
    cited = _validate_citations(payloads)
    _validate_phase05_boundary(payloads)
    claim_map = _validate_claims(payloads, project_root, config, parents)
    payloads = dict(payloads)
    payloads["config.json"] = _canonical(config)
    payloads["preflight.json"] = _canonical(
        {
            "schema_version": SCHEMA_VERSION,
            "source_mode": config["source_mode"],
            "parents": {
                name: {
                    "path": value["path"],
                    "terminal": "complete.json",
                    "sha256sums_sha256": value["sha256sums_sha256"],
                }
                for name, value in sorted(parents.items())
            },
            "bundle_file_count": len([name for name in payloads if name not in {"config.json", "preflight.json"}]),
        }
    )
    payloads["metadata/environment_receipt.json"] = _canonical(_environment_receipt(project_root))
    payloads["metadata/limitations_ledger.json"] = _canonical(_limitations())
    payloads["metadata/artifact_claim_map.json"] = _canonical(
        {"claims": claim_map, "citation_keys": list(cited)}
    )
    if set(payloads) != set(payload_names) - {"manifest.json"}:
        raise SubmissionBundleVerificationError("rebuilt payload inventory mismatch")
    config_sha256 = _sha_bytes(payloads["config.json"])
    payload_sha256 = {
        name: _sha_bytes(payloads[name])
        for name in sorted(payloads)
        if name != "manifest.json"
    }
    run_id = hashlib.sha256(
        _canonical(
            {
                "schema_version": SCHEMA_VERSION,
                "config_sha256": config_sha256,
                "payload_sha256": payload_sha256,
            }
        )
    ).hexdigest()
    rebuilt_manifest = _canonical(
        {
            "artifact_schema_version": SCHEMA_VERSION,
            "bundle_name": config["bundle_name"],
            "run_id": run_id,
            "source_mode": config["source_mode"],
            "archive_format": config["artifact_contract"]["archive_format"],
            "payload_files": list(payload_names),
            "parent_paths": {
                name: value["path"]
                for name, value in sorted(parents.items())
            },
            "bundle_sha256": payload_sha256,
        }
    )
    payloads["manifest.json"] = rebuilt_manifest
    payloads["complete.json"] = _canonical({"status": "complete", "run_id": run_id})
    payloads["SHA256SUMS"] = _sha256sums_bytes(sorted(payloads), payloads)
    for name, rebuilt in payloads.items():
        current = files.get(name)
        if current != rebuilt:
            raise SubmissionBundleVerificationError(f"rebuilt payload mismatch: {name}")
    if manifest["run_id"] != run_id or run_path.name != run_id:
        raise SubmissionBundleVerificationError("run_id mismatch")
    return SubmissionBundleVerificationSummary(run_path, run_id, "complete", len(payload_names))
