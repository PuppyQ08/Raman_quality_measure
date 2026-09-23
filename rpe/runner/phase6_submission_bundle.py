"""Phase 6 Step 11 deterministic submission bundle runner."""
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
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/submission_bundle_v1.json"
SCHEMA_VERSION = "phase6-submission-bundle-v1"
FORBIDDEN_PATH_RE = re.compile(
    r"(^|/)(raw|restricted|bacteria|rruff|non_authoritative[^/]*)(/|_|$)"
)
MARKDOWN_LINK_RE = re.compile(r"(?<!\!)\[[^\]]+\]\(([^)]+)\)")
CITATION_RE = re.compile(r"@([A-Za-z0-9_:-]+)")
BIBTEX_KEY_RE = re.compile(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,")


class SubmissionBundleError(ValueError):
    """The Step 11 source bundle cannot be assembled under the frozen rules."""


@dataclass(frozen=True)
class SubmissionBundleSummary:
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


def repo_relative_path(path: Path, project_root: Path = ROOT) -> str:
    resolved_project_root = Path(project_root).resolve()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(resolved_project_root).as_posix()
    except ValueError as error:
        raise SubmissionBundleError(f"path is outside project root: {path}") from error


def _load_config(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    document = json.loads(raw)
    if raw != _canonical(document):
        raise SubmissionBundleError("config must be canonical JSON")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise SubmissionBundleError("unexpected Step 11 config schema")
    source_mode = document.get("source_mode")
    if source_mode not in {"synthetic_fixture", "formal"}:
        raise SubmissionBundleError("source_mode must be synthetic_fixture or formal")
    return document


def _relative_posix(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SubmissionBundleError(f"{field} must be a non-empty relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise SubmissionBundleError(
            f"{field} must stay inside its declared root and cannot reference forbidden or restricted paths"
        )
    normalized = pure.as_posix()
    if FORBIDDEN_PATH_RE.search(normalized.lower()) is not None:
        raise SubmissionBundleError(f"{field} matches a forbidden or restricted path pattern")
    return normalized


def _bundle_posix(value: str, *, field: str) -> str:
    normalized = _relative_posix(value, field=field)
    if normalized == "SHA256SUMS":
        raise SubmissionBundleError(f"{field} may not shadow SHA256SUMS")
    return normalized


def _parse_sha256sums(path: Path) -> dict[str, str]:
    ledger: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise SubmissionBundleError(f"{path}: invalid SHA256SUMS row")
        if parts[1] in ledger:
            raise SubmissionBundleError(f"{path}: duplicate SHA256SUMS row")
        ledger[parts[1]] = parts[0]
    return ledger


def _validate_parent(parent_name: str, parent_config: Mapping[str, object], project_root: Path) -> dict[str, object]:
    parent_path_value = parent_config.get("path")
    if not isinstance(parent_path_value, str) or not parent_path_value:
        raise SubmissionBundleError(f"{parent_name}.path must be set explicitly")
    if Path(parent_path_value).is_absolute():
        raise SubmissionBundleError(f"{parent_name}.path must be relative to project_root")
    root = (project_root / parent_path_value).resolve()
    if not root.is_dir():
        raise SubmissionBundleError(f"{parent_name} parent path does not exist: {parent_path_value}")
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise SubmissionBundleError(f"{parent_name} SHA256SUMS is missing")
    inventory = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file()
    }
    terminal = str(parent_config.get("required_terminal", "complete.json"))
    if terminal not in inventory:
        raise SubmissionBundleError(f"{parent_name} terminal marker mismatch")
    terminal_document = json.loads((root / terminal).read_text(encoding="utf-8"))
    if "complete" not in str(terminal_document.get("status", "")):
        raise SubmissionBundleError(f"{parent_name} terminal marker is not complete")
    ledger = _parse_sha256sums(checksum_path)
    covered = inventory - {"SHA256SUMS"}
    if set(ledger) != covered:
        raise SubmissionBundleError(f"{parent_name} ledger inventory mismatch")
    for relative, digest in ledger.items():
        if _sha_file(root / relative) != digest:
            raise SubmissionBundleError(f"{parent_name} ledger checksum mismatch: {relative}")
    required_payloads = parent_config.get("required_payloads", ())
    if not isinstance(required_payloads, list) or not all(isinstance(value, str) for value in required_payloads):
        raise SubmissionBundleError(f"{parent_name}.required_payloads must be a string list")
    missing = [value for value in required_payloads if value not in inventory]
    if missing:
        raise SubmissionBundleError(f"{parent_name} required payload mismatch: {missing}")
    return {
        "name": parent_name,
        "root": root,
        "path": parent_path_value,
        "terminal": terminal,
        "inventory": tuple(sorted(inventory)),
        "sha256sums_sha256": _sha_file(checksum_path),
    }


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
    normalized_bundle = _bundle_posix(payload_path, field="claim payload_path")
    if normalized_bundle in bundle_payloads:
        return normalized_bundle
    if Path(payload_path).is_absolute():
        raise SubmissionBundleError("claim payload_path must be relative")
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
        raise SubmissionBundleError(f"claim payload_path is outside declared parent authorities: {payload_path}")
    mapping = _bundle_source_to_target(config)
    if source_key not in mapping:
        raise SubmissionBundleError(f"claim payload_path is not mapped into the bundle: {payload_path}")
    return _bundle_posix(mapping[source_key], field="claim payload_path")


def _collect_bundle_payloads(
    config: Mapping[str, object],
    parents: Mapping[str, Mapping[str, object]],
) -> dict[str, bytes]:
    bundle_files = config.get("bundle_files")
    if not isinstance(bundle_files, list):
        raise SubmissionBundleError("bundle_files must be a list")
    payloads: dict[str, bytes] = {}
    for index, entry in enumerate(bundle_files):
        if not isinstance(entry, Mapping):
            raise SubmissionBundleError(f"bundle_files[{index}] must be an object")
        parent_name = entry.get("parent")
        if parent_name not in parents:
            raise SubmissionBundleError(f"bundle_files[{index}].parent is unknown")
        source_path = _relative_posix(str(entry.get("source_path", "")), field=f"bundle_files[{index}].source_path")
        bundle_path = _bundle_posix(str(entry.get("bundle_path", "")), field=f"bundle_files[{index}].bundle_path")
        source_file = Path(parents[parent_name]["root"]) / source_path
        if not source_file.is_file():
            raise SubmissionBundleError(f"bundle_files[{index}] source file is missing")
        if bundle_path in payloads:
            raise SubmissionBundleError(f"bundle_files[{index}] duplicates bundle path {bundle_path}")
        payloads[bundle_path] = source_file.read_bytes()
    return payloads


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
        raise SubmissionBundleError("references.bib does not define any BibTeX citation keys")
    return keys


def _parse_selector(selector: str) -> dict[str, str]:
    if not selector:
        raise SubmissionBundleError("claim row_selector must be non-empty")
    if selector.startswith("{"):
        try:
            document = json.loads(selector)
        except json.JSONDecodeError as error:
            raise SubmissionBundleError("claim row_selector JSON is invalid") from error
        if (
            not isinstance(document, dict)
            or not document
            or any(not isinstance(key, str) or not key or value == "" for key, value in document.items())
        ):
            raise SubmissionBundleError("claim row_selector JSON must be a non-empty object")
        return {key: str(value) for key, value in document.items()}
    parsed: dict[str, str] = {}
    for item in selector.split(";"):
        if "=" not in item:
            raise SubmissionBundleError("claim row_selector must use key=value selectors")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or value == "":
            raise SubmissionBundleError("claim row_selector contains an empty key or value")
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
        raise SubmissionBundleError("claim row_selector does not resolve to exactly one row")
    return matches[0]


def _validate_manuscript_links(payloads: Mapping[str, bytes]) -> None:
    manuscript_path = "paper/manuscript.md"
    text = payloads[manuscript_path].decode("utf-8")
    for target in _markdown_links(text):
        normalized = posixpath.normpath(
            PurePosixPath(manuscript_path).parent.joinpath(target).as_posix()
        )
        if normalized not in payloads:
            raise SubmissionBundleError(f"Markdown link target is missing from bundle: {target}")


def _validate_citations(payloads: Mapping[str, bytes]) -> tuple[str, ...]:
    manuscript = payloads["paper/manuscript.md"].decode("utf-8")
    references = payloads["paper/references.bib"].decode("utf-8")
    cited = set(_citation_keys_from_text(manuscript))
    defined = set(_citation_keys_from_bibtex(references))
    missing = sorted(cited - defined)
    if missing:
        raise SubmissionBundleError(f"BibTeX citation keys are unresolved: {missing}")
    return tuple(sorted(cited))


def _validate_phase05_boundary(payloads: Mapping[str, bytes]) -> None:
    text = payloads["paper/manuscript.md"].decode("utf-8").lower()
    preserved_failure = re.search(
        r"(?:phase 0\.5\s+)?original(?:\s+phase 0\.5)?\s+confirmatory gate"
        r"\s+(?:(?:was|is|remained|at)\s+)?(?:not_met|not\s+met)\b",
        text,
    )
    if preserved_failure is None:
        raise SubmissionBundleError("Phase 0.5 wording must preserve the original confirmatory gate not_met boundary")
    if "internal screening" not in text:
        raise SubmissionBundleError("Phase 0.5 wording must mention the internal screening as separate")
    if re.search(r"confirmatory gate\s+(?:was\s+)?(?:passed|met)\b", text):
        raise SubmissionBundleError("Phase 0.5 wording crosses the confirmatory gate/internal screening boundary")


def _validate_claims_matrix(
    payloads: Mapping[str, bytes],
    project_root: Path,
    config: Mapping[str, object],
    parents: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    rows = list(csv.DictReader(payloads["paper/claims_matrix.csv"].decode("utf-8").splitlines()))
    required = {
        "claim_id",
        "manuscript_section",
        "claim_text",
        "evidence_class",
        "authority_path",
        "authority_sha256",
        "payload_path",
        "row_selector",
        "allowed_scope",
        "prohibited_scope",
        "verification_state",
    }
    if not rows:
        raise SubmissionBundleError("claims_matrix.csv must contain at least one claim")
    if set(rows[0]) != required:
        raise SubmissionBundleError("claims_matrix.csv header mismatch")
    claim_map: list[dict[str, object]] = []
    for row in rows:
        authority_path = row["authority_path"]
        if Path(authority_path).is_absolute():
            raise SubmissionBundleError("claim authority_path must be relative")
        authority = (project_root / authority_path).resolve()
        if authority.is_file():
            authority_digest = _sha_file(authority)
        elif authority.is_dir() and authority in {
            Path(parent["root"]).resolve() for parent in parents.values()
        }:
            ledger = authority / "SHA256SUMS"
            if not ledger.is_file():
                raise SubmissionBundleError(f"claim artifact authority has no checksum ledger: {authority_path}")
            authority_digest = _sha_file(ledger)
        else:
            raise SubmissionBundleError(f"claim authority is missing: {authority_path}")
        if authority_digest != row["authority_sha256"]:
            raise SubmissionBundleError(f"claim authority digest mismatch: {authority_path}")
        payload_path = _resolve_claim_payload_path(
            row["payload_path"],
            project_root=project_root,
            config=config,
            parents=parents,
            bundle_payloads=payloads,
        )
        if payload_path not in payloads:
            raise SubmissionBundleError(f"claim payload_path is missing from bundle: {payload_path}")
        selector = _parse_selector(row["row_selector"])
        matched = _select_csv_row(payloads[payload_path], selector)
        if not row["allowed_scope"] or not row["prohibited_scope"]:
            raise SubmissionBundleError("claim scope fields must be non-empty")
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


def _render_environment_receipt(project_root: Path) -> dict[str, object]:
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


def _render_limitations_ledger() -> dict[str, object]:
    return {
        "license_gate": "pending_owner_license_selection",
        "redistribution_gate": "pending_owner_review_rruff_and_bacteria",
        "renderer_state": "deferred_pending_locked_renderer_and_venue_choice",
        "venue_state": "pending_owner_choice",
        "submission_state": "not_submitted_pending_owner_authorization",
        "leaderboard_state": "deferred_no_redistributable_hidden_gt",
    }


def _write_payloads(target: Path, payloads: Mapping[str, bytes]) -> None:
    for relative, payload in payloads.items():
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)


def _sha256sums_bytes(names: Iterable[str], payloads: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{_sha_bytes(payloads[name])}  {name}\n"
        for name in names
    ).encode("utf-8")


def build_phase6_submission_bundle(
    output_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    project_root: Path = ROOT,
) -> SubmissionBundleSummary:
    config_path = Path(config_path)
    project_root = Path(project_root)
    config = _load_config(config_path)
    if config["source_mode"] == "formal":
        for name in ("step4_final_figures", "step8", "step9", "step10"):
            parent_info = dict(config.get("parents", {})).get(name)
            if not isinstance(parent_info, Mapping):
                raise SubmissionBundleError(f"formal build requires explicit {name} parent path")
    parents = {
        name: _validate_parent(name, value, project_root)
        for name, value in dict(config.get("parents", {})).items()
    }
    required_parents = {"step8", "step9", "step10"}
    if config["source_mode"] == "formal":
        required_parents = {"step4_final_figures", "step8", "step9", "step10"}
    if set(parents) != required_parents:
        raise SubmissionBundleError("Step 11 requires explicit verified upstream parent paths")
    payloads = _collect_bundle_payloads(config, parents)
    _validate_manuscript_links(payloads)
    cited_keys = _validate_citations(payloads)
    _validate_phase05_boundary(payloads)
    claim_map = _validate_claims_matrix(payloads, project_root, config, parents)
    generated = {
        "config.json": _canonical(config),
        "preflight.json": _canonical(
            {
                "schema_version": SCHEMA_VERSION,
                "source_mode": config["source_mode"],
                "parents": {
                    name: {
                        "path": value["path"],
                        "terminal": value["terminal"],
                        "sha256sums_sha256": value["sha256sums_sha256"],
                    }
                    for name, value in sorted(parents.items())
                },
                "bundle_file_count": len(payloads),
            }
        ),
        "metadata/environment_receipt.json": _canonical(_render_environment_receipt(project_root)),
        "metadata/limitations_ledger.json": _canonical(_render_limitations_ledger()),
        "metadata/artifact_claim_map.json": _canonical(
            {
                "claims": claim_map,
                "citation_keys": list(cited_keys),
            }
        ),
    }
    payloads = dict(payloads)
    payloads.update(generated)
    payload_names = tuple(config["artifact_contract"]["payload_files"])
    expected_payloads = set(payload_names)
    if set(payloads) != expected_payloads - {"manifest.json"}:
        raise SubmissionBundleError("artifact payload inventory does not match the frozen contract")
    config_sha256 = _sha_bytes(payloads["config.json"])
    source_digests = {
        name: _sha_bytes(payloads[name])
        for name in sorted(payloads)
        if name not in {"manifest.json"}
    }
    run_id = hashlib.sha256(
        _canonical(
            {
                "schema_version": SCHEMA_VERSION,
                "config_sha256": config_sha256,
                "payload_sha256": source_digests,
            }
        )
    ).hexdigest()
    manifest = {
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
        "bundle_sha256": source_digests,
    }
    payloads["manifest.json"] = _canonical(manifest)
    complete = _canonical({"status": "complete", "run_id": run_id})
    all_files = dict(payloads)
    all_files["complete.json"] = complete
    all_files["SHA256SUMS"] = _sha256sums_bytes(sorted(all_files), all_files)
    target = Path(output_root) / run_id
    if target.exists():
        raise SubmissionBundleError("output collision: target run directory already exists")
    _write_payloads(target, all_files)
    return SubmissionBundleSummary(target, run_id, "complete", len(payload_names))
