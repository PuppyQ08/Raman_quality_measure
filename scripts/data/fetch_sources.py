from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "metadata" / "sources.json"
SCHEMA_VERSION = "rpe-source-registry-v1"
ENTRY_REQUIRED_KEYS = {"artifact_id", "bytes", "path", "sha256", "url"}
ENTRY_OPTIONAL_KEYS = {"license_status", "md5", "source_page"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX32 = re.compile(r"^[0-9a-f]{32}$")
ARTIFACT_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class SourceRegistryError(ValueError):
    """Raised when the source registry is not canonical or safe."""


class DownloadError(RuntimeError):
    """Raised when source download or verification fails."""


def _canonical_bytes(value: object) -> bytes:
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


def _safe_relative_path(value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise SourceRegistryError("entry path must be a non-empty string")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise SourceRegistryError("entry path must be normalized and relative")
    return path


def _validate_entry(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SourceRegistryError("registry entries must be objects")
    keys = set(value)
    if keys - ENTRY_REQUIRED_KEYS - ENTRY_OPTIONAL_KEYS:
        raise SourceRegistryError("registry entry has unknown keys")
    if not ENTRY_REQUIRED_KEYS <= keys:
        raise SourceRegistryError("registry entry is missing required keys")
    artifact_id = value["artifact_id"]
    if not isinstance(artifact_id, str) or not ARTIFACT_ID.fullmatch(artifact_id):
        raise SourceRegistryError("artifact_id is invalid")
    byte_count = value["bytes"]
    if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count <= 0:
        raise SourceRegistryError("bytes must be a positive integer")
    _safe_relative_path(value["path"])
    digest = value["sha256"]
    if not isinstance(digest, str) or not HEX64.fullmatch(digest):
        raise SourceRegistryError("sha256 must be lowercase hexadecimal")
    url = value["url"]
    if not isinstance(url, str) or not url.startswith("https://"):
        raise SourceRegistryError("registry URLs must use HTTPS")
    if "source_page" in value:
        source_page = value["source_page"]
        if not isinstance(source_page, str) or not source_page.startswith("https://"):
            raise SourceRegistryError("source_page must use HTTPS")
    if "license_status" in value and not isinstance(value["license_status"], str):
        raise SourceRegistryError("license_status must be a string")
    if "md5" in value:
        md5 = value["md5"]
        if not isinstance(md5, str) or not HEX32.fullmatch(md5):
            raise SourceRegistryError("md5 must be lowercase hexadecimal")
    return dict(value)


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict[str, object]:
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SourceRegistryError(f"cannot load registry: {error}") from error
    if not isinstance(document, dict) or set(document) != {"groups", "schema_version"}:
        raise SourceRegistryError("registry must contain only groups and schema_version")
    if document["schema_version"] != SCHEMA_VERSION:
        raise SourceRegistryError("unsupported registry schema_version")
    groups = document["groups"]
    if not isinstance(groups, dict) or not groups:
        raise SourceRegistryError("groups must be a non-empty object")
    normalized_groups: dict[str, list[dict[str, object]]] = {}
    seen_paths: set[str] = set()
    for group, entries in groups.items():
        if not isinstance(group, str) or not ARTIFACT_ID.fullmatch(group):
            raise SourceRegistryError("group name is invalid")
        if not isinstance(entries, list) or not entries:
            raise SourceRegistryError("each group must contain entries")
        normalized_entries = [_validate_entry(entry) for entry in entries]
        ids = [str(entry["artifact_id"]) for entry in normalized_entries]
        if len(ids) != len(set(ids)):
            raise SourceRegistryError("artifact_id values must be unique within a group")
        for entry in normalized_entries:
            relative_path = str(entry["path"])
            if relative_path in seen_paths:
                raise SourceRegistryError("entry paths must be globally unique")
            seen_paths.add(relative_path)
        normalized_groups[group] = normalized_entries
    normalized: dict[str, object] = {
        "groups": normalized_groups,
        "schema_version": SCHEMA_VERSION,
    }
    if raw != _canonical_bytes(normalized):
        raise SourceRegistryError("registry must be canonical JSON")
    return normalized


def _digests(path: Path) -> tuple[int, str, str]:
    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    byte_count = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            byte_count += len(chunk)
            sha256.update(chunk)
            md5.update(chunk)
    return byte_count, sha256.hexdigest(), md5.hexdigest()


def verify_artifact(entry: Mapping[str, object], data_root: Path) -> bool:
    target = data_root / _safe_relative_path(entry["path"])
    if not target.is_file():
        return False
    byte_count, sha256, md5 = _digests(target)
    if byte_count != entry["bytes"] or sha256 != entry["sha256"]:
        return False
    return "md5" not in entry or md5 == entry["md5"]


def fetch_artifact(
    entry: Mapping[str, object],
    data_root: Path,
    *,
    timeout_seconds: int = 120,
) -> Path:
    target = data_root / _safe_relative_path(entry["path"])
    if target.exists():
        if verify_artifact(entry, data_root):
            return target
        raise DownloadError(f"existing file failed verification: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    if partial.exists():
        partial.unlink()
    request = urllib.request.Request(
        str(entry["url"]),
        headers={"User-Agent": "raman-spec-bench-source-fetch/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with partial.open("xb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        byte_count, sha256, md5 = _digests(partial)
        if byte_count != entry["bytes"]:
            raise DownloadError(
                f"byte-count mismatch for {entry['artifact_id']}: "
                f"expected {entry['bytes']}, got {byte_count}"
            )
        if sha256 != entry["sha256"]:
            raise DownloadError(f"SHA-256 mismatch for {entry['artifact_id']}")
        if "md5" in entry and md5 != entry["md5"]:
            raise DownloadError(f"MD5 mismatch for {entry['artifact_id']}")
        partial.replace(target)
    except (OSError, ValueError, urllib.error.URLError) as error:
        if partial.exists():
            partial.unlink()
        if isinstance(error, DownloadError):
            raise
        raise DownloadError(f"download failed for {entry['artifact_id']}: {error}") from error
    except DownloadError:
        if partial.exists():
            partial.unlink()
        raise
    return target


def _canonical_line(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def main(argv: Sequence[str] | None = None, *, default_group: str | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "raw")
    if default_group is None:
        parser.add_argument("--group", required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--list", action="store_true")
    actions.add_argument("--verify-only", action="store_true")
    actions.add_argument("--download", action="store_true")
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--accept-source-terms", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.delay_seconds < 0 or args.timeout_seconds <= 0:
        parser.error("delay and timeout values must be positive")
    registry = load_registry(args.registry)
    group = default_group if default_group is not None else args.group
    groups = registry["groups"]
    if group not in groups:
        parser.error(f"unknown source group: {group}")
    entries = list(groups[group])
    if args.artifact:
        requested = set(args.artifact)
        entries = [entry for entry in entries if entry["artifact_id"] in requested]
        found = {str(entry["artifact_id"]) for entry in entries}
        if found != requested:
            parser.error(f"unknown artifact IDs: {sorted(requested - found)}")
    if args.download and not args.accept_source_terms:
        parser.error("--download requires --accept-source-terms")
    failures = 0
    for index, entry in enumerate(entries):
        target = args.data_root / str(entry["path"])
        if args.list:
            result = {
                "artifact_id": entry["artifact_id"],
                "bytes": entry["bytes"],
                "license_status": entry.get("license_status"),
                "path": target.as_posix(),
                "source_page": entry.get("source_page"),
                "url": entry["url"],
            }
        elif args.verify_only:
            valid = verify_artifact(entry, args.data_root)
            failures += not valid
            result = {
                "artifact_id": entry["artifact_id"],
                "path": target.as_posix(),
                "status": "verified" if valid else "missing_or_invalid",
            }
        else:
            try:
                fetched = fetch_artifact(
                    entry,
                    args.data_root,
                    timeout_seconds=args.timeout_seconds,
                )
                result = {
                    "artifact_id": entry["artifact_id"],
                    "path": fetched.as_posix(),
                    "status": "verified",
                }
            except DownloadError as error:
                failures += 1
                result = {
                    "artifact_id": entry["artifact_id"],
                    "error": str(error),
                    "status": "failed",
                }
            if index + 1 < len(entries) and args.delay_seconds:
                time.sleep(args.delay_seconds)
        print(_canonical_line(result))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
