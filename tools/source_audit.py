from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


VALID_STATUSES = {"verified", "failed", "blocked"}


def _require(record: dict[str, Any], key: str) -> None:
    if record.get(key) in (None, "", [], {}):
        raise ValueError(f"{record.get('artifact_id', '<unknown>')}: {key} is required")


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("source", "artifact_id", "status"):
        _require(record, key)
    if record["status"] not in VALID_STATUSES:
        raise ValueError(f"{record['artifact_id']}: unsupported status {record['status']!r}")
    if record["status"] == "verified":
        for key in ("local_path", "bytes", "sha256", "checks"):
            _require(record, key)
        if len(record["sha256"]) != 64:
            raise ValueError(f"{record['artifact_id']}: sha256 must contain 64 hex characters")
    else:
        _require(record, "error")
    return record


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_file(
    source: str,
    artifact_id: str,
    path: Path,
    source_url: str,
) -> dict[str, Any]:
    path = Path(path)
    checks = ["nonempty"]
    format_details: dict[str, Any] = {}
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"{artifact_id}: file is missing or empty: {path}")

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise ValueError(f"{artifact_id}: ZIP CRC failed for {corrupt_member}")
            format_details["member_count"] = len(archive.infolist())
            format_details["uncompressed_bytes"] = sum(
                member.file_size for member in archive.infolist()
            )
        checks.append("zip_crc_ok")

    return validate_record(
        {
            "source": source,
            "artifact_id": artifact_id,
            "status": "verified",
            "source_url": source_url,
            "local_path": path.as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "checks": checks,
            "format_details": format_details,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
    )


def validate_dataset_arrays(
    spectra: Any,
    targets: Any,
    wavenumbers: Any,
) -> dict[str, Any]:
    spectra_array = np.asarray(spectra)
    targets_array = np.asarray(targets)
    wavenumbers_array = np.asarray(wavenumbers)
    if spectra_array.dtype == object:
        try:
            spectra_array = spectra_array.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("spectra must be numeric") from error
    if wavenumbers_array.dtype == object:
        try:
            wavenumbers_array = wavenumbers_array.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("wavenumbers must be numeric") from error

    if spectra_array.ndim != 2 or spectra_array.shape[0] == 0 or spectra_array.shape[1] == 0:
        raise ValueError(f"spectra must be a nonempty 2D array, got {spectra_array.shape}")
    if targets_array.ndim not in (1, 2):
        raise ValueError(f"targets must be 1D or 2D, got {targets_array.shape}")
    if wavenumbers_array.ndim != 1 or wavenumbers_array.shape[0] == 0:
        raise ValueError(f"wavenumbers must be a nonempty 1D array, got {wavenumbers_array.shape}")
    if spectra_array.shape[0] != targets_array.shape[0]:
        raise ValueError(
            f"spectra rows ({spectra_array.shape[0]}) do not match "
            f"targets rows ({targets_array.shape[0]})"
        )
    if spectra_array.shape[1] != wavenumbers_array.shape[0]:
        raise ValueError(
            f"spectra columns ({spectra_array.shape[1]}) do not match "
            f"wavenumbers length ({wavenumbers_array.shape[0]})"
        )
    if not np.isfinite(spectra_array).all():
        raise ValueError("spectra contain non-finite values")
    if not np.isfinite(wavenumbers_array).all():
        raise ValueError("wavenumbers contain non-finite values")

    return {
        "spectra_shape": list(spectra_array.shape),
        "targets_shape": list(targets_array.shape),
        "wavenumbers_shape": list(wavenumbers_array.shape),
        "spectra_dtype": str(spectra_array.dtype),
        "wavenumbers_dtype": str(wavenumbers_array.dtype),
        "finite_spectra": True,
        "finite_wavenumbers": True,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    checked = [validate_record(dict(record)) for record in records]
    checked.sort(key=lambda record: (record["source"], record["artifact_id"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as output:
        for record in checked:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            output.write("\n")
    temporary_path.replace(path)
