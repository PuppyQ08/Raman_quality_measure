"""Deterministic Phase 6 Step 9 release-metadata artifact builder."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import mlcroissant as mlc
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase6/configs/release_metadata_v1.json"
RUN_DOMAIN = b"rpe-phase6-release-metadata-v1\0"


class ReleaseMetadataError(ValueError):
    """The frozen Step 9 contract is violated."""


@dataclass(frozen=True)
class ReleaseMetadataConfig:
    path: Path
    byte_count: int
    sha256: str
    schema_version: str
    document: Mapping[str, object]


@dataclass(frozen=True)
class ReleaseMetadataSummary:
    path: Path
    run_id: str
    status: str
    release_row_count: int


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _csv_bytes(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> bytes:
    lines = [",".join(fields) + "\n"]
    for row in rows:
        values: list[str] = []
        for field in fields:
            value = row.get(field)
            text = "" if value is None else str(value)
            if any(token in text for token in [",", "\"", "\n"]):
                text = "\"" + text.replace("\"", "\"\"") + "\""
            values.append(text)
        lines.append(",".join(values) + "\n")
    return "".join(lines).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_canonical(row) for row in rows)


def _sha256sums(payloads: Mapping[str, bytes]) -> bytes:
    return "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {name}\n"
        for name, payload in sorted(payloads.items())
    ).encode("utf-8")


def _parse_sha256sums(path: Path, required_names: set[str]) -> Mapping[str, str]:
    lines = (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    listed: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or parts[1] in listed:
            raise ReleaseMetadataError("invalid Step8 SHA256SUMS ledger")
        listed[parts[1]] = parts[0]
    actual_names = {
        item.name
        for item in path.iterdir()
        if item.is_file() and item.name != "SHA256SUMS"
    }
    if set(listed) != actual_names:
        raise ReleaseMetadataError("Step8 SHA256SUMS inventory mismatch")
    if not required_names.issubset(listed):
        raise ReleaseMetadataError("Step8 SHA256SUMS required payload mismatch")
    for name, digest in listed.items():
        if _sha_file(path / name) != digest:
            raise ReleaseMetadataError(f"Step8 SHA256SUMS mismatch: {name}")
    return MappingProxyType(listed)


def _code_authority() -> Mapping[str, Mapping[str, object]]:
    names = (
        "rpe/runner/phase6_release_metadata.py",
        "rpe/runner/phase6_release_metadata_verifier.py",
        "tools/run_phase6_release_metadata.py",
        "experiments/phase6/configs/release_metadata_v1.json",
    )
    return MappingProxyType(
        {
            name: MappingProxyType(
                {
                    "byte_count": int((ROOT / name).stat().st_size),
                    "sha256": _sha_file(ROOT / name),
                }
            )
            for name in names
            if (ROOT / name).is_file()
        }
    )


def _artifact_contract(config: ReleaseMetadataConfig) -> tuple[str, ...]:
    payloads = tuple(config.document["artifact_contract"]["payload_files"])
    expected = (
        "config.json",
        "authority_bridge.json",
        "preflight.json",
        "release_matrix.csv",
        "data_card.md",
        "metadata_index.parquet",
        "croissant.json",
        "limitations.jsonl",
        "environment.json",
        "manifest.json",
    )
    if payloads != expected:
        raise ReleaseMetadataError("artifact_contract: payload inventory mismatch")
    return payloads


def load_phase6_release_metadata_config(path: Path = DEFAULT_CONFIG) -> ReleaseMetadataConfig:
    raw = Path(path).read_bytes()
    document = json.loads(raw)
    if raw != _canonical(document):
        raise ReleaseMetadataError("config must be canonical JSON")
    if document.get("schema_version") != "phase6-release-metadata-v1":
        raise ReleaseMetadataError("unexpected config schema")
    _artifact_contract(
        ReleaseMetadataConfig(
            path=Path(path),
            byte_count=len(raw),
            sha256=_sha_bytes(raw),
            schema_version=str(document["schema_version"]),
            document=_freeze(document),
        )
    )
    return ReleaseMetadataConfig(
        path=Path(path),
        byte_count=len(raw),
        sha256=_sha_bytes(raw),
        schema_version=str(document["schema_version"]),
        document=_freeze(document),
    )


def _source_manifest_rows() -> Mapping[str, Mapping[str, object]]:
    path = ROOT / "data/raw/source_manifest.jsonl"
    rows: dict[str, Mapping[str, object]] = {}
    for line in path.read_bytes().splitlines():
        row = json.loads(line)
        rows[str(row["artifact_id"])] = MappingProxyType(row)
    return MappingProxyType(rows)


def _release_matrix_rows(config: ReleaseMetadataConfig, *, mode: str) -> tuple[Mapping[str, object], ...]:
    source_manifest = _source_manifest_rows()
    rows = (
        {
            "artifact_id": "rruff_record_level_derivatives",
            "title": "RRUFF record-level derivatives",
            "relative_path": "data/unified/rruff_raman_raw",
            "media_type": "application/x-directory",
            "disposition": "local_only_pending_redistribution_review",
            "source_kind": "local_restricted_derivative",
            "record_count": 5244,
            "byte_count": int((ROOT / "data/unified/rruff_raman_raw/arrays.h5").stat().st_size),
            "sha256": _sha_file(ROOT / "data/unified/rruff_raman_raw/arrays.h5"),
            "repository_license": "pending_owner_license_selection",
            "source_license": "redistribution_unconfirmed",
            "reason": "RRUFF redistribution terms remain unconfirmed for record-level derivatives.",
        },
        {
            "artifact_id": "bacteria_id_records",
            "title": "Bacteria-ID source records",
            "relative_path": str(source_manifest["official_dropbox_zip"]["local_path"]),
            "media_type": "application/zip",
            "disposition": "external_reference_only",
            "source_kind": "upstream_reference",
            "record_count": int(source_manifest["official_dropbox_zip"]["format_details"]["total_samples"]),
            "byte_count": int(source_manifest["official_dropbox_zip"]["bytes"]),
            "sha256": str(source_manifest["official_dropbox_zip"]["sha256"]),
            "repository_license": "pending_owner_license_selection",
            "source_license": "redistribution_requires_review",
            "reason": "Bacteria-ID redistribution still requires review; no record bytes are distributed.",
        },
        {
            "artifact_id": "sugar_source_evidence",
            "title": "Sugar mixtures source evidence",
            "relative_path": str(source_manifest["sugar_mixtures_low_snr"]["local_path"]),
            "media_type": "application/json",
            "disposition": "external_reference_only",
            "source_kind": "upstream_reference",
            "record_count": int(source_manifest["sugar_mixtures_low_snr"]["format_details"]["spectra_shape"][0])
            + int(source_manifest["sugar_mixtures_high_snr"]["format_details"]["spectra_shape"][0]),
            "byte_count": int(source_manifest["sugar_mixtures_low_snr"]["bytes"])
            + int(source_manifest["sugar_mixtures_high_snr"]["bytes"]),
            "sha256": _sha_bytes(
                (
                    str(source_manifest["sugar_mixtures_low_snr"]["sha256"])
                    + "|"
                    + str(source_manifest["sugar_mixtures_high_snr"]["sha256"])
                ).encode("utf-8")
            ),
            "repository_license": "pending_owner_license_selection",
            "source_license": "cc_by_4_0_source_evidence_only",
            "reason": "Sugar source evidence is CC BY 4.0, but derived artifact terms remain separate.",
        },
        {
            "artifact_id": "release_matrix_csv",
            "title": "Release matrix",
            "relative_path": "release_matrix.csv",
            "media_type": "text/csv",
            "disposition": "metadata_and_rebuild_instructions_only",
            "source_kind": "generated_metadata",
            "record_count": 6,
            "byte_count": None,
            "sha256": None,
            "repository_license": "pending_owner_license_selection",
            "source_license": "pending_owner_license_selection",
            "reason": "Generated release metadata remains pending owner license selection.",
        },
        {
            "artifact_id": "metadata_index_parquet",
            "title": "Metadata-only interoperability index",
            "relative_path": "metadata_index.parquet",
            "media_type": "application/vnd.apache.parquet",
            "disposition": "metadata_and_rebuild_instructions_only",
            "source_kind": "generated_metadata",
            "record_count": 6,
            "byte_count": None,
            "sha256": None,
            "repository_license": "pending_owner_license_selection",
            "source_license": "pending_owner_license_selection",
            "reason": "Parquet contains only paths, hashes, counts, and status metadata.",
        },
        {
            "artifact_id": "croissant_metadata",
            "title": "Croissant release metadata",
            "relative_path": "croissant.json",
            "media_type": "application/ld+json",
            "disposition": "metadata_and_rebuild_instructions_only",
            "source_kind": "generated_metadata",
            "record_count": 20,
            "byte_count": None,
            "sha256": None,
            "repository_license": "pending_owner_license_selection",
            "source_license": "pending_owner_license_selection",
            "reason": f"Validated with installed mlcroissant in {mode} mode against metadata-only surface.",
        },
    )
    return tuple(MappingProxyType(dict(row)) for row in rows)


def _parquet_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    table = pa.table(
        {
            "artifact_id": [str(row["artifact_id"]) for row in rows],
            "relative_path": [str(row["relative_path"]) for row in rows],
            "media_type": [str(row["media_type"]) for row in rows],
            "disposition": [str(row["disposition"]) for row in rows],
            "source_kind": [str(row["source_kind"]) for row in rows],
            "record_count": [int(row["record_count"]) for row in rows],
            "byte_count": [
                None if row["byte_count"] is None else int(row["byte_count"])
                for row in rows
            ],
            "sha256": [None if row["sha256"] is None else str(row["sha256"]) for row in rows],
            "restricted_record_bytes_present": [False for _ in rows],
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        data_page_version="1.0",
    )
    return sink.getvalue().to_pybytes()


def _data_card(config: ReleaseMetadataConfig) -> bytes:
    text = "\n".join(
        [
            "# Phase 6 Release Metadata Data Card",
            "",
            "## Summary",
            "This Step 9 artifact publishes only metadata and rebuild instructions.",
            "",
            "## Source Boundaries",
            "- RRUFF record-level derivatives: local_only_pending_redistribution_review",
            "- Bacteria-ID records: redistribution_requires_review; no record bytes distributed",
            "- Sugar source evidence: CC BY 4.0 source evidence only; derived terms are separate",
            "- Repository-wide code, document, and data licenses: pending_owner_license_selection",
            "",
            "## Scientific Limits",
            "- Leaderboard state: deferred_no_redistributable_hidden_gt",
            "- Source revision status: unavailable_no_valid_git_repository",
            "- GPU reproducibility claim: not made",
            "",
            "## Interoperability",
            "- metadata_index.parquet is metadata-only and contains no spectral arrays",
            "- croissant.json describes only the allowed metadata surface",
            "",
        ]
    )
    return text.encode("utf-8")


def _limitations_rows(step8_receipt: Mapping[str, object] | None) -> tuple[Mapping[str, object], ...]:
    rows = (
        {
            "limitation_id": "pending_owner_license_selection",
            "state": "open",
            "detail": "Repository-wide code, document, and data license selection is owner-only.",
        },
        {
            "limitation_id": "deferred_no_redistributable_hidden_gt",
            "state": "deferred",
            "detail": "No hosted leaderboard is released because no redistributable hidden GT exists.",
        },
        {
            "limitation_id": "unavailable_no_valid_git_repository",
            "state": "fixed",
            "detail": "The workspace has no valid git repository; file receipts replace commit identity.",
        },
        {
            "limitation_id": "formal_step8_parent_required",
            "state": "satisfied" if step8_receipt is not None else "fixture_not_applicable",
            "detail": "Formal Step 9 builds require an explicit verified Step 8 publication-core artifact path.",
        },
    )
    return tuple(MappingProxyType(dict(row)) for row in rows)


def _environment_payload(config: ReleaseMetadataConfig, *, mode: str) -> bytes:
    value = {
        "python_version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "mlcroissant_version": str(config.document["validator"]["python_package_version"]),
        "pyarrow_version": str(config.document["validator"]["pyarrow_version"]),
        "source_revision_status": str(config.document["release_rules"]["source_revision_status"]),
        "leaderboard_state": str(config.document["release_rules"]["leaderboard_state"]),
        "gpu_reproducibility_claim": "not_claimed",
        "mode": mode,
    }
    return _canonical(value)


def _croissant_context() -> Mapping[str, object]:
    return MappingProxyType(
        {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "equivalentProperty": "cr:equivalentProperty",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": {"@id": "cr:extract", "@type": "@json"},
            "field": "cr:field",
            "fileObject": "cr:fileObject",
            "fileProperty": "cr:fileProperty",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "samplingRate": "cr:samplingRate",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
            "dataCollection": "rai:dataCollection",
            "dataCollectionType": "rai:dataCollectionType",
            "dataCollectionMissingData": "rai:dataCollectionMissingData",
            "dataCollectionRawData": "rai:dataCollectionRawData",
            "dataCollectionTimeFrame": "rai:dataCollectionTimeFrame",
            "dataImputationProtocol": "rai:dataImputationProtocol",
            "dataPreprocessingProtocol": "rai:dataPreprocessingProtocol",
            "dataDataManipulationProtocol": "rai:dataDataManipulationProtocol",
            "dataAnnotationProtocol": "rai:dataAnnotationProtocol",
            "dataAnnotationPlatform": "rai:dataAnnotationPlatform",
            "dataAnnotationAnalysis": "rai:dataAnnotationAnalysis",
            "annotationsPerItem": "rai:annotationsPerItem",
            "annotatorDemographics": "rai:annotatorDemographics",
            "machineAnnotationTools": "rai:machineAnnotationTools",
            "dataBiases": "rai:dataBiases",
            "dataUseCases": "rai:dataUseCases",
            "dataLimitations": "rai:dataLimitations",
            "dataSocialImpact": "rai:dataSocialImpact",
            "personalSensitiveInformation": "rai:personalSensitiveInformation",
            "dataReleaseMaintenancePlan": "rai:dataReleaseMaintenancePlan",
        }
    )


def _croissant_payload(
    config: ReleaseMetadataConfig,
    matrix_rows: Sequence[Mapping[str, object]],
) -> bytes:
    distributions = []
    for row in matrix_rows:
        if str(row["relative_path"]).startswith("data/"):
            continue
        entry = {
            "@id": str(row["artifact_id"]),
            "@type": "cr:FileObject",
            "name": str(row["relative_path"]),
            "contentUrl": str(row["relative_path"]),
            "encodingFormat": str(row["media_type"]),
            "sha256": "0" * 64 if row["sha256"] is None else str(row["sha256"]),
        }
        distributions.append(entry)
    croissant = {
        "@context": dict(_croissant_context()),
        "@type": "sc:Dataset",
        "name": "Raman Preprocessing Evaluation Release Metadata",
        "description": "Phase 6 Step 9 metadata-only release surface without restricted record bytes.",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "citeAs": "Local Phase 6 Step 9 metadata fixture.",
        "datePublished": "2026-08-30",
        "license": "https://example.invalid/pending_owner_license_selection",
        "version": "1.0.0",
        "distribution": distributions,
        "dataCollection": "not_applicable_metadata_only_release_surface",
        "dataCollectionType": ["not_applicable"],
        "dataCollectionMissingData": "Restricted records are intentionally absent from the release surface.",
        "dataCollectionRawData": "No raw or restricted record bytes are redistributed in Step 9.",
        "dataCollectionTimeFrame": ["2026-08-14T00:00:00", "2026-08-30T00:00:00"],
        "dataImputationProtocol": "not_applicable",
        "dataPreprocessingProtocol": [
            "Metadata projection only.",
            "No spectral arrays or record-level RRUFF derivatives are included.",
        ],
        "dataDataManipulationProtocol": "Content-addressed metadata assembly with byte-stable ordering.",
        "dataAnnotationProtocol": ["not_applicable"],
        "dataAnnotationPlatform": ["not_applicable"],
        "dataAnnotationAnalysis": ["not_applicable"],
        "annotationsPerItem": "not_applicable",
        "annotatorDemographics": ["not_applicable"],
        "machineAnnotationTools": ["not_applicable"],
        "dataBiases": [
            "Release scope is dominated by redistribution and licensing constraints.",
            "The artifact intentionally favors metadata and rebuild instructions over record bytes.",
        ],
        "dataUseCases": [
            "metadata audit",
            "rebuild instructions",
            "license boundary review",
        ],
        "dataLimitations": [
            "No hosted leaderboard is released.",
            "No GPU reproducibility claim is made.",
            "Repository-wide license selection remains owner-only.",
        ],
        "dataSocialImpact": "not_applicable_for_metadata_only_release_surface",
        "personalSensitiveInformation": ["not_applicable"],
        "dataReleaseMaintenancePlan": "Owner review is required before any redistribution status changes.",
    }
    return _canonical(croissant)


def _validate_croissant(payload: bytes) -> Mapping[str, object]:
    document = json.loads(payload)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"ConjunctiveGraph is deprecated, use Dataset instead\.",
                category=DeprecationWarning,
                module=r"rdflib\.plugins\.parsers\.jsonld",
            )
            mlc.Dataset(document)
        return MappingProxyType({"status": "passed", "validator": "mlcroissant.Dataset"})
    except mlc.ValidationError as error:
        return MappingProxyType(
            {
                "status": "failed",
                "validator": "mlcroissant.Dataset",
                "error": str(error),
            }
        )


def _step8_receipt(step8_path: Path | None) -> Mapping[str, object] | None:
    if step8_path is None:
        return None
    path = Path(step8_path).resolve()
    if not path.is_dir():
        raise ReleaseMetadataError("step8 publication-core path must be an existing directory")
    required = ("manifest.json", "complete.json", "SHA256SUMS")
    for name in required:
        if not (path / name).is_file():
            raise ReleaseMetadataError(f"step8 publication-core receipt missing {name}")
    try:
        relative_path = path.relative_to(ROOT)
    except ValueError as error:
        raise ReleaseMetadataError("step8 publication-core path must be inside the repository root") from error
    _parse_sha256sums(path, {"manifest.json", "complete.json"})
    complete = json.loads((path / "complete.json").read_bytes())
    manifest = json.loads((path / "manifest.json").read_bytes())
    if str(complete.get("status")) != "complete" or str(manifest.get("status")) != "complete":
        raise ReleaseMetadataError("step8 publication-core receipt must be complete")
    return MappingProxyType(
        {
            "path": str(relative_path),
            "run_id": str(complete["run_id"]),
            "manifest_sha256": _sha_file(path / "manifest.json"),
            "complete_sha256": _sha_file(path / "complete.json"),
            "sha256sums_sha256": _sha_file(path / "SHA256SUMS"),
        }
    )


def _run_id(
    config: ReleaseMetadataConfig,
    *,
    mode: str,
    step8_receipt: Mapping[str, object] | None,
) -> str:
    payload = {
        "config_sha256": config.sha256,
        "mode": mode,
        "step8_receipt": None if step8_receipt is None else _json_ready(step8_receipt),
        "code_authority": _json_ready(_code_authority()),
    }
    return f"{config.document['artifact_contract']['run_prefix']}{_sha_bytes(RUN_DOMAIN + _canonical(payload))}"


def _payloads(
    config: ReleaseMetadataConfig,
    *,
    mode: str,
    step8_receipt: Mapping[str, object] | None,
    run_id: str,
) -> Mapping[str, bytes]:
    matrix_rows = _release_matrix_rows(config, mode=mode)
    matrix_fields = (
        "artifact_id",
        "title",
        "relative_path",
        "media_type",
        "disposition",
        "source_kind",
        "record_count",
        "byte_count",
        "sha256",
        "repository_license",
        "source_license",
        "reason",
    )
    release_matrix_csv = _csv_bytes(matrix_rows, matrix_fields)
    metadata_index_parquet = _parquet_bytes(matrix_rows)
    croissant_json = _croissant_payload(config, matrix_rows)
    croissant_validation = _validate_croissant(croissant_json)
    if croissant_validation["status"] != "passed":
        raise ReleaseMetadataError(f"croissant validation failed: {croissant_validation['error']}")
    authority_bridge = {
        "authorities": _json_ready(config.document["authorities"]),
        "code_authority": _json_ready(_code_authority()),
        "source_revision_status": str(config.document["release_rules"]["source_revision_status"]),
        "step8_publication_core_receipt": None if step8_receipt is None else _json_ready(step8_receipt),
    }
    preflight = {
        "status": "passed",
        "mode": mode,
        "formal_step8_gate": {
            "required": bool(config.document["formal_build_requires_step8_receipt"]),
            "state": "passed" if step8_receipt is not None else "fixture_not_applicable",
        },
        "croissant_validation": _json_ready(croissant_validation),
    }
    manifest = {
        "schema_version": "phase6-release-metadata-artifact-v1",
        "run_id": run_id,
        "status": "complete",
        "mode": mode,
        "release_row_count": len(matrix_rows),
        "configured_payload_count": int(config.document["artifact_contract"]["configured_payload_count"]),
        "payload_files": list(config.document["artifact_contract"]["payload_files"]),
        "source_revision_status": str(config.document["release_rules"]["source_revision_status"]),
        "leaderboard_state": str(config.document["release_rules"]["leaderboard_state"]),
    }
    payloads = {
        "config.json": _canonical(config.document),
        "authority_bridge.json": _canonical(authority_bridge),
        "preflight.json": _canonical(preflight),
        "release_matrix.csv": release_matrix_csv,
        "data_card.md": _data_card(config),
        "metadata_index.parquet": metadata_index_parquet,
        "croissant.json": croissant_json,
        "limitations.jsonl": _jsonl_bytes(_limitations_rows(step8_receipt)),
        "environment.json": _environment_payload(config, mode=mode),
        "manifest.json": _canonical(manifest),
        "complete.json": _canonical({"run_id": run_id, "status": "complete"}),
    }
    payloads["SHA256SUMS"] = _sha256sums(payloads)
    return MappingProxyType(payloads)


def build_phase6_release_metadata(
    output_root: Path,
    *,
    config_path: Path = DEFAULT_CONFIG,
    mode: str = "formal",
    step8_publication_core_path: Path | None = None,
) -> ReleaseMetadataSummary:
    config = load_phase6_release_metadata_config(config_path)
    if mode not in {"formal", "fixture"}:
        raise ReleaseMetadataError("mode must be fixture or formal")
    if mode == "formal" and step8_publication_core_path is None:
        raise ReleaseMetadataError("formal step8 publication-core path is required and must be verified")
    step8_receipt = _step8_receipt(step8_publication_core_path)
    run_id = _run_id(config, mode=mode, step8_receipt=step8_receipt)
    path = Path(output_root) / run_id
    path.mkdir(parents=True, exist_ok=True)
    payloads = _payloads(config, mode=mode, step8_receipt=step8_receipt, run_id=run_id)
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    return ReleaseMetadataSummary(
        path=path,
        run_id=run_id,
        status="complete",
        release_row_count=6,
    )


def build_live_release_metadata_config_document() -> Mapping[str, object]:
    authorities = {}
    for name, relpath in (
        ("phase0_step01_report", "reports/phase0/step01_data_source_audit.md"),
        ("phase4_step38_report", "reports/phase4/step38_phase4_closure_synthesis.md"),
        ("source_manifest", "data/raw/source_manifest.jsonl"),
        ("source_versions", "data/raw/source_versions.json"),
        ("requirements_lock", "env/requirements.lock"),
        ("phase3_requirements_lock", "env/phase3-requirements.lock"),
    ):
        path = ROOT / relpath
        authorities[name] = {
            "path": relpath,
            "byte_count": int(path.stat().st_size),
            "sha256": _sha_file(path),
        }
    return {
        "schema_version": "phase6-release-metadata-v1",
        "artifact_contract": {
            "run_prefix": "phase6-release-metadata-",
            "configured_payload_count": 10,
            "payload_files": [
                "config.json",
                "authority_bridge.json",
                "preflight.json",
                "release_matrix.csv",
                "data_card.md",
                "metadata_index.parquet",
                "croissant.json",
                "limitations.jsonl",
                "environment.json",
                "manifest.json",
            ],
        },
        "formal_build_requires_step8_receipt": True,
        "runtime_modes": {"default_mode": "formal", "allowed_modes": ["formal", "fixture"]},
        "paper_copy_targets": ["paper/croissant.json"],
        "release_rules": {
            "allowed_dispositions": [
                "public_release_candidate",
                "metadata_and_rebuild_instructions_only",
                "local_only_pending_redistribution_review",
                "external_reference_only",
                "excluded",
            ],
            "source_revision_status": "unavailable_no_valid_git_repository",
            "leaderboard_state": "deferred_no_redistributable_hidden_gt",
        },
        "validator": {
            "python_package": "mlcroissant",
            "python_package_version": importlib.metadata.version("mlcroissant"),
            "pyarrow_version": importlib.metadata.version("pyarrow"),
        },
        "authorities": authorities,
    }

