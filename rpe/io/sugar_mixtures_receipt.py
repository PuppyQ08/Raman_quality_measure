from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping, Sequence

from rpe.io.sugar_mixtures_models import (
    SugarMixturesCompanionSummary,
    SugarMixturesDatasetSummary,
    SugarMixturesInspection,
    SugarMixturesValidationError,
    _SugarComparisonSummary,
)
from rpe.io.sugar_mixtures_source import _PRODUCTION_SOURCE_CONTRACT
from rpe.io.store import DATASET_FILES, UnifiedDataset, validate_dataset


RECEIPT_NAME = "sugar_mixtures_raman_conversion.json"
RECEIPT_SCHEMA_VERSION = "1.0.0"
ADAPTER_VERSION = "0.1.0"
SCHEMA_VERSION = "0.1.0"
DATASET_ID = "sugar_mixtures_raman"

SOURCE_CONTRACT_DOMAIN = b"rpe-sugar-receipt-source-contract-v1\0"
SEMANTIC_CONTRACT_DOMAIN = b"rpe-sugar-receipt-semantic-contract-v1\0"
POLICY_CONTRACT_DOMAIN = b"rpe-sugar-receipt-policy-contract-v1\0"

PRODUCTION_SOURCE_CONTRACT_BYTES = 3_371_667
PRODUCTION_SOURCE_CONTRACT_SHA256 = (
    "3b92c106b3c212c93d8065832f8141f03a87327138e3866a91ad9c3bb0a14244"
)
PRODUCTION_SEMANTIC_CONTRACT_BYTES = 4_150
PRODUCTION_SEMANTIC_CONTRACT_SHA256 = (
    "b9d08417461fb632bdc9348596a289867849f5e7bf3d2c6281933561487cabaa"
)
PRODUCTION_POLICY_CONTRACT_BYTES = 2_652
PRODUCTION_POLICY_CONTRACT_SHA256 = (
    "27fb85f9c7ea44580839acfbd2dcc9614ce9b00f1947a710df163e9bf9c2cf21"
)

_RECEIPT_KEYS = {
    "receipt_schema_version",
    "adapter_version",
    "schema_version",
    "dataset_id",
    "source_contract",
    "source_contract_sha256",
    "semantic_contract",
    "semantic_contract_sha256",
    "preprocessing_evidence",
    "license",
    "limitations",
    "policy_contract_sha256",
    "outputs",
}
_SOURCE_KEYS = {
    "release",
    "archive",
    "central_directory_safety",
    "inventory_counts",
    "member_roles",
    "canonical_members",
    "relevant_evidence_members",
    "evidence_snapshots",
    "excluded_inventory",
}
_SEMANTIC_KEYS = {
    "dataset",
    "provenance",
    "targets",
    "preprocessing",
    "precision",
    "views",
    "derived_endmembers",
    "auxiliary_axes",
    "acquisition_json",
    "comparison",
}
_COMPANION_NAMES = {
    "views": "sugar_mixtures_raman_views.json",
    "derived_endmembers": (
        "sugar_mixtures_raman_derived_reference_endmembers.h5"
    ),
    "auxiliary_axes": "sugar_mixtures_raman_auxiliary_axes.h5",
    "acquisition_json": (
        "sugar_mixtures_raman_acquisition_json_text.jsonl"
    ),
}
_OUTPUT_NAMES = {DATASET_ID, *_COMPANION_NAMES.values()}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RELEASE = {
    "record_id": 10779223,
    "doi": "10.5281/zenodo.10779223",
    "concept_doi": "10.5281/zenodo.10779222",
    "url": "https://doi.org/10.5281/zenodo.10779223",
    "title": (
        'Research data supporting "Hyperspectral unmixing for Raman '
        'spectroscopy via physics-constrained autoencoders"'
    ),
    "publication_date": "2024-10-16",
    "status": "published",
    "access_right": "open",
    "creators": [
        "Georgiev, Dimitar",
        "Fernández-Galiana, Álvaro",
        "Vilms Pedersen, Simon",
        "Papadopoulos, Georgios",
        "Stevens, Molly M.",
        "Barahona, Mauricio",
    ],
}
_LIMITATION_VALUES = {
    "adapter_defined_train_test_split": False,
    "condition_difference_not_integration_time_only": True,
    "croissant_included": False,
    "derived_endmembers_not_clean_targets": True,
    "high_low_not_pointwise_pairs": True,
    "high_snr_not_clean_target": True,
    "instrument_internal_operations_unverified": True,
    "local_timestamp_timezone_unknown": True,
    "lod_loq_not_in_adapter": True,
    "nominal_concentrations_not_assay_measured": True,
    "normalization_applied": False,
    "physical_clean_ground_truth_present": False,
    "prepared_views_not_canonical_records": True,
    "public_bundle_review_required": True,
    "resampling_applied": False,
    "sample_age_and_temporal_effects_present": True,
    "source_declared_unprocessed_not_detector_native": True,
    "source_nonfinite_metadata_tagged": True,
    "statistical_preprocessing_probe_not_in_adapter": True,
    "synthetic_subtree_excluded": True,
    "water_not_analyte_target": True,
}


def _fatal(
    path: str,
    reason: str,
    code: str,
) -> SugarMixturesValidationError:
    return SugarMixturesValidationError(path, reason, code)


def _json_native(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _json_native(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _json_native(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_md5_sha256(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            md5.update(block)
            sha256.update(block)
    return md5.hexdigest(), sha256.hexdigest()


def _require_output_topology(
    receipt_path: Path,
    core_path: Path,
    companion_paths: Mapping[str, Path],
) -> None:
    receipt_path = Path(receipt_path)
    core_path = Path(core_path)
    staging_parent = receipt_path.parent
    if staging_parent.is_symlink() or not staging_parent.is_dir():
        raise _fatal(
            "receipt.outputs",
            "staging parent must be a non-symlink directory",
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )
    if set(companion_paths) != set(_COMPANION_NAMES):
        raise _fatal(
            "receipt.companion_paths",
            "companion path key set differs from contract",
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )
    if (
        receipt_path.name != RECEIPT_NAME
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
    ):
        raise _fatal(
            "receipt.outputs.sugar_mixtures_raman_conversion.json",
            "receipt must be a direct non-symlink regular file",
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )
    if (
        core_path.parent != staging_parent
        or core_path.name != DATASET_ID
        or core_path.is_symlink()
        or not core_path.is_dir()
    ):
        raise _fatal(
            f"receipt.outputs.{DATASET_ID}",
            "core must be a direct non-symlink directory",
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )
    for child in core_path.iterdir():
        if child.is_symlink() or not child.is_file():
            raise _fatal(
                f"receipt.outputs.{DATASET_ID}.{child.name}",
                "core entries must be direct non-symlink regular files",
                "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
            )
    for key, basename in _COMPANION_NAMES.items():
        path = Path(companion_paths[key])
        if (
            path.parent != staging_parent
            or path.name != basename
            or path.is_symlink()
            or not path.is_file()
        ):
            raise _fatal(
                f"receipt.outputs.{basename}",
                "companion must be a direct non-symlink regular file",
                "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
            )
    attributed_paths = [
        *(core_path / name for name in sorted(DATASET_FILES)),
        *(Path(companion_paths[key]) for key in sorted(_COMPANION_NAMES)),
    ]
    physical_identities = {
        (path.stat().st_dev, path.stat().st_ino)
        for path in attributed_paths
    }
    if len(physical_identities) != len(attributed_paths):
        raise _fatal(
            "receipt.outputs",
            "attributed paths must resolve to unique physical files",
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )


def _contract_identity(
    domain: bytes,
    document: Mapping[str, object],
) -> tuple[int, str]:
    payload = _canonical_json_bytes(document)
    return len(payload), hashlib.sha256(domain + payload).hexdigest()


def _source_contract(
    inspection: SugarMixturesInspection,
) -> dict[str, object]:
    statistics = inspection.source_statistics
    snapshot_roles = {
        "evidence/zenodo/10779223.json": "source_record_metadata",
        "receipts/sugar_mixtures_high_snr.json": (
            "loader_comparison_only"
        ),
        "receipts/sugar_mixtures_low_snr.json": (
            "loader_comparison_only"
        ),
    }
    canonical_members = [
        {
            "source_member": member.source_member,
            "bytes": member.bytes,
            "crc32": f"{member.crc32:08x}",
            "sha256": member.sha256,
            "condition": member.condition,
            "record_id": member.record_id,
        }
        for member in sorted(
            inspection.accepted_members,
            key=lambda value: value.source_member.encode("utf-8"),
        )
    ]
    evidence_members = [
        {
            "source_member": member.source_member,
            "role": member.role,
            "bytes": member.bytes,
            "crc32": f"{member.crc32:08x}",
            "sha256": member.sha256,
        }
        for member in sorted(
            inspection.relevant_evidence_members,
            key=lambda value: value.source_member.encode("utf-8"),
        )
    ]
    return {
        "release": dict(_RELEASE),
        "archive": {
            "basename": _PRODUCTION_SOURCE_CONTRACT.archive_name,
            "url": (
                "https://zenodo.org/api/records/10779223/files/"
                "Raw%20data.zip/content"
            ),
            "bytes": statistics["archive_bytes"],
            "md5": inspection.archive_md5,
            "sha256": inspection.archive_sha256,
            "retrieved_date": "2026-08-14",
            "member_count": statistics["archive_member_count"],
            "uncompressed_bytes": statistics[
                "archive_uncompressed_bytes"
            ],
            "zip_crc_passed": True,
            "central_directory_inventory_sha256": (
                inspection.central_directory_inventory_sha256
            ),
        },
        "central_directory_safety": {
            key: value
            for key, value in inspection.central_directory_safety.items()
        },
        "inventory_counts": {
            "total_members": statistics["archive_member_count"],
            "directory_entries": inspection.central_directory_safety[
                "directory_entries"
            ],
            "relevant_regular_members": statistics[
                "relevant_regular_members"
            ],
            "canonical_members": len(canonical_members),
            "excluded_synthetic_regular_members": statistics[
                "excluded_synthetic_regular_members"
            ],
        },
        "member_roles": [
            {
                key: value
                for key, value in inspection.member_roles[role].items()
            }
            for role in sorted(
                inspection.member_roles,
                key=lambda value: value.encode("utf-8"),
            )
        ],
        "canonical_members": canonical_members,
        "relevant_evidence_members": evidence_members,
        "evidence_snapshots": [
            {
                "source_artifact": snapshot.source_artifact,
                "role": snapshot_roles[snapshot.source_artifact],
                "bytes": snapshot.bytes,
                "sha256": snapshot.sha256,
            }
            for snapshot in sorted(
                inspection.evidence_snapshots,
                key=lambda value: value.source_artifact.encode("utf-8"),
            )
        ],
        "excluded_inventory": {
            "reason": "unrelated_synthetic_data_product",
            "regular_member_count": statistics[
                "excluded_synthetic_regular_members"
            ],
            "uncompressed_bytes": statistics[
                "excluded_synthetic_uncompressed_bytes"
            ],
        },
    }


def _license_document(
    inspection: SugarMixturesInspection,
) -> dict[str, object]:
    evidence = {
        snapshot.source_artifact: snapshot
        for snapshot in inspection.evidence_snapshots
    }
    source_evidence = evidence.get("evidence/zenodo/10779223.json")
    if source_evidence is None:
        raise _fatal(
            "receipt.license.source_evidence_artifact",
            "source-specific Zenodo evidence is missing",
            "RECEIPT_SOURCE_CONTRACT_INVALID",
        )
    return {
        "text": "CC BY 4.0",
        "identifier": "CC-BY-4.0",
        "status": "standardized",
        "source_license_id": "cc-by-4.0",
        "source_specific": True,
        "access_right": "open",
        "attribution_required": True,
        "public_redistribution_gate": "allow_with_attribution",
        "public_bundle_review_required": True,
        "source_evidence_artifact": (
            "evidence/zenodo/10779223.json"
        ),
        "source_evidence_sha256": source_evidence.sha256,
    }


def _limitations(
    inspection: SugarMixturesInspection,
) -> dict[str, object]:
    return {
        **_LIMITATION_VALUES,
        "synthetic_subtree_excluded_members": (
            inspection.source_statistics[
                "excluded_synthetic_regular_members"
            ]
        ),
    }


def _record_ids(core_path: Path) -> set[str]:
    with UnifiedDataset.open(core_path) as dataset:
        return set(dataset.record_ids)


def _validated_companions(
    *,
    raw_root: Path,
    inspection: SugarMixturesInspection,
    core: SugarMixturesDatasetSummary,
    companions: Mapping[str, SugarMixturesCompanionSummary],
) -> dict[str, Mapping[str, object]]:
    from rpe.io.sugar_mixtures_companions import (
        _validate_acquisition_json,
        _validate_auxiliary_axes,
        _validate_derived_endmembers,
        _validate_views,
    )

    if set(companions) != set(_COMPANION_NAMES):
        raise _fatal(
            "receipt.companions",
            "companion summary key set differs from contract",
            "RECEIPT_SEMANTIC_CONTRACT_INVALID",
        )
    record_ids = _record_ids(core.path)
    expected_ids = {
        member.record_id for member in inspection.accepted_members
    }
    if record_ids != expected_ids:
        raise _fatal(
            "receipt.core.record_ids",
            "core record IDs differ from source inspection",
            "RECEIPT_SEMANTIC_CONTRACT_INVALID",
        )
    for key, expected_name in _COMPANION_NAMES.items():
        summary = companions[key]
        if (
            Path(summary.path).name != expected_name
            or Path(summary.path).parent != Path(core.path).parent
            or Path(summary.path).is_symlink()
            or not Path(summary.path).is_file()
            or summary.bytes != Path(summary.path).stat().st_size
            or summary.sha256 != _sha256_file(Path(summary.path))
        ):
            raise _fatal(
                f"receipt.companions.{key}",
                "companion summary differs from staged physical file",
                "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
            )
    return {
        "views": _validate_views(
            companions["views"].path,
            inspection,
            record_ids=record_ids,
        ),
        "derived_endmembers": _validate_derived_endmembers(
            companions["derived_endmembers"].path,
            raw_root,
            inspection,
            record_ids=record_ids,
        ),
        "auxiliary_axes": _validate_auxiliary_axes(
            companions["auxiliary_axes"].path,
            inspection,
        ),
        "acquisition_json": _validate_acquisition_json(
            companions["acquisition_json"].path,
            raw_root,
            inspection,
            record_ids=record_ids,
        ),
    }


def _comparison_document(
    comparison: _SugarComparisonSummary,
) -> dict[str, int]:
    return {
        "source_rows_compared": comparison.source_rows_compared,
        "source_points_compared": comparison.source_points_compared,
        "consolidated_intensity_cells_compared": (
            comparison.consolidated_intensity_cells_compared
        ),
        "prepared_abundance_rows_compared": (
            comparison.prepared_abundance_rows_compared
        ),
        "prepared_endmembers_compared": (
            comparison.prepared_endmembers_compared
        ),
        "intensity_mismatches": comparison.intensity_mismatches,
        "axis_mismatches": comparison.axis_mismatches,
        "target_mismatches": comparison.target_mismatches,
        "metadata_mismatches": comparison.metadata_mismatches,
        "preprocessing_status_mismatches": (
            comparison.preprocessing_status_mismatches
        ),
        "companion_reference_mismatches": (
            comparison.companion_reference_mismatches
        ),
    }


def _validate_comparison_summary(
    inspection: SugarMixturesInspection,
    comparison: _SugarComparisonSummary,
) -> None:
    statistics = inspection.source_statistics
    record_count = int(statistics["record_count"])
    point_count = int(statistics["points_per_record"])
    expected = {
        "source_rows_compared": record_count,
        "source_points_compared": record_count * point_count,
        "consolidated_intensity_cells_compared": record_count * point_count,
        "prepared_abundance_rows_compared": sum(
            len(inspection.view_record_ids[view_id])
            for view_id in (
                "high_snr",
                "high_snr_no_refs",
                "low_snr",
                "low_snr_no_refs",
            )
        ),
        "prepared_endmembers_compared": 10,
        "intensity_mismatches": 0,
        "axis_mismatches": 0,
        "target_mismatches": 0,
        "metadata_mismatches": 0,
        "preprocessing_status_mismatches": 0,
        "companion_reference_mismatches": 0,
    }
    observed = _comparison_document(comparison)
    if observed != expected:
        raise _fatal(
            "receipt.semantic_contract.comparison",
            "source comparison summary differs from fixed equations",
            "RECEIPT_SEMANTIC_CONTRACT_INVALID",
        )


def _semantic_contract(
    *,
    inspection: SugarMixturesInspection,
    core: SugarMixturesDatasetSummary,
    validations: Mapping[str, Mapping[str, object]],
    comparison: _SugarComparisonSummary,
) -> dict[str, object]:
    statistics = inspection.source_statistics
    views = validations["views"]
    endmembers = validations["derived_endmembers"]
    auxiliary = validations["auxiliary_axes"]
    acquisition = validations["acquisition_json"]
    return {
        "dataset": {
            "dataset_id": DATASET_ID,
            "record_count": core.record_count,
            "canonical_source_members": len(inspection.accepted_members),
            "rejected_canonical_members": statistics[
                "rejected_canonical_members"
            ],
            "sample_count": statistics["sample_count"],
            "points_per_record": statistics["points_per_record"],
            "axis_groups": core.axis_group_count,
            "core_axis_id": inspection.core_axis_id,
            "eligible_records": core.eligible_records,
            "condition_counts": {
                "high_snr": statistics["high_records"],
                "low_snr": statistics["low_records"],
            },
            "record_role_counts": {
                "mixture": statistics["mixture_records"],
                "pure_reference": statistics["pure_reference_records"],
            },
            "target_presence_counts": {
                "baseline": 0,
                "class_label": 0,
                "clean": 0,
                "concentration": 0,
                "concentrations": core.record_count,
                "peaks": 0,
            },
            "identity_map_sha256": statistics["identity_map_sha256"],
        },
        "provenance": {
            "source_url": "https://doi.org/10.5281/zenodo.10779223",
            "license": "CC BY 4.0",
            "license_status": "standardized",
            "retrieved_date": "2026-08-14",
            "source_artifact_count": core.record_count,
            "sha256_scope": "uncompressed_source_member_bytes",
        },
        "targets": {
            "target_names": [
                "sucrose_nominal_mol_l",
                "fructose_nominal_mol_l",
                "maltose_nominal_mol_l",
                "glucose_nominal_mol_l",
            ],
            "concentration_units": {
                "fructose_nominal_mol_l": "mol/L",
                "glucose_nominal_mol_l": "mol/L",
                "maltose_nominal_mol_l": "mol/L",
                "sucrose_nominal_mol_l": "mol/L",
            },
            "levels_mol_l": [0.0, 0.08, 0.2, 0.32, 1.0],
            "well_target_contract_sha256": statistics[
                "well_target_contract_sha256"
            ],
            "record_target_map_sha256": statistics[
                "record_target_map_sha256"
            ],
        },
        "preprocessing": {
            "status_counts": {
                "known_corrected": 0,
                "known_raw": core.record_count,
                "unknown": 0,
            },
            "eligible_records": core.eligible_records,
            "evidence_id": "sugar-source-no-preprocessing-claim-v1",
            "evidence_document_sha256": statistics[
                "preprocessing_evidence_document_sha256"
            ],
            "record_map_sha256": statistics[
                "record_preprocessing_map_sha256"
            ],
        },
        "precision": {
            key: statistics[key]
            for key in (
                "intensity_float32_max_abs_error",
                "axis_float32_max_abs_error_cm1",
                "wavelength_float32_max_abs_error_nm",
            )
        },
        "views": {
            "companion_file": _COMPANION_NAMES["views"],
            "file_bytes": views["bytes"],
            "file_sha256": views["sha256"],
            "view_count": views["view_count"],
            "view_counts": {
                key: value for key, value in views["view_counts"].items()
            },
            "view_memberships": views["view_memberships"],
        },
        "derived_endmembers": {
            "companion_file": _COMPANION_NAMES["derived_endmembers"],
            "semantic_role": "derived_reference_endmember",
            "endmember_count": endmembers["endmember_count"],
            "logical_content_sha256": endmembers[
                "logical_content_sha256"
            ],
            "high_float32_value_sha256": endmembers[
                "high_float32_value_sha256"
            ],
            "low_float32_value_sha256": endmembers[
                "low_float32_value_sha256"
            ],
        },
        "auxiliary_axes": {
            "companion_file": _COMPANION_NAMES["auxiliary_axes"],
            "auxiliary_axis_set_id": auxiliary[
                "auxiliary_axis_set_id"
            ],
            "pixel_axis_id": auxiliary["pixel_axis_id"],
            "wavelength_axis_id": auxiliary["wavelength_axis_id"],
            "logical_content_sha256": auxiliary[
                "logical_content_sha256"
            ],
        },
        "acquisition_json": {
            "companion_file": _COMPANION_NAMES["acquisition_json"],
            "line_count": acquisition["lines"],
            "file_bytes": acquisition["bytes"],
            "file_sha256": acquisition["sha256"],
            "unique_source_json_texts": acquisition[
                "unique_source_json_texts"
            ],
            "unique_normalized_metadata_objects": acquisition[
                "unique_normalized_metadata_objects"
            ],
            "source_nonfinite_values": acquisition[
                "source_nonfinite_values"
            ],
            "source_nonfinite_representation": (
                "tagged_source_nan"
            ),
            "raw_text_inventory_sha256": acquisition[
                "raw_text_inventory_sha256"
            ],
            "normalized_metadata_sha256": statistics[
                "normalized_metadata_sha256"
            ],
            "core_metadata_sha256": statistics[
                "core_metadata_sha256"
            ],
        },
        "comparison": _comparison_document(comparison),
    }


def _output_attributions(
    core: SugarMixturesDatasetSummary,
    companions: Mapping[str, SugarMixturesCompanionSummary],
) -> dict[str, object]:
    core_path = Path(core.path)
    if (
        core_path.name != DATASET_ID
        or core_path.is_symlink()
        or not core_path.is_dir()
        or set(path.name for path in core_path.iterdir()) != set(DATASET_FILES)
    ):
        raise _fatal(
            "receipt.outputs.sugar_mixtures_raman",
            "core topology differs from five-file dataset contract",
            "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
        )
    core_files = {}
    for name in sorted(DATASET_FILES):
        path = core_path / name
        details = core.files.get(name)
        if (
            path.is_symlink()
            or not path.is_file()
            or not isinstance(details, Mapping)
            or details.get("bytes") != path.stat().st_size
            or details.get("sha256") != _sha256_file(path)
        ):
            raise _fatal(
                f"receipt.outputs.{DATASET_ID}.{name}",
                "core summary differs from staged physical file",
                "RECEIPT_OUTPUT_ATTRIBUTION_INVALID",
            )
        core_files[name] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    outputs: dict[str, object] = {
        DATASET_ID: {
            "artifact_type": "dataset",
            "files": core_files,
        }
    }
    for key, basename in _COMPANION_NAMES.items():
        summary = companions[key]
        outputs[basename] = {
            "artifact_type": "file",
            "files": {
                basename: {
                    "bytes": summary.bytes,
                    "sha256": summary.sha256,
                }
            },
        }
    return outputs


def _production_contract_guard(
    inspection: SugarMixturesInspection,
    *,
    source_identity: tuple[int, str],
    semantic_identity: tuple[int, str],
    policy_identity: tuple[int, str],
) -> None:
    if inspection.archive_sha256 != _PRODUCTION_SOURCE_CONTRACT.archive_sha256:
        return
    expected = {
        "source": (
            source_identity,
            (
                PRODUCTION_SOURCE_CONTRACT_BYTES,
                PRODUCTION_SOURCE_CONTRACT_SHA256,
            ),
        ),
        "semantic": (
            semantic_identity,
            (
                PRODUCTION_SEMANTIC_CONTRACT_BYTES,
                PRODUCTION_SEMANTIC_CONTRACT_SHA256,
            ),
        ),
        "policy": (
            policy_identity,
            (
                PRODUCTION_POLICY_CONTRACT_BYTES,
                PRODUCTION_POLICY_CONTRACT_SHA256,
            ),
        ),
    }
    for name, (observed, required) in expected.items():
        if observed != required:
            raise _fatal(
                f"receipt.{name}_contract_sha256",
                f"expected {required}, observed {observed}",
                "RECEIPT_STATIC_CONTRACT_MISMATCH",
            )


def _build_receipt_document(
    *,
    raw_root: Path,
    inspection: SugarMixturesInspection,
    core: SugarMixturesDatasetSummary,
    companions: Mapping[str, SugarMixturesCompanionSummary],
    comparison: _SugarComparisonSummary,
) -> Mapping[str, object]:
    from rpe.io.sugar_mixtures import (
        _preprocessing_evidence_document,
        _require_matching_inspection,
    )

    raw_root = Path(raw_root).resolve()
    _require_matching_inspection(raw_root, inspection)
    validation = validate_dataset(core.path, verify_checksums=True)
    if (
        validation.dataset_id != DATASET_ID
        or validation.record_count != core.record_count
        or validation.axis_group_count != core.axis_group_count
        or core.eligible_records != core.record_count
    ):
        raise _fatal(
            "receipt.core",
            "core summary differs from independently validated dataset",
            "RECEIPT_SEMANTIC_CONTRACT_INVALID",
        )
    _validate_comparison_summary(inspection, comparison)
    validations = _validated_companions(
        raw_root=raw_root,
        inspection=inspection,
        core=core,
        companions=companions,
    )
    source = _source_contract(inspection)
    semantic = _semantic_contract(
        inspection=inspection,
        core=core,
        validations=validations,
        comparison=comparison,
    )
    preprocessing = _json_native(
        _preprocessing_evidence_document(raw_root, inspection)
    )
    license_document = _license_document(inspection)
    limitations = _limitations(inspection)
    policy = {
        "license": license_document,
        "limitations": limitations,
        "preprocessing_evidence": [preprocessing],
    }
    source_identity = _contract_identity(SOURCE_CONTRACT_DOMAIN, source)
    semantic_identity = _contract_identity(
        SEMANTIC_CONTRACT_DOMAIN,
        semantic,
    )
    policy_identity = _contract_identity(POLICY_CONTRACT_DOMAIN, policy)
    _production_contract_guard(
        inspection,
        source_identity=source_identity,
        semantic_identity=semantic_identity,
        policy_identity=policy_identity,
    )
    return {
        "receipt_schema_version": RECEIPT_SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "source_contract": source,
        "source_contract_sha256": source_identity[1],
        "semantic_contract": semantic,
        "semantic_contract_sha256": semantic_identity[1],
        "preprocessing_evidence": [preprocessing],
        "license": license_document,
        "limitations": limitations,
        "policy_contract_sha256": policy_identity[1],
        "outputs": _output_attributions(core, companions),
    }


def _write_receipt(
    path: Path,
    document: Mapping[str, object],
) -> tuple[int, str]:
    path = Path(path)
    payload = _canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except OSError as error:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise _fatal(
            "receipt.write",
            str(error),
            "RECEIPT_WRITE_FAILED",
        ) from error
    return len(payload), hashlib.sha256(payload).hexdigest()


def _read_receipt(path: Path) -> Mapping[str, object]:
    path = Path(path)
    try:
        payload = path.read_bytes()
        document = json.loads(
            payload,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {value}")
            ),
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise _fatal(
            "receipt",
            str(error),
            "RECEIPT_JSON_NONCANONICAL",
        ) from error
    if not isinstance(document, Mapping):
        raise _fatal(
            "receipt",
            "receipt must be an object",
            "RECEIPT_KEYS_INVALID",
        )
    try:
        canonical = _canonical_json_bytes(document)
    except (TypeError, ValueError) as error:
        raise _fatal(
            "receipt",
            str(error),
            "RECEIPT_JSON_NONCANONICAL",
        ) from error
    if payload != canonical:
        raise _fatal(
            "receipt",
            "receipt is not canonical finite JSON",
            "RECEIPT_JSON_NONCANONICAL",
        )
    return document


def _first_mismatch(
    observed: object,
    expected: object,
    path: str,
) -> tuple[str, str] | None:
    if isinstance(expected, Mapping):
        if not isinstance(observed, Mapping):
            return path, "expected object"
        if set(observed) != set(expected):
            return path, "object key set differs"
        for key in sorted(expected):
            mismatch = _first_mismatch(
                observed[key],
                expected[key],
                f"{path}.{key}",
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(expected, list):
        if not isinstance(observed, list):
            return path, "expected array"
        if len(observed) != len(expected):
            return path, "array length differs"
        for index, (observed_item, expected_item) in enumerate(
            zip(observed, expected, strict=True)
        ):
            mismatch = _first_mismatch(
                observed_item,
                expected_item,
                f"{path}[{index}]",
            )
            if mismatch is not None:
                return mismatch
        return None
    if type(observed) is not type(expected):
        return path, "scalar type differs"
    if isinstance(expected, float) and not math.isfinite(observed):
        return path, "number must be finite"
    if observed != expected:
        return path, "value differs"
    return None


def _validate_source_files(
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> None:
    archive_path = (
        Path(raw_root).resolve() / _PRODUCTION_SOURCE_CONTRACT.archive_name
    )
    try:
        if archive_path.is_symlink() or not archive_path.is_file():
            raise OSError("archive must be a non-symlink regular file")
        observed_bytes = archive_path.stat().st_size
        observed_md5, observed_sha256 = _archive_md5_sha256(archive_path)
    except OSError as error:
        raise _fatal(
            "receipt.source_contract.archive",
            str(error),
            "RECEIPT_SOURCE_CONTRACT_INVALID",
        ) from error
    expected_bytes = inspection.source_statistics["archive_bytes"]
    if observed_bytes != expected_bytes:
        raise _fatal(
            "receipt.source_contract.archive.bytes",
            f"expected {expected_bytes}, observed {observed_bytes}",
            "RECEIPT_SOURCE_CONTRACT_INVALID",
        )
    if observed_sha256 != inspection.archive_sha256:
        raise _fatal(
            "receipt.source_contract.archive.sha256",
            (
                f"expected {inspection.archive_sha256}, "
                f"observed {observed_sha256}"
            ),
            "RECEIPT_SOURCE_CONTRACT_INVALID",
        )
    if observed_md5 != inspection.archive_md5:
        raise _fatal(
            "receipt.source_contract.archive.md5",
            f"expected {inspection.archive_md5}, observed {observed_md5}",
            "RECEIPT_SOURCE_CONTRACT_INVALID",
        )
    expected_members = {
        member.source_member: (
            member.bytes,
            member.crc32,
            member.sha256,
        )
        for member in (
            *inspection.accepted_members,
            *inspection.relevant_evidence_members,
        )
    }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise _fatal(
                    "receipt.source_contract.archive.zip_crc_passed",
                    f"CRC failed for {bad_member}",
                    "RECEIPT_SOURCE_CONTRACT_INVALID",
                )
            infos: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                infos.setdefault(info.filename, []).append(info)
            for name, (expected_bytes, expected_crc32, expected_sha256) in (
                expected_members.items()
            ):
                matches = infos.get(name, ())
                if len(matches) != 1:
                    raise _fatal(
                        f"receipt.source_contract.members.{name}",
                        "source member is missing or duplicated",
                        "RECEIPT_SOURCE_CONTRACT_INVALID",
                    )
                payload = archive.read(matches[0])
                if (
                    len(payload) != expected_bytes
                    or matches[0].CRC != expected_crc32
                    or hashlib.sha256(payload).hexdigest()
                    != expected_sha256
                ):
                    raise _fatal(
                        f"receipt.source_contract.members.{name}",
                        "source member bytes differ from inspection",
                        "RECEIPT_SOURCE_CONTRACT_INVALID",
                    )
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            "receipt.source_contract.archive",
            str(error),
            "RECEIPT_SOURCE_CONTRACT_INVALID",
        ) from error
    evidence_root = Path(raw_root).resolve().parent.parent
    for snapshot in inspection.evidence_snapshots:
        path = evidence_root / snapshot.source_artifact
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != snapshot.bytes
            or _sha256_file(path) != snapshot.sha256
        ):
            raise _fatal(
                (
                    "receipt.source_contract.evidence_snapshots."
                    f"{snapshot.source_artifact}"
                ),
                "retained evidence differs from inspection",
                "RECEIPT_SOURCE_CONTRACT_INVALID",
            )


def _validate_receipt(
    path: Path,
    *,
    raw_root: Path,
    inspection: SugarMixturesInspection,
    core_path: Path,
    companion_paths: Mapping[str, Path],
    comparison: _SugarComparisonSummary,
) -> Mapping[str, object]:
    from rpe.io.sugar_mixtures import (
        _compare_sugar_dataset_to_source,
    )

    _require_output_topology(path, core_path, companion_paths)
    document = _read_receipt(path)
    if set(document) != _RECEIPT_KEYS:
        raise _fatal(
            "receipt",
            "top-level key set differs from contract",
            "RECEIPT_KEYS_INVALID",
        )
    if (
        not isinstance(document.get("source_contract"), Mapping)
        or set(document["source_contract"]) != _SOURCE_KEYS
        or not isinstance(document.get("semantic_contract"), Mapping)
        or set(document["semantic_contract"]) != _SEMANTIC_KEYS
        or not isinstance(document.get("outputs"), Mapping)
        or set(document["outputs"]) != _OUTPUT_NAMES
    ):
        raise _fatal(
            "receipt",
            "nested contract key set differs",
            "RECEIPT_KEYS_INVALID",
        )
    for name in (
        "source_contract_sha256",
        "semantic_contract_sha256",
        "policy_contract_sha256",
    ):
        if (
            not isinstance(document[name], str)
            or _SHA256_PATTERN.fullmatch(document[name]) is None
        ):
            raise _fatal(
                f"receipt.{name}",
                "contract digest must be lowercase SHA256",
                "RECEIPT_DIGEST_INVALID",
            )
    _validate_source_files(raw_root, inspection)
    validation = validate_dataset(core_path, verify_checksums=True)
    core = SugarMixturesDatasetSummary(
        path=Path(core_path),
        dataset_id=validation.dataset_id,
        record_count=validation.record_count,
        eligible_records=validation.preprocessing_status_counts[
            "known_raw"
        ],
        axis_group_count=validation.axis_group_count,
        output_bytes=sum(
            (Path(core_path) / name).stat().st_size
            for name in DATASET_FILES
        ),
        files={
            name: {
                "bytes": (Path(core_path) / name).stat().st_size,
                "sha256": _sha256_file(Path(core_path) / name),
            }
            for name in sorted(DATASET_FILES)
        },
    )
    role_by_key = {
        "views": "record_views",
        "derived_endmembers": "derived_reference_endmembers",
        "auxiliary_axes": "auxiliary_axes",
        "acquisition_json": "acquisition_json_text",
    }
    item_count_by_key = {
        "views": 8,
        "derived_endmembers": 10,
        "auxiliary_axes": 1,
        "acquisition_json": validation.record_count,
    }
    companions = {
        key: SugarMixturesCompanionSummary(
            path=Path(companion_paths[key]),
            artifact_role=role_by_key[key],
            bytes=Path(companion_paths[key]).stat().st_size,
            sha256=_sha256_file(Path(companion_paths[key])),
            logical_content_sha256=None,
            item_count=item_count_by_key[key],
        )
        for key in _COMPANION_NAMES
    }
    fresh_comparison = _compare_sugar_dataset_to_source(
        raw_root,
        core_path,
        inspection,
    )
    if fresh_comparison != comparison:
        raise _fatal(
            "receipt.semantic_contract.comparison",
            "fresh source comparison differs from supplied summary",
            "RECEIPT_SEMANTIC_CONTRACT_INVALID",
        )
    expected = _build_receipt_document(
        raw_root=raw_root,
        inspection=inspection,
        core=core,
        companions=companions,
        comparison=fresh_comparison,
    )
    mismatch = _first_mismatch(document, expected, "receipt")
    if mismatch is not None:
        mismatch_path, reason = mismatch
        raise _fatal(
            mismatch_path,
            reason,
            "RECEIPT_CONTRACT_MISMATCH",
        )
    return document
