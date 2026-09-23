from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class SugarMixturesValidationError(ValueError):
    def __init__(self, path: str, reason: str, code: str) -> None:
        super().__init__(f"{path}: {reason} [{code}]")
        self.path = path
        self.reason = reason
        self.code = code


@dataclass(frozen=True)
class _SugarAcceptedMember:
    source_member: str
    source_basename: str
    bytes: int
    crc32: int
    sha256: str
    condition: str
    source_samp: int
    source_row: str
    source_column: int
    source_plate: int
    source_round: int
    source_measurement: int
    source_repetition: int
    well_key: str
    sample_id: str
    acquisition_id: str
    record_id: str
    record_role: str
    pure_component: str | None


@dataclass(frozen=True)
class _SugarRecipe:
    well_key: str
    source_samp: int
    component_volumes_ul: Mapping[str, int]
    total_volume_ul: int
    component_fractions: Mapping[str, float]
    concentrations_mol_l: Mapping[str, float]


@dataclass(frozen=True)
class _SugarEvidenceMember:
    source_member: str
    role: str
    bytes: int
    crc32: int
    sha256: str


@dataclass(frozen=True)
class _SugarEvidenceSnapshot:
    source_artifact: str
    role: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class _SugarSourceContract:
    archive_name: str
    archive_bytes: int
    archive_md5: str
    archive_sha256: str
    archive_member_count: int
    archive_uncompressed_bytes: int
    central_directory_inventory_sha256: str
    expected_role_counts: Mapping[str, int]
    expected_role_bytes: Mapping[str, int]
    expected_role_sha256: Mapping[str, str]
    expected_evidence_snapshots: Mapping[str, tuple[int, str]]
    expected_counts: Mapping[str, int | float | str]


@dataclass(frozen=True)
class SugarMixturesInspection:
    raw_root: Path
    archive_md5: str
    archive_sha256: str
    central_directory_inventory_sha256: str
    central_directory_safety: Mapping[str, int]
    member_roles: Mapping[str, Mapping[str, int | str]]
    accepted_members: Sequence[_SugarAcceptedMember]
    relevant_evidence_members: Sequence[_SugarEvidenceMember]
    evidence_snapshots: Sequence[_SugarEvidenceSnapshot]
    recipes: Mapping[str, _SugarRecipe]
    view_record_ids: Mapping[str, Sequence[str]]
    view_source_members: Mapping[str, Sequence[str]]
    endmember_constituent_record_ids: Mapping[str, Sequence[str]]
    endmember_constituent_source_members: Mapping[str, Sequence[str]]
    source_statistics: Mapping[str, int | float | str]
    core_axis_id: str
    pixel_axis_id: str
    wavelength_axis_id: str
    auxiliary_axis_set_id: str


@dataclass(frozen=True)
class SugarMixturesDatasetSummary:
    path: Path
    dataset_id: str
    record_count: int
    eligible_records: int
    axis_group_count: int
    output_bytes: int
    files: Mapping[str, Mapping[str, int | str]]


@dataclass(frozen=True)
class SugarMixturesCompanionSummary:
    path: Path
    artifact_role: str
    bytes: int
    sha256: str
    logical_content_sha256: str | None
    item_count: int


@dataclass(frozen=True)
class SugarMixturesConversionSummary:
    core: SugarMixturesDatasetSummary
    views: SugarMixturesCompanionSummary
    derived_endmembers: SugarMixturesCompanionSummary
    auxiliary_axes: SugarMixturesCompanionSummary
    acquisition_json: SugarMixturesCompanionSummary
    receipt_path: Path
    receipt_bytes: int
    receipt_sha256: str
    source_contract_sha256: str
    semantic_contract_sha256: str
    policy_contract_sha256: str
    output_snapshot_sha256: str
    transaction_mode: str
    committed: bool


@dataclass(frozen=True)
class SugarMixturesRuntimeMetrics:
    total_seconds: float
    inspection_seconds: float
    staged_build_seconds: float
    core_build_seconds: float
    views_build_seconds: float
    derived_endmembers_build_seconds: float
    auxiliary_axes_build_seconds: float
    acquisition_json_build_seconds: float
    receipt_build_seconds: float
    validation_seconds: float
    peak_rss_raw: int
    peak_rss_bytes: int
    core_output_bytes: int
    views_output_bytes: int
    derived_endmembers_output_bytes: int
    auxiliary_axes_output_bytes: int
    acquisition_json_output_bytes: int
    receipt_bytes: int
    combined_output_bytes: int
    source_archive_bytes: int
    canonical_relevant_source_bytes: int
    compression_ratio_archive: float
    compression_ratio_relevant: float
    core_axis_groups: int
    core_lazy_open_seconds: float
    core_representative_record_read_seconds: float
    core_representative_record_id: str
    views_open_validate_seconds: float
    acquisition_json_stream_validate_seconds: float
    auxiliary_axes_open_validate_seconds: float
    derived_endmembers_open_validate_seconds: float


@dataclass(frozen=True)
class _SugarComparisonSummary:
    source_rows_compared: int
    source_points_compared: int
    consolidated_intensity_cells_compared: int
    prepared_abundance_rows_compared: int
    prepared_endmembers_compared: int
    intensity_mismatches: int
    axis_mismatches: int
    target_mismatches: int
    metadata_mismatches: int
    preprocessing_status_mismatches: int
    companion_reference_mismatches: int


@dataclass(frozen=True)
class _SugarStagedData:
    core: SugarMixturesDatasetSummary
    views: SugarMixturesCompanionSummary
    derived_endmembers: SugarMixturesCompanionSummary
    auxiliary_axes: SugarMixturesCompanionSummary
    acquisition_json: SugarMixturesCompanionSummary
    receipt_path: Path
    receipt_bytes: int
    receipt_sha256: str
    comparison: _SugarComparisonSummary
    output_hashes: Mapping[str, str]


@dataclass(frozen=True)
class _SugarStagingReview:
    artifact_summaries: Mapping[str, Mapping[str, object]]
    receipt_sha256: str
    comparison: _SugarComparisonSummary
    source_statistics: Mapping[str, int | float | str]
    output_hashes: Mapping[str, str]
    metrics: SugarMixturesRuntimeMetrics
    feasibility_failures: Sequence[str]


@dataclass(frozen=True)
class _SugarPublicationResult:
    mode: str
    committed: bool
    output_snapshot_sha256: str
    staging_parent: Path | None


@dataclass(frozen=True)
class _SugarOutputRootLock:
    output_root: Path
    directory_fd: int
    created_output_root: bool


@dataclass(frozen=True)
class _SugarFinalState:
    classification: str
    output_snapshot_sha256: str | None


@dataclass(frozen=True)
class _SugarRecoveryPlan:
    document: Mapping[str, object]
    recovery_plan_sha256: str


@dataclass(frozen=True)
class _SugarRecoveryResult:
    document: Mapping[str, object]
