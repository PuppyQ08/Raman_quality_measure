from __future__ import annotations

import datetime
import csv
import hashlib
import io
import json
import math
import os
import re
import resource
import shutil
import struct
import tempfile
import time
import weakref
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Iterator, Mapping

import h5py
import numpy as np
import pandas as pd

from rpe.io.schema import (
    LicenseStatus,
    PreprocessingStatus,
    Provenance,
    RamanRecord,
    SpectrumMetadata,
    Targets,
    axis_id,
    validate_record,
)
from rpe.io.sugar_mixtures_models import (
    SugarMixturesCompanionSummary,
    SugarMixturesDatasetSummary,
    SugarMixturesInspection,
    SugarMixturesRuntimeMetrics,
    SugarMixturesValidationError,
    _SugarComparisonSummary,
    _SugarAcceptedMember,
    _SugarRecipe,
    _SugarSourceContract,
    _SugarStagedData,
    _SugarStagingReview,
)
from rpe.io.sugar_mixtures_source import (
    CLAIM_README,
    HIGH_DIRECTORY,
    LOW_DIRECTORY,
    ORACLE_MEMBERS,
    PREPARED_PREFIX,
    _PRODUCTION_SOURCE_CONTRACT,
    _inspect_sugar_mixtures_source,
    _reparse_verified_member,
)
from rpe.io.store import (
    DATASET_FILES,
    DatasetValidationError,
    UnifiedDataset,
    _inspect_dataset,
    validate_dataset,
    write_dataset,
)

DATASET_ID = "sugar_mixtures_raman"
SOURCE_URL = "https://doi.org/10.5281/zenodo.10779223"
SOURCE_LICENSE = "CC BY 4.0"
RETRIEVED_DATE = date(2026, 8, 14)
INSTRUMENT = "B-Raman custom Raman microspectroscopy platform"
PREPROCESSING_EVIDENCE_ID = "sugar-source-no-preprocessing-claim-v1"
PREPROCESSING_STATE = "source_declared_unprocessed"
COMPONENTS = (
    "sucrose",
    "fructose",
    "maltose",
    "glucose",
    "water",
)
TARGET_COMPONENTS = COMPONENTS[:-1]
_OBJECTIVE_PATTERN = re.compile(
    r"^(?:N PLAN )?(?P<magnification>[0-9]+(?:\.[0-9]+)?)x/"
    r"(?P<na>[0-9]+(?:\.[0-9]+)?)$"
)
_ORACLE_BY_BASENAME = {
    Path(member).name: member
    for member in ORACLE_MEMBERS
}
_STAGED_TIMINGS: dict[
    int,
    tuple[weakref.ReferenceType[_SugarStagedData], Mapping[str, float]],
] = {}
_SUGAR_METRIC_THRESHOLDS = (
    ("peak_rss_bytes", 2_147_483_648),
    ("combined_output_bytes", 536_870_912),
    ("inspection_seconds", 120.0),
    ("staged_build_seconds", 600.0),
    ("validation_seconds", 600.0),
    ("total_seconds", 1_500.0),
    ("core_lazy_open_seconds", 30.0),
    ("core_representative_record_read_seconds", 2.0),
    ("views_open_validate_seconds", 10.0),
    ("acquisition_json_stream_validate_seconds", 120.0),
    ("auxiliary_axes_open_validate_seconds", 10.0),
    ("derived_endmembers_open_validate_seconds", 10.0),
)
_SUGAR_STAGED_FILE_NAMES = {
    *(f"{DATASET_ID}/{name}" for name in DATASET_FILES),
    "sugar_mixtures_raman_views.json",
    "sugar_mixtures_raman_derived_reference_endmembers.h5",
    "sugar_mixtures_raman_auxiliary_axes.h5",
    "sugar_mixtures_raman_acquisition_json_text.jsonl",
    "sugar_mixtures_raman_conversion.json",
}
_REQUIRED_FREE_DISK_BYTES = 536_870_912 + 1_073_741_824


def _inspect_sugar_mixtures(
    raw_root: Path,
    contract: _SugarSourceContract,
) -> SugarMixturesInspection:
    return _inspect_sugar_mixtures_source(raw_root, contract)


def inspect_sugar_mixtures(
    raw_root: Path,
) -> SugarMixturesInspection:
    return _inspect_sugar_mixtures(
        raw_root,
        _PRODUCTION_SOURCE_CONTRACT,
    )


def _fatal(
    path: str,
    reason: str,
    code: str,
) -> SugarMixturesValidationError:
    return SugarMixturesValidationError(path, reason, code)


def _canonical_json_bytes(
    value: object,
    *,
    newline: bool,
) -> bytes:
    def project(current):
        if isinstance(current, Mapping):
            return {
                key: project(item)
                for key, item in current.items()
            }
        if isinstance(current, (tuple, list)):
            return [project(item) for item in current]
        return current

    encoded = json.dumps(
        project(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    if newline:
        encoded += "\n"
    return encoded.encode("utf-8")


def _update_length_prefixed(digest, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _readonly_float32(values: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype="<f4")
    immutable = np.frombuffer(array.tobytes(order="C"), dtype="<f4")
    immutable.setflags(write=False)
    return immutable


def _float32_max_abs_error(values: np.ndarray) -> float:
    stored = np.asarray(values, dtype="<f4").astype(np.float64)
    return float(np.max(np.abs(values - stored)))


def _require_matching_inspection(
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> None:
    resolved_root = Path(raw_root).resolve()
    if inspection.raw_root != resolved_root:
        raise _fatal(
            "inspection.raw_root",
            (
                f"expected {resolved_root}, "
                f"observed {inspection.raw_root}"
            ),
            "INSPECTION_ROOT_MISMATCH",
        )
    statistics = inspection.source_statistics
    expected = {
        "archive_md5": inspection.archive_md5,
        "archive_sha256": inspection.archive_sha256,
        "central_directory_inventory_sha256": (
            inspection.central_directory_inventory_sha256
        ),
        "core_axis_id": inspection.core_axis_id,
        "pixel_axis_id": inspection.pixel_axis_id,
        "wavelength_axis_id": inspection.wavelength_axis_id,
        "auxiliary_axis_set_id": inspection.auxiliary_axis_set_id,
    }
    for name, value in expected.items():
        if statistics.get(name) != value:
            raise _fatal(
                f"inspection.source_statistics.{name}",
                "inspection identity summary is internally inconsistent",
                "INSPECTION_STALE",
            )
    archive_path = (
        resolved_root / _PRODUCTION_SOURCE_CONTRACT.archive_name
    )
    try:
        archive_bytes = archive_path.stat().st_size
    except OSError as error:
        raise _fatal(
            "inspection.archive",
            str(error),
            "INSPECTION_STALE",
        ) from error
    if archive_bytes != statistics.get("archive_bytes"):
        raise _fatal(
            "inspection.archive.bytes",
            (
                f"expected {statistics.get('archive_bytes')}, "
                f"observed {archive_bytes}"
            ),
            "INSPECTION_STALE",
        )


def _normalized_acquisition_metadata(
    member: _SugarAcceptedMember,
    source: Mapping[str, object],
) -> Mapping[str, object]:
    path = f"records.{member.record_id}.source_acquisition_metadata"
    try:
        acquired = datetime.datetime.strptime(
            source["Date"],
            "%Y-%m-%d %H:%M:%S",
        )
    except (TypeError, ValueError) as error:
        raise _fatal(
            f"{path}.acquired_at_local",
            str(error),
            "METADATA_NORMALIZATION_INVALID",
        ) from error
    ambient_temperature_text = source["Temperature [C]"]
    ambient_humidity_text = source["Humidity [1/100]"]
    normalized = {
        "acquired_at_local": acquired.isoformat(timespec="seconds"),
        "acquired_at_timezone": None,
        "stage_position_um": [
            float(value)
            for value in source["Position [um]"]
        ],
        "ambient_temperature_c": (
            None
            if ambient_temperature_text == "N/A"
            else float(ambient_temperature_text)
        ),
        "ambient_temperature_source_text": ambient_temperature_text,
        "ambient_humidity_hundredth": (
            None
            if ambient_humidity_text == "N/A"
            else float(ambient_humidity_text)
        ),
        "ambient_humidity_source_text": ambient_humidity_text,
        "spectro_temp_c": int(source["Spectro_Temp [C]"]),
        "laser_power_mw": float(source["Laser power [mW]"]),
        "excitation_nm": float(source["Excitation wavelength [nm]"]),
        "objective_name": source["Objective"],
        "objective_maker": source["Objective_Maker"],
        "objective_magnification": float(
            source["Objective_Magnification"]
        ),
        "objective_na": float(source["Objective_NA"]),
        "objective_wd_source": float(source["Objective_WD"]),
        "objective_immersion": source["Objective_Immersion"],
        "objective_tube_lens_f_source": float(
            source["Objective_Tube_Lens_f"]
        ),
        "integration_time_s": float(
            source["Spectrometer integration time [s]"]
        ),
        "n_accumulations": int(
            source["Spectrometer number of accumulations"]
        ),
        "spectrometer_temperature_c": int(
            source["Spectrometer temperature [C]"]
        ),
        "spectrometer_acquisition_mode_code": int(
            source["Spectrometer acquisition mode"]
        ),
        "spectrometer_read_mode_code": int(
            source["Spectrometer read mode"]
        ),
        "spectrometer_trigger_mode_code": int(
            source["Spectrometer trigger mode"]
        ),
        "magnification": float(source["Magnification"]),
        "spot_size_source": float(source["Spot_Size"]),
        "laser_offset_source": [
            {"source_nonfinite_number": "NaN"},
            {"source_nonfinite_number": "NaN"},
        ],
    }
    if ambient_temperature_text != "N/A" or ambient_humidity_text != "N/A":
        raise _fatal(
            path,
            "ambient source text must preserve the exact source N/A values",
            "METADATA_REDUNDANCY_MISMATCH",
        )
    if (
        normalized["magnification"]
        != normalized["objective_magnification"]
    ):
        raise _fatal(
            f"{path}.magnification",
            "Magnification must equal Objective_Magnification",
            "METADATA_REDUNDANCY_MISMATCH",
        )
    objective_match = _OBJECTIVE_PATTERN.fullmatch(
        str(normalized["objective_name"])
    )
    if (
        objective_match is None
        or float(objective_match["magnification"])
        != normalized["objective_magnification"]
        or float(objective_match["na"]) != normalized["objective_na"]
    ):
        raise _fatal(
            f"{path}.objective_name",
            "Objective text must agree with numeric magnification and NA",
            "METADATA_REDUNDANCY_MISMATCH",
        )
    if (
        normalized["excitation_nm"] != 785.0
        or normalized["n_accumulations"] != 1
    ):
        raise _fatal(
            path,
            "excitation and accumulation fields differ from source contract",
            "METADATA_REDUNDANCY_MISMATCH",
        )
    expected_integration = (
        5.0 if member.condition == "high_snr" else 0.5
    )
    expected_laser_power = (
        36.3 if member.condition == "high_snr" else 30.3
    )
    if (
        normalized["integration_time_s"] != expected_integration
        or normalized["laser_power_mw"] != expected_laser_power
    ):
        raise _fatal(
            path,
            "condition disagrees with integration time or laser power",
            "METADATA_CONDITION_MISMATCH",
        )
    return MappingProxyType(normalized)


def _source_metadata(
    member: _SugarAcceptedMember,
    recipe: _SugarRecipe,
    inspection: SugarMixturesInspection,
    acquisition: Mapping[str, object],
) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "source_member": member.source_member,
            "source_member_sha256": member.sha256,
            "source_basename": member.source_basename,
            "source_samp": member.source_samp,
            "source_row": member.source_row,
            "source_column": member.source_column,
            "source_plate": member.source_plate,
            "source_round": member.source_round,
            "source_measurement": member.source_measurement,
            "source_repetition": member.source_repetition,
            "source_acquisition_id": member.acquisition_id,
            "source_acquisition_condition": member.condition,
            "source_record_role": member.record_role,
            "source_auxiliary_axis_set_id": (
                inspection.auxiliary_axis_set_id
            ),
            "source_acquisition_metadata": acquisition,
            "source_sucrose_volume_ul": (
                recipe.component_volumes_ul["sucrose"]
            ),
            "source_fructose_volume_ul": (
                recipe.component_volumes_ul["fructose"]
            ),
            "source_maltose_volume_ul": (
                recipe.component_volumes_ul["maltose"]
            ),
            "source_glucose_volume_ul": (
                recipe.component_volumes_ul["glucose"]
            ),
            "source_water_volume_ul": (
                recipe.component_volumes_ul["water"]
            ),
            "source_total_volume_ul": recipe.total_volume_ul,
            "source_component_fractions": MappingProxyType(
                dict(recipe.component_fractions)
            ),
            "source_preprocessing_state": PREPROCESSING_STATE,
            "source_preprocessing_evidence_id": (
                PREPROCESSING_EVIDENCE_ID
            ),
        }
    )


def _targets(recipe: _SugarRecipe) -> Targets:
    return Targets(
        concentrations=MappingProxyType(
            {
                f"{component}_nominal_mol_l": (
                    recipe.component_volumes_ul[component]
                    / recipe.total_volume_ul
                )
                for component in TARGET_COMPONENTS
            }
        )
    )


def _well_target_contract_sha256(
    inspection: SugarMixturesInspection,
) -> str:
    digest = hashlib.sha256(b"rpe-sugar-well-target-contract-v1\0")
    for well_key in sorted(
        inspection.recipes,
        key=lambda value: value.encode("utf-8"),
    ):
        recipe = inspection.recipes[well_key]
        entry = {
            "well": well_key,
            "concentrations": dict(recipe.concentrations_mol_l),
            "source_component_fractions": dict(
                recipe.component_fractions
            ),
            "source_volumes_ul": {
                component: recipe.component_volumes_ul[component]
                for component in COMPONENTS
            },
            "source_total_volume_ul": recipe.total_volume_ul,
        }
        encoded = _canonical_json_bytes(entry, newline=True)
        _update_length_prefixed(digest, encoded)
    return digest.hexdigest()


def _identity_map_sha256(
    members: tuple[_SugarAcceptedMember, ...],
) -> str:
    digest = hashlib.sha256(b"rpe-sugar-identity-map-v1\0")
    for member in members:
        for value in (
            member.record_id,
            member.sample_id,
            member.acquisition_id,
            member.source_member,
        ):
            _update_length_prefixed(digest, value.encode("utf-8"))
    return digest.hexdigest()


def _validate_member_identity(member: _SugarAcceptedMember) -> None:
    expected_samp = (
        12 * (ord(member.source_row) - ord("A"))
        + member.source_column
    )
    expected_well = (
        f"{member.source_row}{member.source_column}_{member.source_plate}"
    )
    expected_sample = (
        f"sugar-well-{member.source_row.lower()}{member.source_column:02d}"
        f"-p{member.source_plate:02d}"
    )
    expected_record = (
        f"{member.condition}-s{member.source_samp:03d}-"
        f"{member.source_row.lower()}{member.source_column:02d}"
        f"-p{member.source_plate:02d}-r{member.source_round:02d}"
        f"-m{member.source_measurement:02d}"
        f"-rep{member.source_repetition:02d}"
    )
    prefix = (
        "Sugar_Concentration_Test"
        if member.condition == "high_snr"
        else "Sugar_Concentration_Test_Fast"
    )
    expected_basename = (
        f"{prefix}_{member.source_samp}_"
        f"{member.source_row}{member.source_column}_"
        f"{member.source_plate}_RD{member.source_round}_"
        f"M{member.source_measurement}_R{member.source_repetition}.csv"
    )
    if (
        member.source_samp != expected_samp
        or member.well_key != expected_well
        or member.sample_id != expected_sample
        or member.acquisition_id != expected_record
        or member.record_id != expected_record
        or member.source_basename != expected_basename
        or not member.source_member.endswith(f"/{expected_basename}")
    ):
        raise _fatal(
            f"inspection.accepted_members.{member.record_id}",
            "member identity fields are not bidirectionally consistent",
            "INSPECTION_IDENTITY_MISMATCH",
        )


def _revalidate_record_authorities(
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> None:
    expected_identity = inspection.source_statistics.get(
        "identity_map_sha256"
    )
    members = tuple(inspection.accepted_members)
    for member in members:
        _validate_member_identity(member)
        try:
            recipe = inspection.recipes[member.well_key]
        except KeyError:
            raise _fatal(
                f"inspection.accepted_members.{member.record_id}",
                f"missing recipe {member.well_key}",
                "INSPECTION_RECORD_ROLE_MISMATCH",
            ) from None
        pure_components = tuple(
            component
            for component, volume in recipe.component_volumes_ul.items()
            if volume == recipe.total_volume_ul
        )
        expected_role = (
            "pure_reference" if pure_components else "mixture"
        )
        expected_component = (
            pure_components[0] if pure_components else None
        )
        if (
            member.record_role != expected_role
            or member.pure_component != expected_component
        ):
            raise _fatal(
                f"inspection.accepted_members.{member.record_id}",
                "record role or pure component disagrees with recipe",
                "INSPECTION_RECORD_ROLE_MISMATCH",
            )
    if _identity_map_sha256(members) != expected_identity:
        raise _fatal(
            "inspection.accepted_members",
            "identity map differs from inspected contract",
            "INSPECTION_IDENTITY_MISMATCH",
        )
    expected_well = inspection.source_statistics.get(
        "well_target_contract_sha256"
    )
    if _well_target_contract_sha256(inspection) != expected_well:
        raise _fatal(
            "inspection.recipes",
            "well target contract differs from inspected contract",
            "WELL_TARGET_CONTRACT_MISMATCH",
        )
    target_members = tuple(
        member
        for member in inspection.relevant_evidence_members
        if member.source_member.endswith("/Sugar_Concentrations.csv")
        and member.role == "target"
    )
    if len(target_members) != 1:
        raise _fatal(
            "inspection.target_member",
            f"expected one target authority, observed {len(target_members)}",
            "TARGET_AUTHORITY_INVALID",
        )
    target_member = target_members[0]
    archive_path = (
        Path(raw_root).resolve()
        / _PRODUCTION_SOURCE_CONTRACT.archive_name
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            matches = tuple(
                info
                for info in archive.infolist()
                if info.filename == target_member.source_member
            )
            if len(matches) != 1:
                raise _fatal(
                    "inspection.target_member",
                    f"expected one target member, observed {len(matches)}",
                    "TARGET_AUTHORITY_INVALID",
                )
            payload = archive.read(matches[0])
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            "inspection.target_member",
            str(error),
            "TARGET_AUTHORITY_INVALID",
        ) from error
    if (
        len(payload) != target_member.bytes
        or matches[0].CRC != target_member.crc32
        or hashlib.sha256(payload).hexdigest() != target_member.sha256
    ):
        raise _fatal(
            "inspection.target_member",
            "target authority differs from verified member binding",
            "TARGET_AUTHORITY_INVALID",
        )
    _preprocessing_evidence_document(raw_root, inspection)


def _preprocessing_evidence_document(
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> Mapping[str, object]:
    _require_matching_inspection(raw_root, inspection)
    member = next(
        (
            evidence
            for evidence in inspection.relevant_evidence_members
            if evidence.source_member == CLAIM_README
            and evidence.role == "raw_support"
        ),
        None,
    )
    if member is None:
        raise _fatal(
            "preprocessing_evidence.source_member",
            "claim README is missing from verified raw-support members",
            "PREPROCESSING_EVIDENCE_INVALID",
        )
    archive_path = (
        Path(raw_root).resolve()
        / _PRODUCTION_SOURCE_CONTRACT.archive_name
    )
    try:
        with zipfile.ZipFile(archive_path) as archive:
            matches = tuple(
                info
                for info in archive.infolist()
                if info.filename == CLAIM_README
            )
            if len(matches) != 1:
                raise _fatal(
                    "preprocessing_evidence.source_member",
                    f"expected one claim README, observed {len(matches)}",
                    "PREPROCESSING_EVIDENCE_INVALID",
                )
            payload = archive.read(matches[0])
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            "preprocessing_evidence.source_member",
            str(error),
            "PREPROCESSING_EVIDENCE_INVALID",
        ) from error
    observed_sha256 = hashlib.sha256(payload).hexdigest()
    if (
        len(payload) != member.bytes
        or matches[0].CRC != member.crc32
        or observed_sha256 != member.sha256
    ):
        raise _fatal(
            "preprocessing_evidence.source_member",
            "claim README differs from verified member binding",
            "PREPROCESSING_EVIDENCE_INVALID",
        )
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise _fatal(
            "preprocessing_evidence.source_member",
            str(error),
            "PREPROCESSING_EVIDENCE_INVALID",
        ) from error
    claim_prefix = (
        "Note that no pre-processing has been performed such that the user "
        "can decide what is best for the analysis."
    )
    claims = tuple(
        (index, line)
        for index, line in enumerate(lines, start=1)
        if line.rstrip() == claim_prefix
    )
    if len(claims) != 1:
        raise _fatal(
            "preprocessing_evidence.source_line",
            f"expected one claim line, observed {len(claims)}",
            "PREPROCESSING_EVIDENCE_INVALID",
        )
    line_number, exact_line = claims[0]
    document = MappingProxyType(
        {
            "evidence_id": PREPROCESSING_EVIDENCE_ID,
            "evidence_type": "source_dataset_statement",
            "source_member": CLAIM_README,
            "source_member_sha256": observed_sha256,
            "source_member_bytes": len(payload),
            "source_line": line_number,
            "source_line_text_exact": exact_line,
            "claim_text_normalized": exact_line.rstrip(),
            "applies_to_source_directories": [
                HIGH_DIRECTORY.rstrip("/"),
                LOW_DIRECTORY.rstrip("/"),
            ],
            "mapped_preprocessing_status": "known_raw",
            "mapped_preprocessing_steps": [],
            "eligible_for_preprocessing_evaluation": True,
            "scope": (
                "source-declared absence of dataset-level preprocessing "
                "for canonical individual measurement CSVs"
            ),
            "limitations": [
                "does not prove detector-native counts",
                (
                    "does not exclude instrument-firmware or "
                    "acquisition/export-layer operations"
                ),
                "does not make High SNR a clean target for Low SNR",
                (
                    "does not classify prepared analysis artifacts as "
                    "canonical records"
                ),
            ],
        }
    )
    expected_digest = inspection.source_statistics.get(
        "preprocessing_evidence_document_sha256"
    )
    observed_digest = hashlib.sha256(
        b"rpe-sugar-preprocessing-evidence-v1\0"
        + _canonical_json_bytes(document, newline=True)
    ).hexdigest()
    if expected_digest and observed_digest != expected_digest:
        raise _fatal(
            "preprocessing_evidence.document_sha256",
            f"expected {expected_digest}, observed {observed_digest}",
            "PREPROCESSING_EVIDENCE_INVALID",
        )
    return document


def _record_from_member(
    parsed,
    member: _SugarAcceptedMember,
    inspection: SugarMixturesInspection,
) -> RamanRecord:
    try:
        recipe = inspection.recipes[member.well_key]
    except KeyError:
        raise _fatal(
            f"records.{member.record_id}.recipe",
            f"missing recipe {member.well_key}",
            "RECIPE_MISSING",
        ) from None
    acquisition = _normalized_acquisition_metadata(
        member,
        parsed.source_metadata,
    )
    intensity_error = _float32_max_abs_error(parsed.intensity)
    axis_error = _float32_max_abs_error(parsed.wavenumber_cm1)
    wavelength_error = _float32_max_abs_error(parsed.wavelength_nm)
    ceilings = inspection.source_statistics
    for name, observed in (
        ("intensity_float32_max_abs_error", intensity_error),
        ("axis_float32_max_abs_error_cm1", axis_error),
        ("wavelength_float32_max_abs_error_nm", wavelength_error),
    ):
        expected = ceilings.get(name)
        if expected is not None and observed > expected:
            raise _fatal(
                f"records.{member.record_id}.{name}",
                f"expected <= {expected}, observed {observed}",
                "FLOAT32_PRECISION_EXCEEDED",
            )
    record = RamanRecord(
        record_id=member.record_id,
        intensity=_readonly_float32(parsed.intensity),
        wavenumber=_readonly_float32(parsed.wavenumber_cm1),
        meta=SpectrumMetadata(
            dataset_id=DATASET_ID,
            sample_id=member.sample_id,
            instrument=INSTRUMENT,
            excitation_nm=float(acquisition["excitation_nm"]),
            integration_time_s=float(
                acquisition["integration_time_s"]
            ),
            n_accumulations=int(acquisition["n_accumulations"]),
            grating=None,
            detector=None,
            preprocessing_status=PreprocessingStatus.KNOWN_RAW,
            preprocessing_steps=(),
            source_metadata=_source_metadata(
                member,
                recipe,
                inspection,
                acquisition,
            ),
        ),
        targets=_targets(recipe),
        provenance=Provenance(
            source_url=SOURCE_URL,
            license=SOURCE_LICENSE,
            license_status=LicenseStatus.STANDARDIZED,
            sha256=member.sha256,
            retrieved_date=RETRIEVED_DATE,
            source_artifact=member.source_member,
        ),
    )
    if axis_id(record.wavenumber) != inspection.core_axis_id:
        raise _fatal(
            f"records.{member.record_id}.core_axis_id",
            "record wavenumber differs from inspected core axis",
            "AXIS_CONTRACT_MISMATCH",
        )
    return validate_record(record)


def iter_sugar_mixtures_records(
    raw_root: Path,
    inspection: SugarMixturesInspection | None = None,
) -> Iterator[RamanRecord]:
    raw_root = Path(raw_root).resolve()
    if inspection is None:
        inspection = inspect_sugar_mixtures(raw_root)
    _require_matching_inspection(raw_root, inspection)
    _revalidate_record_authorities(raw_root, inspection)
    members = tuple(inspection.accepted_members)
    if tuple(member.record_id for member in members) != tuple(
        sorted(
            (member.record_id for member in members),
            key=lambda value: value.encode("utf-8"),
        )
    ):
        raise _fatal(
            "inspection.accepted_members",
            "members must be sorted by UTF-8 record ID",
            "INSPECTION_STALE",
        )
    archive_path = raw_root / _PRODUCTION_SOURCE_CONTRACT.archive_name
    expected_points = int(
        inspection.source_statistics["points_per_record"]
    )
    target_digest = hashlib.sha256(
        b"rpe-sugar-record-target-map-v1\0"
    )
    preprocessing_digest = hashlib.sha256(
        b"rpe-sugar-record-preprocessing-map-v1\0"
    )
    precision_maxima = {
        "intensity_float32_max_abs_error": 0.0,
        "axis_float32_max_abs_error_cm1": 0.0,
        "wavelength_float32_max_abs_error_nm": 0.0,
    }
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                infos_by_name.setdefault(info.filename, []).append(info)
            for member in members:
                matches = infos_by_name.get(member.source_member, [])
                if len(matches) != 1:
                    raise _fatal(
                        f"archive.members.{member.source_member}",
                        (
                            "expected one member binding, "
                            f"observed {len(matches)}"
                        ),
                        "MEMBER_BINDING_MISMATCH",
                    )
                parsed = _reparse_verified_member(
                    raw_root,
                    member,
                    archive=archive,
                    info=matches[0],
                    expected_points=expected_points,
                )
                record = _record_from_member(
                    parsed,
                    member,
                    inspection,
                )
                precision_maxima["intensity_float32_max_abs_error"] = max(
                    precision_maxima["intensity_float32_max_abs_error"],
                    _float32_max_abs_error(parsed.intensity),
                )
                precision_maxima["axis_float32_max_abs_error_cm1"] = max(
                    precision_maxima["axis_float32_max_abs_error_cm1"],
                    _float32_max_abs_error(parsed.wavenumber_cm1),
                )
                precision_maxima["wavelength_float32_max_abs_error_nm"] = max(
                    precision_maxima["wavelength_float32_max_abs_error_nm"],
                    _float32_max_abs_error(parsed.wavelength_nm),
                )
                target_entry = _canonical_json_bytes(
                    {
                        "record_id": record.record_id,
                        "concentrations": dict(
                            record.targets.concentrations
                        ),
                    },
                    newline=True,
                )
                _update_length_prefixed(target_digest, target_entry)
                status_entry = _canonical_json_bytes(
                    {
                        "record_id": record.record_id,
                        "preprocessing_status": (
                            record.meta.preprocessing_status.value
                        ),
                        "preprocessing_steps": [],
                        "eligible_for_preprocessing_evaluation": (
                            record.meta.eligible_for_preprocessing_evaluation
                        ),
                        "source_preprocessing_state": (
                            record.meta.source_metadata[
                                "source_preprocessing_state"
                            ]
                        ),
                        "source_preprocessing_evidence_id": (
                            record.meta.source_metadata[
                                "source_preprocessing_evidence_id"
                            ]
                        ),
                    },
                    newline=True,
                )
                _update_length_prefixed(
                    preprocessing_digest,
                    status_entry,
                )
                yield record
                del parsed
            expected_target_digest = inspection.source_statistics.get(
                "record_target_map_sha256"
            )
            if (
                expected_target_digest
                and target_digest.hexdigest() != expected_target_digest
            ):
                raise _fatal(
                    "records.target_map_sha256",
                    (
                        f"expected {expected_target_digest}, "
                        f"observed {target_digest.hexdigest()}"
                    ),
                    "RECORD_TARGET_CONTRACT_MISMATCH",
                )
            expected_status_digest = inspection.source_statistics.get(
                "record_preprocessing_map_sha256"
            )
            if (
                expected_status_digest
                and preprocessing_digest.hexdigest()
                != expected_status_digest
            ):
                raise _fatal(
                    "records.preprocessing_map_sha256",
                    (
                        f"expected {expected_status_digest}, "
                        f"observed {preprocessing_digest.hexdigest()}"
                    ),
                    "RECORD_PREPROCESSING_CONTRACT_MISMATCH",
                )
            for name, observed in precision_maxima.items():
                expected = inspection.source_statistics.get(name)
                if expected is not None and observed != expected:
                    raise _fatal(
                        f"records.{name}",
                        f"expected {expected}, observed {observed}",
                        "FLOAT32_PRECISION_CONTRACT_MISMATCH",
                    )
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _fatal(
            "records.archive",
            str(error),
            "MEMBER_BINDING_MISMATCH",
        ) from error


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_sugar_core_staged(
    raw_root: Path,
    core_path: Path,
    inspection: SugarMixturesInspection,
    *,
    timings: dict[str, float] | None = None,
) -> SugarMixturesDatasetSummary:
    raw_root = Path(raw_root).resolve()
    core_path = Path(core_path)
    _require_matching_inspection(raw_root, inspection)
    started = time.perf_counter()
    write_dataset(
        iter_sugar_mixtures_records(raw_root, inspection),
        core_path,
        dataset_id=DATASET_ID,
        class_labels={},
        concentration_units={
            "fructose_nominal_mol_l": "mol/L",
            "glucose_nominal_mol_l": "mol/L",
            "maltose_nominal_mol_l": "mol/L",
            "sucrose_nominal_mol_l": "mol/L",
        },
    )
    if timings is not None:
        timings["core_build_seconds"] = time.perf_counter() - started
    validation_started = time.perf_counter()
    validation = validate_dataset(core_path, verify_checksums=True)
    second_validation = _inspect_dataset(
        core_path,
        verify_checksums=True,
        exhaustive_arrays=True,
    ).summary
    if validation != second_validation:
        raise _fatal(
            "core.validation",
            "independent exhaustive validations differ",
            "CORE_VALIDATION_MISMATCH",
        )
    if (
        validation.dataset_id != DATASET_ID
        or validation.record_count
        != inspection.source_statistics["record_count"]
        or validation.axis_group_count != 1
        or dict(validation.preprocessing_status_counts)
        != {"known_raw": validation.record_count, "known_corrected": 0, "unknown": 0}
        or dict(validation.target_presence_counts)
        != {
            "baseline": 0,
            "class_label": 0,
            "clean": 0,
            "concentration": 0,
            "concentrations": validation.record_count,
            "peaks": 0,
        }
    ):
        raise _fatal(
            "core.validation",
            "validated core summary differs from Sugar contract",
            "CORE_VALIDATION_MISMATCH",
        )
    files = MappingProxyType(
        {
            name: MappingProxyType(
                {
                    "bytes": (core_path / name).stat().st_size,
                    "sha256": _sha256_path(core_path / name),
                }
            )
            for name in sorted(DATASET_FILES)
        }
    )
    if timings is not None:
        timings["validation_seconds"] = (
            time.perf_counter() - validation_started
        )
    return SugarMixturesDatasetSummary(
        path=core_path,
        dataset_id=DATASET_ID,
        record_count=validation.record_count,
        eligible_records=validation.preprocessing_status_counts["known_raw"],
        axis_group_count=validation.axis_group_count,
        output_bytes=sum(int(details["bytes"]) for details in files.values()),
        files=files,
    )


def _comparison_fatal(path: str, reason: str) -> SugarMixturesValidationError:
    return _fatal(path, reason, "SOURCE_COMPARISON_MISMATCH")


def _read_comparison_json(path: Path) -> object:
    try:
        payload = path.read_bytes()
        value = json.loads(
            payload,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {constant}")
            ),
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise _comparison_fatal(
            f"comparison.dataset.{path.name}",
            str(error),
        ) from error
    if payload != _canonical_json_bytes(value, newline=True):
        raise _comparison_fatal(
            f"comparison.dataset.{path.name}",
            "document is not canonical JSON",
        )
    return value


def _read_comparison_records(path: Path) -> tuple[Mapping[str, object], ...]:
    try:
        lines = path.read_bytes().splitlines(keepends=True)
    except OSError as error:
        raise _comparison_fatal(
            "comparison.dataset.records.jsonl",
            str(error),
        ) from error
    result = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(
                line,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant {constant}")
                ),
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ) as error:
            raise _comparison_fatal(
                f"comparison.dataset.records.jsonl[{index}]",
                str(error),
            ) from error
        if (
            not isinstance(value, Mapping)
            or line != _canonical_json_bytes(value, newline=True)
        ):
            raise _comparison_fatal(
                f"comparison.dataset.records.jsonl[{index}]",
                "record is not a canonical object",
            )
        result.append(value)
    return tuple(result)


def _comparison_record_id_from_basename(
    basename: str,
    condition: str,
) -> str:
    pattern = (
        r"^Sugar_Concentration_Test_"
        if condition == "high_snr"
        else r"^Sugar_Concentration_Test_Fast_"
    )
    match = re.fullmatch(
        pattern
        + (
            r"(?P<samp>\d+)_(?P<row>[A-H])(?P<column>\d+)_"
            r"(?P<plate>\d+)_RD(?P<round>\d+)_M(?P<measurement>\d+)_"
            r"R(?P<repetition>\d+)$"
        ),
        basename.removesuffix(".csv"),
    )
    if match is None:
        raise _comparison_fatal(
            f"comparison.source.{basename}",
            "prepared/oracle filename does not match source grammar",
        )
    values = {
        name: int(value) if name != "row" else value
        for name, value in match.groupdict().items()
    }
    return (
        f"{condition}-s{values['samp']:03d}-"
        f"{values['row'].lower()}{values['column']:02d}"
        f"-p{values['plate']:02d}-r{values['round']:02d}"
        f"-m{values['measurement']:02d}-rep{values['repetition']:02d}"
    )


def _comparison_recipe_values(
    recipe: _SugarRecipe,
) -> tuple[list[int], list[float], dict[str, float]]:
    volumes = [recipe.component_volumes_ul[name] for name in COMPONENTS]
    fractions = [value / recipe.total_volume_ul for value in volumes]
    targets = {
        f"{name}_nominal_mol_l": (
            recipe.component_volumes_ul[name] / recipe.total_volume_ul
        )
        for name in TARGET_COMPONENTS
    }
    return volumes, fractions, targets


def _read_verified_evidence_payload(
    archive: zipfile.ZipFile,
    infos_by_name: Mapping[str, list[zipfile.ZipInfo]],
    evidence_by_name: Mapping[str, object],
    source_member: str,
) -> bytes:
    member = evidence_by_name.get(source_member)
    matches = infos_by_name.get(source_member, ())
    if member is None or len(matches) != 1:
        raise _comparison_fatal(
            f"comparison.evidence.{source_member}",
            "evidence binding is missing or duplicated",
        )
    info = matches[0]
    try:
        payload = archive.read(info)
    except (OSError, RuntimeError, zipfile.BadZipFile, KeyError) as error:
        raise _comparison_fatal(
            f"comparison.evidence.{source_member}",
            str(error),
        ) from error
    if (
        len(payload) != member.bytes
        or info.CRC != member.crc32
        or hashlib.sha256(payload).hexdigest() != member.sha256
    ):
        raise _comparison_fatal(
            f"comparison.evidence.{source_member}",
            "evidence bytes differ from inspection binding",
        )
    return payload


def _compare_sugar_dataset_to_source(
    raw_root: Path,
    dataset_path: Path,
    inspection: SugarMixturesInspection,
) -> _SugarComparisonSummary:
    raw_root = Path(raw_root).resolve()
    dataset_path = Path(dataset_path)
    _require_matching_inspection(raw_root, inspection)
    expected_record_count = len(inspection.accepted_members)
    expected_points_per_record = inspection.source_statistics.get(
        "points_per_record"
    )
    expected_prepared_rows = sum(
        len(inspection.view_record_ids[view_id])
        for view_id in (
            "high_snr",
            "high_snr_no_refs",
            "low_snr",
            "low_snr_no_refs",
        )
    )
    if (
        inspection.source_statistics.get("record_count")
        != expected_record_count
        or isinstance(expected_points_per_record, bool)
        or not isinstance(expected_points_per_record, int)
        or expected_points_per_record <= 0
    ):
        raise _comparison_fatal(
            "comparison.counts",
            "inspection comparison counts are internally inconsistent",
        )
    manifest = _read_comparison_json(dataset_path / "dataset.json")
    if not isinstance(manifest, Mapping):
        raise _comparison_fatal(
            "comparison.dataset.dataset.json",
            "manifest must be an object",
        )
    records = _read_comparison_records(dataset_path / "records.jsonl")
    expected_units = {
        "fructose_nominal_mol_l": "mol/L",
        "glucose_nominal_mol_l": "mol/L",
        "maltose_nominal_mol_l": "mol/L",
        "sucrose_nominal_mol_l": "mol/L",
    }
    if manifest.get("concentration_units") != expected_units:
        raise _comparison_fatal(
            "comparison.dataset.concentration_units",
            "manifest target units differ from contract",
        )
    if (
        manifest.get("dataset_id") != DATASET_ID
        or manifest.get("record_count") != len(inspection.accepted_members)
        or manifest.get("class_labels") != {}
        or manifest.get("concentration_unit") is not None
        or not isinstance(manifest.get("source_artifacts"), list)
    ):
        raise _comparison_fatal(
            "comparison.dataset.dataset.json",
            "manifest fixed fields differ from contract",
        )
    members_by_record_id = {
        member.record_id: member for member in inspection.accepted_members
    }
    if tuple(record.get("record_id") for record in records) != tuple(
        sorted(members_by_record_id)
    ):
        raise _comparison_fatal(
            "comparison.dataset.records.jsonl",
            "record IDs differ from inspection order",
        )
    expected_provenance = {
        (
            SOURCE_URL,
            SOURCE_LICENSE,
            "standardized",
            member.sha256,
            RETRIEVED_DATE.isoformat(),
            member.source_member,
        )
        for member in inspection.accepted_members
    }
    manifest_provenance = {
        (
            artifact.get("source_url"),
            artifact.get("license"),
            artifact.get("license_status"),
            artifact.get("sha256"),
            artifact.get("retrieved_date"),
            artifact.get("source_artifact"),
        )
        for artifact in manifest["source_artifacts"]
        if isinstance(artifact, Mapping)
    }
    if manifest_provenance != expected_provenance:
        raise _comparison_fatal(
            "comparison.dataset.source_artifacts",
            "manifest source artifacts differ from inspection",
        )

    source_rows = 0
    source_points = 0
    consolidated_cells = 0
    prepared_abundance_rows = 0
    prepared_endmembers = 0
    intensity_mismatches = 0
    axis_mismatches = 0
    target_mismatches = 0
    metadata_mismatches = 0
    preprocessing_mismatches = 0
    companion_mismatches = 0
    archive_path = raw_root / _PRODUCTION_SOURCE_CONTRACT.archive_name
    evidence_by_name = {
        member.source_member: member
        for member in inspection.relevant_evidence_members
    }
    source_intensity_by_record_id: dict[str, np.ndarray] = {}
    source_axis = None
    stored_axes: dict[str, np.ndarray] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive, h5py.File(
            dataset_path / "arrays.h5",
            "r",
        ) as arrays:
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                infos_by_name.setdefault(info.filename, []).append(info)
            for document in records:
                record_id = document["record_id"]
                member = members_by_record_id[record_id]
                matches = infos_by_name.get(member.source_member, ())
                if len(matches) != 1:
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.identity",
                        "source binding is missing or duplicated",
                    )
                parsed = _reparse_verified_member(
                    raw_root,
                    member,
                    archive=archive,
                    info=matches[0],
                    expected_points=int(
                        inspection.source_statistics["points_per_record"]
                    ),
                )
                array_ref = document.get("array_ref")
                if not isinstance(array_ref, Mapping):
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.identity",
                        "array_ref is invalid",
                    )
                try:
                    group = arrays["axes"][array_ref["axis_id"]]
                    stored_intensity = np.asarray(
                        group["intensity"][array_ref["row"]]
                    )
                    if array_ref["axis_id"] not in stored_axes:
                        stored_axes[array_ref["axis_id"]] = np.asarray(
                            group["wavenumber"][:]
                        )
                    stored_axis = stored_axes[array_ref["axis_id"]]
                except (KeyError, TypeError, ValueError, OSError) as error:
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.identity",
                        str(error),
                    ) from error
                expected_intensity = np.ascontiguousarray(
                    parsed.intensity,
                    dtype="<f4",
                )
                expected_axis = np.ascontiguousarray(
                    parsed.wavenumber_cm1,
                    dtype="<f4",
                )
                if not np.array_equal(stored_intensity, expected_intensity):
                    intensity_mismatches += 1
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.intensity",
                        "stored intensity differs from direct ZIP source",
                    )
                if not np.array_equal(stored_axis, expected_axis):
                    axis_mismatches += 1
                    raise _comparison_fatal(
                        "comparison.dataset.arrays.h5",
                        "stored axis differs from direct ZIP source",
                    )
                expected_record = _record_from_member(
                    parsed,
                    member,
                    inspection,
                )
                meta = document.get("meta")
                source_metadata = (
                    meta.get("source_metadata")
                    if isinstance(meta, Mapping)
                    else None
                )
                if (
                    document.get("record_id") != member.record_id
                    or not isinstance(meta, Mapping)
                    or meta.get("sample_id") != member.sample_id
                    or not isinstance(source_metadata, Mapping)
                    or source_metadata.get("source_acquisition_id")
                    != member.acquisition_id
                    or source_metadata.get("source_member")
                    != member.source_member
                    or source_metadata.get("source_samp")
                    != member.source_samp
                ):
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.identity",
                        "stored identity differs from inspection",
                    )
                if source_metadata.get(
                    "source_acquisition_metadata"
                ) != dict(expected_record.meta.source_metadata[
                    "source_acquisition_metadata"
                ]):
                    metadata_mismatches += 1
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.metadata",
                        "normalized acquisition metadata differs",
                    )
                recipe = inspection.recipes[member.well_key]
                volume_fields = {
                    f"source_{name}_volume_ul": recipe.component_volumes_ul[name]
                    for name in COMPONENTS
                }
                if (
                    any(source_metadata.get(key) != value for key, value in volume_fields.items())
                    or source_metadata.get("source_total_volume_ul")
                    != recipe.total_volume_ul
                    or source_metadata.get("source_component_fractions")
                    != dict(recipe.component_fractions)
                ):
                    target_mismatches += 1
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.recipe",
                        "stored source recipe differs from target authority",
                    )
                targets = document.get("targets")
                expected_targets = dict(expected_record.targets.concentrations)
                if (
                    not isinstance(targets, Mapping)
                    or targets.get("concentrations") != expected_targets
                    or targets.get("concentration") is not None
                ):
                    target_mismatches += 1
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.targets",
                        "stored named targets differ from recipe",
                    )
                if (
                    meta.get("preprocessing_status") != "known_raw"
                    or meta.get("preprocessing_steps") != []
                    or source_metadata.get("source_preprocessing_state")
                    != PREPROCESSING_STATE
                    or source_metadata.get("source_preprocessing_evidence_id")
                    != PREPROCESSING_EVIDENCE_ID
                ):
                    preprocessing_mismatches += 1
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.preprocessing",
                        "stored preprocessing evidence differs",
                    )
                provenance = document.get("provenance")
                if (
                    not isinstance(provenance, Mapping)
                    or provenance.get("source_url") != SOURCE_URL
                    or provenance.get("license") != SOURCE_LICENSE
                    or provenance.get("license_status") != "standardized"
                    or provenance.get("sha256") != member.sha256
                    or provenance.get("source_artifact") != member.source_member
                    or provenance.get("retrieved_date")
                    != RETRIEVED_DATE.isoformat()
                ):
                    raise _comparison_fatal(
                        f"comparison.records.{record_id}.provenance",
                        "stored provenance differs from source binding",
                    )
                source_intensity_by_record_id[record_id] = np.array(
                    parsed.intensity,
                    copy=True,
                )
                if source_axis is None:
                    source_axis = np.array(
                        parsed.wavenumber_cm1,
                        copy=True,
                    )
                source_rows += 1
                source_points += parsed.intensity.size
                del parsed

            for condition, spectra_basename, metadata_basename in (
                (
                    "high_snr",
                    "Sugar_Concentration_Test_ALL_spectra.csv",
                    "Sugar_Concentration_Test_ALL_metadata.csv",
                ),
                (
                    "low_snr",
                    "Sugar_Concentration_Test_Fast_ALL_spectra.csv",
                    "Sugar_Concentration_Test_Fast_ALL_metadata.csv",
                ),
            ):
                spectra_member = _ORACLE_BY_BASENAME[spectra_basename]
                metadata_member = _ORACLE_BY_BASENAME[metadata_basename]
                spectra_payload = _read_verified_evidence_payload(
                    archive,
                    infos_by_name,
                    evidence_by_name,
                    spectra_member,
                )
                metadata_payload = _read_verified_evidence_payload(
                    archive,
                    infos_by_name,
                    evidence_by_name,
                    metadata_member,
                )
                try:
                    spectra_rows = list(
                        csv.reader(
                            io.StringIO(spectra_payload.decode("utf-8"))
                        )
                    )
                    metadata_rows = list(
                        csv.DictReader(
                            io.StringIO(metadata_payload.decode("utf-8"))
                        )
                    )
                except (UnicodeDecodeError, csv.Error) as error:
                    raise _comparison_fatal(
                        f"comparison.oracle.{condition}",
                        str(error),
                    ) from error
                header = spectra_rows[0]
                if header[0] != "cm-1":
                    raise _comparison_fatal(
                        f"comparison.oracle.{condition}",
                        "first spectra column must be cm-1",
                    )
                names = header[1:]
                if names != [row["filename"] for row in metadata_rows]:
                    raise _comparison_fatal(
                        f"comparison.oracle.{condition}",
                        "spectra and metadata filename order differ",
                    )
                axis = np.array(
                    [float(row[0]) for row in spectra_rows[1:]],
                    dtype=np.float64,
                )
                columns = {
                    name: np.array(
                        [
                            float(row[index + 1])
                            for row in spectra_rows[1:]
                        ],
                        dtype=np.float64,
                    )
                    for index, name in enumerate(names)
                }
                for metadata_row in metadata_rows:
                    basename = metadata_row["filename"]
                    record_id = _comparison_record_id_from_basename(
                        basename,
                        condition,
                    )
                    member = members_by_record_id[record_id]
                    source_intensity = source_intensity_by_record_id[
                        record_id
                    ]
                    if (
                        not np.array_equal(columns[basename], source_intensity)
                        or source_axis is None
                        or not np.array_equal(axis, source_axis)
                    ):
                        companion_mismatches += 1
                        raise _comparison_fatal(
                            f"comparison.oracle.{condition}",
                            "consolidated values differ from direct source",
                        )
                    recipe = inspection.recipes[member.well_key]
                    expected_volumes = [
                        *[
                            recipe.component_volumes_ul[name]
                            for name in COMPONENTS
                        ],
                        recipe.total_volume_ul,
                    ]
                    observed_volumes = [
                        int(metadata_row[name])
                        for name in (
                            "Sucrose [ul]",
                            "Fructose [ul]",
                            "Maltose [ul]",
                            "Glucose [ul]",
                            "Water [ul]",
                            "Total Volume [ul]",
                        )
                    ]
                    if observed_volumes != expected_volumes:
                        target_mismatches += 1
                        raise _comparison_fatal(
                            f"comparison.oracle.{condition}",
                            "consolidated recipe differs from source table",
                        )
                    consolidated_cells += source_intensity.size

            for view in (
                "High SNR",
                "High SNR (no refs)",
                "Low SNR",
                "Low SNR (no refs)",
            ):
                condition = (
                    "high_snr" if view.startswith("High") else "low_snr"
                )
                prefix = f"{PREPARED_PREFIX}{view}/"
                payloads = {
                    basename: _read_verified_evidence_payload(
                        archive,
                        infos_by_name,
                        evidence_by_name,
                        f"{prefix}{basename}",
                    )
                    for basename in (
                        "data.pkl",
                        "spectral_axis.pkl",
                        "gt_abundance_image.pkl",
                        "gt_endmembers.pkl",
                        "metadata.csv",
                    )
                }
                try:
                    prepared_data = np.asarray(
                        pd.read_pickle(io.BytesIO(payloads["data.pkl"]))
                    )
                    prepared_axis = np.asarray(
                        pd.read_pickle(
                            io.BytesIO(payloads["spectral_axis.pkl"])
                        )
                    )
                    abundance = np.asarray(
                        pd.read_pickle(
                            io.BytesIO(
                                payloads["gt_abundance_image.pkl"]
                            )
                        )
                    )
                    endmembers = np.asarray(
                        pd.read_pickle(
                            io.BytesIO(payloads["gt_endmembers.pkl"])
                        )
                    )
                    prepared_metadata = list(
                        csv.DictReader(
                            io.StringIO(
                                payloads["metadata.csv"].decode("utf-8")
                            )
                        )
                    )
                except (
                    AttributeError,
                    EOFError,
                    ImportError,
                    ModuleNotFoundError,
                    ValueError,
                    UnicodeDecodeError,
                    csv.Error,
                ) as error:
                    raise _comparison_fatal(
                        f"comparison.prepared.{view}",
                        str(error),
                    ) from error
                if not prepared_metadata:
                    raise _comparison_fatal(
                        f"comparison.prepared.{view}.membership",
                        "prepared metadata is empty",
                    )
                identifier_column = next(
                    (
                        name
                        for name in ("", "Unnamed: 0")
                        if name in prepared_metadata[0]
                    ),
                    None,
                )
                if identifier_column is None:
                    raise _comparison_fatal(
                        f"comparison.prepared.{view}.membership",
                        "prepared metadata identifier column is missing",
                    )
                filenames = [
                    row[identifier_column] for row in prepared_metadata
                ]
                expected_ids = [
                    _comparison_record_id_from_basename(
                        basename,
                        condition,
                    )
                    for basename in filenames
                ]
                expected_view_id = (
                    "high_snr_no_refs"
                    if view == "High SNR (no refs)"
                    else "low_snr_no_refs"
                    if view == "Low SNR (no refs)"
                    else "high_snr"
                    if view == "High SNR"
                    else "low_snr"
                )
                expected_view_ids = tuple(
                    inspection.view_record_ids[expected_view_id]
                )
                if (
                    len(expected_ids) != len(set(expected_ids))
                    or set(expected_ids) != set(expected_view_ids)
                ):
                    companion_mismatches += 1
                    raise _comparison_fatal(
                        f"comparison.prepared.{view}.membership",
                        "prepared members differ from inspection view",
                    )
                expected_data = np.stack(
                    [
                        source_intensity_by_record_id[record_id]
                        for record_id in expected_ids
                    ]
                )
                if (
                    not np.array_equal(prepared_data, expected_data)
                    or source_axis is None
                    or not np.array_equal(
                        prepared_axis.astype(np.float64),
                        source_axis,
                    )
                ):
                    companion_mismatches += 1
                    raise _comparison_fatal(
                        f"comparison.prepared.{view}.data",
                        "prepared data or axis differs from source",
                    )
                expected_abundance = np.stack(
                    [
                        np.array(
                            _comparison_recipe_values(
                                inspection.recipes[
                                    members_by_record_id[record_id].well_key
                                ]
                            )[1],
                            dtype=np.float64,
                        )
                        for record_id in expected_ids
                    ]
                )
                if not np.array_equal(
                    abundance.astype(np.float64),
                    expected_abundance,
                ):
                    target_mismatches += 1
                    raise _comparison_fatal(
                        f"comparison.prepared.{view}.abundance",
                        "prepared abundance differs from source recipe",
                    )
                prepared_abundance_rows += len(expected_ids)
                if view in {"High SNR", "Low SNR"}:
                    pure_medians = []
                    for component in COMPONENTS:
                        values = [
                            source_intensity_by_record_id[member.record_id]
                            for member in inspection.accepted_members
                            if member.condition == condition
                            and member.pure_component == component
                        ]
                        pure_medians.append(
                            np.median(np.stack(values), axis=0)
                        )
                    if not np.array_equal(
                        endmembers.astype(np.float64),
                        np.stack(pure_medians),
                    ):
                        companion_mismatches += 1
                        raise _comparison_fatal(
                            f"comparison.prepared.{view}.endmembers",
                            "prepared endmembers differ from source medians",
                        )
                    prepared_endmembers += len(COMPONENTS)
    except SugarMixturesValidationError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise _comparison_fatal(
            "comparison.source",
            str(error),
        ) from error
    summary = _SugarComparisonSummary(
        source_rows_compared=source_rows,
        source_points_compared=source_points,
        consolidated_intensity_cells_compared=consolidated_cells,
        prepared_abundance_rows_compared=prepared_abundance_rows,
        prepared_endmembers_compared=prepared_endmembers,
        intensity_mismatches=intensity_mismatches,
        axis_mismatches=axis_mismatches,
        target_mismatches=target_mismatches,
        metadata_mismatches=metadata_mismatches,
        preprocessing_status_mismatches=preprocessing_mismatches,
        companion_reference_mismatches=companion_mismatches,
    )
    if (
        summary.source_rows_compared != expected_record_count
        or summary.source_points_compared
        != expected_record_count * expected_points_per_record
        or summary.consolidated_intensity_cells_compared
        != expected_record_count * expected_points_per_record
        or summary.prepared_abundance_rows_compared
        != expected_prepared_rows
        or summary.prepared_endmembers_compared
        != 2 * len(COMPONENTS)
        or any(
            getattr(summary, name) != 0
            for name in (
                "intensity_mismatches",
                "axis_mismatches",
                "target_mismatches",
                "metadata_mismatches",
                "preprocessing_status_mismatches",
                "companion_reference_mismatches",
            )
        )
    ):
        raise _comparison_fatal(
            "comparison.counts",
            "comparison counts or mismatch counters differ from contract",
        )
    return summary


def _write_views(
    path: Path,
    inspection: SugarMixturesInspection,
) -> SugarMixturesCompanionSummary:
    from rpe.io.sugar_mixtures_companions import _write_views as writer

    return writer(path, inspection)


def _validate_views(
    path: Path,
    inspection: SugarMixturesInspection,
    *,
    record_ids: set[str],
) -> Mapping[str, object]:
    from rpe.io.sugar_mixtures_companions import _validate_views as validator

    return validator(path, inspection, record_ids=record_ids)


def _write_derived_endmembers(
    path: Path,
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> SugarMixturesCompanionSummary:
    from rpe.io.sugar_mixtures_companions import (
        _write_derived_endmembers as writer,
    )

    return writer(path, raw_root, inspection)


def _validate_derived_endmembers(
    path: Path,
    raw_root: Path,
    inspection: SugarMixturesInspection,
    *,
    record_ids: set[str],
) -> Mapping[str, object]:
    from rpe.io.sugar_mixtures_companions import (
        _validate_derived_endmembers as validator,
    )

    return validator(
        path,
        raw_root,
        inspection,
        record_ids=record_ids,
    )


def _write_auxiliary_axes(
    path: Path,
    inspection: SugarMixturesInspection,
) -> SugarMixturesCompanionSummary:
    from rpe.io.sugar_mixtures_companions import (
        _write_auxiliary_axes as writer,
    )

    return writer(path, inspection)


def _validate_auxiliary_axes(
    path: Path,
    inspection: SugarMixturesInspection,
) -> Mapping[str, object]:
    from rpe.io.sugar_mixtures_companions import (
        _validate_auxiliary_axes as validator,
    )

    return validator(path, inspection)


def _write_acquisition_json(
    path: Path,
    raw_root: Path,
    inspection: SugarMixturesInspection,
) -> SugarMixturesCompanionSummary:
    from rpe.io.sugar_mixtures_companions import (
        _write_acquisition_json as writer,
    )

    return writer(path, raw_root, inspection)


def _validate_acquisition_json(
    path: Path,
    raw_root: Path,
    inspection: SugarMixturesInspection,
    *,
    record_ids: set[str],
) -> Mapping[str, object]:
    from rpe.io.sugar_mixtures_companions import (
        _validate_acquisition_json as validator,
    )

    return validator(
        path,
        raw_root,
        inspection,
        record_ids=record_ids,
    )


def _build_receipt_document(**kwargs) -> Mapping[str, object]:
    from rpe.io.sugar_mixtures_receipt import (
        _build_receipt_document as builder,
    )

    return builder(**kwargs)


def _write_receipt(
    path: Path,
    document: Mapping[str, object],
) -> tuple[int, str]:
    from rpe.io.sugar_mixtures_receipt import _write_receipt as writer

    return writer(path, document)


def _validate_receipt(path: Path, **kwargs) -> Mapping[str, object]:
    from rpe.io.sugar_mixtures_receipt import _validate_receipt as validator

    return validator(path, **kwargs)


def _build_sugar_mixtures_staged(
    raw_root: Path,
    staging_parent: Path,
    inspection: SugarMixturesInspection,
    *,
    timings: dict[str, float] | None = None,
) -> _SugarStagedData:
    raw_root = Path(raw_root).resolve()
    staging_parent = Path(staging_parent)
    staged_timings: dict[str, float] = {}
    _require_matching_inspection(raw_root, inspection)
    if staging_parent.exists():
        if staging_parent.is_symlink() or not staging_parent.is_dir():
            raise _fatal(
                "staging_parent",
                "must be a non-symlink directory",
                "STAGING_PARENT_INVALID",
            )
        if any(staging_parent.iterdir()):
            raise _fatal(
                "staging_parent",
                "must be empty before staged build",
                "STAGING_PARENT_NOT_EMPTY",
            )
    else:
        staging_parent.mkdir(parents=True)

    core_path = staging_parent / DATASET_ID
    companion_paths = {
        "views": staging_parent / "sugar_mixtures_raman_views.json",
        "derived_endmembers": (
            staging_parent
            / "sugar_mixtures_raman_derived_reference_endmembers.h5"
        ),
        "auxiliary_axes": (
            staging_parent / "sugar_mixtures_raman_auxiliary_axes.h5"
        ),
        "acquisition_json": (
            staging_parent
            / "sugar_mixtures_raman_acquisition_json_text.jsonl"
        ),
    }

    core = _build_sugar_core_staged(
        raw_root,
        core_path,
        inspection,
        timings=staged_timings,
    )
    record_ids = {
        member.record_id for member in inspection.accepted_members
    }

    def timed(name: str, operation):
        started = time.perf_counter()
        result = operation()
        staged_timings[name] = time.perf_counter() - started
        return result

    views = timed(
        "views_build_seconds",
        lambda: _write_views(companion_paths["views"], inspection),
    )
    derived_endmembers = timed(
        "derived_endmembers_build_seconds",
        lambda: _write_derived_endmembers(
            companion_paths["derived_endmembers"],
            raw_root,
            inspection,
        ),
    )
    auxiliary_axes = timed(
        "auxiliary_axes_build_seconds",
        lambda: _write_auxiliary_axes(
            companion_paths["auxiliary_axes"],
            inspection,
        ),
    )
    acquisition_json = timed(
        "acquisition_json_build_seconds",
        lambda: _write_acquisition_json(
            companion_paths["acquisition_json"],
            raw_root,
            inspection,
        ),
    )
    companions = MappingProxyType(
        {
            "views": views,
            "derived_endmembers": derived_endmembers,
            "auxiliary_axes": auxiliary_axes,
            "acquisition_json": acquisition_json,
        }
    )

    validation_seconds = staged_timings.get("validation_seconds", 0.0)
    validation_started = time.perf_counter()
    _validate_views(
        views.path,
        inspection,
        record_ids=record_ids,
    )
    _validate_derived_endmembers(
        derived_endmembers.path,
        raw_root,
        inspection,
        record_ids=record_ids,
    )
    _validate_auxiliary_axes(auxiliary_axes.path, inspection)
    _validate_acquisition_json(
        acquisition_json.path,
        raw_root,
        inspection,
        record_ids=record_ids,
    )
    comparison = _compare_sugar_dataset_to_source(
        raw_root,
        core.path,
        inspection,
    )
    validation_seconds += time.perf_counter() - validation_started
    validation_started = time.perf_counter()
    receipt_document = _build_receipt_document(
        raw_root=raw_root,
        inspection=inspection,
        core=core,
        companions=companions,
        comparison=comparison,
    )
    validation_seconds += time.perf_counter() - validation_started
    receipt_path = staging_parent / "sugar_mixtures_raman_conversion.json"
    receipt_started = time.perf_counter()
    receipt_bytes, receipt_sha256 = _write_receipt(
        receipt_path,
        receipt_document,
    )
    staged_timings["receipt_build_seconds"] = (
        time.perf_counter() - receipt_started
    )
    validation_started = time.perf_counter()
    _validate_receipt(
        receipt_path,
        raw_root=raw_root,
        inspection=inspection,
        core_path=core.path,
        companion_paths=companion_paths,
        comparison=comparison,
    )
    validation_seconds += time.perf_counter() - validation_started
    staged_timings["validation_seconds"] = validation_seconds

    output_hashes = MappingProxyType(
        {
            path.relative_to(staging_parent).as_posix(): _sha256_path(path)
            for path in sorted(
                (
                    candidate
                    for candidate in staging_parent.rglob("*")
                    if candidate.is_file()
                ),
                key=lambda value: value.relative_to(
                    staging_parent
                ).as_posix(),
            )
        }
    )
    if len(output_hashes) != 10:
        raise _fatal(
            "staging_parent.files",
            f"expected ten regular files, observed {len(output_hashes)}",
            "STAGED_FILE_COUNT_MISMATCH",
        )
    staged = _SugarStagedData(
        core=core,
        views=views,
        derived_endmembers=derived_endmembers,
        auxiliary_axes=auxiliary_axes,
        acquisition_json=acquisition_json,
        receipt_path=receipt_path,
        receipt_bytes=receipt_bytes,
        receipt_sha256=receipt_sha256,
        comparison=comparison,
        output_hashes=output_hashes,
    )
    if timings is not None:
        timings.update(staged_timings)
    key = id(staged)

    def discard_timing(reference) -> None:
        current = _STAGED_TIMINGS.get(key)
        if current is not None and current[0] is reference:
            _STAGED_TIMINGS.pop(key, None)

    reference = weakref.ref(staged, discard_timing)
    _STAGED_TIMINGS[key] = (
        reference,
        MappingProxyType(dict(staged_timings)),
    )
    return staged


def _linux_ru_maxrss_to_bytes(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Linux ru_maxrss must be a non-negative integer")
    return value * 1024


def _measure_sugar_core_access(
    dataset_path: Path,
) -> tuple[float, float, str]:
    open_started = time.perf_counter()
    dataset = UnifiedDataset.open(
        Path(dataset_path),
        verify_checksums=True,
    )
    open_seconds = time.perf_counter() - open_started
    try:
        record_ids = sorted(dataset.record_ids)
        if not record_ids:
            raise _fatal(
                "core.record_ids",
                "must not be empty",
                "CORE_ACCESS_PROBE_INVALID",
            )
        probe = hashlib.sha256(
            b"rpe-sugar-read-probe-v1\0" + DATASET_ID.encode("utf-8")
        ).digest()
        index = int.from_bytes(probe[:8], "little") % len(record_ids)
        record_id = record_ids[index]
        read_started = time.perf_counter()
        dataset.get(record_id)
        read_seconds = time.perf_counter() - read_started
    finally:
        dataset.close()
    return open_seconds, read_seconds, record_id


def _measure_views_access(
    path: Path,
    *,
    source_member_by_record_id: Mapping[str, str],
) -> float:
    started = time.perf_counter()
    from rpe.io.sugar_mixtures_companions import (
        _read_canonical_json,
        _validate_view_relations,
    )

    document = _read_canonical_json(Path(path))
    if (
        set(document)
        != {"artifact_role", "artifact_schema_version", "dataset_id", "views"}
        or document["artifact_role"] != "record_views"
        or document["dataset_id"] != DATASET_ID
        or not isinstance(document["views"], list)
        or len(document["views"]) != 8
    ):
        raise _fatal(
            "views.access_probe",
            "views document differs from complete role contract",
            "VIEWS_ACCESS_PROBE_INVALID",
        )
    observed: dict[str, set[str]] = {}
    expected_order = (
        "high_snr",
        "high_snr_no_refs",
        "high_snr_pure_reference",
        "low_snr",
        "low_snr_no_refs",
        "low_snr_pure_reference",
        "no_refs",
        "pure_reference",
    )
    expected_semantics = {
        "high_snr": (
            "source_acquisition_condition",
            {"source_acquisition_condition": "high_snr"},
        ),
        "high_snr_no_refs": (
            "condition_role_intersection",
            {
                "source_acquisition_condition": "high_snr",
                "source_record_role": "mixture",
            },
        ),
        "high_snr_pure_reference": (
            "condition_role_intersection",
            {
                "source_acquisition_condition": "high_snr",
                "source_record_role": "pure_reference",
            },
        ),
        "low_snr": (
            "source_acquisition_condition",
            {"source_acquisition_condition": "low_snr"},
        ),
        "low_snr_no_refs": (
            "condition_role_intersection",
            {
                "source_acquisition_condition": "low_snr",
                "source_record_role": "mixture",
            },
        ),
        "low_snr_pure_reference": (
            "condition_role_intersection",
            {
                "source_acquisition_condition": "low_snr",
                "source_record_role": "pure_reference",
            },
        ),
        "no_refs": (
            "source_reference_exclusion",
            {"source_record_role": "mixture"},
        ),
        "pure_reference": (
            "source_record_role",
            {"source_record_role": "pure_reference"},
        ),
    }
    for index, (expected_id, view) in enumerate(
        zip(expected_order, document["views"], strict=True)
    ):
        if (
            not isinstance(view, Mapping)
            or view.get("view_id") != expected_id
            or (
                view.get("semantic_role"),
                view.get("selector"),
            )
            != expected_semantics[expected_id]
            or view.get("train_test_semantics") is not False
            or not isinstance(view.get("record_ids"), list)
            or view.get("record_count") != len(view["record_ids"])
            or len(view["record_ids"]) != len(set(view["record_ids"]))
            or view["record_ids"] != sorted(view["record_ids"])
        ):
            raise _fatal(
                f"views.access_probe[{index}]",
                "view IDs/count/order differ from contract",
                "VIEWS_ACCESS_PROBE_INVALID",
            )
        record_ids = tuple(view["record_ids"])
        expected_digest = hashlib.sha256(
            b"rpe-sugar-view-record-ids-v1\0"
        )
        for record_id in sorted(
            record_ids,
            key=lambda value: value.encode("utf-8"),
        ):
            encoded = record_id.encode("utf-8")
            expected_digest.update(struct.pack("<Q", len(encoded)))
            expected_digest.update(encoded)
        if view.get("record_id_sha256") != expected_digest.hexdigest():
            raise _fatal(
                f"views.access_probe[{index}].record_id_sha256",
                "view record digest differs",
                "VIEWS_ACCESS_PROBE_INVALID",
            )
        source_digest = hashlib.sha256(
            b"rpe-sugar-view-source-members-v1\0"
        )
        try:
            source_members = tuple(
                sorted(
                    (
                        source_member_by_record_id[record_id]
                        for record_id in record_ids
                    ),
                    key=lambda value: value.encode("utf-8"),
                )
            )
        except KeyError as error:
            raise _fatal(
                f"views.access_probe[{index}].record_ids",
                f"record ID has no source binding: {error}",
                "VIEWS_ACCESS_PROBE_INVALID",
            ) from error
        for source_member in source_members:
            encoded = source_member.encode("utf-8")
            source_digest.update(struct.pack("<Q", len(encoded)))
            source_digest.update(encoded)
        if (
            view.get("source_member_path_sha256")
            != source_digest.hexdigest()
        ):
            raise _fatal(
                f"views.access_probe[{index}].source_member_path_sha256",
                "view source-member digest differs",
                "VIEWS_ACCESS_PROBE_INVALID",
            )
        observed[expected_id] = set(record_ids)
    _validate_view_relations(observed)
    if (
        observed["high_snr"] | observed["low_snr"]
        != set(source_member_by_record_id)
    ):
        raise _fatal(
            "views.access_probe.record_ids",
            "condition views do not cover the core record universe",
            "VIEWS_ACCESS_PROBE_INVALID",
        )
    return time.perf_counter() - started


def _measure_acquisition_json_access(
    path: Path,
    *,
    record_ids: set[str],
    source_bindings: Mapping[str, tuple[str, str]],
) -> float:
    started = time.perf_counter()
    from rpe.io.sugar_mixtures_companions import _parse_jsonl_line

    previous_record_id = None
    observed_record_ids = set()
    line_count = 0
    with Path(path).open("rb") as source:
        for line_count, line in enumerate(source, start=1):
            document = _parse_jsonl_line(line, index=line_count - 1)
            record_id = document["record_id"]
            if (
                previous_record_id is not None
                and record_id <= previous_record_id
            ):
                raise _fatal(
                    "acquisition_json.access_probe.record_id",
                    "record IDs must be strictly sorted",
                    "ACQUISITION_JSON_ACCESS_PROBE_INVALID",
                )
            previous_record_id = record_id
            observed_record_ids.add(record_id)
            expected_binding = source_bindings.get(record_id)
            if (
                expected_binding is None
                or (
                    document["source_member"],
                    document["source_member_sha256"],
                )
                != expected_binding
            ):
                raise _fatal(
                    "acquisition_json.access_probe.source_binding",
                    "JSONL source binding differs from inspection",
                    "ACQUISITION_JSON_ACCESS_PROBE_INVALID",
                )
    if line_count <= 0:
        raise _fatal(
            "acquisition_json.access_probe",
            "must contain at least one line",
            "ACQUISITION_JSON_ACCESS_PROBE_INVALID",
        )
    if observed_record_ids != record_ids or line_count != len(record_ids):
        raise _fatal(
            "acquisition_json.access_probe.record_ids",
            "JSONL record IDs differ from core dataset",
            "ACQUISITION_JSON_ACCESS_PROBE_INVALID",
        )
    return time.perf_counter() - started


def _measure_auxiliary_axes_access(
    path: Path,
    *,
    expected_logical_content_sha256: str,
) -> float:
    started = time.perf_counter()
    path = Path(path)
    with h5py.File(path, "r") as artifact:
        if (
            set(artifact) != {"metadata_json", "axes"}
            or set(artifact.attrs)
            != {
                "artifact_role",
                "artifact_schema_version",
                "dataset_id",
                "logical_content_sha256",
            }
            or set(artifact["axes"].attrs)
            or artifact.attrs.get("artifact_role") != "auxiliary_axes"
            or artifact.attrs.get("artifact_schema_version") != "1.0.0"
            or artifact.attrs.get("dataset_id") != DATASET_ID
            or artifact.attrs.get("logical_content_sha256")
            != expected_logical_content_sha256
        ):
            raise _fatal(
                "auxiliary_axes.access_probe",
                "object set differs from contract",
                "AUXILIARY_ACCESS_PROBE_INVALID",
            )
        if (
            artifact.attrs.get("logical_content_sha256")
            != expected_logical_content_sha256
        ):
            raise _fatal(
                "auxiliary_axes.access_probe.logical_content_sha256",
                "logical digest differs from staged summary",
                "AUXILIARY_ACCESS_PROBE_INVALID",
            )
        metadata = artifact["metadata_json"][...]
        metadata_dataset = artifact["metadata_json"]
        pixel_dataset = artifact["axes"]["pixel"]
        wavelength_dataset = artifact["axes"]["wavelength_nm"]
        expected_layouts = (
            (
                metadata_dataset,
                "|u1",
                metadata_dataset.shape,
            ),
            (
                pixel_dataset,
                "<u2",
                pixel_dataset.shape,
            ),
            (
                wavelength_dataset,
                "<f4",
                wavelength_dataset.shape,
            ),
        )
        if any(
            dataset.dtype.str != dtype
            or dataset.chunks != shape
            or dataset.compression != "gzip"
            or dataset.compression_opts != 4
            or dataset.shuffle is not True
            or dataset.fletcher32 is not True
            for dataset, dtype, shape in expected_layouts
        ):
            raise _fatal(
                "auxiliary_axes.access_probe.layout",
                "dtype/chunk/filter layout differs from contract",
                "AUXILIARY_ACCESS_PROBE_INVALID",
            )
        pixel = pixel_dataset[...]
        wavelength = wavelength_dataset[...]
        if (
            metadata.size <= 0
            or pixel.ndim != 1
            or wavelength.ndim != 1
            or pixel.shape != wavelength.shape
        ):
            raise _fatal(
                "auxiliary_axes.access_probe",
                "stored arrays differ from complete role contract",
                "AUXILIARY_ACCESS_PROBE_INVALID",
            )
        observed_digest = hashlib.sha256(
            b"rpe-sugar-auxiliary-axes-logical-v1\0"
        )
        for payload in (
            metadata.tobytes(order="C"),
            np.ascontiguousarray(pixel, dtype="<u2").tobytes(order="C"),
            np.ascontiguousarray(wavelength, dtype="<f4").tobytes(order="C"),
        ):
            observed_digest.update(struct.pack("<Q", len(payload)))
            observed_digest.update(payload)
        if observed_digest.hexdigest() != expected_logical_content_sha256:
            raise _fatal(
                "auxiliary_axes.access_probe.logical_content_sha256",
                "logical content differs from staged summary",
                "AUXILIARY_ACCESS_PROBE_INVALID",
            )
    return time.perf_counter() - started


def _measure_derived_endmembers_access(
    path: Path,
    *,
    expected_logical_content_sha256: str,
) -> float:
    started = time.perf_counter()
    path = Path(path)
    with h5py.File(path, "r") as artifact:
        if (
            set(artifact) != {"metadata_json", "intensity"}
            or set(artifact.attrs)
            != {
                "artifact_role",
                "artifact_schema_version",
                "dataset_id",
                "logical_content_sha256",
            }
            or artifact.attrs.get("artifact_role")
            != "derived_reference_endmembers"
            or artifact.attrs.get("artifact_schema_version") != "1.0.0"
            or artifact.attrs.get("dataset_id") != DATASET_ID
            or artifact.attrs.get("logical_content_sha256")
            != expected_logical_content_sha256
        ):
            raise _fatal(
                "derived_endmembers.access_probe",
                "object set differs from contract",
                "ENDMEMBER_ACCESS_PROBE_INVALID",
            )
        if (
            artifact.attrs.get("logical_content_sha256")
            != expected_logical_content_sha256
        ):
            raise _fatal(
                "derived_endmembers.access_probe.logical_content_sha256",
                "logical digest differs from staged summary",
                "ENDMEMBER_ACCESS_PROBE_INVALID",
            )
        metadata_dataset = artifact["metadata_json"]
        intensity_dataset = artifact["intensity"]
        if (
            metadata_dataset.dtype.str != "|u1"
            or metadata_dataset.chunks != metadata_dataset.shape
            or metadata_dataset.compression != "gzip"
            or metadata_dataset.compression_opts != 4
            or metadata_dataset.shuffle is not True
            or metadata_dataset.fletcher32 is not True
            or intensity_dataset.dtype.str != "<f4"
            or intensity_dataset.chunks
            != (1, 1, intensity_dataset.shape[-1])
            or intensity_dataset.compression != "gzip"
            or intensity_dataset.compression_opts != 4
            or intensity_dataset.shuffle is not True
            or intensity_dataset.fletcher32 is not True
        ):
            raise _fatal(
                "derived_endmembers.access_probe.layout",
                "dtype/chunk/filter layout differs from contract",
                "ENDMEMBER_ACCESS_PROBE_INVALID",
            )
        metadata = metadata_dataset[...]
        intensity = intensity_dataset[...]
        if (
            metadata.size <= 0
            or intensity.ndim != 3
            or intensity.shape[:2] != (2, 5)
        ):
            raise _fatal(
                "derived_endmembers.access_probe",
                "must read exactly ten endmember arrays",
                "ENDMEMBER_ACCESS_PROBE_INVALID",
            )
        observed_digest = hashlib.sha256(
            b"rpe-sugar-derived-endmembers-logical-v1\0"
        )
        metadata_payload = metadata.tobytes(order="C")
        observed_digest.update(
            struct.pack("<Q", len(metadata_payload))
        )
        observed_digest.update(metadata_payload)
        for condition_index in range(intensity.shape[0]):
            for component_index in range(intensity.shape[1]):
                payload = np.ascontiguousarray(
                    intensity[condition_index, component_index],
                    dtype="<f4",
                ).tobytes(order="C")
                observed_digest.update(struct.pack("<Q", len(payload)))
                observed_digest.update(payload)
        if observed_digest.hexdigest() != expected_logical_content_sha256:
            raise _fatal(
                "derived_endmembers.access_probe.logical_content_sha256",
                "logical content differs from staged summary",
                "ENDMEMBER_ACCESS_PROBE_INVALID",
            )
    return time.perf_counter() - started


def _staged_timings(staged: _SugarStagedData) -> Mapping[str, float]:
    current = _STAGED_TIMINGS.pop(id(staged), None)
    if current is None or current[0]() is not staged:
        raise _fatal(
            "metrics.staged_timings",
            "staged build timings are unavailable or stale",
            "RUNTIME_METRICS_INVALID",
        )
    return current[1]


def _measure_sugar_staged(
    staged: _SugarStagedData,
    *,
    total_seconds: float,
    inspection_seconds: float,
    peak_rss_raw: int,
) -> SugarMixturesRuntimeMetrics:
    timings = _staged_timings(staged)
    required_timing_names = {
        "core_build_seconds",
        "views_build_seconds",
        "derived_endmembers_build_seconds",
        "auxiliary_axes_build_seconds",
        "acquisition_json_build_seconds",
        "receipt_build_seconds",
        "validation_seconds",
    }
    if set(timings) != required_timing_names:
        raise _fatal(
            "metrics.staged_timings",
            "staged timing key set differs from contract",
            "RUNTIME_METRICS_INVALID",
        )
    (
        core_lazy_open_seconds,
        core_representative_record_read_seconds,
        core_representative_record_id,
    ) = _measure_sugar_core_access(staged.core.path)
    receipt = json.loads(staged.receipt_path.read_bytes())
    source_bindings = {
        member["record_id"]: (
            member["source_member"],
            member["sha256"],
        )
        for member in receipt["source_contract"]["canonical_members"]
    }
    source_member_by_record_id = {
        record_id: binding[0]
        for record_id, binding in source_bindings.items()
    }
    views_open_validate_seconds = _measure_views_access(
        staged.views.path,
        source_member_by_record_id=source_member_by_record_id,
    )
    with UnifiedDataset.open(staged.core.path) as core_dataset:
        core_record_ids = set(core_dataset.record_ids)
    acquisition_json_stream_validate_seconds = (
        _measure_acquisition_json_access(
            staged.acquisition_json.path,
            record_ids=core_record_ids,
            source_bindings=source_bindings,
        )
    )
    auxiliary_axes_open_validate_seconds = (
        _measure_auxiliary_axes_access(
            staged.auxiliary_axes.path,
            expected_logical_content_sha256=(
                staged.auxiliary_axes.logical_content_sha256 or ""
            ),
        )
    )
    derived_endmembers_open_validate_seconds = (
        _measure_derived_endmembers_access(
            staged.derived_endmembers.path,
            expected_logical_content_sha256=(
                staged.derived_endmembers.logical_content_sha256 or ""
            ),
        )
    )
    current_rss_raw = int(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    )
    measured_peak_rss_raw = max(peak_rss_raw, current_rss_raw)
    archive = receipt["source_contract"]["archive"]
    roles = {
        role["role"]: role
        for role in receipt["source_contract"]["member_roles"]
    }
    source_archive_bytes = archive["bytes"]
    canonical_relevant_source_bytes = sum(
        roles[name]["uncompressed_bytes"]
        for name in ("canonical_high", "canonical_low", "target")
    )
    combined_output_bytes = (
        staged.core.output_bytes
        + staged.views.bytes
        + staged.derived_endmembers.bytes
        + staged.auxiliary_axes.bytes
        + staged.acquisition_json.bytes
        + staged.receipt_bytes
    )
    staged_build_seconds = sum(
        timings[name]
        for name in (
            "core_build_seconds",
            "views_build_seconds",
            "derived_endmembers_build_seconds",
            "auxiliary_axes_build_seconds",
            "acquisition_json_build_seconds",
            "receipt_build_seconds",
        )
    )
    return SugarMixturesRuntimeMetrics(
        total_seconds=total_seconds,
        inspection_seconds=inspection_seconds,
        staged_build_seconds=staged_build_seconds,
        core_build_seconds=timings["core_build_seconds"],
        views_build_seconds=timings["views_build_seconds"],
        derived_endmembers_build_seconds=timings[
            "derived_endmembers_build_seconds"
        ],
        auxiliary_axes_build_seconds=timings[
            "auxiliary_axes_build_seconds"
        ],
        acquisition_json_build_seconds=timings[
            "acquisition_json_build_seconds"
        ],
        receipt_build_seconds=timings["receipt_build_seconds"],
        validation_seconds=timings["validation_seconds"],
        peak_rss_raw=measured_peak_rss_raw,
        peak_rss_bytes=_linux_ru_maxrss_to_bytes(measured_peak_rss_raw),
        core_output_bytes=staged.core.output_bytes,
        views_output_bytes=staged.views.bytes,
        derived_endmembers_output_bytes=staged.derived_endmembers.bytes,
        auxiliary_axes_output_bytes=staged.auxiliary_axes.bytes,
        acquisition_json_output_bytes=staged.acquisition_json.bytes,
        receipt_bytes=staged.receipt_bytes,
        combined_output_bytes=combined_output_bytes,
        source_archive_bytes=source_archive_bytes,
        canonical_relevant_source_bytes=canonical_relevant_source_bytes,
        compression_ratio_archive=(
            combined_output_bytes / source_archive_bytes
        ),
        compression_ratio_relevant=(
            combined_output_bytes / canonical_relevant_source_bytes
        ),
        core_axis_groups=staged.core.axis_group_count,
        core_lazy_open_seconds=core_lazy_open_seconds,
        core_representative_record_read_seconds=(
            core_representative_record_read_seconds
        ),
        core_representative_record_id=core_representative_record_id,
        views_open_validate_seconds=views_open_validate_seconds,
        acquisition_json_stream_validate_seconds=(
            acquisition_json_stream_validate_seconds
        ),
        auxiliary_axes_open_validate_seconds=(
            auxiliary_axes_open_validate_seconds
        ),
        derived_endmembers_open_validate_seconds=(
            derived_endmembers_open_validate_seconds
        ),
    )


def _sugar_feasibility_failures(
    staged: _SugarStagedData,
    metrics: SugarMixturesRuntimeMetrics,
    comparison: _SugarComparisonSummary,
) -> tuple[str, ...]:
    failures: list[str] = []

    def fail(name: str) -> None:
        if name not in failures:
            failures.append(name)

    numeric_names = tuple(
        name
        for name in SugarMixturesRuntimeMetrics.__dataclass_fields__
        if name != "core_representative_record_id"
    )
    for name in numeric_names:
        value = getattr(metrics, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            fail(name)
    integer_names = (
        "peak_rss_raw",
        "peak_rss_bytes",
        "core_output_bytes",
        "views_output_bytes",
        "derived_endmembers_output_bytes",
        "auxiliary_axes_output_bytes",
        "acquisition_json_output_bytes",
        "receipt_bytes",
        "combined_output_bytes",
        "source_archive_bytes",
        "canonical_relevant_source_bytes",
        "core_axis_groups",
    )
    for name in integer_names:
        value = getattr(metrics, name)
        if isinstance(value, bool) or not isinstance(value, int):
            fail(name)
    if (
        not isinstance(metrics.core_representative_record_id, str)
        or not metrics.core_representative_record_id
    ):
        fail("core_representative_record_id")
    build_names = (
        "core_build_seconds",
        "views_build_seconds",
        "derived_endmembers_build_seconds",
        "auxiliary_axes_build_seconds",
        "acquisition_json_build_seconds",
        "receipt_build_seconds",
    )
    component_names = (
        "inspection_seconds",
        "staged_build_seconds",
        "validation_seconds",
        "core_lazy_open_seconds",
        "core_representative_record_read_seconds",
        "views_open_validate_seconds",
        "acquisition_json_stream_validate_seconds",
        "auxiliary_axes_open_validate_seconds",
        "derived_endmembers_open_validate_seconds",
    )

    def finite_nonnegative(name: str) -> bool:
        value = getattr(metrics, name)
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
        )

    valid_combined_output = finite_nonnegative("combined_output_bytes")
    expected_staged_seconds = (
        sum(getattr(metrics, name) for name in build_names)
        if all(finite_nonnegative(name) for name in build_names)
        else None
    )
    expected_component_seconds = (
        sum(getattr(metrics, name) for name in component_names)
        if all(finite_nonnegative(name) for name in component_names)
        else None
    )
    if (
        expected_component_seconds is None
        or not finite_nonnegative("total_seconds")
        or metrics.total_seconds < expected_component_seconds
    ):
        fail("total_seconds")
    formula_checks = (
        (
            "staged_build_seconds",
            metrics.staged_build_seconds,
            expected_staged_seconds,
        ),
        (
            "peak_rss_bytes",
            metrics.peak_rss_bytes,
            (
                metrics.peak_rss_raw * 1024
                if isinstance(metrics.peak_rss_raw, int)
                and not isinstance(metrics.peak_rss_raw, bool)
                and metrics.peak_rss_raw >= 0
                else None
            ),
        ),
        (
            "core_output_bytes",
            metrics.core_output_bytes,
            staged.core.output_bytes,
        ),
        ("views_output_bytes", metrics.views_output_bytes, staged.views.bytes),
        (
            "derived_endmembers_output_bytes",
            metrics.derived_endmembers_output_bytes,
            staged.derived_endmembers.bytes,
        ),
        (
            "auxiliary_axes_output_bytes",
            metrics.auxiliary_axes_output_bytes,
            staged.auxiliary_axes.bytes,
        ),
        (
            "acquisition_json_output_bytes",
            metrics.acquisition_json_output_bytes,
            staged.acquisition_json.bytes,
        ),
        ("receipt_bytes", metrics.receipt_bytes, staged.receipt_bytes),
        (
            "combined_output_bytes",
            metrics.combined_output_bytes,
            staged.core.output_bytes
            + staged.views.bytes
            + staged.derived_endmembers.bytes
            + staged.auxiliary_axes.bytes
            + staged.acquisition_json.bytes
            + staged.receipt_bytes,
        ),
        (
            "compression_ratio_archive",
            metrics.compression_ratio_archive,
            (
                metrics.combined_output_bytes / metrics.source_archive_bytes
                if valid_combined_output
                and isinstance(metrics.source_archive_bytes, (int, float))
                and not isinstance(metrics.source_archive_bytes, bool)
                and metrics.source_archive_bytes > 0
                else None
            ),
        ),
        (
            "compression_ratio_relevant",
            metrics.compression_ratio_relevant,
            (
                metrics.combined_output_bytes
                / metrics.canonical_relevant_source_bytes
                if valid_combined_output
                and isinstance(
                    metrics.canonical_relevant_source_bytes,
                    (int, float),
                )
                and not isinstance(
                    metrics.canonical_relevant_source_bytes,
                    bool,
                )
                and metrics.canonical_relevant_source_bytes > 0
                else None
            ),
        ),
        (
            "core_axis_groups",
            metrics.core_axis_groups,
            staged.core.axis_group_count,
        ),
    )
    for name, observed, expected in formula_checks:
        if expected is None or observed != expected:
            fail(name)

    for name, limit in _SUGAR_METRIC_THRESHOLDS:
        value = getattr(metrics, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or value >= limit
        ):
            fail(name)

    scientific_checks = (
        ("canonical_records", staged.core.record_count, 9_800),
        ("eligible_records", staged.core.eligible_records, 9_800),
        ("core_axis_groups", staged.core.axis_group_count, 1),
        ("view_entries", staged.views.item_count, 8),
        (
            "view_memberships",
            (
                29_400
                if staged.core.record_count == 9_800
                and staged.views.item_count == 8
                else None
            ),
            29_400,
        ),
        ("endmembers", staged.derived_endmembers.item_count, 10),
        ("auxiliary_axis_sets", staged.auxiliary_axes.item_count, 1),
        (
            "acquisition_json_lines",
            staged.acquisition_json.item_count,
            9_800,
        ),
        (
            "staged_regular_files",
            set(staged.output_hashes),
            _SUGAR_STAGED_FILE_NAMES,
        ),
        (
            "source_rows_compared",
            comparison.source_rows_compared,
            9_800,
        ),
        (
            "source_points_compared",
            comparison.source_points_compared,
            19_600_000,
        ),
        (
            "consolidated_intensity_cells_compared",
            comparison.consolidated_intensity_cells_compared,
            19_600_000,
        ),
        (
            "prepared_abundance_rows_compared",
            comparison.prepared_abundance_rows_compared,
            19_400,
        ),
        (
            "prepared_endmembers_compared",
            comparison.prepared_endmembers_compared,
            10,
        ),
        ("intensity_mismatches", comparison.intensity_mismatches, 0),
        ("axis_mismatches", comparison.axis_mismatches, 0),
        ("target_mismatches", comparison.target_mismatches, 0),
        ("metadata_mismatches", comparison.metadata_mismatches, 0),
        (
            "preprocessing_status_mismatches",
            comparison.preprocessing_status_mismatches,
            0,
        ),
        (
            "companion_reference_mismatches",
            comparison.companion_reference_mismatches,
            0,
        ),
    )
    for name, observed, expected in scientific_checks:
        if observed != expected:
            fail(name)
    expected_hashes = {
        **{
            f"{DATASET_ID}/{name}": details["sha256"]
            for name, details in staged.core.files.items()
        },
        staged.views.path.name: staged.views.sha256,
        staged.derived_endmembers.path.name: staged.derived_endmembers.sha256,
        staged.auxiliary_axes.path.name: staged.auxiliary_axes.sha256,
        staged.acquisition_json.path.name: staged.acquisition_json.sha256,
        staged.receipt_path.name: staged.receipt_sha256,
    }
    if dict(staged.output_hashes) != expected_hashes:
        fail("staged_regular_files")
    failure_order = (
        *(name for name, _ in _SUGAR_METRIC_THRESHOLDS),
        *(
            name
            for name in SugarMixturesRuntimeMetrics.__dataclass_fields__
            if name
            not in {threshold_name for threshold_name, _ in _SUGAR_METRIC_THRESHOLDS}
        ),
        *(
            "canonical_records",
            "eligible_records",
            "core_axis_groups",
            "view_entries",
            "view_memberships",
            "endmembers",
            "auxiliary_axis_sets",
            "acquisition_json_lines",
            "staged_regular_files",
            "source_rows_compared",
            "source_points_compared",
            "consolidated_intensity_cells_compared",
            "prepared_abundance_rows_compared",
            "prepared_endmembers_compared",
            "intensity_mismatches",
            "axis_mismatches",
            "target_mismatches",
            "metadata_mismatches",
            "preprocessing_status_mismatches",
            "companion_reference_mismatches",
        ),
    )
    rank = {name: index for index, name in enumerate(failure_order)}
    return tuple(
        sorted(
            failures,
            key=lambda name: rank.get(name, len(rank)),
        )
    )


def _artifact_summary(
    staged: _SugarStagedData,
) -> Mapping[str, Mapping[str, object]]:
    receipt = json.loads(staged.receipt_path.read_bytes())
    semantic = receipt["semantic_contract"]
    return MappingProxyType(
        {
            "core": MappingProxyType(
                {
                    "dataset_id": staged.core.dataset_id,
                    "record_count": staged.core.record_count,
                    "eligible_records": staged.core.eligible_records,
                    "axis_group_count": staged.core.axis_group_count,
                    "output_bytes": staged.core.output_bytes,
                }
            ),
            "views": MappingProxyType(
                {
                    "artifact_role": staged.views.artifact_role,
                    "bytes": staged.views.bytes,
                    "sha256": staged.views.sha256,
                    "item_count": staged.views.item_count,
                    "view_memberships": semantic["views"][
                        "view_memberships"
                    ],
                }
            ),
            "derived_endmembers": MappingProxyType(
                {
                    "artifact_role": staged.derived_endmembers.artifact_role,
                    "bytes": staged.derived_endmembers.bytes,
                    "sha256": staged.derived_endmembers.sha256,
                    "logical_content_sha256": (
                        staged.derived_endmembers.logical_content_sha256
                    ),
                    "item_count": staged.derived_endmembers.item_count,
                }
            ),
            "auxiliary_axes": MappingProxyType(
                {
                    "artifact_role": staged.auxiliary_axes.artifact_role,
                    "bytes": staged.auxiliary_axes.bytes,
                    "sha256": staged.auxiliary_axes.sha256,
                    "logical_content_sha256": (
                        staged.auxiliary_axes.logical_content_sha256
                    ),
                    "item_count": staged.auxiliary_axes.item_count,
                    "auxiliary_axis_set_id": semantic["auxiliary_axes"][
                        "auxiliary_axis_set_id"
                    ],
                }
            ),
            "acquisition_json": MappingProxyType(
                {
                    "artifact_role": staged.acquisition_json.artifact_role,
                    "bytes": staged.acquisition_json.bytes,
                    "sha256": staged.acquisition_json.sha256,
                    "item_count": staged.acquisition_json.item_count,
                    "line_count": semantic["acquisition_json"][
                        "line_count"
                    ],
                }
            ),
            "receipt": MappingProxyType(
                {
                    "bytes": staged.receipt_bytes,
                    "sha256": staged.receipt_sha256,
                }
            ),
        }
    )


def _stage_sugar_mixtures_for_review(
    raw_root: Path,
    output_root: Path,
) -> _SugarStagingReview:
    started = time.perf_counter()
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    if output_root.is_symlink() or (
        output_root.exists() and not output_root.is_dir()
    ):
        raise _fatal(
            "output_root",
            "must be a non-symlink directory",
            "OUTPUT_ROOT_INVALID",
        )
    output_root_existed = output_root.exists()
    output_root.mkdir(parents=True, exist_ok=True)
    staging_parent: Path | None = None
    review: _SugarStagingReview | None = None
    try:
        free_disk_bytes = shutil.disk_usage(output_root).free
        if (
            isinstance(free_disk_bytes, bool)
            or not isinstance(free_disk_bytes, int)
            or free_disk_bytes <= _REQUIRED_FREE_DISK_BYTES
        ):
            raise _fatal(
                "output_root.free_disk_bytes",
                (
                    f"requires > {_REQUIRED_FREE_DISK_BYTES}, "
                    f"observed {free_disk_bytes}"
                ),
                "INSUFFICIENT_FREE_SPACE",
            )
        inspection_started = time.perf_counter()
        inspection = _inspect_sugar_mixtures(
            raw_root,
            _PRODUCTION_SOURCE_CONTRACT,
        )
        inspection_seconds = time.perf_counter() - inspection_started
        allocated_path = Path(
            tempfile.mkdtemp(
                prefix=".sugar-mixtures.staging-",
                dir=output_root,
            )
        )
        if (
            allocated_path.parent != output_root
            or not allocated_path.name.startswith(
                ".sugar-mixtures.staging-"
            )
            or allocated_path.is_symlink()
            or not allocated_path.is_dir()
        ):
            raise _fatal(
                "staging_parent",
                "allocator returned an unowned or invalid staging path",
                "STAGING_PARENT_INVALID",
            )
        staging_parent = allocated_path
        staged = _build_sugar_mixtures_staged(
            raw_root,
            staging_parent,
            inspection,
        )
        peak_rss_raw = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        )
        metrics = _measure_sugar_staged(
            staged,
            total_seconds=time.perf_counter() - started,
            inspection_seconds=inspection_seconds,
            peak_rss_raw=peak_rss_raw,
        )
        metrics = replace(
            metrics,
            total_seconds=time.perf_counter() - started,
        )
        failures = _sugar_feasibility_failures(
            staged,
            metrics,
            staged.comparison,
        )
        review = _SugarStagingReview(
            artifact_summaries=_artifact_summary(staged),
            receipt_sha256=staged.receipt_sha256,
            comparison=staged.comparison,
            source_statistics=MappingProxyType(
                dict(inspection.source_statistics)
            ),
            output_hashes=staged.output_hashes,
            metrics=metrics,
            feasibility_failures=failures,
        )
    finally:
        if staging_parent is not None and staging_parent.exists():
            shutil.rmtree(staging_parent)
        if (
            not output_root_existed
            and output_root.is_dir()
            and not any(output_root.iterdir())
        ):
            output_root.rmdir()
    if review is None:
        raise RuntimeError("Sugar staging review did not produce a result")
    final_metrics = replace(
        review.metrics,
        total_seconds=time.perf_counter() - started,
    )
    final_peak_rss_raw = max(
        final_metrics.peak_rss_raw,
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    final_metrics = replace(
        final_metrics,
        peak_rss_raw=final_peak_rss_raw,
        peak_rss_bytes=_linux_ru_maxrss_to_bytes(final_peak_rss_raw),
    )
    final_failures = list(review.feasibility_failures)
    peak_invalid = final_metrics.peak_rss_bytes >= 2_147_483_648
    if peak_invalid and "peak_rss_bytes" not in final_failures:
        following = {
            name
            for name, _ in _SUGAR_METRIC_THRESHOLDS[1:]
        } | {
            "canonical_records",
            "eligible_records",
            "core_axis_groups",
            "view_entries",
            "view_memberships",
            "endmembers",
            "auxiliary_axis_sets",
            "acquisition_json_lines",
            "staged_regular_files",
            "source_rows_compared",
            "source_points_compared",
            "consolidated_intensity_cells_compared",
            "prepared_abundance_rows_compared",
            "prepared_endmembers_compared",
            "intensity_mismatches",
            "axis_mismatches",
            "target_mismatches",
            "metadata_mismatches",
            "preprocessing_status_mismatches",
            "companion_reference_mismatches",
        }
        insertion = next(
            (
                index
                for index, name in enumerate(final_failures)
                if name in following
            ),
            len(final_failures),
        )
        final_failures.insert(insertion, "peak_rss_bytes")
    total_invalid = (
        isinstance(final_metrics.total_seconds, bool)
        or not isinstance(final_metrics.total_seconds, (int, float))
        or not math.isfinite(final_metrics.total_seconds)
        or final_metrics.total_seconds < 0
        or final_metrics.total_seconds >= 1_500.0
        or final_metrics.total_seconds
        < (
            final_metrics.inspection_seconds
            + final_metrics.staged_build_seconds
            + final_metrics.validation_seconds
            + final_metrics.core_lazy_open_seconds
            + final_metrics.core_representative_record_read_seconds
            + final_metrics.views_open_validate_seconds
            + final_metrics.acquisition_json_stream_validate_seconds
            + final_metrics.auxiliary_axes_open_validate_seconds
            + final_metrics.derived_endmembers_open_validate_seconds
        )
    )
    if not total_invalid and "total_seconds" in final_failures:
        final_failures.remove("total_seconds")
    if total_invalid and "total_seconds" not in final_failures:
        following = {
            name
            for name, _ in _SUGAR_METRIC_THRESHOLDS[
                tuple(
                    name for name, _ in _SUGAR_METRIC_THRESHOLDS
                ).index("total_seconds")
                + 1 :
            ]
        } | {
            "canonical_records",
            "eligible_records",
            "core_axis_groups",
            "view_entries",
            "view_memberships",
            "endmembers",
            "auxiliary_axis_sets",
            "acquisition_json_lines",
            "staged_regular_files",
            "source_rows_compared",
            "source_points_compared",
            "consolidated_intensity_cells_compared",
            "prepared_abundance_rows_compared",
            "prepared_endmembers_compared",
            "intensity_mismatches",
            "axis_mismatches",
            "target_mismatches",
            "metadata_mismatches",
            "preprocessing_status_mismatches",
            "companion_reference_mismatches",
        }
        insertion = next(
            (
                index
                for index, name in enumerate(final_failures)
                if name in following
            ),
            len(final_failures),
        )
        final_failures.insert(insertion, "total_seconds")
    return replace(
        review,
        metrics=final_metrics,
        feasibility_failures=tuple(final_failures),
    )
