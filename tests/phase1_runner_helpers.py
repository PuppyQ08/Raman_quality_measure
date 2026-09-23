from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.io.schema import (  # noqa: E402
    LicenseStatus,
    PeakTarget,
    PreprocessingStep,
    PreprocessingStatus,
    Provenance,
    RamanRecord,
    SpectrumMetadata,
    Targets,
)
from rpe.io.store import UnifiedDataset, write_dataset  # noqa: E402
from rpe.perturb import load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase1_config import (  # noqa: E402
    ArtifactIdentity,
    LocatedArtifactIdentity,
    Phase1CoreConfig,
)
from rpe.runner.phase1_perturbations import run_perturbation_cell  # noqa: E402
from rpe.runner.phase1_selection import (  # noqa: E402
    SelectedSourceRow,
    SourceInventoryRow,
    load_source_inventory,
    load_phase1_source,
)
from rpe.runner.phase1_types import CellStatus  # noqa: E402
from rpe.runner.phase1_perturbations import (  # noqa: E402
    estimate_p10_peak_bytes,
    run_shard_payload,
)


FIXTURE_DATASET_ID = "rruff_raman_raw"
FIXTURE_GLOBAL_SEED = 20260817
EXPECTED_RECORD_COUNT = 12
_FIXTURE_FILE_SHA = "1" * 64
SWEEP_CONFIG = ROOT / "experiments" / "shared" / "raman_perturbation_sweep_v1.json"
_UNSET = object()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_identity(path: Path) -> ArtifactIdentity:
    return ArtifactIdentity(
        byte_count=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _canonical_json_bytes(value: object) -> bytes:
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


def _fixture_axis() -> np.ndarray:
    return np.linspace(120.0, 1310.0, 120, dtype=np.float32)


def _fixture_intensity() -> np.ndarray:
    axis = _fixture_axis()
    first_peak = np.exp(-0.5 * ((axis - 420.0) / 35.0) ** 2)
    second_peak = 0.7 * np.exp(-0.5 * ((axis - 910.0) / 55.0) ** 2)
    baseline = 0.12 + 0.00008 * (axis - axis.min())
    return np.asarray(baseline + first_peak + second_peak, dtype=np.float32)


def increasing_axis_float32() -> np.ndarray:
    return _fixture_axis().copy()


def increasing_intensity_float32() -> np.ndarray:
    return _fixture_intensity().copy()


def decreasing_axis_float32() -> np.ndarray:
    return increasing_axis_float32()[::-1].copy()


def decreasing_intensity_float32() -> np.ndarray:
    return increasing_intensity_float32()[::-1].copy()


def inventory_rows(
    mapping: Mapping[int, Sequence[tuple[str, str]]],
) -> tuple[SourceInventoryRow, ...]:
    rows = []
    for class_label, pairs in mapping.items():
        for record_id, sample_id in pairs:
            rows.append(
                SourceInventoryRow(
                    record_id=record_id,
                    sample_id=sample_id,
                    class_label=class_label,
                    mineral_name=f"mineral-{class_label}",
                    axis_id=f"axis-{class_label}",
                )
            )
    return tuple(sorted(rows, key=lambda row: row.record_id))


def _fixture_record(
    *,
    record_id: str,
    sample_id: str | None,
    class_label: int | None,
    mineral_name: str,
    decreasing: bool,
    preprocessing_status: PreprocessingStatus = PreprocessingStatus.KNOWN_RAW,
) -> RamanRecord:
    if decreasing:
        axis = decreasing_axis_float32()
        intensity = decreasing_intensity_float32()
    else:
        axis = increasing_axis_float32()
        intensity = increasing_intensity_float32()
    return RamanRecord(
        record_id=record_id,
        intensity=intensity,
        wavenumber=axis,
        meta=SpectrumMetadata(
            dataset_id=FIXTURE_DATASET_ID,
            sample_id=sample_id,
            instrument="fixture-scope",
            excitation_nm=785.0,
            integration_time_s=1.0,
            n_accumulations=2,
            grating="1200-grooves-mm",
            detector="fixture-detector",
            preprocessing_status=preprocessing_status,
            preprocessing_steps=(
                ()
                if preprocessing_status is PreprocessingStatus.KNOWN_RAW
                else (
                    PreprocessingStep(
                        operation="fixture_step",
                        description="fixture documented preprocessing step",
                        evidence="fixture evidence",
                    ),
                )
            ),
            source_metadata={
                "fixture_group": f"class-{0 if class_label is None else class_label}",
                "mineral_name": mineral_name,
            },
        ),
        targets=Targets(
            peaks=(
                PeakTarget(
                    pos_cm1=420.0,
                    height=1.0,
                    fwhm=30.0,
                    assignment="peak-a",
                ),
                PeakTarget(
                    pos_cm1=910.0,
                    height=0.7,
                    fwhm=45.0,
                    assignment="peak-b",
                ),
            ),
            class_label=class_label,
        ),
        provenance=Provenance(
            source_url=f"https://example.org/{record_id}",
            license="CC-BY-4.0",
            license_status=LicenseStatus.STANDARDIZED,
            sha256=hashlib.sha256(f"source::{record_id}".encode("utf-8")).hexdigest(),
            retrieved_date=date(2026, 8, 18),
            source_artifact=f"{record_id}.txt",
        ),
    )


def _fixture_records() -> list[RamanRecord]:
    return [
        _fixture_record(
            record_id="a-00",
            sample_id="sample-a",
            class_label=0,
            mineral_name="Mineral-A",
            decreasing=False,
        ),
        _fixture_record(
            record_id="a-01",
            sample_id="sample-a",
            class_label=0,
            mineral_name="Mineral-A",
            decreasing=False,
        ),
        _fixture_record(
            record_id="a-02",
            sample_id="sample-b",
            class_label=0,
            mineral_name="Mineral-A",
            decreasing=False,
        ),
        _fixture_record(
            record_id="decreasing-record",
            sample_id="sample-b",
            class_label=0,
            mineral_name="Mineral-A",
            decreasing=True,
        ),
        _fixture_record(
            record_id="b-00",
            sample_id="sample-c",
            class_label=1,
            mineral_name="Mineral-B",
            decreasing=False,
        ),
        _fixture_record(
            record_id="b-01",
            sample_id="sample-d",
            class_label=1,
            mineral_name="Mineral-B",
            decreasing=False,
        ),
        _fixture_record(
            record_id="b-02",
            sample_id="sample-d",
            class_label=1,
            mineral_name="Mineral-B",
            decreasing=False,
        ),
        _fixture_record(
            record_id="b-03",
            sample_id="sample-e",
            class_label=1,
            mineral_name="Mineral-B",
            decreasing=False,
        ),
        _fixture_record(
            record_id="c-00",
            sample_id="sample-f",
            class_label=2,
            mineral_name="Mineral-C",
            decreasing=False,
        ),
        _fixture_record(
            record_id="c-01",
            sample_id="sample-f",
            class_label=2,
            mineral_name="Mineral-C",
            decreasing=False,
        ),
        _fixture_record(
            record_id="c-02",
            sample_id="sample-g",
            class_label=2,
            mineral_name="Mineral-C",
            decreasing=False,
        ),
        _fixture_record(
            record_id="c-03",
            sample_id="sample-h",
            class_label=2,
            mineral_name="Mineral-C",
            decreasing=False,
        ),
    ]


def write_phase1_fixture_dataset(
    root: Path,
    *,
    record_count: int = EXPECTED_RECORD_COUNT,
) -> Path:
    if record_count < 1 or record_count > EXPECTED_RECORD_COUNT:
        raise ValueError("record_count must be between 1 and 12 inclusive")
    root = Path(root)
    dataset_path = root / FIXTURE_DATASET_ID
    write_dataset(
        _fixture_records()[:record_count],
        dataset_path,
        dataset_id=FIXTURE_DATASET_ID,
        class_labels={
            0: "Mineral-A",
            1: "Mineral-B",
            2: "Mineral-C",
        },
    )
    return dataset_path


def write_custom_phase1_fixture_dataset(
    root: Path,
    *,
    records: Sequence[RamanRecord],
) -> Path:
    dataset_path = Path(root) / FIXTURE_DATASET_ID
    write_dataset(
        records,
        dataset_path,
        dataset_id=FIXTURE_DATASET_ID,
        class_labels={
            0: "Mineral-A",
            1: "Mineral-B",
            2: "Mineral-C",
        },
    )
    return dataset_path


def phase1_fixture_records() -> tuple[RamanRecord, ...]:
    return tuple(_fixture_records())


def phase1_fixture_config(
    dataset_path: Path,
    *,
    subset_size: int,
    shard_source_count: int,
) -> Phase1CoreConfig:
    dataset_path = Path(dataset_path)
    root = dataset_path.parent
    related_root = root / "phase1_fixture_related"
    related_root.mkdir(exist_ok=True)

    conversion_receipt = related_root / "conversion_receipt.json"
    if not conversion_receipt.exists():
        conversion_receipt.write_bytes(
            _canonical_json_bytes(
                {
                    "dataset_id": FIXTURE_DATASET_ID,
                    "kind": "conversion_receipt",
                    "record_count": EXPECTED_RECORD_COUNT,
                }
            )
        )
    pair_index = related_root / "pair_index.jsonl"
    if not pair_index.exists():
        pair_index.write_bytes(
            _canonical_json_bytes(
                {
                    "dataset_id": FIXTURE_DATASET_ID,
                    "kind": "pair_index",
                }
            )
        )

    shared_sweep_path = root / "phase1_fixture_shared_sweep.json"
    if not shared_sweep_path.exists():
        shared_sweep_path.write_bytes(
            _canonical_json_bytes({"global_seed": FIXTURE_GLOBAL_SEED})
        )

    source_files = {
        name: _artifact_identity(dataset_path / name)
        for name in (
            "SHA256SUMS",
            "SHA256SUMS.sha256",
            "arrays.h5",
            "dataset.json",
            "records.jsonl",
        )
    }
    related_provenance = {
        "conversion_receipt": LocatedArtifactIdentity(
            path=conversion_receipt,
            identity=_artifact_identity(conversion_receipt),
        ),
        "pair_index": LocatedArtifactIdentity(
            path=pair_index,
            identity=_artifact_identity(pair_index),
        ),
    }
    shared_sweep_identity = _artifact_identity(shared_sweep_path)

    with (dataset_path / "records.jsonl").open("rb") as stream:
        source_record_count = sum(1 for _ in stream)

    return Phase1CoreConfig(
        path=root / "phase1_fixture_config.json",
        file_byte_count=1,
        file_sha256=_FIXTURE_FILE_SHA,
        scientific_config_byte_count=1,
        scientific_config_sha256=_FIXTURE_FILE_SHA,
        schema_version="phase1-fixture-v1",
        experiment_id="phase1-fixture",
        source_dataset_id=dataset_path.name,
        source_dataset_path=dataset_path,
        source_record_count=source_record_count,
        source_class_count=3,
        source_files=source_files,
        related_provenance=related_provenance,
        shared_sweep_path=shared_sweep_path,
        shared_sweep_identity=shared_sweep_identity,
        subset_size=subset_size,
        selection_algorithm=(
            "rruff_class_min1_hamilton_capacity_sample_round_robin_sha256_v1"
        ),
        shard_source_count=shard_source_count,
        materialized_perturbation_ids=(
            "p01",
            "p02",
            "p03",
            "p04",
            "p05",
            "p08",
            "p09",
            "p10",
            "p11",
            "p12",
        ),
        deferred_perturbation_ids=("p06", "p07"),
        deferred_reason_code="missing_explicit_baseline",
        deferred_dependency="phase2_semisynthetic",
        peak_not_applicable_reason_codes=(
            "false_peak_placement_impossible",
            "no_detected_peak",
        ),
        distribution_status="fixture-only",
        core_gate={
            "allowed_failed_cell_count": 0,
            "float_relative_tolerance": 1e-12,
            "p01_p04_min_class_fraction": 0.95,
            "p01_p04_min_source_fraction": 0.95,
            "p05_min_class_fraction": 0.9,
            "p05_min_source_fraction": 0.9,
            "p08_p12_required_source_fraction": 1.0,
        },
    )


def _copied_float64_vector(values: np.ndarray) -> np.ndarray:
    return np.array(np.asarray(values, dtype="<f8"), dtype="<f8", copy=True)


def _thawed_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _thawed_json(item)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_thawed_json(item) for item in value)
    return value


def _fixture_complete_phase1_cell(perturbation_id: str):
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = write_phase1_fixture_dataset(Path(tmp))
        config = phase1_fixture_config(
            dataset_path,
            subset_size=8,
            shard_source_count=4,
        )
        sweep = load_perturbation_sweep_config(SWEEP_CONFIG)
        with UnifiedDataset.open(dataset_path, verify_checksums=True) as dataset:
            source = load_phase1_source(
                dataset,
                SelectedSourceRow(
                    selection_rank=0,
                    record_id="a-00",
                    sample_id="sample-a",
                    class_label=0,
                    mineral_name="Mineral-A",
                    axis_id="axis-0",
                ),
            )
        cell = run_perturbation_cell(source, perturbation_id, config, sweep)
    if cell.status is not CellStatus.COMPLETE:
        raise AssertionError(f"expected COMPLETE fixture cell for {perturbation_id}")
    return cell


def complete_fixture_cell(perturbation_id: str) -> "NativeCellView":
    from rpe.runner.phase1_gates import native_cell_view

    return native_cell_view(_fixture_complete_phase1_cell(perturbation_id))


def _rebuilt_record(
    record,
    *,
    output_spectrum_id: object = _UNSET,
    alpha: object = _UNSET,
    alpha_float64_le_hex: object = _UNSET,
    axis_cm1: object = _UNSET,
    intensity: object = _UNSET,
    diagnostics: object = _UNSET,
):
    return type(record)(
        output_spectrum_id=(
            record.output_spectrum_id
            if output_spectrum_id is _UNSET
            else output_spectrum_id
        ),
        alpha=record.alpha if alpha is _UNSET else alpha,
        alpha_float64_le_hex=(
            record.alpha_float64_le_hex
            if alpha_float64_le_hex is _UNSET
            else alpha_float64_le_hex
        ),
        axis_cm1=(
            _copied_float64_vector(record.axis_cm1)
            if axis_cm1 is _UNSET
            else _copied_float64_vector(axis_cm1)
        ),
        intensity=(
            _copied_float64_vector(record.intensity)
            if intensity is _UNSET
            else _copied_float64_vector(intensity)
        ),
        diagnostics=(
            _thawed_json(record.diagnostics)
            if diagnostics is _UNSET
            else diagnostics
        ),
    )


def _rebuilt_cell(
    cell,
    *,
    perturbation_id: object = _UNSET,
    state_digest: object = _UNSET,
    status: object = _UNSET,
    reason_code: object = _UNSET,
    records: object = _UNSET,
):
    return type(cell)(
        source_spectrum_id=cell.source_spectrum_id,
        source_record_id=cell.source_record_id,
        sample_id=cell.sample_id,
        class_label=cell.class_label,
        source_axis_cm1=_copied_float64_vector(cell.source_axis_cm1),
        source_intensity=_copied_float64_vector(cell.source_intensity),
        perturbation_id=(
            cell.perturbation_id
            if perturbation_id is _UNSET
            else perturbation_id
        ),
        state_digest=cell.state_digest if state_digest is _UNSET else state_digest,
        status=cell.status if status is _UNSET else status,
        reason_code=cell.reason_code if reason_code is _UNSET else reason_code,
        records=cell.records if records is _UNSET else records,
    )


def _mutated_diagnostics(record, **updates: object):
    diagnostics = dict(_thawed_json(record.diagnostics))
    diagnostics.update(updates)
    return diagnostics


def mutate_p01_removed_area(cell) -> "NativeCellView":
    records = list(cell.records)
    records[-1] = _rebuilt_record(
        records[-1],
        diagnostics=_mutated_diagnostics(records[-1], removed_component_area=0.0),
    )
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p02_selected_count(cell) -> "NativeCellView":
    records = list(cell.records)
    records[-1] = _rebuilt_record(
        records[-1],
        diagnostics=_mutated_diagnostics(
            records[-1],
            selected_or_inserted_peak_count=0,
        ),
    )
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p03_inserted_area(cell) -> "NativeCellView":
    records = list(cell.records)
    records[-1] = _rebuilt_record(
        records[-1],
        diagnostics=_mutated_diagnostics(
            records[-1],
            inserted_component_area=float(
                records[-1].diagnostics["inserted_component_area"]
            )
            + 1.0,
        ),
    )
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p04_removed_area(cell) -> "NativeCellView":
    records = list(cell.records)
    records[-1] = _rebuilt_record(
        records[-1],
        diagnostics=_mutated_diagnostics(records[-1], removed_component_area=0.0),
    )
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p05_centers(cell) -> "NativeCellView":
    records = list(cell.records)
    records[-1] = _rebuilt_record(
        records[-1],
        diagnostics=_mutated_diagnostics(
            records[-1],
            inserted_peak_centers_cm1=(123.456, 789.012),
        ),
    )
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p08_residual(cell) -> "NativeCellView":
    records = list(cell.records)
    records[-1] = _rebuilt_record(
        records[-1],
        diagnostics=_mutated_diagnostics(
            records[-1],
            realized_added_baseline_rms=float(
                records[-1].diagnostics["realized_added_baseline_rms"]
            )
            + 1.0,
        ),
    )
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p09_residual(cell) -> "NativeCellView":
    records = list(cell.records)
    mutated = _copied_float64_vector(records[-1].intensity)
    mutated[0] = float(mutated[0] + 1.0)
    records[-1] = _rebuilt_record(records[-1], intensity=mutated)
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p10_residual(cell) -> "NativeCellView":
    records = list(cell.records)
    records[-1] = _rebuilt_record(
        records[-1],
        diagnostics=_mutated_diagnostics(
            records[-1],
            sigma=float(records[-1].diagnostics["sigma"]) * 0.5,
        ),
    )
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p11_intensity(cell) -> "NativeCellView":
    records = list(cell.records)
    mutated = _copied_float64_vector(records[-1].intensity)
    mutated[0] = float(mutated[0] + 1.0)
    records[-1] = _rebuilt_record(records[-1], intensity=mutated)
    return _rebuilt_cell(cell, records=tuple(records))


def mutate_p12_axis_order(cell) -> "NativeCellView":
    records = list(cell.records)
    mutated = _copied_float64_vector(records[-1].axis_cm1)
    mutated[10] = float(mutated[9] - 0.5)
    records[-1] = _rebuilt_record(records[-1], axis_cm1=mutated)
    return _rebuilt_cell(cell, records=tuple(records))


def write_phase1_fixture_shard_payload(
    root: Path,
    *,
    source_count: int = 2,
):
    if source_count < 1 or source_count > 4:
        raise ValueError("source_count must be between 1 and 4 inclusive")
    dataset_path = write_phase1_fixture_dataset(Path(root))
    config = phase1_fixture_config(
        dataset_path,
        subset_size=max(source_count, 4),
        shard_source_count=source_count,
    )
    sweep = load_perturbation_sweep_config(SWEEP_CONFIG)
    inventory_by_record_id = {
        row.record_id: row
        for row in load_source_inventory(config)
    }
    selected_ids = (
        "a-00",
        "decreasing-record",
        "b-00",
        "c-00",
    )[:source_count]
    selected_rows = tuple(
        SelectedSourceRow(
            selection_rank=index,
            record_id=inventory_by_record_id[record_id].record_id,
            sample_id=inventory_by_record_id[record_id].sample_id,
            class_label=inventory_by_record_id[record_id].class_label,
            mineral_name=inventory_by_record_id[record_id].mineral_name,
            axis_id=inventory_by_record_id[record_id].axis_id,
        )
        for index, record_id in enumerate(selected_ids)
    )
    with UnifiedDataset.open(dataset_path, verify_checksums=True) as dataset:
        sources = tuple(
            load_phase1_source(dataset, row)
            for row in selected_rows
        )
    memory_budget_bytes = sum(
        estimate_p10_peak_bytes(source.spectrum.axis_cm1.size)
        for source in sources
    )
    return run_shard_payload(
        sources,
        config,
        sweep,
        shard_index=0,
        worker_count=1,
        memory_budget_bytes=memory_budget_bytes,
    )


def assert_stored_shard_matches_payload(stored, payload) -> None:
    assert stored.run_id
    assert stored.shard_index == payload.shard_index
    assert stored.source_count == len(payload.sources)
    assert stored.cell_count == len(payload.cells)
    assert stored.record_count == sum(len(cell.records) for cell in payload.cells)

    for stored_source, source in zip(stored.sources, payload.sources, strict=True):
        assert stored_source.source_spectrum_id == source.spectrum.spectrum_id
        assert stored_source.source_record_id == source.selection.record_id
        assert stored_source.sample_id == source.selection.sample_id
        assert stored_source.class_label == source.selection.class_label
        assert stored_source.mineral_name == source.selection.mineral_name
        assert _thawed_json(stored_source.provenance) == _thawed_json(source.provenance)
        assert np.array_equal(stored_source.axis_cm1, source.spectrum.axis_cm1)
        assert np.array_equal(stored_source.intensity, source.spectrum.intensity)
        assert not np.shares_memory(stored_source.axis_cm1, source.spectrum.axis_cm1)
        assert not np.shares_memory(stored_source.intensity, source.spectrum.intensity)
        assert not stored_source.axis_cm1.flags.writeable
        assert not stored_source.intensity.flags.writeable

    for stored_cell, payload_cell in zip(stored.cells, payload.cells, strict=True):
        assert stored_cell.source_spectrum_id == payload_cell.source.spectrum.spectrum_id
        assert stored_cell.perturbation_id == payload_cell.perturbation_id
        assert stored_cell.status == payload_cell.status.value
        assert stored_cell.reason_code == payload_cell.reason_code
        expected_state_digest = (
            None if payload_cell.state is None else payload_cell.state.state_digest
        )
        assert stored_cell.state_digest == expected_state_digest
        assert _thawed_json(stored_cell.native_gate) == _thawed_json(
            payload_cell.evidence.native_gate
        )
        assert len(stored_cell.records) == len(payload_cell.records)
        for stored_record, payload_record in zip(
            stored_cell.records,
            payload_cell.records,
            strict=True,
        ):
            result = payload_record.result
            assert stored_record.source_spectrum_id == result.source_spectrum_id
            assert stored_record.perturbation_id == result.perturbation_id
            assert stored_record.output_spectrum_id == result.output.spectrum_id
            assert stored_record.alpha == result.alpha
            assert stored_record.alpha_float64_le_hex == payload_record.alpha_float64_le_hex
            assert stored_record.state_digest == result.state_digest
            assert stored_record.axis_behavior == result.axis_behavior.value
            assert stored_record.axis_changed == result.axis_changed
            assert stored_record.intensity_changed == result.intensity_changed
            assert _thawed_json(stored_record.diagnostics) == _thawed_json(
                result.diagnostics
            )
            assert np.array_equal(stored_record.axis_cm1, result.output.axis_cm1)
            assert np.array_equal(stored_record.intensity, result.output.intensity)
            assert not np.shares_memory(stored_record.axis_cm1, result.output.axis_cm1)
            assert not np.shares_memory(
                stored_record.intensity,
                result.output.intensity,
            )
            assert not stored_record.axis_cm1.flags.writeable
            assert not stored_record.intensity.flags.writeable
