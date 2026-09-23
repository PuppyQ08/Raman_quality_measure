from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from phase1_runner_helpers import (  # noqa: E402
    complete_fixture_cell,
    mutate_p01_removed_area,
    mutate_p02_selected_count,
    mutate_p03_inserted_area,
    mutate_p04_removed_area,
    mutate_p05_centers,
    mutate_p08_residual,
    mutate_p09_residual,
    mutate_p10_residual,
    mutate_p11_intensity,
    mutate_p12_axis_order,
    phase1_fixture_config,
    write_phase1_fixture_dataset,
)
from rpe.io.store import UnifiedDataset  # noqa: E402
from rpe.perturb import (  # noqa: E402
    AxisBehavior,
    Perturbation,
    PerturbationContext,
    PerturbationResult,
    load_perturbation_sweep_config,
)
import rpe.runner.phase1_perturbations as phase1_perturbations  # noqa: E402
from rpe.runner.phase1_gates import (  # noqa: E402
    CellGateResult,
    CoreGateResult,
    NativeCellView,
    NativeRecordView,
    Phase1GateError,
    evaluate_core_gate,
    native_cell_view,
    stable_population_rms,
    validate_cell_native_gate,
)
from rpe.runner.phase1_selection import SelectedSourceRow, load_phase1_source  # noqa: E402
from rpe.runner.phase1_types import CellStatus  # noqa: E402


SWEEP_CONFIG = ROOT / "experiments" / "shared" / "raman_perturbation_sweep_v1.json"
_UNSET = object()

COMPLETE_IDS = ("p01", "p02", "p03", "p04", "p05", "p08", "p09", "p10", "p11", "p12")
ALL_IDS = tuple(f"p{index:02d}" for index in range(1, 13))
GATE_MUTATIONS = (
    ("p01", mutate_p01_removed_area, "native_gate.p01.removed_component_area"),
    ("p02", mutate_p02_selected_count, "native_gate.p02.selected_count"),
    ("p03", mutate_p03_inserted_area, "native_gate.p03.inserted_component_area"),
    ("p04", mutate_p04_removed_area, "native_gate.p04.removed_component_area"),
    ("p05", mutate_p05_centers, "native_gate.p05.inserted_centers"),
    ("p08", mutate_p08_residual, "native_gate.p08.residual_rms"),
    ("p09", mutate_p09_residual, "native_gate.p09.residual_rms"),
    ("p10", mutate_p10_residual, "native_gate.p10.residual_rms"),
    ("p11", mutate_p11_intensity, "native_gate.p11.intensity"),
    ("p12", mutate_p12_axis_order, "native_gate.p12.axis"),
)


def fixture_sweep():
    return load_perturbation_sweep_config(SWEEP_CONFIG)


def fixture_config(*, subset_size: int, shard_source_count: int):
    with tempfile.TemporaryDirectory() as tmp:
        dataset_path = write_phase1_fixture_dataset(Path(tmp))
        return phase1_fixture_config(
            dataset_path,
            subset_size=subset_size,
            shard_source_count=shard_source_count,
        )


def _copied_vector(values: np.ndarray) -> np.ndarray:
    copied = np.array(np.asarray(values, dtype="<f8"), dtype="<f8", copy=True)
    copied.setflags(write=False)
    return copied


def _thaw_json(value):
    if isinstance(value, MappingProxyType):
        value = dict(value)
    if isinstance(value, dict):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw_json(item) for item in value)
    return value


def _record_copy(
    record: NativeRecordView,
    *,
    output_spectrum_id: str | None = None,
    alpha: float | None = None,
    alpha_float64_le_hex: str | None = None,
    axis_cm1: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    diagnostics=None,
) -> NativeRecordView:
    return NativeRecordView(
        output_spectrum_id=record.output_spectrum_id if output_spectrum_id is None else output_spectrum_id,
        alpha=record.alpha if alpha is None else alpha,
        alpha_float64_le_hex=(
            record.alpha_float64_le_hex
            if alpha_float64_le_hex is None
            else alpha_float64_le_hex
        ),
        axis_cm1=_copied_vector(record.axis_cm1 if axis_cm1 is None else axis_cm1),
        intensity=_copied_vector(record.intensity if intensity is None else intensity),
        diagnostics=_thaw_json(record.diagnostics) if diagnostics is None else diagnostics,
    )


def _cell_copy(
    cell: NativeCellView,
    *,
    source_spectrum_id: str | None = None,
    source_record_id: str | None = None,
    sample_id: str | None = None,
    class_label: int | None = None,
    source_axis_cm1: np.ndarray | None = None,
    source_intensity: np.ndarray | None = None,
    perturbation_id: str | None = None,
    state_digest=_UNSET,
    status=_UNSET,
    reason_code=_UNSET,
    records: tuple[NativeRecordView, ...] | object = _UNSET,
) -> NativeCellView:
    return NativeCellView(
        source_spectrum_id=cell.source_spectrum_id if source_spectrum_id is None else source_spectrum_id,
        source_record_id=cell.source_record_id if source_record_id is None else source_record_id,
        sample_id=cell.sample_id if sample_id is None else sample_id,
        class_label=cell.class_label if class_label is None else class_label,
        source_axis_cm1=_copied_vector(cell.source_axis_cm1 if source_axis_cm1 is None else source_axis_cm1),
        source_intensity=_copied_vector(cell.source_intensity if source_intensity is None else source_intensity),
        perturbation_id=cell.perturbation_id if perturbation_id is None else perturbation_id,
        state_digest=cell.state_digest if state_digest is _UNSET else state_digest,
        status=cell.status if status is _UNSET else status,
        reason_code=cell.reason_code if reason_code is _UNSET else reason_code,
        records=cell.records if records is _UNSET else records,
    )


def _count_classes(cells: tuple[NativeCellView, ...], perturbation_id: str) -> int:
    return len(
        {
            cell.class_label
            for cell in cells
            if cell.perturbation_id == perturbation_id and cell.status is CellStatus.COMPLETE
        }
    )


def _required_count(total: int, threshold: float) -> int:
    return int((Decimal(total) * Decimal(str(threshold))).to_integral_value(rounding=ROUND_CEILING))


def _synthetic_complete_record(alpha: float, source_axis: np.ndarray, source_intensity: np.ndarray) -> NativeRecordView:
    return NativeRecordView(
        output_spectrum_id=f"synthetic-{struct.pack('<d', alpha).hex()}",
        alpha=alpha,
        alpha_float64_le_hex=struct.pack("<d", alpha).hex(),
        axis_cm1=_copied_vector(source_axis),
        intensity=_copied_vector(source_intensity),
        diagnostics={"placeholder": float(alpha)},
    )


def _synthetic_complete_cell(
    *,
    source_index: int,
    class_label: int,
    perturbation_id: str,
    records: tuple[NativeRecordView, ...],
    source_axis: np.ndarray,
    source_intensity: np.ndarray,
) -> NativeCellView:
    return NativeCellView(
        source_spectrum_id=f"synthetic-spectrum-{source_index}",
        source_record_id=f"synthetic-record-{source_index}",
        sample_id=f"sample-{class_label}",
        class_label=class_label,
        source_axis_cm1=_copied_vector(source_axis),
        source_intensity=_copied_vector(source_intensity),
        perturbation_id=perturbation_id,
        state_digest="a" * 64,
        status=CellStatus.COMPLETE,
        reason_code=None,
        records=records,
    )


def _synthetic_deferred_cell(
    *,
    source_index: int,
    class_label: int,
    perturbation_id: str,
    source_axis: np.ndarray,
    source_intensity: np.ndarray,
    reason_code: str,
) -> NativeCellView:
    return NativeCellView(
        source_spectrum_id=f"synthetic-spectrum-{source_index}",
        source_record_id=f"synthetic-record-{source_index}",
        sample_id=f"sample-{class_label}",
        class_label=class_label,
        source_axis_cm1=_copied_vector(source_axis),
        source_intensity=_copied_vector(source_intensity),
        perturbation_id=perturbation_id,
        state_digest=None,
        status=CellStatus.NOT_APPLICABLE,
        reason_code=reason_code,
        records=(),
    )


def _synthetic_failed_cell(
    *,
    source_index: int,
    class_label: int,
    perturbation_id: str,
    source_axis: np.ndarray,
    source_intensity: np.ndarray,
) -> NativeCellView:
    return NativeCellView(
        source_spectrum_id=f"synthetic-spectrum-{source_index}",
        source_record_id=f"synthetic-record-{source_index}",
        sample_id=f"sample-{class_label}",
        class_label=class_label,
        source_axis_cm1=_copied_vector(source_axis),
        source_intensity=_copied_vector(source_intensity),
        perturbation_id=perturbation_id,
        state_digest=None,
        status=CellStatus.FAILED,
        reason_code=None,
        records=(),
    )


def _aggregate_cells(
    *,
    selected_source_count: int,
    selected_class_count: int,
    complete_ids_by_source: dict[int, set[str]] | None = None,
    failed_ids_by_source: dict[int, set[str]] | None = None,
    deferred_reason_code: str = "missing_explicit_baseline",
    peak_na_ids_by_source: dict[int, dict[str, str]] | None = None,
) -> tuple[NativeCellView, ...]:
    complete_ids_by_source = complete_ids_by_source or {}
    failed_ids_by_source = failed_ids_by_source or {}
    peak_na_ids_by_source = peak_na_ids_by_source or {}
    source_axis = np.linspace(100.0, 200.0, 9, dtype="<f8")
    source_intensity = np.linspace(1.0, 2.0, 9, dtype="<f8")
    shared_records = tuple(
        _synthetic_complete_record(alpha, source_axis, source_intensity)
        for alpha in fixture_sweep().alpha_grid
    )
    cells: list[NativeCellView] = []
    for source_index in range(selected_source_count):
        class_label = source_index % selected_class_count
        complete_ids = complete_ids_by_source.get(source_index, set(COMPLETE_IDS))
        failed_ids = failed_ids_by_source.get(source_index, set())
        peak_na_ids = peak_na_ids_by_source.get(source_index, {})
        for perturbation_id in ALL_IDS:
            if perturbation_id in {"p06", "p07"}:
                cells.append(
                    _synthetic_deferred_cell(
                        source_index=source_index,
                        class_label=class_label,
                        perturbation_id=perturbation_id,
                        source_axis=source_axis,
                        source_intensity=source_intensity,
                        reason_code=deferred_reason_code,
                    )
                )
            elif perturbation_id in failed_ids:
                cells.append(
                    _synthetic_failed_cell(
                        source_index=source_index,
                        class_label=class_label,
                        perturbation_id=perturbation_id,
                        source_axis=source_axis,
                        source_intensity=source_intensity,
                    )
                )
            elif perturbation_id in peak_na_ids:
                cells.append(
                    _synthetic_deferred_cell(
                        source_index=source_index,
                        class_label=class_label,
                        perturbation_id=perturbation_id,
                        source_axis=source_axis,
                        source_intensity=source_intensity,
                        reason_code=peak_na_ids[perturbation_id],
                    )
                )
            elif perturbation_id in complete_ids:
                cells.append(
                    _synthetic_complete_cell(
                        source_index=source_index,
                        class_label=class_label,
                        perturbation_id=perturbation_id,
                        records=shared_records,
                        source_axis=source_axis,
                        source_intensity=source_intensity,
                    )
                )
            else:
                cells.append(
                    _synthetic_deferred_cell(
                        source_index=source_index,
                        class_label=class_label,
                        perturbation_id=perturbation_id,
                        source_axis=source_axis,
                        source_intensity=source_intensity,
                        reason_code="unexpected-not-applicable",
                    )
                )
    return tuple(cells)


class RunnerGatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sweep = fixture_sweep()

    def test_all_real_complete_fixture_cells_pass_and_return_finite_immutable_witnesses(self) -> None:
        for perturbation_id in COMPLETE_IDS:
            with self.subTest(perturbation_id=perturbation_id):
                cell = complete_fixture_cell(perturbation_id)
                gate = validate_cell_native_gate(
                    cell,
                    self.sweep,
                    tolerance=1e-12,
                )
                self.assertIsInstance(gate, CellGateResult)
                self.assertTrue(gate.passed)
                self.assertEqual(gate.perturbation_id, perturbation_id)
                self.assertIsInstance(gate.witness, MappingProxyType)
                for value in gate.witness.values():
                    self._assert_finite_json(value)

    def test_gate_mutation_table_hits_stable_paths(self) -> None:
        for perturbation_id, mutate, expected_path in GATE_MUTATIONS:
            with self.subTest(perturbation_id=perturbation_id):
                with self.assertRaisesRegex(Phase1GateError, expected_path):
                    validate_cell_native_gate(
                        mutate(complete_fixture_cell(perturbation_id)),
                        self.sweep,
                        tolerance=1e-12,
                    )

    def test_p09_uses_sigma_times_noise_mean_square_formula(self) -> None:
        cell = complete_fixture_cell("p09")
        gate = validate_cell_native_gate(cell, self.sweep, tolerance=1e-12)
        sigma = gate.witness["sigma"]
        noise_mean_square = gate.witness["noise_mean_square"]
        observed = gate.witness["observed_residual_rms"]
        for sig, nms, rms in zip(sigma[1:], noise_mean_square[1:], observed[1:], strict=True):
            self.assertAlmostEqual(rms, sig * math.sqrt(nms), places=12)

    def test_stable_population_rms_handles_large_and_tiny_values(self) -> None:
        huge = np.asarray([1e308, -1e308], dtype="<f8")
        tiny = np.asarray([5e-324, 5e-324], dtype="<f8")
        self.assertTrue(math.isfinite(stable_population_rms(huge)))
        self.assertEqual(stable_population_rms(tiny), 5e-324)

    def test_stable_population_rms_rejects_invalid_inputs(self) -> None:
        invalid_cases = (
            ("values", [1.0, 2.0]),
            ("values.dimension", np.ones((2, 2), dtype="<f8")),
            ("values.shape", np.asarray([], dtype="<f8")),
            ("values.finite", np.asarray([1.0, np.nan], dtype="<f8")),
        )
        for expected_path, value in invalid_cases:
            with self.subTest(path=expected_path):
                with self.assertRaisesRegex(Phase1GateError, expected_path):
                    stable_population_rms(value)

    def test_alpha_zero_identity_duplicate_ids_relative_tolerance_and_nan_rejection(self) -> None:
        cell = complete_fixture_cell("p08")
        bad_zero = _cell_copy(
            cell,
            records=(
                _record_copy(cell.records[0], intensity=cell.records[0].intensity + 1.0),
                *cell.records[1:],
            ),
        )
        with self.assertRaisesRegex(Phase1GateError, "native_gate.alpha_zero.intensity"):
            validate_cell_native_gate(bad_zero, self.sweep, tolerance=1e-12)

        duplicate_ids = _cell_copy(
            cell,
            records=(
                cell.records[0],
                _record_copy(cell.records[1], output_spectrum_id=cell.records[0].output_spectrum_id),
                *cell.records[2:],
            ),
        )
        with self.assertRaisesRegex(Phase1GateError, "native_gate.output_spectrum_id"):
            validate_cell_native_gate(duplicate_ids, self.sweep, tolerance=1e-12)

        with self.assertRaisesRegex(Phase1GateError, "tolerance"):
            validate_cell_native_gate(cell, self.sweep, tolerance=True)
        with self.assertRaisesRegex(Phase1GateError, "tolerance"):
            validate_cell_native_gate(cell, self.sweep, tolerance="0.1")
        with self.assertRaisesRegex(Phase1GateError, "tolerance"):
            validate_cell_native_gate(cell, self.sweep, tolerance=float("nan"))
        with self.assertRaisesRegex(Phase1GateError, "tolerance"):
            validate_cell_native_gate(cell, self.sweep, tolerance=-1.0)

        p01 = complete_fixture_cell("p01")
        target = p01.records[-1]
        baseline_gate = validate_cell_native_gate(p01, self.sweep, tolerance=1e-12)
        expected = baseline_gate.witness["observed_removed_component_area"][-1]
        boundary = expected / (1.0 - 1e-12)
        boundary_cell = _cell_copy(
            p01,
            records=(
                *p01.records[:-1],
                _record_copy(
                    target,
                    diagnostics={"removed_component_area": boundary},
                ),
            ),
        )
        self.assertTrue(
            validate_cell_native_gate(boundary_cell, self.sweep, tolerance=1e-12).passed
        )
        above = math.nextafter(boundary, math.inf)
        above_cell = _cell_copy(
            p01,
            records=(
                *p01.records[:-1],
                _record_copy(
                    target,
                    diagnostics={"removed_component_area": above},
                ),
            ),
        )
        with self.assertRaisesRegex(Phase1GateError, "native_gate.p01.removed_component_area"):
            validate_cell_native_gate(above_cell, self.sweep, tolerance=1e-12)

        with self.assertRaisesRegex(Phase1GateError, "diagnostics.bad"):
            _cell_copy(
                p01,
                records=(
                    *p01.records[:-1],
                    _record_copy(target, diagnostics={"bad": float("nan")}),
                ),
            )

    def test_alpha_bytes_preserve_negative_zero(self) -> None:
        record = NativeRecordView(
            output_spectrum_id="negzero",
            alpha=-0.0,
            alpha_float64_le_hex=struct.pack("<d", -0.0).hex(),
            axis_cm1=_copied_vector(np.asarray([1.0, 2.0], dtype="<f8")),
            intensity=_copied_vector(np.asarray([3.0, 4.0], dtype="<f8")),
            diagnostics={},
        )
        self.assertEqual(record.alpha_float64_le_hex, "0000000000000080")

    def test_public_gate_rejects_non_authoritative_sweep_config(self) -> None:
        cell = complete_fixture_cell("p01")
        bad_sweep = replace(
            self.sweep,
            sha256="0" * 64,
        )
        with self.assertRaisesRegex(Phase1GateError, "sweep.sha256"):
            validate_cell_native_gate(cell, bad_sweep, tolerance=1e-12)

    def test_missing_required_diagnostic_keys_raise_normalized_paths(self) -> None:
        p02 = complete_fixture_cell("p02")
        bad_peak = _cell_copy(
            p02,
            records=(
                *p02.records[:-1],
                _record_copy(p02.records[-1], diagnostics={}),
            ),
        )
        with self.assertRaisesRegex(Phase1GateError, "native_gate.p02.selected_count"):
            validate_cell_native_gate(bad_peak, self.sweep, tolerance=1e-12)

        p08 = complete_fixture_cell("p08")
        bad_residual = _cell_copy(
            p08,
            records=(
                *p08.records[:-1],
                _record_copy(p08.records[-1], diagnostics={}),
            ),
        )
        with self.assertRaisesRegex(Phase1GateError, "native_gate.p08.residual_rms"):
            validate_cell_native_gate(bad_residual, self.sweep, tolerance=1e-12)

    def test_native_cell_view_makes_defensive_copies_without_aliasing_or_mutation(self) -> None:
        base = complete_fixture_cell("p09")
        with self.assertRaises(ValueError):
            base.source_axis_cm1[0] = 1.0
        with self.assertRaises(ValueError):
            base.records[0].intensity[0] = 1.0

        with patch.object(phase1_perturbations, "validate_cell_native_gate") as gate_mock:
            gate_mock.return_value = CellGateResult(
                perturbation_id="p09",
                passed=True,
                witness={"ok": 1},
            )
            with patch.object(phase1_perturbations, "native_cell_view", wraps=native_cell_view) as view_mock:
                with self.subTest("fresh-view-does-not-alias-live-cell"):
                    from phase1_runner_helpers import _fixture_complete_phase1_cell  # noqa: PLC0415

                    live = _fixture_complete_phase1_cell("p09")
                    live_source_axis_before = live.source.spectrum.axis_cm1.copy()
                    config = fixture_config(subset_size=8, shard_source_count=4)
                    gate_cell = phase1_perturbations.run_perturbation_cell(
                        live.source,
                        "p09",
                        config,
                        self.sweep,
                    )
                    self.assertIsNotNone(gate_cell)
                    self.assertTrue(np.array_equal(live.source.spectrum.axis_cm1, live_source_axis_before))
                    self.assertGreaterEqual(view_mock.call_count, 1)

    def test_runner_integration_success_stores_witness_and_native_invalid_result_becomes_failed(self) -> None:
        from phase1_runner_helpers import _fixture_complete_phase1_cell  # noqa: PLC0415
        config = fixture_config(subset_size=8, shard_source_count=4)

        with self.subTest("success"):
            base_cell = _fixture_complete_phase1_cell("p01")
            expected_witness = {"validated": (0.0, 1.0)}
            with patch.object(
                phase1_perturbations,
                "validate_cell_native_gate",
                return_value=CellGateResult("p01", True, expected_witness),
            ):
                rerun = phase1_perturbations.run_perturbation_cell(
                    base_cell.source,
                    "p01",
                    config,
                    self.sweep,
                )
                self.assertEqual(rerun.status, CellStatus.COMPLETE)
                self.assertEqual(dict(rerun.evidence.native_gate), expected_witness)

        with self.subTest("gate-failure-downgrades-to-failed"):
            base_cell = _fixture_complete_phase1_cell("p01")
            with patch.object(
                phase1_perturbations,
                "validate_cell_native_gate",
                side_effect=Phase1GateError("native_gate.p01.removed_component_area", "fixture"),
            ):
                rerun = phase1_perturbations.run_perturbation_cell(
                    base_cell.source,
                    "p01",
                    config,
                    self.sweep,
                )
                self.assertEqual(rerun.status, CellStatus.FAILED)
                self.assertEqual(rerun.records, ())
                self.assertIsNotNone(rerun.state)
                self.assertEqual(rerun.evidence.exception_type, "Phase1GateError")
                self.assertEqual(
                    rerun.evidence.exception_path,
                    "native_gate.p01.removed_component_area",
                )

    def test_native_cell_view_accepts_retained_state_for_real_p05_not_applicable_and_gate_failed_cells(self) -> None:
        from phase1_runner_helpers import _fixture_complete_phase1_cell  # noqa: PLC0415
        from tests.test_phase1_runner_execution import fixture_source, fixture_config as execution_fixture_config  # noqa: PLC0415
        from rpe.evaluation import Spectrum1D  # noqa: PLC0415

        execution_config = execution_fixture_config(
            peak_reason_codes=(
                "false_peak_placement_impossible",
                "insufficient_points_for_peak_model",
                "invalid_peak_component",
                "no_detected_peak",
                "nonpositive_intensity_range",
                "zero_false_peak_insertion_capacity",
            )
        )
        source = fixture_source()
        single_peak_source = replace(
            source,
            spectrum=Spectrum1D(
                spectrum_id=source.spectrum.spectrum_id,
                sample_id=source.spectrum.sample_id,
                axis_cm1=np.array(
                    (100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0),
                    dtype="<f8",
                ),
                intensity=np.array(
                    (0.2, 0.5, 1.5, 4.0, 1.5, 0.5, 0.2),
                    dtype="<f8",
                ),
            ),
        )
        na_cell = phase1_perturbations.run_perturbation_cell(
            single_peak_source,
            "p05",
            execution_config,
            self.sweep,
        )
        na_view = native_cell_view(na_cell)
        self.assertEqual(na_view.status, CellStatus.NOT_APPLICABLE)
        self.assertIsNotNone(na_view.state_digest)

        base_cell = _fixture_complete_phase1_cell("p01")
        config = fixture_config(subset_size=8, shard_source_count=4)
        with patch.object(
            phase1_perturbations,
            "validate_cell_native_gate",
            side_effect=Phase1GateError("native_gate.p01.removed_component_area", "fixture"),
        ):
            failed_cell = phase1_perturbations.run_perturbation_cell(
                base_cell.source,
                "p01",
                config,
                self.sweep,
            )
        failed_view = native_cell_view(failed_cell)
        self.assertEqual(failed_view.status, CellStatus.FAILED)
        self.assertIsNotNone(failed_view.state_digest)

    def test_core_gate_small_denominator_threshold_cases(self) -> None:
        config = replace(
            fixture_config(subset_size=40, shard_source_count=4),
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
        cells = _aggregate_cells(selected_source_count=40, selected_class_count=20)
        result = evaluate_core_gate(cells, config, selected_source_count=40, selected_class_count=20)
        self.assertEqual(result.core_dataset_gate, "pass")

        for perturbation_id in ("p01", "p02", "p03", "p04"):
            with self.subTest(perturbation_id=perturbation_id):
                required = _required_count(40, 0.95)
                drop_sources = set(range(required - 1, 40))
                cells = _aggregate_cells(
                    selected_source_count=40,
                    selected_class_count=20,
                    complete_ids_by_source={
                        index: set(COMPLETE_IDS) - {perturbation_id}
                        if index in drop_sources
                        else set(COMPLETE_IDS)
                        for index in range(40)
                    },
                )
                failed = evaluate_core_gate(cells, config, selected_source_count=40, selected_class_count=20)
                self.assertTrue(any(code.startswith(f"threshold.{perturbation_id}.source") for code in failed.failures))

                one_class_drop = {index for index in range(40) if index % 20 == 19}
                cells = _aggregate_cells(
                    selected_source_count=40,
                    selected_class_count=20,
                    complete_ids_by_source={
                        index: set(COMPLETE_IDS) - {perturbation_id}
                        if index in one_class_drop
                        else set(COMPLETE_IDS)
                        for index in range(40)
                    },
                )
                equality = evaluate_core_gate(cells, config, selected_source_count=40, selected_class_count=20)
                self.assertFalse(any(code.startswith(f"threshold.{perturbation_id}.class") for code in equality.failures))

                two_class_drop = {index for index in range(40) if index % 20 in {18, 19}}
                cells = _aggregate_cells(
                    selected_source_count=40,
                    selected_class_count=20,
                    complete_ids_by_source={
                        index: set(COMPLETE_IDS) - {perturbation_id}
                        if index in two_class_drop
                        else set(COMPLETE_IDS)
                        for index in range(40)
                    },
                )
                failed = evaluate_core_gate(cells, config, selected_source_count=40, selected_class_count=20)
                self.assertTrue(any(code.startswith(f"threshold.{perturbation_id}.class") for code in failed.failures))

        required = _required_count(40, 0.9)
        self.assertEqual(required, 36)
        missing = set(range(required, 40))
        cells = _aggregate_cells(
            selected_source_count=40,
            selected_class_count=20,
            complete_ids_by_source={
                index: set(COMPLETE_IDS) - {"p05"} if index in missing else set(COMPLETE_IDS)
                for index in range(40)
            },
        )
        passed = evaluate_core_gate(cells, config, selected_source_count=40, selected_class_count=20)
        self.assertFalse(any(code.startswith("threshold.p05.source") for code in passed.failures))
        fail_cells = _aggregate_cells(
            selected_source_count=40,
            selected_class_count=20,
            complete_ids_by_source={
                index: set(COMPLETE_IDS) - {"p05"} if index in missing | {35} else set(COMPLETE_IDS)
                for index in range(40)
            },
        )
        failed = evaluate_core_gate(fail_cells, config, selected_source_count=40, selected_class_count=20)
        self.assertTrue(any(code.startswith("threshold.p05.source") for code in failed.failures))

        p05_one_class_drop = {index for index in range(40) if index % 20 == 19}
        cells = _aggregate_cells(
            selected_source_count=40,
            selected_class_count=20,
            complete_ids_by_source={
                index: set(COMPLETE_IDS) - {"p05"} if index in p05_one_class_drop else set(COMPLETE_IDS)
                for index in range(40)
            },
        )
        equality = evaluate_core_gate(cells, config, selected_source_count=40, selected_class_count=20)
        self.assertFalse(any(code.startswith("threshold.p05.class") for code in equality.failures))

        p05_three_class_drop = {index for index in range(40) if index % 20 in {17, 18, 19}}
        cells = _aggregate_cells(
            selected_source_count=40,
            selected_class_count=20,
            complete_ids_by_source={
                index: set(COMPLETE_IDS) - {"p05"} if index in p05_three_class_drop else set(COMPLETE_IDS)
                for index in range(40)
            },
        )
        failed = evaluate_core_gate(cells, config, selected_source_count=40, selected_class_count=20)
        self.assertTrue(any(code.startswith("threshold.p05.class") for code in failed.failures))

        for perturbation_id in ("p08", "p09", "p10", "p11", "p12"):
            with self.subTest(perturbation_id=perturbation_id):
                cells = _aggregate_cells(
                    selected_source_count=40,
                    selected_class_count=20,
                    complete_ids_by_source={
                        **{index: set(COMPLETE_IDS) for index in range(40)},
                        39: set(COMPLETE_IDS) - {perturbation_id},
                    },
                )
                failed = evaluate_core_gate(cells, config, selected_source_count=40, selected_class_count=20)
                self.assertTrue(any(code.startswith(f"threshold.{perturbation_id}.source") for code in failed.failures))

    def test_core_gate_rejects_duplicate_missing_unknown_and_inconsistent_coverage(self) -> None:
        config = replace(
            fixture_config(subset_size=2, shard_source_count=1),
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
        cells = list(_aggregate_cells(selected_source_count=2, selected_class_count=2))
        duplicate = tuple(cells + [cells[0]])
        result = evaluate_core_gate(duplicate, config, selected_source_count=2, selected_class_count=2)
        self.assertTrue(any(code.startswith("coverage.duplicate") for code in result.failures))

        missing = tuple(cells[:-1])
        result = evaluate_core_gate(missing, config, selected_source_count=2, selected_class_count=2)
        self.assertTrue(any(code.startswith("coverage.missing") for code in result.failures))

        unknown = list(cells)
        unknown[0] = _cell_copy(unknown[0], perturbation_id="p99")
        result = evaluate_core_gate(tuple(unknown), config, selected_source_count=2, selected_class_count=2)
        self.assertTrue(any(code.startswith("coverage.unknown_perturbation_id") for code in result.failures))

        inconsistent = list(cells)
        inconsistent[1] = _cell_copy(inconsistent[1], sample_id="other-sample")
        result = evaluate_core_gate(tuple(inconsistent), config, selected_source_count=2, selected_class_count=2)
        self.assertTrue(any(code.startswith("coverage.inconsistent_source_metadata") for code in result.failures))

    def test_core_gate_rejects_bad_deferred_status_reason_peak_allowlist_p08_p12_na_failed_count_and_denominator_mismatch(self) -> None:
        config = replace(
            fixture_config(subset_size=2, shard_source_count=1),
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
        base = _aggregate_cells(selected_source_count=2, selected_class_count=2)
        bad_deferred = list(base)
        bad_deferred[5] = _cell_copy(
            bad_deferred[5],
            status=CellStatus.COMPLETE,
            state_digest="a" * 64,
            reason_code=None,
            records=bad_deferred[0].records,
        )
        result = evaluate_core_gate(tuple(bad_deferred), config, selected_source_count=2, selected_class_count=2)
        self.assertTrue(any(code.startswith("deferred.") for code in result.failures))

        peak_na = _aggregate_cells(
            selected_source_count=2,
            selected_class_count=2,
            peak_na_ids_by_source={0: {"p01": "false_peak_placement_impossible"}},
        )
        result = evaluate_core_gate(peak_na, config, selected_source_count=2, selected_class_count=2)
        self.assertFalse(any(code.startswith("coverage.") for code in result.failures))

        peak_na_bad = _aggregate_cells(
            selected_source_count=2,
            selected_class_count=2,
            peak_na_ids_by_source={0: {"p01": "not-allowed"}},
        )
        result = evaluate_core_gate(peak_na_bad, config, selected_source_count=2, selected_class_count=2)
        self.assertTrue(any(code.startswith("coverage.invalid_peak_not_applicable_reason") for code in result.failures))

        p08_na = _aggregate_cells(
            selected_source_count=2,
            selected_class_count=2,
            peak_na_ids_by_source={0: {"p08": "not-allowed"}},
        )
        result = evaluate_core_gate(p08_na, config, selected_source_count=2, selected_class_count=2)
        self.assertTrue(any(code.startswith("coverage.unexpected_not_applicable") for code in result.failures))

        failed_cells = _aggregate_cells(
            selected_source_count=2,
            selected_class_count=2,
            failed_ids_by_source={0: {"p08"}},
        )
        result = evaluate_core_gate(failed_cells, config, selected_source_count=2, selected_class_count=2)
        self.assertTrue(any(code.startswith("failed_cell_count") for code in result.failures))

        result = evaluate_core_gate(base, config, selected_source_count=3, selected_class_count=2)
        self.assertTrue(any(code.startswith("coverage.source_count") for code in result.failures))
        result = evaluate_core_gate(base, config, selected_source_count=2, selected_class_count=3)
        self.assertTrue(any(code.startswith("coverage.class_count") for code in result.failures))

    def test_core_gate_streams_lazy_large_iterable_and_counts_exact_deferred_rows(self) -> None:
        config = replace(
            fixture_config(subset_size=10000, shard_source_count=100),
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
        source_axis = np.linspace(100.0, 200.0, 9, dtype="<f8")
        source_intensity = np.linspace(1.0, 2.0, 9, dtype="<f8")
        shared_records = tuple(
            _synthetic_complete_record(alpha, source_axis, source_intensity)
            for alpha in fixture_sweep().alpha_grid
        )

        class LazyCells:
            def __iter__(self_inner):
                for source_index in range(10000):
                    class_label = source_index % 2480
                    for perturbation_id in ALL_IDS:
                        if perturbation_id in {"p06", "p07"}:
                            yield _synthetic_deferred_cell(
                                source_index=source_index,
                                class_label=class_label,
                                perturbation_id=perturbation_id,
                                source_axis=source_axis,
                                source_intensity=source_intensity,
                                reason_code="missing_explicit_baseline",
                            )
                        else:
                            yield _synthetic_complete_cell(
                                source_index=source_index,
                                class_label=class_label,
                                perturbation_id=perturbation_id,
                                records=shared_records,
                                source_axis=source_axis,
                                source_intensity=source_intensity,
                            )

        result = evaluate_core_gate(
            LazyCells(),
            config,
            selected_source_count=10000,
            selected_class_count=2480,
        )
        self.assertIsInstance(result, CoreGateResult)
        self.assertEqual(result.core_dataset_gate, "pass")
        self.assertEqual(result.full_phase1_gate, "deferred_missing_p06_p07")
        self.assertEqual(result.deferred_cell_count, 20000)
        self.assertEqual(result.failed_cell_count, 0)

    def test_core_gate_result_constructor_enforces_exact_counts_and_allowed_gate_pairs(self) -> None:
        complete_counts = {perturbation_id: 0 for perturbation_id in ALL_IDS}
        class_counts = {perturbation_id: 0 for perturbation_id in ALL_IDS}
        result = CoreGateResult(
            core_dataset_gate="pass",
            full_phase1_gate="deferred_missing_p06_p07",
            selected_source_count=1,
            selected_class_count=1,
            complete_cell_counts=complete_counts,
            complete_class_counts=class_counts,
            deferred_cell_count=2,
            failed_cell_count=0,
            failures=(),
        )
        self.assertEqual(result.core_dataset_gate, "pass")

        with self.assertRaisesRegex(Phase1GateError, "complete_cell_counts"):
            CoreGateResult(
                core_dataset_gate="pass",
                full_phase1_gate="deferred_missing_p06_p07",
                selected_source_count=1,
                selected_class_count=1,
                complete_cell_counts={**complete_counts, "extra": 1},
                complete_class_counts=class_counts,
                deferred_cell_count=2,
                failed_cell_count=0,
                failures=(),
            )

        with self.assertRaisesRegex(Phase1GateError, "full_phase1_gate"):
            CoreGateResult(
                core_dataset_gate="pass",
                full_phase1_gate="fail",
                selected_source_count=1,
                selected_class_count=1,
                complete_cell_counts=complete_counts,
                complete_class_counts=class_counts,
                deferred_cell_count=2,
                failed_cell_count=0,
                failures=(),
            )

    def _assert_finite_json(self, value) -> None:
        if isinstance(value, MappingProxyType):
            value = dict(value)
        if isinstance(value, dict):
            for nested in value.values():
                self._assert_finite_json(nested)
            return
        if isinstance(value, tuple):
            for nested in value:
                self._assert_finite_json(nested)
            return
        if isinstance(value, float):
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
