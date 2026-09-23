from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np

from rpe.io.schema import JsonValue
from rpe.perturb import (
    PerturbationSweepConfig,
    PerturbationSweepConfigError,
    derive_perturbed_spectrum_id,
)
from rpe.perturb.sweep import validate_perturbation_sweep_config
from rpe.runner.phase1_config import Phase1CoreConfig
from rpe.runner.phase1_types import CellStatus, Phase1Cell


_HEX_DIGITS = frozenset("0123456789abcdef")
_COMPLETE_IDS = ("p01", "p02", "p03", "p04", "p05", "p08", "p09", "p10", "p11", "p12")
_PERTURBATION_ORDER = tuple(f"p{index:02d}" for index in range(1, 13))
_DEFERRED_IDS = frozenset({"p06", "p07"})
_PEAK_IDS = frozenset({"p01", "p02", "p03", "p04", "p05"})


class Phase1GateError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise Phase1GateError(path, "must be a nonempty string")
    return value


def _nonnegative_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase1GateError(path, "must be a nonnegative integer")
    return value


def _finite_float(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase1GateError(path, "must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise Phase1GateError(path, "must be a finite real number")
    return result


def _lower_hex(path: str, value: object, *, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise Phase1GateError(
            path,
            f"must be a lowercase {length}-character hexadecimal string",
        )
    return value


def _copy_float64_vector(path: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise Phase1GateError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise Phase1GateError(f"{path}.dtype", "must be little-endian float64")
    if value.ndim != 1:
        raise Phase1GateError(f"{path}.dimension", "must be one-dimensional")
    if value.size == 0:
        raise Phase1GateError(f"{path}.shape", "must be nonempty")
    if not np.isfinite(value).all():
        raise Phase1GateError(f"{path}.finite", "contains non-finite values")
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


def _freeze_json(path: str, value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Phase1GateError(path, "must be finite")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(f"{path}[{index}]", item)
            for index, item in enumerate(value)
        )
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in sorted(value.items()):
            frozen_key = _nonempty_string(f"{path}.key", key)
            frozen[frozen_key] = _freeze_json(f"{path}.{frozen_key}", item)
        return MappingProxyType(frozen)
    raise Phase1GateError(path, "must be canonical-JSON-compatible")


def _frozen_json_mapping(path: str, value: object) -> Mapping[str, JsonValue]:
    frozen = _freeze_json(path, value)
    if not isinstance(frozen, Mapping):
        raise Phase1GateError(path, "must be a mapping")
    return frozen


def _diagnostic_value(
    record: "NativeRecordView",
    *,
    operator_path: str,
    key: str,
) -> JsonValue:
    if key not in record.diagnostics:
        raise Phase1GateError(operator_path, f"missing diagnostic {key!r}")
    return record.diagnostics[key]


def _diagnostic_float(
    record: "NativeRecordView",
    *,
    operator_path: str,
    key: str,
    nonnegative: bool = False,
) -> float:
    value = _diagnostic_value(record, operator_path=operator_path, key=key)
    if nonnegative:
        return _finite_nonnegative_float(operator_path, value)
    return _finite_float(operator_path, value)


def _diagnostic_count(
    record: "NativeRecordView",
    *,
    operator_path: str,
    key: str,
) -> int:
    return _nonnegative_int(
        operator_path,
        _diagnostic_value(record, operator_path=operator_path, key=key),
    )


def _strictly_increasing(path: str, values: np.ndarray) -> None:
    if np.any(values[1:] <= values[:-1]):
        raise Phase1GateError(path, "must be strictly increasing")


def _pack_float64_hex(value: float) -> str:
    return struct.pack("<d", value).hex()


def _finite_nonnegative_float(path: str, value: object) -> float:
    result = _finite_float(path, value)
    if result < 0.0:
        raise Phase1GateError(path, "must be nonnegative")
    return result


def _relation_close(observed: float, expected: float, *, tolerance: float) -> bool:
    return abs(observed - expected) <= tolerance * max(1.0, abs(observed), abs(expected))


def _freeze_witness(value: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return _frozen_json_mapping("witness", value)


def _longdouble_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    observed = np.trapezoid(
        np.asarray(y, dtype=np.longdouble),
        np.asarray(x, dtype=np.longdouble),
    )
    result = float(observed)
    if not math.isfinite(result):
        raise Phase1GateError("native_gate.integration", "must remain finite")
    return result


def _records_alpha_hex(sweep: PerturbationSweepConfig) -> tuple[str, ...]:
    return tuple(struct.pack("<d", float(alpha)).hex() for alpha in sweep.alpha_grid)


def _json_ready_sequence(values: list[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class NativeRecordView:
    output_spectrum_id: str
    alpha: float
    alpha_float64_le_hex: str
    axis_cm1: np.ndarray
    intensity: np.ndarray
    diagnostics: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "output_spectrum_id",
            _nonempty_string("output_spectrum_id", self.output_spectrum_id),
        )
        alpha = _finite_float("alpha", self.alpha)
        object.__setattr__(self, "alpha", alpha)
        expected_hex = _pack_float64_hex(alpha)
        alpha_hex = _lower_hex("alpha_float64_le_hex", self.alpha_float64_le_hex, length=16)
        if alpha_hex != expected_hex:
            raise Phase1GateError("alpha_float64_le_hex", "must equal little-endian float64 hex")
        object.__setattr__(self, "alpha_float64_le_hex", alpha_hex)
        object.__setattr__(self, "axis_cm1", _copy_float64_vector("axis_cm1", self.axis_cm1))
        object.__setattr__(self, "intensity", _copy_float64_vector("intensity", self.intensity))
        if self.axis_cm1.shape != self.intensity.shape:
            raise Phase1GateError("intensity.shape", "must match axis_cm1")
        object.__setattr__(
            self,
            "diagnostics",
            _frozen_json_mapping("diagnostics", self.diagnostics),
        )


@dataclass(frozen=True)
class NativeCellView:
    source_spectrum_id: str
    source_record_id: str
    sample_id: str
    class_label: int
    source_axis_cm1: np.ndarray
    source_intensity: np.ndarray
    perturbation_id: str
    state_digest: str | None
    status: CellStatus
    reason_code: str | None
    records: tuple[NativeRecordView, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_spectrum_id",
            _nonempty_string("source_spectrum_id", self.source_spectrum_id),
        )
        object.__setattr__(
            self,
            "source_record_id",
            _nonempty_string("source_record_id", self.source_record_id),
        )
        object.__setattr__(self, "sample_id", _nonempty_string("sample_id", self.sample_id))
        object.__setattr__(
            self,
            "class_label",
            _nonnegative_int("class_label", self.class_label),
        )
        object.__setattr__(
            self,
            "source_axis_cm1",
            _copy_float64_vector("source_axis_cm1", self.source_axis_cm1),
        )
        object.__setattr__(
            self,
            "source_intensity",
            _copy_float64_vector("source_intensity", self.source_intensity),
        )
        if self.source_axis_cm1.shape != self.source_intensity.shape:
            raise Phase1GateError("source_intensity.shape", "must match source_axis_cm1")
        _strictly_increasing("source_axis_cm1", self.source_axis_cm1)
        object.__setattr__(
            self,
            "perturbation_id",
            _nonempty_string("perturbation_id", self.perturbation_id),
        )
        if self.state_digest is not None:
            object.__setattr__(
                self,
                "state_digest",
                _lower_hex("state_digest", self.state_digest, length=64),
            )
        if not isinstance(self.status, CellStatus):
            raise Phase1GateError("status", "must be CellStatus")
        if not isinstance(self.records, tuple):
            raise Phase1GateError("records", "must be a tuple")
        validated_records: list[NativeRecordView] = []
        for index, record in enumerate(self.records):
            if not isinstance(record, NativeRecordView):
                raise Phase1GateError(f"records[{index}]", "must be NativeRecordView")
            validated_records.append(record)
            if record.axis_cm1.shape != record.intensity.shape:
                raise Phase1GateError(f"records[{index}].intensity.shape", "must match axis_cm1")
            if record.axis_cm1.shape != self.source_axis_cm1.shape:
                raise Phase1GateError(f"records[{index}].axis_cm1.shape", "must match source_axis_cm1")
            if record.intensity.shape != self.source_intensity.shape:
                raise Phase1GateError(f"records[{index}].intensity.shape", "must match source_intensity")
        object.__setattr__(self, "records", tuple(validated_records))

        if self.status is CellStatus.COMPLETE:
            if self.state_digest is None:
                raise Phase1GateError("state_digest", "must be present when status is complete")
            if self.reason_code is not None:
                raise Phase1GateError("reason_code", "must be absent when status is complete")
            if len(self.records) != 9:
                raise Phase1GateError("records", "must contain exactly nine outputs when status is complete")
        elif self.status is CellStatus.NOT_APPLICABLE:
            object.__setattr__(self, "reason_code", _nonempty_string("reason_code", self.reason_code))
            if self.records:
                raise Phase1GateError("records", "must be empty when status is not_applicable")
        else:
            if self.reason_code is not None:
                raise Phase1GateError("reason_code", "must be absent when status is failed")
            if self.records:
                raise Phase1GateError("records", "must be empty when status is failed")


@dataclass(frozen=True)
class CellGateResult:
    perturbation_id: str
    passed: bool
    witness: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "perturbation_id",
            _nonempty_string("perturbation_id", self.perturbation_id),
        )
        if not isinstance(self.passed, bool):
            raise Phase1GateError("passed", "must be bool")
        object.__setattr__(self, "witness", _freeze_witness(dict(self.witness)))


@dataclass(frozen=True)
class CoreGateResult:
    core_dataset_gate: str
    full_phase1_gate: str
    selected_source_count: int
    selected_class_count: int
    complete_cell_counts: Mapping[str, int]
    complete_class_counts: Mapping[str, int]
    deferred_cell_count: int
    failed_cell_count: int
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "core_dataset_gate",
            _nonempty_string("core_dataset_gate", self.core_dataset_gate),
        )
        object.__setattr__(
            self,
            "full_phase1_gate",
            _nonempty_string("full_phase1_gate", self.full_phase1_gate),
        )
        object.__setattr__(
            self,
            "selected_source_count",
            _nonnegative_int("selected_source_count", self.selected_source_count),
        )
        object.__setattr__(
            self,
            "selected_class_count",
            _nonnegative_int("selected_class_count", self.selected_class_count),
        )
        object.__setattr__(
            self,
            "complete_cell_counts",
            self._freeze_count_mapping("complete_cell_counts", self.complete_cell_counts),
        )
        object.__setattr__(
            self,
            "complete_class_counts",
            self._freeze_count_mapping("complete_class_counts", self.complete_class_counts),
        )
        object.__setattr__(
            self,
            "deferred_cell_count",
            _nonnegative_int("deferred_cell_count", self.deferred_cell_count),
        )
        object.__setattr__(
            self,
            "failed_cell_count",
            _nonnegative_int("failed_cell_count", self.failed_cell_count),
        )
        if not isinstance(self.failures, tuple):
            raise Phase1GateError("failures", "must be a tuple")
        frozen_failures = tuple(_nonempty_string(f"failures[{index}]", value) for index, value in enumerate(self.failures))
        if tuple(sorted(set(frozen_failures))) != frozen_failures:
            raise Phase1GateError("failures", "must be sorted and unique")
        object.__setattr__(self, "failures", frozen_failures)
        if self.core_dataset_gate not in {"pass", "fail"}:
            raise Phase1GateError("core_dataset_gate", "must be 'pass' or 'fail'")
        if self.full_phase1_gate not in {"deferred_missing_p06_p07", "fail"}:
            raise Phase1GateError(
                "full_phase1_gate",
                "must be 'deferred_missing_p06_p07' or 'fail'",
            )
        if self.core_dataset_gate == "pass" and self.full_phase1_gate != "deferred_missing_p06_p07":
            raise Phase1GateError("full_phase1_gate", "core pass requires deferred_missing_p06_p07")
        if self.core_dataset_gate == "fail" and self.full_phase1_gate != "fail":
            raise Phase1GateError("full_phase1_gate", "core fail requires fail")

    @staticmethod
    def _freeze_count_mapping(path: str, value: Mapping[str, int]) -> Mapping[str, int]:
        if not isinstance(value, Mapping):
            raise Phase1GateError(path, "must be a mapping")
        if tuple(value.keys()) != _PERTURBATION_ORDER:
            raise Phase1GateError(path, "must contain exact P01-P12 keys in order")
        frozen: dict[str, int] = {}
        for perturbation_id in _PERTURBATION_ORDER:
            if perturbation_id not in value:
                raise Phase1GateError(path, "must contain all P01-P12 keys")
            frozen[perturbation_id] = _nonnegative_int(f"{path}.{perturbation_id}", value[perturbation_id])
        if tuple(frozen.keys()) != _PERTURBATION_ORDER:
            raise Phase1GateError(path, "must follow exact P01-P12 order")
        return MappingProxyType(frozen)


def native_cell_view(cell: Phase1Cell) -> NativeCellView:
    if not isinstance(cell, Phase1Cell):
        raise Phase1GateError("cell", "must be Phase1Cell")
    source_axis = _copy_float64_vector("source_axis_cm1", cell.source.spectrum.axis_cm1)
    source_intensity = _copy_float64_vector("source_intensity", cell.source.spectrum.intensity)
    if np.shares_memory(source_axis, cell.source.spectrum.axis_cm1):
        raise Phase1GateError("source_axis_cm1", "must not alias the live source axis")
    if np.shares_memory(source_intensity, cell.source.spectrum.intensity):
        raise Phase1GateError("source_intensity", "must not alias the live source intensity")

    records: list[NativeRecordView] = []
    for index, record in enumerate(cell.records):
        output_axis = _copy_float64_vector(f"records[{index}].axis_cm1", record.result.output.axis_cm1)
        output_intensity = _copy_float64_vector(f"records[{index}].intensity", record.result.output.intensity)
        if np.shares_memory(output_axis, cell.source.spectrum.axis_cm1):
            raise Phase1GateError(f"records[{index}].axis_cm1", "must not alias the live source axis")
        if np.shares_memory(output_intensity, cell.source.spectrum.intensity):
            raise Phase1GateError(f"records[{index}].intensity", "must not alias the live source intensity")
        if np.shares_memory(output_axis, source_axis) or np.shares_memory(output_intensity, source_intensity):
            raise Phase1GateError(f"records[{index}]", "must not share memory with copied source arrays")
        records.append(
            NativeRecordView(
                output_spectrum_id=record.result.output.spectrum_id,
                alpha=record.result.alpha,
                alpha_float64_le_hex=record.alpha_float64_le_hex,
                axis_cm1=output_axis,
                intensity=output_intensity,
                diagnostics=record.result.diagnostics,
            )
        )

    state_digest = None if cell.state is None else cell.state.state_digest
    return NativeCellView(
        source_spectrum_id=cell.source.spectrum.spectrum_id,
        source_record_id=cell.source.selection.record_id,
        sample_id=cell.source.selection.sample_id,
        class_label=cell.source.selection.class_label,
        source_axis_cm1=source_axis,
        source_intensity=source_intensity,
        perturbation_id=cell.perturbation_id,
        state_digest=state_digest,
        status=cell.status,
        reason_code=cell.reason_code,
        records=tuple(records),
    )


def stable_population_rms(values: np.ndarray) -> float:
    if not isinstance(values, np.ndarray):
        raise Phase1GateError("values", "must be a numpy.ndarray")
    if values.ndim != 1:
        raise Phase1GateError("values.dimension", "must be one-dimensional")
    if values.size == 0:
        raise Phase1GateError("values.shape", "must be nonempty")
    if not np.isfinite(values).all():
        raise Phase1GateError("values.finite", "contains non-finite values")
    scale = 0.0
    sumsq = 1.0
    for index, raw_value in enumerate(values):
        value = abs(float(raw_value))
        if not math.isfinite(value):
            raise Phase1GateError(f"values[{index}]", "must be finite")
        if value == 0.0:
            continue
        if scale < value:
            ratio = 0.0 if scale == 0.0 else scale / value
            sumsq = 1.0 + sumsq * ratio * ratio
            scale = value
        else:
            ratio = value / scale
            sumsq += ratio * ratio
    if scale == 0.0:
        return 0.0
    result = scale * math.sqrt(sumsq / float(values.size))
    if not math.isfinite(result):
        raise Phase1GateError("values", "must have finite RMS")
    return result


def _monotone_zero_then_positive(path: str, values: list[float], *, strict_positive: bool, allow_plateau: bool = False) -> None:
    if values[0] != 0.0:
        raise Phase1GateError(path, "must be zero at alpha=0")
    previous = 0.0
    for index, value in enumerate(values[1:], start=1):
        if strict_positive and value <= 0.0:
            raise Phase1GateError(path, f"must be positive at alpha index {index}")
        if allow_plateau:
            if value < previous:
                raise Phase1GateError(path, "must be nondecreasing over positive alphas")
        elif value <= previous:
            raise Phase1GateError(path, "must be strictly increasing over positive alphas")
        previous = value


def _nonnegative_integer_sequence(path: str, values: list[object]) -> tuple[int, ...]:
    parsed = tuple(_nonnegative_int(f"{path}[{index}]", value) for index, value in enumerate(values))
    return parsed


def _validate_preconditions(
    cell: NativeCellView,
    sweep: PerturbationSweepConfig,
    *,
    tolerance: object,
) -> float:
    if not isinstance(cell, NativeCellView):
        raise Phase1GateError("cell", "must be NativeCellView")
    if not isinstance(sweep, PerturbationSweepConfig):
        raise Phase1GateError("sweep", "must be PerturbationSweepConfig")
    try:
        validate_perturbation_sweep_config(sweep)
    except PerturbationSweepConfigError as error:
        raise Phase1GateError(f"sweep.{error.path}", error.reason) from error
    if cell.status is not CellStatus.COMPLETE:
        raise Phase1GateError("status", "must be complete")
    if cell.perturbation_id not in _COMPLETE_IDS:
        raise Phase1GateError("perturbation_id", "must be a materialized core perturbation")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        raise Phase1GateError("tolerance", "must be a finite nonnegative real number")
    parsed_tolerance = float(tolerance)
    if not math.isfinite(parsed_tolerance) or parsed_tolerance < 0.0:
        raise Phase1GateError("tolerance", "must be a finite nonnegative real number")

    expected_alpha_hex = _records_alpha_hex(sweep)
    observed_alpha_hex = tuple(record.alpha_float64_le_hex for record in cell.records)
    if observed_alpha_hex != expected_alpha_hex:
        raise Phase1GateError("native_gate.alpha_order", "must follow the frozen alpha order")

    seen_output_ids: set[str] = set()
    for index, record in enumerate(cell.records):
        if record.output_spectrum_id in seen_output_ids:
            raise Phase1GateError("native_gate.output_spectrum_id", "must be unique within the cell")
        seen_output_ids.add(record.output_spectrum_id)
        expected_id = derive_perturbed_spectrum_id(
            cell.source_spectrum_id,
            cell.perturbation_id,
            record.alpha,
            cell.state_digest,
            sweep.sha256,
        )
        if record.output_spectrum_id != expected_id:
            raise Phase1GateError("native_gate.output_spectrum_id", "must equal the recomputed perturbed spectrum id")
        if np.shares_memory(record.axis_cm1, cell.source_axis_cm1) or np.shares_memory(record.intensity, cell.source_intensity):
            raise Phase1GateError("native_gate.alias", "must not share memory with source arrays")
        if np.shares_memory(record.axis_cm1, record.intensity):
            raise Phase1GateError("native_gate.alias", "record axis and intensity must not share memory")
        if not np.array_equal(record.axis_cm1, cell.source_axis_cm1) and cell.perturbation_id in {"p01", "p02", "p03", "p04", "p05", "p08", "p09", "p10"}:
            raise Phase1GateError(f"native_gate.{cell.perturbation_id}.axis", "axis must equal the source axis at every alpha")

    zero_record = cell.records[0]
    if not np.array_equal(zero_record.axis_cm1, cell.source_axis_cm1):
        raise Phase1GateError("native_gate.alpha_zero.axis", "must equal the source axis at alpha=0")
    if not np.array_equal(zero_record.intensity, cell.source_intensity):
        raise Phase1GateError("native_gate.alpha_zero.intensity", "must equal the source intensity at alpha=0")
    return parsed_tolerance


def validate_cell_native_gate(
    cell: NativeCellView,
    sweep: PerturbationSweepConfig,
    *,
    tolerance: float,
) -> CellGateResult:
    parsed_tolerance = _validate_preconditions(cell, sweep, tolerance=tolerance)
    perturbation_id = cell.perturbation_id

    if perturbation_id == "p01":
        diagnostic_area: list[float] = []
        observed_area: list[float] = []
        for record in cell.records:
            observed = _longdouble_trapezoid(cell.source_intensity - record.intensity, cell.source_axis_cm1)
            expected = _diagnostic_float(
                record,
                operator_path="native_gate.p01.removed_component_area",
                key="removed_component_area",
            )
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p01.removed_component_area", "diagnostic and observed areas must agree")
            diagnostic_area.append(expected)
            observed_area.append(observed)
        _monotone_zero_then_positive("native_gate.p01.removed_component_area", diagnostic_area, strict_positive=True)
        witness = {
            "removed_component_area": _json_ready_sequence(diagnostic_area),
            "observed_removed_component_area": _json_ready_sequence(observed_area),
        }
    elif perturbation_id == "p02":
        selected_count = _nonnegative_integer_sequence(
            "native_gate.p02.selected_count",
            [
                _diagnostic_count(
                    record,
                    operator_path="native_gate.p02.selected_count",
                    key="selected_or_inserted_peak_count",
                )
                for record in cell.records
            ],
        )
        if selected_count[0] != 0:
            raise Phase1GateError("native_gate.p02.selected_count", "must be zero at alpha=0")
        diagnostic_area = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p02.removed_component_area",
                key="removed_component_area",
                nonnegative=True,
            )
            for record in cell.records
        ]
        observed_area = []
        for record, expected in zip(cell.records, diagnostic_area, strict=True):
            observed = _longdouble_trapezoid(cell.source_intensity - record.intensity, cell.source_axis_cm1)
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p02.removed_component_area", "diagnostic and observed areas must agree")
            observed_area.append(observed)
        if tuple(selected_count[1:]) != tuple(sorted(selected_count[1:])):
            raise Phase1GateError("native_gate.p02.selected_count", "must be nondecreasing over positive alphas")
        if any(right < left for left, right in zip(diagnostic_area, diagnostic_area[1:])):
            raise Phase1GateError("native_gate.p02.removed_component_area", "must be nondecreasing over alpha")
        witness = {
            "selected_count": selected_count,
            "removed_component_area": _json_ready_sequence(diagnostic_area),
            "observed_removed_component_area": _json_ready_sequence(observed_area),
        }
    elif perturbation_id == "p03":
        sigma = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p03.sigma",
                key="broadened_sigma_cm1",
                nonnegative=True,
            )
            for record in cell.records
        ]
        inserted = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p03.inserted_component_area",
                key="inserted_component_area",
                nonnegative=True,
            )
            for record in cell.records
        ]
        source_component = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p03.inserted_component_area",
                key="source_component_area",
                nonnegative=True,
            )
            for record in cell.records
        ]
        removed = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p03.inserted_component_area",
                key="removed_component_area",
                nonnegative=True,
            )
            for record in cell.records
        ]
        observed_inserted = []
        for record, expected_inserted, expected_source, removed_area in zip(
            cell.records,
            inserted,
            source_component,
            removed,
            strict=True,
        ):
            observed = _longdouble_trapezoid(record.intensity - cell.source_intensity, cell.source_axis_cm1) + removed_area
            if not _relation_close(observed, expected_inserted, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p03.inserted_component_area", "diagnostic and observed inserted areas must agree")
            if not _relation_close(expected_inserted, expected_source, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p03.inserted_component_area", "inserted and source component areas must agree")
            observed_inserted.append(observed)
        _monotone_zero_then_positive("native_gate.p03.sigma", sigma, strict_positive=True)
        witness = {
            "broadened_sigma_cm1": _json_ready_sequence(sigma),
            "inserted_component_area": _json_ready_sequence(inserted),
            "observed_inserted_component_area": _json_ready_sequence(observed_inserted),
            "source_component_area": _json_ready_sequence(source_component),
        }
    elif perturbation_id == "p04":
        selected_count = _nonnegative_integer_sequence(
            "native_gate.p04.selected_count",
            [
                _diagnostic_count(
                    record,
                    operator_path="native_gate.p04.selected_count",
                    key="selected_or_inserted_peak_count",
                )
                for record in cell.records
            ],
        )
        if selected_count[0] != 0:
            raise Phase1GateError("native_gate.p04.selected_count", "must be zero at alpha=0")
        diagnostic_area = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p04.removed_component_area",
                key="removed_component_area",
                nonnegative=True,
            )
            for record in cell.records
        ]
        observed_area = []
        for record, expected in zip(cell.records, diagnostic_area, strict=True):
            observed = _longdouble_trapezoid(cell.source_intensity - record.intensity, cell.source_axis_cm1)
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p04.removed_component_area", "diagnostic and observed areas must agree")
            observed_area.append(observed)
        if tuple(selected_count[1:]) != tuple(sorted(selected_count[1:])):
            raise Phase1GateError("native_gate.p04.selected_count", "must be nondecreasing over positive alphas")
        if any(right < left for left, right in zip(diagnostic_area, diagnostic_area[1:])):
            raise Phase1GateError("native_gate.p04.removed_component_area", "must be nondecreasing over alpha")
        witness = {
            "selected_count": selected_count,
            "removed_component_area": _json_ready_sequence(diagnostic_area),
            "observed_removed_component_area": _json_ready_sequence(observed_area),
        }
    elif perturbation_id == "p05":
        selected_count = _nonnegative_integer_sequence(
            "native_gate.p05.selected_count",
            [
                _diagnostic_count(
                    record,
                    operator_path="native_gate.p05.selected_count",
                    key="selected_or_inserted_peak_count",
                )
                for record in cell.records
            ],
        )
        centers = []
        inserted_area = []
        observed_area = []
        max_count = 0
        previous: tuple[float, ...] = ()
        for index, record in enumerate(cell.records):
            raw_centers = _diagnostic_value(
                record,
                operator_path="native_gate.p05.inserted_centers",
                key="inserted_peak_centers_cm1",
            )
            if not isinstance(raw_centers, tuple):
                raise Phase1GateError("native_gate.p05.inserted_centers", "must be a tuple")
            parsed_centers = tuple(_finite_float(f"native_gate.p05.inserted_centers[{index}]", value) for value in raw_centers)
            if len(parsed_centers) != selected_count[index]:
                raise Phase1GateError("native_gate.p05.inserted_centers", "selected count must equal the inserted-center length")
            if parsed_centers != previous[: len(parsed_centers)] and parsed_centers != previous and previous != parsed_centers[: len(previous)]:
                raise Phase1GateError("native_gate.p05.inserted_centers", "must be nested prefixes over alpha")
            previous = parsed_centers
            centers.append(parsed_centers)
            max_count = max(max_count, len(parsed_centers))
            expected = _diagnostic_float(
                record,
                operator_path="native_gate.p05.inserted_component_area",
                key="inserted_component_area",
                nonnegative=True,
            )
            observed = _longdouble_trapezoid(record.intensity - cell.source_intensity, cell.source_axis_cm1)
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p05.inserted_component_area", "diagnostic and observed inserted areas must agree")
            inserted_area.append(expected)
            observed_area.append(observed)
        if selected_count[0] != 0:
            raise Phase1GateError("native_gate.p05.selected_count", "must be zero at alpha=0")
        if centers[0] != ():
            raise Phase1GateError("native_gate.p05.inserted_centers", "must be empty at alpha=0")
        if inserted_area[0] != 0.0:
            raise Phase1GateError("native_gate.p05.inserted_component_area", "must be zero at alpha=0")
        if tuple(selected_count[1:]) != tuple(sorted(selected_count[1:])):
            raise Phase1GateError("native_gate.p05.selected_count", "must be nondecreasing over positive alphas")
        if max_count <= 0:
            raise Phase1GateError("native_gate.p05.inserted_centers", "must be positive at max alpha for a complete cell")
        witness = {
            "selected_count": selected_count,
            "inserted_centers": tuple(centers),
            "inserted_component_area": _json_ready_sequence(inserted_area),
            "observed_inserted_component_area": _json_ready_sequence(observed_area),
        }
    elif perturbation_id == "p08":
        diagnostic_rms = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p08.residual_rms",
                key="realized_added_baseline_rms",
                nonnegative=True,
            )
            for record in cell.records
        ]
        observed_rms = [stable_population_rms(record.intensity - cell.source_intensity) for record in cell.records]
        for observed, expected in zip(observed_rms, diagnostic_rms, strict=True):
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p08.residual_rms", "diagnostic and observed RMS must agree")
        _monotone_zero_then_positive("native_gate.p08.residual_rms", diagnostic_rms, strict_positive=True)
        _monotone_zero_then_positive("native_gate.p08.residual_rms", observed_rms, strict_positive=True)
        witness = {
            "diagnostic_residual_rms": _json_ready_sequence(diagnostic_rms),
            "observed_residual_rms": _json_ready_sequence(observed_rms),
        }
    elif perturbation_id == "p09":
        sigma = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p09.residual_rms",
                key="sigma",
                nonnegative=True,
            )
            for record in cell.records
        ]
        noise_mean_square = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p09.noise_mean_square",
                key="noise_mean_square",
                nonnegative=True,
            )
            for record in cell.records
        ]
        if any(not _relation_close(value, noise_mean_square[0], tolerance=parsed_tolerance) for value in noise_mean_square[1:]):
            raise Phase1GateError("native_gate.p09.noise_mean_square", "must remain consistent across the cell")
        observed_rms = [stable_population_rms(record.intensity - cell.source_intensity) for record in cell.records]
        for observed, sigma_value, mean_square in zip(observed_rms, sigma, noise_mean_square, strict=True):
            expected = sigma_value * math.sqrt(mean_square)
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p09.residual_rms", "observed RMS must agree with sigma * sqrt(noise_mean_square)")
        _monotone_zero_then_positive("native_gate.p09.residual_rms", sigma, strict_positive=True)
        _monotone_zero_then_positive("native_gate.p09.residual_rms", observed_rms, strict_positive=True)
        witness = {
            "sigma": _json_ready_sequence(sigma),
            "noise_mean_square": _json_ready_sequence(noise_mean_square),
            "observed_residual_rms": _json_ready_sequence(observed_rms),
        }
    elif perturbation_id == "p10":
        sigma = [
            _diagnostic_float(
                record,
                operator_path="native_gate.p10.residual_rms",
                key="sigma",
                nonnegative=True,
            )
            for record in cell.records
        ]
        observed_rms = [stable_population_rms(record.intensity - cell.source_intensity) for record in cell.records]
        for observed, expected in zip(observed_rms, sigma, strict=True):
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p10.residual_rms", "diagnostic and observed RMS must agree")
        _monotone_zero_then_positive("native_gate.p10.residual_rms", sigma, strict_positive=True)
        _monotone_zero_then_positive("native_gate.p10.residual_rms", observed_rms, strict_positive=True)
        witness = {
            "sigma": _json_ready_sequence(sigma),
            "observed_residual_rms": _json_ready_sequence(observed_rms),
        }
    elif perturbation_id == "p11":
        diagnostic_offset = []
        observed_offset = []
        for record in cell.records:
            if not np.array_equal(record.intensity, cell.source_intensity):
                raise Phase1GateError("native_gate.p11.intensity", "must exactly equal the source intensity")
            _strictly_increasing("native_gate.p11.axis", record.axis_cm1)
            observed = float(np.max(np.abs(record.axis_cm1 - cell.source_axis_cm1)))
            expected = _diagnostic_float(
                record,
                operator_path="native_gate.p11.offset",
                key="realized_max_abs_offset_cm1",
                nonnegative=True,
            )
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p11.offset", "diagnostic and observed offsets must agree")
            diagnostic_offset.append(expected)
            observed_offset.append(observed)
        _monotone_zero_then_positive("native_gate.p11.offset", diagnostic_offset, strict_positive=True)
        _monotone_zero_then_positive("native_gate.p11.offset", observed_offset, strict_positive=True)
        witness = {
            "diagnostic_max_abs_offset_cm1": _json_ready_sequence(diagnostic_offset),
            "observed_max_abs_offset_cm1": _json_ready_sequence(observed_offset),
        }
    else:
        diagnostic_offset = []
        observed_offset = []
        for record in cell.records:
            if not np.array_equal(record.intensity, cell.source_intensity):
                raise Phase1GateError("native_gate.p12.intensity", "must exactly equal the source intensity")
            _strictly_increasing("native_gate.p12.axis", record.axis_cm1)
            observed = float(np.max(np.abs(record.axis_cm1 - cell.source_axis_cm1)))
            expected = _diagnostic_float(
                record,
                operator_path="native_gate.p12.offset",
                key="realized_max_abs_offset_cm1",
                nonnegative=True,
            )
            if not _relation_close(observed, expected, tolerance=parsed_tolerance):
                raise Phase1GateError("native_gate.p12.offset", "diagnostic and observed offsets must agree")
            diagnostic_offset.append(expected)
            observed_offset.append(observed)
        _monotone_zero_then_positive("native_gate.p12.offset", diagnostic_offset, strict_positive=True)
        _monotone_zero_then_positive("native_gate.p12.offset", observed_offset, strict_positive=True)
        witness = {
            "diagnostic_max_abs_offset_cm1": _json_ready_sequence(diagnostic_offset),
            "observed_max_abs_offset_cm1": _json_ready_sequence(observed_offset),
        }

    return CellGateResult(
        perturbation_id=perturbation_id,
        passed=True,
        witness=witness,
    )


def _required_count(total: int, threshold: float) -> int:
    return int(
        (Decimal(total) * Decimal(str(threshold))).to_integral_value(
            rounding=ROUND_CEILING
        )
    )


def evaluate_core_gate(
    cells: Iterable[NativeCellView],
    config: Phase1CoreConfig,
    *,
    selected_source_count: int,
    selected_class_count: int,
) -> CoreGateResult:
    if not isinstance(config, Phase1CoreConfig):
        raise Phase1GateError("config", "must be Phase1CoreConfig")
    source_count = _nonnegative_int("selected_source_count", selected_source_count)
    class_count = _nonnegative_int("selected_class_count", selected_class_count)
    if source_count <= 0:
        raise Phase1GateError("selected_source_count", "must be positive")
    if class_count <= 0:
        raise Phase1GateError("selected_class_count", "must be positive")

    failures: list[str] = []
    complete_sources: dict[str, set[str]] = {perturbation_id: set() for perturbation_id in _PERTURBATION_ORDER}
    complete_classes: dict[str, set[int]] = {perturbation_id: set() for perturbation_id in _PERTURBATION_ORDER}
    source_metadata: dict[str, tuple[str, str, int, str, str]] = {}
    source_to_ids: dict[str, set[str]] = {}
    record_id_to_source: dict[str, str] = {}
    seen_classes: set[int] = set()
    deferred_counts = {perturbation_id: 0 for perturbation_id in _DEFERRED_IDS}
    failed_cell_count = 0

    for index, cell in enumerate(cells):
        if not isinstance(cell, NativeCellView):
            failures.append(f"coverage.invalid_cell_type.{index}")
            continue
        source_id = cell.source_spectrum_id
        seen_classes.add(cell.class_label)
        source_to_ids.setdefault(source_id, set())
        if cell.status is CellStatus.FAILED:
            failed_cell_count += 1

        axis_digest = hashlib.sha256(
            np.asarray(cell.source_axis_cm1, dtype="<f8").tobytes()
        ).hexdigest()
        intensity_digest = hashlib.sha256(
            np.asarray(cell.source_intensity, dtype="<f8").tobytes()
        ).hexdigest()

        known = source_metadata.get(source_id)
        if known is None:
            source_metadata[source_id] = (
                cell.source_record_id,
                cell.sample_id,
                cell.class_label,
                axis_digest,
                intensity_digest,
            )
        else:
            if (
                known[0] != cell.source_record_id
                or known[1] != cell.sample_id
                or known[2] != cell.class_label
                or known[3] != axis_digest
                or known[4] != intensity_digest
            ):
                failures.append(f"coverage.inconsistent_source_metadata.{source_id}")

        existing_source = record_id_to_source.get(cell.source_record_id)
        if existing_source is None:
            record_id_to_source[cell.source_record_id] = source_id
        elif existing_source != source_id:
            failures.append(f"coverage.duplicate_source_record_id.{cell.source_record_id}")

        if cell.perturbation_id not in _PERTURBATION_ORDER:
            failures.append(f"coverage.unknown_perturbation_id.{cell.perturbation_id}")
            continue
        if cell.perturbation_id in source_to_ids[source_id]:
            failures.append(f"coverage.duplicate.{source_id}.{cell.perturbation_id}")
        else:
            source_to_ids[source_id].add(cell.perturbation_id)

        if cell.perturbation_id in _DEFERRED_IDS:
            if cell.status is not CellStatus.NOT_APPLICABLE:
                failures.append(f"deferred.status.{source_id}.{cell.perturbation_id}")
            elif cell.reason_code != config.deferred_reason_code:
                failures.append(f"deferred.reason.{source_id}.{cell.perturbation_id}")
            elif cell.records:
                failures.append(f"deferred.records.{source_id}.{cell.perturbation_id}")
            else:
                deferred_counts[cell.perturbation_id] += 1
            continue

        if cell.status is CellStatus.COMPLETE:
            complete_sources[cell.perturbation_id].add(source_id)
            complete_classes[cell.perturbation_id].add(cell.class_label)
        elif cell.status is CellStatus.NOT_APPLICABLE:
            if cell.perturbation_id in _PEAK_IDS:
                if cell.reason_code not in config.peak_not_applicable_reason_codes:
                    failures.append(f"coverage.invalid_peak_not_applicable_reason.{source_id}.{cell.perturbation_id}")
            else:
                failures.append(f"coverage.unexpected_not_applicable.{source_id}.{cell.perturbation_id}")
        else:
            failures.append(f"coverage.invalid_status.{source_id}.{cell.perturbation_id}")

    observed_source_ids = set(source_metadata)
    if len(observed_source_ids) != source_count:
        failures.append(f"coverage.source_count.expected_{source_count}.observed_{len(observed_source_ids)}")
    if len(seen_classes) != class_count:
        failures.append(f"coverage.class_count.expected_{class_count}.observed_{len(seen_classes)}")

    for source_id, ids in source_to_ids.items():
        for perturbation_id in _PERTURBATION_ORDER:
            if perturbation_id not in ids:
                failures.append(f"coverage.missing.{source_id}.{perturbation_id}")

    for perturbation_id in _DEFERRED_IDS:
        if deferred_counts[perturbation_id] != source_count:
            failures.append(f"deferred.count.{perturbation_id}.expected_{source_count}.observed_{deferred_counts[perturbation_id]}")

    if failed_cell_count > int(config.core_gate["allowed_failed_cell_count"]):
        failures.append(
            f"failed_cell_count.expected_le_{int(config.core_gate['allowed_failed_cell_count'])}.observed_{failed_cell_count}"
        )

    p01_p04_source_required = _required_count(source_count, float(config.core_gate["p01_p04_min_source_fraction"]))
    p01_p04_class_required = _required_count(class_count, float(config.core_gate["p01_p04_min_class_fraction"]))
    p05_source_required = _required_count(source_count, float(config.core_gate["p05_min_source_fraction"]))
    p05_class_required = _required_count(class_count, float(config.core_gate["p05_min_class_fraction"]))
    p08_p12_source_required = _required_count(source_count, float(config.core_gate["p08_p12_required_source_fraction"]))

    for perturbation_id in ("p01", "p02", "p03", "p04"):
        if len(complete_sources[perturbation_id]) < p01_p04_source_required:
            failures.append(f"threshold.{perturbation_id}.source")
        if len(complete_classes[perturbation_id]) < p01_p04_class_required:
            failures.append(f"threshold.{perturbation_id}.class")
    if len(complete_sources["p05"]) < p05_source_required:
        failures.append("threshold.p05.source")
    if len(complete_classes["p05"]) < p05_class_required:
        failures.append("threshold.p05.class")
    for perturbation_id in ("p08", "p09", "p10", "p11", "p12"):
        if len(complete_sources[perturbation_id]) < p08_p12_source_required:
            failures.append(f"threshold.{perturbation_id}.source")

    complete_cell_counts = MappingProxyType(
        {
            perturbation_id: len(complete_sources[perturbation_id])
            for perturbation_id in _PERTURBATION_ORDER
        }
    )
    complete_class_counts = MappingProxyType(
        {
            perturbation_id: len(complete_classes[perturbation_id])
            for perturbation_id in _PERTURBATION_ORDER
        }
    )
    deferred_cell_count = deferred_counts["p06"] + deferred_counts["p07"]
    unique_failures = tuple(sorted(set(failures)))
    if unique_failures:
        core_dataset_gate = "fail"
        full_phase1_gate = "fail"
    else:
        core_dataset_gate = "pass"
        full_phase1_gate = "deferred_missing_p06_p07"
    return CoreGateResult(
        core_dataset_gate=core_dataset_gate,
        full_phase1_gate=full_phase1_gate,
        selected_source_count=source_count,
        selected_class_count=class_count,
        complete_cell_counts=complete_cell_counts,
        complete_class_counts=complete_class_counts,
        deferred_cell_count=deferred_cell_count,
        failed_cell_count=failed_cell_count,
        failures=unique_failures,
    )
