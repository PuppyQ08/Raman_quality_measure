from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from rpe.io.schema import JsonValue
from rpe.perturb import PerturbationResult, PerturbationState
from rpe.runner.phase1_selection import Phase1Source


_EXPECTED_ALPHA_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8)
_PERTURBATION_ORDER = tuple(f"p{index:02d}" for index in range(1, 13))
_HEX_DIGITS = frozenset("0123456789abcdef")


class Phase1RunnerError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


class CellStatus(str, Enum):
    COMPLETE = "complete"
    NOT_APPLICABLE = "not_applicable"
    FAILED = "failed"


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise Phase1RunnerError(path, "must be a nonempty string")
    return value


def _positive_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase1RunnerError(path, "must be a positive integer")
    return value


def _nonnegative_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase1RunnerError(path, "must be a nonnegative integer")
    return value


def _freeze_json(path: str, value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Phase1RunnerError(path, "must be finite")
        return value
    if isinstance(value, tuple):
        return tuple(
            _freeze_json(f"{path}[{index}]", item)
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
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
    raise Phase1RunnerError(path, "must be canonical-JSON-compatible")


def _frozen_json_mapping(path: str, value: object) -> Mapping[str, JsonValue]:
    frozen = _freeze_json(path, value)
    if not isinstance(frozen, Mapping):
        raise Phase1RunnerError(path, "must be a mapping")
    return frozen


def _lower_hex(path: str, value: object, *, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX_DIGITS for character in value)
    ):
        raise Phase1RunnerError(
            path,
            f"must be a lowercase {length}-character hexadecimal string",
        )
    return value


def _exception_triplet(
    *,
    status: CellStatus,
    reason_code: str | None,
    exception_type: str | None,
    exception_path: str | None,
    exception_message: str | None,
) -> tuple[str | None, str | None, str | None]:
    if reason_code is not None:
        _nonempty_string("reason_code", reason_code)
    fields = (exception_type, exception_path, exception_message)
    any_present = any(field is not None for field in fields)
    all_present = all(field is not None for field in fields)
    if any_present and not all_present:
        raise Phase1RunnerError(
            "exception_fields",
            "must be all present or all absent",
        )
    if status is CellStatus.COMPLETE:
        if reason_code is not None:
            raise Phase1RunnerError(
                "reason_code",
                "must be absent when status is complete",
            )
        if any_present:
            raise Phase1RunnerError(
                "exception_fields",
                "must be absent when status is complete",
            )
        return (None, None, None)
    elif status is CellStatus.NOT_APPLICABLE:
        if reason_code is None:
            raise Phase1RunnerError(
                "reason_code",
                "must be present when status is not_applicable",
            )
        if not any_present:
            return (None, None, None)
        assert all_present
        assert exception_type is not None
        assert exception_path is not None
        assert exception_message is not None
        return (
            _nonempty_string("exception_type", exception_type),
            _nonempty_string("exception_path", exception_path),
            _nonempty_string("exception_message", exception_message),
        )
    elif status is CellStatus.FAILED:
        if reason_code is not None:
            raise Phase1RunnerError(
                "reason_code",
                "must be absent when status is failed",
            )
        if not all_present:
            raise Phase1RunnerError(
                "exception_fields",
                "must be present when status is failed",
            )
        assert exception_type is not None
        assert exception_path is not None
        assert exception_message is not None
        return (
            _nonempty_string("exception_type", exception_type),
            _nonempty_string("exception_path", exception_path),
            _nonempty_string("exception_message", exception_message),
        )
    return (None, None, None)


def _check_no_alias(path: str, source: np.ndarray, output: np.ndarray) -> None:
    if np.shares_memory(source, output):
        raise Phase1RunnerError(path, "must not share memory with source spectrum")


def _float64_le_hex(value: float) -> str:
    return struct.pack("<d", value).hex()


def _validate_sources_sequence(
    sources: tuple[Phase1Source, ...],
) -> None:
    if not isinstance(sources, tuple) or not sources:
        raise Phase1RunnerError("sources", "must be a nonempty tuple")
    previous_rank: int | None = None
    seen_record_ids: set[str] = set()
    for index, source in enumerate(sources):
        if not isinstance(source, Phase1Source):
            raise Phase1RunnerError(
                f"sources[{index}]",
                "must be Phase1Source",
            )
        rank = _nonnegative_int(
            f"sources[{index}].selection.selection_rank",
            source.selection.selection_rank,
        )
        if previous_rank is not None and rank <= previous_rank:
            raise Phase1RunnerError(
                f"sources[{index}].selection.selection_rank",
                "must be strictly increasing and unique",
            )
        previous_rank = rank
        record_id = _nonempty_string(
            f"sources[{index}].selection.record_id",
            source.selection.record_id,
        )
        if record_id in seen_record_ids:
            raise Phase1RunnerError(
                f"sources[{index}].selection.record_id",
                "must be globally unique",
            )
        seen_record_ids.add(record_id)


@dataclass(frozen=True)
class PerturbedRecord:
    source: Phase1Source
    result: PerturbationResult
    alpha_float64_le_hex: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, Phase1Source):
            raise Phase1RunnerError("source", "must be Phase1Source")
        if not isinstance(self.result, PerturbationResult):
            raise Phase1RunnerError("result", "must be PerturbationResult")
        expected_hex = struct.pack("<d", self.result.alpha).hex()
        if self.alpha_float64_le_hex != expected_hex:
            raise Phase1RunnerError(
                "alpha_float64_le_hex",
                "must equal little-endian float64 hex",
            )
        if self.result.source_spectrum_id != self.source.spectrum.spectrum_id:
            raise Phase1RunnerError(
                "result.source_spectrum_id",
                "must match source.spectrum.spectrum_id",
            )
        _check_no_alias(
            "result.output.axis_cm1",
            self.source.spectrum.axis_cm1,
            self.result.output.axis_cm1,
        )
        _check_no_alias(
            "result.output.intensity",
            self.source.spectrum.intensity,
            self.result.output.intensity,
        )

    @property
    def alpha(self) -> float:
        return self.result.alpha


@dataclass(frozen=True)
class CellEvidence:
    status: CellStatus
    reason_code: str | None
    exception_type: str | None
    exception_path: str | None
    exception_message: str | None
    native_gate: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if not isinstance(self.status, CellStatus):
            raise Phase1RunnerError("status", "must be CellStatus")
        exception_type, exception_path, exception_message = _exception_triplet(
            status=self.status,
            reason_code=self.reason_code,
            exception_type=self.exception_type,
            exception_path=self.exception_path,
            exception_message=self.exception_message,
        )
        object.__setattr__(self, "reason_code", self.reason_code)
        object.__setattr__(self, "exception_type", exception_type)
        object.__setattr__(self, "exception_path", exception_path)
        object.__setattr__(self, "exception_message", exception_message)
        object.__setattr__(
            self,
            "native_gate",
            _frozen_json_mapping("native_gate", self.native_gate),
        )


@dataclass(frozen=True)
class Phase1Cell:
    source: Phase1Source
    perturbation_id: str
    state: PerturbationState | None
    records: tuple[PerturbedRecord, ...]
    evidence: CellEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.source, Phase1Source):
            raise Phase1RunnerError("source", "must be Phase1Source")
        perturbation_id = _nonempty_string("perturbation_id", self.perturbation_id)
        object.__setattr__(self, "perturbation_id", perturbation_id)
        if self.state is not None:
            if not isinstance(self.state, PerturbationState):
                raise Phase1RunnerError(
                    "state",
                    "must implement PerturbationState",
                )
            if self.state.perturbation_id != perturbation_id:
                raise Phase1RunnerError(
                    "state.perturbation_id",
                    "must match cell perturbation_id",
                )
            if self.state.spectrum_id != self.source.spectrum.spectrum_id:
                raise Phase1RunnerError(
                    "state.spectrum_id",
                    "must match source.spectrum.spectrum_id",
                )
        if not isinstance(self.records, tuple):
            raise Phase1RunnerError("records", "must be a tuple")
        if not isinstance(self.evidence, CellEvidence):
            raise Phase1RunnerError("evidence", "must be CellEvidence")

        if self.evidence.status is CellStatus.COMPLETE:
            if self.state is None:
                raise Phase1RunnerError(
                    "state",
                    "must be present when status is complete",
                )
            if len(self.records) != len(_EXPECTED_ALPHA_GRID):
                raise Phase1RunnerError(
                    "records",
                    "must contain exactly nine outputs when status is complete",
                )
        elif self.records:
            raise Phase1RunnerError(
                "records",
                "must be empty unless status is complete",
            )

        seen_output_ids: set[str] = set()
        observed_alphas: list[float] = []
        for index, record in enumerate(self.records):
            if not isinstance(record, PerturbedRecord):
                raise Phase1RunnerError(
                    f"records[{index}]",
                    "must be PerturbedRecord",
                )
            if record.source is not self.source:
                raise Phase1RunnerError(
                    f"records[{index}].source",
                    "must be the same Phase1Source object as cell.source",
                )
            if record.result.perturbation_id != perturbation_id:
                raise Phase1RunnerError(
                    f"records[{index}].result.perturbation_id",
                    "must match cell perturbation_id",
                )
            if self.state is not None and record.result.state_digest != self.state.state_digest:
                raise Phase1RunnerError(
                    f"records[{index}].result.state_digest",
                    "must match cell.state.state_digest",
                )
            output_id = record.result.output.spectrum_id
            if output_id in seen_output_ids:
                raise Phase1RunnerError(
                    f"records[{index}].result.output.spectrum_id",
                    "must be unique within the cell",
                )
            seen_output_ids.add(output_id)
            observed_alphas.append(record.result.alpha)
        if self.evidence.status is CellStatus.COMPLETE and tuple(
            _float64_le_hex(alpha) for alpha in observed_alphas
        ) != tuple(_float64_le_hex(alpha) for alpha in _EXPECTED_ALPHA_GRID):
            raise Phase1RunnerError(
                "records",
                "must follow the frozen alpha order",
            )

    @property
    def status(self) -> CellStatus:
        return self.evidence.status

    @property
    def reason_code(self) -> str | None:
        return self.evidence.reason_code


@dataclass(frozen=True)
class Phase1ShardPayload:
    shard_index: int
    sources: tuple[Phase1Source, ...]
    cells: tuple[Phase1Cell, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "shard_index",
            _nonnegative_int("shard_index", self.shard_index),
        )
        if not isinstance(self.cells, tuple):
            raise Phase1RunnerError("cells", "must be a tuple")
        _validate_sources_sequence(self.sources)

        expected_cell_count = len(self.sources) * len(_PERTURBATION_ORDER)
        if len(self.cells) != expected_cell_count:
            raise Phase1RunnerError(
                "cells",
                "must contain exactly twelve cells per source",
            )
        position = 0
        for source_index, source in enumerate(self.sources):
            for expected_id in _PERTURBATION_ORDER:
                cell = self.cells[position]
                if not isinstance(cell, Phase1Cell):
                    raise Phase1RunnerError(
                        f"cells[{position}]",
                        "must be Phase1Cell",
                    )
                if cell.source is not source:
                    raise Phase1RunnerError(
                        f"cells[{position}].source",
                        "must match the shard source object at this position",
                    )
                if cell.perturbation_id != expected_id:
                    raise Phase1RunnerError(
                        f"cells[{position}].perturbation_id",
                        "must follow exact P01-P12 order within each source",
                    )
                position += 1
