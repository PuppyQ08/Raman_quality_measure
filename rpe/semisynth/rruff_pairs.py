from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from rpe.io.store import UnifiedDataset
from rpe.semisynth.contracts import Phase2Config
from rpe.semisynth.splits import RruffPairAssignment, RruffPairRole


_AXIS_RELATIONS = frozenset(
    {
        "exact_equal",
        "processed_exact_contiguous_subset_of_raw",
        "raw_exact_contiguous_subset_of_processed",
    }
)
_HEX = frozenset("0123456789abcdef")
_PROCESSED_TEMPLATE_QUALIFIER = (
    "algorithmically_processed_signal_template_not_physical_clean_gt"
)


class SemiSyntheticPairError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty(path: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise SemiSyntheticPairError(path, "must be a nonempty string")
    return value


def _sha256(path: str, value: object) -> str:
    parsed = _nonempty(path, value)
    if len(parsed) != 64 or any(character not in _HEX for character in parsed):
        raise SemiSyntheticPairError(path, "must be lowercase SHA256")
    return parsed


def _slice(path: str, value: object, length: int) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise SemiSyntheticPairError(path, "must be an integer (start, stop) tuple")
    start, stop = value
    if start < 0 or stop <= start or stop - start != length:
        raise SemiSyntheticPairError(path, "must be a valid half-open slice")
    return start, stop


def _vector(path: str, value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise SemiSyntheticPairError(path, "must be a numpy.ndarray")
    if value.dtype != np.dtype("<f8"):
        raise SemiSyntheticPairError(f"{path}.dtype", "must be float64")
    if value.ndim != 1:
        raise SemiSyntheticPairError(f"{path}.dimension", "must be one-dimensional")
    if value.size < 32:
        raise SemiSyntheticPairError(f"{path}.size", "must contain at least 32 points")
    if not np.isfinite(value).all():
        raise SemiSyntheticPairError(f"{path}.finite", "contains non-finite values")
    copied = np.ascontiguousarray(value).copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True)
class RruffEqualAxisPair:
    pair_id: str
    group_key: str
    role: RruffPairRole
    excitation_stratum: str | None
    axis_relation: str
    raw_record_id: str
    processed_record_id: str
    raw_slice: tuple[int, int]
    processed_slice: tuple[int, int]
    raw_source_member_sha256: str
    processed_source_member_sha256: str
    processed_template_qualifier: str
    axis_cm1: np.ndarray
    raw_intensity: np.ndarray
    processed_intensity: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "group_key",
            "raw_record_id",
            "processed_record_id",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if not isinstance(self.role, RruffPairRole):
            raise SemiSyntheticPairError("role", "must be RruffPairRole")
        if self.excitation_stratum is not None and self.excitation_stratum not in {
            "green_514",
            "green_532",
            "nir_780",
            "nir_785",
        }:
            raise SemiSyntheticPairError(
                "excitation_stratum", "is not a frozen stratum"
            )
        if self.axis_relation not in _AXIS_RELATIONS:
            raise SemiSyntheticPairError("axis_relation", "is not pointwise exact")
        if self.processed_template_qualifier != _PROCESSED_TEMPLATE_QUALIFIER:
            raise SemiSyntheticPairError(
                "processed_template_qualifier",
                "must state that processed RRUFF is not physical clean GT",
            )
        object.__setattr__(
            self,
            "raw_source_member_sha256",
            _sha256("raw_source_member_sha256", self.raw_source_member_sha256),
        )
        object.__setattr__(
            self,
            "processed_source_member_sha256",
            _sha256(
                "processed_source_member_sha256",
                self.processed_source_member_sha256,
            ),
        )
        arrays = {
            name: _vector(name, getattr(self, name))
            for name in ("axis_cm1", "raw_intensity", "processed_intensity")
        }
        if len({array.size for array in arrays.values()}) != 1:
            raise SemiSyntheticPairError("arrays.shape", "must be equal length")
        if not np.all(np.diff(arrays["axis_cm1"]) > 0.0):
            raise SemiSyntheticPairError("axis_cm1.increasing", "must be strict")
        object.__setattr__(
            self, "raw_slice", _slice("raw_slice", self.raw_slice, arrays["axis_cm1"].size)
        )
        object.__setattr__(
            self,
            "processed_slice",
            _slice(
                "processed_slice",
                self.processed_slice,
                arrays["axis_cm1"].size,
            ),
        )
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


def _parse_row_slice(path: str, value: object) -> tuple[int, int]:
    if not isinstance(value, Mapping) or set(value) != {"start", "stop"}:
        raise SemiSyntheticPairError(path, "must contain start and stop")
    start = value["start"]
    stop = value["stop"]
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start
    ):
        raise SemiSyntheticPairError(path, "is not a valid half-open slice")
    return start, stop


def _one_member(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise SemiSyntheticPairError(path, "must contain exactly one member")
    return value[0]


def _verify_pair_index(path: Path, config: Phase2Config) -> None:
    identity = config.sources["rruff_pair_index"]
    try:
        if path.stat().st_size != identity.byte_count:
            raise SemiSyntheticPairError("pair_index", "byte count mismatch")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise SemiSyntheticPairError("pair_index", str(error)) from error
    if digest.hexdigest() != identity.sha256:
        raise SemiSyntheticPairError("pair_index", "SHA256 mismatch")


def load_rruff_equal_axis_pairs(
    pair_index_path: Path,
    raw_dataset_path: Path,
    processed_dataset_path: Path,
    assignments: Iterable[RruffPairAssignment],
    *,
    config: Phase2Config,
    verify_checksums: bool,
) -> tuple[RruffEqualAxisPair, ...]:
    if not isinstance(config, Phase2Config):
        raise SemiSyntheticPairError("config", "must be Phase2Config")
    materialized = tuple(assignments)
    if not materialized or any(
        not isinstance(assignment, RruffPairAssignment) for assignment in materialized
    ):
        raise SemiSyntheticPairError("assignments", "must contain assignments")
    pair_ids = tuple(assignment.pair_id for assignment in materialized)
    if len(pair_ids) != len(set(pair_ids)):
        raise SemiSyntheticPairError("assignments.pair_id", "must be unique")
    pair_index_path = Path(pair_index_path)
    if verify_checksums:
        _verify_pair_index(pair_index_path, config)
    requested = set(pair_ids)
    rows: dict[str, Mapping[str, object]] = {}
    try:
        with pair_index_path.open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                row = json.loads(line)
                if not isinstance(row, Mapping):
                    raise SemiSyntheticPairError(
                        f"pair_index[{index}]", "must be an object"
                    )
                pair_id = row.get("pair_id")
                if pair_id in requested:
                    if pair_id in rows:
                        raise SemiSyntheticPairError(
                            "pair_index.pair_id", "is duplicated"
                        )
                    rows[str(pair_id)] = row
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemiSyntheticPairError("pair_index", str(error)) from error
    missing = requested - set(rows)
    if missing:
        raise SemiSyntheticPairError("pair_index", f"missing {sorted(missing)[0]}")

    loaded: list[RruffEqualAxisPair] = []
    with UnifiedDataset.open(
        Path(raw_dataset_path), verify_checksums=verify_checksums
    ) as raw_dataset, UnifiedDataset.open(
        Path(processed_dataset_path), verify_checksums=verify_checksums
    ) as processed_dataset:
        for assignment in materialized:
            row = rows[assignment.pair_id]
            if (
                row.get("pair_status") != "paired_unique"
                or row.get("pointwise_comparable_without_interpolation") is not True
                or row.get("axis_relation") != assignment.axis_relation
            ):
                raise SemiSyntheticPairError(assignment.pair_id, "is not the assigned exact pair")
            raw_member = _one_member("raw_members", row.get("raw_members"))
            processed_member = _one_member(
                "processed_members", row.get("processed_members")
            )
            if (
                raw_member.get("record_id") != assignment.raw_record_id
                or processed_member.get("record_id") != assignment.processed_record_id
            ):
                raise SemiSyntheticPairError(assignment.pair_id, "record IDs drifted")
            raw_slice = _parse_row_slice("raw_slice", row.get("raw_slice"))
            processed_slice = _parse_row_slice(
                "processed_slice", row.get("processed_slice")
            )
            raw_record = raw_dataset.get(assignment.raw_record_id)
            processed_record = processed_dataset.get(assignment.processed_record_id)
            raw_selector = slice(*raw_slice)
            processed_selector = slice(*processed_slice)
            raw_axis = np.asarray(raw_record.wavenumber[raw_selector], dtype="<f8")
            processed_axis = np.asarray(
                processed_record.wavenumber[processed_selector], dtype="<f8"
            )
            if not np.array_equal(raw_axis, processed_axis):
                raise SemiSyntheticPairError(assignment.pair_id, "exact axes drifted")
            loaded.append(
                RruffEqualAxisPair(
                    pair_id=assignment.pair_id,
                    group_key=assignment.group_key,
                    role=assignment.role,
                    excitation_stratum=assignment.excitation_stratum,
                    axis_relation=assignment.axis_relation,
                    raw_record_id=assignment.raw_record_id,
                    processed_record_id=assignment.processed_record_id,
                    raw_slice=raw_slice,
                    processed_slice=processed_slice,
                    raw_source_member_sha256=str(
                        raw_member.get("source_member_sha256")
                    ),
                    processed_source_member_sha256=str(
                        processed_member.get("source_member_sha256")
                    ),
                    processed_template_qualifier=config.template_qualifiers[
                        "rruff_processed"
                    ],
                    axis_cm1=raw_axis,
                    raw_intensity=np.asarray(
                        raw_record.intensity[raw_selector], dtype="<f8"
                    ),
                    processed_intensity=np.asarray(
                        processed_record.intensity[processed_selector], dtype="<f8"
                    ),
                )
            )
    return tuple(loaded)


__all__ = [
    "RruffEqualAxisPair",
    "SemiSyntheticPairError",
    "load_rruff_equal_axis_pairs",
]
