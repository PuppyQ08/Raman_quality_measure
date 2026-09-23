from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from rpe.io.store import UnifiedDataset
from rpe.semisynth.contracts import Phase2Config, Phase2ConfigError


class RruffPairRole(str, Enum):
    EXTRACTION_FIT = "extraction_fit"
    SIGNAL_TEMPLATE = "signal_template"
    REAL_HOLDOUT = "real_holdout"


class Phase2SplitError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class RruffPairAssignment:
    pair_id: str
    group_key: str
    role: RruffPairRole
    excitation_stratum: str | None
    raw_record_id: str
    processed_record_id: str
    axis_relation: str

    def __post_init__(self) -> None:
        for name in (
            "pair_id",
            "group_key",
            "raw_record_id",
            "processed_record_id",
            "axis_relation",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise Phase2SplitError(name, "must be a nonempty string")
        if not isinstance(self.role, RruffPairRole):
            raise Phase2SplitError("role", "must be RruffPairRole")
        if self.excitation_stratum is not None and self.excitation_stratum not in {
            "green_514",
            "green_532",
            "nir_780",
            "nir_785",
        }:
            raise Phase2SplitError("excitation_stratum", "is not supported")


@dataclass(frozen=True)
class Phase2SplitSummary:
    assignments: tuple[RruffPairAssignment, ...]
    group_count: int
    pair_counts: Mapping[str, int]
    group_counts: Mapping[str, int]
    stratum_pair_counts: Mapping[str, Mapping[str, int]]
    ledger_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.assignments, tuple) or not self.assignments:
            raise Phase2SplitError("assignments", "must be a nonempty tuple")
        if any(
            not isinstance(assignment, RruffPairAssignment)
            for assignment in self.assignments
        ):
            raise Phase2SplitError("assignments", "has an invalid item")
        object.__setattr__(
            self, "pair_counts", MappingProxyType(dict(sorted(self.pair_counts.items())))
        )
        object.__setattr__(
            self, "group_counts", MappingProxyType(dict(sorted(self.group_counts.items())))
        )
        object.__setattr__(
            self,
            "stratum_pair_counts",
            MappingProxyType(
                {
                    role: MappingProxyType(dict(sorted(counts.items())))
                    for role, counts in sorted(self.stratum_pair_counts.items())
                }
            ),
        )

    @property
    def assignment_count(self) -> int:
        return len(self.assignments)


def _length_prefixed(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def assign_rruff_group_role(
    group_key: str, *, config: Phase2Config
) -> RruffPairRole:
    if not isinstance(group_key, str) or not group_key:
        raise Phase2SplitError("group_key", "must be a nonempty string")
    if not isinstance(config, Phase2Config):
        raise Phase2SplitError("config", "must be Phase2Config")
    digest = hashlib.sha256(
        config.split_domain.encode("utf-8")
        + b"\0"
        + struct.pack("<Q", config.split_seed)
        + _length_prefixed(group_key)
    ).digest()
    unit = int.from_bytes(digest[:8], "little") / 2**64
    if unit < config.fit_threshold:
        return RruffPairRole.EXTRACTION_FIT
    if unit < config.holdout_threshold:
        return RruffPairRole.SIGNAL_TEMPLATE
    return RruffPairRole.REAL_HOLDOUT


def excitation_stratum(
    excitation_nm: float | None, *, config: Phase2Config
) -> str | None:
    if excitation_nm is None:
        return None
    if isinstance(excitation_nm, bool) or not isinstance(
        excitation_nm, (int, float)
    ):
        raise Phase2SplitError("excitation_nm", "must be finite or None")
    value = float(excitation_nm)
    if not math.isfinite(value):
        raise Phase2SplitError("excitation_nm", "must be finite or None")
    if not isinstance(config, Phase2Config):
        raise Phase2SplitError("config", "must be Phase2Config")
    for name, bounds in config.split_strata.items():
        lower = bounds["lower_inclusive"]
        if "upper_exclusive" in bounds:
            matches = lower <= value < bounds["upper_exclusive"]
        else:
            matches = lower <= value <= bounds["upper_inclusive"]
        if matches:
            return name
    return None


def _canonical_assignment(assignment: RruffPairAssignment) -> bytes:
    value = {
        "axis_relation": assignment.axis_relation,
        "excitation_stratum": assignment.excitation_stratum,
        "group_key": assignment.group_key,
        "pair_id": assignment.pair_id,
        "processed_record_id": assignment.processed_record_id,
        "raw_record_id": assignment.raw_record_id,
        "role": assignment.role.value,
    }
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


def _read_pair_rows(path: Path) -> list[Mapping[str, object]]:
    rows: list[Mapping[str, object]] = []
    try:
        with Path(path).open(encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise Phase2SplitError(
                        f"pair_index[{index}]", "must be an object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase2SplitError("pair_index", str(error)) from error
    return rows


def _verify_source(path: Path, *, byte_count: int, sha256: str) -> None:
    if path.stat().st_size != byte_count:
        raise Phase2SplitError(path.name, "byte count mismatch")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != sha256:
        raise Phase2SplitError(path.name, "SHA256 mismatch")


def split_rruff_pairs(
    pair_index_path: Path,
    raw_dataset_path: Path,
    *,
    config: Phase2Config,
    verify_checksums: bool,
) -> Phase2SplitSummary:
    if not isinstance(config, Phase2Config):
        raise Phase2SplitError("config", "must be Phase2Config")
    pair_index_path = Path(pair_index_path)
    raw_dataset_path = Path(raw_dataset_path)
    if verify_checksums:
        pair_identity = config.sources["rruff_pair_index"]
        _verify_source(
            pair_index_path,
            byte_count=pair_identity.byte_count,
            sha256=pair_identity.sha256,
        )
    rows = [
        row
        for row in _read_pair_rows(pair_index_path)
        if row.get("pair_status") == "paired_unique"
        and row.get("pointwise_comparable_without_interpolation") is True
    ]
    if len(rows) != 15764:
        raise Phase2SplitError("pair_index", "pointwise pair count mismatch")

    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for index, row in enumerate(rows):
        rruff_ids = row.get("rruff_ids")
        if not isinstance(rruff_ids, list) or not rruff_ids:
            raise Phase2SplitError(f"pair_index[{index}].rruff_ids", "is invalid")
        parsed = tuple(str(value) for value in rruff_ids)
        for value in parsed:
            find(value)
        for value in parsed[1:]:
            union(parsed[0], value)

    components: dict[str, set[str]] = defaultdict(set)
    for value in parent:
        components[find(value)].add(value)
    group_key_by_id = {
        value: "|".join(sorted(components[find(value)])) for value in parent
    }

    assignments: list[RruffPairAssignment] = []
    with UnifiedDataset.open(
        raw_dataset_path, verify_checksums=verify_checksums
    ) as dataset:
        for index, row in enumerate(rows):
            raw_members = row.get("raw_members")
            processed_members = row.get("processed_members")
            if (
                not isinstance(raw_members, list)
                or len(raw_members) != 1
                or not isinstance(processed_members, list)
                or len(processed_members) != 1
            ):
                raise Phase2SplitError(f"pair_index[{index}]", "is not one-to-one")
            raw_record_id = str(raw_members[0]["record_id"])
            processed_record_id = str(processed_members[0]["record_id"])
            record = dataset.get(raw_record_id)
            rruff_ids = tuple(str(value) for value in row["rruff_ids"])
            group_key = group_key_by_id[rruff_ids[0]]
            assignments.append(
                RruffPairAssignment(
                    pair_id=str(row["pair_id"]),
                    group_key=group_key,
                    role=assign_rruff_group_role(group_key, config=config),
                    excitation_stratum=excitation_stratum(
                        record.meta.excitation_nm, config=config
                    ),
                    raw_record_id=raw_record_id,
                    processed_record_id=processed_record_id,
                    axis_relation=str(row["axis_relation"]),
                )
            )
    assignments.sort(key=lambda value: value.pair_id.encode("utf-8"))
    frozen_assignments = tuple(assignments)
    payload = b"".join(_canonical_assignment(value) for value in frozen_assignments)
    ledger_sha256 = hashlib.sha256(
        config.ledger_domain.encode("utf-8") + b"\0" + payload
    ).hexdigest()
    pair_counts = Counter(value.role.value for value in frozen_assignments)
    groups_by_role: dict[str, set[str]] = defaultdict(set)
    strata: dict[str, Counter[str]] = {
        role.value: Counter(
            {
                "green_514": 0,
                "green_532": 0,
                "nir_780": 0,
                "nir_785": 0,
                "unstratified": 0,
            }
        )
        for role in RruffPairRole
    }
    for value in frozen_assignments:
        role = value.role.value
        groups_by_role[role].add(value.group_key)
        strata[role][value.excitation_stratum or "unstratified"] += 1
    group_counts = {role: len(values) for role, values in groups_by_role.items()}
    if dict(sorted(pair_counts.items())) != dict(config.expected_pair_counts):
        raise Phase2SplitError("pair counts", "do not match frozen config")
    if dict(sorted(group_counts.items())) != dict(config.expected_group_counts):
        raise Phase2SplitError("group counts", "do not match frozen config")
    observed_strata = {
        role: dict(sorted(counts.items())) for role, counts in strata.items()
    }
    if observed_strata != {
        role: dict(counts)
        for role, counts in config.expected_stratum_pair_counts.items()
    }:
        raise Phase2SplitError("stratum counts", "do not match frozen config")
    if ledger_sha256 != config.expected_ledger_sha256:
        raise Phase2SplitError("ledger SHA256", "does not match frozen config")
    all_groups = [groups_by_role[role.value] for role in RruffPairRole]
    if any(
        left & right
        for index, left in enumerate(all_groups)
        for right in all_groups[index + 1 :]
    ):
        raise Phase2SplitError("group leakage", "one group spans multiple roles")
    return Phase2SplitSummary(
        assignments=frozen_assignments,
        group_count=len(components),
        pair_counts=pair_counts,
        group_counts=group_counts,
        stratum_pair_counts=observed_strata,
        ledger_sha256=ledger_sha256,
    )


__all__ = [
    "Phase2SplitError",
    "Phase2SplitSummary",
    "RruffPairAssignment",
    "RruffPairRole",
    "assign_rruff_group_role",
    "excitation_stratum",
    "split_rruff_pairs",
]
