from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

import numpy as np

from rpe.io.rruff_source import RruffValidationError

if TYPE_CHECKING:
    from rpe.io.rruff import RruffInspection


_PAIR_KEYS = {
    "pair_id",
    "archive",
    "measurement_key",
    "mineral_names",
    "rruff_ids",
    "source_multiplicity",
    "accepted_multiplicity",
    "raw_members",
    "processed_members",
    "pair_status",
    "axis_relation",
    "pointwise_comparable_without_interpolation",
    "raw_slice",
    "processed_slice",
}
_MULTIPLICITY_KEYS = {"raw", "processed"}
_MEMBER_KEYS = {
    "member_id",
    "record_id",
    "source_member",
    "source_member_sha256",
    "conversion_status",
    "rejection_code",
}
_PAIR_STATUSES = (
    "paired_unique",
    "raw_only",
    "processed_only",
    "ambiguous",
    "rejected_only",
)
_AXIS_RELATIONS = (
    "exact_equal",
    "processed_exact_contiguous_subset_of_raw",
    "raw_exact_contiguous_subset_of_processed",
    "overlap_requires_alignment",
    "no_axis_overlap",
)
_POINTWISE_RELATIONS = {
    "exact_equal",
    "processed_exact_contiguous_subset_of_raw",
    "raw_exact_contiguous_subset_of_processed",
}


@dataclass(frozen=True)
class RruffPairMember:
    member_id: str
    record_id: str | None
    source_member: str
    source_member_sha256: str
    conversion_status: str
    rejection_code: str | None


@dataclass(frozen=True)
class RruffPairInspection:
    pair_id: str
    archive: str
    measurement_key: str
    mineral_names: tuple[str, ...]
    rruff_ids: tuple[str, ...]
    raw_members: tuple[RruffPairMember, ...]
    processed_members: tuple[RruffPairMember, ...]
    pair_status: str
    axis_relation: str | None
    raw_slice: tuple[int, int] | None
    processed_slice: tuple[int, int] | None


def _pair_id(archive: str, measurement_key: str) -> str:
    return hashlib.sha256(
        b"rpe-rruff-pair-v1\0"
        + archive.encode("utf-8")
        + b"\0"
        + measurement_key.encode("utf-8")
    ).hexdigest()


def _record_id(kind: str, member_id: str) -> str:
    return f"{kind}-{member_id}"


def _pair_status(
    *,
    source_raw: int,
    source_processed: int,
    accepted_raw: int,
    accepted_processed: int,
) -> str:
    if accepted_raw == 0 and accepted_processed == 0:
        return "rejected_only"
    if (
        source_raw > 1
        or source_processed > 1
        or accepted_raw > 1
        or accepted_processed > 1
    ):
        return "ambiguous"
    if accepted_raw == 1 and accepted_processed == 1:
        return "paired_unique"
    if accepted_raw == 1 and accepted_processed == 0:
        return "raw_only"
    if accepted_raw == 0 and accepted_processed == 1:
        return "processed_only"
    raise ValueError(
        "unsupported accepted multiplicity "
        f"{accepted_raw} raw + {accepted_processed} processed"
    )


def _contiguous_subsequence_start(
    container: np.ndarray,
    candidate: np.ndarray,
) -> int | None:
    if candidate.size > container.size:
        return None
    candidate_length = candidate.size
    for start in range(container.size - candidate_length + 1):
        if np.array_equal(
            container[start : start + candidate_length],
            candidate,
        ):
            return start
    return None


def _axis_relation(
    raw_axis: np.ndarray,
    processed_axis: np.ndarray,
) -> tuple[
    str,
    tuple[int, int] | None,
    tuple[int, int] | None,
]:
    if np.array_equal(raw_axis, processed_axis):
        return (
            "exact_equal",
            (0, raw_axis.size),
            (0, processed_axis.size),
        )
    processed_start = _contiguous_subsequence_start(
        raw_axis,
        processed_axis,
    )
    if processed_start is not None:
        return (
            "processed_exact_contiguous_subset_of_raw",
            (
                processed_start,
                processed_start + processed_axis.size,
            ),
            (0, processed_axis.size),
        )
    raw_start = _contiguous_subsequence_start(
        processed_axis,
        raw_axis,
    )
    if raw_start is not None:
        return (
            "raw_exact_contiguous_subset_of_processed",
            (0, raw_axis.size),
            (raw_start, raw_start + raw_axis.size),
        )
    if (
        float(np.max(raw_axis)) < float(np.min(processed_axis))
        or float(np.max(processed_axis)) < float(np.min(raw_axis))
    ):
        return "no_axis_overlap", None, None
    return "overlap_requires_alignment", None, None


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


def _member_document(member: RruffPairMember) -> dict[str, object]:
    return {
        "member_id": member.member_id,
        "record_id": member.record_id,
        "source_member": member.source_member,
        "source_member_sha256": member.source_member_sha256,
        "conversion_status": member.conversion_status,
        "rejection_code": member.rejection_code,
    }


def _slice_document(
    value: tuple[int, int] | None,
) -> dict[str, int] | None:
    if value is None:
        return None
    return {"start": value[0], "stop": value[1]}


def _pair_document(pair: RruffPairInspection) -> dict[str, object]:
    raw_members = [_member_document(member) for member in pair.raw_members]
    processed_members = [
        _member_document(member) for member in pair.processed_members
    ]
    return {
        "pair_id": pair.pair_id,
        "archive": pair.archive,
        "measurement_key": pair.measurement_key,
        "mineral_names": list(pair.mineral_names),
        "rruff_ids": list(pair.rruff_ids),
        "source_multiplicity": {
            "raw": len(raw_members),
            "processed": len(processed_members),
        },
        "accepted_multiplicity": {
            "raw": sum(
                member["conversion_status"] == "accepted"
                for member in raw_members
            ),
            "processed": sum(
                member["conversion_status"] == "accepted"
                for member in processed_members
            ),
        },
        "raw_members": raw_members,
        "processed_members": processed_members,
        "pair_status": pair.pair_status,
        "axis_relation": pair.axis_relation,
        "pointwise_comparable_without_interpolation": (
            pair.axis_relation in _POINTWISE_RELATIONS
        ),
        "raw_slice": _slice_document(pair.raw_slice),
        "processed_slice": _slice_document(pair.processed_slice),
    }


def _write_rruff_pair_index(
    path: Path,
    inspection: "RruffInspection",
) -> None:
    path = Path(path)
    if not path.parent.is_dir():
        raise RruffValidationError(
            "pair_index.parent",
            "must be an existing directory",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    if path.exists():
        raise RruffValidationError(
            "pair_index",
            "already exists",
            "SOURCE_MEMBER_COUNT_MISMATCH",
        )
    try:
        with path.open("xb") as output:
            for pair in inspection.pairs:
                output.write(_canonical_json_bytes(_pair_document(pair)))
    except RruffValidationError:
        raise
    except (OSError, TypeError, ValueError) as error:
        cleanup_error = None
        if path.exists():
            try:
                path.unlink()
            except OSError as current_error:
                cleanup_error = current_error
        reason = str(error)
        if cleanup_error is not None:
            reason += f"; partial-file cleanup failed: {cleanup_error}"
        raise RruffValidationError(
            "pair_index",
            reason,
            "SOURCE_MEMBER_COUNT_MISMATCH",
        ) from error


def _validation_error(
    path: str,
    reason: str,
) -> RruffValidationError:
    return RruffValidationError(
        path,
        reason,
        "SOURCE_MEMBER_COUNT_MISMATCH",
    )


def _require_keys(
    path: str,
    value: object,
    expected: set[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise _validation_error(path, "must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise _validation_error(
            path,
            (
                f"key mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
        )
    return value


def _require_member_array(
    path: str,
    value: object,
) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        raise _validation_error(path, "must be a list")
    members = []
    observed_order = []
    for index, item in enumerate(value):
        member = _require_keys(
            f"{path}[{index}]",
            item,
            _MEMBER_KEYS,
        )
        status = member["conversion_status"]
        if status not in {"accepted", "rejected"}:
            raise _validation_error(
                path,
                f"invalid conversion status {status!r}",
            )
        if status == "accepted":
            if (
                not isinstance(member["record_id"], str)
                or member["record_id"] == ""
                or member["rejection_code"] is not None
            ):
                raise _validation_error(
                    path,
                    "accepted member has invalid null semantics",
                )
        elif (
            member["record_id"] is not None
            or not isinstance(member["rejection_code"], str)
            or member["rejection_code"] == ""
        ):
            raise _validation_error(
                path,
                "rejected member has invalid null semantics",
            )
        if (
            not isinstance(member["member_id"], str)
            or not isinstance(member["source_member"], str)
            or not isinstance(member["source_member_sha256"], str)
        ):
            raise _validation_error(path, "member identity fields are invalid")
        try:
            observed_order.append(
                (
                    member["member_id"].encode("ascii"),
                    member["source_member"].encode("utf-8"),
                )
            )
        except UnicodeEncodeError as error:
            raise _validation_error(
                path,
                f"member sort key is not encodable: {error}",
            ) from error
        members.append(member)
    if observed_order != sorted(observed_order):
        raise _validation_error(path, "members are not canonically sorted")
    return members


def _require_slice(path: str, value: object) -> None:
    if value is None:
        return
    parsed = _require_keys(path, value, {"start", "stop"})
    for name in ("start", "stop"):
        current = parsed[name]
        if (
            isinstance(current, bool)
            or not isinstance(current, int)
            or current < 0
        ):
            raise _validation_error(path, "slice indexes must be non-negative integers")
    if parsed["stop"] <= parsed["start"]:
        raise _validation_error(path, "slice stop must exceed start")


def _validate_row_structure(
    row_path: str,
    value: object,
) -> Mapping[str, object]:
    row = _require_keys(row_path, value, _PAIR_KEYS)
    for name in ("source_multiplicity", "accepted_multiplicity"):
        multiplicity = _require_keys(
            f"{row_path}.{name}",
            row[name],
            _MULTIPLICITY_KEYS,
        )
        for kind in ("raw", "processed"):
            current = multiplicity[kind]
            if (
                isinstance(current, bool)
                or not isinstance(current, int)
                or current < 0
            ):
                raise _validation_error(
                    f"{row_path}.{name}",
                    "multiplicity values must be non-negative integers",
                )
    _require_member_array(f"{row_path}.raw_members", row["raw_members"])
    _require_member_array(
        f"{row_path}.processed_members",
        row["processed_members"],
    )
    if row["pair_status"] not in _PAIR_STATUSES:
        raise _validation_error(
            f"{row_path}.pair_status",
            f"invalid status {row['pair_status']!r}",
        )
    if (
        row["axis_relation"] is not None
        and row["axis_relation"] not in _AXIS_RELATIONS
    ):
        raise _validation_error(
            f"{row_path}.axis_relation",
            f"invalid relation {row['axis_relation']!r}",
        )
    if type(row["pointwise_comparable_without_interpolation"]) is not bool:
        raise _validation_error(
            f"{row_path}.pointwise_comparable_without_interpolation",
            "must be Boolean",
        )
    _require_slice(f"{row_path}.raw_slice", row["raw_slice"])
    _require_slice(f"{row_path}.processed_slice", row["processed_slice"])
    if not isinstance(row["mineral_names"], list) or not all(
        isinstance(item, str) for item in row["mineral_names"]
    ):
        raise _validation_error(
            f"{row_path}.mineral_names",
            "must be a list of strings",
        )
    if not isinstance(row["rruff_ids"], list) or not all(
        isinstance(item, str) for item in row["rruff_ids"]
    ):
        raise _validation_error(
            f"{row_path}.rruff_ids",
            "must be a list of strings",
        )
    return row


def _read_pair_index(path: Path) -> list[Mapping[str, object]]:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise _validation_error("pair_index", str(error)) from error
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith(b"\n") for line in lines):
        raise _validation_error(
            "pair_index",
            "must contain complete newline-terminated JSON lines",
        )
    rows = []
    for index, line in enumerate(lines):
        row_path = f"pair_index[{index}]"
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _validation_error(row_path, str(error)) from error
        try:
            canonical = _canonical_json_bytes(value)
        except (TypeError, ValueError) as error:
            raise _validation_error(row_path, str(error)) from error
        if line != canonical:
            raise _validation_error(row_path, "line is not canonical JSON")
        rows.append(_validate_row_structure(row_path, value))
    return rows


def _field_path(index: int, field: str) -> str:
    return f"pair_index[{index}].{field}"


def _compare_row(
    index: int,
    observed: Mapping[str, object],
    expected: Mapping[str, object],
) -> None:
    for field in (
        "pair_id",
        "archive",
        "measurement_key",
        "mineral_names",
        "rruff_ids",
        "source_multiplicity",
        "accepted_multiplicity",
        "raw_members",
        "processed_members",
        "pair_status",
        "axis_relation",
        "pointwise_comparable_without_interpolation",
        "raw_slice",
        "processed_slice",
    ):
        if observed[field] != expected[field]:
            raise _validation_error(
                _field_path(index, field),
                (
                    f"expected {expected[field]!r}, "
                    f"observed {observed[field]!r}"
                ),
            )


def _record_id_sets(
    rows: list[Mapping[str, object]],
) -> tuple[set[str], set[str]]:
    raw_ids: set[str] = set()
    processed_ids: set[str] = set()
    for row in rows:
        for member in row["raw_members"]:
            if member["conversion_status"] == "accepted":
                raw_ids.add(str(member["record_id"]))
        for member in row["processed_members"]:
            if member["conversion_status"] == "accepted":
                processed_ids.add(str(member["record_id"]))
    return raw_ids, processed_ids


def _summary(rows: list[Mapping[str, object]]) -> Mapping[str, int]:
    counts = {
        "rows": len(rows),
        **{status: 0 for status in _PAIR_STATUSES},
        **{relation: 0 for relation in _AXIS_RELATIONS},
        "pointwise_comparable_pairs": 0,
    }
    for row in rows:
        counts[str(row["pair_status"])] += 1
        relation = row["axis_relation"]
        if relation is not None:
            counts[str(relation)] += 1
        if row["pointwise_comparable_without_interpolation"]:
            counts["pointwise_comparable_pairs"] += 1
    return MappingProxyType(counts)


def _validate_rruff_pair_index(
    path: Path,
    inspection: "RruffInspection",
    *,
    raw_record_ids: set[str] | None = None,
    processed_record_ids: set[str] | None = None,
) -> Mapping[str, int]:
    path = Path(path)
    if (raw_record_ids is None) != (processed_record_ids is None):
        raise _validation_error(
            "pair_index.record_ids",
            "raw and processed record-ID sets must be supplied together",
        )
    rows = _read_pair_index(path)
    observed_order = [
        (
            str(row["archive"]).encode("utf-8"),
            str(row["measurement_key"]).encode("utf-8"),
        )
        for row in rows
    ]
    if observed_order != sorted(observed_order) or len(observed_order) != len(
        set(observed_order)
    ):
        raise _validation_error(
            "pair_index.order",
            "rows must be strictly sorted and unique",
        )
    expected_rows = [_pair_document(pair) for pair in inspection.pairs]
    if len(rows) != len(expected_rows):
        raise _validation_error(
            "pair_index.order",
            f"expected {len(expected_rows)} rows, observed {len(rows)}",
        )
    for index, (observed, expected) in enumerate(
        zip(rows, expected_rows, strict=True)
    ):
        _compare_row(index, observed, expected)

    observed_raw_ids, observed_processed_ids = _record_id_sets(rows)
    if raw_record_ids is not None and observed_raw_ids != set(raw_record_ids):
        raise _validation_error(
            "pair_index.record_ids.raw",
            (
                f"expected {sorted(raw_record_ids)}, "
                f"observed {sorted(observed_raw_ids)}"
            ),
        )
    if (
        processed_record_ids is not None
        and observed_processed_ids != set(processed_record_ids)
    ):
        raise _validation_error(
            "pair_index.record_ids.processed",
            (
                f"expected {sorted(processed_record_ids)}, "
                f"observed {sorted(observed_processed_ids)}"
            ),
        )
    return _summary(rows)
