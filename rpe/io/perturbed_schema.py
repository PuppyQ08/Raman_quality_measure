from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Iterable, Mapping, Sequence

import numpy as np


class PerturbedStoreError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


STORE_SCHEMA_VERSION = "phase1-perturbed-store-v1"
SHARD_SCHEMA_VERSION = "phase1-perturbed-shard-v1"
SHARD_FILES = frozenset({"arrays.h5", "records.jsonl", "cells.jsonl", "receipt.json"})

_LOGICAL_ARRAY_DOMAIN = b"rpe-phase1-logical-arrays-v1\0"


def canonical_json_value(value: object) -> object:
    if isinstance(value, np.generic):
        raise PerturbedStoreError(
            "canonical JSON",
            f"unsupported type {type(value).__name__}",
        )
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PerturbedStoreError("canonical JSON", "float must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [canonical_json_value(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in sorted(value.items()):
            if not isinstance(key, str) or key == "":
                raise PerturbedStoreError(
                    "canonical JSON",
                    "keys must be nonempty strings",
                )
            normalized[key] = canonical_json_value(item)
        return normalized
    raise PerturbedStoreError(
        "canonical JSON",
        f"unsupported type {type(value).__name__}",
    )


def canonical_json_bytes(value: object) -> bytes:
    normalized = canonical_json_value(value)
    return (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    payload = bytearray()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PerturbedStoreError(
                f"canonical JSONL[{index}]",
                "row must be a mapping",
            )
        payload.extend(canonical_json_bytes(row))
    return bytes(payload)


def float64_le_bytes(values: np.ndarray) -> bytes:
    if not isinstance(values, np.ndarray):
        raise PerturbedStoreError("float64", "must be a numpy.ndarray")
    if values.dtype != np.dtype("<f8"):
        raise PerturbedStoreError("float64", "dtype must equal little-endian float64")
    if values.ndim != 1:
        raise PerturbedStoreError("float64", "array must be one-dimensional")
    if values.size == 0:
        raise PerturbedStoreError("float64", "array must be nonempty")
    if not np.isfinite(values).all():
        raise PerturbedStoreError("float64", "array must contain only finite values")
    return np.ascontiguousarray(values, dtype="<f8").tobytes()


def logical_array_digest(
    axes: Sequence[np.ndarray],
    sources: Sequence[np.ndarray],
    records: Sequence[np.ndarray],
) -> str:
    digest = hashlib.sha256()
    digest.update(_LOGICAL_ARRAY_DOMAIN)
    for collection in (axes, sources, records):
        digest.update(struct.pack("<Q", len(collection)))
        for values in collection:
            encoded = float64_le_bytes(values)
            digest.update(struct.pack("<Q", values.size))
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
    return digest.hexdigest()
