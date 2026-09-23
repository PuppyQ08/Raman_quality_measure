from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import numpy as np

from rpe.io.perturbed_schema import canonical_json_bytes
from rpe.io.perturbed_store import (
    ShardReceipt,
    StoredCell,
    StoredShard,
    StoredSource,
    read_perturbed_shard,
    write_perturbed_shard,
)
from rpe.io.schema import JsonValue
from rpe.io.store import UnifiedDataset
from rpe.perturb import PerturbationSweepConfig, load_perturbation_sweep_config
from rpe.runner.phase1_config import (
    PHASE1_DATA_CODE_PATHS,
    REPO_ROOT,
    Phase1CoreConfig,
    code_snapshot_digest,
    code_snapshot_document,
    load_phase1_core_config,
    source_snapshot_digest,
)
from rpe.runner.phase1_gates import (
    CoreGateResult,
    NativeCellView,
    NativeRecordView,
    evaluate_core_gate,
    validate_cell_native_gate,
)
from rpe.runner.phase1_perturbations import (
    estimate_p10_peak_bytes,
    run_shard_payload,
)
from rpe.runner.phase1_selection import (
    Phase1Source,
    load_phase1_source,
    load_source_inventory,
    select_source_rows,
    source_subset_jsonl_bytes,
)
from rpe.runner.phase1_types import CellStatus


FORMAL_RUN_SCHEMA_VERSION = "phase1-perturbed-run-v1"
FORMAL_GATE_SCHEMA_VERSION = "phase1-perturbed-gate-v1"
FORMAL_MARKER_SCHEMA_VERSION = "phase1-perturbed-marker-v1"
_RUN_ID_DOMAIN = b"rpe-phase1-perturbation-run-id-v1\0"
_RUN_ID_PREFIX = "phase1-rruff-core10k-"
_FORMAL_CONFIG_PATH = Path("experiments/phase1/configs/rruff_raw_core10k_v1.json")
_DEFAULT_OUTPUT_ROOT = Path("data/perturbed/phase1_rruff_core10k")
_PERTURBATION_ORDER = tuple(f"p{index:02d}" for index in range(1, 13))
_STATUS_ORDER = ("complete", "failed", "not_applicable")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Phase1FormalError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _positive_int(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase1FormalError(path, "must be a positive integer")
    return value


def _nonempty_string(path: str, value: object) -> str:
    if not isinstance(value, str) or not value:
        raise Phase1FormalError(path, "must be a nonempty string")
    return value


def _lower_hex(path: str, value: object) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise Phase1FormalError(
            path, "must be a lowercase 64-character hexadecimal string"
        )
    return value


def _safe_run_id(value: object) -> str:
    run_id = _nonempty_string("run_id", value)
    if _SAFE_COMPONENT.fullmatch(run_id) is None or run_id in {".", ".."}:
        raise Phase1FormalError("run_id", "must be one safe path component")
    return run_id


def _length_prefixed_utf8(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def derive_phase1_formal_run_id(
    *,
    schema_version: str,
    scientific_config_sha256: str,
    sweep_config_sha256: str,
    source_snapshot_sha256: str,
    selected_source_subset_sha256: str,
    data_code_snapshot_sha256: str,
) -> str:
    values = (
        _nonempty_string("schema_version", schema_version),
        _lower_hex("scientific_config_sha256", scientific_config_sha256),
        _lower_hex("sweep_config_sha256", sweep_config_sha256),
        _lower_hex("source_snapshot_sha256", source_snapshot_sha256),
        _lower_hex(
            "selected_source_subset_sha256", selected_source_subset_sha256
        ),
        _lower_hex("data_code_snapshot_sha256", data_code_snapshot_sha256),
    )
    digest = hashlib.sha256()
    digest.update(_RUN_ID_DOMAIN)
    for value in values:
        digest.update(_length_prefixed_utf8(value))
    return _RUN_ID_PREFIX + digest.hexdigest()


def _freeze_string_tuple(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise Phase1FormalError(path, "must be a tuple")
    parsed = tuple(
        _nonempty_string(f"{path}[{index}]", item)
        for index, item in enumerate(value)
    )
    if parsed != tuple(sorted(set(parsed))):
        raise Phase1FormalError(path, "must be sorted and unique")
    return parsed


@dataclass(frozen=True)
class Phase1FormalSummary:
    path: Path
    run_id: str
    source_count: int
    class_count: int
    shard_count: int
    cell_count: int
    record_count: int
    core_dataset_gate: str
    full_phase1_gate: str
    checked_files: tuple[str, ...] = field(default=())

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise Phase1FormalError("path", "must be a pathlib.Path")
        object.__setattr__(self, "run_id", _safe_run_id(self.run_id))
        for name in (
            "source_count",
            "class_count",
            "shard_count",
            "cell_count",
            "record_count",
        ):
            object.__setattr__(self, name, _positive_int(name, getattr(self, name)))
        if self.core_dataset_gate not in {"pass", "fail"}:
            raise Phase1FormalError("core_dataset_gate", "must be pass or fail")
        if self.full_phase1_gate not in {"deferred_missing_p06_p07", "fail"}:
            raise Phase1FormalError(
                "full_phase1_gate", "must be deferred_missing_p06_p07 or fail"
            )
        object.__setattr__(
            self,
            "checked_files",
            _freeze_string_tuple("checked_files", self.checked_files),
        )


@dataclass(frozen=True)
class _IdentityContext:
    subset_bytes: bytes
    subset_sha256: str
    source_snapshot_sha256: str
    code_snapshot: Mapping[str, Mapping[str, int | str]]
    code_snapshot_sha256: str
    run_id: str


@dataclass
class _ReadbackStats:
    operator_status_counts: dict[str, dict[str, int]] = field(
        default_factory=lambda: {
            perturbation_id: {status: 0 for status in _STATUS_ORDER}
            for perturbation_id in _PERTURBATION_ORDER
        }
    )
    reason_counts: dict[str, int] = field(default_factory=dict)
    alpha_record_counts: dict[str, int] = field(default_factory=dict)
    cell_count: int = 0
    record_count: int = 0


@dataclass(frozen=True)
class _ReadbackResult:
    gate: CoreGateResult
    stats: _ReadbackStats
    receipts: tuple[ShardReceipt, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_context(
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    sources: tuple[Phase1Source, ...],
) -> _IdentityContext:
    subset_bytes = source_subset_jsonl_bytes(sources)
    subset_sha256 = hashlib.sha256(subset_bytes).hexdigest()
    source_sha256 = source_snapshot_digest(config)
    code_snapshot = code_snapshot_document(REPO_ROOT, PHASE1_DATA_CODE_PATHS)
    code_sha256 = code_snapshot_digest(REPO_ROOT, PHASE1_DATA_CODE_PATHS)
    run_id = derive_phase1_formal_run_id(
        schema_version=FORMAL_RUN_SCHEMA_VERSION,
        scientific_config_sha256=config.scientific_config_sha256,
        sweep_config_sha256=sweep.sha256,
        source_snapshot_sha256=source_sha256,
        selected_source_subset_sha256=subset_sha256,
        data_code_snapshot_sha256=code_sha256,
    )
    return _IdentityContext(
        subset_bytes=subset_bytes,
        subset_sha256=subset_sha256,
        source_snapshot_sha256=source_sha256,
        code_snapshot=code_snapshot,
        code_snapshot_sha256=code_sha256,
        run_id=run_id,
    )


def _validate_inputs(
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    sources: Sequence[Phase1Source],
) -> tuple[Phase1Source, ...]:
    if not isinstance(config, Phase1CoreConfig):
        raise Phase1FormalError("config", "must be Phase1CoreConfig")
    if not isinstance(sweep, PerturbationSweepConfig):
        raise Phase1FormalError("sweep", "must be PerturbationSweepConfig")
    if sweep.sha256 != config.shared_sweep_identity.sha256:
        raise Phase1FormalError("sweep.sha256", "must match config")
    if sweep.byte_count != config.shared_sweep_identity.byte_count:
        raise Phase1FormalError("sweep.byte_count", "must match config")
    if not isinstance(sources, Sequence) or not sources:
        raise Phase1FormalError("sources", "must be a nonempty sequence")
    parsed = tuple(sources)
    if any(not isinstance(source, Phase1Source) for source in parsed):
        raise Phase1FormalError("sources", "must contain only Phase1Source")
    if len(parsed) != config.subset_size:
        raise Phase1FormalError("sources", "count must equal config.subset_size")
    ranks = tuple(source.selection.selection_rank for source in parsed)
    if ranks != tuple(range(len(parsed))):
        raise Phase1FormalError(
            "sources.selection_rank", "must be consecutive from zero"
        )
    record_ids = tuple(source.selection.record_id for source in parsed)
    if record_ids != tuple(sorted(record_ids)) or len(set(record_ids)) != len(record_ids):
        raise Phase1FormalError("sources.record_id", "must be sorted and unique")
    return parsed


def _stored_native_view(
    source: StoredSource,
    cell: StoredCell,
) -> NativeCellView:
    return NativeCellView(
        source_spectrum_id=source.source_spectrum_id,
        source_record_id=source.source_record_id,
        sample_id=source.sample_id,
        class_label=source.class_label,
        source_axis_cm1=source.axis_cm1,
        source_intensity=source.intensity,
        perturbation_id=cell.perturbation_id,
        state_digest=cell.state_digest,
        status=CellStatus(cell.status),
        reason_code=cell.reason_code,
        records=tuple(
            NativeRecordView(
                output_spectrum_id=record.output_spectrum_id,
                alpha=record.alpha,
                alpha_float64_le_hex=record.alpha_float64_le_hex,
                axis_cm1=record.axis_cm1,
                intensity=record.intensity,
                diagnostics=record.diagnostics,
            )
            for record in cell.records
        ),
    )


def _source_matches(
    stored: StoredSource,
    expected: Phase1Source,
    *,
    path: str,
) -> None:
    pairs = (
        (stored.source_spectrum_id, expected.spectrum.spectrum_id, "spectrum_id"),
        (stored.source_record_id, expected.selection.record_id, "record_id"),
        (stored.sample_id, expected.selection.sample_id, "sample_id"),
        (stored.class_label, expected.selection.class_label, "class_label"),
        (stored.mineral_name, expected.selection.mineral_name, "mineral_name"),
        (dict(stored.provenance), dict(expected.provenance), "provenance"),
    )
    for observed, wanted, name in pairs:
        if observed != wanted:
            raise Phase1FormalError(f"{path}.{name}", "does not match selection")
    if not np.array_equal(stored.axis_cm1, expected.spectrum.axis_cm1):
        raise Phase1FormalError(f"{path}.axis_cm1", "does not match source")
    if not np.array_equal(stored.intensity, expected.spectrum.intensity):
        raise Phase1FormalError(f"{path}.intensity", "does not match source")


def _readback(
    run_path: Path,
    *,
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    sources: tuple[Phase1Source, ...],
) -> _ReadbackResult:
    shard_count = math.ceil(len(sources) / config.shard_source_count)
    receipts: list[ShardReceipt] = []
    stats = _ReadbackStats(
        alpha_record_counts={
            struct.pack("<d", alpha).hex(): 0 for alpha in sweep.alpha_grid
        }
    )

    def views() -> Iterator[NativeCellView]:
        for shard_index in range(shard_count):
            shard_path = run_path / "shards" / f"{shard_index:05d}"
            try:
                stored = read_perturbed_shard(shard_path)
            except Exception as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit, MemoryError)):
                    raise
                raise Phase1FormalError(
                    f"shards/{shard_index:05d}", str(error)
                ) from error
            if stored.run_id != run_path.name:
                raise Phase1FormalError(
                    f"shards/{shard_index:05d}.run_id", "does not match run"
                )
            if stored.shard_index != shard_index:
                raise Phase1FormalError(
                    f"shards/{shard_index:05d}.shard_index", "does not match path"
                )
            start = shard_index * config.shard_source_count
            expected_sources = sources[start : start + config.shard_source_count]
            if len(stored.sources) != len(expected_sources):
                raise Phase1FormalError(
                    f"shards/{shard_index:05d}.source_count", "partition mismatch"
                )
            for source_index, (observed, expected) in enumerate(
                zip(stored.sources, expected_sources, strict=True)
            ):
                _source_matches(
                    observed,
                    expected,
                    path=f"shards/{shard_index:05d}.sources[{source_index}]",
                )
            try:
                first_cell = json.loads(
                    (shard_path / "cells.jsonl").read_bytes().splitlines()[0]
                )
            except (OSError, IndexError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise Phase1FormalError(
                    f"shards/{shard_index:05d}.cells.jsonl", str(error)
                ) from error
            if first_cell.get("scientific_config_sha256") != config.scientific_config_sha256:
                raise Phase1FormalError(
                    f"shards/{shard_index:05d}.scientific_config_sha256",
                    "does not match config",
                )
            if first_cell.get("sweep_config_sha256") != sweep.sha256:
                raise Phase1FormalError(
                    f"shards/{shard_index:05d}.sweep_config_sha256",
                    "does not match sweep",
                )
            receipts.append(stored.receipt)
            stats.cell_count += stored.cell_count
            stats.record_count += stored.record_count
            for source_index, source in enumerate(stored.sources):
                cells = stored.cells[source_index * 12 : (source_index + 1) * 12]
                for cell in cells:
                    stats.operator_status_counts[cell.perturbation_id][cell.status] += 1
                    if cell.reason_code is not None:
                        stats.reason_counts[cell.reason_code] = (
                            stats.reason_counts.get(cell.reason_code, 0) + 1
                        )
                    for record in cell.records:
                        stats.alpha_record_counts[record.alpha_float64_le_hex] += 1
                    view = _stored_native_view(source, cell)
                    if view.status is CellStatus.COMPLETE:
                        result = validate_cell_native_gate(
                            view,
                            sweep,
                            tolerance=float(config.core_gate["float_relative_tolerance"]),
                        )
                        if dict(result.witness) != dict(cell.native_gate):
                            raise Phase1FormalError(
                                f"shards/{shard_index:05d}.native_gate",
                                "stored witness does not match recomputation",
                            )
                    yield view

    gate = evaluate_core_gate(
        views(),
        config,
        selected_source_count=len(sources),
        selected_class_count=len(
            {source.selection.class_label for source in sources}
        ),
    )
    if len(receipts) != shard_count:
        raise Phase1FormalError("shards", "readback did not cover every shard")
    return _ReadbackResult(gate=gate, stats=stats, receipts=tuple(receipts))


def _gate_document(
    *,
    run_id: str,
    result: _ReadbackResult,
) -> Mapping[str, object]:
    gate = result.gate
    return {
        "alpha_record_counts": dict(result.stats.alpha_record_counts),
        "complete_cell_counts": dict(gate.complete_cell_counts),
        "complete_class_counts": dict(gate.complete_class_counts),
        "core_dataset_gate": gate.core_dataset_gate,
        "deferred_cell_count": gate.deferred_cell_count,
        "failed_cell_count": gate.failed_cell_count,
        "failures": list(gate.failures),
        "full_phase1_gate": gate.full_phase1_gate,
        "operator_status_counts": result.stats.operator_status_counts,
        "reason_counts": dict(sorted(result.stats.reason_counts.items())),
        "run_id": run_id,
        "schema_version": FORMAL_GATE_SCHEMA_VERSION,
        "selected_class_count": gate.selected_class_count,
        "selected_source_count": gate.selected_source_count,
    }


def _manifest_document(
    *,
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    identity: _IdentityContext,
    result: _ReadbackResult,
    worker_count: int,
    memory_budget_bytes: int,
) -> Mapping[str, object]:
    shard_entries = []
    for receipt in result.receipts:
        first_rank = receipt.shard_index * config.shard_source_count
        shard_entries.append(
            {
                "cell_count": receipt.cell_count,
                "first_selection_rank": first_rank,
                "last_selection_rank": first_rank + receipt.source_count - 1,
                "logical_content_sha256": receipt.logical_content_sha256,
                "path": f"shards/{receipt.shard_index:05d}",
                "record_count": receipt.record_count,
                "shard_index": receipt.shard_index,
                "source_count": receipt.source_count,
                "source_ids_sha256": receipt.source_ids_sha256,
            }
        )
    return {
        "claim_boundary": "local_rebuild_artifact_not_redistribution_cleared",
        "counts": {
            "cell_count": result.stats.cell_count,
            "class_count": result.gate.selected_class_count,
            "record_count": result.stats.record_count,
            "shard_count": len(result.receipts),
            "source_count": result.gate.selected_source_count,
        },
        "data_code_snapshot": {
            path: dict(file_identity)
            for path, file_identity in identity.code_snapshot.items()
        },
        "data_code_snapshot_sha256": identity.code_snapshot_sha256,
        "distribution_status": config.distribution_status,
        "experiment_id": config.experiment_id,
        "git_commit": None,
        "operational": {
            "memory_budget_bytes": memory_budget_bytes,
            "worker_count": worker_count,
        },
        "operational_config": {
            "byte_count": config.file_byte_count,
            "sha256": config.file_sha256,
        },
        "run_id": identity.run_id,
        "schema_version": FORMAL_RUN_SCHEMA_VERSION,
        "scientific_config": {
            "byte_count": config.scientific_config_byte_count,
            "sha256": config.scientific_config_sha256,
        },
        "selected_source_subset_sha256": identity.subset_sha256,
        "selection": {
            "algorithm": config.selection_algorithm,
            "shard_source_count": config.shard_source_count,
        },
        "shards": shard_entries,
        "source_dataset_id": config.source_dataset_id,
        "source_snapshot_sha256": identity.source_snapshot_sha256,
        "sweep": {
            "byte_count": sweep.byte_count,
            "global_seed": sweep.global_seed,
            "sha256": sweep.sha256,
            "sweep_id": sweep.sweep_id,
        },
        "vcs_limitation": "workspace_not_git_do_not_infer_revision",
        "vcs_status": "unavailable",
    }


def _marker_document(run_id: str) -> Mapping[str, object]:
    return {
        "core_dataset_gate": "pass",
        "full_phase1_gate": "deferred_missing_p06_p07",
        "run_id": run_id,
        "schema_version": FORMAL_MARKER_SCHEMA_VERSION,
        "state": "complete",
    }


def _failed_marker_document(run_id: str) -> Mapping[str, object]:
    return {
        "core_dataset_gate": "fail",
        "full_phase1_gate": "fail",
        "run_id": run_id,
        "schema_version": FORMAL_MARKER_SCHEMA_VERSION,
        "state": "failed",
    }


def _root_inventory(run_path: Path) -> str:
    base = {
        "SHA256SUMS",
        "gate.json",
        "manifest.json",
        "shards",
        "source_subset.jsonl",
    }
    if not run_path.is_dir() or run_path.is_symlink():
        raise Phase1FormalError("run_path", "must be a non-symlink directory")
    names = {path.name for path in run_path.iterdir()}
    marker_names = names & {"complete.json", "failed.json"}
    if len(marker_names) != 1 or names != base | marker_names:
        raise Phase1FormalError("run_path.entries", "must match exact formal layout")
    for name in (base | marker_names) - {"shards"}:
        path = run_path / name
        if not path.is_file() or path.is_symlink():
            raise Phase1FormalError(name, "must be a non-symlink regular file")
    shards = run_path / "shards"
    if not shards.is_dir() or shards.is_symlink():
        raise Phase1FormalError("shards", "must be a non-symlink directory")
    return next(iter(marker_names))


def _checksum_member_paths(run_path: Path) -> tuple[str, ...]:
    members: list[str] = []
    for path in run_path.rglob("*"):
        if path.is_symlink():
            raise Phase1FormalError("SHA256SUMS", "symlinks are forbidden")
        if not path.is_file():
            continue
        relative = path.relative_to(run_path).as_posix()
        if relative in {"SHA256SUMS", "complete.json", "failed.json"}:
            continue
        pure = PurePosixPath(relative)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise Phase1FormalError("SHA256SUMS", "member path is not normalized")
        members.append(relative)
    return tuple(sorted(members))


def _write_checksums(run_path: Path) -> tuple[str, ...]:
    members = _checksum_member_paths(run_path)
    payload = "".join(
        f"{_sha256_file(run_path / relative)}  {relative}\n"
        for relative in members
    )
    (run_path / "SHA256SUMS").write_text(
        payload, encoding="utf-8", newline="\n"
    )
    return members


def _read_checksums(run_path: Path) -> tuple[str, ...]:
    expected_members = _checksum_member_paths(run_path)
    try:
        raw = (run_path / "SHA256SUMS").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise Phase1FormalError("SHA256SUMS", str(error)) from error
    lines = raw.splitlines(keepends=True)
    if not lines or any(not line.endswith("\n") for line in lines):
        raise Phase1FormalError("SHA256SUMS", "must use newline-terminated lines")
    parsed: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        body = line[:-1]
        if body.count("  ") != 1:
            raise Phase1FormalError(
                f"SHA256SUMS[{index}]", "must use two-space separator"
            )
        digest, relative = body.split("  ", 1)
        _lower_hex(f"SHA256SUMS[{index}].sha256", digest)
        parsed.append((relative, digest))
    members = tuple(relative for relative, _ in parsed)
    if members != expected_members:
        raise Phase1FormalError("SHA256SUMS.members", "member set or order mismatch")
    for relative, expected in parsed:
        observed = _sha256_file(run_path / relative)
        if observed != expected:
            raise Phase1FormalError(
                f"SHA256SUMS.{relative}", "file digest mismatch"
            )
    return members


def _read_canonical_json(path: Path) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase1FormalError(path.name, str(error)) from error
    if not isinstance(value, Mapping):
        raise Phase1FormalError(path.name, "must be a JSON object")
    if raw != canonical_json_bytes(value):
        raise Phase1FormalError(path.name, "must use canonical JSON bytes")
    return value


def _embedded_identity_context(
    *,
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    sources: tuple[Phase1Source, ...],
    manifest: Mapping[str, object],
) -> _IdentityContext:
    snapshot_value = manifest.get("data_code_snapshot")
    if not isinstance(snapshot_value, Mapping) or not snapshot_value:
        raise Phase1FormalError(
            "manifest.json.data_code_snapshot", "must be a nonempty object"
        )
    snapshot: dict[str, Mapping[str, int | str]] = {}
    previous_path: str | None = None
    for index, (path, identity_value) in enumerate(snapshot_value.items()):
        if not isinstance(path, str) or not path:
            raise Phase1FormalError(
                f"manifest.json.data_code_snapshot[{index}]",
                "path must be a nonempty string",
            )
        pure = PurePosixPath(path)
        if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
            raise Phase1FormalError(
                f"manifest.json.data_code_snapshot.{path}",
                "path must be normalized and relative",
            )
        if previous_path is not None and path <= previous_path:
            raise Phase1FormalError(
                "manifest.json.data_code_snapshot", "paths must be sorted and unique"
            )
        previous_path = path
        if not isinstance(identity_value, Mapping) or set(identity_value) != {
            "byte_count",
            "sha256",
        }:
            raise Phase1FormalError(
                f"manifest.json.data_code_snapshot.{path}",
                "must contain exact byte_count and sha256",
            )
        byte_count = identity_value.get("byte_count")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise Phase1FormalError(
                f"manifest.json.data_code_snapshot.{path}.byte_count",
                "must be a nonnegative integer",
            )
        sha256 = _lower_hex(
            f"manifest.json.data_code_snapshot.{path}.sha256",
            identity_value.get("sha256"),
        )
        snapshot[path] = {"byte_count": byte_count, "sha256": sha256}
    snapshot_bytes = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    code_sha256 = hashlib.sha256(
        b"rpe-phase1-code-snapshot-v1\0" + snapshot_bytes
    ).hexdigest()
    if manifest.get("data_code_snapshot_sha256") != code_sha256:
        raise Phase1FormalError(
            "manifest.json.data_code_snapshot_sha256",
            "does not match embedded code snapshot",
        )
    subset_bytes = source_subset_jsonl_bytes(sources)
    subset_sha256 = hashlib.sha256(subset_bytes).hexdigest()
    source_sha256 = source_snapshot_digest(config)
    run_id = derive_phase1_formal_run_id(
        schema_version=FORMAL_RUN_SCHEMA_VERSION,
        scientific_config_sha256=config.scientific_config_sha256,
        sweep_config_sha256=sweep.sha256,
        source_snapshot_sha256=source_sha256,
        selected_source_subset_sha256=subset_sha256,
        data_code_snapshot_sha256=code_sha256,
    )
    return _IdentityContext(
        subset_bytes=subset_bytes,
        subset_sha256=subset_sha256,
        source_snapshot_sha256=source_sha256,
        code_snapshot=MappingProxyType(snapshot),
        code_snapshot_sha256=code_sha256,
        run_id=run_id,
    )


def _summary(
    run_path: Path,
    result: _ReadbackResult,
    checked_files: tuple[str, ...],
) -> Phase1FormalSummary:
    return Phase1FormalSummary(
        path=run_path,
        run_id=run_path.name,
        source_count=result.gate.selected_source_count,
        class_count=result.gate.selected_class_count,
        shard_count=len(result.receipts),
        cell_count=result.stats.cell_count,
        record_count=result.stats.record_count,
        core_dataset_gate=result.gate.core_dataset_gate,
        full_phase1_gate=result.gate.full_phase1_gate,
        checked_files=checked_files,
    )


def build_phase1_formal_run(
    output_root: Path,
    *,
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    sources: Sequence[Phase1Source],
    worker_count: int,
    memory_budget_bytes: int,
) -> Phase1FormalSummary:
    parsed_sources = _validate_inputs(config, sweep, sources)
    workers = _positive_int("worker_count", worker_count)
    budget = _positive_int("memory_budget_bytes", memory_budget_bytes)
    largest_p10_estimate = max(
        estimate_p10_peak_bytes(source.spectrum.intensity.size)
        for source in parsed_sources
    )
    if budget < largest_p10_estimate:
        raise Phase1FormalError(
            "memory_budget_bytes",
            "must admit the largest P10 estimate before output creation",
        )
    identity = _identity_context(config, sweep, parsed_sources)
    output_root = Path(output_root)
    run_path = output_root / identity.run_id
    if run_path.exists() or run_path.is_symlink():
        raise Phase1FormalError("output", "already exists")
    output_root.mkdir(parents=True, exist_ok=True)
    run_path.mkdir()
    (run_path / "shards").mkdir()
    (run_path / "source_subset.jsonl").write_bytes(identity.subset_bytes)

    shard_count = math.ceil(len(parsed_sources) / config.shard_source_count)
    for shard_index in range(shard_count):
        start = shard_index * config.shard_source_count
        shard_sources = parsed_sources[start : start + config.shard_source_count]
        payload = run_shard_payload(
            shard_sources,
            config,
            sweep,
            shard_index=shard_index,
            worker_count=min(workers, len(shard_sources)),
            memory_budget_bytes=budget,
        )
        write_perturbed_shard(
            payload,
            run_path / "shards" / f"{shard_index:05d}",
            run_id=identity.run_id,
            scientific_config_sha256=config.scientific_config_sha256,
            sweep_config_sha256=sweep.sha256,
        )

    result = _readback(
        run_path,
        config=config,
        sweep=sweep,
        sources=parsed_sources,
    )
    gate_document = _gate_document(run_id=identity.run_id, result=result)
    manifest_document = _manifest_document(
        config=config,
        sweep=sweep,
        identity=identity,
        result=result,
        worker_count=workers,
        memory_budget_bytes=budget,
    )
    (run_path / "gate.json").write_bytes(canonical_json_bytes(gate_document))
    (run_path / "manifest.json").write_bytes(
        canonical_json_bytes(manifest_document)
    )
    _write_checksums(run_path)
    if (
        result.gate.core_dataset_gate != "pass"
        or result.gate.full_phase1_gate != "deferred_missing_p06_p07"
    ):
        (run_path / "failed.json").write_bytes(
            canonical_json_bytes(_failed_marker_document(identity.run_id))
        )
        raise Phase1FormalError("gate", "formal core gate failed")
    (run_path / "complete.json").write_bytes(
        canonical_json_bytes(_marker_document(identity.run_id))
    )
    return verify_phase1_formal_run(
        run_path,
        config=config,
        sweep=sweep,
        sources=parsed_sources,
    )


def verify_phase1_formal_run(
    run_path: Path,
    *,
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
    sources: Sequence[Phase1Source],
) -> Phase1FormalSummary:
    parsed_sources = _validate_inputs(config, sweep, sources)
    run_path = Path(run_path)
    marker_name = _root_inventory(run_path)
    expected_shards = tuple(
        f"{index:05d}"
        for index in range(
            math.ceil(len(parsed_sources) / config.shard_source_count)
        )
    )
    observed_shards = tuple(sorted(path.name for path in (run_path / "shards").iterdir()))
    if observed_shards != expected_shards:
        raise Phase1FormalError("shards", "partition inventory mismatch")
    checked_files = _read_checksums(run_path)
    manifest = _read_canonical_json(run_path / "manifest.json")
    identity = _embedded_identity_context(
        config=config,
        sweep=sweep,
        sources=parsed_sources,
        manifest=manifest,
    )
    if run_path.name != identity.run_id:
        raise Phase1FormalError("run_path", "basename does not match run identity")
    if (run_path / "source_subset.jsonl").read_bytes() != identity.subset_bytes:
        raise Phase1FormalError(
            "source_subset.jsonl", "does not match selected sources"
        )
    gate = _read_canonical_json(run_path / "gate.json")
    marker = _read_canonical_json(run_path / marker_name)
    operational = manifest.get("operational")
    if not isinstance(operational, Mapping):
        raise Phase1FormalError("manifest.json.operational", "must be an object")
    workers = _positive_int(
        "manifest.json.operational.worker_count", operational.get("worker_count")
    )
    budget = _positive_int(
        "manifest.json.operational.memory_budget_bytes",
        operational.get("memory_budget_bytes"),
    )
    result = _readback(
        run_path,
        config=config,
        sweep=sweep,
        sources=parsed_sources,
    )
    expected_manifest = _manifest_document(
        config=config,
        sweep=sweep,
        identity=identity,
        result=result,
        worker_count=workers,
        memory_budget_bytes=budget,
    )
    expected_gate = _gate_document(run_id=identity.run_id, result=result)
    if manifest != expected_manifest:
        raise Phase1FormalError("manifest.json", "does not match recomputation")
    if gate != expected_gate:
        raise Phase1FormalError("gate.json", "does not match recomputation")
    if result.gate.core_dataset_gate == "pass":
        expected_marker_name = "complete.json"
        expected_marker = _marker_document(identity.run_id)
    else:
        expected_marker_name = "failed.json"
        expected_marker = _failed_marker_document(identity.run_id)
    if marker_name != expected_marker_name or marker != expected_marker:
        raise Phase1FormalError(marker_name, "does not match recomputed gate")
    return _summary(run_path, result, checked_files)


def _load_formal_sources(
    config: Phase1CoreConfig,
    sweep: PerturbationSweepConfig,
) -> tuple[Phase1Source, ...]:
    inventory = load_source_inventory(config)
    rows = select_source_rows(
        inventory,
        global_seed=sweep.global_seed,
        subset_size=config.subset_size,
    )
    with UnifiedDataset.open(
        REPO_ROOT / config.source_dataset_path,
        verify_checksums=False,
    ) as dataset:
        return tuple(load_phase1_source(dataset, row) for row in rows)


def build_rruff_phase1_formal(
    *,
    output_root: Path = REPO_ROOT / _DEFAULT_OUTPUT_ROOT,
    worker_count: int = 32,
    memory_budget_bytes: int = 64 * 2**30,
) -> Phase1FormalSummary:
    config = load_phase1_core_config(REPO_ROOT / _FORMAL_CONFIG_PATH)
    sweep = load_perturbation_sweep_config(REPO_ROOT / config.shared_sweep_path)
    sources = _load_formal_sources(config, sweep)
    return build_phase1_formal_run(
        output_root,
        config=config,
        sweep=sweep,
        sources=sources,
        worker_count=worker_count,
        memory_budget_bytes=memory_budget_bytes,
    )


def verify_rruff_phase1_formal(run_path: Path) -> Phase1FormalSummary:
    config = load_phase1_core_config(REPO_ROOT / _FORMAL_CONFIG_PATH)
    sweep = load_perturbation_sweep_config(REPO_ROOT / config.shared_sweep_path)
    sources = _load_formal_sources(config, sweep)
    return verify_phase1_formal_run(
        run_path,
        config=config,
        sweep=sweep,
        sources=sources,
    )


def _summary_json(summary: Phase1FormalSummary) -> str:
    return json.dumps(
        {
            "cell_count": summary.cell_count,
            "checked_file_count": len(summary.checked_files),
            "class_count": summary.class_count,
            "core_dataset_gate": summary.core_dataset_gate,
            "full_phase1_gate": summary.full_phase1_gate,
            "path": str(summary.path),
            "record_count": summary.record_count,
            "run_id": summary.run_id,
            "shard_count": summary.shard_count,
            "source_count": summary.source_count,
        },
        sort_keys=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or verify the formal Phase 1 RRUFF perturbation run."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument(
        "--output-root", type=Path, default=REPO_ROOT / _DEFAULT_OUTPUT_ROOT
    )
    build.add_argument("--worker-count", type=int, default=32)
    build.add_argument("--memory-budget-gib", type=int, default=64)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        summary = build_rruff_phase1_formal(
            output_root=arguments.output_root,
            worker_count=arguments.worker_count,
            memory_budget_bytes=arguments.memory_budget_gib * 2**30,
        )
    else:
        summary = verify_rruff_phase1_formal(arguments.run_path)
    print(_summary_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FORMAL_GATE_SCHEMA_VERSION",
    "FORMAL_MARKER_SCHEMA_VERSION",
    "FORMAL_RUN_SCHEMA_VERSION",
    "Phase1FormalError",
    "Phase1FormalSummary",
    "build_phase1_formal_run",
    "build_rruff_phase1_formal",
    "derive_phase1_formal_run_id",
    "main",
    "verify_phase1_formal_run",
    "verify_rruff_phase1_formal",
]
