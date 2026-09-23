from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.evaluation import Spectrum1D
from rpe.io.store import UnifiedDataset
from rpe.methods.catalog import Phase3System, TaskLine, load_classical_catalog
from rpe.methods.classical.denoising import (
    DenoisingFitContext,
    DenoisingFitError,
    DenoisingRunStatus,
    DenoisingWarning,
    FittedDenoiser,
    fit_denoising_system,
    run_stateless_denoising_system,
    transform_fitted_denoiser,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "experiments/phase3/configs/denoising_v1_formal_coverage.json"
CONFIG_BYTES = 4659
CONFIG_SHA256 = "248b3fcfce517ada2cd3536865d809b5be6f684981c612776b84ba22670406ab"
RUN_DOMAIN = b"rpe-phase3-denoising-v1-formal-coverage-v1\0"
STATELESS_FAMILIES = frozenset({"savitzky_golay", "wavelet", "whittaker_smoothing"})
FITTED_FAMILIES = frozenset({"pca_reconstruction", "svd_reconstruction"})
SUCCESS = frozenset({"complete", "complete_with_warning"})


class DenoisingFormalVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class _Summary:
    path: Path
    run_id: str
    status: str
    is_formal_run: bool
    system_count: int
    fit_receipt_count: int
    transform_receipt_count: int
    promoted_system_count: int
    warning_count: int


@dataclass(frozen=True)
class _Source:
    cohort_id: str
    record_id: str
    class_label: int
    source_order: int
    point_count: int
    axis_sha256: str
    intensity_sha256: str
    spectrum: Spectrum1D
    sample_id: str | None
    mineral_name: str | None


@dataclass(frozen=True)
class _Inputs:
    stateless: tuple[_Source, ...]
    train: tuple[_Source, ...]
    validation: tuple[_Source, ...]
    test: tuple[_Source, ...]
    axis_sha256: str
    train_ids_sha256: str
    validation_ids_sha256: str
    test_ids_sha256: str
    train_matrix_sha256: str
    validation_matrix_sha256: str
    test_matrix_sha256: str


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def _json_ready(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    return value


def _read_config(root: Path) -> tuple[Mapping[str, object], bytes]:
    raw = (root / CONFIG_PATH.relative_to(ROOT)).read_bytes()
    value = json.loads(raw)
    if (
        not isinstance(value, Mapping)
        or raw != _canonical(value)
        or len(raw) != CONFIG_BYTES
        or hashlib.sha256(raw).hexdigest() != CONFIG_SHA256
    ):
        raise DenoisingFormalVerificationError("frozen config identity mismatch")
    for section in ("authorities",):
        values = value[section]
        assert isinstance(values, Mapping)
        for name, identity in values.items():
            assert isinstance(identity, Mapping)
            path = root / str(identity["path"])
            if (
                not path.is_file()
                or path.stat().st_size != int(identity["byte_count"])
                or _sha(path) != str(identity["sha256"])
            ):
                raise DenoisingFormalVerificationError(
                    f"authority {name} identity mismatch"
                )
    for name in ("catalog", "phase3_lock"):
        identity = value[name]
        assert isinstance(identity, Mapping)
        path = root / str(identity["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(identity["byte_count"])
            or _sha(path) != str(identity["sha256"])
        ):
            raise DenoisingFormalVerificationError(f"{name} identity mismatch")
    rruff = value["rruff"]
    bacteria = value["bacteria"]
    assert isinstance(rruff, Mapping)
    assert isinstance(bacteria, Mapping)
    for name, identity in (
        ("rruff.phase1_manifest", rruff["phase1_manifest"]),
        ("bacteria.d1_config", bacteria["d1_config"]),
        ("bacteria.d1_seed0_result", bacteria["d1_seed0_result"]),
    ):
        assert isinstance(identity, Mapping)
        path = root / str(identity["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(identity["byte_count"])
            or _sha(path) != str(identity["sha256"])
        ):
            raise DenoisingFormalVerificationError(f"{name} identity mismatch")
    if (
        _sha(root / str(bacteria["dataset_path"]) / "SHA256SUMS")
        != str(bacteria["dataset_sha256sums_sha256"])
    ):
        raise DenoisingFormalVerificationError(
            "Bacteria SHA256SUMS identity mismatch"
        )
    phase1_manifest = json.loads(
        (root / str(rruff["phase1_manifest"]["path"])).read_bytes()
    )
    if (
        phase1_manifest.get("selected_source_subset_sha256")
        != rruff["source_ledger_sha256"]
        or phase1_manifest.get("scientific_config", {}).get("sha256")
        != rruff["scientific_config_sha256"]
        or phase1_manifest.get("source_snapshot_sha256")
        != rruff["source_snapshot_sha256"]
    ):
        raise DenoisingFormalVerificationError(
            "Phase 1 manifest scientific identity mismatch"
        )
    return value, raw


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows = []
    for line in path.read_bytes().splitlines(keepends=True):
        value = json.loads(line)
        if not isinstance(value, Mapping) or line != _canonical(value):
            raise DenoisingFormalVerificationError(
                f"{path.name} must be canonical JSONL"
            )
        rows.append(value)
    return rows


def _slice_sha(stream, offset: int, byte_count: int, label: str) -> str:
    stream.seek(offset)
    remaining = byte_count
    digest = hashlib.sha256()
    while remaining:
        chunk = stream.read(min(remaining, 1024 * 1024))
        if not chunk:
            raise DenoisingFormalVerificationError(f"{label} truncated slice")
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _validate_stream_contracts(path: Path) -> None:
    fit_rows = _read_jsonl(path / "fit_receipts.jsonl")
    transform_rows = _read_jsonl(path / "transform_receipts.jsonl")
    state_by_system: dict[str, str] = {}
    expected_state_offset = 0
    with (path / "fitted_states.f64le").open("rb") as state_stream:
        for row in fit_rows:
            if row["fit_state_sha256"] is None:
                if row["fit_state_offset_bytes"] is not None or row["fit_state_byte_count"] is not None:
                    raise DenoisingFormalVerificationError(
                        "fit receipt identity failure carries state"
                    )
                continue
            if int(row["fit_state_offset_bytes"]) != expected_state_offset:
                raise DenoisingFormalVerificationError(
                    "fit receipt identity state offset mismatch"
                )
            diagnostics = row["diagnostics"]
            if not isinstance(diagnostics, Mapping):
                raise DenoisingFormalVerificationError(
                    "fit receipt identity diagnostics malformed"
                )
            descriptors = diagnostics.get("arrays")
            if not isinstance(descriptors, list):
                raise DenoisingFormalVerificationError(
                    "fit receipt identity array descriptors malformed"
                )
            reconstructed: dict[str, np.ndarray] = {}
            for descriptor in descriptors:
                if not isinstance(descriptor, Mapping):
                    raise DenoisingFormalVerificationError(
                        "fit receipt identity descriptor malformed"
                    )
                offset = int(descriptor["offset"])
                byte_count = int(descriptor["byte_count"])
                if (
                    offset != expected_state_offset
                    or _slice_sha(
                        state_stream, offset, byte_count, "fit receipt identity"
                    )
                    != descriptor["sha256"]
                ):
                    raise DenoisingFormalVerificationError(
                        "fit receipt identity state descriptor mismatch"
                    )
                state_stream.seek(offset)
                array = np.frombuffer(
                    state_stream.read(byte_count), dtype="<f8"
                ).reshape(tuple(int(value) for value in descriptor["shape"]))
                reconstructed[str(descriptor["role"])] = array
                expected_state_offset += byte_count
            warnings = tuple(
                DenoisingWarning(str(value["category"]), str(value["message"]))
                for value in row["warnings"]
            )
            fitted = FittedDenoiser(
                system_id=str(row["system_id"]),
                family_id=str(row["family_id"]),
                method_id=str(row["method_id"]),
                n_components=int(diagnostics["n_components"]),
                axis_sha256=str(row["axis_sha256"]),
                training_record_ledger_sha256=str(
                    row["training_record_ledger_sha256"]
                ),
                training_matrix_sha256=str(row["training_matrix_sha256"]),
                context_sha256=str(row["context_sha256"]),
                mean=reconstructed.get("mean"),
                components=reconstructed["components"],
                explained_variance=reconstructed["explained_variance"],
                warnings=warnings,
            )
            if fitted.state_sha256 != row["fit_state_sha256"]:
                raise DenoisingFormalVerificationError(
                    "fit receipt identity state hash mismatch"
                )
            state_by_system[str(row["system_id"])] = fitted.state_sha256
        if expected_state_offset != (path / "fitted_states.f64le").stat().st_size:
            raise DenoisingFormalVerificationError(
                "fit receipt identity unindexed state bytes"
            )
    expected_output_offset = 0
    last_key = None
    with (path / "denoised_outputs.f64le").open("rb") as output_stream:
        for row in transform_rows:
            role = str(row["source_role"])
            key = (
                0 if role == "stateless" else 1,
                str(row["system_id"]),
                {"stateless": 0, "validation": 0, "test": 1}[role],
                int(row["source_order"]),
            )
            if last_key is not None and key <= last_key:
                raise DenoisingFormalVerificationError(
                    "output stream receipt order mismatch"
                )
            last_key = key
            output_sha = row["output_sha256"]
            if output_sha is None:
                if (
                    row["output_offset_bytes"] is not None
                    or row["output_byte_count"] is not None
                ):
                    raise DenoisingFormalVerificationError(
                        "output stream failure carries output"
                    )
            else:
                offset = int(row["output_offset_bytes"])
                byte_count = int(row["output_byte_count"])
                if (
                    offset != expected_output_offset
                    or byte_count != int(row["point_count"]) * 8
                    or _slice_sha(output_stream, offset, byte_count, "output stream")
                    != output_sha
                ):
                    raise DenoisingFormalVerificationError(
                        "output stream descriptor mismatch"
                    )
                expected_output_offset += byte_count
            if role in {"validation", "test"} and output_sha is not None:
                if (
                    state_by_system.get(str(row["system_id"]))
                    != row["fitted_state_sha256"]
                ):
                    raise DenoisingFormalVerificationError(
                        "fit receipt identity transform state binding mismatch"
                    )
        if expected_output_offset != (path / "denoised_outputs.f64le").stat().st_size:
            raise DenoisingFormalVerificationError(
                "output stream contains unindexed bytes"
            )


def _source_row(source: _Source, role: str) -> dict[str, object]:
    return {
        "axis_sha256": source.axis_sha256,
        "class_label": source.class_label,
        "cohort_id": source.cohort_id,
        "intensity_sha256": source.intensity_sha256,
        "mineral_name": source.mineral_name,
        "point_count": source.point_count,
        "record_id": source.record_id,
        "role": role,
        "sample_id": source.sample_id,
        "source_order": source.source_order,
    }


def _system_row(system: Phase3System) -> dict[str, object]:
    return {
        "family_id": system.family_id,
        "hyperparameters": _json_ready(system.hyperparameters),
        "kind": "fitted" if system.family_id in FITTED_FAMILIES else "stateless",
        "method_id": system.method_id,
        "system_id": system.system_id,
    }


def _load_inputs(config: Mapping[str, object], root: Path) -> _Inputs:
    rruff = config["rruff"]
    bacteria = config["bacteria"]
    expected = config["expected"]
    assert isinstance(rruff, Mapping)
    assert isinstance(bacteria, Mapping)
    assert isinstance(expected, Mapping)
    ledger_path = root / str(rruff["source_ledger_path"])
    if (
        ledger_path.stat().st_size != int(rruff["source_ledger_size"])
        or _sha(ledger_path) != str(rruff["source_ledger_sha256"])
    ):
        raise DenoisingFormalVerificationError("RRUFF source ledger mismatch")
    rows = [json.loads(line) for line in ledger_path.read_bytes().splitlines()]
    stateless: list[_Source] = []
    with UnifiedDataset.open(root / str(rruff["dataset_path"]), verify_checksums=True) as dataset:
        for order, row in enumerate(rows):
            if int(row["selection_rank"]) != order:
                raise DenoisingFormalVerificationError("RRUFF source order mismatch")
            record = dataset.get(str(row["record_id"]))
            axis = np.asarray(record.wavenumber, dtype="<f8")
            intensity = np.asarray(record.intensity, dtype="<f8")
            if np.all(np.diff(axis) < 0.0):
                axis = np.ascontiguousarray(axis[::-1])
                intensity = np.ascontiguousarray(intensity[::-1])
            spectrum = Spectrum1D(str(row["record_id"]), record.meta.sample_id, axis, intensity)
            if (
                _array_sha(axis) != row["normalized_axis_float64_sha256"]
                or _array_sha(intensity) != row["normalized_intensity_float64_sha256"]
            ):
                raise DenoisingFormalVerificationError("RRUFF source hash mismatch")
            stateless.append(
                _Source(
                    "rruff_core10k", str(row["record_id"]), int(row["class_label"]),
                    order, intensity.size, _array_sha(axis), _array_sha(intensity), spectrum,
                    str(row["sample_id"]), str(row["mineral_name"]),
                )
            )
    if (
        len(stateless) != int(expected["stateless_source_count"])
        or sum(value.point_count for value in stateless)
        != int(expected["stateless_source_point_count"])
    ):
        raise DenoisingFormalVerificationError("RRUFF frozen count mismatch")

    split_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("finetune", "reference", "test")}
    split_labels: dict[str, list[np.ndarray]] = {key: [] for key in split_arrays}
    split_ids: dict[str, list[str]] = {key: [] for key in split_arrays}
    with BacteriaIdBatchLoader(root / str(bacteria["dataset_path"]), batch_size=4096) as loader:
        raw_axis = None
        for batch in loader.iter_batches():
            split_arrays[batch.source_split].append(np.asarray(batch.intensity, dtype="<f4"))
            split_labels[batch.source_split].append(np.asarray(batch.class_labels, dtype="<i8"))
            split_ids[batch.source_split].extend(batch.record_ids)
            if raw_axis is None:
                raw_axis = np.asarray(batch.wavenumber, dtype="<f8")
    assert raw_axis is not None
    axis = np.ascontiguousarray(raw_axis[::-1]) if np.all(np.diff(raw_axis) < 0.0) else np.ascontiguousarray(raw_axis)
    if _array_sha(axis) != str(bacteria["axis_sha256"]):
        raise DenoisingFormalVerificationError("Bacteria axis mismatch")
    arrays = {key: np.concatenate(value, axis=0) for key, value in split_arrays.items()}
    labels = {key: np.concatenate(value, axis=0) for key, value in split_labels.items()}
    generator = np.random.default_rng(int(bacteria["seed"]))
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_label in range(30):
        indices = np.flatnonzero(labels["finetune"] == class_label)
        shuffled = generator.permutation(indices)
        cut = int(bacteria["validation_per_class"])
        validation_indices.extend(shuffled[:cut].tolist())
        train_indices.extend(shuffled[cut:].tolist())
    train_indices.sort()
    validation_indices.sort()

    train_matrix = np.concatenate(
        (arrays["reference"], arrays["finetune"][train_indices]), axis=0
    )[:, ::-1]
    validation_matrix = arrays["finetune"][validation_indices, ::-1]
    test_matrix = arrays["test"][:, ::-1]
    train_labels = np.concatenate(
        (labels["reference"], labels["finetune"][train_indices])
    )
    validation_labels = labels["finetune"][validation_indices]
    test_labels = labels["test"]
    train_ids = tuple(split_ids["reference"]) + tuple(
        split_ids["finetune"][index] for index in train_indices
    )
    validation_ids = tuple(split_ids["finetune"][index] for index in validation_indices)
    test_ids = tuple(split_ids["test"])

    def materialize(
        cohort: str, matrix: np.ndarray, current_labels: np.ndarray, record_ids: Sequence[str]
    ) -> tuple[_Source, ...]:
        result = []
        for order, record_id in enumerate(record_ids):
            intensity = np.ascontiguousarray(matrix[order], dtype="<f8")
            spectrum = Spectrum1D(record_id, None, axis, intensity)
            result.append(
                _Source(
                    cohort, record_id, int(current_labels[order]), order, intensity.size,
                    _array_sha(axis), _array_sha(intensity), spectrum, None, None,
                )
            )
        return tuple(result)

    train = materialize("bacteria_train", train_matrix, train_labels, train_ids)
    validation = materialize(
        "bacteria_validation", validation_matrix, validation_labels, validation_ids
    )
    test = materialize("bacteria_test", test_matrix, test_labels, test_ids)

    identity_values = {
        "train_record_ids_sha256": hashlib.sha256(("\n".join(train_ids) + "\n").encode()).hexdigest(),
        "validation_record_ids_sha256": hashlib.sha256(("\n".join(validation_ids) + "\n").encode()).hexdigest(),
        "test_record_ids_sha256": hashlib.sha256(("\n".join(test_ids) + "\n").encode()).hexdigest(),
        "train_matrix_sha256": _array_sha(train_matrix),
        "validation_matrix_sha256": _array_sha(validation_matrix),
        "test_matrix_sha256": _array_sha(test_matrix),
    }
    for key, observed in identity_values.items():
        if observed != str(bacteria[key]):
            raise DenoisingFormalVerificationError(f"Bacteria {key} mismatch")
    return _Inputs(
        tuple(stateless), train, validation, test, str(bacteria["axis_sha256"]),
        identity_values["train_record_ids_sha256"],
        identity_values["validation_record_ids_sha256"],
        identity_values["test_record_ids_sha256"],
        identity_values["train_matrix_sha256"],
        identity_values["validation_matrix_sha256"],
        identity_values["test_matrix_sha256"],
    )


_INPUTS: _Inputs | None = None
_SYSTEMS: tuple[Phase3System, ...] = ()


def _pack_result(result) -> dict[str, object]:
    return {
        "system_id": result.system_id,
        "family_id": result.family_id,
        "method_id": result.method_id,
        "status": result.status.value,
        "output": result.denoised_intensity,
        "output_sha256": result.output_sha256,
        "warnings": tuple((value.category, value.message) for value in result.warnings),
        "diagnostics": dict(result.diagnostics),
        "error_code": result.error_code,
        "error_message": result.error_message,
    }


def _stateless_job(task: tuple[int, int, int]) -> tuple[int, list[dict[str, object]]]:
    if _INPUTS is None:
        raise RuntimeError("verifier inputs are not initialized")
    system_index, start, stop = task
    with threadpool_limits(limits=1):
        return start, [
            _pack_result(run_stateless_denoising_system(_SYSTEMS[system_index], source.spectrum))
            for source in _INPUTS.stateless[start:stop]
        ]


def _fitted_job(system_index: int) -> dict[str, object]:
    if _INPUTS is None:
        raise RuntimeError("verifier inputs are not initialized")
    system = _SYSTEMS[system_index]
    context = DenoisingFitContext(
        "d1_seed0_train", "bacteria_id_reference_increasing_float64_v1",
        tuple(value.record_id for value in _INPUTS.train),
    )
    try:
        with threadpool_limits(limits=1):
            fitted = fit_denoising_system(
                system, tuple(value.spectrum for value in _INPUTS.train), context
            )
            results = [
                _pack_result(transform_fitted_denoiser(fitted, source.spectrum))
                for source in (*_INPUTS.validation, *_INPUTS.test)
            ]
    except DenoisingFitError as error:
        return {
            "failure": (error.path, error.reason, error.status.value),
            "results": [],
        }
    return {
        "failure": None,
        "axis_sha256": fitted.axis_sha256,
        "components": fitted.components,
        "context_sha256": fitted.context_sha256,
        "explained_variance": fitted.explained_variance,
        "mean": fitted.mean,
        "n_components": fitted.n_components,
        "results": results,
        "state_sha256": fitted.state_sha256,
        "training_matrix_sha256": fitted.training_matrix_sha256,
        "training_record_ledger_sha256": fitted.training_record_ledger_sha256,
        "warnings": tuple((value.category, value.message) for value in fitted.warnings),
    }


def _warnings(values: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"category": category, "message": message} for category, message in values]


def _warning_rows(
    values: Sequence[tuple[str, str]], kind: str, system_id: str,
    record_id: str | None, role: str,
) -> list[dict[str, object]]:
    return [
        {
            "category": category, "message": message, "receipt_kind": kind,
            "record_id": record_id, "sequence": sequence,
            "source_role": role, "system_id": system_id,
        }
        for sequence, (category, message) in enumerate(values)
    ]


def _code_identity(root: Path) -> dict[str, dict[str, object]]:
    paths = (
        "rpe/methods/catalog.py",
        "rpe/methods/classical/denoising.py",
        "rpe/runner/d1_bacteria_id.py",
        "rpe/runner/phase3_denoising_formal.py",
        "rpe/runner/phase3_denoising_formal_verifier.py",
        "tools/run_phase3_denoising_formal.py",
    )
    return {
        name: {"byte_count": (root / name).stat().st_size, "sha256": _sha(root / name)}
        for name in paths
    }


def _software_identity() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "scipy", "scikit-learn", "PyWavelets", "h5py")
    }


def _build_reconstruction(
    config: Mapping[str, object], config_raw: bytes, inputs: _Inputs,
    systems: tuple[Phase3System, ...], output_root: Path, worker_count: int, root: Path,
) -> _Summary:
    global _INPUTS, _SYSTEMS
    _INPUTS = inputs
    _SYSTEMS = systems
    catalog_value = config["catalog"]
    lock_value = config["phase3_lock"]
    expected = config["expected"]
    policy = config["policy"]
    assert isinstance(catalog_value, Mapping)
    assert isinstance(lock_value, Mapping)
    assert isinstance(expected, Mapping)
    assert isinstance(policy, Mapping)
    input_identity = {
        "axis_sha256": inputs.axis_sha256,
        "stateless_source_count": len(inputs.stateless),
        "fit_source_count": len(inputs.train),
        "validation_source_count": len(inputs.validation),
        "test_source_count": len(inputs.test),
        "train_record_ids_sha256": inputs.train_ids_sha256,
        "validation_record_ids_sha256": inputs.validation_ids_sha256,
        "test_record_ids_sha256": inputs.test_ids_sha256,
        "train_matrix_sha256": inputs.train_matrix_sha256,
        "validation_matrix_sha256": inputs.validation_matrix_sha256,
        "test_matrix_sha256": inputs.test_matrix_sha256,
        "stateless_sources_sha256": hashlib.sha256(
            b"".join(_canonical(_source_row(value, "stateless")) for value in inputs.stateless)
        ).hexdigest(),
    }
    identity = {
        "catalog_id": str(catalog_value["catalog_id"]),
        "catalog_sha256": str(catalog_value["sha256"]),
        "code_identity": _code_identity(root),
        "config_sha256": CONFIG_SHA256,
        "input_identity": input_identity,
        "phase3_lock_sha256": str(lock_value["sha256"]),
        "software_identity": _software_identity(),
        "system_ids": [value.system_id for value in systems],
    }
    run_id = hashlib.sha256(RUN_DOMAIN + _canonical(identity)).hexdigest()
    path = output_root / f"phase3-denoising-v1-formal-{run_id}"
    path.mkdir(parents=True)
    (path / "config.json").write_bytes(config_raw)
    source_rows = [
        *(_source_row(value, "stateless") for value in inputs.stateless),
        *(_source_row(value, "fit") for value in inputs.train),
        *(_source_row(value, "validation") for value in inputs.validation),
        *(_source_row(value, "test") for value in inputs.test),
    ]
    (path / "sources.jsonl").write_bytes(b"".join(_canonical(value) for value in source_rows))
    (path / "systems.jsonl").write_bytes(
        b"".join(_canonical(_system_row(value)) for value in systems)
    )
    stateless_systems = tuple(value for value in systems if value.family_id in STATELESS_FAMILIES)
    fitted_systems = tuple(value for value in systems if value.family_id in FITTED_FAMILIES)
    fit_rows: list[dict[str, object]] = []
    transform_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    with (path / "denoised_outputs.f64le").open("wb") as output, (
        path / "fitted_states.f64le"
    ).open("wb") as states:
        for system in stateless_systems:
            system_index = systems.index(system)
            tasks = [
                (system_index, start, min(start + 16, len(inputs.stateless)))
                for start in range(0, len(inputs.stateless), 16)
            ]
            executor = ProcessPoolExecutor(
                max_workers=worker_count, mp_context=multiprocessing.get_context("fork")
            )
            try:
                for start, results in executor.map(_stateless_job, tasks, chunksize=1):
                    for relative, result in enumerate(results):
                        source = inputs.stateless[start + relative]
                        raw = None
                        if result["output"] is not None:
                            raw = np.ascontiguousarray(result["output"], dtype="<f8").tobytes()
                        offset = output.tell() if raw is not None else None
                        if raw is not None:
                            output.write(raw)
                        transform_rows.append(
                            {
                                "axis_sha256": source.axis_sha256,
                                "cohort": "stateless",
                                "diagnostics": result["diagnostics"],
                                "error_code": result["error_code"],
                                "error_message": result["error_message"],
                                "family_id": system.family_id,
                                "fitted_state_sha256": None,
                                "method_id": system.method_id,
                                "output_byte_count": None if raw is None else len(raw),
                                "output_offset_bytes": offset,
                                "output_sha256": result["output_sha256"],
                                "point_count": source.point_count,
                                "record_id": source.record_id,
                                "source_order": source.source_order,
                                "source_role": "stateless",
                                "status": result["status"],
                                "system_id": system.system_id,
                                "warnings": _warnings(result["warnings"]),
                            }
                        )
                        warning_rows.extend(
                            _warning_rows(result["warnings"], "transform", system.system_id, source.record_id, "stateless")
                        )
            finally:
                executor.shutdown()
        fit_indices = [systems.index(value) for value in fitted_systems]
        fit_executor = ProcessPoolExecutor(
            max_workers=min(worker_count, len(fit_indices)),
            mp_context=multiprocessing.get_context("fork"),
        )
        try:
            for system, result in zip(
                fitted_systems, fit_executor.map(_fitted_job, fit_indices, chunksize=1), strict=True
            ):
                failure = result["failure"]
                ordered_sources = (*inputs.validation, *inputs.test)
                if failure is not None:
                    error_path, error_reason, error_status = failure
                    fit_rows.append(
                        {
                            "axis_sha256": None, "context_sha256": None,
                            "diagnostics": {}, "error_code": error_path,
                            "error_message": error_reason, "family_id": system.family_id,
                            "fit_state_byte_count": None, "fit_state_offset_bytes": None,
                            "fit_state_sha256": None, "method_id": system.method_id,
                            "representation_id": "bacteria_id_reference_increasing_float64_v1",
                            "split_id": "d1_seed0_train", "status": error_status,
                            "system_id": system.system_id, "training_matrix_sha256": None,
                            "training_record_ledger_sha256": None, "warnings": [],
                        }
                    )
                    for source_index, source in enumerate(ordered_sources):
                        role = "validation" if source_index < len(inputs.validation) else "test"
                        transform_rows.append(
                            {
                                "axis_sha256": source.axis_sha256, "cohort": "fitted",
                                "diagnostics": {}, "error_code": error_path,
                                "error_message": error_reason, "family_id": system.family_id,
                                "fitted_state_sha256": None, "method_id": system.method_id,
                                "output_byte_count": None, "output_offset_bytes": None,
                                "output_sha256": None, "point_count": source.point_count,
                                "record_id": source.record_id, "source_order": source.source_order,
                                "source_role": role, "status": DenoisingRunStatus.FAILED_FIT.value,
                                "system_id": system.system_id, "warnings": [],
                            }
                        )
                    continue
                state_offset = states.tell()
                descriptors = []
                for role, array in (
                    ("mean", result["mean"]),
                    ("components", result["components"]),
                    ("explained_variance", result["explained_variance"]),
                ):
                    if array is None:
                        continue
                    raw = np.ascontiguousarray(array, dtype="<f8").tobytes()
                    descriptors.append(
                        {
                            "byte_count": len(raw), "offset": states.tell(), "role": role,
                            "sha256": hashlib.sha256(raw).hexdigest(), "shape": list(array.shape),
                        }
                    )
                    states.write(raw)
                fit_warnings = result["warnings"]
                fit_rows.append(
                    {
                        "axis_sha256": result["axis_sha256"],
                        "context_sha256": result["context_sha256"],
                        "diagnostics": {"arrays": descriptors, "n_components": result["n_components"]},
                        "error_code": None, "error_message": None,
                        "family_id": system.family_id,
                        "fit_state_byte_count": states.tell() - state_offset,
                        "fit_state_offset_bytes": state_offset,
                        "fit_state_sha256": result["state_sha256"],
                        "method_id": system.method_id,
                        "representation_id": "bacteria_id_reference_increasing_float64_v1",
                        "split_id": "d1_seed0_train",
                        "status": "complete_with_warning" if fit_warnings else "complete",
                        "system_id": system.system_id,
                        "training_matrix_sha256": result["training_matrix_sha256"],
                        "training_record_ledger_sha256": result["training_record_ledger_sha256"],
                        "warnings": _warnings(fit_warnings),
                    }
                )
                warning_rows.extend(
                    _warning_rows(fit_warnings, "fit", system.system_id, None, "fit")
                )
                for source_index, (source, current) in enumerate(
                    zip(ordered_sources, result["results"], strict=True)
                ):
                    role = "validation" if source_index < len(inputs.validation) else "test"
                    raw = None
                    if current["output"] is not None:
                        raw = np.ascontiguousarray(current["output"], dtype="<f8").tobytes()
                    offset = output.tell() if raw is not None else None
                    if raw is not None:
                        output.write(raw)
                    transform_rows.append(
                        {
                            "axis_sha256": source.axis_sha256, "cohort": "fitted",
                            "diagnostics": current["diagnostics"],
                            "error_code": current["error_code"],
                            "error_message": current["error_message"],
                            "family_id": system.family_id,
                            "fitted_state_sha256": result["state_sha256"],
                            "method_id": system.method_id,
                            "output_byte_count": None if raw is None else len(raw),
                            "output_offset_bytes": offset,
                            "output_sha256": current["output_sha256"],
                            "point_count": source.point_count, "record_id": source.record_id,
                            "source_order": source.source_order, "source_role": role,
                            "status": current["status"], "system_id": system.system_id,
                            "warnings": _warnings(current["warnings"]),
                        }
                    )
                    warning_rows.extend(
                        _warning_rows(current["warnings"], "transform", system.system_id, source.record_id, role)
                    )
        finally:
            fit_executor.shutdown()
    (path / "fit_receipts.jsonl").write_bytes(b"".join(_canonical(value) for value in fit_rows))
    (path / "transform_receipts.jsonl").write_bytes(
        b"".join(_canonical(value) for value in transform_rows)
    )
    (path / "warnings.jsonl").write_bytes(b"".join(_canonical(value) for value in warning_rows))

    fit_by_system = {str(value["system_id"]): value for value in fit_rows}
    transforms_by_system: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for value in transform_rows:
        transforms_by_system[str(value["system_id"])].append(value)
    system_summaries = []
    promoted = []
    successful_by_role: dict[tuple[str, str], set[str]] = {}
    for system in systems:
        rows = transforms_by_system[system.system_id]
        kind = "fitted" if system.family_id in FITTED_FAMILIES else "stateless"
        transform_counts = Counter(str(value["status"]) for value in rows)
        validation_counts = Counter(
            str(value["status"]) for value in rows if value["source_role"] == "validation"
        )
        test_counts = Counter(
            str(value["status"]) for value in rows if value["source_role"] == "test"
        )
        fit_counts = Counter() if kind == "stateless" else Counter(
            [str(fit_by_system[system.system_id]["status"])]
        )
        successful = sum(value in SUCCESS for value in transform_counts.elements())
        coverage_promoted = successful == len(rows) and (
            kind == "stateless" or str(fit_by_system[system.system_id]["status"]) in SUCCESS
        )
        if coverage_promoted:
            promoted.append(system.system_id)
        for role in (("stateless",) if kind == "stateless" else ("validation", "test")):
            successful_by_role[(system.system_id, role)] = {
                str(value["record_id"]) for value in rows
                if value["source_role"] == role and value["status"] in SUCCESS
            }
        system_summaries.append(
            {
                "coverage_promoted": coverage_promoted, "family_id": system.family_id,
                "fit_status_counts": dict(fit_counts), "kind": kind,
                "successful_fraction": successful / len(rows),
                "successful_transform_count": successful, "system_id": system.system_id,
                "test_status_counts": dict(test_counts), "transform_count": len(rows),
                "transform_status_counts": dict(transform_counts),
                "validation_status_counts": dict(validation_counts),
            }
        )
    (path / "system_summary.jsonl").write_bytes(
        b"".join(_canonical(value) for value in system_summaries)
    )
    family_rows = []
    for family in sorted({value.family_id for value in systems}):
        family_systems = [value for value in systems if value.family_id == family]
        count = sum(value.system_id in promoted for value in family_systems)
        family_rows.append(
            {
                "family_id": family, "promoted_system_count": count,
                "qualifying_family": count >= int(policy["minimum_promoted_per_family"]),
                "system_count": len(family_systems),
            }
        )
    (path / "family_summary.jsonl").write_bytes(
        b"".join(_canonical(value) for value in family_rows)
    )
    warning_counter = Counter(
        (str(value["system_id"]), str(value["source_role"]), str(value["category"]))
        for value in warning_rows
    )
    warning_summary = []
    for system in systems:
        roles = ("stateless",) if system.family_id in STATELESS_FAMILIES else ("fit", "validation", "test")
        denominators = {
            "stateless": len(inputs.stateless), "fit": 1,
            "validation": len(inputs.validation), "test": len(inputs.test),
        }
        categories = sorted(
            {category for sid, role, category in warning_counter if sid == system.system_id and role in roles}
        )
        warning_summary.append(
            {
                "family_id": system.family_id,
                "role_receipt_counts": {role: denominators[role] for role in roles},
                "system_id": system.system_id,
                "warning_counts": {
                    role: {
                        category: warning_counter[(system.system_id, role, category)]
                        for category in categories
                        if warning_counter[(system.system_id, role, category)]
                    }
                    for role in roles
                },
                "warning_receipt_fractions": {
                    role: (
                        float(bool(fit_by_system[system.system_id]["warnings"]))
                        if role == "fit"
                        else sum(
                            1 for row in transform_rows
                            if row["system_id"] == system.system_id
                            and row["source_role"] == role and row["warnings"]
                        ) / denominators[role]
                    )
                    for role in roles
                },
            }
        )
    (path / "warning_summary.jsonl").write_bytes(
        b"".join(_canonical(value) for value in warning_summary)
    )
    stateless_promoted = tuple(
        value.system_id for value in stateless_systems if value.system_id in promoted
    )
    fitted_promoted = tuple(
        value.system_id for value in fitted_systems if value.system_id in promoted
    )

    def common(record_ids: Sequence[str], system_ids: Sequence[str], role: str) -> int:
        return sum(
            all(record_id in successful_by_role[(system_id, role)] for system_id in system_ids)
            for record_id in record_ids
        ) if system_ids else 0

    stateless_common = common(
        tuple(value.record_id for value in inputs.stateless), stateless_promoted, "stateless"
    )
    validation_common = common(
        tuple(value.record_id for value in inputs.validation), fitted_promoted, "validation"
    )
    test_common = common(tuple(value.record_id for value in inputs.test), fitted_promoted, "test")
    cohort_summary = {
        "fitted": {
            "fit_source_count": len(inputs.train),
            "promoted_system_count": len(fitted_promoted),
            "test_common_successful_fraction": test_common / len(inputs.test),
            "test_common_successful_record_count": test_common,
            "test_source_count": len(inputs.test),
            "union_common_successful_fraction": (validation_common + test_common) / (len(inputs.validation) + len(inputs.test)),
            "union_common_successful_record_count": validation_common + test_common,
            "validation_common_successful_fraction": validation_common / len(inputs.validation),
            "validation_common_successful_record_count": validation_common,
            "validation_source_count": len(inputs.validation),
        },
        "stateless": {
            "common_successful_fraction": stateless_common / len(inputs.stateless),
            "common_successful_record_count": stateless_common,
            "promoted_system_count": len(stateless_promoted),
            "source_count": len(inputs.stateless),
        },
    }
    (path / "cohort_summary.json").write_bytes(_canonical(cohort_summary))
    qualifying = [
        value["family_id"] for value in family_rows if value["qualifying_family"]
    ]
    promotion = {
        "coverage_promoted_K": len(promoted),
        "fitted_promoted_system_ids": list(fitted_promoted),
        "phase5_eligible_K": "not_evaluated",
        "planned_K": 60, "runnable_K": 60,
        "phase5_power_status": str(config["phase5_power_status"]),
        "promoted_system_ids": sorted(promoted),
        "qualifying_family_ids": qualifying,
        "stateless_promoted_system_ids": list(stateless_promoted),
        "subset_full_tau_status": str(policy["subset_full_tau_status"]),
    }
    (path / "promotion.json").write_bytes(_canonical(promotion))
    is_formal = (
        len(systems) == int(expected["system_count"])
        and len(inputs.stateless) == int(expected["stateless_source_count"])
        and len(inputs.train) == int(expected["fit_source_count"])
        and len(inputs.validation) == int(expected["validation_source_count"])
        and len(inputs.test) == int(expected["test_source_count"])
        and len(fit_rows) == int(expected["fit_receipt_count"])
        and len(transform_rows) == int(expected["total_transform_receipt_count"])
    )
    gate = {
        "formal_execution_complete": is_formal,
        "phase5_power_status": str(config["phase5_power_status"]),
        "strict_promotion": "complete_cartesian_and_100_percent_success",
        "subset_full_tau_status": str(policy["subset_full_tau_status"]),
    }
    (path / "gate.json").write_bytes(_canonical(gate))
    status = "complete" if is_formal else "fixture_complete"
    manifest = {
        **identity, "claim_boundary": str(config["claim_boundary"]),
        "fit_receipt_count": len(fit_rows), "is_formal_run": is_formal,
        "run_id": run_id, "schema_version": "phase3-denoising-v1-formal-artifact-v1",
        "status": status, "system_count": len(systems),
        "transform_receipt_count": len(transform_rows),
        "warning_count": len(warning_rows),
    }
    (path / "manifest.json").write_bytes(_canonical(manifest))
    marker_name = "complete.json" if is_formal else "failed.json"
    (path / marker_name).write_bytes(
        _canonical({"is_formal_run": is_formal, "run_id": run_id, "status": status})
    )
    names = sorted(value.name for value in path.iterdir() if value.is_file())
    (path / "SHA256SUMS").write_text(
        "".join(f"{_sha(path / name)}  {name}\n" for name in names), encoding="utf-8"
    )
    return _Summary(
        path, run_id, status, is_formal, len(systems), len(fit_rows),
        len(transform_rows), len(promoted), len(warning_rows),
    )


def _validate_authority(path: Path, root: Path) -> _Summary:
    checksums = (path / "SHA256SUMS").read_text(encoding="utf-8")
    parsed = []
    for line in checksums.splitlines():
        digest, name = line.split("  ")
        if not (path / name).is_file() or _sha(path / name) != digest:
            raise DenoisingFormalVerificationError(f"checksum mismatch for {name}")
        parsed.append((name, digest))
    if checksums != "".join(f"{digest}  {name}\n" for name, digest in sorted(parsed)):
        raise DenoisingFormalVerificationError("checksum order mismatch")
    actual = {value.name for value in path.iterdir() if value.is_file()}
    expected = {name for name, _ in parsed} | {"SHA256SUMS"}
    if actual != expected:
        raise DenoisingFormalVerificationError("artifact inventory mismatch")
    config, config_raw = _read_config(root)
    if (path / "config.json").read_bytes() != config_raw:
        raise DenoisingFormalVerificationError("artifact config mismatch")
    manifest = json.loads((path / "manifest.json").read_bytes())
    if manifest["code_identity"] != _code_identity(root):
        raise DenoisingFormalVerificationError("artifact code identity mismatch")
    marker_names = {"complete.json", "failed.json"} & actual
    if len(marker_names) != 1:
        raise DenoisingFormalVerificationError("terminal marker mismatch")
    marker = json.loads((path / next(iter(marker_names))).read_bytes())
    if marker["run_id"] != manifest["run_id"]:
        raise DenoisingFormalVerificationError("terminal run identity mismatch")
    _validate_stream_contracts(path)
    promotion = json.loads((path / "promotion.json").read_bytes())
    return _Summary(
        path, str(manifest["run_id"]), str(manifest["status"]),
        bool(manifest["is_formal_run"]), int(manifest["system_count"]),
        int(manifest["fit_receipt_count"]), int(manifest["transform_receipt_count"]),
        len(promotion["promoted_system_ids"]), int(manifest["warning_count"]),
    )


def verify_phase3_denoising_formal(
    path: Path, *, worker_count: int, project_root: Path = ROOT
) -> _Summary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise DenoisingFormalVerificationError("worker_count must be positive")
    root = Path(project_root)
    authority = _validate_authority(Path(path), root)
    if not authority.is_formal_run:
        # Fixture artifacts are structurally checked by the focused test path; they
        # are never publication authorities and therefore are not rebuilt here.
        return authority
    config, config_raw = _read_config(root)
    operational = config["operational"]
    assert isinstance(operational, Mapping)
    if worker_count == int(operational["worker_processes"]):
        raise DenoisingFormalVerificationError(
            "independent worker count must differ from production"
        )
    catalog_value = config["catalog"]
    assert isinstance(catalog_value, Mapping)
    catalog = load_classical_catalog(root / str(catalog_value["path"]), project_root=root)
    systems = tuple(
        sorted(
            (value for value in catalog.systems if value.task_line is TaskLine.DENOISING),
            key=lambda value: value.system_id,
        )
    )
    system_ids_sha256 = hashlib.sha256(
        ("\n".join(value.system_id for value in systems) + "\n").encode("utf-8")
    ).hexdigest()
    if (
        catalog.catalog_id != catalog_value["catalog_id"]
        or system_ids_sha256 != catalog_value["denoising_system_ids_sha256"]
        or len(systems) != 60
    ):
        raise DenoisingFormalVerificationError(
            "catalog denoising view identity mismatch"
        )
    inputs = _load_inputs(config, root)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=".denoising-formal-verify-", dir=Path(path).parent)
    )
    try:
        rebuilt = _build_reconstruction(
            config, config_raw, inputs, systems, temporary_root, worker_count, root
        )
        authority_files = sorted(
            value.relative_to(path).as_posix() for value in Path(path).rglob("*")
            if value.is_file()
        )
        rebuilt_files = sorted(
            value.relative_to(rebuilt.path).as_posix() for value in rebuilt.path.rglob("*")
            if value.is_file()
        )
        if authority_files != rebuilt_files:
            raise DenoisingFormalVerificationError("independent inventory mismatch")
        for name in authority_files:
            first_path, second_path = Path(path) / name, rebuilt.path / name
            if first_path.stat().st_size != second_path.stat().st_size:
                raise DenoisingFormalVerificationError(f"independent size mismatch: {name}")
            with first_path.open("rb") as first, second_path.open("rb") as second:
                while True:
                    left, right = first.read(4 * 1024 * 1024), second.read(4 * 1024 * 1024)
                    if left != right:
                        raise DenoisingFormalVerificationError(
                            f"independent byte mismatch: {name}"
                        )
                    if not left:
                        break
    finally:
        shutil.rmtree(temporary_root)
    return authority


__all__ = ["verify_phase3_denoising_formal"]
