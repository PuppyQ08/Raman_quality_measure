from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import platform
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import h5py
import numpy as np
import scipy

from rpe.downstream.rruff import (
    D5LibraryQuerySplit,
    D5RawCohort,
    load_d5_native_spectra,
    load_d5_raw_cohort,
)
from rpe.downstream.rruff_matching import match_d5_protocol_a_values
from rpe.evaluation import Spectrum1D, SpectrumPairInput, evaluate_metric
from rpe.metrics import MSEMetric, Wasserstein1Metric
from rpe.perturb import (
    P11GlobalWavenumberShift,
    P9GaussianWhiteNoise,
    PerturbationContext,
    PerturbationSweepConfig,
    load_perturbation_sweep_config,
)
from rpe.runner.phase4_d5_canary_authority import CONFIG_BYTES, CONFIG_SHA256


ROOT = Path(__file__).resolve().parents[2]
CONFIG_RELATIVE_PATH = "experiments/phase4/configs/d5_p09_p11_protocol_a_canary_v1.json"
SCHEMA_VERSION = "phase4-d5-p09-p11-protocol-a-canary-config-v1"
EXPERIMENT_ID = "phase4-d5-p09-p11-protocol-a-canary-v1"
ARTIFACT_SCHEMA_VERSION = "phase4-d5-p09-p11-protocol-a-canary-artifact-v1"
RUN_PREFIX = "phase4-d5-p09-p11-canary-"
CANARY_PAYLOAD_FILES = (
    "config.json",
    "sources.jsonl",
    "split.json",
    "library.jsonl",
    "response_records.jsonl",
    "response_curve.csv",
    "manifest.json",
    "complete.json",
)
CODE_RELATIVE_PATHS = (
    "rpe/downstream/rruff.py",
    "rpe/downstream/rruff_matching.py",
    "rpe/metrics/fidelity.py",
    "rpe/metrics/transport.py",
    "rpe/perturb/axis_transform.py",
    "rpe/perturb/gaussian_noise.py",
    "rpe/runner/phase4_d5_canary.py",
    "tools/run_phase4_d5_canary.py",
)
CODE_AUTHORITY_PATHS = CODE_RELATIVE_PATHS
FORBIDDEN_KEYS = {
    "alignment_gap",
    "bootstrap",
    "holm",
    "p_value",
    "permutation",
    "significant",
}
RESPONSE_KEYS = {
    "alpha",
    "alpha_float64_le_hex",
    "axis_changed",
    "cohort_index",
    "diagnostics",
    "downstream_intensity_sha256",
    "group_id",
    "intensity_changed",
    "mineral_name",
    "mse",
    "output_axis_sha256",
    "output_intensity_sha256",
    "perturbation_id",
    "query_order",
    "record_id",
    "rruff_id",
    "state_digest",
    "top1_class_label",
    "top1_correct",
    "top1_score",
    "top5_class_labels",
    "top5_correct",
    "top5_scores",
    "true_class_label",
    "wasserstein_1_cm1",
}
SOURCE_KEYS = {
    "class_label",
    "cohort_index",
    "group_id",
    "mineral_name",
    "native_axis_sha256",
    "native_intensity_sha256",
    "point_count",
    "record_id",
    "role",
    "rruff_id",
}
LIBRARY_KEYS = {
    "cohort_index",
    "downstream_intensity_sha256",
    "library_order",
    "record_id",
}
MANIFEST_KEYS = {
    "artifact_schema_version",
    "claim_boundary",
    "code",
    "config",
    "counts",
    "downstream_grid",
    "environment",
    "experiment_id",
    "library_matrix_sha256",
    "metric_ids",
    "perturbation_ids",
    "protocol",
    "run_id",
    "run_identity",
    "synthetic_fixture",
}


class Phase4D5CanaryError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class Phase4D5CanaryConfig:
    path: Path
    byte_count: int
    sha256: str
    document: Mapping[str, object]
    design_sha256: str
    sweep_sha256: str
    d5_config_sha256: str
    dataset_sha256sums_sha256: str
    seed: int
    split_sha256: str
    cohort_record_count: int
    query_count: int
    library_count: int
    query_record_ids_sha256: str
    library_record_ids_sha256: str
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    metric_ids: tuple[str, ...]
    grid_start_cm1: float
    grid_stop_cm1: float
    grid_step_cm1: float
    grid_point_count: int
    max_in_range_native_gap_cm1: float
    expected_response_count: int
    expected_curve_count: int
    claim_boundary: str
    code_authority: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class Phase4D5CanarySummary:
    path: Path
    run_id: str
    query_count: int
    library_count: int
    response_count: int
    curve_count: int


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


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(array).tobytes(order="C"))


def _ids_digest(values: Sequence[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Phase4D5CanaryError(path, "must be an object")
    return value


def _tuple_strings(path: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise Phase4D5CanaryError(path, "must be a nonempty list")
    if any(not isinstance(item, str) or item == "" for item in value):
        raise Phase4D5CanaryError(path, "must contain nonempty strings")
    return tuple(value)


def _tuple_floats(path: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise Phase4D5CanaryError(path, "must be a nonempty list")
    converted = tuple(float(item) for item in value)
    if any(not math.isfinite(item) for item in converted):
        raise Phase4D5CanaryError(path, "must contain finite values")
    return converted


def _integer(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Phase4D5CanaryError(path, "must be a nonnegative integer")
    return value


def _number(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase4D5CanaryError(path, "must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise Phase4D5CanaryError(path, "must be a finite number")
    return converted


def _config_from_document(
    path: Path,
    raw: bytes,
    document: Mapping[str, object],
) -> Phase4D5CanaryConfig:
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D5CanaryError("config contract", "schema or experiment mismatch")
    design = _object("design", document.get("design"))
    sweep = _object("sweep", document.get("sweep"))
    d5 = _object("d5", document.get("d5"))
    dataset = _object("dataset", document.get("dataset"))
    split = _object("split", document.get("split"))
    downstream = _object("downstream", document.get("downstream"))
    expected = _object("expected", document.get("expected"))
    claim_boundary = document.get("claim_boundary")
    if not isinstance(claim_boundary, str) or claim_boundary == "":
        raise Phase4D5CanaryError("claim_boundary", "must be nonempty")
    return Phase4D5CanaryConfig(
        path=path,
        byte_count=len(raw),
        sha256=_sha256_bytes(raw),
        document=MappingProxyType(dict(document)),
        design_sha256=str(design.get("sha256")),
        sweep_sha256=str(sweep.get("sha256")),
        d5_config_sha256=str(d5.get("sha256")),
        dataset_sha256sums_sha256=str(dataset.get("dataset_sha256sums_sha256")),
        seed=_integer("seed", document.get("seed")),
        split_sha256=str(split.get("split_sha256")),
        cohort_record_count=_integer("dataset.cohort_record_count", dataset.get("cohort_record_count")),
        query_count=_integer("expected.query_count", expected.get("query_count")),
        library_count=_integer("expected.library_count", expected.get("library_count")),
        query_record_ids_sha256=str(split.get("query_record_ids_sha256")),
        library_record_ids_sha256=str(split.get("library_record_ids_sha256")),
        perturbation_ids=_tuple_strings("perturbation_ids", document.get("perturbation_ids")),
        alpha_grid=_tuple_floats("alpha_grid", document.get("alpha_grid")),
        metric_ids=_tuple_strings("metric_ids", document.get("metric_ids")),
        grid_start_cm1=_number("downstream.grid_start_cm1", downstream.get("grid_start_cm1")),
        grid_stop_cm1=_number("downstream.grid_stop_cm1", downstream.get("grid_stop_cm1")),
        grid_step_cm1=_number("downstream.grid_step_cm1", downstream.get("grid_step_cm1")),
        grid_point_count=_integer("downstream.grid_point_count", downstream.get("grid_point_count")),
        max_in_range_native_gap_cm1=_number(
            "downstream.max_in_range_native_gap_cm1",
            downstream.get("max_in_range_native_gap_cm1"),
        ),
        expected_response_count=_integer("expected.response_count", expected.get("response_count")),
        expected_curve_count=_integer("expected.curve_count", expected.get("curve_count")),
        claim_boundary=claim_boundary,
        code_authority=MappingProxyType(
            {
                str(relative): MappingProxyType(dict(_object(f"code_authority.{relative}", identity)))
                for relative, identity in _object(
                    "code_authority",
                    document.get("code_authority"),
                ).items()
            }
        ),
    )


def load_phase4_d5_canary_config(path: Path) -> Phase4D5CanaryConfig:
    path = Path(path)
    try:
        raw = path.read_bytes()
        document = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5CanaryError("config", str(error)) from error
    if raw != _canonical_json_bytes(document):
        raise Phase4D5CanaryError("config noncanonical", "must be canonical JSON")
    if len(raw) != CONFIG_BYTES or _sha256_bytes(raw) != CONFIG_SHA256:
        raise Phase4D5CanaryError("config identity", "bytes or SHA256 mismatch")
    config = _config_from_document(path, raw, _object("config", document))
    expected = (
        config.design_sha256,
        config.sweep_sha256,
        config.d5_config_sha256,
        config.dataset_sha256sums_sha256,
        config.seed,
        config.split_sha256,
        config.cohort_record_count,
        config.query_count,
        config.library_count,
        config.query_record_ids_sha256,
        config.library_record_ids_sha256,
        config.perturbation_ids,
        config.alpha_grid,
        config.metric_ids,
        config.grid_start_cm1,
        config.grid_stop_cm1,
        config.grid_step_cm1,
        config.grid_point_count,
        config.max_in_range_native_gap_cm1,
        config.expected_response_count,
        config.expected_curve_count,
    )
    frozen = (
        "ea098b8a65906391dc1d9e9a25f3e3f055502f03c9e020c19228497e84efa85f",
        "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186",
        "94f59739a2e3583c6ae33ab5f4ab489cb1f26d3c9f8cdeb689a2448c9c01e009",
        "f0cb09af8d80cdf3d52af9ed01bc1ae48abba94fefdff712c30d6e8c8bfe4995",
        0,
        "5d292560b28bf41c207c6fbe88d9e5025c4f03764240a41d6abf0da14138e5e5",
        3770,
        1318,
        2452,
        "59fb906a47ceb996ef9ba29995172c25931a01840bfb9b4a314ed118ee6c556b",
        "3586c18224d625cfccf26b1d56e1e90b879de376d4dbd7455a161805f5bff87c",
        ("p09", "p11"),
        (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8),
        ("mse", "wasserstein_1_cm1"),
        204.0,
        1800.0,
        2.0,
        799,
        3.0,
        23724,
        18,
    )
    if expected != frozen:
        raise Phase4D5CanaryError("config scientific identity", "frozen values mismatch")
    return config


def _code_document() -> dict[str, object]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha256_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _validate_code_authority(config: Phase4D5CanaryConfig) -> dict[str, object]:
    if tuple(config.code_authority) != CODE_AUTHORITY_PATHS:
        raise Phase4D5CanaryError(
            "code authority paths",
            "must equal the frozen ordered scientific code paths",
        )
    observed = _code_document()
    expected = {
        relative: dict(config.code_authority[relative])
        for relative in config.code_authority
    }
    if observed != expected:
        raise Phase4D5CanaryError(
            "code authority",
            "current bytes or SHA256 do not match the frozen config",
        )
    return observed


def _environment_document() -> dict[str, object]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scipy": scipy.__version__,
    }


def _find_split(cohort: D5RawCohort, seed: int) -> D5LibraryQuerySplit:
    matches = [split for split in cohort.splits if split.seed == seed]
    if len(matches) != 1:
        raise Phase4D5CanaryError("split", "seed must identify exactly one split")
    return matches[0]


def _grid(config: Phase4D5CanaryConfig) -> np.ndarray:
    values = np.arange(
        config.grid_start_cm1,
        config.grid_stop_cm1 + config.grid_step_cm1 / 2.0,
        config.grid_step_cm1,
        dtype="<f8",
    )
    if values.size != config.grid_point_count:
        raise Phase4D5CanaryError("downstream grid", "point count mismatch")
    return values


def _project_spectrum(
    spectrum: Spectrum1D,
    grid: np.ndarray,
    max_gap: float,
) -> np.ndarray:
    axis = spectrum.axis_cm1
    if float(axis[0]) > float(grid[0]) or float(axis[-1]) < float(grid[-1]):
        raise Phase4D5CanaryError(
            "downstream support",
            f"{spectrum.spectrum_id} would require extrapolation",
        )
    left = int(np.searchsorted(axis, grid[0], side="right") - 1)
    right = int(np.searchsorted(axis, grid[-1], side="left"))
    if left < 0 or right >= axis.size:
        raise Phase4D5CanaryError("downstream support", "invalid interpolation bounds")
    support = axis[left : right + 1]
    if support.size < 2 or float(np.max(np.diff(support))) > max_gap:
        raise Phase4D5CanaryError("downstream support", "native gap exceeds frozen maximum")
    values = np.asarray(np.interp(grid, axis, spectrum.intensity), dtype="<f4")
    if not np.isfinite(values).all():
        raise Phase4D5CanaryError("downstream projection", "contains nonfinite values")
    return values


def _metric_value(metric: object, source: Spectrum1D, candidate: Spectrum1D) -> float:
    result = evaluate_metric(metric, SpectrumPairInput(reference=source, candidate=candidate))
    output = result.outputs[0]
    return float(output.value)


def _class_macro(true_labels: np.ndarray, correct: np.ndarray) -> float:
    return float(
        np.mean(
            [
                np.mean(correct[true_labels == label])
                for label in np.unique(true_labels)
            ],
            dtype=np.float64,
        )
    )


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _curve_bytes(rows: Sequence[Mapping[str, object]]) -> bytes:
    fieldnames = (
        "perturbation_id",
        "alpha",
        "query_count",
        "mean_mse",
        "median_mse",
        "mean_wasserstein_1_cm1",
        "median_wasserstein_1_cm1",
        "top1_macro_class_accuracy",
        "top5_macro_class_accuracy",
        "axis_changed_count",
        "intensity_changed_count",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for perturbation_id in ("p09", "p11"):
        alphas = sorted(
            {float(row["alpha"]) for row in rows if row["perturbation_id"] == perturbation_id}
        )
        for alpha in alphas:
            selected = [
                row
                for row in rows
                if row["perturbation_id"] == perturbation_id and float(row["alpha"]) == alpha
            ]
            true_labels = np.asarray([row["true_class_label"] for row in selected], dtype="<i8")
            top1 = np.asarray([row["top1_correct"] for row in selected], dtype=np.float64)
            top5 = np.asarray([row["top5_correct"] for row in selected], dtype=np.float64)
            mse = np.asarray([row["mse"] for row in selected], dtype=np.float64)
            w1 = np.asarray([row["wasserstein_1_cm1"] for row in selected], dtype=np.float64)
            writer.writerow(
                {
                    "perturbation_id": perturbation_id,
                    "alpha": repr(alpha),
                    "query_count": len(selected),
                    "mean_mse": repr(float(np.mean(mse))),
                    "median_mse": repr(float(np.median(mse))),
                    "mean_wasserstein_1_cm1": repr(float(np.mean(w1))),
                    "median_wasserstein_1_cm1": repr(float(np.median(w1))),
                    "top1_macro_class_accuracy": repr(_class_macro(true_labels, top1)),
                    "top5_macro_class_accuracy": repr(_class_macro(true_labels, top5)),
                    "axis_changed_count": sum(bool(row["axis_changed"]) for row in selected),
                    "intensity_changed_count": sum(bool(row["intensity_changed"]) for row in selected),
                }
            )
    return stream.getvalue().encode("utf-8")


def _run_identity(config: Phase4D5CanaryConfig, code: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    identity = {
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config_sha256": config.sha256,
        "dataset_sha256sums_sha256": config.dataset_sha256sums_sha256,
        "design_sha256": config.design_sha256,
        "environment": _environment_document(),
        "split_sha256": config.split_sha256,
        "sweep_sha256": config.sweep_sha256,
    }
    return RUN_PREFIX + _sha256_bytes(_canonical_json_bytes(identity)), identity


def _validate_inputs(
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    config: Phase4D5CanaryConfig,
) -> D5LibraryQuerySplit:
    if len(cohort.record_ids) != config.cohort_record_count:
        raise Phase4D5CanaryError("cohort count", "does not match config")
    if len(native_spectra) != len(cohort.record_ids):
        raise Phase4D5CanaryError("native spectra", "must align with the cohort")
    expected_ids = tuple(f"rruff_raman_raw::{record_id}" for record_id in cohort.record_ids)
    if tuple(spectrum.spectrum_id for spectrum in native_spectra) != expected_ids:
        raise Phase4D5CanaryError("native spectra identity", "record order mismatch")
    if sweep.sha256 != config.sweep_sha256 or tuple(sweep.alpha_grid) != config.alpha_grid:
        raise Phase4D5CanaryError("sweep identity", "config or alpha grid mismatch")
    split = _find_split(cohort, config.seed)
    if split.split_sha256 != config.split_sha256:
        raise Phase4D5CanaryError("split identity", "SHA256 mismatch")
    query_ids = tuple(cohort.record_ids[int(index)] for index in split.query_indices)
    library_ids = tuple(cohort.record_ids[int(index)] for index in split.library_indices)
    if (
        len(query_ids) != config.query_count
        or len(library_ids) != config.library_count
        or _ids_digest(query_ids) != config.query_record_ids_sha256
        or _ids_digest(library_ids) != config.library_record_ids_sha256
    ):
        raise Phase4D5CanaryError("split records", "counts or digests mismatch")
    return split


def build_phase4_d5_canary_from_inputs(
    output_dir: Path,
    *,
    cohort: D5RawCohort,
    native_spectra: tuple[Spectrum1D, ...],
    sweep: PerturbationSweepConfig,
    config: Phase4D5CanaryConfig,
) -> Phase4D5CanarySummary:
    split = _validate_inputs(cohort, native_spectra, sweep, config)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise Phase4D5CanaryError("output_dir", "must not already exist")
    output_dir.mkdir(parents=True)
    grid = _grid(config)
    code = _validate_code_authority(config)
    run_id, run_identity = _run_identity(config, code)
    query_indices = tuple(int(index) for index in split.query_indices)
    library_indices = tuple(int(index) for index in split.library_indices)

    library_matrix = np.stack(
        [
            _project_spectrum(
                native_spectra[index],
                grid,
                config.max_in_range_native_gap_cm1,
            )
            for index in library_indices
        ]
    ).astype("<f4", copy=False)
    context = PerturbationContext(
        sweep_id=sweep.sweep_id,
        sweep_config_sha256=sweep.sha256,
        global_seed=sweep.global_seed,
    )
    operators = {
        "p09": P9GaussianWhiteNoise(sweep),
        "p11": P11GlobalWavenumberShift(sweep),
    }
    states = {
        perturbation_id: tuple(
            operator.prepare(native_spectra[index], context)
            for index in query_indices
        )
        for perturbation_id, operator in operators.items()
    }
    metric_rows: dict[tuple[int, str, float], dict[str, object]] = {}
    matching_rows: dict[tuple[int, str, float], dict[str, object]] = {}
    mse_metric = MSEMetric()
    w1_metric = Wasserstein1Metric()

    for perturbation_id in config.perturbation_ids:
        operator = operators[perturbation_id]
        for alpha in config.alpha_grid:
            projected = []
            condition_results = []
            for query_order, (cohort_index, state) in enumerate(
                zip(query_indices, states[perturbation_id], strict=True)
            ):
                source = native_spectra[cohort_index]
                result = operator.apply(source, alpha, state)
                output = result.output
                downstream = _project_spectrum(
                    output,
                    grid,
                    config.max_in_range_native_gap_cm1,
                )
                projected.append(downstream)
                metric_rows[(query_order, perturbation_id, alpha)] = {
                    "alpha": float(alpha),
                    "alpha_float64_le_hex": struct.pack("<d", float(alpha)).hex(),
                    "axis_changed": result.axis_changed,
                    "diagnostics": _jsonable(result.diagnostics),
                    "downstream_intensity_sha256": _array_sha256(downstream),
                    "intensity_changed": result.intensity_changed,
                    "mse": _metric_value(mse_metric, source, output),
                    "output_axis_sha256": _array_sha256(output.axis_cm1),
                    "output_intensity_sha256": _array_sha256(output.intensity),
                    "perturbation_id": perturbation_id,
                    "state_digest": result.state_digest,
                    "wasserstein_1_cm1": _metric_value(w1_metric, source, output),
                }
                condition_results.append(result)
            query_matrix = np.stack(projected).astype("<f4", copy=False)
            condition_id = f"{perturbation_id}:{struct.pack('<d', float(alpha)).hex()}"
            matching = match_d5_protocol_a_values(
                cohort,
                split,
                condition_id=condition_id,
                query_record_ids=tuple(
                    cohort.record_ids[index] for index in query_indices
                ),
                library_record_ids=tuple(
                    cohort.record_ids[index] for index in library_indices
                ),
                query_values=query_matrix,
                library_values=library_matrix,
            )
            for query_order in range(len(query_indices)):
                top_k = min(5, matching.ranked_class_labels.shape[1])
                matching_rows[(query_order, perturbation_id, alpha)] = {
                    "top1_class_label": int(matching.ranked_class_labels[query_order, 0]),
                    "top1_correct": bool(matching.top1_correct[query_order]),
                    "top1_score": float(matching.ranked_class_scores[query_order, 0]),
                    "top5_class_labels": [
                        int(value)
                        for value in matching.ranked_class_labels[query_order, :top_k]
                    ],
                    "top5_correct": bool(matching.top5_correct[query_order]),
                    "top5_scores": [
                        float(value)
                        for value in matching.ranked_class_scores[query_order, :top_k]
                    ],
                }

    response_rows: list[dict[str, object]] = []
    for query_order, cohort_index in enumerate(query_indices):
        for perturbation_id in config.perturbation_ids:
            for alpha in config.alpha_grid:
                row = {
                    "cohort_index": cohort_index,
                    "group_id": cohort.group_ids[cohort_index],
                    "mineral_name": cohort.mineral_names[cohort_index],
                    "query_order": query_order,
                    "record_id": cohort.record_ids[cohort_index],
                    "rruff_id": cohort.rruff_ids[cohort_index],
                    "true_class_label": int(cohort.class_labels[cohort_index]),
                    **metric_rows[(query_order, perturbation_id, alpha)],
                    **matching_rows[(query_order, perturbation_id, alpha)],
                }
                response_rows.append(row)
    if len(response_rows) != config.expected_response_count:
        raise Phase4D5CanaryError("response count", "does not match config")

    roles = {
        index: "query" if index in set(query_indices) else "library"
        for index in range(len(cohort.record_ids))
    }
    source_rows = []
    for index, spectrum in enumerate(native_spectra):
        source_rows.append(
            {
                "class_label": int(cohort.class_labels[index]),
                "cohort_index": index,
                "group_id": cohort.group_ids[index],
                "mineral_name": cohort.mineral_names[index],
                "native_axis_sha256": _array_sha256(spectrum.axis_cm1),
                "native_intensity_sha256": _array_sha256(spectrum.intensity),
                "point_count": int(spectrum.axis_cm1.size),
                "record_id": cohort.record_ids[index],
                "role": roles[index],
                "rruff_id": cohort.rruff_ids[index],
            }
        )
    library_rows = [
        {
            "cohort_index": cohort_index,
            "downstream_intensity_sha256": _array_sha256(library_matrix[order]),
            "library_order": order,
            "record_id": cohort.record_ids[cohort_index],
        }
        for order, cohort_index in enumerate(library_indices)
    ]
    split_document = {
        "library_count": len(library_indices),
        "library_record_ids_sha256": _ids_digest(
            tuple(cohort.record_ids[index] for index in library_indices)
        ),
        "query_count": len(query_indices),
        "query_record_ids_sha256": _ids_digest(
            tuple(cohort.record_ids[index] for index in query_indices)
        ),
        "seed": split.seed,
        "split_sha256": split.split_sha256,
    }
    config_bytes = _canonical_json_bytes(dict(config.document))
    source_bytes = b"".join(_canonical_json_bytes(row) for row in source_rows)
    library_bytes = b"".join(_canonical_json_bytes(row) for row in library_rows)
    response_bytes = b"".join(_canonical_json_bytes(row) for row in response_rows)
    curve_bytes = _curve_bytes(response_rows)
    curve_count = curve_bytes.count(b"\n") - 1
    if curve_count != config.expected_curve_count:
        raise Phase4D5CanaryError("curve count", "does not match config")
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "claim_boundary": config.claim_boundary,
        "code": code,
        "config": {"bytes": len(config_bytes), "sha256": _sha256_bytes(config_bytes)},
        "counts": {
            "curve": curve_count,
            "library": len(library_indices),
            "query": len(query_indices),
            "response": len(response_rows),
            "source": len(source_rows),
        },
        "downstream_grid": {
            "interpolation": "linear_no_extrapolation",
            "point_count": int(grid.size),
            "start_cm1": float(grid[0]),
            "step_cm1": config.grid_step_cm1,
            "stop_cm1": float(grid[-1]),
        },
        "environment": _environment_document(),
        "experiment_id": EXPERIMENT_ID,
        "library_matrix_sha256": _array_sha256(library_matrix),
        "metric_ids": list(config.metric_ids),
        "perturbation_ids": list(config.perturbation_ids),
        "protocol": "A",
        "run_id": run_id,
        "run_identity": run_identity,
        "synthetic_fixture": bool(config.document.get("synthetic_fixture", False)),
    }
    complete = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
    }
    payloads = {
        "config.json": config_bytes,
        "sources.jsonl": source_bytes,
        "split.json": _canonical_json_bytes(split_document),
        "library.jsonl": library_bytes,
        "response_records.jsonl": response_bytes,
        "response_curve.csv": curve_bytes,
        "manifest.json": _canonical_json_bytes(manifest),
        "complete.json": _canonical_json_bytes(complete),
    }
    for name in CANARY_PAYLOAD_FILES:
        (output_dir / name).write_bytes(payloads[name])
    checksum_bytes = "".join(
        f"{_sha256_bytes(payloads[name])}  {name}\n"
        for name in CANARY_PAYLOAD_FILES
    ).encode("utf-8")
    (output_dir / "SHA256SUMS").write_bytes(checksum_bytes)
    return Phase4D5CanarySummary(
        path=output_dir,
        run_id=run_id,
        query_count=len(query_indices),
        library_count=len(library_indices),
        response_count=len(response_rows),
        curve_count=curve_count,
    )


def build_phase4_d5_canary(output_root: Path) -> Phase4D5CanarySummary:
    config = load_phase4_d5_canary_config(ROOT / CONFIG_RELATIVE_PATH)
    d5_path = ROOT / "experiments/phase05/configs/d5_rruff_protocol.json"
    dataset_path = ROOT / "data/unified/rruff_raman_raw"
    sweep = load_perturbation_sweep_config(
        ROOT / "experiments/shared/raman_perturbation_sweep_v1.json"
    )
    cohort = load_d5_raw_cohort(d5_path, dataset_path)
    native = load_d5_native_spectra(dataset_path, cohort.record_ids)
    run_id, _ = _run_identity(config, _code_document())
    return build_phase4_d5_canary_from_inputs(
        Path(output_root) / run_id,
        cohort=cohort,
        native_spectra=native,
        sweep=sweep,
        config=config,
    )


def _read_json(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase4D5CanaryError(path.name, str(error)) from error
    if raw != _canonical_json_bytes(value):
        raise Phase4D5CanaryError(path.name, "must be canonical JSON")
    return _object(path.name, value)


def _read_jsonl(path: Path) -> list[Mapping[str, object]]:
    rows = []
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise Phase4D5CanaryError(f"{path.name}[{index}]", str(error)) from error
        if line != _canonical_json_bytes(value):
            raise Phase4D5CanaryError(f"{path.name}[{index}]", "must be canonical JSON")
        rows.append(_object(f"{path.name}[{index}]", value))
    return rows


def _recursive_keys(value: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys


def _require_exact_keys(
    path: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    observed = set(value)
    if observed != expected:
        raise Phase4D5CanaryError(
            f"{path} schema",
            f"missing={sorted(expected-observed)!r}; extra={sorted(observed-expected)!r}",
        )


def _validate_artifact_authority(
    config_document: Mapping[str, object],
    manifest: Mapping[str, object],
) -> None:
    synthetic = bool(manifest.get("synthetic_fixture"))
    if not synthetic:
        authoritative = load_phase4_d5_canary_config(ROOT / CONFIG_RELATIVE_PATH)
        if _canonical_json_bytes(config_document) != (ROOT / CONFIG_RELATIVE_PATH).read_bytes():
            raise Phase4D5CanaryError(
                "config authority",
                "artifact config differs from the frozen repository config",
            )
        expected_code = _validate_code_authority(authoritative)
    else:
        code_authority = _object("code_authority", config_document.get("code_authority"))
        expected_code = {str(key): dict(_object(str(key), value)) for key, value in code_authority.items()}
    if manifest.get("code") != expected_code:
        raise Phase4D5CanaryError(
            "manifest code authority",
            "does not match the frozen config/current code",
        )
    if manifest.get("environment") != _environment_document():
        raise Phase4D5CanaryError(
            "environment authority",
            "manifest environment differs from current scientific environment",
        )
    if manifest.get("claim_boundary") != config_document.get("claim_boundary"):
        raise Phase4D5CanaryError(
            "claim boundary",
            "manifest differs from config",
        )
    if manifest.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise Phase4D5CanaryError(
            "artifact schema version",
            "does not match the frozen version",
        )
    if manifest.get("experiment_id") != EXPERIMENT_ID or manifest.get("protocol") != "A":
        raise Phase4D5CanaryError(
            "experiment authority",
            "experiment or protocol mismatch",
        )


def _validate_response_science(row: Mapping[str, object]) -> None:
    _require_exact_keys("response row", row, RESPONSE_KEYS)
    alpha = float(row["alpha"])
    if row["alpha_float64_le_hex"] != struct.pack("<d", alpha).hex():
        raise Phase4D5CanaryError("alpha encoding", "does not match float64 bytes")
    perturbation_id = str(row["perturbation_id"])
    diagnostics = _object("response diagnostics", row["diagnostics"])
    mse = float(row["mse"])
    w1 = float(row["wasserstein_1_cm1"])
    if not all(math.isfinite(value) and value >= 0.0 for value in (mse, w1)):
        raise Phase4D5CanaryError("response metric", "must be finite and nonnegative")
    if perturbation_id == "p09":
        signal_rms = float(diagnostics.get("signal_rms", math.nan))
        sigma = float(diagnostics.get("sigma", math.nan))
        noise_mean_square = float(diagnostics.get("noise_mean_square", math.nan))
        if not math.isclose(sigma, alpha * signal_rms, rel_tol=1e-15, abs_tol=1e-15):
            raise Phase4D5CanaryError("P9 sigma formula", "sigma != alpha * signal_rms")
        expected_mse = sigma * sigma * noise_mean_square
        if not math.isclose(mse, expected_mse, rel_tol=2e-13, abs_tol=1e-12):
            raise Phase4D5CanaryError("P9 MSE formula", "mse != sigma^2 * noise mean square")
        if bool(row["axis_changed"]) or bool(row["intensity_changed"]) is not (alpha > 0.0 and sigma != 0.0):
            raise Phase4D5CanaryError("P9 change flags", "do not match alpha/sigma")
    elif perturbation_id == "p11":
        requested = float(diagnostics.get("requested_max_abs_offset_cm1", math.nan))
        realized = float(diagnostics.get("realized_max_abs_offset_cm1", math.nan))
        expected_shift = 4.0 * alpha
        if not math.isclose(requested, expected_shift, rel_tol=0.0, abs_tol=1e-15):
            raise Phase4D5CanaryError("P11 shift formula", "requested shift != 4 * alpha")
        if not math.isclose(realized, expected_shift, rel_tol=0.0, abs_tol=1e-9):
            raise Phase4D5CanaryError("P11 shift formula", "realized shift != 4 * alpha")
        if not math.isclose(w1, expected_shift, rel_tol=0.0, abs_tol=1e-9):
            raise Phase4D5CanaryError("P11 W1 formula", "W1 != physical shift")
        if mse != 0.0 or bool(row["intensity_changed"]) or bool(row["axis_changed"]) is not (alpha > 0.0):
            raise Phase4D5CanaryError("P11 change semantics", "MSE/axis/intensity contract failed")
        if diagnostics.get("transform") != "global_additive_shift" or diagnostics.get("interpolation") != "none":
            raise Phase4D5CanaryError("P11 diagnostics", "transform/interpolation mismatch")
    else:
        raise Phase4D5CanaryError("perturbation_id", "must be p09 or p11")


def verify_phase4_d5_canary(
    path: Path,
    *,
    reexecute: bool = True,
) -> Mapping[str, object]:
    path = Path(path)
    expected_files = set(CANARY_PAYLOAD_FILES) | {"SHA256SUMS"}
    observed_files = {item.name for item in path.iterdir() if item.is_file()}
    if observed_files != expected_files:
        raise Phase4D5CanaryError("file set", "missing or extra files")
    checksum_lines = (path / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    if len(checksum_lines) != len(CANARY_PAYLOAD_FILES):
        raise Phase4D5CanaryError("checksum", "line count mismatch")
    for line, name in zip(checksum_lines, CANARY_PAYLOAD_FILES, strict=True):
        expected = f"{_sha256_file(path / name)}  {name}"
        if line != expected:
            raise Phase4D5CanaryError("checksum", f"mismatch for {name}")

    config_document = _read_json(path / "config.json")
    manifest = _read_json(path / "manifest.json")
    complete = _read_json(path / "complete.json")
    split = _read_json(path / "split.json")
    sources = _read_jsonl(path / "sources.jsonl")
    library = _read_jsonl(path / "library.jsonl")
    rows = _read_jsonl(path / "response_records.jsonl")
    _require_exact_keys("manifest", manifest, MANIFEST_KEYS)
    _require_exact_keys(
        "split",
        split,
        {
            "library_count",
            "library_record_ids_sha256",
            "query_count",
            "query_record_ids_sha256",
            "seed",
            "split_sha256",
        },
    )
    for source in sources:
        _require_exact_keys("source row", source, SOURCE_KEYS)
    for entry in library:
        _require_exact_keys("library row", entry, LIBRARY_KEYS)
    for row in rows:
        _validate_response_science(row)
    all_payloads: list[object] = [config_document, manifest, complete, split, sources, library, rows]
    if set().union(*(_recursive_keys(item) for item in all_payloads)) & FORBIDDEN_KEYS:
        raise Phase4D5CanaryError("artifact claim boundary", "contains inference keys")
    _validate_artifact_authority(config_document, manifest)
    counts = _object("manifest.counts", manifest.get("counts"))
    if (
        int(counts.get("source", -1)) != len(sources)
        or int(counts.get("library", -1)) != len(library)
        or int(counts.get("response", -1)) != len(rows)
        or complete.get("status") != "complete"
        or complete.get("run_id") != manifest.get("run_id")
    ):
        raise Phase4D5CanaryError("manifest counts", "do not match payloads")
    perturbation_ids = tuple(config_document.get("perturbation_ids", ()))
    alpha_grid = tuple(float(value) for value in config_document.get("alpha_grid", ()))
    query_count = int(counts["query"])
    expected_order = [
        (query, perturbation_id, alpha)
        for query in range(query_count)
        for perturbation_id in perturbation_ids
        for alpha in alpha_grid
    ]
    observed_order = [
        (int(row["query_order"]), str(row["perturbation_id"]), float(row["alpha"]))
        for row in rows
    ]
    if observed_order != expected_order:
        raise Phase4D5CanaryError("response order", "does not match canonical grid")
    by_key = {
        (int(row["query_order"]), str(row["perturbation_id"]), float(row["alpha"])): row
        for row in rows
    }
    for query in range(query_count):
        p09_mse = [float(by_key[(query, "p09", alpha)]["mse"]) for alpha in alpha_grid]
        if p09_mse[0] != 0.0 or p09_mse != sorted(p09_mse) or any(value <= 0.0 for value in p09_mse[1:]):
            raise Phase4D5CanaryError("P9 MSE", "identity/positive monotonicity failed")
        for alpha in alpha_grid:
            p11 = by_key[(query, "p11", alpha)]
            if float(p11["mse"]) != 0.0:
                raise Phase4D5CanaryError("P11 MSE", "must be exactly zero")
            if alpha > 0.0 and (
                not bool(p11["axis_changed"])
                or bool(p11["intensity_changed"])
                or float(p11["wasserstein_1_cm1"]) <= 0.0
            ):
                raise Phase4D5CanaryError("P11 semantics", "axis/W1 contract failed")
        left = by_key[(query, "p09", 0.0)]
        right = by_key[(query, "p11", 0.0)]
        for key in (
            "downstream_intensity_sha256",
            "top1_class_label",
            "top1_score",
            "top1_correct",
            "top5_class_labels",
            "top5_scores",
            "top5_correct",
        ):
            if left[key] != right[key]:
                raise Phase4D5CanaryError("alpha-zero collapse", f"mismatch for {key}")
    if (path / "response_curve.csv").read_bytes() != _curve_bytes(rows):
        raise Phase4D5CanaryError("curve projection", "does not match response rows")
    run_identity = _object("run_identity", manifest.get("run_identity"))
    if run_identity.get("environment") != manifest.get("environment"):
        raise Phase4D5CanaryError("run identity environment", "does not match manifest")
    expected_run_id = RUN_PREFIX + _sha256_bytes(_canonical_json_bytes(run_identity))
    if manifest.get("run_id") != expected_run_id:
        raise Phase4D5CanaryError("run_id", "does not match run identity")
    if reexecute:
        if bool(manifest.get("synthetic_fixture")):
            raise Phase4D5CanaryError("reexecute", "synthetic input is unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            rebuilt = build_phase4_d5_canary(Path(temporary)).path
            for name in (*CANARY_PAYLOAD_FILES, "SHA256SUMS"):
                if (path / name).read_bytes() != (rebuilt / name).read_bytes():
                    raise Phase4D5CanaryError("reexecute", f"byte mismatch for {name}")
    return MappingProxyType(
        {
            "curve_count": int(counts["curve"]),
            "library_count": int(counts["library"]),
            "query_count": query_count,
            "response_count": len(rows),
            "run_id": str(manifest["run_id"]),
            "status": "passed",
        }
    )


__all__ = [
    "CANARY_PAYLOAD_FILES",
    "Phase4D5CanaryConfig",
    "Phase4D5CanaryError",
    "Phase4D5CanarySummary",
    "build_phase4_d5_canary",
    "build_phase4_d5_canary_from_inputs",
    "load_phase4_d5_canary_config",
    "verify_phase4_d5_canary",
]
