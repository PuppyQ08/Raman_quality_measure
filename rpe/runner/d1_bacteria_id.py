from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import h5py
import numpy as np
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)

from rpe.downstream.bacteria_id import BacteriaIdBatchLoader
from rpe.methods.classical.savitzky_golay import (
    SavitzkyGolayPipeline,
    load_savitzky_golay_pipeline,
)


SCHEMA_VERSION = "phase05-d1-runner-v1"
RESULT_SCHEMA_VERSION = "phase05-d1-result-v1"
EXPERIMENT_ID = "d1_bacteria_id_pca20_lr_sg11"
RUNNER_CONFIG_SHA256 = (
    "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"
)
RUNNER_CONFIG_BYTES = 1051
RETAINED_SNAPSHOT_SHA256 = (
    "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02"
)
EXPECTED_CONDITIONS = (
    ("released_input_control", None),
    ("released_input_plus_sg", "d1_sg11_poly3_interp.json"),
)
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
C_GRID = (0.01, 0.1, 1.0, 10.0)
METRICS = (
    "accuracy",
    "macro_f1",
    "balanced_accuracy",
    "per_class_recall",
    "confusion_matrix",
)
COSINE_QUANTILES = (0.5, 0.9, 0.95, 0.99, 1.0)
ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "rpe/downstream/bacteria_id.py",
    "rpe/io/schema.py",
    "rpe/io/store.py",
    "rpe/methods/classical/savitzky_golay.py",
    "rpe/runner/d1_bacteria_id.py",
    "tools/run_phase05.py",
)
THREAD_ENVIRONMENT_KEYS = (
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class D1RunnerValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


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


def _reject_nonfinite(value: str) -> None:
    raise D1RunnerValidationError(
        "config nonfinite",
        f"unsupported JSON constant {value!r}",
    )


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D1RunnerValidationError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise D1RunnerValidationError(path, "keys must be strings")
    return value


def _exact_keys(
    path: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        raise D1RunnerValidationError(
            path,
            (
                f"key mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
        )


def _equal(path: str, observed: object, expected: object) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise D1RunnerValidationError(
            path,
            f"must equal {expected!r}",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class D1ConditionConfig:
    condition_id: str
    pipeline_config: str | None


@dataclass(frozen=True)
class D1RunnerConfig:
    path: Path
    sha256: str
    byte_count: int
    experiment_id: str
    batch_size: int
    expected_class_count: int
    validation_per_class: int
    conditions: tuple[D1ConditionConfig, ...]
    seeds: tuple[int, ...]
    pca_n_components: int
    pca_solver: str
    pca_whiten: bool
    c_grid: tuple[float, ...]
    max_iter: int
    tol: float
    l1_ratio: float
    cosine_quantiles: tuple[float, ...]
    test_chunk_size: int
    train_chunk_size: int


def load_d1_runner_config(path: Path) -> D1RunnerConfig:
    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    except D1RunnerValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D1RunnerValidationError(
            path.name or "config",
            str(error),
        ) from error
    document = _object("config", value)
    if raw != _canonical_json_bytes(document):
        raise D1RunnerValidationError(
            "config noncanonical",
            "must use canonical JSON encoding",
        )
    digest = hashlib.sha256(raw).hexdigest()
    _exact_keys(
        "config",
        document,
        {
            "schema_version",
            "experiment_id",
            "dataset",
            "conditions",
            "seeds",
            "pca",
            "classifier",
            "leakage_audit",
            "metrics",
        },
    )
    _equal("schema_version", document["schema_version"], SCHEMA_VERSION)
    _equal("experiment_id", document["experiment_id"], EXPERIMENT_ID)

    dataset = _object("dataset", document["dataset"])
    _exact_keys(
        "dataset",
        dataset,
        {
            "dataset_id",
            "batch_size",
            "expected_class_count",
            "retained_snapshot_sha256",
            "train_splits",
            "validation_split",
            "validation_per_class",
            "test_split",
        },
    )
    for key, expected in (
        ("dataset_id", "bacteria_id_reference"),
        ("batch_size", 4096),
        ("expected_class_count", 30),
        ("retained_snapshot_sha256", RETAINED_SNAPSHOT_SHA256),
        ("train_splits", ["reference", "finetune"]),
        ("validation_split", "finetune"),
        ("validation_per_class", 10),
        ("test_split", "test"),
    ):
        _equal(f"dataset.{key}", dataset[key], expected)

    raw_conditions = document["conditions"]
    if not isinstance(raw_conditions, list) or len(raw_conditions) != 2:
        raise D1RunnerValidationError(
            "conditions",
            "must contain exactly two conditions",
        )
    conditions = []
    for index, expected in enumerate(EXPECTED_CONDITIONS):
        condition = _object(f"conditions[{index}]", raw_conditions[index])
        _exact_keys(
            f"conditions[{index}]",
            condition,
            {"condition_id", "pipeline_config"},
        )
        _equal(
            f"conditions[{index}].condition_id",
            condition["condition_id"],
            expected[0],
        )
        _equal(
            f"conditions[{index}].pipeline_config",
            condition["pipeline_config"],
            expected[1],
        )
        conditions.append(
            D1ConditionConfig(
                condition_id=expected[0],
                pipeline_config=expected[1],
            )
        )
    _equal("seeds", document["seeds"], list(EXPECTED_SEEDS))

    pca = _object("pca", document["pca"])
    _exact_keys(
        "pca",
        pca,
        {"n_components", "svd_solver", "whiten"},
    )
    for key, expected in (
        ("n_components", 20),
        ("svd_solver", "randomized"),
        ("whiten", False),
    ):
        _equal(f"pca.{key}", pca[key], expected)

    classifier = _object("classifier", document["classifier"])
    _exact_keys(
        "classifier",
        classifier,
        {
            "type",
            "solver",
            "regularization",
            "l1_ratio",
            "c_grid",
            "max_iter",
            "tol",
            "class_weight",
            "refit_with_validation",
        },
    )
    for key, expected in (
        ("type", "logistic_regression"),
        ("solver", "lbfgs"),
        ("regularization", "l2"),
        ("l1_ratio", 0.0),
        ("c_grid", list(C_GRID)),
        ("max_iter", 1000),
        ("tol", 0.0001),
        ("class_weight", None),
        ("refit_with_validation", False),
    ):
        _equal(f"classifier.{key}", classifier[key], expected)

    audit = _object("leakage_audit", document["leakage_audit"])
    _exact_keys(
        "leakage_audit",
        audit,
        {"cosine_quantiles", "test_chunk_size", "train_chunk_size"},
    )
    for key, expected in (
        ("cosine_quantiles", list(COSINE_QUANTILES)),
        ("test_chunk_size", 64),
        ("train_chunk_size", 4096),
    ):
        _equal(f"leakage_audit.{key}", audit[key], expected)
    _equal("metrics", document["metrics"], list(METRICS))
    if len(raw) != RUNNER_CONFIG_BYTES:
        raise D1RunnerValidationError(
            "config bytes",
            f"must equal {RUNNER_CONFIG_BYTES}",
        )
    if digest != RUNNER_CONFIG_SHA256:
        raise D1RunnerValidationError(
            "config sha256",
            f"must equal {RUNNER_CONFIG_SHA256}",
        )

    return D1RunnerConfig(
        path=path,
        sha256=digest,
        byte_count=len(raw),
        experiment_id=EXPERIMENT_ID,
        batch_size=4096,
        expected_class_count=30,
        validation_per_class=10,
        conditions=tuple(conditions),
        seeds=EXPECTED_SEEDS,
        pca_n_components=20,
        pca_solver="randomized",
        pca_whiten=False,
        c_grid=C_GRID,
        max_iter=1000,
        tol=0.0001,
        l1_ratio=0.0,
        cosine_quantiles=COSINE_QUANTILES,
        test_chunk_size=64,
        train_chunk_size=4096,
    )


@dataclass(frozen=True)
class _SplitData:
    intensity: np.ndarray
    labels: np.ndarray
    record_ids: tuple[str, ...]
    source_splits: tuple[str, ...]
    source_rows: np.ndarray


def _load_dataset(
    dataset_path: Path,
    batch_size: int,
) -> tuple[dict[str, _SplitData], Mapping[str, object]]:
    rows: dict[str, list[np.ndarray]] = {
        "finetune": [],
        "reference": [],
        "test": [],
    }
    labels: dict[str, list[np.ndarray]] = {
        key: [] for key in rows
    }
    record_ids: dict[str, list[str]] = {
        key: [] for key in rows
    }
    source_splits: dict[str, list[str]] = {
        key: [] for key in rows
    }
    source_rows: dict[str, list[np.ndarray]] = {
        key: [] for key in rows
    }
    with BacteriaIdBatchLoader(
        dataset_path,
        batch_size=batch_size,
    ) as loader:
        for batch in loader.iter_batches():
            split = batch.source_split
            rows[split].append(np.asarray(batch.intensity, dtype="<f4"))
            labels[split].append(np.asarray(batch.class_labels, dtype="<i8"))
            record_ids[split].extend(batch.record_ids)
            source_splits[split].extend(
                [split] * len(batch.record_ids)
            )
            source_rows[split].append(
                np.asarray(batch.source_rows, dtype="<i8")
            )
    split_data = {
        split: _SplitData(
            intensity=np.concatenate(rows[split], axis=0),
            labels=np.concatenate(labels[split], axis=0),
            record_ids=tuple(record_ids[split]),
            source_splits=tuple(source_splits[split]),
            source_rows=np.concatenate(source_rows[split], axis=0),
        )
        for split in rows
    }
    expected_labels = set(range(30))
    for split, data in split_data.items():
        observed_labels = {int(value) for value in np.unique(data.labels)}
        if observed_labels != expected_labels:
            raise D1RunnerValidationError(
                f"{split} class coverage",
                (
                    f"missing={sorted(expected_labels - observed_labels)}, "
                    f"unexpected={sorted(observed_labels - expected_labels)}"
                ),
            )
    dataset_path = Path(dataset_path)
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(dataset_path.iterdir())
        if path.is_file()
    }
    return split_data, {
        "dataset_id": dataset_path.name,
        "files": files,
        "record_count": sum(len(data.labels) for data in split_data.values()),
        "split_counts": {
            split: len(data.labels)
            for split, data in split_data.items()
        },
    }


def _stratified_split(
    split: _SplitData,
    *,
    seed: int,
    validation_per_class: int,
    expected_class_count: int,
) -> tuple[_SplitData, _SplitData]:
    generator = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []
    for class_label in range(expected_class_count):
        indices = np.flatnonzero(split.labels == class_label)
        if len(indices) <= validation_per_class:
            raise D1RunnerValidationError(
                f"finetune class {class_label}",
                (
                    f"requires more than {validation_per_class} records; "
                    f"observed {len(indices)}"
                ),
            )
        shuffled = generator.permutation(indices)
        validation_indices.extend(shuffled[:validation_per_class].tolist())
        train_indices.extend(shuffled[validation_per_class:].tolist())
    train_indices = np.asarray(sorted(train_indices), dtype=np.int64)
    validation_indices = np.asarray(
        sorted(validation_indices),
        dtype=np.int64,
    )

    def select(indices: np.ndarray) -> _SplitData:
        return _SplitData(
            intensity=split.intensity[indices],
            labels=split.labels[indices],
            record_ids=tuple(split.record_ids[index] for index in indices),
            source_splits=tuple(
                split.source_splits[index] for index in indices
            ),
            source_rows=split.source_rows[indices],
        )

    return select(train_indices), select(validation_indices)


def _combine(first: _SplitData, second: _SplitData) -> _SplitData:
    return _SplitData(
        intensity=np.concatenate((first.intensity, second.intensity), axis=0),
        labels=np.concatenate((first.labels, second.labels), axis=0),
        record_ids=first.record_ids + second.record_ids,
        source_splits=first.source_splits + second.source_splits,
        source_rows=np.concatenate((first.source_rows, second.source_rows)),
    )


def _condition_arrays(
    condition_id: str,
    pipeline: SavitzkyGolayPipeline | None,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    if condition_id == "released_input_control":
        return tuple(np.array(array, copy=True) for array in arrays)
    if condition_id == "released_input_plus_sg" and pipeline is not None:
        return tuple(
            np.array(pipeline.transform(array), copy=True)
            for array in arrays
        )
    raise D1RunnerValidationError(
        "condition",
        f"unsupported condition {condition_id!r}",
    )


def _fit_condition(
    config: D1RunnerConfig,
    condition: D1ConditionConfig,
    *,
    seed: int,
    train: _SplitData,
    validation: _SplitData,
    test: _SplitData,
    pipeline: SavitzkyGolayPipeline | None,
) -> Mapping[str, object]:
    started = perf_counter()
    train_x, validation_x, test_x = _condition_arrays(
        condition.condition_id,
        pipeline,
        train.intensity,
        validation.intensity,
        test.intensity,
    )
    pca = PCA(
        n_components=config.pca_n_components,
        whiten=config.pca_whiten,
        svd_solver=config.pca_solver,
        random_state=seed,
    )
    train_features = pca.fit_transform(train_x)
    validation_features = pca.transform(validation_x)
    test_features = pca.transform(test_x)
    validation_scores = []
    selected_model = None
    selected_c = None
    selected_accuracy = -1.0
    convergence_warning_count = 0
    for c_value in config.c_grid:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model = LogisticRegression(
                C=c_value,
                l1_ratio=config.l1_ratio,
                solver="lbfgs",
                max_iter=config.max_iter,
                tol=config.tol,
                class_weight=None,
                random_state=seed,
            )
            model.fit(train_features, train.labels)
        convergence_warnings = [
            warning
            for warning in captured
            if issubclass(warning.category, ConvergenceWarning)
        ]
        if convergence_warnings:
            raise D1RunnerValidationError(
                f"{condition.condition_id}.classifier.C={c_value}",
                str(convergence_warnings[0].message),
            )
        unexpected_warnings = [
            warning
            for warning in captured
            if not issubclass(warning.category, ConvergenceWarning)
        ]
        if unexpected_warnings:
            raise D1RunnerValidationError(
                f"{condition.condition_id}.classifier.C={c_value}",
                (
                    f"unexpected {unexpected_warnings[0].category.__name__}: "
                    f"{unexpected_warnings[0].message}"
                ),
            )
        prediction = model.predict(validation_features)
        score = float(accuracy_score(validation.labels, prediction))
        validation_scores.append({"accuracy": score, "c": c_value})
        if score > selected_accuracy:
            selected_accuracy = score
            selected_c = c_value
            selected_model = model
        convergence_warning_count += len(convergence_warnings)
    if selected_model is None or selected_c is None:
        raise RuntimeError("no logistic-regression candidate was selected")
    predicted = selected_model.predict(test_features)
    class_labels = np.arange(config.expected_class_count)
    per_class_recall = recall_score(
        test.labels,
        predicted,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test.labels,
        predicted,
        labels=class_labels,
    )
    predictions = [
        {
            "correct": bool(predicted_label == true_label),
            "predicted_class": int(predicted_label),
            "record_id": record_id,
            "true_class": int(true_label),
        }
        for record_id, true_label, predicted_label in zip(
            test.record_ids,
            test.labels,
            predicted,
            strict=True,
        )
    ]
    return {
        "condition_id": condition.condition_id,
        "metrics": {
            "accuracy": float(accuracy_score(test.labels, predicted)),
            "balanced_accuracy": float(
                balanced_accuracy_score(test.labels, predicted)
            ),
            "confusion_matrix": matrix.astype(int).tolist(),
            "macro_f1": float(
                f1_score(
                    test.labels,
                    predicted,
                    labels=class_labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "per_class_recall": per_class_recall.astype(float).tolist(),
        },
        "model": {
            "class_weight": None,
            "classifier": "logistic_regression",
            "convergence_warnings": convergence_warning_count,
            "l1_ratio": config.l1_ratio,
            "max_iter": config.max_iter,
            "pca_n_components": config.pca_n_components,
            "pca_solver": config.pca_solver,
            "pca_whiten": config.pca_whiten,
            "refit_with_validation": False,
            "regularization": "l2",
            "solver": "lbfgs",
            "tol": config.tol,
        },
        "predictions": predictions,
        "runtime_seconds": perf_counter() - started,
        "selected_c": selected_c,
        "validation_scores": validation_scores,
    }


def _overlap_count(
    left: set[object],
    right: set[object],
) -> int:
    return len(left & right)


def _digest_counts(
    left: np.ndarray,
    right: np.ndarray,
) -> int:
    left_counts: dict[bytes, int] = {}
    right_counts: dict[bytes, int] = {}
    for row in left:
        key = np.ascontiguousarray(row, dtype="<f4").tobytes()
        left_counts[key] = left_counts.get(key, 0) + 1
    for row in right:
        key = np.ascontiguousarray(row, dtype="<f4").tobytes()
        right_counts[key] = right_counts.get(key, 0) + 1
    return sum(
        left_count * right_counts.get(key, 0)
        for key, left_count in left_counts.items()
    )


def _nearest_cosine_quantiles(
    train: np.ndarray,
    test: np.ndarray,
    *,
    quantiles: tuple[float, ...],
    test_chunk_size: int,
    train_chunk_size: int,
) -> Mapping[str, float]:
    train64 = np.asarray(train, dtype=np.float64)
    test64 = np.asarray(test, dtype=np.float64)
    train_norm = np.linalg.norm(train64, axis=1)
    test_norm = np.linalg.norm(test64, axis=1)
    if np.any(train_norm == 0) or np.any(test_norm == 0):
        raise D1RunnerValidationError(
            "leakage_audit.cosine",
            "zero-norm spectrum is unsupported",
        )
    nearest = np.full(len(test64), -np.inf, dtype=np.float64)
    for test_start in range(0, len(test64), test_chunk_size):
        test_stop = min(test_start + test_chunk_size, len(test64))
        test_block = test64[test_start:test_stop]
        block_max = np.full(len(test_block), -np.inf, dtype=np.float64)
        for train_start in range(0, len(train64), train_chunk_size):
            train_stop = min(train_start + train_chunk_size, len(train64))
            similarities = (
                test_block @ train64[train_start:train_stop].T
            ) / (
                test_norm[test_start:test_stop, None]
                * train_norm[None, train_start:train_stop]
            )
            block_max = np.maximum(block_max, similarities.max(axis=1))
        nearest[test_start:test_stop] = block_max
    observed = np.quantile(nearest, np.asarray(quantiles, dtype=np.float64))
    return {
        str(quantile): float(value)
        for quantile, value in zip(quantiles, observed, strict=True)
    }


def _leakage_audit(
    config: D1RunnerConfig,
    train: _SplitData,
    validation: _SplitData,
    test: _SplitData,
) -> Mapping[str, object]:
    record_sets = {
        "train": set(train.record_ids),
        "validation": set(validation.record_ids),
        "test": set(test.record_ids),
    }
    coordinate_sets = {
        name: {
            (source_split, int(source_row))
            for source_split, source_row in zip(
                data.source_splits,
                data.source_rows,
                strict=True,
            )
        }
        for name, data in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }
    pairs = (
        ("train_validation", "train", "validation"),
        ("train_test", "train", "test"),
        ("validation_test", "validation", "test"),
    )
    split_lookup = {
        "train": train,
        "validation": validation,
        "test": test,
    }
    return {
        "exact_intensity_duplicate_pair_counts": {
            name: _digest_counts(
                split_lookup[left].intensity,
                split_lookup[right].intensity,
            )
            for name, left, right in pairs
        },
        "nearest_train_cosine_similarity_quantiles": (
            _nearest_cosine_quantiles(
                train.intensity,
                test.intensity,
                quantiles=config.cosine_quantiles,
                test_chunk_size=config.test_chunk_size,
                train_chunk_size=config.train_chunk_size,
            )
        ),
        "record_id_overlap_counts": {
            name: _overlap_count(record_sets[left], record_sets[right])
            for name, left, right in pairs
        },
        "source_coordinate_overlap_counts": {
            name: _overlap_count(
                coordinate_sets[left],
                coordinate_sets[right],
            )
            for name, left, right in pairs
        },
        "unique_source_coordinate_counts": {
            name: len(values)
            for name, values in coordinate_sets.items()
        },
    }


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except (OSError, UnicodeDecodeError, IndexError):
            pass
    return platform.processor() or "unknown"


def _environment_document() -> Mapping[str, object]:
    driver_path = Path("/proc/driver/nvidia/version")
    driver = None
    if driver_path.is_file():
        try:
            driver = driver_path.read_text(encoding="utf-8").splitlines()[0]
        except (OSError, UnicodeDecodeError):
            driver = None
    return {
        "compute_backend": "cpu",
        "cpu_count": os.cpu_count(),
        "cpu_model": _cpu_model(),
        "cuda_runtime": None,
        "git_commit": None,
        "gpu_name": None,
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "thread_environment": {
            key: os.environ.get(key)
            for key in THREAD_ENVIRONMENT_KEYS
        },
        "vcs_status": "unavailable",
        "nvidia_driver": driver,
    }


def _code_document() -> Mapping[str, Mapping[str, object]]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
        for relative_path in CODE_PATHS
    }


def _split_document(
    train: _SplitData,
    validation: _SplitData,
    test: _SplitData,
    *,
    validation_per_class: int,
) -> Mapping[str, object]:
    def one(data: _SplitData) -> Mapping[str, object]:
        return {
            "class_counts": {
                str(class_label): int(np.count_nonzero(data.labels == class_label))
                for class_label in range(30)
            },
            "count": len(data.labels),
            "record_ids_sha256": hashlib.sha256(
                ("\n".join(data.record_ids) + "\n").encode("utf-8")
            ).hexdigest(),
        }

    return {
        "test": one(test),
        "train": one(train),
        "validation": one(validation),
        "validation_per_class": validation_per_class,
    }


def run_d1_experiment(
    config_path: Path,
    dataset_path: Path,
    *,
    seed: int,
    result_level: str,
) -> Mapping[str, object]:
    started = perf_counter()
    config = load_d1_runner_config(config_path)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in config.seeds:
        raise D1RunnerValidationError(
            "seed",
            f"must be one of {config.seeds}",
        )
    if result_level not in {"smoke", "provisional"}:
        raise D1RunnerValidationError(
            "result_level",
            "single-seed runner must use smoke or provisional; "
            "complete_cell requires the five-seed aggregator",
        )
    dataset_path = Path(dataset_path)
    if result_level == "provisional":
        snapshot_path = dataset_path / "SHA256SUMS.sha256"
        try:
            snapshot_sha256 = _sha256_file(snapshot_path)
        except OSError as error:
            raise D1RunnerValidationError(
                "retained snapshot",
                str(error),
            ) from error
        if snapshot_sha256 != RETAINED_SNAPSHOT_SHA256:
            raise D1RunnerValidationError(
                "retained snapshot",
                (
                    f"SHA256SUMS.sha256 hash must equal "
                    f"{RETAINED_SNAPSHOT_SHA256}; observed {snapshot_sha256}"
                ),
            )
    split_data, dataset_document = _load_dataset(
        dataset_path,
        config.batch_size,
    )
    finetune_train, validation = _stratified_split(
        split_data["finetune"],
        seed=seed,
        validation_per_class=config.validation_per_class,
        expected_class_count=config.expected_class_count,
    )
    train = _combine(split_data["reference"], finetune_train)
    test = split_data["test"]
    pipeline_configs: dict[str, Mapping[str, object]] = {}
    pipelines: dict[str, SavitzkyGolayPipeline] = {}
    for condition in config.conditions:
        if condition.pipeline_config is None:
            continue
        pipeline_path = config.path.parent / condition.pipeline_config
        pipelines[condition.condition_id] = load_savitzky_golay_pipeline(
            pipeline_path
        )
        pipeline_configs[condition.condition_id] = {
            "bytes": pipeline_path.stat().st_size,
            "path": condition.pipeline_config,
            "sha256": _sha256_file(pipeline_path),
        }
    conditions = [
        _fit_condition(
            config,
            condition,
            seed=seed,
            train=train,
            validation=validation,
            test=test,
            pipeline=pipelines.get(condition.condition_id),
        )
        for condition in config.conditions
    ]
    return {
        "code": _code_document(),
        "conditions": conditions,
        "config": {
            "bytes": config.byte_count,
            "path": config.path.name,
            "sha256": config.sha256,
        },
        "dataset": dataset_document,
        "environment": _environment_document(),
        "experiment_id": config.experiment_id,
        "leakage_audit": _leakage_audit(
            config,
            train,
            validation,
            test,
        ),
        "pipeline_configs": pipeline_configs,
        "result_level": result_level,
        "runtime_seconds": perf_counter() - started,
        "schema_version": RESULT_SCHEMA_VERSION,
        "seed": seed,
        "split": _split_document(
            train,
            validation,
            test,
            validation_per_class=config.validation_per_class,
        ),
        "status": "completed",
    }


__all__ = [
    "D1RunnerConfig",
    "D1RunnerValidationError",
    "load_d1_runner_config",
    "run_d1_experiment",
]
