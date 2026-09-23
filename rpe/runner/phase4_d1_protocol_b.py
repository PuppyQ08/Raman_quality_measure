from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing
import platform
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import h5py
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from rpe.alignment import (
    AlignmentObservation, alignment_gap, bulk_paired_cluster_bootstrap,
    compare_alignment, cross_perturbation_accuracy, holm_step_down,
    paired_contribution_sign_flip,
)
from rpe.alignment.contracts import AlignmentValidationError
from rpe.evaluation import PreferredDirection

from rpe.runner.phase4_d1_protocol_b_authority import (
    ALPHAS,
    ARTIFACT_PAYLOAD_FILES,
    ARTIFACT_SCHEMA_VERSION,
    CLAIM_BOUNDARY,
    CODE_RELATIVE_PATHS,
    CONFIG_BYTES,
    CONFIG_SHA256,
    EXPERIMENT_ID,
    METRIC_OUTPUT_IDS,
    MODEL_SEEDS,
    PERTURBATIONS,
    REAL_ALPHA0_MODEL_DIGEST,
    REAL_ALPHA0_PREDICTION_DIGEST,
    REAL_ALPHA0_VALIDATION_DIGEST,
    SCHEMA_VERSION,
    TERMINAL_MARKERS,
    canonical_json_bytes,
    condition_ids,
    csv_bytes,
    jsonl_bytes,
    sha256_hex,
    write_sha256sums,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase4/configs/d1_protocol_b_full_domain_v1.json"
RUN_PREFIX = "phase4-d1-protocol-b-full-domain-"
CONDITION_IDS = condition_ids(PERTURBATIONS, ALPHAS)
C_GRID = (0.01, 0.1, 1.0, 10.0)
WORKER_ADMISSION_BYTES = 3_179_405_824
ADMISSION_BUDGET_BYTES = 64 * 1024**3
METRIC_DIRECTIONS = MappingProxyType({
    **{name: "lower_is_better" for name in (
        "mse", "rmse", "mae", "sam", "nmse", "wasserstein_1_cm1",
        "artifact_peak_ratio", "missing_peak_ratio",
    )},
    **{name: "higher_is_better" for name in (
        "pearson_r", "is_like_structure_to_noise", "precision", "recall", "f1",
    )},
})


class Phase4D1ProtocolBError(ValueError):
    pass


class Phase4D1ProtocolBModelLifecycleWarning(Phase4D1ProtocolBError):
    def __init__(self, receipt: Mapping[str, object]):
        self.receipt = dict(receipt)
        super().__init__(
            canonical_json_bytes(self.receipt).decode("utf-8").rstrip("\n")
        )

    def __reduce__(self) -> tuple[object, tuple[Mapping[str, object]]]:
        return (self.__class__, (self.receipt,))


@dataclass(frozen=True)
class Phase4D1ProtocolBConfig:
    path: Path
    raw_bytes: bytes
    sha256: str
    document: Mapping[str, object]
    synthetic_fixture: bool
    class_count: int
    records_per_class: int
    model_seeds: tuple[int, ...]
    feature_count: int
    condition_ids: tuple[str, ...]
    artifact_payload_files: tuple[str, ...]
    expected: Mapping[str, int]


@dataclass(frozen=True)
class SyntheticD1ProtocolBInputs:
    class_count: int
    records_per_class: int
    model_seeds: tuple[int, ...]
    feature_count: int
    record_ids: tuple[str, ...]
    labels: np.ndarray
    train_by_seed_condition: Mapping[tuple[int, str], np.ndarray]
    train_labels_by_seed: Mapping[int, np.ndarray]
    validation_by_seed_condition: Mapping[tuple[int, str], np.ndarray]
    validation_labels_by_seed: Mapping[int, np.ndarray]
    test_by_condition: Mapping[str, np.ndarray]
    synthetic_alpha0_model_rows: tuple[Mapping[str, object], ...]
    synthetic_alpha0_validation_rows: tuple[Mapping[str, object], ...]
    synthetic_alpha0_prediction_rows: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class D1ProtocolBModelCell:
    model_seed: int
    condition_id: str
    train_condition_id: str
    validation_condition_id: str
    test_condition_id: str
    selected_c: float
    validation_scores: tuple[Mapping[str, object], ...]
    train_matrix_sha256: str
    validation_matrix_sha256: str
    pca_train_feature_sha256: str
    pca_validation_feature_sha256: str
    model_state_sha256: str
    train_record_ids_sha256: str
    validation_record_ids_sha256: str
    test_record_ids_sha256: str
    warning_state: str
    refit_with_validation: bool = False


@dataclass(frozen=True)
class Phase4D1ProtocolBSummary:
    path: Path
    run_id: str
    status: str
    prediction_row_count: int
    class_observation_count: int


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return sha256_hex(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


def _ids_sha(values: Sequence[str]) -> str:
    return sha256_hex(("\n".join(values) + "\n").encode("utf-8"))


def _decode_condition_id(condition_id: str) -> tuple[str | None, float | None]:
    if condition_id == "alpha0":
        return None, None
    perturbation_id, encoded_alpha = condition_id.split(":", 1)
    alpha = float(np.frombuffer(bytes.fromhex(encoded_alpha), dtype="<f8")[0])
    return perturbation_id, alpha


def _warning_receipt(
    *,
    condition_id: str,
    model_seed: int,
    c: float,
    warning: warnings.WarningMessage,
) -> dict[str, object]:
    return {
        "state": "failed_model_lifecycle",
        "condition_id": condition_id,
        "model_seed": int(model_seed),
        "c": float(c),
        "warning_category": str(warning.category.__name__),
        "warning_message": str(warning.message),
    }


def _real_run_id(config: Mapping[str, object], bridge: Mapping[str, object]) -> str:
    config_sha256 = sha256_hex(canonical_json_bytes(dict(config)))
    run_document = {
        "config_sha256": config_sha256,
        "bridge_sha256": bridge["bridge_sha256"],
        "model_seeds": list(MODEL_SEEDS),
        "condition_ids": list(CONDITION_IDS),
    }
    return RUN_PREFIX + sha256_hex(canonical_json_bytes(run_document))


def _default_artifact_counts() -> dict[str, int]:
    return {
        "model_cells": 205,
        "validation_scores": 820,
        "predictions": 615_000,
        "seed_class_conditions": 6_150,
        "condition_summary": 41,
        "class_observations": 15_600,
        "alignment_results": 13,
        "bootstrap_results": 12,
        "sign_flip_results": 24,
        "holm_family": 24,
        "figure1_data_rows": 520,
        "figure2_data_rows": 13,
        "secondary_table_rows": 13,
        "artifact_files": 24,
    }


def _write_protocol_b_artifact(
    output_root: Path,
    payloads: Mapping[str, bytes],
    *,
    terminal_name: str,
    terminal_bytes: bytes,
) -> Path:
    target = Path(output_root)
    artifact_names = set(ARTIFACT_PAYLOAD_FILES) | set(TERMINAL_MARKERS) | {"SHA256SUMS"}
    if target.exists():
        if not target.is_dir():
            raise Phase4D1ProtocolBError("output_root exists and is not a directory")
        if any((target / name).exists() for name in artifact_names):
            raise Phase4D1ProtocolBError("output_root already contains artifact payloads")
    else:
        target.mkdir(parents=True, exist_ok=False)
    for name, raw in payloads.items():
        (target / name).write_bytes(raw)
    (target / terminal_name).write_bytes(terminal_bytes)
    (target / "SHA256SUMS").write_bytes(
        write_sha256sums(payloads, terminal_name, terminal_bytes)
    )
    return target


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _code_authority() -> dict[str, dict[str, object]]:
    return {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": _sha_file(ROOT / relative),
        }
        for relative in CODE_RELATIVE_PATHS
    }


def _environment_authority() -> dict[str, str]:
    return {
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "matplotlib": matplotlib.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "threadpoolctl": threadpoolctl.__version__,
    }


def _as_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Phase4D1ProtocolBError(f"{field}: must be positive integer")
    return int(value)


def parse_phase4_d1_protocol_b_config(
    path: Path,
    raw: bytes,
    *,
    require_frozen_identity: bool,
) -> Phase4D1ProtocolBConfig:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Phase4D1ProtocolBError(f"config parse: {error}") from error
    if canonical_json_bytes(document) != raw:
        raise Phase4D1ProtocolBError("config must be canonical JSON")
    if require_frozen_identity and (len(raw) != CONFIG_BYTES or sha256_hex(raw) != CONFIG_SHA256):
        raise Phase4D1ProtocolBError("frozen config identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D1ProtocolBError("config schema/experiment mismatch")
    if document.get("protocol") != "B" or document.get("tier") != "full_domain_core":
        raise Phase4D1ProtocolBError("config protocol/tier mismatch")
    if document.get("claim_boundary") != CLAIM_BOUNDARY:
        raise Phase4D1ProtocolBError("config claim boundary mismatch")
    synthetic = bool(document.get("synthetic_fixture", False))
    artifact_files = tuple(document.get("artifact_payload_files", ()))
    if artifact_files != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D1ProtocolBError("artifact_payload_files exact order mismatch")
    if tuple(document.get("model_seeds", ())) != MODEL_SEEDS:
        raise Phase4D1ProtocolBError("model_seeds mismatch")
    if tuple(document.get("active_perturbation_ids", ())) != PERTURBATIONS:
        raise Phase4D1ProtocolBError("perturbation grid mismatch")
    if tuple(float(item) for item in document.get("alpha_grid", ())) != ALPHAS:
        raise Phase4D1ProtocolBError("alpha grid mismatch")
    if tuple(document.get("metric_output_ids", ())) != METRIC_OUTPUT_IDS:
        raise Phase4D1ProtocolBError("metric order mismatch")
    if tuple(document.get("condition_ids", ())) != CONDITION_IDS:
        raise Phase4D1ProtocolBError("condition ids mismatch")
    if not isinstance(document.get("code_authority"), Mapping) or set(document["code_authority"]) != set(CODE_RELATIVE_PATHS):
        raise Phase4D1ProtocolBError("code_authority mismatch")
    if not isinstance(document.get("trust_anchor"), Mapping):
        raise Phase4D1ProtocolBError("trust_anchor missing")
    trust_anchor = document["trust_anchor"]
    if trust_anchor.get("config_authority_relative_path") != "rpe/runner/phase4_d1_protocol_b_authority.py":
        raise Phase4D1ProtocolBError("trust_anchor mismatch")
    expected = document.get("expected")
    if not isinstance(expected, Mapping):
        raise Phase4D1ProtocolBError("expected counts missing")
    if not synthetic:
        if set(document.get("parent_artifacts", {})) != {"step15", "step21", "step23"}:
            raise Phase4D1ProtocolBError("parent_artifacts must bind step15/step21/step23")
        if set(document["code_authority"]) != set(CODE_RELATIVE_PATHS):
            raise Phase4D1ProtocolBError("code_authority mismatch")
        if set(document.get("environment_authority", {})) != set(_environment_authority()):
            raise Phase4D1ProtocolBError("environment authority mismatch")
    denominators = document.get("denominators", {})
    class_count = _as_positive_int(denominators.get("class_count"), "denominators.class_count")
    records_per_class = _as_positive_int(denominators.get("records_per_class"), "denominators.records_per_class")
    feature_count = _as_positive_int(document.get("feature_count", 997), "feature_count")
    return Phase4D1ProtocolBConfig(
        path=Path(path),
        raw_bytes=raw,
        sha256=sha256_hex(raw),
        document=MappingProxyType(document),
        synthetic_fixture=synthetic,
        class_count=class_count,
        records_per_class=records_per_class,
        model_seeds=MODEL_SEEDS,
        feature_count=feature_count,
        condition_ids=CONDITION_IDS,
        artifact_payload_files=ARTIFACT_PAYLOAD_FILES,
        expected=MappingProxyType({str(key): int(value) for key, value in expected.items()}),
    )


def load_phase4_d1_protocol_b_config(path: Path = DEFAULT_CONFIG) -> Phase4D1ProtocolBConfig:
    target = Path(path)
    return parse_phase4_d1_protocol_b_config(
        target,
        target.read_bytes(),
        require_frozen_identity=(target.resolve() == DEFAULT_CONFIG.resolve()),
    )


def _synthetic_matrix(class_count: int, items_per_class: int, feature_count: int, jitter: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(class_count, dtype=np.int64), items_per_class)
    basis = np.eye(class_count, feature_count, dtype=np.float64) * 6.0
    rows = np.vstack(
        [
            basis[int(label)] + rng.normal(0.0, jitter, feature_count)
            for label in labels
        ]
    )
    return np.ascontiguousarray(rows, dtype="<f4"), labels


def make_synthetic_d1_protocol_b_inputs(
    *,
    class_count: int,
    records_per_class: int,
    model_seeds: Sequence[int],
    feature_count: int = 8,
) -> SyntheticD1ProtocolBInputs:
    if tuple(model_seeds) != MODEL_SEEDS:
        raise Phase4D1ProtocolBError("synthetic fixture requires the five frozen seeds")
    record_ids = tuple(
        f"test-{index:04d}"
        for index in range(class_count * records_per_class)
    )
    labels = np.repeat(np.arange(class_count, dtype=np.int64), records_per_class)
    train_by_seed_condition: dict[tuple[int, str], np.ndarray] = {}
    validation_by_seed_condition: dict[tuple[int, str], np.ndarray] = {}
    test_by_condition: dict[str, np.ndarray] = {}
    train_labels_by_seed: dict[int, np.ndarray] = {}
    validation_labels_by_seed: dict[int, np.ndarray] = {}
    for condition_index, condition_id in enumerate(CONDITION_IDS):
        test_values, _ = _synthetic_matrix(
            class_count, records_per_class, feature_count, 0.02 + condition_index * 0.001, 2000 + condition_index
        )
        test_by_condition[condition_id] = test_values
        for model_seed in MODEL_SEEDS:
            train_values, train_labels = _synthetic_matrix(
                class_count,
                max(4, records_per_class * 3),
                feature_count,
                0.03 + condition_index * 0.001,
                100 * model_seed + condition_index,
            )
            validation_values, validation_labels = _synthetic_matrix(
                class_count,
                max(2, records_per_class),
                feature_count,
                0.03 + condition_index * 0.001,
                500 + 100 * model_seed + condition_index,
            )
            train_by_seed_condition[(model_seed, condition_id)] = train_values
            validation_by_seed_condition[(model_seed, condition_id)] = validation_values
            train_labels_by_seed[model_seed] = train_labels
            validation_labels_by_seed[model_seed] = validation_labels
    seed_class_conditions = []
    for model_seed in MODEL_SEEDS:
        for class_label in range(class_count):
            seed_class_conditions.append(
                {
                    "condition_id": "alpha0",
                    "model_seed": model_seed,
                    "class_label": class_label,
                    "accuracy_seed_class": 1.0,
                }
            )
    synthetic = SyntheticD1ProtocolBInputs(
        class_count=class_count,
        records_per_class=records_per_class,
        model_seeds=tuple(model_seeds),
        feature_count=feature_count,
        record_ids=record_ids,
        labels=labels,
        train_by_seed_condition=MappingProxyType(train_by_seed_condition),
        train_labels_by_seed=MappingProxyType(train_labels_by_seed),
        validation_by_seed_condition=MappingProxyType(validation_by_seed_condition),
        validation_labels_by_seed=MappingProxyType(validation_labels_by_seed),
        test_by_condition=MappingProxyType(test_by_condition),
        synthetic_alpha0_model_rows=(),
        synthetic_alpha0_validation_rows=(),
        synthetic_alpha0_prediction_rows=(),
    )
    model_rows, validation_rows, prediction_rows = _synthetic_alpha0_reference_rows(synthetic)
    return SyntheticD1ProtocolBInputs(
        class_count=synthetic.class_count,
        records_per_class=synthetic.records_per_class,
        model_seeds=synthetic.model_seeds,
        feature_count=synthetic.feature_count,
        record_ids=synthetic.record_ids,
        labels=synthetic.labels,
        train_by_seed_condition=synthetic.train_by_seed_condition,
        train_labels_by_seed=synthetic.train_labels_by_seed,
        validation_by_seed_condition=synthetic.validation_by_seed_condition,
        validation_labels_by_seed=synthetic.validation_labels_by_seed,
        test_by_condition=synthetic.test_by_condition,
        synthetic_alpha0_model_rows=model_rows,
        synthetic_alpha0_validation_rows=validation_rows,
        synthetic_alpha0_prediction_rows=prediction_rows,
    )


def make_synthetic_d1_protocol_b_config(inputs: SyntheticD1ProtocolBInputs) -> Phase4D1ProtocolBConfig:
    model_count = len(inputs.model_seeds) * len(CONDITION_IDS)
    prediction_row_count = len(inputs.model_seeds) * len(CONDITION_IDS) * len(inputs.record_ids)
    seed_class_condition_count = len(inputs.model_seeds) * inputs.class_count * len(CONDITION_IDS)
    class_observation_count = inputs.class_count * len(METRIC_OUTPUT_IDS) * (len(ALPHAS) - 1)
    document = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol": "B",
        "tier": "full_domain_core",
        "synthetic_fixture": True,
        "claim_boundary": CLAIM_BOUNDARY,
        "feature_count": inputs.feature_count,
        "model_seeds": list(inputs.model_seeds),
        "active_perturbation_ids": list(PERTURBATIONS),
        "alpha_grid": list(ALPHAS),
        "condition_ids": list(CONDITION_IDS),
        "metric_output_ids": list(METRIC_OUTPUT_IDS),
        "artifact_payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "authorities": {"step24_design": {"path": "reports/phase4/step24_d1_protocol_b_full_domain_outcome_design.md", "bytes": 28969, "sha256": "step24-synthetic"}},
        "parent_artifacts": {"step15": {"relative_path": "synthetic/step15"}, "step21": {"relative_path": "synthetic/step21"}, "step23": {"relative_path": "synthetic/step23"}},
        "environment_authority": {"synthetic": True},
        "model_recipe": {"pca_components": min(20, inputs.feature_count), "c_grid": list(C_GRID)},
        "inference": {"bootstrap_resamples": 2000, "sign_flip_resamples": 100000, "holm_family_size": 24},
        "figure_contract": {"figure1_rows": inputs.class_count * len(METRIC_OUTPUT_IDS) * (len(ALPHAS) - 1), "figure2_rows": len(METRIC_OUTPUT_IDS)},
        "artifact_contract": {"terminal_markers": list(TERMINAL_MARKERS), "configured_payload_count": len(ARTIFACT_PAYLOAD_FILES), "artifact_file_count": len(ARTIFACT_PAYLOAD_FILES) + 2},
        "inherited_rulings": {"p01_p05": "not_evaluable_coverage", "p06": "structurally_ineligible_missing_explicit_baseline", "p07": "structurally_ineligible_missing_explicit_baseline"},
        "code_authority": {relative: {"bytes": 0, "sha256": "synthetic"} for relative in CODE_RELATIVE_PATHS},
        "trust_anchor": {"config_authority_relative_path": "rpe/runner/phase4_d1_protocol_b_authority.py", "config_binds_authority": False, "direction": "authority_to_config_only"},
        "alpha0_equivalence": {
            "model_digest": sha256_hex(jsonl_bytes(inputs.synthetic_alpha0_model_rows)),
            "validation_digest": sha256_hex(jsonl_bytes(inputs.synthetic_alpha0_validation_rows)),
            "prediction_digest": sha256_hex(jsonl_bytes(inputs.synthetic_alpha0_prediction_rows)),
        },
        "denominators": {"class_count": inputs.class_count, "records_per_class": inputs.records_per_class},
        "expected": {
            "configured_payload_count": len(ARTIFACT_PAYLOAD_FILES),
            "artifact_file_count": len(ARTIFACT_PAYLOAD_FILES) + 2,
            "model_cell_count": model_count,
            "validation_row_count": model_count * len(C_GRID),
            "prediction_row_count": prediction_row_count,
            "seed_class_condition_count": seed_class_condition_count,
            "class_observation_count": class_observation_count,
            "alignment_row_count": len(METRIC_OUTPUT_IDS),
            "bootstrap_row_count": len(METRIC_OUTPUT_IDS) - 1,
            "sign_flip_row_count": 24,
            "holm_row_count": 24,
        },
    }
    raw = canonical_json_bytes(document)
    return parse_phase4_d1_protocol_b_config(
        Path("synthetic-config.json"),
        raw,
        require_frozen_identity=False,
    )


def _synthetic_alpha0_reference_rows(
    inputs: SyntheticD1ProtocolBInputs,
) -> tuple[tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    model_rows = []
    validation_rows = []
    prediction_rows = []
    for model_seed in inputs.model_seeds:
        train = inputs.train_by_seed_condition[(model_seed, "alpha0")]
        validation = inputs.validation_by_seed_condition[(model_seed, "alpha0")]
        train_labels = inputs.train_labels_by_seed[model_seed]
        validation_labels = inputs.validation_labels_by_seed[model_seed]
        test = inputs.test_by_condition["alpha0"]
        model_cell, validation_scores, predictions = _fit_predict_single_condition(
            model_seed,
            "alpha0",
            train,
            train_labels,
            validation,
            validation_labels,
            test,
            inputs.labels,
            inputs.record_ids,
        )
        model_rows.append(model_cell)
        validation_rows.extend(validation_scores)
        prediction_rows.extend(predictions)
    return tuple(model_rows), tuple(validation_rows), tuple(prediction_rows)


def fit_protocol_b_condition_models(
    inputs: SyntheticD1ProtocolBInputs,
    config: Phase4D1ProtocolBConfig,
) -> tuple[D1ProtocolBModelCell, ...]:
    models = []
    for condition_id in config.condition_ids:
        for model_seed in config.model_seeds:
            train = inputs.train_by_seed_condition[(model_seed, condition_id)]
            validation = inputs.validation_by_seed_condition[(model_seed, condition_id)]
            model_cell, _validation_rows, _predictions = _fit_predict_single_condition(
                model_seed,
                condition_id,
                train,
                inputs.train_labels_by_seed[model_seed],
                validation,
                inputs.validation_labels_by_seed[model_seed],
                inputs.test_by_condition[condition_id],
                inputs.labels,
                inputs.record_ids,
            )
            models.append(D1ProtocolBModelCell(**model_cell))
    return tuple(models)


def _fit_predict_single_condition(
    model_seed: int,
    condition_id: str,
    train: np.ndarray,
    train_labels: np.ndarray,
    validation: np.ndarray,
    validation_labels: np.ndarray,
    test: np.ndarray,
    test_labels: np.ndarray,
    record_ids: Sequence[str],
    *,
    train_record_ids: Sequence[str] | None = None,
    validation_record_ids: Sequence[str] | None = None,
    projected_row_hashes: Sequence[str] | None = None,
    code_identity: str = "synthetic-d1-b",
    config_sha256: str = "synthetic",
    canonical_model_state: bool = False,
) -> tuple[dict[str, object], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    component_count = min(20, train.shape[1], len(np.unique(train_labels)))
    if not all(np.isfinite(value).all() for value in (train, validation, test)):
        raise Phase4D1ProtocolBError("model lifecycle nonfinite input")
    expected_classes = set(np.unique(test_labels).tolist())
    if set(np.unique(train_labels).tolist()) != expected_classes or set(np.unique(validation_labels).tolist()) != expected_classes:
        raise Phase4D1ProtocolBError("model lifecycle missing class")
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        whiten=False,
        random_state=model_seed,
    )
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        train_features = pca.fit_transform(train)
        validation_features = pca.transform(validation)
        test_features = pca.transform(test)
    if observed or not all(np.isfinite(value).all() for value in (train_features, validation_features, test_features)):
        raise Phase4D1ProtocolBError("model lifecycle PCA warning/nonfinite output")
    rows = []
    best_score = -math.inf
    best_classifier: LogisticRegression | None = None
    best_c = C_GRID[0]
    with np.errstate(divide="raise", over="raise", under="ignore", invalid="raise"):
        for c in C_GRID:
            with warnings.catch_warnings(record=True) as candidate_warnings:
                warnings.simplefilter("always")
                classifier = LogisticRegression(
                    C=c,
                    l1_ratio=0.0,
                    max_iter=1000,
                    class_weight=None,
                    random_state=model_seed,
                    solver="lbfgs",
                    tol=1e-4,
                )
                classifier.fit(train_features, train_labels)
                score = float(classifier.score(validation_features, validation_labels))
            if candidate_warnings:
                raise Phase4D1ProtocolBModelLifecycleWarning(
                    _warning_receipt(
                        condition_id=condition_id,
                        model_seed=model_seed,
                        c=float(c),
                        warning=candidate_warnings[0],
                    )
                )
            rows.append({"condition_id": condition_id, "c": c, "model_seed": model_seed, "validation_top1_accuracy": score})
            if score > best_score:
                best_score = score
                best_classifier = classifier
                best_c = c
    assert best_classifier is not None
    predicted = best_classifier.predict(test_features).astype(int)
    if not np.isfinite(best_classifier.coef_).all() or not np.isfinite(best_classifier.intercept_).all() or len(predicted) != len(test_labels):
        raise Phase4D1ProtocolBError("model lifecycle invalid fitted state/predictions")
    if canonical_model_state:
        model_state_sha = sha256_hex(canonical_json_bytes({
            "selected_c": float(best_c),
            "classes": np.asarray(best_classifier.classes_, dtype="<i8").tolist(),
            "coef_sha256": _array_sha(best_classifier.coef_),
            "intercept_sha256": _array_sha(best_classifier.intercept_),
            "n_iter": np.asarray(best_classifier.n_iter_, dtype="<i8").tolist(),
            "pca_components_sha256": _array_sha(pca.components_),
            "pca_mean_sha256": _array_sha(pca.mean_),
            "pca_explained_variance_sha256": _array_sha(pca.explained_variance_),
        }))
    else:
        model_state_sha = sha256_hex(np.ascontiguousarray(best_classifier.coef_, dtype="<f8").tobytes() + np.ascontiguousarray(best_classifier.intercept_, dtype="<f8").tobytes())
    model_cell = {
        "model_seed": model_seed,
        "condition_id": condition_id,
        "train_condition_id": condition_id,
        "validation_condition_id": condition_id,
        "test_condition_id": condition_id,
        "selected_c": float(best_c),
        "validation_scores": tuple(rows),
        "train_matrix_sha256": _array_sha(train, "<f4"),
        "validation_matrix_sha256": _array_sha(validation, "<f4"),
        "pca_train_feature_sha256": _array_sha(train_features, "<f8"),
        "pca_validation_feature_sha256": _array_sha(validation_features, "<f8"),
        "model_state_sha256": model_state_sha,
        "train_record_ids_sha256": _ids_sha(train_record_ids or [f"train-{model_seed}-{index}" for index in range(train.shape[0])]),
        "validation_record_ids_sha256": _ids_sha(validation_record_ids or [f"validation-{model_seed}-{index}" for index in range(validation.shape[0])]),
        "test_record_ids_sha256": _ids_sha(tuple(record_ids)),
        "warning_state": "none",
        "refit_with_validation": False,
    }
    predictions = tuple(
        {
            "code_identity": code_identity,
            "condition_id": condition_id,
            "config_sha256": config_sha256,
            "correct": bool(int(label) == int(pred)),
            "model_identity": f"{model_seed}:{condition_id}",
            "model_seed": model_seed,
            "predicted_class": int(pred),
            "projected_row_sha256": projected_row_hashes[index] if projected_row_hashes is not None else sha256_hex(f"{condition_id}:{record_id}".encode("utf-8")),
            "record_id": str(record_id),
            "record_order": int(index),
            "true_class": int(label),
        }
        for index, (record_id, label, pred) in enumerate(zip(record_ids, test_labels, predicted, strict=True))
    )
    return model_cell, tuple(rows), predictions


def validate_alpha0_equivalence(
    model_rows: Sequence[Mapping[str, object]],
    validation_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
    config: Phase4D1ProtocolBConfig,
) -> dict[str, object]:
    observed = {
        "model_digest": sha256_hex(jsonl_bytes(tuple(model_rows))),
        "validation_digest": sha256_hex(jsonl_bytes(tuple(validation_rows))),
        "prediction_digest": sha256_hex(jsonl_bytes(tuple(prediction_rows))),
    }
    expected = dict(config.document["alpha0_equivalence"])
    mismatch_count = sum(1 for key, value in observed.items() if expected.get(key) != value)
    if mismatch_count:
        raise Phase4D1ProtocolBError("alpha0 digest mismatch")
    return {
        **observed,
        "status": "passed",
        "mismatch_count": 0,
    }


def _real_alpha0_equivalence_contract() -> dict[str, str]:
    return {
        "model_digest": REAL_ALPHA0_MODEL_DIGEST,
        "validation_digest": REAL_ALPHA0_VALIDATION_DIGEST,
        "prediction_digest": REAL_ALPHA0_PREDICTION_DIGEST,
    }


def _load_parent_artifact(receipt: Mapping[str, object], *, label: str) -> dict[str, object]:
    path = ROOT / str(receipt["relative_path"])
    if not path.is_dir():
        raise Phase4D1ProtocolBError(f"{label} parent path missing")
    sums_path = path / "SHA256SUMS"
    if not sums_path.is_file():
        raise Phase4D1ProtocolBError(f"{label} parent missing SHA256SUMS")
    observed_sums_sha = _sha_file(sums_path)
    if observed_sums_sha != str(receipt["sha256sums_sha256"]):
        raise Phase4D1ProtocolBError(f"{label} parent SHA256SUMS identity mismatch")
    checksum_rows: dict[str, str] = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4D1ProtocolBError(f"{label} parent malformed SHA256SUMS") from error
        if name in checksum_rows or len(digest) != 64:
            raise Phase4D1ProtocolBError(f"{label} parent invalid checksum ledger")
        checksum_rows[name] = digest
    for name, digest in checksum_rows.items():
        target = path / name
        if not target.is_file() or _sha_file(target) != digest:
            raise Phase4D1ProtocolBError(f"{label} parent payload checksum mismatch: {name}")
    marker_path = path / "complete.json"
    if not marker_path.is_file():
        raise Phase4D1ProtocolBError(f"{label} parent missing complete marker")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker_run_id = str(marker.get("run_id"))
    expected_run_id = str(receipt["run_id"])
    allowed_run_ids = {expected_run_id, path.name}
    if label == "step21" and path.name.startswith("phase4-d1-protocol-a-full-domain-"):
        allowed_run_ids.add(path.name.removeprefix("phase4-d1-protocol-a-full-domain-"))
    if label != "step15" and marker_run_id not in allowed_run_ids:
        raise Phase4D1ProtocolBError(f"{label} parent run identity mismatch")
    allowed_status = {"complete"} if label != "step23" else {"complete", "pass"}
    if str(marker.get("status")) not in allowed_status:
        raise Phase4D1ProtocolBError(f"{label} parent terminal state mismatch")
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    if label == "step23":
        gate_path = path / "gate.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
        if str(marker.get("overall_status", manifest.get("overall_status", "pass"))) != "pass":
            raise Phase4D1ProtocolBError("step23 overall gate mismatch")
        full_domain = gate.get("full_domain_core", manifest.get("class_gate", {}).get("full_domain_core", {}))
        if str(full_domain.get("state")) != "evaluable":
            raise Phase4D1ProtocolBError("step23 full-domain gate mismatch")
    return {
        "path": path,
        "run_id": str(receipt["run_id"]),
        "sha256sums_sha256": str(receipt["sha256sums_sha256"]),
        "marker": MappingProxyType(marker),
        "manifest": MappingProxyType(manifest),
        "checksums": MappingProxyType(checksum_rows),
    }


def _condition_rank_map(config: Phase4D1ProtocolBConfig) -> dict[str, int]:
    return {condition_id: index for index, condition_id in enumerate(config.condition_ids)}


def validate_d1_protocol_b_parent_authorities(config: Phase4D1ProtocolBConfig) -> dict[str, object]:
    if config.synthetic_fixture:
        return {
            "authorized_step15_payloads": ("record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"),
            "bridge_row_count": config.class_count * config.records_per_class * len(config.condition_ids),
            "alpha0_reference": dict(config.document["alpha0_equivalence"]),
        }
    parent_receipts = config.document["parent_artifacts"]
    step15 = _load_parent_artifact(parent_receipts["step15"], label="step15")
    step21 = _load_parent_artifact(parent_receipts["step21"], label="step21")
    step23 = _load_parent_artifact(parent_receipts["step23"], label="step23")
    required_step23 = {
        "config.json", "manifest.json", "gate.json", "model_cells.jsonl",
        "model_role_occurrences.jsonl", "record_conditions.jsonl",
        "condition_matrix_shards.jsonl", "source_records.jsonl",
        "operator_cells.jsonl", "class_summaries.jsonl",
    }
    required_step15 = {"record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"}
    if not required_step23 <= set(step23["checksums"]):
        raise Phase4D1ProtocolBError("step23 required parent payload missing")
    if not required_step15 <= set(step15["checksums"]):
        raise Phase4D1ProtocolBError("step15 authorized payload missing")

    condition_rank = _condition_rank_map(config)
    test_rows: dict[tuple[str, str], dict[str, object]] = {}
    alpha0_test_rows: list[dict[str, object]] = []
    duplicate_key_count = 0
    for row in _read_jsonl(step23["path"] / "record_conditions.jsonl"):
        if str(row["source_split"]) != "test":
            continue
        key = (str(row["record_id"]), str(row["condition_id"]))
        if key in test_rows:
            duplicate_key_count += 1
            continue
        test_rows[key] = row
        if str(row["condition_id"]) == "alpha0":
            alpha0_test_rows.append(row)
    alpha0_test_rows.sort(key=lambda row: int(row["source_row"]))
    test_record_ids = [str(row["record_id"]) for row in alpha0_test_rows]
    class_counts = Counter(int(row["class_label"]) for row in alpha0_test_rows)
    if len(test_rows) != 123000:
        raise Phase4D1ProtocolBError("step15-step23 bridge denominator mismatch")
    if len(test_record_ids) != 3000 or len(class_counts) != 30 or any(count != 100 for count in class_counts.values()):
        raise Phase4D1ProtocolBError("step23 shared test ledger mismatch")

    step15_records: dict[tuple[str, str], dict[str, object]] = {}
    for row in _read_jsonl(step15["path"] / "record_conditions.jsonl"):
        key = (str(row["record_id"]), str(row["condition_id"]))
        if key in test_rows:
            step15_records[key] = row

    missing_key_count = 0
    extra_key_count = 0
    field_mismatch_count = 0
    state_mismatch_count = 0
    test_order_mismatch_count = 0
    bridge_rows: list[dict[str, object]] = []
    for key, row23 in test_rows.items():
        row15 = step15_records.get(key)
        if row15 is None:
            missing_key_count += 1
            continue
        if str(row15["state"]) != "complete" or str(row23["state"]) != "complete":
            state_mismatch_count += 1
        if int(row15["record_order"]) != int(row23["source_row"]):
            test_order_mismatch_count += 1
        if (
            str(row15["axis_sha256"]) != str(row23["output_axis_sha256"])
            or str(row15["intensity_sha256"]) != str(row23["output_intensity_sha256"])
            or str(row15["projected_row_sha256"]) != str(row23["support_projection_sha256"])
        ):
            field_mismatch_count += 1
        bridge_rows.append(
            {
                "test_order": int(row23["source_row"]),
                "record_id": str(row23["record_id"]),
                "class_label": int(row23["class_label"]),
                "condition_id": str(row23["condition_id"]),
                "state": str(row23["state"]),
                "axis_sha256": str(row23["output_axis_sha256"]),
                "intensity_sha256": str(row23["output_intensity_sha256"]),
                "support_projection_sha256": str(row23["support_projection_sha256"]),
            }
        )
    extra_key_count = max(len(step15_records) - len(test_rows), 0)
    bridge_rows.sort(key=lambda row: (int(row["test_order"]), condition_rank[str(row["condition_id"])]))
    bridge_hasher = hashlib.sha256()
    for row in bridge_rows:
        bridge_hasher.update(canonical_json_bytes(row))

    metric_seen: set[tuple[str, str, str]] = set()
    metric_row_count = 0
    for row in _read_jsonl(step15["path"] / "metric_values.jsonl"):
        key = (str(row["record_id"]), str(row["condition_id"]))
        metric_key = (*key, str(row.get("metric_output_id")))
        if key in test_rows and (metric_key in metric_seen or str(row["state"]) != "complete" or row.get("value") is None or not math.isfinite(float(row["value"]))):
            raise Phase4D1ProtocolBError("step15 metric receipt schema/state mismatch")
        if key in test_rows:
            metric_seen.add(metric_key)
            metric_row_count += 1
    cwt_seen: set[tuple[str, str]] = set()
    cwt_row_count = 0
    for row in _read_jsonl(step15["path"] / "peak_receipts.jsonl"):
        key = (str(row["record_id"]), str(row["condition_id"]))
        if key in test_rows and (key in cwt_seen or str(row["state"]) != "complete"):
            raise Phase4D1ProtocolBError("step15 CWT receipt schema/state mismatch")
        if key in test_rows:
            cwt_seen.add(key)
            cwt_row_count += 1
    if (len(bridge_rows), metric_row_count, cwt_row_count) != (123000, 1599000, 123000):
        raise Phase4D1ProtocolBError("step15 authorized payload denominator mismatch")
    if any(value != 0 for value in (missing_key_count, extra_key_count, duplicate_key_count, field_mismatch_count, test_order_mismatch_count, state_mismatch_count)):
        raise Phase4D1ProtocolBError("step15-step23 bridge mismatch")
    return {
        "authorized_step15_payloads": ("record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"),
        "bridge_row_count": len(bridge_rows),
        "metric_row_count": metric_row_count,
        "cwt_row_count": cwt_row_count,
        "bridge_sha256": bridge_hasher.hexdigest(),
        "test_record_ids_sha256": _ids_sha(test_record_ids),
        "condition_ids": tuple(config.condition_ids),
        "step15": step15,
        "step21": step21,
        "step23": step23,
        "parent_checksum_trees": {
            "step15": dict(step15["checksums"]),
            "step21": dict(step21["checksums"]),
            "step23": dict(step23["checksums"]),
        },
        "bridge_rows": tuple(bridge_rows),
        "mismatch_counts": {
            "missing_key_count": missing_key_count,
            "extra_key_count": extra_key_count,
            "duplicate_key_count": duplicate_key_count,
            "field_mismatch_count": field_mismatch_count,
            "test_order_mismatch_count": test_order_mismatch_count,
            "state_mismatch_count": state_mismatch_count,
        },
        "alpha0_reference": _real_alpha0_equivalence_contract(),
    }


@lru_cache(maxsize=1)
def _step23_shard_metadata(path: str) -> dict[str, dict[str, object]]:
    rows = _read_jsonl(Path(path) / "condition_matrix_shards.jsonl")
    if len(rows) != 66:
        raise Phase4D1ProtocolBError("step23 shard count mismatch")
    output: dict[str, dict[str, object]] = {}
    for expected_id, row in enumerate(rows):
        filename = str(row["filename"])
        if filename in output or int(row["shard_id"]) != expected_id or int(row["rows_per_shard"]) != 1000:
            raise Phase4D1ProtocolBError("step23 shard order/layout mismatch")
        target = Path(path) / filename
        expected_bytes = int(row["byte_count"])
        if target.stat().st_size != expected_bytes or _sha_file(target) != str(row["sha256"]):
            raise Phase4D1ProtocolBError(f"step23 shard checksum mismatch: {filename}")
        if expected_bytes != 41 * 1000 * 997 * 4:
            raise Phase4D1ProtocolBError("step23 shard byte layout mismatch")
        output[filename] = row
    return output


def _load_step23_condition_receipts(path: Path, condition_id: str) -> dict[tuple[str, int], dict[str, object]]:
    receipts: dict[tuple[str, int], dict[str, object]] = {}
    for row in _read_jsonl(path / "record_conditions.jsonl"):
        if str(row["condition_id"]) != condition_id:
            continue
        key = (str(row["source_split"]), int(row["source_row"]))
        if key in receipts:
            raise Phase4D1ProtocolBError("step23 duplicate condition receipt")
        receipts[key] = row
    if len(receipts) != 66000:
        raise Phase4D1ProtocolBError("step23 condition receipt denominator mismatch")
    return receipts


def _load_step23_role_occurrences(path: Path, model_seed: int) -> dict[str, list[dict[str, object]]]:
    role_rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in _read_jsonl(path / "model_role_occurrences.jsonl"):
        if int(row["model_seed"]) != model_seed:
            continue
        role_rows[str(row["role"])].append(row)
    for rows in role_rows.values():
        rows.sort(key=lambda row: int(row["role_order"]))
        if tuple(int(row["role_order"]) for row in rows) != tuple(range(len(rows))):
            raise Phase4D1ProtocolBError("step23 role order mismatch")
    return role_rows


def gather_protocol_b_role_matrices(
    config: Phase4D1ProtocolBConfig,
    bridge: Mapping[str, object],
    *,
    condition_id: str,
    model_seed: int,
) -> dict[str, object]:
    if condition_id not in config.condition_ids:
        raise Phase4D1ProtocolBError("unknown condition_id")
    if model_seed not in config.model_seeds:
        raise Phase4D1ProtocolBError("unknown model_seed")
    step23_path = Path(bridge["step23"]["path"])
    receipts = _load_step23_condition_receipts(step23_path, condition_id)
    role_occurrences = _load_step23_role_occurrences(step23_path, model_seed)
    shard_meta = _step23_shard_metadata(str(step23_path))
    memmaps: dict[str, np.memmap] = {}

    def load_row(receipt: Mapping[str, object]) -> np.ndarray:
        filename = str(receipt["filename"])
        support_point_count = int(receipt["support_point_count"])
        metadata = shard_meta.get(filename)
        if metadata is None or support_point_count != 997 or int(receipt["shard_byte_offset"]) % 4:
            raise Phase4D1ProtocolBError("step23 invalid shard receipt layout")
        if filename not in memmaps:
            memmaps[filename] = np.memmap(step23_path / filename, dtype="<f4", mode="r")
        offset = int(receipt["shard_byte_offset"]) // 4
        if offset < 0 or offset + support_point_count > memmaps[filename].size:
            raise Phase4D1ProtocolBError("step23 receipt offset outside shard")
        values = np.asarray(memmaps[filename][offset : offset + support_point_count], dtype="<f4")
        if not np.isfinite(values).all() or not np.any(values):
            raise Phase4D1ProtocolBError("step23 invalid padded/nonfinite matrix row")
        projection_sha = _array_sha(values, "<f4")
        if projection_sha != str(receipt["support_projection_sha256"]):
            raise Phase4D1ProtocolBError(f"step23 shard receipt mismatch: {condition_id}/{receipt['record_id']}")
        return values.copy()

    matrices: dict[str, np.ndarray] = {}
    role_counts: dict[str, int] = {}
    test_orders: tuple[int, ...] = ()
    test_ids: tuple[str, ...] = ()
    for role in ("train", "validation", "test"):
        rows = role_occurrences.get(role, [])
        vectors: list[np.ndarray] = []
        source_orders: list[int] = []
        record_ids: list[str] = []
        for row in rows:
            key = (str(row["source_split"]), int(row["source_row"]))
            receipt = receipts.get(key)
            if receipt is None:
                raise Phase4D1ProtocolBError(f"step23 role matrix missing receipt: {role}/{key}")
            if str(receipt["state"]) != "complete":
                raise Phase4D1ProtocolBError(f"step23 role matrix incomplete receipt: {role}/{key}")
            vectors.append(load_row(receipt))
            source_orders.append(int(row["source_row"]))
            record_ids.append(str(row["record_id"]))
        if not vectors:
            raise Phase4D1ProtocolBError(f"step23 missing role rows: {role}")
        matrices[role] = np.ascontiguousarray(np.vstack(vectors), dtype="<f4")
        role_counts[role] = len(vectors)
        if role == "test":
            test_orders = tuple(source_orders)
            test_ids = tuple(record_ids)
    expected_counts = {"train": 62700, "validation": 300, "test": 3000}
    if role_counts != expected_counts or tuple(test_orders) != tuple(range(3000)):
        raise Phase4D1ProtocolBError("step23 role cardinality/test-local-order mismatch")
    return {
        **matrices,
        "role_counts": role_counts,
        "test_orders": test_orders,
        "test_record_ids_sha256": _ids_sha(test_ids),
        "shard_count": len(shard_meta),
        "role_record_ids": {role: tuple(str(row["record_id"]) for row in role_occurrences[role]) for role in ("train", "validation", "test")},
        "role_labels": {role: tuple(int(row["class_label"]) for row in role_occurrences[role]) for role in ("train", "validation", "test")},
        "role_projection_hashes": {role: tuple(str(receipts[(str(row["source_split"]), int(row["source_row"]))]["support_projection_sha256"]) for row in role_occurrences[role]) for role in ("train", "validation", "test")},
    }


def reconstruct_d1_protocol_b_outcome_inputs(
    config: Phase4D1ProtocolBConfig,
) -> SyntheticD1ProtocolBInputs:
    if config.synthetic_fixture:
        return make_synthetic_d1_protocol_b_inputs(
            class_count=config.class_count,
            records_per_class=config.records_per_class,
            model_seeds=config.model_seeds,
            feature_count=config.feature_count,
        )
    bridge = validate_d1_protocol_b_parent_authorities(config)
    alpha0_rows = [
        row for row in bridge["bridge_rows"]
        if str(row["condition_id"]) == "alpha0"
    ]
    alpha0_rows.sort(key=lambda row: int(row["test_order"]))
    record_ids = tuple(str(row["record_id"]) for row in alpha0_rows)
    labels = np.asarray([int(row["class_label"]) for row in alpha0_rows], dtype=np.int64)
    train_by_seed_condition: dict[tuple[int, str], np.ndarray] = {}
    validation_by_seed_condition: dict[tuple[int, str], np.ndarray] = {}
    test_by_condition: dict[str, np.ndarray] = {}
    train_labels_by_seed: dict[int, np.ndarray] = {}
    validation_labels_by_seed: dict[int, np.ndarray] = {}
    step23_path = Path(bridge["step23"]["path"])
    for model_seed in config.model_seeds:
        roles = _load_step23_role_occurrences(step23_path, model_seed)
        train_labels_by_seed[model_seed] = np.asarray(
            [int(row["class_label"]) for row in roles.get("train", ())],
            dtype=np.int64,
        )
        validation_labels_by_seed[model_seed] = np.asarray(
            [int(row["class_label"]) for row in roles.get("validation", ())],
            dtype=np.int64,
        )
        for condition_id in config.condition_ids:
            gathered = gather_protocol_b_role_matrices(
                config,
                bridge,
                condition_id=condition_id,
                model_seed=model_seed,
            )
            train_by_seed_condition[(model_seed, condition_id)] = gathered["train"]
            validation_by_seed_condition[(model_seed, condition_id)] = gathered["validation"]
            if condition_id not in test_by_condition:
                test_by_condition[condition_id] = gathered["test"]
    return SyntheticD1ProtocolBInputs(
        class_count=config.class_count,
        records_per_class=config.records_per_class,
        model_seeds=config.model_seeds,
        feature_count=config.feature_count,
        record_ids=record_ids,
        labels=labels,
        train_by_seed_condition=MappingProxyType(train_by_seed_condition),
        train_labels_by_seed=MappingProxyType(train_labels_by_seed),
        validation_by_seed_condition=MappingProxyType(validation_by_seed_condition),
        validation_labels_by_seed=MappingProxyType(validation_labels_by_seed),
        test_by_condition=MappingProxyType(test_by_condition),
        synthetic_alpha0_model_rows=(),
        synthetic_alpha0_validation_rows=(),
        synthetic_alpha0_prediction_rows=(),
    )


def _render_protocol_b_figures(
    figure1_rows: Sequence[Mapping[str, object]],
    figure2_rows: Sequence[Mapping[str, object]],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    matplotlib.rcParams["svg.hashsalt"] = "rpe-phase4-d1-protocol-b-v1"
    metric_names = list(METRIC_OUTPUT_IDS)
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    metric_means = []
    for metric in metric_names:
        rows = [row for row in figure1_rows if str(row["metric_output_id"]) == metric]
        values = [float(row["harm"]) for row in rows]
        metric_means.append(float(np.mean(values)) if values else 0.0)
    axes[0].barh(np.arange(len(metric_names)), metric_means, color="#1f77b4")
    axes[0].set_yticks(np.arange(len(metric_names)), metric_names)
    axes[0].set_title("Mean Harm")

    condition_names = [str(row.get("metric_output_id", "")) for row in figure2_rows]
    ag_values = [0.0 if row.get("ag") is None else float(row["ag"]) for row in figure2_rows]
    axes[1].barh(np.arange(len(condition_names)), ag_values, color="#2ca02c")
    axes[1].set_yticks(np.arange(len(condition_names)), condition_names)
    axes[1].set_title("Alignment Gap")
    figure.tight_layout()
    for kind in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
        payloads[f"figure1_d1_protocol_b_full_domain.{kind}"] = stream.getvalue()
    plt.close(figure)

    figure, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
    for axis, field, color in zip(
        axes,
        ("ag", "acc_cross", "d_ag", "d_acc"),
        ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"),
        strict=True,
    ):
        values = [0.0 if row.get(field) is None else float(row[field]) for row in figure2_rows]
        axis.barh(np.arange(len(figure2_rows)), values, color=color)
        axis.set_title(field)
        axis.set_yticks(np.arange(len(figure2_rows)), condition_names if axis is axes[0] else [])
    figure.tight_layout()
    for kind in ("png", "svg"):
        stream = io.BytesIO()
        figure.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
        payloads[f"figure2_d1_protocol_b_full_domain.{kind}"] = stream.getvalue()
    plt.close(figure)
    return payloads


def _build_synthetic_payloads(
    inputs: SyntheticD1ProtocolBInputs,
    config: Phase4D1ProtocolBConfig,
    *,
    worker_count: int,
    bootstrap_resamples: int,
    sign_flip_resamples: int,
) -> tuple[dict[str, bytes], int, int]:
    model_rows = []
    validation_rows = []
    prediction_rows = []
    for condition_id in config.condition_ids:
        for model_seed in config.model_seeds:
            train = inputs.train_by_seed_condition[(model_seed, condition_id)]
            validation = inputs.validation_by_seed_condition[(model_seed, condition_id)]
            model_cell, scores, predictions = _fit_predict_single_condition(
                model_seed,
                condition_id,
                train,
                inputs.train_labels_by_seed[model_seed],
                validation,
                inputs.validation_labels_by_seed[model_seed],
                inputs.test_by_condition[condition_id],
                inputs.labels,
                inputs.record_ids,
            )
            model_rows.append(model_cell)
            validation_rows.extend(scores)
            prediction_rows.extend(predictions)
    alpha0_models = tuple(row for row in model_rows if row["condition_id"] == "alpha0")
    alpha0_validation = tuple(row for row in validation_rows if row["condition_id"] == "alpha0")
    alpha0_predictions = tuple(row for row in prediction_rows if row["condition_id"] == "alpha0")
    alpha0_receipt = validate_alpha0_equivalence(alpha0_models, alpha0_validation, alpha0_predictions, config)

    seed_class_conditions = []
    class_observations = []
    alignment_results = []
    bootstrap_results = []
    sign_flip_results = []
    holm_family = []
    figure1_rows = []
    figure2_rows = []
    target_class_observation_count = inputs.class_count * len(METRIC_OUTPUT_IDS) * (len(ALPHAS) - 1)
    for condition_id in config.condition_ids:
        if condition_id == "alpha0":
            continue
        for class_label in range(inputs.class_count):
            class_predictions = [
                row for row in prediction_rows
                if row["condition_id"] == condition_id and row["true_class"] == class_label
            ]
            accuracy = float(sum(1 for row in class_predictions if row["correct"]) / len(class_predictions))
            for model_seed in config.model_seeds:
                seed_class_conditions.append(
                    {
                        "condition_id": condition_id,
                        "model_seed": model_seed,
                        "class_label": class_label,
                        "accuracy_seed_class": accuracy,
                    }
                )
            if len(class_observations) >= target_class_observation_count:
                continue
            for metric_index, metric_output_id in enumerate(METRIC_OUTPUT_IDS):
                if len(class_observations) >= target_class_observation_count:
                    break
                class_observations.append(
                    {
                        "class_label": class_label,
                        "condition_id": condition_id,
                        "metric_output_id": metric_output_id,
                        "harm": float((metric_index + 1) * 0.01 + class_label * 0.001),
                    }
                )
                figure1_rows.append(
                    {
                        "class_label": class_label,
                        "condition_id": condition_id,
                        "metric_output_id": metric_output_id,
                        "harm": float((metric_index + 1) * 0.01 + class_label * 0.001),
                    }
                )
    for metric_output_id in METRIC_OUTPUT_IDS:
        alignment_results.append(
            {
                "metric_output_id": metric_output_id,
                "ag": 0.25 if metric_output_id == "mse" else 0.2,
                "acc_cross": 0.75,
                "state": "evaluable",
            }
        )
        figure2_rows.append(
            {
                "metric_output_id": metric_output_id,
                "ag": 0.25 if metric_output_id == "mse" else 0.2,
                "acc_cross": 0.75,
                "d_ag": 0.0 if metric_output_id == "mse" else 0.05,
                "d_acc": 0.0 if metric_output_id == "mse" else -0.01,
            }
        )
        if metric_output_id != "mse":
            bootstrap_results.append(
                {
                    "metric_output_id": metric_output_id,
                    "interval_low": 0.01,
                    "interval_high": 0.09,
                    "resamples": bootstrap_resamples,
                }
            )
    candidate_metrics = [metric for metric in METRIC_OUTPUT_IDS if metric != "mse"][:12]
    for metric_output_id in candidate_metrics:
        sign_flip_results.extend(
            [
                {"metric_output_id": metric_output_id, "contrast": "d_ag", "p_value": 0.5, "resamples": sign_flip_resamples},
                {"metric_output_id": metric_output_id, "contrast": "d_acc", "p_value": 0.5, "resamples": sign_flip_resamples},
            ]
        )
        holm_family.extend(
            [
                {"metric_output_id": metric_output_id, "contrast": "d_ag", "adjusted_p_value": 1.0, "family_id": "d1_protocol_b_full_domain_secondary_24"},
                {"metric_output_id": metric_output_id, "contrast": "d_acc", "adjusted_p_value": 1.0, "family_id": "d1_protocol_b_full_domain_secondary_24"},
            ]
        )
    condition_summary = []
    for condition_id in config.condition_ids:
        condition_predictions = [row for row in prediction_rows if row["condition_id"] == condition_id]
        micro = float(sum(1 for row in condition_predictions if row["correct"]) / len(condition_predictions))
        condition_summary.append(
            {
                "condition_id": condition_id,
                "macro_top1_accuracy": micro,
                "micro_top1_accuracy": micro,
                "macro_f1": micro,
            }
        )
    secondary_table = tuple(figure2_rows)
    rendered_figures = _render_protocol_b_figures(figure1_rows, figure2_rows)
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "condition_count": len(config.condition_ids),
        "model_seed_count": len(config.model_seeds),
        "prediction_row_count": len(prediction_rows),
        "class_observation_count": len(class_observations),
        "alpha0_equivalence": alpha0_receipt,
    }
    payloads = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": canonical_json_bytes(
            {
                "parent_artifacts": config.document["parent_artifacts"],
                "authorized_step15_payloads": ["record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"],
            }
        ),
        "preflight.json": canonical_json_bytes(
            {
                "worker_count": int(worker_count),
                "process_start_method": "spawn",
                "blas_thread_limit": 1,
                "worker_count_limit": len(config.model_seeds),
                "admission_bytes_per_job": WORKER_ADMISSION_BYTES,
                "admission_budget_bytes": ADMISSION_BUDGET_BYTES,
                "configured_payload_count": len(ARTIFACT_PAYLOAD_FILES),
            }
        ),
        "alpha0_equivalence.json": canonical_json_bytes(alpha0_receipt),
        "model_cells.jsonl": jsonl_bytes(tuple(model_rows)),
        "validation_scores.jsonl": jsonl_bytes(tuple(validation_rows)),
        "predictions.jsonl": jsonl_bytes(tuple(prediction_rows)),
        "seed_class_conditions.jsonl": jsonl_bytes(tuple(seed_class_conditions)),
        "condition_summary.csv": csv_bytes(tuple(condition_summary)),
        "class_observations.jsonl": jsonl_bytes(tuple(class_observations)),
        "alignment_results.jsonl": jsonl_bytes(tuple(alignment_results)),
        "bootstrap_results.jsonl": jsonl_bytes(tuple(bootstrap_results)),
        "sign_flip_results.jsonl": jsonl_bytes(tuple(sign_flip_results)),
        "holm_family.jsonl": jsonl_bytes(tuple(holm_family)),
        "figure1_d1_protocol_b_full_domain.png": rendered_figures["figure1_d1_protocol_b_full_domain.png"],
        "figure1_d1_protocol_b_full_domain.svg": rendered_figures["figure1_d1_protocol_b_full_domain.svg"],
        "figure1_d1_protocol_b_full_domain_data.csv": csv_bytes(tuple(figure1_rows)),
        "figure2_d1_protocol_b_full_domain.png": rendered_figures["figure2_d1_protocol_b_full_domain.png"],
        "figure2_d1_protocol_b_full_domain.svg": rendered_figures["figure2_d1_protocol_b_full_domain.svg"],
        "figure2_d1_protocol_b_full_domain_data.csv": csv_bytes(tuple(figure2_rows)),
        "d1_protocol_b_full_domain_secondary_table.csv": csv_bytes(secondary_table),
        "manifest.json": canonical_json_bytes(manifest),
    }
    return payloads, len(prediction_rows), len(class_observations)


def _real_metric_rows(bridge: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    rows = _read_jsonl(Path(bridge["step15"]["path"]) / "metric_values.jsonl")
    permitted = {(str(row["record_id"]), str(row["condition_id"])) for row in bridge["bridge_rows"]}
    selected = [row for row in rows if (str(row["record_id"]), str(row["condition_id"])) in permitted]
    if len(selected) != 1_599_000:
        raise Phase4D1ProtocolBError("authorized Step15 metric selection mismatch")
    return tuple(selected)


def _aggregate_real_outcome(
    predictions: Sequence[Mapping[str, object]], metric_values: Sequence[Mapping[str, object]],
    config: Phase4D1ProtocolBConfig, *, bootstrap_resamples: int, sign_flip_resamples: int,
) -> Mapping[str, tuple[Mapping[str, object], ...]]:
    classes = tuple(range(config.class_count)); positive = config.condition_ids[1:]
    per_seed: dict[tuple[int, int, str], list[bool]] = defaultdict(list)
    class_orders: dict[int, set[int]] = defaultdict(set)
    for row in predictions:
        per_seed[(int(row["model_seed"]), int(row["true_class"]), str(row["condition_id"]))].append(bool(row["correct"]))
        class_orders[int(row["true_class"])].add(int(row["record_order"]))
    required = {(seed, label, condition) for seed in config.model_seeds for label in classes for condition in config.condition_ids}
    if set(per_seed) != required or any(len(values) != config.records_per_class for values in per_seed.values()):
        raise Phase4D1ProtocolBError("prediction seed/class/condition denominator mismatch")
    accuracy = {key: float(np.mean(value)) for key, value in per_seed.items()}
    metric_map: dict[tuple[int, str, str], float] = {}
    for row in metric_values:
        key = (int(row["record_order"]), str(row["condition_id"]), str(row["metric_output_id"]))
        if key in metric_map or row.get("state") != "complete": raise Phase4D1ProtocolBError("metric canonical schema mismatch")
        metric_map[key] = float(row["value"])
    observations: list[Mapping[str, object]] = []; tables: dict[str, tuple[AlignmentObservation, ...] | None] = {}; states: dict[str, str] = {}
    for metric in METRIC_OUTPUT_IDS:
        table: list[AlignmentObservation] = []; complete = True
        for condition in positive:
            perturbation, encoded = condition.split(":", 1); alpha = float(np.frombuffer(bytes.fromhex(encoded), dtype="<f8")[0])
            for label in classes:
                orders = sorted(class_orders[label]); baseline = [metric_map.get((order, "alpha0", metric)) for order in orders]; current = [metric_map.get((order, condition, metric)) for order in orders]
                if None in baseline or None in current: complete = False; harm = None
                else:
                    delta = [now - old for old, now in zip(baseline, current, strict=True)]
                    harm = float(np.mean(delta if METRIC_DIRECTIONS[metric] == "lower_is_better" else [-item for item in delta]))
                downstream = float(
                    np.mean([accuracy[(seed, label, "alpha0")] for seed in config.model_seeds])
                    - np.mean([accuracy[(seed, label, condition)] for seed in config.model_seeds])
                )
                observations.append({"metric_output_id": metric, "class_label": label, "condition_id": condition, "perturbation_id": perturbation, "alpha": alpha, "metric_harm": harm, "downstream_harm": downstream, "state": "complete" if harm is not None else "not_evaluable_metric_incomplete"})
                if harm is not None: table.append(AlignmentObservation(str(label), perturbation, alpha, harm, downstream))
        if complete:
            try: alignment_gap(table); cross_perturbation_accuracy(table); tables[metric] = tuple(table); states[metric] = "complete"
            except AlignmentValidationError as error:
                if error.path != "constant downstream": raise
                tables[metric] = None; states[metric] = "not_evaluable_constant_downstream"
        else: tables[metric] = None; states[metric] = "not_evaluable_metric_incomplete"
    alignment: list[Mapping[str, object]]=[]; bootstrap: list[Mapping[str, object]]=[]; sign: list[Mapping[str, object]]=[]; holm: list[Mapping[str, object]]=[]
    reference = tables["mse"]; pvalues: dict[str, float]={}; contrasts: dict[str, float]={}; comparisons: dict[str, object]={}; boots: dict[str, object]={}
    if reference is None: raise Phase4D1ProtocolBError("MSE alignment is not evaluable")
    ref_gap, ref_acc = alignment_gap(reference), cross_perturbation_accuracy(reference); ref_boot = bulk_paired_cluster_bootstrap(reference, reference, resamples=bootstrap_resamples, random_seed=20260817)
    alignment.append({"metric_output_id":"mse","state":"complete","ag":ref_gap.alignment_gap,"ag_raw":ref_gap.raw_alignment_gap,"ag_interval":ref_boot.reference_ag_interval,"acc_cross":ref_acc.accuracy,"acc_interval":ref_boot.reference_acc_interval,"d_ag":None,"d_ag_interval":None,"d_acc":None,"d_acc_interval":None})
    for metric in METRIC_OUTPUT_IDS[1:]:
        candidate=tables[metric]
        if candidate is None:
            bootstrap.append({"metric_output_id":metric,"state":states[metric],"resamples":None})
            for statistic in ("d_ag","d_acc"): pvalues[f"{metric}:{statistic}"]=1.0; sign.append({"metric_output_id":metric,"statistic":statistic,"state":"not_tested_metric_incomplete","contrast":None,"p_value":1.0,"resamples":None})
            alignment.append({"metric_output_id":metric,"state":states[metric],"ag":None,"ag_raw":None,"ag_interval":None,"acc_cross":None,"acc_interval":None,"d_ag":None,"d_ag_interval":None,"d_acc":None,"d_acc_interval":None}); continue
        comparison=compare_alignment(reference,candidate); boot=bulk_paired_cluster_bootstrap(reference,candidate,resamples=bootstrap_resamples,random_seed=20260817); comparisons[metric]=comparison; boots[metric]=boot
        bootstrap.append({"metric_output_id":metric,"state":"complete","resamples":bootstrap_resamples,"candidate_ag_interval":boot.candidate_ag_interval,"candidate_acc_interval":boot.candidate_acc_interval,"d_ag_interval":boot.d_ag_interval,"d_acc_interval":boot.d_acc_interval})
        alignment.append({"metric_output_id":metric,"state":"complete","ag":comparison.candidate_gap.alignment_gap,"ag_raw":comparison.candidate_gap.raw_alignment_gap,"ag_interval":boot.candidate_ag_interval,"acc_cross":comparison.candidate_accuracy.accuracy,"acc_interval":boot.candidate_acc_interval,"d_ag":comparison.d_ag,"d_ag_interval":boot.d_ag_interval,"d_acc":comparison.d_acc,"d_acc_interval":boot.d_acc_interval})
        for statistic, values, reduction in (("d_ag",comparison.ag_contribution_differences,"sum"),("d_acc",comparison.acc_contribution_differences,"mean")):
            outcome=paired_contribution_sign_flip([item.value for item in values],aggregation=reduction,resamples=sign_flip_resamples,random_seed=20260817); key=f"{metric}:{statistic}"; pvalues[key]=outcome.p_value; contrasts[key]=getattr(comparison,statistic); sign.append({"metric_output_id":metric,"statistic":statistic,"state":"complete","contrast":contrasts[key],"p_value":outcome.p_value,"resamples":sign_flip_resamples})
    for result in holm_step_down(pvalues,alpha=0.05):
        metric, statistic=result.hypothesis_id.split(":",1); favorable=contrasts.get(result.hypothesis_id,0.0)>0; holm.append({"metric_output_id":metric,"statistic":statistic,"raw_p_value":result.raw_p_value,"adjusted_p_value":result.adjusted_p_value,"rank":result.rank,"family_size":result.family_size,"family_state":"complete" if result.hypothesis_id in contrasts else "not_tested_metric_incomplete","favorable":favorable,"rejected":bool(result.rejected and favorable)})
    if (len(observations),len(alignment),len(bootstrap),len(sign),len(holm)) != (15600,13,12,24,24): raise Phase4D1ProtocolBError("aggregation artifact denominator mismatch")
    return MappingProxyType({"class_observations":tuple(observations),"alignment_results":tuple(alignment),"bootstrap_results":tuple(bootstrap),"sign_flip_results":tuple(sign),"holm_family":tuple(holm),"seed_accuracy":MappingProxyType(accuracy)})


def _real_condition_summary(predictions: Sequence[Mapping[str, object]], config: Phase4D1ProtocolBConfig) -> tuple[Mapping[str, object], ...]:
    rows=[]
    for condition in config.condition_ids:
        selected=[row for row in predictions if row["condition_id"] == condition]
        if len(selected) != 15000: raise Phase4D1ProtocolBError("condition summary denominator mismatch")
        f1=[]
        for label in range(config.class_count):
            tp=sum(int(row["true_class"]) == label and int(row["predicted_class"]) == label for row in selected); fp=sum(int(row["true_class"]) != label and int(row["predicted_class"]) == label for row in selected); fn=sum(int(row["true_class"]) == label and int(row["predicted_class"]) != label for row in selected)
            f1.append(0.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
        per_class=[float(np.mean([row["correct"] for row in selected if int(row["true_class"]) == label])) for label in range(config.class_count)]
        rows.append({"condition_id":condition,"prediction_count":len(selected),"macro_top1_accuracy":float(np.mean(per_class)),"micro_top1_accuracy":float(np.mean([row["correct"] for row in selected])),"macro_f1":float(np.mean(f1))})
    return tuple(rows)


def _index_step23_role_ledger(path: Path) -> dict[int, dict[str, list[dict[str, object]]]]:
    indexed: dict[int, dict[str, list[dict[str, object]]]] = {seed: defaultdict(list) for seed in MODEL_SEEDS}
    for row in _read_jsonl(path / "model_role_occurrences.jsonl"):
        seed = int(row["model_seed"]); role = str(row["role"])
        if seed not in indexed or role not in ("train", "validation", "test"):
            raise Phase4D1ProtocolBError("step23 role ledger schema mismatch")
        indexed[seed][role].append(row)
    expected = {"train": 62700, "validation": 300, "test": 3000}
    for seed, roles in indexed.items():
        if set(roles) != set(expected): raise Phase4D1ProtocolBError("step23 missing role ledger")
        for role, count in expected.items():
            roles[role].sort(key=lambda row: int(row["role_order"]))
            if len(roles[role]) != count or [int(row["role_order"]) for row in roles[role]] != list(range(count)):
                raise Phase4D1ProtocolBError("step23 role-order ledger mismatch")
    return indexed


def _index_step23_condition_ledger(path: Path, config: Phase4D1ProtocolBConfig) -> dict[str, dict[tuple[str, int], dict[str, object]]]:
    indexed = {condition: {} for condition in config.condition_ids}
    for row in _read_jsonl(path / "record_conditions.jsonl"):
        condition = str(row["condition_id"]); key = (str(row["source_split"]), int(row["source_row"]))
        if condition not in indexed or key in indexed[condition]: raise Phase4D1ProtocolBError("step23 condition ledger schema mismatch")
        indexed[condition][key] = row
    if any(len(rows) != 66000 for rows in indexed.values()): raise Phase4D1ProtocolBError("step23 indexed receipt denominator mismatch")
    return indexed


def _load_indexed_condition_block(path: Path, receipts: Mapping[tuple[str, int], Mapping[str, object]]) -> dict[tuple[str, int], np.ndarray]:
    metadata = _step23_shard_metadata(str(path)); blocks: dict[str, np.memmap] = {}; rows: dict[tuple[str, int], np.ndarray] = {}
    for key, receipt in receipts.items():
        filename=str(receipt["filename"]); offset=int(receipt["shard_byte_offset"]); points=int(receipt["support_point_count"])
        if filename not in metadata or points != 997 or offset % 4: raise Phase4D1ProtocolBError("indexed receipt layout mismatch")
        if filename not in blocks: blocks[filename]=np.memmap(path / filename,dtype="<f4",mode="r")
        start=offset//4; values=np.asarray(blocks[filename][start:start+points],dtype="<f4")
        if len(values) != points or not np.isfinite(values).all() or not np.any(values) or _array_sha(values,"<f4") != str(receipt["support_projection_sha256"]):
            raise Phase4D1ProtocolBError("indexed condition receipt hash/state mismatch")
        if str(receipt["state"]) != "complete": raise Phase4D1ProtocolBError("indexed condition incomplete receipt")
        rows[key]=values.copy()
    if len(rows) != 66000: raise Phase4D1ProtocolBError("indexed condition block cardinality mismatch")
    return rows


def _indexed_seed_job(job: tuple[int, str, dict[tuple[str, int], np.ndarray], dict[str, list[dict[str, object]]], str, str]) -> tuple[int, dict[str, object], tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
    seed, condition, source_rows, roles, code_identity, config_sha = job
    with threadpool_limits(limits=1, user_api="blas"):
        assembled={}
        for role in ("train","validation","test"):
            ordered=roles[role]; keys=[(str(row["source_split"]),int(row["source_row"])) for row in ordered]
            assembled[role]=np.ascontiguousarray(np.vstack([source_rows[key] for key in keys]),dtype="<f4")
        cell,scores,predictions=_fit_predict_single_condition(seed,condition,assembled["train"],np.asarray([row["class_label"] for row in roles["train"]],dtype=np.int64),assembled["validation"],np.asarray([row["class_label"] for row in roles["validation"]],dtype=np.int64),assembled["test"],np.asarray([row["class_label"] for row in roles["test"]],dtype=np.int64),tuple(str(row["record_id"]) for row in roles["test"]),train_record_ids=tuple(str(row["record_id"]) for row in roles["train"]),validation_record_ids=tuple(str(row["record_id"]) for row in roles["validation"]),projected_row_hashes=tuple(_array_sha(source_rows[(str(row["source_split"]),int(row["source_row"]))],"<f4") for row in roles["test"]),code_identity=code_identity,config_sha256=config_sha,canonical_model_state=True)
    return seed,cell,scores,predictions


def _real_alpha0_receipt(
    current: Mapping[str, str],
    parent: Mapping[str, str],
    expected: Mapping[str, str],
) -> Mapping[str, object]:
    fields = ("model_digest", "validation_digest", "prediction_digest")
    if any(set(source) != set(fields) for source in (current, parent, expected)):
        raise Phase4D1ProtocolBError("alpha0 receipt schema mismatch")
    mismatches = {
        field: int(current[field] != parent[field] or current[field] != expected[field])
        for field in fields
    }
    return MappingProxyType({
        **{field: current[field] for field in fields},
        "parent_digests": {field: parent[field] for field in fields},
        "expected_digests": {field: expected[field] for field in fields},
        "mismatch_counts": mismatches,
        "mismatch_count": sum(mismatches.values()),
        "status": "failed" if any(mismatches.values()) else "passed",
    })


def _alpha0_failure_receipt(alpha_receipt: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "state": "failed_alpha0_equivalence",
            "condition_id": "alpha0",
            "model_seed": None,
            "c": None,
            "current_digests": {
                "model_digest": str(alpha_receipt["model_digest"]),
                "validation_digest": str(alpha_receipt["validation_digest"]),
                "prediction_digest": str(alpha_receipt["prediction_digest"]),
            },
            "parent_digests": dict(alpha_receipt["parent_digests"]),
            "expected_digests": dict(alpha_receipt["expected_digests"]),
            "mismatch_counts": dict(alpha_receipt["mismatch_counts"]),
            "mismatch_count": int(alpha_receipt["mismatch_count"]),
        }
    )


def _real_artifact_metadata(
    config: Mapping[str, object],
    bridge: Mapping[str, object],
    alpha0_receipt: Mapping[str, object],
    matrix_receipt_sha256: str,
    projection: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[str, Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    run_id = _real_run_id(config, bridge)
    checksum_trees = {
        name: dict(tree)
        for name, tree in dict(bridge["parent_checksum_trees"]).items()
    }
    authorized = ["record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"]
    forbidden = sorted(set(checksum_trees["step15"]) - set(authorized))
    bridge_doc = {
        "parent_artifacts": config["parent_artifacts"],
        "parent_checksum_trees": checksum_trees,
        "authorized_step15_payloads": authorized,
        "forbidden_step15_payloads": forbidden,
        "condition_bridge_sha256": bridge["bridge_sha256"],
        "test_record_ids_sha256": bridge["test_record_ids_sha256"],
        "condition_ids": list(bridge["condition_ids"]),
        "bridge_row_count": int(bridge["bridge_row_count"]),
        "metric_row_count": int(bridge["metric_row_count"]),
        "cwt_row_count": int(bridge["cwt_row_count"]),
        "mismatch_counts": dict(bridge["mismatch_counts"]),
        "alpha0_qc_scope": "post_computation_step21_model_validation_prediction_projection_only",
    }
    preflight = {
        "status": "complete",
        "authority_bridge_state": "complete",
        "matrix_input_state": "ready",
        "row_hash_state": "complete",
        "offset_state": "complete",
        "production_worker_count": 5,
        "verifier_worker_count": 4,
        "process_start_method": "spawn",
        "blas_thread_limit": 1,
        "admission_bytes_per_job": WORKER_ADMISSION_BYTES,
        "admission_budget_bytes": ADMISSION_BUDGET_BYTES,
        "condition_count": 41,
        "source_record_count": 66_000,
        "role_occurrence_count": 330_000,
        "condition_receipt_count": 2_706_000,
        "condition_matrix_shard_count": 66,
        "condition_matrix_bytes": 10_791_528_000,
        "matrix_receipt_sha256": matrix_receipt_sha256,
        "configured_payload_count": 22,
    }
    counts = _default_artifact_counts()
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "complete",
        "experiment_id": EXPERIMENT_ID,
        "protocol": "B",
        "tier": "full_domain_core",
        "claim_boundary": CLAIM_BOUNDARY,
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "condition_count": 41,
        "model_seed_count": 5,
        "prediction_row_count": 615_000,
        "class_observation_count": 15_600,
        "counts": counts,
        "metric_states": {
            row["metric_output_id"]: row["state"]
            for row in projection["alignment_results"]
        },
        "alpha0_equivalence": dict(alpha0_receipt),
        "authorities": config["authorities"],
        "parent_artifacts": config["parent_artifacts"],
        "code_authority": config["code_authority"],
        "environment_authority": config["environment_authority"],
        "inherited_rulings": config["inherited_rulings"],
    }
    return run_id, MappingProxyType(bridge_doc), MappingProxyType(preflight), MappingProxyType(manifest)


def _completed_model_cell(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized["state"] = "complete"
    return normalized


def _completed_validation_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized["state"] = "complete"
    return normalized


def _completed_prediction_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized["state"] = "complete"
    return normalized


def _completed_seed_class_row(row: Mapping[str, object]) -> dict[str, object]:
    normalized = dict(row)
    normalized["state"] = "complete"
    return normalized


def _completed_condition_summary_row(
    predictions: Sequence[Mapping[str, object]],
    *,
    condition_id: str,
    class_count: int,
) -> dict[str, object]:
    f1_scores = []
    for label in range(class_count):
        tp = sum(
            int(row["true_class"]) == label and int(row["predicted_class"]) == label
            for row in predictions
        )
        fp = sum(
            int(row["true_class"]) != label and int(row["predicted_class"]) == label
            for row in predictions
        )
        fn = sum(
            int(row["true_class"]) == label and int(row["predicted_class"]) != label
            for row in predictions
        )
        f1_scores.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    per_class = [
        float(np.mean([row["correct"] for row in predictions if int(row["true_class"]) == label]))
        for label in range(class_count)
    ]
    return {
        "condition_id": condition_id,
        "prediction_count": len(predictions),
        "macro_top1_accuracy": float(np.mean(per_class)),
        "micro_top1_accuracy": float(np.mean([row["correct"] for row in predictions])),
        "macro_f1": float(np.mean(f1_scores)),
        "state": "complete",
    }


def _null_model_cell(
    *,
    condition_id: str,
    model_seed: int,
    state: str,
) -> dict[str, object]:
    return {
        "model_seed": int(model_seed),
        "condition_id": condition_id,
        "train_condition_id": condition_id,
        "validation_condition_id": condition_id,
        "test_condition_id": condition_id,
        "selected_c": None,
        "validation_scores": None,
        "train_matrix_sha256": None,
        "validation_matrix_sha256": None,
        "pca_train_feature_sha256": None,
        "pca_validation_feature_sha256": None,
        "model_state_sha256": None,
        "train_record_ids_sha256": None,
        "validation_record_ids_sha256": None,
        "test_record_ids_sha256": None,
        "warning_state": state,
        "refit_with_validation": False,
        "state": state,
    }


def _null_validation_row(
    *,
    condition_id: str,
    model_seed: int,
    c: float,
    state: str,
) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "c": float(c),
        "model_seed": int(model_seed),
        "validation_top1_accuracy": None,
        "state": state,
    }


def _null_prediction_row(
    *,
    condition_id: str,
    model_seed: int,
    record_id: str,
    record_order: int,
    true_class: int,
    code_identity: str,
    config_sha256: str,
    state: str,
) -> dict[str, object]:
    return {
        "code_identity": code_identity,
        "condition_id": condition_id,
        "config_sha256": config_sha256,
        "correct": None,
        "model_identity": f"{model_seed}:{condition_id}",
        "model_seed": int(model_seed),
        "predicted_class": None,
        "projected_row_sha256": None,
        "record_id": record_id,
        "record_order": int(record_order),
        "true_class": int(true_class),
        "state": state,
    }


def _null_seed_class_row(
    *,
    condition_id: str,
    model_seed: int,
    class_label: int,
    state: str,
) -> dict[str, object]:
    return {
        "model_seed": int(model_seed),
        "class_label": int(class_label),
        "condition_id": condition_id,
        "accuracy_seed_class": None,
        "state": state,
    }


def _null_condition_summary_row(condition_id: str, state: str) -> dict[str, object]:
    return {
        "condition_id": condition_id,
        "prediction_count": None,
        "macro_top1_accuracy": None,
        "micro_top1_accuracy": None,
        "macro_f1": None,
        "state": state,
    }


def _failed_figure1_rows(
    failure_receipt: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    endpoint_state = str(failure_receipt["state"])
    failed_condition_id = str(failure_receipt["condition_id"])
    rows = []
    for metric_output_id in METRIC_OUTPUT_IDS:
        for perturbation_id in PERTURBATIONS:
            for alpha in ALPHAS[1:]:
                condition_id = f"{perturbation_id}:{np.float64(alpha).tobytes().hex()}"
                rows.append(
                    {
                        "metric_output_id": metric_output_id,
                        "perturbation_id": perturbation_id,
                        "alpha": float(alpha),
                        "mean_metric_harm": None,
                        "mean_downstream_harm": None,
                        "state": (
                            "failed_model_lifecycle"
                            if endpoint_state == "failed_model_lifecycle" and condition_id == failed_condition_id
                            else "not_tested_endpoint_closed"
                        ),
                    }
                )
    return tuple(rows)


def _failed_figure2_rows() -> tuple[Mapping[str, object], ...]:
    return tuple(
        {
            "metric_output_id": metric_output_id,
            "ag": None,
            "ag_raw": None,
            "ag_interval": None,
            "acc_cross": None,
            "acc_interval": None,
            "d_ag": None,
            "d_ag_interval": None,
            "d_acc": None,
            "d_acc_interval": None,
            "d_ag_raw_p": None,
            "d_ag_adjusted_p": None,
            "d_ag_rank": None,
            "d_ag_favorable": None,
            "d_ag_rejected": None,
            "d_acc_raw_p": None,
            "d_acc_adjusted_p": None,
            "d_acc_rank": None,
            "d_acc_favorable": None,
            "d_acc_rejected": None,
            "state": "not_tested_endpoint_closed",
        }
        for metric_output_id in METRIC_OUTPUT_IDS
    )


def _render_failed_protocol_b_figures(
    figure1_rows: Sequence[Mapping[str, object]],
    figure2_rows: Sequence[Mapping[str, object]],
    failure_receipt: Mapping[str, object],
) -> Mapping[str, bytes]:
    payloads: dict[str, bytes] = {}
    state_order = tuple(
        state
        for state in ("failed_alpha0_equivalence", "failed_model_lifecycle", "not_tested_endpoint_closed")
        if state == "not_tested_endpoint_closed"
        or any(str(row["state"]) == state for row in figure1_rows)
    )
    state_counts_figure1 = Counter(str(row["state"]) for row in figure1_rows)
    grey_palette = {
        "failed_alpha0_equivalence": "#6f6f6f",
        "failed_model_lifecycle": "#6f6f6f",
        "not_tested_endpoint_closed": "#b8b8b8",
    }
    failure_lines = ["Protocol B endpoint closed", f"state={failure_receipt['state']}"]
    for key in ("condition_id", "model_seed", "c", "warning_category", "warning_message"):
        if key in failure_receipt and failure_receipt[key] is not None:
            failure_lines.append(f"{key}={failure_receipt[key]}")
    with matplotlib.rc_context(
        {
            "font.family": "DejaVu Sans",
            "lines.linewidth": 1.5,
            "svg.hashsalt": "rpe-phase4-d1-protocol-b-v1",
        }
    ):
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        flat = axes.ravel()
        flat[0].barh(
            np.arange(len(state_order)),
            [state_counts_figure1.get(state, 0) for state in state_order],
            color=[grey_palette[state] for state in state_order],
        )
        flat[0].set_yticks(np.arange(len(state_order)), state_order)
        flat[0].set_title("figure1 csv states")
        flat[1].axis("off")
        flat[1].text(
            0.0,
            1.0,
            "\n".join(failure_lines),
            va="top",
            ha="left",
            fontsize=10,
            wrap=True,
            color="#4f4f4f",
        )
        for axis in flat[2:]:
            axis.set_axis_off()
        fig.tight_layout(pad=1.05)
        for kind in ("png", "svg"):
            stream = io.BytesIO()
            fig.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
            payloads[f"figure1_d1_protocol_b_full_domain.{kind}"] = stream.getvalue()
        plt.close(fig)

        fig, axes = plt.subplots(1, 4, figsize=(14, 8), sharey=True)
        metric_names = [str(row["metric_output_id"]) for row in figure2_rows]
        values = [len(metric_names)] * len(metric_names)
        for axis, title in zip(axes, ("ag", "acc_cross", "d_ag", "d_acc"), strict=True):
            axis.barh(np.arange(len(metric_names)), values, color="#b8b8b8")
            axis.set_title(title)
            axis.set_yticks(np.arange(len(metric_names)), metric_names if axis is axes[0] else [])
            axis.margins(x=0.05, y=0.05)
        fig.tight_layout(pad=1.05)
        for kind in ("png", "svg"):
            stream = io.BytesIO()
            fig.savefig(stream, format=kind, dpi=300, metadata={"Date": None, "Creator": "raman-preproc-eval"})
            payloads[f"figure2_d1_protocol_b_full_domain.{kind}"] = stream.getvalue()
        plt.close(fig)
    return MappingProxyType(payloads)


def _build_real_failure_payloads(
    config: Phase4D1ProtocolBConfig,
    bridge: Mapping[str, object],
    *,
    alpha0_receipt: Mapping[str, object],
    alpha0_model_rows: Sequence[Mapping[str, object]],
    alpha0_validation_rows: Sequence[Mapping[str, object]],
    alpha0_prediction_rows: Sequence[Mapping[str, object]],
    alpha0_seed_class_rows: Sequence[Mapping[str, object]],
    base_test_rows: Sequence[Mapping[str, object]],
    code_identity: str,
    failure_receipt: Mapping[str, object],
) -> tuple[str, dict[str, bytes]]:
    run_id = _real_run_id(config.document, bridge)
    endpoint_state = str(failure_receipt["state"])
    checksum_trees = {
        name: dict(tree)
        for name, tree in dict(bridge["parent_checksum_trees"]).items()
    }
    authorized = ["record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"]
    forbidden = sorted(set(checksum_trees["step15"]) - set(authorized))
    bridge_doc = {
        "parent_artifacts": config.document["parent_artifacts"],
        "parent_checksum_trees": checksum_trees,
        "authorized_step15_payloads": authorized,
        "forbidden_step15_payloads": forbidden,
        "condition_bridge_sha256": bridge["bridge_sha256"],
        "test_record_ids_sha256": bridge["test_record_ids_sha256"],
        "condition_ids": list(bridge["condition_ids"]),
        "bridge_row_count": int(bridge["bridge_row_count"]),
        "metric_row_count": int(bridge["metric_row_count"]),
        "cwt_row_count": int(bridge["cwt_row_count"]),
        "mismatch_counts": dict(bridge["mismatch_counts"]),
        "alpha0_qc_scope": "post_computation_step21_model_validation_prediction_projection_only",
    }
    failed_condition_id = str(failure_receipt["condition_id"])
    failed_model_seed = (
        None if failure_receipt.get("model_seed") is None else int(failure_receipt["model_seed"])
    )
    failed_c = None if failure_receipt.get("c") is None else float(failure_receipt["c"])
    model_rows = [_completed_model_cell(row) for row in alpha0_model_rows]
    validation_rows = [_completed_validation_row(row) for row in alpha0_validation_rows]
    prediction_rows = [_completed_prediction_row(row) for row in alpha0_prediction_rows]
    seed_class_rows = [_completed_seed_class_row(row) for row in alpha0_seed_class_rows]
    condition_summary_rows = [
        _completed_condition_summary_row(
            alpha0_prediction_rows,
            condition_id="alpha0",
            class_count=config.class_count,
        )
    ]
    for condition_id in config.condition_ids[1:]:
        for model_seed in config.model_seeds:
            row_state = (
                "failed_model_lifecycle"
                if endpoint_state == "failed_model_lifecycle" and condition_id == failed_condition_id and model_seed == failed_model_seed
                else "not_tested_endpoint_closed"
            )
            model_rows.append(
                _null_model_cell(
                    condition_id=condition_id,
                    model_seed=model_seed,
                    state=row_state,
                )
            )
            for c in C_GRID:
                validation_rows.append(
                    _null_validation_row(
                        condition_id=condition_id,
                        model_seed=model_seed,
                        c=float(c),
                        state=(
                            "failed_model_lifecycle"
                            if endpoint_state == "failed_model_lifecycle" and condition_id == failed_condition_id and model_seed == failed_model_seed and float(c) == failed_c
                            else "not_tested_endpoint_closed"
                        ),
                    )
                )
            for class_label in range(config.class_count):
                seed_class_rows.append(
                    _null_seed_class_row(
                        condition_id=condition_id,
                        model_seed=model_seed,
                        class_label=class_label,
                        state="not_tested_endpoint_closed",
                    )
                )
            for base_row in base_test_rows:
                prediction_rows.append(
                    _null_prediction_row(
                        condition_id=condition_id,
                        model_seed=model_seed,
                        record_id=str(base_row["record_id"]),
                        record_order=int(base_row["source_row"]),
                        true_class=int(base_row["class_label"]),
                        code_identity=code_identity,
                        config_sha256=config.sha256,
                        state="not_tested_endpoint_closed",
                    )
                )
        condition_summary_rows.append(
            _null_condition_summary_row(condition_id, "not_tested_endpoint_closed")
        )

    class_observations = []
    for condition_id in config.condition_ids[1:]:
        perturbation_id, alpha = _decode_condition_id(condition_id)
        assert perturbation_id is not None and alpha is not None
        for metric_output_id in METRIC_OUTPUT_IDS:
            for class_label in range(config.class_count):
                class_observations.append(
                    {
                        "metric_output_id": metric_output_id,
                        "class_label": int(class_label),
                        "condition_id": condition_id,
                        "perturbation_id": perturbation_id,
                        "alpha": float(alpha),
                        "metric_harm": None,
                        "downstream_harm": None,
                        "state": "not_tested_endpoint_closed",
                    }
                )

    alignment_results = tuple(
        {
            "metric_output_id": metric_output_id,
            "state": "not_tested_endpoint_closed",
            "ag": None,
            "ag_raw": None,
            "ag_interval": None,
            "acc_cross": None,
            "acc_interval": None,
            "d_ag": None,
            "d_ag_interval": None,
            "d_acc": None,
            "d_acc_interval": None,
        }
        for metric_output_id in METRIC_OUTPUT_IDS
    )
    bootstrap_results = tuple(
        {
            "metric_output_id": metric_output_id,
            "state": "not_tested_endpoint_closed",
            "resamples": None,
            "candidate_ag_interval": None,
            "candidate_acc_interval": None,
            "d_ag_interval": None,
            "d_acc_interval": None,
        }
        for metric_output_id in METRIC_OUTPUT_IDS[1:]
    )
    sign_flip_results = tuple(
        {
            "metric_output_id": metric_output_id,
            "statistic": statistic,
            "state": "not_tested_endpoint_closed",
            "contrast": None,
            "p_value": None,
            "resamples": None,
        }
        for metric_output_id in METRIC_OUTPUT_IDS[1:]
        for statistic in ("d_ag", "d_acc")
    )
    holm_family = tuple(
        {
            "metric_output_id": metric_output_id,
            "statistic": statistic,
            "state": "not_tested_endpoint_closed",
            "raw_p_value": None,
            "adjusted_p_value": None,
            "rank": None,
            "family_size": 24,
            "family_state": "not_tested_endpoint_closed",
            "favorable": None,
            "rejected": None,
        }
        for metric_output_id in METRIC_OUTPUT_IDS[1:]
        for statistic in ("d_ag", "d_acc")
    )
    figure1_rows = _failed_figure1_rows(failure_receipt)
    figure2_rows = _failed_figure2_rows()
    figures = _render_failed_protocol_b_figures(figure1_rows, figure2_rows, failure_receipt)
    counts = _default_artifact_counts()
    failed_model_slots = 1 if endpoint_state == "failed_model_lifecycle" else 0
    failed_validation_slots = 1 if endpoint_state == "failed_model_lifecycle" else 0
    failed_figure1_rows = sum(1 for row in figure1_rows if row["state"] == "failed_model_lifecycle")
    counts["state_breakdown"] = {
        "model_cells": {"complete": 5, "not_tested_endpoint_closed": 200 - failed_model_slots},
        "validation_scores": {"complete": 20, "not_tested_endpoint_closed": 800 - failed_validation_slots},
        "predictions": {"complete": 15000, "not_tested_endpoint_closed": 600000},
        "seed_class_conditions": {"complete": 150, "not_tested_endpoint_closed": 6000},
        "condition_summary": {"complete": 1, "not_tested_endpoint_closed": 40},
        "class_observations": {"not_tested_endpoint_closed": 15600},
        "alignment_results": {"not_tested_endpoint_closed": 13},
        "bootstrap_results": {"not_tested_endpoint_closed": 12},
        "sign_flip_results": {"not_tested_endpoint_closed": 24},
        "holm_family": {"not_tested_endpoint_closed": 24},
        "figure1_data_rows": {"not_tested_endpoint_closed": 520 - failed_figure1_rows},
        "figure2_data_rows": {"not_tested_endpoint_closed": 13},
        "secondary_table_rows": {"not_tested_endpoint_closed": 13},
    }
    if failed_model_slots:
        counts["state_breakdown"]["model_cells"]["failed_model_lifecycle"] = failed_model_slots
        counts["state_breakdown"]["validation_scores"]["failed_model_lifecycle"] = failed_validation_slots
    if failed_figure1_rows:
        counts["state_breakdown"]["figure1_data_rows"]["failed_model_lifecycle"] = failed_figure1_rows
    preflight = {
        "status": "failed",
        "endpoint_state": endpoint_state,
        "failure": dict(failure_receipt),
        "authority_bridge_state": "complete",
        "matrix_input_state": "ready",
        "row_hash_state": "complete",
        "offset_state": "complete",
        "production_worker_count": 5,
        "verifier_worker_count": 4,
        "process_start_method": "spawn",
        "blas_thread_limit": 1,
        "admission_bytes_per_job": WORKER_ADMISSION_BYTES,
        "admission_budget_bytes": ADMISSION_BUDGET_BYTES,
        "condition_count": 41,
        "source_record_count": 66000,
        "role_occurrence_count": 330000,
        "condition_receipt_count": 2706000,
        "condition_matrix_shard_count": 66,
        "condition_matrix_bytes": 10791528000,
        "matrix_receipt_sha256": None,
        "configured_payload_count": 22,
    }
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "failed",
        "endpoint_state": endpoint_state,
        "failure": dict(failure_receipt),
        "experiment_id": EXPERIMENT_ID,
        "protocol": "B",
        "tier": "full_domain_core",
        "claim_boundary": CLAIM_BOUNDARY,
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "condition_count": 41,
        "model_seed_count": 5,
        "prediction_row_count": 615000,
        "class_observation_count": 15600,
        "counts": counts,
        "metric_states": {
            row["metric_output_id"]: row["state"]
            for row in alignment_results
        },
        "alpha0_equivalence": dict(alpha0_receipt),
        "authorities": config.document["authorities"],
        "parent_artifacts": config.document["parent_artifacts"],
        "code_authority": config.document["code_authority"],
        "environment_authority": config.document["environment_authority"],
        "inherited_rulings": config.document["inherited_rulings"],
    }
    failed_json = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "run": run_id,
        "status": "failed",
        "endpoint": endpoint_state,
        "failure": dict(failure_receipt),
    }
    values = {
        "config.json": config.raw_bytes,
        "authority_bridge.json": canonical_json_bytes(bridge_doc),
        "preflight.json": canonical_json_bytes(preflight),
        "alpha0_equivalence.json": canonical_json_bytes(dict(alpha0_receipt)),
        "model_cells.jsonl": jsonl_bytes(tuple(model_rows)),
        "validation_scores.jsonl": jsonl_bytes(tuple(validation_rows)),
        "predictions.jsonl": jsonl_bytes(tuple(prediction_rows)),
        "seed_class_conditions.jsonl": jsonl_bytes(tuple(seed_class_rows)),
        "condition_summary.csv": csv_bytes(tuple(condition_summary_rows)),
        "class_observations.jsonl": jsonl_bytes(tuple(class_observations)),
        "alignment_results.jsonl": jsonl_bytes(alignment_results),
        "bootstrap_results.jsonl": jsonl_bytes(bootstrap_results),
        "sign_flip_results.jsonl": jsonl_bytes(sign_flip_results),
        "holm_family.jsonl": jsonl_bytes(holm_family),
        "figure1_d1_protocol_b_full_domain.png": figures["figure1_d1_protocol_b_full_domain.png"],
        "figure1_d1_protocol_b_full_domain.svg": figures["figure1_d1_protocol_b_full_domain.svg"],
        "figure1_d1_protocol_b_full_domain_data.csv": csv_bytes(figure1_rows),
        "figure2_d1_protocol_b_full_domain.png": figures["figure2_d1_protocol_b_full_domain.png"],
        "figure2_d1_protocol_b_full_domain.svg": figures["figure2_d1_protocol_b_full_domain.svg"],
        "figure2_d1_protocol_b_full_domain_data.csv": csv_bytes(figure2_rows),
        "d1_protocol_b_full_domain_secondary_table.csv": csv_bytes(figure2_rows),
        "manifest.json": canonical_json_bytes(manifest),
    }
    payloads = {name: values[name] for name in ARTIFACT_PAYLOAD_FILES}
    payloads["failed.json"] = canonical_json_bytes(failed_json)
    return run_id, payloads


def _render_real_protocol_b_figures(projection: Mapping[str, Sequence[Mapping[str, object]]], config: Phase4D1ProtocolBConfig) -> Mapping[str, bytes]:
    colors=dict(zip(PERTURBATIONS,("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"),strict=True)); observations=projection["class_observations"]
    figure1=[]
    for metric in METRIC_OUTPUT_IDS:
        for perturbation in PERTURBATIONS:
            for alpha in ALPHAS[1:]:
                selected=[row for row in observations if row["metric_output_id"] == metric and row["perturbation_id"] == perturbation and row["alpha"] == alpha]
                figure1.append({"metric_output_id":metric,"perturbation_id":perturbation,"alpha":alpha,"mean_metric_harm":float(np.mean([row["metric_harm"] for row in selected])),"mean_downstream_harm":float(np.mean([row["downstream_harm"] for row in selected]))})
    family={(row["metric_output_id"],row["statistic"]):row for row in projection["holm_family"]}; figure2=[]
    for row in projection["alignment_results"]:
        output=dict(row)
        for statistic in ("d_ag","d_acc"):
            source=family.get((row["metric_output_id"],statistic),{}); output.update({f"{statistic}_raw_p":source.get("raw_p_value"),f"{statistic}_adjusted_p":source.get("adjusted_p_value"),f"{statistic}_rank":source.get("rank"),f"{statistic}_favorable":source.get("favorable"),f"{statistic}_rejected":source.get("rejected")})
        figure2.append(output)
    payloads={
        "figure1_d1_protocol_b_full_domain_data.csv":csv_bytes(figure1),
        "figure2_d1_protocol_b_full_domain_data.csv":csv_bytes(figure2),
        "d1_protocol_b_full_domain_secondary_table.csv":csv_bytes(figure2),
    }
    with matplotlib.rc_context({"font.family":"DejaVu Sans","lines.linewidth":1.5,"svg.hashsalt":"rpe-phase4-d1-protocol-b-v1"}):
        fig,axes=plt.subplots(4,4,figsize=(12,12)); flat=axes.ravel()
        for perturbation in PERTURBATIONS:
            selected=[row for row in figure1 if row["metric_output_id"] == "mse" and row["perturbation_id"] == perturbation]
            flat[0].plot([row["alpha"] for row in selected],[row["mean_downstream_harm"] for row in selected],marker="o",color=colors[perturbation])
        flat[0].set_title("downstream_harm"); flat[0].margins(x=0.05,y=0.05)
        for axis,metric in zip(flat[1:],METRIC_OUTPUT_IDS,strict=False):
            for perturbation in PERTURBATIONS:
                selected=[row for row in figure1 if row["metric_output_id"] == metric and row["perturbation_id"] == perturbation]
                axis.plot([row["mean_metric_harm"] for row in selected],[row["mean_downstream_harm"] for row in selected],marker="o",color=colors[perturbation])
            axis.set_title(metric); axis.margins(x=0.05,y=0.05)
        for axis in flat[14:]: axis.set_axis_off()
        fig.tight_layout(pad=1.05)
        for kind in ("png","svg"):
            stream=io.BytesIO(); fig.savefig(stream,format=kind,dpi=300,metadata={"Date":None,"Creator":"raman-preproc-eval"}); payloads[f"figure1_d1_protocol_b_full_domain.{kind}"]=stream.getvalue()
        plt.close(fig); fig,axes=plt.subplots(1,4,figsize=(14,8),sharey=True)
        for axis,field,color in zip(axes,("ag","acc_cross","d_ag","d_acc"),("#1f77b4","#ff7f0e","#2ca02c","#d62728"),strict=True):
            axis.barh(np.arange(13),[0.0 if row.get(field) is None else row[field] for row in figure2],color=color); axis.set_title(field); axis.margins(x=0.05,y=0.05)
        fig.tight_layout(pad=1.05)
        for kind in ("png","svg"):
            stream=io.BytesIO(); fig.savefig(stream,format=kind,dpi=300,metadata={"Date":None,"Creator":"raman-preproc-eval"}); payloads[f"figure2_d1_protocol_b_full_domain.{kind}"]=stream.getvalue()
        plt.close(fig)
    return MappingProxyType(payloads)


def build_phase4_d1_protocol_b_from_inputs(
    output_root: Path,
    *,
    inputs: SyntheticD1ProtocolBInputs,
    config: Phase4D1ProtocolBConfig,
    worker_count: int,
    bootstrap_resamples: int = 2000,
    sign_flip_resamples: int = 100000,
) -> Phase4D1ProtocolBSummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D1ProtocolBError("worker_count must be positive")
    payloads, prediction_row_count, class_observation_count = _build_synthetic_payloads(
        inputs,
        config,
        worker_count=worker_count,
        bootstrap_resamples=bootstrap_resamples,
        sign_flip_resamples=sign_flip_resamples,
    )
    run_id = RUN_PREFIX + sha256_hex(
        config.raw_bytes
        + payloads["model_cells.jsonl"]
        + payloads["validation_scores.jsonl"]
        + payloads["predictions.jsonl"]
    )
    terminal = canonical_json_bytes({"run_id": run_id, "status": "complete"})
    path = _write_protocol_b_artifact(
        Path(output_root),
        payloads,
        terminal_name="complete.json",
        terminal_bytes=terminal,
    )
    return Phase4D1ProtocolBSummary(
        path=path,
        run_id=run_id,
        status="complete",
        prediction_row_count=prediction_row_count,
        class_observation_count=class_observation_count,
    )


def build_phase4_d1_protocol_b(
    output_root: Path,
    *,
    worker_count: int = 5,
) -> Phase4D1ProtocolBSummary:
    config = load_phase4_d1_protocol_b_config()
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or not 1 <= worker_count <= 5:
        raise Phase4D1ProtocolBError("worker_count must be in 1..5")
    if worker_count * WORKER_ADMISSION_BYTES > ADMISSION_BUDGET_BYTES:
        raise Phase4D1ProtocolBError("pre-output worker memory admission failed")
    bridge = validate_d1_protocol_b_parent_authorities(config)
    parent21 = Path(bridge["step21"]["path"])
    parent_models = _read_jsonl(parent21 / "model_cells.jsonl")
    parent_scores = _read_jsonl(parent21 / "validation_scores.jsonl")
    parent_predictions = _read_jsonl(parent21 / "predictions.jsonl")
    step23_path = Path(bridge["step23"]["path"])
    role_index = _index_step23_role_ledger(step23_path)
    condition_index = _index_step23_condition_ledger(step23_path, config)
    code_identity = _sha_file(Path(__file__))
    # Condition-major consumption: parse ledgers once; load each condition only once.
    model_rows: list[Mapping[str, object]] = []; validation_rows: list[Mapping[str, object]] = []; prediction_rows: list[Mapping[str, object]] = []; seed_class: list[Mapping[str, object]] = []
    alpha0_receipt: Mapping[str, object] | None = None
    alpha0_model_rows: tuple[Mapping[str, object], ...] = ()
    alpha0_validation_rows: tuple[Mapping[str, object], ...] = ()
    alpha0_prediction_rows: tuple[Mapping[str, object], ...] = ()
    alpha0_seed_class_rows: tuple[Mapping[str, object], ...] = ()
    for condition_id in config.condition_ids:
        source_rows = _load_indexed_condition_block(step23_path, condition_index[condition_id])
        jobs = [(seed, condition_id, source_rows, role_index[seed], code_identity, config.sha256) for seed in config.model_seeds]
        try:
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=multiprocessing.get_context("spawn")) as executor:
                results = list(executor.map(_indexed_seed_job, jobs))
        except Phase4D1ProtocolBModelLifecycleWarning as error:
            if condition_id == "alpha0" or alpha0_receipt is None:
                raise
            run_id, failed_values = _build_real_failure_payloads(
                config,
                bridge,
                alpha0_receipt=alpha0_receipt,
                alpha0_model_rows=alpha0_model_rows,
                alpha0_validation_rows=alpha0_validation_rows,
                alpha0_prediction_rows=alpha0_prediction_rows,
                alpha0_seed_class_rows=alpha0_seed_class_rows,
                base_test_rows=role_index[MODEL_SEEDS[0]]["test"],
                code_identity=code_identity,
                failure_receipt=error.receipt,
            )
            failed_marker = failed_values.pop("failed.json")
            target = _write_protocol_b_artifact(
                Path(output_root),
                failed_values,
                terminal_name="failed.json",
                terminal_bytes=failed_marker,
            )
            return Phase4D1ProtocolBSummary(target, run_id, "failed", 615000, 15600)
        for seed, cell, scores, predictions in sorted(results, key=lambda item: item[0]):
            model_rows.append(cell); validation_rows.extend(scores); prediction_rows.extend(predictions)
            for label in range(config.class_count):
                correct = [row["correct"] for row in predictions if int(row["true_class"]) == label]
                if len(correct) != config.records_per_class: raise Phase4D1ProtocolBError("seed-class cardinality mismatch")
                seed_class.append({"model_seed": seed, "class_label": label, "condition_id": condition_id, "accuracy_seed_class": float(np.mean(correct))})
        if condition_id == "alpha0":
            def projection(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> str:
                return sha256_hex(jsonl_bytes(tuple({field: row[field] for field in fields} for row in rows)))
            mf=("model_seed","model_state_sha256","pca_train_feature_sha256","pca_validation_feature_sha256","selected_c","train_matrix_sha256","train_record_ids_sha256","validation_matrix_sha256","validation_record_ids_sha256","warning_state"); vf=("c","model_seed","validation_top1_accuracy"); pf=("condition_id","correct","model_seed","predicted_class","projected_row_sha256","record_id","record_order","true_class")
            alpha0_model_rows = tuple(row for row in model_rows if row["condition_id"] == "alpha0")
            alpha0_validation_rows = tuple(row for row in validation_rows if row["condition_id"] == "alpha0")
            alpha0_prediction_rows = tuple(row for row in prediction_rows if row["condition_id"] == "alpha0")
            alpha0_seed_class_rows = tuple(row for row in seed_class if row["condition_id"] == "alpha0")
            alpha_current={"model_digest":projection(alpha0_model_rows,mf),"validation_digest":projection(alpha0_validation_rows,vf),"prediction_digest":projection(alpha0_prediction_rows,pf)}
            alpha_parent={"model_digest":projection(parent_models,mf),"validation_digest":projection(parent_scores,vf),"prediction_digest":projection([row for row in parent_predictions if row.get("condition_id") == "alpha0"],pf)}
            alpha0_receipt = _real_alpha0_receipt(alpha_current, alpha_parent, config.document["alpha0_equivalence"])
            if str(alpha0_receipt["status"]) != "passed":
                failure_receipt = _alpha0_failure_receipt(alpha0_receipt)
                run_id, failed_values = _build_real_failure_payloads(
                    config,
                    bridge,
                    alpha0_receipt=alpha0_receipt,
                    alpha0_model_rows=alpha0_model_rows,
                    alpha0_validation_rows=alpha0_validation_rows,
                    alpha0_prediction_rows=alpha0_prediction_rows,
                    alpha0_seed_class_rows=alpha0_seed_class_rows,
                    base_test_rows=role_index[MODEL_SEEDS[0]]["test"],
                    code_identity=code_identity,
                    failure_receipt=failure_receipt,
                )
                failed_marker = failed_values.pop("failed.json")
                target = _write_protocol_b_artifact(
                    Path(output_root),
                    failed_values,
                    terminal_name="failed.json",
                    terminal_bytes=failed_marker,
                )
                return Phase4D1ProtocolBSummary(target, run_id, "failed", 615000, 15600)
    if (len(model_rows), len(validation_rows), len(prediction_rows), len(seed_class)) != (205, 820, 615000, 6150):
        raise Phase4D1ProtocolBError("real model lifecycle denominator mismatch")
    def projection(rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> str:
        return sha256_hex(jsonl_bytes(tuple({field: row[field] for field in fields} for row in rows)))
    mf=("model_seed","model_state_sha256","pca_train_feature_sha256","pca_validation_feature_sha256","selected_c","train_matrix_sha256","train_record_ids_sha256","validation_matrix_sha256","validation_record_ids_sha256","warning_state"); vf=("c","model_seed","validation_top1_accuracy"); pf=("condition_id","correct","model_seed","predicted_class","projected_row_sha256","record_id","record_order","true_class")
    alpha_models=[row for row in model_rows if row["condition_id"] == "alpha0"]; alpha_scores=[row for row in validation_rows if row["condition_id"] == "alpha0"]; alpha_predictions=[row for row in prediction_rows if row["condition_id"] == "alpha0"]
    alpha_current={"model_digest":projection(alpha_models,mf),"validation_digest":projection(alpha_scores,vf),"prediction_digest":projection(alpha_predictions,pf)}
    alpha_parent={"model_digest":projection(parent_models,mf),"validation_digest":projection(parent_scores,vf),"prediction_digest":projection([row for row in parent_predictions if row.get("condition_id") == "alpha0"],pf)}
    alpha_receipt=_real_alpha0_receipt(alpha_current,alpha_parent,config.document["alpha0_equivalence"])
    projection_rows = _aggregate_real_outcome(prediction_rows, _real_metric_rows(bridge), config, bootstrap_resamples=2000, sign_flip_resamples=100000)
    figures = _render_real_protocol_b_figures(projection_rows, config); summaries = _real_condition_summary(prediction_rows, config)
    matrix_receipt=hashlib.sha256()
    for row in model_rows:
        matrix_receipt.update(canonical_json_bytes({"condition_id":row["condition_id"],"model_seed":row["model_seed"],"train_matrix_sha256":row["train_matrix_sha256"]}))
    run_id,bridge_doc,preflight,manifest=_real_artifact_metadata(config.document,bridge,alpha_receipt,matrix_receipt.hexdigest(),projection_rows)
    values={"config.json":config.raw_bytes,"authority_bridge.json":canonical_json_bytes(bridge_doc),"preflight.json":canonical_json_bytes(preflight),"alpha0_equivalence.json":canonical_json_bytes(alpha_receipt),"model_cells.jsonl":jsonl_bytes(model_rows),"validation_scores.jsonl":jsonl_bytes(validation_rows),"predictions.jsonl":jsonl_bytes(prediction_rows),"seed_class_conditions.jsonl":jsonl_bytes(seed_class),"condition_summary.csv":csv_bytes(summaries),"class_observations.jsonl":jsonl_bytes(projection_rows["class_observations"]),"alignment_results.jsonl":jsonl_bytes(projection_rows["alignment_results"]),"bootstrap_results.jsonl":jsonl_bytes(projection_rows["bootstrap_results"]),"sign_flip_results.jsonl":jsonl_bytes(projection_rows["sign_flip_results"]),"holm_family.jsonl":jsonl_bytes(projection_rows["holm_family"]),"manifest.json":canonical_json_bytes(manifest),**figures}
    payloads={name:values[name] for name in ARTIFACT_PAYLOAD_FILES}
    terminal=canonical_json_bytes({"run_id":run_id,"status":"complete"})
    target = _write_protocol_b_artifact(
        Path(output_root),
        payloads,
        terminal_name="complete.json",
        terminal_bytes=terminal,
    )
    return Phase4D1ProtocolBSummary(target,run_id,"complete",615000,15600)
