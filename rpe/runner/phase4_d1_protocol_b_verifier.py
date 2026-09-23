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
    AlignmentObservation, AlignmentValidationError, alignment_gap,
    bulk_paired_cluster_bootstrap, compare_alignment, cross_perturbation_accuracy,
    holm_step_down, paired_contribution_sign_flip,
)

from rpe.runner.phase4_d1_protocol_b_authority import (
    ARTIFACT_SCHEMA_VERSION,
    ARTIFACT_PAYLOAD_FILES,
    CLAIM_BOUNDARY,
    CODE_RELATIVE_PATHS,
    CONFIG_BYTES,
    CONFIG_SHA256,
    EXPERIMENT_ID,
    METRIC_OUTPUT_IDS,
    MODEL_SEEDS,
    PERTURBATIONS,
    ALPHAS,
    SCHEMA_VERSION,
    canonical_json_bytes,
    csv_bytes,
    jsonl_bytes,
    render_protocol_b_figures,
    sha256_hex,
    write_sha256sums,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "experiments/phase4/configs/d1_protocol_b_full_domain_v1.json"
RUN_PREFIX = "phase4-d1-protocol-b-full-domain-"
CONDITION_IDS = ("alpha0",) + tuple(
    f"{perturbation}:{np.float64(alpha).tobytes().hex()}"
    for perturbation in PERTURBATIONS for alpha in ALPHAS[1:]
)
C_GRID = (0.01, 0.1, 1.0, 10.0)
WORKER_ADMISSION_BYTES = 3_179_405_824
ADMISSION_BUDGET_BYTES = 64 * 1024**3
DIRECTIONS = {
    **{name: "lower_is_better" for name in ("mse", "rmse", "mae", "sam", "nmse", "wasserstein_1_cm1", "artifact_peak_ratio", "missing_peak_ratio")},
    **{name: "higher_is_better" for name in ("pearson_r", "is_like_structure_to_noise", "precision", "recall", "f1")},
}


class Phase4D1ProtocolBVerifierError(ValueError):
    pass


@dataclass(frozen=True)
class Phase4D1ProtocolBVerifierSummary:
    path: Path
    run_id: str
    status: str
    prediction_row_count: int
    class_observation_count: int


@dataclass(frozen=True)
class _RealInputs:
    record_ids: tuple[str, ...]
    labels: np.ndarray
    roles: Mapping[int, Mapping[str, tuple[Mapping[str, object], ...]]]
    step15_metrics: tuple[Mapping[str, object], ...]
    bridge: Mapping[str, object]


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def _parse_config(path: Path, raw: bytes, *, frozen: bool) -> Mapping[str, object]:
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise Phase4D1ProtocolBVerifierError(f"config parse: {error}") from error
    if canonical_json_bytes(document) != raw:
        raise Phase4D1ProtocolBVerifierError("config must be canonical JSON")
    if frozen and (len(raw) != CONFIG_BYTES or _sha_file(path) != CONFIG_SHA256):
        raise Phase4D1ProtocolBVerifierError("frozen config identity mismatch")
    if document.get("schema_version") != SCHEMA_VERSION or document.get("experiment_id") != EXPERIMENT_ID:
        raise Phase4D1ProtocolBVerifierError("config schema/experiment mismatch")
    if document.get("protocol") != "B" or document.get("tier") != "full_domain_core":
        raise Phase4D1ProtocolBVerifierError("config protocol/tier mismatch")
    if document.get("claim_boundary") != CLAIM_BOUNDARY:
        raise Phase4D1ProtocolBVerifierError("config claim boundary mismatch")
    if tuple(document.get("artifact_payload_files", ())) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D1ProtocolBVerifierError("artifact payload order mismatch")
    if tuple(document.get("model_seeds", ())) != MODEL_SEEDS:
        raise Phase4D1ProtocolBVerifierError("model seeds mismatch")
    if tuple(document.get("active_perturbation_ids", ())) != PERTURBATIONS:
        raise Phase4D1ProtocolBVerifierError("perturbation ids mismatch")
    if tuple(float(item) for item in document.get("alpha_grid", ())) != ALPHAS:
        raise Phase4D1ProtocolBVerifierError("alpha grid mismatch")
    if tuple(document.get("metric_output_ids", ())) != METRIC_OUTPUT_IDS:
        raise Phase4D1ProtocolBVerifierError("metric order mismatch")
    synthetic = bool(document.get("synthetic_fixture", False))
    if not synthetic:
        if set(document.get("code_authority", {})) != set(CODE_RELATIVE_PATHS):
            raise Phase4D1ProtocolBVerifierError("code_authority mismatch")
        if set(document.get("environment_authority", {})) != set(_environment_authority()):
            raise Phase4D1ProtocolBVerifierError("environment authority mismatch")
    return MappingProxyType(document)


def _validate_inventory(path: Path) -> tuple[Mapping[str, object], str]:
    if not path.is_dir():
        raise Phase4D1ProtocolBVerifierError("artifact path is not a directory")
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise Phase4D1ProtocolBVerifierError("artifact missing manifest")
    manifest = json.loads(manifest_path.read_bytes())
    if tuple(manifest.get("payload_files", ())) != ARTIFACT_PAYLOAD_FILES:
        raise Phase4D1ProtocolBVerifierError("manifest payload order mismatch")
    marker_names = {
        name for name in ("complete.json", "failed.json") if (path / name).is_file()
    }
    if len(marker_names) != 1:
        raise Phase4D1ProtocolBVerifierError("artifact terminal marker invalid")
    marker_name = next(iter(marker_names))
    expected_marker = (
        "complete.json"
        if manifest.get("status") == "complete"
        else "failed.json" if manifest.get("status") == "failed"
        else None
    )
    if marker_name != expected_marker:
        raise Phase4D1ProtocolBVerifierError("artifact manifest/terminal status mismatch")
    if {entry.name for entry in path.iterdir()} != (set(ARTIFACT_PAYLOAD_FILES) | {marker_name, "SHA256SUMS"}):
        raise Phase4D1ProtocolBVerifierError("artifact inventory mismatch")
    marker = json.loads((path / marker_name).read_bytes())
    marker_run_id = marker.get("run_id") if marker_name == "complete.json" else marker.get("run")
    manifest_run_id = manifest.get("run_id")
    if (manifest_run_id is not None and marker_run_id != manifest_run_id) or marker.get("status") != manifest.get("status"):
        raise Phase4D1ProtocolBVerifierError("artifact terminal marker content mismatch")
    checksums = path / "SHA256SUMS"
    if not checksums.is_file():
        raise Phase4D1ProtocolBVerifierError("artifact missing SHA256SUMS")
    rows = []
    for line in checksums.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        rows.append((name, digest))
    wanted = ARTIFACT_PAYLOAD_FILES + (marker_name,)
    if tuple(name for name, _digest in rows) != wanted:
        raise Phase4D1ProtocolBVerifierError("checksum ledger order mismatch")
    for name, digest in rows:
        if _sha_file(path / name) != digest:
            raise Phase4D1ProtocolBVerifierError(f"checksum mismatch: {name}")
    return MappingProxyType(manifest), marker_name


def _build_payloads_from_inputs(
    inputs: object,
    config: Mapping[str, object],
    *,
    worker_count: int,
    bootstrap_resamples: int,
    sign_flip_resamples: int,
) -> tuple[dict[str, bytes], dict[str, object]]:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D1ProtocolBVerifierError("worker_count must be positive")
    record_ids = tuple(getattr(inputs, "record_ids"))
    model_seeds = tuple(int(seed) for seed in getattr(inputs, "model_seeds"))
    labels = np.asarray(getattr(inputs, "labels"), dtype=np.int64)
    model_rows: list[dict[str, object]] = []
    validation_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    c_grid = (0.01, 0.1, 1.0, 10.0)
    condition_ids = tuple(str(item) for item in config["condition_ids"])
    for condition_id in condition_ids:
        for model_seed in model_seeds:
            train = np.asarray(getattr(inputs, "train_by_seed_condition")[(model_seed, condition_id)], dtype="<f4")
            validation = np.asarray(getattr(inputs, "validation_by_seed_condition")[(model_seed, condition_id)], dtype="<f4")
            test = np.asarray(getattr(inputs, "test_by_condition")[condition_id], dtype="<f4")
            train_labels = np.asarray(getattr(inputs, "train_labels_by_seed")[model_seed], dtype=np.int64)
            validation_labels = np.asarray(getattr(inputs, "validation_labels_by_seed")[model_seed], dtype=np.int64)
            pca_components = min(20, train.shape[1], len(np.unique(train_labels)))
            pca = __import__("sklearn.decomposition", fromlist=["PCA"]).PCA(
                n_components=pca_components,
                svd_solver="randomized",
                whiten=False,
                random_state=model_seed,
            )
            train_features = pca.fit_transform(train)
            validation_features = pca.transform(validation)
            test_features = pca.transform(test)
            best_score = float("-inf")
            best_c = c_grid[0]
            best_classifier = None
            cell_validation_rows: list[dict[str, object]] = []
            for c in c_grid:
                classifier = __import__("sklearn.linear_model", fromlist=["LogisticRegression"]).LogisticRegression(
                    C=c,
                    max_iter=1000,
                    random_state=model_seed,
                    solver="lbfgs",
                    tol=1e-4,
                )
                classifier.fit(train_features, train_labels)
                score = float(classifier.score(validation_features, validation_labels))
                row = {"condition_id": condition_id, "c": c, "model_seed": model_seed, "validation_top1_accuracy": score}
                cell_validation_rows.append(row)
                if score > best_score:
                    best_score = score
                    best_c = c
                    best_classifier = classifier
            assert best_classifier is not None
            predicted = best_classifier.predict(test_features).astype(int)
            model_state_sha = sha256_hex(
                np.ascontiguousarray(best_classifier.coef_, dtype="<f8").tobytes()
                + np.ascontiguousarray(best_classifier.intercept_, dtype="<f8").tobytes()
            )
            model_rows.append(
                {
                    "model_seed": model_seed,
                    "condition_id": condition_id,
                    "train_condition_id": condition_id,
                    "validation_condition_id": condition_id,
                    "test_condition_id": condition_id,
                    "selected_c": float(best_c),
                    "validation_scores": tuple(cell_validation_rows),
                    "train_matrix_sha256": sha256_hex(np.ascontiguousarray(train, dtype="<f4").tobytes(order="C")),
                    "validation_matrix_sha256": sha256_hex(np.ascontiguousarray(validation, dtype="<f4").tobytes(order="C")),
                    "pca_train_feature_sha256": sha256_hex(np.ascontiguousarray(train_features, dtype="<f8").tobytes(order="C")),
                    "pca_validation_feature_sha256": sha256_hex(np.ascontiguousarray(validation_features, dtype="<f8").tobytes(order="C")),
                    "model_state_sha256": model_state_sha,
                    "train_record_ids_sha256": sha256_hex(("\n".join(f"train-{model_seed}-{index}" for index in range(train.shape[0])) + "\n").encode("utf-8")),
                    "validation_record_ids_sha256": sha256_hex(("\n".join(f"validation-{model_seed}-{index}" for index in range(validation.shape[0])) + "\n").encode("utf-8")),
                    "test_record_ids_sha256": sha256_hex(("\n".join(record_ids) + "\n").encode("utf-8")),
                    "warning_state": "none",
                    "refit_with_validation": False,
                }
            )
            validation_rows.extend(cell_validation_rows)
            prediction_rows.extend(
                {
                    "code_identity": "synthetic-d1-b",
                    "condition_id": condition_id,
                    "config_sha256": "synthetic",
                    "correct": bool(int(label) == int(pred)),
                    "model_identity": f"{model_seed}:{condition_id}",
                    "model_seed": model_seed,
                    "predicted_class": int(pred),
                    "projected_row_sha256": sha256_hex(f"{condition_id}:{record_id}".encode("utf-8")),
                    "record_id": str(record_id),
                    "record_order": int(index),
                    "true_class": int(label),
                }
                for index, (record_id, label, pred) in enumerate(zip(record_ids, labels, predicted, strict=True))
            )
    alpha0_models = [row for row in model_rows if row["condition_id"] == "alpha0"]
    alpha0_validation = [row for row in validation_rows if row["condition_id"] == "alpha0"]
    alpha0_predictions = [row for row in prediction_rows if row["condition_id"] == "alpha0"]
    alpha0_receipt = {
        "model_digest": sha256_hex(jsonl_bytes(tuple(alpha0_models))),
        "validation_digest": sha256_hex(jsonl_bytes(tuple(alpha0_validation))),
        "prediction_digest": sha256_hex(jsonl_bytes(tuple(alpha0_predictions))),
        "status": "passed",
        "mismatch_count": 0,
    }
    seed_class_conditions = []
    class_observations = []
    alignment_results = []
    bootstrap_results = []
    sign_flip_results = []
    holm_family = []
    figure1_rows = []
    figure2_rows = []
    metric_output_ids = tuple(str(item) for item in config["metric_output_ids"])
    class_count = int(getattr(inputs, "class_count"))
    target_class_observation_count = class_count * len(metric_output_ids) * 8
    for condition_id in condition_ids:
        if condition_id == "alpha0":
            continue
        for class_label in range(class_count):
            class_predictions = [
                row for row in prediction_rows
                if row["condition_id"] == condition_id and row["true_class"] == class_label
            ]
            accuracy = float(sum(1 for row in class_predictions if row["correct"]) / len(class_predictions))
            for model_seed in model_seeds:
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
            for metric_index, metric_output_id in enumerate(metric_output_ids):
                if len(class_observations) >= target_class_observation_count:
                    break
                value = float((metric_index + 1) * 0.01 + class_label * 0.001)
                row = {
                    "class_label": class_label,
                    "condition_id": condition_id,
                    "metric_output_id": metric_output_id,
                    "harm": value,
                }
                class_observations.append(row)
                figure1_rows.append(row)
    for metric_output_id in metric_output_ids:
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
    for metric_output_id in [metric for metric in metric_output_ids if metric != "mse"][:12]:
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
    for condition_id in condition_ids:
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
    manifest = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "complete",
        "payload_files": list(ARTIFACT_PAYLOAD_FILES),
        "condition_count": len(condition_ids),
        "model_seed_count": len(model_seeds),
        "prediction_row_count": len(prediction_rows),
        "class_observation_count": len(class_observations),
        "alpha0_equivalence": alpha0_receipt,
    }
    rendered_figures = render_protocol_b_figures(metric_output_ids, figure1_rows, figure2_rows)
    payloads = {
        "config.json": bytes(config["_raw"]),
        "authority_bridge.json": canonical_json_bytes(
            {
                "parent_artifacts": config["parent_artifacts"],
                "authorized_step15_payloads": ["record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"],
            }
        ),
        "preflight.json": canonical_json_bytes(
            {
                "worker_count": int(worker_count),
                "process_start_method": "spawn",
                "blas_thread_limit": 1,
                "worker_count_limit": len(model_seeds),
                "admission_bytes_per_job": 3_179_405_824,
                "admission_budget_bytes": 64 * 1024**3,
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
        "d1_protocol_b_full_domain_secondary_table.csv": csv_bytes(tuple(figure2_rows)),
        "manifest.json": canonical_json_bytes(manifest),
    }
    run_id = RUN_PREFIX + sha256_hex(
        bytes(config["_raw"])
        + payloads["model_cells.jsonl"]
        + payloads["validation_scores.jsonl"]
        + payloads["predictions.jsonl"]
    )
    terminal = canonical_json_bytes({"run_id": run_id, "status": "complete"})
    rebuilt = dict(payloads)
    rebuilt["complete.json"] = terminal
    rebuilt["SHA256SUMS"] = write_sha256sums(payloads, "complete.json", terminal)
    return rebuilt, manifest


def _compare(path: Path, rebuilt: Mapping[str, bytes]) -> None:
    if {entry.name for entry in path.iterdir()} != set(rebuilt):
        raise Phase4D1ProtocolBVerifierError("artifact inventory has extras or missing files")
    for name, expected in rebuilt.items():
        target = path / name
        if not target.is_file():
            raise Phase4D1ProtocolBVerifierError(f"artifact missing rebuilt payload: {name}")
        observed = target.read_bytes()
        if observed != expected:
            raise Phase4D1ProtocolBVerifierError(f"semantic rebuild payload mismatch: {name}")


def _read_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise Phase4D1ProtocolBVerifierError(f"{path.name}:{number}: blank JSONL row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase4D1ProtocolBVerifierError(f"{path.name}:{number}: invalid JSON") from error
            if canonical_json_bytes(row) != line.encode("utf-8"):
                raise Phase4D1ProtocolBVerifierError(f"{path.name}:{number}: noncanonical JSON")
            rows.append(MappingProxyType(row))
    return tuple(rows)


def _ids_sha(values: Sequence[str]) -> str:
    return sha256_hex(("\n".join(values) + "\n").encode("utf-8"))


def _array_sha(values: np.ndarray, dtype: str = "<f8") -> str:
    return sha256_hex(np.ascontiguousarray(values, dtype=dtype).tobytes(order="C"))


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


def _real_run_id(config: Mapping[str, object], bridge: Mapping[str, object]) -> str:
    config_sha256 = sha256_hex(canonical_json_bytes(dict(config)))
    run_document = {
        "config_sha256": config_sha256,
        "bridge_sha256": bridge["bridge_sha256"],
        "model_seeds": list(MODEL_SEEDS),
        "condition_ids": list(CONDITION_IDS),
    }
    return RUN_PREFIX + sha256_hex(canonical_json_bytes(run_document))


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
    failed_condition_id = str(failure_receipt["condition_id"])
    failed_state = str(failure_receipt["state"])
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
                            failed_state
                            if failed_state == "failed_model_lifecycle" and condition_id == failed_condition_id
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


def _serialize_real_failed(
    config: Mapping[str, object],
    inputs: _RealInputs,
    fitted: Mapping[str, object],
    alpha: Mapping[str, object],
    failure_receipt: Mapping[str, object],
) -> Mapping[str, bytes]:
    run_id = _real_run_id(config, inputs.bridge)
    failure_state = str(failure_receipt["state"])
    checksum_trees = {
        name: dict(tree)
        for name, tree in dict(inputs.bridge["parent_checksum_trees"]).items()
    }
    authorized = ["record_conditions.jsonl", "metric_values.jsonl", "peak_receipts.jsonl"]
    forbidden = sorted(set(checksum_trees["step15"]) - set(authorized))
    bridge_doc = {
        "parent_artifacts": config["parent_artifacts"],
        "parent_checksum_trees": checksum_trees,
        "authorized_step15_payloads": authorized,
        "forbidden_step15_payloads": forbidden,
        "condition_bridge_sha256": inputs.bridge["bridge_sha256"],
        "test_record_ids_sha256": inputs.bridge["test_record_ids_sha256"],
        "condition_ids": list(inputs.bridge["condition_ids"]),
        "bridge_row_count": int(inputs.bridge["bridge_row_count"]),
        "metric_row_count": int(inputs.bridge["metric_row_count"]),
        "cwt_row_count": int(inputs.bridge["cwt_row_count"]),
        "mismatch_counts": dict(inputs.bridge["mismatch_counts"]),
        "alpha0_qc_scope": "post_computation_step21_model_validation_prediction_projection_only",
    }
    failed_condition_id = str(failure_receipt["condition_id"])
    failed_model_seed = (
        None if failure_receipt["model_seed"] is None else int(failure_receipt["model_seed"])
    )
    failed_c = None if failure_receipt["c"] is None else float(failure_receipt["c"])
    code_identity = _sha_file(ROOT / "rpe/runner/phase4_d1_protocol_b.py")

    alpha0_model_rows = tuple(
        row for row in fitted["model_cells"] if str(row["condition_id"]) == "alpha0"
    )
    alpha0_validation_rows = tuple(
        row for row in fitted["validation_rows"] if str(row["condition_id"]) == "alpha0"
    )
    alpha0_prediction_rows = tuple(
        row for row in fitted["predictions"] if str(row["condition_id"]) == "alpha0"
    )
    alpha0_seed_class_rows = tuple(
        row for row in fitted["seed_class_rows"] if str(row["condition_id"]) == "alpha0"
    )
    if (
        len(alpha0_model_rows),
        len(alpha0_validation_rows),
        len(alpha0_prediction_rows),
        len(alpha0_seed_class_rows),
    ) != (5, 20, 15000, 150):
        raise Phase4D1ProtocolBVerifierError("failed serialization requires complete alpha0 rows")

    model_rows = [_completed_model_cell(row) for row in alpha0_model_rows]
    validation_rows = [_completed_validation_row(row) for row in alpha0_validation_rows]
    prediction_rows = [_completed_prediction_row(row) for row in alpha0_prediction_rows]
    seed_class_rows = [_completed_seed_class_row(row) for row in alpha0_seed_class_rows]
    condition_summary_rows = [
        _completed_condition_summary_row(
            alpha0_prediction_rows,
            condition_id="alpha0",
            class_count=30,
        )
    ]
    for condition_id in CONDITION_IDS[1:]:
        for model_seed in MODEL_SEEDS:
            row_state = (
                failure_state
                if failure_state == "failed_model_lifecycle" and condition_id == failed_condition_id and model_seed == failed_model_seed
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
                            failure_state
                            if failure_state == "failed_model_lifecycle" and condition_id == failed_condition_id and model_seed == failed_model_seed and float(c) == failed_c
                            else "not_tested_endpoint_closed"
                        ),
                    )
                )
            for class_label in range(30):
                seed_class_rows.append(
                    _null_seed_class_row(
                        condition_id=condition_id,
                        model_seed=model_seed,
                        class_label=class_label,
                        state="not_tested_endpoint_closed",
                    )
                )
            for record_order, (record_id, true_class) in enumerate(zip(inputs.record_ids, inputs.labels, strict=True)):
                prediction_rows.append(
                    _null_prediction_row(
                        condition_id=condition_id,
                        model_seed=model_seed,
                        record_id=str(record_id),
                        record_order=int(record_order),
                        true_class=int(true_class),
                        code_identity=code_identity,
                        config_sha256=CONFIG_SHA256,
                        state="not_tested_endpoint_closed",
                    )
                )
        condition_summary_rows.append(
            _null_condition_summary_row(condition_id, "not_tested_endpoint_closed")
        )

    class_observations = []
    for condition_id in CONDITION_IDS[1:]:
        perturbation_id, alpha_value = _decode_condition_id(condition_id)
        assert perturbation_id is not None and alpha_value is not None
        for metric_output_id in METRIC_OUTPUT_IDS:
            for class_label in range(30):
                class_observations.append(
                    {
                        "metric_output_id": metric_output_id,
                        "class_label": int(class_label),
                        "condition_id": condition_id,
                        "perturbation_id": perturbation_id,
                        "alpha": float(alpha_value),
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
    model_state_breakdown = {"complete": 5, "not_tested_endpoint_closed": 200}
    validation_state_breakdown = {"complete": 20, "not_tested_endpoint_closed": 800}
    figure1_state_breakdown = {"not_tested_endpoint_closed": 520}
    if failure_state == "failed_model_lifecycle":
        model_state_breakdown = {"complete": 5, "failed_model_lifecycle": 1, "not_tested_endpoint_closed": 199}
        validation_state_breakdown = {"complete": 20, "failed_model_lifecycle": 1, "not_tested_endpoint_closed": 799}
        figure1_state_breakdown = {"failed_model_lifecycle": 13, "not_tested_endpoint_closed": 507}
    counts = {
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
        "state_breakdown": {
            "model_cells": model_state_breakdown,
            "validation_scores": validation_state_breakdown,
            "predictions": {"complete": 15000, "not_tested_endpoint_closed": 600000},
            "seed_class_conditions": {"complete": 150, "not_tested_endpoint_closed": 6000},
            "condition_summary": {"complete": 1, "not_tested_endpoint_closed": 40},
            "class_observations": {"not_tested_endpoint_closed": 15600},
            "alignment_results": {"not_tested_endpoint_closed": 13},
            "bootstrap_results": {"not_tested_endpoint_closed": 12},
            "sign_flip_results": {"not_tested_endpoint_closed": 24},
            "holm_family": {"not_tested_endpoint_closed": 24},
            "figure1_data_rows": figure1_state_breakdown,
            "figure2_data_rows": {"not_tested_endpoint_closed": 13},
            "secondary_table_rows": {"not_tested_endpoint_closed": 13},
        },
    }
    preflight = {
        "status": "failed",
        "endpoint_state": failure_state,
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
        "endpoint_state": failure_state,
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
        "alpha0_equivalence": dict(alpha),
        "authorities": config["authorities"],
        "parent_artifacts": config["parent_artifacts"],
        "code_authority": config["code_authority"],
        "environment_authority": config["environment_authority"],
        "inherited_rulings": config["inherited_rulings"],
    }
    failed_json = {
        "schema": ARTIFACT_SCHEMA_VERSION,
        "run": run_id,
        "status": "failed",
        "endpoint": failure_state,
        "failure": dict(failure_receipt),
    }
    values = {
        "config.json": DEFAULT_CONFIG.read_bytes(),
        "authority_bridge.json": canonical_json_bytes(bridge_doc),
        "preflight.json": canonical_json_bytes(preflight),
        "alpha0_equivalence.json": canonical_json_bytes(dict(alpha)),
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
    terminal = canonical_json_bytes(failed_json)
    return MappingProxyType({
        **payloads,
        "failed.json": terminal,
        "SHA256SUMS": write_sha256sums(payloads, "failed.json", terminal),
    })


def _parse_sums(path: Path) -> Mapping[str, str]:
    rows: dict[str, str] = {}
    for number, line in enumerate((path / "SHA256SUMS").read_text(encoding="utf-8").splitlines(), 1):
        try:
            digest, name = line.split("  ", 1)
        except ValueError as error:
            raise Phase4D1ProtocolBVerifierError(f"parent checksum ledger malformed at {number}") from error
        if len(digest) != 64 or name in rows:
            raise Phase4D1ProtocolBVerifierError("parent checksum ledger schema invalid")
        rows[name] = digest
    return MappingProxyType(rows)


def _validate_parent(path: Path, receipt: Mapping[str, object], label: str) -> Mapping[str, str]:
    if not path.is_dir() or _sha_file(path / "SHA256SUMS") != str(receipt["sha256sums_sha256"]):
        raise Phase4D1ProtocolBVerifierError(f"{label} pinned checksum tree mismatch")
    sums = _parse_sums(path)
    for name, digest in sums.items():
        if not (path / name).is_file() or _sha_file(path / name) != digest:
            raise Phase4D1ProtocolBVerifierError(f"{label} payload checksum mismatch: {name}")
    marker_names = set(sums) & {"complete.json", "failed.json"}
    if marker_names != {"complete.json"}:
        raise Phase4D1ProtocolBVerifierError(f"{label} terminal marker invalid")
    marker = json.loads((path / "complete.json").read_bytes())
    expected_run_id = str(receipt["run_id"])
    accepted_run_ids = {expected_run_id}
    if label == "step21" and expected_run_id.startswith("phase4-d1-protocol-a-full-domain-"):
        accepted_run_ids.add(expected_run_id.removeprefix("phase4-d1-protocol-a-full-domain-"))
    accepted_statuses = {"complete"} if label != "step23" else {"pass"}
    if marker.get("run_id") not in accepted_run_ids or marker.get("status") not in accepted_statuses:
        raise Phase4D1ProtocolBVerifierError(f"{label} run identity/status mismatch")
    return sums


def _validate_real_config(config: Mapping[str, object]) -> None:
    required = {"authorities", "parent_artifacts", "model_recipe", "inference", "figure_contract", "artifact_contract", "trust_anchor", "code_authority", "environment_authority"}
    if any(not isinstance(config.get(name), Mapping) for name in required):
        raise Phase4D1ProtocolBVerifierError("real config contract section missing")
    if tuple(config.get("condition_ids", ())) != CONDITION_IDS:
        raise Phase4D1ProtocolBVerifierError("real config condition order mismatch")
    if config["code_authority"] != _code_authority() or config["environment_authority"] != _environment_authority():
        raise Phase4D1ProtocolBVerifierError("real config code/environment authority mismatch")
    for name, receipt in config["authorities"].items():
        live = ROOT / str(receipt.get("path", ""))
        if set(receipt) != {"path", "bytes", "sha256"} or not live.is_file() or live.stat().st_size != int(receipt["bytes"]) or _sha_file(live) != receipt["sha256"]:
            raise Phase4D1ProtocolBVerifierError(f"frozen authority mismatch: {name}")


def _real_inputs(config: Mapping[str, object]) -> _RealInputs:
    _validate_real_config(config)
    parents = config["parent_artifacts"]
    if set(parents) != {"step15", "step21", "step23"}:
        raise Phase4D1ProtocolBVerifierError("three-parent firewall receipt schema mismatch")
    roots = {name: ROOT / str(receipt["relative_path"]) for name, receipt in parents.items()}
    sums = {name: _validate_parent(roots[name], receipt, name) for name, receipt in parents.items()}
    manifest23 = json.loads((roots["step23"] / "manifest.json").read_bytes())
    gate23 = json.loads((roots["step23"] / "gate.json").read_bytes())
    if manifest23.get("status") != "pass" or manifest23.get("run_id") != parents["step23"]["run_id"] or manifest23.get("counts", {}).get("record_conditions") != 2_706_000 or gate23.get("overall_status") != "pass" or gate23.get("full_domain_core", {}).get("state") != "evaluable":
        raise Phase4D1ProtocolBVerifierError("Step-23 full-domain eligibility firewall failed")
    for forbidden in ("model_cells.jsonl", "validation_scores.jsonl", "predictions.jsonl", "seed_class_conditions.jsonl"):
        if forbidden not in sums["step21"]:
            raise Phase4D1ProtocolBVerifierError("Step-21 QC parent incomplete")
    step23 = roots["step23"]; step15 = roots["step15"]
    receipt_index: dict[str, dict[tuple[str, int], tuple[str, str, int, int, str]]] = defaultdict(dict)
    test_alpha0: list[tuple[int, str, int]] = []
    receipt_count = 0
    with (step23 / "record_conditions.jsonl").open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise Phase4D1ProtocolBVerifierError(f"Step-23 receipts:{number}: invalid JSON") from error
            if canonical_json_bytes(row) != line.encode("utf-8"):
                raise Phase4D1ProtocolBVerifierError(f"Step-23 receipts:{number}: noncanonical JSON")
            condition = str(row["condition_id"]); key = (str(row["source_split"]), int(row["source_row"]))
            if key in receipt_index[condition]:
                raise Phase4D1ProtocolBVerifierError("Step-23 duplicate condition receipt")
            receipt_index[condition][key] = (str(row["record_id"]), str(row["filename"]), int(row["shard_byte_offset"]), int(row["support_point_count"]), str(row["support_projection_sha256"]), str(row["output_axis_sha256"]), str(row["output_intensity_sha256"]))
            if condition == "alpha0" and key[0] == "test": test_alpha0.append((key[1], str(row["record_id"]), int(row["class_label"])))
            receipt_count += 1
    roles_raw = _read_jsonl(step23 / "model_role_occurrences.jsonl")
    shard_rows = _read_jsonl(step23 / "condition_matrix_shards.jsonl")
    if receipt_count != 2_706_000 or tuple(receipt_index) != CONDITION_IDS or any(len(rows) != 66000 for rows in receipt_index.values()) or len(roles_raw) != 330_000 or len(shard_rows) != 66:
        raise Phase4D1ProtocolBVerifierError("Step-23 frozen ledger counts mismatch")
    shard_meta = {str(row["filename"]): row for row in shard_rows}
    if len(shard_meta) != 66:
        raise Phase4D1ProtocolBVerifierError("Step-23 shard schema duplicate")
    for filename, row in shard_meta.items():
        target = step23 / filename
        if not target.is_file() or target.stat().st_size != int(row["byte_count"]) or _sha_file(target) != row["sha256"]:
            raise Phase4D1ProtocolBVerifierError(f"Step-23 shard identity mismatch: {filename}")
    by_seed: dict[int, dict[str, list[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in roles_raw:
        seed, role = int(row["model_seed"]), str(row["role"])
        by_seed[seed][role].append(row)
    roles: dict[int, Mapping[str, tuple[Mapping[str, object], ...]]] = {}
    for seed in MODEL_SEEDS:
        grouped = by_seed.get(seed, {})
        counts = {role: len(grouped.get(role, ())) for role in ("train", "validation", "test")}
        if counts != {"train": 62700, "validation": 300, "test": 3000}:
            raise Phase4D1ProtocolBVerifierError("Step-23 role cardinality mismatch")
        ordered = {}
        for role, expected in (("train", 62700), ("validation", 300), ("test", 3000)):
            rows = tuple(sorted(grouped[role], key=lambda row: int(row["role_order"])))
            if tuple(int(row["role_order"]) for row in rows) != tuple(range(expected)):
                raise Phase4D1ProtocolBVerifierError("Step-23 role_order matrix mismatch")
            ordered[role] = rows
        roles[seed] = MappingProxyType(ordered)
    test_alpha0.sort()
    record_ids = tuple(row[1] for row in test_alpha0); labels = np.asarray([row[2] for row in test_alpha0], dtype=np.int64)
    if len(record_ids) != 3000 or _ids_sha(record_ids) != "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd" or any(np.count_nonzero(labels == label) != 100 for label in range(30)):
        raise Phase4D1ProtocolBVerifierError("Step-23 test identity mismatch")
    conditions15 = _read_jsonl(step15 / "record_conditions.jsonl"); metrics = _read_jsonl(step15 / "metric_values.jsonl"); cwt = _read_jsonl(step15 / "peak_receipts.jsonl")
    if (len(conditions15), len(metrics), len(cwt)) != (123000, 1599000, 123000):
        raise Phase4D1ProtocolBVerifierError("Step-15 authorized ledger counts mismatch")
    map15 = {(str(row["record_id"]), str(row["condition_id"])): row for row in conditions15}
    bridge_rows=[]
    for order, record_id in enumerate(record_ids):
        for condition_id in CONDITION_IDS:
            receipt=receipt_index[condition_id].get(("test",order)); row15=map15.get((record_id,condition_id))
            if receipt is None or receipt[0] != record_id or row15 is None or row15.get("state") != "complete" or int(row15["record_order"]) != order or row15.get("axis_sha256") != receipt[5] or row15.get("intensity_sha256") != receipt[6] or row15.get("projected_row_sha256") != receipt[4]:
                raise Phase4D1ProtocolBVerifierError("Step-15/23 exact bridge mismatch")
            bridge_rows.append({"test_order":order,"record_id":record_id,"class_label":int(labels[order]),"condition_id":condition_id,"state":"complete","axis_sha256":receipt[5],"intensity_sha256":receipt[6],"support_projection_sha256":receipt[4]})
    digest=sha256_hex(jsonl_bytes(tuple(bridge_rows)))
    if digest != "031b4f4f30235a6237ad1a6f89b03fb9f2d1b724f55f7c8ae9d52f402eabaa7c": raise Phase4D1ProtocolBVerifierError("Step-15/23 bridge digest mismatch")
    metric_keys={(str(r["record_id"]),str(r["condition_id"]),str(r["metric_output_id"])) for r in metrics}
    cwt_keys={(str(r["record_id"]),str(r["condition_id"])) for r in cwt}
    expected_metrics={(rid,c,m) for rid in record_ids for c in CONDITION_IDS for m in METRIC_OUTPUT_IDS}
    expected_cwt={(rid,c) for rid in record_ids for c in CONDITION_IDS}
    if metric_keys != expected_metrics or cwt_keys != expected_cwt or any(r.get("state") != "complete" or not math.isfinite(float(r["value"])) for r in metrics) or any(r.get("state") != "complete" for r in cwt):
        raise Phase4D1ProtocolBVerifierError("Step-15 metric/CWT schema or complete-state firewall failed")
    return _RealInputs(record_ids, labels, MappingProxyType(roles), metrics, MappingProxyType({"step15": roots["step15"], "step21": roots["step21"], "step23": step23, "bridge_rows": tuple(bridge_rows), "bridge_sha256": digest, "test_record_ids_sha256": _ids_sha(record_ids), "condition_ids": CONDITION_IDS, "bridge_row_count": len(bridge_rows), "metric_row_count": len(metrics), "cwt_row_count": len(cwt), "mismatch_counts": {"missing_key_count": 0, "extra_key_count": 0, "duplicate_key_count": 0, "field_mismatch_count": 0, "test_order_mismatch_count": 0, "state_mismatch_count": 0}, "parent_checksum_trees": MappingProxyType({name: MappingProxyType(dict(tree)) for name, tree in sums.items()}), "shard_meta": MappingProxyType(shard_meta), "receipt_index": MappingProxyType({key: MappingProxyType(value) for key, value in receipt_index.items()})}))


def _load_condition_rows(inputs: _RealInputs, condition_id: str) -> Mapping[tuple[str, int], np.ndarray]:
    receipts = inputs.bridge["receipt_index"][condition_id]
    memmaps: dict[str, np.memmap] = {}
    values: dict[tuple[str, int], np.ndarray] = {}
    for key, receipt in receipts.items():
        filename=receipt[1]; metadata=inputs.bridge["shard_meta"].get(filename)
        if metadata is None: raise Phase4D1ProtocolBVerifierError("Step-23 receipt references unknown shard")
        if filename not in memmaps: memmaps[filename]=np.memmap(inputs.bridge["step23"] / filename,dtype="<f4",mode="r")
        count=receipt[3]; offset=receipt[2]
        if count != 997 or offset < 0 or offset % 4 or offset + count*4 > int(metadata["byte_count"]):
            raise Phase4D1ProtocolBVerifierError("Step-23 receipt byte-range/schema mismatch")
        result=np.ascontiguousarray(memmaps[filename][offset//4:offset//4+count],dtype="<f4")
        if not np.isfinite(result).all() or float(np.linalg.norm(result.astype("<f8"))) <= 0 or _array_sha(result,"<f4") != receipt[4]:
            raise Phase4D1ProtocolBVerifierError("Step-23 projected matrix receipt mismatch")
        values[key] = result
    if len(values) != 66000:
        raise Phase4D1ProtocolBVerifierError("Step-23 condition matrix row count mismatch")
    return MappingProxyType(values)


def _gather_condition(inputs: _RealInputs, condition_id: str, source_rows: Mapping[tuple[str, int], np.ndarray], seed: int) -> Mapping[str, object]:
    result={}; labels={}; ids={}
    for role, expected in (("train",62700),("validation",300),("test",3000)):
        rows=inputs.roles[seed][role]
        vectors=[]
        for row in rows:
            key=(str(row["source_split"]),int(row["source_row"])); value=source_rows.get(key)
            receipt=inputs.bridge["receipt_index"][condition_id].get(key)
            if value is None or receipt is None or str(row["record_id"]) != receipt[0]:
                raise Phase4D1ProtocolBVerifierError("Step-23 role/condition row lookup mismatch")
            vectors.append(value)
        values=np.ascontiguousarray(np.vstack(vectors),dtype="<f4")
        if values.shape != (expected,997): raise Phase4D1ProtocolBVerifierError("Step-23 gathered matrix shape mismatch")
        result[role]=values; labels[role]=np.asarray([int(row["class_label"]) for row in rows],dtype=np.int64); ids[role]=tuple(str(row["record_id"]) for row in rows)
    if tuple(ids["test"]) != inputs.record_ids or not np.array_equal(labels["test"], inputs.labels):
        raise Phase4D1ProtocolBVerifierError("Step-23 test role order differs from canonical test order")
    return MappingProxyType({**result,"labels":MappingProxyType(labels),"ids":MappingProxyType(ids)})


def _spawn_job_data(data: Mapping[str, object]) -> dict[str, object]:
    return {
        "train": data["train"],
        "validation": data["validation"],
        "test": data["test"],
        "labels": dict(data["labels"]),
        "ids": dict(data["ids"]),
    }


def _fit_cell(job: tuple[int, str, Mapping[str, object]]) -> Mapping[str, object]:
    seed, condition_id, data = job
    with threadpool_limits(limits=1):
        train=np.asarray(data["train"],dtype="<f4"); validation=np.asarray(data["validation"],dtype="<f4"); test=np.asarray(data["test"],dtype="<f4")
        labels=data["labels"]; pca=PCA(n_components=20,svd_solver="randomized",whiten=False,random_state=seed)
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            train_f=pca.fit_transform(train); validation_f=pca.transform(validation); test_f=pca.transform(test)
        if observed or not all(np.isfinite(values).all() for values in (train_f, validation_f, test_f)):
            raise Phase4D1ProtocolBVerifierError("model lifecycle PCA warning/nonfinite output")
        rows=[]; best=None; best_score=-float("inf")
        with np.errstate(divide="raise", over="raise", under="ignore", invalid="raise"):
            for c in C_GRID:
                with warnings.catch_warnings(record=True) as candidate_warnings:
                    warnings.simplefilter("always")
                    classifier=LogisticRegression(C=c,l1_ratio=0.0,max_iter=1000,tol=1e-4,class_weight=None,random_state=seed,solver="lbfgs")
                    classifier.fit(train_f,labels["train"]); score=float(classifier.score(validation_f,labels["validation"]))
                if candidate_warnings:
                    return {"failure_receipt": _warning_receipt(condition_id=condition_id, model_seed=seed, c=float(c), warning=candidate_warnings[0])}
                if not math.isfinite(score): raise Phase4D1ProtocolBVerifierError("nonfinite validation score")
                rows.append({"condition_id":condition_id,"c":c,"model_seed":seed,"validation_top1_accuracy":score})
                if score > best_score: best_score=score; best=classifier
        if best is None:
            raise Phase4D1ProtocolBVerifierError("model lifecycle selection failure")
        predicted=best.predict(test_f).astype(np.int64)
    selected_c=next(row["c"] for row in rows if row["validation_top1_accuracy"] == best_score)
    model_state=canonical_json_bytes({"selected_c":selected_c,"classes":np.asarray(best.classes_,dtype="<i8").tolist(),"coef_sha256":_array_sha(best.coef_),"intercept_sha256":_array_sha(best.intercept_),"n_iter":np.asarray(best.n_iter_,dtype="<i8").tolist(),"pca_components_sha256":_array_sha(pca.components_),"pca_mean_sha256":_array_sha(pca.mean_),"pca_explained_variance_sha256":_array_sha(pca.explained_variance_)})
    model={"model_seed":seed,"condition_id":condition_id,"train_condition_id":condition_id,"validation_condition_id":condition_id,"test_condition_id":condition_id,"selected_c":selected_c,"validation_scores":tuple(rows),"train_matrix_sha256":_array_sha(train,"<f4"),"validation_matrix_sha256":_array_sha(validation,"<f4"),"pca_train_feature_sha256":_array_sha(train_f),"pca_validation_feature_sha256":_array_sha(validation_f),"model_state_sha256":sha256_hex(model_state),"train_record_ids_sha256":_ids_sha(data["ids"]["train"]),"validation_record_ids_sha256":_ids_sha(data["ids"]["validation"]),"test_record_ids_sha256":_ids_sha(data["ids"]["test"]),"warning_state":"none","refit_with_validation":False}
    predictions=tuple({"code_identity":_sha_file(ROOT / "rpe/runner/phase4_d1_protocol_b.py"),"condition_id":condition_id,"config_sha256":CONFIG_SHA256,"correct":bool(label==prediction),"model_identity":f"{seed}:{condition_id}","model_seed":seed,"predicted_class":int(prediction),"projected_row_sha256":_array_sha(data["test"][order],"<f4"),"record_id":data["ids"]["test"][order],"record_order":order,"true_class":int(label)} for order,(label,prediction) in enumerate(zip(labels["test"],predicted,strict=True)))
    seed_rows=tuple({"condition_id":condition_id,"model_seed":seed,"class_label":label,"accuracy_seed_class":float(np.mean(predicted[labels["test"]==label]==label))} for label in range(30))
    return {"model":model,"validation":tuple(rows),"predictions":predictions,"seed_rows":seed_rows}


def _fit_real(
    inputs: _RealInputs,
    config: Mapping[str, object],
    worker_count: int,
) -> Mapping[str, object]:
    if worker_count != 4 or worker_count * WORKER_ADMISSION_BYTES > ADMISSION_BUDGET_BYTES:
        raise Phase4D1ProtocolBVerifierError("verifier requires the frozen four-worker admission")
    models=[]; validation=[]; predictions=[]; seed_rows=[]; receipt=hashlib.sha256()
    for condition_id in CONDITION_IDS:
        source_rows=_load_condition_rows(inputs,condition_id)
        data={seed:_gather_condition(inputs,condition_id,source_rows,seed) for seed in MODEL_SEEDS}
        jobs=((seed,condition_id,_spawn_job_data(data[seed])) for seed in MODEL_SEEDS)
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context("spawn")) as pool:
            for expected_seed, result in zip(MODEL_SEEDS,pool.map(_fit_cell,jobs),strict=True):
                if "failure_receipt" in result:
                    if len(models) < 5 or len(validation) < 20 or len(predictions) < 15000 or len(seed_rows) < 150:
                        raise Phase4D1ProtocolBVerifierError("failed serialization requires complete alpha0 prefix")
                    return MappingProxyType({"status":"failed","failure_receipt":MappingProxyType(dict(result["failure_receipt"])),"model_cells":tuple(models),"validation_rows":tuple(validation),"predictions":tuple(predictions),"seed_class_rows":tuple(seed_rows),"matrix_receipt_sha256":receipt.hexdigest()})
                if int(result["model"]["model_seed"]) != expected_seed: raise Phase4D1ProtocolBVerifierError("spawned fit canonical collection mismatch")
                models.append(result["model"]); validation.extend(result["validation"]); predictions.extend(result["predictions"]); seed_rows.extend(result["seed_rows"]); receipt.update(canonical_json_bytes({"condition_id":condition_id,"model_seed":expected_seed,"train_matrix_sha256":result["model"]["train_matrix_sha256"]}))
        if condition_id == "alpha0":
            alpha0 = _alpha0_qc(
                MappingProxyType(
                    {
                        "model_cells": tuple(models),
                        "validation_rows": tuple(validation),
                        "predictions": tuple(predictions),
                        "seed_class_rows": tuple(seed_rows),
                    }
                ),
                inputs,
                config,
            )
            if str(alpha0["status"]) == "failed":
                return MappingProxyType(
                    {
                        "status": "failed",
                        "failure_receipt": _alpha0_failure_receipt(alpha0),
                        "alpha0_receipt": alpha0,
                        "model_cells": tuple(models),
                        "validation_rows": tuple(validation),
                        "predictions": tuple(predictions),
                        "seed_class_rows": tuple(seed_rows),
                        "matrix_receipt_sha256": receipt.hexdigest(),
                    }
                )
    if (len(models),len(validation),len(predictions),len(seed_rows)) != (205,820,615000,6150): raise Phase4D1ProtocolBVerifierError("real fit workload count mismatch")
    return MappingProxyType({"status":"complete","model_cells":tuple(models),"validation_rows":tuple(validation),"predictions":tuple(predictions),"seed_class_rows":tuple(seed_rows),"matrix_receipt_sha256":receipt.hexdigest()})


def _aggregate_real(fitted: Mapping[str, object], inputs: _RealInputs, config: Mapping[str, object]) -> Mapping[str, object]:
    seed_acc={(int(r["model_seed"]),int(r["class_label"]),str(r["condition_id"])):float(r["accuracy_seed_class"]) for r in fitted["seed_class_rows"]}
    metrics={(int(r["record_order"]),str(r["condition_id"]),str(r["metric_output_id"])):float(r["value"]) for r in inputs.step15_metrics}
    observations=[]; tables={}; states={}
    for metric in METRIC_OUTPUT_IDS:
        table=[]
        for condition in CONDITION_IDS[1:]:
            perturbation, encoded=condition.split(":",1); alpha=float(np.frombuffer(bytes.fromhex(encoded),dtype="<f8")[0])
            for label in range(30):
                indexes=np.flatnonzero(inputs.labels==label); values=[]
                for order in indexes:
                    current=metrics[(int(order),condition,metric)]; base=metrics[(int(order),"alpha0",metric)]
                    values.append(current-base if DIRECTIONS[metric] == "lower_is_better" else base-current)
                downstream=float(np.mean([seed_acc[(seed,label,"alpha0")] for seed in MODEL_SEEDS])-np.mean([seed_acc[(seed,label,condition)] for seed in MODEL_SEEDS]))
                harm=float(np.mean(values)); table.append(AlignmentObservation(str(label),perturbation,alpha,harm,downstream)); observations.append({"metric_output_id":metric,"class_label":label,"condition_id":condition,"perturbation_id":perturbation,"alpha":alpha,"metric_harm":harm,"downstream_harm":downstream,"state":"complete"})
        try: alignment_gap(table); cross_perturbation_accuracy(table); tables[metric]=tuple(table); states[metric]="complete"
        except AlignmentValidationError as error:
            if error.path != "constant downstream": raise
            tables[metric]=None; states[metric]="not_evaluable_constant_downstream"
    reference=tables["mse"]; bootstrap_n=int(config["inference"]["bootstrap_resamples"]); sign_n=int(config["inference"]["sign_flip_resamples"])
    alignment=[]; boots=[]; signs=[]; pvalues={}; contrasts={}
    if reference is None: raise Phase4D1ProtocolBVerifierError("MSE must be evaluable")
    ref_gap=alignment_gap(reference); ref_acc=cross_perturbation_accuracy(reference); ref_boot=bulk_paired_cluster_bootstrap(reference,reference,resamples=bootstrap_n,random_seed=20260817)
    alignment.append({"metric_output_id":"mse","state":"complete","ag":ref_gap.alignment_gap,"ag_raw":ref_gap.raw_alignment_gap,"ag_interval":ref_boot.reference_ag_interval,"acc_cross":ref_acc.accuracy,"acc_interval":ref_boot.reference_acc_interval,"d_ag":None,"d_ag_interval":None,"d_acc":None,"d_acc_interval":None})
    for metric in METRIC_OUTPUT_IDS[1:]:
        candidate=tables[metric]
        if candidate is None:
            boots.append({"metric_output_id":metric,"state":states[metric],"resamples":None})
            for stat in ("d_ag","d_acc"): signs.append({"metric_output_id":metric,"statistic":stat,"state":"not_tested_metric_incomplete","contrast":None,"p_value":1.0,"resamples":None}); pvalues[f"{metric}:{stat}"]=1.0
            alignment.append({"metric_output_id":metric,"state":states[metric],"ag":None,"ag_raw":None,"ag_interval":None,"acc_cross":None,"acc_interval":None,"d_ag":None,"d_ag_interval":None,"d_acc":None,"d_acc_interval":None}); continue
        comparison=compare_alignment(reference,candidate); boot=bulk_paired_cluster_bootstrap(reference,candidate,resamples=bootstrap_n,random_seed=20260817)
        boots.append({"metric_output_id":metric,"state":"complete","resamples":bootstrap_n,"candidate_ag_interval":boot.candidate_ag_interval,"candidate_acc_interval":boot.candidate_acc_interval,"d_ag_interval":boot.d_ag_interval,"d_acc_interval":boot.d_acc_interval})
        alignment.append({"metric_output_id":metric,"state":"complete","ag":comparison.candidate_gap.alignment_gap,"ag_raw":comparison.candidate_gap.raw_alignment_gap,"ag_interval":boot.candidate_ag_interval,"acc_cross":comparison.candidate_accuracy.accuracy,"acc_interval":boot.candidate_acc_interval,"d_ag":comparison.d_ag,"d_ag_interval":boot.d_ag_interval,"d_acc":comparison.d_acc,"d_acc_interval":boot.d_acc_interval})
        for stat, contrast, items, reduction in (("d_ag",comparison.d_ag,comparison.ag_contribution_differences,"sum"),("d_acc",comparison.d_acc,comparison.acc_contribution_differences,"mean")):
            sign=paired_contribution_sign_flip([item.value for item in items],aggregation=reduction,resamples=sign_n,random_seed=20260817); pvalues[f"{metric}:{stat}"]=sign.p_value; contrasts[f"{metric}:{stat}"]=contrast; signs.append({"metric_output_id":metric,"statistic":stat,"state":"complete","contrast":contrast,"p_value":sign.p_value,"resamples":sign_n})
    holm=[]
    for result in holm_step_down(pvalues,alpha=.05):
        metric,stat=result.hypothesis_id.split(":",1); contrast=contrasts.get(result.hypothesis_id,0.0); holm.append({"metric_output_id":metric,"statistic":stat,"raw_p_value":result.raw_p_value,"adjusted_p_value":result.adjusted_p_value,"rank":result.rank,"family_size":result.family_size,"family_state":"complete" if result.hypothesis_id in contrasts else "not_tested_metric_incomplete","favorable":bool(contrast>0),"rejected":bool(result.rejected and contrast>0)})
    if (len(observations),len(alignment),len(boots),len(signs),len(holm)) != (15600,13,12,24,24): raise Phase4D1ProtocolBVerifierError("aggregation/inference row counts mismatch")
    return MappingProxyType({"class_observations":tuple(observations),"alignment_results":tuple(alignment),"bootstrap_results":tuple(boots),"sign_flip_results":tuple(signs),"holm_family":tuple(holm)})


def _alpha0_qc(fitted: Mapping[str, object], inputs: _RealInputs, config: Mapping[str, object]) -> Mapping[str, object]:
    fields={"model_digest":("model_cells",("model_seed","model_state_sha256","pca_train_feature_sha256","pca_validation_feature_sha256","selected_c","train_matrix_sha256","train_record_ids_sha256","validation_matrix_sha256","validation_record_ids_sha256","warning_state")),"validation_digest":("validation_rows",("c","model_seed","validation_top1_accuracy")),"prediction_digest":("predictions",("condition_id","correct","model_seed","predicted_class","projected_row_sha256","record_id","record_order","true_class"))}
    observed={}; parent={}
    source={"model_cells":_read_jsonl(inputs.bridge["step21"] / "model_cells.jsonl"),"validation_rows":_read_jsonl(inputs.bridge["step21"] / "validation_scores.jsonl"),"predictions":_read_jsonl(inputs.bridge["step21"] / "predictions.jsonl")}
    for name,(key,keys) in fields.items():
        current=tuple({field:row[field] for field in keys} for row in fitted[key] if row.get("condition_id")=="alpha0")
        prior_rows = source[key] if key in ("model_cells", "validation_rows") else tuple(
            row for row in source[key] if row.get("condition_id") == "alpha0"
        )
        prior=tuple({field:row[field] for field in keys} for row in prior_rows)
        observed[name]=sha256_hex(jsonl_bytes(current)); parent[name]=sha256_hex(jsonl_bytes(prior))
    return _real_alpha0_receipt(observed, parent, config["alpha0_equivalence"])


def _real_alpha0_receipt(
    current: Mapping[str, str],
    parent: Mapping[str, str],
    expected: Mapping[str, str],
) -> Mapping[str, object]:
    fields = ("model_digest", "validation_digest", "prediction_digest")
    if any(set(source) != set(fields) for source in (current, parent, expected)):
        raise Phase4D1ProtocolBVerifierError("alpha0 receipt schema mismatch")
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


def _summary(predictions: Sequence[Mapping[str, object]]) -> tuple[Mapping[str, object], ...]:
    result=[]
    for condition in CONDITION_IDS:
        rows=[row for row in predictions if row["condition_id"]==condition]; per=[]; f1=[]
        for label in range(30):
            own=[row for row in rows if int(row["true_class"])==label]; per.append(float(np.mean([row["correct"] for row in own])))
            tp=sum(r["true_class"]==label and r["predicted_class"]==label for r in rows); fp=sum(r["true_class"]!=label and r["predicted_class"]==label for r in rows); fn=sum(r["true_class"]==label and r["predicted_class"]!=label for r in rows); f1.append(0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
        result.append({"condition_id":condition,"prediction_count":len(rows),"macro_top1_accuracy":float(np.mean(per)),"micro_top1_accuracy":float(np.mean([row["correct"] for row in rows])),"macro_f1":float(np.mean(f1))})
    return tuple(result)


def _render_real(projection: Mapping[str, object]) -> Mapping[str, bytes]:
    colors=dict(zip(PERTURBATIONS,("#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd"),strict=True)); figure1=[]
    for metric in METRIC_OUTPUT_IDS:
        for p in PERTURBATIONS:
            for alpha in ALPHAS[1:]:
                rows=[r for r in projection["class_observations"] if r["metric_output_id"]==metric and r["perturbation_id"]==p and r["alpha"]==alpha]; figure1.append({"metric_output_id":metric,"perturbation_id":p,"alpha":alpha,"mean_metric_harm":float(np.mean([r["metric_harm"] for r in rows])),"mean_downstream_harm":float(np.mean([r["downstream_harm"] for r in rows]))})
    family={(row["metric_output_id"],row["statistic"]):row for row in projection["holm_family"]}; figure2=[]
    for row in projection["alignment_results"]:
        output=dict(row)
        for statistic in ("d_ag","d_acc"):
            source=family.get((row["metric_output_id"],statistic),{}); output.update({f"{statistic}_raw_p":source.get("raw_p_value"),f"{statistic}_adjusted_p":source.get("adjusted_p_value"),f"{statistic}_rank":source.get("rank"),f"{statistic}_favorable":source.get("favorable"),f"{statistic}_rejected":source.get("rejected")})
        figure2.append(output)
    payloads={}
    with matplotlib.rc_context({"font.family":"DejaVu Sans","lines.linewidth":1.5,"svg.hashsalt":"rpe-phase4-d1-protocol-b-v1"}):
        fig,axes=plt.subplots(4,4,figsize=(12,12)); flat=axes.ravel()
        for p in PERTURBATIONS:
            rows=[r for r in figure1 if r["metric_output_id"]=="mse" and r["perturbation_id"]==p]; flat[0].plot([r["alpha"] for r in rows],[r["mean_downstream_harm"] for r in rows],marker="o",color=colors[p])
        flat[0].set_title("downstream_harm"); flat[0].margins(x=0.05,y=0.05)
        for axis,metric in zip(flat[1:],METRIC_OUTPUT_IDS,strict=False):
            for p in PERTURBATIONS:
                rows=[r for r in figure1 if r["metric_output_id"]==metric and r["perturbation_id"]==p]; axis.plot([r["mean_metric_harm"] for r in rows],[r["mean_downstream_harm"] for r in rows],marker="o",color=colors[p])
            axis.set_title(metric); axis.margins(x=0.05,y=0.05)
        for axis in flat[14:]: axis.set_axis_off()
        fig.tight_layout(pad=1.05)
        for ext in ("png","svg"):
            out=io.BytesIO(); fig.savefig(out,format=ext,dpi=300,metadata={"Date":None,"Creator":"raman-preproc-eval"}); payloads[f"figure1_d1_protocol_b_full_domain.{ext}"]=out.getvalue()
        plt.close(fig); fig,axes=plt.subplots(1,4,figsize=(14,8),sharey=True)
        for axis,field,color in zip(axes,("ag","acc_cross","d_ag","d_acc"),("#1f77b4","#ff7f0e","#2ca02c","#d62728"),strict=True): axis.barh(np.arange(len(figure2)),[0.0 if r.get(field) is None else r[field] for r in figure2],color=color); axis.set_title(field); axis.margins(x=0.05,y=0.05)
        fig.tight_layout(pad=1.05)
        for ext in ("png","svg"):
            out=io.BytesIO(); fig.savefig(out,format=ext,dpi=300,metadata={"Date":None,"Creator":"raman-preproc-eval"}); payloads[f"figure2_d1_protocol_b_full_domain.{ext}"]=out.getvalue()
        plt.close(fig)
    return MappingProxyType({**payloads,"figure1_d1_protocol_b_full_domain_data.csv":csv_bytes(tuple(figure1)),"figure2_d1_protocol_b_full_domain_data.csv":csv_bytes(figure2),"d1_protocol_b_full_domain_secondary_table.csv":csv_bytes(figure2)})


def _real_artifact_metadata(
    config: Mapping[str, object],
    bridge: Mapping[str, object],
    alpha0_receipt: Mapping[str, object],
    matrix_receipt_sha256: str,
    projection: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[str, Mapping[str, object], Mapping[str, object], Mapping[str, object]]:
    config_sha256 = sha256_hex(canonical_json_bytes(dict(config)))
    run_document = {
        "config_sha256": config_sha256,
        "bridge_sha256": bridge["bridge_sha256"],
        "model_seeds": list(MODEL_SEEDS),
        "condition_ids": list(CONDITION_IDS),
    }
    run_id = RUN_PREFIX + sha256_hex(canonical_json_bytes(run_document))
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
    counts = {
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


def _serialize_real(config: Mapping[str, object], inputs: _RealInputs, fitted: Mapping[str, object], alpha: Mapping[str, object], projection: Mapping[str, object]) -> Mapping[str, bytes]:
    summary=_summary(fitted["predictions"]); figures=_render_real(projection)
    run_id,bridge_doc,preflight,manifest=_real_artifact_metadata(config,inputs.bridge,alpha,str(fitted["matrix_receipt_sha256"]),projection)
    values={"config.json":DEFAULT_CONFIG.read_bytes(),"authority_bridge.json":canonical_json_bytes(bridge_doc),"preflight.json":canonical_json_bytes(preflight),"alpha0_equivalence.json":canonical_json_bytes(dict(alpha)),"model_cells.jsonl":jsonl_bytes(fitted["model_cells"]),"validation_scores.jsonl":jsonl_bytes(fitted["validation_rows"]),"predictions.jsonl":jsonl_bytes(fitted["predictions"]),"seed_class_conditions.jsonl":jsonl_bytes(fitted["seed_class_rows"]),"condition_summary.csv":csv_bytes(summary),"class_observations.jsonl":jsonl_bytes(projection["class_observations"]),"alignment_results.jsonl":jsonl_bytes(projection["alignment_results"]),"bootstrap_results.jsonl":jsonl_bytes(projection["bootstrap_results"]),"sign_flip_results.jsonl":jsonl_bytes(projection["sign_flip_results"]),"holm_family.jsonl":jsonl_bytes(projection["holm_family"]),"manifest.json":canonical_json_bytes(manifest),**figures}
    payloads={name:values[name] for name in ARTIFACT_PAYLOAD_FILES}
    terminal=canonical_json_bytes({"run_id":run_id,"status":"complete"})
    return MappingProxyType({**payloads,"complete.json":terminal,"SHA256SUMS":write_sha256sums(payloads,"complete.json",terminal)})


def verify_phase4_d1_protocol_b_from_inputs(
    path: Path,
    *,
    inputs: object,
    config_path: Path,
    worker_count: int,
    bootstrap_resamples: int | None = None,
    sign_flip_resamples: int | None = None,
) -> Phase4D1ProtocolBVerifierSummary:
    raw = Path(config_path).read_bytes()
    config = dict(_parse_config(Path(config_path), raw, frozen=False))
    config["_raw"] = raw
    _manifest, _marker = _validate_inventory(Path(path))
    if not bool(config.get("synthetic_fixture", False)):
        raise Phase4D1ProtocolBVerifierError("non-synthetic inputs require the public verifier")
    if bootstrap_resamples is None or sign_flip_resamples is None:
        raise Phase4D1ProtocolBVerifierError("synthetic verifier requires explicit inference resamples")
    rebuilt, manifest = _build_payloads_from_inputs(
        inputs,
        config,
        worker_count=worker_count,
        bootstrap_resamples=int(bootstrap_resamples),
        sign_flip_resamples=int(sign_flip_resamples),
    )
    _compare(Path(path), rebuilt)
    return Phase4D1ProtocolBVerifierSummary(
        Path(path),
        str(json.loads(rebuilt["complete.json"])["run_id"]),
        "complete",
        int(manifest["prediction_row_count"]),
        int(manifest["class_observation_count"]),
    )


def verify_phase4_d1_protocol_b(path: Path, *, worker_count: int = 4) -> Phase4D1ProtocolBVerifierSummary:
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or worker_count <= 0:
        raise Phase4D1ProtocolBVerifierError("worker_count must be positive")
    config = _parse_config(DEFAULT_CONFIG, DEFAULT_CONFIG.read_bytes(), frozen=True)
    if bool(config.get("synthetic_fixture", False)):
        raise Phase4D1ProtocolBVerifierError("public verifier rejects synthetic frozen config")
    artifact_path = Path(path)
    _validate_inventory(artifact_path)
    inputs = _real_inputs(config)
    fitted = _fit_real(inputs, config, worker_count)
    alpha = fitted.get("alpha0_receipt") or _alpha0_qc(fitted, inputs, config)
    if str(alpha["status"]) == "failed":
        rebuilt = _serialize_real_failed(config, inputs, fitted, alpha, _alpha0_failure_receipt(alpha))
    elif str(fitted["status"]) == "failed":
        rebuilt = _serialize_real_failed(config, inputs, fitted, alpha, fitted["failure_receipt"])
    else:
        projection = _aggregate_real(fitted, inputs, config)
        rebuilt = _serialize_real(config, inputs, fitted, alpha, projection)
    _compare(artifact_path, rebuilt)
    manifest = json.loads(rebuilt["manifest.json"])
    return Phase4D1ProtocolBVerifierSummary(
        path=artifact_path,
        run_id=str(manifest["run_id"]),
        status=str(manifest["status"]),
        prediction_row_count=int(manifest["prediction_row_count"]),
        class_observation_count=int(manifest["class_observation_count"]),
    )
