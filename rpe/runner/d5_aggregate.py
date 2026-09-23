from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np

from rpe.downstream.rruff import (
    CLASS_LABELS_SHA256,
    CONFIG_SHA256,
    EXPECTED_CLASS_COUNT,
    EXPECTED_GROUP_COUNT,
    EXPECTED_RECORD_COUNT,
    EXPECTED_RRUFF_ID_COUNT,
    GROUP_IDS_SHA256,
    RECORD_IDS_SHA256,
)
from rpe.downstream.rruff_matching import (
    CONDITION_IDS,
    CONTROL_CONDITION_ID,
    SG_CONDITION_ID,
)
from rpe.runner.d5_rruff import (
    EXPERIMENT_ID,
    FROZEN_SPLITS,
    RETAINED_MATRIX_SHA256,
    SG_CONFIG_SHA256,
)


SCHEMA_VERSION = "phase05-d5-complete-cell-v1"
RESULT_SCHEMA_VERSION = "phase05-d5-result-v1"
SEEDS = (0, 1, 2, 3, 4)
CLASS_COUNT = 681
BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_RESAMPLES = 100_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260817
EFFECT_THRESHOLD_PERCENTAGE_POINTS = 2.0
SIGNIFICANCE_THRESHOLD = 0.05
ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "rpe/runner/d5_aggregate.py",
    "tools/aggregate_phase05_d5.py",
)
RUNNER_CODE_PATHS = (
    "rpe/downstream/rruff.py",
    "rpe/downstream/rruff_matching.py",
    "rpe/methods/classical/savitzky_golay.py",
    "rpe/runner/d5_rruff.py",
    "tools/run_phase05_d5.py",
)


class D5AggregateValidationError(ValueError):
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
    raise D5AggregateValidationError(
        "nonfinite",
        f"unsupported JSON constant {value!r}",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D5AggregateValidationError(path, "must be an object")
    return value


def _expected_dataset() -> dict[str, object]:
    return {
        "class_count": EXPECTED_CLASS_COUNT,
        "class_labels_sha256": CLASS_LABELS_SHA256,
        "dataset_id": "rruff_raman_raw",
        "group_count": EXPECTED_GROUP_COUNT,
        "group_ids_sha256": GROUP_IDS_SHA256,
        "matrix_sha256": RETAINED_MATRIX_SHA256,
        "record_count": EXPECTED_RECORD_COUNT,
        "record_ids_sha256": RECORD_IDS_SHA256,
        "rruff_id_count": EXPECTED_RRUFF_ID_COUNT,
    }


def _expected_runner_code() -> dict[str, object]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
        for relative_path in RUNNER_CODE_PATHS
    }


def _read_canonical_json(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise D5AggregateValidationError(label, "file does not exist")
    raw = path.read_bytes()
    try:
        document = json.loads(raw, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as error:
        raise D5AggregateValidationError(label, str(error)) from error
    if raw != _canonical_json_bytes(document):
        raise D5AggregateValidationError(label, "must use canonical JSON")
    return _object(label, document)


def _validate_outcome(
    row: Mapping[str, object],
    *,
    seed: int,
) -> tuple[int, str, dict[str, tuple[bool, bool]]]:
    class_label = row.get("true_class_label")
    mineral_name = row.get("mineral_name")
    if (
        isinstance(class_label, bool)
        or not isinstance(class_label, int)
        or class_label < 0
        or not isinstance(mineral_name, str)
        or mineral_name == ""
    ):
        raise D5AggregateValidationError(
            f"seed {seed} class metadata",
            "invalid class label or mineral name",
        )
    conditions = _object(
        f"seed {seed} outcome conditions",
        row.get("conditions"),
    )
    if set(conditions) != set(CONDITION_IDS):
        raise D5AggregateValidationError(
            f"seed {seed} conditions",
            "must equal control and SG",
        )
    correctness = {}
    for condition_id in CONDITION_IDS:
        condition = _object(
            f"seed {seed} {condition_id}",
            conditions[condition_id],
        )
        top1_label = condition.get("top1_class_label")
        top1_correct = condition.get("top1_correct")
        top5_labels = condition.get("top5_class_labels")
        top5_scores = condition.get("top5_scores")
        top5_correct = condition.get("top5_correct")
        if (
            isinstance(top1_label, bool)
            or not isinstance(top1_label, int)
            or not isinstance(top1_correct, bool)
            or not isinstance(top5_labels, list)
            or not 1 <= len(top5_labels) <= 5
            or len(set(top5_labels)) != len(top5_labels)
            or not isinstance(top5_scores, list)
            or len(top5_scores) != len(top5_labels)
            or condition.get("top1_score") != top5_scores[0]
            or top1_label != top5_labels[0]
            or top1_correct != (top1_label == class_label)
            or not isinstance(top5_correct, bool)
            or top5_correct != (class_label in top5_labels)
        ):
            raise D5AggregateValidationError(
                f"seed {seed} correctness",
                "top-k labels, scores, or correctness are inconsistent",
            )
        for index, score in enumerate(top5_scores):
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not np.isfinite(float(score))
                or not -1.0 <= float(score) <= 1.0
            ):
                raise D5AggregateValidationError(
                    f"seed {seed} correctness",
                    "top-k score is invalid",
                )
            if index:
                previous = float(top5_scores[index - 1])
                previous_label = top5_labels[index - 1]
                if float(score) > previous or (
                    float(score) == previous
                    and top5_labels[index] < previous_label
                ):
                    raise D5AggregateValidationError(
                        f"seed {seed} correctness",
                        "top-k ranking order is invalid",
                    )
        correctness[condition_id] = (top1_correct, top5_correct)
    return class_label, mineral_name, correctness


def _validate_seed_result(
    root: Path,
    seed: int,
) -> tuple[
    Mapping[str, object],
    list[Mapping[str, object]],
    Mapping[str, object],
    Mapping[str, object],
]:
    result_path = root / f"seed{seed}_complete.json"
    payload_path = root / f"paired_outcomes_seed{seed}_complete.jsonl"
    result = _read_canonical_json(result_path, f"seed {seed} result")
    if (
        result.get("schema_version") != RESULT_SCHEMA_VERSION
        or result.get("experiment_id") != EXPERIMENT_ID
        or result.get("status") != "completed"
        or result.get("result_level") != "complete_seed"
        or result.get("result_label") != f"COMPLETE SEED — {seed + 1}/5"
        or result.get("seed") != seed
        or result.get("seeds_completed") != seed + 1
        or result.get("seeds_required") != 5
    ):
        raise D5AggregateValidationError(
            f"seed {seed} identity",
            "result identity mismatch",
        )
    protocol = _object(
        f"seed {seed} protocol",
        result.get("protocol_config"),
    )
    if protocol.get("sha256") != CONFIG_SHA256:
        raise D5AggregateValidationError(
            f"seed {seed} protocol",
            "SHA256 mismatch",
        )
    sg_config = _object(
        f"seed {seed} SG config",
        result.get("sg_config"),
    )
    if sg_config.get("sha256") != SG_CONFIG_SHA256:
        raise D5AggregateValidationError(
            f"seed {seed} SG config",
            "SHA256 mismatch",
        )
    dataset = _object(
        f"seed {seed} dataset",
        result.get("dataset"),
    )
    if (
        dataset.get("dataset_id") != "rruff_raman_raw"
        or dataset.get("class_count") != CLASS_COUNT
        or dataset.get("record_count") != 3770
        or dataset.get("rruff_id_count") != 1936
        or dataset.get("group_count") != 1934
    ):
        raise D5AggregateValidationError(
            f"seed {seed} dataset",
            "frozen cohort counts mismatch",
        )
    outcomes = result.get("paired_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise D5AggregateValidationError(
            f"seed {seed} paired outcomes",
            "must be a nonempty list",
        )
    if not payload_path.is_file():
        raise D5AggregateValidationError(
            f"seed {seed} paired outcome artifact",
            "file does not exist",
        )
    payload = payload_path.read_bytes()
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(outcomes):
        raise D5AggregateValidationError(
            f"seed {seed} paired outcome artifact",
            "line count mismatch",
        )
    projected = []
    for line in lines:
        row = json.loads(line, parse_constant=_reject_nonfinite)
        if line != _canonical_json_bytes(row):
            raise D5AggregateValidationError(
                f"seed {seed} paired outcome artifact",
                "line is not canonical",
            )
        projected.append(row)
    if projected != outcomes:
        raise D5AggregateValidationError(
            f"seed {seed} paired outcome artifact",
            "does not equal embedded outcomes",
        )
    artifact = result.get("paired_outcomes_artifact")
    expected_artifact = {
        "bytes": len(payload),
        "lines": len(lines),
        "path": payload_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if artifact != expected_artifact:
        raise D5AggregateValidationError(
            f"seed {seed} paired outcome artifact",
            "bytes, lines, path, or SHA256 mismatch",
        )
    class_counts: dict[int, int] = defaultdict(int)
    record_ids = set()
    for row in outcomes:
        document = _object(f"seed {seed} outcome", row)
        class_label, _, _ = _validate_outcome(document, seed=seed)
        class_counts[class_label] += 1
        record_id = document.get("record_id")
        if not isinstance(record_id, str) or record_id in record_ids:
            raise D5AggregateValidationError(
                f"seed {seed} paired outcomes",
                "record IDs must be unique strings",
            )
        record_ids.add(record_id)
    if len(class_counts) != CLASS_COUNT:
        raise D5AggregateValidationError(
            f"seed {seed} classes",
            f"must contain {CLASS_COUNT} classes",
        )
    split = _object(f"seed {seed} split", result.get("split"))
    if (
        split.get("seed") != seed
        or split.get("class_count") != CLASS_COUNT
        or split.get("query_count") != len(outcomes)
    ):
        raise D5AggregateValidationError(
            f"seed {seed} split",
            "seed, class, or query count mismatch",
        )
    for key, expected in FROZEN_SPLITS[seed].items():
        if split.get(key) != expected:
            raise D5AggregateValidationError(
                f"seed {seed} frozen split",
                f"{key} must equal {expected!r}",
            )
    return (
        result,
        outcomes,
        {
            "bytes": result_path.stat().st_size,
            "path": result_path.name,
            "seed": seed,
            "sha256": _sha256_file(result_path),
        },
        {
            **expected_artifact,
            "seed": seed,
        },
    )


def _paired_cluster_statistics(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if (
        left.shape != (CLASS_COUNT,)
        or right.shape != (CLASS_COUNT,)
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise D5AggregateValidationError(
            "class values",
            f"must be two finite vectors of length {CLASS_COUNT}",
        )
    differences = right - left
    generator = np.random.default_rng(RANDOM_SEED)
    indices = generator.integers(
        0,
        CLASS_COUNT,
        size=(BOOTSTRAP_RESAMPLES, CLASS_COUNT),
    )
    left_means = left[indices].mean(axis=1) * 100.0
    right_means = right[indices].mean(axis=1) * 100.0
    difference_means = differences[indices].mean(axis=1) * 100.0
    alpha = (1.0 - CONFIDENCE_LEVEL) / 2.0

    def interval(values: np.ndarray) -> list[float]:
        lower, upper = np.quantile(
            values,
            [alpha, 1.0 - alpha],
        )
        return [float(lower), float(upper)]

    bootstrap = {
        "confidence_level": CONFIDENCE_LEVEL,
        "difference_ci": interval(difference_means),
        "difference_mean": float(differences.mean() * 100.0),
        "interval": "percentile",
        "left_ci": interval(left_means),
        "left_mean": float(left.mean() * 100.0),
        "paired": True,
        "random_seed": RANDOM_SEED,
        "resamples": BOOTSTRAP_RESAMPLES,
        "right_ci": interval(right_means),
        "right_mean": float(right.mean() * 100.0),
        "sample_size": CLASS_COUNT,
        "unit": "mineral_class_cluster",
    }

    observed = float(differences.mean())
    threshold = abs(observed)
    extreme = 0
    completed = 0
    permutation_generator = np.random.default_rng(RANDOM_SEED)
    batch_size = 1024
    while completed < PERMUTATION_RESAMPLES:
        current = min(batch_size, PERMUTATION_RESAMPLES - completed)
        bits = permutation_generator.integers(
            0,
            2,
            size=(current, CLASS_COUNT),
            dtype=np.int8,
        )
        signs = bits * np.int8(2) - np.int8(1)
        permuted = (
            signs.astype(np.float64, copy=False)
            * differences[None, :]
        ).mean(axis=1)
        extreme += int(np.sum(np.abs(permuted) + 1e-15 >= threshold))
        completed += current
    permutation = {
        "alternative": "two-sided",
        "extreme_resamples": extreme,
        "method": "monte_carlo_paired_sign_flip",
        "observed_mean": observed,
        "p_value": (extreme + 1) / (PERMUTATION_RESAMPLES + 1),
        "p_value_correction": "plus_one",
        "random_seed": RANDOM_SEED,
        "resamples": PERMUTATION_RESAMPLES,
        "sample_size": CLASS_COUNT,
        "unit": "mineral_class_cluster",
        "zero_difference": "included",
    }
    return bootstrap, permutation


def _metric_summary(
    left: np.ndarray,
    right: np.ndarray,
) -> Mapping[str, object]:
    bootstrap, permutation = _paired_cluster_statistics(left, right)
    return {
        "bootstrap": bootstrap,
        "control_mean_percentage": float(left.mean() * 100.0),
        "difference_mean_percentage_points": float(
            (right - left).mean() * 100.0
        ),
        "permutation": permutation,
        "sg_mean_percentage": float(right.mean() * 100.0),
    }


def aggregate_d5_results(result_root: Path) -> dict[str, object]:
    result_root = Path(result_root)
    seed_results = []
    input_results = []
    payload_artifacts = []
    baseline_dataset = None
    baseline_code = None
    expected_dataset = _expected_dataset()
    expected_runner_code = _expected_runner_code()
    class_metadata: dict[int, str] = {}
    values: dict[int, dict[str, dict[str, list[bool]]]] = defaultdict(
        lambda: {
            CONTROL_CONDITION_ID: {"top1": [], "top5": []},
            SG_CONDITION_ID: {"top1": [], "top5": []},
        }
    )
    per_seed = []
    for seed in SEEDS:
        result, outcomes, input_document, payload_document = (
            _validate_seed_result(result_root, seed)
        )
        dataset = result["dataset"]
        code = result["code"]
        if dataset != expected_dataset:
            raise D5AggregateValidationError(
                f"seed {seed} current dataset",
                "does not equal the frozen retained cohort",
            )
        if code != expected_runner_code:
            raise D5AggregateValidationError(
                f"seed {seed} current code",
                "does not equal current runner sources",
            )
        if baseline_dataset is None:
            baseline_dataset = dataset
            baseline_code = code
        elif dataset != baseline_dataset:
            raise D5AggregateValidationError(
                f"seed {seed} dataset",
                "differs across seeds",
            )
        elif code != baseline_code:
            raise D5AggregateValidationError(
                f"seed {seed} code",
                "differs across seeds",
            )
        seed_class_values = {
            condition_id: {"top1": [], "top5": []}
            for condition_id in CONDITION_IDS
        }
        by_class: dict[
            int,
            dict[str, dict[str, list[bool]]],
        ] = defaultdict(
            lambda: {
                CONTROL_CONDITION_ID: {"top1": [], "top5": []},
                SG_CONDITION_ID: {"top1": [], "top5": []},
            }
        )
        for raw_row in outcomes:
            row = _object(f"seed {seed} outcome", raw_row)
            class_label, mineral_name, correctness = _validate_outcome(
                row,
                seed=seed,
            )
            previous_name = class_metadata.setdefault(
                class_label,
                mineral_name,
            )
            if previous_name != mineral_name:
                raise D5AggregateValidationError(
                    f"seed {seed} class metadata",
                    f"class {class_label} mineral name differs",
                )
            for condition_id, (top1, top5) in correctness.items():
                by_class[class_label][condition_id]["top1"].append(top1)
                by_class[class_label][condition_id]["top5"].append(top5)
                values[class_label][condition_id]["top1"].append(top1)
                values[class_label][condition_id]["top5"].append(top5)
        if set(by_class) != set(class_metadata):
            raise D5AggregateValidationError(
                f"seed {seed} classes",
                "class set differs across seeds",
            )
        for class_label in sorted(by_class):
            for condition_id in CONDITION_IDS:
                for metric in ("top1", "top5"):
                    seed_class_values[condition_id][metric].append(
                        float(
                            np.mean(
                                by_class[class_label][condition_id][metric]
                            )
                        )
                    )
        per_seed.append(
            {
                "control_top1_macro_class_accuracy": float(
                    np.mean(seed_class_values[CONTROL_CONDITION_ID]["top1"])
                ),
                "control_top5_macro_class_accuracy": float(
                    np.mean(seed_class_values[CONTROL_CONDITION_ID]["top5"])
                ),
                "query_count": len(outcomes),
                "seed": seed,
                "sg_top1_macro_class_accuracy": float(
                    np.mean(seed_class_values[SG_CONDITION_ID]["top1"])
                ),
                "sg_top5_macro_class_accuracy": float(
                    np.mean(seed_class_values[SG_CONDITION_ID]["top5"])
                ),
            }
        )
        seed_results.append(result)
        input_results.append(input_document)
        payload_artifacts.append(payload_document)

    if len(class_metadata) != CLASS_COUNT:
        raise D5AggregateValidationError(
            "class metadata",
            f"must contain {CLASS_COUNT} classes",
        )
    per_class = []
    vectors = {
        condition_id: {"top1": [], "top5": []}
        for condition_id in CONDITION_IDS
    }
    for class_label in sorted(class_metadata):
        row = {
            "class_label": class_label,
            "mineral_name": class_metadata[class_label],
            "query_count": len(
                values[class_label][CONTROL_CONDITION_ID]["top1"]
            ),
        }
        for condition_id in CONDITION_IDS:
            for metric in ("top1", "top5"):
                vector = values[class_label][condition_id][metric]
                if not vector:
                    raise D5AggregateValidationError(
                        f"class {class_label}",
                        "has no paired values",
                    )
                mean = float(np.mean(vector))
                vectors[condition_id][metric].append(mean)
                row[f"{condition_id}_{metric}_accuracy"] = mean
        row["top1_difference_percentage_points"] = 100.0 * (
            row[f"{SG_CONDITION_ID}_top1_accuracy"]
            - row[f"{CONTROL_CONDITION_ID}_top1_accuracy"]
        )
        row["top5_difference_percentage_points"] = 100.0 * (
            row[f"{SG_CONDITION_ID}_top5_accuracy"]
            - row[f"{CONTROL_CONDITION_ID}_top5_accuracy"]
        )
        per_class.append(row)

    top1_control = np.asarray(
        vectors[CONTROL_CONDITION_ID]["top1"],
        dtype=np.float64,
    )
    top1_sg = np.asarray(
        vectors[SG_CONDITION_ID]["top1"],
        dtype=np.float64,
    )
    top5_control = np.asarray(
        vectors[CONTROL_CONDITION_ID]["top5"],
        dtype=np.float64,
    )
    top5_sg = np.asarray(
        vectors[SG_CONDITION_ID]["top5"],
        dtype=np.float64,
    )
    primary = _metric_summary(top1_control, top1_sg)
    secondary = _metric_summary(top5_control, top5_sg)
    effect = (
        primary["difference_mean_percentage_points"]
        > EFFECT_THRESHOLD_PERCENTAGE_POINTS
    )
    significant = (
        primary["permutation"]["p_value"] < SIGNIFICANCE_THRESHOLD
    )
    return {
        "aggregate_code": {
            relative_path: {
                "bytes": (ROOT / relative_path).stat().st_size,
                "sha256": _sha256_file(ROOT / relative_path),
            }
            for relative_path in CODE_PATHS
        },
        "class_count": CLASS_COUNT,
        "conditions": list(CONDITION_IDS),
        "d5_gate_success": effect and significant,
        "effect_exceeds_threshold": effect,
        "effect_threshold_percentage_points": (
            EFFECT_THRESHOLD_PERCENTAGE_POINTS
        ),
        "experiment_id": EXPERIMENT_ID,
        "input_results": input_results,
        "paired_outcome_artifacts": payload_artifacts,
        "per_class": per_class,
        "per_seed": per_seed,
        "primary_metric": "top1_macro_class_accuracy",
        "primary_top1_macro_class_summary": primary,
        "protocol_config_sha256": CONFIG_SHA256,
        "result_level": "complete_cell",
        "schema_version": SCHEMA_VERSION,
        "secondary_top5_macro_class_summary": secondary,
        "seeds": list(SEEDS),
        "significance_threshold": SIGNIFICANCE_THRESHOLD,
        "significant_at_configured_threshold": significant,
        "statistics_contract": {
            "bootstrap": {
                "confidence_level": CONFIDENCE_LEVEL,
                "interval": "percentile",
                "paired": True,
                "random_seed": RANDOM_SEED,
                "resamples": BOOTSTRAP_RESAMPLES,
            },
            "per_class_value": (
                "mean_query_correctness_across_all_records_and_five_splits"
            ),
            "permutation": {
                "alternative": "two-sided",
                "method": "monte_carlo_paired_sign_flip",
                "p_value_correction": "plus_one",
                "random_seed": RANDOM_SEED,
                "resamples": PERMUTATION_RESAMPLES,
            },
            "unit": "mineral_class_cluster",
        },
        "statistics_unit": "mineral_class_cluster",
        "status": "completed",
    }


__all__ = [
    "D5AggregateValidationError",
    "aggregate_d5_results",
]
