from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from rpe.stats.paired import (
    exact_two_sided_sign_flip,
    paired_seed_bootstrap,
)


SCHEMA_VERSION = "phase05-d2-aggregate-v1"
RESULT_SCHEMA_VERSION = "phase05-d2-complete-cells-v1"
EXPERIMENT_ID = "d2_bacteria_id_few_shot_pca20_lr_sg11"
RUNNER_SHA = (
    "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7"
)
SELECTION_SHA = (
    "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36"
)
CONFIG_BYTES = 882
CONFIG_SHA256 = (
    "25162255aabaf073f3c6365172122513f85b3c3d8a2fa79cbe04d1faf3fe197d"
)
SEEDS = (0, 1, 2, 3, 4)
SHOT_COUNTS = (5, 10, 20)
CONTROL = "released_input_control"
SG = "released_input_plus_sg"
CONDITIONS = (CONTROL, SG)
ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "rpe/runner/d2_aggregate.py",
    "rpe/stats/paired.py",
    "tools/aggregate_phase05_d2.py",
)


class D2AggregateValidationError(ValueError):
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
    raise D2AggregateValidationError(
        "config nonfinite",
        f"unsupported JSON constant {value!r}",
    )


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D2AggregateValidationError(path, "must be an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class D2AggregateConfig:
    path: Path
    sha256: str
    byte_count: int
    seeds: tuple[int, ...]
    shot_counts: tuple[int, ...]
    resamples: int
    confidence_level: float
    random_seed: int
    effect_threshold: float
    significance_threshold: float


def load_d2_aggregate_config(path: Path) -> D2AggregateConfig:
    path = Path(path)
    raw = path.read_bytes()
    try:
        document = json.loads(raw, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as error:
        raise D2AggregateValidationError("config", str(error)) from error
    if raw != _canonical_json_bytes(document):
        raise D2AggregateValidationError(
            "config noncanonical",
            "must use canonical JSON",
        )
    expected = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "runner_config_sha256": RUNNER_SHA,
        "selection_artifact_sha256": SELECTION_SHA,
        "seeds": list(SEEDS),
        "shot_counts": list(SHOT_COUNTS),
        "conditions": list(CONDITIONS),
        "primary_metric": "accuracy",
        "effect": {
            "name": "sg_minus_control_percentage_points",
            "scale": 100.0,
        },
        "bootstrap": {
            "unit": "seed",
            "paired": True,
            "resamples": 2000,
            "confidence_level": 0.95,
            "interval": "percentile",
            "random_seed": 20260817,
        },
        "permutation": {
            "unit": "seed",
            "method": "exact_sign_flip",
            "alternative": "two-sided",
            "zero_difference": "included",
            "patterns": 32,
        },
        "effect_threshold_percentage_points": 2.0,
        "significance_threshold": 0.05,
        "gate_role": "counts_if_any_shot_meets_effect_and_significance",
    }
    if document != expected:
        raise D2AggregateValidationError(
            "config",
            "does not equal the frozen D2 aggregate contract",
        )
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTES or digest != CONFIG_SHA256:
        raise D2AggregateValidationError(
            "config identity",
            "bytes or SHA256 mismatch",
        )
    return D2AggregateConfig(
        path=path,
        sha256=digest,
        byte_count=len(raw),
        seeds=SEEDS,
        shot_counts=SHOT_COUNTS,
        resamples=2000,
        confidence_level=0.95,
        random_seed=20260817,
        effect_threshold=2.0,
        significance_threshold=0.05,
    )


def _read_result(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise D2AggregateValidationError(label, "result does not exist")
    raw = path.read_bytes()
    document = json.loads(raw, parse_constant=_reject_nonfinite)
    if raw != _canonical_json_bytes(document):
        raise D2AggregateValidationError(label, "result is not canonical")
    return document


def _conditions(
    result: Mapping[str, object],
    label: str,
) -> Mapping[str, Mapping[str, object]]:
    raw = result.get("conditions")
    if not isinstance(raw, list) or [
        item.get("condition_id") for item in raw
    ] != list(CONDITIONS):
        raise D2AggregateValidationError(
            f"{label} conditions",
            "must equal control then SG",
        )
    return {item["condition_id"]: _object(label, item) for item in raw}


def _validate_predictions(
    condition: Mapping[str, object],
    label: str,
) -> tuple[tuple[str, ...], tuple[int, ...], float]:
    predictions = condition.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 3000:
        raise D2AggregateValidationError(
            f"{label} predictions",
            "must contain 3,000 rows",
        )
    ids = []
    true = []
    correct = 0
    for item in predictions:
        record_id = item.get("record_id")
        true_class = item.get("true_class")
        predicted = item.get("predicted_class")
        observed_correct = item.get("correct")
        if (
            not isinstance(record_id, str)
            or isinstance(true_class, bool)
            or not isinstance(true_class, int)
            or isinstance(predicted, bool)
            or not isinstance(predicted, int)
            or not isinstance(observed_correct, bool)
            or observed_correct != (true_class == predicted)
        ):
            raise D2AggregateValidationError(
                f"{label} prediction",
                "invalid prediction row",
            )
        ids.append(record_id)
        true.append(true_class)
        correct += int(observed_correct)
    if len(set(ids)) != 3000:
        raise D2AggregateValidationError(
            f"{label} prediction IDs",
            "must be unique",
        )
    metrics = _object(f"{label} metrics", condition.get("metrics"))
    accuracy = metrics.get("accuracy")
    implied = correct / 3000
    if not isinstance(accuracy, (int, float)) or not np.isclose(
        float(accuracy),
        implied,
        rtol=0.0,
        atol=1e-15,
    ):
        raise D2AggregateValidationError(
            f"{label} accuracy mismatch",
            f"reported {accuracy!r}; predictions imply {implied}",
        )
    return tuple(ids), tuple(true), float(accuracy)


def _validate_prediction_artifact(
    root: Path,
    result: Mapping[str, object],
    shot: int,
    seed: int,
) -> Mapping[str, object]:
    path = root / f"predictions_seed{seed}_{shot}shot.jsonl"
    if not path.is_file():
        raise D2AggregateValidationError(
            f"predictions seed {seed} shot {shot}",
            "artifact does not exist",
        )
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if len(lines) != 6000:
        raise D2AggregateValidationError(
            f"predictions seed {seed} shot {shot}",
            "must contain 6,000 lines",
        )
    conditions = _conditions(result, f"seed {seed} shot {shot}")
    offset = 0
    for condition_id in CONDITIONS:
        embedded = conditions[condition_id]["predictions"]
        projected = []
        for line in lines[offset : offset + 3000]:
            item = json.loads(line)
            if line != _canonical_json_bytes(item):
                raise D2AggregateValidationError(
                    f"predictions seed {seed} shot {shot}",
                    "line is not canonical",
                )
            if item.get("condition_id") != condition_id:
                raise D2AggregateValidationError(
                    f"predictions seed {seed} shot {shot}",
                    "condition order mismatch",
                )
            projected.append(
                {key: value for key, value in item.items() if key != "condition_id"}
            )
        if projected != embedded:
            raise D2AggregateValidationError(
                f"prediction projection seed {seed} shot {shot}",
                "does not match embedded predictions",
            )
        offset += 3000
    return {
        "bytes": len(raw),
        "lines": 6000,
        "path": path.name,
        "seed": seed,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "shot_count": shot,
    }


def _validate_contract(
    result: Mapping[str, object],
    shot: int,
    seed: int,
) -> None:
    label = f"seed {seed} shot {shot}"
    if (
        result.get("schema_version") != "phase05-d2-result-v1"
        or result.get("status") != "completed"
        or result.get("result_level") != "provisional"
        or result.get("experiment_id") != EXPERIMENT_ID
        or result.get("seed") != seed
        or result.get("shot_count") != shot
    ):
        raise D2AggregateValidationError(label, "result identity mismatch")
    config = _object(f"{label} config", result.get("config"))
    if config.get("sha256") != RUNNER_SHA:
        raise D2AggregateValidationError(f"{label} config", "SHA mismatch")
    selection = _object(f"{label} selection", result.get("selection"))
    if selection.get("artifact_sha256") != SELECTION_SHA:
        raise D2AggregateValidationError(f"{label} selection", "artifact SHA mismatch")
    split = _object(f"{label} split", result.get("split"))
    expected = {
        "train": (shot * 30, shot),
        "validation": (300, 10),
        "test": (3000, 100),
    }
    for name, (count, per_class) in expected.items():
        part = _object(f"{label} split {name}", split.get(name))
        if part.get("count") != count or part.get("class_counts") != {
            str(class_label): per_class for class_label in range(30)
        }:
            raise D2AggregateValidationError(
                f"{label} split",
                f"{name} contract mismatch",
            )
    audit = _object(f"{label} leakage", result.get("leakage_audit"))
    zero = {
        "train_test": 0,
        "train_validation": 0,
        "validation_test": 0,
    }
    if (
        audit.get("record_id_overlap_counts") != zero
        or audit.get("source_coordinate_overlap_counts") != zero
        or audit.get("unique_source_coordinate_counts")
        != {"train": shot * 30, "validation": 300, "test": 3000}
    ):
        raise D2AggregateValidationError(
            f"{label} leakage",
            "contract mismatch",
        )


def _code_document() -> Mapping[str, Mapping[str, object]]:
    return {
        path: {
            "bytes": (ROOT / path).stat().st_size,
            "sha256": _sha256_file(ROOT / path),
        }
        for path in CODE_PATHS
    }


@dataclass(frozen=True)
class D2AggregateConfig:
    path: Path
    sha256: str
    byte_count: int
    seeds: tuple[int, ...]
    shot_counts: tuple[int, ...]
    resamples: int
    confidence_level: float
    random_seed: int
    effect_threshold: float
    significance_threshold: float


def aggregate_d2_results(
    config_path: Path,
    result_root: Path,
) -> Mapping[str, object]:
    config = load_d2_aggregate_config(config_path)
    result_root = Path(result_root)
    first_dataset = None
    first_pipeline = None
    first_code = None
    first_environment = None
    first_test_digest = None
    cells = {}
    inputs = []
    prediction_artifacts = []
    for shot in config.shot_counts:
        control_values = []
        sg_values = []
        per_seed = []
        for seed in config.seeds:
            path = result_root / f"seed{seed}_{shot}shot.json"
            result = _read_result(path, f"seed {seed} shot {shot}")
            _validate_contract(result, shot, seed)
            conditions = _conditions(result, f"seed {seed} shot {shot}")
            control_ids, control_true, control_accuracy = _validate_predictions(
                conditions[CONTROL], f"seed {seed} shot {shot} control"
            )
            sg_ids, sg_true, sg_accuracy = _validate_predictions(
                conditions[SG], f"seed {seed} shot {shot} SG"
            )
            if control_ids != sg_ids or control_true != sg_true:
                raise D2AggregateValidationError(
                    f"seed {seed} shot {shot} paired prediction",
                    "condition IDs or true labels differ",
                )
            test_digest = hashlib.sha256(
                ("\n".join(control_ids) + "\n").encode()
            ).hexdigest()
            if test_digest != result["split"]["test"]["record_ids_sha256"]:
                raise D2AggregateValidationError(
                    f"seed {seed} shot {shot} test identity",
                    "prediction IDs do not match split digest",
                )
            identities = (
                result.get("dataset"),
                result.get("pipeline_configs"),
                result.get("code"),
                result.get("environment"),
                test_digest,
            )
            if first_dataset is None:
                (
                    first_dataset,
                    first_pipeline,
                    first_code,
                    first_environment,
                    first_test_digest,
                ) = identities
            elif identities != (
                first_dataset,
                first_pipeline,
                first_code,
                first_environment,
                first_test_digest,
            ):
                raise D2AggregateValidationError(
                    f"seed {seed} shot {shot} identity",
                    "dataset/pipeline/code/environment/test drift",
                )
            inputs.append(
                {
                    "bytes": path.stat().st_size,
                    "path": path.name,
                    "seed": seed,
                    "sha256": _sha256_file(path),
                    "shot_count": shot,
                }
            )
            prediction_artifacts.append(
                _validate_prediction_artifact(
                    result_root,
                    result,
                    shot,
                    seed,
                )
            )
            control_values.append(control_accuracy)
            sg_values.append(sg_accuracy)
            per_seed.append(
                {
                    "control_accuracy": control_accuracy,
                    "difference_percentage_points": 100.0
                    * (sg_accuracy - control_accuracy),
                    "seed": seed,
                    "sg_accuracy": sg_accuracy,
                }
            )
        bootstrap = paired_seed_bootstrap(
            control_values,
            sg_values,
            resamples=config.resamples,
            confidence_level=config.confidence_level,
            random_seed=config.random_seed,
            scale=100.0,
        )
        differences = [
            right - left
            for left, right in zip(control_values, sg_values, strict=True)
        ]
        permutation = exact_two_sided_sign_flip(differences)
        effect = bootstrap["difference_mean"] > config.effect_threshold
        significant = permutation["p_value"] < config.significance_threshold
        cells[str(shot)] = {
            "bootstrap": bootstrap,
            "control_mean_percentage": bootstrap["left_mean"],
            "difference_mean_percentage_points": bootstrap["difference_mean"],
            "effect_exceeds_threshold": effect,
            "gate_success": effect and significant,
            "per_seed": per_seed,
            "permutation": permutation,
            "sg_mean_percentage": bootstrap["right_mean"],
            "shot_count": shot,
            "significant_at_configured_threshold": significant,
        }
    successes = [
        int(shot)
        for shot, cell in cells.items()
        if cell["gate_success"]
    ]
    return {
        "aggregate_code": _code_document(),
        "aggregate_config": {
            "bytes": config.byte_count,
            "path": config.path.name,
            "sha256": config.sha256,
        },
        "cells": cells,
        "conditions": list(CONDITIONS),
        "d2_gate_success": bool(successes),
        "effect_threshold_percentage_points": config.effect_threshold,
        "experiment_id": EXPERIMENT_ID,
        "gate_role": "counts_if_any_shot_meets_effect_and_significance",
        "gate_success_shot_counts": successes,
        "input_results": inputs,
        "prediction_artifacts": prediction_artifacts,
        "primary_metric": "accuracy",
        "result_level": "complete_cells",
        "runner_config_sha256": RUNNER_SHA,
        "schema_version": RESULT_SCHEMA_VERSION,
        "seeds": list(config.seeds),
        "selection_artifact_sha256": SELECTION_SHA,
        "shot_counts": list(config.shot_counts),
        "significance_threshold": config.significance_threshold,
        "status": "completed",
        "test_record_ids_sha256": first_test_digest,
    }


__all__ = [
    "D2AggregateConfig",
    "D2AggregateValidationError",
    "aggregate_d2_results",
    "load_d2_aggregate_config",
]
