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


SCHEMA_VERSION = "phase05-d1-aggregate-v1"
RESULT_SCHEMA_VERSION = "phase05-d1-complete-cell-v1"
EXPERIMENT_ID = "d1_bacteria_id_pca20_lr_sg11"
RUNNER_CONFIG_SHA256 = (
    "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"
)
AGGREGATE_CONFIG_SHA256 = (
    "fd72d323beaaf1c9d4d52b7dee6a0bc1c8a4dc5987b3da218c8f480d6e6f4051"
)
AGGREGATE_CONFIG_BYTES = 695
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
CONTROL = "released_input_control"
SG = "released_input_plus_sg"
EXPECTED_CONDITIONS = (CONTROL, SG)
ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_CODE_PATHS = (
    "rpe/runner/d1_aggregate.py",
    "rpe/stats/paired.py",
    "tools/aggregate_phase05.py",
)


class D1AggregateValidationError(ValueError):
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
    raise D1AggregateValidationError(
        "config nonfinite",
        f"unsupported JSON constant {value!r}",
    )


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D1AggregateValidationError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise D1AggregateValidationError(path, "keys must be strings")
    return value


def _exact_keys(
    path: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        raise D1AggregateValidationError(
            path,
            (
                f"key mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
        )


def _equal(path: str, observed: object, expected: object) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise D1AggregateValidationError(
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
class D1AggregateConfig:
    path: Path
    sha256: str
    byte_count: int
    seeds: tuple[int, ...]
    resamples: int
    confidence_level: float
    random_seed: int
    sign_flip_patterns: int
    effect_threshold_percentage_points: float


def load_d1_aggregate_config(path: Path) -> D1AggregateConfig:
    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    except D1AggregateValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D1AggregateValidationError(
            path.name or "config",
            str(error),
        ) from error
    document = _object("config", value)
    if raw != _canonical_json_bytes(document):
        raise D1AggregateValidationError(
            "config noncanonical",
            "must use canonical JSON encoding",
        )
    _exact_keys(
        "config",
        document,
        {
            "schema_version",
            "experiment_id",
            "runner_config_sha256",
            "seeds",
            "conditions",
            "primary_metric",
            "effect",
            "bootstrap",
            "permutation",
            "effect_threshold_percentage_points",
            "gate_role",
        },
    )
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("experiment_id", EXPERIMENT_ID),
        ("runner_config_sha256", RUNNER_CONFIG_SHA256),
        ("seeds", list(EXPECTED_SEEDS)),
        ("conditions", list(EXPECTED_CONDITIONS)),
        ("primary_metric", "accuracy"),
        ("effect_threshold_percentage_points", 2.0),
        ("gate_role", "control_not_counted"),
    ):
        _equal(key, document[key], expected)
    effect = _object("effect", document["effect"])
    _exact_keys("effect", effect, {"name", "scale"})
    _equal(
        "effect.name",
        effect["name"],
        "sg_minus_control_percentage_points",
    )
    _equal("effect.scale", effect["scale"], 100.0)
    bootstrap = _object("bootstrap", document["bootstrap"])
    _exact_keys(
        "bootstrap",
        bootstrap,
        {
            "unit",
            "paired",
            "resamples",
            "confidence_level",
            "interval",
            "random_seed",
        },
    )
    for key, expected in (
        ("unit", "seed"),
        ("paired", True),
        ("resamples", 2000),
        ("confidence_level", 0.95),
        ("interval", "percentile"),
        ("random_seed", 20260817),
    ):
        _equal(f"bootstrap.{key}", bootstrap[key], expected)
    permutation = _object("permutation", document["permutation"])
    _exact_keys(
        "permutation",
        permutation,
        {
            "unit",
            "method",
            "alternative",
            "zero_difference",
            "patterns",
        },
    )
    for key, expected in (
        ("unit", "seed"),
        ("method", "exact_sign_flip"),
        ("alternative", "two-sided"),
        ("zero_difference", "included"),
        ("patterns", 32),
    ):
        _equal(f"permutation.{key}", permutation[key], expected)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != AGGREGATE_CONFIG_BYTES:
        raise D1AggregateValidationError(
            "config bytes",
            f"must equal {AGGREGATE_CONFIG_BYTES}",
        )
    if digest != AGGREGATE_CONFIG_SHA256:
        raise D1AggregateValidationError(
            "config sha256",
            f"must equal {AGGREGATE_CONFIG_SHA256}",
        )
    return D1AggregateConfig(
        path=path,
        sha256=digest,
        byte_count=len(raw),
        seeds=EXPECTED_SEEDS,
        resamples=2000,
        confidence_level=0.95,
        random_seed=20260817,
        sign_flip_patterns=32,
        effect_threshold_percentage_points=2.0,
    )


def _read_canonical_json(path: Path, error_path: str) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    except D1AggregateValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D1AggregateValidationError(error_path, str(error)) from error
    document = _object(error_path, value)
    if raw != _canonical_json_bytes(document):
        raise D1AggregateValidationError(
            f"{error_path} noncanonical",
            "must use canonical JSON encoding",
        )
    return document


def _condition_map(
    result: Mapping[str, object],
    seed: int,
) -> Mapping[str, Mapping[str, object]]:
    raw = result.get("conditions")
    if not isinstance(raw, list) or len(raw) != 2:
        raise D1AggregateValidationError(
            f"seed {seed} conditions",
            "must contain exactly two conditions",
        )
    observed = []
    mapping = {}
    for index, item in enumerate(raw):
        condition = _object(
            f"seed {seed} conditions[{index}]",
            item,
        )
        condition_id = condition.get("condition_id")
        if not isinstance(condition_id, str):
            raise D1AggregateValidationError(
                f"seed {seed} condition_id",
                "must be a string",
            )
        observed.append(condition_id)
        mapping[condition_id] = condition
    if tuple(observed) != EXPECTED_CONDITIONS:
        raise D1AggregateValidationError(
            f"seed {seed} conditions",
            f"must equal {EXPECTED_CONDITIONS!r}",
        )
    return mapping


def _validate_prediction_artifact(
    result_root: Path,
    result: Mapping[str, object],
    seed: int,
) -> Mapping[str, object]:
    path = result_root / f"predictions_{seed}.jsonl"
    if not path.is_file():
        raise D1AggregateValidationError(
            f"predictions seed {seed}",
            "artifact does not exist",
        )
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    conditions = _condition_map(result, seed)
    expected_count = sum(
        len(condition.get("predictions", []))
        for condition in conditions.values()
    )
    if len(lines) != expected_count:
        raise D1AggregateValidationError(
            f"predictions seed {seed}",
            f"expected {expected_count} lines; observed {len(lines)}",
        )
    offset = 0
    for condition_id in EXPECTED_CONDITIONS:
        embedded = conditions[condition_id].get("predictions")
        if not isinstance(embedded, list):
            raise D1AggregateValidationError(
                f"seed {seed} {condition_id} predictions",
                "must be a list",
            )
        projected = []
        for line_index, line in enumerate(
            lines[offset : offset + len(embedded)],
            start=offset,
        ):
            try:
                item = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise D1AggregateValidationError(
                    f"predictions seed {seed} line {line_index}",
                    str(error),
                ) from error
            parsed = _object(
                f"predictions seed {seed} line {line_index}",
                item,
            )
            if line != _canonical_json_bytes(parsed):
                raise D1AggregateValidationError(
                    f"predictions seed {seed} line {line_index}",
                    "must use canonical JSON encoding",
                )
            _exact_keys(
                f"predictions seed {seed} line {line_index}",
                parsed,
                {
                    "condition_id",
                    "record_id",
                    "true_class",
                    "predicted_class",
                    "correct",
                },
            )
            if parsed["condition_id"] != condition_id:
                raise D1AggregateValidationError(
                    f"predictions seed {seed} line {line_index}",
                    f"condition_id must equal {condition_id!r}",
                )
            projected.append(
                {
                    key: value
                    for key, value in parsed.items()
                    if key != "condition_id"
                }
            )
        if projected != embedded:
            raise D1AggregateValidationError(
                f"prediction projection seed {seed}",
                f"does not match embedded {condition_id} predictions",
            )
        offset += len(embedded)
    return {
        "bytes": len(raw),
        "lines": len(lines),
        "path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _metric(
    condition: Mapping[str, object],
    seed: int,
    condition_id: str,
) -> float:
    metrics = _object(
        f"seed {seed} {condition_id} metrics",
        condition.get("metrics"),
    )
    accuracy = metrics.get("accuracy")
    if (
        isinstance(accuracy, bool)
        or not isinstance(accuracy, (int, float))
        or not np.isfinite(float(accuracy))
        or not 0.0 <= float(accuracy) <= 1.0
    ):
        raise D1AggregateValidationError(
            f"seed {seed} {condition_id} accuracy",
            "must be finite and in [0, 1]",
        )
    return float(accuracy)


def _validated_predictions(
    condition: Mapping[str, object],
    seed: int,
    condition_id: str,
) -> tuple[tuple[str, ...], tuple[int, ...], float]:
    raw = condition.get("predictions")
    if not isinstance(raw, list) or not raw:
        raise D1AggregateValidationError(
            f"seed {seed} {condition_id} predictions",
            "must be a non-empty list",
        )
    record_ids = []
    true_classes = []
    correct_count = 0
    for index, item in enumerate(raw):
        parsed = _object(
            f"seed {seed} {condition_id} predictions[{index}]",
            item,
        )
        _exact_keys(
            f"seed {seed} {condition_id} predictions[{index}]",
            parsed,
            {"record_id", "true_class", "predicted_class", "correct"},
        )
        record_id = parsed["record_id"]
        true_class = parsed["true_class"]
        predicted_class = parsed["predicted_class"]
        correct = parsed["correct"]
        if not isinstance(record_id, str) or not record_id:
            raise D1AggregateValidationError(
                f"seed {seed} {condition_id} record_id",
                "must be a non-empty string",
            )
        if (
            isinstance(true_class, bool)
            or not isinstance(true_class, int)
            or isinstance(predicted_class, bool)
            or not isinstance(predicted_class, int)
        ):
            raise D1AggregateValidationError(
                f"seed {seed} {condition_id} class",
                "true and predicted classes must be integers",
            )
        if not isinstance(correct, bool) or correct != (
            predicted_class == true_class
        ):
            raise D1AggregateValidationError(
                f"seed {seed} {condition_id} correct",
                "does not match true and predicted classes",
            )
        record_ids.append(record_id)
        true_classes.append(true_class)
        correct_count += int(correct)
    if len(record_ids) != len(set(record_ids)):
        raise D1AggregateValidationError(
            f"seed {seed} {condition_id} record_id",
            "must be unique",
        )
    observed_accuracy = correct_count / len(raw)
    reported_accuracy = _metric(condition, seed, condition_id)
    if not np.isclose(
        observed_accuracy,
        reported_accuracy,
        rtol=0.0,
        atol=1e-15,
    ):
        raise D1AggregateValidationError(
            f"seed {seed} {condition_id} accuracy mismatch",
            (
                f"reported {reported_accuracy}; "
                f"predictions imply {observed_accuracy}"
            ),
        )
    return (
        tuple(record_ids),
        tuple(true_classes),
        reported_accuracy,
    )


def _validate_d1_contract(
    result: Mapping[str, object],
    seed: int,
) -> None:
    dataset = _object(f"seed {seed} dataset", result.get("dataset"))
    if dataset.get("record_count") != 66000 or dataset.get(
        "split_counts"
    ) != {
        "finetune": 3000,
        "reference": 60000,
        "test": 3000,
    }:
        raise D1AggregateValidationError(
            f"seed {seed} split contract",
            "dataset counts do not match the retained D1 contract",
        )
    split = _object(f"seed {seed} split", result.get("split"))
    if split.get("validation_per_class") != 10:
        raise D1AggregateValidationError(
            f"seed {seed} split contract",
            "validation_per_class must equal 10",
        )
    expected = {
        "train": (62700, 2090),
        "validation": (300, 10),
        "test": (3000, 100),
    }
    for split_name, (expected_count, expected_per_class) in expected.items():
        document = _object(
            f"seed {seed} split.{split_name}",
            split.get(split_name),
        )
        if document.get("count") != expected_count:
            raise D1AggregateValidationError(
                f"seed {seed} split contract",
                (
                    f"{split_name} count must equal {expected_count}; "
                    f"observed {document.get('count')!r}"
                ),
            )
        if document.get("class_counts") != {
            str(class_label): expected_per_class
            for class_label in range(30)
        }:
            raise D1AggregateValidationError(
                f"seed {seed} split contract",
                f"{split_name} class counts do not match the D1 contract",
            )
    audit = _object(
        f"seed {seed} leakage_audit",
        result.get("leakage_audit"),
    )
    zero_pairs = {
        "train_test": 0,
        "train_validation": 0,
        "validation_test": 0,
    }
    for field in (
        "record_id_overlap_counts",
        "source_coordinate_overlap_counts",
    ):
        if audit.get(field) != zero_pairs:
            raise D1AggregateValidationError(
                f"seed {seed} leakage contract",
                f"{field} must report zero overlap for all split pairs",
            )
    if audit.get("unique_source_coordinate_counts") != {
        "train": 62700,
        "validation": 300,
        "test": 3000,
    }:
        raise D1AggregateValidationError(
            f"seed {seed} leakage contract",
            "unique source-coordinate counts do not match split counts",
        )


def _aggregate_code_document() -> Mapping[str, Mapping[str, object]]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
        for relative_path in AGGREGATE_CODE_PATHS
    }


def aggregate_d1_results(
    config_path: Path,
    result_root: Path,
) -> Mapping[str, object]:
    config = load_d1_aggregate_config(config_path)
    result_root = Path(result_root)
    results = []
    input_results = []
    prediction_artifacts = []
    for seed in config.seeds:
        path = result_root / f"{seed}.json"
        if not path.is_file():
            raise D1AggregateValidationError(
                f"seed {seed}",
                "result artifact does not exist",
            )
        result = _read_canonical_json(path, f"seed {seed}")
        if result.get("schema_version") != "phase05-d1-result-v1":
            raise D1AggregateValidationError(
                f"seed {seed} schema_version",
                "must equal phase05-d1-result-v1",
            )
        if result.get("status") != "completed":
            raise D1AggregateValidationError(
                f"seed {seed} status",
                "must equal completed",
            )
        if result.get("result_level") != "provisional":
            raise D1AggregateValidationError(
                f"seed {seed} result_level",
                "must equal provisional",
            )
        if result.get("seed") != seed:
            raise D1AggregateValidationError(
                f"seed {seed} seed",
                "does not match filename",
            )
        if result.get("experiment_id") != EXPERIMENT_ID:
            raise D1AggregateValidationError(
                f"seed {seed} experiment_id",
                f"must equal {EXPERIMENT_ID!r}",
            )
        _validate_d1_contract(result, seed)
        config_document = _object(
            f"seed {seed} config",
            result.get("config"),
        )
        if config_document.get("sha256") != RUNNER_CONFIG_SHA256:
            raise D1AggregateValidationError(
                f"seed {seed} config",
                f"sha256 must equal {RUNNER_CONFIG_SHA256}",
            )
        condition_map = _condition_map(result, seed)
        input_results.append(
            {
                "bytes": path.stat().st_size,
                "path": path.name,
                "seed": seed,
                "sha256": _sha256_file(path),
            }
        )
        prediction_artifacts.append(
            {
                "seed": seed,
                **_validate_prediction_artifact(
                    result_root,
                    result,
                    seed,
                ),
            }
        )
        results.append((result, condition_map))

    first_result = results[0][0]
    first_dataset = first_result.get("dataset")
    first_pipeline = first_result.get("pipeline_configs")
    first_code = first_result.get("code")
    first_environment = first_result.get("environment")
    first_test = _object(
        "seed 0 split.test",
        _object("seed 0 split", first_result.get("split")).get("test"),
    )
    test_digest = first_test.get("record_ids_sha256")
    dataset_identity = _canonical_json_bytes(first_dataset)
    pipeline_identity = _canonical_json_bytes(first_pipeline)
    code_identity = _canonical_json_bytes(first_code)
    environment_identity = _canonical_json_bytes(first_environment)
    control_values = []
    sg_values = []
    per_seed = []
    for seed, (result, conditions) in zip(
        config.seeds,
        results,
        strict=True,
    ):
        if _canonical_json_bytes(result.get("dataset")) != dataset_identity:
            raise D1AggregateValidationError(
                f"seed {seed} dataset",
                "identity differs from seed 0",
            )
        if _canonical_json_bytes(
            result.get("pipeline_configs")
        ) != pipeline_identity:
            raise D1AggregateValidationError(
                f"seed {seed} pipeline_configs",
                "identity differs from seed 0",
            )
        if _canonical_json_bytes(result.get("code")) != code_identity:
            raise D1AggregateValidationError(
                f"seed {seed} code identity",
                "differs from seed 0",
            )
        if _canonical_json_bytes(
            result.get("environment")
        ) != environment_identity:
            raise D1AggregateValidationError(
                f"seed {seed} environment identity",
                "differs from seed 0",
            )
        test = _object(
            f"seed {seed} split.test",
            _object(f"seed {seed} split", result.get("split")).get("test"),
        )
        if test.get("record_ids_sha256") != test_digest:
            raise D1AggregateValidationError(
                f"seed {seed} test record identity",
                "differs from seed 0",
            )
        control_ids, control_true, control_value = _validated_predictions(
            conditions[CONTROL],
            seed,
            CONTROL,
        )
        sg_ids, sg_true, sg_value = _validated_predictions(
            conditions[SG],
            seed,
            SG,
        )
        if control_ids != sg_ids or control_true != sg_true:
            raise D1AggregateValidationError(
                f"seed {seed} paired prediction identity",
                "condition record IDs or true labels differ",
            )
        if hashlib.sha256(
            ("\n".join(control_ids) + "\n").encode("utf-8")
        ).hexdigest() != test_digest:
            raise D1AggregateValidationError(
                f"seed {seed} test record identity",
                "embedded predictions do not match split digest",
            )
        control_values.append(control_value)
        sg_values.append(sg_value)
        per_seed.append(
            {
                "control_accuracy": control_value,
                "difference_percentage_points": (
                    100.0 * (sg_value - control_value)
                ),
                "seed": seed,
                "sg_accuracy": sg_value,
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
    if permutation["patterns"] != config.sign_flip_patterns:
        raise RuntimeError("sign-flip pattern count does not match config")
    return {
        "aggregate_code": _aggregate_code_document(),
        "aggregate_config": {
            "bytes": config.byte_count,
            "path": config.path.name,
            "sha256": config.sha256,
        },
        "conditions": list(EXPECTED_CONDITIONS),
        "effect_exceeds_threshold": (
            bootstrap["difference_mean"]
            > config.effect_threshold_percentage_points
        ),
        "effect_threshold_percentage_points": (
            config.effect_threshold_percentage_points
        ),
        "experiment_id": EXPERIMENT_ID,
        "gate_role": "control_not_counted",
        "input_results": input_results,
        "per_seed": per_seed,
        "prediction_artifacts": prediction_artifacts,
        "primary_metric": "accuracy",
        "primary_metric_summary": {
            "bootstrap": bootstrap,
            "control_mean_percentage": bootstrap["left_mean"],
            "difference_mean_percentage_points": (
                bootstrap["difference_mean"]
            ),
            "permutation": permutation,
            "sg_mean_percentage": bootstrap["right_mean"],
        },
        "result_level": "complete_cell",
        "runner_config_sha256": RUNNER_CONFIG_SHA256,
        "schema_version": RESULT_SCHEMA_VERSION,
        "seeds": list(config.seeds),
        "status": "completed",
        "test_record_ids_sha256": test_digest,
    }


__all__ = [
    "D1AggregateConfig",
    "D1AggregateValidationError",
    "aggregate_d1_results",
    "load_d1_aggregate_config",
]
