from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np

from rpe.downstream.sugar_quantitative import (
    CONFIG_SHA256,
    FOLD_RECORD_IDS_SHA256,
    FOLD_SOURCE_MEMBERS_SHA256,
    MIXTURE_RECORD_IDS_SHA256,
    TARGET_NAMES,
)
from rpe.runner.d4_sugar import (
    CONDITION_IDS,
    CONTROL_CONDITION_ID,
    EXPERIMENT_ID,
    FOLD_WELL_IDS_SHA256,
    SG_CONDITION_ID,
    paired_predictions_jsonl_bytes,
    validate_d4_result,
)


SCHEMA_VERSION = "phase05-d4-complete-cell-v1"
RESULT_SCHEMA_VERSION = "phase05-d4-result-v1"
SEEDS = (0, 1, 2, 3, 4)
WELL_COUNT = 240
RECORD_COUNT = 7680
TARGET_COUNT = 4
TARGET_RANGE = 0.32
BOOTSTRAP_RESAMPLES = 10_000
PERMUTATION_RESAMPLES = 100_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 20260817
EFFECT_THRESHOLD_PERCENTAGE_POINTS = 2.0
SIGNIFICANCE_THRESHOLD = 0.05
ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_CODE_PATHS = (
    "rpe/runner/d4_aggregate.py",
    "tools/aggregate_phase05_d4.py",
)
RUNNER_CODE_PATHS = (
    "rpe/downstream/sugar_quantitative.py",
    "rpe/runner/d4_sugar.py",
    "tools/run_phase05_d4.py",
)


class D4AggregateValidationError(ValueError):
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
    raise D4AggregateValidationError(
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
        raise D4AggregateValidationError(path, "must be an object")
    return value


def _read_canonical_json(path: Path, label: str) -> Mapping[str, object]:
    if not path.is_file():
        raise D4AggregateValidationError(label, "file does not exist")
    raw = path.read_bytes()
    try:
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    except json.JSONDecodeError as error:
        raise D4AggregateValidationError(label, str(error)) from error
    document = _object(label, value)
    if raw != _canonical_json_bytes(document):
        raise D4AggregateValidationError(label, "must use canonical JSON")
    return document


def _code_document(paths: tuple[str, ...]) -> dict[str, object]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
        for relative_path in paths
    }


def _primary_from_sse(
    sse: np.ndarray,
    counts: np.ndarray | int,
) -> np.ndarray:
    return np.mean(
        np.sqrt(sse / np.asarray(counts)[..., None]) / TARGET_RANGE,
        axis=-1,
    )


def paired_well_primary_statistics(
    control_sse: np.ndarray,
    sg_sse: np.ndarray,
    counts: np.ndarray,
    *,
    bootstrap_resamples: int,
    permutation_resamples: int,
    confidence_level: float,
    random_seed: int,
) -> dict[str, object]:
    control_sse = np.asarray(control_sse, dtype="<f8")
    sg_sse = np.asarray(sg_sse, dtype="<f8")
    counts = np.asarray(counts)
    if (
        control_sse.ndim != 2
        or control_sse.shape[1] != TARGET_COUNT
        or sg_sse.shape != control_sse.shape
        or counts.shape != (control_sse.shape[0],)
        or control_sse.shape[0] < 2
        or not np.isfinite(control_sse).all()
        or not np.isfinite(sg_sse).all()
        or np.any(control_sse < 0.0)
        or np.any(sg_sse < 0.0)
        or not np.issubdtype(counts.dtype, np.integer)
        or np.any(counts <= 0)
    ):
        raise D4AggregateValidationError(
            "well statistics arrays",
            "must be aligned finite nonnegative SSE matrices and positive counts",
        )
    if (
        isinstance(bootstrap_resamples, bool)
        or not isinstance(bootstrap_resamples, int)
        or bootstrap_resamples <= 0
        or isinstance(permutation_resamples, bool)
        or not isinstance(permutation_resamples, int)
        or permutation_resamples <= 0
        or isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
        or isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise D4AggregateValidationError(
            "well statistics config",
            "resamples, confidence, and random seed are invalid",
        )
    well_count = control_sse.shape[0]
    total_count = int(counts.sum())
    control = float(_primary_from_sse(control_sse.sum(axis=0), total_count))
    sg = float(_primary_from_sse(sg_sse.sum(axis=0), total_count))
    effect = 100.0 * (control - sg)

    generator = np.random.default_rng(random_seed)
    bootstrap_control = np.empty(bootstrap_resamples, dtype="<f8")
    bootstrap_sg = np.empty(bootstrap_resamples, dtype="<f8")
    batch_size = 256
    completed = 0
    while completed < bootstrap_resamples:
        current = min(batch_size, bootstrap_resamples - completed)
        indices = generator.integers(
            0,
            well_count,
            size=(current, well_count),
        )
        sampled_counts = counts[indices].sum(axis=1)
        control_values = _primary_from_sse(
            control_sse[indices].sum(axis=1),
            sampled_counts,
        )
        sg_values = _primary_from_sse(
            sg_sse[indices].sum(axis=1),
            sampled_counts,
        )
        bootstrap_control[completed : completed + current] = control_values
        bootstrap_sg[completed : completed + current] = sg_values
        completed += current
    bootstrap_effect = 100.0 * (bootstrap_control - bootstrap_sg)
    alpha = (1.0 - float(confidence_level)) / 2.0

    def interval(values: np.ndarray) -> list[float]:
        lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
        return [float(lower), float(upper)]

    midpoint = (control_sse + sg_sse) / 2.0
    half_difference = (control_sse - sg_sse) / 2.0
    threshold = abs(effect)
    extreme = 0
    completed = 0
    permutation_generator = np.random.default_rng(random_seed)
    while completed < permutation_resamples:
        current = min(256, permutation_resamples - completed)
        bits = permutation_generator.integers(
            0,
            2,
            size=(current, well_count),
            dtype=np.int8,
        )
        signs = bits * np.int8(2) - np.int8(1)
        signed = signs.astype(np.float64, copy=False) @ half_difference
        center = midpoint.sum(axis=0)[None, :]
        permuted_control = _primary_from_sse(center + signed, total_count)
        permuted_sg = _primary_from_sse(center - signed, total_count)
        permuted_effect = 100.0 * (permuted_control - permuted_sg)
        extreme += int(
            np.sum(np.abs(permuted_effect) + 1e-15 >= threshold)
        )
        completed += current
    return {
        "bootstrap": {
            "confidence_level": float(confidence_level),
            "control_ci": interval(bootstrap_control),
            "effect_ci_percentage_points": interval(bootstrap_effect),
            "interval": "percentile",
            "paired": True,
            "random_seed": random_seed,
            "resamples": bootstrap_resamples,
            "sample_size": well_count,
            "sg_ci": interval(bootstrap_sg),
            "unit": "physical_well",
        },
        "control_macro_normalized_rmse": control,
        "effect_percentage_points": effect,
        "permutation": {
            "alternative": "two-sided",
            "extreme_resamples": extreme,
            "method": "monte_carlo_paired_sign_flip",
            "observed_effect_percentage_points": effect,
            "p_value": (extreme + 1) / (permutation_resamples + 1),
            "p_value_correction": "plus_one",
            "random_seed": random_seed,
            "resamples": permutation_resamples,
            "sample_size": well_count,
            "unit": "physical_well",
            "zero_difference": "included",
        },
        "sg_macro_normalized_rmse": sg,
        "statistics_unit": "physical_well",
    }


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
    payload_path = root / f"paired_predictions_seed{seed}_complete.jsonl"
    result = _read_canonical_json(result_path, f"seed {seed} result")
    try:
        validate_d4_result(result)
    except ValueError as error:
        raise D4AggregateValidationError(
            f"seed {seed} result",
            str(error),
        ) from error
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
        raise D4AggregateValidationError(
            f"seed {seed} identity",
            "result identity mismatch",
        )
    if result.get("code") != _code_document(RUNNER_CODE_PATHS):
        raise D4AggregateValidationError(
            f"seed {seed} current code",
            "does not equal current runner sources",
        )
    split = _object(f"seed {seed} split", result.get("split"))
    if (
        split.get("test_record_ids_sha256") != FOLD_RECORD_IDS_SHA256[seed]
        or split.get("test_source_members_sha256")
        != FOLD_SOURCE_MEMBERS_SHA256[seed]
        or split.get("test_well_ids_sha256") != FOLD_WELL_IDS_SHA256[seed]
    ):
        raise D4AggregateValidationError(
            f"seed {seed} frozen split",
            "test fold identity mismatch",
        )
    if not payload_path.is_file():
        raise D4AggregateValidationError(
            f"seed {seed} paired predictions",
            "file does not exist",
        )
    payload = payload_path.read_bytes()
    expected_payload = paired_predictions_jsonl_bytes(result)
    if payload != expected_payload:
        raise D4AggregateValidationError(
            f"seed {seed} paired predictions",
            "does not equal embedded canonical projection",
        )
    artifact = result.get("paired_predictions_artifact")
    expected_artifact = {
        "bytes": len(payload),
        "lines": payload.count(b"\n"),
        "path": payload_path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if artifact != expected_artifact:
        raise D4AggregateValidationError(
            f"seed {seed} paired predictions",
            "artifact identity mismatch",
        )
    rows = result.get("paired_predictions")
    if not isinstance(rows, list) or len(rows) != 1536:
        raise D4AggregateValidationError(
            f"seed {seed} paired predictions",
            "must contain 1536 rows",
        )
    return (
        result,
        rows,
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


def _pooled_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, object]:
    errors = predicted - true
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    denominator = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    if np.any(denominator <= 0.0):
        raise D4AggregateValidationError(
            "pooled metrics",
            "each target must vary",
        )
    r2 = 1.0 - np.sum(errors**2, axis=0) / denominator
    return {
        "macro_mae_mol_l": float(mae.mean()),
        "macro_normalized_rmse": float(np.mean(rmse / TARGET_RANGE)),
        "macro_r2": float(r2.mean()),
        "per_analyte": {
            name: {
                "mae_mol_l": float(mae[index]),
                "r2": float(r2[index]),
                "rmse_mol_l": float(rmse[index]),
            }
            for index, name in enumerate(TARGET_NAMES)
        },
        "sample_count": int(true.shape[0]),
    }


def aggregate_d4_results(result_root: Path) -> dict[str, object]:
    result_root = Path(result_root)
    input_results = []
    payload_artifacts = []
    seed_results = []
    per_seed = []
    all_rows: list[Mapping[str, object]] = []
    baseline_cohort = None
    baseline_code = None
    for seed in SEEDS:
        result, rows, result_artifact, payload_artifact = (
            _validate_seed_result(result_root, seed)
        )
        cohort = result["cohort"]
        code = result["code"]
        if baseline_cohort is None:
            baseline_cohort = cohort
            baseline_code = code
        elif cohort != baseline_cohort:
            raise D4AggregateValidationError(
                f"seed {seed} cohort",
                "differs across seeds",
            )
        elif code != baseline_code:
            raise D4AggregateValidationError(
                f"seed {seed} code",
                "differs across seeds",
            )
        condition_map = {
            condition["condition_id"]: condition
            for condition in result["conditions"]
        }
        per_seed.append(
            {
                "control_macro_normalized_rmse": condition_map[
                    CONTROL_CONDITION_ID
                ]["test_metrics"]["macro_normalized_rmse"],
                "effect_percentage_points": result[
                    "descriptive_effect_percentage_points"
                ],
                "seed": seed,
                "selected_n_components": {
                    condition_id: condition_map[condition_id][
                        "selected_n_components"
                    ]
                    for condition_id in CONDITION_IDS
                },
                "sg_macro_normalized_rmse": condition_map[
                    SG_CONDITION_ID
                ]["test_metrics"]["macro_normalized_rmse"],
            }
        )
        all_rows.extend(rows)
        seed_results.append(result)
        input_results.append(result_artifact)
        payload_artifacts.append(payload_artifact)
    record_ids = [str(row["record_id"]) for row in all_rows]
    if (
        len(all_rows) != RECORD_COUNT
        or len(set(record_ids)) != RECORD_COUNT
        or hashlib.sha256(
            ("\n".join(sorted(record_ids)) + "\n").encode("utf-8")
        ).hexdigest()
        != MIXTURE_RECORD_IDS_SHA256
    ):
        raise D4AggregateValidationError(
            "exhaustive records",
            "five test folds must contain every retained record exactly once",
        )
    by_well: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in all_rows:
        by_well[str(row["well_id"])].append(row)
    if (
        len(by_well) != WELL_COUNT
        or any(len(rows) != 32 for rows in by_well.values())
    ):
        raise D4AggregateValidationError(
            "exhaustive wells",
            "must contain 240 physical wells with 32 records each",
        )
    true = np.asarray([row["true_targets"] for row in all_rows], dtype="<f8")
    predictions = {
        condition_id: np.asarray(
            [
                row["predictions"][condition_id]
                for row in all_rows
            ],
            dtype="<f8",
        )
        for condition_id in CONDITION_IDS
    }
    pooled_metrics = {
        condition_id: _pooled_metrics(true, predictions[condition_id])
        for condition_id in CONDITION_IDS
    }
    well_rows = []
    control_sse = []
    sg_sse = []
    counts = []
    for well_id in sorted(by_well):
        rows = by_well[well_id]
        well_true = np.asarray(
            [row["true_targets"] for row in rows],
            dtype="<f8",
        )
        current_sse = {}
        for condition_id in CONDITION_IDS:
            current_prediction = np.asarray(
                [row["predictions"][condition_id] for row in rows],
                dtype="<f8",
            )
            current_sse[condition_id] = np.sum(
                (current_prediction - well_true) ** 2,
                axis=0,
            )
        control_sse.append(current_sse[CONTROL_CONDITION_ID])
        sg_sse.append(current_sse[SG_CONDITION_ID])
        counts.append(len(rows))
        well_rows.append(
            {
                "control_sse": [
                    float(value)
                    for value in current_sse[CONTROL_CONDITION_ID]
                ],
                "record_count": len(rows),
                "sg_sse": [
                    float(value)
                    for value in current_sse[SG_CONDITION_ID]
                ],
                "true_targets": [
                    float(value) for value in well_true[0]
                ],
                "well_id": well_id,
            }
        )
    statistics = paired_well_primary_statistics(
        np.asarray(control_sse, dtype="<f8"),
        np.asarray(sg_sse, dtype="<f8"),
        np.asarray(counts, dtype="<i8"),
        bootstrap_resamples=BOOTSTRAP_RESAMPLES,
        permutation_resamples=PERMUTATION_RESAMPLES,
        confidence_level=CONFIDENCE_LEVEL,
        random_seed=RANDOM_SEED,
    )
    pooled_control = pooled_metrics[CONTROL_CONDITION_ID][
        "macro_normalized_rmse"
    ]
    pooled_sg = pooled_metrics[SG_CONDITION_ID][
        "macro_normalized_rmse"
    ]
    if (
        not np.isclose(
            statistics["control_macro_normalized_rmse"],
            pooled_control,
            rtol=0.0,
            atol=1e-15,
        )
        or not np.isclose(
            statistics["sg_macro_normalized_rmse"],
            pooled_sg,
            rtol=0.0,
            atol=1e-15,
        )
    ):
        raise D4AggregateValidationError(
            "primary statistics",
            "well SSE reconstruction differs from pooled predictions",
        )
    effect = statistics["effect_percentage_points"]
    significant = (
        statistics["permutation"]["p_value"] < SIGNIFICANCE_THRESHOLD
    )
    exceeds = effect > EFFECT_THRESHOLD_PERCENTAGE_POINTS
    lod_loq_by_seed = {
        condition_id: {
            target_name: [
                seed_result["conditions"][
                    list(CONDITION_IDS).index(condition_id)
                ]["lod_loq"][target_name]
                for seed_result in seed_results
            ]
            for target_name in TARGET_NAMES
        }
        for condition_id in CONDITION_IDS
    }
    return {
        "aggregate_code": _code_document(AGGREGATE_CODE_PATHS),
        "conditions": list(CONDITION_IDS),
        "d4_gate_success": exceeds and significant,
        "effect_exceeds_threshold": exceeds,
        "effect_threshold_percentage_points": (
            EFFECT_THRESHOLD_PERCENTAGE_POINTS
        ),
        "experiment_id": EXPERIMENT_ID,
        "input_results": input_results,
        "lod_loq_by_seed": lod_loq_by_seed,
        "paired_prediction_artifacts": payload_artifacts,
        "per_seed": per_seed,
        "per_well": well_rows,
        "pooled_metrics": pooled_metrics,
        "primary_summary": statistics,
        "protocol_config_sha256": CONFIG_SHA256,
        "result_level": "complete_cell",
        "schema_version": SCHEMA_VERSION,
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
            "estimand": (
                "pooled_out_of_fold_macro_normalized_rmse_recomputed_"
                "from_resampled_well_sse"
            ),
            "permutation": {
                "alternative": "two-sided",
                "method": "monte_carlo_paired_sign_flip",
                "p_value_correction": "plus_one",
                "random_seed": RANDOM_SEED,
                "resamples": PERMUTATION_RESAMPLES,
            },
            "unit": "physical_well",
        },
        "statistics_unit": "physical_well",
        "status": "completed",
    }


__all__ = [
    "D4AggregateValidationError",
    "aggregate_d4_results",
    "paired_well_primary_statistics",
]
