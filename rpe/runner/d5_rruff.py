from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Mapping

import h5py
import numpy as np
import scipy

from rpe.downstream.rruff import (
    CLASS_LABELS_SHA256,
    CONFIG_BYTES,
    CONFIG_SHA256,
    DATASET_ID,
    EXPECTED_CLASS_COUNT,
    EXPECTED_GROUP_COUNT,
    EXPECTED_RECORD_COUNT,
    EXPECTED_RRUFF_ID_COUNT,
    GROUP_IDS_SHA256,
    RECORD_IDS_SHA256,
    D5RawCohort,
    load_d5_protocol_config,
    load_d5_raw_cohort,
)
from rpe.downstream.rruff_matching import (
    CONDITION_IDS,
    CONTROL_CONDITION_ID,
    SG_CONDITION_ID,
    D5MatchingResult,
    match_d5_condition,
    prepare_d5_conditions,
)


SCHEMA_VERSION = "phase05-d5-result-v1"
EXPERIMENT_ID = "d5_rruff_raw_library_matching_sg11"
RESULT_LEVELS = ("smoke", "provisional", "complete_seed")
SEEDS_REQUIRED = 5
SG_CONFIG_BYTES = 338
SG_CONFIG_SHA256 = (
    "775c3dba4c4d6bb13ab0ac44abf3e48011f90ce8bc268f7f79632ed35cb039aa"
)
RETAINED_MATRIX_SHA256 = (
    "5a9776e787609b84e0ffb2f82685a77f507748243e74769695300b221d4b52ee"
)
FROZEN_SPLITS = (
    {
        "library_count": 2452,
        "library_group_count": 1253,
        "library_group_ids_sha256": (
            "c0288e4428a5f19e1e468a6e295305fbcb40de2030628d0aa76e8c2cf96572e8"
        ),
        "library_record_ids_sha256": (
            "3586c18224d625cfccf26b1d56e1e90b879de376d4dbd7455a161805f5bff87c"
        ),
        "query_count": 1318,
        "query_group_count": 681,
        "query_group_ids_sha256": (
            "04dd41d8b464f91436d57afd54eb88a04e2572154cd58e19e8613d3d6d004147"
        ),
        "query_record_ids_sha256": (
            "59fb906a47ceb996ef9ba29995172c25931a01840bfb9b4a314ed118ee6c556b"
        ),
        "seed": 0,
        "split_sha256": (
            "5d292560b28bf41c207c6fbe88d9e5025c4f03764240a41d6abf0da14138e5e5"
        ),
    },
    {
        "library_count": 2439,
        "library_group_count": 1253,
        "library_group_ids_sha256": (
            "bb1adcc843533c606059c2f1d6ac9f79dbdab56487c596a9783cf485b3a5fb78"
        ),
        "library_record_ids_sha256": (
            "23fbd1f72559000e6cb339266fc021793ba813658f48f9c0bb5821f686b6aacb"
        ),
        "query_count": 1331,
        "query_group_count": 681,
        "query_group_ids_sha256": (
            "f11aefa4c58ea0d1fdedeabd2cbdceae8df20cc96d75bab104d47696e52221bd"
        ),
        "query_record_ids_sha256": (
            "ebb9e7669c7bd23430cf1d4ea1c75cd4d478736f19930656eeed6bcb3b4f431f"
        ),
        "seed": 1,
        "split_sha256": (
            "ee6703717075c7ab4094fcd68886866edfa1237e204f386a5622cb73538dd7c1"
        ),
    },
    {
        "library_count": 2443,
        "library_group_count": 1253,
        "library_group_ids_sha256": (
            "af0ee4b329fcb1e2451ee4837cdca12651e643d74c521b93b3e22eefed579561"
        ),
        "library_record_ids_sha256": (
            "235ad6221bccc2c449e34837303356913e063ba945b76a14a2ad987b184195b6"
        ),
        "query_count": 1327,
        "query_group_count": 681,
        "query_group_ids_sha256": (
            "baae60f424d6a5943b249a938b7001fc5d7ae0a144ef051f2fb76d173c93b704"
        ),
        "query_record_ids_sha256": (
            "5a2c8500bf03dbb3a29d481c826aafc3561c3af7c7d05bc7704dec6dd06a6ef0"
        ),
        "seed": 2,
        "split_sha256": (
            "91c2bc9c8fd669bb39f7ee29b79eecd2baa3573e37b9941e89b2beebb6593335"
        ),
    },
    {
        "library_count": 2449,
        "library_group_count": 1253,
        "library_group_ids_sha256": (
            "262ad3f317a9fbae46dbe78a03329f1c2fc309879abcb3e0c546b5341d5f7372"
        ),
        "library_record_ids_sha256": (
            "3dfb99347da04bf657f618e88ccfb35202f1649a82da310cd49554dcfd279def"
        ),
        "query_count": 1321,
        "query_group_count": 681,
        "query_group_ids_sha256": (
            "6e90304e499af9f2d8f817ed6fc1d997949901b7455e402eee87986383a028ca"
        ),
        "query_record_ids_sha256": (
            "5f7119f8e459314bdac220841db3d1ca70c7cc6e9c5c0c0a1b2563ef773b4e98"
        ),
        "seed": 3,
        "split_sha256": (
            "01d736bcd75a872fbffbd6fac7cd2cd406bc90ad532227a24ed41fdb19dbb1cb"
        ),
    },
    {
        "library_count": 2446,
        "library_group_count": 1253,
        "library_group_ids_sha256": (
            "ef69da621edc45f200192ce7a57173e3e18f2a3b30c2d8e4a8ef8e8f1f331949"
        ),
        "library_record_ids_sha256": (
            "11066cea169bd4fd06330ab0d3f2181cfd26e3965bd0e656ea33c10a5f26bd17"
        ),
        "query_count": 1324,
        "query_group_count": 681,
        "query_group_ids_sha256": (
            "b165672f139c2ed1a906dca6c19f4b6bc9bdc393740978895279f620c649029f"
        ),
        "query_record_ids_sha256": (
            "812ea09a2205a067924fb87b6b7ceb2d2b2ce183897bc82cfc67f25522e11085"
        ),
        "seed": 4,
        "split_sha256": (
            "f3d08a18e88e159b1097a0c2e823bb83d1e2aa0405f4bac3390de57618d97788"
        ),
    },
)
ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "rpe/downstream/rruff.py",
    "rpe/downstream/rruff_matching.py",
    "rpe/methods/classical/savitzky_golay.py",
    "rpe/runner/d5_rruff.py",
    "tools/run_phase05_d5.py",
)
FORBIDDEN_INFERENCE_KEYS = {
    "bootstrap",
    "confidence_interval",
    "gate_success",
    "p_value",
    "permutation",
    "significant",
}


class D5RunnerValidationError(ValueError):
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(values: tuple[str, ...] | list[str] | set[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("utf-8")
    ).hexdigest()


def _class_labels_digest(values: set[int]) -> str:
    return hashlib.sha256(
        ("\n".join(str(value) for value in sorted(values)) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D5RunnerValidationError(path, "must be an object")
    return value


def _code_document() -> dict[str, object]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
        for relative_path in CODE_PATHS
    }


def _environment_document() -> dict[str, object]:
    return {
        "compute_backend": "cpu",
        "h5py": h5py.__version__,
        "machine": platform.machine(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "scipy": scipy.__version__,
    }


def _validate_level_and_seed(
    *,
    seed: int,
    result_level: str,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise D5RunnerValidationError("seed", "must be an integer")
    if result_level not in RESULT_LEVELS:
        raise D5RunnerValidationError(
            "result_level",
            f"must be one of {RESULT_LEVELS!r}",
        )
    if result_level == "provisional" and seed != 0:
        raise D5RunnerValidationError(
            "provisional seed",
            "must equal 0 for PROVISIONAL — 1/5 SEEDS",
        )
    if result_level == "complete_seed" and seed not in range(SEEDS_REQUIRED):
        raise D5RunnerValidationError(
            "complete seed",
            "must be an integer in [0, 4]",
        )


def _find_split(cohort: D5RawCohort, seed: int):
    matches = [split for split in cohort.splits if split.seed == seed]
    if len(matches) != 1:
        raise D5RunnerValidationError(
            "seed",
            f"must identify exactly one cohort split; observed {len(matches)}",
        )
    return matches[0]


def _validate_provisional_cohort(cohort: D5RawCohort) -> None:
    class_labels = set(int(value) for value in cohort.class_labels)
    observed = (
        len(cohort.record_ids),
        len(class_labels),
        len(set(cohort.rruff_ids)),
        len(set(cohort.group_ids)),
    )
    expected = (
        EXPECTED_RECORD_COUNT,
        EXPECTED_CLASS_COUNT,
        EXPECTED_RRUFF_ID_COUNT,
        EXPECTED_GROUP_COUNT,
    )
    if observed != expected:
        raise D5RunnerValidationError(
            "provisional cohort counts",
            f"expected {expected!r}; observed {observed!r}",
        )
    identities = (
        _ids_digest(cohort.record_ids),
        _class_labels_digest(class_labels),
        _ids_digest(set(cohort.group_ids)),
    )
    expected_identities = (
        RECORD_IDS_SHA256,
        CLASS_LABELS_SHA256,
        GROUP_IDS_SHA256,
    )
    if identities != expected_identities:
        raise D5RunnerValidationError(
            "provisional cohort identities",
            "record, class, or group digest mismatch",
        )


def _class_macro_accuracy(
    true_class_labels: np.ndarray,
    correct: np.ndarray,
) -> float:
    class_values = []
    for class_label in np.unique(true_class_labels):
        class_values.append(
            float(np.mean(correct[true_class_labels == class_label]))
        )
    return float(np.mean(np.asarray(class_values, dtype=np.float64)))


def _metrics(result: D5MatchingResult) -> dict[str, object]:
    top1 = np.asarray(result.top1_correct, dtype=np.float64)
    top5 = np.asarray(result.top5_correct, dtype=np.float64)
    return {
        "query_count": int(top1.size),
        "top1_correct_count": int(top1.sum()),
        "top1_macro_class_accuracy": _class_macro_accuracy(
            result.true_class_labels,
            top1,
        ),
        "top1_micro_accuracy": float(top1.mean()),
        "top5_correct_count": int(top5.sum()),
        "top5_macro_class_accuracy": _class_macro_accuracy(
            result.true_class_labels,
            top5,
        ),
        "top5_micro_accuracy": float(top5.mean()),
    }


def _condition_document(
    matching: D5MatchingResult,
    condition_values: np.ndarray,
) -> dict[str, object]:
    return {
        "condition_id": matching.condition_id,
        "condition_matrix_sha256": _array_sha256(condition_values),
        "metrics": _metrics(matching),
        "ranked_class_labels_sha256": _array_sha256(
            matching.ranked_class_labels
        ),
        "ranked_class_scores_sha256": _array_sha256(
            matching.ranked_class_scores
        ),
    }


def _paired_outcomes(
    cohort: D5RawCohort,
    control: D5MatchingResult,
    sg: D5MatchingResult,
) -> list[dict[str, object]]:
    if (
        control.query_record_ids != sg.query_record_ids
        or not np.array_equal(
            control.query_indices,
            sg.query_indices,
        )
        or not np.array_equal(
            control.true_class_labels,
            sg.true_class_labels,
        )
    ):
        raise D5RunnerValidationError(
            "paired outcomes",
            "condition query identities or true labels differ",
        )
    outcomes = []
    for row, query_index in enumerate(control.query_indices):
        index = int(query_index)
        conditions = {}
        for matching in (control, sg):
            top_k = min(5, matching.ranked_class_labels.shape[1])
            conditions[matching.condition_id] = {
                "top1_class_label": int(
                    matching.ranked_class_labels[row, 0]
                ),
                "top1_correct": bool(matching.top1_correct[row]),
                "top1_score": float(
                    matching.ranked_class_scores[row, 0]
                ),
                "top5_class_labels": [
                    int(value)
                    for value in matching.ranked_class_labels[row, :top_k]
                ],
                "top5_correct": bool(matching.top5_correct[row]),
                "top5_scores": [
                    float(value)
                    for value in matching.ranked_class_scores[row, :top_k]
                ],
            }
        outcomes.append(
            {
                "conditions": conditions,
                "group_id": cohort.group_ids[index],
                "mineral_name": cohort.mineral_names[index],
                "query_index": index,
                "record_id": cohort.record_ids[index],
                "rruff_id": cohort.rruff_ids[index],
                "true_class_label": int(cohort.class_labels[index]),
            }
        )
    return outcomes


def _outcomes_jsonl_bytes(
    outcomes: list[Mapping[str, object]],
) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in outcomes)


def paired_outcomes_jsonl_bytes(
    result: Mapping[str, object],
) -> bytes:
    outcomes = result.get("paired_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise D5RunnerValidationError(
            "paired outcomes",
            "must be a nonempty list",
        )
    if any(not isinstance(row, Mapping) for row in outcomes):
        raise D5RunnerValidationError(
            "paired outcomes",
            "every row must be an object",
        )
    return _outcomes_jsonl_bytes(outcomes)


def _paired_summary(
    outcomes: list[Mapping[str, object]],
) -> dict[str, object]:
    summary = {}
    for metric in ("top1_correct", "top5_correct"):
        both = control_only = sg_only = neither = 0
        for row in outcomes:
            conditions = row["conditions"]
            left = bool(conditions[CONTROL_CONDITION_ID][metric])
            right = bool(conditions[SG_CONDITION_ID][metric])
            if left and right:
                both += 1
            elif left:
                control_only += 1
            elif right:
                sg_only += 1
            else:
                neither += 1
        summary[metric] = {
            "both_correct": both,
            "control_only_correct": control_only,
            "neither_correct": neither,
            "sg_only_correct": sg_only,
        }
    return summary


def _descriptive_deltas(
    conditions: list[Mapping[str, object]],
) -> dict[str, float]:
    by_condition = {
        str(condition["condition_id"]): condition["metrics"]
        for condition in conditions
    }
    return {
        metric: 100.0
        * (
            float(by_condition[SG_CONDITION_ID][metric])
            - float(by_condition[CONTROL_CONDITION_ID][metric])
        )
        for metric in (
            "top1_macro_class_accuracy",
            "top5_macro_class_accuracy",
            "top1_micro_accuracy",
            "top5_micro_accuracy",
        )
    }


def _dataset_document(cohort: D5RawCohort) -> dict[str, object]:
    return {
        "class_count": len(set(int(value) for value in cohort.class_labels)),
        "class_labels_sha256": _class_labels_digest(
            set(int(value) for value in cohort.class_labels)
        ),
        "dataset_id": cohort.dataset_id,
        "group_count": len(set(cohort.group_ids)),
        "group_ids_sha256": _ids_digest(set(cohort.group_ids)),
        "matrix_sha256": _array_sha256(cohort.intensity),
        "record_count": len(cohort.record_ids),
        "record_ids_sha256": _ids_digest(cohort.record_ids),
        "rruff_id_count": len(set(cohort.rruff_ids)),
    }


def _expected_provisional_dataset() -> dict[str, object]:
    return {
        "class_count": EXPECTED_CLASS_COUNT,
        "class_labels_sha256": CLASS_LABELS_SHA256,
        "dataset_id": DATASET_ID,
        "group_count": EXPECTED_GROUP_COUNT,
        "group_ids_sha256": GROUP_IDS_SHA256,
        "matrix_sha256": RETAINED_MATRIX_SHA256,
        "record_count": EXPECTED_RECORD_COUNT,
        "record_ids_sha256": RECORD_IDS_SHA256,
        "rruff_id_count": EXPECTED_RRUFF_ID_COUNT,
    }


def _split_document(
    cohort: D5RawCohort,
    matching: D5MatchingResult,
    outcomes: list[Mapping[str, object]],
) -> dict[str, object]:
    split = _find_split(cohort, matching.seed)
    query_indices = [int(value) for value in split.query_indices]
    library_indices = [int(value) for value in split.library_indices]
    query_groups = {cohort.group_ids[index] for index in query_indices}
    library_groups = {cohort.group_ids[index] for index in library_indices}
    return {
        "class_count": len(set(int(value) for value in cohort.class_labels)),
        "library_count": len(library_indices),
        "library_group_count": len(library_groups),
        "library_group_ids_sha256": _ids_digest(library_groups),
        "library_record_ids_sha256": _ids_digest(
            [cohort.record_ids[index] for index in library_indices]
        ),
        "query_count": len(query_indices),
        "query_group_count": len(query_groups),
        "query_group_ids_sha256": _ids_digest(query_groups),
        "query_metadata_sha256": _outcome_metadata_sha256(outcomes),
        "query_record_ids_sha256": _ids_digest(
            [cohort.record_ids[index] for index in query_indices]
        ),
        "seed": matching.seed,
        "split_sha256": matching.split_sha256,
    }


def _outcome_metadata_sha256(
    outcomes: list[Mapping[str, object]],
) -> str:
    projection = [
        {
            "group_id": row["group_id"],
            "mineral_name": row["mineral_name"],
            "query_index": row["query_index"],
            "record_id": row["record_id"],
            "rruff_id": row["rruff_id"],
            "true_class_label": row["true_class_label"],
        }
        for row in outcomes
    ]
    return hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()


def _recursive_keys(value: object) -> set[str]:
    keys = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys


def run_d5_provisional_from_cohort(
    cohort: D5RawCohort,
    sg_config_path: Path,
    *,
    seed: int,
    result_level: str,
) -> dict[str, object]:
    _validate_level_and_seed(seed=seed, result_level=result_level)
    if result_level in {"provisional", "complete_seed"}:
        _validate_provisional_cohort(cohort)
    split = _find_split(cohort, seed)
    sg_config_path = Path(sg_config_path)
    if (
        not sg_config_path.is_file()
        or sg_config_path.stat().st_size != SG_CONFIG_BYTES
        or _sha256_file(sg_config_path) != SG_CONFIG_SHA256
    ):
        raise D5RunnerValidationError(
            "SG config identity",
            "bytes or SHA256 mismatch",
        )

    started = perf_counter()
    condition_matrices = prepare_d5_conditions(cohort, sg_config_path)
    matching = [
        match_d5_condition(cohort, split, condition)
        for condition in condition_matrices
    ]
    outcomes = _paired_outcomes(cohort, matching[0], matching[1])
    outcome_bytes = _outcomes_jsonl_bytes(outcomes)
    condition_documents = [
        _condition_document(current, matrix.values)
        for current, matrix in zip(
            matching,
            condition_matrices,
            strict=True,
        )
    ]
    result = {
        "claim_boundary": {
            "descriptive_only": True,
            "five_seed_cell": False,
            "formal_inference": False,
            "retained_raw_comparison": result_level
            in {"provisional", "complete_seed"},
        },
        "code": _code_document(),
        "conditions": condition_documents,
        "dataset": _dataset_document(cohort),
        "descriptive_deltas_percentage_points": _descriptive_deltas(
            condition_documents
        ),
        "environment": _environment_document(),
        "experiment_id": EXPERIMENT_ID,
        "paired_outcomes": outcomes,
        "paired_outcomes_artifact": {
            "bytes": len(outcome_bytes),
            "lines": len(outcomes),
            "path": (
                f"paired_outcomes_seed{seed}_complete.jsonl"
                if result_level == "complete_seed"
                else f"paired_outcomes_seed{seed}.jsonl"
            ),
            "sha256": hashlib.sha256(outcome_bytes).hexdigest(),
        },
        "paired_summary": _paired_summary(outcomes),
        "protocol_config": {
            "bytes": CONFIG_BYTES,
            "path": "d5_rruff_protocol.json",
            "sha256": CONFIG_SHA256,
        },
        "result_label": (
            "PROVISIONAL — 1/5 SEEDS"
            if result_level == "provisional"
            else (
                f"COMPLETE SEED — {seed + 1}/5"
                if result_level == "complete_seed"
                else "SMOKE — PROVIDED COHORT"
            )
        ),
        "result_level": result_level,
        "runtime_seconds": perf_counter() - started,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "seeds_completed": (
            seed + 1 if result_level == "complete_seed" else 1
        ),
        "seeds_required": SEEDS_REQUIRED,
        "sg_config": {
            "bytes": SG_CONFIG_BYTES,
            "path": sg_config_path.name,
            "sha256": SG_CONFIG_SHA256,
        },
        "split": _split_document(cohort, matching[0], outcomes),
        "status": "completed",
    }
    validate_d5_provisional_result(result)
    return result


def run_d5_provisional_experiment(
    config_path: Path,
    dataset_path: Path,
    sg_config_path: Path,
    *,
    seed: int,
    result_level: str,
) -> dict[str, object]:
    _validate_level_and_seed(seed=seed, result_level=result_level)
    config = load_d5_protocol_config(Path(config_path))
    if config.sha256 != CONFIG_SHA256:
        raise D5RunnerValidationError(
            "protocol config",
            "SHA256 mismatch",
        )
    cohort = load_d5_raw_cohort(config_path, dataset_path)
    return run_d5_provisional_from_cohort(
        cohort,
        sg_config_path,
        seed=seed,
        result_level=result_level,
    )


def _validate_outcome_row(
    row: Mapping[str, object],
) -> None:
    true_class = row.get("true_class_label")
    conditions = _object("paired outcomes conditions", row.get("conditions"))
    if set(conditions) != set(CONDITION_IDS):
        raise D5RunnerValidationError(
            "paired outcomes",
            "condition IDs mismatch",
        )
    for condition_id in CONDITION_IDS:
        condition = _object(
            "paired outcomes condition",
            conditions[condition_id],
        )
        top5_labels = condition.get("top5_class_labels")
        top5_scores = condition.get("top5_scores")
        if (
            not isinstance(top5_labels, list)
            or not 1 <= len(top5_labels) <= 5
            or len(set(top5_labels)) != len(top5_labels)
            or not isinstance(top5_scores, list)
            or len(top5_scores) != len(top5_labels)
            or condition.get("top1_class_label") != top5_labels[0]
            or condition.get("top1_score") != top5_scores[0]
            or condition.get("top1_correct")
            is not (top5_labels[0] == true_class)
            or condition.get("top5_correct")
            is not (true_class in top5_labels)
            or any(
                not isinstance(score, (int, float))
                or isinstance(score, bool)
                or not np.isfinite(float(score))
                or not -1.0 <= float(score) <= 1.0
                for score in top5_scores
            )
        ):
            raise D5RunnerValidationError(
                "paired outcomes",
                "invalid top-k labels, scores, or correctness",
            )
        for index in range(1, len(top5_scores)):
            previous_score = float(top5_scores[index - 1])
            current_score = float(top5_scores[index])
            previous_label = top5_labels[index - 1]
            current_label = top5_labels[index]
            if current_score > previous_score or (
                current_score == previous_score
                and current_label < previous_label
            ):
                raise D5RunnerValidationError(
                    "ranked scores",
                    "must descend, with ascending class labels for ties",
                )


def _metrics_from_outcomes(
    outcomes: list[Mapping[str, object]],
    condition_id: str,
) -> dict[str, object]:
    true_labels = np.asarray(
        [int(row["true_class_label"]) for row in outcomes],
        dtype="<i8",
    )
    top1 = np.asarray(
        [
            bool(row["conditions"][condition_id]["top1_correct"])
            for row in outcomes
        ],
        dtype=np.bool_,
    )
    top5 = np.asarray(
        [
            bool(row["conditions"][condition_id]["top5_correct"])
            for row in outcomes
        ],
        dtype=np.bool_,
    )
    matching = type(
        "_MetricProjection",
        (),
        {
            "true_class_labels": true_labels,
            "top1_correct": top1,
            "top5_correct": top5,
        },
    )()
    return _metrics(matching)


def validate_d5_provisional_result(
    result: Mapping[str, object],
) -> None:
    root = _object("result", result)
    if root.get("schema_version") != SCHEMA_VERSION:
        raise D5RunnerValidationError(
            "schema_version",
            f"must equal {SCHEMA_VERSION!r}",
        )
    if root.get("experiment_id") != EXPERIMENT_ID:
        raise D5RunnerValidationError(
            "experiment_id",
            f"must equal {EXPERIMENT_ID!r}",
        )
    if root.get("status") != "completed":
        raise D5RunnerValidationError("status", "must equal completed")
    result_level = root.get("result_level")
    seed = root.get("seed")
    _validate_level_and_seed(seed=seed, result_level=result_level)
    expected_label = (
        "PROVISIONAL — 1/5 SEEDS"
        if result_level == "provisional"
        else (
            f"COMPLETE SEED — {seed + 1}/5"
            if result_level == "complete_seed"
            else "SMOKE — PROVIDED COHORT"
        )
    )
    if root.get("result_label") != expected_label:
        raise D5RunnerValidationError(
            "result label",
            f"must equal {expected_label!r}",
        )
    if result_level in {"provisional", "complete_seed"}:
        if root.get("dataset") != _expected_provisional_dataset():
            raise D5RunnerValidationError(
                "provisional dataset",
                "must equal the frozen retained cohort identities",
            )
        claim_boundary = _object(
            "claim_boundary",
            root.get("claim_boundary"),
        )
        if claim_boundary.get("retained_raw_comparison") is not True:
            raise D5RunnerValidationError(
                "provisional dataset",
                "must declare retained_raw_comparison=true",
            )
    expected_completed = (
        seed + 1 if result_level == "complete_seed" else 1
    )
    if (
        root.get("seeds_completed") != expected_completed
        or root.get("seeds_required") != 5
    ):
        raise D5RunnerValidationError(
            "seed completion",
            f"must equal {expected_completed}/5",
        )
    if _recursive_keys(root) & FORBIDDEN_INFERENCE_KEYS:
        raise D5RunnerValidationError(
            "inference boundary",
            "contains forbidden inference fields",
        )
    outcomes = root.get("paired_outcomes")
    if not isinstance(outcomes, list) or not outcomes:
        raise D5RunnerValidationError(
            "paired outcomes",
            "must be a nonempty list",
        )
    record_ids = []
    query_indices = []
    for row in outcomes:
        document = _object("paired outcomes row", row)
        _validate_outcome_row(document)
        record_id = document.get("record_id")
        query_index = document.get("query_index")
        if (
            not isinstance(record_id, str)
            or record_id == ""
            or isinstance(query_index, bool)
            or not isinstance(query_index, int)
            or query_index < 0
        ):
            raise D5RunnerValidationError(
                "paired outcomes",
                "invalid record or query identity",
            )
        record_ids.append(record_id)
        query_indices.append(query_index)
    if (
        len(set(record_ids)) != len(record_ids)
        or len(set(query_indices)) != len(query_indices)
    ):
        raise D5RunnerValidationError(
            "paired outcomes",
            "record IDs and query indices must be unique",
        )
    conditions = root.get("conditions")
    if (
        not isinstance(conditions, list)
        or [
            condition.get("condition_id")
            for condition in conditions
        ]
        != list(CONDITION_IDS)
    ):
        raise D5RunnerValidationError(
            "conditions",
            "must equal control then SG",
        )
    for condition in conditions:
        condition_id = str(condition["condition_id"])
        observed = condition.get("metrics")
        expected = _metrics_from_outcomes(outcomes, condition_id)
        if observed != expected:
            raise D5RunnerValidationError(
                "metrics",
                f"{condition_id} does not match paired outcomes",
            )
    expected_deltas = _descriptive_deltas(conditions)
    if root.get("descriptive_deltas_percentage_points") != expected_deltas:
        raise D5RunnerValidationError(
            "descriptive deltas",
            "do not match condition metrics",
        )
    expected_summary = _paired_summary(outcomes)
    if root.get("paired_summary") != expected_summary:
        raise D5RunnerValidationError(
            "paired summary",
            "does not match paired outcomes",
        )
    payload = paired_outcomes_jsonl_bytes(root)
    artifact = root.get("paired_outcomes_artifact")
    expected_artifact = {
        "bytes": len(payload),
        "lines": len(outcomes),
        "path": (
            f"paired_outcomes_seed{seed}_complete.jsonl"
            if result_level == "complete_seed"
            else f"paired_outcomes_seed{seed}.jsonl"
        ),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if artifact != expected_artifact:
        raise D5RunnerValidationError(
            "paired outcomes artifact",
            "bytes, lines, path, or SHA256 mismatch",
        )
    split = _object("split", root.get("split"))
    if (
        split.get("seed") != seed
        or split.get("query_count") != len(outcomes)
        or split.get("query_record_ids_sha256") != _ids_digest(record_ids)
        or split.get("query_metadata_sha256")
        != _outcome_metadata_sha256(outcomes)
    ):
        raise D5RunnerValidationError(
            "outcome metadata",
            "seed, query count, record IDs, or metadata projection mismatch",
        )
    if result_level in {"provisional", "complete_seed"}:
        frozen_split = FROZEN_SPLITS[seed]
        for key, expected in frozen_split.items():
            if split.get(key) != expected:
                raise D5RunnerValidationError(
                    "provisional split",
                    f"{key} must equal {expected!r}",
                )
    protocol = _object("protocol_config", root.get("protocol_config"))
    if protocol != {
        "bytes": CONFIG_BYTES,
        "path": "d5_rruff_protocol.json",
        "sha256": CONFIG_SHA256,
    }:
        raise D5RunnerValidationError(
            "protocol config",
            "identity mismatch",
        )
    sg_config = _object("sg_config", root.get("sg_config"))
    if sg_config != {
        "bytes": SG_CONFIG_BYTES,
        "path": "d1_sg11_poly3_interp.json",
        "sha256": SG_CONFIG_SHA256,
    }:
        raise D5RunnerValidationError(
            "SG config",
            "identity mismatch",
        )
    code = _object("code", root.get("code"))
    if code != _code_document():
        raise D5RunnerValidationError(
            "code",
            "source identities differ from current runner",
        )
    runtime = root.get("runtime_seconds")
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or not np.isfinite(float(runtime))
        or float(runtime) < 0.0
    ):
        raise D5RunnerValidationError(
            "runtime_seconds",
            "must be a non-negative finite number",
        )
    _canonical_json_bytes(root)


__all__ = [
    "D5RunnerValidationError",
    "paired_outcomes_jsonl_bytes",
    "run_d5_provisional_experiment",
    "run_d5_provisional_from_cohort",
    "validate_d5_provisional_result",
]
