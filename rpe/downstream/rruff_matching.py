from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rpe.downstream.rruff import (
    CONFIG_SHA256,
    DATASET_ID,
    D5LibraryQuerySplit,
    D5RawCohort,
    GRID_START_CM1,
    GRID_STEP_CM1,
    GRID_STOP_CM1,
)
from rpe.methods.classical.savitzky_golay import (
    load_savitzky_golay_pipeline,
)


CONTROL_CONDITION_ID = "raw_aligned_control"
SG_CONDITION_ID = "raw_aligned_plus_sg11"
CONDITION_IDS = (CONTROL_CONDITION_ID, SG_CONDITION_ID)
EXPECTED_FEATURE_COUNT = 801
UNIT_NORM_ATOL = 1e-12


class D5MatchingValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class D5ConditionMatrix:
    condition_id: str
    protocol_config_sha256: str
    record_ids: tuple[str, ...]
    values: np.ndarray


@dataclass(frozen=True)
class D5MatchingResult:
    condition_id: str
    protocol_config_sha256: str
    seed: int
    split_sha256: str
    query_indices: np.ndarray
    query_record_ids: tuple[str, ...]
    true_class_labels: np.ndarray
    ranked_class_labels: np.ndarray
    ranked_class_scores: np.ndarray
    top1_correct: np.ndarray
    top5_correct: np.ndarray


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _validate_cohort(cohort: D5RawCohort) -> None:
    if not isinstance(cohort, D5RawCohort):
        raise D5MatchingValidationError(
            "cohort",
            "must be a D5RawCohort",
        )
    if cohort.protocol_config_sha256 != CONFIG_SHA256:
        raise D5MatchingValidationError(
            "protocol config",
            f"must equal {CONFIG_SHA256}",
        )
    if cohort.dataset_id != DATASET_ID:
        raise D5MatchingValidationError(
            "dataset_id",
            f"must equal {DATASET_ID!r}",
        )
    intensity = cohort.intensity
    if (
        not isinstance(intensity, np.ndarray)
        or intensity.dtype != np.dtype("<f4")
        or intensity.ndim != 2
        or intensity.shape[1] != EXPECTED_FEATURE_COUNT
        or intensity.shape[0] == 0
    ):
        raise D5MatchingValidationError(
            "cohort intensity",
            "must be a nonempty little-endian float32 matrix with 801 columns",
        )
    if not np.isfinite(intensity).all():
        raise D5MatchingValidationError(
            "cohort intensity finite",
            "must contain only finite values",
        )
    expected_wavenumber = np.arange(
        GRID_START_CM1,
        GRID_STOP_CM1 + GRID_STEP_CM1 / 2.0,
        GRID_STEP_CM1,
        dtype="<f4",
    )
    if (
        not isinstance(cohort.wavenumber, np.ndarray)
        or cohort.wavenumber.dtype != np.dtype("<f4")
        or not np.array_equal(
            cohort.wavenumber,
            expected_wavenumber,
        )
    ):
        raise D5MatchingValidationError(
            "frozen wavenumber",
            "must equal the 200..1800 cm^-1 grid at 2 cm^-1 spacing",
        )
    row_count = intensity.shape[0]
    if (
        not isinstance(cohort.class_labels, np.ndarray)
        or cohort.class_labels.dtype != np.dtype("<i8")
        or cohort.class_labels.shape != (row_count,)
        or any(
            len(values) != row_count
            for values in (
                cohort.record_ids,
                cohort.mineral_names,
                cohort.rruff_ids,
                cohort.pin_ids,
                cohort.group_ids,
            )
        )
    ):
        raise D5MatchingValidationError(
            "cohort rows",
            "class labels and identity fields must align with intensity rows",
        )


def _l2_normalize(path: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype="<f8")
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise D5MatchingValidationError(
            f"{path} finite",
            "must be a finite two-dimensional matrix",
        )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 0.0):
        raise D5MatchingValidationError(
            f"{path} zero L2 norm",
            "every spectrum must have a positive finite L2 norm",
        )
    normalized = np.asarray(
        matrix / norms[:, None],
        dtype="<f8",
    ).copy()
    if not np.isfinite(normalized).all():
        raise D5MatchingValidationError(
            f"{path} normalized finite",
            "normalization produced non-finite values",
        )
    return _read_only(normalized)


def prepare_d5_conditions(
    cohort: D5RawCohort,
    sg_config_path: Path,
) -> tuple[D5ConditionMatrix, D5ConditionMatrix]:
    _validate_cohort(cohort)
    pipeline = load_savitzky_golay_pipeline(Path(sg_config_path))
    control = D5ConditionMatrix(
        condition_id=CONTROL_CONDITION_ID,
        protocol_config_sha256=cohort.protocol_config_sha256,
        record_ids=cohort.record_ids,
        values=_l2_normalize(
            CONTROL_CONDITION_ID,
            cohort.intensity,
        ),
    )
    transformed = pipeline.transform(cohort.intensity)
    sg = D5ConditionMatrix(
        condition_id=SG_CONDITION_ID,
        protocol_config_sha256=cohort.protocol_config_sha256,
        record_ids=cohort.record_ids,
        values=_l2_normalize(
            SG_CONDITION_ID,
            transformed,
        ),
    )
    return control, sg


def _validated_indices(
    path: str,
    values: np.ndarray,
    row_count: int,
) -> np.ndarray:
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype("<i8")
        or values.ndim != 1
        or values.size == 0
    ):
        raise D5MatchingValidationError(
            path,
            "must be a nonempty one-dimensional int64 array",
        )
    if (
        np.any(values < 0)
        or np.any(values >= row_count)
        or np.unique(values).size != values.size
    ):
        raise D5MatchingValidationError(
            path,
            "must contain unique in-range row indices",
        )
    return values


def _validate_condition(
    condition: D5ConditionMatrix,
    cohort: D5RawCohort,
) -> None:
    if not isinstance(condition, D5ConditionMatrix):
        raise D5MatchingValidationError(
            "condition",
            "must be a D5ConditionMatrix",
        )
    if condition.condition_id not in CONDITION_IDS:
        raise D5MatchingValidationError(
            "condition_id",
            f"must be one of {CONDITION_IDS!r}",
        )
    if condition.protocol_config_sha256 != cohort.protocol_config_sha256:
        raise D5MatchingValidationError(
            "condition protocol config",
            "must equal the cohort protocol config SHA256",
        )
    if condition.record_ids != cohort.record_ids:
        raise D5MatchingValidationError(
            "condition row identity",
            "record IDs must equal the cohort row order",
        )
    row_count = cohort.intensity.shape[0]
    values = condition.values
    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.dtype("<f8")
        or values.shape != (row_count, EXPECTED_FEATURE_COUNT)
        or not np.isfinite(values).all()
    ):
        raise D5MatchingValidationError(
            "condition values",
            "must be a finite float64 matrix aligned to the cohort",
        )
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(
        norms,
        np.ones(row_count, dtype=np.float64),
        rtol=0.0,
        atol=UNIT_NORM_ATOL,
    ):
        raise D5MatchingValidationError(
            "condition unit L2 norm",
            "every condition row must have unit L2 norm",
        )


def _validated_frozen_split(
    cohort: D5RawCohort,
    split: D5LibraryQuerySplit,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(split, D5LibraryQuerySplit):
        raise D5MatchingValidationError(
            "split",
            "must be a D5LibraryQuerySplit",
        )
    row_count = cohort.intensity.shape[0]
    frozen_splits = [
        frozen
        for frozen in cohort.splits
        if frozen.seed == split.seed
    ]
    if (
        len(frozen_splits) != 1
        or split.split_sha256 != frozen_splits[0].split_sha256
        or not np.array_equal(split.query_indices, frozen_splits[0].query_indices)
        or not np.array_equal(split.library_indices, frozen_splits[0].library_indices)
    ):
        raise D5MatchingValidationError(
            "frozen split",
            "seed, SHA256, and indices must equal one cohort split",
        )
    query_indices = _validated_indices(
        "query indices",
        split.query_indices,
        row_count,
    )
    library_indices = _validated_indices(
        "library indices",
        split.library_indices,
        row_count,
    )
    if np.intersect1d(query_indices, library_indices, assume_unique=True).size:
        raise D5MatchingValidationError(
            "split records",
            "query and library indices must be disjoint",
        )
    if not np.array_equal(
        np.sort(np.concatenate((query_indices, library_indices))),
        np.arange(row_count, dtype="<i8"),
    ):
        raise D5MatchingValidationError(
            "complete partition",
            "query and library must cover every cohort row exactly once",
        )
    query_groups = {cohort.group_ids[int(index)] for index in query_indices}
    library_groups = {cohort.group_ids[int(index)] for index in library_indices}
    if query_groups & library_groups:
        raise D5MatchingValidationError(
            "leakage group",
            "a leakage group crosses query and library",
        )
    return query_indices, library_indices


def _match_normalized_values(
    cohort: D5RawCohort,
    split: D5LibraryQuerySplit,
    *,
    condition_id: str,
    query_indices: np.ndarray,
    library_indices: np.ndarray,
    query_values: np.ndarray,
    library_values: np.ndarray,
) -> D5MatchingResult:
    true_class_labels = np.asarray(cohort.class_labels[query_indices], dtype="<i8")
    library_class_labels = np.asarray(cohort.class_labels[library_indices], dtype="<i8")
    unique_library_classes = np.unique(library_class_labels)
    missing = sorted(
        set(int(value) for value in true_class_labels)
        - set(int(value) for value in unique_library_classes)
    )
    if missing:
        raise D5MatchingValidationError(
            "true query class",
            f"missing from library: {missing}",
        )

    similarities = np.clip(query_values @ library_values.T, -1.0, 1.0)
    class_scores = np.empty(
        (query_indices.size, unique_library_classes.size),
        dtype="<f8",
    )
    for column, class_label in enumerate(unique_library_classes):
        class_scores[:, column] = np.max(
            similarities[:, library_class_labels == class_label],
            axis=1,
        )
    rank_order = np.empty(class_scores.shape, dtype="<i8")
    for row in range(class_scores.shape[0]):
        rank_order[row] = np.lexsort((unique_library_classes, -class_scores[row]))
    ranked_class_labels = np.asarray(unique_library_classes[rank_order], dtype="<i8")
    ranked_class_scores = np.asarray(
        np.take_along_axis(class_scores, rank_order, axis=1),
        dtype="<f8",
    )
    top1_correct = ranked_class_labels[:, 0] == true_class_labels
    top_k = min(5, ranked_class_labels.shape[1])
    top5_correct = np.any(
        ranked_class_labels[:, :top_k] == true_class_labels[:, None],
        axis=1,
    )
    return D5MatchingResult(
        condition_id=condition_id,
        protocol_config_sha256=cohort.protocol_config_sha256,
        seed=split.seed,
        split_sha256=split.split_sha256,
        query_indices=_read_only(np.asarray(query_indices, dtype="<i8").copy()),
        query_record_ids=tuple(cohort.record_ids[int(index)] for index in query_indices),
        true_class_labels=_read_only(np.asarray(true_class_labels, dtype="<i8").copy()),
        ranked_class_labels=_read_only(ranked_class_labels.copy()),
        ranked_class_scores=_read_only(ranked_class_scores.copy()),
        top1_correct=_read_only(np.asarray(top1_correct, dtype=np.bool_).copy()),
        top5_correct=_read_only(np.asarray(top5_correct, dtype=np.bool_).copy()),
    )


def match_d5_protocol_a_values(
    cohort: D5RawCohort,
    split: D5LibraryQuerySplit,
    *,
    condition_id: str,
    query_record_ids: tuple[str, ...],
    library_record_ids: tuple[str, ...],
    query_values: np.ndarray,
    library_values: np.ndarray,
) -> D5MatchingResult:
    _validate_cohort(cohort)
    if not isinstance(condition_id, str) or condition_id == "":
        raise D5MatchingValidationError(
            "condition_id",
            "must be a nonempty string",
        )
    query_indices, library_indices = _validated_frozen_split(cohort, split)
    expected_query_ids = tuple(
        cohort.record_ids[int(index)] for index in query_indices
    )
    expected_library_ids = tuple(
        cohort.record_ids[int(index)] for index in library_indices
    )
    if query_record_ids != expected_query_ids:
        raise D5MatchingValidationError(
            "query record IDs",
            "must equal the frozen query row order",
        )
    if library_record_ids != expected_library_ids:
        raise D5MatchingValidationError(
            "library record IDs",
            "must equal the frozen library row order",
        )
    query = np.asarray(query_values)
    library = np.asarray(library_values)
    if query.ndim != 2 or query.shape[0] != query_indices.size:
        raise D5MatchingValidationError(
            "query values shape",
            "must align with frozen query rows",
        )
    if library.ndim != 2 or library.shape[0] != library_indices.size:
        raise D5MatchingValidationError(
            "library values shape",
            "must align with frozen library rows",
        )
    if query.shape[1] == 0 or query.shape[1] != library.shape[1]:
        raise D5MatchingValidationError(
            "feature count",
            "query and library must share one positive feature count",
        )
    normalized_query = _l2_normalize("query values", query)
    normalized_library = _l2_normalize("library values", library)
    return _match_normalized_values(
        cohort,
        split,
        condition_id=condition_id,
        query_indices=query_indices,
        library_indices=library_indices,
        query_values=normalized_query,
        library_values=normalized_library,
    )


def match_d5_condition(
    cohort: D5RawCohort,
    split: D5LibraryQuerySplit,
    condition: D5ConditionMatrix,
) -> D5MatchingResult:
    _validate_cohort(cohort)
    query_indices, library_indices = _validated_frozen_split(cohort, split)
    _validate_condition(condition, cohort)
    return _match_normalized_values(
        cohort,
        split,
        condition_id=condition.condition_id,
        query_indices=query_indices,
        library_indices=library_indices,
        query_values=condition.values[query_indices],
        library_values=condition.values[library_indices],
    )


__all__ = [
    "D5ConditionMatrix",
    "D5MatchingResult",
    "D5MatchingValidationError",
    "match_d5_condition",
    "match_d5_protocol_a_values",
    "prepare_d5_conditions",
]
