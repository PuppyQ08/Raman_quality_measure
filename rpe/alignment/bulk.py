from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

from rpe.alignment.contracts import (
    AlignmentObservation,
    AlignmentValidationError,
)
from rpe.alignment.core import AG_TOLERANCE, _canonical_table, compare_alignment
from rpe.alignment.inference import (
    AlignmentBootstrapResult,
    _interval,
    _positive_integer,
    _probability,
    _seed,
)


def _table_arrays(
    observations: tuple[AlignmentObservation, ...],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], tuple[float, ...]]:
    canonical, cluster_ids, perturbation_ids, alpha_grid = _canonical_table(observations)
    cluster_count = len(cluster_ids)
    condition_count = len(perturbation_ids) * len(alpha_grid)
    metric = np.asarray([row.metric_harm for row in canonical], dtype=np.float64).reshape(
        cluster_count,
        condition_count,
    )
    downstream = np.asarray(
        [row.downstream_harm for row in canonical],
        dtype=np.float64,
    ).reshape(cluster_count, condition_count)
    return metric, downstream, cluster_ids, perturbation_ids, alpha_grid


def _cross_pair_indices(
    perturbation_ids: tuple[str, ...],
    alpha_grid: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    condition_perturbations = [
        perturbation_id
        for perturbation_id in perturbation_ids
        for _ in alpha_grid
    ]
    left_indices: list[int] = []
    right_indices: list[int] = []
    for left_index, left_perturbation in enumerate(condition_perturbations):
        for right_index in range(left_index + 1, len(condition_perturbations)):
            if left_perturbation == condition_perturbations[right_index]:
                continue
            left_indices.append(left_index)
            right_indices.append(right_index)
    return (
        np.asarray(left_indices, dtype=np.int64),
        np.asarray(right_indices, dtype=np.int64),
    )


def _cluster_accuracy_values(
    metric: np.ndarray,
    downstream: np.ndarray,
    left_indices: np.ndarray,
    right_indices: np.ndarray,
) -> np.ndarray:
    metric_difference = metric[:, left_indices] - metric[:, right_indices]
    downstream_difference = downstream[:, left_indices] - downstream[:, right_indices]
    double_tie = (metric_difference == 0.0) & (downstream_difference == 0.0)
    metric_tie = (metric_difference == 0.0) & ~double_tie
    downstream_tie = (downstream_difference == 0.0) & ~double_tie
    agreement = (
        ~double_tie
        & ~metric_tie
        & ~downstream_tie
        & ((metric_difference > 0.0) == (downstream_difference > 0.0))
    )
    score = (
        agreement.astype(np.float64)
        + 0.5 * (double_tie | metric_tie | downstream_tie).astype(np.float64)
    )
    return np.mean(score, axis=1, dtype=np.float64)


def _weighted_isotonic_predictions(
    metric: np.ndarray,
    downstream: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray:
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    model.fit(metric, downstream, sample_weight=sample_weight)
    return np.asarray(model.predict(metric), dtype=np.float64)


def _weighted_sst(downstream: np.ndarray, observation_weights: np.ndarray) -> float:
    total_weight = float(np.sum(observation_weights))
    mean = float(np.sum(observation_weights * downstream) / total_weight)
    return float(np.sum(observation_weights * (downstream - mean) ** 2))


def _weighted_alignment_gap(
    metric: np.ndarray,
    downstream: np.ndarray,
    cluster_weights: np.ndarray,
    alpha_count: int,
) -> float:
    cluster_count, condition_count = metric.shape
    observation_weights = np.repeat(cluster_weights, condition_count).astype(np.float64, copy=False)
    flat_metric = metric.reshape(cluster_count * condition_count)
    flat_downstream = downstream.reshape(cluster_count * condition_count)
    sst = _weighted_sst(flat_downstream, observation_weights)
    if not np.isfinite(sst) or sst <= 0.0:
        raise AlignmentValidationError("constant downstream", "global SST must be positive")
    pooled = _weighted_isotonic_predictions(flat_metric, flat_downstream, observation_weights)
    pooled_sse = float(np.sum(observation_weights * (flat_downstream - pooled) ** 2))
    separate_sse = 0.0
    perturbation_count = condition_count // alpha_count
    for perturbation_index in range(perturbation_count):
        start = perturbation_index * alpha_count
        stop = start + alpha_count
        metric_slice = metric[:, start:stop].reshape(cluster_count * alpha_count)
        downstream_slice = downstream[:, start:stop].reshape(cluster_count * alpha_count)
        weight_slice = np.repeat(cluster_weights, alpha_count).astype(np.float64, copy=False)
        separate = _weighted_isotonic_predictions(metric_slice, downstream_slice, weight_slice)
        separate_sse += float(np.sum(weight_slice * (downstream_slice - separate) ** 2))
    raw_gap = (pooled_sse - separate_sse) / sst
    if raw_gap < -AG_TOLERANCE:
        raise AlignmentValidationError("alignment gap", f"separate fit is worse by {raw_gap!r}")
    return 0.0 if raw_gap < 0.0 else float(raw_gap)


def bulk_paired_cluster_bootstrap(
    reference: tuple[AlignmentObservation, ...] | list[AlignmentObservation],
    candidate: tuple[AlignmentObservation, ...] | list[AlignmentObservation],
    *,
    resamples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 20260817,
) -> AlignmentBootstrapResult:
    count = _positive_integer("resamples", resamples)
    confidence = _probability("confidence_level", confidence_level, strict=True)
    seed = _seed(random_seed)
    comparison = compare_alignment(reference, candidate)
    sample_size = len(comparison.reference_gap.cluster_ids)
    if sample_size < 2:
        raise AlignmentValidationError("clusters", "must contain at least two")
    reference_metric, reference_downstream, _, perturbation_ids, alpha_grid = _table_arrays(tuple(reference))
    candidate_metric, candidate_downstream, _, _, _ = _table_arrays(tuple(candidate))
    if not np.array_equal(reference_downstream, candidate_downstream):
        raise AlignmentValidationError("paired downstream", "harms must be bit-equal")
    left_indices, right_indices = _cross_pair_indices(perturbation_ids, alpha_grid)
    reference_cluster_accuracy = _cluster_accuracy_values(
        reference_metric,
        reference_downstream,
        left_indices,
        right_indices,
    )
    candidate_cluster_accuracy = _cluster_accuracy_values(
        candidate_metric,
        candidate_downstream,
        left_indices,
        right_indices,
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(0, sample_size, size=(count, sample_size))
    cluster_weights = np.zeros((count, sample_size), dtype=np.int64)
    np.add.at(
        cluster_weights,
        (np.repeat(np.arange(count, dtype=np.int64), sample_size), draws.reshape(-1)),
        1,
    )
    duplicate_count = int(np.count_nonzero(np.any(cluster_weights > 1, axis=1)))
    values = np.empty((count, 6), dtype=np.float64)
    alpha_count = len(alpha_grid)
    for index in range(count):
        current_weights = cluster_weights[index]
        try:
            reference_gap = _weighted_alignment_gap(
                reference_metric,
                reference_downstream,
                current_weights,
                alpha_count,
            )
            candidate_gap = _weighted_alignment_gap(
                candidate_metric,
                candidate_downstream,
                current_weights,
                alpha_count,
            )
        except AlignmentValidationError as error:
            raise AlignmentValidationError(
                f"bootstrap resample {index}",
                error.reason,
            ) from error
        reference_accuracy = float(
            np.dot(current_weights, reference_cluster_accuracy) / sample_size
        )
        candidate_accuracy = float(
            np.dot(current_weights, candidate_cluster_accuracy) / sample_size
        )
        values[index] = (
            reference_gap,
            candidate_gap,
            reference_gap - candidate_gap,
            reference_accuracy,
            candidate_accuracy,
            candidate_accuracy - reference_accuracy,
        )
    return AlignmentBootstrapResult(
        confidence_level=confidence,
        resamples=count,
        random_seed=seed,
        sample_size=sample_size,
        duplicate_cluster_resamples=duplicate_count,
        reference_ag_interval=_interval(values[:, 0], confidence),
        candidate_ag_interval=_interval(values[:, 1], confidence),
        d_ag_interval=_interval(values[:, 2], confidence),
        reference_acc_interval=_interval(values[:, 3], confidence),
        candidate_acc_interval=_interval(values[:, 4], confidence),
        d_acc_interval=_interval(values[:, 5], confidence),
    )


__all__ = ["bulk_paired_cluster_bootstrap"]
