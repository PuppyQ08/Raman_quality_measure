from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression

from rpe.alignment.contracts import (
    AlignmentComparison,
    AlignmentGapResult,
    AlignmentObservation,
    AlignmentValidationError,
    ClusterPairContribution,
    ClusterValue,
    CrossPerturbationAccuracyResult,
)
from rpe.evaluation import PreferredDirection


AG_TOLERANCE = 1e-12


def _finite(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentValidationError(path, "must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise AlignmentValidationError(path, "must be finite")
    return converted


def orient_harm(
    baseline: float,
    current: float,
    direction: PreferredDirection,
    *,
    target_value: float | None = None,
) -> float:
    left = _finite("baseline", baseline)
    right = _finite("current", current)
    if not isinstance(direction, PreferredDirection):
        raise AlignmentValidationError("direction", "must be PreferredDirection")
    if direction is PreferredDirection.LOWER_IS_BETTER:
        return right - left
    if direction is PreferredDirection.HIGHER_IS_BETTER:
        return left - right
    if direction is PreferredDirection.TARGET_VALUE:
        if target_value is None:
            raise AlignmentValidationError("target_value", "must be supplied")
        target = _finite("target_value", target_value)
        return abs(right - target) - abs(left - target)
    raise AlignmentValidationError(
        "direction non_monotonic",
        "cannot be oriented for confirmatory alignment",
    )


def _canonical_table(
    observations: Sequence[AlignmentObservation],
) -> tuple[
    tuple[AlignmentObservation, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[float, ...],
]:
    if isinstance(observations, (str, bytes)) or not isinstance(observations, Sequence):
        raise AlignmentValidationError("observations", "must be a sequence")
    if not observations or any(not isinstance(row, AlignmentObservation) for row in observations):
        raise AlignmentValidationError(
            "observations",
            "must contain AlignmentObservation values",
        )
    canonical = tuple(sorted(observations))
    keys = [(row.cluster_id, row.perturbation_id, row.alpha) for row in canonical]
    if len(set(keys)) != len(keys):
        raise AlignmentValidationError("duplicate condition", "cluster-condition keys must be unique")
    cluster_ids = tuple(sorted({row.cluster_id for row in canonical}))
    perturbation_ids = tuple(sorted({row.perturbation_id for row in canonical}))
    if len(perturbation_ids) < 2:
        raise AlignmentValidationError("perturbations", "must contain at least two")
    alpha_by_perturbation = {
        perturbation_id: tuple(
            sorted({row.alpha for row in canonical if row.perturbation_id == perturbation_id})
        )
        for perturbation_id in perturbation_ids
    }
    first_grid = alpha_by_perturbation[perturbation_ids[0]]
    if len(first_grid) < 2 or any(
        alpha_by_perturbation[perturbation_id] != first_grid
        for perturbation_id in perturbation_ids[1:]
    ):
        raise AlignmentValidationError(
            "alpha support",
            "every perturbation must share at least two identical positive alphas",
        )
    expected_conditions = {
        (perturbation_id, alpha)
        for perturbation_id in perturbation_ids
        for alpha in first_grid
    }
    for cluster_id in cluster_ids:
        observed_conditions = {
            (row.perturbation_id, row.alpha)
            for row in canonical
            if row.cluster_id == cluster_id
        }
        if observed_conditions != expected_conditions:
            raise AlignmentValidationError(
                "complete Cartesian grid",
                f"cluster {cluster_id!r} is incomplete",
            )
    return canonical, cluster_ids, perturbation_ids, first_grid


def _isotonic_predictions(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    model = IsotonicRegression(increasing=True, out_of_bounds="clip")
    return np.asarray(
        model.fit_transform(x, y, sample_weight=np.ones(x.size, dtype=np.float64)),
        dtype=np.float64,
    )


def alignment_gap(
    observations: Sequence[AlignmentObservation],
) -> AlignmentGapResult:
    canonical, cluster_ids, perturbation_ids, alpha_grid = _canonical_table(observations)
    x = np.asarray([row.metric_harm for row in canonical], dtype=np.float64)
    y = np.asarray([row.downstream_harm for row in canonical], dtype=np.float64)
    mean = float(np.mean(y))
    sst = float(np.sum((y - mean) ** 2))
    if not math.isfinite(sst) or sst <= 0.0:
        raise AlignmentValidationError(
            "constant downstream",
            "global SST must be positive",
        )
    pooled = _isotonic_predictions(x, y)
    separate = np.empty(y.shape, dtype=np.float64)
    for perturbation_id in perturbation_ids:
        indices = np.asarray(
            [index for index, row in enumerate(canonical) if row.perturbation_id == perturbation_id],
            dtype=np.int64,
        )
        separate[indices] = _isotonic_predictions(x[indices], y[indices])
    pooled_squared = (y - pooled) ** 2
    separate_squared = (y - separate) ** 2
    sse_pooled = float(np.sum(pooled_squared))
    sse_separate = float(np.sum(separate_squared))
    raw_gap = (sse_pooled - sse_separate) / sst
    if raw_gap < -AG_TOLERANCE:
        raise AlignmentValidationError(
            "alignment gap",
            f"separate fit is worse by {raw_gap!r}",
        )
    gap = 0.0 if raw_gap < 0.0 else float(raw_gap)
    contributions = tuple(
        ClusterValue(
            cluster_id=cluster_id,
            value=float(
                np.sum(
                    (pooled_squared - separate_squared)[
                        [index for index, row in enumerate(canonical) if row.cluster_id == cluster_id]
                    ]
                )
                / sst
            ),
        )
        for cluster_id in cluster_ids
    )
    if not math.isclose(
        sum(item.value for item in contributions),
        gap,
        rel_tol=0.0,
        abs_tol=AG_TOLERANCE,
    ):
        raise AlignmentValidationError("cluster contributions", "must sum to AG")
    return AlignmentGapResult(
        observation_count=len(canonical),
        cluster_ids=cluster_ids,
        perturbation_ids=perturbation_ids,
        alpha_grid=alpha_grid,
        sst=sst,
        sse_pooled=sse_pooled,
        sse_separate=sse_separate,
        r2_pooled=1.0 - sse_pooled / sst,
        r2_separate=1.0 - sse_separate / sst,
        raw_alignment_gap=float(raw_gap),
        alignment_gap=gap,
        pooled_predictions=tuple(float(value) for value in pooled),
        separate_predictions=tuple(float(value) for value in separate),
        cluster_contributions=contributions,
    )


def _cluster_pair_contribution(
    cluster_id: str,
    rows: tuple[AlignmentObservation, ...],
) -> ClusterPairContribution:
    agreement = disagreement = metric_tie = downstream_tie = double_tie = 0
    pair_count = 0
    score = 0.0
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if left.perturbation_id == right.perturbation_id:
                continue
            pair_count += 1
            metric_difference = left.metric_harm - right.metric_harm
            downstream_difference = left.downstream_harm - right.downstream_harm
            if metric_difference == 0.0 and downstream_difference == 0.0:
                double_tie += 1
                score += 0.5
            elif metric_difference == 0.0:
                metric_tie += 1
                score += 0.5
            elif downstream_difference == 0.0:
                downstream_tie += 1
                score += 0.5
            elif (metric_difference > 0.0) is (downstream_difference > 0.0):
                agreement += 1
                score += 1.0
            else:
                disagreement += 1
    if pair_count == 0:
        raise AlignmentValidationError("cross pairs", "must contain at least one pair")
    return ClusterPairContribution(
        cluster_id=cluster_id,
        pair_count=pair_count,
        strict_agreement_count=agreement,
        strict_disagreement_count=disagreement,
        metric_tie_count=metric_tie,
        downstream_tie_count=downstream_tie,
        double_tie_count=double_tie,
        accuracy=score / pair_count,
    )


def cross_perturbation_accuracy(
    observations: Sequence[AlignmentObservation],
) -> CrossPerturbationAccuracyResult:
    canonical, cluster_ids, _, _ = _canonical_table(observations)
    by_cluster: dict[str, list[AlignmentObservation]] = defaultdict(list)
    for row in canonical:
        by_cluster[row.cluster_id].append(row)
    contributions = tuple(
        _cluster_pair_contribution(cluster_id, tuple(by_cluster[cluster_id]))
        for cluster_id in cluster_ids
    )
    return CrossPerturbationAccuracyResult(
        cluster_ids=cluster_ids,
        pair_count=sum(item.pair_count for item in contributions),
        strict_agreement_count=sum(item.strict_agreement_count for item in contributions),
        strict_disagreement_count=sum(item.strict_disagreement_count for item in contributions),
        metric_tie_count=sum(item.metric_tie_count for item in contributions),
        downstream_tie_count=sum(item.downstream_tie_count for item in contributions),
        double_tie_count=sum(item.double_tie_count for item in contributions),
        accuracy=float(np.mean([item.accuracy for item in contributions])),
        cluster_contributions=contributions,
    )


def compare_alignment(
    reference_observations: Sequence[AlignmentObservation],
    candidate_observations: Sequence[AlignmentObservation],
) -> AlignmentComparison:
    reference, reference_clusters, _, _ = _canonical_table(reference_observations)
    candidate, candidate_clusters, _, _ = _canonical_table(candidate_observations)
    reference_keys = tuple((row.cluster_id, row.perturbation_id, row.alpha) for row in reference)
    candidate_keys = tuple((row.cluster_id, row.perturbation_id, row.alpha) for row in candidate)
    if reference_keys != candidate_keys or reference_clusters != candidate_clusters:
        raise AlignmentValidationError("paired tables", "condition keys must be identical")
    if any(
        left.downstream_harm != right.downstream_harm
        for left, right in zip(reference, candidate, strict=True)
    ):
        raise AlignmentValidationError("paired downstream", "harms must be bit-equal")
    reference_gap = alignment_gap(reference)
    candidate_gap = alignment_gap(candidate)
    reference_accuracy = cross_perturbation_accuracy(reference)
    candidate_accuracy = cross_perturbation_accuracy(candidate)
    reference_ag = {item.cluster_id: item.value for item in reference_gap.cluster_contributions}
    candidate_ag = {item.cluster_id: item.value for item in candidate_gap.cluster_contributions}
    reference_acc = {item.cluster_id: item.accuracy for item in reference_accuracy.cluster_contributions}
    candidate_acc = {item.cluster_id: item.accuracy for item in candidate_accuracy.cluster_contributions}
    ag_differences = tuple(
        ClusterValue(cluster_id, reference_ag[cluster_id] - candidate_ag[cluster_id])
        for cluster_id in reference_clusters
    )
    acc_differences = tuple(
        ClusterValue(cluster_id, candidate_acc[cluster_id] - reference_acc[cluster_id])
        for cluster_id in reference_clusters
    )
    return AlignmentComparison(
        reference_gap=reference_gap,
        candidate_gap=candidate_gap,
        reference_accuracy=reference_accuracy,
        candidate_accuracy=candidate_accuracy,
        d_ag=reference_gap.alignment_gap - candidate_gap.alignment_gap,
        d_acc=candidate_accuracy.accuracy - reference_accuracy.accuracy,
        ag_contribution_differences=ag_differences,
        acc_contribution_differences=acc_differences,
    )


__all__ = [
    "AG_TOLERANCE",
    "alignment_gap",
    "compare_alignment",
    "cross_perturbation_accuracy",
    "orient_harm",
]
