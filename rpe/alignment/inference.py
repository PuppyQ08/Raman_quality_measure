from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from rpe.alignment.contracts import (
    AlignmentObservation,
    AlignmentValidationError,
)
from rpe.alignment.core import compare_alignment


@dataclass(frozen=True)
class AlignmentBootstrapResult:
    confidence_level: float
    resamples: int
    random_seed: int
    sample_size: int
    duplicate_cluster_resamples: int
    reference_ag_interval: tuple[float, float]
    candidate_ag_interval: tuple[float, float]
    d_ag_interval: tuple[float, float]
    reference_acc_interval: tuple[float, float]
    candidate_acc_interval: tuple[float, float]
    d_acc_interval: tuple[float, float]


@dataclass(frozen=True)
class SignFlipResult:
    aggregation: str
    alternative: str
    observed: float
    extreme_resamples: int
    p_value: float
    p_value_correction: str
    random_seed: int
    resamples: int
    sample_size: int


@dataclass(frozen=True)
class HolmResult:
    hypothesis_id: str
    raw_p_value: float
    adjusted_p_value: float
    rank: int
    family_size: int
    alpha: float
    rejected: bool


def _positive_integer(path: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AlignmentValidationError(path, "must be a positive integer")
    return value


def _seed(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AlignmentValidationError("random_seed", "must be a nonnegative integer")
    return value


def _probability(path: str, value: object, *, strict: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentValidationError(path, "must be a finite probability")
    converted = float(value)
    lower_ok = converted > 0.0 if strict else converted >= 0.0
    if not math.isfinite(converted) or not lower_ok or converted > 1.0:
        raise AlignmentValidationError(path, "must be a finite probability")
    return converted


def _interval(values: np.ndarray, confidence_level: float) -> tuple[float, float]:
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return float(lower), float(upper)


def _bootstrap_table(
    observations: Sequence[AlignmentObservation],
    sampled_clusters: np.ndarray,
    original_cluster_ids: tuple[str, ...],
) -> tuple[AlignmentObservation, ...]:
    by_cluster = {
        cluster_id: tuple(row for row in observations if row.cluster_id == cluster_id)
        for cluster_id in original_cluster_ids
    }
    output = []
    for copy_index, sampled_index in enumerate(sampled_clusters):
        source_id = original_cluster_ids[int(sampled_index)]
        new_id = f"bootstrap-{copy_index:08d}::{source_id}"
        output.extend(
            AlignmentObservation(
                cluster_id=new_id,
                perturbation_id=row.perturbation_id,
                alpha=row.alpha,
                metric_harm=row.metric_harm,
                downstream_harm=row.downstream_harm,
            )
            for row in by_cluster[source_id]
        )
    return tuple(output)


def paired_cluster_bootstrap(
    reference: Sequence[AlignmentObservation],
    candidate: Sequence[AlignmentObservation],
    *,
    resamples: int = 2000,
    confidence_level: float = 0.95,
    random_seed: int = 20260817,
) -> AlignmentBootstrapResult:
    count = _positive_integer("resamples", resamples)
    confidence = _probability("confidence_level", confidence_level, strict=True)
    seed = _seed(random_seed)
    comparison = compare_alignment(reference, candidate)
    cluster_ids = comparison.reference_gap.cluster_ids
    sample_size = len(cluster_ids)
    if sample_size < 2:
        raise AlignmentValidationError("clusters", "must contain at least two")
    generator = np.random.Generator(np.random.PCG64(seed))
    draws = generator.integers(0, sample_size, size=(count, sample_size))
    values = np.empty((count, 6), dtype=np.float64)
    duplicate_count = 0
    for index, sampled in enumerate(draws):
        if np.unique(sampled).size < sample_size:
            duplicate_count += 1
        reference_table = _bootstrap_table(reference, sampled, cluster_ids)
        candidate_table = _bootstrap_table(candidate, sampled, cluster_ids)
        try:
            current = compare_alignment(reference_table, candidate_table)
        except AlignmentValidationError as error:
            raise AlignmentValidationError(
                f"bootstrap resample {index}",
                error.reason,
            ) from error
        values[index] = (
            current.reference_gap.alignment_gap,
            current.candidate_gap.alignment_gap,
            current.d_ag,
            current.reference_accuracy.accuracy,
            current.candidate_accuracy.accuracy,
            current.d_acc,
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


def paired_contribution_sign_flip(
    contributions: Sequence[float],
    *,
    aggregation: str,
    resamples: int = 100000,
    random_seed: int = 20260817,
) -> SignFlipResult:
    if isinstance(contributions, (str, bytes)) or not isinstance(contributions, Sequence) or not contributions:
        raise AlignmentValidationError("contributions", "must be a nonempty sequence")
    values = np.asarray(contributions, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise AlignmentValidationError("contributions finite", "must contain finite values")
    if aggregation not in {"sum", "mean"}:
        raise AlignmentValidationError("aggregation", "must be sum or mean")
    count = _positive_integer("resamples", resamples)
    seed = _seed(random_seed)
    reducer = np.sum if aggregation == "sum" else np.mean
    observed = float(reducer(values))
    threshold = abs(observed)
    generator = np.random.Generator(np.random.PCG64(seed))
    extreme = 0
    completed = 0
    while completed < count:
        current = min(1024, count - completed)
        bits = generator.integers(
            0,
            2,
            size=(current, values.size),
            dtype=np.int8,
        )
        signs = bits * np.int8(2) - np.int8(1)
        signed = signs.astype(np.float64, copy=False) * values[None, :]
        permuted = reducer(signed, axis=1)
        extreme += int(np.sum(np.abs(permuted) + 1e-15 >= threshold))
        completed += current
    return SignFlipResult(
        aggregation=aggregation,
        alternative="two-sided",
        observed=observed,
        extreme_resamples=extreme,
        p_value=(extreme + 1) / (count + 1),
        p_value_correction="plus_one",
        random_seed=seed,
        resamples=count,
        sample_size=values.size,
    )


def holm_step_down(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> tuple[HolmResult, ...]:
    if not isinstance(p_values, Mapping) or not p_values:
        raise AlignmentValidationError("p_values", "must be a nonempty mapping")
    if any(not isinstance(key, str) or key == "" for key in p_values):
        raise AlignmentValidationError("p_values", "hypothesis IDs must be nonempty strings")
    threshold = _probability("alpha", alpha, strict=True)
    ordered = []
    for hypothesis_id, value in p_values.items():
        raw = _probability("p_values", value, strict=False)
        ordered.append((raw, hypothesis_id))
    ordered.sort(key=lambda item: (item[0], item[1]))
    family_size = len(ordered)
    adjusted_by_id: dict[str, tuple[float, int]] = {}
    running = 0.0
    for index, (raw, hypothesis_id) in enumerate(ordered):
        adjusted = min(1.0, (family_size - index) * raw)
        running = max(running, adjusted)
        adjusted_by_id[hypothesis_id] = (running, index + 1)
    return tuple(
        HolmResult(
            hypothesis_id=hypothesis_id,
            raw_p_value=float(p_values[hypothesis_id]),
            adjusted_p_value=adjusted_by_id[hypothesis_id][0],
            rank=adjusted_by_id[hypothesis_id][1],
            family_size=family_size,
            alpha=threshold,
            rejected=adjusted_by_id[hypothesis_id][0] < threshold,
        )
        for hypothesis_id in sorted(p_values)
    )


__all__ = [
    "AlignmentBootstrapResult",
    "HolmResult",
    "SignFlipResult",
    "holm_step_down",
    "paired_cluster_bootstrap",
    "paired_contribution_sign_flip",
]
