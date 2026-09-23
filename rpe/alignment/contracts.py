from __future__ import annotations

import math
from dataclasses import dataclass


class AlignmentValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _nonempty(path: str, value: object) -> str:
    if not isinstance(value, str) or value == "":
        raise AlignmentValidationError(path, "must be a nonempty string")
    return value


def _finite(path: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AlignmentValidationError(path, "must be a finite real number")
    converted = float(value)
    if not math.isfinite(converted):
        raise AlignmentValidationError(path, "must be finite")
    return converted


@dataclass(frozen=True, order=True)
class AlignmentObservation:
    cluster_id: str
    perturbation_id: str
    alpha: float
    metric_harm: float
    downstream_harm: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "cluster_id", _nonempty("cluster_id", self.cluster_id))
        object.__setattr__(
            self,
            "perturbation_id",
            _nonempty("perturbation_id", self.perturbation_id),
        )
        alpha = _finite("alpha", self.alpha)
        if alpha <= 0.0:
            raise AlignmentValidationError("alpha", "must be a positive alpha")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "metric_harm", _finite("metric_harm", self.metric_harm))
        object.__setattr__(
            self,
            "downstream_harm",
            _finite("downstream_harm", self.downstream_harm),
        )


@dataclass(frozen=True)
class ClusterValue:
    cluster_id: str
    value: float


@dataclass(frozen=True)
class AlignmentGapResult:
    observation_count: int
    cluster_ids: tuple[str, ...]
    perturbation_ids: tuple[str, ...]
    alpha_grid: tuple[float, ...]
    sst: float
    sse_pooled: float
    sse_separate: float
    r2_pooled: float
    r2_separate: float
    raw_alignment_gap: float
    alignment_gap: float
    pooled_predictions: tuple[float, ...]
    separate_predictions: tuple[float, ...]
    cluster_contributions: tuple[ClusterValue, ...]


@dataclass(frozen=True)
class ClusterPairContribution:
    cluster_id: str
    pair_count: int
    strict_agreement_count: int
    strict_disagreement_count: int
    metric_tie_count: int
    downstream_tie_count: int
    double_tie_count: int
    accuracy: float


@dataclass(frozen=True)
class CrossPerturbationAccuracyResult:
    cluster_ids: tuple[str, ...]
    pair_count: int
    strict_agreement_count: int
    strict_disagreement_count: int
    metric_tie_count: int
    downstream_tie_count: int
    double_tie_count: int
    accuracy: float
    cluster_contributions: tuple[ClusterPairContribution, ...]


@dataclass(frozen=True)
class AlignmentComparison:
    reference_gap: AlignmentGapResult
    candidate_gap: AlignmentGapResult
    reference_accuracy: CrossPerturbationAccuracyResult
    candidate_accuracy: CrossPerturbationAccuracyResult
    d_ag: float
    d_acc: float
    ag_contribution_differences: tuple[ClusterValue, ...]
    acc_contribution_differences: tuple[ClusterValue, ...]


__all__ = [
    "AlignmentComparison",
    "AlignmentGapResult",
    "AlignmentObservation",
    "AlignmentValidationError",
    "ClusterPairContribution",
    "ClusterValue",
    "CrossPerturbationAccuracyResult",
]
