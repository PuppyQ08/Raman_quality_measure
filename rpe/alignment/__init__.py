"""Pure preregistered Phase 4 alignment statistics."""

from rpe.alignment.contracts import (
    AlignmentComparison,
    AlignmentGapResult,
    AlignmentObservation,
    AlignmentValidationError,
    ClusterPairContribution,
    ClusterValue,
    CrossPerturbationAccuracyResult,
)
from rpe.alignment.core import (
    alignment_gap,
    compare_alignment,
    cross_perturbation_accuracy,
    orient_harm,
)
from rpe.alignment.inference import (
    AlignmentBootstrapResult,
    HolmResult,
    SignFlipResult,
    holm_step_down,
    paired_cluster_bootstrap,
    paired_contribution_sign_flip,
)
from rpe.alignment.bulk import bulk_paired_cluster_bootstrap

__all__ = [
    "AlignmentComparison",
    "AlignmentBootstrapResult",
    "AlignmentGapResult",
    "AlignmentObservation",
    "AlignmentValidationError",
    "ClusterPairContribution",
    "ClusterValue",
    "CrossPerturbationAccuracyResult",
    "HolmResult",
    "SignFlipResult",
    "alignment_gap",
    "bulk_paired_cluster_bootstrap",
    "compare_alignment",
    "cross_perturbation_accuracy",
    "holm_step_down",
    "orient_harm",
    "paired_cluster_bootstrap",
    "paired_contribution_sign_flip",
]
