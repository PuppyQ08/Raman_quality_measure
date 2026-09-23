"""Typed contracts for Raman perturbation sweeps."""

from rpe.perturb.baseline_distortion import (
    AXIS_BEHAVIOR as P8_AXIS_BEHAVIOR,
    BASELINE_MODEL,
    BaselineDistortionError,
    P8BaselineDistortionState,
    P8LowOrderBaselineDistortion,
)
from rpe.perturb.baseline_residual import (
    BaselineReference,
    BaselineResidualError,
    BaselineResidualState,
    P6BaselineUndercorrection,
    P7BaselineOvercorrection,
)
from rpe.perturb.axis_transform import (
    P11AxisTransformError,
    P11AxisTransformState,
    P11GlobalWavenumberShift,
    P12AxisTransformError,
    P12QuadraticWavenumberWarp,
)
from rpe.perturb.contracts import (
    AxisBehavior,
    Perturbation,
    PerturbationContext,
    PerturbationContractError,
    PerturbationResult,
    PerturbationState,
    derive_perturbed_spectrum_id,
    validate_perturbation_result,
)
from rpe.perturb.correlated_noise import (
    CorrelatedNoiseError,
    P10CorrelatedNoise,
    P10CorrelatedNoiseState,
)
from rpe.perturb.peak_family import (
    PeakFamilyError,
    PeakFamilyPreparedState,
    P1GlobalPeakAttenuation,
    P2SelectiveWeakPeakAttenuation,
    P3PeakBroadening,
    P4WeakPeakDeletion,
    P5FalsePeakInsertion,
)
from rpe.perturb.gaussian_noise import (
    GaussianNoiseError,
    P9GaussianNoiseState,
    P9GaussianWhiteNoise,
)
from rpe.perturb.sweep import (
    PerturbationSweepConfig,
    PerturbationSweepConfigError,
    SeedDerivationConfig,
    derive_state_seed_material,
    load_perturbation_sweep_config,
)


__all__ = [
    "AxisBehavior",
    "BASELINE_MODEL",
    "BaselineDistortionError",
    "BaselineReference",
    "BaselineResidualError",
    "BaselineResidualState",
    "CorrelatedNoiseError",
    "GaussianNoiseError",
    "P8_AXIS_BEHAVIOR",
    "P8BaselineDistortionState",
    "P8LowOrderBaselineDistortion",
    "P10CorrelatedNoise",
    "P10CorrelatedNoiseState",
    "P11AxisTransformError",
    "P11AxisTransformState",
    "P11GlobalWavenumberShift",
    "P12AxisTransformError",
    "P12QuadraticWavenumberWarp",
    "Perturbation",
    "PerturbationContext",
    "PerturbationContractError",
    "PerturbationResult",
    "PerturbationState",
    "PerturbationSweepConfig",
    "PerturbationSweepConfigError",
    "P9GaussianNoiseState",
    "P9GaussianWhiteNoise",
    "P1GlobalPeakAttenuation",
    "P2SelectiveWeakPeakAttenuation",
    "P3PeakBroadening",
    "P4WeakPeakDeletion",
    "P5FalsePeakInsertion",
    "P6BaselineUndercorrection",
    "P7BaselineOvercorrection",
    "SeedDerivationConfig",
    "PeakFamilyError",
    "PeakFamilyPreparedState",
    "derive_perturbed_spectrum_id",
    "derive_state_seed_material",
    "load_perturbation_sweep_config",
    "validate_perturbation_result",
]
