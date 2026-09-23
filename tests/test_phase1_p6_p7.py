from __future__ import annotations

import hashlib
import math
import sys
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.perturb import (  # noqa: E402
    BaselineReference,
    BaselineResidualError,
    BaselineResidualState,
    P6BaselineUndercorrection,
    P7BaselineOvercorrection,
    PerturbationContext,
    load_perturbation_sweep_config,
    validate_perturbation_result,
)


SWEEP_CONFIG = (
    ROOT / "experiments" / "shared" / "raman_perturbation_sweep_v1.json"
)
SWEEP_BYTES = 559
SWEEP_SHA256 = (
    "b32e75ffe0d124a2aec80bbae23624f01ca15bfed75184401af7a2e26d7f2186"
)


def source_spectrum(
    *,
    spectrum_id: str = "clean-spectrum",
    axis: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0),
    intensity: tuple[float, ...] = (10.0, 20.0, 30.0, 40.0),
) -> Spectrum1D:
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.array(axis, dtype="<f8"),
        intensity=np.array(intensity, dtype="<f8"),
    )


def baseline_reference(
    *,
    spectrum_id: str = "clean-spectrum",
    provenance_id: str = "semisynth-baseline-fixture-v1",
    axis: tuple[float, ...] = (100.0, 200.0, 300.0, 400.0),
    baseline: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0),
) -> BaselineReference:
    return BaselineReference(
        spectrum_id=spectrum_id,
        provenance_id=provenance_id,
        axis_cm1=np.array(axis, dtype="<f8"),
        baseline_intensity=np.array(baseline, dtype="<f8"),
    )


def perturbation_context() -> PerturbationContext:
    return PerturbationContext(
        sweep_id="raman_perturbation_alpha_v1",
        sweep_config_sha256=SWEEP_SHA256,
        global_seed=20260817,
    )


class SharedSweepAndReferenceTest(unittest.TestCase):
    def test_shared_config_identity_contains_p06_and_p07(self) -> None:
        raw = SWEEP_CONFIG.read_bytes()
        self.assertEqual(len(raw), SWEEP_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SWEEP_SHA256)
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertIn("p06", config.perturbation_ids)
        self.assertIn("p07", config.perturbation_ids)

    def test_reference_copies_arrays_and_is_transitively_immutable(self) -> None:
        axis = np.array([100.0, 200.0, 300.0, 400.0], dtype="<f8")
        baseline = np.array([2.0, 4.0, 6.0, 8.0], dtype="<f8")
        reference = BaselineReference(
            spectrum_id="clean-spectrum",
            provenance_id="paired-baseline-v1",
            axis_cm1=axis,
            baseline_intensity=baseline,
        )
        axis[0] = 999.0
        baseline[0] = 999.0

        np.testing.assert_array_equal(
            reference.axis_cm1,
            np.array([100.0, 200.0, 300.0, 400.0], dtype="<f8"),
        )
        np.testing.assert_array_equal(
            reference.baseline_intensity,
            np.array([2.0, 4.0, 6.0, 8.0], dtype="<f8"),
        )
        self.assertFalse(reference.axis_cm1.flags.writeable)
        self.assertFalse(reference.baseline_intensity.flags.writeable)
        with self.assertRaises(ValueError):
            reference.baseline_intensity[0] = 0.0
        with self.assertRaises(FrozenInstanceError):
            reference.provenance_id = "changed"

    def test_reference_rejects_bad_ids_arrays_axis_and_zero_baseline(self) -> None:
        with self.assertRaisesRegex(BaselineResidualError, "spectrum_id"):
            baseline_reference(spectrum_id="")
        with self.assertRaisesRegex(BaselineResidualError, "provenance_id"):
            baseline_reference(provenance_id="")
        with self.assertRaisesRegex(BaselineResidualError, "array length|shape"):
            baseline_reference(baseline=(2.0, 4.0, 6.0))
        with self.assertRaisesRegex(BaselineResidualError, "strictly increasing"):
            baseline_reference(axis=(100.0, 200.0, 200.0, 400.0))
        with self.assertRaisesRegex(BaselineResidualError, "positive.*RMS|RMS.*positive"):
            baseline_reference(baseline=(0.0, 0.0, 0.0, 0.0))
        with self.assertRaisesRegex(BaselineResidualError, "finite"):
            baseline_reference(baseline=(2.0, 4.0, math.nan, 8.0))
        with self.assertRaisesRegex(BaselineResidualError, "dtype"):
            BaselineReference(
                spectrum_id="clean-spectrum",
                provenance_id="wrong-dtype",
                axis_cm1=np.array([100.0, 200.0], dtype="<f4"),
                baseline_intensity=np.array([1.0, 2.0], dtype="<f8"),
            )


class BaselineResidualOperatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = source_spectrum()
        self.reference = baseline_reference()
        self.p6 = P6BaselineUndercorrection(self.config, self.reference)
        self.p7 = P7BaselineOvercorrection(self.config, self.reference)

    def test_prepare_rejects_baseline_source_id_or_axis_mismatch(self) -> None:
        wrong_id = P6BaselineUndercorrection(
            self.config,
            baseline_reference(spectrum_id="other-spectrum"),
        )
        with self.assertRaisesRegex(BaselineResidualError, "baseline spectrum_id"):
            wrong_id.prepare(self.source, self.context)

        wrong_axis = P7BaselineOvercorrection(
            self.config,
            baseline_reference(axis=(100.0, 200.0, 310.0, 400.0)),
        )
        with self.assertRaisesRegex(BaselineResidualError, "baseline axis"):
            wrong_axis.prepare(self.source, self.context)

    def test_prepare_is_deterministic_content_bound_and_state_is_immutable(self) -> None:
        state = self.p6.prepare(self.source, self.context)
        repeated = self.p6.prepare(source_spectrum(), self.context)

        self.assertIsInstance(state, BaselineResidualState)
        self.assertEqual(state, repeated)
        self.assertEqual(state.perturbation_id, "p06")
        self.assertEqual(state.spectrum_id, "clean-spectrum")
        self.assertEqual(state.baseline_provenance_id, "semisynth-baseline-fixture-v1")
        self.assertEqual(state.baseline_rms, math.sqrt(30.0))
        self.assertFalse(state.baseline_intensity.flags.writeable)
        np.testing.assert_array_equal(state.baseline_intensity, self.reference.baseline_intensity)
        with self.assertRaises(ValueError):
            state.baseline_intensity[0] = 0.0
        with self.assertRaises(FrozenInstanceError):
            state.baseline_rms = 0.0

    def test_literal_half_alpha_outputs_and_diagnostics_match_oracle(self) -> None:
        p6_state = self.p6.prepare(self.source, self.context)
        p7_state = self.p7.prepare(self.source, self.context)
        under = self.p6.apply(self.source, 0.5, p6_state)
        over = self.p7.apply(self.source, 0.5, p7_state)

        np.testing.assert_array_equal(
            under.output.intensity,
            np.array([11.0, 22.0, 33.0, 44.0], dtype="<f8"),
        )
        np.testing.assert_array_equal(
            over.output.intensity,
            np.array([9.0, 18.0, 27.0, 36.0], dtype="<f8"),
        )
        for result, state, sign in ((under, p6_state, 1), (over, p7_state, -1)):
            validate_perturbation_result(self.source, state, result, self.config)
            np.testing.assert_array_equal(result.output.axis_cm1, self.source.axis_cm1)
            self.assertFalse(result.axis_changed)
            self.assertTrue(result.intensity_changed)
            self.assertEqual(result.diagnostics["baseline_provenance_id"], self.reference.provenance_id)
            self.assertEqual(result.diagnostics["baseline_sha256"], state.baseline_sha256)
            self.assertEqual(result.diagnostics["baseline_rms"], math.sqrt(30.0))
            self.assertEqual(result.diagnostics["residual_sign"], sign)
            self.assertEqual(result.diagnostics["requested_alpha"], 0.5)
            self.assertAlmostEqual(
                float(result.diagnostics["realized_residual_rmse"]),
                0.5 * math.sqrt(30.0),
                places=14,
            )
            self.assertTrue(result.diagnostics["axis_preserved"])
            self.assertEqual(
                result.diagnostics["baseline_source"],
                "explicit_external_reference",
            )

    def test_alpha_zero_is_exact_identity_and_does_not_mutate_inputs(self) -> None:
        source_axis_before = self.source.axis_cm1.copy()
        source_intensity_before = self.source.intensity.copy()
        baseline_before = self.reference.baseline_intensity.copy()

        for operator in (self.p6, self.p7):
            state = operator.prepare(self.source, self.context)
            result = operator.apply(self.source, 0.0, state)
            self.assertFalse(result.axis_changed)
            self.assertFalse(result.intensity_changed)
            np.testing.assert_array_equal(result.output.axis_cm1, self.source.axis_cm1)
            np.testing.assert_array_equal(result.output.intensity, self.source.intensity)
            self.assertIsNot(result.output.axis_cm1, self.source.axis_cm1)
            self.assertIsNot(result.output.intensity, self.source.intensity)
            self.assertEqual(result.diagnostics["realized_residual_rmse"], 0.0)

        np.testing.assert_array_equal(self.source.axis_cm1, source_axis_before)
        np.testing.assert_array_equal(self.source.intensity, source_intensity_before)
        np.testing.assert_array_equal(self.reference.baseline_intensity, baseline_before)

    def test_positive_grid_has_strictly_monotone_rmse_and_p6_p7_symmetry(self) -> None:
        p6_state = self.p6.prepare(self.source, self.context)
        p7_state = self.p7.prepare(self.source, self.context)
        realized: list[float] = []
        for alpha in self.config.alpha_grid:
            if alpha == 0.0:
                continue
            under = self.p6.apply(self.source, alpha, p6_state)
            over = self.p7.apply(self.source, alpha, p7_state)
            under_rmse = float(under.diagnostics["realized_residual_rmse"])
            over_rmse = float(over.diagnostics["realized_residual_rmse"])
            self.assertAlmostEqual(under_rmse, alpha * math.sqrt(30.0), places=14)
            self.assertAlmostEqual(over_rmse, alpha * math.sqrt(30.0), places=14)
            self.assertAlmostEqual(under_rmse, over_rmse, places=14)
            np.testing.assert_array_equal(
                under.output.intensity + over.output.intensity,
                2.0 * self.source.intensity,
            )
            realized.append(under_rmse)
        self.assertTrue(all(right > left for left, right in zip(realized, realized[1:])))

    def test_apply_accepts_rebuilt_source_and_rejects_source_or_state_drift(self) -> None:
        state = self.p6.prepare(self.source, self.context)
        rebuilt = source_spectrum()
        result = self.p6.apply(rebuilt, 0.1, state)
        validate_perturbation_result(rebuilt, state, result, self.config)

        with self.assertRaisesRegex(BaselineResidualError, "source intensity"):
            self.p6.apply(
                source_spectrum(intensity=(10.0, 20.0, 30.0, 41.0)),
                0.1,
                state,
            )
        with self.assertRaisesRegex(BaselineResidualError, "state.state_digest"):
            self.p6.apply(rebuilt, 0.1, replace(state, state_digest="0" * 64))
        with self.assertRaisesRegex(BaselineResidualError, "state.*baseline|baseline.*state"):
            self.p6.apply(
                rebuilt,
                0.1,
                replace(
                    state,
                    baseline_intensity=np.array([2.0, 4.0, 6.0, 9.0], dtype="<f8"),
                ),
            )

    def test_state_cannot_cross_operator_or_reference_provenance(self) -> None:
        p6_state = self.p6.prepare(self.source, self.context)
        with self.assertRaisesRegex(BaselineResidualError, "state.perturbation_id"):
            self.p7.apply(self.source, 0.1, p6_state)

        other_provenance = P6BaselineUndercorrection(
            self.config,
            baseline_reference(provenance_id="other-provenance"),
        )
        with self.assertRaisesRegex(BaselineResidualError, "baseline_provenance_id"):
            other_provenance.apply(self.source, 0.1, p6_state)

        other_content = P6BaselineUndercorrection(
            self.config,
            baseline_reference(baseline=(2.0, 4.0, 6.0, 10.0)),
        )
        with self.assertRaisesRegex(BaselineResidualError, "baseline_sha256"):
            other_content.apply(self.source, 0.1, p6_state)

    def test_wrong_context_config_and_alpha_are_rejected(self) -> None:
        with self.assertRaisesRegex(BaselineResidualError, "context.sweep_id"):
            self.p6.prepare(
                self.source,
                replace(self.context, sweep_id="wrong"),
            )
        bad_config = replace(self.config, global_seed=self.config.global_seed + 1)
        with self.assertRaisesRegex(BaselineResidualError, "config.global_seed"):
            P7BaselineOvercorrection(bad_config, self.reference)

        state = self.p6.prepare(self.source, self.context)
        with self.assertRaisesRegex(BaselineResidualError, "alpha"):
            self.p6.apply(self.source, 0.123, state)

    def test_positive_round_to_identity_underflow_and_overflow_fail_closed(self) -> None:
        huge_source = source_spectrum(intensity=(2.0**53,) * 4)
        unit_reference = baseline_reference(baseline=(1.0,) * 4)
        rounded = P6BaselineUndercorrection(self.config, unit_reference)
        rounded_state = rounded.prepare(huge_source, self.context)
        with self.assertRaisesRegex(BaselineResidualError, "rounds entirely to identity"):
            rounded.apply(huge_source, 0.05, rounded_state)

        minimum = float(np.nextafter(0.0, 1.0))
        zero_source = source_spectrum(intensity=(0.0,) * 4)
        tiny_reference = baseline_reference(baseline=(minimum,) * 4)
        underflow = P7BaselineOvercorrection(self.config, tiny_reference)
        underflow_state = underflow.prepare(zero_source, self.context)
        with self.assertRaisesRegex(BaselineResidualError, "rounds entirely to identity"):
            underflow.apply(zero_source, 0.05, underflow_state)

        max_source = source_spectrum(intensity=(1.0e308,) * 4)
        max_reference = baseline_reference(baseline=(1.0e308,) * 4)
        overflow = P6BaselineUndercorrection(self.config, max_reference)
        overflow_state = overflow.prepare(max_source, self.context)
        with self.assertRaisesRegex(BaselineResidualError, "finite output"):
            overflow.apply(max_source, 0.8, overflow_state)

    def test_public_imports_exist_for_both_operators(self) -> None:
        self.assertEqual(self.p6.perturbation_id, "p06")
        self.assertEqual(self.p7.perturbation_id, "p07")


if __name__ == "__main__":
    unittest.main(verbosity=2)
