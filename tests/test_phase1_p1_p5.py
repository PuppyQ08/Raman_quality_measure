from __future__ import annotations

import hashlib
import sys
import unittest
from dataclasses import fields, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.perturb import (  # noqa: E402
    P1GlobalPeakAttenuation,
    P2SelectiveWeakPeakAttenuation,
    P3PeakBroadening,
    P4WeakPeakDeletion,
    P5FalsePeakInsertion,
    PeakFamilyError,
    PeakFamilyPreparedState,
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


def peak_fixture(
    *,
    spectrum_id: str = "two-peak-fixture",
    axis: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
) -> Spectrum1D:
    if axis is None:
        axis = np.linspace(100.0, 219.0, 120, dtype="<f8")
    if intensity is None:
        intensity = np.full(axis.shape, 2.0, dtype="<f8")
        intensity[15:26] = np.linspace(2.0, 8.0, 11, dtype="<f8")
        intensity[26:36] = np.linspace(7.4, 2.0, 10, dtype="<f8")
        intensity[75:86] = np.linspace(2.0, 5.0, 11, dtype="<f8")
        intensity[86:96] = np.linspace(4.7, 2.0, 10, dtype="<f8")
    return Spectrum1D(
        spectrum_id=spectrum_id,
        sample_id="sample-a",
        axis_cm1=np.asarray(axis, dtype="<f8"),
        intensity=np.asarray(intensity, dtype="<f8"),
    )


def constant_fixture() -> Spectrum1D:
    axis = np.linspace(100.0, 119.0, 20, dtype="<f8")
    return Spectrum1D(
        spectrum_id="constant-spectrum",
        sample_id="sample-c",
        axis_cm1=axis,
        intensity=np.full(axis.shape, 3.0, dtype="<f8"),
    )


def short_fixture() -> Spectrum1D:
    return Spectrum1D(
        spectrum_id="short-spectrum",
        sample_id="sample-s",
        axis_cm1=np.array([100.0, 101.0, 102.0, 103.0], dtype="<f8"),
        intensity=np.array([1.0, 2.0, 3.0, 2.0], dtype="<f8"),
    )


def nonfinite_fixture() -> Spectrum1D:
    return Spectrum1D(
        spectrum_id="nonfinite-spectrum",
        sample_id="sample-n",
        axis_cm1=np.array([100.0, 110.0, 120.0, 130.0, 140.0], dtype="<f8"),
        intensity=np.array([1.0, 2.0, np.nan, 2.0, 1.0], dtype="<f8"),
    )


def perturbation_context() -> PerturbationContext:
    return PerturbationContext(
        sweep_id="raman_perturbation_alpha_v1",
        sweep_config_sha256=SWEEP_SHA256,
        global_seed=20260817,
    )


class SharedSweepConfigIdentityTest(unittest.TestCase):
    def test_shared_config_identity_matches_retained_phase1_phase4_file(self) -> None:
        raw = SWEEP_CONFIG.read_bytes()
        self.assertEqual(len(raw), SWEEP_BYTES)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), SWEEP_SHA256)
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(config.sha256, SWEEP_SHA256)
        for perturbation_id in ("p01", "p02", "p03", "p04", "p05"):
            self.assertIn(perturbation_id, config.perturbation_ids)


class SharedPreparedStateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = peak_fixture()
        self.operator = P1GlobalPeakAttenuation(self.config)

    def test_state_dataclass_has_required_public_fields(self) -> None:
        self.assertEqual(
            tuple(field.name for field in fields(PeakFamilyPreparedState)),
            (
                "perturbation_id",
                "spectrum_id",
                "sweep_config_sha256",
                "state_digest",
                "detector_id",
                "source_axis_sha256",
                "source_intensity_sha256",
                "peak_indices",
                "peak_positions_cm1",
                "prominences",
                "support_bounds",
                "component_matrix",
                "component_heights",
                "component_areas",
                "median_fwhm_cm1",
                "candidate_false_centers_cm1",
                "candidate_false_peak",
                "deterministic_random_material",
            ),
        )

    def test_prepare_is_deterministic_and_freezes_two_peak_decomposition(self) -> None:
        state = self.operator.prepare(self.source, self.context)
        repeated = self.operator.prepare(self.source, self.context)

        self.assertEqual(state.perturbation_id, "p01")
        self.assertEqual(state.detector_id, "internal_find_peaks_prominence_5pct_range")
        self.assertEqual(state.state_digest, repeated.state_digest)
        self.assertEqual(state.peak_indices, (25, 85))
        self.assertEqual(state.support_bounds, ((0, 55), (55, 119)))
        np.testing.assert_allclose(
            np.asarray(state.peak_positions_cm1),
            np.array([125.0, 185.0], dtype="<f8"),
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(state.component_matrix.dtype, np.dtype("<f8"))
        self.assertFalse(state.component_matrix.flags.writeable)
        self.assertEqual(state.component_matrix.shape, (2, self.source.axis_cm1.size))
        np.testing.assert_array_equal(
            state.component_matrix,
            repeated.component_matrix,
        )
        self.assertEqual(state.component_heights, (6.0, 3.0))
        self.assertEqual(state.component_areas, (60.0, 30.0))
        self.assertAlmostEqual(state.median_fwhm_cm1, 10.0, places=12)
        self.assertEqual(state.candidate_false_centers_cm1, ())
        self.assertEqual(len(state.candidate_false_peak), self.source.axis_cm1.size)
        self.assertFalse(state.candidate_false_peak.flags.writeable)
        np.testing.assert_array_equal(
            state.candidate_false_peak,
            np.zeros(self.source.axis_cm1.size, dtype="<f8"),
        )

    def test_prepare_rejects_too_short_constant_and_nonfinite_inputs(self) -> None:
        with self.assertRaisesRegex(PeakFamilyError, "at least five points"):
            self.operator.prepare(short_fixture(), self.context)
        with self.assertRaisesRegex(PeakFamilyError, "positive finite intensity range"):
            self.operator.prepare(constant_fixture(), self.context)
        with self.assertRaisesRegex(Exception, "intensity finite|source intensity"):
            self.operator.prepare(nonfinite_fixture(), self.context)


class P1GlobalPeakAttenuationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = peak_fixture()
        self.operator = P1GlobalPeakAttenuation(self.config)
        self.state = self.operator.prepare(self.source, self.context)

    def test_zero_alpha_is_exact_identity_and_positive_alpha_scales_both_peaks(self) -> None:
        zero = self.operator.apply(self.source, 0.0, self.state)
        self.assertFalse(zero.axis_changed)
        self.assertFalse(zero.intensity_changed)
        np.testing.assert_array_equal(zero.output.axis_cm1, self.source.axis_cm1)
        np.testing.assert_array_equal(zero.output.intensity, self.source.intensity)

        result = self.operator.apply(self.source, 0.5, self.state)
        validate_perturbation_result(self.source, self.state, result, self.config)
        self.assertFalse(result.axis_changed)
        self.assertTrue(result.intensity_changed)
        self.assertEqual(result.diagnostics["internal_peak_model"], self.state.detector_id)
        self.assertEqual(result.diagnostics["detected_peak_count"], 2)
        self.assertEqual(result.diagnostics["selected_or_inserted_peak_count"], 2)
        self.assertEqual(result.diagnostics["alpha"], 0.5)
        self.assertEqual(result.diagnostics["source_component_area"], 90.0)
        self.assertEqual(result.diagnostics["removed_component_area"], 45.0)
        self.assertTrue(result.diagnostics["axis_preserved"])
        self.assertEqual(result.output.intensity[25], 5.0)
        self.assertEqual(result.output.intensity[85], 3.5)

    def test_apply_accepts_content_identical_reconstruction_and_rejects_drift(self) -> None:
        rebuilt = peak_fixture()
        result = self.operator.apply(rebuilt, 0.05, self.state)
        validate_perturbation_result(rebuilt, self.state, result, self.config)

        drifted = peak_fixture(intensity=peak_fixture().intensity + np.eye(1, 120, 25, dtype="<f8")[0])
        with self.assertRaisesRegex(PeakFamilyError, "source intensity"):
            self.operator.apply(drifted, 0.05, self.state)


class P2SelectiveWeakPeakAttenuationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = peak_fixture()
        self.operator = P2SelectiveWeakPeakAttenuation(self.config)
        self.state = self.operator.prepare(self.source, self.context)

    def test_weak_peak_is_selected_first_and_count_is_monotone(self) -> None:
        weak_first = self.operator.apply(self.source, 0.05, self.state)
        self.assertEqual(weak_first.diagnostics["selected_or_inserted_peak_count"], 1)
        self.assertEqual(weak_first.output.intensity[25], 8.0)
        self.assertEqual(weak_first.output.intensity[85], 3.5)
        self.assertEqual(weak_first.diagnostics["removed_component_area"], 15.0)

        both = self.operator.apply(self.source, 0.8, self.state)
        validate_perturbation_result(self.source, self.state, both, self.config)
        self.assertEqual(both.diagnostics["selected_or_inserted_peak_count"], 2)
        self.assertEqual(both.output.intensity[25], 5.0)
        self.assertEqual(both.output.intensity[85], 3.5)
        self.assertEqual(both.diagnostics["removed_component_area"], 45.0)


class P3PeakBroadeningTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = peak_fixture()
        self.operator = P3PeakBroadening(self.config)
        self.state = self.operator.prepare(self.source, self.context)

    def test_broadening_uses_median_fwhm_area_conservation_and_monotone_sigma(self) -> None:
        previous_sigma = 0.0
        for alpha in self.config.alpha_grid:
            result = self.operator.apply(self.source, alpha, self.state)
            validate_perturbation_result(self.source, self.state, result, self.config)
            sigma = float(result.diagnostics["broadened_sigma_cm1"])
            if alpha == 0.0:
                self.assertFalse(result.intensity_changed)
                self.assertEqual(sigma, 0.0)
                continue
            self.assertGreater(sigma, previous_sigma)
            previous_sigma = sigma
            self.assertAlmostEqual(
                float(result.diagnostics["source_component_area"]),
                float(result.diagnostics["inserted_component_area"]),
                places=12,
            )

        half = self.operator.apply(self.source, 0.5, self.state)
        self.assertEqual(half.diagnostics["detected_peak_count"], 2)
        self.assertEqual(half.diagnostics["selected_or_inserted_peak_count"], 2)
        self.assertAlmostEqual(float(half.diagnostics["broadened_sigma_cm1"]), 5.0, places=12)
        self.assertLess(half.output.intensity[25], self.source.intensity[25])
        self.assertLess(half.output.intensity[85], self.source.intensity[85])


class P4WeakPeakDeletionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = peak_fixture()
        self.operator = P4WeakPeakDeletion(self.config)
        self.state = self.operator.prepare(self.source, self.context)

    def test_deletes_weak_then_strong_peak_by_monotone_count(self) -> None:
        weak_deleted = self.operator.apply(self.source, 0.05, self.state)
        self.assertEqual(weak_deleted.diagnostics["selected_or_inserted_peak_count"], 1)
        self.assertEqual(weak_deleted.output.intensity[25], 8.0)
        self.assertEqual(weak_deleted.output.intensity[85], 2.0)
        self.assertEqual(weak_deleted.diagnostics["removed_component_area"], 30.0)

        both_deleted = self.operator.apply(self.source, 0.8, self.state)
        validate_perturbation_result(self.source, self.state, both_deleted, self.config)
        self.assertEqual(both_deleted.diagnostics["selected_or_inserted_peak_count"], 2)
        self.assertEqual(both_deleted.output.intensity[25], 2.0)
        self.assertEqual(both_deleted.output.intensity[85], 2.0)
        self.assertEqual(both_deleted.diagnostics["removed_component_area"], 90.0)


class P5FalsePeakInsertionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = peak_fixture()
        self.operator = P5FalsePeakInsertion(self.config)
        self.state = self.operator.prepare(self.source, self.context)

    def test_inserted_count_is_floor_alpha_times_peak_count_and_sets_are_nested(self) -> None:
        none = self.operator.apply(self.source, 0.4, self.state)
        self.assertEqual(none.diagnostics["selected_or_inserted_peak_count"], 0)
        np.testing.assert_array_equal(none.output.intensity, self.source.intensity)

        one = self.operator.apply(self.source, 0.5, self.state)
        self.assertEqual(one.diagnostics["selected_or_inserted_peak_count"], 1)
        self.assertEqual(one.diagnostics["inserted_peak_centers_cm1"], self.state.candidate_false_centers_cm1[:1])
        self.assertGreater(one.diagnostics["inserted_component_area"], 0.0)
        self.assertGreater(float(one.output.intensity.max()), float(self.source.intensity.max()))

        two = self.operator.apply(self.source, 0.8, self.state)
        validate_perturbation_result(self.source, self.state, two, self.config)
        self.assertEqual(two.diagnostics["selected_or_inserted_peak_count"], 1)
        self.assertEqual(two.diagnostics["inserted_peak_centers_cm1"], self.state.candidate_false_centers_cm1[:1])
        self.assertEqual(two.diagnostics["false_peak_height"], 4.5)
        self.assertEqual(two.diagnostics["false_peak_fwhm_cm1"], 10.0)


class SharedFailurePathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.context = perturbation_context()
        self.source = peak_fixture()

    def test_wrong_context_alpha_state_and_source_are_rejected(self) -> None:
        operator = P4WeakPeakDeletion(self.config)
        with self.assertRaisesRegex(PeakFamilyError, "context.sweep_id"):
            operator.prepare(self.source, replace(self.context, sweep_id="wrong"))

        state = operator.prepare(self.source, self.context)
        with self.assertRaisesRegex(PeakFamilyError, "alpha"):
            operator.apply(self.source, 0.123, state)
        with self.assertRaisesRegex(PeakFamilyError, "state.sweep_config_sha256"):
            operator.apply(
                self.source,
                0.05,
                replace(state, sweep_config_sha256="0" * 64),
            )
        with self.assertRaisesRegex(PeakFamilyError, "state.state_digest"):
            operator.apply(
                self.source,
                0.05,
                replace(state, state_digest="0" * 64),
            )
        with self.assertRaisesRegex(PeakFamilyError, "source spectrum_id"):
            operator.apply(
                peak_fixture(spectrum_id="other-spectrum"),
                0.05,
                state,
            )

    def test_positive_alpha_that_rounds_to_identity_fails_closed(self) -> None:
        axis = np.linspace(100.0, 219.0, 120, dtype="<f8")
        baseline = np.full(axis.shape, float(2**53), dtype="<f8")
        intensity = baseline.copy()
        intensity[15:26] = baseline[15:26] + np.linspace(0.0, 12.0, 11, dtype="<f8")
        intensity[26:36] = baseline[26:36] + np.linspace(10.8, 0.0, 10, dtype="<f8")
        intensity[75:86] = baseline[75:86] + np.linspace(0.0, 6.0, 11, dtype="<f8")
        intensity[86:96] = baseline[86:96] + np.linspace(5.4, 0.0, 10, dtype="<f8")
        huge = peak_fixture(
            spectrum_id="huge-spectrum",
            axis=axis,
            intensity=intensity,
        )
        operator = P1GlobalPeakAttenuation(self.config)
        state = operator.prepare(huge, self.context)
        with self.assertRaisesRegex(PeakFamilyError, "rounds entirely to identity"):
            operator.apply(huge, 0.05, state)

    def test_false_peak_placement_is_p5_only_and_does_not_block_p1_p4(self) -> None:
        axis = np.linspace(100.0, 200.0, 201, dtype="<f8")
        intensity = np.asarray(
            1.0
            + 10.0 * np.exp(-0.5 * ((axis - 130.0) / 8.0) ** 2)
            + 8.0 * np.exp(-0.5 * ((axis - 170.0) / 8.0) ** 2),
            dtype="<f8",
        )
        crowded = peak_fixture(
            spectrum_id="crowded-false-peak-domain",
            axis=axis,
            intensity=intensity,
        )

        for operator_class in (
            P1GlobalPeakAttenuation,
            P2SelectiveWeakPeakAttenuation,
            P3PeakBroadening,
            P4WeakPeakDeletion,
        ):
            with self.subTest(operator=operator_class.__name__):
                state = operator_class(self.config).prepare(crowded, self.context)
                self.assertEqual(state.candidate_false_centers_cm1, ())
                np.testing.assert_array_equal(
                    state.candidate_false_peak,
                    np.zeros(axis.size, dtype="<f8"),
                )

        with self.assertRaisesRegex(PeakFamilyError, "false peak candidates"):
            P5FalsePeakInsertion(self.config).prepare(crowded, self.context)

    def test_public_imports_exist_for_all_peak_family_operators(self) -> None:
        config = load_perturbation_sweep_config(SWEEP_CONFIG)
        self.assertEqual(P1GlobalPeakAttenuation(config).perturbation_id, "p01")
        self.assertEqual(P2SelectiveWeakPeakAttenuation(config).perturbation_id, "p02")
        self.assertEqual(P3PeakBroadening(config).perturbation_id, "p03")
        self.assertEqual(P4WeakPeakDeletion(config).perturbation_id, "p04")
        self.assertEqual(P5FalsePeakInsertion(config).perturbation_id, "p05")


if __name__ == "__main__":
    unittest.main(verbosity=2)
