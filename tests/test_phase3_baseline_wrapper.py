from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.methods import load_classical_catalog  # noqa: E402
from rpe.methods.classical import (  # noqa: E402
    BaselineCanaryError,
    BaselineCanarySource,
    BaselineRunStatus,
    build_baseline_canary_from_sources,
    load_rruff_baseline_canary_sources,
    run_baseline_system,
    verify_baseline_canary_artifact,
)


CATALOG_PATH = ROOT / "experiments" / "phase3" / "configs" / "classical_system_catalog_v1.json"
RRUFF_DATASET = ROOT / "data" / "unified" / "rruff_raman_raw"
PHASE1_RUN = (
    ROOT
    / "data"
    / "perturbed"
    / "phase1_rruff_core10k"
    / "phase1-rruff-core10k-ca17c81ec5ff89e5cbb9170e2945b5a2a4f10d242f268abef41aae062ad549f4"
)


def oracle_spectrum() -> Spectrum1D:
    axis = np.linspace(100.0, 400.0, 32, dtype="<f8")
    intensity = (
        0.01 * axis
        + 3.0
        + 20.0 * np.exp(-0.5 * ((axis - 250.0) / 12.0) ** 2)
    ).astype("<f8")
    return Spectrum1D(
        spectrum_id="baseline-wrapper-oracle",
        sample_id="oracle",
        axis_cm1=axis,
        intensity=intensity,
    )


def baseline_system(family: str, expected: dict[str, object]):
    catalog = load_classical_catalog(CATALOG_PATH)
    matches = [
        system
        for system in catalog.systems
        if system.task_line.value == "baseline_correction"
        and system.family_id == family
        and all(system.hyperparameters.get(key) == value for key, value in expected.items())
    ]
    if len(matches) != 1:
        raise AssertionError((family, expected, len(matches)))
    return matches[0]


class BaselineWrapperTest(unittest.TestCase):
    def test_all_fourteen_families_have_literal_output_oracles(self) -> None:
        cases = (
            ("asls", {"lam": 1e6, "p": 0.01}, "df3c58578191be2207ed562207bf7a8ae7a71cccc32cc1745cc47a161ccb96a3", BaselineRunStatus.COMPLETE),
            ("iasls", {"lam": 1e6, "p": 0.01}, "f34911265cd397fe975060d97178d8b388178fd400163c6c1a3763fb16286824", BaselineRunStatus.COMPLETE),
            ("airpls", {"lam": 1e6}, "262dae5c17a39073f551c6b56855c51e40fe1de09f549ad464ecd82c95265d42", BaselineRunStatus.COMPLETE),
            ("arpls", {"lam": 1e5}, "f44e288fa6e690159f0b0ebb3d92a1f97abf62ffd23676f16ddd7cc3857c93e8", BaselineRunStatus.COMPLETE),
            ("drpls", {"lam": 1e5, "eta": 0.5}, "949dbfc07b633ab81da7ab376226e45adf2813fa5b36578f543207160609aa2c", BaselineRunStatus.COMPLETE),
            ("iarpls", {"lam": 1e5}, "2600eef0efefd3658ebe9eadf22f9c9e9c4615a144bbd9837e70355560a3b2d2", BaselineRunStatus.FAILED_CONVERGENCE),
            ("aspls", {"lam": 1e5, "asymmetric_coef": 0.5}, "cb385f86197346023c3471675250c8b66dc6519aaddd949d59c6f1d00d5994d4", BaselineRunStatus.COMPLETE),
            ("psalsa", {"lam": 1e5, "p": 0.5}, "c703623bcfb86947a89b1d1f9fddd11bb8947113af83907548a2093126d4a29a", BaselineRunStatus.COMPLETE),
            ("modpoly", {"poly_order": 2, "use_original": False, "mask_initial_peaks": False}, "90def458981efdfdba5bf947898c4a825ded2ae218bc749daa949b7b388d08b7", BaselineRunStatus.COMPLETE),
            ("imodpoly", {"poly_order": 2, "num_std": 1.0}, "ed3f90ed640fb55505ea54c3c9ee34dcea0b3b271ffc8013a022d8791142f2ac", BaselineRunStatus.COMPLETE),
            ("penalized_poly", {"poly_order": 2, "cost_function": "asymmetric_truncated_quadratic"}, "bdeae6c74f31c57dd975ccce9eaf0a989b3fdb3b59fb211cdc7a7b401d330060", BaselineRunStatus.COMPLETE),
            ("snip", {"max_half_window_cm1": 20.0, "filter_order": 2}, "ce265ac2b445043fee6a53bb2079aafddf6ff0d65565e6b8e17ed7ba58141d49", BaselineRunStatus.COMPLETE),
            ("morphological", {"half_window_cm1": 24.0}, "82112fec7f06901a1a94db833e9ee62a2135609a6874de1886230ddedf1c341a", BaselineRunStatus.COMPLETE),
            ("beads", {"freq_cutoff": 0.005, "asymmetry": 6.0}, "bcb169b8afa39c6bcca8e2e6883499092541286303dca648ea39c58391ee1dac", BaselineRunStatus.COMPLETE),
        )
        spectrum = oracle_spectrum()
        for family, selector, expected_sha, expected_status in cases:
            with self.subTest(family=family):
                result = run_baseline_system(baseline_system(family, selector), spectrum)
                self.assertIs(result.status, expected_status)
                self.assertEqual(result.baseline_sha256, expected_sha)
                self.assertIsNotNone(result.baseline_estimate)
                self.assertIsNotNone(result.corrected_intensity)
                assert result.baseline_estimate is not None
                assert result.corrected_intensity is not None
                np.testing.assert_array_equal(
                    result.corrected_intensity,
                    spectrum.intensity - result.baseline_estimate,
                )
                self.assertFalse(result.baseline_estimate.flags.writeable)
                self.assertFalse(result.corrected_intensity.flags.writeable)
                self.assertRegex(result.corrected_sha256 or "", r"^[0-9a-f]{64}$")
        iarpls = run_baseline_system(baseline_system("iarpls", {"lam": 1e5}), spectrum)
        self.assertEqual(tuple(warning.category for warning in iarpls.warnings), ("ParameterWarning",))
        self.assertGreater(iarpls.diagnostics["tol_last"], 0.001)
        beads = run_baseline_system(baseline_system("beads", {"freq_cutoff": 0.005, "asymmetry": 6.0}), spectrum)
        self.assertRegex(beads.diagnostics["signal_sha256"], r"^[0-9a-f]{64}$")

    def test_physical_window_outside_axis_domain_is_not_applicable(self) -> None:
        spectrum = Spectrum1D(
            spectrum_id="coarse-axis",
            sample_id=None,
            axis_cm1=np.array([0.0, 1000.0, 2000.0, 3000.0], dtype="<f8"),
            intensity=np.array([1.0, 2.0, 3.0, 4.0], dtype="<f8"),
        )
        mor = run_baseline_system(
            baseline_system("morphological", {"half_window_cm1": 8.0}),
            spectrum,
        )
        self.assertIs(mor.status, BaselineRunStatus.NOT_APPLICABLE)
        self.assertEqual(mor.error_code, "converted_half_window_out_of_domain")
        self.assertIsNone(mor.baseline_estimate)

    def test_unexpected_backend_exception_is_failed_runtime(self) -> None:
        system = baseline_system("asls", {"lam": 1e6, "p": 0.01})
        with patch(
            "rpe.methods.classical.baseline.Baseline.asls",
            side_effect=RuntimeError("injected backend failure"),
        ):
            result = run_baseline_system(system, oracle_spectrum())
        self.assertIs(result.status, BaselineRunStatus.FAILED_RUNTIME)
        self.assertEqual(result.error_code, "RuntimeError")
        self.assertIsNone(result.baseline_estimate)


class BaselineCanaryTest(unittest.TestCase):
    def test_rruff_source_selection_uses_four_literal_frozen_band_winners(self) -> None:
        sources = load_rruff_baseline_canary_sources(PHASE1_RUN, RRUFF_DATASET)
        self.assertEqual(
            tuple(
                (source.band_id, source.selection_rank, source.record_id, source.point_count)
                for source in sources
            ),
            (
                ("b1_351_1726", 0, "raw-0003d1b556e9bc8434a8a7972d5f", 1331),
                ("b2_1727_2326", 10, "raw-002525ef577ac9ed746f80d1bfea", 2313),
                ("b3_2327_3395", 2, "raw-0005cc5ef4a23f3ad7c4dfa6ad90", 2516),
                ("b4_3396_23775", 61, "raw-016f962d269c54b622268d06b2d1", 3396),
            ),
        )
        self.assertEqual(
            tuple(source.axis_sha256 for source in sources),
            (
                "489815ede50d4202ee7f957f230b60447898356f5993fcd4c814524d9f2b452b",
                "3a9d737b1cdbfa0d078f1e6355ec6c0d74eeba4251d4370a822dbd8cbba268c1",
                "546518ca0c33206e34e15207be70c3fa45cf612db212873f7bf6a46a37d778f1",
                "4693c60c23d684d275ea3c79ae5ae98604d02ef5dd957c0116316a97dc0ff0be",
            ),
        )
        self.assertTrue(all(not source.spectrum.intensity.flags.writeable for source in sources))

    def test_rruff_source_loader_rejects_frozen_array_hash_drift(self) -> None:
        subset_rows = [
            json.loads(line)
            for line in (PHASE1_RUN / "source_subset.jsonl").read_bytes().splitlines()
        ]
        subset_rows[0]["normalized_intensity_float64_sha256"] = "0" * 64
        subset_bytes = b"".join(
            (
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            for row in subset_rows
        )
        subset_sha = hashlib.sha256(subset_bytes).hexdigest()
        manifest = json.loads((PHASE1_RUN / "manifest.json").read_bytes())
        manifest["selected_source_subset_sha256"] = subset_sha
        with tempfile.TemporaryDirectory() as temp_dir:
            run_path = Path(temp_dir)
            (run_path / "source_subset.jsonl").write_bytes(subset_bytes)
            (run_path / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with patch(
                "rpe.methods.classical.baseline_canary._PHASE1_SUBSET_SHA256",
                subset_sha,
            ), self.assertRaisesRegex(BaselineCanaryError, "normalized intensity"):
                load_rruff_baseline_canary_sources(run_path, RRUFF_DATASET)

    def test_fixture_artifacts_are_byte_deterministic_and_verifiable(self) -> None:
        spectrum = oracle_spectrum()
        source = BaselineCanarySource(
            band_id="fixture_band",
            selection_rank=0,
            record_id="fixture-record",
            sample_id="fixture-sample",
            class_label=1,
            mineral_name="fixture-mineral",
            point_count=spectrum.intensity.size,
            axis_sha256=hashlib.sha256(spectrum.axis_cm1.tobytes()).hexdigest(),
            intensity_sha256=hashlib.sha256(spectrum.intensity.tobytes()).hexdigest(),
            spectrum=spectrum,
        )
        systems = (
            baseline_system("airpls", {"lam": 1e6}),
            baseline_system("iarpls", {"lam": 1e5}),
        )
        catalog = load_classical_catalog(CATALOG_PATH)
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_baseline_canary_from_sources(
                (source,), systems, Path(first_dir), catalog=catalog, project_root=ROOT
            )
            second = build_baseline_canary_from_sources(
                (source,), systems, Path(second_dir), catalog=catalog, project_root=ROOT
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertTrue(first.canary_passed)
            self.assertEqual(first.attempt_count, 2)
            self.assertEqual(first.status_counts["complete"], 1)
            self.assertEqual(first.status_counts["failed_convergence"], 1)
            expected_names = {
                "SHA256SUMS",
                "complete.json",
                "manifest.json",
                "source_subset.jsonl",
                "summary.json",
                "system_results.jsonl",
            }
            self.assertEqual({path.name for path in first.path.iterdir()}, expected_names)
            for name in expected_names:
                self.assertEqual(
                    (first.path / name).read_bytes(),
                    (second.path / name).read_bytes(),
                )
            verified = verify_baseline_canary_artifact(first.path)
            self.assertEqual(verified.run_id, first.run_id)
            self.assertEqual(verified.attempt_count, 2)
            marker_path = first.path / "complete.json"
            marker_bytes = marker_path.read_bytes()
            checksum_path = first.path / "SHA256SUMS"
            checksum_bytes = checksum_path.read_bytes()
            marker = json.loads(marker_bytes)
            marker["status"] = "failed"
            marker_path.write_text(
                json.dumps(
                    marker,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            checksum_path.write_text(
                checksum_bytes.decode("utf-8").replace(
                    hashlib.sha256(marker_bytes).hexdigest(),
                    hashlib.sha256(marker_path.read_bytes()).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BaselineCanaryError, "marker"):
                verify_baseline_canary_artifact(first.path)
            marker_path.write_bytes(marker_bytes)
            checksum_path.write_bytes(checksum_bytes)
            manifest_path = first.path / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["catalog_id"] = "0" * 64
            manifest_payload = (
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
            manifest_path.write_bytes(manifest_payload)
            checksum_path.write_text(
                checksum_path.read_text(encoding="utf-8").replace(
                    hashlib.sha256((second.path / "manifest.json").read_bytes()).hexdigest(),
                    hashlib.sha256(manifest_payload).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(BaselineCanaryError, "run_id"):
                verify_baseline_canary_artifact(first.path)

    def test_builder_rejects_source_and_catalog_membership_drift(self) -> None:
        spectrum = oracle_spectrum()
        source = BaselineCanarySource(
            band_id="fixture_band", selection_rank=0, record_id="fixture-record",
            sample_id="fixture-sample", class_label=1, mineral_name="fixture",
            point_count=spectrum.intensity.size,
            axis_sha256=hashlib.sha256(spectrum.axis_cm1.tobytes()).hexdigest(),
            intensity_sha256=hashlib.sha256(spectrum.intensity.tobytes()).hexdigest(),
            spectrum=spectrum,
        )
        catalog = load_classical_catalog(CATALOG_PATH)
        system = baseline_system("airpls", {"lam": 1e6})
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaisesRegex(
            BaselineCanaryError, "source.point_count"
        ):
            build_baseline_canary_from_sources(
                (replace(source, point_count=31),),
                (system,),
                Path(temp_dir),
                catalog=catalog,
                project_root=ROOT,
            )
        with tempfile.TemporaryDirectory() as temp_dir, self.assertRaisesRegex(
            BaselineCanaryError, "catalog membership"
        ):
            build_baseline_canary_from_sources(
                (source,),
                (replace(system, system_id="0" * 64),),
                Path(temp_dir),
                catalog=catalog,
                project_root=ROOT,
            )


if __name__ == "__main__":
    unittest.main()
