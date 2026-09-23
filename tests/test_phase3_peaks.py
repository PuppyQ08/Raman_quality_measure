from __future__ import annotations

import hashlib
import json
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Peak1D, Spectrum1D  # noqa: E402
from rpe.methods import TaskLine, load_classical_catalog  # noqa: E402
from rpe.methods.classical import (  # noqa: E402
    PeakCanaryError,
    PeakRunStatus,
    PeakWrapperError,
    build_peak_canary_artifact,
    load_peak_canary_config,
    load_peak_canary_sources,
    run_peak_detection_system,
    verify_peak_canary_artifact,
)
from rpe.methods.classical import peaks as peak_module  # noqa: E402


CATALOG_PATH = ROOT / "experiments" / "phase3" / "configs" / "classical_system_catalog_v1.json"
CONFIG_PATH = ROOT / "experiments" / "phase3" / "configs" / "peak_detection_v1_canary.json"


def system(family: str, **selector: object):
    catalog = load_classical_catalog(CATALOG_PATH)
    matches = [
        value
        for value in catalog.systems
        if value.task_line is TaskLine.PEAK_DETECTION
        and value.family_id == family
        and all(value.hyperparameters.get(key) == expected for key, expected in selector.items())
    ]
    if len(matches) != 1:
        raise AssertionError((family, selector, len(matches)))
    return matches[0]


def spectrum(
    spectrum_id: str,
    intensity: np.ndarray,
    *,
    axis: np.ndarray | None = None,
) -> Spectrum1D:
    values = np.asarray(intensity, dtype="<f8")
    physical_axis = (
        np.arange(values.size, dtype="<f8")
        if axis is None
        else np.asarray(axis, dtype="<f8")
    )
    return Spectrum1D(spectrum_id, None, physical_axis, values)


def canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


class PeakWrapperOracleTest(unittest.TestCase):
    def test_triangular_peak_has_hand_index_prominence_fwhm_and_area(self) -> None:
        detector = system("find_peaks", prominence_fraction=0.005)
        result = run_peak_detection_system(
            detector, spectrum("triangle", np.array([0, 1, 2, 1, 0], dtype="<f8"))
        )
        self.assertIs(result.status, PeakRunStatus.COMPLETE)
        self.assertEqual(len(result.peaks), 1)
        peak = result.peaks[0]
        self.assertEqual(peak.index, 2)
        self.assertEqual(peak.position_cm1, 2.0)
        self.assertEqual(peak.height, 2.0)
        self.assertEqual(peak.prominence, 2.0)
        self.assertEqual((peak.left_base_index, peak.right_base_index), (0, 4))
        self.assertEqual(peak.contour_height, 0.0)
        self.assertEqual(peak.fwhm_cm1, 2.0)
        self.assertEqual((peak.area_left_cm1, peak.area_right_cm1), (0.0, 4.0))
        self.assertEqual(peak.area, 4.0)

    def test_gaussian_fwhm_matches_analytic_value(self) -> None:
        axis = np.linspace(-10.0, 10.0, 2001, dtype="<f8")
        values = np.exp(-0.5 * axis * axis).astype("<f8")
        result = run_peak_detection_system(
            system("find_peaks", prominence_fraction=0.005),
            spectrum("gaussian", values, axis=axis),
        )
        self.assertIs(result.status, PeakRunStatus.COMPLETE)
        self.assertEqual(len(result.peaks), 1)
        self.assertLess(
            abs(result.peaks[0].fwhm_cm1 - 2.0 * math.sqrt(2.0 * math.log(2.0))),
            0.02,
        )

    def test_plateau_negative_offset_endpoints_and_empty_output_contract(self) -> None:
        detector = system("find_peaks", prominence_fraction=0.005)
        plateau = run_peak_detection_system(
            detector,
            spectrum("plateau", np.array([-5, -4, -3, -3, -3, -4, -5], dtype="<f8")),
        )
        self.assertEqual(tuple(value.index for value in plateau.peaks), (3,))
        self.assertEqual(plateau.peaks[0].height, -3.0)
        endpoints = run_peak_detection_system(
            detector, spectrum("endpoints", np.array([3, 0, 1, 0, 3], dtype="<f8"))
        )
        self.assertEqual(tuple(value.index for value in endpoints.peaks), (2,))
        constant = run_peak_detection_system(detector, spectrum("constant", np.ones(32)))
        self.assertIs(constant.status, PeakRunStatus.NOT_APPLICABLE)
        self.assertEqual(constant.error_code, "nonpositive_robust_range")
        cwt_constant = run_peak_detection_system(
            system("find_peaks_cwt", widths_cm1=(1.0, 2.0, 3.0, 4.0), min_snr=1),
            spectrum("cwt-empty", np.ones(32)),
        )
        self.assertIs(cwt_constant.status, PeakRunStatus.COMPLETE)
        self.assertEqual(cwt_constant.peaks, ())

    def test_half_up_width_conversion_and_unsupported_bank(self) -> None:
        converted, diagnostics = peak_module._convert_width_bank(
            np.array([0.0, 1.4, 2.8, 4.2, 5.6], dtype="<f8"),
            (1.0, 2.0, 4.0, 8.0),
        )
        self.assertEqual(converted, (1, 3, 6))
        self.assertEqual(diagnostics["physical_to_point"], ((1.0, 1), (2.0, 1), (4.0, 3), (8.0, 6)))
        with self.assertRaisesRegex(PeakWrapperError, "unsupported_width_bank"):
            peak_module._convert_width_bank(
                np.array([0.0, 10.0, 20.0, 30.0], dtype="<f8"),
                (1.0, 2.0, 3.0, 4.0),
            )

    def test_cwt_candidates_are_filtered_and_uniformly_characterized(self) -> None:
        axis = np.arange(101, dtype="<f8")
        values = np.exp(-0.5 * ((axis - 50.0) / 5.0) ** 2)
        source = spectrum("cwt-gaussian", values, axis=axis)
        cwt = run_peak_detection_system(
            system("find_peaks_cwt", widths_cm1=(1.0, 2.0, 3.0, 4.0), min_snr=1),
            source,
        )
        direct = run_peak_detection_system(
            system("find_peaks", prominence_fraction=0.005), source
        )
        self.assertEqual(tuple(value.index for value in cwt.peaks), (50,))
        direct_peak = next(value for value in direct.peaks if value.index == 50)
        self.assertEqual(cwt.peaks[0], direct_peak)
        self.assertEqual(cwt.diagnostics["candidate_count"], 2)
        self.assertEqual(cwt.diagnostics["rejected_non_admissible"], 1)

    def test_result_and_projection_are_immutable_and_content_hashed(self) -> None:
        result = run_peak_detection_system(
            system("find_peaks", prominence_fraction=0.005),
            spectrum("immutable", np.array([0, 1, 2, 1, 0], dtype="<f8")),
        )
        self.assertRegex(result.peaks_sha256 or "", r"^[0-9a-f]{64}$")
        peak = result.peaks[0]
        projected = peak.to_peak1d()
        self.assertIsInstance(projected, Peak1D)
        self.assertEqual(
            projected,
            Peak1D(
                position_cm1=2.0,
                height=2.0,
                fwhm_cm1=2.0,
                area=4.0,
                prominence=2.0,
            ),
        )
        with self.assertRaises((AttributeError, TypeError)):
            peak.index = 3  # type: ignore[misc]

    def test_mspd_catalog_system_is_rejected_as_unimplemented(self) -> None:
        mspd = system("mspd", max_scale_cm1=4.0, ridge_vote_fraction=0.25)
        with self.assertRaisesRegex(PeakWrapperError, "MSPD"):
            run_peak_detection_system(
                mspd, spectrum("mspd", np.array([0, 1, 2, 1, 0], dtype="<f8"))
            )

    def test_unexpected_characterization_exception_is_failed_runtime(self) -> None:
        detector = system("find_peaks", prominence_fraction=0.005)
        with patch(
            "rpe.methods.classical.peaks.peak_prominences",
            side_effect=RuntimeError("injected characterization failure"),
        ):
            result = run_peak_detection_system(
                detector,
                spectrum("runtime", np.array([0, 1, 2, 1, 0], dtype="<f8")),
            )
        self.assertIs(result.status, PeakRunStatus.FAILED_RUNTIME)
        self.assertEqual(result.error_code, "RuntimeError")
        self.assertEqual(result.peaks, ())


class PeakCanaryTest(unittest.TestCase):
    def test_real_sources_match_six_frozen_ids_and_hashes(self) -> None:
        config = load_peak_canary_config(CONFIG_PATH, project_root=ROOT)
        sources = load_peak_canary_sources(config, project_root=ROOT)
        self.assertEqual(
            tuple(value.record_id for value in sources),
            (
                "raw-0003d1b556e9bc8434a8a7972d5f",
                "raw-002525ef577ac9ed746f80d1bfea",
                "raw-0005cc5ef4a23f3ad7c4dfa6ad90",
                "raw-016f962d269c54b622268d06b2d1",
                "test-000000",
                "test-002900",
            ),
        )
        self.assertEqual(len(sources), 6)
        self.assertEqual(
            (sources[4].axis_sha256, sources[4].intensity_sha256),
            (
                "4ceda8f9376a140fba543b8aa185801829a9c1fe04e820a6f47c57c7bec92a5d",
                "98c5b92880fe6aa02527660c3030361e61d1d9899d541876471d28b0f92d6bcc",
            ),
        )
        self.assertEqual(
            sources[5].intensity_sha256,
            "bcf8dab1a7580444ed4de207c7e66f76772b4f8e374eb3cf82702bc0d2a09590",
        )
        for source in sources:
            robust_range = float(
                np.quantile(source.spectrum.intensity, 0.99, method="linear")
                - np.quantile(source.spectrum.intensity, 0.01, method="linear")
            )
            self.assertGreater(robust_range, 0.0)
            for bank in (
                (1.0, 2.0, 3.0, 4.0),
                (1.0, 2.0, 4.0, 8.0),
                (2.0, 4.0, 6.0, 8.0),
                (2.0, 4.0, 8.0, 12.0),
            ):
                converted, _ = peak_module._convert_width_bank(source.spectrum.axis_cm1, bank)
                self.assertGreaterEqual(len(converted), 2)

    def test_reduced_artifact_is_verifiable_and_byte_deterministic(self) -> None:
        config = load_peak_canary_config(CONFIG_PATH, project_root=ROOT)
        sources = load_peak_canary_sources(config, project_root=ROOT)[:2]
        systems = (
            system("find_peaks", prominence_fraction=0.005),
            system("find_peaks_cwt", widths_cm1=(1.0, 2.0, 3.0, 4.0), min_snr=1),
        )
        catalog = load_classical_catalog(CATALOG_PATH)
        with tempfile.TemporaryDirectory() as first_root, tempfile.TemporaryDirectory() as second_root:
            first = build_peak_canary_artifact(
                sources,
                systems,
                Path(first_root),
                config=config,
                catalog=catalog,
                project_root=ROOT,
            )
            second = build_peak_canary_artifact(
                sources,
                systems,
                Path(second_root),
                config=config,
                catalog=catalog,
                project_root=ROOT,
            )
            self.assertEqual(first.run_id, second.run_id)
            self.assertFalse(first.is_frozen_canary)
            first_files = {
                path.relative_to(first.path).as_posix(): path.read_bytes()
                for path in first.path.rglob("*")
                if path.is_file()
            }
            second_files = {
                path.relative_to(second.path).as_posix(): path.read_bytes()
                for path in second.path.rglob("*")
                if path.is_file()
            }
            self.assertEqual(first_files, second_files)
            verified = verify_peak_canary_artifact(first.path, project_root=ROOT)
            self.assertEqual(verified.run_id, first.run_id)
            self.assertEqual(verified.receipt_count, 4)

    def test_verifier_rejects_rebound_peak_and_summary_drift(self) -> None:
        config = load_peak_canary_config(CONFIG_PATH, project_root=ROOT)
        sources = load_peak_canary_sources(config, project_root=ROOT)[:1]
        systems = (system("find_peaks", prominence_fraction=0.005),)
        catalog = load_classical_catalog(CATALOG_PATH)
        with tempfile.TemporaryDirectory() as temp_root:
            artifact = build_peak_canary_artifact(
                sources,
                systems,
                Path(temp_root),
                config=config,
                catalog=catalog,
                project_root=ROOT,
            )
            checksum_path = artifact.path / "SHA256SUMS"
            checksum_original = checksum_path.read_bytes()
            peaks_path = artifact.path / "peaks.jsonl"
            peaks_original = peaks_path.read_bytes()
            rows = [json.loads(line) for line in peaks_original.splitlines()]
            self.assertTrue(rows)
            rows[0]["prominence"] += 1.0
            changed = b"".join(canonical(row) for row in rows)
            peaks_path.write_bytes(changed)
            checksum_path.write_text(
                checksum_original.decode().replace(
                    hashlib.sha256(peaks_original).hexdigest(),
                    hashlib.sha256(changed).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PeakCanaryError, "peak ledger"):
                verify_peak_canary_artifact(artifact.path, project_root=ROOT)
            peaks_path.write_bytes(peaks_original)
            checksum_path.write_bytes(checksum_original)

            summary_path = artifact.path / "summary.json"
            summary_original = summary_path.read_bytes()
            document = json.loads(summary_original)
            document["receipt_count"] += 1
            changed_summary = canonical(document)
            summary_path.write_bytes(changed_summary)
            checksum_path.write_text(
                checksum_original.decode().replace(
                    hashlib.sha256(summary_original).hexdigest(),
                    hashlib.sha256(changed_summary).hexdigest(),
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PeakCanaryError, "summary"):
                verify_peak_canary_artifact(artifact.path, project_root=ROOT)


if __name__ == "__main__":
    unittest.main()
