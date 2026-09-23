from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.perturb import load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase1_preview import (  # noqa: E402
    PREVIEW_PAYLOAD_FILES,
    build_mse_preview_from_sources,
)
from rpe.runner.phase1_selection import (  # noqa: E402
    Phase1Source,
    SelectedSourceRow,
)


SWEEP_PATH = ROOT / "experiments" / "shared" / "raman_perturbation_sweep_v1.json"


def _source(rank: int, *, scale: float) -> Phase1Source:
    axis = np.linspace(100.0, 900.0, 64, dtype="<f8")
    intensity = np.asarray(
        scale
        * (
            0.2
            + np.exp(-0.5 * ((axis - 350.0) / 45.0) ** 2)
            + 0.6 * np.exp(-0.5 * ((axis - 680.0) / 70.0) ** 2)
        ),
        dtype="<f8",
    )
    record_id = f"record-{rank:02d}"
    spectrum = Spectrum1D(
        spectrum_id=f"rruff_raman_raw::{record_id}",
        sample_id=f"sample-{rank:02d}",
        axis_cm1=axis,
        intensity=intensity,
    )
    return Phase1Source(
        selection=SelectedSourceRow(
            selection_rank=rank,
            record_id=record_id,
            sample_id=spectrum.sample_id or "",
            class_label=rank,
            mineral_name=f"Mineral-{rank}",
            axis_id=f"axis-{rank}",
        ),
        spectrum=spectrum,
        original_axis_orientation="increasing",
        source_axis_float32_sha256=hashlib.sha256(
            axis.astype("<f4").tobytes()
        ).hexdigest(),
        source_intensity_float32_sha256=hashlib.sha256(
            intensity.astype("<f4").tobytes()
        ).hexdigest(),
        normalized_axis_float64_sha256=hashlib.sha256(axis.tobytes()).hexdigest(),
        normalized_intensity_float64_sha256=hashlib.sha256(
            intensity.tobytes()
        ).hexdigest(),
        provenance=MappingProxyType(
            {
                "license": None,
                "license_status": "not_stated",
                "retrieved_date": "2026-08-17",
                "sha256": "a" * 64,
                "source_artifact": "fixture",
                "source_url": "https://example.invalid/fixture",
            }
        ),
    )


def _provenance() -> dict[str, object]:
    return {
        "formal_config_sha256": "b" * 64,
        "formal_selected_source_count": 10000,
        "selection_rule": "formal_core10k_selection_rank_prefix",
        "source_snapshot_sha256": "c" * 64,
    }


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class Phase1MsePreviewTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sweep = load_perturbation_sweep_config(SWEEP_PATH)
        self.sources = (_source(0, scale=1.0), _source(1, scale=2.0))

    def test_preview_uses_frozen_science_and_exposes_p11_mse_blindness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "preview"
            summary = build_mse_preview_from_sources(
                output,
                sources=self.sources,
                sweep=self.sweep,
                provenance=_provenance(),
            )

            rows = _load_jsonl(output / "response_records.jsonl")
            self.assertEqual(summary.sample_count, 2)
            self.assertEqual(summary.row_count, 36)
            self.assertEqual(len(rows), 36)
            self.assertEqual(
                [
                    (row["selection_rank"], row["perturbation_id"], row["alpha"])
                    for row in rows
                ],
                [
                    (rank, perturbation_id, alpha)
                    for rank in (0, 1)
                    for perturbation_id in ("p09", "p11")
                    for alpha in self.sweep.alpha_grid
                ],
            )

            for rank in (0, 1):
                p09 = [
                    row
                    for row in rows
                    if row["selection_rank"] == rank
                    and row["perturbation_id"] == "p09"
                ]
                p11 = [
                    row
                    for row in rows
                    if row["selection_rank"] == rank
                    and row["perturbation_id"] == "p11"
                ]
                p09_mse = [float(row["mse"]) for row in p09]
                self.assertEqual(p09_mse[0], 0.0)
                self.assertEqual(p09_mse, sorted(p09_mse))
                self.assertTrue(all(value > 0.0 for value in p09_mse[1:]))
                self.assertFalse(bool(p09[0]["intensity_changed"]))
                self.assertTrue(all(bool(row["intensity_changed"]) for row in p09[1:]))

                self.assertTrue(all(float(row["mse"]) == 0.0 for row in p11))
                self.assertFalse(bool(p11[0]["axis_changed"]))
                self.assertTrue(all(bool(row["axis_changed"]) for row in p11[1:]))
                self.assertTrue(all(not bool(row["intensity_changed"]) for row in p11))

            with (output / "response_curve.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                curve_rows = list(csv.DictReader(stream))
            self.assertEqual(len(curve_rows), 18)
            self.assertEqual(
                [(row["perturbation_id"], float(row["alpha"])) for row in curve_rows],
                [
                    (perturbation_id, alpha)
                    for perturbation_id in ("p09", "p11")
                    for alpha in self.sweep.alpha_grid
                ],
            )
            self.assertTrue(all(int(row["spectrum_count"]) == 2 for row in curve_rows))

            manifest = json.loads((output / "manifest.json").read_bytes())
            self.assertEqual(
                manifest["claim_boundary"],
                "exploratory_preview_not_final_paper_evidence",
            )
            self.assertEqual(
                manifest["p07"],
                {
                    "dependency": "phase2_semisynthetic_or_separately_approved_physical_baseline",
                    "reason": "missing_explicit_baseline",
                    "status": "not_run_missing_explicit_baseline",
                },
            )
            self.assertEqual(manifest["preprocessing"], {"interpolation": "none", "normalization": "none"})
            self.assertEqual(
                manifest["mse_interpretation"]["p11"],
                "index_aligned_intensity_mse_is_zero_despite_physical_axis_shift",
            )
            self.assertEqual(
                manifest["preview_sample"]["statistical_role"],
                "deterministic_pipeline_smoke_test_not_stratified_representative_or_inferential",
            )

            checksum_lines = (output / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
            self.assertEqual(
                [line.split("  ", 1)[1] for line in checksum_lines],
                list(PREVIEW_PAYLOAD_FILES),
            )
            for line in checksum_lines:
                expected, name = line.split("  ", 1)
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)

    def test_repeated_preview_artifacts_are_byte_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            for output in (first, second):
                build_mse_preview_from_sources(
                    output,
                    sources=self.sources,
                    sweep=self.sweep,
                    provenance=_provenance(),
                )
            for name in (*PREVIEW_PAYLOAD_FILES, "SHA256SUMS"):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                    name,
                )


if __name__ == "__main__":
    unittest.main()
