from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.rruff import (  # noqa: E402
    CONFIG_SHA256,
    D5LibraryQuerySplit,
    D5RawCohort,
    load_d5_native_spectra,
)
from rpe.downstream.rruff_matching import (  # noqa: E402
    D5MatchingValidationError,
    match_d5_protocol_a_values,
)
from rpe.evaluation import Spectrum1D  # noqa: E402
from rpe.perturb import load_perturbation_sweep_config  # noqa: E402
from rpe.runner.phase4_d5_canary import (  # noqa: E402
    CANARY_PAYLOAD_FILES,
    Phase4D5CanaryConfig,
    Phase4D5CanaryError,
    build_phase4_d5_canary_from_inputs,
    load_phase4_d5_canary_config,
    verify_phase4_d5_canary,
)


DATASET = ROOT / "data" / "unified" / "rruff_raman_raw"
SWEEP_PATH = ROOT / "experiments" / "shared" / "raman_perturbation_sweep_v1.json"
CANARY_CONFIG = (
    ROOT
    / "experiments"
    / "phase4"
    / "configs"
    / "d5_p09_p11_protocol_a_canary_v1.json"
)


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _synthetic_cohort() -> D5RawCohort:
    axis = _read_only(np.arange(200.0, 1802.0, 2.0, dtype="<f4"))
    values = _read_only(np.ones((6, 801), dtype="<f4"))
    labels = _read_only(np.array([0, 1, 2, 0, 1, 2], dtype="<i8"))
    query = _read_only(np.array([0, 1, 2], dtype="<i8"))
    library = _read_only(np.array([3, 4, 5], dtype="<i8"))
    split = D5LibraryQuerySplit(
        seed=0,
        query_indices=query,
        library_indices=library,
        split_sha256="a" * 64,
    )
    return D5RawCohort(
        protocol_config_sha256=CONFIG_SHA256,
        dataset_id="rruff_raman_raw",
        intensity=values,
        wavenumber=axis,
        class_labels=labels,
        record_ids=("q0", "q1", "q2", "l0", "l1", "l2"),
        mineral_names=("m0", "m1", "m2", "m0", "m1", "m2"),
        rruff_ids=("rq0", "rq1", "rq2", "rl0", "rl1", "rl2"),
        pin_ids=(None,) * 6,
        group_ids=("gq0", "gq1", "gq2", "gl0", "gl1", "gl2"),
        splits=(split,),
    )


def _tree_snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in root.rglob("*")
        if path.is_file()
    }


class D5NativeSpectrumLoaderTest(unittest.TestCase):
    def test_native_loader_preserves_requested_order_axis_and_source_bytes(self) -> None:
        record_ids = []
        expected = {}
        with (DATASET / "records.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if len(record_ids) in (0, 1):
                    record_ids.append(row["record_id"])
                    expected[row["record_id"]] = row
                if len(record_ids) == 2:
                    break
        requested = tuple(reversed(record_ids))
        before = _tree_snapshot(DATASET)
        spectra = load_d5_native_spectra(DATASET, requested)
        after = _tree_snapshot(DATASET)

        self.assertEqual(after, before)
        self.assertEqual(
            tuple(spectrum.spectrum_id for spectrum in spectra),
            tuple(f"rruff_raman_raw::{record_id}" for record_id in requested),
        )
        for spectrum, record_id in zip(spectra, requested, strict=True):
            self.assertEqual(spectrum.sample_id, expected[record_id]["meta"]["sample_id"])
            self.assertEqual(spectrum.axis_cm1.dtype, np.dtype("<f8"))
            self.assertEqual(spectrum.intensity.dtype, np.dtype("<f8"))
            self.assertFalse(spectrum.axis_cm1.flags.writeable)
            self.assertFalse(spectrum.intensity.flags.writeable)
            self.assertNotEqual(spectrum.axis_cm1.size, 801)
            self.assertTrue(np.all(np.diff(spectrum.axis_cm1) > 0.0))

    def test_native_loader_rejects_duplicate_or_missing_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "record_ids.*unique"):
            load_d5_native_spectra(DATASET, ("x", "x"))
        with self.assertRaisesRegex(ValueError, "missing"):
            load_d5_native_spectra(DATASET, ("not-a-retained-record",))


class D5ProtocolAMatcherTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = _synthetic_cohort()
        self.split = self.cohort.splits[0]
        self.query = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="<f8",
        )
        self.library = self.query.copy()
        self.query_ids = tuple(
            self.cohort.record_ids[int(index)] for index in self.split.query_indices
        )
        self.library_ids = tuple(
            self.cohort.record_ids[int(index)] for index in self.split.library_indices
        )

    def test_protocol_a_matches_literal_classes_and_tie_order(self) -> None:
        result = match_d5_protocol_a_values(
            self.cohort,
            self.split,
            condition_id="phase4-test",
            query_record_ids=self.query_ids,
            library_record_ids=self.library_ids,
            query_values=self.query,
            library_values=self.library,
        )
        np.testing.assert_array_equal(result.ranked_class_labels[:, 0], [0, 1, 2])
        np.testing.assert_array_equal(result.top1_correct, [True, True, True])
        np.testing.assert_allclose(result.ranked_class_scores[:, 0], [1.0, 1.0, 1.0])

        tied = np.ones((3, 3), dtype="<f8")
        tie_result = match_d5_protocol_a_values(
            self.cohort,
            self.split,
            condition_id="phase4-tie",
            query_record_ids=self.query_ids,
            library_record_ids=self.library_ids,
            query_values=tied,
            library_values=tied,
        )
        np.testing.assert_array_equal(tie_result.ranked_class_labels[:, 0], [0, 0, 0])

    def test_protocol_a_rejects_shape_finite_norm_or_split_drift(self) -> None:
        cases = (
            (self.query[:, :2], self.library, "feature count"),
            (np.full((3, 3), np.nan), self.library, "finite"),
            (np.zeros((3, 3)), self.library, "zero L2 norm"),
        )
        for query, library, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(D5MatchingValidationError, expected):
                    match_d5_protocol_a_values(
                        self.cohort,
                        self.split,
                        condition_id="phase4-test",
                        query_record_ids=self.query_ids,
                        library_record_ids=self.library_ids,
                        query_values=query,
                        library_values=library,
                    )

        drifted = D5LibraryQuerySplit(
            seed=0,
            query_indices=self.split.query_indices,
            library_indices=self.split.library_indices,
            split_sha256="b" * 64,
        )
        with self.assertRaisesRegex(D5MatchingValidationError, "frozen split"):
            match_d5_protocol_a_values(
                self.cohort,
                drifted,
                condition_id="phase4-test",
                query_record_ids=self.query_ids,
                library_record_ids=self.library_ids,
                query_values=self.query,
                library_values=self.library,
            )

    def test_protocol_a_rejects_explicit_row_identity_drift(self) -> None:
        query_ids = tuple(
            self.cohort.record_ids[int(index)] for index in self.split.query_indices
        )
        library_ids = tuple(
            self.cohort.record_ids[int(index)] for index in self.split.library_indices
        )
        with self.assertRaisesRegex(D5MatchingValidationError, "query record IDs"):
            match_d5_protocol_a_values(
                self.cohort,
                self.split,
                condition_id="phase4-test",
                query_record_ids=tuple(reversed(query_ids)),
                library_record_ids=library_ids,
                query_values=self.query[::-1],
                library_values=self.library,
            )


def _native_spectra(cohort: D5RawCohort) -> tuple[Spectrum1D, ...]:
    axis = np.arange(180.0, 1822.0, 2.0, dtype="<f8")
    spectra = []
    for index, (record_id, class_label, rruff_id) in enumerate(
        zip(cohort.record_ids, cohort.class_labels, cohort.rruff_ids, strict=True)
    ):
        center = 420.0 + 360.0 * int(class_label)
        intensity = (
            1.0
            + np.exp(-0.5 * ((axis - center) / 22.0) ** 2)
            + 0.15 * np.exp(-0.5 * ((axis - (center + 80.0)) / 35.0) ** 2)
        )
        if index < 3:
            intensity = intensity + 0.005 * np.sin(axis / (11.0 + index))
        spectra.append(
            Spectrum1D(
                spectrum_id=f"rruff_raman_raw::{record_id}",
                sample_id=rruff_id,
                axis_cm1=axis,
                intensity=np.asarray(intensity, dtype="<f8"),
            )
        )
    return tuple(spectra)


def _ids_digest(values: tuple[str, ...]) -> str:
    return hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _synthetic_config(cohort: D5RawCohort) -> Phase4D5CanaryConfig:
    sweep = load_perturbation_sweep_config(SWEEP_PATH)
    split = cohort.splits[0]
    query_ids = tuple(cohort.record_ids[int(index)] for index in split.query_indices)
    library_ids = tuple(cohort.record_ids[int(index)] for index in split.library_indices)
    code_paths = (
        "rpe/downstream/rruff.py",
        "rpe/downstream/rruff_matching.py",
        "rpe/metrics/fidelity.py",
        "rpe/metrics/transport.py",
        "rpe/perturb/axis_transform.py",
        "rpe/perturb/gaussian_noise.py",
        "rpe/runner/phase4_d5_canary.py",
        "tools/run_phase4_d5_canary.py",
    )
    code_authority = {
        relative: {
            "bytes": (ROOT / relative).stat().st_size,
            "sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
        for relative in code_paths
    }
    document = {
        "alpha_grid": list(sweep.alpha_grid),
        "claim_boundary": "deterministic_vertical_slice_not_alignment_inference_or_paper_figure",
        "code_authority": code_authority,
        "experiment_id": "phase4-d5-p09-p11-protocol-a-canary-v1",
        "perturbation_ids": ["p09", "p11"],
        "schema_version": "phase4-d5-p09-p11-protocol-a-canary-config-v1",
        "synthetic_fixture": True,
    }
    raw = canonical_json_bytes(document)
    return Phase4D5CanaryConfig(
        path=Path("synthetic-config.json"),
        byte_count=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        document=MappingProxyType(document),
        design_sha256="d" * 64,
        sweep_sha256=sweep.sha256,
        d5_config_sha256=CONFIG_SHA256,
        dataset_sha256sums_sha256="e" * 64,
        seed=0,
        split_sha256=split.split_sha256,
        cohort_record_count=len(cohort.record_ids),
        query_count=len(query_ids),
        library_count=len(library_ids),
        query_record_ids_sha256=_ids_digest(query_ids),
        library_record_ids_sha256=_ids_digest(library_ids),
        perturbation_ids=("p09", "p11"),
        alpha_grid=sweep.alpha_grid,
        metric_ids=("mse", "wasserstein_1_cm1"),
        grid_start_cm1=204.0,
        grid_stop_cm1=1800.0,
        grid_step_cm1=2.0,
        grid_point_count=799,
        max_in_range_native_gap_cm1=3.0,
        expected_response_count=len(query_ids) * 18,
        expected_curve_count=18,
        claim_boundary=document["claim_boundary"],
        code_authority=MappingProxyType(
            {
                relative: MappingProxyType(dict(identity))
                for relative, identity in code_authority.items()
            }
        ),
    )


def canonical_json_bytes(value: object) -> bytes:
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


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class Phase4D5CanaryConfigTest(unittest.TestCase):
    def test_loads_exact_frozen_config_and_rejects_modified_copy(self) -> None:
        config = load_phase4_d5_canary_config(CANARY_CONFIG)
        self.assertEqual(config.perturbation_ids, ("p09", "p11"))
        self.assertEqual(config.grid_point_count, 799)
        self.assertEqual(config.expected_response_count, 23724)
        self.assertEqual(config.expected_curve_count, 18)
        self.assertEqual(config.seed, 0)

        changed = json.loads(CANARY_CONFIG.read_text())
        changed["downstream"]["grid_start_cm1"] = 202.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / CANARY_CONFIG.name
            path.write_bytes(canonical_json_bytes(changed))
            with self.assertRaisesRegex(Phase4D5CanaryError, "config identity"):
                load_phase4_d5_canary_config(path)


class Phase4D5CanaryArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cohort = _synthetic_cohort()
        self.native = _native_spectra(self.cohort)
        self.sweep = load_perturbation_sweep_config(SWEEP_PATH)
        self.config = _synthetic_config(self.cohort)

    def _build(self, path: Path):
        return build_phase4_d5_canary_from_inputs(
            path,
            cohort=self.cohort,
            native_spectra=self.native,
            sweep=self.sweep,
            config=self.config,
        )

    def test_builds_literal_rows_curves_and_claim_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact"
            summary = self._build(output)
            self.assertEqual(summary.query_count, 3)
            self.assertEqual(summary.library_count, 3)
            self.assertEqual(summary.response_count, 54)
            self.assertEqual(summary.curve_count, 18)

            rows = _jsonl(output / "response_records.jsonl")
            self.assertEqual(len(rows), 54)
            self.assertEqual(
                [
                    (row["query_order"], row["perturbation_id"], row["alpha"])
                    for row in rows
                ],
                [
                    (query, perturbation, alpha)
                    for query in range(3)
                    for perturbation in ("p09", "p11")
                    for alpha in self.sweep.alpha_grid
                ],
            )
            for query in range(3):
                p09 = [row for row in rows if row["query_order"] == query and row["perturbation_id"] == "p09"]
                p11 = [row for row in rows if row["query_order"] == query and row["perturbation_id"] == "p11"]
                p09_mse = [float(row["mse"]) for row in p09]
                self.assertEqual(p09_mse[0], 0.0)
                self.assertEqual(p09_mse, sorted(p09_mse))
                self.assertTrue(all(value > 0.0 for value in p09_mse[1:]))
                self.assertTrue(all(float(row["mse"]) == 0.0 for row in p11))
                self.assertTrue(all(float(row["wasserstein_1_cm1"]) > 0.0 for row in p11[1:]))
                self.assertTrue(all(bool(row["axis_changed"]) for row in p11[1:]))
                self.assertTrue(all(not bool(row["intensity_changed"]) for row in p11))
                for left, right in zip(p09[:1], p11[:1], strict=True):
                    for key in (
                        "downstream_intensity_sha256",
                        "top1_class_label",
                        "top1_score",
                        "top1_correct",
                        "top5_class_labels",
                        "top5_scores",
                        "top5_correct",
                    ):
                        self.assertEqual(left[key], right[key])

            curves = (output / "response_curve.csv").read_text().splitlines()
            self.assertEqual(len(curves), 19)
            manifest = json.loads((output / "manifest.json").read_bytes())
            self.assertEqual(manifest["claim_boundary"], self.config.claim_boundary)
            self.assertEqual(manifest["downstream_grid"]["point_count"], 799)
            forbidden = {"alignment_gap", "bootstrap", "holm", "p_value", "permutation", "significant"}
            self.assertFalse(recursive_keys(manifest) & forbidden)
            self.assertTrue((output / "complete.json").is_file())
            self.assertFalse((output / "failed.json").exists())
            self.assertEqual(
                [line.split("  ", 1)[1] for line in (output / "SHA256SUMS").read_text().splitlines()],
                list(CANARY_PAYLOAD_FILES),
            )
            verify_phase4_d5_canary(output, reexecute=False)

    def test_repeated_artifacts_match_and_tampering_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            self._build(first)
            self._build(second)
            for name in (*CANARY_PAYLOAD_FILES, "SHA256SUMS"):
                self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

            changed = root / "changed"
            changed.mkdir()
            for name in (*CANARY_PAYLOAD_FILES, "SHA256SUMS"):
                (changed / name).write_bytes((first / name).read_bytes())
            rows = _jsonl(changed / "response_records.jsonl")
            rows[0]["mse"] = 1.0
            (changed / "response_records.jsonl").write_bytes(
                b"".join(canonical_json_bytes(row) for row in rows)
            )
            with self.assertRaisesRegex(Phase4D5CanaryError, "checksum"):
                verify_phase4_d5_canary(changed, reexecute=False)

    def test_self_consistent_scientific_tampering_fails_authority_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "artifact"
            self._build(artifact)
            rows = _jsonl(artifact / "response_records.jsonl")
            target = next(
                row
                for row in rows
                if row["perturbation_id"] == "p11" and row["alpha"] == 0.8
            )
            target["wasserstein_1_cm1"] = 99.0
            (artifact / "response_records.jsonl").write_bytes(
                b"".join(canonical_json_bytes(row) for row in rows)
            )
            # Reproject the curve with a deliberately self-consistent wrong W1.
            curve_path = artifact / "response_curve.csv"
            curve_rows = list(csv.DictReader(curve_path.read_text().splitlines()))
            curve_row = next(
                row
                for row in curve_rows
                if row["perturbation_id"] == "p11" and float(row["alpha"]) == 0.8
            )
            changed_values = [
                float(row["wasserstein_1_cm1"])
                for row in rows
                if row["perturbation_id"] == "p11" and row["alpha"] == 0.8
            ]
            curve_row["mean_wasserstein_1_cm1"] = repr(float(np.mean(changed_values)))
            curve_row["median_wasserstein_1_cm1"] = repr(float(np.median(changed_values)))
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=curve_rows[0].keys(), lineterminator="\n")
            writer.writeheader()
            writer.writerows(curve_rows)
            curve_path.write_text(stream.getvalue())
            # Refresh checksums so this is not a trivial checksum-corruption test.
            checksum = "".join(
                f"{hashlib.sha256((artifact / name).read_bytes()).hexdigest()}  {name}\n"
                for name in CANARY_PAYLOAD_FILES
            )
            (artifact / "SHA256SUMS").write_text(checksum)
            with self.assertRaisesRegex(Phase4D5CanaryError, "P11 W1 formula"):
                verify_phase4_d5_canary(artifact, reexecute=False)

    def test_rejects_downstream_extrapolation(self) -> None:
        short = list(self.native)
        original = short[0]
        short[0] = Spectrum1D(
            spectrum_id=original.spectrum_id,
            sample_id=original.sample_id,
            axis_cm1=np.arange(300.0, 1822.0, 2.0, dtype="<f8"),
            intensity=original.intensity[-761:],
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(Phase4D5CanaryError, "downstream support"):
                build_phase4_d5_canary_from_inputs(
                    Path(temporary) / "artifact",
                    cohort=self.cohort,
                    native_spectra=tuple(short),
                    sweep=self.sweep,
                    config=self.config,
                )


def recursive_keys(value: object) -> set[str]:
    keys = set()
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(recursive_keys(item))
    return keys


if __name__ == "__main__":
    unittest.main()
