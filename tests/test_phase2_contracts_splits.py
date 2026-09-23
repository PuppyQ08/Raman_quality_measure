from __future__ import annotations

import hashlib
import importlib.metadata
import json
import struct
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.semisynth import (  # noqa: E402
    Phase2ConfigError,
    RruffPairRole,
    SemiSyntheticContractError,
    SemiSyntheticRecord,
    assign_rruff_group_role,
    excitation_stratum,
    load_phase2_config,
    split_rruff_pairs,
)


CONFIG = ROOT / "experiments" / "phase2" / "configs" / "semisynth_v1.json"
LOCK = ROOT / "env" / "requirements.lock"
PAIR_INDEX = ROOT / "data" / "unified" / "rruff_raman_pairs.jsonl"
RAW_DATASET = ROOT / "data" / "unified" / "rruff_raman_raw"


class Phase2ConfigTest(unittest.TestCase):
    def test_config_lock_and_dependency_are_exact(self) -> None:
        lock = LOCK.read_bytes()
        self.assertEqual(len(lock), 126)
        self.assertEqual(
            hashlib.sha256(lock).hexdigest(),
            "3057d1a4f5198139b1e57dd9cca92b05e9c362a5f10bc87fad3a6041c54a9963",
        )
        self.assertEqual(
            lock,
            b"h5py==3.16.0\n"
            b"joblib==1.5.3\n"
            b"matplotlib==3.11.1\n"
            b"numpy==2.5.2\n"
            b"pandas==3.0.5\n"
            b"pybaselines==1.2.1\n"
            b"scikit-learn==1.9.0\n"
            b"scipy==1.18.0\n",
        )
        self.assertEqual(importlib.metadata.version("pybaselines"), "1.2.1")

        raw = CONFIG.read_bytes()
        self.assertEqual(len(raw), 4722)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "8ba338509dbbb4b74a8675f1875bf1048d40359ea4f617b0d087b146f60b0af1",
        )
        document = json.loads(raw)
        self.assertEqual(
            raw,
            (
                json.dumps(
                    document,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8"),
        )
        config = load_phase2_config(CONFIG)
        self.assertEqual(config.schema_version, "phase2-semisynth-config-v1")
        self.assertEqual(config.global_seed, 20260818)
        self.assertEqual(config.legendre_degree, 6)
        self.assertEqual(config.pilot_per_extractor, 2048)
        self.assertEqual(config.full_per_extractor, 65536)
        self.assertEqual(config.extractor_ids, ("airpls", "arpls", "mor"))
        self.assertEqual(config.minimum_systems, 80)
        self.assertEqual(config.minimum_method_families, 10)
        self.assertEqual(config.tau_b_threshold, 0.85)
        self.assertEqual(
            config.template_qualifiers["rruff_processed"],
            "algorithmically_processed_signal_template_not_physical_clean_gt",
        )

    def test_loader_rejects_noncanonical_and_identity_drift(self) -> None:
        with self.assertRaisesRegex(Phase2ConfigError, "config identity"):
            load_phase2_config(ROOT / "phase1_fixture_shared_sweep.json")

    def test_loader_rejects_bound_lock_and_small_source_content_drift(self) -> None:
        document = json.loads(CONFIG.read_bytes())

        def linked_root(path: Path) -> None:
            relative_paths = [document["dependencies"]["lock_path"]]
            relative_paths.extend(
                identity["path"] for identity in document["sources"].values()
            )
            for relative_path in relative_paths:
                destination = path / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.symlink_to(ROOT / relative_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            linked_root(project_root)
            lock_path = project_root / document["dependencies"]["lock_path"]
            lock_path.unlink()
            lock_path.write_bytes(b"x" * LOCK.stat().st_size)
            with self.assertRaisesRegex(Phase2ConfigError, "dependency lock.*SHA256"):
                load_phase2_config(CONFIG, project_root=project_root)

        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            linked_root(project_root)
            source = document["sources"]["rruff_raw_checksum_index"]
            source_path = project_root / source["path"]
            source_path.unlink()
            source_path.write_bytes(b"x" * source["byte_count"])
            with self.assertRaisesRegex(
                Phase2ConfigError, "sources.rruff_raw_checksum_index.*SHA256"
            ):
                load_phase2_config(CONFIG, project_root=project_root)


class Phase2SplitTest(unittest.TestCase):
    def test_group_role_hash_and_stratum_boundaries_have_literal_oracles(self) -> None:
        config = load_phase2_config(CONFIG)
        cases = (
            (
                "R040041",
                "ed959c9d93e1a1b7886bb681af711e8af318e644bc5d85fe3440d19ea2c096c1",
                RruffPairRole.SIGNAL_TEMPLATE,
            ),
            (
                "R070224",
                "26f6fde4be0d1a67411bd6232382a5b721840f5910f466bdc93871f7ce4eb5ef",
                RruffPairRole.EXTRACTION_FIT,
            ),
            (
                "R050336",
                "925d371a6308dfeb586e3e2c8d57b9da0802f688d1596885900f918086b65885",
                RruffPairRole.REAL_HOLDOUT,
            ),
        )
        for group_key, expected_digest, expected_role in cases:
            with self.subTest(group_key=group_key):
                encoded = group_key.encode("utf-8")
                literal = hashlib.sha256(
                    b"rpe-phase2-rruff-pair-split-v1\0"
                    + struct.pack("<Q", 20260818)
                    + struct.pack("<Q", len(encoded))
                    + encoded
                ).hexdigest()
                self.assertEqual(literal, expected_digest)
                self.assertEqual(
                    assign_rruff_group_role(group_key, config=config),
                    expected_role,
                )

        strata = (
            (509.999, None),
            (510.0, "green_514"),
            (517.999, "green_514"),
            (518.0, None),
            (528.0, "green_532"),
            (535.999, "green_532"),
            (536.0, None),
            (775.0, "nir_780"),
            (782.999, "nir_780"),
            (783.0, "nir_785"),
            (790.0, "nir_785"),
            (790.001, None),
            (None, None),
        )
        for excitation, expected in strata:
            with self.subTest(excitation=excitation):
                self.assertEqual(
                    excitation_stratum(excitation, config=config), expected
                )

    def test_group_role_and_stratum_semantics_are_config_driven(self) -> None:
        config = load_phase2_config(CONFIG)
        self.assertEqual(
            assign_rruff_group_role("G0000", config=config),
            RruffPairRole.REAL_HOLDOUT,
        )
        threshold_config = replace(
            config, fit_threshold=0.95, holdout_threshold=0.975
        )
        self.assertEqual(
            assign_rruff_group_role("G0000", config=threshold_config),
            RruffPairRole.EXTRACTION_FIT,
        )
        domain_config = replace(config, split_domain="alternate-phase2-split-v1")
        self.assertEqual(
            assign_rruff_group_role("G0000", config=domain_config),
            RruffPairRole.SIGNAL_TEMPLATE,
        )

        shifted_strata = {
            name: dict(bounds) for name, bounds in config.split_strata.items()
        }
        shifted_strata["green_514"] = {
            "lower_inclusive": 500.0,
            "upper_exclusive": 518.0,
        }
        stratum_config = replace(config, split_strata=shifted_strata)
        self.assertIsNone(excitation_stratum(509.0, config=config))
        self.assertEqual(
            excitation_stratum(509.0, config=stratum_config), "green_514"
        )

    def test_retained_pair_split_has_exact_counts_digest_and_no_group_leakage(self) -> None:
        config = load_phase2_config(CONFIG)
        summary = split_rruff_pairs(
            PAIR_INDEX,
            RAW_DATASET,
            config=config,
            verify_checksums=True,
        )
        self.assertEqual(summary.assignment_count, 15764)
        self.assertEqual(summary.group_count, 3919)
        self.assertEqual(
            dict(summary.pair_counts),
            {
                "extraction_fit": 9371,
                "real_holdout": 3266,
                "signal_template": 3127,
            },
        )
        self.assertEqual(
            dict(summary.group_counts),
            {
                "extraction_fit": 2333,
                "real_holdout": 825,
                "signal_template": 761,
            },
        )
        self.assertEqual(
            summary.ledger_sha256,
            "0b6323973a3afabc840c57a051a7eeb31cb89efcb2bdea0546c5b15141d95c5e",
        )
        self.assertEqual(
            dict(summary.stratum_pair_counts["signal_template"]),
            {
                "green_514": 1232,
                "green_532": 944,
                "nir_780": 674,
                "nir_785": 277,
                "unstratified": 0,
            },
        )
        role_by_group: dict[str, set[RruffPairRole]] = {}
        for assignment in summary.assignments:
            role_by_group.setdefault(assignment.group_key, set()).add(
                assignment.role
            )
        self.assertTrue(all(len(roles) == 1 for roles in role_by_group.values()))
        self.assertEqual(
            summary.assignments[0].pair_id,
            "0004138fdc9671354816ac8a318aca5e822624161ee3d0d7cda161924361e369",
        )
        self.assertEqual(
            summary.assignments[-1].pair_id,
            "fff4de21d781d2d3f250b909b8297798b49c3eb4fa38563914a5d68b26b632f6",
        )


class SemiSyntheticRecordContractTest(unittest.TestCase):
    def test_record_copies_arrays_and_enforces_exact_decomposition(self) -> None:
        axis = np.array([100.0, 200.0, 300.0, 400.0], dtype="<f8")
        template = np.array([1.0, 2.0, 1.5, 0.5], dtype="<f8")
        background = np.array([0.2, 0.4, 0.7, 1.0], dtype="<f8")
        noise = np.array([0.1, -0.1, 0.2, -0.2], dtype="<f8")
        observed = template + background + noise
        record = SemiSyntheticRecord(
            record_id="semisynth-fixture",
            extractor_id="airpls",
            excitation_stratum="green_532",
            template_id="processed-template",
            template_qualifier=(
                "algorithmically_processed_signal_template_not_physical_clean_gt"
            ),
            axis_cm1=axis,
            y_observed=np.asarray(observed, dtype="<f8"),
            s_template=template,
            b_true=background,
            n_true=noise,
            peaks_template=(),
            provenance={"seed": 20260818, "role": "fixture"},
        )
        axis[0] = 999.0
        template[0] = 999.0
        np.testing.assert_array_equal(
            record.axis_cm1,
            np.array([100.0, 200.0, 300.0, 400.0], dtype="<f8"),
        )
        np.testing.assert_allclose(
            record.y_observed,
            record.s_template + record.b_true + record.n_true,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(record.axis_cm1.flags.writeable)
        self.assertFalse(record.y_observed.flags.writeable)
        self.assertEqual(dict(record.provenance), {"role": "fixture", "seed": 20260818})
        with self.assertRaises(ValueError):
            record.y_observed[0] = 0.0

    def test_record_rejects_wrong_dtype_shape_negative_background_and_drift(self) -> None:
        good = {
            "record_id": "semisynth-fixture",
            "extractor_id": "airpls",
            "excitation_stratum": "green_532",
            "template_id": "template",
            "template_qualifier": (
                "algorithmically_processed_signal_template_not_physical_clean_gt"
            ),
            "axis_cm1": np.array([100.0, 200.0, 300.0], dtype="<f8"),
            "s_template": np.array([1.0, 2.0, 3.0], dtype="<f8"),
            "b_true": np.array([0.1, 0.2, 0.3], dtype="<f8"),
            "n_true": np.array([0.0, 0.0, 0.0], dtype="<f8"),
            "peaks_template": (),
            "provenance": {},
        }
        good["y_observed"] = good["s_template"] + good["b_true"]
        cases = (
            (
                "dtype",
                "axis_cm1.dtype",
                {**good, "axis_cm1": np.array([1.0, 2.0, 3.0], dtype="<f4")},
            ),
            (
                "shape",
                "array shape",
                {**good, "n_true": np.array([0.0, 0.0], dtype="<f8")},
            ),
            (
                "negative background",
                "b_true.nonnegative",
                {**good, "b_true": np.array([0.1, -0.2, 0.3], dtype="<f8")},
            ),
            (
                "decomposition drift",
                "decomposition",
                {
                    **good,
                    "y_observed": np.array([1.1, 2.2, 3.4], dtype="<f8"),
                },
            ),
        )
        for label, expected, payload in cases:
            with self.subTest(label=label), self.assertRaisesRegex(
                SemiSyntheticContractError, expected
            ):
                SemiSyntheticRecord(**payload)


if __name__ == "__main__":
    unittest.main()
