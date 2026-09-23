from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.methods import (  # noqa: E402
    AvailabilityStatus,
    Phase3CatalogError,
    TaskLine,
    audit_classical_catalog_availability,
    build_classical_catalog_document,
    canonical_catalog_bytes,
    load_classical_catalog,
)


CATALOG = ROOT / "experiments" / "phase3" / "configs" / "classical_system_catalog_v1.json"
LOCK = ROOT / "env" / "phase3-requirements.lock"
OLD_LOCK = ROOT / "env" / "requirements.lock"
DESIGN = ROOT / "reports" / "phase3" / "step01_classical_system_registry_design.md"
FALLBACK = ROOT / "reports" / "phase2" / "step04_phase2_fallback_decision.md"


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


class Phase3ClassicalCatalogTest(unittest.TestCase):
    def test_catalog_has_exact_identity_counts_and_separated_views(self) -> None:
        raw = CATALOG.read_bytes()
        document = json.loads(raw)
        self.assertEqual(raw, canonical(document))
        self.assertEqual(raw, canonical_catalog_bytes(build_classical_catalog_document(ROOT)))
        catalog = load_classical_catalog(CATALOG)
        self.assertEqual(catalog.byte_count, len(raw))
        self.assertEqual(catalog.sha256, hashlib.sha256(raw).hexdigest())
        self.assertRegex(catalog.catalog_id, r"^[0-9a-f]{64}$")
        self.assertEqual(len(catalog.systems), 306)
        self.assertEqual(len({system.system_id for system in catalog.systems}), 306)
        self.assertEqual(
            {
                view.task_line.value: (view.system_count, view.family_count)
                for view in catalog.views
            },
            {
                "baseline_correction": (210, 14),
                "denoising": (60, 5),
                "peak_detection": (36, 3),
            },
        )
        self.assertEqual(
            tuple(system.system_id for system in catalog.systems),
            tuple(sorted(system.system_id for system in catalog.systems)),
        )
        self.assertTrue(all(system.method_seed is None for system in catalog.systems))
        self.assertTrue(
            all(len(system.ordered_composition) == 1 for system in catalog.systems)
        )
        self.assertTrue(
            all(
                system.availability is AvailabilityStatus.IMPLEMENTATION_MISSING
                for system in catalog.systems
            )
        )
        self.assertEqual(
            {view.task_line for view in catalog.views},
            {
                TaskLine.BASELINE_CORRECTION,
                TaskLine.DENOISING,
                TaskLine.PEAK_DETECTION,
            },
        )
        status_by_task = {
            view.task_line: view.phase5_power_status for view in catalog.views
        }
        self.assertEqual(
            status_by_task[TaskLine.BASELINE_CORRECTION],
            "planned_target_k210_not_yet_powered",
        )
        for system in catalog.systems:
            if system.task_line is TaskLine.PEAK_DETECTION:
                self.assertEqual(
                    system.downstream_eligibility, ("peak_assignment_benchmark",)
                )
                self.assertEqual(
                    system.protocol_eligibility, ("main_table", "phase5")
                )
            else:
                self.assertEqual(
                    system.downstream_eligibility, ("D1", "D2", "D3", "D4", "D5")
                )
                self.assertEqual(
                    system.protocol_eligibility,
                    ("main_table", "phase5", "protocol_a", "protocol_b"),
                )

    def test_literal_grid_boundaries_and_fallback_eligibility_are_frozen(self) -> None:
        catalog = load_classical_catalog(CATALOG)
        by_family: dict[tuple[str, str], list[object]] = {}
        for system in catalog.systems:
            by_family.setdefault((system.task_line.value, system.family_id), []).append(system)

        baseline_expected = {
            "asls",
            "iasls",
            "airpls",
            "arpls",
            "drpls",
            "iarpls",
            "aspls",
            "psalsa",
            "modpoly",
            "imodpoly",
            "penalized_poly",
            "snip",
            "morphological",
            "beads",
        }
        self.assertEqual(
            {family for task, family in by_family if task == "baseline_correction"},
            baseline_expected,
        )
        self.assertTrue(
            all(
                len(values) == (15 if task == "baseline_correction" else 12)
                for (task, _), values in by_family.items()
            )
        )
        airpls = by_family[("baseline_correction", "airpls")]
        self.assertEqual(
            sorted(system.hyperparameters["lam"] for system in airpls),
            [
                100.0,
                300.0,
                1000.0,
                3000.0,
                10000.0,
                30000.0,
                100000.0,
                300000.0,
                1000000.0,
                3000000.0,
                10000000.0,
                30000000.0,
                100000000.0,
                300000000.0,
                1000000000.0,
            ],
        )
        morphology = by_family[("baseline_correction", "morphological")]
        self.assertEqual(
            sorted(system.hyperparameters["half_window_cm1"] for system in morphology),
            [8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 80.0, 96.0, 128.0, 160.0, 192.0, 256.0, 320.0, 400.0],
        )
        wavelet = by_family[("denoising", "wavelet")]
        self.assertEqual(
            {
                (
                    system.hyperparameters["wavelet"],
                    system.hyperparameters["threshold_mode"],
                    system.hyperparameters["threshold_strategy"],
                )
                for system in wavelet
            },
            {
                (wavelet_name, mode, strategy)
                for wavelet_name in ("db4", "db6", "sym8")
                for mode in ("soft", "hard")
                for strategy in ("universal_mad", "bayes_shrink")
            },
        )
        find_peaks = by_family[("peak_detection", "find_peaks")]
        self.assertEqual(
            sorted(system.hyperparameters["prominence_fraction"] for system in find_peaks),
            [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        )
        mspd = by_family[("peak_detection", "mspd")]
        self.assertEqual(
            {
                (system.hyperparameters["max_scale_cm1"], system.hyperparameters["ridge_vote_fraction"])
                for system in mspd
            },
            {
                (scale, fraction)
                for scale in (4.0, 8.0, 12.0, 16.0)
                for fraction in (0.25, 0.5, 0.75)
            },
        )
        for system in catalog.systems:
            if system.task_line is TaskLine.BASELINE_CORRECTION:
                self.assertNotIn(
                    "semisynthetic_clean_gt_fidelity", system.metric_eligibility
                )
                self.assertEqual(
                    system.evidence["phase2_fallback"],
                    "downstream_reference_free_half_split_only",
                )

    def test_lock_and_bound_report_identities_are_exact(self) -> None:
        lock = LOCK.read_bytes()
        self.assertEqual(
            lock,
            b"h5py==3.16.0\n"
            b"joblib==1.5.3\n"
            b"matplotlib==3.11.1\n"
            b"numpy==2.5.2\n"
            b"optuna==4.9.0\n"
            b"pandas==3.0.5\n"
            b"pybaselines==1.2.1\n"
            b"PyWavelets==1.9.0\n"
            b"scikit-learn==1.9.0\n"
            b"scipy==1.18.0\n",
        )
        catalog = load_classical_catalog(CATALOG)
        self.assertEqual(catalog.dependency_lock.byte_count, len(lock))
        self.assertEqual(
            catalog.dependency_lock.sha256, hashlib.sha256(lock).hexdigest()
        )
        document = json.loads(CATALOG.read_bytes())
        for name, path in (("design", DESIGN), ("phase2_fallback", FALLBACK)):
            identity = document["evidence"][name]
            raw = path.read_bytes()
            self.assertEqual(identity["byte_count"], len(raw))
            self.assertEqual(identity["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertEqual(
            hashlib.sha256(OLD_LOCK.read_bytes()).hexdigest(),
            "3057d1a4f5198139b1e57dd9cca92b05e9c362a5f10bc87fad3a6041c54a9963",
        )

    def test_availability_audit_is_truthful_and_does_not_promote_systems(self) -> None:
        catalog = load_classical_catalog(CATALOG)
        audit = audit_classical_catalog_availability(catalog)
        self.assertEqual(audit["planned_system_count"], 306)
        self.assertEqual(audit["catalog_runnable_system_count"], 0)
        self.assertEqual(audit["installed"]["pybaselines"], "1.2.1")
        self.assertEqual(audit["installed"]["scipy"], "1.18.0")
        self.assertEqual(audit["installed"]["scikit-learn"], "1.9.0")
        self.assertEqual(audit["installed"]["joblib"], "1.5.3")
        self.assertEqual(audit["installed"]["optuna"], "4.9.0")
        self.assertEqual(audit["installed"]["PyWavelets"], "1.9.0")
        self.assertTrue(audit["wrappers"]["baseline"])
        self.assertTrue(audit["wrappers"]["denoising"])
        self.assertTrue(audit["wrappers"]["peaks"])
        self.assertTrue(
            all(
                system.availability is AvailabilityStatus.IMPLEMENTATION_MISSING
                for system in catalog.systems
            )
        )
        with self.assertRaises(TypeError):
            audit["installed"]["optuna"] = "present"

    def test_catalog_contract_is_recursively_immutable(self) -> None:
        catalog = load_classical_catalog(CATALOG)
        system = catalog.systems[0]
        with self.assertRaises(TypeError):
            system.hyperparameters["new"] = 1
        with self.assertRaises(TypeError):
            system.input_contract["axis"] = "changed"
        with self.assertRaises(TypeError):
            system.evidence["phase2_fallback"] = "changed"

    def test_loader_rejects_semantic_identity_and_count_drift(self) -> None:
        original = json.loads(CATALOG.read_bytes())

        def assert_rejected(mutator, expected: str) -> None:
            document = copy.deepcopy(original)
            mutator(document)
            with tempfile.TemporaryDirectory() as temp_dir:
                path = Path(temp_dir) / "catalog.json"
                path.write_bytes(canonical(document))
                with self.assertRaisesRegex(Phase3CatalogError, expected):
                    load_classical_catalog(path, enforce_frozen_identity=False)

        assert_rejected(
            lambda d: d["systems"][1].__setitem__(
                "system_id", d["systems"][0]["system_id"]
            ),
            "system identity",
        )
        assert_rejected(
            lambda d: d["systems"][0]["hyperparameters"].__setitem__("lam", 17.0),
            "system identity",
        )
        assert_rejected(
            lambda d: d["systems"][0].__setitem__("method_seed", 7),
            "method_seed",
        )
        assert_rejected(
            lambda d: d["systems"][0].__setitem__("availability", "reported_only"),
            "availability",
        )
        assert_rejected(lambda d: d["systems"].pop(), "system counts")
        assert_rejected(
            lambda d: d["views"][0].__setitem__("system_count", 306),
            "view counts",
        )

    def test_loader_rejects_noncanonical_json_and_lock_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "catalog.json"
            path.write_text(json.dumps(json.loads(CATALOG.read_bytes()), indent=2), encoding="utf-8")
            with self.assertRaisesRegex(Phase3CatalogError, "canonical"):
                load_classical_catalog(path, enforce_frozen_identity=False)

        document = json.loads(CATALOG.read_bytes())
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            lock_path = project_root / document["dependency_lock"]["path"]
            lock_path.parent.mkdir(parents=True)
            lock_path.write_bytes(b"x" * document["dependency_lock"]["byte_count"])
            for key in ("design", "phase2_fallback"):
                source = ROOT / document["evidence"][key]["path"]
                target = project_root / document["evidence"][key]["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(source)
            with self.assertRaisesRegex(Phase3CatalogError, "dependency_lock.*SHA256"):
                load_classical_catalog(CATALOG, project_root=project_root)

    def test_loader_rejects_availability_snapshot_drift_without_identity_gate(self) -> None:
        document = json.loads(CATALOG.read_bytes())
        document["availability_audit"]["installed"]["optuna"] = "4.9.0"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "catalog.json"
            path.write_bytes(canonical(document))
            with self.assertRaisesRegex(Phase3CatalogError, "availability_audit"):
                load_classical_catalog(path, enforce_frozen_identity=False)


if __name__ == "__main__":
    unittest.main()
