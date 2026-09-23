from __future__ import annotations

import copy
import hashlib
import json
import re
import sys
import tempfile
import unittest
from collections import UserDict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase1_config import (  # noqa: E402
    ArtifactIdentity,
    LocatedArtifactIdentity,
    PHASE1_DATA_CODE_PATHS,
    PHASE1_SUMMARY_CODE_PATHS,
    Phase1CoreConfig,
    Phase1ConfigError,
    code_snapshot_digest,
    code_snapshot_document,
    load_phase1_core_config,
    scientific_config_bytes,
)


CONFIG = ROOT / "experiments/phase1/configs/rruff_raw_core10k_v1.json"


class Phase1CoreConfigTest(unittest.TestCase):
    def test_frozen_code_path_constants_are_exact_and_lexicographically_sorted(self):
        expected_data_paths = (
            "rpe/evaluation/__init__.py",
            "rpe/evaluation/contracts.py",
            "rpe/io/__init__.py",
            "rpe/io/perturbed_schema.py",
            "rpe/io/perturbed_store.py",
            "rpe/io/schema.py",
            "rpe/io/store.py",
            "rpe/perturb/__init__.py",
            "rpe/perturb/axis_transform.py",
            "rpe/perturb/baseline_distortion.py",
            "rpe/perturb/baseline_residual.py",
            "rpe/perturb/contracts.py",
            "rpe/perturb/correlated_noise.py",
            "rpe/perturb/gaussian_noise.py",
            "rpe/perturb/peak_family.py",
            "rpe/perturb/sweep.py",
            "rpe/runner/__init__.py",
            "rpe/runner/phase1_config.py",
            "rpe/runner/phase1_formal.py",
            "rpe/runner/phase1_gates.py",
            "rpe/runner/phase1_perturbations.py",
            "rpe/runner/phase1_selection.py",
            "rpe/runner/phase1_types.py",
        )
        expected_summary_paths = ()
        self.assertEqual(PHASE1_DATA_CODE_PATHS, expected_data_paths)
        self.assertEqual(PHASE1_SUMMARY_CODE_PATHS, expected_summary_paths)
        self.assertEqual(
            PHASE1_DATA_CODE_PATHS,
            tuple(sorted(PHASE1_DATA_CODE_PATHS)),
        )
        self.assertEqual(
            PHASE1_SUMMARY_CODE_PATHS,
            tuple(sorted(PHASE1_SUMMARY_CODE_PATHS)),
        )

    def test_retained_config_has_exact_file_and_scientific_identities(self):
        raw = CONFIG.read_bytes()
        self.assertEqual(len(raw), 2350)
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(),
            "6fc3502d44c22df223e6df03c6f2e1e257c1de53a15539d3f64409cfe9e141cd",
        )
        config = load_phase1_core_config(CONFIG)
        self.assertEqual(config.scientific_config_byte_count, 2122)
        self.assertEqual(
            config.scientific_config_sha256,
            "f44a1b51a9ff0eb7f4221d45d452ce7354bae5734432e54a0fda87617babd37e",
        )
        self.assertEqual(config.subset_size, 10_000)
        self.assertEqual(config.shard_source_count, 100)
        self.assertEqual(
            config.materialized_perturbation_ids,
            ("p01", "p02", "p03", "p04", "p05", "p08", "p09", "p10", "p11", "p12"),
        )
        self.assertEqual(config.deferred_perturbation_ids, ("p06", "p07"))

    def test_scientific_projection_removes_only_recursive_path_keys(self):
        document = {
            "dataset_path": "/a",
            "sha256": "f" * 64,
            "nested": {"artifact_path": "/b", "pathology": "kept"},
        }
        projected = json.loads(scientific_config_bytes(document))
        self.assertEqual(
            projected,
            {"nested": {"pathology": "kept"}, "sha256": "f" * 64},
        )

    def test_scientific_projection_handles_nested_mappings_and_wraps_bad_inputs(self):
        document = UserDict(
            {
                "dataset_path": "/outer",
                "pathology": "kept",
                "sequence": (
                    UserDict(
                        {
                            "artifact_path": "/inner",
                            "pathology": "still-kept",
                        }
                    ),
                    [
                        UserDict(
                            {
                                "code_path": "/code",
                                "other": {"pathology": "present"},
                            }
                        )
                    ],
                ),
            }
        )
        projected = json.loads(scientific_config_bytes(document))
        self.assertEqual(
            projected,
            {
                "pathology": "kept",
                "sequence": [
                    {"pathology": "still-kept"},
                    [{"other": {"pathology": "present"}}],
                ],
            },
        )

        bad_cases = (
            (
                "nonfinite",
                {"value": float("nan")},
                "document.value",
            ),
            (
                "unsupported value",
                {"value": object()},
                "document.value",
            ),
            (
                "bad mapping key",
                UserDict({1: "bad"}),
                "document",
            ),
        )
        for label, payload, expected in bad_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(Phase1ConfigError, re.escape(expected)):
                    scientific_config_bytes(payload)

    def test_loader_rejects_noncanonical_unknown_missing_nonfinite_or_wrong_identity(
        self,
    ):
        base = json.loads(CONFIG.read_text(encoding="utf-8"))
        cases = (
            (
                "unknown root key",
                "config.keys",
                lambda d: d.__setitem__("unknown", 1),
            ),
            (
                "missing root key",
                "config.keys",
                lambda d: d.pop("schema_version"),
            ),
            (
                "duplicate materialized ID",
                "materialized_perturbation_ids",
                lambda d: d["materialized_perturbation_ids"].append("p01"),
            ),
            (
                "forbidden alpha copy",
                "config.keys",
                lambda d: d.__setitem__("alpha_grid", [0.0, 0.5]),
            ),
            (
                "forbidden seed copy",
                "config.keys",
                lambda d: d.__setitem__("global_seed", 20260817),
            ),
            (
                "wrong arrays hash",
                "source.files.arrays.h5.sha256",
                lambda d: d["source"]["files"]["arrays.h5"].__setitem__(
                    "sha256",
                    "0" * 64,
                ),
            ),
            (
                "wrong arrays bytes",
                "source.files.arrays.h5.byte_count",
                lambda d: d["source"]["files"]["arrays.h5"].__setitem__(
                    "byte_count",
                    1,
                ),
            ),
            (
                "wrong shard size",
                "storage.shard_source_count",
                lambda d: d["storage"].__setitem__("shard_source_count", 99),
            ),
            (
                "wrong subset size",
                "selection.subset_size",
                lambda d: d["selection"].__setitem__("subset_size", 9999),
            ),
            (
                "invalid threshold",
                "core_gate.p05_min_source_fraction",
                lambda d: d["core_gate"].__setitem__(
                    "p05_min_source_fraction",
                    1.1,
                ),
            ),
            (
                "non-string path",
                "source.dataset_path",
                lambda d: d["source"].__setitem__("dataset_path", 1),
            ),
        )
        for label, expected_path, mutate in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                document = copy.deepcopy(base)
                mutate(document)
                path = Path(tmp) / CONFIG.name
                path.write_text(
                    json.dumps(document, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(Phase1ConfigError, expected_path):
                    load_phase1_core_config(path)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / CONFIG.name
            path.write_text('{"schema_version": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(Phase1ConfigError, "config nonfinite"):
                load_phase1_core_config(path)

    def test_code_snapshot_helpers_require_sorted_relative_in_root_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alpha = root / "alpha.txt"
            beta = root / "nested" / "beta.txt"
            beta.parent.mkdir()
            alpha.write_text("alpha\n", encoding="utf-8")
            beta.write_text("beta\n", encoding="utf-8")

            relative_paths = ("alpha.txt", "nested/beta.txt")
            document = code_snapshot_document(root, relative_paths)

            self.assertEqual(tuple(document.keys()), relative_paths)
            self.assertEqual(document["alpha.txt"]["byte_count"], 6)
            self.assertEqual(
                document["alpha.txt"]["sha256"],
                hashlib.sha256(b"alpha\n").hexdigest(),
            )
            self.assertEqual(
                code_snapshot_digest(root, relative_paths),
                hashlib.sha256(
                    b"rpe-phase1-code-snapshot-v1\0"
                    + json.dumps(
                        {
                            key: dict(value)
                            for key, value in document.items()
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8"),
                ).hexdigest(),
            )

            bad_cases = (
                (("nested/beta.txt", "alpha.txt"), "relative_paths"),
                (("alpha.txt", "alpha.txt"), "relative_paths"),
                (("/abs.txt",), "relative_paths[0]"),
                (("missing.txt",), "relative_paths[0]"),
            )
            for paths, expected in bad_cases:
                with self.subTest(paths=paths):
                    with self.assertRaisesRegex(
                        Phase1ConfigError,
                        re.escape(expected),
                    ):
                        code_snapshot_document(root, paths)

    def test_direct_identity_and_core_config_constructors_validate_and_freeze(self):
        source_files = {
            "zeta": ArtifactIdentity(3, "b" * 64),
            "alpha": ArtifactIdentity(2, "a" * 64),
        }
        related = {
            "pair": LocatedArtifactIdentity(
                Path("b/location.json"),
                ArtifactIdentity(4, "c" * 64),
            ),
            "conversion": LocatedArtifactIdentity(
                Path("a/location.json"),
                ArtifactIdentity(5, "d" * 64),
            ),
        }
        core_gate = {"z": 2.0, "a": 1}

        config = Phase1CoreConfig(
            path=Path("config.json"),
            file_byte_count=10,
            file_sha256="1" * 64,
            scientific_config_byte_count=9,
            scientific_config_sha256="2" * 64,
            schema_version="schema",
            experiment_id="exp",
            source_dataset_id="dataset",
            source_dataset_path=Path("data/source"),
            source_record_count=3,
            source_class_count=2,
            source_files=source_files,
            related_provenance=related,
            shared_sweep_path=Path("experiments/shared.json"),
            shared_sweep_identity=ArtifactIdentity(6, "3" * 64),
            subset_size=2,
            selection_algorithm="selection",
            shard_source_count=1,
            materialized_perturbation_ids=("p02", "p01"),
            deferred_perturbation_ids=("p07", "p06"),
            deferred_reason_code="deferred",
            deferred_dependency="dependency",
            peak_not_applicable_reason_codes=("b", "a"),
            distribution_status="status",
            core_gate=core_gate,
        )

        source_files["beta"] = ArtifactIdentity(7, "4" * 64)
        related["other"] = LocatedArtifactIdentity(
            Path("elsewhere.json"),
            ArtifactIdentity(8, "5" * 64),
        )
        core_gate["extra"] = 3

        self.assertEqual(tuple(config.source_files.keys()), ("alpha", "zeta"))
        self.assertEqual(
            tuple(config.related_provenance.keys()),
            ("conversion", "pair"),
        )
        self.assertEqual(tuple(config.core_gate.keys()), ("a", "z"))
        self.assertEqual(config.materialized_perturbation_ids, ("p01", "p02"))
        self.assertEqual(config.deferred_perturbation_ids, ("p06", "p07"))
        self.assertEqual(config.peak_not_applicable_reason_codes, ("a", "b"))
        with self.assertRaises(TypeError):
            config.source_files["new"] = ArtifactIdentity(1, "6" * 64)
        with self.assertRaises(TypeError):
            config.core_gate["new"] = 1

        bad_cases = (
            ("artifact byte_count", lambda: ArtifactIdentity(-1, "a" * 64), "byte_count"),
            ("artifact sha", lambda: ArtifactIdentity(1, "A" * 64), "sha256"),
            (
                "located path",
                lambda: LocatedArtifactIdentity("not-a-path", ArtifactIdentity(1, "a" * 64)),
                "path",
            ),
            (
                "located identity",
                lambda: LocatedArtifactIdentity(Path("ok"), object()),
                "identity",
            ),
            (
                "config path",
                lambda: Phase1CoreConfig(
                    path="config.json",
                    file_byte_count=1,
                    file_sha256="1" * 64,
                    scientific_config_byte_count=1,
                    scientific_config_sha256="2" * 64,
                    schema_version="schema",
                    experiment_id="exp",
                    source_dataset_id="dataset",
                    source_dataset_path=Path("data/source"),
                    source_record_count=1,
                    source_class_count=1,
                    source_files={},
                    related_provenance={},
                    shared_sweep_path=Path("shared.json"),
                    shared_sweep_identity=ArtifactIdentity(1, "3" * 64),
                    subset_size=1,
                    selection_algorithm="selection",
                    shard_source_count=1,
                    materialized_perturbation_ids=(),
                    deferred_perturbation_ids=(),
                    deferred_reason_code="reason",
                    deferred_dependency="dependency",
                    peak_not_applicable_reason_codes=(),
                    distribution_status="status",
                    core_gate={},
                ),
                "path",
            ),
            (
                "config bad source mapping value",
                lambda: Phase1CoreConfig(
                    path=Path("config.json"),
                    file_byte_count=1,
                    file_sha256="1" * 64,
                    scientific_config_byte_count=1,
                    scientific_config_sha256="2" * 64,
                    schema_version="schema",
                    experiment_id="exp",
                    source_dataset_id="dataset",
                    source_dataset_path=Path("data/source"),
                    source_record_count=1,
                    source_class_count=1,
                    source_files={"bad": object()},
                    related_provenance={},
                    shared_sweep_path=Path("shared.json"),
                    shared_sweep_identity=ArtifactIdentity(1, "3" * 64),
                    subset_size=1,
                    selection_algorithm="selection",
                    shard_source_count=1,
                    materialized_perturbation_ids=("p01",),
                    deferred_perturbation_ids=("p06",),
                    deferred_reason_code="reason",
                    deferred_dependency="dependency",
                    peak_not_applicable_reason_codes=("reason",),
                    distribution_status="status",
                    core_gate={"a": 1},
                ),
                "source_files.bad",
            ),
            (
                "config bad core gate value",
                lambda: Phase1CoreConfig(
                    path=Path("config.json"),
                    file_byte_count=1,
                    file_sha256="1" * 64,
                    scientific_config_byte_count=1,
                    scientific_config_sha256="2" * 64,
                    schema_version="schema",
                    experiment_id="exp",
                    source_dataset_id="dataset",
                    source_dataset_path=Path("data/source"),
                    source_record_count=1,
                    source_class_count=1,
                    source_files={},
                    related_provenance={},
                    shared_sweep_path=Path("shared.json"),
                    shared_sweep_identity=ArtifactIdentity(1, "3" * 64),
                    subset_size=1,
                    selection_algorithm="selection",
                    shard_source_count=1,
                    materialized_perturbation_ids=("p01",),
                    deferred_perturbation_ids=("p06",),
                    deferred_reason_code="reason",
                    deferred_dependency="dependency",
                    peak_not_applicable_reason_codes=("reason",),
                    distribution_status="status",
                    core_gate={"a": object()},
                ),
                "core_gate.a",
            ),
        )
        for label, factory, expected in bad_cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(Phase1ConfigError, re.escape(expected)):
                    factory()


if __name__ == "__main__":
    unittest.main()
