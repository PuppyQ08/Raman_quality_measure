from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.sugar_quantitative import (  # noqa: E402
    D4LoaderValidationError,
    load_d4_protocol_config,
    load_d4_sugar_cohort,
)


CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d4_sugar_protocol.json"
)
AUDIT = ROOT / "reports" / "phase05" / "d4_step01_protocol_audit.json"
ARCHIVE = (
    ROOT
    / "data"
    / "raw"
    / "ramanbench"
    / "cache"
    / "10779223"
    / "Raw data.zip"
)
CONFIG_SHA256 = (
    "69a5dba7ab45cbc421f988439e4f3b3aa9b99505d6ff13cbb3d3b89389d071c8"
)
AUDIT_SHA256 = (
    "7d6ea22b7a96a5e7546d6fd6bc51e68024b59a5214888b0d83c61908aafbd506"
)
MIXTURE_RECORD_IDS_SHA256 = (
    "9b102daf59be106e9fe46ef092373ae5881403ee45b3a033e2ec5c732365f76d"
)
BLANK_RECORD_IDS_SHA256 = (
    "5bda1918b3088228a57c4d97281b285a9dd5debc43db0f76a094620a6ff28ecf"
)
FOLD_RECORD_IDS_SHA256 = (
    "6bad59e135ea206d0c98ae602a89b9c43fa5dd3da015342d504210796bc0b19d",
    "1df98cb38c2ab7495b7682d0cfc58fb8ccadba01696653c60a8bbe518be944f2",
    "d48c935eafe9ea833b562bfd3309215e4bcd2bfdc249d38b6e89c6675825e7b3",
    "8e2e9c243a257d299010ca9a9c4971acab9f86b2dc87fb07ab94532679223daf",
    "fd5bf3debe8ca65a80482547a57ff5d2bb7e4d560cb6c56c4b7635366f4842a0",
)
TARGET_MEMBER = (
    "Raw data/Experimental data from sugar mixtures/Raw data files/"
    "Sugar_Concentrations.csv"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def ids_digest(values: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("utf-8")
    ).hexdigest()


class D4SugarLoaderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.before = (ARCHIVE.stat().st_size, sha256_file(ARCHIVE))
        cls.cohort = load_d4_sugar_cohort(CONFIG, ARCHIVE)
        cls.after = (ARCHIVE.stat().st_size, sha256_file(ARCHIVE))
        cls.audit = json.loads(AUDIT.read_text())

    def test_config_loads_exact_frozen_read_contract(self):
        config = load_d4_protocol_config(CONFIG)

        self.assertEqual(config.sha256, CONFIG_SHA256)
        self.assertEqual(config.byte_count, 4799)
        self.assertEqual(config.audit_sha256, AUDIT_SHA256)
        self.assertEqual(config.expected_mixture_records, 7680)
        self.assertEqual(config.expected_blank_records, 32)
        self.assertEqual(config.target_count, 4)
        self.assertEqual(config.fold_count, 5)
        self.assertEqual(config.records_per_well, 32)
        self.assertTrue(config.raman_data_loader_forbidden)

    def test_loads_exact_read_only_mixture_and_blank_matrices(self):
        cohort = self.cohort

        self.assertEqual(self.after, self.before)
        self.assertEqual(cohort.protocol_config_sha256, CONFIG_SHA256)
        self.assertEqual(cohort.intensity.shape, (7680, 2000))
        self.assertEqual(cohort.intensity.dtype, np.dtype("<f4"))
        self.assertEqual(cohort.targets.shape, (7680, 4))
        self.assertEqual(cohort.targets.dtype, np.dtype("<f8"))
        self.assertEqual(cohort.blank_intensity.shape, (32, 2000))
        self.assertEqual(cohort.blank_targets.shape, (32, 4))
        self.assertEqual(cohort.wavenumber.shape, (2000,))
        self.assertEqual(cohort.wavenumber.dtype, np.dtype("<f4"))
        self.assertEqual(float(cohort.wavenumber[0]), 142.17398071289062)
        self.assertEqual(float(cohort.wavenumber[-1]), 3684.83544921875)
        self.assertEqual(
            hashlib.sha256(cohort.wavenumber.tobytes()).hexdigest(),
            "9b0b88641a767abc74439e21539f7c41366adcf41c9bc3ab60d6de91021431d7",
        )
        for array in (
            cohort.intensity,
            cohort.targets,
            cohort.blank_intensity,
            cohort.blank_targets,
            cohort.wavenumber,
            cohort.rounds,
            cohort.repetitions,
        ):
            self.assertTrue(np.isfinite(array).all())
            self.assertFalse(array.flags.writeable)
        with self.assertRaises(ValueError):
            cohort.intensity[0, 0] = -1
        with self.assertRaises(ValueError):
            cohort.targets[0, 0] = -1

        self.assertEqual(len(cohort.record_ids), 7680)
        self.assertEqual(len(set(cohort.record_ids)), 7680)
        self.assertEqual(len(set(cohort.well_ids)), 240)
        self.assertEqual(ids_digest(cohort.record_ids), MIXTURE_RECORD_IDS_SHA256)
        self.assertEqual(
            ids_digest(cohort.blank_record_ids),
            BLANK_RECORD_IDS_SHA256,
        )
        self.assertEqual(set(cohort.blank_well_ids), {"E1_3"})
        self.assertTrue(np.all(cohort.blank_targets == 0.0))
        self.assertEqual(set(map(int, cohort.rounds)), set(range(1, 9)))
        self.assertEqual(set(map(int, cohort.repetitions)), {1, 2, 3, 4})
        for well_id in set(cohort.well_ids):
            indices = [
                index
                for index, observed in enumerate(cohort.well_ids)
                if observed == well_id
            ]
            self.assertEqual(
                {
                    (
                        int(cohort.rounds[index]),
                        int(cohort.repetitions[index]),
                    )
                    for index in indices
                },
                {
                    (source_round, repetition)
                    for source_round in range(1, 9)
                    for repetition in range(1, 5)
                },
            )
        self.assertEqual(
            cohort.target_names,
            (
                "sucrose_nominal_mol_l",
                "fructose_nominal_mol_l",
                "maltose_nominal_mol_l",
                "glucose_nominal_mol_l",
            ),
        )

    def test_targets_equal_direct_seven_column_source_table(self):
        cohort = self.cohort
        with zipfile.ZipFile(ARCHIVE) as archive:
            rows = list(
                csv.DictReader(
                    archive.read(TARGET_MEMBER)
                    .decode("utf-8-sig")
                    .splitlines()
                )
            )
        expected = {
            row["Cell Number"]: np.array(
                [
                    float(row["Sucrose [ul]"]) / 375.0,
                    float(row["Fructose [ul]"]) / 375.0,
                    float(row["Maltose [ul]"]) / 375.0,
                    float(row["Glucose [ul]"]) / 375.0,
                ],
                dtype="<f8",
            )
            for row in rows
        }

        for index, well_id in enumerate(cohort.well_ids):
            np.testing.assert_array_equal(
                cohort.targets[index],
                expected[well_id],
            )
        self.assertEqual(
            sorted(set(cohort.targets.ravel())),
            [0.0, 0.08, 0.2, 0.32],
        )

    def test_folds_and_seed_rotations_preserve_complete_wells(self):
        cohort = self.cohort
        all_indices = set(range(7680))

        self.assertEqual(len(cohort.folds), 5)
        for fold, indices in enumerate(cohort.folds):
            self.assertFalse(indices.flags.writeable)
            selected = [cohort.record_ids[int(index)] for index in indices]
            self.assertEqual(len(indices), 1536)
            self.assertEqual(ids_digest(selected), FOLD_RECORD_IDS_SHA256[fold])
            counts = {
                well_id: sum(
                    cohort.well_ids[int(index)] == well_id
                    for index in indices
                )
                for well_id in set(
                    cohort.well_ids[int(index)] for index in indices
                )
            }
            self.assertEqual(len(counts), 48)
            self.assertEqual(set(counts.values()), {32})

        self.assertEqual(len(cohort.splits), 5)
        for seed, split in enumerate(cohort.splits):
            train = set(map(int, split.train_indices))
            validation = set(map(int, split.validation_indices))
            test = set(map(int, split.test_indices))
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)
            self.assertEqual(train | validation | test, all_indices)
            self.assertEqual(len(train), 4608)
            self.assertEqual(len(validation), 1536)
            self.assertEqual(len(test), 1536)
            self.assertEqual(split.seed, seed)
            self.assertEqual(split.test_fold, seed)
            self.assertEqual(split.validation_fold, (seed + 1) % 5)
            for well_id in set(cohort.well_ids):
                memberships = {
                    "train": any(cohort.well_ids[index] == well_id for index in train),
                    "validation": any(
                        cohort.well_ids[index] == well_id
                        for index in validation
                    ),
                    "test": any(cohort.well_ids[index] == well_id for index in test),
                }
                self.assertEqual(sum(memberships.values()), 1)

    def test_rejects_mutated_config_without_reading_source(self):
        changed = json.loads(CONFIG.read_text())
        changed["cohort"]["record_count"] = 7679
        path = ROOT / "tests" / ".d4-mutated-config.json"
        self.addCleanup(path.unlink, missing_ok=True)
        path.write_text(
            json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n"
        )

        with self.assertRaisesRegex(
            D4LoaderValidationError,
            "config identity",
        ):
            load_d4_protocol_config(path)


if __name__ == "__main__":
    unittest.main()
