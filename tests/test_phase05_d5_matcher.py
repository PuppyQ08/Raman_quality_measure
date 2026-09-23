from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.rruff import (  # noqa: E402
    D5LibraryQuerySplit,
    D5RawCohort,
)
from rpe.downstream.rruff_matching import (  # noqa: E402
    D5ConditionMatrix,
    D5MatchingValidationError,
    match_d5_condition,
    prepare_d5_conditions,
)


CONFIG_SHA256 = (
    "94f59739a2e3583c6ae33ab5f4ab489cb1f26d3c9f8cdeb689a2448c9c01e009"
)
SG_CONFIG = (
    ROOT
    / "experiments"
    / "phase05"
    / "configs"
    / "d1_sg11_poly3_interp.json"
)
CONTROL = "raw_aligned_control"
SG = "raw_aligned_plus_sg11"


def read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def synthetic_cohort(
    intensity: np.ndarray,
    *,
    class_labels: np.ndarray | None = None,
    group_ids: tuple[str, ...] | None = None,
    split: D5LibraryQuerySplit | None = None,
) -> D5RawCohort:
    row_count = intensity.shape[0]
    if class_labels is None:
        class_labels = np.arange(row_count, dtype="<i8")
    if group_ids is None:
        group_ids = tuple(f"group-{index:02d}" for index in range(row_count))
    if split is None:
        split = D5LibraryQuerySplit(
            seed=0,
            query_indices=read_only(np.array([0], dtype="<i8")),
            library_indices=read_only(
                np.arange(1, row_count, dtype="<i8")
            ),
            split_sha256="0" * 64,
        )
    return D5RawCohort(
        protocol_config_sha256=CONFIG_SHA256,
        dataset_id="rruff_raman_raw",
        intensity=read_only(np.asarray(intensity, dtype="<f4")),
        wavenumber=read_only(
            np.arange(200.0, 1800.0 + 1.0, 2.0, dtype="<f4")
        ),
        class_labels=read_only(np.asarray(class_labels, dtype="<i8")),
        record_ids=tuple(f"record-{index:02d}" for index in range(row_count)),
        mineral_names=tuple(
            f"mineral-{int(label)}" for label in class_labels
        ),
        rruff_ids=tuple(f"R{index:06d}" for index in range(row_count)),
        pin_ids=(None,) * row_count,
        group_ids=group_ids,
        splits=(split,),
    )


def normalized_vector(
    coordinate: int,
    *,
    size: int = 801,
) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float64)
    vector[coordinate] = 1.0
    return vector


def cosine_vector(
    cosine: float,
    *,
    size: int = 801,
) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float64)
    vector[2] = cosine
    vector[3] = math.sqrt(1.0 - cosine**2)
    return vector


class D5ConditionPreparationTest(unittest.TestCase):
    def test_applies_frozen_sg_to_same_rows_then_l2_normalizes(self):
        coordinate = np.linspace(-1.0, 1.0, 801, dtype=np.float32)
        intensity = np.stack(
            (
                np.float32(3.0)
                + coordinate
                + np.float32(0.2) * np.sin(np.float32(31.0) * coordinate),
                np.float32(2.0)
                - np.float32(0.5) * coordinate
                + np.float32(0.1) * np.cos(np.float32(27.0) * coordinate),
            )
        ).astype("<f4")
        before = intensity.copy()
        cohort = synthetic_cohort(intensity)

        conditions = prepare_d5_conditions(cohort, SG_CONFIG)

        self.assertEqual(
            tuple(condition.condition_id for condition in conditions),
            (CONTROL, SG),
        )
        control, sg = conditions
        self.assertEqual(
            control.protocol_config_sha256,
            CONFIG_SHA256,
        )
        self.assertEqual(
            sg.protocol_config_sha256,
            CONFIG_SHA256,
        )
        self.assertEqual(control.record_ids, cohort.record_ids)
        self.assertEqual(sg.record_ids, cohort.record_ids)
        np.testing.assert_array_equal(cohort.intensity, before)
        expected_control = np.asarray(before, dtype=np.float64)
        expected_control /= np.linalg.norm(
            expected_control,
            axis=1,
            keepdims=True,
        )
        expected_sg = np.asarray(
            savgol_filter(
                before,
                window_length=11,
                polyorder=3,
                deriv=0,
                axis=-1,
                mode="interp",
            ),
            dtype="<f4",
        ).astype(np.float64)
        expected_sg /= np.linalg.norm(expected_sg, axis=1, keepdims=True)
        np.testing.assert_allclose(
            control.values,
            expected_control,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            sg.values,
            expected_sg,
            rtol=0.0,
            atol=1e-15,
        )
        self.assertEqual(control.values.dtype, np.dtype("<f8"))
        self.assertEqual(sg.values.dtype, np.dtype("<f8"))
        self.assertFalse(control.values.flags.writeable)
        self.assertFalse(sg.values.flags.writeable)
        self.assertFalse(np.shares_memory(control.values, cohort.intensity))
        self.assertFalse(np.shares_memory(sg.values, cohort.intensity))
        np.testing.assert_allclose(
            np.linalg.norm(control.values, axis=1),
            np.ones(2),
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            np.linalg.norm(sg.values, axis=1),
            np.ones(2),
            rtol=0.0,
            atol=1e-15,
        )

    def test_rejects_zero_norm_nonfinite_or_wrong_cohort_identity(self):
        valid = np.ones((2, 801), dtype="<f4")
        cases = (
            (
                "zero L2 norm",
                synthetic_cohort(
                    np.stack(
                        (np.zeros(801, dtype="<f4"), valid[1])
                    )
                ),
            ),
            (
                "finite",
                synthetic_cohort(
                    np.stack(
                        (
                            np.full(801, np.nan, dtype="<f4"),
                            valid[1],
                        )
                    )
                ),
            ),
        )
        for expected, cohort in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    D5MatchingValidationError,
                    expected,
                ):
                    prepare_d5_conditions(cohort, SG_CONFIG)

        wrong = synthetic_cohort(valid)
        wrong = D5RawCohort(
            **{
                **wrong.__dict__,
                "protocol_config_sha256": "f" * 64,
            }
        )
        with self.assertRaisesRegex(
            D5MatchingValidationError,
            "protocol config",
        ):
            prepare_d5_conditions(wrong, SG_CONFIG)

        wrong_axis = synthetic_cohort(valid)
        shifted_axis = wrong_axis.wavenumber.copy()
        shifted_axis += np.float32(1.0)
        shifted_axis.setflags(write=False)
        wrong_axis = D5RawCohort(
            **{
                **wrong_axis.__dict__,
                "wavenumber": shifted_axis,
            }
        )
        with self.assertRaisesRegex(
            D5MatchingValidationError,
            "frozen wavenumber",
        ):
            prepare_d5_conditions(wrong_axis, SG_CONFIG)


class D5CosineMatcherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        vectors = [
            normalized_vector(0),
            normalized_vector(1),
            normalized_vector(2),
            cosine_vector(0.9),
            normalized_vector(0),
            cosine_vector(0.8),
            normalized_vector(1),
            cosine_vector(0.2),
            cosine_vector(0.7),
            cosine_vector(0.6),
            normalized_vector(0),
            cosine_vector(0.5),
            cosine_vector(0.4),
        ]
        labels = np.array(
            [7, 4, 9, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9],
            dtype="<i8",
        )
        split = D5LibraryQuerySplit(
            seed=0,
            query_indices=read_only(np.array([0, 1, 2], dtype="<i8")),
            library_indices=read_only(
                np.arange(3, len(vectors), dtype="<i8")
            ),
            split_sha256="1" * 64,
        )
        cls.cohort = synthetic_cohort(
            np.asarray(vectors, dtype="<f4"),
            class_labels=labels,
            split=split,
        )
        cls.condition = D5ConditionMatrix(
            condition_id=CONTROL,
            protocol_config_sha256=CONFIG_SHA256,
            record_ids=cls.cohort.record_ids,
            values=read_only(np.asarray(vectors, dtype="<f8")),
        )

    def test_uses_max_record_per_class_unique_ranks_and_class_label_ties(self):
        result = match_d5_condition(
            self.cohort,
            self.cohort.splits[0],
            self.condition,
        )

        self.assertEqual(result.condition_id, CONTROL)
        self.assertEqual(
            result.protocol_config_sha256,
            CONFIG_SHA256,
        )
        self.assertEqual(
            result.split_sha256,
            self.cohort.splits[0].split_sha256,
        )
        self.assertEqual(
            result.query_record_ids,
            ("record-00", "record-01", "record-02"),
        )
        np.testing.assert_array_equal(
            result.query_indices,
            np.array([0, 1, 2], dtype="<i8"),
        )
        np.testing.assert_array_equal(
            result.true_class_labels,
            np.array([7, 4, 9], dtype="<i8"),
        )
        np.testing.assert_array_equal(
            result.ranked_class_labels[0],
            np.array([2, 7, 1, 3, 4, 5, 6, 8, 9], dtype="<i8"),
        )
        np.testing.assert_array_equal(
            result.ranked_class_labels[1],
            np.array([4, 1, 2, 3, 5, 6, 7, 8, 9], dtype="<i8"),
        )
        np.testing.assert_array_equal(
            result.ranked_class_labels[2],
            np.array([1, 3, 5, 6, 8, 9, 2, 4, 7], dtype="<i8"),
        )
        np.testing.assert_allclose(
            result.ranked_class_scores[2, :6],
            np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            rtol=0.0,
            atol=1e-15,
        )
        self.assertGreaterEqual(float(result.ranked_class_scores.min()), -1.0)
        self.assertLessEqual(float(result.ranked_class_scores.max()), 1.0)
        np.testing.assert_array_equal(
            result.top1_correct,
            np.array([False, True, False]),
        )
        np.testing.assert_array_equal(
            result.top5_correct,
            np.array([True, True, False]),
        )
        for row in result.ranked_class_labels:
            self.assertEqual(len(row), len(set(int(value) for value in row)))
        for array in (
            result.query_indices,
            result.true_class_labels,
            result.ranked_class_labels,
            result.ranked_class_scores,
            result.top1_correct,
            result.top5_correct,
        ):
            self.assertFalse(array.flags.writeable)

    def test_rejects_group_leakage_missing_true_class_or_invalid_condition(self):
        split = self.cohort.splits[0]
        leaking_groups = list(self.cohort.group_ids)
        leaking_groups[3] = leaking_groups[0]
        leaking = D5RawCohort(
            **{
                **self.cohort.__dict__,
                "group_ids": tuple(leaking_groups),
            }
        )
        with self.assertRaisesRegex(
            D5MatchingValidationError,
            "leakage group",
        ):
            match_d5_condition(leaking, split, self.condition)

        missing_class = D5LibraryQuerySplit(
            seed=0,
            query_indices=read_only(np.array([0, 10], dtype="<i8")),
            library_indices=read_only(
                np.array(
                    [
                        index
                        for index in range(len(self.cohort.record_ids))
                        if int(self.cohort.class_labels[index]) != 7
                    ],
                    dtype="<i8",
                )
            ),
            split_sha256="2" * 64,
        )
        missing_class_cohort = D5RawCohort(
            **{
                **self.cohort.__dict__,
                "splits": (missing_class,),
            }
        )
        with self.assertRaisesRegex(
            D5MatchingValidationError,
            "true query class",
        ):
            match_d5_condition(
                missing_class_cohort,
                missing_class,
                self.condition,
            )

        invalid_values = self.condition.values.copy()
        invalid_values[0] *= 2.0
        invalid = D5ConditionMatrix(
            condition_id=CONTROL,
            protocol_config_sha256=CONFIG_SHA256,
            record_ids=self.cohort.record_ids,
            values=read_only(invalid_values),
        )
        with self.assertRaisesRegex(
            D5MatchingValidationError,
            "unit L2 norm",
        ):
            match_d5_condition(self.cohort, split, invalid)

    def test_rejects_incomplete_record_partition(self):
        incomplete = D5LibraryQuerySplit(
            seed=0,
            query_indices=read_only(np.array([0], dtype="<i8")),
            library_indices=read_only(
                np.arange(3, len(self.cohort.record_ids), dtype="<i8")
            ),
            split_sha256="3" * 64,
        )
        incomplete_cohort = D5RawCohort(
            **{
                **self.cohort.__dict__,
                "splits": (incomplete,),
            }
        )

        with self.assertRaisesRegex(
            D5MatchingValidationError,
            "complete partition",
        ):
            match_d5_condition(
                incomplete_cohort,
                incomplete,
                self.condition,
            )

    def test_rejects_unfrozen_split_or_condition_row_identity(self):
        frozen = self.cohort.splits[0]
        unfrozen = D5LibraryQuerySplit(
            seed=frozen.seed,
            query_indices=read_only(frozen.query_indices[::-1].copy()),
            library_indices=read_only(frozen.library_indices[::-1].copy()),
            split_sha256="4" * 64,
        )
        with self.assertRaisesRegex(
            D5MatchingValidationError,
            "frozen split",
        ):
            match_d5_condition(
                self.cohort,
                unfrozen,
                self.condition,
            )

        reordered = D5ConditionMatrix(
            condition_id=CONTROL,
            protocol_config_sha256=CONFIG_SHA256,
            record_ids=tuple(reversed(self.cohort.record_ids)),
            values=self.condition.values,
        )
        with self.assertRaisesRegex(
            D5MatchingValidationError,
            "condition row identity",
        ):
            match_d5_condition(
                self.cohort,
                frozen,
                reordered,
            )


if __name__ == "__main__":
    unittest.main()
