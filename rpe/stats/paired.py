from __future__ import annotations

import itertools
import numbers
from typing import Mapping, Sequence

import numpy as np


EXPECTED_SAMPLE_SIZE = 5


class PairedStatisticsValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def _vector(path: str, values: Sequence[float]) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise PairedStatisticsValidationError(path, "must be a sequence")
    if len(values) != EXPECTED_SAMPLE_SIZE:
        raise PairedStatisticsValidationError(
            path,
            f"must contain exactly {EXPECTED_SAMPLE_SIZE} values",
        )
    if any(
        isinstance(value, bool) or not isinstance(value, numbers.Real)
        for value in values
    ):
        raise PairedStatisticsValidationError(
            path,
            "must contain only real non-Boolean numbers",
        )
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise PairedStatisticsValidationError(
            f"{path} finite",
            "must contain only finite values",
        )
    return array


def paired_seed_bootstrap(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int,
    confidence_level: float,
    random_seed: int,
    scale: float,
) -> Mapping[str, object]:
    left_array = _vector("left", left)
    right_array = _vector("right", right)
    if (
        isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples <= 0
    ):
        raise PairedStatisticsValidationError(
            "resamples",
            "must be a positive integer",
        )
    if (
        isinstance(random_seed, bool)
        or not isinstance(random_seed, int)
        or random_seed < 0
    ):
        raise PairedStatisticsValidationError(
            "random_seed",
            "must be a non-negative integer",
        )
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, numbers.Real)
        or not 0.0 < float(confidence_level) < 1.0
    ):
        raise PairedStatisticsValidationError(
            "confidence_level",
            "must be a real number strictly between zero and one",
        )
    if (
        isinstance(scale, bool)
        or not isinstance(scale, numbers.Real)
        or not np.isfinite(float(scale))
        or float(scale) <= 0.0
    ):
        raise PairedStatisticsValidationError(
            "scale",
            "must be a positive finite real number",
        )
    generator = np.random.default_rng(random_seed)
    indices = generator.integers(
        0,
        EXPECTED_SAMPLE_SIZE,
        size=(resamples, EXPECTED_SAMPLE_SIZE),
    )
    left_means = left_array[indices].mean(axis=1) * float(scale)
    right_means = right_array[indices].mean(axis=1) * float(scale)
    paired_differences = right_array - left_array
    differences = (
        paired_differences[indices].mean(axis=1) * float(scale)
    )
    alpha = (1.0 - float(confidence_level)) / 2.0

    def interval(values: np.ndarray) -> list[float]:
        lower, upper = np.quantile(
            values,
            [alpha, 1.0 - alpha],
        )
        return [float(lower), float(upper)]

    return {
        "confidence_level": float(confidence_level),
        "difference_ci": interval(differences),
        "difference_mean": float(
            paired_differences.mean() * float(scale)
        ),
        "interval": "percentile",
        "left_ci": interval(left_means),
        "left_mean": float(left_array.mean() * float(scale)),
        "paired": True,
        "random_seed": random_seed,
        "resamples": resamples,
        "right_ci": interval(right_means),
        "right_mean": float(right_array.mean() * float(scale)),
        "sample_size": EXPECTED_SAMPLE_SIZE,
        "unit": "seed",
    }


def exact_two_sided_sign_flip(
    differences: Sequence[float],
) -> Mapping[str, object]:
    difference_array = _vector("differences", differences)
    observed = float(difference_array.mean())
    tolerance = 1e-15
    permuted = []
    for signs in itertools.product((-1.0, 1.0), repeat=EXPECTED_SAMPLE_SIZE):
        permuted.append(
            float(
                np.mean(
                    difference_array
                    * np.asarray(signs, dtype=np.float64)
                )
            )
        )
    extreme = sum(
        abs(value) + tolerance >= abs(observed)
        for value in permuted
    )
    patterns = len(permuted)
    return {
        "alternative": "two-sided",
        "extreme_patterns": extreme,
        "observed_mean": observed,
        "p_value": extreme / patterns,
        "patterns": patterns,
        "sample_size": EXPECTED_SAMPLE_SIZE,
        "unit": "seed",
        "zero_difference": "included",
    }


__all__ = [
    "PairedStatisticsValidationError",
    "exact_two_sided_sign_flip",
    "paired_seed_bootstrap",
]
