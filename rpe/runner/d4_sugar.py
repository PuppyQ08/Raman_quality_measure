from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression

from rpe.downstream.sugar_quantitative import (
    ARCHIVE_BYTES,
    ARCHIVE_SHA256,
    AXIS_SHA256,
    BLANK_RECORD_IDS_SHA256,
    BLANK_SOURCE_MEMBERS_SHA256,
    CONFIG_BYTES,
    CONFIG_SHA256,
    EXPERIMENT_ID,
    FOLD_RECORD_IDS_SHA256,
    FOLD_SOURCE_MEMBERS_SHA256,
    MIXTURE_RECORD_IDS_SHA256,
    MIXTURE_SOURCE_MEMBERS_SHA256,
    TARGET_NAMES,
    D4SugarCohort,
    D4WellSplit,
    load_d4_protocol_config,
    load_d4_sugar_cohort,
)


SCHEMA_VERSION = "phase05-d4-result-v1"
RESULT_LEVELS = ("smoke", "provisional", "complete_seed")
SEEDS_REQUIRED = 5
CONTROL_CONDITION_ID = "low_snr_raw_control"
SG_CONDITION_ID = "low_snr_raw_plus_sg11"
CONDITION_IDS = (CONTROL_CONDITION_ID, SG_CONDITION_ID)
N_COMPONENTS_GRID = (2, 4, 8, 16, 32)
TARGET_RANGE = 0.32
TARGET_LEVELS = (0.0, 0.08, 0.2, 0.32)
ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "rpe/downstream/sugar_quantitative.py",
    "rpe/runner/d4_sugar.py",
    "tools/run_phase05_d4.py",
)
RETAINED_MATRIX_SHA256 = (
    "6b31d9cce62e164bdf4c4807ad974f98ab2657062460a0ea55cf341eb6bd875e"
)
RETAINED_TARGETS_SHA256 = (
    "9199f29cf7d25462edc54a73bb558c4cc84bba2d8ea9bb2708b1e46cbb020386"
)
RETAINED_BLANK_MATRIX_SHA256 = (
    "d64b0f6259c2089d371e97acd6f322dae8c0798d3db4e6319cddcefe90c22176"
)
RETAINED_WELL_IDS_SHA256 = (
    "8e94b18cee84a78df3d43c55221b7db5564d1e75854b8d623e5bf6c18e8f8c82"
)
RETAINED_BLANK_WELL_IDS_SHA256 = (
    "e6ed04b63409805df7633ff054a4ee844d78093059ece73a73c58aa030cecc18"
)
RETAINED_ROUNDS_SHA256 = (
    "62e462619b393a1770588c4e9a58cbdaf076e5df982eb43542cd1303ea81eb9e"
)
RETAINED_REPETITIONS_SHA256 = (
    "4e9826d9c5767662e360f70b5ea8603ea839f61924a72a6127883e95195adf97"
)
FOLD_WELL_IDS_SHA256 = (
    "6249d6b4e1064ff1ab874b48df8e205ccd18c2345b2a4042574f046f0355307a",
    "daf80529b91c5c6a2bb00a512848730a876482728268d65024a70ff41d57a857",
    "3e7fb946507b5d3b432f16dab35ace2b3b89f709c14ffe203c92ef01aa0ce1c1",
    "8d3bef948d43fcd954b17873862b29f1516854351aa7fa56f3f954873e3f5c24",
    "ad6ee7924c0bd2023e274cdb42c5a3ddaf8ce5b96cfde5b006e7d687c0ba462d",
)
TECHNICAL_LOD_STATUS = (
    "technical_repeatability_estimate_not_validated_analytical_limit"
)
FORBIDDEN_INFERENCE_KEYS = {
    "bootstrap",
    "confidence_interval",
    "gate_success",
    "p_value",
    "permutation",
    "significant",
}


class D4RunnerValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


@dataclass(frozen=True)
class D4ConditionMatrix:
    condition_id: str
    protocol_config_sha256: str
    record_ids: tuple[str, ...]
    values: np.ndarray


def _read_only(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_strings_sha256(values: tuple[str, ...]) -> str:
    return hashlib.sha256(
        ("\n".join(values) + "\n").encode("utf-8")
    ).hexdigest()


def _sorted_strings_sha256(values: set[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(values)) + "\n").encode("utf-8")
    ).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
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


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D4RunnerValidationError(path, "must be an object")
    return value


def _recursive_keys(value: object) -> set[str]:
    keys = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.add(str(key))
            keys.update(_recursive_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.update(_recursive_keys(item))
    return keys


def _finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and np.isfinite(float(value))
    )


def _sha256_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _code_document() -> dict[str, object]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
        for relative_path in CODE_PATHS
    }


def _cohort_document(cohort: D4SugarCohort) -> dict[str, object]:
    return {
        "axis_sha256": _array_sha256(cohort.wavenumber),
        "blank_matrix_sha256": _array_sha256(cohort.blank_intensity),
        "blank_record_count": int(cohort.blank_intensity.shape[0]),
        "blank_record_ids_sha256": _ordered_strings_sha256(
            cohort.blank_record_ids
        ),
        "blank_source_members_sha256": _ordered_strings_sha256(
            cohort.blank_source_members
        ),
        "blank_well_ids_sha256": _ordered_strings_sha256(
            cohort.blank_well_ids
        ),
        "feature_count": int(cohort.intensity.shape[1]),
        "matrix_sha256": _array_sha256(cohort.intensity),
        "record_count": int(cohort.intensity.shape[0]),
        "record_ids_sha256": _ordered_strings_sha256(cohort.record_ids),
        "repetitions_sha256": _array_sha256(cohort.repetitions),
        "rounds_sha256": _array_sha256(cohort.rounds),
        "source_members_sha256": _ordered_strings_sha256(
            cohort.source_members
        ),
        "targets_sha256": _array_sha256(cohort.targets),
        "well_count": len(set(cohort.well_ids)),
        "well_ids_sha256": _ordered_strings_sha256(cohort.well_ids),
    }


def _expected_retained_cohort() -> dict[str, object]:
    return {
        "axis_sha256": AXIS_SHA256,
        "blank_matrix_sha256": RETAINED_BLANK_MATRIX_SHA256,
        "blank_record_count": 32,
        "blank_record_ids_sha256": BLANK_RECORD_IDS_SHA256,
        "blank_source_members_sha256": BLANK_SOURCE_MEMBERS_SHA256,
        "blank_well_ids_sha256": RETAINED_BLANK_WELL_IDS_SHA256,
        "feature_count": 2000,
        "matrix_sha256": RETAINED_MATRIX_SHA256,
        "record_count": 7680,
        "record_ids_sha256": MIXTURE_RECORD_IDS_SHA256,
        "repetitions_sha256": RETAINED_REPETITIONS_SHA256,
        "rounds_sha256": RETAINED_ROUNDS_SHA256,
        "source_members_sha256": MIXTURE_SOURCE_MEMBERS_SHA256,
        "targets_sha256": RETAINED_TARGETS_SHA256,
        "well_count": 240,
        "well_ids_sha256": RETAINED_WELL_IDS_SHA256,
    }


def _validate_level_and_seed(
    *,
    seed: int,
    result_level: str,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise D4RunnerValidationError("seed", "must be an integer")
    if result_level not in RESULT_LEVELS:
        raise D4RunnerValidationError(
            "result level",
            f"must be one of {RESULT_LEVELS!r}",
        )
    if result_level == "provisional" and seed != 0:
        raise D4RunnerValidationError(
            "provisional seed",
            "must equal 0 for PROVISIONAL — 1/5 SEEDS",
        )
    if result_level == "complete_seed" and seed not in range(SEEDS_REQUIRED):
        raise D4RunnerValidationError(
            "complete seed",
            "must be an integer in [0, 4]",
        )


def _validate_provisional_cohort(cohort: D4SugarCohort) -> None:
    observed = _cohort_document(cohort)
    expected = _expected_retained_cohort()
    if observed != expected:
        raise D4RunnerValidationError(
            "provisional cohort",
            "must equal the frozen retained Sugar cohort identities",
        )


def retained_sugar_comparison_for_level(result_level: str) -> bool:
    return result_level in {"provisional", "complete_seed"}


def _validate_cohort(cohort: D4SugarCohort) -> None:
    if not isinstance(cohort, D4SugarCohort):
        raise D4RunnerValidationError(
            "cohort",
            "must be a D4SugarCohort",
        )
    if cohort.protocol_config_sha256 != CONFIG_SHA256:
        raise D4RunnerValidationError(
            "protocol config",
            f"must equal {CONFIG_SHA256}",
        )
    intensity = cohort.intensity
    targets = cohort.targets
    row_count = len(cohort.record_ids)
    if (
        not isinstance(intensity, np.ndarray)
        or intensity.dtype != np.dtype("<f4")
        or intensity.ndim != 2
        or intensity.shape[0] != row_count
        or intensity.shape[1] < max(N_COMPONENTS_GRID)
        or not np.isfinite(intensity).all()
    ):
        raise D4RunnerValidationError(
            "cohort intensity",
            "must be a finite float32 matrix with enough features",
        )
    if (
        not isinstance(targets, np.ndarray)
        or targets.dtype != np.dtype("<f8")
        or targets.shape != (row_count, len(TARGET_NAMES))
        or not np.isfinite(targets).all()
    ):
        raise D4RunnerValidationError(
            "cohort targets",
            "must be a finite float64 matrix with four targets",
        )
    if cohort.target_names != TARGET_NAMES:
        raise D4RunnerValidationError(
            "target names",
            f"must equal {TARGET_NAMES!r}",
        )
    wavenumber = cohort.wavenumber
    if (
        not isinstance(wavenumber, np.ndarray)
        or wavenumber.dtype != np.dtype("<f4")
        or wavenumber.shape != (intensity.shape[1],)
        or not np.isfinite(wavenumber).all()
        or not np.all(np.diff(wavenumber) > 0.0)
    ):
        raise D4RunnerValidationError(
            "wavenumber",
            "must be an aligned finite strictly increasing float32 axis",
        )
    if any(
        len(values) != row_count
        for values in (
            cohort.well_ids,
            cohort.source_members,
            cohort.rounds,
            cohort.repetitions,
        )
    ):
        raise D4RunnerValidationError(
            "cohort rows",
            "identity arrays must align with records",
        )
    if (
        any(
            not isinstance(value, str) or value == ""
            for values in (
                cohort.record_ids,
                cohort.well_ids,
                cohort.source_members,
            )
            for value in values
        )
        or len(set(cohort.record_ids)) != row_count
        or len(set(cohort.source_members)) != row_count
        or not isinstance(cohort.rounds, np.ndarray)
        or cohort.rounds.dtype != np.dtype("<i8")
        or cohort.rounds.shape != (row_count,)
        or np.any(cohort.rounds <= 0)
        or not isinstance(cohort.repetitions, np.ndarray)
        or cohort.repetitions.dtype != np.dtype("<i8")
        or cohort.repetitions.shape != (row_count,)
        or np.any(cohort.repetitions <= 0)
    ):
        raise D4RunnerValidationError(
            "cohort identities",
            "record/source IDs must be unique and acquisition arrays positive",
        )
    if (
        not isinstance(cohort.blank_intensity, np.ndarray)
        or cohort.blank_intensity.dtype != np.dtype("<f4")
        or cohort.blank_intensity.ndim != 2
        or cohort.blank_intensity.shape[1] != intensity.shape[1]
        or cohort.blank_intensity.shape[0] < 2
        or not np.isfinite(cohort.blank_intensity).all()
        or not isinstance(cohort.blank_targets, np.ndarray)
        or cohort.blank_targets.dtype != np.dtype("<f8")
        or cohort.blank_targets.shape
        != (cohort.blank_intensity.shape[0], len(TARGET_NAMES))
        or not np.isfinite(cohort.blank_targets).all()
        or not np.all(cohort.blank_targets == 0.0)
    ):
        raise D4RunnerValidationError(
            "blank cohort",
            "blank arrays must be finite, aligned, and all-zero target",
        )
    blank_count = cohort.blank_intensity.shape[0]
    if (
        any(
            len(values) != blank_count
            for values in (
                cohort.blank_record_ids,
                cohort.blank_well_ids,
                cohort.blank_source_members,
            )
        )
        or any(
            not isinstance(value, str) or value == ""
            for values in (
                cohort.blank_record_ids,
                cohort.blank_well_ids,
                cohort.blank_source_members,
            )
            for value in values
        )
        or len(set(cohort.blank_record_ids)) != blank_count
        or len(set(cohort.blank_source_members)) != blank_count
    ):
        raise D4RunnerValidationError(
            "blank identities",
            "blank record, well, and source identities must be aligned",
        )


def prepare_d4_conditions(
    cohort: D4SugarCohort,
    config_path: Path,
) -> tuple[D4ConditionMatrix, D4ConditionMatrix]:
    _validate_cohort(cohort)
    config = load_d4_protocol_config(Path(config_path))
    if config.sha256 != cohort.protocol_config_sha256:
        raise D4RunnerValidationError(
            "protocol config",
            "cohort and loaded config SHA256 differ",
        )
    control_values = np.asarray(cohort.intensity, dtype="<f4").copy()
    sg_values = np.asarray(
        savgol_filter(
            cohort.intensity,
            window_length=11,
            polyorder=3,
            deriv=0,
            axis=-1,
            mode="interp",
        ),
        dtype="<f4",
    ).copy()
    if not np.isfinite(sg_values).all():
        raise D4RunnerValidationError(
            "SG values",
            "contains non-finite values",
        )
    return (
        D4ConditionMatrix(
            condition_id=CONTROL_CONDITION_ID,
            protocol_config_sha256=cohort.protocol_config_sha256,
            record_ids=cohort.record_ids,
            values=_read_only(control_values),
        ),
        D4ConditionMatrix(
            condition_id=SG_CONDITION_ID,
            protocol_config_sha256=cohort.protocol_config_sha256,
            record_ids=cohort.record_ids,
            values=_read_only(sg_values),
        ),
    )


def regression_metrics(
    true: np.ndarray,
    predicted: np.ndarray,
    target_names: tuple[str, ...],
) -> dict[str, object]:
    true = np.asarray(true, dtype="<f8")
    predicted = np.asarray(predicted, dtype="<f8")
    if (
        true.ndim != 2
        or predicted.shape != true.shape
        or true.shape[1] != len(target_names)
        or true.shape[0] == 0
        or not np.isfinite(true).all()
        or not np.isfinite(predicted).all()
    ):
        raise D4RunnerValidationError(
            "metrics arrays",
            "must be aligned finite nonempty matrices",
        )
    errors = predicted - true
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    mae = np.mean(np.abs(errors), axis=0)
    denominator = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    if np.any(denominator <= 0.0):
        raise D4RunnerValidationError(
            "R2 denominator",
            "each target must vary",
        )
    r2 = 1.0 - np.sum(errors**2, axis=0) / denominator
    return {
        "macro_mae_mol_l": float(mae.mean()),
        "macro_normalized_rmse": float(np.mean(rmse / TARGET_RANGE)),
        "macro_r2": float(r2.mean()),
        "per_analyte": {
            target_name: {
                "mae_mol_l": float(mae[index]),
                "r2": float(r2[index]),
                "rmse_mol_l": float(rmse[index]),
            }
            for index, target_name in enumerate(target_names)
        },
        "sample_count": int(true.shape[0]),
    }


def technical_lod_loq(
    train_true: np.ndarray,
    train_predicted: np.ndarray,
    blank_predicted: np.ndarray,
    target_names: tuple[str, ...],
) -> dict[str, object]:
    train_true = np.asarray(train_true, dtype="<f8")
    train_predicted = np.asarray(train_predicted, dtype="<f8")
    blank_predicted = np.asarray(blank_predicted, dtype="<f8")
    target_count = len(target_names)
    if (
        train_true.ndim != 2
        or train_true.shape[1] != target_count
        or train_predicted.shape != train_true.shape
        or blank_predicted.ndim != 2
        or blank_predicted.shape[1] != target_count
        or blank_predicted.shape[0] < 2
        or not np.isfinite(train_true).all()
        or not np.isfinite(train_predicted).all()
        or not np.isfinite(blank_predicted).all()
    ):
        raise D4RunnerValidationError(
            "LOD arrays",
            "must be finite aligned matrices",
        )
    estimates = {}
    for index, target_name in enumerate(target_names):
        true = train_true[:, index]
        predicted = train_predicted[:, index]
        centered_true = true - true.mean()
        denominator = float(np.sum(centered_true**2))
        if denominator <= 0.0:
            raise D4RunnerValidationError(
                f"{target_name} positive finite slope",
                "training target must vary",
            )
        slope = float(
            np.sum(centered_true * (predicted - predicted.mean()))
            / denominator
        )
        sigma = float(np.std(blank_predicted[:, index], ddof=1))
        if (
            not np.isfinite(slope)
            or slope <= 0.0
            or not np.isfinite(sigma)
            or sigma < 0.0
        ):
            raise D4RunnerValidationError(
                f"{target_name} positive finite slope",
                "slope must be positive finite and sigma nonnegative",
            )
        estimates[target_name] = {
            "blank_replicates": int(blank_predicted.shape[0]),
            "ich_lod_mol_l": float(3.3 * sigma / slope),
            "ich_loq_mol_l": float(10.0 * sigma / slope),
            "iupac_lod_mol_l": float(3.0 * sigma / slope),
            "sigma": sigma,
            "slope": slope,
            "status": TECHNICAL_LOD_STATUS,
        }
    return estimates


def _validate_split(
    cohort: D4SugarCohort,
    split: D4WellSplit,
) -> None:
    row_count = len(cohort.record_ids)
    arrays = (
        split.train_indices,
        split.validation_indices,
        split.test_indices,
    )
    for name, values in zip(
        ("train", "validation", "test"),
        arrays,
        strict=True,
    ):
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype("<i8")
            or values.ndim != 1
            or values.size == 0
            or np.any(values < 0)
            or np.any(values >= row_count)
            or np.unique(values).size != values.size
        ):
            raise D4RunnerValidationError(
                f"{name} indices",
                "must be unique nonempty in-range int64 values",
            )
    train, validation, test = [set(map(int, values)) for values in arrays]
    if (
        train & validation
        or train & test
        or validation & test
        or train | validation | test != set(range(row_count))
    ):
        raise D4RunnerValidationError(
            "split records",
            "must be a disjoint complete partition",
        )
    well_sets = [
        {cohort.well_ids[index] for index in values}
        for values in (train, validation, test)
    ]
    if (
        well_sets[0] & well_sets[1]
        or well_sets[0] & well_sets[2]
        or well_sets[1] & well_sets[2]
    ):
        raise D4RunnerValidationError(
            "split wells",
            "a physical well crosses split roles",
        )
    if (
        isinstance(split.seed, bool)
        or not isinstance(split.seed, int)
        or split.seed not in range(5)
        or len(cohort.folds) != 5
    ):
        raise D4RunnerValidationError(
            "split folds",
            "seed and supplied folds must follow the fixed five-fold protocol",
        )
    expected_test_fold = split.seed
    expected_validation_fold = (split.seed + 1) % 5
    expected_train_folds = tuple(
        fold
        for fold in range(5)
        if fold not in {expected_test_fold, expected_validation_fold}
    )
    fold_sets = []
    for fold, values in enumerate(cohort.folds):
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.dtype("<i8")
            or values.ndim != 1
            or values.size == 0
            or np.any(values < 0)
            or np.any(values >= row_count)
            or np.unique(values).size != values.size
        ):
            raise D4RunnerValidationError(
                "split folds",
                f"fold {fold} must contain unique in-range int64 indices",
            )
        fold_sets.append(set(map(int, values)))
    fold_by_row = {
        row_index: fold
        for fold, rows in enumerate(fold_sets)
        for row_index in rows
    }
    well_fold_sets: dict[str, set[int]] = {}
    for row_index, well_id in enumerate(cohort.well_ids):
        well_fold_sets.setdefault(well_id, set()).add(
            fold_by_row.get(row_index, -1)
        )
    if (
        any(
            fold_sets[left] & fold_sets[right]
            for left in range(5)
            for right in range(left + 1, 5)
        )
        or set().union(*fold_sets) != set(range(row_count))
        or any(len(folds) != 1 for folds in well_fold_sets.values())
        or split.test_fold != expected_test_fold
        or split.validation_fold != expected_validation_fold
        or split.train_folds != expected_train_folds
        or test != fold_sets[expected_test_fold]
        or validation != fold_sets[expected_validation_fold]
        or train
        != set().union(
            *(fold_sets[fold] for fold in expected_train_folds)
        )
    ):
        raise D4RunnerValidationError(
            "split folds",
            "labels and indices must match the fixed five-fold rotation",
        )
    expected_levels = set(TARGET_LEVELS)
    if any(
        {
            float(value)
            for value in cohort.targets[
                np.asarray(sorted(fold_sets[fold]), dtype="<i8"),
                target_index,
            ]
        }
        != expected_levels
        for fold in range(5)
        for target_index in range(len(TARGET_NAMES))
    ):
        raise D4RunnerValidationError(
            "split target levels",
            "each fold and nominal analyte must contain all four fixed levels",
        )


def _find_split(cohort: D4SugarCohort, seed: int) -> D4WellSplit:
    matches = [split for split in cohort.splits if split.seed == seed]
    if len(matches) != 1:
        raise D4RunnerValidationError(
            "seed",
            f"must identify exactly one supplied split; observed {len(matches)}",
        )
    return matches[0]


def _fit_condition(
    cohort: D4SugarCohort,
    split: D4WellSplit,
    condition: D4ConditionMatrix,
) -> tuple[dict[str, object], np.ndarray]:
    if (
        condition.protocol_config_sha256 != cohort.protocol_config_sha256
        or condition.record_ids != cohort.record_ids
        or condition.values.shape != cohort.intensity.shape
    ):
        raise D4RunnerValidationError(
            "condition provenance",
            "config, records, or shape mismatch",
        )
    train = split.train_indices
    validation = split.validation_indices
    test = split.test_indices
    validation_scores = []
    models = {}
    for n_components in N_COMPONENTS_GRID:
        if n_components > min(
            train.size,
            condition.values.shape[1],
        ):
            raise D4RunnerValidationError(
                "n_components",
                "grid exceeds train samples or features",
            )
        model = PLSRegression(
            n_components=n_components,
            scale=True,
            max_iter=500,
            tol=1e-6,
            copy=True,
        )
        model.fit(condition.values[train], cohort.targets[train])
        validation_prediction = np.asarray(
            model.predict(condition.values[validation]),
            dtype="<f8",
        )
        metrics = regression_metrics(
            cohort.targets[validation],
            validation_prediction,
            cohort.target_names,
        )
        validation_scores.append(
            {
                "macro_normalized_rmse": metrics[
                    "macro_normalized_rmse"
                ],
                "n_components": n_components,
            }
        )
        models[n_components] = model
    selected = min(
        validation_scores,
        key=lambda row: (
            row["macro_normalized_rmse"],
            row["n_components"],
        ),
    )["n_components"]
    model = models[selected]
    train_prediction = np.asarray(
        model.predict(condition.values[train]),
        dtype="<f8",
    )
    test_prediction = np.asarray(
        model.predict(condition.values[test]),
        dtype="<f8",
    )
    blank_values = (
        cohort.blank_intensity
        if condition.condition_id == CONTROL_CONDITION_ID
        else np.asarray(
            savgol_filter(
                cohort.blank_intensity,
                window_length=11,
                polyorder=3,
                deriv=0,
                axis=-1,
                mode="interp",
            ),
            dtype="<f4",
        )
    )
    blank_prediction = np.asarray(
        model.predict(blank_values),
        dtype="<f8",
    )
    return (
        {
            "condition_id": condition.condition_id,
            "condition_matrix_sha256": _array_sha256(condition.values),
            "lod_loq": technical_lod_loq(
                cohort.targets[train],
                train_prediction,
                blank_prediction,
                cohort.target_names,
            ),
            "selected_n_components": selected,
            "test_metrics": regression_metrics(
                cohort.targets[test],
                test_prediction,
                cohort.target_names,
            ),
            "test_predictions": test_prediction.tolist(),
            "validation_scores": validation_scores,
        },
        test_prediction,
    )


def _paired_predictions(
    cohort: D4SugarCohort,
    split: D4WellSplit,
    predictions: Mapping[str, np.ndarray],
) -> list[dict[str, object]]:
    rows = []
    for position, index in enumerate(split.test_indices):
        row_index = int(index)
        rows.append(
            {
                "predictions": {
                    condition_id: [
                        float(value)
                        for value in prediction[position]
                    ]
                    for condition_id, prediction in predictions.items()
                },
                "record_id": cohort.record_ids[row_index],
                "repetition": int(cohort.repetitions[row_index]),
                "round": int(cohort.rounds[row_index]),
                "source_member": cohort.source_members[row_index],
                "true_targets": [
                    float(value) for value in cohort.targets[row_index]
                ],
                "well_id": cohort.well_ids[row_index],
            }
        )
    return rows


def _split_document(
    cohort: D4SugarCohort,
    split: D4WellSplit,
) -> dict[str, object]:
    test_record_ids = tuple(
        cohort.record_ids[int(index)]
        for index in split.test_indices
    )
    test_source_members = tuple(
        cohort.source_members[int(index)]
        for index in split.test_indices
    )
    test_well_ids = {
        cohort.well_ids[int(index)]
        for index in split.test_indices
    }
    return {
        "seed": split.seed,
        "test_count": int(split.test_indices.size),
        "test_fold": split.test_fold,
        "test_record_ids_sha256": _sorted_strings_sha256(
            set(test_record_ids)
        ),
        "test_source_members_sha256": _ordered_strings_sha256(
            test_source_members
        ),
        "test_well_count": len(test_well_ids),
        "test_well_ids_sha256": _sorted_strings_sha256(test_well_ids),
        "train_count": int(split.train_indices.size),
        "train_folds": list(split.train_folds),
        "train_well_count": len(
            {cohort.well_ids[int(index)] for index in split.train_indices}
        ),
        "validation_count": int(split.validation_indices.size),
        "validation_fold": split.validation_fold,
        "validation_well_count": len(
            {
                cohort.well_ids[int(index)]
                for index in split.validation_indices
            }
        ),
    }


def _paired_predictions_jsonl_bytes(
    rows: list[Mapping[str, object]],
) -> bytes:
    return b"".join(_canonical_json_bytes(row) for row in rows)


def paired_predictions_jsonl_bytes(
    result: Mapping[str, object],
) -> bytes:
    rows = result.get("paired_predictions")
    if (
        not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise D4RunnerValidationError(
            "paired predictions",
            "must be a nonempty list of objects",
        )
    return _paired_predictions_jsonl_bytes(rows)


def run_d4_result_from_cohort(
    cohort: D4SugarCohort,
    config_path: Path,
    *,
    seed: int,
    result_level: str,
) -> dict[str, object]:
    _validate_level_and_seed(seed=seed, result_level=result_level)
    _validate_cohort(cohort)
    if result_level in {"provisional", "complete_seed"}:
        _validate_provisional_cohort(cohort)
    config = load_d4_protocol_config(Path(config_path))
    if config.sha256 != cohort.protocol_config_sha256:
        raise D4RunnerValidationError(
            "protocol config",
            "cohort and config SHA256 differ",
        )
    split = _find_split(cohort, seed)
    _validate_split(cohort, split)
    conditions = prepare_d4_conditions(cohort, config_path)
    condition_documents = []
    predictions = {}
    for condition in conditions:
        document, test_prediction = _fit_condition(
            cohort,
            split,
            condition,
        )
        condition_documents.append(document)
        predictions[condition.condition_id] = test_prediction
    by_condition = {
        document["condition_id"]: document for document in condition_documents
    }
    paired_predictions = _paired_predictions(
        cohort,
        split,
        predictions,
    )
    paired_payload = _paired_predictions_jsonl_bytes(paired_predictions)
    result = {
        "claim_boundary": {
            "descriptive_only": True,
            "formal_inference": False,
            "retained_sugar_comparison": (
                retained_sugar_comparison_for_level(result_level)
            ),
        },
        "code": _code_document(),
        "cohort": _cohort_document(cohort),
        "conditions": condition_documents,
        "descriptive_effect_percentage_points": 100.0
        * (
            by_condition[CONTROL_CONDITION_ID]["test_metrics"][
                "macro_normalized_rmse"
            ]
            - by_condition[SG_CONDITION_ID]["test_metrics"][
                "macro_normalized_rmse"
            ]
        ),
        "experiment_id": EXPERIMENT_ID,
        "paired_predictions": paired_predictions,
        "paired_predictions_artifact": {
            "bytes": len(paired_payload),
            "lines": len(paired_predictions),
            "path": (
                f"paired_predictions_seed{seed}_complete.jsonl"
                if result_level == "complete_seed"
                else f"paired_predictions_seed{seed}.jsonl"
            ),
            "sha256": hashlib.sha256(paired_payload).hexdigest(),
        },
        "protocol_config": {
            "bytes": CONFIG_BYTES,
            "path": Path(config_path).name,
            "sha256": CONFIG_SHA256,
        },
        "result_label": (
            "PROVISIONAL — 1/5 SEEDS"
            if result_level == "provisional"
            else (
                f"COMPLETE SEED — {seed + 1}/5"
                if result_level == "complete_seed"
                else "SMOKE — PROVIDED COHORT"
            )
        ),
        "result_level": result_level,
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "seeds_completed": (
            seed + 1 if result_level == "complete_seed" else 1
        ),
        "seeds_required": SEEDS_REQUIRED,
        "source_archive": (
            {
                "bytes": ARCHIVE_BYTES,
                "path": "Raw data.zip",
                "sha256": ARCHIVE_SHA256,
            }
            if result_level in {"provisional", "complete_seed"}
            else None
        ),
        "split": _split_document(cohort, split),
        "status": "completed",
        "target_names": list(cohort.target_names),
    }
    validate_d4_result(result)
    return result


def run_d4_smoke_from_cohort(
    cohort: D4SugarCohort,
    config_path: Path,
    *,
    seed: int,
) -> dict[str, object]:
    return run_d4_result_from_cohort(
        cohort,
        config_path,
        seed=seed,
        result_level="smoke",
    )


def run_d4_provisional_experiment(
    config_path: Path,
    archive_path: Path,
    *,
    seed: int,
    result_level: str,
) -> dict[str, object]:
    _validate_level_and_seed(seed=seed, result_level=result_level)
    config = load_d4_protocol_config(Path(config_path))
    if config.sha256 != CONFIG_SHA256:
        raise D4RunnerValidationError(
            "protocol config",
            "SHA256 mismatch",
        )
    cohort = load_d4_sugar_cohort(config_path, archive_path)
    return run_d4_result_from_cohort(
        cohort,
        config_path,
        seed=seed,
        result_level=result_level,
    )


def _metrics_from_paired(
    rows: list[Mapping[str, object]],
    condition_id: str,
    target_names: tuple[str, ...],
) -> Mapping[str, object]:
    true = np.asarray(
        [row["true_targets"] for row in rows],
        dtype="<f8",
    )
    predicted = np.asarray(
        [row["predictions"][condition_id] for row in rows],
        dtype="<f8",
    )
    return regression_metrics(true, predicted, target_names)


def _validate_lod_loq(value: object) -> None:
    document = _object("LOD/LOQ", value)
    if set(document) != set(TARGET_NAMES):
        raise D4RunnerValidationError(
            "LOD/LOQ",
            "must contain exactly the four frozen targets",
        )
    expected_keys = {
        "blank_replicates",
        "ich_lod_mol_l",
        "ich_loq_mol_l",
        "iupac_lod_mol_l",
        "sigma",
        "slope",
        "status",
    }
    for target_name in TARGET_NAMES:
        estimate = _object(
            f"LOD/LOQ {target_name}",
            document[target_name],
        )
        blank_replicates = estimate.get("blank_replicates")
        slope = estimate.get("slope")
        sigma = estimate.get("sigma")
        if (
            set(estimate) != expected_keys
            or isinstance(blank_replicates, bool)
            or not isinstance(blank_replicates, int)
            or blank_replicates < 2
            or not _finite_number(slope)
            or float(slope) <= 0.0
            or not _finite_number(sigma)
            or float(sigma) < 0.0
            or estimate.get("status") != TECHNICAL_LOD_STATUS
        ):
            raise D4RunnerValidationError(
                "LOD/LOQ",
                f"{target_name} has an invalid technical estimate contract",
            )
        expected = {
            "ich_lod_mol_l": 3.3 * float(sigma) / float(slope),
            "ich_loq_mol_l": 10.0 * float(sigma) / float(slope),
            "iupac_lod_mol_l": 3.0 * float(sigma) / float(slope),
        }
        if any(
            not _finite_number(estimate.get(key))
            or float(estimate[key]) != formula_value
            for key, formula_value in expected.items()
        ):
            raise D4RunnerValidationError(
                "LOD/LOQ",
                f"{target_name} formula values do not match sigma and slope",
            )


def validate_d4_result(
    result: Mapping[str, object],
) -> None:
    root = _object("result", result)
    if set(root) != {
        "claim_boundary",
        "code",
        "cohort",
        "conditions",
        "descriptive_effect_percentage_points",
        "experiment_id",
        "paired_predictions",
        "paired_predictions_artifact",
        "protocol_config",
        "result_label",
        "result_level",
        "schema_version",
        "seed",
        "seeds_completed",
        "seeds_required",
        "source_archive",
        "split",
        "status",
        "target_names",
    }:
        raise D4RunnerValidationError(
            "result fields",
            "must equal the fixed smoke result schema",
        )
    result_level = root.get("result_level")
    seed = root.get("seed")
    _validate_level_and_seed(seed=seed, result_level=result_level)
    expected_label = (
        "PROVISIONAL — 1/5 SEEDS"
        if result_level == "provisional"
        else (
            f"COMPLETE SEED — {seed + 1}/5"
            if result_level == "complete_seed"
            else "SMOKE — PROVIDED COHORT"
        )
    )
    if (
        root.get("schema_version") != SCHEMA_VERSION
        or root.get("experiment_id") != EXPERIMENT_ID
        or root.get("status") != "completed"
        or root.get("result_label") != expected_label
    ):
        raise D4RunnerValidationError(
            "result identity",
            "schema, experiment, status, level, or label mismatch",
        )
    if (
        root.get("seeds_completed")
        != (seed + 1 if result_level == "complete_seed" else 1)
        or root.get("seeds_required") != SEEDS_REQUIRED
    ):
        raise D4RunnerValidationError(
            "seed completion",
            "does not match result level and seed",
        )
    if _recursive_keys(root) & FORBIDDEN_INFERENCE_KEYS:
        raise D4RunnerValidationError(
            "inference boundary",
            "contains forbidden inference fields",
        )
    claim_boundary = _object(
        "claim boundary",
        root.get("claim_boundary"),
    )
    expected_claim_boundary = {
        "descriptive_only": True,
        "formal_inference": False,
        "retained_sugar_comparison": (
            retained_sugar_comparison_for_level(result_level)
        ),
    }
    if claim_boundary != expected_claim_boundary:
        raise D4RunnerValidationError(
            "claim boundary",
            "must match the result-level claim boundary",
        )
    code = _object("code provenance", root.get("code"))
    if code != _code_document():
        raise D4RunnerValidationError(
            "code provenance",
            "must match the current fixed runner and loader identities",
        )
    cohort = _object("cohort provenance", root.get("cohort"))
    cohort_keys = {
        "axis_sha256",
        "blank_matrix_sha256",
        "blank_record_count",
        "blank_record_ids_sha256",
        "blank_source_members_sha256",
        "blank_well_ids_sha256",
        "feature_count",
        "matrix_sha256",
        "record_count",
        "record_ids_sha256",
        "repetitions_sha256",
        "rounds_sha256",
        "source_members_sha256",
        "targets_sha256",
        "well_count",
        "well_ids_sha256",
    }
    if (
        set(cohort) != cohort_keys
        or any(
            not _sha256_string(cohort.get(key))
            for key in cohort_keys
            if key.endswith("_sha256")
        )
        or any(
            isinstance(cohort.get(key), bool)
            or not isinstance(cohort.get(key), int)
            or cohort[key] <= 0
            for key in (
                "blank_record_count",
                "feature_count",
                "record_count",
                "well_count",
            )
        )
        or cohort["feature_count"] < max(N_COMPONENTS_GRID)
        or cohort["well_count"] > cohort["record_count"]
    ):
        raise D4RunnerValidationError(
            "cohort provenance",
            "fingerprints and positive dimensions must follow the fixed schema",
        )
    if (
        result_level in {"provisional", "complete_seed"}
        and cohort != _expected_retained_cohort()
    ):
        raise D4RunnerValidationError(
            "provisional cohort",
            "must equal the frozen retained Sugar cohort identities",
        )
    target_names = root.get("target_names")
    if target_names != list(TARGET_NAMES):
        raise D4RunnerValidationError(
            "target names",
            f"must equal {TARGET_NAMES!r}",
        )
    conditions = root.get("conditions")
    if (
        not isinstance(conditions, list)
        or any(not isinstance(condition, Mapping) for condition in conditions)
        or [condition.get("condition_id") for condition in conditions]
        != list(CONDITION_IDS)
    ):
        raise D4RunnerValidationError(
            "conditions",
            "must equal control then SG",
        )
    paired = root.get("paired_predictions")
    if not isinstance(paired, list) or not paired:
        raise D4RunnerValidationError(
            "paired predictions",
            "must be a nonempty list",
        )
    records = set()
    for row in paired:
        document = _object("paired prediction row", row)
        record_id = document.get("record_id")
        true_targets = document.get("true_targets")
        predictions = document.get("predictions")
        if (
            set(document)
            != {
                "predictions",
                "record_id",
                "repetition",
                "round",
                "source_member",
                "true_targets",
                "well_id",
            }
            or
            not isinstance(record_id, str)
            or record_id == ""
            or record_id in records
            or not isinstance(document.get("well_id"), str)
            or document.get("well_id") == ""
            or not isinstance(document.get("source_member"), str)
            or document.get("source_member") == ""
            or isinstance(document.get("round"), bool)
            or not isinstance(document.get("round"), int)
            or document.get("round") <= 0
            or isinstance(document.get("repetition"), bool)
            or not isinstance(document.get("repetition"), int)
            or document.get("repetition") <= 0
            or not isinstance(true_targets, list)
            or len(true_targets) != len(TARGET_NAMES)
            or any(not _finite_number(value) for value in true_targets)
            or not isinstance(predictions, Mapping)
            or set(predictions) != set(CONDITION_IDS)
            or any(
                not isinstance(values, list)
                or len(values) != len(TARGET_NAMES)
                or any(not _finite_number(value) for value in values)
                for values in predictions.values()
            )
        ):
            raise D4RunnerValidationError(
                "paired predictions",
                "invalid row identity, target, or prediction shape",
            )
        records.add(record_id)
    paired_payload = paired_predictions_jsonl_bytes(root)
    expected_artifact = {
        "bytes": len(paired_payload),
        "lines": len(paired),
        "path": (
            f"paired_predictions_seed{seed}_complete.jsonl"
            if result_level == "complete_seed"
            else f"paired_predictions_seed{seed}.jsonl"
        ),
        "sha256": hashlib.sha256(paired_payload).hexdigest(),
    }
    if root.get("paired_predictions_artifact") != expected_artifact:
        raise D4RunnerValidationError(
            "paired predictions artifact",
            "bytes, lines, path, or SHA256 mismatch",
        )
    for condition in conditions:
        condition_id = condition["condition_id"]
        if set(condition) != {
            "condition_id",
            "condition_matrix_sha256",
            "lod_loq",
            "selected_n_components",
            "test_metrics",
            "test_predictions",
            "validation_scores",
        }:
            raise D4RunnerValidationError(
                "condition provenance",
                f"{condition_id} fields differ from the fixed schema",
            )
        if not _sha256_string(condition.get("condition_matrix_sha256")):
            raise D4RunnerValidationError(
                "condition provenance",
                f"{condition_id} matrix SHA256 is invalid",
            )
        _validate_lod_loq(condition.get("lod_loq"))
        expected_metrics = _metrics_from_paired(
            paired,
            condition_id,
            TARGET_NAMES,
        )
        if condition.get("test_metrics") != expected_metrics:
            raise D4RunnerValidationError(
                "metrics",
                f"{condition_id} differs from paired predictions",
            )
        predictions = [
            row["predictions"][condition_id] for row in paired
        ]
        if condition.get("test_predictions") != predictions:
            raise D4RunnerValidationError(
                "paired predictions",
                f"{condition_id} projection differs",
            )
        validation_scores = condition.get("validation_scores")
        if (
            not isinstance(validation_scores, list)
            or any(not isinstance(row, Mapping) for row in validation_scores)
            or any(
                set(row) != {"macro_normalized_rmse", "n_components"}
                or not _finite_number(row.get("macro_normalized_rmse"))
                or float(row["macro_normalized_rmse"]) < 0.0
                for row in validation_scores
            )
            or [row.get("n_components") for row in validation_scores]
            != list(N_COMPONENTS_GRID)
            or condition.get("selected_n_components")
            != min(
                validation_scores,
                key=lambda row: (
                    row["macro_normalized_rmse"],
                    row["n_components"],
                ),
            )["n_components"]
        ):
            raise D4RunnerValidationError(
                "component selection",
                "validation grid or selected component mismatch",
            )
    by_condition = {
        condition["condition_id"]: condition for condition in conditions
    }
    if (
        by_condition[CONTROL_CONDITION_ID]["condition_matrix_sha256"]
        != cohort["matrix_sha256"]
        or any(
            estimate["blank_replicates"] != cohort["blank_record_count"]
            for condition in conditions
            for estimate in condition["lod_loq"].values()
        )
    ):
        raise D4RunnerValidationError(
            "cohort provenance",
            "control matrix or blank repeat projections do not match",
        )
    expected_effect = 100.0 * (
        by_condition[CONTROL_CONDITION_ID]["test_metrics"][
            "macro_normalized_rmse"
        ]
        - by_condition[SG_CONDITION_ID]["test_metrics"][
            "macro_normalized_rmse"
        ]
    )
    if root.get("descriptive_effect_percentage_points") != expected_effect:
        raise D4RunnerValidationError(
            "descriptive effect",
            "does not match test metrics",
        )
    protocol = _object("protocol config", root.get("protocol_config"))
    if protocol != {
        "bytes": CONFIG_BYTES,
        "path": "d4_sugar_protocol.json",
        "sha256": CONFIG_SHA256,
    }:
        raise D4RunnerValidationError(
            "protocol config",
            "identity mismatch",
        )
    expected_source_archive = (
        {
            "bytes": ARCHIVE_BYTES,
            "path": "Raw data.zip",
            "sha256": ARCHIVE_SHA256,
        }
        if result_level in {"provisional", "complete_seed"}
        else None
    )
    if root.get("source_archive") != expected_source_archive:
        raise D4RunnerValidationError(
            "source archive",
            "identity does not match result level",
        )
    split = _object("split", root.get("split"))
    expected_validation_fold = (
        (seed + 1) % 5
        if not isinstance(seed, bool) and isinstance(seed, int)
        else None
    )
    expected_train_folds = (
        [
            fold
            for fold in range(5)
            if fold not in {seed, expected_validation_fold}
        ]
        if expected_validation_fold is not None
        else None
    )
    count_keys = (
        "test_count",
        "test_well_count",
        "train_count",
        "train_well_count",
        "validation_count",
        "validation_well_count",
    )
    if (
        set(split)
        != {
            "seed",
            "test_count",
            "test_fold",
            "test_record_ids_sha256",
            "test_source_members_sha256",
            "test_well_count",
            "test_well_ids_sha256",
            "train_count",
            "train_folds",
            "train_well_count",
            "validation_count",
            "validation_fold",
            "validation_well_count",
        }
        or expected_validation_fold is None
        or seed not in range(5)
        or any(
            isinstance(split.get(key), bool)
            or not isinstance(split.get(key), int)
            or split[key] <= 0
            for key in count_keys
        )
        or split.get("seed") != seed
        or split.get("test_count") != len(paired)
        or split.get("test_fold") != seed
        or split.get("validation_fold") != expected_validation_fold
        or split.get("train_folds") != expected_train_folds
        or split.get("test_well_count")
        != len({row["well_id"] for row in paired})
        or split.get("test_record_ids_sha256")
        != _sorted_strings_sha256(
            {str(row["record_id"]) for row in paired}
        )
        or split.get("test_source_members_sha256")
        != _ordered_strings_sha256(
            tuple(str(row["source_member"]) for row in paired)
        )
        or split.get("test_well_ids_sha256")
        != _sorted_strings_sha256(
            {str(row["well_id"]) for row in paired}
        )
        or split.get("train_count")
        + split.get("validation_count")
        + split.get("test_count")
        != cohort["record_count"]
        or split.get("train_well_count")
        + split.get("validation_well_count")
        + split.get("test_well_count")
        != cohort["well_count"]
    ):
        raise D4RunnerValidationError(
            "split",
            "fields, folds, counts, or test-well projection mismatch",
        )
    if result_level in {"provisional", "complete_seed"} and (
        split.get("test_count") != 1536
        or split.get("test_well_count") != 48
        or split.get("test_record_ids_sha256")
        != FOLD_RECORD_IDS_SHA256[seed]
        or split.get("test_source_members_sha256")
        != FOLD_SOURCE_MEMBERS_SHA256[seed]
        or split.get("test_well_ids_sha256")
        != FOLD_WELL_IDS_SHA256[seed]
    ):
        raise D4RunnerValidationError(
            "provisional split",
            "must equal the frozen retained seed test fold",
        )
    _canonical_json_bytes(root)


def validate_d4_smoke_result(
    result: Mapping[str, object],
) -> None:
    validate_d4_result(result)


__all__ = [
    "D4ConditionMatrix",
    "D4RunnerValidationError",
    "paired_predictions_jsonl_bytes",
    "prepare_d4_conditions",
    "regression_metrics",
    "retained_sugar_comparison_for_level",
    "run_d4_provisional_experiment",
    "run_d4_result_from_cohort",
    "run_d4_smoke_from_cohort",
    "technical_lod_loq",
    "validate_d4_result",
    "validate_d4_smoke_result",
]
