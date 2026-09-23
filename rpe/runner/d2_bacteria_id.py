from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from rpe.methods.classical.savitzky_golay import (
    SavitzkyGolayPipeline,
    load_savitzky_golay_pipeline,
)
from rpe.runner.d1_bacteria_id import (
    _SplitData,
    _code_document as _d1_code_document,
    _environment_document,
    _fit_condition,
    _leakage_audit,
    _load_dataset,
    _split_document,
    load_d1_runner_config,
)
from rpe.runner.d2_selection import (
    validate_d2_few_shot_selection,
)


SCHEMA_VERSION = "phase05-d2-runner-v1"
RESULT_SCHEMA_VERSION = "phase05-d2-result-v1"
EXPERIMENT_ID = "d2_bacteria_id_few_shot_pca20_lr_sg11"
CONFIG_BYTES = 814
CONFIG_SHA256 = (
    "3ad248a536a362c97c5f583979f643f4a9a77dd1a6ecf65c7bea978951b17eb7"
)
D1_CONFIG_SHA256 = (
    "f4b5aa686486044d6da55725dc5b6f13ad3c4a7dd96ee5a2a3aa21a9cc7ba046"
)
ACCEPTED_SELECTION_SHA256 = (
    "7eb48fa23d25d2f701282631a656e1bd2050c039c916b05f87befe9bda53bb36"
)
SELECTION_CONFIG_SHA256 = (
    "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"
)
RETAINED_SNAPSHOT_SHA256 = (
    "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02"
)
TEST_RECORD_IDS_SHA256 = (
    "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"
)
EXPECTED_SEEDS = (0, 1, 2, 3, 4)
EXPECTED_SHOT_COUNTS = (5, 10, 20)
EXPECTED_CONDITIONS = (
    "released_input_control",
    "released_input_plus_sg",
)
ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "rpe/downstream/bacteria_id.py",
    "rpe/io/schema.py",
    "rpe/io/store.py",
    "rpe/methods/classical/savitzky_golay.py",
    "rpe/runner/d1_bacteria_id.py",
    "rpe/runner/d2_bacteria_id.py",
    "tools/run_phase05_d2.py",
)


class D2RunnerValidationError(ValueError):
    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


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


def _reject_nonfinite(value: str) -> None:
    raise D2RunnerValidationError(
        "config nonfinite",
        f"unsupported JSON constant {value!r}",
    )


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D2RunnerValidationError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise D2RunnerValidationError(path, "keys must be strings")
    return value


def _exact_keys(
    path: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        raise D2RunnerValidationError(
            path,
            (
                f"key mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
        )


def _equal(path: str, observed: object, expected: object) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise D2RunnerValidationError(
            path,
            f"must equal {expected!r}",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(record_ids: tuple[str, ...] | list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(record_ids) + "\n").encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class D2RunnerConfig:
    path: Path
    sha256: str
    byte_count: int
    experiment_id: str
    model_config_path: str
    model_config_sha256: str
    accepted_selection_sha256: str
    accepted_selection_config_sha256: str
    seeds: tuple[int, ...]
    shot_counts: tuple[int, ...]
    batch_size: int


def load_d2_runner_config(path: Path) -> D2RunnerConfig:
    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    except D2RunnerValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D2RunnerValidationError(
            path.name or "config",
            str(error),
        ) from error
    document = _object("config", value)
    if raw != _canonical_json_bytes(document):
        raise D2RunnerValidationError(
            "config noncanonical",
            "must use canonical JSON encoding",
        )
    _exact_keys(
        "config",
        document,
        {
            "schema_version",
            "experiment_id",
            "model_config",
            "selection",
            "dataset",
            "seeds",
            "result_levels",
        },
    )
    _equal("schema_version", document["schema_version"], SCHEMA_VERSION)
    _equal("experiment_id", document["experiment_id"], EXPERIMENT_ID)
    _equal("seeds", document["seeds"], list(EXPECTED_SEEDS))
    _equal("result_levels", document["result_levels"], ["smoke", "provisional"])
    model = _object("model_config", document["model_config"])
    _exact_keys("model_config", model, {"path", "sha256"})
    _equal(
        "model_config.path",
        model["path"],
        "d1_bacteria_id_pca20_lr_sg11.json",
    )
    _equal("model_config.sha256", model["sha256"], D1_CONFIG_SHA256)
    selection = _object("selection", document["selection"])
    _exact_keys(
        "selection",
        selection,
        {
            "schema_version",
            "accepted_sha256",
            "accepted_config_sha256",
            "shot_counts",
        },
    )
    _equal(
        "selection.schema_version",
        selection["schema_version"],
        "phase05-d2-selection-artifact-v1",
    )
    _equal(
        "selection.accepted_sha256",
        selection["accepted_sha256"],
        ACCEPTED_SELECTION_SHA256,
    )
    _equal(
        "selection.accepted_config_sha256",
        selection["accepted_config_sha256"],
        SELECTION_CONFIG_SHA256,
    )
    _equal(
        "selection.shot_counts",
        selection["shot_counts"],
        list(EXPECTED_SHOT_COUNTS),
    )
    dataset = _object("dataset", document["dataset"])
    _exact_keys(
        "dataset",
        dataset,
        {
            "dataset_id",
            "batch_size",
            "retained_snapshot_sha256",
            "test_record_ids_sha256",
        },
    )
    for key, expected in (
        ("dataset_id", "bacteria_id_reference"),
        ("batch_size", 4096),
        ("retained_snapshot_sha256", RETAINED_SNAPSHOT_SHA256),
        ("test_record_ids_sha256", TEST_RECORD_IDS_SHA256),
    ):
        _equal(f"dataset.{key}", dataset[key], expected)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTES:
        raise D2RunnerValidationError(
            "config bytes",
            f"must equal {CONFIG_BYTES}",
        )
    if digest != CONFIG_SHA256:
        raise D2RunnerValidationError(
            "config sha256",
            f"must equal {CONFIG_SHA256}",
        )
    return D2RunnerConfig(
        path=path,
        sha256=digest,
        byte_count=len(raw),
        experiment_id=EXPERIMENT_ID,
        model_config_path="d1_bacteria_id_pca20_lr_sg11.json",
        model_config_sha256=D1_CONFIG_SHA256,
        accepted_selection_sha256=ACCEPTED_SELECTION_SHA256,
        accepted_selection_config_sha256=SELECTION_CONFIG_SHA256,
        seeds=EXPECTED_SEEDS,
        shot_counts=EXPECTED_SHOT_COUNTS,
        batch_size=4096,
    )


def _read_selection(path: Path) -> tuple[Mapping[str, object], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    except D2RunnerValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D2RunnerValidationError(
            "selection",
            str(error),
        ) from error
    document = _object("selection", value)
    if raw != _canonical_json_bytes(document):
        raise D2RunnerValidationError(
            "selection noncanonical",
            "must use canonical JSON encoding",
        )
    if (
        document.get("schema_version")
        != "phase05-d2-selection-artifact-v1"
        or document.get("status") != "frozen"
        or document.get("seeds") != list(EXPECTED_SEEDS)
        or document.get("shot_counts") != list(EXPECTED_SHOT_COUNTS)
        or document.get("conditions") != list(EXPECTED_CONDITIONS)
    ):
        raise D2RunnerValidationError(
            "selection",
            "does not match the D2 frozen artifact schema",
        )
    return document, raw


def _select_data(
    source: _SplitData,
    record_ids: list[str],
    *,
    path: str,
) -> _SplitData:
    index_by_id = {
        record_id: index
        for index, record_id in enumerate(source.record_ids)
    }
    if len(record_ids) != len(set(record_ids)):
        raise D2RunnerValidationError(path, "record IDs must be unique")
    try:
        indices = np.asarray(
            [index_by_id[record_id] for record_id in record_ids],
            dtype=np.int64,
        )
    except KeyError as error:
        raise D2RunnerValidationError(
            path,
            f"unknown record ID {error.args[0]!r}",
        ) from error
    return _SplitData(
        intensity=source.intensity[indices],
        labels=source.labels[indices],
        record_ids=tuple(record_ids),
        source_splits=tuple(
            source.source_splits[index]
            for index in indices
        ),
        source_rows=source.source_rows[indices],
    )


def _selected_ids(
    selection: Mapping[str, object],
    *,
    seed: int,
    shot_count: int,
) -> tuple[list[str], list[str], Mapping[str, object]]:
    selections = selection.get("selections")
    if not isinstance(selections, list) or len(selections) != 5:
        raise D2RunnerValidationError(
            "selection.selections",
            "must contain five seeds",
        )
    seed_documents = [
        _object(f"selection seed {index}", value)
        for index, value in enumerate(selections)
        if isinstance(value, Mapping) and value.get("seed") == seed
    ]
    if len(seed_documents) != 1:
        raise D2RunnerValidationError(
            "selection seed",
            f"must contain exactly one seed {seed}",
        )
    seed_document = seed_documents[0]
    classes = seed_document.get("classes")
    if not isinstance(classes, list) or len(classes) != 30:
        raise D2RunnerValidationError(
            "selection classes",
            "must contain 30 class documents",
        )
    train_ids = []
    validation_ids = []
    for class_label, raw_class in enumerate(classes):
        class_document = _object(
            f"selection class {class_label}",
            raw_class,
        )
        if class_document.get("class_label") != class_label:
            raise D2RunnerValidationError(
                f"selection class {class_label}",
                "class labels must be ordered 0..29",
            )
        train = _object(
            f"selection class {class_label} train_record_ids",
            class_document.get("train_record_ids"),
        ).get(str(shot_count))
        validation = class_document.get("validation_record_ids")
        if (
            not isinstance(train, list)
            or len(train) != shot_count
            or not isinstance(validation, list)
            or len(validation) != 10
        ):
            raise D2RunnerValidationError(
                f"selection class {class_label}",
                "train or validation count mismatch",
            )
        train_ids.extend(train)
        validation_ids.extend(validation)
    expected_train_digest = _object(
        "selection train_record_ids_sha256",
        seed_document.get("train_record_ids_sha256"),
    ).get(str(shot_count))
    expected_validation_digest = seed_document.get(
        "validation_record_ids_sha256"
    )
    if _ids_digest(train_ids) != expected_train_digest:
        raise D2RunnerValidationError(
            "selection train digest",
            "does not match exact selected IDs",
        )
    if _ids_digest(validation_ids) != expected_validation_digest:
        raise D2RunnerValidationError(
            "selection validation digest",
            "does not match exact selected IDs",
        )
    if set(train_ids) & set(validation_ids):
        raise D2RunnerValidationError(
            "selection overlap",
            "train and validation IDs must be disjoint",
        )
    return train_ids, validation_ids, {
        "train_record_ids_sha256": expected_train_digest,
        "validation_record_ids_sha256": expected_validation_digest,
    }


def _code_document() -> Mapping[str, Mapping[str, object]]:
    document = dict(_d1_code_document())
    document.pop("tools/run_phase05.py", None)
    for relative_path in CODE_PATHS:
        document[relative_path] = {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
    return document


def run_d2_few_shot_experiment(
    config_path: Path,
    dataset_path: Path,
    selection_path: Path,
    *,
    seed: int,
    shot_count: int,
    result_level: str,
) -> Mapping[str, object]:
    started = perf_counter()
    config = load_d2_runner_config(config_path)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed not in config.seeds
    ):
        raise D2RunnerValidationError(
            "seed",
            f"must be one of {config.seeds}",
        )
    if (
        isinstance(shot_count, bool)
        or not isinstance(shot_count, int)
        or shot_count not in config.shot_counts
    ):
        raise D2RunnerValidationError(
            "shot_count",
            f"must be one of {config.shot_counts}",
        )
    if result_level not in {"smoke", "provisional"}:
        raise D2RunnerValidationError(
            "result_level",
            "must be smoke or provisional",
        )
    selection_path = Path(selection_path)
    selection, selection_raw = _read_selection(selection_path)
    selection_sha256 = hashlib.sha256(selection_raw).hexdigest()
    selection_code = _object(
        "selection code",
        selection.get("code"),
    )
    for relative_path, details_value in selection_code.items():
        details = _object(
            f"selection code {relative_path}",
            details_value,
        )
        source_path = ROOT / relative_path
        if not source_path.is_file() or details != {
            "bytes": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        }:
            raise D2RunnerValidationError(
                "selection code",
                f"{relative_path} does not match current source",
            )
    dataset_path = Path(dataset_path)
    model_config_path = config.path.parent / config.model_config_path
    model_config = load_d1_runner_config(model_config_path)
    if model_config.sha256 != config.model_config_sha256:
        raise D2RunnerValidationError(
            "model_config",
            "does not match the D2 runner contract",
        )
    if result_level == "provisional":
        if selection_sha256 != config.accepted_selection_sha256:
            raise D2RunnerValidationError(
                "selection",
                (
                    f"provisional requires {config.accepted_selection_sha256}; "
                    f"observed {selection_sha256}"
                ),
            )
        selection_config_path = (
            config.path.parent / "d2_few_shot_selection.json"
        )
        validate_d2_few_shot_selection(
            selection,
            selection_config_path,
            dataset_path,
        )
        snapshot_sha256 = _sha256_file(
            dataset_path / "SHA256SUMS.sha256"
        )
        if snapshot_sha256 != RETAINED_SNAPSHOT_SHA256:
            raise D2RunnerValidationError(
                "retained snapshot",
                "does not match the D2 runner contract",
            )
        if selection.get("test", {}).get(
            "record_ids_sha256"
        ) != TEST_RECORD_IDS_SHA256:
            raise D2RunnerValidationError(
                "test identity",
                "does not match the retained D2 test split",
            )
    split_data, dataset_document = _load_dataset(
        dataset_path,
        config.batch_size,
    )
    train_ids, validation_ids, selection_digests = _selected_ids(
        selection,
        seed=seed,
        shot_count=shot_count,
    )
    train = _select_data(
        split_data["finetune"],
        train_ids,
        path="selection train",
    )
    validation = _select_data(
        split_data["finetune"],
        validation_ids,
        path="selection validation",
    )
    test = split_data["test"]
    if _ids_digest(list(test.record_ids)) != selection["test"][
        "record_ids_sha256"
    ]:
        raise D2RunnerValidationError(
            "test identity",
            "dataset test IDs do not match selection artifact",
        )
    for class_label in range(30):
        if np.count_nonzero(train.labels == class_label) != shot_count:
            raise D2RunnerValidationError(
                "selection train class balance",
                f"class {class_label} does not contain {shot_count} records",
            )
        if np.count_nonzero(validation.labels == class_label) != 10:
            raise D2RunnerValidationError(
                "selection validation class balance",
                f"class {class_label} does not contain ten records",
            )
    pipeline_configs = {}
    pipelines: dict[str, SavitzkyGolayPipeline] = {}
    for condition in model_config.conditions:
        if condition.pipeline_config is None:
            continue
        pipeline_path = model_config.path.parent / condition.pipeline_config
        pipelines[condition.condition_id] = load_savitzky_golay_pipeline(
            pipeline_path
        )
        pipeline_configs[condition.condition_id] = {
            "bytes": pipeline_path.stat().st_size,
            "path": condition.pipeline_config,
            "sha256": _sha256_file(pipeline_path),
        }
    conditions = [
        _fit_condition(
            model_config,
            condition,
            seed=seed,
            train=train,
            validation=validation,
            test=test,
            pipeline=pipelines.get(condition.condition_id),
        )
        for condition in model_config.conditions
    ]
    code = dict(_code_document())
    for relative_path in CODE_PATHS:
        code[relative_path] = {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
    return {
        "code": code,
        "conditions": conditions,
        "config": {
            "bytes": config.byte_count,
            "path": config.path.name,
            "sha256": config.sha256,
        },
        "dataset": dataset_document,
        "environment": _environment_document(),
        "experiment_id": config.experiment_id,
        "leakage_audit": _leakage_audit(
            model_config,
            train,
            validation,
            test,
        ),
        "model_config": {
            "bytes": model_config.byte_count,
            "path": model_config.path.name,
            "sha256": model_config.sha256,
        },
        "pipeline_configs": pipeline_configs,
        "result_level": result_level,
        "runtime_seconds": perf_counter() - started,
        "schema_version": RESULT_SCHEMA_VERSION,
        "seed": seed,
        "selection": {
            "artifact_bytes": len(selection_raw),
            "artifact_path": selection_path.name,
            "artifact_sha256": selection_sha256,
            "config_sha256": selection["config"]["sha256"],
            "code": selection_code,
            **selection_digests,
        },
        "shot_count": shot_count,
        "split": _split_document(
            train,
            validation,
            test,
            validation_per_class=10,
        ),
        "status": "completed",
    }


__all__ = [
    "D2RunnerConfig",
    "D2RunnerValidationError",
    "load_d2_runner_config",
    "run_d2_few_shot_experiment",
]
