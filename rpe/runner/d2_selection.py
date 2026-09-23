from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from rpe.downstream.bacteria_id import BacteriaIdBatchLoader


SCHEMA_VERSION = "phase05-d2-selection-v1"
ARTIFACT_SCHEMA_VERSION = "phase05-d2-selection-artifact-v1"
EXPERIMENT_ID = "d2_bacteria_id_few_shot"
CONFIG_BYTES = 858
CONFIG_SHA256 = (
    "d2a9f2eee2332a36f82e26289612f1490b56c548c4f2d35998cadfa1931da138"
)
RETAINED_SNAPSHOT_SHA256 = (
    "605866e2953479534e1830d759790afe71f6a61ffa38f3dbd239895dfa39be02"
)
TEST_RECORD_IDS_SHA256 = (
    "0bede952a2633e33796d7f3b960ddcb1386269d0d27ef9f6da062053c45df4dd"
)
SEEDS = (0, 1, 2, 3, 4)
SHOT_COUNTS = (5, 10, 20)
CONDITIONS = (
    "released_input_control",
    "released_input_plus_sg",
)
ROOT = Path(__file__).resolve().parents[2]
CODE_PATHS = (
    "rpe/downstream/bacteria_id.py",
    "rpe/io/schema.py",
    "rpe/io/store.py",
    "rpe/runner/d2_selection.py",
    "tools/freeze_d2_selection.py",
)


class D2SelectionValidationError(ValueError):
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
    raise D2SelectionValidationError(
        "config nonfinite",
        f"unsupported JSON constant {value!r}",
    )


def _object(path: str, value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise D2SelectionValidationError(path, "must be an object")
    if any(not isinstance(key, str) for key in value):
        raise D2SelectionValidationError(path, "keys must be strings")
    return value


def _exact_keys(
    path: str,
    value: Mapping[str, object],
    expected: set[str],
) -> None:
    actual = set(value)
    if actual != expected:
        raise D2SelectionValidationError(
            path,
            (
                f"key mismatch: missing={sorted(expected - actual)}, "
                f"unexpected={sorted(actual - expected)}"
            ),
        )


def _equal(path: str, observed: object, expected: object) -> None:
    if type(observed) is not type(expected) or observed != expected:
        raise D2SelectionValidationError(
            path,
            f"must equal {expected!r}",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _ids_digest(record_ids: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256(
        ("\n".join(record_ids) + "\n").encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class D2SelectionConfig:
    path: Path
    sha256: str
    byte_count: int
    seeds: tuple[int, ...]
    shot_counts: tuple[int, ...]
    validation_per_class: int
    source_records_per_class: int
    source_split: str
    test_split: str
    train_prefix_length: int
    validation_slice: tuple[int, int]
    conditions: tuple[str, ...]
    batch_size: int


def load_d2_selection_config(path: Path) -> D2SelectionConfig:
    path = Path(path)
    try:
        raw = path.read_bytes()
        value = json.loads(raw, parse_constant=_reject_nonfinite)
    except D2SelectionValidationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise D2SelectionValidationError(
            path.name or "config",
            str(error),
        ) from error
    document = _object("config", value)
    if raw != _canonical_json_bytes(document):
        raise D2SelectionValidationError(
            "config noncanonical",
            "must use canonical JSON encoding",
        )
    _exact_keys(
        "config",
        document,
        {
            "schema_version",
            "experiment_id",
            "dataset",
            "conditions",
            "seeds",
            "shot_counts",
            "validation_per_class",
            "selection",
        },
    )
    for key, expected in (
        ("schema_version", SCHEMA_VERSION),
        ("experiment_id", EXPERIMENT_ID),
        ("conditions", list(CONDITIONS)),
        ("seeds", list(SEEDS)),
        ("shot_counts", list(SHOT_COUNTS)),
        ("validation_per_class", 10),
    ):
        _equal(key, document[key], expected)
    dataset = _object("dataset", document["dataset"])
    _exact_keys(
        "dataset",
        dataset,
        {
            "dataset_id",
            "batch_size",
            "retained_snapshot_sha256",
            "source_split",
            "source_records_per_class",
            "expected_class_count",
            "test_split",
            "test_record_ids_sha256",
        },
    )
    for key, expected in (
        ("dataset_id", "bacteria_id_reference"),
        ("batch_size", 4096),
        ("retained_snapshot_sha256", RETAINED_SNAPSHOT_SHA256),
        ("source_split", "finetune"),
        ("source_records_per_class", 100),
        ("expected_class_count", 30),
        ("test_split", "test"),
        ("test_record_ids_sha256", TEST_RECORD_IDS_SHA256),
    ):
        _equal(f"dataset.{key}", dataset[key], expected)
    selection = _object("selection", document["selection"])
    _exact_keys(
        "selection",
        selection,
        {
            "method",
            "bit_generator",
            "seed_sequence_components",
            "candidate_order",
            "train_prefix_length",
            "validation_slice",
            "nested_train_subsets",
            "unused_records_per_class",
        },
    )
    for key, expected in (
        ("method", "class_local_pcg64_permutation"),
        ("bit_generator", "PCG64"),
        ("seed_sequence_components", ["seed", "class_label"]),
        ("candidate_order", "source_row_ascending"),
        ("train_prefix_length", 20),
        ("validation_slice", [20, 30]),
        ("nested_train_subsets", True),
        ("unused_records_per_class", 70),
    ):
        _equal(f"selection.{key}", selection[key], expected)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != CONFIG_BYTES:
        raise D2SelectionValidationError(
            "config bytes",
            f"must equal {CONFIG_BYTES}",
        )
    if digest != CONFIG_SHA256:
        raise D2SelectionValidationError(
            "config sha256",
            f"must equal {CONFIG_SHA256}",
        )
    return D2SelectionConfig(
        path=path,
        sha256=digest,
        byte_count=len(raw),
        seeds=SEEDS,
        shot_counts=SHOT_COUNTS,
        validation_per_class=10,
        source_records_per_class=100,
        source_split="finetune",
        test_split="test",
        train_prefix_length=20,
        validation_slice=(20, 30),
        conditions=CONDITIONS,
        batch_size=4096,
    )


def _dataset_inventory(
    dataset_path: Path,
    batch_size: int,
) -> tuple[dict[int, list[tuple[int, str]]], list[str], Mapping[str, object]]:
    candidates = {class_label: [] for class_label in range(30)}
    test_ids = []
    split_counts = {"finetune": 0, "reference": 0, "test": 0}
    with BacteriaIdBatchLoader(
        dataset_path,
        batch_size=batch_size,
    ) as loader:
        for batch in loader.iter_batches():
            split_counts[batch.source_split] += len(batch.record_ids)
            if batch.source_split == "finetune":
                for record_id, source_row, class_label in zip(
                    batch.record_ids,
                    batch.source_rows,
                    batch.class_labels,
                    strict=True,
                ):
                    candidates[int(class_label)].append(
                        (int(source_row), record_id)
                    )
            elif batch.source_split == "test":
                test_ids.extend(batch.record_ids)
    for class_label, values in candidates.items():
        values.sort()
        if len(values) != 100:
            raise D2SelectionValidationError(
                f"finetune class {class_label}",
                f"must contain 100 records; observed {len(values)}",
            )
    test_digest = _ids_digest(test_ids)
    if len(test_ids) != 3000 or test_digest != TEST_RECORD_IDS_SHA256:
        raise D2SelectionValidationError(
            "test record identity",
            (
                f"expected count=3000 sha256={TEST_RECORD_IDS_SHA256}; "
                f"observed count={len(test_ids)} sha256={test_digest}"
            ),
        )
    dataset_path = Path(dataset_path)
    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(dataset_path.iterdir())
        if path.is_file()
    }
    return candidates, test_ids, {
        "dataset_id": dataset_path.name,
        "files": files,
        "record_count": sum(split_counts.values()),
        "split_counts": split_counts,
    }


def _expected_class_selection(
    *,
    seed: int,
    class_label: int,
    candidates: list[tuple[int, str]],
    shot_counts: tuple[int, ...],
    train_prefix_length: int,
    validation_slice: tuple[int, int],
) -> Mapping[str, object]:
    ordered_candidates = [
        record_id
        for _, record_id in candidates
    ]
    generator = np.random.Generator(
        np.random.PCG64(
            np.random.SeedSequence([seed, class_label])
        )
    )
    permutation = generator.permutation(len(ordered_candidates))
    selected_order = [
        ordered_candidates[int(index)]
        for index in permutation
    ]
    ordered_train = selected_order[:train_prefix_length]
    validation = selected_order[
        validation_slice[0] : validation_slice[1]
    ]
    train_ids = {
        str(shot_count): ordered_train[:shot_count]
        for shot_count in shot_counts
    }
    return {
        "candidate_count": len(ordered_candidates),
        "class_label": class_label,
        "ordered_train_record_ids": ordered_train,
        "source_split": "finetune",
        "train_record_ids": train_ids,
        "train_record_ids_sha256": {
            str(shot_count): _ids_digest(train_ids[str(shot_count)])
            for shot_count in shot_counts
        },
        "validation_record_ids": validation,
        "validation_record_ids_sha256": _ids_digest(validation),
    }


def _code_document() -> Mapping[str, Mapping[str, object]]:
    return {
        relative_path: {
            "bytes": (ROOT / relative_path).stat().st_size,
            "sha256": _sha256_file(ROOT / relative_path),
        }
        for relative_path in CODE_PATHS
    }


def build_d2_few_shot_selection(
    config_path: Path,
    dataset_path: Path,
) -> Mapping[str, object]:
    config = load_d2_selection_config(config_path)
    dataset_path = Path(dataset_path)
    snapshot_path = dataset_path / "SHA256SUMS.sha256"
    snapshot_sha256 = _sha256_file(snapshot_path)
    if snapshot_sha256 != RETAINED_SNAPSHOT_SHA256:
        raise D2SelectionValidationError(
            "retained snapshot",
            (
                f"must equal {RETAINED_SNAPSHOT_SHA256}; "
                f"observed {snapshot_sha256}"
            ),
        )
    candidates, test_ids, dataset = _dataset_inventory(
        dataset_path,
        config.batch_size,
    )
    selections = []
    for seed in config.seeds:
        class_documents = []
        all_train = {
            shot_count: []
            for shot_count in config.shot_counts
        }
        all_validation = []
        for class_label in range(30):
            class_document = _expected_class_selection(
                seed=seed,
                class_label=class_label,
                candidates=candidates[class_label],
                shot_counts=config.shot_counts,
                train_prefix_length=config.train_prefix_length,
                validation_slice=config.validation_slice,
            )
            class_documents.append(class_document)
            for shot_count in config.shot_counts:
                all_train[shot_count].extend(
                    class_document["train_record_ids"][str(shot_count)]
                )
            all_validation.extend(
                class_document["validation_record_ids"]
            )
        selections.append(
            {
                "classes": class_documents,
                "seed": seed,
                "train_record_ids_sha256": {
                    str(shot_count): _ids_digest(
                        all_train[shot_count]
                    )
                    for shot_count in config.shot_counts
                },
                "validation_record_ids_sha256": _ids_digest(
                    all_validation
                ),
            }
        )
    artifact = {
        "code": _code_document(),
        "conditions": list(config.conditions),
        "config": {
            "bytes": config.byte_count,
            "path": config.path.name,
            "sha256": config.sha256,
        },
        "dataset": dataset,
        "experiment_id": EXPERIMENT_ID,
        "seeds": list(config.seeds),
        "selections": selections,
        "shot_counts": list(config.shot_counts),
        "status": "frozen",
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "test": {
            "count": len(test_ids),
            "record_ids_sha256": _ids_digest(test_ids),
            "source_split": config.test_split,
        },
        "validation_per_class": config.validation_per_class,
    }
    validate_d2_few_shot_selection(
        artifact,
        config_path,
        dataset_path,
    )
    return artifact


def validate_d2_few_shot_selection(
    artifact: Mapping[str, object],
    config_path: Path,
    dataset_path: Path,
) -> None:
    config = load_d2_selection_config(config_path)
    document = _object("artifact", artifact)
    _exact_keys(
        "artifact",
        document,
        {
            "schema_version",
            "status",
            "experiment_id",
            "config",
            "dataset",
            "conditions",
            "code",
            "seeds",
            "shot_counts",
            "validation_per_class",
            "test",
            "selections",
        },
    )
    for key, expected in (
        ("schema_version", ARTIFACT_SCHEMA_VERSION),
        ("status", "frozen"),
        ("experiment_id", EXPERIMENT_ID),
        ("conditions", list(CONDITIONS)),
        ("seeds", list(SEEDS)),
        ("shot_counts", list(SHOT_COUNTS)),
        ("validation_per_class", 10),
    ):
        _equal(key, document[key], expected)
    config_document = _object("artifact.config", document["config"])
    if config_document != {
        "bytes": config.byte_count,
        "path": config.path.name,
        "sha256": config.sha256,
    }:
        raise D2SelectionValidationError(
            "artifact.config",
            "does not match the frozen config",
        )
    if document["code"] != _code_document():
        raise D2SelectionValidationError(
            "artifact.code",
            "does not match current selection source files",
        )
    candidates, test_ids, expected_dataset = _dataset_inventory(
        Path(dataset_path),
        config.batch_size,
    )
    if document["dataset"] != expected_dataset:
        raise D2SelectionValidationError(
            "artifact.dataset",
            "does not match the retained dataset",
        )
    test = _object("artifact.test", document["test"])
    if test != {
        "count": len(test_ids),
        "record_ids_sha256": _ids_digest(test_ids),
        "source_split": "test",
    }:
        raise D2SelectionValidationError(
            "artifact.test",
            "test identity does not match the retained test split",
        )
    selections = document["selections"]
    if not isinstance(selections, list) or len(selections) != 5:
        raise D2SelectionValidationError(
            "artifact.selections",
            "must contain exactly five seed documents",
        )
    observed_seeds = []
    for seed_index, seed_document_value in enumerate(selections):
        seed_document = _object(
            f"artifact.selections[{seed_index}]",
            seed_document_value,
        )
        _exact_keys(
            f"artifact.selections[{seed_index}]",
            seed_document,
            {
                "seed",
                "classes",
                "train_record_ids_sha256",
                "validation_record_ids_sha256",
            },
        )
        seed = seed_document["seed"]
        if seed not in SEEDS or isinstance(seed, bool):
            raise D2SelectionValidationError(
                f"artifact.selections[{seed_index}].seed",
                "must be one of 0..4",
            )
        observed_seeds.append(seed)
        classes = seed_document["classes"]
        if not isinstance(classes, list) or len(classes) != 30:
            raise D2SelectionValidationError(
                f"seed {seed} classes",
                "must contain exactly 30 class documents",
            )
        all_train = {shot_count: [] for shot_count in SHOT_COUNTS}
        all_validation = []
        for class_index, class_value in enumerate(classes):
            class_document = _object(
                f"seed {seed} classes[{class_index}]",
                class_value,
            )
            _exact_keys(
                f"seed {seed} classes[{class_index}]",
                class_document,
                {
                    "candidate_count",
                    "class_label",
                    "source_split",
                    "ordered_train_record_ids",
                    "train_record_ids",
                    "train_record_ids_sha256",
                    "validation_record_ids",
                    "validation_record_ids_sha256",
                },
            )
            class_label = class_document["class_label"]
            if class_label != class_index:
                raise D2SelectionValidationError(
                    f"seed {seed} class_label",
                    "class documents must be ordered 0..29",
                )
            if (
                class_document["candidate_count"] != 100
                or class_document["source_split"] != "finetune"
            ):
                raise D2SelectionValidationError(
                    f"seed {seed} class {class_label} candidate",
                    "must contain 100 finetune candidates",
                )
            valid_candidates = {
                record_id
                for _, record_id in candidates[class_label]
            }
            ordered = class_document["ordered_train_record_ids"]
            validation = class_document["validation_record_ids"]
            if (
                not isinstance(ordered, list)
                or len(ordered) != 20
                or len(set(ordered)) != 20
                or not set(ordered) <= valid_candidates
            ):
                raise D2SelectionValidationError(
                    f"seed {seed} class {class_label} nested",
                    "ordered train IDs must be 20 unique class candidates",
                )
            if (
                not isinstance(validation, list)
                or len(validation) != 10
                or len(set(validation)) != 10
                or not set(validation) <= valid_candidates
                or set(ordered) & set(validation)
            ):
                raise D2SelectionValidationError(
                    f"seed {seed} class {class_label} overlap",
                    "validation IDs must be ten unique disjoint candidates",
                )
            train_ids = _object(
                f"seed {seed} class {class_label} train_record_ids",
                class_document["train_record_ids"],
            )
            train_digests = _object(
                f"seed {seed} class {class_label} train_record_ids_sha256",
                class_document["train_record_ids_sha256"],
            )
            for shot_count in SHOT_COUNTS:
                selected = train_ids.get(str(shot_count))
                if selected != ordered[:shot_count]:
                    raise D2SelectionValidationError(
                        f"seed {seed} class {class_label} nested",
                        f"{shot_count}-shot IDs must be the ordered prefix",
                    )
                if train_digests.get(str(shot_count)) != _ids_digest(
                    selected
                ):
                    raise D2SelectionValidationError(
                        f"seed {seed} class {class_label} digest",
                        f"{shot_count}-shot digest mismatch",
                    )
                all_train[shot_count].extend(selected)
            if class_document[
                "validation_record_ids_sha256"
            ] != _ids_digest(validation):
                raise D2SelectionValidationError(
                    f"seed {seed} class {class_label} digest",
                    "validation digest mismatch",
                )
            expected_class_document = _expected_class_selection(
                seed=seed,
                class_label=class_label,
                candidates=candidates[class_label],
                shot_counts=config.shot_counts,
                train_prefix_length=config.train_prefix_length,
                validation_slice=config.validation_slice,
            )
            if class_document != expected_class_document:
                raise D2SelectionValidationError(
                    f"seed {seed} class {class_label} deterministic selection",
                    "does not equal the frozen class-local PCG64 selection",
                )
            all_validation.extend(validation)
        expected_train_digests = {
            str(shot_count): _ids_digest(all_train[shot_count])
            for shot_count in SHOT_COUNTS
        }
        if seed_document[
            "train_record_ids_sha256"
        ] != expected_train_digests:
            raise D2SelectionValidationError(
                f"seed {seed} train digest",
                "global train digest mismatch",
            )
        if seed_document[
            "validation_record_ids_sha256"
        ] != _ids_digest(all_validation):
            raise D2SelectionValidationError(
                f"seed {seed} validation digest",
                "global validation digest mismatch",
            )
    if tuple(observed_seeds) != SEEDS:
        raise D2SelectionValidationError(
            "artifact.seeds",
            "seed documents must be ordered 0..4",
        )


__all__ = [
    "D2SelectionConfig",
    "D2SelectionValidationError",
    "build_d2_few_shot_selection",
    "load_d2_selection_config",
    "validate_d2_few_shot_selection",
]
