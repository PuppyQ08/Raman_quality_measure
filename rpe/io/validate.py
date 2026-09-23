from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from rpe.io.schema import SchemaValidationError
from rpe.io.store import DatasetValidationError, ValidationSummary, validate_dataset


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _summary_document(summary: ValidationSummary) -> dict[str, object]:
    return {
        "axis_group_count": summary.axis_group_count,
        "checked_files": list(summary.checked_files),
        "dataset_id": summary.dataset_id,
        "preprocessing_status_counts": dict(
            summary.preprocessing_status_counts
        ),
        "record_count": summary.record_count,
        "status": "verified",
        "target_presence_counts": dict(summary.target_presence_counts),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-checksums", action="store_true")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_dataset(
            args.dataset,
            verify_checksums=not args.no_checksums,
        )
    except (
        DatasetValidationError,
        SchemaValidationError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(
            _canonical_json(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "status": "failed",
                }
            ),
            file=sys.stderr,
        )
        return 1

    print(_canonical_json(_summary_document(summary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
