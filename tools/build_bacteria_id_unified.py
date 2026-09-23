from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.io.bacteria_id import (  # noqa: E402
    BacteriaIdConversionSummary,
    BacteriaIdRuntimeMetrics,
    BacteriaIdValidationError,
    _build_bacteria_id_unified_with_metrics,
)
from rpe.io.schema import SchemaValidationError  # noqa: E402
from rpe.io.store import DatasetValidationError  # noqa: E402


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _dataset_document(summary) -> dict[str, object]:
    return {
        "axis_id": summary.axis_id,
        "dataset_id": summary.dataset_id,
        "files": {
            name: dict(details)
            for name, details in summary.files.items()
        },
        "output_bytes": summary.output_bytes,
        "record_count": summary.record_count,
    }


def _summary_document(
    summary: BacteriaIdConversionSummary,
    metrics: BacteriaIdRuntimeMetrics,
) -> dict[str, object]:
    return {
        "status": "written",
        "reference": _dataset_document(summary.reference),
        "clinical": _dataset_document(summary.clinical),
        "receipt": {
            "path": summary.receipt_path.as_posix(),
            "sha256": summary.receipt_sha256,
        },
        "metrics": asdict(metrics),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary, metrics = _build_bacteria_id_unified_with_metrics(
            args.raw_root,
            args.output_root,
            overwrite=args.overwrite,
        )
    except (
        BacteriaIdValidationError,
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

    print(_canonical_json(_summary_document(summary, metrics)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
