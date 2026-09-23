from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.d1_aggregate import (  # noqa: E402
    D1AggregateValidationError,
    aggregate_d1_results,
)
from rpe.stats.paired import PairedStatisticsValidationError  # noqa: E402


def _canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = aggregate_d1_results(
            args.config,
            args.result_root,
        )
    except (
        D1AggregateValidationError,
        PairedStatisticsValidationError,
        OSError,
        ValueError,
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
    print(_canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
