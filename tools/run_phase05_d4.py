from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.downstream.sugar_quantitative import (  # noqa: E402
    D4LoaderValidationError,
)
from rpe.runner.d4_sugar import (  # noqa: E402
    D4RunnerValidationError,
    run_d4_provisional_experiment,
)


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
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--result-level",
        choices=("provisional", "complete_seed"),
        required=True,
    )
    args = parser.parse_args(argv)
    try:
        result = run_d4_provisional_experiment(
            args.config,
            args.archive,
            seed=args.seed,
            result_level=args.result_level,
        )
    except (
        D4LoaderValidationError,
        D4RunnerValidationError,
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
