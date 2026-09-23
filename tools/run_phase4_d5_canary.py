from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_d5_canary import (  # noqa: E402
    Phase4D5CanaryError,
    build_phase4_d5_canary,
    verify_phase4_d5_canary,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.verify is not None:
            result = dict(verify_phase4_d5_canary(args.verify, reexecute=True))
        else:
            summary = build_phase4_d5_canary(args.output_root)
            result = {
                "curve_count": summary.curve_count,
                "library_count": summary.library_count,
                "path": str(summary.path),
                "query_count": summary.query_count,
                "response_count": summary.response_count,
                "run_id": summary.run_id,
            }
    except (OSError, ValueError, Phase4D5CanaryError) as error:
        print(
            json.dumps(
                {"error": str(error), "error_type": type(error).__name__, "status": "failed"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
