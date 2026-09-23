from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_d5_protocol_b_eligibility import (  # noqa: E402
    Phase4D5ProtocolBEligibilityError,
    build_phase4_d5_protocol_b_eligibility,
)
from rpe.runner.phase4_d5_protocol_b_eligibility_verifier import (  # noqa: E402
    verify_phase4_d5_protocol_b_eligibility,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--worker-count", type=int, default=16)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    verify.add_argument("--worker-count", type=int, default=12)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "build":
            summary = build_phase4_d5_protocol_b_eligibility(
                arguments.output_root,
                worker_count=arguments.worker_count,
            )
        else:
            summary = verify_phase4_d5_protocol_b_eligibility(
                arguments.run_path,
                worker_count=arguments.worker_count,
            )
    except (OSError, ValueError, Phase4D5ProtocolBEligibilityError) as error:
        print(
            json.dumps(
                {
                    "error": str(error),
                    "error_type": type(error).__name__,
                    "status": "failed",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "class_summary_count": summary.class_summary_count,
                "operator_cell_count": summary.operator_cell_count,
                "path": str(summary.path),
                "record_condition_count": summary.record_condition_count,
                "role_occurrence_count": summary.role_occurrence_count,
                "run_id": summary.run_id,
                "status": summary.status,
                "unique_record_count": summary.unique_record_count,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
