from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_d4_protocol_b_eligibility import (  # noqa: E402
    Phase4D4ProtocolBEligibilityError,
    build_phase4_d4_protocol_b_eligibility,
)
from rpe.runner.phase4_d4_protocol_b_eligibility_verifier import (  # noqa: E402
    Phase4D4ProtocolBEligibilityVerifierError,
    verify_phase4_d4_protocol_b_eligibility,
)


def _summary_json(summary) -> str:
    return json.dumps(
        {
            "path": str(summary.path),
            "run_id": summary.run_id,
            "status": summary.status,
            "mixture_record_count": summary.mixture_record_count,
            "role_condition_summary_count": summary.role_condition_summary_count,
            "model_condition_readiness_count": summary.model_condition_readiness_count,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "build":
            summary = build_phase4_d4_protocol_b_eligibility(arguments.output_root)
        else:
            summary = verify_phase4_d4_protocol_b_eligibility(arguments.run_path)
    except (
        OSError,
        ValueError,
        Phase4D4ProtocolBEligibilityError,
        Phase4D4ProtocolBEligibilityVerifierError,
    ) as error:
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
    print(_summary_json(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
