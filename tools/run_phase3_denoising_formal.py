from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase3_denoising_formal import (
    DenoisingFormalError,
    build_phase3_denoising_formal,
)
from rpe.runner.phase3_denoising_formal_verifier import (
    verify_phase3_denoising_formal,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or independently verify Phase 3 denoising formal coverage"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    build.add_argument("--worker-count", type=int, default=8)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    verify.add_argument("--worker-count", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            summary = build_phase3_denoising_formal(
                args.output_root, worker_count=args.worker_count
            )
            status = "built"
        else:
            summary = verify_phase3_denoising_formal(
                args.run_path, worker_count=args.worker_count, project_root=ROOT
            )
            status = "verified"
    except (DenoisingFormalError, OSError, ValueError) as error:
        print(
            json.dumps(
                {"error": str(error), "error_type": type(error).__name__, "status": "error"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "path": str(summary.path),
                "run_id": summary.run_id,
                "status": status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
