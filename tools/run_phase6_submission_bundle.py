from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_submission_bundle import build_phase6_submission_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or independently verify the Phase 6 Step 11 submission bundle"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--output-root", required=True)
    build.add_argument("--config", required=True)
    build.add_argument("--project-root", default=str(ROOT))

    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", required=True)
    verify.add_argument("--project-root", default=str(ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        summary = build_phase6_submission_bundle(
            Path(args.output_root),
            config_path=Path(args.config),
            project_root=Path(args.project_root),
        )
    else:
        from rpe.runner.phase6_submission_bundle_verifier import (
            verify_phase6_submission_bundle,
        )

        summary = verify_phase6_submission_bundle(
            Path(args.run_path),
            project_root=Path(args.project_root),
        )
    print(f"{summary.run_id}\t{summary.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
