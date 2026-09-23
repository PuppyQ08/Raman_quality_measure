from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_appendix_audits import build_phase6_appendix_audits  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or independently verify Phase 6 appendix audits")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", required=True)
    build.add_argument("--worker-count", required=True, type=int)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", required=True)
    verify.add_argument("--worker-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    if args.worker_count <= 0:
        _parser().error("--worker-count must be positive")
    if args.command == "build":
        summary = build_phase6_appendix_audits(Path(args.output_root), worker_count=args.worker_count)
    else:
        from rpe.runner.phase6_appendix_audits_verifier import verify_phase6_appendix_audits
        summary = verify_phase6_appendix_audits(Path(args.run_path), worker_count=args.worker_count)
    print(f"{summary.run_id}\t{summary.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
