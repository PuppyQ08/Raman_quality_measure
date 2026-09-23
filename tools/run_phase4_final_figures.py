from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase4_final_figures import build_phase4_final_figures  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or independently verify Phase 4 final figures"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", type=Path, required=True)
    return parser


def _print_json(document: dict[str, object], *, stream: object = sys.stdout) -> None:
    print(json.dumps(document, sort_keys=True, separators=(",", ":")), file=stream)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "build":
            summary = build_phase4_final_figures(arguments.output_root, worker_count=1)
            payload = {
                "figure1_size_px": list(summary.figure1_size_px),
                "figure2_size_px": list(summary.figure2_size_px),
                "path": str(summary.path),
                "run_id": summary.run_id,
                "status": summary.status,
            }
        else:
            module = importlib.import_module("rpe.runner.phase4_final_figures_verifier")
            summary = module.verify_phase4_final_figures(arguments.run_path, worker_count=1)
            payload = {
                "path": str(summary.path),
                "run_id": summary.run_id,
                "status": summary.status,
                "verified_file_count": summary.verified_file_count,
            }
    except (OSError, ValueError) as error:
        _print_json(
            {
                "error": str(error),
                "error_type": type(error).__name__,
                "status": "failed",
            },
            stream=sys.stderr,
        )
        return 1
    _print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
