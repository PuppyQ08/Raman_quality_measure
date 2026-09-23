from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rpe.runner.phase6_release_metadata import (  # noqa: E402
    DEFAULT_CONFIG,
    _canonical,
    build_live_release_metadata_config_document,
    build_phase6_release_metadata,
)
from rpe.runner.phase6_release_metadata_verifier import (  # noqa: E402
    verify_phase6_release_metadata,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or verify Phase 6 release metadata")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build")
    build.add_argument("--output-root", required=True)
    build.add_argument("--config", default=str(DEFAULT_CONFIG))
    build.add_argument("--mode", choices=("formal", "fixture"), default="formal")
    build.add_argument("--step8-publication-core-path")

    verify = commands.add_parser("verify")
    verify.add_argument("--run-path", required=True)
    verify.add_argument("--config", default=str(DEFAULT_CONFIG))

    freeze = commands.add_parser("freeze-config")
    freeze.add_argument("--output", required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        summary = build_phase6_release_metadata(
            Path(args.output_root),
            config_path=Path(args.config),
            mode=args.mode,
            step8_publication_core_path=None
            if args.step8_publication_core_path is None
            else Path(args.step8_publication_core_path),
        )
        print(f"{summary.run_id}\t{summary.path}")
        return 0
    if args.command == "verify":
        summary = verify_phase6_release_metadata(
            Path(args.run_path),
            config_path=Path(args.config),
        )
        print(f"{summary.run_id}\t{summary.path}")
        return 0
    Path(args.output).write_bytes(_canonical(build_live_release_metadata_config_document()))
    print(Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
