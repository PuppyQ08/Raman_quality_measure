from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.data.fetch_sources import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(default_group="bacteria_id"))
