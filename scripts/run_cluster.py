"""Run PEARL experiments across a single multi-GPU node without installing."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pearl.cli.cluster_run import main


if __name__ == "__main__":
    raise SystemExit(main())
