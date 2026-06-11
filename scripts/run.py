"""Run PEARL experiments without installing the console script."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pearl.cli.run import main


if __name__ == "__main__":
    main()
