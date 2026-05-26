"""Runtime bootstrap helpers for repo-local scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def bootstrap_project() -> None:
    """Backward-compatible bootstrap entry point for repo scripts."""

    ensure_src_on_path()
