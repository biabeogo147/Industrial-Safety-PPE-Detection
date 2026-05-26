"""Evaluate simple recall@k for scene graph predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.scene_graph.evaluate import recall_at_k


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate scene graph retrieval recall.")
    parser.add_argument("--retrieved", required=True, help="Path to ranked predictions JSON.")
    parser.add_argument("--relevant", required=True, help="Path to relevant labels JSON.")
    parser.add_argument("--k", type=int, default=20, help="Cutoff for recall.")
    args = parser.parse_args()
    retrieved = json.loads(Path(args.retrieved).read_text(encoding="utf-8"))
    relevant = set(json.loads(Path(args.relevant).read_text(encoding="utf-8")))
    print(json.dumps({"recall_at_k": recall_at_k(retrieved, relevant, args.k)}, indent=2))


if __name__ == "__main__":
    main()
