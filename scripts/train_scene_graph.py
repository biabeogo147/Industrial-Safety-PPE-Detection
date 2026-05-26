"""Summarize scene graph training data."""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.scene_graph.train import scene_graph_summary_as_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize scene graph training data.")
    parser.add_argument("--dataset", required=True, help="Path to scene graph dataset JSON.")
    args = parser.parse_args()
    print(json.dumps(scene_graph_summary_as_dict(args.dataset), indent=2))


if __name__ == "__main__":
    main()
