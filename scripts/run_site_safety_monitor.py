"""Run the end-to-end Site Safety Monitor flow."""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.pipelines.run_monitor import run_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Site Safety Monitor on one regulation and scene pair.")
    parser.add_argument("--regulation", required=True, help="Path to regulation triples JSON.")
    parser.add_argument("--scene", required=True, help="Path to scene triples JSON.")
    parser.add_argument("--worker-id", default=None, help="Optional explicit worker identifier.")
    args = parser.parse_args()
    print(
        json.dumps(
            run_case(
                regulation_path=args.regulation,
                scene_path=args.scene,
                worker_id=args.worker_id,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
