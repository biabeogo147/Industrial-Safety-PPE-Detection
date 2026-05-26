"""Summarize text IE training data."""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.text_ie.train import training_summary_as_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize regulation training data.")
    parser.add_argument("--dataset", required=True, help="Path to regulation dataset JSON.")
    args = parser.parse_args()
    print(json.dumps(training_summary_as_dict(args.dataset), indent=2))


if __name__ == "__main__":
    main()
