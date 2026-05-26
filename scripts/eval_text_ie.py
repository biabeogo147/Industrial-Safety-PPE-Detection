"""Evaluate exact triple extraction metrics from JSON lists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.text_ie.metrics import triple_f1


def _load_tuple_set(path: str) -> set[tuple[str, str, str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {tuple(item) for item in payload}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate triple extraction results.")
    parser.add_argument("--predicted", required=True, help="Path to predicted triples JSON.")
    parser.add_argument("--gold", required=True, help="Path to gold triples JSON.")
    args = parser.parse_args()
    precision, recall, f1 = triple_f1(
        predicted=_load_tuple_set(args.predicted),
        gold=_load_tuple_set(args.gold),
    )
    print(json.dumps({"precision": precision, "recall": recall, "f1": f1}, indent=2))


if __name__ == "__main__":
    main()
