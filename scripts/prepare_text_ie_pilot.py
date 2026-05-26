"""Freeze a pilot annotation set from the manufacturing text IE candidate pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.data.text_ie_candidates import (
    load_candidate_sentences_jsonl,
    write_annotation_seed_jsonl,
    write_candidate_pool_jsonl,
)
from site_safety_monitor.data.text_ie_pilot import (
    PilotSelectionConfig,
    build_pilot_scope_markdown,
    freeze_pilot_candidates,
    split_pilot_candidates,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--total-size", type=int, default=60)
    parser.add_argument("--train-size", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=9)
    parser.add_argument("--test-size", type=int, default=9)
    parser.add_argument("--max-tokens", type=int, default=56)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = PilotSelectionConfig(
        total_size=args.total_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        max_tokens=args.max_tokens,
    )
    candidates = load_candidate_sentences_jsonl(args.input_jsonl)
    selected = freeze_pilot_candidates(candidates, config=config)
    train, val, test = split_pilot_candidates(selected, config=config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_candidate_pool_jsonl(output_dir / "pilot_selected_candidates.jsonl", selected)
    write_annotation_seed_jsonl(output_dir / "pilot_train_seed.jsonl", train)
    write_annotation_seed_jsonl(output_dir / "pilot_val_seed.jsonl", val)
    write_annotation_seed_jsonl(output_dir / "pilot_test_seed.jsonl", test)
    (output_dir / "pilot_scope.md").write_text(
        build_pilot_scope_markdown(selected, config=config),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "selected": len(selected),
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
