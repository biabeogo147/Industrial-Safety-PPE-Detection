"""Convert gold text IE annotations into BIEO training-ready records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.data.text_ie_annotations import load_gold_annotations_jsonl
from site_safety_monitor.text_ie.dataset import (
    MissingTokenizerDependencyError,
    encode_gold_record_for_training,
    gold_annotation_to_record,
    write_encoded_dataset_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--predicates",
        nargs="+",
        default=["be_equipped_with", "perform_operations", "occurrence"],
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    annotations = load_gold_annotations_jsonl(args.annotations)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded_records: list[dict] = []
    try:
        for annotation in annotations:
            record = gold_annotation_to_record(annotation)
            encoded = encode_gold_record_for_training(
                text=record.text,
                tokens=record.tokens,
                triples=record.triples,
                predicates=args.predicates,
                max_length=args.max_length,
                encoder_name=args.encoder_name,
            )
            encoded["sentence_id"] = record.sentence_id
            encoded_records.append(encoded)
    except MissingTokenizerDependencyError as exc:
        raise SystemExit(str(exc)) from exc
    output_path = output_dir / "encoded.jsonl"
    write_encoded_dataset_jsonl(output_path, encoded_records)
    print(
        json.dumps(
            {
                "annotations": len(annotations),
                "encoded_records": len(encoded_records),
                "output_path": str(output_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
