"""Materialize the frozen full manufacturing text IE gold set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.data.text_ie_annotations import load_gold_annotations_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-jsonl", required=True)
    parser.add_argument("--overrides-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser


def load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def load_overrides(path: str | Path) -> tuple[dict[str, list[dict]], set[str]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    annotations = payload.get("annotations", [])
    reviewed_empty_ids = set(payload.get("reviewed_empty_ids", []))
    overrides: dict[str, list[dict]] = {}
    for record in annotations:
        sentence_id = record["sentence_id"]
        if sentence_id in overrides:
            raise ValueError(f"Duplicate override for sentence_id={sentence_id}")
        overrides[sentence_id] = record.get("triples", [])
    duplicate_ids = sorted(sentence_id for sentence_id in reviewed_empty_ids if sentence_id in overrides)
    if duplicate_ids:
        raise ValueError(f"Sentence ids cannot be both annotated and reviewed-empty: {duplicate_ids}")
    return overrides, reviewed_empty_ids


def materialize_annotations(
    seed_records: list[dict],
    overrides: dict[str, list[dict]],
    reviewed_empty_ids: set[str],
) -> list[dict]:
    output: list[dict] = []
    seen_ids: set[str] = set()
    for record in seed_records:
        sentence_id = record["sentence_id"]
        seen_ids.add(sentence_id)
        frozen = dict(record)
        frozen["triples"] = overrides.get(sentence_id, [])
        output.append(frozen)

    unknown_ids = sorted(
        sentence_id
        for sentence_id in set(overrides) | reviewed_empty_ids
        if sentence_id not in seen_ids
    )
    if unknown_ids:
        raise ValueError(f"Overrides reference unknown sentence ids: {unknown_ids}")

    missing_review_ids = sorted(
        sentence_id
        for sentence_id in seen_ids
        if sentence_id not in overrides and sentence_id not in reviewed_empty_ids
    )
    if missing_review_ids:
        raise ValueError(f"Some seed records were not reviewed: {missing_review_ids}")
    return output


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def summarize(records: list[dict]) -> dict[str, int]:
    return {
        "sentences": len(records),
        "annotated_sentences": sum(1 for record in records if record.get("triples")),
        "triples": sum(len(record.get("triples", [])) for record in records),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seed_records = load_jsonl(args.seed_jsonl)
    overrides, reviewed_empty_ids = load_overrides(args.overrides_json)
    materialized = materialize_annotations(seed_records, overrides, reviewed_empty_ids)
    write_jsonl(args.output_jsonl, materialized)
    validated = load_gold_annotations_jsonl(args.output_jsonl)

    print(
        json.dumps(
            {
                "summary": summarize(materialized),
                "reviewed_empty_sentences": len(reviewed_empty_ids),
                "validated_sentences": len(validated),
                "output_jsonl": args.output_jsonl,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
