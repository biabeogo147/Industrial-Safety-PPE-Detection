"""Prepare BERT-ready OSHA sentence corpus from interim crawl artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.data.bert_corpus import build_bert_corpus, export_bert_corpus, load_bert_corpus_rules
from site_safety_monitor.data.corpus_models import SentenceRecord
from site_safety_monitor.data.osha_corpus import DEFAULT_OUTPUT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--rules-path", default=None)
    return parser


def _load_sentences(path: Path) -> list[SentenceRecord]:
    records: list[SentenceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            records.append(
                SentenceRecord(
                    sentence_id=payload["sentence_id"],
                    source_standard=payload["source_standard"],
                    section_ref=payload["section_ref"],
                    source_url=payload["source_url"],
                    text=payload["text"],
                    language=payload.get("language", "en"),
                    domain_tags=tuple(payload.get("domain_tags", [])),
                )
            )
    return records


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    interim_path = Path(args.output_root) / "text_corpus" / "interim" / "sentences.jsonl"
    sentences = _load_sentences(interim_path)
    rules = load_bert_corpus_rules(args.rules_path) if args.rules_path else load_bert_corpus_rules()
    examples = build_bert_corpus(sentences, rules=rules)
    output_paths = export_bert_corpus(examples, output_root=args.output_root)
    print(
        json.dumps(
            {
                "examples": len(examples),
                "output_paths": {name: str(path) for name, path in output_paths.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
