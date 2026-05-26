"""Export annotation candidates from the BERT-ready manufacturing corpus."""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.data.text_ie_candidates import (
    build_candidate_pool,
    load_bert_examples_jsonl,
    write_annotation_seed_jsonl,
    write_candidate_pool_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--domain-tag", default=None)
    parser.add_argument(
        "--annotation-template-jsonl",
        default=None,
        help="Optional blank-annotation export with triples=[] records.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    examples = load_bert_examples_jsonl(args.input_jsonl)
    candidates = build_candidate_pool(
        examples,
        top_k=args.top_k,
        domain_tag=args.domain_tag,
    )
    write_candidate_pool_jsonl(args.output_jsonl, candidates)
    if args.annotation_template_jsonl:
        write_annotation_seed_jsonl(args.annotation_template_jsonl, candidates)
    print(
        json.dumps(
            {
                "input_examples": len(examples),
                "candidates": len(candidates),
                "output_jsonl": args.output_jsonl,
                "annotation_template_jsonl": args.annotation_template_jsonl,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
