"""Crawl OSHA 1910 pages into raw and interim corpus artifacts."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_project

bootstrap_project()

from site_safety_monitor.data.osha_corpus import DEFAULT_MANIFEST_PATH, DEFAULT_OUTPUT_ROOT, crawl_sources


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--source-manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sections, sentences = crawl_sources(
        output_root=args.output_root,
        manifest_path=args.source_manifest,
        limit=args.limit,
        refresh=args.refresh,
    )
    print(
        json.dumps(
            {
                "sections": len(sections),
                "sentences": len(sentences),
                "output_root": args.output_root,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
