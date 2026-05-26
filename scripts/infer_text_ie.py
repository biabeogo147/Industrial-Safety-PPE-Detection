"""Run manufacturing text IE inference from raw text."""

from __future__ import annotations

import argparse
import json

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from site_safety_monitor.text_ie.dataset import MissingTokenizerDependencyError
from site_safety_monitor.text_ie.infer import infer_text_triples
from site_safety_monitor.text_ie.model import MissingMLDependencyError
from site_safety_monitor.text_ie.train import load_training_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_training_config(args.config)
    try:
        triples = infer_text_triples(
            text=args.text,
            checkpoint_path=args.checkpoint,
            encoder_name=config.encoder_name,
            predicates=config.predicates,
            tag_space=list(build_tag_space(config.predicates)),
            max_length=config.max_length,
            threshold=config.threshold,
        )
    except (MissingMLDependencyError, MissingTokenizerDependencyError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            [triple.as_normalized_tuple() for triple in triples],
            indent=2,
        )
    )


def build_tag_space(predicates):
    from site_safety_monitor.text_ie.dataset import build_label_vocabulary

    tag_space, _ = build_label_vocabulary(predicates)
    return tag_space


if __name__ == "__main__":
    main()
