"""Prepare clean OSHA sentence corpora for later BERT tokenization."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from site_safety_monitor.data.corpus_models import BertSentenceExample, SentenceRecord
from site_safety_monitor.data.osha_corpus import DEFAULT_OUTPUT_ROOT


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = PACKAGE_ROOT / "configs" / "site_safety_monitor" / "corpus" / "bert_corpus_rules.yaml"
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TITLE_PATTERN = re.compile(r"^\s*\d{4}\.\d+[a-z0-9()\-]*\s*-\s+", re.IGNORECASE)
_ALPHA_TOKEN_PATTERN = re.compile(r"[a-zA-Z]")
_LABEL_PATTERN = re.compile(r"^(title|part number title|subpart|subpart title|standard number)\s*:", re.IGNORECASE)
_FEDERAL_REGISTER_PATTERN = re.compile(r"^\[\s*\d+\s+fr\b", re.IGNORECASE)
_MOJIBAKE_REPLACEMENTS = {
    "â€™": "'",
    "â€œ": '"',
    "â€\x9d": '"',
    "â€“": "-",
    "Ã©": "é",
}


def load_bert_corpus_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def normalize_corpus_text(text: str) -> str:
    normalized = _WHITESPACE_PATTERN.sub(" ", text).strip()
    for source, target in _MOJIBAKE_REPLACEMENTS.items():
        normalized = normalized.replace(source, target)
    return normalized


def whitespace_tokenize(text: str) -> tuple[str, ...]:
    return tuple(token for token in normalize_corpus_text(text).split(" ") if token)


def is_bert_ready_sentence(sentence: SentenceRecord, rules: dict) -> bool:
    normalized = normalize_corpus_text(sentence.text)
    lowered = normalized.lower()
    if not normalized:
        return False
    if _TITLE_PATTERN.match(normalized):
        return False
    if _LABEL_PATTERN.match(normalized):
        return False
    if _FEDERAL_REGISTER_PATTERN.match(normalized):
        return False
    if any(keyword in lowered for keyword in rules.get("exclude_keywords", [])):
        return False
    if len(normalized) < int(rules.get("min_characters", 24)):
        return False

    tokens = whitespace_tokenize(normalized)
    alpha_tokens = [token for token in tokens if _ALPHA_TOKEN_PATTERN.search(token)]
    if len(alpha_tokens) < int(rules.get("min_alpha_tokens", 4)):
        return False
    return True


def sentence_to_bert_example(sentence: SentenceRecord) -> BertSentenceExample:
    normalized_text = normalize_corpus_text(sentence.text)
    tokens = whitespace_tokenize(normalized_text)
    return BertSentenceExample(
        sentence_id=sentence.sentence_id,
        source_standard=sentence.source_standard,
        section_ref=sentence.section_ref,
        source_url=sentence.source_url,
        text=sentence.text,
        normalized_text=normalized_text,
        tokens=tokens,
        token_count=len(tokens),
        language=sentence.language,
        domain_tags=sentence.domain_tags,
    )


def build_bert_corpus(
    sentences: list[SentenceRecord],
    rules: dict | None = None,
) -> list[BertSentenceExample]:
    active_rules = rules or load_bert_corpus_rules()
    examples: list[BertSentenceExample] = []
    seen_texts: set[str] = set()
    for sentence in sentences:
        if not is_bert_ready_sentence(sentence, active_rules):
            continue
        example = sentence_to_bert_example(sentence)
        if example.normalized_text in seen_texts:
            continue
        seen_texts.add(example.normalized_text)
        examples.append(example)
    return examples


def export_bert_corpus(
    examples: list[BertSentenceExample],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    processed_root = Path(output_root) / "text_corpus" / "processed" / "bert_input"
    processed_root.mkdir(parents=True, exist_ok=True)

    output_paths: dict[str, Path] = {}
    all_path = processed_root / "all.jsonl"
    output_paths["all"] = all_path
    _write_jsonl(all_path, [asdict(example) for example in examples])

    train, val, test = _split_examples(examples)
    for split_name, split_examples in (("train", train), ("val", val), ("test", test)):
        path = processed_root / f"{split_name}.jsonl"
        output_paths[split_name] = path
        _write_jsonl(path, [asdict(example) for example in split_examples])
    return output_paths


def _split_examples(
    examples: list[BertSentenceExample],
) -> tuple[list[BertSentenceExample], list[BertSentenceExample], list[BertSentenceExample]]:
    if not examples:
        return [], [], []
    total = len(examples)
    train_cutoff = max(1, round(total * 0.7))
    val_cutoff = max(train_cutoff + 1, round(total * 0.85)) if total > 1 else train_cutoff
    train = examples[:train_cutoff]
    val = examples[train_cutoff:val_cutoff]
    test = examples[val_cutoff:]
    if not val and test:
        val, test = test[:1], test[1:]
    return train, val, test


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
