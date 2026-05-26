"""Dataset helpers for manufacturing text IE."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from site_safety_monitor.data.text_ie_annotations import (
    GoldSentenceAnnotation,
    load_gold_annotations_jsonl,
)
from site_safety_monitor.text_ie.alignment import expand_word_labels_to_subwords
from site_safety_monitor.text_ie.codec import BIEOCodec


@dataclass(frozen=True)
class RegulationSentence:
    sentence_id: str
    text: str
    tokens: tuple[str, ...]
    triples: tuple[dict, ...]


@dataclass(frozen=True)
class GoldTextIERecord:
    sentence_id: str
    text: str
    tokens: tuple[str, ...]
    triples: tuple[dict, ...]
    source_standard: str | None = None
    section_ref: str | None = None
    source_url: str | None = None
    domain_tags: tuple[str, ...] = ()


class MissingTokenizerDependencyError(RuntimeError):
    """Raised when the tokenizer dependency is not installed."""


def load_regulation_dataset(path: str | Path) -> list[RegulationSentence]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    return [
        RegulationSentence(
            sentence_id=record["sentence_id"],
            text=record["text"],
            tokens=tuple(record["tokens"]),
            triples=tuple(record.get("triples", [])),
        )
        for record in records
    ]


def build_gold_text_ie_record(
    sentence_id: str,
    text: str,
    tokens: list[str] | tuple[str, ...],
    triples: list[dict] | tuple[dict, ...],
    source_standard: str | None = None,
    section_ref: str | None = None,
    source_url: str | None = None,
    domain_tags: list[str] | tuple[str, ...] = (),
) -> GoldTextIERecord:
    record = GoldTextIERecord(
        sentence_id=sentence_id,
        text=text,
        tokens=tuple(tokens),
        triples=tuple(triples),
        source_standard=source_standard,
        section_ref=section_ref,
        source_url=source_url,
        domain_tags=tuple(domain_tags),
    )
    _validate_gold_record(record)
    return record


def load_gold_text_ie_dataset(path: str | Path) -> list[GoldTextIERecord]:
    annotations = load_gold_annotations_jsonl(path)
    return [gold_annotation_to_record(annotation) for annotation in annotations]


def gold_annotation_to_record(annotation: GoldSentenceAnnotation) -> GoldTextIERecord:
    return build_gold_text_ie_record(
        sentence_id=annotation.sentence_id,
        text=annotation.text,
        tokens=annotation.tokens,
        triples=[
            {
                "subject_span": triple.subject_span,
                "predicate": triple.predicate,
                "object_span": triple.object_span,
            }
            for triple in annotation.triples
        ],
        source_standard=annotation.source_standard,
        section_ref=annotation.section_ref,
        source_url=annotation.source_url,
        domain_tags=annotation.domain_tags,
    )


def build_label_vocabulary(predicates: list[str] | tuple[str, ...]) -> tuple[tuple[str, ...], dict[str, int]]:
    tag_space = BIEOCodec(predicates).tag_space()
    return tag_space, {tag: index for index, tag in enumerate(tag_space)}


def create_tokenizer(encoder_name: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise MissingTokenizerDependencyError(
            "Install the 'ml' optional dependencies to build the text IE tokenizer."
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(encoder_name)
    except OSError as exc:  # pragma: no cover - depends on local/runtime model assets
        raise MissingTokenizerDependencyError(
            "Unable to load the tokenizer. Use a local Hugging Face checkpoint directory "
            "or pre-download the encoder assets before building the text IE dataset."
        ) from exc


def encode_gold_record_for_training(
    text: str,
    tokens: list[str] | tuple[str, ...],
    triples: list[dict] | tuple[dict, ...],
    predicates: list[str] | tuple[str, ...],
    max_length: int,
    tokenizer=None,
    encoder_name: str | None = None,
) -> dict:
    resolved_tokenizer = tokenizer
    if resolved_tokenizer is None:
        if encoder_name is None:
            raise ValueError("Either tokenizer or encoder_name must be provided.")
        resolved_tokenizer = create_tokenizer(encoder_name)

    token_list = list(tokens)
    codec = BIEOCodec(predicates)
    tag_space, label_to_id = build_label_vocabulary(predicates)
    word_labels = codec.encode(token_list, list(triples))

    encoded = resolved_tokenizer(
        token_list,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )
    word_ids = list(encoded.word_ids())
    subword_labels = expand_word_labels_to_subwords(word_labels, word_ids)
    label_matrix, label_mask = _encode_subword_labels(subword_labels, tag_space, label_to_id)

    return {
        "input_ids": list(encoded["input_ids"]),
        "attention_mask": list(encoded["attention_mask"]),
        "label_matrix": label_matrix,
        "label_mask": label_mask,
        "word_ids": word_ids,
        "tokens": token_list,
        "tag_space": list(tag_space),
        "subword_labels": subword_labels,
    }


def write_encoded_dataset_jsonl(path: str | Path, records: list[dict]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _validate_gold_record(record: GoldTextIERecord) -> None:
    token_count = len(record.tokens)
    for triple in record.triples:
        for key in ("subject_span", "object_span"):
            start, end = triple[key]
            if start < 0 or end < 0 or start > end or end >= token_count:
                raise ValueError(f"Invalid {key} {triple[key]} for {token_count} tokens.")


def _encode_subword_labels(
    subword_labels: list[list[str]],
    tag_space: tuple[str, ...],
    label_to_id: dict[str, int],
) -> tuple[list[list[int]], list[int]]:
    label_matrix: list[list[int]] = []
    label_mask: list[int] = []
    for labels in subword_labels:
        row = [0] * len(tag_space)
        if not labels:
            label_matrix.append(row)
            label_mask.append(0)
            continue
        for label in labels:
            row[label_to_id[label]] = 1
        label_matrix.append(row)
        label_mask.append(1)
    return label_matrix, label_mask
