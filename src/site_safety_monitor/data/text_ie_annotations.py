"""Contracts and IO helpers for gold manufacturing text IE annotations."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


_VALID_PREDICATES = {
    "be_equipped_with",
    "perform_operations",
    "occurrence",
}


@dataclass(frozen=True)
class GoldTripleAnnotation:
    """One gold triple with token spans for one sentence."""

    subject_span: tuple[int, int]
    predicate: str
    object_span: tuple[int, int]
    subject_text: str | None = None
    object_text: str | None = None

    def validate(self, tokens: tuple[str, ...]) -> None:
        if self.predicate not in _VALID_PREDICATES:
            raise ValueError(f"Unsupported predicate: {self.predicate}")
        _validate_span(self.subject_span, tokens=tokens, role="subject")
        _validate_span(self.object_span, tokens=tokens, role="object")


@dataclass(frozen=True)
class GoldSentenceAnnotation:
    """One annotated sentence used for BERT text IE training."""

    sentence_id: str
    text: str
    tokens: tuple[str, ...]
    triples: tuple[GoldTripleAnnotation, ...]
    source_standard: str | None = None
    section_ref: str | None = None
    source_url: str | None = None
    domain_tags: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        for triple in self.triples:
            triple.validate(self.tokens)


def load_gold_annotations_jsonl(path: str | Path) -> list[GoldSentenceAnnotation]:
    annotations: list[GoldSentenceAnnotation] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            annotations.append(gold_sentence_from_dict(payload))
    return annotations


def write_gold_annotations_jsonl(path: str | Path, annotations: list[GoldSentenceAnnotation]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            annotation.validate()
            handle.write(json.dumps(_annotation_as_dict(annotation), ensure_ascii=False))
            handle.write("\n")


def gold_sentence_from_dict(payload: dict) -> GoldSentenceAnnotation:
    triples = tuple(
        GoldTripleAnnotation(
            subject_span=tuple(triple["subject_span"]),
            predicate=triple["predicate"],
            object_span=tuple(triple["object_span"]),
            subject_text=triple.get("subject_text"),
            object_text=triple.get("object_text"),
        )
        for triple in payload.get("triples", [])
    )
    annotation = GoldSentenceAnnotation(
        sentence_id=payload["sentence_id"],
        text=payload["text"],
        tokens=tuple(payload["tokens"]),
        triples=triples,
        source_standard=payload.get("source_standard"),
        section_ref=payload.get("section_ref"),
        source_url=payload.get("source_url"),
        domain_tags=tuple(payload.get("domain_tags", [])),
    )
    annotation.validate()
    return annotation


def _annotation_as_dict(annotation: GoldSentenceAnnotation) -> dict:
    payload = asdict(annotation)
    payload["tokens"] = list(annotation.tokens)
    payload["domain_tags"] = list(annotation.domain_tags)
    payload["triples"] = [
        {
            "subject_span": list(triple.subject_span),
            "predicate": triple.predicate,
            "object_span": list(triple.object_span),
            "subject_text": triple.subject_text,
            "object_text": triple.object_text,
        }
        for triple in annotation.triples
    ]
    return payload


def _validate_span(span: tuple[int, int], tokens: tuple[str, ...], role: str) -> None:
    if len(span) != 2:
        raise ValueError(f"{role} span must contain exactly two indices.")
    start, end = span
    if start < 0 or end < 0:
        raise ValueError(f"{role} span indices must be non-negative.")
    if start > end:
        raise ValueError(f"{role} span start must be <= end.")
    if end >= len(tokens):
        raise ValueError(f"{role} span index {end} is out of range for {len(tokens)} tokens.")
