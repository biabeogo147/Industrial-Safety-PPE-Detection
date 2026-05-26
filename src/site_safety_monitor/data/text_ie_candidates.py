"""Candidate-pool preparation for manual manufacturing text IE annotation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from site_safety_monitor.data.corpus_models import BertSentenceExample


_KEYWORD_WEIGHTS = {
    "shall": 2,
    "must": 2,
    "ppe": 3,
    "protection": 2,
    "helmet": 2,
    "gloves": 2,
    "glasses": 2,
    "respiratory": 2,
    "noise": 2,
    "machine": 2,
    "welding": 2,
    "injury": 2,
    "hazard": 2,
}


@dataclass(frozen=True)
class CandidateSentence:
    sentence_id: str
    source_standard: str
    section_ref: str
    source_url: str
    text: str
    normalized_text: str
    tokens: tuple[str, ...]
    token_count: int
    language: str
    domain_tags: tuple[str, ...]
    candidate_score: int


def load_bert_examples_jsonl(path: str | Path) -> list[BertSentenceExample]:
    examples: list[BertSentenceExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            examples.append(
                BertSentenceExample(
                    sentence_id=payload["sentence_id"],
                    source_standard=payload["source_standard"],
                    section_ref=payload["section_ref"],
                    source_url=payload["source_url"],
                    text=payload["text"],
                    normalized_text=payload["normalized_text"],
                    tokens=tuple(payload["tokens"]),
                    token_count=payload["token_count"],
                    language=payload.get("language", "en"),
                    domain_tags=tuple(payload.get("domain_tags", [])),
                )
            )
    return examples


def load_candidate_sentences_jsonl(path: str | Path) -> list[CandidateSentence]:
    candidates: list[CandidateSentence] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            candidates.append(
                CandidateSentence(
                    sentence_id=payload["sentence_id"],
                    source_standard=payload["source_standard"],
                    section_ref=payload["section_ref"],
                    source_url=payload["source_url"],
                    text=payload["text"],
                    normalized_text=payload["normalized_text"],
                    tokens=tuple(payload["tokens"]),
                    token_count=payload["token_count"],
                    language=payload.get("language", "en"),
                    domain_tags=tuple(payload.get("domain_tags", [])),
                    candidate_score=payload["candidate_score"],
                )
            )
    return candidates


def score_candidate_sentence(sentence: str) -> int:
    lowered = sentence.lower()
    score = 0
    for keyword, weight in _KEYWORD_WEIGHTS.items():
        if keyword in lowered:
            score += weight
    return score


def build_candidate_pool(
    examples: list[BertSentenceExample],
    top_k: int | None = None,
    domain_tag: str | None = None,
) -> list[CandidateSentence]:
    candidates: list[CandidateSentence] = []
    for example in examples:
        if domain_tag and domain_tag not in example.domain_tags:
            continue
        score = score_candidate_sentence(example.normalized_text)
        if score <= 0:
            continue
        candidates.append(
            CandidateSentence(
                sentence_id=example.sentence_id,
                source_standard=example.source_standard,
                section_ref=example.section_ref,
                source_url=example.source_url,
                text=example.text,
                normalized_text=example.normalized_text,
                tokens=example.tokens,
                token_count=example.token_count,
                language=example.language,
                domain_tags=example.domain_tags,
                candidate_score=score,
            )
        )
    candidates.sort(key=lambda item: (-item.candidate_score, item.sentence_id))
    return candidates[:top_k] if top_k is not None else candidates


def write_candidate_pool_jsonl(path: str | Path, candidates: list[CandidateSentence]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            payload = asdict(candidate)
            payload["tokens"] = list(candidate.tokens)
            payload["domain_tags"] = list(candidate.domain_tags)
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def candidate_to_annotation_seed(candidate: CandidateSentence) -> dict:
    return {
        "sentence_id": candidate.sentence_id,
        "text": candidate.text,
        "tokens": list(candidate.tokens),
        "triples": [],
        "source_standard": candidate.source_standard,
        "section_ref": candidate.section_ref,
        "source_url": candidate.source_url,
        "domain_tags": list(candidate.domain_tags),
    }


def write_annotation_seed_jsonl(path: str | Path, candidates: list[CandidateSentence]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            handle.write(json.dumps(candidate_to_annotation_seed(candidate), ensure_ascii=False))
            handle.write("\n")
