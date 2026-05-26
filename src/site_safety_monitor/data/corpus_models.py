"""Contracts for manufacturing-domain text corpus preparation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OshaSourcePage:
    """Official OSHA page metadata used by the crawler."""

    standard_number: str
    title: str
    url: str
    domain_tags: tuple[str, ...]
    crawl_priority: int = 100


@dataclass(frozen=True)
class SectionRecord:
    """Cleaned section text extracted from a source page."""

    source_standard: str
    section_ref: str
    source_url: str
    title: str
    text: str
    domain_tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SentenceRecord:
    """Sentence-level corpus row with provenance."""

    sentence_id: str
    source_standard: str
    section_ref: str
    source_url: str
    text: str
    language: str = "en"
    domain_tags: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class BertSentenceExample:
    """Clean sentence record prepared for downstream BERT tokenization."""

    sentence_id: str
    source_standard: str
    section_ref: str
    source_url: str
    text: str
    normalized_text: str
    tokens: tuple[str, ...]
    token_count: int
    language: str = "en"
    domain_tags: tuple[str, ...] = field(default_factory=tuple)
