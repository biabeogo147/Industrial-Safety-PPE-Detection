"""OSHA manufacturing corpus crawling and parsing helpers."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

import yaml

from site_safety_monitor.data.corpus_models import OshaSourcePage, SectionRecord, SentenceRecord


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_PATH = PACKAGE_ROOT / "configs" / "site_safety_monitor" / "corpus" / "osha_1910_sources.yaml"
DEFAULT_OUTPUT_ROOT = Path(r"E:\data\SH17\site_safety_monitor")

_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_TEXT_TAGS = {"p", "li"}
_HEADING_TAGS = {"h1", "h2", "h3", "h4"}


class _OshaSectionParser(HTMLParser):
    """Small HTML parser that groups paragraph text under headings."""

    def __init__(self) -> None:
        super().__init__()
        self.current_heading = "Overview"
        self.current_tag: str | None = None
        self.current_buffer: list[str] = []
        self.sections: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _TEXT_TAGS | _HEADING_TAGS:
            self.current_tag = tag
            self.current_buffer = []

    def handle_data(self, data: str) -> None:
        if self.current_tag:
            self.current_buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != self.current_tag:
            return
        text = _normalize_whitespace("".join(self.current_buffer))
        if not text:
            self.current_tag = None
            self.current_buffer = []
            return
        if tag in _HEADING_TAGS:
            self.current_heading = text
        elif tag in _TEXT_TAGS:
            self.sections.append((self.current_heading, text))
        self.current_tag = None
        self.current_buffer = []


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def split_into_sentences(text: str) -> list[str]:
    """Split regulatory prose while keeping direct OSHA wording intact."""

    normalized = _normalize_whitespace(text)
    if not normalized:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_PATTERN.split(normalized) if part.strip()]


def load_source_manifest(path: str | Path = DEFAULT_MANIFEST_PATH) -> list[OshaSourcePage]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [
        OshaSourcePage(
            standard_number=record["standard_number"],
            title=record["title"],
            url=record["url"],
            domain_tags=tuple(record.get("domain_tags", [])),
            crawl_priority=int(record.get("crawl_priority", 100)),
        )
        for record in payload.get("sources", [])
    ]


def extract_sections_from_html(source: OshaSourcePage, html: str) -> list[SectionRecord]:
    parser = _OshaSectionParser()
    parser.feed(html)
    sections = [
        SectionRecord(
            source_standard=source.standard_number,
            section_ref=f"{source.standard_number}:{index:03d}",
            source_url=source.url,
            title=heading,
            text=text,
            domain_tags=source.domain_tags,
        )
        for index, (heading, text) in enumerate(parser.sections, start=1)
    ]
    if sections:
        return sections
    fallback_text = _normalize_whitespace(re.sub(r"<[^>]+>", " ", html))
    if not fallback_text:
        return []
    return [
        SectionRecord(
            source_standard=source.standard_number,
            section_ref=f"{source.standard_number}:000",
            source_url=source.url,
            title=source.title,
            text=fallback_text,
            domain_tags=source.domain_tags,
        )
    ]


def section_to_sentence_records(section: SectionRecord) -> list[SentenceRecord]:
    return [
        SentenceRecord(
            sentence_id=f"{section.section_ref}:{index:03d}",
            source_standard=section.source_standard,
            section_ref=section.section_ref,
            source_url=section.source_url,
            text=sentence,
            domain_tags=section.domain_tags,
        )
        for index, sentence in enumerate(split_into_sentences(section.text), start=1)
    ]


def fetch_source_html(source: OshaSourcePage, output_root: str | Path, refresh: bool = False) -> Path:
    raw_dir = Path(output_root) / "text_corpus" / "raw" / "osha_1910"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_path = raw_dir / f"{source.standard_number}.html"
    if output_path.exists() and not refresh:
        return output_path
    with urlopen(source.url) as response:
        payload = response.read().decode("utf-8", errors="replace")
    output_path.write_text(payload, encoding="utf-8")
    return output_path


def crawl_sources(
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    limit: int | None = None,
    refresh: bool = False,
) -> tuple[list[SectionRecord], list[SentenceRecord]]:
    sources = sorted(load_source_manifest(manifest_path), key=lambda item: item.crawl_priority)
    if limit is not None:
        sources = sources[:limit]

    sections: list[SectionRecord] = []
    sentences: list[SentenceRecord] = []

    for source in sources:
        html_path = fetch_source_html(source, output_root=output_root, refresh=refresh)
        html = html_path.read_text(encoding="utf-8")
        extracted_sections = extract_sections_from_html(source, html)
        sections.extend(extracted_sections)
        for section in extracted_sections:
            sentences.extend(section_to_sentence_records(section))

    interim_root = Path(output_root) / "text_corpus" / "interim"
    interim_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(interim_root / "sections.jsonl", [section.__dict__ for section in sections])
    _write_jsonl(interim_root / "sentences.jsonl", [sentence.__dict__ for sentence in sentences])
    return sections, sentences


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
