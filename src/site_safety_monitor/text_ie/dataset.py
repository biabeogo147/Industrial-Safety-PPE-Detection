"""Dataset loading helpers for regulation relation extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RegulationSentence:
    sentence_id: str
    text: str
    tokens: tuple[str, ...]
    triples: tuple[dict, ...]


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
