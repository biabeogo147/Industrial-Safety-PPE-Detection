"""Training helpers for the text IE module."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from site_safety_monitor.text_ie.dataset import load_regulation_dataset


@dataclass(frozen=True)
class TextIETrainingConfig:
    encoder_name: str = "bert-base-chinese"
    learning_rate: float = 1e-5
    validation_metric: str = "f1"
    early_stopping_patience: int = 10


@dataclass(frozen=True)
class TextIESummary:
    num_sentences: int
    num_triples: int


def summarize_training_data(dataset_path: str | Path) -> TextIESummary:
    records = load_regulation_dataset(dataset_path)
    return TextIESummary(
        num_sentences=len(records),
        num_triples=sum(len(record.triples) for record in records),
    )


def training_summary_as_dict(dataset_path: str | Path) -> dict:
    return asdict(summarize_training_data(dataset_path))
