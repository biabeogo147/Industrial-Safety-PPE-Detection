"""Optional transformer-backed model construction for text IE."""

from __future__ import annotations

from dataclasses import dataclass


class MissingMLDependencyError(RuntimeError):
    """Raised when an optional ML dependency is not installed."""


@dataclass(frozen=True)
class TransformerTaggerConfig:
    encoder_name: str = "bert-base-chinese"
    num_labels: int = 0
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    approximate_parameter_count: int = 110_000_000


def build_bert_bieo_tagger(config: TransformerTaggerConfig):
    """Build a torch/transformers model lazily when dependencies exist."""

    try:
        from torch import nn
        from transformers import AutoModel
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise MissingMLDependencyError(
            "Install the 'ml' optional dependencies to build the transformer tagger."
        ) from exc

    class BertBIEOTagger(nn.Module):
        def __init__(self, encoder_name: str, num_labels: int) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_name)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

        def forward(self, input_ids, attention_mask):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            return self.classifier(outputs.last_hidden_state)

    return BertBIEOTagger(
        encoder_name=config.encoder_name,
        num_labels=config.num_labels,
    )
