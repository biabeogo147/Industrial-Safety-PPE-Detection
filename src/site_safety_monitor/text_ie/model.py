"""Optional transformer-backed model construction for text IE."""

from __future__ import annotations

from dataclasses import dataclass


class MissingMLDependencyError(RuntimeError):
    """Raised when an optional ML dependency is not installed."""


@dataclass(frozen=True)
class TransformerTaggerConfig:
    encoder_name: str = "bert-base-cased"
    num_labels: int = 0
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    approximate_parameter_count: int = 110_000_000
    dropout: float = 0.1


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
        def __init__(self, encoder_name: str, num_labels: int, dropout: float) -> None:
            super().__init__()
            try:
                self.encoder = AutoModel.from_pretrained(encoder_name)
            except OSError as exc:
                raise MissingMLDependencyError(
                    "Unable to load the transformer encoder. Use a local Hugging Face "
                    "checkpoint directory or pre-download the encoder assets before training."
                ) from exc
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        def forward(self, input_ids, attention_mask, labels=None, label_mask=None):
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            sequence_output = self.dropout(outputs.last_hidden_state)
            logits = self.classifier(sequence_output)
            result = {"logits": logits}
            if labels is not None:
                loss = self.loss_fn(logits, labels.float())
                if label_mask is not None:
                    mask = label_mask.unsqueeze(-1).float()
                    denominator = max(float(mask.sum().item() * logits.shape[-1]), 1.0)
                    loss = (loss * mask).sum() / denominator
                else:
                    loss = loss.mean()
                result["loss"] = loss
            return result

    return BertBIEOTagger(
        encoder_name=config.encoder_name,
        num_labels=config.num_labels,
        dropout=config.dropout,
    )
