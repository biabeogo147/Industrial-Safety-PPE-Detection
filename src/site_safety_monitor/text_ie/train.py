"""Training helpers for the manufacturing text IE module."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

from site_safety_monitor.data.text_ie_annotations import load_gold_annotations_jsonl
from site_safety_monitor.text_ie.dataset import (
    build_label_vocabulary,
    create_tokenizer,
    encode_gold_record_for_training,
    gold_annotation_to_record,
    load_gold_text_ie_dataset,
)
from site_safety_monitor.text_ie.evaluate import (
    decode_word_level_triples,
    evaluate_prediction_records,
    logits_to_predicted_tags,
)
from site_safety_monitor.text_ie.model import MissingMLDependencyError, TransformerTaggerConfig, build_bert_bieo_tagger
from site_safety_monitor.text_ie.schema import TEXT_IE_PREDICATES


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "site_safety_monitor"
    / "text_ie"
    / "manufacturing.yaml"
)


@dataclass(frozen=True)
class TextIETrainingConfig:
    encoder_name: str = "bert-base-cased"
    predicates: tuple[str, ...] = tuple(TEXT_IE_PREDICATES)
    train_annotations: str = ""
    val_annotations: str = ""
    output_dir: str = ""
    max_length: int = 256
    batch_size: int = 8
    num_epochs: int = 5
    learning_rate: float = 1e-5
    validation_metric: str = "f1"
    early_stopping_patience: int = 10
    threshold: float = 0.5
    dropout: float = 0.1


@dataclass(frozen=True)
class TextIESummary:
    num_sentences: int
    num_triples: int


@dataclass(frozen=True)
class TrainingResult:
    best_f1: float
    best_checkpoint_path: str
    epochs_completed: int


def build_training_config(**overrides) -> TextIETrainingConfig:
    config = TextIETrainingConfig()
    return replace(config, **overrides)


def load_training_config(path: str | Path = DEFAULT_CONFIG_PATH) -> TextIETrainingConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return build_training_config(
        encoder_name=payload.get("encoder_name", "bert-base-cased"),
        predicates=tuple(payload.get("predicates", TEXT_IE_PREDICATES)),
        train_annotations=payload.get("train_annotations", ""),
        val_annotations=payload.get("val_annotations", ""),
        output_dir=payload.get("output_dir", ""),
        max_length=int(payload.get("max_length", 256)),
        batch_size=int(payload.get("batch_size", 8)),
        num_epochs=int(payload.get("num_epochs", 5)),
        learning_rate=float(payload.get("learning_rate", 1e-5)),
        validation_metric=payload.get("validation_metric", "f1"),
        early_stopping_patience=int(payload.get("early_stopping_patience", 10)),
        threshold=float(payload.get("threshold", 0.5)),
        dropout=float(payload.get("dropout", 0.1)),
    )


def summarize_training_data(dataset_path: str | Path) -> TextIESummary:
    records = load_gold_text_ie_dataset(dataset_path)
    return TextIESummary(
        num_sentences=len(records),
        num_triples=sum(len(record.triples) for record in records),
    )


def training_summary_as_dict(dataset_path: str | Path) -> dict:
    return asdict(summarize_training_data(dataset_path))


def train_text_ie_model(config: TextIETrainingConfig) -> TrainingResult:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise MissingMLDependencyError(
            "Install the 'ml' optional dependencies to train the text IE model."
        ) from exc

    tokenizer = create_tokenizer(config.encoder_name)
    train_records = _encode_annotations(
        load_gold_annotations_jsonl(config.train_annotations),
        predicates=config.predicates,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    val_records = _encode_annotations(
        load_gold_annotations_jsonl(config.val_annotations),
        predicates=config.predicates,
        tokenizer=tokenizer,
        max_length=config.max_length,
    )
    if not train_records:
        raise ValueError("No training annotations were loaded for text IE training.")
    if not val_records:
        raise ValueError("No validation annotations were loaded for text IE training.")

    _, label_to_id = build_label_vocabulary(config.predicates)
    model = build_bert_bieo_tagger(
        TransformerTaggerConfig(
            encoder_name=config.encoder_name,
            num_labels=len(label_to_id),
            dropout=config.dropout,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    train_loader = DataLoader(train_records, batch_size=config.batch_size, shuffle=True, collate_fn=_collate_batch)
    val_loader = DataLoader(val_records, batch_size=config.batch_size, shuffle=False, collate_fn=_collate_batch)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_checkpoint_path = output_dir / "best_model.pt"
    patience = 0
    epochs_completed = 0

    for _epoch_index in range(config.num_epochs):
        epochs_completed += 1
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                label_mask=batch["label_mask"],
            )
            outputs["loss"].backward()
            optimizer.step()

        metrics = _evaluate_model(
            model=model,
            data_loader=val_loader,
            predicates=config.predicates,
            threshold=config.threshold,
        )
        current_f1 = metrics["f1"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "config": asdict(config),
                },
                best_checkpoint_path,
            )
            (output_dir / "best_metrics.json").write_text(
                json.dumps(metrics, indent=2),
                encoding="utf-8",
            )
        else:
            patience += 1
            if patience >= config.early_stopping_patience:
                break

    return TrainingResult(
        best_f1=best_f1,
        best_checkpoint_path=str(best_checkpoint_path),
        epochs_completed=epochs_completed,
    )


def _encode_annotations(annotations, *, predicates, tokenizer, max_length: int) -> list[dict]:
    encoded_records: list[dict] = []
    for annotation in annotations:
        record = gold_annotation_to_record(annotation)
        encoded = encode_gold_record_for_training(
            text=record.text,
            tokens=record.tokens,
            triples=record.triples,
            predicates=predicates,
            max_length=max_length,
            tokenizer=tokenizer,
        )
        encoded["gold_triples"] = list(record.triples)
        encoded["tokens"] = list(record.tokens)
        encoded_records.append(encoded)
    return encoded_records


def _collate_batch(records: list[dict]):
    import torch

    return {
        "input_ids": torch.tensor([record["input_ids"] for record in records], dtype=torch.long),
        "attention_mask": torch.tensor([record["attention_mask"] for record in records], dtype=torch.long),
        "labels": torch.tensor([record["label_matrix"] for record in records], dtype=torch.float),
        "label_mask": torch.tensor([record["label_mask"] for record in records], dtype=torch.long),
        "word_ids": [record["word_ids"] for record in records],
        "tokens": [record["tokens"] for record in records],
        "gold_triples": [record["gold_triples"] for record in records],
        "tag_space": records[0]["tag_space"] if records else [],
    }


def _evaluate_model(model, data_loader, *, predicates, threshold: float) -> dict[str, float]:
    records: list[dict] = []
    model.eval()
    import torch

    with torch.no_grad():
        for batch in data_loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            for index, tokens in enumerate(batch["tokens"]):
                predicted_subword_tags = logits_to_predicted_tags(
                    logits=outputs["logits"][index],
                    tag_space=batch["tag_space"],
                    label_mask=batch["label_mask"][index].tolist(),
                    threshold=threshold,
                )
                predicted_triples = decode_word_level_triples(
                    tokens=tokens,
                    predicted_subword_tags=predicted_subword_tags,
                    word_ids=batch["word_ids"][index],
                    predicates=predicates,
                )
                records.append(
                    {
                        "tokens": tokens,
                        "predicted_triples": predicted_triples,
                        "gold_triples": batch["gold_triples"][index],
                    }
                )
    return evaluate_prediction_records(records)
