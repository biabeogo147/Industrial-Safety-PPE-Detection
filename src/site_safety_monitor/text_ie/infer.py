"""Inference helpers for manufacturing text IE."""

from __future__ import annotations

from pathlib import Path

from site_safety_monitor.data.bert_corpus import normalize_corpus_text, whitespace_tokenize
from site_safety_monitor.text_ie.dataset import create_tokenizer
from site_safety_monitor.text_ie.evaluate import decode_word_level_triples, logits_to_predicted_tags
from site_safety_monitor.text_ie.model import TransformerTaggerConfig, build_bert_bieo_tagger
from site_safety_monitor.text_ie.predict import decoded_triples_to_text_triples


def decode_prediction_bundle(
    tokens: list[str],
    predicted_tags: list[list[str]],
    predicates: list[str] | tuple[str, ...],
) -> list[dict]:
    return decode_word_level_triples(
        tokens=tokens,
        predicted_subword_tags=predicted_tags,
        word_ids=list(range(len(tokens))),
        predicates=predicates,
    )


def infer_text_triples(
    text: str,
    *,
    checkpoint_path: str | Path,
    encoder_name: str,
    predicates: list[str] | tuple[str, ...],
    tag_space: list[str] | tuple[str, ...],
    max_length: int,
    threshold: float,
):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Install the 'ml' optional dependencies to run text IE inference.") from exc

    tokenizer = create_tokenizer(encoder_name)
    tokens = list(whitespace_tokenize(normalize_corpus_text(text)))
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )
    word_ids = list(encoded.word_ids())
    model = build_bert_bieo_tagger(
        TransformerTaggerConfig(
            encoder_name=encoder_name,
            num_labels=len(tag_space),
        )
    )
    checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        outputs = model(
            input_ids=torch.tensor([encoded["input_ids"]], dtype=torch.long),
            attention_mask=torch.tensor([encoded["attention_mask"]], dtype=torch.long),
        )
    predicted_tags = logits_to_predicted_tags(
        logits=outputs["logits"][0],
        tag_space=tag_space,
        label_mask=[1 if word_id is not None else 0 for word_id in word_ids],
        threshold=threshold,
    )
    decoded = decode_word_level_triples(
        tokens=tokens,
        predicted_subword_tags=predicted_tags,
        word_ids=word_ids,
        predicates=predicates,
    )
    return decoded_triples_to_text_triples(tokens=tokens, triples=decoded)
