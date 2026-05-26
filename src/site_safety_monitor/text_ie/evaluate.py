"""Evaluation helpers for manufacturing text IE."""

from __future__ import annotations

import math

from site_safety_monitor.text_ie.alignment import collapse_subword_labels_to_words
from site_safety_monitor.text_ie.codec import BIEOCodec
from site_safety_monitor.text_ie.metrics import triple_f1


def logits_to_predicted_tags(
    logits,
    tag_space: list[str] | tuple[str, ...],
    label_mask: list[int],
    threshold: float,
) -> list[list[str]]:
    rows = _to_nested_list(logits)
    predicted: list[list[str]] = []
    for row, mask_value in zip(rows, label_mask, strict=False):
        if not mask_value:
            predicted.append([])
            continue
        scores = [_sigmoid(value) for value in row]
        labels = [tag for tag, score in zip(tag_space, scores, strict=False) if score >= threshold]
        non_outside = [label for label in labels if label != "O"]
        if non_outside:
            labels = non_outside
        if not labels:
            labels = ["O"]
        predicted.append(sorted(labels))
    return predicted


def decode_word_level_triples(
    tokens: list[str] | tuple[str, ...],
    predicted_subword_tags: list[list[str]],
    word_ids: list[int | None],
    predicates: list[str] | tuple[str, ...],
) -> list[dict]:
    word_tags = collapse_subword_labels_to_words(
        subword_labels=predicted_subword_tags,
        word_ids=word_ids,
        num_words=len(tokens),
    )
    return BIEOCodec(predicates).decode(list(tokens), word_tags)


def triples_to_surface_set(
    tokens: list[str] | tuple[str, ...],
    triples: list[dict] | tuple[dict, ...],
) -> set[tuple[str, str, str]]:
    output: set[tuple[str, str, str]] = set()
    for triple in triples:
        subject_start, subject_end = triple["subject_span"]
        object_start, object_end = triple["object_span"]
        output.add(
            (
                " ".join(tokens[subject_start : subject_end + 1]),
                triple["predicate"],
                " ".join(tokens[object_start : object_end + 1]),
            )
        )
    return output


def is_overlap_sentence(triples: list[dict] | tuple[dict, ...]) -> bool:
    subjects = [tuple(triple["subject_span"]) for triple in triples]
    objects = [tuple(triple["object_span"]) for triple in triples]
    return len(subjects) != len(set(subjects)) or len(objects) != len(set(objects))


def evaluate_prediction_records(records: list[dict]) -> dict[str, float]:
    predicted_all: set[tuple[str, str, str]] = set()
    gold_all: set[tuple[str, str, str]] = set()
    overlap_predicted: set[tuple[str, str, str]] = set()
    overlap_gold: set[tuple[str, str, str]] = set()

    sentence_exact_matches = 0
    for record in records:
        predicted = triples_to_surface_set(record["tokens"], record["predicted_triples"])
        gold = triples_to_surface_set(record["tokens"], record["gold_triples"])
        predicted_all |= predicted
        gold_all |= gold
        if predicted == gold:
            sentence_exact_matches += 1
        if is_overlap_sentence(record["gold_triples"]):
            overlap_predicted |= predicted
            overlap_gold |= gold

    precision, recall, f1 = triple_f1(predicted_all, gold_all)
    overlap_precision, overlap_recall, overlap_f1 = triple_f1(overlap_predicted, overlap_gold)
    total = len(records) if records else 1
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "sentence_exact_match": sentence_exact_matches / total,
        "overlap_precision": overlap_precision,
        "overlap_recall": overlap_recall,
        "overlap_f1": overlap_f1,
    }


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-value)
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _to_nested_list(logits) -> list[list[float]]:
    if hasattr(logits, "detach"):
        return logits.detach().cpu().tolist()
    return [[float(value) for value in row] for row in logits]
