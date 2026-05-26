"""Metrics for exact triple extraction."""

from __future__ import annotations


def triple_f1(
    predicted: set[tuple[str, str, str]],
    gold: set[tuple[str, str, str]],
) -> tuple[float, float, float]:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
