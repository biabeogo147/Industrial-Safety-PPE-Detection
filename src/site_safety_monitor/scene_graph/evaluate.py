"""Evaluation helpers for relation parsing."""

from __future__ import annotations


SCENE_GRAPH_EVAL_TASKS = ("SGGen", "SGCls", "PredCls")
SCENE_GRAPH_RECALL_CUTOFFS = (20, 50, 100)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)
