"""Lightweight predicate fusion head."""

from __future__ import annotations


class PredicateFusionHead:
    """Combine lightweight visual logits with semantic logits."""

    def __init__(self, input_dim: int, num_predicates: int) -> None:
        self.input_dim = input_dim
        self.num_predicates = num_predicates

    def forward(
        self,
        subject_features: list[list[float]],
        relation_features: list[list[float]],
        object_features: list[list[float]],
        semantic_logits: list[list[float]],
    ) -> list[list[float]]:
        fused: list[list[float]] = []
        for subject_row, relation_row, object_row, semantic_row in zip(
            subject_features,
            relation_features,
            object_features,
            semantic_logits,
            strict=False,
        ):
            merged = subject_row + relation_row + object_row
            total = sum(merged) if merged else 0.0
            visual_row = [
                total / float(self.input_dim + predicate_index + 1)
                for predicate_index in range(self.num_predicates)
            ]
            fused.append(
                [
                    visual_logit + semantic_logit
                    for visual_logit, semantic_logit in zip(
                        visual_row,
                        semantic_row,
                        strict=False,
                    )
                ]
            )
        return fused

    __call__ = forward
