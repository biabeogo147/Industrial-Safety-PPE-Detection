"""Empirical predicate prior conditioned on subject and object classes."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import log

from site_safety_monitor.core.normalize import normalize_label, normalize_predicate


NEG_INF = -1e9


class SemanticPrior:
    """Estimate predicate priors from relation counts."""

    def __init__(self, smoothing: float = 0.0) -> None:
        self.smoothing = smoothing
        self.counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.predicate_vocabulary: set[str] = set()

    def fit(self, relations: list[tuple[str, str, str]]) -> None:
        for subject_label, predicate, object_label in relations:
            normalized_subject = normalize_label(subject_label)
            normalized_object = normalize_label(object_label)
            normalized_predicate = normalize_predicate(predicate)
            self.counts[(normalized_subject, normalized_object)][normalized_predicate] += 1
            self.predicate_vocabulary.add(normalized_predicate)

    def probabilities_for(self, subject_label: str, object_label: str) -> dict[str, float]:
        normalized_subject = normalize_label(subject_label)
        normalized_object = normalize_label(object_label)
        counter = self.counts[(normalized_subject, normalized_object)]
        vocabulary = sorted(self.predicate_vocabulary)
        if not vocabulary:
            return {}
        denominator = sum(counter.values()) + self.smoothing * len(vocabulary)
        if denominator == 0:
            return {predicate: 0.0 for predicate in vocabulary}
        return {
            predicate: (counter[predicate] + self.smoothing) / denominator
            for predicate in vocabulary
        }

    def logits_for(self, subject_label: str, object_label: str) -> dict[str, float]:
        probabilities = self.probabilities_for(subject_label, object_label)
        return {
            predicate: log(probability) if probability > 0 else NEG_INF
            for predicate, probability in probabilities.items()
        }
