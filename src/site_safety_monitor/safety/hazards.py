"""Helpers for deriving hazards from regulation triples."""

from __future__ import annotations

from collections import defaultdict

from site_safety_monitor.core.triples import TextTriple


def required_ppe_from_text(text_triples: list[TextTriple]) -> set[str]:
    return {
        triple.normalized_object
        for triple in text_triples
        if triple.normalized_predicate == "be_equipped_with"
    }


def required_ppe_relation_set(text_triples: list[TextTriple]) -> set[tuple[str, str]]:
    return {
        ("be_equipped_with", triple.normalized_object)
        for triple in text_triples
        if triple.normalized_predicate == "be_equipped_with"
    }


def operation_hazard_map(text_triples: list[TextTriple]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for triple in text_triples:
        if triple.normalized_predicate == "occurrence":
            mapping[triple.normalized_subject].append(triple.normalized_object)
    return dict(mapping)


def operations_from_text(text_triples: list[TextTriple]) -> list[str]:
    return [
        triple.normalized_object
        for triple in text_triples
        if triple.normalized_predicate == "perform_operations"
    ]
