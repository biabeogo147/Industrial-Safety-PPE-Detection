"""Prediction-side helpers for text relation extraction."""

from __future__ import annotations

from site_safety_monitor.core.triples import TextTriple


def decoded_triples_to_text_triples(tokens: list[str], triples: list[dict]) -> list[TextTriple]:
    output: list[TextTriple] = []
    for triple in triples:
        subject_span = triple["subject_span"]
        object_span = triple["object_span"]
        output.append(
            TextTriple(
                subject=" ".join(tokens[subject_span[0] : subject_span[1] + 1]),
                predicate=triple["predicate"],
                object=" ".join(tokens[object_span[0] : object_span[1] + 1]),
                subject_span=subject_span,
                object_span=object_span,
            )
        )
    return output
