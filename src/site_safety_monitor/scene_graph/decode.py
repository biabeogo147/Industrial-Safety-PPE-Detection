"""Decode relation annotations into visual triples."""

from __future__ import annotations

from site_safety_monitor.core.triples import VisualTriple


def decode_relations_to_visual_triples(
    objects_by_id: dict[int, dict],
    relations: list[dict],
) -> list[VisualTriple]:
    triples: list[VisualTriple] = []
    for relation in relations:
        subject = objects_by_id[relation["subject_id"]]
        object_ = objects_by_id[relation["object_id"]]
        triples.append(
            VisualTriple(
                subject_id=subject.get("instance_id", f"{subject['label']}_{subject['id']}"),
                subject_label=subject["label"],
                predicate=relation["predicate"],
                object_id=object_.get("instance_id", f"{object_['label']}_{object_['id']}"),
                object_label=object_["label"],
                confidence=relation.get("confidence"),
            )
        )
    return triples
