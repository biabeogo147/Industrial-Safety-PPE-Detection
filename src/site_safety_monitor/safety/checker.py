"""Paper-style PPE compliance and hazard inference."""

from __future__ import annotations

from site_safety_monitor.core.triples import ComplianceDecision, TextTriple, VisualTriple
from site_safety_monitor.safety.hazards import (
    operation_hazard_map,
    operations_from_text,
    required_ppe_relation_set,
)


def evaluate_worker(
    worker_id: str,
    text_triples: list[TextTriple],
    visual_triples: list[VisualTriple],
) -> ComplianceDecision:
    worker_visuals = [
        triple
        for triple in visual_triples
        if triple.subject_id == worker_id and triple.normalized_subject_label == "worker"
    ]
    if not worker_visuals:
        return ComplianceDecision(worker_id=worker_id, compliance="N/A")

    textual_ppe_triples = required_ppe_relation_set(text_triples)
    if not textual_ppe_triples:
        return ComplianceDecision(worker_id=worker_id, compliance="N/A")

    visual_ppe_triples = {
        ("be_equipped_with", triple.normalized_object_label)
        for triple in worker_visuals
        if triple.normalized_predicate in {"wear", "be_equipped_with"}
    }

    if textual_ppe_triples.issubset(visual_ppe_triples):
        return ComplianceDecision(
            worker_id=worker_id,
            compliance="Yes",
            missing_requirements=[],
            hazards=[],
        )

    missing_requirements = sorted(
        requirement_object
        for _, requirement_object in textual_ppe_triples - visual_ppe_triples
    )

    hazard_lookup = operation_hazard_map(text_triples)
    hazards: list[str] = []
    for operation in operations_from_text(text_triples):
        hazards.extend(hazard_lookup.get(operation, []))

    return ComplianceDecision(
        worker_id=worker_id,
        compliance="No",
        missing_requirements=missing_requirements,
        hazards=sorted(dict.fromkeys(hazards)),
    )
