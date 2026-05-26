"""End-to-end runner for the initial Site Safety Monitor flow."""

from __future__ import annotations

import json
from pathlib import Path

from site_safety_monitor.core.triples import TextTriple, VisualTriple
from site_safety_monitor.safety.checker import evaluate_worker


def _load_text_triples(path: str | Path) -> list[TextTriple]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TextTriple(**triple) for triple in payload["triples"]]


def _load_visual_triples(path: str | Path) -> list[VisualTriple]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [VisualTriple(**triple) for triple in payload["triples"]]


def run_case(regulation_path: str | Path, scene_path: str | Path, worker_id: str | None = None) -> dict:
    text_triples = _load_text_triples(regulation_path)
    visual_triples = _load_visual_triples(scene_path)
    resolved_worker_id = worker_id or _infer_worker_id(visual_triples)
    decision = evaluate_worker(
        worker_id=resolved_worker_id,
        text_triples=text_triples,
        visual_triples=visual_triples,
    )
    return {
        "worker_id": decision.worker_id,
        "compliance": decision.compliance,
        "missing_requirements": decision.missing_requirements,
        "hazards": decision.hazards,
    }


def _infer_worker_id(visual_triples: list[VisualTriple]) -> str:
    for triple in visual_triples:
        if triple.normalized_subject_label == "worker":
            return triple.subject_id
    return "worker_0"
