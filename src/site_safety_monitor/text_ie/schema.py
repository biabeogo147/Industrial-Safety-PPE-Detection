"""Default schema definitions for regulation information extraction."""

from __future__ import annotations

from site_safety_monitor.core.schema import PAPER_TEXT_RELATION_SCHEMA

TEXT_IE_PREDICATES = tuple(
    predicate for predicate in PAPER_TEXT_RELATION_SCHEMA.predicates
)
