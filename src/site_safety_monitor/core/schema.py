"""Schema metadata for relation extraction and safety checking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RelationTemplate:
    """Defines one valid subject-predicate-object pattern."""

    subject_type: str
    predicate: str
    object_type: str


@dataclass(frozen=True)
class RelationSchema:
    """Container for valid relation templates."""

    name: str
    templates: tuple[RelationTemplate, ...]
    documented_relation_count: int | None = None

    @property
    def predicates(self) -> tuple[str, ...]:
        return tuple(template.predicate for template in self.templates)


PAPER_TEXT_RELATION_SCHEMA = RelationSchema(
    name="paper2023_text_schema_public_subset",
    templates=(
        RelationTemplate("person", "be_equipped_with", "ppe"),
        RelationTemplate("person", "perform_operations", "operation"),
        RelationTemplate("operation", "occurrence", "injury"),
    ),
    documented_relation_count=8,
)

PAPER_VISUAL_RELATION_SCHEMA = RelationSchema(
    name="paper2023_visual_schema_self_built_dataset",
    templates=(
        RelationTemplate("worker", "wear", "hard_hat"),
        RelationTemplate("worker", "wear", "eye_protection"),
        RelationTemplate("worker", "wear", "hand_protection"),
        RelationTemplate("worker", "wear", "face_protection"),
        RelationTemplate("worker", "hold", "welding_tool"),
    ),
)

DEFAULT_RELATION_SCHEMA = RelationSchema(
    name="site_safety_monitor_v1",
    templates=PAPER_TEXT_RELATION_SCHEMA.templates + PAPER_VISUAL_RELATION_SCHEMA.templates,
    documented_relation_count=PAPER_TEXT_RELATION_SCHEMA.documented_relation_count,
)
