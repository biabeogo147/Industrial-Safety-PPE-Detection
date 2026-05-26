"""Core triple and decision models."""

from __future__ import annotations

from dataclasses import dataclass, field

from site_safety_monitor.core.normalize import normalize_label, normalize_predicate


@dataclass(frozen=True)
class TextTriple:
    """Triple extracted from regulation text."""

    subject: str
    predicate: str
    object: str
    subject_span: tuple[int, int] | None = None
    object_span: tuple[int, int] | None = None

    @property
    def normalized_subject(self) -> str:
        return normalize_label(self.subject)

    @property
    def normalized_predicate(self) -> str:
        return normalize_predicate(self.predicate)

    @property
    def normalized_object(self) -> str:
        return normalize_label(self.object)

    def as_normalized_tuple(self) -> tuple[str, str, str]:
        return (
            self.normalized_subject,
            self.normalized_predicate,
            self.normalized_object,
        )


@dataclass(frozen=True)
class VisualTriple:
    """Triple extracted from image parsing or relation detection."""

    subject_id: str
    subject_label: str
    predicate: str
    object_id: str
    object_label: str
    confidence: float | None = None

    @property
    def normalized_subject_label(self) -> str:
        return normalize_label(self.subject_label)

    @property
    def normalized_predicate(self) -> str:
        return normalize_predicate(self.predicate)

    @property
    def normalized_object_label(self) -> str:
        return normalize_label(self.object_label)

    def as_requirement_tuple(self) -> tuple[str, str]:
        return (self.normalized_predicate, self.normalized_object_label)


@dataclass(frozen=True)
class HazardTriple:
    """Operation-to-hazard triple used during hazard inference."""

    operation: str
    predicate: str
    hazard: str

    @property
    def normalized_operation(self) -> str:
        return normalize_label(self.operation)

    @property
    def normalized_predicate(self) -> str:
        return normalize_predicate(self.predicate)

    @property
    def normalized_hazard(self) -> str:
        return normalize_label(self.hazard)


@dataclass(frozen=True)
class ComplianceDecision:
    """Final safety decision for one worker or scene."""

    worker_id: str
    compliance: str
    missing_requirements: list[str] = field(default_factory=list)
    hazards: list[str] = field(default_factory=list)
