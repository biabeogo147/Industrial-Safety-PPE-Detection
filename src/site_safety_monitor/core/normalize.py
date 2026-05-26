"""Normalization helpers shared across product modules."""

from __future__ import annotations

import re


_SEPARATOR_PATTERN = re.compile(r"[_\-]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")

_LABEL_ALIASES = {
    "burns": "burn",
    "face protections": "face protection",
    "hand protections": "hand protection",
    "hard hats": "hard hat",
    "helmet": "hard hat",
    "helmets": "hard hat",
    "work at height": "working at height",
    "welding operations": "welding operation",
    "welding tools": "welding tool",
}

_PREDICATE_ALIASES = {
    "be equipped with": "be_equipped_with",
    "be equipped with wear": "be_equipped_with",
    "holding": "hold",
    "perform operations": "perform_operations",
    "perform operation": "perform_operations",
    "perform...operations": "perform_operations",
    "wearing": "wear",
}


def normalize_label(value: str) -> str:
    """Normalize free-form labels for human-readable matching."""

    cleaned = _SEPARATOR_PATTERN.sub(" ", value.strip().lower())
    collapsed = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    return _LABEL_ALIASES.get(collapsed, collapsed)


def normalize_predicate(value: str) -> str:
    """Normalize predicate identifiers to snake_case."""

    cleaned = _SEPARATOR_PATTERN.sub(" ", value.strip().lower())
    collapsed = _WHITESPACE_PATTERN.sub(" ", cleaned).strip()
    aliased = _PREDICATE_ALIASES.get(collapsed, collapsed)
    return aliased.replace(" ", "_")
