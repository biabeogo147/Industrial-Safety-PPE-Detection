"""Pilot selection helpers for manufacturing text IE annotation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from site_safety_monitor.data.text_ie_candidates import CandidateSentence, candidate_to_annotation_seed


_EXCLUDE_PATTERNS = (
    re.compile(r"^note[:\s]", re.IGNORECASE),
    re.compile(r"incorporated by reference", re.IGNORECASE),
    re.compile(r"american national standard", re.IGNORECASE),
    re.compile(r"astm ", re.IGNORECASE),
    re.compile(r"effective on ", re.IGNORECASE),
    re.compile(r"paragraphs? \([a-z0-9]+\) through \([a-z0-9]+\)", re.IGNORECASE),
    re.compile(r"\bmeans\b", re.IGNORECASE),
    re.compile(r"\bplhcp\b", re.IGNORECASE),
    re.compile(r"\baudiogram", re.IGNORECASE),
    re.compile(r"\bprogram administrator\b", re.IGNORECASE),
    re.compile(r"\bcopy of (the )?(written )?respiratory protection program\b", re.IGNORECASE),
    re.compile(r"\bmaximum use concentration\b", re.IGNORECASE),
    re.compile(r"\bassigned protection factor\b", re.IGNORECASE),
    re.compile(r"\bfit test\b", re.IGNORECASE),
    re.compile(r"\brecords?\b", re.IGNORECASE),
    re.compile(r"\bmonitoring\b", re.IGNORECASE),
    re.compile(r"\bnoise measurements?\b", re.IGNORECASE),
)

_HAZARD_PATTERNS = (
    "hazard",
    "hazards",
    "injury",
    "injuries",
    "lacerations",
    "amputation",
    "hearing loss",
    "exposed",
    "exposure",
)

_ACTOR_PATTERNS = (
    "employee",
    "employees",
    "worker",
    "workers",
    "operator",
    "operators",
)

_OPERATION_PATTERNS = (
    "welding",
    "cutting",
    "brazing",
    "point of operation",
    "machine",
    "noise exposure",
)


@dataclass(frozen=True)
class PilotSelectionConfig:
    total_size: int = 60
    train_size: int = 42
    val_size: int = 9
    test_size: int = 9
    max_tokens: int = 56
    min_tokens: int = 6
    respiratory_quota: int = 8
    welding_quota: int = 8
    machine_guarding_quota: int = 8
    hearing_quota: int = 6
    hazard_quota: int = 6
    ppe_quota: int = 12


def should_exclude_from_pilot(candidate: CandidateSentence, *, config: PilotSelectionConfig) -> bool:
    if candidate.token_count < config.min_tokens or candidate.token_count > config.max_tokens:
        return True
    lowered = candidate.normalized_text.lower()
    for pattern in _EXCLUDE_PATTERNS:
        if pattern.search(lowered):
            return True
    if "payment requirements" in lowered:
        return True
    if not any(pattern in lowered for pattern in _ACTOR_PATTERNS) and not any(
        pattern in lowered for pattern in _OPERATION_PATTERNS
    ):
        return True
    return False


def score_annotation_readiness(candidate: CandidateSentence) -> int:
    lowered = candidate.normalized_text.lower()
    score = candidate.candidate_score
    if "shall" in lowered:
        score += 2
    if "must" in lowered:
        score += 2
    if "employee" in lowered or "employees" in lowered or "worker" in lowered or "workers" in lowered:
        score += 2
    if "use " in lowered or "using " in lowered:
        score += 2
    if "protect" in lowered or "protected" in lowered:
        score += 2
    if "operation" in lowered or "operations" in lowered:
        score += 2
    if has_explicit_hazard_signal(candidate):
        score += 3
    if 8 <= candidate.token_count <= 40:
        score += 2
    return score


def has_explicit_hazard_signal(candidate: CandidateSentence) -> bool:
    lowered = candidate.normalized_text.lower()
    return any(pattern in lowered for pattern in _HAZARD_PATTERNS)


def freeze_pilot_candidates(
    candidates: list[CandidateSentence],
    *,
    config: PilotSelectionConfig | None = None,
) -> list[CandidateSentence]:
    active_config = config or PilotSelectionConfig()
    eligible = [
        candidate
        for candidate in candidates
        if not should_exclude_from_pilot(candidate, config=active_config)
    ]
    ordered = sorted(
        eligible,
        key=lambda candidate: (-score_annotation_readiness(candidate), candidate.sentence_id),
    )

    selected: list[CandidateSentence] = []
    selected_ids: set[str] = set()

    def reserve(quota: int, predicate) -> None:
        for candidate in ordered:
            if len(selected) >= active_config.total_size:
                return
            if quota <= sum(1 for item in selected if predicate(item)):
                return
            if candidate.sentence_id in selected_ids:
                continue
            if predicate(candidate):
                selected.append(candidate)
                selected_ids.add(candidate.sentence_id)

    reserve(active_config.respiratory_quota, lambda candidate: "respiratory" in candidate.domain_tags)
    reserve(active_config.welding_quota, lambda candidate: "welding" in candidate.domain_tags)
    reserve(active_config.machine_guarding_quota, lambda candidate: "machine_guarding" in candidate.domain_tags)
    reserve(active_config.hearing_quota, lambda candidate: "hearing" in candidate.domain_tags or "noise" in candidate.domain_tags)
    reserve(active_config.hazard_quota, has_explicit_hazard_signal)
    reserve(active_config.ppe_quota, lambda candidate: "ppe" in candidate.domain_tags)

    for candidate in ordered:
        if len(selected) >= active_config.total_size:
            break
        if candidate.sentence_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.sentence_id)

    return selected[: active_config.total_size]


def split_pilot_candidates(
    candidates: list[CandidateSentence],
    *,
    config: PilotSelectionConfig | None = None,
) -> tuple[list[CandidateSentence], list[CandidateSentence], list[CandidateSentence]]:
    active_config = config or PilotSelectionConfig()
    ordered = sorted(candidates, key=lambda candidate: (-score_annotation_readiness(candidate), candidate.sentence_id))
    train = ordered[: active_config.train_size]
    val = ordered[active_config.train_size : active_config.train_size + active_config.val_size]
    test = ordered[
        active_config.train_size + active_config.val_size : active_config.train_size + active_config.val_size + active_config.test_size
    ]
    return train, val, test


def build_pilot_scope_markdown(candidates: list[CandidateSentence], *, config: PilotSelectionConfig | None = None) -> str:
    active_config = config or PilotSelectionConfig()
    train, val, test = split_pilot_candidates(candidates, config=active_config)
    overlap_like_count = sum(
        1
        for candidate in candidates
        if "ppe" in candidate.domain_tags
        and ("operations" in candidate.domain_tags or has_explicit_hazard_signal(candidate))
    )
    lines = [
        "# Manufacturing Text IE Pilot Scope",
        "",
        f"- Total pilot size: `{len(candidates)}`",
        f"- Train/Val/Test: `{len(train)}/{len(val)}/{len(test)}`",
        f"- Respiratory candidates: `{sum(1 for candidate in candidates if 'respiratory' in candidate.domain_tags)}`",
        f"- Welding candidates: `{sum(1 for candidate in candidates if 'welding' in candidate.domain_tags)}`",
        f"- Machine guarding candidates: `{sum(1 for candidate in candidates if 'machine_guarding' in candidate.domain_tags)}`",
        f"- Hearing/noise candidates: `{sum(1 for candidate in candidates if 'hearing' in candidate.domain_tags or 'noise' in candidate.domain_tags)}`",
        f"- PPE candidates: `{sum(1 for candidate in candidates if 'ppe' in candidate.domain_tags)}`",
        f"- Explicit hazard-like candidates: `{sum(1 for candidate in candidates if has_explicit_hazard_signal(candidate))}`",
        f"- Overlap-like candidates: `{overlap_like_count}`",
        "",
        "## Selection Rules",
        "",
        f"- Maximum token count: `{active_config.max_tokens}`",
        f"- Minimum token count: `{active_config.min_tokens}`",
        "- Excluded first-pass categories: standards-by-reference, payment/effective-date notes, and long low-signal administrative clauses.",
        "- Selection priority favors sentences with actor mentions, explicit PPE/operation language, and hazard/exposure cues.",
    ]
    return "\n".join(lines)


def pilot_candidates_to_annotation_seeds(candidates: list[CandidateSentence]) -> list[dict]:
    return [candidate_to_annotation_seed(candidate) for candidate in candidates]
