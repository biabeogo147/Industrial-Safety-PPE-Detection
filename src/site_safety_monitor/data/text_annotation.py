"""Sentence filtering and lightweight triple seeding for manufacturing text."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import yaml

from site_safety_monitor.data.corpus_models import SentenceRecord, TripleAnnotation
from site_safety_monitor.data.osha_corpus import DEFAULT_OUTPUT_ROOT


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = PACKAGE_ROOT / "configs" / "site_safety_monitor" / "corpus" / "annotation_rules.yaml"

_PPE_PATTERNS = {
    "head_protection": re.compile(r"\b(helmet|hard hat|head protection)\b", re.IGNORECASE),
    "eye_protection": re.compile(r"\b(safety glasses|goggles|eye protection|glasses)\b", re.IGNORECASE),
    "face_protection": re.compile(r"\b(face shield|face guard|face protection)\b", re.IGNORECASE),
    "respiratory_protection": re.compile(r"\b(respirator|respiratory protection|face mask)\b", re.IGNORECASE),
    "hearing_protection": re.compile(r"\b(hearing protectors|hearing protection|earmuffs|earplugs)\b", re.IGNORECASE),
    "hand_protection": re.compile(r"\b(gloves|hand protection)\b", re.IGNORECASE),
    "foot_protection": re.compile(r"\b(foot protection|protective footwear|safety shoes)\b", re.IGNORECASE),
    "protective_clothing": re.compile(r"\b(protective clothing|coveralls|safety suit|medical suit)\b", re.IGNORECASE),
}
_OPERATION_PATTERNS = {
    "welding": re.compile(r"\b(welding|cutting|brazing)\b", re.IGNORECASE),
    "machine_operation": re.compile(r"\b(machine|point of operation|power-transmission)\b", re.IGNORECASE),
    "high_noise_operation": re.compile(r"\b(noise|high noise|occupational noise)\b", re.IGNORECASE),
}
_HAZARD_PATTERNS = {
    "hearing_loss": re.compile(r"\b(hearing loss)\b", re.IGNORECASE),
    "eye_injury": re.compile(r"\b(eye injury|injury to the eyes|face injury)\b", re.IGNORECASE),
    "amputation": re.compile(r"\b(amputation|amputations)\b", re.IGNORECASE),
    "respiratory_irritation": re.compile(r"\b(respiratory irritation|respiratory hazard|inhalation hazard)\b", re.IGNORECASE),
}


def load_annotation_rules(path: str | Path = DEFAULT_RULES_PATH) -> dict[str, list[str]]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def should_keep_sentence(sentence: str, rules: dict[str, list[str]]) -> bool:
    lowered = sentence.lower()
    if any(keyword in lowered for keyword in rules.get("exclude_keywords", [])):
        return False
    return any(keyword in lowered for keyword in rules.get("keep_keywords", []))


def sentence_to_seed_triples(sentence: str) -> set[tuple[str, str, str]]:
    triples: set[tuple[str, str, str]] = set()
    for entity, pattern in _PPE_PATTERNS.items():
        if pattern.search(sentence):
            triples.add(("worker", "be_equipped_with", entity))

    matched_operations = [
        entity for entity, pattern in _OPERATION_PATTERNS.items() if pattern.search(sentence)
    ]
    for operation in matched_operations:
        triples.add(("worker", "perform_operations", operation))

    matched_hazards = [entity for entity, pattern in _HAZARD_PATTERNS.items() if pattern.search(sentence)]
    for operation in matched_operations:
        for hazard in matched_hazards:
            triples.add((operation, "occurrence", hazard))
    return triples


def annotate_sentence_records(
    sentences: list[SentenceRecord],
    rules: dict[str, list[str]] | None = None,
) -> tuple[list[SentenceRecord], list[TripleAnnotation]]:
    active_rules = rules or load_annotation_rules()
    kept_sentences: list[SentenceRecord] = []
    annotations: list[TripleAnnotation] = []
    for sentence in sentences:
        if not should_keep_sentence(sentence.text, active_rules):
            continue
        triples = sentence_to_seed_triples(sentence.text)
        if not triples:
            continue
        kept_sentences.append(sentence)
        for subject, predicate, obj in sorted(triples):
            annotations.append(
                TripleAnnotation(
                    sentence_id=sentence.sentence_id,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                )
            )
    return kept_sentences, annotations


def export_processed_text_dataset(
    sentences: list[SentenceRecord],
    annotations: list[TripleAnnotation],
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Path]:
    processed_root = Path(output_root) / "text_corpus" / "processed" / "triples"
    processed_root.mkdir(parents=True, exist_ok=True)

    sentence_map = {sentence.sentence_id: sentence for sentence in sentences}
    grouped_annotations: dict[str, list[TripleAnnotation]] = {}
    for annotation in annotations:
        grouped_annotations.setdefault(annotation.sentence_id, []).append(annotation)

    records: list[dict] = []
    for sentence_id, sentence in sentence_map.items():
        sentence_annotations = grouped_annotations.get(sentence_id, [])
        if not sentence_annotations:
            continue
        records.append(
            {
                "sentence_id": sentence.sentence_id,
                "standard_number": sentence.source_standard,
                "section_ref": sentence.section_ref,
                "source_url": sentence.source_url,
                "text": sentence.text,
                "triples": [asdict(annotation) for annotation in sentence_annotations],
            }
        )

    split_names = ("train", "val", "test")
    split_records = _split_records(records)
    output_paths: dict[str, Path] = {}
    for split_name, split_payload in zip(split_names, split_records, strict=True):
        path = processed_root / f"{split_name}.jsonl"
        output_paths[split_name] = path
        with path.open("w", encoding="utf-8") as handle:
            for record in split_payload:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
    return output_paths


def _split_records(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if not records:
        return [], [], []
    total = len(records)
    train_cutoff = max(1, round(total * 0.7))
    val_cutoff = max(train_cutoff + 1, round(total * 0.85)) if total > 1 else train_cutoff
    train = records[:train_cutoff]
    val = records[train_cutoff:val_cutoff]
    test = records[val_cutoff:]
    if not val and test:
        val, test = test[:1], test[1:]
    return train, val, test
