"""Dataset loading helpers for relation annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneObject:
    id: int
    label: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class SceneRelation:
    subject_id: int
    predicate: str
    object_id: int


@dataclass(frozen=True)
class SceneAnnotation:
    image_id: str
    image_path: str
    objects: tuple[SceneObject, ...]
    relations: tuple[SceneRelation, ...]


def load_scene_graph_dataset(path: str | Path) -> list[SceneAnnotation]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload if isinstance(payload, list) else [payload]
    annotations: list[SceneAnnotation] = []
    for record in records:
        annotations.append(
            SceneAnnotation(
                image_id=record["image_id"],
                image_path=record["image_path"],
                objects=tuple(
                    SceneObject(
                        id=item["id"],
                        label=item["label"],
                        bbox=tuple(item["bbox"]),
                    )
                    for item in record.get("objects", [])
                ),
                relations=tuple(
                    SceneRelation(
                        subject_id=item["subject_id"],
                        predicate=item["predicate"],
                        object_id=item["object_id"],
                    )
                    for item in record.get("relations", [])
                ),
            )
        )
    return annotations
