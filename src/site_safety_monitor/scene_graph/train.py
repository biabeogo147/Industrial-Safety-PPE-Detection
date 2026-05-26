"""Training-side helpers for the scene graph module."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from site_safety_monitor.scene_graph.dataset import load_scene_graph_dataset


@dataclass(frozen=True)
class SceneGraphTrainingConfig:
    detector_family: str = "mask_rcnn"
    detector_backbone: str = "resnext101_fpn"
    semantic_embedding_source: str = "fasttext_common_crawl_2m"
    learning_rate: float = 1e-3
    optimizer: str = "momentum_sgd"
    train_split: float = 0.8
    test_split: float = 0.2


@dataclass(frozen=True)
class SceneGraphSummary:
    num_images: int
    num_objects: int
    num_relations: int


def summarize_scene_graph_data(dataset_path: str | Path) -> SceneGraphSummary:
    records = load_scene_graph_dataset(dataset_path)
    return SceneGraphSummary(
        num_images=len(records),
        num_objects=sum(len(record.objects) for record in records),
        num_relations=sum(len(record.relations) for record in records),
    )


def scene_graph_summary_as_dict(dataset_path: str | Path) -> dict:
    return asdict(summarize_scene_graph_data(dataset_path))
