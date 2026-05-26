"""Prepare SH17 into canonical manufacturing-friendly exports."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SH17_ROOT = Path(r"E:\data\SH17")
DEFAULT_OUTPUT_ROOT = DEFAULT_SH17_ROOT / "site_safety_monitor"
DEFAULT_MAPPING_PATH = PACKAGE_ROOT / "data" / "ontology" / "sh17_mapping.yaml"

_PERSON_CLASS = "worker"
_WEARABLE_ROLES = {"ppe", "ppe_auxiliary"}
_MAPPING_LABELS = {
    "person": "Person",
    "ear": "Ear",
    "ear-mufs": "Earmuffs",
    "face": "Face",
    "face-guard": "Face-guard",
    "face-mask": "Face-mask-medical",
    "foot": "Foot",
    "tool": "Tools",
    "glasses": "Glasses",
    "gloves": "Gloves",
    "helmet": "Helmet",
    "hands": "Hands",
    "head": "Head",
    "medical-suit": "Medical-suit",
    "shoes": "Shoes",
    "safety-suit": "Safety-suit",
    "safety-vest": "Safety-vest",
}


def load_sh17_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> dict[str, dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_sh17_names(sh17_root: str | Path = DEFAULT_SH17_ROOT) -> dict[int, str]:
    payload = yaml.safe_load((Path(sh17_root) / "sh17.local.yaml").read_text(encoding="utf-8"))
    names = payload["names"]
    return {int(index): value for index, value in names.items()}


def map_sh17_label(label: str, mapping_path: str | Path = DEFAULT_MAPPING_PATH) -> str:
    mapping = load_sh17_mapping(mapping_path)
    return mapping[label]["canonical_id"]


def parse_yolo_label_file(path: str | Path, names_by_id: dict[int, str]) -> list[dict]:
    detections: list[dict] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        class_id_str, x_center, y_center, width, height = line.split()
        class_id = int(class_id_str)
        label = names_by_id[class_id]
        x_center_f = float(x_center)
        y_center_f = float(y_center)
        width_f = float(width)
        height_f = float(height)
        detections.append(
            {
                "id": index,
                "class_id": class_id,
                "label": label,
                "bbox": _yolo_to_bbox(x_center_f, y_center_f, width_f, height_f),
            }
        )
    return detections


def derive_scene_objects(
    detections: list[dict],
    mapping: dict[str, dict],
) -> list[dict]:
    objects: list[dict] = []
    for detection in detections:
        original_label = detection["label"]
        title_label = _to_mapping_label(original_label)
        mapping_info = mapping[title_label]
        objects.append(
            {
                "id": detection["id"],
                "original_label": original_label,
                "canonical_id": mapping_info["canonical_id"],
                "role": mapping_info["role"],
                "bbox": detection["bbox"],
            }
        )
    return objects


def derive_relations(objects: list[dict]) -> list[dict]:
    persons = [item for item in objects if item["canonical_id"] == _PERSON_CLASS]
    if not persons:
        return []

    relations: list[dict] = []
    non_person_objects = [item for item in objects if item["canonical_id"] != _PERSON_CLASS]
    for item in non_person_objects:
        person = _closest_person(item["bbox"], persons)
        if person is None:
            continue
        if item["role"] in _WEARABLE_ROLES and _center_inside(item["bbox"], person["bbox"], margin=0.2):
            relations.append(
                {
                    "subject_id": person["id"],
                    "predicate": "wear",
                    "object_id": item["id"],
                }
            )
        elif item["canonical_id"] == "tool" and _center_inside(item["bbox"], person["bbox"], margin=0.35):
            relations.append(
                {
                    "subject_id": person["id"],
                    "predicate": "hold",
                    "object_id": item["id"],
                }
            )
    return relations


def export_sh17_derivative(
    sh17_root: str | Path = DEFAULT_SH17_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    limit: int | None = None,
) -> dict[str, Path]:
    sh17_root_path = Path(sh17_root)
    output_root_path = Path(output_root)
    vision_root = output_root_path / "vision" / "sh17_derived"
    splits_root = vision_root / "splits"
    vision_root.mkdir(parents=True, exist_ok=True)
    splits_root.mkdir(parents=True, exist_ok=True)

    mapping = load_sh17_mapping()
    names_by_id = load_sh17_names(sh17_root_path)
    label_files = sorted((sh17_root_path / "labels").glob("*.txt"))
    if limit is not None:
        label_files = label_files[:limit]

    image_records: list[dict] = []
    object_records: list[dict] = []
    relation_records: list[dict] = []

    for label_file in label_files:
        image_id = label_file.stem
        detections = parse_yolo_label_file(label_file, names_by_id)
        objects = derive_scene_objects(detections, mapping)
        relations = derive_relations(objects)
        metadata = _load_metadata(sh17_root_path / "meta-data" / f"{image_id}.json")

        image_records.append(
            {
                "image_id": image_id,
                "image_path": str(sh17_root_path / "images" / f"{image_id}.jpg"),
                "width": metadata.get("width"),
                "height": metadata.get("height"),
                "source_url": metadata.get("url"),
            }
        )
        object_records.append({"image_id": image_id, "objects": objects})
        relation_records.append({"image_id": image_id, "relations": relations})

    image_path = vision_root / "images_manifest.jsonl"
    objects_path = vision_root / "objects.jsonl"
    relations_path = vision_root / "relations.jsonl"
    _write_jsonl(image_path, image_records)
    _write_jsonl(objects_path, object_records)
    _write_jsonl(relations_path, relation_records)
    split_paths = _export_split_manifests(sh17_root_path, splits_root)

    return {
        "images_manifest": image_path,
        "objects": objects_path,
        "relations": relations_path,
        **split_paths,
    }


def _export_split_manifests(sh17_root: Path, splits_root: Path) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for split_name, source_name in (("train", "train_files_fullpath.txt"), ("val", "val_files_fullpath.txt")):
        source_path = sh17_root / source_name
        image_ids = [Path(line.strip()).stem for line in source_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        output_path = splits_root / f"{split_name}.json"
        output_path.write_text(json.dumps(image_ids, indent=2), encoding="utf-8")
        outputs[split_name] = output_path
    return outputs


def _load_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _to_mapping_label(label: str) -> str:
    return _MAPPING_LABELS[label]


def _yolo_to_bbox(x_center: float, y_center: float, width: float, height: float) -> list[float]:
    half_width = width / 2.0
    half_height = height / 2.0
    return [
        max(0.0, x_center - half_width),
        max(0.0, y_center - half_height),
        min(1.0, x_center + half_width),
        min(1.0, y_center + half_height),
    ]


def _center_inside(bbox: list[float], target_bbox: list[float], margin: float) -> bool:
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    expanded = _expand_bbox(target_bbox, margin)
    return expanded[0] <= x_center <= expanded[2] and expanded[1] <= y_center <= expanded[3]


def _expand_bbox(bbox: list[float], margin: float) -> list[float]:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return [
        max(0.0, bbox[0] - width * margin),
        max(0.0, bbox[1] - height * margin),
        min(1.0, bbox[2] + width * margin),
        min(1.0, bbox[3] + height * margin),
    ]


def _closest_person(bbox: list[float], persons: list[dict]) -> dict | None:
    best_person: dict | None = None
    best_distance: float | None = None
    x_center = (bbox[0] + bbox[2]) / 2.0
    y_center = (bbox[1] + bbox[3]) / 2.0
    for person in persons:
        person_x = (person["bbox"][0] + person["bbox"][2]) / 2.0
        person_y = (person["bbox"][1] + person["bbox"][3]) / 2.0
        distance = (person_x - x_center) ** 2 + (person_y - y_center) ** 2
        if best_distance is None or distance < best_distance:
            best_person = person
            best_distance = distance
    return best_person
