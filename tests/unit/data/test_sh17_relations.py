from site_safety_monitor.data.sh17_prepare import derive_relations


def test_derive_relations_links_wearables_and_tools_to_worker():
    objects = [
        {
            "id": 0,
            "original_label": "person",
            "canonical_id": "worker",
            "role": "actor",
            "bbox": [0.2, 0.1, 0.8, 0.95],
        },
        {
            "id": 1,
            "original_label": "helmet",
            "canonical_id": "head_protection",
            "role": "ppe",
            "bbox": [0.38, 0.08, 0.62, 0.26],
        },
        {
            "id": 2,
            "original_label": "tool",
            "canonical_id": "tool",
            "role": "object",
            "bbox": [0.62, 0.52, 0.85, 0.88],
        },
    ]

    relations = derive_relations(objects)

    assert {"subject_id": 0, "predicate": "wear", "object_id": 1} in relations
    assert {"subject_id": 0, "predicate": "hold", "object_id": 2} in relations
