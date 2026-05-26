from pathlib import Path

import yaml


def test_sh17_labels_map_to_canonical_ppe_groups():
    path = Path("data/ontology/sh17_mapping.yaml")
    with path.open("r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)

    assert mapping["Helmet"]["canonical_id"] == "head_protection"
    assert mapping["Glasses"]["canonical_id"] == "eye_protection"
    assert mapping["Earmuffs"]["canonical_id"] == "hearing_protection"
