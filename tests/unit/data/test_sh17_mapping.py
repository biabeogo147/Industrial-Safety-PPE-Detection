from site_safety_monitor.data.sh17_prepare import map_sh17_label


def test_map_sh17_label_normalizes_into_canonical_ids():
    assert map_sh17_label("Helmet") == "head_protection"
    assert map_sh17_label("Gloves") == "hand_protection"
    assert map_sh17_label("Tools") == "tool"
