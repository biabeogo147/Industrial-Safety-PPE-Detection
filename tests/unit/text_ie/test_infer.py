from site_safety_monitor.text_ie.infer import decode_prediction_bundle


def test_decode_prediction_bundle_returns_triples():
    bundle = decode_prediction_bundle(
        tokens=["Workers", "shall", "use", "eye", "protection", "."],
        predicted_tags=[
            ["SUBJ:be_equipped_with:B", "SUBJ:be_equipped_with:E"],
            ["O"],
            ["O"],
            ["OBJ:be_equipped_with:B"],
            ["OBJ:be_equipped_with:E"],
            ["O"],
        ],
        predicates=["be_equipped_with"],
    )

    assert bundle[0]["predicate"] == "be_equipped_with"
