from site_safety_monitor.data.text_ie_annotations import GoldTripleAnnotation


def test_gold_triple_annotation_preserves_spans_and_predicate():
    triple = GoldTripleAnnotation(
        subject_span=(0, 0),
        predicate="be_equipped_with",
        object_span=(5, 6),
    )

    assert triple.subject_span == (0, 0)
    assert triple.predicate == "be_equipped_with"
    assert triple.object_span == (5, 6)
