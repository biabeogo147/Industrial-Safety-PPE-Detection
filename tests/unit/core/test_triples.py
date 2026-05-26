from site_safety_monitor.core.triples import TextTriple, VisualTriple


def test_text_and_visual_triples_normalize_labels():
    text_triple = TextTriple(
        subject="Workers",
        predicate="be_equipped_with",
        object="Hard Hats",
    )
    visual_triple = VisualTriple(
        subject_id="worker_0",
        subject_label="Worker",
        predicate="Wear",
        object_id="hat_0",
        object_label="Hard Hat",
    )

    assert text_triple.normalized_subject == "workers"
    assert text_triple.normalized_object == "hard hat"
    assert visual_triple.normalized_subject_label == "worker"
    assert visual_triple.normalized_predicate == "wear"
    assert visual_triple.normalized_object_label == "hard hat"


def test_paper_aliases_normalize_operations_and_hazards():
    operation_triple = TextTriple(
        subject="Welding Operations",
        predicate="Perform...Operations",
        object="Burns",
    )

    assert operation_triple.normalized_subject == "welding operation"
    assert operation_triple.normalized_predicate == "perform_operations"
    assert operation_triple.normalized_object == "burn"
