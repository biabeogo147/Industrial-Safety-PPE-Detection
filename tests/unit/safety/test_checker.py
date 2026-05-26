from site_safety_monitor.core.triples import TextTriple, VisualTriple
from site_safety_monitor.safety.checker import evaluate_worker


def test_missing_required_ppe_returns_no_and_activates_hazard():
    text_triples = [
        TextTriple("workers", "be_equipped_with", "hard hat"),
        TextTriple("workers", "perform_operations", "work at height"),
        TextTriple("work at height", "occurrence", "head injury from falls"),
    ]
    visual_triples = [
        VisualTriple("worker_0", "worker", "wear", "glove_0", "hand protection"),
    ]

    result = evaluate_worker(worker_id="worker_0", text_triples=text_triples, visual_triples=visual_triples)

    assert result.compliance == "No"
    assert result.missing_requirements == ["hard hat"]
    assert result.hazards == ["head injury from falls"]


def test_matching_all_required_ppe_returns_yes():
    text_triples = [
        TextTriple("workers", "be_equipped_with", "face protection"),
        TextTriple("workers", "be_equipped_with", "hand protection"),
        TextTriple("workers", "perform_operations", "welding operations"),
        TextTriple("welding operation", "occurrence", "burns"),
    ]
    visual_triples = [
        VisualTriple("worker_0", "worker", "wearing", "face_0", "face protection"),
        VisualTriple("worker_0", "worker", "wear", "hand_0", "hand protection"),
    ]

    result = evaluate_worker(worker_id="worker_0", text_triples=text_triples, visual_triples=visual_triples)

    assert result.compliance == "Yes"
    assert result.missing_requirements == []
    assert result.hazards == []


def test_no_detected_worker_returns_not_applicable():
    text_triples = [
        TextTriple("workers", "be_equipped_with", "hard hat"),
    ]

    result = evaluate_worker(worker_id="worker_0", text_triples=text_triples, visual_triples=[])

    assert result.compliance == "N/A"
