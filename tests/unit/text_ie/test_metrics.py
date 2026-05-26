from site_safety_monitor.text_ie.metrics import triple_f1


def test_triple_f1_requires_exact_subject_predicate_object_match():
    predicted = {
        ("workers", "be_equipped_with", "hard hats"),
        ("workers", "perform_operations", "work at height"),
    }
    gold = {
        ("workers", "be_equipped_with", "hard hats"),
        ("workers", "perform_operations", "welding operations"),
    }

    precision, recall, f1 = triple_f1(predicted=predicted, gold=gold)

    assert round(precision, 3) == 0.5
    assert round(recall, 3) == 0.5
    assert round(f1, 3) == 0.5
