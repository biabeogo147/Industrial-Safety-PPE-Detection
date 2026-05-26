from site_safety_monitor.scene_graph.semantic_prior import SemanticPrior


def test_semantic_prior_learns_predicate_distribution_per_object_pair():
    relations = [
        ("worker", "wear", "hard_hat"),
        ("worker", "wear", "hard_hat"),
        ("worker", "hold", "welding_tool"),
    ]

    prior = SemanticPrior()
    prior.fit(relations)

    hard_hat_scores = prior.logits_for(subject_label="worker", object_label="hard_hat")
    tool_scores = prior.logits_for(subject_label="worker", object_label="welding_tool")

    assert hard_hat_scores["wear"] > hard_hat_scores["hold"]
    assert tool_scores["hold"] > tool_scores["wear"]


def test_semantic_prior_uses_empirical_probabilities():
    relations = [
        ("Worker", "Wearing", "Hard Hats"),
        ("worker", "wear", "hard hat"),
        ("worker", "hold", "welding tool"),
    ]

    prior = SemanticPrior()
    prior.fit(relations)

    probabilities = prior.probabilities_for(subject_label="worker", object_label="hard hat")

    assert probabilities["wear"] == 1.0
    assert probabilities["hold"] == 0.0
