from site_safety_monitor.scene_graph.model import PredicateFusionHead


def test_predicate_fusion_head_adds_semantic_and_visual_logits():
    head = PredicateFusionHead(input_dim=12, num_predicates=3)
    subject = [[1.0, 1.0, 1.0, 1.0]]
    relation = [[1.0, 1.0, 1.0, 1.0]]
    object_ = [[1.0, 1.0, 1.0, 1.0]]
    semantic_logits = [[0.1, 0.3, 0.2]]

    logits = head(subject, relation, object_, semantic_logits)

    assert len(logits) == 1
    assert len(logits[0]) == 3
    assert logits[0][1] > logits[0][0]
