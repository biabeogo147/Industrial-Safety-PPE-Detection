from site_safety_monitor.data.text_ie_candidates import CandidateSentence, candidate_to_annotation_seed


def test_candidate_to_annotation_seed_starts_with_empty_triples():
    candidate = CandidateSentence(
        sentence_id="s1",
        source_standard="1910.133",
        section_ref="1910.133(a)(1)",
        source_url="https://example.test/1910.133",
        text="Employees shall use eye protection.",
        normalized_text="Employees shall use eye protection.",
        tokens=("Employees", "shall", "use", "eye", "protection", "."),
        token_count=6,
        language="en",
        domain_tags=("manufacturing", "ppe"),
        candidate_score=7,
    )

    seed = candidate_to_annotation_seed(candidate)

    assert seed["sentence_id"] == "s1"
    assert seed["triples"] == []
    assert seed["tokens"] == ["Employees", "shall", "use", "eye", "protection", "."]
