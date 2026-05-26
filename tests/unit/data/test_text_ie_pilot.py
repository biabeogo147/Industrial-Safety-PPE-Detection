from site_safety_monitor.data.text_ie_candidates import CandidateSentence
from site_safety_monitor.data.text_ie_pilot import (
    PilotSelectionConfig,
    build_pilot_scope_markdown,
    freeze_pilot_candidates,
    split_pilot_candidates,
    should_exclude_from_pilot,
)


def _candidate(sentence_id: str, text: str, tags: tuple[str, ...], score: int, token_count: int = 12) -> CandidateSentence:
    tokens = tuple(text.split())
    return CandidateSentence(
        sentence_id=sentence_id,
        source_standard="1910.000",
        section_ref=sentence_id,
        source_url="https://example.test",
        text=text,
        normalized_text=text,
        tokens=tokens,
        token_count=token_count,
        language="en",
        domain_tags=tags,
        candidate_score=score,
    )


def test_freeze_pilot_candidates_reserves_core_buckets():
    config = PilotSelectionConfig(
        total_size=6,
        train_size=4,
        val_size=1,
        test_size=1,
        respiratory_quota=1,
        welding_quota=1,
        machine_guarding_quota=1,
        hearing_quota=1,
        hazard_quota=1,
        ppe_quota=1,
    )
    selected = freeze_pilot_candidates(
        [
            _candidate("resp", "Employees shall use respirators when exposed to hazards.", ("ppe", "respiratory"), 10),
            _candidate("weld", "Workers shall use protective clothing during welding operations.", ("welding", "operations"), 9),
            _candidate("guard", "Machine guarding shall protect the operator from injury.", ("machine_guarding", "operations"), 9),
            _candidate("hear", "Protection against the effects of noise exposure shall be provided.", ("noise", "hearing"), 9),
            _candidate("ppe", "Employees shall use eye protection.", ("ppe",), 8),
            _candidate("extra", "Workers shall use gloves when handling materials.", ("ppe", "hand"), 7),
        ],
        config=config,
    )

    selected_ids = {candidate.sentence_id for candidate in selected}

    assert "resp" in selected_ids
    assert "weld" in selected_ids
    assert "guard" in selected_ids
    assert "hear" in selected_ids
    assert "ppe" in selected_ids
    assert len(selected) == 6


def test_split_pilot_candidates_respects_requested_sizes():
    config = PilotSelectionConfig(total_size=6, train_size=4, val_size=1, test_size=1)
    candidates = [
        _candidate(f"s{index}", f"Employees shall use eye protection {index}.", ("ppe",), 10 - index)
        for index in range(6)
    ]

    train, val, test = split_pilot_candidates(candidates, config=config)

    assert len(train) == 4
    assert len(val) == 1
    assert len(test) == 1


def test_build_pilot_scope_markdown_reports_counts():
    config = PilotSelectionConfig(total_size=2, train_size=1, val_size=1, test_size=0)
    candidates = [
        _candidate("resp", "Employees shall use respirators when exposed to hazards.", ("ppe", "respiratory"), 10),
        _candidate("weld", "Workers shall use protective clothing during welding operations.", ("welding", "operations"), 9),
    ]

    markdown = build_pilot_scope_markdown(candidates, config=config)

    assert "Total pilot size: `2`" in markdown
    assert "Respiratory candidates: `1`" in markdown


def test_should_exclude_from_pilot_filters_definition_like_sentences():
    candidate = _candidate(
        "definition",
        "Maximum use concentration means the maximum atmospheric concentration of a hazardous substance.",
        ("ppe", "respiratory"),
        10,
    )

    assert should_exclude_from_pilot(candidate, config=PilotSelectionConfig())
