from site_safety_monitor.text_ie.codec import BIEOCodec


def test_codec_round_trips_overlapping_triples():
    tokens = ["Workers", "should", "wear", "face", "protection", "during", "welding", "operations"]
    triples = [
        {
            "subject_span": (0, 0),
            "predicate": "be_equipped_with",
            "object_span": (3, 4),
        },
        {
            "subject_span": (0, 0),
            "predicate": "perform_operations",
            "object_span": (6, 7),
        },
    ]

    codec = BIEOCodec(predicates=["be_equipped_with", "perform_operations"])
    encoded = codec.encode(tokens=tokens, triples=triples)
    decoded = codec.decode(tokens=tokens, tags=encoded)

    assert decoded == triples


def test_codec_exposes_paper_style_tag_count():
    codec = BIEOCodec(predicates=["be_equipped_with", "perform_operations", "occurrence"])

    assert len(codec.tag_space()) == (4 * 3) + 2


def test_codec_decodes_one_subject_multiple_objects_for_same_predicate():
    tokens = ["Workers", "wear", "face", "protection", "and", "hand", "protection"]
    triples = [
        {
            "subject_span": (0, 0),
            "predicate": "be_equipped_with",
            "object_span": (2, 3),
        },
        {
            "subject_span": (0, 0),
            "predicate": "be_equipped_with",
            "object_span": (5, 6),
        },
    ]

    codec = BIEOCodec(predicates=["be_equipped_with"])
    encoded = codec.encode(tokens=tokens, triples=triples)
    decoded = codec.decode(tokens=tokens, tags=encoded)

    assert decoded == triples
