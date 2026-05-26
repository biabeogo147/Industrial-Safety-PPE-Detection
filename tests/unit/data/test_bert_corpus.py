from site_safety_monitor.data.bert_corpus import (
    build_bert_corpus,
    sentence_to_bert_example,
)
from site_safety_monitor.data.corpus_models import SentenceRecord


def test_sentence_to_bert_example_normalizes_and_tokenizes():
    sentence = SentenceRecord(
        sentence_id="1910.132:001:001",
        source_standard="1910.132",
        section_ref="1910.132:001",
        source_url="https://example.com",
        text="Workers   shall use   protective helmets.",
        domain_tags=("ppe",),
    )

    example = sentence_to_bert_example(sentence)

    assert example.normalized_text == "Workers shall use protective helmets."
    assert example.tokens == ("Workers", "shall", "use", "protective", "helmets.")
    assert example.token_count == 5


def test_build_bert_corpus_filters_titles_and_deduplicates():
    rules = {
        "exclude_keywords": ["appendix"],
        "min_characters": 20,
        "min_alpha_tokens": 4,
    }
    sentences = [
        SentenceRecord(
            sentence_id="s1",
            source_standard="1910.133",
            section_ref="1910.133:001",
            source_url="https://example.com/a",
            text="1910.133 - Eye and face protection.",
        ),
        SentenceRecord(
            sentence_id="s2",
            source_standard="1910.133",
            section_ref="1910.133:002",
            source_url="https://example.com/b",
            text="Protective equipment shall be provided and maintained in a reliable condition.",
        ),
        SentenceRecord(
            sentence_id="s3",
            source_standard="1910.133",
            section_ref="1910.133:003",
            source_url="https://example.com/c",
            text="Protective equipment shall be provided and maintained in a reliable condition.",
        ),
    ]

    examples = build_bert_corpus(sentences, rules=rules)

    assert len(examples) == 1
    assert examples[0].sentence_id == "s2"
