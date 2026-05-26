from site_safety_monitor.data.osha_corpus import split_into_sentences
from site_safety_monitor.data.bert_corpus import build_bert_corpus
from site_safety_monitor.data.corpus_models import SentenceRecord


def test_split_into_sentences_keeps_regulatory_language():
    text = (
        "The employer shall assess the workplace to determine if hazards are present. "
        "The employer shall select PPE that properly fits each affected employee."
    )

    sentences = split_into_sentences(text)

    assert len(sentences) == 2
    assert sentences[1].startswith("The employer shall select PPE")

def test_build_bert_corpus_keeps_clean_regulatory_sentences():
    rules = {
        "exclude_keywords": ["appendix"],
        "min_characters": 20,
        "min_alpha_tokens": 4,
    }
    sentences = [
        SentenceRecord(
            sentence_id="1910.132:001:001",
            source_standard="1910.132",
            section_ref="1910.132:001",
            source_url="https://example.com",
            text="The employer shall select and require employees to use appropriate hand protection.",
            domain_tags=("ppe", "hand"),
        )
    ]

    examples = build_bert_corpus(sentences, rules=rules)

    assert len(examples) == 1
    assert examples[0].source_standard == "1910.132"
    assert examples[0].tokens[0] == "The"
