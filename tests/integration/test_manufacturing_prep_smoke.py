from site_safety_monitor.data.osha_corpus import split_into_sentences
from site_safety_monitor.data.text_annotation import sentence_to_seed_triples


def test_split_into_sentences_keeps_regulatory_language():
    text = (
        "The employer shall assess the workplace to determine if hazards are present. "
        "The employer shall select PPE that properly fits each affected employee."
    )

    sentences = split_into_sentences(text)

    assert len(sentences) == 2
    assert sentences[1].startswith("The employer shall select PPE")


def test_sentence_to_seed_triples_extracts_ppe_operation_and_hazard():
    sentence = "Workers exposed to high noise levels shall use hearing protectors to prevent hearing loss."

    triples = sentence_to_seed_triples(sentence)

    assert ("worker", "be_equipped_with", "hearing_protection") in triples
    assert ("worker", "perform_operations", "high_noise_operation") in triples
    assert ("high_noise_operation", "occurrence", "hearing_loss") in triples
