from site_safety_monitor.text_ie.dataset import (
    build_gold_text_ie_record,
    encode_gold_record_for_training,
)


class FakeEncoding(dict):
    def __init__(self, input_ids, attention_mask, word_ids):
        super().__init__(input_ids=input_ids, attention_mask=attention_mask)
        self._word_ids = word_ids

    def word_ids(self):
        return list(self._word_ids)


class FakeTokenizer:
    def __call__(
        self,
        tokens,
        *,
        is_split_into_words,
        truncation,
        padding,
        max_length,
        return_attention_mask,
    ):
        del is_split_into_words, truncation, padding, max_length, return_attention_mask
        return FakeEncoding(
            input_ids=[101, 1001, 1002, 1003, 1004, 1005, 102, 0],
            attention_mask=[1, 1, 1, 1, 1, 1, 1, 0],
            word_ids=[None, 0, 1, 2, 3, 4, None, None],
        )


def test_build_gold_text_ie_record_keeps_tokens_and_triples():
    record = build_gold_text_ie_record(
        sentence_id="s1",
        text="Workers shall use eye protection.",
        tokens=["Workers", "shall", "use", "eye", "protection", "."],
        triples=[
            {
                "subject_span": (0, 0),
                "predicate": "be_equipped_with",
                "object_span": (3, 4),
            }
        ],
    )

    assert record.sentence_id == "s1"
    assert len(record.triples) == 1


def test_encode_gold_record_for_training_outputs_label_ids():
    encoded = encode_gold_record_for_training(
        text="Workers shall use eye protection.",
        tokens=["Workers", "shall", "use", "eye", "protection"],
        triples=[
            {
                "subject_span": (0, 0),
                "predicate": "be_equipped_with",
                "object_span": (3, 4),
            }
        ],
        predicates=["be_equipped_with"],
        max_length=32,
        tokenizer=FakeTokenizer(),
    )

    assert "input_ids" in encoded
    assert "label_matrix" in encoded
    assert encoded["label_mask"][0] == 0
    assert encoded["label_mask"][1] == 1
