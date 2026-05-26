from site_safety_monitor.text_ie.alignment import expand_word_labels_to_subwords


def test_expand_word_labels_to_subwords_copies_parent_labels():
    word_labels = [
        ["SUBJ:be_equipped_with:B"],
        ["O"],
        ["OBJ:be_equipped_with:B"],
        ["OBJ:be_equipped_with:E"],
    ]
    word_ids = [None, 0, 1, 2, 2, 3, None]

    labels = expand_word_labels_to_subwords(word_labels=word_labels, word_ids=word_ids)

    assert labels[1] == ["SUBJ:be_equipped_with:B"]
    assert labels[3] == ["OBJ:be_equipped_with:B"]
    assert labels[4] == ["OBJ:be_equipped_with:B"]
