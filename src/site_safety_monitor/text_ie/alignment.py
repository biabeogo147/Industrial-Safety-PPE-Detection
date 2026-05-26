"""Word-to-subword alignment helpers for BERT-style tokenization."""

from __future__ import annotations


def expand_word_labels_to_subwords(
    word_labels: list[list[str]],
    word_ids: list[int | None],
) -> list[list[str]]:
    labels: list[list[str]] = []
    for word_id in word_ids:
        if word_id is None:
            labels.append([])
            continue
        labels.append(list(word_labels[word_id]))
    return labels


def collapse_subword_labels_to_words(
    subword_labels: list[list[str]],
    word_ids: list[int | None],
    num_words: int,
) -> list[list[str]]:
    word_sets: list[set[str]] = [set() for _ in range(num_words)]
    for labels, word_id in zip(subword_labels, word_ids, strict=False):
        if word_id is None:
            continue
        word_sets[word_id].update(labels)

    output: list[list[str]] = []
    for labels in word_sets:
        if not labels:
            output.append(["O"])
            continue
        output.append(sorted(labels))
    return output
