"""BIEO codec for schema-driven relation extraction."""

from __future__ import annotations

from collections import defaultdict


class BIEOCodec:
    """Encode and decode schema-driven overlapping triples."""

    def __init__(self, predicates: list[str] | tuple[str, ...]) -> None:
        self.predicates = tuple(predicates)

    def tag_space(self) -> tuple[str, ...]:
        tags: list[str] = []
        for predicate in self.predicates:
            tags.append(f"SUBJ:{predicate}:B")
            tags.append(f"OBJ:{predicate}:B")
        for predicate in self.predicates:
            tags.append(f"SUBJ:{predicate}:E")
            tags.append(f"OBJ:{predicate}:E")
        tags.extend(["I", "O"])
        return tuple(tags)

    def encode(self, tokens: list[str], triples: list[dict]) -> list[list[str]]:
        matrix: list[set[str]] = [set() for _ in tokens]
        for triple in triples:
            predicate = triple["predicate"]
            subj_start, subj_end = triple["subject_span"]
            obj_start, obj_end = triple["object_span"]

            matrix[subj_start].add(f"SUBJ:{predicate}:B")
            matrix[subj_end].add(f"SUBJ:{predicate}:E")
            for index in range(subj_start + 1, subj_end):
                matrix[index].add("I")

            matrix[obj_start].add(f"OBJ:{predicate}:B")
            matrix[obj_end].add(f"OBJ:{predicate}:E")
            for index in range(obj_start + 1, obj_end):
                matrix[index].add("I")

        output: list[list[str]] = []
        for token_tags in matrix:
            if not token_tags:
                output.append(["O"])
                continue
            ordered_tags = sorted(
                token_tags,
                key=lambda item: (
                    item.endswith(":B") is False,
                    item.endswith(":E") is False,
                    item,
                ),
            )
            output.append(ordered_tags)
        return output

    def decode(self, tokens: list[str], tags: list[list[str]]) -> list[dict]:
        span_positions: dict[str, dict[str, dict[str, list[tuple[int, str]]]]] = defaultdict(
            lambda: defaultdict(lambda: {"B": [], "E": []})
        )

        for index, token_tags in enumerate(tags):
            for tag in token_tags:
                if tag == "O" or tag == "I":
                    continue
                role, predicate, boundary = tag.split(":")
                span_positions[predicate][role][boundary].append((index, tag))

        decoded: list[dict] = []
        for predicate in self.predicates:
            predicate_spans = span_positions.get(predicate)
            if not predicate_spans:
                continue
            subject_spans = self._pair_spans(predicate_spans["SUBJ"])
            object_spans = self._pair_spans(predicate_spans["OBJ"])
            if len(subject_spans) == 1 and len(object_spans) >= 1:
                candidate_pairs = [(subject_spans[0], object_span) for object_span in object_spans]
            elif len(object_spans) == 1 and len(subject_spans) >= 1:
                candidate_pairs = [(subject_span, object_spans[0]) for subject_span in subject_spans]
            elif len(subject_spans) == len(object_spans):
                candidate_pairs = list(zip(subject_spans, object_spans, strict=False))
            else:
                candidate_pairs = [
                    (subject_span, object_span)
                    for subject_span in subject_spans
                    for object_span in object_spans
                ]

            for subject_span, object_span in candidate_pairs:
                decoded.append(
                    {
                        "subject_span": subject_span,
                        "predicate": predicate,
                        "object_span": object_span,
                    }
                )
        return decoded

    @staticmethod
    def _pair_spans(boundaries: dict[str, list[tuple[int, str]]]) -> list[tuple[int, int]]:
        starts = [index for index, _ in boundaries["B"]]
        ends = [index for index, _ in boundaries["E"]]
        return list(zip(starts, ends, strict=False))
