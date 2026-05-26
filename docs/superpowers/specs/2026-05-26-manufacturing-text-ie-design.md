# Manufacturing Text IE Design

**Scope**

- Build the text-side training and inference workflow that stays faithful to the `2023` paper's `BERT + BIEO + triple decoding` design.
- Use the prepared `OSHA 1910` manufacturing corpus as the raw text source.
- Replace the paper's original construction-domain Chinese regulation dataset with a new manufacturing-domain English dataset, while preserving the same task formulation.

**Primary Goal**

Create a production-style `text IE` subsystem that can:

1. take curated OSHA manufacturing sentences,
2. convert them into a manually annotated `schema-driven triple extraction` dataset,
3. train a `BERT-based sequence tagging model` with `BIEO` supervision,
4. decode predictions back into relational triples compatible with the existing safety pipeline.

**Non-Goals**

- No weak-label-only training.
- No seq2seq reformulation.
- No LLM-based extraction baseline in this phase.
- No ontology or symbolic reasoning work.
- No attempt to fully reproduce the paper's scene-graph branch in this subproject.

## Why This Is the Right Next Step

The current repo has:

- a clean `OSHA 1910` sentence corpus,
- a `BIEOCodec`,
- a lightweight transformer model scaffold,
- a baseline safety layer that already consumes textual triples.

What is still missing is the core paper-faithful `text IE learning loop`:

- gold annotation
- BIEO training supervision
- real BERT training
- real inference from raw sentences

This subproject fills exactly that gap.

## Paper-Faithful Principles

This design keeps the following parts of the paper intact:

- `schema-driven extraction`
- `joint entity-relation extraction`
- `BERT-based contextual encoding`
- `BIEO tagging`
- support for `overlapping triples`
- `triple-level exact-match evaluation`

What changes from the original paper:

- the source language changes from Chinese to English
- the domain shifts from construction outfield test to manufacturing/general industry
- the BERT checkpoint should be English-compatible instead of `bert-base-chinese`

These changes are necessary because the new corpus is OSHA English text, but they do not change the underlying method.

## Task Definition

The `text IE` task remains:

- input: one regulation sentence
- output: zero, one, or multiple relational triples

Canonical triple schema for this phase:

- `worker -> be_equipped_with -> ppe`
- `worker -> perform_operations -> operation`
- `operation -> occurrence -> injury_or_hazard`

Example:

- sentence: `Employees exposed to flying particles shall use eye protection during grinding operations.`
- triples:
  - `<employees, be_equipped_with, eye protection>`
  - `<employees, perform_operations, grinding operations>`
  - optional third triple only if the sentence explicitly states a consequence relation

## Data Pipeline

### Stage 1: Corpus Curation

Input source:

- `E:\data\SH17\site_safety_monitor\text_corpus\processed\bert_input\all.jsonl`

This file is not yet a training set. It is the sentence pool used to build the text IE dataset.

The first curation pass should remove or separately mark:

- purely navigational or structural sentences
- legal amendment notes
- effective-date-only sentences
- payment and procurement notes that cannot support image-grounded checking
- sentences whose semantics fall outside the current schema

Output:

- an `annotation candidate pool`

### Stage 2: Manual Annotation

Each kept sentence must be annotated with:

- `tokens`
- `triples`
- for each triple:
  - `subject_span`
  - `predicate`
  - `object_span`

The annotation format must support multiple triples per sentence and overlapping spans.

### Stage 3: BIEO Supervision

Gold triples are converted into BIEO multi-tag supervision using `BIEOCodec`.

This is the paper-faithful supervision target.

### Stage 4: Training

Model:

- English `BERT` encoder
- token-level classifier head
- multi-label BIEO prediction

Recommended default checkpoint:

- `bert-base-cased`

Reason:

- it stays architecturally close to the paper's BERT choice
- it matches the OSHA English corpus

### Stage 5: Inference

Inference flow:

1. raw sentence
2. tokenizer
3. model forward pass
4. predicted tag sets
5. `BIEOCodec.decode(...)`
6. `decoded_triples_to_text_triples(...)`

Output:

- normalized textual triples compatible with `Site Safety Monitor`

## Annotation Design

### Predicate Set

The initial annotation predicate set should stay intentionally small:

- `be_equipped_with`
- `perform_operations`
- `occurrence`

This keeps the first dataset aligned with the manufacturing baseline instead of over-expanding too early.

### Span Rules

The annotation guideline should specify:

- shortest semantically complete subject span
- shortest semantically complete object span
- preserve domain phrases such as `eye protection`, `respiratory protection`, `welding operations`
- annotate surface forms as written, not normalized canonical IDs

### Overlap Rules

Overlapping examples must be preserved, not simplified away.

Typical pattern:

- one subject shared by multiple relations
- one sentence containing both PPE requirement and operation context

This is important because the paper explicitly motivates BIEO through overlapping triple extraction.

## Dataset Layout

Recommended data layers:

```text
E:\data\SH17\site_safety_monitor\
  text_ie\
    candidates\
    annotations\
    processed\
      bieo\
      train.jsonl
      val.jsonl
      test.jsonl
```

Recommended repo-local control files:

```text
configs\
  site_safety_monitor\
    text_ie\
data\
  annotation_guidelines\
scripts\
  prepare_text_ie_candidates.py
  build_text_ie_dataset.py
  train_text_ie.py
  infer_text_ie.py
```

## Training Design

### Model

The model should remain simple:

- pretrained BERT encoder
- dropout
- linear token classification head

No CRF is required for the first faithful version unless we later decide it materially improves overlap decoding without deviating too much from the paper.

### Token Alignment

The annotation spans are word-level, but BERT tokenization may split words into subwords.

The dataset builder must therefore:

- tokenize with the selected Hugging Face tokenizer
- align word-level BIEO tags to subword positions
- define a consistent rule for non-first subword pieces

Recommended first rule:

- copy the parent word's label set to all subword pieces

This is simple and stable for multi-label tagging.

### Optimization Defaults

Use paper-inspired defaults where possible:

- learning rate: `1e-5`
- early stopping patience: `10`
- monitor: validation triple-level `F1`

Batch size, max length, and gradient accumulation can be set according to available hardware.

## Evaluation Design

Primary metric:

- triple-level exact match `precision / recall / F1`

A triple is correct only if:

- subject span matches
- predicate matches
- object span matches

Secondary diagnostics:

- sentence-level exact match
- overlap-only subset F1
- predicate-wise F1
- error buckets:
  - missed subject span
  - missed object span
  - wrong predicate
  - incomplete overlap decode

## Integration Design

The `text IE` output should plug into the existing baseline through:

- `decoded_triples_to_text_triples(...)`
- existing `TextTriple`
- existing safety checker input contract

This means the downstream safety logic does not need redesign.

## Risks and Controls

- `Risk`: OSHA sentences are more legalistic than the paper's source data.
  - `Control`: curate the candidate pool before annotation.
- `Risk`: overlap cases may be underrepresented.
  - `Control`: explicitly sample overlap-heavy sentences during annotation.
- `Risk`: English checkpoint choice may change results relative to the paper.
  - `Control`: keep the method identical and document the checkpoint substitution clearly.
- `Risk`: sentence-only corpus may still include some low-value clauses.
  - `Control`: maintain an annotation exclusion guideline and review loop.

## Recommended Execution Order

1. write annotation guideline
2. build candidate pool
3. manually annotate a pilot set
4. implement dataset builder and BIEO alignment
5. train a first real BERT model
6. run inference and evaluate overlap-heavy sentences
7. expand annotation only after the full loop is stable
