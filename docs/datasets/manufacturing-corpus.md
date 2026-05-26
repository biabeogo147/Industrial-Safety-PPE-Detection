# Manufacturing Corpus

`Manufacturing Corpus` is the preparation layer that adapts `Site Safety Monitor` from the original construction-oriented paper setting into the `manufacturing / general industry` setting required by `SH17`.

## Why OSHA 1910

`SH17` is a PPE dataset for `manufacturing industry`, not a construction dataset. The text backbone therefore uses `OSHA 1910 general industry` rather than `OSHA 1926 construction`.

The initial official source set is:

- `1910.132` general PPE requirements
- `1910.133` eye and face protection
- `1910.134` respiratory protection
- `1910.135` head protection
- `1910.136` foot protection
- `1910.138` hand protection
- `1910.95` occupational noise exposure
- `1910.212` machine guarding
- `1910.252` welding, cutting, and brazing

This source family matches the SH17 label space much better because SH17 includes `helmet`, `glasses`, `face-guard`, `face-mask`, `earmuffs`, `gloves`, `shoes`, `safety-suit`, and `tools`.

## Current Scope

At this stage we keep only two prepared assets:

- `SH17` as the vision dataset root at `E:\data\SH17`
- `OSHA 1910` as a clean sentence corpus formatted for later BERT use

We do **not** keep:

- ontology-lite files
- weakly generated text triples
- SH17-derived scene-graph or relation files

## What Lives in Repo

The repo stores light control artifacts:

- `configs/site_safety_monitor/corpus/osha_1910_sources.yaml`
- `configs/site_safety_monitor/corpus/bert_corpus_rules.yaml`
- `src/site_safety_monitor/data/*`
- `scripts/crawl_osha_corpus.py`
- `scripts/prepare_bert_corpus.py`

## What Lives in E:\data\SH17

Heavier crawl and corpus outputs stay under:

```text
E:\data\SH17\site_safety_monitor\
  text_corpus\
    raw\
    interim\
    processed\
  text_ie\
    candidates\
    annotations\
    processed\
      bieo\
    artifacts\
```

This keeps the repo small while still keeping the prepared assets next to the source dataset.

## BERT-Ready Format

The processed corpus is sentence-level and keeps provenance for later labeling and BERT tokenization.

Each exported JSONL row contains:

- `sentence_id`
- `source_standard`
- `section_ref`
- `source_url`
- `text`
- `normalized_text`
- `tokens`
- `token_count`
- `language`
- `domain_tags`

WordPiece tokenization is intentionally left to the chosen BERT tokenizer at training time. The corpus export only guarantees that the sentence text is clean, deduplicated, and consistently structured.

## Integration Checklist

The next baseline integration phase should touch these existing files first:

- `src/site_safety_monitor/core/normalize.py`
- `src/site_safety_monitor/text_ie/dataset.py`
- `configs/site_safety_monitor/text_ie/default.yaml`

The goal of that next phase is not to redesign the baseline. It is to swap in:

- BERT-ready manufacturing sentences from the OSHA corpus
- a later manually labeled text-IE dataset built from those sentences

## Next Text IE Phase

The approved next step is:

- manually annotate a manufacturing-domain text IE dataset from the OSHA candidate pool
- train an English `BERT + BIEO` model
- decode model outputs back into relational triples compatible with the baseline

The current exported annotation assets are:

- `E:\data\SH17\site_safety_monitor\text_ie\candidates\all_candidates.jsonl`
- `E:\data\SH17\site_safety_monitor\text_ie\candidates\annotation_seed.jsonl`

`annotation_seed.jsonl` keeps the candidate sentence metadata and tokens, but starts with `triples: []` so it can be filled manually into a gold annotation set.

The detailed design and plan live in:

- `docs/superpowers/specs/2026-05-26-manufacturing-text-ie-design.md`
- `docs/superpowers/plans/2026-05-26-manufacturing-text-ie-plan.md`
- `docs/superpowers/plans/2026-05-26-manufacturing-text-ie-training-readiness-plan.md`
