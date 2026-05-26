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

## What Lives in Repo

The repo stores light control artifacts:

- `configs/site_safety_monitor/corpus/osha_1910_sources.yaml`
- `configs/site_safety_monitor/corpus/annotation_rules.yaml`
- `data/ontology/canonical_entities.yaml`
- `data/ontology/aliases.yaml`
- `data/ontology/sh17_mapping.yaml`
- `src/site_safety_monitor/data/*`
- `scripts/crawl_osha_corpus.py`
- `scripts/prepare_text_triples.py`
- `scripts/prepare_sh17_derivative.py`

## What Lives in E:\data\SH17

Heavier crawl and derived dataset outputs should stay under:

```text
E:\data\SH17\site_safety_monitor\
  text_corpus\
    raw\
    interim\
    processed\
  vision\
    sh17_derived\
```

This keeps the repo small while still keeping the prepared assets next to the source dataset.

## Corpus Schema

The text side stays faithful to the 2023 paper's triple style:

- `worker -> be_equipped_with -> ppe`
- `worker -> perform_operations -> operation`
- `operation -> occurrence -> injury_or_hazard`

Examples:

- `<worker, be_equipped_with, hearing_protection>`
- `<worker, perform_operations, high_noise_operation>`
- `<high_noise_operation, occurrence, hearing_loss>`

## SH17 Mapping

The ontology-lite layer maps SH17 labels into canonical IDs:

- `Helmet` -> `head_protection`
- `Glasses` -> `eye_protection`
- `Face-guard` -> `face_protection`
- `Face-mask-medical` -> `respiratory_protection`
- `Earmuffs` -> `hearing_protection`
- `Gloves` -> `hand_protection`
- `Shoes` -> `foot_protection`
- `Safety-suit` -> `protective_clothing`
- `Medical-suit` -> `protective_clothing`
- `Safety-vest` -> `high_visibility_clothing`
- `Tools` -> `tool`
- `Person` -> `worker`

Body-part labels such as `Head`, `Face`, `Ear`, `Hands`, and `Foot` are treated as `support_region` classes rather than PPE requirement nodes.

## Relation Quality Note

The local SH17 export currently exposes object detection labels and metadata, but not explicit gold scene-graph relations or the paper's original `on/off` attributes in the YOLO label files.

Because of that, the exported `wear` and `hold` relations are:

- `derived`
- `heuristic`
- `not gold relation annotations`

The current derivation uses:

- canonical class mapping
- nearest-person assignment
- center-inside-person geometry checks with margin expansion

This is good enough for baseline preparation, but it must be reported honestly in experiments.

## Integration Checklist

The next baseline integration phase should touch these existing files first:

- `src/site_safety_monitor/core/normalize.py`
- `src/site_safety_monitor/text_ie/dataset.py`
- `src/site_safety_monitor/scene_graph/dataset.py`
- `configs/site_safety_monitor/text_ie/default.yaml`
- `configs/site_safety_monitor/scene_graph/default.yaml`

The goal of that next phase is not to redesign the baseline. It is to swap in:

- manufacturing text triples from the OSHA corpus
- canonical label normalization from ontology-lite
- SH17-derived object and relation manifests
