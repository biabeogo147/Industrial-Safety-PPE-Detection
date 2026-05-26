# Manufacturing Corpus Prep Design

**Scope**

- Prepare the `manufacturing / general industry` data assets that must exist before connecting a new text corpus and ontology-like vocabulary into the current `Site Safety Monitor` baseline.
- Keep the runtime decision flow faithful to the 2023 paper: `text triples -> visual triples -> safety checking`.
- Adapt the domain from `construction` to `manufacturing`, because `SH17` is a manufacturing PPE dataset rather than a construction dataset.

**Primary Goal**

Build a clean preparation layer for three assets:

1. `Regulation text corpus` from official manufacturing/general-industry safety standards.
2. `Ontology-lite vocabulary` that normalizes labels across regulations and SH17.
3. `SH17-derived vision dataset` prepared for PPE compliance and lightweight visual relation use.

These assets should be ready to plug into the existing baseline without changing the baseline's core reasoning flow.

**Non-Goals**

- No full ontology stack such as `OWL`, `Protégé`, or `Prolog`.
- No replacement of the current baseline logic with paper-2022 reasoning.
- No attempt to force SH17 into a full scene-graph dataset without explicit derived annotations.
- No expansion into non-official safety blogs, vendor guides, or mixed-jurisdiction corpora for v1.

## Domain Pivot

The key design decision is to treat `SH17` as a `manufacturing / general industry` dataset. That changes the text source selection:

- Reject `OSHA 1926 construction` as the main corpus for this phase.
- Use `OSHA 1910 general industry` as the legal and semantic backbone.

Recommended official source family:

- `1910.132` general PPE requirements
- `1910.133` eye and face protection
- `1910.134` respiratory protection
- `1910.135` head protection
- `1910.136` foot protection
- `1910.138` hand protection
- `1910.95` occupational noise exposure
- `1910.212` machine guarding
- `1910.252` welding, cutting, and brazing

This backbone matches SH17 much better than construction-specific standards because SH17 contains classes such as `earmuffs`, `face-guard`, `gloves`, `safety-suit`, `medical-suit`, and `tools`.

## Design Principles

- `Official-first`: only crawl from official OSHA pages for the initial corpus.
- `Schema-faithful`: keep the paper-2023 triple style instead of inventing a different reasoning interface.
- `Vocabulary-first`: normalize labels aggressively before model training or integration.
- `Data-outside-repo`: keep large raw and processed datasets on `E:\data`, while the repo stores manifests, schemas, mappings, and light fixtures.
- `Traceable provenance`: every text sample must keep section and URL provenance.

## Target Artifacts

### 1. Manufacturing Regulation Corpus

The prepared text corpus should exist in three stages:

- `source manifest`
- `raw crawled sections`
- `sentence-level triple dataset`

Recommended storage layout outside the repo:

```text
E:\data\site_safety_monitor\
  text_corpus\
    raw\
      osha_1910\
    interim\
      sections\
      sentences\
    processed\
      triples\
```

The repo should store only lightweight control files:

```text
configs\
  site_safety_monitor\
    corpus\
data\
  manifests\
  ontology\
docs\
  datasets\
```

### 2. Ontology-Lite Vocabulary

This is not a formal ontology engine. It is a normalized domain vocabulary containing:

- canonical entity IDs
- aliases and spelling variants
- SH17-to-canonical label mapping
- relation vocabulary
- hazard vocabulary
- optional lightweight hierarchy

Example canonical entities:

- `worker`
- `hand_protection`
- `hearing_protection`
- `head_protection`
- `eye_protection`
- `face_protection`
- `respiratory_protection`
- `foot_protection`
- `protective_clothing`
- `tool`
- `machine_operation`
- `welding`
- `hearing_loss`
- `eye_injury`
- `amputation`
- `respiratory_irritation`

### 3. SH17-Derived Vision Prep

SH17 is currently an object detection dataset with useful `on/off` tags, not a full visual relation dataset.

The preparation layer should therefore produce:

- a `PPE-focused class subset`
- derived worker-to-PPE relation candidates
- derived worker-to-tool relation candidates
- metadata-backed train/val/test manifests

The first derived relations can be weakly defined from SH17 labels:

- `worker -> wear -> helmet`
- `worker -> wear -> glasses`
- `worker -> wear -> gloves`
- `worker -> wear -> shoes`
- `worker -> wear -> face_guard`
- `worker -> wear -> face_mask`
- `worker -> wear -> earmuffs`
- `worker -> wear -> safety_suit`
- `worker -> hold -> tool`

These are not perfect scene-graph annotations, but they are sufficient to prepare a relation-ready derivative that matches the baseline contracts.

## Schema Alignment with the Current Baseline

The baseline should keep its paper-2023 reasoning flow. The prep work must map into the current contracts rather than redefine them.

### Text Schema

The regulation corpus should be annotated into a constrained schema:

- `worker -> be_equipped_with -> ppe`
- `worker -> perform_operations -> operation`
- `operation -> occurrence -> injury_or_hazard`

Example:

- `<worker, be_equipped_with, hearing_protection>`
- `<worker, perform_operations, high_noise_operation>`
- `<high_noise_operation, occurrence, hearing_loss>`

### Visual Schema

The SH17 derivative should support:

- `worker -> wear -> ppe`
- `worker -> hold -> tool`

### Normalization Layer

The ontology-lite layer must bridge:

- OSHA wording such as `protective helmet`, `eye protection`, `hand protection`, `protective footwear`
- SH17 labels such as `Helmet`, `Glasses`, `Gloves`, `Shoes`

Examples:

- `protective helmet` -> `head_protection`
- `hard hat` -> `head_protection`
- `helmet` -> `head_protection`
- `safety glasses` -> `eye_protection`
- `glasses` -> `eye_protection`
- `earplugs` -> `hearing_protection`
- `earmuffs` -> `hearing_protection`
- `face shield` -> `face_protection`
- `face-guard` -> `face_protection`
- `respirator` -> `respiratory_protection`
- `face-mask-medical` -> `respiratory_protection`

## Data Preparation Workstreams

### Workstream A: Source Manifest and Crawl Rules

Define an official manifest for the OSHA pages to crawl, including:

- standard number
- section title
- canonical URL
- crawl priority
- domain tag such as `ppe`, `noise`, `machine_guarding`, `welding`

This keeps the corpus reproducible and prevents scope creep.

### Workstream B: Section and Sentence Extraction

Convert each source page into:

- raw page snapshot
- section-level text
- sentence-level rows with provenance

Each sentence row should keep:

- `sentence_id`
- `source_standard`
- `section_ref`
- `source_url`
- `text`
- `language`
- `domain_tags`

### Workstream C: Triple Annotation

Each retained sentence should be annotated into one or more schema triples.

Examples of retained sentence families:

- PPE requirement sentences
- operation/process sentences
- hazard/injury consequence sentences

Examples of excluded sentence families:

- payment and employer responsibility clauses that do not affect image-to-rule matching
- procedural certification language
- document navigation or appendix boilerplate

### Workstream D: Ontology-Lite Files

Create machine-readable files for:

- canonical entities
- aliases
- predicates
- SH17 label mapping
- hazard mapping

These files should become the normalization authority for the repo.

### Workstream E: SH17 Derivative Dataset

Create a derived dataset from `E:\data\SH17` that:

- filters or tags the PPE-relevant classes
- converts labels into canonical IDs
- derives simple `wear` and `hold` relations from `on/off` labels and person overlap heuristics
- exports manifests compatible with the existing scene-graph dataset contract

## Acceptance Criteria

This prep phase is complete when the following are true:

- A reproducible `OSHA 1910` source manifest exists.
- The repo contains a stable ontology-lite vocabulary and SH17 mapping files.
- A sentence-level text dataset exists with provenance and triple annotations.
- A derived SH17 dataset exists with canonical class mapping and relation-ready exports.
- The prepared assets can be consumed by the current baseline adapters without changing core baseline reasoning.

## Risks and Controls

- `Risk`: SH17 does not contain gold relation annotations.
  - `Control`: treat the first relation dataset as a documented derived dataset, not as ground-truth scene-graph gold.
- `Risk`: OSHA wording is broader than what images can validate.
  - `Control`: keep only image-verifiable clauses in the first supervised text set.
- `Risk`: manufacturing PPE classes do not map 1:1 to SH17.
  - `Control`: use canonical abstract classes such as `head_protection` and `hearing_protection`.
- `Risk`: body-part labels in SH17 may tempt overfitting.
  - `Control`: keep body-part labels as support features, not mandatory PPE requirement nodes.

## Recommended Sequencing

1. Lock source manifest and canonical vocabulary.
2. Build the text corpus and triple annotations.
3. Build SH17 mapping and derived relation exports.
4. Only then connect the prepared assets into the current baseline loaders and configs.
