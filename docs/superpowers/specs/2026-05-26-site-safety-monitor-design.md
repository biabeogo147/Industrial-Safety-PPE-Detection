# Site Safety Monitor Design

**Scope**

- Reproduce the architecture and decision flow of `Automatic Construction Hazard Identification Integrating On-Site Scene Graphs with Information Extraction in Outfield Test` as faithfully as possible.
- Keep the implementation dataset-agnostic so future work can swap in new image datasets and new regulation text corpora without changing the core pipeline.
- Exclude all ontology, Prolog, and paper-2022 reasoning work from this phase.

**Primary Goal**

Build the first production-style implementation that preserves the paper's three-stage pipeline:

1. `BERT-based information extraction` from regulation text into relational triples.
2. `On-site scene parsing` from images into object detections and visual relational triples.
3. `Automatic safety checking` by matching textual and visual triples to produce PPE compliance and hazard inference outputs.

**Non-Goals**

- No ontology layer.
- No Prolog or symbolic rule engine beyond the deterministic matching logic described in the 2023 paper.
- No replacement of the initial v1 models with newer detectors or relation transformers during this phase.
- No paper-2022 enhancements, even if they would improve semantic normalization.

## Fidelity Targets

The v1 system should preserve the following ideas from the paper:

- Text IE is `schema-driven` and uses a `BERT + BIEO tagging` style pipeline to recover multiple triples per sentence.
- Scene parsing is not plain object detection; it must output `visual triples` and include a `semantic prior` over predicates conditioned on subject and object classes.
- Safety checking is based on `triple matching`, not direct image classification.
- PPE checking returns `Yes`, `No`, or `N/A`.
- Hazard inference is produced from missing PPE plus operation-to-injury triples extracted from regulations.

The v1 system does **not** need the paper's original datasets. Instead, it must expose clean interfaces so that equivalent datasets can be plugged in later.

## Proposed Repository Layout

```text
configs/
  site_safety_monitor/
    text_ie/
    scene_graph/
    safety/
docs/
  products/
  superpowers/
    specs/
    plans/
scripts/
  train_text_ie.py
  eval_text_ie.py
  train_scene_graph.py
  eval_scene_graph.py
  run_site_safety_monitor.py
src/
  site_safety_monitor/
    core/
    data/
    text_ie/
    scene_graph/
    safety/
    pipelines/
tests/
  fixtures/
  unit/
  integration/
```

## Component Boundaries

### 1. `core`

Shared domain models and normalization utilities.

Responsibilities:

- Define `TextTriple`, `VisualTriple`, `HazardTriple`, and `ComplianceDecision`.
- Define relation schema metadata shared by text IE and safety checking.
- Normalize labels and predicates without introducing ontology logic.

This layer is the contract surface for the rest of the project.

### 2. `data`

Dataset adapters and format conversion.

Responsibilities:

- Load regulation text annotations into a schema-driven format compatible with the 2023 architecture.
- Load scene graph annotations from a Visual Genome-like relation format or a custom equivalent.
- Convert dataset-specific labels into the project's canonical vocabulary.

This layer is what makes the codebase reusable with future datasets.

### 3. `text_ie`

Paper-2023-style information extraction pipeline.

Responsibilities:

- Store the relation schema used to define valid triples.
- Encode and decode `BIEO` tag sequences for overlapping triple extraction.
- Build training samples for a transformer sequence tagger.
- Train, evaluate, and run inference for the text IE model.

The first implementation should support any transformer checkpoint, but default to a BERT-style model so the architecture still mirrors the paper.

### 4. `scene_graph`

Paper-2023-style on-site scene parsing pipeline.

Responsibilities:

- Wrap an object detector that exposes ROI-aligned subject, object, and relation-region features.
- Compute a `semantic prior` over predicates using empirical `p(predicate | subject_class, object_class)`.
- Fuse semantic logits with visual logits for predicate prediction.
- Decode object detections plus predicate scores into `VisualTriple` outputs.

This module must preserve the paper's semantic-plus-visual fusion logic even if the exact training dataset changes.

### 5. `safety`

Deterministic safety checking layer.

Responsibilities:

- Select PPE-related textual triples from regulation triples.
- Match textual triples and visual triples at worker level.
- Apply the paper's `Yes / No / N/A` PPE compliance logic.
- Infer hazard presence from operation-to-injury triples when required PPE is missing.

This module is where the system becomes end-to-end usable.

### 6. `pipelines`

Experiment runners and orchestration.

Responsibilities:

- Train text IE independently.
- Train scene graph parsing independently.
- Run end-to-end inference using text IE output, image parsing output, and safety checking.
- Save structured artifacts for later analysis.

## Data Contracts

### Regulation Text Contract

The codebase should accept a dataset format that can represent:

- sentence text
- token list
- subject span
- predicate label
- object span
- optional operation and injury context

Recommended canonical training format:

```json
{
  "sentence_id": "reg-001",
  "text": "Workers should be equipped with hard hats during work at height.",
  "tokens": ["Workers", "should", "be", "equipped", "with", "hard", "hats", "during", "work", "at", "height", "."],
  "triples": [
    {
      "subject_text": "Workers",
      "subject_span": [0, 0],
      "predicate": "be_equipped_with",
      "object_text": "hard hats",
      "object_span": [5, 6]
    },
    {
      "subject_text": "Workers",
      "subject_span": [0, 0],
      "predicate": "perform_operations",
      "object_text": "work at height",
      "object_span": [8, 10]
    }
  ]
}
```

### Scene Graph Contract

The codebase should accept a dataset format that can represent:

- image path
- object bounding boxes and labels
- relation annotations as subject-object-predicate triples

Recommended canonical training format:

```json
{
  "image_id": "img-001",
  "image_path": "path/to/image.jpg",
  "objects": [
    {"id": 0, "label": "worker", "bbox": [100, 50, 240, 420]},
    {"id": 1, "label": "hard_hat", "bbox": [140, 55, 210, 120]}
  ],
  "relations": [
    {"subject_id": 0, "predicate": "wear", "object_id": 1}
  ]
}
```

### Safety Checking Contract

The safety checker should consume:

- a set of textual triples tied to one regulation or operation context
- a set of visual triples tied to one image or one worker instance

And it should produce:

```json
{
  "worker_id": "worker_0",
  "compliance": "No",
  "missing_requirements": ["hard_hat"],
  "hazards": ["head_injury_from_falls"]
}
```

## Baseline Architecture Details

### Text IE

The text pipeline should follow the paper's structure closely:

- schema definition
- sentence tokenization
- BIEO tag generation for subject and object spans
- transformer encoder
- token classification head
- triple decoding from predicted tag sequences

The implementation should keep schema and tag codec separate from model code so that future datasets only change configuration and annotation files.

### Scene Graph

The scene graph pipeline should preserve the paper's decomposition:

- object detector provides subject, object, and relation-region features
- semantic prior branch estimates predicate likelihood from `(subject_class, object_class)`
- visual branch predicts relation logits from fused subject, relation, and object features
- final predicate score is computed from `semantic_logits + visual_logits`

This module should expose a detector adapter so the implementation remains faithful while still allowing practical detector integration.

### Safety Checking

The safety logic should mirror the paper's rule flow exactly:

- if no worker is detected for the relevant context, return `N/A`
- if all PPE triples required by regulation are observed visually, return `Yes`
- if any required PPE triple is absent or only partially matched, return `No`
- when `No`, use operation-to-injury regulation triples to report the corresponding hazard

No ontology-backed expansion should happen here in this phase.

## Testing Strategy

### Unit Tests

- BIEO encoding and decoding for overlapping triples
- label normalization and triple equality
- semantic prior estimation from relation counts
- predicate logit fusion behavior
- PPE compliance logic for `Yes`, `No`, and `N/A`

### Integration Tests

- regulation annotations -> decoded text triples
- image annotations -> decoded visual triples
- text triples + visual triples -> safety decision

### End-to-End Smoke Test

One fixture scenario for:

- `work at height` with missing hard hat -> `No` + `head injury from falls`
- `welding` with required hand and face protection present -> `Yes` + no activated hazard

## Execution Plan Shape

Implementation should proceed in this order:

1. project scaffold and domain contracts
2. text IE data encoding and metrics
3. text IE model and inference CLI
4. scene graph data pipeline and semantic prior
5. scene graph relation head and decoding
6. safety checking engine
7. end-to-end pipeline and experiment configs

This order keeps the pipeline testable in layers while preserving the paper's architecture.

## Success Criteria

The v1 system is considered ready when all of the following are true:

- a text IE module can train and decode paper-style triples from regulation data
- a scene graph module can output visual triples with semantic-plus-visual predicate fusion
- a safety checker can produce `Yes`, `No`, and `N/A` decisions from triples
- an end-to-end runner can reproduce the paper's two canonical flows:
  - missing PPE activates a hazard
  - present PPE suppresses that hazard
- the codebase accepts dataset adapters instead of hard-coding the original paper datasets

## Risks and Guardrails

- The paper does not fully specify every annotation detail. Guardrail: isolate dataset assumptions in `data/` and `configs/`.
- The paper's original scene graph dataset is narrow. Guardrail: keep predicate and label vocabularies configurable.
- Future datasets will differ from the paper's label names. Guardrail: centralize normalization and canonical naming in `core/`.
- Reproduction drift is easy if the code collapses to plain detection. Guardrail: require visual triple output and semantic prior fusion in the v1 system definition.
