# Manufacturing Corpus Prep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare official manufacturing-domain text data, ontology-lite vocabulary files, and SH17-derived vision assets so they can plug into the current `Site Safety Monitor` baseline without changing baseline reasoning.

**Architecture:** The preparation work is split into three streams: `text corpus`, `ontology-lite`, and `SH17 derivative exports`. Heavy data stays on `E:\data`, while the repo stores manifests, configs, mappings, scripts, and lightweight fixtures. The resulting assets must match the existing baseline contracts for `text_ie`, `scene_graph`, and `safety`.

**Tech Stack:** Python, requests/httpx or Playwright-free HTTP fetching, BeautifulSoup/lxml, YAML, JSONL, Pytest, and the existing `site_safety_monitor` package.

---

## File Structure

The implementation should introduce or modify the following focused files:

- `configs/site_safety_monitor/corpus/osha_1910_sources.yaml`
  - Official source manifest for manufacturing/general-industry standards.
- `configs/site_safety_monitor/corpus/annotation_rules.yaml`
  - Sentence retention and exclusion rules for corpus preparation.
- `data/ontology/canonical_entities.yaml`
  - Canonical IDs for PPE, operations, and hazards.
- `data/ontology/aliases.yaml`
  - Synonym and normalization map across OSHA text and SH17 labels.
- `data/ontology/sh17_mapping.yaml`
  - Mapping from SH17 labels and tags into canonical IDs.
- `src/site_safety_monitor/data/corpus_models.py`
  - Dataclasses for source pages, sentence records, and triple annotations.
- `src/site_safety_monitor/data/osha_corpus.py`
  - Load, crawl, parse, and clean OSHA 1910 text sources.
- `src/site_safety_monitor/data/text_annotation.py`
  - Sentence filtering, triple record serialization, and provenance utilities.
- `src/site_safety_monitor/data/sh17_prepare.py`
  - SH17 canonical mapping and derived relation export pipeline.
- `scripts/crawl_osha_corpus.py`
  - CLI entry point to build raw and interim corpus artifacts on `E:\data`.
- `scripts/prepare_text_triples.py`
  - CLI entry point to create processed sentence and triple datasets.
- `scripts/prepare_sh17_derivative.py`
  - CLI entry point to export canonical SH17-derived manifests.
- `tests/unit/data/test_corpus_models.py`
  - Contract tests for text corpus records.
- `tests/unit/data/test_alias_mapping.py`
  - Canonicalization tests for ontology-lite files.
- `tests/unit/data/test_sh17_mapping.py`
  - Mapping tests from SH17 labels to canonical IDs.
- `tests/integration/test_manufacturing_prep_smoke.py`
  - End-to-end smoke test for the prep pipeline using tiny fixtures.
- `docs/datasets/manufacturing-corpus.md`
  - Human-readable description of the new data assets and provenance rules.

### Task 1: Add the Source Manifest and Corpus Contracts

**Files:**
- Create: `configs/site_safety_monitor/corpus/osha_1910_sources.yaml`
- Create: `src/site_safety_monitor/data/corpus_models.py`
- Create: `tests/unit/data/test_corpus_models.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.data.corpus_models import OshaSourcePage


def test_osha_source_page_requires_standard_and_url():
    page = OshaSourcePage(
        standard_number="1910.132",
        title="General requirements",
        url="https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.132",
        domain_tags=("ppe",),
    )

    assert page.standard_number == "1910.132"
    assert page.domain_tags == ("ppe",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/data/test_corpus_models.py -v`
Expected: FAIL because the module or dataclass does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class OshaSourcePage:
    standard_number: str
    title: str
    url: str
    domain_tags: tuple[str, ...]
```

- [ ] **Step 4: Add the first source manifest entries**

Include at least:

- `1910.132`
- `1910.133`
- `1910.134`
- `1910.135`
- `1910.136`
- `1910.138`
- `1910.95`
- `1910.212`
- `1910.252`

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/data/test_corpus_models.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add configs/site_safety_monitor/corpus/osha_1910_sources.yaml src/site_safety_monitor/data/corpus_models.py tests/unit/data/test_corpus_models.py
git commit -m "feat: add manufacturing corpus source manifest"
```

### Task 2: Create the Ontology-Lite Vocabulary Files

**Files:**
- Create: `data/ontology/canonical_entities.yaml`
- Create: `data/ontology/aliases.yaml`
- Create: `data/ontology/sh17_mapping.yaml`
- Create: `tests/unit/data/test_alias_mapping.py`

- [ ] **Step 1: Write the failing test**

```python
import yaml


def test_sh17_labels_map_to_canonical_ppe_groups():
    with open("data/ontology/sh17_mapping.yaml", "r", encoding="utf-8") as handle:
        mapping = yaml.safe_load(handle)

    assert mapping["Helmet"]["canonical_id"] == "head_protection"
    assert mapping["Glasses"]["canonical_id"] == "eye_protection"
    assert mapping["Earmuffs"]["canonical_id"] == "hearing_protection"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/data/test_alias_mapping.py -v`
Expected: FAIL because the ontology files do not exist yet.

- [ ] **Step 3: Write the minimal ontology-lite files**

Populate canonical IDs for:

- `worker`
- `head_protection`
- `eye_protection`
- `face_protection`
- `respiratory_protection`
- `hearing_protection`
- `hand_protection`
- `foot_protection`
- `protective_clothing`
- `tool`
- `welding`
- `machine_operation`
- `high_noise_operation`
- `eye_injury`
- `hearing_loss`
- `amputation`

- [ ] **Step 4: Add alias coverage**

Include aliases such as:

- `helmet`
- `hard hat`
- `protective helmet`
- `glasses`
- `safety glasses`
- `face shield`
- `face-guard`
- `respirator`
- `face-mask-medical`
- `earmuffs`
- `earplugs`
- `gloves`
- `protective footwear`
- `safety shoes`

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/data/test_alias_mapping.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add data/ontology/canonical_entities.yaml data/ontology/aliases.yaml data/ontology/sh17_mapping.yaml tests/unit/data/test_alias_mapping.py
git commit -m "feat: add manufacturing ontology-lite vocabulary"
```

### Task 3: Build the OSHA Crawl and Section Extraction Pipeline

**Files:**
- Create: `src/site_safety_monitor/data/osha_corpus.py`
- Create: `scripts/crawl_osha_corpus.py`
- Create: `configs/site_safety_monitor/corpus/annotation_rules.yaml`
- Test: `tests/integration/test_manufacturing_prep_smoke.py`

- [ ] **Step 1: Write the failing integration test**

```python
from site_safety_monitor.data.osha_corpus import split_into_sentences


def test_split_into_sentences_keeps_regulatory_language():
    text = (
        "The employer shall assess the workplace to determine if hazards are present. "
        "The employer shall select PPE that properly fits each affected employee."
    )

    sentences = split_into_sentences(text)

    assert len(sentences) == 2
    assert sentences[1].startswith("The employer shall select PPE")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_manufacturing_prep_smoke.py -v`
Expected: FAIL because the parser utilities do not exist yet.

- [ ] **Step 3: Write the minimal parser implementation**

Provide:

- source manifest loader
- HTML fetcher with deterministic output paths under `E:\data\site_safety_monitor\text_corpus\raw`
- section extraction
- sentence splitting
- sentence provenance records

- [ ] **Step 4: Wire the CLI**

Support arguments:

- `--output-root`
- `--source-manifest`
- `--limit`
- `--refresh`

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/integration/test_manufacturing_prep_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/site_safety_monitor/data/osha_corpus.py scripts/crawl_osha_corpus.py configs/site_safety_monitor/corpus/annotation_rules.yaml tests/integration/test_manufacturing_prep_smoke.py
git commit -m "feat: add osha general industry corpus preparation"
```

### Task 4: Create Sentence Filtering and Triple Export for the Text Dataset

**Files:**
- Create: `src/site_safety_monitor/data/text_annotation.py`
- Create: `scripts/prepare_text_triples.py`
- Modify: `tests/integration/test_manufacturing_prep_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.data.text_annotation import sentence_to_seed_triples


def test_sentence_to_seed_triples_extracts_ppe_operation_and_hazard():
    sentence = "Workers exposed to high noise levels shall use hearing protectors."

    triples = sentence_to_seed_triples(sentence)

    assert ("worker", "be_equipped_with", "hearing_protection") in triples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_manufacturing_prep_smoke.py -v`
Expected: FAIL because the annotation utilities do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- sentence retention filtering
- seed rule extraction for obvious PPE sentences
- JSONL export for sentence and triple records
- provenance preservation

- [ ] **Step 4: Add output paths**

Write processed data under:

- `E:\data\site_safety_monitor\text_corpus\processed\triples\train.jsonl`
- `E:\data\site_safety_monitor\text_corpus\processed\triples\val.jsonl`
- `E:\data\site_safety_monitor\text_corpus\processed\triples\test.jsonl`

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/integration/test_manufacturing_prep_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/site_safety_monitor/data/text_annotation.py scripts/prepare_text_triples.py tests/integration/test_manufacturing_prep_smoke.py
git commit -m "feat: export manufacturing text triples"
```

### Task 5: Prepare the SH17 Canonical Mapping and Derived Relations

**Files:**
- Create: `src/site_safety_monitor/data/sh17_prepare.py`
- Create: `scripts/prepare_sh17_derivative.py`
- Create: `tests/unit/data/test_sh17_mapping.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.data.sh17_prepare import map_sh17_label


def test_map_sh17_label_normalizes_into_canonical_ids():
    assert map_sh17_label("Helmet") == "head_protection"
    assert map_sh17_label("Gloves") == "hand_protection"
    assert map_sh17_label("Tools") == "tool"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/data/test_sh17_mapping.py -v`
Expected: FAIL because the preparation module does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- SH17 label mapping loader
- canonical label conversion
- PPE-focused subset selection
- worker-to-PPE candidate relation derivation
- worker-to-tool candidate relation derivation

- [ ] **Step 4: Define derivative outputs**

Export to:

- `E:\data\site_safety_monitor\vision\sh17_derived\images_manifest.jsonl`
- `E:\data\site_safety_monitor\vision\sh17_derived\objects.jsonl`
- `E:\data\site_safety_monitor\vision\sh17_derived\relations.jsonl`
- `E:\data\site_safety_monitor\vision\sh17_derived\splits\train.json`
- `E:\data\site_safety_monitor\vision\sh17_derived\splits\val.json`

- [ ] **Step 5: Run the test to verify it passes**

Run: `python -m pytest tests/unit/data/test_sh17_mapping.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/site_safety_monitor/data/sh17_prepare.py scripts/prepare_sh17_derivative.py tests/unit/data/test_sh17_mapping.py
git commit -m "feat: prepare sh17 manufacturing derivative dataset"
```

### Task 6: Add Documentation and Baseline Integration Checklist

**Files:**
- Create: `docs/datasets/manufacturing-corpus.md`
- Modify: `docs/products/site_safety_monitor.md`

- [ ] **Step 1: Write the documentation draft**

Document:

- why `OSHA 1910` was chosen
- how SH17 maps into canonical PPE groups
- what is and is not considered a gold relation label
- where the prepared assets live on `E:\data`

- [ ] **Step 2: Add the integration checklist**

Document exact next touch points in the current baseline:

- `src/site_safety_monitor/core/normalize.py`
- `src/site_safety_monitor/text_ie/dataset.py`
- `src/site_safety_monitor/scene_graph/dataset.py`
- `configs/site_safety_monitor/text_ie/default.yaml`
- `configs/site_safety_monitor/scene_graph/default.yaml`

- [ ] **Step 3: Verify the docs against the spec**

Read both:

- `docs/superpowers/specs/2026-05-26-manufacturing-corpus-prep-design.md`
- `docs/superpowers/plans/2026-05-26-manufacturing-corpus-prep-plan.md`

Confirm the checklist covers all required artifacts.

- [ ] **Step 4: Commit**

```bash
git add docs/datasets/manufacturing-corpus.md docs/products/site_safety_monitor.md
git commit -m "docs: add manufacturing corpus preparation guide"
```

### Task 7: Full Prep Verification

**Files:**
- Verify: `tests/unit/data/test_corpus_models.py`
- Verify: `tests/unit/data/test_alias_mapping.py`
- Verify: `tests/unit/data/test_sh17_mapping.py`
- Verify: `tests/integration/test_manufacturing_prep_smoke.py`

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/unit/data tests/integration/test_manufacturing_prep_smoke.py -v
```

Expected: PASS

- [ ] **Step 2: Run corpus preparation smoke command**

Run:

```bash
python scripts/crawl_osha_corpus.py --output-root E:\data\site_safety_monitor --limit 2
python scripts/prepare_text_triples.py --output-root E:\data\site_safety_monitor
python scripts/prepare_sh17_derivative.py --sh17-root E:\data\SH17 --output-root E:\data\site_safety_monitor
```

Expected:

- raw OSHA pages exist
- processed text triples exist
- derived SH17 relation files exist

- [ ] **Step 3: Sanity-check output contracts**

Confirm that:

- at least one text record contains `source_url` and `standard_number`
- at least one text triple uses `be_equipped_with`
- at least one SH17-derived relation uses `wear` or `hold`

- [ ] **Step 4: Commit**

```bash
git add configs data docs scripts src tests
git commit -m "test: verify manufacturing corpus prep pipeline"
```
