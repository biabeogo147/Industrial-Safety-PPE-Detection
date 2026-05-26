# Manufacturing Text IE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a paper-faithful manufacturing-domain `BERT + BIEO + triple decoding` workflow for text information extraction using curated OSHA sentences and manual annotations.

**Architecture:** The implementation is split into four focused layers: `candidate preparation`, `manual annotation contracts`, `BIEO dataset building`, and `real BERT training/inference`. The existing `Site Safety Monitor` safety logic stays untouched; this work only makes the text branch real and trainable.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, YAML, JSONL, and Pytest.

---

## File Structure

The implementation should introduce or modify the following focused files:

- `configs/site_safety_monitor/text_ie/manufacturing.yaml`
  - runtime defaults for checkpoint, max length, batch size, and training paths
- `data/annotation_guidelines/manufacturing_text_ie.md`
  - manual annotation rules for spans, predicates, overlap handling, and exclusions
- `src/site_safety_monitor/data/text_ie_candidates.py`
  - candidate-pool generation and curation helpers
- `src/site_safety_monitor/data/text_ie_annotations.py`
  - gold annotation contracts and readers/writers
- `src/site_safety_monitor/text_ie/alignment.py`
  - word-to-subword alignment for BERT tokenization
- `src/site_safety_monitor/text_ie/dataset.py`
  - dataset builder for gold examples and BIEO label tensors
- `src/site_safety_monitor/text_ie/model.py`
  - extend current scaffold into a trainable multi-label token classifier
- `src/site_safety_monitor/text_ie/train.py`
  - actual training loop, validation, checkpointing, early stopping
- `src/site_safety_monitor/text_ie/infer.py`
  - sentence inference pipeline from raw text to decoded triples
- `src/site_safety_monitor/text_ie/evaluate.py`
  - triple-level exact-match evaluation and overlap subset metrics
- `scripts/prepare_text_ie_candidates.py`
  - export annotation candidate pool from BERT-ready corpus
- `scripts/build_text_ie_dataset.py`
  - convert manual annotations into training-ready JSONL/tensors
- `scripts/train_text_ie.py`
  - real training entry point
- `scripts/infer_text_ie.py`
  - real inference entry point
- `tests/unit/data/test_text_ie_annotations.py`
  - annotation contract tests
- `tests/unit/text_ie/test_alignment.py`
  - token/subword alignment tests
- `tests/unit/text_ie/test_dataset_builder.py`
  - BIEO dataset conversion tests
- `tests/unit/text_ie/test_infer.py`
  - inference decode tests
- `tests/integration/test_text_ie_training_smoke.py`
  - tiny end-to-end train/eval smoke test

### Task 1: Define the Annotation Guideline and Gold Annotation Contract

**Files:**
- Create: `data/annotation_guidelines/manufacturing_text_ie.md`
- Create: `src/site_safety_monitor/data/text_ie_annotations.py`
- Create: `tests/unit/data/test_text_ie_annotations.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.data.text_ie_annotations import GoldTripleAnnotation


def test_gold_triple_annotation_preserves_spans_and_predicate():
    triple = GoldTripleAnnotation(
        subject_span=(0, 0),
        predicate="be_equipped_with",
        object_span=(5, 6),
    )

    assert triple.subject_span == (0, 0)
    assert triple.predicate == "be_equipped_with"
    assert triple.object_span == (5, 6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/data/test_text_ie_annotations.py -v`
Expected: FAIL because the annotation contract module does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- `GoldTripleAnnotation`
- `GoldSentenceAnnotation`
- JSONL reader/writer helpers

- [ ] **Step 4: Write the annotation guideline**

Document:

- predicate set
- span rules
- overlap rules
- exclusion rules
- examples of correct and incorrect span boundaries

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/data/test_text_ie_annotations.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add data/annotation_guidelines/manufacturing_text_ie.md src/site_safety_monitor/data/text_ie_annotations.py tests/unit/data/test_text_ie_annotations.py
git commit -m "feat: define manufacturing text IE annotation contracts"
```

### Task 2: Build the Annotation Candidate Export

**Files:**
- Create: `src/site_safety_monitor/data/text_ie_candidates.py`
- Create: `scripts/prepare_text_ie_candidates.py`
- Test: `tests/integration/test_text_ie_training_smoke.py`

- [ ] **Step 1: Write the failing integration test**

```python
from site_safety_monitor.data.text_ie_candidates import score_candidate_sentence


def test_score_candidate_sentence_prioritizes_schema_relevant_text():
    sentence = "Employees exposed to flying particles shall use eye protection."

    score = score_candidate_sentence(sentence)

    assert score > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_text_ie_training_smoke.py -v`
Expected: FAIL because the candidate-prep module does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- candidate sentence scoring
- candidate pool export from `bert_input/all.jsonl`
- optional top-k or per-domain sampling helpers

- [ ] **Step 4: Wire the CLI**

Support:

- `--input-jsonl`
- `--output-jsonl`
- `--top-k`
- `--domain-tag`

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/integration/test_text_ie_training_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/site_safety_monitor/data/text_ie_candidates.py scripts/prepare_text_ie_candidates.py tests/integration/test_text_ie_training_smoke.py
git commit -m "feat: export manufacturing text IE candidate pool"
```

### Task 3: Implement Gold Annotation Loading and Dataset Splits

**Files:**
- Modify: `src/site_safety_monitor/text_ie/dataset.py`
- Create: `tests/unit/text_ie/test_dataset_builder.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.text_ie.dataset import build_gold_text_ie_record


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/text_ie/test_dataset_builder.py -v`
Expected: FAIL because the helper does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- gold record dataclass or adapter
- split loaders for `train/val/test`
- validation checks for span boundaries

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/text_ie/test_dataset_builder.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/text_ie/dataset.py tests/unit/text_ie/test_dataset_builder.py
git commit -m "feat: load gold manufacturing text IE dataset"
```

### Task 4: Add Word-to-Subword Alignment for BERT

**Files:**
- Create: `src/site_safety_monitor/text_ie/alignment.py`
- Create: `tests/unit/text_ie/test_alignment.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.text_ie.alignment import expand_word_labels_to_subwords


def test_expand_word_labels_to_subwords_copies_parent_labels():
    word_labels = [["SUBJ:be_equipped_with:B"], ["O"], ["OBJ:be_equipped_with:B",], ["OBJ:be_equipped_with:E"]]
    word_ids = [None, 0, 1, 2, 2, 3, None]

    labels = expand_word_labels_to_subwords(word_labels=word_labels, word_ids=word_ids)

    assert labels[1] == ["SUBJ:be_equipped_with:B"]
    assert labels[3] == ["OBJ:be_equipped_with:B"]
    assert labels[4] == ["OBJ:be_equipped_with:B"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/text_ie/test_alignment.py -v`
Expected: FAIL because the alignment helper does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- word-id alignment handling
- special token masking
- label propagation to subword pieces

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/text_ie/test_alignment.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/text_ie/alignment.py tests/unit/text_ie/test_alignment.py
git commit -m "feat: add text IE subword alignment"
```

### Task 5: Build the Real BIEO Training Dataset

**Files:**
- Modify: `src/site_safety_monitor/text_ie/dataset.py`
- Create: `scripts/build_text_ie_dataset.py`
- Modify: `tests/unit/text_ie/test_dataset_builder.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.text_ie.dataset import encode_gold_record_for_training


def test_encode_gold_record_for_training_outputs_label_ids():
    encoded = encode_gold_record_for_training(
        text="Workers shall use eye protection.",
        tokens=["Workers", "shall", "use", "eye", "protection", "."],
        triples=[
            {
                "subject_span": (0, 0),
                "predicate": "be_equipped_with",
                "object_span": (3, 4),
            }
        ],
        predicates=["be_equipped_with"],
        max_length=32,
    )

    assert "input_ids" in encoded
    assert "label_matrix" in encoded
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/text_ie/test_dataset_builder.py -v`
Expected: FAIL because the training encoder does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- tokenizer integration
- BIEO tag generation from gold triples
- label-to-id mapping
- truncation and padding

- [ ] **Step 4: Wire the CLI**

Support:

- `--annotations`
- `--output-dir`
- `--encoder-name`
- `--max-length`

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/unit/text_ie/test_dataset_builder.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/site_safety_monitor/text_ie/dataset.py scripts/build_text_ie_dataset.py tests/unit/text_ie/test_dataset_builder.py
git commit -m "feat: build manufacturing text IE training dataset"
```

### Task 6: Turn the Text IE Model Scaffold into a Trainable Token Classifier

**Files:**
- Modify: `src/site_safety_monitor/text_ie/model.py`
- Modify: `tests/unit/text_ie/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.text_ie.model import TransformerTaggerConfig, build_bert_bieo_tagger


def test_tagger_config_exposes_num_labels():
    config = TransformerTaggerConfig(encoder_name="bert-base-cased", num_labels=10)
    assert config.num_labels == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/text_ie/test_metrics.py -v`
Expected: FAIL or be insufficient because the model path is not yet exercised for training.

- [ ] **Step 3: Write the minimal implementation**

Add:

- dropout
- logits output contract
- optional loss computation for multi-label token tagging
- English checkpoint compatibility in config defaults

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/text_ie -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/text_ie/model.py tests/unit/text_ie
git commit -m "feat: finalize bert bieo text IE model"
```

### Task 7: Implement Real Training, Validation, and Early Stopping

**Files:**
- Modify: `src/site_safety_monitor/text_ie/train.py`
- Create: `src/site_safety_monitor/text_ie/evaluate.py`
- Modify: `scripts/train_text_ie.py`
- Create: `tests/integration/test_text_ie_training_smoke.py`

- [ ] **Step 1: Write the failing integration test**

```python
from site_safety_monitor.text_ie.train import build_training_config


def test_build_training_config_uses_paper_like_defaults():
    config = build_training_config()

    assert config.learning_rate == 1e-5
    assert config.early_stopping_patience == 10
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_text_ie_training_smoke.py -v`
Expected: FAIL because the real training path does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- dataloader setup
- optimizer and scheduler
- training loop
- validation loop
- checkpoint save/load
- early stopping on validation triple-level `F1`

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `python -m pytest tests/integration/test_text_ie_training_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/text_ie/train.py src/site_safety_monitor/text_ie/evaluate.py scripts/train_text_ie.py tests/integration/test_text_ie_training_smoke.py
git commit -m "feat: train manufacturing bert text IE model"
```

### Task 8: Implement Real Inference from Raw Sentence to Triples

**Files:**
- Create: `src/site_safety_monitor/text_ie/infer.py`
- Create: `scripts/infer_text_ie.py`
- Create: `tests/unit/text_ie/test_infer.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.text_ie.infer import decode_prediction_bundle


def test_decode_prediction_bundle_returns_triples():
    bundle = decode_prediction_bundle(
        tokens=["Workers", "shall", "use", "eye", "protection", "."],
        predicted_tags=[
            ["SUBJ:be_equipped_with:B"],
            ["O"],
            ["O"],
            ["OBJ:be_equipped_with:B"],
            ["OBJ:be_equipped_with:E"],
            ["O"],
        ],
        predicates=["be_equipped_with"],
    )

    assert bundle[0]["predicate"] == "be_equipped_with"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/text_ie/test_infer.py -v`
Expected: FAIL because the inference module does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Implement:

- tokenizer path
- model checkpoint loading
- thresholding or label selection
- BIEO decoding
- conversion to `TextTriple`

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/text_ie/test_infer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/text_ie/infer.py scripts/infer_text_ie.py tests/unit/text_ie/test_infer.py
git commit -m "feat: infer manufacturing text IE triples"
```

### Task 9: Connect Text IE Output to the Existing Baseline Contract

**Files:**
- Modify: `src/site_safety_monitor/text_ie/predict.py`
- Modify: `docs/products/site_safety_monitor.md`

- [ ] **Step 1: Write the integration note**

Document:

- how raw sentence inference becomes `TextTriple`
- where the safety layer consumes those triples
- which parts remain paper-faithful versus domain-adapted

- [ ] **Step 2: Implement the minimal bridge**

Ensure the inference output uses the current `TextTriple` contract and normalized predicates.

- [ ] **Step 3: Run focused tests**

Run: `python -m pytest tests/unit/text_ie tests/integration/test_end_to_end.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/site_safety_monitor/text_ie/predict.py docs/products/site_safety_monitor.md
git commit -m "feat: bridge manufacturing text IE output into baseline"
```

### Task 10: Full Verification

**Files:**
- Verify: `tests/unit/data/test_text_ie_annotations.py`
- Verify: `tests/unit/text_ie/test_alignment.py`
- Verify: `tests/unit/text_ie/test_dataset_builder.py`
- Verify: `tests/unit/text_ie/test_infer.py`
- Verify: `tests/integration/test_text_ie_training_smoke.py`

- [ ] **Step 1: Run focused text IE tests**

Run:

```bash
python -m pytest tests/unit/data/test_text_ie_annotations.py tests/unit/text_ie tests/integration/test_text_ie_training_smoke.py -v
```

Expected: PASS

- [ ] **Step 2: Run one tiny training smoke command**

Run:

```bash
python scripts/train_text_ie.py --config configs/site_safety_monitor/text_ie/manufacturing.yaml
```

Expected:

- training starts
- one checkpoint is saved
- validation F1 is reported

- [ ] **Step 3: Run one inference smoke command**

Run:

```bash
python scripts/infer_text_ie.py --checkpoint <checkpoint-path> --text "Employees shall use eye protection during grinding operations."
```

Expected:

- decoded triples are printed

- [ ] **Step 4: Commit**

```bash
git add configs data docs scripts src tests
git commit -m "test: verify manufacturing text IE training and inference"
```
