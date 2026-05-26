# Manufacturing Text IE Training Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current OSHA manufacturing text assets into a real, repeatable `manual annotation -> BERT training -> inference` workflow.

**Architecture:** The repo already contains the core code path for `candidate export`, `gold annotation IO`, `BIEO dataset building`, `training`, and `inference`. This plan focuses on the operational layer that is still missing: dataset curation, annotation quality control, local checkpoint preparation, run configuration, model evaluation, and inference acceptance criteria.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, JSONL, YAML, Pytest, and local files under `E:\data\SH17\site_safety_monitor`.

---

## Current Starting Point

Current assets already available:

- `E:\data\SH17\site_safety_monitor\text_corpus\processed\bert_input\all.jsonl`
- `E:\data\SH17\site_safety_monitor\text_ie\candidates\all_candidates.jsonl`
- `E:\data\SH17\site_safety_monitor\text_ie\candidates\annotation_seed.jsonl`
- `data/annotation_guidelines/manufacturing_text_ie.md`
- `scripts/build_text_ie_dataset.py`
- `scripts/train_text_ie.py`
- `scripts/infer_text_ie.py`

Current candidate pool size:

- `157` candidate sentences

Current core assumption:

- keep the paper's `schema-driven + BIEO + triple decoding` method
- substitute the original paper's Chinese encoder with an English-compatible BERT checkpoint

---

## File Structure

This phase should produce or update the following artifacts:

- `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_train.jsonl`
  - first manually labeled training split
- `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_val.jsonl`
  - first manually labeled validation split
- `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_test.jsonl`
  - first manually labeled held-out split
- `E:\data\SH17\site_safety_monitor\text_ie\processed\bieo\pilot_train_encoded.jsonl`
  - encoded BIEO supervision for train
- `E:\data\SH17\site_safety_monitor\text_ie\processed\bieo\pilot_val_encoded.jsonl`
  - encoded BIEO supervision for val
- `E:\data\SH17\site_safety_monitor\text_ie\processed\bieo\pilot_test_encoded.jsonl`
  - encoded BIEO supervision for test
- `E:\data\SH17\site_safety_monitor\text_ie\artifacts\pilot_run_001\best_model.pt`
  - first usable checkpoint
- `E:\data\SH17\site_safety_monitor\text_ie\artifacts\pilot_run_001\best_metrics.json`
  - validation metrics for the saved checkpoint
- `configs/site_safety_monitor/text_ie/manufacturing.yaml`
  - runtime config pointing to the active encoder and annotation paths
- `docs/datasets/manufacturing-corpus.md`
  - updated when a real gold dataset exists
- `docs/products/site_safety_monitor.md`
  - updated when text IE becomes genuinely trainable on gold data

---

## Phase Overview

The work should happen in this order:

1. freeze the annotation scope
2. create a high-signal pilot gold dataset
3. validate annotation quality before training
4. prepare a local BERT checkpoint path
5. build BIEO datasets
6. run the first real training cycle
7. evaluate and inspect errors
8. validate raw-sentence inference
9. only then promote the text IE branch into the baseline pipeline

---

### Task 1: Freeze the Pilot Annotation Scope

**Files:**
- Read: `E:\data\SH17\site_safety_monitor\text_ie\candidates\all_candidates.jsonl`
- Read: `E:\data\SH17\site_safety_monitor\text_ie\candidates\annotation_seed.jsonl`
- Read: `data/annotation_guidelines/manufacturing_text_ie.md`

- [ ] **Step 1: Define the pilot size**

Use a small but representative pilot first:

- target `60` labeled sentences total
- target `42` train / `9` val / `9` test

- [ ] **Step 2: Lock required coverage buckets**

The pilot must include at least:

- `12+` PPE sentences
- `10+` eye/face protection sentences
- `8+` respiratory sentences
- `8+` welding or hot-work sentences
- `8+` machine-guarding or tool-operation sentences
- `6+` hearing/noise sentences
- `6+` explicit hazard/injury consequence sentences

- [ ] **Step 3: Lock overlap coverage**

Require at least:

- `15` sentences with `2+` triples
- `5` sentences with explicit `operation -> occurrence -> hazard`

- [ ] **Step 4: Define exclusions for the pilot**

Do not annotate first-pass sentences that are:

- policy boilerplate
- employer administrative duties without image-groundable semantics
- duplicated legal restatements
- extremely long sentences that conflate many clauses and will distort the pilot

- [ ] **Step 5: Record the frozen sampling rule**

Write a short note in the annotation workspace describing:

- pilot size
- bucket targets
- exclusion rules
- split sizes

Expected result:

- a stable pilot scope before any annotation starts

### Task 2: Produce the Pilot Gold Annotation Set

**Files:**
- Create: `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_train.jsonl`
- Create: `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_val.jsonl`
- Create: `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_test.jsonl`
- Read: `E:\data\SH17\site_safety_monitor\text_ie\candidates\annotation_seed.jsonl`
- Read: `data/annotation_guidelines/manufacturing_text_ie.md`

- [ ] **Step 1: Create a manual annotation worksheet**

Work from `annotation_seed.jsonl` and keep only the selected pilot sentences.

- [ ] **Step 2: Annotate word-level spans**

For each sentence, add:

- `subject_span`
- `predicate`
- `object_span`
- optional `subject_text`
- optional `object_text`

- [ ] **Step 3: Preserve overlap exactly**

Do not simplify sentences that contain:

- one subject with multiple relations
- PPE plus operation in the same sentence
- operation plus hazard in the same sentence

- [ ] **Step 4: Split the pilot**

Create:

- `pilot_train.jsonl`
- `pilot_val.jsonl`
- `pilot_test.jsonl`

The three splits must preserve domain coverage instead of being purely random.

- [ ] **Step 5: Sanity-check file format**

Every record must include:

- `sentence_id`
- `text`
- `tokens`
- `triples`
- provenance metadata

Expected result:

- the first gold dataset exists and is ready for validation

### Task 3: Run Annotation Quality Control Before Training

**Files:**
- Read: `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_train.jsonl`
- Read: `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_val.jsonl`
- Read: `E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_test.jsonl`
- Use: `src/site_safety_monitor/data/text_ie_annotations.py`

- [ ] **Step 1: Validate JSONL format through the loader**

Run:

```powershell
& 'D:\Anaconda\envs\Industrial-Safety-PPE-Detection\python.exe' -c "from site_safety_monitor.data.text_ie_annotations import load_gold_annotations_jsonl; print(len(load_gold_annotations_jsonl(r'E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_train.jsonl')))"
```

Expected:

- the loader succeeds with no span or predicate error

- [ ] **Step 2: Review span consistency**

Spot-check all records for:

- punctuation accidentally included in the span
- object spans that are too broad
- missing actor spans

- [ ] **Step 3: Review predicate consistency**

Check that:

- PPE requirement uses `be_equipped_with`
- task/activity uses `perform_operations`
- explicit consequence uses `occurrence`

- [ ] **Step 4: Count overlap examples**

Before training, confirm the pilot still contains the planned overlap cases.

Expected result:

- no malformed annotations enter BERT training

### Task 4: Prepare the Encoder as a Local Runtime Dependency

**Files:**
- Modify: `configs/site_safety_monitor/text_ie/manufacturing.yaml`
- Read: `src/site_safety_monitor/text_ie/dataset.py`
- Read: `src/site_safety_monitor/text_ie/model.py`

- [ ] **Step 1: Choose the runtime checkpoint policy**

Use one of these, in order:

1. local directory checkpoint
2. pre-downloaded Hugging Face cache
3. direct remote checkpoint only if networking is intentional

- [ ] **Step 2: Lock the encoder path**

Recommended first encoder:

- `bert-base-cased`

But in runtime config, prefer a local resolved path such as:

- `E:\data\SH17\site_safety_monitor\models\bert-base-cased`

- [ ] **Step 3: Verify tokenizer and model load offline**

Before building datasets, verify:

- tokenizer can load
- encoder can load
- no network fetch is required during the run

- [ ] **Step 4: Record the checkpoint source**

Keep a small text note near the model directory describing:

- source model name
- download date
- any tokenizer quirks

Expected result:

- training and inference no longer depend on ad-hoc model downloads

### Task 5: Build the Real BIEO Dataset Files

**Files:**
- Use: `scripts/build_text_ie_dataset.py`
- Create: `E:\data\SH17\site_safety_monitor\text_ie\processed\bieo\pilot_train_encoded.jsonl`
- Create: `E:\data\SH17\site_safety_monitor\text_ie\processed\bieo\pilot_val_encoded.jsonl`
- Create: `E:\data\SH17\site_safety_monitor\text_ie\processed\bieo\pilot_test_encoded.jsonl`

- [ ] **Step 1: Encode the train split**

Run:

```powershell
& 'D:\Anaconda\envs\Industrial-Safety-PPE-Detection\python.exe' scripts\build_text_ie_dataset.py --annotations "E:\data\SH17\site_safety_monitor\text_ie\annotations\pilot_train.jsonl" --output-dir "E:\data\SH17\site_safety_monitor\text_ie\processed\bieo\pilot_train" --encoder-name "E:\data\SH17\site_safety_monitor\models\bert-base-cased" --max-length 256
```

- [ ] **Step 2: Encode the val split**

Run the same command for `pilot_val.jsonl`.

- [ ] **Step 3: Encode the test split**

Run the same command for `pilot_test.jsonl`.

- [ ] **Step 4: Inspect one encoded sample**

Check:

- `input_ids`
- `attention_mask`
- `label_matrix`
- `label_mask`
- `word_ids`
- `subword_labels`

- [ ] **Step 5: Check truncation risk**

If many pilot sentences are close to the max length, either:

- shorten the candidate policy
- or raise `max_length`

Expected result:

- all three splits are converted into BIEO-ready model inputs

### Task 6: Run the First Real Training Cycle

**Files:**
- Modify: `configs/site_safety_monitor/text_ie/manufacturing.yaml`
- Use: `scripts/train_text_ie.py`
- Create: `E:\data\SH17\site_safety_monitor\text_ie\artifacts\pilot_run_001\best_model.pt`
- Create: `E:\data\SH17\site_safety_monitor\text_ie\artifacts\pilot_run_001\best_metrics.json`

- [ ] **Step 1: Point the config at pilot annotations**

Set:

- `train_annotations`
- `val_annotations`
- `output_dir`
- `encoder_name`

- [ ] **Step 2: Lock the first-run hyperparameters**

Recommended first run:

- `max_length: 256`
- `batch_size: 4` or `8`
- `num_epochs: 10`
- `learning_rate: 1e-5`
- `early_stopping_patience: 10`
- `threshold: 0.5`

- [ ] **Step 3: Run training**

Run:

```powershell
& 'D:\Anaconda\envs\Industrial-Safety-PPE-Detection\python.exe' scripts\train_text_ie.py --config "configs\site_safety_monitor\text_ie\manufacturing.yaml"
```

- [ ] **Step 4: Confirm checkpoint outputs**

Expect:

- `best_model.pt`
- `best_metrics.json`

- [ ] **Step 5: Archive the exact run config**

Copy or snapshot the exact YAML next to the artifact folder so the run is reproducible.

Expected result:

- one reproducible baseline text IE model exists

### Task 7: Evaluate the First Model Before Expanding the Dataset

**Files:**
- Read: `E:\data\SH17\site_safety_monitor\text_ie\artifacts\pilot_run_001\best_metrics.json`
- Read: `src/site_safety_monitor/text_ie/evaluate.py`

- [ ] **Step 1: Review triple-level F1**

Primary decision metric:

- triple-level exact-match `F1`

- [ ] **Step 2: Review failure patterns**

Manually inspect errors in these buckets:

- missed subject span
- missed object span
- wrong predicate
- partial overlap decode

- [ ] **Step 3: Review predicate balance**

If one predicate collapses, check whether the pilot annotation mix is skewed.

- [ ] **Step 4: Decide go / no-go**

Only expand the dataset if:

- loader is stable
- training runs end-to-end
- inference returns structurally valid triples

Expected result:

- the team learns whether the problem is annotation quality, data coverage, or modeling

### Task 8: Validate Raw-Sentence Inference

**Files:**
- Use: `scripts/infer_text_ie.py`
- Read: `src/site_safety_monitor/text_ie/infer.py`

- [ ] **Step 1: Run inference on held-out pilot sentences**

Use sentences from `pilot_test.jsonl` first.

- [ ] **Step 2: Run inference on unseen OSHA sentences**

Pick `5-10` candidates from `all_candidates.jsonl` that were not annotated.

- [ ] **Step 3: Compare decoded triples to annotation expectations**

Look for:

- missing PPE object phrase
- operation extracted but hazard dropped
- subject span drifting to `employer` instead of `employee`

- [ ] **Step 4: Check normalization compatibility**

Confirm the decoded triples can cleanly map into the existing baseline triple contract.

Expected result:

- inference is usable on raw regulation text, not just in training-time tests

### Task 9: Promote from Pilot to Full Dataset Only After the Loop Is Stable

**Files:**
- Update later: `E:\data\SH17\site_safety_monitor\text_ie\annotations\train.jsonl`
- Update later: `E:\data\SH17\site_safety_monitor\text_ie\annotations\val.jsonl`
- Update later: `E:\data\SH17\site_safety_monitor\text_ie\annotations\test.jsonl`

- [ ] **Step 1: Expand annotation only after pilot success**

Do not jump directly from `0` to all `157` candidates.

- [ ] **Step 2: Grow in bounded batches**

Recommended expansion:

- pilot `60`
- phase 2 `100`
- full set only if the model still benefits

- [ ] **Step 3: Keep the held-out test set stable**

Do not rewrite or reshuffle test examples casually once the first real evaluation starts.

Expected result:

- the dataset grows with evidence instead of guesswork

---

## Acceptance Criteria

This phase is complete when all of the following are true:

- a manually labeled pilot dataset exists
- the pilot contains meaningful overlap cases
- the encoder loads from a stable local path
- `build_text_ie_dataset.py` succeeds on train/val/test
- `train_text_ie.py` saves a real checkpoint and metric file
- `infer_text_ie.py` returns structurally valid triples from raw text
- the team has reviewed first-run errors before scaling annotation

## Recommended Immediate Next Action

The best next move is not training yet. It is:

1. freeze the `60`-sentence pilot scope
2. manually annotate it carefully
3. run one end-to-end pilot training cycle

That will tell us whether the bottleneck is:

- annotation quality
- sentence selection
- encoder choice
- or decoding behavior
