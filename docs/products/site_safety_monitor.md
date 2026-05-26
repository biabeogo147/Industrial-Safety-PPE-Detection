# Site Safety Monitor

`Site Safety Monitor` is the first productized implementation of the 2023 multimodal hazard-identification workflow we studied.

## Scope

- Schema-driven text information extraction for safety regulations
- Scene parsing outputs represented as visual triples
- Triple-based PPE compliance checking and hazard inference
- The current logic follows the publicly documented parts of the 2023 paper as closely as possible.

## Paper-Faithful Notes

- Text IE uses the paper's documented `schema-driven + BIEO` direction.
- The BIEO tag space follows the paper's `4|N| + 2` rule for `B/E` subject-object tags plus `I/O`.
- Text IE defaults are aligned to the paper where specified: `BERT-Base Chinese`, `lr=1e-5`, and early stopping patience `10`.
- Scene-graph defaults are aligned to the paper where specified: `Mask R-CNN` family, `ResNeXt-101-FPN` result backbone, `fastText` semantic prior source, `lr=1e-3`, and `momentum SGD`.
- Safety checking follows the paper's subset logic:
  - no detected worker -> `N/A`
  - textual PPE triple set is a subset of visual PPE triple set -> `Yes`
  - partial or empty match against required PPE triples -> `No`
- Hazard inference is activated from `operation -> occurrence -> injury` text triples when PPE compliance fails.
- The paper mentions `8` text relation types, but the publication does not list all of them explicitly. The code therefore implements the full publicly recoverable schema and keeps the schema extension point explicit.

## Current Layout

- `src/site_safety_monitor/core`: shared contracts and normalization
- `src/site_safety_monitor/text_ie`: schema-driven text IE helpers
- `src/site_safety_monitor/scene_graph`: relation priors, decoding, and evaluation helpers
- `src/site_safety_monitor/safety`: compliance and hazard logic
- `src/site_safety_monitor/pipelines`: end-to-end runner

## Verification

- `D:\Anaconda\envs\Industrial-Safety-PPE-Detection\python.exe -m pytest tests\unit tests\integration -v`
  - Result: `7 passed`
- `D:\Anaconda\envs\Industrial-Safety-PPE-Detection\python.exe scripts\run_site_safety_monitor.py --regulation tests\fixtures\regulations\work_at_height.json --scene tests\fixtures\scenes\work_at_height.json`
  - Result: `No` compliance with hazard `head injury from falls`
