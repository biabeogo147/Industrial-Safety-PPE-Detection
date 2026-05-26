# Site Safety Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first production-style release of `Site Safety Monitor`, keeping the 2023 paper's `text IE -> visual triples -> safety checking` flow before any improvements.

**Architecture:** The codebase will be split into three primary modules that mirror the paper: `text_ie`, `scene_graph`, and `safety`. Shared contracts live in `core`, dataset-specific loading lives in `data`, and orchestration plus experiment entry points live in `pipelines` and `scripts`.

**Tech Stack:** Python, PyTorch, Hugging Face Transformers, Detectron2-compatible detector adapter, YAML configs, and Pytest.

---

### Task 1: Scaffold the Project and Shared Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `src/site_safety_monitor/__init__.py`
- Create: `src/site_safety_monitor/core/triples.py`
- Create: `src/site_safety_monitor/core/schema.py`
- Create: `src/site_safety_monitor/core/normalize.py`
- Create: `tests/unit/core/test_triples.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.core.triples import TextTriple, VisualTriple


def test_text_and_visual_triples_normalize_labels():
    text_triple = TextTriple(
        subject="Workers",
        predicate="be_equipped_with",
        object="Hard Hats",
    )
    visual_triple = VisualTriple(
        subject_id="worker_0",
        subject_label="Worker",
        predicate="Wear",
        object_id="hat_0",
        object_label="Hard Hat",
    )

    assert text_triple.normalized_subject == "workers"
    assert text_triple.normalized_object == "hard hats"
    assert visual_triple.normalized_subject_label == "worker"
    assert visual_triple.normalized_predicate == "wear"
    assert visual_triple.normalized_object_label == "hard hat"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/core/test_triples.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing class errors.

- [ ] **Step 3: Write the minimal implementation**

```python
from dataclasses import dataclass


def normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", " ")


@dataclass(frozen=True)
class TextTriple:
    subject: str
    predicate: str
    object: str

    @property
    def normalized_subject(self) -> str:
        return normalize_label(self.subject)

    @property
    def normalized_object(self) -> str:
        return normalize_label(self.object)


@dataclass(frozen=True)
class VisualTriple:
    subject_id: str
    subject_label: str
    predicate: str
    object_id: str
    object_label: str

    @property
    def normalized_subject_label(self) -> str:
        return normalize_label(self.subject_label)

    @property
    def normalized_predicate(self) -> str:
        return normalize_label(self.predicate)

    @property
    def normalized_object_label(self) -> str:
        return normalize_label(self.object_label)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/core/test_triples.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/site_safety_monitor/core tests/unit/core/test_triples.py
git commit -m "feat: scaffold site safety monitor core contracts"
```

### Task 2: Implement Schema-Driven BIEO Encoding for Text IE

**Files:**
- Create: `src/site_safety_monitor/text_ie/schema.py`
- Create: `src/site_safety_monitor/text_ie/codec.py`
- Create: `src/site_safety_monitor/text_ie/dataset.py`
- Create: `tests/unit/text_ie/test_bieo_codec.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.text_ie.codec import BIEOCodec


def test_codec_round_trips_overlapping_triples():
    tokens = ["Workers", "should", "wear", "face", "protection", "during", "welding", "operations"]
    triples = [
        {
            "subject_span": (0, 0),
            "predicate": "be_equipped_with",
            "object_span": (3, 4),
        },
        {
            "subject_span": (0, 0),
            "predicate": "perform_operations",
            "object_span": (6, 7),
        },
    ]

    codec = BIEOCodec(predicates=["be_equipped_with", "perform_operations"])
    encoded = codec.encode(tokens=tokens, triples=triples)
    decoded = codec.decode(tokens=tokens, tags=encoded)

    assert decoded == triples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/text_ie/test_bieo_codec.py -v`
Expected: FAIL because `BIEOCodec` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
class BIEOCodec:
    def __init__(self, predicates: list[str]) -> None:
        self.predicates = predicates

    def encode(self, tokens: list[str], triples: list[dict]) -> list[list[str]]:
        matrix = [["O"] for _ in tokens]
        for triple in triples:
            predicate = triple["predicate"]
            subj_start, subj_end = triple["subject_span"]
            obj_start, obj_end = triple["object_span"]
            matrix[subj_start].append(f"SUBJ-{predicate}-B")
            matrix[subj_end].append(f"SUBJ-{predicate}-E")
            for index in range(subj_start + 1, subj_end):
                matrix[index].append("I")
            matrix[obj_start].append(f"OBJ-{predicate}-B")
            matrix[obj_end].append(f"OBJ-{predicate}-E")
            for index in range(obj_start + 1, obj_end):
                matrix[index].append("I")
        return matrix

    def decode(self, tokens: list[str], tags: list[list[str]]) -> list[dict]:
        subject_starts: dict[str, int] = {}
        object_starts: dict[str, int] = {}
        triples: list[dict] = []
        for index, token_tags in enumerate(tags):
            for tag in token_tags:
                if tag.startswith("SUBJ-") and tag.endswith("-B"):
                    predicate = tag[len("SUBJ-") : -len("-B")]
                    subject_starts[predicate] = index
                if tag.startswith("SUBJ-") and tag.endswith("-E"):
                    predicate = tag[len("SUBJ-") : -len("-E")]
                    subject_span = (subject_starts[predicate], index)
                    for later_index, later_tags in enumerate(tags):
                        if later_index < index:
                            continue
                        for later_tag in later_tags:
                            if later_tag == f"OBJ-{predicate}-B":
                                object_starts[predicate] = later_index
                            if later_tag == f"OBJ-{predicate}-E" and predicate in object_starts:
                                triples.append(
                                    {
                                        "subject_span": subject_span,
                                        "predicate": predicate,
                                        "object_span": (object_starts[predicate], later_index),
                                    }
                                )
                                object_starts.pop(predicate, None)
                                break
                        else:
                            continue
                        break
        return triples
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/text_ie/test_bieo_codec.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/text_ie tests/unit/text_ie/test_bieo_codec.py
git commit -m "feat: add schema-driven BIEO codec"
```

### Task 3: Build the Text IE Model, Metrics, and CLI

**Files:**
- Create: `src/site_safety_monitor/text_ie/model.py`
- Create: `src/site_safety_monitor/text_ie/metrics.py`
- Create: `src/site_safety_monitor/text_ie/train.py`
- Create: `src/site_safety_monitor/text_ie/predict.py`
- Create: `scripts/train_text_ie.py`
- Create: `scripts/eval_text_ie.py`
- Create: `tests/unit/text_ie/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.text_ie.metrics import triple_f1


def test_triple_f1_requires_exact_subject_predicate_object_match():
    predicted = {
        ("workers", "be_equipped_with", "hard hats"),
        ("workers", "perform_operations", "work at height"),
    }
    gold = {
        ("workers", "be_equipped_with", "hard hats"),
        ("workers", "perform_operations", "welding operations"),
    }

    precision, recall, f1 = triple_f1(predicted=predicted, gold=gold)

    assert round(precision, 3) == 0.5
    assert round(recall, 3) == 0.5
    assert round(f1, 3) == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/text_ie/test_metrics.py -v`
Expected: FAIL because `triple_f1` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
def triple_f1(predicted: set[tuple[str, str, str]], gold: set[tuple[str, str, str]]) -> tuple[float, float, float]:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1
```

- [ ] **Step 4: Expand to the model and runner**

```python
from transformers import AutoModel
from torch import nn


class BertBIEOTagger(nn.Module):
    def __init__(self, encoder_name: str, num_labels: int) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state)
```

Run: `python -m pytest tests/unit/text_ie/test_metrics.py -v`
Expected: PASS for the metric test.

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/text_ie scripts/train_text_ie.py scripts/eval_text_ie.py tests/unit/text_ie/test_metrics.py
git commit -m "feat: add text IE model and evaluation"
```

### Task 4: Implement Scene Graph Dataset Support and Semantic Prior

**Files:**
- Create: `src/site_safety_monitor/scene_graph/dataset.py`
- Create: `src/site_safety_monitor/scene_graph/semantic_prior.py`
- Create: `tests/unit/scene_graph/test_semantic_prior.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.scene_graph.semantic_prior import SemanticPrior


def test_semantic_prior_learns_predicate_distribution_per_object_pair():
    relations = [
        ("worker", "wear", "hard_hat"),
        ("worker", "wear", "hard_hat"),
        ("worker", "hold", "welding_tool"),
    ]

    prior = SemanticPrior()
    prior.fit(relations)

    hard_hat_scores = prior.logits_for(subject_label="worker", object_label="hard_hat")
    tool_scores = prior.logits_for(subject_label="worker", object_label="welding_tool")

    assert hard_hat_scores["wear"] > hard_hat_scores["hold"]
    assert tool_scores["hold"] > tool_scores["wear"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/scene_graph/test_semantic_prior.py -v`
Expected: FAIL because `SemanticPrior` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from collections import Counter, defaultdict
from math import log


class SemanticPrior:
    def __init__(self) -> None:
        self.counts = defaultdict(Counter)

    def fit(self, relations: list[tuple[str, str, str]]) -> None:
        for subject_label, predicate, object_label in relations:
            self.counts[(subject_label, object_label)][predicate] += 1

    def logits_for(self, subject_label: str, object_label: str) -> dict[str, float]:
        counter = self.counts[(subject_label, object_label)]
        total = sum(counter.values()) or 1
        return {predicate: log(count / total) for predicate, count in counter.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/scene_graph/test_semantic_prior.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/scene_graph tests/unit/scene_graph/test_semantic_prior.py
git commit -m "feat: add scene graph semantic prior"
```

### Task 5: Build the Relation Fusion Head and Triple Decoder

**Files:**
- Create: `src/site_safety_monitor/scene_graph/model.py`
- Create: `src/site_safety_monitor/scene_graph/decode.py`
- Create: `src/site_safety_monitor/scene_graph/train.py`
- Create: `src/site_safety_monitor/scene_graph/evaluate.py`
- Create: `scripts/train_scene_graph.py`
- Create: `scripts/eval_scene_graph.py`
- Create: `tests/unit/scene_graph/test_relation_head.py`

- [ ] **Step 1: Write the failing test**

```python
import torch

from site_safety_monitor.scene_graph.model import PredicateFusionHead


def test_predicate_fusion_head_adds_semantic_and_visual_logits():
    head = PredicateFusionHead(input_dim=12, num_predicates=3)
    subject = torch.ones(1, 4)
    relation = torch.ones(1, 4)
    object_ = torch.ones(1, 4)
    semantic_logits = torch.tensor([[0.1, 0.3, 0.2]])

    logits = head(subject, relation, object_, semantic_logits)

    assert logits.shape == (1, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/scene_graph/test_relation_head.py -v`
Expected: FAIL because `PredicateFusionHead` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
import torch
from torch import nn


class PredicateFusionHead(nn.Module):
    def __init__(self, input_dim: int, num_predicates: int) -> None:
        super().__init__()
        self.visual_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, num_predicates),
        )

    def forward(self, subject_features, relation_features, object_features, semantic_logits):
        visual_inputs = torch.cat([subject_features, relation_features, object_features], dim=-1)
        visual_logits = self.visual_mlp(visual_inputs)
        return visual_logits + semantic_logits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/scene_graph/test_relation_head.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/scene_graph scripts/train_scene_graph.py scripts/eval_scene_graph.py tests/unit/scene_graph/test_relation_head.py
git commit -m "feat: add predicate fusion head for scene graph parsing"
```

### Task 6: Implement Paper-Style Safety Checking

**Files:**
- Create: `src/site_safety_monitor/safety/checker.py`
- Create: `src/site_safety_monitor/safety/hazards.py`
- Create: `tests/unit/safety/test_checker.py`

- [ ] **Step 1: Write the failing test**

```python
from site_safety_monitor.core.triples import TextTriple, VisualTriple
from site_safety_monitor.safety.checker import evaluate_worker


def test_missing_required_ppe_returns_no_and_activates_hazard():
    text_triples = [
        TextTriple("workers", "be_equipped_with", "hard hat"),
        TextTriple("workers", "perform_operations", "work at height"),
        TextTriple("work at height", "occurrence", "head injury from falls"),
    ]
    visual_triples = [
        VisualTriple("worker_0", "worker", "wear", "glove_0", "hand protection"),
    ]

    result = evaluate_worker(worker_id="worker_0", text_triples=text_triples, visual_triples=visual_triples)

    assert result.compliance == "No"
    assert result.missing_requirements == ["hard hat"]
    assert result.hazards == ["head injury from falls"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/safety/test_checker.py -v`
Expected: FAIL because `evaluate_worker` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
from dataclasses import dataclass


@dataclass
class ComplianceResult:
    worker_id: str
    compliance: str
    missing_requirements: list[str]
    hazards: list[str]


def evaluate_worker(worker_id, text_triples, visual_triples):
    required_ppe = {
        triple.normalized_object
        for triple in text_triples
        if triple.predicate == "be_equipped_with"
    }
    observed_ppe = {
        triple.normalized_object_label
        for triple in visual_triples
        if triple.subject_id == worker_id and triple.normalized_predicate == "wear"
    }
    if not visual_triples:
        return ComplianceResult(worker_id, "N/A", [], [])
    missing = sorted(required_ppe - observed_ppe)
    if not missing:
        return ComplianceResult(worker_id, "Yes", [], [])
    hazards = [
        triple.normalized_object
        for triple in text_triples
        if triple.predicate == "occurrence"
    ]
    return ComplianceResult(worker_id, "No", missing, hazards)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/safety/test_checker.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/safety tests/unit/safety/test_checker.py
git commit -m "feat: implement paper-style PPE compliance logic"
```

### Task 7: Add End-to-End Pipeline, Fixtures, and Reproducibility Configs

**Files:**
- Create: `src/site_safety_monitor/pipelines/run_monitor.py`
- Create: `scripts/run_site_safety_monitor.py`
- Create: `configs/site_safety_monitor/text_ie/default.yaml`
- Create: `configs/site_safety_monitor/scene_graph/default.yaml`
- Create: `configs/site_safety_monitor/safety/default.yaml`
- Create: `tests/fixtures/regulations/work_at_height.json`
- Create: `tests/fixtures/scenes/work_at_height.json`
- Create: `tests/integration/test_end_to_end.py`
- Create: `docs/products/site_safety_monitor.md`

- [ ] **Step 1: Write the failing integration test**

```python
from site_safety_monitor.pipelines.run_monitor import run_case


def test_end_to_end_work_at_height_case_returns_no_and_head_injury_hazard():
    result = run_case(
        regulation_path="tests/fixtures/regulations/work_at_height.json",
        scene_path="tests/fixtures/scenes/work_at_height.json",
    )

    assert result["compliance"] == "No"
    assert result["hazards"] == ["head injury from falls"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/integration/test_end_to_end.py -v`
Expected: FAIL because `run_case` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
import json

from site_safety_monitor.core.triples import TextTriple, VisualTriple
from site_safety_monitor.safety.checker import evaluate_worker


def run_case(regulation_path: str, scene_path: str) -> dict:
    with open(regulation_path, "r", encoding="utf-8") as handle:
        regulation_payload = json.load(handle)
    with open(scene_path, "r", encoding="utf-8") as handle:
        scene_payload = json.load(handle)

    text_triples = [TextTriple(**triple) for triple in regulation_payload["triples"]]
    visual_triples = [VisualTriple(**triple) for triple in scene_payload["triples"]]
    result = evaluate_worker(worker_id="worker_0", text_triples=text_triples, visual_triples=visual_triples)

    return {
        "worker_id": result.worker_id,
        "compliance": result.compliance,
        "missing_requirements": result.missing_requirements,
        "hazards": result.hazards,
    }
```

- [ ] **Step 4: Run the integration test**

Run: `python -m pytest tests/integration/test_end_to_end.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/site_safety_monitor/pipelines scripts/run_site_safety_monitor.py configs/site_safety_monitor tests/fixtures tests/integration/test_end_to_end.py docs/products/site_safety_monitor.md
git commit -m "feat: add end-to-end site safety monitor runner"
```

### Task 8: Verify the Baseline Matches the Intended Paper Scope

**Files:**
- Modify: `docs/products/site_safety_monitor.md`

- [ ] **Step 1: Run the unit test suite**

Run: `python -m pytest tests/unit -v`
Expected: PASS

- [ ] **Step 2: Run the integration suite**

Run: `python -m pytest tests/integration -v`
Expected: PASS

- [ ] **Step 3: Run the product smoke command**

Run: `python scripts/run_site_safety_monitor.py --regulation tests/fixtures/regulations/work_at_height.json --scene tests/fixtures/scenes/work_at_height.json`
Expected: JSON output with `"compliance": "No"` and `"hazards": ["head injury from falls"]`

- [ ] **Step 4: Update the baseline doc with verification notes**

```markdown
## Verification

- Unit tests pass.
- Integration tests pass.
- The `work at height` smoke case reproduces the paper's missing-hard-hat hazard activation flow.
- The codebase remains limited to the 2023 paper's text IE, scene graph, and safety checking stages.
```

- [ ] **Step 5: Commit**

```bash
git add docs/products/site_safety_monitor.md
git commit -m "docs: record site safety monitor verification"
```
