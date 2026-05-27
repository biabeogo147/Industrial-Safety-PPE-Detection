# Site Safety Monitor Playbook

## 1. Chuẩn Bị

### 1.1 Vào root repo

```bash
cd /path/to/Industrial-Safety-PPE-Detection
```

### 1.2 Khai báo biến dùng chung

Các lệnh bên dưới giả định bạn dùng các biến này:

```bash
export PROJECT_ROOT="$(pwd)"
export DATA_ROOT="${DATA_ROOT:-./runtime_data/site_safety_monitor}"
export MODEL_ROOT="${MODEL_ROOT:-$DATA_ROOT/models}"
```

Ý nghĩa:

- `PROJECT_ROOT`: root của repo
- `DATA_ROOT`: nơi chứa corpus, annotation, artifact ngoài source code
- `MODEL_ROOT`: nơi chứa local checkpoint cho BERT

Nếu muốn đặt data ở chỗ khác, chỉ cần đổi `DATA_ROOT` trước khi chạy.

### 1.3 Tạo thư mục runtime

```bash
mkdir -p "$DATA_ROOT"
mkdir -p "$MODEL_ROOT"
```

## 2. Quick Start

### 2.1 Chạy test toàn repo

```bash
python -m pytest tests/unit tests/integration -v
```

Kỳ vọng hiện tại:

- `31 passed`

### 2.2 Chạy smoke test baseline safety checker

```bash
python scripts/run_site_safety_monitor.py \
  --regulation tests/fixtures/regulations/work_at_height.json \
  --scene tests/fixtures/scenes/work_at_height.json
```

Kỳ vọng output:

- `compliance = No`
- thiếu `hard hat`
- hazard `head injury from falls`

## 3. Workflow Chuẩn Cho Text IE

Đây là thứ tự nên chạy nếu muốn đi từ raw OSHA text tới train/infer BERT.

### Bước 1: Crawl OSHA corpus

Script:

- `scripts/crawl_osha_corpus.py`

Mục đích:

- tải HTML OSHA
- parse thành `sections.jsonl`
- split thành `sentences.jsonl`

Lệnh:

```bash
python scripts/crawl_osha_corpus.py \
  --output-root "$DATA_ROOT"
```

Options hay dùng:

- `--source-manifest`: đổi manifest nguồn crawl
- `--limit`: crawl ít nguồn để test nhanh
- `--refresh`: ép tải lại thay vì dùng file đã có

Output:

- `"$DATA_ROOT"/text_corpus/raw/osha_1910/*.html`
- `"$DATA_ROOT"/text_corpus/interim/sections.jsonl`
- `"$DATA_ROOT"/text_corpus/interim/sentences.jsonl`

### Bước 2: Build BERT-ready corpus

Script:

- `scripts/prepare_bert_corpus.py`

Mục đích:

- đọc `sentences.jsonl`
- lọc câu nhiễu
- normalize text
- export corpus sentence-level sẵn cho BERT tokenization

Lệnh:

```bash
python scripts/prepare_bert_corpus.py \
  --output-root "$DATA_ROOT"
```

Option hay dùng:

- `--rules-path`: thay rule filtering

Output:

- `"$DATA_ROOT"/text_corpus/processed/bert_input/all.jsonl`
- `"$DATA_ROOT"/text_corpus/processed/bert_input/train.jsonl`
- `"$DATA_ROOT"/text_corpus/processed/bert_input/val.jsonl`
- `"$DATA_ROOT"/text_corpus/processed/bert_input/test.jsonl`

### Bước 3: Export annotation candidates

Script:

- `scripts/prepare_text_ie_candidates.py`

Mục đích:

- score candidate sentences
- export candidate pool
- export blank annotation seed với `triples: []`

Lệnh:

```bash
mkdir -p "$DATA_ROOT/text_ie/candidates"

python scripts/prepare_text_ie_candidates.py \
  --input-jsonl "$DATA_ROOT/text_corpus/processed/bert_input/all.jsonl" \
  --output-jsonl "$DATA_ROOT/text_ie/candidates/all_candidates.jsonl" \
  --annotation-template-jsonl "$DATA_ROOT/text_ie/candidates/annotation_seed.jsonl"
```

Options hay dùng:

- `--top-k`
- `--domain-tag`

Output:

- `"$DATA_ROOT"/text_ie/candidates/all_candidates.jsonl`
- `"$DATA_ROOT"/text_ie/candidates/annotation_seed.jsonl`

### Bước 4: Freeze pilot annotation set

Script:

- `scripts/prepare_text_ie_pilot.py`

Mục đích:

- chọn một pilot set sạch từ candidate pool
- tạo split `train/val/test` ở mức seed
- sinh `pilot_scope.md`

Lệnh ví dụ:

```bash
mkdir -p "$DATA_ROOT/text_ie/annotations/pilot"

python scripts/prepare_text_ie_pilot.py \
  --input-jsonl "$DATA_ROOT/text_ie/candidates/all_candidates.jsonl" \
  --output-dir "$DATA_ROOT/text_ie/annotations/pilot" \
  --total-size 48 \
  --train-size 36 \
  --val-size 6 \
  --test-size 6 \
  --max-tokens 56
```

Output:

- `"$DATA_ROOT"/text_ie/annotations/pilot/pilot_selected_candidates.jsonl`
- `"$DATA_ROOT"/text_ie/annotations/pilot/pilot_train_seed.jsonl`
- `"$DATA_ROOT"/text_ie/annotations/pilot/pilot_val_seed.jsonl`
- `"$DATA_ROOT"/text_ie/annotations/pilot/pilot_test_seed.jsonl`
- `"$DATA_ROOT"/text_ie/annotations/pilot/pilot_scope.md`

Lưu ý:

- đây mới là seed
- chưa phải gold annotation cuối cùng

### Bước 5: Annotate gold text IE dataset

Đây là bước thủ công.

Đầu vào:

- `"$DATA_ROOT"/text_ie/annotations/pilot/pilot_*_seed.jsonl`

Guideline:

- `data/annotation_guidelines/manufacturing_text_ie.md`

Mỗi record cần điền:

- `triples`
- mỗi triple cần:
  - `subject_span`
  - `predicate`
  - `object_span`
  - nên có `subject_text`
  - nên có `object_text`

Khi chốt pilot, nên promote thành:

- `"$DATA_ROOT"/text_ie/annotations/train.jsonl`
- `"$DATA_ROOT"/text_ie/annotations/val.jsonl`
- `"$DATA_ROOT"/text_ie/annotations/test.jsonl`

## 4. Cấu Hình Relative Cho BERT

Script `train_text_ie.py` đọc YAML config, nên để portable bạn nên tạo một config local dùng relative path.

Ví dụ:

```bash
mkdir -p configs/site_safety_monitor/text_ie
```

Tạo file `configs/site_safety_monitor/text_ie/local.yaml`:

```yaml
encoder_name: ./runtime_data/site_safety_monitor/models/bert-base-cased
predicates:
  - be_equipped_with
  - perform_operations
  - occurrence
train_annotations: ./runtime_data/site_safety_monitor/text_ie/annotations/train.jsonl
val_annotations: ./runtime_data/site_safety_monitor/text_ie/annotations/val.jsonl
output_dir: ./runtime_data/site_safety_monitor/text_ie/artifacts/pilot_run_001
max_length: 256
batch_size: 4
num_epochs: 6
learning_rate: 1.0e-5
validation_metric: f1
early_stopping_patience: 10
threshold: 0.5
dropout: 0.1
```

Lưu ý:

- file này giả định bạn chạy lệnh từ root repo
- nếu bạn đổi `DATA_ROOT`, hãy sửa YAML tương ứng hoặc generate YAML của riêng bạn

## 5. Local BERT Checkpoint

Repo không tự động tải model. Bạn cần có local checkpoint trước khi train/infer.

Ví dụ, nếu bạn đã có `bert-base-cased` ở nơi khác, hãy copy vào:

```bash
mkdir -p "$MODEL_ROOT/bert-base-cased"
```

Sau đó config nên trỏ tới:

```text
./runtime_data/site_safety_monitor/models/bert-base-cased
```

Thư mục model tối thiểu nên có tokenizer/model files của Hugging Face.

## 6. Build BIEO-ready Dataset

Script:

- `scripts/build_text_ie_dataset.py`

Mục đích:

- đọc gold annotation JSONL
- tokenize bằng BERT tokenizer
- convert triples thành BIEO supervision
- export `encoded.jsonl`

### Train split

```bash
python scripts/build_text_ie_dataset.py \
  --annotations "$DATA_ROOT/text_ie/annotations/train.jsonl" \
  --output-dir "$DATA_ROOT/text_ie/processed/bieo/pilot_train" \
  --encoder-name "$MODEL_ROOT/bert-base-cased" \
  --max-length 256
```

### Val split

```bash
python scripts/build_text_ie_dataset.py \
  --annotations "$DATA_ROOT/text_ie/annotations/val.jsonl" \
  --output-dir "$DATA_ROOT/text_ie/processed/bieo/pilot_val" \
  --encoder-name "$MODEL_ROOT/bert-base-cased" \
  --max-length 256
```

### Test split

```bash
python scripts/build_text_ie_dataset.py \
  --annotations "$DATA_ROOT/text_ie/annotations/test.jsonl" \
  --output-dir "$DATA_ROOT/text_ie/processed/bieo/pilot_test" \
  --encoder-name "$MODEL_ROOT/bert-base-cased" \
  --max-length 256
```

Output:

- `encoded.jsonl` trong từng thư mục output

Field quan trọng:

- `input_ids`
- `attention_mask`
- `label_matrix`
- `label_mask`
- `word_ids`
- `tokens`
- `subword_labels`

## 7. Train BERT Text IE

Script:

- `scripts/train_text_ie.py`

Khuyến nghị:

- dùng config local riêng, ví dụ `configs/site_safety_monitor/text_ie/local.yaml`

Lệnh:

```bash
python scripts/train_text_ie.py \
  --config configs/site_safety_monitor/text_ie/local.yaml
```

Output:

- `best_model.pt`
- `best_metrics.json`

Ví dụ nếu dùng config ở trên:

- `./runtime_data/site_safety_monitor/text_ie/artifacts/pilot_run_001/best_model.pt`
- `./runtime_data/site_safety_monitor/text_ie/artifacts/pilot_run_001/best_metrics.json`

Lưu ý hiện tại:

- repo đã chạy được vòng train/infer thật
- pilot đầu tiên vẫn có thể cho `F1 = 0.0`
- nếu vậy, ưu tiên sửa dữ liệu pilot trước khi nghĩ tới tuning model

## 8. Run Raw-sentence Inference

Script:

- `scripts/infer_text_ie.py`

Lệnh:

```bash
python scripts/infer_text_ie.py \
  --config configs/site_safety_monitor/text_ie/local.yaml \
  --checkpoint "$DATA_ROOT/text_ie/artifacts/pilot_run_001/best_model.pt" \
  --text "Employees exposed to the hazards created by welding, cutting, or brazing operations shall be protected by personal protective equipment."
```

Output:

- JSON list triples chuẩn hóa

Nếu output là `[]`, điều đó thường có nghĩa:

- load model OK
- forward pass OK
- decoder OK
- nhưng pilot hiện tại chưa đủ tốt để model học được

## 9. Evaluate Text IE Từ JSON Đơn Giản

Script:

- `scripts/eval_text_ie.py`

Format input:

```json
[
  ["workers", "be_equipped_with", "hard hat"],
  ["workers", "perform_operations", "work at height"]
]
```

Lệnh:

```bash
python scripts/eval_text_ie.py \
  --predicted ./predicted.json \
  --gold ./gold.json
```

## 10. Chạy Safety Baseline

Script:

- `scripts/run_site_safety_monitor.py`

Mục đích:

- nhận regulation triples JSON
- nhận scene triples JSON
- chạy logic compliance + hazard inference

Lệnh:

```bash
python scripts/run_site_safety_monitor.py \
  --regulation tests/fixtures/regulations/work_at_height.json \
  --scene tests/fixtures/scenes/work_at_height.json
```

Input regulation mẫu:

```json
{
  "triples": [
    {"subject": "workers", "predicate": "be_equipped_with", "object": "hard hat"},
    {"subject": "workers", "predicate": "perform_operations", "object": "work at height"},
    {"subject": "work at height", "predicate": "occurrence", "object": "head injury from falls"}
  ]
}
```

Input scene mẫu:

```json
{
  "triples": [
    {
      "subject_id": "worker_0",
      "subject_label": "worker",
      "predicate": "wear",
      "object_id": "glove_0",
      "object_label": "hand protection"
    }
  ]
}
```

Output mẫu:

```json
{
  "worker_id": "worker_0",
  "compliance": "No",
  "missing_requirements": ["hard hat"],
  "hazards": ["head injury from falls"]
}
```

Lưu ý:

- script này không nhận raw text
- nếu muốn đi từ raw regulation text, bạn phải chạy text IE trước

## 11. Scene Graph Scripts

Phần này hiện chưa phải pipeline train hoàn chỉnh.

### `scripts/train_scene_graph.py`

Hiện tại thực chất chỉ summarize dataset:

```bash
python scripts/train_scene_graph.py \
  --dataset ./path/to/scene_graph_dataset.json
```

### `scripts/eval_scene_graph.py`

Tính `recall@k`:

```bash
python scripts/eval_scene_graph.py \
  --retrieved ./path/to/retrieved.json \
  --relevant ./path/to/relevant.json \
  --k 20
```

## 12. Module Map

### `src/site_safety_monitor/data`

- `osha_corpus.py`: crawl và parse OSHA HTML
- `bert_corpus.py`: lọc và export BERT-ready corpus
- `text_ie_candidates.py`: score/export candidate pool
- `text_ie_pilot.py`: freeze pilot annotation set
- `text_ie_annotations.py`: gold annotation IO + validation

### `src/site_safety_monitor/text_ie`

- `codec.py`: BIEO encode/decode
- `alignment.py`: word-to-subword alignment
- `dataset.py`: build training records cho BERT
- `model.py`: BERT token classifier
- `train.py`: training loop + checkpoint
- `infer.py`: raw sentence -> triples
- `evaluate.py`: evaluate predicted triples
- `predict.py`: convert decoded triples sang `TextTriple`

### `src/site_safety_monitor/safety`

- `checker.py`: PPE compliance logic
- `hazards.py`: hazard inference helpers

### `src/site_safety_monitor/pipelines`

- `run_monitor.py`: end-to-end runner cho regulation + scene

## 13. Troubleshooting

### Tokenizer hoặc model không load được

Kiểm tra:

- `encoder_name` trong YAML config
- local checkpoint có đủ file tokenizer/model không
- bạn có đang chạy từ root repo không nếu config dùng relative path

### `F1 = 0.0`

Điều này thường là vấn đề dữ liệu pilot, không phải lỗi script.

Nguyên nhân hay gặp:

- pilot quá nhỏ
- positive sentences quá ít
- `val` quá mỏng
- overlap cases chưa đủ

### Inference trả `[]`

Điều này thường có nghĩa:

- model load OK
- decoder load OK
- nhưng model chưa học được pattern đủ tốt

### `run_site_safety_monitor.py` không nhận raw text

Đây là chủ ý.

Script này chỉ nhận:

- regulation triples
- scene triples

Nếu muốn đi từ raw text, bạn phải chạy qua `infer_text_ie.py` trước.

## 14. Daily Commands

### Verify repo

```bash
python -m pytest tests/unit tests/integration -v
```

### Rebuild BERT corpus

```bash
python scripts/prepare_bert_corpus.py \
  --output-root "$DATA_ROOT"
```

### Rebuild annotation candidates

```bash
python scripts/prepare_text_ie_candidates.py \
  --input-jsonl "$DATA_ROOT/text_corpus/processed/bert_input/all.jsonl" \
  --output-jsonl "$DATA_ROOT/text_ie/candidates/all_candidates.jsonl" \
  --annotation-template-jsonl "$DATA_ROOT/text_ie/candidates/annotation_seed.jsonl"
```

### Freeze pilot

```bash
python scripts/prepare_text_ie_pilot.py \
  --input-jsonl "$DATA_ROOT/text_ie/candidates/all_candidates.jsonl" \
  --output-dir "$DATA_ROOT/text_ie/annotations/pilot" \
  --total-size 48 \
  --train-size 36 \
  --val-size 6 \
  --test-size 6 \
  --max-tokens 56
```

### Train BERT

```bash
python scripts/train_text_ie.py \
  --config configs/site_safety_monitor/text_ie/local.yaml
```

### Run safety checker

```bash
python scripts/run_site_safety_monitor.py \
  --regulation tests/fixtures/regulations/work_at_height.json \
  --scene tests/fixtures/scenes/work_at_height.json
```
