from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from site_safety_monitor.data.text_ie_annotations import (
    GoldSentenceAnnotation,
    GoldTripleAnnotation,
    write_gold_annotations_jsonl,
)
from site_safety_monitor.data.text_ie_candidates import score_candidate_sentence
from site_safety_monitor.text_ie.dataset import build_label_vocabulary
from site_safety_monitor.text_ie.infer import infer_text_triples
from site_safety_monitor.text_ie.train import build_training_config, load_training_config, train_text_ie_model


def test_score_candidate_sentence_prioritizes_schema_relevant_text():
    sentence = "Employees exposed to flying particles shall use eye protection."

    score = score_candidate_sentence(sentence)

    assert score > 0


def test_build_training_config_uses_paper_like_defaults():
    config = build_training_config()

    assert config.learning_rate == 1e-5
    assert config.early_stopping_patience == 10


def test_train_and_infer_with_local_tiny_bert_checkpoint():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    fixtures_root = Path(__file__).resolve().parents[1] / "fixtures"
    with tempfile.TemporaryDirectory(dir=fixtures_root) as temp_dir:
        tmp_path = Path(temp_dir)
        encoder_dir = _create_local_tiny_bert(tmp_path / "tiny_bert")
        train_path, val_path = _write_gold_splits(tmp_path / "annotations")
        output_dir = tmp_path / "artifacts"
        config_path = tmp_path / "training.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "encoder_name": str(encoder_dir),
                    "predicates": [
                        "be_equipped_with",
                        "perform_operations",
                        "occurrence",
                    ],
                    "train_annotations": str(train_path),
                    "val_annotations": str(val_path),
                    "output_dir": str(output_dir),
                    "max_length": 32,
                    "batch_size": 1,
                    "num_epochs": 1,
                    "learning_rate": 1.0e-5,
                    "validation_metric": "f1",
                    "early_stopping_patience": 2,
                    "threshold": 0.5,
                    "dropout": 0.1,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        config = load_training_config(config_path)
        result = train_text_ie_model(config)

        assert Path(result.best_checkpoint_path).exists()
        assert (output_dir / "best_metrics.json").exists()

        tag_space, _ = build_label_vocabulary(config.predicates)
        triples = infer_text_triples(
            text="Employees shall use eye protection .",
            checkpoint_path=result.best_checkpoint_path,
            encoder_name=config.encoder_name,
            predicates=config.predicates,
            tag_space=tag_space,
            max_length=config.max_length,
            threshold=config.threshold,
        )

        assert isinstance(triples, list)


def _create_local_tiny_bert(path: Path) -> Path:
    from transformers import BertConfig, BertModel, BertTokenizer

    path.mkdir(parents=True, exist_ok=True)
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        ".",
        "Employees",
        "Workers",
        "shall",
        "use",
        "eye",
        "protection",
        "perform",
        "welding",
        "operations",
    ]
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab), encoding="utf-8")

    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(path)

    model = BertModel(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
        )
    )
    model.save_pretrained(path)
    return path


def _write_gold_splits(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    train_path = root / "train.jsonl"
    val_path = root / "val.jsonl"
    annotations = [
        GoldSentenceAnnotation(
            sentence_id="s1",
            text="Employees shall use eye protection .",
            tokens=("Employees", "shall", "use", "eye", "protection", "."),
            triples=(
                GoldTripleAnnotation(
                    subject_span=(0, 0),
                    predicate="be_equipped_with",
                    object_span=(3, 4),
                    subject_text="Employees",
                    object_text="eye protection",
                ),
            ),
            source_standard="1910.133",
            section_ref="1910.133(a)(1)",
            source_url="https://example.test/1910.133",
            domain_tags=("manufacturing", "ppe"),
        ),
        GoldSentenceAnnotation(
            sentence_id="s2",
            text="Workers perform welding operations .",
            tokens=("Workers", "perform", "welding", "operations", "."),
            triples=(
                GoldTripleAnnotation(
                    subject_span=(0, 0),
                    predicate="perform_operations",
                    object_span=(2, 3),
                    subject_text="Workers",
                    object_text="welding operations",
                ),
            ),
            source_standard="1910.252",
            section_ref="1910.252(a)(1)",
            source_url="https://example.test/1910.252",
            domain_tags=("manufacturing", "welding"),
        ),
    ]
    write_gold_annotations_jsonl(train_path, annotations)
    write_gold_annotations_jsonl(val_path, annotations)
    return train_path, val_path
