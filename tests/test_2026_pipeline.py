from pathlib import Path

import numpy as np

from thai_ner_2026.data import parse_lst20_file
from thai_ner_2026.metrics import build_compute_metrics, lst20_label_to_iob2


def test_parse_lst20_file(tmp_path: Path):
    path = tmp_path / "sample.txt"
    path.write_text(
        "นาย\tNN\tB_TTL\tO\n"
        "สมชาย\tNN\tB_PER\tO\n"
        "ใจดี\tNN\tE_PER\tO\n"
        "\n"
        "กรุงเทพ\tNN\tNOT_A_REAL_TAG\tO\n",
        encoding="utf-8",
    )

    examples = parse_lst20_file(path)

    assert len(examples) == 2
    assert examples[0]["tokens"] == ["นาย", "สมชาย", "ใจดี"]
    assert examples[0]["ner_tags"] == ["B_TTL", "B_PER", "E_PER"]
    assert examples[1]["ner_tags"] == ["O"]


def test_lst20_label_to_iob2():
    assert lst20_label_to_iob2("O") == "O"
    assert lst20_label_to_iob2("B_PER") == "B-PER"
    assert lst20_label_to_iob2("I_ORG") == "I-ORG"
    assert lst20_label_to_iob2("E_LOC") == "I-LOC"


def test_compute_metrics_perfect_prediction():
    labels = ["O", "B_PER", "I_PER", "E_PER"]
    compute = build_compute_metrics(labels)

    logits = np.zeros((1, 4, len(labels)), dtype=float)
    target = np.array([[0, 1, 2, 3]])
    for index, label_id in enumerate(target[0]):
        logits[0, index, label_id] = 10.0

    result = compute((logits, target))
    assert result["f1"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def _write_tiny_corpus(root: Path):
    sample = (
        "นาย\tNN\tB_TTL\tO\n"
        "สมชาย\tNN\tB_PER\tO\n"
        "ใจดี\tNN\tE_PER\tO\n"
        "\n"
        "กรุงเทพมหานคร\tNN\tB_LOC\tO\n"
        "\n"
    )
    for folder in ("train", "eval", "test"):
        split = root / folder
        split.mkdir(parents=True, exist_ok=True)
        (split / "sample.txt").write_text(sample, encoding="utf-8")


def test_corpus_audit(tmp_path: Path):
    from thai_ner_2026.audit import audit_corpus

    _write_tiny_corpus(tmp_path)
    result = audit_corpus(tmp_path)

    assert result["totals"]["files"] == 3
    assert result["totals"]["sentences"] == 6
    assert result["totals"]["tokens"] == 12
    assert result["totals"]["entity_tokens"] == 12
    assert result["splits"]["train"]["label_counts"]["B_PER"] == 1


def test_benchmark_reporting(tmp_path: Path):
    import json

    from thai_ner_2026.reporting import write_run_artifacts

    run = {
        "training": {
            "model_name": "test/model",
            "epochs": 1,
            "learning_rate": 2e-5,
            "train_batch_size": 2,
            "eval_batch_size": 2,
            "max_length": 64,
            "seed": 42,
        },
        "dataset_audit": {
            "totals": {
                "files": 3,
                "sentences": 6,
                "tokens": 12,
                "entity_tokens": 12,
            }
        },
        "metrics": {
            "test_precision": 0.9,
            "test_recall": 0.8,
            "test_f1": 0.847,
            "test_accuracy": 0.95,
            "test_loss": 0.2,
        },
        "environment": {
            "timestamp_utc": "2026-09-21T00:00:00+00:00",
            "python": "3.11",
            "platform": "test",
            "hardware": {"cuda_available": False, "cuda_device": None},
        },
    }

    write_run_artifacts(tmp_path, run)

    payload = json.loads((tmp_path / "benchmark_run.json").read_text(encoding="utf-8"))
    report = (tmp_path / "BENCHMARK_RESULT.md").read_text(encoding="utf-8")

    assert payload["metrics"]["test_f1"] == 0.847
    assert "| F1 | 0.8470 |" in report
    assert "**Base model:** test/model" in report
