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
