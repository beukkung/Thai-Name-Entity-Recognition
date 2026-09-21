"""CI smoke test for the 2026 WangchanBERTa modernization."""

from __future__ import annotations

import tempfile
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from thai_ner_2026.data import load_lst20_split
from thai_ner_2026.labels import LABEL2ID
from thai_ner_2026.preprocessing import tokenize_and_align_batch

MODEL = "airesearch/wangchanberta-base-att-spm-uncased"


def build_tiny_lst20(root: Path) -> None:
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


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL)
    print(f"Loaded config: model_type={config.model_type}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    print(f"Loaded tokenizer: {type(tokenizer).__name__}; is_fast={tokenizer.is_fast}")
    if not tokenizer.is_fast:
        raise RuntimeError("WangchanBERTa did not load as a fast tokenizer.")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_tiny_lst20(root)
        train_examples = load_lst20_split(root, "train")
        assert len(train_examples) == 2
        assert train_examples[0]["ner_tags"] == ["B_TTL", "B_PER", "E_PER"]

        batch = {
            "tokens": [train_examples[0]["tokens"]],
            "ner_tags": [train_examples[0]["ner_tags"]],
        }
        encoded = tokenize_and_align_batch(batch, tokenizer, LABEL2ID, max_length=64)
        assert len(encoded["input_ids"][0]) == len(encoded["labels"][0])
        assert any(label != -100 for label in encoded["labels"][0])

    print("2026 WangchanBERTa pipeline smoke test passed.")


if __name__ == "__main__":
    main()
