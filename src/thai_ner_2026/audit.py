"""Corpus audit helpers for reproducible LST20 benchmark runs."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from .data import load_lst20_split


def summarize_split(data_dir: str | Path, split: str) -> dict:
    examples = load_lst20_split(data_dir, split)
    token_count = sum(len(example["tokens"]) for example in examples)
    entity_token_count = sum(
        1
        for example in examples
        for tag in example["ner_tags"]
        if tag != "O"
    )
    label_counts = Counter(
        tag for example in examples for tag in example["ner_tags"]
    )
    files = sorted(
        (Path(data_dir) / {"train": "train", "validation": "eval", "test": "test"}[split]).glob("*.txt")
    )

    return {
        "split": split,
        "files": len(files),
        "sentences": len(examples),
        "tokens": token_count,
        "entity_tokens": entity_token_count,
        "label_counts": dict(sorted(label_counts.items())),
    }


def audit_corpus(data_dir: str | Path) -> dict:
    data_dir = Path(data_dir).expanduser().resolve()
    splits = {
        split: summarize_split(data_dir, split)
        for split in ("train", "validation", "test")
    }
    return {
        "data_dir": str(data_dir),
        "splits": splits,
        "totals": {
            "files": sum(item["files"] for item in splits.values()),
            "sentences": sum(item["sentences"] for item in splits.values()),
            "tokens": sum(item["tokens"] for item in splits.values()),
            "entity_tokens": sum(item["entity_tokens"] for item in splits.values()),
        },
    }
