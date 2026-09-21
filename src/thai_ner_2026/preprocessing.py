"""Tokenization and word-to-subtoken label alignment."""

from __future__ import annotations

from typing import Mapping


def tokenize_and_align_batch(
    examples: Mapping[str, list],
    tokenizer,
    label2id: Mapping[str, int],
    max_length: int = 256,
):
    """Tokenize pre-split words and assign labels to the first subtoken only.

    Subtokens after the first piece of each source word receive -100 so they are
    ignored by the token-classification loss, matching the standard Hugging Face
    recipe for NER.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError(
            "The 2026 pipeline requires a fast tokenizer for reliable word_ids() "
            "alignment. Load WangchanBERTa with use_fast=True."
        )

    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )

    aligned_labels: list[list[int]] = []
    for batch_index, word_labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        labels: list[int] = []

        for word_id in word_ids:
            if word_id is None:
                labels.append(-100)
            elif word_id != previous_word_id:
                labels.append(label2id[word_labels[word_id]])
            else:
                labels.append(-100)
            previous_word_id = word_id

        aligned_labels.append(labels)

    tokenized["labels"] = aligned_labels
    return tokenized
