"""LST20 corpus reader for the 2026 modernization.

The official LST20 corpus is downloaded manually from an authorized source. This
module reads the extracted train/eval/test text files directly instead of
depending on the historical Hugging Face dataset loading script.
"""

from __future__ import annotations

from pathlib import Path

from .labels import LST20_NER_TAGS

SPLIT_DIRS = {
    "train": "train",
    "validation": "eval",
    "test": "test",
}
VALID_NER_TAGS = set(LST20_NER_TAGS)


def _flush_example(
    examples: list[dict],
    file_name: str,
    sentence_id: int,
    tokens: list[str],
    pos_tags: list[str],
    ner_tags: list[str],
    clause_tags: list[str],
) -> None:
    if not tokens:
        return
    if not (len(tokens) == len(pos_tags) == len(ner_tags) == len(clause_tags)):
        raise ValueError(
            f"Column lengths became inconsistent while parsing {file_name!r} "
            f"sentence {sentence_id}."
        )
    examples.append(
        {
            "id": f"{file_name}:{sentence_id}",
            "fname": file_name,
            "tokens": tokens.copy(),
            "pos_tags": pos_tags.copy(),
            "ner_tags": ner_tags.copy(),
            "clause_tags": clause_tags.copy(),
        }
    )


def parse_lst20_file(path: str | Path, *, strict: bool = True) -> list[dict]:
    """Parse one LST20 tab-separated file into sentence-level examples.

    In strict mode (the default), an unknown NER label raises immediately
    instead of silently becoming O. This prevents a malformed or changed
    corpus from producing a misleading benchmark.
    """
    path = Path(path)
    examples: list[dict] = []
    tokens: list[str] = []
    pos_tags: list[str] = []
    ner_tags: list[str] = []
    clause_tags: list[str] = []
    sentence_id = 0

    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if line.strip() == "":
                if tokens:
                    _flush_example(
                        examples,
                        path.name,
                        sentence_id,
                        tokens,
                        pos_tags,
                        ner_tags,
                        clause_tags,
                    )
                    sentence_id += 1
                    tokens.clear()
                    pos_tags.clear()
                    ner_tags.clear()
                    clause_tags.clear()
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                raise ValueError(
                    f"Expected at least 4 tab-separated columns in {path} "
                    f"at line {line_number}, got {len(parts)}: {line!r}"
                )

            token, pos_tag, ner_tag, clause_tag = parts[:4]
            if ner_tag not in VALID_NER_TAGS:
                if strict:
                    expected = ", ".join(sorted(VALID_NER_TAGS))
                    raise ValueError(
                        f"Unsupported LST20 NER tag {ner_tag!r} in {path} "
                        f"at line {line_number}. Expected one of: {expected}"
                    )
                ner_tag = "O"

            tokens.append(token)
            pos_tags.append(pos_tag)
            ner_tags.append(ner_tag)
            clause_tags.append(clause_tag)

    if tokens:
        _flush_example(
            examples,
            path.name,
            sentence_id,
            tokens,
            pos_tags,
            ner_tags,
            clause_tags,
        )

    return examples


def load_lst20_split(
    data_dir: str | Path,
    split: str,
    limit: int | None = None,
) -> list[dict]:
    """Load one split from an extracted LST20Corpus directory."""
    if split not in SPLIT_DIRS:
        raise ValueError(f"Unknown split {split!r}; choose from {sorted(SPLIT_DIRS)}")
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive integer when provided")

    split_dir = Path(data_dir).expanduser().resolve() / SPLIT_DIRS[split]
    if not split_dir.exists():
        raise FileNotFoundError(
            f"{split_dir} does not exist. Expected an extracted LST20 corpus "
            "with train/, eval/, and test/ directories."
        )

    examples: list[dict] = []
    for path in sorted(split_dir.glob("*.txt")):
        examples.extend(parse_lst20_file(path))
        if limit is not None and len(examples) >= limit:
            return examples[:limit]

    if not examples:
        raise FileNotFoundError(f"No LST20 .txt examples found in {split_dir}")

    return examples


def load_lst20_dataset_dict(
    data_dir: str | Path,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    test_limit: int | None = None,
):
    """Return a Hugging Face DatasetDict for the extracted corpus."""
    from datasets import Dataset, DatasetDict

    return DatasetDict(
        {
            "train": Dataset.from_list(
                load_lst20_split(data_dir, "train", train_limit)
            ),
            "validation": Dataset.from_list(
                load_lst20_split(data_dir, "validation", validation_limit)
            ),
            "test": Dataset.from_list(
                load_lst20_split(data_dir, "test", test_limit)
            ),
        }
    )
