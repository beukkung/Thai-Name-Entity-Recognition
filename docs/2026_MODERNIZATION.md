# 2026 Modernization Update

## Why this update exists

The original notebooks in this repository were created in **2022** as Thai NER experiments using BiLSTM, WangchanBERTa, and PyThaiNLP.

In **September 2026**, the repository was revisited and modernized as a continuation of that work. The goal is not to rewrite history or present the 2022 notebooks as newly created. Instead, the repository now shows an explicit evolution:

**2022 experiments → 2026 reproducibility, CI, and ML engineering refresh**

## What changed in 2026

The modernization adds:

- professional repository documentation
- CI-tested PyThaiNLP quickstart
- notebook validation in GitHub Actions
- a modern source package under `src/thai_ner_2026/`
- a direct LST20 corpus reader that does not depend on the historical dataset script
- WangchanBERTa token-classification training with current Hugging Face APIs
- word-to-subtoken label alignment
- entity-level precision, recall, F1, and accuracy
- automated unit tests
- a WangchanBERTa tokenizer/config smoke test
- a modern walkthrough notebook
- clearer separation between historical artifacts and current maintained code


## 2022 vs 2026: what is different?

| Area | Original 2022 notebooks | Maintained 2026 path |
|---|---|---|
| Experiment shape | Cell-by-cell exploratory scripts | Reusable package, CLI, notebooks, and tests |
| NER engine | Historical `ThaiNameTagger` / older APIs | Explicit PyThaiNLP `thainer-v2` quickstart plus WangchanBERTa fine-tuning |
| Tokenization | Notebook-specific preprocessing | Fast-tokenizer `word_ids()` alignment with ignored special/non-first subtokens |
| Training | Hidden local paths and inline installs | Configurable CLI with seed, validation, checkpoint selection, resume, and recorded run metadata |
| Evaluation | Predictions and token-level inspection | Entity-level precision, recall, F1, and token accuracy with a fixed protocol |
| Data handling | Assumes local files without a preflight contract | Authorized LST20 acquisition, strict parser, corpus audit, and split-aware benchmark |
| Quality control | No continuous execution contract | Notebook validation, CI smoke tests, unit tests, and generated benchmark artifacts |
| Claims | Informal experiment output | No benchmark claim until the complete run is recorded and reproducible |

The update is intentionally additive: the historical notebooks are preserved for learning and provenance, while new work is clearly marked as a 2026 continuation.

## What current Thai NER practice looks like in 2026

The maintained path follows the current, widely used Transformer token-classification workflow:

1. Start with a lightweight baseline so the task and labels can be inspected quickly.
2. Fine-tune a Thai pretrained encoder such as WangchanBERTa with a token-classification head when task-specific accuracy matters.
3. Keep the corpus word labels, then align them to SentencePiece subtokens with a fast tokenizer. Special tokens and non-first subtokens are ignored with `-100`.
4. Use dynamic padding through `DataCollatorForTokenClassification` instead of padding every example to the global maximum.
5. Select the checkpoint on validation entity-level F1, then evaluate the selected checkpoint once on the held-out test split.
6. Record dataset provenance, label mapping, seed, hyperparameters, package versions, hardware, and generated metrics as run artifacts.
7. Add per-domain evaluation and error analysis before claiming the model is suitable for production text. LST20 results alone do not establish performance on banking, chat, OCR, legal, or social-media text.
8. Use a nested-NER model only when entities can overlap; this is a different task from the flat LST20 label space and is not silently mixed into this benchmark.

These choices correspond to the current Hugging Face token-classification recipe and the current PyThaiNLP NER API. See the [Hugging Face token-classification guide](https://huggingface.co/docs/transformers/tasks/token_classification), [PyThaiNLP tagging documentation](https://pythainlp.org/docs/5.3/api/tag.html), and [WangchanBERTa model card](https://huggingface.co/airesearch/wangchanberta-base-att-spm-uncased).

## 2026 scope boundary

This repository now provides a **reproducible modernization path**, not a claim that the original 2022 notebooks have become historically new or that this project is state of the art.

A complete 2026 benchmark still requires an authorized LST20 copy and a full run on the documented protocol. The smoke test and CI prove that the pipeline wiring works; they are not a substitute for a full-corpus result.

## Design principle

The original notebooks remain in the repository root as historical learning artifacts.

New work lives in explicit 2026 paths:

```text
src/thai_ner_2026/
notebooks/*_2026.ipynb
docs/2026_MODERNIZATION.md
requirements-2026.txt
```

This makes it clear which code belongs to the original experiment and which code represents the later engineering update.

## Dataset policy

LST20 is not redistributed in this repository.

The 2026 pipeline expects the user to obtain the corpus from the authorized source, extract it locally, and provide the path through `--data-dir`.

Expected structure:

```text
LST20Corpus/
├── train/
├── eval/
└── test/
```

## Run the 2026 training pipeline

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

pip install -r requirements-2026.txt

thai-ner-train \
  --data-dir /path/to/LST20Corpus \
  --output-dir outputs/wangchanberta-lst20-2026
```

For a short pipeline check before a full training run:

```bash
thai-ner-train \
  --data-dir /path/to/LST20Corpus \
  --smoke-limit 100 \
  --epochs 1
```

## Evaluation

The model keeps the original LST20 31-class label space for training.

For entity-level evaluation, LST20 `B_` / `I_` / `E_` tags are normalized to IOB2-style labels for `seqeval`:

- `B_PER` → `B-PER`
- `I_PER` → `I-PER`
- `E_PER` → `I-PER`

Reported metrics:

- precision
- recall
- F1
- token accuracy

## Reproducibility boundary

The 2026 source pipeline is actively validated by CI, but a full LST20 benchmark is intentionally **not** run in GitHub Actions because:

1. the corpus requires manual authorized acquisition, and
2. full WangchanBERTa fine-tuning is a GPU-scale workload.

CI therefore validates the parts that can be verified responsibly without redistributing the corpus:

- Python tests
- LST20 parser behavior on synthetic fixtures
- label normalization
- WangchanBERTa configuration availability
- WangchanBERTa fast tokenizer availability
- word/subtoken label alignment

## Next milestone

The next meaningful artifact is a fully recorded benchmark run with:

- exact environment
- GPU/runtime
- random seed
- training hyperparameters
- precision / recall / F1
- saved model artifacts or model card
- comparison against the original 2022 approaches

Until that run is completed, this repository should be described as a **modernized reproducible training pipeline**, not as a new state-of-the-art benchmark.
