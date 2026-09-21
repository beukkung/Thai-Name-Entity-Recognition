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
