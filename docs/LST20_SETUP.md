# LST20 Setup for the 2026 Benchmark

## Official acquisition path

LST20 is not bundled with this repository.

The official Hugging Face dataset card instructs users to manually obtain the corpus from **AI for Thai**, where login is required. This repository follows that access model and does not attempt to bypass authentication or redistribute corpus files.

After obtaining and extracting the corpus, point the pipeline to the extracted directory.

Expected layout:

```text
LST20Corpus/
├── train/
│   └── *.txt
├── eval/
│   └── *.txt
└── test/
    └── *.txt
```

## Validate before training

```bash
pip install -r requirements-2026.txt

thai-ner-audit --data-dir /path/to/LST20Corpus
```

The audit checks that the expected split folders contain parsable text files and reports:

- file count
- sentence count
- token count
- entity-labelled token count
- label distribution

This creates a reproducibility checkpoint before model training begins.

## Short smoke run

```bash
thai-ner-train \
  --data-dir /path/to/LST20Corpus \
  --smoke-limit 100 \
  --epochs 1 \
  --output-dir outputs/smoke-2026
```

## Full benchmark

```bash
thai-ner-train \
  --data-dir /path/to/LST20Corpus \
  --epochs 3 \
  --learning-rate 2e-5 \
  --train-batch-size 8 \
  --eval-batch-size 8 \
  --max-length 256 \
  --seed 42 \
  --output-dir outputs/wangchanberta-lst20-2026
```

The training command automatically writes benchmark metadata and metrics alongside the saved model.
