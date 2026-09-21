# Benchmark workspace

This directory is reserved for **recorded 2026 benchmark results**.

The repository does not commit LST20 itself. A benchmark result should only be added after running the maintained pipeline against an authorized local copy of the corpus.

## Recommended workflow

```bash
pip install -r requirements-2026.txt

thai-ner-audit --data-dir /path/to/LST20Corpus

thai-ner-train \
  --data-dir /path/to/LST20Corpus \
  --output-dir outputs/wangchanberta-lst20-2026 \
  --epochs 3 \
  --seed 42
```

A successful training run writes:

```text
outputs/wangchanberta-lst20-2026/
├── best-model/
├── benchmark_run.json
└── BENCHMARK_RESULT.md
```

## Publishing a result

Before copying a result into this directory, verify that the run records:

- dataset audit
- exact base model
- seed
- learning rate
- batch sizes
- max sequence length
- number of epochs
- Python and package versions
- precision / recall / F1 / accuracy
- run timestamp

Do not publish a benchmark table with guessed or manually transcribed metrics.
