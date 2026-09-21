# Benchmark Protocol — 2026 Modernization

This document defines the default evaluation contract for the maintained WangchanBERTa + LST20 path.

## Purpose

The goal is a **reproducible project benchmark**, not a state-of-the-art claim.

The original experiments were created in 2022. The protocol below belongs to the September 2026 modernization and should be used for new reported results unless a future experiment explicitly documents a deviation.

## Dataset

- **Corpus:** LST20
- **Acquisition:** authorized local copy obtained from AI for Thai
- **Train split:** `train/`
- **Validation split:** `eval/`
- **Test split:** `test/`
- **Redistribution:** corpus files are not committed to this repository

Run the corpus audit before training:

```bash
thai-ner-audit --data-dir /path/to/LST20Corpus
```

## Base model

```text
airesearch/wangchanberta-base-att-spm-uncased
```

## Label space

Training preserves the original 31 LST20 NER labels:

- `O`
- `B_*`
- `I_*`
- `E_*`

For entity-level scoring only, `E_*` tags are normalized to `I-*` so sequences can be evaluated in IOB2 form with `seqeval`.

## Token / subtoken alignment

- input examples are pre-tokenized at the LST20 word level
- WangchanBERTa uses a fast tokenizer
- only the first subtoken receives the source word's NER label
- later subtokens receive `-100` and are ignored by the loss

This rule must remain fixed when comparing 2026 runs.

## Default training configuration

| Parameter | Default |
|---|---:|
| Epochs | 3 |
| Learning rate | 2e-5 |
| Train batch size | 8 |
| Eval batch size | 8 |
| Max sequence length | 256 |
| Weight decay | 0.01 |
| Seed | 42 |

A run that changes these values is still valid, but the changes must be preserved in `benchmark_run.json`.

## Model selection

- evaluate on the validation split each epoch
- save each epoch
- load the checkpoint with the best validation F1
- evaluate the selected checkpoint once on the test split

Do not tune hyperparameters against the test set.

## Primary metric

**Entity-level F1** is the primary benchmark metric.

Also report:

- entity-level precision
- entity-level recall
- token accuracy
- test loss

## Publishable-result requirements

A result may be copied into the repository benchmark table only when all of the following are available:

1. successful corpus audit
2. complete train / validation / test run
3. generated `benchmark_run.json`
4. generated `BENCHMARK_RESULT.md`
5. Python/package versions
6. seed and hyperparameters
7. hardware metadata
8. exact base model identifier
9. no manual editing of the metric values

## Recommended command

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

## Interpretation

The resulting score describes performance under this specific LST20 benchmark protocol.

It should not be treated as evidence of equivalent performance on banking documents, chats, social media, OCR text, legal text, healthcare text, or other Thai-language domains without separate evaluation.
