# Model Card Template — WangchanBERTa LST20 NER (2026)

> **Status:** Template awaiting a complete authorized LST20 training run.

## Model details

- **Project:** Thai Named Entity Recognition
- **Project history:** original experiments in 2022; modernized training pipeline in 2026
- **Base model:** `airesearch/wangchanberta-base-att-spm-uncased`
- **Task:** Thai Named Entity Recognition / token classification
- **Dataset:** LST20
- **Label space:** 31 LST20 NER tags

## Intended use

This model is intended for research, learning, experimentation, and evaluation of Thai NER workflows.

Potential applications include information extraction, document analytics, entity-aware search, and downstream Thai NLP systems.

## Training data

The model should be trained only from an authorized local copy of LST20.

The dataset is intentionally not redistributed in this repository.

## Training configuration

Populate this section from `benchmark_run.json`.

| Parameter | Value |
|---|---|
| Base model | pending |
| Epochs | pending |
| Learning rate | pending |
| Train batch size | pending |
| Eval batch size | pending |
| Max sequence length | pending |
| Seed | pending |
| Hardware | pending |

## Evaluation

Populate only after a complete test-set evaluation.

| Metric | Value |
|---|---:|
| Precision | pending |
| Recall | pending |
| F1 | pending |
| Token accuracy | pending |

## Limitations

- Performance may vary by news genre, entity type, spelling variation, domain, and tokenization behavior.
- A score on LST20 should not be assumed to represent production performance in banking, social media, conversational Thai, OCR text, or other domains.
- Entity-level errors can propagate to downstream retrieval, analytics, or automation systems.
- The model should be evaluated on the intended target domain before deployment.

## Reproducibility

A publishable run should include the generated `benchmark_run.json` and `BENCHMARK_RESULT.md`, plus hardware/runtime notes.

## Ethical and legal considerations

Respect the licenses and terms of the upstream dataset, pretrained model, and libraries. Do not redistribute restricted corpus files through this repository.
