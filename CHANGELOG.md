# Project Timeline / Changelog

## September 2026 — Modernization continuation

The repository was revisited as a continuation of the original Thai NER experiments.

Highlights:

- rebuilt project documentation
- added CI-tested PyThaiNLP quickstart
- separated historical 2022 notebooks from maintained 2026 code
- introduced reusable `src/thai_ner_2026/` package
- modernized WangchanBERTa + LST20 training workflow
- added direct LST20 parsing
- added word/subtoken label alignment
- added entity-level precision / recall / F1 evaluation
- added automated tests and real WangchanBERTa tokenizer smoke checks
- added corpus audit and reproducible benchmark artifacts
- added benchmark protocol and model-card template

The 2026 work is an engineering and reproducibility update. It does not retroactively change the date or provenance of the original experiments.


## September 2026 — Current-practice follow-up

The continuation was tightened after executing the maintained quickstart in GitHub Actions:

- switched the maintained PyThaiNLP notebook to the explicit `thainer-v2` engine
- added the PyTorch and Transformers dependencies required by that engine
- validated all maintained notebooks for Python syntax
- made LST20 parsing strict by default so unknown labels cannot silently become `O`
- added resumable training, gradient accumulation, mixed precision, gradient checkpointing, and run metadata for those settings
- documented the practical differences between the 2022 experiments and the 2026 workflow
- confirmed the quickstart notebook, package tests, and WangchanBERTa integration smoke test pass in CI

A full LST20 benchmark remains data- and compute-dependent and is not represented by the quickstart CI result.

## 2022 — Original experiments

Initial notebook-based Thai Named Entity Recognition experiments:

- BiLSTM sequence labeling
- WangchanBERTa / Transformer NER experimentation
- PyThaiNLP baseline experimentation
- LST20-based preprocessing and prediction workflows
