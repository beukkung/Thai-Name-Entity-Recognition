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

## 2022 — Original experiments

Initial notebook-based Thai Named Entity Recognition experiments:

- BiLSTM sequence labeling
- WangchanBERTa / Transformer NER experimentation
- PyThaiNLP baseline experimentation
- LST20-based preprocessing and prediction workflows
