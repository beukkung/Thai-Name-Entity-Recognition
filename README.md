# Thai Named Entity Recognition 🇹🇭

[![Notebook CI](https://github.com/beukkung/Thai-Name-Entity-Recognition/actions/workflows/notebook-ci.yml/badge.svg)](https://github.com/beukkung/Thai-Name-Entity-Recognition/actions/workflows/notebook-ci.yml)
![Thai NLP](https://img.shields.io/badge/Thai-NLP-0A66C2)
![Named Entity Recognition](https://img.shields.io/badge/Task-Named%20Entity%20Recognition-6f42c1)
![LST20](https://img.shields.io/badge/Dataset-LST20-2ea44f)
![WangchanBERTa](https://img.shields.io/badge/Model-WangchanBERTa-orange)
![PyThaiNLP](https://img.shields.io/badge/Toolkit-PyThaiNLP-blue)
![2026 Modernization](https://img.shields.io/badge/Update-2026%20Modernization-8A2BE2)

> [!IMPORTANT]
> **Project timeline:** the original Thai NER experiments were created in **2022**. In **September 2026**, this repository was actively revisited as a **modernization / continuation update**—adding reproducible project structure, CI, a maintained PyThaiNLP quickstart, and a modern WangchanBERTa + LST20 training pipeline. The historical notebooks remain intact so the evolution of the project is explicit.

➡️ **[Read the 2026 Modernization Notes](docs/2026_MODERNIZATION.md)**

---

A compact research repository for experimenting with **Thai Named Entity Recognition (NER)** using multiple modeling approaches, including **BiLSTM**, **WangchanBERTa / Transformer-based NER**, and a **PyThaiNLP baseline**.

The project is centered on Thai-language sequence labeling and uses the **LST20 corpus** as the main reference dataset in the notebook workflow.

> This repository is primarily an educational and experimental project. It is useful for studying different Thai NER approaches, comparing modeling strategies, and understanding an end-to-end sequence-labeling workflow in Jupyter/Google Colab.

---

## ⚡ Quickstart — CI-tested notebook

For a runnable first experience, use the modern PyThaiNLP notebook:

➡️ **[Open `notebooks/00_quickstart_pythainlp.ipynb`](notebooks/00_quickstart_pythainlp.ipynb)**  
➡️ **[Open in Google Colab](https://colab.research.google.com/github/beukkung/Thai-Name-Entity-Recognition/blob/main/notebooks/00_quickstart_pythainlp.ipynb)**

This quickstart uses the current PyThaiNLP `thainer-v2` NER engine and:

- does **not** require the LST20 dataset
- uses the current PyThaiNLP public NER API
- is executed automatically by GitHub Actions
- contains a small assertion so CI fails when inference does not return tagged tokens

### Notebook status

| Notebook | Status | CI execution | Notes |
|---|---|---|---|
| `notebooks/00_quickstart_pythainlp.ipynb` | ✅ 2026 Modern | ✅ Yes | Recommended zero-dataset starting point |
| `notebooks/10_wangchanberta_lst20_2026.ipynb` | 🧪 2026 Modernization | ⚙️ Pipeline tested | Modern WangchanBERTa + LST20 training walkthrough |
| `notebooks/20_lst20_benchmark_2026.ipynb` | 📊 2026 Benchmark | ⏳ Requires authorized LST20 | Audit → full run → reproducible result artifacts |
| `ThaiNER-PlyThaiNLP.ipynb` | ⚠️ Legacy (2022) | No | Historical code; old API / logic requires modernization |
| `ThaiNER-BILSTM.ipynb` | ⚠️ Legacy (2022) | No | Depends on local corpus/preprocessing and old TensorFlow-era code |
| `ThaiNER-BERT.ipynb` | ⚠️ Legacy (2022) | No | Depends on LST20, Google Drive paths, and historical Transformer stack |

> **Important:** “Notebook exists” is not the same as “reproducible benchmark.” The modern quickstart is continuously executed; the three original notebooks are preserved as historical experiments until their training pipelines are migrated and benchmarked.

---

## ✨ What this repository contains

| Notebook | Approach | Main purpose |
|---|---|---|
| `notebooks/00_quickstart_pythainlp.ipynb` | 2026 PyThaiNLP | CI-tested runnable entry point |
| `notebooks/10_wangchanberta_lst20_2026.ipynb` | 2026 WangchanBERTa | Modern reproducible LST20 training workflow |
| `notebooks/20_lst20_benchmark_2026.ipynb` | 2026 Benchmark runner | Corpus audit + recorded full benchmark workflow |
| `ThaiNER-BILSTM.ipynb` | Bidirectional LSTM | Builds a neural sequence-labeling workflow for Thai NER |
| `ThaiNER-BERT.ipynb` | WangchanBERTa / Transformer | Fine-tunes a Thai pretrained transformer for token classification |
| `ThaiNER-PlyThaiNLP.ipynb` | PyThaiNLP | Provides a lightweight baseline using `ThaiNameTagger` |

The notebooks cover different parts of the NER workflow: loading data, preprocessing, mapping entity labels, model training or inference, evaluation, and generating prediction outputs.

---

## 🧠 Named Entity Recognition

Named Entity Recognition identifies spans of text that refer to real-world entities such as people, organizations, locations, dates, quantities, titles, and other domain-specific categories.

For Thai text, NER can support use cases such as:

- document information extraction
- search and indexing
- customer-service analytics
- knowledge graph construction
- news and social-media analysis
- downstream NLP pipelines
- entity-aware retrieval and summarization

---

## 📚 Dataset

The notebooks are designed around the **LST20 corpus**, a Thai NLP dataset containing annotations for tasks including:

- token boundaries
- part-of-speech tags
- named entities
- clause boundaries

The Transformer notebook loads LST20 using the Hugging Face `datasets` workflow.

### Example NER label families

The notebook includes entity classes such as:

- `PER` — Person
- `ORG` — Organization
- `LOC` — Location
- `DTM` — Date / Time
- `MEA` — Measurement
- `NUM` — Number
- `TTL` — Title
- `DES` — Designation
- `BRN` — Brand
- `TRM` — Term

The LST20 labels follow a span-tagging convention with prefixes such as `B_`, `I_`, and `E_`.

> Dataset files are **not bundled in this repository**. The 2026 workflow documents NECTEC's official Open-D LST20 resource and provides a local fetch/validation helper; corpus files remain excluded from Git. See [LST20 setup](docs/LST20_SETUP.md).

---

## 🗂 Repository structure

```text
Thai-Name-Entity-Recognition/
├── ThaiNER-BILSTM.ipynb
├── ThaiNER-BERT.ipynb
├── ThaiNER-PlyThaiNLP.ipynb
├── notebooks/
│   ├── 00_quickstart_pythainlp.ipynb
│   ├── 10_wangchanberta_lst20_2026.ipynb
│   └── 20_lst20_benchmark_2026.ipynb
├── src/thai_ner_2026/
│   ├── data.py
│   ├── labels.py
│   ├── audit.py
│   ├── metrics.py
│   ├── preprocessing.py
│   ├── reporting.py
│   └── train.py
├── tests/
│   └── test_2026_pipeline.py
├── benchmark/
│   ├── README.md
│   └── results/
├── CHANGELOG.md
├── README.md
├── requirements.txt
├── requirements-modern.txt
├── requirements-2026.txt
├── pyproject.toml
├── CONTRIBUTING.md
├── scripts/
│   ├── validate_notebooks.py
│   └── fetch_lst20_opend.py
├── .github/workflows/
│   └── notebook-ci.yml
├── docs/
│   ├── REPRODUCIBILITY.md
│   ├── 2026_MODERNIZATION.md
│   ├── LST20_SETUP.md
│   ├── BENCHMARK_PROTOCOL_2026.md
│   └── MODEL_CARD_2026_TEMPLATE.md
├── .gitignore
└── .gitattributes
```

---

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone https://github.com/beukkung/Thai-Name-Entity-Recognition.git
cd Thai-Name-Entity-Recognition
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install the modern quickstart environment

```bash
pip install -r requirements-modern.txt
```

```bash
# Execute the CI-tested quickstart locally
jupyter nbconvert \
  --to notebook \
  --execute notebooks/00_quickstart_pythainlp.ipynb \
  --output /tmp/00_quickstart_pythainlp.executed.ipynb \
  --ExecutePreprocessor.timeout=600
```

For the historical notebooks, `requirements.txt` is a reference list rather than a guaranteed lock file. See the reproducibility notes before recreating the 2022 environments.

Some notebooks were originally created in Google Colab and contain notebook-specific installation commands. Because several dependencies have changed since the original 2022 experiments, see [Reproducibility notes](docs/REPRODUCIBILITY.md) before attempting to rerun every cell unchanged.

---

## 🧪 Recommended notebook order

If you are exploring the repository today:

1. Start with **`notebooks/00_quickstart_pythainlp.ipynb`** for a CI-tested Thai NER inference example.
2. Continue with **`notebooks/10_wangchanberta_lst20_2026.ipynb`** for the maintained **2026 WangchanBERTa + LST20** pipeline.
3. Then inspect the root-level **2022 notebooks** to understand the original BiLSTM, PyThaiNLP, and WangchanBERTa experiments.

This ordering makes the repository's history explicit: **use the 2026 code to reproduce current workflows; use the 2022 notebooks to study the original experiments.**

---

## 🆕 2026 WangchanBERTa modernization

The maintained training path now lives in `src/thai_ner_2026/` and uses current Hugging Face token-classification patterns:

```bash
pip install -r requirements-2026.txt

thai-ner-train \
  --data-dir /path/to/LST20Corpus \
  --epochs 3 \
  --output-dir outputs/wangchanberta-lst20-2026
```

The 2026 pipeline adds:

- direct parsing of authorized local LST20 `train/`, `eval/`, and `test/` files
- WangchanBERTa fast-tokenizer word/subtoken alignment
- a 31-label LST20 token-classification head
- entity-level precision / recall / F1 with `seqeval`
- reproducible CLI training arguments and fixed seed
- automated parser/metric unit tests
- CI smoke testing against the real WangchanBERTa tokenizer and configuration

A full benchmark score is intentionally **not claimed yet**. Publishing one responsibly requires a complete LST20 training run with recorded hardware, environment, hyperparameters, and metrics.

➡️ See **[docs/2026_MODERNIZATION.md](docs/2026_MODERNIZATION.md)** for the design and reproducibility boundary.

➡️ See **[2026 current practice and modernization notes](docs/2026_MODERNIZATION.md#what-current-thai-ner-practice-looks-like-in-2026)** for the explicit 2022 → 2026 comparison.

---

## 📊 2026 benchmark-ready workflow

The repository is now prepared to produce a **recorded, reproducible benchmark** from a local LST20 corpus.

NECTEC's official Open-D resource is documented in [docs/LST20_SETUP.md](docs/LST20_SETUP.md). For a local download + ZIP validation + extraction:

```bash
python scripts/fetch_lst20_opend.py
```

If the ZIP was downloaded in a browser, use `--existing-zip --zip-path /path/to/opend_lst20_corpus.zip`.

Then audit the corpus:

```bash
thai-ner-audit --data-dir /path/to/LST20Corpus
```

Then run the fixed 2026 protocol:

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

A successful run automatically produces:

```text
outputs/wangchanberta-lst20-2026/
├── best-model/
├── benchmark_run.json
└── BENCHMARK_RESULT.md
```

The JSON records the model, hyperparameters, dataset audit, package versions, timestamp, hardware metadata, and test metrics. This prevents benchmark values from being manually copied without experiment context.

See:

- [LST20 setup](docs/LST20_SETUP.md)
- [2026 benchmark protocol](docs/BENCHMARK_PROTOCOL_2026.md)
- [2026 model-card template](docs/MODEL_CARD_2026_TEMPLATE.md)
- [project timeline / changelog](CHANGELOG.md)

> LST20 itself is not redistributed by this repository. The full benchmark should run from a local copy obtained through NECTEC's official source, then only the generated benchmark metadata/metrics should be published.

---

## 🔬 Approaches

### 1. BiLSTM

The BiLSTM notebook explores sequence modeling using bidirectional recurrent layers. A bidirectional model can use both left and right context when predicting a token's label.

This approach is useful for learning the mechanics of classical neural sequence labeling before moving to large pretrained language models.

### 2. WangchanBERTa / Transformer

The Transformer notebook uses the Thai pretrained model:

```text
airesearch/wangchanberta-base-att-spm-uncased
```

The notebook combines Hugging Face tooling, `thai2transformers`, and `simpletransformers` to prepare and train a token-classification / NER workflow.

### 3. PyThaiNLP baseline

The maintained 2026 quickstart demonstrates NER inference with the current PyThaiNLP API:

```python
from pythainlp.tag import NER

ner = NER("thainer-v2")
entities = ner.tag("นายสมชายทำงานที่ธนาคารแห่งหนึ่งในกรุงเทพมหานคร")
```

This is useful as a quick baseline before investing in custom model training.

---

## ⚠️ Reproducibility note

This repository was originally developed in **2022**, and parts of the notebooks reference older library versions and APIs.

Examples include historical versions of:

- TensorFlow / Keras
- PyTorch
- Transformers
- thai2transformers
- Simple Transformers
- PyThaiNLP
- Hugging Face Datasets

As a result, a modern Python environment may require dependency adjustments.

For more detail, see:

➡️ [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)

---

## 📌 Current project status

**Status:** Active **2026 modernization** of a 2022 Thai NER research project.

Completed in the 2026 continuation:

- ✅ professional repository documentation
- ✅ CI-tested PyThaiNLP quickstart
- ✅ reusable `src/` package
- ✅ direct LST20 parser
- ✅ modern WangchanBERTa training CLI
- ✅ subtoken label alignment
- ✅ entity-level evaluation utilities
- ✅ automated unit and integration smoke tests
- ✅ LST20 corpus audit CLI
- ✅ fixed 2026 benchmark protocol
- ✅ automatic benchmark JSON + Markdown result artifacts
- ✅ model-card template and benchmark workspace

Next meaningful milestone:

- ✅ document and validate the official NECTEC Open-D LST20 source
- ⏳ run and record the complete WangchanBERTa + LST20 benchmark
- ⏳ publish the generated precision / recall / F1 result
- ⏳ modernize the historical BiLSTM path only if it adds comparison value

---

## 🏷️ Project topics

**thai-nlp** · **named-entity-recognition** · **ner** · **lst20** · **wangchanberta** · **pythainlp** · **bilstm** · **transformers** · **jupyter-notebook** · **natural-language-processing**

These keywords are also reflected in the repository documentation and citation metadata to improve discoverability.

---

## 🤝 Contributing

Contributions that improve reproducibility, documentation, evaluation, or compatibility with modern NLP libraries are welcome.

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

---

## 🙏 Acknowledgements and attribution

This repository builds on the Thai NLP ecosystem and uses or references work from projects including:

- **LST20** for Thai language annotations
- **WangchanBERTa** / AIResearch for Thai pretrained language modeling
- **PyThaiNLP** for Thai NLP tooling
- **Hugging Face** for datasets and Transformer tooling
- **Simple Transformers** for high-level NER training utilities

The `ThaiNER-BERT.ipynb` notebook also states that parts of the code were modified from an external Google Colab source. That attribution is preserved in the notebook itself.

Please cite and follow the licenses / terms of the upstream datasets, libraries, and pretrained models when reusing this work.

---

## 👤 Maintainer

**beukkung**

GitHub: [@beukkung](https://github.com/beukkung)

---

## ⭐ If this repository is useful

If this project helps you learn Thai NLP or compare Thai NER approaches, consider starring the repository. It helps make the project easier for others to discover.
