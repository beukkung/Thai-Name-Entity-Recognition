# Thai Named Entity Recognition 🇹🇭

A compact research repository for experimenting with **Thai Named Entity Recognition (NER)** using multiple modeling approaches, including **BiLSTM**, **WangchanBERTa / Transformer-based NER**, and a **PyThaiNLP baseline**.

The project is centered on Thai-language sequence labeling and uses the **LST20 corpus** as the main reference dataset in the notebook workflow.

> This repository is primarily an educational and experimental project. It is useful for studying different Thai NER approaches, comparing modeling strategies, and understanding an end-to-end sequence-labeling workflow in Jupyter/Google Colab.

---

## ✨ What this repository contains

| Notebook | Approach | Main purpose |
|---|---|---|
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

> Dataset files are **not bundled in this repository**. Please obtain the dataset from its official or authorized source and comply with its applicable terms of use.

---

## 🗂 Repository structure

```text
Thai-Name-Entity-Recognition/
├── ThaiNER-BILSTM.ipynb
├── ThaiNER-BERT.ipynb
├── ThaiNER-PlyThaiNLP.ipynb
├── README.md
├── requirements.txt
├── CONTRIBUTING.md
├── docs/
│   └── REPRODUCIBILITY.md
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

### 3. Install the common dependencies

```bash
pip install -r requirements.txt
```

Some notebooks were originally created in Google Colab and contain notebook-specific installation commands. Because several dependencies have changed since the original 2022 experiments, see [Reproducibility notes](docs/REPRODUCIBILITY.md) before attempting to rerun every cell unchanged.

---

## 🧪 Recommended notebook order

If you are exploring the project for the first time:

1. Start with **`ThaiNER-PlyThaiNLP.ipynb`** to understand a simple baseline.
2. Continue with **`ThaiNER-BILSTM.ipynb`** to study a recurrent neural-network approach.
3. Use **`ThaiNER-BERT.ipynb`** for a pretrained Transformer-based approach with WangchanBERTa.

This progression makes it easier to compare increasingly sophisticated approaches while keeping the same underlying NER problem in view.

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

The PyThaiNLP notebook demonstrates NER inference with:

```python
from pythainlp.tag.named_entity import ThaiNameTagger
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

**Status:** Research / educational archive with documented experiments.

The repository preserves the original notebook-based experimentation while improving documentation around setup, intent, limitations, and reuse.

Potential future improvements include:

- refactoring common preprocessing into reusable Python modules
- adding a deterministic evaluation script
- reporting precision / recall / F1 consistently across approaches
- adding a modern Hugging Face token-classification pipeline
- adding automated notebook smoke tests
- introducing experiment tracking and model artifacts
- publishing a small inference demo

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
