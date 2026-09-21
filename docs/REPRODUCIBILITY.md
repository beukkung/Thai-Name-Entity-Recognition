# Reproducibility Notes

## Background

The notebooks in this repository were created in 2022 and use library APIs that have changed significantly since then.

The goal of this document is to help readers distinguish between:

1. the original experimental code,
2. historical package assumptions, and
3. what may need to change in a modern environment.


## Maintained 2026 quickstart

`notebooks/00_quickstart_pythainlp.ipynb` is the supported low-cost entry point. It uses the current PyThaiNLP `NER("thainer-v2")` API and is executed by CI. This is separate from the root-level `ThaiNER-PlyThaiNLP.ipynb`, which preserves the original 2022 `pythainlp==3.0.3` experiment.

The quickstart proves that a clean inference notebook runs. It does not prove that a full LST20 training benchmark can run without the separately acquired corpus and a suitable compute budget.

## Notebook-by-notebook notes

### ThaiNER-BERT.ipynb

This notebook references historical installs such as:

```text
torch==1.5.0
torchtext==0.4.0
torchvision==0.6.0
transformers==3.5.0
thai2transformers==0.1.2
```

It also uses:

- Hugging Face `datasets`
- `simpletransformers`
- WangchanBERTa
- `thai2transformers`

Modern versions of these packages may not be API-compatible with the original notebook.

The notebook intentionally limits some preprocessing to the first 1,000 rows for faster experimentation. Treat this as a demonstration setting, not a full benchmark configuration.

### ThaiNER-BILSTM.ipynb

The BiLSTM notebook uses TensorFlow / Keras and scikit-learn-style preprocessing.

When modernizing it, verify:

- tokenizer behavior
- sequence padding
- label encoding
- train / validation / test split behavior
- mask handling
- metric calculation

Older Keras and TensorFlow code can behave differently under current releases.

### ThaiNER-PlyThaiNLP.ipynb

The notebook explicitly installs:

```text
pythainlp==3.0.3
```

Modern PyThaiNLP releases may expose different NER model behavior or package requirements.

## Recommended reproducibility strategy

For historical reproduction:

1. Use Google Colab or an isolated environment.
2. Start from the package versions stated inside each notebook.
3. Adjust Python and CUDA versions only as needed for package compatibility.
4. Record every modification required to make the notebook execute.
5. Do not compare new results with old results unless dataset split, preprocessing, labels, and metrics are identical.

For modernization:

1. Create a fresh environment.
2. Use current maintained libraries.
3. Re-implement preprocessing explicitly.
4. Use a standard token-classification evaluation setup.
5. Report entity-level precision, recall, and F1.
6. Pin the final working environment in a lock file or environment specification.

## Dataset handling

Dataset files are intentionally not stored in the repository.

Users should obtain LST20 from an authorized source and preserve the expected train / validation / test structure used by their chosen loader.

## Reproducibility checklist

Before publishing a result, record:

- Python version
- operating system / runtime
- CPU / GPU
- dataset version
- train / validation / test split
- random seed
- tokenizer
- maximum sequence length
- batch size
- learning rate
- epochs
- label mapping
- evaluation metric definition
- package versions

## Benchmark warning

The notebooks are research and learning artifacts. Unless an experiment explicitly fixes the above settings and reports a reproducible evaluation procedure, its outputs should not be interpreted as a formal benchmark.
