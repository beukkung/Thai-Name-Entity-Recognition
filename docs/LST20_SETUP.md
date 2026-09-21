# LST20 Setup for the 2026 Benchmark

## Official acquisition path

LST20 is **not bundled** with this repository.

As of the September 2026 modernization, NECTEC publishes an official Open-D resource for LST20:

```text
https://opend-portal.nectec.or.th/dataset/d1364791-84bc-4b65-9904-79aa0aa2c5a6/resource/063e2392-1eba-4099-a732-fbaf1ba9a293/download/opend_lst20_corpus.zip
```

The repository uses that URL only as an **upstream source reference**. Corpus files remain local and are excluded from Git by `.gitignore`.

NECTEC's published usage conditions permit non-commercial research and open-source use subject to the stated terms, while modification and redistribution of the corpus are restricted. Review the current resource page before use.

## Option A — local helper

From the repository root:

```bash
python scripts/fetch_lst20_opend.py
```

The helper:

1. downloads from the official Open-D URL
2. validates the ZIP signature and archive integrity
3. checks the current documented resource size
4. extracts into `data/`
5. discovers the directory containing `train/`, `eval/`, and `test/`
6. prints the next audit and smoke-training commands

If you already downloaded the official ZIP in your browser:

```bash
python scripts/fetch_lst20_opend.py \
  --zip-path /path/to/opend_lst20_corpus.zip \
  --existing-zip
```

This is useful when the Open-D route is slow from cloud-hosted compute.

## Cloud-runner note

During the September 2026 repository modernization, the official Open-D endpoint was tested from GitHub-hosted Ubuntu and macOS runners.

The endpoint returned the expected `application/zip` resource and a `Content-Length` of **16,117,011 bytes**, but the transfer path from those hosted runners was extremely slow. Because of that network behavior, the project does **not** run a full Open-D corpus download automatically in normal CI.

This is a network-path limitation, not evidence that the official resource is unavailable.

## Expected extracted layout

The maintained parser expects a corpus root containing:

```text
LST20Corpus/
├── train/
│   └── *.txt
├── eval/
│   └── *.txt
└── test/
    └── *.txt
```

The helper searches recursively for this structure after extraction, so an extra top-level directory in the ZIP is fine.

## Validate before training

```bash
pip install -r requirements-2026.txt

thai-ner-audit --data-dir /path/to/LST20Corpus
```

The audit reports:

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

## Data handling rule

Do **not** commit, attach to a GitHub release, or publish the LST20 ZIP/extracted corpus through this repository.

Only generated code, benchmark metadata, aggregate metrics, and other outputs allowed by the applicable terms should be committed.
