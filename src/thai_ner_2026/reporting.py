"""Benchmark result serialization and Markdown reporting."""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict:
    packages = [
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "sentencepiece",
        "seqeval",
        "numpy",
    ]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: _pkg_version(name) for name in packages},
    }


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_benchmark_summary(run: dict) -> str:
    metrics = run.get("metrics", {})
    args = run.get("training", {})
    env = run.get("environment", {})
    audit = run.get("dataset_audit", {})
    totals = audit.get("totals", {})

    def metric(name: str) -> str:
        value = metrics.get(name)
        return "—" if value is None else f"{float(value):.4f}"

    return f"""# 2026 LST20 NER Benchmark Result

> Generated automatically by the 2026 modernization pipeline.

## Result

| Metric | Value |
|---|---:|
| Precision | {metric("test_precision")} |
| Recall | {metric("test_recall")} |
| F1 | {metric("test_f1")} |
| Token accuracy | {metric("test_accuracy")} |
| Test loss | {metric("test_loss")} |

## Training configuration

- **Base model:** {args.get("model_name", "—")}
- **Epochs:** {args.get("epochs", "—")}
- **Learning rate:** {args.get("learning_rate", "—")}
- **Train batch size:** {args.get("train_batch_size", "—")}
- **Eval batch size:** {args.get("eval_batch_size", "—")}
- **Max length:** {args.get("max_length", "—")}
- **Seed:** {args.get("seed", "—")}

## Dataset audit

- **Files:** {totals.get("files", "—")}
- **Sentences:** {totals.get("sentences", "—")}
- **Tokens:** {totals.get("tokens", "—")}
- **Entity-labelled tokens:** {totals.get("entity_tokens", "—")}

## Environment

- **Timestamp (UTC):** {env.get("timestamp_utc", "—")}
- **Python:** {env.get("python", "—")}
- **Platform:** {env.get("platform", "—")}

This file records one experiment run. It should not be interpreted as a new state-of-the-art claim unless the benchmark protocol has been independently reviewed.
"""


def write_run_artifacts(output_dir: str | Path, run: dict) -> None:
    output_dir = Path(output_dir)
    save_json(output_dir / "benchmark_run.json", run)
    (output_dir / "BENCHMARK_RESULT.md").write_text(
        render_benchmark_summary(run),
        encoding="utf-8",
    )
