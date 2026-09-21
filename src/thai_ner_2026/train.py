"""Fine-tune WangchanBERTa for LST20 NER using the 2026 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from .audit import audit_corpus
from .data import load_lst20_dataset_dict
from .labels import ID2LABEL, LABEL2ID, LST20_NER_TAGS
from .metrics import build_compute_metrics
from .preprocessing import tokenize_and_align_batch
from .reporting import collect_environment, write_run_artifacts

DEFAULT_MODEL = "airesearch/wangchanberta-base-att-spm-uncased"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune WangchanBERTa on a locally downloaded LST20 corpus."
    )
    parser.add_argument("--data-dir", required=True, help="Path to extracted LST20Corpus")
    parser.add_argument("--output-dir", default="outputs/wangchanberta-lst20-2026")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument(
        "--fp16", action="store_true", help="Use FP16 mixed precision on supported hardware."
    )
    precision.add_argument(
        "--bf16", action="store_true", help="Use BF16 mixed precision on supported hardware."
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for lower activation memory during training.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Optional checkpoint directory to resume training from.",
    )
    parser.add_argument(
        "--smoke-limit",
        type=int,
        default=None,
        help="Optional number of examples per split for a quick local smoke run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Auditing LST20 corpus before training...")
    dataset_audit = audit_corpus(args.data_dir)

    dataset = load_lst20_dataset_dict(
        args.data_dir,
        train_limit=args.smoke_limit,
        validation_limit=args.smoke_limit,
        test_limit=args.smoke_limit,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            "A fast WangchanBERTa tokenizer is required for word/subtoken alignment."
        )

    tokenized = dataset.map(
        lambda batch: tokenize_and_align_batch(
            batch,
            tokenizer=tokenizer,
            label2id=LABEL2ID,
            max_length=args.max_length,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing and aligning NER labels",
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=len(LST20_NER_TAGS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=args.logging_steps,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(LST20_NER_TAGS),
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    print(test_metrics)

    trainer.save_model(str(output_dir / "best-model"))
    tokenizer.save_pretrained(str(output_dir / "best-model"))

    run_record = {
        "project_era": "2026-modernization",
        "dataset": "LST20",
        "training": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "logging_steps": args.logging_steps,
            "fp16": args.fp16,
            "bf16": args.bf16,
            "gradient_checkpointing": args.gradient_checkpointing,
            "seed": args.seed,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "smoke_limit": args.smoke_limit,
        },
        "dataset_audit": dataset_audit,
        "metrics": {
            key: float(value) if isinstance(value, (int, float)) else value
            for key, value in test_metrics.items()
        },
        "environment": collect_environment(),
    }
    write_run_artifacts(output_dir, run_record)

    print(f"Benchmark artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
