"""Entity-level evaluation helpers for LST20."""

from __future__ import annotations

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score


def lst20_label_to_iob2(label: str) -> str:
    """Convert LST20 B_/I_/E_ labels to the IOB2 form expected by seqeval."""
    if label == "O":
        return label
    prefix, entity = label.split("_", 1)
    if prefix == "E":
        prefix = "I"
    return f"{prefix}-{entity}"


def build_compute_metrics(label_list: list[str]):
    """Create a Trainer compute_metrics callback with entity-level scores."""

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []

        for prediction, label_ids in zip(predictions, labels):
            pred_seq: list[str] = []
            label_seq: list[str] = []
            for pred_id, label_id in zip(prediction, label_ids):
                if label_id == -100:
                    continue
                pred_seq.append(lst20_label_to_iob2(label_list[int(pred_id)]))
                label_seq.append(lst20_label_to_iob2(label_list[int(label_id)]))
            true_predictions.append(pred_seq)
            true_labels.append(label_seq)

        return {
            "precision": precision_score(true_labels, true_predictions, zero_division=0),
            "recall": recall_score(true_labels, true_predictions, zero_division=0),
            "f1": f1_score(true_labels, true_predictions, zero_division=0),
            "accuracy": accuracy_score(true_labels, true_predictions),
        }

    return compute_metrics
