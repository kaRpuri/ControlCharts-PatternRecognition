"""Scoring a trained model: accuracy, per class report, confusion matrix."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import keras
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import CLASS_LABELS, CLASS_NAMES

logger = logging.getLogger(__name__)

DISPLAY_NAMES = [CLASS_NAMES[label] for label in CLASS_LABELS]


@dataclass
class Evaluation:
    """How a model scored on a labelled set.

    Attributes:
        accuracy: Overall accuracy, between 0 and 1.
        confusion: Confusion matrix, shape (6, 6), true class by row.
        report: Per class precision, recall and f1, keyed by class name.
        true_indices: True class index per sample.
        predicted_indices: Predicted class index per sample.
        class_names: Display names in class index order.
    """

    accuracy: float
    confusion: np.ndarray
    report: dict[str, dict[str, float]]
    true_indices: np.ndarray
    predicted_indices: np.ndarray
    class_names: list[str]

    def per_class_accuracy(self) -> dict[str, float]:
        """Recall for each class, which is its row of the confusion matrix."""
        totals = self.confusion.sum(axis=1)
        return {
            name: float(self.confusion[i, i] / totals[i]) if totals[i] else 0.0
            for i, name in enumerate(self.class_names)
        }


def evaluate(model: keras.Model, x: np.ndarray, y: np.ndarray) -> Evaluation:
    """Score a model on normalized series with one-hot labels.

    Args:
        model: The trained model.
        x: Normalized series, shape (n, 60).
        y: One-hot labels, shape (n, 6).

    Returns:
        The evaluation.
    """
    probabilities = model.predict(x, verbose=0)
    predicted = probabilities.argmax(axis=1)
    true = y.argmax(axis=1)

    report = classification_report(
        true,
        predicted,
        labels=list(range(len(CLASS_LABELS))),
        target_names=DISPLAY_NAMES,
        output_dict=True,
        zero_division=0,
    )

    evaluation = Evaluation(
        accuracy=float(accuracy_score(true, predicted)),
        confusion=confusion_matrix(true, predicted, labels=list(range(len(CLASS_LABELS)))),
        report={k: v for k, v in report.items() if k in DISPLAY_NAMES},
        true_indices=true,
        predicted_indices=predicted,
        class_names=DISPLAY_NAMES,
    )
    logger.info("accuracy %.4f on %d samples", evaluation.accuracy, len(true))
    return evaluation


def format_report(evaluation: Evaluation) -> str:
    """Render the per class report as a fixed width table.

    Args:
        evaluation: The evaluation to render.

    Returns:
        The table as a string, without a trailing newline.
    """
    header = f"{'class':<18}{'precision':>10}{'recall':>9}{'f1':>7}{'support':>9}"
    lines = [header, "-" * len(header)]
    for name in evaluation.class_names:
        row = evaluation.report.get(name)
        if row is None:
            continue
        lines.append(
            f"{name:<18}{row['precision']:>10.3f}{row['recall']:>9.3f}"
            f"{row['f1-score']:>7.3f}{int(row['support']):>9d}"
        )
    lines.append("-" * len(header))
    lines.append(f"{'accuracy':<18}{evaluation.accuracy:>10.4f}")
    return "\n".join(lines)
