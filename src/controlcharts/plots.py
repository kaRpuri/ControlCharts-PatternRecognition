"""Figures: example patterns, training curves, confusion matrix, control charts."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

from .data import CLASS_LABELS, CLASS_NAMES, Dataset  # noqa: E402
from .evaluate import Evaluation  # noqa: E402
from .predict import ProcessReport  # noqa: E402

logger = logging.getLogger(__name__)


def _finish(fig: plt.Figure, path: str | Path) -> Path:
    """Tidy up, save and close a figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def plot_example_patterns(dataset: Dataset, path: str | Path, seed: int = 0) -> Path:
    """One example series per class, in a grid.

    Args:
        dataset: Labelled series to draw from.
        path: Destination image.
        seed: Seed for picking which example of each class to show.

    Returns:
        The path written.
    """
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

    for ax, label in zip(axes.ravel(), CLASS_LABELS, strict=True):
        candidates = np.flatnonzero(dataset.labels == label)
        series = dataset.series[rng.choice(candidates)]
        ax.plot(series, color="black", linewidth=1)
        ax.axhline(series.mean(), color="tab:blue", linestyle="--", linewidth=1)
        ax.set_title(CLASS_NAMES[label])
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("sample")
    for ax in axes[:, 0]:
        ax.set_ylabel("measurement")

    fig.suptitle("Control chart patterns", fontsize=13)
    return _finish(fig, path)


def plot_training_curves(history: dict[str, list[float]], path: str | Path) -> Path:
    """Accuracy and loss per epoch, training against validation.

    Args:
        history: Per epoch metrics from training.
        path: Destination image.

    Returns:
        The path written.
    """
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))

    ax_acc.plot(epochs, history["accuracy"], label="train")
    ax_acc.plot(epochs, history["val_accuracy"], label="validation")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    ax_loss.plot(epochs, history["loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("categorical cross-entropy")
    ax_loss.set_title("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    return _finish(fig, path)


def plot_confusion_matrix(evaluation: Evaluation, path: str | Path) -> Path:
    """Confusion matrix as an annotated heatmap, true class by row.

    Args:
        evaluation: The scored evaluation.
        path: Destination image.

    Returns:
        The path written.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        evaluation.confusion,
        annot=True,
        fmt="g",
        cmap="Blues",
        cbar=False,
        xticklabels=evaluation.class_names,
        yticklabels=evaluation.class_names,
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix, accuracy {evaluation.accuracy:.1%}")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return _finish(fig, path)


def plot_control_charts(report: ProcessReport, path: str | Path, name: str = "") -> Path:
    """X-bar and R charts with their control limits and specification limits.

    Args:
        report: The analyzed process.
        path: Destination image.
        name: Characteristic name for the title.

    Returns:
        The path written.
    """
    chart = report.chart
    fig, (ax_x, ax_r) = plt.subplots(2, figsize=(11, 8), sharex=True)

    ax_x.plot(chart.subgroup_means, linestyle="-", marker="o", markersize=3, color="black")
    ax_x.axhline(chart.ucl_x, color="red", linestyle="--", label="control limits")
    ax_x.axhline(chart.lcl_x, color="red", linestyle="--")
    ax_x.axhline(chart.center_x, color="blue", label="centre line")
    if report.capability is not None:
        ax_x.axhline(report.capability.usl, color="green", linestyle=":", label="specification")
        ax_x.axhline(report.capability.lsl, color="green", linestyle=":")
    outside = [
        i for i, v in enumerate(chart.subgroup_means) if v >= chart.ucl_x or v <= chart.lcl_x
    ]
    if outside:
        ax_x.plot(
            outside,
            chart.subgroup_means[outside],
            "o",
            color="red",
            markersize=6,
            label="out of control",
        )
    ax_x.set_title(f"X-bar chart{f' ({name})' if name else ''}")
    ax_x.set_ylabel("subgroup mean")
    ax_x.grid(True, alpha=0.3)
    ax_x.legend(loc="best", fontsize=8)

    ax_r.plot(chart.subgroup_ranges, linestyle="-", marker="o", markersize=3, color="black")
    ax_r.axhline(chart.ucl_r, color="red", linestyle="--")
    # Below subgroup size 7 the D3 constant is zero, so there is no lower
    # limit to draw and a range of zero is not a violation.
    if chart.lcl_r > 0:
        ax_r.axhline(chart.lcl_r, color="red", linestyle="--")
    ax_r.axhline(chart.center_r, color="blue")
    ax_r.set_title("R chart")
    ax_r.set_xlabel("subgroup")
    ax_r.set_ylabel("subgroup range")
    ax_r.grid(True, alpha=0.3)

    return _finish(fig, path)
