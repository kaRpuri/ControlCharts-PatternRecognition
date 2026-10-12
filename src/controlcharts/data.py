"""The six control chart patterns, and loading or generating series of them.

The shipped dataset is the UCI Synthetic Control Chart Time Series set: 600
series of 60 points, 100 per class, in class order. The generator here
reproduces the same six pattern families from the equations in that dataset's
documentation, which is what lets the tests build small samples without
shipping extra data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SERIES_LENGTH = 60

# Alphabetical order. This is the order sklearn's LabelEncoder produces from
# the dataset labels, and therefore the order of the trained model's outputs.
# It must not be reordered without retraining.
CLASS_LABELS: tuple[str, ...] = (
    "Cyclic",
    "Decreasing_trend",
    "Downward_shift",
    "Increasing_trend",
    "Normal",
    "Upward_shift",
)

# Human readable names used in reports and figures.
CLASS_NAMES: dict[str, str] = {
    "Cyclic": "Cyclic",
    "Decreasing_trend": "Decreasing trend",
    "Downward_shift": "Downward shift",
    "Increasing_trend": "Increasing trend",
    "Normal": "Normal",
    "Upward_shift": "Upward shift",
}

NUM_CLASSES = len(CLASS_LABELS)

# Generator parameters, following the UCI dataset definition.
BASE_MEAN = 30.0
NOISE_RANGE = (-3.0, 3.0)
CYCLE_AMPLITUDE = (10.0, 15.0)
CYCLE_PERIOD = (10.0, 15.0)
TREND_GRADIENT = (0.2, 0.5)
SHIFT_MAGNITUDE = (7.5, 20.0)


@dataclass(frozen=True)
class Dataset:
    """A set of labelled control chart series.

    Attributes:
        series: Raw measurement series, shape (n, 60).
        labels: Class label per series, shape (n,), values from CLASS_LABELS.
    """

    series: np.ndarray
    labels: np.ndarray

    def __post_init__(self) -> None:
        if len(self.series) != len(self.labels):
            raise ValueError(
                f"series has {len(self.series)} rows but labels has {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.series)

    @property
    def label_indices(self) -> np.ndarray:
        """Class index per series, matching the order of CLASS_LABELS."""
        lookup = {name: i for i, name in enumerate(CLASS_LABELS)}
        return np.array([lookup[label] for label in self.labels])


def load_csv(path: str | Path) -> Dataset:
    """Load the UCI synthetic control chart CSV shipped in ``data/``.

    The file carries two leading index columns, written by the export tool as
    unnamed, and a trailing ``label`` column.

    Args:
        path: Path to the CSV.

    Returns:
        The dataset.

    Raises:
        ValueError: If the file does not have 60 measurement columns, or holds
            a label outside CLASS_LABELS.
    """
    frame = pd.read_csv(path)
    frame = frame.drop(columns=[c for c in frame.columns if c.startswith("nan")])

    if "label" not in frame.columns:
        raise ValueError(f"{path} has no label column")
    labels = frame["label"].to_numpy()

    unknown = sorted(set(labels) - set(CLASS_LABELS))
    if unknown:
        raise ValueError(f"{path} holds unknown labels {unknown}")

    series = frame.drop(columns=["label"]).to_numpy(dtype=float)
    if series.shape[1] != SERIES_LENGTH:
        raise ValueError(f"{path} has {series.shape[1]} measurement columns, expected 60")

    return Dataset(series=series, labels=labels)


def generate(
    samples_per_class: int = 100,
    length: int = SERIES_LENGTH,
    seed: int | None = None,
) -> Dataset:
    """Generate synthetic control chart series, one block per class.

    Args:
        samples_per_class: Series to generate for each of the six classes.
        length: Points per series.
        seed: Seed for the generator. Pass an int for a reproducible set.

    Returns:
        A dataset of ``6 * samples_per_class`` series, in class order.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(length, dtype=float)

    blocks: list[np.ndarray] = []
    labels: list[str] = []

    for label in CLASS_LABELS:
        block = BASE_MEAN + rng.uniform(*NOISE_RANGE, size=(samples_per_class, length))

        if label == "Cyclic":
            amplitude = rng.uniform(*CYCLE_AMPLITUDE, size=(samples_per_class, 1))
            period = rng.uniform(*CYCLE_PERIOD, size=(samples_per_class, 1))
            block = block + amplitude * np.sin(2 * np.pi * steps / period)

        elif label in ("Increasing_trend", "Decreasing_trend"):
            gradient = rng.uniform(*TREND_GRADIENT, size=(samples_per_class, 1))
            sign = 1.0 if label == "Increasing_trend" else -1.0
            block = block + sign * gradient * steps

        elif label in ("Upward_shift", "Downward_shift"):
            magnitude = rng.uniform(*SHIFT_MAGNITUDE, size=(samples_per_class, 1))
            shift_at = rng.integers(length // 3, 2 * length // 3, size=(samples_per_class, 1))
            sign = 1.0 if label == "Upward_shift" else -1.0
            block = block + sign * magnitude * (steps >= shift_at)

        blocks.append(block)
        labels.extend([label] * samples_per_class)

    return Dataset(series=np.vstack(blocks), labels=np.array(labels))


def save_csv(dataset: Dataset, path: str | Path) -> Path:
    """Write a dataset in the same column layout as the shipped CSV.

    Args:
        dataset: The dataset to write.
        path: Destination file.

    Returns:
        The path written.
    """
    frame = pd.DataFrame(dataset.series, columns=[str(i) for i in range(dataset.series.shape[1])])
    frame["label"] = dataset.labels
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path
