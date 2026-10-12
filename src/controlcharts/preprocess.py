"""Normalization and the train/test split.

Each series is divided by its own largest absolute value, so a series is
described by its shape rather than its absolute level. That is what lets a
model trained on the synthetic set, which sits around a mean of 30, score a
real process running at 40 mm or 22 mm without rescaling.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split

from .data import NUM_CLASSES, Dataset


def normalize(series: np.ndarray) -> np.ndarray:
    """Scale each series by its own maximum absolute value.

    Args:
        series: Raw series, shape (n, length) or (length,).

    Returns:
        Normalized series, same shape and dtype float.

    Raises:
        ValueError: If any series is all zeros, which cannot be scaled.
    """
    values = np.atleast_2d(np.asarray(series, dtype=float))
    peak = np.abs(values).max(axis=1, keepdims=True)

    if np.any(peak == 0):
        raise ValueError("cannot normalize an all-zero series")

    scaled = values / peak
    return scaled.reshape(np.shape(series)) if np.ndim(series) == 1 else scaled


def one_hot(label_indices: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Expand class indices into one-hot rows.

    Args:
        label_indices: Class index per sample, shape (n,).
        num_classes: Number of classes.

    Returns:
        One-hot array, shape (n, num_classes), dtype float32.
    """
    encoded = np.zeros((len(label_indices), num_classes), dtype=np.float32)
    encoded[np.arange(len(label_indices)), label_indices] = 1.0
    return encoded


@dataclass(frozen=True)
class Split:
    """A prepared train and test split.

    Attributes:
        x_train: Normalized training series, shape (n_train, 60).
        x_test: Normalized test series, shape (n_test, 60).
        y_train: One-hot training labels, shape (n_train, 6).
        y_test: One-hot test labels, shape (n_test, 6).
    """

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

    @property
    def num_features(self) -> int:
        """Points per series."""
        return self.x_train.shape[1]


def prepare(dataset: Dataset, test_size: float = 0.10, seed: int = 101) -> Split:
    """Normalize a dataset and split it into train and test sets.

    The split is stratified, so every class keeps its share in both halves.
    The original notebook split without stratifying and without a fixed
    shuffle seed, which made its exact split unrecoverable.

    Args:
        dataset: The labelled series to prepare.
        test_size: Fraction held out for testing.
        seed: Seed for the split.

    Returns:
        The prepared split.
    """
    x = normalize(dataset.series)
    y = one_hot(dataset.label_indices)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=dataset.label_indices,
    )
    return Split(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)
