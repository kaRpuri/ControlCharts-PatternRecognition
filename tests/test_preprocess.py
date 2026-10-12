"""Normalization, one-hot encoding and the split."""

from __future__ import annotations

import numpy as np
import pytest

from controlcharts.data import NUM_CLASSES, Dataset
from controlcharts.preprocess import normalize, one_hot, prepare


def test_normalize_scales_each_series_to_unit_peak() -> None:
    series = np.array([[1.0, 2.0, 4.0], [10.0, 5.0, 1.0]])
    scaled = normalize(series)

    assert np.allclose(np.abs(scaled).max(axis=1), 1.0)
    assert np.allclose(scaled[0], [0.25, 0.5, 1.0])
    assert np.allclose(scaled[1], [1.0, 0.5, 0.1])


def test_normalize_is_per_series_not_global() -> None:
    """A large series must not shrink a small one sharing the batch."""
    series = np.array([[1.0, 2.0], [1000.0, 2000.0]])
    scaled = normalize(series)
    assert np.allclose(scaled[0], scaled[1])


def test_normalize_handles_negative_values() -> None:
    scaled = normalize(np.array([[-8.0, 2.0, 4.0]]))
    assert np.isclose(np.abs(scaled).max(), 1.0)
    assert scaled[0, 0] == -1.0


def test_normalize_preserves_a_one_dimensional_shape() -> None:
    scaled = normalize(np.array([1.0, 2.0, 4.0]))
    assert scaled.shape == (3,)
    assert np.isclose(scaled.max(), 1.0)


def test_normalize_makes_level_irrelevant() -> None:
    """This is why a model trained around 30 can score a process at 40 mm."""
    shape = np.sin(np.linspace(0, 4 * np.pi, 60))
    assert np.allclose(normalize(shape + 30.0), normalize((shape + 30.0) * 1.3333333333))


def test_normalize_rejects_an_all_zero_series() -> None:
    with pytest.raises(ValueError, match="all-zero"):
        normalize(np.zeros((1, 10)))


def test_one_hot_rows_are_indicator_vectors() -> None:
    encoded = one_hot(np.array([0, 3, 5]))

    assert encoded.shape == (3, NUM_CLASSES)
    assert np.array_equal(encoded.sum(axis=1), np.ones(3))
    assert np.array_equal(encoded.argmax(axis=1), [0, 3, 5])


def test_prepare_split_sizes_and_normalization(tiny_dataset: Dataset) -> None:
    split = prepare(tiny_dataset, test_size=0.25, seed=0)

    assert len(split.x_train) + len(split.x_test) == len(tiny_dataset)
    assert len(split.x_test) == round(0.25 * len(tiny_dataset))
    assert split.num_features == 60
    assert np.allclose(np.abs(split.x_train).max(axis=1), 1.0)
    assert np.allclose(np.abs(split.x_test).max(axis=1), 1.0)


def test_prepare_is_stratified(shipped_dataset: Dataset) -> None:
    """Every class keeps its share in the test half."""
    split = prepare(shipped_dataset, test_size=0.10, seed=101)
    counts = split.y_test.argmax(axis=1)

    assert len(split.x_test) == 60
    assert sorted(np.bincount(counts, minlength=NUM_CLASSES)) == [10] * NUM_CLASSES


def test_prepare_is_reproducible(tiny_dataset: Dataset) -> None:
    first = prepare(tiny_dataset, test_size=0.25, seed=42)
    second = prepare(tiny_dataset, test_size=0.25, seed=42)
    third = prepare(tiny_dataset, test_size=0.25, seed=43)

    assert np.array_equal(first.x_train, second.x_train)
    assert not np.array_equal(first.x_train, third.x_train)


def test_train_and_test_sets_are_disjoint(tiny_dataset: Dataset) -> None:
    split = prepare(tiny_dataset, test_size=0.25, seed=0)
    train_rows = {row.tobytes() for row in split.x_train}
    assert not any(row.tobytes() in train_rows for row in split.x_test)
