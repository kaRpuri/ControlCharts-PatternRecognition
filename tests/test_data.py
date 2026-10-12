"""Dataset loading, generation and label handling."""

from __future__ import annotations

import collections
from pathlib import Path

import numpy as np
import pytest

from controlcharts.data import (
    CLASS_LABELS,
    CLASS_NAMES,
    NUM_CLASSES,
    SERIES_LENGTH,
    Dataset,
    generate,
    load_csv,
    save_csv,
)


def test_class_labels_are_alphabetical() -> None:
    """The model's output order comes from LabelEncoder, which sorts.

    Reordering this tuple silently remaps every prediction, so it is pinned.
    """
    assert list(CLASS_LABELS) == sorted(CLASS_LABELS)
    assert NUM_CLASSES == 6
    assert set(CLASS_NAMES) == set(CLASS_LABELS)


def test_shipped_dataset_shape_and_balance(shipped_dataset: Dataset) -> None:
    dataset = shipped_dataset

    assert dataset.series.shape == (600, SERIES_LENGTH)
    assert len(dataset) == 600
    assert dict(collections.Counter(dataset.labels)) == dict.fromkeys(CLASS_LABELS, 100)


def test_label_indices_match_class_order(shipped_dataset: Dataset) -> None:
    indices = shipped_dataset.label_indices

    assert indices.min() == 0
    assert indices.max() == NUM_CLASSES - 1
    for index, label in zip(indices, shipped_dataset.labels, strict=True):
        assert CLASS_LABELS[index] == label


def test_generate_shapes_and_balance(tiny_dataset: Dataset) -> None:
    assert tiny_dataset.series.shape == (6 * 8, SERIES_LENGTH)
    assert dict(collections.Counter(tiny_dataset.labels)) == dict.fromkeys(CLASS_LABELS, 8)
    assert np.all(np.isfinite(tiny_dataset.series))


def test_generate_is_reproducible() -> None:
    first = generate(samples_per_class=4, seed=7)
    second = generate(samples_per_class=4, seed=7)
    third = generate(samples_per_class=4, seed=8)

    assert np.array_equal(first.series, second.series)
    assert not np.array_equal(first.series, third.series)


def test_generated_patterns_have_their_defining_shape() -> None:
    """Each family is separable on the statistic that defines it."""
    dataset = generate(samples_per_class=30, seed=1)

    def block(label: str) -> np.ndarray:
        return dataset.series[dataset.labels == label]

    # Trends move between the first and last third; normal series do not.
    def drift(series: np.ndarray) -> np.ndarray:
        return series[:, -20:].mean(axis=1) - series[:, :20].mean(axis=1)

    assert drift(block("Increasing_trend")).mean() > 5
    assert drift(block("Decreasing_trend")).mean() < -5
    assert abs(drift(block("Normal")).mean()) < 1

    # Shifts also move, and by more than the noise band.
    assert drift(block("Upward_shift")).mean() > 5
    assert drift(block("Downward_shift")).mean() < -5

    # Cyclic series swing much wider than normal ones without net drift.
    assert block("Cyclic").std(axis=1).mean() > block("Normal").std(axis=1).mean() * 2
    assert abs(drift(block("Cyclic")).mean()) < 5


def test_round_trip_through_csv(tiny_dataset: Dataset, tmp_path: Path) -> None:
    path = save_csv(tiny_dataset, tmp_path / "out.csv")
    reloaded = load_csv(path)

    assert np.allclose(reloaded.series, tiny_dataset.series)
    assert np.array_equal(reloaded.labels, tiny_dataset.labels)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="series has 2 rows but labels has 1"):
        Dataset(series=np.zeros((2, SERIES_LENGTH)), labels=np.array(["Normal"]))


def test_unknown_label_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    row = ",".join("1.0" for _ in range(SERIES_LENGTH))
    header = ",".join(str(i) for i in range(SERIES_LENGTH))
    path.write_text(f"{header},label\n{row},Sawtooth\n")

    with pytest.raises(ValueError, match="unknown labels"):
        load_csv(path)


def test_wrong_column_count_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "short.csv"
    path.write_text("0,1,2,label\n1.0,2.0,3.0,Normal\n")

    with pytest.raises(ValueError, match="expected 60"):
        load_csv(path)
