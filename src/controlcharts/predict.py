"""Classify process measurements with a trained model.

A CSV of raw measurements is read, split into subgroups, reduced to its
X-bar and R series, and each of those two series is classified. This is the
pipeline the give_output function in OneFunction.ipynb performed inline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from .data import CLASS_LABELS, CLASS_NAMES, SERIES_LENGTH
from .preprocess import normalize
from .spc import Capability, ControlChart, capability, control_chart, violations

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """One classified series.

    Attributes:
        chart: Which chart the series came from, "x-bar" or "range".
        pattern: Predicted pattern, as a display name.
        confidence: Probability the model gave the predicted pattern.
        probabilities: Probability per class, in CLASS_LABELS order.
    """

    chart: str
    pattern: str
    confidence: float
    probabilities: np.ndarray

    def ranked(self) -> list[tuple[str, float]]:
        """All classes with their probabilities, most likely first."""
        names = [CLASS_NAMES[label] for label in CLASS_LABELS]
        order = np.argsort(self.probabilities)[::-1]
        return [(names[i], float(self.probabilities[i])) for i in order]


@dataclass
class ProcessReport:
    """Everything the predict command found for one characteristic.

    Attributes:
        source: The CSV that was read.
        chart: Control chart statistics.
        capability: Capability indices, or None if no specification was given.
        predictions: One prediction per chart.
        violations: Out of control and capability findings.
    """

    source: str
    chart: ControlChart
    capability: Capability | None
    predictions: list[Prediction]
    violations: list[str]

    @property
    def in_control(self) -> bool:
        """Whether the process passed every check."""
        return not self.violations


def load_measurements(path: str | Path) -> np.ndarray:
    """Read a CSV of process measurements into subgroups.

    The file is laid out as the notebook expected it: each row is one
    repeated measurement across the run, so a column is one subgroup. The
    array is returned transposed, with subgroups as rows.

    Args:
        path: Path to the CSV.

    Returns:
        Subgroups as rows, shape (num_subgroups, subgroup_size).

    Raises:
        ValueError: If the file holds no numeric measurements.
    """
    frame = pd.read_csv(path)
    values = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    if values.size == 0 or np.all(np.isnan(values)):
        raise ValueError(f"{path} holds no numeric measurements")
    if np.any(np.isnan(values)):
        raise ValueError(f"{path} holds missing or non-numeric values")

    return values.T


def classify_series(model: keras.Model, series: np.ndarray, chart: str) -> Prediction:
    """Classify one series of process statistics.

    Args:
        model: The trained classifier.
        series: The series to classify, length 60.
        chart: Label for which chart this is, used in the result.

    Returns:
        The prediction.

    Raises:
        ValueError: If the series is not 60 points long.
    """
    series = np.asarray(series, dtype=float).ravel()
    if series.size != SERIES_LENGTH:
        raise ValueError(
            f"expected {SERIES_LENGTH} points for the {chart} chart, got {series.size}"
        )

    probabilities = model.predict(normalize(series).reshape(1, -1), verbose=0)[0]
    index = int(probabilities.argmax())

    return Prediction(
        chart=chart,
        pattern=CLASS_NAMES[CLASS_LABELS[index]],
        confidence=float(probabilities[index]),
        probabilities=probabilities,
    )


def analyze(
    model: keras.Model,
    path: str | Path,
    usl: float | None = None,
    lsl: float | None = None,
) -> ProcessReport:
    """Run the full pipeline on one measurement file.

    Args:
        model: The trained classifier.
        path: CSV of process measurements.
        usl: Upper specification limit, or None to skip capability.
        lsl: Lower specification limit, or None to skip capability.

    Returns:
        The report.
    """
    subgroups = load_measurements(path)
    chart = control_chart(subgroups)

    capability_result = None
    if usl is not None and lsl is not None:
        capability_result = capability(chart, usl, lsl)

    predictions = [
        classify_series(model, chart.subgroup_means, "x-bar"),
        classify_series(model, chart.subgroup_ranges, "range"),
    ]

    report = ProcessReport(
        source=str(path),
        chart=chart,
        capability=capability_result,
        predictions=predictions,
        violations=violations(chart, capability_result),
    )
    logger.info(
        "%s: x-bar %s (%.3f), range %s (%.3f), %d finding(s)",
        path,
        predictions[0].pattern,
        predictions[0].confidence,
        predictions[1].pattern,
        predictions[1].confidence,
        len(report.violations),
    )
    return report
