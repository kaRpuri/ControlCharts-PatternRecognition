"""Loading measurements, classifying a series, and the full analyze path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from controlcharts.data import CLASS_NAMES
from controlcharts.model import build_model
from controlcharts.predict import analyze, classify_series, load_measurements

VALID_PATTERNS = set(CLASS_NAMES.values())


def test_load_measurements_transposes_into_subgroups(tiny_csv: Path) -> None:
    """The CSV holds repeats as rows, so a column is one subgroup."""
    subgroups = load_measurements(tiny_csv)
    assert subgroups.shape == (60, 4)
    assert np.all(np.isfinite(subgroups))


def test_load_measurements_reads_quoted_numbers(real_csv: Path) -> None:
    """The production files quote every value."""
    subgroups = load_measurements(real_csv)
    assert subgroups.shape == (60, 4)
    assert subgroups.dtype == float


def test_load_measurements_rejects_missing_values(tmp_path: Path) -> None:
    path = tmp_path / "holes.csv"
    path.write_text("0,1\n1.0,\n2.0,3.0\n")
    with pytest.raises(ValueError, match="missing or non-numeric"):
        load_measurements(path)


def test_classify_series_returns_a_known_pattern() -> None:
    model = build_model()
    prediction = classify_series(model, np.linspace(1.0, 2.0, 60), "x-bar")

    assert prediction.chart == "x-bar"
    assert prediction.pattern in VALID_PATTERNS
    assert 0.0 <= prediction.confidence <= 1.0
    assert np.isclose(prediction.probabilities.sum(), 1.0, atol=1e-5)


def test_classify_series_rejects_the_wrong_length() -> None:
    model = build_model()
    with pytest.raises(ValueError, match="expected 60 points"):
        classify_series(model, np.ones(10), "x-bar")


def test_ranked_is_sorted_and_complete() -> None:
    model = build_model()
    ranked = classify_series(model, np.linspace(1.0, 2.0, 60), "x-bar").ranked()

    assert len(ranked) == len(VALID_PATTERNS)
    assert {name for name, _ in ranked} == VALID_PATTERNS
    assert ranked == sorted(ranked, key=lambda pair: pair[1], reverse=True)


def test_analyze_on_a_tiny_csv(tiny_csv: Path) -> None:
    model = build_model()
    report = analyze(model, tiny_csv, usl=25.5, lsl=24.5)

    assert report.chart.subgroup_size == 4
    assert len(report.chart.subgroup_means) == 60
    assert {p.chart for p in report.predictions} == {"x-bar", "range"}
    assert report.capability is not None
    assert isinstance(report.in_control, bool)


def test_analyze_without_a_specification_skips_capability(tiny_csv: Path) -> None:
    report = analyze(build_model(), tiny_csv)
    assert report.capability is None


def test_analyze_reproduces_the_notebook_statistics(real_csv: Path) -> None:
    """The chart statistics match the values printed in OneFunction.ipynb.

    The notebook reported a grand mean of 39.97266666666667 and a mean range
    of 0.02633333333333212 for this file.
    """
    report = analyze(build_model(), real_csv)

    assert report.chart.center_x == pytest.approx(39.97266666666667, abs=1e-12)
    assert report.chart.center_r == pytest.approx(0.02633333333333212, abs=1e-12)
