"""Control chart statistics and capability.

Several tests pin behaviour that differs from the original notebook. Those
differences are corrections, and each is named in the test it appears in.
"""

from __future__ import annotations

import numpy as np
import pytest

from controlcharts.spc import (
    CONTROL_CONSTANTS,
    capability,
    control_chart,
    subgroup_statistics,
    violations,
)


def test_subgroup_statistics() -> None:
    measurements = np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 10.0, 10.0, 10.0]])
    means, ranges = subgroup_statistics(measurements)

    assert np.allclose(means, [2.5, 10.0])
    assert np.allclose(ranges, [3.0, 0.0])


def test_control_limits_follow_the_shewhart_constants(subgroups: np.ndarray) -> None:
    chart = control_chart(subgroups)
    a2, d3, d4, d2 = CONTROL_CONSTANTS[4]

    assert np.isclose(chart.ucl_x, chart.center_x + a2 * chart.center_r)
    assert np.isclose(chart.lcl_x, chart.center_x - a2 * chart.center_r)
    assert np.isclose(chart.ucl_r, d4 * chart.center_r)
    assert np.isclose(chart.lcl_r, d3 * chart.center_r)
    assert np.isclose(chart.sigma, chart.center_r / d2)


def test_centre_lines_are_the_plain_means(subgroups: np.ndarray) -> None:
    chart = control_chart(subgroups)
    assert np.isclose(chart.center_x, subgroups.mean())
    assert np.isclose(chart.center_r, (subgroups.max(axis=1) - subgroups.min(axis=1)).mean())


def test_unknown_subgroup_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="no control constants"):
        control_chart(np.zeros((10, 11)))


def test_capability_is_dimensionless() -> None:
    """Correction: the notebook's Cpk was a raw distance, not an index.

    With the specification exactly six sigma wide and the mean centred,
    Cp and Cpk are both 1 by definition.
    """
    chart = control_chart(np.array([[9.0, 11.0]] * 40))
    sigma = chart.sigma
    result = capability(chart, usl=10 + 3 * sigma, lsl=10 - 3 * sigma)

    assert np.isclose(result.cp, 1.0)
    assert np.isclose(result.cpk, 1.0)


def test_cpk_falls_below_cp_when_the_mean_is_off_centre() -> None:
    chart = control_chart(np.array([[9.0, 11.0]] * 40))
    sigma = chart.sigma
    centred = capability(chart, usl=10 + 4 * sigma, lsl=10 - 4 * sigma)
    offset = capability(chart, usl=10 + 6 * sigma, lsl=10 - 2 * sigma)

    assert np.isclose(centred.cp, offset.cp)
    assert offset.cpk < centred.cpk


def test_cpk_is_negative_when_the_mean_is_out_of_specification() -> None:
    chart = control_chart(np.array([[9.0, 11.0]] * 40))
    result = capability(chart, usl=9.5, lsl=8.0)
    assert result.cpk < 0


def test_capability_rejects_bad_limits(subgroups: np.ndarray) -> None:
    chart = control_chart(subgroups)
    with pytest.raises(ValueError, match="must be greater than"):
        capability(chart, usl=1.0, lsl=2.0)


def test_capability_rejects_zero_sigma() -> None:
    chart = control_chart(np.ones((10, 4)))
    with pytest.raises(ValueError, match="sigma is zero"):
        capability(chart, usl=2.0, lsl=0.0)


def test_a_stable_process_triggers_no_run_rules() -> None:
    """A process with no assignable cause shows no runs and few breaches.

    An isolated point outside the limits is expected even when nothing is
    wrong, since the limits sit near three sigma, so this bounds the count
    rather than demanding zero.
    """
    rng = np.random.default_rng(0)
    chart = control_chart(10.0 + rng.normal(0, 0.05, size=(60, 4)))
    found = violations(chart)

    assert not any("runs above" in line or "runs below" in line for line in found)
    assert len(found) <= 3, found


def test_every_out_of_limit_point_is_found() -> None:
    """Correction: the notebook scanned only the first and last subgroup.

    Its loop read `for val in (0, len(xbar) - 1)`, which iterates a two
    element tuple, so a spike anywhere in between was invisible.
    """
    rng = np.random.default_rng(0)
    measurements = 10.0 + rng.normal(0, 0.05, size=(60, 4))
    measurements[30] += 5.0
    chart = control_chart(measurements)

    found = violations(chart)
    assert any("X-bar subgroup 30" in line for line in found)


def test_a_zero_range_is_not_an_r_chart_violation() -> None:
    """Correction: for subgroup sizes below 7 the R chart has no lower limit.

    D3 is zero there, so the notebook's `value <= lcl_r` test flagged every
    subgroup whose measurements happened to be identical.
    """
    rng = np.random.default_rng(0)
    measurements = 10.0 + rng.normal(0, 0.05, size=(60, 4))
    measurements[10] = 10.0  # four identical readings, range exactly zero
    chart = control_chart(measurements)

    assert chart.lcl_r == 0.0
    assert not any("R subgroup 10" in line for line in violations(chart))


def test_a_sustained_run_above_the_centre_line_is_found() -> None:
    measurements = np.zeros((60, 4))
    measurements[:50] = 11.0
    measurements[50:] = 1.0
    measurements += np.random.default_rng(0).normal(0, 0.01, size=(60, 4))
    chart = control_chart(measurements)

    assert any("runs above its centre line" in line for line in violations(chart))


def test_capability_findings_are_reported(subgroups: np.ndarray) -> None:
    chart = control_chart(subgroups)
    tight = capability(chart, usl=chart.center_x + 0.01, lsl=chart.center_x - 0.01)
    found = violations(chart, tight)

    assert any("Cp is" in line for line in found)
