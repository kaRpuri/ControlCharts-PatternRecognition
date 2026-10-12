"""X-bar and R control chart statistics and process capability.

Ported from the give_output function in OneFunction.ipynb. Three defects in
that function are corrected here, each noted at the point it occurs:

1. Every rule scan was written ``for val in (0, len(xbar) - 1)``, which
   iterates a two element tuple rather than a range, so only the first and
   last subgroup were ever examined. The local name ``range`` shadowed the
   builtin, which is what forced that workaround.
2. Cp divided the specification width by the control limit width. Control
   limits describe the process spread of subgroup means, not of individual
   parts, so this overstates Cp. The estimate here uses the standard
   sigma = Rbar / d2.
3. Cpk was ``min(USL - mean, mean - LSL)``, a distance in millimetres with
   no division by 3 sigma, so it was not a capability index at all and the
   Cpk < 0 test compared a raw distance against zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Shewhart control chart constants by subgroup size. A2 sets the X-bar
# limits, D3 and D4 the R limits, and d2 converts Rbar into a sigma estimate.
CONTROL_CONSTANTS: dict[int, tuple[float, float, float, float]] = {
    2: (1.880, 0.0, 3.267, 1.128),
    3: (1.023, 0.0, 2.574, 1.693),
    4: (0.729, 0.0, 2.282, 2.059),
    5: (0.577, 0.0, 2.114, 2.326),
    6: (0.483, 0.0, 2.004, 2.534),
    7: (0.419, 0.076, 1.924, 2.704),
    8: (0.373, 0.136, 1.864, 2.847),
    9: (0.337, 0.184, 1.816, 2.970),
    10: (0.308, 0.223, 1.777, 3.078),
}

# Fraction of points on one side of the centre line that counts as a run.
RUN_FRACTION = 0.7


@dataclass(frozen=True)
class ControlChart:
    """X-bar and R chart statistics for one measured characteristic.

    Attributes:
        subgroup_means: Mean of each subgroup, the X-bar series.
        subgroup_ranges: Range of each subgroup, the R series.
        center_x: Grand mean, the X-bar centre line.
        center_r: Mean range, the R centre line.
        ucl_x: Upper control limit of the X-bar chart.
        lcl_x: Lower control limit of the X-bar chart.
        ucl_r: Upper control limit of the R chart.
        lcl_r: Lower control limit of the R chart.
        sigma: Process sigma estimated as Rbar / d2.
        subgroup_size: Measurements per subgroup.
    """

    subgroup_means: np.ndarray
    subgroup_ranges: np.ndarray
    center_x: float
    center_r: float
    ucl_x: float
    lcl_x: float
    ucl_r: float
    lcl_r: float
    sigma: float
    subgroup_size: int


@dataclass(frozen=True)
class Capability:
    """Process capability against a specification.

    Attributes:
        cp: Specification width over six sigma. Below 1 means the spread
            alone cannot hold the tolerance.
        cpk: Cp penalized for being off centre. Below 1 means the process as
            centred cannot hold the tolerance; below 0 means the mean is
            outside specification.
        usl: Upper specification limit.
        lsl: Lower specification limit.
    """

    cp: float
    cpk: float
    usl: float
    lsl: float


def subgroup_statistics(measurements: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and range of each subgroup.

    Args:
        measurements: Subgroups as rows, shape (num_subgroups, subgroup_size).

    Returns:
        ``(means, ranges)``, each shape (num_subgroups,).
    """
    values = np.asarray(measurements, dtype=float)
    return values.mean(axis=1), values.max(axis=1) - values.min(axis=1)


def control_chart(measurements: np.ndarray) -> ControlChart:
    """Compute X-bar and R chart limits from subgrouped measurements.

    Args:
        measurements: Subgroups as rows, shape (num_subgroups, subgroup_size).

    Returns:
        The chart statistics.

    Raises:
        ValueError: If the subgroup size has no tabulated constants.
    """
    values = np.asarray(measurements, dtype=float)
    subgroup_size = values.shape[1]

    if subgroup_size not in CONTROL_CONSTANTS:
        raise ValueError(
            f"no control constants for subgroup size {subgroup_size}; "
            f"tabulated sizes are {sorted(CONTROL_CONSTANTS)}"
        )

    a2, d3, d4, d2 = CONTROL_CONSTANTS[subgroup_size]
    means, ranges = subgroup_statistics(values)
    center_x, center_r = float(means.mean()), float(ranges.mean())

    return ControlChart(
        subgroup_means=means,
        subgroup_ranges=ranges,
        center_x=center_x,
        center_r=center_r,
        ucl_x=center_x + a2 * center_r,
        lcl_x=center_x - a2 * center_r,
        ucl_r=d4 * center_r,
        lcl_r=d3 * center_r,
        # Correction 2: sigma from Rbar / d2, not from the control limit width.
        sigma=center_r / d2,
        subgroup_size=subgroup_size,
    )


def capability(chart: ControlChart, usl: float, lsl: float) -> Capability:
    """Process capability indices against a specification.

    Args:
        chart: Chart statistics for the characteristic.
        usl: Upper specification limit.
        lsl: Lower specification limit.

    Returns:
        Cp and Cpk.

    Raises:
        ValueError: If the specification limits are not ordered, or sigma is
            zero, which makes both indices undefined.
    """
    if usl <= lsl:
        raise ValueError(f"usl {usl} must be greater than lsl {lsl}")
    if chart.sigma == 0:
        raise ValueError("sigma is zero, capability is undefined")

    # Correction 3: both indices are dimensionless, divided through by sigma.
    return Capability(
        cp=(usl - lsl) / (6 * chart.sigma),
        cpk=min(usl - chart.center_x, chart.center_x - lsl) / (3 * chart.sigma),
        usl=usl,
        lsl=lsl,
    )


def violations(chart: ControlChart, capability_result: Capability | None = None) -> list[str]:
    """Out of control and not capable conditions, as readable lines.

    Args:
        chart: Chart statistics to scan.
        capability_result: Capability to check, or None to skip those checks.

    Returns:
        One line per condition found. An empty list means in control.
    """
    found: list[str] = []

    # Correction 1: scan every subgroup, not just the first and the last.
    for index, value in enumerate(chart.subgroup_means):
        if value >= chart.ucl_x or value <= chart.lcl_x:
            found.append(f"X-bar subgroup {index} is outside the control limits at {value:.4f}")

    # For subgroup sizes below 7 the D3 constant is zero, so the R chart has
    # no lower control limit and a range of zero is not a violation. The
    # notebook tested `value <= lcl_r` unconditionally, which flagged every
    # subgroup whose four measurements happened to be identical.
    for index, value in enumerate(chart.subgroup_ranges):
        above = value >= chart.ucl_r
        below = chart.lcl_r > 0 and value <= chart.lcl_r
        if above or below:
            found.append(f"R subgroup {index} is outside the control limits at {value:.4f}")

    below_x = int(np.sum(chart.subgroup_means < chart.center_x))
    below_r = int(np.sum(chart.subgroup_ranges < chart.center_r))
    total = len(chart.subgroup_means)
    threshold = RUN_FRACTION * total

    if below_x >= threshold:
        found.append(f"X-bar chart runs below its centre line, {below_x} of {total} points")
    elif total - below_x >= threshold:
        found.append(f"X-bar chart runs above its centre line, {total - below_x} of {total} points")

    if below_r >= threshold:
        found.append(f"R chart runs below its centre line, {below_r} of {total} points")
    elif total - below_r >= threshold:
        found.append(f"R chart runs above its centre line, {total - below_r} of {total} points")

    if capability_result is not None:
        if capability_result.cp < 1:
            found.append(
                f"process spread cannot hold the tolerance, Cp is {capability_result.cp:.3f}"
            )
        if capability_result.cpk < 0:
            found.append("process mean is outside the specification limits")
        elif capability_result.cpk < 1:
            found.append(f"process is not capable as centred, Cpk is {capability_result.cpk:.3f}")
        if capability_result.usl < chart.ucl_x or capability_result.lsl > chart.lcl_x:
            found.append("control limits fall outside the specification limits")

    return found
