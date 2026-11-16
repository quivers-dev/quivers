"""Training-time knot selection for retained spline hazard bases."""

from __future__ import annotations

import math
from numbers import Real
from collections.abc import Sequence

from quivers._splines import MSplineBasis


def cox_spline_basis(
    event_times: Sequence[float],
    observed_times: Sequence[float],
    *,
    df: int = 5,
    degree: int = 3,
) -> MSplineBasis:
    """Choose event-quantile knots within the full observed-time boundary span.

    Parameters
    ----------
    event_times : Sequence[float]
        Finite nonnegative times of exact, uncensored events.
    observed_times : Sequence[float]
        Finite nonnegative training follow-up times, including censored rows.
    df : int
        Requested number of normalized M-spline components, including the
        intercept component. Must be at least ``degree + 1``.
    degree : int
        Polynomial degree; three gives cubic M-splines.

    Returns
    -------
    MSplineBasis
        Immutable knots, degree, zero lower boundary, and constant hazard tail.
        The upper boundary exceeds maximum follow-up by one fiftieth of its
        full span, or one fiftieth of the maximum when the span is zero.

    Raises
    ------
    ValueError
        If training evidence cannot determine the requested distinct interior
        event quantiles or a positive finite boundary. Supply an explicit
        MSplineBasis when automatic knot selection is underdetermined.

    Notes
    -----
    This function belongs to training-data preparation. Prediction retains its
    returned basis and never selects knots from fresh covariates or responses.
    Censoring times can extend support but cannot move interior event quantiles.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 0:
        raise ValueError("spline degree must be a nonnegative integer")
    if isinstance(df, bool) or not isinstance(df, int) or df < degree + 1:
        raise ValueError("spline df must be an integer at least degree + 1")
    for values in (event_times, observed_times):
        if any(
            isinstance(t, bool)
            or not isinstance(t, Real)
            or not math.isfinite(t)
            or t < 0
            for t in values
        ):
            raise ValueError("training spline times must be finite and nonnegative")
    if not observed_times:
        raise ValueError("automatic spline knots require observed follow-up times")
    lower, maximum = min(observed_times), max(observed_times)
    if maximum <= 0:
        raise ValueError(
            "automatic spline knots require positive follow-up; supply an explicit basis"
        )
    if event_times and max(event_times) > maximum:
        raise ValueError("event times cannot exceed the full observed-time span")
    span = maximum - lower
    upper = maximum + (span if span else maximum) / 50
    if not math.isfinite(upper):
        raise ValueError(
            "automatic spline upper boundary is not finite; supply an explicit basis"
        )
    count = df - degree - 1
    events = sorted(map(float, event_times))
    if count and not events:
        raise ValueError(
            "event quantiles require exact events; supply an explicit basis"
        )
    knots = []
    for j in range(1, count + 1):
        rank = (len(events) - 1) * j / (count + 1)
        index = math.floor(rank)
        fraction = rank - index
        following = min(index + 1, len(events) - 1)
        knots.append((1 - fraction) * events[index] + fraction * events[following])
    if any(a >= b for a, b in zip((0.0, *knots), (*knots, upper))):
        raise ValueError(
            "event quantiles are not distinct interior knots; supply an explicit basis"
        )
    return MSplineBasis((0.0, float(upper)), tuple(knots), degree)
