"""Incomplete Beta tails with one scalar/vector continued-fraction recurrence."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from quivers.qiec._count_mixtures import _log_gamma_increment, _stirling_error
from quivers._quadrature_constants import _NODES_8, _WEIGHTS_8


def _raise(change: float, error: Exception) -> None:
    """Raise the error of a fraction that did not converge.

    Parameters
    ----------
    change : float
        The last multiplicative update, unused.
    error : Exception
        The error.

    Raises
    ------
    Exception
        ``error``.
    """
    del change
    raise error


def beta_fraction(
    a: Any,
    b: Any,
    x: Any,
    *,
    nonzero: Callable[[Any], Any],
    converged: Callable[[Any], bool],
    unconverged: Callable[[Any, Exception], None] = _raise,
    max_iterations: int = 4096,
) -> Any:
    """Evaluate the incomplete Beta fraction using scalar or tensor arithmetic.

    The same recurrence serves the reference and native backends. At least
    64 terms are retained so termination at an integer parameter does not
    discard parameter derivatives of the continued fraction.

    Parameters
    ----------
    a, b, x : Any
        Concentrations and evaluation point in scalar or tensor arithmetic.
    nonzero : Callable
        Stabilizes a recurrence denominator without changing its sign.
    converged : Callable
        Checks convergence of the final multiplicative update.
    unconverged : Callable
        Called with the last update and the error once the budget is
        exhausted; by default it raises the error. When it returns, the
        fraction so far is returned.
    max_iterations : int
        Finite recurrence budget.

    Returns
    -------
    Any
        Continued-fraction value in the input arithmetic.

    Raises
    ------
    Exception
        What ``unconverged`` raises if the recurrence exhausts its
        convergence budget, by default a ``RuntimeError``.
    """
    c = 1.0
    change = d = 1.0 / nonzero(1.0 - (a + b) * x / (a + 1.0))
    result = d
    for m in range(1, max_iterations + 1):
        twice = 2 * m
        coefficient = m * ((b - m) / (a + twice)) * (x / (a + twice - 1))
        d = 1.0 / nonzero(1.0 + coefficient * d)
        c = nonzero(1.0 + coefficient / c)
        result = result * d * c
        coefficient = -((a + m) / (a + twice)) * ((a + b + m) / (a + twice + 1)) * x
        d = 1.0 / nonzero(1.0 + coefficient * d)
        c = nonzero(1.0 + coefficient / c)
        change = d * c
        result = result * change
        if m >= 64 and converged(change):
            return result
    unconverged(
        change,
        RuntimeError(
            "incomplete Beta fraction did not converge within "
            f"{max_iterations} iterations"
        ),
    )
    return result


def _central_polynomial(value: float) -> float:
    """Evaluate ``(log1p(value)-value)/value**2`` near zero.

    Parameters
    ----------
    value : float
        Small relative displacement from the Beta mean.

    Returns
    -------
    float
        Truncated Taylor polynomial, including its value at zero.
    """
    result = -1.0 / 18
    for degree in range(17, 1, -1):
        result = (-1.0 if degree % 2 == 0 else 1.0) / degree + value * result
    return result


def log_beta_front(a: float, b: float, x: float) -> float:
    """Compute the incomplete Beta prefactor without central cancellation.

    Parameters
    ----------
    a, b : float
        Positive concentrations.
    x : float
        Point strictly inside the unit interval.

    Returns
    -------
    float
        Log of ``x**a * (1-x)**b / Beta(a,b)``.
    """
    total = a + b
    if min(a, b) < 8:
        small, large = min(a, b), max(a, b)
        log_beta = math.lgamma(small) - _log_gamma_increment(large, small)
        return math.fsum((a * math.log(x), b * math.log1p(-x), -log_beta))
    p, q = a / total, b / total
    delta = x - p
    if abs(delta) < 0.01 * min(p, q):
        u, v = delta / p, -delta / q
        centered = a * u * u * _central_polynomial(u) + b * v * v * _central_polynomial(
            v
        )
    else:
        centered = math.fsum(
            (a * (math.log(x) - math.log(p)), b * (math.log1p(-x) - math.log(q)))
        )
    return math.fsum(
        (
            centered,
            0.5 * (math.log(a) + math.log(b) - math.log(total)),
            -0.91893853320467274178,
            -_stirling_error(a),
            -_stirling_error(b),
            _stirling_error(total),
        )
    )


def log_beta_tails(a: float, b: float, x: float) -> tuple[float, float]:
    """Return log regularized Beta CDF and survival probabilities.

    Parameters
    ----------
    a, b : float
        Finite positive concentrations with a finite sum.
    x : float
        Evaluation point; points outside the unit interval retain exact tails.

    Returns
    -------
    tuple[float, float]
        Log CDF and log survival probability.

    Raises
    ------
    ValueError
        If concentrations are invalid or the point is NaN.
    RuntimeError
        If the continued fraction exceeds its bounded convergence budget.
    """
    if not (a > 0 and b > 0 and math.isfinite(a + b)) or math.isnan(x):
        raise ValueError(
            "Beta tails require finite positive concentrations and a non-NaN point"
        )
    if x <= 0:
        return -math.inf, 0.0
    if x >= 1:
        return 0.0, -math.inf
    swapped = x > (a + 1) / (a + b + 2)
    aa, bb, xx = (b, a, 1 - x) if swapped else (a, b, x)
    fraction = beta_fraction(
        aa,
        bb,
        xx,
        nonzero=lambda value: math.copysign(max(abs(value), 1e-300), value),
        converged=lambda value: abs(value - 1) <= 8 * math.ulp(1.0),
    )
    lower = min(0.0, log_beta_front(aa, bb, xx) + math.log(fraction) - math.log(aa))
    upper = (
        math.log1p(-math.exp(lower))
        if lower <= -0.6931471805599453
        else math.log(-math.expm1(lower))
        if lower < 0
        else -math.inf
    )
    return (upper, lower) if swapped else (lower, upper)


def log_beta_interval_mass(a: float, b: float, low: float, high: float) -> float:
    """Compute Beta interval mass from stable tails or a local positive rule.

    Parameters
    ----------
    a, b : float
        Finite positive concentrations.
    low, high : float
        Interval endpoints, clipped to the Beta support for integration.

    Returns
    -------
    float
        Log interval probability, with negative infinity for zero mass.

    Raises
    ------
    ValueError
        If a concentration or endpoint is invalid.
    RuntimeError
        If a tail calculation exceeds its bounded recurrence budget.
    """
    low_cdf, low_sf = log_beta_tails(a, b, low)
    high_cdf, high_sf = log_beta_tails(a, b, high)
    lower, upper = max(0.0, low), min(1.0, high)
    if lower >= upper:
        return -math.inf
    width = upper - lower
    if lower > 0 and upper < 1:
        variation = width * (abs(a - 1) / lower + abs(b - 1) / (1 - upper))
        if variation < 1e-4:
            terms = []
            for node, weight in zip(_NODES_8, _WEIGHTS_8, strict=True):
                point = lower + width * ((1 + node) / 2)
                terms.append(
                    math.log(weight / 2)
                    + log_beta_front(a, b, point)
                    - math.log(point)
                    - math.log1p(-point)
                )
            largest = max(terms)
            return (
                math.log(width)
                + largest
                + math.log(math.fsum(math.exp(term - largest) for term in terms))
            )
    first, second = (high_cdf, low_cdf) if high_cdf < low_sf else (low_sf, high_sf)
    if first == -math.inf or first <= second:
        return -math.inf
    difference = second - first
    return first + (
        math.log1p(-math.exp(difference))
        if difference < -math.log(2)
        else math.log(-math.expm1(difference))
    )
