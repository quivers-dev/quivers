"""Reference interval masses preserve raw measure weights and endpoint atoms."""

from __future__ import annotations

import math

from quivers.qiec._beta_tails import log_beta_interval_mass
from quivers.qiec._gamma_tails import _gamma_interval_log_mass, _regularized_gamma_logs
from quivers._quadrature_constants import _NODES_8, _WEIGHTS_8
from quivers.qiec.primitives import _log1mexp, _log_ndtr
from quivers.qiec.reference_support import DistributionError


def _closed(arguments, name):
    """Read optional endpoint metadata without coercing numeric values.

    Parameters
    ----------
    arguments : Mapping
        Retained family arguments.
    name : str
        Endpoint policy name.

    Returns
    -------
    bool
        Closed by default.

    Raises
    ------
    DistributionError
        If the supplied endpoint policy is not Boolean.
    """
    value = arguments.get(name, True)
    if type(value) is not bool:
        raise DistributionError(f"{name} must be Boolean")
    return value


def _intersect(a, b, ac, bc, *, lower):
    """Intersect two optional bounds, retaining the stricter endpoint policy.

    Parameters
    ----------
    a, b : float or None
        Bounds to intersect.
    ac, bc : bool
        Their endpoint membership.
    lower : bool
        Whether these are lower bounds.

    Returns
    -------
    tuple
        Intersection bound and membership.
    """
    if a is None:
        return b, bc
    if b is None:
        return a, ac
    if a == b:
        return a, ac and bc
    return (a, ac) if (a > b if lower else a < b) else (b, bc)


def _difference(larger, smaller):
    """Subtract nonnegative masses in log space, retaining exact zero.

    Parameters
    ----------
    larger, smaller : float
        Log masses.

    Returns
    -------
    float
        Log of their positive difference or negative infinity.
    """
    return larger + _log1mexp(smaller - larger) if smaller < larger else -math.inf


def log_interval_mass(base, low, high, *, lower_closed=True, upper_closed=True):
    """Integrate a raw scalar measure over an endpoint-aware interval.

    Parameters
    ----------
    base : object
        Reference sampleable value.
    low, high : float or None
        Optional interval endpoints.
    lower_closed, upper_closed : bool
        Whether an endpoint belongs to the event.

    Returns
    -------
    float
        Raw log mass, before any explicit normalization.

    Raises
    ------
    DistributionError
        If endpoints are invalid or a finite summation budget is exhausted.
    """
    from quivers.qiec import reference_families as rf

    if type(lower_closed) is not bool or type(upper_closed) is not bool:
        raise DistributionError("interval endpoint membership must be Boolean")
    if any(x is not None and math.isnan(x) for x in (low, high)):
        raise DistributionError("interval endpoints must not be NaN")
    lower = -math.inf if low is None else low
    upper = math.inf if high is None else high
    if lower > upper or lower == upper and not (lower_closed and upper_closed):
        return -math.inf
    family = getattr(base, "family", None)
    arguments = getattr(base, "arguments", {})
    if family == "PointMass":
        atom = arguments["value"]
        inside = (atom > lower or lower_closed and atom == lower) and (
            atom < upper or upper_closed and atom == upper
        )
        return 0.0 if inside else -math.inf
    if family in {"Restrict", "Truncated"}:
        low, lower_closed = _intersect(
            low,
            arguments.get("low"),
            lower_closed,
            _closed(arguments, "lower_closed"),
            lower=True,
        )
        high, upper_closed = _intersect(
            high,
            arguments.get("high"),
            upper_closed,
            _closed(arguments, "upper_closed"),
            lower=False,
        )
        mass = log_interval_mass(
            arguments["base"],
            low,
            high,
            lower_closed=lower_closed,
            upper_closed=upper_closed,
        )
        if family == "Truncated":
            mass -= log_interval_mass(
                arguments["base"],
                arguments.get("low"),
                arguments.get("high"),
                lower_closed=_closed(arguments, "lower_closed"),
                upper_closed=_closed(arguments, "upper_closed"),
            )
        return mass
    if family == "Normalize":
        return log_interval_mass(
            arguments["base"],
            low,
            high,
            lower_closed=lower_closed,
            upper_closed=upper_closed,
        ) - rf._log_mass_of(arguments["base"])
    if family == "Mixture":
        weights, components = (
            rf._mixture_weights(arguments),
            rf._components(arguments, "Mixture"),
        )
        if len(weights) != len(components):
            raise DistributionError("Mixture weights and components differ in number")
        return rf._logsumexp(
            [
                math.log(w)
                + log_interval_mass(
                    c, low, high, lower_closed=lower_closed, upper_closed=upper_closed
                )
                for w, c in zip(weights, components, strict=True)
                if w > 0
            ]
        )
    if family == "Beta":
        return log_beta_interval_mass(
            rf._positive_parameter(arguments, "concentration1", family),
            rf._positive_parameter(arguments, "concentration0", family),
            lower,
            upper,
        )
    if family == "Gamma":
        return _gamma_interval_log_mass(
            rf._positive_parameter(arguments, "concentration", family),
            rf._positive_parameter(arguments, "rate", family),
            lower,
            upper,
        )
    if family == "Normal":
        scale = rf._positive_parameter(arguments, "scale", family)
        loc = rf.real_parameter(arguments, "loc", family)
        a, b = (lower - loc) / scale, (upper - loc) / scale
        width = upper - lower
        if (
            math.isfinite(a)
            and math.isfinite(b)
            and 0 < width / scale < 0.01 / (1 + max(abs(a), abs(b)))
        ):
            midpoint = (lower + width / 2 - loc) / scale
            half = width / (2 * scale)
            return math.log(half) + rf._logsumexp(
                [
                    math.log(weight)
                    - 0.5 * (midpoint + half * node) ** 2
                    - 0.5 * math.log(2 * math.pi)
                    for node, weight in zip(_NODES_8, _WEIGHTS_8, strict=True)
                ]
            )
        return (
            _difference(_log_ndtr(-a), _log_ndtr(-b))
            if a > 0
            else _difference(_log_ndtr(b), _log_ndtr(a))
        )
    if family == "Poisson":
        rate = rf.real_parameter(arguments, "rate", family)
        if rate < 0 or not math.isfinite(rate):
            raise DistributionError("Poisson rate must be finite and nonnegative")

        def tails(value, closed):
            """Read a Poisson half-line with stable regularized-Gamma tails.

            Parameters
            ----------
            value : float
                Cutoff in extended real coordinates.
            closed : bool
                Whether the lower half-line includes the cutoff.

            Returns
            -------
            tuple[float, float]
                Lower and upper log tail masses.
            """
            if value == math.inf:
                return 0.0, -math.inf
            if value == -math.inf:
                return -math.inf, 0.0
            count = math.floor(value) if closed else math.ceil(value) - 1
            if count < 0:
                return -math.inf, 0.0
            sf, cdf = _regularized_gamma_logs(count + 1, rate)
            return cdf, sf

        low_cdf, low_sf = tails(lower, not lower_closed)
        high_cdf, high_sf = tails(upper, upper_closed)
        return (
            _difference(high_cdf, low_cdf)
            if high_cdf < low_sf
            else _difference(low_sf, high_sf)
        )
    if family == "ExtendedBeta":
        from quivers.qiec._extended_beta import distribution

        kernel = distribution(arguments)
        low_cdf = (kernel.log_cdf_left if lower_closed else kernel.log_cdf)(lower)
        low_sf = (
            kernel.log_survival_function_left
            if lower_closed
            else kernel.log_survival_function
        )(lower)
        high_cdf = (kernel.log_cdf if upper_closed else kernel.log_cdf_left)(upper)
        high_sf = (
            kernel.log_survival_function
            if upper_closed
            else kernel.log_survival_function_left
        )(upper)
        return (
            _difference(high_cdf, low_cdf)
            if high_cdf < low_sf
            else _difference(low_sf, high_sf)
        )
    if family == "BernsteinHazard":
        from quivers.qiec._bernstein_hazard import distribution

        kernel = distribution(arguments)
        low_cdf, low_sf = kernel.log_cdf(lower), kernel.log_survival_function(lower)
        high_cdf, high_sf = kernel.log_cdf(upper), kernel.log_survival_function(upper)
        return (
            _difference(high_cdf, low_cdf)
            if high_cdf < low_sf
            else _difference(low_sf, high_sf)
        )
    if rf._base_is_discrete(base):
        if lower == math.inf or upper == -math.inf:
            return -math.inf
        start = 0 if lower == -math.inf else math.ceil(lower)
        if not lower_closed and start == lower:
            start += 1
        stop = (
            upper
            if upper_closed
            else math.ceil(upper) - 1
            if math.isfinite(upper)
            else upper
        )
        total = -math.inf
        small = 0
        for count in range(start, start + 100000):
            if count > stop:
                break
            term = float(base.log_prob(count))
            total = rf._logsumexp((total, term))
            small = small + 1 if term < total - 40 else 0
            if upper == math.inf and small > 50 and count > 50:
                break
        else:
            raise DistributionError(
                "reference interval mass exhausted its discrete summation budget"
            )
        return total
    if lower == upper:
        return -math.inf
    mass = rf._truncated_mass(base, lower, upper)
    return math.log(mass) if mass > 0 else -math.inf
