"""Scalar regularized Gamma tails for checked reference interval measures."""

import math

from quivers.qiec._count_mixtures import _stirling_error
from quivers.qiec.reference_support import DistributionError

_MAX_ITERATIONS = 10000
_TOLERANCE = 8.881784197001252e-16
_TINY = 1e-300


def _log_complement(value: float) -> float:
    """Complement a log probability without tail subtraction.

    Parameters
    ----------
    value : float
        Nonpositive log probability.

    Returns
    -------
    float
        Log of one minus the probability.
    """
    if value == 0:
        return -math.inf
    return (
        math.log1p(-math.exp(value))
        if value < -math.log(2)
        else math.log(-math.expm1(value))
    )


def _log_gamma1p(shape: float) -> float:
    """Evaluate log Gamma near one without losing small shape increments.

    Parameters
    ----------
    shape : float
        Positive increment from one.

    Returns
    -------
    float
        Log Gamma of one plus the increment.
    """
    if shape >= 0.01:
        return math.lgamma(1 + shape)
    coefficients = (
        -0.5772156649015328606,
        0.8224670334241132182,
        -0.4006856343865314285,
        0.2705808084277845479,
        -0.2073855510286739853,
        0.1695571769974081899,
        -0.1440498967688461181,
        0.1255096695247430424,
        -0.1113342658695646905,
        0.1000994575127818085,
    )
    result = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        result = coefficient + shape * result
    return shape * result


def _log_kernel(shape: float, point: float) -> float:
    """Evaluate the common incomplete-Gamma kernel without central cancellation.

    Parameters
    ----------
    shape : float
        Positive Gamma shape.
    point : float
        Positive unit-rate evaluation point.

    Returns
    -------
    float
        ``shape*log(point)-point-logGamma(shape)``.
    """
    if shape < 16:
        return shape * math.log(point) - point - math.lgamma(shape)
    relative = (point - shape) / shape
    if abs(relative) < 0.1:
        polynomial = -1 / 18
        for power in range(17, 1, -1):
            polynomial = (-1 if power % 2 == 0 else 1) / power + relative * polynomial
        deviance = shape * relative * relative * polynomial
    else:
        deviance = shape * (math.log(point) - math.log(shape)) - (point - shape)
    return (
        0.5 * math.log(shape)
        - 0.91893853320467274178
        - _stirling_error(shape)
        + deviance
    )


def _regularized_gamma_logs(shape: float, point: float) -> tuple[float, float]:
    """Evaluate both regularized Gamma tails directly in log space.

    Parameters
    ----------
    shape : float
        Finite strictly positive Gamma shape.
    point : float
        Nonnegative unit-rate point, possibly positive infinity.

    Returns
    -------
    tuple[float, float]
        Log lower-tail and strict upper-tail probabilities.

    Raises
    ------
    DistributionError
        If parameters are invalid or a series/fraction does not converge
        within ten thousand terms.
    """
    if not math.isfinite(shape) or shape <= 0 or math.isnan(point) or point < 0:
        raise DistributionError(
            "Gamma tails require finite positive shape and nonnegative point"
        )
    if point == 0:
        return -math.inf, 0.0
    if math.isinf(point):
        return 0.0, -math.inf
    if shape == 1 or (shape == 2 and point >= 0.01):
        upper = -point if shape == 1 else -point + math.log1p(point)
        return _log_complement(upper), upper
    if point <= 1 and shape < 1:
        # Integrate the exponential series. Each nonconstant term is O(shape),
        # preserving log P near zero even when shape is much smaller than eps.
        factor, total, compensation = 1.0, 0.0, 0.0
        for index in range(1, _MAX_ITERATIONS + 1):
            factor *= -point / index
            term = factor * shape / (shape + index)
            corrected = term - compensation
            update = total + corrected
            compensation = (update - total) - corrected
            total = update
            if abs(term) <= _TOLERANCE * abs(total):
                lower = min(
                    0.0,
                    shape * math.log(point) - _log_gamma1p(shape) + math.log1p(total),
                )
                return lower, _log_complement(lower)
        raise DistributionError(
            "Gamma lower-tail series did not converge within 10000 terms"
        )
    kernel = _log_kernel(shape, point)
    if point < shape + 1 and shape >= 0.01:
        term, total = 1.0, 1.0
        for index in range(1, _MAX_ITERATIONS + 1):
            term *= point / (shape + index)
            total += term
            if term <= _TOLERANCE * total:
                lower = min(0.0, kernel - math.log(shape) + math.log(total))
                return lower, _log_complement(lower)
        raise DistributionError(
            "Gamma lower-tail series did not converge within 10000 terms"
        )
    b = (point - shape) + 1
    c, d = 1 / _TINY, 1 / b
    log_fraction = math.log(d)
    for index in range(1, _MAX_ITERATIONS + 1):
        numerator = -index * (index - shape)
        b += 2
        d = numerator * d + b
        d = math.copysign(max(abs(d), _TINY), d)
        c = b + numerator / c
        c = math.copysign(max(abs(c), _TINY), c)
        d = 1 / d
        delta = d * c
        if not math.isfinite(delta) or delta <= 0:
            raise DistributionError(
                "Gamma upper-tail continued fraction exceeded numerical capacity"
            )
        log_fraction += math.log(delta)
        if abs(delta - 1) <= _TOLERANCE:
            upper = min(0.0, kernel + log_fraction)
            return _log_complement(upper), upper
    raise DistributionError(
        "Gamma upper-tail continued fraction did not converge within 10000 terms"
    )


def _gamma_interval_log_mass(
    shape: float, rate: float, low: float, high: float
) -> float:
    """Score a Gamma interval through its better-conditioned log tail.

    Parameters
    ----------
    shape : float
        Finite positive shape.
    rate : float
        Finite positive inverse scale.
    low : float
        Lower interval endpoint, possibly negative infinity.
    high : float
        Upper interval endpoint, possibly positive infinity.

    Returns
    -------
    float
        Log interval probability, negative infinity for an empty interval.

    Raises
    ------
    DistributionError
        If parameters are invalid or the tail algorithm does not converge.
    """
    if not math.isfinite(shape) or shape <= 0 or not math.isfinite(rate) or rate <= 0:
        raise DistributionError(
            "Gamma interval mass requires finite positive shape and rate"
        )
    if math.isnan(low) or math.isnan(high):
        raise DistributionError("Gamma interval endpoints cannot be NaN")
    if low >= high or high <= 0:
        return -math.inf
    if low > 0 and math.isfinite(high):
        width = high - low
        unit_low = low * rate
        unit_width = width * rate
        relative_width = width / low
        if unit_low > 0 and math.isfinite(unit_low):
            variation = (
                unit_width * abs((shape - 1) / unit_low - 1)
                + abs(shape - 1) * relative_width * relative_width
            )
            if relative_width <= 1e-4 and variation <= 0.01:
                # Log-tail subtraction loses digits for short intervals. Eight
                # positive Gauss nodes instead integrate the smooth local density.
                nodes_weights = (
                    (0.9602898564975363, 0.1012285362903763),
                    (0.7966664774136267, 0.2223810344533745),
                    (0.5255324099163290, 0.3137066458778873),
                    (0.1834346424956498, 0.3626837833783620),
                )
                terms = []
                for node, weight in nodes_weights:
                    for sign in (-1, 1):
                        point = (low + width * ((1 + sign * node) / 2)) * rate
                        terms.append(
                            math.log(weight / 2)
                            + _log_kernel(shape, point)
                            - math.log(point)
                        )
                largest = max(terms)
                return (
                    math.log(width)
                    + math.log(rate)
                    + largest
                    + math.log(math.fsum(math.exp(term - largest) for term in terms))
                )
    lower_cdf, lower_survival = _regularized_gamma_logs(shape, max(0.0, low) * rate)
    upper_cdf, upper_survival = _regularized_gamma_logs(shape, high * rate)
    first, second = (
        (upper_cdf, lower_cdf)
        if upper_cdf <= lower_survival
        else (lower_survival, upper_survival)
    )
    if first == -math.inf or first <= second:
        return -math.inf
    return first + _log_complement(second - first)
