"""Scalar COM-Poisson reference density and log-concave rejection sampler."""

import math
import random
from collections.abc import Mapping
from typing import cast

from quivers.qiec._count_mixtures import _deviance, _stirling_error
from quivers.qiec.reference_support import DistributionError, real_parameter


def _parameters(a: Mapping[str, object]) -> tuple[float, float, int]:
    """Validate parameters and resolve the rate-independent modal count.

    Parameters
    ----------
    a : Mapping[str, object]
        Exactly one rate representation and a dispersion.

    Returns
    -------
    tuple[float, float, int]
        Log rate, dispersion, and modal count.

    Raises
    ------
    DistributionError
        If parameters or modal count are invalid.
    """
    if ("rate" in a) == ("log_rate" in a):
        raise DistributionError("COMPoisson requires exactly one of rate and log_rate")
    nu = real_parameter(a, "dispersion", "COMPoisson")
    if "rate" in a:
        rate = real_parameter(a, "rate", "COMPoisson")
        if rate < 0:
            raise DistributionError("COMPoisson rate must be nonnegative")
        lr = math.log(rate) if rate else -math.inf
    else:
        lr = real_parameter(a, "log_rate", "COMPoisson")
    if not math.isfinite(nu) or nu <= 0 or (not math.isfinite(lr) and lr != -math.inf):
        raise DistributionError(
            "COMPoisson requires finite positive dispersion and a valid log rate"
        )
    if lr / nu > math.log(2**53):
        raise DistributionError(
            "COMPoisson mode exceeds exact Float64 count capacity 2**53"
        )
    return lr, nu, math.floor(math.exp(lr / nu))


def _relative(k: int, lr: float, nu: float, mode: int) -> float:
    """Evaluate a stable modal log-weight ratio.

    Parameters
    ----------
    k, mode : int
        Candidate and modal counts.
    lr, nu : float
        Log rate and dispersion.

    Returns
    -------
    float
        Relative log weight.
    """
    log_rho = lr / nu
    if log_rho < -700:
        return k * lr - nu * math.lgamma(k + 1) if k else 0.0
    rho = math.exp(log_rho)

    def poisson(count):
        """Evaluate a stable Poisson log mass.

        Parameters
        ----------
        count : int
            Nonnegative count.

        Returns
        -------
        float
            Poisson log mass.
        """
        return (
            -_stirling_error(count)
            - 0.5 * math.log(2 * math.pi * count)
            - _deviance(count, log_rho)
            if count
            else -rho
        )

    return nu * (poisson(k) - poisson(mode)) + (k - mode) * (lr - nu * log_rho)


def density(a: Mapping[str, object], value: object) -> float:
    """Score a count using certified geometric normalizer tails.

    Parameters
    ----------
    a : Mapping[str, object]
        Rate representation and dispersion.
    value : object
        Candidate scalar count.

    Returns
    -------
    float
        Log mass, negative infinity off support.

    Raises
    ------
    DistributionError
        If the tail bound cannot be certified within the term budget.
    """
    lr, nu, mode = _parameters(a)
    try:
        point = float(cast(float, value))
    except TypeError, ValueError:
        return -math.inf
    if not math.isfinite(point) or point < 0 or point != math.floor(point):
        return -math.inf
    if lr == -math.inf:
        return 0.0 if point == 0 else -math.inf
    total = 1.0
    for radius in range(1, 262144):
        upper, lower = mode + radius, mode - radius
        uw = math.exp(_relative(upper, lr, nu, mode))
        lw = math.exp(_relative(lower, lr, nu, mode)) if lower >= 0 else 0.0
        total += uw + lw
        ur = lr - nu * math.log(upper + 1)
        ll = max(0, lower)
        lratio = nu * math.log(ll) - lr if ll else -math.inf
        if ur < 0 and lratio < 0:
            tail = uw * math.exp(ur) / -math.expm1(ur)
            if ll:
                tail += lw * math.exp(lratio) / -math.expm1(lratio)
            if tail <= 1e-13 * total:
                return _relative(int(point), lr, nu, mode) - math.log(total)
    raise DistributionError("COMPoisson reference normalizer could not certify tails")


def sample(a: Mapping[str, object], rng: random.Random) -> int:
    """Draw from an exact center and geometric-tail rejection envelope.

    Parameters
    ----------
    a : Mapping[str, object]
        Rate representation and dispersion.
    rng : random.Random
        Random generator.

    Returns
    -------
    int
        Count sample.

    Raises
    ------
    DistributionError
        If sampling exceeds its count or proposal capacity.
    """
    lr, nu, mode = _parameters(a)
    if lr == -math.inf:
        return 0
    width = max(1, math.ceil(math.sqrt(math.exp(lr / nu) / nu)))
    left, right = max(0, mode - width), mode + width
    if right > 2**53:
        raise DistributionError(
            "COMPoisson sampling exceeds exact count capacity 2**53"
        )
    ll = nu * math.log(left) - lr if left else -math.inf
    rr = lr - nu * math.log(right + 1)
    lw = _relative(left, lr, nu, mode)
    rw = _relative(right, lr, nu, mode)
    lm = math.exp(lw + ll) / -math.expm1(ll) if left else 0.0
    rm = math.exp(rw + rr) / -math.expm1(rr)
    center = right - left + 1
    for _ in range(10000):
        pick = rng.random() * (lm + center + rm)
        uniform = max(rng.random(), 2.2250738585072014e-308)
        if pick < lm:
            j = 1 + math.floor(math.log(uniform) / ll)
            candidate, envelope = left - j, lw + j * ll
        elif pick >= lm + center:
            j = 1 + math.floor(math.log(uniform) / rr)
            candidate, envelope = right + j, rw + j * rr
        else:
            candidate, envelope = rng.randint(left, right), 0.0
        if candidate > 2**53:
            raise DistributionError(
                "COMPoisson sampling exceeds exact count capacity 2**53"
            )
        if (
            candidate >= 0
            and math.log(max(rng.random(), 2.2250738585072014e-308))
            <= _relative(candidate, lr, nu, mode) - envelope
        ):
            return candidate
    raise DistributionError(
        "COMPoisson reference sampler exhausted its proposal budget"
    )
