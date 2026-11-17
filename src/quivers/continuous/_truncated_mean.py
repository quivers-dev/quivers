"""Deterministic expectations of scalar laws truncated to an interval.

For a law with distribution function ``F`` and an interval ``I``, the mean of
the law conditioned on ``I`` is ``m(I) / F(I)`` with the partial first moment
``m(I) = int_I y dF(y)``. Three deterministic rules are used, in this order.

Closed forms
    A law may state its own partial first moment through a
    ``log_partial_first_moment`` method, as the zero-one-inflated beta law
    does. Otherwise the size-biased law of many families is a member of a
    known family: ``y f(y) = E[Y] g(y)`` for a density ``g``, so
    ``m(I) = E[Y] G(I)`` and the truncated mean is ``E[Y] G(I) / F(I)``, a
    ratio of two interval masses that the stable log tails evaluate without
    cancellation. This covers the gamma, exponential, Weibull, lognormal,
    shifted lognormal, Frechet, inverse Gaussian and beta laws, the positive
    hurdle laws, and the Poisson, binomial, negative binomial and geometric
    counts. The normal and Student-t laws use their standard closed forms.
    These include every family for which brms computes a truncated mean,
    with the same formulas for the continuous ones. brms sums counts over a
    capped range, where the identities here are exact for an unbounded
    range.

Finite sums
    Any other count law with a finite truncated range is summed term by term.
    With no upper bound and a finite untruncated mean, the partial moment is
    the mean minus the finite sum below the lower bound.

Quadrature
    The remaining continuous laws are integrated by Gauss-Legendre panels
    whose breakpoints are quantiles of the truncated law, found by bisection
    on its interval masses, so the nodes follow the mass wherever it lies.
    The rule also integrates the density itself, and a result is returned
    only when that integral reproduces the interval mass less an atom at
    zero. A law with an atom elsewhere, or a density the panels do not
    resolve, is thus refused instead of being integrated wrongly.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor
from torch.distributions import (
    Beta,
    Binomial,
    Distribution,
    Exponential,
    Gamma,
    Geometric,
    LogNormal,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
    Weibull,
)

from quivers._quadrature_constants import _NODES_32, _WEIGHTS_32
from quivers.continuous._atoms import log_atom_mass
from quivers.continuous._closed_probability import (
    ClosedBinomial,
    ClosedNegativeBinomial,
    ClosedPoisson,
)
from quivers.continuous._interval import _reference, log_interval_mass
from quivers.continuous._positive_hurdle import _PositiveHurdle
from quivers.continuous._response_time import InverseGaussian
from quivers.continuous._shape_families import Frechet, ShiftedLogNormal

MAX_SUMMED_COUNTS = 1 << 16
"""Largest number of integers a truncated count mean sums term by term."""

MIN_COMPLEMENT_MASS = 1e-6
"""Smallest interval mass for which a count mean is taken as the untruncated
mean minus a finite sum; below it the subtraction loses too many digits."""

QUADRATURE_TOLERANCE = 1e-8
"""Largest relative error of the quadrature's own interval mass for which its
mean is returned."""

_LOWER_LEVELS = (1e-10, 1e-7, 1e-4, 1e-2, 0.1, 0.3, 0.5)
_UPPER_LEVELS = (0.3, 0.1, 1e-2, 1e-4, 1e-7, 1e-10)
_BISECTIONS = 64

_Closed = bool | Tensor


def _wide(value: Tensor | float) -> Tensor:
    """Return a value as a float64 tensor.

    Parameters
    ----------
    value : Tensor or float
        Parameter or bound.

    Returns
    -------
    Tensor
        The value in float64.
    """
    return torch.as_tensor(value).to(torch.float64)


def _log_mass(
    base: Distribution,
    low: Tensor,
    high: Tensor,
    lower_closed: _Closed = True,
    upper_closed: _Closed = True,
) -> Tensor:
    """Log mass of an interval whose ends may be infinite, in float64.

    Parameters
    ----------
    base : Distribution
        Scalar law.
    low, high : Tensor
        Interval ends; an infinite end leaves that side unbounded.
    lower_closed, upper_closed : bool or Tensor
        Endpoint membership.

    Returns
    -------
    Tensor
        Float64 log masses, negative infinity for an empty interval.
    """
    shape = torch.broadcast_shapes(base.batch_shape, low.shape, high.shape)
    low, high = low.expand(shape), high.expand(shape)
    empty = high < low
    finite_low = torch.where(torch.isinf(low) | empty, 0.0, low)
    finite_high = torch.where(torch.isinf(high) | empty, 1.0, high)
    both = log_interval_mass(
        base,
        finite_low,
        finite_high,
        lower_closed=lower_closed,
        upper_closed=upper_closed,
    ).double()
    above = log_interval_mass(
        base, finite_low, None, lower_closed=lower_closed
    ).double()
    below = log_interval_mass(
        base, None, finite_high, upper_closed=upper_closed
    ).double()
    open_low, open_high = low == -torch.inf, high == torch.inf
    result = torch.where(
        open_low & open_high,
        0.0,
        torch.where(open_low, below, torch.where(open_high, above, both)),
    )
    return torch.where(empty, -torch.inf, result)


def size_biased_log_moment(
    base: Distribution,
    low: Tensor,
    high: Tensor,
    lower_closed: _Closed,
    upper_closed: _Closed,
) -> Tensor | None:
    """Log partial first moment of a nonnegative law from its size-biased law.

    Parameters
    ----------
    base : Distribution
        Scalar law on the nonnegative reals or the nonnegative integers.
    low, high : Tensor
        Float64 interval ends, possibly infinite.
    lower_closed, upper_closed : bool or Tensor
        Endpoint membership.

    Returns
    -------
    Tensor or None
        Float64 ``log int_I y dF(y)``, or None when the family has no
        size-biased closed form at these parameters.
    """
    if isinstance(base, _PositiveHurdle):
        inner = size_biased_log_moment(
            base._base(), low.clamp_min(0.0), high, True, True
        )
        assert inner is not None
        return torch.log1p(-_wide(base.zero_prob)) + inner
    if isinstance(base, Gamma | Exponential):
        shape = (
            _wide(base.concentration)
            if isinstance(base, Gamma)
            else torch.ones_like(_wide(base.rate))
        )
        rate = _wide(base.rate)
        return shape.log() - rate.log() + _log_mass(Gamma(shape + 1, rate), low, high)
    if isinstance(base, Weibull):
        scale, power = _wide(base.scale), _wide(base.concentration)
        order = 1 + 1 / power

        def hazard(point: Tensor) -> Tensor:
            """Map a response to its cumulative hazard ``(y / scale)**shape``.

            Parameters
            ----------
            point : Tensor
                Interval end, possibly nonpositive or infinite.

            Returns
            -------
            Tensor
                The hazard, zero at or below zero and infinite at infinity.
            """
            inside = (point > 0) & torch.isfinite(point)
            ratio = torch.where(inside, point, 1.0).log() - scale.log()
            return torch.where(
                inside,
                torch.exp(power * ratio),
                torch.where(point <= 0, 0.0, torch.inf),
            )

        tilted = Gamma(order, torch.ones_like(order))
        return (
            scale.log()
            + torch.lgamma(order)
            + _log_mass(tilted, hazard(low), hazard(high))
        )
    if isinstance(base, LogNormal):
        loc, scale = _wide(base.loc), _wide(base.scale)
        return (
            loc
            + scale.square() / 2
            + _log_mass(LogNormal(loc + scale.square(), scale), low, high)
        )
    if isinstance(base, ShiftedLogNormal):
        shift = _wide(base.shift)
        inner = LogNormal(_wide(base.loc), _wide(base.scale))
        moved_low, moved_high = low - shift, high - shift
        moment = size_biased_log_moment(inner, moved_low, moved_high, True, True)
        assert moment is not None
        mass = _log_mass(inner, moved_low, moved_high)
        positive = shift > 0
        shifted = torch.where(positive, shift, 1.0).log() + mass
        return torch.where(positive, torch.logaddexp(shifted, moment), moment)
    if isinstance(base, Frechet):
        scale, power = _wide(base.scale), _wide(base.concentration)
        finite = power > 1
        order = torch.where(finite, 1 - 1 / power, 1.0)

        def inverse_hazard(point: Tensor) -> Tensor:
            """Map a response to ``(y / scale)**(-shape)``, which decreases in it.

            Parameters
            ----------
            point : Tensor
                Interval end, possibly nonpositive or infinite.

            Returns
            -------
            Tensor
                The value, infinite at or below zero and zero at infinity.
            """
            inside = (point > 0) & torch.isfinite(point)
            ratio = torch.where(inside, point, 1.0).log() - scale.log()
            return torch.where(
                inside,
                torch.exp(-power * ratio),
                torch.where(point <= 0, torch.inf, 0.0),
            )

        tilted = Gamma(order, torch.ones_like(order))
        moment = (
            scale.log()
            + torch.lgamma(order)
            + _log_mass(tilted, inverse_hazard(high), inverse_hazard(low))
        )
        # With a tail shape of at most one the size-biased law is not a
        # gamma law; quadrature covers bounded intervals.
        return moment if bool(finite.all()) else None
    if isinstance(base, InverseGaussian):
        mean = _wide(base.mu)
        wide = InverseGaussian(mean, _wide(base.shape))

        def reciprocal(point: Tensor) -> Tensor:
            """Map a response to ``mean**2 / y``, which decreases in it.

            Parameters
            ----------
            point : Tensor
                Interval end, possibly nonpositive or infinite.

            Returns
            -------
            Tensor
                The value, infinite at or below zero and zero at infinity.
            """
            inside = (point > 0) & torch.isfinite(point)
            return torch.where(
                inside,
                mean.square() / torch.where(inside, point, 1.0),
                torch.where(point <= 0, torch.inf, 0.0),
            )

        # If X is inverse Gaussian with mean mu, the size-biased law of X is
        # the law of mu**2 / X.
        return mean.log() + _log_mass(wide, reciprocal(high), reciprocal(low))
    if isinstance(base, Beta):
        first, second = _wide(base.concentration1), _wide(base.concentration0)
        return (
            first.log()
            - (first + second).log()
            + _log_mass(Beta(first + 1, second), low, high)
        )
    if isinstance(base, Poisson):
        rate = _wide(base.rate)
        tilted = ClosedPoisson(rate)
        return rate.log() + _log_mass(
            tilted, low - 1, high - 1, lower_closed, upper_closed
        )
    if isinstance(base, NegativeBinomial | Geometric):
        if isinstance(base, NegativeBinomial):
            count, logits = _wide(base.total_count), _wide(base.logits)
        else:
            # A geometric count of failures before the first success is a
            # negative binomial count with one success and the failure odds.
            logits = -_wide(base.logits)
            count = torch.ones_like(logits)
        tilted = ClosedNegativeBinomial(count + 1, logits=logits)
        return (
            count.log()
            + logits
            + _log_mass(tilted, low - 1, high - 1, lower_closed, upper_closed)
        )
    if isinstance(base, Binomial):
        count, probability = _wide(base.total_count), _wide(base.probs)
        occupied = (count > 0) & (probability > 0)
        tilted = ClosedBinomial(
            torch.where(occupied, count - 1, 0.0),
            probs=torch.where(occupied, probability, 0.5),
        )
        moment = torch.where(occupied, count * probability, 1.0).log() + _log_mass(
            tilted, low - 1, high - 1, lower_closed, upper_closed
        )
        return torch.where(occupied, moment, -torch.inf)
    return None


def _normal_mean(base: Normal, low: Tensor, high: Tensor, log_mass: Tensor) -> Tensor:
    """Mean of a truncated normal law.

    Parameters
    ----------
    base : Normal
        Untruncated law.
    low, high : Tensor
        Float64 interval ends, possibly infinite.
    log_mass : Tensor
        Log mass of the interval.

    Returns
    -------
    Tensor
        ``mu + sigma (phi(a) - phi(b)) / (Phi(b) - Phi(a))`` at the
        standardized ends, with each density ratio formed in log space.
    """
    loc, scale = _wide(base.loc), _wide(base.scale)

    def ratio(point: Tensor) -> Tensor:
        """Standard normal density at an end over the interval mass.

        Parameters
        ----------
        point : Tensor
            Interval end, possibly infinite.

        Returns
        -------
        Tensor
            The ratio, zero at an infinite end.
        """
        finite = torch.isfinite(point)
        z = (torch.where(finite, point, loc) - loc) / scale
        log_density = -0.5 * z.square() - 0.5 * math.log(2 * math.pi)
        return torch.where(finite, torch.exp(log_density - log_mass), 0.0)

    return loc + scale * (ratio(low) - ratio(high))


def _student_mean(
    base: StudentT, low: Tensor, high: Tensor, log_mass: Tensor
) -> Tensor:
    """Mean of a truncated Student-t law with more than one degree of freedom.

    Parameters
    ----------
    base : StudentT
        Untruncated law.
    low, high : Tensor
        Float64 interval ends, possibly infinite.
    log_mass : Tensor
        Log mass of the interval.

    Returns
    -------
    Tensor
        The location plus the scale times
        ``G (A(a) - A(b))`` with ``A(z) = (nu + z**2)**(-(nu - 1) / 2)`` and
        ``G = Gamma((nu - 1) / 2) nu**(nu / 2) / (2 F(I) Gamma(nu / 2)
        Gamma(1 / 2))``; positive or negative infinity where the degrees of
        freedom are at most one and the interval is unbounded, and NaN where
        they are at most one and the interval is bounded.
    """
    df, loc, scale = _wide(base.df), _wide(base.loc), _wide(base.scale)
    defined = df > 1
    safe = torch.where(defined, df, 2.0)
    log_front = (
        torch.lgamma((safe - 1) / 2)
        + safe / 2 * safe.log()
        - math.log(2.0)
        - log_mass
        - torch.lgamma(safe / 2)
        - math.lgamma(0.5)
    )

    def term(point: Tensor) -> Tensor:
        """Evaluate ``G (nu + z**2)**(-(nu - 1) / 2)`` at a standardized end.

        Parameters
        ----------
        point : Tensor
            Interval end, possibly infinite.

        Returns
        -------
        Tensor
            The term, zero at an infinite end.
        """
        finite = torch.isfinite(point)
        z = (torch.where(finite, point, loc) - loc) / scale
        value = torch.exp(log_front - (safe - 1) / 2 * torch.log(safe + z.square()))
        return torch.where(finite, value, 0.0)

    mean = loc + scale * (term(low) - term(high))
    open_low, open_high = torch.isinf(low), torch.isinf(high)
    undefined = torch.where(
        open_low & open_high,
        torch.nan,
        torch.where(open_high, torch.inf, torch.where(open_low, -torch.inf, torch.nan)),
    )
    return torch.where(defined, mean, undefined)


def _count_mean(
    base: Distribution,
    low: Tensor,
    high: Tensor,
    lower_closed: _Closed,
    upper_closed: _Closed,
    log_mass: Tensor,
) -> Tensor:
    """Mean of a truncated count law by a finite sum.

    Parameters
    ----------
    base : Distribution
        Count law on the nonnegative integers.
    low, high : Tensor
        Float64 interval ends, possibly infinite.
    lower_closed, upper_closed : bool or Tensor
        Endpoint membership.
    log_mass : Tensor
        Log mass of the interval.

    Returns
    -------
    Tensor
        The truncated mean. A bounded range is summed directly. An unbounded
        range uses the untruncated mean minus the sum below the lower bound.
        Cost is one log-probability evaluation per summed integer and batch
        entry.

    Raises
    ------
    NotImplementedError
        If a range holds more than `MAX_SUMMED_COUNTS` integers, or an
        unbounded range has no finite untruncated mean or holds less than
        `MIN_COMPLEMENT_MASS` of the law.
    """
    shape = log_mass.shape
    lower_flag = torch.as_tensor(lower_closed).expand(shape)
    upper_flag = torch.as_tensor(upper_closed).expand(shape)
    low, high = low.expand(shape), high.expand(shape)
    first = torch.where(lower_flag, low.ceil(), low.floor() + 1).clamp_min(0.0)
    support_high = getattr(base.support, "upper_bound", None)
    last = torch.where(upper_flag, high.floor(), high.ceil() - 1)
    if support_high is not None:
        last = torch.minimum(last, _wide(support_high).expand(shape))
    bounded = torch.isfinite(last)

    def partial(start: Tensor, stop: Tensor) -> Tensor:
        """Sum ``x p(x)`` over the integers of ``[start, stop]`` per entry.

        Parameters
        ----------
        start, stop : Tensor
            First and last integer of each entry's range; an empty range has ``stop < start``.

        Returns
        -------
        Tensor
            Float64 partial first moments.

        Raises
        ------
        NotImplementedError
            If a range holds more than `MAX_SUMMED_COUNTS` integers.
        """
        width = (stop - start + 1).clamp_min(0.0)
        longest = int(width.max().item()) if width.numel() else 0
        if longest > MAX_SUMMED_COUNTS:
            raise NotImplementedError(
                f"a truncated count mean would sum {longest} integers, more than "
                f"{MAX_SUMMED_COUNTS}"
            )
        total = torch.zeros(shape, dtype=torch.float64)
        for begin in range(0, longest, 1024):
            offsets = torch.arange(
                begin, min(begin + 1024, longest), dtype=torch.float64
            ).reshape(-1, *([1] * len(shape)))
            points = start + offsets
            active = points <= stop
            safe = torch.where(active, points, start.clamp_min(0.0))
            log_probability = base.log_prob(safe).double()
            total = total + torch.where(
                active & (points > 0),
                torch.exp(log_probability + torch.where(points > 0, points, 1.0).log()),
                0.0,
            ).sum(0)
        return total

    if bool(bounded.all()):
        return partial(first, last) / log_mass.exp()
    try:
        mean = _wide(base.mean).expand(shape)
    except NotImplementedError as error:
        raise NotImplementedError(
            f"{type(base).__name__} has no closed-form mean for a truncated "
            "range without an upper bound"
        ) from error
    if not bool(torch.isfinite(mean)[~bounded].all()):
        raise NotImplementedError(
            f"{type(base).__name__} has no finite mean on a truncated range "
            "without an upper bound"
        )
    if bool((log_mass[~bounded] < math.log(MIN_COMPLEMENT_MASS)).any()):
        raise NotImplementedError(
            "a truncated count range without an upper bound holds less than "
            f"{MIN_COMPLEMENT_MASS} of the law, where the mean minus the sum "
            "below the bound loses accuracy"
        )
    stop = torch.where(bounded, last, first - 1)
    start = torch.where(bounded, first, torch.zeros_like(first))
    summed = partial(start, stop)
    return torch.where(bounded, summed, mean - summed) / log_mass.exp()


def _quadrature_mean(
    base: Distribution,
    low: Tensor,
    high: Tensor,
    lower_closed: _Closed,
    upper_closed: _Closed,
    log_mass: Tensor,
    kinks: tuple[Tensor, ...] = (),
) -> Tensor:
    """Mean of a truncated continuous law by quantile-panel quadrature.

    The breakpoints are the quantiles of the truncated law at 13 fixed
    levels, each found by 64 bisections on the interval mass from the nearer
    end, plus the given kinks of the density. Every bounded panel carries a
    32-point Gauss-Legendre rule. An unbounded end panel maps
    ``t = q + w s / (1 - s)``, with ``w`` the width of the adjacent panel,
    onto the same rule. An atom at zero, as in a hurdle law, adds nothing to
    the first moment and is removed from the mass the rule must reproduce.
    Cost is about 1300 interval masses and 450 densities per batch entry.

    Parameters
    ----------
    base : Distribution
        Scalar law with a density away from zero.
    low, high : Tensor
        Float64 interval ends, possibly infinite.
    lower_closed, upper_closed : bool or Tensor
        Endpoint membership.
    log_mass : Tensor
        Log mass of the interval.
    kinks : tuple[Tensor, ...]
        Points where the density is not smooth.

    Returns
    -------
    Tensor
        The truncated mean.

    Raises
    ------
    NotImplementedError
        If the interval is unbounded and the law has no finite mean, or the
        rule does not reproduce the interval mass to `QUADRATURE_TOLERANCE`
        (to 64 units of roundoff for a law that holds its parameters in
        single precision).
    ValueError
        If a quantile of the truncated law cannot be bracketed.
    """
    shape = log_mass.shape
    low, high = low.expand(shape), high.expand(shape)
    support = base.support
    support_low = getattr(support, "lower_bound", None)
    support_high = getattr(support, "upper_bound", None)
    if support_low is not None:
        low = torch.maximum(low, _wide(support_low).expand(shape))
    if support_high is not None:
        high = torch.minimum(high, _wide(support_high).expand(shape))
    unbounded = torch.isinf(low) | torch.isinf(high)
    if bool(unbounded.any()):
        try:
            finite_mean = torch.isfinite(_wide(base.mean).expand(shape))
        except NotImplementedError as error:
            raise NotImplementedError(
                f"{type(base).__name__} certifies no finite mean on an "
                "unbounded truncated range"
            ) from error
        if not bool(finite_mean[unbounded].all()):
            raise NotImplementedError(
                f"{type(base).__name__} has no finite mean on an unbounded "
                "truncated range"
            )

    targets = torch.tensor(
        [math.log(level) for level in (*_LOWER_LEVELS, *_UPPER_LEVELS)],
        dtype=torch.float64,
    ).reshape(-1, *([1] * len(shape)))
    from_low = torch.tensor(
        [True] * len(_LOWER_LEVELS) + [False] * len(_UPPER_LEVELS)
    ).reshape(targets.shape)

    def below(point: Tensor) -> Tensor:
        """Decide whether a point lies below each target quantile.

        Parameters
        ----------
        point : Tensor
            Candidate points, one leading entry per target level.

        Returns
        -------
        Tensor
            Boolean mask, true where the truncated quantile lies above the point.
        """
        lower_part = _log_mass(base, low, point, lower_closed, True) - log_mass
        upper_part = _log_mass(base, point, high, False, upper_closed) - log_mass
        return torch.where(from_low, lower_part < targets, upper_part > targets)

    anchor = torch.where(
        torch.isfinite(low), low, torch.where(torch.isfinite(high), high, 0.0)
    )
    left = low.expand(len(targets), *shape).clone()
    right = high.expand(len(targets), *shape).clone()
    step = torch.ones_like(left)
    for _ in range(2100):
        open_left, open_right = torch.isinf(left), torch.isinf(right)
        if not bool((open_left | open_right).any()):
            break
        probe_left, probe_right = anchor - step, anchor + step
        left = torch.where(open_left & below(probe_left), probe_left, left)
        right = torch.where(open_right & ~below(probe_right), probe_right, right)
        step = step * 2
    else:
        raise ValueError("a truncated quantile could not be bracketed")
    for _ in range(_BISECTIONS):
        middle = left + (right - left) / 2
        is_below = below(middle)
        left = torch.where(is_below, middle, left)
        right = torch.where(is_below, right, middle)
    quantiles = left + (right - left) / 2
    inner_low, inner_high = quantiles.amin(0), quantiles.amax(0)
    extra = [
        torch.minimum(torch.maximum(kink.double().expand(shape), inner_low), inner_high)
        for kink in kinks
    ]
    points = torch.sort(torch.cat((quantiles, *(e.unsqueeze(0) for e in extra))), 0)[0]
    edges = torch.cat(
        (
            torch.where(torch.isfinite(low), low, points[0]).unsqueeze(0),
            points,
            torch.where(torch.isfinite(high), high, points[-1]).unsqueeze(0),
        )
    )
    nodes = torch.tensor(_NODES_32, dtype=torch.float64).reshape(
        -1, 1, *([1] * len(shape))
    )
    weights = torch.tensor(_WEIGHTS_32, dtype=torch.float64).reshape(nodes.shape)

    def integrate(points: Tensor, jacobian: Tensor) -> tuple[Tensor, Tensor]:
        """Integrate the truncated density and its first moment over panels.

        Parameters
        ----------
        points : Tensor
            Quadrature nodes, on node and panel axes before the batch axes.
        jacobian : Tensor
            Derivative of each node's map from the reference rule; zero disables a node.

        Returns
        -------
        tuple[Tensor, Tensor]
            The integral of the normalized density and of the response times it.
        """
        usable = torch.isfinite(points) & (points != 0)
        safe = torch.where(usable, points, 1.0)
        density = torch.exp(base.log_prob(safe).double() - log_mass)
        density = torch.where(usable & (jacobian > 0), density, 0.0)
        weighted = weights * jacobian * density
        return weighted.sum((0, 1)), (weighted * safe).sum((0, 1))

    start, stop = edges[:-1], edges[1:]
    half = (stop - start) / 2
    mass, moment = integrate(
        start + half + half * nodes, half.expand(nodes.shape[0], *half.shape)
    )
    unit = (nodes + 1) / 2
    for open_end, origin, width, sign in (
        (torch.isinf(high), points[-1], points[-1] - points[-2], 1.0),
        (torch.isinf(low), points[0], points[1] - points[0], -1.0),
    ):
        if not bool(open_end.any()):
            continue
        scale = torch.where(open_end & (width > 0), width, 0.0)
        tail_points = origin + sign * scale * unit / (1 - unit)
        jacobian = (scale / (1 - unit).square() / 2).expand(tail_points.shape)
        tail_mass, tail_moment = integrate(tail_points, jacobian)
        mass, moment = mass + tail_mass, moment + tail_moment
    holds_zero = (low <= 0) & (high >= 0)
    try:
        log_atom = log_atom_mass(base, torch.zeros(shape, dtype=torch.float64))
    except NotImplementedError:
        # A law that reports no atoms is checked against the whole mass.
        log_atom = torch.full(shape, -torch.inf, dtype=torch.float64)
    atom = torch.where(holds_zero, torch.exp(log_atom.double() - log_mass), 0.0)
    expected = 1 - atom
    # The density is evaluated in the law's own precision.
    tolerance = max(QUADRATURE_TOLERANCE, 64 * torch.finfo(_reference(base).dtype).eps)
    if not bool(((mass - expected).abs() <= tolerance).all()):
        worst = float((mass - expected).abs().max())
        raise NotImplementedError(
            f"quadrature of a truncated {type(base).__name__} mean missed the "
            f"interval mass by {worst:.3g}, above {tolerance:.3g}; "
            "request a Monte Carlo expectation with mc_samples"
        )
    return moment


def truncated_mean(
    base: Distribution,
    low: Tensor | None,
    high: Tensor | None,
    *,
    lower_closed: _Closed = True,
    upper_closed: _Closed = True,
) -> Tensor:
    """Mean of a scalar law conditioned on an interval, without sampling.

    Parameters
    ----------
    base : Distribution
        Untruncated scalar law.
    low, high : Tensor or None
        Interval ends; None leaves that side unbounded.
    lower_closed, upper_closed : bool or Tensor
        Endpoint membership, which matters for count laws.

    Returns
    -------
    Tensor
        Float64 means on the broadcast batch shape. An entry is infinite or
        NaN where the truncated mean does not exist.

    Raises
    ------
    NotImplementedError
        If the law has an event shape, a count range is too long to sum, an
        unbounded interval has no certified finite mean, or quadrature does
        not reproduce the interval mass.
    """
    if base.event_shape:
        raise NotImplementedError("a truncated mean requires a scalar law")
    lower = _wide(-torch.inf if low is None else low)
    upper = _wide(torch.inf if high is None else high)
    log_mass = _log_mass(base, lower, upper, lower_closed, upper_closed)
    shape = log_mass.shape
    lower, upper = lower.expand(shape), upper.expand(shape)
    if isinstance(base, Normal):
        return _normal_mean(base, lower, upper, log_mass)
    if isinstance(base, StudentT):
        heavy = _wide(base.df).expand(shape) <= 1
        bounded = torch.isfinite(lower) & torch.isfinite(upper)
        if not bool((heavy & bounded).any()):
            return _student_mean(base, lower, upper, log_mass)
        if bool(bounded.all()):
            return _quadrature_mean(
                base, lower, upper, lower_closed, upper_closed, log_mass, (base.loc,)
            )
        raise NotImplementedError(
            "a Student-t law with at most one degree of freedom has no closed "
            "truncated mean when bounded and unbounded rows are mixed"
        )
    own = getattr(base, "log_partial_first_moment", None)
    moment: Tensor | None
    if callable(own):
        moment = cast(
            Tensor,
            own(lower, upper, lower_closed=lower_closed, upper_closed=upper_closed),
        )
    else:
        moment = size_biased_log_moment(base, lower, upper, lower_closed, upper_closed)
    if moment is not None:
        return torch.exp(moment - log_mass)
    if getattr(base.support, "is_discrete", False):
        return _count_mean(base, lower, upper, lower_closed, upper_closed, log_mass)
    # A location parameter marks the only point where the densities of the
    # location-scale laws are not smooth (the asymmetric Laplace kink).
    location = getattr(base, "loc", None)
    kinks = (location,) if isinstance(location, Tensor) else ()
    return _quadrature_mean(
        base, lower, upper, lower_closed, upper_closed, log_mass, kinks
    )


__all__ = [
    "MAX_SUMMED_COUNTS",
    "MIN_COMPLEMENT_MASS",
    "QUADRATURE_TOLERANCE",
    "size_biased_log_moment",
    "truncated_mean",
]
