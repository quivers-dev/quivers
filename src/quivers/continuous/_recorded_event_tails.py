"""Paired log tails for the response laws scored by censoring and truncation.

A censored or truncated response is scored through the log cumulative
distribution function (CDF) ``log P(Y <= y)`` and the log survival function
``log P(Y > y)`` of its law. This module provides both tails, with first and
second derivatives in every parameter, for the laws whose PyTorch
implementation lacks a differentiable CDF:

* Student's t, through the regularized incomplete beta function
  (Abramowitz and Stegun, 1964, section 26.7);
* the von Mises law on ``[-pi, pi]``, through its Bessel series with a
  computed truncation bound, and a graded Gauss-Legendre rule for tail masses
  below the series' absolute accuracy and for very large concentrations;
* the Conway-Maxwell-Poisson law, through certified partial sums of its
  normalizing series;
* the exponential law and the Gamma rate product, with every infinite or
  nonpositive point masked before differentiation;
* zero atoms combined with a positive Gamma or LogNormal component, a
  zero-one inflated Beta, and a finite Normal mixture, by composition of their
  component tails.

Each function returns float64 tails that keep exact zero masses as negative
infinity and never pass a non-finite intermediate into a derivative.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor
from torch.distributions import Distribution, StudentT, VonMises

from quivers._positions import anywhere, demand, entry_count, largest
from quivers._quadrature_constants import _NODES_8, _WEIGHTS_8
from quivers.continuous._beta_tails import log_beta_tails, tensor_beta_fraction
from quivers.continuous._closed_probability import _log_gamma_increment
from quivers.continuous._com_poisson import COMPoisson, _relative_weight
from quivers.continuous._gamma_tails import log_gamma_tails
from quivers.continuous._ordered import _log1mexp

_EPS = torch.finfo(torch.float64).eps
_LOG_HALF = -math.log(2.0)
_LOG_TWO_PI = math.log(2.0 * math.pi)
_LOG_SQRT_PI = 0.5 * math.log(math.pi)

#: Absolute truncation tolerance of the von Mises Bessel series.
VON_MISES_SERIES_TOLERANCE = 1e-17

#: Smallest tail mass taken from the von Mises series; smaller masses are
#: integrated directly so their relative accuracy is retained.
VON_MISES_SERIES_FLOOR = 1e-4

#: Largest concentration scored by the Bessel series; larger concentrations
#: are integrated directly, at a cost logarithmic in the concentration.
VON_MISES_SERIES_MAX_CONCENTRATION = 1e4

#: Relative truncation tolerance of a certified COM-Poisson tail sum.
COM_POISSON_TAIL_RTOL = 1e-15


def _log_complement(log_probability: Tensor) -> Tensor:
    """Return ``log(1 - p)`` from ``log p`` with a safe derivative at ``p = 1``.

    Parameters
    ----------
    log_probability : Tensor
        Nonpositive log probabilities.

    Returns
    -------
    Tensor
        Log complementary probabilities; negative infinity where ``p = 1``.
    """
    below = log_probability < 0
    safe = torch.where(below, log_probability, -1.0)
    return torch.where(below, _log1mexp(safe), -torch.inf)


def _log_add(first: Tensor, second: Tensor) -> Tensor:
    """Add two probabilities in log space, keeping a zero sum differentiable.

    Parameters
    ----------
    first, second : Tensor
        Log probabilities, possibly negative infinity.

    Returns
    -------
    Tensor
        ``log(exp(first) + exp(second))`` with finite derivatives when both
        terms are zero.
    """
    first, second = torch.broadcast_tensors(first, second)
    largest = torch.maximum(first, second).detach()
    finite = torch.isfinite(largest)
    shift = torch.where(finite, largest, 0.0)
    total = (
        torch.where(finite, first - shift, 0.0).exp()
        + torch.where(finite, second - shift, 0.0).exp()
    )
    return torch.where(finite, shift + total.log(), largest)


def _log_mass(probability: Tensor) -> Tensor:
    """Take the log of a probability parameter, with exact zero masses.

    Parameters
    ----------
    probability : Tensor
        Probabilities in ``[0, 1]``.

    Returns
    -------
    Tensor
        Log probabilities, negative infinity at zero with a zero derivative.
    """
    positive = probability > 0
    return torch.where(
        positive, torch.where(positive, probability, 1.0).log(), -torch.inf
    )


def scaled_tail_point(rate: Tensor, value: Tensor) -> Tensor:
    """Form the Gamma tail argument ``rate * value`` for any real query.

    Negative points map to zero and infinite points to infinity without a
    derivative through the infinite product.

    Parameters
    ----------
    rate : Tensor
        Positive rates.
    value : Tensor
        Real query points, including infinities.

    Returns
    -------
    Tensor
        Float64 nonnegative scaled points.
    """
    rate, point = torch.broadcast_tensors(rate.double(), value.double())
    finite = torch.isfinite(point)
    scaled = rate * torch.where(finite, point, 0.0).clamp_min(0)
    return torch.where(finite, scaled, torch.where(point > 0, torch.inf, 0.0))


def log_exponential_tails(rate: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
    """Log CDF and survival of an exponential law at real points.

    Parameters
    ----------
    rate : Tensor
        Positive rates.
    value : Tensor
        Real query points, including infinities.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 ``log P(Y <= value)`` and ``log P(Y > value)``. The survival
        ``-rate * value`` is exact; the CDF is its stable complement.
    """
    rate, point = torch.broadcast_tensors(rate.double(), value.double())
    inside = (point > 0) & torch.isfinite(point)
    exponent = -rate * torch.where(inside, point, 1.0)
    return (
        torch.where(
            inside, _log1mexp(exponent), torch.where(point <= 0, -torch.inf, 0.0)
        ),
        torch.where(inside, exponent, torch.where(point <= 0, 0.0, -torch.inf)),
    )


def log_student_t_tails(
    df: Tensor, loc: Tensor, scale: Tensor, value: Tensor
) -> tuple[Tensor, Tensor]:
    """Log CDF and survival of Student's t law.

    With ``t = (value - loc) / scale`` and ``a = df / 2``, the far tail is
    ``P(T > |t|) = I_x(a, 1/2) / 2`` at ``x = df / (df + t**2)``. Near the
    centre the signed central mass is evaluated as

    ``P(0 < T <= t) = t (df + t**2)**(-1/2) x**a C(z) / B(1/2, a)``,

    with ``z = t**2 / (df + t**2)`` and ``C`` the continued fraction of
    ``I_z(1/2, a)``. This form is analytic in ``t`` at zero, so the location
    derivative there is the density. Both forms are differentiated through
    their converged recurrences, giving first and second derivatives in
    every argument. Cost: O(n m) for n points and m continued-fraction terms
    (m is at most 4096 and typically below 100).

    Parameters
    ----------
    df : Tensor
        Positive finite degrees of freedom.
    loc : Tensor
        Finite locations.
    scale : Tensor
        Positive finite scales.
    value : Tensor
        Real query points, including infinities.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 ``log P(Y <= value)`` and ``log P(Y > value)``.

    Raises
    ------
    ValueError
        If a parameter is invalid or a query point is NaN.
    """
    nu, mu, sigma, point = torch.broadcast_tensors(
        df.double(), loc.double(), scale.double(), value.double()
    )
    demand(
        torch.isfinite(nu)
        & (nu > 0)
        & torch.isfinite(mu)
        & torch.isfinite(sigma)
        & (sigma > 0)
        & ~point.isnan(),
        lambda: ValueError(
            "Student-t tails require positive finite df and scale, a finite "
            "location, and a non-NaN point"
        ),
    )
    finite = torch.isfinite(point)
    t = torch.where(finite, (torch.where(finite, point, 0.0) - mu) / sigma, 1.0)
    square = t.square()
    a = nu / 2
    z = square / (nu + square)
    central = z < 1.5 / (a + 2.5)

    central_a = torch.where(central, a, 1.0)
    central_nu = torch.where(central, nu, 2.0)
    central_t = torch.where(central, t, 0.0)
    central_z = torch.where(central, z, 0.0)
    fraction = tensor_beta_fraction(
        torch.full_like(central_a, 0.5), central_a, central_z
    )
    log_beta = _LOG_SQRT_PI - _log_gamma_increment(
        central_a, torch.full_like(central_a, 0.5)
    )
    signed = (
        central_t
        / (central_nu + central_t.square()).sqrt()
        * (-central_a * torch.log1p(central_t.square() / central_nu) - log_beta).exp()
        * fraction
    )
    central_cdf = torch.log(0.5 + signed)
    central_sf = torch.log(0.5 - signed)

    tail = ~central
    tail_a = torch.where(tail, a, 1.0)
    tail_x = torch.where(tail, nu / torch.where(tail, nu + square, nu + 1.0), 0.5)
    log_far, _ = log_beta_tails(tail_a, torch.full_like(tail_a, 0.5), tail_x)
    log_far = log_far + _LOG_HALF
    log_near = torch.log1p(-torch.where(tail, log_far, -1.0).exp())
    negative = t < 0
    tail_cdf = torch.where(negative, log_far, log_near)
    tail_sf = torch.where(negative, log_near, log_far)

    log_cdf = torch.where(central, central_cdf, tail_cdf)
    log_sf = torch.where(central, central_sf, tail_sf)
    return (
        torch.where(finite, log_cdf, torch.where(point < 0, -torch.inf, 0.0)),
        torch.where(finite, log_sf, torch.where(point < 0, 0.0, -torch.inf)),
    )


def _bessel_ratio_terms(concentration: Tensor) -> tuple[Tensor, int]:
    """Return ``I_j(kappa) / I_0(kappa)`` for ``j = 1..J`` and certify ``J``.

    Successive ratios ``r_j = I_j / I_(j-1)`` satisfy the backward recurrence
    ``r_j = 1 / (2 j / kappa + r_(j+1))`` (Abramowitz and Stegun, 1964, 9.6.26).
    The recurrence starts beyond ``kappa`` from the bound
    ``r_j < kappa / (2 j)``, where its error contracts by at least ``r_j**2``
    per step, and keeps derivatives only for the retained terms.

    Parameters
    ----------
    concentration : Tensor
        Positive finite concentrations, float64.

    Returns
    -------
    tuple[Tensor, int]
        Ratios with a trailing term axis of length ``J``, and ``J``.

    Raises
    ------
    ValueError
        If the truncation bound exceeds `VON_MISES_SERIES_TOLERANCE`.
    """
    peak = largest(concentration, 1.0)
    terms = int(math.ceil(12.0 * math.sqrt(peak))) + 40
    start = terms + int(math.ceil(peak + 12.0 * math.sqrt(peak))) + 40
    with torch.no_grad():
        ratio = concentration / (2.0 * start)
        for j in range(start - 1, terms, -1):
            ratio = 1.0 / (2.0 * j / concentration + ratio)
        boundary = ratio.clone()
    ratio = boundary
    ratios: list[Tensor] = []
    for j in range(terms, 0, -1):
        ratio = 1.0 / (2.0 * j / concentration + ratio)
        ratios.append(ratio)
    ratios.reverse()
    products = torch.cumprod(torch.stack(ratios, dim=-1), dim=-1)
    with torch.no_grad():
        # Ratios decrease in their order, so the omitted terms are bounded
        # by a geometric series with the first omitted ratio.
        bound = (
            (2.0 / math.pi)
            * products[..., -1]
            * boundary
            / ((terms + 1) * (1.0 - boundary))
        )
    demand(
        ~(bound > VON_MISES_SERIES_TOLERANCE),
        lambda: ValueError(
            "von Mises CDF series could not certify its truncation bound "
            f"{VON_MISES_SERIES_TOLERANCE:g} for concentration {peak:g}"
        ),
    )
    return products, terms


def _von_mises_log_integral(
    loc: Tensor, concentration: Tensor, near: Tensor, far: Tensor
) -> Tensor:
    """Integrate the von Mises density over an interval graded from one end.

    Panels grow geometrically by a factor 1.05 away from the endpoint with
    the larger density, so that the density varies by a bounded factor on
    each eight-point Gauss-Legendre panel. Cost: O(n P) for n intervals and
    P panels, with P logarithmic in the concentration.

    Parameters
    ----------
    loc, concentration : Tensor
        Locations and positive concentrations.
    near, far : Tensor
        Interval endpoints; the density is largest at ``near``.

    Returns
    -------
    Tensor
        Log masses of the intervals.
    """
    length = (far - near).abs()
    slope = concentration * length * ((near - loc).sin().abs() + length) + 1.0
    first = (1e-3 / slope).clamp_max(1.0)
    growth = 1.05
    shortest = -largest(-first, -1.0)
    panels = int(math.ceil(math.log(1.0 / shortest) / math.log(growth))) + 1
    steps = torch.arange(panels + 1, dtype=torch.float64, device=near.device)
    edges = torch.cat(
        (
            torch.zeros(first.shape + (1,), dtype=torch.float64, device=near.device),
            (first.unsqueeze(-1) * growth ** steps[:-1]).clamp_max(1.0),
        ),
        dim=-1,
    )
    lower, upper = edges[..., :-1], edges[..., 1:]
    width = upper - lower
    nodes = near.new_tensor(_NODES_8)
    weights = near.new_tensor(_WEIGHTS_8)
    fractions = ((lower + upper) / 2).unsqueeze(-1) + (width / 2).unsqueeze(-1) * nodes
    points = (
        near.unsqueeze(-1).unsqueeze(-1)
        + (far - near).unsqueeze(-1).unsqueeze(-1) * fractions
    )
    kappa = concentration.unsqueeze(-1).unsqueeze(-1)
    # cos(d) - 1 = -2 sin(d / 2)**2 keeps its relative accuracy near the mode,
    # where the concentration multiplies the rounding error of cos(d) - 1.
    half = (points - loc.unsqueeze(-1).unsqueeze(-1)) / 2
    log_density = -2.0 * kappa * half.sin().square()
    occupied = width > 0
    log_width = torch.where(occupied, width, 1.0).log()
    log_panel = torch.logsumexp(log_density + weights.log(), dim=-1) + torch.where(
        occupied, log_width - math.log(2.0), -torch.inf
    )
    return (
        torch.logsumexp(log_panel, dim=-1)
        + length.log()
        - _LOG_TWO_PI
        - torch.special.i0e(concentration).log()
    )


def _von_mises_log_mass(
    loc: Tensor, concentration: Tensor, low: Tensor, high: Tensor
) -> Tensor:
    """Integrate the von Mises density over ``[low, high]`` inside the circle.

    The interval is split at the mode and the antimode, so the density is
    monotone on each piece, and each piece is integrated by
    `_von_mises_log_integral` from its endpoint of larger density.

    Parameters
    ----------
    loc, concentration : Tensor
        Locations and positive concentrations.
    low, high : Tensor
        Interval endpoints in ``[-pi, pi]`` with ``low <= high``.

    Returns
    -------
    Tensor
        Log masses; negative infinity for empty intervals.
    """
    mode = torch.atan2(loc.sin(), loc.cos())
    antimode = torch.atan2(-loc.sin(), -loc.cos())
    first = torch.minimum(mode, antimode).clamp(min=low, max=high)
    second = torch.maximum(mode, antimode).clamp(min=low, max=high)
    pieces = []
    for start, stop in ((low, first), (first, second), (second, high)):
        occupied = stop > start
        safe_start = torch.where(occupied, start, 0.0)
        safe_stop = torch.where(occupied, stop, 1.0)
        stop_nearer = (safe_stop - loc).cos() >= (safe_start - loc).cos()
        near = torch.where(stop_nearer, safe_stop, safe_start)
        far = torch.where(stop_nearer, safe_start, safe_stop)
        mass = _von_mises_log_integral(loc, concentration, near, far)
        pieces.append(torch.where(occupied, mass, -torch.inf))
    return torch.logsumexp(torch.stack(pieces), dim=0)


def log_von_mises_tails(
    loc: Tensor, concentration: Tensor, value: Tensor
) -> tuple[Tensor, Tensor]:
    """Log CDF and survival of the von Mises law on ``[-pi, pi]``.

    The CDF is the density integrated from ``-pi``, the convention of brms's
    ``pvon_mises`` and of Stan's ``von_mises_cdf``:

    ``F(y) = (y + pi) / (2 pi) + (1 / pi) sum_j rho_j (sin(j (y - mu))
    + sin(j (pi + mu))) / j``,

    with ``rho_j = I_j(kappa) / I_0(kappa)`` (Mardia and Jupp, 2000, chapter 3).
    The series is truncated at ``J = ceil(12 sqrt(kappa)) + 40`` terms with a
    computed absolute bound below `VON_MISES_SERIES_TOLERANCE`. A tail whose
    series mass is below `VON_MISES_SERIES_FLOOR` is integrated directly on
    graded Gauss-Legendre panels split at the mode and antimode, which keeps
    its relative accuracy where the series would only be accurate in absolute
    terms. Concentrations above `VON_MISES_SERIES_MAX_CONCENTRATION` take both
    tails from that integral, whose panel count grows only logarithmically in
    the concentration. Points below ``-pi`` have CDF zero and points above
    ``pi`` CDF one. Cost: O(n J) for n points, with ``J`` at most about 1240,
    or O(n P) panels on the integral path.

    Parameters
    ----------
    loc : Tensor
        Finite locations.
    concentration : Tensor
        Positive finite concentrations.
    value : Tensor
        Real query points, including infinities.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 ``log P(Y <= value)`` and ``log P(Y > value)``.

    Raises
    ------
    ValueError
        If a parameter is invalid, a point is NaN, or the series bound fails.
    """
    mu, kappa, point = torch.broadcast_tensors(
        loc.double(), concentration.double(), value.double()
    )
    demand(
        torch.isfinite(mu) & torch.isfinite(kappa) & (kappa > 0) & ~point.isnan(),
        lambda: ValueError(
            "von Mises tails require a finite location, a positive finite "
            "concentration, and a non-NaN point"
        ),
    )
    inside = (point > -math.pi) & (point < math.pi)
    y = torch.where(inside, point, 0.0)
    large = kappa > VON_MISES_SERIES_MAX_CONCENTRATION
    series_kappa = torch.where(large, 1.0, kappa)
    ratios, terms = _bessel_ratio_terms(series_kappa)
    order = torch.arange(1, terms + 1, dtype=torch.float64, device=y.device)
    oscillation = (order * (y - mu).unsqueeze(-1)).sin() + (
        order * (math.pi + mu).unsqueeze(-1)
    ).sin()
    series = (ratios * oscillation / order).sum(-1) / math.pi
    cdf = (y + math.pi) / (2.0 * math.pi) + series
    sf = (math.pi - y) / (2.0 * math.pi) - series
    lower_direct = inside & (large | (cdf < VON_MISES_SERIES_FLOOR))
    upper_direct = inside & (large | (sf < VON_MISES_SERIES_FLOOR))
    log_cdf = torch.where(cdf > 0, torch.where(cdf > 0, cdf, 1.0).log(), -torch.inf)
    log_sf = torch.where(sf > 0, torch.where(sf > 0, sf, 1.0).log(), -torch.inf)
    if anywhere(lower_direct | upper_direct):
        left = torch.full_like(y, -math.pi)
        right = torch.full_like(y, math.pi)
        direct_cdf = _von_mises_log_mass(mu, kappa, left, y)
        direct_sf = _von_mises_log_mass(mu, kappa, y, right)
        # A large concentration takes both tails from the integral; otherwise
        # only the small tail is integrated and the other is its complement.
        log_cdf = torch.where(
            large,
            direct_cdf,
            torch.where(
                lower_direct,
                direct_cdf,
                torch.where(upper_direct, _log_complement(direct_sf), log_cdf),
            ),
        )
        log_sf = torch.where(
            large,
            direct_sf,
            torch.where(
                upper_direct,
                direct_sf,
                torch.where(lower_direct, _log_complement(direct_cdf), log_sf),
            ),
        )
    return (
        torch.where(inside, log_cdf, torch.where(point <= -math.pi, -torch.inf, 0.0)),
        torch.where(inside, log_sf, torch.where(point <= -math.pi, 0.0, -torch.inf)),
    )


def log_com_poisson_tails(law: COMPoisson, value: Tensor) -> tuple[Tensor, Tensor]:
    """Log CDF and survival of a Conway-Maxwell-Poisson law.

    The smaller tail is summed from the query outward. Successive weights
    ``w_(j+1) / w_j = lambda / (j + 1)**nu`` decrease monotonically away from
    the mode, so after a partial sum ending at weight ``w_J`` with next ratio
    ``r < 1`` the omitted mass is at most ``w_J r / (1 - r)``. Summation stops
    once that bound is below `COM_POISSON_TAIL_RTOL` times the partial sum;
    a downward sum that reaches zero is exact. The other tail is the stable
    complement. The normalizer is the law's own certified centred sum.
    Cost: O(n m) for n points and m summed cells, at most the law's
    ``max_terms``.

    Parameters
    ----------
    law : COMPoisson
        Count law with certified normalization.
    value : Tensor
        Real query thresholds; ``P(Y <= floor(value))`` is the CDF.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 ``log P(Y <= value)`` and ``log P(Y > value)``.

    Raises
    ------
    ValueError
        If a point is NaN or a tail cannot be certified within ``max_terms``.
    """
    point = torch.as_tensor(value, dtype=torch.float64, device=law.dispersion.device)
    demand(
        ~point.isnan(),
        lambda: ValueError("COM-Poisson tails require non-NaN query points"),
    )
    mode, log_sum, _, _ = law._statistics()
    log_rate, nu, _ = law._parameters()
    shape = torch.broadcast_shapes(point.shape, mode.shape)
    point, mode, log_sum = (
        point.expand(shape),
        mode.expand(shape),
        log_sum.expand(shape),
    )
    log_rate, nu = log_rate.expand(shape), nu.expand(shape)
    degenerate = (law.log_rate == -torch.inf).expand(shape)
    count = torch.where(torch.isfinite(point), point, 0.0).floor()
    active = torch.isfinite(point) & (count >= 0) & ~degenerate
    downward = active & (count < mode)
    direction = torch.where(downward, -1.0, 1.0)
    begin = torch.where(downward, count, count + 1)
    accumulated = torch.full_like(count, -torch.inf)
    pending = active.clone()
    width = max(1, min(64, 131072 // max(1, entry_count(count))))
    offset = 0
    log_tolerance = math.log(COM_POISSON_TAIL_RTOL)
    while offset < law.max_terms and anywhere(pending):
        steps = torch.arange(
            offset, offset + width, dtype=torch.float64, device=count.device
        )
        cells = begin.unsqueeze(-1) + direction.unsqueeze(-1) * steps
        valid = cells >= 0
        weights = _relative_weight(
            cells.clamp_min(0),
            log_rate.unsqueeze(-1),
            nu.unsqueeze(-1),
            mode.unsqueeze(-1),
        )
        weights = torch.where(valid & pending.unsqueeze(-1), weights, -torch.inf)
        accumulated = _log_add(accumulated, torch.logsumexp(weights, dim=-1))
        with torch.no_grad():
            last = cells[..., -1]
            reached_zero = downward & (last <= 0)
            log_ratio = torch.where(
                downward,
                nu * last.clamp_min(1).log() - log_rate,
                log_rate - nu * (last + 1).log(),
            )
            contracting = log_ratio < 0
            last_weight = _relative_weight(last.clamp_min(0), log_rate, nu, mode)
            safe_ratio = torch.where(contracting, log_ratio, -1.0)
            log_bound = last_weight + safe_ratio - torch.log(-torch.expm1(safe_ratio))
            certified = contracting & (log_bound <= log_tolerance + accumulated)
            pending = pending & ~(reached_zero | certified)
        offset += width
    demand(
        ~pending,
        lambda: ValueError(
            "COM-Poisson tail sum could not certify its truncation within "
            f"max_terms={law.max_terms}"
        ),
    )
    log_tail = (accumulated - log_sum).clamp_max(0.0)
    log_cdf = torch.where(downward, log_tail, _log_complement(log_tail))
    log_sf = torch.where(downward, _log_complement(log_tail), log_tail)
    below = point < 0
    log_cdf = torch.where(active, log_cdf, torch.where(below, -torch.inf, 0.0))
    log_sf = torch.where(active, log_sf, torch.where(below, 0.0, -torch.inf))
    zero_rate_cdf = torch.where(point >= 0, 0.0, -torch.inf)
    zero_rate_sf = torch.where(point >= 0, -torch.inf, 0.0)
    return (
        torch.where(degenerate, zero_rate_cdf, log_cdf),
        torch.where(degenerate, zero_rate_sf, log_sf),
    )


def log_zero_atom_tails(
    zero_probability: Tensor,
    positive_tails: tuple[Tensor, Tensor],
    value: Tensor,
    *,
    left: bool,
) -> tuple[Tensor, Tensor]:
    """Compose tails of a zero atom with a continuous positive component.

    ``P(Y <= y) = p0 + (1 - p0) G(y)`` and ``P(Y > y) = (1 - p0) S(y)`` for
    ``y >= 0``. The left-limit variant scores ``P(Y < y)`` and
    ``P(Y >= y)``, which differ only at ``y = 0``.

    Parameters
    ----------
    zero_probability : Tensor
        Probability of the zero atom.
    positive_tails : tuple[Tensor, Tensor]
        Log CDF and survival of the positive component at ``value``.
    value : Tensor
        Real query points.
    left : bool
        Whether to return the left limits.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 log CDF and log survival of the composed law.
    """
    probability = zero_probability.double()
    point = value.double()
    log_atom = _log_mass(probability)
    log_rest = _log_mass(1 - probability)
    component_cdf, component_sf = positive_tails
    includes_atom = point > 0 if left else point >= 0
    log_cdf = torch.where(
        includes_atom, _log_add(log_atom, log_rest + component_cdf), -torch.inf
    )
    log_sf = torch.where(includes_atom, log_rest + component_sf, 0.0)
    return log_cdf, log_sf


def log_hurdle_tails(
    zero_probability: Tensor, component: Distribution, value: Tensor, *, left: bool
) -> tuple[Tensor, Tensor]:
    """Log tails of a zero atom with a Gamma or LogNormal positive component.

    Gamma tails use `log_gamma_tails`, differentiable in shape and rate;
    LogNormal tails use the log normal CDF of the log response.

    Parameters
    ----------
    zero_probability : Tensor
        Probability of the zero atom.
    component : Distribution
        Positive Gamma or LogNormal component.
    value : Tensor
        Real query points.
    left : bool
        Whether to return the left limits ``P(Y < y)`` and ``P(Y >= y)``.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 log CDF and log survival.

    Raises
    ------
    TypeError
        If the component is neither Gamma nor LogNormal.
    """
    point = value.double()
    positive = (point > 0) & torch.isfinite(point)
    if isinstance(component, torch.distributions.Gamma):
        tails = log_gamma_tails(
            component.concentration, scaled_tail_point(component.rate, point)
        )
    elif isinstance(component, torch.distributions.LogNormal):
        loc, scale = component.loc.double(), component.scale.double()
        safe = torch.where(positive, point, 1.0)
        standardized = (safe.log() - loc) / scale
        tails = (
            torch.where(
                positive,
                torch.special.log_ndtr(standardized),
                torch.where(point <= 0, -torch.inf, 0.0),
            ),
            torch.where(
                positive,
                torch.special.log_ndtr(-standardized),
                torch.where(point <= 0, 0.0, -torch.inf),
            ),
        )
    else:
        raise TypeError(
            f"hurdle tails require a Gamma or LogNormal component, not "
            f"{type(component).__name__}"
        )
    return log_zero_atom_tails(zero_probability, tails, point, left=left)


def log_zero_one_inflated_beta_tails(
    mu: Tensor,
    phi: Tensor,
    zoi: Tensor,
    coi: Tensor,
    value: Tensor,
    *,
    left: bool,
) -> tuple[Tensor, Tensor]:
    """Log tails of a Beta law with atoms at zero and one.

    ``P(Y = 0) = zoi (1 - coi)``, ``P(Y = 1) = zoi coi``, and the interior
    carries ``(1 - zoi)`` times ``Beta(mu phi, (1 - mu) phi)``. With
    ``coi = 0`` this is brms's ``zero_inflated_beta`` with ``zi = zoi``.

    Parameters
    ----------
    mu, phi : Tensor
        Interior Beta mean and precision.
    zoi, coi : Tensor
        Atom probability and conditional probability of the one atom.
    value : Tensor
        Real query points.
    left : bool
        Whether to return the left limits ``P(Y < y)`` and ``P(Y >= y)``.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 log CDF and log survival.
    """
    point = value.double()
    mean, precision = mu.double(), phi.double()
    inflation, one_share = zoi.double(), coi.double()
    beta_cdf, beta_sf = log_beta_tails(mean * precision, (1 - mean) * precision, point)
    log_zero = _log_mass(inflation * (1 - one_share))
    log_one = _log_mass(inflation * one_share)
    log_interior = _log_mass(1 - inflation)
    has_zero = point > 0 if left else point >= 0
    has_one = point > 1 if left else point >= 1
    log_cdf = torch.where(
        has_one,
        0.0,
        torch.where(has_zero, _log_add(log_zero, log_interior + beta_cdf), -torch.inf),
    )
    log_sf = torch.where(
        has_one,
        -torch.inf,
        torch.where(has_zero, _log_add(log_interior + beta_sf, log_one), 0.0),
    )
    return log_cdf, log_sf


def log_normal_mixture_tails(
    weights: Tensor, loc: Tensor, scale: Tensor, value: Tensor
) -> tuple[Tensor, Tensor]:
    """Log tails of a finite Normal mixture with components on the last axis.

    Each tail is the weighted log-sum of component log tails, so neither side
    is a rounded complement. Cost: O(n K) for n points and K components.

    Parameters
    ----------
    weights : Tensor
        Mixing weights on the simplex along the last axis.
    loc, scale : Tensor
        Component locations and positive scales along the last axis.
    value : Tensor
        Real query points without the component axis.

    Returns
    -------
    tuple[Tensor, Tensor]
        Float64 log CDF and log survival.
    """
    point = value.double().unsqueeze(-1)
    finite = torch.isfinite(point)
    standardized = (torch.where(finite, point, 0.0) - loc.double()) / scale.double()
    log_weight = _log_mass(weights.double())
    log_cdf = torch.logsumexp(log_weight + torch.special.log_ndtr(standardized), dim=-1)
    log_sf = torch.logsumexp(log_weight + torch.special.log_ndtr(-standardized), dim=-1)
    point = point.squeeze(-1)
    finite = finite.squeeze(-1)
    return (
        torch.where(finite, log_cdf, torch.where(point < 0, -torch.inf, 0.0)),
        torch.where(finite, log_sf, torch.where(point < 0, 0.0, -torch.inf)),
    )


def invert_log_cdf(
    log_cdf: Callable[[Tensor], Tensor],
    log_probability: Tensor,
    low: Tensor,
    high: Tensor,
    *,
    iterations: int = 200,
) -> Tensor:
    """Invert a continuous log CDF by bisection inside ``[low, high]``.

    Infinite ends are first replaced by finite brackets, doubling their
    distance from the other end until the target lies inside. Bisection then
    halves the bracket ``iterations`` times, so the returned point is within
    ``2**-iterations`` of the bracket width (float64 resolution for the
    default) of the exact quantile. Cost: O(n (b + iterations)) log-CDF
    evaluations for n targets and b doubling steps.

    Parameters
    ----------
    log_cdf : Callable[[Tensor], Tensor]
        Nondecreasing log CDF of a continuous law, evaluated elementwise.
    log_probability : Tensor
        Target log probabilities strictly inside the bracket's CDF range.
    low, high : Tensor
        Bracketing points, possibly infinite, broadcast with the targets.
    iterations : int
        Number of bisection steps.

    Returns
    -------
    Tensor
        Float64 quantiles inside ``[low, high]``.

    Raises
    ------
    ValueError
        If an infinite end cannot be bracketed within 2100 doublings.
    """
    target, lower, upper = torch.broadcast_tensors(
        log_probability.double(), low.double(), high.double()
    )
    lower, upper = lower.clone(), upper.clone()
    anchor = torch.where(
        torch.isfinite(lower),
        lower,
        torch.where(torch.isfinite(upper), upper, torch.zeros_like(upper)),
    )
    step = torch.ones_like(target)
    for _ in range(2100):
        open_low = torch.isinf(lower)
        open_high = torch.isinf(upper)
        if not bool((open_low | open_high).any()):
            break
        probe_low = anchor - step
        probe_high = anchor + step
        lower = torch.where(open_low & (log_cdf(probe_low) < target), probe_low, lower)
        upper = torch.where(
            open_high & (log_cdf(probe_high) >= target), probe_high, upper
        )
        step = step * 2
    else:
        raise ValueError("quantile inversion could not bracket an infinite end")
    for _ in range(iterations):
        middle = lower + (upper - lower) / 2
        below = log_cdf(middle) < target
        lower = torch.where(below, middle, lower)
        upper = torch.where(below, upper, middle)
    return upper


def recorded_event_log_tails(
    base: Distribution, value: Tensor
) -> tuple[Tensor, Tensor] | None:
    """Dispatch the tails of a PyTorch or count law that lacks its own.

    Parameters
    ----------
    base : Distribution
        Candidate response law.
    value : Tensor
        Real query points.

    Returns
    -------
    tuple[Tensor, Tensor] or None
        Log CDF and survival for Student-t, von Mises, and COM-Poisson laws;
        None for any other law.
    """
    if isinstance(base, StudentT):
        return log_student_t_tails(base.df, base.loc, base.scale, value)
    if isinstance(base, VonMises):
        return log_von_mises_tails(base.loc, base.concentration, value)
    if isinstance(base, COMPoisson):
        return log_com_poisson_tails(base, value)
    return None


__all__ = [
    "COM_POISSON_TAIL_RTOL",
    "invert_log_cdf",
    "VON_MISES_SERIES_FLOOR",
    "VON_MISES_SERIES_MAX_CONCENTRATION",
    "VON_MISES_SERIES_TOLERANCE",
    "log_com_poisson_tails",
    "log_exponential_tails",
    "log_hurdle_tails",
    "log_normal_mixture_tails",
    "log_student_t_tails",
    "log_von_mises_tails",
    "log_zero_atom_tails",
    "log_zero_one_inflated_beta_tails",
    "recorded_event_log_tails",
    "scaled_tail_point",
]
