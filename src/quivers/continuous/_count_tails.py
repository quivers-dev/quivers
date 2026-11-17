"""Private incomplete-Beta count tails with retained logit coordinates."""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import Tensor
from torch.nn import functional as F

from quivers.continuous._beta_tails import _log_front
from quivers.continuous._closed_probability import (
    _log_gamma_increment,
    _beta_binomial_log_mass,
)
from quivers.continuous._ordered import _log1mexp
from quivers.qiec._beta_tails import beta_fraction


def _fraction(a: Tensor, b: Tensor, x: Tensor) -> Tensor:
    """Evaluate the shared bounded incomplete-Beta recurrence.

    Parameters
    ----------
    a, b, x : Tensor
        Valid recurrence arguments; b may be zero for an explicit NB limit.

    Returns
    -------
    Tensor
        Continued-fraction factor.
    """

    def nonzero(value):
        """Keep recurrence denominators nonzero without changing their sign.

        Parameters
        ----------
        value : Tensor
            Continued-fraction denominator.

        Returns
        -------
        Tensor
            Sign-preserving denominator bounded away from zero.
        """
        return torch.where(
            value.abs() < 1e-300, torch.where(value < 0, -1e-300, 1e-300), value
        )

    return beta_fraction(
        a,
        b,
        x,
        nonzero=nonzero,
        converged=lambda change: bool(
            ((change - 1).abs() <= 8 * torch.finfo(torch.float64).eps).all()
        ),
    )


def _beta_log_pair(a, b, log_x, log_q):
    """Retain Beta log tails when a finite logit coordinate underflows.

    Parameters
    ----------
    a, b, log_x, log_q : Tensor
        Positive shapes and complementary log probability coordinates.

    Returns
    -------
    tuple[Tensor, Tensor]
        Log CDF and log survival.
    """
    interior = torch.isfinite(log_x) & torch.isfinite(log_q)
    aa, bb = torch.where(interior, a, 1.0), torch.where(interior, b, 1.0)
    lx, lq = (
        torch.where(interior, log_x, -math.log(2)),
        torch.where(interior, log_q, -math.log(2)),
    )
    swap = lx > (aa + 1).log() - (aa + bb + 2).log()
    a, b = torch.where(swap, bb, aa), torch.where(swap, aa, bb)
    lx, lq = torch.where(swap, lq, lx), torch.where(swap, lx, lq)
    x = lx.exp()
    representable = (x > 0) & (x < 1)
    safe_x = torch.where(representable, x, 0.5)
    # The central prefactor avoids lgamma subtraction. Log coordinates
    # then retain tails beyond ordinary probability underflow.
    front = _log_front(a, b, safe_x)
    extreme = (
        _log_front(a, b, torch.full_like(x, 0.5))
        + a * (lx + math.log(2))
        + b * (lq + math.log(2))
    )
    front = torch.where(representable, front, extreme)
    lower = (front + _fraction(a, b, x).log() - a.log()).clamp_max(0)
    safe_lower = torch.where(lower < 0, lower, -1.0)
    upper = torch.where(lower < 0, _log1mexp(safe_lower), -torch.inf)
    cdf, sf = torch.where(swap, upper, lower), torch.where(swap, lower, upper)
    return (
        torch.where(interior, cdf, torch.where(torch.isneginf(log_x), -torch.inf, 0.0)),
        torch.where(interior, sf, torch.where(torch.isneginf(log_x), 0.0, -torch.inf)),
    )


def _coordinates(parameter: Tensor, *, logits: bool):
    """Read complementary probability and log-probability coordinates.

    Parameters
    ----------
    parameter : Tensor
        Valid probability or finite logit values.
    logits : bool
        Whether the parameter stores logits.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        p, q, log p, and log q, retaining separate finite logit tails.
    """
    if logits:
        return (
            parameter.sigmoid(),
            (-parameter).sigmoid(),
            F.logsigmoid(parameter),
            F.logsigmoid(-parameter),
        )
    p, q = parameter, 1 - parameter
    return (
        p,
        q,
        torch.where(p > 0, torch.where(p > 0, p, 1.0).log(), -torch.inf),
        torch.where(q > 0, torch.where(q > 0, q, 1.0).log(), -torch.inf),
    )


def binomial_log_tails(
    total: Tensor, value: Tensor, parameter: Tensor, *, logits=False
):
    """Evaluate validated binomial log tails with first/second endpoint AD.

    Parameters
    ----------
    total : Tensor
        Valid nonnegative integer trial counts.
    value : Tensor
        Real query values, including infinities; thresholds use floor.
    parameter : Tensor
        Valid closed probabilities or finite logits.
    logits : bool
        Whether parameter stores logits.

    Returns
    -------
    tuple[Tensor, Tensor]
        Log P(Y<=value) and log P(Y>value).

    Raises
    ------
    ValueError
        If a query is NaN.
    """
    total, value, parameter = torch.broadcast_tensors(
        total.double(), value.double(), parameter.double()
    )
    if bool(value.isnan().any()):
        raise ValueError("count CDF query must not be NaN")
    p, q, lp, lq = _coordinates(parameter, logits=logits)
    k = value.floor()
    active = (k >= 0) & (k < total)
    a, b = torch.where(active, total - k, 1.0), torch.where(active, k + 1, 1.0)
    cdf, sf = _beta_log_pair(a, b, lq, lp)
    # These exact short-tail forms preserve the finite endpoint
    # derivatives that an incomplete-Beta support mask would discard.
    n = torch.where(total > 0, total, 1.0)
    low0 = n * lq
    low1 = (n - 1) * lq + torch.log1p((n - 1) * p)
    high0 = n * lp
    high1 = (n - 1) * lp + torch.log1p((n - 1) * q)
    cdf = torch.where(k == 0, low0, torch.where((k == 1) & (total > 1), low1, cdf))
    sf = torch.where(
        k == total - 1, high0, torch.where((k == total - 2) & (total > 1), high1, sf)
    )
    return (
        torch.where(active, cdf, torch.where(k < 0, -torch.inf, 0.0)),
        torch.where(active, sf, torch.where(k < 0, 0.0, -torch.inf)),
    )


def negative_binomial_log_tails(
    total: Tensor, value: Tensor, parameter: Tensor, *, logits=False
):
    """Evaluate validated NB tails, retaining the total_count=0 derivative.

    Parameters
    ----------
    total : Tensor
        Valid nonnegative NB shapes, including the degenerate zero limit.
    value : Tensor
        Real query values; thresholds use floor.
    parameter : Tensor
        Success probability in [0,1), or finite logits.
    logits : bool
        Whether parameter stores logits.

    Returns
    -------
    tuple[Tensor, Tensor]
        Log P(Y<=value) and log P(Y>value).

    Raises
    ------
    ValueError
        If a query is NaN.
    """
    total, value, parameter = torch.broadcast_tensors(
        total.double(), value.double(), parameter.double()
    )
    if bool(value.isnan().any()):
        raise ValueError("count CDF query must not be NaN")
    p, q, lp, lq = _coordinates(parameter, logits=logits)
    k = value.floor()
    active = (k >= 0) & torch.isfinite(k)
    short = active & (k <= 1)
    small = active & ~short & (total <= 0.1) & ((p > 0) | logits)
    r = torch.where(small, total, 1.0)
    a = torch.where(small, k + 1, 1.0)
    # The survival I_p(k + 1, r) uses its continued fraction only where it
    # converges, p < (a + 1) / (a + r + 2); beyond, the CDF I_q(r, k + 1)
    # is evaluated in the reflected orientation, where q = 1 - p is small.
    reflected = small & (p * (a + r + 2) > a + 1)
    direct = small & ~reflected
    x = torch.where(direct, p, 0.5)
    log_factor = (
        a * torch.where(direct, lp, -math.log(2))
        + r * torch.where(direct, lq, -math.log(2))
        + _log_gamma_increment(a, r)
        - torch.lgamma(1 + r)
        - a.log()
        + _fraction(a, r, x).log()
    )
    tail = r * log_factor.exp()
    cdf_small = torch.log1p(-tail)
    sf_small = torch.where(
        r > 0, torch.where(r > 0, r, 1.0).log() + log_factor, -torch.inf
    )
    # Inactive rows use fixed safe coordinates so their derivatives stay finite.
    r_reflected = torch.where(reflected, r, 1.0)
    a_reflected = torch.where(reflected, a, 1.0)
    log_reflected = (
        r_reflected * torch.where(reflected, lq, -math.log(2))
        + a_reflected * torch.where(reflected, lp, -math.log(2))
        + _log_gamma_increment(a_reflected, r_reflected)
        - torch.lgamma(1 + r_reflected)
        + _fraction(r_reflected, a_reflected, torch.where(reflected, q, 0.5)).log()
    ).clamp_max(0.0)
    cdf_small = torch.where(reflected, log_reflected, cdf_small)
    sf_small = torch.where(
        reflected,
        _log1mexp(torch.where(log_reflected < 0, log_reflected, -1.0)),
        sf_small,
    )
    # A large shape with a small count sums the k + 1 masses of the CDF
    # directly; the continued fraction would need about sqrt(shape) terms.
    # The survival is the CDF's complement while the CDF is below one half.
    summed = active & ~small & (total > 1e3) & (k <= 64)
    count = torch.arange(65, dtype=torch.float64, device=k.device)
    summed_r = torch.where(summed, total, 1.0).unsqueeze(-1)
    log_masses = (
        _log_gamma_increment(summed_r, count)
        - torch.lgamma(count + 1)
        + summed_r * torch.where(summed, lq, -math.log(2)).unsqueeze(-1)
        + count * torch.where(summed, lp, -math.log(2)).unsqueeze(-1)
    )
    log_masses = torch.where(
        count <= torch.where(summed, k, 0.0).unsqueeze(-1), log_masses, -torch.inf
    )
    cdf_summed = torch.logsumexp(log_masses, dim=-1).clamp_max(0.0)
    complement = summed & (cdf_summed < -math.log(2))
    sf_summed = _log1mexp(torch.where(complement, cdf_summed, -1.0))
    generic = active & ~small & (total > 0) & ~complement
    generic_r = torch.where(generic, total, 1.0)
    generic_k = torch.where(generic, k + 1, 1.0)
    cdf, sf = _beta_log_pair(generic_r, generic_k, lq, lp)
    cdf = torch.where(summed, cdf_summed, cdf)
    sf = torch.where(complement, sf_summed, sf)
    cdf, sf = torch.where(small, cdf_small, cdf), torch.where(small, sf_small, sf)
    first = total * lq
    second = total * lq + torch.log1p(total * p)
    cdf = torch.where(k == 0, first, torch.where(k == 1, second, cdf))
    safe_cdf = torch.where(cdf < 0, cdf, -1.0)
    sf = torch.where(short & (cdf < 0), _log1mexp(safe_cdf), sf)
    sf = torch.where(total == 0, -torch.inf, sf)
    return (
        torch.where(active, cdf, torch.where(k < 0, -torch.inf, 0.0)),
        torch.where(active, sf, torch.where(k < 0, 0.0, -torch.inf)),
    )


def inflated_count_tails(zero_prob, value, component_log_cdf, component_log_sf):
    """Combine a count CDF and its shared structural-zero mixture algebra.

    Parameters
    ----------
    zero_prob, value : Tensor
        Valid closed atom probabilities and scalar queries.
    component_log_cdf, component_log_sf : Tensor
        Exact component log tails, broadcast with the query and probability.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        Log CDF, log survival, CDF, and survival. Probability-scale
        outputs retain finite atom derivatives when log derivatives overflow.
    """
    from quivers.continuous._zero_inflated_count import _zero_atom_log_mass

    zero_prob, value, lc, ls = torch.broadcast_tensors(
        zero_prob, value, component_log_cdf, component_log_sf
    )
    probability = torch.where(value >= 0, zero_prob, 0.0)
    cdf = _zero_atom_log_mass(probability, lc)
    sf = torch.log1p(-probability) + ls
    from quivers.continuous._count_cdf import probability_tails

    component_cdf, component_sf = probability_tails(lc, ls)
    direct_cdf = probability + (1 - probability) * component_cdf
    direct_sf = (1 - probability) * component_sf
    return cdf, sf, direct_cdf, direct_sf


def _beta_binomial_tail_sum(n, a, b, counts, upper, max_terms):
    """Sum one complete finite tail through shared mass and ratio kernels.

    Parameters
    ----------
    n, a, b : Tensor
        Flattened validated counts and concentrations.
    counts : Tensor
        Positive numbers of tail cells to include.
    upper : Tensor
        Whether to begin at the upper support endpoint.
    max_terms : int
        Maximum complete tail length per row.

    Returns
    -------
    Tensor
        Exact finite-tail log sums, up to ordinary floating arithmetic.

    Raises
    ------
    RuntimeError
        If a requested complete tail exceeds its resource budget.
    """
    if bool((counts > max_terms).any()):
        raise RuntimeError(
            "BetaBinomial CDF finite tail exceeds its max_terms resource budget"
        )
    pieces, positions = [], []
    # Power-of-two buckets bound count padding by a factor of two; batches
    # retain at most32 rows and4096 count cells at a time.
    high = 1
    low = 0
    while low < max_terms:
        selected = ((counts > low) & (counts <= high)).nonzero().flatten()
        for start in range(0, len(selected), 32):
            at = selected[start : start + 32]
            nn, aa, bb, cc, uu = (v[at] for v in (n, a, b, counts, upper))
            previous = _beta_binomial_log_mass(torch.where(uu, nn, 0.0), nn, aa, bb)
            total = torch.full_like(previous, -torch.inf)
            span = int(cc.max().item())
            for offset in range(0, span, 4096):
                steps = torch.arange(
                    offset, min(span, offset + 4096), dtype=nn.dtype, device=nn.device
                )
                valid = steps[None, :] < cc[:, None]
                transition = valid & (steps[None, :] > 0)
                j = torch.where(uu[:, None], nn[:, None] - steps, steps)
                j = torch.where(
                    transition, j, torch.where(uu[:, None], nn[:, None] - 1, 1.0)
                )
                lj = torch.where(uu[:, None], 1.0, j)
                uj = torch.where(uu[:, None], j, nn[:, None] - 1)
                lower_ratio = (
                    (nn[:, None] - lj + 1).log()
                    - lj.log()
                    + (aa[:, None] + lj - 1).log()
                    - (bb[:, None] + nn[:, None] - lj).log()
                )
                upper_ratio = (
                    (uj + 1).log()
                    - (nn[:, None] - uj).log()
                    + (bb[:, None] + nn[:, None] - uj - 1).log()
                    - (aa[:, None] + uj).log()
                )
                ratio = torch.where(
                    transition, torch.where(uu[:, None], upper_ratio, lower_ratio), 0.0
                )
                values = previous[:, None] + ratio.cumsum(-1)
                previous = values[:, -1]
                any_valid = valid.any(-1)
                safe = torch.where(
                    any_valid[:, None], torch.where(valid, values, -torch.inf), 0.0
                )
                block = torch.where(any_valid, safe.logsumexp(-1), -torch.inf)
                total = torch.logaddexp(total, block)
            pieces.append(total)
            positions.append(at)
        low, high = high, high * 2
    return torch.cat(pieces)[torch.cat(positions).argsort()]


def beta_binomial_log_tails(
    total, value, concentration1, concentration0, *, max_terms=262144
):
    """Evaluate finite beta-binomial tails without an arbitrary series cutoff.

    Parameters
    ----------
    total, value, concentration1, concentration0 : Tensor
        Valid trial counts, real query values, and positive concentrations.
    max_terms : int
        Complete finite-tail cell budget per flattened row.

    Returns
    -------
    tuple[Tensor, Tensor]
        Log CDF and log strict survival. The complementary calculation
        always starts from a directly summed tail of mass at most one half.

    Raises
    ------
    ValueError
        If a query is NaN or the budget is not a positive integer.
    RuntimeError
        If either required complete tail exceeds its explicit resource budget.
    """
    if not isinstance(max_terms, int) or isinstance(max_terms, bool) or max_terms < 1:
        raise ValueError("BetaBinomial CDF max_terms must be a positive integer")
    total, value, a, b = torch.broadcast_tensors(
        total.double(), value.double(), concentration1.double(), concentration0.double()
    )
    if bool(value.isnan().any()):
        raise ValueError("count CDF query must not be NaN")
    shape = total.shape
    n, k, a, b = (v.flatten() for v in (total, value.floor(), a, b))
    active = (k >= 0) & (k < n)
    anchor = 0 * a + 0 * b
    cdf = torch.where(k < 0, -torch.inf, anchor)
    sf = torch.where(k < 0, anchor, -torch.inf)
    if not bool(active.any()):
        return cdf.reshape(shape), sf.reshape(shape)
    nn, kk, aa, bb = (v[active] for v in (n, k, a, b))
    upper = (nn - kk) < (kk + 1)
    counts = torch.where(upper, nn - kk, kk + 1)
    tail = _beta_binomial_tail_sum(nn, aa, bb, counts, upper, max_terms)
    reverse = tail > -math.log(2)
    if bool(reverse.any()):
        opposite = torch.where(
            upper[reverse], kk[reverse] + 1, nn[reverse] - kk[reverse]
        )
        new = _beta_binomial_tail_sum(
            nn[reverse], aa[reverse], bb[reverse], opposite, ~upper[reverse], max_terms
        )
        tail = tail.masked_scatter(reverse, new)
        upper = upper ^ reverse
    if bool((tail > 0).any()):
        raise RuntimeError(
            "BetaBinomial CDF finite-tail normalization exceeded floating capacity"
        )
    other = _log1mexp(tail)
    cdf = cdf.masked_scatter(active, torch.where(upper, other, tail))
    sf = sf.masked_scatter(active, torch.where(upper, tail, other))
    return cdf.reshape(shape), sf.reshape(shape)


__all__: list[str] = []


def bernoulli_log_tails(parameter, value, *, logits=False):
    """Evaluate Bernoulli tails in the original closed or logit coordinates.

    Parameters
    ----------
    parameter, value : Tensor
        Probabilities or logits and real query thresholds.
    logits : bool
        Whether parameters are finite logits.

    Returns
    -------
    tuple[Tensor, Tensor]
        Inclusive log CDF and strict log survival.

    Raises
    ------
    ValueError
        If a query contains NaN.
    """
    parameter, value = torch.broadcast_tensors(
        parameter.double(), torch.as_tensor(value, device=parameter.device).double()
    )
    if bool(value.isnan().any()):
        raise ValueError("count CDF query must not be NaN")
    _, _, lp, lq = _coordinates(parameter, logits=logits)
    return (
        torch.where(value < 0, -torch.inf, torch.where(value < 1, lq, 0.0)),
        torch.where(value < 0, 0.0, torch.where(value < 1, lp, -torch.inf)),
    )


def categorical_log_tails(probs, value, *, logits=False):
    """Evaluate categorical prefix/suffix masses without subtracting tails.

    Parameters
    ----------
    probs : Tensor
        Normalized probabilities or log weights with an intrinsic category axis.
    value : Tensor
        Real query thresholds, independent of that category axis.
    logits : bool
        Whether the native parameter stores log weights.

    Returns
    -------
    tuple[Tensor, Tensor]
        Inclusive log CDF and strict log survival.

    Raises
    ------
    ValueError
        If a query contains NaN.
    """
    value = torch.as_tensor(value, device=probs.device).double()
    if bool(value.isnan().any()):
        raise ValueError("count CDF query must not be NaN")
    shape = torch.broadcast_shapes(probs.shape[:-1], value.shape)
    probs = probs.double().expand(shape + (probs.shape[-1],))
    k = value.expand(shape).floor()
    selected = k.clamp(0, probs.shape[-1] - 1).long().unsqueeze(-1)
    if logits:
        normalized = probs - torch.logsumexp(probs, -1, keepdim=True)
        lower = normalized.logcumsumexp(-1).gather(-1, selected).squeeze(-1)
        suffix = normalized.flip(-1).logcumsumexp(-1).flip(-1)
        suffix = torch.cat(
            [suffix[..., 1:], torch.full_like(suffix[..., :1], -torch.inf)], -1
        )
        upper = suffix.gather(-1, selected).squeeze(-1)
        return (
            torch.where(
                k < 0, -torch.inf, torch.where(k >= probs.shape[-1] - 1, 0.0, lower)
            ),
            torch.where(
                k < 0, 0.0, torch.where(k >= probs.shape[-1] - 1, -torch.inf, upper)
            ),
        )
    lower = probs.cumsum(-1).gather(-1, selected).squeeze(-1)
    suffix = probs.flip(-1).cumsum(-1).flip(-1)
    suffix = torch.cat([suffix[..., 1:], torch.zeros_like(suffix[..., :1])], -1)
    upper = suffix.gather(-1, selected).squeeze(-1)
    lower = torch.where(k < 0, 0.0, torch.where(k >= probs.shape[-1] - 1, 1.0, lower))
    upper = torch.where(k < 0, 1.0, torch.where(k >= probs.shape[-1] - 1, 0.0, upper))
    return (
        torch.where(lower > 0, torch.where(lower > 0, lower, 1.0).log(), -torch.inf),
        torch.where(upper > 0, torch.where(upper > 0, upper, 1.0).log(), -torch.inf),
    )


class _PoissonLogTails(torch.autograd.Function):
    """Fast Gamma-CDF values with analytic, transformable Poisson derivatives."""

    @staticmethod
    def vmap(info, in_dims, *args):
        """Batch native tail evaluation once, keeping convergence checks physical.

        Parameters
        ----------
        info : object
            Torch batching context with the mapped extent.
        in_dims : tuple[int or None, int or None]
            Mapped axes for rate and integer threshold.
        args : tuple[Tensor, Tensor]
            Physical native rate and integer-threshold tensors.

        Returns
        -------
        tuple
            Paired tail outputs and their leading mapped axes.
        """
        values = tuple(
            value.unsqueeze(0).expand((info.batch_size,) + value.shape)
            if axis is None
            else value.movedim(axis, 0)
            for value, axis in zip(args, in_dims, strict=True)
        )
        return _PoissonLogTails.apply(*values), (0, 0)

    @staticmethod
    def forward(rate, count):
        """Evaluate complementary Gamma tails, restoring underflowed logarithms.

        Parameters
        ----------
        rate, count : Tensor
            Positive rates and nonnegative integer thresholds.

        Returns
        -------
        tuple[Tensor, Tensor]
            Paired log count tails.
        """
        from quivers.continuous._gamma_tails import log_gamma_tails

        cdf = torch.special.gammaincc(count + 1, rate)
        sf = torch.special.gammainc(count + 1, rate)
        restore = ((cdf == 0) | (sf == 0)) & (rate > 0)
        lc = torch.where(cdf > 0, torch.where(cdf > 0, cdf, 1.0).log(), -torch.inf)
        ls = torch.where(sf > 0, torch.where(sf > 0, sf, 1.0).log(), -torch.inf)
        if bool(restore.any()):
            gamma_cdf, gamma_sf = log_gamma_tails(count[restore] + 1, rate[restore])
            lc = lc.masked_scatter(restore, gamma_sf)
            ls = ls.masked_scatter(restore, gamma_cdf)
        return lc, ls

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Retain output coordinates for second-order and forward AD.

        Parameters
        ----------
        ctx : object
            Automatic-differentiation context.
        inputs, output : tuple[Tensor, Tensor]
            Rate/count inputs and paired log tails.
        """
        ctx.save_for_backward(*inputs, *output)
        ctx.save_for_forward(*inputs, *output)

    @staticmethod
    def _partials(rate, count, lc, ls):
        """Differentiate the exact Poisson-tail identity through its stable PMF.

        Parameters
        ----------
        rate, count, lc, ls : Tensor
            Native coordinates and paired log tails.

        Returns
        -------
        tuple[Tensor, Tensor]
            Log-CDF and log-survival partials with respect to rate.
        """
        from quivers.continuous._closed_probability import ClosedPoisson

        mass = ClosedPoisson(rate, validate_args=False).log_prob(count)
        cdf = torch.where(
            torch.isfinite(lc),
            (mass - torch.where(torch.isfinite(lc), lc, 0.0)).exp(),
            0.0,
        )
        sf = torch.where(
            torch.isfinite(ls),
            (mass - torch.where(torch.isfinite(ls), ls, 0.0)).exp(),
            0.0,
        )
        return -cdf, sf

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Apply analytic rate partials to reverse-mode derivatives.

        Parameters
        ----------
        ctx : object
            Automatic-differentiation context.
        grad_outputs : tuple[Tensor, Tensor]
            Paired tail upstream derivatives.

        Returns
        -------
        tuple[Tensor, None]
            Rate derivative and no derivative for the integer threshold.
        """
        dc, ds = _PoissonLogTails._partials(*ctx.saved_tensors)
        gc, gs = grad_outputs
        return gc * dc + gs * ds, None

    @staticmethod
    def jvp(ctx, *grad_inputs):
        """Apply the same analytic rate partials in forward mode.

        Parameters
        ----------
        ctx : object
            Automatic-differentiation context.
        grad_inputs : tuple[Tensor, Tensor]
            Rate and threshold tangents.

        Returns
        -------
        tuple[Tensor, Tensor]
            Paired log-tail tangents.
        """
        dc, ds = _PoissonLogTails._partials(*ctx.saved_tensors)
        rate_tangent, _ = grad_inputs
        return rate_tangent * dc, rate_tangent * ds


def poisson_log_tails(rate, value):
    """Evaluate stable Poisson tails with exact zero-rate boundary derivatives.

    Parameters
    ----------
    rate, value : Tensor
        Valid nonnegative rates and real thresholds, including infinities.

    Returns
    -------
    tuple[Tensor, Tensor]
        Inclusive log CDF and strict log survival.

    Raises
    ------
    ValueError
        If a query contains NaN.
    """
    value = torch.as_tensor(value, device=rate.device).double()
    if bool(value.isnan().any()):
        raise ValueError("count CDF query must not be NaN")
    rate, value = torch.broadcast_tensors(rate.double(), value)
    count = value.floor()
    active = (count >= 0) & torch.isfinite(count)
    safe_count = torch.where(active, count, 0.0)
    safe_rate = torch.where(active, rate, 1.0)
    cdf, sf = _PoissonLogTails.apply(safe_rate, safe_count)
    # Short polynomials retain derivatives where the probability of the
    # complementary event is exactly zero at the closed rate boundary.
    first = -rate
    second = -rate + torch.log1p(rate)
    cdf = torch.where(count == 0, first, torch.where(count == 1, second, cdf))
    safe_cdf = torch.where(cdf < 0, cdf, -1.0)
    short_sf = torch.where(cdf < 0, _log1mexp(safe_cdf), -torch.inf)
    sf = torch.where(count <= 1, short_sf, sf)
    return (
        torch.where(active, cdf, torch.where(count < 0, -torch.inf, 0.0)),
        torch.where(active, sf, torch.where(count < 0, 0.0, -torch.inf)),
    )


def positive_count_log_tails(
    value, log_cdf, log_sf, log_zero_mass, log_first, log_first_two
):
    """Condition scalar count tails to be positive without subtracting probabilities.

    Parameters
    ----------
    value, log_cdf, log_sf, log_zero_mass : Tensor
        Query and unrestricted count log tails and zero-cell log mass.
    log_first, log_first_two : Tensor
        Smooth log conditional CDFs at thresholds one and two, retaining
        the degenerate component's parameter derivatives.

    Returns
    -------
    tuple[Tensor, Tensor]
        Inclusive and strict log tails of the positive conditional law.
    """
    value, lc, ls, l0, first, second = torch.broadcast_tensors(
        value, log_cdf, log_sf, log_zero_mass, log_first, log_first_two
    )
    positive_mass = l0 < 0
    safe_zero = torch.where(positive_mass, l0, -1.0)
    normalizer = _log1mexp(safe_zero)
    retained = lc > l0
    difference = torch.where(retained, l0 - lc, -1.0)
    cumulative = lc + _log1mexp(difference) - normalizer
    cumulative = torch.where(retained, cumulative, -torch.inf)
    survival = ls - normalizer
    finite_survival = survival < 0
    safe_survival = torch.where(finite_survival, survival, -1.0)
    complementary = torch.where(finite_survival, _log1mexp(safe_survival), -torch.inf)
    cumulative = torch.where(survival < -math.log(2), complementary, cumulative)
    cumulative = torch.where(positive_mass, cumulative, 0.0)
    survival = torch.where(positive_mass, survival, -torch.inf)
    k = value.floor()
    cumulative = torch.where(k == 1, first, torch.where(k == 2, second, cumulative))
    safe_cumulative = torch.where(cumulative < 0, cumulative, -1.0)
    short_survival = torch.where(cumulative < 0, _log1mexp(safe_cumulative), -torch.inf)
    survival = torch.where((k == 1) | (k == 2), short_survival, survival)
    return (
        torch.where(k < 1, -torch.inf, torch.where(torch.isposinf(k), 0.0, cumulative)),
        torch.where(k < 1, 0.0, torch.where(torch.isposinf(k), -torch.inf, survival)),
    )


def geometric_log_tails(parameter, value, *, logits=False):
    """Evaluate proper geometric tails without rounding finite logit coordinates.

    Parameters
    ----------
    parameter, value : Tensor
        Proper success probabilities or logits and real query thresholds.
    logits : bool
        Whether parameters store original log odds.

    Returns
    -------
    tuple[Tensor, Tensor]
        Inclusive log CDF and strict log survival.

    Raises
    ------
    ValueError
        If a query is NaN.
    """
    parameter, value = torch.broadcast_tensors(
        parameter.double(), torch.as_tensor(value, device=parameter.device).double()
    )
    if bool(value.isnan().any()):
        raise ValueError("count CDF query must not be NaN")
    from quivers.continuous._zero_truncated_poisson import positive_poisson_log_mean

    p, q, success, failure = _coordinates(parameter, logits=logits)
    count = value.floor()
    finite = (count >= 0) & torch.isfinite(count)
    multiplicity = torch.where(finite, count + 1, 1.0)
    exponent = multiplicity * failure
    small = -exponent < 0.01
    safe_rate = torch.where(small, -exponent, 0.0)
    if logits:
        extreme = parameter < -30
        safe_parameter = torch.where(extreme, -1.0, parameter)
        log_hazard = torch.where(
            extreme,
            parameter - torch.where(extreme, parameter, 0.0).exp() / 2,
            F.softplus(safe_parameter).log(),
        )
    else:
        log_hazard = (-torch.where(small, failure, -1.0)).log()
    small_cdf = multiplicity.log() + log_hazard - positive_poisson_log_mean(safe_rate)
    negative = exponent < 0
    cumulative = torch.where(
        small,
        small_cdf,
        torch.where(
            negative, _log1mexp(torch.where(negative, exponent, -1.0)), -torch.inf
        ),
    )
    cumulative = torch.where(
        count == 0,
        success,
        torch.where(count == 1, success + torch.log1p(q), cumulative),
    )
    return (
        torch.where(finite, cumulative, torch.where(count < 0, -torch.inf, 0.0)),
        torch.where(finite, exponent, torch.where(count < 0, 0.0, -torch.inf)),
    )


def distribution_count_log_tails(distribution, value):
    """Adapt bare Torch count laws to the same native paired-tail kernels.

    Parameters
    ----------
    distribution : torch.distributions.Distribution
        Scalar counting distribution with retained constructor coordinates.
    value : Tensor
        Real query thresholds, independent of any category axis.

    Returns
    -------
    tuple[Tensor, Tensor] or None
        Inclusive/strict log tails, or None for an unrelated distribution.

    Raises
    ------
    ValueError
        If a bare Geometric has cached both coordinates and their original
        provenance is unavailable. Use ClosedGeometric to retain it explicitly.
    """
    from torch import distributions as D

    if isinstance(distribution, D.Poisson):
        return poisson_log_tails(distribution.rate, value)
    if isinstance(distribution, D.Geometric):
        stored = distribution.__dict__
        if "probs" in stored and "logits" in stored:
            raise ValueError(
                "bare Torch Geometric probability provenance is ambiguous after "
                "both probs and logits are cached; use ClosedGeometric"
            )
        logits = "logits" in stored
        parameter = stored["logits" if logits else "probs"]
        if bool(
            (
                torch.isnan(parameter)
                | (torch.isneginf(parameter) if logits else (parameter <= 0))
            ).any()
        ):
            raise ValueError(
                "Geometric requires a proper success probability in (0, 1]"
            )
        return geometric_log_tails(parameter, value, logits=logits)
    if isinstance(
        distribution, (D.Bernoulli, D.Binomial, D.NegativeBinomial, D.Categorical)
    ):
        raw = cast(Tensor, distribution._param)
        logits = raw is not distribution.__dict__.get("probs")
        if isinstance(distribution, D.Categorical):
            return categorical_log_tails(raw, value, logits=logits)
        if isinstance(distribution, D.Bernoulli):
            return bernoulli_log_tails(raw, value, logits=logits)
        kernel = (
            binomial_log_tails
            if isinstance(distribution, D.Binomial)
            else negative_binomial_log_tails
        )
        return kernel(distribution.total_count, value, raw, logits=logits)
    return None
