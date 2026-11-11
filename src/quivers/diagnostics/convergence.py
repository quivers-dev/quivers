"""Convergence diagnostics for chain-by-draw tensors.

The functions here take draws laid out ``(num_chains, num_draws,
*event_shape)`` and return one value per event coordinate, with shape
``event_shape``. They implement the estimators of Vehtari, Gelman,
Simpson, Carpenter, and Bürkner (2021,
`doi:10.1214/20-BA1221 <https://doi.org/10.1214/20-BA1221>`_):

* `rhat`: rank-normalized split :math:`\\hat R`, reported as the
  maximum of the bulk (rank-normalized) and tail (folded,
  rank-normalized) statistics.
* `ess`: effective sample size for the bulk (rank-normalized split
  chains), the tails (the smaller of the 5% and 95% quantile ESS by
  default), the mean, the standard deviation, or one quantile.
* `mcse`: Monte Carlo standard error of the mean, the standard
  deviation, or one quantile.
* `summary`: every diagnostic above for each site of a mapping of
  draws, as a `ConvergenceSummary`.

Autocorrelations pool the chains and include the between-chain
variance, and the effective sample size truncates the autocorrelation
sum with Geyer's initial monotone sequence. The definitions agree with
ArviZ's ``rhat``, ``ess``, and ``mcse`` on finite, non-constant input.

The module needs only torch. `quivers.inference.MCMCResult` computes
its ``r_hat``, ``ess``, and ``ess_tail`` through these functions, so
diagnostics of derived quantities and of draws from custom sampling
loops are directly comparable with the diagnostics a run reports.

Degenerate coordinates return NaN rather than a number that would
read as a valid diagnostic:

* a coordinate with any non-finite value (NaN or infinite);
* a coordinate with fewer than four draws per chain;
* a constant coordinate, whose largest and smallest draws differ by
  less than the float64 resolution (``1e-15``); a chain stuck at one
  value has no estimable mixing, so its effective sample size is NaN
  rather than the total draw count ArviZ reports;
* for `rhat` only, fewer than two chains.

The diagnostics are not differentiable: draws are detached and
computed in float64 on the CPU, and results return on the input's
device as float64.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from typing import Literal

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel

EssMethod = Literal["bulk", "tail", "mean", "sd", "quantile"]
"""The quantity an effective sample size refers to: ``"bulk"``
(rank-normalized split chains, for location summaries), ``"tail"``
(the smaller of two tail-quantile ESS values), ``"mean"`` (the
unnormalized split chains), ``"sd"`` (the squared deviations from the
mean), or ``"quantile"`` (the indicator of falling at or below one
quantile)."""

McseMethod = Literal["mean", "sd", "quantile"]
"""The posterior summary a Monte Carlo standard error refers to: the
``"mean"``, the standard deviation (``"sd"``), or one ``"quantile"``."""

_RESOLUTION = float(torch.finfo(torch.float64).resolution)
_MIN_DRAWS = 4
_DEFAULT_TAIL_PROB = 0.05
# Standard normal CDF at -1 and +1: the central one-sd interval of the
# quantile estimate's beta approximation.
_ONE_SD_LOWER = 0.1586553
_ONE_SD_UPPER = 0.8413447
_CF_EPS = 1e-15
_CF_TINY = 1e-300


class ConvergenceSummary(ValueModel):
    """Posterior summaries and convergence diagnostics for one site.

    Every tensor field except ``quantiles`` and ``mcse_quantiles`` has
    the site's event shape (the shape after the chain and draw axes);
    those two have shape ``(len(probs), *event_shape)``. All are
    float64.

    Attributes
    ----------
    mean : torch.Tensor
        Posterior mean over all chains and draws.
    sd : torch.Tensor
        Posterior standard deviation over all chains and draws
        (one degree of freedom subtracted).
    probs : tuple[float, ...]
        The probabilities at which ``quantiles`` are taken.
    quantiles : torch.Tensor
        Posterior quantiles at ``probs`` (linear interpolation between
        order statistics of the pooled draws); NaN for a coordinate
        with a non-finite draw.
    r_hat : torch.Tensor
        Rank-normalized split :math:`\\hat R` (see `rhat`).
    ess_bulk : torch.Tensor
        Bulk effective sample size (see `ess`).
    ess_tail : torch.Tensor
        Tail effective sample size at the 5% and 95% quantiles.
    mcse_mean : torch.Tensor
        Monte Carlo standard error of the mean.
    mcse_sd : torch.Tensor
        Monte Carlo standard error of the standard deviation.
    mcse_quantiles : torch.Tensor
        Monte Carlo standard error of each quantile in ``quantiles``.
    """

    mean: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    sd: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    probs: tuple[float, ...]
    quantiles: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    r_hat: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    ess_bulk: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    ess_tail: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    mcse_mean: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    mcse_sd: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    mcse_quantiles: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def _as_chains(draws: torch.Tensor) -> torch.Tensor:
    """Validate a ``(chains, draws, *event)`` tensor and flatten its event.

    Parameters
    ----------
    draws : torch.Tensor
        Draws laid out ``(num_chains, num_draws, *event_shape)``.

    Returns
    -------
    torch.Tensor
        Detached float64 CPU tensor of shape
        ``(num_chains, num_draws, prod(event_shape))``.

    Raises
    ------
    TypeError
        If ``draws`` is not a tensor or holds complex values.
    ValueError
        If ``draws`` has fewer than two axes, or no chains or draws.
    """
    if not isinstance(draws, torch.Tensor):
        raise TypeError(f"draws must be a torch.Tensor, got {type(draws).__name__}")
    if draws.is_complex():
        raise TypeError("draws must be real-valued, got a complex tensor")
    if draws.dim() < 2:
        raise ValueError(
            "draws must be laid out (num_chains, num_draws, *event_shape); "
            f"got shape {tuple(draws.shape)}"
        )
    if draws.shape[0] == 0 or draws.shape[1] == 0:
        raise ValueError(
            f"draws need at least one chain and one draw; got shape {tuple(draws.shape)}"
        )
    num_coordinates = math.prod(draws.shape[2:])
    return (
        draws.detach()
        .to(device="cpu", dtype=torch.float64)
        .reshape(draws.shape[0], draws.shape[1], num_coordinates)
    )


def _is_degenerate(chains: torch.Tensor, min_chains: int) -> bool:
    """Whether one coordinate's ``(chains, draws)`` matrix has no estimate.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate's draws, shape ``(num_chains, num_draws)``.
    min_chains : int
        The fewest chains the statistic needs.

    Returns
    -------
    bool
        True for too few chains or draws, a non-finite draw, or a
        constant coordinate.
    """
    if chains.shape[0] < min_chains or chains.shape[1] < _MIN_DRAWS:
        return True
    if not bool(torch.isfinite(chains).all()):
        return True
    return float(chains.max() - chains.min()) < _RESOLUTION


def _per_coordinate(
    draws: torch.Tensor,
    statistic: Callable[[torch.Tensor], float],
    *,
    min_chains: int,
) -> torch.Tensor:
    """Apply a scalar statistic to every event coordinate of ``draws``.

    Parameters
    ----------
    draws : torch.Tensor
        Draws laid out ``(num_chains, num_draws, *event_shape)``.
    statistic : Callable[[torch.Tensor], float]
        Maps one coordinate's ``(num_chains, num_draws)`` float64
        matrix to a float.
    min_chains : int
        The fewest chains the statistic needs.

    Returns
    -------
    torch.Tensor
        Float64 tensor of shape ``event_shape`` on ``draws.device``,
        NaN at degenerate coordinates.
    """
    chains = _as_chains(draws)
    out = torch.empty(chains.shape[2], dtype=torch.float64)
    for coordinate in range(chains.shape[2]):
        values = chains[:, :, coordinate]
        out[coordinate] = (
            math.nan if _is_degenerate(values, min_chains) else statistic(values)
        )
    return out.reshape(draws.shape[2:]).to(device=draws.device)


def _split_chains(chains: torch.Tensor) -> torch.Tensor:
    """Split each chain in half, dropping the middle draw when odd.

    Parameters
    ----------
    chains : torch.Tensor
        Shape ``(num_chains, num_draws)``.

    Returns
    -------
    torch.Tensor
        Shape ``(2 * num_chains, num_draws // 2)``.
    """
    half = chains.shape[1] // 2
    return torch.cat([chains[:, :half], chains[:, -half:]], dim=0)


def _z_scale(chains: torch.Tensor) -> torch.Tensor:
    """Rank-normalize draws with average ranks and Blom's normal scores.

    Parameters
    ----------
    chains : torch.Tensor
        Any float64 tensor; ranks pool all of its elements.

    Returns
    -------
    torch.Tensor
        The normal score of each element's average rank, same shape.
    """
    flat = chains.reshape(-1)
    sorted_values, order = torch.sort(flat, stable=True)
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    last_rank = torch.cumsum(counts, dim=0).to(dtype=torch.float64)
    first_rank = last_rank - counts.to(dtype=torch.float64) + 1.0
    average_rank = torch.repeat_interleave((first_rank + last_rank) / 2.0, counts)
    ranks = torch.empty_like(flat)
    ranks[order] = average_rank
    probabilities = (ranks - 3.0 / 8.0) / (flat.numel() + 1.0 / 4.0)
    return (math.sqrt(2.0) * torch.erfinv(2.0 * probabilities - 1.0)).reshape(
        chains.shape
    )


def _linear_quantile(sorted_values: torch.Tensor, prob: float) -> torch.Tensor:
    """Quantile by linear interpolation between order statistics.

    This is Hyndman and Fan's type 7 definition, NumPy's default,
    including its two-sided interpolation formula.

    Parameters
    ----------
    sorted_values : torch.Tensor
        Values sorted ascending along the first axis.
    prob : float
        Probability in ``[0, 1]``.

    Returns
    -------
    torch.Tensor
        The quantile, with the first axis reduced.
    """
    size = sorted_values.shape[0]
    position = (size - 1) * prob
    lower = min(int(math.floor(position)), size - 1)
    upper = min(lower + 1, size - 1)
    fraction = position - lower
    below = sorted_values[lower]
    above = sorted_values[upper]
    difference = above - below
    if fraction >= 0.5:
        return above - difference * (1.0 - fraction)
    return below + difference * fraction


def _quantile(chains: torch.Tensor, prob: float) -> float:
    """Pooled linear-interpolation quantile of every element of ``chains``.

    Parameters
    ----------
    chains : torch.Tensor
        Any float64 tensor.
    prob : float
        Probability in ``[0, 1]``.

    Returns
    -------
    float
        The quantile.
    """
    return float(_linear_quantile(torch.sort(chains.reshape(-1)).values, prob))


def _rhat_core(chains: torch.Tensor) -> float:
    """Potential scale reduction factor of already transformed chains.

    Parameters
    ----------
    chains : torch.Tensor
        Shape ``(num_chains, num_draws)``.

    Returns
    -------
    float
        :math:`\\sqrt{(B/W + n - 1) / n}` with ``B`` the between-chain
        and ``W`` the mean within-chain variance.
    """
    num_draws = chains.shape[1]
    between = num_draws * float(chains.mean(dim=1).var(unbiased=True))
    within = float(chains.var(dim=1, unbiased=True).mean())
    ratio = between / within if within != 0.0 else _divide_by_zero(between)
    return math.sqrt((ratio + num_draws - 1) / num_draws)


def _divide_by_zero(numerator: float) -> float:
    """IEEE division of ``numerator`` by zero.

    Parameters
    ----------
    numerator : float
        The dividend.

    Returns
    -------
    float
        ``inf`` for a positive numerator, ``nan`` for zero.
    """
    return math.inf if numerator > 0.0 else math.nan


def _autocov(chains: torch.Tensor) -> torch.Tensor:
    """Biased per-chain autocovariance at every lag, by FFT.

    Parameters
    ----------
    chains : torch.Tensor
        Shape ``(num_chains, num_draws)``.

    Returns
    -------
    torch.Tensor
        Shape ``(num_chains, num_draws)``; entry ``[m, t]`` is chain
        ``m``'s autocovariance at lag ``t``.
    """
    num_draws = chains.shape[1]
    centered = chains - chains.mean(dim=1, keepdim=True)
    pad = 1
    while pad < 2 * num_draws:
        pad *= 2
    spectrum = torch.fft.rfft(centered, n=pad, dim=1)
    covariance = torch.fft.irfft(spectrum * torch.conj(spectrum), n=pad, dim=1)
    return covariance[:, :num_draws] / num_draws


def _ess_core(chains: torch.Tensor) -> float:
    """Cross-chain effective sample size of already transformed chains.

    Autocorrelations combine the mean within-chain autocovariance with
    the between-chain variance, and the sum is truncated by Geyer's
    initial positive sequence and made monotone. A transformed matrix
    that is constant (an indicator that never changes, say) returns
    the total draw count, which is the estimator's limit as the
    autocorrelations vanish.

    Parameters
    ----------
    chains : torch.Tensor
        Shape ``(num_chains, num_draws)``, float64.

    Returns
    -------
    float
        The effective sample size.
    """
    if float(chains.max() - chains.min()) < _RESOLUTION:
        return float(chains.numel())
    num_chains, num_draws = chains.shape
    autocovariance: list[float] = _autocov(chains).mean(dim=0).tolist()
    mean_variance = autocovariance[0] * num_draws / (num_draws - 1.0)
    pooled_variance = mean_variance * (num_draws - 1.0) / num_draws
    if num_chains > 1:
        pooled_variance += float(chains.mean(dim=1).var(unbiased=True))

    rho = [0.0] * num_draws
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = 1.0 - (mean_variance - autocovariance[1]) / pooled_variance
    rho[1] = rho_odd

    lag = 1
    while lag < num_draws - 3 and rho_even + rho_odd > 0.0:
        rho_even = 1.0 - (mean_variance - autocovariance[lag + 1]) / pooled_variance
        rho_odd = 1.0 - (mean_variance - autocovariance[lag + 2]) / pooled_variance
        if rho_even + rho_odd >= 0.0:
            rho[lag + 1] = rho_even
            rho[lag + 2] = rho_odd
        lag += 2

    max_lag = lag - 2
    if rho_even > 0.0:
        rho[max_lag + 1] = rho_even

    lag = 1
    while lag <= max_lag - 2:
        previous_pair = rho[lag - 1] + rho[lag]
        if rho[lag + 1] + rho[lag + 2] > previous_pair:
            rho[lag + 1] = previous_pair / 2.0
            rho[lag + 2] = previous_pair / 2.0
        lag += 2

    if any(math.isnan(value) for value in rho):
        return math.nan
    total = num_chains * num_draws
    tau = -1.0 + 2.0 * math.fsum(rho[: max_lag + 1]) + math.fsum(
        rho[max_lag + 1 : max_lag + 2]
    )
    tau = max(tau, 1.0 / math.log10(total))
    return total / tau


def _rank_rhat(chains: torch.Tensor) -> float:
    """Rank-normalized split R-hat: the larger of bulk and folded.

    When the folded draws are constant (draws symmetric about the
    median at one distance), the tail statistic is undefined and the
    bulk statistic is returned.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.

    Returns
    -------
    float
        The statistic.
    """
    split = _split_chains(chains)
    bulk = _rhat_core(_z_scale(split))
    folded = torch.abs(split - _quantile(split, 0.5))
    tail = _rhat_core(_z_scale(folded))
    if math.isnan(tail):
        return bulk
    return max(bulk, tail)


def _ess_bulk(chains: torch.Tensor) -> float:
    """Bulk ESS: ESS of the rank-normalized split chains.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.

    Returns
    -------
    float
        The effective sample size.
    """
    return _ess_core(_z_scale(_split_chains(chains)))


def _ess_mean(chains: torch.Tensor) -> float:
    """ESS for the mean: ESS of the untransformed split chains.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.

    Returns
    -------
    float
        The effective sample size.
    """
    return _ess_core(_split_chains(chains))


def _ess_sd(chains: torch.Tensor) -> float:
    """ESS for the standard deviation: ESS of the squared deviations.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.

    Returns
    -------
    float
        The effective sample size.
    """
    return _ess_core(_split_chains((chains - chains.mean()) ** 2))


def _ess_quantile(chains: torch.Tensor, prob: float) -> float:
    """ESS for one quantile: ESS of the at-or-below indicator.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.
    prob : float
        The quantile's probability.

    Returns
    -------
    float
        The effective sample size.
    """
    indicator = (chains <= _quantile(chains, prob)).to(dtype=torch.float64)
    return _ess_core(_split_chains(indicator))


def _ess_tail(chains: torch.Tensor, prob: float) -> float:
    """Tail ESS: the smaller quantile ESS at ``prob`` and ``1 - prob``.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.
    prob : float
        One tail probability; the other is its complement.

    Returns
    -------
    float
        The effective sample size.
    """
    low, high = sorted((prob, 1.0 - prob))
    return min(_ess_quantile(chains, low), _ess_quantile(chains, high))


def _mcse_mean(chains: torch.Tensor) -> float:
    """Monte Carlo standard error of the mean.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.

    Returns
    -------
    float
        ``sd / sqrt(ess_mean)``.
    """
    return float(chains.std(unbiased=True)) / math.sqrt(_ess_mean(chains))


def _mcse_sd(chains: torch.Tensor) -> float:
    """Monte Carlo standard error of the standard deviation.

    The variance of the variance estimate uses the fourth central
    moment and the ESS of the squared deviations; the delta method
    carries it to the standard deviation.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.

    Returns
    -------
    float
        The standard error.
    """
    centered = chains - chains.mean()
    squared = centered**2
    effective = _ess_mean(squared)
    variance = float(squared.mean())
    variance_of_variance = (float((squared**2).mean()) - variance**2) / effective
    return math.sqrt(variance_of_variance / variance / 4.0)


def _mcse_quantile(chains: torch.Tensor, prob: float) -> float:
    """Monte Carlo standard error of one quantile.

    The quantile's ESS sets a beta distribution for the probability
    level of the order statistic that estimates it; the error is half
    the distance between the order statistics at that distribution's
    central one-sd interval.

    Parameters
    ----------
    chains : torch.Tensor
        One coordinate, shape ``(num_chains, num_draws)``.
    prob : float
        The quantile's probability.

    Returns
    -------
    float
        The standard error.
    """
    effective = _ess_quantile(chains, prob)
    if not math.isfinite(effective):
        return math.nan
    alpha = effective * prob + 1.0
    beta = effective * (1.0 - prob) + 1.0
    lower_level = _beta_ppf(_ONE_SD_LOWER, alpha, beta)
    upper_level = _beta_ppf(_ONE_SD_UPPER, alpha, beta)
    sorted_values = torch.sort(chains.reshape(-1)).values
    size = sorted_values.numel()
    lower_index = int(math.floor(max(lower_level * size - 1.0, 0.0)))
    upper_index = int(math.ceil(min(upper_level * size - 1.0, size - 1.0)))
    return float(sorted_values[upper_index] - sorted_values[lower_index]) / 2.0


def _beta_continued_fraction(x: float, alpha: float, beta: float) -> float:
    """Continued fraction of the regularized incomplete beta function.

    Evaluated by the modified Lentz method; converges quickly for
    ``x < (alpha + 1) / (alpha + beta + 2)``.

    Parameters
    ----------
    x : float
        Point in ``(0, 1)``.
    alpha : float
        First shape parameter, positive.
    beta : float
        Second shape parameter, positive.

    Returns
    -------
    float
        The continued fraction's value.

    Raises
    ------
    RuntimeError
        If the fraction does not converge.
    """
    total = alpha + beta
    c = 1.0
    d = 1.0 - total * x / (alpha + 1.0)
    d = 1.0 / (d if abs(d) > _CF_TINY else _CF_TINY)
    value = d
    max_terms = 200 + int(20.0 * math.sqrt(max(alpha, beta)))
    for m in range(1, max_terms + 1):
        m2 = 2 * m
        even = m * (beta - m) * x / ((alpha - 1.0 + m2) * (alpha + m2))
        d = 1.0 + even * d
        d = 1.0 / (d if abs(d) > _CF_TINY else _CF_TINY)
        c = 1.0 + even / c
        c = c if abs(c) > _CF_TINY else _CF_TINY
        value *= d * c
        odd = -(alpha + m) * (total + m) * x / ((alpha + m2) * (alpha + 1.0 + m2))
        d = 1.0 + odd * d
        d = 1.0 / (d if abs(d) > _CF_TINY else _CF_TINY)
        c = 1.0 + odd / c
        c = c if abs(c) > _CF_TINY else _CF_TINY
        step = d * c
        value *= step
        if abs(step - 1.0) < _CF_EPS:
            return value
    raise RuntimeError(
        f"incomplete beta continued fraction did not converge for x={x}, "
        f"alpha={alpha}, beta={beta}"
    )


def _beta_cdf(x: float, alpha: float, beta: float) -> float:
    """Regularized incomplete beta function :math:`I_x(\\alpha, \\beta)`.

    Parameters
    ----------
    x : float
        Point in ``[0, 1]``.
    alpha : float
        First shape parameter, positive.
    beta : float
        Second shape parameter, positive.

    Returns
    -------
    float
        The Beta(alpha, beta) distribution function at ``x``.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        alpha * math.log(x)
        + beta * math.log1p(-x)
        - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
    )
    if x < (alpha + 1.0) / (alpha + beta + 2.0):
        return math.exp(log_front) * _beta_continued_fraction(x, alpha, beta) / alpha
    return 1.0 - math.exp(log_front) * _beta_continued_fraction(
        1.0 - x, beta, alpha
    ) / beta


def _beta_ppf(level: float, alpha: float, beta: float) -> float:
    """Quantile function of the Beta(alpha, beta) distribution.

    Bisection on the distribution function until the bracket stops
    shrinking in float64.

    Parameters
    ----------
    level : float
        Probability in ``(0, 1)``.
    alpha : float
        First shape parameter, positive.
    beta : float
        Second shape parameter, positive.

    Returns
    -------
    float
        The ``level`` quantile.
    """
    low, high = 0.0, 1.0
    while True:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            return middle
        if _beta_cdf(middle, alpha, beta) < level:
            low = middle
        else:
            high = middle


def _check_prob(prob: float, name: str) -> float:
    """Validate a probability strictly inside ``(0, 1)``.

    Parameters
    ----------
    prob : float
        The probability.
    name : str
        The argument name, for the error message.

    Returns
    -------
    float
        ``prob`` as a float.

    Raises
    ------
    ValueError
        If ``prob`` is not strictly between 0 and 1.
    """
    value = float(prob)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1, got {prob}")
    return value


def rhat(draws: torch.Tensor) -> torch.Tensor:
    """Rank-normalized split :math:`\\hat R` per event coordinate.

    Each chain is split in half (the middle draw is dropped when the
    draw count is odd), the pooled draws are replaced by the normal
    scores of their average ranks, and the potential scale reduction
    factor is computed for those scores (the bulk statistic) and for
    the scores of the absolute deviations from the median (the tail
    statistic). The reported value is the larger of the two, following
    Vehtari et al. (2021). Values near 1 are consistent with the chains
    sampling one distribution; the authors suggest 1.01 as a threshold.

    Parameters
    ----------
    draws : torch.Tensor
        Draws laid out ``(num_chains, num_draws, *event_shape)``.

    Returns
    -------
    torch.Tensor
        Float64 tensor of shape ``event_shape`` on ``draws.device``.
        NaN for a coordinate with fewer than two chains, fewer than
        four draws per chain, a non-finite draw, or a constant value.
        ``inf`` when each split chain is constant but the chains
        differ.

    Raises
    ------
    TypeError
        If ``draws`` is not a real tensor.
    ValueError
        If ``draws`` has fewer than two axes, or no chains or draws.
    """
    return _per_coordinate(draws, _rank_rhat, min_chains=2)


def ess(
    draws: torch.Tensor,
    *,
    method: EssMethod = "bulk",
    prob: float | None = None,
) -> torch.Tensor:
    """Effective sample size per event coordinate.

    All methods split each chain in half and pool the autocorrelations
    across chains (Vehtari et al., 2021):

    * ``"bulk"``: ESS of the rank-normalized draws, the measure for
      location summaries such as the mean and median.
    * ``"tail"``: the smaller of the ``"quantile"`` ESS at ``prob``
      and ``1 - prob`` (``prob`` defaults to 0.05, giving the 5% and
      95% quantiles), the measure for interval endpoints.
    * ``"mean"``: ESS of the untransformed draws.
    * ``"sd"``: ESS of the squared deviations from the mean.
    * ``"quantile"``: ESS of the indicator of falling at or below the
      ``prob`` quantile; ``prob`` is required.

    Parameters
    ----------
    draws : torch.Tensor
        Draws laid out ``(num_chains, num_draws, *event_shape)``.
    method : EssMethod
        Which effective sample size to compute.
    prob : float or None
        The tail probability for ``"tail"`` (default 0.05) or the
        quantile's probability for ``"quantile"`` (required). Must be
        ``None`` for the other methods.

    Returns
    -------
    torch.Tensor
        Float64 tensor of shape ``event_shape`` on ``draws.device``.
        NaN for a coordinate with fewer than four draws per chain, a
        non-finite draw, or a constant value.

    Raises
    ------
    TypeError
        If ``draws`` is not a real tensor.
    ValueError
        If ``draws`` has fewer than two axes or no chains or draws, if
        ``method`` is unknown, or if ``prob`` is missing, out of
        ``(0, 1)``, or given to a method that takes none.
    """
    statistic: Callable[[torch.Tensor], float]
    if method == "tail":
        tail_prob = _check_prob(_DEFAULT_TAIL_PROB if prob is None else prob, "prob")

        def statistic(chains: torch.Tensor) -> float:
            return _ess_tail(chains, tail_prob)

    elif method == "quantile":
        if prob is None:
            raise ValueError('ess(method="quantile") requires prob')
        quantile_prob = _check_prob(prob, "prob")

        def statistic(chains: torch.Tensor) -> float:
            return _ess_quantile(chains, quantile_prob)

    elif method in ("bulk", "mean", "sd"):
        if prob is not None:
            raise ValueError(f'ess(method="{method}") takes no prob')
        statistic = {"bulk": _ess_bulk, "mean": _ess_mean, "sd": _ess_sd}[method]
    else:
        raise ValueError(
            f"unknown ESS method {method!r}; expected one of "
            "'bulk', 'tail', 'mean', 'sd', 'quantile'"
        )
    return _per_coordinate(draws, statistic, min_chains=1)


def mcse(
    draws: torch.Tensor,
    *,
    method: McseMethod = "mean",
    prob: float | None = None,
) -> torch.Tensor:
    """Monte Carlo standard error per event coordinate.

    * ``"mean"``: the posterior standard deviation over the square
      root of the ``"mean"`` ESS.
    * ``"sd"``: the delta-method error of the standard deviation, from
      the fourth central moment and the ESS of the squared deviations.
    * ``"quantile"``: half the distance between the order statistics
      at the central one-sd interval of the beta distribution that the
      quantile's ESS implies for its probability level (Vehtari et
      al., 2021); ``prob`` is required.

    Parameters
    ----------
    draws : torch.Tensor
        Draws laid out ``(num_chains, num_draws, *event_shape)``.
    method : McseMethod
        Which summary's standard error to compute.
    prob : float or None
        The quantile's probability for ``"quantile"`` (required).
        Must be ``None`` for the other methods.

    Returns
    -------
    torch.Tensor
        Float64 tensor of shape ``event_shape`` on ``draws.device``.
        NaN for a coordinate with fewer than four draws per chain, a
        non-finite draw, or a constant value.

    Raises
    ------
    TypeError
        If ``draws`` is not a real tensor.
    ValueError
        If ``draws`` has fewer than two axes or no chains or draws, if
        ``method`` is unknown, or if ``prob`` is missing, out of
        ``(0, 1)``, or given to a method that takes none.
    """
    statistic: Callable[[torch.Tensor], float]
    if method == "quantile":
        if prob is None:
            raise ValueError('mcse(method="quantile") requires prob')
        quantile_prob = _check_prob(prob, "prob")

        def statistic(chains: torch.Tensor) -> float:
            return _mcse_quantile(chains, quantile_prob)

    elif method in ("mean", "sd"):
        if prob is not None:
            raise ValueError(f'mcse(method="{method}") takes no prob')
        statistic = {"mean": _mcse_mean, "sd": _mcse_sd}[method]
    else:
        raise ValueError(
            f"unknown MCSE method {method!r}; expected one of 'mean', 'sd', 'quantile'"
        )
    return _per_coordinate(draws, statistic, min_chains=1)


def _pooled_quantiles(draws: torch.Tensor, probs: Sequence[float]) -> torch.Tensor:
    """Linear-interpolation quantiles of the pooled draws per coordinate.

    Parameters
    ----------
    draws : torch.Tensor
        Draws laid out ``(num_chains, num_draws, *event_shape)``.
    probs : Sequence[float]
        Probabilities in ``(0, 1)``.

    Returns
    -------
    torch.Tensor
        Float64, shape ``(len(probs), *event_shape)`` on
        ``draws.device``; NaN at coordinates with a non-finite draw.
    """
    chains = _as_chains(draws)
    pooled = chains.reshape(-1, chains.shape[2])
    finite = torch.isfinite(pooled).all(dim=0)
    sorted_values = torch.sort(pooled, dim=0).values
    rows = [
        torch.where(finite, _linear_quantile(sorted_values, p), math.nan)
        for p in probs
    ]
    stacked = torch.stack(rows) if rows else pooled.new_empty((0, pooled.shape[1]))
    return stacked.reshape(len(probs), *draws.shape[2:]).to(device=draws.device)


def summary(
    samples: Mapping[str, torch.Tensor],
    *,
    probs: Sequence[float] = (0.05, 0.5, 0.95),
) -> dict[str, ConvergenceSummary]:
    """Posterior summaries and convergence diagnostics for every site.

    Parameters
    ----------
    samples : Mapping[str, torch.Tensor]
        Per-site draws, each laid out
        ``(num_chains, num_draws, *event_shape)``, such as
        `quivers.inference.MCMCResult.samples`.
    probs : Sequence[float]
        Probabilities of the quantiles to report, each in ``(0, 1)``.

    Returns
    -------
    dict[str, ConvergenceSummary]
        One summary per site, in the order of ``samples``.

    Raises
    ------
    TypeError
        If a site's draws are not a real tensor.
    ValueError
        If a site's draws have fewer than two axes or no chains or
        draws, or a probability lies outside ``(0, 1)``.
    """
    levels = tuple(_check_prob(p, "probs") for p in probs)
    summaries: dict[str, ConvergenceSummary] = {}
    for name, draws in samples.items():
        pooled = _as_chains(draws)
        event_shape = draws.shape[2:]
        mean = pooled.mean(dim=(0, 1)).reshape(event_shape)
        sd = pooled.reshape(-1, pooled.shape[2]).std(dim=0, unbiased=True)
        mcse_rows = [mcse(draws, method="quantile", prob=p) for p in levels]
        summaries[name] = ConvergenceSummary(
            mean=mean.to(device=draws.device),
            sd=sd.reshape(event_shape).to(device=draws.device),
            probs=levels,
            quantiles=_pooled_quantiles(draws, levels),
            r_hat=rhat(draws),
            ess_bulk=ess(draws, method="bulk"),
            ess_tail=ess(draws, method="tail"),
            mcse_mean=mcse(draws, method="mean"),
            mcse_sd=mcse(draws, method="sd"),
            mcse_quantiles=(
                torch.stack(mcse_rows)
                if mcse_rows
                else torch.empty((0, *event_shape), dtype=torch.float64)
            ).to(device=draws.device),
        )
    return summaries


__all__ = [
    "EssMethod",
    "McseMethod",
    "ConvergenceSummary",
    "rhat",
    "ess",
    "mcse",
    "summary",
]
