"""Simulation-based calibration (SBC) of Bayesian inference procedures.

SBC (Talts, Betancourt, Simpson, Vehtari, and Gelman, 2018,
`doi:10.48550/arXiv.1804.06788 <https://doi.org/10.48550/arXiv.1804.06788>`_)
checks an inference procedure against the model it targets. Each
simulation draws parameters from the prior and data from the model
given those parameters, fits the simulated data, and records the rank
of each true quantity among the posterior draws of that quantity. If
the procedure samples the exact posterior, every rank is uniform on
``{0, ..., L}`` for ``L`` posterior draws, whatever the model; a
posterior that is too narrow piles ranks at the ends, one too wide
piles them in the middle, and a biased one tilts them.

* `sbc` runs the whole procedure for a program and a caller-supplied
  fit (MCMC, a guide, or any sampler returning chain-by-draw tensors)
  and returns an `SBCResult`.
* `sbc_ranks` is the rank primitive, with ties between the true value
  and posterior draws broken uniformly at random, so that discrete and
  mixed quantities have uniform ranks too.
* `sbc_uniformity` tests the ranks against the discrete uniform
  distribution with the simultaneous ECDF statistic of Säilynoja,
  Bürkner, and Vehtari (2022,
  `doi:10.1007/s11222-022-10090-6 <https://doi.org/10.1007/s11222-022-10090-6>`_),
  calibrated by Monte Carlo under the exact discrete null.
* `sbc_coverage` reports how often the central rank band holds the
  true value, against that band's exact null mass, with
  Clopper-Pearson intervals adjusted for the number of statistics.

The module needs only torch; `quivers.diagnostics.to_datatree_from_sbc`
converts an `SBCResult` for ArviZ's PIT-ECDF plots.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from fractions import Fraction
import math
from typing import Literal, Protocol, cast

import didactic.api as dx
import torch

from quivers._values import OPAQUE_BY_IDENTITY, ValueModel
from quivers.continuous.programs import MonadicProgram
from quivers.diagnostics.convergence import _beta_ppf, ess
from quivers.inference.guides.base import Guide
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry, _require_one_run
from quivers.inference.trace import Trace, trace

SBCStatistic = Callable[
    [Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]], torch.Tensor
]
"""A test quantity: ``statistic(parameters, data)``. ``parameters``
maps each latent site to a batch of draws with a leading draw axis,
shape ``(n, *constrained_shape)``; ``data`` is the simulated
observations together with the host data, as handed to the fit. The
result has shape ``(n,)`` for one quantity or ``(n, *shape)`` for an
array of them, one row per draw."""

SBCPosterior = MCMCResult | Guide | Mapping[str, torch.Tensor]
"""What a fit returns: an `MCMCResult`, a fitted `Guide` (sampled
``num_guide_draws`` times), or a mapping of site name to draws laid
out ``(num_chains, num_draws, *constrained_shape)``."""

SBCThinning = Literal["ess"] | int
"""How posterior draws are thinned before ranking: ``"ess"`` keeps
about one draw per effective sample, an integer keeps every k-th draw
of each chain."""

_INT64_MAX = 2**63 - 1
_REFERENCE_CHUNK = 1024


class _RngBackend(Protocol):
    """The device-state operations used by a seeded calibration scope."""

    def device_count(self) -> int:
        """Return the number of available devices.

        Returns
        -------
        int
            Devices whose random state can be saved and restored.
        """
        ...

    def set_rng_state(self, state: torch.Tensor, device: int) -> None:
        """Set one device's generator state.

        Parameters
        ----------
        state : torch.Tensor
            State produced by a generator for this backend.
        device : int
            Index of the device whose state is set.
        """
        ...


@contextmanager
def _seeded_rng(seed: int) -> Iterator[None]:
    """Isolate CPU and available accelerator random streams.

    Local generators provide seeded states for the global generators used
    by distribution samplers and fitting callbacks. Every modified state
    is restored on exit, including when a callback raises.

    Parameters
    ----------
    seed : int
        Seed used independently on CPU and each available device.

    Yields
    ------
    None
        A scope in which global draws use the explicit seed.

    Raises
    ------
    RuntimeError
        If an available backend cannot construct or restore a generator.
    """
    backend_names = dict.fromkeys(
        ("cuda", "mps", "xpu", "mtia", torch._C._get_privateuse1_backend_name())
    )
    with ExitStack() as stack:
        stack.enter_context(torch.random.fork_rng(devices=[]))
        backends: list[tuple[str, _RngBackend, range]] = []
        for name in backend_names:
            module = getattr(torch, name, None)
            if module is None:
                continue
            backend = cast(_RngBackend, module)
            devices = range(backend.device_count())
            if not devices:
                continue
            stack.enter_context(
                torch.random.fork_rng(devices=devices, device_type=name)
            )
            backends.append((name, backend, devices))
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(seed).get_state())
        for name, backend, devices in backends:
            for index in devices:
                generator = torch.Generator(device=torch.device(name, index))
                backend.set_rng_state(generator.manual_seed(seed).get_state(), index)
        yield


class SBCResult(ValueModel):
    """The rank statistics of a simulation-based calibration run.

    Attributes
    ----------
    ranks : torch.Tensor
        Int64, shape ``(num_simulations, num_statistics)``: the rank of
        each true quantity among ``num_draws`` posterior draws, in
        ``{0, ..., num_draws}``, with ties broken uniformly at random.
    num_draws : int
        ``L``, the posterior draws each rank is taken against, after
        thinning and equalizing across simulations.
    statistic_names : tuple[str, ...]
        The name of each column of ``ranks``: ``"site"`` for a scalar
        site, ``"site[i,j]"`` for one coordinate of an array site,
        and likewise for a named test quantity.
    truths : torch.Tensor
        Float64, shape ``(num_simulations, num_statistics)``: the value
        of each quantity at the parameters drawn from the prior.
    thinning : torch.Tensor
        Int64, shape ``(num_simulations,)``: the interval each
        simulation's chains were thinned by.
    retained : torch.Tensor
        Int64, shape ``(num_simulations,)``: the draws each simulation
        kept after thinning, before all are cut to ``num_draws``.
    seed : int
        The seed that reproduces the run.
    """

    ranks: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    num_draws: int
    statistic_names: tuple[str, ...]
    truths: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    thinning: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    retained: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    seed: int


class SBCCoverage(ValueModel):
    """Coverage of the central rank band, per statistic.

    A rank in the band ``[lower_rank, upper_rank]`` means the true
    value lies inside the central posterior interval spanned by the
    corresponding order statistics. Under calibration the probability
    of that event is exactly ``null_mass``, which differs from the
    requested ``mass`` because ranks are discrete.

    Attributes
    ----------
    mass : float
        The requested central mass.
    null_mass : float
        The band's exact probability under uniform ranks,
        ``(upper_rank - lower_rank + 1) / (num_draws + 1)``.
    lower_rank : int
        The smallest rank in the band.
    upper_rank : int
        The largest rank in the band.
    family_error : float
        The error rate shared by all statistics' intervals; each
        interval has confidence ``1 - family_error / num_statistics``.
    coverage : torch.Tensor
        Float64, shape ``(num_statistics,)``: the fraction of
        simulations whose rank lies in the band.
    lower : torch.Tensor
        Float64, shape ``(num_statistics,)``: the Clopper-Pearson lower
        bound of each coverage.
    upper : torch.Tensor
        Float64, shape ``(num_statistics,)``: the Clopper-Pearson upper
        bound of each coverage.
    consistent : torch.Tensor
        Bool, shape ``(num_statistics,)``: whether ``null_mass`` lies
        inside each interval.
    """

    mass: float
    null_mass: float
    lower_rank: int
    upper_rank: int
    family_error: float
    coverage: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    lower: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    upper: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)
    consistent: torch.Tensor = dx.field(opaque=True, extras=OPAQUE_BY_IDENTITY)


def sbc_ranks(
    truth: torch.Tensor,
    draws: torch.Tensor,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Rank true values among posterior draws, breaking ties at random.

    The rank of a true value ``t`` among draws ``d_1, ..., d_L`` is
    ``#{d_l < t} + U``, with ``U`` uniform on ``{0, ..., #{d_l = t}}``.
    With continuous draws ties have probability zero and ``U = 0``;
    with discrete or mixed quantities the random tie-break is what
    makes the rank uniform on ``{0, ..., L}`` under calibration.

    Parameters
    ----------
    truth : torch.Tensor
        True values, any shape ``S``.
    draws : torch.Tensor
        Posterior draws, shape ``(L, *S)``.
    generator : torch.Generator
        Source of the tie-breaking draws.

    Returns
    -------
    torch.Tensor
        Int64 ranks, shape ``S``, in ``{0, ..., L}``.

    Raises
    ------
    ValueError
        If ``draws`` is not shaped ``(L, *truth.shape)`` with ``L >= 1``,
        or either tensor holds a non-finite value.
    """
    if draws.dim() != truth.dim() + 1 or tuple(draws.shape[1:]) != tuple(truth.shape):
        raise ValueError(
            f"sbc_ranks: draws must have shape (L, *truth.shape) = "
            f"(L, *{tuple(truth.shape)}); got {tuple(draws.shape)}"
        )
    if draws.shape[0] < 1:
        raise ValueError("sbc_ranks: need at least one posterior draw")
    if not bool(torch.isfinite(truth).all()) or not bool(torch.isfinite(draws).all()):
        raise ValueError("sbc_ranks: truth and draws must be finite")
    below = (draws < truth).sum(dim=0)
    ties = (draws == truth).sum(dim=0)
    uniform = torch.rand(
        truth.shape, generator=generator, dtype=torch.float64, device="cpu"
    ).to(device=truth.device)
    tie_share = torch.floor(uniform * (ties + 1).to(dtype=torch.float64))
    return (below + tie_share.to(dtype=below.dtype)).to(dtype=torch.int64)


def _check_ranks(ranks: torch.Tensor, num_draws: int, consumer: str) -> torch.Tensor:
    """Validate ranks and lay them out ``(num_simulations, num_statistics)``.

    Parameters
    ----------
    ranks : torch.Tensor
        Shape ``(num_simulations,)`` or ``(num_simulations,
        num_statistics)``.
    num_draws : int
        ``L``; ranks lie in ``{0, ..., L}``.
    consumer : str
        The function name the error reports.

    Returns
    -------
    torch.Tensor
        Int64 CPU ranks of shape ``(num_simulations, num_statistics)``.

    Raises
    ------
    ValueError
        If the layout, dtype, or range is wrong, or ``num_draws < 1``.
    """
    if num_draws < 1:
        raise ValueError(f"{consumer}: num_draws must be >= 1, got {num_draws}")
    if ranks.dim() not in (1, 2) or ranks.shape[0] < 1:
        raise ValueError(
            f"{consumer}: ranks must be shaped (num_simulations,) or "
            f"(num_simulations, num_statistics); got {tuple(ranks.shape)}"
        )
    if ranks.dtype.is_floating_point or ranks.is_complex() or ranks.dtype == torch.bool:
        raise ValueError(f"{consumer}: ranks must be integers, got {ranks.dtype}")
    table = ranks.detach().to(device="cpu", dtype=torch.int64)
    if table.dim() == 1:
        table = table.unsqueeze(1)
    if bool((table < 0).any()) or bool((table > num_draws).any()):
        raise ValueError(f"{consumer}: ranks must lie in {{0, ..., {num_draws}}}")
    return table


def _binomial_tails(
    num_simulations: int, num_draws: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Binomial tail tables of the rank ECDF under uniform ranks.

    Under uniform ranks on ``{0, ..., L}``, the number of ``N`` ranks
    at or below ``k`` is Binomial(N, (k + 1) / (L + 1)).

    Parameters
    ----------
    num_simulations : int
        ``N``.
    num_draws : int
        ``L``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Float64 tables of shape ``(L, N + 1)``: ``P(count <= c)`` and
        ``P(count >= c)`` at threshold ``k`` (row) and count ``c``
        (column), each accumulated from its own tail for accuracy.
    """
    n = num_simulations
    probability = (
        torch.arange(1, num_draws + 1, dtype=torch.float64) / (num_draws + 1)
    ).unsqueeze(1)
    counts = torch.arange(n + 1, dtype=torch.float64).unsqueeze(0)
    log_choose = (
        math.lgamma(n + 1) - torch.lgamma(counts + 1.0) - torch.lgamma(n - counts + 1.0)
    )
    log_pmf = (
        log_choose
        + counts * torch.log(probability)
        + (n - counts) * torch.log1p(-probability)
    )
    pmf = torch.exp(log_pmf)
    lower = torch.cumsum(pmf, dim=1)
    upper = torch.flip(torch.cumsum(torch.flip(pmf, dims=(1,)), dim=1), dims=(1,))
    return lower, upper


def _ecdf_statistic(
    ranks: torch.Tensor,
    num_draws: int,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    """The simultaneous ECDF statistic of each column of ranks.

    For every threshold ``k`` in ``{0, ..., L - 1}`` the count of ranks
    at or below ``k`` gets its two-sided binomial tail probability
    ``min(1, 2 min(P(count <= c), P(count >= c)))``; the statistic is
    the smallest over thresholds. Small values mean the ECDF leaves the
    uniform reference somewhere.

    Parameters
    ----------
    ranks : torch.Tensor
        Int64, shape ``(N, M)``: ``M`` independent sets of ``N`` ranks.
    num_draws : int
        ``L``.
    lower : torch.Tensor
        ``P(count <= c)`` table from `_binomial_tails`.
    upper : torch.Tensor
        ``P(count >= c)`` table from `_binomial_tails`.

    Returns
    -------
    torch.Tensor
        Float64, shape ``(M,)``.
    """
    num_columns = ranks.shape[1]
    histogram = torch.zeros(num_columns, num_draws + 1, dtype=torch.int64)
    histogram.scatter_add_(1, ranks.T, torch.ones_like(ranks.T))
    at_or_below = torch.cumsum(histogram, dim=1)[:, :num_draws]
    thresholds = torch.arange(num_draws).unsqueeze(0)
    tail = torch.minimum(lower[thresholds, at_or_below], upper[thresholds, at_or_below])
    return torch.clamp(2.0 * tail, max=1.0).min(dim=1).values


def sbc_uniformity(
    ranks: torch.Tensor,
    num_draws: int,
    *,
    num_reference: int = 10_000,
    seed: int = 0,
) -> torch.Tensor:
    """Monte Carlo p-values for the uniformity of SBC ranks.

    The test statistic is the smallest two-sided binomial tail
    probability of the rank ECDF over every threshold, the quantity
    behind the simultaneous ECDF bands of Säilynoja, Bürkner, and
    Vehtari (2022). Its null distribution is simulated from
    ``num_reference`` sets of ranks drawn exactly from the discrete
    uniform distribution on ``{0, ..., L}``, so the test has its
    nominal size at any ``L`` and any number of simulations, where a
    continuous Kolmogorov-Smirnov test or an unequal-bin histogram
    would not. The p-value is ``(1 + #{reference <= observed}) /
    (num_reference + 1)``, a valid p-value for a Monte Carlo test.

    Each column is tested separately; with several statistics, adjust
    for multiplicity (for instance by comparing the smallest p-value
    with ``alpha / num_statistics``).

    Parameters
    ----------
    ranks : torch.Tensor
        Integer ranks, shape ``(num_simulations,)`` or
        ``(num_simulations, num_statistics)``, in ``{0, ..., num_draws}``.
    num_draws : int
        ``L``, the number of posterior draws each rank is taken
        against.
    num_reference : int
        Monte Carlo draws of the null distribution.
    seed : int
        Seed of the reference draws, so the p-values are reproducible.

    Returns
    -------
    torch.Tensor
        Float64 p-values, shape ``()`` for one-dimensional ``ranks``
        and ``(num_statistics,)`` otherwise.

    Raises
    ------
    ValueError
        If ``ranks`` is not an integer tensor of the stated layout with
        values in ``{0, ..., num_draws}``, or ``num_reference < 1``.
    """
    table = _check_ranks(ranks, num_draws, "sbc_uniformity")
    if num_reference < 1:
        raise ValueError(
            f"sbc_uniformity: num_reference must be >= 1, got {num_reference}"
        )
    num_simulations = table.shape[0]
    lower, upper = _binomial_tails(num_simulations, num_draws)
    observed = _ecdf_statistic(table, num_draws, lower, upper)
    generator = torch.Generator().manual_seed(seed)
    reference_parts: list[torch.Tensor] = []
    remaining = num_reference
    while remaining > 0:
        size = min(remaining, _REFERENCE_CHUNK)
        reference_ranks = torch.randint(
            0, num_draws + 1, (num_simulations, size), generator=generator
        )
        reference_parts.append(
            _ecdf_statistic(reference_ranks, num_draws, lower, upper)
        )
        remaining -= size
    reference = torch.cat(reference_parts)
    at_most = (reference.unsqueeze(0) <= observed.unsqueeze(1)).sum(dim=1)
    p_values = (1.0 + at_most.to(dtype=torch.float64)) / (num_reference + 1.0)
    return p_values[0] if ranks.dim() == 1 else p_values


def _clopper_pearson(
    successes: int, trials: int, confidence: float
) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval.

    Parameters
    ----------
    successes : int
        Observed successes.
    trials : int
        Number of trials.
    confidence : float
        Confidence level in ``(0, 1)``.

    Returns
    -------
    tuple[float, float]
        The lower and upper bounds.
    """
    tail = (1.0 - confidence) / 2.0
    low = 0.0 if successes == 0 else _beta_ppf(tail, successes, trials - successes + 1)
    high = (
        1.0
        if successes == trials
        else _beta_ppf(1.0 - tail, successes + 1, trials - successes)
    )
    return low, high


def sbc_coverage(
    ranks: torch.Tensor,
    num_draws: int,
    *,
    mass: float = 0.9,
    family_error: float = 0.05,
) -> SBCCoverage:
    """Coverage of the central rank band against its exact null mass.

    The band holds the ranks ``lower_rank, ..., L - lower_rank`` with
    ``lower_rank = floor((1 - mass) / 2 * (L + 1))``, the narrowest
    symmetric band whose null mass is at least ``mass``. With ``L``
    finite, that exact mass, ``null_mass``, is the reference the
    observed coverage is judged against, not ``mass`` itself. Each
    statistic's coverage gets a Clopper-Pearson interval (Clopper and
    Pearson, 1934, `doi:10.1093/biomet/26.4.404
    <https://doi.org/10.1093/biomet/26.4.404>`_) at confidence
    ``1 - family_error / num_statistics`` (a Bonferroni adjustment),
    so that under calibration every interval holds ``null_mass``
    with probability at least ``1 - family_error``.

    Parameters
    ----------
    ranks : torch.Tensor
        Integer ranks, shape ``(num_simulations,)`` or
        ``(num_simulations, num_statistics)``, in ``{0, ..., num_draws}``.
    num_draws : int
        ``L``.
    mass : float
        Requested central mass, in ``(0, 1)``.
    family_error : float
        Error rate shared by all statistics' intervals, in ``(0, 1)``.

    Returns
    -------
    SBCCoverage
        The band, its null mass, and each statistic's coverage and
        interval.

    Raises
    ------
    ValueError
        If ``ranks`` is malformed, or ``mass`` or ``family_error`` lies
        outside ``(0, 1)``.
    """
    table = _check_ranks(ranks, num_draws, "sbc_coverage")
    if not 0.0 < mass < 1.0:
        raise ValueError(f"sbc_coverage: mass must lie in (0, 1), got {mass}")
    if not 0.0 < family_error < 1.0:
        raise ValueError(
            f"sbc_coverage: family_error must lie in (0, 1), got {family_error}"
        )
    # The band edge is computed in exact rational arithmetic on the
    # shortest decimal spelling of ``mass``, so that a mass such as 0.8
    # with L = 9 yields the band 1..8 rather than losing a rank to
    # binary rounding.
    tail_mass = (1 - Fraction(repr(float(mass)))) / 2
    lower_rank = math.floor(tail_mass * (num_draws + 1))
    upper_rank = num_draws - lower_rank
    null_mass = (upper_rank - lower_rank + 1) / (num_draws + 1)
    trials, num_statistics = table.shape
    inside = ((table >= lower_rank) & (table <= upper_rank)).sum(dim=0).tolist()
    confidence = 1.0 - family_error / num_statistics
    bounds = [_clopper_pearson(int(k), trials, confidence) for k in inside]
    low = torch.tensor([b[0] for b in bounds], dtype=torch.float64)
    high = torch.tensor([b[1] for b in bounds], dtype=torch.float64)
    return SBCCoverage(
        mass=float(mass),
        null_mass=null_mass,
        lower_rank=lower_rank,
        upper_rank=upper_rank,
        family_error=float(family_error),
        coverage=torch.tensor(inside, dtype=torch.float64) / trials,
        lower=low,
        upper=high,
        consistent=(low <= null_mass) & (null_mass <= high),
    )


def _coordinate_names(name: str, shape: tuple[int, ...]) -> list[str]:
    """Name every coordinate of an array quantity.

    Parameters
    ----------
    name : str
        The quantity's name.
    shape : tuple[int, ...]
        Its shape.

    Returns
    -------
    list[str]
        ``[name]`` for a scalar, ``"name[i,j]"`` per coordinate
        otherwise, in row-major order.
    """
    if not shape:
        return [name]
    names: list[str] = []
    for flat in range(math.prod(shape)):
        index: list[int] = []
        rest = flat
        for extent in reversed(shape):
            index.append(rest % extent)
            rest //= extent
        names.append(f"{name}[{','.join(str(i) for i in reversed(index))}]")
    return names


def _site_value(recorded: Trace, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    """One run's value of a latent site, at its constrained shape.

    Parameters
    ----------
    recorded : Trace
        A one-run trace.
    name : str
        The latent site.
    shape : tuple[int, ...]
        The site's `LatentSite.constrained_shape`.

    Returns
    -------
    torch.Tensor
        The recorded value, reshaped to ``shape``.

    Raises
    ------
    ValueError
        If the site is not recorded under its own name (a joint draw of
        several variables) or holds a different number of elements.
    """
    record = recorded.sites.get(name)
    if record is None:
        raise ValueError(
            f"sbc: latent site {name!r} is not recorded under its own name "
            "(a joint draw of several variables); give statistics= that "
            "read the posterior draws directly"
        )
    value = record.value.detach()
    if value.numel() != math.prod(shape):
        raise ValueError(
            f"sbc: latent site {name!r} recorded {tuple(value.shape)}, which "
            f"does not hold one draw of shape {shape}"
        )
    return value.reshape(shape)


def _posterior_draws(
    posterior: SBCPosterior,
    x: torch.Tensor,
    num_guide_draws: int,
) -> dict[str, torch.Tensor]:
    """Lay a fit's posterior out as per-site chain-by-draw tensors.

    Parameters
    ----------
    posterior : SBCPosterior
        The fit's result.
    x : torch.Tensor
        The one-row program input, for drawing from a guide.
    num_guide_draws : int
        Draws taken from a guide.

    Returns
    -------
    dict[str, torch.Tensor]
        Site name to draws, ``(num_chains, num_draws, *shape)``.

    Raises
    ------
    TypeError
        If ``posterior`` is none of the accepted forms.
    ValueError
        If a draw tensor has fewer than two axes, or the sites disagree
        on the chain and draw counts.
    """
    if isinstance(posterior, MCMCResult):
        draws = dict(posterior.samples)
    elif isinstance(posterior, Guide):
        with torch.no_grad():
            rows = [
                posterior.registry.trace_row(posterior.rsample(x), 0)
                for _ in range(num_guide_draws)
            ]
        draws = {
            name: torch.stack([row[name] for row in rows]).unsqueeze(0)
            for name in posterior.registry.names
        }
    elif isinstance(posterior, Mapping):
        draws = dict(posterior)
    else:
        raise TypeError(
            "sbc: fit must return an MCMCResult, a Guide, or a mapping of "
            f"site name to (num_chains, num_draws, ...) draws; got "
            f"{type(posterior).__name__}"
        )
    layout: tuple[int, int] | None = None
    for name, values in draws.items():
        if values.dim() < 2:
            raise ValueError(
                f"sbc: posterior draws of {name!r} must be laid out "
                f"(num_chains, num_draws, ...); got {tuple(values.shape)}"
            )
        leading = (int(values.shape[0]), int(values.shape[1]))
        if layout is None:
            layout = leading
        elif leading != layout:
            raise ValueError(
                f"sbc: posterior sites disagree on (num_chains, num_draws): "
                f"{name!r} has {leading}, another site has {layout}"
            )
    return {name: values.detach() for name, values in draws.items()}


def _evaluate_statistics(
    statistics: Mapping[str, SBCStatistic] | None,
    registry: LatentRegistry,
    parameters: Mapping[str, torch.Tensor],
    data: Mapping[str, torch.Tensor],
) -> tuple[list[str], torch.Tensor]:
    """Evaluate the test quantities on a batch of parameter draws.

    Parameters
    ----------
    statistics : Mapping[str, SBCStatistic] or None
        Named test quantities, or None for every latent coordinate.
    registry : LatentRegistry
        The latent sites.
    parameters : Mapping[str, torch.Tensor]
        Site name to draws with a leading draw axis,
        ``(n, *constrained_shape)``.
    data : Mapping[str, torch.Tensor]
        The simulated observations and host data.

    Returns
    -------
    tuple[list[str], torch.Tensor]
        The coordinate names and a float64 ``(n, num_statistics)``
        tensor of their values.

    Raises
    ------
    KeyError
        If a latent site has no posterior draws.
    ValueError
        If a test quantity returns a tensor without the leading draw
        axis.
    """
    names: list[str] = []
    columns: list[torch.Tensor] = []
    if statistics is None:
        for site in registry.sites.values():
            if site.name not in parameters:
                raise KeyError(f"sbc: the posterior has no draws of {site.name!r}")
            values = parameters[site.name]
            count = values.shape[0]
            names.extend(_coordinate_names(site.name, site.constrained_shape))
            columns.append(values.reshape(count, -1))
    else:
        count = next(iter(parameters.values())).shape[0]
        for name, statistic in statistics.items():
            values = statistic(parameters, data).detach()
            if values.dim() < 1 or values.shape[0] != count:
                raise ValueError(
                    f"sbc: statistic {name!r} must return one row per draw, "
                    f"shape ({count}, ...); got {tuple(values.shape)}"
                )
            names.extend(_coordinate_names(name, tuple(values.shape[1:])))
            columns.append(values.reshape(count, -1))
    stacked = torch.cat([c.to(device="cpu", dtype=torch.float64) for c in columns], 1)
    return names, stacked


def _thinning_interval(draws: torch.Tensor, thin: SBCThinning) -> int:
    """The thinning interval for one simulation's statistic draws.

    Parameters
    ----------
    draws : torch.Tensor
        Shape ``(num_chains, num_draws, num_statistics)``.
    thin : SBCThinning
        ``"ess"`` or a fixed interval.

    Returns
    -------
    int
        The interval. Under ``"ess"`` it is
        ``ceil(num_chains * num_draws / ess)`` with ``ess`` the smallest
        bulk or tail effective sample size over the statistics whose
        draws vary; statistics constant across the posterior draws
        carry no dependence to thin out, and if all are constant the
        interval is 1.

    Raises
    ------
    ValueError
        If ``thin`` is not ``"ess"`` or a positive integer, or ``"ess"``
        is asked of fewer than four draws per chain.
    """
    if thin == "ess":
        if draws.shape[1] < 4:
            raise ValueError(
                'sbc: thin="ess" needs at least four posterior draws per chain; '
                f"got {draws.shape[1]}"
            )
        effective = torch.minimum(ess(draws, method="bulk"), ess(draws, method="tail"))
        varying = effective[~torch.isnan(effective)]
        if varying.numel() == 0:
            return 1
        total = draws.shape[0] * draws.shape[1]
        return max(1, math.ceil(total / float(varying.min())))
    if isinstance(thin, bool) or not isinstance(thin, int) or thin < 1:
        raise ValueError(f'sbc: thin must be "ess" or a positive integer, got {thin!r}')
    return thin


def sbc(
    model: MonadicProgram,
    x: torch.Tensor,
    fit: Callable[[dict[str, torch.Tensor], int], SBCPosterior],
    *,
    observed_names: Iterable[str],
    num_simulations: int,
    seed: int,
    data: Mapping[str, torch.Tensor] | None = None,
    statistics: Mapping[str, SBCStatistic] | None = None,
    thin: SBCThinning = "ess",
    num_guide_draws: int = 1000,
) -> SBCResult:
    """Run simulation-based calibration of an inference procedure.

    Each of ``num_simulations`` simulations

    1. runs ``model`` forward at the one-row input ``x`` and the host
       ``data``, drawing every latent from its prior and every site in
       ``observed_names`` from the model given those latents;
    2. calls ``fit(observations, fit_seed)`` with the simulated
       observations merged with ``data``, and reads its posterior
       draws (see `SBCPosterior`);
    3. evaluates the test quantities at the prior draw and at every
       posterior draw, thins the posterior draws (``thin``), and keeps
       them.

    The posterior draws of all simulations are then cut to the
    smallest retained count ``L`` (taking evenly spaced draws), and
    each true quantity is ranked among its ``L`` draws with
    `sbc_ranks`. Following Talts et al. (2018), ``thin="ess"`` keeps
    about one draw per effective sample, since autocorrelated draws
    make the ranks of even an exact sampler non-uniform.

    The procedure is reproducible from ``seed``: it derives three
    seeds per simulation, for the prior simulation, the fit, and the
    tie-breaks; the prior simulation and the fit run with Torch's
    CPU and available accelerator generators forked and initialized from
    locally seeded generators. Neither scope depends on or changes the
    caller's random state, and a fit that draws from global generators is
    reproducible too. States are restored even if the fit raises.

    Parameters
    ----------
    model : MonadicProgram
        The program whose inference is checked.
    x : torch.Tensor
        A one-row program input: one run, as MCMC and the guides take.
    fit : Callable[[dict[str, torch.Tensor], int], SBCPosterior]
        The inference procedure: ``fit(observations, seed)``, where
        ``observations`` holds the simulated observed sites and the
        host data, and ``seed`` is an integer the procedure may pass
        to a seeded sampler (for instance ``MCMC(..., seed=seed)``).
    observed_names : Iterable[str]
        The sites treated as data.
    num_simulations : int
        Number of simulations.
    seed : int
        Seed of the whole procedure.
    data : Mapping[str, torch.Tensor] or None
        Host data the program reads by name (covariates), held fixed.
    statistics : Mapping[str, SBCStatistic] or None
        Named test quantities; None ranks every coordinate of every
        latent site. Quantities that read the data, such as a
        log-likelihood, can detect miscalibration that the parameters
        alone miss (Modrák et al., 2025,
        `doi:10.1214/23-BA1404 <https://doi.org/10.1214/23-BA1404>`_).
    thin : SBCThinning
        ``"ess"`` (the default) or a fixed interval; use ``1`` for a
        fit whose draws are independent.
    num_guide_draws : int
        Draws taken from a fit that returns a `Guide`.

    Returns
    -------
    SBCResult
        The ranks and the bookkeeping that reproduces them.

    Raises
    ------
    ValueError
        If ``x`` has more than one row, ``num_simulations`` or
        ``num_guide_draws`` is less than one, an observed name is not a
        site of the model, ``thin`` is invalid, a simulation's latent
        sites or statistics differ from the first simulation's, or a
        true value or posterior draw is non-finite.
    TypeError
        If ``fit`` returns none of the accepted posterior forms.
    """
    _require_one_run(x, "sbc")
    if num_simulations < 1:
        raise ValueError(f"sbc: num_simulations must be >= 1, got {num_simulations}")
    if num_guide_draws < 1:
        raise ValueError(f"sbc: num_guide_draws must be >= 1, got {num_guide_draws}")
    observed = frozenset(observed_names)
    host = dict(data or {})
    master = torch.Generator().manual_seed(seed)
    seeds: list[list[int]] = torch.randint(
        0, _INT64_MAX, (num_simulations, 3), generator=master
    ).tolist()

    statistic_names: list[str] | None = None
    truths: list[torch.Tensor] = []
    kept: list[torch.Tensor] = []
    intervals: list[int] = []
    for simulation_seed, fit_seed, _ in seeds:
        with _seeded_rng(simulation_seed):
            with torch.no_grad():
                simulated = trace(model, x, observations=dict(host))
                missing = sorted(observed - simulated.sites.keys())
                if missing:
                    raise ValueError(
                        f"sbc: observed names {missing} are not sample sites "
                        "of the model"
                    )
                observations = {
                    name: simulated.sites[name].value.detach() for name in observed
                }
                fit_data = {**host, **observations}
                registry = LatentRegistry.from_trace(
                    model, trace(model, x, observations=dict(fit_data))
                )
        truth = {
            site.name: _site_value(simulated, site.name, site.constrained_shape)
            for site in registry.sites.values()
        }
        with _seeded_rng(fit_seed):
            posterior = _posterior_draws(
                fit(dict(fit_data), fit_seed), x, num_guide_draws
            )
        names, truth_values = _evaluate_statistics(
            statistics,
            registry,
            {name: value.unsqueeze(0) for name, value in truth.items()},
            fit_data,
        )
        num_chains, num_draws = next(iter(posterior.values())).shape[:2]
        flat_posterior = {
            name: values.reshape(num_chains * num_draws, *values.shape[2:])
            for name, values in posterior.items()
        }
        _, draw_values = _evaluate_statistics(
            statistics, registry, flat_posterior, fit_data
        )
        if statistic_names is None:
            statistic_names = names
        elif names != statistic_names:
            raise ValueError(
                "sbc: the simulations disagree on the latent sites or statistics: "
                f"{names} against {statistic_names}"
            )
        if not bool(torch.isfinite(truth_values).all()):
            raise ValueError("sbc: a prior draw of a statistic is not finite")
        if not bool(torch.isfinite(draw_values).all()):
            raise ValueError("sbc: the fit returned a non-finite posterior draw")
        chains = draw_values.reshape(num_chains, num_draws, -1)
        interval = _thinning_interval(chains, thin)
        intervals.append(interval)
        truths.append(truth_values[0])
        kept.append(chains[:, ::interval].reshape(-1, chains.shape[2]))

    retained = [int(draws.shape[0]) for draws in kept]
    common = min(retained)
    ranks: list[torch.Tensor] = []
    for (_, _, tie_seed), truth_row, draws in zip(seeds, truths, kept, strict=True):
        positions = (torch.arange(common) * draws.shape[0]) // common
        ranks.append(
            sbc_ranks(
                truth_row,
                draws[positions],
                generator=torch.Generator().manual_seed(tie_seed),
            )
        )
    return SBCResult(
        ranks=torch.stack(ranks),
        num_draws=common,
        statistic_names=tuple(statistic_names or ()),
        truths=torch.stack(truths),
        thinning=torch.tensor(intervals, dtype=torch.int64),
        retained=torch.tensor(retained, dtype=torch.int64),
        seed=seed,
    )


__all__ = [
    "SBCStatistic",
    "SBCPosterior",
    "SBCThinning",
    "SBCResult",
    "SBCCoverage",
    "sbc",
    "sbc_ranks",
    "sbc_uniformity",
    "sbc_coverage",
]
