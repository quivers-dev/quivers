"""Families built on a restriction of a base measure draw from the restriction.

`Restrict` draws from its base conditioned on the interval: by inverting
the base's CDF when it has one, and otherwise by redrawing each entry
outside the interval. Every half family, truncated normal, and
zero-truncated count is checked plated and unplated: each draw lies in
the support, and the draws' quantiles match an independent reference
(a torch distribution of the same law, or the exact quantile function)
to within a tolerance set from the Monte Carlo error of a quantile
estimate, so a sampler that reflects or clips the wrong draws fails.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.distributions as D

from quivers.continuous.measure import Restrict
from quivers.dsl import loads
from quivers.inference import trace

#: Draws per quantile check.
N = 20_000

#: The probability levels whose quantiles are compared.
LEVELS = (0.25, 0.5, 0.9)


def _quantile_tolerance(reference: torch.Tensor, level: float) -> float:
    """Five standard errors of a sample quantile at ``level``.

    The standard error of the ``level`` sample quantile of ``N`` draws
    is ``sqrt(level (1 - level) / N) / f(q)`` for the density ``f`` at
    the quantile ``q``; the density is estimated from a large reference
    sample by the share of it within a small window around ``q``.

    Parameters
    ----------
    reference : torch.Tensor
        A large sample of the reference law.
    level : float
        The probability level.

    Returns
    -------
    float
        The tolerance.
    """
    width = float(torch.quantile(reference, min(level + 0.02, 0.999))) - float(
        torch.quantile(reference, max(level - 0.02, 0.001))
    )
    density = 0.04 / width
    return 5.0 * math.sqrt(level * (1.0 - level) / N) / density + 1e-4


def _assert_matches(draws: torch.Tensor, reference: torch.Tensor) -> None:
    for level in LEVELS:
        got = float(torch.quantile(draws.reshape(-1), level))
        want = float(torch.quantile(reference.reshape(-1), level))
        tolerance = _quantile_tolerance(reference.reshape(-1), level)
        assert abs(got - want) < tolerance, (level, got, want, tolerance)


#: Each restricted law: (the restriction, an independent reference
#: sampler of the same law, the support's lower and upper end).
CASES = {
    "half-normal": (
        lambda: Restrict(D.Normal(0.0, 1.5), low=0.0),
        lambda n: D.HalfNormal(1.5).sample((n,)),
        0.0,
        math.inf,
    ),
    "half-cauchy": (
        lambda: Restrict(D.Cauchy(0.0, 2.5), low=0.0),
        lambda n: D.HalfCauchy(2.5).sample((n,)),
        0.0,
        math.inf,
    ),
    "half-student-t": (
        lambda: Restrict(D.StudentT(4.0, 0.0, 1.5), low=0.0),
        lambda n: D.StudentT(4.0, 0.0, 1.5).sample((n,)).abs(),
        0.0,
        math.inf,
    ),
    "half-laplace": (
        lambda: Restrict(D.Laplace(0.0, 1.0), low=0.0),
        lambda n: D.Exponential(1.0).sample((n,)),
        0.0,
        math.inf,
    ),
    "truncated-normal-off-centre": (
        lambda: Restrict(D.Normal(1.0, 2.0), low=0.0, high=1.0),
        lambda n: _rejection_reference(D.Normal(1.0, 2.0), 0.0, 1.0, n),
        0.0,
        1.0,
    ),
    "normal-above-its-mean": (
        lambda: Restrict(D.Normal(-1.0, 1.0), low=0.5),
        lambda n: _rejection_reference(D.Normal(-1.0, 1.0), 0.5, math.inf, n),
        0.5,
        math.inf,
    ),
    "normal-far-tail": (
        lambda: Restrict(D.Normal(0.0, 1.0), low=4.0, high=5.0),
        lambda n: _tail_reference(n),
        4.0,
        5.0,
    ),
}


def _rejection_reference(
    base: D.Distribution, low: float, high: float, n: int
) -> torch.Tensor:
    kept: list[torch.Tensor] = []
    total = 0
    while total < n:
        draws = base.sample((4 * n,))
        inside = draws[(draws >= low) & (draws <= high)]
        kept.append(inside)
        total += inside.numel()
    return torch.cat(kept)[:n]


def _tail_reference(n: int) -> torch.Tensor:
    # The standard normal above 4 is the exact exponential-tilt law
    # sampled by the Robert (1995) rejection scheme.
    kept: list[torch.Tensor] = []
    total = 0
    rate = (4.0 + math.sqrt(4.0**2 + 4.0)) / 2.0
    while total < n:
        z = 4.0 + D.Exponential(rate).sample((4 * n,))
        accept = torch.rand(4 * n) <= torch.exp(-((z - rate) ** 2) / 2.0)
        inside = z[accept & (z <= 5.0)]
        kept.append(inside)
        total += inside.numel()
    return torch.cat(kept)[:n]


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_restriction_draws_from_its_conditioned_base(case: str) -> None:
    restricted, reference, low, high = CASES[case]
    torch.manual_seed(0)
    draws = restricted().sample(torch.Size((N,)))
    assert bool((draws >= low).all()) and bool((draws <= high).all())
    _assert_matches(draws, reference(N))


def test_a_zero_truncated_count_draws_its_conditioned_mass() -> None:
    torch.manual_seed(0)
    draws = Restrict(D.Poisson(0.7), low=1.0).sample(torch.Size((N,)))
    assert bool((draws >= 1).all())
    probabilities = D.Poisson(0.7).log_prob(torch.arange(1.0, 6.0)).exp()
    probabilities = probabilities / (1.0 - math.exp(-0.7))
    for count, probability in zip(range(1, 6), probabilities, strict=True):
        share = float((draws == count).float().mean())
        tolerance = 5.0 * math.sqrt(float(probability) * (1 - probability) / N)
        assert abs(share - float(probability)) < tolerance, (count, share)


def test_a_restriction_too_small_to_reject_into_raises() -> None:
    restricted = Restrict(D.Poisson(0.01), low=40.0)
    with pytest.raises(RuntimeError, match="stayed outside"):
        restricted.sample(torch.Size((4,)))


#: Each restricted family as a program site, its support, and its
#: reference law.
FAMILIES = {
    "HalfNormal(scale=1.5)": (0.0, math.inf, lambda n: D.HalfNormal(1.5).sample((n,))),
    "HalfCauchy(scale=2.5)": (0.0, math.inf, lambda n: D.HalfCauchy(2.5).sample((n,))),
    "HalfStudentT(df=4.0, scale=1.5)": (
        0.0,
        math.inf,
        lambda n: D.StudentT(4.0, 0.0, 1.5).sample((n,)).abs(),
    ),
    "TruncatedNormal(mu=1.0, sigma=2.0, low=0.0, high=1.0)": (
        0.0,
        1.0,
        lambda n: _rejection_reference(D.Normal(1.0, 2.0), 0.0, 1.0, n),
    ),
}


def _plated(family: str) -> str:
    return (
        f"object A : FinSet {N}\nobject Val : Real 1\nprogram p : A -> Val\n"
        f"    sample w : A <- {family}\n    return w\nexport p\n"
    )


def _unplated(family: str) -> str:
    return (
        "object A : FinSet 3\nobject Val : Real 1\nprogram p : A -> Val\n"
        f"    sample w <- {family}\n    return w\nexport p\n"
    )


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_plated_restricted_site_draws_its_law(family: str) -> None:
    low, high, reference = FAMILIES[family]
    torch.manual_seed(0)
    w = trace(loads(_plated(family)).morphism, torch.zeros(1, 1)).sites["w"].value
    assert w.numel() == N
    assert bool((w >= low).all()) and bool((w <= high).all())
    _assert_matches(w, reference(N))


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_an_unplated_restricted_site_draws_its_law(family: str) -> None:
    low, high, reference = FAMILIES[family]
    model = loads(_unplated(family)).morphism
    torch.manual_seed(0)
    draws = torch.stack(
        [
            trace(model, torch.zeros(1, 1)).sites["w"].value.reshape(())
            for _ in range(2000)
        ]
    )
    assert bool((draws >= low).all()) and bool((draws <= high).all())
    # 2000 draws: the quantile tolerance is recomputed for that size.
    for level in LEVELS:
        ref = reference(N)
        got = float(torch.quantile(draws, level))
        want = float(torch.quantile(ref, level))
        tolerance = _quantile_tolerance(ref, level) * math.sqrt(N / 2000)
        assert abs(got - want) < tolerance, (level, got, want)


def test_a_plated_half_normal_with_a_sampled_scale_stays_positive() -> None:
    source = (
        "object A : FinSet 500\nobject Val : Real 1\nprogram p : A -> Val\n"
        "    sample s <- HalfNormal(scale=1.0)\n"
        "    sample w : A <- HalfNormal(scale=s)\n    return w\nexport p\n"
    )
    torch.manual_seed(0)
    w = trace(loads(source).morphism, torch.zeros(1, 1)).sites["w"].value
    assert bool((w >= 0).all())
