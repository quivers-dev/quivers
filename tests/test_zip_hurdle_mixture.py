"""Tests for the zero-inflated / hurdle / mixture distributions and
their DSL inline observe surface.

Three slices, mirroring `tests/test_ordered_logistic.py`:

1. The standalone
   [`ZeroInflatedPoisson`][quivers.continuous._zip_hurdle.ZeroInflatedPoisson],
   [`HurdlePoisson`][quivers.continuous._zip_hurdle.HurdlePoisson],
   and [`MixtureNormal`][quivers.continuous._zip_hurdle.MixtureNormal]
   distributions: log-prob agreement with the closed-form mixture /
   hurdle / Gaussian-mixture density, batch broadcasting, support
   constraints.
2. DSL `observe ... <- Family(...)` surfaces for each.
3. The conditional [`role=kernel`] paths through the corresponding
   `Conditional*` classes are registered in
   [`_get_family_registry`][quivers.dsl.compiler._prelude._get_family_registry].
"""

from __future__ import annotations

import math

import torch

from quivers.continuous import (
    HurdlePoisson,
    MixtureNormal,
    ZeroInflatedPoisson,
    ZeroOneInflatedBeta,
)
from quivers.dsl import loads
from quivers.dsl.compiler._prelude import _get_family_registry
from quivers.inference.trace import trace


def test_zero_inflated_poisson_log_prob() -> None:
    pi = torch.tensor([0.3, 0.5, 0.1])
    rate = torch.tensor([2.0, 1.0, 5.0])
    dist = ZeroInflatedPoisson(pi, rate)
    y = torch.tensor([0, 1, 3])
    actual = dist.log_prob(y)
    poisson = torch.distributions.Poisson(rate)
    expected_zero = torch.log(pi + (1.0 - pi) * (-rate).exp())
    expected_pos = torch.log(1.0 - pi) + poisson.log_prob(y)
    expected = torch.where(y == 0, expected_zero, expected_pos)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_zero_inflated_poisson_mean() -> None:
    pi = torch.tensor([0.0, 0.5, 1.0])
    rate = torch.tensor([3.0, 3.0, 3.0])
    dist = ZeroInflatedPoisson(pi.clamp(1e-6, 1.0 - 1e-6), rate)
    torch.testing.assert_close(
        dist.mean,
        torch.tensor([3.0, 1.5, 0.0]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_hurdle_poisson_log_prob() -> None:
    pi = torch.tensor([0.2, 0.5, 0.7])
    rate = torch.tensor([1.5, 3.0, 0.8])
    dist = HurdlePoisson(pi, rate)
    y = torch.tensor([0, 2, 5])
    actual = dist.log_prob(y)
    poisson = torch.distributions.Poisson(rate)
    log_survival = (1.0 - (-rate).exp()).log()
    expected_zero = pi.log()
    expected_pos = (1.0 - pi).log() + poisson.log_prob(y) - log_survival
    expected = torch.where(y == 0, expected_zero, expected_pos)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_hurdle_poisson_zero_branch_dominates_at_high_pi() -> None:
    pi = torch.tensor([0.99])
    rate = torch.tensor([2.0])
    dist = HurdlePoisson(pi, rate)
    assert dist.log_prob(torch.tensor([0])).item() > math.log(0.98)


def test_mixture_normal_log_prob_matches_explicit_mixture() -> None:
    weights = torch.tensor([0.3, 0.4, 0.3])
    loc = torch.tensor([-2.0, 0.0, 2.0])
    scale = torch.tensor([0.5, 1.0, 0.5])
    dist = MixtureNormal(weights, loc, scale)
    y = torch.tensor([-1.0, 0.5, 2.0, -3.0])
    actual = dist.log_prob(y)
    components = torch.distributions.Normal(loc, scale)
    log_p = components.log_prob(y.unsqueeze(-1))
    expected = torch.logsumexp(torch.log(weights) + log_p, dim=-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def test_mixture_normal_sample_shape() -> None:
    weights = torch.tensor([[0.3, 0.7], [0.5, 0.5]])
    loc = torch.tensor([[-1.0, 1.0], [0.0, 2.0]])
    scale = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
    dist = MixtureNormal(weights, loc, scale)
    assert dist.batch_shape == torch.Size([2])
    samples = dist.sample(torch.Size([4]))
    assert samples.shape == torch.Size([4, 2])


def test_dsl_observe_zero_inflated_poisson() -> None:
    program = loads("""
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(loc=0.0, scale=1.0)
    observe y : Resp <- ZeroInflatedPoisson(zero_prob=pi, rate=rate)
    return y
export m
""")
    y = torch.tensor([0, 1, 2, 0]).long()
    pi = torch.tensor([0.3, 0.3, 0.3, 0.3])
    rate = torch.tensor([1.5, 1.5, 1.5, 1.5])
    tr = trace(
        program.morphism,
        torch.zeros(4, 1),
        observations={"y": y, "pi": pi, "rate": rate},
    )
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


def test_dsl_observe_hurdle_poisson() -> None:
    program = loads("""
object Resp : FinSet 4
program m : Resp -> Resp
    sample foo <- Normal(loc=0.0, scale=1.0)
    observe y : Resp <- HurdlePoisson(zero_prob=pi, rate=rate)
    return y
export m
""")
    y = torch.tensor([0, 2, 1, 0]).long()
    pi = torch.tensor([0.3, 0.3, 0.3, 0.3])
    rate = torch.tensor([1.5, 1.5, 1.5, 1.5])
    tr = trace(
        program.morphism,
        torch.zeros(4, 1),
        observations={"y": y, "pi": pi, "rate": rate},
    )
    assert tr.log_joint is not None
    assert torch.isfinite(tr.log_joint).all()


def test_conditional_classes_registered() -> None:
    """Every new family must appear in the conditional-path registry
    so `morphism f : A -> B [role=kernel] ~ Family` declarations
    compile rather than raising `undefined morphism or distribution
    family`.
    """
    registry = _get_family_registry()
    for name in (
        "Poisson",
        "NegativeBinomial",
        "Binomial",
        "OrderedLogistic",
        "ZeroInflatedPoisson",
        "HurdlePoisson",
        "MixtureNormal",
        "ZeroOneInflatedBeta",
    ):
        assert name in registry, f"missing {name!r} in family registry"


# ---------------------------------------------------------------------------
# ZeroOneInflatedBeta
# ---------------------------------------------------------------------------


def _zoib_params() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor(0.6),
        torch.tensor(4.0),
        torch.tensor(0.3),
        torch.tensor(0.25),
    )


def test_zoib_endpoint_masses_match_closed_form() -> None:
    """The two point masses are `zoi (1 - coi)` at 0 and `zoi coi` at 1."""
    mu, phi, zoi, coi = _zoib_params()
    dist = ZeroOneInflatedBeta(mu, phi, zoi, coi)
    assert math.isclose(
        float(dist.log_prob(torch.tensor(0.0))),
        math.log(0.3 * 0.75),
        rel_tol=1e-6,
    )
    assert math.isclose(
        float(dist.log_prob(torch.tensor(1.0))),
        math.log(0.3 * 0.25),
        rel_tol=1e-6,
    )


def test_zoib_interior_matches_scaled_beta() -> None:
    """Inside the open interval the density is the beta component
    scaled by the probability `1 - zoi` of not landing on an endpoint,
    under the mean-precision parameterisation."""
    mu, phi, zoi, coi = _zoib_params()
    dist = ZeroOneInflatedBeta(mu, phi, zoi, coi)
    y = torch.tensor(0.4)
    beta = torch.distributions.Beta(mu * phi, (1.0 - mu) * phi)
    expected = math.log(1.0 - float(zoi)) + float(beta.log_prob(y))
    assert math.isclose(float(dist.log_prob(y)), expected, rel_tol=1e-6)


def test_zoib_normalises_to_one() -> None:
    """The two masses plus the integral over the open interval is 1, so
    the mixture is a probability distribution rather than merely a
    positive weighting."""
    mu, phi, zoi, coi = _zoib_params()
    dist = ZeroOneInflatedBeta(mu, phi, zoi, coi)
    grid = torch.linspace(1e-6, 1.0 - 1e-6, 200_001, dtype=torch.float64)
    interior = float(torch.trapz(torch.exp(dist.log_prob(grid)), grid))
    endpoints = float(zoi) * (1.0 - float(coi)) + float(zoi) * float(coi)
    assert math.isclose(endpoints + interior, 1.0, abs_tol=1e-4)


def test_zoib_broadcasts_over_a_batch() -> None:
    """Batched parameters score a batch of observations elementwise."""
    dist = ZeroOneInflatedBeta(
        torch.tensor([0.2, 0.6, 0.8]),
        torch.tensor([3.0, 4.0, 9.0]),
        torch.tensor([0.1, 0.3, 0.5]),
        torch.tensor([0.5, 0.25, 0.75]),
    )
    lp = dist.log_prob(torch.tensor([0.0, 0.4, 1.0]))
    assert lp.shape == (3,)
    assert bool(torch.isfinite(lp).all())
