"""The hurdle Poisson retains its zero-rate limit and tiny positive rates."""

import math
import random
from decimal import Decimal, localcontext

import pytest
import torch

from quivers.continuous._zip_hurdle import HurdlePoisson
from quivers.continuous._zero_truncated_poisson import positive_poisson_mean
from quivers.dsl import loads
from quivers.qiec.distributions import RuntimeDistribution


@pytest.mark.parametrize("rate", [0.0, 1e-16, 1e-8, 1.0, 40.0])
def test_hurdle_mean_preserves_the_truncated_poisson_limit(rate: float) -> None:
    dist = HurdlePoisson(torch.tensor(0.3), torch.tensor(rate, dtype=torch.float64))
    conditional = rate / -math.expm1(-rate) if rate > 0 else 1.0
    assert dist.mean.item() == pytest.approx(0.7 * conditional, rel=1e-7)
    if rate < 1e-12:
        wrong = 0.7 * rate / 1e-12
        assert abs(dist.mean.item() - wrong) > 0.6


@pytest.mark.parametrize("zero_prob", [0.0, 0.3, 1.0])
def test_zero_rate_hurdle_is_a_bernoulli_distribution(zero_prob: float) -> None:
    dist = HurdlePoisson(torch.tensor(zero_prob), torch.tensor(0.0))
    actual = dist.log_prob(torch.tensor([0.0, 1.0, 2.0])).exp()
    torch.testing.assert_close(actual, torch.tensor([zero_prob, 1 - zero_prob, 0.0]))


@pytest.mark.parametrize("rate", [1e-16, 0.005, 0.01, 0.2, 8.0, 20.0])
def test_hurdle_density_matches_high_precision_pmf(rate: float) -> None:
    dist = HurdlePoisson(
        torch.tensor(0.3, dtype=torch.float64), torch.tensor(rate, dtype=torch.float64)
    )
    reference = RuntimeDistribution("HurdlePoisson", {"zero_prob": 0.3, "rate": rate})
    with localcontext() as context:
        context.prec = 80
        r = Decimal(str(rate))
        survival = 1 - (-r).exp()
        for k in [1, 2, 5]:
            mass = (
                Decimal("0.7")
                * r**k
                * (-r).exp()
                / (Decimal(math.factorial(k)) * survival)
            )
            expected = float(mass.ln())
            assert dist.log_prob(
                torch.tensor(float(k), dtype=torch.float64)
            ).item() == pytest.approx(expected, rel=1e-12, abs=1e-12)
            assert reference.log_prob(k) == pytest.approx(
                expected, rel=1e-12, abs=1e-12
            )


def test_hurdle_zero_rate_has_finite_limiting_gradients() -> None:
    rate = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    mean = positive_poisson_mean(rate)
    assert torch.autograd.grad(mean, rate)[0].item() == pytest.approx(0.5)
    dist = HurdlePoisson(torch.tensor(0.3, dtype=torch.float64), rate)
    score = dist.log_prob(torch.tensor(1.0))
    assert torch.autograd.grad(score, rate)[0].item() == pytest.approx(-0.5)


@pytest.mark.parametrize("rate", [0.0, 1e-16, 0.3, 4.0, 20.0])
def test_hurdle_draw_frequencies_match_independent_pmf(rate: float) -> None:
    dist = HurdlePoisson(
        torch.tensor(0.3, dtype=torch.float64), torch.tensor(rate, dtype=torch.float64)
    )
    n = 40000
    with torch.random.fork_rng():
        torch.manual_seed(119)
        draws = dist.sample(torch.Size([n]))
    assert draws.shape == (n,)
    assert (draws >= 0).all()
    for count in [0, 1, 2, 3]:
        if count == 0:
            expected = 0.3
        elif rate == 0:
            expected = 0.7 if count == 1 else 0.0
        else:
            expected = (
                0.7
                * rate**count
                * math.exp(-rate)
                / (math.factorial(count) * -math.expm1(-rate))
            )
        actual = (draws == count).double().mean().item()
        allowance = 6 * math.sqrt(expected * (1 - expected) / n) + 1 / n
        assert abs(actual - expected) <= allowance
    if rate < 1e-12:
        assert (draws <= 1).all()
        assert (draws == 1).double().mean().item() > 0.65


@pytest.mark.parametrize("rate", [0.0, 1e-16])
@pytest.mark.parametrize("literal", [False, True])
def test_compiled_hurdle_keeps_boundary_density(rate: float, literal: bool) -> None:
    argument = repr(rate) if literal else "rate"
    source = f"""
object Obs : FinSet 2
object Out : Real 1
program p : Obs -> Out
    sample a <- Normal(loc=0.0, scale=1.0)
    observe y : Obs <- HurdlePoisson(zero_prob=0.3, rate={argument})
    return a
export p
"""
    values = {
        "a": torch.tensor(0.0),
        "y": torch.tensor([0.0, 1.0]),
        "rate": torch.tensor(rate, dtype=torch.float64),
    }
    actual = loads(source).morphism.log_joint(torch.zeros(1, 1), values).item()
    expected = math.log(0.3) + math.log(0.7) - 0.5 * math.log(2 * math.pi)
    assert actual == pytest.approx(expected, abs=1e-6)


def test_reference_hurdle_samples_its_zero_rate_limit() -> None:
    rng = random.Random(119)
    dist = RuntimeDistribution("HurdlePoisson", {"zero_prob": 0.3, "rate": 0.0})
    draws = [dist.sample(rng) for _ in range(2000)]
    assert set(draws) == {0, 1}
    assert abs(draws.count(0) / 2000 - 0.3) < 6 * math.sqrt(0.21 / 2000)
    assert dist.log_prob(2) == -math.inf
