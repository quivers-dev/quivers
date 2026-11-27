"""Positive hurdle laws match independent atom and continuous-component oracles."""

import math
import random

import numpy as np
import pytest
import scipy.stats
import torch

from quivers.continuous import HurdleGamma, HurdleLogNormal
from quivers.continuous.families import (
    ConditionalHurdleGamma,
    ConditionalHurdleLogNormal,
)
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.inference.trace import trace
from quivers.qiec.distributions import ReferenceBackend


CASES = [
    (HurdleGamma, (0.3, 0.4, 1.7), scipy.stats.gamma(0.4, scale=1 / 1.7)),
    (HurdleGamma, (0.2, 3.0, 0.7), scipy.stats.gamma(3.0, scale=1 / 0.7)),
    (
        HurdleLogNormal,
        (0.25, -0.2, 0.7),
        scipy.stats.lognorm(0.7, scale=math.exp(-0.2)),
    ),
]


@pytest.mark.parametrize("cls, parameters, base", CASES)
def test_density_cdf_and_response_moments_match_scipy(cls, parameters, base):
    zero = parameters[0]
    law = cls(*(torch.tensor(p, dtype=torch.float64) for p in parameters))
    points = torch.tensor([0.001, 0.1, 0.5, 2.0, 10.0, 100.0], dtype=torch.float64)
    np.testing.assert_allclose(
        law.log_prob(points).numpy(),
        math.log1p(-zero) + base.logpdf(points),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        law.cdf(points).numpy(), zero + (1 - zero) * base.cdf(points), atol=1e-12
    )
    assert law.log_prob(torch.tensor(0.0)).item() == pytest.approx(math.log(zero))
    assert law.mean.item() == pytest.approx((1 - zero) * base.mean())
    assert law.variance.item() == pytest.approx(
        (1 - zero) * base.var() + zero * (1 - zero) * base.mean() ** 2
    )
    torch.testing.assert_close(
        law.cdf(torch.tensor([-torch.inf, -1.0, 0.0, torch.inf], dtype=torch.float64)),
        torch.tensor([0.0, 0.0, zero, 1.0], dtype=torch.float64),
    )
    assert torch.isnan(law.cdf(torch.tensor(torch.nan)))
    assert torch.isneginf(law.log_prob(torch.tensor(torch.inf)))


@pytest.mark.parametrize("cls, parameters, base", CASES)
def test_atom_and_positive_density_gradients_match_finite_differences(
    cls, parameters, base
):
    values = tuple(
        torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in parameters
    )
    response = torch.tensor([0.0, 0.2, 1.3], dtype=torch.float64)
    assert torch.autograd.gradcheck(lambda *args: cls(*args).log_prob(response), values)
    assert torch.autograd.gradcheck(lambda *args: cls(*args).mean, values)
    assert torch.autograd.gradcheck(lambda *args: cls(*args).variance, values)
    if cls is HurdleLogNormal:
        assert torch.autograd.gradcheck(lambda *args: cls(*args).cdf(response), values)


@pytest.mark.parametrize("cls, parameters, base", CASES)
@pytest.mark.parametrize("zero", [0.0, 1.0])
def test_inflation_endpoints_use_safe_inactive_branches(cls, parameters, base, zero):
    values = tuple(
        torch.tensor(p, dtype=torch.float64, requires_grad=True)
        for p in (zero, *parameters[1:])
    )
    law = cls(*values, validate_args=True)
    response = torch.tensor(0.2 if zero == 0 else 0.0, dtype=torch.float64)
    score = law.log_prob(response)
    expected = base.logpdf(0.2) if zero == 0 else 0.0
    assert score.item() == pytest.approx(expected)
    assert all(torch.isfinite(part) for part in torch.autograd.grad(score, values))
    assert torch.isneginf(law.log_prob(torch.tensor(0.0 if zero == 0 else 0.2)))
    with torch.random.fork_rng(devices=[]):
        samples = law.sample(torch.Size([20]))
    assert bool(samples.eq(0).all()) if zero == 1 else bool(samples.gt(0).all())


@pytest.mark.parametrize("cls, parameters, base", CASES)
def test_sampling_matches_atom_first_and_second_moments(cls, parameters, base):
    probability = parameters[0]
    law = cls(*(torch.tensor(p, dtype=torch.float64) for p in parameters))
    count = 20000
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(719)
        sample = law.sample(torch.Size([count]))
    assert abs(sample.eq(0).double().mean().item() - probability) < 5 * math.sqrt(
        probability * (1 - probability) / count
    )
    for order in [1, 2]:
        mean = (1 - probability) * base.moment(order)
        variance = (1 - probability) * base.moment(2 * order) - mean**2
        assert abs(sample.pow(order).mean().item() - mean) < 5 * math.sqrt(
            variance / count
        )


@pytest.mark.parametrize("cls, parameters, base", CASES)
def test_broadcast_expand_support_and_validation(cls, parameters, base):
    values = [torch.tensor(p, dtype=torch.float64) for p in parameters]
    values[0] = torch.tensor([[0.2], [0.4]], dtype=torch.float64)
    values[1] = values[1].expand(3)
    law = cls(*values, validate_args=True)
    assert law.batch_shape == (2, 3)
    assert law.expand(torch.Size([4, 2, 3])).sample(torch.Size([2])).shape == (
        2,
        4,
        2,
        3,
    )
    with pytest.raises(ValueError):
        law.log_prob(torch.tensor(-1.0))
    assert torch.isneginf(
        cls(*parameters, validate_args=False).log_prob(torch.tensor(-1.0))
    )
    for invalid in [-0.1, 1.1]:
        with pytest.raises(ValueError):
            cls(invalid, *parameters[1:], validate_args=True)
    with pytest.raises(ValueError):
        cls(parameters[0], parameters[1], 0.0, validate_args=True)


@pytest.mark.parametrize("cls", [ConditionalHurdleGamma, ConditionalHurdleLogNormal])
def test_conditional_native_wrapper_scores_and_samples_with_gradients(cls):
    morphism = cls(Euclidean(name="X", dim=2), Euclidean(name="Y", dim=1))
    x = torch.tensor([[0.2, -0.1], [-0.3, 0.5]], requires_grad=True)
    values = torch.tensor([[0.0], [0.7]])
    scores = morphism.log_prob(x, values)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(torch.autograd.grad(scores.sum(), x)[0]).all()
    assert morphism.rsample(x).shape == (2, 1)


@pytest.mark.parametrize(
    "name, parameters, base",
    [
        ("HurdleGamma", (0.3, 0.4, 1.7), scipy.stats.gamma(0.4, scale=1 / 1.7)),
        (
            "HurdleLogNormal",
            (0.25, -0.2, 0.7),
            scipy.stats.lognorm(0.7, scale=math.exp(-0.2)),
        ),
    ],
)
@pytest.mark.parametrize("rows", [1, 2])
def test_fixed_mixed_inline_and_reference_execution_agree(name, parameters, base, rows):
    fields = (
        ("zero_prob", "concentration", "rate")
        if name == "HurdleGamma"
        else ("zero_prob", "loc", "scale")
    )
    arguments = dict(zip(fields, parameters, strict=True))
    backend = ReferenceBackend()
    for point in [0.0, 0.2, 1.3]:
        expected = (
            math.log(parameters[0])
            if point == 0
            else math.log1p(-parameters[0]) + base.logpdf(point)
        )
        assert backend.family_log_prob(name, arguments, point) == pytest.approx(
            expected
        )
        source = f"object Row : FinSet {rows}\nprogram model : Row -> Row\n    observe y : Row <- {name}({', '.join(f'{name}={value}' for name, value in zip(fields, parameters, strict=True))})\n    return y\nexport model\n"
        program = loads(source).morphism
        recorded = trace(
            program,
            torch.zeros(1, 1, dtype=torch.long),
            {"y": torch.tensor([point] * rows)},
        )
        assert recorded.sites["y"].log_prob.sum().item() == pytest.approx(
            rows * expected, abs=1e-6
        )
        mixed = source.replace(str(parameters[0]), "hu", 1)
        dynamic = torch.tensor(parameters[0], requires_grad=True)
        recorded = trace(
            loads(mixed).morphism,
            torch.zeros(1, 1, dtype=torch.long),
            {"y": torch.tensor([point] * rows), "hu": dynamic},
        )
        assert recorded.sites["y"].log_prob.sum().item() == pytest.approx(
            rows * expected, abs=1e-6
        )
        assert torch.isfinite(
            torch.autograd.grad(recorded.sites["y"].log_prob.sum(), dynamic)[0]
        )
    generator = random.Random(17)
    samples = np.array(
        [backend.family_sample(name, arguments, generator) for _ in range(10000)]
    )
    mean = (1 - parameters[0]) * base.mean()
    variance = (1 - parameters[0]) * base.var() + parameters[0] * (
        1 - parameters[0]
    ) * base.mean() ** 2
    assert abs(samples.mean() - mean) < 5 * math.sqrt(variance / len(samples))
    for probability in [0.0, 1.0]:
        endpoint = arguments | {"zero_prob": probability}
        valid_point = 0.5 if probability == 0 else 0.0
        expected = base.logpdf(valid_point) if probability == 0 else 0.0
        assert backend.family_log_prob(name, endpoint, valid_point) == pytest.approx(
            expected
        )
        assert (
            backend.family_log_prob(name, endpoint, 0.0 if probability == 0 else 0.5)
            == -math.inf
        )
