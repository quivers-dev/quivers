"""Conditional plate draws use the same row parameters as their density."""

import pytest
import torch

from quivers.continuous.inline import MixedInlineDistribution, make_fixed_normal
from quivers.continuous.plate import PlateDraw
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.inference.trace import trace


def _conditional_normal():
    return MixedInlineDistribution(
        Euclidean(name="Params", dim=2),
        Euclidean(name="Response", dim=1),
        param_spec=[("var", None), ("var", None)],
        dist_builder=lambda parameters: torch.distributions.Normal(
            parameters[0], parameters[1]
        ),
    )


def test_per_row_conditioning_matches_independent_draws_density_and_gradients():
    mean = torch.tensor([-2.0, 0.5, 3.0], requires_grad=True)
    scale = torch.tensor([0.1, 0.5, 2.0], requires_grad=True)
    family = PlateDraw(3, _conditional_normal())
    parameters = torch.stack((mean, scale), dim=-1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        actual = family.rsample(parameters)
        state = torch.get_rng_state().clone()
        torch.manual_seed(17)
        expected = torch.distributions.Normal(mean, scale).rsample()
        assert torch.equal(state, torch.get_rng_state())
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        family.log_prob(parameters, actual).reshape(()),
        torch.distributions.Normal(mean, scale).log_prob(actual).sum(),
    )
    actual_gradients = torch.autograd.grad(
        actual.sum(), (mean, scale), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected.sum(), (mean, scale))
    for actual_part, expected_part in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_part, expected_part)


@pytest.mark.parametrize("fixed", [False, True])
def test_shared_conditioning_draws_independent_values(fixed):
    family = (
        make_fixed_normal(1.5, 0.7, Euclidean(name="Response", dim=1))
        if fixed
        else _conditional_normal()
    )
    plate = PlateDraw(4, family)
    parameters = torch.tensor([[1.5, 0.7]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        actual = plate.rsample(parameters)
        torch.manual_seed(31)
        expected = torch.distributions.Normal(1.5, 0.7).rsample(torch.Size((4,)))
    torch.testing.assert_close(actual, expected)
    assert actual.unique().numel() == 4


@pytest.mark.parametrize("runs", [1, 3])
def test_group_conditioning_and_independent_run_axes(runs):
    source = """object Group : FinSet 3
object Input : Real 1
object Output : Real 1
program p(mean : Tensor[Real]([Group]), scale : Tensor[Real]([Group])) : Input -> Output
    sample coefficients : Group <- Normal(loc=mean, scale=scale)
    return coefficients
export p
"""
    program = loads(source).morphism
    means = torch.tensor([-100.0, 0.0, 100.0])
    scales = torch.full((3,), 1e-5)
    recorded = trace(program, torch.zeros(runs, 1), {"mean": means, "scale": scales})
    assert recorded.sites["coefficients"].value.shape == (
        (3,) if runs == 1 else (runs, 3)
    )
    expected = means if runs == 1 else means.expand(runs, 3)
    torch.testing.assert_close(
        recorded.sites["coefficients"].value, expected, atol=1e-3, rtol=0
    )


def test_lookup_conditioning_uses_every_group_entry():
    from quivers.continuous.families import ConditionalNormal
    from quivers.continuous.param_source import LookupSource
    from quivers.core.objects import FinSet

    lookup = LookupSource(3, 2)
    with torch.no_grad():
        lookup.table.copy_(
            torch.tensor([[-100.0, -10.0], [0.0, -10.0], [100.0, -10.0]])
        )
    family = ConditionalNormal(
        FinSet(name="Group", cardinality=3),
        Euclidean(name="Response", dim=1),
        param_source=lookup,
    )
    plate = PlateDraw(3, family)
    values = plate.rsample()
    torch.testing.assert_close(values, lookup.table[:, 0], atol=1e-3, rtol=0)
    gradient = torch.autograd.grad(values.sum(), lookup.table)[0]
    torch.testing.assert_close(gradient[:, 0], torch.ones(3))
