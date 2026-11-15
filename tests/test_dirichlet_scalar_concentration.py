"""Runtime symmetric Dirichlet inputs retain dimension and gradients."""

import pytest
import torch

from quivers.continuous.inline import make_inline_distribution
from quivers.continuous.spaces import Euclidean, Simplex
from quivers.dsl import loads
from quivers.dsl.ast_nodes import DrawArgName, DrawArgNamed
from quivers.inference.trace import trace


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("scalar", [False, True])
def test_runtime_dirichlet_concentrations_match_density_and_gradient_oracles(
    batched, scalar
):
    shape = () if scalar else (3,)
    width = 1 if scalar else 3
    family, names = make_inline_distribution(
        "Dirichlet",
        (DrawArgNamed(parameter="concentration", value=DrawArgName(text="alpha")),),
        Simplex(name="P", dim=3),
        variable_types={"alpha": Euclidean(name="Alpha", dim=width)},
        value_extents={"alpha": shape},
    )
    assert names == ("alpha",)
    if scalar:
        alpha = torch.tensor([[0.8], [1.5]]) if batched else torch.tensor(1.3)
    else:
        alpha = (
            torch.tensor([[0.8, 1.0, 2.0], [1.5, 0.7, 2.3]])
            if batched
            else torch.tensor([0.8, 1.5, 2.0])
        )
    alpha.requires_grad_()
    concentration = (
        alpha.expand(2, 3)
        if scalar and batched
        else alpha.expand(3)
        if scalar
        else alpha
    )
    point = (
        torch.tensor([[0.2, 0.3, 0.5], [0.4, 0.35, 0.25]])
        if batched
        else torch.tensor([[0.2, 0.3, 0.5]])
    )
    actual = family.log_prob(alpha, point)
    expected = (
        torch.lgamma(concentration.sum(-1))
        - torch.lgamma(concentration).sum(-1)
        + ((concentration - 1) * point.log()).sum(-1)
    )
    torch.testing.assert_close(actual, expected.reshape(actual.shape))
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), alpha, retain_graph=True)[0],
        torch.autograd.grad(expected.sum(), alpha)[0],
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(23)
        draws = family.rsample(alpha)
        torch.manual_seed(23)
        reference = torch.distributions.Dirichlet(concentration).rsample()
    torch.testing.assert_close(draws, reference.reshape(draws.shape))
    weighted = (draws * torch.tensor([0.2, 0.7, 1.1])).sum()
    derivative = torch.autograd.grad(weighted, alpha)[0]
    assert torch.isfinite(derivative).all()
    assert derivative.abs().sum() > 0


@pytest.mark.parametrize("runs", [1, 3])
def test_runtime_scalar_concentration_generates_each_plate_simplex(runs):
    source = """object Group : FinSet 2
object Category : FinSet 3
object Input : Real 1
object Output : Real 3
program prior(alpha : Real) : Input -> Output
    sample theta : Group <- Dirichlet(concentration=alpha) [over=Category]
    return theta
export prior
"""
    program = loads(source).morphism
    alpha = torch.tensor(1.4, requires_grad=True)
    recorded = trace(program, torch.zeros(runs, 1), {"alpha": alpha})
    draws = recorded.sites["theta"].value
    assert draws.shape == ((2, 3) if runs == 1 else (runs, 2, 3))
    torch.testing.assert_close(draws.sum(-1), torch.ones(draws.shape[:-1]))
    expected = torch.distributions.Dirichlet(alpha.expand(3)).log_prob(draws).sum(-1)
    torch.testing.assert_close(recorded.log_joint, expected.reshape(runs))
    assert torch.isfinite(torch.autograd.grad(recorded.log_joint.sum(), alpha)[0])


def test_runtime_dirichlet_rejects_incompatible_vector_width():
    family, _ = make_inline_distribution(
        "Dirichlet",
        (DrawArgNamed(parameter="concentration", value=DrawArgName(text="alpha")),),
        Simplex(name="P", dim=3),
        variable_types={"alpha": Euclidean(name="Alpha", dim=2)},
        value_extents={"alpha": (2,)},
    )
    with pytest.raises(ValueError, match="concentration has width 2"):
        family.rsample(torch.tensor([0.8, 1.5]))
