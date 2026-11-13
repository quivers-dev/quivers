"""Independent boundary and interior checks for inflated Beta distributions."""

import math

import pytest
import torch
from scipy.special import betaln

from quivers.continuous._zip_hurdle import ZeroOneInflatedBeta


@pytest.mark.parametrize(
    "zoi,coi", [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (1e-12, 1e-10)]
)
def test_boundary_atoms_preserve_exact_weights(zoi: float, coi: float) -> None:
    params = [torch.tensor(x, dtype=torch.float64) for x in (0.2, 3.0, zoi, coi)]
    dist = ZeroOneInflatedBeta(*params)
    weights = torch.tensor([zoi * (1 - coi), zoi * coi], dtype=torch.float64)
    actual = dist.log_prob(torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(actual, weights.log(), rtol=1e-12, atol=1e-12)
    clipped = weights.clamp_min(1e-6).log()
    assert not torch.allclose(actual, clipped)


@pytest.mark.parametrize(
    "mu,phi,y", [(0.2, 1e-10, 0.3), (1e-10, 3.0, 0.2), (0.2, 3.0, 1e-10)]
)
def test_interior_density_preserves_parameters(mu: float, phi: float, y: float) -> None:
    zoi, coi = 0.1, 0.3
    params = [torch.tensor(x, dtype=torch.float64) for x in (mu, phi, zoi, coi)]
    value = torch.tensor(y, dtype=torch.float64)
    actual = ZeroOneInflatedBeta(*params).log_prob(value)
    a, b = mu * phi, (1 - mu) * phi
    expected = (
        math.log1p(-zoi)
        + (a - 1) * math.log(y)
        + (b - 1) * math.log1p(-y)
        - betaln(a, b)
    )
    assert actual.item() == pytest.approx(expected, rel=1e-12, abs=1e-12)
    clipped_mu, clipped_phi, clipped_y = max(mu, 1e-6), max(phi, 1e-6), max(y, 1e-6)
    wrong = math.log1p(-zoi) + torch.distributions.Beta(
        torch.tensor(clipped_mu * clipped_phi, dtype=torch.float64),
        torch.tensor((1 - clipped_mu) * clipped_phi, dtype=torch.float64),
    ).log_prob(torch.tensor(clipped_y, dtype=torch.float64))
    assert abs(actual - wrong) > 1.0


@pytest.mark.parametrize("mu", [0.0, 1.0])
def test_beta_mean_requires_open_interval(mu: float) -> None:
    with pytest.raises(ValueError):
        ZeroOneInflatedBeta(*(torch.tensor(x) for x in (mu, 3.0, 0.2, 0.5)))


def test_outside_support_is_not_a_boundary_atom() -> None:
    args = [torch.tensor(x, dtype=torch.float64) for x in (0.2, 3.0, 0.2, 0.5)]
    values = torch.tensor([-0.1, 1.1], dtype=torch.float64)
    with pytest.raises(ValueError):
        ZeroOneInflatedBeta(*args, validate_args=True).log_prob(values)
    assert torch.isneginf(
        ZeroOneInflatedBeta(*args, validate_args=False).log_prob(values)
    ).all()


@pytest.mark.parametrize(
    "zoi,coi,y", [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0), (0.0, 0.0, 0.3), (0.0, 1.0, 0.3)]
)
def test_unused_components_do_not_poison_gradients(
    zoi: float, coi: float, y: float
) -> None:
    params = [
        torch.tensor(x, dtype=torch.float64, requires_grad=True)
        for x in (0.2, 3.0, zoi, coi)
    ]
    result = ZeroOneInflatedBeta(*params).log_prob(torch.tensor(y, dtype=torch.float64))
    gradients = torch.autograd.grad(result, params)
    assert all(torch.isfinite(g) for g in gradients)


def test_sampling_broadcasts_all_parameter_batches() -> None:
    dist = ZeroOneInflatedBeta(
        torch.full((2,), 0.2),
        torch.full((2,), 3.0),
        torch.tensor([[0.0], [1.0], [1.0]]),
        torch.tensor([[0.5], [0.0], [1.0]]),
    )
    draws = dist.sample(torch.Size([5]))
    assert draws.shape == (5, 3, 2)
    assert (draws[:, 0] > 0).all() and (draws[:, 0] < 1).all()
    assert (draws[:, 1] == 0).all()
    assert (draws[:, 2] == 1).all()
