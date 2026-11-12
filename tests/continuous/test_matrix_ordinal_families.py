"""Inline OrderedProbit, InverseWishart, and MatrixNormal, and the positive-definite bijector.

Each distribution is checked against an independent SciPy density in
double precision, its sampler against closed-form moments with a
tolerance set from the Monte Carlo error, and a deliberately wrong
construction is shown to fail the same checks. The positive-definite
bijector's log-Jacobian is checked against the determinant of its
autograd Jacobian, and inference over an inverse-Wishart covariance
recovers the conjugate posterior mean. The two execution paths agree on
each family at host and literal parameters.
"""

from __future__ import annotations

import math

import pytest
import scipy.stats
import torch

from quivers.continuous._matrix import (
    InverseWishart,
    MatrixNormal,
    PositiveDefiniteTransform,
)
from quivers.continuous._ordered import OrderedProbit
from quivers.continuous.families import ConditionalMatrixNormal
from quivers.continuous.spaces import Euclidean
from quivers.dsl import loads
from quivers.inference.mcmc import MCMC, NUTSKernel
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace

_F = torch.float64
_FACTOR = torch.tensor([[1.0, 0.0, 0.0], [0.3, 0.8, 0.0], [-0.2, 0.1, 1.2]], dtype=_F)
_ROWS = torch.tensor([[1.0, 0.3], [0.3, 2.0]], dtype=_F)
_COLS = torch.tensor([[1.0, 0.2, 0.0], [0.2, 1.5, 0.1], [0.0, 0.1, 0.7]], dtype=_F)
_LOC = torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.0, -0.5]], dtype=_F)


def _scale(factor: torch.Tensor) -> torch.Tensor:
    """The conventional inverse-Wishart scale a Wishart factor names.

    Parameters
    ----------
    factor : torch.Tensor
        The Cholesky factor ``L`` of the Wishart whose inverse is drawn.

    Returns
    -------
    torch.Tensor
        ``(L L^T)^-1``.
    """
    return torch.linalg.inv(factor @ factor.T)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_inverse_wishart_scores_the_scipy_density(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(3, 3, generator=generator, dtype=_F)
    point = noise @ noise.T + 0.5 * torch.eye(3, dtype=_F)
    density = InverseWishart(torch.tensor(6.0, dtype=_F), _FACTOR).log_prob(point)
    expected = scipy.stats.invwishart(df=6.0, scale=_scale(_FACTOR).numpy()).logpdf(
        point.numpy()
    )
    assert float(density) == pytest.approx(float(expected), rel=1e-10)
    # Reading the factor as the conventional scale's is another density.
    misread = scipy.stats.invwishart(
        df=6.0, scale=(_FACTOR @ _FACTOR.T).numpy()
    ).logpdf(point.numpy())
    assert abs(float(density) - float(misread)) > 1e-2


def test_inverse_wishart_draws_have_the_closed_form_mean() -> None:
    torch.manual_seed(4)
    df = 9.0
    draws = InverseWishart(torch.tensor(df, dtype=_F), _FACTOR).rsample(
        torch.Size((20000,))
    )
    mean = _scale(_FACTOR) / (df - 3 - 1)
    # The entries' standard deviations are below 0.4 at these values, so
    # the mean of 20000 draws is within 4 standard errors at 0.012.
    torch.testing.assert_close(draws.mean(0), mean, atol=0.012, rtol=0.0)
    assert (draws.mean(0) - (_FACTOR @ _FACTOR.T) / (df - 4)).abs().max() > 0.05


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matrix_normal_scores_the_scipy_density(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    point = torch.randn(2, 3, generator=generator, dtype=_F)
    density = MatrixNormal(_LOC, _ROWS, _COLS).log_prob(point)
    expected = scipy.stats.matrix_normal(
        mean=_LOC.numpy(), rowcov=_ROWS.numpy(), colcov=_COLS.numpy()
    ).logpdf(point.numpy())
    assert float(density) == pytest.approx(float(expected), rel=1e-10)
    swapped = scipy.stats.matrix_normal(
        mean=_LOC.numpy(),
        rowcov=_ROWS.numpy(),
        colcov=_COLS.flip(0).flip(1).numpy(),
    ).logpdf(point.numpy())
    assert abs(float(density) - float(swapped)) > 1e-2


def test_matrix_normal_draws_have_the_kronecker_covariance() -> None:
    torch.manual_seed(5)
    draws = MatrixNormal(_LOC, _ROWS, _COLS).rsample(torch.Size((40000,)))
    flat = (draws - _LOC).reshape(-1, 6)
    covariance = flat.T @ flat / flat.shape[0]
    expected = torch.kron(_ROWS, _COLS)
    # Each sample covariance entry has a standard deviation below
    # sqrt(2 * 3.0^2 / 40000) < 0.022; four of them is 0.09.
    torch.testing.assert_close(covariance, expected, atol=0.09, rtol=0.0)
    # The Kronecker product in the other order lays the entries out by
    # column, which is not this covariance.
    assert (covariance - torch.kron(_COLS, _ROWS)).abs().max() > 0.2


def test_the_conditional_matrix_normal_scores_its_kronecker_covariance() -> None:
    torch.manual_seed(6)
    morphism = ConditionalMatrixNormal(
        Euclidean(name="x", dim=1), Euclidean(name="y", dim=6), rows=2, cols=3
    ).double()
    x = torch.randn(1, 1, dtype=_F)
    distribution = morphism._get_dist(x)
    point = torch.randn(1, 2, 3, dtype=_F)
    expected = scipy.stats.matrix_normal(
        mean=distribution.loc[0].detach().numpy(),
        rowcov=distribution.row_covariance[0].detach().numpy(),
        colcov=distribution.col_covariance[0].detach().numpy(),
    ).logpdf(point[0].numpy())
    assert float(morphism.log_prob(x, point)) == pytest.approx(
        float(expected), rel=1e-9
    )


@pytest.mark.parametrize("eta", [-1.3, 0.4, 2.2])
def test_ordered_probit_scores_the_normal_cdf_differences(eta: float) -> None:
    cuts = [-1.0, 0.0, 1.5]
    distribution = OrderedProbit(
        torch.tensor(eta, dtype=_F), torch.tensor(cuts, dtype=_F)
    )
    bounds = [-math.inf, *cuts, math.inf]
    for category in range(4):
        expected = math.log(
            scipy.stats.norm.cdf(bounds[category + 1] - eta)
            - scipy.stats.norm.cdf(bounds[category] - eta)
        )
        density = distribution.log_prob(torch.tensor(category))
        assert float(density) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("eta", [-30.0, 30.0])
def test_ordered_probit_keeps_far_categories_finite(eta: float) -> None:
    cuts = torch.tensor([-1.0, 0.0, 1.5], dtype=_F)
    log_probs = OrderedProbit(torch.tensor(eta, dtype=_F), cuts)._category_log_probs()
    assert torch.isfinite(log_probs).all()
    # Far categories match SciPy's log survival and log distribution
    # functions, which are accurate in the tails.
    if eta > 0:
        expected = scipy.stats.norm.logcdf(-1.0 - eta)
        assert float(log_probs[0]) == pytest.approx(float(expected), rel=1e-9)
    else:
        expected = scipy.stats.norm.logsf(1.5 - eta)
        assert float(log_probs[3]) == pytest.approx(float(expected), rel=1e-9)
    assert float(log_probs.exp().sum()) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("dim", [1, 2, 4])
def test_the_positive_definite_bijector_has_its_autograd_jacobian(dim: int) -> None:
    torch.manual_seed(dim)
    transform = PositiveDefiniteTransform(dim)
    x = torch.randn(dim * (dim + 1) // 2, dtype=_F)
    y = transform(x)
    rows, cols = torch.tril_indices(dim, dim)
    jacobian = torch.autograd.functional.jacobian(lambda z: transform(z)[rows, cols], x)
    expected = torch.linalg.slogdet(jacobian).logabsdet
    assert float(transform.log_abs_det_jacobian(x, y)) == pytest.approx(
        float(expected), rel=1e-10, abs=1e-10
    )
    torch.testing.assert_close(transform.inv(y), x, rtol=1e-10, atol=1e-10)
    assert bool(torch.all(torch.linalg.eigvalsh(y) > 0))
    # Leaving out the exponential's contribution is a different Jacobian.
    diagonal = x[(rows == cols).nonzero().squeeze(-1)]
    assert (
        abs(
            float(transform.log_abs_det_jacobian(x, y) - diagonal.sum())
            - float(expected)
        )
        > 1e-3
    )


_IW_PROGRAM = """
object Row : FinSet 6
object K : FinSet 2
program p : Row -> Row
    sample S : K <- InverseWishart(df=5.0, scale_tril=[[1.0, 0.0], [0.3, 0.8]])
    observe y : Row <- MultivariateNormal(loc=[0.0, 0.0], covariance_matrix=S) [over=K]
    return y
export p
"""

_Y = torch.tensor(
    [[0.4, -1.2], [1.1, 0.3], [-0.7, 0.9], [0.2, 0.1], [-1.5, -0.4], [0.8, 1.6]]
)


def test_an_inverse_wishart_site_is_positive_definite_in_the_registry() -> None:
    program = loads(_IW_PROGRAM)
    recorded = trace(program.morphism, torch.zeros(1, 1), observations={"y": _Y})
    registry = LatentRegistry.from_trace(program.morphism, recorded)
    site = registry.sites["S"]
    assert site.constrained_shape == (2, 2)
    assert site.unconstrained_dim == 3
    assert isinstance(site.bijector, PositiveDefiniteTransform)


def test_nuts_recovers_the_conjugate_inverse_wishart_posterior_mean() -> None:
    torch.manual_seed(11)
    program = loads(_IW_PROGRAM)
    result = MCMC(NUTSKernel(), num_warmup=150, num_samples=400, num_chains=1).run(
        program.morphism, torch.zeros(1, 1), {"y": _Y}
    )
    factor = torch.tensor([[1.0, 0.0], [0.3, 0.8]], dtype=_F)
    y = _Y.to(_F)
    scale = _scale(factor) + y.T @ y
    degrees = 5.0 + y.shape[0]
    mean = scale / (degrees - 2 - 1)
    draws = result.samples["S"].to(_F).reshape(-1, 2, 2)
    # The posterior standard deviation of each entry is below 0.6; with at
    # least 100 effective draws the Monte Carlo error is below 0.06, and
    # 0.24 is four of it.
    torch.testing.assert_close(draws.mean(0), mean, atol=0.24, rtol=0.0)
    prior_only = _scale(factor) / (5.0 - 3.0)
    assert (draws.mean(0) - prior_only).abs().max() > 0.2


def _nested(value: torch.Tensor) -> float | tuple[object, ...]:
    """A tensor as the reference machine's nested tuples.

    Parameters
    ----------
    value : torch.Tensor
        The tensor.

    Returns
    -------
    float | tuple[object, ...]
        A number, or nested tuples of numbers.
    """
    if value.dim() == 0:
        return float(value.item())
    return tuple(_nested(item) for item in value)


def test_both_paths_score_host_matrix_parameters_as_scipy_does() -> None:
    source = """
object One : FinSet 1
object K : FinSet 3
object R : FinSet 2
object C : FinSet 3
program p : One -> One
    sample S : K <- InverseWishart(df=6.0, scale_tril=L)
    sample M <- MatrixNormal(loc=M0, row_covariance=U, col_covariance=V) [over=[R, C]]
    return S
export p
"""
    program = loads(source)
    sigma = torch.tensor(
        [[2.0, 0.3, 0.1], [0.3, 1.0, -0.2], [0.1, -0.2, 0.8]], dtype=_F
    )
    matrix = torch.tensor([[0.3, 0.9, -0.2], [0.1, 0.4, -1.0]], dtype=_F)
    data = {"L": _FACTOR, "M0": _LOC, "U": _ROWS, "V": _COLS}
    expected = float(
        scipy.stats.invwishart(df=6.0, scale=_scale(_FACTOR).numpy()).logpdf(
            sigma.numpy()
        )
        + scipy.stats.matrix_normal(
            mean=_LOC.numpy(), rowcov=_ROWS.numpy(), colcov=_COLS.numpy()
        ).logpdf(matrix.numpy())
    )
    values = {
        name: value.float() for name, value in {**data, "S": sigma, "M": matrix}.items()
    }
    compiled = float(program.morphism.log_joint(torch.zeros(1, 1), values).sum())
    reference = program.run(
        "p",
        data={name: _nested(value) for name, value in data.items()},
        sites={"S": _nested(sigma), "M": _nested(matrix)},
    ).log_joint
    assert compiled == pytest.approx(expected, rel=1e-5)
    assert reference == pytest.approx(expected, rel=1e-9)
