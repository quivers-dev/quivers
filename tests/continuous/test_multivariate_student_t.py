"""Independent marginal law, geometry, and differentiation oracles."""

from unittest.mock import patch
import random

import numpy as np
import pytest
from scipy import stats
import torch

from quivers.continuous._multivariate_student_t import MultivariateStudentT


def _host(value):
    return value.item() if value.ndim == 0 else tuple(_host(part) for part in value)


@pytest.mark.parametrize("df", [0.3, 1.0, 2.0, 7.5, 1000.0])
@pytest.mark.parametrize(
    "parameterization", ["scale_tril", "covariance_matrix", "precision_matrix"]
)
def test_density_and_broadcasting_match_scipy(df, parameterization):
    loc = torch.tensor([[0.3, -0.4, 0.1], [-0.7, 0.2, 0.8]], dtype=torch.double)
    factor = torch.tensor(
        [[1.2, 0, 0], [0.3, 0.7, 0], [-0.2, 0.1, 0.9]], dtype=torch.double
    )
    scale = factor @ factor.T
    parameter = {
        "scale_tril": factor,
        "covariance_matrix": scale,
        "precision_matrix": torch.linalg.inv(scale),
    }[parameterization]
    distribution = MultivariateStudentT(df, loc, **{parameterization: parameter})
    point = torch.tensor([[[-0.1, 0.7, 1.1]], [[0.8, -1.2, 0.2]]], dtype=torch.double)
    expected = np.array(
        [
            [
                stats.multivariate_t.logpdf(
                    row[0].numpy(), mean.numpy(), scale.numpy(), df
                )
                for mean in loc
            ]
            for row in point
        ]
    )
    torch.testing.assert_close(
        distribution.log_prob(point), torch.from_numpy(expected), atol=2e-12, rtol=2e-12
    )
    assert distribution.batch_shape == (2,)
    assert distribution.event_shape == (3,)
    assert distribution.expand((4, 2)).rsample((5,)).shape == (5, 4, 2, 3)


def test_scalar_reduction_all_head_gradients_and_forward_second_order():
    df = torch.tensor(3.7, dtype=torch.double, requires_grad=True)
    loc = torch.tensor([0.4], dtype=torch.double, requires_grad=True)
    factor = torch.tensor([[0.8]], dtype=torch.double, requires_grad=True)
    point = torch.tensor([1.1], dtype=torch.double, requires_grad=True)

    def score(df, loc, factor, point):
        return MultivariateStudentT(
            df, loc, scale_tril=factor, validate_args=False
        ).log_prob(point)

    actual = score(df, loc, factor, point)
    oracle = torch.distributions.StudentT(df, loc[0], factor[0, 0]).log_prob(point[0])
    torch.testing.assert_close(actual, oracle)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual, (df, loc, factor, point), retain_graph=True),
        torch.autograd.grad(oracle, (df, loc, factor, point)),
        strict=True,
    ):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert torch.autograd.gradcheck(score, (df, loc, factor, point))
    assert torch.autograd.gradgradcheck(score, (df, loc, factor, point))
    forward = torch.func.jacfwd(score, argnums=(0, 1, 2, 3))(df, loc, factor, point)
    reverse = torch.func.jacrev(score, argnums=(0, 1, 2, 3))(df, loc, factor, point)
    for left, right in zip(forward, reverse, strict=True):
        torch.testing.assert_close(left, right)


def test_multivariate_scale_gradients_match_independent_closed_form():
    df = torch.tensor([1.2, 6.4], dtype=torch.double, requires_grad=True)
    loc = torch.tensor([0.2, -0.5], dtype=torch.double, requires_grad=True)
    factor_raw = torch.tensor(
        [[0.8, 0.0], [0.3, 1.1]], dtype=torch.double, requires_grad=True
    )
    point = torch.tensor([[1.2, 0.7], [-0.4, 0.2]], dtype=torch.double)
    factor = factor_raw.tril()
    covariance = factor @ factor.T
    residual = point - loc
    squared = torch.einsum(
        "...i,ij,...j->...", residual, torch.linalg.inv(covariance), residual
    )
    oracle = (
        torch.lgamma((df + 2) / 2)
        - torch.lgamma(df / 2)
        - torch.log(df * torch.pi)
        - torch.linalg.slogdet(covariance)[1] / 2
        - (df + 2) / 2 * torch.log1p(squared / df)
    )
    actual = MultivariateStudentT(df, loc, scale_tril=factor).log_prob(point)
    torch.testing.assert_close(actual, oracle)
    for left, right in zip(
        torch.autograd.grad(actual.sum(), (df, loc, factor_raw), retain_graph=True),
        torch.autograd.grad(oracle.sum(), (df, loc, factor_raw)),
        strict=True,
    ):
        torch.testing.assert_close(left, right)


def test_sample_moments_cross_covariance_and_pathwise_head_gradients():
    df = torch.tensor(8.0, dtype=torch.double, requires_grad=True)
    loc = torch.tensor([0.3, -0.2], dtype=torch.double, requires_grad=True)
    factor = torch.tensor(
        [[0.9, 0.0], [0.4, 0.7]], dtype=torch.double, requires_grad=True
    )
    distribution = MultivariateStudentT(df, loc, scale_tril=factor)
    with torch.random.fork_rng():
        torch.manual_seed(341)
        draws = distribution.rsample((80000,))
    torch.testing.assert_close(draws.mean(0), loc, atol=0.015, rtol=0)
    torch.testing.assert_close(
        torch.cov(draws.T), (factor @ factor.T) * 8 / 6, atol=0.025, rtol=0
    )
    torch.testing.assert_close(
        distribution.variance, (factor @ factor.T).diagonal() * 8 / 6
    )
    for gradient in torch.autograd.grad(draws.square().mean(), (df, loc, factor)):
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_moment_boundaries_and_large_df_normal_limit():
    distribution = MultivariateStudentT(
        torch.tensor([0.5, 1.0, 1.5, 2.0, 3.0], dtype=torch.double),
        torch.zeros(1, dtype=torch.double),
        scale_tril=torch.eye(1, dtype=torch.double),
    )
    assert torch.isnan(distribution.mean[:2]).all()
    assert torch.isnan(distribution.variance[:2]).all()
    assert torch.isinf(distribution.variance[2:4]).all()
    torch.testing.assert_close(
        distribution.variance[4], torch.tensor([3.0], dtype=torch.double)
    )
    loc, factor = torch.zeros(2, dtype=torch.double), torch.eye(2, dtype=torch.double)
    point = torch.tensor([0.2, -0.8], dtype=torch.double)
    expected = torch.distributions.MultivariateNormal(loc, scale_tril=factor).log_prob(
        point
    )
    torch.testing.assert_close(
        MultivariateStudentT(1e16, loc, scale_tril=factor).log_prob(point),
        expected,
        atol=1e-13,
        rtol=1e-13,
    )


def test_factor_path_expand_and_batch_selection_do_not_factor_or_invert():
    from quivers.continuous._batch_selection import select_batch

    loc = torch.zeros(4, 2, dtype=torch.double)
    factor = torch.eye(2, dtype=torch.double)
    with (
        patch(
            "torch.linalg.cholesky",
            side_effect=AssertionError("unexpected factorization"),
        ),
        patch("torch.linalg.inv", side_effect=AssertionError("unexpected inverse")),
    ):
        distribution = MultivariateStudentT(
            torch.tensor([3.0, 4.0, 5.0, 6.0]), loc, scale_tril=factor
        )
        expanded = distribution.expand((3, 4))
        expanded.log_prob(loc)
        expanded.rsample((2,))
        selected = select_batch(distribution, torch.tensor([True, False, True, False]))
        torch.testing.assert_close(selected.df, distribution.df[[0, 2]])
        torch.testing.assert_close(
            selected.log_prob(loc[[0, 2]]), distribution.log_prob(loc)[[0, 2]]
        )


@pytest.mark.parametrize("df", [0.0, -1.0, float("nan")])
def test_invalid_degrees_of_freedom_fail(df):
    with pytest.raises(ValueError):
        MultivariateStudentT(df, torch.zeros(2), scale_tril=torch.eye(2))


@pytest.mark.parametrize(
    "parameterization", ["scale_tril", "covariance_matrix", "precision_matrix"]
)
@pytest.mark.parametrize("rows", [2, 3])
def test_checked_source_native_reference_and_all_parameter_gradients(
    parameterization, rows
):
    from quivers.dsl import loads
    from quivers.inference import trace

    source = f"""object Row:FinSet {rows}
object Dim:FinSet 2
program model(df:Real,loc:Tensor[Real]([Row,Dim]),factor:Tensor[Real]([Dim,Dim]),y:Tensor[Real]([Row,Dim])):Row->Row
    observe y:Row <- MultivariateStudentT(df=df,loc=loc,{parameterization}=factor) [over=Dim]
    return y
export model
"""
    df = torch.tensor(4.7, dtype=torch.double, requires_grad=True)
    loc = (
        torch.linspace(-0.3, 0.5, rows * 2, dtype=torch.double)
        .reshape(rows, 2)
        .requires_grad_()
    )
    lower = torch.tensor([[1.2, 0.0], [0.3, 0.8]], dtype=torch.double)
    scale = lower @ lower.T
    factor = (
        {
            "scale_tril": lower,
            "covariance_matrix": scale,
            "precision_matrix": torch.linalg.inv(scale),
        }[parameterization]
        .clone()
        .requires_grad_()
    )
    y = torch.linspace(0.7, -0.2, rows * 2, dtype=torch.double).reshape(rows, 2)
    program = loads(source)
    values = {"df": df, "loc": loc, "factor": factor, "y": y}
    recorded = trace(program.morphism, torch.zeros(1, rows, dtype=torch.double), values)
    oracle = (
        MultivariateStudentT(df, loc, **{parameterization: factor}).log_prob(y).sum()
    )
    torch.testing.assert_close(recorded.log_joint.sum(), oracle)
    for actual, expected in zip(
        torch.autograd.grad(
            recorded.log_joint.sum(), (df, loc, factor), retain_graph=True
        ),
        torch.autograd.grad(oracle, (df, loc, factor)),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    reference = program.run(
        "model",
        data={key: _host(value.detach()) for key, value in values.items()},
        sites={},
    )
    assert reference.log_joint == pytest.approx(float(oracle.detach()), abs=2e-12)


def test_reference_sampling_moments_and_invalid_gaussian_geometry():
    from quivers.qiec.distributions import ReferenceBackend
    from quivers.qiec.reference_support import DistributionError

    backend = ReferenceBackend()
    arguments = {"df": 8.0, "loc": (0.3, -0.2), "scale_tril": ((0.9, 0.0), (0.4, 0.7))}
    rng = random.Random(39)
    draws = np.asarray(
        [backend.family_sample("MultivariateStudentT", arguments, rng) for _ in range(16000)]
    )
    np.testing.assert_allclose(draws.mean(0), [0.3, -0.2], atol=0.025, rtol=0)
    np.testing.assert_allclose(
        np.cov(draws.T),
        np.array([[0.81, 0.36], [0.36, 0.65]]) * 8 / 6,
        atol=0.045,
        rtol=0,
    )
    for name in (
        "MultivariateNormal",
        "MultivariateStudentT",
        "AdditiveLogRatioNormal",
    ):
        malformed = arguments | {"scale_tril": ((0.9, 0.2), (0.4, 0.7))}
        with pytest.raises(DistributionError, match="Cholesky"):
            backend.family_sample(name, malformed, random.Random(4))


def test_conditional_wrapper_reuses_gaussian_head_unpacking():
    from quivers.continuous.families import ConditionalMultivariateStudentT
    from quivers.continuous.spaces import Euclidean

    kernel = ConditionalMultivariateStudentT(
        Euclidean(name="X", dim=2), Euclidean(name="Y", dim=3)
    ).double()
    x = torch.tensor([[0.2, -0.3], [0.4, 0.7]], dtype=torch.double)
    law = kernel._get_dist(x)
    assert law.batch_shape == (2,)
    assert law.event_shape == (3,)
    score = law.log_prob(torch.ones(2, 3, dtype=torch.double)).sum()
    gradients = torch.autograd.grad(score, tuple(kernel.parameters()))
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all(value.abs().sum() > 0 for value in gradients)
