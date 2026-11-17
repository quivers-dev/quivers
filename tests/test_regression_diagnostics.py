"""Bayesian variance ratios agree with independent row-variance calculations."""

import numpy as np
import pytest
import torch

from quivers.diagnostics.regression import bayesian_r2


@pytest.mark.parametrize("event", [(), (3,), (2, 3)])
@pytest.mark.parametrize("model", [False, True])
def test_r_squared_matches_numpy_and_retains_sample_and_event_axes(event, model):
    rng = np.random.default_rng(234)
    means = rng.normal(size=(2, 5, 8, *event))
    observed = rng.normal(size=(8, *event))
    variances = np.exp(rng.normal(size=(2, 5, 8, *event)))
    vp = np.var(means, axis=2, ddof=1)
    if model:
        expected = vp / (vp + np.mean(variances, axis=2))
        result = bayesian_r2(
            torch.tensor(means), residual_variance=torch.tensor(variances)
        )
    else:
        expected = vp / (vp + np.var(observed - means, axis=2, ddof=1))
        result = bayesian_r2(torch.tensor(means), observed=torch.tensor(observed))
    np.testing.assert_allclose(result.numpy(), expected, atol=2e-15, rtol=2e-15)
    assert result.shape == (2, 5, *event)


@pytest.mark.parametrize("factor", [1e-300, 1e300])
def test_residual_ratio_is_scale_invariant_without_overflow_or_underflow(factor):
    means = torch.tensor([[[1.0, 2.0, 4.0, 8.0]]], dtype=torch.double)
    response = torch.tensor([2.0, 2.0, 3.0, 9.0], dtype=torch.double)
    expected = bayesian_r2(means, observed=response)
    actual = bayesian_r2(means * factor, observed=response * factor)
    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


def test_model_variance_and_constant_mean_policies():
    means = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 3.0, 3.0]]], dtype=torch.double)
    variance = torch.tensor([1e300, 1e300, 1e300], dtype=torch.double)
    result = bayesian_r2(means, residual_variance=variance)
    assert result[0, 0] == pytest.approx(1e-300, rel=1e-14, abs=0)
    assert result[0, 1] == 0
    assert torch.isnan(bayesian_r2(means[:, 1:], residual_variance=0.0)).all()
    assert torch.isnan(bayesian_r2(means[:, 1:], observed=torch.full((3,), 3.0))).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_model_variance_requires_finite_nonnegative_values(bad):
    with pytest.raises(ValueError):
        bayesian_r2(torch.ones(1, 2, 4), residual_variance=bad)


@pytest.mark.parametrize("shape", [(4,), (2, 4), (1, 2, 1), (1, 0, 4)])
def test_sample_and_row_axes_are_required(shape):
    with pytest.raises(ValueError, match="chain, draw"):
        bayesian_r2(torch.ones(shape), residual_variance=1.0)


def test_invalid_or_ambiguous_response_information_is_refused():
    means = torch.zeros(1, 2, 4)
    for kwargs in [
        {},
        {"observed": torch.zeros(4), "residual_variance": 1.0},
        {"observed": torch.zeros(1, 4)},
        {"residual_variance": torch.ones(3)},
    ]:
        with pytest.raises(ValueError):
            bayesian_r2(means, **kwargs)
    with pytest.raises(TypeError):
        bayesian_r2(means.to(torch.complex128), residual_variance=1.0)
    with pytest.raises(ValueError, match="integer"):
        bayesian_r2(
            torch.full((1, 2, 4), 2**53 + 1, dtype=torch.int64), residual_variance=1.0
        )
