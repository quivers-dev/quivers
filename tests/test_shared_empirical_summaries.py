"""Retained and recorded-chain summaries share exact carriers and scaled reductions."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import tstd
import torch

from quivers.diagnostics import PosteriorAnalysis, hypothesis_summary, summary
from quivers.diagnostics._numeric import real_samples


@pytest.mark.parametrize("kind", ["mcmc", "variational", "prior"])
def test_all_retained_kinds_match_independent_numpy_scipy_and_contrast(kind):
    values = np.random.default_rng(17).normal(
        size=(2 if kind == "mcmc" else 1, 20, 2, 3)
    )
    tensor = torch.tensor(values)
    levels = (0.1, 0.5, 0.9)
    result = PosteriorAnalysis(
        samples={"x": tensor}, log_likelihood={}, kind=kind
    ).summary(probs=levels)["x"]
    contrast = hypothesis_summary(tensor, probs=levels)
    pooled = values.reshape(-1, 2, 3)
    np.testing.assert_allclose(result.mean, np.mean(pooled, axis=0), atol=1e-15)
    np.testing.assert_allclose(result.sd, tstd(pooled, axis=0), atol=1e-15)
    np.testing.assert_allclose(
        result.quantiles, np.quantile(pooled, levels, axis=0), atol=1e-15
    )
    torch.testing.assert_close(result.mean, contrast.estimate, atol=0, rtol=0)
    torch.testing.assert_close(result.sd, contrast.est_error, atol=0, rtol=0)
    torch.testing.assert_close(result.quantiles, contrast.quantiles, atol=0, rtol=0)
    assert (result.convergence is not None) is (kind == "mcmc")


@pytest.mark.parametrize("kind", ["mcmc", "variational", "prior"])
@pytest.mark.parametrize("value", [1e308, -1e308, 1e-320])
def test_finite_constant_extremes_keep_exact_location_zero_dispersion(kind, value):
    tensor = torch.full((2 if kind == "mcmc" else 1, 8), value, dtype=torch.float64)
    result = PosteriorAnalysis(
        samples={"x": tensor}, log_likelihood={}, kind=kind
    ).summary()["x"]
    assert result.mean.item() == value
    assert result.sd.item() == 0
    assert bool((result.quantiles == value).all())
    if result.convergence is not None:
        assert bool(torch.isnan(result.convergence.ess_bulk))
        assert bool(torch.isnan(result.convergence.r_hat))


def test_opposite_extremes_use_scaled_mean_sd_and_one_quantile_kernel():
    values = torch.tensor([[-1e308, 1e308] * 4] * 2, dtype=torch.float64)
    result = summary({"x": values}, probs=(0.1, 0.5, 0.9))["x"]
    assert result.mean.item() == 0
    oracle = tstd(np.array([-1.0, 1.0] * 8)) * 1e308
    assert result.sd.item() == pytest.approx(oracle, rel=2e-15)
    np.testing.assert_allclose(result.quantiles, [-1e308, 0, 1e308], rtol=1e-15)


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64])
def test_adjacent_exact_integer_draws_retain_small_spread_at_capacity(dtype):
    values = torch.tensor([[2**53 - 3, 2**53 - 2, 2**53 - 1, 2**53]], dtype=dtype)
    result = PosteriorAnalysis(
        samples={"count": values}, log_likelihood={}, kind="prior"
    ).summary(probs=(0.5,))["count"]
    assert result.sd.item() == pytest.approx(tstd(np.arange(4)), rel=2e-15)
    assert result.mean.item() == float(2**53 - 2)
    assert result.quantiles.item() == float(2**53 - 2)
    assert result.convergence is None
    direct = summary({"count": values})["count"]
    torch.testing.assert_close(direct.sd, result.sd)


@pytest.mark.parametrize(
    "bad,dtype",
    [(2**53 + 1, torch.int64), (-(2**53 + 1), torch.int64), (2**64 - 1, torch.uint64)],
)
def test_original_integer_overflow_is_not_silently_rounded_in_any_summary(bad, dtype):
    values = torch.tensor([[0, bad]], dtype=dtype)
    for function in (
        lambda: PosteriorAnalysis(
            samples={"x": values}, log_likelihood={}, kind="prior"
        ).summary(),
        lambda: summary({"x": values}),
        lambda: real_samples(values, label="permissive", finite=False),
    ):
        with pytest.raises(ValueError, match=r"2\*\*53"):
            function()


@pytest.mark.parametrize("kind", ["mcmc", "variational", "prior"])
def test_nonfinite_coordinates_remain_nan_without_deleting_draws_or_sites(kind):
    rng = np.random.default_rng(87)
    values = torch.tensor(rng.normal(size=(2 if kind == "mcmc" else 1, 12, 4)))
    clean = values[..., 0].numpy().reshape(-1).copy()
    for j, bad in enumerate([torch.nan, torch.inf, -torch.inf], start=1):
        values[0, 1, j] = bad
    result = PosteriorAnalysis(
        samples={"x": values}, log_likelihood={}, kind=kind
    ).summary(probs=(0.1, 0.5, 0.9))["x"]
    assert result.mean[0].item() == pytest.approx(np.mean(clean), abs=2e-15)
    assert result.sd[0].item() == pytest.approx(tstd(clean), abs=2e-15)
    np.testing.assert_allclose(
        result.quantiles[:, 0], np.quantile(clean, [0.1, 0.5, 0.9]), atol=2e-15
    )
    assert bool(torch.isnan(result.mean[1:]).all())
    assert bool(torch.isnan(result.sd[1:]).all())
    assert bool(torch.isnan(result.quantiles[:, 1:]).all())
    if kind == "mcmc":
        for name in ("r_hat", "ess_bulk", "ess_tail", "mcse_mean", "mcse_sd"):
            diagnostics = getattr(result.convergence, name)
            assert bool(torch.isnan(diagnostics[1:]).all())
            assert math.isfinite(diagnostics[0].item())
        assert bool(torch.isnan(result.convergence.mcse_quantiles[:, 1:]).all())
    else:
        assert result.convergence is None
    # Strict numerical diagnostics keep their original refusal policy.
    with pytest.raises(ValueError, match="finite samples"):
        real_samples(values, label="strict")
    with pytest.raises(ValueError, match="finite samples"):
        hypothesis_summary(values)


@pytest.mark.parametrize("kind", ["mcmc", "variational", "prior"])
def test_single_draw_sample_sd_remains_undefined_without_reducing_event_axes(kind):
    values = torch.tensor([[[7.0, 8.0]]])
    result = PosteriorAnalysis(
        samples={"x": values}, log_likelihood={}, kind=kind
    ).summary(probs=())["x"]
    np.testing.assert_array_equal(result.mean, [7.0, 8.0])
    assert bool(torch.isnan(result.sd).all())
    assert result.quantiles.shape == (0, 2)


def test_boolean_prior_summary_uses_numeric_carrier_without_fake_diagnostics():
    values = torch.tensor([[True, False, True, False]])
    result = PosteriorAnalysis(
        samples={"x": values}, log_likelihood={}, kind="prior"
    ).summary()["x"]
    assert result.mean.item() == 0.5
    assert result.sd.item() == pytest.approx(tstd(np.array([1.0, 0.0, 1.0, 0.0])))
    assert result.convergence is None
