"""Empirical contrast evidence agrees with exact counts and independent summaries."""

from __future__ import annotations

import copy
import math
import pickle
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from scipy.stats import median_abs_deviation
import torch

from quivers.diagnostics import HypothesisSummary, PosteriorAnalysis, hypothesis_summary
from quivers.formulas import fit
from quivers.modeling import Model, VariationalMethod


@pytest.mark.parametrize("robust", [False, True])
def test_matrix_uncertainty_matches_numpy_and_scipy(robust):
    rng = np.random.default_rng(14)
    raw = rng.lognormal(size=(3, 17, 2, 3)) - 0.7
    levels = (0.9, 0.025, 0.5, 0.9)
    result = hypothesis_summary(torch.tensor(raw), probs=levels, robust=robust)
    pooled = raw.reshape(-1, 2, 3)
    estimate = np.median(pooled, axis=0) if robust else pooled.mean(axis=0)
    dispersion = (
        median_abs_deviation(pooled, axis=0, scale="normal")
        if robust
        else pooled.std(axis=0, ddof=1)
    )
    np.testing.assert_allclose(result.estimate, estimate, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(result.est_error, dispersion, rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(
        result.quantiles, np.quantile(pooled, levels, axis=0), rtol=2e-14
    )
    assert result.probs == levels and result.robust is robust
    assert result.sample_shape == (3, 17) and result.kind == "unspecified"
    assert not result.estimate.requires_grad


def test_strict_regions_ties_and_asymmetric_prior_odds_are_distinct():
    result = hypothesis_summary(
        torch.tensor([[-2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]]),
        prior=torch.tensor([[-3.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    )
    assert result.below.probability.item() == 2 / 8
    assert result.equal_probability.item() == 2 / 8
    assert result.above.probability.item() == 4 / 8
    assert result.prior_equal_probability.item() == 4 / 8
    assert result.above.prior_probability.item() == 1 / 8
    assert result.above.log_odds.item() == 0
    assert result.above.log_bayes_factor.item() == pytest.approx(math.log(7))
    assert result.below.log_bayes_factor.item() == pytest.approx(math.log(5 / 9))
    assert not hasattr(result, "equal_log_bayes_factor")
    assert not hasattr(result, "post_prob")


def test_equality_of_joint_derived_draws_is_preserved_not_permuted():
    a = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    b = a.clone()
    result = hypothesis_summary(a - b, prior=a - b)
    assert result.equal_probability.item() == result.prior_equal_probability.item() == 1
    assert result.above.probability.item() == result.below.probability.item() == 0
    assert bool(torch.isnan(result.above.log_bayes_factor))
    assert torch.equal(a, b)


@pytest.mark.parametrize("posterior", [[-2.0, -1.0], [1.0, 2.0], [0.0, 0.0]])
@pytest.mark.parametrize(
    "prior", [None, [-1.0, 1.0], [1.0, 2.0], [-2.0, -1.0], [0.0, 0.0]]
)
def test_boundary_odds_have_no_pseudocount_or_undefined_prior_correction(
    posterior, prior
):
    result = hypothesis_summary(
        torch.tensor([posterior]),
        prior=torch.tensor([prior]) if prior is not None else None,
    )
    expected = sum(x > 0 for x in posterior) / len(posterior)
    expected_log = math.inf if expected == 1 else -math.inf
    assert result.above.log_odds.item() == expected_log
    if prior is None:
        assert result.above.log_bayes_factor is None
    elif prior == [-1.0, 1.0]:
        assert result.above.log_bayes_factor.item() == expected_log
    else:
        assert bool(torch.isnan(result.above.log_bayes_factor))


def test_null_uses_event_broadcasting_and_keeps_original_summary_units():
    values = torch.arange(48.0, dtype=torch.float64).reshape(2, 4, 2, 3)
    cutoff = torch.tensor([3.0, 9.0, 30.0], dtype=torch.float64)
    result = hypothesis_summary(values, null=cutoff, prior=values[:1, :2])
    np.testing.assert_allclose(result.estimate, values.numpy().mean(axis=(0, 1)))
    np.testing.assert_allclose(
        result.below.probability, (values.numpy() < cutoff.numpy()).mean(axis=(0, 1))
    )
    assert result.null.shape == (2, 3) and result.prior_sample_shape == (1, 2)
    assert result.equal_probability.shape == result.above.log_odds.shape == (2, 3)
    with pytest.raises(ValueError, match="event axes"):
        hypothesis_summary(values, null=torch.zeros(2, 4, 2, 3))


def test_named_axes_labels_and_provenance_reuse_retained_validation():
    values = torch.arange(16.0).reshape(2, 4, 2)
    coords = {
        "component": ["a[0]", 2],
        "chain": ["left", "right"],
        "draw": [10, 20, 30, 40],
    }
    result = hypothesis_summary(
        values,
        name="contrast[raw]",
        dims=["component"],
        coords=coords,
        kind="variational",
    )
    assert result.name == "contrast[raw]" and result.dimension_names == ("component",)
    assert result.coordinate_labels == (("a[0]", 2),)
    assert result.chain_labels == ("left", "right") and result.draw_labels == (
        10,
        20,
        30,
        40,
    )
    assert result.kind == "variational"
    analysis = PosteriorAnalysis(
        samples={"contrast[raw]": values}, log_likelihood={}, kind="mcmc"
    )
    covariance = analysis.covariance(
        dims={"contrast[raw]": ["component"]}, coords=coords
    )
    assert covariance.dimension_names[0] == result.dimension_names
    assert covariance.coordinate_labels == (("a[0]",), (2,))
    for restored in (copy.deepcopy(result), pickle.loads(pickle.dumps(result))):
        assert restored.dimension_names == result.dimension_names
        torch.testing.assert_close(restored.above.probability, result.above.probability)


@pytest.mark.parametrize("robust", [False, True])
def test_constant_single_and_extreme_draws_have_explicit_dispersion(robust):
    constant = hypothesis_summary(
        torch.full((2, 4), 1e308, dtype=torch.float64), robust=robust
    )
    assert constant.estimate.item() == 1e308 and constant.est_error.item() == 0
    single = hypothesis_summary(torch.tensor([[7.0]]), robust=robust)
    assert single.estimate.item() == 7
    assert (
        single.est_error.item() == 0 if robust else bool(torch.isnan(single.est_error))
    )
    opposite = hypothesis_summary(
        torch.tensor([[-1e308, 1e308]], dtype=torch.float64),
        probs=(0.25, 0.5, 0.75),
        robust=robust,
    )
    assert opposite.estimate.item() == 0
    np.testing.assert_allclose(opposite.quantiles, [-5e307, 0, 5e307], rtol=1e-15)
    assert math.isfinite(opposite.est_error.item())
    tiny = hypothesis_summary(
        torch.full((1, 4), 1e-320, dtype=torch.float64), robust=robust
    )
    assert tiny.estimate.item() == 1e-320 and tiny.est_error.item() == 0


def test_high_offset_spread_and_null_comparison_avoid_cancellation_or_subtraction():
    values = torch.tensor(
        [[1e308, math.nextafter(1e308, math.inf)]], dtype=torch.float64
    )
    result = hypothesis_summary(values, null=-1e308)
    expected_sd = (values[0, 1].item() - values[0, 0].item()) / math.sqrt(2)
    assert result.est_error.item() == pytest.approx(expected_sd, rel=2e-15)
    assert result.above.probability.item() == 1
    assert bool(torch.isfinite(result.estimate))


@pytest.mark.parametrize("robust", [False, True])
def test_unrepresentable_dispersion_is_refused(robust):
    with pytest.raises(ValueError, match="Float64 capacity"):
        hypothesis_summary(
            torch.tensor([[-1.7e308, 1.7e308]], dtype=torch.float64), robust=robust
        )


@pytest.mark.parametrize("dtype", [torch.int64, torch.uint64, torch.bool])
def test_original_count_and_boolean_carriers(dtype):
    raw = (
        [[0, 1, 1, 0]]
        if dtype == torch.bool
        else [[2**53 - 3, 2**53 - 2, 2**53 - 1, 2**53]]
    )
    result = hypothesis_summary(
        torch.tensor(raw, dtype=dtype),
        null=1 if dtype == torch.bool else 2**53 - 1,
        kind="prior",
    )
    assert result.estimate.dtype == torch.float64
    assert (
        result.equal_probability.item() == 0.5
        if dtype == torch.bool
        else result.equal_probability.item() == 0.25
    )
    if dtype != torch.bool:
        assert result.est_error.item() == pytest.approx(np.std(np.arange(4), ddof=1))


@pytest.mark.parametrize(
    "value,dtype",
    [(2**53 + 1, torch.int64), (-(2**53 + 1), torch.int64), (2**64 - 1, torch.uint64)],
)
def test_original_integer_capacity_refused_before_float_cast(value, dtype):
    with pytest.raises(ValueError, match=r"2\*\*53"):
        hypothesis_summary(torch.tensor([[0, value]], dtype=dtype))
    with pytest.raises(ValueError, match=r"2\*\*53"):
        hypothesis_summary(torch.ones(1, 2), null=value)


@pytest.mark.parametrize("shape", [(), (4,), (0, 4), (1, 0), (1, 4, 0)])
def test_missing_or_empty_sample_axes_refused(shape):
    with pytest.raises(ValueError, match="chain and draw"):
        hypothesis_summary(torch.empty(shape))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"probs": [True]},
        {"probs": [0.0]},
        {"probs": [1.0]},
        {"probs": [float("nan")]},
        {"null": float("inf")},
        {"null": torch.zeros(3)},
        {"robust": 1},
        {"kind": "fake"},
        {"name": ""},
        {"dims": ["chain"]},
        {"dims": "x"},
        {"coords": {"draw": [1]}},
        {"prior": torch.ones(1, 3, 2)},
    ],
)
def test_invalid_options_refuse_without_reinterpreting_axes(kwargs):
    with pytest.raises((ValueError, TypeError)):
        hypothesis_summary(torch.ones(2, 4), **kwargs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_missing_samples_are_not_dropped_or_imputed(value):
    with pytest.raises(ValueError, match="finite samples"):
        hypothesis_summary(torch.tensor([[0.0, value]]))
    with pytest.raises(ValueError, match="finite samples"):
        hypothesis_summary(torch.ones(1, 4), prior=torch.tensor([[value]]))


def test_complex_inputs_refused_and_empty_probability_axis_supported():
    with pytest.raises(TypeError, match="real sample tensor"):
        hypothesis_summary(torch.ones(1, 4, dtype=torch.complex128))
    result = hypothesis_summary(torch.ones(1, 4, 2), probs=())
    assert result.quantiles.shape == (0, 2)


@pytest.mark.parametrize("backend", ["source", "formula"])
def test_actual_fits_reuse_retained_draws_without_sampling_or_rng_changes(backend):
    if backend == "source":
        model = Model("""object Out : Real 1
program p[n:Nat](x:Tensor[Real]([n]),y:Tensor[Real]([n])) : n -> Out
    sample a <- Normal(loc=0.0,scale=1.0)
    observe y : n <- Normal(loc=a,scale=1.0)
    return a
export p
""")
        fitted = model.fit(
            {"x": torch.arange(4.0), "y": torch.tensor([0.2, 0.4, 0.1, 0.7])},
            method=VariationalMethod(num_steps=1, seed=2),
        )
    else:
        fitted = fit(
            "y~x",
            data=pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [0.2, 0.4, 0.1, 0.7]}),
            fixed_parameters={"sigma": 1.0},
            method="svi",
            num_steps=1,
            seed=2,
        )
    analysis = fitted.analysis(num_samples=8, seed=7)
    name, values = next(iter(analysis.samples.items()))
    state = torch.get_rng_state().clone()
    with patch.object(
        type(fitted.posterior), "rsample", side_effect=AssertionError("replayed guide")
    ):
        result = hypothesis_summary(values, name=name, kind=analysis.kind)
        retained = analysis.summary(probs=result.probs)[name]
    assert isinstance(result, HypothesisSummary)
    torch.testing.assert_close(result.estimate, retained.mean)
    torch.testing.assert_close(result.est_error, retained.sd)
    torch.testing.assert_close(result.quantiles, retained.quantiles)
    assert torch.equal(state, torch.get_rng_state())


def test_available_accelerator_preserves_device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA Float64 device unavailable")
    values = torch.tensor([[0.0, 1.0, 2.0]], device="cuda", requires_grad=True)
    result = hypothesis_summary(values, prior=values.detach())
    assert result.estimate.device == values.device and not result.estimate.requires_grad
    assert result.above.log_bayes_factor.device == values.device
    with pytest.raises(ValueError, match="same device"):
        hypothesis_summary(values, prior=values.cpu())
