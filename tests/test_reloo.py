"""Exact-refit adapters preserve the target and delegate PSIS to ArviZ."""

import math
from didactic.fields import ValidationError
from unittest.mock import patch

import numpy as np
import pytest
from scipy.stats import norm
import torch

from quivers.diagnostics import PosteriorAnalysis, FoldPrediction, cross_validate
from quivers.diagnostics.cross_validation import FoldPlan


def _conjugate():
    data = torch.tensor([-0.6, 0.2, 0.8, 2.5], dtype=torch.float64)
    standardized = torch.tensor(norm.ppf((np.arange(8192) + 0.5) / 8192))
    variance = 1 / (len(data) + 1)
    theta = variance * data.sum() + math.sqrt(variance) * standardized
    scores = torch.distributions.Normal(theta[:, None], 1).log_prob(data)
    analysis = PosteriorAnalysis(
        samples={"theta": theta[None]},
        log_likelihood={"y": scores[None]},
        observed_data={"y": data},
        kind="variational",
        coords={"case": ["d", "a", "c", "b"]},
        group_dims={
            "log_likelihood": {"y": ["case"]},
            "observed_data": {"y": ["case"]},
        },
    )
    calls = []

    def refit(split):
        calls.append(split)
        train = data[list(split.train)]
        variance = 1 / (len(train) + 1)
        theta = variance * train.sum() + math.sqrt(variance) * standardized
        raw = torch.distributions.Normal(theta[:, None], 1).log_prob(
            data[list(split.test)]
        )
        return FoldPrediction(log_likelihood={"y": raw[None]})

    return data, analysis, refit, calls


def test_all_refits_match_independent_normal_predictive_oracle_and_shared_cv():
    data, analysis, refit, calls = _conjugate()
    before = torch.get_rng_state().clone()
    result = analysis.reloo(refit, k_threshold=-math.inf, seed=47)
    expected = []
    for row in range(len(data)):
        train = data[torch.arange(len(data)) != row]
        expected.append(
            norm.logpdf(
                float(data[row]), loc=float(train.sum() / 4), scale=math.sqrt(1.25)
            )
        )
    np.testing.assert_allclose(result.elpd_i, expected, atol=2e-4, rtol=0)
    np.testing.assert_array_equal(result.elpd_i.coords["case"], ["d", "a", "c", "b"])
    assert [split.test for split in calls] == [(0,), (1,), (2,), (3,)]
    assert [split.seed for split in calls] == [48, 49, 50, 51]
    torch.testing.assert_close(torch.get_rng_state(), before)
    exact = cross_validate(FoldPlan(assignments=tuple(range(4)), seed=47), refit)
    np.testing.assert_allclose(
        result.elpd_i, exact.log_predictive_density, atol=1e-12, rtol=0
    )
    assert float(result.elpd) == pytest.approx(float(exact.elpd))
    np.testing.assert_array_equal(result.pareto_k, 0)


def test_selected_bad_pareto_rows_reuse_existing_psis_and_leave_other_terms_unchanged():
    _, analysis, refit, calls = _conjugate()
    original = analysis.loo()
    original.pareto_k.values[:] = [0.1, 0.9, 0.2, 0.95]
    before = original.elpd_i.copy(deep=True)
    with patch.object(PosteriorAnalysis, "loo", return_value=original):
        result = analysis.reloo(refit, k_threshold=0.7)
    assert [split.test for split in calls] == [(1,), (3,)]
    np.testing.assert_array_equal(result.elpd_i.values[[0, 2]], before.values[[0, 2]])
    np.testing.assert_array_equal(original.elpd_i, before)
    with patch.object(PosteriorAnalysis, "loo", return_value=original):
        compact = analysis.reloo(refit, k_threshold=math.inf, pointwise=False)
    assert compact.elpd_i is None and compact.pareto_k is None
    assert len(calls) == 2


@pytest.mark.parametrize("threshold", [None, -math.inf])
def test_undefined_constant_likelihood_tail_fits_are_refitted(threshold):
    _, original, refit, calls = _conjugate()
    analysis = original.with_(
        log_likelihood={"y": original.log_likelihood["y"][:, :1].expand(1, 8192, 4)}
    )
    assert np.isnan(analysis.loo().pareto_k).all()
    criterion = analysis.reloo(refit, k_threshold=threshold)
    assert len(calls) == 4
    assert (criterion.pareto_k == 0).all()


@pytest.mark.parametrize(
    "kind", ["missing", "wrong_rows", "integer", "nan", "complex", "empty"]
)
def test_bad_refit_draws_fail_through_shared_raw_score_validator(kind):
    _, analysis, _, _ = _conjugate()
    values = {
        "missing": {"other": torch.zeros(1, 5, 1)},
        "wrong_rows": {"y": torch.zeros(1, 5, 2)},
        "integer": {"y": torch.zeros(1, 5, 1, dtype=torch.int64)},
        "complex": {"y": torch.zeros(1, 5, 1, dtype=torch.complex128)},
        "nan": {"y": torch.full((1, 5, 1), math.nan)},
        "empty": {"y": torch.empty(1, 0, 1)},
    }[kind]
    with pytest.raises((ValueError, TypeError)):
        analysis.reloo(
            lambda split: FoldPrediction(log_likelihood=values), k_threshold=-math.inf
        )


@pytest.mark.parametrize(
    "options",
    [{"k_threshold": math.nan}, {"k_threshold": True}, {"seed": -1}, {"pointwise": 1}],
)
def test_invalid_options_fail_before_refits(options):
    _, analysis, refit, calls = _conjugate()
    with pytest.raises((ValueError, TypeError, ValidationError)):
        analysis.reloo(refit, **options)
    assert not calls


@pytest.mark.parametrize("source", [False, True])
def test_existing_source_and_formula_recipes_feed_the_same_exact_refit_adapter(source):
    import pandas as pd
    from quivers.formulas import fit
    from quivers.modeling import Model, VariationalMethod

    values = [-0.6, 0.2, 0.8, 1.5]
    if source:
        specification = Model("""object Row : FinSet _
object Out : Real 1
program regression(y: Tensor[Real]([Row])) : Row -> Out
    sample intercept <- Normal(loc=0.0, scale=1.0)
    observe y : Row <- Normal(loc=intercept, scale=1.0)
    return intercept
export regression
""")
        result = specification.fit(
            {"y": torch.tensor(values)}, method=VariationalMethod(num_steps=2, seed=3)
        )
        callback = result.refit_callback(
            row_axis="Row",
            method_factory=lambda method, seed: VariationalMethod(
                num_steps=2, seed=seed
            ),
            num_samples=16,
        )
    else:
        result = fit(
            "y ~ 1",
            data=pd.DataFrame({"y": values}),
            fixed_parameters={"sigma": 1.0},
            method="svi",
            num_steps=2,
            seed=3,
        )
        callback = result.refit_callback(
            method_factory=lambda settings, seed: {**settings, "seed": seed},
            num_samples=16,
        )
    analysis = result.analysis(num_samples=32, seed=29)
    criterion = analysis.reloo(callback, k_threshold=-math.inf, seed=11)
    oracle = cross_validate(FoldPlan(assignments=(0, 1, 2, 3), seed=11), callback)
    np.testing.assert_allclose(
        criterion.elpd_i, oracle.log_predictive_density, atol=1e-12, rtol=0
    )
