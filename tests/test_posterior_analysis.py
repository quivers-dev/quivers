"""Shared fit analysis matches independent empirical and likelihood oracles."""

import math
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import arviz as az
import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import logsumexp

from quivers.diagnostics import PosteriorAnalysis, PosteriorSummary, WAICResult, waic
from quivers.formulas import fit
from quivers.inference import MCMCResult
from quivers.modeling import Model, ModelFit, VariationalMethod

SOURCE = """object Row : FinSet _
object Out : Real 1
program p(x : Tensor[Real]([Row]), y : Tensor[Real]([Row])) : Row -> Out
    sample intercept <- Normal(loc=0.0, scale=1.0)
    sample beta_x <- Normal(loc=0.0, scale=1.0)
    let mu = intercept + beta_x * x
    observe y : Row <- Normal(loc=mu, scale=0.7)
    return intercept
export p
"""


def _frame():
    return pd.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, 2.0],
            "y": [0.2, 0.7, -0.3, 0.5],
            "w": [0.0, 0.5, 2.0, 1.0],
        }
    )


def _posterior():
    generator = torch.Generator().manual_seed(41)
    return MCMCResult(
        samples={
            "intercept": 0.3 + 0.1 * torch.randn(2, 32, generator=generator),
            "beta_x": -0.2 + 0.1 * torch.randn(2, 32, generator=generator),
        },
        log_densities=torch.zeros(2, 32),
        acceptance_rates=torch.ones(2),
        divergence_counts=torch.zeros(2, dtype=torch.long),
        num_warmup=0,
        num_samples=32,
        sample_stats={"diverged": torch.zeros(2, 32), "tree_depth": torch.ones(2, 32)},
    )


@pytest.mark.parametrize("source", [False, True])
def test_source_and_formula_analysis_retain_raw_scores_coordinates_and_real_diagnostics(
    source,
):
    frame = _frame()
    posterior = _posterior()
    if source:
        bound = Model(SOURCE).bind(
            {
                key: torch.tensor(frame[key].to_numpy(), dtype=torch.float32)
                for key in ("x", "y")
            }
        )
        fitted = ModelFit(model=bound, posterior=posterior)
    else:
        fitted = fit(
            "y ~ x",
            data=frame,
            weights="w",
            fixed_parameters={"sigma": 0.7},
            method="svi",
            num_steps=1,
        ).with_(posterior=posterior)
    state = torch.get_rng_state().clone()
    result = fitted.analysis(num_samples=16, seed=7)
    assert torch.equal(state, torch.get_rng_state())
    assert result.kind == "mcmc"
    assert result.samples["intercept"].shape == (2, 16)
    location = posterior.samples["intercept"][:, :16, None] + posterior.samples[
        "beta_x"
    ][:, :16, None] * torch.tensor(frame.x.to_numpy(), dtype=torch.float32)
    expected = torch.distributions.Normal(location, 0.7).log_prob(
        torch.tensor(frame.y.to_numpy(), dtype=torch.float32)
    )
    torch.testing.assert_close(result.log_likelihood["y"], expected)
    torch.testing.assert_close(
        fitted.log_likelihood(num_samples=16, seed=7)["y"], expected
    )
    # Zero-weight training rows still contribute their raw conditional density.
    assert not result.log_likelihood["y"][..., 0].eq(0).any()
    summaries = result.summary(probs=(0.1, 0.5, 0.9))
    assert isinstance(summaries["intercept"], PosteriorSummary)
    assert summaries["intercept"].convergence is not None
    values = posterior.samples["intercept"][:, :16].double().flatten()
    torch.testing.assert_close(summaries["intercept"].mean, values.mean())
    torch.testing.assert_close(summaries["intercept"].sd, values.std())
    torch.testing.assert_close(
        summaries["intercept"].quantiles,
        values.quantile(torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)),
    )
    tree = result.to_datatree()
    assert tree["posterior"].dataset["intercept"].shape == (2, 16)
    np.testing.assert_allclose(tree["log_likelihood"].dataset["y"], expected)
    assert tree["sample_stats"].dataset["diverging"].dtype == np.dtype("bool")
    assert tree["sample_stats"].dataset["tree_depth"].dtype == np.dtype("int64")
    assert tree.attrs["quivers_posterior_kind"] == "mcmc"
    criterion = result.loo()
    oracle = az.loo(tree, pointwise=True, var_name="y")
    assert criterion.elpd == pytest.approx(oracle.elpd)
    np.testing.assert_allclose(criterion.pareto_k, oracle.pareto_k)
    assert result.waic().num_samples == 32


@pytest.mark.parametrize("source", [False, True])
def test_variational_analysis_draws_once_and_never_invents_chain_diagnostics(source):
    frame = _frame()
    if source:
        bound = Model(SOURCE).bind(
            {
                key: torch.tensor(frame[key].to_numpy(), dtype=torch.float32)
                for key in ("x", "y")
            }
        )
        fitted = bound.infer(VariationalMethod(num_steps=1, seed=2))
    else:
        fitted = fit(
            "y ~ x",
            data=frame,
            fixed_parameters={"sigma": 0.7},
            method="svi",
            num_steps=1,
            seed=2,
        )
    state = torch.get_rng_state().clone()
    with patch.object(
        type(fitted.posterior),
        "rsample",
        autospec=True,
        side_effect=type(fitted.posterior).rsample,
    ) as sample:
        result = fitted.analysis(num_samples=32, seed=7)
        assert sample.call_count == 32
        assert result.kind == "variational" and not result.sample_stats
        assert all(s.convergence is None for s in result.summary().values())
        result.waic()
        tree = result.to_datatree()
        criterion = result.loo()
        oracle = az.loo(tree, var_name="y", pointwise=True, reff=1.0)
        assert criterion.elpd == pytest.approx(oracle.elpd)
        assert sample.call_count == 32
    assert torch.equal(state, torch.get_rng_state())
    repeated = fitted.analysis(num_samples=32, seed=7)
    for name, values in result.samples.items():
        torch.testing.assert_close(values, repeated.samples[name], rtol=0, atol=0)


@pytest.mark.parametrize("shift", [0.0, -10000.0])
def test_waic_matches_independent_loo_sample_variance_formula(shift):
    generator = np.random.default_rng(41)
    scores = generator.normal(-2, 0.8, size=(3, 20, 2, 4)) + shift
    result = waic(torch.tensor(scores, dtype=torch.float64))
    pooled = scores.reshape(60, 2, 4)
    variances = pooled.var(axis=0, ddof=1)
    pointwise = logsumexp(pooled, axis=0) - math.log(60) - variances
    assert isinstance(result, WAICResult)
    np.testing.assert_allclose(result.elpd_i, pointwise, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.variance_i, variances, rtol=2e-12, atol=1e-12)
    assert result.elpd_waic == pytest.approx(pointwise.sum())
    assert result.p_waic == pytest.approx(variances.sum())
    assert result.waic == pytest.approx(-2 * pointwise.sum())
    assert result.se == pytest.approx(
        math.sqrt(pointwise.size * pointwise.flatten().var(ddof=1))
    )
    assert result.warning and result.num_observations == 8 and result.num_samples == 60


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2, 0),
        torch.full((1, 2, 1), -torch.inf),
        torch.full((1, 2, 1), torch.nan),
    ],
)
def test_waic_refuses_invalid_empirical_inputs(value):
    with pytest.raises(ValueError):
        waic(value)


def test_waic_single_unit_has_explicit_undefined_standard_error():
    result = waic(torch.tensor([[[-1.0], [-1.0]]]))
    assert math.isnan(result.se) and not result.warning
    assert result.elpd_waic == pytest.approx(-1.0) and result.p_waic == 0


@pytest.mark.parametrize(
    "value",
    [
        torch.zeros(1, 2, 1, dtype=torch.long),
        torch.zeros(1, 2, 1, dtype=torch.complex64),
        None,
    ],
)
def test_waic_requires_real_floating_scores(value):
    with pytest.raises(TypeError):
        waic(value)


@pytest.mark.parametrize("num_samples", [0, -1, True, 1.5])
@pytest.mark.parametrize("source", [False, True])
def test_analysis_rejects_invalid_draw_requests_without_consuming_rng(
    num_samples, source
):
    frame = _frame()
    if source:
        fitted = ModelFit(
            model=Model(SOURCE).bind(
                {
                    k: torch.tensor(frame[k].to_numpy(), dtype=torch.float32)
                    for k in ("x", "y")
                }
            ),
            posterior=_posterior(),
        )
    else:
        fitted = fit("y ~ x", data=frame, method="svi", num_steps=1)
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="positive integer"):
        fitted.analysis(num_samples=num_samples)
    assert torch.equal(state, torch.get_rng_state())


def test_multivariate_analysis_requires_explicit_likelihood_selection_and_shared_axes():
    result = PosteriorAnalysis(
        samples={"x": torch.zeros(2, 5)},
        log_likelihood={
            "a": torch.full((2, 5, 3), -1.0),
            "b": torch.full((2, 5, 4), -2.0),
        },
    )
    with pytest.raises(ValueError, match="var_name"):
        result.waic()
    assert result.waic(var_name="a").num_observations == 3
    with pytest.raises(ValueError, match="unknown"):
        result.waic(var_name="missing")
    with pytest.raises(ValueError, match="chain/draw"):
        result.with_(log_likelihood={"a": torch.zeros(2, 4, 3)}).to_datatree()
    with pytest.raises(ValueError, match="pseudo-chain"):
        result.with_(kind="variational").summary()


def test_comparison_reuses_retained_scores_and_matches_arviz_stacking():
    from quivers.diagnostics import compare

    generator = torch.Generator().manual_seed(19)
    samples = {"theta": torch.randn(2, 80, generator=generator)}
    observed = {"y": torch.tensor([0.0, 1.0, 0.0, 1.0])}
    scores = torch.randn(2, 80, 4, generator=generator) * 0.1 - 1.0
    first = PosteriorAnalysis(
        samples=samples, log_likelihood={"y": scores}, observed_data=observed
    )
    second = PosteriorAnalysis(
        samples=samples, log_likelihood={"y": scores - 0.1}, observed_data=observed
    )
    result = compare({"first": first, "second": second}, round_to="none")
    oracle = az.compare({"first": first.loo(), "second": second.loo()}, round_to="none")
    for column in ("elpd", "p", "elpd_diff", "weight", "se", "dse"):
        np.testing.assert_allclose(
            result[column], oracle[column], rtol=2e-12, atol=2e-12
        )
    assert float(result.loc["first", "weight"]) > 0.99
    with pytest.raises(ValueError, match="same observed responses"):
        compare(
            {"first": first, "second": second.with_(observed_data={"y": torch.ones(4)})}
        )
    with pytest.raises(ValueError, match="observation units"):
        compare(
            {
                "first": first,
                "second": second.with_(log_likelihood={"y": scores[:, :, :3]}),
            }
        )
    with pytest.raises(ValueError, match="response data"):
        compare({"first": first, "second": second.with_(observed_data={})})


def test_empty_observation_units_still_allow_posterior_summaries():
    result = PosteriorAnalysis(
        samples={"theta": torch.arange(10.0).reshape(2, 5)},
        log_likelihood={"y": torch.empty(2, 5, 0)},
        observed_data={"y": torch.empty(0)},
    )
    assert result.summary()["theta"].mean.item() == 4.5
    assert result.to_datatree()["log_likelihood"].dataset["y"].shape == (2, 5, 0)
    with pytest.raises(ValueError, match="nonempty"):
        result.waic()


@pytest.mark.parametrize("kind", ["mcmc", "variational"])
def test_analysis_datatree_labels_event_axes_and_preserves_tensor_values(kind):
    samples = torch.arange(24.0).reshape(2, 4, 3).requires_grad_()
    scores = torch.arange(40.0).reshape(2, 4, 5) / -10
    observed = torch.arange(5)
    statistics = (
        {
            "lp": torch.arange(8.0).reshape(2, 4),
            "diverged": torch.tensor([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            "tree_depth": torch.full((2, 4), 3.0),
            "num_steps": torch.full((2, 4), 7.0),
            "accept_prob": torch.full((2, 4), 0.75),
            "component__num_steps": torch.full((2, 4), 11.0),
        }
        if kind == "mcmc"
        else {}
    )
    if kind == "variational":
        samples = samples[:1]
        scores = scores[:1]
    result = PosteriorAnalysis(
        samples={"beta": samples},
        log_likelihood={"y": scores},
        observed_data={"y": observed},
        sample_stats=statistics,
        kind=kind,
    )
    tree = result.to_datatree(
        coords={"coefficient": ["a", "b", "c"], "row": list("vwxyz")},
        dims={"beta": ["coefficient"], "y": ["row"]},
    )
    assert tree["posterior"]["beta"].dims == ("chain", "draw", "coefficient")
    assert tree["log_likelihood"]["y"].dims == ("chain", "draw", "row")
    assert tree["observed_data"]["y"].dims == ("row",)
    assert tree["posterior"]["beta"].coefficient.values.tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(
        tree["posterior"]["beta"], np.arange(24).reshape(2, 4, 3)[: samples.shape[0]]
    )
    np.testing.assert_allclose(
        tree["log_likelihood"]["y"],
        np.arange(40).reshape(2, 4, 5)[: samples.shape[0]] / -10,
        rtol=1e-7,
    )
    np.testing.assert_array_equal(tree["observed_data"]["y"], np.arange(5))
    assert samples.requires_grad and result.samples["beta"] is samples
    assert tree.attrs["quivers_posterior_kind"] == kind
    if kind == "mcmc":
        assert tree["sample_stats"]["diverging"].dtype == np.dtype("bool")
        assert tree["sample_stats"]["tree_depth"].dtype == np.dtype("int64")
        assert tree["sample_stats"]["n_steps"].dtype == np.dtype("int64")
        assert tree["sample_stats"]["component__num_steps"].dtype == np.dtype("float32")
        np.testing.assert_array_equal(tree["sample_stats"]["acceptance_rate"], 0.75)
        assert statistics["diverged"].dtype == torch.float32
    else:
        assert "sample_stats" not in tree.children


def test_torch_only_analysis_does_not_import_optional_arviz_dependencies():
    # Fresh interpreter: collection has already imported ArviZ in this process.
    script = r"""
import importlib.abc
import sys
class OptionalDependencyGuard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'arviz', 'xarray'}:
            raise ModuleNotFoundError('optional diagnostics dependency blocked')
sys.meta_path.insert(0, OptionalDependencyGuard())
import torch
from quivers.diagnostics import PosteriorAnalysis
analysis = PosteriorAnalysis(
    samples={'theta': torch.arange(8.).reshape(1,8)},
    log_likelihood={'y': torch.zeros(1,8,2)},
    observed_data={'y': torch.zeros(2)},
    sample_stats={}, kind='variational',
)
assert analysis.summary()['theta'].mean.item() == 3.5
assert analysis.waic().num_observations == 2
assert 'arviz' not in sys.modules and 'xarray' not in sys.modules
try:
    analysis.to_datatree()
except ModuleNotFoundError as exc:
    assert 'optional diagnostics dependency blocked' in str(exc)
else:
    raise AssertionError('conversion must require optional diagnostics packages')
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(Path(__file__).parents[1] / "src"), environment.get("PYTHONPATH", ""))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        text=True,
        capture_output=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
