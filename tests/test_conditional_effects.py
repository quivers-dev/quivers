"""Explicit conditioning and empirical averaging retain joint uncertainty."""

from collections.abc import Sequence

import pandas as pd
import pytest
import torch
from didactic.fields import ValidationError

from quivers.diagnostics import PredictionGrid, prediction_grid
from quivers.formulas import fit
from quivers.inference import PriorDraws


def _fit(family="gaussian", formula="y~x+z"):
    data = pd.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [-2.0, 2.0, -2.0, 2.0],
            "g": ["a", "b", "a", "b"],
        }
    )
    fitted = fit(
        formula,
        data=data,
        family=family,
        fixed_parameters={"sigma": 1.0} if family == "gaussian" else None,
        method="svi",
        num_steps=1,
        seed=11,
    )
    samples = dict(fitted.prior_draws(num_samples=4, seed=12).samples)
    if "intercept" in samples:
        samples["intercept"] = torch.tensor([[-0.5, 0.0, 0.5, 1.0]]).reshape_as(
            samples["intercept"]
        )
    if "beta_x" in samples:
        samples["beta_x"] = torch.tensor([[0.2, 0.4, 0.6, 0.8]]).reshape_as(
            samples["beta_x"]
        )
    if "beta_z" in samples:
        samples["beta_z"] = torch.ones_like(samples["beta_z"])
    return fitted.with_(posterior=PriorDraws(samples=samples, num_samples=4))


def test_conditioning_keeps_explicit_rows_and_paired_posterior_coordinates():
    fitted = _fit()
    frame = pd.DataFrame({"z": [-2.0, 2.0]})
    grid = prediction_grid({"x": [0.0, 1.0]}, conditions=frame)
    frame.loc[0, "z"] = 999.0
    result = fitted.conditional_effects(grid)
    intercept = fitted.posterior.samples["intercept"]
    slope = fitted.posterior.samples["beta_x"]
    expected = (
        intercept[:, :, None, None]
        + slope[:, :, None, None] * torch.tensor([0.0, 1.0])[None, None, :, None]
        + torch.tensor([-2.0, 2.0])
    )
    torch.testing.assert_close(result.draws, expected)
    assert result.grid.points == ((0.0,), (1.0,))
    assert result.kind == "prior" and result.method == "epred"
    # Shared draws make the point contrast exactly the slope, not two
    # independently sampled response means with destroyed covariance.
    torch.testing.assert_close(
        result.draws[:, :, 1, 0] - result.draws[:, :, 0, 0], slope
    )
    summary = result.summary()
    torch.testing.assert_close(summary.mean, expected.double().mean((0, 1)))
    covariance = result.analysis().covariance().covariance
    torch.testing.assert_close(
        covariance, torch.cov(expected.flatten(0, 1).flatten(1).double().T)
    )
    assert summary.convergence is None


def test_empirical_marginalization_precedes_summaries_and_differs_from_mean_conditioning():
    fitted = _fit("bernoulli")
    empirical = prediction_grid(
        {"x": [0.0, 1.0]},
        conditions=pd.DataFrame({"z": [-2.0, 2.0]}),
        marginalize=True,
        weights=[1.0, 3.0],
    )
    conditional = prediction_grid(
        {"x": [0.0, 1.0]}, conditions=pd.DataFrame({"z": [1.0]})
    )
    result = fitted.conditional_effects(empirical)
    mean_conditioned = fitted.conditional_effects(conditional)
    a = fitted.posterior.samples["intercept"][:, :, None, None]
    b = fitted.posterior.samples["beta_x"][:, :, None, None]
    row_means = torch.sigmoid(
        a
        + b * torch.tensor([0.0, 1.0])[None, None, :, None]
        + torch.tensor([-2.0, 2.0])
    )
    expected = (row_means.double() * torch.tensor([0.25, 0.75])).sum(-1)
    torch.testing.assert_close(result.draws, expected)
    assert result.draws.shape == (1, 4, 2)
    assert not torch.allclose(result.draws, mean_conditioned.draws.squeeze(-1).double())
    torch.testing.assert_close(
        result.summary(probs=[0.5]).quantiles[0],
        torch.quantile(expected[0], 0.5, dim=0),
    )


@pytest.mark.parametrize("method", ["epred", "linpred", "predict"])
def test_formula_adapter_makes_one_joint_prediction_call(method, monkeypatch):
    fitted = _fit()
    grid = prediction_grid({"x": [0.0, 1.0]}, conditions=pd.DataFrame({"z": [0.0]}))
    name = "posterior_" + method
    actual = getattr(type(fitted), name)
    calls = []

    def counted(self, **options):
        calls.append(options)
        return actual(self, **options)

    monkeypatch.setattr(type(fitted), name, counted)
    rng = torch.random.get_rng_state().clone()
    result = fitted.conditional_effects(grid, method=method, re_formula="~0", seed=42)
    assert len(calls) == 1
    assert len(calls[0]["new_data"]) == 2 and calls[0]["seed"] == 42
    assert calls[0]["re_formula"] == "~0"
    assert result.draws.shape == (1, 4, 2, 1)
    assert torch.equal(rng, torch.random.get_rng_state())


def test_known_group_effects_and_exclusion_reuse_shared_predictor_policy():
    fitted = _fit(formula="y~x+(1|g)")
    samples = dict(fitted.posterior.samples)
    samples["sigma_g_Intercept"] = torch.ones_like(samples["sigma_g_Intercept"])
    samples["z_g_Intercept"] = torch.tensor(
        [[[1.0, -1.0], [2.0, -2.0], [3.0, -3.0], [4.0, -4.0]]]
    )
    fitted = fitted.with_(posterior=PriorDraws(samples=samples, num_samples=4))
    grid = prediction_grid(
        {"x": [0.0, 1.0]}, conditions=pd.DataFrame({"g": ["a", "b"]})
    )
    included = fitted.conditional_effects(grid)
    excluded = fitted.conditional_effects(grid, re_formula="~0")
    expected = samples["z_g_Intercept"][:, :, None, :].expand(1, 4, 2, 2)
    torch.testing.assert_close(included.draws - excluded.draws, expected)
    assert (included.draws[:, :, 1, :] - included.draws[:, :, 0, :]).shape == (1, 4, 2)


def test_new_group_draw_is_shared_across_grid_points_and_strict_by_default():
    fitted = _fit(formula="y~x+(1|g)")
    grid = prediction_grid({"x": [0.0, 1.0]}, conditions=pd.DataFrame({"g": ["fresh"]}))
    with pytest.raises(ValueError, match="level|group"):
        fitted.conditional_effects(grid)
    result = fitted.conditional_effects(
        grid, allow_new_levels=True, sample_new_levels="gaussian", seed=4
    )
    torch.testing.assert_close(
        result.draws[:, :, 1, 0] - result.draws[:, :, 0, 0],
        fitted.posterior.samples["beta_x"],
    )


def test_categorical_predictions_keep_component_probabilities_and_labels():
    fitted = _fit("categorical")
    grid = prediction_grid({"x": [0.0, 1.0]}, conditions=pd.DataFrame({"z": [0.0]}))
    result = fitted.conditional_effects(grid)
    assert result.draws.shape == (1, 4, 2, 1, 2)
    assert result.labels == (0, 1)
    torch.testing.assert_close(result.draws.sum(-1), torch.ones(1, 4, 2, 1))


def test_allocation_bound_is_checked_before_iterating_cartesian_axes():
    class Huge(Sequence):
        def __len__(self):
            return 10**8

        def __getitem__(self, index):
            raise AssertionError("oversized axes were materialized")

    with pytest.raises(ValueError, match="exceeding max_rows"):
        prediction_grid({"x": Huge(), "z": Huge()}, conditions=pd.DataFrame(index=[0]))


def test_multiple_axes_order_overrides_and_extreme_reference_weights():
    grid = prediction_grid(
        {"x": [0, 1], "z": [2, 3]},
        conditions=pd.DataFrame({"x": [float("nan"), float("nan")], "g": ["a", "b"]}),
        marginalize=True,
        weights=[1e308, 1e308],
    )
    assert grid.points == ((0, 2), (0, 3), (1, 2), (1, 3))
    assert grid.weights == (0.5, 0.5)
    assert grid.to_frame().to_dict("list") == {
        "x": [0, 0, 0, 0, 1, 1, 1, 1],
        "g": ["a", "b"] * 4,
        "z": [2, 2, 3, 3, 2, 2, 3, 3],
    }
    with pytest.raises(TypeError, match="sequence"):
        prediction_grid({"g": "abc"}, conditions=pd.DataFrame(index=[0]))


def test_mcmc_chains_and_variational_provenance_are_preserved():
    frame = pd.DataFrame({"y": [0.0, 1.0, 0.0, 1.0], "x": [0.0, 1.0, 2.0, 3.0]})
    grid = prediction_grid({"x": [0.0, 1.0]}, conditions=pd.DataFrame(index=[0]))
    fitted = fit(
        "y~x",
        data=frame,
        fixed_parameters={"sigma": 1.0},
        method="hmc",
        num_warmup=1,
        num_samples=3,
        num_chains=2,
        seed=13,
    )
    result = fitted.conditional_effects(grid, num_samples=2)
    assert result.draws.shape == (2, 2, 2, 1) and result.kind == "mcmc"
    fitted = fit(
        "y~x",
        data=frame,
        fixed_parameters={"sigma": 1.0},
        method="svi",
        num_steps=1,
        seed=13,
    )
    result = fitted.conditional_effects(grid, num_samples=3)
    assert result.draws.shape == (1, 3, 2, 1) and result.kind == "variational"
    assert result.summary().convergence is None


@pytest.mark.parametrize(
    "weights", [[0, 0], [True, 1], [-1, 2], [float("inf"), 1], [1]]
)
def test_bad_empirical_weights_are_refused(weights):
    with pytest.raises(ValueError, match="reference weights"):
        prediction_grid(
            {"x": [0.0]},
            conditions=pd.DataFrame({"z": [0.0, 1.0]}),
            marginalize=True,
            weights=weights,
        )


def test_grid_rejects_bad_prediction_layout_and_duplicate_columns():
    grid = prediction_grid({"x": [0.0, 1.0]}, conditions=pd.DataFrame(index=[0]))
    with pytest.raises(ValueError, match="matching chain"):
        grid.result(torch.zeros(1, 4, 3), kind="prior")
    with pytest.raises(ValueError, match="unique column"):
        prediction_grid(
            {"x": [0.0]}, conditions=pd.DataFrame([[0, 1]], columns=["z", "z"])
        )
    with pytest.raises(ValidationError, match="inconsistent layout"):
        PredictionGrid(
            columns=("x",),
            rows=((0.0,),),
            axes=("x",),
            points=((0.0,),),
            num_conditions=2,
        )


@pytest.mark.parametrize(
    "options",
    [
        {"new_data": pd.DataFrame({"x": [0.0]})},
        {"method": "bad"},
        {"method": "predict", "transform": True},
        {"method": "linpred", "mc_samples": 4},
    ],
)
def test_incompatible_options_fail_before_prediction(options, monkeypatch):
    fitted = _fit()

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid grid options reached prediction")

    monkeypatch.setattr(type(fitted), "posterior_epred", forbidden)
    grid = prediction_grid({"x": [0.0]}, conditions=pd.DataFrame({"z": [0.0]}))
    with pytest.raises(ValueError):
        fitted.conditional_effects(grid, **options)
