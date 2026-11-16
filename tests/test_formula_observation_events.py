"""Recorded likelihood events and sampling laws share the formula pipeline."""

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.inference.trace import trace
from quivers.inference.mcmc.driver import MCMCResult


def _frame():
    return pd.DataFrame(
        dict(
            y=[-0.2, 0.3, 0.8, 0.1],
            x=[-0.4, 0.2, 0.5, 0.8],
            c=[-1, 0, 1, 2],
            end=[math.nan, math.nan, math.nan, 1.1],
            w=[0.0, 0.5, 1.0, 2.0],
            se=[0.2, 0.3, 0.4, 0.5],
        )
    )


@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_recorded_event_likelihood_and_every_coefficient_gradient_match_direct_math(
    truncated, weighted
):
    frame = _frame()
    formula = (
        "y | cens(c,end)"
        + (" + trunc(lb=-1.,ub=2.)" if truncated else "")
        + (" + weights(w)" if weighted else "")
        + " ~ x"
    )
    parsed = formula_from_data(formula, frame)
    lens = FormulaToQVRModule(
        families["gaussian"],
        fixed_parameters={"sigma": 1.1},
        user_priors={"beta_x": "Normal(0.0,5.0)"},
    )
    module, _ = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    intercept = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    slope = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(frame.y.to_numpy(), dtype=torch.float64)
    x = torch.tensor(frame.x.to_numpy(), dtype=torch.float64)
    recorded = trace(
        program,
        torch.zeros(1, 4),
        lens.fixed_column_observations(parsed)
        | {"y": y, "intercept": intercept, "beta_x": slope},
    )
    normal = torch.distributions.Normal(intercept + slope * x, 1.1)

    def cdf(value):
        return torch.special.ndtr((value - normal.loc) / 1.1)

    low = torch.tensor(-1.0 if truncated else -torch.inf, dtype=torch.float64)
    high = torch.tensor(2.0 if truncated else torch.inf, dtype=torch.float64)
    denominator = cdf(high) - cdf(low)
    expected = (
        torch.stack(
            (
                torch.log(cdf(y)[0] - cdf(low)[0]),
                normal.log_prob(y)[1],
                torch.log(cdf(high)[2] - cdf(y)[2]),
                torch.log(cdf(torch.tensor(1.1, dtype=torch.float64))[3] - cdf(y)[3]),
            )
        )
        - denominator.log()
    )
    site = recorded.sites["y"]
    torch.testing.assert_close(
        site.sampleable.raw_log_prob_pointwise(y), expected, rtol=2e-7, atol=2e-7
    )
    weights = (
        torch.tensor(frame.w.to_numpy(), dtype=torch.float64)
        if weighted
        else torch.ones(4, dtype=torch.float64)
    )
    prior = torch.distributions.Normal(0.0, 5.0)
    oracle = (
        (weights * expected).sum() + prior.log_prob(intercept) + prior.log_prob(slope)
    )
    torch.testing.assert_close(recorded.log_joint.sum(), oracle, rtol=2e-7, atol=2e-7)
    actual_grad = torch.autograd.grad(
        recorded.log_joint.sum(), (intercept, slope), retain_graph=True
    )
    expected_grad = torch.autograd.grad(oracle, (intercept, slope))
    for actual, wanted in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, wanted, rtol=2e-7, atol=2e-7)
    # Canonical source roundtrip preserves the typed recording options.
    from quivers.dsl.emit import module_to_source

    source = module_to_source(module)
    assert module_to_source(parse(source)) == source


@pytest.mark.parametrize("sigma", [False, True])
def test_known_standard_errors_use_combined_scale_and_drop_excluded_sigma_prior(sigma):
    frame = _frame()
    formula = f"y | se(se,sigma={sigma}) ~ x"
    parsed = formula_from_data(formula, frame)
    lens = FormulaToQVRModule(
        families["gaussian"], fixed_parameters={"sigma": 0.7} if sigma else {}
    )
    module, _ = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    y = torch.tensor(frame.y.to_numpy(), dtype=torch.float64)
    recorded = trace(
        program,
        torch.zeros(1, 4),
        lens.fixed_column_observations(parsed)
        | {"y": y, "intercept": torch.tensor(0.2), "beta_x": torch.tensor(-0.3)},
    )
    wanted = torch.hypot(
        torch.tensor(frame.se.to_numpy()), torch.tensor(0.7 if sigma else 0.0)
    )
    torch.testing.assert_close(
        recorded.sites["formula_sigma"].value.reshape(-1), wanted, rtol=1e-7, atol=1e-7
    )
    assert ("sigma" in get_prior(formula, data=frame)) is sigma
    if not sigma:
        assert "sigma" not in recorded.sites


def _recorded(fitted):
    return fitted.with_(
        posterior=MCMCResult(
            samples={
                "intercept": torch.full((2, 3), 0.2),
                "beta_x": torch.full((2, 3), -0.3),
            },
            log_densities=torch.zeros(2, 3),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.long),
            num_warmup=0,
            num_samples=3,
        )
    )


def test_fresh_predictions_use_truncated_sampling_law_without_censor_response_or_weights():
    frame = _frame()
    fitted = _recorded(
        fit(
            "y | cens(c,end) + trunc(lb=-1.,ub=2.) + weights(w) ~ x",
            data=frame,
            fixed_parameters={"sigma": 1.1},
            method="svi",
            num_steps=1,
            seed=3,
        )
    )
    fresh = pd.DataFrame({"x": [0.1, 0.8]})
    samples = fitted.posterior_predict(new_data=fresh, seed=4)
    assert samples.shape == (2, 3, 2)
    assert bool(((samples >= -1) & (samples <= 2)).all())
    assert torch.unique(samples).numel() > 2
    means = fitted.posterior_epred(new_data=fresh, mc_samples=128, seed=4)
    assert means.shape == (2, 3, 2) and bool(((means > -1) & (means < 2)).all())


def test_subset_is_shared_by_primary_auxiliary_neural_and_trials_bindings():
    frame = pd.DataFrame(
        {
            "y": [0.0, math.nan, 2.0],
            "x": [0.1, math.nan, 0.3],
            "z": [0.2, math.nan, 0.4],
            "keep": [True, False, True],
            "n": [0, -1, 3],
        }
    )
    result = fit(
        "y | subset(keep) + trials(n) ~ x",
        data=frame,
        family="binomial",
        method="svi",
        num_steps=1,
        seed=2,
    )
    assert result.formula.response_values.shape == (2,)
    assert result.binomial_trials == "formula_trials"
    assert result.observations["formula_trials"].tolist() == [0, 3]
    gaussian = fit(
        "y | subset(keep) ~ x",
        data=frame,
        distributional={"sigma": "~z"},
        method="svi",
        num_steps=1,
        seed=2,
    )
    assert gaussian.distributional["sigma"].fixed_columns[-1].data.tolist() == [
        0.2,
        0.4,
    ]


def test_fresh_known_errors_are_required_and_validated_before_posterior_draws():
    fitted = _recorded(
        fit("y | se(se,sigma=False) ~ x", data=_frame(), method="svi", num_steps=1)
    )
    with pytest.raises(ValueError, match="standard_error.*existing"):
        fitted.posterior_predict(new_data=pd.DataFrame({"x": [0.2]}))
    with pytest.raises(ValueError, match="nonnegative"):
        fitted.posterior_predict(new_data=pd.DataFrame({"x": [0.2], "se": [-0.1]}))
    actual = fitted.posterior_epred(
        new_data=pd.DataFrame({"x": [0.2, 0.6], "se": [0.1, 0.9]})
    )
    wanted = torch.tensor([0.14, 0.02], dtype=torch.float64).expand(2, 3, 2)
    torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=1e-6)


def test_compiled_known_error_combination_is_stable_and_preserves_scale_gradient():
    errors = [0.0, 0.3, 1e200]
    frame = pd.DataFrame({"y": [0.0, 0.0, 0.0], "se": errors})
    parsed = formula_from_data("y | se(se) ~ 1", frame)
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(parsed)
    sigma = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    recorded = trace(
        Compiler(module).compile().morphism,
        torch.zeros(1, 3),
        lens.fixed_column_observations(parsed)
        | {"y": torch.zeros(3), "intercept": torch.tensor(0.0), "sigma": sigma},
    )
    actual = recorded.sites["formula_sigma"].value.reshape(-1)
    expected = torch.tensor(
        [math.hypot(error, 0.8) for error in errors], dtype=torch.float64
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), sigma)[0], (0.8 / expected).sum()
    )


@pytest.mark.parametrize("specification", ["n", "3"])
def test_fresh_trial_modifier_uses_retained_authored_source(specification):
    frame = pd.DataFrame({"y": [0, 1, 2], "x": [0.1, 0.2, 0.3], "n": [0, 2, 3]})
    fitted = fit(
        f"y | trials({specification}) ~ x",
        data=frame,
        family="binomial",
        method="svi",
        num_steps=1,
        seed=4,
    )
    fresh = pd.DataFrame({"x": [0.2, 0.6], "n": [0, 5]})
    result = fitted.posterior_epred(new_data=fresh, num_samples=2, seed=5)
    assert result.shape == (1, 2, 2)
    assert bool((result >= 0).all())
    if specification == "n":
        assert bool((result[..., 0] == 0).all())
        assert bool((result[..., 1] <= 5).all())
    else:
        assert bool((result <= 3).all())
