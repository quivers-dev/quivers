"""Formula updates retain original fitting inputs without another fitting engine."""

import inspect

import pandas as pd
import pytest
import torch

from quivers.formulas import fit
from quivers.inference import MCMCResult
from quivers.modeling import refit_callback
from quivers.diagnostics.cross_validation import FoldPlan, cross_validate


def _data():
    return pd.DataFrame({"y": [0.3, 0.6, 0.2, 0.8], "x": [-1.0, 0.0, 1.0, 2.0]})


def test_formula_update_clones_training_frame_and_retains_complete_signature():
    data = _data()
    fitted = fit("y ~ x", data=data, method="svi", num_steps=2, seed=17)
    old_program, old_guide = fitted.program, fitted.posterior
    data.loc[:, "x"] = 100.0
    updated = fitted.update()
    assert updated is not fitted and updated.program is not old_program
    assert updated.posterior is not old_guide and fitted.posterior is old_guide
    assert updated.formula == fitted.formula
    assert updated.losses == fitted.losses
    assert set(fitted.refit_recipe.method) == set(inspect.signature(fit).parameters) - {
        "data",
        "snapshot_policy",
    }
    with pytest.raises(TypeError):
        fitted.refit_recipe.method["num_steps"] = 100
    changed = updated.update(data=_data().iloc[:2], formula="y ~ 1", num_steps=0)
    assert changed.formula.response_values.shape == (2,)
    assert len(changed.formula.fixed_columns) == 1
    assert changed.formula.fixed_columns[0].is_intercept


def test_formula_method_transitions_clear_only_inherited_incompatible_budgets():
    original = fit("y~1", data=_data(), method="svi", num_steps=0, seed=3)
    sampled = original.update(method="hmc", num_warmup=2, num_samples=4, num_chains=1)
    assert isinstance(sampled.posterior, MCMCResult)
    assert sampled.posterior.samples["intercept"].shape[:2] == (1, 4)
    variational = sampled.update(method="svi", num_steps=0)
    assert variational.losses == ()
    with pytest.raises(ValueError, match="num_steps.*SVI"):
        original.update(
            method="hmc", num_steps=3, num_warmup=0, num_samples=1, num_chains=1
        )
    with pytest.raises(ValueError, match="num_samples.*MCMC"):
        sampled.update(method="svi", num_steps=0, num_samples=2)


def test_unknown_formula_override_fails_before_preparation(monkeypatch):
    import quivers.formulas._fit as module

    fitted = fit("y~1", data=_data(), method="svi", num_steps=0)
    calls = []
    original = module._prepare_formula

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_prepare_formula", counted)
    with pytest.raises(TypeError, match="unexpected keyword"):
        fitted.update(nmu_steps=3)
    assert calls == []


def test_formula_neural_recipe_retains_initial_module_and_original_subset_rows():
    data = _data().assign(keep=[True, False, True, True])
    predictor = torch.nn.Linear(1, 1)
    with torch.no_grad():
        predictor.weight.fill_(0.4)
        predictor.bias.fill_(0.2)
    initial = {
        name: value.detach().clone() for name, value in predictor.state_dict().items()
    }
    predictor_data = torch.arange(4.0).reshape(-1, 1)
    fitted = fit(
        "y | subset(keep) ~ 1",
        data=data,
        method="svi",
        num_steps=2,
        seed=8,
        predictor=predictor,
        predictor_data=predictor_data,
    )
    with torch.no_grad():
        predictor.weight.fill_(99)
        predictor.bias.fill_(99)
        predictor_data.add_(100)
    data.loc[:, "keep"] = False
    updated = fitted.update(num_steps=0)
    assert updated.predictor is not predictor
    for name, value in updated.predictor.state_dict().items():
        torch.testing.assert_close(value, initial[name])
    torch.testing.assert_close(
        updated.predictor_data, torch.tensor([[0.0], [2.0], [3.0]])
    )
    assert updated.formula.response_values.shape == (3,)


def test_formula_refit_callback_uses_existing_raw_heldout_accessor():
    fitted = fit("y~x", data=_data(), method="svi", num_steps=1, seed=3)
    callback = refit_callback(
        fitted.refit_recipe,
        lambda data, rows: data.iloc[list(rows)].copy(),
        lambda fitted, data, seed: fitted.log_likelihood(
            new_data=data, num_samples=4, seed=seed
        ),
        method_factory=lambda settings, seed: {**settings, "seed": seed},
    )
    result = cross_validate(
        FoldPlan(assignments=(0, 0, 1, 1), seed=6), callback, save_fits=True
    )
    assert result.complete and result.log_predictive_density.shape == (4,)
    for fold, fitted in result.fits.items():
        assert fitted.formula.response_values.shape == (2,)
        assert fitted.refit_recipe.method["seed"] == 7 + fold


def test_formula_refit_reference_policy_is_explicit():
    data = _data()
    fitted = fit(
        "y~1", data=data, method="svi", num_steps=0, snapshot_policy="reference"
    )
    data.loc[:, "y"] += 1
    updated = fitted.update()
    assert updated.refit_recipe.snapshot_policy == "reference"
    torch.testing.assert_close(
        torch.tensor(updated.formula.response_values), torch.tensor(data.y.to_numpy())
    )


def test_formula_update_target_and_every_coefficient_gradient_match_normal_oracle():
    original = fit(
        "y~x",
        data=_data(),
        method="svi",
        num_steps=0,
        fixed_prior="Normal(0.0,2.0)",
        fixed_parameters={"sigma": 1.0},
    )
    frame = pd.DataFrame({"y": [0.25, -0.5, 0.75], "x": [0.5, 2.0, -1.0]})
    updated = original.update(data=frame)
    theta = torch.tensor([0.1, 0.7], dtype=torch.float64, requires_grad=True)
    actual = updated.program.log_joint(
        torch.zeros(1, 1),
        {
            **updated.observations,
            "intercept": theta[0],
            "beta_x": theta[1],
        },
    ).sum()
    x = torch.tensor(frame.x.to_numpy())
    y = torch.tensor(frame.y.to_numpy())
    # The legacy fixed_prior setting rescales each slope by its training
    # column's RMS; a refit must use the replacement training column.
    scales = torch.stack((x.new_tensor(2.0), 2 / x.square().mean().sqrt()))
    expected = (
        torch.distributions.Normal(torch.zeros_like(scales), scales)
        .log_prob(theta)
        .sum()
        + torch.distributions.Normal(theta[0] + theta[1] * x, 1.0).log_prob(y).sum()
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, theta)[0], torch.autograd.grad(expected, theta)[0]
    )
