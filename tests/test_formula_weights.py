"""Formula likelihood weights preserve priors, retained rows, and predictive laws."""

import math
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import (
    ResponseModifiers,
    FixedColumn,
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    formula_to_qvr,
    get_prior,
)
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.trace import trace


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.2, 0.7, -0.3, 0.5],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "w": [0.0, 0.5, 2.0, 1.0],
            "hours": [0.5, 1.0, 2.0, 4.0],
        }
    )


def test_weights_are_retained_immutable_data_without_coefficients_or_priors(data):
    formula = formula_from_data("y ~ x", data, weights="w")
    weight = formula.modifiers.weights
    assert weight is not None and weight.name == "w"
    np.testing.assert_array_equal(weight.data, data.w)
    assert not weight.data.flags.writeable
    assert pickle.loads(pickle.dumps(formula)) == formula
    lens = FormulaToQVRModule(families["gaussian"])
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    assert "[weights=formula_weights]" in module_to_source(module)
    assert set(lens.get_prior(formula)) == {"intercept", "beta_x", "sigma"}
    assert "formula_weights" in lens.fixed_column_observations(formula)
    data.loc[0, "w"] = 99.0
    assert weight.data[0] == 0
    with pytest.raises(ValueError, match="unused prior overrides"):
        get_prior(
            "y ~ x",
            data=data,
            weights="w",
            priors={"formula_weights": "Normal(0.0, 1.0)"},
        )


def test_weighted_gaussian_joint_and_gradients_use_unweighted_priors(data):
    formula = formula_from_data("y ~ x", data, weights="w")
    lens = FormulaToQVRModule(
        families["gaussian"],
        fixed_parameters={"sigma": 0.7},
        user_priors={"intercept": "Normal(0.0, 1.0)", "beta_x": "Normal(0.0, 1.0)"},
    )
    module, _ = lens.forward(formula)
    values = {
        "intercept": torch.tensor(0.3, requires_grad=True),
        "beta_x": torch.tensor(-0.2, requires_grad=True),
    }
    observations = (
        lens.fixed_column_observations(formula)
        | values
        | {"y": torch.tensor(data.y.to_numpy(), dtype=torch.float32)}
    )
    recorded = trace(
        Compiler(module).compile().morphism,
        torch.zeros(1, 1, dtype=torch.long),
        observations,
    )
    mu = values["intercept"] + values["beta_x"] * observations["x"]
    raw = (
        -0.5 * ((observations["y"] - mu) / 0.7).square()
        - math.log(0.7)
        - 0.5 * math.log(2 * math.pi)
    )
    prior = sum(
        -0.5 * (value.square() + math.log(2 * math.pi)) for value in values.values()
    )
    expected = prior + (observations["formula_weights"] * raw).sum()
    actual = recorded.log_joint.sum()
    torch.testing.assert_close(actual, expected)
    for observed, oracle in zip(
        torch.autograd.grad(actual, tuple(values.values()), retain_graph=True),
        torch.autograd.grad(expected, tuple(values.values())),
        strict=True,
    ):
        torch.testing.assert_close(observed, oracle)
    sampleable = recorded.sites["y"].sampleable
    torch.testing.assert_close(
        sampleable.raw_log_prob_pointwise(recorded.sites["y"].value).reshape(-1), raw
    )


def test_exposure_and_weights_have_separate_poisson_roles(data):
    frame = data.assign(y=[0, 1, 3, 2])
    formula = formula_from_data("y ~ x + exposure(hours)", frame, weights="w")
    lens = FormulaToQVRModule(families["poisson"])
    program = Compiler(lens.forward(formula)[0]).compile().morphism
    observations = lens.fixed_column_observations(formula) | {
        "y": torch.tensor(frame.y.to_numpy(), dtype=torch.float32),
        "intercept": torch.tensor(0.1, requires_grad=True),
        "beta_x": torch.tensor(-0.2, requires_grad=True),
    }
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    rate = (
        observations["intercept"] + observations["beta_x"] * observations["x"]
    ).exp() * torch.tensor(frame.hours.to_numpy(), dtype=torch.float32)
    raw = observations["y"] * rate.log() - rate - torch.lgamma(observations["y"] + 1)
    actual = recorded.sites["y"].log_prob.sum()
    expected = (observations["formula_weights"] * raw).sum()
    torch.testing.assert_close(actual, expected)
    variables = (observations["intercept"], observations["beta_x"])
    for observed, oracle in zip(
        torch.autograd.grad(actual, variables, retain_graph=True),
        torch.autograd.grad(expected, variables),
        strict=True,
    ):
        torch.testing.assert_close(observed, oracle)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_weighted_fit_source_and_fixed_posterior_predictions(data, method):
    result = fit(
        "y ~ x",
        data=data,
        weights="w",
        fixed_parameters={"sigma": 0.7},
        method=method,
        num_steps=2 if method == "svi" else None,
        num_samples=2 if method != "svi" else None,
        num_warmup=2 if method != "svi" else None,
        num_chains=1 if method != "svi" else None,
        seed=13,
    )
    assert result.qvr_source == formula_to_qvr(
        "y ~ x", data=data, weights="w", fixed_parameters={"sigma": 0.7}
    )
    loads(result.qvr_source)
    posterior = MCMCResult(
        samples={
            "intercept": torch.full((1, 2), 0.3),
            "beta_x": torch.full((1, 2), -0.2),
        },
        log_densities=torch.zeros(1, 2),
        acceptance_rates=torch.ones(1),
        divergence_counts=torch.zeros(1),
        num_warmup=0,
        num_samples=2,
    )
    result = result.with_(posterior=posterior)
    expected = torch.tensor(0.3 - 0.2 * data.x.to_numpy(), dtype=torch.float32).expand(
        1, 2, 4
    )
    torch.testing.assert_close(result.posterior_linpred(), expected)
    torch.testing.assert_close(result.posterior_epred(), expected)
    changed = result.with_(
        observations=dict(result.observations) | {"formula_weights": torch.zeros(4)}
    )
    torch.testing.assert_close(result.posterior_epred(), changed.posterior_epred())
    assert set(result.fixed_effects()) == {"Intercept", "x"}


def test_categorical_probabilities_ignore_observation_weights_at_fixed_coefficients(
    data,
):
    result = fit(
        "y ~ x",
        data=data.assign(y=[0, 1, 2, 0]),
        family="categorical",
        weights="w",
        method="svi",
        num_steps=2,
        seed=17,
    )
    first = result.posterior_epred(num_samples=3, seed=11)
    changed = result.with_(
        observations=dict(result.observations) | {"formula_weights": torch.zeros(4)}
    )
    torch.testing.assert_close(first, changed.posterior_epred(num_samples=3, seed=11))
    torch.testing.assert_close(first.sum(-1), torch.ones(1, 3, 4))


@pytest.mark.parametrize("entry", [formula_from_data, formula_to_qvr, get_prior, fit])
@pytest.mark.parametrize(
    "column, pattern",
    [
        ([-1.0, 1.0, 1.0, 1.0], "nonnegative"),
        ([np.inf, 1.0, 1.0, 1.0], "finite"),
        ([True, False, True, False], "numeric"),
        (["0.0", "1.0", "2.0", "3.0"], "numeric"),
        ([1j, 1j, 1j, 1j], "numeric"),
    ],
)
def test_weight_column_validation_precedes_compilation(data, entry, column, pattern):
    with pytest.raises(ValueError, match=pattern):
        if entry is formula_from_data:
            entry("y ~ x", data.assign(w=column), weights="w")
        else:
            entry("y ~ x", data=data.assign(w=column), weights="w")


def test_missing_selector_dropped_rows_and_reserved_input_names_are_rejected(data):
    with pytest.raises(ValueError, match="existing dataframe column"):
        formula_from_data("y ~ x", data, weights="missing")
    # As brms's na_omit, a row missing a predictor or a weight is removed from
    # the weights and the design together.
    for column in ("x", "w"):
        with pytest.warns(UserWarning, match="excluded from the model"):
            dropped = formula_from_data(
                "y ~ x", data.assign(**{column: [np.nan, 0.0, 1.0, 2.0]}), weights="w"
            )
        assert dropped.modifiers.weights is not None
        assert dropped.modifiers.weights.data.shape == (3,)
        assert dropped.response_values.shape == (3,)
    with pytest.raises(ValueError, match="formula_weights is reserved"):
        formula_to_qvr(
            "y ~ formula_weights", data=data.assign(formula_weights=data.x), weights="w"
        )


def test_manually_constructed_formula_weight_shape_is_checked(data):
    formula = formula_from_data("y ~ x", data)
    malformed = formula.with_(
        modifiers=ResponseModifiers(
            weights=FixedColumn(term="w", name="w", qvr_name="w", data=np.ones(3))
        )
    )
    with pytest.raises(ValueError, match="align with every"):
        FormulaToQVRModule(families["gaussian"]).forward(malformed)


def test_distributional_scale_priors_are_not_weighted(data):
    formula = formula_from_data("y ~ x", data, weights="w")
    scale = formula_from_data("y ~ x", data)
    names = ("intercept", "beta_x", "dpar_sigma_intercept", "dpar_sigma_beta_x")
    lens = FormulaToQVRModule(
        families["gaussian"],
        distributional={"sigma": scale},
        user_priors={name: "Normal(0.0, 1.0)" for name in names},
    )
    values = {
        name: torch.tensor(value, requires_grad=True)
        for name, value in zip(names, (0.3, -0.2, -0.4, 0.1), strict=True)
    }
    observations = (
        lens.fixed_column_observations(formula)
        | values
        | {"y": torch.tensor(data.y.to_numpy(), dtype=torch.float32)}
    )
    program = Compiler(lens.forward(formula)[0]).compile().morphism
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    mean = values["intercept"] + values["beta_x"] * observations["x"]
    sigma = (
        values["dpar_sigma_intercept"] + values["dpar_sigma_beta_x"] * observations["x"]
    ).exp()
    raw = (
        -0.5 * ((observations["y"] - mean) / sigma).square()
        - sigma.log()
        - 0.5 * math.log(2 * math.pi)
    )
    prior = sum(
        -0.5 * (value.square() + math.log(2 * math.pi)) for value in values.values()
    )
    expected = prior + (observations["formula_weights"] * raw).sum()
    torch.testing.assert_close(recorded.log_joint.sum(), expected)
    for observed, oracle in zip(
        torch.autograd.grad(
            recorded.log_joint.sum(), tuple(values.values()), retain_graph=True
        ),
        torch.autograd.grad(expected, tuple(values.values())),
        strict=True,
    ):
        torch.testing.assert_close(observed, oracle)


def test_weight_input_has_explicit_row_type_and_response_names_cannot_shadow_it(data):
    source = formula_to_qvr("y ~ x", data=data, weights="w")
    assert "formula_weights : Tensor[Real]([Resp])" in source
    with pytest.raises(ValueError, match="formula_weights is reserved"):
        formula_to_qvr(
            "formula_weights ~ x",
            data=data.rename(columns={"y": "formula_weights"}),
            weights="w",
        )
    formula = formula_from_data("y ~ x", data, weights="w")
    with pytest.raises(ValueError, match="formula_weights is reserved"):
        FormulaToQVRModule(
            families["gaussian"], predictor_name="formula_weights"
        ).forward(formula)


def test_weights_with_shared_correlated_location_scale_groups(data):
    frame = data.assign(g=["a", "a", "b", "b"])
    location = formula_from_data("y ~ x + (1 + x | shared | g)", frame, weights="w")
    scale = formula_from_data("y ~ x + (1 | shared | g)", frame)
    lens = FormulaToQVRModule(families["gaussian"], distributional={"sigma": scale})
    module, complement = lens.forward(location)
    assert lens.backward(module, complement) == location
    values = {
        "intercept": torch.tensor(0.3, requires_grad=True),
        "beta_x": torch.tensor(-0.2, requires_grad=True),
        "dpar_sigma_intercept": torch.tensor(-0.4, requires_grad=True),
        "dpar_sigma_beta_x": torch.tensor(0.1, requires_grad=True),
        "sigma_g_Intercept": torch.tensor(0.4),
        "sigma_g_x": torch.tensor(0.7),
        "dpar_sigma_sigma_g_Intercept": torch.tensor(0.2),
        "z_g_Intercept": torch.tensor([0.2, -0.5]),
        "z_g_x": torch.tensor([1.0, 0.3]),
        "dpar_sigma_z_g_Intercept": torch.tensor([-0.1, 0.4]),
        "cor_shared_shared": torch.tensor(
            [[1.0, 0.0, 0.0], [0.3, math.sqrt(0.91), 0.0], [-0.2, 0.1, math.sqrt(0.95)]]
        ),
    }
    observations = (
        lens.fixed_column_observations(location)
        | values
        | {
            "g_idx": torch.tensor([0, 0, 1, 1]),
            "y": torch.tensor(frame.y.to_numpy(), dtype=torch.float32),
        }
    )
    recorded = trace(
        Compiler(module).compile().morphism,
        torch.zeros(1, 1, dtype=torch.long),
        observations,
    )
    z = torch.stack(
        (values["z_g_Intercept"], values["z_g_x"], values["dpar_sigma_z_g_Intercept"]),
        dim=-1,
    )
    coefficients = z @ values["cor_shared_shared"].T * torch.tensor([0.4, 0.7, 0.2])
    per_row = coefficients[observations["g_idx"]]
    x = observations["x"]
    mu = values["intercept"] + values["beta_x"] * x + per_row[:, 0] + per_row[:, 1] * x
    sigma = (
        values["dpar_sigma_intercept"] + values["dpar_sigma_beta_x"] * x + per_row[:, 2]
    ).exp()
    raw = (
        -0.5 * ((observations["y"] - mu) / sigma).square()
        - sigma.log()
        - 0.5 * math.log(2 * math.pi)
    )
    expected = (observations["formula_weights"] * raw).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    differentiable = tuple(
        values[name]
        for name in ("intercept", "beta_x", "dpar_sigma_intercept", "dpar_sigma_beta_x")
    )
    for observed, oracle in zip(
        torch.autograd.grad(actual, differentiable, retain_graph=True),
        torch.autograd.grad(expected, differentiable),
        strict=True,
    ):
        torch.testing.assert_close(observed, oracle)
