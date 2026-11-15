"""Offset and exposure terms have fixed coefficients in native regressions."""

import math
import pickle

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.trace import trace


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.4, 1.1, -0.3, 0.8],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.3, 0.0, 0.5],
            "time": [0.5, 1.0, 2.0, 4.0],
            "g": ["a", "a", "b", "b"],
        }
    )


def test_offsets_are_retained_without_coefficients_or_priors(data):
    formula = formula_from_data("y ~ x + offset(z) + exposure(time)", data)
    assert [column.name for column in formula.fixed_columns] == ["Intercept", "x"]
    assert [column.name for column in formula.offsets] == [
        "offset(z)",
        "exposure(time)",
    ]
    np.testing.assert_array_equal(formula.offsets[0].data, data.z)
    np.testing.assert_allclose(formula.offsets[1].data, np.log(data.time))
    assert not formula.offsets[0].data.flags.writeable
    assert pickle.loads(pickle.dumps(formula)) == formula
    lens = FormulaToQVRModule(families["gaussian"])
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    source = module_to_source(module)
    assert "beta_offset" not in source and "beta_exposure" not in source
    assert set(lens.get_prior(formula)) == {"intercept", "beta_x", "sigma"}
    assert set(get_prior("y ~ 0 + offset(z)", data=data)) == {"sigma"}
    reparsed = loads(source).morphism
    assert reparsed is not None


def test_gaussian_offset_density_and_input_coefficient_gradients(data):
    formula = formula_from_data("y ~ x + offset(z)", data)
    lens = FormulaToQVRModule(families["gaussian"], fixed_parameters={"sigma": 0.7})
    program = Compiler(lens.forward(formula)[0]).compile().morphism
    intercept = torch.tensor(0.3, requires_grad=True)
    slope = torch.tensor(-0.2, requires_grad=True)
    offset = torch.tensor(data.z.to_numpy(), dtype=torch.float32, requires_grad=True)
    observations = lens.fixed_column_observations(formula) | {
        "y": torch.tensor(data.y.to_numpy(), dtype=torch.float32),
        "intercept": intercept,
        "beta_x": slope,
        "formula_offset_0": offset,
    }
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    eta = intercept + slope * observations["x"] + offset
    expected = (
        -0.5 * ((observations["y"] - eta) / 0.7).square()
        - math.log(0.7)
        - 0.5 * math.log(2 * math.pi)
    ).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(
        actual, (intercept, slope, offset), retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected, (intercept, slope, offset))
    for observed, oracle in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, oracle)


def test_poisson_exposure_is_observation_time_multiplier(data):
    frame = data.assign(y=[0, 1, 3, 2])
    formula = formula_from_data("y ~ x + exposure(time)", frame)
    lens = FormulaToQVRModule(families["poisson"])
    program = Compiler(lens.forward(formula)[0]).compile().morphism
    observations = lens.fixed_column_observations(formula) | {
        "y": torch.tensor(frame.y.to_numpy(), dtype=torch.float64),
        "intercept": torch.tensor(0.1, requires_grad=True),
        "beta_x": torch.tensor(0.2, requires_grad=True),
    }
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    rate = torch.exp(
        observations["intercept"].double()
        + observations["beta_x"].double() * observations["x"].double()
    ) * torch.tensor(frame.time.to_numpy(), dtype=torch.float32)
    response = observations["y"].double()
    expected = (response * rate.log() - rate - torch.lgamma(response + 1)).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    variables = (observations["intercept"], observations["beta_x"])
    for observed, oracle in zip(
        torch.autograd.grad(actual, variables, retain_graph=True),
        torch.autograd.grad(expected, variables),
        strict=True,
    ):
        torch.testing.assert_close(observed, oracle)


def test_distributional_offsets_enter_scale_before_inverse_link(data):
    location = formula_from_data("y ~ 0 + offset(z)", data)
    scale = formula_from_data("y ~ 1 + exposure(time)", data)
    lens = FormulaToQVRModule(families["gaussian"], distributional={"sigma": scale})
    program = Compiler(lens.forward(location)[0]).compile().morphism
    observations = lens.fixed_column_observations(location) | {
        "y": torch.tensor(data.y.to_numpy(), dtype=torch.float32),
        "dpar_sigma_intercept": torch.tensor(-0.4, requires_grad=True),
    }
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    sigma = observations["dpar_sigma_intercept"].exp() * torch.tensor(
        data.time.to_numpy(), dtype=torch.float32
    )
    mu = torch.tensor(data.z.to_numpy(), dtype=torch.float32)
    oracle = torch.distributions.Normal(mu, sigma).log_prob(observations["y"]).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, oracle)
    variable = observations["dpar_sigma_intercept"]
    torch.testing.assert_close(
        torch.autograd.grad(actual, variable, retain_graph=True)[0],
        torch.autograd.grad(oracle, variable)[0],
    )


def test_categorical_offsets_apply_to_each_nonreference_logit(data):
    frame = data.assign(y=[0, 1, 2, 0])
    formula = formula_from_data("y ~ 0 + offset(z)", frame)
    lens = FormulaToQVRModule(families["categorical"])
    program = Compiler(lens.forward(formula)[0]).compile().morphism
    observations = lens.fixed_column_observations(formula) | {
        "y": torch.tensor(frame.y.to_numpy(), dtype=torch.float32)
    }
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    z = torch.tensor(data.z.to_numpy(), dtype=torch.float32)
    logits = torch.stack((torch.zeros_like(z), z, z), dim=-1)
    oracle = logits.log_softmax(-1).gather(1, observations["y"].long()[:, None]).sum()
    torch.testing.assert_close(recorded.sites["y"].log_prob.sum(), oracle)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_fitting_and_posterior_helpers_retain_native_offsets(data, method):
    result = fit(
        "y ~ x + offset(z) + exposure(time)",
        data=data.assign(y=[0, 1, 3, 2]),
        family="poisson",
        method=method,
        num_steps=2 if method == "svi" else None,
        num_samples=2 if method != "svi" else None,
        num_warmup=2 if method != "svi" else None,
        num_chains=1 if method != "svi" else None,
        seed=13,
    )
    samples = {"intercept": torch.full((1, 2), 0.3), "beta_x": torch.full((1, 2), -0.2)}
    result = result.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(1, 2),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1),
            num_warmup=0,
            num_samples=2,
        )
    )
    eta = torch.tensor(
        0.3
        - 0.2 * data.x.to_numpy()
        + data.z.to_numpy()
        + np.log(data.time.to_numpy()),
        dtype=torch.float32,
    )
    torch.testing.assert_close(result.posterior_linpred(), eta.expand(1, 2, 4))
    torch.testing.assert_close(
        result.posterior_epred(), eta.double().exp().expand(1, 2, 4)
    )
    assert set(result.fixed_effects()) == {"Intercept", "x"}
    assert "formula_offset_1" in result.qvr_source
    loads(result.qvr_source)


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ x:offset(z)",
        "y ~ (offset(z)|g)",
        "y ~ I(offset(z) + 1)",
        "y ~ x:exposure(time)",
        "y ~ I(exposure(time))",
    ],
)
def test_offset_interactions_groups_and_wrappers_are_rejected(data, formula):
    with pytest.raises(ValueError, match="offset and exposure"):
        formula_from_data(formula, data)


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_exposures_must_be_strictly_positive_and_finite(data, value):
    with pytest.raises(ValueError, match="positive finite|response row"):
        formula_from_data(
            "y ~ exposure(time)", data.assign(time=[value, 1.0, 2.0, 3.0])
        )


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_offsets_must_be_finite(data, value):
    with pytest.raises(ValueError, match="finite"):
        formula_from_data("y ~ offset(z)", data.assign(z=[value, 1.0, 2.0, 3.0]))


def test_generated_offset_input_names_cannot_shadow_covariates(data):
    with pytest.raises(ValueError, match="offset input names are reserved"):
        fit(
            "y ~ formula_offset_0 + offset(z)",
            data=data.assign(formula_offset_0=1.0),
            method="svi",
            num_steps=1,
        )


@pytest.mark.parametrize("name", ["offset", "exposure"])
def test_modifier_meanings_cannot_be_overridden(data, name):
    with pytest.raises(ValueError, match="reserved formula modifiers"):
        formula_from_data("y ~ 1", data, extra_namespace={name: np.exp})


def test_constant_and_removed_offsets_follow_formula_algebra(data):
    formula = formula_from_data("y ~ offset(0.5) + offset(z) - offset(z)", data)
    assert len(formula.offsets) == 1
    np.testing.assert_array_equal(formula.offsets[0].data, np.full(4, 0.5))
