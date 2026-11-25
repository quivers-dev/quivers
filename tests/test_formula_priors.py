"""Prior inspection describes the exact distributions compiled for fitting."""

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    formula_to_qvr,
    get_prior,
)
from quivers.inference.trace import trace


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.1, 0.5, 0.2, 0.7],
            "x": [1.0, 2.0, 3.0, 4.0],
            "g": ["a", "a", "b", "b"],
        }
    )


def test_inspected_default_scale_is_actual_density_scale(data):
    priors = get_prior("y ~ x", data=data)
    scale = 5 / math.sqrt(7.5)
    assert priors == {
        "intercept": "Normal(loc=0.0, scale=5.0)",
        "beta_x": f"Normal(loc=0.0, scale={scale})",
        "sigma": "HalfCauchy(scale=2.0)",
    }
    formula = formula_from_data("y ~ x", data)
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    recorded = trace(
        program,
        torch.zeros(1, 1, dtype=torch.long),
        lens.fixed_column_observations(formula)
        | {"y": torch.tensor(data.y), "beta_x": torch.tensor(0.4)},
    )
    torch.testing.assert_close(
        recorded.sites["beta_x"].log_prob.sum(),
        torch.distributions.Normal(0.0, scale).log_prob(torch.tensor(0.4)),
    )


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_prior_names_match_correlated_distributional_coefficients(
    data, parameterization
):
    priors = get_prior(
        "y ~ x + (1 + x | joint | g)",
        data=data,
        distributional={"sigma": "~ x + (1 | joint | g)"},
        reparameterize=parameterization,
        priors={
            "dpar_sigma_beta_x": "Normal(loc=0.2, scale=0.3)",
            "cor_shared_joint": "LKJCholesky(2.0)",
        },
    )
    assert priors["dpar_sigma_beta_x"] == "Normal(loc=0.2, scale=0.3)"
    assert priors["cor_shared_joint"] == "LKJCholesky(concentration=2.0)"
    assert "dpar_sigma_sigma_g_Intercept" in priors
    assert "sigma_g_x" in priors
    assert "sigma" not in priors
    assert all(
        not name.startswith(("z_", "alpha_", "beta_g_", "dpar_sigma_z_"))
        for name in priors
    )


@pytest.mark.parametrize(
    "name", ["beta_typo", "z_g_Intercept", "alpha_g", "cor_missing"]
)
def test_unused_prior_overrides_fail_before_fitting(data, name):
    with pytest.raises(ValueError, match="unused prior overrides"):
        fit(
            "y ~ x + (1 | g)",
            data=data,
            priors={name: "Normal(0.0, 1.0)"},
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="available parameters"):
        get_prior("y ~ x + (1 | g)", data=data, priors={name: "Normal(0.0, 1.0)"})


def test_mixture_prior_inspection_has_all_component_hyperpriors(data):
    priors = get_prior("y ~ x", data=data, family="mixture", mixture_components=3)
    assert {
        "theta",
        "mixture_intercept_spacing",
        "sigma1",
        "sigma2",
        "sigma3",
        "dpar_mu1_intercept",
        "dpar_mu3_beta_x",
    } <= priors.keys()
    assert "dpar_mu2_intercept" not in priors
    source = formula_to_qvr(
        "y ~ x", data=data, family="mixture", mixture_components=3, priors=priors
    )
    assert "sample sigma3" in source


def test_categorical_inspection_preserves_reference_coefficient_prior_names(data):
    data = data.assign(y=[0, 1, 2, 0])
    priors = get_prior("y ~ x + (1 | g)", data=data, family="categorical")
    assert {"intercept", "beta_x", "sigma_g_Intercept"} == priors.keys()
    assert "Normal(loc=0.0, scale=5.0)" == priors["intercept"]


def test_inspection_preserves_rng_and_does_not_run_inference(data):
    state = torch.random.get_rng_state()
    get_prior("y ~ x + (1 + x | g)", data=data)
    assert torch.equal(torch.random.get_rng_state(), state)


def test_fixed_auxiliary_parameters_are_absent_from_prior_inspection(data):
    priors = get_prior(
        "y ~ x",
        data=data,
        family="asym_laplace",
        fixed_parameters={"quantile": 0.9, "sigma": 0.7},
    )
    assert priors.keys() == {"intercept", "beta_x"}
    source = formula_to_qvr(
        "y ~ x",
        data=data,
        family="asym_laplace",
        fixed_parameters={"quantile": 0.9, "sigma": 0.7},
        priors=priors,
    )
    assert "sample sigma" not in source
    assert "let sigma = 0.7" in source


def test_fixed_boundary_inflation_probability_is_valid(data):
    source = formula_to_qvr(
        "y ~ x",
        data=data.assign(y=[0, 1, 2, 0]),
        family="zero_inflated_poisson",
        fixed_parameters={"zi": 0.0},
    )
    assert "let zi = 0.0" in source
