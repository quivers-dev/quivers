"""Correlated formula coefficients agree with multivariate Gaussian laws."""

import math
import pickle

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import (
    FormulaToQVRModule,
    GroupCovariance,
    families,
    fit,
    formula_from_data,
    formula_to_qvr,
)
from quivers.inference.trace import trace


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.1, 1.0, -0.5, 0.7],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "g": ["a", "a", "b", "b"],
        }
    )


@pytest.mark.parametrize(
    "bar, correlated, shared",
    [("|", True, False), ("||", False, False), ("| shared |", True, True)],
)
def test_bar_notation_retains_correlation_blocks(data, bar, correlated, shared):
    formula = formula_from_data(f"y ~ x + (1 + x {bar} g)", data)
    assert len(formula.random_terms) == 2
    assert all(
        (term.correlation_id is not None) == correlated and term.shared == shared
        for term in formula.random_terms
    )
    source = formula_to_qvr(f"y ~ x + (1 + x {bar} g)", data=data)
    assert ("LKJCholesky" in source) == correlated
    loads(source)
    lens = FormulaToQVRModule(families["gaussian"])
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    assert pickle.loads(pickle.dumps(formula)) == formula


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_correlated_group_density_and_gradient_match_gaussian_oracle(
    data, parameterization
):
    formula = formula_from_data("y ~ x + (1 + x | g)", data)
    lens = FormulaToQVRModule(
        families["gaussian"],
        reparameterize=parameterization,
        user_priors={"beta_x": "Normal(0.0, 5.0)"},
    )
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    program = Compiler(module).compile().morphism
    rho = torch.tensor(0.6, requires_grad=True)
    zero = rho * 0
    factor = torch.stack(
        (
            torch.stack((zero + 1, zero)),
            torch.stack((rho, torch.sqrt(1 - rho.square()))),
        )
    )
    scales = torch.tensor([0.4, 0.7], requires_grad=True)
    innovations = torch.tensor([[0.2, 1.0], [-0.5, 0.3]])
    coefficients = innovations @ factor.T * scales
    group = torch.tensor([0, 0, 1, 1])
    inputs = lens.fixed_column_observations(formula)
    response = torch.tensor(data.y.to_numpy(), dtype=torch.float32)
    values = {
        "cor_g": factor.unsqueeze(0),
        "sigma_g_Intercept": scales[0],
        "sigma_g_x": scales[1],
        "intercept": torch.tensor(0.2),
        "beta_x": torch.tensor(-0.3),
        "sigma": torch.tensor(0.8),
    }
    if parameterization == "noncentered":
        values |= {"z_g_Intercept": innovations[:, 0], "z_g_x": innovations[:, 1]}
        group_density = torch.distributions.Normal(0.0, 1.0).log_prob(innovations).sum()
    else:
        values |= {"group_coefficients_g": coefficients}
        group_density = (
            torch.distributions.MultivariateNormal(
                torch.zeros(2), scale_tril=scales[:, None] * factor
            )
            .log_prob(coefficients)
            .sum()
        )
    recorded = trace(
        program,
        torch.zeros(1, 1, dtype=torch.long),
        inputs | values | {"g_idx": group, "y": response},
    )
    mean = (
        0.2
        - 0.3 * inputs["x"]
        + coefficients[group, 0]
        + coefficients[group, 1] * inputs["random_x"]
    )
    likelihood = (
        -0.5 * ((response - mean) / 0.8).square()
        - math.log(0.8)
        - 0.5 * math.log(2 * math.pi)
    ).sum()
    expected = (
        likelihood
        + group_density
        + torch.distributions.LKJCholesky(2, 1.0).log_prob(factor)
        + torch.distributions.HalfNormal(1.0).log_prob(scales).sum()
        + torch.distributions.Normal(0.0, 5.0).log_prob(values["intercept"])
        + torch.distributions.Normal(0.0, 5.0).log_prob(values["beta_x"])
        + torch.distributions.HalfCauchy(2.0).log_prob(values["sigma"])
    )
    torch.testing.assert_close(recorded.sites["y"].log_prob.sum(), likelihood)
    actual = recorded.log_joint.sum()
    torch.testing.assert_close(actual, expected)
    for actual_part, expected_part in zip(
        torch.autograd.grad(actual, (rho, scales), retain_graph=True),
        torch.autograd.grad(expected, (rho, scales)),
        strict=True,
    ):
        torch.testing.assert_close(actual_part, expected_part, atol=1e-5, rtol=1e-5)
    loads(module_to_source(module))


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_correlated_groups_fit_with_svi_and_hamiltonian_sampling(
    data, parameterization
):
    for method in ("svi", "hmc", "nuts"):
        counts = (
            {"num_steps": 2}
            if method == "svi"
            else {"num_warmup": 2, "num_samples": 3, "num_chains": 1}
        )
        result = fit(
            "y ~ x + (1 + x | g)",
            data=data,
            method=method,
            reparameterize=parameterization,
            seed=7,
            **counts,
        )
        assert "sample cor_g" in result.qvr_source


def test_shared_ids_join_location_and_scale_coefficients(data):
    result = fit(
        "y ~ 1 + (1 | effects | g)",
        data=data,
        distributional={"sigma": "~ 1 + (1 | effects | g)"},
        method="svi",
        num_steps=2,
    )
    assert "object FormulaCorrelation_shared_effects : FinSet 2" in result.qvr_source
    assert result.qvr_source.count("<- LKJCholesky") == 1
    assert "dpar_sigma_z_g_Intercept" in result.qvr_source
    loads(result.qvr_source)


def test_separate_blocks_are_independent_unless_their_id_is_shared(data):
    independent = formula_to_qvr("y ~ x + (1 | g) + (0 + x | g)", data=data)
    assert "LKJCholesky" not in independent
    shared = formula_to_qvr("y ~ x + (1 | block | g) + (0 + x | block | g)", data=data)
    assert shared.count("<- LKJCholesky") == 1


def test_random_slope_covariates_are_bound_without_fixed_slope(data):
    result = fit("y ~ 1 + (1 + x | g)", data=data, method="svi", num_steps=2)
    assert "x" not in result.observations
    torch.testing.assert_close(
        result.observations["random_x"],
        torch.tensor(data.x.to_numpy(), dtype=torch.float32),
    )


def test_random_factor_columns_are_distinct_coefficients(data):
    data = data.assign(factor=["a", "b", "c", "a"])
    result = fit("y ~ 1 + (1 + factor | g)", data=data, method="svi", num_steps=2)
    assert len(result.formula.random_terms) == 3
    assert "FormulaCorrelation_g : FinSet 3" in result.qvr_source


def test_categorical_group_covariance_couples_slopes_and_reference_logits():
    data = pd.DataFrame(
        {
            "y": [0, 1, 2, 0, 1, 2],
            "x": [-1.0, 0.5, 1.5, 0.0, -0.3, 2.0],
            "g": ["a", "a", "b", "b", "c", "c"],
        }
    )
    formula = formula_from_data("y ~ 1 + (1 + x | p | g)", data)
    lens = FormulaToQVRModule(families["categorical"])
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    assert "FormulaCorrelation_shared_p : FinSet 4" in module_to_source(module)
    program = Compiler(module).compile().morphism
    raw_correlation = torch.linspace(-0.5, 0.5, 6, requires_grad=True)
    factor = torch.distributions.transforms.CorrCholeskyTransform()(raw_correlation)
    scales = torch.tensor([0.4, 0.7, 0.3, 0.6], requires_grad=True)
    z = torch.tensor(
        [[0.2, 1.0, -0.3, 0.1], [-0.5, 0.3, 0.7, -0.2], [0.8, -0.6, 0.5, 0.4]]
    )
    coefficient = z @ factor.T * scales
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    inputs = lens.fixed_column_observations(formula)
    values = {
        "cor_shared_p": factor.unsqueeze(0),
        "sigma_g_Intercept": scales[:2],
        "sigma_g_x": scales[2:],
        "z_g_Intercept": z[:, :2].reshape(-1),
        "z_g_x": z[:, 2:].reshape(-1),
        "intercept": torch.tensor([0.1, -0.2]),
        "g_idx": groups,
        "y": torch.tensor(data.y.to_numpy()),
    }
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), inputs | values)
    nonreference = (
        values["intercept"]
        + coefficient[groups, :2]
        + coefficient[groups, 2:] * inputs["random_x"][:, None]
    )
    logits = torch.cat((torch.zeros(6, 1), nonreference), dim=-1)
    expected = torch.log_softmax(logits, dim=-1)[torch.arange(6), values["y"]].sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    for actual_part, expected_part in zip(
        torch.autograd.grad(actual, (raw_correlation, scales), retain_graph=True),
        torch.autograd.grad(expected, (raw_correlation, scales)),
        strict=True,
    ):
        torch.testing.assert_close(actual_part, expected_part, atol=1e-5, rtol=1e-5)
    loads(module_to_source(module))


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_correlated_categorical_models_fit_with_all_frontend_samplers(method):
    data = pd.DataFrame({"y": [0, 1, 2, 0, 1, 2], "g": ["a", "a", "b", "b", "c", "c"]})
    counts = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 3, "num_chains": 1}
    )
    result = fit(
        "y ~ 1 + (1 | p | g)",
        data=data,
        family="categorical",
        method=method,
        seed=7,
        **counts,
    )
    assert "FormulaCorrelation_shared_p : FinSet 2" in result.qvr_source


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
@pytest.mark.parametrize("known_covariance", [False, True])
def test_shared_location_scale_covariance_matches_joint_gaussian(
    data, parameterization, known_covariance
):
    formula = formula_from_data("y ~ 1 + (1 | joint | g)", data)
    scale_formula = formula_from_data("y ~ 1 + (1 | joint | g)", data)
    dtype = torch.float64 if known_covariance else torch.float32
    covariance = torch.tensor([[1.4, 0.2], [0.2, 0.8]], dtype=dtype)
    lens = FormulaToQVRModule(
        families["gaussian"],
        distributional={"sigma": scale_formula},
        reparameterize=parameterization,
        group_covariances={"g": GroupCovariance(levels=("a", "b"), matrix=covariance)}
        if known_covariance
        else None,
    )
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    rho = torch.tensor(-0.4, dtype=dtype, requires_grad=True)
    zero = rho * 0
    factor = torch.stack(
        (
            torch.stack((zero + 1, zero)),
            torch.stack((rho, torch.sqrt(1 - rho.square()))),
        )
    )
    scales = torch.tensor([0.3, 0.6], dtype=dtype, requires_grad=True)
    z = torch.tensor([[0.3, -0.4], [-0.2, 0.7]], dtype=dtype)
    coefficients = (
        (torch.linalg.cholesky(covariance) @ z if known_covariance else z)
        @ factor.T
        * scales
    )
    group = torch.tensor([0, 0, 1, 1])
    response = torch.tensor(data.y.to_numpy(), dtype=torch.float32)
    values = {
        "cor_shared_joint": factor.unsqueeze(0),
        "sigma_g_Intercept": scales[0],
        "dpar_sigma_sigma_g_Intercept": scales[1],
        "intercept": torch.tensor(0.2),
        "dpar_sigma_intercept": torch.tensor(-0.4),
        "g_idx": group,
        "y": response,
    }
    if parameterization == "centered":
        values |= {"group_coefficients_shared_joint": coefficients}
        group_density = (
            torch.distributions.MultivariateNormal(
                torch.zeros(2), scale_tril=scales[:, None] * factor
            )
            .log_prob(coefficients)
            .sum()
        )
    else:
        values |= {"z_g_Intercept": z[:, 0], "dpar_sigma_z_g_Intercept": z[:, 1]}
        group_density = torch.distributions.Normal(0.0, 1.0).log_prob(z).sum()
    if known_covariance:
        values.pop("group_coefficients_shared_joint", None)
        values.pop("z_g_Intercept", None)
        values.pop("dpar_sigma_z_g_Intercept", None)
        values.update(lens.fixed_column_observations(formula))
        values["group_covariance_shared_joint"] = (
            coefficients if parameterization == "centered" else z
        )
        if parameterization == "centered":
            coefficient_covariance = scales[:, None] * factor
            coefficient_covariance = coefficient_covariance @ coefficient_covariance.T
            group_density = torch.distributions.MultivariateNormal(
                torch.zeros(4, dtype=dtype),
                covariance_matrix=torch.kron(
                    covariance, coefficient_covariance.contiguous()
                ),
            ).log_prob(coefficients.reshape(-1))
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), values)
    expected_likelihood = (
        torch.distributions.Normal(
            0.2 + coefficients[group, 0], torch.exp(-0.4 + coefficients[group, 1])
        )
        .log_prob(response)
        .sum()
    )
    expected = (
        expected_likelihood
        + group_density
        + torch.distributions.LKJCholesky(2, 1.0).log_prob(factor)
        + torch.distributions.HalfNormal(1.0).log_prob(scales).sum()
        + torch.distributions.Normal(0.0, 5.0).log_prob(torch.tensor([0.2, -0.4])).sum()
    )
    torch.testing.assert_close(recorded.log_joint.sum(), expected)
    torch.testing.assert_close(recorded.sites["y"].log_prob.sum(), expected_likelihood)
    for actual, oracle in zip(
        torch.autograd.grad(recorded.log_joint.sum(), (rho, scales), retain_graph=True),
        torch.autograd.grad(expected, (rho, scales)),
        strict=True,
    ):
        torch.testing.assert_close(actual, oracle, atol=1e-5, rtol=1e-5)


def test_quoted_bar_in_random_slope_contrast_is_not_a_block_separator(data):
    data = data.assign(factor=["a|b", "c", "a|b", "c"])
    formula = formula_from_data(
        'y ~ 1 + (1 + C(factor, Treatment(reference="a|b")) | g)', data
    )
    assert len(formula.random_terms) == 2
    assert len({term.correlation_id for term in formula.random_terms}) == 1


def test_shared_id_cannot_alias_different_grouping_factors(data):
    data = data.assign(other=["c", "d", "c", "d"])
    with pytest.raises(ValueError, match="ambiguous across groups"):
        formula_to_qvr("y ~ 1 + (1 | shared | g) + (1 | shared | other)", data=data)


@pytest.mark.parametrize("family", ["gaussian", "categorical"])
def test_correlation_prior_requires_lkj_and_accepts_named_override(data, family):
    if family == "categorical":
        data = data.assign(y=[0, 1, 2, 0])
    site = "cor_g" if family == "gaussian" else "cor_g_component1"
    with pytest.raises(ValueError, match="correlation priors must use LKJCholesky"):
        formula_to_qvr(
            "y ~ 1 + (1 + x | g)",
            data=data,
            family=family,
            priors={site: "Normal(0.0, 1.0)"},
        )
    source = formula_to_qvr(
        "y ~ 1 + (1 + x | g)",
        data=data,
        family=family,
        priors={site: "LKJCholesky(2.0)"},
    )
    assert "LKJCholesky(concentration=2.0)" in source
