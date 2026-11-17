"""Distributional regressions agree with independently evaluated likelihoods."""

import math

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
    formula_to_qvr,
)
from quivers.inference.trace import trace


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.1, 1.0, -0.5, 0.7],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.0, 1.0, -1.0, 0.5],
            "g": ["a", "a", "b", "b"],
        }
    )


def test_gaussian_scale_regression_has_correct_density_and_gradients(data):
    location = formula_from_data("y ~ x", data)
    scale = formula_from_data("y ~ z", data)
    lens = FormulaToQVRModule(families["gaussian"], distributional={"sigma": scale})
    module, complement = lens.forward(location)
    assert lens.backward(module, complement) == location
    program = Compiler(module).compile().morphism
    source = module_to_source(module)
    assert "let sigma = exp(dpar_sigma_eta)" in source
    loads(source)
    values = {
        "intercept": torch.tensor(0.2),
        "beta_x": torch.tensor(-0.3),
        "dpar_sigma_intercept": torch.tensor(-0.2, requires_grad=True),
        "dpar_sigma_beta_z": torch.tensor(0.4, requires_grad=True),
    }
    observations = (
        lens.fixed_column_observations(location)
        | {"y": torch.tensor(data.y.to_numpy(), dtype=torch.float32)}
        | values
    )
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    mu = values["intercept"] + values["beta_x"] * observations["x"]
    log_scale = (
        values["dpar_sigma_intercept"]
        + values["dpar_sigma_beta_z"] * observations["dpar_sigma_z"]
    )
    expected = (
        -0.5 * ((observations["y"] - mu) * torch.exp(-log_scale)).square()
        - log_scale
        - 0.5 * math.log(2 * math.pi)
    ).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(
        actual, tuple(values.values())[2:], retain_graph=True
    )
    expected_gradient = torch.autograd.grad(expected, tuple(values.values())[2:])
    for actual_part, expected_part in zip(
        actual_gradient, expected_gradient, strict=True
    ):
        torch.testing.assert_close(actual_part, expected_part)


@pytest.mark.parametrize("reparameterize", ["centered", "noncentered"])
def test_distributional_group_effects_run_and_keep_coefficient_names(
    data, reparameterize
):
    result = fit(
        "y ~ x + (1 | g)",
        data=data,
        distributional={"sigma": "~ z + (1 | g)"},
        method="svi",
        num_steps=2,
        reparameterize=reparameterize,
        seed=13,
    )
    assert "dpar_sigma_intercept" in result.qvr_source
    assert "dpar_sigma_sigma_g_Intercept" in result.qvr_source
    assert "sigma <- HalfCauchy" not in result.qvr_source
    assert result.distributional["sigma"].group_levels == {"g": ("a", "b")}
    assert np.array_equal(
        result.observations["dpar_sigma_z"].numpy(), data.z.to_numpy()
    )
    loads(result.qvr_source)


def test_distributional_only_group_is_bound_and_seed_restored(data):
    initial = torch.get_rng_state().clone()
    result = fit(
        "y ~ 1",
        data=data,
        distributional={"sigma": "sigma ~ 1 + (1 | g)"},
        method="svi",
        num_steps=2,
        seed=31,
    )
    assert result.observations["g_idx"].tolist() == [0, 0, 1, 1]
    assert torch.equal(initial, torch.get_rng_state())


@pytest.mark.parametrize(
    "family, parameter",
    [
        ("beta", "phi"),
        ("gamma", "shape"),
        ("negative_binomial", "disp"),
        ("student_t", "nu"),
        ("zero_inflated_poisson", "zi"),
        ("zero_one_inflated_beta", "coi"),
    ],
)
def test_distributional_parameters_compile_for_every_auxiliary_family(
    data, family, parameter
):
    # The shared response is real-valued; each family scores a response in
    # its own support.
    if families[family].native.discrete:
        data = data.assign(y=[0, 2, 1, 3])
    elif families[family].response_interval != families["gaussian"].response_interval:
        data = data.assign(y=[0.1, 0.9, 0.5, 0.7])
    source = formula_to_qvr(
        "y ~ x", data=data, family=family, distributional={parameter: "~ z"}
    )
    assert f"dpar_{parameter}_beta_z" in source
    loads(source)


@pytest.mark.parametrize(
    "distributional, message",
    [
        ({"bogus": "~ x"}, "unknown distributional"),
        ({"sigma": "x"}, "requires '~'"),
        ({"sigma": "phi ~ x"}, "mismatched left-hand side"),
    ],
)
def test_invalid_distributional_formulas_fail_before_inference(
    data, distributional, message
):
    with pytest.raises(ValueError, match=message):
        formula_to_qvr("y ~ x", data=data, distributional=distributional)


@pytest.mark.parametrize(
    "link",
    [
        "identity",
        "logit",
        "log",
        "inverse",
        "probit",
        "probit_approx",
        "log1p",
        "cloglog",
        "cauchit",
        "sqrt",
        "inverse_square",
        "logm1",
        "softplus",
        "tan_half",
        "squareplus",
        "softit",
    ],
)
def test_link_density_and_autograd_agree_with_independent_functions(data, link):
    family = families["gaussian"].with_links(mu=link)
    parsed = formula_from_data("y ~ 1", data)
    lens = FormulaToQVRModule(family)
    module, _ = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    eta = torch.tensor(0.7, requires_grad=True)
    inverses = {
        "identity": lambda x: x,
        "logit": torch.sigmoid,
        "log": torch.exp,
        "inverse": torch.reciprocal,
        "probit": lambda x: torch.special.ndtr(x),
        "probit_approx": lambda x: torch.sigmoid(0.07056 * x**3 + 1.5976 * x),
        "log1p": torch.expm1,
        "cloglog": lambda x: -torch.expm1(-torch.exp(x)),
        "cauchit": lambda x: 0.5 + torch.atan(x) / math.pi,
        "sqrt": torch.square,
        "inverse_square": torch.rsqrt,
        "logm1": lambda x: 1 + torch.exp(x),
        "softplus": torch.nn.functional.softplus,
        "tan_half": lambda x: 2 * torch.atan(x),
        "squareplus": lambda x: (x + torch.sqrt(x * x + 4)) / 2,
        "softit": lambda x: (
            torch.nn.functional.softplus(x) / (1 + torch.nn.functional.softplus(x))
        ),
    }
    response = torch.tensor(data.y.to_numpy(), dtype=torch.float32)
    recorded = trace(
        program,
        torch.zeros(1, 1, dtype=torch.long),
        {"intercept": eta, "sigma": torch.tensor(1.0), "y": response},
    )
    actual = recorded.sites["y"].log_prob.sum()
    expected = (
        torch.distributions.Normal(inverses[link](eta), 1.0).log_prob(response).sum()
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, eta, retain_graph=True)[0],
        torch.autograd.grad(expected, eta)[0],
    )
    loads(module_to_source(module))


@pytest.mark.parametrize("eta_value", [-80.0, 0.0, 1000.0])
def test_softplus_scale_link_is_stable_and_differentiable_at_zero(data, eta_value):
    family = families["gaussian"].with_links(sigma="softplus")
    location = formula_from_data("y ~ 1", data)
    lens = FormulaToQVRModule(family, distributional={"sigma": location})
    module, _ = lens.forward(location)
    program = Compiler(module).compile().morphism
    eta = torch.tensor(eta_value, requires_grad=True)
    recorded = trace(
        program,
        torch.zeros(1, 1, dtype=torch.long),
        {
            "intercept": torch.tensor(0.0),
            "dpar_sigma_intercept": eta,
            "y": torch.tensor(data.y.to_numpy(), dtype=torch.float32),
        },
    )
    actual = recorded.sites["sigma"].value.reshape(-1)[0]
    expected = torch.nn.functional.softplus(eta)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, eta, retain_graph=True)[0],
        torch.autograd.grad(expected, eta)[0],
    )


@pytest.mark.parametrize(
    "family_name", ["lognormal", "exponential", "geometric", "weibull", "von_mises"]
)
def test_added_families_use_brms_parameterization(data, family_name):
    response_values = (
        [0.5, 1.0, 2.0, 3.0] if family_name != "geometric" else [0, 1, 2, 3]
    )
    data = data.assign(y=response_values)
    parsed = formula_from_data("y ~ 1", data)
    lens = FormulaToQVRModule(families[family_name])
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
    program = Compiler(module).compile().morphism
    eta = torch.tensor(0.3, requires_grad=True)
    mu = torch.exp(eta)
    parameters = {
        "intercept": eta,
        "sigma": torch.tensor(0.7),
        "shape": torch.tensor(1.8),
        "kappa": torch.tensor(2.0),
    }
    distributions = {
        "lognormal": lambda: torch.distributions.LogNormal(eta, parameters["sigma"]),
        "exponential": lambda: torch.distributions.Exponential(1 / mu),
        # Count-law arithmetic promotes the computed predictor probability;
        # the predictor itself and its coefficient gradients retain Float32.
        "geometric": lambda: torch.distributions.Geometric(
            probs=(1 / (1 + mu)).double()
        ),
        "weibull": lambda: torch.distributions.Weibull(
            mu / torch.exp(torch.lgamma(1 + 1 / parameters["shape"])),
            parameters["shape"],
        ),
        "von_mises": lambda: torch.distributions.VonMises(
            2 * torch.atan(eta), parameters["kappa"]
        ),
    }
    response = torch.tensor(response_values, dtype=torch.float32)
    recorded = trace(
        program, torch.zeros(1, 1, dtype=torch.long), {"y": response} | parameters
    )
    actual = recorded.sites["y"].log_prob.sum()
    expected = distributions[family_name]().log_prob(response).sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, eta, retain_graph=True)[0],
        torch.autograd.grad(expected, eta)[0],
    )
    loads(module_to_source(module))


def test_link_validation_and_conflicting_direct_prior(data):
    with pytest.raises(ValueError, match="unknown family parameters"):
        families["gaussian"].with_links(phi="log")
    with pytest.raises(ValueError, match="unknown links"):
        families["gaussian"].with_links(mu="unknown")
    with pytest.raises(ValueError, match="require the softmax"):
        families["categorical"].with_links(mu="identity")
    with pytest.raises(ValueError, match="must name their coefficients"):
        formula_to_qvr(
            "y ~ x",
            data=data,
            distributional={"sigma": "~ z"},
            priors={"sigma": "HalfNormal(1.0)"},
        )


def test_beta_binomial_density_with_varying_trials_and_distributional_precision(data):
    data = data.assign(y=[1, 2, 1, 6], trials=[3, 5, 6, 7])
    parsed = formula_from_data("y ~ 1", data)
    precision = formula_from_data("y ~ z", data)
    lens = FormulaToQVRModule(
        families["beta_binomial"],
        binomial_trials="trials",
        distributional={"phi": precision},
    )
    module, _ = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    eta = torch.tensor(0.3, requires_grad=True)
    log_precision = torch.tensor(0.5, requires_grad=True)
    response = torch.tensor(data.y.to_numpy(), dtype=torch.float32)
    trials = torch.tensor(data.trials.to_numpy(), dtype=torch.long)
    observations = lens.fixed_column_observations(parsed) | {
        "y": response,
        "trials": trials,
        "intercept": eta,
        "dpar_phi_intercept": log_precision,
        "dpar_phi_beta_z": torch.tensor(0.2),
    }
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    precision = torch.exp(
        log_precision.double() + 0.2 * observations["dpar_phi_z"].double()
    )
    alpha = torch.sigmoid(eta.double()) * precision
    beta = (1 - torch.sigmoid(eta.double())) * precision
    response = response.double()
    trials = trials.double()
    expected = (
        torch.lgamma(trials + 1)
        - torch.lgamma(response + 1)
        - torch.lgamma(trials - response + 1)
        + torch.lgamma(response + alpha)
        + torch.lgamma(trials - response + beta)
        - torch.lgamma(trials + alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
    ).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(
        actual, (eta, log_precision), retain_graph=True
    )
    expected_gradients = torch.autograd.grad(expected, (eta, log_precision))
    for actual_part, expected_part in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(actual_part, expected_part)
    emitted = formula_to_qvr(
        "y ~ 1",
        data=data,
        family="beta_binomial",
        binomial_trials="trials",
        distributional={"phi": "~ z"},
    )
    loads(emitted)
    result = fit(
        "y ~ 1",
        data=data,
        family="beta_binomial",
        binomial_trials="trials",
        distributional={"phi": "~ z"},
        method="svi",
        num_steps=2,
    )
    assert result.observations["trials"].tolist() == [3, 5, 6, 7]


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_cumulative_link_selects_the_correct_ordered_likelihood(data, link):
    data = data.assign(y=[0, 1, 2, 1])
    family = families["cumulative"].with_links(mu=link)
    parsed = formula_from_data("y ~ 1", data)
    lens = FormulaToQVRModule(family)
    module, _ = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    eta = torch.tensor(0.3, requires_grad=True)
    recorded = trace(
        program,
        torch.zeros(1, 1, dtype=torch.long),
        {
            "y": torch.tensor(data.y.to_numpy()),
            "intercept": eta,
            "cutpoint_log_spacing": torch.tensor([0.0]),
        },
    )
    boundaries = torch.tensor([-0.5, 0.5]) - eta
    cdf = (
        torch.sigmoid(boundaries) if link == "logit" else torch.special.ndtr(boundaries)
    )
    probs = torch.stack((cdf[0], cdf[1] - cdf[0], 1 - cdf[1]))
    expected = torch.log(probs)[torch.tensor(data.y.to_numpy())].sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, eta, retain_graph=True)[0],
        torch.autograd.grad(expected, eta)[0],
    )
    loads(module_to_source(module))


def test_trial_counts_keep_integer_precision_and_reject_bad_support(data):
    from quivers.formulas._fit import _binomial_trial_tensor

    exact = 2**24 + 1
    data = data.assign(trials=[exact, 2, 3, 4])
    actual = _binomial_trial_tensor(data, "trials", n_obs=4)
    assert actual.dtype == torch.long
    assert actual[0].item() == exact
    for counts in ([True] * 4, [1.5] * 4, [float("nan")] * 4, [float(2**63)] * 4):
        with pytest.raises(ValueError, match="binomial_trials"):
            _binomial_trial_tensor(data.assign(trials=counts), "trials", n_obs=4)


def test_repeated_link_updates_preserve_ordinal_link_selection():
    family = families["cumulative"].with_links(mu="probit").with_links()
    assert family.location_link.name == "probit"
    assert family.observe_family == "OrderedProbit"


def test_distributional_design_transforms_and_factor_columns_are_bound_once(data):
    result = fit(
        "y ~ x",
        data=data,
        distributional={"sigma": "~ poly(z, 2) + g"},
        method="svi",
        num_steps=2,
        seed=29,
    )
    predictor = result.distributional["sigma"]
    columns = [column for column in predictor.fixed_columns if not column.is_intercept]
    assert len(columns) == 3
    for column in columns:
        torch.testing.assert_close(
            result.observations[f"dpar_sigma_{column.qvr_name}"],
            torch.tensor(column.data.copy(), dtype=torch.float32),
        )
    loads(result.qvr_source)


@pytest.mark.parametrize("sampler", ["hmc", "nuts"])
def test_distributional_regression_runs_through_hamiltonian_samplers(data, sampler):
    result = fit(
        "y ~ 1",
        data=data,
        distributional={"sigma": "~ 1"},
        method=sampler,
        num_warmup=2,
        num_samples=3,
        num_chains=1,
        seed=7,
    )
    assert result.posterior.samples["dpar_sigma_intercept"].shape[:2] == (1, 3)
    assert torch.isfinite(result.posterior.log_densities).all()


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_fixed_quantile_regression_fits_without_quantile_latent(data, method):
    counts = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 3, "num_chains": 1}
    )
    result = fit(
        "y ~ x + (1 | g)",
        data=data,
        family="asym_laplace",
        fixed_parameters={"quantile": 0.8},
        method=method,
        seed=7,
        **counts,
    )
    assert "let quantile = 0.8" in result.qvr_source
    assert "sample quantile" not in result.qvr_source
    assert result.fixed_parameters == {"quantile": 0.8}
    loads(result.qvr_source)


def test_quantile_regression_and_auxiliary_links_match_check_loss(data):
    formula = formula_from_data("y ~ x", data)
    auxiliary = formula_from_data("y ~ z", data)
    lens = FormulaToQVRModule(
        families["asym_laplace"],
        distributional={"sigma": auxiliary},
        fixed_parameters={"quantile": 0.8},
    )
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    values = {
        "intercept": torch.tensor(0.3, requires_grad=True),
        "beta_x": torch.tensor(-0.2, requires_grad=True),
        "dpar_sigma_intercept": torch.tensor(-0.3, requires_grad=True),
        "dpar_sigma_beta_z": torch.tensor(0.1, requires_grad=True),
    }
    inputs = lens.fixed_column_observations(formula)
    response = torch.tensor(data.y.to_numpy(), dtype=torch.float32)
    recorded = trace(
        program, torch.zeros(1, 1, dtype=torch.long), inputs | values | {"y": response}
    )
    mu = values["intercept"] + values["beta_x"] * inputs["x"]
    scale = (
        values["dpar_sigma_intercept"]
        + values["dpar_sigma_beta_z"] * inputs["dpar_sigma_z"]
    ).exp()
    residual = (response - mu) / scale
    expected = (
        math.log(0.8 * 0.2) - scale.log() - residual * (0.8 - (residual < 0).float())
    ).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    for result, oracle in zip(
        torch.autograd.grad(actual, tuple(values.values()), retain_graph=True),
        torch.autograd.grad(expected, tuple(values.values())),
        strict=True,
    ):
        torch.testing.assert_close(result, oracle)


@pytest.mark.parametrize(
    "fixed, distributional, priors, message",
    [
        ({"sigma": -1.0}, None, None, "strictly positive"),
        ({"sigma": True}, None, None, "finite real"),
        ({"missing": 1.0}, None, None, "unknown fixed"),
        ({"sigma": float("nan")}, None, None, "finite real"),
        ({"sigma": 0.8}, {"sigma": "~ z"}, None, "cannot have distributional"),
        ({"sigma": 0.8}, None, {"sigma": "HalfNormal(1.0)"}, "unused prior"),
    ],
)
def test_fixed_parameters_refuse_invalid_or_conflicting_contracts(
    data, fixed, distributional, priors, message
):
    with pytest.raises(ValueError, match=message):
        formula_to_qvr(
            "y ~ x",
            data=data,
            fixed_parameters=fixed,
            distributional=distributional,
            priors=priors,
        )


@pytest.mark.parametrize("quantile", [0.0, 1.0, -0.5, 1.5])
def test_fixed_quantile_refuses_boundary_or_invalid_probabilities(data, quantile):
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        formula_to_qvr(
            "y ~ x",
            data=data,
            family="asym_laplace",
            fixed_parameters={"quantile": quantile},
        )


@pytest.mark.parametrize(
    "family, auxiliary", [("exgaussian", "beta"), ("inverse_gaussian", "shape")]
)
def test_response_time_formula_likelihood_and_gradients_match_mean_parameterization(
    data, family, auxiliary
):
    import scipy.stats

    data = data.assign(y=[0.1, 0.8, 1.5, 2.0])
    formula = formula_from_data("y ~ x", data)
    scale = formula_from_data("y ~ z", data)
    fixed = {"sigma": 0.7} if family == "exgaussian" else {}
    lens = FormulaToQVRModule(
        families[family], distributional={auxiliary: scale}, fixed_parameters=fixed
    )
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    program = Compiler(module).compile().morphism
    inputs = lens.fixed_column_observations(formula)
    values = {
        "intercept": torch.tensor(0.4, requires_grad=True),
        "beta_x": torch.tensor(0.1, requires_grad=True),
        f"dpar_{auxiliary}_intercept": torch.tensor(-0.2, requires_grad=True),
        f"dpar_{auxiliary}_beta_z": torch.tensor(0.1, requires_grad=True),
    }
    response = torch.tensor(data.y.to_numpy(), dtype=torch.float32)
    recorded = trace(
        program, torch.zeros(1, 1, dtype=torch.long), inputs | values | {"y": response}
    )
    location = values["intercept"] + values["beta_x"] * inputs["x"]
    parameter = (
        values[f"dpar_{auxiliary}_intercept"]
        + values[f"dpar_{auxiliary}_beta_z"] * inputs[f"dpar_{auxiliary}_z"]
    ).exp()
    if family == "exgaussian":
        mu = location
        xi = mu - parameter
        standardized = (response - xi) / 0.7
        ratio = 0.7 / parameter
        expected = (
            -parameter.log()
            + 0.5 * ratio.square()
            - ratio * standardized
            + torch.special.log_ndtr(standardized - ratio)
        ).sum()
        scipy_density = (
            scipy.stats.exponnorm(
                parameter.detach().numpy() / 0.7, loc=xi.detach().numpy(), scale=0.7
            )
            .logpdf(response.numpy())
            .sum()
        )
    else:
        mu = location.exp()
        expected = (
            0.5 * (parameter.log() - math.log(2 * math.pi) - 3 * response.log())
            - parameter * (response - mu).square() / (2 * mu.square() * response)
        ).sum()
        scipy_density = (
            scipy.stats.invgauss(
                mu.detach().numpy() / parameter.detach().numpy(),
                scale=parameter.detach().numpy(),
            )
            .logpdf(response.numpy())
            .sum()
        )
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    assert actual.item() == pytest.approx(scipy_density, abs=2e-5)
    for actual_gradient, expected_gradient in zip(
        torch.autograd.grad(actual, tuple(values.values()), retain_graph=True),
        torch.autograd.grad(expected, tuple(values.values())),
        strict=True,
    ):
        torch.testing.assert_close(actual_gradient, expected_gradient)
    loads(module_to_source(module))


@pytest.mark.parametrize("family", ["exgaussian", "inverse_gaussian"])
@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_response_time_formulas_fit_with_all_samplers(data, family, method):
    counts = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 3, "num_chains": 1}
    )
    result = fit(
        "y ~ x + (1 | g)",
        data=data.assign(y=[0.1, 0.8, 1.5, 2.0]),
        family=family,
        method=method,
        seed=7,
        **counts,
    )
    assert families[family].observe_family in result.qvr_source
