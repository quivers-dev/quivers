"""Zero-inflated beta regression uses the exact zero-atom/Beta mixture."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.continuous._zip_hurdle import ZeroOneInflatedBeta
from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import formula_to_qvr
from quivers.formulas import FormulaToQVRModule, families, fit, formula_from_data
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.trace import trace
from quivers.transpile import UnsupportedConstruct, available_targets, transpile
from quivers.dsl.parser import parse


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.0, 0.2, 0.7, 0.0],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.3, 0.0, 0.5],
        }
    )


def _interior_log_density(value, mean, precision):
    alpha = mean * precision
    beta = (1 - mean) * precision
    return (
        torch.lgamma(alpha + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + (alpha - 1) * value.log()
        + (beta - 1) * torch.log1p(-value)
    )


def test_native_zero_only_mixture_has_exact_density_mean_and_gradients():
    mu = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    zi = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    coi = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    distribution = ZeroOneInflatedBeta(mu, phi, zi, coi, validate_args=True)
    response = torch.tensor([0.0, 0.2, 0.7], dtype=torch.float64)
    actual = distribution.log_prob(response).sum()
    expected = (
        zi.log()
        + 2 * torch.log1p(-zi)
        + _interior_log_density(response[1:], mu, phi).sum()
    )
    torch.testing.assert_close(actual, expected)
    variables = (mu, phi, zi)
    for observed, oracle in zip(
        torch.autograd.grad(actual, variables, retain_graph=True),
        torch.autograd.grad(expected, variables),
        strict=True,
    ):
        assert torch.isfinite(observed)
        torch.testing.assert_close(observed, oracle)
    # The selected zero atom has log(1-coi), whose derivative at zero is -1.
    torch.testing.assert_close(
        torch.autograd.grad(actual, coi)[0], torch.tensor(-1.0, dtype=torch.float64)
    )
    torch.testing.assert_close(distribution.mean, (1 - zi) * mu)
    assert torch.isneginf(distribution.log_prob(torch.tensor(1.0)))


@pytest.mark.parametrize("zi, response", [(0.0, 0.2), (1.0, 0.0)])
def test_native_zero_only_mixture_boundary_gradients_are_finite(zi, response):
    mu = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    inflation = torch.tensor(zi, dtype=torch.float64, requires_grad=True)
    distribution = ZeroOneInflatedBeta(
        mu, phi, inflation, torch.zeros(()), validate_args=True
    )
    score = distribution.log_prob(torch.tensor(response, dtype=torch.float64))
    expected = (
        _interior_log_density(torch.tensor(response), mu, phi)
        if zi == 0
        else torch.zeros((), dtype=torch.float64)
    )
    torch.testing.assert_close(score, expected)
    gradient = torch.autograd.grad(score, (mu, phi, inflation))
    assert all(torch.isfinite(value) for value in gradient)
    if zi == 1:
        torch.testing.assert_close(gradient[0], torch.zeros_like(mu))
        torch.testing.assert_close(gradient[1], torch.zeros_like(phi))
    impossible = torch.tensor(0.0 if zi == 0 else 0.2)
    assert torch.isneginf(distribution.log_prob(impossible))


def test_native_zero_only_sampling_has_correct_atom_and_response_moments():
    mu, phi, zi = 0.3, 8.0, 0.25
    distribution = ZeroOneInflatedBeta(
        *(torch.tensor(value, dtype=torch.float64) for value in (mu, phi, zi, 0.0))
    )
    count = 20000
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(832)
        samples = distribution.sample(torch.Size([count]))
    assert torch.all((samples >= 0) & (samples < 1))
    observed_zero = samples.eq(0).double().mean()
    assert abs(observed_zero.item() - zi) < 5 * math.sqrt(zi * (1 - zi) / count)
    expected_mean = (1 - zi) * mu
    variance = (1 - zi) * mu * (1 - mu) / (phi + 1) + zi * (1 - zi) * mu**2
    assert abs(samples.mean().item() - expected_mean) < 5 * math.sqrt(variance / count)


def test_distributional_zero_inflation_and_precision_match_independent_density(data):
    location = formula_from_data("y ~ x", data)
    lens = FormulaToQVRModule(
        families["zero_inflated_beta"],
        distributional={
            "phi": formula_from_data("y ~ z", data),
            "zi": formula_from_data("y ~ z", data),
        },
    )
    module, complement = lens.forward(location)
    assert lens.backward(module, complement) == location
    source = module_to_source(module)
    assert "coi=0.0" in source
    assert "sample coi" not in source and "sample zoi" not in source
    loads(source)
    program = Compiler(module).compile().morphism
    values = {
        name: torch.tensor(value, requires_grad=True)
        for name, value in {
            "intercept": -0.3,
            "beta_x": 0.2,
            "dpar_phi_intercept": 1.0,
            "dpar_phi_beta_z": -0.1,
            "dpar_zi_intercept": -0.8,
            "dpar_zi_beta_z": 0.3,
        }.items()
    }
    observations = (
        lens.fixed_column_observations(location)
        | values
        | {"y": torch.tensor(data.y.to_numpy(), dtype=torch.float32)}
    )
    recorded = trace(program, torch.zeros(1, 1, dtype=torch.long), observations)
    x = observations["x"]
    z = observations["dpar_phi_z"]
    mu = (values["intercept"] + values["beta_x"] * x).sigmoid()
    phi = (values["dpar_phi_intercept"] + values["dpar_phi_beta_z"] * z).exp()
    zi = (values["dpar_zi_intercept"] + values["dpar_zi_beta_z"] * z).sigmoid()
    zero = observations["y"].eq(0)
    expected = (
        zi[zero].log().sum()
        + torch.log1p(-zi[~zero]).sum()
        + _interior_log_density(observations["y"][~zero], mu[~zero], phi[~zero]).sum()
    )
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected)
    for observed, oracle in zip(
        torch.autograd.grad(actual, tuple(values.values()), retain_graph=True),
        torch.autograd.grad(expected, tuple(values.values())),
        strict=True,
    ):
        assert torch.isfinite(observed)
        torch.testing.assert_close(observed, oracle)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_zero_inflated_beta_fitting_prior_and_posterior_mean(data, method):
    result = fit(
        "y ~ x",
        data=data,
        family="zero_inflated_beta",
        method=method,
        fixed_parameters={"phi": 8.0, "zi": 0.25},
        num_steps=2 if method == "svi" else None,
        num_samples=2 if method != "svi" else None,
        num_warmup=2 if method != "svi" else None,
        num_chains=1 if method != "svi" else None,
        seed=13,
    )
    result = result.with_(
        posterior=MCMCResult(
            samples={
                "intercept": torch.full((1, 2), -0.3),
                "beta_x": torch.full((1, 2), 0.2),
            },
            log_densities=torch.zeros(1, 2),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1),
            num_warmup=0,
            num_samples=2,
        )
    )
    mu = torch.tensor(-0.3 + 0.2 * data.x.to_numpy(), dtype=torch.float32).sigmoid()
    torch.testing.assert_close(
        result.posterior_linpred(transform=True), mu.expand(1, 2, 4)
    )
    torch.testing.assert_close(result.posterior_epred(), (0.75 * mu).expand(1, 2, 4))
    assert set(result.draws()) == {"intercept", "beta_x"}
    assert "coi=0.0" in result.qvr_source


@pytest.mark.parametrize("value", [1.0, -0.1, 1.1, np.inf, -np.inf])
def test_zero_inflated_beta_response_domain_checked_before_fitting(data, value):
    with pytest.raises(ValueError, match=r"finite response values in \[0, 1\)"):
        fit(
            "y ~ x",
            data=data.assign(y=[value, 0.2, 0.3, 0.0]),
            family="zero_inflated_beta",
            method="svi",
            num_steps=1,
        )


@pytest.mark.parametrize("zi", [0.0, 1.0])
def test_fixed_inflation_endpoints_fit_without_invalid_inactive_gradients(data, zi):
    frame = data.assign(y=[0.1, 0.2, 0.3, 0.7] if zi == 0 else np.zeros(4))
    result = fit(
        "y ~ x",
        data=frame,
        family="zero_inflated_beta",
        method="svi",
        fixed_parameters={"zi": zi, "phi": 8.0},
        num_steps=2,
        seed=17,
    )
    conditional_mean = result.posterior_linpred(transform=True, num_samples=3, seed=2)
    response_mean = result.posterior_epred(num_samples=3, seed=2)
    torch.testing.assert_close(response_mean, conditional_mean * (1 - zi))


def test_zero_inflation_prior_inspection_excludes_absent_one_atom(data):
    from quivers.formulas import get_prior

    priors = get_prior("y ~ x", data=data, family="zero_inflated_beta")
    assert set(priors) == {"intercept", "beta_x", "phi", "zi"}
    with pytest.raises(ValueError, match="unused prior overrides"):
        get_prior(
            "y ~ x",
            data=data,
            family="zero_inflated_beta",
            priors={"coi": "Beta(2.0, 2.0)"},
        )


@pytest.mark.parametrize("target", sorted(available_targets()))
def test_external_backend_refusal_identifies_native_zero_beta_family(data, target):
    source = formula_to_qvr(
        "y ~ 1",
        data=data,
        family="zero_inflated_beta",
        fixed_parameters={"phi": 8.0, "zi": 0.25},
    )
    with pytest.raises(UnsupportedConstruct) as caught:
        transpile(parse(source), target=target)
    assert any("ZeroOneInflatedBeta" in kind for kind in caught.value.kinds)
