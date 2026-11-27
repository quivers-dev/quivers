"""COM-Poisson formulas retain rate semantics across both runtimes and predictions."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import gammaln, logsumexp

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.inference import trace
from quivers.inference.mcmc.driver import MCMCResult
from quivers.qiec.program_runtime import run_program


def _data():
    return pd.DataFrame(
        {
            "y": [0, 1, 3, 2],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.3, 0.1, 0.5],
            "w": [0.0, 0.5, 1.0, 2.0],
        }
    )


def _series_torch(lr, nu, y):
    counts = torch.arange(1000, dtype=torch.double).unsqueeze(-1)
    weights = counts * lr - nu * torch.lgamma(counts + 1)
    return y * lr - nu * torch.lgamma(y + 1) - torch.logsumexp(weights, dim=0)


def _mean(lr, nu):
    counts = np.arange(10000, dtype=float)
    weights = counts * lr - nu * gammaln(counts + 1)
    return np.sum(counts * np.exp(weights - logsumexp(weights)))


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("distributional", [False, True])
def test_com_formula_source_density_and_all_parameter_gradients(
    weighted, distributional
):
    data = _data()
    formula = formula_from_data("y ~ x", data, weights="w" if weighted else None)
    predictors = {"shape": formula_from_data("y ~ z", data)} if distributional else None
    lens = FormulaToQVRModule(families["com_poisson"], distributional=predictors)
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    source = module_to_source(module)
    assert "COMPoisson(log_rate=eta, dispersion=shape)" in source
    assert "let mu = exp(eta)" not in source
    loads(source)
    values = {
        "intercept": torch.tensor(0.2, requires_grad=True),
        "beta_x": torch.tensor(-0.15, requires_grad=True),
    }
    if distributional:
        values |= {
            "dpar_shape_intercept": torch.tensor(0.3, requires_grad=True),
            "dpar_shape_beta_z": torch.tensor(0.2, requires_grad=True),
        }
    else:
        values["shape"] = torch.tensor(1.4, requires_grad=True)
    inputs = (
        lens.fixed_column_observations(formula)
        | values
        | {"y": torch.tensor(data.y.to_numpy())}
    )
    recorded = trace(
        Compiler(module).compile().morphism, torch.zeros(1, 1, dtype=torch.long), inputs
    )
    lr = values["intercept"].double() + values["beta_x"].double() * inputs["x"].double()
    nu = (
        (
            values["dpar_shape_intercept"].double()
            + values["dpar_shape_beta_z"].double() * inputs["dpar_shape_z"].double()
        ).exp()
        if distributional
        else values["shape"].double()
    )
    scores = _series_torch(lr, nu, inputs["y"].double())
    if weighted:
        scores = scores * inputs["formula_weights"]
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, scores.sum(), atol=1e-7, rtol=1e-7)
    for got, expected in zip(
        torch.autograd.grad(actual, tuple(values.values()), retain_graph=True),
        torch.autograd.grad(scores.sum(), tuple(values.values())),
        strict=True,
    ):
        torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-6)
    # Check the complete emitted program, including latent priors, through QIEC.
    checked = lower_qvr_to_qiec(parse(source))
    reference_data = {
        name: tuple(value.reshape(-1).tolist()) if value.ndim else value.item()
        for name, value in inputs.items()
    }
    reference = run_program(
        checked,
        "model",
        data={key: value for key, value in reference_data.items() if key not in values},
        sites={key: value.item() for key, value in values.items()},
    )
    assert float(reference.log_joint) == pytest.approx(
        recorded.log_joint.sum().item(), abs=2e-6
    )


def _recorded(result, values, draws=3):
    return result.with_(
        posterior=MCMCResult(
            samples={
                name: value.expand(2, draws, *value.shape).clone()
                for name, value in values.items()
            },
            log_densities=torch.zeros(2, draws),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.long),
            num_warmup=0,
            num_samples=draws,
        )
    )


def _fitted(*, distributional=False, weighted=False, family="com_poisson"):
    return fit(
        "y ~ x",
        data=_data(),
        family=family,
        distributional={"shape": "~ z"} if distributional else None,
        fixed_parameters={} if distributional else {"shape": 1.4},
        weights="w" if weighted else None,
        method="svi",
        num_steps=1,
        seed=19,
        initial_values={"intercept": 0.5, "beta_x": 0.0},
    )


@pytest.mark.parametrize("distributional", [False, True])
@pytest.mark.parametrize("fresh", [False, True])
def test_com_predictions_distinguish_rate_from_response_expectation(
    distributional, fresh
):
    values = {"intercept": torch.tensor(0.2), "beta_x": torch.tensor(-0.15)}
    if distributional:
        values |= {
            "dpar_shape_intercept": torch.tensor(0.3),
            "dpar_shape_beta_z": torch.tensor(0.2),
        }
    result = _recorded(
        _fitted(distributional=distributional, weighted=True), values, draws=8
    )
    data = (
        pd.DataFrame({"x": [2.0, -1.0, 0.0], "z": [0.5, 0.2, -0.3]})
        if fresh
        else _data()
    )
    kwargs = {"new_data": data} if fresh else {}
    eta = 0.2 - 0.15 * torch.tensor(data.x.to_numpy(), dtype=torch.float32)
    nu = (
        (0.3 + 0.2 * torch.tensor(data.z.to_numpy(), dtype=torch.float32)).exp()
        if distributional
        else torch.full_like(eta, 1.4)
    )
    torch.testing.assert_close(
        result.posterior_linpred(**kwargs), eta.expand(2, 8, len(data))
    )
    torch.testing.assert_close(
        result.posterior_linpred(transform=True, **kwargs),
        eta.exp().expand(2, 8, len(data)),
    )
    means = torch.tensor(
        [
            _mean(lr, power)
            for lr, power in zip(
                eta.double().tolist(), nu.double().tolist(), strict=True
            )
        ],
        dtype=torch.double,
    )
    torch.testing.assert_close(
        result.posterior_epred(**kwargs),
        means.expand(2, 8, len(data)),
        atol=1e-7,
        rtol=1e-7,
    )
    assert not torch.allclose(means, eta.double().exp())
    predictive = result.posterior_predict(**kwargs, seed=7)
    assert predictive.shape == (2, 8, len(data)) and predictive.dtype == torch.int64
    assert (predictive >= 0).all()
    torch.testing.assert_close(predictive, result.posterior_predict(**kwargs, seed=7))
    if distributional:
        expected = 0.3 + 0.2 * torch.tensor(data.z.to_numpy(), dtype=torch.float32)
        torch.testing.assert_close(
            result.posterior_linpred(parameter="shape", **kwargs),
            expected.expand(2, 8, len(data)),
        )
        torch.testing.assert_close(
            result.posterior_linpred(parameter="shape", transform=True, **kwargs),
            nu.expand(2, 8, len(data)),
        )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_com_fit_methods_with_stable_user_priors_and_released_prior_inspection(method):
    kwargs = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 2, "num_chains": 1}
    )
    result = fit(
        "y ~ x",
        data=_data(),
        family="com_poisson",
        priors={
            "intercept": "Normal(0.0, 0.2)",
            "beta_x": "Normal(0.0, 0.1)",
            "shape": "Gamma(20.0, 20.0)",
        },
        method=method,
        seed=7,
        initial_values={"intercept": 0.0, "beta_x": 0.0, "shape": 1.0},
        **kwargs,
    )
    assert all(
        torch.isfinite(value).all() for value in result.draws(num_samples=2).values()
    )
    assert result.posterior_epred(num_samples=2).shape[-1] == 4
    priors = get_prior("y ~ x", data=_data(), family="com_poisson")
    assert "Gamma" in priors["shape"] and "0.01" in priors["shape"]


@pytest.mark.parametrize("link", ["identity", "sqrt", "softplus", "squareplus"])
def test_com_alternative_location_links_compile_with_rate_semantics(link):
    family = families["com_poisson"].with_links(mu=link)
    result = _recorded(
        _fitted(family=family),
        {"intercept": torch.tensor(0.5), "beta_x": torch.tensor(0.0)},
    )
    rate = {
        "identity": 0.5,
        "sqrt": 0.25,
        "softplus": math.log1p(math.exp(0.5)),
        "squareplus": (0.5 + math.sqrt(4.25)) / 2,
    }[link]
    torch.testing.assert_close(
        result.posterior_linpred(transform=True), torch.full((2, 3, 4), rate)
    )
    expected = _mean(math.log(rate), 1.4)
    torch.testing.assert_close(
        result.posterior_epred(),
        torch.full((2, 3, 4), expected, dtype=torch.double),
        atol=1e-7,
        rtol=1e-7,
    )


def test_com_default_scalar_prior_initialization_can_fit():
    result = fit(
        "y ~ x",
        data=_data(),
        family="com_poisson",
        method="svi",
        num_steps=2,
        seed=7,
        initial_values={"intercept": 0.0, "beta_x": 0.0, "shape": 1.0},
    )
    assert torch.isfinite(result.posterior_epred(num_samples=2)).all()


def test_log_link_generated_program_preserves_overflowing_rate():
    data = pd.DataFrame({"y": [100, 99]})
    formula = formula_from_data("y ~ 1", data)
    lens = FormulaToQVRModule(
        families["com_poisson"], fixed_parameters={"shape": 1000.0}
    )
    module, _ = lens.forward(formula)
    source = module_to_source(module)
    intercept = 1000 * math.log(100)
    inputs = {
        "y": torch.tensor([100, 99]),
        "intercept": torch.tensor(intercept, dtype=torch.double),
    }
    recorded = trace(
        loads(source).morphism, torch.zeros(1, 1, dtype=torch.long), inputs
    )
    reference = run_program(
        lower_qvr_to_qiec(parse(source)),
        "model",
        data={"y": (100, 99)},
        sites={"intercept": intercept},
    )
    assert math.isfinite(reference.log_joint)
    assert recorded.log_joint.sum().item() == pytest.approx(
        reference.log_joint.item(), abs=1e-7
    )
    assert "mu" not in recorded.sites
    assert torch.isfinite(recorded.sites["y"].log_prob).all()


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_com_group_and_exposure_likelihood_gradients_match_independent_oracle(
    parameterization,
):
    data = _data().assign(g=["a", "b", "a", "b"], e=[1.0, 2.0, 0.5, 3.0])
    formula = formula_from_data("y ~ x + offset(log(e)) + (1|g)", data, weights="w")
    lens = FormulaToQVRModule(
        families["com_poisson"],
        fixed_parameters={"shape": 1.4},
        reparameterize=parameterization,
    )
    module, _ = lens.forward(formula)
    values = {
        "intercept": torch.tensor(0.2, requires_grad=True),
        "beta_x": torch.tensor(-0.15, requires_grad=True),
        "sigma_g_Intercept": torch.tensor(0.3, requires_grad=True),
    }
    group_name = "alpha_g" if parameterization == "centered" else "z_g_Intercept"
    values[group_name] = torch.tensor([0.5, -0.5], requires_grad=True)
    groups = torch.tensor(formula.group_indices["g"])
    inputs = (
        lens.fixed_column_observations(formula)
        | values
        | {"g_idx": groups, "y": torch.tensor(data.y.to_numpy())}
    )
    recorded = trace(
        Compiler(module).compile().morphism, torch.zeros(1, 1, dtype=torch.long), inputs
    )
    effect = (
        values[group_name]
        if parameterization == "centered"
        else values["sigma_g_Intercept"] * values[group_name]
    )
    eta = (
        inputs["formula_offset_0"]
        + values["intercept"]
        + values["beta_x"] * inputs["x"]
        + effect[groups]
    ).double()
    expected = (
        _series_torch(eta, torch.tensor(1.4).double(), inputs["y"].double())
        * inputs["formula_weights"]
    ).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    targets = tuple(
        value
        for name, value in values.items()
        if parameterization == "noncentered" or name != "sigma_g_Intercept"
    )
    for got, want in zip(
        torch.autograd.grad(actual, targets, retain_graph=True),
        torch.autograd.grad(expected, targets),
        strict=True,
    ):
        torch.testing.assert_close(got, want, atol=3e-6, rtol=3e-6)
