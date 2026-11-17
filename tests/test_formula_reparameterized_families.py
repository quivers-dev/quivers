"""Additional released brms parameterizations preserve densities and predictions."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.inference import trace
from quivers.inference.mcmc.driver import MCMCResult

NAMES = ["discrete_weibull", "zero_inflated_asym_laplace", "negbinomial2"]


def _data(name):
    return pd.DataFrame(
        {
            "y": [0.0, -0.2, 0.4, 0.0]
            if name == "zero_inflated_asym_laplace"
            else [0.0, 1.0, 3.0, 0.0],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.3, 0.1, 0.5],
            "w": [0.0, 0.5, 1.0, 2.0],
        }
    )


def _score(name, mu, auxiliary, y):
    if name == "discrete_weibull":
        logq = mu.log()
        left = y.pow(auxiliary["shape"]) * logq
        right = (y + 1).pow(auxiliary["shape"]) * logq
        return left + torch.log(-torch.expm1(right - left))
    if name == "negbinomial2":
        shape = auxiliary["sigma"].reciprocal()
        return (
            torch.lgamma(y + shape)
            - torch.lgamma(shape)
            - torch.lgamma(y + 1)
            + shape * (shape.log() - (shape + mu).log())
            + y * (mu.log() - (shape + mu).log())
        )
    scale, quantile, zi = (auxiliary[k] for k in ("sigma", "quantile", "zi"))
    residual = (y - mu) / scale
    loss = residual * (quantile - (residual < 0).to(residual.dtype))
    continuous = quantile.log() + torch.log1p(-quantile) - scale.log() - loss
    return torch.where(y == 0, zi.log(), torch.log1p(-zi) + continuous)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("weighted", [False, True])
def test_distributional_scores_and_all_coefficients_match_independent_oracles(
    name, weighted
):
    frame = _data(name)
    family = families[name]
    parsed = formula_from_data("y ~ x", frame, weights="w" if weighted else None)
    auxiliary_formula = formula_from_data("y ~ z", frame)
    lens = FormulaToQVRModule(
        family, distributional={p.name: auxiliary_formula for p in family.aux_params}
    )
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
    source = module_to_source(module)
    program = loads(source).morphism
    values = {
        "intercept": torch.tensor(-0.2, requires_grad=True),
        "beta_x": torch.tensor(0.15, requires_grad=True),
    }
    auxiliary = {}
    z = torch.tensor(frame.z.to_numpy(), dtype=torch.float32)
    for i, parameter in enumerate(family.aux_params):
        a = values[f"dpar_{parameter.name}_intercept"] = torch.tensor(
            -0.4 + 0.2 * i, requires_grad=True
        )
        b = values[f"dpar_{parameter.name}_beta_z"] = torch.tensor(
            0.1 + 0.1 * i, requires_grad=True
        )
        predictor = a + b * z
        auxiliary[parameter.name] = (
            predictor.sigmoid()
            if parameter.name in {"zi", "quantile"}
            else predictor.exp()
        )
    y = torch.tensor(frame.y.to_numpy(), dtype=torch.float32)
    recorded = trace(
        program,
        torch.zeros(1, 1),
        lens.fixed_column_observations(parsed) | values | {"y": y},
    )
    eta = values["intercept"] + values["beta_x"] * torch.tensor(
        frame.x.to_numpy(), dtype=torch.float32
    )
    mu = (
        eta.sigmoid()
        if name == "discrete_weibull"
        else eta.exp()
        if name == "negbinomial2"
        else eta
    )
    expected_rows = _score(name, mu, auxiliary, y)
    weights = (
        torch.tensor(frame.w.to_numpy(), dtype=torch.float32)
        if weighted
        else torch.ones_like(y)
    )
    expected = (expected_rows * weights).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual.double(), expected.double(), rtol=3e-6, atol=3e-6)
    actual_grad = torch.autograd.grad(actual, tuple(values.values()), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, tuple(values.values()))
    for a, e in zip(actual_grad, expected_grad, strict=True):
        assert torch.isfinite(a)
        torch.testing.assert_close(a, e, rtol=5e-6, atol=5e-6)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_fits_use_both_scalar_and_regressed_auxiliaries(name, method):
    frame = _data(name)
    kwargs = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 2, "num_chains": 1}
    )
    if name == "discrete_weibull":
        # This deliberately short sampler smoke needs a controlled shape;
        # tiny powers can produce mathematically valid counts beyond Int64.
        kwargs.update(
            priors={
                "dpar_shape_intercept": "Normal(0.7, 0.1)",
                "dpar_shape_beta_z": "Normal(0.0, 0.1)",
            },
            initial_values={"dpar_shape_intercept": 0.7, "dpar_shape_beta_z": 0.0},
        )
    result = fit(
        "y ~ x",
        data=frame,
        family=name,
        distributional={families[name].aux_params[0].name: "~ z"},
        method=method,
        seed=7,
        **kwargs,
    )
    assert all(torch.isfinite(v).all() for v in result.draws(num_samples=2).values())
    assert torch.isfinite(result.posterior_predict(num_samples=2, seed=5)).all()


@pytest.mark.parametrize("name", NAMES)
def test_new_rows_match_independent_moments_and_sampling(name):
    fixed = (
        {"shape": 2.0}
        if name == "discrete_weibull"
        else {"sigma": 0.6}
        if name == "negbinomial2"
        else {"sigma": 0.6, "quantile": 0.3, "zi": 0.25}
    )
    result = fit(
        "y ~ x",
        data=_data(name),
        family=name,
        fixed_parameters=fixed,
        method="svi",
        num_steps=1,
        seed=13,
    )
    draws = 200
    result = result.with_(
        posterior=MCMCResult(
            samples={
                "intercept": torch.full((2, draws), -0.2),
                "beta_x": torch.full((2, draws), 0.1),
            },
            log_densities=torch.zeros(2, draws),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.long),
            num_warmup=0,
            num_samples=draws,
        )
    )
    fresh = pd.DataFrame({"x": np.linspace(-1, 2, 24)})
    eta = -0.2 + 0.1 * fresh.x.to_numpy()
    if name == "discrete_weibull":
        q = stats.logistic.cdf(eta)
        k = np.arange(1, 100)[:, None]
        expected = (q[None, :] ** (k**2)).sum(axis=0)
    elif name == "negbinomial2":
        expected = np.exp(eta)
    else:
        expected = 0.75 * (eta + 0.6 * (1 - 2 * 0.3) / (0.3 * (1 - 0.3)))
    state = torch.get_rng_state().clone()
    means = result.posterior_epred(new_data=fresh)
    sampled = result.posterior_predict(new_data=fresh, seed=17)
    assert torch.equal(state, torch.get_rng_state())
    assert means.shape == sampled.shape == (2, draws, len(fresh))
    np.testing.assert_allclose(means[0, 0].numpy(), expected, rtol=3e-6, atol=3e-6)
    centered = sampled.double() - torch.tensor(expected)
    standard_error = centered.std().item() / math.sqrt(centered.numel())
    assert abs(centered.mean().item()) < 6 * standard_error
    if name != "zero_inflated_asym_laplace":
        assert ((sampled >= 0) & (sampled == sampled.round())).all()
    else:
        frequency = sampled.eq(0).double().mean().item()
        assert frequency == pytest.approx(
            0.25, abs=6 * math.sqrt(0.25 * 0.75 / sampled.numel())
        )


def test_negbinomial2_prior_is_on_reciprocal_dispersion():
    frame = _data("negbinomial2")
    assert (
        get_prior("y ~ 1", data=frame, family="negbinomial2")["sigma"]
        == "Gamma(concentration=0.4, rate=0.3)"
    )
    parsed = formula_from_data("y ~ 1", frame)
    lens = FormulaToQVRModule(families["negbinomial2"])
    module, _ = lens.forward(parsed)
    sigma = torch.tensor(0.6, requires_grad=True)
    recorded = trace(
        Compiler(module).compile().morphism,
        torch.zeros(1, 1),
        {
            "intercept": torch.tensor(0.0),
            "sigma": sigma,
            "y": torch.tensor(frame.y.to_numpy(), dtype=torch.float32),
        },
    )
    actual = recorded.sites["sigma"].log_prob.sum()
    expected = torch.distributions.Gamma(0.4, 0.3).log_prob(sigma)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual, sigma)[0], torch.autograd.grad(expected, sigma)[0]
    )
