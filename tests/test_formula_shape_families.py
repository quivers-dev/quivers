"""Shape-family regressions match independent densities and response moments."""

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import special, stats

from quivers.dsl import Compiler, loads
from quivers.dsl.emit import module_to_source
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
    links,
)
from quivers.inference import trace
from quivers.inference.mcmc.driver import MCMCResult


CASES = {
    "skew_normal": ({"sigma": 0.6, "alpha": -2.0}, 0.3),
    "shifted_lognormal": ({"sigma": 0.6, "ndt": 0.03}, -0.5),
    "frechet": ({"nu": 3.5}, math.log(0.8)),
    "gen_extreme_value": ({"sigma": 0.6, "xi": 0.0}, 0.3),
}


def _oracle(name, mu, parameters, values):
    if name == "skew_normal":
        alpha = parameters["alpha"]
        delta = alpha / np.sqrt(1 + alpha**2)
        scale = parameters["sigma"] / np.sqrt(1 - 2 * delta**2 / np.pi)
        location = mu - scale * delta * np.sqrt(2 / np.pi)
        return stats.skewnorm.logpdf(values, alpha, loc=location, scale=scale)
    if name == "shifted_lognormal":
        return stats.lognorm.logpdf(
            values, parameters["sigma"], loc=parameters["ndt"], scale=np.exp(mu)
        )
    if name == "frechet":
        return stats.invweibull.logpdf(
            values, parameters["nu"], scale=mu / special.gamma(1 - 1 / parameters["nu"])
        )
    return stats.genextreme.logpdf(
        values, -parameters["xi"], loc=mu, scale=parameters["sigma"]
    )


@pytest.mark.parametrize("name", CASES)
def test_formula_density_and_location_gradient_match_scipy(name):
    fixed, intercept = CASES[name]
    frame = pd.DataFrame({"y": [0.1, 0.2, 0.5, 1.5], "x": [-1.0, 0.0, 1.0, 2.0]})
    parsed = formula_from_data("y ~ x", frame)
    lens = FormulaToQVRModule(families[name], fixed_parameters=fixed)
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
    source = module_to_source(module)
    program = loads(source).morphism
    beta = torch.tensor(0.1, requires_grad=True)
    observations = lens.fixed_column_observations(parsed) | {
        "y": torch.tensor(frame.y.to_numpy(), dtype=torch.float32),
        "intercept": torch.tensor(intercept),
        "beta_x": beta,
    }
    actual = trace(program, torch.zeros(1, 1), observations).sites["y"].log_prob
    eta = intercept + 0.1 * frame.x.to_numpy()
    mu = np.exp(eta) if name == "frechet" else eta
    expected = _oracle(name, mu, fixed, frame.y.to_numpy())
    np.testing.assert_allclose(
        actual.detach().numpy().reshape(-1), [expected.sum()], rtol=3e-6, atol=3e-6
    )

    def oracle(coefficient):
        predictor = intercept + coefficient * frame.x.to_numpy()
        location = np.exp(predictor) if name == "frechet" else predictor
        return _oracle(name, location, fixed, frame.y.to_numpy()).sum()

    epsilon = 1e-5
    gradient = torch.autograd.grad(actual.sum(), beta)[0]
    assert gradient.item() == pytest.approx(
        (oracle(0.1 + epsilon) - oracle(0.1 - epsilon)) / (2 * epsilon),
        rel=3e-5,
        abs=3e-5,
    )


@pytest.mark.parametrize("name", CASES)
def test_fitted_new_data_predictions_use_response_means_and_laws(name):
    fixed, intercept = CASES[name]
    frame = pd.DataFrame({"y": [0.1, 0.2, 0.5, 1.5], "x": [-1.0, 0.0, 1.0, 2.0]})
    result = fit(
        "y ~ x",
        data=frame,
        family=name,
        fixed_parameters=fixed,
        method="svi",
        num_steps=1,
        seed=4,
    )
    draws = 150
    result = result.with_(
        posterior=MCMCResult(
            samples={
                "intercept": torch.full((2, draws), intercept),
                "beta_x": torch.full((2, draws), 0.1),
            },
            log_densities=torch.zeros(2, draws),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.long),
            num_warmup=0,
            num_samples=draws,
        )
    )
    fresh = pd.DataFrame({"x": np.linspace(-1, 2, 20)})
    eta = intercept + 0.1 * fresh.x.to_numpy()
    if name == "frechet":
        expected = np.exp(eta)
    elif name == "shifted_lognormal":
        expected = fixed["ndt"] + np.exp(eta + fixed["sigma"] ** 2 / 2)
    elif name == "gen_extreme_value":
        expected = eta + np.euler_gamma * fixed["sigma"]
    else:
        expected = eta
    state = torch.get_rng_state().clone()
    means = result.posterior_epred(new_data=fresh)
    samples = result.posterior_predict(new_data=fresh, seed=19)
    assert torch.equal(state, torch.get_rng_state())
    assert samples.shape == means.shape == (2, draws, len(fresh))
    np.testing.assert_allclose(means[0, 0].numpy(), expected, rtol=2e-6, atol=2e-6)
    # Pool each row after subtracting its analytic mean. Fixed posterior
    # parameters leave only independent response noise in this check.
    centered = samples.double() - torch.tensor(expected)
    standard_error = centered.std().item() / math.sqrt(centered.numel())
    assert abs(centered.mean().item()) < 6 * standard_error
    if name in {"frechet", "shifted_lognormal"}:
        assert bool((samples > fixed.get("ndt", 0)).all())


def test_scalar_shift_prior_uses_minimum_response_and_fits_small_times():
    frame = pd.DataFrame({"y": [0.001, 0.003, 0.005, 0.002]})
    priors = get_prior("y ~ 1", data=frame, family="shifted_lognormal")
    assert priors["ndt"] == "Uniform(low=0.0, high=0.001)"
    result = fit(
        "y ~ 1",
        data=frame,
        family="shifted_lognormal",
        method="svi",
        num_steps=3,
        seed=5,
    )
    assert torch.isfinite(result.draws(num_samples=3)["ndt"]).all()
    assert bool((result.draws(num_samples=3)["ndt"] < 0.001).all())
    assert torch.isfinite(result.posterior_epred(num_samples=2)).all()


def test_frechet_shape_prior_is_normalized_and_constrained_above_one():
    frame = pd.DataFrame({"y": [0.1, 0.3, 0.5]})
    prior = get_prior("y ~ 1", data=frame, family="frechet")["nu"]
    assert "Restrict" in prior and "Gamma" in prior
    parsed = formula_from_data("y ~ 1", frame)
    lens = FormulaToQVRModule(families["frechet"])
    module, _ = lens.forward(parsed)
    program = Compiler(module).compile().morphism
    observations = lens.fixed_column_observations(parsed) | {
        "y": torch.tensor([0.1, 0.3, 0.5]),
        "intercept": torch.tensor(0.0),
        "nu": torch.tensor(3.0),
    }
    recorded = trace(program, torch.zeros(1, 1), observations)
    expected = stats.gamma.logpdf(3, 2, scale=10) - stats.gamma.logsf(1, 2, scale=10)
    assert recorded.sites["nu"].log_prob.item() == pytest.approx(expected, abs=1e-6)
    result = fit(
        "y ~ 1", data=frame, family="frechet", method="svi", num_steps=2, seed=17
    )
    assert bool((result.draws(num_samples=5)["nu"] > 1).all())


def test_frechet_prior_override_is_conditioned_on_the_mean_domain():
    frame = pd.DataFrame({"y": [0.1, 0.3, 0.5]})
    prior = get_prior(
        "y ~ 1", data=frame, family="frechet", priors={"nu": "Normal(0.0, 2.0)"}
    )["nu"]
    assert prior == "Restrict(base=Normal(loc=0.0, scale=2.0), low=1.0)"
    result = fit(
        "y ~ 1",
        data=frame,
        family="frechet",
        priors={"nu": "Normal(0.0, 2.0)"},
        method="svi",
        num_steps=2,
    )
    assert bool((result.draws(num_samples=4)["nu"] > 1).all())


def test_frechet_unsafe_regression_link_is_refused():
    frechet = families["frechet"]
    with pytest.raises(ValueError, match=r"admissible links are \['logm1'\]"):
        frechet.with_links(nu="identity")
    nu = frechet.aux_params[0].with_(link=links["identity"])
    with pytest.raises(ValueError, match=r"must exceed 1.*Link\.image"):
        FormulaToQVRModule(frechet.with_(aux_params=(nu,)))


@pytest.mark.parametrize("shape", [1.01, 1.7, 3.0])
def test_frechet_restricted_override_intersects_domain_and_normalizes(shape):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.double)
    try:
        frame = pd.DataFrame({"y": [0.8, 1.4]})
        parsed = formula_from_data("y ~ 1", frame)
        lens = FormulaToQVRModule(
            families["frechet"],
            user_priors={
                "intercept": "Normal(0.0, 1.0)",
                "nu": "Restrict(base=Normal(loc=0.0, scale=2.0), low=0.0)",
            },
        )
        module, _ = lens.forward(parsed)
        source = loads(module_to_source(module))
        nu = torch.tensor(shape, requires_grad=True)
        recorded = trace(
            source.morphism,
            torch.zeros(1, 1),
            {
                "intercept": torch.tensor(0.0),
                "nu": nu,
                "y": torch.tensor(frame.y.to_numpy()),
            },
        )
        expected_prior = stats.norm.logpdf(shape, scale=2) - stats.norm.logsf(
            1, scale=2
        )
        assert recorded.sites["nu"].log_prob.item() == pytest.approx(
            expected_prior, abs=2e-12
        )
        assert torch.autograd.grad(recorded.sites["nu"].log_prob.sum(), nu)[
            0
        ].item() == pytest.approx(-shape / 4, abs=2e-12)
        expected = (
            expected_prior
            + stats.norm.logpdf(0)
            + stats.invweibull.logpdf(
                frame.y.to_numpy(), shape, scale=1 / special.gamma(1 - 1 / shape)
            ).sum()
        )
        assert recorded.log_joint.sum().item() == pytest.approx(expected, abs=2e-11)
        reference = source.run(
            "model", data={"y": tuple(frame.y)}, sites={"intercept": 0.0, "nu": shape}
        )
        assert reference.log_joint == pytest.approx(expected, abs=2e-11)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    "name,fixed",
    [
        ("frechet", {"nu": 1.0}),
        ("shifted_lognormal", {"ndt": -0.1}),
        ("shifted_lognormal", {"ndt": 0.2}),
    ],
)
def test_invalid_fixed_parameter_support_is_rejected_before_fitting(name, fixed):
    with pytest.raises(ValueError):
        fit(
            "y ~ 1",
            data=pd.DataFrame({"y": [0.1, 0.3]}),
            family=name,
            fixed_parameters=fixed,
            method="svi",
            num_steps=1,
        )
