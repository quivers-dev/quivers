"""Native formula predictions preserve sampling laws, row independence, and RNG state."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import Family, families, fit
from quivers.qiec.families import FAMILIES
from quivers.inference.mcmc.driver import MCMCResult


def _posterior(result, values, chains=2, draws=100):
    return result.with_(
        posterior=MCMCResult(
            samples={
                name: value.expand(chains, draws, *value.shape).clone()
                for name, value in values.items()
            },
            log_densities=torch.zeros(chains, draws),
            acceptance_rates=torch.ones(chains),
            divergence_counts=torch.zeros(chains, dtype=torch.long),
            num_warmup=0,
            num_samples=draws,
        )
    )


def test_gaussian_shared_parameters_produce_independent_rows_and_correct_moments(
    monkeypatch,
):
    rows, draws = 40, 100
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": np.zeros(rows)}),
        fixed_parameters={"sigma": 0.7},
        method="svi",
        num_steps=1,
    )
    result = _posterior(result, {"intercept": torch.tensor(0.3)}, draws=draws)

    def reject_recompile(*args, **kwargs):
        raise AssertionError("predictive sampling must reuse the fitted program")

    monkeypatch.setattr("quivers.formulas._fit.Compiler.compile", reject_recompile)
    samples = result.posterior_predict(seed=71)
    assert samples.shape == (2, draws, rows)
    residual = samples - 0.3
    count = residual.numel()
    assert abs(residual.mean().item()) < 5 * 0.7 / math.sqrt(count)
    assert abs(residual.square().mean().item() - 0.7**2) < 5 * 0.7**2 * math.sqrt(
        2 / count
    )
    covariance = (residual[..., 0] * residual[..., 1]).mean()
    assert abs(covariance.item()) < 5 * 0.7**2 / math.sqrt(2 * draws)
    assert not torch.equal(samples[..., 0], samples[..., 1])
    assert not samples.requires_grad


def test_bernoulli_shared_probability_has_correct_moment_and_row_independence():
    rows, draws = 30, 80
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": np.zeros(rows)}),
        family="bernoulli",
        method="svi",
        num_steps=1,
    )
    probability = 0.3
    result = _posterior(
        result,
        {"intercept": torch.tensor(math.log(probability / (1 - probability)))},
        draws=draws,
    )
    samples = result.posterior_predict(seed=39)
    assert torch.all((samples == 0) | (samples == 1))
    count = samples.numel()
    assert abs(samples.double().mean().item() - probability) < 5 * math.sqrt(
        probability * (1 - probability) / count
    )
    centered = samples.double() - probability
    covariance = (centered[..., 0] * centered[..., 1]).mean()
    assert abs(covariance.item()) < 5 * probability * (1 - probability) / math.sqrt(
        2 * draws
    )


def test_guide_and_response_noise_use_independent_streams():
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0.0]}),
        fixed_parameters={"sigma": 1.0},
        method="svi",
        num_steps=1,
    )
    with torch.no_grad():
        result.posterior.loc_intercept.zero_()
        result.posterior.log_scale_intercept.zero_()
    count, seed = 400, 29
    means = result.draws(num_samples=count, seed=seed)["intercept"].reshape(-1)
    predictions = result.posterior_predict(num_samples=count, seed=seed).reshape(-1)
    noise = predictions - means
    # Under an independent N(0,1) posterior mean and N(0,1) response
    # residual, E[Y^2]=2 and E[mean * residual]=0. Reusing a seed for
    # both streams on one row would instead make Y=2*mean exactly.
    assert abs(predictions.square().mean().item() - 2) < 5 * math.sqrt(8 / count)
    assert abs((means * noise).mean().item()) < 5 / math.sqrt(count)


@pytest.mark.parametrize("family", ["gaussian", "categorical", "hurdle_gamma"])
def test_seed_chain_axes_and_fixed_draw_weight_invariance(family):
    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 0] if family == "categorical" else [0.0, 0.2, 0.7, 0.0],
            "w": [0.0, 0.5, 1.0, 2.0],
        }
    )
    result = fit(
        "y ~ 1",
        data=frame,
        family=family,
        weights="w",
        method="svi",
        num_steps=1,
        seed=13,
    )
    before = torch.get_rng_state().clone()
    first = result.posterior_predict(num_samples=5, seed=23)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(first, result.posterior_predict(num_samples=5, seed=23))
    assert not torch.equal(first, result.posterior_predict(num_samples=5, seed=24))
    changed = result.with_(
        observations=dict(result.observations) | {"formula_weights": torch.zeros(4)}
    )
    assert torch.equal(first, changed.posterior_predict(num_samples=5, seed=23))
    assert first.shape == (1, 5, 4)


def test_binomial_prediction_uses_per_row_trials():
    trials = torch.tensor([1, 2, 5, 10])
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0, 1, 2, 4], "trials": trials.numpy()}),
        family="binomial",
        binomial_trials="trials",
        method="svi",
        num_steps=1,
    )
    result = _posterior(result, {"intercept": torch.tensor(0.0)}, draws=100)
    samples = result.posterior_predict(seed=47)
    assert samples.dtype == torch.long
    assert torch.all((samples >= 0) & (samples <= trials))
    expected = trials.double() * 0.5
    standard_error = (trials.double() * 0.25 / 200).sqrt()
    assert torch.all(
        (samples.double().mean((0, 1)) - expected).abs() < 5 * standard_error
    )


@pytest.mark.parametrize("family", ["categorical", "cumulative"])
def test_category_predictions_return_valid_codes_not_probability_vectors(family):
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0, 1, 2, 0]}),
        family=family,
        method="svi",
        num_steps=1,
    )
    samples = result.posterior_predict(num_samples=10, seed=17)
    assert samples.shape == (1, 10, 4)
    assert samples.dtype == torch.long
    assert torch.all((samples >= 0) & (samples < len(result.category_labels)))


@pytest.mark.parametrize(
    "family", ["zero_inflated_beta", "hurdle_gamma", "hurdle_lognormal"]
)
def test_inflated_prediction_includes_correct_zero_atom_and_response_mean(family):
    count, rows, probability = 150, 20, 0.25
    frame = pd.DataFrame({"y": [0.2] * rows})
    if family == "zero_inflated_beta":
        fixed = {"phi": 8.0, "zi": probability}
        intercept, component_mean, component_var = 0.0, 0.5, 0.25 / 9
    elif family == "hurdle_gamma":
        fixed = {"shape": 2.0, "hu": probability}
        intercept, component_mean, component_var = math.log(0.5), 0.5, 0.5**2 / 2
    else:
        fixed = {"sigma": 0.4, "hu": probability}
        intercept = math.log(0.5) - 0.4**2 / 2
        component_mean, component_var = 0.5, (math.exp(0.4**2) - 1) * 0.5**2
    result = fit(
        "y ~ 1",
        data=frame,
        family=family,
        fixed_parameters=fixed,
        method="svi",
        num_steps=1,
    )
    result = _posterior(
        result, {"intercept": torch.tensor(intercept)}, chains=1, draws=count
    )
    samples = result.posterior_predict(seed=59)
    total = samples.numel()
    assert abs(samples.eq(0).double().mean().item() - probability) < 5 * math.sqrt(
        probability * (1 - probability) / total
    )
    mean = (1 - probability) * component_mean
    variance = (1 - probability) * component_var + probability * (
        1 - probability
    ) * component_mean**2
    assert abs(samples.mean().item() - mean) < 5 * math.sqrt(variance / total)


@pytest.mark.parametrize("family", sorted(families))
def test_every_formula_family_has_native_training_row_predictions(family):
    specification = families[family]
    native_family = FAMILIES[specification.observe_family]
    if native_family.discrete:
        response = (
            [0, 1, 1, 0]
            if specification.trials_parameter is not None
            or native_family.finite_support is not None
            else [0, 1, 2, 0]
        )
    else:
        response = [0.1, 0.2, 0.7, 0.4]
    vector = (
        specification.response is not None
        and specification.response.kind != "categorical"
    )
    frame = pd.DataFrame({"y": response, "choice": [0, 1, 0, 1]})
    if vector:
        events = (
            [[0, 0, 0], [1, 2, 0], [2, 1, 4], [0, 3, 1]]
            if specification.trials_parameter
            else [[0.2, 0.3, 0.5], [0.3, 0.2, 0.5], [0.1, 0.3, 0.6], [0.4, 0.3, 0.3]]
        )
        frame = pd.DataFrame(events, columns=["a", "b", "c"])
    result = fit(
        "cbind(a,b,c) ~ 1" if vector else "y ~ 1",
        data=frame,
        family=family,
        decision="choice" if family == "wiener" else None,
        method="svi",
        num_steps=1,
        seed=13,
    )
    predicted = result.posterior_predict(num_samples=2, seed=19)
    assert predicted.shape == ((1, 2, 4, 3) if vector else (1, 2, 4))
    assert torch.isfinite(predicted).all()


@pytest.mark.parametrize("count", [0, -1, True])
def test_invalid_predictive_draw_count_preserves_rng(count):
    result = fit(
        "y ~ 1", data=pd.DataFrame({"y": [0.0, 0.5]}), method="svi", num_steps=1
    )
    state = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="positive integer"):
        result.posterior_predict(num_samples=count)
    assert torch.equal(state, torch.get_rng_state())


def test_native_sampling_failure_restores_rng(monkeypatch):
    result = fit(
        "y ~ 1", data=pd.DataFrame({"y": [0.0, 0.5]}), method="svi", num_steps=1
    )

    def fail_after_sampling(self, input):
        torch.rand(11)
        raise RuntimeError("response sampler failed")

    monkeypatch.setattr(
        "quivers.continuous.plate.VectorisedObserve.rsample", fail_after_sampling
    )
    state = torch.get_rng_state().clone()
    with pytest.raises(RuntimeError, match="response sampler failed"):
        result.posterior_predict(num_samples=2)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("family", ["cauchy", "von_mises"])
def test_predictions_do_not_require_a_defined_response_expectation(family):
    selected = (
        Family(
            name="cauchy",
            location_link=families["gaussian"].location_link,
            observe_family="Cauchy",
            extra_observe_args=("sigma",),
            aux_params=families["gaussian"].aux_params,
        )
        if family == "cauchy"
        else family
    )
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0.1, 0.2, 0.7, 0.4]}),
        family=selected,
        method="svi",
        num_steps=1,
    )
    with pytest.raises((TypeError, ValueError), match="angular|undefined"):
        result.posterior_epred(num_samples=2)
    samples = result.posterior_predict(num_samples=5, seed=23)
    assert samples.shape == (1, 5, 4)
    assert torch.isfinite(samples).all()
    if family == "von_mises":
        assert torch.all(samples.abs() <= math.pi)


def test_mcmc_prediction_caps_draws_per_chain():
    result = fit(
        "y ~ 1",
        data=pd.DataFrame({"y": [0.0, 0.5]}),
        fixed_parameters={"sigma": 1.0},
        method="svi",
        num_steps=1,
    )
    result = _posterior(result, {"intercept": torch.tensor(0.0)}, draws=3)
    assert result.posterior_predict(num_samples=2).shape == (2, 2, 2)
    assert result.posterior_predict(num_samples=20).shape == (2, 3, 2)


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_correlated_groups_offsets_and_distributional_scale_match_response_law(
    parameterization,
):
    frame = pd.DataFrame(
        {
            "y": [0.1, 0.8, -0.2, 0.4],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.5, 0.0, 0.7],
            "g": ["a", "a", "b", "b"],
        }
    )
    result = fit(
        "y ~ x + offset(z) + (1 + x | g)",
        data=frame,
        distributional={"sigma": "~ z"},
        method="svi",
        num_steps=1,
        reparameterize=parameterization,
    )
    factor = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    scales = torch.tensor([0.4, 0.7])
    standardized = torch.tensor([[0.2, 1.0], [-0.5, 0.3]])
    coefficients = standardized @ factor.T * scales
    values = {
        "intercept": torch.tensor(0.3),
        "beta_x": torch.tensor(-0.2),
        "cor_g": factor,
        "sigma_g_Intercept": scales[0],
        "sigma_g_x": scales[1],
        "dpar_sigma_intercept": torch.tensor(-0.4),
        "dpar_sigma_beta_z": torch.tensor(0.1),
    }
    values |= (
        {"z_g_Intercept": standardized[:, 0], "z_g_x": standardized[:, 1]}
        if parameterization == "noncentered"
        else {"alpha_g": coefficients[:, 0], "beta_g_x": coefficients[:, 1]}
    )
    result = _posterior(result, values, draws=150)
    x, z = (torch.tensor(frame[name].to_numpy()).float() for name in ("x", "z"))
    groups = torch.tensor([0, 0, 1, 1])
    mean = 0.3 - 0.2 * x + z + coefficients[groups, 0] + coefficients[groups, 1] * x
    scale = (-0.4 + 0.1 * z).exp()
    residual = (result.posterior_predict(seed=53) - mean) / scale
    count = residual.shape[0] * residual.shape[1]
    assert torch.all(residual.mean((0, 1)).abs() < 5 / math.sqrt(count))
    assert torch.all(
        (residual.square().mean((0, 1)) - 1).abs() < 5 * math.sqrt(2 / count)
    )
