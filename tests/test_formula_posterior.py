"""Formula-aware posterior summaries preserve native prediction semantics."""

import pandas as pd
import pytest
import torch

from quivers.formulas import fit, families, Family
from quivers.inference.mcmc.driver import MCMCResult
from quivers.continuous._zip_hurdle import ZeroOneInflatedBeta


@pytest.fixture
def data():
    return pd.DataFrame(
        {
            "y": [0.1, 0.8, -0.2, 0.4],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.5, 0.0, 0.7],
            "g": ["a", "a", "b", "b"],
        }
    )


def _recorded(result, values, chains=2, count=3):
    samples = {
        name: value.expand(chains, count, *value.shape).clone()
        for name, value in values.items()
    }
    posterior = MCMCResult(
        samples=samples,
        log_densities=torch.zeros(chains, count),
        acceptance_rates=torch.ones(chains),
        divergence_counts=torch.zeros(chains),
        num_warmup=0,
        num_samples=count,
    )
    return result.with_(posterior=posterior)


def test_population_coefficients_linpred_epred_and_chain_axes(data):
    result = fit("y ~ x", data=data, method="svi", num_steps=1)
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.3),
            "beta_x": torch.tensor(-0.2),
            "sigma": torch.tensor(0.7),
        },
    )
    expected = 0.3 - 0.2 * torch.tensor(data.x.to_numpy(), dtype=torch.float32)
    assert result.draws()["beta_x"].shape == (2, 3)
    assert result.fixed_effects()["Intercept"].shape == (2, 3)
    assert result.fixed_effects()["x"].eq(-0.2).all()
    assert result.random_effects() == {}
    assert result.coefficients() == {}
    torch.testing.assert_close(result.posterior_linpred(), expected.expand(2, 3, 4))
    torch.testing.assert_close(result.posterior_epred(), expected.expand(2, 3, 4))
    assert result.posterior_linpred(num_samples=1).shape == (2, 1, 4)


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_correlated_native_group_bindings_reconstruct_coefficients(
    data, parameterization
):
    result = fit(
        "y ~ x + (1 + x | g)",
        data=data,
        method="svi",
        num_steps=1,
        reparameterize=parameterization,
    )
    factor = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    scale = torch.tensor([0.4, 0.7])
    z = torch.tensor([[0.2, 1.0], [-0.5, 0.3]])
    coefficients = z @ factor.T * scale
    values = {
        "intercept": torch.tensor(0.3),
        "beta_x": torch.tensor(-0.2),
        "sigma": torch.tensor(0.7),
        "cor_g": factor,
        "sigma_g_Intercept": scale[0],
        "sigma_g_x": scale[1],
    }
    values |= (
        {"z_g_Intercept": z[:, 0], "z_g_x": z[:, 1]}
        if parameterization == "noncentered"
        else {"group_coefficients_g": coefficients}
    )
    result = _recorded(result, values)
    groups = result.random_effects()["g"]
    assert result.group_levels == {"g": ("a", "b")}
    torch.testing.assert_close(groups["Intercept"], coefficients[:, 0].expand(2, 3, 2))
    torch.testing.assert_close(groups["x"], coefficients[:, 1].expand(2, 3, 2))
    combined = result.coefficients()["g"]
    torch.testing.assert_close(combined["Intercept"], groups["Intercept"] + 0.3)
    torch.testing.assert_close(combined["x"], groups["x"] - 0.2)
    indices = torch.tensor([0, 0, 1, 1])
    x = torch.tensor(data.x.to_numpy(), dtype=torch.float32)
    expected = 0.3 - 0.2 * x + coefficients[indices, 0] + coefficients[indices, 1] * x
    torch.testing.assert_close(result.posterior_linpred(), expected.expand(2, 3, 4))


def test_distributional_coefficients_linpred_and_inverse_link(data):
    result = fit(
        "y ~ x", data=data, distributional={"sigma": "~ z"}, method="svi", num_steps=1
    )
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.3),
            "beta_x": torch.tensor(-0.2),
            "dpar_sigma_intercept": torch.tensor(-0.4),
            "dpar_sigma_beta_z": torch.tensor(0.1),
        },
    )
    expected = -0.4 + 0.1 * torch.tensor(data.z.to_numpy(), dtype=torch.float32)
    torch.testing.assert_close(
        result.posterior_linpred(parameter="sigma"), expected.expand(2, 3, 4)
    )
    torch.testing.assert_close(
        result.posterior_linpred(parameter="sigma", transform=True),
        expected.exp().expand(2, 3, 4),
    )
    assert result.fixed_effects(parameter="sigma")["z"].eq(0.1).all()


def test_binomial_epred_is_count_and_transformed_linpred_is_probability(data):
    result = fit(
        "y ~ x",
        data=data.assign(y=[0, 1, 2, 0], trials=[1, 2, 3, 4]),
        family="binomial",
        binomial_trials="trials",
        method="svi",
        num_steps=1,
    )
    result = _recorded(
        result, {"intercept": torch.tensor(0.3), "beta_x": torch.tensor(-0.2)}
    )
    eta = 0.3 - 0.2 * torch.tensor(data.x.to_numpy(), dtype=torch.float32)
    torch.testing.assert_close(
        result.posterior_linpred(transform=True), eta.sigmoid().expand(2, 3, 4)
    )
    torch.testing.assert_close(
        result.posterior_epred(),
        (eta.double().sigmoid() * torch.arange(1, 5)).expand(2, 3, 4),
    )


@pytest.mark.parametrize("link", ["logit", "probit"])
def test_ordinal_epred_returns_threshold_probabilities_and_labels(data, link):
    result = fit(
        "y ~ x",
        data=data.assign(y=[0, 1, 2, 0]),
        family=families["cumulative"].with_links(mu=link),
        method="svi",
        num_steps=1,
    )
    draws = result.draws(num_samples=2, seed=3)
    assert result.category_labels == (0, 1, 2)
    probability = result.posterior_epred(num_samples=2, seed=3)
    assert probability.shape == (1, 2, 4, 3)
    torch.testing.assert_close(probability.sum(-1), torch.ones(1, 2, 4))
    # Cutpoints are centered cumulative positive spacings in the native program.
    spacing = draws["cutpoint_log_spacing"].squeeze(-1).exp()
    cutpoint = torch.stack((-0.5 * spacing, 0.5 * spacing), dim=-1)
    eta = draws["intercept"][..., None] + draws["beta_x"][..., None] * torch.tensor(
        data.x.to_numpy(), dtype=torch.float32
    )
    offsets = cutpoint[..., None, :] - eta[..., None]
    cdf = offsets.sigmoid() if link == "logit" else torch.special.ndtr(offsets)
    expected = torch.cat(
        (cdf[..., :1], cdf[..., 1:] - cdf[..., :1], 1 - cdf[..., 1:]), dim=-1
    )
    torch.testing.assert_close(probability, expected)
    with pytest.raises(ValueError, match="thresholds"):
        result.posterior_linpred(transform=True, num_samples=1)


def test_categorical_full_logits_reference_probabilities_and_group_coefficients(data):
    result = fit(
        "y ~ 1 + (1 | p | g)",
        data=data.assign(y=[0, 1, 2, 0]),
        family="categorical",
        method="svi",
        num_steps=1,
    )
    factor = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    scale = torch.tensor([0.4, 0.7])
    z = torch.tensor([[0.2, 1.0], [-0.5, 0.3]])
    coefficient = z @ factor.T * scale
    intercept = torch.tensor([0.3, -0.2])
    result = _recorded(
        result,
        {
            "intercept": intercept,
            "sigma_g_Intercept": scale,
            "cor_shared_p": factor,
            "z_g_Intercept": z.reshape(-1),
        },
    )
    expected = torch.cat(
        (torch.zeros(4, 1), intercept + coefficient[torch.tensor([0, 0, 1, 1])]), dim=-1
    )
    torch.testing.assert_close(result.posterior_linpred(), expected.expand(2, 3, 4, 3))
    torch.testing.assert_close(
        result.posterior_epred(), expected.softmax(-1).expand(2, 3, 4, 3)
    )
    torch.testing.assert_close(
        result.random_effects()["g"]["Intercept"], coefficient.expand(2, 3, 2, 2)
    )
    torch.testing.assert_close(
        result.coefficients()["g"]["Intercept"],
        (coefficient + intercept).expand(2, 3, 2, 2),
    )
    assert result.category_labels == (0, 1, 2)


def test_guide_draws_and_summaries_preserve_rng_and_reproduce_seed(data):
    result = fit("y ~ x + (1 | g)", data=data, method="svi", num_steps=1)
    state = torch.random.get_rng_state()
    a = result.posterior_epred(num_samples=2, seed=31)
    b = result.posterior_epred(num_samples=2, seed=31)
    torch.testing.assert_close(a, b)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert result.draws(num_samples=2)["z_g_Intercept"].shape == (1, 2, 2)


def test_zero_one_inflated_beta_mean_matches_mixture_components():
    mu = torch.tensor([0.2, 0.7], requires_grad=True)
    zoi = torch.tensor([0.3, 0.8], requires_grad=True)
    coi = torch.tensor([0.4, 0.1], requires_grad=True)
    law = ZeroOneInflatedBeta(mu, torch.tensor(3.0), zoi, coi)
    expected = zoi * (coi * 1.0 + (1 - coi) * 0.0) + (1 - zoi) * mu
    torch.testing.assert_close(law.mean, expected)
    gradients = torch.autograd.grad(law.mean.sum(), (mu, zoi, coi))
    for actual, expected_gradient in zip(
        gradients, (1 - zoi, coi - mu, zoi), strict=True
    ):
        torch.testing.assert_close(actual, expected_gradient)


@pytest.mark.parametrize("count", [0, -1, True])
def test_posterior_summary_refuses_invalid_count(data, count):
    result = fit("y ~ x", data=data, method="svi", num_steps=1)
    with pytest.raises(ValueError, match="positive integer"):
        result.posterior_epred(num_samples=count)


def test_undefined_mean_refusal_and_missing_auxiliary_predictor(data):
    cauchy = Family(
        name="cauchy",
        location_link=families["gaussian"].location_link,
        observe_family="Cauchy",
        observe_args=("mu", "sigma"),
        aux_params=families["gaussian"].aux_params,
    )
    result = fit("y ~ x", data=data, family=cauchy, method="svi", num_steps=1)
    with pytest.raises(ValueError, match="undefined or nonfinite"):
        result.posterior_epred(num_samples=1)
    scale = result.posterior_linpred(parameter="sigma", num_samples=1)
    assert scale.shape == (1, 1, len(data))
    assert torch.isfinite(scale).all()


def test_mixture_epred_weights_component_means_and_quantile_epred_is_mean(data):
    mixture = fit("y ~ 1", data=data, family="mixture", method="svi", num_steps=1)
    mixture = _recorded(
        mixture,
        {
            "dpar_mu1_intercept": torch.tensor(0.3),
            "mixture_intercept_spacing": torch.tensor([1.2]),
            "sigma1": torch.tensor(0.5),
            "sigma2": torch.tensor(0.9),
            "theta": torch.tensor([0.35, 0.65]),
        },
    )
    means = torch.tensor([0.3, 0.3 + 1.2])
    expected = (torch.tensor([0.35, 0.65]) * means).sum()
    torch.testing.assert_close(mixture.posterior_epred(), expected.expand(2, 3, 4))
    assert abs(expected.item() - 0.3) > 0.1
    quantile = fit(
        "y ~ 1",
        data=data,
        family="asym_laplace",
        fixed_parameters={"sigma": 0.7, "quantile": 0.8},
        method="svi",
        num_steps=1,
    )
    quantile = _recorded(quantile, {"intercept": torch.tensor(0.3)})
    torch.testing.assert_close(
        quantile.posterior_epred(),
        torch.tensor(0.3 + 0.7 * (1 - 2 * 0.8) / (0.8 * 0.2)).expand(2, 3, 4),
    )
    torch.testing.assert_close(quantile.posterior_linpred(), torch.full((2, 3, 4), 0.3))


def test_inflated_beta_epred_includes_boundary_mass(data):
    result = fit(
        "y ~ 1",
        data=data.assign(y=[0.0, 0.4, 0.7, 1.0]),
        family="zero_one_inflated_beta",
        fixed_parameters={"phi": 3.0, "zoi": 0.3, "coi": 0.4},
        method="svi",
        num_steps=1,
    )
    result = _recorded(result, {"intercept": torch.tensor(0.3)})
    expected = 0.3 * 0.4 + (1 - 0.3) * torch.tensor(0.3).sigmoid()
    torch.testing.assert_close(result.posterior_epred(), expected.expand(2, 3, 4))


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_independent_group_coefficients_include_random_only_covariates(
    data, parameterization
):
    result = fit(
        "y ~ 1 + (1 + x || g)",
        data=data,
        method="svi",
        num_steps=1,
        reparameterize=parameterization,
    )
    scale = torch.tensor([0.4, 0.7])
    z = torch.tensor([[0.2, 1.0], [-0.5, 0.3]])
    coefficient = z * scale
    values = {
        "intercept": torch.tensor(0.3),
        "sigma": torch.tensor(0.7),
        "sigma_g_Intercept": scale[0],
        "sigma_g_x": scale[1],
    }
    values |= (
        {"z_g_Intercept": z[:, 0], "z_g_x": z[:, 1]}
        if parameterization == "noncentered"
        else {"alpha_g": coefficient[:, 0], "beta_g_x": coefficient[:, 1]}
    )
    result = _recorded(result, values)
    combined = result.coefficients()["g"]
    torch.testing.assert_close(combined["x"], coefficient[:, 1].expand(2, 3, 2))
    indices = torch.tensor([0, 0, 1, 1])
    expected = (
        0.3
        + coefficient[indices, 0]
        + coefficient[indices, 1] * torch.tensor(data.x.to_numpy(), dtype=torch.float32)
    )
    torch.testing.assert_close(result.posterior_linpred(), expected.expand(2, 3, 4))


def test_shared_group_effects_use_distributional_native_bindings(data):
    result = fit(
        "y ~ 1 + (1 | joint | g)",
        data=data,
        distributional={"sigma": "~ 1 + (1 | joint | g)"},
        method="svi",
        num_steps=1,
    )
    factor = torch.tensor([[1.0, 0.0], [-0.4, (1 - 0.4**2) ** 0.5]])
    scale = torch.tensor([0.3, 0.6])
    z = torch.tensor([[0.3, -0.4], [-0.2, 0.7]])
    coefficient = z @ factor.T * scale
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.2),
            "dpar_sigma_intercept": torch.tensor(-0.4),
            "cor_shared_joint": factor,
            "sigma_g_Intercept": scale[0],
            "dpar_sigma_sigma_g_Intercept": scale[1],
            "z_g_Intercept": z[:, 0],
            "dpar_sigma_z_g_Intercept": z[:, 1],
        },
    )
    torch.testing.assert_close(
        result.random_effects(parameter="sigma")["g"]["Intercept"],
        coefficient[:, 1].expand(2, 3, 2),
    )
    expected = (-0.4 + coefficient[torch.tensor([0, 0, 1, 1]), 1]).exp()
    torch.testing.assert_close(
        result.posterior_linpred(parameter="sigma", transform=True),
        expected.expand(2, 3, 4),
    )


def test_real_mcmc_draws_and_prediction_do_not_recompile(data, monkeypatch):
    result = fit(
        "y ~ x", data=data, method="hmc", num_chains=2, num_warmup=2, num_samples=3
    )

    def reject_recompile(*args, **kwargs):
        raise AssertionError("posterior summaries must reuse the fitted native program")

    monkeypatch.setattr("quivers.formulas._fit.Compiler.compile", reject_recompile)
    draws = result.draws()
    expected = draws["intercept"][..., None] + draws["beta_x"][
        ..., None
    ] * torch.tensor(data.x.to_numpy(), dtype=torch.float32)
    torch.testing.assert_close(result.posterior_linpred(), expected)
    torch.testing.assert_close(result.posterior_epred(), expected)
    assert result.fixed_effects()["x"].shape == (2, 3)


def test_wrong_posterior_sites_are_refused_before_replay(data):
    result = fit("y ~ x", data=data, method="svi", num_steps=1)
    result = _recorded(
        result, {"intercept": torch.tensor(0.3), "sigma": torch.tensor(0.7)}
    )
    with pytest.raises(ValueError, match="posterior sites do not match"):
        result.posterior_epred()
