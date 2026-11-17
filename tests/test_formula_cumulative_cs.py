"""Independent likelihood and derivative checks for constrained linear ordinal CS."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.formulas import families, formula_to_qvr
from quivers.inference.trace import trace


@pytest.fixture(autouse=True)
def _double_literal_carrier():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("family", ("cumulative", "hurdle_cumulative"))
@pytest.mark.parametrize("thresholds", ("flexible", "sum_to_zero", "equidistant"))
def test_linear_cs_feasible_spacing_matches_cdf_and_every_derivative(
    family, thresholds
):
    response = [0, 1, 2, 3, 1, 2] if family == "cumulative" else [0, 1, 2, 3, 4, 2]
    frame = pd.DataFrame(
        {
            "y": response,
            "x": [-0.4, 0.3, -1.1, 0.8, 0.2, 0.1],
            "z": [-2.0, -0.4, 0.7, 2.0, 1.0, -1.0],
        }
    )
    descriptor = families[family].with_thresholds(thresholds)
    source = formula_to_qvr(
        "y~x+cs(z)",
        data=frame,
        family=descriptor,
        priors={
            "intercept": "Normal(0.0, 1.0)",
            "beta_x": "Normal(0.0, 0.5)",
            "beta_cs_z": "Normal(0.0, 0.25)",
        },
        fixed_parameters={
            "disc": 1.25,
            **({"hu": 0.25} if family == "hurdle_cumulative" else {}),
        },
    )
    assert module_to_source(parse(module_to_source(parse(source)))) == module_to_source(
        parse(source)
    )
    compiled = loads(source)
    intercept = torch.tensor(0.3, dtype=torch.double, requires_grad=True)
    beta = torch.tensor(-0.4, dtype=torch.double, requires_grad=True)
    cs = torch.tensor([1.3, -0.5, 1.1], dtype=torch.double, requires_grad=True)
    spacing = torch.tensor(
        0.8 if thresholds == "equidistant" else [math.log(0.8), math.log(1.2)],
        dtype=torch.double,
        requires_grad=True,
    )
    x = torch.tensor(frame.x.to_numpy(), dtype=torch.double)
    z = torch.tensor(frame.z.to_numpy(), dtype=torch.double)
    values = {
        "intercept": intercept,
        "beta_x": beta,
        "beta_cs_z": cs,
        "cutpoint_spacing"
        if thresholds == "equidistant"
        else "cutpoint_log_spacing": spacing,
    }
    recorded = trace(
        compiled.morphism,
        torch.zeros(1, 1),
        {"x": x, "cs_z": z, "y": torch.tensor(response), **values},
    )
    effects = z[:, None] * cs
    adjacent = effects[:, 1:] - effects[:, :-1]
    # Compute the two support inequalities separately, without the source reducer.
    bounds = torch.stack(
        [
            torch.stack([torch.tensor(0.0, dtype=torch.double), *adjacent[:, j]]).max()
            for j in range(2)
        ]
    )
    gaps = (
        bounds.max() + spacing
        if thresholds == "equidistant"
        else bounds + spacing.exp()
    )
    if gaps.ndim == 0:
        gaps = gaps.expand(2)
    raw = torch.cat((torch.zeros(1, dtype=torch.double), gaps.cumsum(0)))
    cuts = raw - raw.mean()
    contrasts = 1.25 * (cuts - (intercept + beta * x)[:, None] - effects)
    assert (contrasts[:, 1:] > contrasts[:, :-1]).all()
    cdf = contrasts.sigmoid()
    probabilities = torch.cat(
        (cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), -1
    )
    if family == "hurdle_cumulative":
        probabilities = torch.cat(
            (
                torch.full((len(frame), 1), 0.25, dtype=torch.double),
                0.75 * probabilities,
            ),
            -1,
        )
    expected = probabilities.log().gather(1, torch.tensor(response)[:, None]).sum()
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-10)
    parameters = (intercept, beta, cs, spacing)
    for got, want in zip(
        torch.autograd.grad(actual, parameters, retain_graph=True),
        torch.autograd.grad(expected, parameters),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=2e-9, atol=2e-9)
    # The reference executor includes unchanged coefficient and explicit slack priors.
    reference = compiled.run(
        "model",
        data={"x": tuple(frame.x), "cs_z": tuple(frame.z), "y": tuple(response)},
        sites={
            name: (
                float(value.detach())
                if value.ndim == 0
                else tuple(value.detach().tolist())
            )
            for name, value in values.items()
        },
    )
    torch.testing.assert_close(recorded.sites["cutpoints"].value.reshape(-1), cuts)
    assert np.isfinite(reference.log_joint)
    assert reference.log_joint == pytest.approx(
        float(recorded.log_joint.detach().sum()), rel=1e-9, abs=1e-9
    )


def _controlled_fit(*, group=False):
    from quivers.formulas import fit
    from quivers.inference.mcmc.driver import MCMCResult

    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 3, 1, 2],
            "x": [-0.4, 0.3, -1.1, 0.8, 0.2, 0.1],
            "z": [-2.0, -0.4, 0.7, 2.0, 1.0, -1.0],
            "g": ["a", "b", "a", "b", "a", "b"],
        }
    )
    model = fit(
        "y~x+cs(z)" + ("+(0+cs(1)||g)" if group else ""),
        data=frame,
        family="cumulative",
        method="svi",
        num_steps=1,
        seed=23,
    )
    values = {
        "intercept": torch.tensor(0.3),
        "beta_x": torch.tensor(-0.4),
        "beta_cs_z": torch.tensor([1.3, -0.5, 1.1]),
        "cutpoint_log_spacing": torch.log(torch.tensor([0.8, 1.2])),
    }
    if group:
        values.update(
            {
                "sigma_g_cs_1": torch.ones(3),
                "z_g_cs_1": torch.tensor([0.0, 0.0, 0.0, 0.2, -0.2, 0.3]),
            }
        )
    samples = {
        name: value.expand(1, 2, *value.shape).clone() for name, value in values.items()
    }
    return model.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(1, 2),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=2,
        )
    ), frame


def test_prediction_retains_actual_cutpoints_across_fresh_rows_and_calls():
    model, training = _controlled_fit()
    fresh = pd.DataFrame({"x": [0.1, 0.9], "z": [0.25, 0.8]})
    before = fresh.copy(deep=True)
    predicted = model.posterior_epred(new_data=fresh)
    cs = torch.tensor([1.3, -0.5, 1.1])
    training_cs = torch.tensor(training.z.to_numpy())[:, None] * cs
    gaps = (training_cs[:, 1:] - training_cs[:, :-1]).amax(0).clamp_min(
        0
    ) + torch.tensor([0.8, 1.2])
    uncentered = torch.cat((torch.zeros(1), gaps.cumsum(0)))
    cuts = uncentered - uncentered.mean()
    eta = (0.3 - 0.4 * torch.tensor(fresh.x.to_numpy()))[:, None] + torch.tensor(
        fresh.z.to_numpy()
    )[:, None] * cs
    cdf = (cuts - eta).sigmoid()
    expected = torch.cat((cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), -1)
    torch.testing.assert_close(
        predicted, expected.expand(1, 2, 2, 4), rtol=1e-6, atol=1e-7
    )
    model.posterior_epred(new_data=fresh.iloc[:1])
    torch.testing.assert_close(model.posterior_epred(new_data=fresh), predicted)
    pd.testing.assert_frame_equal(fresh, before)
    assert model.posterior_predict(new_data=fresh, seed=8).shape == (1, 2, 2)
    torch.testing.assert_close(
        model.posterior_linpred(new_data=fresh), eta.expand(1, 2, 2, 3)
    )


def test_prediction_rejects_crossings_without_recomputing_or_discarding_draws():
    model, _ = _controlled_fit()
    before = {name: value.clone() for name, value in model.posterior.samples.items()}
    with pytest.raises(ValueError, match="chain 0, draw 0.*ordinal_effective_spacing"):
        model.posterior_epred(new_data=pd.DataFrame({"x": [0.0], "z": [200.0]}))
    for name, value in before.items():
        torch.testing.assert_close(model.posterior.samples[name], value)


@pytest.mark.parametrize("family", ("cumulative", "hurdle_cumulative"))
def test_grouped_linear_cs_uses_each_group_domain_and_all_profile_coordinates(family):
    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 3, 1, 2] if family == "cumulative" else [0, 1, 2, 3, 4, 2],
            "x": [-0.4, 0.3, -1.1, 0.8, 0.2, 0.1],
            "z": [-2.0, -0.4, 0.7, 2.0, 1.0, -1.0],
            "g": ["a", "b", "a", "b", "a", "b"],
        }
    )
    source = formula_to_qvr(
        "y~x+cs(z)+(0+cs(z)||g)",
        data=frame,
        family=family,
        thresholds_by="g",
        fixed_parameters={
            "disc": 1.0,
            **({"hu": 0.25} if family == "hurdle_cumulative" else {}),
        },
    )
    assert source.count("sample sigma_g_Intercept ") == 1
    compiled = loads(source)
    raw_values = {
        "intercept": 0.3,
        "beta_x": -0.4,
        "beta_cs_z": [1.3, -0.5, 1.1],
        "sigma_g_Intercept": 0.6,
        "z_g_Intercept": [-0.3, 0.7],
        "sigma_g_cs_z": [0.4, 0.7, 0.5],
        "z_g_cs_z": [0.2, -0.1, 0.4, -0.3, 0.5, -0.2],
        "cutpoint_log_spacing_mean": [-0.2, 0.1],
        "cutpoint_log_spacing_scale": [0.3, 0.4],
        "cutpoint_log_spacing_noise": [0.2, -0.3, -0.4, 0.1],
    }
    values = {
        name: torch.tensor(value, requires_grad=True)
        for name, value in raw_values.items()
    }
    index = torch.tensor([0, 1, 0, 1, 0, 1])
    x, z = torch.tensor(frame.x.to_numpy()), torch.tensor(frame.z.to_numpy())
    observed = {
        "x": x,
        "cs_z": z,
        "random_cs_z": z,
        "g_idx": index,
        "y": torch.tensor(frame.y.to_numpy()),
        **values,
    }
    recorded = trace(compiled.morphism, torch.zeros(1, 1), observed)
    coefficients = values["beta_cs_z"] + values["sigma_g_cs_z"] * values[
        "z_g_cs_z"
    ].reshape(2, 3)
    effects = z[:, None] * coefficients[index]
    bounds = torch.stack(
        [
            (effects[index == group, 1:] - effects[index == group, :-1])
            .amax(0)
            .clamp_min(0)
            for group in range(2)
        ]
    )
    log_slack = values["cutpoint_log_spacing_mean"] + values[
        "cutpoint_log_spacing_scale"
    ] * values["cutpoint_log_spacing_noise"].reshape(2, 2)
    gaps = bounds + log_slack.exp()
    raw = torch.cat((torch.zeros(2, 1), gaps.cumsum(-1)), -1)
    cuts = raw - raw.mean(-1, keepdim=True)
    group_location = values["sigma_g_Intercept"] * values["z_g_Intercept"]
    eta = values["intercept"] + values["beta_x"] * x + group_location[index]
    contrasts = cuts[index] - effects - eta[:, None]
    assert (contrasts[:, 1:] > contrasts[:, :-1]).all()
    cdf = contrasts.sigmoid()
    probabilities = torch.cat(
        (cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), -1
    )
    if family == "hurdle_cumulative":
        probabilities = torch.cat(
            (torch.full((len(frame), 1), 0.25), 0.75 * probabilities), -1
        )
    expected = (
        probabilities.log().gather(1, torch.tensor(frame.y.to_numpy())[:, None]).sum()
    )
    actual = recorded.sites["y"].log_prob.sum()
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-9)
    parameters = tuple(values.values())
    for got, want in zip(
        torch.autograd.grad(actual, parameters, retain_graph=True),
        torch.autograd.grad(expected, parameters),
        strict=True,
    ):
        torch.testing.assert_close(got, want, rtol=1e-8, atol=1e-8)
    torch.testing.assert_close(
        recorded.sites["cutpoint_matrix"].value.reshape(2, 3), cuts
    )
    # Centered shape and one location span all three cutpoints, without a second location latent.
    torch.testing.assert_close(cuts.mean(-1), torch.zeros(2), atol=1e-15, rtol=0)
    profile = cuts - group_location[:, None]
    torch.testing.assert_close(profile.mean(-1), -group_location)
    assert not torch.equal(bounds[0], bounds[1])
    native_profile = (
        recorded.sites["cutpoint_matrix"].value.reshape(2, 3)
        - (values["sigma_g_Intercept"] * values["z_g_Intercept"])[:, None]
    )
    for group in range(2):
        rows = []
        for boundary in range(3):
            noise_gradient, location_gradient = torch.autograd.grad(
                native_profile[group, boundary],
                (values["cutpoint_log_spacing_noise"], values["z_g_Intercept"]),
                retain_graph=True,
            )
            rows.append(
                torch.cat(
                    (
                        noise_gradient[2 * group : 2 * group + 2],
                        location_gradient[group : group + 1],
                    )
                )
            )
        assert torch.linalg.matrix_rank(torch.stack(rows)) == 3
    reference = compiled.run(
        "model",
        data={
            "x": tuple(frame.x),
            "cs_z": tuple(frame.z),
            "random_cs_z": tuple(frame.z),
            "g_idx": tuple(index.tolist()),
            "y": tuple(frame.y),
        },
        sites={
            name: float(value.detach())
            if value.ndim == 0
            else tuple(value.detach().tolist())
            for name, value in values.items()
        },
    )
    assert reference.log_joint == pytest.approx(
        float(recorded.log_joint.detach().sum()), rel=1e-9, abs=1e-9
    )


@pytest.mark.parametrize("family", ("cumulative", "hurdle_cumulative"))
def test_threshold_group_location_is_added_once_and_binary_profile_is_supported(family):
    from quivers.formulas import fit

    y = [0, 1, 0, 1] if family == "cumulative" else [0, 1, 2, 1]
    frame = pd.DataFrame(
        {"y": y, "x": [0.0, 0.2, -0.3, 0.1], "g": ["a", "b", "a", "b"]}
    )
    plain = formula_to_qvr("y~x", data=frame, family=family, thresholds_by="g")
    explicit = formula_to_qvr("y~x+(1|g)", data=frame, family=family, thresholds_by="g")
    assert plain == explicit
    assert plain.count("sample sigma_g_Intercept ") == 1
    assert "cutpoint_log_spacing" not in plain
    result = fit(
        "y~x",
        data=frame,
        family=family,
        thresholds_by="g",
        method="svi",
        num_steps=1,
        seed=11,
    )
    assert (
        sum(
            term.group == "g" and term.slope == "Intercept"
            for term in result.formula.random_terms
        )
        == 1
    )
    assert result.posterior_epred(
        new_data=frame.drop(columns="y"), num_samples=1
    ).shape == (1, 1, 4, len(set(y)))


@pytest.mark.parametrize("policy", ("feasible_uncertainty", "feasible_gaussian"))
def test_new_cs_group_policies_are_explicit_conditional_and_reproducible(policy):
    model, _ = _controlled_fit(group=True)
    fresh = pd.DataFrame(
        {"x": [0.0, 0.0, 0.0], "z": [2.8, 2.8, 2.8], "g": ["fresh", "fresh", "a"]}
    )
    with pytest.raises(ValueError, match="explicit.*feasible_uncertainty"):
        model.posterior_epred(new_data=fresh, allow_new_levels=True)
    state = torch.random.get_rng_state().clone()
    kwargs = dict(
        new_data=fresh, allow_new_levels=True, sample_new_levels=policy, seed=18
    )
    predicted = model.posterior_epred(**kwargs)
    profiles = model.posterior_thresholds(**kwargs)
    assert model.threshold_labels == ((0, 1), (1, 2), (2, 3))
    assert (profiles[..., 1:] > profiles[..., :-1]).all()
    torch.testing.assert_close(profiles[..., 0, :], profiles[..., 1, :])
    cdf = (profiles - 0.3).sigmoid()
    expected = torch.cat(
        (cdf[..., :1], cdf[..., 1:] - cdf[..., :-1], 1 - cdf[..., -1:]), -1
    )
    torch.testing.assert_close(predicted, expected)
    torch.testing.assert_close(model.posterior_epred(**kwargs), predicted)
    assert torch.equal(torch.random.get_rng_state(), state)
    if policy == "feasible_uncertainty":
        # Donor b crosses at z=2.8; donor a alone remains feasible. Rejection
        # conditions the donor law without dropping either posterior draw.
        torch.testing.assert_close(predicted[..., 0, :], predicted[..., 2, :])
    assert model.posterior_predict(**kwargs).shape == (1, 2, 3)


def test_infeasible_new_group_request_fails_with_draws_preserved(monkeypatch):
    import quivers.formulas._posterior as posterior

    model, _ = _controlled_fit(group=True)
    monkeypatch.setattr(posterior, "_MAX_FEASIBLE_GROUP_ATTEMPTS", 3)
    fresh = pd.DataFrame({"x": [0.0], "z": [200.0], "g": ["fresh"]})
    before = {name: value.clone() for name, value in model.posterior.samples.items()}
    with pytest.raises(ValueError, match="chain 0, draw 0.*3 attempts.*retained"):
        model.posterior_epred(
            new_data=fresh,
            allow_new_levels=True,
            sample_new_levels="feasible_uncertainty",
        )
    for name, value in before.items():
        torch.testing.assert_close(model.posterior.samples[name], value)


def _grouped_controlled_fit():
    from quivers.formulas import fit
    from quivers.inference.mcmc.driver import MCMCResult

    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 3, 1, 2],
            "x": [-0.4, 0.3, -1.1, 0.8, 0.2, 0.1],
            "z": [-2.0, -0.4, 0.7, 2.0, 1.0, -1.0],
            "g": ["a", "b", "a", "b", "a", "b"],
        }
    )
    result = fit(
        "y~x+cs(z)+(0+cs(z)||g)",
        data=frame,
        family="cumulative",
        thresholds_by="g",
        method="svi",
        num_steps=1,
        seed=17,
    )
    raw = {
        "intercept": 0.3,
        "beta_x": -0.4,
        "beta_cs_z": [1.3, -0.5, 1.1],
        "sigma_g_Intercept": 0.6,
        "z_g_Intercept": [-0.3, 0.7],
        "sigma_g_cs_z": [0.4, 0.7, 0.5],
        "z_g_cs_z": [0.2, -0.1, 0.4, -0.3, 0.5, -0.2],
        "cutpoint_log_spacing_mean": [-0.2, 0.1],
        "cutpoint_log_spacing_scale": [0.3, 0.4],
        "cutpoint_log_spacing_noise": [0.2, -0.3, -0.4, 0.1],
    }
    values = {name: torch.tensor(value) for name, value in raw.items()}
    result = result.with_(
        posterior=MCMCResult(
            samples={
                name: value.expand(1, 3, *value.shape).clone()
                for name, value in values.items()
            },
            log_densities=torch.zeros(1, 3),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=3,
        )
    )
    recorded = trace(
        result.program, torch.zeros(1, 1), dict(result.observations) | values
    )
    return result, values, recorded.sites["cutpoint_matrix"].value.reshape(2, 3)


def test_grouped_cs_known_subset_and_exclusions_retain_full_training_profiles():
    model, values, cuts = _grouped_controlled_fit()
    fresh = pd.DataFrame(
        {"x": [0.2, -0.5, 0.1], "z": [-0.3, 0.6, 0.1], "g": ["b", "b", "b"]}
    )
    coefficients = (
        values["beta_cs_z"]
        + values["sigma_g_cs_z"] * values["z_g_cs_z"].reshape(2, 3)[1]
    )
    expected_profile = (
        cuts[1] - 0.6 * 0.7 - torch.tensor(fresh.z.to_numpy())[:, None] * coefficients
    )
    observed = model.posterior_thresholds(new_data=fresh)
    torch.testing.assert_close(
        observed, expected_profile.expand(1, 3, 3, 3), rtol=1e-6, atol=1e-7
    )
    excluded_profile = (
        cuts[1] - torch.tensor(fresh.z.to_numpy())[:, None] * values["beta_cs_z"]
    )
    torch.testing.assert_close(
        model.posterior_thresholds(new_data=fresh, re_formula="~0"),
        excluded_profile.expand(1, 3, 3, 3),
        rtol=1e-6,
        atol=1e-7,
    )
    population = 0.3 - 0.4 * torch.tensor(fresh.x.to_numpy())
    cdf = (expected_profile - population[:, None]).sigmoid()
    expected = torch.cat((cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), -1)
    torch.testing.assert_close(
        model.posterior_epred(new_data=fresh),
        expected.expand(1, 3, 3, 4),
        rtol=1e-6,
        atol=1e-7,
    )


@pytest.mark.parametrize("policy", ("feasible_uncertainty", "feasible_gaussian"))
def test_grouped_cs_new_profiles_preserve_known_cuts_and_joint_support(policy):
    model, values, cuts = _grouped_controlled_fit()
    fresh = pd.DataFrame(
        {"x": [0.0, 0.0, 0.0], "z": [0.1, 0.1, 0.1], "g": ["fresh", "fresh", "b"]}
    )
    kwargs = dict(
        new_data=fresh, allow_new_levels=True, sample_new_levels=policy, seed=22
    )
    profiles = model.posterior_thresholds(**kwargs)
    assert (profiles[..., 1:] > profiles[..., :-1]).all()
    torch.testing.assert_close(profiles[..., 0, :], profiles[..., 1, :])
    coefficients = values["beta_cs_z"] + values["sigma_g_cs_z"] * values[
        "z_g_cs_z"
    ].reshape(2, 3)
    alternatives = cuts - (0.6 * values["z_g_Intercept"])[:, None] - 0.1 * coefficients
    torch.testing.assert_close(
        profiles[..., 2, :], alternatives[1].expand(1, 3, 3), rtol=1e-6, atol=1e-7
    )
    if policy == "feasible_uncertainty":
        assert (
            torch.isclose(profiles[..., 0, None, :], alternatives, atol=1e-6, rtol=1e-6)
            .all(-1)
            .any(-1)
            .all()
        )
    cdf = (profiles - 0.3).sigmoid()
    expected = torch.cat(
        (cdf[..., :1], cdf[..., 1:] - cdf[..., :-1], 1 - cdf[..., -1:]), -1
    )
    torch.testing.assert_close(
        model.posterior_epred(**kwargs), expected, rtol=1e-6, atol=1e-7
    )
    assert model.posterior_predict(**kwargs).shape == (1, 3, 3)


def test_duplicate_cs_intercept_and_full_threshold_profile_is_diagnosed():
    frame = pd.DataFrame({"y": [0, 1, 2, 1], "g": ["a", "b", "a", "b"]})
    with pytest.raises(ValueError, match="duplicates the full thresholds_by profile"):
        formula_to_qvr(
            "y~1+(0+cs(1)||g)", data=frame, family="cumulative", thresholds_by="g"
        )


def test_training_thresholds_raw_likelihood_weights_and_predictive_support_agree():
    from quivers.formulas import fit

    model, frame = _controlled_fit()
    probability = model.posterior_epred()
    response = torch.tensor(frame.y.to_numpy())
    expected = (
        probability.log()
        .gather(-1, response.expand(1, 2, -1).unsqueeze(-1))
        .squeeze(-1)
    )
    torch.testing.assert_close(model.log_likelihood()["y"], expected)
    profiles = model.posterior_thresholds()
    population = 0.3 - 0.4 * torch.tensor(frame.x.to_numpy())
    cdf = (profiles - population[:, None]).sigmoid()
    oracle = torch.cat(
        (cdf[..., :1], cdf[..., 1:] - cdf[..., :-1], 1 - cdf[..., -1:]), -1
    )
    torch.testing.assert_close(probability, oracle)
    weighted = fit(
        "y~x+cs(z)",
        data=frame.assign(w=[0.0, 1.0, 2.0, 0.25, 3.0, 0.0]),
        weights="w",
        family="cumulative",
        method="svi",
        num_steps=1,
        seed=23,
    ).with_(posterior=model.posterior)
    torch.testing.assert_close(weighted.posterior_epred(), probability)
    torch.testing.assert_close(weighted.log_likelihood()["y"], expected)
    sampled = weighted.posterior_predict(seed=41)
    assert sampled.dtype == torch.long
    assert ((sampled >= 0) & (sampled < 4)).all()
    torch.testing.assert_close(weighted.posterior_predict(seed=41), sampled)


def test_fixed_cs_intercept_has_an_independent_threshold_null_direction():
    cutpoints = torch.tensor([-1.2, -0.1, 1.3])
    specific_intercept = torch.tensor([0.1, -0.2, 0.3])
    intercept = torch.tensor(0.2)
    centered_shift = torch.tensor([0.05, -0.08, 0.03])
    original = cutpoints - specific_intercept - intercept
    shifted = (
        (cutpoints + centered_shift) - (specific_intercept + centered_shift) - intercept
    )
    torch.testing.assert_close(original, shifted)
    cdf = original.sigmoid()
    alternative_cdf = shifted.sigmoid()
    torch.testing.assert_close(
        torch.cat((cdf[:1], cdf[1:] - cdf[:-1], 1 - cdf[-1:])),
        torch.cat(
            (
                alternative_cdf[:1],
                alternative_cdf[1:] - alternative_cdf[:-1],
                1 - alternative_cdf[-1:],
            )
        ),
    )
    # The unchanged location direction is independent of the centered shape direction.
    torch.testing.assert_close(
        cutpoints - (specific_intercept - 0.4) - (intercept + 0.4), original
    )
    frame = pd.DataFrame({"y": [0, 1, 2, 3], "x": [0.0, 0.2, -0.3, 0.1]})
    with pytest.raises(
        ValueError, match="fixed cs\\(1\\).*ordinary intercept and thresholds"
    ):
        formula_to_qvr(
            "y~x+cs(1)",
            data=frame,
            family="cumulative",
            priors={"beta_cs_1": "Normal(0.0, 0.5)"},
        )
