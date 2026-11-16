"""Grouped threshold coordinates preserve order, identified shape and prediction."""

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import loads
from quivers.dsl.emit import module_to_source
from quivers.dsl.parser import parse
from quivers.formulas import families, fit, formula_from_data
from quivers.formulas.compile import FormulaToQVRModule
from quivers.inference.trace import trace


@pytest.fixture(autouse=True)
def _double_literal_carrier():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def _host(value):
    return value.item() if value.ndim == 0 else tuple(_host(x) for x in value.detach())


@pytest.mark.parametrize("family", ["cumulative", "hurdle_cumulative"])
@pytest.mark.parametrize("policy", ["flexible", "sum_to_zero", "equidistant"])
@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
def test_group_profile_cs_joint_density_all_gradients_and_reference(
    family, policy, coordinates
):
    response = [0, 1, 2, 3, 1, 2] if family == "cumulative" else [0, 1, 2, 3, 4, 2]
    frame = pd.DataFrame(
        {
            "y": response,
            "x": [-0.4, 0.3, -1.1, 0.8, 0.2, 0.1],
            "z": [-2.0, -0.4, 0.7, 2.0, 1.0, -1.0],
            "g": ["a", "b"] * 3,
        }
    )
    formula = formula_from_data("y~x+cs(z)+(1|g)", frame)
    lens = FormulaToQVRModule(
        families[family].with_thresholds(policy),
        thresholds_by="g",
        reparameterize=coordinates,
        fixed_parameters={
            "disc": 1.25,
            **({"hu": 0.25} if family == "hurdle_cumulative" else {}),
        },
    )
    module, _ = lens.forward(formula)
    source = module_to_source(module)
    assert module_to_source(parse(source)) == source
    compiled = loads(source)
    width = 1 if policy == "equidistant" else 2
    intercept = torch.tensor(0.3, requires_grad=True)
    beta = torch.tensor(-0.4, requires_grad=True)
    cs = torch.tensor([1.3, -0.5, 1.1], requires_grad=True)
    mean = torch.tensor([-0.3, 0.2][:width], requires_grad=True)
    scale = torch.tensor([0.4, 0.7][:width], requires_grad=True)
    raw = torch.tensor([0.2, -0.4, 0.6, -0.5][: 2 * width], requires_grad=True)
    location_raw = torch.tensor([0.1, -0.2], requires_grad=True)
    location_scale = torch.tensor(0.8, requires_grad=True)
    centered = coordinates == "centered"
    ell = raw.reshape(2, width) if centered else mean + scale * raw.reshape(2, width)
    location = location_raw if centered else location_scale * location_raw
    values = {
        "intercept": intercept,
        "beta_x": beta,
        "beta_cs_z": cs,
        "sigma_g_Intercept": location_scale,
        "alpha_g" if centered else "z_g_Intercept": location_raw,
        "cutpoint_log_spacing_mean": mean,
        "cutpoint_log_spacing_scale": scale,
        "cutpoint_log_spacing_value" if centered else "cutpoint_log_spacing_noise": raw,
    }
    inputs = lens.fixed_column_observations(formula)
    codes = torch.tensor(formula.group_indices["g"])
    inputs.update(g_idx=codes, y=torch.tensor(response))
    recorded = trace(compiled.morphism, torch.zeros(1, 1), inputs | values)

    x, z = inputs["x"].to(cs), inputs["cs_z"].to(cs)
    effects = z[:, None] * cs
    differences = effects[:, 1:] - effects[:, :-1]
    # Unequal per-gap bounds distinguish a shared maximum from per-gap shifts.
    bound = torch.stack(
        [
            torch.stack(
                [
                    torch.cat((torch.zeros(1), differences[codes == group, gap])).max()
                    for gap in range(2)
                ]
            )
            for group in range(2)
        ]
    )
    if policy == "equidistant":
        gaps = (bound.max(-1).values[:, None] + ell.exp()).expand(2, 2)
    else:
        gaps = bound + ell.exp()
    origin = torch.cat((torch.zeros(2, 1), gaps.cumsum(-1)), -1)
    cuts = origin - origin.mean(-1, keepdim=True)
    torch.testing.assert_close(
        recorded.sites["cutpoint_matrix"].value.reshape(2, 3), cuts
    )
    contrast = 1.25 * (
        cuts[codes] - location[codes, None] - (intercept + beta * x)[:, None] - effects
    )
    cdf = contrast.sigmoid()
    probabilities = torch.cat(
        (cdf[:, :1], cdf[:, 1:] - cdf[:, :-1], 1 - cdf[:, -1:]), -1
    )
    if family == "hurdle_cumulative":
        probabilities = torch.cat(
            (torch.full((len(frame), 1), 0.25), 0.75 * probabilities), -1
        )
    raw_name = (
        "cutpoint_log_spacing_value" if centered else "cutpoint_log_spacing_noise"
    )
    expected = probabilities.log().gather(1, torch.tensor(response)[:, None]).sum()
    expected += (
        torch.distributions.Normal(mean, scale).log_prob(raw.reshape(2, width)).sum()
        if centered
        else torch.distributions.Normal(0.0, 1.0).log_prob(raw).sum()
    )
    # Include all three hierarchical priors to exercise every head gradient.
    expected += torch.distributions.Normal(0.0, 1.0).log_prob(mean).sum()
    expected += torch.distributions.HalfNormal(0.5).log_prob(scale).sum()
    expected += (
        torch.distributions.Normal(0.0, location_scale).log_prob(location_raw).sum()
        if centered
        else torch.distributions.Normal(0.0, 1.0).log_prob(location_raw).sum()
    )
    actual = sum(
        recorded.sites[name].log_prob.sum()
        for name in (
            "y",
            raw_name,
            "cutpoint_log_spacing_mean",
            "cutpoint_log_spacing_scale",
            "alpha_g" if centered else "z_g_Intercept",
        )
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    parameters = (intercept, beta, cs, mean, scale, raw, location_raw, location_scale)
    got = torch.autograd.grad(actual, parameters, retain_graph=True, allow_unused=True)
    want = torch.autograd.grad(expected, parameters, allow_unused=True)
    for a, b, parameter in zip(got, want, parameters, strict=True):
        torch.testing.assert_close(
            a if a is not None else torch.zeros_like(parameter),
            b if b is not None else torch.zeros_like(parameter),
            atol=2e-9,
            rtol=2e-9,
        )
    reference = compiled.run(
        "model",
        data={k: _host(v) for k, v in inputs.items()},
        sites={k: _host(v) for k, v in values.items()},
    )
    assert reference.log_joint == pytest.approx(recorded.log_joint.item(), rel=1e-9)


@pytest.mark.parametrize("policy", ["flexible", "sum_to_zero", "equidistant"])
@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
def test_profile_policy_predicts_known_and_new_groups_without_reestimating_cuts(
    policy, coordinates
):
    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 3] * 2,
            "x": np.linspace(-0.6, 0.6, 8),
            "g": ["a"] * 4 + ["b"] * 4,
        }
    )
    result = fit(
        "y~x+cs(x)+(1|g)",
        data=frame,
        family=families["cumulative"].with_thresholds(policy),
        thresholds_by="g",
        reparameterize=coordinates,
        method="svi",
        num_steps=1,
        seed=13,
    )
    original = result.posterior_thresholds(num_samples=2, seed=17)
    known = result.posterior_thresholds(
        new_data=frame.iloc[[5, 1]], num_samples=2, seed=17
    )
    torch.testing.assert_close(known, original[:, :, [5, 1]])
    fresh = pd.DataFrame({"x": [0.0, 0.0, 0.0], "g": ["a", "new", "new"]})
    state = torch.random.get_rng_state().clone()
    for new_policy in ("feasible_uncertainty", "feasible_gaussian"):
        options = dict(
            new_data=fresh,
            allow_new_levels=True,
            sample_new_levels=new_policy,
            num_samples=2,
            seed=17,
        )
        cuts = result.posterior_thresholds(**options)
        assert bool((cuts.diff(dim=-1) > 0).all())
        torch.testing.assert_close(cuts[:, :, 1], cuts[:, :, 2])
        if policy == "equidistant":
            torch.testing.assert_close(
                cuts.diff(dim=-1)[..., 0], cuts.diff(dim=-1)[..., 1]
            )
        selected = result.posterior_thresholds(re_formula="~0", **options)
        torch.testing.assert_close(selected.diff(dim=-1), cuts.diff(dim=-1))
        probabilities = result.posterior_epred(**options)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, 2, 3))
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("family", ["cumulative", "hurdle_cumulative"])
@pytest.mark.parametrize("policy", ["flexible", "sum_to_zero", "equidistant"])
def test_binary_positive_profile_has_only_its_reused_group_location(family, policy):
    response = [0, 1, 0, 1] if family == "cumulative" else [0, 1, 2, 1]
    frame = pd.DataFrame({"y": response, "g": ["a", "b", "a", "b"]})
    lens = FormulaToQVRModule(
        families[family].with_thresholds(policy),
        thresholds_by="g",
        reparameterize="centered",
        fixed_parameters={"hu": 0.2} if family == "hurdle_cumulative" else {},
    )
    formula = formula_from_data("y~1+(1|g)", frame)
    module, _ = lens.forward(formula)
    source = module_to_source(module)
    assert "sample cutpoint_log_spacing" not in source
    location = torch.tensor([0.2, -0.3], requires_grad=True)
    values = {
        "intercept": torch.tensor(0.1),
        "sigma_g_Intercept": torch.tensor(0.7),
        "alpha_g": location,
    }
    inputs = lens.fixed_column_observations(formula)
    inputs.update(
        g_idx=torch.tensor(formula.group_indices["g"]), y=torch.tensor(response)
    )
    recorded = trace(loads(source).morphism, torch.zeros(1, 1), inputs | values)
    shape = recorded.sites["cutpoint_matrix"].value.reshape(2, 1)
    torch.testing.assert_close(shape, torch.zeros(2, 1))
    profile = shape - location[:, None]
    jacobian = torch.stack(
        [
            torch.autograd.grad(profile[g, 0], location, retain_graph=True)[0]
            for g in range(2)
        ]
    )
    torch.testing.assert_close(jacobian, -torch.eye(2))


@pytest.mark.parametrize("policy", ["flexible", "sum_to_zero", "equidistant"])
@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
def test_group_threshold_hyperpriors_use_existing_named_override_catalog(
    policy, coordinates
):
    frame = pd.DataFrame({"y": [0, 1, 2, 3] * 2, "g": ["a", "b"] * 4})
    formula = formula_from_data("y~1+(1|g)", frame)
    lens = FormulaToQVRModule(
        families["cumulative"].with_thresholds(policy),
        thresholds_by="g",
        reparameterize=coordinates,
        user_priors={
            "cutpoint_log_spacing_mean": "Normal(0.2, 0.7)",
            "cutpoint_log_spacing_scale": "Normal(-0.1, 0.6)",
        },
    )
    catalog = lens.get_prior(formula)
    assert catalog["cutpoint_log_spacing_mean"] == "Normal(loc=0.2, scale=0.7)"
    assert catalog["cutpoint_log_spacing_scale"].startswith("Restrict(base=Normal(")
    assert "low=0.0" in catalog["cutpoint_log_spacing_scale"]
    module, _ = lens.forward(formula)
    compiled = loads(module_to_source(module))
    width = 1 if policy == "equidistant" else 2
    mean = torch.tensor([-0.3, 0.4][:width], requires_grad=True)
    scale = torch.tensor([0.6, 0.8][:width], requires_grad=True)
    values = {
        "intercept": torch.tensor(0.1),
        "sigma_g_Intercept": torch.tensor(0.5),
        "alpha_g" if coordinates == "centered" else "z_g_Intercept": torch.zeros(2),
        "cutpoint_log_spacing_mean": mean,
        "cutpoint_log_spacing_scale": scale,
        "cutpoint_log_spacing_value"
        if coordinates == "centered"
        else "cutpoint_log_spacing_noise": torch.zeros(2 * width),
    }
    inputs = {
        "g_idx": torch.tensor(formula.group_indices["g"]),
        "y": torch.tensor(frame.y.to_numpy()),
    }
    recorded = trace(compiled.morphism, torch.zeros(1, 1), inputs | values)
    actual = sum(
        recorded.sites[name].log_prob.sum()
        for name in ("cutpoint_log_spacing_mean", "cutpoint_log_spacing_scale")
    )
    expected = (
        torch.distributions.Normal(0.2, 0.7).log_prob(mean).sum()
        + (
            torch.distributions.Normal(-0.1, 0.6).log_prob(scale)
            - torch.log1p(-torch.distributions.Normal(-0.1, 0.6).cdf(torch.tensor(0.0)))
        ).sum()
    )
    torch.testing.assert_close(actual, expected)
    for got, want in zip(
        torch.autograd.grad(actual, (mean, scale), retain_graph=True),
        torch.autograd.grad(expected, (mean, scale)),
        strict=True,
    ):
        torch.testing.assert_close(got, want)
    reference = compiled.run(
        "model",
        data={k: _host(v) for k, v in inputs.items()},
        sites={k: _host(v) for k, v in values.items()},
    )
    assert reference.log_joint == pytest.approx(recorded.log_joint.item(), rel=1e-9)
