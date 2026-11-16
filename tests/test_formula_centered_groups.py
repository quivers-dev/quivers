"""Centered group coordinates share likelihoods and the Gaussian measure law."""

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import loads
from quivers.formulas import families, fit, formula_from_data
from quivers.formulas.compile import FormulaToQVRModule
from quivers.dsl.emit import module_to_source
from quivers.inference.trace import trace


def _host(value):
    return (
        value.item() if value.ndim == 0 else tuple(_host(row) for row in value.detach())
    )


@pytest.mark.parametrize("groups", [2, 3])
def test_plated_computed_mvn_parameters_preserve_checked_feature_layout(groups):
    source = f"""object G:FinSet {groups}
object D:FinSet 3
object Out:Real 1
program model(s:Real,rho:Real):Out->Out
    let loc=factor i:D in 0.1*real(i)
    let L=factor i:D, j:D in select(i==j,s,select(i>j,rho,0.0))
    sample b:G <- MultivariateNormal(loc=loc,scale_tril=L) [over=D,iid_over=G]
    return b
export model
"""
    compiled = loads(source)
    s = torch.tensor(0.8, dtype=torch.double, requires_grad=True)
    rho = torch.tensor(0.2, dtype=torch.double, requires_grad=True)
    value = torch.linspace(-0.4, 0.6, groups * 3, dtype=torch.double).reshape(groups, 3)
    actual = trace(
        compiled.morphism, torch.zeros(1, 1), {"s": s, "rho": rho, "b": value}
    ).log_joint.sum()
    factor = s * torch.eye(3, dtype=torch.double) + rho * torch.tril(
        torch.ones(3, 3, dtype=torch.double), diagonal=-1
    )
    expected = (
        torch.distributions.MultivariateNormal(
            torch.arange(3, dtype=torch.double) * 0.1, scale_tril=factor
        )
        .log_prob(value)
        .sum()
    )
    torch.testing.assert_close(actual, expected)
    got = torch.autograd.grad(actual, (s, rho), retain_graph=True)
    want = torch.autograd.grad(expected, (s, rho))
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b)
    reference = compiled.run(
        "model", data={"s": 0.8, "rho": 0.2, "out": (0.0,)}, sites={"b": _host(value)}
    )
    assert reference.log_joint == pytest.approx(expected.item())


@pytest.mark.parametrize("groups", [2, 5])
def test_mixed_width_centered_joint_and_change_of_variables_match_independent_mvn(
    groups,
):
    n = 10
    frame = pd.DataFrame(
        {
            "y": np.arange(n) % 4,
            "x": np.linspace(-0.8, 0.8, n),
            "g": [str(i % groups) for i in range(n)],
        }
    )
    formula = formula_from_data("y~1+(1+cs(x)|joint|g)", frame)
    aux = formula_from_data("y~1+(1|joint|g)", frame)
    raw = torch.linspace(-0.15, 0.2, 10, dtype=torch.double, requires_grad=True)
    L = torch.distributions.transforms.CorrCholeskyTransform()(raw)
    scales = torch.tensor(
        [0.4, 0.6, 0.8, 0.9, 0.3], dtype=torch.double, requires_grad=True
    )
    z = torch.linspace(
        -0.7, 0.8, groups * 5, dtype=torch.double, requires_grad=True
    ).reshape(groups, 5)
    b = z @ L.T * scales
    centered = FormulaToQVRModule(
        families["acat"], distributional={"disc": aux}, reparameterize="centered"
    )
    module, complement = centered.forward(formula)
    assert centered.backward(module, complement) == formula
    layout = centered.group_layouts[0]
    assert [member.width for member in layout.members] == [1, 3, 1]
    assert [member.offset for member in layout.members] == [0, 1, 4]
    compiled = loads(module_to_source(module))
    inputs = centered.fixed_column_observations(formula)
    inputs.update(
        {
            "g_idx": torch.tensor(formula.group_indices["g"]),
            "y": torch.tensor(frame.y.to_numpy()),
        }
    )
    sites = {
        "cor_shared_joint": L,
        "sigma_g_Intercept": scales[0],
        "sigma_g_cs_x": scales[1:4],
        "dpar_disc_sigma_g_Intercept": scales[4],
        "group_coefficients_shared_joint": b,
        "intercept": torch.tensor(0.2),
        "dpar_disc_intercept": torch.tensor(-0.1),
        "cutpoint_offset": torch.tensor([0.3, -0.2], dtype=torch.double),
    }
    recorded = trace(compiled.morphism, torch.zeros(1, 1), inputs | sites)
    actual = recorded.sites["group_coefficients_shared_joint"].log_prob.sum()
    expected = (
        torch.distributions.MultivariateNormal(
            torch.zeros(5, dtype=torch.double), scale_tril=scales[:, None] * L
        )
        .log_prob(b)
        .sum()
    )
    torch.testing.assert_close(actual, expected)
    log_jacobian = groups * (scales.log().sum() + L.diagonal().log().sum())
    transformed = actual + log_jacobian
    standard = torch.distributions.Normal(0.0, 1.0).log_prob(z).sum()
    torch.testing.assert_close(transformed, standard)
    got = torch.autograd.grad(
        transformed, (raw, scales, z), create_graph=True, retain_graph=True
    )
    want = torch.autograd.grad(
        standard + 0 * (raw.sum() + scales.sum()), (raw, scales, z), create_graph=True
    )
    for a, c in zip(got, want, strict=True):
        torch.testing.assert_close(a, c, atol=2e-12, rtol=2e-12)
    reference = compiled.run(
        "model",
        data={key: _host(value) for key, value in inputs.items()},
        sites={key: _host(value) for key, value in sites.items()},
    )
    assert reference.log_joint == pytest.approx(recorded.log_joint.item(), rel=1e-7)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("family", ["cumulative", "hurdle_cumulative"])
def test_centered_full_threshold_profile_and_cs_groups_fit_and_predict(method, family):
    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 3, 0, 1, 2, 3],
            "x": [-1.0, -0.5, 0.0, 0.5, -0.8, -0.3, 0.2, 0.7],
            "g": ["a"] * 4 + ["b"] * 4,
        }
    )
    options = (
        {"num_steps": 1}
        if method == "svi"
        else {"num_warmup": 1, "num_samples": 2, "num_chains": 1}
    )
    result = fit(
        "y~x+(1+cs(x)|g)",
        data=frame,
        family=family,
        thresholds_by="g",
        reparameterize="centered",
        method=method,
        seed=11,
        **options,
    )
    draws = result.draws(num_samples=2)
    assert "cutpoint_log_spacing_value" in draws
    assert "cutpoint_log_spacing_noise" not in draws
    assert "group_coefficients_g" in draws
    expected_width = 2 if family == "hurdle_cumulative" else 3
    cuts = result.posterior_thresholds(num_samples=2)
    assert cuts.shape == (1, 2, 8, expected_width)
    assert (cuts.diff(dim=-1) > 0).all()
    fresh = pd.DataFrame({"x": [-0.4, 0.1], "g": ["a", "new"]})
    before = torch.random.get_rng_state()
    for policy in ("feasible_uncertainty", "feasible_gaussian"):
        predicted = result.posterior_epred(
            new_data=fresh,
            allow_new_levels=True,
            sample_new_levels=policy,
            num_samples=2,
            seed=9,
        )
        assert predicted.shape == (1, 2, 2, 4)
        torch.testing.assert_close(predicted.sum(-1), torch.ones(1, 2, 2).to(predicted))
    assert torch.equal(before, torch.random.get_rng_state())


def test_centered_mixed_block_new_groups_replay_joint_covariance_once_per_request():
    from quivers.formulas._prediction import prepare_prediction
    from quivers.formulas._group_prediction import remap_draw
    from quivers.inference.registry import LatentRegistry

    frame = pd.DataFrame(
        {"y": [0, 1, 2, 3] * 2, "x": np.linspace(-0.5, 0.5, 8), "g": ["a", "b"] * 4}
    )
    result = fit(
        "y~1+(1+cs(x)|joint|g)",
        data=frame,
        family="acat",
        distributional={"disc": "~1+(1|joint|g)"},
        reparameterize="centered",
        method="svi",
        num_steps=1,
        seed=13,
    )
    latents = {name: value[0, 0] for name, value in result.draws(num_samples=1).items()}
    L = torch.distributions.transforms.CorrCholeskyTransform()(torch.full((10,), 0.4))
    sigma = torch.tensor([0.4, 0.6, 0.8, 0.9, 0.3])
    known = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [-0.1, -0.2, -0.3, -0.4, -0.5]])
    latents.update(
        {
            "cor_shared_joint": L,
            "sigma_g_Intercept": sigma[0],
            "sigma_g_cs_x": sigma[1:4],
            "dpar_disc_sigma_g_Intercept": sigma[4],
            "group_coefficients_shared_joint": known,
        }
    )
    fresh = pd.DataFrame(
        {"x": np.zeros(1025), "g": ["a", *(f"new{i}" for i in range(1024))]}
    )
    context = prepare_prediction(
        result,
        fresh,
        None,
        allow_new_levels=True,
        sample_new_levels="gaussian",
        seed=31,
    )
    assert context.prior_program is not None
    assert not any(
        getattr(step, "names", ()) == ("y",) for step in context.prior_program.steps
    )
    registry = LatentRegistry.from_model(context.program, set(context.observations))
    shapes = {name: registry[name].constrained_shape for name in registry.names}
    state = torch.random.get_rng_state()
    values = remap_draw(context, latents, shapes, torch.Generator().manual_seed(31), {})
    assert torch.equal(state, torch.random.get_rng_state())
    sampled = values["group_coefficients_shared_joint"]
    torch.testing.assert_close(sampled[0], known[0])
    standardized = sampled[1:] / sigma
    torch.testing.assert_close(standardized.mean(0), torch.zeros(5), atol=0.1, rtol=0)
    torch.testing.assert_close(standardized.T.cov(), L @ L.T, atol=0.12, rtol=0)


@pytest.mark.parametrize("family", ["cumulative", "hurdle_cumulative"])
@pytest.mark.parametrize("policy", ["flexible", "sum_to_zero", "equidistant"])
def test_centered_threshold_conditional_density_derivatives_and_full_profile_rank(
    family,
    policy,
):
    categories = 4 if family == "cumulative" else 5
    frame = pd.DataFrame(
        {
            "y": np.arange(10) % categories,
            "x": np.linspace(-0.6, 0.6, 10),
            "g": ["a", "b"] * 5,
        }
    )
    formula = formula_from_data("y~x+(1|g)", frame)
    lens = FormulaToQVRModule(
        families[family].with_thresholds(policy),
        thresholds_by="g",
        reparameterize="centered",
        fixed_parameters={"hu": 0.2} if family == "hurdle_cumulative" else {},
    )
    module, _ = lens.forward(formula)
    compiled = loads(module_to_source(module))
    width = 1 if policy == "equidistant" else 2
    mean = torch.tensor([-0.2, 0.3][:width], dtype=torch.double, requires_grad=True)
    scale = torch.tensor([0.4, 0.6][:width], dtype=torch.double, requires_grad=True)
    ell = torch.tensor(
        [0.1, -0.2, 0.3, 0.4][: 2 * width], dtype=torch.double, requires_grad=True
    )
    location = torch.tensor([0.15, -0.25], dtype=torch.double, requires_grad=True)
    sites = {
        "intercept": torch.tensor(0.2),
        "beta_x": torch.tensor(-0.1),
        "sigma_g_Intercept": torch.tensor(0.7),
        "alpha_g": location,
        "cutpoint_log_spacing_mean": mean,
        "cutpoint_log_spacing_scale": scale,
        "cutpoint_log_spacing_value": ell,
    }
    inputs = lens.fixed_column_observations(formula)
    inputs.update(
        {
            "g_idx": torch.tensor(formula.group_indices["g"]),
            "y": torch.tensor(frame.y.to_numpy()),
        }
    )
    recorded = trace(compiled.morphism, torch.zeros(1, 1), inputs | sites)
    actual = recorded.sites["cutpoint_log_spacing_value"].log_prob.sum()
    expected = (
        torch.distributions.Normal(mean, scale).log_prob(ell.reshape(2, width)).sum()
    )
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual, (mean, scale, ell), retain_graph=True),
        torch.autograd.grad(expected, (mean, scale, ell), retain_graph=True),
        strict=True,
    ):
        torch.testing.assert_close(a, b)
    shape = recorded.sites["cutpoint_matrix"].value.reshape(2, 3)
    torch.testing.assert_close(
        shape.mean(-1), torch.zeros(2, dtype=torch.double), atol=1e-15, rtol=0
    )
    if policy == "equidistant":
        torch.testing.assert_close(shape.diff(dim=-1)[:, 0], shape.diff(dim=-1)[:, 1])
    profile = shape - location[:, None]
    for group in range(2):
        rows = []
        for boundary in range(3):
            dshape, dlocation = torch.autograd.grad(
                profile[group, boundary], (ell, location), retain_graph=True
            )
            rows.append(
                torch.cat(
                    (
                        dshape[group * width : group * width + width],
                        dlocation[group : group + 1],
                    )
                )
            )
        assert torch.linalg.matrix_rank(torch.stack(rows)) == width + 1
    reference = compiled.run(
        "model",
        data={key: _host(value) for key, value in inputs.items()},
        sites={key: _host(value) for key, value in sites.items()},
    )
    assert reference.log_joint == pytest.approx(recorded.log_joint.item(), rel=1e-7)


@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
def test_independent_scalar_terms_use_the_same_explicit_layout_contract(coordinates):
    from quivers.inference.mcmc.driver import MCMCResult

    frame = pd.DataFrame(
        {
            "y": [0.2, 0.4, 0.6, 0.8],
            "x": [0.0, 1.0, 0.0, 1.0],
            "g": ["a", "a", "b", "b"],
        }
    )
    result = fit(
        "y~x+(1+x||g)",
        data=frame,
        fixed_parameters={"sigma": 1.0},
        reparameterize=coordinates,
        method="svi",
        num_steps=1,
        seed=7,
    )
    assert result.prediction_plan is not None
    layouts = result.prediction_plan.group_layouts
    assert len(layouts) == 2
    assert all(
        layout.coordinate_axis is None and layout.width == 1 for layout in layouts
    )
    values = {
        "intercept": torch.tensor(0.3),
        "beta_x": torch.tensor(0.2),
        "sigma_g_Intercept": torch.tensor(0.4),
        "sigma_g_x": torch.tensor(0.7),
    }
    if coordinates == "centered":
        values.update(
            alpha_g=torch.tensor([1.0, 3.0]), beta_g_x=torch.tensor([-0.5, 0.25])
        )
    else:
        values.update(
            z_g_Intercept=torch.tensor([2.5, 7.5]),
            z_g_x=torch.tensor([-0.5 / 0.7, 0.25 / 0.7]),
        )
    assert {name for layout in layouts for name in layout.raw_sites} == set(values) - {
        "intercept",
        "beta_x",
        "sigma_g_Intercept",
        "sigma_g_x",
    }
    result = result.with_(
        posterior=MCMCResult(
            samples={name: value[None, None] for name, value in values.items()},
            log_densities=torch.zeros(1, 1),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=1,
        )
    )
    fresh = pd.DataFrame({"x": [2.0, -1.0], "g": ["b", "a"]})
    torch.testing.assert_close(
        result.posterior_linpred(new_data=fresh), torch.tensor([[[4.2, 1.6]]])
    )


@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
def test_layout_preserves_distinct_raw_sites_and_sums_joint_member_widths(coordinates):
    from quivers.formulas._group_layout import GroupBlockLayout, GroupMemberLayout

    layout = GroupBlockLayout(
        group="g",
        identity="block",
        coordinate_axis="D",
        parameterization=coordinates,
        members=(
            GroupMemberLayout(
                prefix="",
                slope="Intercept",
                axis=None,
                width=1,
                offset=0,
                coefficient="a",
                raw_site="joint",
            ),
            GroupMemberLayout(
                prefix="",
                slope="cs_x",
                axis="T",
                width=3,
                offset=1,
                coefficient="b",
                raw_site="separate",
            ),
            GroupMemberLayout(
                prefix="dpar_sigma_",
                slope="Intercept",
                axis=None,
                width=1,
                offset=4,
                coefficient="s",
                raw_site="joint",
            ),
        ),
    )
    assert layout.width == 5
    assert layout.raw_sites == {"joint": 2, "separate": 3}


def test_plated_parameter_layout_keeps_integer_validation_before_float_packing():
    source = """object G:FinSet 2
program model(n:Tensor[Int]([G]),p:Tensor[Real]([G])):G->G
    sample y:G<-Binomial(total_count=n,probs=p)
    return y
export model
"""
    compiled = loads(source)
    with pytest.raises(ValueError, match="2\\*\\*53|2\\^53|9007199254740992"):
        trace(
            compiled.morphism,
            torch.zeros(1, 1),
            {
                "n": torch.tensor([1, 2**53 + 1]),
                "p": torch.tensor([0.5, 0.5]),
                "y": torch.zeros(2),
            },
        )


def test_interleaved_lens_forwards_keep_each_matching_group_layout_complement():
    import pickle

    a = formula_from_data(
        "y~1+(1+cs(x)|g)",
        pd.DataFrame(
            {"y": [0, 1, 2, 3] * 2, "x": np.linspace(-0.5, 0.5, 8), "g": ["a", "b"] * 4}
        ),
    )
    b = formula_from_data(
        "y~x+(1|h)",
        pd.DataFrame(
            {
                "y": [0, 1, 2, 3] * 2,
                "x": np.linspace(-0.2, 0.2, 8),
                "h": ["c"] * 4 + ["d"] * 4,
            }
        ),
    )
    lens = FormulaToQVRModule(families["acat"], reparameterize="centered")
    module_a, complement_a = lens.forward(a)
    module_b, complement_b = lens.forward(b)
    assert complement_a.group_layouts != complement_b.group_layouts
    assert lens.backward(module_a, complement_a) == a
    assert lens.backward(module_b, complement_b) == b
    copied = pickle.loads(pickle.dumps(complement_a))
    assert copied == complement_a
    assert hash(copied) == hash(complement_a)
    assert lens.backward(module_a, copied) == a
