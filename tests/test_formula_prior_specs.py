"""Compositional prior selectors and sampled shrinkage reuse one emitted model."""

from contextlib import contextmanager
import math

import didactic.api as dx
import pandas as pd
import pytest
import torch

from quivers.dsl.ast_nodes import ProgramDecl, SampleStep
from quivers.dsl import Compiler
from quivers.dsl.emit import module_to_source
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.formulas import FormulaToQVRModule, families, formula_from_data
from quivers.formulas._priors import (
    PriorSpec,
    PriorSet,
    PriorTarget,
    RegularizedHorseshoe,
    R2D2,
    normalize_priors,
    resolve_priors,
)
from quivers.inference.trace import trace
from quivers.qiec.program_runtime import run_program


@contextmanager
def _double():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.double)
        yield
    finally:
        torch.set_default_dtype(previous)


def _targets():
    return [
        PriorTarget(site="beta_x", class_="b", coef="x", dpar="mu"),
        PriorTarget(site="beta_z", class_="b", coef="z", dpar="mu"),
        PriorTarget(site="intercept", class_="Intercept", coef="Intercept", dpar="mu"),
        PriorTarget(site="dpar_sigma_beta_x", class_="b", coef="x", dpar="sigma"),
        PriorTarget(site="nlpar_a_beta_x", class_="b", coef="x", nlpar="a"),
        PriorTarget(
            site="sigma_g_x", class_="sd", coef="x", group="g", dpar="mu", lower=0
        ),
        PriorTarget(site="cor_g", class_="cor", group="g"),
    ]


def test_selector_resolution_reuses_original_labels_and_strict_refinements():
    broad = PriorSpec(distribution="Normal(0,2)")
    location = PriorSpec(distribution="Normal(0,1)", dpar="mu")
    slope = PriorSpec(distribution="Normal(.3,.2)", dpar="mu", coef="x")
    selected = resolve_priors(broad + location + slope, _targets())
    assert selected["beta_x"] == slope
    assert selected["beta_z"] == location
    assert selected["dpar_sigma_beta_x"] == selected["nlpar_a_beta_x"] == broad
    assert "intercept" not in selected
    assert "sigma_g_x" not in selected


@pytest.mark.parametrize(
    "first,second",
    [
        ({"coef": "x"}, {"dpar": "mu"}),
        ({"coef": "x"}, {"coef": "x"}),
    ],
)
def test_incomparable_or_duplicate_selectors_refuse_ambiguous_override(first, second):
    rules = PriorSpec(distribution="Normal(0,1)", **first) + PriorSpec(
        distribution="Normal(1,2)", **second
    )
    with pytest.raises(ValueError, match="conflicting prior selectors.*beta_x"):
        resolve_priors(rules, _targets())


@pytest.mark.parametrize(
    "selector",
    [
        {"coef": "missing"},
        {"group": "unknown", "class_": "sd"},
        {"nlpar": "missing"},
        {"dpar": "unknown"},
        {"site": "z_g_x", "class_": None},
    ],
)
def test_unused_prior_refusal_identifies_selector_and_available_sites(selector):
    with pytest.raises(ValueError, match="unused prior selector.*available targets"):
        resolve_priors(
            normalize_priors(PriorSpec(distribution="Normal(0,1)", **selector)),
            _targets(),
        )


def test_fully_shadowed_rules_fail_and_legacy_names_use_same_resolution():
    broad = PriorSpec(distribution="Normal(0,1)", dpar="mu", coef="x")
    exact = PriorSpec(distribution="Normal(1,2)", dpar="mu", coef="x", site="beta_x")
    with pytest.raises(ValueError, match="completely shadowed"):
        resolve_priors(broad + exact, _targets())
    legacy = normalize_priors({"beta_x": "Normal(0,1)", "sigma_g_x": "HalfNormal(.4)"})
    assert set(resolve_priors(legacy, _targets())) == {"beta_x", "sigma_g_x"}
    assert normalize_priors(legacy) is legacy
    assert normalize_priors(None) == PriorSet()


@pytest.mark.parametrize(
    "factory,kwargs",
    [
        (RegularizedHorseshoe, {"df": 0}),
        (RegularizedHorseshoe, {"scale_global": math.inf}),
        (RegularizedHorseshoe, {"par_ratio": 0}),
        (RegularizedHorseshoe, {"df_slab": -1}),
        (R2D2, {"mean_R2": 1.0}),
        (R2D2, {"mean_R2": 0.0}),
        (R2D2, {"prec_R2": 0.0}),
        (R2D2, {"cons_D2": ()}),
        (R2D2, {"cons_D2": (0.5, -1.0)}),
        (PriorSpec, {"distribution": "", "class_": "b"}),
        (PriorSpec, {"distribution": "Normal(0,1)", "dpar": "mu", "nlpar": "a"}),
        (PriorSpec, {"distribution": "Normal(0,1)", "lower": 2.0, "upper": 1.0}),
    ],
)
def test_invalid_prior_specification_is_refused_before_compilation(factory, kwargs):
    with pytest.raises((TypeError, ValueError, dx.ValidationError)):
        factory(**kwargs)


def _frame():
    return pd.DataFrame(
        {
            "y": [0.1, 0.2, 0.5, 0.3],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.4, -0.2, 0.5, -0.1],
            "g": ["a", "b", "a", "b"],
        }
    )


def _model(
    hierarchy, *, group=False, categorical=False, concentration=None, bounded=False
):
    frame = _frame()
    if categorical:
        frame.y = [0, 1, 2, 1]
    formula = formula_from_data("y~0+x+z" + ("+(1||g)" if group else ""), frame)
    rules = PriorSpec(
        distribution=hierarchy,
        block="shared",
        lower=-0.5 if bounded else None,
        upper=0.8 if bounded else None,
    )
    if group:
        rules = rules + PriorSpec(
            distribution=hierarchy,
            class_="sd",
            block="shared",
            lower=0.1 if bounded else None,
            upper=0.9 if bounded else None,
        )
    lens = FormulaToQVRModule(
        families["categorical" if categorical else "gaussian"], user_priors=rules
    )
    updated, complement = lens.forward(formula)
    assert complement.prior_targets == lens.prior_targets
    checked = lower_qvr_to_qiec(updated)
    program = Compiler(updated).compile().morphism
    observations = lens.fixed_column_observations(formula) | {
        "y": torch.tensor(
            frame.y.to_numpy(), dtype=torch.long if categorical else torch.double
        )
    }
    observations = {
        name: value.double() if value.is_floating_point() else value
        for name, value in observations.items()
    }
    observations |= {
        group + "_idx": torch.tensor(indices, dtype=torch.long)
        for group, indices in formula.group_indices.items()
    }
    return frame, formula, lens, updated, checked, program, observations


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("group", [False, True])
@pytest.mark.parametrize("kind", ["horseshoe", "r2d2"])
def test_sampled_shrinkage_joint_all_latent_gradients_and_reference_match_torch(
    kind, group, bounded
):
    with _double():
        width = 3 if group else 2
        hierarchy = (
            RegularizedHorseshoe(
                df=3.0, df_global=2.0, df_slab=5.0, scale_global=0.3, scale_slab=1.2
            )
            if kind == "horseshoe"
            else R2D2(mean_R2=0.4, prec_R2=3.0, cons_D2=0.7)
        )
        frame, formula, lens, module, checked, program, observations = _model(
            hierarchy, group=group, bounded=bounded
        )
        coefficient = torch.tensor([0.2, -0.15], requires_grad=True)
        sigma = torch.tensor(0.6, requires_grad=True)
        sites = {"beta_x": coefficient[0], "beta_z": coefficient[1], "sigma": sigma}
        if kind == "horseshoe":
            global_ = torch.tensor(0.4, requires_grad=True)
            slab = torch.tensor(0.8, requires_grad=True)
            local = torch.tensor(
                [0.7, 1.4] + ([0.9] if group else []), requires_grad=True
            )
            sites |= {
                "prior_shared_global": global_,
                "prior_shared_slab": slab,
                "prior_shared_local": local,
            }
            product = sigma * global_ * local
            c2 = hierarchy.scale_slab**2 * slab
            scales = product * torch.sqrt(c2 / (c2 + product.square()))
            density = (
                torch.distributions.StudentT(
                    hierarchy.df_global, 0.0, hierarchy.scale_global
                ).log_prob(global_)
                + math.log(2)
                + torch.distributions.InverseGamma(
                    hierarchy.df_slab / 2, hierarchy.df_slab / 2
                ).log_prob(slab)
                + (
                    torch.distributions.StudentT(hierarchy.df, 0.0, 1.0).log_prob(local)
                    + math.log(2)
                ).sum()
            )
            parameters = [coefficient, sigma, global_, slab, local]
        else:
            r2 = torch.tensor(0.35, requires_grad=True)
            logits = torch.tensor(
                [-0.2, 0.3] + ([0.1] if group else []), requires_grad=True
            )
            allocation = logits.softmax(0)
            sites |= {"prior_shared_R2": r2, "prior_shared_allocation": allocation}
            scales = sigma * torch.sqrt(allocation * r2 / (1 - r2))
            density = torch.distributions.Beta(
                hierarchy.mean_R2 * hierarchy.prec_R2,
                (1 - hierarchy.mean_R2) * hierarchy.prec_R2,
            ).log_prob(r2)
            density += torch.distributions.Dirichlet(
                torch.full((width,), hierarchy.cons_D2)
            ).log_prob(allocation)
            parameters = [coefficient, sigma, r2, logits]
        mean = coefficient[0] * observations["x"] + coefficient[1] * observations["z"]
        density += (
            torch.distributions.Normal(torch.zeros(2), scales[:2])
            .log_prob(coefficient)
            .sum()
        )
        if bounded:
            standard = torch.distributions.Normal(0.0, 1.0)
            normalizer = standard.cdf(0.8 / scales[:2]) - standard.cdf(
                -0.5 / scales[:2]
            )
            density -= normalizer.log().sum()
        density += torch.distributions.HalfCauchy(2.0).log_prob(sigma)
        if group:
            scale = torch.tensor(0.3, requires_grad=True)
            z = torch.tensor([-0.5, 0.4], requires_grad=True)
            sites |= {"sigma_g_Intercept": scale, "z_g_Intercept": z}
            mean += scale * z[observations["g_idx"]]
            density += torch.distributions.HalfNormal(scales[2]).log_prob(scale)
            if bounded:
                mass = 2 * (
                    standard.cdf(0.9 / scales[2]) - standard.cdf(0.1 / scales[2])
                )
                density -= mass.log()
            density += torch.distributions.Normal(0.0, 1.0).log_prob(z).sum()
            parameters.extend((scale, z))
        density += (
            torch.distributions.Normal(mean, sigma).log_prob(observations["y"]).sum()
        )
        recorded = trace(program, torch.zeros(1, 1), observations | sites)
        torch.testing.assert_close(
            recorded.log_joint.sum(), density, atol=2e-11, rtol=2e-12
        )
        actual = torch.autograd.grad(
            recorded.log_joint.sum(), parameters, retain_graph=True
        )
        expected = torch.autograd.grad(density, parameters)
        for a, e in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, e, atol=3e-10, rtol=3e-10)
        host_data = {key: tuple(value.tolist()) for key, value in observations.items()}
        host_sites = {
            key: tuple(value.detach().tolist()) if value.ndim else value.item()
            for key, value in sites.items()
        }
        reference = run_program(
            checked, "model", data=host_data, sites=host_sites
        ).log_joint
        assert reference == pytest.approx(density.item(), abs=2e-11)
        source = module_to_source(module)
        assert source.count("sample beta_x ") == 1
        assert "prior_shared" in source


def test_integrated_selectors_preserve_catalogue_and_compositional_bounds():
    from quivers.formulas import get_prior

    frame = _frame()
    rules = (
        PriorSpec(distribution="Normal(0,1)")
        + PriorSpec(distribution="Normal(.2,.3)", coef="x", dpar="mu")
        + PriorSpec(distribution="HalfNormal(.4)", class_="sd", group="g")
        + PriorSpec(distribution="LKJCholesky(3)", class_="cor", group="g")
    )
    formula = formula_from_data("y~x+z+(1+x|g)", frame)
    lens = FormulaToQVRModule(families["gaussian"], user_priors=rules)
    module, complement = lens.forward(formula)
    actual = lens.get_prior(formula)
    assert actual["beta_x"] == "Normal(loc=0.2, scale=0.3)"
    assert actual["beta_z"] == "Normal(loc=0, scale=1)"
    assert actual["sigma_g_x"] == "HalfNormal(scale=0.4)"
    assert actual["cor_g"] == "LKJCholesky(concentration=3)"
    assert complement.prior_targets == lens.prior_targets
    assert lens.backward(module, complement) == formula
    bounded = get_prior(
        "y~x",
        data=frame,
        priors=PriorSpec(distribution="Normal(0,1)", lower=0, upper=2),
    )
    assert "Restrict" in bounded["beta_x"]
    assert "low=0.0" in bounded["beta_x"]


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("hierarchy", [RegularizedHorseshoe(par_ratio=0.4), R2D2()])
@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_shrinkage_fit_prior_view_and_fresh_rows_share_one_model(
    hierarchy, method, bounded
):
    from quivers.formulas import fit

    fitted = fit(
        "y~0+x+z",
        data=_frame(),
        fixed_parameters={"sigma": 0.7},
        priors=PriorSpec(
            distribution=hierarchy,
            block="shared",
            lower=0.1 if bounded else None,
            upper=0.8 if bounded else None,
        ),
        method=method,
        num_steps=2 if method == "svi" else None,
        num_samples=2 if method != "svi" else None,
        num_warmup=2 if method != "svi" else None,
        num_chains=1 if method != "svi" else None,
        seed=818,
    )
    before = torch.random.get_rng_state().clone()
    fresh = pd.DataFrame({"x": [0.0, 2.0], "z": [1.0, -0.5]})
    draws = fitted.draws(num_samples=2, seed=71)
    if bounded:
        assert all(
            ((draws[name] >= 0.1) & (draws[name] <= 0.8)).all()
            for name in ("beta_x", "beta_z")
        )
    expected = draws["beta_x"][..., None] * torch.tensor([0.0, 2.0])
    expected += draws["beta_z"][..., None] * torch.tensor([1.0, -0.5])
    actual = fitted.posterior_epred(new_data=fresh, num_samples=2, seed=71)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(before, torch.random.get_rng_state())
    prior = fitted.prior_view(num_samples=2, seed=19)
    if bounded:
        assert all(
            ((prior.draws()[name] >= 0.1) & (prior.draws()[name] <= 0.8)).all()
            for name in ("beta_x", "beta_z")
        )
    assert prior.posterior_predict(new_data=fresh).shape == (1, 2, 2)
    assert "prior_shared" in fitted.qvr_source


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("hierarchy", [RegularizedHorseshoe(), R2D2()])
def test_component_axes_use_one_hierarchy_and_keep_reference_labels(hierarchy, bounded):
    with _double():
        frame, formula, lens, module, checked, program, observations = _model(
            hierarchy,
            categorical=True,
        )
        target = next(t for t in lens.prior_targets if t.site == "beta_x")
        assert target.axes == ("FormulaCategoryLogit",)
        assert target.shape == (2,)
        assert target.component_labels == ("1", "2")
        assert "object FormulaPrior_shared : FinSet 4" in module_to_source(module)
        recorded = trace(program, torch.zeros(1, 1), observations)
        assert torch.isfinite(recorded.log_joint).all()
        latent_names = {
            step.vars[0]
            for declaration in module.statements
            if isinstance(declaration, ProgramDecl)
            for step in declaration.draws
            if isinstance(step, SampleStep)
        }
        values = {
            n: tuple(s.value.reshape(-1).tolist())
            if s.value.numel() > 1
            else s.value.item()
            for n, s in recorded.sites.items()
            if n in latent_names
        }
        host = {n: tuple(v.tolist()) for n, v in observations.items()}
        reference = run_program(checked, "model", data=host, sites=values).log_joint
        assert reference == pytest.approx(recorded.log_joint.sum().item(), abs=3e-10)


def test_conditional_prior_dependencies_and_cycles_use_existing_source_walker():
    from quivers.formulas import formula_to_qvr

    source = formula_to_qvr(
        "y~0+x+z", data=_frame(), priors={"beta_x": "Normal(beta_z,1)"}
    )
    assert source.index("sample beta_z") < source.index("sample beta_x")
    with pytest.raises(ValueError, match="conditional prior dependency cycle"):
        formula_to_qvr("y~0+x+z", data=_frame(), priors={"beta_x": "Normal(beta_x,1)"})


def test_single_coordinate_r2d2_has_no_degenerate_simplex_and_validates_blocks():
    from quivers.formulas import formula_to_qvr

    source = formula_to_qvr(
        "y~0+x", data=_frame(), priors=PriorSpec(distribution=R2D2())
    )
    assert "sample prior_0_allocation" not in source
    assert "let prior_0_allocation" in source
    with pytest.raises(ValueError, match="cons_D2 must have length 2"):
        formula_to_qvr(
            "y~0+x+z",
            data=_frame(),
            priors=PriorSpec(distribution=R2D2(cons_D2=(0.5,))),
        )
    conflict = PriorSpec(distribution=R2D2(), coef="x", block="shared") + PriorSpec(
        distribution=RegularizedHorseshoe(), coef="z", block="shared"
    )
    with pytest.raises(ValueError, match="inconsistent hierarchy"):
        formula_to_qvr("y~0+x+z", data=_frame(), priors=conflict)
    with pytest.raises(ValueError, match="requires b or sd"):
        formula_to_qvr(
            "y~x+(1+x|g)",
            data=_frame(),
            priors=PriorSpec(distribution=R2D2(), class_="cor"),
        )


def test_distributional_and_nonlinear_prior_scopes_share_same_catalogue():
    from quivers.formulas import get_prior

    actual = get_prior(
        "y~x",
        data=_frame(),
        distributional={"sigma": "~x"},
        priors=PriorSpec(distribution="Normal(.2,.3)", dpar="sigma"),
    )
    assert actual["dpar_sigma_beta_x"] == "Normal(loc=0.2, scale=0.3)"
    assert actual["beta_x"] != actual["dpar_sigma_beta_x"]
    nonlinear = get_prior(
        "y~a*x",
        data=_frame(),
        nonlinear={"a": "~0+z"},
        priors=PriorSpec(distribution=R2D2(autoscale=False), nlpar="a"),
    )
    assert "nlpar_a_beta_z" in nonlinear
    assert "prior_0_R2" in nonlinear


@pytest.mark.parametrize(
    "family,width", [("dirichlet", 2), ("dirichlet2", 3), ("logistic_normal", 2)]
)
def test_vector_response_hierarchy_reuses_component_layout_and_prediction(
    family, width
):
    from quivers.formulas import fit

    frame = _frame().assign(
        a=[0.2, 0.3, 0.4, 0.2], b=[0.3, 0.4, 0.2, 0.5], c=[0.5, 0.3, 0.4, 0.3]
    )
    result = fit(
        "cbind(a,b,c)~0+x",
        data=frame,
        family=family,
        priors=PriorSpec(distribution=R2D2(autoscale=False), lower=-0.5, upper=0.8),
        method="svi",
        num_steps=1,
        seed=81,
    )
    assert result.draws(num_samples=1)["beta_x"].shape[-1] == width
    assert result.posterior_predict(
        new_data=pd.DataFrame({"x": [0.0, 1.0]}), num_samples=2
    ).shape == (1, 2, 2, 3)
    if family != "logistic_normal":
        mean = result.posterior_epred(
            new_data=pd.DataFrame({"x": [0.0, 1.0]}), num_samples=2
        )
        torch.testing.assert_close(mean.sum(-1), torch.ones(1, 2, 2))


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_shared_sd_shrinkage_retains_correlated_distributional_replay(parameterization):
    from quivers.formulas import fit

    hierarchy = RegularizedHorseshoe(df=3, df_global=3)
    rules = PriorSpec(distribution=hierarchy, block="joint") + PriorSpec(
        distribution=hierarchy, class_="sd", block="joint"
    )
    result = fit(
        "y~x+(1+x|joint|g)",
        data=_frame(),
        distributional={"sigma": "~1+(1|joint|g)"},
        priors=rules,
        reparameterize=parameterization,
        method="svi",
        num_steps=1,
        seed=81,
    )
    draws = result.draws(num_samples=2)
    assert draws["prior_joint_local"].shape[-1] == 4
    assert result.posterior_epred(
        new_data=pd.DataFrame({"x": [0.0, 1.0], "g": ["b", "a"]}), num_samples=2
    ).shape == (1, 2, 2)
    prior = result.prior_view(num_samples=2)
    assert prior.posterior_predict(
        new_data=pd.DataFrame({"x": [0.0], "g": ["a"]})
    ).shape == (1, 2, 1)
