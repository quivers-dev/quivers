"""Spline hazard formulas agree with independent spline and likelihood oracles."""

import math
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.interpolate import BSpline
from scipy.integrate import quad

from quivers.continuous import MSplineBasis
from quivers.dsl import Compiler, loads, module_to_source, parse
from quivers.formulas import (
    BaselineSpec,
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.inference import trace
from quivers.qiec.program_runtime import run_program


@pytest.fixture(autouse=True)
def double_literals():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def _independent_basis(basis, values):
    degree = basis.degree
    knots = (
        (basis.boundary_knots[0],) * (degree + 1)
        + basis.interior_knots
        + (basis.boundary_knots[1],) * (degree + 1)
    )
    splines = BSpline(knots, np.eye(basis.size), degree, extrapolate=False)
    scales = (degree + 1) / (
        np.asarray(knots[degree + 1 :]) - np.asarray(knots[: -(degree + 1)])
    )
    clipped = np.clip(values, 0, basis.boundary_knots[1])
    hazard = splines(clipped) * scales
    integral = (
        splines.antiderivative()(clipped) - splines.antiderivative()(0)
    ) * scales
    tail = np.maximum(values - basis.boundary_knots[1], 0)
    integral = integral + tail[:, None] * splines(basis.boundary_knots[1]) * scales
    return hazard, integral


@pytest.mark.parametrize("weighted", [False, True])
def test_formula_score_reference_and_all_parameter_gradients_match_independent_splines(
    weighted,
):
    frame = pd.DataFrame(
        {
            "y": [0.1, 0.5, 1.0, 1.5, 3.0],
            "x": [-1.0, 0.1, 1.0, 2.0, -0.4],
            "w": [0.0, 0.5, 1.0, 2.0, 0.3],
        }
    )
    basis = MSplineBasis((0.0, 2.0), (0.7,), degree=2)
    family = families["cox"].with_baseline(basis)
    formula = formula_from_data("y~x", frame, weights="w" if weighted else None)
    lens = FormulaToQVRModule(family)
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    source = module_to_source(module)
    assert module_to_source(parse(source)) == source
    checked = Compiler(module).qiec_module
    program = loads(source).morphism
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    intercept = torch.tensor(0.2, requires_grad=True)
    beta = torch.tensor(-0.3, requires_grad=True)
    inputs = lens.fixed_column_observations(formula) | {
        "y": torch.tensor(frame.y.to_numpy()),
        "sbhaz": weights,
        "intercept": intercept,
        "beta_x": beta,
    }
    # Supply Float64 oracle inputs explicitly; the high-level fit adapter's
    # default design buffers are Float32 and retain those rounded values.
    inputs["x"] = torch.tensor(frame.x.to_numpy())
    if weighted:
        inputs["formula_weights"] = torch.tensor(frame.w.to_numpy())
    recorded = trace(program, torch.zeros(1, 1), inputs)
    m, i = _independent_basis(basis, frame.y.to_numpy())
    rate = (intercept + beta * torch.tensor(frame.x.to_numpy())).exp()
    log_density = (
        rate.log()
        + (torch.tensor(m) @ weights).log()
        - rate * (torch.tensor(i) @ weights)
    )
    row_weights = torch.tensor(frame.w.to_numpy()) if weighted else torch.ones(5)
    likelihood = (row_weights * log_density).sum()
    torch.testing.assert_close(recorded.sites["y"].log_prob.sum(), likelihood)
    prior = (
        torch.distributions.Dirichlet(torch.ones(4)).log_prob(weights)
        + torch.distributions.Normal(0.0, 5.0).log_prob(intercept)
        + torch.distributions.Normal(
            0.0, 5.0 / np.sqrt(np.mean(frame.x.to_numpy() ** 2))
        ).log_prob(beta)
    )
    expected = likelihood + prior
    torch.testing.assert_close(recorded.log_joint.sum(), expected)
    leaves = (weights, intercept, beta)
    actual_grad = torch.autograd.grad(
        recorded.log_joint.sum(), leaves, retain_graph=True
    )
    expected_grad = torch.autograd.grad(expected, leaves)
    for actual, truth in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(actual, truth, rtol=1e-9, atol=1e-9)
    host_data = {
        name: tuple(value.detach().tolist())
        for name, value in inputs.items()
        if value.ndim
    }
    host_sites = {
        "sbhaz": tuple(weights.detach().tolist()),
        "intercept": float(intercept.detach()),
        "beta_x": float(beta.detach()),
    }
    host_data.pop("sbhaz")
    assert run_program(
        checked, "model", data=host_data, sites=host_sites
    ).log_joint == pytest.approx(expected.item(), rel=1e-10)


@pytest.mark.parametrize("link", ["log", "identity", "softplus", "squareplus"])
def test_family_links_preserve_serializable_basis_and_effective_prior(link):
    basis = MSplineBasis((0.0, 2.0), (0.5,), degree=2)
    family = families["cox"].with_baseline(basis).with_links(mu=link)
    assert (
        family.baseline.resolve(
            formula_from_data("y~1", pd.DataFrame({"y": [0.1, 0.5, 1.0, 1.5]}))
        )
        == basis
    )
    assert pickle.loads(pickle.dumps(family)) == family
    assert family.model_validate_json(family.model_dump_json()) == family
    priors = get_prior(
        "y~1",
        data=pd.DataFrame({"y": [0.1, 0.5, 1.0, 1.5]}),
        family=family,
        priors={"sbhaz": "Dirichlet(2.0)"},
    )
    assert "sbhaz" in priors


def test_prediction_retains_baseline_and_uses_explicit_finite_mean_budget():
    frame = pd.DataFrame(
        {"y": [0.1, 0.5, 1.0, 1.5, 2.0], "x": [-1.0, 0.0, 1.0, 0.3, 0.2]}
    )
    result = fit("y~x", data=frame, family="cox", method="svi", num_steps=1, seed=4)
    before = torch.get_rng_state().clone()
    new = pd.DataFrame({"x": [0.2, -0.1]})
    raw = result.posterior_linpred(new_data=new, num_samples=3, seed=7)
    rate = result.posterior_linpred(transform=True, new_data=new, num_samples=3, seed=7)
    torch.testing.assert_close(rate, raw.exp())
    values = result.posterior_predict(new_data=new, num_samples=3, seed=7)
    assert values.shape == (1, 3, 2) and (values > 0).all()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(
        values, result.posterior_predict(new_data=new, num_samples=3, seed=7)
    )
    with pytest.raises(TypeError, match="analytic response expectation"):
        result.posterior_epred(new_data=new, num_samples=1)
    expected = result.posterior_epred(
        new_data=new, num_samples=2, mc_samples=200, seed=7
    )
    assert expected.shape == (1, 2, 2) and torch.isfinite(expected).all()
    assert torch.equal(before, torch.get_rng_state())
    # This AST, including literal knots and coefficients, is the retained plan.
    original = module_to_source(result.prediction_plan.module)
    result.posterior_predict(
        new_data=pd.DataFrame({"x": [100.0]}), num_samples=1, seed=3
    )
    assert module_to_source(result.prediction_plan.module) == original


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_small_inference_smoke(method):
    options = (
        {"num_steps": 1}
        if method == "svi"
        else {"num_warmup": 1, "num_samples": 1, "num_chains": 1}
    )
    result = fit(
        "y~1",
        data=pd.DataFrame({"y": [0.2, 0.4, 0.8, 1.1, 1.5]}),
        family="cox",
        method=method,
        seed=12,
        **options,
    )
    assert torch.isfinite(result.log_likelihood(num_samples=1)["y"]).all()


@pytest.mark.parametrize(
    "values", [[0.0, 0.5, 1.0], [-0.1, 0.5, 1.0], [0.2, float("inf"), 1.0]]
)
def test_invalid_response_times_refuse(values):
    with pytest.raises(ValueError, match="positive|finite"):
        fit(
            "y~1",
            data=pd.DataFrame({"y": values}),
            family="cox",
            method="svi",
            num_steps=0,
        )


def test_explicit_finite_moment_monte_carlo_matches_independent_survival_integral():
    from quivers.continuous import CoxPH
    from quivers.formulas._posterior import response_mean

    basis = MSplineBasis((0.0, 2.0), (0.7,), degree=2)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
    rate = 0.8
    law = CoxPH(basis, weights, rate=rate)

    def survival(t):
        _, i = _independent_basis(basis, np.array([t]))
        return math.exp(-rate * float(i[0] @ weights.numpy()))

    mean = (
        quad(survival, 0, 2, epsabs=1e-12)[0]
        + quad(survival, 2, np.inf, epsabs=1e-12)[0]
    )
    second = (
        quad(lambda t: 2 * t * survival(t), 0, 2, epsabs=1e-12)[0]
        + quad(lambda t: 2 * t * survival(t), 2, np.inf, epsabs=1e-12)[0]
    )
    count = 20000
    with torch.random.fork_rng():
        torch.manual_seed(57)
        estimate = response_mean(law, mc_samples=count).item()
    assert abs(estimate - mean) < 5 * math.sqrt((second - mean * mean) / count)
    # A budget does not silently replace an existing analytic mean or make
    # an unverified infinite-mean law eligible for Monte Carlo expectations.
    assert (
        response_mean(torch.distributions.Exponential(torch.tensor(2.0)), mc_samples=1)
        == 0.5
    )
    with pytest.raises(ValueError, match="undefined|nonfinite"):
        response_mean(torch.distributions.Cauchy(0.0, 1.0), mc_samples=10)


@pytest.mark.parametrize(
    "configuration",
    [
        {"boundary_knots": (0.0,)},
        {"boundary_knots": (0.0, 1.0, 2.0)},
        {"boundary_knots": (0.1, 1.0)},
        {"interior_knots": (0.5,)},
        {"df": 3, "degree": 3},
    ],
)
def test_baseline_configuration_is_checked_instead_of_silently_repaired(configuration):
    formula = formula_from_data("y~1", pd.DataFrame({"y": [0.1, 0.5, 1.0, 1.5]}))
    with pytest.raises(ValueError):
        BaselineSpec(**configuration).resolve(formula)


def test_vector_prior_uses_existing_parsed_auxiliary_prior_path():
    frame = pd.DataFrame({"y": [0.1, 0.5, 1.0, 1.5]})
    family = families["cox"].with_baseline(MSplineBasis((0.0, 2.0), degree=1))
    priors = get_prior(
        "y~1", data=frame, family=family, priors={"sbhaz": "Dirichlet([2.0,3.0])"}
    )
    assert "2.0" in priors["sbhaz"] and "3.0" in priors["sbhaz"]
    result = fit(
        "y~1",
        data=frame,
        family=family,
        priors={"sbhaz": "Dirichlet([2.0,3.0])"},
        method="svi",
        num_steps=1,
        seed=5,
    )
    assert result.posterior_predict(num_samples=2).shape == (1, 2, 4)


@pytest.mark.parametrize("count", [64, 130])
def test_shared_monte_carlo_mean_preserves_large_subnormal_zero_and_mixed_sign_values(
    count,
):
    from quivers.formulas._posterior import response_mean

    class FiniteLaw(torch.distributions.Distribution):
        arg_constraints = {}
        support = torch.distributions.constraints.real
        finite_moment_orders = frozenset({1})

        def __init__(self):
            super().__init__(batch_shape=torch.Size([4]))
            self.offset = 0

        @property
        def mean(self):
            raise NotImplementedError

        def sample(self, sample_shape=torch.Size()):
            size = sample_shape[0]
            values = torch.tensor(
                [1e307, 1e-320, 0.0, 1e307], dtype=torch.float64
            ).repeat(size, 1)
            signs = (torch.arange(self.offset, self.offset + size) % 2) * 2 - 1
            values[:, 3] *= signs
            self.offset += size
            return values

    result = response_mean(FiniteLaw(), count)
    expected = torch.tensor([1e307, 1e-320, 0.0, 0.0], dtype=torch.float64)
    assert torch.equal(result, expected)


@pytest.mark.parametrize("groups", [2, 3])
@pytest.mark.parametrize("weighted", [False, True])
def test_group_baseline_source_joint_and_all_gradients_match_independent_splines(
    groups, weighted
):
    frame = pd.DataFrame(
        {
            "y": [0.15, 0.45, 0.8, 1.2, 1.8, 2.7],
            "x": [-1.0, 0.2, 0.6, -0.4, 0.9, 0.1],
            "g": [f"g{index % groups}" for index in [1, 0, 2, 1, 2, 0]],
            "w": [0.0, 0.5, 1.0, 2.0, 0.7, 1.0],
        }
    )
    formula = formula_from_data(
        "y~x", frame, groups=("g",), weights="w" if weighted else None
    )
    assert formula.random_terms == ()
    basis = MSplineBasis((0.0, 2.0), (0.7,), degree=1)
    family = families["cox"].with_baseline(basis, by="g")
    lens = FormulaToQVRModule(family, user_priors={"sbhaz": "Dirichlet([2.,3.,4.])"})
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    assert family.model_validate_json(family.model_dump_json()) == family
    source = module_to_source(module)
    assert module_to_source(parse(source)) == source
    checked = Compiler(module).qiec_module
    profiles = torch.tensor(
        [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.1, 0.7, 0.2]][:groups], requires_grad=True
    )
    intercept = torch.tensor(0.2, requires_grad=True)
    beta = torch.tensor(-0.3, requires_grad=True)
    codes = torch.tensor(formula.group_indices["g"])
    inputs = lens.fixed_column_observations(formula) | {
        "x": torch.tensor(frame.x.to_numpy()),
        "g_idx": codes,
        "y": torch.tensor(frame.y.to_numpy()),
        "sbhaz": profiles,
        "intercept": intercept,
        "beta_x": beta,
    }
    m, integral = _independent_basis(basis, frame.y.to_numpy())
    rate = (intercept + beta * inputs["x"]).exp()
    selected = profiles[codes]
    rows = (
        rate.log()
        + (torch.tensor(m) * selected).sum(-1).log()
        - rate * (torch.tensor(integral) * selected).sum(-1)
    )
    row_weights = inputs["formula_weights"] if weighted else torch.ones(len(frame))
    expected = (rows * row_weights).sum() + torch.distributions.Dirichlet(
        torch.tensor([2.0, 3.0, 4.0])
    ).log_prob(profiles).sum()
    expected = (
        expected
        + torch.distributions.Normal(0.0, 5.0).log_prob(intercept)
        + torch.distributions.Normal(
            0.0, 5.0 / np.sqrt(np.mean(frame.x.to_numpy() ** 2))
        ).log_prob(beta)
    )
    leaves = (profiles, intercept, beta)
    for runtime in [Compiler(module).compile().morphism, loads(source).morphism]:
        recorded = trace(runtime, torch.zeros(1, 1), inputs)
        torch.testing.assert_close(recorded.log_joint.sum(), expected)
        for actual, truth in zip(
            torch.autograd.grad(recorded.log_joint.sum(), leaves, retain_graph=True),
            torch.autograd.grad(expected, leaves, retain_graph=True),
            strict=True,
        ):
            torch.testing.assert_close(actual, truth, atol=1e-9, rtol=1e-9)
    host = {
        name: value.detach().tolist() if value.ndim else float(value.detach())
        for name, value in inputs.items()
    }

    def tuples(value):
        return (
            tuple(tuples(item) for item in value) if isinstance(value, list) else value
        )

    host = {name: tuples(value) for name, value in host.items()}
    sites = {name: host.pop(name) for name in ("sbhaz", "intercept", "beta_x")}
    assert run_program(
        checked, "model", data=host, sites=sites
    ).log_joint == pytest.approx(expected.item(), rel=1e-10)
    # Independent integrated basis values certify per-group normalization.
    _, endpoint = _independent_basis(basis, np.array([2.0]))
    torch.testing.assert_close(
        torch.tensor(endpoint) @ profiles.T, torch.ones(1, groups)
    )


def test_structural_group_coding_reuses_effect_coding_without_adding_terms():
    frame = pd.DataFrame({"y": [0.2, 0.5, 0.7], "g": ["z", "a", "z"]})
    structural = formula_from_data("y~1", frame, groups=("g",))
    effects = formula_from_data("y~(1|g)", frame)
    assert structural.group_levels == effects.group_levels == {"g": ("a", "z")}
    assert structural.group_indices == effects.group_indices == {"g": (1, 0, 1)}
    assert structural.random_terms == ()
    for invalid in [frame.drop(columns="g"), frame.assign(g=["a", None, "b"])]:
        with pytest.raises(ValueError, match="grouping factor"):
            formula_from_data("y~1", invalid, groups=("g",))


def test_censored_knot_selection_uses_exact_events_and_full_followup_span():
    frame = pd.DataFrame(
        {
            "y": [0.2, 0.4, 0.8, 1.5, 9.0, 10.0],
            "status": [0, 0, 0, 0, 1, 2],
            "upper": [None, None, None, None, None, 30.0],
        }
    )
    formula = formula_from_data("y | cens(status,upper) ~ 1", frame)
    spec = families["cox"].with_baseline(df=5, degree=2).baseline
    basis = spec.resolve(formula)
    np.testing.assert_allclose(
        basis.interior_knots, np.quantile([0.2, 0.4, 0.8, 1.5], [1 / 3, 2 / 3])
    )
    assert basis.boundary_knots == pytest.approx((0.0, 30.0 + (30.0 - 0.2) / 50))
    only_censored = formula_from_data(
        "y | cens(status) ~ 1", pd.DataFrame({"y": [1.0, 2.0], "status": [1, 1]})
    )
    with pytest.raises(ValueError, match="exact events"):
        spec.resolve(only_censored)
    assert families["cox"].with_baseline(
        MSplineBasis((0.0, 3.0), (), degree=1)
    ).baseline.resolve(only_censored).boundary_knots == (0.0, 3.0)


@pytest.mark.parametrize("policy", ["uncertainty", "old_levels", "gaussian"])
def test_group_baseline_predictions_retain_profiles_under_effect_selection(policy):
    frame = pd.DataFrame(
        {
            "y": [0.2, 0.5, 1.0, 1.5],
            "x": [-1.0, 0.0, 1.0, 0.5],
            "g": ["a", "b", "a", "b"],
        }
    )
    result = fit(
        "y~x",
        data=frame,
        family=families["cox"].with_baseline(
            MSplineBasis((0.0, 2.0), (), degree=1), by="g"
        ),
        method="svi",
        num_steps=1,
    )
    new = pd.DataFrame({"x": [0.0, 0.5, 0.5], "g": ["a", "new", "new"]})
    before = torch.get_rng_state().clone()
    values = result.posterior_predict(
        new_data=new,
        num_samples=2,
        seed=13,
        allow_new_levels=True,
        sample_new_levels=policy,
        re_formula="~0",
    )
    assert values.shape == (1, 2, 3) and (values > 0).all()
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(
        values,
        result.posterior_predict(
            new_data=new,
            num_samples=2,
            seed=13,
            allow_new_levels=True,
            sample_new_levels=policy,
            re_formula="~0",
        ),
    )
    with pytest.raises(ValueError, match="unknown groups|missing predictors"):
        result.posterior_predict(
            new_data=new.drop(columns="g"), num_samples=1, re_formula="~0"
        )


@pytest.mark.parametrize("policy", ["uncertainty", "old_levels", "gaussian"])
def test_baseline_new_identity_policy_reuses_joint_donors_and_conditional_simplex_law(
    policy,
):
    from quivers.formulas._prediction import prepare_prediction
    from quivers.formulas._group_prediction import remap_draw
    from quivers.inference.registry import LatentRegistry

    frame = pd.DataFrame({"y": [0.2, 0.5, 1.0, 1.5], "g": ["a", "b", "a", "b"]})
    result = fit(
        "y~1+(1|g)",
        data=frame,
        family=families["cox"].with_baseline(
            MSplineBasis((0.0, 2.0), (0.7,), degree=1), by="g"
        ),
        method="svi",
        num_steps=1,
    )
    context = prepare_prediction(
        result,
        pd.DataFrame({"g": ["new", "a", "new"]}),
        None,
        allow_new_levels=True,
        sample_new_levels=policy,
    )
    registry = LatentRegistry.from_model(
        context.program, observed_names=set(context.observations)
    )
    shapes = {name: registry[name].constrained_shape for name in registry.names}
    latents = {
        "intercept": torch.tensor(0.0),
        "sigma_g_Intercept": torch.tensor(0.4),
        "z_g_Intercept": torch.tensor([1.0, 3.0]),
        "sbhaz": torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]),
    }
    generator = torch.Generator().manual_seed(17)
    fixed_donors = {}
    samples = []
    indices = []
    before = torch.get_rng_state().clone()
    for _ in range(256 if policy == "gaussian" else 16):
        donors = {}
        values = remap_draw(context, latents, shapes, generator, fixed_donors, donors)
        torch.testing.assert_close(values["sbhaz"][1], latents["sbhaz"][0])
        torch.testing.assert_close(
            values["z_g_Intercept"][1], latents["z_g_Intercept"][0]
        )
        assert (values["sbhaz"] > 0).all()
        torch.testing.assert_close(values["sbhaz"].sum(-1), torch.ones(2))
        samples.append(values["sbhaz"][0])
        if policy != "gaussian":
            index = int(donors["g"][0])
            indices.append(index)
            torch.testing.assert_close(values["sbhaz"][0], latents["sbhaz"][index])
            torch.testing.assert_close(
                values["z_g_Intercept"][0], latents["z_g_Intercept"][index]
            )
    assert torch.equal(before, torch.get_rng_state())
    if policy == "gaussian":
        matrix = torch.stack(samples)
        torch.testing.assert_close(
            matrix.mean(0), torch.full((3,), 1 / 3), atol=0.07, rtol=0.0
        )
        assert not torch.isclose(matrix, latents["sbhaz"][0]).all(-1).any()
        assert not torch.isclose(matrix, latents["sbhaz"][1]).all(-1).any()
    elif policy == "old_levels":
        assert len(set(indices)) == 1
    else:
        assert set(indices) == {0, 1}


def test_normalized_group_profiles_have_no_duplicate_rate_scale_coordinate():
    # Independent M-spline values identify two shape coordinates per group
    # plus one population rate. There is no free profile multiplier.
    basis = MSplineBasis((0.0, 2.0), (0.7,), degree=1)
    matrix, _ = _independent_basis(basis, np.array([0.1, 0.3, 0.6, 1.0, 1.4, 1.9]))
    design = torch.tensor(matrix)

    def log_hazards(coordinates):
        free = coordinates[:4].reshape(2, 2)
        profiles = torch.cat((free, 1 - free.sum(-1, keepdim=True)), dim=-1)
        return ((design @ profiles.T).log() + coordinates[4]).reshape(-1)

    jacobian = torch.func.jacrev(log_hazards)(torch.tensor([0.2, 0.3, 0.6, 0.3, 0.1]))
    assert torch.linalg.matrix_rank(jacobian) == 5
