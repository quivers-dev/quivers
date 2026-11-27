"""Nonlinear parameter graphs reuse checked QVR, prior, and posterior paths."""

import math

import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.dsl.qiec_lowering import lower_qvr_to_qiec
from quivers.formulas import NonlinearExpression, fit, formula_to_qvr, get_prior
from quivers.formulas._fit import _prepare_formula_entry
from quivers.formulas._prediction import prepare_prediction
from quivers.inference.trace import trace
from quivers.inference.mcmc.driver import MCMCResult
from quivers.qiec.program_runtime import run_program


def _frame():
    return pd.DataFrame(
        {
            "y": [0.2, 0.4, 0.8, 1.0],
            "x": [0.125, 0.25, 0.5, 1.0],
            "z": [-0.5, 0.25, 0.5, 1.0],
            "g": ["a", "b", "a", "b"],
            "w": [0.0, 0.5, 1.0, 2.0],
        }
    )


def _recorded(result, values):
    original = result.draws(num_samples=1)
    samples = {
        name: value[0, 0].expand(2, 3, *value.shape[2:]).clone()
        for name, value in original.items()
    }
    samples.update(
        {
            name: value.expand(2, 3, *value.shape).clone()
            for name, value in values.items()
        }
    )
    return result.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(2, 3),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.long),
            num_warmup=0,
            num_samples=3,
        )
    )


def test_nonlinear_joint_all_gradients_and_checked_reference_agree_with_independent_equations():
    frame = _frame()
    nonlinear = {
        "a": "~1+z",
        "log_rate": "~1",
        "rate": NonlinearExpression(expression="exp(log_rate)"),
        "s0": "~1",
        "s1": "~1",
    }
    distributional = {"sigma": NonlinearExpression(expression="s0+s1*x")}
    names = [
        "nlpar_a_intercept",
        "nlpar_a_beta_z",
        "nlpar_log_rate_intercept",
        "nlpar_s0_intercept",
        "nlpar_s1_intercept",
    ]
    prepared = _prepare_formula_entry(
        "y ~ a*(1-exp(-rate*x))",
        data=frame,
        nonlinear=nonlinear,
        distributional=distributional,
        weights="w",
        priors={name: "Normal(0.0,2.0)" for name in names},
    )
    lens, parsed = prepared.lens, prepared.formula
    module, complement = lens.forward(parsed)
    assert lens.backward(module, complement) == parsed
    checked = lower_qvr_to_qiec(module)
    theta = torch.tensor(
        [1.5, 0.2, -0.3, -0.5, 0.1], dtype=torch.float64, requires_grad=True
    )
    observations = lens.fixed_column_observations(parsed) | {
        "y": torch.tensor(frame.y.to_numpy(), dtype=torch.float64)
    }
    observations = {
        name: value.double() if value.is_floating_point() else value
        for name, value in observations.items()
    }
    observations["nlcov_x"] = observations["nlcov_x"].clone().requires_grad_()
    sites = dict(zip(names, theta.unbind(), strict=True))
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        native = trace(
            Compiler(module).compile().morphism, torch.zeros(1, 1), observations | sites
        )
    finally:
        torch.set_default_dtype(old)
    x, z = observations["nlcov_x"], observations["nlpar_a_z"]
    mu = (theta[0] + theta[1] * z) * (1 - torch.exp(-theta[2].exp() * x))
    scale = torch.exp(theta[3] + theta[4] * x)
    likelihood = torch.distributions.Normal(mu, scale).log_prob(observations["y"])
    expected = (
        likelihood * observations["formula_weights"]
    ).sum() + torch.distributions.Normal(
        torch.zeros_like(theta), torch.full_like(theta, 2.0)
    ).log_prob(theta).sum()
    torch.testing.assert_close(native.log_joint.sum(), expected)
    actual_grads = torch.autograd.grad(
        native.log_joint.sum(), (theta, x), retain_graph=True
    )
    oracle_grads = torch.autograd.grad(expected, (theta, x))
    for actual, oracle in zip(actual_grads, oracle_grads, strict=True):
        torch.testing.assert_close(actual, oracle)
    reference = run_program(
        checked,
        "model",
        data={
            name: tuple(value.detach().tolist()) for name, value in observations.items()
        },
        sites={name: float(value.detach()) for name, value in sites.items()},
    )
    assert reference.log_joint == pytest.approx(float(expected.detach()), abs=2e-11)
    assert set(
        get_prior(
            "y ~ a*(1-exp(-rate*x))",
            data=frame,
            nonlinear=nonlinear,
            distributional=distributional,
        )
    ) == set(names)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_nonlinear_fit_native_methods_have_finite_gradients_and_predictions(method):
    counts = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 2, "num_chains": 1}
    )
    result = fit(
        "y ~ a*exp(-rate*x)",
        data=_frame(),
        nonlinear={"a": "~1", "rate": "~1"},
        fixed_parameters={"sigma": 0.5},
        priors={
            "nlpar_a_intercept": "Normal(1.0,0.5)",
            "nlpar_rate_intercept": "Normal(0.0,0.2)",
        },
        initial_values={"nlpar_a_intercept": 1.0, "nlpar_rate_intercept": 0.1},
        method=method,
        seed=9,
        **counts,
    )
    assert all(
        bool(torch.isfinite(value).all())
        for value in result.draws(num_samples=2).values()
    )
    assert bool(torch.isfinite(result.posterior_epred(num_samples=2)).all())
    assert result.posterior_predict(num_samples=2, seed=3).shape == (1, 2, 4)


def test_derived_parameter_and_distributional_roots_replay_on_new_rows_once(
    monkeypatch,
):
    result = fit(
        "y ~ a*(1-exp(-rate*x))",
        data=_frame(),
        nonlinear={
            "a": "~1",
            "log_rate": "~1",
            "rate": NonlinearExpression(expression="exp(log_rate)"),
            "log_sigma": "~1",
        },
        distributional={"sigma": NonlinearExpression(expression="log_sigma+x")},
        method="svi",
        num_steps=1,
    )
    result = _recorded(
        result,
        {
            "nlpar_a_intercept": torch.tensor(2.0),
            "nlpar_log_rate_intercept": torch.tensor(math.log(0.5)),
            "nlpar_log_sigma_intercept": torch.tensor(-1.0),
        },
    )
    compile_ = Compiler.compile
    calls = []

    def counted(self):
        calls.append(self)
        return compile_(self)

    monkeypatch.setattr(Compiler, "compile", counted)
    fresh = pd.DataFrame({"x": [0.25, 0.75, 1.25]})
    expected = 2 * (1 - torch.exp(-0.5 * torch.tensor(fresh.x.to_numpy())))
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh), expected.expand(2, 3, 3)
    )
    assert len(calls) == 1
    torch.testing.assert_close(
        result.posterior_linpred(parameter="rate", new_data=fresh),
        torch.full((2, 3, 3), 0.5),
    )
    torch.testing.assert_close(
        result.posterior_linpred(parameter="sigma", new_data=fresh, transform=True),
        torch.exp(-1 + torch.tensor(fresh.x.to_numpy())).expand(2, 3, 3),
    )
    assert set(result.fixed_effects(parameter="a")) == {"Intercept"}
    assert "nlpar_a_intercept" in result.qvr_source


@pytest.mark.parametrize(
    "formula,nonlinear,message",
    [
        (
            "y ~ a",
            {
                "a": NonlinearExpression(expression="b"),
                "b": NonlinearExpression(expression="a"),
            },
            "cyclic",
        ),
        ("y ~ a+x", {"a": "~1", "b": "~1"}, "unused"),
        ("y ~ a+missing", {"a": "~1"}, "unknown nonlinear symbols"),
        ("y ~ a", {"a": "other~1"}, "mismatched"),
        ("y ~ a", {"a": "1"}, "requires '~'"),
        ("y ~ a", {"a": "~1", "mu": "~1"}, "reserved"),
        ("y ~ a", {"a": NonlinearExpression(expression="sigma")}, "family outputs"),
        ("y ~ a", {"a": NonlinearExpression(expression="1; return 2")}, "one nonempty"),
    ],
)
def test_nonlinear_graph_dependency_errors_are_explicit(formula, nonlinear, message):
    with pytest.raises(ValueError, match=message):
        formula_to_qvr(formula, data=_frame(), nonlinear=nonlinear)


def test_nonlinear_raw_integer_covariates_keep_checked_carrier_on_new_rows():
    frame = _frame().assign(k=[1, 2, 3, 4])
    result = fit(
        "y ~ a*k",
        data=frame,
        nonlinear={"a": "~1"},
        fixed_parameters={"sigma": 0.5},
        method="svi",
        num_steps=1,
    )
    assert "nlcov_k : Tensor[Int]" in result.qvr_source
    assert result.posterior_epred(
        new_data=pd.DataFrame({"k": [2, 5]}), num_samples=1
    ).shape == (1, 1, 2)
    with pytest.raises(ValueError, match="numeric carrier"):
        result.posterior_epred(new_data=pd.DataFrame({"k": [2.0, 5.0]}), num_samples=1)


@pytest.mark.parametrize(
    "name",
    [
        "categorical",
        "multinomial",
        "dirichlet",
        "dirichlet2",
        "dirichlet_multinomial",
        "logistic_normal",
    ],
)
@pytest.mark.parametrize("rows", [2, 3])
def test_labelled_nonlinear_response_roots_match_native_laws_and_all_head_gradients(
    name, rows
):
    from quivers.formulas import families
    from quivers.continuous._additive_log_ratio import AdditiveLogRatioNormal
    from quivers.continuous._multinomial import Multinomial, DirichletMultinomial

    frame = pd.DataFrame(
        {
            "x": [0.25, 0.5, 0.75][:rows],
            "A": [0.25, 0.2, 0.3][:rows],
            "B": [0.25, 0.3, 0.2][:rows],
            "C": [0.5, 0.5, 0.5][:rows],
            "y": [0, 1, 2][:rows],
        }
    )
    if name == "categorical":
        frame["y"] = pd.Categorical(frame.y, categories=[0, 1, 2])
    if name in {"multinomial", "dirichlet_multinomial"}:
        frame[["A", "B", "C"]] = pd.DataFrame([[1, 2, 3], [0, 1, 2], [2, 0, 1]][:rows])
    response = "y" if name == "categorical" else "cbind(A,B,C)"
    family = families[name]
    if name == "logistic_normal":
        family = family.with_(response=family.response.with_(reference="B"))
    labels = (
        tuple(str(index) for index in range(rows))
        if name == "categorical"
        else ("A", "B", "C")
    )
    free = tuple(labels[index] for index in family.response.coordinate_indices(labels))
    # Reverse authored order: matching is by component name, never mapping position.
    components = dict(
        reversed(
            [
                (label, "n*x+p" if index == 0 else "p+x" if index == 1 else "n-p*x")
                for index, label in enumerate(free)
            ]
        )
    )
    nonlinear = {"n": "~1", "p": "~1"}
    fixed = (
        {"phi": 3.0}
        if name in {"dirichlet", "dirichlet_multinomial"}
        else {"sigma": 0.5}
        if name == "logistic_normal"
        else {}
    )
    prepared = _prepare_formula_entry(
        f"{response}~0",
        data=frame,
        family=family,
        nonlinear=nonlinear,
        distributional={"mu": NonlinearExpression(expression=components)},
        fixed_parameters=fixed,
        priors={
            "nlpar_n_intercept": "Normal(0.0,2.0)",
            "nlpar_p_intercept": "Normal(0.0,2.0)",
        },
    )
    lens, parsed = prepared.lens, prepared.formula
    module, _ = lens.forward(parsed)
    theta = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    observations = lens.fixed_column_observations(parsed)
    from quivers.formulas._response import response_tensor
    from quivers.formulas.formula import _qvr_name

    observations[_qvr_name(parsed.response_name)] = response_tensor(parsed, family)
    sites = {"nlpar_n_intercept": theta[0], "nlpar_p_intercept": theta[1]}
    if name == "logistic_normal":
        sites["rescor"] = torch.eye(2, dtype=torch.float64)
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        actual = trace(
            Compiler(module).compile().morphism, torch.zeros(1, 1), observations | sites
        )
    finally:
        torch.set_default_dtype(old)
    x = torch.tensor(frame.x.to_numpy())
    coordinates = torch.stack(
        [
            theta[0] * x + theta[1]
            if index == 0
            else theta[1] + x
            if index == 1
            else theta[0] - theta[1] * x
            for index in range(len(free))
        ],
        -1,
    )
    if name == "logistic_normal":
        law = AdditiveLogRatioNormal(
            coordinates, torch.eye(2, dtype=torch.float64) * 0.5, reference=1
        )
    else:
        full = theta.new_zeros((rows, len(labels)))
        for index, coordinate in enumerate(family.response.coordinate_indices(labels)):
            full[:, coordinate] = coordinates[:, index]
        probs = full.softmax(-1)
        if name == "categorical":
            law = torch.distributions.Categorical(logits=full)
        elif name == "multinomial":
            law = Multinomial(observations["formula_trials"], logits=full)
        elif name == "dirichlet":
            law = torch.distributions.Dirichlet(probs * 3)
        elif name == "dirichlet_multinomial":
            law = DirichletMultinomial(observations["formula_trials"], probs * 3)
        else:
            law = torch.distributions.Dirichlet(full.exp())
    expected = law.log_prob(observations[_qvr_name(parsed.response_name)]).sum()
    raw = (
        actual.sites[_qvr_name(parsed.response_name)]
        .sampleable.raw_log_prob_pointwise(
            observations[_qvr_name(parsed.response_name)]
        )
        .sum()
    )
    torch.testing.assert_close(raw, expected)
    torch.testing.assert_close(
        torch.autograd.grad(raw, theta, retain_graph=True)[0],
        torch.autograd.grad(expected, theta)[0],
    )

    def host(value):
        if isinstance(value, list):
            return tuple(host(item) for item in value)
        return value

    reference_data = {key: host(value.tolist()) for key, value in observations.items()}
    if name == "categorical":
        reference_data[_qvr_name(parsed.response_name)] = tuple(
            int(value)
            for value in observations[_qvr_name(parsed.response_name)].tolist()
        )
    reference = run_program(
        lower_qvr_to_qiec(module),
        "model",
        data=reference_data,
        sites={key: host(value.detach().tolist()) for key, value in sites.items()},
    )
    assert reference.log_joint == pytest.approx(
        float(actual.log_joint.sum().detach()), abs=2e-11
    )


def test_nonlinear_group_parameters_reuse_known_and_new_group_prediction_metadata():
    result = fit(
        "y ~ a*exp(b*x)",
        data=_frame(),
        nonlinear={"a": "~1+(1||g)", "b": "~1"},
        fixed_parameters={"sigma": 0.5},
        method="svi",
        num_steps=1,
    )
    result = _recorded(
        result,
        {
            "nlpar_a_intercept": torch.tensor(1.0),
            "nlpar_b_intercept": torch.tensor(0.2),
            "nlpar_a_sigma_g_Intercept": torch.tensor(0.3),
            "nlpar_a_z_g_Intercept": torch.tensor([1.0, -1.0]),
        },
    )
    fresh = pd.DataFrame({"g": ["b", "a", "b"], "x": [0.25, 0.5, 0.75]})
    expected = torch.tensor([0.7, 1.3, 0.7]) * torch.exp(
        0.2 * torch.tensor([0.25, 0.5, 0.75])
    )
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh),
        expected.expand(2, 3, 3),
        check_dtype=False,
    )
    expected_no_group = torch.exp(0.2 * torch.tensor([0.25, 0.5, 0.75]))
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh, re_formula="~0"),
        expected_no_group.expand(2, 3, 3),
        check_dtype=False,
    )
    assert result.random_effects(parameter="a")["g"]["Intercept"].shape == (2, 3, 2)
    assert result.group_levels == {"g": ("a", "b")}
    new = pd.DataFrame({"g": ["new", "new"], "x": [0.25, 0.25]})
    predicted = result.posterior_epred(
        new_data=new, allow_new_levels=True, sample_new_levels="gaussian", seed=17
    )
    torch.testing.assert_close(predicted[..., 0], predicted[..., 1])


def test_labelled_intermediate_nodes_reindex_by_name_and_replay_fitted_component_order():
    from quivers.formulas import families

    frame = pd.DataFrame(
        {
            "A": [0.2, 0.3, 0.25],
            "B": [0.3, 0.2, 0.25],
            "C": [0.5, 0.5, 0.5],
            "x": [0.25, 0.5, 0.75],
        }
    )
    family = families["logistic_normal"].with_(
        response=families["logistic_normal"].response.with_(reference="B")
    )
    nonlinear = {
        "n": "~1",
        "p": "~1",
        "t": NonlinearExpression(expression={"C": "n*x", "A": "p+x"}),
    }
    result = fit(
        "cbind(A,B,C)~0",
        data=frame,
        family=family,
        nonlinear=nonlinear,
        distributional={
            "mu": NonlinearExpression(expression={"C": "t*t", "A": "exp(t)"})
        },
        fixed_parameters={"sigma": 0.5},
        method="svi",
        num_steps=1,
    )
    result = _recorded(
        result,
        {
            "nlpar_n_intercept": torch.tensor(0.2),
            "nlpar_p_intercept": torch.tensor(-0.1),
            "rescor": torch.eye(2),
        },
    )
    fresh = pd.DataFrame({"x": [0.125, 0.5]})
    x = torch.tensor(fresh.x.to_numpy())
    expected = torch.stack([torch.exp(-0.1 + x), (0.2 * x).square()], -1)
    torch.testing.assert_close(
        result.posterior_linpred(new_data=fresh), expected.expand(2, 3, 2, 2)
    )
    intermediate = torch.stack([0.2 * x, -0.1 + x], -1)
    torch.testing.assert_close(
        result.posterior_linpred(parameter="t", new_data=fresh),
        intermediate.expand(2, 3, 2, 2),
    )
    assert result.posterior_predict(new_data=fresh).shape == (2, 3, 2, 3)


@pytest.mark.parametrize(
    "components,message",
    [
        ({"A": "a*x", "B": "a*x"}, "component labels"),
        ({"B": "a*x", "C": "a*x", "D": "a*x"}, "component labels"),
    ],
)
def test_nonlinear_component_contract_refuses_missing_reference_and_extra_roots(
    components, message
):
    frame = pd.DataFrame(
        {"A": [0.2, 0.3], "B": [0.3, 0.2], "C": [0.5, 0.5], "x": [0.25, 0.5]}
    )
    with pytest.raises(ValueError, match=message):
        formula_to_qvr(
            "cbind(A,B,C)~0",
            data=frame,
            family="dirichlet",
            nonlinear={"a": "~1"},
            distributional={"mu": NonlinearExpression(expression=components)},
        )


def test_an_explicit_mu_root_does_not_silently_discard_the_response_design():
    with pytest.raises(ValueError, match="requires.*'~0'"):
        formula_to_qvr(
            "y~x",
            data=_frame(),
            nonlinear={"a": "~1"},
            distributional={"mu": NonlinearExpression(expression="a*x")},
        )


def test_nonlinear_distributional_only_graph_and_component_scale_use_one_family_link():
    from quivers.formulas import families

    frame = pd.DataFrame(
        {
            "A": [0.2, 0.3, 0.25],
            "B": [0.3, 0.2, 0.25],
            "C": [0.5, 0.5, 0.5],
            "x": [0.25, 0.5, 0.75],
        }
    )
    family = families["logistic_normal"].with_(
        response=families["logistic_normal"].response.with_(reference="B")
    )
    result = fit(
        "cbind(A,B,C)~0",
        data=frame,
        family=family,
        nonlinear={"s": "~1"},
        distributional={
            "mu": NonlinearExpression(expression={"C": "x", "A": "-x"}),
            "sigma": NonlinearExpression(expression={"C": "s-x", "A": "s+x"}),
        },
        method="svi",
        num_steps=1,
    )
    result = _recorded(
        result, {"nlpar_s_intercept": torch.tensor(-1.0), "rescor": torch.eye(2)}
    )
    x = torch.tensor([0.125, 0.5], dtype=torch.float64)
    expected = torch.stack([(-1 + x).exp(), (-1 - x).exp()], -1)
    torch.testing.assert_close(
        result.posterior_linpred(
            parameter="sigma", transform=True, new_data=pd.DataFrame({"x": x.numpy()})
        ),
        expected.expand(2, 3, 2, 2),
    )
    plain = fit(
        "y~x",
        data=_frame(),
        distributional={"sigma": NonlinearExpression(expression="-1+x*x")},
        method="svi",
        num_steps=1,
    )
    sigma = plain.posterior_linpred(parameter="sigma", transform=True, num_samples=1)
    torch.testing.assert_close(
        sigma,
        torch.exp(-1 + torch.tensor(_frame().x.to_numpy()).square()).expand(1, 1, 4),
    )


@pytest.mark.parametrize("expression", ["unknown_function(a)", "a+true"])
def test_nonlinear_expressions_use_the_existing_qiec_function_and_type_checker(
    expression,
):
    from quivers.dsl.qiec_lowering import QiecDiagnosticError

    with pytest.raises(QiecDiagnosticError):
        formula_to_qvr("y~" + expression, data=_frame(), nonlinear={"a": "~1"})


def test_shared_group_ids_keep_joint_nonlinear_parameter_donors():
    result = fit(
        "y~a*exp(b*x)",
        data=_frame(),
        nonlinear={"a": "~1+(1|joint|g)", "b": "~1+(1|joint|g)"},
        fixed_parameters={"sigma": 0.5},
        method="svi",
        num_steps=1,
    )
    factor = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    z = torch.tensor([[1.0, -0.5], [-0.3, 0.8]])
    scales = torch.tensor([0.3, 0.4])
    effects = z @ factor.T * scales
    result = _recorded(
        result,
        {
            "cor_shared_joint": factor,
            "nlpar_a_intercept": torch.tensor(1.0),
            "nlpar_b_intercept": torch.tensor(0.2),
            "nlpar_a_sigma_g_Intercept": scales[0],
            "nlpar_b_sigma_g_Intercept": scales[1],
            "nlpar_a_z_g_Intercept": z[:, 0],
            "nlpar_b_z_g_Intercept": z[:, 1],
        },
    )
    assert len(result.prediction_plan.group_layouts) == 1
    fresh = pd.DataFrame({"g": ["b", "a", "b"], "x": [0.25, 0.5, 0.75]})
    idx = torch.tensor([1, 0, 1])
    expected = (1 + effects[idx, 0]) * torch.exp(
        (0.2 + effects[idx, 1]) * torch.tensor(fresh.x.to_numpy())
    )
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh), expected.expand(2, 3, 3)
    )
    new = pd.DataFrame({"g": ["fresh", "fresh"], "x": [0.5, 0.5]})
    observed = result.posterior_epred(
        new_data=new, allow_new_levels=True, sample_new_levels="uncertainty", seed=6
    )
    candidates = (1 + effects[:, 0]) * torch.exp((0.2 + effects[:, 1]) * 0.5)
    assert bool(
        ((observed - candidates[0]).abs() < 1e-6)
        .logical_or((observed - candidates[1]).abs() < 1e-6)
        .all()
    )
    torch.testing.assert_close(observed[..., 0], observed[..., 1])


def test_nonlinear_leaf_stateful_transforms_and_factor_coding_reuse_fitted_designs():
    frame = _frame().assign(c=["a", "b", "a", "b"])
    result = fit(
        "y~a*exp(b*x)",
        data=frame,
        nonlinear={"a": "~poly(z,2)+c", "b": "~1"},
        fixed_parameters={"sigma": 0.5},
        method="svi",
        num_steps=1,
    )
    fresh = pd.DataFrame({"x": [0.25, 0.5], "z": [10.0, 20.0], "c": ["b", "a"]})
    context = prepare_prediction(result, fresh, None)
    fitted = result.prediction_plan.designs["a"].common.evaluate_new_data(fresh)
    matrix = torch.as_tensor(fitted.design_matrix, dtype=torch.float32)
    columns = [
        column
        for column in result.predictor_graph.leaves["a"].fixed_columns
        if not column.is_intercept
    ]
    for position, column in enumerate(columns, start=1):
        torch.testing.assert_close(
            context.observations[f"nlpar_a_{column.qvr_name}"], matrix[:, position]
        )
    assert result.posterior_epred(new_data=fresh, num_samples=2).shape == (1, 2, 2)
    with pytest.raises(ValueError, match="unknown levels"):
        result.posterior_epred(new_data=fresh.assign(c="unfitted"), num_samples=1)


@pytest.mark.parametrize(
    "distributional",
    [
        {"bogus": "~ x"},
        {
            "sigma": NonlinearExpression(expression="exp(a * x)"),
            "a": "~ 1",
            "b": "~ 1",
        },
    ],
    ids=["no nonlinear formula", "not read by the nonlinear formula"],
)
def test_a_formula_no_nonlinear_formula_reads_is_an_unknown_parameter(
    distributional,
):
    with pytest.raises(ValueError, match=r"unknown distributional parameters \["):
        formula_to_qvr("y ~ x", data=_frame(), distributional=distributional)


def test_a_formula_a_nonlinear_formula_reads_is_a_nonlinear_parameter():
    distributional = {
        "sigma": NonlinearExpression(expression="exp(a * x)"),
        "a": "~ 1",
    }
    source = formula_to_qvr("y ~ x", data=_frame(), distributional=distributional)
    assert "nlpar_a_intercept" in source
