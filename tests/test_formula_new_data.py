"""Fresh formula predictions preserve fitted encodings and native laws."""

import math
import pickle

from formulae.config import config
import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import fit, formula_from_data
from quivers.formulas._prediction import prepare_prediction
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.inference.trace import trace


def _recorded(result, values=None, chains=2, count=3):
    draws = result.draws(num_samples=1)
    # Derived coefficient sites (ordered or centered intercepts) are not latent.
    latent = LatentRegistry.from_model(
        result.program, observed_names=set(result.observations)
    ).names
    samples = {
        name: value[0, 0].expand(chains, count, *value.shape[2:]).clone()
        for name, value in draws.items()
        if name in latent
    }
    for name, value in (values or {}).items():
        samples[name] = value.expand(chains, count, *value.shape).clone()
    return result.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(chains, count),
            acceptance_rates=torch.ones(chains),
            divergence_counts=torch.zeros(chains, dtype=torch.long),
            num_warmup=0,
            num_samples=count,
        )
    )


def _fit(formula="y ~ x", **kwargs):
    frame = pd.DataFrame(
        {
            "y": [0.1, 0.8, -0.2, 0.4],
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [0.2, -0.5, 0.0, 0.7],
            "g": ["a", "a", "b", "b"],
            "w": [0.0, 0.5, 1.0, 2.0],
            "e": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return fit(formula, data=frame, method="svi", num_steps=1, **kwargs)


def test_new_rows_compile_once_preserve_training_priors_and_exact_gradients(
    monkeypatch,
):
    result = _recorded(
        _fit(fixed_parameters={"sigma": 0.7}),
        {
            "intercept": torch.tensor(0.3),
            "beta_x": torch.tensor(-0.2),
        },
    )
    original = result.qvr_source
    observations = {name: value.clone() for name, value in result.observations.items()}
    from quivers.formulas._prediction import Compiler

    compile_ = Compiler.compile
    calls = []

    def counted(self):
        calls.append(self)
        return compile_(self)

    monkeypatch.setattr(Compiler, "compile", counted)
    frame = pd.DataFrame({"x": [10.0, 20.0]})
    expected = torch.tensor([-1.7, -3.7]).expand(2, 3, 2)
    torch.testing.assert_close(result.posterior_linpred(new_data=frame), expected)
    assert len(calls) == 1
    context = prepare_prediction(result, frame, None)
    theta = torch.tensor([0.3, -0.2], requires_grad=True)
    recorded = trace(
        context.program,
        torch.zeros(1, 1, dtype=torch.long),
        dict(context.observations) | {"intercept": theta[0], "beta_x": theta[1]},
    )
    mean = theta[0] + theta[1] * torch.tensor([10.0, 20.0])
    oracle = (
        torch.distributions.Normal(mean, 0.7).log_prob(context.observations["y"]).sum()
        + torch.distributions.Normal(0.0, 5.0).log_prob(theta[0])
        + torch.distributions.Normal(0.0, 5.0 / math.sqrt(1.5)).log_prob(theta[1])
    )
    actual = recorded.log_joint.sum()
    torch.testing.assert_close(actual, oracle)
    torch.testing.assert_close(
        torch.autograd.grad(actual, theta, retain_graph=True)[0],
        torch.autograd.grad(oracle, theta)[0],
    )
    assert result.qvr_source == original
    assert all(
        torch.equal(result.observations[name], value)
        for name, value in observations.items()
    )


def test_stateful_transforms_and_contrasts_retain_training_state_and_column_order():
    frame = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "c": ["a", "b", "c", "a"],
        }
    )
    result = fit(
        "y ~ center(x) + scale(x) + poly(x, 2) + c",
        data=frame,
        fixed_parameters={"sigma": 1.0},
        method="svi",
        num_steps=1,
    )
    coefficients = [0.3, 0.2, -0.1, 0.4, 0.5, 0.6, 0.7]
    values = {
        "intercept" if col.is_intercept else f"beta_{col.qvr_name}": torch.tensor(value)
        for col, value in zip(result.formula.fixed_columns, coefficients, strict=True)
    }
    result = _recorded(result, values)
    fresh = pd.DataFrame({"x": [10.0, 12.0], "c": ["c", "a"]})
    matrix = torch.tensor(
        [
            # R scale() divides by the n - 1 deviation, sqrt(5 / 3) for x.
            [1.0, 8.5, 8.5 / math.sqrt(5 / 3), 8.5 / math.sqrt(5), 35.5, 0.0, 1.0],
            [1.0, 10.5, 10.5 / math.sqrt(5 / 3), 10.5 / math.sqrt(5), 54.5, 0.0, 0.0],
        ]
    )
    expected = (matrix @ torch.tensor(coefficients)).expand(2, 3, 2)
    before = tuple(col.data.copy() for col in result.formula.fixed_columns)
    fitted_matrix = result.prediction_plan.designs["mu"].common.design_matrix.copy()
    a = result.posterior_linpred(new_data=fresh)
    result.posterior_linpred(new_data=pd.DataFrame({"x": [-20.0], "c": ["b"]}))
    torch.testing.assert_close(a, expected)
    torch.testing.assert_close(result.posterior_linpred(new_data=fresh), a)
    assert all(
        np.array_equal(col.data, saved)
        for col, saved in zip(result.formula.fixed_columns, before, strict=True)
    )
    np.testing.assert_array_equal(
        result.prediction_plan.designs["mu"].common.design_matrix, fitted_matrix
    )
    assert pickle.loads(pickle.dumps(result.formula)) == result.formula
    assert result.formula == formula_from_data(result.formula.formula, frame)
    with pytest.raises(
        TypeError, match="qvr-formula-prediction/v1.*cannot be serialized"
    ):
        pickle.dumps(result.prediction_plan)


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_known_groups_transformed_slopes_and_shared_auxiliary_identity(
    parameterization,
):
    result = _fit(
        "y ~ x + (1 | shared | g)",
        distributional={"sigma": "~ z + (1 | shared | g)"},
        reparameterize=parameterization,
    )
    result = _recorded(result)
    frame = pd.DataFrame(
        {"x": [5.0, -2.0, 7.0], "z": [-1.0, 2.0, 0.0], "g": ["b", "a", "b"]}
    )
    draws = result.draws()
    location = result.random_effects()["g"]["Intercept"]
    scale = result.random_effects(parameter="sigma")["g"]["Intercept"]
    groups = torch.tensor([1, 0, 1])
    expected = (
        draws["intercept"][..., None]
        + draws["beta_x"][..., None] * torch.tensor([5.0, -2.0, 7.0])
        + location[..., groups]
    )
    aux = (
        draws["dpar_sigma_intercept"][..., None]
        + draws["dpar_sigma_beta_z"][..., None] * torch.tensor([-1.0, 2.0, 0.0])
        + scale[..., groups]
    )
    torch.testing.assert_close(result.posterior_linpred(new_data=frame), expected)
    torch.testing.assert_close(
        result.posterior_linpred(new_data=frame, parameter="sigma"), aux
    )
    torch.testing.assert_close(
        result.posterior_linpred(new_data=frame, parameter="sigma", transform=True),
        aux.exp(),
    )


def test_random_polynomial_slopes_use_fitted_basis_with_two_groups():
    frame = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "g": ["a", "a", "b", "b"],
        }
    )
    result = _recorded(
        fit("y ~ 1 + (0 + poly(x,2) || g)", data=frame, method="svi", num_steps=1)
    )
    fresh = pd.DataFrame({"x": [10.0, 12.0], "g": ["b", "a"]})
    effects = result.random_effects()["g"]
    columns = torch.tensor([[8.5 / math.sqrt(5), 35.5], [10.5 / math.sqrt(5), 54.5]])
    expected = result.draws()["intercept"][..., None].expand(2, 3, 2).clone()
    for index, name in enumerate(effects):
        expected += effects[name][..., torch.tensor([1, 0])] * columns[:, index]
    torch.testing.assert_close(result.posterior_linpred(new_data=fresh), expected)


def test_offsets_exposures_auxiliaries_and_weightless_frames():
    result = _recorded(
        _fit(
            "y ~ x + offset(z) + exposure(e)",
            weights="w",
            distributional={"sigma": "~ z + offset(x) + exposure(e)"},
        ),
        {
            "intercept": torch.tensor(0.3),
            "beta_x": torch.tensor(-0.2),
            "dpar_sigma_intercept": torch.tensor(-0.4),
            "dpar_sigma_beta_z": torch.tensor(0.1),
        },
    )
    frame = pd.DataFrame({"x": [3.0, 4.0], "z": [0.5, -1.0], "e": [2.0, 4.0]})
    expected = (
        0.3
        - 0.2 * torch.tensor([3.0, 4.0])
        + torch.tensor([0.5, -1.0])
        + torch.log(torch.tensor([2.0, 4.0]))
    )
    aux = (
        -0.4
        + 0.1 * torch.tensor([0.5, -1.0])
        + torch.tensor([3.0, 4.0])
        + torch.log(torch.tensor([2.0, 4.0]))
    )
    torch.testing.assert_close(
        result.posterior_epred(new_data=frame), expected.expand(2, 3, 2)
    )
    torch.testing.assert_close(
        result.posterior_linpred(new_data=frame, parameter="sigma"), aux.expand(2, 3, 2)
    )
    context = prepare_prediction(result, frame, None)
    assert context.observations["formula_weights"].eq(1).all()
    assert torch.equal(
        result.posterior_predict(new_data=frame, seed=9),
        result.posterior_predict(new_data=frame.assign(w=[99.0, float("nan")]), seed=9),
    )


@pytest.mark.parametrize("family", ["binomial", "beta_binomial"])
def test_fresh_binomial_trials_are_counts_not_probabilities(family):
    frame = pd.DataFrame({"y": [0, 1, 2, 1], "n": [2, 3, 4, 5]})
    result = _recorded(
        fit(
            "y ~ 1",
            data=frame,
            family=family,
            binomial_trials="n",
            method="svi",
            num_steps=1,
        ),
        {"intercept": torch.tensor(0.0)},
    )
    fresh = pd.DataFrame({"n": [1, 7, 3]})
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh),
        torch.tensor([0.5, 3.5, 1.5], dtype=torch.float64).expand(2, 3, 3),
    )
    sample = result.posterior_predict(new_data=fresh, seed=8)
    assert sample.shape == (2, 3, 3)
    assert (
        (sample >= 0) & (sample <= torch.tensor([1, 7, 3])) & (sample == sample.round())
    ).all()


@pytest.mark.parametrize("family", ["categorical", "cumulative"])
def test_training_category_labels_survive_fewer_new_rows(family):
    frame = pd.DataFrame({"y": [0, 1, 2, 0], "x": [-1.0, 0.0, 1.0, 2.0]})
    result = _recorded(
        fit("y ~ x", data=frame, family=family, method="svi", num_steps=1)
    )
    fresh = pd.DataFrame({"x": [5.0]})
    probabilities = result.posterior_epred(new_data=fresh)
    assert probabilities.shape == (2, 3, 1, 3)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 3, 1))
    samples = result.posterior_predict(new_data=fresh, seed=7)
    assert samples.shape == (2, 3, 1)
    assert ((samples >= 0) & (samples < 3) & (samples == samples.round())).all()
    assert result.category_labels == (0, 1, 2)


@pytest.mark.parametrize(
    "fresh,match",
    [
        (pd.DataFrame({"other": [1.0]}), "missing predictors"),
        (pd.DataFrame({"x": [float("nan")]}), "finite numeric"),
        (pd.DataFrame({"x": [float("inf")]}), "finite numeric"),
        (pd.DataFrame({"x": []}), "nonempty"),
    ],
)
@pytest.mark.parametrize(
    "method", ["posterior_linpred", "posterior_epred", "posterior_predict"]
)
def test_invalid_predictors_fail_before_posterior_sampling(
    monkeypatch, fresh, match, method
):
    result = _fit()

    def no_draws(*args, **kwargs):
        raise AssertionError("invalid new_data must fail before posterior sampling")

    monkeypatch.setattr(type(result), "draws", no_draws)
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match=match):
        getattr(result, method)(new_data=fresh, num_samples=1)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["category", "group"])
def test_unknown_levels_refused_even_if_formulae_global_config_is_silent(kind):
    frame = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0],
            "c": ["a", "b", "a", "b"],
            "g": ["a", "a", "b", "b"],
        }
    )
    result = fit(
        "y ~ c" if kind == "category" else "y ~ (1|g)",
        data=frame,
        method="svi",
        num_steps=1,
    )
    old = config["EVAL_UNSEEN_CATEGORIES"]
    config["EVAL_UNSEEN_CATEGORIES"] = "silent"
    try:
        with pytest.raises(ValueError, match="unknown levels|unknown groups"):
            result.posterior_epred(
                new_data=pd.DataFrame({"c": ["new"], "g": ["new"]}), num_samples=1
            )
        assert config["EVAL_UNSEEN_CATEGORIES"] == "silent"
    finally:
        config["EVAL_UNSEEN_CATEGORIES"] = old


@pytest.mark.parametrize(
    "column,value,kwargs",
    [
        ("e", 0.0, {}),
        ("e", float("inf"), {}),
        ("z", float("nan"), {}),
        ("n", -1.0, {"family": "binomial", "binomial_trials": "n"}),
        ("n", 1.5, {"family": "binomial", "binomial_trials": "n"}),
        ("n", True, {"family": "binomial", "binomial_trials": "n"}),
    ],
)
def test_bad_offsets_exposures_and_trials_fail_before_draws(
    monkeypatch, column, value, kwargs
):
    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 1],
            "n": [2, 3, 4, 5],
            "e": [1.0, 2.0, 3.0, 4.0],
            "z": [0.0, 1.0, 2.0, 3.0],
        }
    )
    result = fit(
        "y ~ offset(z) + exposure(e)", data=frame, method="svi", num_steps=1, **kwargs
    )
    monkeypatch.setattr(
        type(result), "draws", lambda *a, **kw: pytest.fail("premature posterior draw")
    )
    fresh = pd.DataFrame({"n": [3], "e": [2.0], "z": [0.0]}).assign(**{column: value})
    with pytest.raises(ValueError):
        result.posterior_predict(new_data=fresh)


def test_neural_prediction_requires_fresh_input_and_restores_module_mode_and_rng():
    predictor = torch.nn.Sequential(
        torch.nn.Linear(1, 1, bias=False), torch.nn.Dropout(0.5)
    )
    with torch.no_grad():
        predictor[0].weight.fill_(2.0)
    predictor.requires_grad_(False)
    result = _fit(
        "y ~ 1",
        predictor=predictor,
        predictor_data=torch.ones(4, 1),
        fixed_parameters={"sigma": 1.0},
    )
    result = _recorded(result, {"intercept": torch.tensor(0.3)})
    fresh = pd.DataFrame(index=range(2))
    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="explicit predictor_data"):
        result.posterior_epred(new_data=fresh)
    value = result.posterior_epred(
        new_data=fresh, predictor_data=torch.tensor([[3.0], [4.0]])
    )
    torch.testing.assert_close(value, torch.tensor([6.3, 8.3]).expand(2, 3, 2))
    assert predictor.training and predictor[1].training
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="expected"):
        result.posterior_epred(new_data=fresh, predictor_data=torch.ones(3, 1))
    assert predictor.training and torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match="requires new_data"):
        result.posterior_epred(predictor_data=torch.ones(2, 1))


def test_constant_new_rows_have_independent_predictive_noise_and_preserve_rng():
    result = _recorded(
        _fit("y ~ 1", fixed_parameters={"sigma": 0.7}),
        {"intercept": torch.tensor(0.3)},
        count=100,
    )
    fresh = pd.DataFrame(index=range(20))
    before = torch.get_rng_state().clone()
    first = result.posterior_predict(new_data=fresh, seed=83)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(first, result.posterior_predict(new_data=fresh, seed=83))
    assert not torch.equal(first[..., 0], first[..., 1])
    residual = first - 0.3
    assert abs(residual.mean().item()) < 5 * 0.7 / math.sqrt(residual.numel())
    assert abs(residual.square().mean().item() - 0.7**2) < 5 * 0.7**2 * math.sqrt(
        2 / residual.numel()
    )


def test_categorical_calls_interactions_and_input_frame_are_preserved():
    training = pd.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0],
            "x": [0.0, 1.0, 2.0, 3.0],
            "c": ["a", "b", "c", "a"],
        }
    )
    result = _recorded(fit("y ~ x * C(c)", data=training, method="svi", num_steps=1))
    fresh = pd.DataFrame(
        {"x": [10.0, 5.0], "c": pd.Categorical(["c", "a"], categories=["c", "b", "a"])}
    )
    saved = fresh.copy(deep=True)
    draws = result.fixed_effects()
    expected = draws["Intercept"][..., None] + draws["x"][..., None] * torch.tensor(
        [10.0, 5.0]
    )
    expected[..., 0] += draws["C(c)_2"] + 10 * draws["x:C(c)_2"]
    torch.testing.assert_close(result.posterior_linpred(new_data=fresh), expected)
    pd.testing.assert_frame_equal(fresh, saved)
    with pytest.raises(ValueError, match="unknown levels"):
        result.posterior_epred(new_data=pd.DataFrame({"x": [1.0], "c": ["new"]}))


def test_auxiliary_only_group_and_missing_auxiliary_inputs_are_preflighted(monkeypatch):
    result = _recorded(_fit("y ~ x", distributional={"sigma": "~ z + (1|g)"}))
    fresh = pd.DataFrame({"x": [4.0, 1.0], "z": [2.0, -1.0], "g": ["b", "b"]})
    assert result.posterior_epred(new_data=fresh).shape == (2, 3, 2)
    monkeypatch.setattr(
        type(result), "draws", lambda *a, **kw: pytest.fail("premature draw")
    )
    for bad in (
        fresh.drop(columns="z"),
        fresh.drop(columns="g"),
        fresh.assign(g="unknown"),
    ):
        with pytest.raises(
            ValueError, match="missing predictors|unknown levels|unknown groups"
        ):
            result.posterior_linpred(new_data=bad)


def test_grouped_ordinal_thresholds_preserve_training_dimensions():
    training = pd.DataFrame(
        {"y": [0, 1, 2, 0, 1, 2], "g": ["a", "a", "a", "b", "b", "b"]}
    )
    result = _recorded(
        fit(
            "y ~ (1|g)",
            data=training,
            family="cumulative",
            thresholds_by="g",
            method="svi",
            num_steps=1,
        )
    )
    fresh = pd.DataFrame({"g": ["b"]})
    probabilities = result.posterior_epred(new_data=fresh)
    assert probabilities.shape == (2, 3, 1, 3)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(2, 3, 1))
    assert result.category_labels == (0, 1, 2)


def test_mixture_component_axis_is_independent_of_new_row_count():
    result = _recorded(_fit("y ~ x", family="mixture", mixture_components=3))
    fresh = pd.DataFrame({"x": [3.0]})
    context = prepare_prediction(result, fresh, None)
    draws = result.draws()
    latents = {name: value[0, 0] for name, value in draws.items()}
    recorded = trace(
        context.program,
        torch.zeros(1, 1, dtype=torch.long),
        dict(context.observations) | latents,
    )
    from quivers.formulas._posterior import response_distribution

    expected = response_distribution(recorded, "y").mean.reshape(1)
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh), expected.expand(2, 3, 1)
    )
    assert result.posterior_predict(new_data=fresh).shape == (2, 3, 1)
