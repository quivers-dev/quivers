"""Formula hypotheses select authored roles and share vectorized group geometry."""

from unittest.mock import patch

import pandas as pd
import pytest
import torch

from quivers.diagnostics import ContrastExpression, hypothesis_summary
from quivers.formulas import PriorSpec, R2D2, RegularizedHorseshoe, families, fit
from quivers.formulas._posterior import posterior_draws
from quivers.inference.trace import trace


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "y": [0.2, 0.4, -0.1, 0.5, -0.2, 0.3],
            "x": [-1.0, 0.0, 1.0, -0.3, 0.2, 0.7],
            "z": [0.5, -0.3, 0.2, 0.6, -0.7, 0.1],
            "g": ["a", "a", "b", "b", "c", "c"],
        }
    )


def test_population_hypothesis_uses_one_draw_stream_and_exact_authored_labels(frame):
    frame["condition"] = pd.Categorical(["red", "blue", "red", "blue", "red", "blue"])
    result = fit("y ~ x + condition", data=frame, method="svi", num_steps=1)
    draws = result.draws(num_samples=8, seed=14)
    fixed = result._fixed_effects(draws, result.formula, "")
    label = next(
        column.name
        for column in result.formula.fixed_columns
        if not column.is_intercept and column.name != "x"
    )
    expression = ContrastExpression(expression="exp(a)-x", bindings={"a": label})
    expected = hypothesis_summary(torch.exp(fixed[label]) - fixed["x"])
    state = torch.random.get_rng_state()
    with patch(
        "quivers.formulas._posterior.posterior_draws", wraps=posterior_draws
    ) as collect:
        actual = result.hypothesis(expression, num_samples=8, seed=14)
    assert collect.call_count == 1
    torch.testing.assert_close(actual.estimate, expected.estimate)
    torch.testing.assert_close(actual.quantiles, expected.quantiles)
    assert actual.kind == "variational" and actual.sample_shape == (1, 8)
    assert torch.equal(state, torch.random.get_rng_state())


@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
@pytest.mark.parametrize("correlated", [False, True])
def test_group_hypotheses_match_native_bindings_without_replay(
    frame, coordinates, correlated
):
    term = "(1+x|g)" if correlated else "(1+x||g)"
    result = fit(
        "y~x+" + term, data=frame, reparameterize=coordinates, method="svi", num_steps=1
    )
    draws = result.draws(num_samples=5, seed=9)
    direct = []
    for i in range(5):
        recorded = trace(
            result.program,
            torch.zeros(1, 1),
            result.observations | {name: value[0, i] for name, value in draws.items()},
        )
        direct.append(
            recorded.sites["beta_g_x"].value.reshape(3)
            - recorded.sites["alpha_g"].value.reshape(3)
        )
    expected = hypothesis_summary(torch.stack(direct).unsqueeze(0))
    with patch(
        "quivers.formulas._posterior.replay",
        side_effect=AssertionError("group extraction must not replay"),
    ):
        actual = result.hypothesis(
            "x-Intercept", scope="ranef", group="g", num_samples=5, seed=9
        )
        combined = result.hypothesis(
            "x-Intercept", scope="coef", group="g", num_samples=5, seed=9
        )
    torch.testing.assert_close(actual.estimate, expected.estimate.double())
    population = result._fixed_effects(draws, result.formula, "")
    torch.testing.assert_close(
        combined.estimate,
        (
            torch.stack(direct).unsqueeze(0)
            + (population["x"] - population["Intercept"]).unsqueeze(-1)
        )
        .double()
        .mean((0, 1)),
    )
    assert actual.dimension_names == ("group:g",)
    assert actual.coordinate_labels == (("a", "b", "c"),)


@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
def test_shared_distributional_block_and_nonlinear_predictors_retain_namespaces(
    frame, coordinates
):
    result = fit(
        "y~x+(1|joint|g)",
        data=frame,
        distributional={"sigma": "~z+(1|joint|g)"},
        reparameterize=coordinates,
        method="svi",
        num_steps=1,
    )
    draws = result.draws(num_samples=4, seed=12)
    expected = result._fixed_effects(
        draws, result.distributional["sigma"], "dpar_sigma_"
    )["z"]
    actual = result.hypothesis("z", parameter="sigma", num_samples=4, seed=12)
    torch.testing.assert_close(actual.estimate, expected.double().mean())
    block = result.hypothesis("joint[0,1]", class_="cor", num_samples=4, seed=12)
    factor = draws[
        next(
            layout.correlation_site
            for layout in result.prediction_plan.group_layouts
            if layout.correlation_site is not None
        )
    ]
    torch.testing.assert_close(
        block.estimate, (factor @ factor.transpose(-1, -2))[..., 0, 1].double().mean()
    )
    result = fit(
        "y~a*exp(b)",
        nonlinear={"a": "~x+(1|g)", "b": "~z"},
        data=frame,
        method="svi",
        num_steps=1,
    )
    actual = result.hypothesis("x-Intercept", parameter="a", num_samples=4, seed=12)
    fixed = result.fixed_effects(parameter="a", num_samples=4, seed=12)
    torch.testing.assert_close(
        actual.estimate, (fixed["x"] - fixed["Intercept"]).double().mean()
    )


@pytest.mark.parametrize("family", ["categorical", "dirichlet", "logistic_normal"])
def test_component_hypotheses_preserve_free_coordinates_and_labels(frame, family):
    if family == "categorical":
        frame["y"] = pd.Categorical(
            ["blue", "red", "green", "blue", "red", "green"],
            categories=["green", "red", "blue"],
        )
        formula = "y~x+(1|g)"
    else:
        frame["left"] = [0.2, 0.3, 0.4, 0.2, 0.1, 0.3]
        frame["middle"] = [0.3, 0.2, 0.2, 0.4, 0.6, 0.2]
        frame["right"] = 1 - frame.left - frame.middle
        formula = "cbind(left,middle,right)~x+(1|g)"
    result = fit(formula, family=family, data=frame, method="svi", num_steps=1)
    actual = result.hypothesis("x-Intercept", num_samples=5, seed=7)
    fixed = result.fixed_effects(num_samples=5, seed=7)
    torch.testing.assert_close(
        actual.estimate, (fixed["x"] - fixed["Intercept"]).double().mean((0, 1))
    )
    assert actual.coordinate_labels == (result.coefficient_labels,)
    grouped = result.hypothesis(
        "Intercept", scope="coef", group="g", num_samples=5, seed=7
    )
    assert grouped.coordinate_labels == (("a", "b", "c"), result.coefficient_labels)


@pytest.mark.parametrize("hierarchy", [RegularizedHorseshoe(), R2D2()])
def test_hierarchical_prior_contrasts_preserve_joint_dependence_and_bounds(
    frame, hierarchy
):
    result = fit(
        "y~x+z",
        data=frame,
        method="svi",
        num_steps=1,
        priors=PriorSpec(distribution=hierarchy, lower=-0.5, upper=0.8),
        sample_prior="yes",
        num_prior_samples=9,
    )
    posterior = result.draws(num_samples=6, seed=8)
    prior = result.prior_view().draws()
    expected = hypothesis_summary(
        posterior["beta_x"] - posterior["beta_z"],
        prior=prior["beta_x"] - prior["beta_z"],
    )
    with patch(
        "quivers.formulas._posterior.posterior_draws", wraps=posterior_draws
    ) as collect:
        actual = result.hypothesis("x-z", num_samples=6, seed=8, prior=True)
    assert collect.call_count == 2
    torch.testing.assert_close(
        actual.above.prior_probability, expected.above.prior_probability
    )
    torch.testing.assert_close(
        actual.below.log_bayes_factor, expected.below.log_bayes_factor, equal_nan=True
    )
    assert actual.prior_sample_shape == (1, 9)


def test_selector_errors_are_explicit_and_exact_latent_scope_remains_available(frame):
    result = fit("y~x+(1|g)", data=frame, method="svi", num_steps=1)
    raw = result.hypothesis("beta_x", class_=None, num_samples=3)
    assert raw.sample_shape == (1, 3)
    for kwargs in (
        {"class_": "missing"},
        {"scope": "coef"},
        {"scope": "ranef", "group": "missing"},
        {"num_prior_samples": 3},
    ):
        with pytest.raises(ValueError):
            result.hypothesis("x", num_samples=3, **kwargs)
    with pytest.raises(ValueError, match="retained parameter-role"):
        result.with_(prediction_plan=None).hypothesis("x", num_samples=3)


@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
def test_mixed_scalar_component_shared_block_matches_every_native_binding(coordinates):
    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 3] * 2,
            "x": [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
            "g": ["a", "b"] * 4,
        }
    )
    result = fit(
        "y~1+(1+cs(x)|joint|g)",
        data=frame,
        family="acat",
        distributional={"disc": "~1+(1|joint|g)"},
        reparameterize=coordinates,
        priors=PriorSpec(distribution=R2D2(), class_="sd", lower=0.1, upper=0.9),
        method="svi",
        num_steps=1,
    )
    draws = result.draws(num_samples=3, seed=17)
    for parameter, prefix, predictor in result._predictor_items():
        actual = result._random_effects(draws, (1, 3), predictor, prefix)
        for layout in result.prediction_plan.group_layouts:
            for member in layout.members:
                if member.prefix != prefix:
                    continue
                expected = []
                for index in range(3):
                    recorded = trace(
                        result.program,
                        torch.zeros(1, 1),
                        result.observations
                        | {name: value[0, index] for name, value in draws.items()},
                    )
                    value = recorded.sites[member.coefficient].value
                    expected.append(
                        value.reshape(member.width, 2).T
                        if member.axis is not None
                        else value.reshape(2)
                    )
                torch.testing.assert_close(
                    actual["g"][member.slope], torch.stack(expected).unsqueeze(0)
                )
    label = next(
        term.slope
        for term in result.formula.random_terms
        if term.column is not None and term.column.category_specific
    )
    summary = result.hypothesis(
        ContrastExpression(expression="effect", bindings={"effect": label}),
        scope="ranef",
        group="g",
        num_samples=3,
        seed=17,
    )
    assert summary.coordinate_labels == (("a", "b"), ((0, 1), (1, 2), (2, 3)))


@pytest.mark.parametrize("coordinates", ["centered", "noncentered"])
@pytest.mark.parametrize("policy", ["flexible", "sum_to_zero", "equidistant"])
def test_grouped_threshold_contrast_preserves_location_adjustment_once(
    coordinates, policy
):
    frame = pd.DataFrame(
        {
            "y": [0, 1, 2, 3] * 2,
            "x": [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
            "g": ["a", "b"] * 4,
        }
    )
    family = families["cumulative"].with_thresholds(policy)
    result = fit(
        "y~x+(1+cs(x)|g)",
        data=frame,
        family=family,
        thresholds_by="g",
        reparameterize=coordinates,
        method="svi",
        num_steps=1,
    )
    draws = result.draws(num_samples=3, seed=15)
    actual = result.hypothesis(
        "Intercept", scope="ranef", group="g", num_samples=3, seed=15
    )
    expected = []
    for index in range(3):
        recorded = trace(
            result.program,
            torch.zeros(1, 1),
            result.observations
            | {name: value[0, index] for name, value in draws.items()},
        )
        expected.append(recorded.sites["alpha_g"].value.reshape(2))
    torch.testing.assert_close(actual.estimate, torch.stack(expected).double().mean(0))
    assert actual.coordinate_labels == (("a", "b"),)
