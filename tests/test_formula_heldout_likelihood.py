"""Fresh recorded outcomes share fitted encodings and raw native scores."""

import math

import pandas as pd
import pytest
import torch

from quivers.formulas import fit
from tests.test_formula_new_data import _fit, _recorded


def test_heldout_normal_scores_preserve_draws_rows_weights_and_fit_state():
    fitted = _recorded(
        _fit("y | weights(w) ~ x", fixed_parameters={"sigma": 0.7}),
        {"intercept": torch.tensor(0.3), "beta_x": torch.tensor(-0.2)},
    )
    before = {key: value.clone() for key, value in fitted.observations.items()}
    original = fitted.log_likelihood()["y"]
    fresh = pd.DataFrame({"x": [10.0, 20.0], "y": [-1.0, -4.0]})
    expected = (
        torch.distributions.Normal(torch.tensor([-1.7, -3.7]), 0.7)
        .log_prob(torch.tensor([-1.0, -4.0]))
        .expand(2, 3, 2)
    )
    torch.testing.assert_close(fitted.log_likelihood(new_data=fresh)["y"], expected)
    # Fresh fitting weights never change raw scores or become required inputs.
    weighted = fresh.assign(w=[0.0, 1000.0])
    torch.testing.assert_close(fitted.log_likelihood(new_data=weighted)["y"], expected)
    torch.testing.assert_close(fitted.log_likelihood()["y"], original)
    for key, value in before.items():
        torch.testing.assert_close(fitted.observations[key], value)


def test_fitted_response_transform_is_reused_before_scoring():
    training = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]})
    fitted = _recorded(
        fit(
            "log(y) ~ x",
            data=training,
            method="svi",
            num_steps=1,
            fixed_parameters={"sigma": 1.0},
        ),
        {"intercept": torch.tensor(0.0), "beta_x": torch.tensor(0.0)},
    )
    fresh = pd.DataFrame({"y": [math.e, math.e**2], "x": [0.0, 1.0]})
    actual = next(iter(fitted.log_likelihood(new_data=fresh).values()))
    expected = torch.distributions.Normal(0.0, 1.0).log_prob(torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(actual, expected.expand(2, 3, 2))


def test_categorical_outcomes_use_training_level_order_with_absent_levels():
    training = pd.DataFrame({"y": ["a", "b", "c", "a"], "x": [0.0, 1.0, 2.0, 3.0]})
    fitted = _recorded(
        fit("y ~ 1", data=training, family="categorical", method="svi", num_steps=1),
        {"intercept": torch.tensor([0.4, -0.3])},
    )
    fresh = pd.DataFrame({"y": ["c", "a"]})
    expected = torch.tensor([0.0, 0.4, -0.3]).log_softmax(-1)[torch.tensor([2, 0])]
    torch.testing.assert_close(
        fitted.log_likelihood(new_data=fresh)["y"], expected.expand(2, 3, 2)
    )
    with pytest.raises((ValueError, KeyError)):
        fitted.log_likelihood(new_data=pd.DataFrame({"y": ["unknown"]}))


@pytest.mark.parametrize("family", ["multinomial", "dirichlet_multinomial"])
def test_vector_counts_infer_heldout_totals_and_validate_explicit_totals(family):
    training = pd.DataFrame({"a": [1, 2, 0], "b": [2, 0, 1], "c": [0, 1, 3]})
    fixed = {"phi": 2.0} if family == "dirichlet_multinomial" else {}
    fitted = _recorded(
        fit(
            "cbind(a,b,c) ~ 1",
            data=training,
            family=family,
            method="svi",
            num_steps=1,
            fixed_parameters=fixed,
        ),
        {"intercept": torch.tensor([0.4, -0.3])},
    )
    fresh = pd.DataFrame({"a": [0, 2], "b": [0, 3], "c": [0, 1]})
    y = torch.tensor([[0.0, 0.0, 0.0], [2.0, 3.0, 1.0]], dtype=torch.float64)
    probability = torch.tensor([0.0, 0.4, -0.3], dtype=torch.float64).softmax(-1)
    expected = torch.lgamma(y.sum(-1) + 1) - torch.lgamma(y + 1).sum(-1)
    if family == "multinomial":
        expected += (y * probability.log()).sum(-1)
    else:
        alpha = probability * 2
        expected += torch.lgamma(torch.tensor(2.0)) - torch.lgamma(y.sum(-1) + 2)
        expected += (torch.lgamma(y + alpha) - torch.lgamma(alpha)).sum(-1)
    actual = next(iter(fitted.log_likelihood(new_data=fresh).values()))
    torch.testing.assert_close(actual, expected.expand(2, 3, 2))
    with pytest.raises(ValueError, match="totals disagree"):
        fitted.log_likelihood(new_data=fresh.assign(n=[0, 8]), trials="n")


def test_subset_and_censored_events_rebind_without_training_masks():
    training = pd.DataFrame(
        {"y": [0.0, 1.0, 2.0], "keep": [True, True, True], "status": [0, 0, 0]}
    )
    fitted = _recorded(
        fit(
            "y | subset(keep) + cens(status) ~ 1",
            data=training,
            method="svi",
            num_steps=1,
            fixed_parameters={"sigma": 1.0},
        ),
        {"intercept": torch.tensor(0.0)},
    )
    fresh = pd.DataFrame(
        {
            "y": [0.5, float("nan"), -0.3],
            "keep": [True, False, True],
            "status": [1, 99, -1],
        }
    )
    expected = torch.stack(
        [
            torch.special.log_ndtr(torch.tensor(-0.5, dtype=torch.float64)),
            torch.special.log_ndtr(torch.tensor(-0.3, dtype=torch.float64)),
        ]
    )
    torch.testing.assert_close(
        fitted.log_likelihood(new_data=fresh)["y"], expected.expand(2, 3, 2)
    )


def test_trial_overrides_score_actual_training_and_fresh_counts():
    training = pd.DataFrame({"y": [1, 2, 0], "n": [2, 3, 1]})
    fitted = _recorded(
        fit(
            "y ~ 1",
            data=training,
            family="binomial",
            binomial_trials="n",
            method="svi",
            num_steps=1,
        ),
        {"intercept": torch.tensor(0.4)},
    )
    probability = torch.tensor(0.4, dtype=torch.float64).sigmoid()
    for frame, arguments in [
        (training, {"trials": 4}),
        (pd.DataFrame({"y": [2, 0], "n": [4, 0]}), {"new_data": True}),
    ]:
        counts = (
            torch.full((len(frame),), 4.0, dtype=torch.float64)
            if "trials" in arguments
            else torch.tensor(frame.n.to_numpy(), dtype=torch.float64)
        )
        expected = torch.distributions.Binomial(counts, probs=probability).log_prob(
            torch.tensor(frame.y.to_numpy(), dtype=torch.float64)
        )
        keywords = {"trials": 4} if "trials" in arguments else {"new_data": frame}
        actual = fitted.log_likelihood(**keywords)["y"]
        torch.testing.assert_close(actual, expected.expand(2, 3, len(frame)))
    with pytest.raises(ValueError, match="between zero"):
        fitted.log_likelihood(trials=1)


def test_trial_modifier_override_does_not_require_the_original_trial_column():
    fitted = _recorded(
        fit(
            "y | trials(n) ~ 1",
            data=pd.DataFrame({"y": [1, 2], "n": [2, 3]}),
            family="binomial",
            method="svi",
            num_steps=1,
        ),
        {"intercept": torch.tensor(0.4)},
    )
    fresh = pd.DataFrame({"y": [1, 3]})
    expected = torch.distributions.Binomial(
        4.0, probs=torch.tensor(0.4, dtype=torch.float64).sigmoid()
    ).log_prob(torch.tensor([1.0, 3.0], dtype=torch.float64))
    torch.testing.assert_close(
        fitted.log_likelihood(new_data=fresh, trials=4)["y"], expected.expand(2, 3, 2)
    )


def test_grouped_heldout_prediction_uses_existing_selection_and_new_level_policy():
    fitted = _recorded(
        _fit("y ~ x + (1 | g)", fixed_parameters={"sigma": 0.7}),
        {"intercept": torch.tensor(0.3), "beta_x": torch.tensor(-0.2)},
    )
    fresh = pd.DataFrame({"x": [0.0, 2.0], "y": [0.4, -0.3], "g": ["new", "new"]})
    with pytest.raises(ValueError, match="new|unknown|unseen"):
        fitted.log_likelihood(new_data=fresh)
    mean = fitted.posterior_linpred(new_data=fresh, allow_new_levels=True, seed=41)
    expected = torch.distributions.Normal(mean, 0.7).log_prob(
        torch.tensor(fresh.y.to_numpy(), dtype=mean.dtype)
    )
    torch.testing.assert_close(
        fitted.log_likelihood(new_data=fresh, allow_new_levels=True, seed=41)["y"],
        expected,
    )
    population = torch.tensor([0.3, -0.1])
    expected_population = torch.distributions.Normal(population, 0.7).log_prob(
        torch.tensor(fresh.y.to_numpy(), dtype=population.dtype)
    )
    torch.testing.assert_close(
        fitted.log_likelihood(new_data=fresh.drop(columns="g"), re_formula="~0")["y"],
        expected_population.expand(2, 3, 2),
    )


@pytest.mark.parametrize("values", [[float("nan")], [float("inf")], [1e100]])
def test_invalid_recorded_outcomes_are_rejected_before_posterior_draws(
    values, monkeypatch
):
    fitted = _fit(fixed_parameters={"sigma": 1.0})

    def forbidden(*args, **kwargs):
        raise AssertionError("posterior draws requested before validation")

    monkeypatch.setattr(type(fitted), "draws", forbidden)
    with pytest.raises(ValueError, match="finite"):
        fitted.log_likelihood(new_data=pd.DataFrame({"y": values, "x": [0.0]}))
