"""Formula R-squared uses retained response scales and conditional variances."""

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.special import expit

from quivers.formulas import fit
from tests.test_formula_posterior import _recorded


def _ratio(mean, observed=None, variance=None):
    explained = np.var(mean, ddof=1)
    unexplained = (
        np.var(observed - mean, ddof=1) if observed is not None else np.mean(variance)
    )
    return explained / (explained + unexplained)


def test_gaussian_model_and_residual_ratios_match_independent_variances():
    data = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0], "y": [0.2, 1.0, 1.5, -0.1]})
    result = fit(
        "y~x", data=data, fixed_parameters={"sigma": 0.5}, method="svi", num_steps=0
    )
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.3, dtype=torch.double),
            "beta_x": torch.tensor(-0.2, dtype=torch.double),
        },
    )
    mean = 0.3 - 0.2 * data.x.to_numpy()
    for method in ("auto", "model"):
        actual = result.bayes_r2(method=method)
        assert actual.shape == (2, 3)
        torch.testing.assert_close(
            actual, torch.full_like(actual, _ratio(mean, variance=0.25))
        )
    residual = result.bayes_r2(method="residual")
    torch.testing.assert_close(
        residual, torch.full_like(residual, _ratio(mean, observed=data.y.to_numpy()))
    )
    fresh = pd.DataFrame({"x": [2.0, 3.0, 4.0], "y": [-0.2, 0.5, -0.9]})
    got = result.bayes_r2(method="residual", new_data=fresh)
    torch.testing.assert_close(
        got,
        torch.full_like(
            got, _ratio(0.3 - 0.2 * fresh.x.to_numpy(), observed=fresh.y.to_numpy())
        ),
    )
    assert result.bayes_r2(new_data=fresh.drop(columns="y")).shape == (2, 3)


def test_heteroscedastic_gaussian_uses_mean_variance_not_variance_of_sigma():
    data = pd.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, 2.0],
            "z": [-0.5, 0.0, 0.5, 1.0],
            "y": [0.0, 1.0, 1.5, 0.0],
        }
    )
    result = fit(
        "y~x", data=data, distributional={"sigma": "~z"}, method="svi", num_steps=0
    )
    result = _recorded(
        result,
        {
            name: torch.tensor(value, dtype=torch.double)
            for name, value in {
                "intercept": 0.3,
                "beta_x": 0.4,
                "dpar_sigma_intercept": -0.5,
                "dpar_sigma_beta_z": 0.2,
            }.items()
        },
    )
    mean = 0.3 + 0.4 * data.x.to_numpy()
    variance = np.exp(2 * (-0.5 + 0.2 * data.z.to_numpy()))
    actual = result.bayes_r2(method="model")
    torch.testing.assert_close(
        actual, torch.full_like(actual, _ratio(mean, variance=variance))
    )


@pytest.mark.parametrize("family", ["bernoulli", "binomial", "poisson"])
def test_count_model_variances_use_response_counts_and_row_trials(family):
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    data = pd.DataFrame({"x": x, "y": [0, 1, 0, 1], "n": [3, 4, 5, 6]})
    kwargs = {"binomial_trials": "n"} if family == "binomial" else {}
    result = fit("y~x", data=data, family=family, method="svi", num_steps=0, **kwargs)
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.2, dtype=torch.double),
            "beta_x": torch.tensor(0.4, dtype=torch.double),
        },
    )
    if family == "poisson":
        mean = np.exp(0.2 + 0.4 * x)
        variance = mean
    else:
        p = expit(0.2 + 0.4 * x)
        trials = data.n.to_numpy() if family == "binomial" else 1
        mean = trials * p
        variance = trials * p * (1 - p)
    actual = result.bayes_r2(method="model")
    torch.testing.assert_close(
        actual, torch.full_like(actual, _ratio(mean, variance=variance))
    )
    if family == "poisson":
        actual = result.bayes_r2()
        torch.testing.assert_close(
            actual, torch.full_like(actual, _ratio(mean, observed=data.y.to_numpy()))
        )


def test_new_response_transform_retains_training_state_and_requires_response_values():
    data = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0, 4.0]})
    result = fit(
        "log(y)~x",
        data=data,
        fixed_parameters={"sigma": 0.4},
        method="svi",
        num_steps=0,
    )
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.3, dtype=torch.double),
            "beta_x": torch.tensor(0.2, dtype=torch.double),
        },
    )
    fresh = pd.DataFrame({"x": [-2.0, 1.0, 3.0], "y": [2.0, 5.0, 9.0]})
    actual = result.bayes_r2(method="residual", new_data=fresh)
    torch.testing.assert_close(
        actual,
        torch.full_like(
            actual,
            _ratio(0.3 + 0.2 * fresh.x.to_numpy(), observed=np.log(fresh.y.to_numpy())),
        ),
    )
    with pytest.raises(ValueError, match="response"):
        result.bayes_r2(method="residual", new_data=fresh.drop(columns="y"))
    explicit = result.bayes_r2(
        method="residual",
        new_data=fresh.drop(columns="y"),
        observed=torch.tensor(np.log(fresh.y.to_numpy())),
    )
    torch.testing.assert_close(actual, explicit)


def test_invalid_estimator_options_and_unordered_categories_are_refused():
    data = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 1.0]})
    result = fit("y~x", data=data, method="svi", num_steps=0)
    for kwargs in [
        {"method": "other"},
        {"method": "model", "mc_samples": 3},
        {"method": "model", "observed": torch.zeros(4)},
    ]:
        with pytest.raises(ValueError):
            result.bayes_r2(**kwargs)
    categorical = fit("y~x", data=data, family="categorical", method="svi", num_steps=0)
    with pytest.raises(ValueError, match="unordered"):
        categorical.bayes_r2()


@pytest.mark.parametrize("hurdle", [False, True])
def test_ordered_category_scores_use_retained_labels_and_complete_probabilities(hurdle):
    labels = ["low", "middle", "high"]
    data = pd.DataFrame(
        {
            "x": [-1.0, 0.0, 1.0, 2.0],
            "y": pd.Categorical(
                ["low", "middle", "high", "middle"], categories=labels, ordered=True
            ),
        }
    )
    if hurdle:
        labels = ["zero", *labels]
        data["y"] = pd.Categorical(
            ["zero", "low", "high", "middle"], categories=labels, ordered=True
        )
    result = fit(
        "y~x",
        data=data,
        family="hurdle_cumulative" if hurdle else "cumulative",
        fixed_parameters={"hu": 0.2} if hurdle else {},
        method="svi",
        num_steps=0,
    )
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.3, dtype=torch.double),
            "beta_x": torch.tensor(0.4, dtype=torch.double),
            "cutpoint_log_spacing": torch.tensor([np.log(1.4)], dtype=torch.double),
        },
    )
    eta = 0.3 + 0.4 * data.x.to_numpy()
    cdf = expit(np.array([-0.7, 0.7])[None, :] - eta[:, None])
    probs = np.column_stack((cdf[:, 0], cdf[:, 1] - cdf[:, 0], 1 - cdf[:, 1]))
    if hurdle:
        probs = np.column_stack((np.full(len(data), 0.2), 0.8 * probs))
    codes = np.arange(len(labels))
    mean = probs @ codes
    variance = np.sum(probs * (codes - mean[:, None]) ** 2, axis=-1)
    model = result.bayes_r2(method="model")
    torch.testing.assert_close(
        model, torch.full_like(model, _ratio(mean, variance=variance))
    )
    actual = result.bayes_r2(method="residual")
    torch.testing.assert_close(
        actual,
        torch.full_like(actual, _ratio(mean, observed=data.y.cat.codes.to_numpy())),
    )
    # New rows use the fitted ordering, even when the frame's category order changes.
    fresh = data.iloc[[2, 0, 3]].copy()
    fresh["y"] = fresh.y.cat.reorder_categories(list(reversed(labels)))
    got = result.bayes_r2(method="residual", new_data=fresh)
    expected = _ratio(mean[[2, 0, 3]], observed=data.y.cat.codes.to_numpy()[[2, 0, 3]])
    torch.testing.assert_close(got, torch.full_like(got, expected))


def test_wiener_model_moments_use_unsigned_time_with_independent_first_passage_oracle():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    data = pd.DataFrame({"x": x, "rt": [0.4, 0.7, 0.5, 0.9], "choice": [0, 1, 0, 1]})
    result = fit(
        "rt|dec(choice)~x",
        data=data,
        family="wiener",
        fixed_parameters={"bs": 1.2, "ndt": 0.1, "bias": 0.5},
        method="svi",
        num_steps=0,
    )
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.6, dtype=torch.double),
            "beta_x": torch.tensor(0.2, dtype=torch.double),
        },
    )
    drift = 0.6 + 0.2 * x
    # QVR fixed numeric constants retain their compiled Float32 literal value.
    half = float(np.float32(1.2)) / 2
    # Exit-time moments for unit Brownian diffusion from [-half, half].
    mean = 0.1 + half * np.tanh(half * drift) / drift
    variance = half * np.tanh(half * drift) / drift**3 - half**2 / (
        drift**2 * np.cosh(half * drift) ** 2
    )
    actual = result.bayes_r2(method="model")
    torch.testing.assert_close(
        actual,
        torch.full_like(actual, _ratio(mean, variance=variance)),
        rtol=2e-7,
        atol=1e-10,
    )
    residual = result.bayes_r2()
    torch.testing.assert_close(
        residual,
        torch.full_like(residual, _ratio(mean, observed=data.rt.to_numpy())),
        rtol=2e-7,
        atol=1e-10,
    )


def test_residual_requires_actual_censored_values_and_new_rows_accept_exact_responses():
    data = pd.DataFrame(
        {"x": [-1.0, 0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 1.0], "status": [0, 1, 0, -1]}
    )
    result = fit(
        "y|cens(status)~x",
        data=data,
        fixed_parameters={"sigma": 0.5},
        method="svi",
        num_steps=0,
    )
    result = _recorded(
        result, {"intercept": torch.tensor(0.3), "beta_x": torch.tensor(0.4)}
    )
    with pytest.raises(ValueError, match="actual responses"):
        result.bayes_r2(method="residual")
    actual_values = torch.tensor([0.0, 1.3, 2.0, 0.7])
    explicit = result.bayes_r2(observed=actual_values)
    expected_mean = 0.3 + 0.4 * data.x.to_numpy()
    torch.testing.assert_close(
        explicit,
        torch.full_like(
            explicit, _ratio(expected_mean, observed=actual_values.numpy())
        ),
        rtol=2e-6,
        atol=1e-8,
    )
    got = result.bayes_r2(method="residual", new_data=data.drop(columns="status"))
    torch.testing.assert_close(
        got,
        torch.full_like(got, _ratio(expected_mean, observed=data.y.to_numpy())),
        rtol=2e-6,
        atol=1e-8,
    )
    with pytest.raises(TypeError, match="tensor"):
        result.bayes_r2(observed=[1.0, 2.0, 3.0, 4.0])


def test_variational_r2_respects_seed_draw_count_and_callers_random_state():
    data = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0], "y": [0.0, 1.0, 2.0, 1.0]})
    result = fit(
        "y~x", data=data, fixed_parameters={"sigma": 0.5}, method="svi", num_steps=1
    )
    before = torch.random.get_rng_state().clone()
    first = result.bayes_r2(num_samples=5, seed=7)
    torch.testing.assert_close(before, torch.random.get_rng_state())
    torch.testing.assert_close(first, result.bayes_r2(num_samples=5, seed=7))
    assert first.shape == (1, 5)
    assert not torch.equal(first, result.bayes_r2(num_samples=5, seed=8))
