"""New group identities preserve joint donor choices and conditional Gaussian laws."""

import pandas as pd
import pytest
import torch

from quivers.formulas import fit
from quivers.formulas._prediction import prepare_prediction
from quivers.inference.mcmc.driver import MCMCResult


def _fit(parameterization="noncentered", formula="y ~ x + (1 + x | g)", **kwargs):
    return fit(
        formula,
        data=pd.DataFrame(
            {
                "y": [0.0, 1.0, 0.0, 1.0],
                "x": [0.0, 1.0, 0.0, 1.0],
                "g": ["a", "a", "b", "b"],
            }
        ),
        method="svi",
        num_steps=1,
        reparameterize=parameterization,
        fixed_parameters={"sigma": 1.0},
        **kwargs,
    )


def _recorded(result, values, count=8):
    return result.with_(
        posterior=MCMCResult(
            samples={
                name: value.expand(2, count, *value.shape).clone()
                for name, value in values.items()
            },
            log_densities=torch.zeros(2, count),
            acceptance_rates=torch.ones(2),
            divergence_counts=torch.zeros(2, dtype=torch.long),
            num_warmup=0,
            num_samples=count,
        )
    )


def _values(centered=False):
    values = {
        "intercept": torch.tensor(0.3),
        "beta_x": torch.tensor(0.2),
        "sigma_g_Intercept": torch.tensor(1.0),
        "sigma_g_x": torch.tensor(1.0),
        "cor_g": torch.eye(2),
    }
    values.update(
        {"group_coefficients_g": torch.tensor([[1.0, 10.0], [3.0, 30.0]])}
        if centered
        else {
            "z_g_Intercept": torch.tensor([1.0, 3.0]),
            "z_g_x": torch.tensor([10.0, 30.0]),
        }
    )
    return values


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_known_levels_compact_gather_and_selected_terms_match_manual_predictors(
    parameterization,
):
    result = _recorded(_fit(parameterization), _values(parameterization == "centered"))
    frame = pd.DataFrame({"x": [2.0, -1.0, 0.0], "g": ["b", "a", "b"]})
    expected = torch.tensor([0.3 + 0.4 + 3 + 60, 0.3 - 0.2 + 1 - 10, 0.3 + 3])
    torch.testing.assert_close(
        result.posterior_linpred(new_data=frame), expected.expand(2, 8, 3)
    )
    no_groups = result.posterior_linpred(
        new_data=frame.drop(columns="g"), re_formula="~0"
    )
    torch.testing.assert_close(no_groups, torch.tensor([0.7, 0.1, 0.3]).expand(2, 8, 3))
    intercept = result.posterior_linpred(new_data=frame, re_formula="~(1 | g)")
    torch.testing.assert_close(intercept, torch.tensor([3.7, 1.1, 3.3]).expand(2, 8, 3))
    slope = result.posterior_linpred(new_data=frame, re_formula="~(0 + x | g)")
    torch.testing.assert_close(slope, torch.tensor([60.7, -9.9, 0.3]).expand(2, 8, 3))
    training = result.posterior_linpred(re_formula="~0")
    torch.testing.assert_close(
        training, torch.tensor([0.3, 0.5, 0.3, 0.5]).expand(2, 8, 4)
    )


@pytest.mark.parametrize("policy", ["uncertainty", "old_levels"])
@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_new_group_donors_are_joint_across_slopes_and_repeated_identities(
    policy, parameterization
):
    result = _recorded(
        _fit(parameterization), _values(parameterization == "centered"), count=20
    )
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 0.0, 1.0, 0.0], "g": ["new", "new", "a", "a", "new2"]}
    )
    predicted = result.posterior_linpred(
        new_data=frame, allow_new_levels=True, sample_new_levels=policy, seed=13
    )
    group_intercept = predicted[..., 0] - 0.3
    group_slope = predicted[..., 1] - 0.5 - group_intercept
    torch.testing.assert_close(group_slope, 10 * group_intercept)
    assert (
        torch.isclose(group_intercept, torch.tensor(1.0))
        .logical_or(torch.isclose(group_intercept, torch.tensor(3.0)))
        .all()
    )
    torch.testing.assert_close(predicted[..., 2], torch.full((2, 20), 1.3))
    torch.testing.assert_close(predicted[..., 3], torch.full((2, 20), 11.5))
    if policy == "old_levels":
        assert group_intercept.unique().numel() == 1
    else:
        assert group_intercept.unique().numel() == 2
    assert torch.equal(
        predicted,
        result.posterior_linpred(
            new_data=frame, allow_new_levels=True, sample_new_levels=policy, seed=13
        ),
    )


@pytest.mark.parametrize("parameterization", ["centered", "noncentered"])
def test_gaussian_new_levels_follow_fitted_covariance_and_preserve_known_levels(
    parameterization,
):
    values = _values(parameterization == "centered")
    values["cor_g"] = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    values["sigma_g_Intercept"] = torch.tensor(0.4)
    values["sigma_g_x"] = torch.tensor(0.7)
    result = _recorded(_fit(parameterization), values, count=400)
    frame = pd.DataFrame({"x": [0.0, 1.0, 0.0], "g": ["new", "new", "a"]})
    predicted = result.posterior_linpred(
        new_data=frame, allow_new_levels=True, sample_new_levels="gaussian", seed=49
    )
    intercept = predicted[..., 0] - 0.3
    slope = predicted[..., 1] - 0.5 - intercept
    assert abs(intercept.mean()) < 0.06
    assert abs(slope.mean()) < 0.1
    covariance = torch.stack([intercept.reshape(-1), slope.reshape(-1)]).cov()
    torch.testing.assert_close(
        covariance, torch.tensor([[0.16, 0.168], [0.168, 0.49]]), atol=0.06, rtol=0.15
    )
    expected_known = 1.3 if parameterization == "centered" else 0.7
    torch.testing.assert_close(predicted[..., 2], torch.full((2, 400), expected_known))


@pytest.mark.parametrize("policy", ["uncertainty", "old_levels", "gaussian"])
def test_missing_group_values_share_one_identity_and_rng_is_restored(policy):
    result = _recorded(_fit(), _values())
    frame = pd.DataFrame({"x": [0.0, 0.0, 0.0], "g": [None, None, "a"]})
    before = torch.random.get_rng_state().clone()
    predicted = result.posterior_epred(
        new_data=frame, allow_new_levels=True, sample_new_levels=policy, seed=4
    )
    assert torch.equal(before, torch.random.get_rng_state())
    torch.testing.assert_close(predicted[..., 0], predicted[..., 1])
    all_missing = result.posterior_epred(
        new_data=frame.drop(columns="g"),
        allow_new_levels=True,
        sample_new_levels=policy,
        seed=4,
    )
    torch.testing.assert_close(all_missing[..., 0], all_missing[..., 2])
    sampled = result.posterior_predict(
        new_data=frame, allow_new_levels=True, sample_new_levels=policy, seed=4
    )
    assert torch.equal(before, torch.random.get_rng_state())
    assert not torch.equal(sampled[..., 0], sampled[..., 1])


def test_compression_does_not_retain_unused_training_group_axes():
    result = _fit()
    context = prepare_prediction(
        result, pd.DataFrame({"x": [0.0, 0.0], "g": ["b", "b"]}), None
    )
    assert context.groups["g"].fitted_indices == (1,)
    assert context.groups["g"].row_indices.tolist() == [0, 0]


@pytest.mark.parametrize(
    "selector", ["~(1 | h)", "~(0 + z | g)", "~x", "y ~ (1 | g)", "~1"]
)
def test_unknown_or_fixed_effect_selectors_are_rejected(selector):
    result = _fit()
    with pytest.raises(ValueError, match="re_formula"):
        result.posterior_epred(re_formula=selector)


def test_new_levels_require_opt_in_and_categorical_predictors_stay_strict():
    result = _fit()
    frame = pd.DataFrame({"x": [0.0], "g": ["new"]})
    with pytest.raises(ValueError, match="allow_new_levels"):
        result.posterior_epred(new_data=frame)
    with pytest.raises(ValueError, match="sample_new_levels"):
        result.posterior_epred(new_data=frame, sample_new_levels="other")
    with pytest.raises(TypeError, match="bool"):
        result.posterior_epred(new_data=frame, allow_new_levels=1)


def test_donor_identity_is_shared_across_auxiliary_and_location_predictors():
    result = fit(
        "y ~ (1 | shared | g)",
        data=pd.DataFrame({"y": [0.0, 1.0, 0.0, 1.0], "g": ["a", "a", "b", "b"]}),
        distributional={"sigma": "~ (1 | shared | g)"},
        method="svi",
        num_steps=1,
    )
    values = {
        "intercept": torch.tensor(0.0),
        "dpar_sigma_intercept": torch.tensor(0.0),
        "sigma_g_Intercept": torch.tensor(1.0),
        "dpar_sigma_sigma_g_Intercept": torch.tensor(1.0),
        "cor_shared_shared": torch.eye(2),
        "z_g_Intercept": torch.tensor([1.0, 3.0]),
        "dpar_sigma_z_g_Intercept": torch.tensor([10.0, 30.0]),
    }
    result = _recorded(result, values, count=20)
    frame = pd.DataFrame({"g": ["new", "new2", "a"]})
    location = result.posterior_linpred(new_data=frame, allow_new_levels=True, seed=7)
    scale_eta = result.posterior_linpred(
        parameter="sigma", new_data=frame, allow_new_levels=True, seed=7
    )
    torch.testing.assert_close(scale_eta, 10 * location)


def test_ordinary_unknown_categories_are_rejected_even_when_new_groups_are_allowed():
    result = fit(
        "y ~ c + (1 | g)",
        data=pd.DataFrame(
            {
                "y": [0.0, 1.0, 0.0, 1.0],
                "c": ["u", "v", "u", "v"],
                "g": ["a", "a", "b", "b"],
            }
        ),
        method="svi",
        num_steps=1,
    )
    with pytest.raises(ValueError, match="unknown levels"):
        result.posterior_epred(
            new_data=pd.DataFrame({"c": ["unseen"], "g": ["new"]}),
            allow_new_levels=True,
        )


def test_excluded_random_slopes_need_no_columns_even_when_shared_by_groups():
    frame = pd.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0],
            "x": [0.0, 1.0, 0.0, 1.0],
            "g": ["a", "a", "b", "b"],
            "h": ["j", "k", "j", "k"],
        }
    )
    result = fit(
        "y ~ 1 + (0 + x | g) + (0 + x | h)", data=frame, method="svi", num_steps=1
    )
    output = result.posterior_epred(
        new_data=pd.DataFrame(index=range(2)), re_formula="~0", num_samples=3
    )
    assert output.shape == (1, 3, 2)
    torch.testing.assert_close(output[..., 0], output[..., 1])


def test_new_missing_identity_does_not_collide_with_literal_na_group_label():
    result = fit(
        "y ~ (1 | g)",
        data=pd.DataFrame({"y": [0.0, 1.0, 0.0, 1.0], "g": ["NA", "NA", "b", "b"]}),
        method="svi",
        num_steps=1,
        fixed_parameters={"sigma": 1.0},
    )
    result = _recorded(
        result,
        {
            "intercept": torch.tensor(0.0),
            "sigma_g_Intercept": torch.tensor(1.0),
            "z_g_Intercept": torch.tensor([100.0, 300.0]),
        },
    )
    output = result.posterior_epred(
        new_data=pd.DataFrame({"g": [None, "NA"]}),
        allow_new_levels=True,
        sample_new_levels="gaussian",
        seed=32,
    )
    torch.testing.assert_close(output[..., 1], torch.full((2, 8), 100.0))
    assert (output[..., 0].abs() < 10).all()


@pytest.mark.parametrize("policy", ["uncertainty", "old_levels", "gaussian"])
def test_categorical_group_major_cells_preserve_logits_and_joint_category_donors(
    policy,
):
    result = fit(
        "y ~ 1 + (1 | p | g)",
        data=pd.DataFrame({"y": [0, 1, 2, 0], "g": ["a", "a", "b", "b"]}),
        family="categorical",
        method="svi",
        num_steps=1,
    )
    factor = torch.tensor([[1.0, 0.0], [0.6, 0.8]])
    scale = torch.tensor([0.4, 0.7])
    z = torch.tensor([[0.2, 1.0], [-0.5, 0.3]])
    intercept = torch.tensor([0.3, -0.2])
    coefficients = z @ factor.T * scale
    result = _recorded(
        result,
        {
            "intercept": intercept,
            "sigma_g_Intercept": scale,
            "cor_shared_p": factor,
            "z_g_Intercept": z.reshape(-1),
        },
        count=12,
    )
    frame = pd.DataFrame({"g": ["b", "new", "new"]})
    logits = result.posterior_linpred(
        new_data=frame, allow_new_levels=True, sample_new_levels=policy, seed=31
    )
    expected_known = torch.cat((torch.zeros(1), intercept + coefficients[1]))
    torch.testing.assert_close(logits[..., 0, :], expected_known.expand(2, 12, 3))
    torch.testing.assert_close(logits[..., 1, :], logits[..., 2, :])
    if policy != "gaussian":
        candidate = logits[..., 1, 1:] - intercept
        distance = (candidate[..., None, :] - coefficients).abs().amax(-1)
        assert (distance.amin(-1) < 1e-6).all()
    probability = result.posterior_epred(
        new_data=frame, allow_new_levels=True, sample_new_levels=policy, seed=31
    )
    torch.testing.assert_close(probability, logits.softmax(-1))
    excluded = result.posterior_linpred(
        new_data=frame.drop(columns="g"), re_formula="~0"
    )
    expected_population = torch.cat((torch.zeros(1), intercept))
    torch.testing.assert_close(excluded, expected_population.expand(2, 12, 3, 3))
