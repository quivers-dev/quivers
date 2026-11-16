"""Included-row identities and neural inputs stay aligned through refits."""

import pandas as pd
import pytest
import torch

from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.effects.observation_scoring import observation_scoring
from quivers.formulas import fit
from quivers.formulas._fit import _predictor_output
from quivers.modeling._row_attachments import AttachedRows, RowAttachment


def _frame():
    return pd.DataFrame(
        {
            "y": [0.2, 99.0, -0.4, 0.6, 99.0, 0.1],
            "x": [1.0, 9.0, -1.0, 0.5, 9.0, 2.0],
            "z": [-0.5, 9.0, 0.5, 1.0, 9.0, -0.25],
            "keep": [True, False, True, True, False, True],
            "w": [0.2, 1.0, 1.0, 0.5, 1.0, 2.0],
        },
        index=[8, 8, 3, 17, 3, 2],
    )


def _neural_fit(*, distributional=False):
    predictor = torch.nn.Linear(2, 1)
    with torch.no_grad():
        predictor.weight.copy_(torch.tensor([[0.3, -0.2]]))
        predictor.bias.fill_(0.1)
    inputs = torch.tensor(
        [[0.0, 1.0], [9.0, 9.0], [1.0, 2.0], [-1.0, 0.5], [9.0, 9.0], [0.5, -1.0]]
    )
    options = (
        {"distributional": {"sigma": "~z"}}
        if distributional
        else {"fixed_parameters": {"sigma": 1.0}}
    )
    fitted = fit(
        "y | subset(keep) + weights(w) ~ x",
        data=_frame(),
        method="svi",
        num_steps=0,
        predictor=predictor,
        predictor_data=inputs,
        **options,
    )
    return fitted, inputs


def _method(settings, seed):
    return {**settings, "seed": seed}


def test_neural_subsets_use_included_positions_and_raw_heldout_density():
    fitted, inputs = _neural_fit()
    callback = fitted.refit_callback(method_factory=_method, num_samples=8)
    plan = FoldPlan(assignments=(0, 0, 1, 1), seed=11)
    included = (0, 2, 3, 5)
    for fold in range(2):
        split = plan.split(fold)
        prediction = callback(split)
        actual = prediction.fit
        train = tuple(included[i] for i in split.train)
        test = tuple(included[i] for i in split.test)
        torch.testing.assert_close(actual.predictor_data, inputs[list(train)])
        torch.testing.assert_close(
            actual.refit_recipe.method["predictor_data"], inputs[list(train)]
        )
        torch.testing.assert_close(
            torch.tensor(actual.formula.response_values),
            torch.tensor(_frame().iloc[list(train)].y.to_numpy()),
        )
        draws = actual.draws(num_samples=8, seed=split.seed)
        neural = inputs[list(test)] @ actual.predictor.weight.T + actual.predictor.bias
        mu = (
            draws["intercept"][..., None]
            + draws["beta_x"][..., None]
            * draws["beta_x"].new_tensor(_frame().iloc[list(test)].x.to_numpy())
            + neural.flatten()
        )
        expected = torch.distributions.Normal(mu, 1.0).log_prob(
            mu.new_tensor(_frame().iloc[list(test)].y.to_numpy())
        )
        torch.testing.assert_close(prediction.log_likelihood["y"], expected)
    assert fitted.formula.modifiers.subset.positions == included


def test_partitioned_location_auxiliary_neural_likelihood_and_every_head_gradient():
    fitted, inputs = _neural_fit(distributional=True)
    split = FoldPlan(assignments=(0, 0, 1, 1)).split(0)
    refitted = fitted.refit_callback(method_factory=_method, num_samples=2)(split).fit
    train = (3, 5)
    values = torch.tensor([0.2, -0.3, 0.1, 0.4], requires_grad=True)
    names = ("intercept", "beta_x", "dpar_sigma_intercept", "dpar_sigma_beta_z")
    assert set(refitted.posterior._registry.names) == set(names)
    bindings = {**refitted.observations, **dict(zip(names, values, strict=True))}
    bindings[refitted.predictor_name] = _predictor_output(
        refitted.predictor, refitted.predictor_data, n_obs=2
    )
    x = torch.zeros(1, 1)
    joint = refitted.program.log_joint(x, bindings).sum()
    with observation_scoring(False):
        prior = refitted.program.log_joint(x, bindings).sum()
    actual = joint - prior
    data = _frame().iloc[list(train)]
    neural = inputs[list(train)] @ refitted.predictor.weight.T + refitted.predictor.bias
    mu = values[0] + values[1] * torch.tensor(data.x.to_numpy()) + neural.flatten()
    scale = (values[2] + values[3] * torch.tensor(data.z.to_numpy())).exp()
    expected = (
        torch.tensor(data.w.to_numpy())
        * torch.distributions.Normal(mu, scale).log_prob(
            torch.tensor(data.y.to_numpy())
        )
    ).sum()
    torch.testing.assert_close(actual, expected)
    parameters = (values, *refitted.predictor.parameters())
    actual_gradient = torch.autograd.grad(actual, parameters, retain_graph=True)
    expected_gradient = torch.autograd.grad(expected, parameters)
    for actual, expected in zip(actual_gradient, expected_gradient, strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "failure", ["shape", "nonfinite", "keyword", "count", "replacement"]
)
def test_bad_attachments_options_and_row_plans_fail_before_fitting(failure):
    fitted, _ = _neural_fit()
    calls = []

    def method(settings, seed):
        calls.append(seed)
        return _method(settings, seed)

    if failure == "shape":
        fitted.refit_recipe.method["predictor_data"].resize_(5, 2)
        with pytest.raises(ValueError, match="original rows"):
            fitted.refit_callback(method_factory=method)
    elif failure == "nonfinite":
        fitted.refit_recipe.method["predictor_data"].fill_(float("nan"))
        with pytest.raises(ValueError, match="finite"):
            fitted.refit_callback(method_factory=method)
    elif failure == "keyword":
        with pytest.raises(TypeError, match="unexpected keyword"):
            fitted.refit_callback(method_factory=method, typo_policy=True)
    elif failure == "count":
        callback = fitted.refit_callback(method_factory=method)
        with pytest.raises(ValueError, match="exactly 4"):
            callback(FoldPlan(assignments=(0, 1)).split(0))
    else:
        callback = fitted.refit_callback(
            method_factory=lambda settings, seed: {
                **settings,
                "predictor_data": torch.zeros(6, 2),
            }
        )
        with pytest.raises(ValueError, match="preserve retained row attachment"):
            callback(FoldPlan(assignments=(0, 0, 1, 1)).split(0))
    assert calls == []


def test_tensor_attachments_select_only_their_declared_axes_and_preserve_gradients():
    value = torch.arange(24.0, requires_grad=True).reshape(2, 6, 2)
    attachment = RowAttachment("training", "scoring", value, axis=1)
    attachment.validate(6)
    constants = torch.eye(6)
    payload = AttachedRows(
        {"rows": torch.arange(6), "constants": constants}, (attachment,)
    )
    selected = payload.partition(
        (5, 1), lambda data, rows: {**data, "rows": data["rows"][list(rows)]}
    )
    assert selected.data["constants"] is constants
    torch.testing.assert_close(
        selected.keywords(scoring=False)["training"], value[:, [5, 1]]
    )
    assert selected.keywords(scoring=True)["scoring"] is selected.attachments[0].value
    gradient = torch.autograd.grad(selected.attachments[0].value.sum(), value)[0]
    expected = torch.zeros_like(value)
    expected[:, [5, 1]] = 1
    torch.testing.assert_close(gradient, expected)


def test_formula_groups_keep_explicit_new_level_policy_and_joint_row_units():
    data = pd.DataFrame(
        {"y": [0.2, 0.4, -0.3, -0.1, 0.7, 0.8], "g": ["a", "a", "b", "b", "c", "c"]}
    )
    fitted = fit(
        "y~1+(1|g)",
        data=data,
        method="svi",
        num_steps=0,
        fixed_parameters={"sigma": 1.0},
    )
    plan = FoldPlan(assignments=(0, 0, 1, 1, 2, 2))
    split = plan.split(0)
    with pytest.raises(ValueError, match="new.*level|new.*group"):
        fitted.refit_callback(method_factory=_method, num_samples=2)(split)
    callback = fitted.refit_callback(
        method_factory=_method,
        num_samples=4,
        allow_new_levels=True,
        sample_new_levels="gaussian",
    )
    result = cross_validate(
        plan, callback, joint="group", groups=tuple(data.g), save_fits=True
    )
    assert result.complete and result.log_predictive_density.shape == (3,)
    assert torch.isfinite(result.log_predictive_density).all()
    for fold, refitted in result.fits.items():
        assert (
            data.g.iloc[list(plan.split(fold).test)].iloc[0]
            not in refitted.group_levels["g"]
        )


def test_equal_size_neural_weights_are_constants_not_row_attachments():
    data = pd.DataFrame({"y": [0.2, 0.4, 0.1, 0.8]})
    predictor = torch.nn.Sequential(
        torch.nn.Linear(4, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1)
    )
    inputs = torch.arange(16.0).reshape(4, 4) / 16
    original = predictor[0].weight.detach().clone()
    fitted = fit(
        "y~1",
        data=data,
        method="svi",
        num_steps=0,
        predictor=predictor,
        predictor_data=inputs,
    )
    refitted = fitted.refit_callback(method_factory=_method, num_samples=2)(
        FoldPlan(assignments=(0, 0, 1, 1)).split(0)
    ).fit
    assert refitted.predictor_data.shape == (2, 4)
    torch.testing.assert_close(refitted.predictor[0].weight, original)


def test_neural_training_refits_are_order_independent_and_preserve_factory_seed():
    fitted, _ = _neural_fit()
    original = {
        key: value.detach().clone()
        for key, value in fitted.predictor.state_dict().items()
    }
    plan = FoldPlan(assignments=(0, 0, 1, 1), seed=19)

    def method(settings, seed):
        return {**settings, "seed": seed + 100, "num_steps": 2}

    callback = fitted.refit_callback(method_factory=method, num_samples=2)
    first = cross_validate(plan, callback, fold_ids=(0, 1), save_fits=True)
    second = cross_validate(plan, callback, fold_ids=(1, 0), save_fits=True)
    for fold in (0, 1):
        left, right = first.fits[fold], second.fits[fold]
        assert left.refit_recipe.method["seed"] == 120 + fold
        assert left.losses == right.losses
        for key, value in left.predictor.state_dict().items():
            torch.testing.assert_close(value, right.predictor.state_dict()[key])
    for key, value in fitted.predictor.state_dict().items():
        torch.testing.assert_close(value, original[key])
