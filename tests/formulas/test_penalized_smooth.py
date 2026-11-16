"""Penalized spline formulas retain Gaussian laws and fitted transformations."""

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler, parse
from quivers.dsl.emit import module_to_source
from quivers.formulas import (
    FormulaToQVRModule,
    PriorSpec,
    families,
    fit,
    formula_from_data,
)
from quivers.inference.trace import trace


FORMULA = "y~s(x,bs='ps',k=6)"


def _frame():
    x = np.linspace(-1, 1, 12)
    return pd.DataFrame(
        {"x": x, "y": 0.1 + 0.4 * x * x, "z": np.linspace(-0.4, 0.8, 12)}
    )


def test_pspline_nullspace_and_penalized_block_have_exact_source_law_and_every_gradient():
    formula = formula_from_data(FORMULA, _frame())
    term = formula.design_terms[0]
    assert term.data.shape == (12, 4)
    assert len(formula.fixed_columns) == 2
    lens = FormulaToQVRModule(families["gaussian"])
    module, complement = lens.forward(formula)
    assert lens.backward(module, complement) == formula
    source = module_to_source(module)
    assert module_to_source(parse(source)) == source
    program = Compiler(module).compile().morphism
    column = formula.fixed_columns[1]
    name = term.qvr_name
    values = lens.fixed_column_observations(formula)
    values["y"] = torch.tensor(formula.response_values.copy(), dtype=torch.float64)
    point = torch.tensor(
        [0.1, 0.3, 0.8, 0.1, -0.2, 0.3, -0.1, 0.7],
        dtype=torch.float64,
        requires_grad=True,
    )

    def actual(v):
        observations = {
            **values,
            "intercept": v[0],
            f"beta_{column.qvr_name}": v[1],
            name + "_scale": v[2],
            name + "_z": v[3:7],
            "sigma": v[7],
        }
        return trace(program, v.new_zeros(1, 1), observations).log_joint.sum()

    def oracle(v):
        fixed = torch.tensor(column.data.copy(), dtype=torch.float64)
        random = torch.tensor(term.data.copy(), dtype=torch.float64)
        mu = v[0] + v[1] * fixed + random @ (v[2] * v[3:7])
        return (
            torch.distributions.Normal(0.0, 5.0).log_prob(v[0])
            + torch.distributions.Normal(
                0.0, 5.0 / fixed.square().mean().sqrt()
            ).log_prob(v[1])
            + torch.distributions.HalfNormal(1.0).log_prob(v[2])
            + torch.distributions.Normal(0.0, 1.0).log_prob(v[3:7]).sum()
            + torch.distributions.HalfCauchy(2.0).log_prob(v[7])
            + torch.distributions.Normal(mu, v[7]).log_prob(values["y"]).sum()
        )

    torch.testing.assert_close(actual(point), oracle(point))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(oracle, point),
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(oracle, point),
    )
    targets = lens.prior_targets
    assert next(t for t in targets if t.site == name + "_scale").class_ == "sds"


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_smooth_fitting_and_prediction_retain_knots_coefficients_and_request_isolation(
    method,
):
    kwargs = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 1, "num_samples": 2, "num_chains": 1}
    )
    result = fit(
        FORMULA, data=_frame(), family="gaussian", method=method, seed=4, **kwargs
    )
    original = result.formula
    fresh = pd.DataFrame({"x": [0.2, 1.3]})
    one = result.posterior_epred(new_data=fresh, num_samples=2, seed=11)
    assert one.shape == (1, 2, 2)
    assert torch.isfinite(one).all()
    result.posterior_epred(
        new_data=pd.DataFrame({"x": [-0.5, 0.5, 1.7]}), num_samples=2, seed=12
    )
    torch.testing.assert_close(
        result.posterior_epred(new_data=fresh, num_samples=2, seed=11), one
    )
    assert result.formula is original
    assert (
        result.posterior_predict(new_data=fresh, num_samples=2, seed=11).shape
        == one.shape
    )


def test_auxiliary_smooth_uses_same_prior_and_prediction_paths_and_typed_override():
    result = fit(
        "y~1+x",
        data=_frame(),
        distributional={"sigma": "~s(z,bs='ps',k=5)"},
        family="gaussian",
        method="svi",
        num_steps=2,
        priors=PriorSpec(distribution="HalfNormal(0.4)", class_="sds", dpar="sigma"),
    )
    assert "HalfNormal(scale=0.4)" in result.qvr_source
    assert torch.isfinite(
        result.posterior_epred(
            new_data=pd.DataFrame({"x": [0.1, 0.2], "z": [0.3, 0.5]}), num_samples=2
        )
    ).all()


@pytest.mark.parametrize(
    "formula",
    [
        "y~s(x)",
        "y~s(x,bs='tp')",
        "y~s(x,bs='ps',k=6)*z",
        "y~(s(x,bs='ps',k=6)|z)",
    ],
)
def test_unimplemented_smooth_constructions_are_never_unpenalized_fixed_columns(
    formula,
):
    with pytest.raises(ValueError, match="thin-plate|standalone|structured smooth"):
        formula_from_data(formula, _frame())


def test_categorical_smooth_retains_response_component_coefficient_axes():
    frame = _frame()
    frame["y"] = np.tile([0, 1, 2], 4)
    result = fit(FORMULA, data=frame, family="categorical", method="svi", num_steps=2)
    result.posterior_epred(new_data=pd.DataFrame({"x": [0.1, 0.8]}), num_samples=2)
    scale = next(
        t
        for t in result._formula_lens().forward(result.formula)[1].prior_targets
        if t.class_ == "sds"
    )
    assert scale.shape == (2,)


def test_categorical_smooth_joint_and_every_component_derivative_match_independent_logits():
    frame = _frame()
    frame["y"] = np.tile([0, 1, 2], 4)
    formula = formula_from_data(FORMULA, frame)
    lens = FormulaToQVRModule(families["categorical"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    term, column = formula.design_terms[0], formula.fixed_columns[1]
    name = term.qvr_name
    inputs = lens.fixed_column_observations(formula)
    inputs["y"] = torch.tensor(formula.response_values.copy(), dtype=torch.long)
    point = torch.tensor(
        [0.1, -0.2, 0.3, 0.15, 0.7, 0.8, 0.1, -0.2, 0.3, -0.1, 0.2, 0.4, -0.3, 0.2],
        dtype=torch.float64,
        requires_grad=True,
    )

    def actual(v):
        values = {
            **inputs,
            "intercept": v[:2],
            f"beta_{column.qvr_name}": v[2:4],
            name + "_scale": v[4:6],
            name + "_z": v[6:],
        }
        return trace(program, v.new_zeros(1, 1), values).log_joint.sum()

    def oracle(v):
        x = v.new_tensor(column.data.copy())
        z = v.new_tensor(term.data.copy())
        eta = v[:2] + x[:, None] * v[2:4] + z @ (v[6:].reshape(4, 2) * v[4:6])
        logits = torch.cat((v.new_zeros(12, 1), eta), dim=1)
        return (
            torch.distributions.Normal(0.0, 5.0).log_prob(v[:2]).sum()
            + torch.distributions.Normal(0.0, 5.0 / x.square().mean().sqrt())
            .log_prob(v[2:4])
            .sum()
            + torch.distributions.HalfNormal(1.0).log_prob(v[4:6]).sum()
            + torch.distributions.Normal(0.0, 1.0).log_prob(v[6:]).sum()
            + torch.distributions.Categorical(logits=logits).log_prob(inputs["y"]).sum()
        )

    torch.testing.assert_close(actual(point), oracle(point))
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(actual, point),
        torch.autograd.functional.jacobian(oracle, point),
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(actual, point),
        torch.autograd.functional.hessian(oracle, point),
    )


@pytest.mark.parametrize("order", [1, 3])
def test_penalty_order_changes_nullity_without_changing_retained_newdata_width(order):
    formula = f"y~s(x,bs='ps',k=7,m={order})"
    model = fit(formula, data=_frame(), family="gaussian", method="svi", num_steps=1)
    assert len(model.formula.fixed_columns) == order
    assert model.formula.design_terms[0].data.shape[1] == 7 - order
    assert model.posterior_epred(
        new_data=pd.DataFrame({"x": [0.1, 0.2]}), num_samples=1
    ).shape == (1, 1, 2)


def test_spline_structural_state_and_artifact_replay_retain_every_coordinate(
    tmp_path, monkeypatch
):
    from scipy.interpolate import BSpline
    from quivers.modeling import save_fit, load_fit
    import quivers.formulas._fit as api

    fitted = fit(
        FORMULA, data=_frame(), family="gaussian", method="svi", num_steps=1, seed=7
    )
    term = fitted.formula.design_terms[0]
    assert term.basis is not None
    raw = BSpline(term.basis.knots, np.eye(6), term.basis.degree)(
        _frame()["x"].to_numpy()
    )
    mapped = raw @ np.asarray(term.coefficient_map)
    np.testing.assert_allclose(mapped[:, term.column_indices], term.data, atol=1e-13)
    np.testing.assert_allclose(
        np.asarray(term.constraint) @ np.asarray(term.coefficient_map), 0, atol=1e-13
    )
    fresh = pd.DataFrame({"x": [0.2, 1.1]})
    expected = fitted.posterior_epred(new_data=fresh, num_samples=2, seed=9)
    path = tmp_path / "smooth.qfit"
    save_fit(fitted, path)

    def forbid(*args, **kwargs):
        raise AssertionError("loading a spline fit must not run inference")

    monkeypatch.setattr(api, "_run_inference", forbid)
    restored = load_fit(path)
    assert restored.formula.design_terms == fitted.formula.design_terms
    torch.testing.assert_close(
        restored.posterior_epred(new_data=fresh, num_samples=2, seed=9), expected
    )
