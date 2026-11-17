"""mgcv smooth terms fit, predict, refit, and restore through the shared path."""

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.dsl import Compiler
from quivers.formulas import (
    FormulaToQVRModule,
    families,
    fit,
    formula_from_data,
    get_prior,
)
from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.inference.trace import trace
from quivers.modeling import load_fit, refit_callback, save_fit

FORMULA = "y ~ s(x) + t2(x, z, k=4) + s(z, by=g, bs='cr', k=5) + g"


def _frame(rows: int = 40) -> pd.DataFrame:
    index = np.arange(1, rows + 1)
    frame = pd.DataFrame(
        {
            "x": (index * 0.618034) % 1,
            "z": (index * 0.414214) % 1,
            "g": np.array(["a", "b"])[index % 2],
        }
    )
    frame["y"] = np.sin(3 * frame["x"]) + frame["z"] ** 2
    return frame


def _fresh() -> pd.DataFrame:
    return pd.DataFrame(
        {"x": [0.1, 0.5, 1.2], "z": [0.3, 0.9, -0.1], "g": ["b", "a", "a"]}
    )


def test_shared_id_scale_joint_and_derivatives_match_independent_oracle():
    frame = _frame()
    formula = formula_from_data("y ~ s(x, k=5, id=1) + s(z, k=5, id=1)", frame)
    terms = formula.design_terms
    [shared] = {t.scale_group for t in terms}
    assert shared and len(terms) == 2
    lens = FormulaToQVRModule(families["gaussian"])
    module, _ = lens.forward(formula)
    program = Compiler(module).compile().morphism
    observed = lens.fixed_column_observations(formula)
    observed["y"] = torch.tensor(formula.response_values.copy(), dtype=torch.float64)
    columns = [c for c in formula.fixed_columns if not c.is_intercept]
    widths = [t.data.shape[1] for t in terms]
    size = 1 + len(columns) + 1 + sum(widths) + 1
    point = torch.linspace(-0.4, 0.6, size, dtype=torch.float64).requires_grad_()
    point.data[1 + len(columns)] = 0.7
    point.data[-1] = 0.9

    def split(v):
        beta = v[1 : 1 + len(columns)]
        scale = v[1 + len(columns)]
        start = 2 + len(columns)
        z = []
        for width in widths:
            z.append(v[start : start + width])
            start += width
        return v[0], beta, scale, z, v[-1]

    def actual(v):
        intercept, beta, scale, z, sigma = split(v)
        values = {**observed, "intercept": intercept, "sigma": sigma}
        values[shared + "_scale"] = scale
        for column, b in zip(columns, beta, strict=True):
            values[f"beta_{column.qvr_name}"] = b
        for term, raw in zip(terms, z, strict=True):
            values[term.qvr_name + "_z"] = raw
        return trace(program, v.new_zeros(1, 1), values).log_joint.sum()

    def oracle(v):
        intercept, beta, scale, z, sigma = split(v)
        mu = intercept.expand(len(frame))
        density = torch.distributions.Normal(0.0, 5.0).log_prob(intercept)
        for column, b in zip(columns, beta, strict=True):
            x = v.new_tensor(column.data.copy())
            mu = mu + b * x
            density = density + torch.distributions.Normal(
                0.0, 5.0 / x.square().mean().sqrt()
            ).log_prob(b)
        for term, raw in zip(terms, z, strict=True):
            mu = mu + v.new_tensor(term.data.copy()) @ (scale * raw)
            density = density + torch.distributions.Normal(0.0, 1.0).log_prob(raw).sum()
        return (
            density
            + torch.distributions.HalfNormal(1.0).log_prob(scale)
            + torch.distributions.HalfCauchy(2.0).log_prob(sigma)
            + torch.distributions.Normal(mu, sigma).log_prob(observed["y"]).sum()
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


def test_prior_catalogue_lists_one_sds_per_block_and_one_per_shared_id():
    frame = _frame()
    priors = get_prior(FORMULA, data=frame)
    scales = [site for site in priors if site.endswith("_scale")]
    # s(x): one block; t2 with two margins: three blocks; by=g: one per level.
    assert len(scales) == 1 + 3 + 2
    shared = get_prior("y ~ s(x, k=5, id=2) + s(z, k=5, id=2)", data=frame)
    assert len([site for site in shared if site.endswith("_scale")]) == 1


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_every_inference_method_predicts_fresh_rows_through_retained_bases(method):
    options = (
        {"num_steps": 3}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 2, "num_chains": 1}
    )
    if method == "nuts":
        # The checks below concern shapes, finiteness and replay consistency,
        # not posterior accuracy, so each transition is capped at seven
        # leapfrog steps.
        options["control"] = {"max_treedepth": 3}
    fitted = fit(
        FORMULA, data=_frame(), family="gaussian", method=method, seed=2, **options
    )
    fresh = _fresh()
    for draws in (
        fitted.posterior_linpred(new_data=fresh, num_samples=2, seed=5),
        fitted.posterior_epred(new_data=fresh, num_samples=2, seed=5),
        fitted.posterior_predict(new_data=fresh, num_samples=2, seed=5),
        *fitted.log_likelihood(
            new_data=fresh.assign(y=[0.1, 0.4, 0.2]), num_samples=2, seed=5
        ).values(),
    ):
        assert draws.shape[-1] == 3 and torch.isfinite(draws).all()
    with pytest.raises(ValueError, match="unknown levels|unseen levels"):
        fitted.posterior_epred(new_data=fresh.assign(g=["c", "a", "a"]), num_samples=1)


def test_training_rows_as_new_data_reproduce_the_training_predictor():
    frame = _frame()
    fitted = fit(FORMULA, data=frame, method="svi", num_steps=2, seed=3)
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=frame, num_samples=3, seed=4),
        fitted.posterior_linpred(num_samples=3, seed=4),
    )


def test_update_kfold_and_artifact_round_trip(tmp_path, monkeypatch):
    import quivers.formulas._fit as api

    frame = _frame()
    fitted = fit(FORMULA, data=frame, method="svi", num_steps=1, seed=7)
    refit = fitted.update(data=frame.iloc[:30].reset_index(drop=True))
    assert refit.formula.response_values.shape == (30,)
    callback = refit_callback(
        fitted.refit_recipe,
        lambda data, rows: data.iloc[list(rows)].copy(),
        lambda refitted, held, seed: refitted.log_likelihood(
            new_data=held, num_samples=2, seed=seed
        ),
        method_factory=lambda settings, seed: {**settings, "seed": seed},
    )
    result = cross_validate(
        FoldPlan(assignments=tuple((i // 2) % 2 for i in range(40)), seed=1), callback
    )
    assert result.complete and result.log_predictive_density.shape == (40,)
    fresh = _fresh()
    expected = fitted.posterior_epred(new_data=fresh, num_samples=2, seed=9)
    path = tmp_path / "smooth.qfit"
    save_fit(fitted, path)

    def forbid(*args, **kwargs):
        raise AssertionError("restoring a smooth fit must not run inference")

    monkeypatch.setattr(api, "_run_inference", forbid)
    restored = load_fit(path)
    assert restored.formula.design_terms == fitted.formula.design_terms
    torch.testing.assert_close(
        restored.posterior_epred(new_data=fresh, num_samples=2, seed=9), expected
    )


def test_conditional_smooths_equal_hand_assembled_coefficient_contractions():
    from quivers.formulas import conditional_smooths

    frame = _frame()
    fitted = fit(FORMULA, data=frame, method="svi", num_steps=2, seed=8)
    fresh = _fresh()
    term = "t2(x, z, k=4)"
    effect = conditional_smooths(fitted, term, fresh, num_samples=3, seed=2)
    draws = fitted.draws(num_samples=3, seed=2)
    design = fitted.prediction_plan.designs["mu"].common.terms[term]
    matrix = torch.as_tensor(np.asarray(design.eval_new_data(fresh)))
    expected = torch.zeros(effect.shape, dtype=torch.float64)
    fixed = [c for c in fitted.formula.fixed_columns if c.term == term]
    for position, column in enumerate(fixed):
        expected += (
            draws["beta_" + column.qvr_name].double()[..., None] * matrix[:, position]
        )
    blocks = [b for b in fitted.formula.design_terms if b.term == term]
    assert len(blocks) == 3
    for block in blocks:
        coefficients = (
            draws[block.qvr_name + "_scale"].double()[..., None]
            * draws[block.qvr_name + "_z"].double()
        )
        expected += torch.einsum(
            "cdk,rk->cdr", coefficients, matrix[:, list(block.column_indices)]
        )
    torch.testing.assert_close(effect, expected)
    with pytest.raises(ValueError, match="not a fitted s"):
        conditional_smooths(fitted, "g", fresh)


def test_brms_half_student_t_sds_default_is_available_as_a_typed_override():
    from quivers.formulas import PriorSpec

    frame = _frame()
    priors = get_prior(
        "y ~ s(x)",
        data=frame,
        priors=PriorSpec(distribution="HalfStudentT(3.0, 2.5)", class_="sds"),
    )
    [scale] = [value for site, value in priors.items() if site.endswith("_scale")]
    assert "HalfStudentT" in scale and "3.0" in scale and "2.5" in scale
