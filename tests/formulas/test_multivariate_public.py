"""Public joint (multivariate) formulas: brms surface, densities, prediction, refits."""

from __future__ import annotations

import json
import math
from pathlib import Path

import didactic.api as dx
import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.diagnostics import prediction_grid
from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.dsl import Compiler
from quivers.formulas import (
    MultivariateFit,
    MultivariateFormula,
    PriorSpec,
    ResponseModel,
    bf,
    families,
    fit,
    formula_to_qvr,
    get_prior,
    mvbf,
    set_rescor,
)
from quivers.formulas import _joint_fit
from quivers.inference import trace
from quivers.inference.prior import PriorDraws
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_multivariate.json").read_text()
)


def _frame(rows: int = 12) -> pd.DataFrame:
    generator = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "y_1": generator.normal(size=rows),
            "y.2": generator.normal(size=rows),
            "k": generator.integers(0, 2, size=rows),
            "c": generator.integers(0, 4, size=rows),
            "x": generator.normal(size=rows),
            "z": generator.normal(size=rows),
            "g": ["a", "b", "c"] * (rows // 3),
        }
    )


def _fixed_posterior(fitted: MultivariateFit, values: dict, draws: int):
    return fitted.with_(
        posterior=PriorDraws(
            samples={
                name: torch.as_tensor(value, dtype=torch.float32)
                .expand(1, draws, *torch.as_tensor(value).shape)
                .clone()
                for name, value in values.items()
            },
            num_samples=draws,
        ),
        response_fits={},
    )


# Surface: brms construction, response names, and the residual-correlation default.


def test_mvbind_repeats_one_right_hand_side_under_brms_response_names():
    spec = bf("mvbind(y_1, y.2) ~ x + (1|p|g)", "sigma ~ z")
    assert isinstance(spec, MultivariateFormula)
    assert spec.response_names == ("y1", "y2")
    assert spec.responses["y2"].formula == "y.2 ~ x + (1|p|g)"
    assert spec.responses["y1"].distributional == {"sigma": "sigma ~ z"}
    # brms names responses by removing every non-alphanumeric character.
    assert [f"N_{name}" for name in spec.response_names] == ORACLE["responses"]


def test_brms_default_and_explicit_residual_correlation_settings():
    gaussian = bf("y_1 ~ x") + bf("y.2 ~ x")
    student = bf("y_1 ~ x", family="student_t") + bf("y.2 ~ x", family="student_t")
    mixed = bf("y_1 ~ x") + bf("k ~ x", family="bernoulli")
    assert gaussian.rescor is ORACLE["default_rescor_gaussian"] is True
    assert student.rescor is ORACLE["default_rescor_student"] is True
    assert mixed.rescor is ORACLE["default_rescor_mixed"] is False
    assert (gaussian + set_rescor(False)).rescor is False
    assert mvbf(bf("y_1 ~ x"), bf("y.2 ~ x"), rescor=False).rescor is False
    assert (bf("y_1 ~ mi(z)") + bf("y.2 ~ x")).rescor is True
    assert (bf("y_1 | mi() ~ x") + bf("y.2 ~ x")).rescor is False


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (lambda: mixed_rescor(), "homogeneous Gaussian or Student"),
        (lambda: bf("y ~ x") + bf("y ~ z"), "appears twice"),
        (lambda: bf("y ~ x") + set_rescor(True), "at least two responses"),
        (lambda: bf("mvbind(y) ~ x"), "at least two response"),
        (lambda: bf("y ~ x", "a ~ 1"), "nonlinear parameters require nl=True"),
        (
            lambda: MultivariateFormula(
                responses={
                    "y_1": ResponseModel(formula="y ~ 1"),
                    "z": ResponseModel(formula="z ~ 1"),
                }
            ),
            "only letters and digits",
        ),
    ],
)
def test_invalid_joint_specifications_are_refused(build, message):
    with pytest.raises((ValueError, dx.ValidationError), match=message):
        build()


def mixed_rescor():
    # The families of a joint formula are checked when it is compiled.
    return formula_to_qvr(
        bf("y_1 ~ x") + bf("k ~ x", family="bernoulli") + set_rescor(True),
        data=_frame(),
    )


def test_fit_refuses_raw_mvbind_strings_and_response_options():
    data = _frame()
    with pytest.raises(ValueError, match="pass bf"):
        fit("mvbind(y_1, z) ~ x", data=data, method="svi", num_steps=1)
    with pytest.raises(ValueError, match="family"):
        fit(
            bf("y_1 ~ x") + bf("z ~ x"),
            data=data,
            family="student_t",  # type: ignore[call-overload]
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match=r"mi\(x\).*requires a response"):
        fit("y_1 ~ mi(x)", data=data, method="svi", num_steps=1)


def test_rescor_refuses_censored_and_row_mismatched_events():
    data = _frame().assign(cens=0, keep=[True, False] * 6)
    with pytest.raises(ValueError, match="censoring and truncation"):
        fit(
            bf("y_1 | cens(cens) ~ x") + bf("z ~ x") + set_rescor(True),
            data=data,
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="identical original response rows"):
        fit(
            bf("y_1 | subset(keep) ~ x") + bf("z ~ x") + set_rescor(True),
            data=data,
            method="svi",
            num_steps=1,
        )


# brms naming and the joint inspection entry points.


def test_parameter_names_match_brms_variables():
    data = _frame()
    rescor = fit(
        bf("mvbind(y_1, y.2) ~ x + (1|p|g)") + set_rescor(True),
        data=data,
        method="svi",
        num_steps=1,
    )
    assert sorted(rescor.variables(num_samples=2)) == sorted(ORACLE["rescor_variables"])
    mixed = fit(
        bf("y_1 ~ x + (1|p|g)", "sigma ~ z + (1|p|g)")
        + bf("k ~ x + (1|p|g)", family="bernoulli")
        + set_rescor(False),
        data=data,
        method="svi",
        num_steps=1,
    )
    named = mixed.variables(num_samples=2)
    assert sorted(named) == sorted(ORACLE["mixed_variables"])
    effects = mixed.random_effects(resp="k", num_samples=2)["g"]["Intercept"]
    for index, level in enumerate(mixed.group_levels["g"]):
        torch.testing.assert_close(
            named[f"r_g__k[{level},Intercept]"], effects[:, :, index]
        )


def test_get_prior_and_source_use_one_joint_catalogue_with_response_selectors():
    data = _frame()
    spec = bf("mvbind(y_1, z) ~ x") + set_rescor(True)
    rule = PriorSpec(distribution="Normal(0.0, 0.5)", class_="b", response="y1")
    priors = get_prior(spec, data=data, priors=rule)
    assert priors["response_y1__beta_x"] == "Normal(loc=0.0, scale=0.5)"
    assert priors["response_z__beta_x"] != priors["response_y1__beta_x"]
    assert priors["rescor"] == "LKJCholesky(concentration=1.0)"
    source = formula_to_qvr(spec, data=data, priors=rule)
    assert "MultivariateNormal" in source and "response_y1__beta_x" in source
    with pytest.raises(ValueError, match="unused"):
        get_prior(
            spec,
            data=data,
            priors=PriorSpec(distribution="Normal(0,1)", response="missing"),
        )


# Independent density oracles.


def test_heterogeneous_shared_group_joint_density_and_gradients():
    torch.set_default_dtype(torch.float64)
    try:
        data = _frame()
        spec = (
            bf("y_1 ~ x + (1|p|g)", "sigma ~ z")
            + bf("k ~ x + (1|p|g)", family="bernoulli")
            + bf("c ~ 1 + (1|p|g)", family="poisson")
        )
        fitted = fit(spec, data=data, method="svi", num_steps=1, seed=2)
        priors = get_prior(spec, data=data)
        (layout,) = fitted.emission.group_layouts
        assert [member.prefix for member in layout.members] == [
            "response_y1__",
            "response_k__",
            "response_c__",
        ]
        factor = torch.tensor(
            [[1.0, 0.0, 0.0], [0.3, math.sqrt(0.91), 0.0], [-0.2, 0.4, math.sqrt(0.8)]],
            requires_grad=True,
        )
        scales = torch.tensor([0.6, 0.4, 0.9], requires_grad=True)
        raw = torch.tensor(
            [[0.2, -0.4, 0.1], [0.5, 0.1, -0.3], [-0.6, 0.2, 0.7]], requires_grad=True
        )
        coefficients = torch.tensor(
            [0.1, -0.3, 0.2, 0.5, -0.4, 0.3, 0.6], requires_grad=True
        )
        names = (
            "response_y1__intercept",
            "response_y1__beta_x",
            "response_y1__dpar_sigma_intercept",
            "response_y1__dpar_sigma_beta_z",
            "response_k__intercept",
            "response_k__beta_x",
            "response_c__intercept",
        )
        sites = {layout.correlation_site: factor}
        for index, member in enumerate(layout.members):
            sites[member.scale_site] = scales[index]
            sites[member.raw_site] = raw[:, index]
        sites.update({name: coefficients[i] for i, name in enumerate(names)})
        result = trace(
            fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
        )
        # Hand-written oracle: default priors read from get_prior, the
        # noncentered shared block, and the three response likelihoods.
        prior_scale = {
            name: float(priors[name].split("scale=")[1].rstrip(")")) for name in names
        }
        log_prior = sum(
            torch.distributions.Normal(0.0, prior_scale[name]).log_prob(coefficients[i])
            for i, name in enumerate(names)
        )
        log_prior = (
            log_prior
            + torch.distributions.LKJCholesky(3, torch.tensor(1.0)).log_prob(factor)
            + torch.distributions.HalfNormal(1.0).log_prob(scales).sum()
            + torch.distributions.Normal(0.0, 1.0).log_prob(raw).sum()
        )
        effects = (raw @ factor.T) * scales
        codes = torch.tensor(pd.Categorical(data.g).codes, dtype=torch.long)
        x = torch.tensor(data.x.to_numpy())
        z = torch.tensor(data.z.to_numpy())
        b = coefficients
        mu = b[0] + b[1] * x + effects[codes, 0]
        sigma = torch.exp(b[2] + b[3] * z)
        logit = b[4] + b[5] * x + effects[codes, 1]
        rate = torch.exp(b[6] + effects[codes, 2])
        y = torch.tensor(data.y_1.to_numpy())
        k = torch.tensor(data.k.to_numpy(), dtype=torch.float64)
        c = torch.tensor(data.c.to_numpy(), dtype=torch.float64)
        likelihood = (
            torch.distributions.Normal(mu, sigma).log_prob(y).sum()
            + torch.distributions.Bernoulli(logits=logit).log_prob(k).sum()
            + torch.distributions.Poisson(rate).log_prob(c).sum()
        )
        scipy_likelihood = (
            stats.norm.logpdf(y, mu.detach(), sigma.detach()).sum()
            + stats.bernoulli.logpmf(k, torch.sigmoid(logit).detach()).sum()
            + stats.poisson.logpmf(c, rate.detach()).sum()
        )
        assert float(likelihood) == pytest.approx(scipy_likelihood, rel=1e-12)
        oracle = log_prior + likelihood
        torch.testing.assert_close(result.log_joint.sum(), oracle)
        leaves = (factor, scales, raw, coefficients)
        for got, wanted in zip(
            torch.autograd.grad(result.log_joint.sum(), leaves, retain_graph=True),
            torch.autograd.grad(oracle, leaves),
            strict=True,
        ):
            torch.testing.assert_close(got, wanted)
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("family", ["gaussian", "student_t"])
def test_pointwise_units_follow_brms_log_lik(family):
    data = _frame()
    fitted = fit(
        bf("mvbind(y_1, y.2) ~ x", family=family) + set_rescor(True),
        data=data,
        method="svi",
        num_steps=1,
    )
    values = {
        "response_y1__intercept": 0.2,
        "response_y1__beta_x": -0.3,
        "response_y2__intercept": -0.1,
        "response_y2__beta_x": 0.4,
        "response_y1__sigma": 0.7,
        "response_y2__sigma": 1.3,
        "rescor": [[1.0, 0.0], [0.6, 0.8]],
    }
    if family == "student_t":
        values["nu"] = 5.0
    fitted = _fixed_posterior(fitted, values, draws=2)
    x = data.x.to_numpy()
    loc = np.stack((0.2 - 0.3 * x, -0.1 + 0.4 * x), axis=-1)
    shape = np.array([[0.49, 0.6 * 0.7 * 1.3], [0.6 * 0.7 * 1.3, 1.69]])
    observed = np.stack((data["y_1"], data["y.2"]), axis=-1)
    if family == "gaussian":
        joint = [
            stats.multivariate_normal(m, shape).logpdf(o) for m, o in zip(loc, observed)
        ]
        marginal = stats.norm.logpdf(observed[:, 0], loc[:, 0], 0.7)
    else:
        joint = [
            stats.multivariate_t(m, shape, df=5.0).logpdf(o)
            for m, o in zip(loc, observed)
        ]
        marginal = stats.t.logpdf(observed[:, 0], 5.0, loc[:, 0], 0.7)
    scores = fitted.log_likelihood()
    assert set(scores) == {"y1__y2"}
    assert scores["y1__y2"].shape == (1, 2, ORACLE["rescor_log_lik_dim"][1])
    torch.testing.assert_close(
        scores["y1__y2"][0, 0].double(), torch.tensor(joint), rtol=1e-5, atol=1e-5
    )
    selected = fitted.log_likelihood(resp="y1")["y1"]
    assert selected.shape[-1] == ORACLE["rescor_log_lik_resp_dim"][1]
    torch.testing.assert_close(
        selected[0, 1].double(), torch.tensor(marginal), rtol=1e-5, atol=1e-5
    )
    fresh = data.iloc[:4]
    torch.testing.assert_close(
        fitted.log_likelihood(new_data=fresh)["y1__y2"][0, 0].double(),
        torch.tensor(joint[:4]),
        rtol=1e-5,
        atol=1e-5,
    )
    predicted = fitted.posterior_predict(seed=3)
    assert predicted["y1"].shape == (1, 2, ORACLE["rescor_predict_dim"][1])


def test_uncorrelated_units_sum_responses_per_row_as_brms_does():
    data = _frame()
    fitted = fit(
        bf("y_1 ~ x") + bf("k ~ x", family="bernoulli"),
        data=data,
        method="svi",
        num_steps=1,
    )
    values = {
        "response_y1__intercept": 0.2,
        "response_y1__beta_x": -0.3,
        "response_y1__sigma": 0.7,
        "response_k__intercept": -0.4,
        "response_k__beta_x": 0.9,
    }
    fitted = _fixed_posterior(fitted, values, draws=1)
    x = data.x.to_numpy()
    gaussian = stats.norm.logpdf(data.y_1, 0.2 - 0.3 * x, 0.7)
    bernoulli = stats.bernoulli.logpmf(data.k, 1 / (1 + np.exp(0.4 - 0.9 * x)))
    torch.testing.assert_close(
        fitted.log_likelihood()["y1__k"][0, 0].double(),
        torch.tensor(gaussian + bernoulli),
        rtol=1e-5,
        atol=1e-6,
    )
    assert fitted.log_likelihood()["y1__k"].shape[-1] == ORACLE["mixed_log_lik_dim"][1]
    torch.testing.assert_close(
        fitted.log_likelihood(resp="k")["k"][0, 0].double(),
        torch.tensor(bernoulli),
        rtol=1e-5,
        atol=1e-6,
    )
    # A deliberately wrong unit (one response) differs from the joint unit.
    assert not torch.allclose(
        fitted.log_likelihood()["y1__k"], fitted.log_likelihood(resp="y1")["y1"]
    )


# New-data prediction and new grouping levels.


def test_new_rows_reuse_fitted_encodings_and_shared_group_effects(
    monkeypatch: pytest.MonkeyPatch,
):
    data = _frame()
    fitted = fit(
        bf("y_1 ~ x + (1|p|g)") + bf("z ~ x + (1|p|g)") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=2,
        seed=5,
    )
    assert isinstance(fitted, MultivariateFit)
    fresh = pd.DataFrame({"x": [0.5, -1.0, 2.0], "g": ["c", "a", "c"]})
    linear = fitted.posterior_linpred(new_data=fresh, num_samples=4, seed=1)
    draws = fitted.draws(num_samples=4, seed=1)

    def rebuild(*_args):
        raise AssertionError(
            "reading joint coefficients must not compile a response fit"
        )

    with monkeypatch.context() as coefficients:
        coefficients.setattr(MultivariateFit, "_response_fit", rebuild)
        for resp in ("y1", "z"):
            fixed = fitted.fixed_effects(resp=resp, num_samples=4, seed=1)
            effects = fitted.random_effects(resp=resp, num_samples=4, seed=1)["g"][
                "Intercept"
            ]
            namespace = fitted.emission.responses[resp].prefix
            assert torch.equal(fixed["Intercept"], draws[f"{namespace}intercept"])
            assert torch.equal(fixed["x"], draws[f"{namespace}beta_x"])
            combined = fitted.coefficients(resp=resp, num_samples=4, seed=1)["g"]
            assert torch.equal(
                combined["Intercept"], fixed["Intercept"][..., None] + effects
            )
            levels = fitted.group_levels["g"]
            codes = [levels.index(label) for label in fresh.g]
            expected = (
                fixed["Intercept"][..., None]
                + fixed["x"][..., None] * torch.tensor(fresh.x.to_numpy()).float()
                + effects[..., codes]
            )
            torch.testing.assert_close(linear[resp], expected)
    means = fitted.posterior_epred(resp="z", new_data=fresh, num_samples=4, seed=1)
    assert set(means) == {"z"}
    torch.testing.assert_close(means["z"], linear["z"])
    with pytest.raises(ValueError, match="allow_new_levels"):
        fitted.posterior_epred(new_data=fresh.assign(g=["new", "a", "c"]))
    with pytest.raises(ValueError, match="unknown responses"):
        fitted.posterior_epred(resp="w")
    with pytest.raises(ValueError, match="unfitted term"):
        fitted.posterior_epred(new_data=fresh, re_formula="~(1|h)")
    excluded = fitted.posterior_linpred(
        new_data=fresh.drop(columns="g"), re_formula="~0", num_samples=4, seed=1
    )
    fixed = fitted.fixed_effects(resp="y1", num_samples=4, seed=1)
    torch.testing.assert_close(
        excluded["y1"],
        fixed["Intercept"][..., None]
        + fixed["x"][..., None] * torch.tensor(fresh.x.to_numpy()).float(),
    )


def test_new_levels_draw_one_donor_jointly_for_every_response():
    data = _frame()
    fitted = fit(
        bf("y_1 ~ 1 + (1|p|g)") + bf("z ~ 1 + (1|p|g)") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=2,
        seed=5,
    )
    fresh = pd.DataFrame({"g": ["new"]})
    for policy in ("uncertainty", "old_levels"):
        linear = fitted.posterior_linpred(
            new_data=fresh,
            allow_new_levels=True,
            sample_new_levels=policy,
            num_samples=6,
            seed=2,
        )
        known = {
            resp: fitted.coefficients(resp=resp, num_samples=6, seed=2)["g"][
                "Intercept"
            ]
            for resp in ("y1", "z")
        }
        for draw in range(6):
            donors = [
                level
                for level in range(3)
                if torch.isclose(known["y1"][0, draw, level], linear["y1"][0, draw, 0])
                and torch.isclose(known["z"][0, draw, level], linear["z"][0, draw, 0])
            ]
            assert donors, (policy, draw)


def test_gaussian_new_levels_keep_the_cross_response_correlation():
    data = _frame()
    fitted = fit(
        bf("y_1 ~ 1 + (1|p|g)") + bf("z ~ 1 + (1|p|g)") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=1,
    )
    (layout,) = fitted.emission.group_layouts
    values = {
        "response_y1__intercept": 0.0,
        "response_z__intercept": 0.0,
        "response_y1__sigma": 1.0,
        "response_z__sigma": 1.0,
        layout.correlation_site: [[1.0, 0.0], [0.9, math.sqrt(0.19)]],
        layout.members[0].scale_site: 1.0,
        layout.members[1].scale_site: 2.0,
        layout.members[0].raw_site: [0.0, 0.0, 0.0],
        layout.members[1].raw_site: [0.0, 0.0, 0.0],
    }
    draws = 4000
    fitted = _fixed_posterior(fitted, values, draws)
    linear = fitted.posterior_linpred(
        new_data=pd.DataFrame({"g": ["new"]}),
        allow_new_levels=True,
        sample_new_levels="gaussian",
        seed=9,
    )
    sample = torch.stack((linear["y1"][0, :, 0], linear["z"][0, :, 0])).double()
    covariance = torch.cov(sample)
    # Five Monte Carlo standard errors of each covariance entry at n = 4000.
    expected = torch.tensor([[1.0, 1.8], [1.8, 4.0]], dtype=torch.float64)
    error = torch.sqrt(
        (expected.square() + torch.outer(expected.diag(), expected.diag())) / draws
    )
    assert torch.all((covariance - expected).abs() < 5 * error)


def test_joint_training_expected_responses_reuse_the_fitted_program(monkeypatch):
    data = _frame().assign(category=pd.Categorical(["red", "green", "blue"] * 4))
    fitted = fit(
        bf("y_1 ~ x") + bf("category ~ x", family="categorical") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=1,
    )

    def reject_compilation(*args, **kwargs):
        pytest.fail("training posterior_epred should reuse the fitted joint program")

    monkeypatch.setattr(Compiler, "compile", reject_compilation)
    expected = fitted.posterior_epred(num_samples=3, seed=12)
    native = fitted.draws(num_samples=3, seed=12)
    intercept = native["response_y1__intercept"].unsqueeze(-1)
    slope = native["response_y1__beta_x"].unsqueeze(-1)
    x = torch.as_tensor(data["x"].to_numpy(copy=True), dtype=intercept.dtype)
    torch.testing.assert_close(expected["y1"], intercept + slope * x)
    linear = fitted.posterior_linpred(resp="y1", num_samples=3, seed=12)["y1"]
    torch.testing.assert_close(linear, expected["y1"])
    probabilities = expected["category"]
    assert probabilities.shape == (1, 3, len(data), 3)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones_like(probabilities[..., 0])
    )


@pytest.mark.parametrize(
    "method",
    ["posterior_linpred", "posterior_epred", "posterior_predict", "log_likelihood"],
)
def test_fresh_joint_rows_compile_one_program(monkeypatch, method):
    data = _frame()
    fitted = fit(
        bf("y_1 ~ x + (1 | shared | g)")
        + bf("z ~ x + (1 | shared | g)")
        + set_rescor(False),
        data=data,
        method="svi",
        num_steps=1,
    )
    compile_model = Compiler.compile
    compilations = []

    def compile_once(compiler, *args, **kwargs):
        compilations.append(compiler)
        return compile_model(compiler, *args, **kwargs)

    monkeypatch.setattr(Compiler, "compile", compile_once)
    fresh = data.iloc[:3].assign(g=["a", "new", "b"])
    draws = getattr(fitted, method)(
        new_data=fresh, allow_new_levels=True, num_samples=2, seed=12
    )
    assert len(compilations) == 1
    assert all(value.shape == (1, 2, 3) for value in draws.values())


def test_correlated_predictions_keep_the_joint_residual_law_on_new_rows():
    data = _frame()
    fitted = fit(
        bf("mvbind(y_1, z) ~ 1") + set_rescor(True),
        data=data,
        method="svi",
        num_steps=1,
    )
    values = {
        "response_y1__intercept": 0.2,
        "response_z__intercept": -0.3,
        "response_y1__sigma": 0.7,
        "response_z__sigma": 1.2,
        "rescor": [[1.0, 0.0], [0.6, 0.8]],
    }
    draws = 3000
    fitted = _fixed_posterior(fitted, values, draws)
    fresh = pd.DataFrame({"x": [0.0]})
    predicted = fitted.posterior_predict(new_data=fresh, seed=11)
    sample = torch.stack((predicted["y1"][0, :, 0], predicted["z"][0, :, 0])).double()
    expected = torch.tensor([[0.49, 0.504], [0.504, 1.44]], dtype=torch.float64)
    error = torch.sqrt(
        (expected.square() + torch.outer(expected.diag(), expected.diag())) / draws
    )
    assert torch.all((torch.cov(sample) - expected).abs() < 5 * error)
    only = fitted.posterior_predict(resp="z", new_data=fresh, seed=11)
    torch.testing.assert_close(only["z"], predicted["z"])
    variables = fitted.variables()
    torch.testing.assert_close(variables["rescor__y1__z"], torch.full((1, draws), 0.6))


# Inference drivers, retained analyses, refits, and artifacts.


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("rescor", [False, True])
def test_joint_model_runs_every_inference_driver(method, rescor):
    data = _frame(6)
    budget = (
        {"num_steps": 2}
        if method == "svi"
        else {"num_warmup": 2, "num_samples": 3, "num_chains": 1}
    )
    state = torch.random.get_rng_state().clone()
    fitted = fit(
        bf("y_1 ~ x") + bf("z ~ 1") + set_rescor(rescor),
        data=data,
        method=method,
        seed=7,
        **budget,
    )
    assert torch.equal(torch.random.get_rng_state(), state)
    means = fitted.posterior_epred(num_samples=3, seed=5)
    assert all(value.shape == (1, 3, 6) for value in means.values())
    analysis = fitted.analysis(num_samples=3)
    assert set(analysis.log_likelihood) == {"y1__z"}


def test_retained_summaries_hypotheses_and_conditional_effects():
    data = _frame()
    fitted = fit(
        bf("y_1 ~ x") + bf("k ~ x", family="bernoulli"),
        data=data,
        method="svi",
        num_steps=2,
        sample_prior="yes",
        num_prior_samples=8,
    )
    named = fitted.variables(num_samples=8, seed=1)
    summary = fitted.hypothesis("b_y1_x - b_k_x", num_samples=8, seed=1)
    torch.testing.assert_close(
        torch.as_tensor(summary.estimate).reshape(()).float(),
        (named["b_y1_x"] - named["b_k_x"]).mean(),
    )
    with_prior = fitted.hypothesis("b_y1_x", prior=True, num_samples=8, seed=1)
    assert with_prior is not None
    grid = prediction_grid({"x": [-1.0, 0.0, 1.0]}, conditions={})
    effects = fitted.conditional_effects(grid, num_samples=4)
    assert set(effects) == {"y1", "k"}
    probability = effects["k"].draws
    assert torch.all((probability >= 0) & (probability <= 1))


def test_update_cross_validation_and_artifact_round_trip(tmp_path, monkeypatch):
    data = _frame()
    fitted = fit(
        bf("y_1 ~ x + (1|p|g)") + bf("k ~ x + (1|p|g)", family="bernoulli"),
        data=data,
        method="svi",
        num_steps=2,
        seed=4,
    )
    refit = fitted.update(num_steps=3)
    assert isinstance(refit, MultivariateFit) and len(refit.losses) == 3
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=3,
    )
    result = cross_validate(FoldPlan(assignments=(0, 1) * 6, seed=2), callback)
    assert result.responses == ("y1__k",)
    assert result.log_predictive_density.shape == (12,)
    assert torch.isfinite(result.log_predictive_density).all()
    single = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=3,
        resp="k",
    )
    split = FoldPlan(assignments=tuple(range(12)), seed=1).split(3)
    prediction = single(split)
    expected = prediction.fit.log_likelihood(
        resp="k", new_data=data.iloc[[3]], num_samples=3, seed=split.seed
    )
    torch.testing.assert_close(prediction.log_likelihood["k"], expected["k"])
    path = tmp_path / "joint.pt"
    save_fit(fitted, path)

    def refuse(*args, **kwargs):
        raise AssertionError("restoration must not run inference")

    monkeypatch.setattr(_joint_fit, "_run_inference", refuse)
    restored = load_fit(path)
    assert isinstance(restored, MultivariateFit)
    fresh = pd.DataFrame({"x": [0.3, 1.0], "g": ["b", "new"]})
    for name in ("posterior_epred", "posterior_linpred"):
        before = getattr(fitted, name)(
            new_data=fresh, allow_new_levels=True, num_samples=3, seed=6
        )
        after = getattr(restored, name)(
            new_data=fresh, allow_new_levels=True, num_samples=3, seed=6
        )
        for resp in before:
            torch.testing.assert_close(before[resp], after[resp])
    assert restored.qvr_source == fitted.qvr_source


# Every single-response family as a response of a heterogeneous joint model.

_ROWS = 8
_COUNTS = np.array([0, 1, 2, 0, 3, 1, 0, 2])
_UNIT = np.array([0.2, 0.4, 0.6, 0.3, 0.7, 0.5, 0.25, 0.8])
_POSITIVE = np.array([0.5, 1.2, 0.8, 2.0, 0.4, 1.5, 0.9, 1.1])
_ORDINAL = np.array([0, 1, 2, 1, 0, 2, 1, 2])


def _family_case(name: str):
    columns: dict[str, np.ndarray | list] = {}
    options: dict = {}
    formula = "r ~ x + (1|p|g)"
    real = {
        "gaussian",
        "student_t",
        "skew_normal",
        "exgaussian",
        "asym_laplace",
        "gen_extreme_value",
        "mixture",
    }
    positive = {
        "gamma",
        "lognormal",
        "weibull",
        "exponential",
        "frechet",
        "inverse_gaussian",
        "shifted_lognormal",
        "cox",
    }
    counts = {
        "poisson",
        "negative_binomial",
        "negbinomial2",
        "geometric",
        "com_poisson",
        "discrete_weibull",
        "hurdle_poisson",
        "hurdle_negbinomial",
        "zero_inflated_poisson",
        "zero_inflated_negbinomial",
    }
    if name in real:
        columns["r"] = _UNIT * 3 - 1
        if name == "student_t":
            # A finite response mean requires nu > 1 at every draw.
            options["fixed_parameters"] = {"nu": 5.0}
    elif name == "zero_inflated_asym_laplace":
        columns["r"] = np.where(_COUNTS == 0, 0.0, _UNIT * 3 - 1)
    elif name in positive:
        columns["r"] = _POSITIVE
    elif name in {"hurdle_gamma", "hurdle_lognormal"}:
        columns["r"] = np.where(_COUNTS == 0, 0.0, _POSITIVE)
    elif name == "von_mises":
        columns["r"] = _UNIT * 2 - 1
    elif name == "wiener":
        columns["r"] = _POSITIVE
        columns["ch"] = [0, 1, 1, 0, 1, 0, 1, 1]
        formula = "r | dec(ch) ~ x + (1|p|g)"
        options["fixed_parameters"] = {"bs": 1.2, "ndt": 0.1, "bias": 0.4}
    elif name == "beta":
        columns["r"] = _UNIT
    elif name == "xbeta":
        columns["r"] = np.where(_COUNTS == 0, 0.0, _UNIT)
        options["fixed_parameters"] = {"phi": 4.0, "kappa": 0.2}
    elif name == "zero_inflated_beta":
        columns["r"] = np.where(_COUNTS == 0, 0.0, _UNIT)
    elif name == "zero_one_inflated_beta":
        columns["r"] = np.where(_COUNTS == 0, 0.0, np.where(_COUNTS == 3, 1.0, _UNIT))
    elif name in counts:
        columns["r"] = _COUNTS
    elif name in {
        "binomial",
        "beta_binomial",
        "zero_inflated_binomial",
        "zero_inflated_beta_binomial",
    }:
        columns["r"] = _COUNTS
        columns["n"] = [4] * _ROWS
        options["binomial_trials"] = "n"
    elif name == "bernoulli":
        columns["r"] = (_COUNTS > 0).astype(int)
    elif name in {"cumulative", "sratio", "cratio", "acat", "categorical"}:
        columns["r"] = _ORDINAL
    elif name == "hurdle_cumulative":
        columns["r"] = _ORDINAL
        options["fixed_parameters"] = {"hu": 0.2}
    elif name in {"multinomial", "dirichlet_multinomial"}:
        columns["a"] = _COUNTS
        columns["b"] = 3 - np.minimum(_COUNTS, 3)
        columns["n"] = columns["a"] + columns["b"]
        options["binomial_trials"] = "n"
        formula = "cbind(a,b) ~ x + (1|p|g)"
    elif name in {"dirichlet", "dirichlet2", "logistic_normal"}:
        columns["a"] = _UNIT
        columns["b"] = 1 - _UNIT
        formula = "cbind(a,b) ~ x + (1|p|g)"
    else:
        raise KeyError(name)
    return formula, columns, options


@pytest.mark.parametrize("name", sorted(families))
def test_every_family_is_a_joint_response_with_new_data_prediction(name):
    generator = np.random.default_rng(1)
    formula, columns, options = _family_case(name)
    data = pd.DataFrame(
        {
            "yg": generator.normal(size=_ROWS),
            "x": generator.normal(size=_ROWS),
            "g": list("abababab"),
            **columns,
        }
    )
    spec = bf("yg ~ x + (1|p|g)") + bf(formula, family=name, **options)
    fitted = fit(spec, data=data, method="svi", num_steps=1, seed=1)
    (other,) = (resp for resp in fitted.response_names if resp != "yg")
    fresh = data.iloc[:3].assign(g=["a", "new", "b"])
    options = {"new_data": fresh, "allow_new_levels": True, "num_samples": 2}
    if name == "von_mises":
        # As for a single response, a circular mean needs an angular convention.
        with pytest.raises(TypeError, match="angular convention"):
            fitted.posterior_epred(mc_samples=16, **options)
        means = fitted.posterior_epred(resp="yg", mc_samples=16, **options)
        means[other] = fitted.posterior_predict(resp=other, **options)[other]
    else:
        means = fitted.posterior_epred(mc_samples=16, **options)
    draws = fitted.posterior_predict(**options)
    linear = fitted.posterior_linpred(**options)
    scores = fitted.log_likelihood(**options)
    for result in (means, draws, linear):
        assert set(result) == {"yg", other}
        assert all(value.shape[:3] == (1, 2, 3) for value in result.values())
    assert torch.isfinite(scores["yg__" + other]).all()
    assert torch.isfinite(fitted.log_likelihood(resp=other, num_samples=2)[other]).all()


# Closed gaps: retained cumulative thresholds, subset() refits, per-level names.


@pytest.mark.parametrize("thresholds_by", [None, "g"])
def test_category_specific_cumulative_response_predicts_fresh_rows_jointly(
    thresholds_by,
):
    data = pd.DataFrame(
        {
            "yg": [0.3, -0.2, 0.8, 0.1, -0.5, 0.4, 0.0, 0.9],
            "r": [0, 1, 2, 3, 1, 2, 0, 3],
            "x": [-0.4, 0.3, -1.1, 0.8, 0.2, 0.1, 0.5, -0.3],
            "z": [-2.0, -0.4, 0.7, 2.0, 1.0, -1.0, 0.3, 0.6],
            "g": list("abababab"),
        }
    )
    options = {"fixed_parameters": {"disc": 1.0}}
    if thresholds_by is not None:
        options["thresholds_by"] = thresholds_by
    spec = bf("yg ~ x") + bf("r ~ x + cs(z) + (1|g)", family="cumulative", **options)
    joint = fit(spec, data=data, method="svi", num_steps=3, seed=2)
    alone = fit(
        "r ~ x + cs(z) + (1|g)",
        data=data,
        family="cumulative",
        method="svi",
        num_steps=1,
        **options,
    )
    draws = joint.draws(num_samples=4, seed=3)
    prefix = "response_r__"
    local = {
        name.removeprefix(prefix): value
        for name, value in draws.items()
        if name.startswith(prefix)
    }
    alone = alone.with_(
        posterior=PriorDraws(samples=local, num_samples=4),
        prediction_plan=alone.prediction_plan,
    )
    fresh = data.iloc[:3].assign(z=[0.1, -0.2, 0.4])
    expected = alone.posterior_epred(new_data=fresh, num_samples=4, seed=3)
    actual = joint.posterior_epred(resp="r", new_data=fresh, num_samples=4, seed=3)
    torch.testing.assert_close(actual["r"], expected)
    excluded = joint.posterior_epred(resp="r", re_formula="~0", num_samples=4, seed=3)
    torch.testing.assert_close(
        excluded["r"], alone.posterior_epred(re_formula="~0", num_samples=4, seed=3)
    )


def test_subset_responses_refit_on_all_other_rows_and_score_included_rows():
    data = _frame().assign(keep=[True, False, True] * 4)
    fitted = fit(
        bf("y_1 | subset(keep) ~ x") + bf("z ~ x") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=2,
        seed=4,
    )
    with pytest.raises(ValueError, match="different subset"):
        fitted.refit_callback(method_factory=lambda settings, seed: settings)
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=3,
        resp="y1",
    )
    included = tuple(i for i, keep in enumerate(data.keep) if keep)
    plan = FoldPlan(assignments=(0, 1) * 4, seed=3)
    split = plan.split(1)
    prediction = callback(split)
    test = [included[i] for i in split.test]
    trained = prediction.fit.refit_recipe.data
    assert sorted(trained.index) == sorted(set(data.index) - set(test))
    expected = prediction.fit.log_likelihood(
        resp="y1", new_data=data.iloc[test], num_samples=3, seed=split.seed
    )
    torch.testing.assert_close(prediction.log_likelihood["y1"], expected["y1"])
    assert prediction.log_likelihood["y1"].shape[-1] == len(test)
    result = cross_validate(plan, callback)
    assert result.log_predictive_density.shape == (len(included),)


def _series_frame() -> pd.DataFrame:
    rows = np.arange(30)
    frame = pd.DataFrame(
        {"x": (rows * 0.618) % 1, "t": rows % 10, "s": np.repeat(np.arange(3), 10)}
    )
    return frame.assign(y=np.sin(3 * frame.x), z=np.cos(2 * frame.x))


@pytest.mark.parametrize("term", ["s(x)", "t2(x, t, k=4)"])
def test_smooth_terms_inside_a_joint_model_smoke(term):
    data = _series_frame()
    fitted = fit(
        bf(f"y ~ {term}") + bf("z ~ x") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=2,
        seed=1,
    )
    fresh = data.iloc[:5]
    means = fitted.posterior_epred(new_data=fresh, num_samples=2)
    assert means["y"].shape == (1, 2, 5)
    assert torch.isfinite(
        fitted.log_likelihood(new_data=fresh, num_samples=2)["y__z"]
    ).all()


def test_autocorrelation_term_inside_a_joint_model_smoke_on_training_rows():
    data = _series_frame()
    fitted = fit(
        bf("y ~ x + ar(time=t, gr=s, p=1)") + bf("z ~ x") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=2,
        seed=1,
    )
    assert fitted.posterior_epred(num_samples=2)["y"].shape == (1, 2, 30)
    assert torch.isfinite(fitted.log_likelihood(num_samples=2)["y__z"]).all()


def test_autocorrelation_term_inside_a_joint_model_scores_new_series_as_alone():
    data = _series_frame()
    formula = "y ~ x + ar(time=t, gr=s, p=1)"
    joint = fit(
        bf(formula) + bf("z ~ x") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=2,
        seed=1,
    )
    alone = fit(formula, data=data, method="svi", num_steps=1, seed=1)
    draws = joint.draws(num_samples=3, seed=4)
    prefix = "response_y__"
    local = {
        name.removeprefix(prefix): value
        for name, value in draws.items()
        if name.startswith(prefix)
    }
    alone = alone.with_(
        posterior=PriorDraws(samples=local, num_samples=3),
        prediction_plan=alone.prediction_plan,
    )
    # Rows 10 to 14 open a new five-row series whose history is its own y.
    fresh = data.iloc[10:15]
    options = {"new_data": fresh, "num_samples": 3, "seed": 4}
    torch.testing.assert_close(
        joint.posterior_epred(resp="y", **options)["y"],
        alone.posterior_epred(**options),
    )
    torch.testing.assert_close(
        joint.log_likelihood(resp="y", **options)["y"],
        alone.log_likelihood(**options)["y"],
    )
    assert joint.posterior_predict(**options)["y"].shape == (1, 3, 5)
