"""brms mi() responses and predictors in a joint formula program."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.diagnostics.cross_validation import FoldPlan
from quivers.formulas import (
    BayesianFit,
    MultivariateFit,
    PriorSpec,
    RegularizedHorseshoe,
    bf,
    default_prior,
    fit,
    formula_to_qvr,
    get_prior,
    set_mecor,
    set_rescor,
    validate_prior,
)
from quivers.inference import trace
from quivers.inference.prior import PriorDraws
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_missing.json").read_text()
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [0.1, 0.4, -0.2, 0.3, 0.5, 0.0, 0.7, -0.1],
            "x": [1.0, np.nan, 0.5, np.nan, 2.0, 1.5, np.nan, 0.2],
            "z": [0.2, 0.1, -0.3, 0.4, 0.0, 0.3, -0.2, 0.6],
            "s": [0.1, 0.2, 0.3, 0.1, 0.2, 0.2, 0.4, 0.3],
        }
    )


def _scale(call: str) -> float:
    return float(call.split("scale=")[1].rstrip(")"))


def _fixed[Fit: (BayesianFit, MultivariateFit)](
    fitted: Fit, values: dict, draws: int = 1
) -> Fit:
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
        **({"response_fits": {}} if isinstance(fitted, MultivariateFit) else {}),
    )


def test_mi_response_and_predictor_joint_density_matches_brms_and_autograd():
    torch.set_default_dtype(torch.float64)
    try:
        data = _frame()
        spec = bf("y ~ mi(x) + mi(x):z") + bf("x | mi() ~ z") + set_rescor(False)
        fitted = fit(spec, data=data, method="svi", num_steps=1)
        priors = get_prior(spec, data=data)
        observed = np.isfinite(data.x.to_numpy())
        missing = int((~observed).sum())
        assert missing == ORACLE["mi"]["Nmi_x"]
        assert fitted.draws(num_samples=1)["mi_x__missing"].shape == (1, 1, missing)
        b = torch.tensor([0.3, 0.8, -0.4, 0.1, 0.6], requires_grad=True)
        sigma = torch.tensor([0.7, 0.5], requires_grad=True)
        imputed = torch.tensor([0.4, -0.3, 1.1], requires_grad=True)
        sites = {
            "response_y__intercept": b[0],
            "response_y__beta_x": b[1],
            "response_y__beta_x_z": b[2],
            "response_x__intercept": b[3],
            "response_x__beta_z": b[4],
            "response_y__sigma": sigma[0],
            "response_x__sigma": sigma[1],
            "mi_x__missing": imputed,
        }
        result = trace(
            fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
        )
        # brms: Yl_x = Y_x with Yl_x[Jmi_x] = Ymi_x, and normal_lpdf over Yl_x.
        z = torch.tensor(data.z.to_numpy())
        yl = torch.tensor(np.nan_to_num(data.x.to_numpy()))
        yl = yl.masked_scatter(torch.tensor(~observed), imputed)
        mu_x = b[3] + b[4] * z
        mu_y = b[0] + b[1] * yl + b[2] * yl * z
        y = torch.tensor(data.y.to_numpy())
        likelihood = (
            torch.distributions.Normal(mu_x, sigma[1]).log_prob(yl).sum()
            + torch.distributions.Normal(mu_y, sigma[0]).log_prob(y).sum()
        )
        assert float(likelihood) == pytest.approx(
            stats.norm.logpdf(yl.detach(), mu_x.detach(), float(sigma[1])).sum()
            + stats.norm.logpdf(y, mu_y.detach(), float(sigma[0])).sum(),
            rel=1e-12,
        )
        names = list(sites)[:5]
        prior = (
            sum(
                torch.distributions.Normal(0.0, _scale(priors[name])).log_prob(b[i])
                for i, name in enumerate(names)
            )
            + torch.distributions.HalfCauchy(2.0).log_prob(sigma).sum()
        )
        oracle = prior + likelihood
        torch.testing.assert_close(result.log_joint.sum(), oracle)
        leaves = (b, sigma, imputed)
        for got, wanted in zip(
            torch.autograd.grad(result.log_joint.sum(), leaves, retain_graph=True),
            torch.autograd.grad(oracle, leaves),
            strict=True,
        ):
            torch.testing.assert_close(got, wanted)
    finally:
        torch.set_default_dtype(torch.float32)


def test_mi_measurement_error_response_uses_latent_truths_for_every_row():
    torch.set_default_dtype(torch.float64)
    try:
        data = _frame()
        spec = bf("y ~ mi(x)") + bf("x | mi(s) ~ z") + set_rescor(False)
        fitted = fit(spec, data=data, method="svi", num_steps=1)
        priors = get_prior(spec, data=data)
        rows = len(data)
        assert ORACLE["me_response"]["Yl"] == "vector[N] Yl;  // latent variable"
        truth = torch.linspace(-0.5, 1.0, rows).requires_grad_(True)
        b = torch.tensor([0.3, 0.8, 0.1, 0.6], requires_grad=True)
        sigma = torch.tensor([0.7, 0.5], requires_grad=True)
        sites = {
            "response_y__intercept": b[0],
            "response_y__beta_x": b[1],
            "response_x__intercept": b[2],
            "response_x__beta_z": b[3],
            "response_y__sigma": sigma[0],
            "response_x__sigma": sigma[1],
            "mi_x__truth": truth,
        }
        result = trace(
            fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
        )
        observed = torch.tensor(np.isfinite(data.x.to_numpy()))
        x = torch.tensor(np.nan_to_num(data.x.to_numpy()))
        s = torch.tensor(data.s.to_numpy())
        z = torch.tensor(data.z.to_numpy())
        y = torch.tensor(data.y.to_numpy())
        oracle = (
            sum(
                torch.distributions.Normal(0.0, _scale(priors[name])).log_prob(b[i])
                for i, name in enumerate(list(sites)[:4])
            )
            + torch.distributions.HalfCauchy(2.0).log_prob(sigma).sum()
            # brms: target += normal_lpdf(Y[Jme] | Yl[Jme], noise[Jme]).
            + torch.distributions.Normal(truth[observed], s[observed])
            .log_prob(x[observed])
            .sum()
            + torch.distributions.Normal(b[2] + b[3] * z, sigma[1])
            .log_prob(truth)
            .sum()
            + torch.distributions.Normal(b[0] + b[1] * truth, sigma[0])
            .log_prob(y)
            .sum()
        )
        torch.testing.assert_close(result.log_joint.sum(), oracle)
        leaves = (truth, b, sigma)
        for got, wanted in zip(
            torch.autograd.grad(result.log_joint.sum(), leaves, retain_graph=True),
            torch.autograd.grad(oracle, leaves),
            strict=True,
        ):
            torch.testing.assert_close(got, wanted)
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        (
            lambda: bf("y ~ mi(k)") + bf("k | mi() ~ z", family="bernoulli"),
            "not supported for family",
        ),
        (lambda: bf("y ~ mi(x)") + bf("z ~ 1"), "requires a response"),
        (lambda: bf("y ~ I(mi(x)**2)") + bf("x | mi() ~ z"), "nonlinearly"),
    ],
)
def test_mi_refusals_follow_brms(spec, message):
    data = _frame().assign(k=[1.0, np.nan, 0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match=message):
        fit(spec(), data=data, method="svi", num_steps=1)


def test_mi_likelihood_units_prediction_and_index_joins():
    data = _frame()
    fitted = fit(
        bf("y ~ mi(x)") + bf("x | mi() ~ z") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=2,
    )
    values = {
        "response_y__intercept": 0.3,
        "response_y__beta_x": 0.8,
        "response_x__intercept": 0.1,
        "response_x__beta_z": 0.6,
        "response_y__sigma": 0.7,
        "response_x__sigma": 0.5,
        "mi_x__missing": [0.4, -0.3, 1.1],
    }
    fitted = _fixed(fitted, values)
    missing = ~np.isfinite(data.x.to_numpy())
    scores = fitted.log_likelihood(resp="x")["x"][0, 0]
    # A missing mi() row has no recorded likelihood (brms log_lik gives NA).
    assert bool((scores[missing] == 0).all())
    expected = stats.norm.logpdf(data.x[~missing], 0.1 + 0.6 * data.z[~missing], 0.5)
    torch.testing.assert_close(
        scores[~missing].double(), torch.tensor(expected), rtol=1e-5, atol=1e-6
    )
    completed = data.x.to_numpy().copy()
    completed[missing] = [0.4, -0.3, 1.1]
    means = fitted.posterior_epred(resp="y")["y"][0, 0]
    torch.testing.assert_close(
        means.double(), torch.tensor(0.3 + 0.8 * completed), rtol=1e-5, atol=1e-6
    )
    fresh = pd.DataFrame({"x": [0.3, 1.2], "z": [0.1, -0.4]})
    torch.testing.assert_close(
        fitted.posterior_epred(new_data=fresh)["y"][0, 0].double(),
        torch.tensor(0.3 + 0.8 * fresh.x.to_numpy()),
        rtol=1e-5,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="brms returns missing predictions"):
        fitted.posterior_epred(new_data=fresh.assign(x=[np.nan, 1.0]))
    sparse = pd.DataFrame(
        {
            "y": [0.1, 0.4, -0.2, 0.3, 0.5, 0.0],
            "idx": [1, 2, 1, 3, 4, 2],
            "x": [1.0, np.nan, 0.5, 2.0, np.nan, np.nan],
            "id": [1, 2, 3, 4, 0, 0],
            "sub": [True, True, True, True, False, False],
            "z": [0.2, 0.1, -0.3, 0.4, 0.0, 0.3],
        }
    )
    indexed = fit(
        bf("y ~ mi(x, idx)")
        + bf("x | mi() + index(id) + subset(sub) ~ z")
        + set_rescor(False),
        data=sparse,
        method="svi",
        num_steps=1,
    )
    assert indexed.missing_specs["x"].latent_positions == (1,)
    torch.testing.assert_close(
        indexed.observations["mi_y__x__source_0"], torch.tensor([0, 1, 0, 2, 3, 1])
    )


def test_mi_source_refit_and_artifact(tmp_path):
    data = _frame()
    fitted = fit(
        bf("y ~ mi(x)") + bf("x | mi() ~ z") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=2,
        seed=3,
    )
    assert "mi_x__missing" in formula_to_qvr(
        bf("y ~ mi(x)") + bf("x | mi() ~ z") + set_rescor(False), data=data
    )
    assert len(fitted.update(num_steps=3).losses) == 3
    path = tmp_path / "mi.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    assert isinstance(restored, MultivariateFit)
    for resp in ("y", "x"):
        torch.testing.assert_close(
            restored.posterior_epred(resp=resp, num_samples=3, seed=2)[resp],
            fitted.posterior_epred(resp=resp, num_samples=3, seed=2)[resp],
        )
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=2,
        resp="y",
    )
    # Held-out units are the rows whose mi(x) values are recorded.
    split = FoldPlan(assignments=(0, 1, 0, 1, 0), seed=1).split(0)
    prediction = callback(split)
    recorded = [row for row, value in enumerate(data.x) if np.isfinite(value)]
    test = [recorded[i] for i in split.test]
    expected = prediction.fit.log_likelihood(
        resp="y", new_data=data.iloc[test], num_samples=2, seed=split.seed
    )
    torch.testing.assert_close(prediction.log_likelihood["y"], expected["y"])


@pytest.mark.parametrize("known_error", [False, True])
def test_bare_mi_uses_single_response_names_and_recorded_scores(known_error, tmp_path):
    data = _frame()
    formula = f"x | mi({'s' if known_error else ''}) ~ z"
    fitted = fit(formula, data=data, method="svi", num_steps=1, seed=3)
    assert isinstance(fitted, BayesianFit)
    assert fitted.formula.formula == formula
    assert fitted.included_rows == tuple(range(len(data)))
    assert fitted.qvr_source == formula_to_qvr(formula, data=data)
    assert set(get_prior(formula, data=data)) == {"intercept", "beta_z", "sigma"}

    actual = torch.tensor(data.x.to_numpy())
    recorded = torch.isfinite(actual)
    z = torch.tensor(data.z.to_numpy())
    mean = 0.2 - 0.4 * z
    truth = torch.linspace(-0.5, 0.5, len(data), dtype=torch.float64)
    latent = (
        {"mi_x__truth": truth} if known_error else {"mi_x__missing": truth[~recorded]}
    )
    view = _fixed(fitted, {"intercept": 0.2, "beta_z": -0.4, "sigma": 0.7, **latent})
    assert set(view.draws()) == {"intercept", "beta_z", "sigma", *latent}
    expected = torch.zeros_like(actual)
    loc = truth[recorded] if known_error else mean[recorded]
    scale = torch.tensor(data.s.to_numpy())[recorded] if known_error else 0.7
    expected[recorded] = torch.distributions.Normal(loc, scale).log_prob(
        actual[recorded]
    )
    torch.testing.assert_close(view.posterior_linpred()[0, 0].double(), mean)
    torch.testing.assert_close(view.posterior_epred()[0, 0].double(), mean)
    torch.testing.assert_close(view.log_likelihood()["x"][0, 0].double(), expected)
    torch.testing.assert_close(
        view.log_likelihood(re_formula="~0")["x"][0, 0].double(), expected
    )
    analysis = view.analysis()
    torch.testing.assert_close(
        analysis.observed_data["x"].double(), actual, equal_nan=True
    )
    torch.testing.assert_close(analysis.log_likelihood["x"][0, 0].double(), expected)

    # An outcome solely on the left side is unnecessary for predicting it.
    fresh = data[["z"]].iloc[[1, 4, 0]]
    torch.testing.assert_close(
        view.posterior_epred(new_data=fresh)[0, 0].double(), mean[[1, 4, 0]]
    )
    assert view.posterior_predict(new_data=fresh).shape == (1, 1, len(fresh))
    assert view.posterior_predict().shape == (1, 1, len(data))

    if not known_error:
        shrinkage = PriorSpec(distribution=RegularizedHorseshoe(), class_="b")
        ordinary = get_prior("x ~ z", data=data.dropna(subset=["x"]), priors=shrinkage)
        assert "prior_0_global" in ordinary
        assert get_prior(formula, data=data, priors=shrinkage).keys() == ordinary.keys()
        assert len(fitted.update(num_steps=2).losses) == 2
        path = tmp_path / "bare-mi.pt"
        save_fit(fitted, path)
        restored = load_fit(path)
        assert isinstance(restored, BayesianFit)
        assert restored.qvr_source == fitted.qvr_source
        for new_data in (None, fresh):
            torch.testing.assert_close(
                restored.posterior_epred(new_data=new_data, num_samples=2, seed=4),
                fitted.posterior_epred(new_data=new_data, num_samples=2, seed=4),
            )


def test_bare_me_replays_true_values_and_fresh_measurements_with_ordinary_names():
    data = _frame().assign(x=np.linspace(0.1, 1.0, 8))
    formula = "y ~ me(x, s) + me(x, s):z"
    rule = {"me_y__x__mean": "Normal(0.0, 2.0)"}
    fitted = fit(formula, data=data, priors=rule, method="svi", num_steps=1)
    authored = fit(bf(formula), data=data, method="svi", num_steps=1)
    assert isinstance(fitted, BayesianFit)
    assert isinstance(authored, MultivariateFit)
    assert fitted.qvr_source == formula_to_qvr(formula, data=data, priors=rule)
    assert "beta_x" in get_prior(formula, data=data)
    assert "response_y__beta_x" in get_prior(bf(formula), data=data)
    assert default_prior(formula, data=data).row("me_y__x__mean").source == "default"
    priors = validate_prior(rule, formula, data=data)
    assert priors.row("me_y__x__mean").source == "user"

    values = {"intercept": 0.3, "beta_x": 0.8, "beta_x_z": -0.4, "sigma": 0.7}
    noise = torch.linspace(-1.0, 1.0, len(data))
    measurement = {"me_y__x__mean": 0.2, "me_y__x__sd": 0.9, "me_y__x__z": noise}
    view = _fixed(fitted, values | measurement)
    joint = _fixed(
        authored, {f"response_y__{k}": v for k, v in values.items()} | measurement
    )
    z = torch.tensor(data.z.to_numpy())
    expected = 0.3 + (0.8 - 0.4 * z) * (0.2 + 0.9 * noise)
    torch.testing.assert_close(view.posterior_epred()[0, 0].double(), expected)
    torch.testing.assert_close(view.posterior_epred(), joint.posterior_epred()["y"])

    fresh = data[["x", "z"]].iloc[[3, 1, 7]]
    expected = 0.3 + (0.8 - 0.4 * torch.tensor(fresh.z.to_numpy())) * torch.tensor(
        fresh.x.to_numpy()
    )
    torch.testing.assert_close(
        view.posterior_linpred(new_data=fresh)[0, 0].double(), expected
    )
    torch.testing.assert_close(
        view.posterior_epred(new_data=fresh)[0, 0].double(), expected
    )
    actual = data.y.loc[fresh.index]
    score = torch.distributions.Normal(expected, 0.7).log_prob(
        torch.tensor(actual.to_numpy())
    )
    torch.testing.assert_close(
        view.log_likelihood(new_data=fresh.assign(y=actual))["y"][0, 0].double(), score
    )
    with pytest.raises(ValueError, match="brms returns missing predictions"):
        view.posterior_epred(new_data=fresh.assign(x=[0.1, np.nan, 0.2]))


@pytest.mark.slow
def test_mi_recovers_the_slope_on_simulated_missing_at_random_data():
    generator = np.random.default_rng(11)
    rows = 40
    z = generator.normal(size=rows)
    x = 0.5 + 1.0 * z + 0.5 * generator.normal(size=rows)
    y = 0.2 + 1.5 * x + 0.3 * generator.normal(size=rows)
    drop = generator.random(rows) < 1 / (1 + np.exp(-z))  # missing at random
    data = pd.DataFrame({"y": y, "x": np.where(drop, np.nan, x), "z": z})
    fitted = fit(
        bf("y ~ mi(x)") + bf("x | mi() ~ z") + set_rescor(False),
        data=data,
        method="nuts",
        num_warmup=100,
        num_samples=100,
        num_chains=1,
        seed=4,
    )
    slope = fitted.variables()["bsp_y_mix"]
    # Tolerance fixed in advance: four posterior standard deviations plus the
    # Monte Carlo error of a 100-draw mean.
    error = slope.std() * (4 + 4 / np.sqrt(100))
    assert abs(float(slope.mean()) - 1.5) < float(error)


def test_mi_parameter_names_match_brms_variables():
    data = _frame()
    fitted = fit(
        bf("y ~ mi(x) + mi(x):z") + bf("x | mi() ~ z") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=1,
    )
    assert sorted(fitted.variables(num_samples=2)) == sorted(ORACLE["mi"]["variables"])
    assert [
        1 + row for row, flag in enumerate(ORACLE["mi"]["log_lik_x"]) if flag
    ] == ORACLE["mi"]["Jmi_x"]


def _complete() -> pd.DataFrame:
    return _frame().assign(
        x=[1.0, 0.1, 0.5, 0.3, 2.0, 1.5, 0.4, 0.2], g=list("abcdabcd")
    )


def test_me_terms_follow_brms_noncentered_correlated_latent_values():
    torch.set_default_dtype(torch.float64)
    try:
        data = _complete()
        spec = bf("y ~ me(x, s) + me(z, s)") + set_mecor(True)
        rule = PriorSpec(distribution="Normal(0.0, 2.0)", class_="meanme")
        assert get_prior(spec, data=data, priors=rule)["me_y__x__mean"] == (
            "Normal(loc=0.0, scale=2.0)"
        )
        fitted = fit(spec, data=data, priors=rule, method="svi", num_steps=1)
        assert sorted(fitted.variables(num_samples=2)) == sorted(
            ORACLE["me"]["variables"]
        )
        rows = len(data)
        factor = torch.tensor(
            [[1.0, 0.0], [0.4, float(np.sqrt(0.84))]], requires_grad=True
        )
        z = torch.linspace(-1.0, 1.0, 2 * rows).reshape(2, rows).requires_grad_(True)
        hyper = torch.tensor([0.5, -0.2, 0.7, 0.3], requires_grad=True)
        b = torch.tensor([0.1, 0.8, -0.4], requires_grad=True)
        sigma = torch.tensor(0.6, requires_grad=True)
        sites = {
            "me_y__x__mean": hyper[0],
            "me_y__z__mean": hyper[1],
            "me_y__x__sd": hyper[2],
            "me_y__z__sd": hyper[3],
            "me_y_rows__cor": factor,
            "me_y__x__z": z[0],
            "me_y__z__z": z[1],
            "response_y__intercept": b[0],
            "response_y__beta_x": b[1],
            "response_y__beta_z": b[2],
            "response_y__sigma": sigma,
        }
        result = trace(
            fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
        )
        # brms: Xme = meanme + transpose(diag(sdme) * Lme * zme).
        latent = hyper[:2, None] + hyper[2:, None] * (torch.tril(factor) @ z)
        s = torch.tensor(data.s.to_numpy())
        noisy = torch.tensor(data[["x", "z"]].to_numpy().T)
        y = torch.tensor(data.y.to_numpy())
        oracle = (
            torch.distributions.Normal(0.0, 2.0).log_prob(hyper[:2]).sum()
            + torch.distributions.HalfNormal(1.0).log_prob(hyper[2:]).sum()
            + torch.distributions.LKJCholesky(2, torch.tensor(1.0)).log_prob(factor)
            + torch.distributions.Normal(0.0, 1.0).log_prob(z).sum()
            + torch.distributions.Normal(0.0, 5.0).log_prob(b).sum()
            + torch.distributions.HalfCauchy(2.0).log_prob(sigma)
            + torch.distributions.Normal(latent, s).log_prob(noisy).sum()
            + torch.distributions.Normal(
                b[0] + b[1] * latent[0] + b[2] * latent[1], sigma
            )
            .log_prob(y)
            .sum()
        )
        torch.testing.assert_close(result.log_joint.sum(), oracle)
        leaves = (factor, z, hyper, b, sigma)
        for got, wanted in zip(
            torch.autograd.grad(result.log_joint.sum(), leaves, retain_graph=True),
            torch.autograd.grad(oracle, leaves),
            strict=True,
        ):
            torch.testing.assert_close(got, wanted)
        means = fitted.posterior_epred(new_data=data.iloc[:3], num_samples=2)["y"]
        draws = fitted.variables(num_samples=2)
        torch.testing.assert_close(
            means,
            draws["b_Intercept"][..., None]
            + draws["bsp_mexs"][..., None] * torch.tensor(data.x[:3].to_numpy())
            + draws["bsp_mezs"][..., None] * torch.tensor(data.z[:3].to_numpy()),
        )
    finally:
        torch.set_default_dtype(torch.float32)


def test_me_grouping_shares_one_latent_value_per_level():
    data = _complete().assign(x=[1.0, 0.1, 0.5, 0.3] * 2, s=[0.1, 0.2, 0.3, 0.1] * 2)
    fitted = fit(bf("y ~ me(x, s, gr = g)"), data=data, method="svi", num_steps=1)
    assert fitted.draws(num_samples=1)["me_y__x__z"].shape == (1, 1, 4)
    torch.testing.assert_close(
        fitted.observations["mi_y__x__source_0"],
        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
    )
    with pytest.raises(ValueError, match="identical measured values"):
        fit(bf("y ~ me(x, s, g)"), data=_complete(), method="svi", num_steps=1)


@pytest.mark.parametrize(
    ("spec", "name"),
    [
        (lambda: bf("y ~ z", "sigma ~ mi(x)"), "bsp_sigma_y_mix"),
        (lambda: bf("y ~ a + b * z", "a ~ mi(x)", "b ~ 1", nl=True), "bsp_y_a_mix"),
    ],
)
def test_mi_predictors_in_distributional_and_nonlinear_formulas(spec, name):
    data = _frame()
    fitted = fit(
        spec() + bf("x | mi() ~ z") + set_rescor(False),
        data=data,
        method="svi",
        num_steps=1,
    )
    assert name in fitted.variables(num_samples=2)
    fresh = data.assign(x=data.x.fillna(0.3)).iloc[:3]
    assert fitted.posterior_epred(new_data=fresh, num_samples=2)["y"].shape == (1, 2, 3)
