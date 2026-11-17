"""brms ``mi()`` responses under estimated residual correlations.

brms scores the completed response vectors of every row under one
multivariate law, ``multi_normal_cholesky_lpdf(Yl | Mu, LSigma)`` or
``multi_student_t_lpdf(Yl | nu, Mu, Sigma)``, with the missing values (or,
under ``mi(sdy)``, every true value) as parameters without a prior. The Stan
statements and the values of ``log_lik``, ``posterior_epred`` and
``posterior_linpred`` at fixed draws are pinned by
``oracles/brms_missing_rescor.R``.

Densities are compared in Float64 with ``torch.distributions`` and SciPy
references written in this file, tolerance 1e-10 for values and first
derivatives and 1e-9 for second derivatives. Fitted programs run in Float32
and are compared with brms at relative 1e-4 and absolute 1e-5.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import multivariate_normal, multivariate_t
from scipy.stats import t as student_t

from quivers.diagnostics.cross_validation import FoldPlan
from quivers.formulas import bf, fit, formula_to_qvr, set_rescor
from quivers.inference import trace
from quivers.inference.prior import PriorDraws
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_missing_rescor.json").read_text()
)
RTOL, ATOL = 1e-4, 1e-5


def _frame() -> pd.DataFrame:
    return pd.DataFrame(ORACLE["data"]).astype({"x": float, "y": float})


def _spec(name: str):
    family = "student_t" if name == "student" else "gaussian"
    weights = "w" if name == "weighted" else None
    if name == "noisy":
        return bf("y ~ mi(x) + z") + bf("x | mi(s) ~ z") + set_rescor(True)
    second = "x | mi() ~ mi(y) + z" if name == "cyclic" else "x | mi() ~ z"
    return (
        bf("y | mi() ~ mi(x) + z", family=family, weights=weights)
        + bf(second, family=family, weights=weights)
        + set_rescor(True)
    )


def _values(name: str) -> dict[str, torch.Tensor]:
    """One fixed point of every latent site, in double precision."""
    values = {
        "response_y__intercept": torch.tensor(0.3),
        "response_y__beta_x": torch.tensor(0.8),
        "response_y__beta_z": torch.tensor(-0.4),
        "response_y__sigma": torch.tensor(0.7),
        "response_x__intercept": torch.tensor(0.1),
        "response_x__beta_z": torch.tensor(0.6),
        "response_x__sigma": torch.tensor(0.5),
        "rescor": torch.linalg.cholesky(torch.tensor([[1.0, -0.45], [-0.45, 1.0]])),
    }
    if name == "noisy":
        values["mi_x__truth"] = torch.linspace(0.2, 1.9, 11)
    else:
        values["mi_y__missing"] = torch.tensor([0.35])
        values["mi_x__missing"] = torch.tensor([0.4, -0.3])
    if name == "student":
        values["nu"] = torch.tensor(4.5)
    if name == "cyclic":
        values["response_x__beta_y"] = torch.tensor(0.25)
    return {key: value.double() for key, value in values.items()}


def _student_log_prob(
    value: torch.Tensor, nu: torch.Tensor, loc: torch.Tensor, factor: torch.Tensor
) -> torch.Tensor:
    """Multivariate Student log density with scale matrix ``factor factor'``."""
    width = value.shape[-1]
    covariance = factor @ factor.T
    deviation = value - loc
    quadratic = (deviation @ torch.linalg.inv(covariance) * deviation).sum(-1)
    return (
        torch.lgamma((nu + width) / 2)
        - torch.lgamma(nu / 2)
        - 0.5 * width * torch.log(nu * math.pi)
        - 0.5 * torch.logdet(covariance)
        - 0.5 * (nu + width) * torch.log1p(quadratic / nu)
    )


def _reference(
    name: str, values: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Write brms's likelihood by hand.

    Returns the total log likelihood, the row locations, the scale factor,
    and the completed response vectors.
    """
    data = _frame()
    if name == "noisy":
        # brms drops the row whose plain response is missing.
        data = data[np.isfinite(data.y)].reset_index(drop=True)
    y = torch.tensor(np.nan_to_num(data.y.to_numpy()))
    x = torch.tensor(np.nan_to_num(data.x.to_numpy()))
    z = torch.tensor(data.z.to_numpy())
    measurement = torch.zeros((), dtype=torch.float64)
    if name == "noisy":
        truth = values["mi_x__truth"]
        seen = torch.tensor(np.isfinite(data.x.to_numpy()))
        measurement = (
            torch.distributions.Normal(
                truth[seen], torch.tensor(data.s.to_numpy())[seen]
            )
            .log_prob(x[seen])
            .sum()
        )
        x = truth
    else:
        y = y.masked_scatter(
            torch.tensor(~np.isfinite(data.y.to_numpy())), values["mi_y__missing"]
        )
        x = x.masked_scatter(
            torch.tensor(~np.isfinite(data.x.to_numpy())), values["mi_x__missing"]
        )
    mu_x = values["response_x__intercept"] + values["response_x__beta_z"] * z
    if name == "cyclic":
        mu_x = mu_x + values["response_x__beta_y"] * y
    mu_y = (
        values["response_y__intercept"]
        + values["response_y__beta_x"] * x
        + values["response_y__beta_z"] * z
    )
    loc = torch.stack([mu_y, mu_x], dim=-1)
    completed = torch.stack([y, x], dim=-1)
    sigma = torch.stack([values["response_y__sigma"], values["response_x__sigma"]])
    # brms: LSigma = diag_pre_multiply(sigma, Lrescor).
    factor = sigma[:, None] * values["rescor"]
    if name == "student":
        rows = _student_log_prob(completed, values["nu"], loc, factor)
    else:
        rows = torch.distributions.MultivariateNormal(loc, scale_tril=factor).log_prob(
            completed
        )
    if name == "weighted":
        rows = rows * torch.tensor(data.w.to_numpy())
    return rows.sum() + measurement, loc, factor, completed


def _likelihood(fitted, sites: dict[str, torch.Tensor]) -> torch.Tensor:
    """Log joint of the fitted program without its prior terms."""
    recorded = trace(
        fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
    )
    assert set(sites) <= set(recorded.sites)
    prior = sum(recorded.sites[name].log_prob.sum() for name in sites)
    return recorded.log_joint.sum() - prior


def test_brms_scores_completed_vectors_under_the_joint_law():
    cases = ORACLE["cases"]
    assert (
        "target += multi_normal_cholesky_lpdf(Yl | Mu, LSigma);"
        in (cases["plain"]["lines"])
    )
    assert {
        "Yl_y[Jmi_y] = Ymi_y;",
        "Yl[n][1] = Yl_y[n];",
        "Yl[n][2] = Yl_x[n];",
    } <= set(cases["plain"]["lines"])
    assert (
        "matrix[nresp, nresp] LSigma = diag_pre_multiply(sigma, Lrescor);"
        in (cases["plain"]["lines"])
    )
    assert (
        "target += multi_student_t_lpdf(Yl | nu, Mu, Sigma);"
        in (cases["student"]["lines"])
    )
    assert "vector[N_x] Yl_x;  // latent variable" in cases["noisy"]["lines"]
    assert (
        "target += normal_lpdf(Y_x[Jme_x] | Yl_x[Jme_x], noise_x[Jme_x]);"
        in cases["noisy"]["lines"]
    )
    assert (
        "target += weights[n] * (multi_normal_cholesky_lpdf(Yl[n] | Mu[n], LSigma));"
        in cases["weighted"]["lines"]
    )
    # brms admits responses that use each other's completed values.
    assert "mu_x[n] += (bsp_x[1]) * Yl_y[n];" in cases["cyclic"]["lines"]
    assert ORACLE["standata"] == {"Jmi_y": 4, "Jmi_x": [2, 7], "N": 12, "nresp": 2}
    source = formula_to_qvr(_spec("plain"), data=_frame())
    # The missing values carry no prior and the joint law is one factor.
    assert "sample mi_x__missing : MiMissing_x <- ImproperFlat(log_density=0.0)" in (
        source
    )
    assert "sample mi_y__missing : MiMissing_y <- ImproperFlat(log_density=0.0)" in (
        source
    )
    assert "score formula_joint_score" in source
    assert "observe formula_joint_response" not in source
    noisy = formula_to_qvr(_spec("noisy"), data=_frame())
    assert "sample mi_x__truth : FormulaJointRow <- ImproperFlat(log_density=0.0)" in (
        noisy
    )
    assert "observe mi_x__observed : MiObserved_x <- Normal(" in noisy


@pytest.mark.parametrize("name", ["plain", "noisy", "student", "weighted", "cyclic"])
def test_joint_density_and_its_derivatives_match_the_hand_written_model(name):
    torch.set_default_dtype(torch.float64)
    try:
        fitted = fit(_spec(name), data=_frame(), method="svi", num_steps=1)
        values = _values(name)
        torch.testing.assert_close(
            _likelihood(fitted, values),
            _reference(name, values)[0],
            rtol=1e-10,
            atol=1e-10,
        )
        latent = "mi_x__truth" if name == "noisy" else "mi_x__missing"
        point = torch.cat(
            [
                values[latent],
                values["response_y__sigma"][None],
                values["rescor"][1],
                values["response_y__beta_x"][None],
            ]
        )
        size = len(values[latent])

        def unpack(flat: torch.Tensor) -> dict[str, torch.Tensor]:
            factor = torch.stack([values["rescor"][0], flat[size + 1 : size + 3]])
            return dict(values) | {
                latent: flat[:size],
                "response_y__sigma": flat[size],
                "rescor": factor,
                "response_y__beta_x": flat[size + 3],
            }

        def model(flat: torch.Tensor) -> torch.Tensor:
            return _likelihood(fitted, unpack(flat))

        def reference(flat: torch.Tensor) -> torch.Tensor:
            return _reference(name, unpack(flat))[0]

        torch.testing.assert_close(
            torch.autograd.functional.jacobian(model, point),
            torch.autograd.functional.jacobian(reference, point),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            torch.autograd.functional.hessian(model, point),
            torch.autograd.functional.hessian(reference, point),
            rtol=1e-9,
            atol=1e-9,
        )
    finally:
        torch.set_default_dtype(torch.float32)


def test_reference_densities_agree_with_scipy_and_differ_from_independence():
    for name in ("plain", "student"):
        values = _values(name)
        total, loc, factor, completed = _reference(name, values)
        covariance = (factor @ factor.T).numpy()
        rows = [
            multivariate_t(mean, covariance, df=float(values["nu"])).logpdf(point)
            if name == "student"
            else multivariate_normal(mean, covariance).logpdf(point)
            for mean, point in zip(loc.numpy(), completed.numpy(), strict=True)
        ]
        assert float(total) == pytest.approx(float(np.sum(rows)), rel=1e-12, abs=1e-12)
    # Scoring each completed response under its own Normal law, the model
    # without residual correlation, is a different density.
    values = _values("plain")
    total, loc, factor, completed = _reference("plain", values)
    scales = torch.stack([values["response_y__sigma"], values["response_x__sigma"]])
    independent = torch.distributions.Normal(loc, scales).log_prob(completed).sum()
    assert abs(float(total - independent)) > 0.5


def _pinned(fitted, values: dict[str, torch.Tensor]):
    """Replace the posterior by fixed draws on ``(chain=1, draw)`` axes."""
    draws = next(iter(values.values())).shape[0]
    return fitted.with_(
        posterior=PriorDraws(
            samples={
                name: value.float()[None].clone() for name, value in values.items()
            },
            num_samples=draws,
        ),
        response_fits={},
    )


def _brms_draws(name: str) -> dict[str, torch.Tensor]:
    """Map brms's draws to latent sites, one leading draw axis."""
    draws = pd.DataFrame(ORACLE["values"][name]["draws"])

    def column(label: str) -> torch.Tensor:
        return torch.tensor(draws[label].to_numpy(dtype=float))

    correlation = column("rescor__y__x")
    factor = torch.zeros(len(draws), 2, 2, dtype=torch.float64)
    factor[:, 0, 0] = 1.0
    factor[:, 1, 0] = correlation
    factor[:, 1, 1] = torch.sqrt(1 - correlation**2)
    values = {
        "response_y__intercept": column("b_y_Intercept"),
        "response_y__beta_x": column("bsp_y_mix"),
        "response_y__beta_z": column("b_y_z"),
        "response_y__sigma": column("sigma_y"),
        "response_x__intercept": column("b_x_Intercept"),
        "response_x__beta_z": column("b_x_z"),
        "response_x__sigma": column("sigma_x"),
        "rescor": factor,
    }
    if name == "noisy":
        values["mi_x__truth"] = torch.stack(
            [column(f"Yl_x[{row}]") for row in range(1, 12)], dim=-1
        )
    else:
        values["mi_y__missing"] = column("Ymi_y[4]")[:, None]
        values["mi_x__missing"] = torch.stack(
            [column("Ymi_x[2]"), column("Ymi_x[7]")], dim=-1
        )
    if name == "student":
        values["nu"] = column("nu")
    return values


def _numbers(value: list) -> torch.Tensor:
    """Parse brms's decimal strings, NA becoming NaN."""
    array = np.array(value, dtype=object)
    return torch.tensor(
        np.vectorize(lambda text: float("nan") if text == "NA" else float(text))(
            array
        ).astype(np.float32)
    )


@pytest.mark.parametrize("name", ["plain", "noisy", "student"])
def test_scores_and_expectations_match_brms_at_its_draws(name):
    oracle = ORACLE["values"][name]
    data = _frame()
    fitted = _pinned(
        fit(_spec(name), data=data, method="svi", num_steps=1, seed=1),
        _brms_draws(name),
    )
    scores = fitted.log_likelihood()["y__x"][0].float()
    wanted = _numbers(oracle["log_lik"])
    # brms reports NA where a response of the row has no recorded value.
    assert bool(torch.isnan(wanted).any())
    torch.testing.assert_close(scores, wanted, rtol=RTOL, atol=ATOL, equal_nan=True)
    marginal = fitted.log_likelihood(resp="y")["y"][0].float()
    if name == "student":
        # brms scores a response subset of a Student joint event with the
        # response's placeholder nu_y = 1 instead of the shared nu. The exact
        # marginal of the joint event is the univariate Student law with the
        # shared nu.
        draws = pd.DataFrame(oracle["draws"])
        assert set(draws["nu_y"]) == {1}
        location = np.array(oracle["epred"], dtype=float)[..., 0]
        scale = draws["sigma_y"].to_numpy()[:, None]
        outcome = data.y.to_numpy()[None]
        brms = np.array(
            [
                [float("nan") if text == "NA" else float(text) for text in row]
                for row in oracle["log_lik_y"]
            ]
        )
        np.testing.assert_allclose(
            student_t(1, location, scale).logpdf(outcome), brms, rtol=1e-10, atol=1e-10
        )
        wanted_marginal = torch.tensor(
            student_t(draws["nu"].to_numpy()[:, None], location, scale).logpdf(outcome),
            dtype=torch.float32,
        )
    else:
        wanted_marginal = _numbers(oracle["log_lik_y"])
    torch.testing.assert_close(
        marginal, wanted_marginal, rtol=RTOL, atol=ATOL, equal_nan=True
    )
    expected = fitted.posterior_epred()
    linear = fitted.posterior_linpred()
    for position, resp in enumerate(("y", "x")):
        torch.testing.assert_close(
            expected[resp][0].float(),
            _numbers(oracle["epred"])[..., position],
            rtol=RTOL,
            atol=ATOL,
        )
        torch.testing.assert_close(
            linear[resp][0].float(),
            _numbers(oracle["linpred"])[..., position],
            rtol=RTOL,
            atol=ATOL,
        )
    # brms draws the true values of fresh rows from the measurement model,
    # so its fresh-row values of the noisy model are random.
    if "new_log_lik" not in oracle or name == "noisy":
        return
    fresh = pd.DataFrame(ORACLE["fresh"])
    torch.testing.assert_close(
        fitted.log_likelihood(new_data=fresh)["y__x"][0].float(),
        _numbers(oracle["new_log_lik"]),
        rtol=RTOL,
        atol=ATOL,
    )
    new = fitted.posterior_epred(new_data=fresh)
    for position, resp in enumerate(("y", "x")):
        torch.testing.assert_close(
            new[resp][0].float(),
            _numbers(oracle["new_epred"])[..., position],
            rtol=RTOL,
            atol=ATOL,
        )


def test_predictive_draws_keep_the_joint_residual_law():
    values = {
        key: value.expand(4000, *value.shape) for key, value in _values("plain").items()
    }
    fitted = _pinned(
        fit(_spec("plain"), data=_frame(), method="svi", num_steps=1, seed=1), values
    )
    _, loc, factor, _ = _reference("plain", _values("plain"))
    predicted = fitted.posterior_predict(seed=5)
    residual = (
        torch.stack([predicted["y"][0], predicted["x"][0]], dim=-1).float()
        - loc.float()
    )
    covariance = torch.einsum("drj,drk->jk", residual, residual) / (
        residual.shape[0] * residual.shape[1]
    )
    wanted = (factor @ factor.T).float()
    # 48000 independent rows: the standard error of a covariance entry is
    # below 0.004, so 0.02 is five standard errors.
    torch.testing.assert_close(covariance, wanted, rtol=0.0, atol=0.02)
    assert float(covariance[0, 1]) < -0.1
    # A strict subset is scored by the marginal of the joint event.
    scores = fitted.log_likelihood(resp="x", num_samples=1)["x"][0, 0]
    data = _frame()
    marginal = torch.distributions.Normal(loc[:, 1], factor[1].norm()).log_prob(
        torch.tensor(np.nan_to_num(data.x.to_numpy()))
    )
    marginal[~torch.tensor(np.isfinite(data.x.to_numpy()))] = float("nan")
    torch.testing.assert_close(
        scores.float(), marginal.float(), rtol=RTOL, atol=ATOL, equal_nan=True
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("name", ["plain", "noisy"])
def test_fit_predict_refit_and_restore(method, name, tmp_path):
    data = _frame()
    settings = (
        {"method": "svi", "num_steps": 10}
        if method == "svi"
        else {"method": method, "num_warmup": 4, "num_samples": 4, "num_chains": 1}
    )
    fitted = fit(_spec(name), data=data, seed=2, **settings)
    draws = fitted.draws(num_samples=3, seed=1)
    assert all(bool(torch.isfinite(value).all()) for value in draws.values())
    variables = set(fitted.variables())
    assert "rescor__y__x" in variables
    assert ("Yl_x[1]" if name == "noisy" else "Ymi_x[2]") in variables
    rows = 11 if name == "noisy" else 12
    epred = fitted.posterior_epred(num_samples=3, seed=1)
    predicted = fitted.posterior_predict(num_samples=3, seed=1)
    for resp in ("y", "x"):
        assert epred[resp].shape == predicted[resp].shape == (1, 3, rows)
        assert bool(torch.isfinite(predicted[resp]).all())
    scores = fitted.log_likelihood(num_samples=3, seed=1)["y__x"]
    assert scores.shape == (1, 3, rows)
    assert int(torch.isnan(scores[0, 0]).sum()) == (2 if name == "noisy" else 3)
    fresh = data.dropna().iloc[:4]
    new = fitted.posterior_predict(new_data=fresh, num_samples=3, seed=1)
    assert new["y"].shape == new["x"].shape == (1, 3, 4)
    assert bool(
        torch.isfinite(
            fitted.log_likelihood(new_data=fresh, num_samples=3, seed=1)["y__x"]
        ).all()
    )
    if method != "svi":
        return
    refitted = fitted.update(num_steps=2)
    assert set(refitted.draws(num_samples=1)) == set(draws)
    path = tmp_path / "rescor.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.posterior_epred(num_samples=3, seed=1)["y"], epred["y"]
    )
    torch.testing.assert_close(
        restored.log_likelihood(num_samples=3, seed=1)["y__x"], scores, equal_nan=True
    )
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=2,
    )
    # Held-out units are the rows whose mi() values are all recorded.
    recorded = [
        row
        for row in range(len(data))
        if np.isfinite(data.x[row]) and np.isfinite(data.y[row])
    ]
    split = FoldPlan(
        assignments=tuple(index % 2 for index in range(len(recorded))), seed=1
    ).split(0)
    prediction = callback(split)
    held_out = prediction.fit.log_likelihood(
        new_data=data.iloc[[recorded[i] for i in split.test]],
        num_samples=2,
        seed=split.seed,
    )
    torch.testing.assert_close(prediction.log_likelihood["y__x"], held_out["y__x"])


def test_refusals_follow_brms():
    data = _frame()
    assert ORACLE["cases"]["censored"]["message"].startswith(
        "Only 'se', 'weights', 'mi' are supported addition arguments"
    )
    with pytest.raises(ValueError, match="censoring and truncation"):
        fit(
            bf("y | mi() ~ mi(x) + z")
            + bf("x | mi() + cens(c) ~ z")
            + set_rescor(True),
            data=data.assign(c=0),
            method="svi",
            num_steps=1,
        )
    with pytest.raises(ValueError, match="identical original response rows"):
        fit(
            bf("y | subset(k) ~ mi(x) + z") + bf("x | mi() ~ z") + set_rescor(True),
            data=data.assign(
                k=[1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1], y=data.y.fillna(0)
            ),
            method="svi",
            num_steps=1,
        )
    fitted = fit(_spec("plain"), data=data, method="svi", num_steps=1)
    # A fresh row with a missing mi() value has no completed value.
    with pytest.raises(ValueError):
        fitted.posterior_epred(new_data=data.iloc[[1]])
    # The missing values carry the improper flat law, which has no draws.
    with pytest.raises(ValueError, match="improper flat"):
        fit(_spec("plain"), data=data, method="svi", num_steps=1, sample_prior="only")
