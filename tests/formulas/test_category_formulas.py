"""Category-specific formulas ``mu<c> ~ ...`` of categorical and simplex families.

Oracles: hand-written torch softmax densities (values, first and second
derivatives), SciPy multinomial and Dirichlet laws, the shared-coefficient
emission at matching coefficients, and brms 2.20.4 names, prior classes and
``log_lik`` / ``posterior_epred`` at brms's draws
(``oracles/brms_category_formulas.R``). Float64 parameters over float32
design columns agree to float64 rounding (1e-10); brms values are compared
at 1e-6 because quivers stores design columns in float32.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.formulas import PriorSpec, bf, families, fit, formula_to_qvr
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.inference._response import response_distribution
from quivers.inference.trace import trace
from quivers.modeling.artifact import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_category_formulas.json").read_text()
)
RTOL = 1e-10
ATOL = 1e-10


def _frame() -> pd.DataFrame:
    data = ORACLE["data"]
    labels = np.array(["A", "B", "C"])[np.asarray(data["y"], dtype=int)]
    return pd.DataFrame(
        {
            "y": pd.Categorical(labels, categories=["A", "B", "C"]),
            "x": np.asarray(data["x"], dtype=np.float64),
            "z": np.asarray(data["z"], dtype=np.float64),
        }
    )


def _latent(fitted) -> tuple[str, ...]:
    return LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names


def _pin(fitted, values):
    assert set(values) == set(_latent(fitted))
    samples = {
        name: torch.as_tensor(np.asarray(value), dtype=torch.float64).unsqueeze(0)
        for name, value in values.items()
    }
    count = next(iter(samples.values())).shape[1]
    return fitted.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(1, count),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=count,
        )
    )


def _column(name: str) -> torch.Tensor:
    return torch.tensor(_frame()[name].to_numpy(), dtype=torch.float32).double()


SITES = (
    "dpar_muB_intercept",
    "dpar_muB_beta_x",
    "dpar_muC_intercept",
    "dpar_muC_beta_z",
)


def _oracle(theta: torch.Tensor) -> torch.Tensor:
    b0, b1, c0, c1 = theta.unbind()
    x, z = _column("x"), _column("z")
    logits = torch.stack((torch.zeros_like(x), b0 + b1 * x, c0 + c1 * z), dim=-1)
    codes = torch.as_tensor(np.asarray(ORACLE["data"]["y"]), dtype=torch.long)
    return torch.log_softmax(logits, dim=-1).gather(-1, codes[:, None]).squeeze(-1)


@pytest.fixture(scope="module")
def categorical():
    return fit(
        "y ~ x",
        data=_frame(),
        family="categorical",
        distributional={"muC": "~ z"},
        method="svi",
        num_steps=0,
    )


def _model(fitted, theta: torch.Tensor) -> torch.Tensor:
    latents = dict(zip(SITES, theta.unbind(), strict=True))
    recorded = trace(
        fitted.program,
        torch.zeros(1, 1, dtype=torch.long),
        dict(fitted.observations) | latents,
    )
    codes = torch.as_tensor(np.asarray(ORACLE["data"]["y"]), dtype=torch.long)
    return response_distribution(recorded, "y").log_prob(codes).reshape(-1)


POINT = torch.tensor([0.3, -0.6, -0.2, 0.8], dtype=torch.float64)


def test_category_specific_density_and_derivatives(categorical):
    assert set(_latent(categorical)) == set(SITES)
    torch.testing.assert_close(
        _model(categorical, POINT), _oracle(POINT), rtol=RTOL, atol=ATOL
    )
    point = POINT.clone().requires_grad_(True)
    (actual,) = torch.autograd.grad(_model(categorical, point).sum(), point)
    (expected,) = torch.autograd.grad(_oracle(point).sum(), point)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        torch.autograd.functional.hessian(
            lambda t: _model(categorical, t).sum(), POINT.clone()
        ),
        torch.autograd.functional.hessian(lambda t: _oracle(t).sum(), POINT.clone()),
        rtol=1e-9,
        atol=1e-9,
    )
    # A wrong variant (category C reading x) differs.
    wrong = POINT.clone()
    assert not torch.allclose(
        _model(categorical, wrong), _oracle(wrong[[0, 1, 2, 2]]), atol=1e-3
    )


def test_identical_category_formulas_reproduce_the_shared_emission():
    frame = _frame()
    shared = fit("y ~ x", data=frame, family="categorical", method="svi", num_steps=0)
    separate = fit(
        "y ~ x",
        data=frame,
        family="categorical",
        distributional={"muB": "~ x"},
        method="svi",
        num_steps=0,
    )
    assert "dpar_muC_beta_x" in _latent(separate)
    intercepts = torch.tensor([0.4, -0.3], dtype=torch.float64)
    slopes = torch.tensor([-0.2, 0.9], dtype=torch.float64)
    shared_scores = (
        trace(
            shared.program,
            torch.zeros(1, 1, dtype=torch.long),
            dict(shared.observations) | {"intercept": intercepts, "beta_x": slopes},
        )
        .sites["y"]
        .log_prob
    )
    separate_scores = (
        trace(
            separate.program,
            torch.zeros(1, 1, dtype=torch.long),
            dict(separate.observations)
            | {
                "dpar_muB_intercept": intercepts[0],
                "dpar_muB_beta_x": slopes[0],
                "dpar_muC_intercept": intercepts[1],
                "dpar_muC_beta_x": slopes[1],
            },
        )
        .sites["y"]
        .log_prob
    )
    torch.testing.assert_close(
        separate_scores.reshape(-1), shared_scores.reshape(-1), rtol=RTOL, atol=ATOL
    )


def test_brms_names_classes_and_scores_at_brms_draws(categorical):
    stan = "\n".join(ORACLE["stan"])
    assert "vector[N] muB" in stan and "vector[N] muC" in stan
    assert "categorical_logit_lpmf" in stan
    refcat = "\n".join(ORACLE["refcat_stan"])
    assert "vector[N] muA" in refcat and "muC" not in refcat.replace("muC =", "")
    classes = {
        entry
        for entry in ORACLE["prior_classes"]
        if entry.split("|")[0] in {"b", "Intercept"}
    }
    targets = {
        f"{target.class_}|{target.coef if target.class_ == 'b' else ''}|{target.dpar}"
        for target in categorical.prediction_plan.prior_targets
    }
    expected = {entry for entry in classes if entry.split("|")[1] != ""} | {
        entry for entry in classes if entry.startswith("Intercept")
    }
    assert expected <= targets
    rows = ORACLE["draws"]
    draws = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    pinned = _pin(
        categorical,
        {
            "dpar_muB_intercept": draws["b_muB_Intercept"],
            "dpar_muB_beta_x": draws["b_muB_x"],
            "dpar_muC_intercept": draws["b_muC_Intercept"],
            "dpar_muC_beta_z": draws["b_muC_z"],
        },
    )
    torch.testing.assert_close(
        pinned.log_likelihood()["y"][0].double(),
        torch.tensor(ORACLE["log_lik"], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        pinned.posterior_epred()[0].double(),
        torch.tensor(ORACLE["epred"], dtype=torch.float64),
        rtol=1e-6,
        atol=1e-6,
    )


def test_reference_category_selects_the_free_parameters():
    frame = _frame()
    family = families["categorical"].with_reference("C")
    source = formula_to_qvr(
        "y ~ x", data=frame, family=family, distributional={"muA": "~ z"}
    )
    assert "dpar_muA_beta_z" in source and "dpar_muB_beta_x" in source
    assert "dpar_muC" not in source
    with pytest.raises(ValueError, match="unknown category parameters"):
        formula_to_qvr(
            "y ~ x", data=frame, family=family, distributional={"muC": "~ z"}
        )


def test_category_priors_select_one_category():
    frame = _frame()
    source = formula_to_qvr(
        "y ~ x",
        data=frame,
        family="categorical",
        distributional={"muC": "~ z"},
        priors=PriorSpec(distribution="Normal(0.0, 0.5)", class_="b", dpar="muC"),
    )
    assert "dpar_muC_beta_z <- Normal(loc=0.0, scale=0.5)" in source
    assert "dpar_muB_beta_x <- Normal(loc=0.0, scale=0.5)" not in source


def test_multinomial_and_dirichlet_category_formulas_match_scipy():
    rng = np.random.default_rng(5)
    rows = 10
    x = np.round(rng.normal(size=rows), 3)
    counts = rng.multinomial(6, [0.2, 0.3, 0.5], rows)
    frame = pd.DataFrame(
        {"x": x, "a": counts[:, 0], "b": counts[:, 1], "c": counts[:, 2]}
    )
    fitted = fit(
        "cbind(a, b, c) ~ x",
        data=frame,
        family="multinomial",
        distributional={"muc": "~ 1"},
        method="svi",
        num_steps=0,
    )
    values = {
        "dpar_mub_intercept": np.array([0.2, -0.1]),
        "dpar_mub_beta_x": np.array([0.5, 0.3]),
        "dpar_muc_intercept": np.array([0.7, 0.4]),
    }
    pinned = _pin(fitted, values)
    xs = x.astype(np.float32).astype(np.float64)
    expected = []
    for d in range(2):
        logits = np.stack(
            (
                np.zeros(rows),
                values["dpar_mub_intercept"][d] + values["dpar_mub_beta_x"][d] * xs,
                np.full(rows, values["dpar_muc_intercept"][d]),
            ),
            axis=-1,
        )
        probs = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        expected.append(stats.multinomial.logpmf(counts, 6, probs))
    torch.testing.assert_close(
        pinned.log_likelihood()["a_b_c"].double()
        if "a_b_c" in pinned.log_likelihood()
        else next(iter(pinned.log_likelihood().values())).double(),
        torch.tensor(np.array(expected)).unsqueeze(0),
        rtol=1e-6,
        atol=1e-6,
    )
    shares = rng.dirichlet([2.0, 3.0, 4.0], rows)
    simplex = pd.DataFrame(
        {"x": x, "a": shares[:, 0], "b": shares[:, 1], "c": shares[:, 2]}
    )
    dirichlet = fit(
        "cbind(a, b, c) ~ x",
        data=simplex,
        family="dirichlet",
        distributional={"mub": "~ 1"},
        method="svi",
        num_steps=0,
    )
    assert {"dpar_mub_intercept", "dpar_muc_beta_x", "phi"} <= set(_latent(dirichlet))
    point = {
        "dpar_mub_intercept": np.array([0.1, 0.3]),
        "dpar_muc_intercept": np.array([0.4, 0.2]),
        "dpar_muc_beta_x": np.array([-0.3, 0.6]),
        "phi": np.array([5.0, 8.0]),
    }
    pinned = _pin(dirichlet, point)
    expected = []
    for d in range(2):
        logits = np.stack(
            (
                np.zeros(rows),
                np.full(rows, point["dpar_mub_intercept"][d]),
                point["dpar_muc_intercept"][d] + point["dpar_muc_beta_x"][d] * xs,
            ),
            axis=-1,
        )
        mean = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
        alpha = mean * point["phi"][d]
        expected.append(
            np.array(
                [
                    stats.dirichlet.logpdf(shares[i].astype(np.float64), alpha[i])
                    for i in range(rows)
                ]
            )
        )
    torch.testing.assert_close(
        next(iter(pinned.log_likelihood().values())).double(),
        torch.tensor(np.array(expected)).unsqueeze(0),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_category_formulas_fit_predict_refit_and_restore(method, tmp_path):
    frame = _frame()
    options = (
        {"num_steps": 30}
        if method == "svi"
        else {"num_warmup": 20, "num_samples": 10, "num_chains": 1}
    )
    fitted = fit(
        "y ~ x",
        data=frame,
        family="categorical",
        distributional={"muC": "~ z"},
        method=method,
        seed=2,
        **options,
    )
    fresh = frame.iloc[:4]
    assert fitted.posterior_epred(
        new_data=fresh.drop(columns="y"), num_samples=3
    ).shape[-2:] == (4, 3)
    assert (
        fitted.posterior_linpred(parameter="muC", new_data=fresh, num_samples=3).shape[
            -1
        ]
        == 4
    )
    assert fitted.posterior_predict(new_data=fresh, num_samples=3).shape[-1] == 4
    assert torch.isfinite(
        fitted.log_likelihood(new_data=fresh, num_samples=3)["y"]
    ).all()
    assert set(fitted.fixed_effects(parameter="muC", num_samples=3)) == {
        "Intercept",
        "z",
    }
    path = tmp_path / "categorical.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.log_likelihood(num_samples=3, seed=1)["y"],
        fitted.log_likelihood(num_samples=3, seed=1)["y"],
    )
    if method == "svi":
        updated = fitted.update(num_steps=3)
        assert "dpar_muC_beta_z" in updated.draws()


def test_category_formulas_inside_a_joint_model():
    frame = _frame().assign(w=lambda f: f.x + 0.1 * f.z)
    spec = bf("y ~ x", "muC ~ z", family="categorical") + bf("w ~ x")
    joint = fit(spec, data=frame, method="svi", num_steps=10, seed=1)
    assert any(name.endswith("dpar_muC_beta_z") for name in joint.draws())
    scores = joint.log_likelihood(resp="y", new_data=frame.iloc[:3])["y"]
    assert bool(torch.isfinite(scores).all())


def test_rejected_category_specifications():
    frame = _frame()
    with pytest.raises(ValueError, match="unknown category parameters"):
        formula_to_qvr(
            "y ~ x", data=frame, family="categorical", distributional={"muA": "~ z"}
        )
    with pytest.raises(ValueError, match="unknown distributional parameters"):
        formula_to_qvr(
            "y ~ x", data=frame, family="gaussian", distributional={"muB": "~ z"}
        )
