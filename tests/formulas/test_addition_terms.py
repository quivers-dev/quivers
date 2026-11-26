"""Response addition terms rate(), thres()/cat(), subset() 0/1 and update_adterms.

Oracles: SciPy Poisson and negative binomial laws at pinned draws (the
denominator multiplies the mean and the negative binomial shape, as brms's
Stan code pinned in ``oracles/brms_addition_terms.json`` does), and brms
2.20.4 ``make_standata`` threshold and row counts and ``update_adterms``
results. Tolerance 1e-6: design columns are stored in float32.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.formulas import bf, families, fit, formula_to_qvr, update_adterms
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.modeling.artifact import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_addition_terms.json").read_text()
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y": [1, 2, 2, 3, 1, 2],
            "x": [0.1, -0.3, 0.5, 1.2, -0.8, 0.4],
            "t": [1.5, 2.0, 3.0, 1.0, 2.5, 4.0],
            "keep": [1, 0, 1, 1, 0, 1],
        }
    )


def _pin(fitted, values):
    latent = LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names
    assert set(values) == set(latent)
    samples = {
        name: torch.as_tensor(np.asarray(value), dtype=torch.float64)[None]
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


def test_update_adterms_matches_brms():
    updates = ORACLE["updates"]
    assert (
        update_adterms("y | weights(w) + trials(n) ~ x", "~ trials(10) + cens(c)")
        == updates["update"]
    )
    assert (
        update_adterms("y | weights(w) ~ x", "~ cens(c)", action="replace")
        == updates["replace"]
    )
    assert update_adterms("y ~ x", "~ rate(t)") == updates["add"]
    assert (
        update_adterms(bf("y | weights(w) ~ x"), "~ subset(k)").formula
        == "y | weights(w) + subset(k) ~ x"
    )
    with pytest.raises(ValueError, match="one-sided"):
        update_adterms("y ~ x", "y ~ rate(t)")
    with pytest.raises(ValueError, match="named call"):
        update_adterms("y ~ x", "~ t")


@pytest.mark.parametrize(
    "family", ["poisson", "negative_binomial", "negbinomial2", "geometric"]
)
def test_rate_scales_the_mean_and_shape_like_brms(family):
    frame = _frame()
    fitted = fit(
        "y | rate(t) ~ x", data=frame, family=family, method="svi", num_steps=0
    )
    values = {
        "intercept": np.array([0.2, -0.1]),
        "beta_x": np.array([0.4, 0.3]),
    }
    if family == "negative_binomial":
        values["disp"] = np.array([2.0, 5.0])
    if family == "negbinomial2":
        values["sigma"] = np.array([0.5, 0.2])
    pinned = _pin(fitted, values)
    x = frame.x.to_numpy(np.float32).astype(np.float64)
    t = frame.t.to_numpy()
    y = frame.y.to_numpy()
    expected, means = [], []
    for d in range(2):
        mu = np.exp(values["intercept"][d] + values["beta_x"][d] * x)
        if family == "poisson":
            scores = stats.poisson.logpmf(y, mu * t)
        else:
            shape = {
                "negative_binomial": lambda: values["disp"][d],
                "negbinomial2": lambda: 1.0 / values["sigma"][d],
                "geometric": lambda: 1.0,
            }[family]()
            # brms: neg_binomial_2(mu * denom, shape * denom).
            n = shape * t
            scores = stats.nbinom.logpmf(y, n, n / (n + mu * t))
        expected.append(scores)
        means.append(mu * t)
    torch.testing.assert_close(
        pinned.log_likelihood()["y"].double(),
        torch.tensor(np.array(expected))[None],
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        pinned.posterior_epred().double(),
        torch.tensor(np.array(means))[None],
        rtol=1e-6,
        atol=1e-6,
    )
    # brms's mu excludes the denominator.
    torch.testing.assert_close(
        pinned.posterior_linpred(transform=True).double(),
        torch.tensor(np.array(means) / t)[None],
        rtol=1e-6,
        atol=1e-6,
    )
    fresh = frame.iloc[:2].drop(columns=["y"])
    torch.testing.assert_close(
        pinned.posterior_epred(new_data=fresh).double(),
        torch.tensor(np.array(means)[:, :2])[None],
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(ValueError):
        pinned.posterior_epred(new_data=fresh.drop(columns=["t"]))
    pins = "\n".join(
        ORACLE["rate"]["negbinomial" if family != "poisson" else "poisson_log"]
    )
    assert "log_denom" in pins


def test_rate_with_identity_link_multiplies_the_mean():
    frame = _frame()
    family = families["poisson"].with_links(mu="identity")
    fitted = fit(
        "y | rate(t) ~ 1", data=frame, family=family, method="svi", num_steps=0
    )
    pinned = _pin(fitted, {"intercept": np.array([1.3, 0.7])})
    expected = np.stack(
        [stats.poisson.logpmf(frame.y, value * frame.t) for value in (1.3, 0.7)]
    )
    torch.testing.assert_close(
        pinned.log_likelihood()["y"].double(),
        torch.tensor(expected)[None],
        rtol=1e-6,
        atol=1e-6,
    )
    assert "mu .* denom" in ORACLE["rate"]["poisson_identity"][-1]


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_rate_fits_and_restores(method, tmp_path):
    frame = pd.concat([_frame()] * 3, ignore_index=True)
    options = (
        {"num_steps": 20}
        if method == "svi"
        else {"num_warmup": 20, "num_samples": 10, "num_chains": 1}
    )
    fitted = fit(
        "y | rate(t) + subset(keep) ~ x",
        data=frame,
        family="negative_binomial",
        method=method,
        seed=1,
        **options,
    )
    assert fitted.formula.response_values.shape == (12,)
    path = tmp_path / "rate.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.log_likelihood(num_samples=3, seed=2)["y"],
        fitted.log_likelihood(num_samples=3, seed=2)["y"],
    )
    if method == "svi":
        assert "rate_shape" in fitted.update(num_steps=2).qvr_source


def test_threshold_counts_and_numeric_subset_match_brms():
    frame = _frame().assign(y=lambda f: f.y - 1)
    source = formula_to_qvr("y | thres(4) ~ x", data=frame, family="cumulative")
    assert f"object FormulaCutpoint : FinSet {ORACLE['nthres']['thres']}" in source
    source = formula_to_qvr("y | cat(5) ~ x", data=frame, family="cumulative")
    assert f"object FormulaCutpoint : FinSet {ORACLE['nthres']['cat']}" in source
    fitted = fit(
        "y | thres(4) ~ x", data=frame, family="cumulative", method="svi", num_steps=0
    )
    assert fitted.posterior_epred(num_samples=2).shape[-1] == 5
    subset = fit("y | subset(keep) ~ x", data=frame, method="svi", num_steps=0)
    assert subset.formula.response_values.shape == (ORACLE["subset_n"],)


@pytest.mark.parametrize(
    ("formula", "family", "message"),
    [
        ("y | rate(t) ~ x", "gaussian", "not supported for the gaussian"),
        ("y | rate(t) ~ x", "categorical", "not supported for the categorical"),
        ("y | rate(z) ~ x", "poisson", "positive"),
        ("y | thres(1) ~ x", "cumulative", "response codes"),
        ("y | thres(2) + cat(3) ~ x", "cumulative", "both declare"),
        ("y | subset(half) ~ x", "gaussian", "Boolean or 0/1"),
    ],
)
def test_rejected_addition_terms(formula, family, message):
    frame = _frame().assign(
        y=lambda f: f.y - 1, z=0.0, g="a", half=[0.5, 1, 1, 1, 1, 1]
    )
    with pytest.raises(ValueError, match=message):
        formula_to_qvr(formula, data=frame, family=family)
