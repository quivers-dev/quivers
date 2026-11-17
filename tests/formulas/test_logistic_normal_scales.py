"""brms ``sigma<cat> ~ ...``: per-category scales of the logistic-normal law.

brms gives every free category ``c`` of a ``logistic_normal`` response its
own scale ``sigma<c>``, the standard deviation of that category's log ratio
to the reference. A formula predicts it through the log link, and a category
without one keeps a scalar ``sigma<c>``. The Stan statements, prior rows,
messages, variable names, and the values of ``log_lik`` and
``posterior_linpred`` at fixed draws are pinned by
``oracles/brms_logistic_normal_scales.R``.

Densities are compared in Float64 with the law written in this file from
brms's Stan function (a multivariate Normal on the log ratios with Cholesky
factor ``diag(sigma_n) L`` plus the Jacobian ``-sum(log y)``): tolerance
1e-10 for values and first derivatives and 1e-8 for second derivatives.
Fitted programs run in Float32 and are compared with brms at relative 1e-4
and absolute 1e-5.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats
import torch

from quivers.diagnostics.cross_validation import FoldPlan
from quivers.formulas import PriorSpec, bf, families, fit
from quivers.inference import trace
from quivers.inference.prior import PriorDraws
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_logistic_normal_scales.json").read_text()
)
RTOL, ATOL = 1e-4, 1e-5
RESPONSE = "cbind(a, b, c) ~ x"
#: Reference component, free components, predicted scales, scalar scales.
MODELS: dict[str, tuple[str, tuple[str, str], dict[str, str], tuple[str, ...]]] = {
    "one": ("a", ("b", "c"), {"sigmab": "~ x"}, ("sigmac",)),
    "both": ("a", ("b", "c"), {"sigmab": "~ x", "sigmac": "~ 1"}, ()),
    "refcat": ("b", ("a", "c"), {"sigmaa": "~ x"}, ("sigmac",)),
}


def _frame(source: str = "data") -> pd.DataFrame:
    return pd.DataFrame(ORACLE[source])


def _fit(name: str, data: pd.DataFrame | None = None, **settings):
    reference, _, scales, _ = MODELS[name]
    settings = {"seed": 1} | (settings or {"method": "svi", "num_steps": 1})
    return fit(
        RESPONSE,
        data=_frame() if data is None else data,
        family=families["logistic_normal"].with_reference(reference),
        distributional=scales,
        **settings,
    )


def _sites(name: str, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map brms's parameters to the fitted program's sites.

    Each value has any leading draw axes. The correlation becomes the
    two-by-two Cholesky factor ``[[1, 0], [r, sqrt(1 - r^2)]]``.
    """
    _, free, scales, scalars = MODELS[name]
    correlation = values[f"lncor__{free[0]}__{free[1]}"]
    one, zero = torch.ones_like(correlation), torch.zeros_like(correlation)
    sites = {
        "intercept": torch.stack(
            [values[f"b_mu{label}_Intercept"] for label in free], -1
        ),
        "beta_x": torch.stack([values[f"b_mu{label}_x"] for label in free], -1),
        "rescor": torch.stack(
            [
                torch.stack([one, zero], -1),
                torch.stack([correlation, (1 - correlation**2).sqrt()], -1),
            ],
            -2,
        ),
    }
    for scale, formula in scales.items():
        sites[f"dpar_{scale}_intercept"] = values[f"b_{scale}_Intercept"]
        if "x" in formula:
            sites[f"dpar_{scale}_beta_x"] = values[f"b_{scale}_x"]
    for scale in scalars:
        sites[scale] = values[scale]
    return sites


def _reference(
    name: str, values: dict[str, torch.Tensor], frame: pd.DataFrame
) -> torch.Tensor:
    """Pointwise logistic-normal log density from brms's Stan function."""
    reference, free, scales, _ = MODELS[name]
    x = torch.tensor(frame.x.to_numpy(), dtype=torch.float64)
    parts = torch.tensor(frame[["a", "b", "c"]].to_numpy(), dtype=torch.float64)
    ratios = torch.stack(
        [
            torch.tensor(np.log(frame[label] / frame[reference]).to_numpy())
            for label in free
        ],
        -1,
    )
    location = torch.stack(
        [
            values[f"b_mu{label}_Intercept"] + values[f"b_mu{label}_x"] * x
            for label in free
        ],
        -1,
    )
    columns = []
    for label in free:
        scale = f"sigma{label}"
        if scale not in scales:
            columns.append(values[scale] * torch.ones_like(x))
            continue
        eta = values[f"b_{scale}_Intercept"] * torch.ones_like(x)
        if "x" in scales[scale]:
            eta = eta + values[f"b_{scale}_x"] * x
        columns.append(eta.exp())
    sigma = torch.stack(columns, -1)
    correlation = values[f"lncor__{free[0]}__{free[1]}"]
    one, zero = torch.ones_like(correlation), torch.zeros_like(correlation)
    cholesky = torch.stack(
        [
            torch.stack([one, zero]),
            torch.stack([correlation, (1 - correlation**2).sqrt()]),
        ]
    )
    law = torch.distributions.MultivariateNormal(
        location, scale_tril=sigma[:, :, None] * cholesky
    )
    return law.log_prob(ratios) - parts.log().sum(-1)


def _point(name: str) -> tuple[list[str], torch.Tensor]:
    _, free, scales, scalars = MODELS[name]
    values = {
        f"b_mu{free[0]}_Intercept": 0.3,
        f"b_mu{free[1]}_Intercept": -0.2,
        f"b_mu{free[0]}_x": 0.4,
        f"b_mu{free[1]}_x": -0.25,
        f"lncor__{free[0]}__{free[1]}": -0.35,
    }
    for scale, formula in scales.items():
        values[f"b_{scale}_Intercept"] = -0.4
        if "x" in formula:
            values[f"b_{scale}_x"] = 0.3
    for scale in scalars:
        values[scale] = 0.7
    return list(values), torch.tensor(list(values.values()), dtype=torch.float64)


def test_brms_predicts_each_scale_through_the_log_link():
    one, both, plain = (ORACLE["cases"][key] for key in ("one", "both", "plain"))
    assert "sigmab = exp(sigmab);" in one["lines"]
    assert "sigma[n] = transpose([sigmab[n], sigmac]);" in one["lines"]
    assert "real<lower=0> sigmac;  // dispersion parameter" in one["lines"]
    assert "sigma[n] = transpose([sigmab[n], sigmac[n]]);" in both["lines"]
    assert "vector[ncat-1] sigma = transpose([sigmab, sigmac]);" in plain["lines"]
    # The reference component is argument five of the density.
    assert any(
        line.endswith("Llncor, 2);") for line in ORACLE["cases"]["refcat"]["lines"]
    )
    rows = pd.DataFrame(one["prior"])
    assert sorted(zip(rows["class"], rows.coef, rows.dpar, rows.lb, strict=True)) == [
        ("Intercept", "", "sigmab", ""),
        ("b", "", "sigmab", ""),
        ("b", "x", "sigmab", ""),
        ("sigmac", "", "", "0"),
    ]
    # brms has no shared 'sigma' and no scale of the reference component.
    for key, parameter in (("shared", "sigma"), ("reference", "sigmaa")):
        assert ORACLE["cases"][key]["message"].startswith(
            f"The parameter '{parameter}' is not a valid distributional"
        )

    fitted = _fit("one", default_priors="brms", method="svi", num_steps=1)
    table = {row.site: row for row in fitted.prior_summary().rows}
    assert table["sigmac"].class_ == "sigmac" and table["sigmac"].lower == 0.0
    assert table["sigmac"].prior == "HalfStudentT(df=3.0, scale=2.5)"
    assert table["dpar_sigmab_beta_x"].prior == "flat"
    assert (table["dpar_sigmab_beta_x"].class_, table["dpar_sigmab_beta_x"].dpar) == (
        "b",
        "sigmab",
    )
    assert table["dpar_sigmab_intercept"].class_ == "Intercept"
    assert "sigma" not in table and "sigmab" not in table


@pytest.mark.parametrize("name", sorted(MODELS))
def test_density_and_derivatives_match_the_logistic_normal_law(name):
    torch.set_default_dtype(torch.float64)
    try:
        fitted = _fit(name)
        frame = _frame()
        names, point = _point(name)

        def model(flat: torch.Tensor) -> torch.Tensor:
            sites = _sites(name, dict(zip(names, flat, strict=True)))
            recorded = trace(
                fitted.program, torch.zeros(1, 1), dict(fitted.observations) | sites
            )
            assert set(sites) <= set(recorded.sites)
            prior = sum(recorded.sites[site].log_prob.sum() for site in sites)
            return recorded.log_joint.sum() - prior

        def reference(flat: torch.Tensor) -> torch.Tensor:
            return _reference(name, dict(zip(names, flat, strict=True)), frame).sum()

        torch.testing.assert_close(
            model(point), reference(point), rtol=1e-10, atol=1e-10
        )
        gradient = torch.autograd.functional.jacobian(model, point)
        assert bool(torch.isfinite(gradient).all())
        torch.testing.assert_close(
            gradient,
            torch.autograd.functional.jacobian(reference, point),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            torch.autograd.functional.hessian(model, point),
            torch.autograd.functional.hessian(reference, point),
            rtol=1e-8,
            atol=1e-8,
        )
        # A scale that ignores its predictor is a different model.
        slope = next(
            index
            for index, key in enumerate(names)
            if key.endswith("_x") and "sigma" in key
        )
        constant = point.clone()
        constant[slope] = 0.0
        assert abs(float(reference(constant) - reference(point))) > 0.1
    finally:
        torch.set_default_dtype(torch.float32)


def test_reference_law_agrees_with_scipy():
    names, point = _point("one")
    values = dict(zip(names, point, strict=True))
    frame = _frame()
    scored = _reference("one", values, frame)
    for row in (0, 7):
        x = float(frame.x[row])
        sigma = np.array([np.exp(-0.4 + 0.3 * x), 0.7])
        correlation = np.array([[1.0, -0.35], [-0.35, 1.0]])
        expected = stats.multivariate_normal(
            mean=[0.3 + 0.4 * x, -0.2 - 0.25 * x],
            cov=correlation * np.outer(sigma, sigma),
        ).logpdf(np.log([frame.b[row] / frame.a[row], frame.c[row] / frame.a[row]]))
        expected -= np.log([frame.a[row], frame.b[row], frame.c[row]]).sum()
        assert abs(float(scored[row]) - expected) < 1e-12


def _pinned(fitted, values: dict[str, torch.Tensor]):
    """Replace the posterior by fixed draws on ``(chain=1, draw)`` axes."""
    draws = next(iter(values.values())).shape[0]
    return fitted.with_(
        posterior=PriorDraws(
            samples={
                name: value.float()[None].clone() for name, value in values.items()
            },
            num_samples=draws,
        )
    )


@pytest.mark.parametrize("name", sorted(MODELS))
def test_scores_predictors_and_names_match_brms_at_its_draws(name):
    oracle = ORACLE["values"][name]
    draws = pd.DataFrame(oracle["draws"])
    values = {
        column: torch.tensor(draws[column].to_numpy(), dtype=torch.float64)
        for column in draws
    }
    fitted = _pinned(_fit(name), _sites(name, values))
    scale = next(iter(MODELS[name][2]))

    def close(actual: torch.Tensor, key: str) -> None:
        torch.testing.assert_close(
            actual.float(),
            torch.tensor(oracle[key], dtype=torch.float32),
            rtol=RTOL,
            atol=ATOL,
        )

    (scores,) = fitted.log_likelihood().values()
    close(scores[0], "log_lik")
    close(fitted.posterior_linpred()[0], "linpred")
    close(fitted.posterior_linpred(parameter=scale)[0], "scale_linpred")
    close(fitted.posterior_linpred(parameter=scale, transform=True)[0], "scale")
    fresh = _frame("fresh")
    (fresh_scores,) = fitted.log_likelihood(new_data=fresh).values()
    close(fresh_scores[0], "new_log_lik")
    close(
        fitted.posterior_linpred(parameter=scale, transform=True, new_data=fresh)[0],
        "new_scale",
    )
    # brms has no expected value of this family.
    assert "Cannot compute expected values" in oracle["epred"]["error"]
    variables = fitted.variables()
    assert sorted(variables) == sorted(oracle["variables"])
    for variable in oracle["variables"]:
        torch.testing.assert_close(
            variables[variable][0].float(),
            torch.tensor(draws[variable].to_numpy(), dtype=torch.float32),
            rtol=RTOL,
            atol=ATOL,
        )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize("name", ["one", "grouped"])
def test_fit_predict_refit_and_restore(method, name, tmp_path):
    data = _frame().assign(g=[f"g{index % 4}" for index in range(len(_frame()))])
    settings = (
        {"method": "svi", "num_steps": 10}
        if method == "svi"
        else {"method": method, "num_warmup": 4, "num_samples": 4, "num_chains": 1}
    )
    if method == "hmc":
        settings["control"] = {"stepsize": 0.02}
    fitted = fit(
        RESPONSE,
        data=data,
        family="logistic_normal",
        distributional={"sigmab": "~ x" if name == "one" else "~ x + (1 | g)"},
        seed=2,
        **settings,
    )
    draws = fitted.draws(num_samples=3, seed=1)
    assert all(bool(torch.isfinite(value).all()) for value in draws.values())
    assert {"sigmac", "dpar_sigmab_intercept", "dpar_sigmab_beta_x"} <= set(draws)
    assert "sigma" not in draws
    names = set(fitted.variables())
    assert {"b_sigmab_Intercept", "b_sigmab_x", "sigmac", "lncor__b__c"} <= names
    assert (name == "grouped") == ("sd_g__sigmab_Intercept" in names)
    predicted = fitted.posterior_predict(num_samples=3, seed=1)
    assert predicted.shape == (1, 3, len(data), 3)
    torch.testing.assert_close(
        predicted.sum(-1), torch.ones_like(predicted.sum(-1)), rtol=1e-5, atol=1e-5
    )
    (scores,) = fitted.log_likelihood(num_samples=3, seed=1).values()
    assert scores.shape == (1, 3, len(data)) and bool(torch.isfinite(scores).all())
    fresh = data.iloc[[1, 4, 10]].assign(x=lambda rows: rows.x + 0.5)
    scale = fitted.posterior_linpred(
        parameter="sigmab", transform=True, new_data=fresh, num_samples=3, seed=1
    )
    assert scale.shape == (1, 3, len(fresh)) and bool((scale > 0).all())
    epred = fitted.posterior_epred(new_data=fresh, num_samples=3, seed=1, mc_samples=8)
    assert epred.shape == (1, 3, len(fresh), 3)
    if method != "svi":
        return
    refitted = fitted.update(num_steps=2)
    assert set(refitted.draws(num_samples=1)) == set(draws)
    path = tmp_path / "scales.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.posterior_linpred(
            parameter="sigmab", transform=True, new_data=fresh, num_samples=3, seed=1
        ),
        scale,
    )
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=2,
    )
    split = FoldPlan(
        assignments=tuple(index % 3 for index in range(len(data))), seed=1
    ).split(0)
    prediction = callback(split)
    held_out = prediction.fit.log_likelihood(
        new_data=data.iloc[list(split.test)], num_samples=2, seed=split.seed
    )
    (key,) = held_out
    torch.testing.assert_close(prediction.log_likelihood[key], held_out[key])


def test_priors_fixed_values_and_joint_models_address_one_scale():
    data = _frame().assign(w=lambda rows: rows.x * 0.5 + rows.a)
    fitted = fit(
        RESPONSE,
        data=data,
        family="logistic_normal",
        distributional={"sigmab": "~ x"},
        priors=(
            PriorSpec(distribution="HalfNormal(0.3)", class_="sigmac"),
            PriorSpec(distribution="Normal(0.0, 0.5)", class_="b", dpar="sigmab"),
        ),
        method="svi",
        num_steps=1,
        seed=1,
    )
    table = {row.site: row for row in fitted.prior_summary().rows}
    assert table["sigmac"].prior == "HalfNormal(scale=0.3)"
    assert table["dpar_sigmab_beta_x"].prior == "Normal(loc=0.0, scale=0.5)"
    assert table["beta_x"].source == "default"
    # A constant scale of one category, with the other estimated or predicted.
    for scales in ({}, {"sigmab": "~ x"}):
        constant = fit(
            RESPONSE,
            data=data,
            family="logistic_normal",
            distributional=scales,
            fixed_parameters={"sigmac": 0.5},
            method="svi",
            num_steps=1,
            seed=1,
        )
        names = set(constant.variables())
        assert "sigmac" not in names
        assert ("b_sigmab_x" in names) == bool(scales)
        assert ("sigmab" in names) != bool(scales)
    joint = fit(
        bf(RESPONSE, "sigmab ~ x", family="logistic_normal") + bf("w ~ x"),
        data=data,
        method="svi",
        num_steps=1,
        seed=1,
    )
    assert {"b_sigmab_cbindabc_x", "sigmac_cbindabc"} <= set(joint.variables())
    predicted = joint.posterior_predict(num_samples=2, seed=1)
    assert predicted["cbindabc"].shape == (1, 2, len(data), 3)


def test_refusals_name_the_scale():
    data = _frame()

    def attempt(**settings):
        settings = {"family": "logistic_normal"} | settings
        return fit(RESPONSE, data=data, method="svi", num_steps=1, **settings)

    # The reference component has no log ratio, so no scale.
    with pytest.raises(ValueError, match=r"unknown scale parameters \['sigmaa'\]"):
        attempt(distributional={"sigmaa": "~ x"})
    with pytest.raises(ValueError, match=r"unknown scale parameters \['sigmad'\]"):
        attempt(fixed_parameters={"sigmad": 1.0})
    with pytest.raises(ValueError, match="not both"):
        attempt(distributional={"sigma": "~ x", "sigmab": "~ x"})
    with pytest.raises(ValueError, match="not both"):
        attempt(distributional={"sigmab": "~ x"}, fixed_parameters={"sigma": 1.0})
    with pytest.raises(ValueError, match="cannot have distributional formulas"):
        attempt(distributional={"sigmab": "~ x"}, fixed_parameters={"sigmab": 1.0})
    with pytest.raises(ValueError, match="must be strictly positive"):
        attempt(fixed_parameters={"sigmac": -1.0})
    # Only the logistic-normal law has per-category scales.
    with pytest.raises(ValueError, match="unknown distributional parameters"):
        attempt(family="dirichlet", distributional={"sigmab": "~ x"})
