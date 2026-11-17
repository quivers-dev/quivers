"""``mo()``, ``gp()``, ``mm()`` and ``gr()`` options in per-category predictors.

Oracles: brms 2.20.4 variable names, prior rows, and ``log_lik`` /
``posterior_epred`` at brms's own draws
(``oracles/brms_category_special_terms.R``), a NumPy evaluation of the
monotonic, multi-membership and Student-t group predictors written here, and
SciPy multinomial, Dirichlet and Dirichlet-multinomial laws.

Tolerances were fixed before comparing. brms stores draws to six significant
digits, so a monotonic simplex is renormalized here and values computed from
those draws agree to 1e-5. NumPy and SciPy references use float64 parameters
over float32 design columns and agree to 1e-6.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.formulas import fit, formula_to_qvr, get_prior
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry
from quivers.modeling.artifact import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_category_special_terms.json").read_text()
)
BRMS_TOL = 1e-5
TOL = 1e-6
SEPARATE = {
    "muB": "~ mo(k) + (1 | mm(g1, g2))",
    "muC": "~ gp(z, k = 5, c = 5/4) + (1 | gr(g1, by = gb))",
}
LEVELS = ("p", "q", "r")


def _frame() -> pd.DataFrame:
    frame = pd.DataFrame(ORACLE["data"])
    frame["y"] = pd.Categorical(frame["y"], categories=["A", "B", "C"])
    return frame


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


def _draw(draws, name, entries=None):
    if entries is None:
        return np.asarray(draws[name], dtype=np.float64)
    return np.stack(
        [np.asarray(draws[f"{name}[{entry}]"], dtype=np.float64) for entry in entries],
        axis=-1,
    )


def _simplex(draws, name, width=3):
    values = _draw(draws, name, range(1, width + 1))
    return values / values.sum(axis=-1, keepdims=True)


def _separate_values(draws):
    intercepts = [f"{level},Intercept" for level in LEVELS]
    sd_member = _draw(draws, "sd_mmg1g2__muB_Intercept")
    sd_by = np.stack(
        [
            _draw(draws, "sd_g1__muC_Intercept:gbu"),
            _draw(draws, "sd_g1__muC_Intercept:gbv"),
        ],
        axis=-1,
    )
    # Levels p and q of g1 lie in by-level u, level r in by-level v.
    return {
        "dpar_muB_simo_mok1": _simplex(draws, "simo_muB_mok1"),
        "dpar_muB_bsp_mo_k": _draw(draws, "bsp_muB_mok"),
        "dpar_muB_intercept": _draw(draws, "b_muB_Intercept"),
        "dpar_muB_sigma_mmg1g2_Intercept": sd_member,
        "dpar_muB_z_mmg1g2_Intercept": _draw(draws, "r_mmg1g2__muB", intercepts)
        / sd_member[:, None],
        "dpar_muC_sdgp_gpz": _draw(draws, "sdgp_muC_gpz"),
        "dpar_muC_lscale_gpz": _draw(draws, "lscale_muC_gpz"),
        "dpar_muC_gp_gpz_z": _draw(draws, "zgp_muC_gpz", range(1, 6)),
        "dpar_muC_intercept": _draw(draws, "b_muC_Intercept"),
        "dpar_muC_sigma_g1_Intercept": sd_by,
        "dpar_muC_z_g1_Intercept": _draw(draws, "r_g1__muC", intercepts)
        / sd_by[:, [0, 0, 1]],
    }


def _copied_values(draws):
    values = {}
    for category in "BC":
        prefix = f"dpar_mu{category}_"
        values |= {
            prefix + "simo_mok1": _simplex(draws, f"simo_mu{category}_mok1"),
            prefix + "bsp_mo_k": _draw(draws, f"bsp_mu{category}_mok"),
            prefix + "intercept": _draw(draws, f"b_mu{category}_Intercept"),
            prefix + "sdgp_gpz": _draw(draws, f"sdgp_mu{category}_gpz"),
            prefix + "lscale_gpz": _draw(draws, f"lscale_mu{category}_gpz"),
            prefix + "gp_gpz_z": _draw(draws, f"zgp_mu{category}_gpz", range(1, 24)),
        }
    return values


@pytest.fixture(scope="module")
def separate():
    return fit(
        "y ~ x",
        data=_frame(),
        family="categorical",
        distributional=SEPARATE,
        method="svi",
        num_steps=0,
    )


@pytest.fixture(scope="module")
def copied():
    return fit(
        "y ~ mo(k) + gp(z)",
        data=_frame(),
        family="categorical",
        method="svi",
        num_steps=0,
    )


def _assert_brms_scores(pinned, case):
    torch.testing.assert_close(
        pinned.log_likelihood()["y"][0].double(),
        torch.tensor(case["log_lik"], dtype=torch.float64),
        rtol=BRMS_TOL,
        atol=BRMS_TOL,
    )
    torch.testing.assert_close(
        pinned.posterior_epred()[0].double(),
        torch.tensor(case["epred"], dtype=torch.float64),
        rtol=BRMS_TOL,
        atol=BRMS_TOL,
    )


def test_separate_category_terms_match_brms_names_and_scores(separate):
    case = ORACLE["separate"]
    assert list(separate.variables()) == case["variables"]
    _assert_brms_scores(_pin(separate, _separate_values(case["draws"])), case)


def test_main_formula_terms_are_copied_to_every_category_as_in_brms(copied):
    case = ORACLE["copied"]
    stan = "\n".join(case["stan"])
    # brms gives each category its own simplex and GP hyperparameters.
    for name in ("simo_muB_1", "simo_muC_1", "sdgp_muB_1", "lscale_muC_1"):
        assert name in stan
    assert list(copied.variables()) == case["variables"]
    _assert_brms_scores(_pin(copied, _copied_values(case["draws"])), case)


def test_a_wrong_simplex_order_disagrees_with_brms(separate):
    case = ORACLE["separate"]
    values = _separate_values(case["draws"])
    values["dpar_muB_simo_mok1"] = values["dpar_muB_simo_mok1"][:, ::-1].copy()
    scores = _pin(separate, values).log_likelihood()["y"][0].double()
    assert (scores - torch.tensor(case["log_lik"])).abs().max() > 1e-3


def test_default_priors_match_the_brms_prior_rows(copied):
    rows = ORACLE["copied"]["prior"]
    brms = {
        (cls, coef, dpar): prior
        for prior, cls, coef, dpar in zip(
            rows["prior"], rows["class"], rows["coef"], rows["dpar"], strict=True
        )
    }
    table = {
        (row.class_, row.coef, row.dpar): row.prior
        for row in copied.prior_summary().rows
    }
    for category in ("muB", "muC"):
        assert brms[("simo", "mok1", category)] == "dirichlet(1)"
        assert table[("simo", "mok1", category)] == (
            "Dirichlet(concentration=[1.0, 1.0, 1.0])"
        )
        assert brms[("lscale", "gpz", category)] == "inv_gamma(1.494197, 0.056607)"
        assert table[("lscale", "gpz", category)] == (
            "InverseGamma(concentration=1.494197, rate=0.056607)"
        )
        assert brms[("sdgp", "", category)] == "student_t(3, 0, 2.5)"
        assert table[("sdgp", "gpz", category)] == "HalfStudentT(df=3.0, scale=2.5)"
        assert ("b", "mok", category) in brms
        assert ("b", "mo(k)", category) in table
    priors = get_prior("y ~ mo(k) + gp(z)", data=_frame(), family="categorical")
    # Every latent except the standard normal GP coordinates has a prior row.
    assert set(priors) == {
        name for name in _latent(copied) if not name.endswith("_gp_gpz_z")
    }


def _monotonic(codes: np.ndarray, simplex: np.ndarray) -> np.ndarray:
    """brms ``mo``: the simplex length times its first ``code`` entries."""
    cumulative = np.concatenate(([0.0], np.cumsum(simplex)))
    return len(simplex) * cumulative[codes]


def _vector_frame(kind: str) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    rows = 12
    if kind == "counts":
        response = rng.multinomial(6, [0.2, 0.3, 0.5], rows).astype(np.float64)
    else:
        response = rng.dirichlet([2.0, 3.0, 4.0], rows)
    frame = pd.DataFrame(
        {
            "a": response[:, 0],
            "b": response[:, 1],
            "c": response[:, 2],
            "k": rng.integers(1, 5, rows),
            "g1": rng.choice(list(LEVELS), rows),
            "g2": rng.choice(list(LEVELS), rows),
        }
    )
    # Every level and every monotonic value occurs.
    frame.loc[:2, "g1"] = list(LEVELS)
    frame.loc[:3, "k"] = [1, 2, 3, 4]
    if kind == "counts":
        frame[["a", "b", "c"]] = frame[["a", "b", "c"]].astype(int)
    return frame, response


VECTOR_FAMILIES = {
    "multinomial": "counts",
    "dirichlet_multinomial": "counts",
    "dirichlet": "shares",
    "logistic_normal": "shares",
}


@pytest.mark.parametrize("family", sorted(VECTOR_FAMILIES))
def test_vector_family_predictors_equal_the_numpy_reference(family):
    frame, response = _vector_frame(VECTOR_FAMILIES[family])
    fitted = fit(
        "cbind(a, b, c) ~ mo(k) + (1 | mm(g1, g2))",
        data=frame,
        family=family,
        method="svi",
        num_steps=0,
    )
    rng = np.random.default_rng(11)
    draws = 2
    values = {}
    for name in _latent(fitted):
        shape = tuple(fitted.draws()[name].shape[2:])
        if len(shape) == 2 and shape[0] == shape[1]:
            # Cholesky factor of the logistic-normal correlation matrix.
            values[name] = np.broadcast_to(np.eye(shape[0]), (draws, *shape)).copy()
        elif "simo" in name:
            values[name] = rng.dirichlet(np.ones(shape[-1]), draws)
        elif "sigma_mm" in name or name in {"phi", "sigma"}:
            values[name] = rng.uniform(0.5, 2.0, (draws, *shape))
        else:
            values[name] = rng.normal(size=(draws, *shape)) * 0.5
    pinned = _pin(fitted, values)
    codes = frame["k"].to_numpy() - frame["k"].min()
    first = pd.Categorical(frame["g1"], categories=LEVELS).codes
    second = pd.Categorical(frame["g2"], categories=LEVELS).codes
    logits = np.zeros((draws, len(frame), 3))
    for position, category in ((1, "mub"), (2, "muc")):
        prefix = f"dpar_{category}_"
        for draw in range(draws):
            effects = (
                values[prefix + "sigma_mmg1g2_Intercept"][draw]
                * values[prefix + "z_mmg1g2_Intercept"][draw]
            )
            logits[draw, :, position] = (
                values[prefix + "intercept"][draw]
                + values[prefix + "bsp_mo_k"][draw]
                * _monotonic(codes, values[prefix + "simo_mok1"][draw])
                + 0.5 * effects[first]
                + 0.5 * effects[second]
            )
        torch.testing.assert_close(
            pinned.posterior_linpred(parameter=category)[0].double(),
            torch.tensor(logits[..., position]),
            rtol=TOL,
            atol=TOL,
        )
    scores = next(iter(pinned.log_likelihood().values()))[0].double()
    assert bool(torch.isfinite(scores).all())
    if family == "logistic_normal":
        return
    mean = np.exp(logits) / np.exp(logits).sum(-1, keepdims=True)
    expected = np.empty((draws, len(frame)))
    for draw in range(draws):
        for row in range(len(frame)):
            if family == "multinomial":
                expected[draw, row] = stats.multinomial.logpmf(
                    response[row], 6, mean[draw, row]
                )
            elif family == "dirichlet":
                expected[draw, row] = stats.dirichlet.logpdf(
                    response[row], mean[draw, row] * values["phi"][draw]
                )
            else:
                expected[draw, row] = stats.dirichlet_multinomial.logpmf(
                    response[row].astype(int),
                    mean[draw, row] * values["phi"][draw],
                    6,
                )
    torch.testing.assert_close(scores, torch.tensor(expected), rtol=TOL, atol=TOL)


def test_student_group_coefficients_in_one_category_equal_the_numpy_reference():
    frame = _frame()
    fitted = fit(
        "y ~ 1",
        data=frame,
        family="categorical",
        distributional={"muC": "~ (1 | gr(g1, dist = 'student'))"},
        method="svi",
        num_steps=0,
    )
    names = set(_latent(fitted))
    assert {"df_g1", "udf_g1", "dpar_muC_z_g1_Intercept"} <= names
    assert not any(name.startswith("dpar_muB_z") for name in names)
    values = {
        "dpar_muB_intercept": np.array([0.3]),
        "dpar_muC_intercept": np.array([-0.2]),
        "df_g1": np.array([4.5]),
        "udf_g1": np.array([[0.4, 0.9, 0.2]]),
        "dpar_muC_sigma_g1_Intercept": np.array([1.3]),
        "dpar_muC_z_g1_Intercept": np.array([[0.5, -1.1, 0.7]]),
    }
    assert set(values) == names
    # brms: r = sd * sqrt(df * udf) * z, a Student-t scale mixture.
    effects = (
        1.3 * np.sqrt(4.5 * values["udf_g1"][0]) * values["dpar_muC_z_g1_Intercept"][0]
    )
    expected = -0.2 + effects[pd.Categorical(frame["g1"], categories=LEVELS).codes]
    torch.testing.assert_close(
        _pin(fitted, values).posterior_linpred(parameter="muC")[0, 0].double(),
        torch.tensor(expected),
        rtol=TOL,
        atol=TOL,
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_category_special_terms_fit_predict_refit_and_restore(method, tmp_path):
    frame = _frame()
    options = (
        {"num_steps": 30}
        if method == "svi"
        else {"num_warmup": 15, "num_samples": 8, "num_chains": 1}
    )
    fitted = fit(
        "y ~ x",
        data=frame,
        family="categorical",
        distributional=SEPARATE,
        method=method,
        seed=3,
        **options,
    )
    # New rows: an unseen monotonic-range interior, GP input and member pair.
    fresh = frame.iloc[:5].assign(z=lambda f: f.z + 0.05, g2=lambda f: f.g1)
    epred = fitted.posterior_epred(new_data=fresh.drop(columns="y"), num_samples=3)
    assert epred.shape[-2:] == (5, 3)
    torch.testing.assert_close(
        epred.sum(-1), torch.ones_like(epred.sum(-1)), rtol=1e-5, atol=1e-5
    )
    for parameter in ("muB", "muC"):
        linpred = fitted.posterior_linpred(
            parameter=parameter, new_data=fresh, num_samples=3
        )
        assert linpred.shape[-1] == 5 and bool(torch.isfinite(linpred).all())
    assert fitted.posterior_predict(new_data=fresh, num_samples=3).shape[-1] == 5
    assert bool(
        torch.isfinite(fitted.log_likelihood(new_data=fresh, num_samples=3)["y"]).all()
    )
    path = tmp_path / "special.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.posterior_epred(new_data=fresh, num_samples=3, seed=1),
        fitted.posterior_epred(new_data=fresh, num_samples=3, seed=1),
    )
    if method == "svi":
        updated = fitted.update(num_steps=3)
        assert "dpar_muB_simo_mok1" in updated.draws()
        folds = fitted.kfold(K=2, seed=1)
        assert np.isfinite(folds.estimate.elpd)


def test_rejected_special_terms():
    frame = _frame()
    # A value outside the fitted monotonic range has no simplex entry.
    fitted = fit(
        "y ~ mo(k)", data=frame, family="categorical", method="svi", num_steps=0
    )
    with pytest.raises(ValueError, match="k"):
        fitted.posterior_epred(new_data=frame.iloc[:2].assign(k=9), num_samples=1)
    # Category-specific cs() terms belong to ordinal families only.
    with pytest.raises(ValueError, match="cs terms"):
        formula_to_qvr(
            "y ~ x", data=frame, family="categorical", distributional={"muC": "~ cs(x)"}
        )
    # The logistic-normal scale predictor keeps one coefficient per category.
    shares, _ = _vector_frame("shares")
    with pytest.raises(ValueError, match="one coefficient per response category"):
        formula_to_qvr(
            "cbind(a, b, c) ~ 1",
            data=shares,
            family="logistic_normal",
            distributional={"sigma": "~ mo(k)"},
        )
    with pytest.raises(ValueError, match="me\\(\\) terms are supported in the main"):
        formula_to_qvr(
            "y ~ x",
            data=frame,
            family="categorical",
            distributional={"muC": "~ me(x, 0.1)"},
        )
    with pytest.raises(ValueError, match="unknown category parameters"):
        formula_to_qvr(
            "y ~ mo(k)", data=frame, family="categorical", distributional={"muA": "~ x"}
        )
