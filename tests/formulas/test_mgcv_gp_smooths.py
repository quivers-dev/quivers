"""mgcv Gaussian process smooths ``s(x, bs = "gp")`` equal brms's Stan data.

Oracle: ``oracles/mgcv_gp_smooths.R`` (brms 2.20.4, mgcv 1.8-42) records, for
every covariance choice of ``m``, the unpenalized design ``Xs`` and penalized
design ``Zs`` (checked there against ``make_standata``) and the matrices brms
evaluates on fresh rows. Penalized columns are compared up to sign.
Unpenalized columns of a one-dimensional null space are compared up to sign,
and those of a larger null space by their span, because brms takes them from
a repeated zero eigenvalue and gives them exchangeable flat priors.

Tolerance, fixed before comparing: mgcv stops its Lanczos eigensolver at a
relative residual of ``eps ** 0.5`` (1.5e-8), so eigenvector error is that
residual over the relative eigengap. A relative tolerance of 1e-6 of each
matrix's largest entry covers it while rejecting any change of correlation
function, range, knots, constraint, or scale. The correlation functions are
also checked against closed forms written here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quivers.formulas import conditional_smooths, fit, formula_from_data
from quivers.formulas._mgcv_smooth import GaussianProcessSmoothBasis, gp_smooth
from quivers.formulas._smooth import SmoothTransform
from quivers.formulas.formula import _formula_and_design
from quivers.modeling import load_fit, save_fit

ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "mgcv_gp_smooths.json").read_text()
)
RTOL = 1e-6


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(ORACLE["train"])
    fresh = pd.DataFrame(ORACLE["fresh"])
    levels = sorted(set(train["g"]))
    train["g"] = pd.Categorical(train["g"], categories=levels)
    fresh["g"] = pd.Categorical(fresh["g"], categories=levels)
    return train, fresh


def _matrix(rows: list, count: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(count, -1)


def _assert_columns_up_to_sign(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    if not expected.size:
        return
    signs = np.sign(np.sum(actual * expected, axis=0))
    scale = max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual * signs, expected, rtol=0, atol=RTOL * scale)


def _assert_same_span(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    if not expected.size:
        return
    left = np.linalg.qr(actual)[0]
    right = np.linalg.qr(expected)[0]
    np.testing.assert_allclose(left @ left.T, right @ right.T, rtol=0, atol=RTOL)


def _smooth(spec: str, train: pd.DataFrame):
    formula, design = _formula_and_design("y ~ " + spec, train)
    [(name, term)] = [
        (name, term)
        for name, term in design.common.terms.items()
        if name != "Intercept"
    ]
    transform = term.components[0].call.stateful_transform
    assert isinstance(transform, SmoothTransform)
    return formula, design, name, transform.state


@pytest.mark.parametrize("entry", ORACLE["entries"], ids=lambda e: e["spec"])
def test_training_and_fresh_designs_equal_brms(entry):
    train, fresh = _frames()
    formula, design, name, state = _smooth(entry["spec"], train)
    columns = design.common.slices[name]
    trained = np.asarray(design.common.design_matrix)[:, columns]
    evaluated = np.asarray(design.common.evaluate_new_data(fresh).design_matrix)[
        :, columns
    ]
    assert len(state.layout) == len(entry["blocks"])
    for layout, block in zip(state.layout, entry["blocks"], strict=True):
        for values, prefix, rows in (
            (trained, "", len(train)),
            (evaluated, "fresh_", len(fresh)),
        ):
            expected = block[prefix + "fixed"]
            fixed = _matrix(expected, rows) if expected else np.zeros((rows, 0))
            compare = (
                _assert_columns_up_to_sign if fixed.shape[1] <= 1 else _assert_same_span
            )
            compare(values[:, list(layout.fixed)], fixed)
            assert len(layout.random) == len(block[prefix + "random"])
            for group, expected in zip(
                layout.random, block[prefix + "random"], strict=True
            ):
                _assert_columns_up_to_sign(
                    values[:, list(group)], _matrix(expected, rows)
                )


@pytest.mark.parametrize(
    ("definition", "closed_form"),
    [
        ((1.0, 0.8), lambda e: (1 - 1.5 * e + 0.5 * e**3) * (e <= 1)),
        ((2.0, 0.8, 1.5), lambda e: np.exp(-(e**1.5))),
        ((3.0, 0.8), lambda e: (1 + e) * np.exp(-e)),
        ((4.0, 0.8), lambda e: (1 + e + e**2 / 3) * np.exp(-e)),
        ((5.0, 0.8), lambda e: (1 + e + 0.4 * e**2 + e**3 / 15) * np.exp(-e)),
    ],
)
def test_correlation_functions_equal_their_closed_forms(definition, closed_form):
    x = np.linspace(-1.0, 1.5, 12)[:, None]
    raw = gp_smooth(x, 12, definition)
    basis = raw.basis
    assert isinstance(basis, GaussianProcessSmoothBasis)
    centered = x - x.mean()
    distance = np.abs(centered - centered.T) / 0.8
    np.testing.assert_allclose(
        raw.design[:, :10],
        closed_form(distance) @ basis.transform,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(raw.design[:, 10], 1.0)
    np.testing.assert_allclose(raw.design[:, 11], centered[:, 0], atol=1e-14)
    # The retained map holds eigenvectors of the knot correlation matrix.
    correlation = closed_form(distance)
    values = np.diag(raw.penalties[0])[:10]
    np.testing.assert_allclose(
        correlation @ basis.transform, basis.transform * values, rtol=0, atol=1e-6
    )
    # A wrong range disagrees.
    assert not np.allclose(
        raw.design[:, :10], closed_form(distance * 1.1) @ basis.transform, atol=1e-3
    )


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_gp_smooths_fit_predict_refit_and_restore(method, tmp_path):
    train, fresh = _frames()
    fresh = fresh.assign(y=np.linspace(0.0, 1.0, len(fresh)))
    options = (
        {"num_steps": 30}
        if method == "svi"
        else {"num_warmup": 15, "num_samples": 8, "num_chains": 1}
    )
    formula = "y ~ s(x, bs = 'gp', k = 6, m = c(-3, 0.5)) + z"
    fitted = fit(formula, data=train, method=method, seed=2, **options)
    linpred = fitted.posterior_linpred(new_data=fresh, num_samples=3)
    assert linpred.shape[-1] == len(fresh) and bool(torch.isfinite(linpred).all())
    assert fitted.posterior_predict(new_data=fresh, num_samples=3).shape[-1] == 7
    assert bool(
        torch.isfinite(fitted.log_likelihood(new_data=fresh, num_samples=3)["y"]).all()
    )
    torch.testing.assert_close(
        fitted.posterior_linpred(new_data=train, num_samples=3, seed=1),
        fitted.posterior_linpred(num_samples=3, seed=1),
        rtol=1e-6,
        atol=1e-6,
    )
    term = next(iter(t.term for t in fitted.formula.design_terms))
    effect = conditional_smooths(fitted, term, fresh, num_samples=3, seed=1)
    assert effect.shape[-1] == len(fresh) and bool(torch.isfinite(effect).all())
    path = tmp_path / "gp_smooth.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.posterior_linpred(new_data=fresh, num_samples=3, seed=1),
        fitted.posterior_linpred(new_data=fresh, num_samples=3, seed=1),
    )
    if method == "svi":
        updated = fitted.update(num_steps=3)
        assert set(updated.draws()) == set(fitted.draws())
        assert np.isfinite(fitted.kfold(K=2, seed=1).estimate.elpd)


@pytest.mark.parametrize(
    ("formula", "message"),
    [
        ("y ~ s(x, bs = 'gp', m = 6)", "type in 1..5"),
        ("y ~ s(x, bs = 'gp', m = 2.5)", "type in 1..5"),
        ("y ~ s(x, bs = 'gp', m = c(2, 0.5, 3))", "power in"),
        ("y ~ s(x, bs = 'gp', m = c(3, 1, 1, 1))", "type in 1..5"),
        ("y ~ s(x, bs = 'gp', k = 45)", "fewer unique covariate combinations"),
        ("y ~ t2(x, z, bs = 'gp')", "margins must use bases"),
        ("y ~ s(x, bs = 'ad')", "brms refuses it as well"),
        ("y ~ s(x, z, bs = 'so')", "brms refuses it as well"),
        ("y ~ s(x, z, bs = 'ds')", "unsupported smooth basis"),
        ("y ~ s(g, bs = 'mrf')", "requires xt = list"),
        ("y ~ s(g, bs = 'gp')", "finite real row vector"),
    ],
)
def test_invalid_gp_smooths_are_rejected(formula, message):
    train, _ = _frames()
    with pytest.raises(ValueError, match=message):
        formula_from_data(formula, train)
