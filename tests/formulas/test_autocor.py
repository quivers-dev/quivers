"""brms autocorrelation terms agree with brms's Stan programs and independent oracles.

Pinned brms values come from ``tests/formulas/oracles/autocor_brms.R``.
Tolerances are fixed from Float64 rounding before comparison: 1e-10 for
closed-form densities and their autograd derivatives.
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, multivariate_t, norm, t as student
import torch

from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.formulas import PriorSpec, fit, get_prior
from quivers.formulas._autocor import parse_autocor_call, split_autocor_terms
from quivers.formulas._residual_dependence import (
    ResidualResponse,
    sar_weights,
    spectrum_tolerance,
)
from quivers.modeling import load_fit, save_fit

RTOL = ATOL = 1e-10
# Fitted formula programs run in the default Float32 dtype.
FIT_RTOL, FIT_ATOL = 2e-5, 2e-6

# Unsorted rows of the R oracle: brms orders by group, then time.
FRAME = pd.DataFrame(
    {
        "y": [0.3, -1.2, 0.8, 2.1, -0.4, 1.5, 0.9],
        "x": [1.0, 0.5, -0.2, 0.7, 1.3, -0.9, 0.1],
        "t": [3, 1, 2, 2, 1, 4, 3],
        "g": ["b", "a", "b", "a", "b", "a", "a"],
    }
)
BRMS_SORTED_Y = [-1.2, 2.1, 0.9, 1.5, -0.4, 0.8, 0.3]
BRMS_J_LAG = [1, 2, 2, 0, 1, 2, 0]
BRMS_ARMA1 = np.array(
    [
        [1.0119047619047621, 0.10476190476190481, 0.041904761904761924],
        [0.10476190476190481, 1.0119047619047621, 0.10476190476190481],
        [0.041904761904761924, 0.10476190476190481, 1.0119047619047621],
    ]
)
BRMS_AR1 = np.array(
    [
        [1.1904761904761905, 0.47619047619047622, 0.19047619047619052],
        [0.47619047619047622, 1.1904761904761905, 0.47619047619047622],
        [0.19047619047619052, 0.47619047619047622, 1.1904761904761905],
    ]
)
BRMS_MA1 = np.array([[1.09, -0.3, 0.0], [-0.3, 1.09, -0.3], [0.0, -0.3, 1.09]])
PATH = np.array(
    [
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.0, 1.0, 0.0],
    ]
)


def _t(values, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def _layout(term, frame=FRAME, *, student_law=False, matrix=None):
    terms = split_autocor_terms(f"y ~ x + {term}")[1]
    from quivers.formulas import families
    from quivers.formulas._autocor import autocor_binding, autocor_family
    from quivers.formulas.formula import ResponseModifiers

    family = autocor_family(
        terms,
        families["student_t" if student_law else "gaussian"],
        data=frame,
        response=frame["y"].to_numpy(),
        response_name="y",
        data2=None if matrix is None else {"M": matrix},
        distributional=None,
        modifiers=ResponseModifiers(),
        caller="test",
    )
    binding = autocor_binding(family)
    assert binding is not None
    return binding.layout


def _stan_arma(eta, y, sigma, ar, ma, groups, nu=None):
    """Transcribe brms's Stan model block on brms-sorted rows."""
    count = len(y)
    lags = max(len(ar), len(ma))
    err_matrix = [[eta.new_zeros(()) for _ in range(lags)] for _ in range(count + 1)]
    mu = list(eta)
    err = [None] * count
    for n in range(count):
        for k in range(len(ma)):
            mu[n] = mu[n] + err_matrix[n][k] * ma[k]
        err[n] = y[n] - mu[n]
        if n + 1 < count:
            available = sum(
                1
                for j in range(max(0, n + 1 - lags), n + 1)
                if groups[j] == groups[n + 1]
            )
            for i in range(available):
                err_matrix[n + 1][i] = err[n - i]
        for k in range(len(ar)):
            mu[n] = mu[n] + err_matrix[n][k] * ar[k]
    mu = torch.stack(mu)
    if nu is None:
        return mu, torch.distributions.Normal(mu, sigma).log_prob(y)
    return mu, torch.distributions.StudentT(nu, mu, sigma).log_prob(y)


def test_formula_terms_parse_with_brms_defaults_and_rejections():
    stripped, (term,) = split_autocor_terms(
        "y ~ x + arma(time = t, gr = g, cov = TRUE)"
    )
    assert stripped == "y ~ x"
    assert (term.p, term.q, term.cov, term.time, term.group) == (1, 1, True, "t", "g")
    assert split_autocor_terms("y ~ ar(p = 2)")[0] == "y ~ 1"
    assert parse_autocor_call("ma(t, g, 3)").q == 3
    assert parse_autocor_call("sar(W, 'error')").sar_type == "error"
    assert parse_autocor_call("car(W, gr = s, type = 'bym2')").car_type == "bym2"
    for text, message in [
        ("y ~ x + arma(p = 0, q = 0)", "At least one"),
        ("y ~ x + ar(p = -1)", "non-negative"),
        ("y ~ x + ar(p = 2, cov = TRUE)", "maximal order one"),
        ("y ~ x:ar(p = 1)", "standalone"),
        ("y ~ x + ar(p = 1) + ma(q = 1)", "one term per autocorrelation class"),
        ("y ~ x + ar(p = 1) + cosy()", "one time-series term"),
        ("y ~ x + ar(lag = 1)", "unknown argument"),
        ("y ~ x + unstr(t)", "missing"),
        ("y ~ x + sar()", "'M' is missing"),
        ("y ~ x + car(W, type = 'foo')", "type must be"),
        ("y ~ x + cosy() + fcor(V)", "one covariance matrix"),
        ("y ~ x + ar(p = 1, cov = TRUE) + sar(W)", "one covariance matrix"),
    ]:
        with pytest.raises(ValueError, match=message):
            split_autocor_terms(text)
    # A conditional ARMA term holds no covariance, so brms admits it beside
    # sar or fcor.
    assert len(split_autocor_terms("y ~ x + ar(p = 1) + sar(W)")[1]) == 2


def test_series_coding_matches_brms_standata():
    layout = _layout("arma(time = t, gr = g, p = 2, q = 1)")
    sorted_y = FRAME["y"].to_numpy()[list(layout.order)]
    np.testing.assert_array_equal(sorted_y, BRMS_SORTED_Y)
    assert (layout.starts, layout.lengths) == ((0, 4), (4, 3))
    lag = [
        min(position + 1, 2) if position + 1 < length else 0
        for length in layout.lengths
        for position in range(length)
    ]
    assert lag == BRMS_J_LAG
    unstr = _layout("unstr(t, g)")
    assert unstr.time_index == (0, 1, 2, 3, 0, 1, 2) and unstr.times == 4
    with pytest.raises(ValueError, match="unique"):
        _layout("ar(time = t, gr = g)", FRAME.assign(t=[1, 1, 2, 2, 1, 4, 3]))


@pytest.mark.parametrize("student_law", [False, True])
def test_conditional_arma_matches_brms_stan_loop_with_derivatives(student_law):
    layout = _layout("arma(time = t, gr = g, p = 2, q = 1)", student_law=student_law)
    eta = _t([0.2, -0.1, 0.4, 0.3, -0.5, 0.1, 0.0], grad=True)
    sigma = _t(0.8, grad=True)
    ar = _t([0.5, -0.3], grad=True)
    ma = _t([0.25], grad=True)
    nu = _t(4.5, grad=True)
    y = _t(FRAME["y"].tolist())
    law = ResidualResponse(
        layout,
        eta,
        sigma.expand(7),
        {"ar_1": ar[0], "ar_2": ar[1], "ma_1": ma[0]},
        nu if student_law else None,
    )
    order = list(layout.order)
    groups = FRAME["g"].to_numpy()[order]
    mu, expected = _stan_arma(
        eta[order], y[order], sigma, ar, ma, groups, nu if student_law else None
    )
    actual = law.log_prob(y)[order]
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        law.loo_log_prob(y)[order], expected, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(law.linear_predictor[order], mu, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(law.mean[order], mu, rtol=RTOL, atol=ATOL)
    parameters = (eta, sigma, ar, ma, *((nu,) if student_law else ()))
    first = torch.autograd.grad(actual.sum(), parameters, create_graph=True)
    reference = torch.autograd.grad(expected.sum(), parameters, create_graph=True)
    for a, b in zip(first, reference, strict=True):
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)
    second = torch.autograd.grad(first[2].sum(), parameters, retain_graph=True)
    second_ref = torch.autograd.grad(reference[2].sum(), parameters)
    for a, b in zip(second, second_ref, strict=True):
        torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL)


def test_conditional_arma_rejects_the_variant_whose_error_includes_ar():
    layout = _layout("arma(time = t, gr = g, p = 1, q = 1)")
    eta = _t([0.0] * 7)
    y = _t(FRAME["y"].tolist())
    law = ResidualResponse(
        layout, eta, _t([1.0] * 7), {"ar_1": _t(0.6), "ma_1": _t(0.4)}
    )
    order = list(layout.order)
    wrong = []
    err_prev = 0.0
    for position, row in enumerate(order):
        if position in (0, 4):
            err_prev = 0.0
        mean = 0.6 * err_prev + 0.4 * err_prev
        wrong.append(norm.logpdf(float(y[row]), mean, 1.0))
        err_prev = float(y[row]) - mean
    assert not np.allclose(law.log_prob(y)[order].numpy(), wrong)


def _brms_arma1(ar, ma, n):
    """Transcribe brms's R get_cor_matrix_arma1."""
    out = np.zeros((n, n))
    gamma0 = (1 + ma**2 + 2 * ar * ma) / (1 - ar**2)
    gamma1 = (1 + ar * ma) * (ar + ma) / (1 - ar**2)
    for i in range(n):
        out[i, i] = gamma0
        for j in range(i):
            out[i, j] = out[j, i] = gamma1 * ar ** (i - j - 1)
    return out


def test_transcribed_brms_correlations_match_pinned_r_output():
    np.testing.assert_allclose(
        _brms_arma1(0.4, -0.3, 3), BRMS_ARMA1, rtol=RTOL, atol=ATOL
    )
    np.testing.assert_allclose(_brms_arma1(0.4, 0.0, 3), BRMS_AR1, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        _brms_arma1(0.0, -0.3, 3), BRMS_MA1, rtol=RTOL, atol=ATOL
    )


def _block_covariance(kind, n, sigma):
    if kind == "arma":
        return sigma**2 * _brms_arma1(0.4, -0.3, n)
    if kind == "ar":
        return sigma**2 * _brms_arma1(0.4, 0.0, n)
    if kind == "ma":
        return sigma**2 * _brms_arma1(0.0, -0.3, n)
    return sigma**2 * (0.7 * np.eye(n) + 0.3)


def _loo_oracle(y, mean, cov, nu=None):
    """Condition each coordinate on the others with explicit Schur complements."""
    out = []
    for i in range(len(y)):
        rest = [j for j in range(len(y)) if j != i]
        inverse = np.linalg.inv(cov[np.ix_(rest, rest)])
        gain = cov[i, rest] @ inverse
        location = mean[i] + gain @ (y[rest] - mean[rest])
        variance = cov[i, i] - gain @ cov[rest, i]
        if nu is None:
            out.append(norm.logpdf(y[i], location, math.sqrt(variance)))
        else:
            quadratic = (y[rest] - mean[rest]) @ inverse @ (y[rest] - mean[rest])
            degrees = nu + len(rest)
            scale = math.sqrt(variance * (nu + quadratic) / degrees)
            out.append(student.logpdf(y[i], degrees, location, scale))
    return np.array(out)


@pytest.mark.parametrize("student_law", [False, True])
@pytest.mark.parametrize(
    "term,coefficients,kind",
    [
        (
            "arma(time = t, gr = g, p = 1, q = 1, cov = TRUE)",
            {"ar_1": 0.4, "ma_1": -0.3},
            "arma",
        ),
        ("ar(time = t, gr = g, cov = TRUE)", {"ar_1": 0.4}, "ar"),
        ("ma(time = t, gr = g, cov = TRUE)", {"ma_1": -0.3}, "ma"),
        ("cosy(time = t, gr = g)", {"cosy": 0.3}, "cosy"),
    ],
)
def test_covariance_structures_match_scipy_per_series_and_loo(
    term, coefficients, kind, student_law
):
    layout = _layout(term, student_law=student_law)
    eta = np.array([0.2, -0.1, 0.4, 0.3, -0.5, 0.1, 0.0])
    sigma, nu = 0.9, 5.0
    y = FRAME["y"].to_numpy()
    law = ResidualResponse(
        layout,
        _t(eta),
        _t([sigma] * 7),
        {name: _t(value) for name, value in coefficients.items()},
        _t(nu) if student_law else None,
    )
    expected = 0.0
    loo = np.zeros(7)
    for start, length in zip(layout.starts, layout.lengths, strict=True):
        rows = list(layout.order[start : start + length])
        cov = _block_covariance(kind, length, sigma)
        if student_law:
            expected += multivariate_t.logpdf(y[rows], eta[rows], cov, df=nu)
        else:
            expected += multivariate_normal.logpdf(y[rows], eta[rows], cov)
        loo[rows] = _loo_oracle(y[rows], eta[rows], cov, nu if student_law else None)
    scores = law.log_prob(_t(y))
    np.testing.assert_allclose(float(scores.sum()), expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        law.loo_log_prob(_t(y)).numpy(), loo, rtol=RTOL, atol=ATOL
    )
    torch.testing.assert_close(law.mean, _t(eta))


@pytest.mark.parametrize("student_law", [False, True])
def test_unstructured_and_fixed_covariances_condition_missing_time_levels(student_law):
    layout = _layout("unstr(t, g)", student_law=student_law)
    correlation = np.array(
        [
            [1.0, 0.3, 0.1, -0.2],
            [0.3, 1.0, 0.25, 0.05],
            [0.1, 0.25, 1.0, 0.4],
            [-0.2, 0.05, 0.4, 1.0],
        ]
    )
    sigma = np.array([0.7, 1.1, 0.9, 1.3, 0.8, 1.0, 1.2])
    eta = np.zeros(7)
    y = FRAME["y"].to_numpy()
    law = ResidualResponse(
        layout,
        _t(eta),
        _t(sigma),
        {},
        _t(4.0) if student_law else None,
        _t(np.linalg.cholesky(correlation)),
    )
    expected = 0.0
    for start, length in zip(layout.starts, layout.lengths, strict=True):
        rows = list(layout.order[start : start + length])
        levels = list(layout.time_index[start : start + length])
        cov = np.outer(sigma[rows], sigma[rows]) * correlation[np.ix_(levels, levels)]
        expected += (
            multivariate_t.logpdf(y[rows], eta[rows], cov, df=4.0)
            if student_law
            else multivariate_normal.logpdf(y[rows], eta[rows], cov)
        )
    np.testing.assert_allclose(
        float(law.log_prob(_t(y)).sum()), expected, rtol=RTOL, atol=ATOL
    )
    matrix = np.array([[0.5, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 0.2]])
    small = FRAME.iloc[:3].reset_index(drop=True)
    fixed = _layout("fcor(M)", small, student_law=student_law, matrix=matrix)
    law = ResidualResponse(
        fixed, _t([0.1, 0.0, -0.1]), _t([1.3] * 3), {}, _t(4.0) if student_law else None
    )
    cov = 1.3**2 * matrix
    expected = (
        multivariate_t.logpdf(small.y, [0.1, 0.0, -0.1], cov, df=4.0)
        if student_law
        else multivariate_normal.logpdf(small.y, [0.1, 0.0, -0.1], cov)
    )
    np.testing.assert_allclose(
        float(law.log_prob(_t(small.y)).sum()), expected, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize("student_law", [False, True])
@pytest.mark.parametrize("kind", ["lag", "error"])
def test_sar_matches_dense_scipy_without_dense_scoring(kind, student_law, monkeypatch):
    frame = pd.DataFrame(
        {"y": [0.2, -0.5, 1.1, 0.4, -0.3], "x": [0.1, 0.4, -0.6, 1.2, 0.3]}
    )
    layout = _layout(
        f"sar(M, type = '{kind}')", frame, student_law=student_law, matrix=PATH
    )
    assert layout.eigenvalues is not None
    np.testing.assert_allclose(layout.eigenvalues[[0, -1]], [-1.0, 1.0], atol=1e-12)
    eta = np.array([0.3, -0.2, 0.1, 0.5, 0.0])
    rho, sigma, nu = 0.35, 0.8, 6.0
    y = frame.y.to_numpy()
    name = "lagsar" if kind == "lag" else "errorsar"
    rho_t = _t(rho, grad=True)

    def forbid(*args, **kwargs):
        raise AssertionError("SAR scoring formed a dense factorization")

    monkeypatch.setattr(torch.linalg, "cholesky", forbid)
    monkeypatch.setattr(torch.linalg, "inv", forbid)
    monkeypatch.setattr(torch.linalg, "solve", forbid)
    law = ResidualResponse(
        layout,
        _t(eta),
        _t([sigma] * 5),
        {name: rho_t},
        _t(nu) if student_law else None,
    )
    score = law.log_prob(_t(y)).sum()
    loo = law.loo_log_prob(_t(y))
    (gradient,) = torch.autograd.grad(score, rho_t)
    score = score.detach()
    monkeypatch.undo()
    operator = np.eye(5) - rho * PATH
    mean = np.linalg.solve(operator, eta) if kind == "lag" else eta
    cov = sigma**2 * np.linalg.inv(operator.T @ operator)
    oracle = (
        (
            lambda r, m: multivariate_t.logpdf(
                y,
                m,
                sigma**2
                * np.linalg.inv((np.eye(5) - r * PATH).T @ (np.eye(5) - r * PATH)),
                df=nu,
            )
        )
        if student_law
        else (
            lambda r, m: multivariate_normal.logpdf(
                y,
                m,
                sigma**2
                * np.linalg.inv((np.eye(5) - r * PATH).T @ (np.eye(5) - r * PATH)),
            )
        )
    )
    np.testing.assert_allclose(float(score), oracle(rho, mean), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(
        loo.detach().numpy(),
        _loo_oracle(y, mean, cov, nu if student_law else None),
        rtol=RTOL,
        atol=ATOL,
    )
    step = 1e-6

    def shifted(r):
        op = np.eye(5) - r * PATH
        return oracle(r, np.linalg.solve(op, eta) if kind == "lag" else eta)

    numeric = (shifted(rho + step) - shifted(rho - step)) / (2 * step)
    np.testing.assert_allclose(float(gradient), numeric, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(law.mean.detach().numpy(), mean, rtol=RTOL, atol=ATOL)


def test_sar_rejects_complex_spectra_and_singular_parameters():
    rotation = np.array([[0.0, 1.0], [-1.0, 0.0]])
    with pytest.raises(ValueError, match="real eigenvalues"):
        sar_weights(rotation)
    frame = pd.DataFrame({"y": [0.2, -0.5, 1.1, 0.4, -0.3], "x": [0.0] * 5})
    layout = _layout("sar(M)", frame, matrix=PATH)
    rho = _t([0.5, 1.0])[:, None].expand(2, 5)
    law = ResidualResponse(
        layout, _t([0.0] * 5).expand(2, 5), _t([1.0] * 5), {"lagsar": rho}
    )
    scores = law.log_prob(_t(frame.y.tolist()))
    assert torch.isfinite(scores[0]).all() and (scores[1] == -torch.inf).all()
    # The largest eigenvalue of the row-stochastic PATH is exactly one, and its
    # computed value lies within rounding of one on either side. The bound is
    # 1e3 * 5 * eps = 1.1e-12: every rho closer to one than that is singular,
    # whichever side the platform rounds to, and rho further away is scored.
    assert spectrum_tolerance(layout.eigenvalues) == pytest.approx(
        5e3 * np.finfo(np.float64).eps, rel=1e-12
    )
    near = [np.nextafter(1.0, 0.0), np.nextafter(1.0, 2.0), 1 - 1e-13, 1 - 1e-11]
    law = ResidualResponse(
        layout,
        _t([0.0] * 5).expand(4, 5),
        _t([1.0] * 5),
        {"lagsar": _t(near)[:, None].expand(4, 5)},
    )
    scores = law.log_prob(_t(frame.y.tolist()))
    assert (scores[:3] == -torch.inf).all() and torch.isfinite(scores[3]).all()


def test_cosy_sampling_reproduces_the_covariance_and_student_radius_per_series():
    layout = _layout("cosy(time = t, gr = g)")
    law = ResidualResponse(layout, _t([0.0] * 7), _t([1.5] * 7), {"cosy": _t(0.3)})
    torch.manual_seed(5)
    draws = law.sample(torch.Size((40000,))).numpy()
    rows = list(layout.order[:4])
    covariance = np.cov(draws[:, rows].T)
    # Monte Carlo standard error of a covariance entry is below 0.02 here.
    np.testing.assert_allclose(covariance, 2.25 * (0.7 * np.eye(4) + 0.3), atol=0.08)
    other = list(layout.order[4:])
    np.testing.assert_allclose(
        np.cov(draws[:, [rows[0], other[0]]].T)[0, 1], 0.0, atol=0.08
    )


def test_missing_arma_history_is_predicted_sequentially():
    frame = FRAME.assign(y=[0.3, -1.2, np.nan, 2.1, -0.4, np.nan, 0.9])
    layout = _layout("ar(time = t, gr = g)", frame)
    law = ResidualResponse(layout, _t([0.0] * 7), _t([1e-9] * 7), {"ar_1": _t(0.5)})
    mean = law.mean.numpy()
    # Group b at t = 2 follows t = 1 (y = -0.4); a vanishing scale makes the
    # drawn history equal its conditional mean.
    assert mean[2] == pytest.approx(-0.2, abs=1e-8)
    assert mean[0] == pytest.approx(0.5 * -0.2, abs=1e-8)
    with pytest.raises(ValueError, match="every response"):
        law._arma_means(law._history(), fill=False)


def _series_frame(seed=3, groups=2, length=6):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(groups):
        e = 0.0
        for t in range(length):
            x = rng.normal()
            e = 0.5 * e + rng.normal(scale=0.5)
            rows.append({"g": f"s{g}", "t": t, "x": x, "y": 0.3 + 0.8 * x + e})
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
def test_formula_arma_fits_with_every_method_and_replays_brms_mu(method):
    data = _series_frame()
    settings = (
        {"method": "svi", "num_steps": 30}
        if method == "svi"
        else {"method": method, "num_warmup": 4, "num_samples": 4, "num_chains": 1}
    )
    fitted = fit("y ~ x + ar(time = t, gr = g, p = 2)", data=data, seed=3, **settings)
    draws = fitted.draws(num_samples=3, seed=1)
    mu = fitted.posterior_linpred(num_samples=3, seed=1)
    scores = fitted.log_likelihood(num_samples=3, seed=1)["y"]
    order = data.sort_values(["g", "t"]).index.to_numpy()
    groups = data.g.to_numpy()[order]
    y = _t(data.y.to_numpy()[order])
    for chain in range(mu.shape[0]):
        for draw in range(mu.shape[1]):
            value = {k: v[chain, draw].double() for k, v in draws.items()}
            eta = value["intercept"] + value["beta_x"] * _t(data.x.to_numpy()[order])
            ar = torch.stack([value["ar_1"], value["ar_2"]]).reshape(2)
            expected_mu, expected = _stan_arma(
                eta, y, value["sigma"], ar, eta.new_zeros(0), groups
            )
            torch.testing.assert_close(
                mu[chain, draw, order].double(),
                expected_mu,
                rtol=FIT_RTOL,
                atol=FIT_ATOL,
            )
            torch.testing.assert_close(
                scores[chain, draw, order].double(),
                expected,
                rtol=FIT_RTOL,
                atol=FIT_ATOL,
            )
    assert fitted.posterior_predict(num_samples=3, seed=2).shape == mu.shape
    epred = fitted.posterior_epred(num_samples=3, seed=1)
    torch.testing.assert_close(epred, mu, rtol=RTOL, atol=ATOL)
    plain = fitted.posterior_epred(num_samples=3, seed=1, incl_autocor=False)
    eta = fitted.posterior_linpred(num_samples=3, seed=1, incl_autocor=False)
    torch.testing.assert_close(plain, eta, rtol=RTOL, atol=ATOL)


def test_formula_prediction_new_series_priors_update_and_artifacts(tmp_path):
    data = _series_frame()
    prior = get_prior("y ~ x + arma(time = t, gr = g, cov = TRUE)", data=data)
    assert "ar_1" in prior and "ma_1" in prior
    fitted = fit(
        "y ~ x + arma(time = t, gr = g, cov = TRUE)",
        data=data,
        method="svi",
        num_steps=20,
        priors=PriorSpec(distribution="Normal(0.0, 0.3)", class_="ar"),
        seed=4,
    )
    assert "Restrict" in fitted.qvr_source
    fresh = data[data.g == "s0"].assign(g="new").copy()
    fresh.loc[fresh.t >= 4, "y"] = np.nan
    predicted = fitted.posterior_predict(new_data=fresh, num_samples=4, seed=2)
    assert predicted.shape == (1, 4, len(fresh)) and torch.isfinite(predicted).all()
    scored = fitted.log_likelihood(new_data=data.iloc[:5], num_samples=4, seed=2)["y"]
    assert scored.shape == (1, 4, 5)
    refitted = fitted.update(num_steps=5)
    assert refitted.posterior_epred(num_samples=2).shape == (1, 2, len(data))
    path = tmp_path / "autocor.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.log_likelihood(num_samples=4, seed=2)["y"],
        fitted.log_likelihood(num_samples=4, seed=2)["y"],
    )


def test_grouped_folds_refit_series_and_score_held_out_groups():
    data = _series_frame(groups=3, length=4)
    fitted = fit(
        "y ~ x + ar(time = t, gr = g)", data=data, method="svi", num_steps=5, seed=1
    )
    assignments = tuple(int(label[1:]) for label in data.g)
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=2,
    )
    result = cross_validate(FoldPlan(assignments=assignments), callback)
    assert torch.isfinite(result.log_predictive_density).all()
    assert result.log_predictive_density.numel() == len(data)


def test_formula_rejections_follow_brms_restrictions():
    data = _series_frame()
    cases = [
        (
            {"family": "poisson", "formula": "y ~ x + ma(time = t, gr = g)"},
            "MA structures",
        ),
        (
            {
                "family": "poisson",
                "formula": "y ~ x + fcor(M)",
                "data2": {"M": np.eye(len(data))},
            },
            "FCOR terms are not implemented",
        ),
        ({"distributional": {"nu": "~ x"}, "family": "student_t"}, "scalar 'nu'"),
        ({"data2": None, "formula": "y ~ x + fcor(M)"}, "data2"),
        (
            {"formula": "y ~ x + car(M, gr = g)", "data2": {"M": np.eye(2)}},
            "Row names",
        ),
        (
            {"weights": "x", "formula": "y ~ x + cosy(time = t, gr = g)"},
            "Invalid addition arguments",
        ),
    ]
    for options, message in cases:
        options = dict(options)
        formula = options.pop("formula", "y ~ x + ar(time = t, gr = g)")
        with pytest.raises(ValueError, match=message):
            fit(
                formula,
                data=data.assign(x=data.x.abs(), y=data.y.abs().round()),
                method="svi",
                num_steps=1,
                **options,
            )
    fitted = fit("y ~ x + unstr(t, g)", data=data, method="svi", num_steps=2)
    with pytest.raises(ValueError, match="new time points"):
        fitted.posterior_epred(new_data=data.assign(t=data.t + 10), num_samples=1)


@pytest.mark.parametrize("method", ["svi", "nuts"])
def test_formula_cosy_student_scores_brms_loo_from_replayed_draws(method):
    data = _series_frame(seed=8, groups=2, length=4)
    settings = (
        {"method": "svi", "num_steps": 20}
        if method == "svi"
        else {"method": "nuts", "num_warmup": 4, "num_samples": 3, "num_chains": 1}
    )
    fitted = fit(
        "y ~ x + cosy(time = t, gr = g)",
        data=data,
        family="student_t",
        seed=2,
        **settings,
    )
    draws = fitted.draws(num_samples=3, seed=1)
    scores = fitted.log_likelihood(num_samples=3, seed=1)["y"]
    y = data.y.to_numpy()
    for draw in range(scores.shape[1]):
        value = {k: v[0, draw].double().numpy() for k, v in draws.items()}
        eta = value["intercept"] + value["beta_x"] * data.x.to_numpy()
        for group in ("s0", "s1"):
            rows = data.index[data.g == group].to_numpy()
            rows = rows[np.argsort(data.t.to_numpy()[rows])]
            n = len(rows)
            cov = value["sigma"] ** 2 * (
                (1 - value["cosy"]) * np.eye(n) + value["cosy"]
            )
            np.testing.assert_allclose(
                scores[0, draw, rows].double().numpy(),
                _loo_oracle(y[rows], eta[rows], cov, float(value["nu"])),
                rtol=FIT_RTOL,
                atol=FIT_ATOL,
            )


def test_formula_sar_fcor_unstr_fit_predict_and_restore(tmp_path):
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"x": rng.normal(size=5)})
    frame["y"] = 0.4 * frame.x + rng.normal(scale=0.3, size=5)
    lag = fit(
        "y ~ x + sar(W, type = 'lag')",
        data=frame,
        data2={"W": PATH},
        method="svi",
        num_steps=20,
        seed=1,
    )
    assert "lagsar" in get_prior("y ~ x + sar(W)", data=frame, data2={"W": PATH})
    draws = lag.draws(num_samples=2, seed=1)
    epred = lag.posterior_epred(num_samples=2, seed=1)
    for draw in range(2):
        value = {k: v[0, draw].double().numpy() for k, v in draws.items()}
        eta = value["intercept"] + value["beta_x"] * frame.x.to_numpy()
        expected = np.linalg.solve(np.eye(5) - value["lagsar"] * PATH, eta)
        np.testing.assert_allclose(
            epred[0, draw].double().numpy(), expected, rtol=FIT_RTOL, atol=FIT_ATOL
        )
    with pytest.raises(ValueError, match=r"data2\['W'\] over the new rows"):
        lag.posterior_predict(new_data=frame.iloc[:3], num_samples=1)
    independent = lag.posterior_predict(
        new_data=frame.iloc[:3], num_samples=2, incl_autocor=False
    )
    assert independent.shape == (1, 2, 3)
    path = tmp_path / "sar.pt"
    save_fit(lag, path)
    torch.testing.assert_close(
        load_fit(path).posterior_epred(num_samples=2, seed=1), epred
    )
    fixed = fit(
        "y ~ x + fcor(V)",
        data=frame,
        data2={"V": np.diag([0.5, 1.0, 1.5, 1.0, 0.5])},
        method="svi",
        num_steps=5,
    )
    assert fixed.log_likelihood(num_samples=2)["y"].shape == (1, 2, 5)
    series = _series_frame(groups=2, length=3)
    unstructured = fit("y ~ x + unstr(t, g)", data=series, method="svi", num_steps=5)
    assert "Lcortime" in unstructured.qvr_source
    assert unstructured.posterior_predict(num_samples=2).shape == (1, 2, 6)
