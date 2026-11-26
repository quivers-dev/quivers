"""Autocorrelation terms combined with response terms and with each other.

Three brms combinations are checked against independent references: known
standard errors added to a time-series covariance, censored and truncated
rows of the conditional ARMA formulation, and a conditional ARMA term beside
a SAR or fixed-covariance term. References are SciPy densities over dense
matrices, hand-written transcriptions of brms's Stan functions, and brms
output pinned by ``oracles/brms_autocor_combinations.R``.

Tolerances are fixed from floating-point error before comparison: 1e-10 for
Float64 densities and autograd derivatives, 1e-6 for central finite
differences with step 1e-5, and relative 1e-4 with absolute 1e-5 for fitted
formula programs, which run in Float32 through tail masses and linear solves.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import multivariate_normal, multivariate_t, norm, t as student_t
import torch

from quivers.diagnostics.cross_validation import FoldPlan, cross_validate
from quivers.formulas import bf, fit, formula_to_qvr, set_rescor
from quivers.formulas._autocor import split_autocor_terms
from quivers.formulas._residual_law import structure_layout
from quivers.formulas._residual_dependence import ResidualResponse
from quivers.inference.registry import LatentRegistry
from quivers.inference.mcmc.driver import MCMCResult
from quivers.modeling import load_fit, save_fit

RTOL = ATOL = 1e-10
FD_TOL = 1e-6
FIT_RTOL, FIT_ATOL = 1e-4, 1e-5
ORACLE = json.loads(
    (Path(__file__).parent / "oracles" / "brms_autocor_combinations.json").read_text()
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(ORACLE["data"])


def _matrix(name: str) -> np.ndarray:
    return np.asarray(ORACLE[name], dtype=np.float64)


def _t(values, grad=False):
    return torch.tensor(values, dtype=torch.float64, requires_grad=grad)


def _layout(text, frame, *, student=False, data2=None, se=None, history=None):
    terms = split_autocor_terms(f"y ~ x + {text}")[1]
    arma = next((term for term in terms if term.kind == "arma" and not term.cov), None)
    structural = [term for term in terms if term is not arma]
    return structure_layout(
        structural[0] if structural else arma,
        arma if structural else None,
        frame.reset_index(drop=True),
        student=student,
        history=history,
        time_levels=None,
        data2=data2,
        variance=None if se is None else np.square(se),
        caller="test",
    )[0]


def _series(frame):
    """Rows of each series in time order, as brms sorts them."""
    return [
        rows.sort_values("t").index.to_numpy()
        for _, rows in frame.groupby("g", sort=True)
    ]


def _correlation(kind, count, *, rho=0.0, theta=0.0, factor=None):
    """Transcribe brms's time-series correlation matrices."""
    lag = np.abs(np.subtract.outer(np.arange(count), np.arange(count)))
    if kind == "cosy":
        return np.where(lag == 0, 1.0, rho)
    if kind == "unstr":
        return (factor @ factor.T)[:count, :count]
    variance = (1 + theta**2 + 2 * rho * theta) / (1 - rho**2)
    first = (1 + rho * theta) * (rho + theta) / (1 - rho**2)
    return np.where(lag == 0, variance, first * rho ** np.clip(lag - 1, 0, None))


def _loo(y, mean, covariance, nu=None):
    """Leave-one-out conditional log densities from the dense covariance."""
    scores = []
    for row in range(len(y)):
        rest = [index for index in range(len(y)) if index != row]
        solve = np.linalg.solve(covariance[np.ix_(rest, rest)], covariance[rest, row])
        centered = y[rest] - mean[rest]
        location = mean[row] + solve @ centered
        variance = covariance[row, row] - covariance[row, rest] @ solve
        if nu is None:
            scores.append(norm.logpdf(y[row], location, math.sqrt(variance)))
            continue
        quadratic = centered @ np.linalg.solve(covariance[np.ix_(rest, rest)], centered)
        degrees = nu + len(rest)
        scale = math.sqrt(variance * (nu + quadratic) / degrees)
        scores.append(student_t.logpdf(y[row], degrees, location, scale))
    return np.array(scores)


SE_KINDS = [
    ("cosy(t, g)", "cosy", {"cosy": 0.35}),
    ("ar(t, g, cov = TRUE)", "arma", {"ar_1": 0.45}),
    ("ma(t, g, cov = TRUE)", "arma", {"ma_1": -0.3}),
    ("arma(t, g, cov = TRUE)", "arma", {"ar_1": 0.4, "ma_1": -0.25}),
    ("unstr(t, g)", "unstr", {}),
]
UNSTR_FACTOR = np.linalg.cholesky(
    np.array(
        [
            [1.0, 0.3, -0.2, 0.1],
            [0.3, 1.0, 0.4, 0.0],
            [-0.2, 0.4, 1.0, 0.25],
            [0.1, 0.0, 0.25, 1.0],
        ]
    )
)


@pytest.mark.parametrize("student_law", [False, True])
@pytest.mark.parametrize(("text", "kind", "coefficients"), SE_KINDS)
def test_known_errors_add_to_each_series_covariance(
    text, kind, coefficients, student_law
):
    # Drop one row so series have unequal lengths and one misses a time level.
    frame = _frame().drop(index=5).sample(frac=1.0, random_state=2)
    frame = frame.reset_index(drop=True)
    se = frame.s.to_numpy()
    layout = _layout(text, frame, student=student_law, se=se)
    eta = 0.2 + 0.7 * frame.x.to_numpy()
    scale = np.exp(-0.4 + 0.3 * frame.x.to_numpy())
    nu = 5.5 if student_law else None
    law = ResidualResponse(
        layout,
        _t(eta),
        _t(scale),
        {name: _t(value) for name, value in coefficients.items()},
        None if nu is None else _t(nu),
        _t(UNSTR_FACTOR) if kind == "unstr" else None,
    )
    y = frame.y.to_numpy()
    scores = law.log_prob(_t(y)).numpy()
    loo = law.loo_log_prob(_t(y)).numpy()
    total = 0.0
    levels = sorted(frame.t.unique())
    for rows in _series(frame):
        times = [levels.index(value) for value in frame.t.to_numpy()[rows]]
        correlation = _correlation(
            kind,
            len(rows),
            rho=coefficients.get("ar_1", coefficients.get("cosy", 0.0)),
            theta=coefficients.get("ma_1", 0.0),
            factor=UNSTR_FACTOR,
        )
        if kind == "unstr":
            correlation = (UNSTR_FACTOR @ UNSTR_FACTOR.T)[np.ix_(times, times)]
        # brms normal_time_het_se_lpdf: sigma R sigma' + diag(se^2).
        covariance = np.outer(scale[rows], scale[rows]) * correlation + np.diag(
            se[rows] ** 2
        )
        expected = (
            multivariate_normal(eta[rows], covariance).logpdf(y[rows])
            if nu is None
            else multivariate_t(eta[rows], covariance, df=nu).logpdf(y[rows])
        )
        np.testing.assert_allclose(scores[rows].sum(), expected, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(
            loo[rows], _loo(y[rows], eta[rows], covariance, nu), rtol=1e-9, atol=1e-9
        )
        # Folding se into the scale instead correlates the measurement error.
        folded = np.sqrt(scale[rows] ** 2 + se[rows] ** 2)
        wrong = multivariate_normal(
            eta[rows], np.outer(folded, folded) * correlation, allow_singular=True
        ).logpdf(y[rows])
        if nu is None:
            assert abs(wrong - expected) > 1e-3
        total += expected
    np.testing.assert_allclose(scores.sum(), total, rtol=RTOL, atol=ATOL)


def test_known_error_covariance_has_exact_first_and_second_derivatives():
    frame = _frame()
    se = _t(frame.s.to_numpy())
    layout = _layout("arma(t, g, cov = TRUE)", frame, se=se.numpy())
    y, x = _t(frame.y.to_numpy()), _t(frame.x.to_numpy())
    blocks = [torch.as_tensor(rows) for rows in _series(frame)]

    def model(parameters):
        intercept, slope, log_sigma, phi, theta = parameters
        law = ResidualResponse(
            layout,
            intercept + slope * x,
            log_sigma.exp().expand(len(frame)),
            {"ar_1": phi, "ma_1": theta},
        )
        return law.log_prob(y).sum()

    def reference(parameters):
        intercept, slope, log_sigma, phi, theta = parameters
        total = y.new_zeros(())
        for rows in blocks:
            count = len(rows)
            lag = (torch.arange(count)[:, None] - torch.arange(count)[None, :]).abs()
            variance = (1 + theta**2 + 2 * phi * theta) / (1 - phi**2)
            first = (1 + phi * theta) * (phi + theta) / (1 - phi**2)
            correlation = torch.where(
                lag == 0, variance, first * phi ** (lag - 1).clamp_min(0)
            )
            covariance = log_sigma.exp() ** 2 * correlation + torch.diag(se[rows] ** 2)
            total = total + torch.distributions.MultivariateNormal(
                intercept + slope * x[rows], covariance_matrix=covariance
            ).log_prob(y[rows])
        return total

    point = _t([0.2, 0.6, -0.3, 0.4, -0.2])
    torch.testing.assert_close(model(point), reference(point), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(model, point),
        torch.autograd.functional.jacobian(reference, point),
        rtol=1e-9,
        atol=1e-9,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(model, point),
        torch.autograd.functional.hessian(reference, point),
        rtol=1e-8,
        atol=1e-8,
    )


def test_known_error_draws_have_the_series_covariance():
    frame = _frame()
    se = frame.s.to_numpy()
    law = ResidualResponse(
        _layout("cosy(t, g)", frame, se=se),
        _t([0.0] * len(frame)),
        _t([0.8] * len(frame)),
        {"cosy": _t(0.5)},
    )
    torch.manual_seed(5)
    draws = law.sample(torch.Size([40000])).numpy()
    rows = _series(frame)[0]
    expected = 0.64 * _correlation("cosy", len(rows), rho=0.5) + np.diag(se[rows] ** 2)
    # Monte Carlo error of a covariance entry is below 4 * 1.2 / sqrt(40000).
    np.testing.assert_allclose(np.cov(draws[:, rows].T), expected, atol=0.025)
    other = _series(frame)[1]
    assert abs(np.cov(draws[:, rows[0]], draws[:, other[0]])[0, 1]) < 0.025


def _brms_arma_mu(eta, y, ar, ma, groups):
    """Transcribe brms's Stan ARMA loop on rows sorted by group and time."""
    count = len(y)
    lags = max(len(ar), len(ma), 1)
    zero = eta[0] * 0
    err_matrix = [[zero for _ in range(lags)] for _ in range(count + 1)]
    mu = list(eta)
    err = [zero] * count
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
    return mu


CODES = {"none": 0, "left": -1, "right": 1, "interval": 2}


def _event_oracle(y, upper, status, mu, sigma, nu, lower_bound, upper_bound):
    """Score censored and truncated rows with SciPy tails.

    Without truncation these are brms's statements. A row that is both
    censored and truncated scores the mass of its censored event inside the
    truncation interval, the shared response-event semantics; brms divides
    the whole tail mass by the interval mass.
    """
    law = norm(mu, sigma) if nu is None else student_t(nu, mu, sigma)
    low = -np.inf if lower_bound is None else lower_bound
    high = np.inf if upper_bound is None else upper_bound
    start = np.where((status == 1) | (status == 2), y, -np.inf)
    stop = np.where(status == -1, y, np.where(status == 2, upper, np.inf))
    start, stop = np.maximum(start, low), np.minimum(stop, high)
    event = np.log(law.cdf(stop) - law.cdf(start))
    if lower_bound is None:
        # Tails by their accurate complements, as brms writes them.
        event = np.where(
            status == 1, law.logsf(y), np.where(status == -1, law.logcdf(y), event)
        )
    scores = np.where(status == 0, law.logpdf(y), event)
    if lower_bound is not None:
        scores = scores - np.log(law.cdf(high) - law.cdf(low))
    return scores


@pytest.mark.parametrize("student_law", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
@pytest.mark.parametrize(
    ("text", "ar", "ma"),
    [
        ("ar(t, g)", [0.5], []),
        ("arma(t, g, p = 2, q = 1)", [0.4, -0.2], [0.3]),
        ("ma(t, g, q = 2)", [], [0.35, 0.1]),
    ],
)
def test_censored_and_truncated_conditional_arma_matches_the_brms_loop(
    text, ar, ma, truncated, student_law
):
    frame = _frame()
    shuffled = frame.sample(frac=1.0, random_state=4).reset_index(drop=True)
    order = shuffled.sort_values(["g", "t"]).index.to_numpy()
    y = shuffled.y.to_numpy()
    status = shuffled.c.map(CODES).to_numpy()
    upper = shuffled.y2.to_numpy()
    eta = 0.1 + 0.6 * shuffled.x.to_numpy()
    sigma, nu = 0.9, (4.5 if student_law else None)
    bounds = (-3.0, 4.0) if truncated else (None, None)
    layout = _layout(text, shuffled, student=student_law, history=y)
    coefficients = {f"ar_{k + 1}": _t(v) for k, v in enumerate(ar)}
    coefficients |= {f"ma_{k + 1}": _t(v) for k, v in enumerate(ma)}

    def score(eta_value, sigma_value):
        law = ResidualResponse(
            layout,
            _t(eta_value),
            _t([sigma_value] * len(y)),
            coefficients,
            None if nu is None else _t(nu),
        )
        return law.recorded_event_log_prob(
            _t(y),
            status=torch.as_tensor(status),
            interval_upper=_t(upper),
            trunc_lower=bounds[0],
            trunc_upper=bounds[1],
        ).numpy()

    def oracle(eta_value, sigma_value, endpoint=y):
        mu = np.array(
            _brms_arma_mu(
                list(eta_value[order]),
                endpoint[order],
                ar,
                ma,
                shuffled.g.to_numpy()[order],
            ),
            dtype=np.float64,
        )
        result = np.empty(len(y))
        result[order] = _event_oracle(
            y[order], upper[order], status[order], mu, sigma_value, nu, *bounds
        )
        return result

    np.testing.assert_allclose(score(eta, sigma), oracle(eta, sigma), rtol=1e-9)
    # brms computes residuals from the recorded lower endpoint, not the upper.
    wrong = oracle(eta, sigma, np.where(status == 2, upper, y))
    assert np.abs(wrong - oracle(eta, sigma)).max() > 1e-3
    # First derivatives by central differences of the independent oracle.
    step = 1e-5
    eta_tensor = _t(eta, grad=True)
    sigma_tensor = _t(sigma, grad=True)
    law = ResidualResponse(
        layout,
        eta_tensor,
        sigma_tensor.expand(len(y)),
        coefficients,
        None if nu is None else _t(nu),
    )
    law.recorded_event_log_prob(
        _t(y),
        status=torch.as_tensor(status),
        interval_upper=_t(upper),
        trunc_lower=bounds[0],
        trunc_upper=bounds[1],
    ).sum().backward()
    assert sigma_tensor.grad is not None and eta_tensor.grad is not None
    expected_sigma = (
        oracle(eta, sigma + step).sum() - oracle(eta, sigma - step).sum()
    ) / (2 * step)
    np.testing.assert_allclose(sigma_tensor.grad.item(), expected_sigma, atol=FD_TOL)
    for row in (0, 5):
        shift = np.zeros(len(y))
        shift[row] = step
        expected = (
            oracle(eta + shift, sigma).sum() - oracle(eta - shift, sigma).sum()
        ) / (2 * step)
        np.testing.assert_allclose(eta_tensor.grad[row].item(), expected, atol=FD_TOL)


def test_censored_gaussian_arma_has_exact_second_derivatives():
    frame = _frame()
    y, x = _t(frame.y.to_numpy()), _t(frame.x.to_numpy())
    status = torch.as_tensor(frame.c.map(CODES).to_numpy())
    upper = _t(frame.y2.to_numpy())
    layout = _layout("arma(t, g)", frame, history=frame.y.to_numpy())
    groups = frame.g.to_numpy()

    def model(parameters):
        intercept, slope, log_sigma, phi, theta = parameters
        law = ResidualResponse(
            layout,
            intercept + slope * x,
            log_sigma.exp().expand(len(frame)),
            {"ar_1": phi, "ma_1": theta},
        )
        return law.recorded_event_log_prob(
            y, status=status, interval_upper=upper, trunc_lower=-3.0, trunc_upper=4.0
        ).sum()

    def reference(parameters):
        intercept, slope, log_sigma, phi, theta = parameters
        mu = torch.stack(
            _brms_arma_mu(list(intercept + slope * x), y, [phi], [theta], groups)
        )
        law = torch.distributions.Normal(mu, log_sigma.exp())
        low, high = _t(-3.0), _t(4.0)
        # The censored event inside the truncation interval.
        start = torch.where((status == 1) | (status == 2), y, low)
        stop = torch.where(status == -1, y, torch.where(status == 2, upper, high))
        event = (law.cdf(stop) - law.cdf(start)).log()
        scores = torch.where(status == 0, law.log_prob(y), event)
        mass = law.cdf(high) - law.cdf(low)
        return (scores - mass.log()).sum()

    point = _t([0.2, 0.6, -0.1, 0.4, 0.3])
    torch.testing.assert_close(model(point), reference(point), rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(model, point),
        torch.autograd.functional.jacobian(reference, point),
        rtol=1e-7,
        atol=1e-7,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(model, point),
        torch.autograd.functional.hessian(reference, point),
        rtol=1e-6,
        atol=1e-6,
    )


def test_truncated_arma_prediction_draws_missing_history_inside_the_bounds():
    frame = _frame().assign(y=lambda rows: rows.y.where(rows.t < 3))
    layout = _layout("ar(t, g)", frame, history=frame.y.to_numpy())
    law = ResidualResponse(
        layout,
        _t([0.0] * len(frame)).expand(200, len(frame)),
        _t([3.0] * len(frame)),
        {"ar_1": _t(0.9)},
    )
    torch.manual_seed(1)
    truncated = law.componentwise_truncation(-0.5, 0.75)
    draws = truncated.sample()
    assert draws.shape == (200, len(frame))
    assert bool(((draws >= -0.5) & (draws <= 0.75)).all())
    # A mean beyond 0.9 * 0.75 in magnitude needs history outside the bounds.
    means = law.linear_predictor
    late = torch.as_tensor(frame.t.to_numpy() == 4)
    assert bool((means[:, late].abs() <= 0.9 * 0.75 + 1e-12).all())
    for kind in ("cosy(t, g)", "ar(t, g, cov = TRUE)"):
        covariance = ResidualResponse(
            _layout(kind, _frame()),
            _t([0.0] * 12),
            _t([1.0] * 12),
            {"cosy": _t(0.2), "ar_1": _t(0.2)},
        )
        with pytest.raises(ValueError, match="Invalid addition arguments"):
            covariance.componentwise_truncation(-1.0, 1.0)
        with pytest.raises(ValueError, match="Invalid addition arguments"):
            covariance.recorded_event_log_prob(_t([0.0] * 12), status=1)


def _sar_oracle(y, mu, sigma, rho, weights, kind, nu=None):
    """Transcribe brms's SAR densities, with their dropped constants restored."""
    count = len(y)
    operator = np.eye(count) - rho * weights
    half = operator @ y - mu if kind == "lag" else operator @ (y - mu)
    log_det = np.log1p(-rho * np.linalg.eigvals(weights).real).sum()
    quadratic = half @ half / sigma**2
    if nu is None:
        return (
            -0.5 * count * math.log(2 * math.pi)
            - count * math.log(sigma)
            + log_det
            - 0.5 * quadratic
        )
    return (
        -0.5 * count * math.log(nu * math.pi)
        + math.lgamma((nu + count) / 2)
        - math.lgamma(nu / 2)
        - count * math.log(sigma)
        + log_det
        - (nu + count) / 2 * math.log1p(quadratic / nu)
    )


COMBINATIONS = [
    ("ar(t, g) + sar(W)", [0.45], [], "lag"),
    ("arma(t, g) + sar(W, type = 'error')", [0.3], [-0.25], "error"),
    ("ma(t, g, q = 2) + sar(W, type = 'lag')", [], [0.3, 0.15], "lag"),
    ("ar(t, g, p = 2) + fcor(V)", [0.4, -0.15], [], "fcor"),
    ("ma(t, g) + fcor(V)", [], [0.35], "fcor"),
]


@pytest.mark.parametrize("student_law", [False, True])
@pytest.mark.parametrize(("text", "ar", "ma", "kind"), COMBINATIONS)
def test_conditional_arma_locates_sar_and_fixed_covariance_laws(
    text, ar, ma, kind, student_law
):
    frame = _frame()
    weights, fixed = _matrix("W"), _matrix("V")
    y = frame.y.to_numpy()
    eta = 0.15 + 0.5 * frame.x.to_numpy()
    sigma, rho, nu = 0.8, 0.3, (6.0 if student_law else None)
    coefficients = {f"ar_{k + 1}": _t(v) for k, v in enumerate(ar)}
    coefficients |= {f"ma_{k + 1}": _t(v) for k, v in enumerate(ma)}
    if kind != "fcor":
        coefficients["lagsar" if kind == "lag" else "errorsar"] = _t(rho)

    def law_on(rows):
        subset = frame.iloc[rows].reset_index(drop=True)
        data2 = {
            "W": weights[np.ix_(rows, rows)],
            "V": fixed[np.ix_(rows, rows)],
        }
        return ResidualResponse(
            _layout(text, subset, student=student_law, data2=data2, history=y[rows]),
            _t(eta[rows]),
            _t([sigma] * len(rows)),
            coefficients,
            None if nu is None else _t(nu),
        )

    identity = np.arange(len(frame))
    law = law_on(identity)
    # The frame is sorted by group and time, so brms's rows are these rows.
    mu = np.array(_brms_arma_mu(list(eta), y, ar, ma, frame.g.to_numpy()))
    operator = np.eye(len(y))
    if kind == "fcor":
        covariance = sigma**2 * fixed
        expected = (
            multivariate_normal(mu, covariance).logpdf(y)
            if nu is None
            else multivariate_t(mu, covariance, df=nu).logpdf(y)
        )
    else:
        expected = _sar_oracle(y, mu, sigma, rho, weights, kind, nu)
        operator = np.linalg.inv(np.eye(len(y)) - rho * weights)
        covariance = sigma**2 * operator @ operator.T
    np.testing.assert_allclose(
        law.log_prob(_t(y)).sum().item(), expected, rtol=RTOL, atol=ATOL
    )
    center = operator @ mu if kind == "lag" else mu
    np.testing.assert_allclose(
        law.loo_log_prob(_t(y)).numpy(),
        _loo(y, center, covariance, nu),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(law.linear_predictor.numpy(), mu, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(law.mean.numpy(), center, rtol=1e-9, atol=1e-9)
    # Dropping the ARMA location is a different model.
    plain = (
        multivariate_normal(eta, covariance).logpdf(y)
        if kind == "fcor"
        else _sar_oracle(y, eta, sigma, rho, weights, kind, None)
    )
    if nu is None:
        assert abs(plain - expected) > 1e-3
    # Matrix rows are the rows as given: permuting rows and matrices together
    # leaves the joint density unchanged and permutes the pointwise scores.
    permutation = np.random.default_rng(3).permutation(len(frame))
    permuted = law_on(permutation)
    torch.testing.assert_close(
        permuted.log_prob(_t(y[permutation])).sum(),
        law.log_prob(_t(y)).sum(),
        rtol=1e-9,
        atol=1e-9,
    )
    torch.testing.assert_close(
        permuted.loo_log_prob(_t(y[permutation])),
        law.loo_log_prob(_t(y))[torch.as_tensor(permutation)],
        rtol=1e-8,
        atol=1e-8,
    )
    missing = ResidualResponse(
        _layout(
            text,
            frame,
            student=student_law,
            data2={"W": weights, "V": fixed},
            history=np.where(frame.t == 2, np.nan, y),
        ),
        _t(eta),
        _t([sigma] * len(y)),
        coefficients,
        None if nu is None else _t(nu),
    )
    with pytest.raises(ValueError, match="require the response of every row"):
        _ = missing.mean


def test_arma_beside_sar_has_exact_first_and_second_derivatives():
    frame = _frame()
    weights = _t(_matrix("W"))
    y, x = _t(frame.y.to_numpy()), _t(frame.x.to_numpy())
    layout = _layout(
        "arma(t, g) + sar(W)",
        frame,
        data2={"W": weights.numpy()},
        history=frame.y.to_numpy(),
    )
    eigenvalues = torch.linalg.eigvals(weights).real
    groups = frame.g.to_numpy()

    def model(parameters):
        intercept, slope, log_sigma, phi, theta, rho = parameters
        law = ResidualResponse(
            layout,
            intercept + slope * x,
            log_sigma.exp().expand(len(frame)),
            {"ar_1": phi, "ma_1": theta, "lagsar": rho},
        )
        return law.log_prob(y).sum()

    def reference(parameters):
        intercept, slope, log_sigma, phi, theta, rho = parameters
        mu = torch.stack(
            _brms_arma_mu(list(intercept + slope * x), y, [phi], [theta], groups)
        )
        half = y - rho * (weights @ y) - mu
        count = len(frame)
        return (
            -0.5 * count * math.log(2 * math.pi)
            - count * log_sigma
            + torch.log1p(-rho * eigenvalues).sum()
            - 0.5 * (half @ half) / log_sigma.exp() ** 2
        )

    point = _t([0.2, 0.6, -0.2, 0.4, -0.3, 0.25])
    torch.testing.assert_close(model(point), reference(point), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(
        torch.autograd.functional.jacobian(model, point),
        torch.autograd.functional.jacobian(reference, point),
        rtol=1e-9,
        atol=1e-9,
    )
    torch.testing.assert_close(
        torch.autograd.functional.hessian(model, point),
        torch.autograd.functional.hessian(reference, point),
        rtol=1e-8,
        atol=1e-8,
    )


# brms case name to the QVR formula and fit options.
CASES = {
    "se_ar_cov_default_sigma": ("y | se(s) ~ x + ar(t, g, cov = TRUE)", {}),
    "se_ar_cov": ("y | se(s, sigma = TRUE) ~ x + ar(t, g, cov = TRUE)", {}),
    "se_arma_cov_student": (
        "y | se(s, sigma = TRUE) ~ x + arma(t, g, cov = TRUE)",
        {"family": "student_t"},
    ),
    "se_cosy": ("y | se(s, sigma = TRUE) ~ x + cosy(t, g)", {}),
    "se_unstr": ("y | se(s, sigma = TRUE) ~ x + unstr(t, g)", {}),
    "se_cosy_het": (
        "y | se(s, sigma = TRUE) ~ x + cosy(t, g)",
        {"distributional": {"sigma": "~ x"}},
    ),
    "se_ar": ("y | se(s, sigma = TRUE) ~ x + ar(t, g)", {}),
    "se_fcor": ("y | se(s, sigma = TRUE) ~ x + fcor(V)", {}),
    "se_sar": ("y | se(s, sigma = TRUE) ~ x + sar(W)", {}),
    "se_sar_error_student": (
        "y | se(s, sigma = TRUE) ~ x + sar(W, type = 'error')",
        {"family": "student_t"},
    ),
    "cens_ar": ("y | cens(c, y2) ~ x + ar(t, g)", {}),
    "trunc_arma": ("y | trunc(lb = -3, ub = 4) ~ x + arma(t, g, 2, 1)", {}),
    "cens_trunc_ma_student": (
        "y | cens(c, y2) + trunc(lb = -3, ub = 4) ~ x + ma(t, g)",
        {"family": "student_t"},
    ),
    "weights_ar": ("y | weights(w) ~ x + ar(t, g)", {}),
    "cens_ar_cov": ("y | cens(c, y2) ~ x + ar(t, g, cov = TRUE)", {}),
    "trunc_cosy": ("y | trunc(lb = -3) ~ x + cosy(t, g)", {}),
    "weights_unstr": ("y | weights(w) ~ x + unstr(t, g)", {}),
    "cens_fcor": ("y | cens(c, y2) ~ x + fcor(V)", {}),
    "cens_sar": ("y | cens(c, y2) ~ x + sar(W)", {}),
    "weights_sar": ("y | weights(w) ~ x + sar(W)", {}),
    "trunc_sar": ("y | trunc(lb = -3) ~ x + sar(W)", {}),
    "ar_sar_lag": ("y ~ x + ar(t, g) + sar(W)", {}),
    "arma_sar_error": ("y ~ x + arma(t, g) + sar(W, type = 'error')", {}),
    "ar_fcor": ("y ~ x + ar(t, g, p = 2) + fcor(V)", {}),
    "ma_fcor_student": ("y ~ x + ma(t, g) + fcor(V)", {"family": "student_t"}),
    "ar_fcor_het": (
        "y ~ x + ar(t, g) + fcor(V)",
        {"distributional": {"sigma": "~ x"}},
    ),
    "ar_cov_sar": ("y ~ x + ar(t, g, cov = TRUE) + sar(W)", {}),
    "cosy_fcor": ("y ~ x + cosy(t, g) + fcor(V)", {}),
    "sar_fcor": ("y ~ x + sar(W) + fcor(V)", {}),
    "ar_sar_poisson": ("k ~ x + ar(t, g) + sar(W)", {"family": "poisson"}),
    "ar_fcor_poisson": ("k ~ x + ar(t, g) + fcor(V)", {"family": "poisson"}),
    "ar_sar_sigma": (
        "y ~ x + ar(t, g) + sar(W)",
        {"distributional": {"sigma": "~ x"}},
    ),
    "cens_ar_sar": ("y | cens(c, y2) ~ x + ar(t, g) + sar(W)", {}),
    "cens_ar_fcor": ("y | cens(c, y2) ~ x + ar(t, g) + fcor(V)", {}),
}


def _data2():
    return {"W": _matrix("W"), "V": _matrix("V")}


def test_every_pinned_brms_case_has_a_decision():
    assert set(CASES) == set(ORACLE["cases"])


@pytest.mark.parametrize("name", sorted(CASES))
def test_admission_and_refusal_follow_brms(name):
    formula, options = CASES[name]
    decision = ORACLE["cases"][name]
    frame = _frame().assign(k=lambda rows: np.round(np.exp(rows.y)).astype(int))

    def build():
        return formula_to_qvr(formula, data=frame, data2=_data2(), **options)

    if decision["status"] == "ok":
        assert "Residual" in build()
        return
    with pytest.raises(ValueError) as refusal:
        build()
    message = str(refusal.value)
    if decision["status"] == "refused":
        # brms's own message, whose trailing period QVR may extend.
        assert decision["message"].rstrip(".") in message
    else:
        # brms writes a Stan program that does not compile.
        assert "sar" in decision["message"]
        assert "has no brms model" in message or "Invalid addition" in message


def test_brms_pairs_the_sar_matrix_with_its_sorted_rows():
    # The difference QVR documents: brms sorts rows by group and time and
    # leaves the matrix as given, so its matrix rows are the sorted rows.
    order = ORACLE["row_order"]
    assert order["matrix_unpermuted"] is True
    assert order["given_y"] != order["standata_y"]
    assert sorted(order["given_y"]) == sorted(order["standata_y"])


def _latent(fitted) -> tuple[str, ...]:
    return LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names


BRMS_SITES = {
    "b_Intercept": "intercept",
    "b_x": "beta_x",
    "sigma": "sigma",
    "nu": "nu",
    "ar[1]": "ar_1",
    "ma[1]": "ma_1",
    "cosy": "cosy",
    "lagsar": "lagsar",
    "errorsar": "errorsar",
}


def _pinned(formula, values, **options):
    """Fit one step and replace the posterior by brms's draws."""
    fitted = fit(
        formula,
        data=_frame(),
        data2=_data2(),
        method="svi",
        num_steps=1,
        seed=1,
        **options,
    )
    draws = {
        BRMS_SITES[name]: np.asarray(column, dtype=np.float64)
        for name, column in values["draws"].items()
        if name in BRMS_SITES
    }
    assert set(draws) == set(_latent(fitted))
    samples = {
        name: torch.as_tensor(value, dtype=torch.float64).unsqueeze(0)
        for name, value in draws.items()
    }
    return fitted.with_(
        posterior=MCMCResult(
            samples=samples,
            log_densities=torch.zeros(1, 3),
            acceptance_rates=torch.ones(1),
            divergence_counts=torch.zeros(1, dtype=torch.long),
            num_warmup=0,
            num_samples=3,
        )
    )


VALUE_CASES = {
    "se_cosy": ("y | se(s, sigma = TRUE) ~ x + cosy(t, g)", {}),
    "se_ar_cov_student": (
        "y | se(s, sigma = TRUE) ~ x + ar(t, g, cov = TRUE)",
        {"family": "student_t"},
    ),
    "cens_ar": ("y | cens(c, y2) ~ x + ar(t, g)", {}),
    "trunc_arma": ("y | trunc(lb = -3, ub = 4) ~ x + arma(t, g)", {}),
    "ar_sar_lag": ("y ~ x + ar(t, g) + sar(W)", {}),
    "ma_sar_error_student": (
        "y ~ x + ma(t, g) + sar(W, type = 'error')",
        {"family": "student_t"},
    ),
    "ar_fcor": ("y ~ x + ar(t, g) + fcor(V)", {}),
}


@pytest.mark.parametrize("name", sorted(VALUE_CASES))
def test_likelihood_and_predictors_equal_brms_at_brms_draws(name):
    formula, options = VALUE_CASES[name]
    values = ORACLE["values"][name]
    fitted = _pinned(formula, values, **options)

    def close(actual, expected):
        np.testing.assert_allclose(
            actual[0].double().numpy(),
            np.asarray(expected, dtype=np.float64),
            rtol=FIT_RTOL,
            atol=FIT_ATOL,
        )

    close(fitted.log_likelihood()["y"], values["log_lik"])
    close(fitted.posterior_linpred(), values["linpred"])
    if isinstance(values["epred"], dict):
        # brms's posterior_epred fails for the error SAR Student law, whose
        # expectation is the linear predictor.
        assert values["epred"]["error"] == "'a' is 0-diml"
        close(fitted.posterior_epred(), values["linpred"])
    elif "trunc" not in name:
        close(fitted.posterior_epred(), values["epred"])
    if "new_log_lik" in values:
        fresh = pd.DataFrame(ORACLE["fresh"])
        fresh2 = {"W": _matrix("fresh_W"), "V": _matrix("fresh_V")}
        close(
            fitted.log_likelihood(new_data=fresh, data2=fresh2)["y"],
            values["new_log_lik"],
        )
        close(fitted.posterior_epred(new_data=fresh, data2=fresh2), values["new_epred"])


FIT_CASES = [
    ("y | se(s, sigma = TRUE) ~ x + cosy(t, g)", {}, {"cosy", "sigma"}),
    ("y | cens(c, y2) + trunc(lb = -3, ub = 4) ~ x + ar(t, g)", {}, {"ar_1"}),
    ("y ~ x + ar(t, g) + sar(W)", {}, {"ar_1", "lagsar"}),
    ("y ~ x + ma(t, g) + fcor(V)", {"family": "student_t"}, {"ma_1", "nu"}),
]


@pytest.mark.parametrize("method", ["svi", "hmc", "nuts"])
@pytest.mark.parametrize(("formula", "options", "sites"), FIT_CASES)
def test_combinations_fit_predict_refit_and_restore(
    formula, options, sites, method, tmp_path
):
    frame = _frame()
    settings = (
        {"method": "svi", "num_steps": 15}
        if method == "svi"
        else {"method": method, "num_warmup": 4, "num_samples": 4, "num_chains": 1}
    )
    fitted = fit(formula, data=frame, data2=_data2(), seed=2, **options, **settings)
    draws = fitted.draws(num_samples=3, seed=1)
    assert sites <= set(draws)
    assert all(bool(torch.isfinite(value).all()) for value in draws.values())
    truncated = "trunc" in formula
    scores = fitted.log_likelihood(num_samples=3, seed=1)["y"]
    assert scores.shape == (1, 3, len(frame)) and bool(torch.isfinite(scores).all())
    predicted = fitted.posterior_predict(num_samples=3, seed=1)
    assert predicted.shape == (1, 3, len(frame))
    if truncated:
        assert bool(((predicted >= -3) & (predicted <= 4)).all())
    fresh = pd.DataFrame(ORACLE["fresh"])
    fresh2 = {"W": _matrix("fresh_W"), "V": _matrix("fresh_V")}
    new_scores = fitted.log_likelihood(
        new_data=fresh, data2=fresh2, num_samples=3, seed=1
    )["y"]
    assert new_scores.shape == (1, 3, len(fresh))
    assert bool(torch.isfinite(new_scores).all())
    new_draws = fitted.posterior_predict(
        new_data=fresh, data2=fresh2, num_samples=3, seed=1
    )
    assert new_draws.shape == (1, 3, len(fresh))
    independent = fitted.posterior_predict(
        new_data=fresh, num_samples=3, seed=1, incl_autocor=False
    )
    assert independent.shape == (1, 3, len(fresh))
    if method != "svi":
        return
    linpred = fitted.posterior_linpred(num_samples=3, seed=1)
    assert linpred.shape == (1, 3, len(frame))
    refitted = fitted.update(num_steps=3)
    assert set(refitted.draws(num_samples=1)) == set(draws)
    path = tmp_path / "combination.pt"
    save_fit(fitted, path)
    restored = load_fit(path)
    torch.testing.assert_close(
        restored.log_likelihood(num_samples=3, seed=1)["y"], scores
    )
    torch.testing.assert_close(
        restored.log_likelihood(new_data=fresh, data2=fresh2, num_samples=3, seed=1)[
            "y"
        ],
        new_scores,
    )


def test_known_errors_without_the_structure_keep_their_variance():
    frame = _frame()
    fitted = fit(
        "y | se(s, sigma = TRUE) ~ x + cosy(t, g)",
        data=frame,
        method="svi",
        num_steps=3,
        seed=1,
    )
    draws = fitted.draws(num_samples=2, seed=1)
    scores = fitted.log_likelihood(num_samples=2, seed=1, incl_autocor=False)["y"]
    for draw in range(2):
        value = {k: v[0, draw].double().numpy() for k, v in draws.items()}
        eta = value["intercept"] + value["beta_x"] * frame.x.to_numpy()
        scale = np.sqrt(value["sigma"] ** 2 + frame.s.to_numpy() ** 2)
        np.testing.assert_allclose(
            scores[0, draw].double().numpy(),
            norm.logpdf(frame.y.to_numpy(), eta, scale),
            rtol=FIT_RTOL,
            atol=FIT_ATOL,
        )
    with pytest.raises(ValueError, match="standard_error|absent|not found|'s'"):
        fitted.posterior_predict(new_data=frame.drop(columns="s"), num_samples=1)


def test_new_rows_need_their_own_matrices():
    frame = _frame()
    fitted = fit(
        "y ~ x + ar(t, g) + sar(W)",
        data=frame,
        data2=_data2(),
        method="svi",
        num_steps=2,
        seed=1,
    )
    fresh = pd.DataFrame(ORACLE["fresh"])
    with pytest.raises(ValueError, match=r"requires data2\['W'\] over the new rows"):
        fitted.posterior_epred(new_data=fresh, num_samples=1)
    with pytest.raises(ValueError, match="number of observations"):
        fitted.posterior_epred(new_data=fresh, data2=_data2(), num_samples=1)
    with pytest.raises(ValueError, match="require the response of every row"):
        fitted.posterior_epred(
            new_data=fresh.assign(y=np.nan),
            data2={"W": _matrix("fresh_W")},
            num_samples=1,
        )
    lag = np.asarray(
        fitted.posterior_epred(
            new_data=fresh, data2={"W": _matrix("fresh_W")}, num_samples=2, seed=3
        )[0].double()
    )
    draws = fitted.draws(num_samples=2, seed=3)
    for draw in range(2):
        value = {k: v[0, draw].double().numpy() for k, v in draws.items()}
        eta = value["intercept"] + value["beta_x"] * fresh.x.to_numpy()
        mu = np.array(
            _brms_arma_mu(
                list(eta), fresh.y.to_numpy(), [value["ar_1"]], [], fresh.g.to_numpy()
            ),
            dtype=np.float64,
        )
        expected = np.linalg.solve(
            np.eye(len(fresh)) - value["lagsar"] * _matrix("fresh_W"), mu
        )
        np.testing.assert_allclose(lag[draw], expected, rtol=FIT_RTOL, atol=FIT_ATOL)


def test_joint_models_keep_one_residual_law_per_response():
    frame = _frame().assign(z=lambda rows: rows.y * 0.5 + rows.x)
    joint = fit(
        bf("y ~ x + ar(t, g) + sar(W)") + bf("z ~ x + cosy(t, g)") + set_rescor(False),
        data=frame,
        data2=_data2(),
        method="svi",
        num_steps=2,
        seed=1,
    )
    fresh = pd.DataFrame(ORACLE["fresh"]).assign(z=lambda rows: rows.y * 0.5 + rows.x)
    fresh2 = {"W": _matrix("fresh_W")}
    with pytest.raises(ValueError, match=r"requires data2\['W'\] over the new rows"):
        joint.posterior_epred(new_data=fresh, num_samples=1)
    options = {"new_data": fresh, "data2": fresh2, "num_samples": 3, "seed": 4}
    epred = joint.posterior_epred(**options)
    assert epred["y"].shape == epred["z"].shape == (1, 3, len(fresh))
    draws = joint.draws(num_samples=3, seed=4)
    for name, formula in (
        ("y", "y ~ x + ar(t, g) + sar(W)"),
        ("z", "z ~ x + cosy(t, g)"),
    ):
        alone = fit(
            formula, data=frame, data2=_data2(), method="svi", num_steps=1, seed=1
        )
        prefix = f"response_{name}__"
        local = {
            site.removeprefix(prefix): value
            for site, value in draws.items()
            if site.startswith(prefix)
        }
        assert set(local) == set(_latent(alone))
        alone = alone.with_(
            posterior=MCMCResult(
                samples=local,
                log_densities=torch.zeros(1, 3),
                acceptance_rates=torch.ones(1),
                divergence_counts=torch.zeros(1, dtype=torch.long),
                num_warmup=0,
                num_samples=3,
            )
        )
        torch.testing.assert_close(
            joint.log_likelihood(resp=name, **options)[name],
            alone.log_likelihood(**options)[name],
        )
        torch.testing.assert_close(epred[name], alone.posterior_epred(**options))
    assert joint.posterior_predict(**options)["y"].shape == (1, 3, len(fresh))


def test_estimated_residual_correlations_refuse_residual_terms():
    frame = _frame().assign(z=lambda rows: rows.y * 0.5 + rows.x)
    # brms's message for a covariance term beside rescor.
    with pytest.raises(ValueError, match="Explicit covariance terms cannot be modeled"):
        formula_to_qvr(
            bf("y ~ x + cosy(t, g)") + bf("z ~ x") + set_rescor(True), data=frame
        )
    with pytest.raises(ValueError, match="Explicit covariance terms cannot be modeled"):
        formula_to_qvr(
            bf("y ~ x + ar(t, g) + sar(W)") + bf("z ~ x") + set_rescor(True),
            data=frame,
            data2=_data2(),
        )
    # brms admits the conditional formulation there; QVR names the gap.
    with pytest.raises(ValueError, match=r"ar\(t, g\) beside estimated residual"):
        formula_to_qvr(
            bf("y ~ x + ar(t, g)") + bf("z ~ x") + set_rescor(True), data=frame
        )


@pytest.mark.parametrize(
    "formula",
    [
        "y | se(s, sigma = TRUE) ~ x + cosy(t, g)",
        "y | cens(c, y2) + trunc(lb = -3, ub = 4) ~ x + ar(t, g)",
    ],
)
def test_grouped_folds_refit_and_score_held_out_series(formula):
    frame = _frame()
    fitted = fit(formula, data=frame, method="svi", num_steps=3, seed=1)
    assignments = tuple("abc".index(label) for label in frame.g)
    callback = fitted.refit_callback(
        method_factory=lambda settings, seed: {**settings, "seed": seed},
        num_samples=2,
    )
    result = cross_validate(FoldPlan(assignments=assignments), callback)
    assert result.log_predictive_density.numel() == len(frame)
    assert bool(torch.isfinite(result.log_predictive_density).all())
