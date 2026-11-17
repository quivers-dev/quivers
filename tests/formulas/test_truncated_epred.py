"""``posterior_epred`` of a truncated response is its truncated mean.

The reference never calls the truncated-mean code: at pinned posterior draws
the untruncated model's own density is integrated against ``y`` over the
truncation interval by graded Gauss-Legendre quadrature (or summed over the
integers for counts) and divided by the same integral of the density. The
Gaussian mixture, which brms truncates component by component, is checked
against SciPy's truncated normal means.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import stats

from quivers.formulas import families, fit

import test_recorded_event_matrix as matrix

# The reference rule is accurate to about 1e-10 relative; the tolerance of
# the censoring matrix is kept, leaving room for float64 rounding of the
# replayed parameters.
RTOL = matrix.RTOL
ATOL = matrix.ATOL

FAMILIES = [
    "gaussian",
    "skew_normal",
    "exgaussian",
    "asym_laplace",
    "gen_extreme_value",
    "lognormal",
    "shifted_lognormal",
    "gamma",
    "exponential",
    "weibull",
    "frechet",
    "inverse_gaussian",
    "hurdle_gamma",
    "hurdle_lognormal",
    "beta",
    "zero_inflated_beta",
    "poisson",
    "negative_binomial",
    "geometric",
    "binomial",
    "beta_binomial",
    "zero_inflated_poisson",
    "hurdle_negbinomial",
]
VARIANTS = ("trunc_lower", "trunc_upper", "trunc_both")


def _kinks(name: str, fitted, plain, rows: int) -> np.ndarray:
    """Per-draw, per-row points where the density is not smooth, or NaN."""
    draws = fitted.posterior.num_samples
    if name not in matrix.KINKS:
        return np.full((draws, rows), np.nan)
    parameter = matrix.KINKS[name]
    drawn = fitted.draws()
    values = (
        drawn[parameter].reshape(1, draws, 1)
        if parameter in drawn
        else plain.posterior_linpred(parameter=parameter, transform=True)
    )
    return np.broadcast_to(values.detach().numpy(), (1, draws, rows))[0]


def _plain(name: str, fitted, frame):
    kind = matrix.KIND[name]
    plain_formula = "y | trials(n) ~ x" if kind == "trials" else "y ~ x"
    return fit(
        plain_formula,
        data=frame,
        family=families[name],
        method="svi",
        num_steps=1,
        seed=0,
    ).with_(posterior=fitted.posterior)


def _reference(name: str, fitted, frame, lower, upper) -> torch.Tensor:
    """Truncated means from the untruncated density, shape (1, draws, rows).

    The numerator integrates ``y f(y)`` and the denominator ``f(y)`` over the
    truncation interval, split at the density's kink for each draw. An atom
    at zero inside the interval joins the denominator only.
    """
    kind = matrix.KIND[name]
    plain = _plain(name, fitted, frame)
    low = -math.inf if lower is None else lower
    high = math.inf if upper is None else upper
    kinks = _kinks(name, fitted, plain, len(frame))
    draws = kinks.shape[0]
    result = torch.empty(1, draws, len(frame), dtype=torch.float64)
    for row in range(len(frame)):
        for draw in range(draws):
            if kind in {"count", "trials"}:
                cap = matrix.TRIALS if kind == "trials" else matrix.COUNT_CAP
                first = max(math.ceil(low), 0) if math.isfinite(low) else 0
                last = cap if math.isinf(high) else min(math.floor(high), cap)
                points = np.arange(first, last + 1, dtype=float)
                weights = np.ones_like(points)
            else:
                support_low, support_high = matrix.SUPPORT[kind]
                a, b = max(low, support_low), min(high, support_high)
                kink = kinks[draw, row]
                cuts = [a, kink, b] if a < kink < b else [a, b]
                rules = [
                    matrix._graded_rule(left, right)
                    for left, right in zip(cuts[:-1], cuts[1:], strict=True)
                ]
                points = np.concatenate([rule[0] for rule in rules])
                weights = np.concatenate([rule[1] for rule in rules])
            density = matrix._plain_log_density(plain, frame, row, points)[0, draw]
            weighted = density.exp() * torch.tensor(weights)
            mass = weighted.sum()
            if name in matrix.ATOM_AT_ZERO and low <= 0 <= high:
                atom = matrix._plain_log_density(plain, frame, row, np.array([0.0]))
                mass = mass + atom[0, draw, 0].exp()
            result[0, draw, row] = (weighted * torch.tensor(points)).sum() / mass
    return result


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", FAMILIES)
def test_truncated_epred_matches_the_integrated_untruncated_density(
    name: str, variant: str
) -> None:
    frame = matrix._frame(matrix.KIND[name])
    formula, lower, upper = matrix._formula(name, variant)
    fitted = matrix._pinned(
        fit(formula, data=frame, family=name, method="svi", num_steps=1, seed=0)
    )
    observed = fitted.posterior_epred().to(torch.float64)
    assert observed.shape == (1, 2, len(frame))
    reference = _reference(name, fitted, frame, lower, upper)
    torch.testing.assert_close(observed, reference, rtol=RTOL, atol=ATOL)
    low = -math.inf if lower is None else lower
    high = math.inf if upper is None else upper
    assert bool(((observed >= low) & (observed <= high)).all())
    # The expectation is deterministic: no Monte Carlo budget, and the same
    # values on a second call and for the rows given as new data.
    torch.testing.assert_close(fitted.posterior_epred().to(torch.float64), observed)
    again = fitted.posterior_epred(new_data=frame).to(torch.float64)
    torch.testing.assert_close(again, observed, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_truncated_mixture_epred_averages_truncated_component_means(
    variant: str,
) -> None:
    frame = matrix._frame("real")
    formula, lower, upper = matrix._formula("mixture", variant)
    fitted = matrix._pinned(
        fit(formula, data=frame, family="mixture", method="svi", num_steps=1, seed=0)
    )
    observed = fitted.posterior_epred().to(torch.float64).numpy()
    draws = {site: value.numpy() for site, value in fitted.draws().items()}
    low = -np.inf if lower is None else lower
    high = np.inf if upper is None else upper
    x = frame.x.to_numpy()
    expected = np.empty_like(observed)
    for d in range(observed.shape[1]):
        first = draws["dpar_mu1_intercept"][0, d]
        gap = draws["mixture_intercept_spacing"][0, d].reshape(-1)[0]
        locations = (
            first + draws["dpar_mu1_beta_x"][0, d] * x,
            first + gap + draws["dpar_mu2_beta_x"][0, d] * x,
        )
        scales = (draws["sigma1"][0, d], draws["sigma2"][0, d])
        weights = draws["theta"][0, d].reshape(-1)
        expected[0, d] = sum(
            weight
            * stats.truncnorm.mean(
                (low - loc) / scale, (high - loc) / scale, loc=loc, scale=scale
            )
            for weight, loc, scale in zip(weights, locations, scales, strict=True)
        )
    np.testing.assert_allclose(observed, expected, rtol=RTOL, atol=ATOL)


def test_fitted_and_conditional_effects_need_no_monte_carlo_budget() -> None:
    frame = matrix._frame("real")
    fitted = fit(
        "y | trunc(lb = -3) ~ x", data=frame, method="svi", num_steps=2, seed=0
    )
    table = fitted.fitted()
    assert len(table.to_frame()) == len(frame)
    assert fitted.conditional_effects("x") is not None
    assert np.isfinite(fitted.residuals().to_frame().to_numpy(dtype=float)).all()
    ratio = fitted.bayes_r2()
    assert bool(torch.isfinite(ratio).all() and ((ratio >= 0) & (ratio <= 1)).all())


def test_a_monte_carlo_budget_remains_an_option_where_no_rule_applies() -> None:
    frame = matrix._frame("pos")
    formula, _, _ = matrix._formula("cox", "trunc_both")
    fitted = fit(formula, data=frame, family="cox", method="svi", num_steps=1, seed=0)
    with pytest.raises(TypeError, match="missed the interval mass"):
        fitted.posterior_epred(num_samples=2)
    sampled = fitted.posterior_epred(num_samples=2, mc_samples=64)
    assert bool(torch.isfinite(sampled).all())
    assert bool(((sampled >= 0.0625) & (sampled <= 20.0)).all())
    one_sided, _, _ = matrix._formula("cox", "trunc_lower")
    open_fit = fit(
        one_sided, data=frame, family="cox", method="svi", num_steps=1, seed=0
    )
    with pytest.raises(TypeError, match="certifies no finite mean"):
        open_fit.posterior_epred(num_samples=2, mc_samples=64)
