"""Every formula family against every ``cens`` and ``trunc`` combination.

The matrix is derived from the family registry: each registered family must
carry a recorded decision in `RECORDED_EVENT_DECISIONS`. A family admitting
the terms is fitted with SVI under left, right and interval censoring, lower,
upper and two-sided truncation, and censoring with truncation; the combined
specification also runs a few HMC and NUTS steps. Its pointwise log
likelihood at pinned posterior draws is compared with an independent oracle:
the uncensored model's own density at the same draws, integrated over each
recorded event and truncation interval by graded Gauss-Legendre quadrature
(or summed over integers for counts), plus any atom at zero. The oracle
never calls a CDF, so it checks the tail code rather than reusing it.
Truncated prediction must stay inside the bounds. A family brms refuses is
refused with an error naming the family and the term.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats

from quivers.formulas import families, fit, formula_to_qvr
from quivers.formulas import _response as response_module
from quivers.formulas._recorded_event_families import RECORDED_EVENT_DECISIONS
from quivers._spline_design import cox_spline_basis
from quivers.inference.mcmc.driver import MCMCResult
from quivers.inference.registry import LatentRegistry

# Quadrature with 20-point panels graded geometrically toward each end is
# accurate to about 1e-10 relative for the smooth densities drawn here; the
# tolerance leaves four orders of margin for float64 log-likelihood rounding.
RTOL = 1e-6
ATOL = 1e-6

KIND = {
    "gaussian": "real",
    "student_t": "real",
    "skew_normal": "real",
    "exgaussian": "real",
    "gen_extreme_value": "real",
    "asym_laplace": "real",
    "zero_inflated_asym_laplace": "real",
    "gamma": "pos",
    "weibull": "pos",
    "exponential": "pos",
    "frechet": "pos",
    "inverse_gaussian": "pos",
    "lognormal": "pos",
    "shifted_lognormal": "pos",
    "cox": "pos",
    "hurdle_gamma": "pos0",
    "hurdle_lognormal": "pos0",
    "beta": "unit",
    "zero_inflated_beta": "unit0",
    "poisson": "count",
    "negative_binomial": "count",
    "negbinomial2": "count",
    "geometric": "count",
    "discrete_weibull": "count",
    "com_poisson": "count",
    "hurdle_poisson": "count",
    "hurdle_negbinomial": "count",
    "zero_inflated_poisson": "count",
    "zero_inflated_negbinomial": "count",
    "binomial": "trials",
    "beta_binomial": "trials",
    "zero_inflated_binomial": "trials",
    "zero_inflated_beta_binomial": "trials",
    "von_mises": "angle",
    "mixture": "real",
}
ATOM_AT_ZERO = {
    "hurdle_gamma",
    "hurdle_lognormal",
    "zero_inflated_beta",
    "zero_inflated_asym_laplace",
}
BOUNDS = {
    "real": (-3.0, 3.0),
    "pos": (0.0625, 20.0),
    "pos0": (0.0, 20.0),
    "unit": (0.0625, 0.9375),
    "unit0": (0.0, 0.9375),
    "count": (1.0, 12.0),
    "trials": (1.0, 5.0),
    "angle": (-3.0, 3.0),
}
SUPPORT = {
    "real": (-math.inf, math.inf),
    "pos": (0.0, math.inf),
    "pos0": (0.0, math.inf),
    "unit": (0.0, 1.0),
    "unit0": (0.0, 1.0),
    "angle": (-math.pi, math.pi),
}
TRIALS = 6
COUNT_CAP = 400
STATUS = np.array([0, 1, -1, 2, 0, -1, 1, 2])
SUPPORTED = sorted(
    name for name, allowed in RECORDED_EVENT_DECISIONS.items() if allowed
)
REFUSED = sorted(
    name for name, allowed in RECORDED_EVENT_DECISIONS.items() if not allowed
)
VARIANTS = ("cens", "trunc_lower", "trunc_upper", "trunc_both", "cens_trunc")


def test_every_registered_family_has_a_recorded_event_decision():
    assert set(families) == set(RECORDED_EVENT_DECISIONS)
    assert set(SUPPORTED) == set(KIND)


def _frame(kind: str) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    n = STATUS.size
    if kind == "real":
        y = np.clip(rng.normal(size=n), -2.5, 2.5)
        up = y + 0.4
    elif kind in {"pos", "pos0"}:
        y = np.clip(rng.gamma(2.0, 1.0, size=n) + 0.1, 0.1, 15.0)
        if kind == "pos0":
            y[[0, 5]] = 0.0
        up = y + 1.0
    elif kind in {"unit", "unit0"}:
        y = rng.uniform(0.1, 0.8, size=n)
        if kind == "unit0":
            y[[0, 5]] = 0.0
        up = y + 0.1
    elif kind == "count":
        y = np.clip(rng.poisson(3.0, size=n), 1, 8).astype(float)
        up = y + 2
    elif kind == "trials":
        y = np.clip(rng.binomial(TRIALS, 0.4, size=n), 1, 4).astype(float)
        up = y + 1
    else:
        y = rng.uniform(-2.5, 2.5, size=n)
        up = y + 0.3
    # Scalar real responses are carried in float32; float32-exact values make
    # the scored endpoints identical to the oracle's.
    y, up = (np.asarray(v, np.float32).astype(np.float64) for v in (y, up))
    return pd.DataFrame(
        {"y": y, "x": rng.normal(size=n), "st": STATUS, "up": up, "n": TRIALS}
    )


def _formula(name: str, variant: str) -> tuple[str, float | None, float | None]:
    lower, upper = BOUNDS[KIND[name]]
    terms = []
    if variant in {"cens", "cens_trunc"}:
        terms.append("cens(st, up)")
    if variant == "trunc_lower":
        terms.append(f"trunc(lb={lower})")
        upper = None
    elif variant == "trunc_upper":
        terms.append(f"trunc(ub={upper})")
        lower = None
    elif variant in {"trunc_both", "cens_trunc"}:
        terms.append(f"trunc(lb={lower}, ub={upper})")
    else:
        lower = upper = None
    if KIND[name] == "trials":
        terms.append("trials(n)")
    return f"y | {' + '.join(terms)} ~ x", lower, upper


def _pinned(fitted, draws: int = 2):
    # Derived coefficient sites (ordered mixture intercepts) are not latent.
    latent = LatentRegistry.from_model(
        fitted.program, observed_names=set(fitted.observations)
    ).names
    samples = {
        site: value[:, :draws].to(torch.float64)
        for site, value in fitted.draws().items()
        if site in latent
    }
    posterior = MCMCResult(
        samples=samples,
        log_densities=torch.zeros(1, draws),
        acceptance_rates=torch.ones(1),
        divergence_counts=torch.zeros(1, dtype=torch.long),
        num_warmup=0,
        num_samples=draws,
    )
    return fitted.with_(posterior=posterior)


def _graded_rule(low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes for ``[low, high]``, resolving singular endpoints.

    The unit coordinate ``r`` is split into 20-point panels: 32 uniform
    panels plus panels halving toward both ends. It is mapped through
    ``s = r**8 / (r**8 + (1 - r)**8)``, which turns an algebraic endpoint
    singularity ``t**(a - 1)`` into the smooth ``r**(8 a - 1)``. Infinite ends
    use ``t = low + s / (1 - s)``. Nodes that round onto a finite endpoint,
    or lie beyond 1e30 where response carriers overflow, are dropped; their
    weight is below 2**-150.
    """
    if math.isinf(low) and math.isinf(high):
        left = _graded_rule(-math.inf, 0.0)
        right = _graded_rule(0.0, math.inf)
        return np.concatenate((left[0], right[0])), np.concatenate((left[1], right[1]))
    nodes, weights = np.polynomial.legendre.leggauss(20)
    grade = 0.5 ** np.arange(1, 21)
    edges = np.unique(np.concatenate((grade, 1.0 - grade, np.linspace(0.0, 1.0, 33))))
    a, b = edges[:-1], edges[1:]
    r = (((a + b) / 2)[:, None] + ((b - a) / 2)[:, None] * nodes).ravel()
    wr = (((b - a) / 2)[:, None] * weights).ravel()
    power = 8
    head, tail = r**power, (1 - r) ** power
    s = head / (head + tail)
    complement = tail / (head + tail)
    ds = power * r ** (power - 1) * (1 - r) ** (power - 1) / (head + tail) ** 2
    if math.isinf(high):
        points, jacobian = low + s / complement, ds / complement**2
    elif math.isinf(low):
        points, jacobian = high - s / complement, ds / complement**2
    else:
        width = high - low
        points = np.where(s < 0.5, low + width * s, high - width * complement)
        jacobian = width * ds
    keep = (points > low) & (points < high) & (np.abs(points) < 1e30) & (jacobian > 0)
    return points[keep], (wr * jacobian)[keep]


def _event_sets(kind: str, frame: pd.DataFrame, status: np.ndarray, lower, upper):
    """Per-row (low, high, low_open) event intervals and the truncation interval."""
    low_t = -math.inf if lower is None else lower
    high_t = math.inf if upper is None else upper
    events = []
    for value, end, code in zip(frame.y, frame.up, status, strict=True):
        if code == 0:
            events.append(None)
        elif code == -1:
            events.append((low_t, min(value, high_t), False))
        elif code == 1:
            events.append((max(value, low_t), high_t, value >= low_t))
        else:
            events.append((max(value, low_t), min(end, high_t), value >= low_t))
    return events, (low_t, high_t)


def _oracle_log_mass(plain, name, kind, frame, row, low, high, low_open, kinks):
    """Integrate or sum the uncensored density of one row over an interval.

    ``kinks`` holds per-draw points where the density is not smooth; the
    continuous integral is split there for each draw separately.
    """
    if kind in {"count", "trials"}:
        cap = TRIALS if kind == "trials" else COUNT_CAP
        first = math.floor(low) + 1 if low_open else math.ceil(max(low, 0.0))
        last = cap if math.isinf(high) else min(math.floor(high), cap)
        grid = np.arange(max(first, 0), last + 1, dtype=float)
        if grid.size == 0:
            return torch.full((1, 2), -torch.inf, dtype=torch.float64)
        values = _plain_log_density(plain, frame, row, grid)
        return torch.logsumexp(values, dim=-1)
    support_low, support_high = SUPPORT[kind]
    a, b = max(low, support_low), min(high, support_high)
    if kind in {"unit", "unit0"} and b >= 1.0 and a > 0.0:
        # Float64 nodes cannot approach 1 closer than its spacing 1.1e-16,
        # which hides about 1e-6 of a Beta mass with a shape below one.
        # Nodes approach 0 without that limit, so the upper tail is taken
        # as the complement of the normalized law's mass on [0, a].
        below = _oracle_log_mass(plain, name, kind, frame, row, -1.0, a, False, kinks)
        return torch.log1p(-below.exp())
    pieces = []
    if b > a:
        per_draw = []
        for draw in range(kinks.shape[-1]):
            cuts = [a, *sorted(k for k in kinks[..., draw].ravel() if a < k < b), b]
            parts = []
            for left, right in zip(cuts[:-1], cuts[1:], strict=True):
                points, weights = _graded_rule(left, right)
                values = _plain_log_density(plain, frame, row, points)[..., draw, :]
                parts.append(
                    torch.logsumexp(values + torch.tensor(np.log(weights)), dim=-1)
                )
            per_draw.append(torch.logsumexp(torch.stack(parts), dim=0))
        pieces.append(torch.stack(per_draw, dim=-1))
    if name in ATOM_AT_ZERO and (low < 0 or (low == 0 and not low_open)) and high >= 0:
        pieces.append(_plain_log_density(plain, frame, row, np.array([0.0]))[..., 0])
    if not pieces:
        return torch.full((1, 2), -torch.inf, dtype=torch.float64)
    return torch.logsumexp(torch.stack(pieces), dim=0)


def _float64_response_dtype(formula, family=None):
    """Carry real oracle nodes in float64; other carriers are unchanged."""
    dtype = _RESPONSE_DTYPE(formula, family)
    return torch.float64 if dtype == torch.float32 else dtype


_RESPONSE_DTYPE = response_module.response_dtype


def _plain_log_density(plain, frame, row, points):
    """Score the uncensored density at quadrature nodes.

    Scalar real responses are carried in float32 by the formula pipeline. The
    oracle evaluates its nodes in float64 instead, so nodes within float32
    resolution of a singular endpoint stay distinct from it.
    """
    data = pd.DataFrame(
        {"y": points, "x": frame.x.iloc[row], "n": TRIALS, "st": 0, "up": 0.0}
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(response_module, "response_dtype", _float64_response_dtype)
        return next(iter(plain.log_likelihood(new_data=data).values()))


KINKS = {
    "asym_laplace": "mu",
    "zero_inflated_asym_laplace": "mu",
    "shifted_lognormal": "ndt",
}
"""Parameter (a latent site or a predicted dpar) locating a density kink."""


def _oracle(name: str, formula: str, frame, fitted, lower, upper):
    kind = KIND[name]
    plain_formula = "y | trials(n) ~ x" if kind == "trials" else "y ~ x"
    family = families[name]
    knots = np.empty(0)
    if name == "cox":
        # The censored fit places its baseline knots from exact event times
        # and interval follow-up; the plain oracle model must share them.
        status = frame.st.to_numpy() if "cens" in formula else np.zeros(len(frame))
        events = frame.y.to_numpy()[status == 0]
        followup = np.concatenate(
            (frame.y.to_numpy(), frame.up.to_numpy()[status == 2])
        )
        basis = cox_spline_basis(events.tolist(), followup.tolist(), df=5, degree=3)
        family = family.with_baseline(basis=basis)
        knots = np.array([*basis.boundary_knots, *basis.interior_knots])
    plain = fit(
        plain_formula, data=frame, family=family, method="svi", num_steps=1, seed=0
    ).with_(posterior=fitted.posterior)
    if name in KINKS:
        parameter = KINKS[name]
        drawn = fitted.draws()
        kinks = (
            drawn[parameter].reshape(drawn[parameter].shape[:2] + (1,))
            if parameter in drawn
            else plain.posterior_linpred(parameter=parameter, transform=True)
        )
        kinks = kinks.detach().numpy()
        kinks = np.broadcast_to(kinks, kinks.shape[:2] + (len(frame),))
    elif name == "cox":
        # The M-spline baseline hazard is piecewise polynomial between knots.
        kinks = np.broadcast_to(
            knots[:, None, None, None], (knots.size, 1, 2, len(frame))
        )
    else:
        kinks = np.empty((1, 2, len(frame)))
        kinks[:] = np.nan
    status = frame.st.to_numpy() if "cens" in formula else np.zeros(len(frame), int)
    events, (low_t, high_t) = _event_sets(kind, frame, status, lower, upper)
    truncated = lower is not None or upper is not None
    rows = []
    for row, event in enumerate(events):
        row_kinks = kinks[..., row].reshape(-1, kinks.shape[-2])
        if event is None:
            numerator = _plain_log_density(
                plain, frame, row, np.array([frame.y.iloc[row]])
            )[..., 0]
        else:
            low, high, low_open = event
            numerator = _oracle_log_mass(
                plain, name, kind, frame, row, low, high, low_open, row_kinks
            )
        if truncated:
            numerator = numerator - _oracle_log_mass(
                plain, name, kind, frame, row, low_t, high_t, False, row_kinks
            )
        rows.append(numerator)
    return torch.stack(rows, dim=-1)


def _mixture_oracle(fitted, frame, formula, lower, upper):
    """Component-wise truncation as in brms: sum_k w_k P_k(event) / P_k(bounds).

    The default ``mixture`` family is ``mixture(gaussian, gaussian)`` with
    ordered intercepts: ``mu_k = intercept_k + beta_k x`` where the second
    intercept is the first plus the sampled positive gap, one scale per
    component, and Dirichlet weights ``theta``.
    """
    draws = {site: value.detach().numpy() for site, value in fitted.draws().items()}
    x, y = frame.x.to_numpy(), frame.y.to_numpy()
    up = frame.up.to_numpy()
    status = frame.st.to_numpy() if "cens" in formula else np.zeros(len(frame), int)
    low_t = -np.inf if lower is None else lower
    high_t = np.inf if upper is None else upper
    chains, count = draws["dpar_mu1_intercept"].shape[:2]
    out = np.empty((chains, count, len(frame)))
    for c in range(chains):
        for d in range(count):
            first = draws["dpar_mu1_intercept"][c, d]
            gap = draws["mixture_intercept_spacing"][c, d].reshape(-1)[0]
            loc = np.stack(
                (
                    first + draws["dpar_mu1_beta_x"][c, d] * x,
                    first + gap + draws["dpar_mu2_beta_x"][c, d] * x,
                ),
                axis=-1,
            )
            scale = np.array([draws["sigma1"][c, d], draws["sigma2"][c, d]])
            weights = draws["theta"][c, d].reshape(-1)

            def cdf(value):
                return stats.norm.cdf(value, loc, scale)

            mass = cdf(high_t) - cdf(low_t)
            code = status[:, None]
            yy, uu = y[:, None], up[:, None]
            event = np.where(
                code == 0,
                stats.norm.pdf(yy, loc, scale),
                np.where(
                    code == -1,
                    cdf(np.minimum(yy, high_t)) - cdf(low_t),
                    np.where(
                        code == 1,
                        cdf(high_t) - cdf(np.maximum(yy, low_t)),
                        cdf(np.minimum(uu, high_t)) - cdf(np.maximum(yy, low_t)),
                    ),
                ),
            )
            out[c, d] = np.log((weights * event / mass).sum(-1))
    return torch.tensor(out)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("name", SUPPORTED)
def test_svi_scores_match_the_density_oracle_and_prediction_respects_bounds(
    name, variant
):
    frame = _frame(KIND[name])
    formula, lower, upper = _formula(name, variant)
    fitted = fit(formula, data=frame, family=name, method="svi", num_steps=3, seed=5)
    for value in fitted.draws().values():
        assert bool(torch.isfinite(value).all())
    pinned = _pinned(fitted)
    actual = next(iter(pinned.log_likelihood().values()))
    assert bool(torch.isfinite(actual).all())
    if name == "mixture":
        expected = _mixture_oracle(pinned, frame, formula, lower, upper)
    else:
        expected = _oracle(name, formula, frame, pinned, lower, upper)
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    draws = pinned.posterior_predict(seed=7)
    assert bool(torch.isfinite(draws).all())
    if lower is not None:
        assert bool((draws >= lower).all())
    if upper is not None:
        assert bool((draws <= upper).all())


SAMPLER_PRIORS = {"com_poisson": {"shape": "Gamma(2.0, 2.0)"}}
"""Priors replacing defaults whose draws leave numerical range before sampling.

The default COM-Poisson ``shape`` prior is brms's ``Gamma(0.01, 0.01)``. The
MCMC driver records one prior draw to discover the latent sites, and such a
draw is often below 1e-10, where the modal count exceeds float64 capacity.
This concerns the uncensored law equally and is reported as a known gap.
"""


@pytest.mark.parametrize("method", ["hmc", "nuts"])
@pytest.mark.parametrize("name", SUPPORTED)
def test_hmc_and_nuts_run_censored_truncated_fits_without_nan(name, method):
    frame = _frame(KIND[name])
    formula, lower, upper = _formula(name, "cens_trunc")
    assert lower is not None and upper is not None
    fitted = fit(
        formula,
        data=frame,
        family=name,
        method=method,
        num_warmup=3,
        num_samples=3,
        num_chains=1,
        seed=5,
        priors=SAMPLER_PRIORS.get(name),
    )
    for value in fitted.draws().values():
        assert bool(torch.isfinite(value).all())
    likelihood = next(iter(fitted.log_likelihood().values()))
    assert bool(torch.isfinite(likelihood).all())
    draws = fitted.posterior_predict(seed=7)
    assert bool(((draws >= lower) & (draws <= upper)).all())


def _refused_frame(name: str) -> tuple[str, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = STATUS.size
    frame = pd.DataFrame({"x": rng.normal(size=n), "st": STATUS})
    extra = ""
    if name == "bernoulli":
        frame["y"] = [0, 1] * (n // 2)
    elif name in {"xbeta", "zero_one_inflated_beta"}:
        frame["y"] = rng.uniform(0.1, 0.8, size=n)
    elif name == "wiener":
        frame["y"] = rng.uniform(0.3, 1.0, size=n)
        frame["dec"] = [0, 1] * (n // 2)
        extra = " + dec(dec)"
    else:
        frame["y"] = pd.Categorical(list("abcabcab"), ordered=name != "categorical")
    frame["up"] = 2.0
    return extra, frame


@pytest.mark.parametrize("term", ["cens(st, up)", "trunc(ub=2)"])
@pytest.mark.parametrize(
    "name", [name for name in REFUSED if families[name].response is None]
)
def test_scalar_families_brms_refuses_are_refused_with_the_family_and_term(name, term):
    extra, frame = _refused_frame(name)
    keyword = term.split("(")[0]
    with pytest.raises(ValueError, match=rf"{keyword}\(\).*'{name}'"):
        formula_to_qvr(f"y | {term}{extra} ~ x", data=frame, family=name)


@pytest.mark.parametrize("term", ["cens(st, up)", "trunc(ub=2)"])
@pytest.mark.parametrize(
    "name", [name for name in REFUSED if families[name].response is not None]
)
def test_vector_families_refuse_recorded_events(name, term):
    rng = np.random.default_rng(0)
    shares = rng.dirichlet([2.0, 2.0, 2.0], STATUS.size)
    frame = pd.DataFrame(
        {"x": rng.normal(size=STATUS.size), "st": STATUS, "up": 2.0}
        | {label: shares[:, k] for k, label in enumerate("abc")}
    )
    if name in {"multinomial", "dirichlet_multinomial"}:
        frame[["a", "b", "c"]] = [2, 1, 3]
    with pytest.raises(ValueError, match="recorded events require one scalar response"):
        formula_to_qvr(f"cbind(a, b, c) | {term} ~ x", data=frame, family=name)
