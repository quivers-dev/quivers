"""Regression tests for audit-confirmed transpile correctness fixes on
the BUGS, JAGS, WebPPL, and Church backends.

Each test transpiles a model exercising one distribution family and
asserts the emitted distribution call is now correct: right name, right
argument order, and right per-target parameterisation (precision vs
rate, scale vs rate, ...). Assertions run on emitted text so they stay
fast and deterministic.

The contract each fix preserves is that the emitted program's joint
log-density equals the QVR model's up to an additive constant.
"""

from __future__ import annotations

from quivers.dsl.parser import parse
from quivers.transpile import transpile


def _scalar_model(step: str) -> str:
    """A one-sample program exercising a single family."""
    return f"object O:FinSet 8\nprogram g:O->O\n    {step}\n    return t\nexport g"


def _emit(step: str, target: str) -> str:
    return transpile(parse(_scalar_model(step)), target=target).decode()


def _dense(text: str) -> str:
    """Strip whitespace so substring assertions ignore layout."""
    return "".join(text.split())


# ---------------------------------------------------------------------------
# StudentT -> dt(mu, tau, k): JAGS / BUGS Student-t is
# location-precision-df parameterised, but torch carries (df, loc,
# scale). The renderer must reorder to (loc, 1/scale^2, df).
# ---------------------------------------------------------------------------


def test_jags_studentt_dt_location_precision_df() -> None:
    out = _dense(_emit("sample t <- StudentT(df=3.0,loc=1.0,scale=2.0)", "jags"))
    assert "dt(1,1/(2*2),3)" in out
    # The old bug emitted the torch arg order verbatim.
    assert "dt(3,1,2)" not in out


def test_bugs_studentt_dt_location_precision_df() -> None:
    out = _dense(_emit("sample t <- StudentT(df=3.0,loc=1.0,scale=2.0)", "bugs"))
    assert "dt(1,1/(2*2),3)" in out
    assert "dt(3,1,2)" not in out


# ---------------------------------------------------------------------------
# Laplace -> ddexp(mu, tau): JAGS / BUGS double-exponential is
# rate-parameterised (tau = 1/scale), not precision (1/scale^2).
# ---------------------------------------------------------------------------


def test_jags_laplace_ddexp_rate() -> None:
    out = _dense(_emit("sample t <- Laplace(loc=0.0,scale=2.0)", "jags"))
    assert "ddexp(0,1/2)" in out
    # The old bug emitted the precision 1/(scale*scale).
    assert "ddexp(0,1/(2*2))" not in out


def test_bugs_laplace_ddexp_rate() -> None:
    out = _dense(_emit("sample t <- Laplace(loc=0.0,scale=2.0)", "bugs"))
    assert "ddexp(0,1/2)" in out
    assert "ddexp(0,1/(2*2))" not in out


# ---------------------------------------------------------------------------
# JAGS / BUGS families that were already correct must stay correct: the
# per-family transform override for Laplace must not perturb Normal,
# Cauchy, or HalfCauchy (all precision-parameterised).
# ---------------------------------------------------------------------------


def test_jags_normal_stays_precision() -> None:
    out = _dense(_emit("sample t <- Normal(loc=1.0,scale=2.0)", "jags"))
    assert "dnorm(1,1/(2*2))" in out


def test_jags_cauchy_stays_precision_df_one() -> None:
    out = _dense(_emit("sample t <- Cauchy(loc=1.0,scale=2.0)", "jags"))
    assert "dt(1,1/(2*2),1)" in out


# ---------------------------------------------------------------------------
# HalfNormal / HalfCauchy lower to the symmetric `dnorm` / `dt` with an
# injected zero location. Without a truncation suffix the emitted
# support wrongly spans all of R, so a latent draw must carry a
# one-sided lower truncation at zero. Both the `jags` and the `bugs`
# backend execute through the JAGS engine (the `bugs` probe image
# installs the `jags` binary and pyjags, and the gallery harness maps
# `"bugs"` to the JAGS probe script), so both emit JAGS's renormalized
# `T(0,)`. The alternative `I(0,)` spelling is JAGS interval censoring,
# which JAGS accepts only when every distribution parameter is fixed and
# rejects at compile time on a latent-parent node (the hierarchical
# `sigma ~ dnorm(0, 1/(tau*tau)) T(0,)` with latent `tau`); `T(0,)` is
# the only spelling that both compiles and preserves the folded prior
# measure.
# ---------------------------------------------------------------------------


_HALF_PRIORS = """object Coef : FinSet 4

program half : Coef -> Coef
    sample tau <- HalfCauchy(scale=1.0)
    sample sigma <- HalfNormal(scale=2.0)
    sample lam : Coef <- HalfCauchy(scale=1.0)
    return tau

export half"""


def test_jags_halfnormal_latent_truncated_at_zero() -> None:
    out = _dense(transpile(parse(_HALF_PRIORS), target="jags").decode())
    assert "dnorm(0,1/(2*2))T(0,)" in out


def test_jags_halfcauchy_latent_truncated_at_zero() -> None:
    out = _dense(transpile(parse(_HALF_PRIORS), target="jags").decode())
    assert "dt(0,1/(1*1),1)T(0,)" in out


def test_bugs_halfnormal_latent_truncated_at_zero() -> None:
    out = _dense(transpile(parse(_HALF_PRIORS), target="bugs").decode())
    assert "dnorm(0,1/(2*2))T(0,)" in out
    # The `bugs` backend runs on the JAGS engine, which rejects the
    # `I(,)` censoring spelling on a latent-parent node.
    assert "dnorm(0,1/(2*2))I(0,)" not in out


def test_bugs_halfcauchy_latent_truncated_at_zero() -> None:
    out = _dense(transpile(parse(_HALF_PRIORS), target="bugs").decode())
    assert "dt(0,1/(1*1),1)T(0,)" in out
    assert "dt(0,1/(1*1),1)I(0,)" not in out


def test_bugs_latent_parent_half_normal_emits_renormalized_truncation() -> None:
    # The canonical hierarchical scale prior: `sigma ~ HalfNormal(tau)`
    # with `tau` itself latent. The `bugs` backend executes through the
    # JAGS engine, which rejects `I(,)` when any parameter of the
    # truncated distribution is latent ("BUGS I(,) notation is only
    # allowed if all parameters are fixed"). The renderer must emit
    # JAGS's renormalized `T(0,)` so the model compiles and the folded
    # prior measure is preserved.
    src = (
        "object Coef : FinSet 4\n"
        "\n"
        "program hier : Coef -> Coef\n"
        "    sample tau <- HalfCauchy(scale=1.0)\n"
        "    sample sigma <- HalfNormal(scale=tau)\n"
        "    return sigma\n"
        "\n"
        "export hier"
    )
    out = _dense(transpile(parse(src), target="bugs").decode())
    # The latent-parent relation carries the renormalized suffix.
    assert "sigma~dnorm(0,1/(tau*tau))T(0,)" in out
    # The censoring spelling that JAGS rejects on a latent parent must
    # never be emitted for the `bugs` backend.
    assert "I(0,)" not in out


def test_jags_bugs_half_truncation_applies_under_a_plate() -> None:
    # `lam : Coef` sits inside a `for` loop; the suffix rides on the
    # indexed LHS relation, not just the scalar one. Both backends run
    # on the JAGS engine and emit the renormalized `T(0,)`.
    for target in ("jags", "bugs"):
        out = _dense(transpile(parse(_HALF_PRIORS), target=target).decode())
        assert "lam[m_Coef]~dt(0,1/(1*1),1)T(0,)" in out


def test_bugs_observed_half_support_is_not_censored() -> None:
    # The JAGS engine reads a truncation suffix on an observed node as
    # censoring, a different likelihood. An observed half-support
    # variate already lies in the support, so no suffix is emitted for
    # it.
    src = (
        "object Obs : FinSet 8\n"
        "\n"
        "program obs : Obs -> Obs\n"
        "    sample sigma <- HalfNormal(scale=1.0)\n"
        "    observe y : Obs <- HalfNormal(scale=sigma)\n"
        "    return sigma\n"
        "\n"
        "export obs"
    )
    for target in ("jags", "bugs"):
        out = _dense(transpile(parse(src), target=target).decode())
        assert "sigma~dnorm(0,1/(1*1))T(0,)" in out
        assert "y[n]~dnorm(0,1/(sigma*sigma))T(0,)" not in out


def test_jags_bugs_truncated_normal_keeps_both_bounds() -> None:
    # The one-sided path must not perturb the two-sided suffix that
    # `TruncatedNormal` emits. Both backends run on the JAGS engine and
    # emit the renormalized `T(low, high)`.
    for target in ("jags", "bugs"):
        out = _dense(
            _emit(
                "sample t <- TruncatedNormal(mu=0.0,sigma=1.0,low=0.0,high=5.0)",
                target,
            )
        )
        assert "dnorm(0,1/(1*1))T(0,5)" in out


# ---------------------------------------------------------------------------
# WebPPL Gamma is scale-parameterised (scale = 1/rate); the torch rate
# must be reciprocated into the scale slot.
# ---------------------------------------------------------------------------


def test_webppl_gamma_rate_reciprocated_to_scale() -> None:
    out = _dense(_emit("sample t <- Gamma(concentration=2.0,rate=5.0)", "webppl"))
    assert "shape:2" in out
    assert "scale:1/5" in out


def test_webppl_exponential_rate_unchanged() -> None:
    # WebPPL's Exponential({a}) is rate-parameterised; no reciprocal.
    out = _dense(_emit("sample t <- Exponential(rate=3.0)", "webppl"))
    assert "Exponential({a:3})" in out


# ---------------------------------------------------------------------------
# WebPPL lacks LogNormal / StudentT / Weibull / NegativeBinomial; the
# renderer grafts a runtime helper defining each so the emitted call
# resolves instead of referencing an undefined name.
# ---------------------------------------------------------------------------


def test_webppl_lognormal_grafts_runtime_helper() -> None:
    out = _emit("sample t <- LogNormal(loc=0.0,scale=1.0)", "webppl")
    assert "var LogNormal = function" in out
    assert _dense("sample(LogNormal({") in _dense(out)


def test_webppl_studentt_grafts_runtime_helper() -> None:
    out = _emit("sample t <- StudentT(df=3.0,loc=1.0,scale=2.0)", "webppl")
    assert "var StudentT = function" in out
    assert _dense("sample(StudentT({") in _dense(out)


def test_webppl_weibull_grafts_runtime_helper() -> None:
    out = _emit("sample t <- Weibull(scale=2.0,concentration=5.0)", "webppl")
    assert "var Weibull = function" in out
    assert _dense("sample(Weibull({") in _dense(out)


def test_webppl_negbinomial_grafts_runtime_helper() -> None:
    out = _emit("sample t <- NegativeBinomial(total_count=10.0,probs=0.3)", "webppl")
    assert "var NegativeBinomial = function" in out
    assert _dense("sample(NegativeBinomial({") in _dense(out)


# ---------------------------------------------------------------------------
# WebPPL array-valued deterministic lets: a `let` combining an
# array-valued prior under scalar operators must broadcast elementwise
# via `_qvr_bcast` rather than coerce arrays to NaN.
# ---------------------------------------------------------------------------


_CHANGEPOINT = """object Step : FinSet 64

program cp : Step -> Step
    sample tau <- Uniform(low=0.0, high=100.0)
    sample rate_before <- Gamma(concentration=2.0, rate=1.0)
    sample rate_after <- Gamma(concentration=2.0, rate=1.0)
    let s = sigmoid(20.0 * (t - tau))
    let rate = (1.0 - s) * rate_before + s * rate_after
    observe y : Step <- Poisson(rate=rate)
    return tau

export cp"""


def test_webppl_array_valued_let_broadcasts() -> None:
    out = transpile(parse(_CHANGEPOINT), target="webppl").decode()
    dense = _dense(out)
    # The `rate` binding combines the array-valued `s` with scalar
    # rates. It is built one row at a time, so every operator reads one
    # entry of `s` and none reaches the whole array.
    assert "((1-s[__i_2])*rate_before)+(s[__i_2]*rate_after)" in dense
    assert "(1-s)*rate_before" not in dense
