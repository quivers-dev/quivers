"""Regression tests for audit-confirmed Stan transpilation defects.

Each test transpiles a model exercising one family and asserts the
emitted Stan distribution call is now correct (right name plus right
argument order / values / parameterisation). Assertions are on the
emitted text, keyed off the repros in
`notes/transpile-audit-findings.md`.

Covered defects:

* `weibull-shape-scale-swapped`: Stan's `weibull(alpha, sigma)` is
  shape-first, so the QVR `(scale, concentration)` args are swapped.
* `negbinomial-stan-neg-binomial-2-unconverted`: Stan's
  `neg_binomial_2(mu, phi)` is mean / dispersion parameterised, so the
  QVR `(total_count, probs)` args are converted to
  `(total_count * probs / (1 - probs), total_count)`.
* `logitnormal-nonexistent-stan`: Stan ships no `logit_normal`, so the
  renderer grafts a `logit_normal_lpdf` / `_rng` runtime helper.

Every draw is scored with an explicit `target += <family>_lpdf /
_lpmf(...)` increment rather than a `~` sampling statement, so the
assertions below read the density-function spelling. The two forms are
not interchangeable: `~` drops every term Stan judges constant with
respect to the parameters, which discards data-dependent normalisers
(the Weibull log-normalising constant, the neg-binomial-2 term in
`total_count`, the LogitNormal change-of-variables Jacobian). Those are
exactly the terms the QVR measure counts, so only the target-increment
form reproduces the declared joint.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from quivers.dsl.parser import parse
from quivers.transpile import transpile


def _stan(source: str) -> str:
    """Transpile a QVR source string to Stan and decode the bytes."""
    return transpile(parse(source), target="stan").decode()


def _nospace(text: str) -> str:
    """Drop spaces so assertions ignore the formatter's comma spacing."""
    return text.replace(" ", "")


_WEIBULL_SRC = """
object Obs : Real 4
morphism weibull_kernel : Obs -> Obs [role=kernel] ~ Weibull(scale=1.5, concentration=2.0)
program weibull_fixture : Obs -> Obs
    sample x <- weibull_kernel
    return x
export weibull_fixture
"""


def test_weibull_shape_scale_order() -> None:
    """`Weibull(scale=1.5, concentration=2.0)` must emit shape-first as
    `weibull_lpdf(x | 2, 1.5)`, not the transposed `(1.5, 2)`."""
    out = _nospace(_stan(_WEIBULL_SRC))
    assert "target+=weibull_lpdf(x[m_Obs]|2,1.5);" in out
    assert "weibull_lpdf(x[m_Obs]|1.5,2)" not in out


_NEGBIN_SRC = """
object Item : FinSet 21
object Out : FinSet 3
object Resp : FinSet 63

program negbin_regression : Resp -> Resp
    sample beta_0 : Out <- Normal(loc=0.0, scale=5.0)
    sample beta_1 : Out <- Normal(loc=0.0, scale=5.0)
    sample dispersion : Out <- Gamma(concentration=2.0, rate=0.5)

    let b0 = beta_0[out_idx]
    let b1 = beta_1[out_idx]
    let disp = dispersion[out_idx]
    let eta = b0 + b1 * x
    let mu = exp(eta)
    let probs = disp / (disp + mu)

    observe y : Resp <- NegativeBinomial(total_count=disp, probs=probs)
    return beta_1
"""


def test_negbinomial_neg_binomial_2_conversion() -> None:
    """`NegativeBinomial(total_count, probs)` maps to Stan's mean /
    dispersion `neg_binomial_2_lpmf(y | mu, phi)` with
    `mu = total_count * probs / (1 - probs)` and `phi = total_count`,
    not the identity `neg_binomial_2_lpmf(y | total_count, probs)`."""
    out = _nospace(_stan(_NEGBIN_SRC))
    assert (
        "target+=neg_binomial_2_lpmf(y[m_Resp]|"
        "disp[m_Resp]*probs[m_Resp]/(1-probs[m_Resp]),"
        "disp[m_Resp]);" in out
    )
    # The uncorrected identity mapping must not survive.
    assert "neg_binomial_2_lpmf(y[m_Resp]|disp[m_Resp],probs[m_Resp])" not in out


_LOGITNORMAL_SRC = """
object Obs : FinSet 8
program logitnormal_fixture : Obs -> Obs
    sample theta <- LogitNormal(mu=0.0, sigma=1.0)
    return theta
export logitnormal_fixture
"""


def test_logitnormal_runtime_helper_grafted() -> None:
    """LogitNormal has no Stan built-in; the renderer grafts a
    `logit_normal_lpdf` helper (with the change-of-variables Jacobian)
    that the `target += logit_normal_lpdf(theta | 0, 1)` increment then
    calls directly."""
    out = _stan(_LOGITNORMAL_SRC)
    assert "real logit_normal_lpdf(real y, real mu, real sigma)" in out
    assert "real logit_normal_rng(real mu, real sigma)" in out
    # The Jacobian term -log(y) - log1m(y) guards the density. It is
    # data-dependent, so a `~` statement would drop it.
    assert "- log(y) - log1m(y)" in out
    assert _nospace("target += logit_normal_lpdf(theta | 0,1);") in _nospace(out)


@pytest.mark.parametrize(
    "family, arguments, bounds",
    [
        ("HalfNormal", "scale=1.0", "<lower=0>"),
        ("HalfCauchy", "scale=1.0", "<lower=0>"),
        ("HalfStudentT", "df=3.0, scale=1.0", "<lower=0>"),
        ("Gamma", "concentration=2.0, rate=1.0", "<lower=0>"),
        ("Beta", "concentration1=2.0, concentration0=3.0", "<lower=0,upper=1>"),
        ("Uniform", "low=-2.0, high=3.0", "<lower=-2,upper=3>"),
    ],
)
def test_promoted_component_vectors_retain_scalar_support(family, arguments, bounds):
    """A vector consumer cannot remove the producer's componentwise support."""
    source = f"""
object Obs : FinSet 3
object Comp : FinSet 2
program components(concentration: Tensor[Real]([Comp])) : Obs -> Obs
    sample weights <- Dirichlet(concentration=concentration) [over=Comp]
    sample loc <- {family}({arguments}) [over=Comp]
    sample scale <- HalfNormal(scale=1.0) [over=Comp]
    observe y : Obs <- MixtureNormal(weights=weights, loc=loc, scale=scale)
    return y
export components
"""
    emitted = _stan(source)
    compact = "".join(emitted.split())
    assert f"vector{bounds}[2]loc;" in compact
    assert "vector<lower=0>[2]scale;" in compact
    binary = shutil.which("stanc")
    if binary is not None:
        done = subprocess.run(
            [binary, "--info", "-"],
            input=emitted.encode(),
            capture_output=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr.decode()
