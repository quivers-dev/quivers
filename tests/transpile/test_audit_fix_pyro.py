"""Regression tests for the Pyro-backend transpile audit fixes.

Each test transpiles a model exercising one family and asserts the
emitted Pyro source now carries the corrected distribution call. The
audit-confirmed defects these guard:

- `lkj-missing-dimension-arg`: `LKJCholesky` lowered to
  `pyro.distributions.LKJCorrCholesky(eta)` without the mandatory
  matrix-dimension leading argument, so `eta` bound to `d` and the
  concentration was dropped. The dimension is now prepended from the
  sample's event axis: `LKJCorrCholesky(4, eta)`.
- `logitnormal-nonexistent-pyro`: `LogitNormal` emitted as
  `pyro.distributions.LogitNormal`, which Pyro does not ship
  (AttributeError). It is now a grafted runtime helper class realising
  the sigmoid-transform of a Normal, called by its bare name.
- `halfstudentt-nonexistent-pyro`: `HalfStudentT` emitted as
  `pyro.distributions.HalfStudentT`, which Pyro does not ship. It is
  now a grafted runtime helper class realising the folded StudentT.
- `matrixnormal-inversewishart-nonexistent-pyro` (MatrixNormal half):
  `MatrixNormal` emitted as `pyro.distributions.MatrixNormal`, which
  Pyro does not ship, and its covariance arguments were broadcast to
  the full `loc` shape `(3, 4)` rather than the per-factor `(3, 3)` /
  `(4, 4)`. It is now a grafted runtime helper class with per-factor
  covariance shapes derived from the structured-lowering axis indices.
- `matrixnormal-inversewishart-nonexistent-pyro` (InverseWishart half):
  `InverseWishart` emitted as `pyro.distributions.InverseWishart`,
  which Pyro does not ship (AttributeError at construction). It is now
  a grafted runtime helper class carrying the closed-form
  inverse-Wishart density on ``(df, scale_tril)``, checked here against
  `scipy.stats.invwishart`.
- `relaxed-temperature-slot-mishandled`: the relaxed families take
  `temperature` as their leading constructor slot; the emitted call
  must place the QVR's first argument there.

Assertions run on the emitted source text (fast, deterministic), plus
execution checks that score the emitted model against an independent
reference density.
"""

from __future__ import annotations

import pathlib

import pytest

import pyro
import torch
from scipy.stats import invwishart

from quivers.transpile import runtime_pyro as runtime
from quivers.dsl.parser import parse
from quivers.transpile import transpile


_FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "families"


def _emit(source: str) -> tuple[str, str]:
    """Transpile `source` to Pyro; return the decoded text and a
    whitespace-stripped variant for spacing-robust substring checks."""
    text = transpile(parse(source), target="pyro").decode("utf-8")
    stripped = "".join(text.split())
    return text, stripped


_LKJ_SOURCE = """
object Dim : FinSet 4

program correlation_model : Dim -> Dim
    sample eta <- HalfNormal(scale=2.0)
    sample chol : Dim <- LKJCholesky(concentration=eta)
    return chol

export correlation_model
"""


_LOGITNORMAL_SOURCE = """
object Obs : FinSet 8

program logitnormal_model : Obs -> Obs
    sample theta <- LogitNormal(mu=0.0, sigma=1.0)
    return theta

export logitnormal_model
"""


_HALFSTUDENTT_SOURCE = """
object Resp : FinSet 1

program half_student_t_model : Resp -> Resp
    sample sigma <- HalfStudentT(df=3.0, scale=1.0)
    return sigma

export half_student_t_model
"""


#: A 3x3 inverse-Wishart with an explicit lower-triangular scale
#: factor ``L``; the scale matrix scored against is ``Psi = L L^T``.
_INVERSEWISHART_SOURCE = """
object Dim : FinSet 3

program inverse_wishart_model : Dim -> Dim
    sample sigma : Dim <- InverseWishart(df=7.0, scale_tril=[[1.0, 0.0, 0.0], [0.2, 1.0, 0.0], [0.1, 0.3, 1.0]])
    return sigma

export inverse_wishart_model
"""


#: The scale Cholesky factor written in `_INVERSEWISHART_SOURCE`.
_INVERSEWISHART_SCALE_TRIL = (
    (1.0, 0.0, 0.0),
    (0.2, 1.0, 0.0),
    (0.1, 0.3, 1.0),
)

#: The degrees of freedom written in `_INVERSEWISHART_SOURCE`.
_INVERSEWISHART_DF = 7.0


_RELAXED_SOURCE = """
object Obs : FinSet 4

program relaxed_model : Obs -> Obs
    sample x <- RelaxedBernoulli(temperature=0.5, probs=0.3)
    sample z : Obs <- RelaxedOneHotCategorical(temperature=0.25, probs=[0.1, 0.2, 0.3, 0.4])
    return z

export relaxed_model
"""


def test_lkj_prepends_matrix_dimension() -> None:
    """`LKJCholesky` over a `FinSet 4` event axis emits
    `LKJCorrCholesky(4, eta)`, dimension first, concentration second.
    """
    text, stripped = _emit(_LKJ_SOURCE)
    assert "LKJCorrCholesky" in text
    # Dimension prepended ahead of the concentration reference.
    assert "LKJCorrCholesky(4,eta)" in stripped, text
    # The pre-fix single-arg form must not survive.
    assert "LKJCorrCholesky(eta)" not in stripped, text


def test_logitnormal_grafts_helper_not_pyro_class() -> None:
    """`LogitNormal` emits a grafted helper class plus a bare-name
    call, never the nonexistent `pyro.distributions.LogitNormal`."""
    text, stripped = _emit(_LOGITNORMAL_SOURCE)
    # The runtime helper class is grafted onto the module.
    assert "class LogitNormal(" in text
    # Called by its bare name with (loc, scale) in order.
    assert "LogitNormal(0,1)" in stripped, text
    # The undefined pyro attribute must not be emitted.
    assert "pyro.distributions.LogitNormal" not in text


def test_halfstudentt_grafts_helper_not_pyro_class() -> None:
    """`HalfStudentT` emits a grafted helper class plus a bare-name
    call, never the nonexistent `pyro.distributions.HalfStudentT`."""
    text, stripped = _emit(_HALFSTUDENTT_SOURCE)
    assert "class HalfStudentT(" in text
    # Called by its bare name with (df, scale) in order.
    assert "HalfStudentT(3,1)" in stripped, text
    assert "pyro.distributions.HalfStudentT" not in text


def test_matrixnormal_grafts_helper_and_passes_its_declared_matrices() -> None:
    """`MatrixNormal` over `Row(3) x Col(4)` emits a grafted helper
    class, never the nonexistent `pyro.distributions.MatrixNormal`, and
    hands it the mean and the two covariance factors as the `(3, 4)`,
    `(3, 3)` and `(4, 4)` matrices the plan declares those inputs to be,
    with no fill: `torch.full` fills only from a number."""
    source = (_FIXTURES / "matrixnormal.qvr").read_text()
    text, stripped = _emit(source)
    assert "class MatrixNormal(" in text
    assert "pyro.distributions.MatrixNormal" not in text
    assert "MatrixNormal(m_loc,m_row_covariance,m_col_covariance)" in stripped, text
    assert "torch.full(" not in stripped, text


def test_inversewishart_grafts_helper_not_pyro_class() -> None:
    """`InverseWishart` emits a grafted helper class plus a bare-name
    call, never the nonexistent `pyro.distributions.InverseWishart`."""
    text, stripped = _emit(_INVERSEWISHART_SOURCE)
    assert "class InverseWishart(" in text
    assert "pyro.distributions.InverseWishart" not in text
    # Called by its bare name with (df, scale_tril) in order.
    assert "InverseWishart(7,torch.tensor([[" in stripped, text


def test_inversewishart_helper_support_is_positive_definite() -> None:
    """The grafted helper declares the positive-definite support, so
    Pyro's `biject_to` supplies the unconstrained-space Jacobian during
    inference rather than treating the matrix as unconstrained."""
    text, _ = _emit(_INVERSEWISHART_SOURCE)
    assert "support = pyro.distributions.constraints.positive_definite" in text


def test_inversewishart_emitted_density_matches_reference() -> None:
    """The emitted model's joint log-density equals the inverse-Wishart
    log density from `scipy.stats.invwishart` on the same draw.

    This is the measure-equivalence check: the helper scores the closed
    form density in the constrained coordinates, so no change of
    variables and no Jacobian correction enters `log_prob`. A source's
    ``InverseWishart(df, scale_tril=L)`` draws the inverse of a
    ``Wishart(df, scale_tril=L)`` draw, whose conventional scale is
    ``(L L^T)^-1``.
    """
    text, _ = _emit(_INVERSEWISHART_SOURCE)
    namespace: dict[str, object] = {"pyro": pyro, "torch": torch}
    exec(text, namespace)  # noqa: S102 - emitted source under test
    model = namespace["model"]
    assert callable(model)
    with pyro.poutine.trace() as tracer:
        draw = model()
    assert tuple(draw.shape) == (3, 3)
    scale_tril = torch.tensor(
        [list(row) for row in _INVERSEWISHART_SCALE_TRIL],
        dtype=draw.dtype,
    )
    scale = torch.linalg.inv(scale_tril @ scale_tril.transpose(-1, -2))
    reference = float(
        invwishart.logpdf(
            draw.detach().numpy(),
            df=_INVERSEWISHART_DF,
            scale=scale.numpy(),
        )
    )
    emitted = float(tracer.trace.log_prob_sum())
    assert emitted == pytest.approx(reference, rel=1e-5, abs=1e-5)


def test_inversewishart_sampler_mean_matches_theory() -> None:
    """The helper's sampler draws from the density it scores: the
    empirical mean of many draws approaches ``Psi / (nu - d - 1)``,
    confirming the Wishart-inverse construction is the right law.
    """
    torch.manual_seed(0)
    scale_tril = torch.tensor([list(row) for row in _INVERSEWISHART_SCALE_TRIL])
    scale = scale_tril @ scale_tril.transpose(-1, -2)
    dist = runtime.InverseWishart(torch.tensor(_INVERSEWISHART_DF), scale_tril)
    draws = dist.sample((40000,))
    expected = scale / (_INVERSEWISHART_DF - 3.0 - 1.0)
    assert torch.allclose(draws.mean(0), expected, atol=0.05)


def test_relaxed_families_place_temperature_first() -> None:
    """The relaxed families take `temperature` as their leading
    constructor slot, as a tensor since their log density scales by it
    through `torch.full_like`; the QVR call's first argument lands there
    and the probability argument follows."""
    text, stripped = _emit(_RELAXED_SOURCE)
    assert "pyro.distributions.RelaxedBernoulli(torch.tensor(0.5),0.3)" in stripped, (
        text
    )
    assert (
        "pyro.distributions.RelaxedOneHotCategorical("
        "torch.tensor(0.25),torch.tensor([0.1,0.2,0.3,0.4]))" in stripped
    ), text


@pytest.mark.parametrize(
    "source",
    [
        _LOGITNORMAL_SOURCE,
        _HALFSTUDENTT_SOURCE,
        _LKJ_SOURCE,
        _INVERSEWISHART_SOURCE,
        _RELAXED_SOURCE,
    ],
    ids=[
        "logitnormal",
        "halfstudentt",
        "lkj",
        "inversewishart",
        "relaxed",
    ],
)
def test_emitted_model_executes(source: str) -> None:
    """The emitted Pyro model runs end-to-end under a `{pyro, torch}`
    namespace: the grafted helpers and the LKJ dimension arg are not
    merely well-formed text but a runnable program."""
    text, _ = _emit(source)
    namespace: dict[str, object] = {"pyro": pyro, "torch": torch}
    exec(text, namespace)  # noqa: S102 - emitted source under test
    model = namespace["model"]
    assert callable(model)
    with pyro.poutine.trace():
        model()
