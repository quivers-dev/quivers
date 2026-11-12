"""The trace axioms of `CompactClosedTrace`, over the Boolean and real semirings.

The canonical trace is checked against an independent statement,
``Tr^U(f)[a, b] = Σ_u f[a, u, b, u]`` written with ``einsum`` (and its
Boolean analogue), and against the axioms of Joyal, Street and Verity
(1996): naturality in ``A`` and ``B``, dinaturality in ``U``, vanishing
over the unit, superposing, and yanking. An algebra that is not a
semiring is refused.
"""

from __future__ import annotations

import pytest
import torch

from quivers.categorical.monoidal import CartesianProductMonoidal
from quivers.categorical.traced import CompactClosedTrace, trace
from quivers.core.algebras import BOOLEAN, PRODUCT_FUZZY, REAL, Semiring
from quivers.core.morphisms import Morphism, identity, observed
from quivers.core.objects import FinSet, ProductSet, SetObject, Unit

A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=3)
U = FinSet(name="U", cardinality=2)


def _random(dom: SetObject, cod: SetObject, algebra: Semiring, seed: int) -> Morphism:
    gen = torch.Generator().manual_seed(seed)
    data = torch.rand(*dom.shape, *cod.shape, generator=gen)
    if algebra is BOOLEAN:
        data = (data > 0.5).float()
    return observed(dom, cod, data, algebra=algebra)


def _close(x: torch.Tensor, y: torch.Tensor) -> None:
    torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


@pytest.fixture(params=["boolean", "real"])
def algebra(request: pytest.FixtureRequest) -> Semiring:
    return BOOLEAN if request.param == "boolean" else REAL


def test_trace_is_the_sum_over_the_feedback_diagonal(algebra: Semiring) -> None:
    f = _random(
        ProductSet(components=(A, U)), ProductSet(components=(B, U)), algebra, 0
    )
    reference = torch.einsum("aubu->ab", f.tensor)
    if algebra is BOOLEAN:
        reference = (reference > 0).float()
    _close(trace(f, U, A, B).tensor, reference)


def test_naturality_and_dinaturality(algebra: Semiring) -> None:
    tr = CompactClosedTrace(algebra)
    au, bu = ProductSet(components=(A, U)), ProductSet(components=(B, U))
    f = _random(au, bu, algebra, 1)
    g = _random(A, A, algebra, 2)
    h = _random(B, B, algebra, 3)
    id_u = identity(U, algebra=algebra)
    left = tr.trace((g @ id_u) >> f >> (h @ id_u), U, A, B).tensor
    _close(left, (g >> tr.trace(f, U, A, B) >> h).tensor)
    # dinaturality in U: Tr(f ; (id_B × k)) = Tr((id_A × k) ; f) for f : A×U → B×V
    v = FinSet(name="V", cardinality=3)
    f2 = _random(au, ProductSet(components=(B, v)), algebra, 4)
    k = _random(v, U, algebra, 5)
    lhs = tr.trace(f2 >> (identity(B, algebra=algebra) @ k), U, A, B).tensor
    rhs = tr.trace((identity(A, algebra=algebra) @ k) >> f2, v, A, B).tensor
    _close(lhs, rhs)


def test_vanishing_superposing_and_yanking(algebra: Semiring) -> None:
    tr = CompactClosedTrace(algebra)
    f = _random(
        ProductSet(components=(A, Unit)), ProductSet(components=(B, Unit)), algebra, 6
    )
    _close(tr.trace(f, Unit, A, B).tensor.reshape(2, 3), f.tensor.reshape(2, 3))
    g = _random(A, A, algebra, 7)
    f3 = _random(
        ProductSet(components=(B, U)), ProductSet(components=(B, U)), algebra, 8
    )
    superposed = tr.trace(
        g @ f3, U, ProductSet(components=(A, B)), ProductSet(components=(A, B))
    ).tensor
    _close(superposed, (g @ tr.trace(f3, U, B, B)).tensor)
    swap = CartesianProductMonoidal(algebra).braiding(U, U)
    _close(tr.trace(swap, U, U, U).tensor, identity(U, algebra=algebra).tensor)


def test_refuses_an_algebra_that_is_not_a_semiring() -> None:
    with pytest.raises(TypeError, match="Semiring"):
        CompactClosedTrace(PRODUCT_FUZZY)  # type: ignore[arg-type]


def test_braiding_is_natural_and_satisfies_the_hexagon(algebra: Semiring) -> None:
    m = CartesianProductMonoidal(algebra)
    f = _random(A, B, algebra, 9)
    g = _random(U, A, algebra, 10)
    # naturality: σ_{B,A} ∘ (f × g) = (g × f) ∘ σ_{A,U}
    _close((m.braiding(A, U) >> (g @ f)).tensor, ((f @ g) >> m.braiding(B, A)).tensor)
    # hexagon (associators are identities on flattened products):
    # σ_{A, B×U} = (σ_{A,B} × id_U) ; (id_B × σ_{A,U})
    c = FinSet(name="C", cardinality=2)
    lhs = m.braiding(A, ProductSet(components=(B, c))).tensor.reshape(12, 12)
    step1 = m.braiding(A, B) @ identity(c, algebra=algebra)
    step2 = identity(B, algebra=algebra) @ m.braiding(A, c)
    _close(lhs, (step1 >> step2).tensor.reshape(12, 12))
