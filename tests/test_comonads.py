"""The comonad laws of `CofreeComonad` and its coKleisli category.

The product (environment) comonad ``W(A) = A × S`` is checked against
an independent statement: its counit and comultiplication are written
out here as index maps, and the comonad laws (both counit laws,
coassociativity, naturality of ε and δ, functoriality of W) and the
coKleisli category laws are checked as tensor equations on small sets,
over the Boolean and real semirings. A deliberately broken
comultiplication fails the same checks.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from quivers.core.algebras import BOOLEAN, PRODUCT_FUZZY, REAL, Semiring
from quivers.core.morphisms import Morphism, identity, observed
from quivers.core.objects import FinSet, ProductSet, SetObject
from quivers.monadic.comonads import CofreeComonad, CoKleisliCategory

S = FinSet(name="S", cardinality=2)
A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=3)
C = FinSet(name="C", cardinality=2)


def _close(x: Morphism, y: Morphism) -> bool:
    return x.tensor.shape == y.tensor.shape and bool(
        torch.allclose(x.tensor, y.tensor, atol=1e-6)
    )


def _random(dom: SetObject, cod: SetObject, algebra: Semiring, seed: int) -> Morphism:
    gen = torch.Generator().manual_seed(seed)
    data = torch.rand(*dom.shape, *cod.shape, generator=gen)
    if algebra is BOOLEAN:
        data = (data > 0.5).float()
    return observed(dom, cod, data, algebra=algebra)


def _reference_counit(a: FinSet, algebra: Semiring) -> torch.Tensor:
    data = torch.zeros(a.cardinality, S.cardinality, a.cardinality)
    for x, s in itertools.product(range(a.cardinality), range(S.cardinality)):
        data[x, s, x] = algebra.unit
    return data


def _reference_comultiply(a: FinSet, algebra: Semiring) -> torch.Tensor:
    n, m = a.cardinality, S.cardinality
    data = torch.zeros(n, m, n, m, m)
    for x, s in itertools.product(range(n), range(m)):
        data[x, s, x, s, s] = algebra.unit
    return data


@pytest.fixture(params=["boolean", "real"])
def algebra(request: pytest.FixtureRequest) -> Semiring:
    return BOOLEAN if request.param == "boolean" else REAL


def test_components_match_the_reference(algebra: Semiring) -> None:
    w = CofreeComonad(S, algebra)
    assert torch.equal(w.counit(A).tensor, _reference_counit(A, algebra))
    assert torch.equal(w.comultiply(A).tensor, _reference_comultiply(A, algebra))


def _comonad_law_failures(w: CofreeComonad, a: SetObject) -> list[str]:
    failures = []
    wa = w.endofunctor.map_object(a)
    delta = w.comultiply(a)
    id_wa = identity(wa, algebra=w.algebra)
    if not _close(delta >> w.counit(wa), id_wa):
        failures.append("left counit")
    if not _close(delta >> w.endofunctor.map_morphism(w.counit(a)), id_wa):
        failures.append("right counit")
    if not _close(
        delta >> w.endofunctor.map_morphism(delta), delta >> w.comultiply(wa)
    ):
        failures.append("coassociativity")
    return failures


def test_counit_and_coassociativity_laws(algebra: Semiring) -> None:
    assert _comonad_law_failures(CofreeComonad(S, algebra), A) == []


def test_counit_and_comultiplication_are_natural(algebra: Semiring) -> None:
    w = CofreeComonad(S, algebra)
    f = _random(A, B, algebra, seed=0)
    wf = w.endofunctor.map_morphism(f)
    assert _close(wf >> w.counit(B), w.counit(A) >> f)
    wwf = w.endofunctor.map_morphism(wf)
    assert _close(wf >> w.comultiply(B), w.comultiply(A) >> wwf)


def test_endofunctor_is_a_functor(algebra: Semiring) -> None:
    w = CofreeComonad(S, algebra)
    f = _random(A, B, algebra, seed=1)
    g = _random(B, C, algebra, seed=2)
    fun = w.endofunctor
    assert _close(fun.map_morphism(f >> g), fun.map_morphism(f) >> fun.map_morphism(g))
    assert _close(
        fun.map_morphism(identity(A, algebra=algebra)),
        identity(fun.map_object(A), algebra=algebra),
    )


def test_cokleisli_category_is_a_category(algebra: Semiring) -> None:
    w = CofreeComonad(S, algebra)
    cat = CoKleisliCategory(w)
    f = _random(ProductSet(components=(A, S)), B, algebra, seed=3)
    g = _random(ProductSet(components=(B, S)), C, algebra, seed=4)
    h = _random(ProductSet(components=(C, S)), A, algebra, seed=5)
    assert _close(cat.compose(cat.compose(f, g), h), cat.compose(f, cat.compose(g, h)))
    assert _close(cat.compose(cat.identity(A), f), f)
    assert _close(cat.compose(f, cat.identity(B)), f)


def test_refuses_an_algebra_that_is_not_a_semiring() -> None:
    with pytest.raises(TypeError, match="Semiring"):
        CofreeComonad(S, PRODUCT_FUZZY)  # type: ignore[arg-type]


def test_base_of_refuses_an_object_without_the_environment() -> None:
    w = CofreeComonad(S, BOOLEAN)
    with pytest.raises(ValueError, match="not of the form"):
        w.base_of(ProductSet(components=(A, B)))


class _ForgetfulDelta(CofreeComonad):
    """δ(a, s) = ((a, s₀), s): forgets the environment in the inner copy."""

    def comultiply(self, obj: SetObject) -> Morphism:
        lawful = super().comultiply(obj)
        n, m = obj.size, S.cardinality
        data = torch.zeros(n, m, n, m, m)
        for x, s in itertools.product(range(n), range(m)):
            data[x, s, x, 0, s] = self.algebra.unit
        data = data.reshape(lawful.tensor.shape)
        return observed(lawful.domain, lawful.codomain, data, algebra=self.algebra)


def test_the_law_checks_reject_a_broken_comultiplication() -> None:
    failures = _comonad_law_failures(_ForgetfulDelta(S, BOOLEAN), A)
    assert "left counit" in failures
