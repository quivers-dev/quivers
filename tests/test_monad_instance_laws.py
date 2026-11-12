"""The monad laws of every shipped monad instance, on small finite sets.

The laws are stated here, independently of `quivers.monadic.laws`, as
equations between composites:

* unit and associativity: ``μ ∘ η_T = id``, ``μ ∘ T(η) = id``,
  ``μ ∘ T(μ) = μ ∘ μ_T``;
* naturality of ``η`` and ``μ`` for functions ``f``;
* the functor laws for functions;
* the Kleisli category laws: ``η ⋆ k = k``, ``k ⋆ η = k`` and
  ``(k ⋆ h) ⋆ l = k ⋆ (h ⋆ l)`` with ``k ⋆ h = k ; T(h) ; μ``, for every
  Kleisli arrow between one-element sets (the continuation monad's
  ``T(T(T(A)))`` is too large to materialize, so its associativity is
  checked in this Kleisli form).

The function-space instances act on functions, so they refuse a
morphism that is not one; a broken monoid is refused by `Writer`.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from quivers.core.algebras import BOOLEAN, GODEL, MARKOV
from quivers.core.morphisms import Morphism, identity, observed
from quivers.core.objects import FinSet, SetObject
from quivers.monadic import (
    Alternative_,
    Continuation,
    FuzzyPowersetMonad,
    Identity,
    Maybe,
    Reader,
    State,
    Writer,
)
from quivers.monadic.typeclasses import Monad
from quivers.stochastic.giry import GiryMonad

ONE = FinSet(name="One", cardinality=1)
A = FinSet(name="A", cardinality=2)
B = FinSet(name="B", cardinality=3)
C = FinSet(name="C", cardinality=2)


def _close(x: Morphism, y: Morphism) -> bool:
    return x.tensor.shape == y.tensor.shape and bool(
        torch.allclose(x.tensor, y.tensor, atol=1e-5)
    )


def _algebra_of(m: Monad):
    return m.pure(ONE).algebra


def _function(dom: SetObject, cod: SetObject, table: list[int], m: Monad) -> Morphism:
    algebra = _algebra_of(m)
    data = torch.full((dom.size, cod.size), float(algebra.zero))
    for i, j in enumerate(table):
        data[i, j] = float(algebra.unit)
    return observed(dom, cod, data.reshape(*dom.shape, *cod.shape), algebra=algebra)


#: (monad, object at which unit and associativity are materialized)
_MONADS: dict[str, tuple[Monad, SetObject]] = {
    "identity": (Identity(), A),
    "maybe": (Maybe(), A),
    "alternative": (Alternative_(), A),
    "reader": (Reader(env=FinSet(name="E", cardinality=2)), A),
    "state": (State(state=FinSet(name="S", cardinality=2)), ONE),
    "writer": (Writer(monoid=FinSet(name="M", cardinality=2)), A),
    "fuzzy-powerset": (FuzzyPowersetMonad(GODEL), A),
    "giry": (GiryMonad(), A),
}


@pytest.mark.parametrize("name", sorted(_MONADS))
def test_unit_and_associativity_laws(name: str) -> None:
    m, a = _MONADS[name]
    ta = m.fmap_obj(a)
    tta = m.fmap_obj(ta)
    mu = m.join(a)
    id_ta = identity(ta, algebra=mu.algebra)
    assert _close(m.pure(ta) >> mu, id_ta), "left unit"
    assert _close(m.fmap(a, ta, m.pure(a)) >> mu, id_ta), "right unit"
    assert _close(m.fmap(tta, ta, mu) >> mu, m.join(ta) >> mu), "associativity"


@pytest.mark.parametrize("name", sorted(_MONADS))
def test_unit_and_multiplication_are_natural(name: str) -> None:
    m, a = _MONADS[name]
    b = B if a is A else ONE
    for table in itertools.product(range(b.size), repeat=a.size):
        f = _function(a, b, list(table), m)
        tf = m.fmap(a, b, f)
        assert _close(m.pure(a) >> tf, f >> m.pure(b))
        tta, ttb = m.fmap_obj(m.fmap_obj(a)), m.fmap_obj(m.fmap_obj(b))
        ttf = m.fmap(m.fmap_obj(a), m.fmap_obj(b), tf)
        assert tta.size == ttf.domain.size and ttb.size == ttf.codomain.size
        assert _close(m.join(a) >> tf, ttf >> m.join(b))


@pytest.mark.parametrize("name", sorted(_MONADS))
def test_functor_laws_on_functions(name: str) -> None:
    m, a = _MONADS[name]
    b = B if a is A else ONE
    c = C if a is A else ONE
    f = _function(a, b, [i % b.size for i in range(a.size)], m)
    g = _function(b, c, [(i + 1) % c.size for i in range(b.size)], m)
    assert _close(m.fmap(a, c, f >> g), m.fmap(a, b, f) >> m.fmap(b, c, g))
    id_a = identity(a, algebra=_algebra_of(m))
    assert _close(m.fmap(a, a, id_a), identity(m.fmap_obj(a), algebra=_algebra_of(m)))


def _kleisli(m: Monad, k: Morphism, h: Morphism, mid: SetObject, end: SetObject) -> Morphism:
    return k >> m.fmap(mid, m.fmap_obj(end), h) >> m.join(end)


def _kleisli_arrows(m: Monad) -> list[Morphism]:
    t_one = m.fmap_obj(ONE)
    return [_function(ONE, t_one, [i], m) for i in range(t_one.size)]


_KLEISLI = {**_MONADS, "continuation": (Continuation(answer=FinSet(name="R", cardinality=2)), ONE)}


@pytest.mark.parametrize("name", sorted(_KLEISLI))
def test_kleisli_category_laws(name: str) -> None:
    m, _ = _KLEISLI[name]
    arrows = _kleisli_arrows(m)
    if len(arrows) > 4:
        arrows = arrows[:4]
    eta = m.pure(ONE)
    for k in arrows:
        assert _close(_kleisli(m, eta, k, ONE, ONE), k), "left identity"
        assert _close(_kleisli(m, k, eta, ONE, ONE), k), "right identity"
    for k, h, l in itertools.product(arrows, repeat=3):
        left = _kleisli(m, _kleisli(m, k, h, ONE, ONE), l, ONE, ONE)
        right = _kleisli(m, k, _kleisli(m, h, l, ONE, ONE), ONE, ONE)
        assert _close(left, right), "associativity"


@pytest.mark.parametrize(
    "monad",
    [
        State(state=FinSet(name="S", cardinality=2)),
        Reader(env=FinSet(name="E", cardinality=2)),
        Continuation(answer=FinSet(name="R", cardinality=2)),
    ],
    ids=["state", "reader", "continuation"],
)
def test_function_space_instances_refuse_a_relation(monad: Monad) -> None:
    relation = observed(ONE, A, torch.tensor([[1.0, 1.0]]))
    with pytest.raises(ValueError, match="not the indicator of one element"):
        monad.fmap(ONE, A, relation)


def test_writer_refuses_an_operation_that_is_not_a_monoid() -> None:
    m = 2
    # m1 ⊕ m2 = m1: associative, but index 0 is not a left unit.
    table = [0.0] * (m**3)
    for a in range(m):
        for b in range(m):
            table[(a * m + b) * m + a] = 1.0
    writer = Writer(monoid=FinSet(name="M", cardinality=m), monoid_op_tensor=tuple(table))
    with pytest.raises(ValueError, match="unit"):
        writer.join(A)


def test_alternative_and_powerset_compose_over_their_semirings() -> None:
    assert _algebra_of(Alternative_()) is BOOLEAN
    assert _algebra_of(GiryMonad()) is MARKOV


def test_giry_refuses_a_kernel_over_another_algebra() -> None:
    f = observed(A, A, torch.full((2, 2), 0.5))
    with pytest.raises(ValueError, match="cannot compose"):
        GiryMonad().kleisli_compose(f, f)
