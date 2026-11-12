"""Currying is the bijection Hom(X × Y, Z) ≅ Hom(X, [Y, Z]) with [Y, Z] = Y × Z.

The curried morphism's entries are checked against the index equations
``curry(f)[x, (y, z)] = f[(x, y), z]`` (right) and
``curry(f)[y, (x, z)] = f[(x, y), z]`` (left), its declared type
against its tensor shape, and uncurrying against the round trip in
both directions. A curried morphism whose codomain drops the absorbed
factor, as the library once declared, fails the type check.
"""

from __future__ import annotations

import pytest
import torch

from quivers.core.algebras import REAL
from quivers.core.morphisms import CurriedMorphism, UncurriedMorphism, observed
from quivers.core.objects import FinSet, ProductSet

X = FinSet(name="X", cardinality=2)
Y = FinSet(name="Y", cardinality=3)
Z = FinSet(name="Z", cardinality=4)


def _f():
    gen = torch.Generator().manual_seed(0)
    return observed(
        ProductSet(components=(X, Y)),
        Z,
        torch.rand(2, 3, 4, generator=gen),
        algebra=REAL,
    )


@pytest.mark.parametrize("direction", ["right", "left"])
def test_declared_type_matches_the_tensor(direction: str) -> None:
    g = CurriedMorphism(_f(), direction=direction)
    assert tuple(g.tensor.shape) == (*g.domain.shape, *g.codomain.shape)
    absorbed = Y if direction == "right" else X
    assert g.codomain == ProductSet(components=(absorbed, Z))
    assert g.codomain != Z


def test_entries_follow_the_currying_equations() -> None:
    f = _f()
    right = CurriedMorphism(f, "right").tensor
    left = CurriedMorphism(f, "left").tensor
    for x in range(2):
        for y in range(3):
            for z in range(4):
                assert right[x, y, z] == f.tensor[x, y, z]
                assert left[y, x, z] == f.tensor[x, y, z]


@pytest.mark.parametrize("direction", ["right", "left"])
def test_uncurry_inverts_curry(direction: str) -> None:
    f = _f()
    back = UncurriedMorphism(CurriedMorphism(f, direction), direction)
    assert back.domain == f.domain and back.codomain == f.codomain
    torch.testing.assert_close(back.tensor, f.tensor)
    g = observed(X, ProductSet(components=(Y, Z)), torch.rand(2, 3, 4), algebra=REAL)
    again = CurriedMorphism(UncurriedMorphism(g, "right"), "right")
    assert again.domain == g.domain and again.codomain == g.codomain
    torch.testing.assert_close(again.tensor, g.tensor)


def test_uncurry_refuses_a_codomain_that_is_not_a_binary_product() -> None:
    with pytest.raises(TypeError, match="binary product"):
        UncurriedMorphism(observed(X, Z, torch.rand(2, 4), algebra=REAL))
