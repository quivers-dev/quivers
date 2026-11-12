"""Which carrier maps are algebra homomorphisms, against the definition.

An `AlgebraHomomorphism` must preserve the tensor, the binary join, the
unit and the zero on its source carrier; then change of base along it
commutes with composition. Every shipped homomorphism is checked on a
grid, and every shipped `CarrierMap` that is not one is shown to fail
an equation with an explicit counterexample.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from quivers.core.algebra_morphisms import (
    COUNTING_FROM_REAL,
    COUNTING_TO_REAL,
    MARKOV_TO_PRODUCT_FUZZY,
    LOG_PROB_MAP,
    PRODUCT_FUZZY_TO_GODEL,
    MAX_PLUS_MAP,
    PROBABILITY_CLAMP,
    PROBABILITY_TO_REAL,
    AlgebraHomomorphism,
    CarrierMap,
    Embedding,
    IdentityHom,
    threshold,
)
from quivers.core.algebras import (
    BOOLEAN,
    COUNTING,
    GODEL,
    PRODUCT_FUZZY,
    REAL,
    Algebra,
)
from quivers.core.morphisms import observed
from quivers.core.objects import FinSet


def _apply(phi: CarrierMap, x: float) -> float:
    return float(phi.apply(torch.tensor([x], dtype=torch.float64))[0])


def _plus(alg: Algebra, a: float, b: float) -> float:
    return float(alg.join(torch.tensor([a, b], dtype=torch.float64), dim=0))


def _times(alg: Algebra, a: float, b: float) -> float:
    return float(
        alg.tensor_op(
            torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
        )
    )


def _failures(phi: CarrierMap, grid: list[float]) -> list[str]:
    v, w = phi.source, phi.target
    out = []
    if abs(_apply(phi, v.unit) - w.unit) > 1e-9:
        out.append("unit")
    if abs(_apply(phi, v.zero) - w.zero) > 1e-9:
        out.append("zero")
    for a, b in itertools.product(grid, repeat=2):
        if (
            abs(
                _apply(phi, _times(v, a, b)) - _times(w, _apply(phi, a), _apply(phi, b))
            )
            > 1e-6
        ):
            out.append(f"tensor at {(a, b)}")
        if (
            abs(_apply(phi, _plus(v, a, b)) - _plus(w, _apply(phi, a), _apply(phi, b)))
            > 1e-6
        ):
            out.append(f"join at {(a, b)}")
    return out


_HOMS = {
    "identity-real": (IdentityHom(REAL), [-1.0, 0.0, 0.5, 2.0]),
    "identity-godel": (IdentityHom(GODEL), [0.0, 0.25, 1.0]),
    "boolean-into-product-fuzzy": (Embedding(BOOLEAN, PRODUCT_FUZZY), [0.0, 1.0]),
    "counting-to-real": (COUNTING_TO_REAL, [0.0, 1.0, 2.0, 3.0]),
}

_NON_HOMS = {
    "expectation": (MARKOV_TO_PRODUCT_FUZZY, [0.0, 0.5, 0.75]),
    "log-prob": (LOG_PROB_MAP, [0.25, 0.5, 1.0]),
    "max-plus": (MAX_PLUS_MAP, [0.25, 0.5, 1.0]),
    "threshold": (threshold(0.5), [0.6, 0.7]),
    "material-implication": (PRODUCT_FUZZY_TO_GODEL, [0.5, 0.75]),
    "probability-clamp": (PROBABILITY_CLAMP, [-1.0, 0.5]),
    "counting-from-real": (COUNTING_FROM_REAL, [0.5, 1.5]),
    "probability-to-real": (PROBABILITY_TO_REAL, [0.5, 0.75]),
}


@pytest.mark.parametrize("name", sorted(_HOMS))
def test_homomorphisms_preserve_the_operations(name: str) -> None:
    phi, grid = _HOMS[name]
    assert isinstance(phi, AlgebraHomomorphism)
    assert _failures(phi, grid) == []


@pytest.mark.parametrize("name", sorted(_NON_HOMS))
def test_other_carrier_maps_fail_an_equation_and_do_not_claim_to_be_homomorphisms(
    name: str,
) -> None:
    phi, grid = _NON_HOMS[name]
    assert isinstance(phi, CarrierMap)
    assert not isinstance(phi, AlgebraHomomorphism)
    assert _failures(phi, grid) != []


def test_change_base_along_a_homomorphism_commutes_with_composition() -> None:
    a, b, c = (FinSet(name=n, cardinality=k) for n, k in zip("ABC", (2, 3, 2)))
    gen = torch.Generator().manual_seed(0)
    f = observed(
        a, b, torch.randint(0, 3, (2, 3), generator=gen).float(), algebra=COUNTING
    )
    g = observed(
        b, c, torch.randint(0, 3, (3, 2), generator=gen).float(), algebra=COUNTING
    )
    lhs = (f >> g).change_base(COUNTING_TO_REAL).tensor
    rhs = (f.change_base(COUNTING_TO_REAL) >> g.change_base(COUNTING_TO_REAL)).tensor
    torch.testing.assert_close(lhs, rhs)
