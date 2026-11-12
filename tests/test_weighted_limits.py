"""Weighted (co)limits of parallel morphisms, against their universal property.

For parallel matrices ``f_j : A → B`` over a complete lattice ``V``,
viewed in the power ``V^{A×B}``, the weighted colimit and limit are
characterized, for every ``g : A → B``, by

* ``W ⊗ f ≤ g ⟺ W(j) ⊗ f_j ≤ g`` for every ``j``;
* ``g ≤ {W, f} ⟺ W(j) ⊗ g ≤ f_j`` for every ``j``.

Both are checked over every ``g`` on a grid of Gödel values, with
weights that are not all units, and the functions refuse an algebra
that is not a lattice.
"""

from __future__ import annotations

import itertools

import pytest
import torch

from quivers.core.algebras import GODEL, PRODUCT_FUZZY, TROPICAL
from quivers.core.morphisms import observed
from quivers.core.objects import FinSet
from quivers.enriched.weighted_limits import (
    Weight,
    representable_weight,
    terminal_weight,
    weighted_colimit_morphisms,
    weighted_limit_morphisms,
)

A = FinSet(name="A", cardinality=1)
B = FinSet(name="B", cardinality=2)
GRID = [0.0, 0.25, 0.5, 1.0]


def _family():
    f1 = observed(A, B, torch.tensor([[0.5, 1.0]]), algebra=GODEL)
    f2 = observed(A, B, torch.tensor([[0.25, 0.0]]), algebra=GODEL)
    weight = Weight(values=torch.tensor([0.5, 1.0]), algebra=GODEL)
    return [f1, f2], weight


def _leq(x: torch.Tensor, y: torch.Tensor) -> bool:
    return bool((x <= y).all())


def test_weighted_colimit_has_its_universal_property():
    fs, w = _family()
    colim = weighted_colimit_morphisms(w, fs, GODEL)
    for g_vals in itertools.product(GRID, repeat=B.size):
        g = torch.tensor([g_vals])
        lhs = _leq(colim, g)
        rhs = all(
            _leq(torch.minimum(w.values[j], fs[j].tensor), g) for j in range(len(fs))
        )
        assert lhs == rhs


def test_weighted_limit_has_its_universal_property():
    fs, w = _family()
    lim = weighted_limit_morphisms(w, fs, GODEL)
    for g_vals in itertools.product(GRID, repeat=B.size):
        g = torch.tensor([g_vals])
        lhs = _leq(g, lim)
        rhs = all(
            _leq(torch.minimum(w.values[j], g), fs[j].tensor) for j in range(len(fs))
        )
        assert lhs == rhs


def test_tropical_weighted_limit_uses_truncated_subtraction():
    f = observed(A, B, torch.tensor([[3.0, 1.0]]), algebra=TROPICAL)
    w = Weight(values=torch.tensor([2.0]), algebra=TROPICAL)
    lim = weighted_limit_morphisms(w, [f], TROPICAL)
    torch.testing.assert_close(lim, torch.tensor([[1.0, 0.0]]))


def test_weights_and_constructions_refuse_a_non_lattice_algebra():
    with pytest.raises(TypeError, match="IdempotentSemiring"):
        Weight(values=torch.tensor([1.0]), algebra=PRODUCT_FUZZY)  # type: ignore[arg-type]
    fs, w = _family()
    with pytest.raises(TypeError, match="IdempotentSemiring"):
        weighted_limit_morphisms(w, fs, PRODUCT_FUZZY)  # type: ignore[arg-type]


def test_weight_requires_one_dimension():
    with pytest.raises(ValueError, match="1D"):
        Weight(values=torch.tensor([[0.5, 1.0]]), algebra=GODEL)


def test_representable_and_terminal_weights():
    j = FinSet(name="J", cardinality=3)
    rep = representable_weight(j, 1, GODEL)
    assert rep.values.tolist() == [0.0, 1.0, 0.0]
    assert terminal_weight(j, GODEL).values.tolist() == [1.0, 1.0, 1.0]
