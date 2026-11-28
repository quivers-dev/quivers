"""Structural cache reuse with live weights and exact budgets."""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    ComposedRelation,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
)
from quivers.core.relations import calculus
from tests.test_relation_composition import _INPUT, _OUTPUT, _SHARED


def test_materialized_composition_reuses_only_the_last_exact_budget_and_no_weights():
    shared = nn.Parameter(torch.tensor([0.25, 1.0], dtype=torch.float64))
    left, right = (
        RationalRelation(
            RelationTopology(
                tapes,
                1,
                2,
                (Arc(0, 1, (0, 0), 0),),
                (Boundary(0, 1),),
                (Boundary(1, 1),),
                ("shared", "unit"),
            ),
            shared,
            algebra=MARKOV,
        )
        for tapes in ((_INPUT, _SHARED), (_SHARED, _OUTPUT))
    )
    composition = ComposedRelation(left, right)
    words = ((0,), (0,))
    with patch.object(
        calculus, "_materialize_product", wraps=calculus._materialize_product
    ) as build:
        first = composition.materialize(max_states=30, max_work=50)
        assert build.call_count == 1
        assert first.weight(words).item() == 0.25**2

        with torch.no_grad():
            shared[0] = 0.5
        reused = composition.materialize(max_states=30, max_work=50)
        assert build.call_count == 1
        assert reused.topology is first.topology
        assert reused.weight(words).item() == first.weight(words).item() == 0.5**2
        (derivative,) = torch.autograd.grad(reused.weight(words), shared)
        assert derivative[0].item() == 1.0

        with pytest.raises(RelationLimitError, match="max_states"):
            composition.materialize(max_states=1, max_work=50)
        with pytest.raises(RelationLimitError, match="max_work"):
            composition.materialize(max_states=30, max_work=0)
        assert build.call_count == 3
        still_cached = composition.materialize(max_states=30, max_work=50)
        assert still_cached.topology is first.topology
        assert build.call_count == 3

        changed = composition.materialize(max_states=31, max_work=51)
        assert build.call_count == 4
        assert (
            changed.topology == first.topology
            and changed.topology is not first.topology
        )
        again = composition.materialize(max_states=30, max_work=50)
        assert build.call_count == 5
        assert again.topology == first.topology and again.topology is not first.topology
