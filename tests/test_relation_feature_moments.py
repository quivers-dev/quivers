"""A packed expectation carrier compared with original path enumeration."""

from fractions import Fraction
from math import log

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.objects import FinSet, StringSet
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationTopology,
)

_TAPE = StringSet(alphabet=FinSet(name="Symbols", cardinality=2))
_WORD = (0, 0)
_ARC_FEATURES = (
    (1, 0, 0),
    (0, 4, 0),
    (2, 1, 0),
    (1, 7, 0),
    (3, 0, 0),
    (128, 10, 0),
    (0, 0, 0),
)
_START_FEATURES = ((1, 3, 0),)
_STOP_FEATURES = ((6, 1, 0), (0, 5, 0))


def _fixture():
    return RelationTopology(
        (_TAPE,),
        1,
        5,
        (
            Arc(0, 1, (None,), 1),
            Arc(0, 1, (None,), 1),
            Arc(1, 1, (0,), 1),
            Arc(0, 2, (0,), 2),
            Arc(2, 3, (0,), 3),
            Arc(0, 2, (0,), 4),
            Arc(4, 0, (None,), 0),
        ),
        (Boundary(4, 3),),
        (Boundary(1, 3), Boundary(3, 3)),
        ("entrance", "same", "alternate", "one", "zero"),
    )


def _enumeration(topology, weights):
    rows = []

    def walk(state, offset, slots, arcs, features):
        if offset == len(_WORD):
            for mark_index, mark in enumerate(topology.final):
                if mark.state == state:
                    owners = (*slots, mark.weight)
                    mass = Fraction(1)
                    for owner in owners:
                        mass *= weights[owner]
                    total = tuple(
                        a + b
                        for a, b in zip(
                            features, _STOP_FEATURES[mark_index], strict=True
                        )
                    )
                    rows.append((mass, total, owners, arcs))
        for arc_index, arc in enumerate(topology.arcs):
            if arc.source != state:
                continue
            (symbol,) = arc.labels
            if symbol is not None and (offset == len(_WORD) or symbol != _WORD[offset]):
                continue
            walk(
                arc.target,
                offset + (symbol is not None),
                (*slots, arc.weight),
                (*arcs, arc_index),
                tuple(
                    a + b
                    for a, b in zip(features, _ARC_FEATURES[arc_index], strict=True)
                ),
            )

    for mark_index, mark in enumerate(topology.initial):
        walk(mark.state, 0, (mark.weight,), (), _START_FEATURES[mark_index])
    return rows


def _features():
    return tuple(
        torch.tensor(rows, dtype=torch.float64).log().requires_grad_()
        for rows in (_ARC_FEATURES, _START_FEATURES, _STOP_FEATURES)
    )


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_packed_moments_match_occurrence_oracle_and_shared_live_gradients(algebra):
    topology = _fixture()
    shift = -1000 if algebra is LOG_PROB else 0
    weights = (Fraction(1), Fraction(1, 4), Fraction(1, 8), Fraction(1), Fraction(0))
    live = nn.Parameter(torch.empty(5, dtype=torch.float64))
    arc, initial, final = _features()

    def update(values):
        native = torch.tensor([float(item) for item in values], dtype=torch.float64)
        if algebra is LOG_PROB:
            native = native.log()
            native[0] += shift
        with torch.no_grad():
            live.copy_(native)

    update(weights)
    query = RationalRelation(topology, live, algebra=algebra).query((_WORD,))
    for weights in (weights, (*weights[:1], Fraction(1, 2), *weights[2:])):
        update(weights)
        rows = _enumeration(topology, weights)
        assert [row[-1] for row in rows] == [
            (6, 0, 2, 2),
            (6, 1, 2, 2),
            (6, 3, 4),
            (6, 5, 4),
        ]
        mass, moments = query.log_feature_moments(
            arc, initial_logs=initial, final_logs=final
        )
        totals = (
            sum(row[0] for row in rows),
            *(
                sum(weight * scores[feature] for weight, scores, _, _ in rows)
                for feature in range(3)
            ),
        )
        values = (mass, *moments)
        for k, (value, total) in enumerate(zip(values, totals, strict=True)):
            if not total:
                assert value.item() == -torch.inf
                assert torch.equal(
                    torch.autograd.grad(value, live, retain_graph=True)[0],
                    torch.zeros_like(live),
                )
                continue
            assert value.item() == pytest.approx(
                shift + log(float(total)), abs=2e-12, rel=0.0
            )
            (gradient,) = torch.autograd.grad(value, live, retain_graph=True)
            expected_derivatives = []
            for slot, base in enumerate(weights):
                derivative = (
                    sum(
                        weight * owners.count(slot) * (1 if k == 0 else scores[k - 1])
                        for weight, scores, owners, _ in rows
                    )
                    / total
                )
                if algebra is MARKOV and base:
                    derivative /= base
                expected_derivatives.append(float(derivative))
            assert gradient.tolist() == pytest.approx(
                expected_derivatives, abs=2e-12, rel=0.0
            )

        (feature_gradient,) = torch.autograd.grad(moments[0], arc)
        expected = [
            float(
                sum(
                    weight * route.count(i) * _ARC_FEATURES[i][0]
                    for weight, _, _, route in rows
                )
                / totals[1]
            )
            for i in range(len(topology.arcs))
        ]
        assert feature_gradient[:, 0].tolist() == pytest.approx(
            expected, abs=2e-12, rel=0.0
        )
        assert torch.count_nonzero(feature_gradient[:, 1:]).item() == 0


def test_feature_moments_zero_empty_and_cyclic_support():
    topology = _fixture()
    weights = torch.ones(5, dtype=torch.float64)
    features, _, _ = _features()
    relation = RationalRelation(topology, weights, algebra=MARKOV)
    mass, moments = relation.query(((1, 1),)).log_feature_moments(features)
    assert mass.item() == -torch.inf
    assert moments.tolist() == [-torch.inf] * 3

    # Cycles in the topology can become an acyclic observed graph when
    # an edge consumes a symbol. A productive epsilon cycle still needs
    # a separate carrier closure, even when the scalar star is permitted.
    loop = RelationTopology(
        (_TAPE,),
        1,
        1,
        (Arc(0, 0, (None,), 1),),
        (Boundary(0, 0),),
        (Boundary(0, 0),),
        ("unit", "loop"),
    )
    cyclic = RationalRelation(
        loop, torch.tensor([1.0, 0.25], dtype=torch.float64), algebra=MARKOV
    ).query(((),), max_closure_size=8)
    assert cyclic.weight().item() == pytest.approx(4 / 3)
    with pytest.raises(ProductiveEpsilonCycleError, match="acyclic productive"):
        cyclic.log_feature_moments(torch.zeros((1, 2), dtype=torch.float64))


def test_feature_moments_validate_original_rows_feature_axis_and_log_carrier():
    relation = RationalRelation(
        _fixture(), torch.ones(5, dtype=torch.float64), algebra=MARKOV
    )
    query = relation.query((_WORD,))
    features, _, _ = _features()
    for invalid in (
        features[:4],
        features.float(),
        torch.full_like(features, torch.nan),
        torch.full_like(features, torch.inf),
    ):
        with pytest.raises(ValueError, match="one log-space row per occurrence"):
            query.log_feature_moments(invalid)
    with pytest.raises(ValueError, match="nonempty feature axis"):
        query.log_feature_moments(features[:, :0])
    with pytest.raises(ValueError, match="final log features"):
        query.log_feature_moments(features, final_logs=features[:2, :2])
