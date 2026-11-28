"""Exact finite-support sweep and truncation diagnostics for loopy beam BP."""

from fractions import Fraction
from itertools import product
from math import prod

import pytest
import torch
from torch import nn

from quivers.core.algebras import MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    FiniteStringBeliefPlan,
    RationalRelation,
    RelationLimitError,
    RelationTopology,
    StringFactor,
    StringFactorGraph,
    StringVariable,
)
from tests.test_relation_composition import _INPUT

_WORDS = ((1,), (), (0,))
_SCOPES = ((0, 1), (1, 2), (0, 2))


def _relation(words, values, arity):
    labels = tuple(word[0] if word else None for word in words)
    arcs = tuple(
        Arc(0, 1, entry, slot)
        for slot, entry in enumerate(product(labels, repeat=arity))
    )
    source = nn.Parameter(torch.tensor([*values, 1.0], dtype=torch.float64))
    return RationalRelation(
        RelationTopology(
            (_INPUT,) * arity,
            arity - 1,
            2,
            arcs,
            (Boundary(0, len(arcs)),),
            (Boundary(1, len(arcs)),),
            tuple(f"weight_{slot}" for slot in range(len(arcs) + 1)),
        ),
        source,
        algebra=MARKOV,
    ), source


def _plan(graph, supports, *, max_entries=120):
    return FiniteStringBeliefPlan(
        graph, supports, max_entries=max_entries, max_states=600, max_work=600
    )


def _normalize(exact):
    exact = tuple(exact)
    total = sum(exact)
    assert total > 0
    return tuple(value / total for value in exact)


def _oracle(weights, damping, sweeps):
    size = len(_WORDS)
    initial = tuple(Fraction(1, size) for _ in _WORDS)
    messages = {(f, v): initial for f, scope in enumerate(_SCOPES) for v in scope}
    adjacent = {
        v: tuple(f for f, scope in enumerate(_SCOPES) if v in scope) for v in range(3)
    }
    for iteration in range(sweeps + 1):
        births = {}
        outgoing = {}
        proposals = {}
        for f, scope in enumerate(_SCOPES):
            for axis, v in enumerate(scope):
                other = scope[1 - axis]
                context = _normalize(
                    prod(messages[g, other][i] for g in adjacent[other] if g != f)
                    for i in range(size)
                )
                raw = [Fraction(0) for _ in _WORDS]
                for a, b in product(range(size), repeat=2):
                    pair = (a, b)
                    raw[pair[axis]] += weights[a * size + b] * context[pair[1 - axis]]
                ranking = sorted(range(size), key=lambda i: (-raw[i], _WORDS[i]))
                retained = tuple(sorted(ranking[:2]))
                removed = ranking[2:]
                new = _normalize(
                    raw[i] if i in retained else Fraction(0) for i in range(size)
                )
                previously = tuple(
                    messages[f, v][i] if new[i] else Fraction(0) for i in range(size)
                )
                if sum(previously):
                    old = _normalize(previously)
                    outgoing[f, v] = tuple(
                        damping * m + (1 - damping) * n
                        for m, n in zip(old, new, strict=True)
                    )
                else:
                    outgoing[f, v] = new
                births[f, v] = (retained, tuple(raw), sum(raw[i] for i in removed))
                proposals[f, v] = new
        if iteration == sweeps:
            break
        messages = outgoing
    marginals = {
        v: _normalize(prod(messages[f, v][i] for f in adjacent[v]) for i in range(size))
        for v in range(3)
    }
    residual = max(
        sum(
            abs(before - after)
            for before, after in zip(messages[key], proposals[key], strict=True)
        )
        / 2
        for key in messages
    )
    return messages, births, marginals, residual


def test_pruned_loopy_sweeps_match_exact_declared_support_and_raw_scales():
    exact = tuple(
        map(Fraction, (0.875, 0.625, 0.25, 0.5, 0.75, 0.125, 0.25, 0.125, 0.0625))
    )
    relation, source = _relation(_WORDS, tuple(map(float, exact)), arity=2)
    names = ("x", "y", "z")
    graph = StringFactorGraph(
        tuple(StringVariable(name, _INPUT) for name in names),
        tuple(
            StringFactor(tuple(names[i] for i in scope), relation) for scope in _SCOPES
        ),
    )
    plan = _plan(graph, {name: _WORDS for name in names})
    obtained = plan.run_pruned(beam=2, max_sweeps=2, tolerance=None, damping=1 / 3)
    messages, raw, beliefs, residual = _oracle(exact, Fraction(1, 3), 2)
    assert obtained.status == "limit" and obtained.sweeps == 2
    assert obtained.exact is False and obtained.truncated
    assert obtained.message_residual == pytest.approx(float(residual))
    assert obtained.log_marginals is not None
    for v, name in enumerate(names):
        assert obtained.log_marginals[name].exp().tolist() == pytest.approx(
            list(map(float, beliefs[v]))
        )
    for f, scope in enumerate(_SCOPES):
        for axis, v in enumerate(scope):
            selected, effects, omitted = raw[f, v]
            record = obtained.proposals[f][axis]
            assert record.tape == _INPUT
            assert record.words == tuple(_WORDS[i] for i in selected)
            assert record.log_weights.exp().tolist() == pytest.approx(
                [float(effects[i]) for i in selected]
            )
            assert record.log_total_mass.exp().item() == pytest.approx(
                float(sum(effects))
            )
            assert record.log_omitted_mass.exp().item() == pytest.approx(float(omitted))
            output = obtained.log_messages[f][axis]
            assert int(torch.isfinite(output).sum()) <= 2
            assert output.exp().tolist() == pytest.approx(
                list(map(float, messages[f, v]))
            )

    (gradient,) = torch.autograd.grad(obtained.log_marginals["x"].exp()[0], source)
    epsilon = Fraction(1, 2**18)
    plus = (exact[0] + epsilon, *exact[1:])
    minus = (exact[0] - epsilon, *exact[1:])
    plus_estimate = _oracle(plus, Fraction(1, 3), 2)[2][0][0]
    minus_estimate = _oracle(minus, Fraction(1, 3), 2)[2][0][0]
    expected = float((plus_estimate - minus_estimate) / (2 * epsilon))
    assert gradient[0].item() == pytest.approx(expected, abs=1e-8)


def test_hard_support_conflict_is_distinct_from_convergence_and_resource_exhaustion():
    words = ((0,), (1,))
    left, _ = _relation(words, (2.0, 1.0), arity=1)
    right, _ = _relation(words, (1.0, 2.0), arity=1)
    graph = StringFactorGraph(
        (StringVariable("x", _INPUT),),
        (StringFactor(("x",), left), StringFactor(("x",), right)),
    )
    plan = _plan(graph, {"x": words})
    assert plan.run(max_sweeps=3).log_marginals["x"].exp().tolist() == pytest.approx(
        [0.5, 0.5]
    )
    result = plan.run_pruned(beam=1, max_sweeps=3, damping=0.9)
    assert result.status == "empty_support"
    assert result.log_marginals is None and result.message_residual is None
    assert result.sweeps == 1 and result.exact is False and result.truncated
    assert result.proposals[0][0].words == ((0,),)
    assert result.proposals[1][0].words == ((1,),)
    assert result.proposals[0][0].log_total_mass.exp().item() == pytest.approx(3)
    assert result.proposals[0][0].log_omitted_mass.exp().item() == pytest.approx(1)
    with pytest.raises(RelationLimitError, match="max_entries"):
        _plan(graph, {"x": words}, max_entries=1)
    with pytest.raises(ValueError, match="positive integer"):
        plan.run_pruned(beam=0, max_sweeps=2)
    with pytest.raises(ValueError, match="at least one sweep"):
        plan.run_pruned(beam=1, max_sweeps=0)


def test_lexicographic_tie_and_convergence_do_not_claim_exactness():
    words = ((1,), (0,))
    relation, _ = _relation(words, (1.0, 1.0), arity=1)
    graph = StringFactorGraph(
        (StringVariable("x", _INPUT),), (StringFactor(("x",), relation),)
    )
    result = _plan(graph, {"x": words}).run_pruned(beam=1, max_sweeps=2)
    assert result.status == "converged"
    assert result.exact is False and result.truncated
    assert result.message_residual == pytest.approx(0)
    record = result.proposals[0][0]
    assert record.boundary_tie is True and record.words == ((0,),)
    assert result.log_marginals is not None
    assert result.log_marginals["x"].exp().tolist() == [0, 1]


def test_damping_uses_new_support_when_the_beam_switches_to_a_disjoint_word():
    words = ((0,), (1,))
    prior, _ = _relation(words, (1.0, 2.0), arity=1)
    channel, _ = _relation(words, (4.0, 3.0, 0.0, 6.0), arity=2)
    graph = StringFactorGraph(
        (StringVariable("x", _INPUT), StringVariable("y", _INPUT)),
        (StringFactor(("y",), prior), StringFactor(("x", "y"), channel)),
    )
    plan = _plan(graph, {"x": words, "y": words})
    first = plan.run_pruned(beam=1, max_sweeps=1, tolerance=None, damping=0.99)
    second = plan.run_pruned(beam=1, max_sweeps=2, tolerance=None, damping=0.99)
    assert first.log_messages[1][0].exp().tolist() == [1.0, 0.0]
    assert second.log_messages[1][0].exp().tolist() == [0.0, 1.0]
    changed = second.proposals[1][0]
    assert changed.words == ((1,),)
    assert changed.log_total_mass.exp().item() == pytest.approx(9)
    assert changed.log_omitted_mass.exp().item() == pytest.approx(3)
