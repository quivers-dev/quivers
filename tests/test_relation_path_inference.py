"""Independent path enumeration checks for acyclic decoding and counts."""

from collections import Counter
from fractions import Fraction
from math import log, prod

import pytest
import torch
from torch import nn

from quivers.core.algebras import LOG_PROB, MARKOV
from quivers.core.relations import (
    Arc,
    Boundary,
    ProductiveEpsilonCycleError,
    RationalRelation,
    RelationTopology,
)
from tests.evidence.oracles import paths
from tests.test_relation_composition import _INPUT, _topology


def _fixture():
    weights = {
        "shared": Fraction(1, 4),
        "tail": Fraction(1, 2),
        "direct": Fraction(3, 8),
        "zero": Fraction(0),
        "start": Fraction(1, 4),
        "unit": Fraction(1),
    }
    machine = paths.Machine(
        arcs=tuple(
            paths.Arc(name=str(i), source=src, target=dst, labels=(label,), weight=w)
            for i, (src, dst, label, w) in enumerate(
                (
                    (0, 1, None, "shared"),
                    (0, 1, None, "shared"),
                    (1, 3, "a", "tail"),
                    (0, 3, "a", "direct"),
                    (0, 2, None, "zero"),
                    (2, 3, "a", "unit"),
                    (4, 3, "a", "unit"),
                    (0, 5, None, "unit"),
                    (5, 3, "a", "zero"),
                )
            )
        ),
        initial=(
            paths.Mark(state=0, weight="unit"),
            paths.Mark(state=4, weight="start"),
        ),
        final=(paths.Mark(state=0, weight="unit"), paths.Mark(state=3, weight="unit")),
    )
    return machine, weights


def _path_id(path):
    return path.start.state, tuple(int(a.name) for a in path.arcs), path.end.state


def _production_id(path):
    return path.initial_state, path.arc_ids, path.final_state


@pytest.mark.parametrize("algebra", [MARKOV, LOG_PROB])
def test_acyclic_rank_counts_and_gradient_against_materialized_paths(algebra):
    machine, values = _fixture()
    topology = _topology(machine, (_INPUT,))
    raw = torch.tensor(
        [float(values[name]) for name in topology.weight_names], dtype=torch.float64
    )
    slots = nn.Parameter(raw if algebra is MARKOV else raw.log())
    relation = RationalRelation(topology, slots, algebra=algebra)
    query = relation.query(((0,),))
    observed = [
        candidate
        for candidate in paths.accepting_paths(machine)
        if candidate.strings() == (("a",),)
    ]
    by_path = {
        _path_id(walk): prod(values[name] for name in walk.factors())
        for walk in observed
    }
    ranked = query.k_best_paths(20)
    identities = [_production_id(walk) for walk in ranked]
    positive = {walk for walk, weight in by_path.items() if weight > 0}
    expected_masses = sorted((by_path[walk] for walk in positive), reverse=True)
    assert set(identities) == positive
    assert [by_path[walk] for walk in identities] == expected_masses
    assert [_production_id(walk) for walk in query.k_best_paths(3)] == identities[:3]
    best = query.best_path()
    assert best is not None
    assert _production_id(best) == identities[0]
    assert query.k_best_paths(0) == ()
    with pytest.raises(ValueError, match="nonnegative integer"):
        query.k_best_paths(True)
    masses = [algebra.to_reals(walk.weight).item() for walk in ranked]
    assert masses == pytest.approx([float(mass) for mass in expected_masses])

    counts = query.expected_arc_counts()
    denominator = sum(by_path.values())
    exact = {
        arc_id: sum(mass * ids.count(arc_id) for (_, ids, _), mass in by_path.items())
        / denominator
        for arc_id in range(len(machine.arcs))
    }
    assert set(counts.arc_ids) == set(exact)
    assert counts.counts.tolist() == pytest.approx(
        [float(exact[i]) for i in counts.arc_ids]
    )
    assert algebra.to_reals(counts.normalizer).item() == pytest.approx(
        float(denominator)
    )
    assert torch.allclose(counts.normalizer, query.weight())

    # Independent reverse-mode oracle uses every full path monomial, with
    # neither packed state charts nor the runtime count formula.
    lookup = dict(zip(topology.weight_names, slots.unbind(), strict=True))
    terms = []
    for walk in observed:
        factors = torch.stack([lookup[name] for name in walk.factors()])
        terms.append(factors.prod() if algebra is MARKOV else factors.sum().exp())
    monomials = torch.stack(terms)
    feature = torch.tensor(
        [sum(int(arc.name) + 1 for arc in walk.arcs) for walk in observed],
        dtype=slots.dtype,
    )
    from_paths = (monomials * feature).sum() / monomials.sum()
    from_chart = counts.counts @ slots.new_tensor([i + 1 for i in counts.arc_ids])
    assert torch.allclose(from_chart, from_paths)
    (observed_grad,) = torch.autograd.grad(from_chart, slots)
    (oracle_grad,) = torch.autograd.grad(from_paths, slots)
    assert torch.all(torch.isfinite(observed_grad))
    assert torch.allclose(observed_grad, oracle_grad, atol=1e-13, rtol=1e-13)

    empty = relation.query(((),))
    no_arcs = empty.best_path()
    assert no_arcs is not None and _production_id(no_arcs) == (0, (), 0)
    assert empty.expected_arc_counts().arc_ids == ()
    impossible = relation.query(((1,),))
    assert impossible.best_path() is None
    with pytest.raises(ValueError, match="positive finite normalizer"):
        impossible.expected_arc_counts()
    with pytest.raises(ValueError, match="positive finite normalizer"):
        impossible.sample_path()


def test_sampling_probability_telescopes_to_independent_path_mass(monkeypatch):
    machine, values = _fixture()
    topology = _topology(machine, (_INPUT,))
    slots = torch.tensor(
        [float(values[name]) for name in topology.weight_names], dtype=torch.float64
    )
    relation = RationalRelation(topology, slots, algebra=MARKOV)
    query = relation.query(((0,),))
    mass = {
        _path_id(walk): prod(values[name] for name in walk.factors())
        for walk in paths.accepting_paths(machine)
        if walk.strings() == (("a",),)
    }
    denominator = sum(mass.values())
    actual_multinomial = torch.multinomial
    probabilities = []

    def track(distribution, n, *, generator):
        choice = actual_multinomial(distribution, n, generator=generator)
        probabilities.append(distribution[choice].item())
        return choice

    monkeypatch.setattr(torch, "multinomial", track)
    generator = torch.Generator().manual_seed(301)
    seen = Counter()
    for _ in range(80):
        sampled = _production_id(query.sample_path(generator=generator))
        seen[sampled] += 1
        assert prod(probabilities) == pytest.approx(float(mass[sampled] / denominator))
        probabilities.clear()
    assert set(seen) == {walk for walk, weight in mass.items() if weight > 0}


def test_expected_arc_count_sums_distinct_offsets_of_one_original_arc():
    mark = (Boundary(0, 1),)
    topology = RelationTopology(
        (_INPUT,), 1, 1, (Arc(0, 0, (0,), 0),), mark, mark, ("emit", "unit")
    )
    weights = nn.Parameter(torch.tensor([-1000.0, 0.0], dtype=torch.float64))
    query = RationalRelation(topology, weights, algebra=LOG_PROB).query(((0,) * 5,))
    counts = query.expected_arc_counts()
    assert counts.arc_ids == (0,)
    assert counts.counts.tolist() == [5.0]
    assert counts.normalizer.item() == -5000.0
    assert query.sample_path().arc_ids == (0,) * 5
    path = query.best_path()
    assert path is not None and path.arc_ids == (0,) * 5
    assert path.weight.item() == -5000.0


def test_exact_ranking_resolves_weights_that_round_to_the_same_product():
    mark = (Boundary(0, 3),)
    topology = RelationTopology(
        (_INPUT,),
        1,
        3,
        (Arc(0, 1, (None,), 0), Arc(1, 2, (0,), 1), Arc(0, 2, (0,), 2)),
        mark,
        (Boundary(2, 3),),
        ("lower", "upper", "direct", "unit"),
    )
    step = 2.0**-52
    raw = [1 - step, 1 + step, 1, 1]
    assert raw[0] * raw[1] == 1
    assert Fraction(raw[0]) * Fraction(raw[1]) < 1
    query = RationalRelation(
        topology, torch.tensor(raw, dtype=torch.float64), algebra=MARKOV
    ).query(((0,),))
    assert [walk.arc_ids for walk in query.k_best_paths(2)] == [(2,), (0, 1)]


def test_productive_cycles_and_weight_domains_refuse_path_operations():
    mark = (Boundary(0, 1),)
    epsilon_cycle = RelationTopology(
        (_INPUT,), 1, 1, (Arc(0, 0, (None,), 0),), mark, mark, ("loop", "unit")
    )
    query = RationalRelation(
        epsilon_cycle, torch.tensor([0.5, 1.0]), algebra=MARKOV
    ).query(((),), max_closure_size=1)
    assert query.weight().item() == 2
    for operation in (query.best_path, query.expected_arc_counts, query.sample_path):
        with pytest.raises(
            ProductiveEpsilonCycleError, match="acyclic productive graph"
        ):
            operation()

    weights = torch.ones(1)
    empty_topology = RelationTopology(
        (_INPUT,), 1, 1, (), (Boundary(0, 0),), (Boundary(0, 0),), ("unit",)
    )
    signed_algebra = RationalRelation(empty_topology, weights).query(((),))
    with pytest.raises(TypeError, match="Markov or LogProb"):
        signed_algebra.best_path()
    with pytest.raises(TypeError, match="floating-point"):
        RationalRelation(empty_topology, weights.long(), algebra=MARKOV).query(
            ((),)
        ).sample_path()


def test_log_probability_two_extreme_paths_use_the_finite_log_normalizer():
    topology = RelationTopology(
        (_INPUT,),
        1,
        2,
        (Arc(0, 1, (0,), 0), Arc(0, 1, (0,), 0)),
        (Boundary(0, 1),),
        (Boundary(1, 1),),
        ("rare", "unit"),
    )
    query = RationalRelation(
        topology,
        torch.tensor([-1000.0, 0.0], dtype=torch.float64),
        algebra=LOG_PROB,
    ).query(((0,),))
    counts = query.expected_arc_counts()
    assert counts.normalizer.item() == pytest.approx(-1000 + log(2), abs=1e-14)
    assert counts.counts.tolist() == pytest.approx([0.5, 0.5], abs=1e-13)
    assert set(counts.arc_ids) == {0, 1}
    assert query.sample_path().arc_ids in {(0,), (1,)}
